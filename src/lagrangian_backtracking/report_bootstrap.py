"""報告層的 exact categorical member bootstrap 純計算核心。

本模組只接受已由報告串流層整理好的 member-level 類別代碼，不讀取 trajectory、
NetCDF、Parquet 或其他檔案，也不保存完整 member 清單。每一筆輸入代表一個固定
順序的有效或缺值 member：整數代碼 ``0`` 到 ``category_count - 1`` 是可計入
分子與類別 raw count 的資料，``None`` 代表該 member 在此 categorical product
沒有可歸類的代碼，但仍然是已知的 member，必須留在重抽樣分母中。這個界線避免
把缺值誤當成某一個物理類別或以零值替代資料狀態。

每個 replicate 對同一組固定順序的 ``N`` 個 member 做 exact nonparametric
member bootstrap。令第 ``i`` 個 member（以零起始索引）處理前尚餘 ``R_i`` 個
抽樣名額，先取

``w_i ~ Binomial(R_i, 1 / (N - i))``，

並以 ``R_(i+1) = R_i - w_i`` 更新；最後一筆直接取得所有剩餘名額。這個條件式
分解與直接從 ``Multinomial(N; 1/N, ..., 1/N)`` 產生每一筆 member weight
完全同分布，但只需保存 ``O(B*C+B)`` 的陣列，其中 ``B`` 是 replicates、``C``
是 categories。輸入 iterable 因而只被消費一次，不會 materialize 成 member
清單；中間 weights 也只存在於計算期間，回傳值不保存它們。

group seed 由固定方法識別碼、base seed 的十進位文字與 ASCII-safe group key
組成 canonical JSON，再取 SHA-256。前 16 bytes 以 big-endian 解讀成
``PCG64DXSM`` 的 128 位元 seed，完整小寫十六進位 digest 同時留在結果中，讓
相同資料契約可重建且不同 group 不會共用未記錄的亂數來源。類別 raw count 是
原始輸入中非 ``None`` 代碼的計數；lower、median、upper 則是每個類別的重抽樣
權重除以 ``N`` 後，以固定 ``linear`` 方法取得 ``(1-confidence)/2``、``0.5``
與 ``1-(1-confidence)/2`` 分位數。

這些區間只表達固定 member sample unit 下的「條件式來源權重」不確定性；它們
不是建立先驗與似然後得到的絕對來源機率，也不是因果歸因。模組刻意不使用
Poisson、aggregate count 近似、I/O 或其他替代抽樣方法。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "BOOTSTRAP_METHOD_ID",
    "BOOTSTRAP_QUANTILE_METHOD",
    "CategoricalBootstrapIntervals",
    "derive_bootstrap_group_seed",
    "bootstrap_categorical_intervals",
]


# 方法識別碼與分位數方法都是結果資料契約的一部分；下游不能根據相近名稱猜測
# 另一種 bootstrap 或 NumPy 的預設 quantile 行為，否則不同報告無法比較。
BOOTSTRAP_METHOD_ID: Final[str] = "sequential_conditional_binomial_multinomial_v1"
BOOTSTRAP_QUANTILE_METHOD: Final[str] = "linear"

_MAX_UINT128: Final[int] = 2**128 - 1
_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def _require_native_int(value: object, *, label: str) -> int:
    """要求欄位是原生 Python ``int``，避免 bool 與 NumPy scalar 偷換型別。

    Python 的 ``bool`` 是 ``int`` 子類別，而 NumPy 整數 scalar 也可能在比較或
    轉型時看起來像原生整數；本模組的 seed、member 數與 category code 都是跨
    執行環境的資料契約，所以使用精確型別檢查，不讓隱式轉型改變 canonical JSON
    或固定寬度計數的意義。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    return value


def _require_uint128(value: object, *, label: str) -> int:
    """驗證可作為 base seed 的 128 位元無號原生整數。"""

    number = _require_native_int(value, label=label)
    if not 0 <= number <= _MAX_UINT128:
        raise ValueError(f"{label} 必須介於 0 與 2^128-1 之間")
    return number


def _require_positive_int(value: object, *, label: str, minimum: int = 1) -> int:
    """驗證正的原生整數，並把下限保留在錯誤訊息中。"""

    number = _require_native_int(value, label=label)
    if number < minimum:
        raise ValueError(f"{label} 必須大於或等於 {minimum}")
    return number


def _require_ascii_group_key(value: object, *, label: str = "group_key") -> str:
    """驗證可穩定放進 canonical JSON 的非空可列印 ASCII group key。

    group key 是不同報告分組的命名空間，而不是檔案路徑；因此保留一般可列印
    ASCII 字元（例如 ``/``、``|`` 或 ``:``）的組合能力，但拒絕控制字元、首尾
    空白與非 ASCII 文字。這樣既不任意限制 receptor/boundary 的既有組合格式，
    也能讓 canonical bytes 在所有支援環境中固定為 ASCII。
    """

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} 必須只含 ASCII 字元") from error
    if any(byte < 0x20 or byte > 0x7E for byte in encoded):
        raise ValueError(f"{label} 不得含 ASCII 控制字元")
    return value


def _canonical_seed_digest(base_seed: int, group_key: str) -> tuple[int, str]:
    """依固定三欄 canonical JSON 產生 PCG64DXSM seed 與完整 digest。

    ``base_seed`` 先以十進位文字放入 JSON，避免不同語言對超過 64 位元 JSON
    number 的解析方式造成差異；method、base_seed、group 三個欄位的名稱也一併
    進入摘要，防止同一個 group 在不同方法版本間重用亂數子流。回傳的整數只
    使用 digest 前 16 bytes，第二項則保留完整 SHA-256 小寫十六進位文字。
    """

    payload = {
        "method": BOOTSTRAP_METHOD_ID,
        "base_seed": str(base_seed),
        "group": group_key,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    digest_bytes = hashlib.sha256(canonical_json).digest()
    return int.from_bytes(digest_bytes[:16], byteorder="big", signed=False), digest_bytes.hex()


def derive_bootstrap_group_seed(base_seed: int, group_key: str) -> tuple[int, str]:
    """由 base seed 與 group key 派生 deterministic ``PCG64DXSM`` seed。

    Args:
        base_seed: ``0`` 到 ``2^128-1`` 的原生 Python ``int``。這是報告規格提供的
            bootstrap 根 seed，不直接作為每一個 group 的亂數 seed。
        group_key: 非空、沒有首尾空白且只含可列印 ASCII 字元的原生 ``str``。它
            應代表已固定輸入順序的 receptor／boundary 或其他報告分組。

    Returns:
        ``(pcg64dxsm_seed, digest)`` 二元 tuple。第一項是 SHA-256 前 16 bytes
        的 big-endian 無號整數，可直接交給 ``np.random.PCG64DXSM``；第二項是
        完整 64 碼小寫 SHA-256。完整 digest 必須隨 typed result 保存，不能只留
        前段 seed。

    Raises:
        TypeError: base seed 或 group key 不是原生契約型別。
        ValueError: seed 超過 128 位元範圍，或 group key 不是安全 ASCII key。
    """

    normalized_seed = _require_uint128(base_seed, label="base_seed")
    normalized_group = _require_ascii_group_key(group_key)
    return _canonical_seed_digest(normalized_seed, normalized_group)


def _readonly_int64_vector(value: object, *, label: str, size: int) -> np.ndarray:
    """複製並驗證一維非負計數，封存為指定長度的唯讀 ``int64`` 陣列。

    先檢查原始 NumPy dtype 與每個 Python 整數的範圍，再轉成 int64；不能讓
    負數、浮點、布林或 uint64 超界值在 NumPy 轉型時靜默截斷。copy 與 write
    flag 同時使用，避免 caller 透過原始陣列或回傳陣列改寫 typed product。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是一維整數陣列") from error
    if raw.ndim != 1 or raw.size != size:
        raise ValueError(f"{label} 必須是長度 {size} 的一維陣列")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{label} 必須是整數 dtype，不接受 bool 或浮點")

    for element in raw.flat:
        number = int(element)
        if number < 0:
            raise ValueError(f"{label} 不可包含負值")
        if number > _INT64_MAX:
            raise ValueError(f"{label} 的值超出 int64 可表示範圍")

    copied = np.array(raw, dtype=np.int64, copy=True)
    copied.setflags(write=False)
    return copied


def _readonly_probability_vector(value: object, *, label: str, size: int) -> np.ndarray:
    """複製並驗證一維有限 ``[0, 1]`` 區間，封存為唯讀 ``float64``。

    分位數是 member weight 除以整數分母所得的比例；有限性、範圍與單調順序
    都屬於 typed product 的 cross-field gate，不能留給 renderer 猜測或修補。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是一維數值陣列") from error
    if raw.ndim != 1 or raw.size != size:
        raise ValueError(f"{label} 必須是長度 {size} 的一維陣列")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{label} 必須是整數或浮點 dtype")

    try:
        copied = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 無法安全轉為 float64") from error
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{label} 必須全部為有限值")
    if np.any(copied < 0.0) or np.any(copied > 1.0):
        raise ValueError(f"{label} 必須全部落在 0 到 1 之間")

    copied.setflags(write=False)
    return copied


def _require_digest(value: object) -> str:
    """要求完整、全小寫且沒有前綴的 SHA-256 十六進位摘要。"""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("digest 必須是 64 碼小寫 SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class CategoricalBootstrapIntervals:
    """一組 categorical member bootstrap 的不可變 raw count 與比例區間。

    ``member_count`` 是輸入 member 數 ``N``，包含代碼為 ``None`` 的 member；
    ``category_count`` 是代碼軸 ``0..C-1`` 的長度。``raw_counts`` 只計非
    ``None`` 的原始 categorical numerator，故其總和可以小於 ``N``，但不能大於
    ``N``。``lower``、``median`` 與 ``upper`` 是每一類重抽樣比例的固定
    ``linear`` quantile，軸都是 category index，數值必須是有限 float64 且遵守
    ``lower <= median <= upper``。

    ``method``、``group`` 與 ``digest`` 保存算法版本、分組命名空間與完整亂數
    provenance；``replicates`` 和 ``confidence`` 使區間的重抽樣數與中心信賴
    水準可被下游正確解讀。所有 NumPy 陣列都在建構時 defensive-copy 並關閉
    寫入權限，frozen dataclass 則防止欄位重新指派。此結果只屬固定 member
    sample unit 下的條件式來源權重 CI，不是絕對來源機率或因果歸因。
    """

    method: str
    group: str
    digest: str
    member_count: int
    category_count: int
    replicates: int
    confidence: float
    raw_counts: np.ndarray
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    quantile_method: str

    def __post_init__(self) -> None:
        """完成 scalar、shape、dtype、範圍、順序與計數守恆的 cross-field 驗證。"""

        if type(self.method) is not str or self.method != BOOTSTRAP_METHOD_ID:
            raise ValueError("method 必須精確符合 BOOTSTRAP_METHOD_ID")
        group = _require_ascii_group_key(self.group, label="group")
        digest = _require_digest(self.digest)
        member_count = _require_positive_int(self.member_count, label="member_count")
        if member_count > _INT64_MAX:
            raise ValueError("member_count 不得超過 int64 可表示範圍")
        category_count = _require_positive_int(self.category_count, label="category_count")
        replicates = _require_positive_int(self.replicates, label="replicates", minimum=2)
        if type(self.confidence) is not float:
            raise TypeError("confidence 必須是原生 float，且不可是 NumPy scalar")
        if not np.isfinite(self.confidence) or not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence 必須是介於 0 與 1 之間的有限值")
        if type(self.quantile_method) is not str or self.quantile_method != BOOTSTRAP_QUANTILE_METHOD:
            raise ValueError("quantile_method 必須精確符合 BOOTSTRAP_QUANTILE_METHOD")

        raw_counts = _readonly_int64_vector(
            self.raw_counts,
            label="raw_counts",
            size=category_count,
        )
        lower = _readonly_probability_vector(self.lower, label="lower", size=category_count)
        median = _readonly_probability_vector(self.median, label="median", size=category_count)
        upper = _readonly_probability_vector(self.upper, label="upper", size=category_count)

        # 以 Python int 加總，避免 category 數量或極大 raw count 讓 NumPy sum
        # 以固定寬度整數繞回；這個不變量同時確認 raw numerator 沒有超過分母。
        raw_total = sum(int(element) for element in raw_counts.flat)
        if raw_total > member_count:
            raise ValueError("raw_counts 的總和不得超過 member_count")
        if np.any(lower > median) or np.any(median > upper):
            raise ValueError("lower、median、upper 必須逐類維持單調不減")

        object.__setattr__(self, "method", self.method)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "member_count", member_count)
        object.__setattr__(self, "category_count", category_count)
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "confidence", self.confidence)
        object.__setattr__(self, "raw_counts", raw_counts)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "quantile_method", self.quantile_method)


def _validate_category_code(value: object, *, category_count: int, index: int) -> int | None:
    """驗證一筆 member code，保留 ``None`` 的缺值／不可分類語意。"""

    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"codes[{index}] 必須是 None 或原生 int，不接受 bool 或 NumPy scalar")
    if not 0 <= value < category_count:
        raise ValueError(f"codes[{index}] 必須落在 0 到 category_count-1")
    return value


def _safe_vector_add_in_place(total: np.ndarray, added: np.ndarray, *, label: str) -> None:
    """先檢查 int64 上限，再安全累加同形狀的非負向量。"""

    if total.shape != added.shape or total.dtype != np.int64 or added.dtype != np.int64:
        raise RuntimeError(f"{label} 的內部 dtype 或 shape 不符")
    if np.any(added < 0) or np.any(total < 0):
        raise RuntimeError(f"{label} 不可含負值")
    if np.any(added > np.int64(_INT64_MAX) - total):
        raise OverflowError(f"{label} 累加將超過 int64 上限")
    total += added


def bootstrap_categorical_intervals(
    codes: Iterable[int | None],
    *,
    member_count: int,
    category_count: int,
    replicates: int,
    confidence: float,
    base_seed: int,
    group_key: str,
) -> CategoricalBootstrapIntervals:
    """以一次串流輸入建立 exact categorical member bootstrap 區間。

    Args:
        codes: 一次性的 member categorical iterable。第 ``i`` 筆只能是原生 Python
            ``int`` 且滿足 ``0 <= code < category_count``，或是 ``None``；輸入順序
            就是重抽樣的 member sample unit 順序。函式會消費並驗證恰好
            ``member_count`` 筆，不把 iterable materialize 成 list。
        member_count: member 分母 ``N``，必須是大於零且不超過 int64 的原生 ``int``。
            ``None`` member 仍然納入這個分母。
        category_count: 類別軸長度 ``C``，必須是正的原生 ``int``；代碼軸為
            ``0..C-1``。
        replicates: bootstrap replicate 數 ``B``，必須大於 1 的原生 ``int``。
        confidence: 中心信賴水準，必須是介於 0 與 1 之間的原生有限 ``float``。
        base_seed: ``0..2^128-1`` 的 bootstrap 根 seed。
        group_key: 非空可列印 ASCII group key；它與方法及 base seed 共同決定
            reproducible random stream。

    Returns:
        :class:`CategoricalBootstrapIntervals`。``raw_counts`` 是原始非 ``None``
        類別計數；三個 interval array 是各類 ``weight / N`` 的線性分位數。結果
        不保留中間 weights，所有輸出陣列都是 defensive、唯讀的 int64／float64。

    Raises:
        TypeError: 參數或 member code 不是指定的原生型別，或 ``codes`` 不可迭代。
        ValueError: 參數超出範圍、member 數量少於或多於 ``member_count``、或 code
            不在類別軸內。
        OverflowError: 計數即將超出 int64 上限；此情況會 fail closed，不回傳繞回值。

    Notes:
        第 ``i`` 筆 member 使用 ``Binomial(R_i, 1/(N-i))``，最後一筆取得餘數，
        因而每個 replicate 的全部 member weights 恰為 ``N``。這是 exact
        ``Multinomial(N; 1/N, ..., 1/N)``，不是 aggregate count 的 Poisson 近似。
        亂數只在 NumPy ``PCG64DXSM`` 上運作，且 weight 抽樣不插入任何物理積分
        stage。CI 僅是固定 member sample unit 下的條件式來源權重 CI。
    """

    normalized_member_count = _require_positive_int(member_count, label="member_count")
    if normalized_member_count > _INT64_MAX:
        raise ValueError("member_count 不得超過 int64 可表示範圍")
    normalized_category_count = _require_positive_int(category_count, label="category_count")
    normalized_replicates = _require_positive_int(replicates, label="replicates", minimum=2)
    if type(confidence) is not float:
        raise TypeError("confidence 必須是原生 float，且不可是 NumPy scalar")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必須是介於 0 與 1 之間的有限值")
    normalized_seed, digest = derive_bootstrap_group_seed(base_seed, group_key)

    try:
        iterator = iter(codes)
    except TypeError as error:
        raise TypeError("codes 必須是可一次串流的 Iterable") from error

    # counts 是唯一的 B×C bootstrap 累加矩陣；remaining、weights、assigned_weights
    # 都是 B 長度的工作向量。這個配置保留固定的 O(B*C+B) RAM，不保存 N 筆 code。
    counts = np.zeros((normalized_replicates, normalized_category_count), dtype=np.int64)
    raw_counts = np.zeros(normalized_category_count, dtype=np.int64)
    remaining = np.full(normalized_replicates, normalized_member_count, dtype=np.int64)
    assigned_weights = np.zeros(normalized_replicates, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64DXSM(normalized_seed))

    for index in range(normalized_member_count):
        try:
            raw_code = next(iterator)
        except StopIteration as error:
            raise ValueError(
                f"codes 成員不足：預期 {normalized_member_count} 筆，於第 {index} 筆結束"
            ) from error
        code = _validate_category_code(
            raw_code,
            category_count=normalized_category_count,
            index=index,
        )

        denominator = normalized_member_count - index
        if denominator == 1:
            # 最後一筆不能再呼叫 binomial；直接取得所有剩餘名額，確保每個
            # replicate 的 member weight 總和嚴格回到 N 而不受浮點機率影響。
            weights = remaining.copy()
            remaining.fill(0)
        else:
            probability = 1.0 / denominator
            weights = np.asarray(rng.binomial(remaining, probability), dtype=np.int64)
            if weights.shape != (normalized_replicates,):
                raise RuntimeError("PCG64DXSM binomial 結果 shape 不符內部契約")
            if np.any(weights < 0) or np.any(weights > remaining):
                raise RuntimeError("conditional binomial 產生超出剩餘名額的 weight")
            remaining -= weights

        _safe_vector_add_in_place(
            assigned_weights,
            weights,
            label="bootstrap weights",
        )

        if code is None:
            continue

        current_raw = int(raw_counts[code])
        if current_raw >= _INT64_MAX:
            raise OverflowError("raw_counts 累加將超過 int64 上限")
        raw_counts[code] = current_raw + 1

        # 使用 column view 更新單一類別；先逐元素檢查上限，避免 NumPy fixed-width
        # 加法在未來替換 RNG 或輸入上限時靜默繞回負值。
        category_column = counts[:, code]
        _safe_vector_add_in_place(
            category_column,
            weights,
            label=f"counts[:, {code}]",
        )

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"codes 成員過多：預期恰好 {normalized_member_count} 筆")

    if np.any(remaining != 0) or np.any(assigned_weights != normalized_member_count):
        raise RuntimeError("每個 bootstrap replicate 的 member weight 未守恆為 member_count")

    alpha = 1.0 - confidence
    probabilities = counts.astype(np.float64, copy=True) / float(normalized_member_count)
    quantiles = np.quantile(
        probabilities,
        (alpha / 2.0, 0.5, 1.0 - alpha / 2.0),
        axis=0,
        method=BOOTSTRAP_QUANTILE_METHOD,
    )

    return CategoricalBootstrapIntervals(
        method=BOOTSTRAP_METHOD_ID,
        group=group_key,
        digest=digest,
        member_count=normalized_member_count,
        category_count=normalized_category_count,
        replicates=normalized_replicates,
        confidence=confidence,
        raw_counts=raw_counts,
        lower=quantiles[0],
        median=quantiles[1],
        upper=quantiles[2],
        quantile_method=BOOTSTRAP_QUANTILE_METHOD,
    )
