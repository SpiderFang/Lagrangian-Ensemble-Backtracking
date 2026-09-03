"""正式報告 release 的可重現繪圖與抽樣規格。

本模組只處理 report spec 的 immutable 資料契約、aggregate spec binding 與小型 JSON
I/O 邊界；不讀取 OCM schema 3／NWW3 schema 1 大型陣列，也不執行統計、繪圖或軌跡
選樣。距離與核密度估計帶寬使用公尺（m），travel age 分位數使用秒（s），檔案大小
使用 bytes；這些單位在後續 renderer 與 report validator 中必須維持一致，不能在繪圖
層把秒默默改成日或把經緯度當成公尺座標。垂向分箱邊界以相對瞬時海面的
positive-down 深度（m）表示；觀測落在最後 edge 之外時，後續 reducer 必須記錄
overflow，不得在本模組裁切或把它改塞進最後一箱。

spec 只登錄已通過 aggregate spec binding 的工程設定。代表軌跡選樣 policy 固定對應
core 的 4 個 season × 2 個 spring/neap strata，讓 one-pass reducer 能在每層保留相同
數量的 SHA-256 最小優先序軌跡；這不是尚未實作的 output-quantile 選樣。即使本機
synthetic／pilot 測試成功建立此檔案，也只代表可重現的報告規格契約成立，不代表真實
OCM／NWW 科學成果、絕對來源機率、因果歸因或觀測驗證；正式 scientific evidence
仍必須在 SERVER 以已驗收產品與完整 provenance 重新建立。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn
from uuid import uuid4

from .aggregate_spec import AggregateSpec

__all__ = [
    "REPORT_SPEC_SCHEMA_VERSION",
    "ReportSpec",
    "load_report_spec",
    "validate_report_spec_against_aggregate_spec",
    "write_report_spec",
]


# schema、選樣 policy、quantile 與 renderer 版本都是 exact contract；不允許下游用
# 相近版本猜測欄位語意，否則同一個 aggregate release 可能產生不可比較的圖表。
REPORT_SPEC_SCHEMA_VERSION: Final[str] = "1.0.0"
_REPRESENTATIVE_SELECTION_POLICY: Final[str] = "stable_hash_core_season_tide_v1"
_TRAVEL_AGE_QUANTILES: Final[tuple[float, ...]] = (0.05, 0.25, 0.5, 0.75, 0.95)
_PATHWAY_FIRST_PASSAGE_QUANTILES: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
_FIGURE_FORMATS: Final[tuple[str, ...]] = ("png", "svg", "pdf")
_RENDERER_STYLE_VERSION: Final[str] = "academic_zh_tw_v1"
_LANGUAGE: Final[str] = "zh-TW"
_RASTER_DPI: Final[int] = 300
_MAX_REPRESENTATIVE_SELECTION_SEED: Final[int] = 2**128 - 1

_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "aggregate_spec_canonical_sha256",
        "primary_kde_bandwidth_m",
        "minimum_kde_raw_count",
        "low_sample_min_member_count",
        "vertical_depth_bin_edges_m",
        "representative_trajectory_count_per_site",
        "representative_selection_policy",
        "representative_selection_seed",
        "travel_age_quantiles",
        "pathway_first_passage_quantiles",
        "figure_formats",
        "raster_dpi",
        "renderer_style_version",
        "language",
    }
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _ReportSpecError(ValueError):
    """內部 schema 例外；公開 loader 會統一轉成不含 path 的 ValueError。"""


class _ReportSpecTargetExists(FileExistsError):
    """只表示 caller 指定的 final target 已存在，供 writer 保留 FileExistsError。"""


def _fail(message: str) -> NoReturn:
    """在 loader 內建立不攜帶檔案路徑的內部失敗。"""

    raise _ReportSpecError(message)


def _require_text(value: object, *, label: str) -> str:
    """要求原生、非空且沒有首尾空白的文字欄位。"""

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_slug(value: object, *, label: str) -> str:
    """要求可安全作為 run／case 識別碼的 ASCII slug。"""

    text = _require_text(value, label=label)
    if _SLUG_RE.fullmatch(text) is None:
        raise ValueError(f"{label} 必須是安全 ASCII slug")
    return text


def _require_sha256(value: object, *, label: str) -> str:
    """要求完整、全小寫的 SHA-256 十六進位摘要。"""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是 64 碼小寫 SHA-256")
    return value


def _require_finite_positive_number(value: object, *, label: str) -> int | float:
    """要求排除 bool／NumPy scalar 的有限原生正數。

    primary KDE bandwidth 可以保留 caller 的原生 int 或 float 型別，但不接受其他可
    轉型物件；這可避免 renderer 在 JSON canonicalization 時悄悄改變數值語意。
    """

    value_type = type(value)
    if value_type is int:
        if value <= 0:
            raise ValueError(f"{label} 必須是正值")
        return value
    if value_type is float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} 必須是有限正值")
        return value
    raise TypeError(f"{label} 必須是原生 int 或 float，且不可是 bool")


def _require_positive_native_int(value: object, *, label: str) -> int:
    """要求排除 bool／NumPy scalar 的原生正 Python ``int``。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是正的原生 int，且不可是 bool")
    if value <= 0:
        raise ValueError(f"{label} 必須大於 0")
    return value


def _require_representative_count(value: object) -> int:
    """要求每站代表軌跡數至少 8 且可被 8 整除。

    core report 固定有 4 個 season × 2 個 spring/neap strata；這個倍數限制讓後續一次
    串流 reducer 能為每個 strata 配置相同數量的候選名額。選樣實際依 SHA-256 最小
    優先序執行，不在此規格中虛構 output-quantile 演算法。
    """

    count = _require_positive_native_int(
        value,
        label="representative_trajectory_count_per_site",
    )
    if count < 8 or count % 8 != 0:
        raise ValueError("representative_trajectory_count_per_site 必須至少 8 且可被 8 整除")
    return count


def _snapshot_vertical_depth_edges(value: object) -> tuple[float, ...]:
    """複製並驗證 positive-down 垂向深度分箱邊界。

    每個 edge 是相對瞬時海面的深度，單位為公尺（m）；輸入只接受非字串、非
    mapping 的 sequence，且每點必須是原生 Python ``int`` 或 ``float``，不接受
    bool／NumPy scalar。內部一律 canonical 成有限 ``float``，第一個 edge 必須是
    0.0 且後續嚴格遞增。最後一個 edge 之外的 observation 不在這裡裁切，後續
    reducer 應保留為 overflow 狀態，以免缺值與超出設計範圍被混成同一個 bin。
    """

    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
    ):
        raise TypeError("vertical_depth_bin_edges_m 必須是非字串、非 mapping sequence")
    copied = tuple(value)
    if len(copied) < 2:
        raise ValueError("vertical_depth_bin_edges_m 至少需要兩個邊界")
    normalized: list[float] = []
    for index, edge in enumerate(copied):
        if type(edge) is int:
            try:
                normalized_edge = float(edge)
            except (OverflowError, ValueError):
                raise ValueError(
                    f"vertical_depth_bin_edges_m[{index}] 必須是有限值"
                ) from None
        elif type(edge) is float:
            normalized_edge = edge
        else:
            raise TypeError(
                f"vertical_depth_bin_edges_m[{index}] 必須是原生 int 或 float，且不可是 bool"
            )
        if not math.isfinite(normalized_edge):
            raise ValueError(f"vertical_depth_bin_edges_m[{index}] 必須是有限值")
        normalized.append(normalized_edge)

    if normalized[0] != 0.0:
        raise ValueError("vertical_depth_bin_edges_m 第一個邊界必須精確為 0.0 m")
    # 只有原始 canonical float 已通過 numerically-zero gate 後，才將 -0.0
    # 統一封存為可觀察的 +0.0，避免把非零起點誤改寫成海面基準。
    normalized[0] = 0.0
    if not all(
        left < right
        for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("vertical_depth_bin_edges_m 必須嚴格遞增")
    return tuple(normalized)


def _require_selection_seed(value: object) -> int:
    """要求 0 到 2^128-1 間的原生 unsigned selection seed。"""

    if type(value) is not int:
        raise TypeError("representative_selection_seed 必須是原生 int，且不可是 bool")
    if not 0 <= value <= _MAX_REPRESENTATIVE_SELECTION_SEED:
        raise ValueError("representative_selection_seed 超出 128 位元無號整數範圍")
    return value


def _snapshot_tuple(value: object, *, label: str) -> tuple[Any, ...]:
    """將 caller sequence defensive copy 成 tuple，拒絕字串與 mapping 歧義。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"{label} 必須是可 materialize 的 sequence")
    try:
        return tuple(value)
    except Exception as error:
        raise ValueError(f"{label} 無法建立 defensive tuple") from error


def _snapshot_fixed_float_tuple(
    value: object,
    *,
    label: str,
    expected: tuple[float, ...],
) -> tuple[float, ...]:
    """複製並要求固定的原生 float quantile tuple。"""

    copied = _snapshot_tuple(value, label=label)
    if any(type(item) is not float for item in copied) or copied != expected:
        raise ValueError(f"{label} 必須精確符合固定 quantile tuple")
    return copied  # type: ignore[return-value]


def _snapshot_fixed_text_tuple(
    value: object,
    *,
    label: str,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    """複製並要求固定的原生文字 tuple。"""

    copied = _snapshot_tuple(value, label=label)
    if any(type(item) is not str for item in copied) or copied != expected:
        raise ValueError(f"{label} 必須精確符合固定格式 tuple")
    return copied  # type: ignore[return-value]


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """以 compact、sort-key、UTF-8 JSON 建立 report spec canonical bytes。"""

    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except Exception as error:
        raise ValueError("report spec 無法建立 canonical JSON") from error


def _canonical_sha256(spec: ReportSpec) -> str:
    """計算排除兩個衍生 hash 後的 canonical payload SHA-256。"""

    payload = spec.to_dict()
    del payload["source_sha256"]
    del payload["canonical_sha256"]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class ReportSpec:
    """報告 renderer、KDE、低樣本遮罩與代表軌跡選樣的 immutable 規格。

    ``primary_kde_bandwidth_m`` 是公尺制帶寬，必須在公開 binding 時精確選自同一份
    ``AggregateSpec.kde_bandwidths_m``；三個 count 是正的原生 Python 整數，代表原始
    KDE 樣本門檻、低樣本 member 門檻與每站代表軌跡數，後者至少 8 且為 8 的倍數。
    ``vertical_depth_bin_edges_m`` 是相對瞬時海面的 positive-down 深度邊界，單位為
    公尺（m），canonical 為嚴格遞增的 float tuple；最後 edge 以外的 observation 必須
    由後續 reducer 記錄 overflow，不得在此裁切。
    固定選樣 policy 對應 4 season × 2 spring/neap strata 的 SHA-256 最小優先序保留，
    並不宣稱 output-quantile 選樣。selection seed 是 128 位元無號整數。travel age
    與 pathway first-passage quantiles 的軸單位均為秒（s），raster DPI 是固定 300；
    這個類別只保存設定，不從軌跡 extent 自動改格網、不補零，也不將 synthetic 工程
    檢查誤稱為真實 OCM／NWW 科學成果。

    所有 list/tuple 輸入會被複製成 tuple，``to_dict`` 則建立新的 plain dict/list
    snapshot。``source_sha256`` 是輸入 JSON raw bytes 的摘要，``canonical_sha256`` 是
    排除兩個 hash 欄位後的 compact canonical JSON 摘要；兩者都是由 loader/writer 推導。
    """

    schema_version: str
    run_id: str
    aggregate_spec_canonical_sha256: str
    primary_kde_bandwidth_m: int | float
    minimum_kde_raw_count: int
    low_sample_min_member_count: int
    vertical_depth_bin_edges_m: tuple[float, ...]
    representative_trajectory_count_per_site: int
    representative_selection_policy: str
    representative_selection_seed: int
    travel_age_quantiles: tuple[float, ...]
    pathway_first_passage_quantiles: tuple[float, ...]
    figure_formats: tuple[str, ...]
    raster_dpi: int
    renderer_style_version: str
    language: str
    source_sha256: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        """完成 schema、數值、固定 renderer policy 與 provenance hash 的基本驗證。"""

        schema_version = _require_text(self.schema_version, label="schema_version")
        if schema_version != REPORT_SPEC_SCHEMA_VERSION:
            raise ValueError("schema_version 不受支援")
        run_id = _require_slug(self.run_id, label="run_id")
        aggregate_hash = _require_sha256(
            self.aggregate_spec_canonical_sha256,
            label="aggregate_spec_canonical_sha256",
        )
        primary_bandwidth = _require_finite_positive_number(
            self.primary_kde_bandwidth_m,
            label="primary_kde_bandwidth_m",
        )
        minimum_kde_raw_count = _require_positive_native_int(
            self.minimum_kde_raw_count,
            label="minimum_kde_raw_count",
        )
        low_sample_min_member_count = _require_positive_native_int(
            self.low_sample_min_member_count,
            label="low_sample_min_member_count",
        )
        vertical_depth_bin_edges_m = _snapshot_vertical_depth_edges(
            self.vertical_depth_bin_edges_m
        )
        representative_count = _require_representative_count(
            self.representative_trajectory_count_per_site,
        )
        selection_policy = _require_text(
            self.representative_selection_policy,
            label="representative_selection_policy",
        )
        if selection_policy != _REPRESENTATIVE_SELECTION_POLICY:
            raise ValueError("representative_selection_policy 不受支援")
        selection_seed = _require_selection_seed(self.representative_selection_seed)
        travel_quantiles = _snapshot_fixed_float_tuple(
            self.travel_age_quantiles,
            label="travel_age_quantiles",
            expected=_TRAVEL_AGE_QUANTILES,
        )
        pathway_quantiles = _snapshot_fixed_float_tuple(
            self.pathway_first_passage_quantiles,
            label="pathway_first_passage_quantiles",
            expected=_PATHWAY_FIRST_PASSAGE_QUANTILES,
        )
        figure_formats = _snapshot_fixed_text_tuple(
            self.figure_formats,
            label="figure_formats",
            expected=_FIGURE_FORMATS,
        )
        if type(self.raster_dpi) is not int or self.raster_dpi != _RASTER_DPI:
            raise ValueError("raster_dpi 必須精確為原生 int 300")
        renderer_style_version = _require_text(
            self.renderer_style_version,
            label="renderer_style_version",
        )
        if renderer_style_version != _RENDERER_STYLE_VERSION:
            raise ValueError("renderer_style_version 不受支援")
        language = _require_text(self.language, label="language")
        if language != _LANGUAGE:
            raise ValueError("language 必須精確為 zh-TW")
        source_sha256 = _require_sha256(self.source_sha256, label="source_sha256")
        canonical_sha256 = _require_sha256(self.canonical_sha256, label="canonical_sha256")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "aggregate_spec_canonical_sha256", aggregate_hash)
        object.__setattr__(self, "primary_kde_bandwidth_m", primary_bandwidth)
        object.__setattr__(self, "minimum_kde_raw_count", minimum_kde_raw_count)
        object.__setattr__(self, "low_sample_min_member_count", low_sample_min_member_count)
        object.__setattr__(self, "vertical_depth_bin_edges_m", vertical_depth_bin_edges_m)
        object.__setattr__(self, "representative_trajectory_count_per_site", representative_count)
        object.__setattr__(self, "representative_selection_policy", selection_policy)
        object.__setattr__(self, "representative_selection_seed", selection_seed)
        object.__setattr__(self, "travel_age_quantiles", travel_quantiles)
        object.__setattr__(self, "pathway_first_passage_quantiles", pathway_quantiles)
        object.__setattr__(self, "figure_formats", figure_formats)
        object.__setattr__(self, "raster_dpi", _RASTER_DPI)
        object.__setattr__(self, "renderer_style_version", renderer_style_version)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "canonical_sha256", canonical_sha256)

    def to_dict(self) -> dict[str, object]:
        """回傳包含 hash、且不共享 tuple/list 的 JSON-ready plain snapshot。

        垂向邊界會以 canonical float list 輸出；其數值是相對瞬時海面的
        positive-down 深度（m），不包含 reducer 對最後 edge 外 observation 的
        overflow 計數或任何裁切結果。
        """

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "aggregate_spec_canonical_sha256": self.aggregate_spec_canonical_sha256,
            "primary_kde_bandwidth_m": self.primary_kde_bandwidth_m,
            "minimum_kde_raw_count": self.minimum_kde_raw_count,
            "low_sample_min_member_count": self.low_sample_min_member_count,
            "vertical_depth_bin_edges_m": list(self.vertical_depth_bin_edges_m),
            "representative_trajectory_count_per_site": self.representative_trajectory_count_per_site,
            "representative_selection_policy": self.representative_selection_policy,
            "representative_selection_seed": self.representative_selection_seed,
            "travel_age_quantiles": list(self.travel_age_quantiles),
            "pathway_first_passage_quantiles": list(self.pathway_first_passage_quantiles),
            "figure_formats": list(self.figure_formats),
            "raster_dpi": self.raster_dpi,
            "renderer_style_version": self.renderer_style_version,
            "language": self.language,
            "source_sha256": self.source_sha256,
            "canonical_sha256": self.canonical_sha256,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """作為 JSON object_pairs_hook，拒絕所有層級的 duplicate key。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON object 不得包含 duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_token: str) -> NoReturn:
    """拒絕 JSON decoder 額外接受的 NaN／Infinity token。"""

    _fail("JSON 不允許 NaN 或 Infinity")


def _reject_non_finite_values(value: object) -> None:
    """遞迴拒絕 1e999 等解析後才顯示為 infinity 的極大指數。"""

    if type(value) is float and not math.isfinite(value):
        _fail("JSON 數值必須是有限值")
    if type(value) is dict:
        for nested_value in value.values():
            _reject_non_finite_values(nested_value)
    elif type(value) is list:
        for nested_value in value:
            _reject_non_finite_values(nested_value)


def _read_regular_file_bytes(path: str | Path) -> bytes:
    """以 lstat、O_NOFOLLOW 與 fstat 讀取既有 non-symlink regular file。

    lstat 先拒絕路徑本身的 symbolic link，O_NOFOLLOW 防止 open 時重新跟隨連結，
    fstat 再確認實際 descriptor 是 regular file 且 identity 沒在 open race 中改變。此
    loader 只處理小型 report spec JSON；不把這個 helper 延伸到大型 trajectory array。
    """

    descriptor: int | None = None
    try:
        filesystem_path = os.fspath(path)
        if not isinstance(filesystem_path, (str, bytes)):
            _fail("path 必須是檔案系統路徑")
        link_status = os.lstat(filesystem_path)
        if stat.S_ISLNK(link_status.st_mode) or not stat.S_ISREG(link_status.st_mode):
            _fail("report spec 輸入必須是 non-symlink regular file")

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(filesystem_path, flags)
        opened_status = os.fstat(descriptor)
        if not stat.S_ISREG(opened_status.st_mode):
            _fail("report spec descriptor 必須指向 regular file")
        if (opened_status.st_dev, opened_status.st_ino) != (
            link_status.st_dev,
            link_status.st_ino,
        ):
            _fail("report spec 輸入檔在開啟期間改變")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except _ReportSpecError:
        raise
    except Exception:
        _fail("report spec 輸入檔無法讀取")
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _load_document(document: object, *, source_sha256: str) -> ReportSpec:
    """驗證已解碼 root 並建立含 raw/canonical hash 的 ReportSpec。

    root 的垂向分箱欄位必須是 JSON array；建構時會 canonical 成嚴格遞增的
    positive-down 深度 float tuple（m），最後 edge 外的 observation 不在 loader
    補入或裁切。
    """

    if type(document) is not dict:
        _fail("JSON root 必須是 object")
    if set(document) != _ROOT_KEYS:
        _fail("JSON root keys 不符合固定 report spec schema")
    for field_name in (
        "vertical_depth_bin_edges_m",
        "travel_age_quantiles",
        "pathway_first_passage_quantiles",
        "figure_formats",
    ):
        if type(document[field_name]) is not list:
            _fail(f"{field_name} 必須是 JSON array")

    provisional = ReportSpec(
        schema_version=document["schema_version"],
        run_id=document["run_id"],
        aggregate_spec_canonical_sha256=document["aggregate_spec_canonical_sha256"],
        primary_kde_bandwidth_m=document["primary_kde_bandwidth_m"],
        minimum_kde_raw_count=document["minimum_kde_raw_count"],
        low_sample_min_member_count=document["low_sample_min_member_count"],
        vertical_depth_bin_edges_m=tuple(document["vertical_depth_bin_edges_m"]),
        representative_trajectory_count_per_site=document["representative_trajectory_count_per_site"],
        representative_selection_policy=document["representative_selection_policy"],
        representative_selection_seed=document["representative_selection_seed"],
        travel_age_quantiles=tuple(document["travel_age_quantiles"]),
        pathway_first_passage_quantiles=tuple(document["pathway_first_passage_quantiles"]),
        figure_formats=tuple(document["figure_formats"]),
        raster_dpi=document["raster_dpi"],
        renderer_style_version=document["renderer_style_version"],
        language=document["language"],
        source_sha256=source_sha256,
        canonical_sha256="0" * 64,
    )
    return ReportSpec(
        schema_version=provisional.schema_version,
        run_id=provisional.run_id,
        aggregate_spec_canonical_sha256=provisional.aggregate_spec_canonical_sha256,
        primary_kde_bandwidth_m=provisional.primary_kde_bandwidth_m,
        minimum_kde_raw_count=provisional.minimum_kde_raw_count,
        low_sample_min_member_count=provisional.low_sample_min_member_count,
        vertical_depth_bin_edges_m=provisional.vertical_depth_bin_edges_m,
        representative_trajectory_count_per_site=provisional.representative_trajectory_count_per_site,
        representative_selection_policy=provisional.representative_selection_policy,
        representative_selection_seed=provisional.representative_selection_seed,
        travel_age_quantiles=provisional.travel_age_quantiles,
        pathway_first_passage_quantiles=provisional.pathway_first_passage_quantiles,
        figure_formats=provisional.figure_formats,
        raster_dpi=provisional.raster_dpi,
        renderer_style_version=provisional.renderer_style_version,
        language=provisional.language,
        source_sha256=provisional.source_sha256,
        canonical_sha256=_canonical_sha256(provisional),
    )


def load_report_spec(path: str | Path) -> ReportSpec:
    """嚴格載入 ReportSpec JSON 並由 raw bytes／canonical payload 推導兩個 hash。

    輸入檔只能是 non-symlink regular file；根層不可包含 ``source_sha256`` 或
    ``canonical_sha256``，也不可有未知／遺漏欄位。JSON 會拒絕 duplicate keys、NaN、
    Infinity 與極大指數；垂向分箱是相對瞬時海面的 positive-down 深度（m），至少兩個
    edge 且嚴格遞增，最後 edge 外的 observation 由後續 reducer 記 overflow 而非裁切；
    所有公開失敗皆固定為不含 path 的 ``ValueError``。成功 spec 只保存報告工程設定，
    不能據此宣稱 local synthetic 是正式 OCM／NWW 科學成果。
    """

    try:
        raw_bytes = _read_regular_file_bytes(path)
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        try:
            text = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("report spec 必須是嚴格 UTF-8")
        try:
            document = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except _ReportSpecError:
            raise
        except Exception:
            _fail("report spec 不是有效 JSON")
        _reject_non_finite_values(document)
        return _load_document(document, source_sha256=source_sha256)
    except Exception:
        raise ValueError("report spec 載入或驗證失敗") from None


def validate_report_spec_against_aggregate_spec(
    report_spec: ReportSpec,
    aggregate_spec: AggregateSpec,
) -> None:
    """把 report spec 的 run、aggregate canonical hash 與 primary 帶寬封閉到 aggregate。

    ``primary_kde_bandwidth_m`` 必須以數值 exact equality 選自 aggregate spec 已登錄的
    三個公尺制帶寬；run id 與 aggregate canonical SHA-256 也必須完全相等。此函式不讀檔、
    不改動任何物件，也不從 geometry 或軌跡結果猜測替代帶寬；local synthetic 通過只
    表示 binding 工程測試成立，不是真實 OCM／NWW 科學成果。
    """

    if type(report_spec) is not ReportSpec:
        raise TypeError("report_spec 必須是 exact ReportSpec")
    if type(aggregate_spec) is not AggregateSpec:
        raise TypeError("aggregate_spec 必須是 exact AggregateSpec")
    if report_spec.run_id != aggregate_spec.run_id:
        raise ValueError("report_spec run_id 與 aggregate_spec 不一致")
    if report_spec.aggregate_spec_canonical_sha256 != aggregate_spec.canonical_sha256:
        raise ValueError("report_spec aggregate canonical hash 與 aggregate_spec 不一致")
    if not any(
        report_spec.primary_kde_bandwidth_m == bandwidth
        for bandwidth in aggregate_spec.kde_bandwidths_m
    ):
        raise ValueError("report_spec primary KDE bandwidth 未精確選自 aggregate_spec")


def _require_ordinary_parent(path: Path) -> None:
    """要求 target parent 預先存在且最後節點是普通 non-symlink directory。"""

    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("report spec target parent 必須是既有普通 non-symlink directory")


def _require_target_absent(path: Path) -> None:
    """以 lstat 拒絕既有 target、directory、symlink 與 broken symlink。"""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("report spec target 無法檢查") from error
    raise _ReportSpecTargetExists("report spec target 已存在")


def _cleanup_owned_partial(
    path: Path | None,
    identity: tuple[int, int] | None,
) -> None:
    """只在 lstat identity 仍相同時清理本次已成功建立的 partial regular file。"""

    if path is None or identity is None:
        return
    try:
        metadata = os.lstat(path)
    except OSError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        return
    if (metadata.st_dev, metadata.st_ino) != identity:
        return
    with suppress(OSError):
        os.unlink(path)


def _require_regular_identity(
    path: Path,
    identity: tuple[int, int],
    *,
    message: str,
) -> None:
    """確認檔案仍是本次 ownership 的普通 non-symlink regular file。

    writer 在 load partial 後與 rename 後都會重新 lstat；不能只相信先前開啟
    descriptor 的 inode，因為 partial 路徑在小型 publish window 內可能被外部
    程序替換。這個 helper 不把路徑放入錯誤訊息，並以 device/inode 同時封閉
    路徑節點與實際檔案 identity。
    """

    try:
        metadata = os.lstat(path)
    except OSError:
        raise ValueError(message) from None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise ValueError(message)


def _fsync_target_parent(path: Path) -> None:
    """以 directory descriptor 對 target parent 做最後的目錄耐久化同步。

    parent 在 writer 開始時已要求是既有普通 non-symlink directory；這裡再以
    lstat/fstat 封閉 open race，並在可用時使用 ``O_DIRECTORY``、否則退回
    ``O_RDONLY``。descriptor 一律在 finally 關閉；若檔案系統或平台拒絕
    fsync，例外會交由 post-replace closure 轉成固定的 durability RuntimeError，
    不會把可能尚未落盤的發布回報成成功。
    """

    descriptor: int | None = None
    try:
        parent_status = os.lstat(path.parent)
        if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(parent_status.st_mode):
            raise ValueError("report spec target parent durability identity 無效")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path.parent, flags)
        opened_status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened_status.st_mode)
            or (opened_status.st_dev, opened_status.st_ino)
            != (parent_status.st_dev, parent_status.st_ino)
        ):
            raise ValueError("report spec target parent durability identity 改變")
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_report_spec(
    target: str | Path,
    *,
    aggregate_spec: AggregateSpec,
    primary_kde_bandwidth_m: int | float,
    minimum_kde_raw_count: int,
    low_sample_min_member_count: int,
    vertical_depth_bin_edges_m: tuple[int | float, ...],
    representative_trajectory_count_per_site: int,
    representative_selection_seed: int,
) -> Path:
    """建立並原子發布一份與 aggregate spec 綁定的 report spec JSON。

    target parent 必須由 caller 預先建立，且最後節點是普通 non-symlink directory；本函式
    不 mkdir，避免輸出因拼字或路徑錯誤污染 source run／aggregate release。輸入 JSON 只
    保存不含兩個衍生 hash 的固定欄位，以 compact、sort-key、UTF-8 bytes 寫入同父唯一
    partial file，再 flush／fsync、load、與完整 expected spec exact 比對及 aggregate
    binding validate；replace 前再次以 lstat 封閉 partial identity 與 target absent，
    ``os.replace`` 後再確認 final identity 並 fsync target parent directory。replace 前
    的失敗依原政策保留 target ``FileExistsError`` 或固定 ``ValueError``；replace 後
    若 identity／directory durability 無法確認，固定拋出
    ``RuntimeError('report spec durability confirmation failed')``，不刪除可能已存在
    的 final。partial 清理只使用本次檔案的 device/inode，不掃描也不刪除其他檔案。

    primary KDE 帶寬使用公尺（m），count 是原始樣本／member／每站代表軌跡的原生正整數，
    垂向分箱是相對瞬時海面的 positive-down 深度（m），且最後 edge 外的 observation
    不得被裁切；representative seed 是 128 位元無號整數，travel age 與 pathway
    quantiles 的單位是秒（s）。本 writer 只建立可重現的工程規格；synthetic local 結果
    不是真實 OCM／NWW 科學成果，也不會建立絕對來源機率或因果歸因。
    """

    partial_path: Path | None = None
    partial_identity: tuple[int, int] | None = None
    published = False
    try:
        target_path = Path(target)
        if type(aggregate_spec) is not AggregateSpec:
            raise TypeError("aggregate_spec 必須是 exact AggregateSpec")

        primary_bandwidth = _require_finite_positive_number(
            primary_kde_bandwidth_m,
            label="primary_kde_bandwidth_m",
        )
        _require_positive_native_int(
            minimum_kde_raw_count,
            label="minimum_kde_raw_count",
        )
        _require_positive_native_int(
            low_sample_min_member_count,
            label="low_sample_min_member_count",
        )
        vertical_edges = _snapshot_vertical_depth_edges(vertical_depth_bin_edges_m)
        representative_count = _require_representative_count(
            representative_trajectory_count_per_site,
        )
        _require_selection_seed(representative_selection_seed)
        if not any(
            primary_bandwidth == bandwidth
            for bandwidth in aggregate_spec.kde_bandwidths_m
        ):
            raise ValueError("primary_kde_bandwidth_m 未精確選自 aggregate_spec")

        _require_ordinary_parent(target_path.parent)
        _require_target_absent(target_path)

        provisional = ReportSpec(
            schema_version=REPORT_SPEC_SCHEMA_VERSION,
            run_id=aggregate_spec.run_id,
            aggregate_spec_canonical_sha256=aggregate_spec.canonical_sha256,
            primary_kde_bandwidth_m=primary_bandwidth,
            minimum_kde_raw_count=minimum_kde_raw_count,
            low_sample_min_member_count=low_sample_min_member_count,
            vertical_depth_bin_edges_m=vertical_edges,
            representative_trajectory_count_per_site=representative_count,
            representative_selection_policy=_REPRESENTATIVE_SELECTION_POLICY,
            representative_selection_seed=representative_selection_seed,
            travel_age_quantiles=_TRAVEL_AGE_QUANTILES,
            pathway_first_passage_quantiles=_PATHWAY_FIRST_PASSAGE_QUANTILES,
            figure_formats=_FIGURE_FORMATS,
            raster_dpi=_RASTER_DPI,
            renderer_style_version=_RENDERER_STYLE_VERSION,
            language=_LANGUAGE,
            source_sha256="0" * 64,
            canonical_sha256="0" * 64,
        )
        input_payload = provisional.to_dict()
        del input_payload["source_sha256"]
        del input_payload["canonical_sha256"]
        serialized = _canonical_json_bytes(input_payload)
        source_sha256 = hashlib.sha256(serialized).hexdigest()
        canonical_sha256 = hashlib.sha256(serialized).hexdigest()
        expected_spec = ReportSpec(
            schema_version=provisional.schema_version,
            run_id=provisional.run_id,
            aggregate_spec_canonical_sha256=provisional.aggregate_spec_canonical_sha256,
            primary_kde_bandwidth_m=provisional.primary_kde_bandwidth_m,
            minimum_kde_raw_count=provisional.minimum_kde_raw_count,
            low_sample_min_member_count=provisional.low_sample_min_member_count,
            vertical_depth_bin_edges_m=provisional.vertical_depth_bin_edges_m,
            representative_trajectory_count_per_site=provisional.representative_trajectory_count_per_site,
            representative_selection_policy=provisional.representative_selection_policy,
            representative_selection_seed=provisional.representative_selection_seed,
            travel_age_quantiles=provisional.travel_age_quantiles,
            pathway_first_passage_quantiles=provisional.pathway_first_passage_quantiles,
            figure_formats=provisional.figure_formats,
            raster_dpi=provisional.raster_dpi,
            renderer_style_version=provisional.renderer_style_version,
            language=provisional.language,
            source_sha256=source_sha256,
            canonical_sha256=canonical_sha256,
        )

        partial_path = target_path.parent / f".{target_path.name}.partial-{uuid4().hex}"
        # xb 使 partial 建立本身不可覆寫；即使 uuid 罕見碰撞，也不會改寫別人的暫存檔。
        with partial_path.open("xb") as handle:
            # 先保存 descriptor identity；若後續寫入、flush 或 fsync 失敗，仍能
            # 只清理這次確實建立的 partial，而不必以不安全的檔名猜測 ownership。
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("report spec partial 必須是 regular file")
            partial_identity = (metadata.st_dev, metadata.st_ino)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        loaded = load_report_spec(partial_path)
        if loaded != expected_spec:
            raise ValueError("report spec partial 與 expected spec 不完全相等")
        validate_report_spec_against_aggregate_spec(loaded, aggregate_spec)
        expected_identity = partial_identity
        if expected_identity is None:
            raise ValueError("report spec partial identity 尚未建立")
        _require_regular_identity(
            partial_path,
            expected_identity,
            message="report spec partial identity 驗證失敗",
        )
        _require_target_absent(target_path)
        os.replace(partial_path, target_path)
        published = True
        partial_path = None
        _require_regular_identity(
            target_path,
            expected_identity,
            message="report spec final identity 驗證失敗",
        )
        _fsync_target_parent(target_path)
        partial_identity = None
        return target_path
    except _ReportSpecTargetExists:
        _cleanup_owned_partial(partial_path, partial_identity)
        raise
    except Exception:
        if published:
            raise RuntimeError("report spec durability confirmation failed") from None
        _cleanup_owned_partial(partial_path, partial_identity)
        raise ValueError("report spec 寫入或驗證失敗") from None
