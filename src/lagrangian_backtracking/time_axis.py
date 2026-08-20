"""跨月份 forcing 時間軸的決定性正規化與缺口盤點。

OCM 與 NWW3 上游產品以月份分片保存，但月份快取可包含前後 halo，且來源時間座標可能
在月界重複或倒序。粒子積分不能直接以目錄月份挑檔，否則同一 UTC 可能因 I/O 順序而
讀到不同樣本。本模組只處理一維 ``int64`` UTC nanoseconds：先依月份輸入順序串接，
再以 stable sort 排序，並對同一 UTC 保留最後出現的樣本。回傳的 chunk/local index
可同步套用到任一大型物理陣列，而不必在建立索引時載入四維 forcing。

本模組不補流速或波浪值。缺口只以相鄰 canonical UTC 的距離描述，讓後續重建器、
arrival-time selector 與 run manifest 使用同一份可稽核事實；觀測值、重建值與域外值
仍必須在物理資料層以不同 provenance/QC 保存。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

NANOSECONDS_PER_HOUR = 3_600_000_000_000
"""一小時的整數 nanoseconds；使用整數可避免缺口分類受浮點誤差影響。"""


@dataclass(frozen=True, slots=True)
class TimeChunk:
    """一個月或一個來源分片的 UTC 軸。

    ``chunk_id`` 通常是 ``YYYYMM``，只作 provenance，不參與排序優先序以外的科學判定；
    ``time_utc_ns`` 必須是一維、嚴格遞增且唯一。若來源分片本身不符合此條件，代表單檔
    已失去可安全索引性，應在進入跨月 canonicalization 前拒絕。
    """

    chunk_id: str
    time_utc_ns: np.ndarray


@dataclass(frozen=True, slots=True)
class GapInterval:
    """相鄰可用 UTC 之間缺少的規則時次區間。

    ``missing_step_count`` 不含左右兩個可用端點。例如逐時資料由 23:00 跳到 01:00，
    ``gap_hours=2``、``missing_step_count=1``。這個定義可直接對應短缺口與長缺口重建
    的 block 長度，也避免把「相鄰時距」誤寫成真正缺少的時次數。
    """

    before_utc_ns: int
    after_utc_ns: int
    gap_hours: float
    missing_step_count: int


@dataclass(frozen=True, slots=True)
class CanonicalTimeAxis:
    """跨分片 canonical UTC 軸、來源索引與完整稽核摘要。"""

    time_utc_ns: np.ndarray
    source_chunk_index: np.ndarray
    source_local_index: np.ndarray
    policy: str
    input_time_count: int
    reordered_time_step_count: int
    dropped_duplicate_time_step_count: int
    expected_timestep_hours: float
    gaps: tuple[GapInterval, ...]

    @property
    def maximum_gap_hours(self) -> float:
        """回傳 canonical 軸最大相鄰時距；無缺口時仍是正常取樣間距。"""

        if self.time_utc_ns.size < 2:
            return 0.0
        return float(np.max(np.diff(self.time_utc_ns)) / NANOSECONDS_PER_HOUR)

    @property
    def missing_step_count(self) -> int:
        """回傳全期規則逐時軸中缺少的時次總數，不計資料範圍外端點。"""

        return sum(item.missing_step_count for item in self.gaps)


def _validated_chunk_time(chunk: TimeChunk) -> np.ndarray:
    """驗證單一分片時間軸並回傳不複製的 ``int64`` view。

    來源月份若已有倒序或重複，無法只靠跨月規則區分是單檔損毀還是合法 halo，因此在
    這一層 fail closed。跨月份之間的重複才由 ``prefer_last`` 契約決定性消解。
    """

    values = np.asarray(chunk.time_utc_ns)
    if values.ndim != 1 or values.size == 0 or values.dtype != np.dtype("int64"):
        raise ValueError(f"{chunk.chunk_id} time_utc_ns 必須是非空一維 int64")
    if values.size >= 2 and np.any(np.diff(values) <= 0):
        raise ValueError(f"{chunk.chunk_id} time_utc_ns 必須嚴格遞增且唯一")
    return values


def canonicalize_time_chunks(
    chunks: Sequence[TimeChunk],
    *,
    policy: Literal["reject", "sort_and_deduplicate_prefer_last"],
    expected_timestep_hours: float,
) -> CanonicalTimeAxis:
    """建立跨月份唯一 UTC 軸並保留每筆樣本的來源位置。

    參數
    ----
    chunks:
        已按正式來源優先序排列的月份分片。相同 UTC 採 ``prefer_last`` 時，後面的 chunk
        優先；同一 chunk 內因已要求唯一，不會出現歧義。
    policy:
        ``reject`` 要求原始串接軸已嚴格遞增；
        ``sort_and_deduplicate_prefer_last`` 則沿用 OCM-SVD-Analysis 的全部可得資料契約。
    expected_timestep_hours:
        正常取樣間距。必須可精確換算為正整數 nanoseconds；目前正式資料為 1 小時。

    回傳值包含 canonical UTC、chunk/local index、重排與去重數，以及所有真正缺口。
    函式不讀取或修改任何物理欄位，因此可安全用於 metadata-only preflight。
    """

    if not chunks:
        raise ValueError("至少需要一個時間分片")
    expected_step_ns = int(round(expected_timestep_hours * NANOSECONDS_PER_HOUR))
    if expected_step_ns <= 0 or not np.isclose(
        expected_step_ns / NANOSECONDS_PER_HOUR,
        expected_timestep_hours,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("expected_timestep_hours 必須可換算為正整數 nanoseconds")

    validated = [_validated_chunk_time(chunk) for chunk in chunks]
    source_time = np.concatenate(validated)
    source_chunk = np.concatenate(
        [np.full(values.size, index, dtype=np.int64) for index, values in enumerate(validated)]
    )
    source_local = np.concatenate(
        [np.arange(values.size, dtype=np.int64) for values in validated]
    )
    original_order = np.arange(source_time.size, dtype=np.int64)

    if policy == "reject":
        if np.any(np.diff(source_time) <= 0):
            raise ValueError("跨分片 UTC 軸有倒序或重複；reject 政策不允許正規化")
        chronological_order = original_order
        retained = original_order
    elif policy == "sort_and_deduplicate_prefer_last":
        # stable sort 會保留相同 UTC 原有的 chunk/local 順序；每組取最後一筆即形成不依賴
        # worker 完成順序或物理數值內容的 deterministic prefer-last 契約。
        chronological_order = np.argsort(source_time, kind="stable")
        sorted_time = source_time[chronological_order]
        group_ends = np.flatnonzero(np.r_[np.diff(sorted_time) != 0, True])
        retained = chronological_order[group_ends]
    else:  # pragma: no cover - Literal 外仍保留 runtime 防線，避免 YAML 動態值繞過型別。
        raise ValueError(f"不支援的時間正規化政策：{policy}")

    canonical_time = source_time[retained]
    canonical_chunk = source_chunk[retained]
    canonical_local = source_local[retained]
    if canonical_time.size < 2 or np.any(np.diff(canonical_time) <= 0):
        raise ValueError("canonical UTC 軸必須至少兩筆且嚴格遞增")

    differences = np.diff(canonical_time)
    gap_indices = np.flatnonzero(differences > expected_step_ns)
    gaps: list[GapInterval] = []
    for index in gap_indices:
        span_ns = int(differences[index])
        # 非整數倍間距代表來源取樣節奏本身不規則，不能把餘數藏在 missing-step 計數中。
        if span_ns % expected_step_ns != 0:
            raise ValueError("canonical UTC 缺口不是 expected timestep 的整數倍")
        gaps.append(
            GapInterval(
                before_utc_ns=int(canonical_time[index]),
                after_utc_ns=int(canonical_time[index + 1]),
                gap_hours=span_ns / NANOSECONDS_PER_HOUR,
                missing_step_count=span_ns // expected_step_ns - 1,
            )
        )

    return CanonicalTimeAxis(
        time_utc_ns=canonical_time,
        source_chunk_index=canonical_chunk,
        source_local_index=canonical_local,
        policy=policy,
        input_time_count=int(source_time.size),
        reordered_time_step_count=int(np.count_nonzero(chronological_order != original_order)),
        dropped_duplicate_time_step_count=int(source_time.size - canonical_time.size),
        expected_timestep_hours=float(expected_timestep_hours),
        gaps=tuple(gaps),
    )
