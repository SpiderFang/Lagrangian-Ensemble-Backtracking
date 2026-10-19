"""WP-63：OCM 日檔接縫跳動（daily-restart seam jump）偵測與到達時刻候選排除。

背景（WP-61 診斷）：OCM 原始日檔在每日重啟處，海面高程會出現 η 的不連續。選擇器的
「上升／下降最快」相位以 ``dη/dt`` 的極值挑時刻，會被這類跳動誤導。本模組以固定、可重現的
規則找出跳動時刻，再把前後各 ``exclusion_radius_hours`` 小時的整點從到達時刻候選中排除。

偵測規則（V1，與 WP-61 診斷 ``diag_jumps.find_glitches`` 相同）：

* 二階差分 ``S[k] = |η[k+1] - 2 η[k] + η[k-1]|``（首尾為 NaN）；
* ``R[k] = S[k] / median(S[k-12..k-1] ∪ S[k+1..k+12])``，取 ``k`` 從 13 到 ``n-14``，鄰域中位數忽略 NaN，
  中位數為 0 或無有限值時 ``R`` 為 NaN；
* 只在 ``reference_years`` 內判定；``p95`` 是 ``reference_years`` 內 ``S`` 的有限值第 95 百分位；
* 跳動：``R >= ratio_threshold`` 且 ``S >= p95_multiplier * p95``。

``scope="union_of_sites"``（預設）：接縫是 OCM 日檔的共同屬性，因此五站各自偵測後取聯集，
每站都用同一份排除清單；``scope="own_site"`` 則每站只用自己的偵測結果。排除清單（UTC ns，遞增）
與其 SHA-256、各站偵測結果、全部參數寫入 provenance。純函式、無亂數、無 I/O。
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

SEAM_JUMP_EXCLUSION_POLICY_ID: Final[str] = "ocm_daily_seam_jump_exclusion_v1"
SEAM_JUMP_SCOPES: Final[tuple[str, ...]] = ("union_of_sites", "own_site")
NS_PER_HOUR: Final[int] = 3_600_000_000_000
_NEIGHBOUR_HALF_WIDTH_DEFAULT: Final[int] = 12


@dataclass(frozen=True, slots=True)
class SeamJumpParameters:
    """接縫跳動偵測與排除的全部門檻；預設值就是 WP-61 的 V1 定義。"""

    ratio_threshold: float = 5.0
    p95_multiplier: float = 1.5
    percentile: float = 95.0
    neighbour_half_width_hours: int = _NEIGHBOUR_HALF_WIDTH_DEFAULT
    exclusion_radius_hours: int = 1
    scope: str = "union_of_sites"

    def __post_init__(self) -> None:
        if not (np.isfinite(self.ratio_threshold) and self.ratio_threshold > 0):
            raise ValueError("ratio_threshold 必須是有限正數")
        if not (np.isfinite(self.p95_multiplier) and self.p95_multiplier >= 0):
            raise ValueError("p95_multiplier 必須是有限非負數")
        if not (np.isfinite(self.percentile) and 0.0 < self.percentile < 100.0):
            raise ValueError("percentile 必須在 (0, 100) 內")
        if isinstance(self.neighbour_half_width_hours, bool) or int(self.neighbour_half_width_hours) < 1:
            raise ValueError("neighbour_half_width_hours 必須是正整數")
        if isinstance(self.exclusion_radius_hours, bool) or int(self.exclusion_radius_hours) < 0:
            raise ValueError("exclusion_radius_hours 必須是非負整數")
        if self.scope not in SEAM_JUMP_SCOPES:
            raise ValueError(f"scope 必須是 {SEAM_JUMP_SCOPES} 之一")

    def as_payload(self) -> dict[str, Any]:
        return {
            "ratio_threshold": float(self.ratio_threshold),
            "p95_multiplier": float(self.p95_multiplier),
            "percentile": float(self.percentile),
            "neighbour_half_width_hours": int(self.neighbour_half_width_hours),
            "exclusion_radius_hours": int(self.exclusion_radius_hours),
            "scope": self.scope,
        }


def _years_of(times_ns: np.ndarray) -> np.ndarray:
    return times_ns.astype("datetime64[ns]").astype("datetime64[Y]").astype(np.int64) + 1970


def find_seam_jump_indices(
    times_ns: np.ndarray,
    elevation_m: np.ndarray,
    *,
    reference_years: Sequence[int],
    parameters: SeamJumpParameters | None = None,
) -> np.ndarray:
    """回傳 ``times_ns`` 軸上被判為接縫跳動的索引（遞增）。"""

    params = parameters or SeamJumpParameters()
    times = np.asarray(times_ns, dtype=np.int64)
    eta = np.asarray(elevation_m, dtype=np.float64)
    if times.ndim != 1 or eta.shape != times.shape:
        raise ValueError("times_ns 與 elevation_m 必須是同長度的一維陣列")
    n = int(times.size)
    half = int(params.neighbour_half_width_hours)
    if n < 2 * half + 4:
        return np.empty(0, dtype=np.int64)
    second = np.full(n, np.nan)
    second[1:-1] = np.abs(eta[2:] - 2.0 * eta[1:-1] + eta[:-2])
    ratio = np.full(n, np.nan)
    window = np.lib.stride_tricks.sliding_window_view(second, 2 * half + 1)  # window[j] 以 j+half 為中心
    centre = np.arange(half + 1, n - half - 1)  # k = half+1 .. n-half-2（與 WP-61 的 13..n-14 一致）
    neigh = np.delete(window[centre - half], half, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(neigh, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[centre] = np.where(med > 0, second[centre] / med, np.nan)
    in_reference = np.isin(_years_of(times), np.asarray([int(y) for y in reference_years], dtype=np.int64))
    ref_values = second[in_reference]
    ref_values = ref_values[np.isfinite(ref_values)]
    if ref_values.size == 0:
        return np.empty(0, dtype=np.int64)
    p_ref = float(np.percentile(ref_values, params.percentile))
    with np.errstate(invalid="ignore"):
        flagged = in_reference & (ratio >= params.ratio_threshold) & (second >= params.p95_multiplier * p_ref)
    return np.flatnonzero(flagged).astype(np.int64)


def expand_exclusion_times(jump_times_ns: Sequence[int], radius_hours: int) -> tuple[int, ...]:
    """把每個跳動時刻展開成前後各 ``radius_hours`` 小時（含自身）的整點，回傳遞增、不重複的 tuple。"""

    out: set[int] = set()
    for value in jump_times_ns:
        for k in range(-int(radius_hours), int(radius_hours) + 1):
            out.add(int(value) + k * NS_PER_HOUR)
    return tuple(sorted(out))


def sha256_of_time_list(values: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(v) for v in values], separators=(",", ":")).encode("utf-8")).hexdigest()


def _iso(value: int) -> str:
    return str(np.datetime64(int(value), "ns"))[:19] + "Z"


@dataclass(frozen=True, slots=True)
class SeamJumpExclusion:
    """排除結果：各站排除的 UTC ns tuple 與可寫進 manifest 的 provenance。"""

    excluded_time_ns_by_site: Mapping[str, tuple[int, ...]]
    provenance: Mapping[str, Any]


def build_seam_jump_exclusion(
    elevation_by_site: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    reference_years: Sequence[int],
    parameters: SeamJumpParameters | None = None,
) -> SeamJumpExclusion:
    """由各站 ``(times_ns, elevation_m)`` 產生各站的排除清單與 provenance。"""

    params = parameters or SeamJumpParameters()
    if not elevation_by_site:
        raise ValueError("elevation_by_site 不可為空")
    detected: dict[str, tuple[int, ...]] = {}
    for site_id in sorted(elevation_by_site):
        times, eta = elevation_by_site[site_id]
        times = np.asarray(times, dtype=np.int64)
        idx = find_seam_jump_indices(times, eta, reference_years=reference_years, parameters=params)
        detected[site_id] = tuple(int(times[i]) for i in idx)
    union = tuple(sorted({t for values in detected.values() for t in values}))
    excluded: dict[str, tuple[int, ...]] = {}
    for site_id in sorted(detected):
        source = union if params.scope == "union_of_sites" else detected[site_id]
        excluded[site_id] = expand_exclusion_times(source, params.exclusion_radius_hours)
    provenance = {
        "policy_id": SEAM_JUMP_EXCLUSION_POLICY_ID,
        "parameters": params.as_payload(),
        "reference_years": [int(y) for y in reference_years],
        "detected_jump_times_utc_ns_by_site": {s: list(v) for s, v in detected.items()},
        "detected_jump_times_utc_by_site": {s: [_iso(t) for t in v] for s, v in detected.items()},
        "union_jump_time_count": len(union),
        "excluded_times_by_site": {
            s: {
                "count": len(v),
                "utc_ns": list(v),
                "sha256": sha256_of_time_list(v),
            }
            for s, v in excluded.items()
        },
    }
    provenance["excluded_times_sha256_all_sites"] = hashlib.sha256(
        json.dumps({s: list(v) for s, v in sorted(excluded.items())}, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SeamJumpExclusion(excluded_time_ns_by_site=excluded, provenance=provenance)
