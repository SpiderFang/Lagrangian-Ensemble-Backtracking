"""WP-63：沉底「壞時刻」排除規則（確定性，純函式部分）。

動機（WP-62 診斷）：受體閘門要求候選面在「每一個沉底整點」都通過 NWW exact-hour 四角支撐與 OCM 垂向支撐。
少數整點只讓一小撮面失敗（例如 NWW 某個四角格點在那一小時缺值）時，這些面會因為「一次失敗」而整面被淘汰，
候選面積被不成比例地削薄。與其放寬閘門，不如把這些「只讓少數面失敗、但合計失敗面積很大」的整點，
從沉底可用整點中拿掉（和 NWW 站點層級的整點排除同一層），其餘閘門一字不改。

規則（輸入為失敗矩陣，輸出為排除整點清單）：
  1. 失敗矩陣 ``failed[面, 整點]``：該面在該整點未通過閘門（NWW 四角或 OCM 垂向任一失敗）。
  2. 每面的失敗整點數 ``c_f``；只保留「稀少失敗面」``min_failed_hours <= c_f <= max_failed_hours``。
  3. 每個整點的「牽連面積」＝稀少失敗面中，在該整點失敗者的面積總和。
  4. 牽連面積 >= ``area_threshold_km2`` 的整點就是壞時刻，排除。
單次通過、不迭代；輸入相同則輸出完全相同（與工作程序數、執行順序無關）。

排除清單及其 SHA-256、矩陣指紋寫進 manifest provenance，見 ``input_derivation`` 的接線。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

BAD_HOUR_RULE_POLICY_ID = "failed_area_bad_hour_rule_v1"
_HOUR_NS = 3_600_000_000_000
_CHUNK_HOURS = 512


@dataclass(frozen=True, slots=True)
class BadHourRuleParameters:
    """規則參數。``radius_m`` 只用來界定「哪些面算在內」（錨點周圍的圓），由呼叫端使用。"""

    radius_m: float
    min_failed_hours: int = 1
    max_failed_hours: int = 5
    area_threshold_km2: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.radius_m)) or float(self.radius_m) <= 0.0:
            raise ValueError("bad-hour rule radius_m 必須是有限正數")
        if (
            isinstance(self.min_failed_hours, bool)
            or isinstance(self.max_failed_hours, bool)
            or not isinstance(self.min_failed_hours, int)
            or not isinstance(self.max_failed_hours, int)
            or self.min_failed_hours < 1
            or self.max_failed_hours < self.min_failed_hours
        ):
            raise ValueError("bad-hour rule 需要 1 <= min_failed_hours <= max_failed_hours（整數）")
        if not math.isfinite(float(self.area_threshold_km2)) or float(self.area_threshold_km2) <= 0.0:
            raise ValueError("bad-hour rule area_threshold_km2 必須是有限正數")

    def to_dict(self) -> dict[str, Any]:
        return {
            "radius_m": float(self.radius_m),
            "min_failed_hours": int(self.min_failed_hours),
            "max_failed_hours": int(self.max_failed_hours),
            "area_threshold_km2": float(self.area_threshold_km2),
        }


@dataclass(frozen=True, slots=True)
class BadHourDerivation:
    """單一站點的規則結果。``affected_area_km2`` 與 ``affected_face_count`` 對齊 ``times_ns``。"""

    times_ns: tuple[int, ...]
    excluded_time_ns: tuple[int, ...]
    affected_area_km2: np.ndarray
    affected_face_count: np.ndarray
    face_count: int
    pool_area_km2: float
    rare_failure_face_count: int
    rare_failure_area_km2: float
    faces_never_failing: int
    faces_valid_after_exclusion: int
    area_valid_before_km2: float
    area_valid_after_km2: float


def _checked_inputs(
    failed: np.ndarray, face_area_m2: np.ndarray, times_ns: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(failed)
    if matrix.dtype != np.bool_ or matrix.ndim != 2:
        raise ValueError("failed 必須是二維 bool 陣列（面 × 整點）")
    area = np.asarray(face_area_m2, dtype=np.float64)
    times = np.asarray(list(times_ns), dtype=np.int64)
    if area.shape != (matrix.shape[0],) or times.shape != (matrix.shape[1],):
        raise ValueError("face_area_m2 或 times_ns 的長度與 failed 的維度不符")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("failed 不可為空")
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("面積必須是有限正數")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times_ns 必須嚴格遞增（排序且無重複）")
    if np.any(times % _HOUR_NS != 0):
        raise ValueError("times_ns 必須全部落在整點")
    return matrix, area, times


def derive_bad_hours_from_failure_matrix(
    failed: np.ndarray,
    face_area_m2: np.ndarray,
    times_ns: Sequence[int],
    parameters: BadHourRuleParameters,
) -> BadHourDerivation:
    """依規則由失敗矩陣算出排除整點。純函式、確定性。"""

    matrix, area, times = _checked_inputs(failed, face_area_m2, times_ns)
    counts = matrix.sum(axis=1)
    rare = (counts >= int(parameters.min_failed_hours)) & (counts <= int(parameters.max_failed_hours))
    rare_idx = np.flatnonzero(rare)
    affected_area_m2 = np.zeros(times.size, dtype=np.float64)
    affected_faces = np.zeros(times.size, dtype=np.int64)
    if rare_idx.size:
        rare_area = area[rare_idx]
        for start in range(0, times.size, _CHUNK_HOURS):
            block = matrix[rare_idx, start : start + _CHUNK_HOURS]
            affected_area_m2[start : start + block.shape[1]] = rare_area @ block.astype(np.float64)
            affected_faces[start : start + block.shape[1]] = block.sum(axis=0)
    threshold_m2 = float(parameters.area_threshold_km2) * 1.0e6
    excluded_mask = affected_area_m2 >= threshold_m2
    excluded = tuple(int(value) for value in times[excluded_mask])
    valid_before = counts == 0
    if (~excluded_mask).any():
        valid_after = ~np.any(matrix[:, ~excluded_mask], axis=1)
    else:
        valid_after = np.zeros(matrix.shape[0], dtype=bool)
    return BadHourDerivation(
        times_ns=tuple(int(value) for value in times),
        excluded_time_ns=excluded,
        affected_area_km2=affected_area_m2 / 1.0e6,
        affected_face_count=affected_faces,
        face_count=int(matrix.shape[0]),
        pool_area_km2=float(area.sum() / 1.0e6),
        rare_failure_face_count=int(rare_idx.size),
        rare_failure_area_km2=float(area[rare_idx].sum() / 1.0e6),
        faces_never_failing=int(valid_before.sum()),
        faces_valid_after_exclusion=int(valid_after.sum()),
        area_valid_before_km2=float(area[valid_before].sum() / 1.0e6),
        area_valid_after_km2=float(area[valid_after].sum() / 1.0e6),
    )


def hours_sha256(hours_ns: Sequence[int]) -> str:
    """與 ``restrict_available_times_to_supported`` 的 ``excluded_hours_sha256`` 同一定義。"""

    return hashlib.sha256(
        json.dumps([int(value) for value in hours_ns], separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def failure_matrix_fingerprint(
    failed: np.ndarray, face_ids: Sequence[int], times_ns: Sequence[int]
) -> str:
    """失敗矩陣、面編號與整點的 SHA-256（位元打包；輸入相同則指紋相同）。"""

    matrix = np.asarray(failed)
    digest = hashlib.sha256()
    digest.update(np.asarray(list(face_ids), dtype=np.int64).tobytes())
    digest.update(np.asarray(list(times_ns), dtype=np.int64).tobytes())
    digest.update(np.packbits(matrix, axis=None).tobytes())
    digest.update(str(matrix.shape).encode("ascii"))
    return digest.hexdigest()


def derivation_audit(
    derivation: BadHourDerivation,
    parameters: BadHourRuleParameters,
    *,
    face_ids: Sequence[int],
    failed: np.ndarray,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """寫進 manifest 的單站稽核：參數、規模、排除清單與其 SHA-256、矩陣指紋。"""

    excluded = list(derivation.excluded_time_ns)
    position = {value: index for index, value in enumerate(derivation.times_ns)}
    record: dict[str, Any] = {
        "policy_id": BAD_HOUR_RULE_POLICY_ID,
        "parameters": parameters.to_dict(),
        "candidate_hour_count": len(derivation.times_ns),
        "candidate_face_count": derivation.face_count,
        "candidate_area_km2": derivation.pool_area_km2,
        "rare_failure_face_count": derivation.rare_failure_face_count,
        "rare_failure_area_km2": derivation.rare_failure_area_km2,
        "excluded_hour_count": len(excluded),
        "excluded_hours_utc_ns": excluded,
        "excluded_hours_sha256": hours_sha256(excluded),
        "affected_area_km2_at_excluded_hours": [
            float(derivation.affected_area_km2[position[value]]) for value in excluded
        ],
        "faces_never_failing": derivation.faces_never_failing,
        "faces_valid_after_exclusion": derivation.faces_valid_after_exclusion,
        "area_valid_before_km2": derivation.area_valid_before_km2,
        "area_valid_after_km2": derivation.area_valid_after_km2,
        "failure_matrix_sha256": failure_matrix_fingerprint(failed, face_ids, derivation.times_ns),
    }
    if extra:
        record.update(dict(extra))
    return record
