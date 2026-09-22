"""WP-17 任務 2：由既有 ``full_horizon_from_deposition`` 結果截斷推導較短回溯結果。

## 背景與正確性論證

兩種回溯模式（見 ``backtrack_horizon.py``）都是從同一個沉底 UTC 開始**逆向**逐小時
運算；差別只在什麼時候停止：

- ``full_horizon_from_deposition`` H：回溯滿 H 天就停（或更早因離開關注區域、沉底、
  資料缺口、數值失敗而提前終止）。
- ``fixed_calendar_window`` H：有效回溯期間是 ``H - bed_residence_age_hours``，比
  ``full_horizon_from_deposition`` 同一個 H 短（或年齡 ≥ H 時完全不運算，見
  ``NATIVE_STATUS_PRE_WINDOW_DEPOSITION``）。

因為運算本身是逐小時往回走、且任一小時的移動只依賴「到那個時刻為止已經算出的狀態」
（不依賴未來會不會被截斷），所以「以較短期間單獨重算」與「用較長期間算好之後、在
較早的那個時刻截斷」對**同一顆粒子、截斷之前**必須給出完全相同的軌跡與終止狀態；
兩者唯一可能不同的地方是「原本比較長的那次計算，之後（更早的時刻）還發生了什麼」，
而這正是截斷要丟棄的部分。這個模組因此只需要：

1. 找出較短 horizon 對應的截止 UTC（``cutoff_utc_ns``，逆向奈秒）。
2. 對截止時刻之前就已經終止的粒子：原封不動保留（它們的終止時刻／位置／狀態與
   單獨重跑一模一樣，因為終止判定只用得到截止時刻之前的資訊）。
3. 對截止時刻仍在追蹤中的粒子：改標記成「回溯達上限」，位置取截止時刻那一筆軌跡
   （逐小時取樣，見 ``hour_major.py``／``backward_driver.py`` 的既有小時輸出間隔），
   與單獨重跑在 ``substep_policy="global_cfl"`` 下的行為位元相同（``local_cfl`` 的
   子步數會依當下整批哪些粒子仍在活動決定，因此只保證統計等價，不保證位元相同，
   這是既有政策，不是本模組引入的新限制，見 ``hour_major.py`` 模組文件）。

``fixed_calendar_window`` 且整個到達時刻 ``pre_window_deposition``（年齡 ≥ H）時，
不做任何截斷計算，直接改用 ``backward_driver.build_pre_window_deposition_result``
（與 ``native-plan`` 對這種情況的處理完全一致）。

本模組**不重新執行任何流場運算**，只讀取既有 ``.npz`` 結果、切片既有陣列；因此
即使來源目錄有數萬顆粒子，推導也是純陣列操作，秒級完成。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..scenarios import ArrivalTime
from .backtrack_horizon import (
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    ResolvedUnitHorizon,
    resolve_unit_backtrack_horizon,
)
from .backward_driver import (
    NATIVE_STATUS_MAX_BACKTRACK,
    BackwardDriverResult,
    DepartureRecord,
    HourlyOverheadDiagnostics,
    SubstepDiagnostics,
    build_pre_window_deposition_result,
)

__all__ = [
    "HorizonDerivationError",
    "HorizonDerivationInfo",
    "derive_backward_driver_result",
    "derive_result_directory",
]

_HOUR_NS: Final[int] = 3_600_000_000_000


class HorizonDerivationError(ValueError):
    """來源結果不足以推導、或推導參數不合法時的 fail-closed 例外。"""


@dataclass(frozen=True, slots=True)
class HorizonDerivationInfo:
    """一次推導的稽核摘要，供 CLI 寫進 manifest／結果摘要。"""

    source_arrival_time_utc_ns: int
    target_backtrack_mode: str
    target_requested_horizon_days: float
    target_effective_horizon_seconds: float
    pre_window_deposition: bool
    bed_residence_age_hours: int | None
    cutoff_utc_ns: int
    hours_run: int
    particle_count: int
    newly_capped_particle_count: int
    unchanged_particle_count: int
    source_exhausted_before_cutoff: bool
    """來源結果本身（全部粒子）在到達這個目標 horizon 之前就已經整批終止；此時
    ``cutoff_utc_ns`` 實際上等於來源最深取樣時刻，沒有任何粒子被改標成新的
    回溯上限，推導結果與來源在（可能較短的）目標長度內完全一致。"""


def derive_backward_driver_result(
    result: BackwardDriverResult,
    *,
    resolved: ResolvedUnitHorizon,
    source_requested_horizon_days: float,
) -> tuple[BackwardDriverResult, HorizonDerivationInfo]:
    """把一個以較長 ``full_horizon_from_deposition`` H_src 算好的結果，截斷推導成
    ``resolved`` 描述的較短目標。

    Args:
        result: 來源 ``.npz`` 讀回的完整 ``BackwardDriverResult``（必須是以
            ``full_horizon_from_deposition`` 模式算出的結果——``fixed_calendar_
            window`` 的結果已經在較早的時刻停止，沒有更多資訊可以再截斷）。
        resolved: 目標 ``(模式, H)`` 的 ``backtrack_horizon.resolve_unit_backtrack_
            horizon`` 輸出（呼叫端負責解析，本函式不重新呼叫
            ``bed_residence.resolve_bed_residence_timing``，維持職責分離）。
        source_requested_horizon_days: 來源結果的 H_src（用於上限檢查，
            ``resolved.requested_horizon_days`` 必須 ``<= source_requested_horizon_
            days``，否則沒有足夠的來源資訊可以推導，fail closed）。
    """

    if resolved.requested_horizon_days > source_requested_horizon_days:
        raise HorizonDerivationError(
            f"目標 H（{resolved.requested_horizon_days} 天）大於來源 H_src"
            f"（{source_requested_horizon_days} 天）；截斷推導只能推導 H <= H_src，"
            "更長的 horizon 必須重新執行運算"
        )

    trajectory_utc_ns = np.asarray(result.trajectory_utc_ns, dtype=np.int64)
    if trajectory_utc_ns.size == 0:
        raise HorizonDerivationError("來源結果沒有任何軌跡取樣，無法推導")
    arrival_utc_ns = int(trajectory_utc_ns[0])
    n = int(result.row_particle_index.shape[0])

    if resolved.pre_window_deposition:
        derived = build_pre_window_deposition_result(
            row_particle_index=result.row_particle_index.copy(),
            x0_m=np.asarray(result.trajectory_x_m[0], dtype=np.float64).copy(),
            y0_m=np.asarray(result.trajectory_y_m[0], dtype=np.float64).copy(),
            z0_m=np.asarray(result.trajectory_z_m[0], dtype=np.float64).copy(),
            arrival_time_utc_ns=arrival_utc_ns,
        )
        info = HorizonDerivationInfo(
            source_arrival_time_utc_ns=arrival_utc_ns,
            target_backtrack_mode=resolved.backtrack_mode,
            target_requested_horizon_days=resolved.requested_horizon_days,
            target_effective_horizon_seconds=resolved.effective_horizon_seconds,
            pre_window_deposition=True,
            bed_residence_age_hours=resolved.bed_residence_age_hours,
            cutoff_utc_ns=arrival_utc_ns,
            hours_run=0,
            particle_count=n,
            newly_capped_particle_count=0,
            unchanged_particle_count=0,
            source_exhausted_before_cutoff=False,
        )
        return derived, info

    horizon_ns = int(resolved.effective_horizon_ns)
    if horizon_ns <= 0:
        raise HorizonDerivationError(
            "resolved.effective_horizon_ns 必須是正值（非 pre-window 情況下有效期間為零"
            "是內部矛盾，請檢查 resolve_unit_backtrack_horizon 的呼叫參數）"
        )

    # ── 驗證軌跡時間軸是固定間隔遞減（逆向），截止時刻才可能剛好落在某個既有
    # 取樣點；native 正式路徑固定逐小時輸出一筆（見模組文件「背景」），這裡明確
    # 檢查而不是默默假設，任何非固定間隔都 fail closed。 ──────────────────────
    if trajectory_utc_ns.size >= 2:
        step_ns = int(trajectory_utc_ns[0] - trajectory_utc_ns[1])
        if step_ns <= 0:
            raise HorizonDerivationError("trajectory_utc_ns 必須嚴格遞減")
        diffs = trajectory_utc_ns[:-1] - trajectory_utc_ns[1:]
        if not np.all(diffs == step_ns):
            raise HorizonDerivationError(
                "trajectory_utc_ns 取樣間隔不固定，無法安全截斷到指定 horizon"
            )

    requested_cutoff_utc_ns = arrival_utc_ns - horizon_ns
    deepest_utc_ns = int(trajectory_utc_ns[-1])
    source_exhausted_before_cutoff = requested_cutoff_utc_ns < deepest_utc_ns
    if source_exhausted_before_cutoff:
        # 來源結果（全部粒子）在到達這個目標 horizon 之前就已經整批終止；沒有更深
        # 的資料可以截斷，直接使用來源已有的最深取樣點，等同於「原樣回傳」。
        cutoff_utc_ns = deepest_utc_ns
        k = int(trajectory_utc_ns.size - 1)
    else:
        remainder = (arrival_utc_ns - requested_cutoff_utc_ns) % (
            step_ns if trajectory_utc_ns.size >= 2 else _HOUR_NS
        )
        if remainder != 0:
            raise HorizonDerivationError(
                "目標 horizon 對應的截止時刻沒有對齊來源軌跡的取樣間隔（預期整點小時），"
                "無法在不內插的情況下截斷"
            )
        step = step_ns if trajectory_utc_ns.size >= 2 else _HOUR_NS
        k = int((arrival_utc_ns - requested_cutoff_utc_ns) // step)
        if k >= trajectory_utc_ns.size or int(trajectory_utc_ns[k]) != requested_cutoff_utc_ns:
            raise HorizonDerivationError(
                "來源結果的軌跡長度不足以推導到這個 horizon（內部一致性檢查失敗）"
            )
        cutoff_utc_ns = requested_cutoff_utc_ns

    final_utc_ns = np.asarray(result.final_utc_ns, dtype=np.int64)
    needs_recap = final_utc_ns < cutoff_utc_ns
    newly_capped = int(np.count_nonzero(needs_recap))

    new_final_status = np.asarray(result.final_status, dtype=np.int32).copy()
    new_final_x = np.asarray(result.final_x_m, dtype=np.float64).copy()
    new_final_y = np.asarray(result.final_y_m, dtype=np.float64).copy()
    new_final_z = np.asarray(result.final_z_m, dtype=np.float64).copy()
    new_final_utc = final_utc_ns.copy()

    cutoff_x = np.asarray(result.trajectory_x_m[k], dtype=np.float64)
    cutoff_y = np.asarray(result.trajectory_y_m[k], dtype=np.float64)
    cutoff_z = np.asarray(result.trajectory_z_m[k], dtype=np.float64)

    new_final_status[needs_recap] = NATIVE_STATUS_MAX_BACKTRACK
    new_final_x[needs_recap] = cutoff_x[needs_recap]
    new_final_y[needs_recap] = cutoff_y[needs_recap]
    new_final_z[needs_recap] = cutoff_z[needs_recap]
    new_final_utc[needs_recap] = cutoff_utc_ns

    new_trajectory_utc_ns = trajectory_utc_ns[: k + 1].copy()
    new_trajectory_x_m = np.asarray(result.trajectory_x_m[: k + 1], dtype=np.float64).copy()
    new_trajectory_y_m = np.asarray(result.trajectory_y_m[: k + 1], dtype=np.float64).copy()
    new_trajectory_z_m = np.asarray(result.trajectory_z_m[: k + 1], dtype=np.float64).copy()
    new_trajectory_status = np.asarray(result.trajectory_status[: k + 1], dtype=np.int32).copy()
    # 注意：截止那一筆（索引 k）*不*改寫成新的回溯上限狀態碼，刻意保留原始值
    # （對被重新截斷的粒子通常是 ACTIVE）。這是刻意mirror既有
    # ``run_backward_batch``／``run_flow_domain_hour_major`` 的既有行為（見
    # ``hour_major._finalize_unit_result`` docstring）：回溯上限的判定發生在
    # 「這一筆軌跡取樣之後、下一小時任何移動之前」，所以 trajectory_status 的
    # 最後一筆本來就停在終止前一小時的狀態，只有獨立的 final_status／
    # final_utc_ns／final_x_m／final_y_m／final_z_m 才反映真正的終止原因與時刻。
    # 若在這裡改寫 trajectory_status，會與直接重跑的結果不再位元相同。

    original_departures_by_row = {d.row_particle_index: d for d in result.departures}
    needs_recap_rows = set(int(row) for row in np.flatnonzero(needs_recap))
    new_departures: list[DepartureRecord] = []
    for row in range(n):
        if row in needs_recap_rows:
            new_departures.append(
                DepartureRecord(
                    row_particle_index=row,
                    status=NATIVE_STATUS_MAX_BACKTRACK,
                    x_m=float(new_final_x[row]),
                    y_m=float(new_final_y[row]),
                    z_m=float(new_final_z[row]),
                    utc_ns=cutoff_utc_ns,
                )
            )
        else:
            record = original_departures_by_row.get(row)
            if record is not None:
                new_departures.append(record)

    derived = BackwardDriverResult(
        row_particle_index=result.row_particle_index.copy(),
        final_status=new_final_status,
        final_x_m=new_final_x,
        final_y_m=new_final_y,
        final_z_m=new_final_z,
        final_utc_ns=new_final_utc,
        trajectory_utc_ns=new_trajectory_utc_ns,
        trajectory_x_m=new_trajectory_x_m,
        trajectory_y_m=new_trajectory_y_m,
        trajectory_z_m=new_trajectory_z_m,
        trajectory_status=new_trajectory_status,
        departures=tuple(new_departures),
        diagnostics=SubstepDiagnostics(),
        overhead_diagnostics=HourlyOverheadDiagnostics(),
        hours_run=k,
    )
    info = HorizonDerivationInfo(
        source_arrival_time_utc_ns=arrival_utc_ns,
        target_backtrack_mode=resolved.backtrack_mode,
        target_requested_horizon_days=resolved.requested_horizon_days,
        target_effective_horizon_seconds=resolved.effective_horizon_seconds,
        pre_window_deposition=False,
        bed_residence_age_hours=resolved.bed_residence_age_hours,
        cutoff_utc_ns=cutoff_utc_ns,
        hours_run=k,
        particle_count=n,
        newly_capped_particle_count=newly_capped,
        unchanged_particle_count=n - newly_capped,
        source_exhausted_before_cutoff=source_exhausted_before_cutoff,
    )
    return derived, info


def derive_result_directory(
    *,
    source_results_dir: str | Path,
    output_dir: str | Path,
    source_backtrack_mode: str,
    source_requested_horizon_days: float,
    target_backtrack_mode: str,
    target_requested_horizon_days: float,
    arrivals_by_time_utc_ns: Mapping[int, ArrivalTime] | None = None,
    maximum_age_days: int | None = None,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    """對來源目錄下每個 ``*.driver_result.npz`` 逐一推導，寫到新目錄，不改動來源檔。

    只接受 ``source_backtrack_mode == full_horizon_from_deposition`` 的來源（見模組
    文件「背景」）：``fixed_calendar_window`` 的結果已經在較早時刻停止，沒有更多
    資訊可以再截斷成任何其他 ``(模式, H)``。

    每個檔名（不含 ``.driver_result.npz`` 副檔名）視為 ``unit_id``；到達（沉底）UTC
    直接從該檔 ``trajectory_utc_ns[0]`` 讀出（不需要另外提供計畫檔），
    ``target_backtrack_mode == fixed_calendar_window`` 時才需要 ``arrivals_by_time_
    utc_ns`` 依這個 UTC 查出對應 ``ArrivalTime`` metadata。

    回傳可直接 ``json.dumps`` 的 manifest dict（同時也會寫入
    ``<output_dir>/manifest.json``），明確記載「由 H_src 截斷推導」與其與單獨重跑
    的差異性質（見模組文件），供人工與後續 ``native-postprocess`` 核對。
    """

    if source_backtrack_mode != BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION:
        raise HorizonDerivationError(
            f"source_backtrack_mode 必須是 {BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION!r}"
            f"，收到 {source_backtrack_mode!r}；fixed_calendar_window 的結果已經截斷過，"
            "沒有更多資訊可以再推導"
        )

    # 延遲 import：hour_major_provider 匯入了 production_wiring／BayTrace handles 等
    # 較重的依賴鏈，horizon_derivation 本身的純陣列邏輯不需要它們，只在真的要讀寫
    # .npz 檔時才載入，避免把這個模組的匯入成本綁死在小時主軸執行路徑上。
    from .hour_major_provider import load_backward_driver_result_npz, save_backward_driver_result_npz

    source_dir = Path(source_results_dir)
    source_files = sorted(source_dir.glob("*.driver_result.npz"))
    if not source_files:
        raise HorizonDerivationError(f"來源目錄 {source_dir} 沒有任何 *.driver_result.npz")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)

    resolved_cache: dict[int, ResolvedUnitHorizon] = {}
    unit_manifests: dict[str, dict[str, Any]] = {}
    for source_file in source_files:
        unit_id = source_file.name[: -len(".driver_result.npz")]
        result = load_backward_driver_result_npz(source_file)
        arrival_utc_ns = int(result.trajectory_utc_ns[0])
        if arrival_utc_ns not in resolved_cache:
            arrival = (
                arrivals_by_time_utc_ns.get(arrival_utc_ns)
                if arrivals_by_time_utc_ns is not None
                else None
            )
            if arrival is None and arrivals_by_time_utc_ns is not None:
                raise HorizonDerivationError(
                    f"到達（沉底）時刻 {arrival_utc_ns}（來自 {source_file.name}）沒有對應的"
                    " ArrivalTime metadata，無法在 fixed_calendar_window 模式下解析有效回溯時間"
                )
            resolved_cache[arrival_utc_ns] = resolve_unit_backtrack_horizon(
                backtrack_mode=target_backtrack_mode,
                requested_horizon_days=target_requested_horizon_days,
                arrival=arrival,
                maximum_age_days=maximum_age_days,
                sampling_seed=sampling_seed,
            )
        resolved = resolved_cache[arrival_utc_ns]
        derived, info = derive_backward_driver_result(
            result, resolved=resolved, source_requested_horizon_days=source_requested_horizon_days
        )
        save_backward_driver_result_npz(output_path / source_file.name, derived)
        unit_manifests[unit_id] = {
            "source_file": source_file.name,
            "source_arrival_time_utc_ns": info.source_arrival_time_utc_ns,
            "target_backtrack_mode": info.target_backtrack_mode,
            "target_requested_horizon_days": info.target_requested_horizon_days,
            "target_effective_horizon_seconds": info.target_effective_horizon_seconds,
            "pre_window_deposition": info.pre_window_deposition,
            "bed_residence_age_hours": info.bed_residence_age_hours,
            "cutoff_utc_ns": info.cutoff_utc_ns,
            "hours_run": info.hours_run,
            "particle_count": info.particle_count,
            "newly_capped_particle_count": info.newly_capped_particle_count,
            "unchanged_particle_count": info.unchanged_particle_count,
            "source_exhausted_before_cutoff": info.source_exhausted_before_cutoff,
        }

    manifest = {
        "manifest_kind": "native_horizon_derivation_v1",
        "derivation_method": "truncate_existing_full_horizon_from_deposition_result_v1",
        "derivation_note_zh": (
            "本目錄的每個 *.driver_result.npz 都是由來源目錄同名檔案截斷推導而來，不是"
            "重新執行流場運算；粒子逐顆結果在截止前與來源檔完全相同。若以 "
            "substep_policy=\"global_cfl\" 直接重新執行同一個 (backtrack_mode, H) 會得到"
            "位元相同的結果，但正式生產路徑預設 substep_policy=\"local_cfl\"，其子步數依"
            "當下整批哪些粒子仍在活動決定，因此與本推導只保證統計等價，不保證位元相同"
            "（見 native_runner.hour_major 模組文件）。"
        ),
        "source_results_dir": str(source_dir),
        "source_backtrack_mode": source_backtrack_mode,
        "source_requested_horizon_days": source_requested_horizon_days,
        "target_backtrack_mode": target_backtrack_mode,
        "target_requested_horizon_days": target_requested_horizon_days,
        "unit_count": len(unit_manifests),
        "units": unit_manifests,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
