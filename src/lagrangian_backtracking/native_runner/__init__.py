"""驅動 BayTrace 原版 njit 批次主迴圈的 LBT 專用數值核心集合。

本子套件放置「直接取代或補齊 BayTrace 批次主迴圈中某一物理項」的獨立模組，例如
有限水深 Stokes 漂流批次核心（見 ``stokes_batch``）。這裡的模組只負責計算物理量並
寫入呼叫端提供的輸出陣列，不讀取原始資料檔、不管理粒子生命週期、也不修改
BayTrace 原始碼；月份選擇、缺值狀態、格網內插與粒子狀態機仍由 LBT 既有的
``forcing.py``、``engine.py`` 等控制層決定。

之所以獨立於主套件的物理模組（例如 ``stokes.py``）之外，是因為這裡實作的是
工作書「2-4 節式(7)」明確規定的簡化公式（波長由有限水深色散關係求出，垂向衰減
採深水型指數項），刻意不套用 ``stokes.py`` 內更完整的雙曲餘弦／正弦有限水深垂向
結構；兩者服務不同工作包，不應互相取代或合併。
"""

from __future__ import annotations

from .forcing_bridge import (
    ForcingBridgeError,
    ForcingGapInventoryReport,
    HourlyFieldPair,
    NativeForcingBridgeReader,
    UnregisteredForcingGapError,
    UnresolvedForcingGapError,
    inventory_forcing_gaps,
    project_face_dryness_to_nodes,
)
from .grid_bridge import (
    NativeGridArrays,
    NativeGridBridgeError,
    SigmaStaticizationReport,
    build_native_grid_arrays,
    evaluate_static_sigma_error,
)
from .particle_input import (
    BatchGroup,
    BatchLayoutReport,
    BayTraceParticleSoA,
    InitialPositionResolver,
    ParticleIdentityTable,
    ParticleInputError,
    build_baytrace_particle_soa,
    build_particle_identity_table,
    group_indices_by_arrival_time,
    group_indices_by_settling_velocity,
    group_indices_by_settling_velocity_and_arrival_time,
    load_particle_identity_table_npz,
    save_particle_identity_table_npz,
    summarize_batch_layout,
    validate_arrival_time_homogeneous_batches,
    validate_batch_partition,
    validate_settling_velocity_homogeneous_batches,
)
from .stokes_batch import (
    STOKES_BATCH_STATUS_DESCRIPTIONS,
    STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED,
    STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED,
    STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE,
    STOKES_BATCH_STATUS_INVALID_PHYSICS,
    STOKES_BATCH_STATUS_OK,
    STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN,
    STOKES_BATCH_STATUS_WAVE_MASK_INVALID,
    STOKES_BATCH_STATUS_WAVE_QC_FLAGGED,
    StokesBatchResult,
    batch_finite_depth_stokes,
    finite_depth_stokes_reference,
)

__all__ = [
    "BatchGroup",
    "BatchLayoutReport",
    "BayTraceParticleSoA",
    "ForcingBridgeError",
    "ForcingGapInventoryReport",
    "HourlyFieldPair",
    "InitialPositionResolver",
    "NativeForcingBridgeReader",
    "NativeGridArrays",
    "NativeGridBridgeError",
    "ParticleIdentityTable",
    "ParticleInputError",
    "STOKES_BATCH_STATUS_DESCRIPTIONS",
    "STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED",
    "STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED",
    "STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE",
    "STOKES_BATCH_STATUS_INVALID_PHYSICS",
    "STOKES_BATCH_STATUS_OK",
    "STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN",
    "STOKES_BATCH_STATUS_WAVE_MASK_INVALID",
    "STOKES_BATCH_STATUS_WAVE_QC_FLAGGED",
    "SigmaStaticizationReport",
    "StokesBatchResult",
    "UnregisteredForcingGapError",
    "UnresolvedForcingGapError",
    "batch_finite_depth_stokes",
    "build_baytrace_particle_soa",
    "build_native_grid_arrays",
    "build_particle_identity_table",
    "evaluate_static_sigma_error",
    "finite_depth_stokes_reference",
    "group_indices_by_arrival_time",
    "group_indices_by_settling_velocity",
    "group_indices_by_settling_velocity_and_arrival_time",
    "inventory_forcing_gaps",
    "load_particle_identity_table_npz",
    "project_face_dryness_to_nodes",
    "save_particle_identity_table_npz",
    "summarize_batch_layout",
    "validate_arrival_time_homogeneous_batches",
    "validate_batch_partition",
    "validate_settling_velocity_homogeneous_batches",
]
