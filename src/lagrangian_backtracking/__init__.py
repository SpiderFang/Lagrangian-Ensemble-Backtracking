"""三維 Lagrangian 系集逆向溯源的可重現科學計算套件。

套件只讀取上游已驗收的 OCM schema 3 與 NWW3 analysis schema 1 產品，不讀取
raw NetCDF，也不修改上游快取。公開 API 先提供設定、preflight、科學資料型別與
純 NumPy 參考核心；正式批次必須由 CLI 產生完整 manifest 與 QC 證據。aggregate
spec、payload 與 release API 只保存公尺制／秒制的條件式來源足跡工程資料；本機
synthetic 測試不等於真實 OCM／NWW3 科學成果。report spec 只登錄 renderer、抽樣與
垂向 positive-down 分箱的可重現設定；report records 則把 synthetic、SERVER pilot、
SERVER formal baseline 與 SERVER scientific evidence 分開，不能因 registry 或 hash
契約通過就把本機工程 smoke 誤認為正式海洋科學證據。套件根目錄另公開停止結果、
方向性跨站連通、來源段—受體與統一報告統計 facade，以及有效成員 pathway／環境／
材質／代表軌跡的一次串流 typed API；這些 API 只組合已驗證 payload 與比例產品，
不在統計層繪圖。report-v1 的 ``ReportRelease``、atomic writer、reader 與 validator
已完成，可對 caller 明示的 products 做來源綁定、固定拓撲與唯讀完整性驗證；共同
staging/格式/校驗基礎已完成，``report_render.py`` 提供固定格式、canonical sidecar、
實際 bytes size／SHA-256 與 immutable staging view。這仍不代表任何 F/T 科學內容已
產生：``report_pipeline.py`` 的建置前唯讀 gate 已完成，涵蓋 complete run、aggregate/spec
binding、formal trajectory v2、MPLCONFIGDIR 與 output/evidence policy；但
``build_report_release``、F01–F12/T01–T06 專屬 artifact adapters、CLI ``report-build``
與正式 SERVER 科學發布仍未完成，不能把 preflight 稱為完整 pipeline 或推定正式報告完成。
"""

from .aggregate_pipeline import build_aggregate_release_payload
from .aggregate_release import (
    read_aggregate_release,
    validate_aggregate_release,
    write_aggregate_release,
)
from .aggregate_release_payload import (
    AGGREGATE_RELEASE_SCHEMA_VERSION,
    AggregateReleasePayload,
)
from .aggregate_spec import (
    AggregateSpec,
    load_aggregate_spec,
    validate_aggregate_spec_against_boundaries,
    write_aggregate_spec_from_boundaries,
)
from .batch_state import PARTICLE_CODE_TO_STATUS, PARTICLE_STATUS_TO_CODE, ParticleBatch
from .checkpoint import (
    CheckpointBinding,
    ExecutionCheckpoint,
    load_execution_checkpoint,
    write_execution_checkpoint,
)
from .config import ProjectConfig, load_config, resolve_flow_domain_id
from .diffusion import (
    DiffusionCoefficients,
    DiffusionModel,
    DiffusionSample,
    SmagorinskySettings,
    SpatialDiffusionProvider,
    brownian_displacement,
    choose_time_step,
    diffusion_displacement,
    resolve_diffusion_sample,
    smagorinsky_horizontal_diffusivity,
)
from .engine import (
    EngineSettings,
    EnvironmentSampleStatus,
    Observation,
    ParticleAdvanceResult,
    ParticleExecutionState,
    ParticleResult,
    advance_particle_once,
    finalize_particle_execution,
    initialize_particle_execution,
    run_particle,
)
from .forcing_window import (
    ForcingCacheStats,
    ForcingWindowManager,
    ManagedForcingProvider,
    ManagedSpatialDiffusionProvider,
    MissingForcingMonth,
)
from .input_derivation import (
    ARTIFACT_FILENAMES,
    DERIVED_INPUT_SCHEMA_VERSION,
    EXPECTED_NWW_HOURLY_STEPS,
    InputDerivationError,
    InputDerivationResult,
    build_input_derivatives,
    create_release_config,
    read_canonical_json,
    validate_input_derivatives,
    validate_release_config,
    write_canonical_json,
)
from .manifests import (
    BoundaryGeometryBundle,
    ScenarioInputs,
    load_arrival_manifest,
    load_arrival_time_manifest,
    load_boundary_geometries,
    load_material_manifest,
    load_receptor_arrival_initial_condition_manifest,
    load_receptor_manifest,
    load_scenario_inputs,
    resolve_manifest_path,
)
from .models import BoundaryEvent, EventType, ParticleState, ParticleStatus, SampleQC, VelocitySample
from .outputs import read_trajectory_shard
from .pilot_calibration import (
    PAIR_SAMPLE_SCHEMA,
    PILOT_CALIBRATION_ARTIFACT_KIND,
    PILOT_CALIBRATION_FILES,
    PILOT_CALIBRATION_SCHEMA_VERSION,
    PilotCalibration,
    build_pilot_calibration,
    read_pilot_calibration,
    validate_pilot_calibration,
)
from .pilot_config import (
    PILOT_EXECUTION_BINDING_SCHEMA_VERSION,
    PILOT_EXECUTION_BINDING_STATUS,
    create_pilot_execution_config,
    validate_pilot_execution_config,
)
from .pilot_selection import (
    PILOT_SCENARIO_SELECTION_POLICY,
    PILOT_SCENARIO_SELECTION_RANKING_POLICY,
    PILOT_SCENARIO_SELECTION_SCHEMA_VERSION,
    PILOT_SCENARIO_SELECTION_STRATUM_FIELDS,
    apply_scenario_selection,
    build_full_scenario_selection,
    canonical_scenario_ids_sha256,
    scenario_ids_sha256,
    select_pilot_scenarios,
    validate_scenario_selection_binding_shape,
)
from .production import (
    HintTrackingVelocityProvider,
    ProductionAdvanceResult,
    ProductionBatch,
    ProductionParticleRuntime,
    run_production_shard,
)
from .provenance import CodeProvenance, collect_code_provenance
from .report_comparison_statistics import (
    ComparisonHDRStatus,
    ComparisonParameterDifference,
    ComparisonRatioDifference,
    ComparisonScalarDifference,
    ReportComparisonStatistics,
    build_report_comparison_statistics,
)
from .report_material_statistics import (
    FISHING_GEAR_MATERIAL_ID,
    MaterialStatistics,
    MaterialStatisticsAccumulator,
    MaterialStatisticsProduct,
    MaterialStatisticsResult,
    MaterialStatisticsSummary,
    build_material_statistics,
)
from .report_matrix_statistics import (
    ConnectivityStatistics,
    OutcomeStatistics,
    build_connectivity_statistics,
    build_outcome_statistics,
)
from .report_pipeline import ReportBuildPreflight, preflight_report_build
from .report_records import (
    REPORT_COMPARISON_ARTIFACT_IDS,
    REPORT_CORE_ARTIFACT_IDS,
    REPORT_FIGURE_IDS,
    REPORT_RELEASE_SCHEMA_VERSION,
    REPORT_TABLE_IDS,
    REPORT_VALIDATION_ARTIFACT_IDS,
    ReportArtifactRecord,
    ReportProductRef,
    ReportRegistry,
)
from .report_release import (
    ReportRelease,
    ReportReleaseWriter,
    read_report_registry,
    read_report_release,
    validate_report_release,
    write_report_release,
)
from .report_render import (
    RenderedArtifact,
    ReportStagingRenderer,
    report_render_style_context,
)
from .report_source_receptor_statistics import (
    SourceReceptorStatistic,
    SourceReceptorStatistics,
    TravelAgeStatistics,
    build_source_receptor_statistics,
)
from .report_spec import (
    REPORT_SPEC_SCHEMA_VERSION,
    ReportSpec,
    load_report_spec,
    validate_report_spec_against_aggregate_spec,
    write_report_spec,
)
from .report_statistics import ReportStatistics, build_report_statistics
from .report_trajectory_stream import (
    EnvironmentCompletenessAccumulator,
    EnvironmentCompletenessSiteStatistics,
    EnvironmentCompletenessStatistics,
    TrajectoryReportAccumulator,
    TrajectoryStreamStatistics,
    build_environment_completeness_statistics,
    build_trajectory_stream_statistics,
)
from .report_validation_evidence import (
    VALIDATION_EVIDENCE_CATEGORIES,
    VALIDATION_EVIDENCE_SCHEMA_VERSION,
    VALIDATION_METRIC_CATEGORIES,
    ValidationEvidence,
    ValidationMetric,
    load_validation_evidence,
    validate_validation_evidence,
    write_validation_evidence,
)
from .run_control import (
    RunController,
    RunExecutionSummary,
    RunWorkspace,
    checkpoint_input_binding_hash,
    initialize_run_workspace,
    load_run_plan,
    load_run_progress,
)
from .run_locking import RunLockBusyError, acquire_run_lock
from .run_validation import benchmark_report, validate_run
from .runner import (
    SCENARIO_ORDERING_POLICY,
    ReferenceParticleRequest,
    ScenarioShard,
    scenario_execution_sort_key,
)
from .runtime import (
    EXPERIMENT_CASE_INCLUDE_STOKES,
    EXPERIMENT_CASE_SPECS,
    ExperimentCaseSpec,
    ValidatedRunStaticInputs,
    initialize_formal_run,
    initialize_pilot_run,
    initialize_run,
    load_validated_run_static_inputs,
    open_formal_run_controller,
    open_pilot_run_controller,
    open_run_controller,
)
from .scenarios import ReceptorArrivalInitialCondition

__all__ = [
    "AGGREGATE_RELEASE_SCHEMA_VERSION",
    "AggregateReleasePayload",
    "AggregateSpec",
    "ARTIFACT_FILENAMES",
    "BoundaryEvent",
    "BoundaryGeometryBundle",
    "CheckpointBinding",
    "CodeProvenance",
    "ComparisonHDRStatus",
    "ComparisonParameterDifference",
    "ComparisonRatioDifference",
    "ComparisonScalarDifference",
    "ConnectivityStatistics",
    "EnvironmentCompletenessAccumulator",
    "EnvironmentCompletenessSiteStatistics",
    "EnvironmentCompletenessStatistics",
    "EnvironmentSampleStatus",
    "EngineSettings",
    "EXPERIMENT_CASE_INCLUDE_STOKES",
    "EXPERIMENT_CASE_SPECS",
    "ExperimentCaseSpec",
    "DERIVED_INPUT_SCHEMA_VERSION",
    "DiffusionCoefficients",
    "DiffusionModel",
    "DiffusionSample",
    "SmagorinskySettings",
    "ExecutionCheckpoint",
    "EventType",
    "ForcingCacheStats",
    "ForcingWindowManager",
    "HintTrackingVelocityProvider",
    "EXPECTED_NWW_HOURLY_STEPS",
    "InputDerivationError",
    "InputDerivationResult",
    "ManagedForcingProvider",
    "ManagedSpatialDiffusionProvider",
    "MissingForcingMonth",
    "Observation",
    "OutcomeStatistics",
    "PARTICLE_CODE_TO_STATUS",
    "PARTICLE_STATUS_TO_CODE",
    "PAIR_SAMPLE_SCHEMA",
    "ParticleAdvanceResult",
    "ParticleBatch",
    "ParticleExecutionState",
    "ParticleResult",
    "ParticleState",
    "ParticleStatus",
    "ProductionAdvanceResult",
    "ProductionBatch",
    "ProductionParticleRuntime",
    "ProjectConfig",
    "PILOT_CALIBRATION_ARTIFACT_KIND",
    "PILOT_CALIBRATION_FILES",
    "PILOT_CALIBRATION_SCHEMA_VERSION",
    "PilotCalibration",
    "PILOT_EXECUTION_BINDING_SCHEMA_VERSION",
    "PILOT_EXECUTION_BINDING_STATUS",
    "PILOT_SCENARIO_SELECTION_RANKING_POLICY",
    "PILOT_SCENARIO_SELECTION_POLICY",
    "PILOT_SCENARIO_SELECTION_SCHEMA_VERSION",
    "PILOT_SCENARIO_SELECTION_STRATUM_FIELDS",
    "REPORT_COMPARISON_ARTIFACT_IDS",
    "REPORT_CORE_ARTIFACT_IDS",
    "REPORT_FIGURE_IDS",
    "REPORT_RELEASE_SCHEMA_VERSION",
    "REPORT_SPEC_SCHEMA_VERSION",
    "REPORT_TABLE_IDS",
    "REPORT_VALIDATION_ARTIFACT_IDS",
    "VALIDATION_EVIDENCE_CATEGORIES",
    "VALIDATION_EVIDENCE_SCHEMA_VERSION",
    "VALIDATION_METRIC_CATEGORIES",
    "FISHING_GEAR_MATERIAL_ID",
    "MaterialStatistics",
    "MaterialStatisticsAccumulator",
    "MaterialStatisticsProduct",
    "MaterialStatisticsResult",
    "MaterialStatisticsSummary",
    "ReportArtifactRecord",
    "ReportBuildPreflight",
    "ReportComparisonStatistics",
    "ReportRelease",
    "ReportReleaseWriter",
    "ReportProductRef",
    "ReportRegistry",
    "RenderedArtifact",
    "ReportSpec",
    "ReportStagingRenderer",
    "ReportStatistics",
    "ReceptorArrivalInitialCondition",
    "RunController",
    "RunExecutionSummary",
    "RunLockBusyError",
    "RunWorkspace",
    "SCENARIO_ORDERING_POLICY",
    "ScenarioShard",
    "ScenarioInputs",
    "SampleQC",
    "SourceReceptorStatistic",
    "SourceReceptorStatistics",
    "SpatialDiffusionProvider",
    "TravelAgeStatistics",
    "TrajectoryReportAccumulator",
    "TrajectoryStreamStatistics",
    "ValidationEvidence",
    "ValidationMetric",
    "ValidatedRunStaticInputs",
    "VelocitySample",
    "advance_particle_once",
    "acquire_run_lock",
    "apply_scenario_selection",
    "benchmark_report",
    "build_aggregate_release_payload",
    "build_connectivity_statistics",
    "build_environment_completeness_statistics",
    "build_input_derivatives",
    "build_material_statistics",
    "build_outcome_statistics",
    "build_full_scenario_selection",
    "build_report_comparison_statistics",
    "preflight_report_build",
    "build_report_statistics",
    "build_source_receptor_statistics",
    "build_trajectory_stream_statistics",
    "brownian_displacement",
    "checkpoint_input_binding_hash",
    "canonical_scenario_ids_sha256",
    "collect_code_provenance",
    "create_release_config",
    "create_pilot_execution_config",
    "choose_time_step",
    "diffusion_displacement",
    "finalize_particle_execution",
    "initialize_particle_execution",
    "initialize_formal_run",
    "initialize_pilot_run",
    "initialize_run",
    "load_arrival_manifest",
    "load_arrival_time_manifest",
    "load_aggregate_spec",
    "load_boundary_geometries",
    "load_config",
    "load_execution_checkpoint",
    "load_run_plan",
    "load_run_progress",
    "load_material_manifest",
    "load_receptor_arrival_initial_condition_manifest",
    "load_receptor_manifest",
    "load_report_spec",
    "load_validation_evidence",
    "load_scenario_inputs",
    "load_validated_run_static_inputs",
    "resolve_manifest_path",
    "resolve_flow_domain_id",
    "open_formal_run_controller",
    "open_pilot_run_controller",
    "open_run_controller",
    "read_aggregate_release",
    "read_canonical_json",
    "read_pilot_calibration",
    "read_report_registry",
    "read_report_release",
    "report_render_style_context",
    "run_particle",
    "run_production_shard",
    "read_trajectory_shard",
    "ReferenceParticleRequest",
    "resolve_diffusion_sample",
    "scenario_execution_sort_key",
    "scenario_ids_sha256",
    "select_pilot_scenarios",
    "smagorinsky_horizontal_diffusivity",
    "validate_aggregate_release",
    "validate_aggregate_spec_against_boundaries",
    "validate_report_release",
    "validate_report_spec_against_aggregate_spec",
    "validate_validation_evidence",
    "validate_run",
    "validate_input_derivatives",
    "validate_pilot_calibration",
    "validate_pilot_execution_config",
    "validate_scenario_selection_binding_shape",
    "validate_release_config",
    "initialize_run_workspace",
    "write_execution_checkpoint",
    "write_aggregate_release",
    "write_aggregate_spec_from_boundaries",
    "write_report_release",
    "write_report_spec",
    "write_validation_evidence",
    "write_canonical_json",
    "build_pilot_calibration",
]

__version__ = "0.1.0"
