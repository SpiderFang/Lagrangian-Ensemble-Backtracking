"""B1b runtime request factory 的契約測試。

測試只替換 ``ForcingWindowManager.from_roots``，不建立實際 OCM/NWW forcing array，
以便驗證 request factory 的身分核對、lazy 邊界、投影位置、dynamic z、face provenance
與扁平資源統計，而不把測試耦合到上游產品檔案內容。
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import box

import lagrangian_backtracking.runtime as runtime
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.config import ProjectConfig, load_config
from lagrangian_backtracking.diffusion import DiffusionCoefficients, SmagorinskySettings
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.manifests import BoundaryGeometryBundle, ScenarioInputs
from lagrangian_backtracking.mesh import MeshLocation
from lagrangian_backtracking.preflight import (
    Finding,
    MonthInventory,
    TimeAxisInventory,
    TimeGapInventory,
)
from lagrangian_backtracking.provenance import CodeProvenance
from lagrangian_backtracking.run_control import RunController, RunWorkspace
from lagrangian_backtracking.run_validation import validate_run
from lagrangian_backtracking.runner import RunUnit
from lagrangian_backtracking.scenarios import (
    ArrivalTime,
    Behavior,
    Receptor,
    ReceptorArrivalInitialCondition,
    Scenario,
    stable_identifier,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"
SITE_ID = "gongliao"
ARRIVAL_TIME_NS = int(datetime(2025, 1, 15, tzinfo=UTC).timestamp()) * 1_000_000_000


@dataclass(frozen=True, slots=True)
class _FakeCacheStats:
    """提供 runtime resource_stats 所需的 OCM/NWW 原生整數計數。"""

    ocm_load_count: int = 2
    nww_load_count: int = 3
    ocm_cache_hit_count: int = 4
    nww_cache_hit_count: int = 5
    ocm_cache_miss_count: int = 6
    nww_cache_miss_count: int = 7
    eviction_count: int = 8
    resident_ndarray_bytes: int = 9

    @property
    def cache_hit_count(self) -> int:
        """回傳 OCM 與 NWW 的合計命中次數。"""

        return self.ocm_cache_hit_count + self.nww_cache_hit_count

    @property
    def cache_miss_count(self) -> int:
        """回傳 OCM 與 NWW 的合計未命中次數。"""

        return self.ocm_cache_miss_count + self.nww_cache_miss_count


class _FakeProvider:
    """只記錄若外部誤觸發 sample 的 facade；factory 本身不應呼叫它。"""

    def __init__(self, manager: _FakeManager) -> None:
        """保存所屬 fake manager，讓意外取樣可以留下可檢查的紀錄。"""

        self._manager = manager

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> None:
        """記錄 provider 取樣輸入；B1b 建立 request 時預期永遠不會進入此方法。"""

        self._manager.sample_calls.append((x_m, y_m, z_m, time_utc_ns))


class _FakeSpatialDiffusionProvider:
    """記錄 Smagorinsky facade wiring，不執行月份取樣。"""

    def __init__(self, manager: _FakeManager, settings: SmagorinskySettings) -> None:
        """保存 fake manager 與 immutable settings，供 request identity 測試。"""

        self.manager = manager
        self.settings = settings

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> None:
        """若誤在 request 建構時取樣便記錄並讓測試可辨識。"""

        self.manager.sample_calls.append((x_m, y_m, z_m, time_utc_ns))


class _FakeMesh:
    """回傳固定 MeshLocation 並記錄 locate 呼叫，不載入任何網格檔。"""

    def __init__(self, manager: _FakeManager, location: MeshLocation) -> None:
        """保存 fake manager 與測試指定的 face provenance。"""

        self._manager = manager
        self._location = location

    def locate(self, x_m: float, y_m: float) -> MeshLocation:
        """記錄公尺座標並回傳預先指定的 native face 位置。"""

        self._manager.locate_calls.append((x_m, y_m))
        return self._location


class _FakeManager:
    """模擬單一 flow domain manager 的 mesh、provider 與 cache 統計介面。"""

    def __init__(self, location: MeshLocation) -> None:
        """建立不含 forcing array 的 fake manager，所有副作用均以 list 記錄。"""

        self.provider_calls: list[tuple[float, bool]] = []
        self.smagorinsky_provider_calls: list[SmagorinskySettings] = []
        self.sample_calls: list[tuple[float, float, float, int]] = []
        self.preload_calls: list[tuple[tuple[str, ...], bool]] = []
        self.locate_calls: list[tuple[float, float]] = []
        self.mesh = _FakeMesh(self, location)
        self._cache_stats = _FakeCacheStats()

    @property
    def cache_stats(self) -> _FakeCacheStats:
        """回傳固定數值 snapshot，供 resource_stats 驗證扁平合計。"""

        return self._cache_stats

    def provider(self, settling_velocity_mps: float, include_stokes: bool) -> _FakeProvider:
        """記錄 provider cache key 對應的建立次數並回傳輕量 facade。"""

        self.provider_calls.append((settling_velocity_mps, include_stokes))
        return _FakeProvider(self)

    def smagorinsky_provider(self, settings: SmagorinskySettings) -> _FakeSpatialDiffusionProvider:
        """記錄 Smagorinsky facade 建立，確保不會在 factory 階段載入月份。"""

        self.smagorinsky_provider_calls.append(settings)
        return _FakeSpatialDiffusionProvider(self, settings)

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        **_: Any,
    ) -> None:
        """提供不應被 factory 使用的 manager sample 觀測點。"""

        self.sample_calls.append((x_m, y_m, z_m, time_utc_ns))

    def preload(self, month_ids: tuple[str, ...], *, include_stokes: bool = False) -> None:
        """提供不應被 factory 使用的 preload 觀測點。"""

        self.preload_calls.append((month_ids, include_stokes))


class _UninspectablePath:
    """呼叫 ``__fspath__`` 即失敗的 sentinel，用來證明 no-Stokes 不查 NWW root。"""

    def __fspath__(self) -> str:
        """若 runtime 觸碰此路徑便讓測試立即失敗。"""

        raise AssertionError("no_stokes 不得探查傳入的 NWW path")


def _configured_project_config() -> ProjectConfig:
    """由範例 YAML 載入真正 ProjectConfig，再以 model_copy 注入 B1b scalar。"""

    config = load_config(EXAMPLE_CONFIG)
    integration = config.integration.model_copy(
        update={
            "dt_min_seconds": 1.0,
            "dt_max_seconds": 10.0,
            "output_interval_seconds": 60.0,
        }
    )
    boundaries = config.boundaries.model_copy(
        update={"max_backtrack_days": 7.0, "maximum_step_count": 1_000}
    )
    scenarios = config.scenarios.model_copy(
        update={
            "members_per_scenario": 2,
            "master_seed": 20260828,
            "seed_policy": "sha256_v1_pcg64dxsm",
        }
    )
    execution = config.execution.model_copy(
        update={
            "max_resident_forcing_months": 2,
            "shard_scenario_count": 1,
            "checkpoint_interval_sweeps": 1,
            "active_chunk_size": 2,
            "production_backend": "numpy_reference",
        }
    )
    physics = deepcopy(config.physics)
    physics["horizontal_diffusion"]["constant_kh_m2ps"] = 2.5
    physics["vertical_diffusion"]["constant_kz_m2ps"] = 0.01
    configured = config.model_copy(
        update={
            "integration": integration,
            "boundaries": boundaries,
            "scenarios": scenarios,
            "execution": execution,
            "physics": physics,
        }
    )
    assert isinstance(configured, ProjectConfig)
    return configured


def _smagorinsky_project_config(
    data: dict[str, Any],
    *,
    floor: Any = 0.01,
    cap: Any = 10.0,
    sensitivity: Any = (0.10, 0.15, 0.20),
) -> ProjectConfig:
    """將 runtime fixture 轉成可執行的 Smagorinsky 設定。

    fixture 原本的 constant Kh 仍保留在範例設定；這裡把它設為 null，驗證 Smagorinsky
    case 不會誤讀常數 Kh，並只使用 sensitivity、Kh floor/cap 與共同 constant Kz。
    ``Any`` 讓參數化測試能注入 null、bool、非有限值或錯誤型別，檢查 parser 的
    fail-closed 行為而不依賴 Pydantic 先行轉型。
    """

    physics = deepcopy(data["config"].physics)
    horizontal = physics["horizontal_diffusion"]
    horizontal["constant_kh_m2ps"] = None
    horizontal["smagorinsky"] = {
        "cs_sensitivity": sensitivity,
        "kh_floor_m2ps": floor,
        "kh_cap_m2ps": cap,
    }
    physics["vertical_diffusion"]["constant_kz_m2ps"] = 0.01
    return data["config"].model_copy(update={"physics": physics})


def _single_behavior() -> Behavior:
    """建立一筆具有明確 sinking 行為與負 settling velocity 的 material。"""

    return Behavior(
        material_id="material-1",
        oca_category_zh="測試類別",
        material_family_zh="測試材質",
        representative_shape_zh="測試形狀",
        settling_velocity_mps=-0.2,
        behavior_class="sinking",
        applicability_condition_zh="僅供測試",
        calibration_status="pilot",
        evidence_grade="test",
    )


def _scenario_records(
    config: ProjectConfig,
) -> tuple[
    Behavior,
    Receptor,
    ArrivalTime,
    ReceptorArrivalInitialCondition,
    Scenario,
]:
    """建立單一 scenario 及其 dynamic pair，刻意讓模板 z 與實際 z 不同。"""

    site = next(item for item in config.study_sites if item.study_site_id == SITE_ID)
    behavior = _single_behavior()
    receptor = Receptor(
        receptor_id="receptor-1",
        study_site_id=site.study_site_id,
        analysis_region_id=site.analysis_region_id,
        lon=121.9,
        lat=25.0,
        z_m_positive_up=-3.0,
        vertical_id="vertical-1",
        metadata={"template_role": "deliberately_not_runtime_z"},
    )
    arrival = ArrivalTime(
        arrival_time_id="arrival-1",
        study_site_id=site.study_site_id,
        time_utc_ns=ARRIVAL_TIME_NS,
        year=2025,
        season="DJF",
        tide_class="spring_proxy",
        phase_or_event="strong_current_event",
        metadata={},
    )
    pair = ReceptorArrivalInitialCondition(
        receptor_id=receptor.receptor_id,
        arrival_time_id=arrival.arrival_time_id,
        study_site_id=site.study_site_id,
        analysis_region_id=site.analysis_region_id,
        flow_domain_id=site.flow_domain_id,
        time_utc_ns=arrival.time_utc_ns,
        vertical_id=receptor.vertical_id,
        z_m_positive_up=-17.0,
        eta_m_positive_up=-1.0,
        bed_z_m_positive_up=-20.0,
        water_column_height_m=19.0,
        height_above_bed_m=3.0,
        zcor_lower_m_positive_up=-18.0,
        zcor_upper_m_positive_up=-16.0,
        vertical_bracket_alpha=0.5,
        source_face_local_index=4,
        source_face_global_index=4004,
        wetdry_elem_value=0,
        wetdry_semantics_id="schism_wetdry_elem_zero_is_wet",
        ocm_month_yyyymm="202501",
        ocm_source_time_index=0,
        ocm_time_origin="observed",
    )
    scenario = Scenario(
        scenario_id=stable_identifier(
            "scn",
            [
                site.study_site_id,
                behavior.material_id,
                receptor.receptor_id,
                arrival.arrival_time_id,
                config.design_version,
            ],
        ),
        study_site_id=site.study_site_id,
        analysis_region_id=site.analysis_region_id,
        material_id=behavior.material_id,
        receptor_id=receptor.receptor_id,
        arrival_time_id=arrival.arrival_time_id,
        settling_velocity_mps=behavior.settling_velocity_mps,
        arrival_time_utc_ns=arrival.time_utc_ns,
        design_version=config.design_version,
    )
    return behavior, receptor, arrival, pair, scenario


def _scenario_inputs(
    behavior: Behavior,
    receptor: Receptor,
    arrival: ArrivalTime,
    pair: ReceptorArrivalInitialCondition,
    scenarios: tuple[Scenario, ...],
) -> ScenarioInputs:
    """以 tuple 建立不含 forcing 的 ScenarioInputs 與自動 pair index。"""

    return ScenarioInputs(
        materials=(behavior,),
        receptors=(receptor,),
        arrival_times=(arrival,),
        scenarios=scenarios,
        file_sha256={},
        canonical_component_hashes={},
        design_version=scenarios[0].design_version,
        initial_conditions=(pair,),
    )


def _geometry_bundle(receptor: Receptor) -> BoundaryGeometryBundle:
    """以 receptor 投影點周圍的公尺制 box 建立 local/flow 幾何 bundle。"""

    projection = DomainProjection(receptor.lon, receptor.lat)
    domain = box(-100.0, -100.0, 100.0, 100.0)
    geometry = BoundaryGeometry(
        own_local_domain=domain,
        flow_domain=domain,
        foreign_local_domains={},
        local_equals_flow=True,
    )
    return BoundaryGeometryBundle(
        geometries={receptor.study_site_id: geometry},
        projections={receptor.study_site_id: projection},
        file_sha256={},
        canonical_component_hashes={},
    )


@pytest.fixture
def runtime_fixture() -> dict[str, Any]:
    """提供所有 B1b 測試共用的實際設定、manifest records 與幾何。"""

    config = _configured_project_config()
    behavior, receptor, arrival, pair, scenario = _scenario_records(config)
    inputs = _scenario_inputs(behavior, receptor, arrival, pair, (scenario,))
    geometries = _geometry_bundle(receptor)
    inputs = replace(
        inputs,
        canonical_component_hashes={
            "material": "1" * 64,
            "receptor": "2" * 64,
            "arrival": "3" * 64,
            "receptor_arrival_initial_condition": "4" * 64,
        },
    )
    geometries = replace(
        geometries,
        canonical_component_hashes={
            "domain": "5" * 64,
            "local": "6" * 64,
            "open_boundary": "7" * 64,
        },
    )
    return {
        "config": config,
        "behavior": behavior,
        "receptor": receptor,
        "arrival": arrival,
        "pair": pair,
        "scenario": scenario,
        "inputs": inputs,
        "geometries": geometries,
    }


def _unit(scenario: Scenario, *, case_id: str = "no_stokes", member_id: int = 1) -> RunUnit:
    """建立只改 member／particle／seed 的固定 RunUnit。"""

    return RunUnit(
        scenario=scenario,
        experiment_case_id=case_id,
        member_id=member_id,
        particle_id=f"particle-{member_id}",
        seed=1000 + member_id,
    )


def _factory(
    data: dict[str, Any],
    tmp_path: Path,
    *,
    case_id: str = "no_stokes",
    nww_root: str | Path | object | None = None,
    geometries: BoundaryGeometryBundle | None = None,
    run_kind: str = "pilot",
) -> runtime.RuntimeRequestFactory:
    """用 ordinary OCM root 建立 pilot/formal factory；manager 由各測試 monkeypatch。"""

    ocm_root = tmp_path / "ocm-native"
    ocm_root.mkdir(exist_ok=True)
    return runtime.RuntimeRequestFactory(
        config=data["config"],
        scenario_inputs=data["inputs"],
        geometries=geometries or data["geometries"],
        ocm_native_root=ocm_root,
        nww_analysis_root=nww_root,
        experiment_case_id=case_id,
        run_kind=run_kind,
    )


def _patch_from_roots(monkeypatch: pytest.MonkeyPatch, location: MeshLocation):
    """以 staticmethod monkeypatch manager factory，並回傳呼叫參數與 fake managers。"""

    from_roots_calls: list[dict[str, Any]] = []
    managers: list[_FakeManager] = []

    def from_roots(**kwargs: Any) -> _FakeManager:
        """記錄 runtime 傳入的 flow、projection、root 與 cache 上限。"""

        from_roots_calls.append(kwargs)
        manager = _FakeManager(location)
        managers.append(manager)
        return manager

    monkeypatch.setattr(runtime.ForcingWindowManager, "from_roots", staticmethod(from_roots))
    return from_roots_calls, managers


def _matching_location(data: dict[str, Any]) -> MeshLocation:
    """建立與 dynamic pair local/global face provenance 完全相同的位置。"""

    pair = data["pair"]
    return MeshLocation(
        triangle_id=7,
        source_face_local_index=pair.source_face_local_index,
        source_face_global_index=pair.source_face_global_index,
        node_indices=(0, 1, 2),
        barycentric_weights=(0.2, 0.3, 0.5),
        triangle_area_m2=50.0,
    )


def test_experiment_case_registry_is_exact_immutable_and_compatibility_mapping_is_derived() -> None:
    """五個固定案例由 frozen spec 唯一登錄，衍生 Stokes mapping 仍不可改寫。"""

    assert list(runtime.EXPERIMENT_CASE_SPECS) == [
        "finite_depth_stokes",
        "no_stokes",
        "smagorinsky_cs_010",
        "smagorinsky_cs_015",
        "smagorinsky_cs_020",
    ]
    assert [
        runtime.EXPERIMENT_CASE_SPECS[case_id].coefficient_cs
        for case_id in runtime.EXPERIMENT_CASE_SPECS
    ] == [None, None, 0.10, 0.15, 0.20]
    assert dict(runtime.EXPERIMENT_CASE_INCLUDE_STOKES) == {
        "finite_depth_stokes": True,
        "no_stokes": False,
        "smagorinsky_cs_010": True,
        "smagorinsky_cs_015": True,
        "smagorinsky_cs_020": True,
    }
    assert all(
        runtime.EXPERIMENT_CASE_SPECS[case_id].include_stokes
        == runtime.EXPERIMENT_CASE_INCLUDE_STOKES[case_id]
        for case_id in runtime.EXPERIMENT_CASE_SPECS
    )
    with pytest.raises(TypeError):
        runtime.EXPERIMENT_CASE_INCLUDE_STOKES["unexpected"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        runtime.EXPERIMENT_CASE_SPECS["unexpected"] = runtime.ExperimentCaseSpec(  # type: ignore[index]
            "unexpected", False, "constant"
        )
    with pytest.raises(FrozenInstanceError):
        runtime.EXPERIMENT_CASE_SPECS["no_stokes"].include_stokes = True  # type: ignore[misc]


def test_valid_no_stokes_is_lazy_shared_and_uses_dynamic_pair(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """no-Stokes 兩個 member 應共用 manager/provider，且只使用 pair 的實際 z。"""

    data = runtime_fixture
    calls, managers = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path, nww_root=_UninspectablePath())
    assert factory.resource_stats() == {
        "manager_count": 0,
        "loads": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "resident_bytes": 0,
    }

    first = factory(_unit(data["scenario"], member_id=1))
    second = factory(_unit(data["scenario"], member_id=2))

    assert len(calls) == 1
    assert calls[0]["flow_domain_id"] == data["pair"].flow_domain_id
    assert calls[0]["nww_root"] is None
    assert calls[0]["ocm_root"] == tmp_path / "ocm-native"
    assert calls[0]["max_resident_months"] == 2
    manager = managers[0]
    assert manager.provider_calls == [(-0.2, False)]
    assert manager.sample_calls == []
    assert manager.preload_calls == []
    assert manager.locate_calls
    assert first.velocity is second.velocity
    assert first.diffusion == DiffusionCoefficients(2.5, 2.5, 0.01)

    state = first.initial_state
    projection = data["geometries"].projections[SITE_ID]
    expected_x, expected_y = projection.project(data["receptor"].lon, data["receptor"].lat)
    assert state.x_m == pytest.approx(float(expected_x))
    assert state.y_m == pytest.approx(float(expected_y))
    assert state.z_m == -17.0
    assert state.z_m != data["receptor"].z_m_positive_up
    assert state.time_utc_ns == data["scenario"].arrival_time_utc_ns
    assert state.particle_id == "particle-1"
    assert state.scenario_id == data["scenario"].scenario_id
    assert state.member_id == 1
    assert state.study_site_id == data["scenario"].study_site_id
    assert state.analysis_region_id == data["scenario"].analysis_region_id
    assert state.receptor_id == data["scenario"].receptor_id
    assert first.behavior_class == "sinking"
    assert first.diffusion.kx_m2ps == 2.5
    assert first.diffusion.ky_m2ps == 2.5
    assert first.diffusion.kz_m2ps == 0.01
    assert first.settings.dt_min_seconds == 1.0
    assert first.settings.dt_max_seconds == 10.0
    assert first.settings.output_interval_seconds == 60.0
    assert first.settings.max_backtrack_seconds == 7.0 * 86400.0
    assert first.settings.earliest_forcing_time_utc_ns == (
        data["scenario"].arrival_time_utc_ns - 7 * 86400 * 1_000_000_000
    )
    assert first.settings.maximum_minimum_clamps == 100


@pytest.mark.parametrize(
    ("case_id", "expected_cs"),
    [
        ("smagorinsky_cs_010", 0.10),
        ("smagorinsky_cs_015", 0.15),
        ("smagorinsky_cs_020", 0.20),
    ],
)
def test_smagorinsky_cases_share_lazy_diffusion_facade_and_keep_stokes_velocity(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    expected_cs: float,
) -> None:
    """三個 Cs case 應接上正確 settings、共享 diffusion facade 且仍要求 Stokes NWW。"""

    data = dict(runtime_fixture)
    data["config"] = _smagorinsky_project_config(data)
    nww_root = tmp_path / "nww-analysis"
    nww_root.mkdir()
    calls, managers = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path, case_id=case_id, nww_root=nww_root)
    first = factory(_unit(data["scenario"], case_id=case_id, member_id=1))
    second = factory(_unit(data["scenario"], case_id=case_id, member_id=2))

    assert len(calls) == 1
    assert calls[0]["nww_root"] == nww_root
    manager = managers[0]
    assert manager.provider_calls == [(-0.2, True)]
    assert len(manager.smagorinsky_provider_calls) == 1
    settings = manager.smagorinsky_provider_calls[0]
    assert settings == SmagorinskySettings(expected_cs, 0.01, 10.0, 0.01)
    assert first.diffusion is second.diffusion
    assert first.diffusion.settings is settings
    assert manager.sample_calls == []
    assert manager.preload_calls == []


def test_constant_cases_ignore_null_smagorinsky_placeholder(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """常數案例仍可在 example config 的 null floor/cap placeholder 上建立 request。"""

    data = dict(runtime_fixture)
    physics = deepcopy(data["config"].physics)
    physics["horizontal_diffusion"]["smagorinsky"] = None
    data["config"] = data["config"].model_copy(update={"physics": physics})
    _patch_from_roots(monkeypatch, _matching_location(data))
    request = _factory(data, tmp_path, case_id="no_stokes", nww_root=_UninspectablePath())(
        _unit(data["scenario"], case_id="no_stokes")
    )
    assert request.diffusion == DiffusionCoefficients(2.5, 2.5, 0.01)


@pytest.mark.parametrize(
    ("smag_override", "expected_match"),
    [
        (None, "smagorinsky 必須是 mapping"),
        ({}, "缺少 physics.horizontal_diffusion.smagorinsky.cs_sensitivity"),
        ({"cs_sensitivity": []}, "cs_sensitivity 不可為空"),
        ({"cs_sensitivity": [0.10, 0.10]}, "cs_sensitivity 不可包含重複值"),
        ({"cs_sensitivity": [True, 0.15, 0.20]}, r"cs_sensitivity\[0\]"),
        ({"cs_sensitivity": [0.10, -0.15, 0.20]}, r"cs_sensitivity\[1\]"),
        ({"cs_sensitivity": [0.10, math.nan, 0.20]}, r"cs_sensitivity\[1\]"),
        ({"cs_sensitivity": [0.15, 0.20]}, "coefficient_cs 必須列在"),
        ({"cs_sensitivity": [0.10, 0.15, 0.20], "kh_floor_m2ps": None}, "kh_floor_m2ps"),
        ({"cs_sensitivity": [0.10, 0.15, 0.20], "kh_floor_m2ps": -1.0}, "kh_floor_m2ps 必須 >="),
        ({"cs_sensitivity": [0.10, 0.15, 0.20], "kh_cap_m2ps": None}, "kh_cap_m2ps"),
        ({"cs_sensitivity": [0.10, 0.15, 0.20], "kh_cap_m2ps": math.inf}, "kh_cap_m2ps"),
        (
            {
                "cs_sensitivity": [0.10, 0.15, 0.20],
                "kh_floor_m2ps": 11.0,
                "kh_cap_m2ps": 10.0,
            },
            "kh_floor_m2ps 不可大於 kh_cap_m2ps",
        ),
    ],
)
def test_invalid_smagorinsky_config_fails_closed(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    smag_override: Any,
    expected_match: str,
) -> None:
    """Smagorinsky parser 的各項錯誤需在有效 root gate 後以精確原因拒絕。"""

    data = dict(runtime_fixture)
    physics = deepcopy(data["config"].physics)
    smag = {
        "cs_sensitivity": [0.10, 0.15, 0.20],
        "kh_floor_m2ps": 0.01,
        "kh_cap_m2ps": 10.0,
    }
    # None 與空 mapping 是測試「mapping 本身缺失／欄位缺失」的替換案例；其餘 dict
    # 只覆寫一個錯誤欄位，讓測試先通過 OCM/NWW root gate 後確實抵達 parser 的對應分支。
    if smag_override is None or smag_override == {}:
        actual_smag = smag_override
    else:
        smag.update(smag_override)
        actual_smag = smag
    physics["horizontal_diffusion"]["constant_kh_m2ps"] = None
    physics["horizontal_diffusion"]["smagorinsky"] = actual_smag
    data["config"] = data["config"].model_copy(update={"physics": physics})
    ocm_root = tmp_path / "ocm-native"
    nww_root = tmp_path / "nww-analysis"
    ocm_root.mkdir()
    nww_root.mkdir()
    with pytest.raises((TypeError, ValueError), match=expected_match):
        runtime.RuntimeRequestFactory(
            config=data["config"],
            scenario_inputs=data["inputs"],
            geometries=data["geometries"],
            ocm_native_root=ocm_root,
            nww_analysis_root=nww_root,
            experiment_case_id="smagorinsky_cs_010",
        )


def test_second_material_same_flow_has_separate_provider_cache(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一 flow 的第二 material 重用 manager，但 provider key 必須區分 settling。"""

    data = runtime_fixture
    second_behavior = replace(
        data["behavior"], material_id="material-2", settling_velocity_mps=-0.5
    )
    second_scenario = replace(
        data["scenario"],
        scenario_id="scenario-2",
        material_id=second_behavior.material_id,
        settling_velocity_mps=second_behavior.settling_velocity_mps,
    )
    inputs = replace(
        data["inputs"],
        materials=(data["behavior"], second_behavior),
        scenarios=(data["scenario"], second_scenario),
    )
    data = {**data, "inputs": inputs}
    _, managers = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path)

    first = factory(_unit(data["scenario"], member_id=1))
    second = factory(_unit(second_scenario, member_id=1))

    assert len(managers) == 1
    assert managers[0].provider_calls == [(-0.2, False), (-0.5, False)]
    assert first.velocity is not second.velocity


@pytest.mark.parametrize("failure_kind", ["scenario", "experiment", "pair", "missing_pair"])
def test_identity_failures_happen_before_manager(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """scenario、case、pair 任一身分矛盾都必須在 manager 建立前拒絕。"""

    data = runtime_fixture
    if failure_kind == "scenario":
        scenario = replace(data["scenario"], settling_velocity_mps=-99.0)
        unit = _unit(scenario)
    elif failure_kind == "experiment":
        unit = _unit(data["scenario"], case_id="finite_depth_stokes")
    elif failure_kind == "pair":
        bad_pair = replace(data["pair"], time_utc_ns=data["pair"].time_utc_ns + 1)
        data = {
            **data,
            "inputs": _scenario_inputs(
                data["behavior"],
                data["receptor"],
                data["arrival"],
                bad_pair,
                (data["scenario"],),
            ),
        }
        unit = _unit(data["scenario"])
    else:
        missing_inputs = replace(
            data["inputs"], initial_conditions=(), initial_conditions_by_pair={}
        )
        data = {**data, "inputs": missing_inputs}
        unit = _unit(data["scenario"])

    calls, _ = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path)
    with pytest.raises(ValueError):
        factory(unit)
    assert calls == []


def test_fractional_horizon_nanoseconds_fails_before_manager(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1e-12 天換算為 86.4 ns 時，必須拒絕而不得 round、floor 或 ceil。"""

    data = runtime_fixture
    boundaries = data["config"].boundaries.model_copy(
        update={"max_backtrack_days": 1e-12}
    )
    data = {**data, "config": data["config"].model_copy(update={"boundaries": boundaries})}
    calls, _ = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path)
    with pytest.raises(ValueError, match="horizon ns"):
        factory(_unit(data["scenario"]))
    assert calls == []


@pytest.mark.parametrize("hours", range(1, 73))
def test_integer_hour_horizons_restore_exact_nanoseconds(hours: int) -> None:
    """整數一至七十二小時以天數 float 傳入，必須還原同一秒數與奈秒，不移動時間窗。"""

    assert runtime._backtrack_horizon(hours / 24) == (hours * 3600.0, hours * 3_600_000_000_000)


@pytest.mark.parametrize("days", [1.0, 7.0, 30.0, 365.0, 0.1, 0.3, 0.1234, 365.123456])
def test_existing_exact_decimal_days_are_unchanged(days: float) -> None:
    """既有十進位天數已能精確轉奈秒的值優先保留，不讓浮點正規化改變原有整數結果。"""

    expected_ns = Decimal(str(days)) * Decimal(86_400) * Decimal(1_000_000_000)
    assert expected_ns == expected_ns.to_integral_value()
    expected = (float(expected_ns / Decimal(1_000_000_000)), int(expected_ns))
    assert runtime._backtrack_horizon(days) == expected
    # 共用工具不得受呼叫端 Decimal 的低精度影響而默默把次奈秒捨入成整數。
    with localcontext() as context:
        context.prec = 6
        assert runtime._backtrack_horizon(days) == expected
        with pytest.raises(ValueError, match="horizon ns"):
            runtime._backtrack_horizon(1e-12)


@pytest.mark.parametrize(("days", "expected_ns"), [
    (1 / 24, 3_600_000_000_000), (1 / 48, 1_800_000_000_000),
    (7 / 24, 25_200_000_000_000), (1e-9 / 86_400, 1), (1e-8 / 86_400, 10), (7.0, 604_800_000_000_000),
])
def test_fractional_day_factory_and_formal_gap_safe_share_exact_window(
    runtime_fixture, tmp_path, monkeypatch, days, expected_ns,
) -> None:
    """真實 factory request 與正式缺口閘門使用同一奈秒窗，窗口起點的 1 ns 邊界不可改動。"""

    data = _formal_test_data(runtime_fixture, gap_safe=True)
    config = data["config"].model_copy(update={
        "boundaries": data["config"].boundaries.model_copy(update={"max_backtrack_days": days}),
    })
    data = {**data, "config": config}
    calls, _ = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path, run_kind="formal")
    unit = _unit(data["scenario"])
    request = factory(unit)
    arrival_ns = unit.scenario.arrival_time_utc_ns
    earliest_ns = arrival_ns - expected_ns
    assert request.settings.max_backtrack_seconds == expected_ns / 1_000_000_000
    assert request.settings.earliest_forcing_time_utc_ns == earliest_ns
    assert request.initial_state.time_utc_ns == arrival_ns
    assert request.initial_state.scenario_id == unit.scenario.scenario_id
    assert request.initial_state.member_id == unit.member_id and unit.seed == 1001
    assert len(calls) == 1
    flow = runtime.resolve_flow_domain_id(config, data["scenario"].analysis_region_id, formal=True)
    flows = runtime._formal_flow_domain_ids(config)
    # 缺口終點在窗口前一奈秒時可通過，碰到起點時則拒絕；兩端仍是閉區間。
    safe_axes = [("ocm_native", key, 3_600_000_000_000,
                  ((earliest_ns - 1_000_000_000, earliest_ns - 1),) if key == flow else ()) for key in flows]
    runtime._validate_formal_ocm_gap_support(config, data["inputs"], ocm_axes=safe_axes)
    touching_axes = [(product, key, step, ((earliest_ns - 1_000_000_000, earliest_ns),) if gaps else ())
                     for product, key, step, gaps in safe_axes]
    with pytest.raises(ValueError, match="相交"):
        runtime._validate_formal_ocm_gap_support(config, data["inputs"], ocm_axes=touching_axes)


@pytest.mark.parametrize("days", [
    1e-12, 1.05e-8 / 86_400, math.nextafter(1 / 24, math.inf), math.nextafter(1 / 24, 0.0),
    float("inf"), float("nan"), 1e308, 107_000.0, 0.0, -1.0, None, True, "1",
])
def test_unrepresentable_horizon_still_fails_before_manager(
    runtime_fixture, tmp_path, monkeypatch, days,
) -> None:
    """真正次奈秒、可分辨的鄰值、非有限／溢位及非法輸入，不得因浮點容許界線變成成功。"""

    data = runtime_fixture
    config = data["config"].model_copy(update={
        "boundaries": data["config"].boundaries.model_copy(update={"max_backtrack_days": days}),
    })
    calls, _ = _patch_from_roots(monkeypatch, _matching_location(data))
    with pytest.raises((TypeError, ValueError)):
        _factory({**data, "config": config}, tmp_path)(_unit(data["scenario"]))
    assert calls == []
    formal = _formal_test_data(data, gap_safe=True)
    formal_config = formal["config"].model_copy(update={"boundaries": config.boundaries})
    flow = runtime.resolve_flow_domain_id(formal_config, data["scenario"].analysis_region_id, formal=True)
    axes = [("ocm_native", key, 3_600_000_000_000, ((0, 1),) if key == flow else ())
            for key in runtime._formal_flow_domain_ids(formal_config)]
    with pytest.raises((TypeError, ValueError)):
        runtime._validate_formal_ocm_gap_support(formal_config, formal["inputs"], ocm_axes=axes)


def test_horizon_earliest_int64_underflow_rejected(runtime_fixture, tmp_path) -> None:
    """合法回溯長度也不能把最早 UTC 奈秒推到檔案格式的有號 64 位整數範圍以外。"""

    factory = _factory(runtime_fixture, tmp_path)
    with pytest.raises(ValueError, match="earliest_forcing_time_utc_ns"):
        factory._build_settings(-(1 << 63))


def test_manifest_material_settling_mismatch_fails_before_manager(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """manifest material 的 settling velocity 與 exact scenario 不一致時必須拒絕。"""

    data = runtime_fixture
    mismatched_behavior = replace(data["behavior"], settling_velocity_mps=-0.3)
    data = {
        **data,
        "inputs": _scenario_inputs(
            mismatched_behavior,
            data["receptor"],
            data["arrival"],
            data["pair"],
            (data["scenario"],),
        ),
    }
    calls, _ = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path)
    with pytest.raises(ValueError, match="settling_velocity_mps"):
        factory(_unit(data["scenario"]))
    assert calls == []


@pytest.mark.parametrize("failure_kind", ["outside", "local_face", "global_face"])
def test_geometry_and_face_provenance_fail_before_provider(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """幾何或 mesh face provenance 錯誤可在 manager 後發現，但 provider 仍不得建立。"""

    data = runtime_fixture
    geometry = data["geometries"][SITE_ID]
    if failure_kind == "outside":
        outside = box(1_000.0, 1_000.0, 1_100.0, 1_100.0)
        altered_geometry = replace(
            geometry, own_local_domain=outside, flow_domain=outside
        )
        bundle = replace(data["geometries"], geometries={SITE_ID: altered_geometry})
        location = _matching_location(data)
    else:
        bundle = data["geometries"]
        location = _matching_location(data)
        if failure_kind == "local_face":
            location = replace(
                location,
                source_face_local_index=data["pair"].source_face_local_index + 1,
            )
        else:
            location = replace(
                location,
                source_face_global_index=data["pair"].source_face_global_index + 1,
            )

    _, managers = _patch_from_roots(monkeypatch, location)
    factory = _factory(data, tmp_path, geometries=bundle)
    with pytest.raises(ValueError):
        factory(_unit(data["scenario"]))
    assert len(managers) == 1
    assert managers[0].provider_calls == []


def test_finite_depth_stokes_requires_and_forwards_nww_root(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """finite-depth Stokes 只有 ordinary NWW root 可成功，且 root 原樣傳給 manager。"""

    nww_root = tmp_path / "nww-analysis"
    nww_root.mkdir()
    calls, _ = _patch_from_roots(monkeypatch, _matching_location(runtime_fixture))
    factory = _factory(runtime_fixture, tmp_path, case_id="finite_depth_stokes", nww_root=nww_root)
    factory(_unit(runtime_fixture["scenario"], case_id="finite_depth_stokes"))
    assert calls[0]["nww_root"] == nww_root


@pytest.mark.parametrize("root_kind", ["none", "missing", "symlink"])
def test_finite_depth_stokes_rejects_missing_or_symlink_nww_root(
    runtime_fixture: dict[str, Any], tmp_path: Path, root_kind: str
) -> None:
    """finite-depth Stokes 不得接受 None、不存在或符號連結的 NWW root。"""

    if root_kind == "none":
        nww_root = None
    elif root_kind == "missing":
        nww_root = tmp_path / "missing-nww"
    else:
        target = tmp_path / "nww-target"
        target.mkdir()
        nww_root = tmp_path / "nww-symlink"
        nww_root.symlink_to(target, target_is_directory=True)
    with pytest.raises((FileNotFoundError, TypeError, ValueError)):
        _factory(
            runtime_fixture,
            tmp_path,
            case_id="finite_depth_stokes",
            nww_root=nww_root,
        )


@pytest.mark.parametrize("root_kind", ["missing", "symlink"])
def test_invalid_ocm_root_fails_closed(
    runtime_fixture: dict[str, Any], tmp_path: Path, root_kind: str
) -> None:
    """OCM root 不存在或為符號連結時，factory 必須拒絕而不建立 manager。"""

    if root_kind == "missing":
        ocm_root = tmp_path / "missing-ocm"
    else:
        target = tmp_path / "ocm-target"
        target.mkdir()
        ocm_root = tmp_path / "ocm-symlink"
        ocm_root.symlink_to(target, target_is_directory=True)
    with pytest.raises((FileNotFoundError, ValueError)):
        runtime.RuntimeRequestFactory(
            config=runtime_fixture["config"],
            scenario_inputs=runtime_fixture["inputs"],
            geometries=runtime_fixture["geometries"],
            ocm_native_root=ocm_root,
            nww_analysis_root=None,
            experiment_case_id="no_stokes",
        )


@pytest.mark.parametrize("run_kind", ["production", "unknown"])
def test_unknown_runtime_run_kind_fails_closed_without_alias(
    runtime_fixture: dict[str, Any], tmp_path: Path, run_kind: str
) -> None:
    """未登錄 runtime mode 不得以 production alias 或其他模式 fallback。"""

    ocm_root = tmp_path / "ocm-native"
    ocm_root.mkdir()
    with pytest.raises(ValueError) as exc_info:
        runtime.RuntimeRequestFactory(
            config=runtime_fixture["config"],
            scenario_inputs=runtime_fixture["inputs"],
            geometries=runtime_fixture["geometries"],
            ocm_native_root=ocm_root,
            nww_analysis_root=None,
            experiment_case_id="no_stokes",
            run_kind=run_kind,
        )
    assert "pilot 或 formal" in str(exc_info.value)


def test_runtime_request_factory_formal_uses_expanded_flow_id_and_pilot_keeps_base_id(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """formal factory 使用 expanded A 區 ID，而 pilot factory 維持現有 base ID。"""

    formal_data = _formal_test_data(runtime_fixture)
    formal_calls, _ = _patch_from_roots(monkeypatch, _matching_location(formal_data))
    formal_factory = _factory(formal_data, tmp_path, run_kind="formal")
    formal_factory(_unit(formal_data["scenario"]))
    assert formal_calls[0]["flow_domain_id"] == (
        "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    )

    pilot_calls, _ = _patch_from_roots(monkeypatch, _matching_location(runtime_fixture))
    pilot_factory = _factory(runtime_fixture, tmp_path, run_kind="pilot")
    pilot_factory(_unit(runtime_fixture["scenario"]))
    assert pilot_calls[0]["flow_domain_id"] == runtime_fixture["pair"].flow_domain_id


def test_unknown_experiment_case_fails_closed(
    runtime_fixture: dict[str, Any], tmp_path: Path
) -> None:
    """未知 experiment case 不得猜測 Stokes 開關或 fallback 到 no-Stokes。"""

    ocm_root = tmp_path / "ocm-native"
    ocm_root.mkdir()
    with pytest.raises(ValueError, match="未知 experiment_case_id"):
        runtime.RuntimeRequestFactory(
            config=runtime_fixture["config"],
            scenario_inputs=runtime_fixture["inputs"],
            geometries=runtime_fixture["geometries"],
            ocm_native_root=ocm_root,
            nww_analysis_root=None,
            experiment_case_id="unknown-case",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kh", None),
        ("kh", True),
        ("kh", -1.0),
        ("kh", math.nan),
        ("kz", None),
        ("kz", True),
        ("kz", -1.0),
        ("kz", math.nan),
        ("dt_min", None),
        ("dt_min", True),
        ("dt_min", 0.0),
        ("dt_min", math.nan),
        ("dt_max", None),
        ("dt_max", True),
        ("dt_max", 0.0),
        ("dt_max", 0.5),
        ("dt_max", math.nan),
        ("output", None),
        ("output", True),
        ("output", 0.0),
        ("output", math.nan),
        ("max_days", None),
        ("max_days", True),
        ("max_days", 0.0),
        ("max_days", -1.0),
        ("max_days", math.nan),
        ("max_steps", None),
        ("max_steps", True),
        ("max_steps", 0),
        ("max_steps", 1.5),
        ("cache_months", None),
        ("cache_months", True),
        ("cache_months", 0),
        ("cache_months", 1.5),
        ("clamps", None),
        ("clamps", True),
        ("clamps", 0),
        ("clamps", 1.5),
    ],
)
def test_invalid_runtime_config_scalars_fail_closed(
    runtime_fixture: dict[str, Any], tmp_path: Path, field: str, value: Any
) -> None:
    """Kh/Kz、時間上限、步數與 cache 容量的非法 scalar 都必須被拒絕。"""

    config = runtime_fixture["config"]
    physics = deepcopy(config.physics)
    integration = config.integration
    boundaries = config.boundaries
    execution = config.execution
    if field == "kh":
        physics["horizontal_diffusion"]["constant_kh_m2ps"] = value
    elif field == "kz":
        physics["vertical_diffusion"]["constant_kz_m2ps"] = value
    elif field == "dt_min":
        integration = integration.model_copy(update={"dt_min_seconds": value})
    elif field == "dt_max":
        integration = integration.model_copy(update={"dt_max_seconds": value})
    elif field == "output":
        integration = integration.model_copy(update={"output_interval_seconds": value})
    elif field == "max_days":
        boundaries = boundaries.model_copy(update={"max_backtrack_days": value})
    elif field == "max_steps":
        boundaries = boundaries.model_copy(update={"maximum_step_count": value})
    elif field == "clamps":
        integration = integration.model_copy(update={"maximum_minimum_clamps": value})
    else:
        execution = execution.model_copy(update={"max_resident_forcing_months": value})
    invalid_config = config.model_copy(
        update={
            "physics": physics,
            "integration": integration,
            "boundaries": boundaries,
            "execution": execution,
        }
    )
    ocm_root = tmp_path / "ocm-native"
    ocm_root.mkdir()
    with pytest.raises((TypeError, ValueError)):
        runtime.RuntimeRequestFactory(
            config=invalid_config,
            scenario_inputs=runtime_fixture["inputs"],
            geometries=runtime_fixture["geometries"],
            ocm_native_root=ocm_root,
            nww_analysis_root=None,
            experiment_case_id="no_stokes",
        )


def test_resource_stats_is_exact_flat_python_int_mapping(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resource_stats 只回傳六個 flat key，且每個值都是原生 Python int。"""

    data = runtime_fixture
    _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path)
    factory(_unit(data["scenario"]))
    stats = factory.resource_stats()
    assert stats == {
        "manager_count": 1,
        "loads": 5,
        "hits": 9,
        "misses": 13,
        "evictions": 8,
        "resident_bytes": 9,
    }
    assert set(stats) == {
        "manager_count",
        "loads",
        "hits",
        "misses",
        "evictions",
        "resident_bytes",
    }
    assert all(type(value) is int for value in stats.values())


_PILOT_DECLARED_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _pilot_inventory(config_hash: str, *, mode: str = "pilot", formal_ready: bool = False) -> dict[str, Any]:
    """建立符合 Slice 3B2a 根欄位契約的 inventory 測試 mapping。

    ``inventories`` 刻意放入 server absolute path，驗證該路徑只會隨 inventory payload
    保存，而不會被 initialize orchestration 複製到 immutable run plan。
    """

    return {
        "created_at_utc": "2026-08-28T00:00:00Z",
        "config_hash": config_hash,
        "mode": mode,
        "formal_ready": formal_ready,
        "inventories": [{"source_path": "/server/input/ocm-native"}],
        "time_axes": [{"axis_id": "arrival_utc", "status": "pilot"}],
        "findings": [{"finding_id": "pilot-1", "severity": "info"}],
    }


def _write_inventory(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 JSON 寫入 inventory fixture；正常 fixture 不含非有限數值。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _pilot_provenance(*, dirty: bool = False) -> CodeProvenance:
    """建立可供 run-control 驗證的 pilot provenance fixture。"""

    return CodeProvenance(
        git_available=True,
        git_commit=_PILOT_DECLARED_COMMIT,
        git_dirty=dirty,
        commit_source="git_repository",
        deployment_tree_sha256="8" * 64,
        deployment_file_count=1,
        uv_lock_sha256="9" * 64,
        python_version="3.11.test",
        platform="test-platform",
        package_version="test-package",
        numpy_version="test-numpy",
        numba_version="test-numba",
        pyarrow_version="test-pyarrow",
    )


def _formal_config(data: dict[str, Any], *, gap_safe: bool = False) -> ProjectConfig:
    """把小型 runtime fixture 補成 formal loader 可接受的設定 snapshot。

    測試只把第一個 domain 換成 expanded formal ID，並填入正式 gate 所需的 manifest、
    diffusion、時間與部署決策；scenario／geometry loader 仍由 monkeypatch 提供一筆小
    fixture，避免建立 50,000 情境或讀取真實 forcing。
    """

    base = data["config"]
    inputs = base.inputs.model_copy(
        update={
            "ocm_gap_reconstruction_manifest": None
            if gap_safe
            else "manifests/ocm-reconstruction.json",
            "ocm_gap_safe_arrival_manifest": "manifests/ocm-gap-safe.json"
            if gap_safe
            else None,
            "nww_full_hourly_analysis_manifest": "manifests/nww-full-hourly.json",
        }
    )
    domains = list(base.domains)
    domains[0] = domains[0].model_copy(
        update={
            "formal_release_flow_domain_id": "northeast_taiwan_common_cache_v4_lbt_south_expanded",
            "formal_release_domain_status": "approved",
        }
    )
    scenarios = base.scenarios.model_copy(
        update={
            "receptor_manifest": "manifests/receptor.json",
            "arrival_time_manifest": "manifests/arrival.json",
            "material_manifest": "manifests/material.json",
            "receptor_arrival_initial_condition_manifest": "manifests/initial-condition.json",
            "members_per_scenario": 2,
            "master_seed": 20260828,
            "seed_policy": "sha256_v1_pcg64dxsm",
        }
    )
    geometry = deepcopy(base.geometry)
    geometry.update(
        {
            "domain_manifest": "manifests/domain.json",
            "local_domain_manifest": "manifests/local.json",
            "open_boundary_manifest": "manifests/open.json",
            "receptor_manifest": "manifests/receptor-geometry.json",
        }
    )
    physics = deepcopy(base.physics)
    settling = dict(physics["settling"])
    settling["material_manifest"] = "manifests/material.json"
    physics["settling"] = settling
    physics["horizontal_diffusion"]["constant_kh_m2ps"] = 2.5
    physics["vertical_diffusion"]["constant_kz_m2ps"] = 0.01
    forcing = deepcopy(base.forcing)
    forcing["ocm"]["wetdry_semantics_decision_status"] = "approved"
    integration = base.integration.model_copy(
        update={"dt_min_seconds": 1.0, "dt_max_seconds": 10.0, "output_interval_seconds": 60.0}
    )
    boundaries = base.boundaries.model_copy(
        update={"max_backtrack_days": 7.0, "maximum_step_count": 1_000}
    )
    execution = base.execution.model_copy(
        update={
            "shard_scenario_count": 1,
            "checkpoint_interval_sweeps": 1,
            "active_chunk_size": 2,
            "production_backend": "numpy_reference",
            "max_resident_forcing_months": 2,
        }
    )
    formal = base.model_copy(
        update={
            "config_status": "approved",
            "inputs": inputs,
            "domains": domains,
            "scenarios": scenarios,
            "geometry": geometry,
            "physics": physics,
            "forcing": forcing,
            "integration": integration,
            "boundaries": boundaries,
            "execution": execution,
        }
    )
    assert isinstance(formal, ProjectConfig)
    formal.assert_formal_release_ready()
    return formal


def _utc_text(value: datetime) -> str:
    """把測試用 UTC datetime 轉成 inventory 契約要求的 Z 尾碼文字。"""

    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _formal_gap_payload(
    *,
    start_utc: datetime | None = None,
    step_count: int = 1,
    boundary: str | None = None,
) -> dict[str, Any]:
    """建立與 preflight 真實格式一致的逐時連續缺口列。

    ``boundary=None`` 表示研究期內部缺口，必須同時保存缺口前後相鄰時次及兩端距離；
    ``start``／``end`` 分別模擬研究期起點／終點缺口，因研究期外沒有可驗證資料，缺少的
    外側端點與 ``gap_hours`` 必須是 ``None``。fixture 只建立時間摘要，不建立 forcing array。
    """

    start = start_utc or datetime(2025, 1, 12, tzinfo=UTC)
    end = start + timedelta(hours=step_count - 1)
    if boundary is None:
        before = start - timedelta(hours=1)
        after = end + timedelta(hours=1)
        gap_hours = (after - before).total_seconds() / 3600.0
    elif boundary == "start":
        before = None
        after = end + timedelta(hours=1)
        gap_hours = None
    elif boundary == "end":
        before = start - timedelta(hours=1)
        after = None
        gap_hours = None
    else:
        raise ValueError("boundary 只接受 None、start 或 end")
    return asdict(
        TimeGapInventory(
            missing_start_utc=_utc_text(start),
            missing_end_utc=_utc_text(end),
            missing_step_count=step_count,
            before_utc=None if before is None else _utc_text(before),
            after_utc=None if after is None else _utc_text(after),
            gap_hours=gap_hours,
        )
    )


def _formal_inventory(
    config: ProjectConfig,
    *,
    ocm_gap: bool = False,
    nww_gap: bool = False,
    ocm_gap_flow_ids: set[str] | None = None,
    gap_start_utc: datetime | None = None,
    gap_step_count: int = 1,
    gap_boundary: str | None = None,
    findings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """建立真實四域×24 月×兩產品 topology 的小型 formal inventory fixture。

    每一列只有 preflight 的 metadata 摘要，沒有網格或 forcing 陣列；因此可測試正式
    gate 的完整集合、schema/status/cache contract、時間軸與缺口語意，而不建立大量情境。
    月份列的 time_count 依該月逐時計算，跨月軸固定使用 2024–2025 的 17,544 個小時。
    """

    products = ("ocm_native", "nww3_analysis")
    inventories: list[dict[str, Any]] = []
    for domain in config.domains:
        flow_domain_id = domain.resolved_flow_domain_id(formal=True)
        for year in sorted(config.inputs.years):
            for month in range(1, 13):
                month_start = datetime(year, month, 1, tzinfo=UTC)
                next_month = (
                    datetime(year + 1, 1, 1, tzinfo=UTC)
                    if month == 12
                    else datetime(year, month + 1, 1, tzinfo=UTC)
                )
                month_end = next_month - timedelta(hours=1)
                month_time_count = int((month_end - month_start).total_seconds() / 3600) + 1
                for product in products:
                    contract = (
                        config.inputs.ocm_contract
                        if product == "ocm_native"
                        else config.inputs.nww_contract
                    )
                    root_env = (
                        config.inputs.ocm_native_root_env
                        if product == "ocm_native"
                        else config.inputs.nww_analysis_root_env
                    )
                    inventories.append(
                        asdict(
                            MonthInventory(
                                product=product,
                                flow_domain_id=flow_domain_id,
                                month=f"{year}{month:02d}",
                                status=contract["accepted_statuses"][0],
                                schema_version=f"{contract['required_schema_major']}.0",
                                cache_kind=contract["accepted_cache_kinds"][0],
                                time_count=month_time_count,
                                time_start_utc=_utc_text(month_start),
                                time_end_utc=_utc_text(month_end),
                                maximum_gap_seconds=0.0,
                                required_arrays_present=True,
                                source_missing_day_count=0 if product == "ocm_native" else None,
                                source_timestamp_repair_file_count=0
                                if product == "ocm_native"
                                else None,
                                source_zero_kept_file_count=0 if product == "ocm_native" else None,
                                source_skipped_overlap_time_step_count=0
                                if product == "ocm_native"
                                else None,
                                path_token=f"${root_env}/{flow_domain_id}/months/{year}{month:02d}",
                            )
                        )
                    )

    expected_period_time_count = 17_544
    period_start = datetime(2024, 1, 1, tzinfo=UTC)
    period_end = datetime(2025, 12, 31, 23, tzinfo=UTC)
    time_axes: list[dict[str, Any]] = []
    for domain in config.domains:
        flow_domain_id = domain.resolved_flow_domain_id(formal=True)
        for product in products:
            if product == "ocm_native":
                has_gap = ocm_gap and (
                    ocm_gap_flow_ids is None or flow_domain_id in ocm_gap_flow_ids
                )
            else:
                has_gap = nww_gap
            gaps = (
                [
                    _formal_gap_payload(
                        start_utc=gap_start_utc,
                        step_count=gap_step_count,
                        boundary=gap_boundary,
                    )
                ]
                if has_gap
                else []
            )
            missing_count = sum(gap["missing_step_count"] for gap in gaps)
            available_count = expected_period_time_count - missing_count
            time_axes.append(
                asdict(
                    TimeAxisInventory(
                        product=product,
                        flow_domain_id=flow_domain_id,
                        policy=config.inputs.time_axis_contract["canonicalization_policy"],
                        expected_timestep_hours=1.0,
                        input_time_count=available_count,
                        canonical_time_count=available_count,
                        reordered_time_step_count=0,
                        dropped_duplicate_time_step_count=0,
                        expected_period_time_count=expected_period_time_count,
                        available_period_time_count=available_count,
                        missing_period_time_count=missing_count,
                        extra_halo_time_count=0,
                        coverage_fraction=available_count / expected_period_time_count,
                        time_start_utc=_utc_text(period_start),
                        time_end_utc=_utc_text(period_end),
                        maximum_internal_gap_hours=(
                            float(gaps[0]["gap_hours"] or 1.0) if has_gap else 0.0
                        ),
                        gaps=gaps,
                    )
                )
            )
    return {
        "created_at_utc": "2026-08-28T00:00:00Z",
        "config_hash": config.config_hash(),
        "mode": "formal_release",
        "formal_ready": True,
        "inventories": inventories,
        "time_axes": time_axes,
        "findings": list(findings or []),
    }


def _formal_test_data(
    data: dict[str, Any], *, gap_safe: bool = False, arrival_time_ns: int | None = None
) -> dict[str, Any]:
    """將單一 runtime fixture 改成 formal expanded flow-domain 與可選 arrival。"""

    formal_config = _formal_config(data, gap_safe=gap_safe)
    arrival = data["arrival"]
    pair = data["pair"]
    scenario = data["scenario"]
    if arrival_time_ns is not None:
        arrival_datetime = datetime.fromtimestamp(arrival_time_ns / 1_000_000_000, tz=UTC)
        arrival = replace(
            arrival,
            time_utc_ns=arrival_time_ns,
            year=arrival_datetime.year,
        )
        pair = replace(pair, time_utc_ns=arrival_time_ns)
        scenario = replace(scenario, arrival_time_utc_ns=arrival_time_ns)
    formal_pair = replace(
        pair,
        flow_domain_id="northeast_taiwan_common_cache_v4_lbt_south_expanded",
    )
    formal_inputs = replace(
        data["inputs"],
        arrival_times=(arrival,),
        scenarios=(scenario,),
        initial_conditions=(formal_pair,),
        initial_conditions_by_pair={
            (formal_pair.receptor_id, formal_pair.arrival_time_id): formal_pair
        },
    )
    return {
        **data,
        "config": formal_config,
        "arrival": arrival,
        "scenario": scenario,
        "inputs": formal_inputs,
        "pair": formal_pair,
    }


def _patch_formal_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, Any],
    *,
    provenance: CodeProvenance | None = None,
) -> dict[str, Any]:
    """替換 formal initializer 的 loader，記錄 formal 旗標並保留小型 fixture。"""

    calls: dict[str, Any] = {}
    selected_provenance = provenance or _pilot_provenance()

    def fake_load_config(path: str | Path, *, formal_release: bool) -> ProjectConfig:
        """回傳已通過 formal config gate 的 fixture config。"""

        calls["config"] = {"path": path, "formal_release": formal_release}
        return data["config"]

    def fake_load_scenario_inputs(config: ProjectConfig, **kwargs: Any) -> ScenarioInputs:
        """回傳小型 dynamic initial-condition fixture，並保存 loader 參數。"""

        calls["scenario"] = {"config": config, "kwargs": kwargs}
        return data["inputs"]

    def fake_load_boundary_geometries(config: ProjectConfig, **kwargs: Any) -> BoundaryGeometryBundle:
        """回傳小型幾何 fixture，並保存 formal loader 參數。"""

        calls["geometry"] = {"config": config, "kwargs": kwargs}
        return data["geometries"]

    def fake_collect_code_provenance(
        project_root: str | Path,
        *,
        declared_git_commit: str | None,
        formal: bool,
    ) -> CodeProvenance:
        """記錄正式 provenance gate 的 formal=True 呼叫。"""

        calls["provenance"] = {
            "project_root": project_root,
            "declared_git_commit": declared_git_commit,
            "formal": formal,
        }
        return selected_provenance

    monkeypatch.setattr(runtime, "load_config", fake_load_config)
    monkeypatch.setattr(runtime, "load_scenario_inputs", fake_load_scenario_inputs)
    monkeypatch.setattr(runtime, "load_boundary_geometries", fake_load_boundary_geometries)
    monkeypatch.setattr(runtime, "collect_code_provenance", fake_collect_code_provenance)
    return calls


def _patch_pilot_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, Any],
    *,
    provenance: CodeProvenance | None = None,
) -> dict[str, Any]:
    """替換 initialize 依賴，記錄 loader／provenance 參數而不重測 manifest loader。"""

    calls: dict[str, Any] = {}
    selected_provenance = provenance or _pilot_provenance()

    def fake_load_config(path: str | Path, *, formal_release: bool) -> ProjectConfig:
        """回傳已在 fixture 準備好的 config，避免重新載入 example 的 null scalar。"""

        calls["config"] = {"path": path, "formal_release": formal_release}
        return data["config"]

    def fake_load_scenario_inputs(config: ProjectConfig, **kwargs: Any) -> ScenarioInputs:
        """回傳含 dynamic initial-condition hash 的 fixture inputs。"""

        calls["scenario"] = {"config": config, "kwargs": kwargs}
        return data["inputs"]

    def fake_load_boundary_geometries(config: ProjectConfig, **kwargs: Any) -> BoundaryGeometryBundle:
        """回傳含 domain/local/open-boundary hash 的 fixture geometry bundle。"""

        calls["geometry"] = {"config": config, "kwargs": kwargs}
        return data["geometries"]

    def fake_collect_code_provenance(
        project_root: str | Path,
        *,
        declared_git_commit: str | None,
        formal: bool,
    ) -> CodeProvenance:
        """記錄 provenance 呼叫，讓測試確認 pilot 不觸發 formal gate。"""

        calls["provenance"] = {
            "project_root": project_root,
            "declared_git_commit": declared_git_commit,
            "formal": formal,
        }
        return selected_provenance

    monkeypatch.setattr(runtime, "load_config", fake_load_config)
    monkeypatch.setattr(runtime, "load_scenario_inputs", fake_load_scenario_inputs)
    monkeypatch.setattr(runtime, "load_boundary_geometries", fake_load_boundary_geometries)
    monkeypatch.setattr(runtime, "collect_code_provenance", fake_collect_code_provenance)
    return calls


def _assert_no_published_workspace(destination: Path, run_id: str) -> None:
    """確認 initialize 失敗後沒有目標 workspace 或遺留 partial 目錄。"""

    assert not (destination / run_id).exists()
    if destination.exists():
        assert not any(item.name.startswith(f".{run_id}.partial-") for item in destination.iterdir())


def _initialize_pilot_test_run(
    data: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any] | None = None,
    run_id: str = "pilot-initialize",
    provenance: CodeProvenance | None = None,
) -> tuple[RunWorkspace, dict[str, Any]]:
    """以 real inventory file 呼叫 public initializer，供少量測試共用固定 setup。"""

    inventory_path = tmp_path / f"{run_id}-inventory.json"
    _write_inventory(inventory_path, payload or _pilot_inventory(data["config"].config_hash()))
    calls = _patch_pilot_orchestration(monkeypatch, data, provenance=provenance)
    workspace = runtime.initialize_pilot_run(
        config_path=EXAMPLE_CONFIG,
        input_inventory_path=inventory_path,
        destination=tmp_path / "runs",
        run_id=run_id,
        experiment_case_id="no_stokes",
        project_root=tmp_path / "project-root",
        declared_git_commit=_PILOT_DECLARED_COMMIT,
    )
    return workspace, calls


def _initialize_formal_test_run(
    data: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any] | None = None,
    run_id: str = "formal-initialize",
) -> tuple[RunWorkspace, dict[str, Any]]:
    """以完整 formal topology fixture 發布 workspace，不接觸實際 forcing。"""

    inventory_path = tmp_path / f"{run_id}-inventory.json"
    _write_inventory(inventory_path, payload or _formal_inventory(data["config"]))
    calls = _patch_formal_orchestration(monkeypatch, data)
    workspace = runtime.initialize_formal_run(
        config_path=EXAMPLE_CONFIG,
        input_inventory_path=inventory_path,
        destination=tmp_path / "runs",
        run_id=run_id,
        experiment_case_id="no_stokes",
        project_root=tmp_path / "project-root",
        declared_git_commit=_PILOT_DECLARED_COMMIT,
    )
    return workspace, calls


def test_initialize_formal_run_orchestrates_formal_loaders_and_full_inventory(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """formal initializer 應使用正式 loader、完整 topology 並發布 formal plan。"""

    data = _formal_test_data(runtime_fixture)
    workspace, calls = _initialize_formal_test_run(data, tmp_path, monkeypatch)

    assert calls["config"] == {"path": EXAMPLE_CONFIG, "formal_release": True}
    assert calls["scenario"]["config"] is data["config"]
    assert calls["scenario"]["kwargs"] == {
        "config_path": EXAMPLE_CONFIG,
        "require_dynamic_initial_conditions": True,
        "formal": True,
    }
    assert calls["geometry"] == {
        "config": data["config"],
        "kwargs": {"config_path": EXAMPLE_CONFIG, "formal": True},
    }
    assert calls["provenance"] == {
        "project_root": tmp_path / "project-root",
        "declared_git_commit": _PILOT_DECLARED_COMMIT,
        "formal": True,
    }
    assert workspace.plan["run_kind"] == "formal"
    stored_inventory = json.loads(
        (workspace.path / "input_inventory.json").read_text(encoding="utf-8")
    )
    assert len(stored_inventory["inventories"]) == 4 * 24 * 2
    assert len(stored_inventory["time_axes"]) == 4 * 2
    assert validate_run(workspace)["valid"]


@pytest.mark.parametrize(
    "years",
    ([2024], [2025, 2024], [2024, 2025, 2026], [2024, 2026]),
)
def test_initialize_formal_run_requires_exact_ordered_research_years(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    years: list[int],
) -> None:
    """正式 initializer 必須拒絕缺年、重排、額外年份及跨年洞。"""

    data = _formal_test_data(runtime_fixture)
    invalid_inputs = data["config"].inputs.model_copy(update={"years": years})
    invalid_config = data["config"].model_copy(update={"inputs": invalid_inputs})
    invalid_data = {**data, "config": invalid_config}
    payload = _formal_inventory(data["config"])
    # inventory 的 hash 必須先綁定被測設定，讓測試確實走到 formal years gate，
    # 而不是因為較早的 hash mismatch 失敗。
    payload["config_hash"] = invalid_config.config_hash()
    inventory_path = tmp_path / "formal-invalid-years.json"
    _write_inventory(inventory_path, payload)
    calls = _patch_formal_orchestration(monkeypatch, invalid_data)

    with pytest.raises(ValueError, match="config.inputs.years"):
        runtime.initialize_formal_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=tmp_path / "runs",
            run_id="formal-invalid-years",
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )

    assert "provenance" not in calls
    _assert_no_published_workspace(tmp_path / "runs", "formal-invalid-years")


@pytest.mark.parametrize(
    ("product", "wrong_component"),
    [
        ("ocm_native", "root"),
        ("ocm_native", "flow"),
        ("ocm_native", "month"),
        ("nww3_analysis", "root"),
        ("nww3_analysis", "flow"),
        ("nww3_analysis", "month"),
    ],
)
def test_initialize_formal_run_rejects_non_exact_month_path_token(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    product: str,
    wrong_component: str,
) -> None:
    """OCM/NWW3 月份列的 root、flow 或 month token 任一錯誤都必須 fail-closed。"""

    data = _formal_test_data(runtime_fixture)
    payload = _formal_inventory(data["config"])
    target = next(row for row in payload["inventories"] if row["product"] == product)
    month = target["month"]
    flow_id = target["flow_domain_id"]
    root_env = (
        data["config"].inputs.ocm_native_root_env
        if product == "ocm_native"
        else data["config"].inputs.nww_analysis_root_env
    )
    if wrong_component == "root":
        target["path_token"] = f"$WRONG_ROOT/{flow_id}/months/{month}"
    elif wrong_component == "flow":
        target["path_token"] = f"${root_env}/wrong-flow/months/{month}"
    else:
        target["path_token"] = f"${root_env}/{flow_id}/months/199901"
    inventory_path = tmp_path / "formal-invalid-path-token.json"
    _write_inventory(inventory_path, payload)
    calls = _patch_formal_orchestration(monkeypatch, data)
    run_id = f"formal-invalid-path-{product}-{wrong_component}"

    with pytest.raises(ValueError, match="path_token"):
        runtime.initialize_formal_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=tmp_path / "runs",
            run_id=run_id,
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )

    assert "provenance" not in calls
    _assert_no_published_workspace(tmp_path / "runs", run_id)


@pytest.mark.parametrize(
    "failure_kind",
    [
        "mode",
        "formal_ready",
        "error_finding",
        "inventory_duplicate",
        "inventory_missing",
        "bad_schema",
        "bad_status",
        "bad_arrays",
        "axis_duplicate",
        "axis_missing",
        "expected_period_count",
        "missing_gap_mismatch",
        "nww_gap",
    ],
)
def test_initialize_formal_run_rejects_invalid_inventory_before_publish(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """formal mode、產品 topology、欄位契約與 NWW 完整性均需 fail-closed。"""

    data = _formal_test_data(runtime_fixture)
    payload = _formal_inventory(
        data["config"],
        ocm_gap=failure_kind == "missing_gap_mismatch",
        nww_gap=failure_kind == "nww_gap",
    )
    if failure_kind == "mode":
        payload["mode"] = "pilot"
    elif failure_kind == "formal_ready":
        payload["formal_ready"] = False
    elif failure_kind == "error_finding":
        payload["findings"] = [
            asdict(
                Finding(
                    severity="error",
                    code="test_error",
                    location="fixture",
                    message="測試用正式輸入錯誤",
                )
            )
        ]
    elif failure_kind == "inventory_duplicate":
        payload["inventories"][-1] = deepcopy(payload["inventories"][0])
    elif failure_kind == "inventory_missing":
        payload["inventories"].pop()
    elif failure_kind == "bad_schema":
        payload["inventories"][0]["schema_version"] = "2.0"
    elif failure_kind == "bad_status":
        payload["inventories"][0]["status"] = "trial_ready"
    elif failure_kind == "bad_arrays":
        payload["inventories"][0]["required_arrays_present"] = False
    elif failure_kind == "axis_duplicate":
        payload["time_axes"][-1] = deepcopy(payload["time_axes"][0])
    elif failure_kind == "axis_missing":
        payload["time_axes"].pop()
    elif failure_kind == "expected_period_count":
        payload["time_axes"][0]["expected_period_time_count"] -= 1
    elif failure_kind == "missing_gap_mismatch":
        axis = payload["time_axes"][0]
        axis["missing_period_time_count"] = 2
        axis["available_period_time_count"] = 17_542
        axis["canonical_time_count"] = 17_542
        axis["input_time_count"] = 17_542
        axis["coverage_fraction"] = 17_542 / 17_544

    destination = tmp_path / "runs"
    run_id = f"formal-invalid-{failure_kind}"
    inventory_path = tmp_path / "formal-inventory.json"
    _write_inventory(inventory_path, payload)
    calls = _patch_formal_orchestration(monkeypatch, data)
    with pytest.raises((TypeError, ValueError)):
        runtime.initialize_formal_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=destination,
            run_id=run_id,
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )
    assert "provenance" not in calls
    _assert_no_published_workspace(destination, run_id)


def test_initialize_formal_run_accepts_ocm_full_coverage(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重建 manifest 已宣告且所有 OCM axis missing=0 時，完整產品可直接通過。"""

    data = _formal_test_data(runtime_fixture)
    workspace, _ = _initialize_formal_test_run(data, tmp_path, monkeypatch, run_id="formal-ocm-full")
    assert workspace.plan["run_kind"] == "formal"


def test_initialize_formal_run_checks_gap_only_for_arrival_flow(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B flow 殘留缺口不得阻擋只屬 A flow 且回溯窗有完整支援的 arrival。"""

    data = _formal_test_data(runtime_fixture, gap_safe=True)
    b_flow_id = runtime.resolve_flow_domain_id(data["config"], "B", formal=True)
    payload = _formal_inventory(
        data["config"],
        ocm_gap=True,
        ocm_gap_flow_ids={b_flow_id},
        gap_start_utc=datetime(2025, 1, 12, tzinfo=UTC),
    )
    workspace, _ = _initialize_formal_test_run(
        data,
        tmp_path,
        monkeypatch,
        payload=payload,
        run_id="formal-other-flow-gap",
    )
    assert workspace.plan["run_kind"] == "formal"


def test_initialize_formal_run_accepts_gap_safe_arrival_window(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OCM residual gap 避開 arrival 回溯窗時，gap-safe manifest 可使 gate 通過。"""

    data = _formal_test_data(runtime_fixture, gap_safe=True)
    payload = _formal_inventory(
        data["config"],
        ocm_gap=True,
        gap_start_utc=datetime(2025, 1, 20, tzinfo=UTC),
    )
    workspace, _ = _initialize_formal_test_run(
        data,
        tmp_path,
        monkeypatch,
        payload=payload,
        run_id="formal-ocm-gap-safe",
    )
    assert workspace.plan["run_kind"] == "formal"


def test_initialize_formal_run_accepts_safe_start_boundary_gap_shape(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight 的研究期起點缺口格式在遠離 arrival 回溯窗時應可通過。"""

    data = _formal_test_data(runtime_fixture, gap_safe=True)
    a_flow_id = runtime.resolve_flow_domain_id(data["config"], "A", formal=True)
    payload = _formal_inventory(
        data["config"],
        ocm_gap=True,
        ocm_gap_flow_ids={a_flow_id},
        gap_start_utc=datetime(2024, 1, 1, tzinfo=UTC),
        gap_boundary="start",
    )
    workspace, _ = _initialize_formal_test_run(
        data,
        tmp_path,
        monkeypatch,
        payload=payload,
        run_id="formal-start-boundary-gap",
    )
    assert workspace.plan["run_kind"] == "formal"


def test_initialize_formal_run_rejects_invalid_start_boundary_endpoint(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """研究期起點缺口的 after 不是緊鄰時次時，正式時間證據必須拒絕。"""

    data = _formal_test_data(runtime_fixture, gap_safe=True)
    a_flow_id = runtime.resolve_flow_domain_id(data["config"], "A", formal=True)
    payload = _formal_inventory(
        data["config"],
        ocm_gap=True,
        ocm_gap_flow_ids={a_flow_id},
        gap_start_utc=datetime(2024, 1, 1, tzinfo=UTC),
        gap_boundary="start",
    )
    a_axis = next(
        axis
        for axis in payload["time_axes"]
        if axis["product"] == "ocm_native" and axis["flow_domain_id"] == a_flow_id
    )
    a_axis["gaps"][0]["after_utc"] = "2024-01-01T02:00:00Z"
    destination = tmp_path / "runs"
    run_id = "formal-invalid-start-boundary"
    inventory_path = tmp_path / "formal-invalid-start-boundary.json"
    _write_inventory(inventory_path, payload)
    calls = _patch_formal_orchestration(monkeypatch, data)
    with pytest.raises(ValueError, match="起點 boundary gap"):
        runtime.initialize_formal_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=destination,
            run_id=run_id,
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )
    assert "provenance" not in calls
    _assert_no_published_workspace(destination, run_id)


@pytest.mark.parametrize("case", ["reconstruction_only", "arrival_intersection", "horizon_outside"])
def test_initialize_formal_run_rejects_unsupported_ocm_gap_safe_cases(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """重建-only、arrival 相交與研究期外回溯窗均不得通過 OCM gap gate。"""

    data = _formal_test_data(
        runtime_fixture,
        gap_safe=case != "reconstruction_only",
        arrival_time_ns=(
            int(datetime(2024, 1, 3, tzinfo=UTC).timestamp()) * 1_000_000_000
            if case == "horizon_outside"
            else None
        ),
    )
    payload = _formal_inventory(
        data["config"],
        ocm_gap=True,
        ocm_gap_flow_ids=(
            {runtime.resolve_flow_domain_id(data["config"], "A", formal=True)}
            if case == "arrival_intersection"
            else None
        ),
        gap_start_utc=(
            datetime(2025, 1, 12, tzinfo=UTC)
            if case == "arrival_intersection"
            else datetime(2025, 1, 20, tzinfo=UTC)
        ),
    )
    destination = tmp_path / "runs"
    run_id = f"formal-gap-invalid-{case}"
    calls = _patch_formal_orchestration(monkeypatch, data)
    inventory_path = tmp_path / "formal-gap-inventory.json"
    _write_inventory(inventory_path, payload)
    with pytest.raises(ValueError):
        runtime.initialize_formal_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=destination,
            run_id=run_id,
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )
    assert "provenance" not in calls
    _assert_no_published_workspace(destination, run_id)


def test_open_generic_and_formal_controller_revalidate_formal_inventory_read_only(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generic/formal wrapper 應重驗 formal inventory、轉送 checkpoint 並保持 lazy。"""

    data = _formal_test_data(runtime_fixture)
    workspace, _ = _initialize_formal_test_run(data, tmp_path, monkeypatch, run_id="formal-open")
    ocm_root = tmp_path / "ocm-native"
    ocm_root.mkdir()
    checkpoint_root = tmp_path / "external-checkpoints"
    checkpoint_root.mkdir()
    progress_path = workspace.path / "run_progress.json"
    before_progress = progress_path.read_bytes()
    real_gate = runtime._validated_formal_inventory
    gate_calls: list[Path] = []

    def recording_gate(
        input_inventory_path: str | Path,
        *,
        expected_config_hash: str,
        config: ProjectConfig,
        scenario_inputs: ScenarioInputs,
    ) -> dict[str, Any]:
        """記錄 open 期間的第二次 formal inventory semantic gate。"""

        gate_calls.append(Path(input_inventory_path))
        return real_gate(
            input_inventory_path,
            expected_config_hash=expected_config_hash,
            config=config,
            scenario_inputs=scenario_inputs,
        )

    def reject_forcing_manager(**_: Any) -> None:
        """formal controller 建立時不得提前建立 forcing manager。"""

        raise AssertionError("open formal controller 不得提前觸發 forcing")

    monkeypatch.setattr(runtime, "_validated_formal_inventory", recording_gate)
    monkeypatch.setattr(
        runtime.ForcingWindowManager,
        "from_roots",
        staticmethod(reject_forcing_manager),
    )
    controller = runtime.open_run_controller(
        workspace,
        config_path=EXAMPLE_CONFIG,
        ocm_native_root=ocm_root,
        resume=True,
        checkpoint_root=checkpoint_root,
    )

    assert isinstance(controller, RunController)
    assert controller.resume is True
    assert controller.checkpoint_root == checkpoint_root
    assert gate_calls == [workspace.path / "input_inventory.json"]
    assert controller.request_factory.resource_stats()["manager_count"] == 0
    assert progress_path.read_bytes() == before_progress

    with pytest.raises(ValueError, match="run_kind 必須 exact 等於 pilot"):
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=ocm_root,
        )
    controller_from_formal_wrapper = runtime.open_formal_run_controller(
        workspace,
        config_path=EXAMPLE_CONFIG,
        ocm_native_root=ocm_root,
    )
    assert isinstance(controller_from_formal_wrapper, RunController)
    assert progress_path.read_bytes() == before_progress


def test_initialize_pilot_run_orchestrates_strict_inputs_and_publishes_valid_workspace(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """happy path 必須只用固定 loader kwargs、mapping inventory 與 pilot plan schema。"""

    data = runtime_fixture
    workspace, calls = _initialize_pilot_test_run(data, tmp_path, monkeypatch)

    assert isinstance(workspace, RunWorkspace)
    assert calls["config"] == {"path": EXAMPLE_CONFIG, "formal_release": False}
    assert calls["scenario"]["config"] is data["config"]
    assert calls["scenario"]["kwargs"] == {
        "config_path": EXAMPLE_CONFIG,
        "require_dynamic_initial_conditions": True,
        "formal": False,
    }
    assert calls["geometry"] == {
        "config": data["config"],
        "kwargs": {"config_path": EXAMPLE_CONFIG, "formal": False},
    }
    assert calls["provenance"] == {
        "project_root": tmp_path / "project-root",
        "declared_git_commit": _PILOT_DECLARED_COMMIT,
        "formal": False,
    }

    plan = workspace.plan
    assert plan["run_kind"] == "pilot"
    assert plan["experiment_case_id"] == "no_stokes"
    assert plan["scenario_count"] == 1
    assert plan["particle_count"] == 2
    assert plan["master_seed"] == 20260828
    assert plan["seed_policy"] == "sha256_v1_pcg64dxsm"
    assert plan["members_per_scenario"] == 2
    assert plan["shard_scenario_count"] == 1
    assert plan["checkpoint_interval_sweeps"] == 1
    assert plan["active_chunk_size"] == 2
    assert plan["component_canonical_hashes"] == {
        "material": "1" * 64,
        "receptor": "2" * 64,
        "arrival": "3" * 64,
        "receptor_arrival_initial_condition": "4" * 64,
    }
    assert plan["geometry_canonical_hashes"] == {
        "domain": "5" * 64,
        "local": "6" * 64,
        "open_boundary": "7" * 64,
    }
    assert "downgrade" not in plan

    stored_inventory = json.loads(
        (workspace.path / "input_inventory.json").read_text(encoding="utf-8")
    )
    expected_inventory = _pilot_inventory(data["config"].config_hash())
    assert stored_inventory == expected_inventory
    assert "/server/input/ocm-native" not in json.dumps(dict(plan), ensure_ascii=False)
    assert validate_run(workspace)["valid"]


@pytest.mark.parametrize(
    ("mode", "formal_ready"),
    [("pilot", False), ("pilot", True), ("formal", False), ("formal", True)],
)
def test_initialize_pilot_run_allows_inventory_mode_and_formal_ready_values(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    formal_ready: bool,
) -> None:
    """pilot 不把 inventory mode 或 formal_ready 當成正式發布 gate。"""

    run_id = f"pilot-inventory-{mode}-{str(formal_ready).lower()}"
    payload = _pilot_inventory(
        runtime_fixture["config"].config_hash(),
        mode=mode,
        formal_ready=formal_ready,
    )
    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture,
        tmp_path,
        monkeypatch,
        payload=payload,
        run_id=run_id,
        provenance=_pilot_provenance(dirty=True),
    )
    assert workspace.plan["run_kind"] == "pilot"


@pytest.mark.parametrize(
    "failure_kind",
    ["extra", "missing", "root_type", "list_item", "bad_utc", "bad_config_hash", "nonfinite", "symlink"],
)
def test_initialize_pilot_run_rejects_strict_inventory_and_leaves_no_workspace(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """inventory 根欄位、型別、UTC、hash、有限數值與 symlink 錯誤都必須 fail-closed。"""

    data = runtime_fixture
    run_id = f"pilot-invalid-inventory-{failure_kind}"
    destination = tmp_path / "runs"
    payload = _pilot_inventory(data["config"].config_hash())
    inventory_path = tmp_path / f"{run_id}.json"
    if failure_kind == "extra":
        payload["unexpected"] = True
        _write_inventory(inventory_path, payload)
    elif failure_kind == "missing":
        del payload["findings"]
        _write_inventory(inventory_path, payload)
    elif failure_kind == "root_type":
        inventory_path.write_text("[]\n", encoding="utf-8")
    elif failure_kind == "list_item":
        payload["inventories"] = ["not-an-object"]
        _write_inventory(inventory_path, payload)
    elif failure_kind == "bad_utc":
        payload["created_at_utc"] = "2026-08-28T08:00:00+08:00"
        _write_inventory(inventory_path, payload)
    elif failure_kind == "bad_config_hash":
        payload["config_hash"] = "f" * 64
        _write_inventory(inventory_path, payload)
    elif failure_kind == "nonfinite":
        payload["findings"] = []
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        inventory_path.write_text(
            raw.replace('"findings": []', '"findings": [{"value": NaN}]'),
            encoding="utf-8",
        )
    else:
        target = tmp_path / "valid-inventory-target.json"
        _write_inventory(target, payload)
        inventory_path.symlink_to(target)

    calls = _patch_pilot_orchestration(monkeypatch, data)
    with pytest.raises((TypeError, ValueError)):
        runtime.initialize_pilot_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=destination,
            run_id=run_id,
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )
    assert "provenance" not in calls
    _assert_no_published_workspace(destination, run_id)


@pytest.mark.parametrize("failure_kind", ["missing_dynamic", "missing_geometry", "extra_dynamic", "bad_hash"])
def test_initialize_pilot_run_rejects_nonexact_canonical_hashes_before_publish(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """dynamic 與三份 geometry hash 必須各自 exact 且每值為小寫 SHA-256。"""

    data = runtime_fixture
    if failure_kind == "missing_dynamic":
        component_hashes = dict(data["inputs"].canonical_component_hashes)
        del component_hashes["receptor_arrival_initial_condition"]
        inputs = replace(data["inputs"], canonical_component_hashes=component_hashes)
        geometries = data["geometries"]
    elif failure_kind == "missing_geometry":
        geometry_hashes = dict(data["geometries"].canonical_component_hashes)
        del geometry_hashes["open_boundary"]
        inputs = data["inputs"]
        geometries = replace(data["geometries"], canonical_component_hashes=geometry_hashes)
    elif failure_kind == "extra_dynamic":
        component_hashes = dict(data["inputs"].canonical_component_hashes)
        component_hashes["unregistered"] = "8" * 64
        inputs = replace(data["inputs"], canonical_component_hashes=component_hashes)
        geometries = data["geometries"]
    else:
        component_hashes = dict(data["inputs"].canonical_component_hashes)
        component_hashes["arrival"] = "G" * 64
        inputs = replace(data["inputs"], canonical_component_hashes=component_hashes)
        geometries = data["geometries"]

    patched_data = {**data, "inputs": inputs, "geometries": geometries}
    inventory_path = tmp_path / f"{failure_kind}.json"
    _write_inventory(inventory_path, _pilot_inventory(data["config"].config_hash()))
    calls = _patch_pilot_orchestration(monkeypatch, patched_data)
    destination = tmp_path / "runs"
    run_id = f"pilot-invalid-hashes-{failure_kind}"
    with pytest.raises(ValueError):
        runtime.initialize_pilot_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=destination,
            run_id=run_id,
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )
    assert "provenance" not in calls
    _assert_no_published_workspace(destination, run_id)


def _config_with_pilot_gate_value(
    config: ProjectConfig, *, section: str, field: str, value: Any
) -> ProjectConfig:
    """只替換一個 pilot gate 欄位，保留其餘已配置 fixture 值。"""

    if section == "scenarios":
        scenarios = config.scenarios.model_copy(update={field: value})
        return config.model_copy(update={"scenarios": scenarios})
    execution = config.execution.model_copy(update={field: value})
    return config.model_copy(update={"execution": execution})


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("scenarios", "members_per_scenario", None),
        ("scenarios", "members_per_scenario", True),
        ("scenarios", "master_seed", -1),
        ("scenarios", "master_seed", True),
        ("execution", "shard_scenario_count", 0),
        ("execution", "shard_scenario_count", True),
        ("execution", "checkpoint_interval_sweeps", 0),
        ("execution", "checkpoint_interval_sweeps", True),
        ("scenarios", "seed_policy", "sha256_v1_other"),
        ("execution", "production_backend", "numba"),
        ("execution", "active_chunk_size", 0),
        ("execution", "active_chunk_size", True),
        ("execution", "max_resident_forcing_months", None),
        ("execution", "max_resident_forcing_months", True),
    ],
)
def test_initialize_pilot_run_rejects_invalid_execution_gate_without_publish(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: Any,
) -> None:
    """seed／分片／checkpoint／backend／cache gate 的非法值不可建立 workspace。"""

    invalid_config = _config_with_pilot_gate_value(
        runtime_fixture["config"], section=section, field=field, value=value
    )
    data = {**runtime_fixture, "config": invalid_config}
    inventory_path = tmp_path / f"invalid-{section}-{field}-{str(value).replace('/', '_')}.json"
    _write_inventory(inventory_path, _pilot_inventory(invalid_config.config_hash()))
    calls = _patch_pilot_orchestration(monkeypatch, data)
    destination = tmp_path / "runs"
    run_id = f"pilot-invalid-config-{section}-{field}-{len(str(value))}"
    with pytest.raises((TypeError, ValueError)):
        runtime.initialize_pilot_run(
            config_path=EXAMPLE_CONFIG,
            input_inventory_path=inventory_path,
            destination=destination,
            run_id=run_id,
            experiment_case_id="no_stokes",
            project_root=tmp_path / "project-root",
            declared_git_commit=_PILOT_DECLARED_COMMIT,
        )
    assert "provenance" not in calls
    _assert_no_published_workspace(destination, run_id)


def test_initialize_pilot_run_accepts_none_active_chunk_size(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """active_chunk_size=None 是允許的 pilot 設定，且不擴張 run plan 欄位。"""

    config = _config_with_pilot_gate_value(
        runtime_fixture["config"], section="execution", field="active_chunk_size", value=None
    )
    data = {**runtime_fixture, "config": config}
    workspace, _ = _initialize_pilot_test_run(data, tmp_path, monkeypatch, run_id="pilot-active-none")
    assert workspace.plan["active_chunk_size"] is None


def _plan_with_binding_mismatch(plan: dict[str, Any], field: str) -> dict[str, Any]:
    """複製磁碟 plan 並只改一個 controller binding 欄位。

    測試故意只在 ``load_run_plan`` 的回傳 mapping 中注入失配，不改寫 workspace 磁碟；
    這樣 ``validate_run`` 仍會看到真實且完整的 immutable plan，測到的是 public open
    API 自己的 binding gate，而不是 validator 先替測試攔截。每個欄位都保留合法型別，
    讓失敗原因精確落在對應的 exact equality，而不混入 plan schema 錯誤。
    """

    mutated = deepcopy(plan)
    if field == "config_hash":
        mutated[field] = "f" * 64
    elif field == "component_canonical_hashes":
        mutated[field]["material"] = "f" * 64
    elif field == "geometry_canonical_hashes":
        mutated[field]["domain"] = "f" * 64
    elif field in {
        "scenario_count",
        "master_seed",
        "members_per_scenario",
        "shard_scenario_count",
        "checkpoint_interval_sweeps",
    }:
        mutated[field] += 1
    elif field == "seed_policy":
        mutated[field] = "sha256_v1_other"
    elif field == "active_chunk_size":
        mutated[field] = 1 if mutated[field] is None else mutated[field] + 1
    else:
        raise AssertionError(f"未登錄的 plan binding 測試欄位：{field}")
    return mutated


def test_open_pilot_run_controller_no_stokes_is_lazy_and_read_only(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """no-Stokes open 必須完成只讀 binding，且不建立 manager、執行 shard 或 reconcile。

    workspace 由 public initializer 真實發布，故 normalized config、plan、progress、
    scenario table 與 seed table 都是實際 run-control 產物。此測試把 forcing manager
    constructor 改成不可觸發的 sentinel，並傳入不可探查的 NWW path，確認開啟 controller
    只建立 lookup 與 execution facade；progress bytes 的前後相等則鎖定沒有隱含狀態寫入。
    """

    data = runtime_fixture
    workspace, calls = _initialize_pilot_test_run(
        data, tmp_path, monkeypatch, run_id="pilot-open-no-stokes"
    )
    ocm_root = tmp_path / "ocm-native"
    ocm_root.mkdir()
    progress_path = workspace.path / "run_progress.json"
    before_progress = progress_path.read_bytes()

    def reject_forcing_manager(**_: Any) -> None:
        """若 open 路徑提前碰觸 forcing manager，立即讓測試失敗。"""

        raise AssertionError("open_pilot_run_controller 不得建立 forcing manager")

    monkeypatch.setattr(
        runtime.ForcingWindowManager,
        "from_roots",
        staticmethod(reject_forcing_manager),
    )
    controller = runtime.open_pilot_run_controller(
        workspace,
        config_path=EXAMPLE_CONFIG,
        ocm_native_root=ocm_root,
        nww_analysis_root=_UninspectablePath(),
    )

    assert isinstance(controller, RunController)
    assert isinstance(controller.request_factory, runtime.RuntimeRequestFactory)
    factory = controller.request_factory
    assert controller.resource_reporter == factory.resource_stats
    assert getattr(controller.resource_reporter, "__self__", None) is factory
    assert factory.resource_stats() == {
        "manager_count": 0,
        "loads": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "resident_bytes": 0,
    }
    assert all(type(value) is int for value in factory.resource_stats().values())

    after_progress = progress_path.read_bytes()
    assert after_progress == before_progress
    progress = json.loads(after_progress.decode("utf-8"))
    assert progress["run_lifecycle"] == "PLANNED"
    assert all(row["lifecycle"] == "PLANNED" for row in progress["shards"].values())
    assert all(row["attempt_count"] == 0 for row in progress["shards"].values())

    assert calls["config"] == {"path": EXAMPLE_CONFIG, "formal_release": False}
    assert calls["scenario"]["config"] is data["config"]
    assert calls["scenario"]["kwargs"] == {
        "config_path": EXAMPLE_CONFIG,
        "require_dynamic_initial_conditions": True,
        "formal": False,
    }
    assert calls["geometry"] == {
        "config": data["config"],
        "kwargs": {"config_path": EXAMPLE_CONFIG, "formal": False},
    }


def test_open_pilot_run_controller_forwards_resume_and_external_checkpoint_root(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resume 與 external checkpoint root 必須原樣傳給 validator 與 controller，且不改 progress。"""

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id="pilot-open-resume"
    )
    ocm_root = tmp_path / "ocm-native"
    ocm_root.mkdir()
    external_root = tmp_path / "external-checkpoints"
    external_root.mkdir()
    progress_path = workspace.path / "run_progress.json"
    before_progress = progress_path.read_bytes()
    real_validate_run = runtime.validate_run
    validation_calls: list[dict[str, Any]] = []

    def recording_validate_run(
        path: str | Path,
        *,
        require_complete: bool,
        checkpoint_root: str | Path | None,
    ) -> dict[str, Any]:
        """記錄參數後仍呼叫真實只讀 validator，避免測試繞過完整性檢查。"""

        validation_calls.append(
            {
                "path": path,
                "require_complete": require_complete,
                "checkpoint_root": checkpoint_root,
            }
        )
        return real_validate_run(
            path,
            require_complete=require_complete,
            checkpoint_root=checkpoint_root,
        )

    monkeypatch.setattr(runtime, "validate_run", recording_validate_run)
    controller = runtime.open_pilot_run_controller(
        workspace,
        config_path=EXAMPLE_CONFIG,
        ocm_native_root=ocm_root,
        resume=True,
        checkpoint_root=external_root,
    )

    assert isinstance(controller, RunController)
    assert controller.resume is True
    assert controller.checkpoint_root == external_root
    assert validation_calls == [
        {
            "path": workspace.path,
            "require_complete": False,
            "checkpoint_root": external_root,
        }
    ]
    assert progress_path.read_bytes() == before_progress


@pytest.mark.parametrize(
    "field",
    [
        "config_hash",
        "component_canonical_hashes",
        "geometry_canonical_hashes",
        "scenario_count",
        "master_seed",
        "seed_policy",
        "members_per_scenario",
        "shard_scenario_count",
        "checkpoint_interval_sweeps",
        "active_chunk_size",
    ],
)
def test_open_pilot_run_controller_rejects_each_plan_binding_before_factory(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """每個 immutable plan execution binding 失配都必須在 factory 建構前 fail closed。

    ``load_run_plan`` 只在記憶體回傳一份單欄位失配的合法 plan copy；磁碟 workspace 保持
    原樣，使 validator 仍完成真實完整性檢查。factory sentinel 的零呼叫是本測試的副作用
    邊界：只讀 binding 失敗不得初始化 manager、provider 或其他 runtime execution state。
    """

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id=f"pilot-open-binding-{field}"
    )
    real_load_run_plan = runtime.load_run_plan
    disk_plan = real_load_run_plan(workspace.path)
    factory_calls: list[dict[str, Any]] = []

    def reject_factory(**kwargs: Any) -> None:
        """記錄任何越過 binding gate 的 factory 建構嘗試。"""

        factory_calls.append(kwargs)
        raise AssertionError("plan binding 失配時不得建立 RuntimeRequestFactory")

    def mismatched_load_run_plan(path: str | Path) -> dict[str, Any]:
        """只改變指定欄位，模擬記憶體中的 plan snapshot 被竄改。"""

        assert Path(path) == workspace.path
        return _plan_with_binding_mismatch(disk_plan, field)

    monkeypatch.setattr(runtime, "load_run_plan", mismatched_load_run_plan)
    monkeypatch.setattr(runtime, "RuntimeRequestFactory", reject_factory)

    with pytest.raises(ValueError):
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
        )
    assert factory_calls == []


@pytest.mark.parametrize(
    ("plan_field", "bad_value"),
    [("run_kind", "formal"), ("experiment_case_id", "unregistered-case")],
)
def test_open_pilot_run_controller_rejects_mode_and_case_before_validator(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_field: str,
    bad_value: str,
) -> None:
    """非 pilot mode 與未知 experiment case 必須早於 validator、config 與 factory 被拒絕。"""

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id=f"pilot-open-invalid-{plan_field}"
    )
    disk_plan = runtime.load_run_plan(workspace.path)
    blocked_calls = {"validator": 0, "config": 0, "factory": 0}

    def invalid_plan_load(path: str | Path) -> dict[str, Any]:
        """保留所有磁碟欄位，只注入 mode 或 experiment case 的單一失配。"""

        assert Path(path) == workspace.path
        plan = deepcopy(disk_plan)
        plan[plan_field] = bad_value
        return plan

    def blocked_validator(*_: Any, **__: Any) -> dict[str, Any]:
        """若安全邊界失效，記錄 validator 呼叫並讓測試明確失敗。"""

        blocked_calls["validator"] += 1
        raise AssertionError("run_kind/experiment case 失配時不得呼叫 validator")

    def blocked_config(*_: Any, **__: Any) -> ProjectConfig:
        """若安全邊界失效，記錄 config loader 呼叫。"""

        blocked_calls["config"] += 1
        raise AssertionError("run_kind/experiment case 失配時不得載入 config")

    def blocked_factory(**_: Any) -> None:
        """若安全邊界失效，記錄 factory 建構呼叫。"""

        blocked_calls["factory"] += 1
        raise AssertionError("run_kind/experiment case 失配時不得建立 factory")

    monkeypatch.setattr(runtime, "load_run_plan", invalid_plan_load)
    monkeypatch.setattr(runtime, "validate_run", blocked_validator)
    monkeypatch.setattr(runtime, "load_config", blocked_config)
    monkeypatch.setattr(runtime, "RuntimeRequestFactory", blocked_factory)

    with pytest.raises(ValueError):
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
        )
    assert blocked_calls == {"validator": 0, "config": 0, "factory": 0}


@pytest.mark.parametrize("bad_resume", [1, "true"])
def test_open_pilot_run_controller_requires_native_bool_resume(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_resume: Any,
) -> None:
    """resume 不接受整數或字串 coercion，且型別錯誤必須發生在 factory 前。"""

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id=f"pilot-open-resume-{type(bad_resume).__name__}"
    )
    factory_calls: list[dict[str, Any]] = []

    def reject_factory(**kwargs: Any) -> None:
        """記錄不應被錯誤型別穿透的 factory 建構。"""

        factory_calls.append(kwargs)
        raise AssertionError("invalid resume 型別時不得建立 factory")

    monkeypatch.setattr(runtime, "RuntimeRequestFactory", reject_factory)
    with pytest.raises(TypeError):
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
            resume=bad_resume,
        )
    assert factory_calls == []


def test_open_pilot_run_controller_rejects_validator_invalid_workspace_without_path_leak(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真實 validator 判定 workspace invalid 時，open 必須只回報 safe errors 且不改 progress。"""

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id="pilot-open-validator-invalid"
    )
    (workspace.path / "unexpected-runtime-file").write_text("tampered\n", encoding="utf-8")
    external_root = tmp_path / "external-checkpoints"
    external_root.mkdir()
    progress_path = workspace.path / "run_progress.json"
    before_progress = progress_path.read_bytes()
    factory_calls: list[dict[str, Any]] = []

    def reject_factory(**kwargs: Any) -> None:
        """若 validator 失敗仍建立 factory，立即暴露順序回歸。"""

        factory_calls.append(kwargs)
        raise AssertionError("validator invalid 時不得建立 factory")

    monkeypatch.setattr(runtime, "RuntimeRequestFactory", reject_factory)
    with pytest.raises(ValueError) as exc_info:
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
            checkpoint_root=external_root,
        )
    message = str(exc_info.value)
    assert str(workspace.path) not in message
    assert str(external_root) not in message
    assert factory_calls == []
    assert progress_path.read_bytes() == before_progress


@pytest.mark.parametrize(
    "normalized_case",
    ["duplicate", "nan", "infinity", "non_object", "symlink"],
)
def test_open_pilot_run_controller_strictly_rejects_invalid_normalized_config(
    runtime_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    normalized_case: str,
) -> None:
    """normalized_config 的 duplicate、非有限值、非 object 與 symlink 都要在 factory 前拒絕。

    這裡刻意只讓 ``validate_run`` 回傳 valid=true，隔離測試 normalized reader 的責任；
    workspace 本身仍由 initializer 真實建立。每個參數化案例使用獨立 ``tmp_path``，因此
    symlink 或損壞 bytes 不需恢復，也不會影響其他測試的 immutable input snapshot。
    """

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id=f"pilot-open-normalized-{normalized_case}"
    )
    normalized_path = workspace.path / "normalized_config.json"
    if normalized_case == "duplicate":
        normalized_path.write_bytes(b'{"duplicate": 1, "duplicate": 2}\n')
    elif normalized_case == "nan":
        normalized_path.write_bytes(b'{"value": NaN}\n')
    elif normalized_case == "infinity":
        normalized_path.write_bytes(b'{"value": Infinity}\n')
    elif normalized_case == "non_object":
        normalized_path.write_bytes(b"[]\n")
    else:
        target = tmp_path / "normalized-config-target.json"
        target.write_bytes(b"{}\n")
        normalized_path.unlink()
        normalized_path.symlink_to(target)

    monkeypatch.setattr(runtime, "validate_run", lambda *_args, **_kwargs: {"valid": True, "errors": []})
    factory_calls: list[dict[str, Any]] = []

    def reject_factory(**kwargs: Any) -> None:
        """記錄 strict normalized reader 不應越過的 factory 建構。"""

        factory_calls.append(kwargs)
        raise AssertionError("invalid normalized_config 時不得建立 factory")

    monkeypatch.setattr(runtime, "RuntimeRequestFactory", reject_factory)
    with pytest.raises(ValueError):
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
        )
    assert factory_calls == []


def test_open_pilot_run_controller_rejects_normalized_config_exact_mismatch_before_factory(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """validator 真實通過時，normalized config 單一欄位失配仍必須在 factory 前拒絕。"""

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id="pilot-open-normalized-mismatch"
    )
    real_reader = runtime._read_strict_json_object

    def mismatched_reader(path: str | Path, *, label: str) -> dict[str, Any]:
        """讀取真實 snapshot 後只改一個欄位，不觸碰磁碟 normalized config。"""

        payload = real_reader(path, label=label)
        payload["config_status"] = "__normalized-config-mismatch__"
        return payload

    monkeypatch.setattr(runtime, "_read_strict_json_object", mismatched_reader)
    factory_calls: list[dict[str, Any]] = []

    def reject_factory(**kwargs: Any) -> None:
        """記錄 exact normalized binding 失敗時不應發生的 factory 建構。"""

        factory_calls.append(kwargs)
        raise AssertionError("normalized config mismatch 時不得建立 factory")

    monkeypatch.setattr(runtime, "RuntimeRequestFactory", reject_factory)
    with pytest.raises(ValueError):
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
        )
    assert factory_calls == []


def test_open_pilot_run_controller_validates_workspace_before_config_and_factory(
    runtime_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """validator invalid 時必須停止在 validate_run，不能提前載入 config、manifest 或 factory。"""

    workspace, _ = _initialize_pilot_test_run(
        runtime_fixture, tmp_path, monkeypatch, run_id="pilot-open-validate-first"
    )
    (workspace.path / "validator-rejects-this-file").write_text("tampered\n", encoding="utf-8")
    real_validate_run = runtime.validate_run
    calls = {"validator": 0, "config": 0, "scenario": 0, "geometry": 0, "factory": 0}

    def recording_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """呼叫真實 validator 並記錄 validate-first 順序邊界。"""

        calls["validator"] += 1
        return real_validate_run(*args, **kwargs)

    def blocked_config(*_: Any, **__: Any) -> ProjectConfig:
        """若 config loader 被提前觸發，留下明確失敗訊號。"""

        calls["config"] += 1
        raise AssertionError("validator 失敗後不得載入 config")

    def blocked_scenario(*_: Any, **__: Any) -> ScenarioInputs:
        """若 scenario loader 被提前觸發，留下明確失敗訊號。"""

        calls["scenario"] += 1
        raise AssertionError("validator 失敗後不得載入 scenario")

    def blocked_geometry(*_: Any, **__: Any) -> BoundaryGeometryBundle:
        """若 geometry loader 被提前觸發，留下明確失敗訊號。"""

        calls["geometry"] += 1
        raise AssertionError("validator 失敗後不得載入 geometry")

    def blocked_factory(**_: Any) -> None:
        """若 factory 被提前觸發，留下明確失敗訊號。"""

        calls["factory"] += 1
        raise AssertionError("validator 失敗後不得建立 factory")

    monkeypatch.setattr(runtime, "validate_run", recording_validator)
    monkeypatch.setattr(runtime, "load_config", blocked_config)
    monkeypatch.setattr(runtime, "load_scenario_inputs", blocked_scenario)
    monkeypatch.setattr(runtime, "load_boundary_geometries", blocked_geometry)
    monkeypatch.setattr(runtime, "RuntimeRequestFactory", blocked_factory)

    with pytest.raises(ValueError):
        runtime.open_pilot_run_controller(
            workspace,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
        )
    assert calls == {
        "validator": 1,
        "config": 0,
        "scenario": 0,
        "geometry": 0,
        "factory": 0,
    }
