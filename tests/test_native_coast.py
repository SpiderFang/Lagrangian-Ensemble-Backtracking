"""核准模型岸界的首接觸、來源綁定及舊行為相容測試，不代表真資料重算驗收。"""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import LineString, box, mapping
from test_boundaries_engine import _state

from lagrangian_backtracking.boundaries import BoundaryGeometry, resolve_horizontal_boundaries
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.models import EventType, ParticleStatus
from lagrangian_backtracking.native_coast import NativeCoast, canonical_sha, file_sha, load_native_coasts


def coast(region=None):
    """一個公尺矩形島嶼，逐原邊建立索引，無速度取樣或亂數。"""
    region = box(2, -1, 4, 1) if region is None else region
    coords = list(region.exterior.coords)
    edges = [
        dict(
            edge_id=f"e{i}",
            coordinates_m=[a, b],
            source_edge_global_index=i,
            source_face_global_index=i,
            owner_count=1,
            has_core_endpoint=True,
        )
        for i, (a, b) in enumerate(zip(coords[:-1], coords[1:], strict=True))
    ]
    return NativeCoast(region, edges, manifest_sha256="a" * 64, region_sha256=canonical_sha(mapping(region)))


def boundary(native=True):
    """外框全為開放邊界，模型岸界只在明示時加入，不套用到底圖或其他空洞。"""
    return BoundaryGeometry(
        box(-10, -10, 10, 10),
        box(-10, -10, 10, 10),
        {},
        local_equals_flow=True,
        native_coast=coast() if native else None,
    )


@pytest.mark.parametrize("end,expected_x", [(3, 2), (5, 2), (20, 2)])
def test_first_contact_and_cross_island(end, expected_x):
    """海→陸、海→海穿島及更晚外框出口，皆在較早模型岸界同步停時／深度／年齡。"""
    previous = _state(0)
    proposed = replace(previous, x_m=end, z_m=-1, time_utc_ns=80_000_000_000, age_seconds=20)
    state, events = resolve_horizontal_boundaries(previous, proposed, boundary())
    assert state.status == ParticleStatus.COAST_CONTACT and state.x_m == expected_x
    fraction = 2 / end
    assert state.age_seconds == 20 * fraction and state.z_m == -5 + 4 * fraction
    assert state.time_utc_ns == 100_000_000_000 + round(-20_000_000_000 * fraction)
    assert len(events) == 1 and events[0].attributes["source_edge_global_index"] >= 0
    assert events[0].boundary_s_m == 1.0


@pytest.mark.parametrize(
    "start,end,contact",
    [
        ((0, 2), (5, 2), False),
        ((0, 1), (5, 1), True),
        ((0, 3), (2, 1), True),
        ((2, 0), (0, 0), False),
        ((2, 0), (3, 0), True),
        ((2, 0), (2, 1), True),
        ((3, 0), (5, 0), False),
        ((0, 0), (0, 0), False),
    ],
)
def test_touch_direction_tangent_and_invalid_start(start, end, contact):
    """切線／共點為接觸；步首朝水離開不重報，朝岸或沿岸停，非法內部state不重分類。"""
    value = coast().first_contact(start, end)
    assert (value is not None) == contact
    if contact and start == (2, 0):
        assert value.fraction == 0


def test_no_setting_and_unregistered_hole_keep_old_path():
    """沒有模型岸界時水中同一路段不被新增停止，反向離開外框仍是flow exit。"""
    p = _state(0)
    q = replace(p, x_m=5)
    state, events = resolve_horizontal_boundaries(p, q, boundary(False))
    assert state == q and events == []
    state, events = resolve_horizontal_boundaries(p, replace(q, x_m=-20), boundary())
    assert state.status == ParticleStatus.FLOW_DOMAIN_EXIT
    assert events[0].event_type == EventType.FLOW_DOMAIN_OPEN_EXIT


def test_local_event_order_and_no_rng_consumption():
    """非終止local在岸界之前保留，岸界後的事件不寫；查詢不消耗外部RNG。"""
    rng = np.random.default_rng(4)
    before = deepcopy(rng.bit_generator.state)
    b = replace(boundary(), local_equals_flow=False, own_local_domain=box(-1, -2, 1, 2))
    s, events = resolve_horizontal_boundaries(_state(0), replace(_state(0), x_m=20), b)
    assert [e.event_type for e in events] == [EventType.LOCAL_DOMAIN_FIRST_EXIT, EventType.COAST_CONTACT]
    assert s.own_local_exit_recorded and before == rng.bit_generator.state


def manifest_fixture(tmp_path):
    """合成核准站資料驗證loader；來源檔／manifest／config各有獨立SHA。"""
    projection = DomainProjection(120, 22)
    c = coast()
    source = tmp_path / "source.json"
    source.write_text("{}")
    rec = dict(
        study_site_id="nanwan",
        component_id="nanwan-boundary-0",
        flow_domain_id="flow",
        projection_wkt=projection._forward.target_crs.to_wkt(),
        region_geometry_m=mapping(c.region),
        region_sha256=c.region_sha256,
        edges=list(c.edges),
        edges_sha256=canonical_sha(list(c.edges)),
    )
    payload = dict(
        schema_version="1.0.0",
        manifest_kind="native_coast_manifest",
        status="approved",
        method="stop_at_first_contact",
        coverage_policy="reviewed_two_components_only_v1",
        source_files=[dict(path=str(source), sha256=file_sha(source))],
        geometry_file_sha256=dict(domain="d", local="l", open_boundary="o"),
        records=[rec],
    )
    path = tmp_path / "native.json"
    path.write_text(json.dumps(payload))
    kwargs = dict(
        config_path=tmp_path / "config.yaml",
        site_flow={"nanwan": "flow"},
        projections={"nanwan": projection},
        geometry_sha256=payload["geometry_file_sha256"],
    )
    return path, payload, kwargs


def test_loader_and_deepcopy_index(tmp_path):
    """可選來源載入後可經runtime defensive deepcopy重建，交點數值及來源不變。"""
    path, _, kwargs = manifest_fixture(tmp_path)
    coasts, _, _ = load_native_coasts(dict(path=str(path), sha256=file_sha(path)), **kwargs)
    c = coasts["nanwan"]
    assert deepcopy(c).first_contact((0, 0), (5, 0)) == c.first_contact((0, 0), (5, 0))


@pytest.mark.parametrize(
    "fault",
    [
        "file_sha",
        "source_sha",
        "site",
        "projection",
        "flow",
        "component",
        "schema",
        "geometry",
        "edge_hash",
        "region_hash",
        "noncore",
        "owner",
    ],
)
def test_loader_rejects_misbound_source(tmp_path, fault):
    """錯誤來源、站點、投影、schema、裁切邊與指紋不能靠可讀JSON繞過安全綁定。"""
    path, p, kwargs = manifest_fixture(tmp_path)
    r = p["records"][0]
    if fault == "source_sha":
        p["source_files"][0]["sha256"] = "0" * 64
    elif fault == "site":
        r["study_site_id"] = "other"
    elif fault == "projection":
        r["projection_wkt"] = DomainProjection(121, 22)._forward.target_crs.to_wkt()
    elif fault == "flow":
        r["flow_domain_id"] = "other"
    elif fault == "component":
        r["component_id"] = "other"
    elif fault == "schema":
        p["schema_version"] = "bad"
    elif fault == "geometry":
        p["geometry_file_sha256"] = {}
    elif fault == "edge_hash":
        r["edges"][0]["coordinates_m"][0] = [9, 9]
    elif fault == "region_hash":
        r["region_sha256"] = "0" * 64
    elif fault in ("noncore", "owner"):
        r["edges"][0]["has_core_endpoint" if fault == "noncore" else "owner_count"] = (
            False if fault == "noncore" else 2
        )
        r["edges_sha256"] = canonical_sha(r["edges"])
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_native_coasts(
            dict(path=str(path), sha256="0" * 64 if fault == "file_sha" else file_sha(path)), **kwargs
        )


def test_config_absent_hash_and_changed_coast_binding():
    """未設定geometry不增加預設欄位；只換岸界版本SHA也會改config hash。"""
    from test_config import _legacy_example_payload

    from lagrangian_backtracking.config import ProjectConfig

    p = _legacy_example_payload()
    old = ProjectConfig.model_validate(p)
    assert "native_coast_manifest" not in old.normalized_payload()["geometry"]
    p["geometry"]["native_coast_manifest"] = dict(path="coast.json", sha256="a" * 64)
    first_hash = ProjectConfig.model_validate(p).config_hash()
    p["geometry"]["native_coast_manifest"]["sha256"] = "b" * 64
    assert first_hash != ProjectConfig.model_validate(p).config_hash()


def test_new_boundary_version_rejects_checkpoint(tmp_path):
    """岸界引用SHA進入config binding，不能用新岸界續接舊checkpoint。"""
    from test_config import _legacy_example_payload

    from lagrangian_backtracking.checkpoint import CheckpointBinding, load_checkpoint, write_checkpoint
    from lagrangian_backtracking.config import ProjectConfig

    payload = _legacy_example_payload()
    payload["geometry"]["native_coast_manifest"] = dict(path="coast.json", sha256="a" * 64)
    old = ProjectConfig.model_validate(payload).config_hash()
    binding = CheckpointBinding(old, "inputs", "baseline", "shard", "sha256-v1", "code")
    checkpoint = write_checkpoint(tmp_path / "checkpoint", binding=binding, states=[_state(0)], sequence=1)
    payload["geometry"]["native_coast_manifest"]["sha256"] = "b" * 64
    changed = replace(binding, config_hash=ProjectConfig.model_validate(payload).config_hash())
    with pytest.raises(ValueError, match="binding 不相容"):
        load_checkpoint(checkpoint, expected_binding=changed)


@pytest.mark.parametrize("qc_name", ["OUTSIDE_HORIZONTAL_DOMAIN", "DRY_FACE", "TIME_GAP"])
@pytest.mark.parametrize("invalid_start", [False, True])
def test_engine_stage_coast_recovery_does_not_reclassify_other_qc(qc_name, invalid_start):
    """只有有效步首後純水平stage失效可沿既有近似線定位；乾點／時間／非法步首不得改岸。"""
    from lagrangian_backtracking.diffusion import DiffusionCoefficients
    from lagrangian_backtracking.engine import EngineSettings, run_particle
    from lagrangian_backtracking.models import SampleQC, VelocitySample

    qc = getattr(SampleQC, qc_name)

    def velocity(x, y, z, utc):
        """合成只在島內失效之流，回溯向東；不提供缺值替代速度。"""
        return VelocitySample(-1, 0, 0, 0, -10, 100, 1, qc if invalid_start or x >= 2 else SampleQC.OK)

    result = run_particle(
        _state(0),
        velocity=velocity,
        boundaries=boundary(),
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0, 0, 0),
        settings=EngineSettings(1, 4, 4, 100, 100, 0),
        rng=np.random.default_rng(9),
    )
    should_coast = qc_name == "OUTSIDE_HORIZONTAL_DOMAIN" and not invalid_start
    assert (result.final_state.status == ParticleStatus.COAST_CONTACT) == should_coast
    if should_coast:
        assert result.final_state.x_m == 2
        event = next(e for e in result.events if e.event_type == EventType.COAST_CONTACT)
        assert event.attributes["requires_dt_halving_validation"] is True
        assert event.attributes["boundary_locator"] == "reference_drift_after_rk_stage_invalid"
    else:
        assert all(e.event_type != EventType.COAST_CONTACT for e in result.events)


def test_manifest_sea_line_rejected_even_with_updated_fingerprint(tmp_path):
    """指紋有效仍不足以把海中線當岸；停止線必須位於所綁region邊界。"""
    path, payload, kwargs = manifest_fixture(tmp_path)
    record = payload["records"][0]
    record["edges"][0]["coordinates_m"] = [[0, 0], [0, 1]]
    record["edges_sha256"] = canonical_sha(record["edges"])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="region邊界"):
        load_native_coasts(dict(path=str(path), sha256=file_sha(path)), **kwargs)


def test_geometry_loader_optional_hashes_and_default_unchanged(tmp_path):
    """正式五站loader保留原三份來源，僅明示設定時新增模型岸界的檔案／canonical綁定。"""
    from test_manifests import _config, _geometry_payloads

    from lagrangian_backtracking.manifests import load_boundary_geometries

    config = _config(tmp_path)
    paths = [tmp_path / name for name in ("domain.json", "local.json", "open.json")]
    for path, document in zip(paths, _geometry_payloads(), strict=True):
        path.write_text(json.dumps(document))
    before = load_boundary_geometries(config, *paths, formal=True)
    manifest, document, _ = manifest_fixture(tmp_path)
    record = document["records"][0]
    record["flow_domain_id"] = next(
        s.flow_domain_id for s in config.study_sites if s.study_site_id == "nanwan"
    )
    record["projection_wkt"] = before.projections["nanwan"]._forward.target_crs.to_wkt()
    document["geometry_file_sha256"] = dict(before.file_sha256)
    manifest.write_text(json.dumps(document))
    config = config.model_copy(
        update={
            "geometry": dict(
                config.geometry, native_coast_manifest=dict(path=str(manifest), sha256=file_sha(manifest))
            )
        }
    )
    after = load_boundary_geometries(config, *paths, formal=True)
    assert set(after.file_sha256) == set(before.file_sha256) | {"native_coast"}
    assert after.canonical_component_hashes["native_coast"] == canonical_sha(document)
    for key, value in before.canonical_component_hashes.items():
        assert after.canonical_component_hashes[key] == value
    assert after["nanwan"].native_coast is not None
    assert all(after[s].native_coast is None for s in after if s != "nanwan")


def test_artifact_extraction_excludes_artificial_open_and_noncore_edges():
    """區域封口不因是region邊界就變岸；原生open邊與非core裁切邊均排除。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_reviewed_native_coast import extract_edges

    region = box(0, 0, 4, 4)
    lines = [LineString([(0, 0), (4, 0)]), LineString([(4, 0), (4, 4)]), LineString([(4, 4), (0, 4)])]
    edges = [
        dict(
            boundary_edge_index=i,
            node_local_indices=[2 * i, 2 * i + 1],
            has_core_endpoint=(i != 2),
            source_edge_global_index=i,
        )
        for i in range(3)
    ]
    component = dict(site="nanwan", component_id="nanwan-boundary-0", edges=edges)
    nodes = {i: dict(global_index=i) for i in range(6)}
    result = extract_edges(component, region, nodes, lines, box(-1, -1, 5, 5), lines[1])
    assert len(result) == 1 and result[0]["source_edge_global_index"] == 0
    assert result[0]["coordinates_m"] == [[0.0, 0.0], [4.0, 0.0]]


def test_real_native_batch_then_model_coast_without_legacy_or_rng_draw(tmp_path, monkeypatch):
    """原版Numba批次器真的完成兩粒子平流後，既有邊界流程攔截穿島；不調legacy或亂數。"""
    from pathlib import Path

    import test_baytrace_ocm_adapter as fixtures
    from numba.core.registry import CPUDispatcher

    from lagrangian_backtracking.baytrace_batch_advection import NativeBayTraceBatchAdvector
    from lagrangian_backtracking.baytrace_native_cpu import load_original_cpu
    from lagrangian_backtracking.baytrace_ocm_adapter import OCMPreparedFrameResolver
    from lagrangian_backtracking.forcing_window import ManagedForcingProvider
    from lagrangian_backtracking.production import HintTrackingVelocityProvider

    source = Path(__file__).resolve().parents[1] / "BayTrace/Baytech_BayTrace_v4.7.2"
    if not source.is_dir():
        source = Path(
            "/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/BayTrace/Baytech_BayTrace_v4.7.2"
        )
    if not source.is_dir():
        pytest.skip("未找到指定PI原版來源")
    mesh = fixtures._mesh()
    month = fixtures._month(mesh, settling_current_mps=0.03)
    manager = fixtures._manager(mesh, {month.month_id: month})
    provider = HintTrackingVelocityProvider(
        ManagedForcingProvider(manager, settling_velocity_mps=-0.02, include_stokes=False), triangle_hint=0
    )
    task = fixtures._task(provider)
    batcher = NativeBayTraceBatchAdvector(
        load_original_cpu(source, tmp_path / "cache"), OCMPreparedFrameResolver(), require_native=True
    )
    import lagrangian_backtracking.baytrace_batch_advection as batch_module

    def forbidden(*args, **kwargs):
        """任何legacy回補皆令測試立即失敗，不能只看native>0宣稱通過。"""
        raise AssertionError("不可呼叫legacy")

    monkeypatch.setattr(batch_module, "_execute_legacy_advection_task", forbidden)
    rng = np.random.default_rng(34)
    before = deepcopy(rng.bit_generator.state)
    results = batcher([task, replace(task, state=replace(task.state, particle_id="p1"))])
    a = np.array([task.state.x_m, task.state.y_m])
    b = np.array([results[0].x_m, results[0].y_m])
    direction = b - a
    width = np.array([-direction[1], direction[0]])
    from shapely.geometry import Polygon

    island = Polygon(
        [
            a + direction * 0.4 + width,
            a + direction * 0.6 + width,
            a + direction * 0.6 - width,
            a + direction * 0.4 - width,
        ]
    )
    geometry = replace(
        boundary(False),
        flow_domain=box(-1e6, -1e6, 1e6, 1e6),
        own_local_domain=box(-1e6, -1e6, 1e6, 1e6),
        native_coast=coast(island),
    )
    stopped, events = resolve_horizontal_boundaries(task.state, results[0], geometry)
    assert stopped.status == ParticleStatus.COAST_CONTACT
    assert events[0].fraction == pytest.approx(0.4)
    report = batcher.report()
    assert report["native_completed_particle_steps"] == 2
    assert report["fallback_particle_steps"] == 0 and report["native_rejected_particle_steps"] == 0
    assert report["group_size_histogram"] == {"2": 1}
    assert isinstance(batcher.kernels.batch_move_rk4, CPUDispatcher)
    assert batcher.kernels.batch_move_rk4.signatures
    assert rng.bit_generator.state == before

    # 再走真正ProductionBatch→engine→原版batcher→擴散→岸界的完整單步生命週期。
    # 兩份相同seed的批次只差有無岸界；同一單步的RNG狀態必須完全一致。
    from test_production_batch_advection import _request, _shard

    from lagrangian_backtracking.diffusion import DiffusionCoefficients
    from lagrangian_backtracking.engine import EngineSettings
    from lagrangian_backtracking.production import ProductionBatch

    shard = _shard(members_per_scenario=2)
    shard = replace(
        shard, scenarios=(replace(shard.scenarios[0], arrival_time_utc_ns=task.state.time_utc_ns),)
    )

    def request_for(unit, *, with_coast):
        """同一合成OCM與極小非零擴散，保留unit身分與初始時刻；不改原版步長或亂數。"""
        request = _request(unit, settings=EngineSettings(1, 1, 1, 10, 20, 0))
        state = replace(request.initial_state, x_m=task.state.x_m, y_m=task.state.y_m, z_m=task.state.z_m)
        return replace(
            request,
            initial_state=state,
            velocity=provider.provider,
            diffusion=DiffusionCoefficients(1e-10, 1e-10, 1e-10),
            boundaries=geometry if with_coast else replace(geometry, native_coast=None),
        )

    plain = ProductionBatch(
        shard,
        master_seed=34,
        request_factory=lambda u: request_for(u, with_coast=False),
        advection_batcher=batcher,
    )
    plain.advance(1)
    # engine實際步首sample與上方獨立AdvectionTask的人工sample不同；用同一engine
    # 單步的合成水中路段佈置測試島，不假設兩種reference會產生相同推進方向。
    end = plain.runtimes[0].execution.state
    direction = np.array([end.x_m, end.y_m]) - a
    width = np.array([-direction[1], direction[0]])
    island = Polygon(
        [
            a + direction * 0.4 + width,
            a + direction * 0.6 + width,
            a + direction * 0.6 - width,
            a + direction * 0.4 - width,
        ]
    )
    geometry = replace(geometry, native_coast=coast(island))
    coastal = ProductionBatch(
        shard,
        master_seed=34,
        request_factory=lambda u: request_for(u, with_coast=True),
        advection_batcher=batcher,
    )
    coastal.advance(1)
    for p, c in zip(plain.runtimes, coastal.runtimes, strict=True):
        assert p.execution.state.status == ParticleStatus.ACTIVE
        assert c.execution.state.status == ParticleStatus.COAST_CONTACT
        assert p.rng.bit_generator.state == c.rng.bit_generator.state
    assert batcher.report()["native_completed_particle_steps"] == 6
    assert batcher.report()["fallback_particle_steps"] == 0
