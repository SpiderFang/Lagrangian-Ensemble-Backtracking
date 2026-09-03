"""aggregate payload pipeline 的 synthetic 工程整合測試。

本檔只驗證 ``build_aggregate_release_payload`` 如何把一個已完成的 pilot run、已載入的
static inputs、AggregateSpec 與單一 trajectory shard 串成 release payload。測試沿用既有
runtime fixture，但不建立 OCM schema 3 或 NWW3 schema 1 forcing array，也不宣稱這些
數值是實際海流、來源足跡、絕對來源機率或觀測成果；所有通過案例都只是工程資料契約、
檔案完整性、資料順序與固定記憶體流程的 synthetic 驗證。

payload 建構本身應只讀 source workspace：static binding 不建立 forcing factory／manager，
trajectory iterator 一次串流一個 shard，事件與 pathway reducer 當下吸收 chunk 後釋放
上游 reference。真正的 OCM／NWW 科學結果仍須以 SERVER 上已驗收的正式產品執行，不能
由本機 fixture 的成功取代。
"""

from __future__ import annotations

import inspect
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_aggregate_release_public_validation as public_validation
import test_runtime_static_inputs as runtime_static_fixture
from shapely.geometry import LineString, box
from test_run_control import _request
from test_runtime import EXAMPLE_CONFIG, _initialize_pilot_test_run

import lagrangian_backtracking.aggregate_pipeline as pipeline
import lagrangian_backtracking.runtime as runtime
from lagrangian_backtracking.aggregate_release import (
    read_aggregate_release,
    validate_aggregate_release,
    write_aggregate_release,
)
from lagrangian_backtracking.aggregate_release_codec import encode_aggregate_release_payload
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload
from lagrangian_backtracking.aggregate_release_records import scenario_inputs_to_strata
from lagrangian_backtracking.aggregate_spec import (
    AggregateSpec,
    SiteBoundarySegments,
    load_aggregate_spec,
    write_aggregate_spec_from_boundaries,
)
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.models import ParticleStatus
from lagrangian_backtracking.run_control import (
    RunController,
    load_run_plan,
    load_run_progress,
)
from lagrangian_backtracking.run_locking import RunLockBusyError, acquire_run_lock

_PAYLOAD_FAILURE = "aggregate release payload 建構失敗"
_FINAL_SUFFIX = ".aggregate-v1"


@dataclass(frozen=True, slots=True)
class _AggregatePipelineFixture:
    """保存一份 module-scoped synthetic pilot 與其幾何導出的規格。

    ``data`` 由既有 runtime static-input helper 建立，workspace 則由真實 pilot initializer
    發布後再用同一個 ``RunController`` 完成唯一 shard。幾何故意是公尺制 200 m 方形與
    底邊開放線；它只讓測試能穩定檢查格線／路徑 shape，不代表真實海岸或 OCM/NWW 產品。
    ``spec_path`` 是 workspace 外的 caller-owned JSON，方便同一份原始 bytes 同時餵給
    payload pipeline 與 public release writer。
    """

    data: dict[str, Any]
    workspace: Path
    spec_path: Path
    spec: AggregateSpec
    site_id: str
    centers: dict[str, tuple[float, float]]
    plan: dict[str, Any]


def _aggregate_runtime_data() -> tuple[dict[str, Any], str]:
    """把既有小型 runtime records 換成 payload 測試所需的固定 synthetic 邊界。

    input records、component canonical hashes 與 projection mapping 都直接重用既有
    ``_runtime_data``；只替換 BoundaryGeometry 的局部／flow polygon 與明示開放線。方形
    四邊皆在公尺座標，底邊 LineString 落在 flow boundary 上，``local_equals_flow``
    讓同一個安全 segment ID 同時代表 local 與 outer 角色。這些 synthetic geometry
    不是實際海岸線，不能用來推論 OCM/NWW 科學結果。
    """

    data = runtime_static_fixture._runtime_data()
    bundle = data["geometries"]
    site_id = next(iter(bundle.geometries))
    open_line = LineString([(-100.0, -100.0), (100.0, -100.0)])
    aggregate_geometry = BoundaryGeometry(
        own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
        flow_domain=box(-100.0, -100.0, 100.0, 100.0),
        foreign_local_domains={},
        own_local_open_boundary=open_line,
        flow_open_boundary=open_line,
        local_equals_flow=True,
        own_local_boundary_segment_id="aggregate-open",
        flow_boundary_segment_id="aggregate-open",
    )
    # 只替換幾何 mapping；原 bundle 的 projection 與 canonical component hashes 必須
    # 原封不動保留，才能測到「規格由目前已驗證 bundle/config 導出」而非另一份 fixture。
    data["geometries"] = replace(bundle, geometries={site_id: aggregate_geometry})
    return data, site_id


def _site_metric_centers(
    data: Mapping[str, Any],
    site_ids: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    """從 config 的 analysis-region ``center_lonlat`` 取得明示投影中心（單位為度）。

    pipeline 及 AggregateSpec validator 都禁止由 projection private state 猜中心；測試
    因此沿用正式 config 的 site→region 關聯，將經緯度只作 AEQD 定義，後續格線仍在公尺
    座標比較。site IDs 若不在 config 會立即失敗，避免 synthetic extra site 偷渡。
    """

    config = data["config"]
    result: dict[str, tuple[float, float]] = {}
    for site_id in site_ids:
        site = next(item for item in config.study_sites if item.study_site_id == site_id)
        domain = next(
            item
            for item in config.domains
            if item.analysis_region_id == site.analysis_region_id
        )
        center = domain.center_lonlat
        result[site_id] = (float(center[0]), float(center[1]))
    return result


def _write_test_spec(
    data: Mapping[str, Any],
    target: Path,
    *,
    run_id: str,
    site_ids: tuple[str, ...] | None = None,
) -> tuple[Path, AggregateSpec, dict[str, tuple[float, float]]]:
    """用 production geometry writer 產生一份 exact run-bound synthetic AggregateSpec。

    cell、boundary bin 與 KDE bandwidth 都是公尺；age 邊界是秒；bootstrap 參數只是
    固定設定值，不會在本測試計算任何不確定性。writer 的輸入是 in-memory Bundle，故不
    回讀 raw NetCDF 或 forcing，也不把測試數值提升為正式科學成果。
    """

    selected_sites = site_ids or tuple(data["geometries"].geometries)
    centers = _site_metric_centers(data, selected_sites)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_aggregate_spec_from_boundaries(
        target,
        data["geometries"],
        run_id=run_id,
        site_metric_centers_deg=centers,
        grid_cell_size_m=100,
        boundary_bin_size_m=100,
        kde_bandwidths_m=(100, 200, 300),
        age_bin_edges_seconds=(0, 10),
        bootstrap_replicates=10,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=123,
    )
    return target, load_aggregate_spec(target), centers


def _initialize_run(
    data: dict[str, Any],
    base: Path,
    *,
    run_id: str,
    checkpoint_root: Path | None = None,
) -> Path:
    """透過既有 initializer 發布 run，再以唯一 request 完成一個 shard。

    ``_initialize_pilot_test_run`` 只建立 immutable plan/progress；真正的 two-member
    trajectory 由既有 test_run_control 的 ``_request`` 和 reference RunController 產生。
    這個 request 使用零擴散 constant flow，是工程 fixture，不是 OCM/NWW forcing 執行。
    ``checkpoint_root`` 僅在 external-checkpoint regression 使用，絕不寫進 payload。
    """

    base.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        workspace, _ = _initialize_pilot_test_run(
            data,
            base,
            patch,
            run_id=run_id,
        )
    controller = RunController(
        workspace,
        request_factory=_request,
        checkpoint_root=checkpoint_root,
    )
    summaries = controller.run_all()
    assert len(summaries) == 1
    assert summaries[0].run_id == run_id
    assert summaries[0].lifecycle == "COMPLETE"
    reconciliation = controller.reconcile()
    assert reconciliation["run_id"] == run_id
    return workspace.path


@pytest.fixture(scope="module")
def aggregate_pipeline_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> _AggregatePipelineFixture:
    """建立一次可重用的 completed pilot template，降低每個案例的 fixture 成本。

    module fixture 只建構一個 scenario×兩個 members 的 synthetic run；各測試只讀它，
    不修改 source tree。payload 及 release 的成功只表示本機工程 contract 通過，不是
    真實 OCM schema 3、NWW3 schema 1 或研究結論。
    """

    data, site_id = _aggregate_runtime_data()
    base = tmp_path_factory.mktemp("aggregate-pipeline")
    workspace = _initialize_run(
        data,
        base,
        run_id="aggregate-pipeline-pilot",
    )
    spec_path, spec, centers = _write_test_spec(
        data,
        base / "aggregate_spec.json",
        run_id="aggregate-pipeline-pilot",
    )
    return _AggregatePipelineFixture(
        data=data,
        workspace=workspace,
        spec_path=spec_path,
        spec=spec,
        site_id=site_id,
        centers=centers,
        plan=load_run_plan(workspace),
    )


def _snapshot_tree(root: Path) -> dict[str, tuple[str, object]]:
    """保存 source workspace 每個節點的型別與 ordinary-file exact bytes。

    payload pipeline 只允許讀取 source；snapshot 同時記錄 directory／symlink topology，
    避免只比檔案內容而漏掉刪除或替換節點。這裡只作用於 synthetic run workspace，沒有
    任何 OCM/NWW raw data，也不把絕對路徑納入 payload。
    """

    snapshot: dict[str, tuple[str, object]] = {}
    paths = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
    for path in paths:
        relative = "." if path == root else path.relative_to(root).as_posix()
        node = path.lstat()
        if stat.S_ISREG(node.st_mode):
            snapshot[relative] = ("file", path.read_bytes())
        elif stat.S_ISDIR(node.st_mode):
            snapshot[relative] = ("directory", "")
        elif stat.S_ISLNK(node.st_mode):
            snapshot[relative] = ("symlink", os.readlink(path))
        else:
            snapshot[relative] = ("other", node.st_mode)
    return snapshot


def _assert_payload_failure(operation: Callable[[], object], *, path_marker: Path) -> None:
    """固定 payload failure 的 public error、cause 與 path 隱私契約。"""

    with pytest.raises(ValueError) as error_info:
        operation()
    error = error_info.value
    assert type(error) is ValueError
    assert str(error) == _PAYLOAD_FAILURE
    assert error.__cause__ is None
    assert str(path_marker) not in str(error)
    assert str(path_marker) not in repr(error)


def _build_payload(
    fixture: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: AggregateSpec | None = None,
    source_root: Path | None = None,
    checkpoint_root: Path | None = None,
) -> Any:
    """安裝既有 static loader fake 後呼叫真實 aggregate pipeline API。

    fake loader 只回傳既有 manifest records 與公尺制 geometry snapshot；它不 fake
    ``build_aggregate_release_payload``，也不建立 forcing。每個呼叫都明示使用本檔的
    synthetic engineering data，不能被誤解為正式 OCM/NWW scientific output。
    """

    runtime_static_fixture._patch_static_loaders(monkeypatch, fixture.data)
    return pipeline.build_aggregate_release_payload(
        source_run_root=source_root or fixture.workspace,
        config_path=EXAMPLE_CONFIG,
        aggregate_spec=spec or fixture.spec,
        checkpoint_root=checkpoint_root,
    )


def _assert_release_topology(root: Path) -> None:
    """驗證 public writer 產物是恰 33 個 ordinary files、沒有子目錄或 symlink。"""

    assert root.name == f"aggregate-pipeline-pilot{_FINAL_SUFFIX}"
    entries = list(root.iterdir())
    assert len(entries) == 33
    for entry in entries:
        node = entry.lstat()
        assert stat.S_ISREG(node.st_mode)
        assert not stat.S_ISLNK(node.st_mode)


def _install_iterator_transform(
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[Any], Any],
) -> None:
    """只竄改 iterator 第一個 in-memory record，保留其餘 production reader 行為。"""

    real_iterator = pipeline.iter_complete_run_trajectory_shards

    def transformed_iterator(path: object, *, checkpoint_root: object = None):
        """從真實 complete iterator 取一筆，再回傳測試指定的 binding 變體。"""

        records = real_iterator(path, checkpoint_root=checkpoint_root)
        first = next(records)
        yield transform(first)
        yield from records

    monkeypatch.setattr(pipeline, "iter_complete_run_trajectory_shards", transformed_iterator)


def _extra_site_spec(spec: AggregateSpec, *, extra_site: str = "synthetic-extra-site") -> AggregateSpec:
    """建立只有額外 pathway site、但沒有任何 trajectory result 的合法 typed spec。

    這個 detached spec 用來確認 pipeline 不會把「spec 宣告的 site 沒有輸入 shard」默默
    當成零結果。測試會暫時略過 geometry validator 以孤立 pathway completeness gate；
    AggregateSpec 自身仍經 constructor 驗證。extra site 與 segment 都是 synthetic 名稱，
    不代表真實 OCM/NWW 邊界。
    """

    source_site = next(iter(spec.site_grids))
    extra_segment = "synthetic-extra-open"
    return replace(
        spec,
        site_grids={**spec.site_grids, extra_site: replace(spec.site_grids[source_site])},
        site_metric_crs={
            **spec.site_metric_crs,
            extra_site: replace(spec.site_metric_crs[source_site]),
        },
        boundary_segment_lengths_m={
            **spec.boundary_segment_lengths_m,
            extra_segment: 200.0,
        },
        site_boundary_segment_ids={
            **spec.site_boundary_segment_ids,
            extra_site: SiteBoundarySegments(
                local_segment_ids=(extra_segment,),
                outer_segment_ids=(extra_segment,),
            ),
        },
    )


def test_public_pipeline_api_has_exact_signature_and_export() -> None:
    """公開 API 只暴露 payload builder，且 keyword-only 參數不得漂移。"""

    assert pipeline.__all__ == ["build_aggregate_release_payload"]
    signature = inspect.signature(pipeline.build_aggregate_release_payload)
    assert tuple(signature.parameters) == (
        "source_run_root",
        "config_path",
        "aggregate_spec",
        "checkpoint_root",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert signature.parameters["checkpoint_root"].default is None


def test_build_payload_success_binds_plan_hashes_counts_and_never_touches_forcing(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 pilot payload 應 exact 綁定 source，且 static 前置階段不接觸 forcing。"""

    fixture = aggregate_pipeline_template
    source_before = _snapshot_tree(fixture.workspace)
    call_counts = {"static": 0, "iterator": 0, "factory": 0, "manager": 0}

    runtime_static_fixture._patch_static_loaders(monkeypatch, fixture.data)
    real_static_loader = pipeline.load_validated_run_static_inputs

    def counted_static_loader(*args: object, **kwargs: object) -> object:
        """記錄 static binding 次數，仍委派 production helper。"""

        call_counts["static"] += 1
        return real_static_loader(*args, **kwargs)

    real_iterator = pipeline.iter_complete_run_trajectory_shards

    def counted_iterator(path: object, *, checkpoint_root: object = None):
        """記錄 iterator factory 次數，仍逐筆 yield production record。"""

        call_counts["iterator"] += 1
        yield from real_iterator(path, checkpoint_root=checkpoint_root)

    def blocked_factory(*args: object, **kwargs: object) -> None:
        """若 static binding 建立 request factory，立即暴露責任邊界回歸。"""

        del args, kwargs
        call_counts["factory"] += 1
        raise AssertionError("aggregate static binding 不得建立 RuntimeRequestFactory")

    def blocked_manager(*args: object, **kwargs: object) -> None:
        """若 static binding 建立 forcing manager，立即暴露 OCM/NWW eager load。"""

        del args, kwargs
        call_counts["manager"] += 1
        raise AssertionError("aggregate static binding 不得建立 ForcingWindowManager")

    monkeypatch.setattr(pipeline, "load_validated_run_static_inputs", counted_static_loader)
    monkeypatch.setattr(pipeline, "iter_complete_run_trajectory_shards", counted_iterator)
    monkeypatch.setattr(runtime, "RuntimeRequestFactory", blocked_factory)
    monkeypatch.setattr(
        runtime.ForcingWindowManager,
        "from_roots",
        staticmethod(blocked_manager),
    )

    payload = pipeline.build_aggregate_release_payload(
        source_run_root=fixture.workspace,
        config_path=EXAMPLE_CONFIG,
        aggregate_spec=fixture.spec,
    )

    assert type(payload) is AggregateReleasePayload
    plan = fixture.plan
    assert payload.run_id == plan["run_id"] == "aggregate-pipeline-pilot"
    assert payload.run_kind == plan["run_kind"] == "pilot"
    assert payload.experiment_case_id == plan["experiment_case_id"] == "no_stokes"
    assert payload.members_per_scenario == plan["members_per_scenario"] == 2
    assert payload.config_hash == plan["config_hash"]
    assert payload.checkpoint_input_binding_hash == plan["checkpoint_input_binding_hash"]

    # source JSON hash 是 exact bytes provenance；測試獨立重新計算，不能只比較 pipeline
    # 自己回傳的欄位彼此相等。這四檔是 immutable run input，不包含 forcing array。
    source_hash_fields = {
        "run_plan.json": "source_run_plan_sha256",
        "run_progress.json": "source_run_progress_sha256",
        "normalized_config.json": "source_normalized_config_sha256",
        "input_inventory.json": "source_input_inventory_sha256",
    }
    for filename, field_name in source_hash_fields.items():
        expected_hash = sha256((fixture.workspace / filename).read_bytes()).hexdigest()
        assert getattr(payload, field_name) == expected_hash

    progress = load_run_progress(fixture.workspace)
    plan_row = plan["shards"][0]
    progress_row = progress["shards"][plan_row["shard_id"]]
    output_root = fixture.workspace / progress_row["output_relative_path"]
    manifest_bytes = (output_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    assert len(payload.shard_bindings) == plan["shard_count"] == 1
    binding = payload.shard_bindings[0]
    assert binding.shard_id == plan_row["shard_id"]
    assert binding.scenario_start_index == plan_row["scenario_start_index"] == 0
    assert binding.scenario_stop_index == plan_row["scenario_stop_index"] == 1
    assert binding.output_relative_path == progress_row["output_relative_path"]
    assert binding.trajectory_manifest_sha256 == sha256(manifest_bytes).hexdigest()
    assert binding.particle_count == plan_row["particle_count"] == 2
    assert binding.observation_count == manifest["observation_count"]
    assert binding.event_count == manifest["event_count"]

    expected_strata = scenario_inputs_to_strata(fixture.data["inputs"], formal=False)
    assert payload.scenario_strata == expected_strata
    assert tuple(item.scenario_id for item in payload.scenario_strata) == (
        fixture.data["scenario"].scenario_id,
    )
    assert payload.event_aggregate.input_particle_count == plan["particle_count"] == 2
    assert set(payload.event_aggregate.site_grid_counts) == {fixture.site_id}
    assert set(payload.event_aggregate.outcome_count_by_site) == {fixture.site_id}
    assert payload.event_aggregate.total_member_count_by_site[fixture.site_id] == 2
    assert payload.event_aggregate.valid_member_denominator_by_site[fixture.site_id] == 2
    assert payload.event_aggregate.outcome_count_by_site[fixture.site_id][ParticleStatus.MAX_AGE.value] == 2
    assert sum(payload.event_aggregate.outcome_count_by_site[fixture.site_id].values()) == 2

    assert set(payload.pathway_by_site) == {fixture.site_id}
    pathway = payload.pathway_by_site[fixture.site_id]
    np.testing.assert_array_equal(pathway.x_edges_m, np.array([-100.0, 0.0, 100.0]))
    np.testing.assert_array_equal(pathway.y_edges_m, np.array([-100.0, 0.0, 100.0]))
    np.testing.assert_array_equal(pathway.age_bin_edges_seconds, np.array([0.0, 10.0]))
    assert pathway.input_particle_count == 2
    assert pathway.input_interval_seconds == 14.0
    assert pathway.allocated_interval_seconds == 14.0
    np.testing.assert_array_equal(
        pathway.unique_particle_count,
        np.array([[0, 0], [0, 2]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        pathway.residence_time_seconds,
        np.array([[0.0, 0.0], [0.0, 14.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        pathway.first_passage_age_histogram,
        np.array([[[0], [0]], [[0], [2]]], dtype=np.int64),
    )
    assert call_counts == {"static": 1, "iterator": 1, "factory": 0, "manager": 0}
    assert _snapshot_tree(fixture.workspace) == source_before


def test_payload_can_be_published_and_read_as_synthetic_end_to_end(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """payload 經 public writer／validator／reader 後，九表十八陣列仍 exact。"""

    fixture = aggregate_pipeline_template
    source_before = _snapshot_tree(fixture.workspace)
    payload = _build_payload(fixture, monkeypatch)
    final_path = write_aggregate_release(
        source_run_root=fixture.workspace,
        aggregate_spec_path=fixture.spec_path,
        payload=payload,
    )
    expected_final = fixture.workspace.parent / f"{fixture.workspace.name}{_FINAL_SUFFIX}"
    assert final_path == expected_final
    _assert_release_topology(final_path)
    report = validate_aggregate_release(final_path)
    assert report["valid"] is True
    assert report["errors"] == []
    read_payload = read_aggregate_release(final_path)
    expected_products = encode_aggregate_release_payload(payload)
    actual_products = encode_aggregate_release_payload(read_payload)
    public_validation._assert_encoded_products_exact(expected_products, actual_products)
    assert _snapshot_tree(fixture.workspace) == source_before


def test_payload_preserves_outer_exclusive_run_gate(
    aggregate_pipeline_template: _AggregatePipelineFixture,
) -> None:
    """外層持有 run gate 時，payload builder 應保留原始 RunLockBusyError。"""

    fixture = aggregate_pipeline_template
    lock_path = fixture.workspace / "locks" / "run_gate.lock"
    parent_before = frozenset(item.name for item in fixture.workspace.parent.iterdir())
    with acquire_run_lock(lock_path, mode="exclusive", blocking=False), pytest.raises(
        RunLockBusyError
    ):
        pipeline.build_aggregate_release_payload(
            source_run_root=fixture.workspace,
            config_path=EXAMPLE_CONFIG,
            aggregate_spec=fixture.spec,
        )
    assert frozenset(item.name for item in fixture.workspace.parent.iterdir()) == parent_before


@pytest.mark.parametrize("tampered_field", ("shard_id", "range", "scenarios", "particle_count"))
def test_payload_rejects_iterator_plan_binding_tamper_without_path_leak(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tampered_field: str,
) -> None:
    """shard ID、scenario range、scenario tuple 或 particle count 不符都必須 fail closed。"""

    fixture = aggregate_pipeline_template

    def tamper(record: Any) -> Any:
        """建立單一 record 的 binding 竄改；不修改 source shard bytes。"""

        if tampered_field == "shard_id":
            return replace(record, shard_id="tampered-shard")
        if tampered_field == "range":
            return replace(record, scenario_start_index=1)
        if tampered_field == "scenarios":
            return replace(record, scenarios=())
        return replace(record, particle_count=record.particle_count + 1)

    _install_iterator_transform(monkeypatch, tamper)
    _assert_payload_failure(
        lambda: _build_payload(fixture, monkeypatch),
        path_marker=tmp_path,
    )


def test_payload_rejects_unknown_iterator_result_site(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """trajectory result 若引用未知 site，不得把它靜默歸到任何 pathway。"""

    def tamper(record: Any) -> Any:
        """只改 detached final_state site identity，保留原 record 與 source bytes。"""

        result = record.results[0]
        state = replace(result.final_state, study_site_id="unknown-site")
        altered = replace(result, final_state=state)
        return replace(record, results=(altered, *record.results[1:]))

    _install_iterator_transform(monkeypatch, tamper)
    _assert_payload_failure(
        lambda: _build_payload(aggregate_pipeline_template, monkeypatch),
        path_marker=tmp_path,
    )


def test_payload_rejects_spec_site_without_pathway_result(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """spec 額外宣告沒有輸入 result 的 site 時，pathway completeness gate 必須失敗。"""

    fixture = aggregate_pipeline_template
    extra_spec = _extra_site_spec(fixture.spec)

    def isolated_spec_validator(*args: object, **kwargs: object) -> None:
        """孤立本案例的 pathway-site gate；detached spec 仍已由自身 constructor 驗證。"""

        del args, kwargs

    monkeypatch.setattr(
        pipeline,
        "validate_aggregate_spec_against_boundaries",
        isolated_spec_validator,
    )
    _assert_payload_failure(
        lambda: _build_payload(fixture, monkeypatch, spec=extra_spec),
        path_marker=tmp_path,
    )


def test_payload_rejects_geometry_spec_mismatch_with_fixed_error(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """AggregateSpec 的公尺格網 bounds 偏離 geometry 時應在 pipeline binding 階段拒絕。"""

    fixture = aggregate_pipeline_template
    grid = fixture.spec.site_grids[fixture.site_id]
    tampered_spec = replace(
        fixture.spec,
        site_grids={fixture.site_id: replace(grid, x_max_m=200.0)},
    )
    _assert_payload_failure(
        lambda: _build_payload(fixture, monkeypatch, spec=tampered_spec),
        path_marker=tmp_path,
    )


def test_payload_rejects_incomplete_run_with_exact_error(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """尚未完成的 run 不得進入 trajectory aggregation。"""

    fixture = aggregate_pipeline_template
    base = tmp_path / "incomplete-setup"
    base.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as setup_patch:
        workspace, _ = _initialize_pilot_test_run(
            fixture.data,
            base,
            setup_patch,
            run_id="aggregate-incomplete-pilot",
        )
    spec_path, spec, _ = _write_test_spec(
        fixture.data,
        base / "aggregate-incomplete-spec.json",
        run_id="aggregate-incomplete-pilot",
    )
    del spec_path
    local_fixture = replace(fixture, workspace=workspace.path, spec=spec)
    _assert_payload_failure(
        lambda: _build_payload(local_fixture, monkeypatch),
        path_marker=tmp_path,
    )


def test_payload_rejects_spec_run_id_mismatch(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """spec run_id 不等於 immutable plan run_id 時應固定失敗且不洩漏 workspace。"""

    tampered_spec = replace(
        aggregate_pipeline_template.spec,
        run_id="another-pilot-run",
    )
    _assert_payload_failure(
        lambda: _build_payload(aggregate_pipeline_template, monkeypatch, spec=tampered_spec),
        path_marker=tmp_path,
    )


def test_payload_rechecks_source_hashes_before_returning(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """第二次 source hash 若看到另一個合法 digest，payload 不得回傳 stale snapshot。"""

    real_hash = pipeline._hash_source_files
    calls = 0

    def changing_hash(source_root: Path) -> dict[str, str]:
        """第一次保留真實 hash，第二次只替換一個合法 64 碼 digest。"""

        nonlocal calls
        calls += 1
        result = real_hash(source_root)
        if calls == 2:
            altered = dict(result)
            altered_value = "a" * 64
            if altered_value == altered["run_progress.json"]:
                altered_value = "b" * 64
            altered["run_progress.json"] = altered_value
            return altered
        return result

    monkeypatch.setattr(pipeline, "_hash_source_files", changing_hash)
    _assert_payload_failure(
        lambda: _build_payload(aggregate_pipeline_template, monkeypatch),
        path_marker=tmp_path,
    )
    assert calls == 2


def test_payload_supports_external_checkpoint_root_without_persisting_path(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RunController 與 payload iterator 可使用 external checkpoint，結果不保存其絕對路徑。"""

    fixture = aggregate_pipeline_template
    base = tmp_path / "external-checkpoint-setup"
    external_root = tmp_path / "external-checkpoints"
    workspace = _initialize_run(
        fixture.data,
        base,
        run_id="aggregate-external-pilot",
        checkpoint_root=external_root,
    )
    spec_path, spec, _ = _write_test_spec(
        fixture.data,
        base / "aggregate-external-spec.json",
        run_id="aggregate-external-pilot",
    )
    del spec_path
    local_fixture = replace(fixture, workspace=workspace, spec=spec)
    payload = _build_payload(
        local_fixture,
        monkeypatch,
        checkpoint_root=external_root,
    )
    assert str(external_root) not in repr(payload)
    assert payload.run_id == "aggregate-external-pilot"

    mismatched_spec = replace(spec, run_id="aggregate-external-other")
    _assert_payload_failure(
        lambda: _build_payload(
            local_fixture,
            monkeypatch,
            spec=mismatched_spec,
            checkpoint_root=external_root,
        ),
        path_marker=external_root,
    )


def test_payload_uses_single_pass_fixed_memory_reducers(
    aggregate_pipeline_template: _AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一個 shard 應逐 chunk add、各 reducer 只 finalize 一次，且不呼叫 pairwise merge。"""

    fixture = aggregate_pipeline_template
    event_calls: list[str] = []
    pathway_calls: list[str] = []
    real_event_accumulator = pipeline.EventAggregateAccumulator
    real_pathway_accumulator = pipeline.StreamingPathwayAccumulator

    class RecordingEventAccumulator(real_event_accumulator):
        """只記錄事件 reducer 的生命週期，實際運算仍由 production class 執行。"""

        def add(self, chunk: object) -> None:
            """記錄目前 shard 的事件 chunk 已被當下吸收。"""

            event_calls.append("add")
            return super().add(chunk)  # type: ignore[arg-type]

        def finalize(self) -> object:
            """記錄 event reducer 的唯一 finalize，仍回傳 production aggregate。"""

            event_calls.append("finalize")
            return super().finalize()

    class RecordingPathwayAccumulator(real_pathway_accumulator):
        """只記錄 pathway reducer 的生命週期，實際運算仍由 production class 執行。"""

        def add(self, chunk: object) -> None:
            """記錄目前站點 pathway chunk 已被當下吸收。"""

            pathway_calls.append("add")
            return super().add(chunk)  # type: ignore[arg-type]

        def finalize(self) -> object:
            """記錄 pathway reducer 的唯一 finalize，仍回傳 production aggregate。"""

            pathway_calls.append("finalize")
            return super().finalize()

    monkeypatch.setattr(pipeline, "EventAggregateAccumulator", RecordingEventAccumulator)
    monkeypatch.setattr(pipeline, "StreamingPathwayAccumulator", RecordingPathwayAccumulator)
    payload = _build_payload(fixture, monkeypatch)
    assert payload.run_id == fixture.plan["run_id"]
    assert event_calls == ["add", "finalize"]
    assert pathway_calls == ["add", "finalize"]

    source_text = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "tuple(iter_complete_run_trajectory_shards" not in source_text
    assert "list(iter_complete_run_trajectory_shards" not in source_text
    assert "merge_event_aggregate_chunks" not in source_text
    assert "merge_streaming_pathway_aggregates" not in source_text
