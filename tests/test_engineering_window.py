"""單站工程性回溯時窗的契約與小型端到端測試。

測試沿用 ``test_input_derivation`` 的 synthetic OCM／OCM surface／NWW3 fixture，先由
既有輸入衍生器建立可載入的三套 component，再交給工程時窗 adapter 重新產生單一
站點、單一 exact-hour arrival、單一材質的 generated artifact。這些資料只用來驗證
loader、來源追溯、時間窗與 run-control binding，不代表正式研究結果或 SERVER 產品。
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import test_input_derivation as input_fixture_module
import yaml

import lagrangian_backtracking.runtime as runtime
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.engineering_window import (
    EngineeringWindowError,
    build_engineering_horizon,
    parse_arrival_utc,
    prepare_engineering_window,
    run_engineering_window,
)
from lagrangian_backtracking.input_derivation import (
    build_input_derivatives,
    create_release_config,
    read_canonical_json,
)
from lagrangian_backtracking.input_horizon import (
    UTC_HOUR_NS,
    HorizonWindow,
    compute_horizon_coverage,
)
from lagrangian_backtracking.manifests import load_scenario_inputs
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.models import SampleQC, VelocitySample

# 讓本檔可直接重用既有 synthetic_support_input_fixture；該 fixture 仍由 pytest 管理
# tmp_path 生命週期，避免在這裡複製一套會與輸入衍生測試漂移的大型資料建立器。
pytest_plugins = ("test_input_derivation",)


_GUISHAN_ARRIVAL_UTC = "2024-02-15T00:00:00Z"
_GUISHAN_MATERIAL_ID = "oca_fishinggear_open_mesh_bundle"


def test_engineering_horizon_supports_common_7_30_60_day_windows() -> None:
    """7、30、60 日都應建立 exact-hour inclusive window，而非只支援固定 H30。"""

    expected_start = {
        7: "2024-02-08T00:00:00Z",
        30: "2024-01-16T00:00:00Z",
        60: "2023-12-17T00:00:00Z",
    }
    expected_months = {
        7: ("202402",),
        30: ("202401", "202402"),
        60: ("202312", "202401", "202402"),
    }
    for days in (7, 30, 60):
        horizon = build_engineering_horizon(_GUISHAN_ARRIVAL_UTC, days)
        assert horizon.support_days == days
        assert horizon.expected_step_count == days * 24 + 1
        assert horizon.start_utc == expected_start[days]
        assert horizon.arrival_utc == _GUISHAN_ARRIVAL_UTC
        assert horizon.months == expected_months[days]


@pytest.mark.parametrize(
    "arrival_utc",
    [
        "2024-02-15T00:30:00Z",
        "2024-02-15T00:00:00+00:00",
        " 2024-02-15T00:00:00Z",
        "2024-02-15T00:00:00Z ",
        "2024-02-15T00:00:00",
        "2024-02-15T00:00:00+08:00Z",
    ],
)
def test_engineering_horizon_rejects_non_exact_or_noncanonical_utc(arrival_utc: str) -> None:
    """arrival 必須是沒有空白、無 offset 且落在整點的 UTC ``Z`` 字串。"""

    with pytest.raises(EngineeringWindowError):
        parse_arrival_utc(arrival_utc)


def test_engineering_horizon_rejects_nonpositive_or_boolean_days() -> None:
    """回溯日數不可由 bool 冒充整數，也不可為零或負數。"""

    for days in (0, -1, True, 1.0):
        with pytest.raises(EngineeringWindowError):
            build_engineering_horizon(_GUISHAN_ARRIVAL_UTC, days)  # type: ignore[arg-type]


def test_engineering_horizon_rejects_one_missing_hour() -> None:
    """H30 只要跨過一個 canonical gap，就必須 fail-closed 而不可以零值補齊。"""

    horizon = build_engineering_horizon(_GUISHAN_ARRIVAL_UTC, 30)
    period_start = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()) * 1_000_000_000
    period_end = int(datetime(2024, 2, 29, 23, tzinfo=UTC).timestamp()) * 1_000_000_000
    missing = int(datetime(2024, 1, 31, tzinfo=UTC).timestamp()) * 1_000_000_000
    product = SimpleNamespace(
        canonical=SimpleNamespace(
            time_utc_ns=np.asarray([period_start, period_end], dtype=np.int64),
            gaps=(
                {
                    "before_utc_ns": missing - UTC_HOUR_NS,
                    "after_utc_ns": missing + UTC_HOUR_NS,
                    "missing_step_count": 1,
                    "gap_hours": 2.0,
                },
            ),
        )
    )

    # 先直接驗證共同 horizon helper 的 missing 節點，保留測試對資料缺口語意的明示。
    window = build_engineering_horizon(_GUISHAN_ARRIVAL_UTC, 30)
    coverage = compute_horizon_coverage(
        HorizonWindow(
            arrival_time_ns=window.arrival_time_ns,
            start_time_ns=window.start_time_ns,
            end_time_ns=window.arrival_time_ns,
            support_days=window.support_days,
            expected_step_count=window.expected_step_count,
        ),
        expected_start_ns=period_start,
        expected_end_ns=period_end,
        canonical_start_ns=period_start,
        canonical_end_ns=period_end,
        canonical_gaps=product.canonical.gaps,
    )
    assert coverage.crossed_gap is True
    assert coverage.missing_time_ns == (missing,)

    from lagrangian_backtracking.engineering_window import _assert_horizon_coverage

    with pytest.raises(EngineeringWindowError, match="H 時窗缺節點"):
        _assert_horizon_coverage(product, horizon, product_label="synthetic_ocm")


@pytest.fixture(scope="module")
def guishan_engineering_artifact(tmp_path_factory: pytest.TempPathFactory):
    """建立一次可重用的龜山島 H30 near-bed artifact。

    既有輸入測試的 fixture 是 function scope，這裡透過其原始函式配合 module scope 的
    temporary root 重用同一套 synthetic source；如此整個檔案只建立一次 30 日母體與
    derived component。來源仍經過真正 ``build_input_derivatives``、release config 與
    adapter loader，沒有以手寫 JSON 或全面 mock 取代資料契約。
    """

    root = tmp_path_factory.mktemp("engineering-window")
    source_tuple = input_fixture_module.synthetic_input_fixture.__wrapped__(root)
    source_tuple = input_fixture_module.synthetic_support_input_fixture.__wrapped__(source_tuple)
    source_config_template, ocm_root, surface_root, nww_root, source_root = source_tuple
    source_payload = yaml.safe_load(source_config_template.read_text(encoding="utf-8"))
    assert isinstance(source_payload, dict)
    scenarios = source_payload["scenarios"]
    execution = source_payload["execution"]
    physics = source_payload["physics"]
    assert isinstance(scenarios, dict) and isinstance(execution, dict)
    assert isinstance(physics, dict)
    horizontal_diffusion = physics["horizontal_diffusion"]
    vertical_diffusion = physics["vertical_diffusion"]
    assert isinstance(horizontal_diffusion, dict) and isinstance(vertical_diffusion, dict)
    # example config 對正式校準刻意保留 null；run smoke 仍需讓真實
    # RuntimeRequestFactory 建立有限係數。設為零代表此測試只驗證
    # workspace/checkpoint 綁定，不把解析 forcing provider 假裝成擴散驗證。
    horizontal_diffusion["constant_kh_m2ps"] = 0.0
    vertical_diffusion["constant_kz_m2ps"] = 0.0
    # engineering run 只測 M=1 的單站小 bundle；這些執行欄位是測試 fixture 的
    # deterministic run-control 綁定，不能把 source 設計中的正式 50,000 情境數帶入。
    scenarios["members_per_scenario"] = 1
    scenarios["master_seed"] = 20260911
    execution.update(
        {
            "shard_scenario_count": 5,
            "checkpoint_interval_sweeps": 1,
            "active_chunk_size": 1,
        }
    )
    source_config_template.write_text(
        yaml.safe_dump(source_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    input_directory = source_root / "derived-inputs"
    build_input_derivatives(
        config_path=source_config_template,
        destination=input_directory,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    source_config = source_root / "source-release.yaml"
    create_release_config(
        config_template_path=source_config_template,
        input_directory=input_directory,
        output_path=source_config,
        formal=False,
        max_backtrack_days=30,
        maximum_step_count=8_640,
    )
    artifact_directory = source_root / "guishan-engineering"
    artifact = prepare_engineering_window(
        source_config=source_config,
        source_input_directory=input_directory,
        study_site_id="guishan",
        arrival_utc=_GUISHAN_ARRIVAL_UTC,
        backtrack_days=30,
        material_id=_GUISHAN_MATERIAL_ID,
        destination=artifact_directory,
        ocm_native_root=ocm_root,
        nww_analysis_root=nww_root,
        project_root=Path(__file__).resolve().parents[1],
        vertical_id="near_bed",
        shard_scenario_count=5,
    )
    return {
        "artifact": artifact,
        "source_config": source_config,
        "input_directory": input_directory,
        "ocm_root": ocm_root,
        "nww_root": nww_root,
        "root": source_root,
    }


def test_prepare_engineering_window_filters_five_near_bed_pairs_and_loader_reads_new_bundle(
    guishan_engineering_artifact: dict[str, Any],
) -> None:
    """prepare 應只保留龜山島五個既有 near-bed XY，且新 manifest 可由正式 loader 讀取。"""

    artifact = guishan_engineering_artifact["artifact"]
    assert artifact.study_site_id == "guishan"
    assert artifact.vertical_id == "near_bed"
    assert (artifact.receptor_count, artifact.pair_count, artifact.scenario_count) == (5, 5, 5)

    config = load_config(artifact.config_path, formal_release=False)
    scenario_inputs = load_scenario_inputs(
        config,
        config_path=artifact.config_path,
        require_dynamic_initial_conditions=True,
        formal=False,
    )
    assert len(scenario_inputs.receptors) == 5
    assert len(scenario_inputs.initial_conditions) == 5
    assert len(scenario_inputs.scenarios) == 5
    assert {item.vertical_id for item in scenario_inputs.receptors} == {"near_bed"}
    assert {item.material_id for item in scenario_inputs.scenarios} == {_GUISHAN_MATERIAL_ID}
    assert all(item.height_above_bed_m > 0.0 for item in scenario_inputs.initial_conditions)

    manifest, _ = read_canonical_json(artifact.manifest_path)
    assert manifest["engineering_only"] is True
    assert manifest["horizon"]["expected_step_count"] == 721
    assert manifest["horizon"]["coverage"][0]["missing_utc"] == []
    assert manifest["horizon"]["nww_endpoint_support"]["arrival_endpoint_only"] is True
    assert manifest["source_lineage"]["old_input_release_reused"] is False
    assert manifest["source_lineage"]["old_pilot_execution_binding_reused"] is False


def test_engineering_prepare_rejects_source_component_lineage_mismatch(
    guishan_engineering_artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """source config 若把 component reference 指到別處，必須在 loader 前拒絕錯綁。"""

    source_config = Path(guishan_engineering_artifact["source_config"])
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    scenarios = payload.get("scenarios")
    assert isinstance(scenarios, dict)
    scenarios["receptor_manifest"] = "derived-inputs/other-receptor.json"
    mismatched_config = tmp_path / "mismatched-source.yaml"
    mismatched_config.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(EngineeringWindowError, match="reference 未指向指定 source directory"):
        prepare_engineering_window(
            source_config=mismatched_config,
            source_input_directory=guishan_engineering_artifact["input_directory"],
            study_site_id="guishan",
            arrival_utc=_GUISHAN_ARRIVAL_UTC,
            backtrack_days=30,
            material_id=_GUISHAN_MATERIAL_ID,
            destination=tmp_path / "should-not-publish",
            ocm_native_root=guishan_engineering_artifact["ocm_root"],
            nww_analysis_root=guishan_engineering_artifact["nww_root"],
            project_root=Path(__file__).resolve().parents[1],
            vertical_id="near_bed",
        )


def test_engineering_artifact_tamper_is_rejected_and_formal_loader_is_blocked(
    guishan_engineering_artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """generated component bytes 竄改要被拒絕，且 engineering config 不得走 formal loader。"""

    original = Path(guishan_engineering_artifact["artifact"].directory)
    tampered = tmp_path / "tampered-engineering"
    shutil.copytree(original, tampered)
    receptor_path = tampered / "manifests" / "receptor.json"
    receptor_payload = json.loads(receptor_path.read_text(encoding="utf-8"))
    receptor_payload["records"][0]["metadata"]["tampered_for_test"] = True
    receptor_path.write_text(
        json.dumps(receptor_payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EngineeringWindowError, match="無法讀取 JSON|component hash 不一致"):
        run_engineering_window(
            artifact=tampered,
            destination=tmp_path / "runs",
            run_id="tampered-run",
            ocm_native_root=guishan_engineering_artifact["ocm_root"],
            nww_analysis_root=guishan_engineering_artifact["nww_root"],
            project_root=Path(__file__).resolve().parents[1],
        )

    with pytest.raises(ValueError, match="正式發布設定未通過"):
        load_config(original / "config.yaml", formal_release=True)


class _AnalyticVelocityProvider:
    """提供固定、有限且不觸碰 forcing array 的 reference 速度樣本。"""

    step_start_sample_reuse_safe = True

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳零水平／零垂向速度，讓測試聚焦 run-control checkpoint binding。"""

        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(
            u_mps=0.0,
            v_mps=0.0,
            w_mps=0.0,
            eta_m=0.0,
            bed_z_m=-20.0,
            horizontal_scale_m=100_000.0,
            vertical_scale_m=20.0,
            qc=SampleQC.OK,
        )


class _AnalyticManager:
    """以真實 NativeMesh 定位 receptor、以解析速度取代月份取樣的最小 manager。"""

    def __init__(self, flow_domain_id: str, projection: object, ocm_root: Path) -> None:
        """讀取 artifact 對應的 static mesh；不讀取 OCM 月份 forcing。"""

        self.flow_domain_id = flow_domain_id
        self.mesh = NativeMesh.from_directory(
            ocm_root / flow_domain_id / "grid",
            projection=projection,
        )
        self.cache_stats = SimpleNamespace(
            ocm_load_count=0,
            nww_load_count=0,
            ocm_cache_hit_count=0,
            ocm_cache_miss_count=0,
            nww_cache_hit_count=0,
            nww_cache_miss_count=0,
            eviction_count=0,
            resident_ndarray_bytes=0,
            cache_hit_count=0,
            cache_miss_count=0,
        )

    def provider(self, settling_velocity_mps: float, include_stokes: bool) -> _AnalyticVelocityProvider:
        """建立與真實 manager 相同呼叫形狀的 reference provider。"""

        del settling_velocity_mps, include_stokes
        return _AnalyticVelocityProvider()


def test_engineering_run_pause_resume_keeps_plan_and_checkpoint_binding(
    guishan_engineering_artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """實際建立 workspace、checkpoint 並 resume；改動 workspace inventory 應拒絕續跑。"""

    managers: list[_AnalyticManager] = []

    def from_roots(**kwargs: Any) -> _AnalyticManager:
        """保留 factory 的真實 root／flow／projection 參數並只替換物理取樣器。"""

        manager = _AnalyticManager(
            str(kwargs["flow_domain_id"]),
            kwargs["projection"],
            Path(kwargs["ocm_root"]),
        )
        managers.append(manager)
        return manager

    monkeypatch.setattr(runtime.ForcingWindowManager, "from_roots", staticmethod(from_roots))
    run_root = tmp_path / "runs"
    checkpoint_root = tmp_path / "checkpoints"
    common_kwargs = {
        "artifact": guishan_engineering_artifact["artifact"].directory,
        "destination": run_root,
        "run_id": "guishan-engineering-smoke",
        "checkpoint_root": checkpoint_root,
        "ocm_native_root": guishan_engineering_artifact["ocm_root"],
        "nww_analysis_root": guishan_engineering_artifact["nww_root"],
        "project_root": Path(__file__).resolve().parents[1],
        "shard_ids": ("0",),
        "sweep_budget": 1,
    }

    first = run_engineering_window(**common_kwargs)
    assert len(first) == 1
    assert first[0].lifecycle == "PAUSED"
    workspace = run_root / "guishan-engineering-smoke"
    plan = json.loads((workspace / "run_plan.json").read_text(encoding="utf-8"))
    progress = json.loads((workspace / "run_progress.json").read_text(encoding="utf-8"))
    assert plan["run_kind"] == "pilot"
    assert plan["scenario_count"] == 5
    assert progress["shards"][first[0].shard_id]["lifecycle"] == "PAUSED"
    assert (workspace / "normalized_config.json").is_file()
    assert managers, "真實 RuntimeRequestFactory 應建立一個 flow manager"

    resumed_kwargs = dict(common_kwargs)
    resumed_kwargs["resume"] = True
    resumed = run_engineering_window(**resumed_kwargs)
    assert len(resumed) == 1
    assert resumed[0].lifecycle == "PAUSED"
    progress_after_resume = json.loads((workspace / "run_progress.json").read_text(encoding="utf-8"))
    row = progress_after_resume["shards"][first[0].shard_id]
    assert row["attempt_count"] == 2
    assert row["checkpoint_sequence"] == 2
    assert row["sweeps_completed"] == 2

    inventory_path = workspace / "input_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["tampered_for_test"] = True
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EngineeringWindowError, match=r"input_inventory\.json: (size|checksum)"):
        run_engineering_window(**resumed_kwargs)
