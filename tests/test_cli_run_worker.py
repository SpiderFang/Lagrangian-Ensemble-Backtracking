"""``run-worker`` 的單程序多 shard orchestration 測試。

測試同時覆蓋 CLI 的批次邊界與既有 run-control 行為：所有 shard ID 必須在第一片前
完成驗證、同一個 controller 依命令列順序重用、PAUSED／例外／鎖衝突會停止後續片，
並輸出可供排程器讀取的摘要與單程序時間。合成 run 只驗證工程連接，不代表 OCM／NWW
科學結果或 SERVER 整機效能。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_run_control import _payload_snapshot, _request, _workspace

import lagrangian_backtracking.cli as cli
from lagrangian_backtracking.cli import main
from lagrangian_backtracking.run_control import RunController, RunExecutionSummary, load_run_plan
from lagrangian_backtracking.run_locking import RunLockBusyError


def _worker_argv(workspace: Path, config: Path, shard_ids: list[str]) -> list[str]:
    """建立所有測試共用的 run-worker 參數，保留指定 shard 的順序。"""

    argv = [
        "run-worker",
        str(workspace),
        "--config",
        str(config),
        "--ocm-native-root",
        str(workspace / "ocm"),
    ]
    for shard_id in shard_ids:
        argv.extend(("--shard-id", shard_id))
    return argv


def _fake_config() -> Any:
    """提供 CLI root 解析所需的最小 config.inputs 物件，不讀取 forcing。"""

    return SimpleNamespace(
        inputs=SimpleNamespace(
            ocm_native_root_env="LBT_TEST_RUN_WORKER_OCM",
            nww_analysis_root_env="LBT_TEST_RUN_WORKER_NWW",
        )
    )


def _patch_worker_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workspace: Path,
    controller: Any,
    open_calls: list[tuple[Path, dict[str, object]]],
    config_calls: list[dict[str, object]] | None = None,
) -> None:
    """隔離 CLI 的 pre-validation/config/open 邊界，保留真實 controller 執行。"""

    config = _fake_config()

    def fake_validate_run(
        path: Path,
        *,
        require_complete: bool,
        checkpoint_root: Path | None,
    ) -> dict[str, object]:
        """讓測試專注在 worker orchestration，不重複讀取完整 validator。"""

        assert path == workspace
        assert require_complete is False
        assert checkpoint_root is None
        return {"valid": True}

    def fake_load_config(path: Path, *, formal_release: bool) -> Any:
        """記錄同 run config 模式，模擬既有 run-shard 的 loader 邊界。"""

        if config_calls is not None:
            config_calls.append({"path": path, "formal_release": formal_release})
        return config

    def fake_open_run_controller(workspace_arg: Path, **kwargs: object) -> Any:
        """記錄 controller 開啟次數，回傳同一個真實或行為替身。"""

        open_calls.append((workspace_arg, kwargs))
        return controller

    monkeypatch.setattr(cli, "validate_run", fake_validate_run)
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "open_run_controller", fake_open_run_controller)


def test_run_worker_opens_one_controller_and_reuses_real_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """真實小型 controller 依指定順序完成兩片，且 request factory 跨片持續使用。"""

    workspace = _workspace(tmp_path, "run-worker-reuse", run_kind="pilot", interval=1)
    config_path = tmp_path / "pilot-config.yaml"
    plan = load_run_plan(workspace)
    shard_ids = [str(row["shard_id"]) for row in plan["shards"]]
    requested = list(reversed(shard_ids))
    factory_calls: list[str] = []

    def counted_factory(unit: Any) -> Any:
        """記錄兩片建立的 particle request，確認共享同一個 factory 生命週期。"""

        factory_calls.append(unit.particle_id)
        return _request(unit)

    controller = RunController(workspace, request_factory=counted_factory)
    open_calls: list[tuple[Path, dict[str, object]]] = []
    config_calls: list[dict[str, object]] = []
    _patch_worker_preflight(
        monkeypatch,
        workspace=workspace,
        controller=controller,
        open_calls=open_calls,
        config_calls=config_calls,
    )

    assert main(_worker_argv(workspace, config_path, requested)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(open_calls) == 1
    assert open_calls[0][0] == workspace
    assert [item["shard_id"] for item in payload["executed_shards"]] == requested
    assert all(item["lifecycle"] == "COMPLETE" for item in payload["executed_shards"])
    assert payload["requested_shard_ids"] == requested
    assert payload["stopped_after_paused_shard_id"] is None
    assert payload["schema_version"] == "1.0.0"
    assert payload["artifact_type"] == "run_worker_execution_summary"
    assert payload["timing"]["includes_python_import"] is False
    assert payload["timing"]["includes_python_startup"] is False
    assert payload["timing"]["preflight_wall_seconds"] <= payload["timing"]["command_wall_seconds"]
    assert payload["timing"]["preflight_process_cpu_seconds"] <= payload["timing"][
        "command_process_cpu_seconds"
    ]
    # 每片一個 scenario、兩個 members；兩片合計四個 request，證明第二片沒有另開 controller。
    assert len(factory_calls) == 4
    assert config_calls == [{"path": config_path, "formal_release": False}]


def test_run_worker_pause_then_new_controller_resume_matches_uninterrupted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PAUSED 後由新 controller resume，結果須與不中斷的同片執行逐值相同。"""

    paused_workspace = _workspace(tmp_path, "run-worker-resume-paused", run_kind="pilot", interval=1)
    baseline_workspace = _workspace(tmp_path, "run-worker-resume-baseline", run_kind="pilot", interval=1)
    config_path = tmp_path / "pilot-config.yaml"
    shard_id = str(load_run_plan(paused_workspace)["shards"][0]["shard_id"])
    controller_calls: list[dict[str, object]] = []
    controllers: list[RunController] = []

    def fake_open_run_controller(workspace_arg: Path, **kwargs: object) -> RunController:
        """每次 CLI invocation 建立新 controller，保留 resume 旗標供驗收。"""

        assert workspace_arg == paused_workspace
        controller_calls.append(dict(kwargs))
        controller = RunController(
            paused_workspace,
            request_factory=_request,
            resume=bool(kwargs["resume"]),
            checkpoint_root=kwargs.get("checkpoint_root"),
        )
        controllers.append(controller)
        return controller

    open_calls: list[tuple[Path, dict[str, object]]] = []
    _patch_worker_preflight(
        monkeypatch,
        workspace=paused_workspace,
        controller=object(),
        open_calls=open_calls,
    )
    # 覆寫 helper 的固定 controller，只讓本案例檢查兩次 invocation 的新物件與 resume。
    monkeypatch.setattr(cli, "open_run_controller", fake_open_run_controller)

    first = _worker_argv(paused_workspace, config_path, [shard_id]) + ["--sweep-budget", "1"]
    assert main(first) == 0
    first_payload = json.loads(capsys.readouterr().out)
    assert first_payload["executed_shards"][0]["lifecycle"] == "PAUSED"

    second = _worker_argv(paused_workspace, config_path, [shard_id]) + ["--resume"]
    assert main(second) == 0
    second_payload = json.loads(capsys.readouterr().out)
    assert second_payload["executed_shards"][0]["lifecycle"] == "COMPLETE"
    assert len(controllers) == 2
    assert controllers[0] is not controllers[1]
    assert [bool(call["resume"]) for call in controller_calls] == [False, True]

    # 同一個 synthetic scenario／seed 的 uninterrupted baseline 用來確認 checkpoint restore
    # 沒有改變 trajectory、event 或 status arrays；run ID 與 runtime metadata 不納入此比較。
    baseline_shard_id = str(load_run_plan(baseline_workspace)["shards"][0]["shard_id"])
    RunController(baseline_workspace, request_factory=_request).run_shard(baseline_shard_id)
    assert _payload_snapshot(paused_workspace, shard_id) == _payload_snapshot(
        baseline_workspace, baseline_shard_id
    )


@pytest.mark.parametrize("bad_kind", ["missing", "duplicate"])
def test_run_worker_validates_all_ids_before_first_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_kind: str,
) -> None:
    """後續 ID 缺失或重複時，不得啟動第一片、載入 config 或開啟 controller。"""

    workspace = _workspace(tmp_path, f"run-worker-{bad_kind}", run_kind="pilot")
    config_path = tmp_path / "pilot-config.yaml"
    shard_ids = [str(row["shard_id"]) for row in load_run_plan(workspace)["shards"]]
    requested = (
        [shard_ids[0], "missing-shard-id"]
        if bad_kind == "missing"
        else [shard_ids[0], shard_ids[0]]
    )
    controller_calls: list[str] = []
    open_calls: list[tuple[Path, dict[str, object]]] = []
    config_calls: list[dict[str, object]] = []
    controller = SimpleNamespace(
        run_shard=lambda shard_id, *, sweep_budget=None: controller_calls.append(shard_id)
    )
    _patch_worker_preflight(
        monkeypatch,
        workspace=workspace,
        controller=controller,
        open_calls=open_calls,
        config_calls=config_calls,
    )

    with pytest.raises(ValueError, match="shard"):
        main(_worker_argv(workspace, config_path, requested))
    assert controller_calls == []
    assert open_calls == []
    assert config_calls == []


def test_run_worker_stops_after_paused_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """第一片 PAUSED 時只輸出已執行摘要，後續片不會被呼叫。"""

    workspace = _workspace(tmp_path, "run-worker-paused", run_kind="pilot")
    config_path = tmp_path / "pilot-config.yaml"
    shard_ids = [str(row["shard_id"]) for row in load_run_plan(workspace)["shards"]]
    calls: list[tuple[str, int | None]] = []

    def run_shard(shard_id: str, *, sweep_budget: int | None = None) -> RunExecutionSummary:
        """第一片回傳 pause，若第二片被錯誤呼叫則測試會因明確 assertion 失敗。"""

        calls.append((shard_id, sweep_budget))
        assert len(calls) == 1
        return RunExecutionSummary(
            run_id="run-worker-paused",
            shard_id=shard_id,
            lifecycle="PAUSED",
            scenario_count=1,
            particle_count=2,
            sweeps_completed=1,
            particle_steps=2,
            output_relative_path=None,
            checkpoint_relative_path="run-worker-paused/checkpoint-00000001",
        )

    controller = SimpleNamespace(run_shard=run_shard)
    open_calls: list[tuple[Path, dict[str, object]]] = []
    _patch_worker_preflight(
        monkeypatch,
        workspace=workspace,
        controller=controller,
        open_calls=open_calls,
    )

    assert main(_worker_argv(workspace, config_path, shard_ids) + ["--sweep-budget", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == [(shard_ids[0], 1)]
    assert len(open_calls) == 1
    assert len(payload["executed_shards"]) == 1
    assert payload["stopped_after_paused_shard_id"] == shard_ids[0]


@pytest.mark.parametrize("error", [RuntimeError("worker failure"), RunLockBusyError("worker busy")])
def test_run_worker_propagates_exception_and_stops_following_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """物理例外與 lock contention 都直接傳出，第二片不得被跳過錯誤繼續執行。"""

    workspace = _workspace(tmp_path, "run-worker-error", run_kind="pilot")
    config_path = tmp_path / "pilot-config.yaml"
    shard_ids = [str(row["shard_id"]) for row in load_run_plan(workspace)["shards"]]
    calls: list[str] = []

    def run_shard(shard_id: str, *, sweep_budget: int | None = None) -> RunExecutionSummary:
        """第一片固定拋出指定例外，模擬 controller 的既有 fail-fast 行為。"""

        del sweep_budget
        calls.append(shard_id)
        raise error

    controller = SimpleNamespace(run_shard=run_shard)
    open_calls: list[tuple[Path, dict[str, object]]] = []
    _patch_worker_preflight(
        monkeypatch,
        workspace=workspace,
        controller=controller,
        open_calls=open_calls,
    )

    with pytest.raises(type(error), match=str(error)):
        main(_worker_argv(workspace, config_path, shard_ids))
    assert calls == [shard_ids[0]]
    assert len(open_calls) == 1
