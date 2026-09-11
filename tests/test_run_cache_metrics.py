"""RunController forcing cache 計數、gauge、續跑與 legacy 語意測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_run_control import _first_shard, _request, _workspace

from lagrangian_backtracking.production import ProductionBatch
from lagrangian_backtracking.run_control import (
    RunController,
    _merge_invocation_forcing_snapshot,
    _merge_metrics,
    load_run_plan,
    load_run_progress,
)
from lagrangian_backtracking.run_validation import benchmark_report

_CACHE_KEYS = ("loads", "hits", "misses", "evictions", "manager_count", "resident_bytes")


def _cache_stats(**overrides: int) -> dict[str, int]:
    """建立完整 reporter snapshot；欄位單位是事件次數、manager 數或 resident bytes。"""

    values = {key: 0 for key in _CACHE_KEYS}
    values.update(overrides)
    return values


def _write_progress(workspace: Path, progress: dict) -> None:
    """僅供測試模擬既有 progress 的合法可恢復狀態。"""

    (workspace / "run_progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_shared_controller_cache_counters_are_invocation_deltas_and_gauges_are_max(
    tmp_path: Path,
) -> None:
    """同一 controller 執行兩片時，累計 loads 不得被第二片重複相加。"""

    workspace = _workspace(tmp_path, "shared-cache-metrics", run_kind="pilot", interval=1)
    counters = {key: 0 for key in _CACHE_KEYS}

    def counted_request(unit):
        """模擬每個 request 建構觸發一次 forcing load 與 gauge 增長。"""

        counters["loads"] += 1
        counters["manager_count"] = counters["loads"]
        counters["resident_bytes"] = counters["loads"] * 10
        return _request(unit)

    controller = RunController(
        workspace,
        request_factory=counted_request,
        resource_reporter=lambda: dict(counters),
    )
    controller.run_all()

    progress = load_run_progress(workspace)
    rows = [row["metrics"] for row in progress["shards"].values()]
    assert counters["loads"] == 4
    assert [row["forcing_cache_stats"]["loads"] for row in rows] == [2, 2]
    assert all(
        row["forcing_cache_stats_semantics"] == "invocation_delta_v1" for row in rows
    )
    report = benchmark_report(workspace)
    summary = report["summary"]
    assert summary["forcing_cache_stats"] == {
        "loads": 4,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "manager_count": 4,
        "resident_bytes": 40,
    }
    assert summary["forcing_cache_stats_precise"] is True


def test_pause_resume_same_and_new_controller_accumulates_once(tmp_path: Path) -> None:
    """同 controller 續跑與新 controller 接續都只加入各自 invocation delta。"""

    workspace = _workspace(tmp_path, "resume-cache-metrics", run_kind="pilot", interval=1)
    counters = {key: 0 for key in _CACHE_KEYS}

    def counted_request(unit):
        """為每個 request 增加一次 loads。"""

        counters["loads"] += 1
        counters["manager_count"] = counters["loads"]
        counters["resident_bytes"] = counters["loads"] * 10
        return _request(unit)

    def reporter():
        """回傳共用 controller 當下的完整 cache snapshot。"""

        return dict(counters)

    controller = RunController(
        workspace,
        request_factory=counted_request,
        resource_reporter=reporter,
        resume=True,
    )
    shard_id = _first_shard(workspace)
    assert controller.run_shard(shard_id, sweep_budget=1).lifecycle == "PAUSED"
    assert load_run_progress(workspace)["shards"][shard_id]["metrics"]["forcing_cache_stats"]["loads"] == 2
    controller.run_shard(shard_id)
    first_row = load_run_progress(workspace)["shards"][shard_id]
    assert first_row["metrics"]["forcing_cache_stats"]["loads"] == 4

    # 第二片交由新 controller 接續，驗證 shared manager 的 baseline 不會把前片 4 次重算。
    second = next(
        row["shard_id"]
        for row in load_run_plan(workspace)["shards"]
        if row["shard_id"] != shard_id
    )
    RunController(
        workspace,
        request_factory=counted_request,
        resource_reporter=reporter,
        resume=True,
    ).run_shard(second)
    report = benchmark_report(workspace)
    assert report["summary"]["forcing_cache_stats"]["loads"] == 6
    assert report["summary"]["forcing_cache_stats_precise"] is True


def test_pause_resume_new_controller_with_reset_counters_keeps_saved_delta(
    tmp_path: Path,
) -> None:
    """新 controller 的 process counter 重置後，仍只加入新 invocation 的 loads。"""

    workspace = _workspace(tmp_path, "resume-reset-cache", run_kind="pilot", interval=1)
    counters = _cache_stats()

    def counted_request(unit):
        """建立 request 時增加目前 process 的 forcing load counter。"""

        counters["loads"] += 1
        counters["manager_count"] = counters["loads"]
        counters["resident_bytes"] = counters["loads"] * 10
        return _request(unit)

    shard_id = _first_shard(workspace)
    first = RunController(
        workspace,
        request_factory=counted_request,
        resource_reporter=lambda: dict(counters),
    )
    first.run_shard(shard_id, sweep_budget=1)
    assert load_run_progress(workspace)["shards"][shard_id]["metrics"]["forcing_cache_stats"]["loads"] == 2

    # 新 process 的 manager cache counter 從零開始；progress 仍帶著上一段 invocation 的
    # delta，兩者相加後應為實際舊 2 + 新 2，而不是把舊值誤視為新 baseline。
    counters.update(_cache_stats())
    second = RunController(
        workspace,
        request_factory=counted_request,
        resource_reporter=lambda: dict(counters),
        resume=True,
    )
    second.run_shard(shard_id)
    row = load_run_progress(workspace)["shards"][shard_id]
    assert counters["loads"] == 2
    assert row["metrics"]["forcing_cache_stats"]["loads"] == 4
    assert row["metrics"]["forcing_cache_stats_semantics"] == "invocation_delta_v1"


def test_checkpoint_high_gauge_is_kept_when_completion_sample_is_lower(
    tmp_path: Path,
) -> None:
    """同 invocation 先觀測高 gauge、完成時降回低值時仍保留 invocation 峰值。"""

    workspace = _workspace(tmp_path, "gauge-peak-cache", run_kind="pilot", interval=1)
    counters = _cache_stats()
    reporter_calls = 0

    def reporter():
        """回傳完整 snapshot，第一個 checkpoint 的 gauge 高於後續完成樣本。"""

        nonlocal reporter_calls
        reporter_calls += 1
        gauge = 0 if reporter_calls == 1 else 100 if reporter_calls == 2 else 10
        return _cache_stats(manager_count=gauge, resident_bytes=gauge * 10)

    def counted_request(unit):
        """建立 request 時增加 forcing load，保持 reporter counter 單調。"""

        counters["loads"] += 1
        return _request(unit)

    # 將 counter 與 gauge reporter 分開，避免測試中的 gauge 變化改變 counter delta。
    def combined_reporter():
        """合併單調 loads 與刻意下降的 gauge。"""

        snapshot = reporter()
        snapshot["loads"] = counters["loads"]
        return snapshot

    shard_id = _first_shard(workspace)
    RunController(
        workspace,
        request_factory=counted_request,
        resource_reporter=combined_reporter,
    ).run_shard(shard_id)
    row = load_run_progress(workspace)["shards"][shard_id]
    assert reporter_calls >= 3
    assert row["metrics"]["forcing_cache_stats"]["manager_count"] == 100
    assert row["metrics"]["forcing_cache_stats"]["resident_bytes"] == 1000


def test_legacy_cache_stats_remain_traceable_without_precise_aggregate(tmp_path: Path) -> None:
    """沒有新語意標記的歷史 cache object 可讀，但不產生精確總量。"""

    workspace = _workspace(tmp_path, "legacy-cache-metrics", run_kind="pilot", interval=1)
    RunController(workspace, request_factory=_request).run_all()
    progress = load_run_progress(workspace)
    for row in progress["shards"].values():
        row["metrics"]["forcing_cache_stats"] = {"loads": 99}
    _write_progress(workspace, progress)

    summary = benchmark_report(workspace)["summary"]
    assert summary["forcing_cache_stats"] == {}
    assert summary["forcing_cache_stats_precise"] is False
    assert summary["forcing_cache_stats_semantics"] == "legacy_unknown"
    assert set(summary["forcing_cache_stats_legacy_by_shard"]) == set(progress["shards"])


def test_missing_cache_measurement_on_either_resume_side_is_legacy() -> None:
    """前次或本次 invocation 缺 cache 量測時，不得把另一側默認成零而宣稱精確。"""

    base = {
        "wall_seconds": 1.0,
        "process_cpu_seconds": 1.0,
        "max_rss_bytes": 1,
        "output_bytes": 0,
        "checkpoint_bytes": 0,
        "particle_steps": 1,
    }
    current = {
        **base,
        "forcing_cache_stats": _cache_stats(loads=2),
        "forcing_cache_stats_semantics": "invocation_delta_v1",
    }
    previous = {
        **base,
        "forcing_cache_stats": _cache_stats(loads=3),
        "forcing_cache_stats_semantics": "invocation_delta_v1",
    }
    assert _merge_metrics(current, {**base})["forcing_cache_stats_semantics"] == "legacy_unknown"
    assert _merge_metrics({**base}, previous)["forcing_cache_stats_semantics"] == "legacy_unknown"


def test_invocation_counter_regression_is_rejected() -> None:
    """同 invocation 的較新 counter 不得回退覆蓋已觀測的事件數。"""

    earlier = _cache_stats(loads=5)
    later = _cache_stats(loads=3)
    with pytest.raises(ValueError, match="counter regression"):
        _merge_invocation_forcing_snapshot(earlier, later)


def test_reporter_error_does_not_mask_original_failure(tmp_path: Path) -> None:
    """reporter malformed 時保持原始 request error，並以 unavailable 保存量測限制。"""

    workspace = _workspace(tmp_path, "reporter-failure", run_kind="pilot", interval=1)
    counters = _cache_stats()
    reporter_calls = 0

    def reporter():
        """第一次提供 baseline，後續模擬 reporter 格式損壞。"""

        nonlocal reporter_calls
        reporter_calls += 1
        if reporter_calls == 1:
            return dict(counters)
        raise ValueError("reporter malformed")

    def failing_factory(unit):
        """模擬 request 建構的原始物理／輸入錯誤。"""

        del unit
        counters["loads"] += 1
        raise RuntimeError("original request failure")

    shard_id = _first_shard(workspace)
    with pytest.raises(RuntimeError, match="original request failure"):
        RunController(
            workspace,
            request_factory=failing_factory,
            resource_reporter=reporter,
        ).run_shard(shard_id)
    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "FAILED"
    assert row["metrics"]["forcing_cache_stats_status"] == "unavailable_due_to_reporter_error"
    assert all(value >= 0 for value in row["metrics"]["forcing_cache_stats"].values())


def test_keyboard_interrupt_during_factory_preserves_cache_snapshot(tmp_path: Path) -> None:
    """batch 尚未建立時中斷仍保存已發生的 cache loads，並維持 RUNNING。"""

    workspace = _workspace(tmp_path, "keyboard-cache-metrics", run_kind="pilot", interval=1)
    counters = _cache_stats()

    def interrupting_factory(unit):
        """模擬第一個 request 已觸發 load 後被 operator 中斷。"""

        del unit
        counters["loads"] += 1
        counters["manager_count"] = 1
        counters["resident_bytes"] = 10
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        RunController(
            workspace,
            request_factory=interrupting_factory,
            resource_reporter=lambda: dict(counters),
        ).run_shard(_first_shard(workspace))
    row = load_run_progress(workspace)["shards"][_first_shard(workspace)]
    assert row["lifecycle"] == "RUNNING"
    assert row["checkpoint_sequence"] == 0
    assert row["metrics"]["forcing_cache_stats"]["loads"] == 1
    assert row["metrics"]["forcing_cache_stats_semantics"] == "invocation_delta_v1"


def test_keyboard_interrupt_after_batch_keeps_checkpoint_cache_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """batch 已建立時 Ctrl-C 產生 checkpoint，且 checkpoint metrics 不遺失 cache loads。"""

    workspace = _workspace(tmp_path, "keyboard-checkpoint-cache", run_kind="pilot", interval=1)
    counters = _cache_stats()

    def counted_request(unit):
        """建立 request 時累計 forcing loads。"""

        counters["loads"] += 1
        counters["manager_count"] = counters["loads"]
        counters["resident_bytes"] = counters["loads"] * 10
        return _request(unit)

    def interrupt(self: ProductionBatch, sweeps: int = 1):
        """模擬 batch 已建立後的 operator 中斷。"""

        del self, sweeps
        raise KeyboardInterrupt

    monkeypatch.setattr(ProductionBatch, "advance", interrupt)
    with pytest.raises(KeyboardInterrupt):
        RunController(
            workspace,
            request_factory=counted_request,
            resource_reporter=lambda: dict(counters),
        ).run_shard(_first_shard(workspace))
    row = load_run_progress(workspace)["shards"][_first_shard(workspace)]
    assert row["lifecycle"] == "PAUSED"
    assert row["metrics"]["forcing_cache_stats"]["loads"] == 2
    assert row["metrics"]["forcing_cache_stats_semantics"] == "invocation_delta_v1"
