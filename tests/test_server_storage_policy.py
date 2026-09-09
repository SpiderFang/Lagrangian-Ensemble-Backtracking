"""SERVER NFS storage gate 的路徑、掛載、空間與程序鎖定契約測試。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_server_storage.py"
SPEC = importlib.util.spec_from_file_location("validate_server_storage", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
storage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = storage
SPEC.loader.exec_module(storage)


def _layout(tmp_path: Path) -> dict[str, Path]:
    """建立一組不依賴真 NFS 的 project、venv 與 NFS 子目錄。"""

    project = tmp_path / "home" / "project"
    (project / ".venv").mkdir(parents=True)
    nfs = tmp_path / "data" / "lbt-results"
    nfs.mkdir(parents=True)
    roots = {
        "project_root": project,
        "project_venv": project / ".venv",
        "result_nfs_root": nfs,
    }
    for label in storage.EXECUTION_ROOT_LABELS:
        path = nfs / label
        path.mkdir()
        roots[label] = path
    roots["minimum_free_gib"] = 1
    return roots


def _mount_probe(nfs_root: Path, different_mount: set[Path] | None = None):
    """回傳固定 mount metadata，讓測試不需要呼叫真實 findmnt。"""

    different_mount = different_mount or set()

    def probe(path: Path) -> Any:
        if path in different_mount:
            return storage.MountInfo("nfs4", "other-server:/other", "/other")
        return storage.MountInfo("nfs4", "server:/export", "/data")

    return probe


def _validate(layout: dict[str, Path], **overrides: Any) -> dict[str, object]:
    """套用固定 mock 探針執行 gate，並允許單一測試替換一項條件。"""

    values: dict[str, Any] = dict(layout)
    values.update(
        {
            "mount_probe": _mount_probe(layout["result_nfs_root"]),
            "write_probe": lambda _path: True,
            "flock_probe": lambda _path: True,
            "disk_usage": lambda _path: SimpleNamespace(free=16 * 1024**3),
        }
    )
    values.update(overrides)
    return storage.validate_storage_policy(**values)


def test_valid_nfs_layout_passes_without_absolute_paths(tmp_path: Path) -> None:
    """合法 layout 通過，snapshot 只能含 label、hash、bytes 與狀態等欄位。"""

    snapshot = _validate(_layout(tmp_path))

    assert snapshot["gate_status"] == "PASS"
    assert not snapshot["issues"]
    encoded = json.dumps(snapshot, ensure_ascii=False)
    assert str(tmp_path) not in encoded
    root_rows = {row["label"]: row for row in snapshot["roots"]}
    assert root_rows["result_nfs_root"]["fstype"] == "nfs4"
    assert root_rows["result_nfs_root"]["source_token_hash"]
    assert root_rows["output_root"]["free_bytes"] == 16 * 1024**3


def test_project_root_and_venv_can_be_outside_nfs(tmp_path: Path) -> None:
    """/home 上的 project 與固定 .venv 不會被誤當成成果逃逸。"""

    snapshot = _validate(_layout(tmp_path))

    assert snapshot["gate_status"] == "PASS"


def test_output_root_outside_nfs_is_rejected(tmp_path: Path) -> None:
    """任何執行成果根離開 NFS root 都必須停止 gate。"""

    layout = _layout(tmp_path)
    escaped = tmp_path / "home" / "escaped-output"
    escaped.mkdir()
    layout["output_root"] = escaped

    snapshot = _validate(layout)

    assert snapshot["gate_status"] == "FAIL"
    assert {issue["code"] for issue in snapshot["issues"]} >= {"output_root_outside_result_root"}


def test_symlink_component_is_rejected(tmp_path: Path) -> None:
    """中間 symlink 即使最後解析到 NFS 內也不得通過。"""

    layout = _layout(tmp_path)
    real_root = layout["result_nfs_root"] / "real-output"
    real_root.mkdir()
    link_root = layout["result_nfs_root"] / "link-output"
    link_root.symlink_to(real_root, target_is_directory=True)
    layout["output_root"] = link_root

    snapshot = _validate(layout)

    assert snapshot["gate_status"] == "FAIL"
    assert {issue["code"] for issue in snapshot["issues"]} >= {"output_root_symlink_component"}


def test_different_mount_source_is_rejected(tmp_path: Path) -> None:
    """同為 NFS 但 source/target 不同時仍不得混用成果。"""

    layout = _layout(tmp_path)
    other = layout["checkpoint_root"]

    snapshot = _validate(
        layout,
        mount_probe=_mount_probe(layout["result_nfs_root"], {other}),
    )

    assert snapshot["gate_status"] == "FAIL"
    assert {
        issue["code"]
        for issue in snapshot["issues"]
        if issue["label"] == "checkpoint_root"
    } >= {"mount_source_or_target_mismatch"}


def test_non_nfs_result_mount_is_rejected(tmp_path: Path) -> None:
    """result root 不是 nfs/nfs4 時整體 gate 必須失敗。"""

    layout = _layout(tmp_path)

    def probe(path: Path) -> Any:
        del path
        return storage.MountInfo("ext4", "/dev/test", "/data")

    snapshot = _validate(layout, mount_probe=probe)

    assert snapshot["gate_status"] == "FAIL"
    assert {issue["code"] for issue in snapshot["issues"]} >= {"result_root_not_nfs"}


@pytest.mark.parametrize(
    "root_label",
    ["uv_cache_root", "mpl_cache_root", "xdg_cache_root", "tmp_root"],
)
def test_cache_and_tmp_roots_cannot_escape_nfs(tmp_path: Path, root_label: str) -> None:
    """UV、matplotlib、XDG 與 TMP 每一個都獨立接受 strict descendant gate。"""

    layout = _layout(tmp_path)
    escaped = tmp_path / "home" / f"escaped-{root_label}"
    escaped.mkdir()
    layout[root_label] = escaped

    snapshot = _validate(layout)

    assert snapshot["gate_status"] == "FAIL"
    assert {
        issue["code"]
        for issue in snapshot["issues"]
        if issue["label"] == root_label
    } >= {f"{root_label}_outside_result_root"}


def test_free_space_failure_is_reported(tmp_path: Path) -> None:
    """任一 root 可用空間低於最低 GiB 即拒絕。"""

    layout = _layout(tmp_path)

    snapshot = _validate(
        layout,
        minimum_free_gib=2,
        disk_usage=lambda _path: SimpleNamespace(free=1),
    )

    assert snapshot["gate_status"] == "FAIL"
    assert {issue["code"] for issue in snapshot["issues"]} >= {"free_space_insufficient"}


def test_write_probe_failure_is_reported(tmp_path: Path) -> None:
    """寫入探針失敗時不能只依賴 os.access 放行。"""

    layout = _layout(tmp_path)

    snapshot = _validate(layout, write_probe=lambda _path: False)

    assert snapshot["gate_status"] == "FAIL"
    assert {issue["code"] for issue in snapshot["issues"]} >= {"write_probe_failed"}


def test_flock_probe_failure_is_reported(tmp_path: Path) -> None:
    """跨程序 flock 探針失敗時停止執行。"""

    layout = _layout(tmp_path)

    snapshot = _validate(layout, flock_probe=lambda _path: False)

    assert snapshot["gate_status"] == "FAIL"
    assert {issue["code"] for issue in snapshot["issues"]} >= {"flock_probe_failed"}


def test_output_scratch_checkpoint_roots_must_not_overlap(tmp_path: Path) -> None:
    """三個核心生命週期根即使在 NFS 內也不得相等或互相包含。"""

    layout = _layout(tmp_path)
    nested_scratch = layout["output_root"] / "nested-scratch"
    nested_scratch.mkdir()
    layout["scratch_root"] = nested_scratch

    snapshot = _validate(layout)

    assert snapshot["gate_status"] == "FAIL"
    assert {
        issue["code"]
        for issue in snapshot["issues"]
        if issue["label"] in {"output_root", "scratch_root"}
    } >= {"output_scratch_checkpoint_overlap"}


def test_write_probe_checks_same_directory_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write probe 必須實際執行同目錄 os.replace，而非只 create/unlink。"""

    calls: list[tuple[Path, Path]] = []
    original_replace = storage.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        calls.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", recording_replace)

    assert storage._write_probe(tmp_path)
    assert len(calls) == 1
    assert calls[0][0].parent == tmp_path
    assert calls[0][1].parent == tmp_path
    assert not list(tmp_path.glob(".lbt-storage-write-*"))


def test_snapshot_output_must_be_strict_nfs_descendant(tmp_path: Path) -> None:
    """gate 快照本身不能藉由 CLI output path 逃到 project 或其他磁碟。"""

    layout = _layout(tmp_path)
    escaped_parent = tmp_path / "home" / "snapshot"
    escaped_parent.mkdir()
    snapshot = _validate(layout)

    with pytest.raises(storage.StoragePolicyError, match="snapshot_output_parent_outside_result_root"):
        storage._write_snapshot(
            escaped_parent / "gate.json",
            snapshot,
            result_nfs_root=layout["result_nfs_root"],
            scratch_root=layout["scratch_root"],
        )


def test_snapshot_output_under_scratch_is_written_atomically(tmp_path: Path) -> None:
    """合法 scratch 子路徑可保存 JSON gate snapshot，且內容不含絕對路徑。"""

    layout = _layout(tmp_path)
    destination = layout["scratch_root"] / "gate.json"
    snapshot = _validate(layout)

    storage._write_snapshot(
        destination,
        snapshot,
        result_nfs_root=layout["result_nfs_root"],
        scratch_root=layout["scratch_root"],
    )

    encoded = destination.read_text(encoding="utf-8")
    assert json.loads(encoded)["gate_status"] == "PASS"
    assert str(tmp_path) not in encoded


def test_snapshot_output_inside_nfs_but_outside_scratch_is_rejected(tmp_path: Path) -> None:
    """快照即使仍在 NFS root，也不能改寫 output 或其他 sibling 子樹。"""

    layout = _layout(tmp_path)
    destination = layout["output_root"] / "gate.json"
    snapshot = _validate(layout)

    with pytest.raises(storage.StoragePolicyError, match="snapshot_output_outside_result_root"):
        storage._write_snapshot(
            destination,
            snapshot,
            result_nfs_root=layout["result_nfs_root"],
            scratch_root=layout["scratch_root"],
        )


def test_findmnt_parser_accepts_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """預設 findmnt probe 解析 JSON 並只回傳 mount 三元組。"""

    output = json.dumps(
        {
            "filesystems": [
                {"fstype": "nfs4", "source": "server:/export", "target": str(tmp_path)}
            ]
        }
    )

    calls: list[list[str]] = []

    def fake_run(*args: Any, **_kwargs: Any) -> Any:
        calls.append(args[0])
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(storage.subprocess, "run", fake_run)
    info = storage._findmnt_mount_probe(tmp_path)

    assert info == storage.MountInfo("nfs4", "server:/export", str(tmp_path))
    assert "--raw" not in calls[0]
    assert "--noheadings" not in calls[0]


def test_findmnt_parser_chooses_deepest_nested_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """巢狀 NFS/autofs 多列時取涵蓋目標的最深 mount。"""

    nested = tmp_path / "nested"
    nested.mkdir()
    output = json.dumps(
        {
            "filesystems": [
                {"fstype": "autofs", "source": "systemd-1", "target": str(tmp_path)},
                {"fstype": "nfs4", "source": "server:/export", "target": str(nested)},
            ]
        }
    )
    monkeypatch.setattr(
        storage.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    info = storage._findmnt_mount_probe(nested)

    assert info == storage.MountInfo("nfs4", "server:/export", str(nested))


def test_findmnt_parser_prefers_nfs_when_mount_targets_tie(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """同一 target 同時回 autofs 與 NFS 時，不能因 findmnt 順序選到 autofs。"""

    output = json.dumps(
        {
            "filesystems": [
                {"fstype": "autofs", "source": "systemd-1", "target": str(tmp_path)},
                {"fstype": "nfs", "source": "server:/export", "target": str(tmp_path)},
            ]
        }
    )
    monkeypatch.setattr(
        storage.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    info = storage._findmnt_mount_probe(tmp_path)

    assert info == storage.MountInfo("nfs", "server:/export", str(tmp_path))


def test_cli_exit_code_and_snapshot_are_machine_readable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """CLI 對 mock gate 以 0/2 表達結果，stdout 不含絕對路徑。"""

    layout = _layout(tmp_path)
    monkeypatch.setattr(
        storage,
        "validate_storage_policy",
        lambda **_kwargs: {
            "schema_version": "1.0.0",
            "minimum_free_bytes": 1,
            "gate_status": "PASS",
            "roots": [],
            "probes": [],
            "issues": [],
        },
    )
    arguments = []
    for name, value in layout.items():
        option = f"--{name.replace('_', '-')}"
        if name != "minimum_free_gib":
            arguments.extend([option, str(value)])
    arguments.extend(["--minimum-free-gib", "1"])

    assert storage.main(arguments) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["gate_status"] == "PASS"
    assert str(tmp_path) not in output


def test_cli_nonzero_on_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """CLI gate failure 使用狀態 2，供 shell runner fail-closed。"""

    layout = _layout(tmp_path)
    monkeypatch.setattr(
        storage,
        "validate_storage_policy",
        lambda **_kwargs: {
            "schema_version": "1.0.0",
            "minimum_free_bytes": 1,
            "gate_status": "FAIL",
            "roots": [],
            "probes": [],
            "issues": [{"label": "tmp_root", "code": "tmp_root_outside_result_root"}],
        },
    )
    arguments = []
    for name, value in layout.items():
        option = f"--{name.replace('_', '-')}"
        if name != "minimum_free_gib":
            arguments.extend([option, str(value)])
    arguments.extend(["--minimum-free-gib", "1"])

    assert storage.main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["gate_status"] == "FAIL"


def test_cli_snapshot_output_escape_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """CLI 不能把 gate snapshot 指向 NFS 外的 /home-like 路徑。"""

    layout = _layout(tmp_path)
    escaped_parent = tmp_path / "home" / "snapshot"
    escaped_parent.mkdir()
    monkeypatch.setattr(
        storage,
        "validate_storage_policy",
        lambda **_kwargs: {
            "schema_version": "1.0.0",
            "minimum_free_bytes": 1,
            "gate_status": "PASS",
            "roots": [],
            "probes": [],
            "issues": [],
        },
    )
    arguments = []
    for name, value in layout.items():
        option = f"--{name.replace('_', '-')}"
        if name != "minimum_free_gib":
            arguments.extend([option, str(value)])
    arguments.extend(
        [
            "--minimum-free-gib",
            "1",
            "--snapshot-output",
            str(escaped_parent / "gate.json"),
        ]
    )

    assert storage.main(arguments) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["gate_status"] == "FAIL"
    assert not (escaped_parent / "gate.json").exists()


def test_tracked_runner_calls_gate_before_package_entrypoint() -> None:
    """tracked runner 必須先 gate，再把 /data package 交給既有 orchestration。"""

    runner = SCRIPT_PATH.with_name("run_b_hsinchu_expanded_matrix.sh").read_text(encoding="utf-8")

    gate_position = runner.index("validate_server_storage.py")
    package_position = runner.index('exec bash "$LBT_EXECUTION_PACKAGE_ROOT/run_server_matrix.sh"')
    assert gate_position < package_position
    for variable in (
        "LBT_RESULT_NFS_ROOT",
        "LBT_EXECUTION_PACKAGE_ROOT",
        "LBT_UV_CACHE_ROOT",
        "LBT_MPL_CACHE_ROOT",
        "LBT_XDG_CACHE_ROOT",
        "LBT_TMP_ROOT",
    ):
        assert variable in runner


def test_a4_forcing_script_requires_explicit_nfs_uv_cache() -> None:
    """A 區 forcing 前處理不能將 uv cache fallback 到 /tmp。"""

    source = (SCRIPT_PATH.parent / "prepare_a_v4_forcing.sh").read_text(encoding="utf-8")

    assert "LBT_UV_CACHE_ROOT" in source
    assert "LBT_RESULT_NFS_ROOT" in source
    assert "cache_relative" in source
    assert "symlink 元件" in source
    assert "/tmp/lbt-a-v4-uv-cache" not in source
    assert "LBT_WORKSPACE_UV_CACHE_ROOT" not in source


def test_preview_rebuild_command_uses_verified_cache_environment() -> None:
    """預覽重跑命令只能引用已驗證的 NFS cache roots。"""

    source = (SCRIPT_PATH.parent / "build_pilot_coastline_preview.py").read_text(encoding="utf-8")

    assert "work/uv-cache" not in source
    assert "work/matplotlib-cache" not in source
    for variable in (
        "LBT_UV_CACHE_ROOT",
        "LBT_MPL_CACHE_ROOT",
        "LBT_XDG_CACHE_ROOT",
        "LBT_TMP_ROOT",
    ):
        assert variable in source


def test_real_cli_rejects_escaped_output_without_writing_path(tmp_path: Path) -> None:
    """真實 CLI 拒絕離開 result root 的輸出，且不洩漏測試絕對路徑。

    SERVER 的 pytest 暫存根也依儲存政策放在 NFS，因此不能假設 ``tmp_path`` 位於
    非 NFS。這裡把 output 指到同一實體檔案系統、但位於邏輯 result root 外的既有
    目錄；無論測試由本機磁碟或 SERVER NFS 執行，都應穩定命中 strict-descendant
    gate，同時不必為了製造反例而在 SERVER ``/home`` 或 ``/tmp`` 建立測試資料。
    """

    layout = _layout(tmp_path)
    escaped_output = layout["project_root"] / "escaped-output"
    escaped_output.mkdir()
    layout["output_root"] = escaped_output
    arguments = [
        "python3",
        str(SCRIPT_PATH),
        "--project-root",
        str(layout["project_root"]),
        "--project-venv",
        str(layout["project_venv"]),
        "--result-nfs-root",
        str(layout["result_nfs_root"]),
    ]
    for label in storage.EXECUTION_ROOT_LABELS:
        arguments.extend([f"--{label.replace('_', '-')}", str(layout[label])])
    arguments.extend(["--minimum-free-gib", "1"])
    completed = subprocess.run(arguments, check=False, capture_output=True, text=True)

    assert completed.returncode == 2
    snapshot = json.loads(completed.stdout)
    assert snapshot["gate_status"] == "FAIL"
    assert {issue["code"] for issue in snapshot["issues"]} >= {
        "output_root_outside_result_root"
    }
    assert str(tmp_path) not in completed.stdout
