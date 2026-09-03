"""程式 deployment fingerprint 與 Git/無 Git 正式閘門測試。"""

from __future__ import annotations

import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from lagrangian_backtracking.provenance import collect_code_provenance

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _project(root: Path) -> None:
    """建立只含固定 fingerprint 範圍的最小部署樹。"""

    source = root / "src" / "lagrangian_backtracking"
    source.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")


def _git_init(root: Path) -> None:
    """在 fixture 建立可由 provenance 讀取的乾淨 Git commit。"""

    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "測試提交"], check=True)


def test_git_clean_and_dirty_are_distinguished(tmp_path: Path) -> None:
    """Git fixture 的 clean/dirty 狀態與 deployment hash 都應可重現。"""

    _project(tmp_path)
    _git_init(tmp_path)
    clean = collect_code_provenance(tmp_path)
    assert clean.git_available is True
    assert clean.git_commit is not None
    assert clean.git_dirty is False
    assert clean == collect_code_provenance(tmp_path)
    (tmp_path / "src" / "lagrangian_backtracking" / "__init__.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    dirty = collect_code_provenance(tmp_path)
    assert dirty.git_dirty is True
    assert dirty.deployment_tree_sha256 != clean.deployment_tree_sha256
    with pytest.raises(ValueError, match="dirty"):
        collect_code_provenance(tmp_path, formal=True)


def test_declared_commit_mismatch_is_rejected(tmp_path: Path) -> None:
    """有 Git 時外部宣告 commit 若非 HEAD，不得建立可能誤綁定的 provenance。"""

    _project(tmp_path)
    _git_init(tmp_path)
    with pytest.raises(ValueError, match="不一致"):
        collect_code_provenance(tmp_path, declared_git_commit=_COMMIT)


def test_no_git_pilot_and_formal_declared_deployment(tmp_path: Path) -> None:
    """無 .git 的部署在 pilot 可為 None，formal 必須使用合法 declared commit。"""

    _project(tmp_path)
    pilot = collect_code_provenance(tmp_path)
    assert pilot.git_available is False
    assert pilot.git_commit is None
    assert pilot.git_dirty is None
    formal = collect_code_provenance(tmp_path, declared_git_commit=_COMMIT, formal=True)
    assert formal.git_commit == _COMMIT
    assert formal.commit_source == "declared_deployment"
    assert formal.git_dirty is None
    with pytest.raises(ValueError, match="40 位"):
        collect_code_provenance(tmp_path, declared_git_commit="bad")


def test_tree_content_and_symlink_are_bound(tmp_path: Path) -> None:
    """內容變更必須改變 hash，source symlink 與根檔 symlink 必須拒絕。"""

    _project(tmp_path)
    first = collect_code_provenance(tmp_path)
    source = tmp_path / "src" / "lagrangian_backtracking" / "__init__.py"
    source.write_text("VALUE = 3\n", encoding="utf-8")
    second = collect_code_provenance(tmp_path)
    assert second.deployment_tree_sha256 != first.deployment_tree_sha256
    source.unlink()
    source.symlink_to(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("VALUE = 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symlink"):
        collect_code_provenance(tmp_path)
    source.unlink()
    (tmp_path / "uv.lock").unlink()
    (tmp_path / "uv.lock").symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="symlink"):
        collect_code_provenance(tmp_path)


def test_provenance_does_not_contain_project_root(tmp_path: Path) -> None:
    """JSON-ready provenance 欄位不應保存本機絕對專案路徑。"""

    _project(tmp_path)
    value = collect_code_provenance(tmp_path).to_dict()
    assert str(tmp_path) not in repr(value)
    assert value["uv_lock_sha256"] == sha256((tmp_path / "uv.lock").read_bytes()).hexdigest()
