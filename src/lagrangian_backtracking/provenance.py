"""建立不含本機絕對路徑的程式部署指紋。

正式 run 必須能回答「實際執行的是哪一份程式、依賴與資料部署」；只保存 Git commit
不夠，因為 SERVER 部署可能沒有 ``.git``，而 dirty working tree 也可能讓同一 commit
對應到不同原始碼。本模組因此固定掃描 ``pyproject.toml``、``uv.lock`` 及套件目錄內
的普通 Python 檔，依相對路徑、檔案大小與檔案 SHA-256 建立 deployment tree 指紋。
輸出只保存版本與雜湊，不保存 ``project_root``，避免把本機或 SERVER 絕對路徑帶入
可攜的 run manifest。
"""

from __future__ import annotations

import json
import platform as platform_module
import re
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .outputs import sha256_file

_PACKAGE_DISTRIBUTION = "lagrangian-ensemble-backtracking"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_TREE_ROOT_FILES = ("pyproject.toml", "uv.lock")


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    """不可變的程式、依賴與部署環境指紋。

    ``git_available`` 區分本機有 Git metadata 與僅有發布檔案的 SERVER 部署；因此在
    沒有 ``.git`` 的 pilot 可以保留 ``git_commit=None``，但正式模式必須由外部宣告一個
    合法 40 位小寫 commit。``git_dirty`` 在無 Git 的部署固定為 ``None``，不把「無法
    判定」誤寫成乾淨。``deployment_file_count`` 是實際納入 canonical tree 的檔案數，
    方便檢查部署漏檔；所有 hash 均為小寫十六進位 SHA-256。
    """

    git_available: bool
    git_commit: str | None
    git_dirty: bool | None
    commit_source: str
    deployment_tree_sha256: str
    deployment_file_count: int
    uv_lock_sha256: str
    python_version: str
    platform: str
    package_version: str
    numpy_version: str
    numba_version: str
    pyarrow_version: str

    def to_dict(self) -> dict[str, Any]:
        """回傳不含路徑、可直接寫入 JSON 的 immutable provenance snapshot。"""

        return asdict(self)


def _run_git(project_root: Path, *arguments: str) -> str:
    """在指定專案根目錄執行唯讀 Git 命令，失敗時保留 fail-fast 語意。"""

    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Git provenance 命令失敗：{' '.join(arguments)}") from exc
    return completed.stdout.strip()


def _validate_commit(value: str, *, label: str) -> str:
    """確認 commit 是資料契約要求的 40 位小寫十六進位字串。"""

    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是 40 位小寫 hexadecimal Git commit")
    return value


def _ensure_regular_file(path: Path, *, label: str) -> None:
    """拒絕 symlink、目錄與不存在檔案，避免 fingerprint 跟隨部署外的資料。"""

    if path.is_symlink():
        raise ValueError(f"{label} 不允許 symlink：{path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} 必須是存在的普通檔案：{path}")


def _tree_entries(project_root: Path) -> list[tuple[str, int, str]]:
    """收集固定部署範圍的 ``(relative path, size, SHA-256)`` 排序清單。

    source 套件下只納入 ``.py``，但會檢查該樹內所有 symlink；這樣不會因為 symlink
    指向外部而產生看似可重現、實際不可部署的指紋。``pyproject.toml`` 與 ``uv.lock``
    是必備根檔，缺任一檔案都拒絕建立 provenance。
    """

    if project_root.is_symlink() or not project_root.is_dir():
        raise ValueError("project_root 必須是非 symlink 的目錄")
    entries: list[tuple[str, Path]] = []
    for relative in _TREE_ROOT_FILES:
        path = project_root / relative
        _ensure_regular_file(path, label=relative)
        entries.append((relative, path))
    source_root = project_root / "src" / "lagrangian_backtracking"
    if source_root.is_symlink() or not source_root.is_dir():
        raise FileNotFoundError("src/lagrangian_backtracking 必須是存在的普通目錄")
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(project_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"deployment tree 不允許 symlink：{relative}")
        if path.is_dir():
            continue
        if path.suffix == ".py":
            if not path.is_file():
                raise FileNotFoundError(f"source Python 檔案不存在：{relative}")
            entries.append((relative, path))
    result: list[tuple[str, int, str]] = []
    for relative, path in entries:
        _ensure_regular_file(path, label=relative)
        result.append((relative, path.stat().st_size, sha256_file(path)))
    result.sort(key=lambda item: item[0])
    return result


def _tree_hash(entries: list[tuple[str, int, str]]) -> str:
    """以 canonical JSON 固定路徑、大小、內容 hash 的欄位界線與順序。"""

    encoded = json.dumps(
        [{"path": relative, "size_bytes": size, "sha256": digest} for relative, size, digest in entries],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _package_version() -> str:
    """讀取目前套件版本；未安裝 editable metadata 時以 ``unknown`` 明示限制。"""

    try:
        return importlib_metadata.version(_PACKAGE_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _version(distribution: str) -> str:
    """讀取可選科學依賴版本，失敗時回傳明示的 ``unavailable``。"""

    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def collect_code_provenance(
    project_root: str | Path,
    declared_git_commit: str | None = None,
    *,
    formal: bool = False,
) -> CodeProvenance:
    """收集程式部署指紋並套用 Git／無 Git 的正式閘門。

    有 ``.git`` 時，commit 由 ``git rev-parse HEAD`` 取得，dirty 狀態由完整
    ``git status --porcelain --untracked-files=all`` 判定；若 caller 另給 declared commit，
    兩者必須完全一致。沒有 ``.git`` 時 pilot 允許 ``git_commit=None``，但 formal 必須由
    caller 提供合法 commit，且此時 ``git_dirty=None``、``commit_source`` 會標為
    ``declared_deployment``。正式模式拒絕 dirty tree。任何 symlink、缺少固定根檔或
    source 目錄錯誤都在產生結果前拒絕，避免建立不完整的 run binding。
    """

    root = Path(project_root)
    entries = _tree_entries(root)
    uv_lock_sha256 = next(digest for relative, _, digest in entries if relative == "uv.lock")
    git_marker = root / ".git"
    git_available = git_marker.exists()
    git_commit: str | None
    git_dirty: bool | None
    if git_available:
        git_commit = _validate_commit(_run_git(root, "rev-parse", "HEAD"), label="git commit")
        if declared_git_commit is not None:
            declared = _validate_commit(declared_git_commit, label="declared_git_commit")
            if declared != git_commit:
                raise ValueError(f"declared_git_commit 與 Git HEAD 不一致：{declared} != {git_commit}")
        status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
        git_dirty = bool(status)
        commit_source = "git_repository"
    else:
        git_commit = None
        git_dirty = None
        if formal:
            if declared_git_commit is None:
                raise ValueError("formal 無 .git 部署必須提供 declared_git_commit")
            git_commit = _validate_commit(declared_git_commit, label="declared_git_commit")
        elif declared_git_commit is not None:
            git_commit = _validate_commit(declared_git_commit, label="declared_git_commit")
        commit_source = "declared_deployment" if git_commit is not None else "no_git_pilot"
    if formal and git_dirty is True:
        raise ValueError("formal provenance 拒絕 dirty Git working tree")
    return CodeProvenance(
        git_available=git_available,
        git_commit=git_commit,
        git_dirty=git_dirty,
        commit_source=commit_source,
        deployment_tree_sha256=_tree_hash(entries),
        deployment_file_count=len(entries),
        uv_lock_sha256=uv_lock_sha256,
        python_version=platform_module.python_version(),
        platform=platform_module.platform(),
        package_version=_package_version(),
        numpy_version=_version("numpy"),
        numba_version=_version("numba"),
        pyarrow_version=_version("pyarrow"),
    )


__all__ = ["CodeProvenance", "collect_code_provenance"]
