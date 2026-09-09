#!/usr/bin/env python3
"""驗證 SERVER 執行資料只能落在同一個 NFS 結果檔案系統。

這個獨立腳本是正式數值 runner 的儲存硬閘門。它把專案程式碼與既有虛擬環境
視為可位於 ``/home`` 的部署內容；輸出、scratch、checkpoint、執行 package 與
Python／繪圖／系統暫存快取則必須是 operator 指定 NFS 結果根目錄的嚴格子目錄。
每個目錄都會檢查實體目錄、路徑元件不可為符號連結、可寫性、剩餘空間及掛載
資訊。結果快照只保存 label、檔案系統型別、來源 token 的雜湊、可用 bytes 與
閘門狀態，避免把 SERVER 絕對路徑寫進可搬移的稽核證據。

正式 runner 會在任何 ``mkdir``、``uv`` 或 ``run-create`` 前執行本腳本。測試可以
把 ``mount_probe``、``write_probe``、``flock_probe`` 與 ``disk_usage`` 注入，因而
不需要一台真的 NFS 主機即可覆蓋拒絕條件。
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

NFS_FILESYSTEM_TYPES = frozenset({"nfs", "nfs4"})
"""正式 SERVER 允許的 NFS 檔案系統型別；大小寫差異會先正規化。"""

SCHEMA_VERSION = "1.0.0"
"""機器可讀 gate 快照的版本，變更欄位契約時必須遞增。"""

RESULT_ROOT_LABEL = "result_nfs_root"
"""NFS 根目錄的快照 label；它本身不是自身的嚴格子目錄。"""

EXECUTION_ROOT_LABELS = (
    "execution_package_root",
    "output_root",
    "scratch_root",
    "checkpoint_root",
    "uv_cache_root",
    "mpl_cache_root",
    "xdg_cache_root",
    "tmp_root",
)
"""必須位於 NFS 結果根目錄下的執行、成果與快取目錄 label。"""


@dataclass(frozen=True)
class MountInfo:
    """``findmnt`` 對單一路徑回報的掛載識別。

    ``source`` 與 ``target`` 僅供本次程序內比較；輸出快照只保留 source 的
    SHA-256 token，避免把主機名稱、export 或絕對掛載路徑暴露給成果使用者。
    """

    fstype: str
    source: str
    target: str


@dataclass(frozen=True)
class StorageRootState:
    """單一資料根目錄的 machine-readable 結果欄位。"""

    label: str
    fstype: str | None
    source_token_hash: str | None
    free_bytes: int | None
    gate_status: str

    def as_dict(self) -> dict[str, object]:
        """以固定欄位順序轉成不含絕對路徑的 JSON 物件。"""

        return {
            "label": self.label,
            "fstype": self.fstype,
            "source_token_hash": self.source_token_hash,
            "free_bytes": self.free_bytes,
            "gate_status": self.gate_status,
        }


@dataclass(frozen=True)
class StorageIssue:
    """驗證失敗的非敏感識別；不保存例外文字或絕對路徑。"""

    label: str
    code: str

    def as_dict(self) -> dict[str, str]:
        """將失敗原因限制為可供自動化判斷的 code 與 label。"""

        return {"label": self.label, "code": self.code}


class StoragePolicyError(RuntimeError):
    """內部驗證錯誤，固定以 code 傳遞而不把路徑帶入 snapshot。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


PathLike = Union[str, os.PathLike[str]]  # noqa: UP007 - 支援 SERVER 的 Python 3.9 呼叫端
MountProbe = Callable[[Path], MountInfo]
WriteProbe = Callable[[Path], bool]
FlockProbe = Callable[[Path], bool]
DiskUsage = Callable[[Path], shutil._ntuple_diskusage]


def _coerce_mount_info(value: object) -> MountInfo:
    """正規化注入探針回傳值，讓測試可使用 dataclass、mapping 或三元組。

    正式路徑由 ``_findmnt_mount_probe`` 直接回傳 ``MountInfo``；測試則可用簡單
    dict／tuple mock，不必模擬 subprocess 的完整 CompletedProcess。所有欄位仍須
    是非空字串，否則以固定 ``findmnt_output_invalid`` code 拒絕。
    """

    if isinstance(value, MountInfo):
        fstype, source, target = value.fstype, value.source, value.target
    elif isinstance(value, Mapping):
        fstype = value.get("fstype")
        source = value.get("source")
        target = value.get("target")
    elif isinstance(value, tuple) and len(value) == 3:
        fstype, source, target = value
    else:
        raise StoragePolicyError("findmnt_output_invalid")
    if not all(isinstance(item, str) and item for item in (fstype, source, target)):
        raise StoragePolicyError("findmnt_output_invalid")
    return MountInfo(fstype=fstype, source=source, target=target)


def _absolute_path(value: PathLike, label: str) -> Path:
    """將 operator 路徑轉成 Path，並先拒絕相對路徑或空值。

    解析工作必須在 gate 中完成，避免 shell 當前目錄或相對路徑讓成果寫入
    未預期位置；錯誤只回傳固定 code，不在快照中記錄輸入路徑。
    """

    path = Path(value)
    if not str(path) or not path.is_absolute():
        raise StoragePolicyError(f"{label}_must_be_absolute")
    return path


def _reject_symlink_components(path: Path, label: str) -> None:
    """逐一檢查既有路徑元件，拒絕任何符號連結與缺失元件。

    單看最終 ``Path.is_symlink`` 不足以防止中間元件把路徑導向另一個檔案系統；
    因此以 ``lstat`` 從根逐段檢查。這也要求 operator 先建立好所有正式目錄，
    runner 才能在通過 gate 後建立其下的 run workspace。
    """

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            component_stat = current.lstat()
        except FileNotFoundError as exc:
            raise StoragePolicyError(f"{label}_component_missing") from exc
        except OSError as exc:
            raise StoragePolicyError(f"{label}_component_stat_failed") from exc
        if stat.S_ISLNK(component_stat.st_mode):
            raise StoragePolicyError(f"{label}_symlink_component")


def _existing_directory(value: PathLike, label: str) -> Path:
    """驗證普通絕對目錄並回傳實體 resolved path。

    專案根與 ``.venv`` 也走同一個 symlink-component 檢查，避免 deployment
    contract 對程式環境與資料環境產生不同的路徑語意；它們可以位於 ``/home``，
    但仍必須是已存在的目錄。
    """

    path = _absolute_path(value, label)
    _reject_symlink_components(path, label)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise StoragePolicyError(f"{label}_resolve_failed") from exc
    if not resolved.is_dir():
        raise StoragePolicyError(f"{label}_not_directory")
    if not os.access(resolved, os.X_OK):
        raise StoragePolicyError(f"{label}_not_traversable")
    return resolved


def _require_strict_descendant(candidate: Path, root: Path, label: str) -> None:
    """要求 candidate 是 NFS 根的嚴格後代，拒絕根本身及域外路徑。"""

    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise StoragePolicyError(f"{label}_outside_result_root") from exc
    if not relative.parts:
        raise StoragePolicyError(f"{label}_must_be_strict_descendant")


def _findmnt_mount_probe(path: Path) -> MountInfo:
    """以 ``findmnt`` 取得 path 的 fstype、source 與 mount target。

    使用 JSON 形式避免 source 或 target 含空白時的欄位歧義；為了方便測試及
    某些舊版 ``findmnt``，若輸出不是 JSON 則接受同一命令的三欄文字格式。命令
    失敗或輸出不完整都視為硬閘門失敗，不以作業系統猜測替代。
    """

    try:
        completed = subprocess.run(
            [
                "findmnt",
                "--json",
                "--target",
                str(path),
                "--output",
                "FSTYPE,SOURCE,TARGET",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StoragePolicyError("findmnt_failed") from exc

    output = completed.stdout.strip()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        fields = output.split(maxsplit=2)
        if len(fields) != 3:
            raise StoragePolicyError("findmnt_output_invalid") from None
        fstype, source, target = fields
        return MountInfo(fstype=fstype, source=source, target=target)

    filesystems = payload.get("filesystems") if isinstance(payload, dict) else None
    if not isinstance(filesystems, list) or not filesystems:
        raise StoragePolicyError("findmnt_output_invalid")

    # autofs 或巢狀 NFS mount 可能回傳多列；必須選取實際涵蓋 path 且 target
    # 最深的一列，不能把第一列誤當成最終寫入檔案系統。target 只在程序內作
    # mount identity 比較，永不寫入 snapshot。
    valid_records: list[tuple[int, int, dict[str, object]]] = []
    for record in filesystems:
        if not isinstance(record, dict):
            continue
        target = record.get("target")
        if not isinstance(target, str) or not target.startswith("/"):
            continue
        try:
            _absolute_target = Path(target).resolve(strict=False)
            path.relative_to(_absolute_target)
        except (OSError, RuntimeError, ValueError):
            continue
        fstype = record.get("fstype")
        nfs_priority = int(isinstance(fstype, str) and fstype.lower() in NFS_FILESYSTEM_TYPES)
        valid_records.append((len(_absolute_target.parts), nfs_priority, record))
    if not valid_records:
        raise StoragePolicyError("findmnt_output_invalid")
    record = max(valid_records, key=lambda item: (item[0], item[1]))[2]
    fstype = record.get("fstype")
    source = record.get("source")
    target = record.get("target")
    if not all(isinstance(item, str) and item for item in (fstype, source, target)):
        raise StoragePolicyError("findmnt_output_invalid")
    return MountInfo(fstype=fstype, source=source, target=target)


def _source_token_hash(source: str) -> str:
    """以 SHA-256 將 mount source 轉成不可逆的稽核 token。"""

    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_probe(path: Path) -> bool:
    """在指定目錄建立、寫入、同步並移除小型探針檔。

    探針只寫入少量 bytes 並使用隨機檔名，確認 NFS permission 與實際寫入路徑
    有效；檔案成功同步後立即清除，不把測試資料留在研究成果目錄。任何錯誤
    都回傳 False，呼叫端會以 ``write_probe_failed`` 停止正式 runner。
    """

    temporary_path = path / f".lbt-storage-write-{os.getpid()}-{secrets.token_hex(8)}.tmp"
    published_path = path / f".lbt-storage-write-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    published_descriptor: int | None = None
    probe_succeeded = False
    try:
        descriptor = os.open(
            temporary_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.write(descriptor, b"lbt-storage-probe\n")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        # 正式成果會透過同目錄 atomic replace 發布；write probe 同樣驗證該
        # NFS 操作，避免只有 create 成功但原子發布失敗的檔案系統被放行。
        os.replace(temporary_path, published_path)
        published_descriptor = os.open(published_path, os.O_RDONLY)
        os.fsync(published_descriptor)
        probe_succeeded = True
    except OSError:
        probe_succeeded = False
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if published_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(published_descriptor)
        for probe_path in (temporary_path, published_path):
            try:
                probe_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # 寫入成功但清除失敗時仍視為 probe 失敗；殘留檔可被 operator
                # 依固定 prefix 找到，不會被當成有效正式成果。
                probe_succeeded = False
    return probe_succeeded


def _flock_child(path_string: str, connection: object) -> None:
    """子程序嘗試取得父程序持有的獨占鎖，僅傳回是否正確被阻擋。"""

    descriptor: int | None = None
    try:
        descriptor = os.open(path_string, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            connection.send(True)
        else:
            connection.send(False)
    except OSError:
        connection.send(False)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        connection.close()


def _flock_probe(path: Path) -> bool:
    """以兩個同主機程序驗證 NFS 上的 ``flock`` 互斥語意。

    父程序先持有鎖，子程序必須在非阻塞模式收到 ``EWOULDBLOCK``；若子程序也
    能取得鎖，代表目前檔案系統的跨程序鎖定不足以保護 run/reconcile topology。
    探針檔只承載鎖，不保存任何研究資料。
    """

    probe_path = path / f".lbt-storage-flock-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    parent_connection = None
    process = None
    try:
        descriptor = os.open(probe_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        context = multiprocessing.get_context(
            "fork" if "fork" in multiprocessing.get_all_start_methods() else None
        )
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(target=_flock_child, args=(str(probe_path), child_connection))
        process.start()
        child_connection.close()
        blocked = parent_connection.poll(5.0) and bool(parent_connection.recv())
        process.join(5.0)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
        return bool(blocked and process.exitcode == 0)
    except (OSError, EOFError):
        return False
    finally:
        if parent_connection is not None:
            parent_connection.close()
        if descriptor is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        try:
            probe_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _positive_free_bytes(value: float | int | str) -> int:
    """將最低可用 GiB 轉為正整數 bytes，拒絕 NaN、無限值與零。"""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StoragePolicyError("minimum_free_gib_invalid") from exc
    if not math.isfinite(number) or number <= 0:
        raise StoragePolicyError("minimum_free_gib_invalid")
    return math.ceil(number * 1024**3)


def _root_state(
    label: str,
    *,
    fstype: str | None,
    source: str | None,
    free_bytes: int | None,
    issues: Sequence[StorageIssue],
) -> StorageRootState:
    """建立不含路徑的 root snapshot state。"""

    failed = any(issue.label == label for issue in issues)
    return StorageRootState(
        label=label,
        fstype=fstype,
        source_token_hash=_source_token_hash(source) if source else None,
        free_bytes=free_bytes,
        gate_status="FAIL" if failed else "PASS",
    )


def validate_storage_policy(
    *,
    project_root: PathLike,
    project_venv: PathLike,
    result_nfs_root: PathLike,
    execution_package_root: PathLike,
    output_root: PathLike,
    scratch_root: PathLike,
    checkpoint_root: PathLike,
    uv_cache_root: PathLike,
    mpl_cache_root: PathLike,
    xdg_cache_root: PathLike,
    tmp_root: PathLike,
    minimum_free_gib: float | int | str,
    mount_probe: MountProbe | None = None,
    write_probe: WriteProbe | None = None,
    flock_probe: FlockProbe | None = None,
    disk_usage: DiskUsage = shutil.disk_usage,
) -> dict[str, object]:
    """驗證整個 SERVER 儲存部署並回傳 machine-readable gate snapshot。

    ``result_nfs_root`` 下的八個 execution roots 必須是已存在、可寫且不含符號
    連結元件的嚴格後代；它們各自需通過同一 NFS mount/source、空間、寫入探針，
    並由 NFS root 上的跨程序 flock probe 證明互斥語意。``project_root`` 與
    ``project_venv`` 只驗證為既有目錄且 ``project_venv`` 精確位於 project root
    的 ``.venv``，因此可合法位於 ``/home``。所有預期失敗都彙整為 code，避免
    例外文字把任何絕對路徑寫入快照。
    """

    issues: list[StorageIssue] = []

    def issue(label: str, code: str) -> None:
        """以穩定 pair 記錄 issue，避免同一條件重複污染快照。"""

        value = StorageIssue(label=label, code=code)
        if value not in issues:
            issues.append(value)

    try:
        minimum_free_bytes = _positive_free_bytes(minimum_free_gib)
    except StoragePolicyError as exc:
        minimum_free_bytes = 0
        issue("gate", exc.code)

    try:
        project_path = _existing_directory(project_root, "project_root")
    except StoragePolicyError as exc:
        project_path = None
        issue("project_root", exc.code)

    try:
        venv_path = _existing_directory(project_venv, "project_venv")
    except StoragePolicyError as exc:
        venv_path = None
        issue("project_venv", exc.code)

    if project_path is not None and venv_path is not None:
        expected_venv = project_path / ".venv"
        if venv_path != expected_venv:
            issue("project_venv", "project_venv_must_be_project_dot_venv")

    try:
        nfs_path = _existing_directory(result_nfs_root, RESULT_ROOT_LABEL)
    except StoragePolicyError as exc:
        nfs_path = None
        issue(RESULT_ROOT_LABEL, exc.code)

    raw_roots: Mapping[str, PathLike] = {
        "execution_package_root": execution_package_root,
        "output_root": output_root,
        "scratch_root": scratch_root,
        "checkpoint_root": checkpoint_root,
        "uv_cache_root": uv_cache_root,
        "mpl_cache_root": mpl_cache_root,
        "xdg_cache_root": xdg_cache_root,
        "tmp_root": tmp_root,
    }
    resolved_roots: dict[str, Path | None] = {}
    for label, raw_path in raw_roots.items():
        try:
            resolved_roots[label] = _existing_directory(raw_path, label)
        except StoragePolicyError as exc:
            resolved_roots[label] = None
            issue(label, exc.code)

    if nfs_path is not None:
        for label, resolved in resolved_roots.items():
            if resolved is not None:
                try:
                    _require_strict_descendant(resolved, nfs_path, label)
                except StoragePolicyError as exc:
                    issue(label, exc.code)

    # output、scratch 與 checkpoint 需要由不同的生命週期管理；互相包含會讓
    # 清理、quota 或 checkpoint archive 誤傷另一類成果，因此即使三者同在
    # NFS mount 也拒絕相等或巢狀 topology。其他 cache 可以依 operator 的
    # deployment layout 放在各自的 NFS 子樹，但不能改變這三個核心根的邊界。
    managed_roots = {
        label: resolved_roots.get(label)
        for label in ("output_root", "scratch_root", "checkpoint_root")
    }
    for left_index, left_label in enumerate(managed_roots):
        left_path = managed_roots[left_label]
        if left_path is None:
            continue
        for right_label in list(managed_roots)[left_index + 1 :]:
            right_path = managed_roots[right_label]
            if right_path is None:
                continue
            try:
                left_path.relative_to(right_path)
                overlap = True
            except ValueError:
                try:
                    right_path.relative_to(left_path)
                    overlap = True
                except ValueError:
                    overlap = False
            if overlap:
                issue(left_label, "output_scratch_checkpoint_overlap")
                issue(right_label, "output_scratch_checkpoint_overlap")

    probe_mount = mount_probe or _findmnt_mount_probe
    mount_records: dict[str, MountInfo | None] = {RESULT_ROOT_LABEL: None}
    if nfs_path is not None:
        try:
            mount_records[RESULT_ROOT_LABEL] = _coerce_mount_info(probe_mount(nfs_path))
        except (StoragePolicyError, OSError, ValueError):
            issue(RESULT_ROOT_LABEL, "findmnt_failed")
    for label, resolved in resolved_roots.items():
        mount_records[label] = None
        if resolved is None:
            continue
        try:
            mount_records[label] = _coerce_mount_info(probe_mount(resolved))
        except (StoragePolicyError, OSError, ValueError):
            issue(label, "findmnt_failed")

    result_mount = mount_records[RESULT_ROOT_LABEL]
    if result_mount is not None:
        if result_mount.fstype.lower() not in NFS_FILESYSTEM_TYPES:
            issue(RESULT_ROOT_LABEL, "result_root_not_nfs")
        for label, candidate_mount in mount_records.items():
            if label == RESULT_ROOT_LABEL or candidate_mount is None:
                continue
            if candidate_mount.fstype.lower() not in NFS_FILESYSTEM_TYPES:
                issue(label, "root_not_nfs")
            if (
                candidate_mount.source != result_mount.source
                or candidate_mount.target != result_mount.target
            ):
                issue(label, "mount_source_or_target_mismatch")

    free_bytes_by_label: dict[str, int | None] = {RESULT_ROOT_LABEL: None}
    fstype_by_label: dict[str, str | None] = {
        RESULT_ROOT_LABEL: result_mount.fstype if result_mount is not None else None
    }
    source_by_label: dict[str, str | None] = {
        RESULT_ROOT_LABEL: result_mount.source if result_mount is not None else None
    }
    for label, candidate_mount in mount_records.items():
        if label == RESULT_ROOT_LABEL:
            continue
        fstype_by_label[label] = candidate_mount.fstype if candidate_mount is not None else None
        source_by_label[label] = candidate_mount.source if candidate_mount is not None else None

    # 空間檢查使用每個實體 root，而不是只檢查 result root；同一 mount 上的 quota
    # 或 bind topology 仍可能讓某一個 execution root 無法建立成果檔案。
    for label, resolved in ((RESULT_ROOT_LABEL, nfs_path), *resolved_roots.items()):
        if resolved is None:
            continue
        try:
            usage = disk_usage(resolved)
            free_bytes = int(usage.free)
            free_bytes_by_label[label] = free_bytes
            if free_bytes < minimum_free_bytes:
                issue(label, "free_space_insufficient")
        except (OSError, AttributeError, TypeError, ValueError):
            issue(label, "disk_usage_failed")

    # execution roots 必須真的可寫；project root／venv 只提供 code/runtime，不
    # 會因為部署權限而被要求可寫。這保留 /home 上唯讀 code checkout 的合法性。
    for label, resolved in resolved_roots.items():
        if resolved is None:
            continue
        if not os.access(resolved, os.W_OK | os.X_OK):
            issue(label, "root_not_writable")
        try:
            if not (write_probe or _write_probe)(resolved):
                issue(label, "write_probe_failed")
        except (OSError, RuntimeError):
            issue(label, "write_probe_failed")

    if nfs_path is not None:
        try:
            if not (flock_probe or _flock_probe)(nfs_path):
                issue(RESULT_ROOT_LABEL, "flock_probe_failed")
        except (OSError, RuntimeError):
            issue(RESULT_ROOT_LABEL, "flock_probe_failed")

    root_states = [
        _root_state(
            RESULT_ROOT_LABEL,
            fstype=fstype_by_label[RESULT_ROOT_LABEL],
            source=source_by_label[RESULT_ROOT_LABEL],
            free_bytes=free_bytes_by_label[RESULT_ROOT_LABEL],
            issues=issues,
        )
    ]
    root_states.extend(
        _root_state(
            label,
            fstype=fstype_by_label.get(label),
            source=source_by_label.get(label),
            free_bytes=free_bytes_by_label.get(label),
            issues=issues,
        )
        for label in EXECUTION_ROOT_LABELS
    )
    probes = [
        {
            "label": "write_probe",
            "gate_status": "FAIL"
            if any(issue.code == "write_probe_failed" for issue in issues)
            else "PASS",
        },
        {
            "label": "same_host_flock_probe",
            "gate_status": "FAIL"
            if any(issue.code == "flock_probe_failed" for issue in issues)
            else "PASS",
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "minimum_free_bytes": minimum_free_bytes,
        "gate_status": "FAIL" if issues else "PASS",
        "roots": [state.as_dict() for state in root_states],
        "probes": probes,
        "issues": [item.as_dict() for item in issues],
    }


def _positive_gib_argument(value: str) -> float:
    """提供 argparse 的正浮點 GiB 型別，錯誤交由 argparse 回傳狀態 2。"""

    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("最低空間必須是正數 GiB") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("最低空間必須是正數 GiB")
    return number


def _parser() -> argparse.ArgumentParser:
    """建立 storage gate CLI；所有資料與快取根都必須由 operator 明示。"""

    parser = argparse.ArgumentParser(
        description="驗證 SERVER 執行成果、package 與快取只能寫入同一 NFS 結果根"
    )
    parser.add_argument("--project-root", required=True, help="可位於 /home 的 Git 專案根")
    parser.add_argument("--project-venv", required=True, help="專案根下既有 .venv")
    parser.add_argument(
        "--result-nfs-root",
        "--nfs-result-root",
        dest="result_nfs_root",
        required=True,
        help="operator 指定的 NFS 結果根",
    )
    for option, dest, help_text in (
        ("--execution-package-root", "execution_package_root", "/data 上執行 package 根"),
        ("--output-root", "output_root", "成果輸出根"),
        ("--scratch-root", "scratch_root", "執行 scratch 根"),
        ("--checkpoint-root", "checkpoint_root", "checkpoint 根"),
        ("--uv-cache-root", "uv_cache_root", "uv cache 根"),
        ("--mpl-cache-root", "mpl_cache_root", "matplotlib cache 根"),
        ("--xdg-cache-root", "xdg_cache_root", "XDG cache 根"),
        ("--tmp-root", "tmp_root", "TMPDIR 根"),
    ):
        parser.add_argument(option, dest=dest, required=True, help=help_text)
    parser.add_argument(
        "--minimum-free-gib",
        required=True,
        type=_positive_gib_argument,
        help="每一個 root 至少保留的可用空間（GiB）",
    )
    parser.add_argument(
        "--snapshot-output",
        help="將同一 machine-readable 快照原子寫入已存在的 NFS scratch parent",
    )
    return parser


def _write_snapshot(
    path_value: PathLike,
    snapshot: Mapping[str, object],
    *,
    result_nfs_root: PathLike,
    scratch_root: PathLike,
) -> None:
    """以同目錄暫存檔後 rename 保存 gate snapshot，不建立未知目錄。

    runner 在 gate 前不得 ``mkdir``，所以 operator 必須先建立 snapshot parent。
    ``result_nfs_root`` 與 ``scratch_root`` 是必要邊界；destination 的 canonical
    path 必須是 scratch 的嚴格後代，且內容由 caller 產生且不含路徑。這裡另行拒絕
    snapshot path 的 symlink，避免通過 gate 後把證據導向 ``/home`` 或其他檔案系統。
    """

    destination = _absolute_path(path_value, "snapshot_output")
    parent = _existing_directory(destination.parent, "snapshot_output_parent")
    nfs_root = _existing_directory(result_nfs_root, RESULT_ROOT_LABEL)
    _require_strict_descendant(parent, nfs_root, "snapshot_output_parent")
    scratch_path = _existing_directory(scratch_root, "scratch_root")
    # parent 可能就是 scratch root（例如 scratch/gate.json）；真正需要是
    # destination 這個檔案的 canonical path 為 scratch 的嚴格後代。
    try:
        destination_resolved = destination.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise StoragePolicyError("snapshot_output_resolve_failed") from exc
    _require_strict_descendant(destination_resolved, scratch_path, "snapshot_output")
    # destination 可以是尚未建立的新檔案；既有 parent 已由上方逐元件檢查，
    # 因此只需另外拒絕 destination 本身是 symlink，不把「檔案尚不存在」誤判。
    if destination.exists() and destination.is_symlink():
        raise StoragePolicyError("snapshot_output_symlink")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=".lbt-storage-gate-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(snapshot, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise StoragePolicyError("snapshot_output_write_failed") from exc
    finally:
        if temporary is not None and temporary.exists():
            with contextlib.suppress(OSError):
                temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """執行 CLI gate，輸出 JSON 並以 0/2 表達通過或拒絕。"""

    arguments = _parser().parse_args(argv)
    policy_arguments = vars(arguments).copy()
    snapshot_output = policy_arguments.pop("snapshot_output")
    snapshot = validate_storage_policy(**policy_arguments)
    try:
        if snapshot_output:
            _write_snapshot(
                snapshot_output,
                snapshot,
                result_nfs_root=policy_arguments["result_nfs_root"],
                scratch_root=policy_arguments["scratch_root"],
            )
    except StoragePolicyError as exc:
        # snapshot parent 本身若不可寫，原始 gate snapshot 仍印到 stdout；這裡
        # 只追加固定 code，不輸出 destination 絕對路徑。
        snapshot = dict(snapshot)
        snapshot["gate_status"] = "FAIL"
        issues = list(snapshot.get("issues", []))
        issues.append({"label": "snapshot_output", "code": exc.code})
        snapshot["issues"] = issues
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
    return 0 if snapshot.get("gate_status") == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
