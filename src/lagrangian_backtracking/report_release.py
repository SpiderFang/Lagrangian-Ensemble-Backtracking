"""report-v1 成果目錄的來源綁定、原子發布與唯讀驗證。

本模組是 renderer 與正式報告管線之間的最後 I/O 邊界。它不繪圖、不重新計算統計量，
只接受 caller 已經建立的 :class:`~lagrangian_backtracking.report_records.ReportRegistry`
與每個產品的明示 staging path，將來源 JSON、registry 及產品以固定 topology 複製成
不可變 sibling release。所有來源快照保留原始 bytes；檔案大小使用 bytes，registry
中的科學距離與網格仍是公尺（m），時間仍是秒（s）。

release 的完整性分成三層：第一層是目錄節點與 exact file inventory，第二層是每個普通
檔案的大小／SHA-256，第三層是 ReportRegistry、ReportSpec、AggregateSpec、aggregate
manifest 與 source-run 文件的語意 binding。reader 與 validator 都以 no-follow ordinary
file I/O 讀取，不能因 manifest 或 caller 提供的 path 而追隨 symbolic link，也不能用
零值、最近值或自動掃描結果補足缺少產品。

本機 synthetic engineering release 通過這些檢查，只代表檔案拓撲、provenance 與工程
資料契約成立；它不把 synthetic bytes 轉成正式 OCM schema 3、NWW3 schema 1 科學證據，
也不允許將條件式來源足跡或相對來源權重解讀為絕對來源機率、因果歸因或觀測驗證。
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Final, NoReturn
from uuid import uuid4

from .aggregate_spec import load_aggregate_spec
from .report_records import (
    REPORT_RELEASE_SCHEMA_VERSION as _REPORT_RECORDS_SCHEMA_VERSION,
)
from .report_records import (
    ReportArtifactRecord,
    ReportProductRef,
    ReportRegistry,
)
from .report_spec import REPORT_SPEC_SCHEMA_VERSION, ReportSpec

__all__ = [
    "REPORT_RELEASE_FINAL_DIRECTORY_SUFFIX",
    "REPORT_RELEASE_SCHEMA_VERSION",
    "REPORT_SOURCE_FILE_NAMES",
    "ReportRelease",
    "ReportReleaseWriter",
    "read_report_registry",
    "read_report_release",
    "validate_report_release",
    "write_report_release",
]


# 報告 release 與 registry 共用同一個版本；若兩層版本漂移，reader 不應猜測欄位語意。
REPORT_RELEASE_SCHEMA_VERSION: Final[str] = _REPORT_RECORDS_SCHEMA_VERSION
REPORT_RELEASE_FINAL_DIRECTORY_SUFFIX: Final[str] = ".report-v1"

# source/ 下的七份檔案是 report release 唯一允許的來源快照。前三份分別是 aggregate
# manifest、aggregate spec、report spec；後四份是 run 控制器的 exact source bytes。
REPORT_SOURCE_FILE_NAMES: Final[tuple[str, ...]] = (
    "aggregate_manifest.json",
    "aggregate_spec.json",
    "report_spec.json",
    "run_plan.json",
    "run_progress.json",
    "normalized_config.json",
    "input_inventory.json",
)

_AGGREGATE_SOURCE_FILE_NAMES: Final[tuple[str, ...]] = (
    "aggregate_manifest.json",
    "aggregate_spec.json",
    "source_run_plan.json",
    "source_run_progress.json",
    "source_normalized_config.json",
    "source_input_inventory.json",
)
_RUN_SOURCE_FILE_NAMES: Final[tuple[str, ...]] = (
    "run_plan.json",
    "run_progress.json",
    "normalized_config.json",
    "input_inventory.json",
)
_AGGREGATE_TO_REPORT_SOURCE_NAME: Final[dict[str, str]] = {
    "aggregate_manifest.json": "aggregate_manifest.json",
    "aggregate_spec.json": "aggregate_spec.json",
    "source_run_plan.json": "run_plan.json",
    "source_run_progress.json": "run_progress.json",
    "source_normalized_config.json": "normalized_config.json",
    "source_input_inventory.json": "input_inventory.json",
}
_ROOT_CONTROL_FILES: Final[frozenset[str]] = frozenset(
    {"report_manifest.json", "figure_registry.json", "table_registry.json"}
)
_ROOT_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {"source", "figures", "tables", "caption_sidecars", "data_sidecars"}
)
_FIGURE_DIRECTORIES: Final[frozenset[str]] = frozenset({"main", "supplement"})

# report manifest 不把自己放進 files mapping，因為它的 bytes 包含 files mapping 本身，
# 把自身 checksum 放入同一份 JSON 會形成無法封閉的自我參照。其餘所有檔案都必須列出。
_MANIFEST_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_kind",
        "experiment_case_id",
        "evidence_class",
        "registry",
        "source",
        "files",
    }
)
_SOURCE_CONTRACT_KEYS: Final[frozenset[str]] = frozenset(
    {"relative_path", "size_bytes", "sha256"}
)
_COMMON_REGISTRY_FIELDS: Final[tuple[str, ...]] = tuple(
    field.name
    for field in fields(ReportRegistry)
    if field.name not in {"figures", "tables"}
)
_REGISTRY_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {*_COMMON_REGISTRY_FIELDS, "registry_kind", "artifacts"}
)
_ARTIFACT_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    field.name for field in fields(ReportArtifactRecord)
)
_PRODUCT_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    field.name for field in fields(ReportProductRef)
)
_REGISTRY_KINDS: Final[frozenset[str]] = frozenset({"figure", "table"})

_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_ALLOWED_STAGES: Final[frozenset[str]] = frozenset(
    {"topology", "name", "manifest", "checksum", "registry", "source", "products", "decoder"}
)


class _ReportReleaseValidationError(ValueError):
    """只攜帶固定 stage/reason 的內部驗證錯誤。

    公開 validator 會把這個例外轉成 JSON-safe 的 ``{"stage", "reason"}`` 物件；例外
    本身不保存原始 path、作業系統文字或第三方套件錯誤，避免 SERVER 絕對位置穿透公開
    CLI。``reason`` 是固定 token，不用動態檔名或輸入內容組成。
    """

    _REASONS: Final[frozenset[str]] = frozenset(
        {
            "root_not_directory",
            "fixed_node_set_mismatch",
            "symbolic_link_or_non_regular_node",
            "manifest_schema_mismatch",
            "manifest_not_canonical",
            "manifest_inventory_mismatch",
            "final_name_mismatch",
            "hash_mismatch",
            "registry_schema_mismatch",
            "registry_closure_mismatch",
            "source_schema_mismatch",
            "source_hash_mismatch",
            "source_binding_mismatch",
            "product_schema_mismatch",
            "unexpected_failure",
        }
    )

    def __init__(self, stage: str, reason: str) -> None:
        if stage not in _ALLOWED_STAGES or reason not in self._REASONS:
            raise ValueError("report release stage/reason 未登錄")
        self.stage = stage
        self.reason = reason
        super().__init__(stage, reason)


class _ReportReleaseFinalExistsError(FileExistsError):
    """標記 writer 的 final collision，與 partial 內部 race 分開處理。"""


def _fail(stage: str, reason: str) -> NoReturn:
    """建立不含路徑的固定內部失敗。"""

    raise _ReportReleaseValidationError(stage, reason)


def _sha256_bytes(raw_bytes: bytes) -> str:
    """以串流無關的 immutable bytes 計算小寫 SHA-256。"""

    if type(raw_bytes) is not bytes:
        raise ValueError("bytes 型別不符")
    return hashlib.sha256(raw_bytes).hexdigest()


def _canonical_json_bytes(document: object, *, newline: bool = True) -> bytes:
    """以 compact、排序鍵名、UTF-8 且禁止非有限數值的規則序列化 JSON。

    manifest 與兩份 registry 使用單一尾端換行，讓檔案能以文字工具穩定顯示；source
    snapshot 則保留 caller 原始 bytes，不會經過這個函式重新排版。
    """

    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except Exception as error:
        raise ValueError("JSON canonicalization 失敗") from error
    return encoded + (b"\n" if newline else b"")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒絕所有層級的 duplicate key，避免後值靜默覆蓋前值。"""

    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("JSON duplicate key")
        document[key] = value
    return document


def _reject_json_constant(_token: str) -> NoReturn:
    """拒絕 JSON decoder 額外接受的 NaN、Infinity 與 -Infinity。"""

    raise ValueError("JSON non-finite constant")


def _reject_nonfinite_json_numbers(value: object) -> None:
    """遞迴拒絕極大指數解析後形成的 infinity。"""

    if type(value) is float and not math.isfinite(value):
        raise ValueError("JSON non-finite number")
    if type(value) is dict:
        for nested in value.values():
            _reject_nonfinite_json_numbers(nested)
    elif type(value) is list:
        for nested in value:
            _reject_nonfinite_json_numbers(nested)


def _strict_json_object(raw_bytes: bytes) -> dict[str, object]:
    """將 exact UTF-8 JSON bytes 解析成根節點普通 dict。

    report release 的 source、registry 與 manifest 都是 object；拒絕 list/scalar root、
    duplicate key、非有限值與錯誤 UTF-8，避免 reader 以寬鬆 coercion 產生另一份 provenance。
    """

    if type(raw_bytes) is not bytes:
        raise ValueError("JSON bytes 型別不符")
    document = json.loads(
        raw_bytes.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )
    _reject_nonfinite_json_numbers(document)
    if type(document) is not dict:
        raise ValueError("JSON root 必須是 object")
    return document


def _require_sha256(value: object) -> str:
    """驗證固定格式的 64 碼小寫 SHA-256。"""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError("SHA-256 contract 不符")
    return value


def _require_size(value: object) -> int:
    """驗證不含 bool/NumPy scalar 的非負 Python bytes 計數。"""

    if type(value) is not int or value < 0:
        raise ValueError("size_bytes contract 不符")
    return value


def _require_relative_path(value: object) -> str:
    """驗證 release 內部使用的安全 POSIX relative path。"""

    if type(value) is not str or not value or value != value.strip():
        raise ValueError("relative path contract 不符")
    if value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError("relative path contract 不符")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("relative path component 不符")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("relative path control character")
    return value


def _validate_product_topology(relative_path: str) -> None:
    """限制產品只落在固定 figures/tables/sidecars 目錄，不允許任意子目錄。

    ``figures/main`` 與 ``figures/supplement`` 是計畫文件登錄的兩個固定出版層；產品
    必須放在其中一層，不能放在 figures root。其他三種產品目錄只接受直接檔案，避免
    caller 用深層 path 隱藏未登錄的目錄拓撲。
    """

    parts = relative_path.split("/")
    if (
        parts[0] == "figures"
        and len(parts) == 3
        and parts[1] in _FIGURE_DIRECTORIES
    ) or (
        parts[0] in {"tables", "caption_sidecars", "data_sidecars"}
        and len(parts) == 2
    ):
        return
    raise ValueError("product path 不在固定 topology")


def _lstat_regular_directory(path: Path) -> os.stat_result:
    """以 lstat 要求 ordinary non-symlink directory，拒絕 symbolic link 目錄。"""

    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise ValueError("directory node 不符")
    return node


def _open_regular_file(path: Path) -> int:
    """以 O_NOFOLLOW/fstat 開啟 ordinary regular file，不追隨最後一層 symlink。"""

    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("file node 不符")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        after = os.fstat(descriptor)
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("file identity 不符")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_file_bytes(path: Path) -> bytes:
    """使用 no-follow descriptor 讀取 exact regular-file bytes。"""

    descriptor = _open_regular_file(path)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _regular_file_size_and_sha256(path: Path) -> tuple[int, str]:
    """以 no-follow descriptor 串流計算普通檔大小與 SHA-256。"""

    descriptor = _open_regular_file(path)
    size_bytes = 0
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            size_bytes += len(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return size_bytes, digest.hexdigest()


def _write_exclusive_durable_bytes(path: Path, raw_bytes: bytes) -> None:
    """以 O_EXCL 建立檔案、寫 exact bytes、flush 並 fsync。

    exclusive create 同時拒絕既有普通檔、symbolic link 與 broken link；partial 內任何
    目標若被其他程序搶先建立，writer 會 fail closed，而不會覆寫對方 bytes。
    """

    if type(raw_bytes) is not bytes:
        raise ValueError("durable bytes 型別不符")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("exclusive target 非普通檔")
        offset = 0
        while offset < len(raw_bytes):
            written = os.write(descriptor, raw_bytes[offset:])
            if written <= 0:
                raise OSError("durable write 未完成")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """以不追隨符號連結的目錄檔案描述元（no-follow directory descriptor）同步普通目錄。

    ``expected_identity`` 只在 final 目錄重新命名後使用，用來確保耐久性同步
    （durability fsync）仍然指向完成發布的同一個 parent 索引節點（inode）；若 parent 在 rename
    後被替換，呼叫端會把這視為「已發布但耐久性未確認」，保留 final 而不清理任何
    partial。
    """

    parent_status = _lstat_regular_directory(path)
    if expected_identity is not None and (
        parent_status.st_dev,
        parent_status.st_ino,
    ) != expected_identity:
        raise ValueError("directory identity changed")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened_status.st_mode)
            or expected_identity is not None
            and (opened_status.st_dev, opened_status.st_ino) != expected_identity
        ):
            raise ValueError("directory descriptor 非目錄")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _ReportReleaseAtomicRenameUnsupported(RuntimeError):
    """目前平台、C 標準函式庫（libc）或檔案系統不保證原子拒覆寫目錄重新命名。"""


class _ReportReleaseAtomicRenameFailure(RuntimeError):
    """原子拒覆寫目錄重新命名後端回報無法安全判定的錯誤。"""


# Linux 核心使用者空間介面（UAPI）``<linux/fs.h>`` 的 RENAME_NOREPLACE 是 bit 0；
# Darwin 軟體開發套件（SDK）``sys/stdio.h`` 的 RENAME_EXCL 是 0x00000004。這些是
# 傳給 C 標準函式庫的旗標，不是系統呼叫編號（syscall number），因此不依賴會隨核心
# 或 CPU 架構變動的硬編碼系統呼叫編號。
_LINUX_RENAME_NOREPLACE: Final[int] = 1 << 0
_DARWIN_RENAME_EXCL: Final[int] = 0x00000004
_UNSUPPORTED_RENAME_ERRNOS: Final[frozenset[int]] = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }
)
# 這組錯誤碼代表旗標、函式或檔案系統不提供可驗證的拒覆寫語意；遇到它們時只能
# 清理本次自有暫存目錄，不能改用一般 rename 造成覆寫風險。
# Linux renameat2 與 Darwin renameatx_np 共用的 C 函式簽章：兩個目錄檔案描述元、
# 兩個以 NUL 結尾的檔名，以及一個無號整數旗標。明示型別可避免 ctypes 依平台預設
# 推斷參數寬度，並把原生錯誤碼保留給後續分類。
_RENAMEAT_ARGTYPES: Final[tuple[object, ...]] = (
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint,
)


def _load_exclusive_rename_backend() -> tuple[Callable[..., int], int]:
    """載入平台的原子拒覆寫目錄重新命名函式（exclusive directory rename）。

    Linux 使用 C 標準函式庫的 ``renameat2(..., RENAME_NOREPLACE)``，Darwin 使用
    軟體開發套件宣告的 ``renameatx_np(..., RENAME_EXCL)``。兩者都以同一個已驗證
    parent 目錄檔案描述元（dirfd）作為來源與目的目錄，因此不會在 rename 期間重新
    解析可被替換的 parent path。其他平台、缺少符號或無法設定應用程式二進位介面
    （ABI）時一律安全失敗（fail closed）；絕不退回 ``os.replace``、``os.rename``
    或逐檔複製。
    """

    if sys.platform.startswith("linux"):
        function_name = "renameat2"
        rename_flags = _LINUX_RENAME_NOREPLACE
    elif sys.platform == "darwin":
        function_name = "renameatx_np"
        rename_flags = _DARWIN_RENAME_EXCL
    else:
        raise _ReportReleaseAtomicRenameUnsupported("平台未提供 exclusive rename")

    try:
        # CDLL(None) 取得目前程序已載入的 libc；use_errno=True 讓 ctypes 在
        # thread-local errno 與 Python 之間保留 native rename 的失敗原因。
        libc = ctypes.CDLL(None, use_errno=True)
        rename_function = getattr(libc, function_name)
        rename_function.argtypes = list(_RENAMEAT_ARGTYPES)
        rename_function.restype = ctypes.c_int
    except AttributeError as error:
        raise _ReportReleaseAtomicRenameUnsupported("libc 未提供 exclusive rename") from error
    except Exception as error:
        raise _ReportReleaseAtomicRenameFailure("exclusive rename ABI 初始化失敗") from error
    return rename_function, rename_flags


def _raise_exclusive_rename_errno(error_number: int | None) -> NoReturn:
    """把 C 標準函式庫錯誤碼（errno）轉成名稱衝突（collision）、不支援或安全失敗錯誤。"""

    if error_number == errno.EEXIST:
        # RENAME_NOREPLACE/RENAME_EXCL 對任何既有 destination node（檔案、目錄、
        # symbolic link）都必須回 EEXIST；write() 會轉成不含路徑的 FileExistsError。
        raise _ReportReleaseFinalExistsError("report release final 已存在")
    if error_number in _UNSUPPORTED_RENAME_ERRNOS:
        raise _ReportReleaseAtomicRenameUnsupported("檔案系統不支援 exclusive rename")
    raise _ReportReleaseAtomicRenameFailure("exclusive rename backend 失敗")


def _call_exclusive_rename(
    rename_function: Callable[..., int],
    *,
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    rename_flags: int,
) -> None:
    """以固定目錄檔案描述元（dirfd）呼叫 C 函式，並封閉原生錯誤碼語意。"""

    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    # use_errno=True 會把 C 標準函式庫在目前執行緒設定的 errno 複製到 ctypes
    # 的執行緒區域；先清零可避免 mock 或異常回傳時誤讀上一個 native 呼叫的錯誤。
    ctypes.set_errno(0)
    try:
        result = rename_function(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            rename_flags,
        )
    except OSError as error:
        _raise_exclusive_rename_errno(error.errno)
    # 原生函式以 0 表示成功、非 0 表示失敗；失敗原因只能由同一執行緒的
    # ctypes.get_errno() 取得，不能用 Python 的 exists 檢查取代原子操作。
    if result != 0:
        _raise_exclusive_rename_errno(ctypes.get_errno())


def _atomic_exclusive_directory_rename(
    source: Path,
    destination: Path,
    *,
    expected_parent_identity: tuple[int, int],
) -> None:
    """以同父目錄檔案描述元原子地搬移自有暫存目錄（partial）到 final 名稱。

    ``renameat2(RENAME_NOREPLACE)`` 與 ``renameatx_np(RENAME_EXCL)`` 把「目的名稱
    不存在」與「目錄項目交換」合併成一個檔案系統操作；因此即使另一程序在最後
    lstat 後建立 final 目錄、檔案或符號連結，也只會得到 EEXIST，對方 inode／內容／
    符號連結目標（link target）不會被覆寫。parent 先以 O_DIRECTORY/O_NOFOLLOW 開啟並
    核對裝置與索引節點（inode），避免路徑競態（path race）將操作導向另一個目錄。
    後端不可用或回報其他錯誤碼時直接安全失敗。
    """

    source = Path(source)
    destination = Path(destination)
    if source.parent != destination.parent or not source.name or not destination.name:
        raise ValueError("exclusive rename 必須使用同父 basename")

    rename_function, rename_flags = _load_exclusive_rename_backend()
    directory_flags = os.O_RDONLY
    try:
        directory_flags |= os.O_DIRECTORY
        directory_flags |= os.O_NOFOLLOW
    except AttributeError as error:
        raise _ReportReleaseAtomicRenameUnsupported("平台缺少安全 directory open 旗標") from error

    parent_descriptor: int | None = None
    try:
        parent_descriptor = os.open(source.parent, directory_flags)
        parent_status = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_status.st_mode)
            or (parent_status.st_dev, parent_status.st_ino) != expected_parent_identity
        ):
            raise ValueError("final parent identity changed")
        _call_exclusive_rename(
            rename_function,
            parent_descriptor=parent_descriptor,
            source_name=source.name,
            destination_name=destination.name,
            rename_flags=rename_flags,
        )
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """建立 JSON-ready mapping 的深層副本，隔離 caller 後續修改。"""

    return deepcopy(dict(value))


def _normalize_source_key(value: object) -> str:
    """把 caller source mapping 的固定別名轉成 report source logical filename。"""

    if type(value) is not str:
        raise ValueError("source key 必須是 str")
    aliases = {
        "aggregate_manifest.json": "aggregate_manifest.json",
        "source/aggregate_manifest.json": "aggregate_manifest.json",
        "aggregate_spec.json": "aggregate_spec.json",
        "source/aggregate_spec.json": "aggregate_spec.json",
        "report_spec.json": "report_spec.json",
        "source/report_spec.json": "report_spec.json",
        "run_plan.json": "run_plan.json",
        "source/run_plan.json": "run_plan.json",
        "source_run_plan.json": "run_plan.json",
        "run_progress.json": "run_progress.json",
        "source/run_progress.json": "run_progress.json",
        "source_run_progress.json": "run_progress.json",
        "normalized_config.json": "normalized_config.json",
        "source/normalized_config.json": "normalized_config.json",
        "source_normalized_config.json": "normalized_config.json",
        "input_inventory.json": "input_inventory.json",
        "source/input_inventory.json": "input_inventory.json",
        "source_input_inventory.json": "input_inventory.json",
    }
    if value not in aliases:
        raise ValueError("source key 不在固定集合")
    return aliases[value]


def _path_value(value: object) -> Path:
    """要求 caller 提供可由 Path 建立的 path-like 值，不接受 bytes。"""

    if isinstance(value, (str, Path)):
        return Path(value)
    raise TypeError("path 必須是 str 或 Path")


def _validate_source_mapping_paths(
    *,
    registry: ReportRegistry,
    source_run_root: str | Path | None,
    aggregate_release_root: str | Path | None,
    report_spec_path: str | Path | None,
    config_path: str | Path | None,
    source_paths: Mapping[str, object] | None,
) -> dict[str, Path]:
    """組合 caller 明示的 source paths 與兩種既有 release root。

    aggregate root 提供 aggregate manifest/spec 與四份 ``source_*`` JSON；source run root
    提供不帶 prefix 的四份 run JSON。兩者稍後以 exact bytes 比較，確保報告不能把另一個
    run 的設定或 progress 拼接進來。``source_paths`` 可逐項覆蓋，但每個 value 仍須是
    caller 明示的 ordinary file path；這個 helper 不掃描目錄找「看起來相近」的檔案。
    """

    result: dict[str, Path] = {}
    if source_paths is not None:
        if not isinstance(source_paths, Mapping):
            raise ValueError("source_paths 必須是 mapping")
        for raw_key, raw_path in source_paths.items():
            logical_name = _normalize_source_key(raw_key)
            if logical_name in result:
                raise ValueError("source_paths key 重複")
            result[logical_name] = _path_value(raw_path)

    aggregate_root = None if aggregate_release_root is None else _path_value(aggregate_release_root)
    run_root = None if source_run_root is None else _path_value(source_run_root)
    if aggregate_root is not None:
        _lstat_regular_directory(aggregate_root)
        aggregate_paths = {
            "aggregate_manifest.json": aggregate_root / "aggregate_manifest.json",
            "aggregate_spec.json": aggregate_root / "aggregate_spec.json",
            "run_plan.json": aggregate_root / "source_run_plan.json",
            "run_progress.json": aggregate_root / "source_run_progress.json",
            "normalized_config.json": aggregate_root / "source_normalized_config.json",
            "input_inventory.json": aggregate_root / "source_input_inventory.json",
        }
        for logical_name, path in aggregate_paths.items():
            result.setdefault(logical_name, path)
    if run_root is not None:
        _lstat_regular_directory(run_root)
        if run_root.name != registry.run_id:
            raise ValueError("source run basename 與 registry run_id 不一致")
        for logical_name in _RUN_SOURCE_FILE_NAMES:
            result.setdefault(logical_name, run_root / logical_name)
    if report_spec_path is not None:
        result["report_spec.json"] = _path_value(report_spec_path)
    if config_path is not None:
        result["normalized_config.json"] = _path_value(config_path)

    if set(result) != set(REPORT_SOURCE_FILE_NAMES):
        raise ValueError("source paths 必須恰好涵蓋七份 source snapshot")
    return result


def _read_optional_root_file(root: Path, name: str) -> bytes | None:
    """讀取存在的固定 root file；不存在時回傳 None，其他節點錯誤則 fail closed。"""

    path = root / name
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    return _read_regular_file_bytes(path)


def _report_spec_from_bytes(raw_bytes: bytes) -> ReportSpec:
    """由 report spec exact bytes 建立帶 raw/canonical hash 的 immutable ReportSpec。

    report spec 的兩個 hash 是 loader 衍生欄位，不應出現在輸入 JSON。這裡重用既有
    ReportSpec constructor 的所有公尺／秒、quantile、代表軌跡與 renderer policy 驗證，
    只在記憶體中重建，不重新排版或覆寫 caller source file。
    """

    document = _strict_json_object(raw_bytes)
    expected_keys = {
        "schema_version",
        "run_id",
        "aggregate_spec_canonical_sha256",
        "primary_kde_bandwidth_m",
        "minimum_kde_raw_count",
        "low_sample_min_member_count",
        "vertical_depth_bin_edges_m",
        "representative_trajectory_count_per_site",
        "representative_selection_policy",
        "representative_selection_seed",
        "travel_age_quantiles",
        "pathway_first_passage_quantiles",
        "figure_formats",
        "raster_dpi",
        "renderer_style_version",
        "language",
    }
    if set(document) != expected_keys:
        raise ValueError("report spec key set 不符")
    sequence_keys = (
        "vertical_depth_bin_edges_m",
        "travel_age_quantiles",
        "pathway_first_passage_quantiles",
        "figure_formats",
    )
    if any(type(document[key]) is not list for key in sequence_keys):
        raise ValueError("report spec sequence 型別不符")

    raw_hash = _sha256_bytes(raw_bytes)
    provisional = ReportSpec(
        schema_version=document["schema_version"],
        run_id=document["run_id"],
        aggregate_spec_canonical_sha256=document["aggregate_spec_canonical_sha256"],
        primary_kde_bandwidth_m=document["primary_kde_bandwidth_m"],
        minimum_kde_raw_count=document["minimum_kde_raw_count"],
        low_sample_min_member_count=document["low_sample_min_member_count"],
        vertical_depth_bin_edges_m=tuple(document["vertical_depth_bin_edges_m"]),
        representative_trajectory_count_per_site=document[
            "representative_trajectory_count_per_site"
        ],
        representative_selection_policy=document["representative_selection_policy"],
        representative_selection_seed=document["representative_selection_seed"],
        travel_age_quantiles=tuple(document["travel_age_quantiles"]),
        pathway_first_passage_quantiles=tuple(document["pathway_first_passage_quantiles"]),
        figure_formats=tuple(document["figure_formats"]),
        raster_dpi=document["raster_dpi"],
        renderer_style_version=document["renderer_style_version"],
        language=document["language"],
        source_sha256=raw_hash,
        canonical_sha256="0" * 64,
    )
    canonical_payload = provisional.to_dict()
    canonical_payload.pop("source_sha256")
    canonical_payload.pop("canonical_sha256")
    canonical_hash = _sha256_bytes(_canonical_json_bytes(canonical_payload, newline=False))
    return ReportSpec(
        schema_version=provisional.schema_version,
        run_id=provisional.run_id,
        aggregate_spec_canonical_sha256=provisional.aggregate_spec_canonical_sha256,
        primary_kde_bandwidth_m=provisional.primary_kde_bandwidth_m,
        minimum_kde_raw_count=provisional.minimum_kde_raw_count,
        low_sample_min_member_count=provisional.low_sample_min_member_count,
        vertical_depth_bin_edges_m=provisional.vertical_depth_bin_edges_m,
        representative_trajectory_count_per_site=provisional.representative_trajectory_count_per_site,
        representative_selection_policy=provisional.representative_selection_policy,
        representative_selection_seed=provisional.representative_selection_seed,
        travel_age_quantiles=provisional.travel_age_quantiles,
        pathway_first_passage_quantiles=provisional.pathway_first_passage_quantiles,
        figure_formats=provisional.figure_formats,
        raster_dpi=provisional.raster_dpi,
        renderer_style_version=provisional.renderer_style_version,
        language=provisional.language,
        source_sha256=provisional.source_sha256,
        canonical_sha256=canonical_hash,
    )


def _mapping_if_dict(value: object) -> dict[str, object] | None:
    """將 optional JSON object 複製成普通 dict；其他型別由上層判定為 schema 錯誤。"""

    return None if value is None else (dict(value) if type(value) is dict else None)


def _compare_optional_field(document: Mapping[str, object], field_name: str, expected: object) -> None:
    """若來源文件明示 identity 欄位，要求它與已驗證 registry 值精確相等。"""

    if field_name in document and document[field_name] != expected:
        raise ValueError("source identity field mismatch")


def _validate_source_semantics(
    *,
    source_bytes: Mapping[str, bytes],
    source_paths: Mapping[str, Path] | None,
    registry: ReportRegistry,
) -> tuple[ReportSpec, Mapping[str, object]]:
    """驗證七份 source snapshot 的 JSON schema、hash 與跨文件 binding。

    ``source_bytes`` 必須先由 no-follow reader 取得，之後才進入 semantic parse；這個
    順序確保 caller 不能在 checksum gate 前以另一份文件影響 registry。若 aggregate
    manifest 是既有完整 aggregate release 的三段式 manifest，這裡核對其 metadata／files
    中與七份 source 相關的欄位；若 synthetic fixture 只保存同樣欄位的較小 object，也會
    依明示欄位核對而不猜測缺少的科學資料。
    """

    if set(source_bytes) != set(REPORT_SOURCE_FILE_NAMES):
        raise ValueError("source bytes key set 不符")
    documents: dict[str, dict[str, object]] = {}
    digests: dict[str, str] = {}
    for name in REPORT_SOURCE_FILE_NAMES:
        raw_bytes = source_bytes[name]
        if type(raw_bytes) is not bytes or not raw_bytes:
            raise ValueError("source JSON bytes 不符")
        documents[name] = _strict_json_object(raw_bytes)
        digests[name] = _sha256_bytes(raw_bytes)

    aggregate_manifest = documents["aggregate_manifest.json"]
    if set(aggregate_manifest) == {"schema_version", "metadata", "files"}:
        if aggregate_manifest["schema_version"] != "1.0.0":
            raise ValueError("aggregate manifest schema mismatch")
        if type(aggregate_manifest["metadata"]) is not dict:
            raise ValueError("aggregate metadata schema mismatch")
        if type(aggregate_manifest["files"]) is not dict:
            raise ValueError("aggregate files schema mismatch")
        if source_bytes["aggregate_manifest.json"] != _canonical_json_bytes(aggregate_manifest):
            raise ValueError("aggregate manifest not canonical")
    metadata = _mapping_if_dict(aggregate_manifest.get("metadata"))
    metadata = aggregate_manifest if metadata is None else metadata

    # aggregate root 的 files contract 只要明示，就必須逐一綁到 exact source bytes；不
    # 依它提供任意路徑，並不因 synthetic manifest 沒有完整 32 檔而放寬已明示欄位。
    aggregate_files = aggregate_manifest.get("files")
    if aggregate_files is not None:
        if type(aggregate_files) is not dict:
            raise ValueError("aggregate files schema mismatch")
        for aggregate_name, report_name in _AGGREGATE_TO_REPORT_SOURCE_NAME.items():
            contract = aggregate_files.get(aggregate_name)
            if contract is None:
                continue
            if type(contract) is not dict:
                raise ValueError("aggregate source contract schema mismatch")
            if (
                contract.get("size_bytes") != len(source_bytes[report_name])
                or contract.get("sha256") != digests[report_name]
            ):
                raise ValueError("aggregate source contract mismatch")

    aggregate_spec_path = None if source_paths is None else source_paths.get("aggregate_spec.json")
    if aggregate_spec_path is None:
        raise ValueError("aggregate spec source path 缺少")
    aggregate_spec = load_aggregate_spec(aggregate_spec_path)
    if aggregate_spec.source_sha256 != digests["aggregate_spec.json"]:
        raise ValueError("aggregate spec source hash mismatch")

    report_spec = _report_spec_from_bytes(source_bytes["report_spec.json"])
    if report_spec.schema_version != REPORT_SPEC_SCHEMA_VERSION:
        raise ValueError("report spec schema mismatch")
    if report_spec.run_id != registry.run_id:
        raise ValueError("report spec run binding mismatch")
    if report_spec.aggregate_spec_canonical_sha256 != aggregate_spec.canonical_sha256:
        raise ValueError("report spec aggregate binding mismatch")
    if aggregate_spec.run_id != registry.run_id:
        raise ValueError("aggregate spec run binding mismatch")

    # registry 保存的是 aggregate manifest/raw run bytes 與 normalized config canonical
    # hash；這些欄位不接受由檔名或 report spec 推導的替代值。
    expected_registry_hashes = {
        "aggregate_manifest_sha256": digests["aggregate_manifest.json"],
        "source_run_plan_sha256": digests["run_plan.json"],
        "source_run_progress_sha256": digests["run_progress.json"],
    }
    for field_name, expected in expected_registry_hashes.items():
        if getattr(registry, field_name) != expected:
            raise ValueError("registry source hash mismatch")
    normalized_config = documents["normalized_config.json"]
    config_canonical = _canonical_json_bytes(normalized_config, newline=False)
    if _sha256_bytes(config_canonical) != registry.config_hash:
        raise ValueError("normalized config canonical hash mismatch")

    run_plan = documents["run_plan.json"]
    run_progress = documents["run_progress.json"]
    for document in (run_plan, run_progress):
        _compare_optional_field(document, "run_id", registry.run_id)
        _compare_optional_field(document, "run_kind", registry.run_kind)
        _compare_optional_field(document, "experiment_case_id", registry.experiment_case_id)
    _compare_optional_field(run_plan, "config_hash", registry.config_hash)
    _compare_optional_field(
        run_plan,
        "checkpoint_input_binding_hash",
        registry.checkpoint_input_binding_hash,
    )
    if "run_lifecycle" in run_progress and run_progress["run_lifecycle"] != "COMPLETE":
        raise ValueError("run progress is not complete")

    plan_files = run_plan.get("files")
    if plan_files is not None:
        if type(plan_files) is not dict:
            raise ValueError("run plan files schema mismatch")
        for plan_name, report_name in (
            ("normalized_config.json", "normalized_config.json"),
            ("input_inventory.json", "input_inventory.json"),
        ):
            contract = plan_files.get(plan_name)
            if contract is None:
                continue
            if type(contract) is not dict:
                raise ValueError("run plan file contract schema mismatch")
            if (
                contract.get("size_bytes") != len(source_bytes[report_name])
                or contract.get("sha256") != digests[report_name]
            ):
                raise ValueError("run plan file contract mismatch")

    # aggregate metadata 與 registry/source bytes 的交叉檢查。metadata 缺少某欄時不捏造
    # 值；欄位一旦明示則必須精確相等，維持 synthetic 與正式 aggregate manifest 的同一
    # binding 方向。
    for field_name, expected in (
        ("run_id", registry.run_id),
        ("run_kind", registry.run_kind),
        ("experiment_case_id", registry.experiment_case_id),
        ("config_hash", registry.config_hash),
        ("checkpoint_input_binding_hash", registry.checkpoint_input_binding_hash),
        ("source_run_plan_sha256", digests["run_plan.json"]),
        ("source_run_progress_sha256", digests["run_progress.json"]),
        ("source_normalized_config_sha256", digests["normalized_config.json"]),
        ("source_input_inventory_sha256", digests["input_inventory.json"]),
        ("aggregate_spec_source_sha256", digests["aggregate_spec.json"]),
        ("aggregate_spec_canonical_sha256", aggregate_spec.canonical_sha256),
    ):
        _compare_optional_field(metadata, field_name, expected)
    return report_spec, MappingProxyType(documents["aggregate_manifest.json"])


def _registry_document(registry: ReportRegistry, *, registry_kind: str) -> dict[str, object]:
    """把 immutable ReportRegistry 的指定 F/T 半部序列化成固定 registry JSON。"""

    if registry_kind not in _REGISTRY_KINDS:
        raise ValueError("registry kind 不符")
    source = registry.to_dict()
    artifacts_key = "figures" if registry_kind == "figure" else "tables"
    artifacts = source.pop(artifacts_key)
    source.pop("figures", None)
    source.pop("tables", None)
    source["registry_kind"] = registry_kind
    source["artifacts"] = artifacts
    return source


def _artifact_from_dict(document: object) -> ReportArtifactRecord:
    """由 registry artifact plain dict 重建 ReportArtifactRecord，不做型別轉換。

    artifact 與其中的 product row 都是公開 JSON 資料契約的一部分；不能只取建構子
    需要的欄位，否則 unknown key 會被靜默捨棄、讓簽名過的 registry bytes 與 reader
    實際採用的語意不一致。因此這裡先以 dataclass 欄位建立 exact-key gate，再交給
    ``ReportArtifactRecord``／``ReportProductRef`` 執行既有欄位值與 closure 驗證。
    """

    if type(document) is not dict or set(document) != _ARTIFACT_ROOT_KEYS:
        raise ValueError("artifact document schema 不符")
    product_documents = document.get("products")
    if type(product_documents) is not list:
        raise ValueError("artifact products schema 不符")
    products_list: list[ReportProductRef] = []
    for product in product_documents:
        if type(product) is not dict or set(product) != _PRODUCT_ROOT_KEYS:
            raise ValueError("product document schema 不符")
        products_list.append(ReportProductRef(**product))
    products = tuple(products_list)
    fields_to_read = (
        "artifact_id",
        "artifact_kind",
        "title_zh",
        "status",
        "evidence_class",
        "input_sha256",
        "raw_sample_count",
        "denominator_name",
        "denominator_count",
        "units",
        "crs_by_site",
        "limitations",
        "unavailable_reason_code",
        "unavailable_reason_zh",
        "component_status",
    )
    values = {field_name: document[field_name] for field_name in fields_to_read}
    values["products"] = products
    return ReportArtifactRecord(**values)


def _registry_from_documents(
    figure_document: object,
    table_document: object,
) -> ReportRegistry:
    """驗證兩份 registry exact schema 並重建 ReportRegistry closure。"""

    if type(figure_document) is not dict or type(table_document) is not dict:
        raise ValueError("registry root 必須是 object")
    if set(figure_document) != _REGISTRY_ROOT_KEYS or set(table_document) != _REGISTRY_ROOT_KEYS:
        raise ValueError("registry root key set mismatch")
    if figure_document["registry_kind"] != "figure" or table_document["registry_kind"] != "table":
        raise ValueError("registry kind mismatch")
    common_values: dict[str, object] = {}
    for field_name in _COMMON_REGISTRY_FIELDS:
        if figure_document[field_name] != table_document[field_name]:
            raise ValueError("figure/table registry common metadata mismatch")
        common_values[field_name] = figure_document[field_name]
    raw_figures = figure_document["artifacts"]
    raw_tables = table_document["artifacts"]
    if type(raw_figures) is not list or type(raw_tables) is not list:
        raise ValueError("registry artifacts 必須是 list")
    figures = tuple(_artifact_from_dict(item) for item in raw_figures)
    tables = tuple(_artifact_from_dict(item) for item in raw_tables)
    return ReportRegistry(figures=figures, tables=tables, **common_values)


def _contract_for_source(raw_bytes: bytes) -> dict[str, object]:
    """建立 source JSON 的 exact manifest contract。"""

    return {"kind": "source_json", "size_bytes": len(raw_bytes), "sha256": _sha256_bytes(raw_bytes)}


def _contract_for_registry(raw_bytes: bytes) -> dict[str, object]:
    """建立 registry JSON 的 exact manifest contract。"""

    return {"kind": "registry_json", "size_bytes": len(raw_bytes), "sha256": _sha256_bytes(raw_bytes)}


def _contract_for_product(
    product: ReportProductRef,
    *,
    artifact_id: str,
) -> dict[str, object]:
    """由已驗證 ReportProductRef 建立含 artifact identity 的 manifest contract。"""

    return {
        "kind": "product",
        "artifact_id": artifact_id,
        "role": product.role,
        "media_type": product.media_type,
        "size_bytes": product.size_bytes,
        "sha256": product.sha256,
    }


def _build_manifest_document(
    *,
    registry: ReportRegistry,
    source_bytes: Mapping[str, bytes],
    figure_registry_bytes: bytes,
    table_registry_bytes: bytes,
) -> dict[str, object]:
    """依固定順序建立 report_manifest，完整列出 source、registry 與產品 inventory。"""

    source_contracts: dict[str, dict[str, object]] = {}
    files: dict[str, dict[str, object]] = {}
    for name in REPORT_SOURCE_FILE_NAMES:
        contract = _contract_for_source(source_bytes[name])
        source_contracts[name] = {
            "relative_path": f"source/{name}",
            "size_bytes": contract["size_bytes"],
            "sha256": contract["sha256"],
        }
        files[f"source/{name}"] = contract

    registry_contracts = {
        "figure_registry.json": _contract_for_registry(figure_registry_bytes),
        "table_registry.json": _contract_for_registry(table_registry_bytes),
    }
    files.update(registry_contracts)
    for artifact in (*registry.figures, *registry.tables):
        for product in artifact.products:
            _validate_product_topology(product.relative_path)
            files[product.relative_path] = _contract_for_product(
                product,
                artifact_id=artifact.artifact_id,
            )

    return {
        "schema_version": registry.schema_version,
        "run_id": registry.run_id,
        "run_kind": registry.run_kind,
        "experiment_case_id": registry.experiment_case_id,
        "evidence_class": registry.evidence_class,
        "registry": {
            name: {
                "relative_path": name,
                "size_bytes": contract["size_bytes"],
                "sha256": contract["sha256"],
            }
            for name, contract in registry_contracts.items()
        },
        "source": source_contracts,
        "files": files,
    }


def _validate_manifest_document(
    document: object,
) -> dict[str, dict[str, object]]:
    """驗證 report_manifest schema、source mapping 與每個 inventory contract 宣告。"""

    if type(document) is not dict or set(document) != _MANIFEST_ROOT_KEYS:
        _fail("manifest", "manifest_schema_mismatch")
    if document["schema_version"] != REPORT_RELEASE_SCHEMA_VERSION:
        _fail("manifest", "manifest_schema_mismatch")
    for name in ("run_id", "run_kind", "experiment_case_id", "evidence_class"):
        if type(document[name]) is not str or not document[name]:
            _fail("manifest", "manifest_schema_mismatch")

    source_document = document["source"]
    if type(source_document) is not dict or set(source_document) != set(REPORT_SOURCE_FILE_NAMES):
        _fail("manifest", "manifest_schema_mismatch")
    contracts: dict[str, dict[str, object]] = {}
    for name in REPORT_SOURCE_FILE_NAMES:
        value = source_document[name]
        expected_relative_path = f"source/{name}"
        if (
            type(value) is not dict
            or set(value) != _SOURCE_CONTRACT_KEYS
            or value["relative_path"] != expected_relative_path
        ):
            _fail("manifest", "manifest_schema_mismatch")
        try:
            size_bytes = _require_size(value["size_bytes"])
            digest = _require_sha256(value["sha256"])
        except Exception:
            _fail("manifest", "manifest_schema_mismatch")
        contracts[expected_relative_path] = {
            "kind": "source_json",
            "size_bytes": size_bytes,
            "sha256": digest,
        }

    registry_document = document["registry"]
    if type(registry_document) is not dict or set(registry_document) != {
        "figure_registry.json",
        "table_registry.json",
    }:
        _fail("manifest", "manifest_schema_mismatch")
    for name in ("figure_registry.json", "table_registry.json"):
        value = registry_document[name]
        if (
            type(value) is not dict
            or set(value) != _SOURCE_CONTRACT_KEYS
            or value["relative_path"] != name
        ):
            _fail("manifest", "manifest_schema_mismatch")
        try:
            contracts[name] = {
                "kind": "registry_json",
                "size_bytes": _require_size(value["size_bytes"]),
                "sha256": _require_sha256(value["sha256"]),
            }
        except Exception:
            _fail("manifest", "manifest_schema_mismatch")

    files = document["files"]
    if type(files) is not dict or not files:
        _fail("manifest", "manifest_schema_mismatch")
    for raw_path, raw_contract in files.items():
        try:
            relative_path = _require_relative_path(raw_path)
        except Exception:
            _fail("manifest", "manifest_schema_mismatch")
        if type(raw_contract) is not dict or "kind" not in raw_contract:
            _fail("manifest", "manifest_schema_mismatch")
        kind = raw_contract["kind"]
        try:
            if kind in {"source_json", "registry_json"}:
                expected_keys = {"kind", "size_bytes", "sha256"}
            elif kind == "product":
                expected_keys = {
                    "kind",
                    "artifact_id",
                    "role",
                    "media_type",
                    "size_bytes",
                    "sha256",
                }
            else:
                raise ValueError("unknown inventory kind")
            if set(raw_contract) != expected_keys:
                raise ValueError("inventory contract key set")
            size_bytes = _require_size(raw_contract["size_bytes"])
            digest = _require_sha256(raw_contract["sha256"])
            if kind == "product":
                _validate_product_topology(relative_path)
                if (
                    type(raw_contract["artifact_id"]) is not str
                    or type(raw_contract["role"]) is not str
                    or type(raw_contract["media_type"]) is not str
                ):
                    raise ValueError("product identity type")
            elif kind == "source_json" and relative_path not in {
                f"source/{name}" for name in REPORT_SOURCE_FILE_NAMES
            }:
                raise ValueError("source inventory path")
            elif kind == "registry_json" and relative_path not in {
                "figure_registry.json",
                "table_registry.json",
            }:
                raise ValueError("registry inventory path")
            contracts[relative_path] = {
                key: deepcopy(raw_contract[key]) for key in raw_contract
            }
            contracts[relative_path]["size_bytes"] = size_bytes
            contracts[relative_path]["sha256"] = digest
        except Exception:
            _fail("manifest", "manifest_schema_mismatch")

    for name in REPORT_SOURCE_FILE_NAMES:
        relative_path = f"source/{name}"
        if contracts.get(relative_path) != {
            "kind": "source_json",
            "size_bytes": source_document[name]["size_bytes"],
            "sha256": source_document[name]["sha256"],
        }:
            _fail("manifest", "manifest_schema_mismatch")
    for name in ("figure_registry.json", "table_registry.json"):
        declared = registry_document[name]
        contract = contracts.get(name)
        if (
            contract is None
            or contract.get("kind") != "registry_json"
            or contract.get("size_bytes") != declared["size_bytes"]
            or contract.get("sha256") != declared["sha256"]
        ):
            _fail("manifest", "manifest_schema_mismatch")
    return contracts


def _enumerate_report_payload(root: Path) -> set[str]:
    """以 lstat 枚舉固定 topology 下的 payload files，拒絕額外目錄或特殊節點。

    枚舉只服務於 exact inventory 比較，不會把未登錄檔案自動納入 manifest。figures root
    必須恰有 main/supplement 兩個固定子目錄；其他資料夾只能直接包含普通檔案。
    """

    try:
        _lstat_regular_directory(root)
        with os.scandir(root) as entries:
            root_names = {entry.name for entry in entries}
        if root_names != _ROOT_CONTROL_FILES | _ROOT_DIRECTORIES:
            _fail("topology", "fixed_node_set_mismatch")
        for name in _ROOT_CONTROL_FILES:
            node = os.lstat(root / name)
            if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
                _fail("topology", "symbolic_link_or_non_regular_node")
        source_root = root / "source"
        _lstat_regular_directory(source_root)
        with os.scandir(source_root) as entries:
            source_names = {entry.name for entry in entries}
        if source_names != set(REPORT_SOURCE_FILE_NAMES):
            _fail("topology", "fixed_node_set_mismatch")
        payload_files = {
            f"source/{name}" for name in REPORT_SOURCE_FILE_NAMES
        } | {"figure_registry.json", "table_registry.json"}
        for name in REPORT_SOURCE_FILE_NAMES:
            node = os.lstat(source_root / name)
            if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
                _fail("topology", "symbolic_link_or_non_regular_node")

        def collect_direct_files(directory: Path, prefix: str) -> None:
            """收集指定層級普通檔，拒絕 category 內額外子目錄。"""

            _lstat_regular_directory(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    node = os.lstat(directory / entry.name)
                    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
                        _fail("topology", "symbolic_link_or_non_regular_node")
                    payload_files.add(f"{prefix}{entry.name}")

        figures_root = root / "figures"
        _lstat_regular_directory(figures_root)
        with os.scandir(figures_root) as entries:
            figure_entries = tuple(entries)
        figure_names = {entry.name for entry in figure_entries}
        # figures root 的兩個出版層是 topology 節點，不是「有產品才建立」的選配目錄。
        # 即使某個 supplement 目前為空，仍要保留 ordinary directory，讓 release 的
        # 物理版面與 downstream 掃描邊界固定；因此先驗證兩者都存在，再檢查是否有
        # direct file 或額外名稱。
        if not _FIGURE_DIRECTORIES.issubset(figure_names):
            _fail("topology", "fixed_node_set_mismatch")
        if figure_names != _FIGURE_DIRECTORIES:
            _fail("topology", "fixed_node_set_mismatch")
        for entry in figure_entries:
            node = os.lstat(figures_root / entry.name)
            if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
                _fail("topology", "symbolic_link_or_non_regular_node")
            collect_direct_files(figures_root / entry.name, f"figures/{entry.name}/")
        for directory_name in ("tables", "caption_sidecars", "data_sidecars"):
            collect_direct_files(root / directory_name, f"{directory_name}/")
        return payload_files
    except _ReportReleaseValidationError:
        raise
    except FileNotFoundError:
        _fail("topology", "fixed_node_set_mismatch")
    except Exception:
        _fail("topology", "root_not_directory")


def _validate_inventory_checksums(
    root: Path,
    contracts: Mapping[str, Mapping[str, object]],
    actual_files: set[str],
) -> None:
    """逐一核對 exact inventory 的 ordinary file size/SHA-256。"""

    if set(contracts) != actual_files:
        _fail("manifest", "manifest_inventory_mismatch")
    for relative_path in sorted(actual_files):
        contract = contracts[relative_path]
        try:
            size_bytes, digest = _regular_file_size_and_sha256(root / relative_path)
        except Exception:
            _fail("checksum", "hash_mismatch")
        if size_bytes != contract["size_bytes"] or digest != contract["sha256"]:
            _fail("checksum", "hash_mismatch")


def _validate_registry_manifest_closure(
    *,
    registry: ReportRegistry,
    manifest_document: Mapping[str, object],
    contracts: Mapping[str, Mapping[str, object]],
) -> None:
    """確認 manifest identity 與每一列 F01-F12/T01-T06 的 products 完全閉合。"""

    for field_name in ("schema_version", "run_id", "run_kind", "experiment_case_id", "evidence_class"):
        if manifest_document[field_name] != getattr(registry, field_name):
            _fail("registry", "registry_closure_mismatch")

    expected_product_paths: dict[str, tuple[str, str, str, str, int, str]] = {}
    for artifact in (*registry.figures, *registry.tables):
        for product in artifact.products:
            try:
                _validate_product_topology(product.relative_path)
            except Exception:
                _fail("registry", "registry_closure_mismatch")
            if product.relative_path in expected_product_paths:
                _fail("registry", "registry_closure_mismatch")
            expected_product_paths[product.relative_path] = (
                artifact.artifact_id,
                product.role,
                product.media_type,
                "product",
                product.size_bytes,
                product.sha256,
            )
    actual_product_paths = {
        path
        for path, contract in contracts.items()
        if contract.get("kind") == "product"
    }
    if actual_product_paths != set(expected_product_paths):
        _fail("registry", "registry_closure_mismatch")
    for path, expected in expected_product_paths.items():
        contract = contracts[path]
        if (
            contract.get("artifact_id"),
            contract.get("role"),
            contract.get("media_type"),
            contract.get("kind"),
            contract.get("size_bytes"),
            contract.get("sha256"),
        ) != expected:
            _fail("registry", "registry_closure_mismatch")


def _validate_product_sidecar_schema(root: Path, registry: ReportRegistry) -> None:
    """對 JSON caption/metadata sidecar 執行 strict object 檢查，不解析大型 binary product。

    caption 與 metadata 是資料契約的可讀欄位；PNG、SVG、PDF、Parquet、NPY 的內容 schema
    由 renderer／table codec 負責，release 層只核對其 registry role、media type、大小與
    checksum，避免在發布層偷偷轉型或重新編碼科學產品。
    """

    for artifact in (*registry.figures, *registry.tables):
        for product in artifact.products:
            if product.role not in {"caption_sidecar_json", "metadata_sidecar_json"}:
                continue
            try:
                _strict_json_object(_read_regular_file_bytes(root / product.relative_path))
            except Exception:
                _fail("products", "product_schema_mismatch")


@dataclass(frozen=True, slots=True)
class ReportRelease:
    """完整 reader 結果的 immutable view。

    ``registry`` 是重新由 figure/table registry JSON 建立的 ReportRegistry；``source_bytes``
    保存七份 source snapshot 的 exact bytes，key 是不帶 ``source/`` prefix 的固定 logical
    filename。``manifest`` 是已通過 canonical/schema 檢查的 JSON snapshot，內容只含
    release 內部相對 path，不保存 caller 的絕對位置。產品本體不 materialize 到記憶體，
    其大小／hash／role 由 manifest 與 registry 提供；需要產品時應以 manifest 列出的 path
    由另一個 no-follow consumer 讀取。
    """

    registry: ReportRegistry
    source_bytes: Mapping[str, bytes]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        """建立 defensive source/manifest snapshot，拒絕缺少七份 source 或 mutable alias。"""

        if type(self.registry) is not ReportRegistry:
            raise TypeError("registry 必須是 exact ReportRegistry")
        if not isinstance(self.source_bytes, Mapping) or set(self.source_bytes) != set(
            REPORT_SOURCE_FILE_NAMES
        ):
            raise ValueError("source_bytes key set 不符")
        source_copy: dict[str, bytes] = {}
        for name in REPORT_SOURCE_FILE_NAMES:
            if type(self.source_bytes[name]) is not bytes:
                raise TypeError("source_bytes value 必須是 bytes")
            source_copy[name] = bytes(self.source_bytes[name])
        if not isinstance(self.manifest, Mapping):
            raise TypeError("manifest 必須是 mapping")
        object.__setattr__(self, "source_bytes", MappingProxyType(source_copy))
        object.__setattr__(self, "manifest", MappingProxyType(deepcopy(dict(self.manifest))))

    @property
    def product_paths(self) -> tuple[str, ...]:
        """回傳依 registry 固定順序排列的全部產品相對 path。"""

        return tuple(
            product.relative_path
            for artifact in (*self.registry.figures, *self.registry.tables)
            for product in artifact.products
        )


def _inspect_report_release(path: str | Path, *, require_final_name: bool) -> ReportRelease:
    """依 topology → manifest → checksum → registry/source/products 順序完整 inspection。"""

    try:
        root = Path(path)
        actual_files = _enumerate_report_payload(root)
        manifest_bytes = _read_regular_file_bytes(root / "report_manifest.json")
        manifest_document = _strict_json_object(manifest_bytes)
        if manifest_bytes != _canonical_json_bytes(manifest_document):
            _fail("manifest", "manifest_not_canonical")
        contracts = _validate_manifest_document(manifest_document)
        expected_final_name = (
            f"{manifest_document['run_id']}{REPORT_RELEASE_FINAL_DIRECTORY_SUFFIX}"
        )
        if require_final_name and root.name != expected_final_name:
            _fail("name", "final_name_mismatch")
        _validate_inventory_checksums(root, contracts, actual_files)

        figure_bytes = _read_regular_file_bytes(root / "figure_registry.json")
        table_bytes = _read_regular_file_bytes(root / "table_registry.json")
        figure_document = _strict_json_object(figure_bytes)
        table_document = _strict_json_object(table_bytes)
        if figure_bytes != _canonical_json_bytes(figure_document) or table_bytes != _canonical_json_bytes(
            table_document
        ):
            _fail("registry", "registry_schema_mismatch")
        try:
            registry = _registry_from_documents(figure_document, table_document)
        except _ReportReleaseValidationError:
            raise
        except Exception:
            _fail("registry", "registry_schema_mismatch")
        _validate_registry_manifest_closure(
            registry=registry,
            manifest_document=manifest_document,
            contracts=contracts,
        )
        source_bytes = {
            name: _read_regular_file_bytes(root / "source" / name)
            for name in REPORT_SOURCE_FILE_NAMES
        }
        try:
            source_paths = {
                name: root / "source" / name for name in REPORT_SOURCE_FILE_NAMES
            }
            _validate_source_semantics(
                source_bytes=source_bytes,
                source_paths=source_paths,
                registry=registry,
            )
        except _ReportReleaseValidationError:
            raise
        except Exception:
            _fail("source", "source_binding_mismatch")
        _validate_product_sidecar_schema(root, registry)
        return ReportRelease(
            registry=registry,
            source_bytes=source_bytes,
            manifest=manifest_document,
        )
    except _ReportReleaseValidationError:
        raise
    except FileNotFoundError:
        _fail("topology", "fixed_node_set_mismatch")
    except Exception:
        _fail("decoder", "unexpected_failure")


def _snapshot_registry(registry: ReportRegistry) -> ReportRegistry:
    """以 to_dict→constructor 建立 registry defensive snapshot，隔離低階 alias/tamper。"""

    if type(registry) is not ReportRegistry:
        raise TypeError("registry 必須是 exact ReportRegistry")
    document = registry.to_dict()
    return ReportRegistry(
        schema_version=document["schema_version"],
        run_id=document["run_id"],
        run_kind=document["run_kind"],
        experiment_case_id=document["experiment_case_id"],
        evidence_class=document["evidence_class"],
        aggregate_manifest_sha256=document["aggregate_manifest_sha256"],
        source_run_plan_sha256=document["source_run_plan_sha256"],
        source_run_progress_sha256=document["source_run_progress_sha256"],
        config_hash=document["config_hash"],
        checkpoint_input_binding_hash=document["checkpoint_input_binding_hash"],
        allow_missing_comparison=document["allow_missing_comparison"],
        allow_missing_validation_evidence=document["allow_missing_validation_evidence"],
        figures=tuple(
            _artifact_from_dict(item) for item in document["figures"]  # type: ignore[arg-type]
        ),
        tables=tuple(
            _artifact_from_dict(item) for item in document["tables"]  # type: ignore[arg-type]
        ),
    )


def _product_staging_paths(
    *,
    registry: ReportRegistry,
    staging_root: str | Path | None,
    product_paths: Mapping[str, object] | None,
) -> dict[str, Path]:
    """建立 registry product path → caller staging file path 的 explicit mapping。"""

    expected = {
        product.relative_path
        for artifact in (*registry.figures, *registry.tables)
        for product in artifact.products
    }
    result: dict[str, Path] = {}
    if product_paths is not None:
        if not isinstance(product_paths, Mapping):
            raise ValueError("product_paths 必須是 mapping")
        for raw_relative_path, raw_staging_path in product_paths.items():
            relative_path = _require_relative_path(raw_relative_path)
            if relative_path in result:
                raise ValueError("product path 重複")
            result[relative_path] = _path_value(raw_staging_path)
    if staging_root is not None:
        root = _path_value(staging_root)
        _lstat_regular_directory(root)
        for relative_path in expected:
            result.setdefault(relative_path, root / relative_path)
    if set(result) != expected:
        raise ValueError("product_paths 必須精確涵蓋 registry products")
    if len({os.fspath(path) for path in result.values()}) != len(result):
        raise ValueError("不同 registry path 不得共用 staging file")
    return result


def _validate_staging_products(
    *,
    registry: ReportRegistry,
    staging_paths: Mapping[str, Path],
) -> dict[str, bytes]:
    """只讀取 registry 明示的 ordinary staging files，並比對 role/size/SHA contract。"""

    product_bytes: dict[str, bytes] = {}
    for artifact in (*registry.figures, *registry.tables):
        for product in artifact.products:
            try:
                _validate_product_topology(product.relative_path)
                raw_bytes = _read_regular_file_bytes(staging_paths[product.relative_path])
            except Exception as error:
                raise ValueError("staging product 不是 ordinary file") from error
            if len(raw_bytes) != product.size_bytes or _sha256_bytes(raw_bytes) != product.sha256:
                raise ValueError("staging product checksum 不符")
            product_bytes[product.relative_path] = raw_bytes
    return product_bytes


@dataclass(frozen=True, slots=True)
class _PreparedReportRelease:
    """writer 在建立 partial 前封存的 registry、source 與 product bytes。"""

    registry: ReportRegistry
    source_bytes: Mapping[str, bytes]
    product_bytes: Mapping[str, bytes]
    source_paths: Mapping[str, Path]


def _prepare_report_inputs(
    *,
    registry: ReportRegistry,
    source_run_root: str | Path | None,
    aggregate_release_root: str | Path | None,
    report_spec_path: str | Path | None,
    config_path: str | Path | None,
    source_paths: Mapping[str, object] | None,
    staging_root: str | Path | None,
    product_paths: Mapping[str, object] | None,
) -> _PreparedReportRelease:
    """完成所有來源／staging read-only binding，成功前不建立任何 report partial。"""

    snapshot = _snapshot_registry(registry)
    resolved_source_paths = _validate_source_mapping_paths(
        registry=snapshot,
        source_run_root=source_run_root,
        aggregate_release_root=aggregate_release_root,
        report_spec_path=report_spec_path,
        config_path=config_path,
        source_paths=source_paths,
    )
    source_bytes = {
        name: _read_regular_file_bytes(resolved_source_paths[name])
        for name in REPORT_SOURCE_FILE_NAMES
    }

    # 如果 caller 同時明示 aggregate root/source run root，兩棵 tree 都以固定 filename
    # 讀取並逐 bytes 比較；不比較整棵目錄，避免把無關 cache 或他人 partial 帶入範圍。
    if aggregate_release_root is not None:
        aggregate_root = _path_value(aggregate_release_root)
        for aggregate_name, report_name in _AGGREGATE_TO_REPORT_SOURCE_NAME.items():
            aggregate_bytes = _read_regular_file_bytes(aggregate_root / aggregate_name)
            if aggregate_bytes != source_bytes[report_name]:
                raise ValueError("aggregate source 與 selected source 不一致")
    if source_run_root is not None:
        run_root = _path_value(source_run_root)
        for name in _RUN_SOURCE_FILE_NAMES:
            run_bytes = _read_regular_file_bytes(run_root / name)
            if run_bytes != source_bytes[name]:
                raise ValueError("source run 與 aggregate source 不一致")

    _validate_source_semantics(
        source_bytes=source_bytes,
        source_paths=resolved_source_paths,
        registry=snapshot,
    )
    staging_paths = _product_staging_paths(
        registry=snapshot,
        staging_root=staging_root,
        product_paths=product_paths,
    )
    product_bytes = _validate_staging_products(
        registry=snapshot,
        staging_paths=staging_paths,
    )
    return _PreparedReportRelease(
        registry=snapshot,
        source_bytes=MappingProxyType(dict(source_bytes)),
        product_bytes=MappingProxyType(dict(product_bytes)),
        source_paths=MappingProxyType(dict(resolved_source_paths)),
    )


def _make_partial_topology(partial_root: Path) -> None:
    """在自有空白 partial 內建立固定 root/source/figure/table/sidecar directories。"""

    _lstat_regular_directory(partial_root)
    for directory_name in _ROOT_DIRECTORIES:
        directory = partial_root / directory_name
        directory.mkdir(mode=0o700, exist_ok=False)
        _lstat_regular_directory(directory)
    figures_root = partial_root / "figures"
    for directory_name in _FIGURE_DIRECTORIES:
        directory = figures_root / directory_name
        directory.mkdir(mode=0o700, exist_ok=False)
        _lstat_regular_directory(directory)


def _write_prepared_partial(partial_root: Path, prepared: _PreparedReportRelease) -> None:
    """把 prepared exact bytes 寫入 partial，建立 manifest 後同步所有節點與目錄。"""

    _make_partial_topology(partial_root)
    registry = prepared.registry
    figure_registry_bytes = _canonical_json_bytes(
        _registry_document(registry, registry_kind="figure")
    )
    table_registry_bytes = _canonical_json_bytes(
        _registry_document(registry, registry_kind="table")
    )

    for name in REPORT_SOURCE_FILE_NAMES:
        _write_exclusive_durable_bytes(
            partial_root / "source" / name,
            prepared.source_bytes[name],
        )
    _write_exclusive_durable_bytes(partial_root / "figure_registry.json", figure_registry_bytes)
    _write_exclusive_durable_bytes(partial_root / "table_registry.json", table_registry_bytes)

    for artifact in (*registry.figures, *registry.tables):
        for product in artifact.products:
            destination = partial_root / product.relative_path
            _validate_product_topology(product.relative_path)
            _write_exclusive_durable_bytes(destination, prepared.product_bytes[product.relative_path])

    manifest_document = _build_manifest_document(
        registry=registry,
        source_bytes=prepared.source_bytes,
        figure_registry_bytes=figure_registry_bytes,
        table_registry_bytes=table_registry_bytes,
    )
    manifest_bytes = _canonical_json_bytes(manifest_document)
    _write_exclusive_durable_bytes(partial_root / "report_manifest.json", manifest_bytes)

    # fsync 每個固定 directory entry；檔案本身已在 exclusive writer 中同步。順序由深至
    # 淺，讓新增 product 與 source/figure 子目錄的 directory entry 都在 partial 驗證前穩定。
    for directory in (
        partial_root / "source",
        partial_root / "figures" / "main",
        partial_root / "figures" / "supplement",
        partial_root / "figures",
        partial_root / "tables",
        partial_root / "caption_sidecars",
        partial_root / "data_sidecars",
        partial_root,
    ):
        _fsync_directory(directory)


class ReportReleaseWriter:
    """建立一份 source-run sibling report-v1 release 的 atomic writer。

    建構子只保存 caller 的設定，不建立目錄或讀檔；真正的來源、registry、staging product
    與 destination 檢查都在 :meth:`write` 內執行。``staging_root`` 是簡單情境下的產品
    根目錄；``product_paths`` 則可逐一明示 ``registry relative_path -> staging file``，
    且兩者可以同時使用以覆蓋個別檔案。source snapshot 可由 aggregate/source roots 與
    report spec path 組合，或用 ``source_paths`` 明示七個 logical filename。

    writer 的 final basename 固定為 ``<registry.run_id>.report-v1``，預設與 source run
    同父；明示 destination 也必須是同一個 parent 與 basename。既有 final、symbolic
    link、broken link、他人 partial 都不會被覆寫或清理；失敗清理只針對本次 UUID partial
    且會再次核對 device/inode ownership。
    """

    def __init__(
        self,
        registry: ReportRegistry,
        *,
        source_run_root: str | Path | None = None,
        aggregate_release_root: str | Path | None = None,
        report_spec_path: str | Path | None = None,
        config_path: str | Path | None = None,
        staging_root: str | Path | None = None,
        product_paths: Mapping[str, object] | None = None,
        source_paths: Mapping[str, object] | None = None,
        destination: str | Path | None = None,
    ) -> None:
        """保存 writer inputs；source/product files 會在 write 時以 no-follow 讀取。"""

        self.registry = registry
        self.source_run_root = source_run_root
        self.aggregate_release_root = aggregate_release_root
        self.report_spec_path = report_spec_path
        self.config_path = config_path
        self.staging_root = staging_root
        self.product_paths = product_paths
        self.source_paths = source_paths
        self.destination = destination

    def _destination_context(self, registry: ReportRegistry) -> tuple[Path, Path, os.stat_result]:
        """解析固定 final parent/path，並確認 parent 是 ordinary non-symlink directory。"""

        base_value = self.source_run_root or self.aggregate_release_root
        if base_value is None:
            if self.destination is None:
                raise ValueError("缺少 destination 或 source root")
            base_path = Path(self.destination).parent
            _lstat_regular_directory(base_path)
        else:
            base_path = Path(base_value)
            _lstat_regular_directory(base_path)
        parent = Path(os.path.abspath(os.path.normpath(os.fspath(base_path).rstrip("/"))))
        if base_value is not None:
            parent = parent.parent
        parent_status = _lstat_regular_directory(parent)
        final_name = f"{registry.run_id}{REPORT_RELEASE_FINAL_DIRECTORY_SUFFIX}"
        expected_final = parent / final_name
        if self.destination is None:
            final_path = expected_final
        else:
            destination = Path(self.destination)
            if ".." in destination.parts:
                raise ValueError("destination 不可含 ..")
            final_path = Path(os.path.abspath(os.path.normpath(os.fspath(destination))))
            if final_path != expected_final:
                raise ValueError("destination 不符合固定 sibling final")
        return parent, final_path, parent_status

    @staticmethod
    def _require_absent(path: Path) -> None:
        """以 lstat 拒絕既有 final、普通檔、目錄與 broken symbolic link。"""

        try:
            os.lstat(path)
        except FileNotFoundError:
            return
        raise _ReportReleaseFinalExistsError("report release final 已存在")

    @staticmethod
    def _cleanup_owned_partial(
        partial_path: Path | None,
        partial_identity: tuple[int, int] | None,
        *,
        published: bool,
    ) -> None:
        """只清理本次 writer 建立且 identity 未改變的 partial。"""

        if partial_path is None or partial_identity is None or published:
            return
        try:
            node = os.lstat(partial_path)
            if (
                stat.S_ISLNK(node.st_mode)
                or not stat.S_ISDIR(node.st_mode)
                or (node.st_dev, node.st_ino) != partial_identity
            ):
                return
            shutil.rmtree(partial_path)
        except Exception:
            # ownership／權限不明時保留現場，避免把他人節點當成本次 partial 刪除。
            return

    def write(self) -> Path:
        """完成來源綁定（source binding）、原子暫存目錄建置與拒覆寫目錄重新命名。

        成功回傳 final ``Path``。任何 rename 前錯誤都轉成不含 path 的固定
        ``ValueError("report release 寫入失敗")``；既有 final 保留 ``FileExistsError`` 讓
        caller 能區分不可覆寫衝突。rename 成功但 parent fsync 失敗時，final 已經存在，
        因此回傳固定 ``RuntimeError("report release 已發布但 parent durability 未確認")``。
        """

        partial_path: Path | None = None
        partial_identity: tuple[int, int] | None = None
        published = False
        try:
            prepared = _prepare_report_inputs(
                registry=self.registry,
                source_run_root=self.source_run_root,
                aggregate_release_root=self.aggregate_release_root,
                report_spec_path=self.report_spec_path,
                config_path=self.config_path,
                source_paths=self.source_paths,
                staging_root=self.staging_root,
                product_paths=self.product_paths,
            )
            parent, final_path, parent_status = self._destination_context(prepared.registry)
            self._require_absent(final_path)
            if (parent_status.st_dev, parent_status.st_ino) != (
                os.lstat(parent).st_dev,
                os.lstat(parent).st_ino,
            ):
                raise ValueError("final parent identity changed")
            final_name = final_path.name
            partial_path = parent / f".{final_name}.partial-{uuid4().hex}"
            partial_path.mkdir(mode=0o700, exist_ok=False)
            partial_status = os.lstat(partial_path)
            if stat.S_ISLNK(partial_status.st_mode) or not stat.S_ISDIR(partial_status.st_mode):
                raise ValueError("partial node 不符")
            partial_identity = (partial_status.st_dev, partial_status.st_ino)

            _write_prepared_partial(partial_path, prepared)
            inspection = _inspect_report_release(partial_path, require_final_name=False)
            if inspection.registry != prepared.registry or dict(inspection.source_bytes) != dict(
                prepared.source_bytes
            ):
                raise ValueError("partial self validation snapshot mismatch")

            # rename 前最後一次檢查只看 parent/final/partial identity；完整內容已在上一步
            # inspection 通過，且 partial ownership 未轉移前不會碰其他 partial。exclusive
            # backend 會再次以同一組 parent identity 開啟 dirfd，封閉最後的 path race。
            current_parent_status = _lstat_regular_directory(parent)
            parent_identity = (parent_status.st_dev, parent_status.st_ino)
            if (current_parent_status.st_dev, current_parent_status.st_ino) != parent_identity:
                raise ValueError("final parent identity changed")
            self._require_absent(final_path)
            partial_status = os.lstat(partial_path)
            if (partial_status.st_dev, partial_status.st_ino) != partial_identity:
                raise ValueError("partial ownership changed")
            _atomic_exclusive_directory_rename(
                partial_path,
                final_path,
                expected_parent_identity=parent_identity,
            )
            published = True
            _fsync_directory(parent, expected_identity=parent_identity)
            return final_path
        except _ReportReleaseFinalExistsError:
            self._cleanup_owned_partial(partial_path, partial_identity, published=published)
            raise FileExistsError("report release final 已存在") from None
        except FileExistsError:
            # partial 內的 exclusive-create race 不是 caller 明示的既有 final collision；
            # 不把作業系統例外原文直接拋出，避免其中附帶 partial 絕對路徑。
            self._cleanup_owned_partial(partial_path, partial_identity, published=published)
            raise ValueError("report release 寫入失敗") from None
        except Exception:
            if published:
                raise RuntimeError(
                    "report release 已發布但 parent durability 未確認"
                ) from None
            self._cleanup_owned_partial(partial_path, partial_identity, published=published)
            raise ValueError("report release 寫入失敗") from None

    def build(self) -> Path:
        """``write`` 的可讀別名，供 pipeline 以 build 語意呼叫。"""

        return self.write()


def write_report_release(
    *,
    registry: ReportRegistry,
    source_run_root: str | Path | None = None,
    aggregate_release_root: str | Path | None = None,
    report_spec_path: str | Path | None = None,
    config_path: str | Path | None = None,
    staging_root: str | Path | None = None,
    product_paths: Mapping[str, object] | None = None,
    source_paths: Mapping[str, object] | None = None,
    destination: str | Path | None = None,
) -> Path:
    """以 :class:`ReportReleaseWriter` 建立 report-v1 sibling final。

    這個函式是 CLI/pipeline 可使用的薄 facade；所有輸入責任與安全限制均由 writer
    執行。``config_path`` 若明示，代表 normalized_config.json 的 ordinary source file，
    並會與 aggregate/source-run snapshot 逐 bytes 綁定；它不是用來覆蓋 registry 的
    canonical hash。成功只代表 engineering release I/O contract 通過。
    """

    return ReportReleaseWriter(
        registry,
        source_run_root=source_run_root,
        aggregate_release_root=aggregate_release_root,
        report_spec_path=report_spec_path,
        config_path=config_path,
        staging_root=staging_root,
        product_paths=product_paths,
        source_paths=source_paths,
        destination=destination,
    ).write()


def read_report_release(path: str | Path) -> ReportRelease:
    """no-follow 完整驗證並讀取一份 final report-v1 release。

    驗證順序固定為 root topology、canonical manifest/exact inventory、普通檔案大小／
    SHA-256、registry closure、source binding 與 sidecar schema。任何失敗都轉成不含
    path、stage 或底層例外文字的固定 ``ValueError("report release 驗證失敗")``；需要
    JSON-safe stage/reason 時請使用 :func:`validate_report_release`。
    """

    try:
        return _inspect_report_release(path, require_final_name=True)
    except Exception:
        raise ValueError("report release 驗證失敗") from None


def read_report_registry(path: str | Path) -> ReportRegistry:
    """完整驗證 final release 後回傳其 immutable ReportRegistry。"""

    return read_report_release(path).registry


def _validation_summary(release: ReportRelease) -> dict[str, object]:
    """建立固定 JSON-safe validator summary，不攜帶 Path 或產品 bytes。"""

    registry = release.registry
    artifacts = (*registry.figures, *registry.tables)
    product_count = sum(len(artifact.products) for artifact in artifacts)
    available_count = sum(artifact.status == "available" for artifact in artifacts)
    return {
        "schema_version": registry.schema_version,
        "run_id": registry.run_id,
        "run_kind": registry.run_kind,
        "experiment_case_id": registry.experiment_case_id,
        "evidence_class": registry.evidence_class,
        "artifact_count": len(artifacts),
        "available_artifact_count": available_count,
        "source_file_count": len(REPORT_SOURCE_FILE_NAMES),
        "product_file_count": product_count,
        "inventory_file_count": len(release.manifest["files"]),
    }


def validate_report_release(path: str | Path) -> dict[str, object]:
    """驗證 report-v1 final 並回傳固定 JSON-safe ``valid/errors/summary``。

    成功時 ``errors`` 是空 list，summary 只含原生字串與整數；失敗時只回傳第一個固定
    ``{"stage": ..., "reason": ...}``，不回傳 exception、absolute path、檔名、SERVER
    路徑或第三方套件訊息。stage 順序可區分 topology、manifest、checksum、registry、
    source 與 products 等責任邊界，但不代表通過工程 validator 就等於科學結果驗證。
    """

    try:
        release = _inspect_report_release(path, require_final_name=True)
    except _ReportReleaseValidationError as error:
        return {
            "valid": False,
            "errors": [{"stage": error.stage, "reason": error.reason}],
            "summary": {},
        }
    except Exception:
        return {
            "valid": False,
            "errors": [{"stage": "decoder", "reason": "unexpected_failure"}],
            "summary": {},
        }
    return {"valid": True, "errors": [], "summary": _validation_summary(release)}
