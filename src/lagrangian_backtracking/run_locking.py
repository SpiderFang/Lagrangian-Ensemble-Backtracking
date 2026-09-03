"""正式 run workspace 的 Unix 檔案鎖。

run controller 使用預先建立的零長度普通檔案作為鎖標記，實際互斥由 Unix
``fcntl.flock`` 提供。鎖檔只保存程序核心狀態，不寫入任何資料，因此不納入 run plan
的 immutable input checksum。這個模組只處理安全開檔、鎖定、釋放與 contention；run
生命週期與 lock topology 由 ``run_control``／``run_validation`` 驗證。
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal


class RunLockBusyError(RuntimeError):
    """表示要求的 run 鎖目前由另一個程序持有。"""


def _open_lock_file(path: Path) -> int:
    """以 no-follow 開啟既有零長度普通鎖檔，避免 symlink/race 逃逸。

    錯誤訊息刻意不包含檔案的絕對路徑，因為 run workspace 可能位於受保護的 SERVER
    目錄；呼叫端只需要知道契約錯誤，不應把現場路徑寫進 failure artifact 或 log。
    """

    flags = os.O_RDWR
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("run lock 必須是可開啟的既有普通檔案") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 0:
            raise ValueError("run lock 必須是零長度普通檔案")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def acquire_run_lock(
    path: str | os.PathLike[str],
    *,
    mode: Literal["shared", "exclusive"] = "exclusive",
    blocking: bool = True,
) -> Iterator[None]:
    """在既有 run lock file 上取得 shared/exclusive 的 ``fcntl`` 鎖。

    Args:
        path: 已預建的零長度普通鎖檔；不可是 symlink、目錄、非零長度檔案或不存在。
        mode: ``shared`` 允許不同 shard worker 同時持有，``exclusive`` 用於單一 shard
            或 reconcile。鎖的粒度與固定順序由上層 controller 決定。
        blocking: ``False`` 代表 contention 立即以 ``RunLockBusyError`` 回傳；``True``
            會等待核心釋放鎖。

    Raises:
        RunLockBusyError: non-blocking 取得鎖時發現其他程序持有。
        ValueError: 鎖檔不存在、是 symlink、不是普通零長度檔案或參數不合法。

    Notes:
        Unix ``flock`` 對同一檔案描述元的 process semantics 是本模組的前提；NFS 的
        locking 行為仍須在目標 SERVER 以 preflight 實測，不能只因本機測試通過就宣稱
        分散式檔案系統具備相同保證。離開 context（包括例外）一定釋放鎖並關閉描述元。
    """

    if mode not in {"shared", "exclusive"}:
        raise ValueError("run lock mode 必須是 shared 或 exclusive")
    lock_path = Path(path)
    if lock_path.is_symlink() or not lock_path.exists() or not lock_path.is_file():
        raise ValueError("run lock 必須是既有普通檔案")
    descriptor = _open_lock_file(lock_path)
    operation = fcntl.LOCK_SH if mode == "shared" else fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, operation)
        except OSError as exc:
            if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RunLockBusyError("run lock busy") from exc
            raise ValueError("run lock 無法取得") from exc
        acquired = True
        yield
    finally:
        if acquired:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            # 原始例外不可被釋放失敗覆蓋；描述元仍需關閉，讓核心回收 owner。
        os.close(descriptor)


__all__ = ["RunLockBusyError", "acquire_run_lock"]
