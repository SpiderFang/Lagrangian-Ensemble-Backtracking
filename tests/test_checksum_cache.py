"""重建 patch 已驗證雜湊快取的行為測試。"""

from __future__ import annotations

import hashlib
import os
import time

from lagrangian_backtracking import forcing_window as fw


def test_checksum_cache_disabled_by_default(tmp_path, monkeypatch):
    """未設定環境變數時不產生任何快取檔，結果與直接計算相同。"""

    monkeypatch.delenv(fw.CHECKSUM_CACHE_DIR_ENV, raising=False)
    target = tmp_path / "a.npy"
    target.write_bytes(b"abc" * 1000)
    assert fw._sha256_file(target) == hashlib.sha256(b"abc" * 1000).hexdigest()
    assert fw._checksum_cache_entry(target) is None


def test_checksum_cache_reuses_value_and_invalidates_on_change(tmp_path, monkeypatch):
    """同一未變動檔案第二次不再讀檔；內容改寫後（大小或修改時間改變）重新計算。"""

    cache = tmp_path / "cache"
    monkeypatch.setenv(fw.CHECKSUM_CACHE_DIR_ENV, str(cache))
    target = tmp_path / "patch.npy"
    target.write_bytes(b"x" * 4096)
    first = fw._sha256_file(target)
    assert first == hashlib.sha256(b"x" * 4096).hexdigest()
    assert fw._checksum_cache_entry(target).is_file()

    # 讀檔被禁止時仍能回傳，證明第二次確實沒有重讀內容。
    original_open = type(target).open

    def forbidden_open(self, *args, **kwargs):
        if self == target:
            raise AssertionError("命中快取時不應重新讀檔")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(type(target), "open", forbidden_open)
    assert fw._sha256_file(target) == first
    monkeypatch.setattr(type(target), "open", original_open)

    # 改寫內容：大小相同但修改時間改變，必須重新計算出新雜湊。
    time.sleep(0.01)
    target.write_bytes(b"y" * 4096)
    os.utime(target, ns=(time.time_ns(), time.time_ns()))
    assert fw._sha256_file(target) == hashlib.sha256(b"y" * 4096).hexdigest()
