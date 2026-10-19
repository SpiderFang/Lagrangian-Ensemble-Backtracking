"""WP-63：壞時刻失敗矩陣的續跑快取（預設關閉；開啟時 round-trip 位元相同）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lagrangian_backtracking import input_derivation as idv


def test_cache_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LBT_BAD_HOUR_CACHE_DIR", raising=False)
    assert idv._bad_hour_cache_file(site_id="s", anchor_lonlat=(1.0, 2.0), radius_m=1.0, hours_ns=np.arange(3), face_count=5) is None


def test_cache_round_trip_and_key_sensitivity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LBT_BAD_HOUR_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("LBT_BAD_HOUR_CACHE_TAG", "t1")
    hours = np.arange(5, dtype=np.int64) * 3_600_000_000_000
    f = idv._bad_hour_cache_file(site_id="s", anchor_lonlat=(1.0, 2.0), radius_m=10000.0, hours_ns=hours, face_count=7)
    assert f is not None and idv._bad_hour_cache_load(f) is None
    rng = np.random.default_rng(1)
    failed = rng.random((11, 5)) < 0.3
    ids = np.arange(11, dtype=np.int64) * 3
    area = rng.random(11) + 1.0
    idv._bad_hour_cache_store(f, failed, ids, area, {"a": 1, "b": [1.5, 2]})
    got = idv._bad_hour_cache_load(f)
    assert got is not None
    assert np.array_equal(got[0], failed) and got[0].dtype == bool
    assert np.array_equal(got[1], ids) and np.array_equal(got[2], area)
    assert got[3] == {"a": 1, "b": [1.5, 2]}
    # 鍵對候選整點、錨點與標籤敏感
    g = idv._bad_hour_cache_file(site_id="s", anchor_lonlat=(1.0, 2.0), radius_m=10000.0, hours_ns=hours[:4], face_count=7)
    h = idv._bad_hour_cache_file(site_id="s", anchor_lonlat=(1.0, 2.001), radius_m=10000.0, hours_ns=hours, face_count=7)
    monkeypatch.setenv("LBT_BAD_HOUR_CACHE_TAG", "t2")
    k = idv._bad_hour_cache_file(site_id="s", anchor_lonlat=(1.0, 2.0), radius_m=10000.0, hours_ns=hours, face_count=7)
    assert len({f, g, h, k}) == 4
    # 損毀檔案視為未命中
    f.write_bytes(b"junk")
    assert idv._bad_hour_cache_load(f) is None
