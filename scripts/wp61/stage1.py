# ruff: noqa: E501, E402, SIM115
"""WP-61 階段 1：重現挑選時用的序列（OCM 海面高程、OCM 流速、NWW3 Hs）與 25 小時潮汐強度，存成快取。只讀。"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

W = Path(os.environ.get("LBT_WORKSPACE", "/home/mustlab/Workspace"))
# WP-63：資料根、母體、程式碼與輸出位置可由環境變數覆寫；未設定時與 run08 版完全相同。
# run09 用：LBT_STAGE1_SURFACE_ROOT=<v7-surface-fix202511-12/ocm_surface>、LBT_STAGE1_OCM_ROOT=<v6-full/ocm_native>、
# LBT_STAGE1_NWW_ROOT=<NAS 24 個月 nww3_analysis>、LBT_STAGE1_NWW_FIRST_MONTH=202401、LBT_STAGE1_SEAM_FROM_MANIFEST=1。
sys.path.insert(0, os.environ.get("LBT_CODE_SRC", str(W / "lbt-repo-wp61/src")))
os.environ["OCM_SURFACE_ROOT"] = os.environ.get("LBT_STAGE1_SURFACE_ROOT", str(W / "lbt-local-data-v6-full/ocm_surface"))
os.environ["OCM_NATIVE_ROOT"] = os.environ.get("LBT_STAGE1_OCM_ROOT", str(W / "lbt-local-data-v6-full/ocm_native"))
os.environ["NWW_ANALYSIS_ROOT"] = os.environ.get("LBT_STAGE1_NWW_ROOT", str(W / "lbt-local-data-v6/nww3_analysis"))
from lagrangian_backtracking import input_derivation as idv
from lagrangian_backtracking.arrival_times import _rolling_rms
from lagrangian_backtracking.config import load_config

INP = Path(
    os.environ.get(
        "LBT_ARRIVAL_POP",
        str(W / "lbt-inputs/formal/v7-class-velocity/20261013_final-sites_h30-h60-h90_common_m10_v7"),
    )
)
OUT = Path(os.environ.get("LBT_ARRIVAL_DATA", str(W / "lbt-wp61/data")))
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config(INP / "common-config.yaml", formal_release=False)
months24 = idv._months_for_config(cfg)
print("months", months24[0], months24[-1], len(months24))
expected = idv._expected_hourly_axis(months24, step_hours=1.0)
arr = json.load(open(INP / "common-input/arrival.json"))
meta = {}
for r in arr["records"]:
    m = r["metadata"]
    meta.setdefault(r["study_site_id"], {})[m["observation_arrival_time_id"]] = m
seam_excluded: dict[str, np.ndarray] = {}
if os.environ.get("LBT_STAGE1_SEAM_FROM_MANIFEST") == "1":
    # run09：接縫跳動排除清單取自母體 manifest 的 provenance（建置時實際使用的那份），並套用到 selector 與 valid 遮罩。
    def _find(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                r = _find(v, key)
                if r is not None:
                    return r
        return None

    _seam = _find(arr.get("provenance", {}), "seam_jump_exclusion")
    if _seam is None:
        raise SystemExit("LBT_STAGE1_SEAM_FROM_MANIFEST=1，但 manifest provenance 沒有 seam_jump_exclusion")
    for _sid, _v in _seam["excluded_times_by_site"].items():
        seam_excluded[_sid] = np.asarray(_v["utc_ns"], dtype=np.int64)
roots = {
    k: idv._env_root(None, getattr(cfg.inputs, k + "_root_env"))
    for k in ("ocm_native", "ocm_surface", "nww_analysis")
}
nww_months = [m for m in months24 if m >= os.environ.get("LBT_STAGE1_NWW_FIRST_MONTH", "202407")]
products = {}
for dom in cfg.domains:
    region = dom.analysis_region_id
    fid = idv._resolve_flow_product_id(cfg, region, roots["ocm_surface"], formal=False)
    nat = idv._load_product(
        product="ocm_native",
        root=roots["ocm_native"],
        root_token=cfg.inputs.ocm_native_root_env,
        flow_domain_id=fid,
        months=months24,
        config=cfg,
    )
    sur = idv._load_product(
        product="ocm_surface",
        root=roots["ocm_surface"],
        root_token=cfg.inputs.ocm_surface_root_env,
        flow_domain_id=fid,
        months=months24,
        config=cfg,
    )
    nw = idv._load_product(
        product="nww3_analysis",
        root=roots["nww_analysis"],
        root_token=cfg.inputs.nww_analysis_root_env,
        flow_domain_id=fid,
        months=nww_months,
        config=cfg,
    )
    products[region] = (nat, sur, nw, idv._load_nww_runtime_cache(nw))
    print(
        region,
        fid,
        "axis equal native/surface:",
        np.array_equal(nat.canonical.time_utc_ns, sur.canonical.time_utc_ns),
        nat.canonical.time_utc_ns.size,
    )
sel = cfg.arrival_time_selection
hz = idv.resolve_configured_horizon(cfg)
gap_on = idv._gap_censoring_enabled(cfg)
print("gap censoring", gap_on, "years", sel.observation_years, sel.policy, sel.tidal_strength_window_hours)
t0 = time.time()
for site in cfg.study_sites:
    sid = site.study_site_id
    nat, sur, nw, cache = products[site.analysis_region_id]
    lon, lat = site.anchor_lonlat
    elev, speed = idv._surface_series_for_location(sur, lon=float(lon), lat=float(lat))
    # NWW 取樣點：manifest 記錄的 metric location（anchor 或 cell center）
    any_meta = next(iter(meta[sid].values()))
    mlon, mlat = float(any_meta["metric_location_lon"]), float(any_meta["metric_location_lat"])
    nww_series, nww_valid = idv._nww_series_for_location(nw, lon=mlon, lat=mlat, cache=cache)
    times = np.asarray(nat.canonical.time_utc_ns, dtype=np.int64)
    eta = np.array([elev.get(int(t), np.nan) for t in times])
    spd = np.array([speed.get(int(t), np.nan) for t in times])
    hs = np.array([nww_series.get(int(t), np.nan) for t in times])
    nvalid = np.array([bool(nww_valid.get(int(t), False)) for t in times])
    # 與 selector 完全相同的重現：呼叫正式函式
    recs = idv._select_arrivals_for_site(
        site_id=sid,
        product=nat,
        nww_series=nww_series,
        nww_valid_for_site=nww_valid,
        ocm_elevation=elev,
        ocm_speed=speed,
        max_backtrack_days=hz.selection_days,
        design_version=cfg.design_version,
        expected_axis=expected,
        strict=True,
        observation_years=tuple(int(y) for y in sel.observation_years),
        replicates=sel.replicates,
        selection_policy=sel.policy,
        tidal_strength_window_hours=sel.tidal_strength_window_hours,
        allow_gap_censored_anchor=gap_on,
        **({"excluded_times_ns": seam_excluded[sid]} if sid in seam_excluded else {}),
    )
    seconds = times.astype(np.float64) / 1e9
    nominal = float(np.median(np.diff(seconds)))
    window = max(int(round(float(sel.tidal_strength_window_hours) * 3600.0 / nominal)), 3)
    strength = _rolling_rms(eta, window)
    deriv = np.gradient(eta, seconds)
    valid = np.isfinite(eta) & np.isfinite(hs) & np.isfinite(spd) & nvalid & np.isin(times, expected)
    if sid in seam_excluded:
        valid &= ~np.isin(times, seam_excluded[sid])
    if not gap_on:
        valid &= idv._backward_window_mask(
            times,
            expected_axis=expected,
            available_axis=nat.canonical.time_utc_ns,
            max_backtrack_days=hz.selection_days,
        )
    np.savez_compressed(
        OUT / f"{sid}.npz",
        times=times,
        eta=eta,
        speed=spd,
        hs=hs,
        strength=strength,
        deriv=deriv,
        valid=valid,
        window=window,
        anchor=np.array([float(lon), float(lat)]),
        metric=np.array([mlon, mlat]),
    )
    # 與 manifest 對照
    man = {int(m["observation_time_utc_ns"]): m for m in meta[sid].values()}
    rec_by_t = {int(r.time_utc_ns): r for r in recs}
    same_times = set(rec_by_t) == set(man)
    ok_label = 0
    max_de = 0.0
    for t, m in man.items():
        r = rec_by_t.get(t)
        if r is None:
            continue
        ok_label += (
            r.tide_class == m["observation_tide_class"]
            and r.phase_or_event == m["observation_phase_or_event"]
        )
        i = int(np.searchsorted(times, t))
        # 事件觀測（高浪／強流）沒有 elevation_m；run09 起龜山島自選事件，重跑選擇器的紀錄會含事件列。
        if m.get("elevation_m") is not None:
            max_de = max(max_de, abs(eta[i] - m["elevation_m"]))
    print(
        sid,
        "selector rerun set equal manifest:",
        same_times,
        "labels match",
        ok_label,
        "/ 50; max |eta - manifest elevation_m| =",
        max_de,
        round(time.time() - t0),
        flush=True,
    )
json.dump(
    {
        "months": months24,
        "sites": {s.study_site_id: list(map(float, s.anchor_lonlat)) for s in cfg.study_sites},
    },
    open(OUT / "config_summary.json", "w"),
)
