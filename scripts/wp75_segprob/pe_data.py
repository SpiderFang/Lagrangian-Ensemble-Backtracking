# ruff: noqa: E501
"""WP-75 試作（幾何期望極座標圖）：五站的觀測扇區計數、幾何與期望值。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wp75_common as C  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402
from lagrangian_backtracking.native_runner import polar_expected as PE  # noqa: E402
from lagrangian_backtracking.native_runner import polar_source as ps  # noqa: E402
from lagrangian_backtracking.native_runner.arc_by_group import group_label_zh  # noqa: E402
from lagrangian_backtracking.native_runner.display_names import material_display_name_zh  # noqa: E402

POSTPROCESS = C.RUN / "finite_depth_stokes/postprocess-coastB"
SEASONS = ("DJF", "MAM", "JJA", "SON")
TIDES = ("spring_proxy", "neap_proxy")


def group_masks(smp) -> tuple[dict[str, np.ndarray], dict[str, str], list[str]]:
    n = smp.n_rows
    g = {"all": np.ones(n, bool)}
    lab = {"all": "全年"}
    for k in SEASONS:
        g[f"season_{k}"] = smp.season == k
        lab[f"season_{k}"] = group_label_zh("season", k)
    for k in TIDES:
        g[f"tide_{k}"] = smp.tide == k
        lab[f"tide_{k}"] = group_label_zh("tide", k)
    mats = sorted(set(smp.material_id.tolist()))
    speed = {m: float(np.median(smp.class_speed_abs_mps[smp.material_id == m])) for m in mats}
    mats = sorted(mats, key=lambda m: (speed[m], m))
    for m in mats:
        g[f"class_{m}"] = smp.material_id == m
        lab[f"class_{m}"] = material_display_name_zh(m)
    return g, lab, mats


def build_all(sites=C.SITES, step_m: float = PE.STEP_M) -> dict:
    regions, site_cfg = C.load_config()
    anchors = F._load_receptor_anchors(C.RELEASE)  # noqa: SLF001
    land = C.load_land()
    samples = ps.load_site_polar_samples(
        POSTPROCESS, {k: tuple(v[:2]) for k, v in anchors.items()}, list(sites)
    )
    out = {}
    for sid in sites:
        sd = C.build_site(sid, land, regions, site_cfg, anchors)
        geo = PE.site_geometry(sd, step_m)
        smp = samples[sid]
        # 錨點必須與分段機率圖一致
        assert (
            abs(smp.anchor_lonlat[0] - sd.anchor[0]) < 1e-9
            and abs(smp.anchor_lonlat[1] - sd.anchor[1]) < 1e-9
        )
        masks, labels, mats = group_masks(smp)
        groups = {}
        for gid, m in masks.items():
            n = int(m.sum())
            counts = np.bincount(smp.sector[m], minlength=PE.N).astype(np.int64)
            n_cross = int((m & (smp.status == C.STATUS_CROSSED)).sum())
            n_coast = int((m & (smp.status == C.STATUS_COAST)).sum())
            obs = 100.0 * counts / n
            pc, pk = 100.0 * n_cross / n, 100.0 * n_coast / n
            exp = PE.expected_pct(geo, pc, pk)
            groups[gid] = dict(
                label=labels[gid],
                n=n,
                counts=counts,
                n_cross=n_cross,
                n_coast=n_coast,
                n_dep=n - n_cross - n_coast,
                obs=obs,
                exp=exp,
                p_cross=pc,
                p_coast=pk,
                old=PE.old_flags(counts, n),
                new=PE.new_flags(obs, exp),
            )
        out[sid] = dict(geo=geo, anchor=sd.anchor, bbox=sd.bbox, groups=groups, mats=mats, sd=sd)
    return out
