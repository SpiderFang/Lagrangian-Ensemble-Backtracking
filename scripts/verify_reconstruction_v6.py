#!/usr/bin/env python3
"""WP-53：以正式載入路徑驗證 v6 重建 patch。

對每個 flow domain：
1. 以 ``_ReconstructionPatchStore`` 載入 patch（驗 checksum、root index、網格指紋）；
2. ``inventory_forcing_gaps``：期望期間內未登錄缺口數必須為 0，並回報觀測／重建小時數；
3. ``NativeForcingBridgeReader`` 逐一解析所有重建小時（精確整點與其後半小時），回報有限值比例、
   最大流速、缺值遮罩；任何例外視為失敗。
不寫入任何檔案（除 ``--report``）。
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from lagrangian_backtracking.forcing_window import _ReconstructionPatchStore
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner.forcing_bridge import NativeForcingBridgeReader, inventory_forcing_gaps

H = 3600 * 10**9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocm-native-root", required=True)
    ap.add_argument("--reconstruction-root", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--first-month", default="202407")
    a = ap.parse_args()
    flow = a.domain
    root = Path(a.ocm_native_root) / flow
    mesh = NativeMesh.from_directory(root / "grid", projection=DomainProjection(121.0, 24.0))
    store = _ReconstructionPatchStore(root=a.reconstruction_root, flow_domain_id=flow, mesh=mesh)
    months = sorted(p.name for p in (root / "months").iterdir() if p.is_dir() and p.name >= a.first_month)
    month_dirs = {m: root / "months" / m for m in months}
    patch_months = sorted(p.name for p in (Path(a.reconstruction_root) / flow / "months").iterdir() if p.is_dir())
    patches = {m: store.load_month(m) for m in patch_months if m in month_dirs}
    patches = {m: p for m, p in patches.items() if p is not None}
    t_first = min(int(np.load(month_dirs[m] / "time_utc_ns.npy")[0]) for m in months)
    t_last = max(int(np.load(month_dirs[m] / "time_utc_ns.npy")[-1]) for m in months)
    inv = inventory_forcing_gaps(month_dirs, expected_start_utc_ns=t_first, expected_end_utc_ns=t_last, patches=patches, flow_id=flow)
    rep = {"domain": flow, "expected_start": str(np.datetime64(t_first, "ns")), "expected_end": str(np.datetime64(t_last, "ns")),
           "expected_hours": inv.expected_hour_count, "observed_hours": inv.observed_hour_count,
           "reconstructed_hours": inv.reconstructed_hour_count, "unregistered_gaps": len(inv.unregistered_gaps)}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    reader = NativeForcingBridgeReader(month_dirs, ibf=-1, patches=patches, load_diffusivity=False)
    rec_times = np.sort(np.concatenate([np.asarray(p.time_utc_ns, dtype=np.int64) for p in patches.values()]))
    worst = 0.0
    bad = []
    fin_min = 1.0
    t0 = time.time()
    for k, t in enumerate(rec_times):
        for off in (0, H // 2):
            try:
                p = reader.resolve(int(t) + off)
                for arr in (p.u_field1, p.u_field2, p.v_field1, p.v_field2):
                    worst = max(worst, float(np.nanmax(np.abs(arr))))
                    fin_min = min(fin_min, float(np.isfinite(arr).mean()))
            except Exception as exc:  # noqa: BLE001
                bad.append((int(t) + off, str(exc)[:160]))
        if k % 40 == 0:
            print(f"  已解析 {k}/{rec_times.size}（{time.time() - t0:.0f}s）", flush=True)
    rep.update({"resolved_reconstructed_hours": int(rec_times.size), "resolve_failures": len(bad), "failure_examples": bad[:5],
                "max_abs_component_mps": worst, "min_finite_fraction": fin_min,
                "patch_months": sorted(patches), "all_on_canonical_axis": bool(inv.unregistered_gaps == () or len(inv.unregistered_gaps) == 0)})
    Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 1 if (bad or inv.unregistered_gaps) else 0


if __name__ == "__main__":
    sys.exit(main())
