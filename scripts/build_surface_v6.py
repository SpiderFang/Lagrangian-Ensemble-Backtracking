#!/usr/bin/env python3
"""WP-53：建立「無缺時」的 ocm_surface v6 根（供 horizon-suite-create 的標籤與到達時刻選取使用）。

來源：``--nas-root``（NAS 的 ocm_surface）與 ``--gaponly-root``（WP-52 以供應者後補日檔補上的月份，
覆蓋 NAS 同名月份）。做法與 native 的 patch 一致（方法 ``ocm_harmonic_residual_bridge_bounded_v2``）：

1. 以「排除重播小時」後的真實表層時間軸找出所有缺口（含原本就缺的小時），
   每段缺口以兩側各最多 96 小時真實表層資料做潮汐調和＋有界殘差橋接，
   ``eta_m``、``surface_z`` 為純量；``u_surface_mps``／``v_surface_mps`` 成對並限制流速大小
   不超過訓練視窗各格點最大值；
2. 重播小時（供應者日檔複製前一日）的列以重建值取代，不保留複製值；
3. 缺的整點插入對應月份（月份規則與原月份一致：該整點之前 1 奈秒所屬 UTC 月）；
4. ``valid_mask_surface`` 與 ``qc_flags`` 沿用原資料語意：兩者都只由靜態網格遮罩決定
   （靜態無效格 qc=1，其餘 0），與原始月份逐位元一致（本腳本在載入時驗證這點）。

處理 2024-01 起的全部月份（母體建置的標籤序列從 2024-01 開始，時間軸要完整）。不修改來源。
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from lagrangian_backtracking.ocm_reconstruction_v2 import (
    GapCluster, HarmonicBridgeConfig, context_times_for_cluster, find_gap_clusters, reconstruct_feature_block,
)

H = 3600 * 10**9
ARR = ("eta_m", "u_surface_mps", "v_surface_mps", "surface_z")
FIRST_MONTH = "202401"


def month_of_hour_end(t: int) -> str:
    """整點資料的月份：與原快取相同，以該整點之前 1 奈秒所屬的 UTC 月份為準。"""
    return datetime.fromtimestamp((int(t) - 1) / 1e9, tz=UTC).strftime("%Y%m")


def src_dir(args, dom: str, m: str) -> Path:
    g = Path(args.gaponly_root) / dom / "months" / m
    return g if g.is_dir() else Path(args.nas_root) / dom / "months" / m


def build_domain(args, dom: str, log) -> dict:
    out = Path(args.output_root) / dom
    if out.exists():
        raise SystemExit(f"輸出已存在，拒絕覆寫：{out}")
    (out / "months").mkdir(parents=True)
    shutil.copytree(Path(args.nas_root) / dom / "grid", out / "grid")
    static = np.load(out / "grid" / "mask_static.npy").astype(bool)
    bad = np.asarray(json.load(open(Path(args.replay_dir) / f"bad_replay_{dom}.json"))["bad"], dtype=np.int64)
    bad_set = {int(v) for v in bad}
    months = sorted(p.name for p in (Path(args.nas_root) / dom / "months").iterdir() if p.is_dir())
    cfg = HarmonicBridgeConfig(context_hours=96, min_context_hours=24, edge_hours=3, edge_decay_hours=1.5)
    data: dict[str, dict] = {}
    for m in months:
        if m < FIRST_MONTH:
            continue
        d = src_dir(args, dom, m)
        t = np.load(d / "time_utc_ns.npy")
        if np.any(np.diff(t) <= 0):
            raise SystemExit(f"{dom}/{m} 時間軸非嚴格遞增")
        rec = {"time": t, "dir": d}
        for n in ARR:
            rec[n] = np.load(d / f"{n}.npy")
        v = np.load(d / "valid_mask_surface.npy")
        q = np.load(d / "qc_flags.npy")
        # 驗證：valid 與 qc 只由靜態遮罩決定（逐列）
        if not (np.all(v == static[None]) and np.all(q == (~static).astype(np.uint16)[None])):
            raise SystemExit(f"{dom}/{m} 的 valid/qc 不只由靜態遮罩決定，需要另行處理")
        data[m] = rec
    # 乾淨時間軸（prefer-last，排除重播小時）
    loc: dict[int, tuple[str, int]] = {}
    for m in sorted(data):
        for i, t in enumerate(data[m]["time"]):
            if int(t) not in bad_set:
                loc[int(t)] = (m, i)
    clean = np.asarray(sorted(loc), dtype=np.int64)
    clusters = find_gap_clusters(clean, bad, flow_id=dom)
    log(f"[{dom}] 乾淨表層小時 {clean.size}，缺口群 {len(clusters)}，共 {sum(c.missing_times_ns.size for c in clusters)} 小時，其中重播取代 {sum(c.superseded_ns.size for c in clusters)}")
    nrow, ncol = static.shape
    recon: dict[int, dict[str, np.ndarray]] = {}
    maxspeed_real = 0.0
    stats = []
    for c in clusters:
        left, right = context_times_for_cluster(clean, c, cfg)
        train_t = np.concatenate((left, right))
        centre = 0.5 * (float(c.missing_times_ns[0]) + float(c.missing_times_ns[-1]))
        th = (train_t.astype(np.float64) - centre) / H
        gh = (c.missing_times_ns.astype(np.float64) - centre) / H

        def rows(name):
            return np.stack([data[loc[int(t)][0]][name][loc[int(t)][1]] for t in train_t]).astype(np.float64)

        eta, _ = reconstruct_feature_block(th, rows("eta_m").reshape(train_t.size, -1), gh, cfg, kind="scalar")
        zz, _ = reconstruct_feature_block(th, rows("surface_z").reshape(train_t.size, -1), gh, cfg, kind="scalar")
        u, v = rows("u_surface_mps"), rows("v_surface_mps")
        uv = np.stack([u, v], axis=-1).reshape(train_t.size, -1)
        pred, _ = reconstruct_feature_block(th, uv, gh, cfg, kind="hvel")
        pred = pred.reshape(gh.size, nrow * ncol, 2)
        for k, t in enumerate(c.missing_times_ns):
            recon[int(t)] = {
                "eta_m": eta[k].reshape(nrow, ncol).astype(np.float32),
                "surface_z": zz[k].reshape(nrow, ncol).astype(np.float32),
                "u_surface_mps": pred[k, :, 0].reshape(nrow, ncol).astype(np.float32),
                "v_surface_mps": pred[k, :, 1].reshape(nrow, ncol).astype(np.float32),
            }
        stats.append({"gap_id": c.gap_id, "hours": int(c.missing_times_ns.size), "superseded": int(c.superseded_ns.size)})
    # 真實最大流速（乾淨列）與重建最大流速
    for m in data:
        sp = np.hypot(data[m]["u_surface_mps"], data[m]["v_surface_mps"])
        maxspeed_real = max(maxspeed_real, float(np.nanmax(sp[[i for i, t in enumerate(data[m]["time"]) if int(t) not in bad_set]])))
    maxspeed_rec = max((float(np.nanmax(np.hypot(r["u_surface_mps"], r["v_surface_mps"]))) for r in recon.values()), default=0.0)
    # 輸出月份
    present_any = {int(t) for m in data for t in data[m]["time"]}
    new_by_month: dict[str, list[int]] = {}
    for t in recon:
        if t in present_any:
            continue  # 重播小時：原月份已有該列，只取代數值，不另插入
        new_by_month.setdefault(month_of_hour_end(t), []).append(t)
    placed = set()
    summary = {}
    for m in months:
        dest = out / "months" / m
        if m < FIRST_MONTH:
            shutil.copytree(Path(args.nas_root) / dom / "months" / m, dest)
            continue
        rec = data[m]
        t0 = rec["time"]
        add = sorted(t for t in new_by_month.get(m, []) if t not in placed)
        # 注意：重播小時若同時存在於原月份，列保留並取代數值；缺的整點才插入
        existing = {int(t) for t in t0}
        add = [t for t in add if t not in existing]
        tn = np.sort(np.concatenate([t0, np.asarray(add, dtype=np.int64)]), kind="stable")
        T = tn.size
        dest.mkdir(parents=True)
        arrs = {n: np.empty((T, nrow, ncol), dtype=np.float32) for n in ARR}
        replaced = 0
        for j, t in enumerate(tn):
            t = int(t)
            if t in recon and (t in bad_set or t not in existing):
                for n in ARR:
                    arrs[n][j] = recon[t][n]
                replaced += 1
            else:
                i = int(np.searchsorted(t0, t))
                for n in ARR:
                    arrs[n][j] = rec[n][i]
        for n in ARR:
            # 靜態無效格維持 NaN（重建函式已對全 NaN 特徵輸出 NaN）
            np.save(dest / f"{n}.npy", arrs[n])
        np.save(dest / "time_utc_ns.npy", tn.astype(np.int64))
        np.save(dest / "valid_mask_surface.npy", np.broadcast_to(static, (T, nrow, ncol)).copy())
        np.save(dest / "qc_flags.npy", np.broadcast_to((~static).astype(np.uint16), (T, nrow, ncol)).copy())
        md = json.loads((rec["dir"] / "metadata.json").read_text(encoding="utf-8"))
        for n in list(md.get("arrays", {})):
            if "shape" in md["arrays"][n]:
                md["arrays"][n]["shape"] = [T] + list(md["arrays"][n]["shape"][1:])
        if replaced:
            md["wp53_reconstruction"] = {
                "method": cfg.method_id,
                "reconstructed_rows": int(replaced),
                "superseded_replay_rows": int(sum(1 for t in tn if int(t) in bad_set)),
                "inserted_rows": len(add),
                "note": "重播小時與原缺整點以表層資料的潮汐調和＋有界殘差橋接重建；其餘列與來源逐位元相同",
            }
        (dest / "metadata.json").write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8")
        qr = rec["dir"] / "quality_report.json"
        if qr.is_file():
            shutil.copy2(qr, dest / "quality_report.json")
        summary[m] = {"rows": int(T), "replaced_or_inserted": int(replaced)}
        placed.update(add)
        del arrs
    return {"domain": dom, "clusters": stats, "max_speed_real": maxspeed_real, "max_speed_reconstructed": maxspeed_rec, "months": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nas-root", required=True)
    ap.add_argument("--gaponly-root", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    def log(s):
        print(time.strftime("%F %T"), s, flush=True)

    rep = build_domain(args, args.domain, log)
    Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"完成 {args.domain}：真實最大流速 {rep['max_speed_real']:.3f}，重建最大 {rep['max_speed_reconstructed']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
