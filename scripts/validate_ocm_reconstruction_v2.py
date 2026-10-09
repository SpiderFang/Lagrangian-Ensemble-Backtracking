#!/usr/bin/env python3
"""方法 v2（潮汐調和＋有界殘差橋接）的阻擋式交叉驗證。

做法：在真實資料中隨機挑多個起點，把連續 24 小時與 48 小時「人為挖掉」（來源物件對重建程式
隱藏這些小時），用與正式重建完全相同的程式重建，與被挖掉的真值比較。挖洞的起點必須使前後各
``context_hours`` 小時的真實資料完整存在。為了讓四區在合理時間內完成，空間上以每區若干段
連續節點（所有層與分量）與若干段連續元素（wetdry）作為樣本，不是只抽幾個點。

比較對象（基準）：
* ``replay24``：把前一日同一小時整份複製過來，這正是供應者「重播日」留在舊快取裡的內容；
* ``linear``：只用緊鄰缺口兩個真實小時做線性內插；
* ``harmonic_only``：只有潮汐調和、沒有殘差橋接（消融）；
* 不同 ``context_hours`` 的方法變體。

輸出 JSON：每個變體、缺口長度、欄位的 RMSE、最大絕對誤差、偏差、相關係數、真值標準差，以及
相位（潮位與水平流速的互相關最佳延遲，單位小時）、水平流速最大值（重建與真值）、zcor 垂向順序
違反格點數、NaN 遮罩是否與真值一致。結果只用來評估方法，不宣稱獨立科學驗證。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from lagrangian_backtracking.ocm_reconstruction import HOURLY_NS
from lagrangian_backtracking.ocm_reconstruction_v2 import (
    CleanMonthsSource,
    GapCluster,
    HarmonicBridgeConfig,
    reconstruct_cluster_field,
    repair_zcor_order,
)

FIELDS = ("hvel", "elev", "zcor", "vertical_velocity", "wetdry_elem")
H = int(HOURLY_NS)


class MaskedSource:
    """對重建程式隱藏指定 UTC 的唯讀來源；讀取仍轉給原來源（隱藏的時刻不會被要求）。"""

    def __init__(self, base: CleanMonthsSource, hidden: np.ndarray) -> None:
        self._base = base
        self._times = np.setdiff1d(base.times_utc_ns, hidden)

    @property
    def times_utc_ns(self) -> np.ndarray:
        return self._times.copy()

    @property
    def field_shapes(self):
        return self._base.field_shapes

    def read_times(self, *args, **kwargs):
        return self._base.read_times(*args, **kwargs)


def flat_range(field: str, a: int, b: int, shapes) -> tuple[int, int]:
    """節點（或元素）索引區間轉成該欄位展平特徵區間。"""

    per = int(np.prod(shapes[field], dtype=np.int64)) // int(shapes["elev" if field != "wetdry_elem" else "wetdry_elem"][0])
    return a * per, b * per


def pick_starts(times: np.ndarray, length: int, context: int, count: int, rng: np.random.Generator) -> list[int]:
    """挑起點：缺口 ``length`` 小時及前後各 ``context`` 小時全部存在於真實時間軸。"""

    present = set(int(t) for t in times)
    first, last = int(times[0]), int(times[-1])
    candidates = []
    # 以 6 小時為步長掃描所有起點，要求整個視窗（含兩側 context）皆存在。
    t = first + context * H
    while t + (length + context) * H <= last:
        window = [t + k * H for k in range(-context, length + context)]
        if all(w in present for w in window):
            candidates.append(t)
        t += 6 * H
    if len(candidates) < count:
        raise RuntimeError(f"可用起點只有 {len(candidates)} 個，少於需求 {count}")
    # 依時間均勻分層後隨機，避免全部集中在同一季。
    candidates = np.asarray(candidates, dtype=np.int64)
    strata = np.array_split(candidates, count)
    return [int(rng.choice(s)) for s in strata]


def lag_hours(truth: np.ndarray, recon: np.ndarray) -> float:
    """以 0.1 小時步長找 ``recon`` 相對 ``truth`` 的最佳時間延遲（-3..3 小時）。"""

    n = truth.size
    t = np.arange(n, dtype=np.float64)
    best, best_rmse = 0.0, np.inf
    for lag in np.arange(-3.0, 3.01, 0.1):
        shifted = np.interp(t, t + lag, recon, left=np.nan, right=np.nan)
        ok = np.isfinite(shifted)
        if ok.sum() < max(6, n // 2):
            continue
        rmse = float(np.sqrt(np.mean((shifted[ok] - truth[ok]) ** 2)))
        if rmse < best_rmse:
            best, best_rmse = float(lag), rmse
    return best


class Acc:
    """累積平方誤差、最大誤差、偏差與真值統計。"""

    def __init__(self) -> None:
        self.n = 0
        self.sse = 0.0
        self.sum_err = 0.0
        self.max_abs = 0.0
        self.sum_truth = 0.0
        self.sum_truth_sq = 0.0
        self.seg_rmse: list[float] = []

    def add(self, truth: np.ndarray, est: np.ndarray) -> None:
        ok = np.isfinite(truth) & np.isfinite(est)
        if not ok.any():
            return
        err = (est[ok] - truth[ok]).astype(np.float64)
        self.n += int(err.size)
        self.sse += float(np.sum(err**2))
        self.sum_err += float(err.sum())
        self.max_abs = max(self.max_abs, float(np.abs(err).max()))
        tv = truth[ok].astype(np.float64)
        self.sum_truth += float(tv.sum())
        self.sum_truth_sq += float(np.sum(tv**2))
        self.seg_rmse.append(float(np.sqrt(np.mean(err**2))))

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        mean_truth = self.sum_truth / self.n
        var_truth = max(self.sum_truth_sq / self.n - mean_truth**2, 0.0)
        return {
            "n": self.n,
            "rmse": (self.sse / self.n) ** 0.5,
            "bias": self.sum_err / self.n,
            "max_abs_err": self.max_abs,
            "truth_std": var_truth**0.5,
            "seg_rmse_min": min(self.seg_rmse),
            "seg_rmse_median": float(np.median(self.seg_rmse)),
            "seg_rmse_max": max(self.seg_rmse),
        }


def run(args: argparse.Namespace) -> dict:
    domain_root = Path(args.domain_root)
    bad = np.asarray(json.load(open(args.bad_hours))["bad"], dtype=np.int64)
    base = CleanMonthsSource(domain_root, exclude_utc_ns=bad)
    shapes = base.field_shapes
    node_count = shapes["elev"][0]
    face_count = shapes["wetdry_elem"][0]
    rng = np.random.default_rng(args.seed)
    blocks_n = sorted(int(x) for x in rng.integers(0, node_count - args.node_block, size=args.node_blocks))
    blocks_f = sorted(int(x) for x in rng.integers(0, face_count - args.face_block, size=args.node_blocks))
    variants = {
        "primary_w168": HarmonicBridgeConfig(context_hours=168, source_fingerprint="validation"),
        "w96": HarmonicBridgeConfig(context_hours=96, source_fingerprint="validation"),
        "w120": HarmonicBridgeConfig(context_hours=120, source_fingerprint="validation"),
        "edge6_w168": HarmonicBridgeConfig(context_hours=168, edge_hours=6, edge_decay_hours=3.0, source_fingerprint="validation"),
        "w240": HarmonicBridgeConfig(context_hours=240, source_fingerprint="validation"),
        "harmonic_only_w168": HarmonicBridgeConfig(context_hours=168, residual_bridge=False, source_fingerprint="validation"),
        "few_constituents_w168": HarmonicBridgeConfig(
            context_hours=168, constituents=("M2", "S2", "N2", "K1", "O1", "M4"), source_fingerprint="validation"
        ),
    }
    times = base.times_utc_ns
    results: dict = {"domain": domain_root.name, "node_blocks": blocks_n, "node_block_size": args.node_block,
                     "face_blocks": blocks_f, "face_block_size": args.face_block, "segments": {}, "metrics": {}}
    starts_all = pick_starts(times, max(args.lengths), max(v.context_hours for v in variants.values()), args.segments, rng)
    for length in args.lengths:
        starts = starts_all
        results["segments"][str(length)] = [int(s) for s in starts]
        acc: dict = {}
        extra: dict = {}
        for seg_i, s in enumerate(starts):
            t0 = time.time()
            missing = s + np.arange(length, dtype=np.int64) * H
            masked = MaskedSource(base, missing)
            cluster = GapCluster(
                gap_id=f"val_{length}_{s}",
                missing_times_ns=missing,
                left_time_ns=int(missing[0]) - H,
                right_time_ns=int(missing[-1]) + H,
                superseded_ns=np.empty(0, dtype=np.int64),
            )
            truth_cache: dict = {}
            for field in FIELDS:
                parts = []
                rngs = blocks_f if field == "wetdry_elem" else blocks_n
                size = args.face_block if field == "wetdry_elem" else args.node_block
                for a in rngs:
                    fr = flat_range(field, a, a + size, shapes)
                    parts.append(base.read_times(missing, field, flat_start=fr[0], flat_stop=fr[1]).astype(np.float64))
                truth_cache[field] = parts
            baselines: dict = {}
            for field in FIELDS:
                rngs = blocks_f if field == "wetdry_elem" else blocks_n
                size = args.face_block if field == "wetdry_elem" else args.node_block
                rep, lin = [], []
                for a in rngs:
                    fr = flat_range(field, a, a + size, shapes)
                    prev = base.read_times(missing - 24 * H, field, flat_start=fr[0], flat_stop=fr[1]).astype(np.float64)
                    left = base.read_times(np.asarray([missing[0] - H]), field, flat_start=fr[0], flat_stop=fr[1]).astype(np.float64)
                    right = base.read_times(np.asarray([missing[-1] + H]), field, flat_start=fr[0], flat_stop=fr[1]).astype(np.float64)
                    w = (np.arange(1, length + 1, dtype=np.float64) / (length + 1))[:, None]
                    rep.append(prev)
                    lin.append(left + w * (right - left))
                baselines[field] = {"replay24": rep, "linear": lin}
            for name, cfg in variants.items():
                for field in FIELDS:
                    rngs = blocks_f if field == "wetdry_elem" else blocks_n
                    size = args.face_block if field == "wetdry_elem" else args.node_block
                    est_parts = []
                    for a in rngs:
                        fr = flat_range(field, a, a + size, shapes)
                        out = np.empty((length, fr[1] - fr[0]), dtype=np.float64)

                        def sink(start, stop, block, out=out, base0=fr[0]):
                            out[:, start - base0 : stop - base0] = block

                        reconstruct_cluster_field(masked, cluster, field, cfg, sink=sink, flat_range=fr)
                        est_parts.append(out)
                    key = (name, field)
                    acc.setdefault(key, Acc()).add(np.concatenate(truth_cache[field], axis=1), np.concatenate(est_parts, axis=1))
                    if name == "primary_w168":
                        truth = np.concatenate(truth_cache[field], axis=1)
                        est = np.concatenate(est_parts, axis=1)
                        if field == "hvel":
                            ts = np.hypot(truth[:, 0::2], truth[:, 1::2]); es = np.hypot(est[:, 0::2], est[:, 1::2])
                            extra.setdefault("hvel_speed_max_truth", []).append(float(np.nanmax(ts)))
                            extra.setdefault("hvel_speed_max_recon", []).append(float(np.nanmax(es)))
                            # 潮流相位：節點×層平均 u 的時間序列
                            extra.setdefault("hvel_u_lag_h", []).append(lag_hours(np.nanmean(truth[:, 0::2], axis=1), np.nanmean(est[:, 0::2], axis=1)))
                            extra.setdefault("hvel_v_lag_h", []).append(lag_hours(np.nanmean(truth[:, 1::2], axis=1), np.nanmean(est[:, 1::2], axis=1)))
                            extra.setdefault("nan_mask_match_hvel", []).append(bool(np.array_equal(np.isnan(truth), np.isnan(est))))
                        elif field == "elev":
                            extra.setdefault("elev_lag_h", []).append(lag_hours(truth.mean(axis=1), est.mean(axis=1)))
                            extra.setdefault("elev_corr", []).append(float(np.corrcoef(truth.ravel(), est.ravel())[0, 1]))
                            extra.setdefault("elev_max_truth", []).append(float(truth.max()))
                            extra.setdefault("elev_max_recon", []).append(float(est.max()))
                            extra.setdefault("elev_min_truth", []).append(float(truth.min()))
                            extra.setdefault("elev_min_recon", []).append(float(est.min()))
                        elif field == "zcor":
                            viol = 0
                            for a_i, a in enumerate(rngs):
                                z = est_parts[a_i].reshape(length, size, -1)
                                viol += repair_zcor_order(z.copy())
                            extra.setdefault("zcor_order_violations_before_repair", []).append(int(viol))
                            extra.setdefault("nan_mask_match_zcor", []).append(bool(np.array_equal(np.isnan(truth), np.isnan(est))))
                        elif field == "vertical_velocity":
                            extra.setdefault("w_max_truth", []).append(float(np.nanmax(np.abs(truth))))
                            extra.setdefault("w_max_recon", []).append(float(np.nanmax(np.abs(est))))
                        elif field == "wetdry_elem":
                            extra.setdefault("wetdry_mismatch_fraction", []).append(float(np.mean(truth != est)))
                            extra.setdefault("wetdry_mismatch_fraction_replay24", []).append(float(np.mean(truth != np.concatenate(baselines[field]["replay24"], axis=1))))
                # 基準只算一次（與變體無關）
            for field in FIELDS:
                truth = np.concatenate(truth_cache[field], axis=1)
                for bname in ("replay24", "linear"):
                    est = np.concatenate(baselines[field][bname], axis=1)
                    acc.setdefault((bname, field), Acc()).add(truth, est)
                if field == "hvel":
                    for bname in ("replay24", "linear"):
                        est = np.concatenate(baselines[field][bname], axis=1)
                        extra.setdefault(f"hvel_speed_max_{bname}", []).append(float(np.nanmax(np.hypot(est[:, 0::2], est[:, 1::2]))))
            print(f"{domain_root.name[:12]} L={length} seg {seg_i + 1}/{len(starts)} {time.time() - t0:.1f}s", flush=True)
        results["metrics"][str(length)] = {f"{k[0]}|{k[1]}": v.summary() for k, v in acc.items()}
        results.setdefault("extra", {})[str(length)] = extra
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain-root", required=True)
    p.add_argument("--bad-hours", required=True, help="dups_all.json 中該區的 bad 清單 JSON：{\"bad\": [...]}")
    p.add_argument("--output", required=True)
    p.add_argument("--segments", type=int, default=12)
    p.add_argument("--lengths", type=int, nargs="+", default=[24, 48])
    p.add_argument("--node-blocks", type=int, default=4)
    p.add_argument("--node-block", type=int, default=800)
    p.add_argument("--face-block", type=int, default=1600)
    p.add_argument("--seed", type=int, default=20261009)
    args = p.parse_args()
    result = run(args)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
