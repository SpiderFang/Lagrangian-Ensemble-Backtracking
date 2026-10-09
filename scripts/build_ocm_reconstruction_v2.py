#!/usr/bin/env python3
"""建立四區 OCM 重建 patch（方法 v2：潮汐調和＋有界殘差橋接），含重播小時的取代宣告。

輸入
----
* ``--source-root``：本機 ``ocm_native`` 根（例如 v5），內有 ``<flow_id>/months/YYYYMM``。
* ``--replay-dir``：每區一個 ``bad_replay_<flow_id>.json``，內容 ``{"bad": [UTC 奈秒...]}``，
  是已確認為「供應者日檔重播前一日」的觀測小時（內容不是該小時真實流場）。
* ``--old-patch-root``：舊 patch 根；只用來原樣沿用本機沒有資料的月份（早於本機資料第一個月）。

做法
----
1. 以排除重播小時後的真實時間軸找出所有內部缺口（含原本就缺的小時），每段連續缺時為一個
   缺口群；重播小時與原缺小時合併成同一群一起重建。
2. 每個（缺口群、欄位、特徵區塊）為一個工作，多程序平行；結果直接寫入各月份 partial 目錄的
   memmap（不經過整月常駐記憶體）。
3. zcor 垂向順序修正、品質旗標、metadata（含 ``supersedes_observed`` 清單）與 sha256，
   最後以原子 rename 發布月份，寫 domain manifest 與整個根的 index，並用
   ``validate_reconstruction_patch`` 驗證。

不修改來源、不覆寫既有輸出（輸出根必須不存在該 domain）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lagrangian_backtracking.ocm_reconstruction import (
    HOURLY_NS,
    ReconstructionError,
    _canonical_json_bytes,
    _month_for_ns,
    _source_mesh_fingerprint,
    _write_domain_manifest,
    validate_reconstruction_patch,
)
from lagrangian_backtracking.ocm_reconstruction_v2 import (
    QUALITY_HARMONIC_BRIDGE,
    QUALITY_SUPERSEDES_OBSERVED,
    CleanMonthsSource,
    GapCluster,
    HarmonicBridgeConfig,
    allocate_partial_month,
    finalize_month,
    find_gap_clusters,
    reconstruct_cluster_field,
    repair_zcor_order,
    write_root_index,
)

FIELDS = ("hvel", "vertical_velocity", "zcor", "elev", "wetdry_elem")


def source_fingerprint_v2(domain_root: Path, mesh_fingerprint: str, bad_hours: np.ndarray) -> str:
    """來源版本指紋（與 ``_source_domain_fingerprint`` 同一做法，但只含本機根實際有的 5 個欄位）。

    本機 v5 根沒有 ``diffusivity.npy``（執行期不讀 OCM 擴散係數），所以不能直接用原函式。
    指紋納入網格、逐月時間軸與 metadata bytes、各陣列相對路徑／大小／header shape 與 dtype，
    再納入「被宣告為重播的 UTC 清單」，因為它決定哪些來源列被取代而不用。
    """

    digest = hashlib.sha256()
    digest.update(mesh_fingerprint.encode("ascii"))
    for directory in sorted(p for p in (domain_root / "months").iterdir() if p.is_dir()):
        digest.update(directory.name.encode("ascii"))
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            digest.update(metadata_path.read_bytes())
        for name in ("time_utc_ns", *FIELDS):
            path = directory / f"{name}.npy"
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            digest.update(str(path.relative_to(domain_root)).encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            if name == "time_utc_ns":
                digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    digest.update(b"replay_hours_declared_superseded")
    digest.update(np.ascontiguousarray(np.sort(bad_hours.astype(np.int64))).tobytes(order="C"))
    return digest.hexdigest()
_SRC: dict[str, CleanMonthsSource] = {}


def _source(domain_root: str, exclude_path: str) -> CleanMonthsSource:
    """每個工作程序只開一次來源（memmap 與時間索引）。"""

    key = f"{domain_root}|{exclude_path}"
    if key not in _SRC:
        bad = np.asarray(json.load(open(exclude_path))["bad"], dtype=np.int64)
        _SRC[key] = CleanMonthsSource(domain_root, exclude_utc_ns=bad)
    return _SRC[key]


def _work(task: dict) -> dict:
    """單一工作：重建某缺口群某欄位的一段特徵範圍，寫入目的 memmap。"""

    cfg = HarmonicBridgeConfig(**{**task["config"], "constituents": tuple(task["config"]["constituents"])})
    src = _source(task["domain_root"], task["exclude_path"])
    cluster = GapCluster(
        gap_id=task["gap_id"],
        missing_times_ns=np.asarray(task["missing"], dtype=np.int64),
        left_time_ns=task["left"],
        right_time_ns=task["right"],
        superseded_ns=np.empty(0, dtype=np.int64),
    )
    field = task["field"]
    target = np.load(task["target_path"], mmap_mode="r+")
    view = target.reshape(target.shape[0], -1)
    r0 = task["row0"]
    rows = len(task["missing"])

    def sink(start: int, stop: int, block: np.ndarray) -> None:
        view[r0 : r0 + rows, start:stop] = block.astype(np.float32)

    stats = reconstruct_cluster_field(src, cluster, field, cfg, sink=sink, flat_range=(task["start"], task["stop"]))
    target.flush()
    del view, target
    return {"gap_id": task["gap_id"], "field": field, **stats}


def build_domain(args: argparse.Namespace, flow_id: str, pool: ProcessPoolExecutor, log) -> dict:
    """建立一個 flow domain 的 patch 並回傳 domain manifest。"""

    domain_root = Path(args.source_root) / flow_id
    exclude_path = str(Path(args.replay_dir) / f"bad_replay_{flow_id}.json")
    bad = np.asarray(json.load(open(exclude_path))["bad"], dtype=np.int64)
    cfg = HarmonicBridgeConfig(
        context_hours=args.context_hours,
        min_context_hours=args.min_context_hours,
        edge_hours=args.edge_hours,
        edge_decay_hours=args.edge_decay_hours,
        feature_block_size=args.feature_block_size,
        source_fingerprint="pending",
    )
    mesh_fp = _source_mesh_fingerprint(domain_root / "grid")
    src_fp = source_fingerprint_v2(domain_root, mesh_fp, bad)
    cfg = HarmonicBridgeConfig(**{**asdict(cfg), "constituents": cfg.constituents, "source_fingerprint": src_fp})
    clean = CleanMonthsSource(domain_root, exclude_utc_ns=bad)
    clusters = find_gap_clusters(clean.times_utc_ns, bad, flow_id=flow_id)
    total_missing = int(sum(c.missing_times_ns.size for c in clusters))
    log(f"[{flow_id}] 缺口群 {len(clusters)} 段，共 {total_missing} 小時；其中重播取代 {int(sum(c.superseded_ns.size for c in clusters))} 小時")
    for c in clusters:
        months = {_month_for_ns(int(t)) for t in c.missing_times_ns}
        if len(months) != 1:
            raise ReconstructionError(f"缺口群跨月，目前不支援：{c.gap_id}")
    out_root = Path(args.output_root)
    domain_out = out_root / flow_id
    if domain_out.exists():
        raise ReconstructionError(f"輸出 domain 已存在，拒絕覆寫：{domain_out}")
    months_root = domain_out / "months"
    months_root.mkdir(parents=True)
    # 以月份分組
    by_month: dict[str, list[GapCluster]] = {}
    for c in clusters:
        by_month.setdefault(_month_for_ns(int(c.missing_times_ns[0])), []).append(c)
    shapes, dtypes = clean.field_shapes, clean.field_dtypes
    partials: dict[str, Path] = {}
    row0: dict[str, int] = {}
    month_times: dict[str, np.ndarray] = {}
    for month, items in sorted(by_month.items()):
        times = np.concatenate([c.missing_times_ns for c in items])
        order = np.argsort(times)
        assert np.all(order == np.arange(times.size)), "缺口群需已依時間排序"
        month_times[month] = times
        partial = months_root / f".{month}.partial-{uuid.uuid4().hex}"
        allocate_partial_month(partial, times, shapes, dtypes)
        partials[month] = partial
        offset = 0
        for c in items:
            row0[c.gap_id] = offset
            offset += c.missing_times_ns.size
    # 工作清單
    tasks = []
    for month, items in sorted(by_month.items()):
        for c in items:
            for field in FIELDS:
                total = int(np.prod(shapes[field], dtype=np.int64))
                chunk = cfg.feature_block_size * args.blocks_per_task
                for start in range(0, total, chunk):
                    tasks.append(
                        {
                            "domain_root": str(domain_root),
                            "exclude_path": exclude_path,
                            "config": asdict(cfg),
                            "gap_id": c.gap_id,
                            "missing": [int(t) for t in c.missing_times_ns],
                            "left": c.left_time_ns,
                            "right": c.right_time_ns,
                            "field": field,
                            "start": start,
                            "stop": min(total, start + chunk),
                            "target_path": str(partials[month] / f"{field}.npy"),
                            "row0": row0[c.gap_id],
                        }
                    )
    log(f"[{flow_id}] 工作數 {len(tasks)}")
    t0 = time.time()
    futures = [pool.submit(_work, t) for t in tasks]
    done = 0
    inconsistent: dict[str, int] = {}
    for fut in as_completed(futures):
        res = fut.result()
        done += 1
        inconsistent[res["field"]] = inconsistent.get(res["field"], 0) + res["inconsistent_features"]
        if done % 25 == 0 or done == len(futures):
            log(f"[{flow_id}] 完成 {done}/{len(futures)}（{time.time() - t0:.0f}s）")
    # 後處理：zcor 垂向順序、品質旗標、metadata、發布
    month_metadata = []
    repaired_total = 0
    for month, items in sorted(by_month.items()):
        partial = partials[month]
        zc = np.load(partial / "zcor.npy", mmap_mode="r+")
        repaired = repair_zcor_order(zc)
        zc.flush()
        del zc
        repaired_total += repaired
        times = month_times[month]
        superseded = np.concatenate([c.superseded_ns for c in items]) if items else np.empty(0, np.int64)
        flags = np.full(times.size, 2 | 8, dtype=np.uint16) | QUALITY_HARMONIC_BRIDGE
        flags[np.isin(times, superseded)] |= QUALITY_SUPERSEDES_OBSERVED
        details = {
            "clusters": [
                {
                    "gap_id": c.gap_id,
                    "hours": int(c.missing_times_ns.size),
                    "superseded_hours": int(c.superseded_ns.size),
                    "left_support_utc_ns": int(c.left_time_ns),
                    "right_support_utc_ns": int(c.right_time_ns),
                }
                for c in items
            ],
            "zcor_order_repaired_cells": int(repaired),
            "partial_nan_features_by_field": inconsistent,
        }
        meta = finalize_month(
            partial,
            months_root / month,
            flow_id=flow_id,
            month=month,
            gap_ids=[c.gap_id for c in items],
            superseded_ns=superseded,
            quality_flags=flags,
            config=cfg,
            source_fingerprint=src_fp,
            extra=details,
        )
        month_metadata.append(meta)
        log(f"[{flow_id}] 發布月份 {month}：{times.size} 列（取代重播 {superseded.size}）；zcor 修正 {repaired}")
    # 沿用舊 patch 中早於本機資料的月份（硬連結）
    first_local = min(p.name for p in (domain_root / "months").iterdir() if p.is_dir())
    old_domain = Path(args.old_patch_root) / flow_id / "months"
    if old_domain.is_dir():
        for old in sorted(old_domain.iterdir()):
            if old.is_dir() and old.name < first_local and not (months_root / old.name).exists():
                dest = months_root / old.name
                dest.mkdir()
                for f in old.iterdir():
                    os.link(f, dest / f.name)
                month_metadata.append(json.loads((old / "metadata.json").read_text(encoding="utf-8")))
                log(f"[{flow_id}] 沿用舊月份 {old.name}（硬連結）")
    manifest = _write_domain_manifest(
        out_root, flow_id, month_metadata, config=cfg, source_fingerprint=src_fp, mesh_fingerprint=mesh_fp
    )
    validate_reconstruction_patch(domain_out)
    sha = (domain_out / "reconstruction-manifest.json.sha256").read_text().strip()
    return {"flow_id": flow_id, "manifest_sha256": sha}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--replay-dir", required=True)
    p.add_argument("--old-patch-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--flow-id", action="append")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--context-hours", type=int, default=96)
    p.add_argument("--min-context-hours", type=int, default=24)
    p.add_argument("--edge-hours", type=int, default=3)
    p.add_argument("--edge-decay-hours", type=float, default=1.5)
    p.add_argument("--feature-block-size", type=int, default=131072)
    p.add_argument("--blocks-per-task", type=int, default=2)
    args = p.parse_args()
    flow_ids = args.flow_id or sorted(p_.name for p_ in Path(args.source_root).iterdir() if (p_ / "months").is_dir())

    def log(msg: str) -> None:
        print(time.strftime("%F %T"), msg, flush=True)

    ctx = mp.get_context("spawn")
    manifests = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for flow_id in flow_ids:
            manifests.append(build_domain(args, flow_id, pool, log))
    write_root_index(Path(args.output_root), manifests)
    log("完成：root index 已寫入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
