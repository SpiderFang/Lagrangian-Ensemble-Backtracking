#!/usr/bin/env python3
"""建立四區 OCM sparse reconstruction patch。

本命令只讀取已驗收 OCM schema 3 的 ``<flow_id>/months/YYYYMM``，並將缺少的逐時
rows 寫成獨立 patch；不修改 source cache，也不把原始月份複製到輸出。實際大型資料
由 ``NpyDomainSource`` memory-map 加上 feature block 讀取，因此可在 SERVER 上以一段
缺口一段缺口處理，而不是一次載入兩年流場。

輸出根目錄拓撲：

``<output-root>/<flow_id>/months/YYYYMM/{...}.npy``
``<output-root>/<flow_id>/reconstruction-manifest.json[.sha256]``
``<output-root>/reconstruction-index.json[.sha256]``

這個建置器不宣稱重建產品已通過獨立科學驗證；blocked-mask 合成驗證由核心 API
另行執行並保存 caller 明示的 validation payload。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from lagrangian_backtracking.ocm_reconstruction import (
    NpyDomainSource,
    ReconstructionConfig,
    ReconstructionError,
    build_reconstruction_patch,
    validate_reconstruction_patch,
)


def _canonical_json_bytes(document: dict[str, Any]) -> bytes:
    """產生固定 JSON bytes，供四區 index 與 sidecar checksum 共用。"""

    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    """建立 CLI parser，將重建設定明確暴露在 command line provenance。"""

    parser = argparse.ArgumentParser(description="建立 OCM 稀疏時間缺口重建 patch")
    parser.add_argument("--source-root", type=Path, required=True, help="OCM schema 3 根目錄")
    parser.add_argument("--output-root", type=Path, required=True, help="immutable patch 根目錄")
    parser.add_argument("--flow-id", action="append", help="指定 flow domain；未指定則掃描 source-root")
    parser.add_argument("--context-hours", type=int, default=96)
    parser.add_argument("--min-state-history", type=int, default=8)
    parser.add_argument("--feature-block-size", type=int, default=4096)
    parser.add_argument("--eof-rank", type=int, default=8)
    parser.add_argument("--randomized-oversampling", type=int, default=4)
    parser.add_argument("--randomized-power-iterations", type=int, default=1)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-4)
    parser.add_argument("--diffusivity-max-m2ps", type=float, default=None)
    parser.add_argument("--source-fingerprint", default="unspecified")
    parser.add_argument("--expected-start-ns", type=int, default=None)
    parser.add_argument("--expected-end-ns", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="沿用已通過完整 checksum 驗證的 domain；不接受或覆寫 partial domain",
    )
    return parser


def _write_index(output_root: Path, domain_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """以 atomic bytes 寫入整個 patch root 的四區／多區 index。"""

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "reconstruction-index.json"
    if path.exists():
        raise ReconstructionError(f"重建 index 已存在，拒絕覆寫：{path}")
    document = {
        "schema_version": "ocm_reconstruction_index_v1",
        "domains": [
            {
                "flow_id": item["flow_id"],
                "manifest_path": f"{item['flow_id']}/reconstruction-manifest.json",
                "manifest_sha256": item["manifest_sha256"],
            }
            for item in sorted(domain_manifests, key=lambda item: str(item["flow_id"]))
        ],
    }
    data = _canonical_json_bytes(document)
    partial = output_root / f".reconstruction-index.partial-{uuid.uuid4().hex}"
    try:
        partial.write_bytes(data)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, path)
        (output_root / "reconstruction-index.json.sha256").write_text(
            hashlib.sha256(data).hexdigest() + "\n", encoding="ascii"
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return document


def _print_progress(payload: dict[str, Any]) -> None:
    """逐行輸出 canonical JSON 進度，讓背景工作可安全監督且不解析人類文字。"""

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    """依序處理 flow domains，任何一區失敗即停止且不發布總 index。"""

    args = _parser().parse_args(argv)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    flow_ids = args.flow_id or sorted(
        path.name for path in source_root.iterdir() if path.is_dir() and (path / "months").is_dir()
    )
    if not flow_ids:
        raise ReconstructionError(f"source-root 沒有 flow domain：{source_root}")
    config = ReconstructionConfig(
        context_hours=args.context_hours,
        min_state_history=args.min_state_history,
        feature_block_size=args.feature_block_size,
        eof_rank=args.eof_rank,
        randomized_oversampling=args.randomized_oversampling,
        randomized_power_iterations=args.randomized_power_iterations,
        ridge_lambda=args.ridge_lambda,
        diffusivity_max_m2ps=args.diffusivity_max_m2ps,
        source_fingerprint=args.source_fingerprint,
    )
    manifests: list[dict[str, Any]] = []
    for flow_id in flow_ids:
        _print_progress({"event": "flow_start", "flow_id": flow_id})
        domain_manifest_path = output_root / flow_id / "reconstruction-manifest.json"
        if (output_root / flow_id).exists():
            if not args.resume:
                raise ReconstructionError(
                    f"domain 已存在；確認是完整舊成果後才可用 --resume：{output_root / flow_id}"
                )
            # resume 只信任完整 manifest／sidecar／array checksum；中途殘留的月份沒有
            # domain manifest，會在此 fail closed，絕不跳過或混入新的 root index。
            validated = validate_reconstruction_patch(output_root / flow_id)
            manifests.append(
                {
                    "flow_id": flow_id,
                    "manifest_sha256": hashlib.sha256(domain_manifest_path.read_bytes()).hexdigest(),
                    "manifest": validated,
                }
            )
            _print_progress({"event": "flow_reused", "flow_id": flow_id})
            continue
        source = NpyDomainSource(source_root / flow_id)
        build_reconstruction_patch(
            source,
            output_root,
            flow_id=flow_id,
            config=config,
            expected_start_ns=args.expected_start_ns,
            expected_end_ns=args.expected_end_ns,
            progress_callback=_print_progress,
        )
        manifests.append(
            {
                "flow_id": flow_id,
                "manifest_sha256": hashlib.sha256(domain_manifest_path.read_bytes()).hexdigest(),
                "manifest": validate_reconstruction_patch(output_root / flow_id),
            }
        )
        _print_progress({"event": "flow_complete", "flow_id": flow_id})
    _write_index(output_root, manifests)
    _print_progress({"event": "root_index_published", "flow_count": len(manifests)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
