#!/usr/bin/env python3
"""單站工程回溯時窗的薄型命令列入口。

本腳本只把命令列參數轉交給
``lagrangian_backtracking.engineering_window`` 的核心 API，不另行建立輸入、改寫
正式 selector 或實作粒子運算。``prepare`` 會從已驗收的 source config／derived
input 建立帶有來源指紋與 H30 時間支援證據的 engineering-only artifact；``run``
則以該 artifact 建立或恢復 pilot workspace，交由既有 RunController 執行指定
shard。兩個子命令都輸出單一 JSON 摘要，方便外部監測器保存 stdout、exit code 與
執行時間。

這個入口的輸入路徑必須由 operator 明示；它不會把結果 fallback 到程式目錄、作業系統
暫存區或未驗收的 raw NetCDF。工程 artifact、pilot workspace 與 checkpoint 都不能
被解讀為正式五站成果；正式模式也不會接受 engineering-only artifact。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lagrangian_backtracking.engineering_window import (
    ENGINEERING_DEFAULT_EXPERIMENT_CASE_ID,
    EngineeringWindowError,
    prepare_engineering_window,
    run_engineering_window,
)


def _add_forcing_root_options(parser: argparse.ArgumentParser) -> None:
    """加入 OCM native、NWW3 analysis 與程式 provenance 的共用路徑參數。

    核心 API 會優先使用這些明示根目錄，若省略則依 source config 的環境變數契約
    解析。CLI 不替 operator 猜測掛載點，避免本機 mirror 或錯誤月份資料被誤綁到
    SERVER 工程 run。
    """

    parser.add_argument(
        "--ocm-native-root",
        type=Path,
        help="已驗收 OCM native schema 3 的資料根目錄；省略時沿用 config 環境變數",
    )
    parser.add_argument(
        "--nww-analysis-root",
        type=Path,
        help="已驗收 NWW3 analysis schema 1 的資料根目錄；省略時沿用 config 環境變數",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="實際執行 checkout，用於保存 code provenance；省略時使用套件所在專案根",
    )


def _build_parser() -> argparse.ArgumentParser:
    """建立不含執行副作用的 argparse parser。

    `prepare` 與 `run` 共用 forcing／code 路徑旗標，但各自只暴露核心 API 真正接受的
    欄位。尤其 `run` 的 `--shard-id` 可重複指定 immutable run plan 中的 ID 或非負
    index；不在 CLI 內產生或猜測 shard 名稱。
    """

    parser = argparse.ArgumentParser(
        description=(
            "建立或執行單站 engineering-only 回溯時窗；不產生正式五站研究成果。"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="驗證 source 並建立單站 engineering artifact，不啟動粒子追蹤",
        description=(
            "從已驗收 source config／derived input 建立單站 engineering-only artifact。"
        ),
    )
    prepare_parser.add_argument("--source-config", required=True, type=Path)
    prepare_parser.add_argument("--source-input-directory", required=True, type=Path)
    prepare_parser.add_argument("--study-site-id", required=True)
    prepare_parser.add_argument(
        "--arrival-utc",
        required=True,
        help="exact-hour UTC Z 字串，例如 2024-08-01T00:00:00Z",
    )
    prepare_parser.add_argument(
        "--backtrack-days",
        required=True,
        type=int,
        help="正整數回溯日數；不寫死為 7、30 或 60",
    )
    prepare_parser.add_argument("--material-id", required=True)
    prepare_parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="新的 artifact 目錄；若已存在則拒絕覆寫",
    )
    prepare_parser.add_argument(
        "--vertical-id",
        choices=("upper_water_column", "mid_upper_water_column", "mid_lower_water_column", "near_bed"),
        help="只從既有 source receptor 過濾的垂向層位；本輪龜山島使用 near_bed",
    )
    prepare_parser.add_argument(
        "--shard-scenario-count",
        type=int,
        default=5,
        help="每片情境數；預設 5，需與本輪單片工程設計一致",
    )
    _add_forcing_root_options(prepare_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="建立或恢復 pilot workspace 並執行工程 shard",
        description=(
            "驗證 engineering artifact，建立或恢復 pilot workspace，交由既有 controller 執行 shard。"
        ),
    )
    run_parser.add_argument("--artifact", required=True, type=Path)
    run_parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="run workspace 的 parent 目錄；workspace 會是 parent/run-id",
    )
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--checkpoint-root", type=Path)
    run_parser.add_argument(
        "--experiment-case-id",
        default=ENGINEERING_DEFAULT_EXPERIMENT_CASE_ID,
        help="實驗案例識別碼；預設為 finite_depth_stokes",
    )
    run_parser.add_argument(
        "--random-stream-id",
        help=(
            "可選共同亂數流識別碼；明示後會寫入 paired run plan／seed table，"
            "讓不同 experiment case 在相同 scenario 與 member 使用相同 seed"
        ),
    )
    run_parser.add_argument(
        "--sweep-budget",
        type=int,
        help="正整數 sweep 預算；到達預算由 controller 決定 checkpoint／pause 行為",
    )
    run_parser.add_argument(
        "--shard-id",
        dest="shard_ids",
        action="append",
        help=(
            "指定 run_plan.json 已存在的 shard ID，可重複指定；也接受非負 index。"
            "省略時執行 plan 宣告的全部 shard"
        ),
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="只恢復既有且 binding 未變更的 workspace，不重新建立 run plan",
    )
    _add_forcing_root_options(run_parser)
    return parser


def _dump_json(payload: dict[str, Any]) -> None:
    """以穩定 UTF-8 JSON 輸出 machine-readable 摘要。

    stdout 只放一份完整 JSON，讓 monitor 或 shell 可以直接保存；中文錯誤訊息仍保留
    原文，並以 `valid=false` 搭配非零 exit code 表示核心拒絕，而不是把拒絕誤標為
    空成功輸出。
    """

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _run_prepare(args: argparse.Namespace) -> int:
    """執行 prepare 子命令並輸出 artifact 摘要。"""

    artifact = prepare_engineering_window(
        source_config=args.source_config,
        source_input_directory=args.source_input_directory,
        study_site_id=args.study_site_id,
        arrival_utc=args.arrival_utc,
        backtrack_days=args.backtrack_days,
        material_id=args.material_id,
        destination=args.destination,
        ocm_native_root=args.ocm_native_root,
        nww_analysis_root=args.nww_analysis_root,
        project_root=args.project_root,
        vertical_id=args.vertical_id,
        shard_scenario_count=args.shard_scenario_count,
    )
    _dump_json({"valid": True, "phase": "prepare", **artifact.to_dict()})
    return 0


def _run_run(args: argparse.Namespace) -> int:
    """執行 run 子命令並輸出每一片的 controller 摘要。"""

    summaries = run_engineering_window(
        artifact=args.artifact,
        destination=args.destination,
        run_id=args.run_id,
        checkpoint_root=args.checkpoint_root,
        ocm_native_root=args.ocm_native_root,
        nww_analysis_root=args.nww_analysis_root,
        project_root=args.project_root,
        experiment_case_id=args.experiment_case_id,
        random_stream_id=args.random_stream_id,
        sweep_budget=args.sweep_budget,
        shard_ids=args.shard_ids,
        resume=args.resume,
    )
    _dump_json(
        {
            "valid": True,
            "phase": "run",
            "run_id": args.run_id,
            "resume": args.resume,
            "shards": [asdict(summary) for summary in summaries],
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令列、呼叫核心 adapter，並將可預期的契約拒絕轉成 JSON 錯誤。

    這裡只捕捉輸入、檔案與工程契約常見的可預期例外；不攔截 `KeyboardInterrupt`，
    也不傳送終止訊號給 controller。checkpoint、resume 與未完成狀態均由核心
    RunController 保存，CLI 不代替它修改 progress。
    """

    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return _run_prepare(args)
        if args.command == "run":
            return _run_run(args)
    except (EngineeringWindowError, FileExistsError, OSError, ValueError) as error:
        _dump_json({"valid": False, "error_type": type(error).__name__, "error": str(error)})
        return 2
    raise AssertionError(f"未處理的工程子命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
