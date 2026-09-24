"""所有可重現流程的命令列入口。

CLI 只接受明示路徑或環境變數，不把本機／SERVER 絕對路徑寫入原始碼或 run plan。
輸出預設為 JSON，方便 runbook、排程器與後續 manifest 驗證使用；錯誤訊息寫到 stderr
並以非零狀態退出，避免 shell 批次把未通過 gate 的結果當成功。run-create 只
建立 pilot 或 formal workspace 並立即做只讀驗證；run-shard 先驗證 workspace，再依
immutable plan 選擇 runtime mode 與 forcing root；run-worker 在同一個 controller 內依
命令列順序連續執行已驗證的多個 shard；run-reconcile 僅採認 checkpoint／progress
狀態，不建立物理 request，也不啟動 forcing。aggregate-spec-create、aggregate-build、
aggregate-validate、report-spec-create 與 report-validate 分別負責建立公尺制格網／秒制
age 規格、串流建立並原子發布 aggregate release、輸出固定 JSON-safe aggregate 驗證報告、
建立不讀取 source run 的報告 renderer 規格，以及唯讀驗證 caller 明示的既有 report-v1
release。report-spec-create 只產生報告規格，不建立報告成果；report-validate 只驗證既有
release，不猜測路徑、不建立或修改任何產品。aggregate／report 流程只保存條件式來源足跡
或相對來源權重的工程統計；本機 synthetic 測試不是 OCM／NWW 科學成果，也不能取代 SERVER
正式資料驗收。
pilot-config-create／pilot-config-validate 只負責把完整 calibration candidate 綁定到
generated pilot execution YAML；它不建立軌跡、不把設定升為 approved，也不把任何 SERVER
或 synthetic 輸入宣稱為科學成果。
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import shlex
import stat
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from shapely.geometry import box

from . import manifests
from .aggregate_pipeline import _build_site_metric_centers, build_aggregate_release_payload
from .aggregate_release import (
    validate_aggregate_release,
    write_aggregate_release,
)
from .aggregate_spec import (
    load_aggregate_spec,
    write_aggregate_spec_from_boundaries,
)
from .boundaries import BoundaryGeometry
from .config import load_config
from .diffusion import DiffusionCoefficients
from .engine import EngineSettings, run_particle
from .geometry import DomainProjection
from .horizon_suite import build_horizon_suite, resume_horizon_suite, validate_horizon_suite
from .input_derivation import (
    build_input_derivatives,
    create_release_config,
    validate_input_derivatives,
    validate_release_config,
)
from .models import ParticleState, VelocitySample
from .native_runner import backtrack_horizon as native_backtrack_horizon
from .native_runner import bed_no_deposition_case as native_bed_no_deposition_case
from .native_runner import figures as native_figures
from .native_runner import horizon_derivation as native_horizon_derivation
from .native_runner import hour_major_provider as native_hour_major_provider
from .native_runner import orchestrator as native_orchestrator
from .native_runner import physics_case as native_physics_case
from .native_runner import postprocess as native_postprocess
from .native_runner import production_wiring as native_production_wiring
from .native_runner.backward_driver import load_baytrace_modules
from .native_runner.forcing_bridge import inventory_forcing_gaps
from .native_runner.particle_input import load_particle_identity_table_npz
from .outputs import validate_trajectory_shard, write_trajectory_shard
from .pilot_calibration import (
    build_pilot_calibration,
    read_pilot_calibration,
    validate_pilot_calibration,
)
from .pilot_config import (
    create_pilot_execution_config,
    validate_pilot_execution_config,
)
from .pilot_matrix_validation import (
    canonical_pilot_matrix_json,
    validate_pilot_matrix,
)
from .preflight import run_preflight
from .provenance import collect_code_provenance
from .report_release import validate_report_release
from .report_spec import (
    load_report_spec,
    validate_report_spec_against_aggregate_spec,
    write_report_spec,
)
from .run_control import RunController, load_run_plan
from .run_locking import RunLockBusyError, acquire_run_lock
from .run_validation import benchmark_report, validate_run
from .runtime import (
    EXPERIMENT_CASE_SPECS,
    initialize_formal_run,
    initialize_pilot_run,
    load_validated_run_static_inputs,
    open_run_controller,
)
from .scenarios import BASELINE_BEHAVIORS, records_as_dicts
from .source_pathway_release import (
    build_source_pathway_release,
    validate_source_pathway_release,
)


def _config_check_parser() -> argparse.ArgumentParser:
    """建立設定驗證子命令 parser，供整合 CLI 與獨立 entry point 共用。"""

    parser = argparse.ArgumentParser(description="驗證 Lagrangian 逆向溯源 YAML 與科學計數契約")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--formal-release", action="store_true", help="額外啟用正式發布衍生閘門")
    return parser


def _preflight_parser() -> argparse.ArgumentParser:
    """建立上游月份 preflight 子命令 parser。"""

    parser = argparse.ArgumentParser(description="唯讀檢查 OCM schema 3 與 NWW3 schema 1 月份產品")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ocm-native-root", type=Path)
    parser.add_argument("--nww-analysis-root", type=Path)
    parser.add_argument("--months", nargs="*", help="限制為指定 YYYYMM；省略時檢查設定年份全部月份")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--formal-release", action="store_true")
    return parser


def _inputs_build_parser() -> argparse.ArgumentParser:
    """建立 SERVER v3 衍生輸入 manifest 的 parser。

    OCM native、OCM surface 與 NWW3 analysis root 都必須由 caller 明示或交由設定中的
    環境變數提供；此 parser 不替 SERVER 猜測任何資料目錄。即使未指定
    ``--formal-release``，CLI 仍固定以 strict 模式 fail-closed，禁止以 synthetic
    constant-field fallback 代替缺失或未驗收的輸入。``--formal-release`` 只控制
    approved status、完整時段與 gap-safe horizon 等正式發布閘門；pilot 建置仍可保留
    ``generated`` 狀態供開發資料稽核，但不得把它誤當正式成果。
    """

    parser = argparse.ArgumentParser(description="建立 OCM/NWW3 SERVER v3 immutable input manifests")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--ocm-native-root", type=Path)
    parser.add_argument("--ocm-surface-root", type=Path)
    parser.add_argument("--nww-analysis-root", type=Path)
    parser.add_argument(
        "--pilot-arrival-utc",
        action="append",
        metavar="SITE=UTC",
        help=(
            "只供非正式 pilot 的 deterministic arrival replacement；目前唯一允許值為 "
            "hsinchu=2024-01-02T01:00:00Z，可重複傳入但同一 site 不得重複"
        ),
    )
    parser.add_argument(
        "--formal-release",
        "--formal",
        dest="formal",
        action="store_true",
        help=(
            "啟用 approved status、完整 17,544 小時與 gap-safe horizon 等正式發布閘門；"
            "CLI 始終禁止 synthetic fallback"
        ),
    )
    return parser


def _inputs_validate_parser() -> argparse.ArgumentParser:
    """建立衍生輸入目錄唯讀驗證 parser，支援 positional 與具名目錄寫法。"""

    parser = argparse.ArgumentParser(description="驗證 SERVER v3 input manifest、hash 與 pair coverage")
    parser.add_argument("directory", nargs="?", type=Path, help="input artifact 目錄")
    parser.add_argument("--input-directory", "--directory", dest="input_directory", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ocm-native-root", type=Path)
    parser.add_argument("--ocm-surface-root", type=Path)
    parser.add_argument("--nww-analysis-root", type=Path)
    parser.add_argument("--formal-release", "--formal", dest="formal", action="store_true")
    return parser


def _release_config_create_parser() -> argparse.ArgumentParser:
    """建立 release config 產生 parser；輸出永遠是新檔案，不覆寫 template。"""

    parser = argparse.ArgumentParser(description="由 immutable input manifests 產生 release config")
    parser.add_argument("--config-template", "--config", dest="config_template", required=True, type=Path)
    parser.add_argument("--input-directory", "--manifests", dest="input_directory", required=True, type=Path)
    parser.add_argument("--output", "--destination", dest="output", required=True, type=Path)
    parser.add_argument(
        "--max-backtrack-days",
        type=float,
        help="只覆寫輸出 release config 的 requested 回溯日數；不改 template 或 input artifact",
    )
    parser.add_argument(
        "--maximum-step-count",
        type=int,
        help="只覆寫輸出 release config 的最大步數；不足以涵蓋 requested horizon 時拒絕",
    )
    parser.set_defaults(formal=True)
    parser.add_argument("--formal-release", "--formal", dest="formal", action="store_true")
    parser.add_argument(
        "--pilot",
        dest="formal",
        action="store_false",
        help="只產生 generated config，不嘗試正式 approved gate",
    )
    return parser


def _release_config_validate_parser() -> argparse.ArgumentParser:
    """建立 release config 與其 artifact hash binding 的唯讀驗證 parser。"""

    parser = argparse.ArgumentParser(description="驗證 release config 的 input path/hash binding")
    parser.add_argument("path", type=Path)
    parser.add_argument("--input-directory", "--manifests", dest="input_directory", type=Path)
    parser.add_argument("--formal-release", "--formal", dest="formal", action="store_true")
    parser.set_defaults(formal=True)
    parser.add_argument("--pilot", dest="formal", action="store_false")
    return parser


def _horizon_suite_create_parser() -> argparse.ArgumentParser:
    """建立共同回溯母體 suite 的建置 parser。

    ``--backtrack-days`` 接受任意數量的整數日數；handler 會再拒絕重複值並排序，
    因此 CLI 不會把 30、60、90 寫成固定選單。建置會把最大日數送入一次
    ``inputs-build``，其餘日數只建立 release config；三個 accepted forcing root 均需
    由 caller 明示，避免在 SERVER 上猜測資料位置。
    """

    parser = argparse.ArgumentParser(description="建立共用輸入母體與多個回溯長度 release suite")
    parser.add_argument("--config-template", required=True, type=Path)
    parser.add_argument("--backtrack-days", required=True, nargs="+", type=_positive_cli_int)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--ocm-native-root", required=True, type=Path)
    parser.add_argument("--ocm-surface-root", required=True, type=Path)
    parser.add_argument("--nww-analysis-root", required=True, type=Path)
    parser.set_defaults(formal=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--formal-release",
        "--formal",
        dest="formal",
        action="store_true",
        help="啟用正式 approved gate（預設模式）",
    )
    mode_group.add_argument(
        "--pilot",
        dest="formal",
        action="store_false",
        help="保留 generated/pilot gate；仍執行完整結構與 hash 驗證",
    )
    return parser


def _horizon_suite_validate_parser() -> argparse.ArgumentParser:
    """建立共同回溯母體 suite 的唯讀 validator parser。"""

    parser = argparse.ArgumentParser(description="驗證共用輸入母體與多個回溯 release suite")
    parser.add_argument("path", type=Path)
    parser.add_argument("--ocm-native-root", type=Path)
    parser.add_argument("--ocm-surface-root", type=Path)
    parser.add_argument("--nww-analysis-root", type=Path)
    parser.set_defaults(formal=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--formal-release",
        "--formal",
        dest="formal",
        action="store_true",
        help="啟用正式 approved gate（預設模式）",
    )
    mode_group.add_argument(
        "--pilot",
        dest="formal",
        action="store_false",
        help="以 generated/pilot 狀態驗證 suite",
    )
    return parser


def _horizon_suite_resume_parser() -> argparse.ArgumentParser:
    """建立 preserved partial recovery parser。

    recovery 必須由 caller 明示原 partial、全新的 destination、完整 horizon 集合與
    三套 accepted forcing root；核心會先唯讀驗證 common-input，並保證此命令不再次
    執行昂貴的 ``inputs-build``。formal／pilot 只切換既有 validator gate，不改變
    input artifact 或 release 的科學設定。
    """

    parser = argparse.ArgumentParser(
        description="從 preserved .partial-* 重建 horizon suite，不重跑 input-build"
    )
    parser.add_argument("--partial", required=True, type=Path)
    parser.add_argument("--backtrack-days", required=True, nargs="+", type=_positive_cli_int)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--ocm-native-root", required=True, type=Path)
    parser.add_argument("--ocm-surface-root", required=True, type=Path)
    parser.add_argument("--nww-analysis-root", required=True, type=Path)
    parser.set_defaults(formal=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--formal-release",
        "--formal",
        dest="formal",
        action="store_true",
        help="啟用正式 approved gate（預設模式）",
    )
    mode_group.add_argument(
        "--pilot",
        dest="formal",
        action="store_false",
        help="以 generated/pilot 狀態 recovery；仍執行完整 closure/hash 驗證",
    )
    return parser


def _pilot_calibrate_parser() -> argparse.ArgumentParser:
    """建立 OCM-only pilot 擴散／步長校準 evidence 建置 parser。

    本命令只接受已驗收 input artifact、明示 OCM native root 與專案根目錄；不從環境變數
    猜測 SERVER 路徑，也不讀取 NWW3。輸出是 immutable calibration evidence 目錄，並非
    正式 runtime 參數或科學成果。
    """

    parser = argparse.ArgumentParser(description="建立 OCM-only pilot 擴散與步長校準 evidence")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--ocm-native-root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--pair-limit-per-site", type=int)
    return parser


def _pilot_calibrate_validate_parser() -> argparse.ArgumentParser:
    """建立 pilot calibration immutable artifact 唯讀 validator parser。"""

    parser = argparse.ArgumentParser(description="驗證 OCM-only pilot calibration evidence")
    parser.add_argument("path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-directory", type=Path)
    return parser


def _pilot_config_create_parser() -> argparse.ArgumentParser:
    """建立 calibration-bound pilot execution config 的 CLI parser。

    source release YAML、accepted input directory、1.1.0 calibration evidence 與所有
    執行 scalar 都必須由 caller 明示；CLI 不替 operator 選擇時間步長、回溯期、M 或
    chunk 大小。``active_chunk_size`` 必須明寫整數或小寫 ``none``，以區分「沿用完整
    active set」與漏傳參數；候選 Kh/Kz/floor/cap 不在 CLI 重複輸入，而由 calibration
    report 綁定，避免命令列與 evidence 產生兩套物理值。
    """

    parser = argparse.ArgumentParser(
        description="由 calibration evidence 建立 generated pilot execution config"
    )
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument(
        "--calibration",
        "--calibration-directory",
        dest="calibration",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", "--destination", dest="output", required=True, type=Path)
    parser.add_argument("--dt-min-seconds", required=True, type=_positive_cli_float)
    parser.add_argument("--dt-max-seconds", required=True, type=_positive_cli_float)
    parser.add_argument("--output-interval-seconds", required=True, type=_positive_cli_float)
    parser.add_argument("--max-backtrack-days", required=True, type=_positive_cli_float)
    parser.add_argument("--maximum-step-count", required=True, type=_positive_cli_int)
    parser.add_argument("--members-per-scenario", required=True, type=_positive_cli_int)
    parser.add_argument("--master-seed", required=True, type=_nonnegative_cli_int)
    parser.add_argument("--shard-scenario-count", required=True, type=_positive_cli_int)
    parser.add_argument("--checkpoint-interval-sweeps", required=True, type=_positive_cli_int)
    parser.add_argument("--active-chunk-size", required=True, type=_optional_positive_cli_int)
    parser.add_argument(
        "--max-resident-forcing-months",
        required=True,
        type=_positive_cli_int,
    )
    return parser


def _pilot_config_validate_parser() -> argparse.ArgumentParser:
    """建立 pilot execution config 唯讀 validator parser。

    validator 必須同時收到 target config、accepted input directory 與 calibration
    evidence directory；不從 target 的 release binding 推測外部位置，也不把 target
    當作 calibration 的 source config。成功／失敗 JSON 均由公開 validator 產生，CLI
    只將 ``valid`` 映射到 shell 的 0／2。
    """

    parser = argparse.ArgumentParser(description="驗證 generated pilot execution config")
    parser.add_argument("path", metavar="CONFIG", type=Path)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument(
        "--calibration",
        "--calibration-directory",
        dest="calibration",
        required=True,
        type=Path,
    )
    return parser


def _pilot_matrix_validate_parser() -> argparse.ArgumentParser:
    """建立跨區 pilot 試跑共同設定驗證 parser。

    每個 positional root 都必須是已發布 run workspace，且同時含有 immutable
    ``run_plan.json`` 與 ``normalized_config.json``。validator 只做 JSON／設定 contract
    比較，不讀 trajectory、forcing 或 checkpoint 大檔；至少兩個 root 的要求由公開
    validator 回報，讓 CLI 能穩定輸出 JSON 並使用狀態 2 表示拒絕。
    """

    parser = argparse.ArgumentParser(
        description="驗證多個區域 pilot run 是否使用相同試跑設定與部署版本"
    )
    parser.add_argument("run_roots", nargs="+", type=Path, metavar="RUN_ROOT")
    return parser


def _synthetic_smoke_parser() -> argparse.ArgumentParser:
    """建立不讀 SERVER 的端到端 constant-flow smoke parser。"""

    parser = argparse.ArgumentParser(description="執行 constant-flow backward engine 並發布可驗證 shard")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _validate_shard_parser() -> argparse.ArgumentParser:
    """建立 shard checksum/CSR/Parquet 驗證 parser。"""

    parser = argparse.ArgumentParser(description="驗證不可變 trajectory shard")
    parser.add_argument("path", type=Path)
    return parser


def _behavior_manifest_parser() -> argparse.ArgumentParser:
    """建立十種已裁決非上浮材質／形狀代理 manifest 輸出 parser。"""

    parser = argparse.ArgumentParser(
        description="輸出 design_baseline_v2 十種非上浮海廢材質／形狀代理 manifest"
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _code_provenance_parser() -> argparse.ArgumentParser:
    """建立程式部署指紋 parser；輸出不含 project root 絕對路徑。"""

    parser = argparse.ArgumentParser(description="收集 Lagrangian 程式與依賴部署指紋")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--declared-git-commit")
    parser.add_argument("--formal", action="store_true", help="啟用正式部署 commit/dirty gate")
    return parser


def _validate_run_parser() -> argparse.ArgumentParser:
    """建立 run workspace 嚴格驗證 parser，支援 runtime external checkpoint root。"""

    parser = argparse.ArgumentParser(description="驗證 run plan、seed、checkpoint 與已發布 shard")
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="run 執行時使用的 external checkpoint root；省略時使用 workspace/checkpoints",
    )
    return parser


def _benchmark_report_parser() -> argparse.ArgumentParser:
    """建立唯讀工程 benchmark 摘要 parser，支援驗證 external checkpoint tree。"""

    parser = argparse.ArgumentParser(description="彙總已驗證 run 的工程資源量測")
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="run 執行時使用的 external checkpoint root；不會寫入輸出 JSON",
    )
    return parser


def _run_create_parser() -> argparse.ArgumentParser:
    """建立 pilot/formal workspace 建立器的 parser。

    ``run-kind`` 固定限制為 ``pilot`` 或 ``formal``，讓 SERVER orchestration 使用同一份
    CLI 介面；``--pilot-scenarios-per-stratum`` 僅供 pilot 工程 sanity／benchmark，formal
    明確禁止。formal 的 strict forcing、產品 topology 與 gap-safe gate 由 runtime
    initializer 執行，失敗時不會靜默降級成 pilot。
    精確選擇另以 --pilot-study-site-id、--pilot-arrival-id、--pilot-material-id 明示，
    三者缺一、重複或與分層 N 混用都拒絕；不依軌跡結果挑選案例。
    """

    parser = argparse.ArgumentParser(description="建立可驗證的 pilot 或 formal run workspace")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-inventory", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-kind", required=True, choices=("pilot", "formal"))
    parser.add_argument(
        "--pilot-scenarios-per-stratum",
        type=_positive_cli_int,
        default=None,
        help="pilot 工程 sanity/benchmark 每個 study_site×receptor.vertical strata 的情境數；formal 禁止",
    )
    # 暫存每次出現值，讓 handler 拒絕重複選項，不採 argparse 預設的最後一值覆蓋。
    for flag, meaning in (
        ("--pilot-study-site-id", "單一研究站點 study_site_id"),
        ("--pilot-arrival-id", "單一到達時間 arrival_time_id 識別碼，非 UTC 或列索引"),
        ("--pilot-material-id", "單一材質 material_id"),
    ):
        parser.add_argument(flag, action="append", help=f"pilot 精確選擇：{meaning}；三項必須一起指定")
    parser.add_argument(
        "--experiment-case",
        required=True,
        dest="experiment_case_id",
        # CLI choices 直接由 runtime 唯一 registry 產生，避免 parser 與 initializer 各自
        # 維護一份可能分歧的 experiment case 清單。
        choices=tuple(EXPERIMENT_CASE_SPECS),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--declared-git-commit")
    return parser


def _run_shard_parser() -> argparse.ArgumentParser:
    """建立單一 pilot/formal shard 執行器的 parser，包含 external root 與可恢復選項。"""

    parser = argparse.ArgumentParser(description="執行 pilot 或 formal run 的單一 shard")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--ocm-native-root", type=Path)
    parser.add_argument("--ocm-reconstruction-root", type=Path)
    parser.add_argument("--nww-analysis-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-priority",
        action="store_true",
        help="啟用連續 forcing fallback；僅將時間缺口轉為可記錄 fallback，不放寬陣列／網格安全檢查",
    )
    parser.add_argument("--sweep-budget", type=_positive_cli_int)
    return parser


def _run_worker_parser() -> argparse.ArgumentParser:
    """建立同一 run 內連續執行多個 shard 的 parser。

    這個入口沿用 ``run-shard`` 的 workspace、config、forcing root、resume、checkpoint
    與 sweep 選項；唯一差異是 ``--shard-id`` 可重複指定，並由 handler 先完整驗證所有
    ID。這裡不提供自動挑選或跨 run 的選項，讓連續執行的 runtime manager 明確只屬於
    caller 指定的單一 immutable workspace。
    """

    parser = argparse.ArgumentParser(description="在同一個 run 內連續執行指定的多個 shard")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    # 使用 append 保留 caller 的命令列順序；handler 會拒絕重複與不存在的 ID。
    parser.add_argument("--shard-id", action="append", required=True)
    parser.add_argument("--ocm-native-root", type=Path)
    parser.add_argument("--ocm-reconstruction-root", type=Path)
    parser.add_argument("--nww-analysis-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-priority",
        action="store_true",
        help="單片 exception/FAILED 隔離並持續執行；由同一 controller 連續服務後續 shard",
    )
    parser.add_argument("--sweep-budget", type=_positive_cli_int)
    return parser


def _run_formal_parallel_parser() -> argparse.ArgumentParser:
    """建立正式完整母體的 dynamic/static worker 平行入口 parser。

    workspace、版本、儲存 gate、log 與 Numba cache 路徑都由操作者明示；worker 數也
    必須明示，不從主機 CPU 數量或研究設定推算。正式 CLI 預設使用 parent queue
    動態派片；``--execution-mode static`` 才保留原固定 contiguous group executor。
    run plan 決定分片與 checkpoint cadence，本命令不接受科學參數或每片步數上限；
    ``--warmup-only`` 只編譯 JIT，不跑粒子。
    """

    parser = argparse.ArgumentParser(
        description="依 immutable run plan 以 persistent worker 執行正式完整母體"
    )
    parser.add_argument("workspace", type=Path, help="已建立且通過驗證的 formal run workspace")
    parser.add_argument("--config", required=True, type=Path, help="與 run plan 綁定的正式設定")
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="乾淨且符合 run plan provenance 的程式根目錄",
    )
    parser.add_argument(
        "--worker-count",
        required=True,
        type=_positive_cli_int,
        help="persistent worker 數；不得大於 shard 數",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("dynamic", "static"),
        default="dynamic",
        help="dynamic 為 parent queue 預設；static 保留原固定分組 executor",
    )
    parser.add_argument(
        "--execution-priority",
        action="store_true",
        help="啟用 persistent parent queue、連續時間 fallback 與單片失敗隔離",
    )
    parser.add_argument(
        "--defer-full-validation",
        action="store_true",
        help=(
            "明示先執行粒子、完成後再以 validate-run 完整驗收；"
            "只允許 --execution-priority 的 dynamic 模式"
        ),
    )
    parser.add_argument(
        "--scratch-root",
        required=True,
        type=Path,
        help="已通過 SERVER 儲存 gate 的 NFS scratch 根目錄",
    )
    parser.add_argument(
        "--log-root",
        required=True,
        type=Path,
        help="scratch root 下既有的平行執行日誌根目錄",
    )
    parser.add_argument(
        "--storage-gate-evidence",
        required=True,
        type=Path,
        help="本次 validate_server_storage.py 產生的 PASS JSON",
    )
    parser.add_argument("--checkpoint-root", type=Path, help="本次 run 使用的既有 external checkpoint root")
    parser.add_argument(
        "--ocm-native-root",
        type=Path,
        help="OCM native 根目錄；省略時使用 config 指定的環境變數",
    )
    parser.add_argument(
        "--ocm-reconstruction-root",
        type=Path,
        help="已核准 OCM reconstruction patch 根目錄；省略時使用 config 指定的環境變數",
    )
    parser.add_argument(
        "--nww-analysis-root",
        type=Path,
        help="NWW3 analysis 根目錄；省略時使用 config 指定的環境變數",
    )
    parser.add_argument(
        "--numba-cache-dir",
        type=Path,
        help="Numba backend 必填；scratch 下的安全環境路徑，現行 cache=False kernels 不寫入磁碟快取",
    )
    parser.add_argument(
        "--cpu-affinity",
        default="auto",
        help="Linux CPU 綁定：auto、none 或明示逗號分隔 CPU ID；平台不支援時記錄安全退回",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="只在恢復既有 RUNNING／PAUSED／FAILED 等 run 時明示",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="單次獨立 JIT 編譯檢查，不啟動粒子；正式 Numba worker 仍會各自在程序內預熱",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.05,
        help="parent 檢查 child 狀態的輪詢間隔；不得為負數",
    )
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=float,
        default=30.0,
        help="失敗／中斷時 bounded child shutdown 的寬限時間；不得為負數",
    )
    parser.add_argument(
        "--worker-response-timeout-seconds",
        type=float,
        default=None,
        help="可選的單片 IPC response timeout；省略時不以任意時間殺合法長片",
    )
    return parser


def _run_reconcile_parser() -> argparse.ArgumentParser:
    """建立只做 checkpoint／progress reconcile 的 parser。"""

    parser = argparse.ArgumentParser(description="採認 run checkpoint 與 progress 狀態")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    return parser


def _aggregate_spec_create_parser() -> argparse.ArgumentParser:
    """建立 aggregate spec 建立器的 parser。

    所有研究數值都必須由 caller 明示；CLI 不從範例設定、run plan 或環境變數猜測
    公尺制格網、邊界分箱、核密度估計帶寬、秒制 age 軸或 bootstrap 參數。真正的
    ``AggregateSpec`` constructor 仍會驗證排序、信賴水準與 age 第一個邊界為 0 秒。
    """

    parser = argparse.ArgumentParser(description="由已驗證 run geometry 建立 AggregateSpec 公尺／秒規格")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--grid-cell-size-m", required=True, type=_positive_cli_float)
    parser.add_argument("--boundary-bin-size-m", required=True, type=_positive_cli_float)
    parser.add_argument(
        "--kde-bandwidths-m",
        required=True,
        nargs=3,
        type=_positive_cli_float,
        metavar=("BANDWIDTH_1_M", "BANDWIDTH_2_M", "BANDWIDTH_3_M"),
    )
    parser.add_argument(
        "--age-bin-edges-seconds",
        required=True,
        nargs="+",
        action=_AtLeastTwoValuesAction,
        type=_finite_cli_float,
        metavar="AGE_EDGE_SECONDS",
    )
    parser.add_argument("--bootstrap-replicates", required=True, type=_positive_cli_int)
    parser.add_argument(
        "--bootstrap-confidence-level",
        required=True,
        type=_confidence_cli_float,
    )
    parser.add_argument("--bootstrap-seed", required=True, type=_nonnegative_cli_int)
    parser.add_argument("--checkpoint-root", type=Path)
    return parser


def _aggregate_build_parser() -> argparse.ArgumentParser:
    """建立 aggregate payload／release writer 的整合 parser。"""

    parser = argparse.ArgumentParser(description="串流建立並原子發布 aggregate release")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    return parser


def _aggregate_validate_parser() -> argparse.ArgumentParser:
    """建立 aggregate release 唯讀驗證 parser。"""

    parser = argparse.ArgumentParser(description="驗證 final aggregate release")
    parser.add_argument("release", type=Path)
    return parser


def _report_validate_parser() -> argparse.ArgumentParser:
    """建立 report-v1 release 唯讀驗證 parser。

    ``release`` 必須由 caller 明示為唯一位置參數；CLI 不接受隱含的工作目錄、
    自動搜尋結果或推測出的 sibling 路徑，才能讓 validator 的輸入與 runbook、
    checksum 證據保持一一對應。此 parser 只負責將文字轉成 ``Path``，不讀取、
    建立或修改 release 內容；實際的 no-follow I/O 與固定 JSON-safe 錯誤收斂由
    ``validate_report_release`` 負責。
    """

    parser = argparse.ArgumentParser(description="驗證 final report-v1 release")
    parser.add_argument("release", type=Path)
    return parser


def _source_pathway_build_parser() -> argparse.ArgumentParser:
    """建立向下沉降 source-pathway-v1 圖與 sidecar 的 parser。

    本命令只接受已驗證 aggregate release 與同一 AggregateSpec hash 的 ReportSpec；
    ``--mplconfigdir`` 是明示的既有 Matplotlib cache 目錄，style／繁中字型 gate
    會在真正 import pyplot 前執行。成果包是獨立於完整 report-v1 F/T registry 的
    條件式來源足跡產品，不讀取 trajectory shard，也不納入零沉降材質。
    """

    parser = argparse.ArgumentParser(description="建立向下沉降粒子 source-pathway-v1 圖與 sidecar")
    parser.add_argument("--aggregate-release", required=True, type=Path)
    parser.add_argument("--report-spec", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--mplconfigdir",
        required=True,
        type=Path,
        help="既有、可寫、非 symbolic link 的絕對 Matplotlib cache 目錄",
    )
    return parser


def _source_pathway_validate_parser() -> argparse.ArgumentParser:
    """建立 source-pathway-v1 唯讀 validator parser。"""

    parser = argparse.ArgumentParser(description="驗證 source-pathway-v1 成果包")
    parser.add_argument("release", type=Path)
    return parser


def _native_plan_parser() -> argparse.ArgumentParser:
    """建立 WP-7 native 工作單元規劃＋缺口安全檢查 parser（見 CLI 參考）。

    只讀取既有粒子身分表（WP-4 ``build_particle_identity_table`` 的 npz 輸出）與
    ``ocm_native`` 月份時間軸，切分工作單元並檢查每個到達時刻的 30 天回溯窗是否
    落在未登錄缺口內；不載入 BayTrace 原始碼、不建構網格、不執行任何 kernel 呼叫，
    因此可以在秒等級完成，供核准者在啟動正式計算前先審查。
    """

    parser = argparse.ArgumentParser(description="規劃 native 逆向批次工作單元並檢查到達時刻缺口安全性")
    parser.add_argument("--identity-table", required=True, type=Path, help="WP-4 粒子身分表 npz 路徑")
    parser.add_argument(
        "--region-to-flow-domain",
        type=Path,
        help="JSON 檔：{analysis_region_id: flow_domain_id}；省略時 flow_domain_id 沿用 analysis_region_id",
    )
    parser.add_argument(
        "--ocm-native-root",
        type=Path,
        default=os.environ.get("OCM_NATIVE_ROOT"),
        help="ocm_native schema 3 根目錄；省略時讀取 OCM_NATIVE_ROOT 環境變數",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="正式設定 YAML；提供時缺口檢查會納入 OCM_RECONSTRUCTION_ROOT 內已核准的重建 patch",
    )
    parser.add_argument("--max-backtrack-days", type=float, default=30.0)
    parser.add_argument(
        "--backtrack-mode",
        choices=tuple(native_backtrack_horizon.SUPPORTED_BACKTRACK_MODES),
        default=None,
        help=(
            "WP-17：實際採用的回溯模式；省略時讀取 LBT_NATIVE_BACKTRACK_MODE 環境變數，"
            "兩者都省略時預設 full_horizon_from_deposition（從沉底日起完整回溯 H 天，"
            "PI 原話對應的正式模式）。fixed_calendar_window（固定日曆終點往前回溯 H 天）"
            "是替代敏感度分析模式，須另外提供 --arrival-manifest 才能解析沉底年齡"
        ),
    )
    parser.add_argument(
        "--acknowledge-config-mode-mismatch",
        action="store_true",
        help=(
            "設定檔 bed_residence_time.backtrack_mode 與 --backtrack-mode 實際採用值不同時，"
            "預設 fail-closed 拒絕規劃；明確加上這個旗標才會放行（差異會寫進計畫檔）"
        ),
    )
    parser.add_argument(
        "--arrival-manifest",
        type=Path,
        default=None,
        help=(
            "WP-17：fixed_calendar_window 模式需要的 arrival.json（含沉底 metadata）；"
            "與 --config 搭配載入並驗證。full_horizon_from_deposition 模式不需要這個檔案"
        ),
    )
    parser.add_argument(
        "--experiment-case",
        choices=tuple(EXPERIMENT_CASE_SPECS),
        default=None,
        help=(
            "WP-18：實際採用的實驗案例；省略時讀取 LBT_NATIVE_EXPERIMENT_CASE 環境變數，"
            "兩者都省略時預設 finite_depth_stokes（含有限水深 Stokes 漂移，保留 WP-18 之前"
            "「只要波浪資料存在就用」的既有行為）。no_stokes 只有流：完全不讀取波浪資料、"
            "不計算 Stokes 漂移。兩種案例的擴散都固定採 Smagorinsky（工作書式 (10)，"
            "horcon=0.2），與 EXPERIMENT_CASE_SPECS 的常數擴散（constant_kh_m2ps）舊選項"
            "無關，見 native_runner.physics_case 模組文件"
        ),
    )
    parser.add_argument(
        "--require-wave-data",
        action="store_true",
        help=(
            "只在 --experiment-case finite_depth_stokes（或省略、採用其預設值）時有意義："
            "NWW3 波浪資料缺漏時 fail-closed，不再靜默退化成沒有 Stokes。建議正式規劃一律"
            "加上，避免波浪資料缺漏被默默忽略"
        ),
    )
    parser.add_argument(
        "--bed-no-deposition",
        dest="bed_no_deposition",
        action="store_true",
        default=None,
        help=(
            "WP-26／WP-28：研究用敏感度案例；省略時讀取 LBT_NATIVE_BED_NO_DEPOSITION 環境"
            "變數，兩者都省略時預設 False（既有行為：粒子連續停留海床即視為沉積並終止，"
            "見 native_runner.backward_driver.BackwardDriverConfig.bed_no_deposition 與"
            " reports/WP-28.md 的完整正名說明）。設為啟用時，粒子接觸海床達沉積門檻"
            "不視為沉積、不終止，由既有平流＋擴散＋既有垂向夾擠繼續驅動——不是鏡射反射，"
            "也不是再懸浮動力學。啟用與否會寫進計畫檔／每份結果摘要／後處理 manifest，"
            "兩種案例的結果必須各自使用不同的 --output 目錄，絕不可混在同一份統計"
        ),
    )
    parser.add_argument(
        "--skip-gap-check",
        action="store_true",
        help="明確略過缺口安全檢查（例如合成資料試跑）；一律在輸出中標記，不靜默略過",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--declared-git-commit")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--config-sha256", default="not_provided")
    parser.add_argument("--output", required=True, type=Path, help="輸出的工作計畫 JSON 路徑")
    parser.add_argument(
        "--schedule",
        choices=("per-unit", "hour-major", "hour-major-segmented"),
        default="per-unit",
        help=(
            "per-unit（預設）：計畫檔逐工作單元寫出，沿用既有一次一顆 kernel 呼叫的"
            "正式排程路徑。hour-major：計畫檔改寫成每個流場區域一個摺疊後的排程單元"
            "（見 native_runner.orchestrator.plan_native_hour_major_work_units 與"
            " native_runner.hour_major 模組文件），配合支援 hourmajor:: 前綴"
            " unit_id 的 --resource-provider 實作使用。hour-major-segmented"
            "（WP-11）：在 hour-major 的基礎上，把每個流場區域依到達時刻再切成"
            " --segments 段（或依 --target-worker-count 自動決定），讓核心數遠多於"
            "流場區域數的伺服器能吃滿全部核心（見"
            " native_runner.orchestrator.plan_native_hour_major_segments）；"
            "缺口安全檢查一律仍在摺疊之前、以原本的逐工作單元粒度進行，不受"
            "這個選項影響"
        ),
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=None,
        help=(
            "只在 --schedule hour-major-segmented 時使用。單一整數：套用到全部流場"
            "區域的段數；或 JSON 檔路徑（{flow_domain_id: K}）：逐區域指定段數；"
            "省略時改用 --target-worker-count 自動決定（兩者都省略則等同 K=1，"
            "與 hour-major 相同）"
        ),
    )
    parser.add_argument(
        "--target-worker-count",
        type=int,
        default=None,
        help=(
            "只在 --schedule hour-major-segmented 且未提供 --segments 時使用："
            "自動切段的目標平行度（通常填伺服器可用核心數）。同時提供"
            " --node-counts／--layer-counts 時，改用 WP-13 正確成本模型"
            "（native_runner.orchestrator.recommend_hour_major_segment_counts_by_workload，"
            "依「流場小時數×節點數×層數」而非粒子數選段數）；否則退回舊版"
            " recommend_hour_major_segment_counts（純粒子數啟發式）"
        ),
    )
    parser.add_argument(
        "--node-counts",
        type=Path,
        default=None,
        help=(
            "WP-13：JSON 檔 {flow_domain_id: 節點數}；與 --layer-counts 一起提供時"
            "才會啟用正確成本模型（自動選段數＋報表列出的預估牆鐘時間／重複成本"
            "比例／各段成本）。缺少任一區域的節點數會在該區域需要成本模型時"
            "fail-closed 報錯，不會用臆測值頂替"
        ),
    )
    parser.add_argument(
        "--layer-counts",
        type=Path,
        default=None,
        help="WP-13：JSON 檔 {flow_domain_id: 垂向層數}；見 --node-counts",
    )
    parser.add_argument(
        "--seconds-per-node-layer-hour",
        type=float,
        default=None,
        help=(
            "WP-13 成本模型的校準常數（見"
            " native_runner.orchestrator._DEFAULT_SECONDS_PER_NODE_LAYER_HOUR"
            " 文件的量測依據與誠實揭露）；省略時使用該模組內建的預設值，"
            "正式伺服器應該重新量測後覆寫"
        ),
    )
    parser.add_argument(
        "--worker-count-for-estimate",
        type=int,
        default=None,
        help=(
            "WP-13：只用於 native-plan 報表估算「總牆鐘時間」"
            "（native_runner.orchestrator.estimate_total_wall_clock_seconds），"
            "不影響實際切段；省略時退回 --target-worker-count"
        ),
    )
    return parser


def _months_overlapping_window(month_ids: Sequence[str], start_utc_ns: int, end_utc_ns: int) -> list[str]:
    """只保留與 ``[start, end]`` 期間重疊的 ``YYYYMM`` 月份，避免載入（並逐檔驗證雜湊）期間外的重建 patch。"""

    start = np.datetime64(int(start_utc_ns), "ns").astype("datetime64[M]")
    end = np.datetime64(int(end_utc_ns), "ns").astype("datetime64[M]")
    kept = []
    for month_id in month_ids:
        month = np.datetime64(f"{month_id[:4]}-{month_id[4:6]}", "M")
        if start <= month <= end:
            kept.append(month_id)
    return kept


def _discover_native_month_dirs(ocm_native_root: Path, flow_domain_id: str) -> dict[str, str]:
    """列出 ``<root>/<flow_domain_id>/months/<YYYYMM>`` 既有月份目錄，供缺口盤點使用。

    只列目錄名稱與路徑，不讀取任何月份內的陣列檔；``inventory_forcing_gaps`` 本身
    只讀取每個月份的 ``time_utc_ns.npy``（見其 docstring），維持整體檢查在秒等級。
    """

    months_dir = Path(ocm_native_root) / flow_domain_id / "months"
    if not months_dir.is_dir():
        raise ValueError(f"找不到 flow_domain_id={flow_domain_id} 的月份目錄：{months_dir}")
    return {entry.name: str(entry) for entry in sorted(months_dir.iterdir()) if entry.is_dir()}


def _native_run_parser() -> argparse.ArgumentParser:
    """建立 WP-7 native 工作單元執行／續跑 parser（見 CLI 參考）。

    真正呼叫 BayTrace 核心的 ``build_worker_resources``／``run_unit`` 由
    ``--resource-provider`` 指定的模組層級 factory 函式提供（見
    ``native_runner.orchestrator`` 模組文件「執行引擎」一節與「已知限制」）；本命令
    本身不假設任何特定的流場區域網格、interest region 或波浪橋接實作方式。
    """

    parser = argparse.ArgumentParser(description="執行（或續跑）native-plan 產生的工作單元")
    parser.add_argument("plan", type=Path, help="native-plan 產生的工作計畫 JSON")
    parser.add_argument("--output", required=True, type=Path, help="完成證據、結果摘要與進度快照目錄")
    parser.add_argument(
        "--resource-provider",
        required=True,
        help=(
            "'模組路徑:函式名稱'；該函式呼叫時不帶參數，須回傳 "
            "(build_worker_resources, run_unit) 兩個模組層級可 pickle 的 callable"
        ),
    )
    parser.add_argument("--worker-count", type=int)
    parser.add_argument(
        "--auto-worker-count", action="store_true", help="依可用記憶體與 CPU 數自動估算 worker 數"
    )
    parser.add_argument("--per-worker-memory-bytes", type=int, help="--auto-worker-count 時必須提供")
    parser.add_argument("--reserved-memory-bytes", type=int, default=0)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--numba-cache-dir", type=Path, default=os.environ.get("NUMBA_CACHE_DIR"))
    parser.add_argument(
        "--allow-unsafe-arrivals",
        action="store_true",
        help="明確覆寫 native-plan 回報的缺口不安全狀態；預設 fail-closed 拒絕啟動",
    )
    parser.add_argument("--baytrace-root", type=Path, default=os.environ.get("LBT_BAYTRACE_ROOT"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--declared-git-commit")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--config-sha256", default="not_provided")
    parser.add_argument("--mp-context", default="spawn", choices=("spawn", "forkserver", "fork"))
    parser.add_argument(
        "--wait-for-staged",
        type=Path,
        default=None,
        help=(
            "WP-13：依資料就緒動態加入。提供一個目錄，本命令啟動後監看"
            " <目錄>/<flow_domain_id>.done 完成標記；已就緒的流場區域立即開跑，"
            "尚未就緒的區域的工作單元先保留、worker 不因此閒置或退出，一偵測到"
            "標記就立刻把該區域排入佇列。全部區域都出現在計畫檔裡的"
            " flow_domain_id 集合就緒、且佇列清空後才正常結束。省略時（預設）"
            "一律視為全部區域已就緒，行為與 WP-13 之前完全相同"
        ),
    )
    return parser


def _native_status_parser() -> argparse.ArgumentParser:
    """建立 WP-7 native 執行進度查詢 parser。"""

    parser = argparse.ArgumentParser(description="讀取 native-run 執行進度快照與完成證據統計")
    parser.add_argument("--output", required=True, type=Path, help="native-run 使用的同一個輸出目錄")
    return parser


def run_native_plan_command(argv: Sequence[str] | None = None) -> int:
    """規劃 native 工作單元、檢查缺口安全性並寫出計畫 JSON。"""

    args = _native_plan_parser().parse_args(argv)
    table = load_particle_identity_table_npz(args.identity_table)

    region_to_flow_domain: dict[str, str] = {}
    if args.region_to_flow_domain is not None:
        region_to_flow_domain = json.loads(args.region_to_flow_domain.read_text(encoding="utf-8"))

    units = native_orchestrator.plan_native_work_units(table, region_to_flow_domain=region_to_flow_domain)

    # WP-17：回溯模式明確化——在缺口檢查與排程摺疊之前，先把每個工作單元的有效回溯
    # 時間、pre-window 判定解析好並烤進 NativeWorkUnit（見 backtrack_horizon.py 與
    # orchestrator.resolve_native_work_unit_backtrack_horizons）。
    effective_backtrack_mode = native_backtrack_horizon.resolve_backtrack_mode(
        cli_backtrack_mode=args.backtrack_mode
    )
    config_declared_backtrack_mode: str | None = None
    if args.config is not None:
        plan_config = load_config(str(args.config), formal_release=False)
        bed_residence_time = getattr(plan_config.scenarios, "bed_residence_time", None)
        if bed_residence_time is not None:
            config_declared_backtrack_mode = bed_residence_time.backtrack_mode
    native_backtrack_horizon.require_mode_acknowledged_if_mismatched(
        effective_backtrack_mode=effective_backtrack_mode,
        config_declared_backtrack_mode=config_declared_backtrack_mode,
        acknowledge_config_mode_mismatch=args.acknowledge_config_mode_mismatch,
    )

    # WP-18：實驗案例明確化（見 physics_case.py 模組文件）——決定本次規劃是否含
    # Stokes 漂移；兩種案例都固定用 Smagorinsky 工作書式 (10) 擴散，與舊登錄表
    # 的常數擴散選項無關。
    resolved_physics_case = native_physics_case.resolve_physics_case(
        cli_experiment_case_id=args.experiment_case
    )
    physics_case_declaration = {
        "experiment_case_id": resolved_physics_case.experiment_case_id,
        "include_stokes": resolved_physics_case.include_stokes,
        "diffusion_method": resolved_physics_case.diffusion_method,
        "require_wave_data": bool(args.require_wave_data),
    }

    # WP-26／WP-28：bed_no_deposition（海床不沉積）敏感度案例明確化，介面與上面
    # 的實驗案例同款（見 bed_no_deposition_case.py 模組文件）。
    resolved_bed_no_deposition = native_bed_no_deposition_case.resolve_bed_no_deposition(
        cli_bed_no_deposition=args.bed_no_deposition
    )
    bed_no_deposition_declaration = {"bed_no_deposition": resolved_bed_no_deposition}

    arrivals_by_time_utc_ns: dict[int, Any] | None = None
    maximum_age_days: int | None = None
    sampling_seed: int | None = None
    if effective_backtrack_mode == native_backtrack_horizon.BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW:
        if args.arrival_manifest is None or args.config is None:
            raise ValueError(
                "--backtrack-mode fixed_calendar_window 必須同時提供 --config 與 --arrival-manifest，"
                "才能解析每個到達時刻的沉底年齡（resolve_bed_residence_timing 需要完整 metadata）"
            )
        fixed_window_config = load_config(str(args.config), formal_release=False)
        bed_residence_time = getattr(fixed_window_config.scenarios, "bed_residence_time", None)
        if bed_residence_time is None:
            raise ValueError(
                "--config 沒有啟用 scenarios.bed_residence_time；fixed_calendar_window 模式"
                "需要沉底年齡分層抽樣設定才有意義"
            )
        arrival_records = manifests.load_arrival_manifest(
            args.arrival_manifest, fixed_window_config, formal=args.formal
        )
        arrivals_by_time_utc_ns = {int(record.time_utc_ns): record for record in arrival_records}
        maximum_age_days = bed_residence_time.maximum_age_days
        # WP-26：改讀 effective_sampling_seed，讓 age_sampling_seed 覆寫（若有明示）
        # 也套用到 fixed_calendar_window 的年齡解析；未明示時逐位元相同。
        sampling_seed = bed_residence_time.effective_sampling_seed

    units = native_orchestrator.resolve_native_work_unit_backtrack_horizons(
        units,
        backtrack_mode=effective_backtrack_mode,
        requested_horizon_days=args.max_backtrack_days,
        arrivals_by_time_utc_ns=arrivals_by_time_utc_ns,
        maximum_age_days=maximum_age_days,
        sampling_seed=sampling_seed,
    )
    pre_window_unit_count = sum(1 for unit in units if unit.pre_window_deposition)

    summary = native_orchestrator.summarize_native_work_plan(units)

    gap_checks: tuple[native_orchestrator.ArrivalGapCheck, ...] = ()
    patch_months_used: dict[str, list[str]] = {}
    if not args.skip_gap_check:
        if args.ocm_native_root is None:
            raise ValueError(
                "缺口檢查需要 --ocm-native-root 或 OCM_NATIVE_ROOT；如需略過請明示 --skip-gap-check"
            )
        max_backtrack_seconds = args.max_backtrack_days * 86400.0
        gap_reports: dict[str, object] = {}
        for region in sorted({unit.analysis_region_id for unit in units}):
            flow_domain_id = region_to_flow_domain.get(region, region)
            region_arrivals = [
                unit.arrival_time_utc_ns for unit in units if unit.analysis_region_id == region
            ]
            expected_start = int(min(region_arrivals) - round(max_backtrack_seconds * 1.0e9))
            expected_end = int(max(region_arrivals))
            month_dirs = _discover_native_month_dirs(args.ocm_native_root, flow_domain_id)
            # 已核准的 OCM 缺口重建 patch：提供 --config 且設定 OCM_RECONSTRUCTION_ROOT
            # 時一併納入盤點，與正式執行時讀取端套用的 patch 一致（否則已重建的小時會被
            # 誤判為未登錄缺口）。
            patches = {}
            if args.config is not None:
                patches = native_production_wiring.load_domain_reconstruction_patch_time_axes(
                    os.environ.get("OCM_RECONSTRUCTION_ROOT"),
                    flow_domain_id,
                    months=_months_overlapping_window(sorted(month_dirs), expected_start, expected_end),
                )
                patch_months_used[region] = sorted(patches)
            gap_reports[region] = inventory_forcing_gaps(
                month_dirs,
                expected_start_utc_ns=expected_start,
                expected_end_utc_ns=expected_end,
                patches=patches,
                flow_id=flow_domain_id,
            )
        gap_checks = native_orchestrator.check_arrival_gap_safety(
            table, gap_reports_by_region=gap_reports, max_backtrack_seconds=max_backtrack_seconds
        )

    manifest = native_orchestrator.build_native_run_manifest(
        project_root=args.project_root,
        declared_git_commit=args.declared_git_commit,
        formal=args.formal,
        config_sha256=args.config_sha256,
        seed_policy="particle_identity_table_master_seed_v1",
        master_seed=int(table.master_seed),
        extra={
            "gap_check_skipped": bool(args.skip_gap_check),
            "schedule": args.schedule,
            "gap_check_reconstruction_patch_months": patch_months_used,
        },
    )

    # 缺口安全檢查（上面）永遠以原本的逐工作單元粒度進行——``check_arrival_gap_
    # safety`` 需要每個工作單元自己的 arrival_time_utc_ns，摺疊後的小時主軸排程
    # 單元用 sentinel 值填那個欄位，不能拿來做這個檢查。只有真正寫進計畫檔的
    # ``units``（供 native-run 排程執行）才依 --schedule 決定要不要摺疊／切段。
    node_counts: dict[str, int] | None = None
    layer_counts: dict[str, int] | None = None
    if args.node_counts is not None:
        node_counts = {
            str(k): int(v) for k, v in json.loads(args.node_counts.read_text(encoding="utf-8")).items()
        }
    if args.layer_counts is not None:
        layer_counts = {
            str(k): int(v) for k, v in json.loads(args.layer_counts.read_text(encoding="utf-8")).items()
        }
    seconds_per_node_layer_hour = (
        args.seconds_per_node_layer_hour
        if args.seconds_per_node_layer_hour is not None
        else native_orchestrator._DEFAULT_SECONDS_PER_NODE_LAYER_HOUR
    )

    hour_major_members: Mapping[str, tuple[native_orchestrator.NativeWorkUnit, ...]] | None = None
    plan_units = units
    segment_reports: tuple[native_orchestrator.HourMajorSegmentPlanReport, ...] = ()
    if args.schedule == "hour-major":
        plan_units, hour_major_members = native_orchestrator.plan_native_hour_major_work_units(units)
    elif args.schedule == "hour-major-segmented":
        max_backtrack_seconds = args.max_backtrack_days * 86400.0
        segment_count: int | Mapping[str, int] | None = None
        if args.segments is not None:
            try:
                segment_count = int(args.segments)
            except ValueError:
                segment_count = json.loads(Path(args.segments).read_text(encoding="utf-8"))
        plan_units, hour_major_members, segment_reports = native_orchestrator.plan_native_hour_major_segments(
            units,
            segment_count=segment_count,
            target_worker_count=args.target_worker_count,
            max_backtrack_seconds=max_backtrack_seconds,
            node_counts=node_counts,
            layer_counts=layer_counts,
            seconds_per_node_layer_hour=seconds_per_node_layer_hour,
        )

    # WP-13：只有在呼叫端提供了 --node-counts／--layer-counts（成本模型的
    # 主要因子）時，才計算並列出各段成本、預估牆鐘時間與重複成本比例；
    # 缺少網格規模時不臆測，報表明確省略這幾個欄位，不會用誤導性的數字
    # 頂替（見 estimate_hour_major_segment_costs 對這個 fail-closed 策略的
    # 說明）。
    cost_model_report: dict[str, Any] | None = None
    if args.schedule == "hour-major-segmented" and node_counts is not None and layer_counts is not None:
        segment_costs = native_orchestrator.estimate_hour_major_segment_costs(
            segment_reports,
            plan_units,
            node_counts=node_counts,
            layer_counts=layer_counts,
            seconds_per_node_layer_hour=seconds_per_node_layer_hour,
        )
        estimate_worker_count = args.worker_count_for_estimate or args.target_worker_count
        if estimate_worker_count is not None and segment_costs:
            wall_clock_seconds, max_unit_cost_seconds = native_orchestrator.estimate_total_wall_clock_seconds(
                segment_costs, worker_count=estimate_worker_count
            )
            no_split_costs = {
                domain: native_orchestrator.estimate_hour_major_unit_cost_seconds(
                    hours_to_synthesize=int(
                        round(report.arrival_span_days * 24.0 + report.max_backtrack_days * 24.0) + 1
                    ),
                    node_count=node_counts[report.flow_domain_id],
                    layer_count=layer_counts[report.flow_domain_id],
                    particle_count=0,
                    seconds_per_node_layer_hour=seconds_per_node_layer_hour,
                )
                for domain, report in {r.flow_domain_id: r for r in segment_reports}.items()
            }
            total_no_split_cost = sum(no_split_costs.values()) or 1.0
            total_cost = sum(segment_costs.values())
            repeat_cost_fraction = (total_cost - total_no_split_cost) / total_no_split_cost
            cost_model_report = {
                "seconds_per_node_layer_hour": seconds_per_node_layer_hour,
                "worker_count_for_estimate": estimate_worker_count,
                "estimated_wall_clock_seconds": wall_clock_seconds,
                "max_single_segment_cost_seconds": max_unit_cost_seconds,
                "total_cost_seconds": total_cost,
                "repeat_cost_fraction_vs_no_split": repeat_cost_fraction,
                "segment_cost_seconds": segment_costs,
            }

    if cost_model_report is not None:
        # 把成本模型算出的各段成本一併存進 manifest，native-run 讀回計畫檔時
        # 可以直接拿來當 run_native_plan 的 unit_costs（依成本由大到小提交，
        # 不需要 native-run 重新知道節點數／層數才能排序），見 run_native_run。
        manifest["extra"]["unit_costs"] = cost_model_report["segment_cost_seconds"]

    backtrack_mode_declaration = {
        "declared_config_backtrack_mode": config_declared_backtrack_mode,
        "effective_backtrack_mode": effective_backtrack_mode,
        "requested_horizon_days": args.max_backtrack_days,
        "acknowledge_config_mode_mismatch": bool(args.acknowledge_config_mode_mismatch),
        "pre_window_unit_count": pre_window_unit_count,
        "arrival_manifest": str(args.arrival_manifest) if args.arrival_manifest is not None else None,
    }

    native_orchestrator.write_native_plan_file(
        args.output,
        units=plan_units,
        gap_checks=gap_checks,
        summary=summary,
        manifest=manifest,
        hour_major_members=hour_major_members,
        backtrack_mode_declaration=backtrack_mode_declaration,
        physics_case_declaration=physics_case_declaration,
        bed_no_deposition_declaration=bed_no_deposition_declaration,
    )

    unsafe = [check for check in gap_checks if not check.is_gap_safe]
    report = {
        "output": str(args.output),
        "schedule": args.schedule,
        "unit_count": summary.unit_count,
        "scheduled_unit_count": len(plan_units),
        "locality_group_count": summary.locality_group_count,
        "particle_count_total": summary.particle_count_total,
        "region_ids": list(summary.region_ids),
        "gap_check_skipped": bool(args.skip_gap_check),
        "is_gap_safe_ready": args.skip_gap_check or len(unsafe) == 0,
        "unsafe_arrival_count": len(unsafe),
        "unsafe_arrivals": [dataclasses.asdict(check) for check in unsafe],
        "segment_plan": [dataclasses.asdict(item) for item in segment_reports],
        "cost_model": cost_model_report,
        "backtrack_horizon": backtrack_mode_declaration,
        "physics_case": physics_case_declaration,
        "bed_no_deposition": bed_no_deposition_declaration,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["is_gap_safe_ready"] else 2


def run_native_run(argv: Sequence[str] | None = None) -> int:
    """執行（或續跑）``native-plan`` 產生的工作單元。"""

    args = _native_run_parser().parse_args(argv)
    units, payload = native_orchestrator.read_native_plan_file(args.plan)

    # WP-17：計畫檔記錄了 native-plan 當初實際採用的回溯模式（見
    # cli.run_native_plan_command 的 backtrack_mode_declaration）；這裡在真正執行
    # 任何工作單元之前，先核對「這次 native-run 進程實際會採用的模式」
    # （LBT_NATIVE_BACKTRACK_MODE，與每個 worker 行程 hour_major_provider._resolve_
    # params 讀到的值同一套解析邏輯）是否與計畫檔一致。native-run 本身不會重新
    # 規劃工作單元——每個 NativeWorkUnit 的 resolved_max_backtrack_seconds／
    # pre_window_deposition 已經在 native-plan 階段依當時的模式烤進計畫檔——所以
    # 這裡的檢查只防止「用一份用模式 A 規劃好的計畫檔，卻在寫進每個結果摘要的
    # provenance 時貼上模式 B 的標籤」這種誤導；不做這個檢查，計算結果本身仍然
    # 正確（因為已經烤進計畫檔），但誰都看不出來 provenance 標籤是錯的。
    # 同一個「承認」環境變數同時涵蓋 WP-17 回溯模式與 WP-18 實驗案例兩種不一致，
    # 不另外新增第二個旗標（見 hour_major_provider.py 模組文件「環境變數契約」）。
    raw_ack = os.environ.get("LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH")
    acknowledge_mismatch = bool(raw_ack) and str(raw_ack).strip() not in ("0", "false", "False", "")

    plan_backtrack_horizon = payload.get("backtrack_horizon")
    if plan_backtrack_horizon is not None:
        plan_effective_mode = plan_backtrack_horizon.get("effective_backtrack_mode")
        run_effective_mode = native_backtrack_horizon.resolve_backtrack_mode(cli_backtrack_mode=None)
        native_backtrack_horizon.require_mode_acknowledged_if_mismatched(
            effective_backtrack_mode=run_effective_mode,
            config_declared_backtrack_mode=plan_effective_mode,
            acknowledge_config_mode_mismatch=acknowledge_mismatch,
        )

    # WP-18：計畫檔記錄了 native-plan 當初實際採用的實驗案例（是否含 Stokes、擴散
    # 方法聲明）；這裡同樣在執行任何工作單元之前核對本次 native-run 進程實際會
    # 採用的案例（LBT_NATIVE_EXPERIMENT_CASE，與每個 worker 行程
    # hour_major_provider._resolve_params 讀到的值同一套解析邏輯）是否與計畫檔
    # 一致，理由與上面 backtrack_horizon 的核對相同：native-run 不重新規劃，
    # 只是避免結果摘要貼上與計畫檔不符的實驗案例標籤。
    plan_physics_case = payload.get("physics_case")
    if plan_physics_case is not None:
        plan_experiment_case_id = plan_physics_case.get("experiment_case_id")
        run_experiment_case_id = native_physics_case.resolve_experiment_case_id(
            cli_experiment_case_id=None
        )
        if run_experiment_case_id != plan_experiment_case_id and not acknowledge_mismatch:
            raise native_physics_case.PhysicsCaseError(
                f"計畫檔實驗案例 experiment_case_id={plan_experiment_case_id!r} 與本次 "
                f"native-run 實際採用的 LBT_NATIVE_EXPERIMENT_CASE={run_experiment_case_id!r} "
                "不一致，拒絕默默執行。請明確設定 LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH=1 "
                "承認此差異，或改用與計畫檔一致的實驗案例"
            )

    # WP-26／WP-28：計畫檔記錄了 native-plan 當初是否啟用 bed_no_deposition（海床
    # 不沉積）敏感度案例；同樣在執行任何工作單元之前核對本次 native-run 進程實際
    # 會採用的值（LBT_NATIVE_BED_NO_DEPOSITION，與每個 worker 行程
    # hour_major_provider._resolve_params 讀到的值同一套解析邏輯）是否與計畫檔一致，
    # 理由與上面 backtrack_horizon／physics_case 的核對相同：避免結果摘要與輸出
    # 目錄貼上與計畫檔不符的標籤，讓兩種案例的結果被誤判成同一批可直接合併統計。
    plan_bed_no_deposition = payload.get("bed_no_deposition")
    if plan_bed_no_deposition is not None:
        plan_bed_no_deposition_value = bool(plan_bed_no_deposition.get("bed_no_deposition", False))
        run_bed_no_deposition_value = native_bed_no_deposition_case.resolve_bed_no_deposition(
            cli_bed_no_deposition=None
        )
        if run_bed_no_deposition_value != plan_bed_no_deposition_value and not acknowledge_mismatch:
            raise native_bed_no_deposition_case.BedNoDepositionCaseError(
                f"計畫檔 bed_no_deposition={plan_bed_no_deposition_value!r} 與本次 native-run 實際"
                f" 採用的 LBT_NATIVE_BED_NO_DEPOSITION={run_bed_no_deposition_value!r} 不一致，拒絕"
                "默默執行。請明確設定 LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH=1 承認此差異，"
                "或改用與計畫檔一致的設定"
            )

    if not payload.get("is_gap_safe_ready", False) and not args.allow_unsafe_arrivals:
        report = {
            "error": "plan_not_gap_safe_ready",
            "message": "native-plan 回報有到達時刻不安全；請先解決缺口或明確加上 --allow-unsafe-arrivals",
            "unsafe_gap_checks": [
                check for check in payload.get("gap_checks", []) if not check.get("is_gap_safe")
            ],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    if args.worker_count is not None:
        worker_count = args.worker_count
    elif args.auto_worker_count:
        if args.per_worker_memory_bytes is None:
            raise ValueError("--auto-worker-count 必須搭配 --per-worker-memory-bytes")
        worker_count = native_orchestrator.estimate_worker_count(
            available_memory_bytes=native_orchestrator.read_available_memory_bytes(),
            per_worker_memory_bytes=args.per_worker_memory_bytes,
            cpu_count=os.cpu_count(),
            reserved_memory_bytes=args.reserved_memory_bytes,
            max_workers=args.max_workers,
        )
    else:
        raise ValueError("必須提供 --worker-count 或 --auto-worker-count")

    module_path, _, func_name = args.resource_provider.partition(":")
    if not module_path or not func_name:
        raise ValueError("--resource-provider 必須是 '模組路徑:函式名稱' 格式")
    provider_module = importlib.import_module(module_path)
    provider_factory = getattr(provider_module, func_name)
    build_worker_resources, run_unit = provider_factory()

    baytrace_dependency_sha256: Mapping[str, str] | None = None
    if args.baytrace_root is not None:
        handles = load_baytrace_modules(args.baytrace_root)
        baytrace_dependency_sha256 = handles.dependency_sha256

    manifest = native_orchestrator.build_native_run_manifest(
        project_root=args.project_root,
        declared_git_commit=args.declared_git_commit,
        formal=args.formal,
        config_sha256=args.config_sha256,
        seed_policy="particle_identity_table_master_seed_v1",
        master_seed=int(payload.get("manifest", {}).get("master_seed", 0)),
        baytrace_source_root=str(args.baytrace_root) if args.baytrace_root else None,
        baytrace_dependency_sha256=baytrace_dependency_sha256,
        extra={"plan_path": str(args.plan), "worker_count": worker_count},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    raw_unit_costs = payload.get("manifest", {}).get("extra", {}).get("unit_costs")
    unit_costs: Mapping[str, float] | None = None
    if isinstance(raw_unit_costs, Mapping):
        unit_costs = {str(k): float(v) for k, v in raw_unit_costs.items()}

    staged_kwargs: dict[str, Any] = {}
    if args.wait_for_staged is not None:
        staged_kwargs["staged_ready_dir"] = args.wait_for_staged
        staged_kwargs["staged_flow_domain_ids"] = sorted({u.flow_domain_id for u in units})

    summary = native_orchestrator.run_native_plan(
        units,
        worker_count=worker_count,
        build_worker_resources=build_worker_resources,
        run_unit=run_unit,
        output_dir=args.output,
        numba_cache_dir=str(args.numba_cache_dir) if args.numba_cache_dir else None,
        mp_context_method=args.mp_context,
        unit_costs=unit_costs,
        **staged_kwargs,
    )
    report = dataclasses.asdict(summary)
    report["worker_count"] = worker_count
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.all_succeeded else 2


def run_native_status(argv: Sequence[str] | None = None) -> int:
    """讀取 native-run 的進度快照與完成證據統計。"""

    args = _native_status_parser().parse_args(argv)
    snapshot = native_orchestrator.read_progress_snapshot(args.output)
    if snapshot is None:
        error_payload = {"error": "no_progress_snapshot_found", "output": str(args.output)}
        print(json.dumps(error_payload, ensure_ascii=False))
        return 2

    status_counts: dict[str, int] = {}
    for marker_path in sorted(Path(args.output).glob("*.done.json")):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = str(marker.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    report = dict(snapshot)
    report["unit_status_counts"] = status_counts
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# ══════════════════════════════════════════════════════════════════════════
# WP-11：native-preflight／native-gap-fill／native-postprocess
# ══════════════════════════════════════════════════════════════════════════


def _native_preflight_parser() -> argparse.ArgumentParser:
    """建立 ``lbt native-preflight`` parser：伺服器上第一步、秒到分鐘級資料契約檢查。

    只呼叫既有 ``production_wiring.preflight_native_inputs``（見該函式文件），不
    另外定義任何檢查規則；本命令的價值在於把它包成一條可以直接接在
    ``run_native_backtrack_server.sh`` 第一步的 CLI 子命令，失敗時以非零狀態碼
    讓伺服器腳本 fail-closed 停止，不繼續往下跑昂貴的網格建構或 kernel 呼叫。
    """

    parser = argparse.ArgumentParser(description="正式 native 資料前置檢查（目錄、節點數、缺口、到達窗）")
    parser.add_argument("--config", required=True, type=Path, help="正式設定 YAML 路徑")
    parser.add_argument(
        "--domain",
        action="append",
        dest="domains",
        default=None,
        help="只檢查指定的 analysis_region_id；可重複指定，省略時檢查設定內全部區域",
    )
    parser.add_argument("--max-backtrack-days", type=float, default=30.0)
    parser.add_argument("--formal", action="store_true", default=True)
    parser.add_argument("--pilot", dest="formal", action="store_false", help="以 pilot（非正式）模式檢查")
    parser.add_argument("--output", type=Path, help="可選：把人類可讀報告額外寫成文字檔")
    return parser


def run_native_preflight_command(argv: Sequence[str] | None = None) -> int:
    """執行正式 native 資料前置檢查，回傳碼反映 ``PreflightReport.is_run_ready``。"""

    args = _native_preflight_parser().parse_args(argv)
    report = native_production_wiring.preflight_native_inputs(
        str(args.config),
        env=os.environ,
        domain_ids=args.domains,
        formal=args.formal,
        max_backtrack_seconds=args.max_backtrack_days * 86400.0,
    )
    text = report.describe()
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if report.is_run_ready else 2


def _native_gap_fill_parser() -> argparse.ArgumentParser:
    """建立 ``lbt native-gap-fill`` parser：逐流場區域呼叫已核准的缺口重建管線。

    重建邏輯本身由 ``native_runner.gap_fill``（另一個並行開發的工作包）提供，本
    命令只以**延遲匯入**（``run_native_gap_fill_command`` 函式內才 import）呼叫
    ``build_gap_fill_patches``／``build_wave_gap_fill_patches``／
    ``validate_gap_fill_blocked`` 三個函式，逐一對每個在 ``--ocm-native-domain-root``
    下找到的流場區域執行，輸出寫到 ``--output-root``（即 ``OCM_RECONSTRUCTION_
    ROOT``）。延遲匯入是刻意的：``gap_fill.py`` 可能在本命令交付時尚未完成，
    延遲匯入讓本檔案本身可以先通過語法檢查與其餘 native-* 命令的測試，只有真正
    執行 ``native-gap-fill`` 時才需要該模組存在。
    """

    parser = argparse.ArgumentParser(description="逐流場區域執行已核准的 OCM／波浪缺口重建")
    parser.add_argument(
        "--ocm-native-domain-root",
        required=True,
        type=Path,
        help="ocm_native schema 3 根目錄（其下每個子目錄是一個 flow_domain_id）",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="重建 patch 輸出根目錄，即後續 OCM_RECONSTRUCTION_ROOT 指向的路徑",
    )
    parser.add_argument("--start-utc", default="2024-01-01T00:00:00+00:00", help="重建區間起點（ISO 8601）")
    parser.add_argument("--end-utc", default="2025-12-31T23:00:00+00:00", help="重建區間終點（ISO 8601）")
    parser.add_argument(
        "--flow-domain-id",
        action="append",
        dest="flow_domain_ids",
        default=None,
        help="只處理指定的 flow_domain_id；可重複指定，省略時處理 --ocm-native-domain-root 下全部子目錄",
    )
    parser.add_argument(
        "--include-wave",
        action="store_true",
        help="同時呼叫 build_wave_gap_fill_patches 重建波浪缺口（需要 --nww-native-domain-root）",
    )
    parser.add_argument("--nww-native-domain-root", type=Path, help="--include-wave 時的 NWW3 根目錄")
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="略過每個區域重建後的 validate_gap_fill_blocked 阻擋式交叉驗證（不建議在正式流程使用）",
    )
    return parser


def _iso8601_to_utc_ns(text: str) -> int:
    """把 ISO 8601 字串轉成 UTC 奈秒；不接受缺時區資訊的字串（避免時區歧義）。"""

    from datetime import datetime

    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"時間字串 {text!r} 缺少時區資訊，native-gap-fill 只接受明示時區的 ISO 8601")
    return int(parsed.timestamp() * 1_000_000_000)


def run_native_gap_fill_command(argv: Sequence[str] | None = None) -> int:
    """逐流場區域執行 ``native_runner.gap_fill`` 的缺口重建管線。

    ``gap_fill`` 模組由另一個並行開發的工作包負責，本函式在呼叫當下才 import，
    找不到時直接以清楚的錯誤訊息與非零狀態碼回報，不假裝已完成重建。
    """

    args = _native_gap_fill_parser().parse_args(argv)
    try:
        from .native_runner import gap_fill as native_gap_fill
    except ImportError as exc:
        error_payload = {
            "error": "gap_fill_module_not_available",
            "message": (
                "native_runner.gap_fill 尚未提供（由另一個工作包負責），"
                "native-gap-fill 無法執行；請確認該模組已交付後重試"
            ),
            "import_error": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    start_utc_ns = _iso8601_to_utc_ns(args.start_utc)
    end_utc_ns = _iso8601_to_utc_ns(args.end_utc)
    if end_utc_ns <= start_utc_ns:
        raise ValueError("--end-utc 必須晚於 --start-utc")

    domain_root = Path(args.ocm_native_domain_root)
    if args.flow_domain_ids:
        flow_domain_ids = list(args.flow_domain_ids)
    else:
        flow_domain_ids = sorted(entry.name for entry in domain_root.iterdir() if entry.is_dir())

    per_domain_reports: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for flow_domain_id in flow_domain_ids:
        domain_report: dict[str, Any] = {"flow_domain_id": flow_domain_id}
        try:
            ocm_patches = native_gap_fill.build_gap_fill_patches(
                domain_root / flow_domain_id,
                args.output_root,
                start_utc_ns=start_utc_ns,
                end_utc_ns=end_utc_ns,
                flow_domain_id=flow_domain_id,
            )
            domain_report["ocm_patches"] = ocm_patches
            if args.include_wave:
                if args.nww_native_domain_root is None:
                    raise ValueError("--include-wave 需要同時提供 --nww-native-domain-root")
                wave_patches = native_gap_fill.build_wave_gap_fill_patches(
                    Path(args.nww_native_domain_root) / flow_domain_id,
                    args.output_root,
                    start_utc_ns=start_utc_ns,
                    end_utc_ns=end_utc_ns,
                    flow_domain_id=flow_domain_id,
                )
                domain_report["wave_patches"] = wave_patches
            if not args.skip_validate:
                domain_report["validation"] = native_gap_fill.validate_gap_fill_blocked(
                    args.output_root, flow_domain_id=flow_domain_id
                )
        except Exception as exc:  # noqa: BLE001 - 單一區域失敗不應中止其餘區域的重建
            domain_report["error"] = str(exc)
            failures.append(flow_domain_id)
        per_domain_reports[flow_domain_id] = domain_report

    report = {
        "output_root": str(args.output_root),
        "start_utc": args.start_utc,
        "end_utc": args.end_utc,
        "flow_domain_ids": flow_domain_ids,
        "failed_flow_domain_ids": failures,
        "per_domain": per_domain_reports,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0 if not failures else 2


def _native_postprocess_parser() -> argparse.ArgumentParser:
    """建立 ``lbt native-postprocess`` parser：合併全部工作單元成果、輸出正式產品。

    只認得 ``hour_major_provider.save_backward_driver_result_npz`` 寫出的
    ``<unit_id>.driver_result.npz`` 檔案格式（見該模組文件「重量級輸出的存放
    格式」一節）；``--results-dir`` 應該是 ``LBT_NATIVE_RESULTS_DIR`` 指向的
    同一個目錄。

    WP-30：``--project-root``／``--declared-git-commit``／``--formal`` 三者比照
    ``native-plan``／``native-run`` 既有同名參數的語意，收集程式部署指紋寫進
    manifest 的 ``code_provenance``（見 ``provenance.collect_code_provenance``）；
    這三個參數在此之前完全沒有開放，導致 native-postprocess 的 manifest 一律拿
    不到 git commit hash，是正式成果報告可追溯性附錄的缺口。``--project-root``
    省略時維持修正前的既有行為（``code_provenance`` 只留一句略過說明）。
    """

    parser = argparse.ArgumentParser(description="合併 native-run 全部工作單元成果，輸出後處理正式產品")
    parser.add_argument("--identity-table", required=True, type=Path, help="WP-4 粒子身分表 npz 路徑")
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="hour_major_provider 寫出的 <unit_id>.driver_result.npz 所在目錄（LBT_NATIVE_RESULTS_DIR）",
    )
    parser.add_argument("--output", required=True, type=Path, help="後處理成果輸出目錄")
    parser.add_argument(
        "--config",
        type=Path,
        help="可選：正式設定 YAML 路徑，提供時輸出的離開紀錄會額外帶經緯度（由各站所屬" "區域投影反算）",
    )
    parser.add_argument(
        "--with-boundary-segments",
        action="store_true",
        help=(
            "WP-17：另外依開放邊界分段（domain／local／open_boundary manifest）統計每站"
            "（與每站×沉降速度）相對來源權重，寫出 boundary_entry_by_site.*／"
            "boundary_entry_by_site_settling_velocity.*；需要同時提供 --config，且該設定的"
            "邊界幾何 manifest 必須可載入，否則 fail closed（不靜默略過這個統計）"
        ),
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help=(
            "載入邊界幾何 manifest 時是否要求正式門檻；同時也是 --project-root 收集"
            "程式部署指紋時的正式門檻（與 native-plan／native-run 的 --formal 相同"
            "語意——啟用時沒有 .git 部署必須另外提供 --declared-git-commit，且拒絕"
            "dirty working tree，見 provenance.collect_code_provenance）"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "WP-30：可選，正式部署所在的程式根目錄；提供時比照 native-plan／"
            "native-run 既有的同名參數，呼叫 provenance.collect_code_provenance "
            "收集程式部署指紋（Git commit、dirty 狀態、部署樹雜湊與依賴版本），"
            "寫入本次後處理 manifest 的 code_provenance，讓正式成果報告的可追溯性"
            "附錄拿得到 git commit hash。省略時 code_provenance 只留一句略過"
            "說明（僅限測試或探索用途），與加入本參數之前的既有行為完全相同"
        ),
    )
    parser.add_argument(
        "--declared-git-commit",
        default=None,
        help=(
            "只在提供 --project-root 且該目錄沒有 .git（例如 SERVER 發布部署）時"
            "才會被使用：由呼叫端宣告的 40 位小寫 Git commit；--formal 且無 .git "
            "時必須提供，否則 collect_code_provenance 會 fail-fast 拒絕產生指紋"
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help=(
            "WP-18／WP-26／WP-28：可選，native-plan 產生的計畫檔路徑；提供時把其中"
            " physics_case 區塊（實際採用的實驗案例、是否含 Stokes、擴散方法聲明）"
            "原樣寫進本次後處理 manifest 的 config.runtime_physics_case，並把"
            " bed_no_deposition 區塊（本次是否啟用海床不沉積敏感度案例）寫進"
            " config.runtime_bed_no_deposition，供人工稽核與避免不同案例的結果"
            "被混用或合併統計。省略時兩個欄位皆為 null，不代表結果沒有 Stokes，"
            "也不代表沒有啟用 bed_no_deposition"
        ),
    )
    parser.add_argument("--trajectory-sample-size-per-group", type=int, default=24)
    parser.add_argument("--trajectory-sample-seed", type=int, default=0)
    parser.add_argument("--kde-grid-cell-count", type=int, default=64)
    parser.add_argument("--kde-minimum-point-count", type=int, default=30)
    parser.add_argument(
        "--azimuth-sector-count",
        type=int,
        default=native_postprocess.DEFAULT_AZIMUTH_SECTOR_COUNT,
        help=(
            "WP-20：只在 --with-boundary-segments 時生效。以站受體位置中心為原點，"
            "依離開點方位角把每站的開放邊界穿越點再細分成幾個等角扇區（預設 16 個，"
            "每 22.5 度，由正北順時針；等於 16 時扇區名稱套用標準羅盤方位，否則退回"
            "sector_NNN 數字命名），輸出 boundary_entry_by_site_azimuth_sector.*"
        ),
    )
    parser.add_argument(
        "--boundary-arc-segment-count",
        type=int,
        default=native_postprocess.DEFAULT_BOUNDARY_ARC_SEGMENT_COUNT,
        help=(
            "WP-20：只在 --with-boundary-segments 時生效。把每站已判定所屬的開放邊界"
            "折線依弧長等分成幾段（預設 20 段），統計各段的相對來源權重，輸出"
            "boundary_entry_by_site_arc_segment.*"
        ),
    )
    return parser


def _build_native_lonlat_projector(
    config: Any,
) -> native_postprocess.LonLatProjector:
    """由設定檔的 ``study_sites``／``domains`` 建立「站點局部座標 → 經緯度」投影器。

    與 ``production_wiring._site_projection_map`` 使用完全相同的
    ``DomainProjection(*domain_config.center_lonlat)`` 建構方式（該函式是私有
    輔助函式，本處為避免跨模組依賴私有名稱而在 CLI 層獨立重建，但邏輯與其
    保持一致，任一方修改投影中心的取得方式都必須同步更新另一方）。
    """

    domain_by_region = {domain.analysis_region_id: domain for domain in config.domains}
    site_to_region = {site.study_site_id: site.analysis_region_id for site in config.study_sites}
    projections: dict[str, DomainProjection] = {}
    for site_id, region_id in site_to_region.items():
        domain_config = domain_by_region.get(region_id)
        if domain_config is not None:
            projections[site_id] = DomainProjection(*domain_config.center_lonlat)

    def projector(
        study_site_id: np.ndarray, x_m: np.ndarray, y_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lon = np.full(x_m.shape, np.nan, dtype=np.float64)
        lat = np.full(x_m.shape, np.nan, dtype=np.float64)
        for site_id, projection in projections.items():
            mask = study_site_id == site_id
            if not np.any(mask):
                continue
            site_lon, site_lat = projection.unproject(x_m[mask], y_m[mask])
            lon[mask] = site_lon
            lat[mask] = site_lat
        return lon, lat

    return projector


def run_native_postprocess_command(argv: Sequence[str] | None = None) -> int:
    """合併 ``native-run`` 全部工作單元成果，輸出離開紀錄、通量與條件式來源足跡。"""

    args = _native_postprocess_parser().parse_args(argv)
    identity_table = load_particle_identity_table_npz(args.identity_table)

    lonlat_projector: native_postprocess.LonLatProjector | None = None
    config = None
    if args.config is not None:
        config = load_config(str(args.config), formal_release=False)
        lonlat_projector = _build_native_lonlat_projector(config)

    boundary_geometries = None
    if args.with_boundary_segments:
        if config is None:
            raise ValueError("--with-boundary-segments 必須同時提供 --config")
        boundary_geometries = manifests.load_boundary_geometries(
            config, config_path=str(args.config), formal=args.formal
        )

    runtime_physics_case: Mapping[str, Any] | None = None
    runtime_bed_no_deposition: Mapping[str, Any] | None = None
    if args.plan is not None:
        _plan_units, plan_payload = native_orchestrator.read_native_plan_file(args.plan)
        runtime_physics_case = plan_payload.get("physics_case")
        runtime_bed_no_deposition = plan_payload.get("bed_no_deposition")

    postprocess_config = native_postprocess.PostprocessConfig(
        trajectory_sample_size_per_group=args.trajectory_sample_size_per_group,
        trajectory_sample_seed=args.trajectory_sample_seed,
        kde_grid_cell_count=args.kde_grid_cell_count,
        kde_minimum_point_count=args.kde_minimum_point_count,
        azimuth_sector_count=args.azimuth_sector_count,
        boundary_arc_segment_count=args.boundary_arc_segment_count,
        # WP-30：只有明確提供 --project-root 才收集程式部署指紋，省略時保留 None，
        # 讓 PostprocessAccumulator.finalize 走既有的「略過收集」分支（見 postprocess.py）。
        project_root=str(args.project_root) if args.project_root is not None else None,
        declared_git_commit=args.declared_git_commit,
        formal_provenance=args.formal,
    )
    accumulator = native_postprocess.PostprocessAccumulator(
        identity_table,
        output_dir=args.output,
        config=postprocess_config,
        lonlat_projector=lonlat_projector,
        boundary_geometries=boundary_geometries,
        runtime_physics_case=runtime_physics_case,
        runtime_bed_no_deposition=runtime_bed_no_deposition,
    )

    result_paths = sorted(Path(args.results_dir).glob("*.driver_result.npz"))
    if not result_paths:
        raise ValueError(f"{args.results_dir} 底下找不到任何 *.driver_result.npz 成果檔")
    for result_path in result_paths:
        unit_id = result_path.name[: -len(".driver_result.npz")]
        driver_result = native_hour_major_provider.load_backward_driver_result_npz(result_path)
        accumulator.add_batch(unit_id, driver_result)

    manifest = accumulator.finalize()
    report = {
        "output": str(args.output),
        "batch_count": len(manifest.batches),
        "particle_count_total": manifest.particle_count_total,
        "status_counts": {str(k): v for k, v in manifest.status_counts.items()},
        "status_descriptions_zh": {str(k): v for k, v in manifest.status_descriptions_zh.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _native_derive_horizon_parser() -> argparse.ArgumentParser:
    """建立 ``lbt native-derive-horizon`` parser（WP-17 任務 2：免重跑截斷推導）。

    只讀取既有 ``*.driver_result.npz``，不建構任何網格、不呼叫任何 BayTrace 核心、
    不查詢任何流場，純陣列切片，因此不需要 ``LBT_BAYTRACE_ROOT``。
    """

    parser = argparse.ArgumentParser(
        description="由既有 full_horizon_from_deposition 結果截斷推導較短 (模式, H)，不重跑"
    )
    parser.add_argument(
        "--source-results-dir", required=True, type=Path,
        help="來源 *.driver_result.npz 所在目錄（必須是 full_horizon_from_deposition 的結果）",
    )
    parser.add_argument(
        "--source-backtrack-mode",
        required=True,
        choices=tuple(native_backtrack_horizon.SUPPORTED_BACKTRACK_MODES),
        help="來源結果實際採用的回溯模式；目前只接受 full_horizon_from_deposition",
    )
    parser.add_argument(
        "--source-horizon-days", required=True, type=float,
        help="來源結果的 H_src（天）；本命令不會從 .npz 檔重新推算，必須由呼叫端明示",
    )
    parser.add_argument(
        "--target-backtrack-mode",
        required=True,
        choices=tuple(native_backtrack_horizon.SUPPORTED_BACKTRACK_MODES),
    )
    parser.add_argument(
        "--target-horizon-days", required=True, type=float,
        help="目標 H（天），必須 <= --source-horizon-days",
    )
    parser.add_argument(
        "--arrival-manifest", type=Path, default=None,
        help="target-backtrack-mode=fixed_calendar_window 時必須提供的 arrival.json",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="target-backtrack-mode=fixed_calendar_window 時必須提供，用來讀取"
        " bed_residence_time.maximum_age_days／sampling_seed 與驗證 arrival manifest",
    )
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output", required=True, type=Path, help="推導結果輸出目錄（必須不存在）")
    return parser


def run_native_derive_horizon_command(argv: Sequence[str] | None = None) -> int:
    """由既有 ``full_horizon_from_deposition`` 結果截斷推導較短 ``(模式, H)``，見
    ``native_runner.horizon_derivation`` 模組文件的正確性論證。"""

    args = _native_derive_horizon_parser().parse_args(argv)

    arrivals_by_time_utc_ns: dict[int, Any] | None = None
    maximum_age_days: int | None = None
    sampling_seed: int | None = None
    if args.target_backtrack_mode == native_backtrack_horizon.BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW:
        if args.arrival_manifest is None or args.config is None:
            raise ValueError(
                "--target-backtrack-mode fixed_calendar_window 必須同時提供 --config 與"
                " --arrival-manifest，才能解析每個到達時刻的沉底年齡"
            )
        derive_config = load_config(str(args.config), formal_release=False)
        bed_residence_time = getattr(derive_config.scenarios, "bed_residence_time", None)
        if bed_residence_time is None:
            raise ValueError("--config 沒有啟用 scenarios.bed_residence_time")
        arrival_records = manifests.load_arrival_manifest(
            args.arrival_manifest, derive_config, formal=args.formal
        )
        arrivals_by_time_utc_ns = {int(record.time_utc_ns): record for record in arrival_records}
        maximum_age_days = bed_residence_time.maximum_age_days
        # WP-26：同上，改讀 effective_sampling_seed。
        sampling_seed = bed_residence_time.effective_sampling_seed

    manifest = native_horizon_derivation.derive_result_directory(
        source_results_dir=args.source_results_dir,
        output_dir=args.output,
        source_backtrack_mode=args.source_backtrack_mode,
        source_requested_horizon_days=args.source_horizon_days,
        target_backtrack_mode=args.target_backtrack_mode,
        target_requested_horizon_days=args.target_horizon_days,
        arrivals_by_time_utc_ns=arrivals_by_time_utc_ns,
        maximum_age_days=maximum_age_days,
        sampling_seed=sampling_seed,
    )
    report = {
        "output": str(args.output),
        "unit_count": manifest["unit_count"],
        "target_backtrack_mode": manifest["target_backtrack_mode"],
        "target_requested_horizon_days": manifest["target_requested_horizon_days"],
        "pre_window_unit_count": sum(
            1 for u in manifest["units"].values() if u["pre_window_deposition"]
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _native_figures_parser() -> argparse.ArgumentParser:
    """建立 ``lbt native-figures`` parser（WP-21：native 正式成果圖表管線）。

    只讀 ``native-postprocess`` 已寫出的 Parquet／CSV 與呼叫端明示的 CRS84 海岸
    GeoJSON；不讀 forcing、不重新積分、不改寫任何輸入。``--output-dir`` 必須是
    尚不存在的全新目錄，見 ``native_runner.figures`` 模組文件。
    """

    parser = argparse.ArgumentParser(description="由 native 後處理輸出獨立繪製每站八張正式成果圖")
    parser.add_argument(
        "--postprocess-dir", required=True, type=Path, help="native-postprocess 輸出目錄（只讀）"
    )
    parser.add_argument(
        "--coastline", required=True, type=Path, help="CRS84 FeatureCollection 海岸 GeoJSON（只讀）"
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="圖表輸出目錄（必須尚不存在）")
    parser.add_argument(
        "--sites", nargs="+", default=None, help="只輸出指定站點（預設輸出資料中全部站點）"
    )
    parser.add_argument(
        "--font-path", type=Path, default=None,
        help="可選：自備中文字型檔；省略時呼叫 resolve_cjk_font()，找不到合格字型時圖面改用英文標籤",
    )
    parser.add_argument(
        "--max-trajectories-per-site", type=int, default=24,
        help="水平總覽／局部／深度年齡三張圖共用的代表軌跡樣本上限",
    )
    parser.add_argument(
        "--dpi", type=int, default=600,
        help=(
            "輸出 PNG 解析度（WP-25：正式交付預設 600 dpi，對齊 OCM-SVD-Analysis 成圖"
            "規範；每張圖同時固定輸出同名 SVG，不受此參數影響）"
        ),
    )
    parser.add_argument(
        "--figure-text", choices=list(native_figures.FIGURE_TEXT_MODES), default="full",
        help=(
            "圖面文字層級（WP-24b）：full（預設）畫出標題／副標／圖例／座標軸標籤，供內部"
            "檢視；none 只畫座標軸與圖例，不畫標題與副標，供正式報告排版——文字改由輸出目錄"
            "的 captions.md 承載。兩種模式都會輸出 captions.md。"
        ),
    )
    return parser


def run_native_figures_command(argv: Sequence[str] | None = None) -> int:
    """執行一次 native-figures：讀 native 後處理輸出，畫每站八張正式成果圖。"""

    args = _native_figures_parser().parse_args(argv)
    command_argv = list(argv if argv is not None else sys.argv[1:])
    command = shlex.join(["lbt", "native-figures", *command_argv])

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=args.postprocess_dir,
        coastline_path=args.coastline,
        output_dir=args.output_dir,
        sites=tuple(args.sites) if args.sites is not None else None,
        font_path=args.font_path,
        max_trajectories_per_site=args.max_trajectories_per_site,
        dpi=args.dpi,
        figure_text=args.figure_text,
    )
    output_dir = native_figures.build_native_figures(config, command=command)
    report = {
        "output_dir": str(output_dir),
        "figure_stems": list(native_figures.FIGURE_STEMS),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _report_spec_create_parser() -> argparse.ArgumentParser:
    """建立 report spec parser，所有 renderer／抽樣研究值都必須明示。

    KDE 帶寬與垂向深度邊界使用公尺（m）；垂向軸是相對瞬時海面的
    positive-down 深度，最後 edge 外的 observation 由後續 reducer 記錄 overflow，
    CLI 不自行裁切。代表軌跡數的 ``>=8``／8 倍數、seed 的 128 位元上限、深度首點
    0 與嚴格遞增，以及帶寬是否精確存在於 AggregateSpec，全部交由 ReportSpec
    writer 的 immutable contract fail closed；這裡只先拒絕缺參數、非有限值、非正整數
    或少於兩個 edge。此命令不接受 run、forcing 或 OCM／NWW3 array，輸出仍只是
    可重現的報告工程規格，不是 local synthetic 的科學成果。
    """

    parser = argparse.ArgumentParser(description="建立與 AggregateSpec 綁定的 report spec")
    parser.add_argument("--aggregate-spec", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--primary-kde-bandwidth-m",
        required=True,
        type=_positive_cli_float,
    )
    parser.add_argument(
        "--minimum-kde-raw-count",
        required=True,
        type=_positive_cli_int,
    )
    parser.add_argument(
        "--low-sample-min-member-count",
        required=True,
        type=_positive_cli_int,
    )
    parser.add_argument(
        "--vertical-depth-bin-edges-m",
        required=True,
        nargs="+",
        action=_AtLeastTwoValuesAction,
        type=_finite_cli_float,
        metavar="DEPTH_EDGE_M",
    )
    parser.add_argument(
        "--representative-trajectory-count-per-site",
        required=True,
        type=_positive_cli_int,
    )
    parser.add_argument(
        "--representative-selection-seed",
        required=True,
        type=_nonnegative_cli_int,
    )
    return parser


class _AtLeastTwoValuesAction(argparse.Action):
    """在 argparse 階段拒絕少於兩個時間或垂向深度邊界的選項值。

    ``argparse`` 的 ``nargs='+'`` 本身允許單一值，但 aggregate age 軸與 report
    positive-down 深度軸都必須至少包含起點與終點；把這個結構限制放在 parser，可讓
    使用者得到標準狀態 2，而不是進入檔案／寫入流程後才發現輸入不足。每一個值仍由
    呼叫端指定的 finite-float type 先驗證，起點與排序等領域語意則留給各自 schema。
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        """保存至少兩個已解析 edge，否則交由 argparse 統一回報錯誤。"""

        del option_string
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise argparse.ArgumentError(self, "至少需要兩個 edge")
        setattr(namespace, self.dest, tuple(values))


def _positive_cli_int(raw: str) -> int:
    """將 CLI 文字轉成嚴格正整數，拒絕零、負數與布林文字。

    argparse 的 type 會先把命令列值交給此函式；回傳真正的 int 供 controller
    使用，並把格式或範圍錯誤轉成 argparse 的狀態 2。明確拒絕 True、False 等
    布林樣式文字，可避免將控制旗標誤當成 sweep 次數。
    """

    try:
        value = int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必須是正整數") from exc
    if isinstance(value, bool) or value < 1:
        raise argparse.ArgumentTypeError("必須是正整數")
    return value


def _nonnegative_cli_int(raw: str) -> int:
    """將 CLI 文字轉成可作 seed 的非負原生整數，拒絕負數與布林樣式。"""

    try:
        value = int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必須是非負整數") from exc
    if type(value) is not int or value < 0:
        raise argparse.ArgumentTypeError("必須是非負整數")
    return value


def _optional_positive_cli_int(raw: str) -> int | None:
    """解析 active chunk 的明示整數或 ``none``，拒絕空字串與其他隱含 fallback。"""

    if raw == "none":
        return None
    return _positive_cli_int(raw)


def _finite_cli_float(raw: str) -> float:
    """將 CLI 文字轉為有限 Python float，不接受 NaN、正負無限或布林值。"""

    if isinstance(raw, bool):
        raise argparse.ArgumentTypeError("必須是有限浮點數")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("必須是有限浮點數") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("必須是有限浮點數")
    return value


def _positive_cli_float(raw: str) -> float:
    """將 CLI 文字轉為正的有限 Python float，供所有公尺制長度與帶寬使用。"""

    value = _finite_cli_float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("必須是正的有限浮點數")
    return value


def _confidence_cli_float(raw: str) -> float:
    """將 CLI 信賴水準限制在 AggregateSpec 要求的開區間 ``(0, 1)``。"""

    value = _finite_cli_float(raw)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("必須是介於 0 與 1 之間的有限浮點數")
    return value


def _validation_error_codes(result: object) -> list[str]:
    """從 validator 結果擷取不含路徑的錯誤碼，供建立失敗例外使用。

    validator 對外保證錯誤是可序列化字串，但錯誤細節未來可能包含操作環境資訊。
    run-create 的例外只需要讓 orchestration 辨識失敗類型，因此取每個錯誤的
    第一段 code，避免 workspace、checkpoint 或 SERVER 絕對路徑洩漏到訊息中。
    """

    if not isinstance(result, Mapping):
        return ["validator_invalid"]
    raw_errors = result.get("errors")
    if not isinstance(raw_errors, (list, tuple)):
        return ["validator_invalid"]
    codes: list[str] = []
    for raw_error in raw_errors:
        if type(raw_error) is not str or not raw_error.strip():
            continue
        first_segment = raw_error.split(":", 1)[0].strip()
        code = first_segment.split(maxsplit=1)[0] if first_segment else ""
        if (
            code
            and not code.startswith(("/", "\\"))
            and not (len(code) >= 3 and code[1] == ":" and code[2] in {"/", "\\"})
        ):
            codes.append(code)
    return codes or ["validator_invalid"]


def _workspace_display_path(workspace: object) -> str:
    """取得 initializer 回傳 workspace 的顯示位置，不改寫其 plan 內容。"""

    path = getattr(workspace, "path", workspace)
    return str(path)


def _optional_root_from_env(value: Path | None, env_name: str) -> Path | None:
    """解析可選 forcing root；缺少設定時回傳 None 而不猜測 SERVER 路徑。"""

    if value is not None:
        return value
    raw = os.environ.get(env_name)
    return Path(raw) if raw else None


def _shard_lifecycle_counts(progress: Mapping[str, object]) -> dict[str, int]:
    """從 reconcile 回傳的 progress 計算各 shard lifecycle 數量。

    progress 的 shards 是以 shard ID 為 key 的 mapping；此摘要只保留生命週期計數，
    不輸出 checkpoint token 或任何 external root，讓 SERVER orchestration 的結果可安全
    傳遞給監控系統。
    """

    raw_shards = progress.get("shards")
    if not isinstance(raw_shards, Mapping):
        return {}
    counts = Counter(
        row.get("lifecycle")
        for row in raw_shards.values()
        if isinstance(row, Mapping) and type(row.get("lifecycle")) is str
    )
    return dict(counts)


def _root_from_argument_or_env(value: Path | None, env_name: str) -> Path:
    """以 CLI 明示值優先，其次讀環境變數；缺少時拒絕猜測 SERVER 路徑。"""

    if value is not None:
        return value
    raw = os.environ.get(env_name)
    if not raw:
        raise ValueError(f"缺少 --root 參數或環境變數 {env_name}")
    return Path(raw)


def open_worker_controller(
    workspace: str | Path,
    *,
    config_path: str | Path,
    ocm_native_root: str | Path,
    nww_analysis_root: str | Path | None = None,
    ocm_reconstruction_root: str | Path | None = None,
    resume: bool = False,
    checkpoint_root: str | Path | None = None,
    continuous_time_fallback: bool = False,
    _defer_full_validation: bool = False,
    _private_dynamic_bootstrap: object | None = None,
) -> RunController:
    """建立一個可連續處理多片的正式 runtime controller。

    這個薄 helper 是 ``run-worker`` 與 dynamic persistent child 共用的 controller
    setup 邊界。它只轉交既有 ``open_run_controller``；forcing schema、粒子 request、
    run gate、shard lock、checkpoint、seed 與 QC 仍完全由既有 controller 負責。caller
    必須先完成 workspace/config/forcing root 的驗證；本函式不偷偷猜測任何路徑，也不會
    自動執行 shard 或取得 reconcile 的 run-gate exclusive lock。
    """

    if type(continuous_time_fallback) is not bool:
        raise TypeError("continuous_time_fallback 必須是 bool")
    if type(_defer_full_validation) is not bool:
        raise TypeError("_defer_full_validation 必須是 bool")
    kwargs: dict[str, object] = {
        "config_path": config_path,
        "ocm_native_root": ocm_native_root,
        "nww_analysis_root": nww_analysis_root,
        "resume": resume,
        "checkpoint_root": checkpoint_root,
    }
    # Optional reconstruction root 在未提供時必須完全省略，而不是傳入
    # ``None``。這保留既有 controller 與測試替身的關鍵字介面；只有正式
    # 使用重建流場時才把實際路徑傳入，避免把「未設定」誤寫成一個新的
    # provenance 欄位。execution-priority 的 fallback 旗標仍獨立處理如下。
    if ocm_reconstruction_root is not None:
        kwargs["ocm_reconstruction_root"] = ocm_reconstruction_root
    # 只有 execution-priority 明確開啟時才傳新參數；舊 controller／測試替身在預設
    # 路徑仍看到完全相同的 kwargs 拓撲，並保留 fail-closed 的預設 forcing 行為。
    if continuous_time_fallback:
        kwargs["continuous_time_fallback"] = True
    if _defer_full_validation:
        # private capability 只能由 dynamic child 轉交；本 helper 不自行建立 token，避免
        # 一般 caller 只靠 bool 取得 startup-only 驗證路徑。
        from .runtime import _DEFERRED_VALIDATION_PRIVATE_TOKEN

        if _private_dynamic_bootstrap is not _DEFERRED_VALIDATION_PRIVATE_TOKEN:
            raise ValueError(
                "_defer_full_validation 只允許 dynamic child private bootstrap 使用"
            )
        kwargs["defer_full_validation"] = True
        kwargs["_private_dynamic_bootstrap"] = _private_dynamic_bootstrap
    return open_run_controller(workspace, **kwargs)


def execute_worker_shard(
    controller: Any,
    shard_id: str,
    *,
    sweep_budget: int | None = None,
) -> dict[str, object]:
    """用既有 controller 執行恰好一片，回傳 JSON-safe 的工程狀態摘要。

    dynamic child 以此 helper 將一個 JSON-line request 對應到一個
    ``controller.run_shard`` 呼叫；它不重建 controller、不重派其他 shard，也不改變
    run-control 的 shared run gate／exclusive shard lock 順序。這裡保留既有 duck-typed
    controller 契約，讓正式 ``RunController`` 與測試中的行為替身都必須提供同一個
    callable ``run_shard`` 方法。回傳 mapping 只包裝既有 ``RunExecutionSummary``，因此
    lifecycle、checkpoint、seed、QC 與輸出 validator 的語意仍由核心 controller 定義。
    """

    if type(shard_id) is not str or not shard_id:
        raise ValueError("shard_id 必須是非空字串")
    run_shard = getattr(controller, "run_shard", None)
    if not callable(run_shard):
        raise TypeError("controller 必須提供 callable run_shard 方法")
    return asdict(run_shard(shard_id, sweep_budget=sweep_budget))


def reconcile_run_for_resume(
    workspace: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """以既有 reconcile 採認 resume 狀態，但禁止在 reconcile 建立粒子 request。

    explicit resume 必須先透過 run-gate exclusive reconcile 修復或採認 checkpoint／output
    pointer，再由 caller 讀取 progress 決定未完成片。reject factory 是安全邊界：若
    reconcile 意外要求物理 request，立即失敗；因此這個 helper 不會與 worker shared gate
    長時間交錯，也不新增 checkpoint schema。
    """

    def reject_request_factory(unit: object) -> NoReturn:
        """reconcile 僅採認狀態，不得建立 forcing 或粒子 request。"""

        del unit
        raise RuntimeError("run-reconcile 不得呼叫 request_factory；reconcile 不做物理計算")

    controller = RunController(
        workspace,
        request_factory=reject_request_factory,
        checkpoint_root=checkpoint_root,
    )
    reconciled = controller.reconcile()
    if not isinstance(reconciled, Mapping):
        raise ValueError("reconcile 回傳值必須是 progress mapping")
    return dict(reconciled)


def _validate_worker_shard_ids(
    plan: Mapping[str, object], requested_ids: Sequence[str]
) -> tuple[str, ...]:
    """驗證 worker 的完整 shard 清單，並保留 caller 指定的執行順序。

    ``run-worker`` 必須在第一片開始前確認所有 ID 都屬於同一份 immutable run plan；否則
    第一片可能已經取得鎖、建立 checkpoint 或讀取 forcing，才發現後續 ID 拼寫錯誤。
    這裡同時拒絕命令列重複、plan 內重複與缺少的 shard，且不做自動排序、補片或跨 run
    搜尋。真正的 scenario、seed、checkpoint、provenance 與 lock 校驗仍由既有
    ``RunController.run_shard`` 保留執行，這個 helper 只負責 worker 的批次入口邊界。
    """

    if not requested_ids:
        raise ValueError("run-worker 至少需要一個 --shard-id")
    if any(type(shard_id) is not str or not shard_id for shard_id in requested_ids):
        raise ValueError("run-worker 的 --shard-id 必須是非空字串")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("run-worker 的 --shard-id 不可重複")

    raw_shards = plan.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("run plan shards 必須是 list")
    plan_ids: list[str] = []
    for row in raw_shards:
        if not isinstance(row, Mapping) or type(row.get("shard_id")) is not str:
            raise ValueError("run plan shard 缺少合法 shard_id")
        plan_ids.append(row["shard_id"])
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("run plan 的 shard_id 不可重複")

    plan_id_set = set(plan_ids)
    missing = [shard_id for shard_id in requested_ids if shard_id not in plan_id_set]
    if missing:
        raise ValueError("run-worker 指定的 shard_id 不存在於 run plan")
    return tuple(requested_ids)


def _parse_pilot_arrival_utc_options(values: Sequence[str] | None) -> dict[str, str]:
    """把 CLI 的重複 ``SITE=UTC`` 選項轉成 builder 使用的 mapping。

    CLI 層只處理選項結構與重複 site；精確 UTC 格式、版本化 hsinchu 時刻、產品軸與
    gap-safe 支援仍由 ``input_derivation`` 的單一 policy gate 驗證。這樣命令列不會
    另行實作一套可與 Python API 漂移的時間語意，也能在 formal 模式於呼叫 builder
    前拒絕 pilot 選項而不建立 destination 或 partial directory。
    """

    if not values:
        return {}
    result: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or raw.count("=") != 1:
            raise ValueError("--pilot-arrival-utc 必須使用 SITE=UTC 格式")
        site_id, utc_value = raw.split("=", 1)
        if (
            not site_id
            or site_id != site_id.strip()
            or not utc_value
            or utc_value != utc_value.strip()
        ):
            raise ValueError("--pilot-arrival-utc 的 SITE 與 UTC 不可為空或含前後空白")
        if site_id in result:
            raise ValueError(f"--pilot-arrival-utc 不可重複指定 site：{site_id}")
        result[site_id] = utc_value
    return result


def run_config_check(argv: Sequence[str] | None = None) -> int:
    """執行設定驗證並輸出 canonical hash。"""

    args = _config_check_parser().parse_args(argv)
    config = load_config(args.config, formal_release=args.formal_release)
    print(
        json.dumps(
            {
                "config": str(args.config),
                "config_hash": config.config_hash(),
                "formal_release_checked": bool(args.formal_release),
                "flow_domain_count": len(config.domains),
                "study_site_count": len(config.study_sites),
                "scenario_count": config.scenarios.scenario_count,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_preflight_command(argv: Sequence[str] | None = None) -> int:
    """執行唯讀月份檢查、原子寫報告，正式模式有 error 時回傳 2。"""

    args = _preflight_parser().parse_args(argv)
    config = load_config(args.config, formal_release=args.formal_release)
    ocm_root = _root_from_argument_or_env(args.ocm_native_root, config.inputs.ocm_native_root_env)
    nww_root = _root_from_argument_or_env(args.nww_analysis_root, config.inputs.nww_analysis_root_env)
    report = run_preflight(
        config,
        ocm_native_root=ocm_root,
        nww_analysis_root=nww_root,
        months=args.months,
        formal_release=args.formal_release,
    )
    report.write_json(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "formal_ready": report.formal_ready,
                "finding_count": len(report.findings),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.formal_ready or not args.formal_release else 2


def run_inputs_build(argv: Sequence[str] | None = None) -> int:
    """建立並輸出 SERVER v3 input artifact 摘要。

    builder 只讀取已驗收的 OCM schema 3／NWW3 analysis schema 1，所有 component、sidecar
    與 artifact closure 會先在 partial directory 完成，成功後才發布 final 目錄。CLI 無論
    是否指定 ``--formal-release`` 都固定傳入 ``strict=True``，因此遇到缺失、未驗收或
    無法取得的資料時會 fail-closed，不會啟用僅供小型 fixture 的 synthetic constant-field
    fallback。``--formal-release`` 仍只控制 approved status、完整時段與 gap-safe horizon
    等正式發布閘門；CLI 最後只印出 JSON-safe 的產物摘要。
    """

    args = _inputs_build_parser().parse_args(argv)
    if args.formal and args.pilot_arrival_utc:
        # 必須在 builder 讀取 config、accepted roots 或建立 destination partial 前拒絕；
        # 這是 pilot-only policy 的副作用邊界，不讓 formal 呼叫留下半成品。
        raise ValueError("--formal-release 不得搭配 --pilot-arrival-utc；pilot 只允許非正式 build")
    pilot_arrival_utc = _parse_pilot_arrival_utc_options(args.pilot_arrival_utc)
    result = build_input_derivatives(
        config_path=args.config,
        destination=args.destination,
        ocm_native_root=args.ocm_native_root,
        ocm_surface_root=args.ocm_surface_root,
        nww_analysis_root=args.nww_analysis_root,
        formal=args.formal,
        # CLI 是 production pipeline 的入口；即使是非正式 pilot，也不得以合成常值場補資料。
        strict=True,
        pilot_arrival_utc=pilot_arrival_utc or None,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_inputs_validate(argv: Sequence[str] | None = None) -> int:
    """唯讀驗證衍生 input 目錄並以 valid 對應 shell exit code。"""

    args = _inputs_validate_parser().parse_args(argv)
    directory = args.input_directory or args.directory
    if directory is None:
        raise ValueError("inputs-validate 必須提供 input artifact 目錄")
    result = validate_input_derivatives(
        directory,
        config_path=args.config,
        formal=args.formal,
        ocm_native_root=args.ocm_native_root,
        ocm_surface_root=args.ocm_surface_root,
        nww_analysis_root=args.nww_analysis_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("valid") is True else 2


def run_release_config_create(argv: Sequence[str] | None = None) -> int:
    """由 input artifact 產生新的 release config，並輸出 approved/generated 狀態。"""

    args = _release_config_create_parser().parse_args(argv)
    # 舊呼叫未提供新 flags 時維持原本的關鍵字集合，讓外部 wrapper／測試的 mock
    # 仍可相容；只在 caller 明示覆寫值時才把新參數傳給核心函式。
    release_kwargs: dict[str, Any] = {
        "config_template_path": args.config_template,
        "input_directory": args.input_directory,
        "output_path": args.output,
        "formal": args.formal,
    }
    if args.max_backtrack_days is not None:
        release_kwargs["max_backtrack_days"] = args.max_backtrack_days
    if args.maximum_step_count is not None:
        release_kwargs["maximum_step_count"] = args.maximum_step_count
    result = create_release_config(
        **release_kwargs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_release_config_validate(argv: Sequence[str] | None = None) -> int:
    """唯讀驗證 release config 的 component path、raw/canonical hash 與正式 gate。"""

    args = _release_config_validate_parser().parse_args(argv)
    result = validate_release_config(
        args.path,
        input_directory=args.input_directory,
        formal=args.formal,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("valid") is True else 2


def run_horizon_suite_create(argv: Sequence[str] | None = None) -> int:
    """建立共用最長回溯輸入與多份 horizon release config，並輸出 JSON 摘要。

    CLI 只負責解析 caller 明示的 template、日數、目的地及三個 accepted forcing
    root；日數保持 argparse 收到的原始順序傳給核心，由核心統一做唯一性與排序檢查。
    ``formal`` 只表達正式／pilot gate 模式，不在 handler 內複製任何輸入衍生或
    release 設定邏輯。核心例外維持既有 CLI 錯誤機制，避免把 partial suite 誤報成成功。
    """

    args = _horizon_suite_create_parser().parse_args(argv)
    result = build_horizon_suite(
        config_template_path=args.config_template,
        backtrack_days=args.backtrack_days,
        destination=args.destination,
        ocm_native_root=args.ocm_native_root,
        ocm_surface_root=args.ocm_surface_root,
        nww_analysis_root=args.nww_analysis_root,
        formal=args.formal,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_horizon_suite_resume(argv: Sequence[str] | None = None) -> int:
    """由 exact preserved partial 重建 suite，輸出新的 atomic destination 摘要。

    handler 不接受 input directory override，也不自行複製或推導 manifest；所有
    closure、source hash、formal gate、六份 release 與 exclusive publish 都由核心
    ``resume_horizon_suite`` 統一處理。原 partial 若失敗，仍保留供稽核。
    """

    args = _horizon_suite_resume_parser().parse_args(argv)
    result = resume_horizon_suite(
        preserved_partial=args.partial,
        destination=args.destination,
        backtrack_days=args.backtrack_days,
        ocm_native_root=args.ocm_native_root,
        ocm_surface_root=args.ocm_surface_root,
        nww_analysis_root=args.nww_analysis_root,
        formal=args.formal,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_horizon_suite_validate(argv: Sequence[str] | None = None) -> int:
    """唯讀驗證 horizon suite，並以 validator 的 ``valid`` 映射 shell 狀態。

    suite path 是唯一的輸入母體識別；common input 必須由核心固定解析為 suite 內的
    ``common-input``，因此 handler 不接受或轉送外部 input directory。三個 forcing
    root 可由 caller 明示以重做來源與 checksum 檢查；驗證報告原樣輸出為 JSON，合法
    回傳 0，任何 ``valid`` 非嚴格 ``True`` 的報告回傳 2。
    """

    args = _horizon_suite_validate_parser().parse_args(argv)
    result = validate_horizon_suite(
        args.path,
        formal=args.formal,
        ocm_native_root=args.ocm_native_root,
        ocm_surface_root=args.ocm_surface_root,
        nww_analysis_root=args.nww_analysis_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if isinstance(result, Mapping) and result.get("valid") is True else 2


def run_pilot_calibrate(argv: Sequence[str] | None = None) -> int:
    """以 accepted OCM product 建立 pilot calibration evidence 並輸出摘要。

    builder 內部先做 release／actual-z／geometry gate，再以 OCM-only lazy manager 取樣；
    handler 只輸出完成狀態與候選摘要，不把 pair 科學資料展開到 stdout。任何 input 或
    immutable publish 失敗都會讓命令以例外退出，避免排程器採認半套目錄。
    """

    args = _pilot_calibrate_parser().parse_args(argv)
    destination = build_pilot_calibration(
        config_path=args.config,
        input_directory=args.input_directory,
        ocm_native_root=args.ocm_native_root,
        destination=args.destination,
        project_root=args.project_root,
        pair_limit_per_site=args.pair_limit_per_site,
    )
    artifact = read_pilot_calibration(destination)
    report = artifact.report
    counts = report.get("counts", {})
    print(
        json.dumps(
            {
                "destination": str(destination),
                "schema_version": report.get("schema_version"),
                "evidence_class": report.get("evidence_class"),
                "completion_status": report.get("completion_status"),
                "selected_pair_count": counts.get("selected_pair_count")
                if isinstance(counts, Mapping)
                else None,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_pilot_calibrate_validate(argv: Sequence[str] | None = None) -> int:
    """唯讀驗證 pilot calibration evidence 並以 valid 對應 shell exit code。"""

    args = _pilot_calibrate_validate_parser().parse_args(argv)
    result = validate_pilot_calibration(
        args.path,
        config_path=args.config,
        input_directory=args.input_directory,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("valid") is True else 2


def run_pilot_config_create(argv: Sequence[str] | None = None) -> int:
    """建立並驗證 generated pilot execution config，輸出 JSON-safe binding 摘要。

    handler 僅把 parser 已轉成原生 Python scalar 的值交給 builder；source release、input
    artifact hash、calibration evidence、gap-safe horizon、atomic publish 與 target
    round-trip 都由 ``create_pilot_execution_config`` 完成。這個命令成功只表示候選值
    已綁定且設定可供 pilot runtime 讀取，不代表通過 trajectory 或 ensemble 科學收斂。
    """

    args = _pilot_config_create_parser().parse_args(argv)
    result = create_pilot_execution_config(
        args.source_config,
        args.input_directory,
        args.calibration,
        args.output,
        dt_min_seconds=args.dt_min_seconds,
        dt_max_seconds=args.dt_max_seconds,
        output_interval_seconds=args.output_interval_seconds,
        max_backtrack_days=args.max_backtrack_days,
        maximum_step_count=args.maximum_step_count,
        members_per_scenario=args.members_per_scenario,
        master_seed=args.master_seed,
        shard_scenario_count=args.shard_scenario_count,
        checkpoint_interval_sweeps=args.checkpoint_interval_sweeps,
        active_chunk_size=args.active_chunk_size,
        max_resident_forcing_months=args.max_resident_forcing_months,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


def run_pilot_config_validate(argv: Sequence[str] | None = None) -> int:
    """唯讀驗證 pilot execution config 並以 valid 對應 shell exit code。"""

    args = _pilot_config_validate_parser().parse_args(argv)
    result = validate_pilot_execution_config(
        args.path,
        input_directory=args.input_directory,
        calibration_directory=args.calibration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("valid") is True else 2


def run_pilot_matrix_validate(argv: Sequence[str] | None = None) -> int:
    """唯讀比較多個 pilot run 的共同設定並輸出 canonical JSON。

    run roots 必須由 caller 明示；validator 會在每個 root 內讀取 immutable plan/config，
    只將 ``valid`` 映射成 shell 狀態 0 或 2。輸出採 canonical JSON（排序 key、無多餘
    空白），讓 SERVER runbook 可以直接保存、hash 或與其他區域的檢查結果比對。
    """

    args = _pilot_matrix_validate_parser().parse_args(argv)
    report = validate_pilot_matrix(args.run_roots)
    print(canonical_pilot_matrix_json(report))
    return 0 if report.get("valid") is True else 2


def _current_design_version() -> str:
    """取得 behavior manifest 的目前設計版本，避免 CLI 固定寫死歷史版本。

    ``CURRENT_DESIGN_VERSION`` 由設定模組維護，代表目前可接受的 material component
    identity。延後 import 是為了讓設定模組在載入期間不形成循環依賴；若舊部署尚未提供
    常數，則從同一個 source checkout 的 example config 讀取其 design_version，仍拒絕
    回退到歷史 v2 或自行猜測版本。
    """

    from . import config as config_module

    current = getattr(config_module, "CURRENT_DESIGN_VERSION", None)
    if isinstance(current, str) and current.strip() and current == current.strip():
        return current
    example_path = Path(__file__).resolve().parents[2] / "configs" / "lagrangian_backtracking.example.yaml"
    try:
        return load_config(example_path, formal_release=False).design_version
    except Exception as exc:
        raise RuntimeError("config 未提供 CURRENT_DESIGN_VERSION，且無法讀取 example config") from exc


def run_behavior_manifest(argv: Sequence[str] | None = None) -> int:
    """原子寫出十種非上浮海廢代理；既有檔案不覆寫。

    輸出同時保存 iOcean 分類名稱、代表材質、形狀、適用條件、證據等級與暫定速度，
    讓後續批次不會只看見匿名速度。官方分類只作臺灣情境命名；速度仍是待現地樣本
    校準的敏感度格點，因此 manifest 明確標示來源與不可作類別平均物性的限制。
    """

    args = _behavior_manifest_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"不可覆寫既有 manifest：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0.0",
        "design_version": _current_design_version(),
        "classification_source": ("海洋保育署 iOcean 海洋廢棄物管理頁；2026-08-27 查閱；只作分類命名"),
        "velocity_unit": "m s-1; z positive-up; all values must be strictly negative",
        "velocity_source": "design_sensitivity_grid_not_oca_measurement",
        "positive_or_zero_velocity_policy": "reject_config",
        "calibration_scope": "provisional_material_shape_proxy_pending_local_measurement",
        "records": records_as_dicts(BASELINE_BEHAVIORS),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "behavior_count": len(BASELINE_BEHAVIORS)},
            ensure_ascii=False,
        )
    )
    return 0


def run_synthetic_smoke(argv: Sequence[str] | None = None) -> int:
    """執行常流 backward 垂直切片，原子發布後立即重驗 shard。

    smoke 不代表真實 forcing 或科學成果；其用途是確認 CLI、signed-time RK4、巢狀邊界、
    ragged arrays、Parquet、manifest 與 checksum 在乾淨環境可串接。
    """

    args = _synthetic_smoke_parser().parse_args(argv)
    geometry = BoundaryGeometry(
        own_local_domain=box(-5_000.0, -2_000.0, 5_000.0, 2_000.0),
        flow_domain=box(-20_000.0, -5_000.0, 20_000.0, 5_000.0),
        foreign_local_domains={"guishan": box(-15_000.0, -2_000.0, -10_000.0, 2_000.0)},
    )

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """提供固定 0.5 m/s 東向流；參數保留以符合 production provider protocol。"""

        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(0.5, 0.0, 0.0, 0.0, -100.0, 1_000.0, 2.0)

    initial = ParticleState(
        particle_id="synthetic-p0000",
        scenario_id="synthetic-s0000",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="synthetic-r0000",
        x_m=0.0,
        y_m=0.0,
        z_m=-10.0,
        time_utc_ns=1_704_067_200_000_000_000,
    )
    result = run_particle(
        initial,
        velocity=velocity,
        boundaries=geometry,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=EngineSettings(
            dt_min_seconds=60.0,
            dt_max_seconds=600.0,
            output_interval_seconds=600.0,
            max_backtrack_seconds=86_400.0,
            maximum_step_count=1_000,
            earliest_forcing_time_utc_ns=1_703_000_000_000_000_000,
        ),
        rng=np.random.default_rng(20260819),
    )
    write_trajectory_shard(
        args.output,
        [result],
        run_metadata={
            "run_kind": "synthetic_smoke_not_scientific_result",
            "config_hash": "synthetic_constant_flow_v1",
            "input_inventory_hash": "synthetic_no_external_input",
            "seed_policy": "fixed_20260819",
        },
    )
    validation = validate_trajectory_shard(args.output)
    print(
        json.dumps(
            {"output": str(args.output), "valid": validation["valid"], "errors": validation["errors"]},
            ensure_ascii=False,
        )
    )
    return 0 if validation["valid"] else 2


def run_validate_shard(argv: Sequence[str] | None = None) -> int:
    """驗證既有 shard 並以 JSON/exit code 回報。"""

    args = _validate_shard_parser().parse_args(argv)
    validation = validate_trajectory_shard(args.path)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 2


def run_code_provenance(argv: Sequence[str] | None = None) -> int:
    """輸出目前部署的 code provenance JSON；正式 gate 失敗時由例外阻止發布。"""

    args = _code_provenance_parser().parse_args(argv)
    provenance = collect_code_provenance(
        args.project_root,
        declared_git_commit=args.declared_git_commit,
        formal=args.formal,
    )
    print(json.dumps(provenance.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_validate_run(argv: Sequence[str] | None = None) -> int:
    """驗證 run workspace 並以 valid/error JSON 與 shell exit code 回報。"""

    args = _validate_run_parser().parse_args(argv)
    result = validate_run(
        args.path,
        require_complete=args.require_complete,
        checkpoint_root=args.checkpoint_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


def run_benchmark_report(argv: Sequence[str] | None = None) -> int:
    """輸出唯讀 engineering benchmark report；不把量測宣稱成科學成果。"""

    args = _benchmark_report_parser().parse_args(argv)
    result = benchmark_report(
        args.path,
        require_complete=args.require_complete,
        checkpoint_root=args.checkpoint_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


def run_create(argv: Sequence[str] | None = None) -> int:
    """建立 pilot/formal run、立即驗證 workspace，並輸出 orchestration 摘要。

    CLI 先解析 config、inventory、destination、run identity 與 experiment case，再依
    ``--run-kind`` 呼叫對應 initializer。formal 的設定、manifest、inventory gate 全部
    在 initializer 內執行；任何 gate 失敗都不會 downgrade 成 pilot。initializer 完成
    immutable plan、scenario／seed table 與目錄的原子發布後，本 handler 立刻以
    ``validate_run(..., require_complete=False)`` 做只讀驗證。失敗例外只帶 validator
    錯誤碼，不帶 workspace 或 SERVER 絕對路徑；成功 JSON 的 workspace 只供 CLI 顯示，
    並不回寫 plan。
    精確選擇的三個識別碼只傳給 pilot 建立器，再由計畫繫結持久保存；此處先拒絕
    缺項、重複、首尾空白與正式／分層形式混用，不自行讀取或裁剪來源清單。
    """

    args = _run_create_parser().parse_args(argv)
    if args.run_kind == "formal" and args.pilot_scenarios_per_stratum is not None:
        raise ValueError("formal run 禁止 --pilot-scenarios-per-stratum")
    exact_options = {
        name: getattr(args, name)
        for name in ("pilot_study_site_id", "pilot_arrival_id", "pilot_material_id")
    }
    exact_requested = any(values is not None for values in exact_options.values())
    if exact_requested:
        if args.run_kind != "pilot":
            raise ValueError("formal run 禁止 pilot_exact")
        if args.pilot_scenarios_per_stratum is not None:
            raise ValueError("pilot_exact 不可與 --pilot-scenarios-per-stratum 混用")
        if any(
            values is None or len(values) != 1 or not values[0] or values[0] != values[0].strip()
            for values in exact_options.values()
        ):
            raise ValueError("pilot_exact 必須各一次明示完整且無首尾空白的三個識別碼")
    initializer = initialize_formal_run if args.run_kind == "formal" else initialize_pilot_run
    initializer_kwargs: dict[str, object] = {
        "config_path": args.config,
        "input_inventory_path": args.input_inventory,
        "destination": args.destination,
        "run_id": args.run_id,
        "experiment_case_id": args.experiment_case_id,
        "project_root": args.project_root,
        "declared_git_commit": args.declared_git_commit,
    }
    # None 代表完整 pilot，不額外把新參數送進舊 caller 的 wrapper；指定 N 時才傳給
    # pilot initializer，formal 已在上方先拒絕，避免任何 manifest／workspace I/O。
    if args.run_kind == "pilot" and args.pilot_scenarios_per_stratum is not None:
        initializer_kwargs["pilot_scenarios_per_stratum"] = args.pilot_scenarios_per_stratum
    if exact_requested:
        initializer_kwargs.update({name: values[0] for name, values in exact_options.items()})
    workspace = initializer(**initializer_kwargs)
    validation = validate_run(workspace, require_complete=False)
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        codes = _validation_error_codes(validation)
        raise ValueError("run-create 建立後驗證失敗：" + "; ".join(codes))

    summary = validation.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("run-create validator summary 缺少或不是 mapping")
    try:
        counts = {key: int(summary[key]) for key in ("scenario_count", "particle_count", "shard_count")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run-create validator summary 缺少必要計數") from exc
    payload = {
        "workspace": _workspace_display_path(workspace),
        "run_id": args.run_id,
        "run_kind": args.run_kind,
        "experiment_case_id": args.experiment_case_id,
        **counts,
        "valid": True,
    }
    # 真實 RunWorkspace 一定帶 immutable plan；測試或外部 wrapper 若只提供 path，保留舊
    # JSON 形狀而不猜測 selection。此處只輸出 mode/count，不輸出 scenario ID 或 SERVER path。
    plan = getattr(workspace, "plan", None)
    selection = plan.get("scenario_selection") if isinstance(plan, Mapping) else None
    if isinstance(selection, Mapping):
        payload.update(
            {
                "selection_mode": selection.get("mode"),
                "source_scenario_count": selection.get("source_scenario_count"),
                "selected_scenario_count": selection.get("selected_scenario_count"),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_shard(argv: Sequence[str] | None = None) -> int:
    """在通過 workspace pre-validation 後執行單一 pilot/formal shard。

    第一個有副作用風險的動作是呼叫 validate_run；在它回報 invalid 前，handler
    不讀取設定、不解析任何 forcing root 環境變數，也不建立 controller。驗證通過後
    才重新載入 immutable plan 並取得 ``run_kind``，再以相同模式呼叫 ``load_config``；
    因此 formal 不可能以 pilot config 開啟。CLI root 優先於設定指定的環境變數，OCM
    root 兩者皆缺時拒絕，NWW root 皆缺時保留 None 交給 runtime 依 experiment case
    fail-closed。root 只作本次開啟 controller 的輸入，不寫入 plan 或輸出 JSON。
    controller 建立後只呼叫指定的 run_shard 一次，將 immutable RunExecutionSummary
    以 dataclass asdict 轉成排序 UTF-8 JSON；PAUSED 與 COMPLETE 都代表本次
    orchestration 呼叫成功並回傳 0。
    """

    args = _run_shard_parser().parse_args(argv)
    validation = validate_run(
        args.workspace,
        require_complete=False,
        checkpoint_root=args.checkpoint_root,
    )
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return 2

    # validator 通過後才讀 immutable plan；invalid workspace 不會因此載入 config、環境
    # 變數、forcing root 或 controller。run-control 仍可能辨識 synthetic plan，但 physical
    # runtime 僅允許 pilot/formal，未知模式不得 fallback。
    plan = load_run_plan(args.workspace)
    run_kind = plan.get("run_kind")
    if type(run_kind) is not str or run_kind not in {"pilot", "formal"}:
        raise ValueError("run plan run_kind 只允許 pilot 或 formal")
    config = load_config(args.config, formal_release=run_kind == "formal")
    input_config = config.inputs
    ocm_root = _root_from_argument_or_env(
        args.ocm_native_root,
        input_config.ocm_native_root_env,
    )
    reconstruction_root = _optional_root_from_env(
        args.ocm_reconstruction_root,
        getattr(input_config, "ocm_reconstruction_root_env", None) or "OCM_RECONSTRUCTION_ROOT",
    )
    nww_root = _optional_root_from_env(
        args.nww_analysis_root,
        input_config.nww_analysis_root_env,
    )
    controller = open_worker_controller(
        args.workspace,
        config_path=args.config,
        ocm_native_root=ocm_root,
        nww_analysis_root=nww_root,
        ocm_reconstruction_root=reconstruction_root,
        resume=args.resume,
        checkpoint_root=args.checkpoint_root,
        continuous_time_fallback=args.execution_priority,
    )
    summary = controller.run_shard(args.shard_id, sweep_budget=args.sweep_budget)
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    return 0


def run_worker(argv: Sequence[str] | None = None) -> int:
    """在同一個 runtime controller 內依指定順序連續執行多個 shard。

    先做一次既有 workspace validator、讀取 immutable plan 並驗證全部 ``--shard-id``；
    只有所有 ID 都存在且不重複時，才載入 config、解析 forcing root、開啟一次
    ``RunController``。之後每片都呼叫同一個 controller 的 ``run_shard``，讓其既有
    RuntimeRequestFactory、流場月份管理器、網格與 lock／checkpoint／provenance gate
    依原契約運作。某片回傳 ``PAUSED`` 即停止後續片；任何例外（包含 lock contention）
    直接向 caller 傳出，不以跳過或部分成功掩蓋錯誤。只有明示
    ``--execution-priority`` 時，才把可隔離的單片例外記為 ``FAILED`` 並繼續後續片；
    這個入口本身不負責 formal provenance override，正式平行入口才會保存該摘要。

    成功時 stdout 只輸出一份 JSON，內含所有實際執行片的 ``RunExecutionSummary``、
    handler 前置時間與整個命令時間。時間從本 handler 開始計算，明確不包含 Python
    啟動與模組匯入；它是單一程序的 command wall／CPU 時間，不是多 worker 整機經過時間。

    ``--sweep-budget`` 會原樣套用到每一片；resume、external checkpoint root、forcing
    root 與 config 也沿用 ``run-shard`` 的參數與驗證邊界，不跨 run 共用任何 runtime。
    """

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    args = _run_worker_parser().parse_args(argv)
    validation = validate_run(
        args.workspace,
        require_complete=False,
        checkpoint_root=args.checkpoint_root,
    )
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return 2

    # validator 通過後才讀 immutable plan；這一步同時阻止 config／forcing／controller
    # 在 workspace invalid 時建立。所有指定 ID 先完整檢查，再進入任何 shard 執行。
    plan = load_run_plan(args.workspace)
    run_kind = plan.get("run_kind")
    if type(run_kind) is not str or run_kind not in {"pilot", "formal"}:
        raise ValueError("run plan run_kind 只允許 pilot 或 formal")
    run_id = plan.get("run_id")
    if type(run_id) is not str:
        raise ValueError("run plan run_id 必須是字串")
    shard_ids = _validate_worker_shard_ids(plan, args.shard_id)

    config = load_config(args.config, formal_release=run_kind == "formal")
    input_config = config.inputs
    ocm_root = _root_from_argument_or_env(
        args.ocm_native_root,
        input_config.ocm_native_root_env,
    )
    reconstruction_root = _optional_root_from_env(
        args.ocm_reconstruction_root,
        getattr(input_config, "ocm_reconstruction_root_env", None) or "OCM_RECONSTRUCTION_ROOT",
    )
    nww_root = _optional_root_from_env(
        args.nww_analysis_root,
        input_config.nww_analysis_root_env,
    )
    controller = open_worker_controller(
        args.workspace,
        config_path=args.config,
        ocm_native_root=ocm_root,
        nww_analysis_root=nww_root,
        ocm_reconstruction_root=reconstruction_root,
        resume=args.resume,
        checkpoint_root=args.checkpoint_root,
        continuous_time_fallback=args.execution_priority,
    )
    preflight_wall = time.perf_counter() - started_wall
    preflight_cpu = time.process_time() - started_cpu

    summaries: list[dict[str, object]] = []
    paused_shard_id: str | None = None
    for shard_id in shard_ids:
        # 每一片共用同一個 controller；run-control 仍會為各 shard 取得自己的 lock，
        # 並以既有 checkpoint／輸出 validator 判定 COMPLETE、PAUSED 或失敗。
        try:
            summary = execute_worker_shard(
                controller,
                shard_id,
                sweep_budget=args.sweep_budget,
            )
        except BaseException as exc:  # noqa: BLE001 - priority mode仍須保留 fatal 邊界
            if not args.execution_priority:
                raise
            # ``run-worker`` 與正式 dynamic child 共用相同的 failure-isolation 分類：
            # 數值／單片例外可以留下 FAILED 摘要後續跑，但記憶體、訊號、鎖與陣列／網格
            # 結構錯誤不能把已初始化的 controller 當成仍可安全重用。lazy import 避免
            # cli 在 dynamic child bootstrap 時與 dynamic_execution 形成模組循環。
            from .dynamic_execution import _worker_exception_is_fatal

            if _worker_exception_is_fatal(exc):
                raise
            summary = {
                "shard_id": shard_id,
                "lifecycle": "FAILED",
                "error_code": "run_shard_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        summaries.append(summary)
        if summary.get("lifecycle") == "PAUSED":
            paused_shard_id = shard_id
            break

    command_wall = time.perf_counter() - started_wall
    command_cpu = time.process_time() - started_cpu
    payload = {
        "artifact_type": "run_worker_execution_summary",
        "schema_version": "1.0.0",
        "run_id": run_id,
        "requested_shard_ids": list(shard_ids),
        "executed_shards": summaries,
        "stopped_after_paused_shard_id": paused_shard_id,
        "timing": {
            "includes_python_import": False,
            "includes_python_startup": False,
            "preflight_wall_seconds": preflight_wall,
            "preflight_process_cpu_seconds": preflight_cpu,
            "command_wall_seconds": command_wall,
            "command_process_cpu_seconds": command_cpu,
            "scope": "單一程序 run-worker handler；不含 Python 啟動與模組匯入",
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_formal_parallel(argv: Sequence[str] | None = None) -> int:
    """執行 formal parallel coordinator 並輸出整機摘要，不將部分成果標成 COMPLETE。"""

    from .parallel_execution import execute_formal_parallel

    args = _run_formal_parallel_parser().parse_args(argv)
    try:
        summary, exit_code = execute_formal_parallel(
            workspace=args.workspace,
            config_path=args.config,
            project_root=args.project_root,
            worker_count=args.worker_count,
            scratch_root=args.scratch_root,
            log_root=args.log_root,
            storage_gate_evidence=args.storage_gate_evidence,
            checkpoint_root=args.checkpoint_root,
            numba_cache_dir=args.numba_cache_dir,
            ocm_native_root=args.ocm_native_root,
            ocm_reconstruction_root=args.ocm_reconstruction_root,
            nww_analysis_root=args.nww_analysis_root,
            resume=args.resume,
            cpu_affinity=args.cpu_affinity,
            warmup_only=args.warmup_only,
            execution_mode=args.execution_mode,
            poll_interval_seconds=args.poll_interval_seconds,
            shutdown_grace_seconds=args.shutdown_grace_seconds,
            response_timeout_seconds=args.worker_response_timeout_seconds,
            execution_priority=args.execution_priority,
            defer_full_validation=args.defer_full_validation,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 必須以機器可讀失敗狀態 fail closed
        print(
            json.dumps(
                {
                    "artifact_type": "formal_parallel_coordinator_error",
                    "schema_version": "1.0.0",
                    "status": "COORDINATOR_ERROR",
                    "valid": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    # 一般例外可能發生於驗證階段，也可能出現在 child 已結束後的摘要／驗證階段；
                    # 未取得正式執行摘要時不得錯誤宣稱「尚未啟動 child」。
                    "children_started": None,
                    "full_validation_deferred": bool(args.defer_full_validation),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def run_reconcile(argv: Sequence[str] | None = None) -> int:
    """只 reconcile checkpoint／progress，禁止載入 config 或進入物理計算。

    本 handler 建立的本地 reject factory 僅作安全網；reconcile 應只讀取並必要時修復
    run-control 的狀態與 checkpoint pointer，若任何路徑意外要求建立 particle request，
    factory 會立即 raise，避免偷偷載入 geometry、forcing 或執行物理計算。reconcile
    完成後才以同一個 workspace 與 external checkpoint root 呼叫只讀 validator，輸出
    run identity、revision、run lifecycle 計數與 validator errors；摘要刻意不包含
    checkpoint path。external root 只存在於本次呼叫參數，不會持久化到 plan 或 JSON。
    """

    args = _run_reconcile_parser().parse_args(argv)

    reconciled = reconcile_run_for_resume(
        args.workspace,
        checkpoint_root=args.checkpoint_root,
    )
    validation = validate_run(
        args.workspace,
        require_complete=False,
        checkpoint_root=args.checkpoint_root,
    )
    raw_errors = validation.get("errors", []) if isinstance(validation, Mapping) else []
    errors = [error for error in raw_errors if type(error) is str] if isinstance(raw_errors, list) else []
    payload = {
        "run_id": reconciled.get("run_id"),
        "run_lifecycle": reconciled.get("run_lifecycle"),
        "revision": reconciled.get("revision"),
        "shard_lifecycle_counts": _shard_lifecycle_counts(reconciled),
        "valid": bool(validation.get("valid")) if isinstance(validation, Mapping) else False,
        "errors": errors,
    }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if payload["valid"] else 2


def _require_ordinary_directory_node(path: Path) -> None:
    """以 ``lstat`` 驗證 aggregate CLI 使用的既有目錄節點。

    run root 與 ``locks`` 是取得 exclusive gate 前的安全邊界；只檢查最後一個節點，
    因而不會把合法的祖先路徑 alias 當成錯誤。此 helper 不呼叫 ``is_dir``，也不跟隨
    symbolic link；上層 handler 會把所有一般失敗收斂成不含路徑的固定錯誤。
    """

    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("aggregate CLI directory node 必須是普通非 symlink 目錄")


def _require_ordinary_file_node(path: Path) -> None:
    """以 ``lstat`` 驗證 aggregate CLI 使用的既有普通非 symlink 檔案。"""

    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("aggregate CLI file node 必須是普通非 symlink 檔案")


def _validate_aggregate_spec_destination(*, source_root: Path, destination: Path) -> None:
    """拒絕把 aggregate spec 寫回 immutable run 或透過 symlink 逃逸。

    ``source_root`` 內的 run plan、progress、checkpoint 與 trajectory topology 是已驗證
    後不可變的輸入；spec 若寫入該目錄或其子層，會把輸入與輸出混在一起，破壞後續
    release pipeline 對 source snapshot 的假設。因此這裡只允許 source sibling 或其他
    已存在的普通目錄。destination 的 parent 先以 ``lstat`` 驗證最後節點不是 symlink，
    再把 source root 與 parent 都以 ``resolve(strict=True)`` 封閉到實際目錄，避免透過
    parent alias／符號連結間接寫入 run。這是路徑安全檢查，不改變 spec writer 自己對
    destination 最終檔案「不可覆寫」的契約；所有公尺制 geometry 與秒制 age 語意仍由
    AggregateSpec 契約驗證，local synthetic 目錄也不因此成為 OCM／NWW 科學成果。
    """

    destination_parent = destination.parent
    _require_ordinary_directory_node(destination_parent)
    source_resolved = source_root.resolve(strict=True)
    destination_parent_resolved = destination_parent.resolve(strict=True)
    if destination_parent_resolved == source_resolved or destination_parent_resolved.is_relative_to(
        source_resolved
    ):
        raise ValueError("aggregate spec destination 不得位於 source run")


def run_aggregate_spec_create(argv: Sequence[str] | None = None) -> int:
    """在 source run gate 內建立並回讀一份 AggregateSpec。

    這個 handler 僅把命令列明示的研究參數交給既有 geometry writer：格網、邊界分箱
    與 KDE 帶寬的距離單位是公尺（m），age 邊界是秒（s），bootstrap 數值則由
    AggregateSpec 契約驗證。它先以 ``lstat`` 檢查 source root、``locks`` 與 gate
    檔案，再以 non-blocking exclusive lock 保護 static binding、中心推導、spec 寫入
    與回讀；static loader 只呼叫一次，且不讀 OCM/NWW 大型 array。成功輸出只包含
    spec basename、版本、run id、站點數與兩個 spec digest，不保存任何本機／SERVER
    絕對路徑。結果仍只是條件式來源足跡規格的工程輸入，不是 synthetic local 或
    OCM/NWW 科學成果。

    ``RunLockBusyError`` 必須原樣傳出，讓排程器可採取重試；其他失敗則統一成固定
    ``ValueError``，避免檔案系統或 parser 細節進入公開錯誤。
    """

    args = _aggregate_spec_create_parser().parse_args(argv)
    try:
        source_root = Path(args.run)
        _require_ordinary_directory_node(source_root)
        locks_root = source_root / "locks"
        _require_ordinary_directory_node(locks_root)
        lock_path = locks_root / "run_gate.lock"
        _require_ordinary_file_node(lock_path)

        with acquire_run_lock(lock_path, mode="exclusive", blocking=False):
            # spec 是由 run geometry 衍生出的輸出，不能寫進 immutable source run；先封閉
            # destination parent 的實際目錄 identity，再進入任何 spec writer I/O。
            _validate_aggregate_spec_destination(
                source_root=source_root,
                destination=Path(args.destination),
            )
            # run plan、config、scenario 與公尺制 geometry 必須在同一個 gate 內建立
            # static snapshot；不能在 lock 外先讀 plan 再把可能改變的現場資料交給 writer。
            static_inputs = load_validated_run_static_inputs(
                source_root,
                config_path=args.config,
                checkpoint_root=args.checkpoint_root,
                require_complete=False,
            )
            run_id = static_inputs.plan["run_id"]
            site_metric_centers_deg = _build_site_metric_centers(
                static_inputs.config,
                static_inputs.geometries,
            )
            write_aggregate_spec_from_boundaries(
                args.destination,
                static_inputs.geometries,
                run_id=run_id,
                site_metric_centers_deg=site_metric_centers_deg,
                grid_cell_size_m=args.grid_cell_size_m,
                boundary_bin_size_m=args.boundary_bin_size_m,
                kde_bandwidths_m=tuple(args.kde_bandwidths_m),
                age_bin_edges_seconds=tuple(args.age_bin_edges_seconds),
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_confidence_level=args.bootstrap_confidence_level,
                bootstrap_seed=args.bootstrap_seed,
            )
            # writer 已做 partial 驗證，但 CLI 仍以公開 loader 回讀目標 bytes，讓成功
            # JSON 的 digest 一定來自實際已發布 spec，而不是 caller 的暫存參照。
            created_spec = load_aggregate_spec(args.destination)

        result = {
            "aggregate_spec_name": args.destination.name,
            "schema_version": created_spec.schema_version,
            "run_id": created_spec.run_id,
            "site_count": len(created_spec.site_grids),
            "source_sha256": created_spec.source_sha256,
            "canonical_sha256": created_spec.canonical_sha256,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except RunLockBusyError:
        raise
    except Exception:
        raise ValueError("aggregate spec 建立失敗") from None


def run_report_spec_create(argv: Sequence[str] | None = None) -> int:
    """建立與 AggregateSpec 綁定的 report spec，並輸出 portable JSON 摘要。

    handler 只依序執行 strict ``load_aggregate_spec``、``write_report_spec``、
    ``load_report_spec`` 與 ``validate_report_spec_against_aggregate_spec``；它不讀取
    source run、forcing、OCM schema 3／NWW3 schema 1 array，也不建立 parent directory。
    帶寬與垂向深度邊界的距離單位是公尺（m），垂向軸採相對瞬時海面的
    positive-down 語意，最後 edge 外的 observation 不由 CLI 裁切。代表數與 seed 的
    邊界、垂向首點／排序及 aggregate 帶寬 membership 都由 ReportSpec writer 的
    immutable contract fail closed；本命令只保存規格，不會把 local synthetic 設定或
    後續條件式來源足跡宣稱為正式 OCM／NWW3 科學成果。

    writer 已完成 rename 但無法確認 final identity／parent durability 時，固定的
    ``RuntimeError('report spec durability confirmation failed')`` 原樣傳出，讓 SERVER
    orchestration 能辨識「檔案可能已存在但耐久性未確認」；其他一般失敗則收斂為不含
    路徑的 ``ValueError('report spec 建立失敗')``。成功 JSON 只含 basename、版本、run
    identity 與兩個 provenance hash，不含任何絕對路徑。
    """

    args = _report_spec_create_parser().parse_args(argv)
    try:
        aggregate_spec = load_aggregate_spec(args.aggregate_spec)
        write_report_spec(
            args.destination,
            aggregate_spec=aggregate_spec,
            primary_kde_bandwidth_m=args.primary_kde_bandwidth_m,
            minimum_kde_raw_count=args.minimum_kde_raw_count,
            low_sample_min_member_count=args.low_sample_min_member_count,
            vertical_depth_bin_edges_m=tuple(args.vertical_depth_bin_edges_m),
            representative_trajectory_count_per_site=args.representative_trajectory_count_per_site,
            representative_selection_seed=args.representative_selection_seed,
        )
        created_spec = load_report_spec(args.destination)
        validate_report_spec_against_aggregate_spec(created_spec, aggregate_spec)
        result = {
            "report_spec_name": args.destination.name,
            "schema_version": created_spec.schema_version,
            "run_id": created_spec.run_id,
            "aggregate_spec_canonical_sha256": created_spec.aggregate_spec_canonical_sha256,
            "source_sha256": created_spec.source_sha256,
            "canonical_sha256": created_spec.canonical_sha256,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except RuntimeError as error:
        if str(error) == "report spec durability confirmation failed":
            raise
        raise ValueError("report spec 建立失敗") from None
    except Exception:
        raise ValueError("report spec 建立失敗") from None


def run_aggregate_build(argv: Sequence[str] | None = None) -> int:
    """串流建立、原子發布並驗證 aggregate release。

    handler 先以既有 strict loader 讀取明示 AggregateSpec，再交給 payload pipeline 在
    source run exclusive gate 內串流消費完整 trajectory shard；pipeline 不建立 forcing
    manager、不讀取 OCM/NWW array，也不重新平流。payload 完成後由既有 writer 重新在
    writer gate 內 binding、寫入與 durability 確認，最後由公開 validator 做完整唯讀檢查。
    所有距離／格線是公尺（m）、travel age 是秒（s），成功結果只輸出 final basename、
    ``valid`` 與 validator 保證的十二欄 portable summary；不輸出 source、config、spec、
    checkpoint 或 destination 絕對路徑。結果描述的是條件式來源足跡／相對來源權重的
    工程統計，不因 local synthetic fixture 通過而成為 OCM/NWW 科學成果。

    ``RunLockBusyError`` 原樣保留。writer 已完成 rename 但 parent durability 未確認的
    固定 ``RuntimeError`` 也原樣保留，讓 SERVER orchestration 能區分「已發布但需處理
    durability 不確定」；其餘失敗收斂為固定 ``ValueError``。
    """

    args = _aggregate_build_parser().parse_args(argv)
    try:
        spec = load_aggregate_spec(args.spec)
        payload = build_aggregate_release_payload(
            source_run_root=args.run,
            config_path=args.config,
            aggregate_spec=spec,
            checkpoint_root=args.checkpoint_root,
        )
        release_path = write_aggregate_release(
            source_run_root=args.run,
            aggregate_spec_path=args.spec,
            payload=payload,
            destination=args.destination,
            checkpoint_root=args.checkpoint_root,
        )
        validation = validate_aggregate_release(release_path)
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            raise ValueError("aggregate release validator 未通過")
        summary = validation.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("aggregate release validator summary 必須是 mapping")
        result = {
            "release_name": release_path.name,
            "valid": True,
            "summary": dict(summary),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except RunLockBusyError:
        raise
    except RuntimeError as error:
        if str(error) == "aggregate release 已發布但 parent durability 未確認":
            raise
        raise ValueError("aggregate build 失敗") from None
    except Exception:
        raise ValueError("aggregate build 失敗") from None


def run_aggregate_validate(argv: Sequence[str] | None = None) -> int:
    """原樣輸出 aggregate release 的 pretty sorted UTF-8 validator report。

    公開 validator 已將 topology、manifest、checksum、source binding、產品與 decoder
    失敗收斂成不含路徑的 JSON-safe 報告；本 handler 不修改或包裝該報告，只以
    ``valid=True`` 對應 shell 狀態 0，其餘報告對應狀態 2。報告中的公尺／秒單位與
    條件式來源足跡語意沿用 release schema；local synthetic 驗證仍不是 OCM/NWW 科學
    成果。
    """

    args = _aggregate_validate_parser().parse_args(argv)
    report = validate_aggregate_release(args.release)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if isinstance(report, Mapping) and report.get("valid") is True else 2


def run_report_validate(argv: Sequence[str] | None = None) -> int:
    """原樣輸出 report-v1 release 的 pretty、sorted、UTF-8 validator report。

    validator 已把 report release 的 topology、exact inventory、檔案大小／SHA-256、
    ReportRegistry closure 與 source binding 失敗收斂為不含路徑及底層例外的 JSON-safe
    報告；handler 因此只解析 caller 明示的 release、呼叫 validator 並原樣序列化回傳。
    ``valid`` 僅在嚴格等於 ``True`` 時映射為 shell 狀態 0，其他固定失敗報告映射為
    狀態 2。這個命令是唯讀工程驗證，不把 local synthetic release 宣稱為 OCM／NWW
    科學成果，也不修改輸入或建立任何輸出檔案。
    """

    args = _report_validate_parser().parse_args(argv)
    report = validate_report_release(args.release)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if isinstance(report, Mapping) and report.get("valid") is True else 2


def run_source_pathway_build(argv: Sequence[str] | None = None) -> int:
    """建立並驗證向下沉降 source-pathway-v1 成果包。

    handler 不把 absolute source／destination 寫入 stdout；成功摘要只回傳 final
    basename、validator summary 與 ``valid``。任何零／正沉降 scenario、來源 hash
    錯配或 renderer gate 失敗都由 build API fail closed；低樣本則保留 raw count，
    並在 sidecar／圖面明示 zero 或 below-minimum 狀態。
    """

    args = _source_pathway_build_parser().parse_args(argv)
    try:
        release_path = build_source_pathway_release(
            aggregate_release=args.aggregate_release,
            report_spec=args.report_spec,
            destination=args.destination,
            mplconfigdir=args.mplconfigdir,
        )
        report = validate_source_pathway_release(release_path)
        if not isinstance(report, Mapping) or report.get("valid") is not True:
            raise ValueError("source pathway validator 未通過")
        summary = report.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("source pathway validator summary 必須是 mapping")
        print(
            json.dumps(
                {"release_name": release_path.name, "valid": True, "summary": dict(summary)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except RuntimeError as error:
        if str(error) == "source pathway 已發布但 parent durability 未確認":
            raise
        raise ValueError("source pathway build 失敗") from None
    except Exception:
        raise ValueError("source pathway build 失敗") from None


def run_source_pathway_validate(argv: Sequence[str] | None = None) -> int:
    """原樣輸出 source-pathway-v1 validator 的 JSON-safe 報告。"""

    args = _source_pathway_validate_parser().parse_args(argv)
    report = validate_source_pathway_release(args.release)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if isinstance(report, Mapping) and report.get("valid") is True else 2


def main(argv: Sequence[str] | None = None) -> int:
    """整合 ``lbt`` 子命令；未知命令由 argparse 以狀態 2 拒絕。"""

    parser = argparse.ArgumentParser(prog="lbt", description="Lagrangian 系集逆向溯源可重現 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("config-check", parents=[_config_check_parser()], add_help=False)
    subparsers.add_parser("preflight", parents=[_preflight_parser()], add_help=False)
    subparsers.add_parser("inputs-build", parents=[_inputs_build_parser()], add_help=False)
    subparsers.add_parser("inputs-validate", parents=[_inputs_validate_parser()], add_help=False)
    subparsers.add_parser(
        "release-config-create",
        parents=[_release_config_create_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "release-config-validate",
        parents=[_release_config_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "horizon-suite-create",
        parents=[_horizon_suite_create_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "horizon-suite-resume",
        parents=[_horizon_suite_resume_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "horizon-suite-validate",
        parents=[_horizon_suite_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "pilot-calibrate",
        parents=[_pilot_calibrate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "pilot-calibrate-validate",
        parents=[_pilot_calibrate_validate_parser()],
        add_help=False,
    )
    # 長名稱是同一個公開命令的可讀 alias，避免 runbook 必須知道內部短命名。
    subparsers.add_parser(
        "pilot-calibration-build",
        parents=[_pilot_calibrate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "pilot-calibration-validate",
        parents=[_pilot_calibrate_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "pilot-config-create",
        parents=[_pilot_config_create_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "pilot-config-validate",
        parents=[_pilot_config_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "pilot-matrix-validate",
        parents=[_pilot_matrix_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser("behavior-manifest", parents=[_behavior_manifest_parser()], add_help=False)
    subparsers.add_parser("synthetic-smoke", parents=[_synthetic_smoke_parser()], add_help=False)
    subparsers.add_parser("validate-shard", parents=[_validate_shard_parser()], add_help=False)
    subparsers.add_parser("code-provenance", parents=[_code_provenance_parser()], add_help=False)
    subparsers.add_parser("validate-run", parents=[_validate_run_parser()], add_help=False)
    subparsers.add_parser("benchmark-report", parents=[_benchmark_report_parser()], add_help=False)
    subparsers.add_parser("run-create", parents=[_run_create_parser()], add_help=False)
    subparsers.add_parser("run-shard", parents=[_run_shard_parser()], add_help=False)
    subparsers.add_parser("run-worker", parents=[_run_worker_parser()], add_help=False)
    subparsers.add_parser(
        "run-formal-parallel",
        parents=[_run_formal_parallel_parser()],
        add_help=False,
    )
    subparsers.add_parser("run-reconcile", parents=[_run_reconcile_parser()], add_help=False)
    subparsers.add_parser(
        "aggregate-spec-create",
        parents=[_aggregate_spec_create_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "aggregate-build",
        parents=[_aggregate_build_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "aggregate-validate",
        parents=[_aggregate_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "report-spec-create",
        parents=[_report_spec_create_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "report-validate",
        parents=[_report_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "source-pathway-build",
        parents=[_source_pathway_build_parser()],
        add_help=False,
    )
    subparsers.add_parser(
        "source-pathway-validate",
        parents=[_source_pathway_validate_parser()],
        add_help=False,
    )
    subparsers.add_parser("native-preflight", parents=[_native_preflight_parser()], add_help=False)
    subparsers.add_parser("native-gap-fill", parents=[_native_gap_fill_parser()], add_help=False)
    subparsers.add_parser("native-plan", parents=[_native_plan_parser()], add_help=False)
    subparsers.add_parser("native-run", parents=[_native_run_parser()], add_help=False)
    subparsers.add_parser("native-status", parents=[_native_status_parser()], add_help=False)
    subparsers.add_parser("native-postprocess", parents=[_native_postprocess_parser()], add_help=False)
    subparsers.add_parser(
        "native-derive-horizon", parents=[_native_derive_horizon_parser()], add_help=False
    )
    subparsers.add_parser("native-figures", parents=[_native_figures_parser()], add_help=False)
    parsed, remainder = parser.parse_known_args(argv)
    # 重新交給共用 handler 解析完整參數，確保獨立與整合 entry point 行為一致。
    command_argv = list(argv if argv is not None else sys.argv[1:])[1:]
    if parsed.command == "config-check":
        return run_config_check(command_argv)
    if parsed.command == "preflight":
        return run_preflight_command(command_argv)
    if parsed.command == "inputs-build":
        return run_inputs_build(command_argv)
    if parsed.command == "inputs-validate":
        return run_inputs_validate(command_argv)
    if parsed.command == "release-config-create":
        return run_release_config_create(command_argv)
    if parsed.command == "release-config-validate":
        return run_release_config_validate(command_argv)
    if parsed.command == "horizon-suite-create":
        return run_horizon_suite_create(command_argv)
    if parsed.command == "horizon-suite-resume":
        return run_horizon_suite_resume(command_argv)
    if parsed.command == "horizon-suite-validate":
        return run_horizon_suite_validate(command_argv)
    if parsed.command in {"pilot-calibrate", "pilot-calibration-build"}:
        return run_pilot_calibrate(command_argv)
    if parsed.command in {"pilot-calibrate-validate", "pilot-calibration-validate"}:
        return run_pilot_calibrate_validate(command_argv)
    if parsed.command == "pilot-config-create":
        return run_pilot_config_create(command_argv)
    if parsed.command == "pilot-config-validate":
        return run_pilot_config_validate(command_argv)
    if parsed.command == "pilot-matrix-validate":
        return run_pilot_matrix_validate(command_argv)
    if parsed.command == "behavior-manifest":
        return run_behavior_manifest(command_argv)
    if parsed.command == "synthetic-smoke":
        return run_synthetic_smoke(command_argv)
    if parsed.command == "validate-shard":
        return run_validate_shard(command_argv)
    if parsed.command == "code-provenance":
        return run_code_provenance(command_argv)
    if parsed.command == "validate-run":
        return run_validate_run(command_argv)
    if parsed.command == "benchmark-report":
        return run_benchmark_report(command_argv)
    if parsed.command == "run-create":
        return run_create(command_argv)
    if parsed.command == "run-shard":
        return run_shard(command_argv)
    if parsed.command == "run-worker":
        return run_worker(command_argv)
    if parsed.command == "run-formal-parallel":
        return run_formal_parallel(command_argv)
    if parsed.command == "run-reconcile":
        return run_reconcile(command_argv)
    if parsed.command == "aggregate-spec-create":
        return run_aggregate_spec_create(command_argv)
    if parsed.command == "aggregate-build":
        return run_aggregate_build(command_argv)
    if parsed.command == "aggregate-validate":
        return run_aggregate_validate(command_argv)
    if parsed.command == "report-spec-create":
        return run_report_spec_create(command_argv)
    if parsed.command == "report-validate":
        return run_report_validate(command_argv)
    if parsed.command == "source-pathway-build":
        return run_source_pathway_build(command_argv)
    if parsed.command == "source-pathway-validate":
        return run_source_pathway_validate(command_argv)
    if parsed.command == "native-preflight":
        return run_native_preflight_command(command_argv)
    if parsed.command == "native-gap-fill":
        return run_native_gap_fill_command(command_argv)
    if parsed.command == "native-plan":
        return run_native_plan_command(command_argv)
    if parsed.command == "native-run":
        return run_native_run(command_argv)
    if parsed.command == "native-status":
        return run_native_status(command_argv)
    if parsed.command == "native-postprocess":
        return run_native_postprocess_command(command_argv)
    if parsed.command == "native-derive-horizon":
        return run_native_derive_horizon_command(command_argv)
    if parsed.command == "native-figures":
        return run_native_figures_command(command_argv)
    parser.error(f"未知命令：{parsed.command}; 其餘參數={remainder}")
    return 2


def config_check_main() -> None:
    """console script wrapper；以回傳碼結束程序。"""

    raise SystemExit(run_config_check())


def preflight_main() -> None:
    """console script wrapper；以回傳碼結束程序。"""

    raise SystemExit(run_preflight_command())


def inputs_build_main() -> None:
    """inputs-build console script wrapper。"""

    raise SystemExit(run_inputs_build())


def inputs_validate_main() -> None:
    """inputs-validate console script wrapper。"""

    raise SystemExit(run_inputs_validate())


def release_config_create_main() -> None:
    """release-config-create console script wrapper。"""

    raise SystemExit(run_release_config_create())


def release_config_validate_main() -> None:
    """release-config-validate console script wrapper。"""

    raise SystemExit(run_release_config_validate())


def synthetic_smoke_main() -> None:
    """synthetic smoke console script wrapper。"""

    raise SystemExit(run_synthetic_smoke())


def validate_shard_main() -> None:
    """trajectory shard validator console script wrapper。"""

    raise SystemExit(run_validate_shard())


def code_provenance_main() -> None:
    """code-provenance console script wrapper。"""

    raise SystemExit(run_code_provenance())


def validate_run_main() -> None:
    """validate-run console script wrapper。"""

    raise SystemExit(run_validate_run())


def benchmark_report_main() -> None:
    """benchmark-report console script wrapper。"""

    raise SystemExit(run_benchmark_report())


def pilot_matrix_validate_main() -> None:
    """pilot-matrix-validate console script wrapper。"""

    raise SystemExit(run_pilot_matrix_validate())
