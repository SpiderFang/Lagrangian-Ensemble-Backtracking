"""所有可重現流程的命令列入口。

CLI 只接受明示路徑或環境變數，不把本機／SERVER 絕對路徑寫入原始碼或 run plan。
輸出預設為 JSON，方便 runbook、排程器與後續 manifest 驗證使用；錯誤訊息寫到 stderr
並以非零狀態退出，避免 shell 批次把未通過 gate 的結果當成功。run-create 只
建立 pilot 或 formal workspace 並立即做只讀驗證；run-shard 先驗證 workspace，再依
immutable plan 選擇 runtime mode 與 forcing root；run-reconcile 僅採認 checkpoint／progress
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
import json
import math
import os
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import numpy as np
from shapely.geometry import box

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
from .input_derivation import (
    build_input_derivatives,
    create_release_config,
    validate_input_derivatives,
    validate_release_config,
)
from .models import ParticleState, VelocitySample
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
    parser.add_argument("--nww-analysis-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sweep-budget", type=_positive_cli_int)
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
    result = build_input_derivatives(
        config_path=args.config,
        destination=args.destination,
        ocm_native_root=args.ocm_native_root,
        ocm_surface_root=args.ocm_surface_root,
        nww_analysis_root=args.nww_analysis_root,
        formal=args.formal,
        # CLI 是 production pipeline 的入口；即使是非正式 pilot，也不得以合成常值場補資料。
        strict=True,
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
    result = create_release_config(
        config_template_path=args.config_template,
        input_directory=args.input_directory,
        output_path=args.output,
        formal=args.formal,
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
        "design_version": "design_baseline_v2_non_rising_oca_proxy",
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
    """

    args = _run_create_parser().parse_args(argv)
    if args.run_kind == "formal" and args.pilot_scenarios_per_stratum is not None:
        raise ValueError("formal run 禁止 --pilot-scenarios-per-stratum")
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
    nww_root = _optional_root_from_env(
        args.nww_analysis_root,
        input_config.nww_analysis_root_env,
    )
    controller = open_run_controller(
        args.workspace,
        config_path=args.config,
        ocm_native_root=ocm_root,
        nww_analysis_root=nww_root,
        resume=args.resume,
        checkpoint_root=args.checkpoint_root,
    )
    summary = controller.run_shard(args.shard_id, sweep_budget=args.sweep_budget)
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    return 0


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

    def reject_request_factory(unit: object) -> NoReturn:
        """reconcile 不得建立粒子 request，因為狀態採認不應做物理計算。"""

        del unit
        raise RuntimeError("run-reconcile 不得呼叫 request_factory；reconcile 不做物理計算")

    controller = RunController(
        args.workspace,
        request_factory=reject_request_factory,
        checkpoint_root=args.checkpoint_root,
    )
    reconciled = controller.reconcile()
    if not isinstance(reconciled, Mapping):
        raise ValueError("reconcile 回傳值必須是 progress mapping")
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
    subparsers.add_parser("behavior-manifest", parents=[_behavior_manifest_parser()], add_help=False)
    subparsers.add_parser("synthetic-smoke", parents=[_synthetic_smoke_parser()], add_help=False)
    subparsers.add_parser("validate-shard", parents=[_validate_shard_parser()], add_help=False)
    subparsers.add_parser("code-provenance", parents=[_code_provenance_parser()], add_help=False)
    subparsers.add_parser("validate-run", parents=[_validate_run_parser()], add_help=False)
    subparsers.add_parser("benchmark-report", parents=[_benchmark_report_parser()], add_help=False)
    subparsers.add_parser("run-create", parents=[_run_create_parser()], add_help=False)
    subparsers.add_parser("run-shard", parents=[_run_shard_parser()], add_help=False)
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
    if parsed.command in {"pilot-calibrate", "pilot-calibration-build"}:
        return run_pilot_calibrate(command_argv)
    if parsed.command in {"pilot-calibrate-validate", "pilot-calibration-validate"}:
        return run_pilot_calibrate_validate(command_argv)
    if parsed.command == "pilot-config-create":
        return run_pilot_config_create(command_argv)
    if parsed.command == "pilot-config-validate":
        return run_pilot_config_validate(command_argv)
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
