"""所有可重現流程的命令列入口。

CLI 只接受明示路徑或環境變數，不把本機／SERVER 絕對路徑寫入原始碼。輸出預設為
JSON，方便 runbook、排程器與後續 manifest 驗證使用；錯誤訊息寫到 stderr 並以非零
狀態退出，避免 shell 批次把未通過 gate 的結果當成功。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from shapely.geometry import box

from .boundaries import BoundaryGeometry
from .config import load_config
from .diffusion import DiffusionCoefficients
from .engine import EngineSettings, run_particle
from .models import ParticleState, VelocitySample
from .outputs import validate_trajectory_shard, write_trajectory_shard
from .preflight import run_preflight
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
    """建立十種已裁決垂向行為 manifest 輸出 parser。"""

    parser = argparse.ArgumentParser(description="輸出 design_baseline_v1 十種垂向行為 manifest")
    parser.add_argument("--output", required=True, type=Path)
    return parser


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


def run_behavior_manifest(argv: Sequence[str] | None = None) -> int:
    """原子寫出十種 behavior records；既有檔案不覆寫。"""

    args = _behavior_manifest_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"不可覆寫既有 manifest：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "design_version": "design_baseline_v1",
        "velocity_unit": "m s-1; z positive-up; negative=sinking; positive=rising",
        "calibration_scope": "behavior_sensitivity_not_named_material_calibration",
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


def main(argv: Sequence[str] | None = None) -> int:
    """整合 ``lbt`` 子命令；未知命令由 argparse 以狀態 2 拒絕。"""

    parser = argparse.ArgumentParser(prog="lbt", description="Lagrangian 系集逆向溯源可重現 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("config-check", parents=[_config_check_parser()], add_help=False)
    subparsers.add_parser("preflight", parents=[_preflight_parser()], add_help=False)
    subparsers.add_parser("behavior-manifest", parents=[_behavior_manifest_parser()], add_help=False)
    subparsers.add_parser("synthetic-smoke", parents=[_synthetic_smoke_parser()], add_help=False)
    subparsers.add_parser("validate-shard", parents=[_validate_shard_parser()], add_help=False)
    parsed, remainder = parser.parse_known_args(argv)
    # 重新交給共用 handler 解析完整參數，確保獨立與整合 entry point 行為一致。
    command_argv = list(argv if argv is not None else sys.argv[1:])[1:]
    if parsed.command == "config-check":
        return run_config_check(command_argv)
    if parsed.command == "preflight":
        return run_preflight_command(command_argv)
    if parsed.command == "behavior-manifest":
        return run_behavior_manifest(command_argv)
    if parsed.command == "synthetic-smoke":
        return run_synthetic_smoke(command_argv)
    if parsed.command == "validate-shard":
        return run_validate_shard(command_argv)
    parser.error(f"未知命令：{parsed.command}; 其餘參數={remainder}")
    return 2


def config_check_main() -> None:
    """console script wrapper；以回傳碼結束程序。"""

    raise SystemExit(run_config_check())


def preflight_main() -> None:
    """console script wrapper；以回傳碼結束程序。"""

    raise SystemExit(run_preflight_command())


def synthetic_smoke_main() -> None:
    """synthetic smoke console script wrapper。"""

    raise SystemExit(run_synthetic_smoke())


def validate_shard_main() -> None:
    """trajectory shard validator console script wrapper。"""

    raise SystemExit(run_validate_shard())
