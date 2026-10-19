#!/usr/bin/env python3
"""run09 資料根檢查（唯讀、fail closed）。

用法：
  check_run09_data_roots.py roots --runtime-ocm DIR --runtime-nww DIR --aprime-ocm DIR --aprime-surface DIR
                                  [--first-month 202404] [--last-month 202512] [--aprime-first-month 202410]
  check_run09_data_roots.py population --arrival ARRIVAL_JSON --horizon-days 180 --max-age-days 90
                                       --runtime-ocm DIR --runtime-nww DIR --aprime-ocm DIR [--floor 2024-04-01]

roots 階段：
  * 只允許下列資料根（路徑須含其中之一）：lbt-local-data-v6、-v6-full、-v7-full、-v7-surface-fix202511-12、NAS；
  * 執行期 OCM（v7-full）四個流場 × [first, last] 月份：time/hvel/vertical_velocity/zcor/elev/wetdry_elem 必須存在且是實檔；
  * 執行期 NWW：四個流場 × 月份：目錄內全部 .npy 必須是實檔；
  * 母體建置 OCM（v6-full）：標頭（metadata.json、time_utc_ns.npy）一律須存在；zcor/elev/wetdry_elem 從 aprime-first-month 起必須是實檔
    （母體建置只在沉底整點讀這三個陣列的數值；更早月份的稀疏占位檔只讀標頭，且建置程式另有 fail-closed 守門）。
population 階段：到達時刻 JSON 的最早觀測時刻 - (H + 最長年齡) 與最早沉底時刻 - H 都不得早於 floor，
  並確認這些時刻涵蓋的月份在執行期資料根都是實檔。
稀疏占位檔判定：檔案 >= 1 MiB 且 st_blocks*512 < 5% 的檔案大小。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ALLOWED = ("lbt-local-data-v6", "lbt-local-data-v6-full", "lbt-local-data-v7-full",
           "lbt-local-data-v7-surface-fix202511-12", "NWW-Preprocessed-Data")
DOMAINS = ("northeast_taiwan_common_cache_v3", "hsinchu_cache_v3", "houwan_nmmba_cache_v3", "lienchiang_common_cache_v3")
OCM_RUNTIME_ARRAYS = ("time_utc_ns.npy", "hvel.npy", "vertical_velocity.npy", "zcor.npy", "elev.npy", "wetdry_elem.npy")
OCM_VALUE_ARRAYS = ("zcor.npy", "elev.npy", "wetdry_elem.npy")
MIN_BYTES = 1 << 20
FRACTION = 0.05


def months(first: str, last: str) -> list[str]:
    y, m = int(first[:4]), int(first[4:])
    out = []
    while f"{y}{m:02d}" <= last:
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def sparse(path: Path) -> bool:
    st = path.stat()
    return st.st_size >= MIN_BYTES and st.st_blocks * 512 < FRACTION * st.st_size


def check_allowed(path: Path, errors: list[str]) -> None:
    real = str(path.resolve())
    parts = real.replace("\\", "/").split("/")
    if not any(a in parts for a in ALLOWED):
        errors.append(f"資料根 {real} 不在允許清單 {ALLOWED}")


def check_arrays(base: Path, names, errors: list[str], label: str) -> None:
    for name in names:
        f = base / name
        if not f.is_file():
            errors.append(f"{label}：缺少 {f}")
        elif sparse(f):
            errors.append(f"{label}：{f} 是稀疏占位檔（st_blocks 遠小於檔案大小）")


def stage_roots(a) -> list[str]:
    errors: list[str] = []
    for root in (a.runtime_ocm, a.runtime_nww, a.aprime_ocm, a.aprime_surface):
        if not Path(root).is_dir():
            errors.append(f"找不到資料根 {root}")
            continue
        check_allowed(Path(root), errors)
    if errors:
        return errors
    for d in DOMAINS:
        grid = Path(a.runtime_ocm) / d / "grid"
        if not (grid / "source_depth_m.npy").is_file():
            errors.append(f"執行期 OCM grid 缺少 {grid}/source_depth_m.npy")
        for mon in ([] if a.aprime_only else months(a.first_month, a.last_month)):
            mdir = Path(a.runtime_ocm) / d / "months" / mon
            if not mdir.is_dir():
                errors.append(f"執行期 OCM 缺少月份目錄 {mdir}")
                continue
            check_arrays(mdir, OCM_RUNTIME_ARRAYS, errors, f"執行期 OCM {d}/{mon}")
            nmdir = Path(a.runtime_nww) / d / "months" / mon
            if not nmdir.is_dir():
                errors.append(f"執行期 NWW 缺少月份目錄 {nmdir}")
            else:
                npys = sorted(p.name for p in nmdir.glob("*.npy"))
                if not npys:
                    errors.append(f"執行期 NWW {nmdir} 沒有任何 .npy")
                check_arrays(nmdir, npys, errors, f"執行期 NWW {d}/{mon}")
        # 母體建置用 OCM：標頭一律檢查，數值陣列從 aprime_first_month 起檢查
        for mon in months("202401", a.last_month):
            mdir = Path(a.aprime_ocm) / d / "months" / mon
            if not mdir.is_dir():
                errors.append(f"母體建置 OCM 缺少月份目錄 {mdir}")
                continue
            check_arrays(mdir, ("metadata.json", "time_utc_ns.npy"), errors, f"母體建置 OCM 標頭 {d}/{mon}")
            if mon >= a.aprime_first_month:
                check_arrays(mdir, OCM_VALUE_ARRAYS, errors, f"母體建置 OCM 數值 {d}/{mon}")
    return errors


def stage_population(a) -> list[str]:
    errors: list[str] = []
    doc = json.load(open(a.arrival, encoding="utf-8"))
    dep = [int(r["time_utc_ns"]) for r in doc["records"]]
    obs = [int(r["metadata"]["observation_time_utc_ns"]) for r in doc["records"]]
    floor = datetime.fromisoformat(a.floor).replace(tzinfo=UTC)
    to_dt = lambda ns: datetime.fromtimestamp(ns / 1e9, tz=UTC)
    first_obs, first_dep = to_dt(min(obs)), to_dt(min(dep))
    sel = first_obs - timedelta(days=a.horizon_days + a.max_age_days)
    exe = first_dep - timedelta(days=a.horizon_days)
    print(f"最早觀測時刻 {first_obs:%F %T}Z；減 {a.horizon_days + a.max_age_days} 日（選取窗）= {sel:%F %T}Z")
    print(f"最早沉底時刻 {first_dep:%F %T}Z；減 {a.horizon_days} 日（執行窗）= {exe:%F %T}Z")
    if sel < floor:
        errors.append(f"選取窗起點 {sel:%F} 早於 {a.floor}")
    if exe < floor:
        errors.append(f"執行窗起點 {exe:%F} 早於 {a.floor}")
    last_dep = to_dt(max(dep))
    need = months(f"{exe.year}{exe.month:02d}", f"{last_dep.year}{last_dep.month:02d}")
    print("執行期需要的月份：", need[0], "～", need[-1], f"（{len(need)} 個）")
    for d in DOMAINS:
        for mon in need:
            mdir = Path(a.runtime_ocm) / d / "months" / mon
            if not mdir.is_dir():
                errors.append(f"執行期 OCM 缺少 {mdir}")
            else:
                check_arrays(mdir, OCM_RUNTIME_ARRAYS, errors, f"執行期 OCM {d}/{mon}")
            nmdir = Path(a.runtime_nww) / d / "months" / mon
            if not nmdir.is_dir():
                errors.append(f"執行期 NWW 缺少 {nmdir}")
    dep_months = sorted({f"{to_dt(t).year}{to_dt(t).month:02d}" for t in dep})
    print("沉底整點所在月份（母體建置讀數值的月份）：", dep_months[0], "～", dep_months[-1])
    for d in DOMAINS:
        for mon in dep_months:
            check_arrays(Path(a.aprime_ocm) / d / "months" / mon, OCM_VALUE_ARRAYS, errors, f"母體建置 OCM 數值 {d}/{mon}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("roots", "population"):
        p = sub.add_parser(name)
        p.add_argument("--runtime-ocm", required=True)
        p.add_argument("--runtime-nww", required=True)
        p.add_argument("--aprime-ocm", required=True)
        if name == "roots":
            p.add_argument("--aprime-surface", required=True)
            p.add_argument("--first-month", default="202404")
            p.add_argument("--last-month", default="202512")
            p.add_argument("--aprime-first-month", default="202410")
            p.add_argument("--aprime-only", action="store_true", help="母體建置前只檢查 APRIME 根與執行期根的存在／grid（執行期月份尚在搬運時使用）")
        else:
            p.add_argument("--arrival", required=True)
            p.add_argument("--horizon-days", type=int, default=180)
            p.add_argument("--max-age-days", type=int, default=90)
            p.add_argument("--floor", default="2024-04-01")
    a = ap.parse_args()
    errors = stage_roots(a) if a.stage == "roots" else stage_population(a)
    if errors:
        shown = errors[:40]
        print(f"資料根檢查失敗（{len(errors)} 項）：", file=sys.stderr)
        for e in shown:
            print("  -", e, file=sys.stderr)
        if len(errors) > len(shown):
            print(f"  ...另有 {len(errors) - len(shown)} 項", file=sys.stderr)
        return 1
    print(f"資料根檢查通過（階段 {a.stage}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
