#!/usr/bin/env python3
"""由 v7（run08）範本產生 run09（v8：回溯 180 日）設定範本。只做文字替換，其餘逐字複製 v7 範本。

用法：make_template_v8.py [--check] [--v7-template PATH] [--output PATH]
                           [--nanwan-anchor LON LAT] [--no-bad-hour-rule] [--bad-hours-file PATH]

替換項（每一項在 v7 範本中都必須恰好出現一次，否則中止）：
  1. inputs.backtrack_support_days 180 -> 270（H180 + 最長沉底年齡 90 日）；
  2. arrival_time_selection：northeast_pair_utc_when_coverage_allows true -> false（龜山島依自己的潮位選到達時刻），
     並新增 seam_jump_exclusion: {}（OCM 日檔接縫跳動前後 ±1 h 不作為到達時刻候選；預設值即 WP-61 V1 定義）；
  3. scenarios.horizontal_receptor_selection_policy -> 方位×距離等面積拉丁方格分層抽樣；
     新增 scenarios.horizontal_receptor_sampling（半徑 10 km；龜山島西半 3、東半 2 個方位區間）；
  4. 南灣錨點 -> --nanwan-anchor（預設 120.770675 21.92041，自原錨點方位 165° 外推 3 km），並標示 anchor_source: config_override_v1；
  5. scenarios.bed_residence_time.deposition_bad_hour_rule（牽連面積 >= 1 km2；預設啟用，--no-bad-hour-rule 關閉）
     或 --bad-hours-file（外部清單，二擇一）。
新竹錨點 (120.45, 24.75)、龜山島錨點 (121.939148, 24.850326)、貢寮、連江一律不動。
安全：輸出檔已存在就中止；--check 只驗證不寫檔。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

W = Path.home() / "Workspace"
DEFAULT_V7 = W / "lbt-inputs/formal/template-v7-class-velocity.yaml"
DEFAULT_OUT = W / "lbt-inputs/formal/template-v8-run09.yaml"
LATIN_POLICY = "azimuth_distance_equal_area_latin_stratified_v1"
OLD_NANWAN = "    anchor_lonlat: [120.763161, 21.946577]\n"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"中止：v7 範本中找不到唯一的「{label}」（出現 {text.count(old)} 次）")
    return text.replace(old, new, 1)


def build(text: str, *, nanwan: tuple[float, float], bad_hour_rule: bool, bad_hours_file: str | None) -> str:
    text = replace_once(text, "  backtrack_support_days: 180\n",
                        "  # run09：H180 加最長 90 日沉底年齡 = 270 日 observation 選時包絡（2025-01-01 往前 270 日為 2024-04-06）。\n"
                        "  backtrack_support_days: 270\n", "backtrack_support_days")
    text = replace_once(text, "  northeast_pair_utc_when_coverage_allows: true\n",
                        "  # run09：貢寮與龜山島各依自己的潮位獨立選到達時刻（不再配對）。\n"
                        "  northeast_pair_utc_when_coverage_allows: false\n"
                        "  # run09：OCM 日檔接縫跳動（WP-61 V1 準則，五站聯集）前後 1 小時不作為到達時刻候選；清單與 SHA-256 寫入 manifest。\n"
                        "  seam_jump_exclusion: {}\n", "northeast_pair_utc_when_coverage_allows")
    text = replace_once(
        text,
        "  horizontal_receptor_selection_policy: seeded_uniform_random_without_replacement_v1\n",
        f"  horizontal_receptor_selection_policy: {LATIN_POLICY}\n"
        "  # run09：以錨點為圓心 10 km 內，方位 × 距離等面積拉丁方格分層隨機抽樣；龜山島西半（180°–360°）3 個、東半（0°–180°）2 個方位區間。\n"
        "  horizontal_receptor_sampling:\n"
        "    radius_m: 10000\n"
        "    strata_count: 5\n"
        "    azimuth_allocation_by_site:\n"
        "      guishan:\n"
        "        west_180_360: 3\n"
        "        east_0_180: 2\n",
        "horizontal_receptor_selection_policy")
    lon, lat = nanwan
    text = replace_once(text, OLD_NANWAN,
                        f"    # run09：南灣錨點自 (120.763161, 21.946577) 往方位 165° 外推 3 km；由設定檔明示覆寫（anchor_source）。\n"
                        f"    anchor_lonlat: [{lon}, {lat}]\n    anchor_source: config_override_v1\n", "南灣錨點")
    anchor_line = "    runtime_horizon_support_days: null\n"
    if bad_hours_file:
        text = replace_once(text, anchor_line, anchor_line + f"    deposition_exclusion_hours_file: {bad_hours_file}\n", "runtime_horizon_support_days")
    elif bad_hour_rule:
        text = replace_once(
            text, anchor_line,
            anchor_line
            + "    # run09：壞時刻規則（WP-62 診斷）：錨點 10 km 內只失敗 1–5 個整點的面，在某整點合計牽連面積 >= 1 km2 者，\n"
            "    # 該整點從沉底可用整點排除；由失敗矩陣確定性導出，清單與 SHA-256 寫入 manifest。\n"
            "    deposition_bad_hour_rule:\n"
            "      radius_m: 10000\n"
            "      min_failed_hours: 1\n"
            "      max_failed_hours: 5\n"
            "      area_threshold_km2: 1.0\n",
            "runtime_horizon_support_days")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--v7-template", type=Path, default=DEFAULT_V7)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--nanwan-anchor", nargs=2, type=float, default=(120.770675, 21.92041), metavar=("LON", "LAT"))
    ap.add_argument("--no-bad-hour-rule", action="store_true")
    ap.add_argument("--bad-hours-file", default=None)
    a = ap.parse_args()
    v7 = a.v7_template.read_text(encoding="utf-8")
    out = build(v7, nanwan=tuple(a.nanwan_anchor), bad_hour_rule=not a.no_bad_hour_rule, bad_hours_file=a.bad_hours_file)
    print(f"v7 範本 SHA-256：{hashlib.sha256(v7.encode('utf-8')).hexdigest()}")
    print(f"v8 範本 SHA-256：{hashlib.sha256(out.encode('utf-8')).hexdigest()}（{len(out.splitlines())} 行）")
    if a.check:
        print("檢查模式：未寫檔")
        return 0
    if a.output.exists():
        print(f"中止：{a.output} 已存在，為避免覆蓋而不寫入", file=sys.stderr)
        return 2
    a.output.write_text(out, encoding="utf-8")
    print(f"已寫入 {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
