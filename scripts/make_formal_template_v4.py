#!/usr/bin/env python
"""WP-51 整合：由伺服器上的 v3 正式範本產生 v4（逐成員沉底年齡）正式範本。

v4 範本以 ``template-v3-25h-newsettling.yaml`` 為基礎，只做下列逐項、可斷言的文字改動，
其餘每個位元組都沿用 v3（因此 arrival policy v2、25 小時潮汐強度時窗、M=10、十檔沉降
速度、受體與邊界設定都不變）：

1. ``scenarios.bed_residence_time`` 新增 ``age_sampling_scope: per_member``
   （每個成員各自抽沉底年齡，50×M 個槽位的分層排列；見 WP-51 報告）；
2. 龜山島錨點換成「距校閱海岸陸地至少 500 m」的新錨點，並帶完整 provenance
   （區塊文字取自 repo 內的 ``configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml``）；
3. ``physics.vertical_diffusion.constant_kz_m2ps`` 0.0013526236792521886 -> 0.0003：
   WP-51 整合後設定檔是 Kz 的唯一來源，v3 範本的 0.00135 從未生效（v3 正式重算實際用的是
   3.0e-4），為使 v4 的實際物理與 v3 相同，且「設定＝實際」，必須把設定改成 3.0e-4；
4. ``integration.output_interval_seconds`` 300 -> 3600、``maximum_horizontal_cfl_fraction``
   0.25 -> 0.5、其餘兩個 native 不讀的 fraction 欄位 -> null：只是把設定改成與 native 實際行為
   一致（這些欄位 native 不讀，不影響結果；註解文字取自 repo 的 formal YAML）。

用法::

    python scripts/make_formal_template_v4.py \\
        --base ~/Workspace/lbt-inputs/formal/template-v3-25h-newsettling.yaml \\
        --repo-formal-config configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml \\
        --output ~/Workspace/lbt-inputs/formal/template-v4-per-member.yaml

輸出已存在時拒絕覆寫。產生後會用 ``load_config`` 載入驗證，並以結構化比對印出 v3 與 v4 的
全部差異路徑；差異超出預期（上列四類）即中止並刪除輸出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# (舊文字, 新文字) 皆須在 v3 範本恰好出現一次。
_ANCHOR_V3 = "    anchor_lonlat: [121.951606, 24.843127]\n"
_AGE_SCOPE_AFTER = "    shared_age_offsets_across_sites: true\n"
_AGE_SCOPE_BLOCK = (
    "    # WP-51：逐成員沉底年齡（50 個觀測 × M 個成員 = 50M 個槽位，分層後隨機排列配對，\n"
    "    # 五站共用；沉底整點須落在各站 canonical available-time set）。見 reports/WP-51*.md。\n"
    "    age_sampling_scope: per_member\n"
)
_KZ_V3 = "    constant_kz_m2ps: 0.0013526236792521886\n"
_OUTPUT_INTERVAL_V3 = "  output_interval_seconds: 300\n"
_CFL_V3 = (
    "  maximum_horizontal_cfl_fraction: 0.25\n"
    "  maximum_vertical_layer_fraction: 0.25\n"
    "  maximum_diffusive_rms_fraction: 0.25\n"
)

# 自 repo formal YAML 取用的文字區段（起訖標記）。
_GUISHAN_START = "    # WP-51 C：錨點由陸地移到海上。"
_GUISHAN_END = "    receptor_core_radius_m: 12500\n"
_KZ_START = "    # WP-51 整合：native-run 實際使用的常數垂向擴散係數"
_KZ_END = "    constant_kz_m2ps: 0.0003\n"
_INTEGRATION_START = "  # WP-51：本區數值必須等於 native_runner 實際行為"
_OUTPUT_INTERVAL_V4 = "  output_interval_seconds: 3600\n"
_CFL_END = "  maximum_diffusive_rms_fraction: null\n"

# 預期的結構化差異路徑（其餘路徑 v3 與 v4 必須完全相同）。
_EXPECTED_DIFF_PREFIXES = (
    "scenarios.bed_residence_time.age_sampling_scope",
    "study_sites[guishan].anchor_lonlat",
    "study_sites[guishan].anchor_policy",
    "study_sites[guishan].anchor_snap_provenance",
    "physics.vertical_diffusion.constant_kz_m2ps",
    "integration.output_interval_seconds",
    "integration.maximum_horizontal_cfl_fraction",
    "integration.maximum_vertical_layer_fraction",
    "integration.maximum_diffusive_rms_fraction",
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"v3 範本中「{label}」應恰好出現一次，實際 {count} 次；範本可能已改版，請人工檢查")
    return text.replace(old, new)


def _span(text: str, start: str, end: str, label: str, *, include_end: bool) -> str:
    begin = text.find(start)
    if begin < 0 or text.count(start) != 1:
        raise SystemExit(f"repo formal YAML 找不到唯一的起點標記：{label}")
    stop = text.find(end, begin)
    if stop < 0:
        raise SystemExit(f"repo formal YAML 找不到終點標記：{label}")
    if include_end:
        stop += len(end)
    return text[begin:stop]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        if value and all(isinstance(item, dict) and "study_site_id" in item for item in value):
            for item in value:
                result.update(_flatten(item, f"{prefix}[{item['study_site_id']}]"))
        else:
            result[prefix] = value
    else:
        result[prefix] = value
    return result


def generate(base_text: str, repo_formal_text: str) -> str:
    guishan_block = _span(repo_formal_text, _GUISHAN_START, _GUISHAN_END, "龜山島錨點區塊", include_end=False)
    kz_block = _span(repo_formal_text, _KZ_START, _KZ_END, "Kz 註解與值", include_end=True)
    integration_block = _span(
        repo_formal_text, _INTEGRATION_START, _CFL_END, "integration 區塊", include_end=True
    )
    # integration 區塊在 repo 內是 output_interval 3600 + cfl 說明 + 三個 fraction 欄位連續出現。
    if _OUTPUT_INTERVAL_V4 not in integration_block:
        raise SystemExit("repo formal YAML 的 integration 區塊不含 output_interval_seconds: 3600")

    text = base_text
    text = _replace_once(text, _ANCHOR_V3, guishan_block, "龜山島 anchor_lonlat")
    text = _replace_once(text, _AGE_SCOPE_AFTER, _AGE_SCOPE_AFTER + _AGE_SCOPE_BLOCK, "shared_age_offsets_across_sites")
    text = _replace_once(text, _KZ_V3, kz_block, "constant_kz_m2ps")
    # output_interval 與三個 fraction 在 v3 範本並不相鄰，分別替換成 repo 區塊中對應的行。
    lines = integration_block.splitlines(keepends=True)
    interval_start = next(i for i, line in enumerate(lines) if line == _OUTPUT_INTERVAL_V4)
    cfl_start = next(i for i, line in enumerate(lines) if line.startswith("  # 實際子步上限"))
    interval_text = "".join(lines[: interval_start + 1])
    cfl_text = "".join(lines[cfl_start:])
    text = _replace_once(text, _OUTPUT_INTERVAL_V3, interval_text, "output_interval_seconds")
    text = _replace_once(text, _CFL_V3, cfl_text, "maximum_*_fraction")
    return text


def _check_diff(base_text: str, new_text: str) -> list[str]:
    old = _flatten(yaml.safe_load(base_text))
    new = _flatten(yaml.safe_load(new_text))
    differing = sorted(key for key in set(old) | set(new) if old.get(key, "<缺>") != new.get(key, "<缺>"))
    unexpected = [key for key in differing if not key.startswith(_EXPECTED_DIFF_PREFIXES)]
    if unexpected:
        raise SystemExit("v3 與 v4 出現預期之外的差異：" + ", ".join(unexpected))
    return differing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="由 v3 正式範本產生 v4（per_member）正式範本")
    parser.add_argument("--base", type=Path, required=True, help="v3 範本（唯讀）")
    parser.add_argument("--repo-formal-config", type=Path, required=True, help="repo 內的正式設定 YAML")
    parser.add_argument("--output", type=Path, required=True, help="輸出路徑（必須尚不存在）")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"拒絕覆寫既有檔案：{args.output}")

    base_text = args.base.read_text(encoding="utf-8")
    new_text = generate(base_text, args.repo_formal_config.read_text(encoding="utf-8"))
    differing = _check_diff(base_text, new_text)

    payload = yaml.safe_load(new_text)
    bed = payload["scenarios"]["bed_residence_time"]
    assert bed["age_sampling_scope"] == "per_member"
    assert payload["scenarios"]["members_per_scenario"] == 10
    assert payload["physics"]["vertical_diffusion"]["constant_kz_m2ps"] == 0.0003

    args.output.write_text(new_text, encoding="utf-8")
    try:
        from lagrangian_backtracking.config import load_config

        config = load_config(args.output)
        guishan = next(site for site in config.study_sites if site.study_site_id == "guishan")
        assert tuple(guishan.anchor_lonlat) == (121.939148, 24.850326)
        assert config.scenarios.bed_residence_time.age_sampling_scope == "per_member"
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"已寫入 {args.output}；v3 與 v4 的結構差異路徑共 {len(differing)} 項：")
    for key in differing:
        print("  ", key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
