"""類別顯示名稱對照表的測試：只改圖面文字，不得影響任何識別值或進入雜湊的欄位。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from lagrangian_backtracking.native_runner.display_names import (
    MATERIAL_DISPLAY_NAME_ZH,
    material_display_name_zh,
)
from lagrangian_backtracking.scenarios import BASELINE_BEHAVIORS

# 官方分類名稱（材質清冊與設定檔記錄，進入材質清冊的檢核碼）。這份字面值就是「不得改動」的證據。
_OFFICIAL_CATEGORY_BY_MATERIAL_ID = {
    "oca_styrofoam_porous_fragment": "保麗龍",
    "oca_wood_waterlogged_elongated": "竹木",
    "oca_wastepaper_folded_fiber": "廢紙",
    "oca_nonrecyclable_flexible_sheet": "其他/不可回收",
    "oca_fishinggear_open_mesh_bundle": "廢漁網漁具",
    "oca_pet_waterfilled_bottle": "寶特瓶",
    "oca_other_recyclable_irregular_fragment": "其他/可回收",
    "oca_aluminum_crushed_cylinder": "鋁罐",
    "oca_steel_rigid_cylinder": "鐵罐",
    "oca_glass_bottle_or_fragment": "玻璃瓶",
}


def test_display_map_covers_exactly_the_ten_material_ids():
    assert set(MATERIAL_DISPLAY_NAME_ZH) == {b.material_id for b in BASELINE_BEHAVIORS}
    assert len(MATERIAL_DISPLAY_NAME_ZH) == 10


def test_only_two_classes_are_renamed_and_the_rest_keep_the_official_name():
    changed = {
        mid: name
        for mid, name in MATERIAL_DISPLAY_NAME_ZH.items()
        if name != _OFFICIAL_CATEGORY_BY_MATERIAL_ID[mid]
    }
    assert changed == {
        "oca_nonrecyclable_flexible_sheet": "不可回收塑膠",
        "oca_other_recyclable_irregular_fragment": "可回收塑膠",
    }


def test_identity_fields_in_behavior_records_are_unchanged_by_display_names():
    # 呼叫顯示名稱對照之後，材質清冊用的記錄（含官方分類名稱與材質識別碼）逐字不變。
    before = json.dumps([asdict(b) for b in BASELINE_BEHAVIORS], ensure_ascii=False, sort_keys=True)
    for b in BASELINE_BEHAVIORS:
        material_display_name_zh(b.material_id)
    after = json.dumps([asdict(b) for b in BASELINE_BEHAVIORS], ensure_ascii=False, sort_keys=True)
    assert hashlib.sha256(before.encode()).hexdigest() == hashlib.sha256(after.encode()).hexdigest()
    assert {b.material_id: b.oca_category_zh for b in BASELINE_BEHAVIORS} == _OFFICIAL_CATEGORY_BY_MATERIAL_ID


def test_unknown_material_id_is_returned_unchanged():
    assert material_display_name_zh("not_a_registered_material") == "not_a_registered_material"
