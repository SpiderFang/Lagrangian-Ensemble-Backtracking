"""圖面與圖說用的「類別顯示名稱」對照表（只管呈現，不進入任何身分或雜湊）。

為什麼獨立成一個模組
--------------------
十個海廢類別在資料契約裡的識別方式有兩層：

1. ``material_id``（例如 ``oca_nonrecyclable_flexible_sheet``）：粒子身分表、後處理表與
   所有 manifest 用來辨識類別的唯一鍵，進入隨機種子命名空間、情境識別碼與輸出檢核碼。
2. ``Behavior.oca_category_zh``（例如 ``其他/不可回收``）：材質清冊（material manifest）
   與設定檔記錄的「iOcean 官方分類名稱」。它是 ``scenarios.BASELINE_BEHAVIORS`` 的欄位，
   會隨 ``asdict`` 寫進材質清冊並一起計入該檔的 SHA-256；``config.EXPECTED_OCA_CATEGORIES_ZH``
   與 ``manifests`` 也會逐字檢核它。**因此官方分類名稱不得為了圖面好讀而改動**，否則會讓
   材質清冊的雜湊與輸入母體不一致。

圖面上「其他／不可回收」與「其他／可回收」兩個名稱過於籠統，容易被誤讀為「其餘雜項」；
實際上兩類分別代表「不可回收的塑膠類薄膜、軟片」與「可回收的塑膠類不規則片塊」。本模組
只在**畫圖與寫圖說的當下**，以 ``material_id`` 查表取得顯示名稱，原始欄位一律不動。

規則
----
* 鍵一律是 ``material_id``；查不到時回傳 ``material_id`` 本身（不 fail-closed，與
  ``figures.SITE_LABELS_ZH`` 同樣屬於純呈現用的別名表，不是資料契約）。
* 只有下列兩類的顯示名稱與官方分類名稱不同，其餘八類逐字沿用官方分類名稱。
* 本模組不 import ``scenarios``，避免呈現層反向依賴契約層；一致性由單元測試
  （``tests/test_native_display_names.py``）核對。
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "MATERIAL_DISPLAY_NAME_ZH",
    "material_display_name_zh",
]

MATERIAL_DISPLAY_NAME_ZH: Mapping[str, str] = {
    "oca_styrofoam_porous_fragment": "保麗龍",
    "oca_wood_waterlogged_elongated": "竹木",
    "oca_wastepaper_folded_fiber": "廢紙",
    # 官方分類名稱為「其他/不可回收」；圖面改稱「不可回收塑膠」。
    "oca_nonrecyclable_flexible_sheet": "不可回收塑膠",
    "oca_fishinggear_open_mesh_bundle": "廢漁網漁具",
    "oca_pet_waterfilled_bottle": "寶特瓶",
    # 官方分類名稱為「其他/可回收」；圖面改稱「可回收塑膠」。
    "oca_other_recyclable_irregular_fragment": "可回收塑膠",
    "oca_aluminum_crushed_cylinder": "鋁罐",
    "oca_steel_rigid_cylinder": "鐵罐",
    "oca_glass_bottle_or_fragment": "玻璃瓶",
}
"""``material_id`` → 圖面／圖說顯示名稱（繁體中文）。只供呈現，不可寫回任何資料契約欄位。"""


def material_display_name_zh(material_id: str) -> str:
    """回傳類別的圖面顯示名稱；未登錄的 ``material_id`` 原樣回傳（純呈現用，不 fail-closed）。"""

    return MATERIAL_DISPLAY_NAME_ZH.get(material_id, material_id)
