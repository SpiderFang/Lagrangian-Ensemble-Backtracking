#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derive.py -- 十類海廢物件沉降速度理論推導（可獨立重算與稽核）

方法：
    力平衡：V g (rho_p - rho_w) = 0.5 rho_w Cd w^2 A_proj
    阻力係數（Schiller-Naumann 形狀項 + 形狀相關高 Re 極限項 cd_inf）：
        Cd(Re) = 24/Re * (1 + 0.15 Re^0.687) + cd_inf
    以固定點迭代求解終端沉降速度 w。

僅依賴標準函式庫；若環境有 numpy 則額外用於彙總統計（幾何平均等），
沒有 numpy 時自動退回純 Python 實作，行為完全等價。

執行方式：
    python3 derive.py
會印出求解器驗證、逐情境結果、十類彙總表，並寫出 results.csv。
"""

import csv
import math
import os
import sys

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - 環境無 numpy 時的退回路徑
    np = None
    HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
RHO_W = 1025.0   # kg/m^3, 海水密度
NU = 1.05e-6     # m^2/s, 海水運動黏滯係數
G = 9.81         # m/s^2, 重力加速度


# ---------------------------------------------------------------------------
# 核心函式
# ---------------------------------------------------------------------------
def cd_shape(Re, cd_inf):
    Re = max(Re, 1e-12)
    return 24.0 / Re * (1.0 + 0.15 * Re ** 0.687) + cd_inf


def solve(V, A_proj, d_A, rho_p, cd_inf):
    """力平衡 V g (rho_p - rho_w) = 0.5 rho_w Cd w^2 A_proj，固定點迭代求 w。"""
    return solve_with(lambda Re: cd_shape(Re, cd_inf), V, A_proj, d_A, rho_p)


def solve_with(cdfun, V, A_proj, d_A, rho_p):
    """與 solve() 完全相同的固定點迭代邏輯，但阻力係數關聯式改由 cdfun(Re) 提供。

    用途：讓求解器驗證區塊可以套用外部文獻的阻力關聯式（如 Clift & Gauvin 1970）
    重新解出 w，藉此與主求解器（Schiller-Naumann，見 cd_shape）互相對照，
    而不需複製一份迭代邏輯、也不影響 solve() 本身的行為。"""
    R = rho_p / RHO_W - 1.0
    if R <= 0:
        return 0.0, 0.0, float("nan")
    num = 2.0 * V * G * R / A_proj
    w = 1e-4
    for _ in range(500):
        Cd = cdfun(w * d_A / NU)
        w_new = (num / Cd) ** 0.5
        if abs(w_new - w) < 1e-12 * max(w_new, 1e-12):
            w = w_new
            break
        w = 0.5 * w + 0.5 * w_new
    Re = w * d_A / NU
    return w, Re, cdfun(Re)


def cd_clift_gauvin(Re):
    """Clift & Gauvin (1970) 寬雷諾數球體阻力關聯式，適用範圍遠超 Schiller-Naumann，
    做為獨立於本案求解器之外的標準球體阻力曲線對照基準。"""
    Re = max(Re, 1e-12)
    return 24.0 / Re * (1.0 + 0.15 * Re ** 0.687) + 0.42 / (1.0 + 4.25e4 * Re ** -1.16)


def w_ferguson_church(D, rho_p, C1=18.0, C2=0.4):
    """Ferguson & Church (2004) 沉積物顆粒沉降速度顯式公式（不需迭代，光滑球體取
    C1=18.0, C2=0.4），做為沉積物領域獨立經驗式之第二重對照基準。"""
    R = rho_p / RHO_W - 1.0
    return R * G * D ** 2 / (C1 * NU + math.sqrt(0.75 * C2 * R * G * D ** 3))


# ---------------------------------------------------------------------------
# 幾何函式
# ---------------------------------------------------------------------------
def geom_slab(L, W, t):
    """矩形薄板，以寬面（L x W）水平下墜。L, W, t 單位 m。回傳 (V, A_proj, d_A)。"""
    V = L * W * t
    A_proj = L * W
    d_A = math.sqrt(4.0 * A_proj / math.pi)
    return V, A_proj, d_A


def geom_cyl(d, L):
    """圓柱體，軸向水平下墜（側面迎流）。d, L 單位 m。回傳 (V, A_proj, d_A)。"""
    V = math.pi / 4.0 * d ** 2 * L
    A_proj = d * L
    d_A = math.sqrt(4.0 * A_proj / math.pi)
    return V, A_proj, d_A


def geom_bluff(V_env):
    """鈍體（以外包絡體積等效球徑近似，適用罐、瓶、碎片等緊實鈍體）。V_env 單位 m^3。
    回傳 (V, A_proj, d_A)，其中 V 即 V_env（力平衡使用的浮力/重力體積）。
    僅適用於投影面積接近球體投影（緊實鈍體）的情境；壓扁後呈扁平片狀者
    應改用 geom_slab（見下方寶特瓶/鋁罐/鐵罐壓扁情境），否則會因低估投影
    面積、高估阻力係數而系統性高估沉降速度。"""
    D_eq = (6.0 * V_env / math.pi) ** (1.0 / 3.0)
    A_proj = math.pi / 4.0 * D_eq ** 2
    d_A = D_eq
    return V_env, A_proj, d_A


def rho_eff_shell(m_solid, rho_solid, V_env):
    """薄壁中空物件（罐、瓶）有效密度：殼體材料體積之外的外包絡體積視為進水。

    若外包絡體積小於殼體實心體積（V_env < m_solid/rho_solid），代表殼體材料
    根本塞不進該外包絡體積，此近似不再合理；此時直接取材料密度 rho_solid
    作為有效密度（見 rho_eff_note 產生對應提示文字）。"""
    V_solid = m_solid / rho_solid
    if V_env < V_solid:
        return rho_solid
    V_water = V_env - V_solid
    return (m_solid + V_water * RHO_W) / V_env


def rho_eff_note(m_solid, rho_solid, V_env):
    """若 rho_eff_shell 觸發體積下限保護，回傳提示字串；否則回傳空字串。"""
    if V_env < m_solid / rho_solid:
        return "外包絡體積已小於殼體實心體積，取材料密度。"
    return ""


# ---------------------------------------------------------------------------
# 小工具：幾何平均（optional numpy）
# ---------------------------------------------------------------------------
def geo_mean(values):
    values = [v for v in values if v > 0]
    if HAVE_NUMPY:
        arr = np.asarray(values, dtype=float)
        return float(np.exp(np.mean(np.log(arr))))
    return math.exp(sum(math.log(v) for v in values) / len(values))


# ---------------------------------------------------------------------------
# 求解器驗證
# ---------------------------------------------------------------------------
def validate():
    print("=" * 78)
    print("求解器驗證：石英砂球體 (rho_p = 2650 kg/m^3, cd_inf = 0.0 即標準球體阻力)")
    print("與兩個可引用之公認關聯式即時計算比對，不使用寫死之期望值")
    print("=" * 78)
    quartz_rho = 2650.0
    diam_mm = [0.1, 0.5, 1.0, 2.0, 5.0]

    header = (
        f"  {'d(mm)':>6s} {'Re':>9s} {'本案 w':>10s} "
        f"{'Clift-Gauvin w':>15s} {'差異%':>8s} "
        f"{'Ferguson-Church w':>18s} {'差異%':>8s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for d_mm in diam_mm:
        d = d_mm / 1000.0
        V = math.pi / 6.0 * d ** 3
        A_proj = math.pi / 4.0 * d ** 2
        d_A = d

        # 本案求解器（Schiller-Naumann 閉合式，cd_inf=0.0）
        w, Re, _ = solve(V, A_proj, d_A, quartz_rho, cd_inf=0.0)

        # 對照一：Clift & Gauvin (1970)，套用與主求解器相同的固定點迭代求 w
        w_cg, _, _ = solve_with(cd_clift_gauvin, V, A_proj, d_A, quartz_rho)
        diff_cg_pct = (w - w_cg) / w_cg * 100.0

        # 對照二：Ferguson & Church (2004)，顯式公式不需迭代
        w_fc = w_ferguson_church(d, quartz_rho)
        diff_fc_pct = (w - w_fc) / w_fc * 100.0

        print(
            f"  {d_mm:6.1f} {Re:9.1f} {w:10.4f} "
            f"{w_cg:15.4f} {diff_cg_pct:+8.1f} "
            f"{w_fc:18.4f} {diff_fc_pct:+8.1f}"
        )

    print(
        "\n  註：Re ≲ 1000 範圍內，本案採用的 Schiller-Naumann 閉合式與標準球體阻力曲線\n"
        "  （Clift & Gauvin 1970）相差 1.5% 以內；Re > 1000 後偏高，5 mm 檔 +12.5%，\n"
        "  此為 Schiller-Naumann 關聯式已知的適用範圍上限（原文適用 Re < 1000），\n"
        "  非程式邏輯錯誤。Ferguson & Church (2004) 為沉積物領域的獨立經驗式，\n"
        "  列為第二重對照（詳見 README 限制章節）。"
    )
    print()


# ---------------------------------------------------------------------------
# 十類物件模板
# ---------------------------------------------------------------------------
CD_INF = {
    "slab": (1.1, 2.0),   # (lo -> w_hi 穩定下墜, hi -> w_lo 翻滾飄盪)
    "cyl": (0.9, 1.3),
    "bluff": (0.6, 1.2),
}


def mm(x):
    return x / 1000.0


CLASSES = [
    {
        "class_id": 1,
        "class_zh": "保麗龍",
        "shape": "slab",
        "scenarios": [
            {
                "scenario": "S 小塊發泡板",
                "geom": geom_slab(mm(10), mm(10), mm(5)),
                "rho_p": 1026.0,
                "note": "小型 EPS 包材碎塊，密度取近飽和海水（微浮力差），"
                        "尺寸代表常見緩衝角料。",
            },
            {
                "scenario": "M 中型保麗龍箱體殘片",
                "geom": geom_slab(mm(50), mm(50), mm(10)),
                "rho_p": 1060.0,
                "note": "常見保麗龍漁箱/包裝箱破片尺寸，密度反映部分吸水與外殼材質。",
            },
            {
                "scenario": "L 大型浮具/保麗龍板",
                "geom": geom_slab(mm(100), mm(100), mm(20)),
                "rho_p": 1100.0,
                "note": "大型保麗龍浮具或蚵棚保麗龍浮筒破片，長期泡水吸附生物膜、"
                        "密度較高。",
            },
        ],
        "current_value": 1e-4,
    },
    {
        "class_id": 2,
        "class_zh": "竹木",
        "shape": "cyl",
        "scenarios": [
            {
                "scenario": "S 細竹枝/小木條",
                "geom": geom_cyl(mm(5), mm(50)),
                "rho_p": 1030.0,
                "note": "漂流細竹枝/小木條，密度略高於淡水生木材，反映海水吸附與部分吸水。",
            },
            {
                "scenario": "M 中型漂流木條",
                "geom": geom_cyl(mm(15), mm(150)),
                "rho_p": 1090.0,
                "note": "中型漂流木/竹材，密度取一般硬木浸水後之常見範圍。",
            },
            {
                "scenario": "L 大型漂流木段",
                "geom": geom_cyl(mm(30), mm(300)),
                "rho_p": 1150.0,
                "note": "大型漂流木段，長期泡水吸水飽和、密度接近或超過部分緻密木種浸水值。",
            },
        ],
        "current_value": 2e-4,
    },
    {
        "class_id": 3,
        "class_zh": "廢紙",
        "shape": "slab",
        "scenarios": [
            {
                "scenario": "S 薄紙片",
                "geom": geom_slab(mm(10), mm(10), mm(0.3)),
                "rho_p": 1050.0,
                "note": "小張濕紙片，密度取略高於海水以反映吸水初期。",
            },
            {
                "scenario": "M 濕紙板碎片",
                "geom": geom_slab(mm(50), mm(50), mm(3)),
                "rho_p": 1120.0,
                "note": "瓦楞紙板/厚紙浸水後碎片，密度反映纖維吸水膨潤。",
            },
            {
                "scenario": "L 飽和厚紙板",
                "geom": geom_slab(mm(100), mm(100), mm(10)),
                "rho_p": 1200.0,
                "note": "大片厚紙板長時間泡水達飽和吸水狀態，密度明顯高於海水。",
            },
        ],
        "current_value": 5e-4,
    },
    {
        "class_id": 4,
        "class_zh": "其他／不可回收",
        "shape": "slab",
        "scenarios": [
            {
                "scenario": "S 極薄雜項薄膜",
                "geom": geom_slab(mm(50), mm(50), mm(0.03)),
                "rho_p": 1030.0,
                "note": "混合材質極薄膜片（如複合包裝膜），密度取略高於海水。",
            },
            {
                "scenario": "M 中厚雜項複合片",
                "geom": geom_slab(mm(150), mm(150), mm(0.5)),
                "rho_p": 1200.0,
                "note": "混合材質複合片（布料、皮革、複合材等），密度取常見混合廢棄物範圍。",
            },
            {
                "scenario": "L 厚實雜項廢片",
                "geom": geom_slab(mm(300), mm(300), mm(2)),
                "rho_p": 1400.0,
                "note": "厚實雜項不可回收廢棄物件（如混合橡塑膠、填充材），密度取偏高範圍。",
            },
        ],
        "current_value": 1e-3,
    },
    {
        "class_id": 5,
        "class_zh": "廢漁網漁具",
        "shape": "cyl",
        "scenarios": [
            {
                "scenario": "S 單根網線",
                "geom": geom_cyl(mm(0.5), mm(50)),
                "rho_p": 1050.0,
                "note": "單根 PA/PE 網線纖維，無配重，密度取略高於海水的一般漁網材質估計。",
            },
            {
                "scenario": "M 網片段",
                "geom": geom_cyl(mm(2), mm(200)),
                "rho_p": 1140.0,
                "note": "小段纏結網片段，密度反映附著生物與泥沙。",
            },
            {
                "scenario": "L 繩索",
                "geom": geom_cyl(mm(20), mm(500)),
                "rho_p": 1200.0,
                "note": "廢棄纜繩/粗繩索，密度取含配重、附著物之偏高估計。",
            },
        ],
        "current_value": 2e-3,
    },
    {
        "class_id": 6,
        "class_zh": "寶特瓶",
        "shape": "mixed",
        "scenarios": [
            {
                "scenario": "S 600 mL 完整進水瓶身",
                "shape": "bluff",
                "geom": geom_bluff(620e-6),
                "rho_p": rho_eff_shell(0.025, 1380.0, 620e-6),
                "cd_inf": CD_INF["bluff"],
                "note": "600 mL 完整寶特瓶，殼重 m_solid=25 g、PET 密度 1380 kg/m^3，"
                        "V_env=620e-6 m^3 為外包絡體積；完整瓶身視為緊實鈍體，"
                        "以外包絡體積等效球徑（geom_bluff）計算阻力；rho_eff 以殼體+"
                        "內部進水之整體密度計算，等效密度約 "
                        f"{rho_eff_shell(0.025, 1380.0, 620e-6):.1f} kg/m^3。"
                        + rho_eff_note(0.025, 1380.0, 620e-6),
            },
            {
                "scenario": "M 壓扁瓶身",
                "shape": "slab",
                "geom": geom_slab(0.25, 0.07, 0.020),
                "rho_p": rho_eff_shell(0.025, 1380.0, 0.25 * 0.07 * 0.020),
                "cd_inf": CD_INF["slab"],
                "note": "壓扁後之瓶身呈扁平片狀（250×70×20 mm），投影面積大、阻力高，"
                        "改以薄板幾何（geom_slab）計算，不再視為緊實鈍體；殼重與材質"
                        "密度假設同 S 情境，V_env 取薄板體積 0.25×0.07×0.020 m^3，"
                        "等效密度約 "
                        f"{rho_eff_shell(0.025, 1380.0, 0.25 * 0.07 * 0.020):.1f} kg/m^3。"
                        + rho_eff_note(0.025, 1380.0, 0.25 * 0.07 * 0.020),
            },
            {
                "scenario": "L 嚴重壓扁瓶身",
                "shape": "slab",
                "geom": geom_slab(0.25, 0.07, 0.008),
                "rho_p": rho_eff_shell(0.025, 1380.0, 0.25 * 0.07 * 0.008),
                "cd_inf": CD_INF["slab"],
                "note": "嚴重壓扁後之瓶身呈更扁平片狀（250×70×8 mm），同以薄板幾何計算，"
                        "殼重與材質密度假設同 S 情境，等效密度約 "
                        f"{rho_eff_shell(0.025, 1380.0, 0.25 * 0.07 * 0.008):.1f} kg/m^3。"
                        + rho_eff_note(0.025, 1380.0, 0.25 * 0.07 * 0.008),
            },
        ],
        "current_value": 5e-3,
    },
    {
        "class_id": 7,
        "class_zh": "其他／可回收",
        "shape": "slab",
        "scenarios": [
            {
                "scenario": "S 薄片可回收料",
                "geom": geom_slab(mm(50), mm(50), mm(3)),
                "rho_p": 1100.0,
                "note": "混合可回收薄片（如硬質塑膠片、複合金屬片），密度取略高於海水。",
            },
            {
                "scenario": "M 中厚可回收料",
                "geom": geom_slab(mm(100), mm(100), mm(10)),
                "rho_p": 1200.0,
                "note": "中等厚度可回收廢棄片材，密度取常見混合可回收物範圍。",
            },
            {
                "scenario": "L 厚實可回收料",
                "geom": geom_slab(mm(200), mm(200), mm(20)),
                "rho_p": 1400.0,
                "note": "厚實可回收廢棄片材（如複合板材），密度取偏高範圍。",
            },
        ],
        "current_value": 1e-2,
    },
    {
        "class_id": 8,
        "class_zh": "鋁罐",
        "shape": "mixed",
        "scenarios": [
            {
                "scenario": "S 完整進水鋁罐",
                "shape": "bluff",
                "geom": geom_bluff(390e-6),
                "rho_p": rho_eff_shell(0.013, 2700.0, 390e-6),
                "cd_inf": CD_INF["bluff"],
                "note": "330 mL 鋁罐之外包絡體積約 390 cm^3；殼重 m_solid=13 g、"
                        "鋁密度 2700 kg/m^3；完整罐身視為緊實鈍體，以外包絡體積等效"
                        "球徑（geom_bluff）計算阻力，等效密度約 "
                        f"{rho_eff_shell(0.013, 2700.0, 390e-6):.1f} kg/m^3。"
                        + rho_eff_note(0.013, 2700.0, 390e-6),
            },
            {
                "scenario": "M 半壓扁鋁罐",
                "shape": "slab",
                "geom": geom_slab(0.10, 0.06, 0.025),
                "rho_p": rho_eff_shell(0.013, 2700.0, 0.10 * 0.06 * 0.025),
                "cd_inf": CD_INF["slab"],
                "note": "半壓扁鋁罐呈扁平片狀（100×60×25 mm），投影面積大、阻力高，"
                        "改以薄板幾何（geom_slab）計算而非緊實鈍體，殼重與材質密度"
                        "假設同 S 情境，等效密度約 "
                        f"{rho_eff_shell(0.013, 2700.0, 0.10 * 0.06 * 0.025):.1f} kg/m^3。"
                        + rho_eff_note(0.013, 2700.0, 0.10 * 0.06 * 0.025),
            },
            {
                "scenario": "L 嚴重壓扁鋁罐",
                "shape": "slab",
                "geom": geom_slab(0.10, 0.06, 0.008),
                "rho_p": rho_eff_shell(0.013, 2700.0, 0.10 * 0.06 * 0.008),
                "cd_inf": CD_INF["slab"],
                "note": "嚴重壓扁鋁罐呈更扁平片狀（100×60×8 mm），同以薄板幾何計算，"
                        "等效密度約 "
                        f"{rho_eff_shell(0.013, 2700.0, 0.10 * 0.06 * 0.008):.1f} kg/m^3。"
                        + rho_eff_note(0.013, 2700.0, 0.10 * 0.06 * 0.008),
            },
        ],
        "current_value": 2e-2,
    },
    {
        "class_id": 9,
        "class_zh": "鐵罐",
        "shape": "mixed",
        "scenarios": [
            {
                "scenario": "S 完整進水鐵罐",
                "shape": "bluff",
                "geom": geom_bluff(425e-6),
                "rho_p": rho_eff_shell(0.055, 7850.0, 425e-6),
                "cd_inf": CD_INF["bluff"],
                "note": "鐵罐殼重 m_solid=55 g、鋼密度 7850 kg/m^3，V_env=425e-6 m^3 "
                        "代表接近完整之外包絡體積；完整罐身視為緊實鈍體，以外包絡"
                        "體積等效球徑（geom_bluff）計算阻力，等效密度約 "
                        f"{rho_eff_shell(0.055, 7850.0, 425e-6):.1f} kg/m^3。"
                        + rho_eff_note(0.055, 7850.0, 425e-6),
            },
            {
                "scenario": "M 半壓扁鐵罐",
                "shape": "slab",
                "geom": geom_slab(0.10, 0.075, 0.030),
                "rho_p": rho_eff_shell(0.055, 7850.0, 0.10 * 0.075 * 0.030),
                "cd_inf": CD_INF["slab"],
                "note": "半壓扁鐵罐呈扁平片狀（100×75×30 mm），投影面積大、阻力高，"
                        "改以薄板幾何（geom_slab）計算而非緊實鈍體，等效密度約 "
                        f"{rho_eff_shell(0.055, 7850.0, 0.10 * 0.075 * 0.030):.1f} kg/m^3。"
                        + rho_eff_note(0.055, 7850.0, 0.10 * 0.075 * 0.030),
            },
            {
                "scenario": "L 嚴重壓扁鐵罐",
                "shape": "slab",
                "geom": geom_slab(0.10, 0.075, 0.012),
                "rho_p": rho_eff_shell(0.055, 7850.0, 0.10 * 0.075 * 0.012),
                "cd_inf": CD_INF["slab"],
                "note": "嚴重壓扁鐵罐呈更扁平片狀（100×75×12 mm），同以薄板幾何計算，"
                        "等效密度約 "
                        f"{rho_eff_shell(0.055, 7850.0, 0.10 * 0.075 * 0.012):.1f} kg/m^3。"
                        + rho_eff_note(0.055, 7850.0, 0.10 * 0.075 * 0.012),
            },
        ],
        "current_value": 5e-2,
    },
    {
        "class_id": 10,
        "class_zh": "玻璃瓶",
        "shape": "mixed",
        "scenarios": [
            {
                "scenario": "S 完整玻璃瓶（進水）",
                "shape": "bluff",
                "geom": geom_bluff(700e-6),
                "rho_p": rho_eff_shell(0.5, 2500.0, 700e-6),
                "cd_inf": CD_INF["bluff"],
                "note": "完整玻璃瓶，殼重 m_solid=500 g、玻璃密度 2500 kg/m^3，"
                        "V_env=700e-6 m^3，內部視為進水，等效密度約 "
                        f"{rho_eff_shell(0.5, 2500.0, 700e-6):.1f} kg/m^3。"
                        + rho_eff_note(0.5, 2500.0, 700e-6),
            },
            {
                "scenario": "M 緻密碎片（等效球徑 20 mm）",
                "shape": "bluff",
                "geom": geom_bluff(math.pi / 6.0 * mm(20) ** 3),
                "rho_p": 2500.0,
                "cd_inf": CD_INF["bluff"],
                "note": "破碎後緻密玻璃碎片，以等效球徑 20 mm 代表，"
                        "密度取實心玻璃密度 2500 kg/m^3（無進水空腔）。",
            },
            {
                "scenario": "L 瓶身碎片（薄板）",
                "shape": "slab",
                "geom": geom_slab(0.05, 0.03, 0.003),
                "rho_p": 2500.0,
                "cd_inf": CD_INF["slab"],
                "note": "破碎玻璃瓶身多呈曲殼薄片狀而非緻密塊體，投影面積大、阻力高，"
                        "改以薄板幾何（50×30×3 mm）取代緻密碎片假設；密度取實心玻璃"
                        "密度 2500 kg/m^3；此情境對應 Chan et al. (2021) 實測之瓶片狀"
                        "曲殼幾何。",
            },
        ],
        "current_value": 1e-1,
    },
]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run():
    validate()

    rows = []
    summary = []

    print("=" * 78)
    print("逐情境結果")
    print("=" * 78)

    for cls in CLASSES:
        class_shape = cls["shape"]
        print(f"\n[{cls['class_id']:2d}] {cls['class_zh']}  (幾何類型: {class_shape})")

        all_w_for_class = []

        for sc in cls["scenarios"]:
            V, A_proj, d_A = sc["geom"]
            rho_p = sc["rho_p"]
            shape_sc = sc.get("shape", class_shape)
            cd_lo, cd_hi = sc.get("cd_inf", CD_INF[shape_sc])  # cd_lo -> w_hi；cd_hi -> w_lo

            w_hi, Re_hi, Cd_hi = solve(V, A_proj, d_A, rho_p, cd_lo)
            w_lo, Re_lo, Cd_lo_val = solve(V, A_proj, d_A, rho_p, cd_hi)

            all_w_for_class.extend([w_lo, w_hi])

            re_min, re_max = sorted([Re_lo, Re_hi])
            cd_min, cd_max = sorted([Cd_hi, Cd_lo_val])

            print(
                f"    - {sc['scenario']:<28s} geom={shape_sc:<5s} cd_inf=[{cd_lo}/{cd_hi}]  "
                f"V={V:.3e} m^3  A_proj={A_proj:.3e} m^2  "
                f"rho_p={rho_p:7.1f}  w_lo={w_lo:.5f}  w_hi={w_hi:.5f} m/s  "
                f"Re=[{re_min:.2f},{re_max:.2f}]  Cd=[{cd_min:.3f},{cd_max:.3f}]"
            )
            print(f"      note: {sc['note']}")

            rows.append({
                "class_id": cls["class_id"],
                "class_zh": cls["class_zh"],
                "scenario": sc["scenario"],
                "geometry": shape_sc,
                "rho_p_kgm3": f"{rho_p:.3f}",
                "V_m3": f"{V:.6e}",
                "A_proj_m2": f"{A_proj:.6e}",
                "cd_inf_lo": cd_lo,
                "cd_inf_hi": cd_hi,
                "w_lo_ms": f"{w_lo:.6f}",
                "w_hi_ms": f"{w_hi:.6f}",
                "Re_lo": f"{re_min:.4f}",
                "Re_hi": f"{re_max:.4f}",
                "note": sc["note"],
            })

        w_min = min(all_w_for_class)
        w_max = max(all_w_for_class)
        w_repr = geo_mean(all_w_for_class)

        summary.append({
            "class_id": cls["class_id"],
            "class_zh": cls["class_zh"],
            "w_min": w_min,
            "w_repr": w_repr,
            "w_max": w_max,
            "current_value": cls["current_value"],
        })

    # -------------------------------------------------------------------
    # 寫出 results.csv
    # -------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "results.csv")
    fieldnames = [
        "class_id", "class_zh", "scenario", "geometry", "rho_p_kgm3",
        "V_m3", "A_proj_m2", "cd_inf_lo", "cd_inf_hi",
        "w_lo_ms", "w_hi_ms", "Re_lo", "Re_hi", "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # -------------------------------------------------------------------
    # 彙總表
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("十類彙總表  (單位: m/s)")
    print("=" * 78)
    header = f"{'class':>2s}  {'類別':<12s}{'w_min':>10s}{'w_repr':>10s}{'w_max':>10s}{'現行值':>10s}{'比值(現行/w_repr)':>20s}"
    print(header)
    print("-" * len(header))
    for s in summary:
        ratio = s["current_value"] / s["w_repr"] if s["w_repr"] > 0 else float("nan")
        print(
            f"{s['class_id']:>2d}  {s['class_zh']:<12s}"
            f"{s['w_min']:>10.5f}{s['w_repr']:>10.5f}{s['w_max']:>10.5f}"
            f"{s['current_value']:>10.5f}{ratio:>20.3f}"
        )

    print(f"\n已寫出: {csv_path}")
    return rows, summary


if __name__ == "__main__":
    run()
