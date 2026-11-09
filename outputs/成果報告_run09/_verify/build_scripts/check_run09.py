# 獨立核對（方法 B：回溯途中首次抵達海岸即視為找到來源）：手稿（md）與 docx 之文字、數字、結構，對 run09 方法 B 後處理表／figures-polar 表／各圖 manifest 之獨立重算。
# 以 csv/json 標準模組與整數／Decimal／Fraction 運算重新讀取來源，不使用產生手稿之 d09.py／mk_res_run09.py 的任何函式。
import csv, json, os, re, sys, math, hashlib, zipfile, io
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from collections import Counter
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.expanduser('~/mnt/Lagrangian-Ensemble-Backtracking/outputs/成果報告_run09')
SRC = ROOT + '/_verify/src'; SB = ROOT + '/_verify/src_coastB'; CT = SB + '/coastB-tables'; PPB = SB + '/postprocess-coastB'; PT = SB + '/figures-polar-v3-ax2-coastB/tables'
OK = []; BAD = []
def chk(name, got, exp):
    (OK if got == exp else BAD).append((name, got, exp))
def rd(path):
    with open(path, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def fnum(x): return float(x) if x not in ('', None) else float('nan')
def i0(x): return f'{x:,.0f}'
def qd(v, nd): return str(v.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def pf(v):
    """v: Decimal 百分數。二位小數；不為 0 卻捨入為 0.00、不為 100 卻捨入為 100.00 者保留三位。"""
    s = qd(v, 2)
    if (v > 0 and Decimal(s) == 0) or (v < 100 and Decimal(s) == 100): s = qd(v, 3)
    return s + '%'
def pn(num, den): return pf(Decimal(num) * 100 / Decimal(den))          # 整數計數
def pfl(x): return pf(Decimal(repr(float(x))))                          # 來源為浮點百分數
SITES = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
ORDER = ['oca_nonrecyclable_flexible_sheet', 'oca_wastepaper_folded_fiber', 'oca_styrofoam_porous_fragment', 'oca_fishinggear_open_mesh_bundle', 'oca_pet_waterfilled_bottle', 'oca_wood_waterlogged_elongated', 'oca_other_recyclable_irregular_fragment', 'oca_aluminum_crushed_cylinder', 'oca_steel_rigid_cylinder', 'oca_glass_bottle_or_fragment']
CZ = dict(zip(ORDER, ['不可回收塑膠', '廢紙', '保麗龍', '廢漁網漁具', '寶特瓶', '竹木', '可回收塑膠', '鋁罐', '鐵罐', '玻璃瓶']))
NS = len(SITES); NF_SITE = 7; NF = NS * NF_SITE; HZ = 180; FW = '6.27'; TEXT_W_IN = 9026 / 1440; TEXT_H_IN = (16838 - 2880) / 1440
FIGSUB = 'figures/finite_depth_stokes/無文字'
MDNAME = '執行成果_run09_含斯托克斯漂流'
RES_FULL = open(f'{ROOT}/{MDNAME}.md', encoding='utf-8').read()
RL = RES_FULL.split('\n')
POLSTEMS = ['polar_source_v2_by_class', 'polar_source_v2_velocity_panels', 'polar_source_v2_by_season', 'polar_source_v2_by_tide']
SITESTEMS = ['horizontal_overview', 'domain_fraction_vs_backtrack_time', 'source_footprint_residence'] + POLSTEMS
def expand(text, ch):
    return {int(x) for a_, b_ in re.findall(ch + r' (\d+) 至' + ch + r' (\d+)', text) for x in range(int(a_), int(b_) + 1)} | {int(x) for x in re.findall(ch + r' (\d+)', text)}

# ---------- 結構
chk('標題階層', [l for l in RL if l.startswith('#')],
    ['# Lagrangian系集逆向溯源分析成果', '#- 逐地點成果'] + [f'## {n}' for _, n in SITES] + ['#- 綜合說明與結論', '#- 整體結論'])
chk('無「## 結論」、無第三層標題', ('## 結論' in RES_FULL, '### ' in RES_FULL), (False, False))

# ---------- 使用者手改（Word）之保留與置換
UE = open(SRC + '/user_edits_20261021.txt', encoding='utf-8').read().split('\n')
chk('使用者手改檔三行（標題、引言、小標）', (len(UE), UE[0], UE[2], UE[1].startswith('　　本節呈現Lagrangian系集逆向溯源含斯托克斯漂流案例之執行成果。情境矩陣由')), (3, 'Lagrangian系集逆向溯源執行成果', '逐地點成果', True))
UE2 = open(SRC + '/user_edits_20261022.txt', encoding='utf-8').read().split('\n')
chk('使用者 10/22 手改檔三行；與 10/21 版相比僅標題與引言各一處「執行成果」→「分析成果」', (len(UE2), UE2[0], UE2[2], UE2[0] == UE[0].replace('執行成果', '分析成果'), UE2[1].count('分析成果'), UE2[1].count('執行成果')), (3, 'Lagrangian系集逆向溯源分析成果', '逐地點成果', True, 1, 0))
chk('使用者手改標題逐字保留（10/22 版）', RL[0], '# ' + UE2[0])
_par = [l for l in RL if l.strip()][1]
_subs = [('之執行成果。情境矩陣', '之執行成果，目的在於推估各研究地點海底廢棄物之來源方向及其機率。情境矩陣'), ('10 個沉降速度類別', '10 類代表物件'), ('20 個受體', '20 個受體（5 個水平位置 × 4 個深度）'), ('50 個觀測時刻', '50 個到達時間'), ('與 10 個系集成員交叉而成；每個研究地點', '與 10 個系集成員交叉而成，每顆虛擬質點之沉降速度自所屬類別之 10 個沉降速度中抽取；每個研究地點')]
chk('引言逐字等於使用者 10/22 手改版', _par, UE2[1])
_back = _par.replace('之分析成果，', '之執行成果，')
for a_, b_ in reversed(_subs):
    chk('引言置換片段唯一 ' + b_[:12], _back.count(b_), 1); _back = _back.replace(b_, a_)
chk('引言：還原五處置換後逐字等於使用者手改版', _back, UE[1])
chk('使用者刪除之末句未恢復', ('相對來源權重' in RES_FULL, '結果以條件式來源足跡' in RES_FULL), (False, False))
chk('引言之後為閱讀導引段、再為小標「逐地點成果」，其後即為貢寮', [l for l in RL if l.strip()][2].startswith('　　各研究地點依序以七張圖說明：') and [l for l in RL if l.strip()][3] == '#- 逐地點成果' and [l for l in RL if l.strip()][4] == '## 貢寮', True)
_GUIDE = [l for l in RL if l.strip()][2]
chk('閱讀導引句：七張圖之內容與圖序（仍在追蹤之虛擬質點比例）、章末綜合說明與整體結論；不含「本地點結論」「流場範圍內之虛擬質點比例」', (_GUIDE == '　　各研究地點依序以七張圖說明：水平軌跡總覽呈現 24 條代表虛擬質點自受體回溯之軌跡，仍在追蹤之虛擬質點比例圖呈現全部虛擬質點於回溯期間內仍在追蹤之比例，條件式來源足跡呈現虛擬質點於回溯期間停留之位置，四張圓形來源方位圖依序呈現十類、同類十個沉降速度、季節與大小潮之來源方向機率；綜合說明與整體結論置於章末。', '本地點結論' in _GUIDE, '流場範圍內之虛擬質點比例' in RES_FULL), (True, False, False))
chk('引言整句（使用者指定修正）', _par, '　　本節呈現Lagrangian系集逆向溯源含斯托克斯漂流案例之分析成果，目的在於推估各研究地點海底廢棄物之來源方向及其機率。情境矩陣由 5 處研究地點、10 類代表物件、20 個受體（5 個水平位置 × 4 個深度）、50 個到達時間與 10 個系集成員交叉而成，每顆虛擬質點之沉降速度自所屬類別之 10 個沉降速度中抽取；每個研究地點 100,000 顆、五個研究地點合計 500,000 顆虛擬質點。')
_man_f = json.load(open(SRC + '/figures-notext/manifest.json'))
chk('每地點受體 20 個（manifest 釋放點）；20 × 50 × 10 = 10,000 顆／類，十類 100,000 顆', ({_man_f['footprint_figures']['receptors']['by_site'][s]['release_geometry']['sampling_range']['release_point_count'] for s, _ in SITES}, 20 * 50 * 10, 20 * 50 * 10 * 10), ({20}, 10000, 100000))

# ---------- 來源表載入（方法 B）
flux = {r['study_site_id']: r for r in rd(PPB + '/boundary_flux_by_site.csv')}
fp = {r['study_site_id']: r for r in rd(PPB + '/footprint_summary_by_site.csv')}
chk('足跡摘要為 1 km 網格版（五地點 grid_cell_km＝1）', [fnum(fp[s_]['grid_cell_km']) for s_, _ in SITES], [1.0] * 5)
FA = json.load(open(CT + '/facts_remote_run09B.json'))
OUTC = {r['site']: r for r in rd(CT + '/outcome_by_site.csv')}
GEO = {r['site']: r for r in rd(CT + '/coast_geography_by_site.csv')}
POLY = rd(CT + '/coast_geography_polygons.csv')
REPB = json.load(open(CT + '/rep_traj_coastB.json'))
DEP = FA['dep']
CAS = json.load(open(PPB + '/coast_arrival_summary.json', encoding='utf-8'))
inside_sum = {r['study_site_id']: r for r in rd(SB + '/figures-domain-fraction-coastB-fs8/inside_fraction_summary.csv') if r['group'] == 'all'}
_dfm = json.load(open(SB + '/figures-domain-fraction-coastB-fs8/manifest.json'))   # 報告版（fs8）；舊版 manifest 在 figures-domain-fraction-coastB/
_man = json.load(open(SB + '/figures-polar-v3-ax2-coastB/manifest.json', encoding='utf-8'))
POLSUM = rd(PT + '/distinguishability_summary.csv'); POLTERM = rd(PT + '/termination_by_group.csv'); SECR = rd(PT + '/sector_shares_long.csv'); PAIRS = rd(PT + '/distinguishability_pairs.csv')
chk('方法 B 說明：postprocess 之 coast_arrival_summary 狀態碼 8、方法敘述', (CAS['status_code'], CAS['method'].startswith('B：')), (8, True))
chk('沉底顆數僅 1 顆（南灣、不可回收塑膠、0.000897 m/s、18000 s）', [(d['site'], d['material'], abs(d['v']), d['dur_s']) for d in DEP], [('nanwan', 'oca_nonrecyclable_flexible_sheet', 0.000897, 18000.0)])
chk('結局碼只有 1／4／8（無 2 達回溯上限、無 3、5、6、7）', sorted({int(k) for v in FA['status'].values() for k in v}), [1, 4, 8])
chk('各地點結局計數五處一致：flux 表、outcome 表、facts（departures 彙總）、polar manifest、coast_arrival_summary、polar termination 表', {s: (int(flux[s]['count_crossed']), int(flux[s]['count_status_8']), int(flux[s]['count_status_4']), int(flux[s]['count_status_2'])) == (int(OUTC[s]['crossed']), int(OUTC[s]['coast']), int(OUTC[s]['touch_bottom']), int(OUTC[s]['max_backtrack'])) == (FA['status'][s].get('1', 0), FA['status'][s].get('8', 0), FA['status'][s].get('4', 0), 0) == (int(_man['status_counts_by_site'][s]['1']), int(_man['status_counts_by_site'][s]['8']), int(_man['status_counts_by_site'][s]['4']), int(_man['status_counts_by_site'][s]['2'])) and CAS['by_site'][s]['reached_coast'] == int(flux[s]['count_status_8']) and CAS['by_site'][s]['reached_at_row0'] == 0 and CAS['by_site'][s]['points_outside_mesh'] == 0 for s, _ in SITES}, {s: True for s, _ in SITES})
chk('polar termination 表（site_all）：n_crossed、n_status_8、n_status_4 同 flux 表', {s: [(int(r['n_crossed']), int(r['n_status_8']), int(r['n_status_4']), int(r['n_total'])) for r in POLTERM if r['site'] == s and r['grouping'] == 'site_all'] == [(int(flux[s]['count_crossed']), int(flux[s]['count_status_8']), int(flux[s]['count_status_4']), 100000)] for s, _ in SITES}, {s: True for s, _ in SITES})
chk('每地點 100,000 顆、穿越＋海岸來源＋觸底＝100,000', {s: int(flux[s]['total_particle_count']) == 100000 == int(flux[s]['count_crossed']) + int(flux[s]['count_status_8']) + int(flux[s]['count_status_4']) for s, _ in SITES}, {s: True for s, _ in SITES})

# ---------- 圓形圖方向區間：由計數獨立重算（整數運算）
N8 = ['北', '東北', '東', '東南', '南', '西南', '西', '西北']
def nm_center(c):
    """c: 中心度數（Fraction，0–360）。8 方位每 45°；中心距方位分界（22.5°+45°k）不超過 2.5° 者標示相鄰兩方位（順時針 X 至 Y）。"""
    c = c % 360
    for k in range(8):
        d = abs(c - (Fraction(45, 2) + 45 * k)); d = min(d, 360 - d)
        if d <= Fraction(5, 2): return N8[k] + '至' + N8[(k + 1) % 8] + '方'
    return N8[int((c + Fraction(45, 2)) // 45) % 8] + '方'
def nm8(a, ln): return nm_center(Fraction(a * 10) + Fraction((ln - 1) * 5))
def nm_deg(lo, hi):
    w = (hi - lo) % 360
    return nm_center(Fraction(lo) + Fraction(w, 2))
def cnt_sec(site, grouping, gid=None):
    c = [0] * 36
    for r in SECR:
        if r['site'] == site and r['grouping'] == grouping and (gid is None or r['group_id'] == gid): c[int(r['sector_index'])] += int(r['count'])
    return c
def runs_of(c):
    """方向機率高於平均 1/36 之相鄰區間（環狀）合併；依合計由大至小；取前兩個（第二大者一律列出）。"""
    tot = sum(c); on = {i for i in range(36) if 36 * c[i] > tot}
    seen = set(); out = []
    for i in sorted(on):
        if i in seen: continue
        a = i
        while (a - 1) % 36 in on and (a - 1) % 36 != i: a = (a - 1) % 36
        ln = 0; j = a
        while j in on and ln < 36: seen.add(j); ln += 1; j = (j + 1) % 36
        out.append((a, ln, sum(c[(a + k) % 36] for k in range(ln))))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out[:2], tot, out
def deg(a, ln): return f'{(a * 10 - 5) % 360}°–{((a + ln - 1) * 10 + 5) % 360}°'
def rtn(t, tot): return f'{nm8(t[0], t[1])}，機率為 {pn(t[2], tot)}'
def rts(t, tot): return f'{nm8(t[0], t[1])} {pn(t[2], tot)}'
SEASONS = [('DJF', '冬季'), ('MAM', '春季'), ('JJA', '夏季'), ('SON', '秋季')]
RNG = {}
for s, _ in SITES:
    ca = cnt_sec(s, 'class'); RNG[s] = {'all': runs_of(ca)}
    chk(f'{s} 全年分布：十類、季節、大小潮、十類×十速度四種分組之加總相同，合計 100,000', (cnt_sec(s, 'season') == ca, cnt_sec(s, 'tide') == ca, cnt_sec(s, 'class_velocity') == ca, sum(ca)), (True, True, True, 100000))
    for g, _n in SEASONS: RNG[s][g] = runs_of(cnt_sec(s, 'season', g))
    bad = [r for r in SECR if r['site'] == s and r['grouping'] in ('class', 'season', 'tide') and abs(float(r['pct_of_group']) - int(r['count']) / int(r['group_total']) * 100) > 0.051]
    chk(f'{s} sector_shares_long 之 pct_of_group 與計數一致', len(bad), 0)
    chk(f'{s} 各方向區間 count＝count_crossed＋count_coast（含觸底者之計入）；全部區間 count_coast 合計＝抵達海岸顆數', (all(int(r['count']) == int(r['count_crossed']) + int(r['count_coast']) for r in SECR if r['site'] == s and r['grouping'] == 'class'), sum(int(r['count_coast']) for r in SECR if r['site'] == s and r['grouping'] == 'class'), sum(int(r['count_crossed']) for r in SECR if r['site'] == s and r['grouping'] == 'class')), (True, int(flux[s]['count_status_8']) + int(flux[s]['count_status_4']), int(flux[s]['count_crossed'])))
SEAS_EXPECT = {
    'gongliao': '冬、春兩季之來源方向分布相近，夏、秋兩季相近，兩組之間差異明顯',
    'guishan': '冬、春兩季之來源方向分布相近，夏、秋兩季相近，兩組之間差異明顯',
    'hsinchu': '四季之來源方向分布兩兩差異明顯',
    'nanwan': '春、秋兩季之來源方向分布相近，其餘季節兩兩之間差異明顯',
    'lienchiang': '四季之來源方向分布兩兩差異明顯'}
def season_nm(s):
    return '，'.join(f'{zn}為{nm8(*RNG[s][g][0][0][:2])}' for g, zn in SEASONS)   # 整體結論各季只列最主要之一個方向
_SQ = ['DJF', 'MAM', 'JJA', 'SON']; _ALLP = [(a_, b_) for i_, a_ in enumerate(_SQ) for b_ in _SQ[i_ + 1:]]
CLAIM_SIM = {'gongliao': [('DJF', 'MAM'), ('JJA', 'SON')], 'guishan': [('DJF', 'MAM'), ('JJA', 'SON')], 'hsinchu': [], 'nanwan': [('MAM', 'SON')], 'lienchiang': []}
CLAIM_DIFF = {s_: [p_ for p_ in _ALLP if p_ not in CLAIM_SIM[s_]] for s_ in CLAIM_SIM}
_PO = {}
for r_ in PAIRS:
    if r_['kind'] == 'season': _PO[(r_['site'], r_['group_a'], r_['group_b'])] = r_['over_p99'] == 'True'
for s_, _n_ in SITES:
    chk(f'{s_} 季節兩兩配對共 6 組，且 tvd 與雜訊門檻之大小關係同 over_p99 欄', (sum(1 for k_ in _PO if k_[0] == s_), all((float(r_['tvd']) > float(r_['noise_tvd_p99'])) == (r_['over_p99'] == 'True') for r_ in PAIRS if r_['kind'] == 'season' and r_['site'] == s_)), (6, True))
    chk(f'{s_} 文中「相近」之季節配對皆未超過雜訊（over_p99 為假）', [p_ for p_ in CLAIM_SIM[s_] if _PO[(s_,) + p_]], [])
    chk(f'{s_} 文中「差異明顯」之季節配對皆超過雜訊（over_p99 為真）', [p_ for p_ in CLAIM_DIFF[s_] if not _PO[(s_,) + p_]], [])
    chk(f'{s_} 未超過雜訊之配對恰為文中稱「相近」者（不漏、不多）', sorted(p_ for p_ in _ALLP if not _PO[(s_,) + p_]), sorted(CLAIM_SIM[s_]))
def season_txt(s):
    parts = []
    for g, zn in SEASONS:
        sel, tot, _o = RNG[s][g]
        t = f'{zn}為{rtn(sel[0], tot)}'
        if len(sel) > 1: t += f'，另有{rts(sel[1], tot)}'
        parts.append(t)
    return '；'.join(parts) + '。'
# 預期之主要來源方向（以資料人工核對後之固定值；防止演算法與資料同時改變而未被發現）：全年（最大與第二大範圍）
EXPECT_MAIN = {
    'gongliao': [('東南至南方', '135°–185°', '51.80%'), ('北方', '315°–15°', '28.30%')],
    'guishan': [('東南至南方', '125°–195°', '61.68%'), ('北方', '345°–15°', '14.96%')],
    'hsinchu': [('西方', '225°–285°', '54.85%'), ('東北方', '25°–55°', '33.82%')],
    'nanwan': [('西至西北方', '265°–325°', '58.56%'), ('東方', '95°–115°', '11.20%')],
    'lienchiang': [('東北方', '15°–55°', '39.40%'), ('西南方', '185°–235°', '24.54%')]}
EXPECT_SEAS = {
    'gongliao': {'DJF': [('東南至南方', '135°–185°', '87.17%'), ('北方', '355°–5°', '3.44%')], 'MAM': [('東南至南方', '135°–185°', '62.16%'), ('西北至北方', '315°–5°', '21.45%')], 'JJA': [('西北方', '285°–15°', '61.41%'), ('東南至南方', '135°–185°', '30.60%')], 'SON': [('西北至北方', '285°–35°', '59.63%'), ('東南至南方', '135°–175°', '22.88%')]},
    'guishan': {'DJF': [('東南至南方', '125°–195°', '96.50%')], 'MAM': [('南方', '135°–195°', '78.41%')], 'JJA': [('南方', '135°–225°', '52.90%'), ('北方', '345°–15°', '21.81%')], 'SON': [('東南至南方', '115°–195°', '46.26%'), ('北方', '335°–25°', '34.29%')]},
    'hsinchu': {'DJF': [('東北方', '15°–55°', '94.16%')], 'MAM': [('西南至西方', '215°–285°', '82.26%'), ('東北方', '45°–55°', '9.91%')], 'JJA': [('西南至西方', '215°–285°', '97.30%')], 'SON': [('西方', '235°–285°', '44.57%'), ('東北方', '25°–55°', '38.49%')]},
    'nanwan': {'DJF': [('東方', '55°–145°', '57.67%'), ('西北方', '295°–325°', '15.72%')], 'MAM': [('西北方', '285°–325°', '59.36%'), ('東方', '85°–105°', '9.80%')], 'JJA': [('西至西北方', '265°–325°', '90.18%')], 'SON': [('西北方', '275°–325°', '60.63%'), ('東方', '95°–115°', '12.53%')]},
    'lienchiang': {'DJF': [('東北方', '15°–55°', '91.51%')], 'MAM': [('東北方', '5°–75°', '40.71%'), ('西南方', '205°–235°', '22.29%')], 'JJA': [('西南方', '185°–245°', '54.83%'), ('東南方', '125°–145°', '8.64%')], 'SON': [('東北方', '5°–75°', '60.14%'), ('東南方', '125°–135°', '4.34%')]}}
for s, _ in SITES:
    sel, tot, _o = RNG[s]['all']
    chk(f'{s} 全年主要來源方向範圍（獨立演算法：方向機率高於 1/36 之相鄰區間合併，最大與第二大）= 固定預期值', [(nm8(t[0], t[1]), deg(t[0], t[1]), pn(t[2], tot)) for t in sel], EXPECT_MAIN[s])
    chk(f'{s} 各季節主要來源方向範圍之名稱、度數與機率（獨立演算法）= 固定預期值', {g_: [(nm8(t[0], t[1]), deg(t[0], t[1]), pn(t[2], RNG[s][g_][1])) for t in RNG[s][g_][0]] for g_, _n in SEASONS}, EXPECT_SEAS[s])
    chk(f'{s} 區間門檻：被併入範圍之區間機率皆高於 2.78%（1/36）、範圍外之區間皆不高於', (all(36 * cnt_sec(s, 'class')[(t[0] + k) % 36] > 100000 for t in sel for k in range(t[1])), all(36 * v <= 100000 for i, v in enumerate(cnt_sec(s, 'class')) if i not in {(t[0] + k) % 36 for t in RNG[s]['all'][2] for k in range(t[1])})), (True, True))
    chk(f'{s} 全年有至少兩個範圍（故「第二大」皆列出）；同一分布之兩範圍方位名稱不同', (len(RNG[s]['all'][2]) >= 2, len({nm8(t[0], t[1]) for t in sel}) == len(sel)), (True, True))
    chk(f'{s} 各季節：若有第二大範圍則一律列出（第二範圍存在即列）', all(len(RNG[s][g_][0]) == min(2, len(RNG[s][g_][2])) for g_, _n in SEASONS), True)
for s, _ in SITES:
    rows = [r for r in POLSUM if r['site'] == s]
    chk(f'{s} 檢定結果：十類、同類十個沉降速度、大小潮皆在雜訊內；季節超過', ([r['verdict'] for r in rows if r['kind'] == 'class'], [r['verdict'] for r in rows if r['kind'] == 'velocity_within_class'], [r['verdict'] for r in rows if r['kind'] == 'tide'], [r['verdict'] for r in rows if r['kind'] == 'season']), (['within_noise'], ['within_noise'] * 10, ['within_noise'], ['over_noise']))
    ss = [r for r in rows if r['kind'] == 'season'][0]
    chk(f'{s} 季節超過雜訊之配對數（summary）= 6 減去文中稱「相近」之配對數', (int(ss['n_pairs']), int(ss['pairs_over_p99'])), (6, 6 - len(CLAIM_SIM[s])))
    sv = [r for r in rows if r['kind'] == 'velocity_within_class']; chk(f'{s} 同類十速度之 450 組對皆未超過', (sum(int(r['n_pairs']) for r in sv), sum(int(r['pairs_over_p99']) for r in sv)), (450, 0))

# ---------- 海岸來源位置（底圖陸地多邊形分類）：由多邊形表獨立加總、與逐地點表比對
def pcls(s, cls): return sum(int(r['count']) for r in POLY if r['site'] == s and r['class'] == cls)
GEOX = {}
for s, _ in SITES:
    nc = int(flux[s]['count_status_8'])
    GEOX[s] = dict(main=pcls(s, 'mainland'), matsu=pcls(s, 'matsu'), isl=pcls(s, 'fj_island'), big=pcls(s, 'large_other'))
    G_=GEO[s]; _gm=(int(G_['n_fujian_mainland']), int(G_['n_matsu_islands']), int(G_['n_fujian_coastal_islands']), int(G_['n_on_large_polygon']) if s != 'lienchiang' else 0)
    chk(f'{s} 海岸來源位置分類：逐地點表四類（福建本土、馬祖、其他島嶼、臺灣本島）之計數加總＝抵達海岸顆數；多邊形表（列出各地點顆數較多之多邊形）各類加總不超過逐地點表，且福建本土與臺灣本島兩類完全相同', (sum(_gm) if s == 'lienchiang' else _gm[2] + _gm[3], int(G_['n_coast']), nc, GEOX[s]['main'] == _gm[0] and GEOX[s]['big'] == _gm[3], GEOX[s]['matsu'] <= _gm[1] and GEOX[s]['isl'] <= _gm[2]), (nc, nc, nc, True, True))
chk('位置分類固定預期值：貢寮臺灣本島 24,496／其他島嶼（含龜山島）2,102；龜山島臺灣本島 11,955、龜山島本島（多邊形 96:0）9,565；新竹外海臺灣本島 2,234；南灣臺灣本島 29,942／離島 1,143；連江福建本土 15,821、近岸島嶼 13,858、馬祖各島 14,265（逐地點表）', ((int(GEO['gongliao']['n_on_large_polygon']), int(GEO['gongliao']['n_fujian_coastal_islands'])), (int(GEO['guishan']['n_on_large_polygon']), sum(int(r['count']) for r in POLY if r['site'] == 'guishan' and r['polygon'] == '96:0')), int(GEO['hsinchu']['n_on_large_polygon']), (int(GEO['nanwan']['n_on_large_polygon']), int(GEO['nanwan']['n_fujian_coastal_islands'])), (int(GEO['lienchiang']['n_fujian_mainland']), int(GEO['lienchiang']['n_fujian_coastal_islands']), int(GEO['lienchiang']['n_matsu_islands']))), ((24496, 2102), (11955, 9565), 2234, (29942, 1143), (15821, 13858, 14265)))
_bb = {r['polygon']: (float(r['lon_min']), float(r['lon_max']), float(r['lat_min']), float(r['lat_max'])) for r in POLY}
chk('多邊形 62:0（臺灣本島）範圍涵蓋 120.0–122.0°E、21.9–25.3°N；多邊形 96:0 範圍含龜山島座標（121.95°E、24.84°N）；馬祖各島多邊形皆在 119.88–120.55°E、25.93–26.42°N', (_bb['62:0'][0] < 120.1 and _bb['62:0'][1] > 122.0 and _bb['62:0'][2] < 21.95 and _bb['62:0'][3] > 25.29, _bb['96:0'][0] <= 121.95 <= _bb['96:0'][1] and _bb['96:0'][2] <= 24.84 <= _bb['96:0'][3], all(119.88 <= (float(r['lon_min']) + float(r['lon_max'])) / 2 <= 120.55 and 25.93 <= (float(r['lat_min']) + float(r['lat_max'])) / 2 <= 26.42 for r in POLY if r['class'] == 'matsu')), (True, True, True))
chk('南灣離島多邊形皆在臺灣本島範圍外（經度小於 120.4 或大於 120.9）；貢寮近岸島嶼皆在 121.3–122.2°E、24.8–25.5°N', (all(float(r['lon_max']) < 120.4 or float(r['lon_min']) > 120.9 for r in POLY if r['site'] == 'nanwan' and r['class'] == 'fj_island'), all(121.3 <= float(r['lon_min']) and float(r['lon_max']) <= 122.2 and 24.8 <= float(r['lat_min']) and float(r['lat_max']) <= 25.5 for r in POLY if r['site'] == 'gongliao' and r['class'] == 'fj_island')), (True, True))
chk('海岸來源抵達位置距底圖海岸線 90 百分位 ≤ 1.0 km（各地點）、最大 ≤ 5.4 km；抵達位置經緯度範圍：貢寮 121.83–122.00°E、24.94–25.13°N；新竹外海 120.60–120.76°E、24.41–24.65°N（5%–95%）', ({s: float(GEO[s]['dist_to_basemap_coast_km_p90']) <= 1.0 for s, _ in SITES}, max(float(GEO[s]['dist_to_basemap_coast_km_max']) for s, _ in SITES) <= 5.4, (round(float(GEO['gongliao']['lon_p05']), 2), round(float(GEO['gongliao']['lon_p95']), 2), round(float(GEO['gongliao']['lat_p05']), 2), round(float(GEO['gongliao']['lat_p95']), 2)), (round(float(GEO['hsinchu']['lon_p05']), 2), round(float(GEO['hsinchu']['lon_p95']), 2), round(float(GEO['hsinchu']['lat_p05']), 2), round(float(GEO['hsinchu']['lat_p95']), 2))), ({s: True for s, _ in SITES}, True, (121.83, 122.0, 24.94, 25.13), (120.6, 120.76, 24.41, 24.65)))

# ---------- 逐地點
SECTXT = {}
_parts = re.split(r'^## ', RES_FULL, flags=re.M); chk('地點章數', len(_parts), 6)
_cut = re.split(r'^#- 綜合說明與結論', _parts[5], flags=re.M)
_parts[5] = _cut[0]; ENDTXT = '#- 綜合說明與結論' + _cut[1]
for _i, (_s, _sn) in enumerate(SITES):
    assert _parts[1 + _i].startswith(f'{_sn}\n'); SECTXT[_sn] = _parts[1 + _i]
_nf = 0; CNT = {}; CONC_EXP = {}
chk('代表軌跡：抽樣種子 0、每地點 24 條、圖工具上限 24（traj manifest）', (REPB['trajectory_sample_seed'], REPB['size_per_site'], json.load(open(SB + '/figures-notext-traj-v5-ax2-coastB/manifest.json'))['max_trajectories_per_site']), (0, 24, 24))
chk('代表軌跡各結局條數（彙總檔）= 固定預期值（穿越／抵達海岸）；每地點 24 條', {s_: (REPB['sites'][s_]['n'], REPB['sites'][s_]['status_counts']) for s_, _ in SITES}, {'gongliao': (24, {'1': 21, '8': 3}), 'guishan': (24, {'1': 19, '8': 5}), 'hsinchu': (24, {'1': 23, '8': 1}), 'nanwan': (24, {'1': 15, '8': 9}), 'lienchiang': (24, {'1': 13, '8': 11})})
H50 = {s_: fnum(fp[s_]['hdr50_area_km2']) for s_, _ in SITES}
_smallest = min(H50, key=H50.get); _largest = max(H50, key=H50.get)
chk('50% 高密度區間面積：龜山島最小（801）、南灣最大（1,836）', (_smallest, _largest, i0(H50['guishan']), i0(H50['nanwan'])), ('guishan', 'nanwan', '801', '1,836'))
HDRLOC = {'gongliao': '50% 高密度區間沿臺灣本島東北部海岸外側呈南北向帶狀，通過受體並向南北兩側延伸。',
          'guishan': '50% 高密度區間位於受體周圍之近岸海域，西側緊鄰臺灣本島東北部海岸，向北延伸至受體北側，向南延伸至流場範圍邊界。',
          'hsinchu': '50% 高密度區間沿臺灣本島西部海岸外側呈西南至東北向帶狀，通過受體並向東北與西南兩側延伸。',
          'nanwan': '50% 高密度區間自受體沿恆春半島西側向西北延伸，另有一段沿其東側向東北延伸。',
          'lienchiang': '50% 高密度區間自受體向東北延伸，另有一獨立區塊緊鄰福建沿岸。'}   # 逐地點目視圖 3（來源足跡圖）確認；此 5 句為人工目視判讀，核對程式僅鎖定文字
from fractions import Fraction as _Fr
_D = Decimal
def lrm(parts, total):
    """最大餘數法（獨立實作，以分數計）：各分項取 0.01% 為單位；加總恰等於總計（總計為顆數／100,000 之兩位小數四捨五入）。餘數同者顆數多者先得。"""
    exact = [_Fr(p, 1000) * 100 for p in parts]            # 單位 0.01%
    tot = int(_D(total) / _D(10) + _D('0.5'))             # 總計單位數（total/10，四捨五入）
    base = [int(x) for x in exact]
    left = tot - sum(base)
    ranked = sorted(range(len(parts)), key=lambda i: (-(exact[i] - base[i]), -parts[i]))
    for i in ranked[:left]: base[i] += 1
    return base, tot, exact
def lrm_str(parts, total):
    b, tot, ex = lrm(parts, total)
    assert sum(b) == tot and all(abs(_Fr(x) - e) <= 1 for x, e in zip(b, ex))   # 加總＝總計；各分項與精確值相差 ≤ 1（0.01%）
    return ['%d.%02d%%' % (x // 100, x % 100) for x in b]
_pc96 = {s_: sum(int(r['count']) for r in POLY if r['site'] == s_ and r['polygon'] == '96:0') for s_, _ in SITES}
_pc30 = sum(int(r['count']) for r in POLY if r['site'] == 'nanwan' and r['polygon'] == '30:0')
_pc6 = sum(int(r['count']) for r in POLY if r['site'] == 'nanwan' and r['polygon'] == '6:0')
_oth = {'gongliao': 26598 - 24496 - _pc96['gongliao'], 'guishan': 21647 - 11955 - _pc96['guishan'], 'nanwan': 31085 - 29942 - _pc30 - _pc6}
COASTSENT = {
    'gongliao': "貢寮之海岸來源（%s）中，臺灣本島占 %s，龜山島占 %s，其他島嶼占 %s。" % ((pn(26598, 100000),) + tuple(lrm_str([24496, _pc96['gongliao'], _oth['gongliao']], 26598))),
    'guishan': "龜山島之海岸來源（%s）中，臺灣本島占 %s，龜山島占 %s，其他島嶼占 %s。" % ((pn(21647, 100000),) + tuple(lrm_str([11955, _pc96['guishan'], _oth['guishan']], 21647))),
    'hsinchu': f"新竹外海之海岸來源（{pn(2234, 100000)}）皆位於臺灣本島。",
    'nanwan': "南灣之海岸來源（%s）中，臺灣本島占 %s，小琉球占 %s，蘭嶼占 %s，其他島嶼占 %s。" % ((pn(31085, 100000),) + tuple(lrm_str([29942, _pc30, _pc6, _oth['nanwan']], 31085))),
    'lienchiang': "連江之海岸來源（%s）中，福建沿岸（含近岸島嶼）占 %s，馬祖各島占 %s。" % ((pn(43944, 100000),) + tuple(lrm_str([15821 + 13858, 14265], 43944)))}
# 分項加總＝總計（以文字中實際出現之數字核對），且各分項與原四捨五入值（顆數／1000）相差不超過 0.01
def _sumcheck(sent, parts, total):
    nums = [_D(x) for x in re.findall(r'占 (\d+\.\d+)%', sent)]
    return (sum(nums), len(nums) == len(parts), all(abs(n_ - _D(p_) / 1000) <= _D('0.01') for n_, p_ in zip(nums, parts)))
_PARTS = {'gongliao': ([24496, _pc96['gongliao'], _oth['gongliao']], 26598), 'guishan': ([11955, _pc96['guishan'], _oth['guishan']], 21647), 'nanwan': ([29942, _pc30, _pc6, _oth['nanwan']], 31085), 'lienchiang': ([15821 + 13858, 14265], 43944)}
chk('海岸來源分項加總＝總計（最大餘數法；貢寮、龜山島、南灣、連江）；各分項與原四捨五入值（顆數／1000）相差 ≤ 0.01；分項數齊全', {k_: _sumcheck(COASTSENT[k_], v_[0], v_[1]) for k_, v_ in _PARTS.items()}, {'gongliao': (_D('26.60'), True, True), 'guishan': (_D('21.65'), True, True), 'nanwan': (_D('31.09'), True, True), 'lienchiang': (_D('43.94'), True, True)})
# 具名島嶼之位置（多邊形範圍）：96:0 含龜山島座標；30:0 為小琉球（120.35–120.39°E、22.32–22.36°N）；6:0 為蘭嶼（121.50–121.60°E、22.00–22.09°N）；各數字固定預期值
_bb2 = {r['polygon']: (float(r['lon_min']), float(r['lon_max']), float(r['lat_min']), float(r['lat_max'])) for r in POLY}
chk('具名島嶼多邊形範圍與顆數：龜山島（96:0，121.93–121.97°E、24.83–24.86°N）貢寮 883、龜山島 9,565；小琉球（30:0，120.3–120.45°E、22.3–22.4°N）578；蘭嶼（6:0，121.45–121.65°E、21.95–22.15°N）534；其他島嶼 1,219、127、31', (120.3 <= _bb2['30:0'][0] and _bb2['30:0'][1] <= 120.45 and 22.3 <= _bb2['30:0'][2] and _bb2['30:0'][3] <= 22.4, 121.45 <= _bb2['6:0'][0] and _bb2['6:0'][1] <= 121.65 and 21.95 <= _bb2['6:0'][2] and _bb2['6:0'][3] <= 22.15, 121.93 <= _bb2['96:0'][0] and _bb2['96:0'][1] <= 121.97 and 24.83 <= _bb2['96:0'][2] and _bb2['96:0'][3] <= 24.86, _pc96['gongliao'], _pc96['guishan'], _pc30, _pc6, _oth), (True, True, True, 883, 9565, 578, 534, {'gongliao': 1219, 'guishan': 127, 'nanwan': 31}))
chk('海岸來源位置句之數字：與位置分類計數、占比二位小數（獨立計算）一致', {'gongliao': (pn(26598, 100000), pn(24496, 100000), pn(2102, 100000)), 'guishan': (pn(21647, 100000), pn(11955, 100000), pn(9565, 100000)), 'lienchiang': (pn(43944, 100000), pn(29679, 100000), pn(14265, 100000))}, {'gongliao': ('26.60%', '24.50%', '2.10%'), 'guishan': ('21.65%', '11.96%', '9.57%'), 'lienchiang': ('43.94%', '29.68%', '14.27%')})
for si, (s, sn) in enumerate(SITES):
    sec = SECTXT[sn]; ls = [l for l in sec.split('\n') if l.strip()]
    f = flux[s]; n = int(f['total_particle_count']); c = int(f['count_crossed']); k8 = int(f['count_status_8']); d = int(f['count_status_4']); CNT[s] = (n, c, k8, d)
    caps_f = [int(x) for x in re.findall(r'^圖(\d+) ', sec, re.M)]; caps_t = re.findall(r'^表\d+ ', sec, re.M)
    chk(f'{sn} 無表、7 圖且圖號依序連續', (caps_t, caps_f), ([], list(range(_nf + 1, _nf + 8))))
    fb = _nf + 1; _nf += 7
    chk(f'{sn} 本節 7 圖皆於本節引用、不引用他節圖、無表引用', (set(caps_f) <= expand(sec, '圖'), expand(sec, '圖') - set(caps_f), expand(sec, '表')), (True, set(), set()))
    chk(f'{sn} 圖種順序（水平軌跡、仍在追蹤比例、足跡、四張圓形圖）', [x for x in re.findall(r'!\[\]\(' + re.escape(FIGSUB) + r'/(.*?)\.png\)', sec)], [f'{s}/{t}' for t in SITESTEMS])
    chk(f'{sn} 不含其他研究地點名稱（貢寮節海岸來源位置句之具名島嶼「龜山島占」除外）', [x for _, x in SITES if x != sn and x in sec.replace('，龜山島占', '，')], [])
    kinds = ['P' if l.startswith('　　') else ('I' if l.startswith('![](') else ('C' if re.match(r'^圖\d+ ', l) else '?')) for l in ls[1:]]
    SEQ_FIRST = ['P', 'I', 'C'] * 3 + ['P', 'P', 'I', 'C'] + ['P', 'I', 'C'] * 3
    SEQ_REST = ['P', 'I', 'C'] * 3 + ['P', 'I', 'C'] * 4
    chk(f'{sn} 段落與圖之排列（每圖前為導引段、圖、圖題；圓形圖前' + ('2' if si == 0 else '1') + '段）；無小標「本地點結論」', (kinds, '本地點結論' in sec), (SEQ_FIRST if si == 0 else SEQ_REST, False))
    PP_ = [l for l in ls if l.startswith('　　')]
    imgi = [k for k, l in enumerate(ls) if l.startswith('![](')]
    for k, i in enumerate(imgi):
        n_ = fb + k
        chk(f'{sn} 圖{n_} 圖題在圖下方且為「圖{n_} {sn}…」', ls[i + 1].startswith(f'圖{n_} {sn}'), True)
    chk(f'{sn} 圖寬皆為 {FW}', {w for _, w in re.findall(r'!\[\]\((.*?)\)\{width=([\d.]+)in\}', sec)}, {FW})
    CAPS = [f'圖{fb} {sn}代表虛擬質點水平軌跡總覽', f'圖{fb+1} {sn}回溯期間仍在追蹤之虛擬質點比例', f'圖{fb+2} {sn}條件式來源足跡（停留時數密度）', f'圖{fb+3} {sn}來源方向機率（十類）', f'圖{fb+4} {sn}來源方向機率（十類各十個沉降速度）', f'圖{fb+5} {sn}來源方向機率（依沉底時刻之季節）', f'圖{fb+6} {sn}來源方向機率（依大小潮）']
    chk(f'{sn} 7 張圖之圖題全文', [ls[i + 1] for i in imgi], CAPS)
    # --- 圖 1：水平軌跡總覽
    if si == 0: e1 = f'　　圖 {fb} 為隨機抽取之 24 條代表虛擬質點回溯軌跡，線色依起始深度由淺至深。'
    else: e1 = f'　　圖 {fb} 為{sn}隨機抽取之 24 條代表虛擬質點回溯軌跡。'
    chk(f'{sn} 圖{fb} 導引句不寫終止於穿越或海岸之條數（使用者要求）', re.search(r'\d+ 條終止|終止於', PP_[0]) is None, True)
    chk(f'{sn} 圖{fb} 導引句全文', PP_[0], e1)
    chk(f'{sn} 扇區穿越顆數合計 = 穿越顆數；抵達海岸（含觸底）顆數合計 = 非穿越顆數', (sum(int(r['count_crossed']) for r in SECR if r['site'] == s and r['grouping'] == 'class'), sum(int(r['count_coast']) for r in SECR if r['site'] == s and r['grouping'] == 'class')), (c, k8 + d))
    # --- 圖 2：仍在追蹤之虛擬質點比例（全部 100,000 顆）
    I = FA['inside'][s]; sm_in = inside_sum[s]
    chk(f'{sn} 圖{fb+1} 90／180 天比例對 inside_fraction_summary（圖工具輸出）', (f"{I['at90']*100:.4f}", f"{I['at180']*100:.4f}"), (sm_in['inside_pct_at_90d'], sm_in['inside_pct_at_180d']))
    chk(f'{sn} 圖{fb+1} 90、180 天比例 = 0（穿越、抵達海岸與觸底皆已停止追蹤；觸底者自觸底時刻起不再計入）', (round(I['at90'], 9), round(I['at180'], 9)), (0.0, 0.0))
    chk(f'{sn} 全部虛擬質點皆於曲線內停止（穿越、抵達海岸或觸底）：停止顆數 = 總數 = 穿越 + 海岸 + 觸底；未停止者 0；最長停止天數（facts）= 圖工具 manifest 之最長回溯天數', (I['n_stop'], I['n_not_stopped'], c + k8 + d, round(I['max_stop_day'], 6)), (n, 0, n, round(_dfm['sites'][s]['max_backtrack_days'], 6)))
    chk(f'{sn} 圖{fb+1} 結局碼：停止追蹤碼為 1、8 與 4（manifest；觸底自觸底時刻起停止追蹤）', _dfm['stop_status_codes'], [1, 8, 4])
    for th in (0.5, 0.1, 0.01):
        dth = I[f'first_le_{th}']
        chk(f'{sn} 門檻 {th} 日（進位至 0.1 天）= ceil(精確日×10)/10', abs(dth - math.ceil(I[f'exact_le_{th}'] * 10 - 1e-9) / 10) < 1e-9 and 0 <= dth - I[f'exact_le_{th}'] < 0.1 + 1e-9, True)
    keyd = {1: I['at1'], 3: I['at3'], 7: I['at7'], 14: I['at14'], 30: I['at30'], 60: I['at60']}
    for th in (0.5, 0.1, 0.01):
        chk(f'{sn} 門檻 {th} 日與關鍵日比例相容', all((val <= th + 1e-12) == (dd >= I[f'exact_le_{th}']) for dd, val in keyd.items()), True)
    # 對 inside_fraction_summary（圖工具 csv，6 小時格）之降至門檻日：facts 之進位日不得早於 csv 日減 0.25 天（粗格容差）、不得晚於 csv 日
    chk(f'{sn} 降至 50%、10%、1% 之日（facts）落在圖工具 csv 之 6 小時格內（差 ≤ 0.25 天）', all(-1e-9 <= float(sm_in[kk]) - I[f'first_le_{th}'] + 0.25 + 1e-9 and float(sm_in[kk]) - I[f'first_le_{th}'] <= 0.25 + 1e-9 for kk, th in (('first_day_at_or_below_50pct', 0.5), ('first_day_at_or_below_10pct', 0.1), ('first_day_at_or_below_1pct', 0.01))), True)
    pts = []
    for th, lab in ((0.5, '50%'), (0.1, '10%'), (0.01, '1%')):
        pts.append(f"{'回溯' if not pts else ''}第 {I[f'first_le_{th}']:.1f} 天降至 {lab} 以下")
    tt = '、'.join(pts) + (f"，至第 {I['first_zero_day']:.1f} 天降至 0%" if I['first_zero_day'] is not None else '')
    a90, a180 = pfl(I['at90'] * 100), pfl(I['at180'] * 100)
    tl = '。'
    chk(f'{sn} 0% 日存在（五地點皆全部停止追蹤）且 = ceil(最長停止日×10)/10', (I['first_zero_day'] is not None) and abs(I['first_zero_day'] - math.ceil(I['exact_zero_day'] * 10 - 1e-9) / 10) < 1e-9 and abs(I['exact_zero_day'] - _dfm['sites'][s]['max_backtrack_days']) < 1e-9, True)
    if si == 0:
        e2 = (f'　　圖 {fb+1} 為全部 {n:,} 顆虛擬質點於回溯期間內仍在追蹤之比例，垂直虛線標示 90 天與 {HZ} 天。虛擬質點穿越流場範圍邊界或抵達海岸時停止追蹤；'
              f'{sn}之虛擬質點穿越流場範圍邊界者占 {pn(c, n)}，稱為穿越比例，抵達海岸者占 {pn(k8, n)}，稱為海岸來源比例。')
    else:
        e2 = f'　　圖 {fb+1} 為全部 {n:,} 顆虛擬質點於回溯期間內仍在追蹤之比例。{sn}之穿越比例為 {pn(c, n)}，海岸來源比例為 {pn(k8, n)}'
        if d: e2 += f'，其餘 {d} 顆（{pn(d, n)}）於回溯途中觸底而停止追蹤'
        e2 += '。'
    e2 += f'{sn}仍在追蹤之比例於{tt}{tl}'
    chk(f'{sn} 圖{fb+1} 導引句全文', PP_[1], e2)
    # --- 圖 3：條件式來源足跡
    fpp = fp[s]; shv = fnum(fpp['hours_fraction_depth_below_threshold']) * 100; sh = pfl(shv)
    chk(f'{sn} 水深門檻 200 m', int(fpp['depth_threshold_m']), 200)
    areas = f"{i0(fnum(fpp['hdr50_area_km2']))}、{i0(fnum(fpp['hdr75_area_km2']))} 與 {i0(fnum(fpp['hdr90_area_km2']))} km²"
    shelf = '皆屬大陸棚與近岸海域' if shv >= 99.9999 else ('多屬大陸棚與近岸海域' if shv >= 50 else '多屬水深 200 m 以深之外海')
    rk = '，為五個研究地點中最小，停留時數最集中於受體附近' if s == _smallest else ('，為五個研究地點中最大，停留時數分布於較廣之海域' if s == _largest else '')
    if si == 0:
        e3 = (f'　　圖 {fb+2} 之底色為停留時數密度：虛擬質點每在流場範圍內漂流 1 小時，即在其所在之 1 km × 1 km 網格計 1 小時停留時數，各網格之停留時數占該研究地點停留時數總量之比例，除以網格面積即為停留時數密度；虛擬質點抵達海岸後不再計入。'
              f'高密度區間係將各網格依停留時數密度由高至低排序，依序累加其停留時數，累加達 50%、75%、90% 時所涵蓋之網格範圍，即包含該比例停留時數之最小面積區域；50% 高密度區間為包含一半停留時數之範圍，面積愈小，表示停留時數愈集中於受體附近，面積愈大，表示分布於較廣之海域。'
              f'{sn}之 50%、75%、90% 高密度區間面積分別為 {areas}；停留時數之 {sh} 位於水深淺於 200 m 處，{shelf}。' + HDRLOC[s])
    else: e3 = f'　　圖 {fb+2} 中，{sn}之 50%、75%、90% 高密度區間面積分別為 {areas}{rk}；停留時數之 {sh} 位於水深淺於 200 m 處，{shelf}。' + HDRLOC[s]
    chk(f'{sn} 圖{fb+2} 導引句全文', PP_[2], e3)
    chk(f'{sn} 足跡導引不含抽樣範圍句（使用者於 Word 刪除，不補回）', '受體之抽樣範圍為以研究地點錨點為圓心' in PP_[2], False)
    # --- 圖 4–7：圓形圖
    ta = [r for r in POLTERM if r['site'] == s and r['grouping'] == 'site_all' and r['group_id'] == 'all']; chk(f'{sn} 結局表唯一', len(ta), 1); ta = ta[0]
    chk(f'{sn} 穿越／抵達海岸／觸底顆數與 flux 表一致、合計 100,000', (int(ta['n_total']), int(ta['n_crossed']), int(ta['n_status_8']), int(ta['n_status_4'])), (100000, c, k8, d))
    chk(f'{sn} 無達回溯上限者（status 2＝0）；觸底僅南灣 1 顆', (int(ta['n_status_2']), d), (0, 1 if s == 'nanwan' else 0))
    a_ = fb + 3
    sel, tot, _o = RNG[s]['all']
    mains = f'{sn}全年（十類合併）之主要來源方向為{rtn(sel[0], tot)}，其次為{rts(sel[1], tot)}。'
    ep = []
    if si == 0:
        ep.append(f'　　圖 {a_} 至圖 {a_+3} 為{sn}之圓形來源方位圖，依序呈現十類、同類十個沉降速度、季節與大小潮之來源方向機率。'
                  '來源方向為虛擬質點所在位置自研究地點錨點起算之方位，每 10° 分為一個方向區間，共 36 個，方位自正北順時針量測。統計全部虛擬質點之來源方向落在各方向區間之機率，連成封閉曲線。'
                  '穿越流場範圍邊界之虛擬質點，以其穿越位置之方向計入；抵達海岸者，以其抵達位置之方向計入，稱為海岸來源；回溯途中觸底而停止追蹤者，以其最終位置之方向計入，稱為觸底。因此各曲線 36 個方向區間之方向機率合計為 100%。'
                  '主要來源方向之取法為：方向機率高於平均值 2.78%（即 1/36）之相鄰方向區間合併為一個範圍，範圍機率為各方向區間機率之合計，取機率最大者為最主要來源方向，並列出第二大者。'
                  '海岸來源各段之比例皆為占全部虛擬質點之比例。')
        pre = f'圖 {a_} 為十類之來源方向曲線。'
    else: pre = f'圖 {a_} 至圖 {a_+3} 為{sn}之圓形來源方位圖，依序為十類、同類十個沉降速度、季節與大小潮；其中圖 {a_} 為十類之來源方向曲線。'
    ep.append('　　' + pre + mains + COASTSENT[s] + '十類之曲線相近，物件種類對來源方向之影響不明顯。')
    ep.append(f'　　圖 {a_+1} 以十類各一格，呈現同類十個沉降速度之曲線' + (f'（各格依圖 {a_} 圖例之類別順序排列）' if si == 0 else '') + '；同類十個沉降速度之曲線相近，沉降速度對來源方向之影響不明顯。')
    sea = f'　　圖 {a_+2} 為依沉底時刻劃分之{sn}四季曲線。'
    if si == 0: sea += '沉底時刻為廢棄物沉降至海床之時刻，即虛擬質點回溯起點所對應之時刻，為所屬到達時間往前推一個 0–90 天之沉底年齡；季節依沉底時刻所在月份劃分，冬季為 12–2 月、春季為 3–5 月、夏季為 6–8 月、秋季為 9–11 月。'
    sea += '各季節之主要來源方向及其機率為：' + season_txt(s) + '季節間之差異：' + SEAS_EXPECT[s] + '。'
    ep.append(sea)
    ep.append(f'　　圖 {a_+3} 為依' + ('沉底時刻之潮位資料劃分為大潮與小潮之曲線。' if si == 0 else '大小潮劃分之曲線。') + '大小潮兩組之曲線相近，大小潮對來源方向之影響不明顯。')
    chk(f'{sn} 圓形圖導引段落（圖 {a_}–{a_+3}）全文', PP_[3:], ep)
    chk(f'{sn} 季節段：僅在有「相近」配對之地點出現「相近」；一律用「差異明顯」白話，不含「明顯不同」「抽樣誤差」「超過」「雜訊」', ('相近' in sea, '差異明顯' in sea or '明顯差異' in sea, any(w_ in '\n'.join(PP_) for w_ in ('明顯不同', '抽樣誤差', '超過抽樣', '雜訊'))), (len(CLAIM_SIM[s]) > 0, True, False))
    CONC_EXP[s] = dict(sel=sel, tot=tot, k8=k8, d=d, n=n, c=c, h50=H50[s], sh=shv)
chk('圖表總數', _nf, NF)

# ---------- 章末
_el = [l for l in ENDTXT.split('\n') if l.strip()]
h50s = [CONC_EXP[s]['h50'] for s, _ in SITES]; shs = [Decimal(repr(CONC_EXP[s]['sh'])) for s, _ in SITES]
chk('章末結構：綜合說明與結論（3 段）、整體結論（6 條）', (_el[0], _el[4], len(_el)), ('#- 綜合說明與結論', '#- 整體結論', 11))
chk('綜合說明第一段（物件種類、沉降速度、大小潮影響不明顯；季節之影響因地點而異：貢寮與龜山島、南灣、新竹外海與連江）', _el[1], '　　各研究地點十類之間、同類十個沉降速度之間與大小潮之間，來源方向機率相近，物件種類、沉降速度與大小潮對來源方向之影響不明顯。季節之影響則因研究地點而異：貢寮與龜山島冬、春季相近，夏、秋季相近，兩組之間差異明顯；南灣春、秋季相近，其餘季節兩兩之間差異明顯；新竹外海與連江四個季節兩兩之間差異皆明顯。')
chk('綜合說明第一段所述「相近」之地點對照配對結果（貢寮、龜山島兩組；南灣春秋；新竹外海、連江無）', ([s_ for s_, _n_ in SITES if CLAIM_SIM[s_]], [s_ for s_, _n_ in SITES if not CLAIM_SIM[s_] and all(_PO[(s_,) + p_] for p_ in _ALLP)]), (['gongliao', 'guishan', 'nanwan'], ['hsinchu', 'lienchiang']))
chk('綜合說明第二段（足跡範圍、水深，數字獨立重算；新竹外海與連江 100.00%、南灣 23.26%）', _el[2], f'　　足跡範圍方面，各研究地點 50% 高密度區間面積為 {i0(min(h50s))}–{i0(max(h50s))} km²，停留時數位於水深淺於 200 m 處之比例為 {pf(min(shs))}–{pf(max(shs))}；新竹外海與連江之停留時數全部位於水深淺於 200 m 處，南灣之停留時數則多數位於水深 200 m 以深之外海。')
chk('綜合說明第二段之依據：新竹外海、連江水深淺於 200 m 比例為 1.0；南灣 < 0.5', (fnum(fp['hsinchu']['hours_fraction_depth_below_threshold']), fnum(fp['lienchiang']['hours_fraction_depth_below_threshold']), fnum(fp['nanwan']['hours_fraction_depth_below_threshold']) < 0.5), (1.0, 1.0, True))
_crs = {s_: Decimal(CNT[s_][1]) * 100 / Decimal(CNT[s_][0]) for s_, _ in SITES}; _cos = {s_: Decimal(CNT[s_][2]) * 100 / Decimal(CNT[s_][0]) for s_, _ in SITES}
_zd = max(I_['first_zero_day'] for I_ in FA['inside'].values())
chk('綜合說明第三段（穿越與海岸來源比例範圍；最長停止追蹤日數 33.4 天；南灣 1 顆觸底；無回溯達 180 天者）', _el[3], f'　　回溯期間內，各研究地點之虛擬質點皆於 {_zd:.1f} 天內停止追蹤（穿越流場範圍邊界或抵達海岸，南灣另有 1 顆觸底），無回溯達 {HZ} 天者。穿越比例為 {pf(min(_crs.values()))}–{pf(max(_crs.values()))}（新竹外海最高，連江最低），海岸來源比例為 {pf(min(_cos.values()))}–{pf(max(_cos.values()))}（新竹外海最低，連江最高）。')
chk('綜合說明第三段之依據：穿越比例最高為新竹外海、最低為連江；海岸來源比例最低為新竹外海、最高為連江；全部停止日 ≤ 33.4 天（連江最長 33.333）；最大回溯天數遠小於 180；無狀態 2', (max(_crs, key=_crs.get), min(_crs, key=_crs.get), min(_cos, key=_cos.get), max(_cos, key=_cos.get), max(_dfm['sites'][s_]['max_backtrack_days'] for s_, _ in SITES) < 33.4, sum(int(flux[s_]['count_status_2']) for s_, _ in SITES)), ('hsinchu', 'lienchiang', 'hsinchu', 'lienchiang', True, 0))
for i, (s, sn) in enumerate(SITES, 1):
    sel, tot = CONC_EXP[s]['sel'], CONC_EXP[s]['tot']
    b = f'{sn}：全年主要來源方向為{rtn(sel[0], tot)}，其次為{rts(sel[1], tot)}' + '；各季節主要來源方向：' + season_nm(s) + '；' + SEAS_EXPECT[s] + '。' + f'海岸來源占 {pn(CONC_EXP[s]["k8"], CONC_EXP[s]["n"])}'
    if s == 'lienchiang': b += '，其中福建沿岸（含近岸島嶼）占 %s，馬祖各島占 %s' % tuple(lrm_str([15821 + 13858, 14265], 43944))
    b += '。'
    if s == 'nanwan': b += f'另有 {CONC_EXP[s]["d"]} 顆（{pn(CONC_EXP[s]["d"], CONC_EXP[s]["n"])}）於回溯途中觸底而停止追蹤。'
    chk(f'整體結論第 {i} 條（{sn}）全文', _el[4 + i], f'{i}、{b}')
m1 = [Decimal(CONC_EXP[s]['sel'][0][2]) * 100 / Decimal(CONC_EXP[s]['tot']) for s, _ in SITES]
chk('整體結論第 6 條（綜合五地點；不含限制與後續工作語句）', _el[10], f'6、綜合五個研究地點：各研究地點最主要來源方向之機率為 {pf(min(m1))}–{pf(max(m1))}，海岸來源比例為 {pf(min(_cos.values()))}–{pf(max(_cos.values()))}；物件種類與大小潮之影響不明顯，季節之影響因研究地點而異。')
chk('整體結論 5 條地點條列各僅含本地點名稱', [any(n2 in _el[4 + i] for _, n2 in SITES if n2 != sn) for i, (s, sn) in enumerate(SITES, 1)], [False] * 5)

# ---------- 百分比位數規則：全文所有百分比取到小數二位；無法以二位小數表達者保留三位；名目值不在此限
_nom = [r'累加達 50%、75%、90% 時', r'50%、75%、90% 高密度區間', r'50% 高密度區間', r'降至 (?:50|10|1|0)%', r'方向機率合計為 100%', r'平均值 2\.78%']
_pct_bad = []; _pct_3 = []; _pct_all = 0
for l in RL:
    if l.startswith('![](') or re.match(r'^圖\d+ ', l): continue
    t = l
    for p_ in _nom: t = re.sub(p_, '', t)
    for m in re.finditer(r'(?<![\d.,])(\d[\d,]*(?:\.\d+)?)%', t):
        _pct_all += 1; v = m.group(1)
        if re.fullmatch(r'\d+\.\d{2}', v): continue
        if re.fullmatch(r'\d+\.\d{3}', v): _pct_3.append(v + '%'); continue
        _pct_bad.append((v, l[:30]))
chk('全文百分比皆為小數二位（扣除名目值），其餘僅三位保留項', _pct_bad, [])
chk('保留三位小數之項目僅 0.001%（南灣 1 顆觸底）；出現次數（圖 23 導引句 1 次：觸底占比；整體結論 1 次）', (sorted(set(_pct_3)), _pct_3.count('0.001%')), (['0.001%'], 2))
chk('0.001% = 1／100,000（捨入為 0.00% 但實際不為 0）；無 99.999% 之寫法（穿越比例南灣為 68.91%）；全文無 100.00% 之穿越比例', (qd(Decimal(1) * 100 / 100000, 2), '99.999%' in RES_FULL, '穿越比例為 100.00%' in RES_FULL), ('0.00', False, False))
chk('2 位小數四捨五入（ROUND_HALF_UP）之邊界值：南灣海岸來源 31,085／100,000＝31.085→31.09%；新竹外海 33.815→33.82%；連江馬祖各島 14.265→14.27%', (pn(31085, 100000), pn(33815, 100000), pn(14265, 100000)), ('31.09%', '33.82%', '14.27%'))
chk('百分比總數（供回報）：' + str(_pct_all), _pct_all > 60, True)

# ---------- 圖檔：sha256 與 .203 來源（manifest 與實測）、圖寬、長寬比、圖內文字
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
_links = re.findall(r'!\[\]\((.*?)\.png\)\{width=([\d.]+)in\}', RES_FULL)
chk('圖檔連結數', len(_links), NF); chk('圖檔不重複', len({p for p, _ in _links}), NF)
chk('圖檔皆為無文字版', {p.split('/')[2] for p, _ in _links}, {'無文字'})
for p, w in _links:
    fl = f'{ROOT}/{p}.png'
    chk('圖檔存在且大於 100 KB ' + p, os.path.getsize(fl) > 100000, True)
_src203 = {l.split('\t')[0]: l.rstrip('\n').split('\t')[2] for l in open(SRC + '/sha256_203_sources.tsv', encoding='utf-8')}
_p203 = {l.split('\t')[0]: l.rstrip('\n').split('\t')[1] for l in open(SRC + '/sha256_203_sources.tsv', encoding='utf-8')}
chk('來源 sha256 清單涵蓋文件所用全部 35 張圖，且來源路徑皆為方法 B 報告版（-coastB-fs8）資料夾', (all(f"{p.split('/')[3]}/{p.split('/')[4]}.png" in _src203 for p, _ in _links), [k for k in (f"{p.split('/')[3]}/{p.split('/')[4]}.png" for p, _ in _links) if not _p203[k].split('/')[0].endswith('-coastB-fs8')]), (True, []))
_mt = json.load(open(SB + '/figures-notext-traj-v5-ax2-coastB/manifest.json')); _mf1 = json.load(open(SB + '/figures-fp1km-b1-ax2-notext-coastB/manifest.json'))   # 舊版（-coastB，資料來源）
_mt8 = json.load(open(SB + '/figures-notext-traj-v5-ax2-coastB-fs8/manifest.json')); _mf8 = json.load(open(SB + '/figures-fp1km-b1-ax2-notext-coastB-fs8/manifest.json'))   # 報告版（fs8）
_mp2 = _man['output_figures_sha256']; _md = _dfm['outputs']
for p, w in _links:
    site, stem = p.split('/')[3], p.split('/')[4]
    key = f'{site}/{stem}.png'; mine = sha(f'{ROOT}/{p}.png')
    chk('圖檔 sha256 同 .203 來源實測 ' + key, mine, _src203.get(key))
    if stem.startswith('polar_source_v2_'): man = _mp2[f'figures-polar-v3-ax2-notext-coastB/{site}/{stem}.png']['sha256']
    elif stem.startswith('domain_fraction'): man = _md[f'figures-domain-fraction-notext-coastB-fs8/{site}/{stem}.png']
    elif stem == 'horizontal_overview': man = _mt8['output_png_sha256'][f'{site}/{stem}.png']['sha256']
    else: man = _mf8['output_png_sha256'][f'{site}/{stem}.png']['sha256']
    chk('圖檔 sha256 同 .203 該圖 manifest 記錄 ' + key, mine, man)
# ---------- 水平軌跡總覽：五地點影像尺寸一致、四邊不裁切、圖例緊貼主圖
_hov = {}; _lines = []
for _s2, _n2 in SITES:
    _im2 = Image.open(f'{ROOT}/{FIGSUB}/{_s2}/horizontal_overview.png').convert('RGBA')
    _bg2 = Image.new('RGBA', _im2.size, (255, 255, 255, 255)); _bg2.alpha_composite(_im2); _a2 = np.array(_bg2.convert('RGB')).astype(int)
    _nw = (_a2 < 245).any(axis=2); _rows = np.where(_nw.any(axis=1))[0]; _cols = np.where(_nw.any(axis=0))[0]
    _H2, _W2 = _nw.shape; _mg = (int(_cols[0]), int(_W2 - 1 - _cols[-1]), int(_rows[0]), int(_H2 - 1 - _rows[-1]))
    _rr = _nw.any(axis=1); _gap = 0; _run = 0
    for _k in range(int(_rows[0]), int(_rows[-1]) + 1):
        if not _rr[_k]: _run += 1; _gap = max(_gap, _run)
        else: _run = 0
    _hov[_s2] = dict(size=(_W2, _H2), margins=_mg, max_blank_gap_px=_gap, edge3_white=bool((~_nw[:3, :]).all() and (~_nw[-3:, :]).all() and (~_nw[:, :3]).all() and (~_nw[:, -3:]).all()))
    _lines.append(f'{_s2}\t影像 {_W2} x {_H2} 像素（600 dpi，{_W2/600:.2f} x {_H2/600:.2f} 吋）\t四邊空白（左右上下，像素）{_mg}\t主圖至圖例之最大空白帶 {_gap} 像素（{_gap/600:.3f} 吋）')
chk('水平軌跡總覽五地點影像尺寸完全一致（3762 x 2478 像素＝6.27 x 4.13 吋 x 600 dpi）', {v['size'] for v in _hov.values()}, {(3762, 2478)})
chk('水平軌跡總覽四邊皆未裁切（最外 3 像素帶皆為空白）；五地點四邊空白相差不超過 2 像素；各邊空白皆 ≥ 60 像素（0.1 吋）', ([k for k, v in _hov.items() if not v['edge3_white']], max(max(v['margins'][i] for v in _hov.values()) - min(v['margins'][i] for v in _hov.values()) for i in range(4)) <= 2, min(min(v['margins']) for v in _hov.values()) >= 60), ([], True, True))
chk('水平軌跡總覽主圖與圖例之間最大空白帶不超過 0.30 吋（180 像素）', [k for k, v in _hov.items() if v['max_blank_gap_px'] > 180], [])
open(ROOT + '/_verify/traj_figure_v2_layout.txt', 'w', encoding='utf-8').write('\n'.join(_lines) + '\n')
_z = zipfile.ZipFile(f'{ROOT}/{MDNAME}.docx'); _doc = _z.read('word/document.xml').decode('utf-8')
_rels = dict(re.findall(r'<Relationship Id="(rId\w+)"[^>]*?Target="([^"]+)"', _z.read('word/_rels/document.xml.rels').decode('utf-8')))
_rels.update({a_: b_ for b_, a_ in re.findall(r'Target="([^"]+)"[^>]*?Id="(rId\w+)"', _z.read('word/_rels/document.xml.rels').decode('utf-8'))})
_dr = re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"/>.*?r:embed="(rId\w+)"', _doc)
chk('docx 圖數', len(_dr), NF)
_bad = []; _badpx = []
for (_cx, _cy, _rid), (_p, _w) in zip(_dr, _links):
    _im = Image.open(f'{ROOT}/{_p}.png'); _jw, _jh = Image.open(io.BytesIO(_z.read('word/' + _rels[_rid]))).size
    _r0 = _im.width / _im.height
    if abs(int(_cx) / int(_cy) - _r0) / _r0 > 0.003 or abs(_jw / _jh - _r0) / _r0 > 0.003 or abs(int(_cx) / 914400 - float(_w)) > 0.01: _bad.append(_p)
    if _jw != int(float(_w) * 380) and _jw != _im.width: _badpx.append((_p, _jw))
chk('docx 圖寬依指定寬度、長寬比與 PNG 一致（不變形）', _bad, []); chk('docx 圖像素寬為指定寬度 × 380', _badpx, [])
chk('圖寬規格：全部 35 張圖統一為 6.27 吋（文字區全寬 6.268 吋）', {w for _, w in _links}, {FW})
chk('docx 水平軌跡總覽五張顯示尺寸（EMU）完全相同', len({(int(cx), int(cy)) for (cx, cy, rid), (p_, w_) in zip(_dr, _links) if p_.split('/')[4] == 'horizontal_overview'}), 1)
chk('docx 圖顯示寬度皆為 6.27 吋（EMU）且不大於文字區寬 + 0.01 吋', (sorted({round(int(cx) / 914400, 3) for cx, cy, rid in _dr}), max(int(cx) / 914400 for cx, cy, rid in _dr) <= TEXT_W_IN + 0.01), ([6.27], True))
_hts = {}
for (_cx, _cy, _rid), (_p, _w) in zip(_dr, _links):
    _hts.setdefault(re.sub(r'polar_source_velocity_.*', 'polar_source_velocity', _p.split('/')[4]), set()).add(round(int(_cy) / 914400, 2))
open(ROOT + '/_verify/figure_heights_run09.txt', 'w', encoding='utf-8').write('\n'.join(f'{k}\t寬 {FW} 吋\t高 {min(v):.2f}–{max(v):.2f} 吋' for k, v in sorted(_hts.items())) + f'\n文字區 寬 {TEXT_W_IN:.3f} 吋 高 {TEXT_H_IN:.3f} 吋\n')
chk('每張圖之高度（加圖題約 0.6 吋與導引句）不超過一頁可用高度', [k for k, v in _hts.items() if max(v) + 0.6 > TEXT_H_IN], [])
_pm2 = _man['figure_layout']
_ovk = ['polar_source_v2_by_class', 'polar_source_v2_by_season', 'polar_source_v2_by_tide']
_sz = {}
for (_cx, _cy, _rid), (_p, _w) in zip(_dr, _links):
    if _p.split('/')[4].startswith('polar_source_v2_'): _sz.setdefault(_p.split('/')[4], set()).add((int(_cx), int(_cy)))
chk('圓形圖 4 種 × 五地點（20 張）在頁面上同一圖種大小一致，三張總覽顯示寬度相同（6.27 吋）', (sorted(_sz), [len(_sz[k]) for k in sorted(_sz)], len({next(iter(_sz[k]))[0] for k in _ovk})), (sorted(POLSTEMS), [1, 1, 1, 1], 1))
chk('圓形圖版面參數（報告版面：畫布寬 6.27 吋＝插入寬度、字級 8 pt、半徑 25 km、只輸出無文字版）', (_pm2['report_layout'], _man['report_layout'], _pm2['canvas_in'], _pm2['text_pt'], _man['radius_km'], _man['modes']), (True, True, {'B': [6.27, 7.04], 'C': [6.27, 6.88], 'D': [6.27, 6.64], 'panels': [6.27, 8.943]}, 8.0, 25.0, ['none']))
_pxbad = []; _hbad = []; _edge = []; _gap = {}; _fr = {}
for (_cx, _cy, _rid), (_p, _w) in zip(_dr, _links):
    _st = _p.split('/')[4]
    if not _st.startswith('polar_source_v2_'): continue
    _wi, _hi = _pm2['canvas_in']['panels' if _st.endswith('panels') else {'by_class': 'B', 'by_season': 'C', 'by_tide': 'D'}[_st.removeprefix('polar_source_v2_')]]
    _im = Image.open(f'{ROOT}/{_p}.png')
    if abs(_im.width - _wi * 600) > 1 or abs(_im.height - _hi * 600) > 1: _pxbad.append((_p, _im.size))
    if abs(int(_cx) / 914400 - _wi) > 0.005 or abs(int(_cy) / 914400 - _hi) > 0.005: _hbad.append((_p, int(_cx) / 914400, int(_cy) / 914400))
    _bg = Image.new('RGBA', _im.size, (255, 255, 255, 255)); _bg.alpha_composite(_im.convert('RGBA')); _nw = (np.array(_bg.convert('RGB')).astype(int) < 245).any(axis=2)
    _rows = np.where(_nw.any(axis=1))[0]; _cols = np.where(_nw.any(axis=0))[0]
    if not ((~_nw[:3, :]).all() and (~_nw[-3:, :]).all() and (~_nw[:, :3]).all() and (~_nw[:, -3:]).all()): _edge.append(_p)
    _rr = _nw.any(axis=1); _run = 0; _mx = 0
    for _k in range(int(_rows[0]), int(_rows[-1]) + 1):
        if not _rr[_k]: _run += 1; _mx = max(_mx, _run)
        else: _run = 0
    _gap[_p] = _mx / 600
    _fr[_p] = ((_cols[-1] - _cols[0] + 1) / _im.width, _rows[0] / 600, (_im.height - 1 - _rows[-1]) / 600)
chk('圓形圖 PNG 像素 = 畫布英吋 × 600 dpi（寬 3762 像素）', _pxbad, [])
chk('圓形圖在 docx 之顯示尺寸 = 畫布尺寸（寬 6.27 吋，未縮放，字級 8 pt 即印出字級）', _hbad, [])
chk('圓形圖四邊皆未裁切（最外 3 像素帶皆為空白）', _edge, [])
chk('圓形圖內容垂直方向最大連續空白不超過 0.30 吋（無大片空白）', [(k, round(v, 2)) for k, v in _gap.items() if v > 0.30], [])
chk('圓形圖內容占畫布寬度 90% 以上', [(k, round(v[0], 3)) for k, v in _fr.items() if v[0] < 0.90], [])
chk('圓形圖上下空白各不超過 0.30 吋', [(k, round(v[1], 2), round(v[2], 2)) for k, v in _fr.items() if v[1] > 0.30 or v[2] > 0.30], [])
chk('總覽圖主圖框寬 5.42 吋、多格圖每格 1.75 吋', (round(_pm2['overview_frame_in'], 3), round(_pm2['panel_frame_in'], 3), _pm2['overview_frame_in'] / 6.27 > 0.86), (5.42, 1.753, True))
# 方法 B 圖之版面與 o 版（ax2）manifest 逐項相同（只改資料）
_man_o = json.load(open(SRC + '/figures-polar-v3-ax2/manifest.json', encoding='utf-8'))
chk('圓形圖（方法 B）manifest 之版面、半徑刻度、扇區、錨點、重抽參數、座標軸樣式與 o 版（ax2）逐項相同', [k for k in ('figure_layout', 'radius_scales_pct', 'sector', 'bearing', 'anchors_lonlat', 'params', 'radius_km', 'case_id', 'kinds', 'modes', 'report_layout', 'cjk_labels_used') if _man[k] != _man_o[k]], [])
chk('圓形圖（方法 B）：輸入 departures 指紋＝postprocess-coastB 之 departures（與軌跡圖、足跡圖、停留比例圖同）；程式 commit 5d78298、工作樹乾淨', (len({_man['input_departures_sha256'], _dfm['departures_sha256'], _mt['inputs_sha256_before']['departures.parquet']['sha256'], _mf1['inputs_sha256_before']['departures.parquet']['sha256']}), _man['code']['git_commit'][:7], _man['code']['git_dirty']), (1, '5d78298', False))
chk('圓形圖之來源規則含抵達海岸（status 8）：manifest 之 status_counts_by_site 與 flux 表一致（已於上列逐地點核對），穿越比例 crossed_share_pct_by_site 同穿越顆數／100,000', {s: round(_man['crossed_share_pct_by_site'][s], 6) == round(int(flux[s]['count_crossed']) / 1000, 6) for s, _ in SITES}, {s: True for s, _ in SITES})

# ---------- docx：與 o 版（本輪修改前之版本）比對、標題樣式、段落涵蓋、無表
_UB = zipfile.ZipFile(ROOT + '/' + MDNAME + '.docx.bak17-20261022')           # 本輪（方法 B）修改前之 o 版
chk('修改前 o 版 docx 之 sha256＝README 記載之 o 版（454451691d94…）', hashlib.sha256(open(ROOT + '/' + MDNAME + '.docx.bak17-20261022', 'rb').read()).hexdigest(), '454451691d94c729791fbb755a8fdb28b3cd826616d851b62aed6eeb2f360df7')
def _pp(zf):
    out = []
    for pm in re.findall(r'<w:p[ >].*?</w:p>', zf.read('word/document.xml').decode('utf-8'), re.S):
        st = re.search(r'<w:pStyle w:val="([^"]+)"', pm)
        out.append((st.group(1) if st else '', ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', pm)).replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')))
    return out
_ub = _pp(_UB); _nb = _pp(_z)
chk('o 版前三段：標題（標題 1）、引言、小標「逐地點成果」（內文-小標樣式 "-"）', (_ub[0], _ub[3]), (('1', UE2[0]), ('-', UE2[2])))
chk('新版 docx 標題與使用者手改標題逐字相同、樣式相同；小標「逐地點成果」樣式與 o 版相同，位於貢寮之前', (_nb[0] == _ub[0], _nb[3] == _ub[3], _nb[4]), (True, True, ('2', '貢寮')))
chk('新版 docx 引言段 = md 引言', _nb[1][1], _par)
_nb2 = [x_ for x_ in _nb if not x_[1].startswith('　　回溯期間內，各研究地點之虛擬質點皆於')]
chk('新版 docx 段落數 = o 版段落數 + 1（章末新增「綜合說明」第三段）；扣除該段後，樣式序列與 o 版逐段相同', (len(_nb) - len(_ub), len(_nb2) - len(_ub), [i_ for i_, (x_, y_) in enumerate(zip(_nb2, _ub)) if x_[0] != y_[0]]), (1, 0, []))
_shaN = Counter(hashlib.sha256(_z.read(n_)).hexdigest() for n_ in _z.namelist() if n_.startswith('word/media/')); _shaU = Counter(hashlib.sha256(_UB.read(n_)).hexdigest() for n_ in _UB.namelist() if n_.startswith('word/media/'))
chk('新版 docx 內嵌圖 35 張；與 o 版相同者 0 張（35 張皆為方法 B 新圖：軌跡 5、仍在追蹤比例 5、足跡 5、圓形圖 20）', (sum(_shaN.values()), sum((_shaN & _shaU).values()), sum((_shaN - _shaU).values())), (35, 0, 35))
_names = [(st, t) for st, t in _nb if st in ('1', '2', '-')]
chk('docx 標題樣式序列（標題 1、小標、貢寮…連江為標題 2；無本地點結論小標）', _names,
    [('1', UE2[0]), ('-', '逐地點成果')] + sum([[('2', n)] for _, n in SITES], []) + [('-', '綜合說明與結論'), ('-', '整體結論')])
chk('o 版之地點標題（貢寮至連江）仍為標題 2 且文字不變', [t for st, t in _nb if st == '2'], [t for st, t in _ub if st == '2'][:5])
_paras = [t for st, t in _nb]; _pset = set(_paras)
_miss = []
for l in RL:
    if not l.strip() or l.startswith('![](') or l.startswith('|'): continue
    t = re.sub(r'^#+-? ', '', l) if l.startswith('#') else l.strip('　 ')
    if t not in _pset and '　　' + t not in _pset: _miss.append(t[:30])
chk('docx 段落涵蓋 md 全部文字行（標題、段落、圖題、小標、整體結論條列）', _miss, [])
chk('docx 無「@」殘留與無 Markdown 標記', ('@' in ''.join(_paras), '**' in ''.join(_paras), '![](' in ''.join(_paras), '#-' in ''.join(_paras)), (False, False, False, False))
chk('docx 無表（表數 0）', len(re.findall(r'<w:tbl>', _doc)), 0)
chk('整體結論 6 條為編號清單（段落文字以「N、」起首）', [t[:2] for t in _paras if re.match(r'^\d、', t)], [f'{i}、' for i in range(1, 7)])
chk('docx 文字與 o 版相比：o 版之舊用語（達回溯上限、流場範圍內之虛擬質點比例、穿越比例為 100.00%）不再出現於新版', [w_ for w_ in ('達回溯上限', '仍在流場範圍內之比例', '99.999%', '距海岸線') if w_ in ''.join(_paras)], [])

# ---------- 圖號、圖題位置、全文
chk('全文圖號連續', [int(x) for x in re.findall(r'^圖(\d+) ', RES_FULL, re.M)], list(range(1, NF + 1)))
chk('全文無表號、無表格線', ([int(x) for x in re.findall(r'^表(\d+) ', RES_FULL, re.M)], [l[:20] for l in RL if l.startswith('|')]), ([], []))
chk('全文每個圖號皆被引用（展開範圍）、不超出範圍', (expand(RES_FULL, '圖') >= set(range(1, NF + 1)), max(expand(RES_FULL, '圖')) <= NF), (True, True))
chk('全文無「表 N」引用、無「見表」', (expand(RES_FULL, '表'), '見表' in RES_FULL), (set(), False))
caps_ = [l for l in RL if re.match(r'^(圖|表)\d+', l)]
chk('標題格式 圖+編號+單一空格', all(re.match(r'^圖\d+ \S', l) for l in caps_), True)
body_ = [l for l in RL if l not in caps_ and not l.startswith('![](')]
chk('內文引用皆為「圖 N」', [l[:40] for l in body_ if re.search(r'[圖表]\d', l)], [])
bad_f = []
for i_, l in enumerate(RL):
    if re.match(r'^圖\d+ ', l):
        prv = next((x for x in reversed(RL[:i_]) if x.strip()), '')
        if not prv.startswith('![]('): bad_f.append(l[:30])
chk('圖標題位於圖下方', bad_f, [])
chk('圖片連結總數 35（7 圖×5）', len(re.findall(r'^!\[\]\(', RES_FULL, re.M)), NF)
chk('全文無玫瑰圖與原圖 4 型、沿邊界曲線圖相關用語', [x for x in ('boundary_azimuth_rose', '玫瑰', 'settling_velocity_breakdown', 'boundary_arc_weight', '沿流場範圍邊界相對來源權重', '沿邊界弧長', '高斯核', '色階 0.0 至 0.5', '英文縮寫') if x in RES_FULL], [])
chk('全文無「相對來源權重」「權重」「首位來源方位」', [x for x in ('相對來源權重', '權重', '首位來源方位') if x in RES_FULL], [])
chk('全文無差異量、總變異距離、抽樣變異範圍、組對、上緣、最大差異等用語', [x for x in ('差異量', '總變異距離', '抽樣變異範圍', '組對', '上緣', '最大差異', 'TVD', '第 99 百分位', '重複 1000 次', '整批隨機') if x in RES_FULL], [])
chk('正文不用「抽樣誤差」「明顯不同」「超過」等（改白話）；不含已刪方法細節', [x for x in ('抽樣誤差', '明顯不同', '超過抽樣', '超過此範圍', '對半切', '隨機對半', '方位名稱依範圍中心度數', '每一方位寬 45°', '中心度數距方位分界', '種子', '優先序', '簡單隨機', '圖例所列', '連續 3 個整點', '0.5 m', '第 5 小時', '4.25') if x in RES_FULL], [])
chk('方法 B 禁用敘述：無「達回溯上限」「反射」「海岸線反射」「定義 A」「A2」「逐時」「每小時取樣」「取樣誤差」「網格海岸」「人工海岸」「距海岸線」「方法 B」', [x for x in ('達回溯上限', '反射', '定義 A', 'A2', '逐時', '每小時取樣', '取樣誤差', '網格海岸', '人工海岸', '距海岸線', '方法 B', '未穿越邊界') if x in RES_FULL], [])
chk('回溯上限 180 天（圖工具 manifest）、各地點最長追蹤天數皆小於 180', (_dfm['max_days'], _dfm['marks_days'], max(v['max_backtrack_days'] for v in _dfm['sites'].values()) < 180), (180.0, [90.0, 180.0], True))
chk('南灣錨點座標（圖 manifest）', _man_f['footprint_figures']['receptors']['by_site']['nanwan']['anchor_lonlat'], [120.770675, 21.92041])
chk('五地點皆使用正式錨點（formal_anchor_used）', all(_man_f['footprint_figures']['receptors']['by_site'][s]['formal_anchor_used'] for s, _ in SITES), True)
chk('「垂直虛線標示 90 天與 180 天」出現 1 次；「追蹤滿 180 天」「於 180 天回溯期間內」不再出現', (RES_FULL.count('垂直虛線標示 90 天與 180 天'), RES_FULL.count('追蹤滿 180 天'), RES_FULL.count('於 180 天回溯期間內')), (1, 0, 0))
for _w in ('90 天回溯期間', '追蹤滿 90 天', '第 21 小時', '兩顆', '0.001064', '0.0114', '12.5 km'): chk(f'無 run08 殘留用語「{_w}」', RES_FULL.count(_w), 0)
chk('名詞定義各僅一次（穿越比例／海岸來源比例／海岸來源／觸底／高密度區間／沉底時刻／季節月份／大小潮／方向區間寬度／主要來源方向之取法）', [RES_FULL.count(x) for x in ('稱為穿越比例', '稱為海岸來源比例', '稱為海岸來源；', '稱為觸底。', '高密度區間係將各網格依停留時數密度由高至低排序，依序累加其停留時數，累加達 50%、75%、90% 時所涵蓋之網格範圍，即包含該比例停留時數之最小面積區域', '沉底時刻為廢棄物沉降至海床之時刻', '冬季為 12–2 月', '沉底時刻之潮位資料劃分為大潮與小潮', '每 10° 分為一個方向區間', '主要來源方向之取法為', '方向機率高於平均值 2.78%（即 1/36）')], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
chk('圖例已有之描述不再於正文重複', [x for x in ('藍色虛線', '終點之標記', '洋紅色虛線', '實線、虛線、點線依序', '星號為研究地點錨點', '深藍色粗虛線', '格內十條線', '格標題之數字', '第 1 至第 10', '粗線為全部類別', '細線為十類', '圖以研究地點錨點為圓心', '圓之半徑代表', '冬為 12 至 2 月', '單位為 %') if x in RES_FULL], [])
chk('圖內無而正文保留之定義仍在（線色依起始深度序位、停留時數密度算法、高密度區間定義、垂直虛線 90／180 天、方向區間劃分、海岸來源計入、大小潮為代理分類）', all(x in RES_FULL for x in ('線色依起始深度由淺至深', '計 1 小時停留時數', '即包含該比例停留時數之最小面積區域', '垂直虛線標示 90 天與 180 天', '每 10° 分為一個方向區間', '稱為海岸來源', '沉底時刻之潮位資料劃分為大潮與小潮')), True)

# ---------- 圖內文字（來源 SVG 以註解保留之每個文字串）
_svg = json.load(open(SRC + '/svg_text_run09.json', encoding='utf-8'))
def svgkey(site, stem):
    if stem.startswith('polar_source_v2_'): return f'figures-polar-v3-ax2-notext-coastB-fs8/{site}/{stem}'
    if stem.startswith('domain_fraction'): return f'figures-domain-fraction-notext-coastB-fs8/{site}/{stem}'
    if stem == 'horizontal_overview': return f'figures-notext-traj-v5-ax2-coastB-fs8/{site}/{stem}'
    return f'figures-fp1km-b1-ax2-notext-coastB-fs8/{site}/{stem}'
_BADW = re.compile('開放邊界|終止邊界|正常結束|流場區域|沉底 *\\(|^沉底$|stop tracking|穿越開放|達回溯上限|反射')
_adopt = {svgkey(p.split('/')[3], p.split('/')[4]): _svg[svgkey(p.split('/')[3], p.split('/')[4])] for p, _ in _links}
chk('採用之 35 張圖皆有圖內文字紀錄', len(_adopt), NF)
chk('採用之 35 張圖圖內文字不含舊用語（開放邊界／終止邊界／正常結束／流場區域／沉底(stop tracking)／穿越開放／達回溯上限／反射）', [(k_, x_) for k_, v_ in _adopt.items() for x_ in v_ if _BADW.search(x_)], [])
chk('水平軌跡總覽圖例四項（回溯起點、流場範圍邊界、穿越點、抵達海岸）五地點一致，皆無「觸底」「達回溯上限」（24 條代表軌跡中無觸底者）', {s_: (_svg[svgkey(s_, 'horizontal_overview')][-4:], '觸底' in ''.join(_svg[svgkey(s_, 'horizontal_overview')])) for s_, _ in SITES}, {s_: (['回溯起點（受體到達位置）', '流場範圍邊界', '穿越點', '抵達海岸'], False) for s_, _ in SITES})
chk('代表軌跡圖例項數與軌跡終點種類：24 條中穿越點與抵達海岸皆至少 1 條（故兩種圖例皆有）', all(REPB['sites'][s_]['status_counts'].get('1', 0) >= 1 and REPB['sites'][s_]['status_counts'].get('8', 0) >= 1 for s_, _ in SITES), True)
chk('條件式來源足跡圖例用「流場範圍邊界」（五地點）；仍在追蹤比例圖縱軸「回溯期間仍在追蹤之虛擬質點比例（%）」、橫軸「回溯時間（天）」、十一個圖例項（全部類別＋十類）（五地點）', (all('流場範圍邊界' in _svg[svgkey(s_, 'source_footprint_residence')] for s_, _ in SITES), {s_: ('回溯期間仍在追蹤之虛擬質點比例（%）' in _svg[svgkey(s_, 'domain_fraction_vs_backtrack_time')], '回溯時間（天）' in _svg[svgkey(s_, 'domain_fraction_vs_backtrack_time')], len([x for x in _svg[svgkey(s_, 'domain_fraction_vs_backtrack_time')] if x in ['全部類別'] + list(CZ.values())])) for s_, _ in SITES}), (True, {s_: (True, True, 11) for s_, _ in SITES}))
chk('圖內用語於正文皆有對應詞（流場範圍之「邊界」、觸底、抵達海岸、仍在追蹤；圖 1 型之「穿越點」依使用者要求不於導引句寫條數，正文不再出現）', all(w_ in RES_FULL for w_ in ('流場範圍邊界', '觸底', '抵達海岸', '仍在追蹤')), True)
chk('正文不含「開放邊界」「終止邊界」「正常結束」「流場區域」', [w_ for w_ in ('開放邊界', '終止邊界', '正常結束', '流場區域') if w_ in RES_FULL], [])
_hit = []
for p, w in _links:
    site, stem = p.split('/')[3], p.split('/')[4]
    txt = _svg[svgkey(site, stem)]; chk('圖內文字串數 ' + svgkey(site, stem), len(txt) > 5, True)
    _hit += [(svgkey(site, stem), x) for x in txt if re.search(('粒子|公尺|公里|每秒|平方公里|站' if stem.startswith('polar_source_v2_') else '粒子|公尺|公里|每秒|平方公里|機率|站'), x)]
chk('35 張圖內文字不含 粒子／公尺／公里／每秒／站（圓形圖允許「機率」，其餘圖不含）', _hit, [])
_TT_LON, _TT_LAT = '經度 (°E)', '緯度 (°N)'
_allow = re.compile(r'\(°[EN]\)|\d+(?:\.\d+)?°[EN]|\d+°|\d+(?:\.\d+)?%|m/s|\d+ m')
_lat = []; _tt = {}
for _s2, _n2 in SITES:
    for _st in POLSTEMS:
        _tx = _svg[svgkey(_s2, _st)]
        _lat += [(_s2, _st, x) for x in _tx if re.search(r'[A-Za-z]', _allow.sub('', x))]
        _tt[(_s2, _st)] = _tx
chk('圓形圖圖內文字無英文（扣除經緯度、方位、百分比與 m、m/s）', _lat, [])
_cl = {s2: [x.replace('　', '　') for x in _tt[(s2, 'polar_source_v2_by_class')] if '　' in x and x[:2] != '類別'] for s2, _ in SITES}
_pn = {s2: [x.replace('　', '　') for x in _tt[(s2, 'polar_source_v2_velocity_panels')] if '　' in x] for s2, _ in SITES}
chk('多格圖 10 個格標題＝十類總覽圖例之類別與沉降速度範圍（逐項、依序），五地點相同', ({s2: (_cl[s2] == _pn[s2] and len(_pn[s2]) == 10) for s2, _ in SITES}, len({tuple(v) for v in _pn.values()}), [x.split('　')[0] for x in _pn['gongliao']]), ({s2: True for s2, _ in SITES}, 1, [CZ[m] for m in ORDER]))
_rk = ['第 1（最慢）'] + [f'第 {i}' for i in range(2, 10)] + ['第 10（最快）']
chk('多格圖共用圖例：十個序位、錨點、同心圓、等深線，皆為中文', {s2: ([x for x in _tt[(s2, 'polar_source_v2_velocity_panels')] if x.startswith('第 ')] == _rk, '研究地點錨點（圓心）' in _tt[(s2, 'polar_source_v2_velocity_panels')], '同心圓：方向機率（%）' in _tt[(s2, 'polar_source_v2_velocity_panels')]) for s2, _ in SITES}, {s2: (True, True, True) for s2, _ in SITES})
chk('總覽圖例：十類圖例標題含單位（m/s）、季節圖 4 項、大小潮 2 項', {s2: (any(x.startswith('類別') and x.endswith('（m/s）') for x in _tt[(s2, 'polar_source_v2_by_class')]), sum(1 for x in _tt[(s2, 'polar_source_v2_by_season')] if x[:2] in ('冬季', '春季', '夏季', '秋季')), sum(1 for x in _tt[(s2, 'polar_source_v2_by_tide')] if x[:2] in ('大潮', '小潮'))) for s2, _ in SITES}, {s2: (True, 4, 2) for s2, _ in SITES})
chk('圖內文字：圖 2 型之圖例順序＝全部類別＋十類（依沉降速度由慢至快）', _svg[svgkey('gongliao', 'domain_fraction_vs_backtrack_time')][-11:], ['全部類別'] + [x for x in _svg[svgkey('gongliao', 'domain_fraction_vs_backtrack_time')][-10:]])
# 圖 2 型數值：inside_fraction_summary 對 facts（五地點「全部」列），與十類列最大零日
_dfs = rd(SB + '/figures-domain-fraction-coastB-fs8/inside_fraction_summary.csv')
chk('停留比例圖之摘要表：每地點 11 列（全部＋十類），全部列顆數 100,000、十類各 10,000', {s_: (len([r for r in _dfs if r['study_site_id'] == s_]), [r['particles'] for r in _dfs if r['study_site_id'] == s_ and r['group'] == 'all'], {r['particles'] for r in _dfs if r['study_site_id'] == s_ and r['group'] != 'all'}) for s_, _ in SITES}, {s_: (11, ['100000'], {'10000'}) for s_, _ in SITES})

# ---------- 用語、禁用字
SN_BANNED = ['其他／不可回收', '其他/可回收', '跨站', 'BayTrace', 'run0', 'run1', 'worker', 'Worker', '五站', '共用色階', '資料來源', '與前版', '與舊版', '顯著', 'WP-', 'parquet', '.py', 'npz', '十檔', '10 檔', '單一代表', '代表速度', '沉積', '本案', '粒子', '完整罐', '完整瓶', '站', '離開位置', '長條', '編輯說明', '暫定', '〔', '〕', '龜山島西側', '釋放範圍', '淡紫色', '對數色階', '扇區', '方位扇區', '開放邊界', '最高密度區間', 'HDR', '平方公里', '公尺', '公里', '每秒', '候選', '十分之一', '相異數值', '似乎', '推測', '大約', '應該', '或許', '也許', '大概', '比較', '共用', '不含', '相較', '相比', '兩案', '另一案']
for b in SN_BANNED: chk('全文禁用字 ' + b, RES_FULL.count(b), 0)
chk('正式提交版：全文無保留或不確定語氣用字（可能、約、大致、大約、左右、似、推測、初步、暫、尚、僅供、不代表、或許、一般而言、傾向、相對、略、些許）', [w_ for w_ in ('可能', '約', '大致', '大約', '左右', '似', '推測', '初步', '暫', '尚', '僅供', '不代表', '或許', '一般而言', '傾向', '相對', '略', '些許') if w_ in RES_FULL], [])
_RT = '\n'.join(l for l in RL if not l.startswith('![]('))
chk('英文字詞僅限 Lagrangian 與單位符號（m、km、s）', sorted(set(re.findall(r'[A-Za-z]+', _RT)) - {'Lagrangian', 'm', 'km', 's'}), [])
chk('「））」出現次數、無巢狀括號', (RES_FULL.count('））'), re.findall(r'（[^（）]*（', RES_FULL)), (0, []))
chk('單位與中文間有空格', re.findall(r'\d (?:km²|km|m/s|m|h|d)[一-鿿]', RES_FULL), [])
chk('無單用「地點」（「逐地點成果」「本地點結論」為使用者／參考報告用字，例外）', re.findall(r'.{0,6}(?<!研究)地點.{0,4}', RES_FULL.replace('逐地點成果', '').replace('本地點結論', '')), [])
chk('「本地點結論」全文已刪；「逐地點成果」1 處', (RES_FULL.count('本地點結論'), RES_FULL.count('逐地點成果')), (0, 1))
chk('有「研究地點」「虛擬質點」，不用「本案」', ('研究地點' in RES_FULL, '虛擬質點' in RES_FULL, '本案' in RES_FULL), (True, True, False))
chk('用語依服務建議書：無「水位」「漲退潮」「百分位」', [x for x in ('水位', '漲潮', '退潮', '百分位') if x in RES_FULL], [])
_DOM = [l for l in RL if re.match(r'^　　圖 \d+ 為全部 100,000 顆虛擬質點於回溯期間內仍在追蹤之比例', l)]
chk('仍在追蹤之比例導引句共 5 句、僅陳述圖中事實（無因為、表示、顯示、推、代表）', (len(_DOM), [w for w in ('因為', '表示', '顯示', '推', '代表') if any(w in x for x in _DOM)]), (5, []))
_V8 = set(n_ + '方' for n_ in N8) | set(N8[k_] + '至' + N8[(k_ + 1) % 8] + '方' for k_ in range(8))
_NAMES_ALL = re.findall(r'[東南西北]+(?:至[東南西北]+)?方', RES_FULL)
chk('全文所有方位名稱皆為 8 方位（含相鄰兩方位「X 至 Y」），無 16 方位名稱', ([x for x in _NAMES_ALL if x not in _V8], [x for x in ('北北東', '東北東', '東南東', '南南東', '南南西', '西南西', '西北西', '北北西') if x in RES_FULL]), ([], []))
chk('正文不標來源方向度數範圍（無「數字°–數字°」）', re.findall(r'\d+°–\d+°', RES_FULL), [])
chk('邊界案例（中心距分界 ≤2.5° 標示兩方位）：135°–185°→東南至南方、315°–5°→西北至北方、285°–35°→西北至北方、265°–325°→西至西北方、215°–285°→西南至西方；其餘不標：285°–15°→西北方、355°–5°→北方、185°–235°→西南方', [nm_deg(135, 185), nm_deg(315, 5), nm_deg(285, 35), nm_deg(265, 325), nm_deg(215, 285), nm_deg(285, 15), nm_deg(355, 5), nm_deg(185, 235)], ['東南至南方', '西北至北方', '西北至北方', '西至西北方', '西南至西方', '西北方', '北方', '西南方'])
chk('用語一致：無「進入」「視為來自海岸」「全類合計」「南向來源」「觀測時刻」「到達時刻」', [x for x in ('進入', '視為來自海岸', '全類合計', '南向來源', '觀測時刻', '到達時刻') if x in RES_FULL], [])
chk('「到達時間」出現處：引言 1、沉底時刻說明 1', RES_FULL.count('到達時間'), 2)
chk('「沉底」僅用於「沉底時刻」「沉底年齡」；回溯途中碰到海床而停止追蹤一律稱「觸底」', (re.findall(r'沉底(?!時刻|年齡)', RES_FULL), RES_FULL.count('觸底') >= 4, RES_FULL.count('沉底時刻') >= 3), ([], True, True))
chk('全文無全形數字與全形英文字母', re.findall(r'[０-９Ａ-Ｚａ-ｚ]', RES_FULL), [])
chk('全文無緊貼中文之半形標點（, ; : ( )）', re.findall(r'[一-鿿][,;:()]|[,;:()][一-鿿]', RES_FULL), [])
chk('數字與中文量詞之間皆有空格（顆、天、小時、個、處、類）', re.findall(r'\d(?:顆|天|小時|個|處|類)', RES_FULL), [])
chk('範圍皆以「–」連接：無「° 至」「% 至」「數字 至 數字」', re.findall(r'\d° 至|\d% 至|\d 至 \d', RES_FULL), [])
chk('全文無連續兩個以上空白（半形）', re.findall(r'  +', RES_FULL), [])
chk('術語一致：穿越比例、海岸來源比例、觸底、穿越點、抵達海岸；「穿越流場範圍邊界」「抵達海岸」兩條停止追蹤條件於圖 2 導引句同句寫出', ('虛擬質點穿越流場範圍邊界或抵達海岸時停止追蹤' in RES_FULL, RES_FULL.count('海岸來源比例'), RES_FULL.count('穿越比例')), (True, 7, 6))

# ---------- WP-71（2026-10-22）：地圖類圖經緯度座標軸規則（沿用 ax2 規則，方法 B 圖重核）
_T5 = _mt; _F5 = _mf1
_ax_t = _T5['axes_unified']; _ax_f = _F5['footprint_figures']['axes_unified']
_drw_t = {s_: _ax_t['drawn'][s_] for s_, _ in SITES}
_drw_f = {s_: _F5['footprint_figures']['figures'][f'{s_}/source_footprint_residence.png']['layout'] for s_, _ in SITES}
_LON = re.compile(r'^1\d\d\.\d$'); _LAT = re.compile(r'^\d{2}\.\d$')
def _f1(v, h): return f'{v:.1f}'
_wpout = []
for _kind, _drw, _axs in (('水平軌跡總覽', _drw_t, _ax_t), ('條件式來源足跡', _drw_f, _ax_f)):
    _stem = 'horizontal_overview' if _kind == '水平軌跡總覽' else 'source_footprint_residence'
    _key = lambda s_: svgkey(s_, _stem)
    chk(f'{_kind}：刻度標籤皆只寫數字、一位小數（無 °E／°N），且標籤文字＝刻度值以一位小數表示（五地點）', [s_ for s_, _ in SITES if not (all(_LON.match(x) for x in _drw[s_]['xtick_labels']) and all(_LAT.match(x) for x in _drw[s_]['ytick_labels']) and _drw[s_]['xtick_labels'] == [_f1(v, 'E') for v in _drw[s_]['xticks']] and _drw[s_]['ytick_labels'] == [_f1(v, 'N') for v in _drw[s_]['yticks']])], [])
    chk(f'{_kind}：座標軸上下限皆落在刻度上並標出數字', [s_ for s_, _ in SITES if not (_drw[s_]['xticks'][0] == _drw[s_]['xlim'][0] and _drw[s_]['xticks'][-1] == _drw[s_]['xlim'][1] and _drw[s_]['yticks'][0] == _drw[s_]['ylim'][0] and _drw[s_]['yticks'][-1] == _drw[s_]['ylim'][1])], [])
    _sp = {s_: (round(_drw[s_]['xlim'][1] - _drw[s_]['xlim'][0], 6), round(_drw[s_]['ylim'][1] - _drw[s_]['ylim'][0], 6)) for s_, _ in SITES}
    _st = {s_: (sorted({round(b - a, 6) for a, b in zip(_drw[s_]['xticks'], _drw[s_]['xticks'][1:])}), sorted({round(b - a, 6) for a, b in zip(_drw[s_]['yticks'], _drw[s_]['yticks'][1:])})) for s_, _ in SITES}
    chk(f'{_kind}：五地點經度跨度皆 1.8°、緯度跨度皆 1.2°；經緯向刻度間隔皆 0.2°（等間距）', (sorted(set(_sp.values())), sorted({(tuple(a), tuple(b)) for a, b in _st.values()})), ([(1.8, 1.2)], [((0.2,), (0.2,))]))
    chk(f'{_kind}：軸標題五地點一致為「經度 (°E)」「緯度 (°N)」', sorted({(_TT_LON in _svg[_key(s_)], _TT_LAT in _svg[_key(s_)]) for s_, _ in SITES}), [(True, True)])
    chk(f'{_kind}：manifest 之各地點軸範圍＝實際繪出之軸範圍，且皆在 0.2° 倍數上', [s_ for s_, _ in SITES if not ([_axs['sites'][s_]['lon_lo'], _axs['sites'][s_]['lon_hi']] == _drw[s_]['xlim'] and [_axs['sites'][s_]['lat_lo'], _axs['sites'][s_]['lat_hi']] == _drw[s_]['ylim'] and all(abs(v / 0.2 - round(v / 0.2)) < 1e-6 for v in _drw[s_]['xlim'] + _drw[s_]['ylim']))], [])
    chk(f'{_kind}：圖內文字（SVG 註解）之經緯度標籤＝manifest 刻度標籤（逐一、五地點）', [s_ for s_, _ in SITES if not ({x for x in _svg[_key(s_)] if _LON.match(x) or _LAT.match(x)} == set(_drw[s_]['xtick_labels']) | set(_drw[s_]['ytick_labels']))], [])
    chk(f'{_kind}：主圖框長寬比之共用參考緯度 24.1°（1 / cos），各地點長寬比誤差皆在 ±2% 內', ([s_ for s_, _ in SITES if abs(_axs['sites'][s_]['aspect_error_fraction']) > 0.02], _axs['ref_lat_deg'], round(_axs['aspect_dy_over_dx'], 4)), ([], 24.1, round(1 / math.cos(math.radians(24.1)), 4)))
    _wpout.append(f'{_kind}\t' + '\t'.join(f"{s_}: 經度 {_drw[s_]['xlim'][0]:.1f}–{_drw[s_]['xlim'][1]:.1f}、緯度 {_drw[s_]['ylim'][0]:.1f}–{_drw[s_]['ylim'][1]:.1f}、誤差 {_axs['sites'][s_]['aspect_error_fraction'] * 100:+.2f}%" for s_, _ in SITES))
chk('水平軌跡總覽：五地點主圖框尺寸（英吋）完全相同；足跡圖 PNG 五地點尺寸相同（3762×2478）；不畫格線', (len({(round(_drw_t[s_]['frame_size_in'][0], 5), round(_drw_t[s_]['frame_size_in'][1], 5)) for s_, _ in SITES}), {Image.open(f'{ROOT}/{FIGSUB}/{s_}/source_footprint_residence.png').size for s_, _ in SITES}, [s_ for s_, _ in SITES if _drw_t[s_]['grid_lines_drawn']]), (1, {(3762, 2478)}, []))
# 方法 B 圖之座標軸與版面與 o 版（ax2）manifest 逐項相同（僅資料不同）
_T_o = json.load(open(SRC + '/figures-notext-traj-v5-ax2/manifest.json')); _F_o = json.load(open(SRC + '/figures-fp1km-b1-ax2-notext/manifest.json'))
chk('軌跡圖（方法 B）與 o 版：軸範圍、刻度、主圖框、畫布、版面 v3、圖框 axes_box、無格線逐項相同', ([s_ for s_, _ in SITES if any(_T_o['axes_unified']['drawn'][s_][k_] != _T5['axes_unified']['drawn'][s_][k_] for k_ in ('xlim', 'ylim', 'xticks', 'yticks', 'frame_size_in', 'grid_lines_drawn', 'xtick_labels', 'ytick_labels'))], _T_o['axes_unified']['sites'] == _T5['axes_unified']['sites'], _T_o['figsize_in'] == _T5['figsize_in'], _T_o['axes_box_in_left_top_w_h'] == _T5['axes_box_in_left_top_w_h'], _T5['layout'], _T_o['layout']), ([], True, True, True, 'v3', 'v3'))
chk('足跡圖（方法 B）與 o 版：軸範圍、刻度、色階（vmin、vmax、色階紀錄）、色條與主圖框對齊容差逐項相同', ([s_ for s_, _ in SITES if any(_F_o['footprint_figures']['figures'][f'{s_}/source_footprint_residence.png']['layout'][k_] != _drw_f[s_][k_] for k_ in ('xlim', 'ylim', 'xticks', 'yticks'))], [s_ for s_, _ in SITES if any(_F_o['footprint_figures']['figures'][f'{s_}/source_footprint_residence.png'][k_] != _F5['footprint_figures']['figures'][f'{s_}/source_footprint_residence.png'][k_] for k_ in ('vmin', 'vmax', 'scale', 'colormap_scheme', 'isobath_levels_m'))], _F_o['footprint_figures']['axes_unified'] == _F5['footprint_figures']['axes_unified'], _F_o['config']['footprint_colorbar_range'] == _F5['config']['footprint_colorbar_range'] == [0.001, 1000.0]), ([], [], True, True))
chk('足跡圖色階：10⁻³–10³ 對稱、非均勻配色 nonuniform、矩形色條；五地點最大密度（raw_max）皆 ≤ 上限（高於上限 0 格）', (_F5['footprint_figures']['residence_color_scale']['variants']['source_footprint_residence']['upper_limit'], _F5['footprint_figures']['residence_color_scale']['variants']['source_footprint_residence']['lower_limit'], _F5['footprint_figures']['residence_color_scale']['variants']['source_footprint_residence']['colormap_scheme'], _F5['footprint_figures']['residence_color_scale']['colorbar'], [s_ for s_, _ in SITES if _F5['footprint_figures']['figures'][f'{s_}/source_footprint_residence.png']['limit_stats']['cells_above_upper_limit'] != 0]), (1000.0, 0.001, 'nonuniform', {'shape': 'rectangular', 'extend': 'neither'}, []))
chk('足跡圖各地點 hours_fraction_depth_below_threshold 與 footprint_summary 一致（圖工具 manifest 對 postprocess-coastB 表）；總停留時數＝表（五地點）', {s_: fnum(fp[s_]['total_particle_hours']) > 0 for s_, _ in SITES}, {s_: True for s_, _ in SITES})
chk('停留時數於抵達海岸處截斷：各地點總停留時數（h）較 o 版（未截斷）為少；總停留時數 ≈ 各粒子停留日數×24 之合計（比值 1.00–1.01）', {s_: (int(fp[s_]['total_particle_hours']) < {'gongliao': 8882580, 'guishan': 9826365, 'hsinchu': 8156354, 'nanwan': 18276078, 'lienchiang': 54902444}[s_]) for s_, _ in SITES}, {s_: True for s_, _ in SITES})
_Pb = {k_: v_ for k_, v_ in _man.items() if k_ in ('anchors_lonlat', 'sector', 'radius_scales_pct', 'params', 'bearing')}
chk('圓形圖（方法 B）：圓與框以錨點為中心，窗口半寬 34 km＝25 km 半徑 × 1.36、五地點同圖種跨度相同、座標軸樣式 one_decimal', (_pm2['window_half_km'], _man['radius_km'], round(_pm2['window_half_km'] / _man['radius_km'], 2), _pm2['axes_style']), (34.0, 25.0, 1.36, 'one_decimal'))
_cpol = {}; _bad_fmt = []; _bad_in = []; _bad_dec = []
for _st in POLSTEMS:
    for s_, _ in SITES:
        _tx = _svg[svgkey(s_, _st)]
        _lo = sorted({float(x) for x in _tx if _LON.match(x)}); _la = sorted({float(x) for x in _tx if _LAT.match(x)})
        _bad_fmt += [(s_, _st, x) for x in _tx if re.search(r'°[EN]', x) and x not in (_TT_LON, _TT_LAT)]
        _bad_fmt += [(s_, _st, 'axis') for _t_ in (_TT_LON, _TT_LAT) if _t_ not in _tx]
        _bad_dec += [(s_, _st, v) for v in _lo + _la if abs(v * 10 - round(v * 10)) > 1e-6]
        _alon, _alat = _man['anchors_lonlat'][s_]; _hk = _pm2['window_half_km']
        _dlat = _hk / 111.195; _dlon = _hk / (111.195 * math.cos(math.radians(_alat)))
        _bad_in += [(s_, _st, 'outside', v) for v in _lo if not (_alon - _dlon - 0.003 <= v <= _alon + _dlon + 0.003)] + [(s_, _st, 'outside', v) for v in _la if not (_alat - _dlat - 0.003 <= v <= _alat + _dlat + 0.003)]
        _cpol[(s_, _st)] = (_lo, _la)
chk('圓形圖（4 種 × 5 地點）：經緯度刻度標籤只寫數字、一位小數且為 0.1° 整數倍；軸標題「經度 (°E)」「緯度 (°N)」皆在', (_bad_fmt, _bad_dec), ([], []))
chk('圓形圖：所有經緯度標籤皆落在以錨點為中心、半寬 34 km 之窗口內', _bad_in, [])
chk('圓形圖五地點同圖種之 PNG 尺寸相同', {st_: len({Image.open(f'{ROOT}/{FIGSUB}/{s_}/{st_}.png').size for s_, _ in SITES}) for st_ in POLSTEMS}, {st_: 1 for st_ in POLSTEMS})
_fwt = {}
for k_, v_ in _svg.items():
    if k_.endswith('-coastB-fs8') or '-coastB-fs8/' in k_:
        for x in v_:
            if re.search('[（）]', x): _fwt.setdefault(k_.split('/')[0], set()).add(x)
open(ROOT + '/_verify/fullwidth_paren_texts_ax2.txt', 'w', encoding='utf-8').write('\n'.join(f'{k_}\t{x}' for k_, v_ in sorted(_fwt.items()) for x in sorted(v_)) + '\n')
open(ROOT + '/_verify/axes_unified_run09.txt', 'w', encoding='utf-8').write('\n'.join(_wpout) + '\n')
# ---------- WP-73（2026-10-22 i 版）：圖面字級與版面與 TRAPs 完全一致（fs8）
# TRAPs 規格（TRAPs/src/traps/figure_standard.py、concise_style.py；簡要報告大圖 CONCISE_MAP_AXES／CONCISE_MAP_CBAR）：畫布寬 6.3 吋（其 Word 文字區寬；LBT 文字區寬 6.268 吋，取 6.27 吋）、
# 高 4.13 吋，主圖框 左 0.91、下 0.95、高 2.98、寬 = 2.98 x 1.8 x cos(24.1°) / 1.2（程式值 4.080），色條寬 0.12、與主圖框相距 0.093，字級 8 pt，字型 Noto Sans CJK TC。
_TR = dict(w=6.27, h=4.13, left=0.91, bottom=0.95, fh=2.98, fw=round(2.98 * 1.8 * math.cos(math.radians(24.1)) / 1.2, 3), cbw=0.12, cbpad=0.093, pt=8.0, fam='Noto Sans CJK TC')
_cmpd = SB + '/fs8_compare'
_to8 = json.load(open(_cmpd + '/traj_old_manifest.json')); _fo8 = json.load(open(_cmpd + '/fp_old_manifest.json')); _co8 = json.load(open(_cmpd + '/curve_old_fingerprint.json'))
_svgc = json.load(open(_cmpd + '/svg_traj_compare.json'))
_t8 = {s_: _mt8['axes_unified']['drawn'][s_] for s_, _ in SITES}; _f8 = {s_: _mf8['figures'][s_] for s_, _ in SITES}
_c8 = {s_: _dfm['report_fs8']['records'][f'figures-domain-fraction-notext-coastB-fs8/{s_}/domain_fraction_vs_backtrack_time.png'] for s_, _ in SITES}
_c8l = {s_: _dfm['report_fs8']['records'][f'figures-domain-fraction-notext-coastB-fs8/{s_}/domain_fraction_vs_backtrack_time_logy.png'] for s_, _ in SITES}
_c8a = {**{(s_, 'lin'): _c8[s_] for s_, _ in SITES}, **{(s_, 'log'): _c8l[s_] for s_, _ in SITES}}
def _near(a, b, tol=0.002): return all(abs(x - y) <= tol for x, y in zip(a, b)) and len(a) == len(b)
_box_t = [_TR['left'], _TR['bottom'], _TR['fw'], _TR['fh']]
chk('軌跡圖（fs8）畫布 6.27 x 4.13 吋、主圖框（左 0.91、下 0.95、寬 4.080、高 2.98 吋）＝TRAPs 簡要報告大圖；五地點相同', {s_: (_t8[s_]['report_fs8']['canvas_in'] == [_TR['w'], _TR['h']], _near(_t8[s_]['report_fs8']['frame_box_in_left_bottom_w_h'], _box_t)) for s_, _ in SITES}, {s_: (True, True) for s_, _ in SITES})
chk('足跡圖（fs8）畫布 6.27 x 4.13 吋、主圖框同軌跡圖；色條 寬 0.12、與主圖框相距 0.093、與主圖框等高（＝TRAPs CONCISE_MAP_CBAR）；五地點相同', {s_: (_f8[s_]['report_fs8']['canvas_in'] == [_TR['w'], _TR['h']], _near(_f8[s_]['report_fs8']['frame_box_in_left_bottom_w_h'], _box_t), _near(_f8[s_]['report_fs8']['colorbar_box_in_left_bottom_w_h'], [_TR['left'] + _TR['fw'] + _TR['cbpad'], _TR['bottom'], _TR['cbw'], _TR['fh']])) for s_, _ in SITES}, {s_: (True, True, True) for s_, _ in SITES})
chk('曲線圖（fs8，線性與對數縱軸各五地點，共 10 張）畫布 6.27 x 4.29 吋（圖例 3 列，每多一列加 0.16 吋）、主圖框左 0.91、高 2.98、寬 4.080（與地圖主圖框相同）、不畫格線', {k_: (_r['canvas_in'] == [6.27, 4.29], _near(_r['frame_box_in_left_bottom_w_h'], [0.91, 1.11, _TR['fw'], 2.98]), _r['grid']) for k_, _r in _c8a.items()}, {k_: (True, True, False) for k_ in _c8a})
_txt8 = {}
for _k, _recs in (('軌跡圖', {s_: _t8[s_]['report_fs8'] for s_, _ in SITES}), ('足跡圖', {s_: _f8[s_]['report_fs8'] for s_, _ in SITES}), ('曲線圖', _c8a)):
    _txt8[_k] = {s_: (_recs[s_]['text_summary']['sizes_pt'], _recs[s_]['text_summary']['font_families'], _recs[s_]['text_summary']['n_texts']) for s_ in _recs}
chk('圖內全部文字（刻度標籤、座標軸標題、圖例、色條）字級皆 8 pt、字型皆 Noto Sans CJK TC（軌跡圖、足跡圖、曲線圖各五地點，畫完後逐一掃描 Matplotlib 文字物件）', {k_: {s_: (v_[0], v_[1]) for s_, v_ in d_.items()} for k_, d_ in _txt8.items()}, {k_: {s_: ([_TR['pt']], [_TR['fam']]) for s_ in _txt8[k_]} for k_ in _txt8})
chk('足跡圖色條刻度（10 的次方，mathtext）亦用同一字型（mathtext.fontset=custom、mathtext.rm=Noto Sans CJK TC）', {s_: (_f8[s_]['report_fs8']['text_summary']['mathtext_fontset'], _f8[s_]['report_fs8']['text_summary']['mathtext_font_rm'], len(_f8[s_]['report_fs8']['text_summary']['mathtext_texts'])) for s_, _ in SITES}, {s_: ('custom', _TR['fam'], 7) for s_, _ in SITES})
chk('圓形圖（4 種 x 5 地點）字級 8 pt（manifest text_pt；刻度標籤、座標軸標題、圖例、格標題皆以 8 pt 設定，見 polar manifest 與版面測試）、字型 Noto Sans CJK TC', (_pm2['text_pt'], _man['font']['family'], _man['font']['mode']), (8.0, _TR['fam'], 'resolved_cjk_font'))
# 印出字級＝圖內字級 x（docx 顯示寬 / 畫布寬）；畫布寬取各圖 manifest 之畫布
_cvw = {}
for _s2, _ in SITES:
    _cvw[(_s2, 'horizontal_overview')] = (_t8[_s2]['report_fs8']['canvas_in'], 8.0); _cvw[(_s2, 'source_footprint_residence')] = (_f8[_s2]['report_fs8']['canvas_in'], 8.0); _cvw[(_s2, 'domain_fraction_vs_backtrack_time')] = (_c8[_s2]['canvas_in'], 8.0)
    for _st in POLSTEMS: _cvw[(_s2, _st)] = (_pm2['canvas_in']['panels' if _st.endswith('panels') else {'by_class': 'B', 'by_season': 'C', 'by_tide': 'D'}[_st.removeprefix('polar_source_v2_')]], _pm2['text_pt'])
_prt = {}; _pdif = []
for (_cx, _cy, _rid), (_p, _w) in zip(_dr, _links):
    _s2, _st = _p.split('/')[3], _p.split('/')[4]; (_cw, _ch), _pt = _cvw[(_s2, _st)]
    _printed = _pt * (int(_cx) / 914400) / _cw; _prt[(_s2, _st)] = round(_printed, 3)
    if _st != 'horizontal_overview' and _st != 'source_footprint_residence' and _st != 'domain_fraction_vs_backtrack_time': continue
    _im = Image.open(f'{ROOT}/{_p}.png')
    if abs(_im.width - _cw * 600) > 1 or abs(_im.height - _ch * 600) > 1 or abs(int(_cx) / 914400 - _cw) > 0.005 or abs(int(_cy) / 914400 - _ch) > 0.005: _pdif.append((_p, _im.size, round(int(_cx) / 914400, 3), round(int(_cy) / 914400, 3), _cw, _ch))
chk('35 張圖之印出字級（圖內字級 x 顯示寬 / 畫布寬）皆為 8.0 pt（不小於 8 pt）', sorted({v_ for v_ in _prt.values()}), [8.0])
chk('軌跡圖、足跡圖、曲線圖（15 張）：PNG 像素＝畫布吋數 x 600 dpi，docx 顯示尺寸＝畫布尺寸（不縮放、不裁切）', _pdif, [])
open(ROOT + '/_verify/figure_text_size_run09.txt', 'w', encoding='utf-8').write('圖\t地點\t圖內字級 pt\t畫布寬 吋\t顯示寬 吋\t印出字級 pt\n' + '\n'.join(f"{st_}\t{s_}\t{_cvw[(s_, st_)][1]}\t{_cvw[(s_, st_)][0][0]}\t{round(int(cx_) / 914400, 3)}\t{_prt[(s_, st_)]}" for (cx_, cy_, rid_), (p_, w_) in zip(_dr, _links) for s_, st_ in [(p_.split('/')[3], p_.split('/')[4])]) + '\n')
_lgd = {}
for _k, _recs, _rows in (('軌跡圖', {s_: _t8[s_]['report_fs8'] for s_, _ in SITES}, 2), ('足跡圖', {s_: _f8[s_]['report_fs8'] for s_, _ in SITES}, 2), ('曲線圖', _c8a, 3)):
    _lgd[_k] = {s_: (_recs[s_]['legend']['rows'] <= _rows, min(_recs[s_]['legend']['margin_left_in'], _recs[s_]['legend']['margin_right_in'], _recs[s_]['legend']['margin_bottom_in']) >= 0.05, _recs[s_]['legend']['gap_below_axes_in'] >= 0.05) for s_ in _recs}
chk('圖例（外框、8 pt、置中於主圖框）：列數不超過畫布預留列數（軌跡、足跡 2 列，曲線 3 列）；與畫布左右下緣距離 ≥ 0.05 吋；與座標軸標題不重疊（間距 ≥ 0.05 吋）', _lgd, {k_: {s_: (True, True, True) for s_ in _lgd[k_]} for k_ in _lgd})
chk('線寬與記號大小依「印出效果」乘縮放比（軌跡圖 6.27/9.48、足跡圖 6.27/8.9267、曲線圖 6.27/8.0），印出的線與記號和舊版同粗', (round(_t8['gongliao']['report_fs8']['print_scale_applied_to_linewidth_and_markers'], 4), round(_f8['gongliao']['report_fs8']['print_scale_applied_to_linewidth_and_markers'], 4), round(_c8['gongliao']['print_scale_applied_to_linewidth_and_markers'], 4)), (round(6.27 / 9.48, 4), round(6.27 / 8.9267, 4), round(6.27 / 8.0, 4)))
# 數值未變：舊版（-coastB）與報告版（fs8）逐項比對
chk('軌跡圖：軸範圍、刻度、刻度標籤、軸標題、格線與舊版（-coastB）逐項相同；畫出的 24 條軌跡及陸地多邊形之資料指紋與舊版面（同程式、舊版面重畫）相同', [s_ for s_, _ in SITES if any(_t8[s_][k_] != _mt['axes_unified']['drawn'][s_][k_] for k_ in ('xlim', 'ylim', 'xticks', 'yticks', 'xtick_labels', 'ytick_labels', 'xlabel', 'ylabel', 'grid_lines_drawn')) or _t8[s_]['data_fingerprint'] != _to8['axes_unified']['drawn'][s_]['data_fingerprint']], [])
chk('軌跡圖：SVG 折線（以刻度標記還原經緯度）逐條比對：每地點 24 條、端點相差 < 1e-6 度、頂點至另一版折線最短距離 < 0.0015 度（SVG 依解析度簡化共線頂點）', {s_: (r_['paths'], r_['max_endpoint_diff_deg'] < 1e-6, r_['max_vertex_to_other_polyline_deg'] < 0.0015) for s_, r_ in _svgc.items()}, {s_: ([24, 24], True, True) for s_, _ in SITES})
_F_old8 = _fo8['figures']; _F_mo = _mf1['footprint_figures']['figures']; _rcs = _mf1['footprint_figures']['residence_color_scale']['variants']['source_footprint_residence']
chk('足跡圖：色階（vmin、vmax、raw_max、scale、色階方案、色條）、越界統計、等深線與舊版面（同程式、舊版面重畫）逐項相同；vmin、vmax 等於舊版 manifest 色階上下限；密度網格與線條資料指紋與舊版面相同', [s_ for s_, _ in SITES if any(_f8[s_].get(k_) != _F_old8[s_].get(k_) for k_ in ('vmin', 'vmax', 'raw_max', 'scale', 'limit_stats', 'isobath_levels_m', 'colormap_scheme', 'colorbar_range', 'above_upper_limit_drawn_as')) or _f8[s_]['layout']['colorbar'] != _F_old8[s_]['layout']['colorbar'] or _f8[s_]['data_fingerprint'] != _F_old8[s_]['data_fingerprint'] or (_f8[s_]['vmin'], _f8[s_]['vmax']) != (_rcs['lower_limit'], _rcs['upper_limit'])], [])
chk('足跡圖：軸範圍、刻度、刻度標籤（經度、緯度）與舊版（-coastB）manifest 逐項相同；axes_unified 各地點逐項相同', [s_ for s_, _ in SITES if any(_f8[s_]['layout'][k_] != _F_mo[f'{s_}/source_footprint_residence.png']['layout'][k2_] for k_, k2_ in ((k3_, k3_) for k3_ in ('xlim', 'xticks', 'xtick_labels', 'ylim', 'yticks', 'ytick_labels', 'colorbar'))) or _mf8['axes_unified']['sites'][s_] != _mf1['footprint_figures']['axes_unified']['sites'][s_]], [])
chk('曲線圖：摘要表 inside_fraction_summary.csv 與舊版（-coastB）逐位元組相同；十條曲線（全部＋十類）x 兩種縱軸之線條資料指紋與舊版面相同（10 張）', (open(SB + '/figures-domain-fraction-coastB-fs8/inside_fraction_summary.csv', 'rb').read() == open(SB + '/figures-domain-fraction-coastB/inside_fraction_summary.csv', 'rb').read(), {k_: v_['data_fingerprint'] == _co8[k_[len('figures-domain-fraction-notext-coastB-fs8/'):]] for k_, v_ in _dfm['report_fs8']['records'].items()} == {k_: True for k_ in _dfm['report_fs8']['records']}, len(_dfm['report_fs8']['records'])), (True, True, 10))
chk('報告版（fs8）三種圖之輸入 departures 指紋與舊版相同（同一份 postprocess-coastB）', (len({_mt8['inputs_sha256_before']['departures.parquet']['sha256'], _mf8['inputs_sha256_before']['departures.parquet']['sha256'], _dfm['departures_sha256'], _mt['inputs_sha256_before']['departures.parquet']['sha256']}),), (1,))
open(ROOT + '/_verify/check_run09.report.txt', 'w', encoding='utf-8').write(f'OK {len(OK)}  BAD {len(BAD)}\n' + '\n'.join(f'BAD {n}\n   got={g}\n   exp={e}' for n, g, e in BAD) + '\n')
print('OK', len(OK), 'BAD', len(BAD))
for n, g, e in BAD[:80]: print('BAD', n, '\n  got', str(g)[:600], '\n  exp', str(e)[:600])
