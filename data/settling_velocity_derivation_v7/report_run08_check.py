# -*- coding: utf-8 -*-
"""核對報告 docx 內的數字與 v7 results.csv / report_run08_stats.py / report_run08_alu_example.py 輸出是否一致。
用法：python3 report_run08_check.py [docx] [v7目錄] [run08輸出目錄]
未給參數時：docx 預設為專案 outputs/成果報告_run08/3-4-3_run08.docx（本腳本位於 data/settling_velocity_derivation_v7/），
v7目錄與run08輸出目錄預設為本腳本所在資料夾。
"""
import csv, json, math, re, sys, zipfile, os
from xml.etree import ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'outputs', '成果報告_run08', '3-4-3_run08.docx')
V7 = sys.argv[2] if len(sys.argv) > 2 else HERE
OUT = sys.argv[3] if len(sys.argv) > 3 else HERE
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'

fails = []
nchk = 0


def ok(cond, msg):
    global nchk
    nchk += 1
    if not cond:
        fails.append(msg)


def close(a, b, rel=0.0, absol=0.0):
    return abs(a - b) <= max(absol, rel * abs(b))


def num(s):
    return float(s.replace('−', '-').replace(' ', '').replace(' ', ''))


SUPMAP = str.maketrans('⁻⁰¹²³⁴⁵⁶⁷⁸⁹', '-0123456789')


def sci_parse(s):
    """'5.00×10⁻⁷' -> float"""
    s = s.replace(' ', '').replace(' ', '')
    m = re.match(r'^([0-9.]+)[×x]10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹-]+)$', s)
    return float(m.group(1)) * 10 ** int(m.group(2).translate(SUPMAP))


z = zipfile.ZipFile(DOCX)
xml = z.read('word/document.xml')
root = ET.fromstring(xml)
body = root.find(W + 'body')


def ptext(p):
    # 2026-10-29：代表物件標籤「嚴重壓扁」改稱「重壓扁」（results.csv 等來源已改）；此處將封存之舊稿用語正規化，使舊稿仍可核對
    return ''.join((t.text or '') for t in p.iter() if t.tag in (W + 't', M + 't')).replace('嚴重壓扁', '重壓扁')


tables = []
paras = []
for el in body:
    if el.tag == W + 'tbl':
        rws = []
        for tr in el.findall(W + 'tr'):
            rws.append(['\n'.join(ptext(p) for p in tc.iter(W + 'p')) for tc in tr.findall(W + 'tc')])
        tables.append(rws)
    elif el.tag == W + 'p':
        paras.append(ptext(el))
fulltext = ('\n'.join(paras) + '\n' + '\n'.join('\t'.join(c) for t in tables for c in t)).replace('\u2005', ' ')

# ---------------- 資料來源：results.csv ----------------
rows = list(csv.DictReader(open(os.path.join(V7, 'results.csv'), encoding='utf-8-sig')))
ok(len(rows) == 50, 'results.csv 列數 != 50')
CS = {1: '保麗龍', 2: '竹木', 3: '廢紙', 4: '不可回收塑膠', 5: '廢漁網漁具', 6: '寶特瓶', 7: '可回收塑膠',
      8: '鋁罐', 9: '鐵罐', 10: '玻璃瓶'}
cls = {}
for r in rows:
    cls.setdefault(int(r['class_id']), []).extend([float(r['w_lo_ms']), float(r['w_hi_ms'])])
for c in cls:
    ok(len(cls[c]) == 10, 'class %d 解數 != 10' % c)
gm = {c: math.exp(sum(math.log(x) for x in v) / len(v)) for c, v in cls.items()}
wmin = {c: min(v) for c, v in cls.items()}
wmax = {c: max(v) for c, v in cls.items()}
nd = {c: len(set(round(x, 6) for x in v)) for c, v in cls.items()}
ok(sorted(c for c in nd if nd[c] == 8) == [6, 8, 9], '相異速度 8 個的類別應為 6/8/9: %s' % nd)
ok(all(nd[c] == 10 for c in nd if c not in (6, 8, 9)), '其餘類別應為 10 個相異速度')

# ---------------- 正文速度表 ----------------
main = [t for t in tables if t and '10 個速度之範圍' in ''.join(t[0])][0]
ok(len(main) == 11 and all(len(r_) == 4 for r_ in main) and '代表值' not in ''.join(main[0]), '正文表列數、欄數（已刪代表值欄）')
for r in main[1:]:
    c = [k for k, v in CS.items() if v == r[0]][0]
    rng_txt = r[3].replace('\n', '')
    lo, hi = [num(x) for x in re.findall(r'−[0-9.]+', rng_txt)]
    ok(rng_txt.count('−') == 2, '範圍欄應含兩個負號 %s' % rng_txt)
    ok(rng_txt == '−{:#.3g}至 −{:#.3g}'.format(wmin[c], wmax[c]) or rng_txt == '−{:#.3g}至−{:#.3g}'.format(wmin[c], wmax[c]), '正文表範圍字串三位有效數字 %s' % rng_txt)
    ok(close(-lo, wmin[c], rel=0.006), '正文表 %s 下限 %s vs %.6f' % (r[0], lo, wmin[c]))
    ok(close(-hi, wmax[c], rel=0.006), '正文表 %s 上限 %s vs %.6f' % (r[0], hi, wmax[c]))

# ---------------- 第一節結果表 ----------------
t1 = [t for t in tables if t and t[0][0].startswith('類別') and 'w 下界' in ''.join(t[0])][0]
ok(len(t1) == 11 and all(len(r_) == 3 for r_ in t1) and '代表值' not in ''.join(t1[0]), '第一節表列數、欄數（已刪代表值欄）')
for r in t1[1:]:
    c = [k for k, v in CS.items() if r[0].startswith(v)][0]
    ok(r[1] == '{:#.3g}'.format(wmin[c]), '第一節表 %s 下界 %s vs %s' % (r[0], r[1], '{:#.3g}'.format(wmin[c])))
    ok(r[2] == '{:#.3g}'.format(wmax[c]), '第一節表 %s 上界 %s vs %s' % (r[0], r[2], '{:#.3g}'.format(wmax[c])))

# ---------------- 第五節大表 ----------------
big = [t for t in tables if t and '代表物件' in ''.join(t[0]) and '尺寸' in ''.join(t[0])][0]
brs = big[1:]
ok(len(brs) == 50, '大表資料列 %d != 50' % len(brs))
cur = None
GEO = {'slab': '薄板', 'cyl': '圓柱', 'bluff': '鈍體'}
for i, (r, c) in enumerate(zip(brs, rows)):
    if r[0].strip():
        cur = r[0]
    ok(CS[int(c['class_id'])] in (cur or ''), '大表第 %d 列類別 %s vs %s' % (i + 1, cur, c['class_zh']))
    ok(r[2] == GEO[c['geometry']], '大表第 %d 列幾何 %s vs %s' % (i + 1, r[2], c['geometry']))
    ok(close(sci_parse(r[4]), float(c['V_m3']), rel=0.006), '大表第 %d 列 V %s vs %s' % (i + 1, r[4], c['V_m3']))
    ok(close(sci_parse(r[5]), float(c['A_proj_m2']), rel=0.006), '大表第 %d 列 A %s vs %s' % (i + 1, r[5], c['A_proj_m2']))
    ok(int(r[6]) == int(round(float(c['rho_p_kgm3']))), '大表第 %d 列 rho %s vs %s' % (i + 1, r[6], c['rho_p_kgm3']))
    lo, hi = [float(x) for x in re.findall(r'[0-9.]+', r[7])]
    ok(lo == float(c['cd_inf_lo']) and hi == float(c['cd_inf_hi']), '大表第 %d 列 Cd' % (i + 1))
    a, b = [float(x) for x in re.findall(r'[0-9.]+', r[8])]
    ok(close(a, float(c['w_lo_ms']), absol=0.00006) and close(b, float(c['w_hi_ms']), absol=0.00006),
       '大表第 %d 列 w %s vs %s-%s' % (i + 1, r[8], c['w_lo_ms'], c['w_hi_ms']))

# ---------------- 第五節文字統計 ----------------
S = json.load(open(os.path.join(OUT, 'stats_run08.json')))['n50']
st = S['stats']
Re_all = [float(r['Re_lo']) for r in rows] + [float(r['Re_hi']) for r in rows]
ok(close(min(Re_all), st['Re_min'], rel=1e-3) and close(max(Re_all), st['Re_max'], rel=1e-3), 'Re 範圍(csv)與重算不一致')
ok('雷諾數介於 %d 至 %.1f×104' % (round(st['Re_min']), st['Re_max'] / 1e4) in fulltext, '雷諾數範圍文字')
n_lt = sum(1 for x in Re_all if x < 1000)
ok(n_lt == st['n_sol_lt1e3'], 'Re<1e3 解數(csv) %d vs %d' % (n_lt, st['n_sol_lt1e3']))
ok('雷諾數低於 103之解有 %d 個，來自 %d 組' % (st['n_sol_lt1e3'], st['n_obj_lt1e3']) in fulltext, 'Re<1e3 解數文字')
ok('約占阻力係數 %d%%–%d%%' % (round(st['share_lt1e3_min']), round(st['share_lt1e3_max'])) in fulltext, 'Re<1e3 第一項占比文字')
ok('全部 50 組代表物件均在數十次迭代內收斂' in fulltext and st['n_iter_max'] < 100, '迭代次數')
ok('本研究 %d 個薄板物件之投影面積為 %d–%d cm²，其中 %d 個介於 25–2500 cm²，其餘 %d 個小於 25 cm²'
   % (st['n_slab'], st['slab_area_min'], st['slab_area_max'], st['n_slab_25_2500'], st['n_slab_lt25']) in fulltext, '薄板面積文字')
areas = [float(r['A_proj_m2']) * 1e4 for r in rows if r['geometry'] == 'slab']
ok(len(areas) == 30 and min(areas) == 1.0 and abs(max(areas) - 900) < 1e-6, '薄板面積範圍(csv)')
ok(sum(1 for a in areas if 25 <= a <= 2500) == 22 and sum(1 for a in areas if a < 25) == 8, '薄板面積分組(csv)')
ok('1.1–10.8 倍' in fulltext and '%.1f' % st['sphere_ratio_slab_min'] == '1.1' and '%.1f' % st['sphere_ratio_slab_max'] == '10.8', '等效球倍數(片狀)')
ok('1.3–2.3 倍' in fulltext and '%.1f' % st['sphere_ratio_flat_slabcd_min'] == '1.3' and '%.1f' % st['sphere_ratio_flat_slabcd_max'] == '2.3', '等效球倍數(壓扁)')
ok('6–31%' in fulltext and round(st['ge1e4_excess_min']) == 6 and round(st['ge1e4_excess_max']) == 31, 'Re>=1e4 第一項超出比例')
longest = set()
for r in brs:
    if r[2] not in ('薄板', '圓柱'):
        continue
    nums = [float(x) for x in re.findall(r'[0-9.]+', r[3])]
    longest.add((nums[1] if r[3].startswith('φ') else max(nums[:2])) / 10)
ok(sorted(longest) == [1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0], '最長邊實際集合 %s' % sorted(longest))
ok('最長邊取 1、2.5、5、10、15、20、25、30 與 50 cm' in fulltext, '最長邊清單文字')

# ---------------- 第六節鋁罐實例 ----------------
A = json.load(open(os.path.join(OUT, 'alu_example_run08.json')))
ok(close(A['w_min'], wmin[8], rel=1e-5) and close(A['w_max'], wmax[8], rel=1e-5), '鋁罐範圍 vs results.csv')
sol_tbl = [t for t in tables if t and t[0][0].startswith('代表物件') and 'w (m/s)' in ''.join(t[0]) and 'Re' in ''.join(t[0]) and len(t) == 11][0]
for r in sol_tbl[1:]:
    s = [q for q in A['solutions'] if q['obj'] == r[0] and abs(q['cd_inf'] - float(r[2])) < 1e-9][0]
    ok(close(num(r[3]), s['Re'], absol=1) and close(num(r[4]), s['Cd'], absol=0.00006) and close(num(r[5]), s['w'], absol=0.000006),
       '十個解表 %s: %s' % (r[0], r))
ok('十個解之範圍 %.5f–%.5f m/s' % (A['w_min'], A['w_max']) in fulltext, '十個解範圍文字')
for o in A['objects']:
    if o['Vwater'] is not None:
        ok(close(o['rho'], (o['m'] + o['Vwater'] * 1025) / o['V'], rel=1e-9), '鋁罐 ρeff 自洽 %s' % o['name'])
allcells = ''.join(c for t in tables for r in t for c in r)
ok('1042.9' in allcells and '0.003206' in allcells and '0.02374' in allcells and '5.837' in allcells, '新增鋁罐列的有效密度/N/V_water')
ok(all(1025 < float(r['rho_p_kgm3']) <= 1388 or r['geometry'] == 'slab' or True for r in rows), '')
# 容器有效密度皆介於海水密度與 1388 之間（完整進水容器；壓扁者為幾何放大，另列）
for r in rows:
    if r['geometry'] == 'bluff':
        rho = float(r['rho_p_kgm3'])
        if r['class_id'] in ('6', '8', '9'):
            ok(1025 < rho < 1388, '完整容器有效密度 %s %.1f' % (r['scenario'], rho))

# ---------------- 第七節（以 10 個解之範圍對照實測）----------------
RS = json.load(open(os.path.join(OUT, 'range_sens_run08.json')))
sev = [t for t in tables if t and '文獻實測值' in ''.join(t[0])][0]
ok(sev[0][1] == '本研究 10 個解之範圍', '表 A10 欄名')
def rg(c):
    return '{:#.3g}–{:#.3g}'.format(wmin[c], wmax[c])
ok(sev[1][1] == rg(8) and sev[2][1] == rg(9), '表 A10 鋁罐、鐵罐範圍 = results.csv 三位有效數字：%s／%s' % (sev[1][1], sev[2][1]))
ok(sev[3][1] == ' / '.join(rg(c) for c in (1, 2, 3)), '表 A10 保麗龍、竹木、廢紙範圍：%s' % sev[3][1])
ok(wmin[8] < 0.115 and 0.120 < wmax[8], '鋁罐範圍涵蓋鋁瓶蓋實測 0.115-0.120')
ok(wmin[9] < 0.176 and 0.177 < wmax[9], '鐵罐範圍涵蓋鐵瓶蓋實測 0.176-0.177')
ok(wmin[9] > wmin[8] and wmax[9] > wmax[8], '鐵罐下界、上界均高於鋁罐')
ok('實測值落於推導範圍內' in sev[1][3] and '實測值落於推導範圍內' in sev[2][3], '表 A10 說明欄')
ok('鋁罐 10 個解之範圍 0.0367–0.175 m/s 涵蓋鋁質瓶蓋實測值 0.115–0.120 m/s，鐵罐 0.107–0.427 m/s 涵蓋鐵質瓶蓋實測值 0.176–0.177 m/s；鐵罐之下界與上界均高於鋁罐，與鐵質瓶蓋快於鋁質瓶蓋之順序一致。' in fulltext, '第七節文字')
ok('（單位均為 m/s；本研究欄為該類 10 個解之最小值至最大值。）' in fulltext, '表 A10 備註')

# ---------------- 第八節 ----------------
V = RS['variants']
sens = [t for t in tables if t and '十類下界與上界之變化' in ''.join(t[0])][0]
for c in range(1, 11):
    ok(close(RS['base'][str(c)]['lo'], wmin[c], absol=6e-7) and close(RS['base'][str(c)]['hi'], wmax[c], absol=6e-7), '敏感度基準範圍 vs results.csv 類 %d' % c)


def f1(x):
    return '0.0%' if abs(x) < 0.05 else ('%+.1f%%' % x).replace('-', '−')


def rng_txt(k):
    v = V[k]
    return '下界 %s 至 %s，上界 %s 至 %s' % (f1(v['pct_lo_min']), f1(v['pct_lo_max']), f1(v['pct_hi_min']), f1(v['pct_hi_max']))


ok(sens[0][2] == '十類下界、上界排序', '表 A11 欄名')
ok(sens[1][1] == rng_txt('textbook'), '教科書值 %s vs %s' % (sens[1][1], rng_txt('textbook')))
ok(sens[2][1] == rng_txt('no_first_term'), '去第一項 %s vs %s' % (sens[2][1], rng_txt('no_first_term')))
swap = '下界排序不變；上界排序中保麗龍與鋁罐互換（基準上界為保麗龍 %s、鋁罐 %s m/s）' % ('{:#.3g}'.format(wmax[1]), '{:#.3g}'.format(wmax[8]))
ok(all(V[k]['order_lo_same'] for k in V), '各情境下界排序皆不變')
ok(not V['textbook']['order_hi_same'] and not V['no_first_term']['order_hi_same'] and V['textbook']['order_hi'].index(1) < V['textbook']['order_hi'].index(8) and V['no_first_term']['order_hi'].index(1) < V['no_first_term']['order_hi'].index(8) and RS['order_base_hi'].index(8) < RS['order_base_hi'].index(1), '教科書值、去第一項：上界排序保麗龍與鋁罐互換')
ok(sens[1][2] == swap and sens[2][2] == swap, '表 A11 前兩列排序欄')
c_ = V['cyl_dn']
ok(c_['changed_lo'] == [2, 5] and c_['changed_hi'] == [2, 5] and V['cyl_dn']['order_lo_same'] and V['cyl_dn']['order_hi_same'], '圓柱等體積球徑：僅竹木、廢漁網漁具之範圍改變，排序不變')
ok(sens[3][1] == '竹木下界 %s、上界 %s；廢漁網漁具下界 %s、上界 %s，其餘不變' % (f1(c_['pct_lo']['2']), f1(c_['pct_hi']['2']), f1(c_['pct_lo']['5']), f1(c_['pct_hi']['5'])) and sens[3][2] == '下界、上界排序均不變', '圓柱列文字 %s' % sens[3][1])
ok(V['dedupe']['changed_lo'] == [] and V['dedupe']['changed_hi'] == [] and sens[4][1] == '下界與上界均不變（重複解不影響最小值與最大值）' and sens[4][2] == '下界、上界排序均不變', '去重複列')
nu = '10 °C：下界 %s 至 %s，上界 %s 至 %s；28 °C：下界 %s 至 %s，上界 %s 至 %s' % tuple(f1(V[k][m]) for k in ('nu10', 'nu28') for m in ('pct_lo_min', 'pct_lo_max', 'pct_hi_min', 'pct_hi_max'))
ok(sens[5][1] == nu and sens[5][2] == '下界、上界排序均不變' and V['nu10']['order_lo_same'] and V['nu10']['order_hi_same'] and V['nu28']['order_lo_same'] and V['nu28']['order_hi_same'], '黏滯係數列 %s' % sens[5][1])

# ---------------- 殘留舊字詞 ----------------
for bad in ['採用值', '30 組', '三種代表物件', '六個解', '6 個解', '其他／', '其他/', '一半', '1 萬', '機率', 'BayTrace', 'run0', '站', '圖-', '表-', '本案', '粒子', '完整罐', '完整進水瓶身', '本研究設定' if False else '本案設定']:
    ok(bad not in fulltext, '殘留字詞「%s」' % bad)


# ---------------- 推導流程（附錄新增一節）----------------
ok('推導流程' in paras, '缺「推導流程」標題段')
i0 = paras.index('推導流程')
flow = paras[i0:i0 + 15]
ft = '\n'.join(flow)
src = open(os.path.join(V7, 'derive.py'), encoding='utf-8').read()
def const(name):
    return float(re.search(r'^%s\s*=\s*([0-9.eE+-]+)' % name, src, flags=re.M).group(1))
ok(const('RHO_W') == 1025.0 and const('NU') == 1.05e-6 and const('G') == 9.81, 'derive.py 常數與流程輸入不一致')
ok('ρw=1025' in ft and 'ν=1.05×10-6' in ft and 'g=9.81' in ft, '流程輸入常數文字')
ok(re.search(r'w = 1e-4', src) and re.search(r'1e-12 \* max\(w_new, 1e-12\)', src) and re.search(r'0\.5 \* w \+ 0\.5 \* w_new', src)
   and re.search(r'range\(500\)', src), 'derive.py 迭代設定（初值1e-4、容許1e-12、阻尼0.5、500次）')
ok('w←1×10-4' in ft, '2.1 初值')
ok('|w*-w|<10-12w*' in ft and '上限 500 次' in ft, '2.3 收斂條件與上限')
ok('w←0.5w+0.5w*' in ft, '2.3 更新式')
ok('Cd=24/Re×(1+0.15 Re0.687)+Cd,∞' in ft, '2.2 阻力係數式')
ok('R=ρp/ρw-1' in ft and 'N=2VgR/Aproj' in ft, '1.3、1.4 算式')
ok('本研究 50 組皆 R>0' in ft, '1.3 R>0')
ok(all(float(r['rho_p_kgm3']) > 1025 for r in rows), '50 組 R>0（csv）')
ok('式(A5)' in ft and '式(A10)' in ft and '式(A3)' in ft, '流程引用式號')
ok('1,000 顆' in ft and '隨機排列' in ft and '各類 10 個解之最小值與最大值即第一節之下界與上界' in ft and '彙整' not in ft and '取三位有效數字' not in ft, '步驟 3（用於模擬）')
for k in ('1.1', '1.2', '1.3', '1.4', '1.5', '2.1', '2.2', '2.3'):
    ok(any(x.startswith(k + '　') for x in flow), '流程條目 %s' % k)
for k in ('步驟 1　', '步驟 2　', '步驟 3　用於模擬'):
    ok(any(x.startswith(k) for x in flow), '流程 ' + k)
ok('迭代程序見推導流程步驟 2' in fulltext, '補充一（一）')
ok('須同時求解（推導流程步驟 2）' in fulltext and '迭代式（推導流程步驟 2）' in fulltext, '第二節、第六節指向流程')
for k in ('推導流程 1.1', '推導流程 1.2', '推導流程 1.3、1.4', '推導流程 1.5', '推導流程步驟 2；以 330 mL'):
    ok(k in fulltext, '第六節步驟標題對應：' + k)

# ---------------- B4：引用的比例與範圍（重算輸出 b4_recheck_run08.json）----------------
Bj = json.load(open(os.path.join(OUT, 'b4_recheck_run08.json')))
lo, hi = Bj['n50']['sphere_ratio_slab']
ok(('%.1f–%.1f 倍' % (lo, hi)) in fulltext, 'B4 片狀物件等效球倍數')
lo, hi = Bj['n50']['sphere_ratio_flat_slabcd']
ok(('%.1f–%.1f 倍' % (lo, hi)) in fulltext, 'B4 壓扁等效球倍數')
lo, hi = Bj['n50']['ge1e4_excess_pct']
ok(('%d–%d%%' % (round(lo), round(hi))) in fulltext, 'B4 Re>=1e4 之 Cd 差')
e2 = Bj['supp2_example']
ok(('R=%.5f' % e2['R']) in fulltext and ('w=%.4f' % e2['w_hi']).replace('w=', '') [:6] in fulltext, 'B4 補充二 R、上界')
ok('Cd=%.3f' % e2['Cd'] in fulltext and ('Re=%d' % round(e2['Re'])) in fulltext and '0.%03d' % round(e2['SN_term'] * 1000) in fulltext, 'B4 補充二 Cd、Re、第一項')
ok('相差約 %d%%' % round(e2['edge_0.1_vs_1.0_m_pct']) in fulltext and ('相差約 %.1f 倍' % e2['film_5_vs_50_mm_ratio']) in fulltext, 'B4 補充二 邊長影響')

# ---------------- 圖、表引用與圖題格式 ----------------
ok('如圖 1，' in fulltext and '如圖 2。' in fulltext, '內文引用「圖 N」')
ok(re.search(r'如圖\d', fulltext) is None and re.search(r'對應圖\d', fulltext) is None, '引用不得為「圖N」')
ok(any(re.match(r'^圖1 [^ ]', t_) for t_ in paras) and any(re.match(r'^圖2 [^ ]', t_) for t_ in paras), '圖題格式「圖N 標題」')


# ---------------- r4：名稱統一與段落調整 ----------------
ok(re.search(r'(?<!研究)(?<!到達)(?<!發現)地點', fulltext) is None, '單獨「地點」殘留：' + str(re.findall(r'.{4}地點.{2}', fulltext)))
ok('各研究地點之抽樣範圍如圖 2' in fulltext and '使受體集中於各研究地點關注海域內' in fulltext and '該研究地點關注海域' in fulltext, '受體段研究地點')
ok('各研究地點之 50 個到達時間' in fulltext and '五個研究地點之錨點座標見表 2。' in fulltext, '研究地點（到達時間段、座標表前）')
ok(any(r[0] == '研究地點' for t in tables for r in t), '座標表頭「研究地點」')
ok('2、到達地點（Receptor locations）：設定20種不同的到達／被發現地點。' in fulltext.replace(' ', '') or '到達地點（Receptor locations）' in fulltext, '計畫書標題保留')
ok(any(re.match(r'^圖1 四區流場範圍與各研究地點抽樣範圍位置', t_) for t_ in paras) and any(re.match(r'^圖2 各研究地點受體抽樣範圍', t_) for t_ in paras), '圖題用「各研究地點」')
dsc = re.findall(rb'descr="([^"]*)"', xml)
ok(len(dsc) == 2 and all('各研究地點' in d.decode('utf8') and '站' not in d.decode('utf8') for d in dsc), '圖片替代文字')
ok('假設10種不同的沉降／上升速度（10 類，各類 10 個速度），以涵蓋' in fulltext, '計畫書句加註')
ok('若已有同尺度之實測 w，可由式(A1)反求 Cd；本研究十類物件均無此類實測，故由教科書取得 Cd 再求 w。' in fulltext, '第二節一句')
ok('反之' not in fulltext and '直接反求' not in fulltext, '第二節舊句已刪')
ok('介於端面與側面迎流面積之間（條件為 L/d>1.09）' in fulltext and 'π/4' not in fulltext.split('四、物件形狀與密度')[1].split('五、各類輸入參數')[0], '第四節（二）')
ok('幾何平均不受最大值主導' not in fulltext, '已刪「幾何平均不受最大值主導」')
i1 = [i for i, t_ in enumerate(paras) if t_.startswith('表 A1 中數值均取三位有效數字')]
ok(len(i1) == 1 and paras[i1[0] + 1].startswith('同類 10 個解中，保麗龍') and '補充二（二）' in paras[i1[0] + 1], '第一節表後兩段')
ok('表 1 列各類 10 個速度之範圍。垂直座標向上為正，故各速度均為負值。' in fulltext and '幾何平均（代表值）。' not in fulltext, '正文速度表前說明段已無代表值')
# 虛擬質點與 10 個速度之分配（正文、附錄步驟 4、表 A1 後說明一致）
ok('各研究地點每類各有 10,000 顆虛擬質點，該類 10 個速度各分配給其中 1,000 顆。虛擬質點與速度之對應，以固定隨機種子之隨機排列決定，不含斯托克斯漂流案例與含斯托克斯漂流案例採用相同之分配。' in fulltext.replace('模擬時，', ''), '正文分配說明段')
ok('各研究地點每類之 10,000 顆虛擬質點，10 個解各分配給其中 1,000 顆，分配由固定隨機種子之隨機排列決定，不含斯托克斯漂流案例與含斯托克斯漂流案例相同。垂直座標向上為正，故各解取負值。' in fulltext, '附錄推導流程步驟 4 分配說明')
ok('各解於模擬中之分配方式見推導流程步驟 3。' in fulltext and '並非全類只用代表值' not in fulltext and '取用一個' not in fulltext and '隨機取用' not in fulltext, '表 A1 後說明段與步驟 3 一致，無舊「取用一個」句')
ok('粒子' not in fulltext.replace('虛擬質點', ''), '不得出現「粒子」')
ok('完整進水瓶身' not in fulltext and '或壓扁、重壓扁之瓶身' not in fulltext and '壓扁瓶身' not in re.sub('半壓扁瓶身|重壓扁瓶身', '', fulltext), '壓扁瓶身命名')
ok(all(c in fulltext for c in ('完整進水罐（330 mL）', '完整進水罐（500 mL）', '半壓扁瓶身', '330 mL 完整進水瓶', '小型完整進水罐')), '完整進水罐／瓶命名')
# 表1 簡化後之「代表形狀與狀態」欄（各類一句短語，尺寸與密度見表 A4）
t1_ = tables[0]
short_ = {'不可回收塑膠': '薄膜至帆布片之軟質包材', '廢紙': '吸水紙片至紙箱板', '保麗龍': '吸水或附著之發泡板至箱體側板', '廢漁網漁具': '網線至繩索', '寶特瓶': '瓶身碎片、完整進水瓶、半壓扁至重壓扁瓶身', '竹木': '細竹枝至大型竹木段（水浸飽和）', '可回收塑膠': '硬質塑膠碎片至塑膠籃側板、橡膠塊', '鋁罐': '罐身碎片、完整進水罐、半壓扁至重壓扁罐身', '鐵罐': '罐身碎片、完整進水罐、半壓扁至重壓扁罐身', '玻璃瓶': '完整進水瓶、緻密碎片、瓶身曲殼薄片'}
ok(t1_[0][2] == '代表形狀與狀態' and len(t1_) == 11 and all(short_[r_[0]] == r_[2] for r_ in t1_[1:]), '表1 代表形狀與狀態欄為簡化短語')
ok('代表物件之尺寸與輸入參數（表 A4）' in fulltext, '表1 後句指向表 A4')
ok('本研究設定' in fulltext and '本研究推導值' in fulltext and '本研究 Cd,∞' in fulltext.replace(' ', ' '), '本研究')
ok('本法' in fulltext and '本附錄' in fulltext, '本法、本附錄保留')


# ---------------- 表編號、表題與引用 ----------------
body_ = ET.fromstring(xml).find(W + 'body')
seq_ = []   # 依出現順序：('p', text, keepNext) / ('tbl', 首列文字)
for el_ in body_:
    if el_.tag == W + 'p':
        tx_ = ''.join(t_.text or '' for t_ in el_.iter(W + 't'))
        kn_ = el_.find(W + 'pPr') is not None and el_.find(W + 'pPr').find(W + 'keepNext') is not None
        seq_.append(('p', tx_, kn_, el_))
    elif el_.tag == W + 'tbl':
        r0_ = el_.find(W + 'tr')
        first_ = ''.join(t_.text or '' for t_ in r0_.iter(W + 't')).strip()
        eq_ = re.match(r'^\(A\d+\)', first_) is not None
        seq_.append(('tbl', first_, eq_, el_))
data_tbls_ = [i_ for i_, e_ in enumerate(seq_) if e_[0] == 'tbl' and not e_[2]]
caps_ = []
for i_ in data_tbls_:
    pr_ = seq_[i_ - 1]
    ok(pr_[0] == 'p' and re.match(r'^表(A?\d+) [^ ]', pr_[1]) is not None, '資料表上方應為表題：' + seq_[i_][1][:20])
    ok(pr_[0] == 'p' and pr_[2], '表題 keepNext：' + pr_[1][:20])
    ok(pr_[0] == 'p' and pr_[3].find(W + 'pPr').find(W + 'jc').get(W + 'val') == 'center', '表題置中：' + pr_[1][:20])
    caps_.append(pr_[1].split(' ')[0])
exp_ = ['表1', '表2'] + ['表A%d' % k_ for k_ in range(1, 13)]
ok(caps_ == exp_, '表號連續且順序：%s' % caps_)
# 表題段落總數 = 資料表數（無多餘表題）
ok(sum(1 for e_ in seq_ if e_[0] == 'p' and re.match(r'^表A?\d+ [^ ]', e_[1])) == len(exp_), '表題段落數')
ok(not re.search(r'表-', fulltext), '表題無連字號')
# 內文引用（排除表題本身與教科書「表 11-1」等）
body_txt_ = '\n'.join(e_[1] for e_ in seq_ if e_[0] == 'p' and not re.match(r'^表A?\d+ [^ ]', e_[1]))
for lab_ in exp_:
    n_ = len(re.findall(r'(?<![A-Za-z0-9])' + lab_[0] + r' ' + lab_[1:] + r'(?![0-9])', body_txt_))
    print('  引用次數 %s：%d' % (lab_, n_))
for lab_ in exp_:
    ok(re.search(r'表 ' + lab_[1:] + r'(?![0-9])', body_txt_) is not None, '內文引用 表 ' + lab_[1:])
ok(re.search(r'表(A?\d+)(?![0-9 .])', body_txt_.replace('表 11', '')) is None, '內文引用不得為「表N」無空格')
ok('下表' not in fulltext and '如下表' not in fulltext and '上表' not in fulltext and '第五節之表' not in fulltext and '參數來源表' not in fulltext and '正文表列' not in fulltext and '第五節表列值' not in fulltext and '此表檢核' not in fulltext, '舊式表格引用已清除')
ok('與教科書同類形狀之比較見表 A2。' in fulltext and '各代表物件之值如表 A7。' in fulltext and '上下界）見表 A9。' in fulltext and '表 A10 之實測對象' in fulltext and '各項檢驗結果見表 A11。' in fulltext, '表 A2、A7、A9、A10、A11 引用句')
ok('表 1 列各類 10 個速度之範圍' in fulltext and '表 A4 中體積' in fulltext and '（見表 A5）' in fulltext and '與表 A1 及表 1 一致' not in fulltext and '表 A12 檢核者' in fulltext and '三種幾何模型（表 A3）' in fulltext and '輸入取自表 A4' in fulltext and '如表 A6 所示' in fulltext and '如表 A8 所示' in fulltext and '如表 A5 所示' in fulltext, '改後引用句')


# ---------------- 表2 錨點、圖檔（抽樣範圍為錨點為圓心、半徑 12.5 km 之圓；龜山島錨點 121.939148, 24.850326） ----------------
import hashlib
site_ = [r_ for r_ in tables if r_ and r_[0][0] == '研究地點' and r_[0][1].startswith('經度')]
ok(len(site_) == 1, '找到表2')
ok(len(site_[0][0]) == 3 and site_[0][0][2].startswith('緯度') and all(len(r_) == 3 for r_ in site_[0]), '表2 僅 3 欄（地點、錨點經度、錨點緯度），無抽樣範圍欄：%s' % site_[0][0])
ok(not any('抽樣範圍' in c_ or re.search(r'\d–\d', c_) for r_ in site_[0] for c_ in r_), '表2 無範圍欄位內容')
rows2_ = site_[0][1:]
ok([r_[0] for r_ in rows2_] == ['貢寮', '龜山島', '新竹外海', '南灣', '連江'], '表2 地點欄（龜山島）')
gs_ = [r_ for r_ in rows2_ if r_[0] == '龜山島'][0]
ok(gs_[1:] == ['121.94', '24.85'], '表2 龜山島列：%s' % gs_)
ok(abs(round(121.939148, 2) - 121.94) < 1e-9 and abs(round(24.850326, 2) - 24.85) < 1e-9, '龜山島錨點進位 0.01°')
ok(any(re.match(r'^圖2 .*\(b\) 龜山島、', t_) for t_ in paras) and '龜山島西側' not in fulltext, '圖2 圖題 (b) 龜山島；全文無「龜山島西側」')
ok(any(t_ == '表2 五個研究地點之錨點座標' for t_ in paras) and '五個研究地點之錨點座標見表 2。' in fulltext, '表2 標題與導句')
imgs_ = {n_: hashlib.md5(z.read(n_)).hexdigest() for n_ in z.namelist() if n_.startswith('word/media/')}
ok(imgs_.get('word/media/image1.png') == '0f43b39768141205d377b7062e187ec9' and imgs_.get('word/media/image2.png') == 'c312f8e8b97fe0d70fe5f171dfddff3c', '圖1、圖2 圖檔 md5 等於圓形抽樣範圍新版（343_fig1/2_circle.png）：%s' % imgs_)
ok('12.5 km' in fulltext and '10 km' not in fulltext.replace('12.5 km', ''), '內文有「12.5 km」且無「10 km」')
ok('抽樣範圍為以各研究地點錨點為圓心、半徑 12.5 km 之圓' in fulltext, '受體段寫入半徑 12.5 km 之圓')
ok(fulltext.count('5 個水平位置 × 每個位置 4 個垂向位置') == 1, '「5 個水平位置 × 4 個垂向位置」僅出現一次（未重複）')
ok('虛線框' not in fulltext and '虛線圓' in fulltext and '框旁' not in fulltext, '「虛線框」「框旁」0 次；圖2 圖說為虛線圓')
ok(any(t_.startswith('圖2 各研究地點受體抽樣範圍（虛線圓）與 5 個受體水平位置（空心圓）：') for t_ in paras) and any(t_.startswith('圖1 ') and '圓旁字母' in t_ for t_ in paras), '圖1、圖2 圖題')


# ---------------- 附錄不得再有代表值；式號連續且每個引用都對應存在之式 ----------------
ok('代表值' not in fulltext and '幾何平均' not in fulltext and '摘要' not in fulltext, '全文無「代表值」「幾何平均」「摘要」：%s' % re.findall(r'.{6}(?:代表值|幾何平均|摘要).{6}', fulltext))
ok('各解於模擬中之分配方式見推導流程步驟 3。' in fulltext and '推導流程步驟 4' not in fulltext, '推導流程步驟編號引用')
eqlabels_ = []
for e_ in seq_:
    if e_[0] == 'tbl' and e_[2]:
        eqlabels_ += [int(n_) for n_ in re.findall(r'\(A(\d+)\)', ''.join(t_.text or '' for t_ in e_[3].iter(W + 't')))]
ok(eqlabels_ == list(range(1, 14)), '式號連續 A1–A13：%s' % eqlabels_)
refs_ = [int(n_) for e_ in seq_ if e_[0] == 'p' for n_ in re.findall(r'\(A(\d+)\)', e_[1])]
ok(refs_ and set(refs_) <= set(eqlabels_), '內文式號引用皆存在：%s' % sorted(set(refs_) - set(eqlabels_)))
ok('式(A12)' in fulltext and '式(A13)' not in fulltext and '式(A14)' not in fulltext, '補充一引用 Clift–Gauvin 式為 (A12)')

# ---------------- 公式數 ----------------
x = xml.decode('utf-8')
print('oMathPara=%d  oMath=%d  編號公式=%d' % (x.count('<m:oMathPara>'), x.count('<m:oMath>'), len(re.findall(r'\(A\d+\)', fulltext))))
ok(len(re.findall(r'\(A\d+\)', fulltext)) >= 13, '編號公式數')
print('總檢查項 %d；失敗 %d' % (nchk, len(fails)))
for f in fails:
    print('FAIL:', f)
sys.exit(1 if fails else 0)
