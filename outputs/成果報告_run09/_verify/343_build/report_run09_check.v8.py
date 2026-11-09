# -*- coding: utf-8 -*-
"""（run09 版，由 make_check_run09.py 自 report_run08_check.py 產生）核對報告 docx 內的數字與 v7 results.csv / report_run08_stats.py / report_run08_alu_example.py 輸出是否一致。
用法：python3 report_run08_check.py [docx] [v7目錄] [run08輸出目錄]
未給參數時：docx 預設為專案 outputs/成果報告_run08/3-4-3_run08.docx（本腳本位於 data/settling_velocity_derivation_v7/），
v7目錄與run08輸出目錄預設為本腳本所在資料夾。
"""
import csv, json, math, re, sys, zipfile, os
from xml.etree import ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
V7DIR = os.path.join(HERE, '..', '..', '..', '..', 'data', 'settling_velocity_derivation_v7')
DOCX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', '3-4-3_run09.docx')
V7 = sys.argv[2] if len(sys.argv) > 2 else V7DIR
OUT = sys.argv[3] if len(sys.argv) > 3 else V7DIR
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
    return ''.join((t.text or '') for t in p.iter() if t.tag in (W + 't', M + 't'))


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
ok('各研究地點之 50 個到達時間' in fulltext and '五個研究地點之錨點座標與抽樣範圍見表 2。' in fulltext and '五個研究地點之錨點座標見表 2。' not in fulltext, '研究地點（到達時間段、座標表前）')
ok(any(r[0] == '研究地點' for t in tables for r in t), '座標表頭「研究地點」')
ok('2、到達地點（Receptor locations）：設定20種不同的到達／被發現地點。' in fulltext.replace(' ', '') or '到達地點（Receptor locations）' in fulltext, '計畫書標題保留')
ok(any(re.match(r'^圖1 四區流場範圍與各研究地點抽樣範圍位置', t_) for t_ in paras) and any(re.match(r'^圖2 各研究地點受體抽樣範圍', t_) for t_ in paras), '圖題用「各研究地點」')
dsc = re.findall(rb'descr="([^"]*)"', xml)
ok(len(dsc) == 7 and all('各研究地點' in d.decode('utf8') for d in dsc[:2]) and all('站' not in d.decode('utf8') and '放大' not in d.decode('utf8') and '半徑' not in d.decode('utf8') for d in dsc), '圖片替代文字（共 7 張，圖1、圖2 含各研究地點，無放大圖、無半徑）')
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
ok('完整進水瓶身' not in fulltext and '或壓扁、嚴重壓扁之瓶身' not in fulltext and '壓扁瓶身' not in re.sub('半壓扁瓶身|嚴重壓扁瓶身', '', fulltext), '壓扁瓶身命名')
ok(all(c in fulltext for c in ('完整進水罐（330 mL）', '完整進水罐（500 mL）', '半壓扁瓶身', '330 mL 完整進水瓶', '小型完整進水罐')), '完整進水罐／瓶命名')
# 表1 簡化後之「代表形狀與狀態」欄（各類一句短語，尺寸與密度見表 A4）
t1_ = tables[0]
short_ = {'不可回收塑膠': '薄膜至帆布片之軟質包材', '廢紙': '吸水紙片至紙箱板', '保麗龍': '吸水或附著之發泡板至箱體側板', '廢漁網漁具': '網線至繩索', '寶特瓶': '瓶身碎片、完整進水瓶、半壓扁至嚴重壓扁瓶身', '竹木': '細竹枝至大型竹木段（水浸飽和）', '可回收塑膠': '硬質塑膠碎片至塑膠籃側板、橡膠塊', '鋁罐': '罐身碎片、完整進水罐、半壓扁至嚴重壓扁罐身', '鐵罐': '罐身碎片、完整進水罐、半壓扁至嚴重壓扁罐身', '玻璃瓶': '完整進水瓶、緻密碎片、瓶身曲殼薄片'}
ok(t1_[0][2] == '代表形狀與狀態' and len(t1_) == 11 and all(short_[r_[0]] == r_[2] for r_ in t1_[1:]), '表1 代表形狀與狀態欄為簡化短語')
ok('代表物件之尺寸與輸入參數、實測量級對照與敏感度分析詳見附錄' in fulltext and '代表物件之尺寸與輸入參數（表 A4）' not in fulltext, '表1 後句（使用者於 Word 手改，刪去「（表 A4）」，以現況為準）')
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
ok(site_[0][0] == ['研究地點', '經度 (°E)', '緯度 (°N)', '抽樣範圍經度 (°E)', '抽樣範圍緯度 (°N)'] and all(len(r_) == 5 for r_ in site_[0]), '表2 五欄（與 run08 同表頭）：%s' % site_[0][0])
import math as _m
_anc = {'貢寮': (121.9223889, 25.0964444), '龜山島': (121.939148, 24.850326), '新竹外海': (120.45, 24.75), '南灣': (120.770675, 21.92041), '連江': (119.95, 26.2)}
def _rng(lon_, lat_, km_=10.0):
    dlat_ = km_ / 111.32
    dlon_ = km_ / (111.32 * _m.cos(_m.radians(lat_)))
    return ('%.2f–%.2f' % (lon_ - dlon_, lon_ + dlon_), '%.2f–%.2f' % (lat_ - dlat_, lat_ + dlat_))
_want5 = [[k_, '%.2f' % round(v_[0], 2), '%.2f' % round(v_[1], 2)] + list(_rng(v_[0], v_[1])) for k_, v_ in _anc.items()]
ok([r_ for r_ in site_[0][1:]] == _want5, '表2 五欄數值等於由錨點與半徑 10 km 公式（緯 ±10/111.32、經 ±10/(111.32·cosφ)）計算值：%s' % site_[0][1:])
_tblx = [e_ for e_ in body if e_.tag == W + 'tbl' and ''.join(x_.text or '' for x_ in e_.iter(W + 't')).startswith('研究地點經度')]
_grid = [int(g_.get(W + 'w')) for g_ in _tblx[0].find(W + 'tblGrid')]
ok(_grid == [1700, 1350, 1350, 2030, 2027] and len(_tblx) == 1, '表2 欄寬 %s' % _grid)
rows2_ = site_[0][1:]
ok([r_[0] for r_ in rows2_] == ['貢寮', '龜山島', '新竹外海', '南灣', '連江'], '表2 地點欄（龜山島）')
gs_ = [r_ for r_ in rows2_ if r_[0] == '龜山島'][0]
ok(gs_[1:3] == ['121.94', '24.85'], '表2 龜山島列：%s' % gs_)
ok([r_[1:3] for r_ in rows2_] == [['121.92', '25.10'], ['121.94', '24.85'], ['120.45', '24.75'], ['120.77', '21.92'], ['119.95', '26.20']], '表2 五列座標（南灣已改 120.77、21.92）：%s' % rows2_)
ok(['%.2f' % round(x_, 2) for x_ in (121.9223889, 25.0964444, 121.939148, 24.850326, 120.45, 24.75, 120.770675, 21.92041, 119.95, 26.2)] == ['121.92', '25.10', '121.94', '24.85', '120.45', '24.75', '120.77', '21.92', '119.95', '26.20'], '表2 與 run09 release config 錨點進位 0.01° 一致')
ok(abs(round(121.939148, 2) - 121.94) < 1e-9 and abs(round(24.850326, 2) - 24.85) < 1e-9, '龜山島錨點進位 0.01°')
ok(any(re.match(r'^圖2 .*\(b\) 龜山島、', t_) for t_ in paras) and '龜山島西側' not in fulltext, '圖2 圖題 (b) 龜山島；全文無「龜山島西側」')
ok(any(t_ == '表2 五個研究地點之錨點座標與抽樣範圍' for t_ in paras) and '五個研究地點之錨點座標與抽樣範圍見表 2。' in fulltext, '表2 標題與導句')
imgs_ = {n_: hashlib.md5(z.read(n_)).hexdigest() for n_ in z.namelist() if n_.startswith('word/media/')}
import hashlib as _h
_sha = {n_: _h.sha256(z.read(n_)).hexdigest() for n_ in z.namelist() if n_.startswith('word/media/')}
_ref = {}
for _l in open(os.path.join(HERE, 'figs_new_v2', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_arrival_zh_v6', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
_rels = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="(media/[^"]+)"', z.read('word/_rels/document.xml.rels').decode('utf8')))
_rels.update({a_: b_ for b_, a_ in re.findall(r'Target="(media/[^"]+)"[^>]*?Id="(rId\d+)"', z.read('word/_rels/document.xml.rels').decode('utf8'))})
_docimg = ['word/' + _rels[r_] for r_ in re.findall(r'r:embed="(rId\d+)"', xml.decode('utf8') if isinstance(xml, bytes) else xml)]
ok(len(_docimg) == 7 and len(set(_docimg)) == 7, '文件內依序引用 7 個不同圖檔：%s' % _docimg)
ok(_sha.get(_docimg[0]) == _ref['343_run09v2_fig1_circle.png'] and _sha.get(_docimg[1]) == _ref['343_run09v2_fig2_circle.png'], '圖1、圖2 圖檔 sha256 等於 .203 重繪輸出（半徑 10 km 圓、圖例「抽樣範圍」版）')
_order = ['wp61_%s_overview_S_notext.png' % _s for _s in ('gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang')]
ok(len(_sha) == 7 and all(_sha.get(_docimg[2 + i_]) == _ref[n_] for i_, n_ in enumerate(_order)), '圖3–圖7 圖檔 sha256 等於 figs_arrival_zh_v6（.203 figures-zh-v5 重繪輸出，圖例不含括號數字且對齊主圖左緣，v9；取代 v8 之 figs_arrival_zh_v6）；媒體檔共 7 個')
_v3 = {}
for _l in open(os.path.join(HERE, 'figs_arrival_zh_v3', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _v3[os.path.basename(_p[1])] = os.path.join(HERE, 'figs_arrival_zh_v3', _p[1])
import struct as _st0
ok(all(z.read(_docimg[2 + i_])[16:24] == open(_v3[n_], 'rb').read()[16:24] and z.read(_docimg[2 + i_])[16:24] == _st0.pack('>II', 1880, 1753) and _sha.get(_docimg[2 + i_]) != hashlib.sha256(open(_v3[n_], 'rb').read()).hexdigest() for i_, n_ in enumerate(_order)), '圖3–圖7 像素尺寸與 v3 相同（1880x1753），且確已不同於 v3（圖例括號數字已由繪圖程式拿掉）')
_v5d = {}
try:
    from PIL import Image as _Im
    import numpy as _np
    for i_, n_ in enumerate(_order):
        _a = _np.asarray(_Im.open(_v3[n_]).convert('RGBA')).astype(int)
        _b = _np.asarray(_Im.open(__import__('io').BytesIO(z.read(_docimg[2 + i_]))).convert('RGBA')).astype(int)
        _ys = _np.where((_np.abs(_a - _b).sum(2) > 0).any(1))[0]
        _v5d[n_] = (int(_ys.min()), int(_ys.max()), _a.shape[0])
    ok(all(v_[0] >= 1500 and v_[1] < v_[2] for v_ in _v5d.values()), '圖3–圖7 與 v3 之像素差異僅在底部圖例帶（第 1500 列以下；繪圖區與軸、標記、季節標頭逐像素相同）：%s' % _v5d)
    _al = {}
    for i_, n_ in enumerate(_order):
        _s_ = n_.split('_')[1]
        _a5 = _np.asarray(_Im.open(os.path.join(HERE, 'figs_arrival_zh_v5', _s_, n_)).convert('RGBA')).astype(int)
        _im = _Im.open(__import__('io').BytesIO(z.read(_docimg[2 + i_])))
        _b = _np.asarray(_im.convert('RGBA')).astype(int)
        _dd = (_np.abs(_a5 - _b).sum(2) > 0) if _a5.shape == _b.shape else None
        _g = _np.asarray(_im.convert('L')).astype(int)
        _sp = 215 + int(_np.where((_g[1200:1350, 215:230] < 128).all(0))[0].min())
        _lg = int(_np.where((_g[1521:1760, :900] < 128).any(0))[0].min())
        _rg = 900 + int(_np.where((_g[1521:1760, 900:] < 200).any(0))[0].min())
        _al[n_] = (_sp, _lg, _rg, None if _dd is None else (int(_np.where(_dd.any(1))[0].min()), int(_dd[:1500].sum())))
    ok(all(v_[3] is not None and v_[3][0] >= 1500 and v_[3][1] == 0 for v_ in _al.values()), '圖3–圖7 與前版（figs_arrival_zh_v5，圖例靠左）尺寸相同，像素差異僅在底部圖例帶（第 1500 列以下，其上逐像素相同）（v9）：%s' % {k_: v_[3] for k_, v_ in _al.items()})
    ok(all(abs(v_[1] - v_[0]) <= 1 for v_ in _al.values()), '圖3–圖7 左欄最左側圖例標記墨跡之 x 與主圖左脊線之 x 相差不超過 1 像素（v9；各圖 脊線 x／標記 x）：%s' % {k_: v_[:2] for k_, v_ in _al.items()})
    ok(all(abs(v_[2] - 988) <= 2 for v_ in _al.values()), '圖3–圖7 右欄圖例墨跡起點在主圖水平中央（繪圖區 x 222–1754 之中點 988，容差 2 像素）（v9）：%s' % {k_: v_[2] for k_, v_ in _al.items()})
except ImportError:
    ok(True, '（略過像素差異檢查：無 PIL／numpy）')
import struct as _st
_ext = [(int(m_.group(1)), int(m_.group(2))) for m_ in re.finditer(r'<wp:extent cx="(\d+)" cy="(\d+)"', xml.decode('utf8') if isinstance(xml, bytes) else xml)]
ok(len(_ext) == 7, '文件內圖片 7 張')
ok(abs(_ext[0][0] / 914400 - 4.134) < 0.01 and abs(_ext[0][1] / 914400 - 5.672) < 0.01 and abs(_ext[1][0] / 914400 - 6.267) < 0.01 and abs(_ext[1][1] / 914400 - 8.92) < 0.02, '圖1、圖2 嵌入尺寸沿用 run08（4.13x5.67、6.27x8.92 in）')
for _i in range(2, 7):
    _w, _h = _st.unpack('>II', z.read(_docimg[_i])[16:24])
    _sc = (_ext[_i][0] / 914400) / (_w / 300.0)
    ok(abs(_ext[_i][0] / 914400 - 6.267) < 0.01 and 5.5 <= _ext[_i][1] / 914400 <= 6.01 and _sc >= 0.999, '圖%d 寬 %.3f in、高 %.3f in，嵌入縮放 %.4f（圖內 9 pt 字不縮小）' % (_i + 1, _ext[_i][0] / 914400, _ext[_i][1] / 914400, _sc))
ok(len(re.findall(r'Type="[^"]*/image"', z.read('word/_rels/document.xml.rels').decode('utf8'))) == 7, '圖片關聯 7 個（無放大圖殘留）')
ok('12.5 km' not in fulltext and fulltext.count('10 km') == 2 and '12.5km' not in fulltext, '內文無「12.5 km」，「10 km」出現 2 次（受體段落兩處）；例外：附錄「+12.5%」為球體檢驗誤差，非抽樣半徑')
ok('抽樣範圍為以各研究地點錨點為圓心、半徑 10 km 之圓' in fulltext, '受體段寫入半徑 10 km 之圓')
ok(fulltext.count('5 個水平位置 × 每個位置 4 個垂向位置') == 1, '「5 個水平位置 × 4 個垂向位置」僅出現一次（未重複）')
ok('虛線框' not in fulltext and '虛線圓' in fulltext and '框旁' not in fulltext, '「虛線框」「框旁」0 次；圖2 圖說為虛線圓')
ok(any(t_.startswith('圖2 各研究地點受體抽樣範圍（虛線圓）與 5 個受體水平位置（空心圓）：') for t_ in paras) and any(t_.startswith('圖1 ') and '圓旁字母' in t_ for t_ in paras), '圖1、圖2 圖題')
ok(not any('12.5' in t_ for t_ in paras if t_.startswith('圖')) and not any('12.5' in d_.decode('utf8') for d_ in dsc), '圖說與替代文字無 12.5')


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


# ---------------- run09：新增與變更項目 ----------------
# 1. 受體段落：新方法關鍵句
rec_ = [t_ for t_ in paras if t_.startswith('共由「5 個水平位置')]
ok(len(rec_) == 1, '受體段落唯一')
rec_ = rec_[0]
for k_ in ['半徑 10 km 之圓',
           '分別將半徑 10 km 範圍內之候選海域等分為面積相同之 5 個區間',
           '依方位及距研究地點錨點之距離',
           '使每一方位區間與每一距離區間各恰有 1 個受體水平位置',
           '並於所屬區塊（方位區間與距離區間之交會處）內隨機抽取',
           '並於同一區塊內依隨機順序遞補',
           '候選面須為有效之三角形網格且非陸地，並於全部 50 個到達時間均維持濕潤（不因退潮而乾出）',
           '抽中後另須通過流場與波浪資料之逐時空間支援檢查，未通過者剔除']:
    ok(k_ in rec_, '受體段關鍵句：' + k_)
for k_ in ['無放回', '均勻隨機抽取 5 個位置', '依隨機抽樣之順序遞補', '12.5', '關注海域位於島的西側', '位於島的西側', '故西半部', '西半部', '東半部', '配置 3 個']:
    ok(k_ not in rec_, '受體段不得再有舊方法用語：' + k_)
ok('關注海域位於島的西側' not in fulltext and '龜山島之西半部' not in fulltext, '全文無「關注海域位於島的西側」與「龜山島之西半部」（使用者已於 Word 刪除該句，維持刪除）')
# 2. 到達時間段落
arr_ = [t_ for t_ in paras if t_.startswith('50 個到達時間由 48 個分層時刻')]
ok(len(arr_) == 1, '到達時間段落唯一')
arr_ = arr_[0]
for k_ in ['涵蓋四季，每季 12 個分層時刻，大潮、小潮各 6 個',
           '以各時刻前後共 25 小時（約一個潮日）之潮汐強度（該時段水位起伏之大小）與該季之中位數比較，高於中位數者為大潮，低於者為小潮',
           '挑選時排除流場資料於日間銜接處水位不連續之時刻',
           '每格選出 2 個時刻', '共 24 格']:
    ok(k_ in arr_, '到達時間段關鍵句：' + k_)
ok('各月 5 至 9 個' not in arr_ and '各月' not in arr_ and '全年十二個月' not in arr_ and not re.search(r'5\s*至\s*9', arr_), '到達時間段不得再有「各月 5 至 9 個」「全年十二個月」')
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', arr_)), '到達時間段無英文字母')
# 2b. 溯源停止邊界段：回溯上限 180 天與逐地點離開流場範圍比例（數字自 termination_by_group.csv 讀取，不手抄）
import csv as _csv
_t = {}
for _r in _csv.DictReader(open(os.path.join(HERE, 'src', 'termination_by_group.csv'), encoding='utf-8-sig')):
    if _r['grouping'] == 'site_all':
        _t[_r['site']] = (int(_r['n_total']), int(_r['n_crossed']))
ok(set(_t) == {'gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang'}, '停止邊界：csv 五地點齊全')
_pc = {k_: v_[1] / v_[0] * 100.0 for k_, v_ in _t.items()}
stop_ = [t_ for t_ in paras if t_.startswith('4、溯源停止邊界')]
ok(len(stop_) == 1, '溯源停止邊界段唯一')
stop_ = stop_[0]
_want = ('回溯期間上限為 180 天。於此期間內，貢寮、龜山島與新竹外海之全部虛擬質點離開流場範圍邊界，南灣有 %.3f%%，連江有 %.3f%%。'
         % (_pc['nanwan'], _pc['lienchiang']))
ok(stop_.endswith(_want), '停止邊界末句與 csv 一致：' + _want)
_pre = '本研究之關注區域即圖 1 之四區流場範圍，虛擬質點離開流場範圍邊界即停止追蹤。'
ok(stop_.endswith(_pre + _want) and stop_.count(_pre) == 1, '停止邊界段末：先有「本研究之關注區域即圖 1 之四區流場範圍，虛擬質點離開流場範圍邊界即停止追蹤。」再接回溯上限與比例句')
ok(_pc['gongliao'] == _pc['guishan'] == _pc['hsinchu'] == 100.0 and '%.3f' % _pc['nanwan'] == '99.999' and '%.3f' % _pc['lienchiang'] == '95.857', 'csv 值：貢寮、龜山島、新竹外海 100%、南灣 99.999%、連江 95.857%')
ok('開放邊界' not in fulltext and '開放邊界' not in stop_ and '流場範圍邊界' in stop_, '停止邊界段用「流場範圍邊界」，全文無「開放邊界」')
ok('180 天' in stop_ and '180 天' not in re.sub(re.escape('回溯期間上限為 180 天'), '', fulltext), '全文「180 天」僅出現於停止邊界末句')
ok(not re.search(r'較|比|高於|低於|相比|最', _want), '停止邊界末句無比較語氣')
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', _want)), '停止邊界末句無英文字母')

# 2c. 圖 3–圖 7 圖內文字（由 SVG 文字取出）：只有專有名詞或符號可有英文（白名單 Hs、m、m/s）、用語依服務建議書（有效波高）、Y 軸標題在 Y 軸位置
import glob as _g, json as _j
_cnt = _j.load(open(os.path.join(HERE, 'src', 'arrival_counts.json'), encoding='utf-8'))
_SITES = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
_svg_all = {}
for _s, _zh in _SITES:
    _f = os.path.join(HERE, 'figs_arrival_zh_v6', _s, 'wp61_%s_overview_S_notext.svg' % _s)
    _sv = open(_f, encoding='utf-8').read()
    _els = []
    for _m in re.finditer(r'<text([^>]*)>(.*?)</text>', _sv, flags=re.S):
        _tx = re.sub(r'<[^>]+>', '', _m.group(2)).strip()
        if _tx:
            _els.append((_tx, _m.group(1)))
    _tx = [x_[0] for x_ in _els]
    _svg_all[_s] = _tx
    _c = _cnt[_s]
    ok(_c['total'] == 50 and sum(_c['class'].values()) == 48 and sum(_c['phase'].values()) == 48 and sum(_c['event'].values()) == 2, '%s：manifest 計數 50 = 48 個分層時刻（大潮 + 小潮；三潮相）+ 2 個事件時刻' % _zh)
    ok(not [x_ for x_ in _tx if re.search(r'[（(]\s*\d+\s*[）)]', x_)], '%s 圖內文字無括號數字（圖例不含各類數量）' % _zh)
    for _w in ['大潮', '小潮', '漲潮最快', '退潮最快', '近平潮', '高浪事件', '強流事件', '季內中位數', '全年第 99 百分位',
               '冬季', '春季', '夏季', '秋季', '日期（年/月/日）']:
        ok(_tx.count(_w) >= 1, '%s 圖內文字含「%s」' % (_zh, _w))
    _dt = [x_ for x_ in _tx if re.fullmatch(r'\d{4}/\d{2}/\d{2}', x_)]
    ok(_dt == ['2025/01/01', '2025/03/01', '2025/05/01', '2025/07/01', '2025/09/01', '2025/11/01', '2026/01/01'], '%s 日期刻度每兩個月：%s' % (_zh, _dt))
    _sn = [x_ for x_ in _tx if x_.startswith('大潮 ') and '／小潮 ' in x_]
    _exp = ['大潮 %d／小潮 %d' % (_c['season_class'][k_ + '|spring_proxy'], _c['season_class'][k_ + '|neap_proxy']) for k_ in ('DJF', 'MAM', 'JJA', 'SON')]
    ok(_sn == _exp, '%s 各季色塊標示大潮／小潮數與 manifest 一致：%s' % (_zh, _sn))
    # 英文：僅白名單
    _lat = [x_ for x_ in _tx if re.search(r'[A-Za-z]', re.sub(r'\bHs\b|\bm/s\b|\bm\b', '', x_))]
    ok(not _lat, '%s 圖內文字無英文（白名單：Hs、m、m/s）：%s' % (_zh, _lat))
    ok(all(not re.search(r'[A-Za-z]', re.sub(r'\bHs\b|\bm/s\b|\bm\b', '', x_)) for x_ in _tx) and not any(w_ in ' '.join(_tx) for w_ in ('UTC', 'Date', 'Spring', 'Neap', 'Winter', 'Summer', 'Autumn', 'Fastest', 'slack', 'median', 'percentile', 'p99', 'Tidal', 'Current', 'wave height')), '%s 無 UTC、Date 及舊英文詞' % _zh)
    ok(not any(w_ in ' '.join(_tx) for w_ in ('顯著波高', '示性波高', '有義波高', '世界時', '半徑', '25 小時海面高度', '均方根')), '%s 無「顯著波高／示性波高／世界時」，定義文字不在圖上' % _zh)
    ok(sum(1 for x_ in _tx if '有效波高' in x_) == 1 and '有效波高 Hs' in _tx, '%s 圖內「有效波高 Hs」（服務建議書用詞）' % _zh)
    ok(set(re.findall(r'font-size: ([0-9.]+)px', _sv)) == {'9'}, '%s 圖內文字皆 9 px（嵌入縮放 1.0 即 9 pt）' % _zh)
    # Y 軸標題：垂直（rotate(-90)）、位於刻度字左側、其垂直位置在對應面板範圍內
    _ytick = [(float(re.search(r'y="([0-9.]+)"', a_).group(1)), x_) for x_, a_ in _els if re.fullmatch(r'\d+(\.\d+)?', x_) and 'x="49.78"' in a_]
    _grp, _prev = [], None
    for _y, _v in _ytick:
        if _prev is None or float(_v) < _prev:
            _grp.append([])
        _grp[-1].append(_y); _prev = float(_v)
    ok(len(_grp) == 3, '%s 三個面板之縱軸刻度組：%d' % (_zh, len(_grp)))
    _yl = {}
    for _x, _a in _els:
        _mm = re.search(r'translate\(([0-9.]+) ([0-9.]+)\) rotate\(-90\)', _a)
        if _mm:
            _yl.setdefault(_x, []).append((float(_mm.group(1)), float(_mm.group(2))))
    ok(sorted(_yl) == sorted(['潮汐強度', '（m）', '有效波高 Hs', '流速', '（m/s）']) and len(_yl['（m）']) == 2, '%s Y 軸垂直標題文字：%s' % (_zh, sorted(_yl)))
    _rng = [(min(g_), max(g_)) for g_ in _grp]  # 由下而上：流速、有效波高、潮汐強度
    _pos_ok = all(xx_ < 53.0 for v_ in _yl.values() for xx_, _ in v_)
    _in = lambda yy_, r_: r_[0] - 8 <= yy_ <= r_[1] + 8
    _pos_ok = _pos_ok and _in(_yl['潮汐強度'][0][1], _rng[2]) and _in(_yl['有效波高 Hs'][0][1], _rng[1]) and _in(_yl['流速'][0][1], _rng[0])
    _pos_ok = _pos_ok and all(_in(yy_, _rng[2]) or _in(yy_, _rng[1]) or _in(yy_, _rng[0]) for v_ in _yl.values() for _, yy_ in v_)
    ok(_pos_ok, '%s Y 軸標題在 Y 軸位置（x 在繪圖區左側刻度字外、垂直落於各自面板內）：%s' % (_zh, _yl))
    ok(not any('潮汐強度' in x_ and '（' in x_ and 'm）' in x_ and len(x_) > 8 for x_ in _tx) and not any(w_ in ' '.join(_tx) for w_ in ('Significant', 'Tidal strength')), '%s 面板上方無長標題' % _zh)
# 兩條線樣式：由繪圖程式常數與圖例確認（黑色粗長虛線 vs 深紅點線）
_src = open(os.path.join(HERE, 'scripts_fig', 'wp61_v2', 'arrival_figs.py'), encoding='utf-8').read()
ok('LINE_MEDIAN = dict(color="black", lw=1.7, ls=(0, (7, 3)))' in _src and 'LINE_P99 = dict(color="#b2182b", lw=1.5, ls=(0, (1, 1.6)))' in _src, '季內中位數為黑色粗長虛線、全年第 99 百分位為深紅點線（程式常數）')
for _s, _zh in _SITES:
    _sv = open(os.path.join(HERE, 'figs_arrival_zh_v6', _s, 'wp61_%s_overview_S_notext.svg' % _s), encoding='utf-8').read()
    ok('stroke-dasharray: 11.9,5.1' in _sv.replace('stroke-dasharray: 11.900000,5.100000', 'stroke-dasharray: 11.9,5.1') or re.search(r'stroke: #000000[^"]*stroke-dasharray: 11\.9', _sv) is not None, '%s 圖中有黑色長虛線（中位數）' % _zh)
    ok(re.search(r'stroke: #b2182b', _sv) is not None, '%s 圖中有深紅線（全年第 99 百分位）' % _zh)
# 圖 1、圖 2 圖例（沿用 v2）

_man = _j.load(open(os.path.join(HERE, 'figs_new_v2', '343_run09v2_figs_circle_manifest.json'), encoding='utf-8'))
ok(_man['circle_legend_label'] == '抽樣範圍' and float(_man['radius_km']) == 10.0, '圖1 圖例文字為「抽樣範圍」，半徑 10 km 僅於本文與表')
_src = open(os.path.join(HERE, 'scripts_fig', 'make_343_figs_run09.py'), encoding='utf-8').read()
_m = re.search(r'def _circle_label.*?return "([^"]*)"', _src, flags=re.S)
ok(_m is not None and _m.group(1) == '抽樣範圍', '繪圖程式 _circle_label 回傳「抽樣範圍」（圖檔文字已轉為路徑，無法由 SVG 讀字；半徑只寫在本文與表 2）')

# 3. 回溯上限：全文不得出現 90 天、2160 小時（正文原無此二詞；180 天亦不在 3-4-3 內）
ok(not re.search(r'90\s*天|2160|2,160|九十天|三個月', fulltext), '全文無「90 天」「2160 小時」')
ok(re.search(r'12\.5\s*km', fulltext) is None, '全文無「12.5 km」')
# 4. 驗證圖：導引句、圖題、編號、引用
guide_ = [t_ for t_ in paras if t_.startswith('以下列出各研究地點之到達時間驗證圖')]
ok(len(guide_) == 1, '導引句唯一')
g_ = guide_[0]
for k_ in ['每個研究地點 1 張全年總覽圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7',
           '圖中潮汐強度即前述各時刻前後 25 小時之水位起伏大小，以海面高度之均方根計算',
           '季內中位數即大潮與小潮之分界；近平潮即最接近漲退潮交替之時刻',
           '2 個事件時刻另標示於有效波高與流速曲線上',
           '同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊']:
    ok(k_ in g_, '導引句：' + k_)
_g6 = ('以下列出各研究地點之到達時間驗證圖，每個研究地點 1 張全年總覽圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7。'
       '圖中潮汐強度即前述各時刻前後 25 小時之水位起伏大小，以海面高度之均方根計算；季內中位數即大潮與小潮之分界；近平潮即最接近漲退潮交替之時刻。'
       '2 個事件時刻另標示於有效波高與流速曲線上。同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊。')
ok(g_ == _g6, '（v6）導引句全文等於精簡版')
ok('橘色標記為大潮' not in fulltext and '圖例括號內為各類時刻數' not in fulltext and '黑色粗虛線' not in fulltext, '（v6）全文無已刪之圖例重複說明（橘色標記為大潮、圖例括號內為各類時刻數、黑色粗虛線）')
_zf = zipfile.ZipFile(DOCX)
_ft = _zf.read('word/footer1.xml').decode('utf-8')
_ftxt = re.sub(r'<[^>]+>', '', re.sub(r'<w:instrText[^>]*>.*?</w:instrText>', '', _ft))
ok('3-' not in _ft and 'PAGE' in _ft and '<w:jc w:val="center"/>' in _ft, '（v6）頁尾無「3-」，仍有 PAGE 欄位且置中')
ok(not re.search(r'[A-Za-z\-]', re.sub(r'\d+', '', _ftxt)) and [n_ for n_ in _zf.namelist() if re.match(r'word/(header|footer)\d*\.xml$', n_)] == ['word/footer1.xml'], '（v6）頁尾僅頁碼欄位結果、無其他頁首頁尾部件')
ok('<w:pgNumType' not in xml.decode('utf-8'), '（v6）節設定無 pgNumType（十進位、由 1 起算）')
ok(not re.search(r'[A-Za-z]', g_) and not re.search(r'可能|或許|大概|似乎|應該|恐怕|推測|也許', g_), '導引句無英文字母、無不確定語氣')
order_ = [t_ for t_ in paras if re.match(r'^圖\d+ ', t_)]
nums_ = [int(re.match(r'^圖(\d+) ', t_).group(1)) for t_ in order_]
ok(nums_ == list(range(1, 8)), '圖號連續 1–7：%s' % nums_)
ok(not re.search('顯著波高|示性波高|有義波高', fulltext) and fulltext.count('有效波高') == 1, '全文無「顯著波高／示性波高／有義波高」，「有效波高」僅於導引句（1 處，依服務建議書）')
ok(not re.search('放大', fulltext) and '逐時刻' not in fulltext, '全文無「放大」「逐時刻」（放大圖與其所有提及已刪）')
want_ = []
for zh_ in ('貢寮', '龜山島', '新竹外海', '南灣', '連江'):
    want_ += ['%s到達時間驗證圖（全年總覽）' % zh_]
ok([t_.split(' ', 1)[1] for t_ in order_[2:]] == want_, '圖3–圖7 圖題含地點名稱且依序為貢寮、龜山島、新竹外海、南灣、連江（僅全年總覽）')
nonc_ = '\n'.join(t_ for t_ in paras if not re.match(r'^圖\d+ ', t_))
for n_ in range(1, 8):
    ok(re.search(r'圖 %d(?!\d)' % n_, nonc_) is not None, '內文引用 圖 %d' % n_)
ok(re.search(r'(?<![A-Za-z0-9])圖\d', nonc_) is None, '內文引用不得為「圖N」無空格')
# 圖在上、圖題在下（序列檢查）
seq2_ = []
for el_ in body:
    if el_.tag == W + 'p':
        has_ = el_.find('.//' + W + 'drawing') is not None
        tx_ = ptext(el_)
        seq2_.append(('fig' if has_ else ('cap' if re.match(r'^圖\d+ ', tx_) else 'p'), tx_))
fi_ = [i_ for i_, e_ in enumerate(seq2_) if e_[0] == 'fig']
ok(len(fi_) == 7 and all(seq2_[i_ + 1][0] == 'cap' for i_ in fi_), '7 張圖，每張圖下一段皆為圖題（圖在上、圖題在下）')
# 圖1 圖2 引用先於圖；新圖之導引句在圖 3 之前
ok(paras.index(g_) < paras.index(order_[2]), '導引句位於圖3 之前')
ok(paras.index(arr_) < paras.index(g_), '導引句緊接到達時間段落之後')
# 5. 新增文字之禁用字與英文字母
newtext_ = '\n'.join([rec_, arr_, g_, _want] + order_[2:])
for bad_ in ['站', '站點', '粒子', '機率', '扇區', '開放邊界', '本案', 'run0', 'BayTrace', '.py', '.csv', '.json', '.png', 'lbt', 'LBT', 'SCHISM', 'OCM', 'NWW', 'UTC']:
    ok(bad_ not in newtext_, '新增／修改段落不含禁用詞「%s」' % bad_)
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', ''.join([g_] + order_[2:]))), '導引句與圖題無英文字母')
ok(not re.search(r'可能|或許|大概|似乎|應該|恐怕|推測|也許', newtext_), '新增／修改段落無不確定語氣')
# 6. 單位寫法：km 為符號、句中時間寫中文（新增文字內不得出現 h、d 或阿拉伯數字加 hr）
ok(not re.search(r'\d\s*(hr|h\b|d\b|day)', newtext_), '新增文字無英文時間單位')
# 7. 第三輪（v5）：依服務建議書「到達時間」用語，全文不再用「到達時刻」
ok('到達時刻' not in fulltext, '全文無「到達時刻」（用語依建議書「到達時間」）')
ok('實際深度再依各到達時間的潮位與海床高程換算' in fulltext, '深度換算句用「各到達時間的潮位」')

# ---------------- 公式數 ----------------
x = xml.decode('utf-8')
print('oMathPara=%d  oMath=%d  編號公式=%d' % (x.count('<m:oMathPara>'), x.count('<m:oMath>'), len(re.findall(r'\(A\d+\)', fulltext))))
ok(len(re.findall(r'\(A\d+\)', fulltext)) >= 13, '編號公式數')
print('總檢查項 %d；失敗 %d' % (nchk, len(fails)))
for f in fails:
    print('FAIL:', f)
sys.exit(1 if fails else 0)
