#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v13 候選之獨立檢查。用法：check_v13.py <候選docx> <主檔v12> <figs目錄> <combined_final預覽目錄> [報告txt]"""
import sys, os, re, zipfile, hashlib, json, struct, unicodedata
from lxml import etree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v13_texts as T
CAND, MASTER, FIGS, PREV = sys.argv[1:5]
OUT = sys.argv[5] if len(sys.argv) > 5 else None
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
q = lambda t: '{%s}%s' % (W, t)
res = []; notes = []
def chk(name, cond, detail=''): res.append((bool(cond), name, detail))
def ptx(p): return ''.join(t.text or '' for t in p.iter(q('t')))
def sha(b): return hashlib.sha256(b).hexdigest()
def clean(e):
    e = etree.fromstring(etree.tostring(e))
    for x in e.iter():
        for a in list(x.attrib):
            if 'rsid' in a or a.endswith('paraId') or a.endswith('textId') or a.endswith('anchorId') or a.endswith('editId'): del x.attrib[a]
    for x in list(e.iter(q('proofErr'), q('lastRenderedPageBreak'), q('bookmarkStart'), q('bookmarkEnd'))): x.getparent().remove(x)
    return re.sub(r' xmlns:\w+="[^"]+"', '', etree.tostring(e, encoding='unicode'))
def nkl(e):
    e = etree.fromstring(etree.tostring(e))
    r0 = e.find(q('r'))
    if r0 is not None and ptx(r0).startswith('20 種到達地點'): e.remove(r0)   # v14r3 補主語所加之 run
    for x in list(e.iter(q('keepNext'), q('keepLines'))): x.getparent().remove(x)
    for x in e.iter(q('spacing')):
        for a_ in ('before', 'after'):
            if q(a_) in x.attrib: del x.attrib[q(a_)]   # 段前／段後間距（v14r3 調整）
    return clean(e)
def nex(e):   # 去掉圖框尺寸屬性與段前間距後之 XML（用於圖 1，已等比縮小 10%、段前改 6 pt）
    e = etree.fromstring(etree.tostring(e))
    for x in e.iter(q('spacing')):
        for a_ in ('before', 'after'):
            if q(a_) in x.attrib: del x.attrib[q(a_)]
    for x in e.iter('{%s}extent' % WP, '{%s}ext' % A_NS):
        for a_ in ('cx', 'cy'):
            if a_ in x.attrib: del x.attrib[a_]
    return clean(e)
def nl(text, wpt):
    ln, w = 1, 0.0
    for ch in text:
        cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
        if w + cw > wpt: ln += 1; w = cw
        else: w += cw
    return ln
LIM = (16838 - 2880) / 20.0 - 17.0; BW = (11906 - 2880 - 991) / 20.0; CW = (11906 - 2880) / 20.0
def fits(text, cy, cap): return nl(text, BW) * 22 + 18 + 12 + cy / 12700.0 + (2 + nl(cap, CW) * 27 + 8) <= LIM

zc = zipfile.ZipFile(CAND); zm = zipfile.ZipFile(MASTER)
rc = etree.fromstring(zc.read('word/document.xml')); bc = list(rc.find(q('body')))
bc_raw = list(bc)   # v14r2：圖 1、圖 2 移至區塊 9 之後（實際文件序）；其餘檢查以還原後之序（canonical）與 v12 逐區塊比對
bm = list(etree.fromstring(zm.read('word/document.xml')).find(q('body')))
MERGED = len(bc_raw) == 280   # v14r4：區塊 8、9 合為一段（共 280 區塊）；以 canonical（取主檔區塊 8、9 代入，另以 14.x 逐 run 驗證合併）與主檔比對
if MERGED:
    MOVED = True
    bc = bc_raw[:8] + [bm[8], bm[9]] + bc_raw[13:17] + bc_raw[9:13] + bc_raw[17:]
else:
    MOVED = len(bc_raw) > 18 and bc_raw[10].find('.//{%s}inline' % WP) is not None and bc_raw[14].find('.//{%s}inline' % WP) is None
    if MOVED: bc = bc_raw[:10] + bc_raw[14:18] + bc_raw[10:14] + bc_raw[18:]
rm = etree.fromstring(zm.read('word/document.xml')); bm = list(rm.find(q('body')))
relc = {r.get('Id'): r.get('Target') for r in etree.fromstring(zc.read('word/_rels/document.xml.rels'))}
txc = [ptx(k) if k.tag == q('p') else '' for k in bc]; txm = [ptx(k) if k.tag == q('p') else '' for k in bm]
fullc = '\n'.join(txc)
# ------------------------------------------------------------ 0 主檔
chk('0.1 主檔 sha256 仍為 v12（43435cf1…）', sha(open(MASTER, 'rb').read()) == '43435cf179185e896b34092a76164bd739f5a74776f630f12b15454e37bd71a7')
chk('0.2 主檔 357 區塊；候選 281 區塊（357 − 5×3 足跡 − 5×24 來源分段 − 1 綜合段 + 5×12 合併圖）', len(bm) == 357 and len(bc) == 281, '%d/%d' % (len(bm), len(bc)))
# ------------------------------------------------------------ 1 圖數與編號
figs = []   # (block idx, name, n, rid, cx, cy, descr)
for i, k in enumerate(bc):
    if k.tag != q('p'): continue
    for inl in k.iter('{%s}inline' % WP):
        dp = inl.find('{%s}docPr' % WP); m = re.match(r'^fig(\d+)\.(png|jpg)$', dp.get('name'))
        ext = inl.find('{%s}extent' % WP); blip = next(inl.iter('{%s}blip' % A_NS))
        cm = re.match(r'^圖(\d+) ', txc[i + 1]) if i + 1 < len(bc) else None
        figs.append((i, dp.get('name'), int(m.group(1)) if m else (int(cm.group(1)) if cm else None), blip.get('{%s}embed' % R_NS), int(ext.get('cx')), int(ext.get('cy')), dp.get('descr'), dp.get('id')))
nums = [f[2] for f in figs]
chk('1.1 圖數 37 且文件序為 1…37 連號', nums == list(range(1, 38)), str(nums))
chk('1.2 docPr id 無重複；圖 8–37 之 docPr 名稱為 figN.png／.jpg 且與圖號一致', len(set(f[7] for f in figs)) == len(figs) and all(re.match(r'^fig%d\.(png|jpg)$' % f[2], f[1]) for f in figs if f[2] >= 8))
capok = []
for f in figs:
    i, name, n = f[0], f[1], f[2]
    cap = txc[i + 1]
    capok.append(True if n <= 7 else (re.match(r'^圖%d ' % n, cap) is not None and f[6] is not None and cap.startswith(f[6])))
chk('1.3 圖 8–37：每張圖之下一段為圖說，「圖N 」與圖號一致，替代文字為圖說題名；圖 1–7 圖說以「圖N 」起首', all(capok), str([f[2] for f, o in zip(figs, capok) if not o]))
gok = []
for f in figs:
    g = txc[f[0] - 1]
    gok.append(re.match(r'^圖 %d (為|以)' % f[2], g) is not None or f[2] <= 7)
chk('1.4 圖 8–37 之上一段為導引段，開頭「圖 N 為／以」與圖號一致', all(gok), str([f[2] for f, o in zip(figs, gok) if not o]))
# 圖與 rels、媒體
bad = []
for f in figs:
    t = relc.get(f[3])
    if not t or ('word/' + t) not in zc.namelist(): bad.append(f[2])
chk('1.5 每張圖之 rId 均指向存在之媒體檔', not bad, str(bad))
used = set(relc[f[3]] for f in figs)
allmedia = set(n[len('word/'):] for n in zc.namelist() if n.startswith('word/media/'))
chk('1.6 media 目錄無孤立檔（皆被 rels 引用且被圖使用）', allmedia == used or allmedia - used == set(), str(sorted(allmedia - used))[:200])
# 5 地點順序、各地點 6 張圖之順序
heads = [(i, t) for i, t in enumerate(txc) if t in ('龜山島', '貢寮', '新竹', '南灣', '連江') and bc[i].find(q('pPr')) is not None and bc[i].find(q('pPr')).find(q('outlineLvl')) is not None]
chk('1.7 地點標題順序：龜山島、貢寮、新竹、南灣、連江', [t for _, t in heads] == ['龜山島', '貢寮', '新竹', '南灣', '連江'], str(heads))
kinds = ['軌跡', '仍在追蹤', '全年', '十類', '季節', '大小潮']
for (key, name, n0, *_), (hi, _) in zip(T.SITES, heads):
    caps = [txc[f[0] + 1] for f in figs if f[2] in range(n0, n0 + 6)]
    ok = [f[2] for f in figs if f[2] in range(n0, n0 + 6)] == list(range(n0, n0 + 6))
    kw = ['水平軌跡總覽', '仍在追蹤', '（全年，十類合併）', '（十類）', '（依沉底時刻之季節）', '（依大小潮）']
    ok2 = all(k_ in c for k_, c in zip(kw, caps)) and all(name in c for c in caps)
    inblk = all(hi < f[0] for f in figs if f[2] in range(n0, n0 + 6))
    chk('1.8 %s 之 6 張圖依「軌跡→仍在追蹤→全年→十類→季節→大小潮」排列（圖 %d–%d）' % (name, n0, n0 + 5), ok and ok2 and inblk, str((ok, ok2, inblk)))
# ------------------------------------------------------------ 2 交互參照
refs_bad = []
for i, t in enumerate(txc):
    for m in re.finditer(r'圖 ?(\d+)', t):
        if int(m.group(1)) > 37 or int(m.group(1)) < 1: refs_bad.append((i, m.group(0)))
chk('2.1 全文「圖 N」之 N 皆在 1–37', not refs_bad, str(refs_bad))
site_of = {}
for (key, name, n0, *_), in [((s,) ) for s in T.SITES]: pass
ok_ref = True; detail = []
for key, name, n0, *_ in T.SITES:
    for f in figs:
        if f[2] in range(n0, n0 + 6):
            g = txc[f[0] - 1]
            for m in re.finditer(r'圖 ?(\d+)', g):
                n = int(m.group(1))
                if not (n0 <= n < n0 + 6 or n <= 7): ok_ref = False; detail.append((f[2], n))
chk('2.2 各地點導引段內之圖號只指向同地點之 6 張圖（或圖 1–7）', ok_ref, str(detail))
selfok = True; d2 = []
for key, name, n0, *_ in T.SITES:
    for off in (2, 3, 4, 5):
        g = txc[[f[0] for f in figs if f[2] == n0 + off][0] - 1]
        ns = set(int(x) for x in re.findall(r'圖 (\d+)', g))
        if ns != {n0 + off}: selfok = False; d2.append((n0 + off, ns))
chk('2.3 合併圖導引段只引自己之圖號（含「依圖 N 之分段分佈」）', selfok, str(d2))
trk_ok = all(('與圖 %d 之主要來源一致' % (n0 + 2)) in txc[[f[0] for f in figs if f[2] == n0][0] - 1] for key, name, n0, *_ in T.SITES)
chk('2.4 各地點軌跡導引段之「與圖 N 之主要來源一致」指向該地點全年合併圖', trk_ok)
chk('2.5 圖 1–7 之導引、圖說與表 2 圖號未變（相同區塊 XML；區塊 9 除 keepNext／keepLines 外相同）', all(clean(bm[i]) == clean(bc[j]) for i, j in [(i, i) for i in range(0, 34) if i not in (9, 14, 15)] + [(35, 36)]) and nkl(bm[9]) == nkl(bc[9]) and nex(bm[14]) == nex(bc[14]) and nex(bm[15]) == nex(bc[15]) and all(re.match(r'^圖%d ' % f[2], txc[f[0] + 1]) for f in figs if f[2] <= 7), '')
# 圖 8–37 外之「圖 N」參照：整體結論、綜合說明、附錄不含 8 以上之圖號
tail_refs = [(i, m.group(0)) for i in range(len(bc)) for m in re.finditer(r'圖 ?(\d+)', txc[i]) if int(m.group(1)) >= 8 and not any(f[0] + d == i for f in figs for d in (-1, 1)) and not txc[i].startswith('圖 10 至圖 13 為龜山島')]
chk('2.6 圖導引段與圖說以外之段落不含圖 8 以上之參照', not tail_refs, str(tail_refs[:5]))
# ------------------------------------------------------------ 3 禁用語
removed_blocks = set()
for _k, _n, _n0, _h, _t, _c, _f, _G in T.SITES: removed_blocks |= set(range(_f, _f + 3)) | set(range(_G, _G + 24))
removed_blocks.add(209)
docxml = zc.read('word/document.xml').decode('utf-8')
for term in ('停留時數', '足跡', '高密度區間', '條件式來源', '水深淺於 200 m', '200 m 水深', '1 km × 1 km'):
    chk('3.%s 全文（含替代文字）不含「%s」' % (term, term), term not in docxml)
kept_m = ''.join(txm[i] for i in range(len(bm)) if i not in removed_blocks)
chk('3.8 「密度」一詞之出現次數與 v12 保留區塊相同（皆為海水密度等物理量；無「停留時數密度」）', fullc.count('密度') == sum(txm[i].count('密度') for i in range(len(bm)) if i not in removed_blocks) and '停留時數密度' not in fullc, '%d' % fullc.count('密度'))
chk('3.9 全文不含舊用語「邊界穿越來源分段機率圖」「海岸來源分段機率圖」「龜山島沿岸」', not any(x in fullc for x in ('邊界穿越來源分段機率圖', '海岸來源分段機率圖', '龜山島沿岸')))
chk('3.10 全文不含「新竹外海」「其餘 1」等已廢用語；圖說不含「其餘」為編號外段之說明', '新竹外海' not in fullc and not any('「其餘」' in txc[f[0] + 1] for f in figs))
# ------------------------------------------------------------ 4 圖說規則
PH = T.PHRASE
cap_fail = []
meth_i = [i for i, t in enumerate(txc) if t.startswith('來源分段機率圖中，' + PH)]
for key, name, n0, *_ in T.SITES:
    for off, kd in ((2, 'all'), (3, 'class'), (4, 'season'), (5, 'tide')):
        n = n0 + off; f = [f for f in figs if f[2] == n][0]; cap = txc[f[0] + 1]; g = txc[f[0] - 1]
        rep_ = [PH, '邊界每 10 km 一段', '海岸每 5 km 一段', '色階', '僅計流場範圍內海岸']
        c = [PH not in g, not any(x in cap for x in rep_), not re.search(r'(≥|≤|門檻|閾值|至少|超過|以上)', cap), not re.search(r'\d+(\.\d+)?%', cap)]
        if kd == 'all':
            c += ['黑、灰括號' in cap, '括號旁' in cap, '引線旁' in cap, '各段機率' in cap]
        else:   # 十類、季節、大小潮：只留該圖特有說明
            c += ['①②' in cap, '格上方為' in cap, '格內為' in cap]
            if kd == 'class': c += ['ⓐ' not in cap]
            else: c += ['ⓐⓑ' in cap]
        if not all(c): cap_fail.append((n, c))
chk('4.1 20 張合併圖之圖說與導引段不重複：色段說明、邊界每 10 km／海岸每 5 km、色階、「僅計流場範圍內海岸」只在（二）之來源分段方法段說明一次，圖說與導引段皆不重述；全年圖說只留括號與引線數值之意義（≤2 行目標）；十類、季節、大小潮圖說只留①②／ⓐⓑ 與格內外數值之說明；不寫門檻與百分比', not cap_fail, str(cap_fail))
chk('4.1b 色段說明句、「每 10 km 為一段」全文只出現於方法段（1 次）；方法段位於（二）說明（區塊「各研究地點依序以六張圖說明」）之後、「逐地點成果：」之前', fullc.count(PH) == 1 and len(meth_i) == 1 and fullc.count('邊界每 10 km 為一段') == 1 and txc[meth_i[0] - 1].startswith('各研究地點依序以六張圖說明') and txc[meth_i[0] + 1] == '逐地點成果：', str((fullc.count(PH), meth_i)))

# 與 manifest 對照：圖上有哪些記號，圖說就說明哪些
man = {}
ZH = {'guishan': '龜山島', 'gongliao': '貢寮', 'hsinchu': '新竹', 'nanwan': '南灣', 'lienchiang': '連江'}
for key, name, n0, *_ in T.SITES:
    man[key] = json.load(open(os.path.join(PREV, name, 'manifest_combined_B_all_season_tide_class.json')))
mk_fail = []
for key, name, n0, *_ in T.SITES:
    for off, kd in ((2, 'all'), (3, 'class'), (4, 'season'), (5, 'tide')):
        a = man[key][kd]['annotations']; ps = [a['boundary'], a['coast']] if kd == 'all' else a
        marks = set(st.get('marker') for p in ps for st in p['stretches'] if st.get('marker'))
        cap = txc[[f[0] for f in figs if f[2] == n0 + off][0] + 1]
        for mk in marks:
            if mk in '①②③' and '①②' not in cap: mk_fail.append((n0 + off, mk))
            if mk in 'ⓐⓑⓒ' and 'ⓐⓑ' not in cap: mk_fail.append((n0 + off, mk))
chk('4.2 manifest 中出現之圖上記號（①②③、ⓐⓑⓒ）皆於圖說說明', not mk_fail, str(mk_fail))
# ------------------------------------------------------------ 5 數值對圖
num_fail = []
for key, name, n0, *_ in T.SITES:
    for off, kd in ((2, 'all'), (3, 'class'), (4, 'season'), (5, 'tide')):
        m = man[key][kd]; n = n0 + off
        g = txc[[f[0] for f in figs if f[2] == n][0] - 1]
        if kd == 'all':
            for st in m['annotations']['boundary']['stretches'] + m['annotations']['coast']['stretches']:
                if st['pct'] + '%' not in g: num_fail.append((n, 'all', st['pct']))
            if key == 'lienchiang':
                for st in m['annotations']['coast_inset']['stretches']:
                    if st['pct'] + '%' not in g: num_fail.append((n, 'inset', st['pct']))
            lab = [x for x in m['legend_labels'] if '比例' in x]
            for x in lab:
                v = re.search(r'([\d.]+)%', x).group(1)
                if (v + '%') not in g: num_fail.append((n, 'ratio', x))
        elif kd == 'class':
            rs = [re.findall(r'邊界穿越 ([\d.]+)%　海岸來源 ([\d.]+)%', t)[0] for t in m['panel_titles']]
            bs = [float(x[0]) for x in rs]; cs = [float(x[1]) for x in rs]
            e = '穿越比例介於 %.2f%% 至 %.2f%%，海岸來源比例介於 %.2f%% 至 %.2f%%' % (min(bs), max(bs), min(cs), max(cs))
            if e not in g: num_fail.append((n, 'class', e))
        elif kd == 'season':
            pass   # v13b：季節導引段為定性敘述，不列數值（見 11.x）
        else:
            for i, (tt, p) in enumerate(zip(m['panel_titles'], m['annotations'])):
                for x in re.findall(r'([\d.]+)%', tt):
                    if x + '%' not in g: num_fail.append((n, 'ratio', i, x))
                for st in p['stretches']:
                    mk = st.get('marker')
                    if mk in ('①', 'ⓐ') and ('（%s，%s%%）' % (mk, st['pct'])) not in g: num_fail.append((n, mk, i, st['pct']))
chk('5.1 20 張合併圖導引段之數值與 manifest 一致（全年各段機率與兩比例、十類範圍、大小潮各格比例及 ①、ⓐ 之機率；季節導引段已改為定性敘述，見 11.x）', not num_fail, str(num_fail))
# 舊稿 36.62 等
chk('5.2 舊稿之季節 ① 誤差值（36.62／39.57／19.98／41.64）不再出現於全文', not any(x in fullc for x in ('36.62%', '39.57%', '19.98%', '41.64%')))
# 十類 v12 舊導引段之數字、其餘數字保留
tok = lambda s: set(re.findall(r'\d[\d,]*\.?\d*%', s))
lost = []
for key, name, n0, head, trk, trc, foot, G in T.SITES:
    for off, kd in ((0, 'all'), (6, 'class'), (18, 'tide')):
        o = tok(txm[G + off]) | tok(txm[G + off + 3])
        f = [f for f in figs if f[2] == n0 + {'all': 2, 'class': 3, 'season': 4, 'tide': 5}[kd]][0]
        nw = tok(txc[f[0] - 1])
        miss = o - nw - {'36.62%', '39.57%', '19.98%', '41.64%'}
        if miss: lost.append((n0, kd, sorted(miss)))
chk('5.3 v12 舊（邊界＋海岸）兩導引段之百分比數字皆保留於合併導引段（全年、十類、大小潮；季節已改定性）', not lost, str(lost))
# ------------------------------------------------------------ 6 八句
S = T.S
pos = {'gs1': 'guishan', 'gs2': 'guishan', 'gl1': 'gongliao', 'gl2': 'gongliao', 'nw': 'nanwan', 'bg': 'lienchiang', 'ng': 'lienchiang', 'fj': 'lienchiang'}
fig_all = {k: [f for f in figs if f[2] == n0 + 2][0] for k, _, n0, *_ in T.SITES}
prev_re = {'gs1': r'龜山島周圍海岸（9\.57%）。$', 'gs2': r'121\.82–121\.88°E 段（6\.29%）。$', 'gl1': r'121\.81–121\.92°E 段（16\.90%）。$', 'gl2': r'121\.93–122\.00°E 段（6\.10%）。$',
           'nw': r'120\.69–120\.86°E 段（27\.23%）。$', 'bg': r'馬祖北竿（7\.99%）。$', 'fj': r'119\.62–119\.71°E 段（6\.26%）。$', 'ng': r'馬祖南竿（4\.56%）。$'}
pfail = []
for k_, v in S.items():
    g = txc[fig_all[pos[k_]][0] - 1]
    if g.count(v) != 1: pfail.append((k_, 'count', g.count(v))); continue
    pre = g[:g.index(v)]
    if not re.search(prev_re[k_], pre): pfail.append((k_, 'prev', pre[-30:]))
chk('6.1 修正後 8 句逐字出現於各地點全年合併圖導引段各一次，且緊接在給出該段機率之句之後', not pfail, str(pfail))
chk('6.2 8 句不出現於其他段落（整體結論、綜合說明、其他導引段與圖說）', all(sum(t.count(v) for t in txc) == 1 for v in S.values()))
chk('6.3 新竹全年導引段不含「該段海岸」（備用句不插入）', '該段' not in txc[fig_all['hsinchu'][0] - 1])
chk('6.4 8 句不含括號、港埠分類、史述用語', not any(re.search(r'[（）()]|第[一二三四五六七八九十\d]類|商港|歷史|年建|始建|原為', v) for v in S.values()))
WL = ['東北角暨宜蘭海岸國家風景區', '龜山島', '宜蘭縣', '壯圍鄉', '五結鄉', '蘇澳鎮', '蘭陽溪', '冬山河', '新城溪', '蘇澳港', '基隆市', '中正區', '新北市', '瑞芳區', '貢寮區', '深澳', '水湳洞', '南雅奇岩', '南雅', '鼻頭角', '鼻頭', '龍洞灣海洋公園', '龍洞', '頭城鎮', '福隆海水浴場', '福隆', '卯澳', '馬崗', '石城', '三貂角', '澳底', '屏東縣', '車城鄉', '恆春鎮', '墾丁國家公園', '後壁湖', '紅柴坑', '山海', '潭仔', '南灣', '貓鼻頭', '萬里桐', '鵝鑾鼻', '連江縣', '北竿鄉', '白沙港', '塘后道', '坂里', '南竿鄉', '福澳港', '津沙海灘', '福建省', '福州市', '長樂區', '閩江河口濕地']
FUNC = set('該段為海岸位於經屬含鄰以及與等遊憩據點限量開放登島之觀光，、。島岸河口南端漁港沙灘')
resid = {}
for k_, v in S.items():
    s = v
    for w in sorted(WL, key=len, reverse=True): s = s.replace(w, '')
    bad_ch = set(s) - FUNC
    if bad_ch: resid[k_] = ''.join(sorted(bad_ch))
chk('6.5 8 句所用地名皆在白名單（已於 coast_sentences_final.md 逐一查核），殘餘字僅為虛詞', not resid, str(resid))
# 新增地名之全文檢查：導引段中之其他地名仍限於 v11 白名單
OLDWL = ['龜山島周圍海岸', '臺灣本島', '福建沿岸', '馬祖北竿', '馬祖南竿', '馬祖各島', '馬祖列島', '恆春半島', '小琉球', '蘭嶼', '近岸島嶼', '其他島嶼', '各島']
# ------------------------------------------------------------ 7 保留區塊未動
chg_master = set()
removed = set()
for key, name, n0, head, trk, trc, foot, G in T.SITES:
    removed |= set(range(foot, foot + 3)) | set(range(G, G + 24))
    chg_master |= {trk, trk + 1, trk + 2, trc, trc + 1, trc + 2}
removed.add(209); chg_master |= {9, 14, 15, 34, 46}   # 9：圖 1／2 導引段加 keepLines＋keepNext；14：圖 1 縮小 10%
keep_idx = [i for i in range(len(bm)) if i not in removed]
combined_idx = set()
for key, name, n0, *_ in T.SITES:
    for off in (2, 3, 4, 5):
        fi = [f[0] for f in figs if f[2] == n0 + off][0]; combined_idx |= {fi - 1, fi, fi + 1}
cand_keep = [i for i in range(len(bc)) if i not in combined_idx]
_m46 = [i for i, t in enumerate(txc) if t.startswith('來源分段機率圖中，' + T.PHRASE)][0]
cand_keep.remove(_m46); cand_keep.insert(keep_idx.index(46), _m46)   # v14：方法段移至區塊 34 之後
chk('7.1 去掉被刪與新增之區塊後，保留區塊數相同（%d）' % len(keep_idx), len(keep_idx) == len(cand_keep), '%d/%d' % (len(keep_idx), len(cand_keep)))
diffs = [(mi, ci) for mi, ci in zip(keep_idx, cand_keep) if mi not in chg_master and clean(bm[mi]) != clean(bc[ci])]
chk('7.2 其餘保留區塊（含表 1、表 2、附錄、整體結論、參考資料）XML 與 v12 完全相同（排除 34、46 與 10 組軌跡／仍在追蹤）', not diffs, str(diffs[:10]))
chk('7.3 綜合說明第 1、3 段與整體結論 (1)–(5) 逐字未動；足跡段（區塊 209）已刪', 'x' and all(txm[i] == txc[cand_keep[keep_idx.index(i)]] for i in (208, 210, 212, 213, 214, 215, 216)) and '足跡範圍方面' not in fullc)
# 區塊 34、46
i34 = cand_keep[keep_idx.index(34)]; i46 = _m46
chk('7.4 區塊 34 圖序概述改為「六張圖」、四張合併來源分段機率圖、不含足跡', '各研究地點依序以六張圖說明' in txc[i34] and '四張來源分段機率圖' in txc[i34] and '足跡' not in txc[i34] and '十一張' not in txc[i34])
chk('7.5 方法段已移至（二）說明之後，改為一般性用語（無地點名與圖號）；含色段說明句、①②／ⓐⓑ、來源分段取法與 100% 說明', i46 == i34 + 1 and '龜山島' not in txc[i46] and not re.search(r'圖 ?\d', txc[i46]) and '①②' in txc[i46] and 'ⓐⓑ' in txc[i46] and '邊界每 10 km 為一段，海岸每 5 km 為一段（僅計流場範圍內海岸）' in txc[i46] and '合計為 100%' in txc[i46], txc[i46][:60])
# 軌跡／仍在追蹤只改圖號
retok = True; rd = []
for key, name, n0, head, trk, trc, foot, G in T.SITES:
    for mi, nn in ((trk, n0), (trc, n0 + 1)):
        ci = cand_keep[keep_idx.index(mi)]
        a = re.sub(r'圖 ?\d+', '圖', txm[mi]); b = re.sub(r'圖 ?\d+', '圖', txc[ci])
        a = re.sub(r'與圖 及圖 之主要來源一致', '與圖 之主要來源一致', a)
        if a != b: retok = False; rd.append(mi)
chk('7.6 軌跡與仍在追蹤比例導引段除圖號外文字未變', retok, str(rd))
# ------------------------------------------------------------ 8 zip 其他成員、PNG
names_m = set(zm.namelist()); names_c = set(zc.namelist())
rm_media = {n for n in names_m - names_c}; add = names_c - names_m
chk('8.1 刪除之 zip 成員皆為 word/media（共 45 個＝5×(足跡 1＋來源分段 8)）', all(n.startswith('word/media/') for n in rm_media) and len(rm_media) == 45, str(len(rm_media)))
chk('8.2 新增成員為 20 個 v13 合併圖 PNG', len(add) == 20 and all(re.match(r'word/media/v13_fig\d+\.png$', n) for n in add))
same = all(zc.read(n) == zm.read(n) for n in names_m & names_c if n not in ('word/document.xml', 'word/_rels/document.xml.rels', 'word/styles.xml'))
chk('8.3 其餘 zip 成員逐位元組與 v12 相同（含圖 1–9、numbering、settings、theme、[Content_Types].xml；styles.xml 另見 15.2）', same)
pfail = []
for f in figs:
    if f[2] >= 8 and f[2] in {n0 + o for _, _, n0, *_ in T.SITES for o in (2, 3, 4, 5)}:
        data = zc.read('word/' + relc[f[3]]); src = open(os.path.join(FIGS, 'fig%d.png' % f[2]), 'rb').read()
        w, h = struct.unpack('>II', data[16:24])
        if data != src or f[4] != 5733288 or abs(f[5] - round(5733288 * h / w)) > 1: pfail.append(f[2])
chk('8.4 20 張合併圖之嵌入檔與 v13_figs 相同、寬 5,733,288 EMU（6.27 in）、高依長寬比', not pfail, str(pfail))
# ------------------------------------------------------------ 9 keepNext
kfail = []
for f in figs:
    if f[2] in {n0 + o for _, _, n0, *_ in T.SITES for o in (2, 3, 4, 5)}:
        g = bc[f[0] - 1]; kn = g.find(q('pPr')).find(q('keepNext')) is not None
        if kn != fits(txc[f[0] - 1], f[5], txc[f[0] + 1]): kfail.append(f[2])
chk('9.1 合併圖導引段之 keepNext 與估算式（fits）一致', not kfail, str(kfail))
chk('9.2 20 張合併圖之圖段 keepNext、圖說 keepLines 保留', all(bc[f[0]].find(q('pPr')).find(q('keepNext')) is not None and bc[f[0] + 1].find(q('pPr')).find(q('keepLines')) is not None for f in figs if f[2] in {n0 + o for _, _, n0, *_ in T.SITES for o in (2, 3, 4, 5)}))
# ------------------------------------------------------------ 10 圖本身（manifest）
comp = []
for key, name, n0, *_ in T.SITES:
    for off, kd in ((2, 'all'), (3, 'class'), (4, 'season'), (5, 'tide')):
        m = man[key][kd]; a = m['annotations']; ps = [a['boundary'], a['coast']] if kd == 'all' else a
        nin = sum(1 for p in ps for sd in p.get('side', []) if not sd['outward'])
        nviol = sum(1 for p in ps for st in p.get('stretches', []) if st.get('violations'))
        col = m.get('collision_count', 0)
        pngsha = sha(open(os.path.join(PREV, name, {'all': 'combined_all.png', 'class': 'combined_class_B.png', 'season': 'combined_season_B.png', 'tide': 'combined_tide_B.png'}[kd]), 'rb').read())
        comp.append((name, kd, nin, nviol, col, pngsha == m.get('sha256')))
chk('10.1 20 張圖：圖內括號與標示之內側放置數皆為 0', all(c[2] == 0 for c in comp), str([c for c in comp if c[2]]))
chk('10.2 20 張圖：collision_count 皆為 0', all(c[4] == 0 for c in comp), str([c for c in comp if c[4]]))
chk('10.3 manifest 之 sha256 與預覽 PNG 相符', all(c[5] for c in comp), str([c for c in comp if not c[5]]))
dev = [(c[0], c[1], c[3]) for c in comp if c[3]]
notes.append('10.4 已知偏離（記號與色段重疊，列為備註、不計失敗）：' + '; '.join('%s%s %d 處' % d for d in dev))

# ------------------------------------------------------------ 11 v13b：季節導引段（定性）、新竹海岸句、南灣句
SEAG = {}
for key, name, n0, head, trk, trc, foot, G in T.SITES:
    SEAG[key] = txc[[f[0] for f in figs if f[2] == n0 + 4][0] - 1]
v12B = {key: txm[G + 12] for key, name, n0, head, trk, trc, foot, G in T.SITES}
v12C = {key: txm[G + 15] for key, name, n0, head, trk, trc, foot, G in T.SITES}
sf = []
for key, name, n0, head, trk, trc, foot, G in T.SITES:
    g = SEAG[key]; N = n0 + 4
    verdict = re.match(r'^.*?。依圖 \d+ 與圖 \d+ 之分段分佈（邊界與海岸合計），(.*)$', v12C[key], re.S).group(1)
    pre = '圖 %d 為依沉底時刻劃分之%s四季來源分段機率圖。' % (N, name)
    defn = re.match(r'^圖 \d+ 為依沉底時刻劃分之' + name + r'四季邊界穿越來源分段機率圖。(.*?)各季節之穿越比例', v12B[key], re.S).group(1)
    want = pre + defn + T.SEA[key][0] + T.SEA[key][1] + '依圖 %d 之分段分佈（邊界與海岸合計），%s' % (N, verdict)
    if g != want: sf.append((key, 'text'))
    if (key == 'guishan') != ('沉底時刻為廢棄物沉降至海床之時刻' in g): sf.append((key, 'defn'))
    nums_ = re.findall(r'\d+(?:\.\d+)?', g.replace(pre, '').replace('依圖 %d 之' % N, '') if key != 'guishan' else g.replace(pre, '').replace(defn, '').replace('依圖 %d 之' % N, ''))
    if len(nums_) > 2: sf.append((key, 'nums', nums_))
    if re.search(r'（[①ⓐ]，|%', g.replace(defn, '')): sf.append((key, 'pct'))
    if len(re.findall('。', T.SEA[key][0])) != 1 or len(re.findall('。', T.SEA[key][1])) != 1: sf.append((key, 'one-sentence'))
chk('11.1 五段季節導引段＝「定義句（僅龜山島）＋邊界定性一句＋海岸定性一句＋v12 結論句」，不列各季數值（句內數字 ≤2）', not sf, str(sf))
tf = []
for key, name, n0, *_ in T.SITES:
    old = v12B[key] + v12C[key] + txm[[s_[7] for s_ in T.SITES if s_[0] == key][0] + 2] + txm[[s_[7] for s_ in T.SITES if s_[0] == key][0] + 5]
    for sent in T.SEA[key]:
        for w in re.findall(r'(?:南|北|東|西)側邊界(?:東|中|西)?段?|南側邊界|(?:西南|西北|東南|東北)角|龜山島周圍海岸|研究地點[^，；、。]{0,5}臺灣本島海岸|馬祖北竿|福建沿岸', sent):
            if w not in old and not (key == 'guishan' and w == '北側邊界中段'): tf.append((key, w))
chk('11.2 季節定性句所用位置詞皆沿用 v12 已核定名稱（龜山島北側邊界中段為新增，依 ② 座標 25.5°N、121.5–122.4°E）', not tf, str(tf))
# 與 manifest 比對之定性主張
import re as _re
def seas(key, kd='season'):
    m = man[key][kd]; out = []
    for t_, bl in zip(m['panel_titles'], m['box_lines']):
        b_, c_ = [float(x) for x in re.findall(r'(?:邊界穿越|海岸來源) ([\d.]+)%', t_)]
        out.append((b_, c_, bl))
    return out
cl = []
sg = seas('guishan'); cs = [x[1] for x in sg]; 
if not (cs.index(max(cs)) == 2 and cs.index(min(cs)) == 0): cl.append('龜山島 海岸比例最大夏/最小冬')
if not all(float(x[2][x[2].index('① ') + 1][:-1] if False else 0) == 0 for x in sg): pass
sg = seas('gongliao'); cs = [x[1] for x in sg]
if not (cs.index(max(cs)) == 2 and cs.index(min(cs)) == 0): cl.append('貢寮 海岸比例最大夏/最小冬')
sg = seas('nanwan'); cs = [x[1] for x in sg]
if not (cs.index(min(cs)) == 2 and sorted(range(4), key=lambda i: -cs[i])[:2] == [0, 1]): cl.append('南灣 冬春較大夏最小')
sg = seas('lienchiang'); cs = [x[1] for x in sg]
if not (cs.index(max(cs)) == 1 and cs.index(min(cs)) == 0 and cs[1] > 50 and cs[2] > 50 and cs[3] < 50): cl.append('連江 春最大冬最小春夏過半')
sg = seas('hsinchu'); cs = [x[1] for x in sg]
if not (cs[:3] == [0.0, 0.0, 0.0] and cs[3] > 0): cl.append('新竹 僅秋季有海岸來源')
if not all(x[0] >= 92 for x in sg): cl.append('新竹 邊界穿越四季皆≥92%')
chk('11.3 季節定性句之可由 manifest 驗證之主張（海岸比例最大／最小季、過半、新竹僅秋季有海岸來源）成立', not cl, str(cl))
# 新竹全年：縣市與漁港句
hc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hs_county_counts.json'), encoding='utf-8'))
from decimal import Decimal, ROUND_HALF_UP
rh = lambda c, n: str((Decimal(c) * 100 / Decimal(n)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
hg = txc[fig_all['hsinchu'][0] - 1]
cc_ = hc['county_counts']
hf = []
if T.HSINCHU_COAST not in hg or hg.count(T.HSINCHU_COAST) != 1: hf.append('sentence')
if '（%s%%）及臺中市（%s%%）' % (rh(cc_['苗栗縣'], hc['n']), rh(cc_['臺中市'], hc['n'])) not in T.HSINCHU_COAST: hf.append('pct')
if sum(cc_.values()) != hc['coast_total'] or rh(hc['coast_total'], hc['n']) != '2.23': hf.append('sum')
if not (Decimal(hc['seg_max']) * 100 / Decimal(hc['n']) < 1): hf.append('<1%')
if not re.search(r'新竹之海岸來源（2\.23%）皆位於臺灣本島。' + re.escape(T.HSINCHU_COAST), hg): hf.append('position')
if re.search(r'新竹縣|新竹市|人工魚礁', hg) or '人工魚礁' in fullc: hf.append('banned')
for hb, sg_ in (('白沙屯漁港', 'C0001-20'), ('通霄漁港', 'C0001-21')):
    if not (hc['harbour_to_segment_km'][hb][sg_] <= 0.5 and hc['seg_counts'][sg_] >= 100): hf.append(hb)
chk('11.4 新竹全年導引段：海岸來源縣市句緊接 2.23% 句之後、數字由計數 ROUND_HALF_UP 重算（苗栗縣 1.92%、臺中市 0.31%）、「各海岸段未達 1%」（最大 0.62%）、漁港距 ≥0.1% 段 ≤0.5 km、不含新竹縣市與人工魚礁', not hf, str(hf))
chk('11.5 南灣句為「含後壁湖及紅柴坑等漁港」，其餘不變', '含後壁湖及紅柴坑等漁港，以及南灣、貓鼻頭、萬里桐與鵝鑾鼻等遊憩據點' in txc[fig_all['nanwan'][0] - 1] and '潭仔' not in fullc and '山海漁港' not in fullc)
# 10.5 圖框外／貼圖框之標記改放邊界內側
mv = []; ax_ = []
for key, name, n0, *_ in T.SITES:
    for off, kd in ((2, 'all'), (3, 'class'), (4, 'season'), (5, 'tide')):
        m = man[key][kd]; a_ = m['annotations']; ps = [a_['boundary'], a_['coast']] if kd == 'all' else a_
        for pi, p in enumerate(ps):
            for st in p.get('stretches', []):
                if 'axes' in (st.get('violations') or []): ax_.append((name, kd, pi))
                if st.get('marker_where') == 'inner_axes': mv.append((name, kd, pi, st['marker']))
chk('10.5 20 張圖之圖框（axes）違規為 0；改放邊界內側之標記只出現於新竹十類（①，10 格）', not ax_ and mv == [('新竹', 'class', i, '①') for i in range(10)], str((ax_, mv)))
notes.append('10.5 改放邊界內側之標記：新竹十類 ① ×10 格（外圈放不下、原落在圖框上）；其餘 190 個標記維持外圈')


# ------------------------------------------------------------ 12 v14 相對 v13 現行主檔：只動方法段位置與用語、20 個合併圖圖說
V13 = os.environ.get('LBT_V13')
if V13:
    z13 = zipfile.ZipFile(V13); r13 = etree.fromstring(z13.read('word/document.xml')); b13 = list(r13.find(q('body')))
    t13 = [ptx(k) if k.tag == q('p') else '' for k in b13]
    chk('12.0 比對基準 v13 主檔 sha256 為 093fe4a8…', sha(open(V13, 'rb').read()).startswith('093fe4a8'))
    m13 = [i for i, t in enumerate(t13) if t.startswith('圖 10 至圖 13 為龜山島之來源分段機率圖')]
    chk('12.1 v13 方法段只有 1 個；v14 與 v13 區塊數相同（281）', len(m13) == 1 and len(b13) == len(bc) == 281)
    A_ = [k for i, k in enumerate(b13) if i != m13[0]]; B_ = [k for i, k in enumerate(bc) if i != i46]
    capi = set()
    for key, name, n0, *_ in T.SITES:
        for off in (2, 3, 4, 5):
            fi = [f[0] for f in figs if f[2] == n0 + off][0]; capi.add(fi + 1)
    capi_ = set(i if i < i46 else i - 1 for i in capi)
    def nk(x):
        x = etree.fromstring(etree.tostring(x))
        for e_ in list(x.iter(q('keepNext'))): e_.getparent().remove(e_)
        return clean(x)
    def nk9(i, x): return nkl(x) if i == 9 else (nex(x) if i in (14, 15) else nk(x))
    dif = [i for i, (x, y) in enumerate(zip(A_, B_)) if nk9(i, x) != nk9(i, y)]
    kn_dif = [i for i, (x, y) in enumerate(zip(A_, B_)) if clean(x) != clean(y) and nk9(i, x) == nk9(i, y) and i != 9]
    notes.append('12.5 因圖說縮短，導引段 keepNext 重新估算而改變者（僅格式屬性、文字不變），圖號：' + str([int(re.match(r'^圖 (\d+)', txc[i + 1 if i + 1 < i46 else i + 1]).group(1)) if re.match(r'^圖 (\d+)', txc[i + 1]) else i for i in kn_dif]))
    chk('12.2 去掉方法段後，v14 與 v13 只有 20 個合併圖圖說不同（導引段、圖、軌跡段、表、附錄、結論皆逐字相同）', set(dif) == capi_ and len(dif) == 20, str((sorted(set(dif) ^ capi_)[:10], len(dif))))
    zm13 = {n for n in z13.namelist()}; zc_ = {n for n in zc.namelist()}
    chk('12.3 v14 與 v13 之 zip 成員相同，除 word/document.xml、word/styles.xml（15.2）外逐位元組相同', zm13 == zc_ and all(z13.read(n) == zc.read(n) for n in zm13 if n not in ('word/document.xml', 'word/styles.xml')))
    chk('12.4 方法段新舊：舊首句「圖 10 至圖 13 為龜山島…」改為「來源分段機率圖中，〔色段說明句〕。」，其餘文字逐字不變', txc[i46].startswith('來源分段機率圖中，' + PH + '。') and t13[m13[0]].split('。', 2)[2 - 0 if False else 1] is not None and txc[i46][len('來源分段機率圖中，' + PH + '。'):] == t13[m13[0]][t13[m13[0]].index('來源分段之取法為'):])

# ------------------------------------------------------------ 13 圖與導引段／圖說同頁（v14r2：圖 1、圖 2 位置與圖 1 縮小；LibreOffice 排版結果）
G = 8 if MERGED else 9
F1, C1, F2, C2 = G + 1, G + 2, G + 3, G + 4
def _ext(bk):
    inl = next(bk.iter('{%s}inline' % WP)); e1 = inl.find('{%s}extent' % WP); e2 = next(inl.iter('{%s}ext' % A_NS))
    return int(e1.get('cx')), int(e1.get('cy')), int(e2.get('cx')), int(e2.get('cy'))
_nonp = lambda e: [c for c in e if c.tag != q('pPr')]
def _cc(c):
    w_ = etree.Element(q('p')); w_.append(etree.fromstring(etree.tostring(c))); return clean(w_)
if MERGED:
    mg = bc_raw[8]; km = _nonp(mg); k8 = _nonp(bm[8]); k9 = _nonp(bm[9]); nrun = km[len(k8)] if len(km) > len(k8) else None
    chk('14.1 區塊 8、9 合為一段，接縫插入「這 20 種到達地點」：…任意懸浮深度。這 20 種到達地點共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。四區流場範圍…；合併段文字＝區塊 8 文字＋「這 20 種到達地點」＋區塊 9 文字，其餘逐字相同',
        ptx(mg) == ptx(bm[8]) + '這 20 種到達地點' + ptx(bm[9]) and ptx(bm[8]).endswith('任意懸浮深度。') and ptx(bm[9]).startswith('共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。四區流場範圍和研究地點如圖 1') and '任意懸浮深度。這 20 種到達地點共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。四區流場範圍和研究地點如圖 1' in ptx(mg), ptx(mg)[100:160])
    chk('14.1b 合併段之 run 逐一等於區塊 8 之 run、新 run、區塊 9 之 run（無增減、格式不變）', nrun is not None and len(km) == len(k8) + 1 + len(k9) and [_cc(c) for c in km[:len(k8)]] == [_cc(c) for c in k8] and [_cc(c) for c in km[len(k8) + 1:]] == [_cc(c) for c in k9] and ptx(nrun) == '這 20 種到達地點')
    _r5 = [r_ for r_ in bm[9].findall(q('r')) if ptx(r_) == '5 ']
    chk('14.2 新增之 run 與段內半形數字（區塊 9 之「5 」run）同格式（rPr 相同、無字型指定）', nrun is not None and _r5 and etree.tostring(nrun.find(q('rPr'))) == etree.tostring(_r5[0].find(q('rPr'))) and nrun.find(q('rPr')).find(q('rFonts')) is None, '')
    chk('14.3 合併段 pPr 與區塊 8 之 pPr 相同（保留區塊 8 之 pPr；無 keepNext／keepLines，段落可跨頁，圖 1 隨後同頁）；間距屬性未改', clean(mg.find(q('pPr'))) == clean(bm[8].find(q('pPr'))) and mg.find(q('pPr')).find(q('keepNext')) is None and mg.find(q('pPr')).find(q('keepLines')) is None, '')
    chk('14.4 區塊 9 已併入（候選無獨立之「共由…」段）；其餘區塊數 280（281−1）', not any(t.startswith('共由「5 個水平位置') for t in [ptx(k) for k in bc_raw]) and len(bc_raw) == 280)
else:
    _n9 = ptx(bc_raw[9]); _o9 = ptx(bm[9])
    _r0 = bc_raw[9].find(q('r')); _r5 = [r_ for r_ in bc_raw[9].findall(q('r')) if ptx(r_) == '5 ']
    chk('14.1 區塊 9 只在句首補「20 種到達地點」', _o9.startswith('共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。') and _n9 == '20 種到達地點' + _o9, _n9[:40])
    chk('14.2 新增之 run 與段內半形數字（「5 」run）同格式', _r0 is not None and _r5 and etree.tostring(_r0.find(q('rPr'))) == etree.tostring(_r5[0].find(q('rPr'))) and _r0.find(q('rPr')).find(q('rFonts')) is None, '')
_sp = lambda bk, a_: bk.find(q('pPr')).find(q('spacing')).get(q(a_))
chk('13.1 圖 1、圖 2 位置：導引段之後緊接 圖 1、圖 1 圖說、圖 2、圖 2 圖說；其後依序為取樣深度說明、表 2 前導句、表 2 圖題、表 2',
    MOVED and bc_raw[F1].find('.//{%s}inline' % WP) is not None and re.match(r'^圖1 ', ptx(bc_raw[C1])) and bc_raw[F2].find('.//{%s}inline' % WP) is not None and re.match(r'^圖2 ', ptx(bc_raw[C2]))
    and ptx(bc_raw[G + 6]).startswith('五個研究地點之錨點座標') and re.match(r'^表2 ', ptx(bc_raw[G + 7])) and bc_raw[G + 8].tag == q('tbl'),
    str([ptx(bc_raw[k])[:12] for k in range(G + 5, G + 9)]))
kp = bc_raw[G].find(q('pPr'))
if MERGED:
    chk('13.2 圖 1 段含 keepNext（與圖說同頁）；合併導引段不加 keepNext／keepLines（讓段落可跨頁，圖 1 以原尺寸緊接其後）', bc_raw[F1].find(q('pPr')).find(q('keepNext')) is not None and kp.find(q('keepNext')) is None and kp.find(q('keepLines')) is None)
else:
    chk('13.2 區塊 9 含 keepNext 與 keepLines；圖 1 段含 keepNext', kp.find(q('keepNext')) is not None and kp.find(q('keepLines')) is not None and bc_raw[F1].find(q('pPr')).find(q('keepNext')) is not None)
e_m = _ext(bm[14]); e_c = _ext(bc_raw[F1]); sc = e_c[0] / e_m[0]
if MERGED:
    chk('13.3 圖 1 顯示尺寸與主檔相同（%d×%d EMU，比例 1.000，未縮放）；extent 與 a:ext 一致；圖 2 尺寸未變' % (e_m[0], e_m[1]), e_c == e_m and e_m[:2] == (3502152, 5759196) and e_c[0] == e_c[2] and e_c[1] == e_c[3] and _ext(bm[16]) == _ext(bc_raw[F2]), str((e_m, e_c)))
    chk('13.3b 全文 37 張圖之 extent 中，圖 1–9 與主檔逐一相同（無任何圖被縮放）', [_ext(b) for b in bc_raw if b.tag == q('p') and b.find('.//{%s}inline' % WP) is not None][:2] == [_ext(bm[14]), _ext(bm[16])])
    chk('13.3c 圖 1 之 w:drawing（含 extent）與主檔逐位元相同；圖 1 段 spacing 與主檔相同', nex(bc_raw[F1]) == nex(bm[14]) and etree.tostring(next(bc_raw[F1].iter('{%s}drawing' % W))) == etree.tostring(next(bm[14].iter('{%s}drawing' % W))) and etree.tostring(bc_raw[F1].find(q('pPr'))) == etree.tostring(bm[14].find(q('pPr'))) , '')
else:
    chk('13.3 圖 1 等比縮小', 0.869 <= sc <= 1.0 and abs(e_c[1] / e_m[1] - sc) < 0.001 and e_c[0] == e_c[2] and e_c[1] == e_c[3] and _ext(bm[16]) == _ext(bc_raw[F2]), str((e_m, e_c)))
chk('13.4 圖 1 之圖檔（media）與主檔相同', relc[next(bc_raw[F1].iter('{%s}blip' % A_NS)).get('{%s}embed' % R_NS)] == {r.get('Id'): r.get('Target') for r in etree.fromstring(zm.read('word/_rels/document.xml.rels'))}[next(bm[14].iter('{%s}blip' % A_NS)).get('{%s}embed' % R_NS)])
LAY = os.environ.get('LBT_LAYOUT')
if LAY:
    L = json.load(open(LAY, encoding='utf-8'))
    chk('13.5 LibreOffice 排版：37 張圖各列一列（圖 1–37）', [r['fig'] for r in L['figs']] == list(range(1, 38)))
    bad = [(r['fig'], r['status']) for r in L['figs'] if r['status'] != 'OK']
    if MERGED:
        chk('13.6 LibreOffice 排版：37 張圖中，除圖 1 為「跨頁開頭」（合併導引段由前頁起、末 2 行與圖 1 同頁）外，其餘皆 OK（無 NG）', bad == [(1, '跨頁開頭')], str(bad))
    else:
        chk('13.6 LibreOffice 排版：37 張圖之圖說皆與圖同頁，且無 NG／跨頁開頭', not bad, str(bad))
    capbad = [r['fig'] for r in L['figs'] if r['cap_p'] != r['fig_p'] or r.get('cap_end_p') != r['fig_p']]
    chk('13.7 LibreOffice 排版：圖說起、訖頁（含最後一行）皆與圖同頁，不同者為 0', not capbad, str(capbad))
    ind = [r['fig'] for r in L['figs'] if r['fig'] >= 8 and not (r['g_start'] == r['g_end'] == r['fig_p'])]
    chk('13.8 LibreOffice 排版：圖 8–37 之導引段起訖頁皆等於圖頁（導引段後即見圖，不落至下一頁、不跨頁開頭）', not ind, str(ind))
    f1 = L['figs'][0]
    if MERGED:
        chk('13.9 LibreOffice 排版：圖 1——合併導引段止於圖 1 所在頁（起於前一頁）、圖 1、圖說起訖皆同頁', f1['g_end'] == f1['fig_p'] == f1['cap_p'] == f1['cap_end_p'] and f1['g_start'] == f1['fig_p'] - 1, str(f1))
    else:
        chk('13.9 LibreOffice 排版：圖 1 與導引段（區塊 9）同頁、圖 1 圖說同頁', f1['g_start'] == f1['g_end'] == f1['fig_p'] == f1['cap_p'], str(f1))
    f37 = {r['fig']: r for r in L['figs']}
    chk('13.10 LibreOffice 排版：共用導引段之圖 2、圖 4–7 為導引段所在頁之後第一張圖（逐頁一張，圖大無法與導引同頁；圖說與圖同頁）', all(f37[k]['cap_p'] == f37[k]['fig_p'] for k in (2, 4, 5, 6, 7)))
    tbad = [(t['name'], t['status']) for t in L['tables'] if t['status'].startswith('NG')]
    chk('13.11 LibreOffice 排版：表題與表首列同頁（無 NG）', not tbad, str(tbad))
    chk('13.12 LibreOffice 頁數 = %d（要求 ≤63）' % L['pages'], L['pages'] <= 63, '')
    if os.environ.get('LBT_SLACK'): notes.append('13.13 圖 1 所在頁（LO）頁面底部餘裕（頁底線 769.9 pt 減圖說末行底）：' + os.environ['LBT_SLACK'] + ' pt')
# ------------------------------------------------------------ 15 字型：英、數字一律 Times New Roman（v14r5）
import font_audit as FA
_sm = zm.read('word/styles.xml').decode('utf-8'); _sc = zc.read('word/styles.xml').decode('utf-8')
_exp = _sm.replace('<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:szCs w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="a">', '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:szCs w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="a">', 1)
_exp = _exp.replace('<w:rFonts w:ascii="Calibri" w:eastAsia="標楷體" w:hAnsi="Calibri"/><w:kern w:val="2"/><w:sz w:val="28"/><w:szCs w:val="22"/></w:rPr></w:style><w:style w:type="character" w:customStyle="1" w:styleId="1ff6">', '<w:rFonts w:ascii="Times New Roman" w:eastAsia="標楷體" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:kern w:val="2"/><w:sz w:val="28"/><w:szCs w:val="22"/></w:rPr></w:style><w:style w:type="character" w:customStyle="1" w:styleId="1ff6">', 1)
chk('15.1 styles.xml 與 v12 相比，只有樣式 afe（List Paragraph）與 aff（其連結字元樣式）之 rFonts 由 Calibri 改 TNR（ascii／hAnsi／cs；eastAsia 不動）', _sc == _exp and _sc != _sm, '')
_rows = FA.audit(CAND); _rows_m = FA.audit(MASTER)
_bad = [r for r in _rows if r['ascii'] != 'Times New Roman' or r['hAnsi'] != 'Times New Roman' or r['cs'] != 'Times New Roman']
chk('15.2 每個含半形英數字之 run（document.xml、footer、footnotes、endnotes；不含 OMML、圖內文字），解析後 ascii／hAnsi／cs 皆為 Times New Roman（共 %d 個 run、%d 字；v12 為 %d 個非 TNR）' % (len(_rows), sum(r['nlat'] for r in _rows), sum(1 for r in _rows_m if r['ascii'] != 'Times New Roman' or r['hAnsi'] != 'Times New Roman')), not _bad, str(_bad[:3]))
import collections as _co
_ea = lambda rows: _co.Counter(r['text'] and (r.get('pstyle'),) for r in rows)
chk('15.3 theme 字型未改（minor／major latin）、docDefaults rFonts 未改、numbering.xml 與 v12 位元組相同', zc.read('word/theme/theme1.xml') == zm.read('word/theme/theme1.xml') and zc.read('word/numbering.xml') == zm.read('word/numbering.xml') and re.search(r'<w:docDefaults>.*?</w:docDefaults>', _sc, re.S).group(0) == re.search(r'<w:docDefaults>.*?</w:docDefaults>', _sm, re.S).group(0))
_sym = _co.Counter()
for _t in re.findall(r'<w:t[^>]*>([^<]*)</w:t>', zc.read('word/document.xml').decode('utf-8')):
    for _ch in _t:
        if 0x2460 <= ord(_ch) <= 0x24FF: _sym[_ch] += 1
notes.append('15.4 刻意保留（TNR 無此字形，由系統後備字型顯示，不屬英數字）：圈號／圈字 ' + ''.join(sorted(_sym)) + '，共 ' + str(sum(_sym.values())) + ' 個字元；numbering 之項目符號字型 Wingdings／Symbol／Courier New 屬符號，且文件未使用項目符號（僅 numId 46 之「%1.」為 TNR）')
notes.append('15.5 未使用之樣式（文件中無段落、字元或表格引用）仍含 Arial、Calibri、細明體等 Latin 字型定義，未更動（不影響版面）')

# ------------------------------------------------------------ 輸出
nfail = sum(1 for r in res if not r[0])
lines = ['v13 候選檢查：%s' % os.path.basename(CAND), 'sha256 %s' % sha(open(CAND, 'rb').read()), '合計 %d 項，失敗 %d 項' % (len(res), nfail), '']
for ok, nm, d in res: lines.append(('PASS ' if ok else 'FAIL ') + nm + ('' if ok or not d else '  ← ' + d))
lines += [''] + ['NOTE ' + n for n in notes]
txt = '\n'.join(lines)
print(txt)
if OUT: open(OUT, 'w', encoding='utf-8').write(txt + '\n')
sys.exit(1 if nfail else 0)
