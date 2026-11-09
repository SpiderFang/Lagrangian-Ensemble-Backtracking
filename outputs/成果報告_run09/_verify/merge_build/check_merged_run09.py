#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成果報告_run09_合併版.docx 之獨立檢查（v4：合併版為主檔，使用者已在 Word 手改）。
不匯入 merge_run09.py；「v2 重建」一項以子程序執行該腳本於暫存目錄，僅作比對基準。
用法：python3 check_merged_run09.py [合併版docx] [pdf(選用)]"""
import sys, os, re, zipfile, hashlib, subprocess, tempfile, json, unicodedata
from decimal import Decimal
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
SRC_A = os.path.join(BASE, '3-4-3_run09.docx')
SRC_B = os.path.join(BASE, '執行成果_run09_含斯托克斯漂流.docx')
SHA_A = '2266588cfaac1140250ef4989e8c7ef1207154bc153a2977c7487f3555d415ad'
SHA_B = 'a22d0be9ed895158dfe96ddde34ae2d5b9762f55eacae96b939d19b1734e1055'
SHA_V2 = '8db2951bd1c29fa4347237aef2ba607a722c37f42edad2f71ec4cc390fb4b25e'
MERGED = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, '成果報告_run09_合併版.docx')
PDF = sys.argv[2] if len(sys.argv) > 2 else None
V10 = 'v10' in os.path.basename(MERGED).lower() and 'v11' not in os.path.basename(MERGED).lower()   # v10：以 v9 升格後之現行主檔（使用者再存，sha 20d83708…）為基準；40 個圖說改寫、5 張仍在追蹤比例圖改選項 B
V9 = 'v9' in os.path.basename(MERGED).lower() or V10   # v9：30 張多格來源分段機率圖改版面（單行標題、圖內數值框）；v10 沿用 v9 之圖與紀錄綱要

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
q = lambda t: '{%s}%s' % (W, t)
sha = lambda b: hashlib.sha256(b).hexdigest()
res = []
def chk(name, cond, detail=''): res.append((bool(cond), name, detail))
SUPERSEDED = []
def chk_old(name, cond, detail=''): SUPERSEDED.append(name.split(' ')[0])   # v8：已由 v8 版取代之舊檢查（保留原碼供對照，結果不計）
def ptext(p): return ''.join(t.text or '' for t in p.iter(q('t')))
def pstyle(p):
    ps = p.find(q('pPr')); s = ps.find(q('pStyle')) if ps is not None else None
    return s.get(q('val')) if s is not None else None
KEEPRE = re.compile(r'<w:keepNext/>|<w:keepLines/>|<w:outlineLvl w:val="\d"/>')
def strip_keep(s): return KEEPRE.sub('', s)
def clean(e):
    e = etree.fromstring(etree.tostring(e))
    for x in e.iter():
        for a in list(x.attrib):
            if 'rsid' in a or a.endswith('paraId') or a.endswith('textId') or a.endswith('anchorId') or a.endswith('editId'): del x.attrib[a]
    for x in list(e.iter(q('proofErr'), q('lastRenderedPageBreak'), q('bookmarkStart'), q('bookmarkEnd'))): x.getparent().remove(x)
    return re.sub(r' xmlns:\w+="[^"]+"', '', etree.tostring(e, encoding='unicode'))
def figs_in(p):
    out = []
    for inl in p.iter('{%s}inline' % WP):
        ext = inl.find('{%s}extent' % WP); blip = next(inl.iter('{%s}blip' % A_NS)); dp = inl.find('{%s}docPr' % WP)
        out.append((int(ext.get('cx')), int(ext.get('cy')), blip.get('{%s}embed' % R_NS), dp.get('id')))
    return out

def load(path):
    z = zipfile.ZipFile(path)
    root = etree.fromstring(z.read('word/document.xml')); body = root.find(q('body'))
    rels = {r.get('Id'): r.get('Target') for r in etree.fromstring(z.read('word/_rels/document.xml.rels'))}
    return z, root, list(body), rels

SENT_E = '各類 10 個速度之跨距較大，表 1 以科學記號列其範圍；模擬時 10 個速度各自直接使用，不取平均值或對數平均。'   # v10：使用者核定，加於表 1 後第一段（區塊 6）末
def canon10(items):
    """使用者於 v9 升格後在 Word 手改，及 v10 使用者核定之文字變更，於比對時對雙方同步正規化（把 v10 新文字還原為舊文字，來源與舊基準本即為舊文字，不受影響）：
    (1) 全文「嚴重壓扁」→「重壓扁」（刪『嚴』字；共 10 個區塊、15 處）；(2) 到達時間驗證圖導引段刪『每個研究地點 1 張全年總覽圖，』；
    (3) 區塊 6 末之核定句（SENT_E）；(4) 5 張仍在追蹤比例圖：導引段首句、前綴「全部 100,000 顆虛擬質點合計，」、句末「十類曲線相近…」、圖說「（十類）；小圖為…」、區塊 34 之「十類虛擬質點各自」。
    新文字之存在與逐字內容由 V10-12、V10-13 在未正規化之原文上另行斷言。"""
    s_ = ''.join(c if len(c) == 1 else '\x00' for _, c in items)
    ed = []   # (start, end, 取代文字或 '')
    for m_ in re.finditer('嚴(?=重壓扁)', s_): ed.append((m_.start(), m_.end(), ''))
    for m_ in re.finditer('(?<=到達時間驗證圖，)每個研究地點 1 張全年總覽圖，', s_): ed.append((m_.start(), m_.end(), ''))
    for m_ in re.finditer(re.escape(SENT_E), s_): ed.append((m_.start(), m_.end(), ''))
    for m_ in re.finditer('為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。', s_): ed.append((m_.start(), m_.end(), '為全部 100,000 顆虛擬質點於回溯期間內仍在追蹤之比例。'))
    for m_ in re.finditer('全部 100,000 顆虛擬質點合計，(?=(?:貢寮|龜山島|新竹外海|南灣|連江)仍在追蹤之比例於回溯第)', s_): ed.append((m_.start(), m_.end(), ''))
    for m_ in re.finditer('(?<=降至 0%。)十類曲線相近(?:，其中不可回收塑膠略慢)?。', s_): ed.append((m_.start(), m_.end(), ''))
    for m_ in re.finditer('(?<=仍在追蹤之虛擬質點比例)（十類）；小圖為回溯前 20 天之放大', s_): ed.append((m_.start(), m_.end(), ''))
    for m_ in re.finditer('呈現十類虛擬質點各自於回溯期間內仍在追蹤之比例', s_): ed.append((m_.start(), m_.end(), '呈現全部虛擬質點於回溯期間內仍在追蹤之比例'))
    if not ed: return tuple(items)
    ed.sort(); out = []; pos = 0
    for a_, b_, r_ in ed:
        if a_ < pos: continue
        out += items[pos:a_]
        if r_: out += [(items[a_][0], ch_) for ch_ in r_]
        pos = b_
    out += items[pos:]
    return tuple(out)
def sig(e, z, rels, nokeep=False):
    """區塊簽章：段落＝(pPr, 逐字(rPr,字), 圖(寬,高,sha))；表格＝各儲存格逐字格式；其餘＝清理後 XML。與 Word 重存之 run 切分、rsid、圖檔更名無關。"""
    if e.tag == q('p'):
        ps = e.find(q('pPr')); ppr = clean(ps) if ps is not None else ''
        if nokeep: ppr = strip_keep(ppr)
        chars = []
        for r in e.iter(q('r')):
            rp = r.find(q('rPr')); rs = clean(rp) if rp is not None else ''
            for t in r.iter(q('t')):
                for ch in (t.text or ''): chars.append((rs, ch))
        figs = tuple((cx, cy, sha(z.read('word/' + rels[em]))[:12]) for cx, cy, em, _ in figs_in(e))
        return ('p', ppr, canon10(chars), figs)
    if e.tag == q('tbl'):
        seq = []
        for p in e.iter(q('p')):
            ps = p.find(q('pPr')); seq.append(('P', clean(ps) if ps is not None else ''))
            for r in p.iter(q('r')):
                rp = r.find(q('rPr')); rs = clean(rp) if rp is not None else ''
                for t in r.iter(q('t')):
                    for ch in (t.text or ''): seq.append((rs, ch))
        tp = e.find(q('tblPr')); tg = e.find(q('tblGrid'))
        return ('tbl', clean(tp) if tp is not None else '', clean(tg) if tg is not None else '', canon10(seq))
    return (e.tag.split('}')[1],)   # sectPr：Word 重存會改 footer 之 rId，不比內容

# ---------------------------------------------------------------- v11：研究地點順序（龜山島、貢寮、新竹、南灣、連江）與「新竹外海」→「新竹」
# 檔名含「v11」即啟用，且只跑本節（v10 之 151 項舊檢查依 v10 順序與文字登記，不適用於換序後之文件）。
# 信任鏈：v10 候選（sha 578c8096…，151 項全過，報告存為 check_merged_run09.report.txt.v10-20261029）
#         ──使用者於 Word 手改（V11-0 登記：五個地點小標去冒號、新竹外海小標改新竹、刪（6）綜合段與其後空段、分頁符移至（5）末）──▶ 現行主檔（sha a0f99778…）
#         ──v11 登記之變更（本節 V11-1…V11-12，逐區塊由主檔重建預期值，與 edit_v11_inplace.py 各自獨立書寫）──▶ v11 候選。
import shutil
V12 = 'v12' in os.path.basename(MERGED).lower()   # v12：v11 升格後使用者再存（sha 9808fcad…）＋「海床高程」→「該處地形水深」一處
V11 = 'v11' in os.path.basename(MERGED).lower() or V12   # v12 沿用全部 v11 檢查（含 V11-C），另加 V12-A…V12-3
MASTER11_PRE = 'a0f9977871d44f6c'; V10C_PRE = '578c8096fe1cb423'
SITES11 = ['龜山島', '貢寮', '新竹', '南灣', '連江']


def v11_main():
    def find_by_sha(pre):
        for f_ in sorted(os.listdir(BASE)):
            if f_.startswith('成果報告_run09_合併版') and '.docx' in f_:
                p_ = os.path.join(BASE, f_)
                if os.path.isfile(p_) and sha(open(p_, 'rb').read()).startswith(pre): return p_
        return None
    mp = find_by_sha(MASTER11_PRE); v10p = find_by_sha(V10C_PRE)
    chk('V11-A 比對基準檔存在：現行主檔（sha a0f99778…，使用者於 v10 後再存）與 v10 候選（sha 578c8096…）', bool(mp) and bool(v10p), str((mp, v10p)))
    if not (mp and v10p):
        fails = [r for r in res if not r[0]]
        print('\n'.join('%s %s %s' % ('OK  ' if ok else 'FAIL', n, d) for ok, n, d in res)); sys.exit(1)
    zb, rb, kb, rlb = load(mp); zc, rc, kc, rlc = load(MERGED); z10, r10, k10, rl10 = load(v10p)
    # ---- 簽章（原文，不經 canon10）
    cl2 = lambda e: clean(e).replace(' w:hint="eastAsia"', '').replace(' xml:space="preserve"', '')
    def sg(e, z, rels):
        if e.tag == q('p'):
            ps = e.find(q('pPr')); chars = []
            for r in e.iter(q('r')):
                rp = r.find(q('rPr')); rs_ = cl2(rp) if rp is not None else ''
                for t in r.iter(q('t')):
                    for ch in (t.text or ''): chars.append((rs_, ch))
            nbr = sum(1 for _ in e.iter(q('br')))
            figs = tuple((cx, cy, sha(z.read('word/' + rels[em]))[:12]) for cx, cy, em, _ in figs_in(e))
            return ('p', cl2(ps) if ps is not None else '', tuple(chars), nbr, figs)
        if e.tag == q('tbl'):
            return ('tbl', cl2(e))
        return (e.tag.split('}')[1],)
    # ================= V11-0 使用者於 v10 後之手改（v10 候選 → 現行主檔）
    s10 = [sg(e, z10, rl10) for e in k10]; sb = [sg(e, zb, rlb) for e in kb]
    txt = lambda e: ptext(e)
    heads10 = [i for i, e in enumerate(k10) if e.tag == q('p') and txt(e) in ('貢寮：', '龜山島：', '新竹外海：', '南灣：', '連江：')]
    ok0 = len(k10) == 359 and len(kb) == 357 and heads10 == [36, 71, 105, 139, 173]
    ex_h = {'貢寮：': '貢寮', '龜山島：': '龜山島', '新竹外海：': '新竹', '南灣：': '南灣', '連江：': '連江'}
    ok0 = ok0 and all(txt(kb[i]) == ex_h[txt(k10[i])] for i in heads10)
    others = [i for i in range(217) if i not in heads10 and i != 216]
    ok0 = ok0 and all(s10[i] == sb[i] for i in others)
    for i in heads10:   # 小標：去冒號（新竹外海→新竹）；段落格式與字元格式（rPr 集合）不變
        ok0 = ok0 and s10[i][1] == sb[i][1] and {r for r, _ in s10[i][2]} == {r for r, _ in sb[i][2]}
    # 區塊 216（（5）連江）：文字與格式相同，現行主檔末多一個分頁符
    ok0 = ok0 and s10[216][:3] == sb[216][:3] and sb[216][3] == s10[216][3] + 1 and s10[216][4] == sb[216][4]
    # v10 之（6）綜合段（217）與其後之空段（218，含分頁符）已刪；其後區塊（參考資料起）逐一相同
    ok0 = ok0 and txt(k10[217]).startswith('（6）綜合五個研究地點：') and txt(k10[218]) == '' and sum(1 for _ in k10[218].iter(q('br'))) == 1
    ok0 = ok0 and all(s10[i + 2] == sb[i] for i in range(217, 357))
    chk('V11-0 使用者於 v10 後之手改（登記）：五個地點小標去冒號、「新竹外海」小標改「新竹」、刪（6）綜合段與其後空段、分頁符移至（5）末；其餘 %d 個區塊與 v10 候選逐字、逐格式、逐圖片相同' % (len(others) + 140), ok0)
    # ================= 重建預期值
    def f11(n):
        if n == 3: return 4
        if n == 4: return 3
        if 8 <= n <= 18: return n + 11
        if 19 <= n <= 29: return n - 11
        return n
    def renum(t): return re.sub(r'圖( ?)(\d+)', lambda m: '圖%s%d' % (m.group(1), f11(int(m.group(2)))), t)
    def rep(t, old, new):
        assert t.count(old) == 1, (old, t.count(old), t[:40]); return t.replace(old, new)
    T = {i: txt(e) for i, e in enumerate(kb) if e.tag == q('p')}
    # 新龜山島（來源：舊 71–104＋舊 46）、新貢寮（來源：舊 36–70 去 46）
    PERM = list(range(357))
    PERM[21:25] = [23, 24, 21, 22]
    B_ = list(range(71, 81)) + [46] + list(range(81, 105)); A_ = [i for i in range(36, 71) if i != 46]
    PERM[36:105] = B_ + A_
    PERM[212], PERM[213] = 213, 212
    assert sorted(PERM) == list(range(357))
    # 舊 43（貢寮）：定義段（含「圖 10 之底色」…「較廣之海域。」）與貢寮數據段
    t43 = T[43]; k43 = t43.index('貢寮之 50%、75%、90% 高密度區間面積分別為')
    def_a = t43[:k43]; assert def_a.startswith('圖 10 之底色為停留時數密度：') and def_a.endswith('表示分布於較廣之海域。')
    t59 = T[59]; i0 = t59.index('沉底時刻為廢棄物沉降至海床之時刻'); i1 = t59.index('各季節之穿越比例及穿越最集中之邊界段為：')
    defn = t59[i0:i1]; assert defn.endswith('秋季為 9–11 月。') and '由於廢棄物實際何時沉底未知' in defn
    LONG = '依沉底時刻之潮位資料劃分為大潮與小潮之'; SHORT = '依大小潮劃分之'
    def expect(src):
        t = T[src]
        if src not in (46, 20): t = renum(t)
        if src == 17: t = rep(t, '(a) 貢寮、(b) 龜山島、(c) 新竹外海', '(a) 龜山島、(b) 貢寮、(c) 新竹')
        elif src == 20: t = rep(t, '貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5', '龜山島見圖 3，貢寮見圖 4，新竹見圖 5')
        elif src == 37: t = rep(t, '為隨機抽取之 24 條代表虛擬質點回溯軌跡，線色依起始深度由淺至深。', '為貢寮隨機抽取之 24 條代表虛擬質點回溯軌跡。')
        elif src == 72: t = rep(t, '回溯軌跡。', '回溯軌跡，線色依起始深度由淺至深。')
        elif src == 40: t = rep(t, '貢寮之虛擬質點穿越流場範圍邊界者占 73.40%，稱為穿越比例，抵達海岸者占 26.60%，稱為海岸來源比例。', '貢寮之穿越比例為 73.40%，海岸來源比例為 26.60%。')
        elif src == 75: t = rep(t, '龜山島之穿越比例為 78.35%，海岸來源比例為 21.65%。', '龜山島之虛擬質點穿越流場範圍邊界者占 78.35%，稱為穿越比例，抵達海岸者占 21.65%，稱為海岸來源比例。')
        elif src == 43: t = '圖 21 中，' + t[t.index('貢寮之 50%、75%、90%'):]
        elif src == 78:
            assert t.startswith('圖 10 中，龜山島之 50%')
            t = '圖 10 之底色' + def_a[len('圖 10 之底色'):] + t[len('圖 10 中，'):]
        elif src == 46: t = rep(t, '為貢寮之來源分段機率圖', '為龜山島之來源分段機率圖')
        elif src == 59: t = t.replace(defn, '', 1) if t.count(defn) == 1 else None
        elif src == 93: t = rep(t, '四季邊界穿越來源分段機率圖。', '四季邊界穿越來源分段機率圖。' + defn)
        elif src in (65, 68): t = rep(t, '為' + LONG, '為' + SHORT)
        elif src in (99, 102): t = rep(t, '為' + SHORT, '為' + LONG)
        elif src == 208: t = rep(t, '貢寮、新竹外海、南灣與連江四季兩兩之間差異明顯；龜山島僅冬、春兩季相近，其餘季節兩兩之間差異明顯。', '龜山島僅冬、春兩季相近，其餘季節兩兩之間差異明顯；貢寮、新竹、南灣與連江四季兩兩之間差異明顯。')
        elif src == 212: t = rep(t, '（1）貢寮：', '（2）貢寮：')
        elif src == 213: t = rep(t, '（2）龜山島：', '（1）龜山島：')
        if V12 and src == 0: t = t + '分析成果'                                              # v12 登記：使用者於 v11 後在 Word 手改（標題加「分析成果」）
        if V12 and src == 10: t = rep(t, '潮位與海床高程換算', '潮位與該處地形水深換算')      # v12 登記：使用者核定之一處用語
        t = t.replace('新竹外海', '新竹')
        t = t.replace('龜山島沿岸', '龜山島周圍海岸')      # 使用者核定（v11 複核）：龜山島作為海岸來源時寫「龜山島周圍海岸」
        return re.sub(r'龜山島占(?= \d)', '龜山島周圍海岸占', t)
    # ================= V11-1 封裝
    chk('V11-1 zip 完整、全部 XML 可解析', zc.testzip() is None and all(etree.fromstring(zc.read(n)) is not None for n in zc.namelist() if n.endswith('.xml') or n.endswith('.rels')))
    zbase = zb
    if V12:   # v12：zip 成員改與 v11 升格後之使用者再存主檔（sha 9808fcad…）比對（Word 重存後媒體檔重新編號、settings／docProps 更新，見 V12-0）
        _p12 = find_by_sha('9808fcad1dfc285f'); zbase = zipfile.ZipFile(_p12) if _p12 else zb
    same = [n for n in zbase.namelist() if n not in (('word/document.xml',) if V12 else ('word/document.xml', 'word/media/image1.png', 'word/media/image2.png'))]
    diffm = [n for n in same if n not in zc.namelist() or zc.read(n) != zbase.read(n)]
    extra = [n for n in zc.namelist() if n not in zbase.namelist()]
    chk('V11-1b 除 word/document.xml 與圖 1、圖 2 兩個媒體檔外，其餘 %d 個 zip 成員（styles、numbering、theme、rels、其餘 60 張圖等）與主檔逐位元組相同、無新增成員' % len(same), not diffm and not extra, str((diffm[:3], extra[:3])))
    # ================= V11-2 逐區塊重建
    chk('V11-2a 區塊數 357（含 sectPr），與主檔相同', len(kc) == 357, str(len(kc)))
    bad_t = []; bad_f = []; bad_i = []
    sc = [sg(e, zc, rlc) for e in kc]
    for j in range(357):
        src = PERM[j]; e = kc[j]; es = kb[src]
        if es.tag == q('p'):
            want = expect(src)
            if want is None or txt(e) != want: bad_t.append((j, src, (txt(e) or '')[:30]))
            # 格式：pPr（導引段不含 keepNext，keepNext 另查）、rPr 集合、分頁符數
            is_guide = j + 1 < 357 and kc[j + 1].tag == q('p') and bool(figs_in(kc[j + 1]))
            a_ = strip_keep(sc[j][1]) if is_guide else sc[j][1]; b_ = strip_keep(sb[src][1]) if is_guide else sb[src][1]
            rset_ok = {r for r, _ in sc[j][2]} == {r for r, _ in sb[src][2]}
            if V12 and src == 0: rset_ok = {r for r, _ in sc[j][2]} == {r for r, _ in sb[src][2]} | {'<w:rPr><w:rFonts/><w:szCs w:val="28"/></w:rPr>'}   # 標題新加「分析成果」之 run 格式
            if a_ != b_ or not rset_ok or sc[j][3] != sb[src][3]: bad_f.append((j, src))
            if figs_in(e) or figs_in(es):
                a_f = sc[j][4]; b_f = sb[src][4]
                if len(a_f) != len(b_f) or any(x[:2] != y[:2] for x, y in zip(a_f, b_f)): bad_i.append((j, src, 'size'))
        elif es.tag == q('tbl'):
            pass
    chk('V11-2b 每個區塊之文字 = 主檔對應來源區塊（換序後）＋登記之變更（圖號互換、第一個地點說明改隨龜山島、新竹外海→新竹、結論與綜合說明重排）；共 %d 個段落逐字相等' % sum(1 for e in kb if e.tag == q('p')), not bad_t, str(bad_t[:5]))
    chk('V11-2c 每個區塊之段落格式（pPr〔導引段不含 keepNext〕、字元格式、分頁符）與來源區塊相同；圖之顯示尺寸（寬、高）與來源區塊相同', not bad_f and not bad_i, str((bad_f[:5], bad_i[:5])))
    # ================= V11-3 表 2 列對調
    tb, tc_ = kb[13], kc[13]
    rows_b = tb.findall(q('tr')); rows_c = tc_.findall(q('tr'))
    exp_rows = [rows_b[0], rows_b[2], rows_b[1]] + rows_b[3:]
    names_c = [''.join(r.itertext()) for r in rows_c]
    chk('V11-3 表 2：列序為 龜山島、貢寮、新竹、南灣、連江；各列內容與格式 = 主檔對應列（貢寮、龜山島對調；新竹外海→新竹）；表格屬性與欄寬不變',
        len(rows_c) == 6 and all(n_.startswith(s_) for n_, s_ in zip(names_c[1:], SITES11))
        and [''.join(r.itertext()).replace('新竹外海', '新竹') for r in exp_rows] == names_c
        and [cl2(r).replace('新竹外海', '新竹') for r in exp_rows] == [cl2(r) for r in rows_c]
        and cl2(tb.find(q('tblPr'))) == cl2(tc_.find(q('tblPr'))) and cl2(tb.find(q('tblGrid'))) == cl2(tc_.find(q('tblGrid'))), str(names_c[:3]))
    # ================= V11-4 順序與圖號
    heads = [i for i, e in enumerate(kc) if e.tag == q('p') and pstyle(e) == 'afe' and txt(e) in SITES11]
    head_ok = [txt(kc[i]) for i in heads] == SITES11 and heads == [36, 71, 105, 139, 173]
    # 圖號：docPr 名稱與圖說號依文件順序為 1…62
    seq = []
    for i, e in enumerate(kc):
        if e.tag == q('p'):
            for inl in e.iter('{%s}inline' % WP):
                nm = inl.find('{%s}docPr' % WP).get('name')
                seq.append((i, int(re.match(r'^fig(\d+)\.', nm).group(1)) if re.match(r'^fig(\d+)\.', nm) else None, txt(kc[i + 1])))
    nums_ok = len(seq) == 62
    cap_ok = True; names_ok = []
    for k, (i, n, c) in enumerate(seq, 1):
        m = re.match(r'^圖(\d+) ', c)
        if not m or int(m.group(1)) != k: cap_ok = False; names_ok.append((k, c[:12]))
        if k >= 8 and n != k: cap_ok = False; names_ok.append((k, 'docPr', n))
    chk('V11-4 地點小標順序為 龜山島、貢寮、新竹、南灣、連江（小標段落位置 36、71、105、139、173）；全文 62 張圖依序為圖 1…62，圖說號與圖 8 起之 docPr 名稱 figN 皆與順序一致', head_ok and nums_ok and cap_ok, str((head_ok, len(seq), names_ok[:4])))
    # 圖 2 圖說字母 (a)…(e) 與表 2 列序、小標順序一致；圖 3–7 圖說依序為 龜山島、貢寮、新竹、南灣、連江
    cap17 = txt(kc[17]); cap37 = [txt(kc[i]) for i in (22, 24, 26, 28, 30)]
    chk('V11-4b 圖 2 圖說為 (a) 龜山島、(b) 貢寮、(c) 新竹、(d) 南灣、(e) 連江；圖 3–7 圖說與導引句依序為龜山島、貢寮、新竹、南灣、連江',
        cap17.endswith('(a) 龜山島、(b) 貢寮、(c) 新竹、(d) 南灣、(e) 連江') and cap37 == ['圖%d %s到達時間驗證圖' % (k, s_) for k, s_ in zip(range(3, 8), SITES11)]
        and '龜山島見圖 3，貢寮見圖 4，新竹見圖 5，南灣見圖 6，連江見圖 7' in txt(kc[20]), str((cap17[-40:], cap37)))
    # 各地點區塊內之圖號引用只指向本區塊之圖（龜山島 8–18、貢寮 19–29、新竹 30–40、南灣 41–51、連江 52–62）
    rng = {36: (8, 18), 71: (19, 29), 105: (30, 40), 139: (41, 51), 173: (52, 62)}
    xref_bad = []
    bounds = [36, 71, 105, 139, 173, 207]
    for a_, b_ in zip(bounds, bounds[1:]):
        lo, hi = rng[a_]
        for i in range(a_, b_):
            if kc[i].tag != q('p'): continue
            for m in re.finditer(r'圖 ?(\d+)', txt(kc[i])):
                if not (lo <= int(m.group(1)) <= hi): xref_bad.append((i, m.group(0)))
    allrefs = sorted({int(m.group(1)) for e in kc if e.tag == q('p') for m in re.finditer(r'圖 ?(\d+)', txt(e))})
    chk('V11-4c 圖號引用：各地點區塊內之「圖 N」皆落在該地點之圖號範圍內；全文引用之圖號皆為 1–62 之既有圖', not xref_bad and allrefs and max(allrefs) <= 62 and min(allrefs) >= 1, str(xref_bad[:5]))
    # 替代文字（docPr descr／name）之圖號與圖說一致
    desc_bad = []
    for i, n, c in seq:
        e = kc[i]
        for inl in e.iter('{%s}inline' % WP):
            d = inl.find('{%s}docPr' % WP).get('descr') or ''
            m = re.match(r'^圖(\d+) ', d)
            if m and int(m.group(1)) != int(re.match(r'^圖(\d+)', c).group(1)): desc_bad.append((i, d[:20]))
    chk('V11-4d 圖之替代文字（descr）凡以「圖N」起首者，N 與圖說一致', not desc_bad, str(desc_bad[:4]))
    # ================= V11-5 圖
    img_bad = []
    for j in range(357):
        src = PERM[j]
        if kc[j].tag == q('p') and figs_in(kc[j]):
            a_, b_ = sc[j][4], sb[src][4]
            if j in (14, 16): continue
            if [x[2] for x in a_] != [x[2] for x in b_]: img_bad.append((j, src))
    chk('V11-5a 圖 3–62 之圖檔（含各圖搬移後）與主檔對應來源區塊逐位元組相同（sha）；貢寮與龜山島兩區塊之圖 11 張對調、圖 3 與圖 4 對調', not img_bad, str(img_bad[:5]))
    FD = os.path.join(os.path.dirname(HERE), 'preview', 'site_order', 'fig12_new')
    f1 = open(os.path.join(FD, '343_run09v5_fig1_circle.png'), 'rb').read(); f2 = open(os.path.join(FD, '343_run09v5_fig2_circle.png'), 'rb').read()
    m1 = zc.read('word/media/image1.png'); m2 = zc.read('word/media/image2.png'); o1 = zb.read('word/media/image1.png'); o2 = zb.read('word/media/image2.png')
    import io
    from PIL import Image
    import numpy as np
    def px(b): return np.array(Image.open(io.BytesIO(b)).convert('RGB'))
    a1, c1 = px(o1), px(m1); a2, c2 = px(o2), px(m2)
    d1 = (a1 != c1).any(2) if a1.shape == c1.shape else np.ones((1, 1), bool); d2 = (a2 != c2).any(2) if a2.shape == c2.shape else np.ones((1, 1), bool)
    ys1, xs1 = np.nonzero(d1); ys2, xs2 = np.nonzero(d2)
    # 圖 1：只有 (a)、(b) 兩個字母標籤處不同（標籤位於星號右側，y 約 939–1178、x 約 1775–1888）；圖 2：只有第一列（y < 1700）之兩個小圖不同
    ok_img = (m1 == f1 and m2 == f2 and m1 != o1 and m2 != o2 and a1.shape == c1.shape and a2.shape == c2.shape
              and ys1.min() >= 930 and ys1.max() <= 1180 and xs1.min() >= 1770 and xs1.max() <= 1890 and ys2.max() < 1700)
    chk('V11-5b 圖 1、圖 2 為依新順序重畫之圖檔（與 preview/site_order/fig12_new 之圖逐位元組相同；像素尺寸同舊圖）；圖 1 與舊圖之差異只在 (a)、(b) 兩個字母標籤（%d 像素），圖 2 只在第一列兩個小圖（龜山島與貢寮對調，%d 像素）' % (d1.sum(), d2.sum()), ok_img, str((d1.sum(), ys1.min(), ys1.max(), xs1.min(), xs1.max(), ys2.max())))
    # ================= V11-6 名稱
    allxml = {n: zc.read(n).decode('utf-8', 'ignore') for n in zc.namelist() if n.endswith('.xml') or n.endswith('.rels')}
    OLDV = ['新竹外海', '龜山島沿岸', '龜山島西', '南灣海域', '連江外海', '貢寮海域', '後灣', '龜山島海域', '新竹海域']
    left = [(n, v) for n, x in allxml.items() for v in OLDV if v in x]
    chk('V11-6a 全部 word/*.xml、docProps 等 XML 部件（含 document.xml 之文字與替代文字、docPr 名稱）不再出現「新竹外海」及其他舊名稱變體（龜山島西側、南灣海域、連江外海、貢寮海域、後灣、龜山島海域、新竹海域）', not left, str(left[:5]))
    allp = lambda kk: '\n'.join(txt(e) for e in kk if e.tag == q('p')) + '\n' + '\n'.join(txt(p) for t_ in kk if t_.tag == q('tbl') for p in t_.iter(q('p')))
    full = allp(kc); fullb = allp(kb)
    chk('V11-6b 「新竹」於全文（含表格）共 %d 處，與主檔「新竹」（含「新竹外海」之 %d 處與小標）之處數相同；「新竹外海」0 處' % (full.count('新竹'), fullb.count('新竹外海')),
        full.count('新竹外海') == 0 and full.count('新竹') == fullb.count('新竹'), str((full.count('新竹'), fullb.count('新竹'))))
    # ================= V11-7 第一個地點才有之說明（隨龜山島）
    P = {i: txt(e) for i, e in enumerate(kc) if e.tag == q('p')}
    first = ['線色依起始深度由淺至深', '稱為穿越比例', '稱為海岸來源比例', '之底色為停留時數密度', '來源分段之取法為', '由於廢棄物實際何時沉底未知', '依沉底時刻之潮位資料劃分為大潮與小潮之']
    loc = {k_: [i for i, t in P.items() if k_ in t] for k_ in first}
    ok7 = (loc['線色依起始深度由淺至深'] == [37] and loc['稱為穿越比例'] == [40] and loc['稱為海岸來源比例'] == [40] and loc['之底色為停留時數密度'] == [43]
           and loc['來源分段之取法為'] == [46] and loc['由於廢棄物實際何時沉底未知'] == [59] and loc['依沉底時刻之潮位資料劃分為大潮與小潮之'] == [65, 68]
           and P[37].startswith('圖 8 為龜山島') and P[40].startswith('圖 9 ') and P[43].startswith('圖 10 之底色') and P[46].startswith('圖 11 至圖 18 為龜山島之來源分段機率圖')
           and P[59].startswith('圖 15 為依沉底時刻劃分之龜山島四季邊界穿越來源分段機率圖。沉底時刻為廢棄物沉降至海床之時刻') and P[65].startswith('圖 17 為依沉底時刻之潮位資料劃分為大潮與小潮之龜山島') and P[68].startswith('圖 18 為依沉底時刻之潮位資料劃分為大潮與小潮之龜山島'))
    chk('V11-7a 第一個地點才有之說明改隨龜山島（各恰出現一次，且位於龜山島區塊）：線色說明（圖 8）、穿越比例與海岸來源比例之定義（圖 9）、停留時數密度與高密度區間之定義（圖 10）、來源分段之取法（圖 11 至圖 18 導引段）、沉底時刻與季節之定義（圖 15；「由於廢棄物實際何時沉底未知」逐字保留）、大小潮定義句（圖 17、圖 18）', ok7, str(loc))
    ok7b = (P[72].startswith('圖 19 為貢寮隨機抽取之 24 條代表虛擬質點回溯軌跡。軌跡') and '貢寮之穿越比例為 73.40%，海岸來源比例為 26.60%。' in P[75] and P[78].startswith('圖 21 中，貢寮之 50%、75%、90% 高密度區間面積分別為')
            and P[93].startswith('圖 26 為依沉底時刻劃分之貢寮四季邊界穿越來源分段機率圖。各季節之穿越比例') and P[99].startswith('圖 28 為依大小潮劃分之貢寮邊界穿越來源分段機率圖。') and P[102].startswith('圖 29 為依大小潮劃分之貢寮海岸來源分段機率圖。')
            and not any(P[i].startswith('圖 2') and '來源分段之取法' in P[i] for i in P))
    chk('V11-7b 貢寮改為其餘地點之簡短寫法（與新竹、南灣、連江之句式相同）；貢寮區塊不再有「圖 N 至圖 M 為…來源分段機率圖」取法段', ok7b)
    # 順序一致：綜合說明、整體結論
    chk('V11-7c 綜合說明「季節之影響明顯」句依新順序（龜山島先）；整體結論（1）龜山島、（2）貢寮、（3）新竹、（4）南灣、（5）連江；（6）綜合段已由使用者刪除（V11-0）',
        '季節之影響明顯：龜山島僅冬、春兩季相近，其餘季節兩兩之間差異明顯；貢寮、新竹、南灣與連江四季兩兩之間差異明顯。' in P[208]
        and [P[i][:3] + P[i][3:].split('：')[0] for i in range(212, 217)] == ['（1）龜山島', '（2）貢寮', '（3）新竹', '（4）南灣', '（5）連江'] and all(P[i].split('：')[0] == P[i][:3] + P[i][3:].split('：')[0] and '：' in P[i] for i in range(212, 217)) and all(pstyle(kc[i]) == 'afe' for i in range(212, 217)), str([P[i][:8] for i in range(212, 217)]))
    # ================= V11-8 keepNext 估算式（v8 起）
    LIMc = (16838 - 2880) / 20.0 - 17.0; BWc = (11906 - 2880 - 991) / 20.0; CWc = (11906 - 2880) / 20.0
    def nlines(text, wpt):
        ln, w = 1, 0.0
        for ch in text:
            cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
            if w + cw > wpt: ln += 1; w = cw
            else: w += cw
        return ln
    kbad = []
    for i, n, c in seq:
        if n is None or n < 8: continue
        g = kc[i - 1]; cy = figs_in(kc[i])[0][1]
        tot = nlines(txt(g), BWc) * 22 + 18 + 12 + cy / 12700.0 + (2 + nlines(c, CWc) * 27 + 8)
        has = g.find(q('pPr')).find(q('keepNext')) is not None
        if has != (tot <= LIMc): kbad.append((n, has, round(tot, 1)))
    chk('V11-8 圖 8–62 導引段 keepNext 與估算式（導引段、圖、圖說合計 ≤ %.1f pt 才與下段同頁）一致（共 55 張）' % LIMc, not kbad, str(kbad[:5]))
    # ================= V11-9 其餘：標題大綱層級、圖說格式（沿主檔，已含於 V11-2c）；成果段數值未動
    nums = lambda t: sorted(re.findall(r'\d+(?:\.\d+)?%', t))
    sA = sorted(x for i in range(36, 105) for x in nums(T.get(i, '')))
    sC = sorted(x for i in range(36, 105) for x in nums(P.get(i, '')))
    chk('V11-9 （二）逐地點成果之全部百分比數值（%d 個）與主檔相同之多重集合（只搬移、未改任何數值）' % len(sA), sA == sC, str(set(sA) ^ set(sC)))
    # ================= V11-C 內容層級檢查（自 v10 檢查移植；以「研究地點」與「圖類型」為鍵，不依區塊位置或圖號）
    # 圖之辨識：圖檔 sha256 → (地點, 類型)。來源：3-4-3 原圖（到達時間驗證圖）、執行成果原圖（軌跡、足跡）、v10 建置紀錄（仍在追蹤比例曲線）、
    # segprob_v3 manifest（全年邊界、全年海岸）、segprob_fullpage/final（其餘 6 類 × 5 地點）。圖號只作為辨識結果之屬性使用。
    import csv, struct
    LOG = json.load(open(os.path.join(HERE, 'v10_build_log.json'), encoding='utf-8')); V10L = LOG['v10']
    SEG = os.path.join(BASE, '_verify', 'preview', 'segprob_v3'); MAN = json.load(open(os.path.join(SEG, 'manifest.json'), encoding='utf-8'))
    FINAL9 = os.path.join(BASE, '_verify', 'preview', 'segprob_fullpage', 'final')
    SITEK = ['gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang']
    ZH = dict(zip(SITEK, ['貢寮', '龜山島', '新竹', '南灣', '連江'])); SKOF = {v_: k_ for k_, v_ in ZH.items()}
    FIGK = ['boundary_all', 'coast_all', 'boundary_class', 'coast_class', 'boundary_season', 'coast_season', 'boundary_tide', 'coast_tide']
    KEYS8 = ['all_b', 'all_c', 'cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c']
    rd = lambda f_: list(csv.DictReader(open(os.path.join(SEG, 'tables', f_), encoding='utf-8-sig')))
    AS = rd('annotated_stretches.csv'); GTR = {(r_['site'], r_['group']): r_ for r_ in rd('group_totals.csv')}
    VER = list(csv.DictReader(open(os.path.join(BASE, '_verify', 'preview', 'segprob_v2', 'tables', 'diff_verdicts_final.csv'), encoding='utf-8-sig')))
    FT = '\n'.join(''.join(t_.text or '' for t_ in e.iter(q('t'))) for e in kc)
    FTB = '\n'.join(''.join(t_.text or '' for t_ in e.iter(q('t'))) for e in kb)
    isp = lambda e: e.tag == q('p')
    def idx_text(kk, pred, start=0):
        for n_ in range(start, len(kk)):
            if isp(kk[n_]) and pred(txt(kk[n_])): return n_
        return None
    ia = idx_text(kc, lambda t: t.startswith('4、溯源停止邊界')); ri0 = idx_text(kc, lambda t: t == '參考資料')
    ins = kc[ia + 1:ri0]
    got = [txt(p_) for p_ in ins if isp(p_) and not figs_in(p_)]
    alltext = [txt(p_) for p_ in kc if isp(p_)]
    n343 = ''.join(txt(e) for e in kc[:ia + 1] if isp(e))
    i_sh = idx_text(kc, lambda t: t == '逐地點成果：'); i_zh = idx_text(kc, lambda t: t == '綜合說明與結論：')
    hd = {nm: idx_text(kc, lambda t, nm=nm: t == nm, i_sh) for nm in SITES11}
    order_ = sorted(hd.values()); endb = {nm: (order_[order_.index(hd[nm]) + 1] if order_.index(hd[nm]) + 1 < 5 else i_zh) for nm in SITES11}
    blocks = {nm: [txt(kc[j_]) for j_ in range(hd[nm] + 1, endb[nm]) if isp(kc[j_])] for nm in SITES11}
    # ---- 辨識
    chk('V11-C0 來源檔（3-4-3_run09.docx、執行成果_run09_含斯托克斯漂流.docx）未被修改（sha 同登記）', sha(open(SRC_A, 'rb').read()) == SHA_A and sha(open(SRC_B, 'rb').read()) == SHA_B)
    zA, _rA, kA, rlA = load(SRC_A); zB, _rB, kB0, rlB = load(SRC_B)
    figsA = [(f_[:2], sha(zA.read('word/' + rlA[f_[2]]))) for p_ in kA if isp(p_) for f_ in figs_in(p_)]
    figsB = [(f_[:2], sha(zB.read('word/' + rlB[f_[2]]))) for p_ in kB0 if isp(p_) for f_ in figs_in(p_)]
    IDM = {}; collide = []
    def reg_(h_, key_):
        if h_ in IDM and IDM[h_] != key_: collide.append((key_, IDM[h_]))
        IDM[h_] = key_
    SZ = {}
    for i_, sk_ in enumerate(SITEK):
        reg_(figsA[2 + i_][1], (sk_, 'arrive')); SZ[(sk_, 'arrive')] = figsA[2 + i_][0]
        reg_(figsB[7 * i_][1], (sk_, 'traj')); SZ[(sk_, 'traj')] = figsB[7 * i_][0]
        reg_(figsB[7 * i_ + 2][1], (sk_, 'foot')); SZ[(sk_, 'foot')] = figsB[7 * i_ + 2][0]
        for k_, fk_ in zip(KEYS8, FIGK):
            if k_ in ('all_b', 'all_c'): reg_(MAN['figures'][sk_][fk_]['sha256'], (sk_, k_))
            else: reg_(sha(open(os.path.join(FINAL9, sk_, fk_ + '.png'), 'rb').read()), (sk_, k_))
    for c_ in V10L['curves']: reg_(c_['png_sha256'], (c_['site'], 'curve'))
    CURV = {c_['site']: c_ for c_ in V10L['curves']}
    FIGAT = {}; unident = []; cfigs = []
    for i_, e in enumerate(kc):
        if isp(e):
            for f_ in figs_in(e):
                h_ = sha(zc.read('word/' + rlc[f_[2]])); k_ = IDM.get(h_)
                cfigs.append((i_, f_, h_, k_))
                if k_ is None: unident.append((i_, h_[:8]))
                else: FIGAT.setdefault(k_, []).append(i_)
    ROLES = ['arrive', 'traj', 'curve', 'foot'] + KEYS8
    capnum = lambda i_: int(re.match(r'^圖(\d+) ', txt(kc[i_ + 1])).group(1))
    FN = {k_: capnum(v_[0]) for k_, v_ in FIGAT.items()}      # (地點, 類型) → 圖說號
    GUIDE = {k_: txt(kc[v_[0] - 1]) for k_, v_ in FIGAT.items()}
    CAPT = {k_: txt(kc[v_[0] + 1]) for k_, v_ in FIGAT.items()}
    okid = (not collide and len(unident) == 2 and len(cfigs) == 62 and all(len(FIGAT.get((sk_, r_), [])) == 1 for sk_ in SITEK for r_ in ROLES))
    chk('V11-C1 62 張圖之辨識：圖檔 sha 對應 (地點, 類型)——5 地點 × 12 類（到達時間驗證、軌跡、仍在追蹤比例、足跡、8 類來源分段機率）= 60 張，各恰 1 張、無 sha 衝突；其餘 2 張為圖 1、圖 2（重畫版）', okid, str((collide[:2], unident[:3], len(cfigs), [(k_, len(v_)) for k_, v_ in FIGAT.items() if len(v_) != 1][:3])))
    # 辨識結果與所在地點區塊一致、類型次序固定
    posbad = []
    for sk_ in SITEK:
        nm = ZH[sk_]
        idxs = [FIGAT[(sk_, r_)][0] for r_ in ROLES[1:]]
        if not all(hd[nm] < i_ < endb[nm] for i_ in idxs) or idxs != sorted(idxs): posbad.append((nm, 'block/order'))
        for r_ in ROLES[1:]:
            if nm not in CAPT[(sk_, r_)] or nm not in GUIDE[(sk_, r_)]: posbad.append((nm, r_, '地點名'))
        if nm not in CAPT[(sk_, 'arrive')]: posbad.append((nm, 'arrive', '地點名'))
    arr = [FIGAT[(sk_, 'arrive')][0] for sk_ in [SKOF[n_] for n_ in SITES11]]
    chk('V11-C2 每個地點之 11 張圖（軌跡、仍在追蹤比例、足跡、8 類來源分段機率）皆位於該地點小標之區塊內，次序為 軌跡→仍在追蹤比例→足跡→全年邊界→全年海岸→十類邊界→十類海岸→季節邊界→季節海岸→大小潮邊界→大小潮海岸；導引段與圖說皆含該地點名；到達時間驗證圖依 龜山島、貢寮、新竹、南灣、連江 序', not posbad and arr == sorted(arr), str(posbad[:4]))
    # ---- 數值：各地點穿越／海岸來源比例（已公布值）
    PUB = {'貢寮': ('73.40', '26.60'), '龜山島': ('78.35', '21.65'), '新竹': ('97.77', '2.23'), '南灣': ('68.91', '31.09'), '連江': ('56.06', '43.94')}
    dres = {}; lastday = {}; sub = {}; area = {}; depth = {}
    for nm, bl in blocks.items():
        jt = '\n'.join(bl)
        m_ = re.search(r'穿越比例為 ([\d.]+)%，海岸來源比例為 ([\d.]+)%', jt) or re.search(r'穿越流場範圍邊界者占 ([\d.]+)%，稱為穿越比例，抵達海岸者占 ([\d.]+)%', jt)
        dres[nm] = (m_.group(1), m_.group(2)) if m_ else None
        m_ = re.search(r'至第 ([\d.]+) 天降至 0%', jt); lastday[nm] = m_.group(1) if m_ else None
        m_ = re.search(r'海岸來源（([\d.]+)%）中，([^。]+)。', jt)
        if m_: sub[nm] = (m_.group(1), re.findall(r'占 ([\d.]+)%', m_.group(2)))
        m_ = re.search(r'高密度區間面積分別為 ([\d,]+)、', jt); area[nm] = m_.group(1) if m_ else None
        m_ = re.search(r'停留時數之 ([\d.]+)% 位於水深淺於 200 m', jt); depth[nm] = m_.group(1) if m_ else None
    chk('V11-C N1 5 地點穿越／海岸來源比例（各地點段，不論用「稱為穿越比例」之定義寫法或簡短寫法）= 已公布值', dres == PUB, str(dres))
    for nm in SITES11:
        a_, b_ = Decimal(dres[nm][0]), Decimal(dres[nm][1])
        chk('V11-C N2 %s：穿越＋海岸來源%s = 100.00' % (nm, '（另 1 顆觸底 0.001%）' if nm == '南灣' else ''), abs(a_ + b_ - 100) <= Decimal('0.0100') and (a_ + b_ == 100 or nm == '南灣'), str(a_ + b_))
    chk('V11-C N3 「33.4 天」：連江最末降至 0% 日 = 全部地點最大值 = 綜合說明之 33.4 天', lastday['連江'] == '33.4' and max(Decimal(v_) for v_ in lastday.values()) == Decimal('33.4') and any('皆於 33.4 天內停止追蹤' in t_ for t_ in got), str(lastday))
    cr = [Decimal(dres[n_][0]) for n_ in SITES11]; co = [Decimal(dres[n_][1]) for n_ in SITES11]
    rngt = '穿越比例為 %s%%–%s%%' % (min(cr), max(cr)); rngc = '海岸來源比例為 %s%%–%s%%' % (min(co), max(co))
    chk('V11-C N4 綜合說明之比例範圍 = 五地點最小最大', any(rngt in t_ and rngc in t_ for t_ in got), rngt + ' ' + rngc)
    concl = {nm: next(t_ for t_ in got if re.match(r'^（\d）%s：' % nm, t_) and len(t_) > 40) for nm in SITES11}
    chk('V11-C N5 整體結論條列為（1）龜山島、（2）貢寮、（3）新竹、（4）南灣、（5）連江（使用者已刪第 6 條綜合段，故原 N5、N7 之「第 6 條範圍」不適用；範圍數字改由 N4、N9、N10 與各地點條列把關）', [re.match(r'^（(\d)）', concl[n_]).group(1) for n_ in SITES11] == ['1', '2', '3', '4', '5'] and not any(t_.startswith('（6）') for t_ in got))
    def _sub_ok(nm, tot, parts):
        if Decimal(tot) != Decimal(dres[nm][1]): return False
        if nm == '龜山島':   # 不寫其他島嶼之數值（「其餘為其他島嶼」）；餘數須落在主檔原值 0.13 之進位誤差內（0.10–0.15）
            return Decimal('0.10') <= Decimal(tot) - sum(Decimal(x) for x in parts) <= Decimal('0.15') and any('龜山島周圍海岸占 9.57%，其餘為其他島嶼' in t_ for t_ in got)
        return sum(Decimal(x) for x in parts) == Decimal(tot)
    bs_ = [nm for nm, (tot, parts) in sub.items() if not _sub_ok(nm, tot, parts)]
    chk('V11-C N8 海岸來源細項加總 = 該地點海岸來源比例（%d 地點）' % len(sub), not bs_ and len(sub) == 4, str(bs_))
    ar = sorted(Decimal(a_.replace(',', '')) for a_ in area.values())
    chk('V11-C N9 綜合說明 50% 高密度區間面積範圍 = 各地點最小最大', any('面積為 %s–%s km²' % (f'{int(ar[0]):,}', f'{int(ar[-1]):,}') in t_ for t_ in got), str(area))
    dv = [Decimal(v_) for v_ in depth.values() if v_ is not None] + [Decimal('100.00')] * sum(1 for v_ in depth.values() if v_ is None)
    chk('V11-C N10 水深淺於 200 m 之比例範圍 = 各地點最小最大', any('%s%%–100.00%%' % min(dv) in t_ for t_ in got), str(depth))
    chk('V11-C N11 顆數：20 × 50 × 10 = 10,000（每類）；× 10 類 = 100,000；× 5 地點 = 500,000，與 3-4-3 每類 10,000 顆一致', 20 * 50 * 10 == 10000 and '每類各有 10,000 顆' in n343 and any('100,000' in t_ and '500,000' in t_ for t_ in got))
    chk('V11-C N12 3-4-3 敘述之 20 種到達地點、50 種到達時間、10 類、五個研究地點確實存在', all(k_ in n343 for k_ in ['設定 20 種不同', '設定 50 種不同', '10 類', '五個研究地點']))
    # ---- X 系列
    item4 = next(t_ for t_ in alltext if t_.startswith('4、溯源停止邊界')); item1 = next(t_ for t_ in alltext if t_.startswith('1、沉降動力參數')); para3 = next(t_ for t_ in alltext if t_.startswith('十種項目對應'))
    chk('V11-C X1 停止邊界段不含任何結果數字（無百分比、無 33.4、無各地點比例、無地點名）', not re.search(r'\d+(\.\d+)?%|33\.4|穿越比例|海岸來源比例|貢寮|龜山島|新竹|南灣|連江', item4), item4[-80:])
    chk('V11-C X2 停止邊界段保留停止條件定義，段末為「即停止追蹤。」，無「180」', '虛擬質點穿越流場範圍邊界或抵達海岸（視為找到來源，稱為海岸來源）即停止追蹤' in item4 and item4.rstrip().endswith('即停止追蹤。') and '180' not in item4, item4[-60:])
    chk('V11-C X3 第 1 項標題無「上升」、為「設定 10 種不同的沉降速度」，無「假設」', '設定 10 種不同的沉降速度' in item1 and '假設' not in item1 and '上升' not in item1)
    chk('V11-C X4 第 1 項「各類速度均為沉降速度。」且無「不含上升情境」', '各類速度均為沉降速度。每類由' in para3 and '上升情境' not in para3)
    up = [t_[max(0, m_.start() - 2):m_.end() + 4] for t_ in alltext for m_ in re.finditer('上升|上浮|浮升|上漂', t_)]
    chk('V11-C X5 全文「上升」等字眼僅餘兩處「潮位上升」', len(up) == 2 and all(x_.startswith('潮位上升') for x_ in up), str(up))
    rt = '\n'.join(got)
    chk('V11-C X6 「穿越比例」「海岸來源比例」之定義恰一處，位於龜山島之仍在追蹤比例導引段（龜山島之虛擬質點穿越流場範圍邊界者占 78.35%，稱為穿越比例，抵達海岸者占 21.65%，稱為海岸來源比例）', rt.count('稱為穿越比例') == 1 and rt.count('稱為海岸來源比例') == 1 and '龜山島之虛擬質點穿越流場範圍邊界者占 78.35%，稱為穿越比例，抵達海岸者占 21.65%，稱為海岸來源比例' in GUIDE[('guishan', 'curve')])
    chk('V11-C X7 33.4 天（連江仍在追蹤比例導引段；綜合說明）與各地點穿越比例（五地點導引段）', '至第 33.4 天降至 0%' in GUIDE[('lienchiang', 'curve')] and '皆於 33.4 天內停止追蹤' in rt and all(('穿越比例為 %s%%' % PUB[n_][0]) in GUIDE[(SKOF[n_], 'curve')] or n_ == '龜山島' for n_ in SITES11))
    # ---- E 系列：百分比小數位數與用語
    pct_bad = []
    for t_ in got:
        for m_ in re.finditer(r'(\d+(?:\.\d+)?)%', t_):
            v_ = m_.group(1)
            if '.' in v_:
                if not (len(v_.split('.')[1]) == 2 or v_ == '0.001'): pct_bad.append(v_)
            elif v_ not in ('0', '1', '10', '50', '75', '90', '100'): pct_bad.append(v_)
    chk('V11-C E1 成果段百分比小數位數：2 位（0.001% 例外）；整數僅限 0、1、10、50、75、90、100', not pct_bad, str(pct_bad[:5]))
    forb = ['各地點', '到達時刻', '沉底時間', '流場邊界', '穿越點', '反射', '取樣誤差', '逐時取樣', '接收點', '錨定點', '海岸源', '觸地', '著底', '受點', '研究站', '研究點']
    fl_ = [f_ for f_ in forb if any(f_ in t_.replace('各研究地點', '') for t_ in got)]
    chk('V11-C E2 成果段無禁用／異形用語（各地點、到達時刻、沉底時間、流場邊界、穿越點、反射、取樣誤差、逐時取樣、接收點、錨定點、海岸源、觸地、著底、受點、研究站、研究點）', not fl_, str(fl_))
    fl3 = [f_ for f_ in forb if any(f_ in t_.replace('各研究地點', '') for t_ in [txt(e) for e in kc[:ia + 1] if isp(e)])]
    chk('V11-C E3 3-4-3 情境段亦無上列異形用語', not fl3, str(fl3))
    req = ['到達時間', '沉底時刻', '流場範圍邊界', '海岸來源', '觸底', '受體', '研究地點', '錨點']
    allres = '\n'.join(got + n343.split('\n'))
    chk('V11-C E4 八個統一術語皆出現於合併文件', all(r_ in allres for r_ in req), str([r_ for r_ in req if r_ not in allres]))
    chk('V11-C E5 成果段不得含「反射」「A2」「方法 A／B／D」', not re.search(r'反射|A2|方法 ?[ABD]', rt))
    chk('V11-C E6 無表情符號', not re.search('[\U0001F300-\U0001FAFF☀-➿]', '\n'.join(alltext)))
    chk('V11-C E7 「臺灣本島」用語（無「台灣本島」「臺灣島」）', not any(('台灣本島' in t_ or '臺灣島' in t_) for t_ in alltext))
    # ---- 版面格式（成果段，與 3-4-3 參照段；左縮依使用者手改 1260）
    rpr_has = lambda p_, nm_: (p_.find(q('pPr')) is not None and p_.find(q('pPr')).find(q(nm_)) is not None)
    def outl(p_):
        ps_ = p_.find(q('pPr')); e_ = ps_.find(q('outlineLvl')) if ps_ is not None else None
        return int(e_.get(q('val'))) if e_ is not None else None
    def ref_para(prefix): return next(e for e in kA if isp(e) and txt(e).startswith(prefix))
    def np_(p_):
        ps_ = p_.find(q('pPr')); return strip_keep(clean(ps_)).replace(' w:hint="eastAsia"', '') if ps_ is not None else ''   # 使用者 Word 重存會於 rPr 加 hint=eastAsia（不影響版面）
    R_L2 = np_(ref_para('（一）')); R_L3 = np_(ref_para('4、溯源')); R_BODY = np_(ref_para('每個水平位置再隨機取'))
    R_CAP = np_(ref_para('圖3 ')); R_FIG = np_(kA[14])
    L4 = R_L3.replace('w:left="980" w:hangingChars="150" w:hanging="420"', 'w:left="1260" w:hangingChars="250" w:hanging="700"')
    L3N = R_L3.replace('w:left="980" w:hangingChars="150" w:hanging="420"', 'w:left="560"')
    L4N = R_L3.replace('w:left="980" w:hangingChars="150" w:hanging="420"', 'w:leftChars="354" w:left="991"')
    okfmt = {k_: 0 for k_ in ('L2', 'L3', 'L4', 'ITEM', 'BODY', 'CAP', 'FIG')}; badf = []
    for p_ in ins:
        if not isp(p_): continue
        t_ = txt(p_); pp_ = np_(p_)
        if figs_in(p_): k_ = 'FIG'; ok_ = pp_ == R_FIG
        elif re.match(r'^圖\d+ ', t_): k_ = 'CAP'; ok_ = pp_ == R_CAP
        elif t_ == '（二）分析成果：': k_ = 'L2'; ok_ = pp_ == R_L2
        elif t_ in ('逐地點成果：', '綜合說明與結論：', '整體結論：'): k_ = 'L3'; ok_ = pp_ == L3N
        elif t_ in SITES11: k_ = 'L4'; ok_ = pp_ == L4N
        elif re.match(r'^（\d）', t_): k_ = 'ITEM'; ok_ = pp_ == L4
        else: k_ = 'BODY'; ok_ = pp_ == R_BODY
        okfmt[k_] += ok_
        if not ok_: badf.append((k_, t_[:20]))
    chk('V11-C V1 各層級段落格式與 3-4-3 對應段落一致（條列左縮 1260 為使用者手改；L3 小標左 560 無懸掛、研究地點小標左 991 與內文對齊）', not badf, str(badf[:3]))
    chk('V11-C V2 各類段落數：L2×1、L3×3、L4×5、條列×5（使用者已刪第 6 條）、圖×55、圖說×55', (okfmt['L2'], okfmt['L3'], okfmt['L4'], okfmt['ITEM'], okfmt['FIG'], okfmt['CAP']) == (1, 3, 5, 5, 55, 55), str(okfmt))
    allowed_rp = {'<w:rPr><w:szCs w:val="28"/></w:rPr>', '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:szCs w:val="28"/></w:rPr>',
                  '<w:rPr><w:color w:val="000000"/><w:szCs w:val="28"/></w:rPr>', '<w:rPr><w:noProof/></w:rPr>'}
    rp_ = set()
    for p_ in ins:
        for r_ in p_.iter(q('r')):
            x_ = r_.find(q('rPr'))
            if x_ is not None: rp_.add(clean(x_))
    chk('V11-C V3 成果段 run 格式僅限 3-4-3 所用之 4 種', rp_ <= allowed_rp, str(rp_ - allowed_rp))
    insxml = ''.join(etree.tostring(p_, encoding='unicode') for p_ in ins)
    chk('V11-C V4 成果段僅用 afe 或無樣式', set(re.findall(r'<w:pStyle w:val="([^"]*)"', insxml)) <= {'afe'})
    chk('V11-C V5 成果段無 w:sz／w:b／eastAsia 直接格式', not re.search(r'<w:sz |<w:b/>|<w:b |<w:i/>|w:eastAsia=', insxml))
    capL = [(n_, txt(p_)) for n_, p_ in enumerate(kc) if isp(p_) and re.match(r'^圖\d+ ', txt(p_))]
    chk('V11-C V6 圖說：「圖」＋「編號＋半形空格」＋說明（%d 個）' % len(capL), len(capL) == 62 and all(re.match(r'^圖\d+ \S', t_) for _, t_ in capL))
    # ---- 結構與標題
    mk = ['Lagrangian系集逆向溯源', '（一）情境矩陣與停止條件設計：', '1、沉降動力參數：', '2、到達地點', '3、到達時間', '4、溯源停止邊界', '（二）分析成果：', '逐地點成果：'] + SITES11 + ['綜合說明與結論：', '整體結論：', '參考資料', '附錄　十類海底廢棄物終端沉降速度之推導']
    pos_ = []; s0 = 0
    for m_ in mk:
        k_ = idx_text(kc, (lambda t, m_=m_: t == m_) if m_ in SITES11 or m_ in ('逐地點成果：', '綜合說明與結論：', '整體結論：', '參考資料') else (lambda t, m_=m_: t.startswith(m_)), s0)
        pos_.append(k_)
        if k_ is not None: s0 = k_ + 1
    chk('V11-C H1 標題依序：標題→（一）→1–4→（二）→逐地點成果→龜山島、貢寮、新竹、南灣、連江→綜合說明與結論→整體結論→參考資料→附錄；參考資料與附錄位於全文最末', None not in pos_ and pos_ == sorted(pos_), str(pos_))
    nf_ = [sum(1 for n_ in range(a_, b_) if isp(kc[n_]) and figs_in(kc[n_])) for a_, b_ in ((hd[n_], endb[n_]) for n_ in SITES11)]
    chk('V11-C H3 每個研究地點各 11 張圖', nf_ == [11] * 5, str(nf_))
    chk('V11-C H5 「詳見附錄」之參照仍成立', any('詳見附錄' in t_ for t_ in alltext) and pos_[-1] is not None)
    # 附錄、參考資料：與 v10 候選逐區塊、逐字、逐格式相同（v10 候選已由 v10 檢查與 3-4-3 逐項比對；此處不重做 v4–v7 之正規化）
    ap_c = idx_text(kc, lambda t: t.startswith('附錄　十類')); ap_10 = idx_text(k10, lambda t: t.startswith('附錄　十類')); rf_c = idx_text(kc, lambda t: t == '參考資料'); rf_10 = idx_text(k10, lambda t: t == '參考資料')
    same_tail = (len(kc) - rf_c == len(k10) - rf_10) and all(sg(kc[rf_c + j_], zc, rlc) == sg(k10[rf_10 + j_], z10, rl10) for j_ in range(len(kc) - rf_c))
    chk('V11-C H6/H7 參考資料與附錄（含表 A1–A12，共 %d 個區塊）與 v10 候選（已與 3-4-3 逐項比對，附錄全文相同〔僅登記之用語統一、科學記號、重壓扁〕）逐字、逐格式、逐圖片相同' % (len(kc) - rf_c), same_tail)
    # ---- 標題層級、分頁保持
    apx_n = ap_c
    exp_lvl = {}
    L1s = {'（一）情境矩陣與停止條件設計：', '（二）分析成果：'}; L2s = {'逐地點成果：', '綜合說明與結論：', '整體結論：', '推導流程'}
    for n_, e in enumerate(kc):
        if not isp(e) or figs_in(e): continue
        t_ = txt(e)
        if n_ == 0 and t_.startswith('Lagrangian系集逆向溯源'): exp_lvl[n_] = (0, True)
        elif t_ in L1s: exp_lvl[n_] = (1, True)
        elif t_ in L2s: exp_lvl[n_] = (2, True)
        elif n_ > apx_n and re.match(r'^([一二三四五六七八]、\S{2,20}|補充[一二]、\S{2,25})$', t_): exp_lvl[n_] = (2, True)
        elif t_ in SITES11 and n_ < apx_n: exp_lvl[n_] = (3, True)
        elif n_ > apx_n and re.match(r'^步驟[一二三四五]　\S.{2,40}$', t_): exp_lvl[n_] = (3, True)
        elif t_ in ('參考資料', '附錄　十類海底廢棄物終端沉降速度之推導'): exp_lvl[n_] = (1, False)
    chk('V11-C K0 標題段共 29 個', len(exp_lvl) == 29, str(len(exp_lvl)))
    chk('V11-C K1 全部標題段皆有 keepNext 與 keepLines', all(rpr_has(kc[n_], 'keepNext') and rpr_has(kc[n_], 'keepLines') for n_ in exp_lvl), str([txt(kc[n_])[:8] for n_ in exp_lvl if not (rpr_has(kc[n_], 'keepNext') and rpr_has(kc[n_], 'keepLines'))]))
    chk('V11-C O1 直接大綱層級值與預期相符（0/1/2/3；27 個直接設定）', all(outl(kc[n_]) == (lv_ if d_ else None) for n_, (lv_, d_) in exp_lvl.items()) and sum(1 for n_, (lv_, d_) in exp_lvl.items() if d_) == 27, str([(txt(kc[n_])[:8], outl(kc[n_]), lv_) for n_, (lv_, d_) in exp_lvl.items() if outl(kc[n_]) != (lv_ if d_ else None)][:5]))
    stz = etree.fromstring(zc.read('word/styles.xml'))
    st21 = next(s_ for s_ in stz.findall(q('style')) if s_.get(q('styleId')) == '21')
    chk('V11-C O2 參考資料與附錄維持 Heading 2 樣式（outlineLvl=1），未加直接層級', st21.find(q('pPr')).find(q('outlineLvl')).get(q('val')) == '1' and all(pstyle(kc[n_]) == '21' for n_, (lv_, d_) in exp_lvl.items() if not d_))
    styled = {s_.get(q('styleId')) for s_ in stz.findall(q('style')) if s_.find(q('pPr')) is not None and s_.find(q('pPr')).find(q('outlineLvl')) is not None}
    others_ = [txt(e)[:15] for n_, e in enumerate(kc) if isp(e) and n_ not in exp_lvl and txt(e).strip() and (outl(e) is not None or pstyle(e) in styled)]
    chk('V11-C O3 除上列標題外，無其他有文字之段落有大綱層級', not others_, str(others_[:5]))
    seq_l = [(exp_lvl[n_][0], txt(kc[n_])) for n_ in sorted(exp_lvl)]
    chk('V11-C O4 大綱層級序列不跳級', all(b_[0] <= a_[0] + 1 for a_, b_ in zip(seq_l, seq_l[1:])) and seq_l[0][0] == 0)
    tcn = [n_ for n_, e in enumerate(kc) if isp(e) and re.match(r'^表(A?\d+) ', txt(e)) and n_ + 1 < len(kc) and kc[n_ + 1].tag == q('tbl')]
    chk('V11-C K2 表標題（緊接表格者）皆有 keepNext 與 keepLines（14 個）', len(tcn) == 14 and all(rpr_has(kc[n_], 'keepNext') and rpr_has(kc[n_], 'keepLines') for n_ in tcn), str(len(tcn)))
    fgn = [n_ for n_, e in enumerate(kc) if isp(e) and figs_in(e)]
    chk('V11-C K3 全部 62 個圖段皆有 keepNext', len(fgn) == 62 and all(rpr_has(kc[n_], 'keepNext') for n_ in fgn))
    intro_n = [n_ - 1 for n_ in fgn if n_ - 1 >= 0 and isp(kc[n_ - 1]) and not figs_in(kc[n_ - 1]) and (n_ - 1) not in exp_lvl and txt(kc[n_ - 1]).strip() and not re.match(r'^(圖|表)(A?\d+) ', txt(kc[n_ - 1]))]
    chk('V11-C K4 圖前導引段共 56 個（到達時間 1 + 55）皆有 keepLines', len(set(intro_n)) == 56 and all(rpr_has(kc[x_], 'keepLines') for x_ in intro_n), str(len(set(intro_n))))
    # ---- 圖檔與圖之尺寸、封裝
    sz_bad = []
    for (sk_, r_), idl in FIGAT.items():
        f_ = figs_in(kc[idl[0]])[0]
        if r_ in ('arrive', 'traj', 'foot') and f_[:2] != SZ[(sk_, r_)]: sz_bad.append((sk_, r_))
    chk('V11-C F3/F4/F5 到達時間驗證圖（5 張）與軌跡、足跡圖（10 張）之圖檔 sha（由辨識證明）與顯示尺寸皆與 3-4-3／執行成果原圖相同；圖 1、圖 2 顯示尺寸與 3-4-3 原圖相同', not sz_bad and all(cfigs[k_][1][:2] == figsA[k_][0] for k_ in (0, 1)), str(sz_bad))
    textw = 11906 - 2 * 1440
    chk('V11-C F7 全部圖寬不超過版心寬（容許 0.5 pt 進位）', all(f_[1][0] <= textw * 635 + 6350 for f_ in cfigs))
    tbls = [e for e in kc if e.tag == q('tbl')]
    chk('V11-C F8 表格數不變（%d，同 3-4-3）' % len(tbls), len(tbls) == len([e for e in kA if e.tag == q('tbl')]))
    tcap = [txt(p_) for p_ in kc if isp(p_) and re.match(r'^表(A?\d+) ', txt(p_))]
    tn = [re.match(r'^表(A?\d+)', t_).group(1) for t_ in tcap]
    chk('V11-C F9 表號：表1、表2、表A1–A12 連續且不變', tn == ['1', '2'] + ['A%d' % n_ for n_ in range(1, 13)], str(tn))
    chk('V11-C F2 每個圖說緊接前一段為圖片段', all(isp(kc[n_ - 1]) and figs_in(kc[n_ - 1]) for n_, _ in capL))
    ids_ = [f_[1][3] for f_ in cfigs]
    chk('V11-C P9/P11 docPr id 全文唯一（62 張圖）；62 張圖各引用一個圖檔、無共用', len(ids_) == len(set(ids_)) == 62 and len({rlc[f_[1][2]] for f_ in cfigs}) == 62)
    chk('V11-C P10 全部 r:embed 指向存在之關係與檔案', all(f_[1][2] in rlc and ('word/' + rlc[f_[1][2]]) in zc.namelist() for f_ in cfigs))
    chk('V11-C P12 zip 無孤立之圖檔（word/media 內檔案 = 關係所引用者）', {n_ for n_ in zc.namelist() if n_.startswith('word/media/')} == {'word/' + t_ for t_ in rlc.values() if t_.startswith('media/')})
    try:
        import docx as _docx; _docx.Document(MERGED); okd = True
    except Exception: okd = False
    ctx = zc.read('[Content_Types].xml').decode()
    chk('V11-C P5/P13 Content_Types 含 jpg 與 png；python-docx 可開啟', 'jpg' in ctx and 'png' in ctx and okd)
    # ---- 40 張來源分段機率圖與 5 張曲線圖：圖檔與尺寸
    okF = []; detF = []
    for sk_ in SITEK:
        for k_, fk_ in zip(KEYS8, FIGK):
            i_ = FIGAT[(sk_, k_)][0]; f_ = figs_in(kc[i_])[0]; data_ = zc.read('word/' + rlc[f_[2]]); w_, h_ = struct.unpack('>II', data_[16:24])
            disk = open(os.path.join(SEG, sk_, MAN['figures'][sk_][fk_]['png']), 'rb').read() if k_ in ('all_b', 'all_c') else open(os.path.join(FINAL9, sk_, fk_ + '.png'), 'rb').read()
            ent_sha = MAN['figures'][sk_][fk_]['sha256'] if k_ in ('all_b', 'all_c') else sha(disk)
            ok_ = data_[:8] == b'\x89PNG\r\n\x1a\n' and sha(data_) == ent_sha == sha(disk) and f_[0] == 5733288 and f_[1] == round(h_ / 600.0 * 914400) and w_ == 3762
            okF.append(ok_)
            if not ok_: detF.append((sk_, k_))
    chk('V11-C F10 40 張來源分段機率圖之 sha 等於 segprob_v3 manifest／segprob_fullpage/final 磁碟檔，寬 6.27 in（5733288 EMU、3762 px／600 dpi），高依像素等比，未縮放', len(okF) == 40 and all(okF), str(detF[:4]))
    cu_bad = []
    for sk_, c_ in CURV.items():
        i_ = FIGAT[(sk_, 'curve')][0]; f_ = figs_in(kc[i_])[0]; data_ = zc.read('word/' + rlc[f_[2]]); w_, h_ = struct.unpack('>II', data_[16:24])
        if not (sha(data_) == c_['png_sha256'] and w_ == 3762 and f_[0] == 5733288 and f_[1] == round(h_ / 600.0 * 914400) == c_['new_cy'] and h_ == c_['h_px']): cu_bad.append(sk_)
    chk('V11-C V10-10 5 張仍在追蹤比例圖（選項 B 曲線圖）：圖檔 sha 同建置紀錄、寬 6.27 in、高依像素等比、未縮放', len(CURV) == 5 and not cu_bad, str(cu_bad))
    rbad2 = []; nchk10 = 0; ncol9 = 0
    for sk_ in SITEK:
        rj = json.load(open(os.path.join(FINAL9, sk_, 'result.json'), encoding='utf-8'))
        if rj['fails']: rbad2.append((sk_, rj['fails'][:2]))
        mf_ = json.load(open(os.path.join(FINAL9, sk_, 'manifest.json'), encoding='utf-8'))
        for fk_ in FIGK[2:]: ncol9 += mf_[fk_]['collision_count']
        rows_ = list(csv.DictReader(open(os.path.join(FINAL9, sk_, 'checks.csv'), encoding='utf-8-sig'))); nchk10 += len(rows_)
        rbad2 += [(sk_, r_['figure'], r_['panel'], r_['check']) for r_ in rows_ if r_['result'] == 'FAIL']
    chk('V11-C V9-10/V10-11 五地點 30 張多格圖之版面與數值檢查（final/<site>/result.json 與 checks.csv，共 %d 列）：失敗 0；數值框、括號、圈號、標籤之碰撞 0' % nchk10, not rbad2 and ncol9 == 0 and nchk10 > 0, str(rbad2[:3]) + ' coll=%d' % ncol9)
    # ---- 圖號引用（以圖類型為鍵）
    nrng = {nm: (min(FN[(SKOF[nm], r_)] for r_ in ROLES[1:]), max(FN[(SKOF[nm], r_)] for r_ in ROLES[1:])) for nm in SITES11}
    rbad = []
    for nm in SITES11:
        lo, hi = nrng[nm]
        for t_ in blocks[nm]:
            if re.match(r'^圖\d+ ', t_): continue
            for m_ in re.finditer(r'圖 (\d+)', t_):
                if not (lo <= int(m_.group(1)) <= hi): rbad.append((nm, m_.group(1)))
    chk('V11-C R6 各地點段落之圖號引用皆在該地點之 11 張圖內（圖號範圍由辨識結果取得：%s）' % '、'.join('%s %d–%d' % (n_, nrng[n_][0], nrng[n_][1]) for n_ in SITES11), not rbad and all(nrng[n_][1] - nrng[n_][0] == 10 for n_ in SITES11), str(rbad[:4]))
    m_all = [(nm, m_) for nm in SITES11 for t_ in blocks[nm] for m_ in re.finditer(r'圖 (\d+) 至圖 (\d+)', t_)]
    a0, b0 = FN[('guishan', 'all_b')], FN[('guishan', 'tid_c')]
    chk('V11-C R4 「圖 a 至圖 b」僅 1 處，位於龜山島（第一個地點）之來源分段取法段，a = 全年邊界圖、b = 大小潮海岸圖（共 8 張）', len(m_all) == 1 and m_all[0][0] == '龜山島' and (int(m_all[0][1].group(1)), int(m_all[0][1].group(2))) == (a0, b0) and b0 - a0 == 7, str([(a_, m_.group(0)) for a_, m_ in m_all]))
    chk('V11-C R5 十類圖不再引用「圖例」順序（各格標題標明類別）', not any('圖例' in t_ for t_ in got))
    refs_all = sorted({int(m_.group(1)) for t_ in alltext for m_ in re.finditer(r'圖 ?(\d+)', t_)})
    chk('V11-C R2 成果段內圖號引用皆在 3–62（既有圖）', all(1 <= n_ <= 62 for n_ in refs_all))
    # ---- 文字修飾（W 系列；與順序無關者）
    hai = [FT[max(0, m_.start() - 8):m_.end() + 6] for m_ in re.finditer('海廢', FT)]
    chk('V11-C W1 全文（含表格）無「垂向」「代理」', not any(w_ in FT for w_ in ('垂向', '代理')))
    chk('V11-C W2 「海廢」僅餘 2 處（十個海廢分類名稱之原名；第 4 項「鎖定海廢移入」為計畫書原文）；「海底廢」後皆接「棄物」；「海漂底廢棄物調查指引」2 處',
        len(hai) == 2 and any('海廢分類名稱' in h_ for h_ in hai) and any('鎖定海廢移入' in h_ for h_ in hai) and all(FT[m_.end():m_.end() + 2] == '棄物' for m_ in re.finditer('海底廢', FT)) and FT.count('海漂底廢棄物調查指引') == 2, str(hai))
    chk('V11-C W3/W17 「海底廢棄物」共 7 處；全文（含表格、圖說、附錄）無「本研究」（含跨 run 殘留）；「本計畫」共 18 處', FT.count('海底廢棄物') == 7 and FT.count('本研究') == 0 and FT.count('本計畫') == 18 and '本研究' not in ''.join(t_.text or '' for t_ in rc.iter(q('t'))), '%d %d %d' % (FT.count('海底廢棄物'), FT.count('本研究'), FT.count('本計畫')))
    lg = re.findall(r'Lagrangian(.)', FT)
    chk('V11-C W4 「Lagrangian」與中文之間：3 處皆無空格', len(lg) == 3 and all('一' <= c_ <= '鿿' for c_ in lg), str(lg))
    item2 = next(t_ for t_ in alltext if t_.startswith('共由「5 個水平位置'))
    chk('V11-C W5 受體選取段：保留半徑 10 km、方位×距離各 5 等面積區間、區塊內隨機抽取；已無「濕潤」「逐時空間支援檢查」「遞補」',
        '受體位置限於全部到達時間皆有流場與波浪資料之海域；符合條件者再依方位及距研究地點錨點之距離' in item2 and '半徑 10 km 範圍內之候選海域等分為面積相同之 5 個區間' in item2 and item2.endswith('內隨機抽取。') and not any(w_ in FT for w_ in ('濕潤', '逐時空間支援檢查', '遞補')))
    chk('V11-C W6 「4 個深度抽樣」；「這 4 個深度的編號只是抽籤順序，不代表表層、中層或底層。」保留', '4 個深度抽樣」構成' in item2 and '這 4 個深度的編號只是抽籤順序，不代表表層、中層或底層。' in FT)
    item3 = next(t_ for t_ in alltext if t_.startswith('50 個到達時間由 48 個分層時刻'))
    chk('V11-C W7 到達時間段：「三種潮相」「以潮位資料判定」；無「日間銜接」「相互重疊」', '四季 × 大潮／小潮 × 三種潮相」共 24 格' in item3 and '低於者為小潮。三種潮相以潮位資料判定，分別為潮位上升最快、潮位下降最快' in item3 and '日間銜接' not in FT and '相互重疊' not in FT)
    chk('V11-C W10 #9 無「未必涵蓋」、有「用以納入分層設計以外之極端動力狀態」；第 4 項為計畫書原文「藉此精確鎖定海廢移入該海域之潛在外部來源與邊界通量路徑。」', '未必涵蓋' not in FT and '用以納入分層設計以外之極端動力狀態' in FT and '藉此精確鎖定海廢移入該海域之潛在外部來源與邊界通量路徑。' in FT and '辨識海底廢棄物' not in FT)
    chk('V11-C W11 （一）3 不含「沉底」；全文「沉底年齡」恰 1 處（龜山島之季節導引段）；無「0–90」', '沉底' not in item3 and item3.endswith('各潮相皆有代表時刻。') and FT.count('沉底年齡') == 1 and '0–90' not in FT and '沉底年齡' in GUIDE[('guishan', 'sea_b')])
    chk('V11-C W19 五處改後句——（一）1 末段「各類 10 個速度係以受力平衡求解，」、附錄三「參照其兩極限相加之概念，黏滯項改採 Schiller–Naumann 式」「本計畫採用之薄板，依教科書（表 11-2）有限矩形板式為 1.14–1.18」、附錄五「尚無實測，故以表 A4 之有效密度作為設定值。」、綜合說明「南灣另有 1 顆觸底）。穿越比例為」',
        all(x_ in FT for x_ in ('各類 10 個速度係以受力平衡求解，', '。參照其兩極限相加之概念，黏滯項改採 Schiller–Naumann 式', '本計畫採用之薄板，依教科書（表 11-2）有限矩形板式為 1.14–1.18', '尚無實測，故以表 A4 之有效密度作為設定值。', '南灣另有 1 顆觸底）。穿越比例為')) and not any(x_ in FT for x_ in ('由本計畫以受力平衡', '本計畫參照其兩極限', '本計畫薄板')))
    chk('V11-C W14 （二）內小標已不編號（無「1、逐地點成果」「（1）龜山島：」獨立成行之小標）；整體結論條列（1）–（5）保留編號且依 龜山島、貢寮、新竹、南灣、連江 序；（一）項目 1–4 與附錄編號不動',
        not any(w_ in FT for w_ in ('1、逐地點成果', '2、綜合說明與結論', '3、整體結論')) and not any(('\n' + '（%d）%s：\n' % (i_ + 1, s_)) in FT for i_, s_ in enumerate(SITES11))
        and all(any(t_.startswith('（%d）%s：' % (i_ + 1, s_)) and len(t_) > 40 for t_ in alltext) for i_, s_ in enumerate(SITES11)) and all(any(t_.startswith('%d、' % k_) for t_ in alltext) for k_ in (1, 2, 3, 4)))
    # 表 1
    t1 = next(e for e in kc if e.tag == q('tbl'))
    okr = 0
    for p_ in t1.iter(q('p')):
        rs_ = list(p_.iter(q('r')))
        for k_, r_ in enumerate(rs_):
            if ''.join(t.text or '' for t in r_.iter(q('t'))) == '至':
                if rs_[k_ - 1].find(q('br')) is not None and ''.join(t.text or '' for t in rs_[k_ + 1].iter(q('t'))).startswith(' '): okr += 1
    chk('V11-C W8 表 1 速度範圍 10 列：「至」前為手動換行、後為半形空格；表 A 系列之「至」皆前後半形空格', okr == 10 and not re.search(r'\d%至|至\+|\S至 [+−]', ''.join(txt(e) for e in kc[ap_c:] if e.tag == q('tbl'))), str(okr))
    GRID_T1 = [1100, 2050, 3407, 1900]; GRID_A10 = [1494, 2300, 2673, 1990]
    _gc = [int(x.get(q('w'))) for x in t1.find(q('tblGrid')).findall(q('gridCol'))]
    chk('V11-C W13 表 1 欄寬 %s（總寬 %d 不變）；第 4 欄 %.1f pt 放得下第二行「至 1.08×10⁻¹」但放不下一行，故兩行、保留「至」前手動換行' % (_gc, sum(_gc), (_gc[3] - 240) / 20.0), _gc == GRID_T1 and sum(_gc) == 8457 and 75 < (_gc[3] - 240) / 20.0 < 130)
    V6F_OLD = [('0.000897', '0.108'), ('0.00661', '0.162'), ('0.00527', '0.187'), ('0.00795', '0.223'), ('0.0293', '0.134'), ('0.0135', '0.303'), ('0.0342', '0.345'), ('0.0367', '0.175'), ('0.107', '0.427'), ('0.195', '0.920')]
    _rows = t1.findall(q('tr'))[1:]; _bad = []; _fmt = set()
    for _r, (_o1, _o2) in zip(_rows, V6F_OLD):
        _p = list(_r.findall(q('tc'))[3].iter(q('p')))[0]; _parts = []; _br = 0
        for _run in _p.findall(q('r')):
            _sup = _run.find(q('rPr')).find(q('vertAlign')) is not None and _run.find(q('rPr')).find(q('vertAlign')).get(q('val')) == 'superscript'
            if _run.find(q('br')) is not None: _br += 1; _parts.append(('br', '')); continue
            for _t in _run.findall(q('t')): _parts.append(('sup' if _sup else 'n', _t.text or ''))
        _mg = []
        for k_, t_ in _parts:
            if _mg and _mg[-1][0] == k_ and k_ == 'sup': _mg[-1] = (k_, _mg[-1][1] + t_)
            else: _mg.append((k_, t_))
        _txt = ''.join(('^{%s}' % t_) if k_ == 'sup' else ('|' if k_ == 'br' else t_) for k_, t_ in _mg)
        _m = re.fullmatch(r'(\d(?:\.\d+)?)×10\^\{(−?\d)\}\|至 (\d(?:\.\d+)?)×10\^\{(−?\d)\}', _txt)
        if not _m or _br != 1: _bad.append(('格式', _txt)); continue
        _fmt.add(re.sub(r'\d', '9', _txt))
        for (_mt, _ex), _o in ((_m.group(1, 2), _o1), (_m.group(3, 4), _o2)):
            if Decimal(_mt + 'E' + _ex.replace('−', '-')) != Decimal(_o) or len(_mt.replace('.', '')) != len(_o.replace('.', '').lstrip('0')): _bad.append(('數值', _txt, _o))
            if not (1 <= Decimal(_mt) < 10): _bad.append(('尾數', _mt))
    chk('V11-C W20 表 1 十列科學記號（尾數×10 與上標指數〔vertAlign=superscript〕），換算後逐一等於原值（Decimal），有效位數照原值，「至」前保留手動換行，十列格式一致',
        len(_rows) == 10 and not _bad and len(_fmt) <= 3 and not any(c_ in ''.join(txt(e) for e in [t1]) for c_ in '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'), str(_bad) + str(_fmt))
    negpat = [r'[−\-]\s?0\.\d+(?![\d%])', r'[−\-]\s?\d(?:\.\d+)?×10', '取負值', '均為負值', '垂直座標向上為正', '向上為正']
    chk('V11-C W22 全文（含表格）無帶負號之沉降速度，無「取負值」「均為負值」「垂直座標向上為正」', not any(re.search(pp_, FT) for pp_ in negpat), str([pp_ for pp_ in negpat if re.search(pp_, FT)]))
    _t10 = next(e for n_, e in enumerate(kc) if e.tag == q('tbl') and txt(kc[n_ - 1]).startswith('表A10 '))
    def _grid(t_): return [int(x.get(q('w'))) for x in t_.find(q('tblGrid')).findall(q('gridCol'))]
    def _tcw_ok(t_): return all([int(tc.find(q('tcPr')).find(q('tcW')).get(q('w'))) for tc in tr.findall(q('tc'))] == _grid(t_) for tr in t_.findall(q('tr')))
    chk('V11-C W23 表 1 欄寬、表 A10 欄寬（各自總寬 8457 不變）；各列 tcW 與 tblGrid 一致；表 1 十個資料列皆 cantSplit', _grid(t1) == GRID_T1 and _grid(_t10) == GRID_A10 and sum(_grid(t1)) == sum(_grid(_t10)) == 8457 and _tcw_ok(t1) and _tcw_ok(_t10)
        and sum(1 for tr in t1.findall(q('tr'))[1:] if tr.find(q('trPr')) is not None and tr.find(q('trPr')).find(q('cantSplit')) is not None) == 10)
    # ---- 40 個導引段、40 個圖說（以 (地點, 類型) 為鍵）
    SEADEF_M = '沉底時刻為廢棄物沉降至海床之時刻，亦即虛擬質點回溯之起點；由於廢棄物實際何時沉底未知，沉底時刻由所屬到達時間往前推一段隨機抽取之沉底年齡決定，每個系集成員各自抽取；季節依沉底時刻所在月份劃分，冬季為 12–2 月、春季為 3–5 月、夏季為 6–8 月、秋季為 9–11 月。'
    GUI0 = {(c_['site'], c_['role'].split(':')[1]): c_['new'] for c_ in LOG['changes'] if c_['kind'] == 'new' and c_['role'].startswith('guide:')}
    FMlog = {(f_['site'], f_['key']): f_ for f_ in LOG['fig_map']}
    LONGD = '依沉底時刻之潮位資料劃分為大潮與小潮之'; SHORTD = '依大小潮劃分之'
    nzg = lambda t: re.sub(r'圖 ?\d+', '圖N', t).replace('新竹外海', '新竹').replace('龜山島沿岸', '龜山島周圍海岸').replace('龜山島占 ', '龜山島周圍海岸占 ').replace(SEADEF_M, '').replace(LONGD, SHORTD)
    gbase = [(sk_, k_) for sk_ in SITEK for k_ in KEYS8 if nzg(GUIDE[(sk_, k_)]) != nzg(GUI0[(sk_, k_)])]
    chk('V11-C G1 40 個來源分段機率導引段 = v10 建置紀錄之核定文字（扣除 v11 登記之變更：圖號、新竹外海→新竹、龜山島沿岸→龜山島周圍海岸、沉底時刻定義與大小潮定義句改隨龜山島）；逐 (地點, 類型) 比對', not gbase, str(gbase[:4]))
    meth = next(t_ for t_ in blocks['龜山島'] if re.match(r'^圖 \d+ 至圖 \d+ 為龜山島之來源分段機率圖', t_))
    texts_new = [GUIDE[(sk_, k_)] for sk_ in SITEK for k_ in KEYS8] + [CAPT[(sk_, k_)] for sk_ in SITEK for k_ in KEYS8] + [meth]
    i_w2 = idx_text(kc, lambda t: t == '（二）分析成果：')
    texts_new += [txt(kc[j_]) for j_ in range(i_w2 + 1, i_sh) if isp(kc[j_])]                       # （二）導言
    texts_new += [t_ for t_ in got if t_.startswith('各研究地點十類之間') or re.match(r'^（\d）(龜山島|貢寮|新竹|南灣|連江)：', t_)]   # 綜合說明、整體結論
    CAPREQ = {'all_b': ['邊界穿越來源分段機率（全年，十類合併）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例', '黑色括號標示機率集中之連續邊界段', '灰色括號標示其餘連續非零邊界段'],
              'all_c': ['海岸來源分段機率（全年，十類合併）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例'],
              'cls_b': ['邊界穿越來源分段機率（十類）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例', '括號標示機率集中之連續邊界段', '圓圈數字', '「其餘」'],
              'cls_c': ['海岸來源分段機率（十類）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例'],
              'sea_b': ['（依沉底時刻之季節）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例', '括號標示機率集中之連續邊界段', '圓圈數字', '「其餘」'],
              'sea_c': ['（依沉底時刻之季節）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例'],
              'tid_b': ['（依大小潮）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例', '括號標示機率集中之連續邊界段', '圓圈數字', '「其餘」'],
              'tid_c': ['（依大小潮）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例']}
    cbad = []
    for sk_ in SITEK:
        for k_ in KEYS8:
            c_ = CAPT[(sk_, k_)].replace(' ', '')
            for ph in CAPREQ[k_]:
                if ph.replace(' ', '') not in c_: cbad.append((ZH[sk_], k_, ph))
            if ZH[sk_] not in CAPT[(sk_, k_)]: cbad.append((ZH[sk_], k_, '地點名'))
    chk('V11-C C1 40 張圖之圖說依圖類型含規定用語（邊界每 10 km 一段；海岸每 5 km 一段、僅計流場範圍內海岸；色階為各段占全部虛擬質點之比例；單張邊界圖之黑、灰色括號句；多格邊界圖之括號、圓圈數字與「其餘」說明），並標明研究地點', not cbad, str(cbad[:4]))
    THR = r'[≥≧≤]|門檻|閾值|至少 ?\d|(?<![\d.])[13]%|2 段|1%以上|3%以上'
    thr = [t_[:20] for t_ in texts_new if re.search(THR, t_)]
    chk('V11-C C2 導引段、圖說、方法段、導言、綜合說明與結論（%d 段）不寫括號門檻（1%%、3%%、≥2 段、≥1%%）' % len(texts_new), not thr, str(thr[:3]))
    chk('V11-C C3 方法段含：邊界每 10 km／海岸每 5 km 分段、依穿越位置與首次接觸位置歸段、歸入邊界段／海岸段／觸底三者之一且只計入一次、機率合計 100%；不含「相近或差異明顯」「稱為相近」「抽樣而產生」「差異範圍」，無 360°、10° 一格、1/36、高於平均',
        all(x_ in meth for x_ in ['邊界每 10 km', '海岸每 5 km', '穿越位置', '首次接觸海岸之位置', '只歸入邊界段、海岸段與觸底三者之一，且只計入一次', '互不重疊', '合計為 100%']) and not re.search(r'360|10°|1/36|高於平均|平均值|相近|差異明顯|抽樣而產生|差異範圍|稱為相近|稱為差異', meth))
    # ---- N13：每個百分比皆出自 segprob_v3 之表
    nbad = []; gbad = []
    for sk_ in SITEK:
        nm = ZH[sk_]; allowed_ = set()
        for r_ in AS:
            if r_['site'] == sk_: allowed_.add('%.2f' % float(r_['pct']))
        for (ss, g_), r_ in GTR.items():
            if ss == sk_: allowed_ |= {'%.2f' % float(r_['crossed_pct']), '%.2f' % float(r_['coast_pct'])}
        for fk in FIGK:
            f_ = MAN['figures'][sk_][fk]
            for t_ in f_.get('panel_titles', []): allowed_ |= set('%.2f' % float(x_) for x_ in re.findall(r'([\d.]+)%', t_))
            anns = f_['annotations']; anns = [anns] if isinstance(anns, dict) else anns
            for a_ in anns:
                for st in a_['stretches']: allowed_.add('%.2f' % float(st.get('shown_pct', st.get('pct'))))
            if f_.get('inset'):
                for st in f_['inset']['annotations']['stretches']: allowed_.add('%.2f' % float(st['pct']))
        allowed_ |= set(sub[nm][1]) if nm in sub else set()
        allowed_ |= {'100.00', '0.00'}
        for k_ in KEYS8:
            for v_ in re.findall(r'(\d+\.\d+)%', GUIDE[(sk_, k_)].replace('<0.01%', '')):
                if v_ not in allowed_: nbad.append((nm, k_, v_))
        need = {'all_b': ['%.2f' % float(r_['pct']) for r_ in AS if r_['site'] == sk_ and r_['kind'] == 'boundary' and r_['group'] == 'all'] + ['%.2f' % float(st['shown_pct']) for st in MAN['figures'][sk_]['boundary_all']['annotations']['stretches']],
                'all_c': ['%.2f' % float(r_['pct']) for r_ in AS if r_['site'] == sk_ and r_['kind'] == 'coast' and r_['group'] == 'all']}
        for k_ in ('sea_b', 'sea_c', 'tid_b', 'tid_c'):
            need[k_] = ['%.2f' % float(x_) for t_ in MAN['figures'][sk_][FIGK[KEYS8.index(k_)]]['panel_titles'] for x_ in re.findall(r'①\s*([\d.]+)%', t_)]
        for k_, lst_ in need.items():
            miss = [v_ for v_ in lst_ if (v_ + '%') not in GUIDE[(sk_, k_)]]
            if miss: gbad.append((nm, k_, miss))
        for k_, gp, fld, lab in (('sea_b', 'season', 'crossed_pct', '穿越'), ('sea_c', 'season', 'coast_pct', '海岸來源'), ('tid_b', 'tide', 'crossed_pct', '穿越'), ('tid_c', 'tide', 'coast_pct', '海岸來源')):
            for m_ in re.finditer(r'(冬季|春季|夏季|秋季|大潮|小潮)%s比例為 ([\d.]+)%%' % lab, GUIDE[(sk_, k_)]):
                gk = {'冬季': 'season_DJF', '春季': 'season_MAM', '夏季': 'season_JJA', '秋季': 'season_SON', '大潮': 'tide_spring_proxy', '小潮': 'tide_neap_proxy'}[m_.group(1)]
                if '%.2f' % float(GTR[(sk_, gk)][fld]) != m_.group(2): nbad.append((nm, k_, m_.group(1), m_.group(2)))
        GK_ = {'冬季': 'season_DJF', '春季': 'season_MAM', '夏季': 'season_JJA', '秋季': 'season_SON', '大潮': 'tide_spring_proxy', '小潮': 'tide_neap_proxy'}
        for k_, labs_ in (('sea_c', ['冬季', '春季', '夏季', '秋季']), ('tid_c', ['大潮', '小潮'])):
            tx_ = GUIDE[(sk_, k_)]
            named0 = set(); namedt = set(re.findall(r'(冬季|春季|夏季|秋季|大潮|小潮)海岸來源比例為 <0\.01%', tx_))
            for m_ in re.finditer(r'((?:冬|春|夏|秋)(?:、(?:冬|春|夏|秋))*季|大潮|小潮)無海岸來源', tx_):
                named0 |= set(x_ + '季' for x_ in re.findall(r'[冬春夏秋]', m_.group(1))) if m_.group(1).endswith('季') else {m_.group(1)}
            z0 = {l_ for l_ in labs_ if int(GTR[(sk_, GK_[l_])]['coast_n']) == 0}
            zt = {l_ for l_ in labs_ if int(GTR[(sk_, GK_[l_])]['coast_n']) > 0 and '%.2f' % float(GTR[(sk_, GK_[l_])]['coast_pct']) == '0.00'}
            if named0 != z0 or namedt != zt or re.search(r'海岸來源比例為 0\.00%', tx_): nbad.append((nm, k_, '零值', sorted(named0), sorted(z0), sorted(namedt), sorted(zt)))
        cl = [r_ for (ss, g_), r_ in GTR.items() if ss == sk_ and g_.startswith('class_')]
        for k_, fld in (('cls_b', 'crossed_pct'), ('cls_c', 'coast_pct')):
            vv = [float(r_[fld]) for r_ in cl]
            if ('介於 %.2f%% 至 %.2f%%' % (min(vv), max(vv))) not in GUIDE[(sk_, k_)]: nbad.append((nm, k_, '十類範圍'))
    chk('V11-C N13 40 個導引段之每個百分比皆出自 segprob_v3 之表（括號值、分組合計、母版海岸子分項）；各分組編號括號之機率、全年邊界與海岸括號之機率皆出現於對應導引段；分組合計、零值寫法、十類範圍相符', not nbad and not gbad, str(nbad[:4]) + str(gbad[:3]))
    nbad2 = []
    for sk_ in SITEK:
        nm = ZH[sk_]; cc = concl[nm]
        mains = ['%.2f' % float(r_['pct']) for r_ in AS if r_['site'] == sk_ and r_['kind'] == 'boundary' and r_['group'] == 'all']
        coasts = ['%.2f' % float(r_['pct']) for r_ in AS if r_['site'] == sk_ and r_['kind'] == 'coast' and r_['group'] == 'all']
        for v_ in mains + coasts:
            if v_ + '%' not in cc: nbad2.append((nm, v_))
        if ('全年穿越比例為 %s%%' % PUB[nm][0]) not in cc or ('海岸來源占 %s%%' % PUB[nm][1]) not in cc: nbad2.append((nm, '總量'))
        if nm == '南灣' and '0.001%' not in cc: nbad2.append((nm, '觸底'))
        if nm == '連江' and not ('29.68%' in cc and '14.26%' in cc): nbad2.append((nm, '子分項'))
    chk('V11-C N6 整體結論（1）–（5）：全年穿越比例與海岸來源比例 = 已公布值；全年邊界與海岸括號值取自 segprob_v3；南灣觸底 0.001%、連江子分項 29.68／14.26 維持；季節部分為定性', not nbad2, str(nbad2[:4]))
    # ---- 判定：diff_verdicts_final.csv
    vbad = []
    for sk_ in SITEK:
        nm = ZH[sk_]; rows = [r_ for r_ in VER if r_['site'] == sk_]
        sea = [r_ for r_ in rows if r_['kind'] == 'season']; sim = [(r_['group_a'], r_['group_b']) for r_ in sea if r_['verdict_final'] == '相近']
        if len(sea) != 6: vbad.append((nm, 'season rows'))
        txt_ = GUIDE[(sk_, 'sea_c')]; cc = concl[nm]
        if not sim: ok1 = '四季之分段分佈兩兩差異明顯' in txt_ and '四季之分段分佈兩兩差異明顯' in cc
        else: ok1 = sim == [('DJF', 'MAM')] and '冬、春兩季之分段分佈相近，其餘季節兩兩之間差異明顯' in txt_ and '冬、春兩季之分段分佈相近，其餘季節兩兩之間差異明顯' in cc
        if not ok1: vbad.append((nm, 'season'))
        if not all(r_['verdict_final'] == '相近' for r_ in rows if r_['kind'] == 'tide') or '大小潮兩組相近' not in GUIDE[(sk_, 'tid_c')]: vbad.append((nm, 'tide'))
        if not all(r_['verdict_final'] == '相近' for r_ in rows if r_['kind'] == 'class') or '十類之間相近' not in GUIDE[(sk_, 'cls_c')]: vbad.append((nm, 'class'))
        if not all(r_['verdict_final'] == '相近' for r_ in rows if r_['kind'] == 'velocity_within_class') or '同類十個沉降速度之間亦相近' not in GUIDE[(sk_, 'cls_c')]: vbad.append((nm, 'velocity'))
    s148 = next(t_ for t_ in got if t_.startswith('各研究地點十類之間'))
    chk('V11-C N14 季節、大小潮、十類（含十速度）之「相近／差異明顯」判定 = diff_verdicts_final.csv（龜山島僅冬–春相近；貢寮、新竹、南灣、連江四季兩兩差異明顯；大小潮、十類、同類十速度皆相近），導引段、整體結論與綜合說明一致；綜合說明依新順序（龜山島先）', not vbad and '季節之影響明顯：龜山島僅冬、春兩季相近，其餘季節兩兩之間差異明顯；貢寮、新竹、南灣與連江四季兩兩之間差異明顯' in s148, str(vbad[:4]))
    qbad = []; lbad = []; sbad = []; wbad = []
    for sk_ in SITEK:
        nm = ZH[sk_]
        for k_ in ('cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c'):
            g_ = GUIDE[(sk_, k_)]
            if re.search(r'°|其次|[②③]', g_): lbad.append((nm, k_))
        for k_, fk in (('sea_b', 'boundary_season'), ('tid_b', 'boundary_tide'), ('sea_c', 'coast_season'), ('tid_c', 'coast_tide')):
            nconc = GUIDE[(sk_, k_)].count('集中於'); ntop = sum(1 for t_ in MAN['figures'][sk_][fk]['panel_titles'] if '①' in t_)
            if nconc != ntop: qbad.append((nm, k_, nconc, ntop))
        for k_ in ('sea_b', 'tid_b'):
            for pos in re.findall(r'集中於([^（]+)（', GUIDE[(sk_, k_)]):
                if not re.fullmatch(r'(?:[東西南北]側邊界(?:[西中東南北]段)?|[東西][南北]角)', pos): wbad.append((nm, k_, pos))
        for k_ in ('sea_c', 'tid_c'):
            for pos in re.findall(r'集中於([^（]+)（', GUIDE[(sk_, k_)]):
                if not re.fullmatch(r'(?:研究地點(?:[東西南北]|東北|東南|西北|西南)側臺灣本島海岸|研究地點周邊(?:臺灣本島|恆春半島南端)?海岸|龜山島周圍海岸|馬祖北竿|馬祖南竿|福建沿岸(?:近岸島嶼)?|馬祖各島)', pos): wbad.append((nm, k_, pos))
        cc = concl[nm]; parts_ = cc.split('；')
        sp_ = [x_ for x_ in parts_ if re.search(r'季.*集中於|四季穿越皆集中於', x_) and '主要集中於' not in x_]
        if len(sp_) != 1 or re.search(r'\d', sp_[0]): sbad.append((nm, '季節定性句', [x_[:30] for x_ in parts_], sp_[:1])); continue
        if '°' not in parts_[0]: sbad.append((nm, '全年範圍'))
        bsr = {r_['segment_id']: r_ for r_ in rd('boundary_segments_%s.csv' % sk_)}
        SIDEN = {'N': '北側', 'S': '南側', 'E': '東側', 'W': '西側'}; CORN = {frozenset('NE'): '東北角', frozenset('NW'): '西北角', frozenset('SW'): '西南角', frozenset('SE'): '東南角'}
        for lab_, pn_ in zip('冬春夏秋', MAN['figures'][sk_]['boundary_season']['annotations']):
            st_ = [x_ for x_ in pn_['stretches'] if x_['marker'] == '①'][0]
            sides_ = {}
            for sid in st_['segments']:
                r_ = bsr[sid]; sd_ = r_['side']; v_ = [float(r_['start_lon']), float(r_['end_lon'])] if sd_ in 'NS' else [float(r_['start_lat']), float(r_['end_lat'])]
                sides_.setdefault(sd_, []).extend(v_)
            big_ = [k_ for k_, v_ in sides_.items() if max(v_) - min(v_) > 0.12]
            if len(big_) == 1 and len(sides_) > 1: sides_ = {big_[0]: sides_[big_[0]]}
            nmx = CORN[frozenset(''.join(sides_))] if len(sides_) == 2 else SIDEN[list(sides_)[0]] + '邊界'
            clause = next((x_ for x_ in re.split(r'，', sp_[0]) if re.search(r'^[冬春夏秋、]*%s' % lab_, x_) or re.search(r'[冬春夏秋、]*%s[冬春夏秋、]*季' % lab_, x_)), '')
            if not clause.endswith(nmx) and not ('四季穿越皆集中於' + nmx) in sp_[0]: sbad.append((nm, lab_, nmx, clause))
    chk('V11-C N15 季節、大小潮、十類導引段不寫經緯度、每格只寫第一名（無「其次」「②③」），「集中於」之數目 = 有括號之面板數', not lbad and not qbad, str(lbad[:3]) + str(qbad[:3]))
    chk('V11-C N16 季節、大小潮導引段所用位置詞限於：邊界（X 側邊界、X 側邊界西／中／東（南／北）段、四個角）、海岸（研究地點某側臺灣本島海岸、研究地點周邊恆春半島南端海岸、龜山島周圍海岸、馬祖北竿、馬祖南竿、福建沿岸、福建沿岸近岸島嶼、馬祖各島）；「龜山島沿岸」已不允許', not wbad, str(wbad[:4]))
    chk('V11-C N17 整體結論 (1)–(5) 之季節部分為定性（冬、春季穿越集中於…，夏、秋季集中於…），不含季節百分比，且保留全年穿越範圍（經緯度）；各季所屬邊界 = 該季最集中（①）括號所在邊界（由 boundary_segments 表獨立推得）', not sbad, str(sbad[:4]))
    import math as _m
    ANC_ = {'gongliao': (121.92, 25.10), 'guishan': (121.94, 24.85), 'hsinchu': (120.45, 24.75), 'nanwan': (120.77, 21.92), 'lienchiang': (119.95, 26.20)}
    def _two(sk, ids):
        cs_ = {r_['segment_id']: r_ for r_ in rd('coast_segments_%s.csv' % sk)}; ax, ay = ANC_[sk]
        km_ = lambda lo, la: ((lo - ax) * 111.32 * _m.cos(_m.radians(ay)), (la - ay) * 110.57)
        def dd(i):
            r = cs_[i]; A = km_(float(r['start_lon']), float(r['start_lat'])); B = km_(float(r['end_lon']), float(r['end_lat']))
            dx, dy = B[0] - A[0], B[1] - A[1]; L = dx * dx + dy * dy; t = max(0.0, min(1.0, -(A[0] * dx + A[1] * dy) / L)) if L else 0.0
            return _m.hypot(A[0] + t * dx, A[1] + t * dy)
        ids = sorted(ids); ring = ids[0].split('-')[0]; near = min([i for i in cs_ if i.startswith(ring + '-')], key=dd)
        if min(dd(i) for i in ids) > 10.0: return False
        e1 = km_(float(cs_[ids[0]]['start_lon']), float(cs_[ids[0]]['start_lat'])); e2 = km_(float(cs_[ids[-1]]['end_lon']), float(cs_[ids[-1]]['end_lat']))
        b1 = _m.degrees(_m.atan2(*e1)); b2 = _m.degrees(_m.atan2(*e2))
        return (near in ids and 0 < ids.index(near) < len(ids) - 1) or abs((b1 - b2 + 180) % 360 - 180) >= 150
    pbad = []; pchg = []
    for sk_ in SITEK:
        nm = ZH[sk_]
        for k_, fk in (('sea_c', 'coast_season'), ('tid_c', 'coast_tide')):
            f_ = MAN['figures'][sk_][fk]; anns_ = [a_ for a_ in (f_['annotations'] if isinstance(f_['annotations'], list) else [f_['annotations']]) if a_['stretches']]
            tops = [t_ for t_ in f_['panel_titles'] if '①' in t_]
            labs_ = re.findall(r'集中於([^（]+)（([\d.]+)%', GUIDE[(sk_, k_)])
            if len(anns_) != len(tops) or len(labs_) != len(tops): pbad.append((nm, k_, len(anns_), len(tops), len(labs_))); continue
            for a_, t_, (lab_, pc_) in zip(anns_, tops, labs_):
                st_ = [x_ for x_ in a_['stretches'] if x_['marker'] == '①'][0]
                if not (lab_.startswith('研究地點') and (lab_.endswith('側臺灣本島海岸') or lab_ == '研究地點周邊臺灣本島海岸')): continue
                exp_ = _two(sk_, st_['segments'])
                if exp_ != (lab_ == '研究地點周邊臺灣本島海岸'): pbad.append((nm, k_, lab_, pc_, exp_))
                if lab_ == '研究地點周邊臺灣本島海岸': pchg.append((nm, k_, pc_))
    chk('V11-C N19 季節、大小潮海岸括號若跨研究地點兩側（含最近海岸段且不在端點，或兩端夾角 ≥ 150°；10 km 內）則稱「研究地點周邊臺灣本島海岸」，否則用方位詞（獨立重算；周邊者 %d 處）' % len(pchg), not pbad, str(pbad[:4]))
    g_c = GUIDE[('guishan', 'all_c')]
    chk('V11-C N18 龜山島海岸來源子分項寫「臺灣本島占 11.96%，龜山島周圍海岸占 9.57%，其餘為其他島嶼」；同段 9.57% 只出現一次；全部導引段、圖說、導言、結論無 9.56%、0.13%；結論中 9.57% 各僅一處',
        '臺灣本島占 11.96%，龜山島周圍海岸占 9.57%，其餘為其他島嶼' in g_c and g_c.count('9.57%') == 1 and not [t_[:20] for t_ in texts_new if '9.56%' in t_ or '0.13%' in t_] and all(t_.count('9.57%') <= 1 for t_ in texts_new))
    # ---- 龜山島作為海岸來源之用語（使用者核定：「龜山島周圍海岸」）
    gs_all = [m_.group(0) for m_ in re.finditer('龜山島(?:周圍海岸|沿岸|本島沿岸)?', FT)]
    GSHAN_HITS = {'圖12': 0}
    chk('V11-C G2 龜山島作為海岸來源（非研究地點）一律寫「龜山島周圍海岸」：全文「龜山島沿岸」0 處、「龜山島占 x%」0 處；「龜山島周圍海岸」共 9 處（圖 12 導引段 2、圖 16 導引段 3、圖 18 導引段 2、貢寮圖 23 導引段 1、整體結論（1）1）；全部 word/*.xml 部件無「龜山島沿岸」',
        FT.count('龜山島沿岸') == 0 and not re.search(r'龜山島占 ?\d', FT) and FT.count('龜山島周圍海岸') == 9
        and FT.count('龜山島周圍海岸') == (GUIDE[('guishan', 'all_c')].count('龜山島周圍海岸') + GUIDE[('guishan', 'sea_c')].count('龜山島周圍海岸') + GUIDE[('guishan', 'tid_c')].count('龜山島周圍海岸') + GUIDE[('gongliao', 'all_c')].count('龜山島周圍海岸') + concl['龜山島'].count('龜山島周圍海岸'))
        and GUIDE[('guishan', 'all_c')].count('龜山島周圍海岸') == 2 and GUIDE[('guishan', 'sea_c')].count('龜山島周圍海岸') == 3 and GUIDE[('guishan', 'tid_c')].count('龜山島周圍海岸') == 2 and GUIDE[('gongliao', 'all_c')].count('龜山島周圍海岸') == 1 and concl['龜山島'].count('龜山島周圍海岸') == 1
        and not any('龜山島沿岸' in v_ for n_, v_ in allxml.items()),
        str((FT.count('龜山島沿岸'), FT.count('龜山島周圍海岸'), [GUIDE[('guishan', k_)].count('龜山島周圍海岸') for k_ in ('all_c', 'sea_c', 'tid_c')])))
    # ---- 殘留用語、地名白名單、待確認語
    resid = ['圓形', '來源方位', '扇區', '玫瑰', '方向機率', '主要來源方向', '來源方向', '方向區間', '正北', '順時針', '1/36', '2.78%', '十速度', 'BayTrace', 'Baytrace', 'baytrace', '反射', '本研究', '曲線之間']
    rb = [w_ for w_ in resid if w_ in FT] + (['360'] if re.search(r'(?<![\d.])360(?![\d.])', FT) else [])
    chk('V11-C T6 全文已無圓形來源方位圖用語（圓形、來源方位、扇區、玫瑰、方向機率、主要來源方向、來源方向、方向區間、正北、順時針、1/36、2.78%、360）、無「十速度」、無 BayTrace、無「反射」、無「本研究」、無「曲線之間」', not rb, str(rb))
    ng = [t_[:15] for t_ in got if re.search(r'^圖 ?\d+ ', t_) and re.search(r'十個沉降速度之曲線|十類各十個', t_)]
    chk('V11-C T7 十速度之圖與導引段已移除，僅保留「同類十個沉降速度之間亦相近」一句（五地點各 1 處）', not ng and sum(t_.count('同類十個沉降速度之間亦相近') for t_ in got) == 5, str(ng))
    PLC = {'龜山島', '龜山島周圍海岸', '臺灣本島', '福建沿岸', '馬祖北竿', '馬祖南竿', '馬祖各島', '馬祖列島', '恆春半島', '小琉球', '蘭嶼', '近岸島嶼', '其他島嶼', '各島'}
    found = set()
    for t_ in texts_new: found |= set(re.findall(r'[一-鿿]{0,3}(?:本島|各島|列島|半島|竿|沿岸|小琉球|蘭嶼|島嶼)', t_))
    unk = {x_ for x_ in found if not any(x_.endswith(p_) or p_.endswith(x_) or p_ in x_ for p_ in PLC)}
    gs_tok = set(re.findall(r'龜山島[一-鿿]{0,4}', '\n'.join(texts_new)))
    gs_bad = {x_ for x_ in gs_tok if any(b_ in x_ for b_ in ('沿岸', '西側', '東側', '海域', '附近', '周邊', '本島', '島西', '島東'))}
    chk('V11-C T8 導引段、圖說、導言、結論所用之地名限於白名單（龜山島周圍海岸〔已取代「龜山島沿岸」〕、臺灣本島、福建沿岸、馬祖北竿、馬祖南竿、馬祖各島、馬祖列島、恆春半島、小琉球、蘭嶼、近岸島嶼、其他島嶼、各島）；「龜山島」之後不接「沿岸」「西側」「海域」等其他寫法', not unk and not gs_bad and '龜山島沿岸' not in PLC, str(unk) + str(gs_bad))
    HEDGE = r'本研究|可能|或許|似乎|尚待|尚未|未知|不確定|應該|大約'
    hg = [t_[:24] for t_ in got if re.search(HEDGE, t_.replace('由於廢棄物實際何時沉底未知，', ''))]
    chk('V11-C T9 成果段（導引段、圖說、導言、結論全部）無「本研究」、無待確認／推測語（可能、或許、大約、似乎、尚待、尚未、未知、不確定、應該）；唯一例外為使用者核定之子句「由於廢棄物實際何時沉底未知，」（龜山島圖 15 導引段，逐字，全文恰 1 處）', not hg and FT.count('由於廢棄物實際何時沉底未知，') == 1 and '由於廢棄物實際何時沉底未知，' in GUIDE[('guishan', 'sea_b')], str(hg[:3]))
    chk('V11-C T10 每個來源分段導引段皆含結果句（百分比）', all(re.search(r'\d+\.\d\d%', GUIDE[(sk_, k_)]) for sk_ in SITEK for k_ in KEYS8))
    gb = GUIDE[('guishan', 'sea_b')]
    chk('V11-C W12 沉底時刻定義（逐字等於使用者核定之主檔文字，含「由於廢棄物實際何時沉底未知，」，其後「季節依沉底時刻所在月份劃分，冬季為 12–2 月…」）恰一處，位於龜山島之季節邊界導引段；貢寮等其餘地點之季節導引段不再重述；不含「分層」「種子」「定義見（一）3」',
        gb.startswith('圖 %d 為依沉底時刻劃分之龜山島四季邊界穿越來源分段機率圖。' % FN[('guishan', 'sea_b')]) and SEADEF_M in gb and FT.count(SEADEF_M) == 1 and all(SEADEF_M not in GUIDE[(sk_, 'sea_b')] for sk_ in SITEK if sk_ != 'guishan') and '分層' not in gb and '種子' not in gb and '定義見（一）3' not in FT)
    fig_intro = [GUIDE[(sk_, r_)] for sk_ in SITEK for r_ in ROLES[1:]]
    chk('V11-C W15 （二）55 張圖之導引段皆以「圖 N」起首並含結果句（百分比、「相近」或「一致」）', len(fig_intro) == 55 and all(t_.startswith('圖 ') and re.search(r'\d+(\.\d+)?%|相近|一致', t_) for t_ in fig_intro))
    trj = []
    for sk_ in SITEK:
        t_ = GUIDE[(sk_, 'traj')]; nums = [FN[(sk_, 'all_b')]] + ([] if sk_ == 'hsinchu' else [FN[(sk_, 'all_c')]])
        trj.append(t_.endswith('與圖 %s 之主要來源一致。' % ' 及圖 '.join(str(x_) for x_ in nums)) and not re.search(r'\d+ 條中|其中 \d+ 條', t_) and '延伸' in t_ and '一致' in t_)
    chk('V11-C W16 五張軌跡圖導引段各有一句結果（延伸／穿越方向、抵達海岸），並呼應各地點全年邊界與海岸來源分段機率圖之主要來源（新竹僅邊界圖）；圖號取自辨識結果；不含條數', all(trj), str(trj))
    # ---- 圖說（V10-6/7/8）
    CAPNEW = {}
    for sk_ in SITEK:
        for k_ in KEYS8: CAPNEW[(sk_, k_)] = next(c_['new'] for c_ in LOG['changes'] if c_.get('fig') == FMlog[(sk_, k_)]['n'] and c_['kind'] == 'new' and c_['role'].startswith('caption:'))
    capbad = []; posw = []; oldbad = []
    for sk_ in SITEK:
        for k_ in KEYS8:
            t_ = CAPT[(sk_, k_)]; n_ = FN[(sk_, k_)]
            want = '圖%d %s' % (n_, FMlog[(sk_, k_)]['caption'].replace('新竹外海', '新竹'))
            if t_ != want: capbad.append((ZH[sk_], k_, '與建置紀錄不同'))
            if re.sub(r'^圖\d+', '', t_) != re.sub(r'^圖\d+', '', CAPNEW[(sk_, k_)]).replace('新竹外海', '新竹'): capbad.append((ZH[sk_], k_, '與 v10 圖說不同'))
            if not rpr_has(kc[FIGAT[(sk_, k_)][0] + 1], 'keepLines'): capbad.append((ZH[sk_], k_, 'keepLines'))
            if ZH[sk_] not in t_: capbad.append((ZH[sk_], k_, '地點名'))
            if re.search(r'數值框|[左右][上下]|角落|各格標題|其後數值|標題', t_): posw.append((ZH[sk_], k_, re.findall(r'數值框|[左右][上下]|角落|各格標題|其後數值|標題', t_)))
            if re.search(THR, t_): capbad.append((ZH[sk_], k_, '門檻'))
            o_ = FMlog[(sk_, k_)].get('v9_caption')
            if o_ is not None and re.findall(r'\d+(?:\.\d+)?', o_) != re.findall(r'\d+(?:\.\d+)?', FMlog[(sk_, k_)]['caption']): oldbad.append((ZH[sk_], k_))
    chk('V11-C V10-6 40 個圖說：文字 ＝ 建置紀錄（圖號 ＋ 新句，地點名依新名稱）、保留 keepLines、含研究地點名、不寫門檻值；與 v9 圖說相比，數字（10 km、5 km 等）完全相同', not capbad and not oldbad, str(capbad[:4]) + str(oldbad[:3]))
    allp = [txt(p_) for p_ in kc if isp(p_)] + [txt(p_) for t_ in kc if t_.tag == q('tbl') for p_ in t_.iter(q('p'))]
    chk('V11-C V10-7 全文（含表格）已無「數值框」「右上／右下／左上／左下」位置詞（圖說）與「各格標題」「其後數值為該段機率」之殘留；40 個圖說無位置詞', not posw and not [t_ for t_ in allp if re.search(r'數值框|各格標題|其後數值為該段機率', t_)], str(posw[:3]))
    rbad = []; n_unnum = {}
    for sk_ in SITEK:
        for k_ in KEYS8:
            t_ = CAPT[(sk_, k_)]; n_ = FN[(sk_, k_)]
            if k_ == 'all_b':
                if not (t_.endswith('，括號旁數值為該段之機率') and '黑色括號標示機率集中之連續邊界段' in t_ and '灰色括號標示其餘連續非零邊界段' in t_): rbad.append((n_, '全年邊界'))
                continue
            if k_ == 'all_c':
                if t_ != '圖%d %s' % (n_, FMlog[(sk_, k_)]['v9_caption'].replace('新竹外海', '新竹')) or '引線旁數值為機率集中之海岸段之機率' not in t_: rbad.append((n_, '全年海岸（應與 v9 圖說相同）'))
                continue
            mf_ = json.load(open(os.path.join(FINAL9, sk_, 'manifest.json'), encoding='utf-8'))[FIGK[KEYS8.index(k_)]]
            anns_ = mf_['annotations']; anns_ = [anns_] if isinstance(anns_, dict) else anns_
            if k_ in ('cls_b', 'sea_b', 'tid_b'):
                pre = '各格內數值依序為' + ('該類之' if k_ == 'cls_b' else '') + '穿越比例、各括號段之機率（圈號）與其餘'
                unnum = any(st_.get('marker', '') == '' for a_ in anns_ for st_ in a_['stretches']); n_unnum[(sk_, k_)] = unnum
                tail = '「其餘」為未編號邊界段（含未編號之括號段）之機率合計' if unnum else '「其餘」為未編號邊界段之機率合計'
                if not (pre in t_ and t_.endswith(tail) and '括號標示機率集中之連續邊界段' in t_ and '圓圈數字依機率由大至小編號' in t_): rbad.append((n_, '邊界', unnum))
                if (not unnum) and '未編號之括號段' in t_: rbad.append((n_, '不應有未編號括號段'))
            if k_ == 'cls_c':
                if not (t_.endswith('各格內數值為該類之海岸來源比例') and '圓圈' not in t_ and '括號' not in t_): rbad.append((n_, '十類海岸'))
            if k_ in ('sea_c', 'tid_c'):
                has_ = [any(re.search(r'[①-⑳]', ln_) for ln_ in bl_) for bl_ in mf_['box_lines']]; nP = len(has_)
                if sum(has_) == 0:
                    if not (t_.endswith('各格內數值為海岸來源比例') and '圓圈' not in t_): rbad.append((n_, '海岸無圈號'))
                elif sum(has_) == nP:
                    if not ('圓圈數字依機率由大至小標示機率集中之海岸段' in t_ and t_.endswith('各格內數值依序為海岸來源比例及各圈號海岸段之機率')): rbad.append((n_, '海岸每格皆有圈號'))
                else:
                    names_ = [mf_['panel_titles'][j_][:2] for j_, h_ in enumerate(has_) if h_]
                    if not (('僅%s格有機率集中之海岸段，以圓圈數字標示' % '、'.join(names_)) in t_ and t_.endswith('各格內數值為海岸來源比例，%s格另列圈號海岸段之機率' % '、'.join(names_))): rbad.append((n_, '海岸部分格有圈號', names_, has_))
    UNN = sorted(k_ for k_, u_ in n_unnum.items() if u_)
    chk('V11-C V10-8 40 個圖說之用語與圖面實際內容相符（依 final manifest）：全年邊界「括號旁數值為該段之機率」；十類、季節、大小潮邊界「各格內數值依序為…穿越比例、各括號段之機率（圈號）與其餘」，有未編號括號段者另註「含未編號之括號段」——依圖類型為鍵，僅貢寮與南灣之季節邊界圖（圖 %s）；海岸圖依各格實有圈號者寫' % '、'.join('%d' % FN[k_] for k_ in UNN),
        not rbad and UNN == [('gongliao', 'sea_b'), ('nanwan', 'sea_b')], str(rbad[:4]) + str(UNN))
    # ---- 曲線圖文字（V10-13）與圖說 keepLines
    FIG10 = {}
    for i_, e in enumerate(k10):
        if isp(e):
            for f_ in figs_in(e):
                k_ = IDM.get(sha(z10.read('word/' + rl10[f_[2]])))
                if k_: FIG10[k_] = i_
    tbad = []
    for sk_ in SITEK:
        nm = ZH[sk_]; g_ = GUIDE[(sk_, 'curve')]; n_ = FN[(sk_, 'curve')]; cp_ = CAPT[(sk_, 'curve')]; go_ = txt(k10[FIG10[(sk_, 'curve')] - 1])
        slow_ = sk_ in ('gongliao', 'guishan', 'nanwan')
        if not g_.startswith('圖 %d 為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。' % n_): tbad.append((nm, '首句'))
        if '全部 100,000 顆虛擬質點合計，%s仍在追蹤之比例於回溯第' % nm not in g_ or '為全部 100,000' in g_: tbad.append((nm, '前綴'))
        stripn = lambda s_: re.findall(r'\d+(?:\.\d+)?%?', re.sub(r'^圖 \d+', '', s_).replace('每類 10,000 顆', '').replace('全部 100,000 顆', ''))
        if stripn(g_) != stripn(go_): tbad.append((nm, '數字', stripn(g_), stripn(go_)))
        if not g_.endswith('降至 0%。十類曲線相近，其中不可回收塑膠略慢。' if slow_ else '降至 0%。十類曲線相近。'): tbad.append((nm, '結果句', slow_))
        if '小圖' in g_ or '全部類別' in g_ or '黑' in g_: tbad.append((nm, '導引段不應提小圖或全部線'))
        if cp_ != '圖%d %s回溯期間仍在追蹤之虛擬質點比例（十類）；小圖為回溯前 20 天之放大' % (n_, nm): tbad.append((nm, '圖說'))
        if rpr_has(kc[FIGAT[(sk_, 'curve')][0] + 1], 'keepLines') != rpr_has(k10[FIG10[(sk_, 'curve')] + 1], 'keepLines'): tbad.append((nm, '曲線圖說 keepLines 與 v10 候選不同'))
    b34 = next((t_ for t_ in got if '各研究地點依序以十一張圖說明' in t_), '')
    chk('V11-C V10-13 5 張仍在追蹤比例圖之文字（依地點與圖類型）：導引段首句「圖 N 為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。」；50%／10%／1% 句加前綴「全部 100,000 顆虛擬質點合計，」、數字與 v10 候選相同；句末「十類曲線相近」（貢寮、龜山島、南灣另加「，其中不可回收塑膠略慢」；新竹、連江不加）；導引段不提小圖；圖說加「（十類）；小圖為回溯前 20 天之放大」；導言改為「呈現十類虛擬質點各自於回溯期間內仍在追蹤之比例」',
        not tbad and '呈現十類虛擬質點各自於回溯期間內仍在追蹤之比例' in b34 and '全部虛擬質點於回溯期間內仍在追蹤' not in b34, str(tbad[:4]))
    n_caps_kl = {(k_): rpr_has(kc[FIGAT[k_][0] + 1], 'keepLines') for k_ in FIGAT}
    chk('V11-C K7 圖說 keepLines：40 個來源分段圖說與 5 個仍在追蹤比例圖說（共 %d 個）皆有 keepLines，與 v10 候選逐圖（同圖檔）一致；其餘圖說不變' % sum(n_caps_kl.values()),
        all(n_caps_kl[(sk_, k_)] for sk_ in SITEK for k_ in KEYS8) and all(rpr_has(kc[FIGAT[k_][0] + 1], 'keepLines') == rpr_has(k10[FIG10[k_] + 1], 'keepLines') for k_ in FIGAT if k_ in FIG10), str(sorted(k_ for k_, v_ in n_caps_kl.items() if v_ and k_[1] not in KEYS8)))
    # ---- 表 1 後第一段（SENT_E）、重壓扁、到達時間導引段
    i_t1 = next(n_ for n_, e in enumerate(kc) if e.tag == q('tbl'))
    t6 = txt(kc[i_t1 + 1]); t4 = txt(kc[i_t1 - 1])
    chk('V11-C V10-12 表 1 後第一段末為使用者核定之一句，逐字：「各類 10 個速度之跨距較大，表 1 以科學記號列其範圍；模擬時 10 個速度各自直接使用，不取平均值或對數平均。」，全文恰 1 處；原有文字（受力平衡求解、詳見附錄）在前；表 1 標題不變',
        FT.count(SENT_E) == 1 and t6.endswith(SENT_E) and t6.startswith('各類 10 個速度係以受力平衡求解，完整推導過程、代表物件之尺寸與輸入參數、實測量級對照與敏感度分析詳見附錄。各類 10 個速度之跨距較大') and t4 == '表1 十類海底廢棄物之代表形狀與終端沉降速度', t6[-60:])
    n_ya = sum(txt(p_).count('重壓扁') for p_ in kc if isp(p_)) + sum(txt(p_).count('重壓扁') for t_ in kc if t_.tag == q('tbl') for p_ in t_.iter(q('p')))
    n_blk = sum(1 for e in kc if '重壓扁' in ''.join(t_.text or '' for t_ in e.iter(q('t'))))
    chk('V11-C V10-H1 使用者手改（全文「嚴重壓扁」→「重壓扁」）：已無「嚴重壓扁」，「重壓扁」共 %d 處（18 處）、分布於 %d 個區塊（11 個，含首段與表 1）' % (n_ya, n_blk), not any('嚴重壓扁' in t_ for t_ in allp) and n_ya == 18 and n_blk == 11 and '重壓扁' in ''.join(t_.text or '' for t_ in t1.iter(q('t'))), str((n_ya, n_blk)))
    t20 = next(t_ for t_ in alltext if t_.startswith('以下列出各研究地點之到達時間驗證圖'))
    chk('V11-C V10-H2 到達時間驗證圖導引段已刪「每個研究地點 1 張全年總覽圖，」，逐字：「以下列出各研究地點之到達時間驗證圖，龜山島見圖 3，貢寮見圖 4，新竹見圖 5，南灣見圖 6，連江見圖 7。…」（圖號 = 5 張驗證圖之辨識結果）',
        t20.startswith('以下列出各研究地點之到達時間驗證圖，' + '，'.join('%s見圖 %d' % (n_, FN[(SKOF[n_], 'arrive')]) for n_ in SITES11) + '。') and '全年總覽' not in t20)
    # ---- 參考資料（V10-R）
    ri1 = next(i_ for i_ in range(ri0 + 1, len(kc)) if isp(kc[i_]) and txt(kc[i_]).startswith('附錄　'))
    refs_ = [txt(kc[i_]) for i_ in range(ri0 + 1, ri1)]
    def _k(t):
        nm_ = unicodedata.normalize('NFKD', t.split(',')[0] if ', ' in t.split('(')[0] and not re.match(r'^[A-Z][A-Za-z ]+\. \(', t) else t.split('. (')[0]).lower()
        nm_ = ''.join(c_ for c_ in nm_ if c_.isalpha() and c_.isascii())
        yr_ = re.search(r'\((\d{4})\)', t).group(1)
        return (nm_, yr_)
    zh_ = [t_ for t_ in refs_ if re.search(r'[一-鿿]', t_[:6])]; en_ = [t_ for t_ in refs_ if t_ not in zh_]
    order_ok = refs_[:len(zh_)] == zh_ and [_k(t_) for t_ in en_] == sorted(_k(t_) for t_ in en_)
    CITE = ['海洋委員會海洋保育署, 2025', 'Bagheri & Bonadonna, 2016', 'Cambridge University Engineering Department, 2003', 'Çengel & Cimbala, 2018', 'Duis & Coors, 2016', 'Ferguson & Church, 2004', 'Forest Products Laboratory, 2010', 'GESAMP, 2015', 'GESAMP, 2019', 'Goral et al., 2023', 'Ioakeimidis et al., 2014', 'ITTC, 2011', 'Kooi et al., 2017', 'Liu et al., 2022', 'MSFD Technical Subgroup on Marine Litter, 2013', 'Nikolaev, 2022', 'Prado & Dremière, 1990', 'Thompson & Taylor, 2008', 'Xu et al., 2024']
    body_ = '\n'.join(txt(p_) for i_, p_ in enumerate(kc) if isp(p_) and not (ri0 <= i_ < ri1)) + '\n' + '\n'.join(txt(p_) for t_ in kc if t_.tag == q('tbl') for p_ in t_.iter(q('p')))
    miss_ = [c_ for c_ in CITE if c_ not in body_]
    pair_ = [(re.match(r'^(.*?)[,(]', t_).group(1).strip(' .'), re.search(r'\((\d{4})\)', t_).group(1)) for t_ in refs_]
    listed_ = {'海洋委員會海洋保育署2025', 'Bagheri2016', 'Cambridge University Engineering Department2003', 'Çengel2018', 'Duis2016', 'Ferguson2004', 'Forest Products Laboratory2010', 'GESAMP2015', 'GESAMP2019', 'Goral2023', 'Ioakeimidis2014', 'ITTC2011', 'Kooi2017', 'Liu2022', 'MSFD Technical Subgroup on Marine Litter2013', 'Nikolaev2022', 'Prado1990', 'Thompson2008', 'Xu2024'}
    got_ = {a_ + b_ for a_, b_ in pair_}
    num_ok = all(kc[i_].find(q('pPr')).find(q('numPr')) is not None for i_ in range(ri0 + 1, ri1))
    sameas = all(sg(kc[i_], zc, rlc) == sg(k10[i_ - (rf_c - rf_10)], z10, rl10) for i_ in range(ri0, ri1))
    chk('V11-C V10-R 參考資料（依期中報告用法）：共 %d 筆，中文 %d 筆置前、英文 %d 筆依第一作者姓氏／機構名 A→Z（同一作者依年份），已編號且懸掛縮排；內文引用為作者—年份括號式（19 種）且逐一對應清單、清單每筆皆在本文或附錄被引用；參考資料段與 v10 候選逐字、逐格式相同' % (len(refs_), len(zh_), len(en_)),
        len(refs_) == 19 and len(zh_) == 1 and order_ok and not miss_ and got_ == listed_ and num_ok and sameas, str((order_ok, miss_, got_ ^ listed_, num_ok, sameas)))

    # ================= V11-PDF（若提供 PDF）
    if PDF:
        def pgs(pdf):
            n = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout).group(1))
            return [re.sub(r'\s+', '', subprocess.run(['pdftotext', '-f', str(p), '-l', str(p), pdf, '-'], capture_output=True, text=True).stdout) for p in range(1, n + 1)]
        Pg = pgs(PDF)
        il = [int(l.split()[0]) for l in subprocess.run(['pdfimages', '-list', PDF], capture_output=True, text=True).stdout.splitlines()[2:] if l.split()[2] == 'image']
        sep = []
        for i, n, c in seq:
            if n is None or n < 8: continue
            cp = [k + 1 for k, t in enumerate(Pg) if re.sub(r'\s+', '', c)[:20] in t]
            if cp != [il[n - 1]]: sep.append((n, il[n - 1], cp))
        chk('V11-PDF LibreOffice：%d 頁（主檔 92 頁）；62 張圖各 1 個；圖 8–62 之圖說皆與圖在同一頁（無圖與圖說分離）' % len(Pg), len(Pg) == 92 and len(il) == 62 and not sep, str(sep[:5]))

    if PDF:
        npg = len(Pg)
        def pg_of(s_, start=1):
            for p_ in range(start, npg + 1):
                if s_ in subprocess.run(['pdftotext', '-f', str(p_), '-l', str(p_), '-layout', PDF, '-'], capture_output=True, text=True).stdout.replace(' ', ''): return p_
        pa, pb, pc, pd = pg_of('（二）分析成果'), pg_of('整體結論：'), pg_of('參考資料', 40), pg_of('附錄十類海底廢棄物', 40)
        chk('V11-C D1 PDF 頁序：（二）< 整體結論 < 參考資料 < 附錄（%s<%s<%s<%s，共 %d 頁）' % (pa, pb, pc, pd, npg), None not in (pa, pb, pc, pd) and pa < pb < pc < pd)
        txtp = subprocess.run(['pdftotext', '-layout', PDF, '-'], capture_output=True, text=True).stdout
        _p13 = subprocess.run(['pdftotext', '-f', '1', '-l', '3', '-layout', PDF, '-'], capture_output=True, text=True).stdout
        _pdfr = _p13.split('\n')
        chk('V11-C D4 PDF 表 1 速度欄每列兩行——「a.bc×10⁻ⁿ」共 20 處、其中「至 a.bc×10⁻ⁿ」10 處（第二行），無單獨成行之「至」',
            len(re.findall(r'(?<![\d.])\d\.\d{2}×10−\d', _p13)) == 20 and len(re.findall(r'至\s+\d\.\d{2}×10−\d', _p13)) == 10 and not any(l_.strip() == '至' for l_ in _pdfr))
        chk('V11-C D2 PDF 圖說 圖1–圖62 皆出現', all(re.search(r'圖 ?%d ' % n_, txtp) for n_ in range(1, 63)))
        st4 = txtp.replace('\n', '').replace(' ', '')
        chk('V11-C D3 PDF 中停止邊界段結尾為「即停止追蹤。」且其後無「33.4」', '即停止追蹤。' in st4 and '即停止追蹤。33.4' not in st4)
        chk('V11-C D5 PDF 全文無「龜山島沿岸」「新竹外海」', '龜山島沿岸' not in st4 and '新竹外海' not in st4)

    if V12:
        mp12 = find_by_sha('9808fcad1dfc285f'); mp11c = find_by_sha('54fd3bcdb6fdcf43')
        chk('V12-A 比對基準檔存在：v11 升格後使用者再存之主檔備份（sha 9808fcad…）與 v11 候選（sha 54fd3bcd…）', bool(mp12) and bool(mp11c), str((mp12, mp11c)))
        z12, r12, k12, rl12 = load(mp12); z11, r11, k11, rl11 = load(mp11c)
        s11 = [sg(e, z11, rl11) for e in k11]; s12 = [sg(e, z12, rl12) for e in k12]
        d12 = [i for i in range(len(k11)) if s11[i] != s12[i]] if len(k11) == len(k12) else None
        RP_NEW = '<w:rPr><w:rFonts/><w:szCs w:val="28"/></w:rPr>'
        ok_t = d12 == [0, 5] and s12[0][1] == s11[0][1] and s12[0][2] == s11[0][2] + tuple((RP_NEW, ch_) for ch_ in '分析成果') and s12[0][3:] == s11[0][3:]
        def cell_sig(tc_):
            out_ = []
            for p_ in tc_.iter(q('p')):
                ps_ = p_.find(q('pPr')); out_.append(('P', cl2(ps_) if ps_ is not None else ''))
                for r_ in p_.iter(q('r')):
                    rp_ = r_.find(q('rPr')); rs_ = cl2(rp_) if rp_ is not None else ''
                    for t_ in r_.iter(q('t')):
                        for ch_ in (t_.text or ''): out_.append((rs_, ch_))
            return out_
        c11 = list(k11[5].iter(q('tc'))); c12 = list(k12[5].iter(q('tc')))
        CELLCHG = {14: ('吸水或附著之發泡板、箱體側板', '生物附著之發泡板、箱體'), 17: ('尼龍、聚乙烯或混合漁具材料', '尼龍、聚乙烯或混合漁具'), 30: ('硬質塑膠碎片、塑膠籃側板、橡膠塊', '硬質塑膠碎片、塑膠籃、橡膠塊')}
        chg_c = [j_ for j_ in range(len(c11)) if cell_sig(c11[j_]) != cell_sig(c12[j_])] if len(c11) == len(c12) == 44 else None
        ok_c = chg_c == sorted(CELLCHG) and all(txt(c11[j_]) == CELLCHG[j_][0] and txt(c12[j_]) == CELLCHG[j_][1] and {x_[0] for x_ in cell_sig(c11[j_]) if len(x_) == 2 and x_[0] != 'P'} == {x_[0] for x_ in cell_sig(c12[j_]) if len(x_) == 2 and x_[0] != 'P'} for j_ in CELLCHG)
        ok_c = ok_c and cl2(k11[5].find(q('tblPr'))) == cl2(k12[5].find(q('tblPr'))) and cl2(k11[5].find(q('tblGrid'))) == cl2(k12[5].find(q('tblGrid'))) and len(k11[5].findall(q('tr'))) == len(k12[5].findall(q('tr')))
        zb_n = z12.namelist(); zc_n = z11.namelist()
        media_h = lambda z_: sorted(sha(z_.read(n_)) for n_ in z_.namelist() if n_.startswith('word/media/'))
        othr = [n_ for n_ in zb_n if not n_.startswith('word/media/') and n_ not in ('word/document.xml', 'word/settings.xml', 'docProps/core.xml', 'docProps/app.xml')]
        ok_z = sorted(zb_n) == sorted(zc_n) and media_h(z11) == media_h(z12) and all(z11.read(n_) == z12.read(n_) for n_ in othr)
        chk('V12-0 使用者於 v11 升格後之手改（登記，v11 候選 → 主檔 9808fcad…）共 4 處：(1) 文件標題「Lagrangian系集逆向溯源」→「Lagrangian系集逆向溯源分析成果」；(2) 表 1 保麗龍列代表形狀「吸水或附著之發泡板、箱體側板」→「生物附著之發泡板、箱體」；(3) 表 1 廢漁網列材質「尼龍、聚乙烯或混合漁具材料」→「尼龍、聚乙烯或混合漁具」；(4) 表 1 可回收塑膠列代表形狀「硬質塑膠碎片、塑膠籃側板、橡膠塊」→「硬質塑膠碎片、塑膠籃、橡膠塊」；其餘 355 個區塊與表 1 其餘 41 格逐字、逐格式、逐圖片（sha）相同；zip 僅差媒體檔重新編號（%d 個圖檔內容 sha 多重集合相同）與 settings、docProps' % len(media_h(z12)),
            ok_t and ok_c and ok_z, str((d12, chg_c, ok_t, ok_c, ok_z)))
        # v12 候選 vs 主檔 9808fcad…：僅 document.xml 之一處
        xml_c = zc.read('word/document.xml'); xml_b = z12.read('word/document.xml')
        ctx_o = '<w:t>海床高程</w:t>'.encode('utf-8'); ctx_n = '<w:t>該處地形水深</w:t>'.encode('utf-8')
        ok_x = xml_b.count(ctx_o) == 1 and xml_c == xml_b.replace(ctx_o, ctx_n) and xml_c.count(ctx_n) == 1
        ok_z2 = sorted(zc.namelist()) == sorted(z12.namelist()) and all(zc.read(n_) == z12.read(n_) for n_ in z12.namelist() if n_ != 'word/document.xml')
        chk('V12-1 v12 候選相對主檔 9808fcad…：除 word/document.xml 外，其餘 %d 個 zip 成員逐位元組相同；document.xml 之原始位元組恰差一處，即單一 run 內 <w:t>海床高程</w:t> → <w:t>該處地形水深</w:t>（rsid、run 格式、其餘位元組不動）' % (len(z12.namelist()) - 1), ok_x and ok_z2, str((ok_x, ok_z2)))
        sc12 = [sg(e, zc, rlc) for e in kc]
        d_c = [i for i in range(357) if sc12[i] != s12[i]]
        b10b, b10c = txt(k12[10]), txt(kc[10])
        chk('V12-2 區塊層級：與主檔 9808fcad… 相比，唯一不同的區塊是（一）2 受體選取段；該段文字 = 原文將「潮位與海床高程換算」改為「潮位與該處地形水深換算」，字元格式（rPr）逐字對應相同、pPr 與分頁符相同', d_c == [10] and b10c == b10b.replace('潮位與海床高程換算', '潮位與該處地形水深換算') and b10b.count('潮位與海床高程換算') == 1
            and {r_ for r_, _ in sc12[10][2]} == {r_ for r_, _ in s12[10][2]} and len(set(r_ for r_, _ in sc12[10][2])) == 1 and sc12[10][1] == s12[10][1] and sc12[10][3:] == s12[10][3:], str(d_c))
        alltx = {n_: ''.join(etree.fromstring(zc.read(n_)).itertext()) for n_ in zc.namelist() if n_.endswith('.xml')}
        sent = '實際深度再依各到達時間的潮位與該處地形水深換算，因此同一受體在不同時刻的深度會隨潮位而異。'
        chk('V12-3 「海床高程」全部 XML 部件共 0 處；「該處地形水深」全文 1 處且在（一）2；整句逐字為「…實際深度再依各到達時間的潮位與該處地形水深換算，因此同一受體在不同時刻的深度會隨潮位而異。」；該句仍保留「潮位」2 處；「海床高程」「高程」於全文（含表、圖說、附錄、替代文字）0 處',
            sum(v_.count('海床高程') for v_ in alltx.values()) == 0 and FT.count('該處地形水深') == 1 and '該處地形水深' in b10c and sent in b10c and b10c.count('潮位') == 2 and '高程' not in FT
            and not any('高程' in (m_ or '') for m_ in re.findall(r'descr="([^"]*)"', zc.read('word/document.xml').decode())), str((FT.count('該處地形水深'), b10c.count('潮位'))))
    # ---- 報告
    shutil.rmtree(tmpd0, ignore_errors=True)
    fails = [r for r in res if not r[0]]
    out = ['%s %s%s' % ('OK  ' if ok else 'FAIL', n, ('  -- ' + d) if (not ok and d) else '') for ok, n, d in res]
    out += ['', '檢查對象：%s（sha256 %s）' % (os.path.basename(MERGED), sha(open(MERGED, 'rb').read())), '比對基準：現行主檔 %s（sha256 %s）' % (os.path.basename(mp), sha(open(mp, 'rb').read())),
            '%s：%d 項，失敗 %d' % ('全部通過' if not fails else '有失敗', len(res), len(fails))]
    open(os.path.join(HERE, 'check_merged_run09.report.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('\n'.join(out))
    sys.exit(1 if fails else 0)


tmpd0 = tempfile.mkdtemp()
if V11: v11_main()


# ---------------------------------------------------------------- 來源與基準
chk('S1 來源 3-4-3 未被修改（sha 同）', sha(open(SRC_A, 'rb').read()) == SHA_A)
chk('S2 來源執行成果未被修改（sha 同）', sha(open(SRC_B, 'rb').read()) == SHA_B)
zm, rootM, kids, relsM = load(MERGED)
za, rootA, kidsA, relsA = load(SRC_A)
zb, rootB, kidsB0, relsB = load(SRC_B)
kidsB = [e for e in kidsB0 if e.tag == q('p')]
tmpd = tempfile.mkdtemp(); v2p = os.path.join(tmpd, 'v2.docx')
keep_backup = {f: open(os.path.join(HERE, f), 'rb').read() for f in ('merge_trace.json', 'keep_plan.json')}
subprocess.run([sys.executable, os.path.join(HERE, 'merge_run09.py'), v2p], capture_output=True, check=True)
for f, d in keep_backup.items(): open(os.path.join(HERE, f), 'wb').write(d)
chk('S3 以 merge_run09.py 重建之 v2（比對基準）sha = 8db2951b…（可重現）', sha(open(v2p, 'rb').read()) == SHA_V2)
z2, root2, kids2, rels2 = load(v2p)

# ---------------------------------------------------------------- 套件
chk('P1 zip 完整（testzip）', zm.testzip() is None)
bad = []
for n in zm.namelist():
    if n.endswith('.xml') or n.endswith('.rels'):
        try: etree.fromstring(zm.read(n))
        except Exception: bad.append(n)
chk('P2 全部 XML 可解析', not bad, str(bad))
chk('P3 styles.xml、numbering.xml、theme、fontTable 與 3-4-3 逐位元組相同（版面定義未動）',
    all(zm.read(n) == za.read(n) for n in ('word/styles.xml', 'word/numbering.xml', 'word/theme/theme1.xml', 'word/fontTable.xml')))
ct = zm.read('[Content_Types].xml').decode()
chk('P5 Content_Types 含 jpg 與 png', 'jpg' in ct and 'png' in ct)
allfig = [(p, f) for p in kids if p.tag == q('p') for f in figs_in(p)]
ids = [f[3] for _, f in allfig]
chk_old('P9 docPr id 全文唯一（42 張圖）', len(ids) == len(set(ids)) == 42)
chk('P10 全部 r:embed 指向存在之關係與檔案', all(f[2] in relsM and ('word/' + relsM[f[2]]) in zm.namelist() for _, f in allfig))
chk_old('P11 42 張圖各引用一個圖檔，無共用', len({relsM[f[2]] for _, f in allfig}) == 42)
try:
    import docx; docx.Document(MERGED); ok = True
except Exception: ok = False
chk('P13 python-docx 可開啟', ok)

# ---------------------------------------------------------------- 與 v2 之差異僅限已登記之手改與 v3 修改
V8 = len(kids) != len(kids2)       # v8：成果段以來源分段機率圖取代圓形來源方位圖（359 區塊）；主檔（299 區塊）= v7 加使用者手改
DELTA = 60 if V8 else 0            # 候選與主檔之區塊數差：移除 61、新增 121
MASTER_PATH = os.path.join(BASE, '成果報告_run09_合併版.docx')
_B17 = os.path.join(BASE, '成果報告_run09_合併版.docx.bak17-20261028')
if os.path.exists(_B17): MASTER_PATH = _B17   # v8 升格後：比對基準為升格前之主檔（bak17，sha 45f8d015…）
if V10:
    import glob
    _c10 = [os.path.join(BASE, '成果報告_run09_合併版.docx')] + sorted(glob.glob(os.path.join(BASE, '成果報告_run09_合併版.docx.bak*')))
    M10P = next((p_ for p_ in _c10 if os.path.exists(p_) and sha(open(p_, 'rb').read()).startswith('20d8370880154b53')), None)
    zb10, rootb10, kb10, relsb10 = load(M10P) if M10P else (None, None, [], {})
if V8:
    zmast, rootmast, kmast, relsmast = load(MASTER_PATH)
    chk('M-1 v8：主檔 成果報告_run09_合併版.docx 未被動過（sha256 以 45f8d015914eeb6d 起首、區塊數 299）', sha(open(MASTER_PATH, 'rb').read()).startswith('45f8d015914eeb6d') and len(kmast) == 299 and len(kids) == 299 + DELTA, '%d %d' % (len(kmast), len(kids)))
else:
    zmast, kmast, relsmast = zm, kids, relsM
sg2 = [sig(e, z2, rels2) for e in kids2]; sgM = [sig(e, zmast, relsmast) for e in kmast]
chk('M0 區塊數同 v2（%d）（v8：主檔 299 區塊同 v2；候選 = 主檔 + %d）' % (len(kids2), DELTA), len(kmast) == len(kids2), '%d vs %d' % (len(kmast), len(kids2)))
def mod_chars(chars, old, new):
    s = ''.join(c for _, c in chars); i = s.index(old); assert s.count(old) == 1, old
    rs = chars[i][0] if chars else ''
    return tuple(chars[:i]) + tuple((rs, c) for c in new) + tuple(chars[i + len(old):])
exp = list(sg2)
txt2 = [ptext(e) if e.tag == q('p') else '' for e in kids2]
i_item1 = next(n for n, t in enumerate(txt2) if t.startswith('1、沉降動力參數'))
i_para3 = next(n for n, t in enumerate(txt2) if t.startswith('十種項目對應'))
i_item4 = next(n for n, t in enumerate(txt2) if t.startswith('4、溯源停止邊界'))
HAND_IND = [n for n, t in enumerate(txt2) if re.match(r'^（[1-6]）', t) and kids2[n].tag == q('p')]
def edit(n, f):
    k, ppr, chars, figs = exp[n]; exp[n] = (k, ppr, f(chars), figs)
edit(i_item1, lambda c: mod_chars(c, '沉降／上升速度', '沉降速度'))                     # 使用者手改
edit(i_para3, lambda c: mod_chars(c, '，不含上升情境', ''))                           # v3
edit(i_para3, lambda c: mod_chars(c, '垂直座標向上為正，故各速度均為負值。', ''))        # 使用者於 v6e 後在 Word 手改（2026-10-27 02:42 UTC 存檔）：刪此句
i_tbl1 = next(n for n, e in enumerate(kids2) if e.tag == q('tbl'))
def _RS(x): return clean(etree.fromstring('<w:rPr xmlns:w="%s">%s</w:rPr>' % (W, x)))
RS_N = _RS('<w:szCs w:val="28"/>'); RS_M = _RS('<w:rFonts w:eastAsia="MS Gothic"/><w:szCs w:val="28"/>')
RS_SM = _RS('<w:rFonts w:eastAsia="MS Gothic"/><w:szCs w:val="28"/><w:vertAlign w:val="superscript"/>'); RS_S = _RS('<w:szCs w:val="28"/><w:vertAlign w:val="superscript"/>')
RS_H1 = _RS('<w:rFonts w:hint="eastAsia"/><w:b/><w:bCs/><w:szCs w:val="28"/>'); RS_H2 = _RS('<w:b/><w:bCs/><w:szCs w:val="28"/>')
def sci_str(v):
    d = v.replace('.', '').lstrip('0'); e = Decimal(v).adjusted()
    return (d[0] + ('.' + d[1:] if len(d) > 1 else '')), e
def tbl1_edit(sg, header=True, sci=True):
    seq = list(sg[3])
    def idxs(): return [n for n, x in enumerate(seq) if x[0] != 'P']
    def rep(old, newc):
        ix = idxs(); st = ''.join(seq[n][1] for n in ix); i = st.index(old); assert st.count(old) == 1, old
        a_, b_ = ix[i], ix[i + len(old) - 1]; assert b_ - a_ == len(old) - 1
        seq[a_:b_ + 1] = newc
    if header:   # 使用者手改：表頭「10 個速度之範圍 (m/s)」→「終端沉降速度範圍 (m/s)」
        rep('10 個速度之範圍', [(RS_H1, c) for c in '終端'] + [(RS_H2, c) for c in '沉降速度範圍'])
    if sci:      # v6f：十列改科學記號（20 個數字，由後往前換）
        ix = idxs(); st = ''.join(seq[n][1] for n in ix)
        ms = list(re.finditer(r'−0\.\d+', st)); assert len(ms) == 20, len(ms)   # v7：連前導負號一併去掉
        for m in reversed(ms):
            a_, b_ = idxs()[m.start()], idxs()[m.end() - 1]
            mant, ex = sci_str(m.group(0)[1:])
            new = [(RS_N, c) for c in mant + '×10'] + ([(RS_SM, '−')] if ex < 0 else []) + [(RS_S, c) for c in str(abs(ex))]
            seq[a_:b_ + 1] = new
    return (sg[0], sg[1], sg[2], tuple(seq))
stop_old = '，各研究地點皆於 33.4 天內停止追蹤，穿越比例與海岸來源比例依序為貢寮 73.40%、26.60%，龜山島 78.35%、21.65%，新竹外海 97.77%、2.23%，南灣 68.91%、31.09%，連江 56.06%、43.94%'
edit(i_item4, lambda c: mod_chars(c, stop_old, ''))                                   # v3
for n in HAND_IND:                                                                     # 使用者手改：左縮 1680 → 1260
    k, ppr, chars, figs = exp[n]
    assert 'w:left="1680"' in ppr
    exp[n] = (k, ppr.replace('w:left="1680"', 'w:left="1260"'), chars, figs)
# ---- v4 登記之修改（343_formal_review.md 三組建議；各以「段落定位片段、舊、新」登記，與 edit_v4_inplace.py 各自獨立書寫）----
V4 = [
    ('共由「5 個水平位置', '候選面須為有效之三角形網格且非陸地，並於全部 50 個到達時間均維持濕潤（不因退潮而乾出）；符合條件者再依', '受體位置限於全部到達時間皆有流場與波浪資料之海域；符合條件者再依'),   # #6
    ('共由「5 個水平位置', '抽中後另須通過流場與波浪資料之逐時空間支援檢查，未通過者剔除，並於同一區塊內依隨機順序遞補。', ''),                                   # #6
    ('共由「5 個水平位置', '垂向位置抽樣', '深度抽樣'),                                     # #5
    ('50 個到達時間由 48 個分層時刻', '挑選時排除流場資料於日間銜接處水位不連續之時刻。', ''),   # #8
    ('50 個到達時間由 48 個分層時刻', '三種潮相代理」共', '三種潮相」共'),                       # #10
    ('50 個到達時間由 48 個分層時刻', '三種潮相代理分別為', '三種潮相以潮位資料判定，分別為'),     # #10
    ('以下列出各研究地點之到達時間驗證圖', '同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊。', ''),   # #11
    ('1、沉降動力參數', '附著生物之海廢在水體中', '附著生物之海底廢棄物在水體中'),               # #4
    ('表1 十類海廢', '十類海廢之代表形狀', '十類海底廢棄物之代表形狀'),                          # #4
    ('2、到達地點（Receptor locations）', '針對底床上的海底廢，', '針對底床上的海底廢棄物，'),   # #4
    ('附錄　十類海廢終端沉降速度之推導', '十類海廢終端', '十類海底廢棄物終端'),                    # #4（附錄標題）
    ('表A1 十類海廢終端沉降速度之下界與上界', '十類海廢終端', '十類海底廢棄物終端'),                # #4（表 A1 標題）
    ('各類海廢設定 5 種代表物件', '各類海廢設定', '各類海底廢棄物設定'),                          # #4
]
V4 += [   # v5：#9、#13（3-4-3 部分），#12 之（一）3 補沉底時刻定義
    ('50 個到達時間由 48 個分層時刻', '用以納入分層設計未必涵蓋之極端動力狀態', '用以納入分層設計以外之極端動力狀態'),   # #9
]
def swap_chars(seq):
    seq = list(seq); i = 0
    while i < len(seq) - 2:
        a_, b_, c_ = seq[i], seq[i + 1], seq[i + 2]
        if all(len(x) == 2 and x[0] != 'P' and len(x[1]) == 1 for x in (a_, b_, c_)) and a_[1] + b_[1] + c_[1] == '本研究':
            seq[i + 1] = (b_[0], '計'); seq[i + 2] = (c_[0], '畫'); i += 3
        else: i += 1
    return tuple(seq)
def swap_sig(sg):
    if sg[0] == 'p': return (sg[0], sg[1], swap_chars(sg[2]), sg[3])
    if sg[0] == 'tbl': return (sg[0], sg[1], sg[2], swap_chars(sg[3]))
    return sg
V5B = ('1、沉降動力參數', '假設10種不同的沉降速度', '設定 10 種不同的沉降速度')   # v5b：#1「假設」→「設定」，數字與中文間加半形空格（第 1 項在 3-4-3 原檔另有手改，故只套用於 v2 基準）
V4 += [   # v5c：第 2、3 項數字與中文間加半形空格
    ('2、到達地點（Receptor locations）', '設定20種不同', '設定 20 種不同'),
    ('3、到達時間（Arrival times）', '設定50種不同', '設定 50 種不同'),
]
def v4_apply(sigs, texts):
    for anchor, o, nw in V4:
        hit = [n for n, t in enumerate(texts) if anchor in t]
        assert len(hit) == 1, (anchor, hit)
        k, ppr, chars, figs = sigs[hit[0]]; sigs[hit[0]] = (k, ppr, mod_chars(chars, o, nw), figs)
v4_apply(exp, txt2)
V6_TRAJ = [   # v6：水平軌跡總覽導引段補結果句（錨點，舊，新）；圖號為合併版編號
    ('圖 8 為隨機抽取之 24 條', '線色依起始深度由淺至深。', '線色依起始深度由淺至深。軌跡多向南延伸並穿越流場範圍南界，其次向北延伸；少數抵達海岸，與圖 11 之主要來源方向一致。'),
    ('圖 15 為龜山島隨機抽取之 24 條', '回溯軌跡。', '回溯軌跡。軌跡向南至東南延伸者居多，其次向北延伸；部分抵達海岸，與圖 18 之主要來源方向一致。'),
    ('圖 22 為新竹外海隨機抽取之 24 條', '回溯軌跡。', '回溯軌跡。軌跡向西南至西方與東北方延伸，穿越流場範圍西界者較多；僅極少數抵達海岸，與圖 25 之主要來源方向一致。'),
    ('圖 29 為南灣隨機抽取之 24 條', '回溯軌跡。', '回溯軌跡。軌跡多向西北延伸並穿越流場範圍西界，少數向東延伸；部分抵達海岸，與圖 32 之主要來源方向一致。'),
    ('圖 36 為連江隨機抽取之 24 條', '回溯軌跡。', '回溯軌跡。軌跡多向東北延伸並穿越流場範圍北界，另有向西南延伸者；多條抵達福建沿岸與馬祖各島，與圖 39 之主要來源方向一致。'),
]
for _a, _o, _n in V6_TRAJ:
    _h = [n for n, t in enumerate(txt2) if t.startswith(_a)]; assert len(_h) == 1
    _k, _p, _c, _f = exp[_h[0]]; exp[_h[0]] = (_k, _p, mod_chars(_c, _o, _n), _f)
V6D = ('圖 9 為全部 100,000 顆虛擬質點於回溯期間內仍在追蹤之比例', '，垂直虛線標示 90 天與 180 天。', '。')   # v6d：貢寮圖 9 導引段刪「垂直虛線標示 90 天與 180 天」
_h = [n for n, t in enumerate(txt2) if t.startswith(V6D[0])]; assert len(_h) == 1
_k, _p, _c, _f = exp[_h[0]]; exp[_h[0]] = (_k, _p, mod_chars(_c, V6D[1], V6D[2]), _f)
V6_L3 = [('1、逐地點成果：', '1、'), ('2、綜合說明與結論：', '2、'), ('3、整體結論：', '3、')]
V6_L4 = [('（1）貢寮：', '（1）'), ('（2）龜山島：', '（2）'), ('（3）新竹外海：', '（3）'), ('（4）南灣：', '（4）'), ('（5）連江：', '（5）')]
for _lst, _o, _n in ((V6_L3, 'w:left="980" w:hangingChars="150" w:hanging="420"', 'w:leftChars="200" w:left="560"'), (V6_L4, 'w:left="1260" w:hangingChars="250" w:hanging="700"', 'w:leftChars="354" w:left="991"')):
    for _t, _pre in _lst:
        _h = [n for n, t in enumerate(txt2) if t == _t and kids2[n].tag == q('p')]; assert len(_h) == 1
        _k, _p, _c, _f = exp[_h[0]]; assert _o in _p, _t
        exp[_h[0]] = (_k, _p.replace(_o, _n), mod_chars(_c, _pre, ''), _f)
V5E = ('圖 13 為依沉底時刻劃分之貢寮四季曲線', '即虛擬質點回溯起點所對應之時刻，為所屬到達時間往前推一個 0–90 天之沉底年齡；', '亦即虛擬質點回溯之起點；由於廢棄物實際何時沉底未知，沉底時刻由所屬到達時間往前推一段隨機抽取之沉底年齡決定，每個系集成員各自抽取；')   # v5e
_h = [n for n, t in enumerate(txt2) if V5E[0] in t]; assert len(_h) == 1
_k, _p, _c, _f = exp[_h[0]]; exp[_h[0]] = (_k, _p, mod_chars(_c, V5E[1], V5E[2]), _f)
_h = [n for n, t in enumerate(txt2) if V5B[0] in t and t.startswith('1、')]; assert len(_h) == 1
_k, _p, _c, _f = exp[_h[0]]; exp[_h[0]] = (_k, _p, mod_chars(_c, V5B[1], V5B[2]), _f)
# 使用者於 v6b 後在 Word 手改（2026-10-27 00:49 UTC 存檔）：(1) 第 4 項刪「或抵達海岸」與「回溯期間上限為 180 天。」（段落標記 rFonts 加 hint）；(2) 三個小標（逐地點成果、綜合說明與結論、整體結論）縮排去掉 leftChars（只留 left=560）
_k, _p, _c, _f = exp[i_item4]
_c = mod_chars(_c, '邊界或抵達海岸為止', '邊界為止'); _c = mod_chars(_c, '回溯期間上限為 180 天。', '')
exp[i_item4] = (_k, _p, _c, _f)   # v6f：使用者於 v6e 後 Word 重存，段落標記 rFonts 之 hint 已不存在（原手改所加者被 Word 正規化掉）
for _t, _pre in V6_L3:
    _h = [n for n, t in enumerate(txt2) if t == _t]; _k, _p, _c, _f = exp[_h[0]]
    assert 'w:leftChars="200" w:left="560"' in _p; exp[_h[0]] = (_k, _p.replace('w:leftChars="200" w:left="560"', 'w:left="560"'), _c, _f)
exp[i_tbl1] = tbl1_edit(exp[i_tbl1])   # 使用者手改表頭 ＋ v6f 科學記號
exp = [swap_sig(x) for x in exp]   # v6c
V6E = [   # v6e：使用者同意之五處（皆在 swap 後之「本計畫」文字上套用；錨點，舊，新）
    ('各類 10 個速度由本', '各類 10 個速度由本計畫以受力平衡求解，', '各類 10 個速度係以受力平衡求解，'),                       # （一）1 末段
    ('（一）計算式：阻力係數', '本計畫參照其兩極限相加之概念', '參照其兩極限相加之概念'),                                    # 附錄三
    ('下界取與沉降姿態最接近之固定物體值', '本計畫薄板依教科書', '本計畫採用之薄板，依教科書'),                              # 附錄三
    ('（一）代表物件輸入：保麗龍', '尚無實測，本計畫以表 A4', '尚無實測，故以表 A4'),                                       # 附錄五
]
V6E_R = ('回溯期間內，各研究地點之虛擬質點皆於 33.4 天內停止追蹤', '，無回溯達 180 天者。', '。')   # 綜合說明與結論
def v6e_apply(sigs, texts, lst=None):
    for anchor, o, nw in (V6E + [V6E_R] if lst is None else lst):
        hit = [n for n, t in enumerate(texts) if anchor in t]
        assert len(hit) == 1, (anchor, hit)
        k, ppr, chars, figs = sigs[hit[0]]; sigs[hit[0]] = (k, ppr, mod_chars(chars, o, nw), figs)
v6e_apply(exp, txt2)   # v6e

# ---------------------------------------------------------------- v7：去負號、附錄速度科學記號、欄寬（以文字錨點定位，exp 與 sgA 各自套用）
_AFT = [q(x) for x in ('rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath')]
_supc = {}
def sup_rs(rs):
    if rs in _supc: return _supc[rs]
    el = etree.fromstring(rs.replace('<w:rPr', '<w:rPr xmlns:w="%s"' % W, 1)) if rs else etree.Element(q('rPr'))
    for v in el.findall(q('vertAlign')): el.remove(v)
    va = etree.Element(q('vertAlign')); va.set(q('val'), 'superscript')
    for n_, c_ in enumerate(el):
        if c_.tag in _AFT: el.insert(n_, va); break
    else: el.append(va)
    _supc[rs] = clean(el); return _supc[rs]
V7_TOK = re.compile(r'(?<![\d.])0\.\d+(?![\d%])')
def _pars(seq):
    segs = []; cur = []
    for n_, x_ in enumerate(seq):
        if x_[0] == 'P':
            if cur: segs.append(cur)
            cur = []
        else: cur.append(n_)
    if cur: segs.append(cur)
    return segs
def sci_seq(seq, fullpar=None, allowed=None, n_expect=None):
    seq = list(seq); cnt = 0
    for idx in reversed(_pars(seq)):
        text = ''.join(seq[n_][1] for n_ in idx)
        if fullpar is not None and not re.fullmatch(fullpar, text): continue
        for m in reversed(list(V7_TOK.finditer(text))):
            if allowed is not None and m.group(0) not in allowed: continue
            a_, b_ = idx[m.start()], idx[m.end() - 1]; rs0 = seq[a_][0]
            mant, ex = sci_str(m.group(0))
            seq[a_:b_ + 1] = [(rs0, c_) for c_ in mant + '×10'] + [(sup_rs(rs0), c_) for c_ in '−' + str(abs(ex))]; cnt += 1
    assert n_expect is None or cnt == n_expect, (cnt, n_expect)
    return seq
def regrid(tg, ws):
    it = iter(ws); return re.sub(r'(<w:gridCol w:w=")\d+(")', lambda m: m.group(1) + str(next(it)) + m.group(2), tg)
GRID_T1 = [1100, 2050, 3407, 1900]; GRID_A10 = [1494, 2300, 2673, 1990]
V7_P = [   # (段落錨點, 刪去之字 或 None, 允許換算之數字集合 或 None, 預期換算數)
    ('步驟 3　用於模擬', '垂直座標向上為正，故各解取負值。', None, 0),
    ('（單位均為 m/s；垂直座標向上為正時取負值。）', '；垂直座標向上為正時取負值', None, 0),
    ('十個解之範圍 0.03671', None, {'0.03671', '0.17548'}, 2),
    ('表 A10 之實測對象', None, {'0.0367', '0.175', '0.115', '0.120', '0.107', '0.427', '0.176', '0.177'}, 8),
    ('（一）薄板之沉速主要由厚度決定', None, {'0.1501'}, 1),
]
V7_T = [   # (表標題錨點, fullpar, allowed, 預期換算數, 新欄寬)
    ('表A1 ', None, None, 20, None), ('表A4 ', r'0\.\d+–0\.\d+', None, 100, None), ('表A9 ', r'0\.\d{5}', None, 10, None),
    ('表A10 ', None, None, 14, GRID_A10), ('表A11 ', None, {'0.187', '0.175'}, 4, None), ('表A12 ', r'0\.\d{4}', None, 10, None),
]
def v7_apply(sigs, texts, kinds):
    for anchor, dele, allowed, n_exp in V7_P:
        hit = [n_ for n_, t_ in enumerate(texts) if t_.startswith(anchor)]; assert len(hit) == 1, (anchor, hit)
        k_, ppr_, chars_, figs_ = sigs[hit[0]]
        if dele is not None: chars_ = mod_chars(chars_, dele, '')
        if allowed is not None: chars_ = tuple(sci_seq(chars_, allowed=allowed, n_expect=n_exp))
        sigs[hit[0]] = (k_, ppr_, chars_, figs_)
    for anchor, fp, allowed, n_exp, gr in V7_T:
        hit = [n_ for n_, t_ in enumerate(texts) if t_.startswith(anchor)]; assert len(hit) == 1, (anchor, hit)
        j_ = hit[0] + 1; assert sigs[j_][0] == 'tbl', anchor
        k_, tp_, tg_, seq_ = sigs[j_]
        sigs[j_] = (k_, tp_, regrid(tg_, gr) if gr else tg_, tuple(sci_seq(seq_, fullpar=fp, allowed=allowed, n_expect=n_exp)))
v7_apply(exp, txt2, None)
exp[i_tbl1] = (exp[i_tbl1][0], exp[i_tbl1][1], regrid(exp[i_tbl1][2], GRID_T1), exp[i_tbl1][3])   # v7：表 1 欄寬
# v8 登記：主檔含使用者於 v7（bak16-20261027）之後在 Word 手改兩處——(1) 第 1 項標題「參數：」後之半形空格刪除；(2) 表 1 三個儲存格「半壓扁至嚴重壓扁」→「半壓扁至重壓扁」（Word 重存另使表 1 速度欄 run 重新切分，不影響逐字簽章）
edit(i_item1, lambda c: mod_chars(c, '參數： 設定', '參數：設定'))
def _tbl_hand(sg):
    seq = list(sg[3]); idx = [n_ for n_, x_ in enumerate(seq) if x_[0] != 'P']; st_ = ''.join(seq[n_][1] for n_ in idx)
    ps = [m_.start() for m_ in re.finditer('半壓扁至嚴重壓扁', st_)]
    if not ps:   # v10：sig() 已將全文「嚴重壓扁」正規化為「重壓扁」（canon10）；此處僅確認表 1 三個儲存格已是「半壓扁至重壓扁」
        assert len(re.findall('半壓扁至重壓扁', st_)) == 3, st_[:80]; return sg
    assert len(ps) == 3, len(ps)
    drop = {idx[p_ + 4] for p_ in ps}
    return (sg[0], sg[1], sg[2], tuple(x_ for n_, x_ in enumerate(seq) if n_ not in drop))
exp[i_tbl1] = _tbl_hand(exp[i_tbl1])
if V8 or True:   # v8：主檔含使用者於 v7 之後在 Word 手改之處（逐字登記；見下）
    def _s(sg): return ''.join(c for _, c in (sg[2] if sg[0] == 'p' else [x for x in sg[3] if x[0] != 'P'])) if sg[0] in ('p', 'tbl') else ''
    if os.environ.get('CHECK_DEBUG'):
        for n_, (a_, b_) in enumerate(zip(exp, sgM)):
            if a_ != b_:
                sa, sb = _s(a_), _s(b_); i_ = next((k for k in range(min(len(sa), len(sb))) if sa[k] != sb[k]), min(len(sa), len(sb)))
                print('DIFF', n_, a_[0], repr(sa[max(0, i_ - 15):i_ + 25]), '|', repr(sb[max(0, i_ - 15):i_ + 25]), 'ppr' if a_[1] != b_[1] else '', file=sys.stderr)
diff = [n for n, (a, b) in enumerate(zip(exp, sgM)) if a != b]
chk('M1 合併版 ＝ v2 ＋ 登記之修改（使用者手改：第 1 項刪「／上升」、%d 個（1）…（6）段左縮 1680→1260；v3：第 1 項刪「，不含上升情境」、停止邊界段刪成果數字；v4：#4 海底廢棄物（第 4 項「海廢移入」於 v6b 還原為計畫書原文「海廢」）、#5 深度、#6 受體選取句、#8、#10、#11、#14 共 15 筆文字修改；v5：#9 共 1 筆（#12 已於 v5d 還原、#13 已於 v6b 還原為計畫書原文）、v5e／v5f：成果段圖 13 導引段 1 筆（最終句不含「0–90 天」）；v6c：全文「本研究」→「本計畫」21 處（含附錄、表；逐字對位）；使用者於 v6b 後手改 2 處（第 4 項刪「或抵達海岸」與「回溯期間上限為 180 天。」、三小標縮排去 leftChars）；v6：（二）內 3 個小標與 5 個研究地點小標去編號並改縮排、5 段水平軌跡導引段補結果句、v5b：#1 共 1 筆、v5c：第 2、3 項加空格共 2 筆（3-4-3 部分））；其餘全部區塊逐字、逐格式、逐圖片相同' % len(HAND_IND), not diff, str(diff[:5]))
chk('M2 登記之手改共 %d 段（1 + %d）' % (1 + len(HAND_IND), len(HAND_IND)), len(HAND_IND) == 11)

# ---------------------------------------------------------------- 結構與標題順序
ia = next(n for n, e in enumerate(kidsA) if e.tag == q('p') and ptext(e).startswith('4、溯源停止邊界'))
NINS = len(kidsB)
ins = kids[ia + 1: ia + 1 + NINS + DELTA]
chk('P7 本文區塊數 = 3-4-3 區塊數 + 成果段落數(%d) + v8 增減(%d)' % (NINS, DELTA), len(kids) == len(kidsA) + NINS + DELTA)
sgA = [sig(e, za, relsA, True) for e in kidsA]
txtA = [ptext(e) if e.tag == q('p') else '' for e in kidsA]
v4_apply(sgA, txtA); sgA = [swap_sig(x) for x in sgA]; v6e_apply(sgA, txtA, V6E); v7_apply(sgA, txtA, None)   # v4 之文字修改同樣套用於 3-4-3 原檔，使 P8 仍只比對「登記之 3 段」與 keep／大綱
sgH = [sig(e, zm, relsM, True) for e in kids[:ia + 1]]; sgT = [sig(e, zm, relsM, True) for e in kids[ia + 1 + NINS + DELTA:]]
dh = [n for n, (a, b) in enumerate(zip(sgA[:ia + 1], sgH)) if a != b]; dt = [n for n, (a, b) in enumerate(zip(sgA[ia + 1:], sgT)) if a != b]
chk('P8 3-4-3 前段、後段（參考資料、附錄）與原檔相同，僅差登記之 4 段（第 1 項、第 1 項末段、停止邊界段、v6f 表 1）與 keep／大綱屬性',
    set(dh) <= {i_item1, i_para3, i_item4, i_tbl1} and not dt and len(sgH) == ia + 1 and len(sgT) == len(kidsA) - ia - 1, '%s %s' % (dh, dt))
imgsA = [(f, za.read('word/' + relsA[f[2]])) for p in kidsA if p.tag == q('p') for f in figs_in(p)]
imgsB = [(f, zb.read('word/' + relsB[f[2]])) for p in kidsB for f in figs_in(p)]
imgsM = [(f, zm.read('word/' + relsM[f[2]])) for _, f in allfig]
caps = [(n, ptext(p)) for n, p in enumerate(kids) if p.tag == q('p') and re.match(r'^圖\d+ ', ptext(p))]
capnums = [int(re.match(r'^圖(\d+)', t).group(1)) for _, t in caps]
chk_old('F1 圖說編號 1–42 連續且依序', capnums == list(range(1, 43)), str(capnums))
chk('F2 每個圖說緊接前一段為圖片段', all(kids[n - 1].tag == q('p') and figs_in(kids[n - 1]) for n, _ in caps))
chk('F3 圖 1–7 圖檔 sha 與 3-4-3 原圖相同，顯示尺寸相同', len(imgsA) == 7 and all(sha(imgsM[k][1]) == sha(imgsA[k][1]) and imgsM[k][0][:2] == imgsA[k][0][:2] for k in range(7)))
chk_old('F4 圖 8–42 圖檔 sha 與執行成果圖 1–35 相同', len(imgsB) == 35 and all(sha(imgsM[7 + k][1]) == sha(imgsB[k][1]) for k in range(35)))
chk_old('F5 圖 8–42 顯示尺寸與執行成果相同', all(imgsM[7 + k][0][:2] == imgsB[k][0][:2] for k in range(35)))
textw = 11906 - 2 * 1440
chk('F7 全部圖寬不超過版心寬（容許 0.5 pt 進位）', all(f[0] <= textw * 635 + 6350 for _, f in allfig))
tbls = [e for e in kids if e.tag == q('tbl')]
chk('F8 表格數不變', len(tbls) == len([e for e in kidsA if e.tag == q('tbl')]))
tcap = [ptext(p) for p in kids if p.tag == q('p') and re.match(r'^表(A?\d+) ', ptext(p))]
tn = [re.match(r'^表(A?\d+)', t).group(1) for t in tcap]
chk('F9 表號：表1、表2、表A1–A12 連續且不變', tn == ['1', '2'] + ['A%d' % n for n in range(1, 13)], str(tn))
def find_idx(prefix, start=0):
    for n in range(start, len(kids)):
        if kids[n].tag == q('p') and ptext(kids[n]).startswith(prefix): return n
mk = ['Lagrangian系集逆向溯源', '（一）情境矩陣與停止條件設計：', '1、沉降動力參數：', '2、到達地點', '3、到達時間', '4、溯源停止邊界',
      '（二）分析成果：', '逐地點成果：', '貢寮：', '龜山島：', '新竹外海：', '南灣：', '連江：',
      '綜合說明與結論：', '整體結論：', '參考資料', '附錄　十類海底廢棄物終端沉降速度之推導']
pos = []; s_ = 0
for m in mk:
    k = find_idx(m, s_); pos.append(k)
    if k is not None: s_ = k + 1
chk('H1 標題依序：標題→（一）→1–4→（二）→逐地點成果→貢寮…連江（v6 起不編號）→綜合說明與結論→整體結論→參考資料→附錄', None not in pos and pos == sorted(pos), str(pos))
chk('H2 「參考資料」「附錄」位於全文最末', pos[-2] > pos[-3] > pos[-4])
sitepos = pos[8:13]
nf = [sum(1 for n in range(a, b) if kids[n].tag == q('p') and figs_in(kids[n])) for a, b in zip(sitepos, sitepos[1:] + [pos[13]])]
chk_old('H3 每個研究地點各 7 張圖', nf == [7] * 5, str(nf))
alltext = [ptext(p) for p in kids if p.tag == q('p')]
chk('H5 「詳見附錄」之參照仍成立', any('詳見附錄' in t for t in alltext) and pos[16] is not None)
apxA = [ptext(p) for p in kidsA[next(n for n, e in enumerate(kidsA) if e.tag == q('p') and ptext(e).startswith('附錄　')):] if p.tag == q('p')]
apxM = [ptext(p) for p in kids[pos[16]:] if p.tag == q('p')]
apxA4 = [t.replace('十類海廢', '十類海底廢棄物').replace('各類海廢設定', '各類海底廢棄物設定').replace('本研究', '本計畫') for t in apxA]
for _a, _o, _n in V6E[1:]:   # v6e：附錄三、附錄五三處（（一）1 末段不在附錄）
    apxA4 = [t.replace(_o, _n) if _a in t else t for t in apxA4]
def _t7(t):   # v7：附錄正文（頂層段落）之刪句與科學記號（文字層級；w:t 順序為尾數×10、上標「−n」）
    for anchor, dele, allowed, n_exp in V7_P:
        if t.startswith(anchor):
            if dele is not None: t = t.replace(dele, '')
            if allowed is not None: t = V7_TOK.sub(lambda m: (lambda mx: mx[0] + '×10' + '−' + str(abs(mx[1])))(sci_str(m.group(0))) if m.group(0) in allowed else m.group(0), t)
    return t
apxA4 = [_t7(t) for t in apxA4]
apxA4 = [t.replace('嚴重壓扁', '重壓扁') for t in apxA4]   # v10 登記：使用者手改（全文「嚴重壓扁」→「重壓扁」）
chk('H6 附錄全文與 3-4-3 相同（v4 僅附錄標題、表 A1 標題、「各類海廢設定」三處用語改為海底廢棄物；v6e 另有附錄三、附錄五三處登記修改）', apxA4 == apxM and sum(a != b for a, b in zip(apxA, apxM)) >= 3, str(sum(a != b for a, b in zip(apxA, apxM))))
refA = [ptext(p) for p in kidsA[next(n for n, e in enumerate(kidsA) if e.tag == q('p') and ptext(e) == '參考資料'):] if p.tag == q('p')][:20]
refM = [ptext(p) for p in kids[pos[15]:] if p.tag == q('p')][:20]
chk('H7 參考資料全文與 3-4-3 相同', refA == refM)
head = kids[:ia + 1]

# ---------------------------------------------------------------- 圖號引用
def refs_bad(block, lo, hi):
    bad = []
    for p in block:
        if p.tag != q('p'): continue
        t = ptext(p)
        if re.match(r'^圖\d+ ', t): continue
        for m in re.finditer(r'圖 ?(\d+)', t):
            if not (lo <= int(m.group(1)) <= hi): bad.append(int(m.group(1)))
    return bad
chk('R1 3-4-3 段內圖號引用皆在 1–7', not refs_bad(head, 1, 7))
chk_old('R2 成果段內圖號引用皆在 8–42', not refs_bad(ins, 8, 42))
badn = []
for n, p in enumerate(ins):
    if p.tag != q('p'): continue
    t = ptext(p)
    if re.match(r'^圖\d+ ', t) or figs_in(p): continue
    if n + 1 < len(ins) and figs_in(ins[n + 1]):
        m = re.match(r'圖 (\d+)', t); capn = int(re.match(r'^圖(\d+)', ptext(ins[n + 2])).group(1))
        if not m or int(m.group(1)) != capn: badn.append((t[:25], capn))
chk('R3 每個「圖 N …」引導段之 N 等於其後圖說之編號', not badn, str(badn[:3]))
badr = []
for t in [ptext(p) for p in ins if p.tag == q('p')]:
    for m in re.finditer(r'圖 (\d+) 至圖 (\d+)', t):
        a, b = int(m.group(1)), int(m.group(2))
        if b - a != 3 or capnums[a - 1] != a: badr.append((a, b))
    for m in re.finditer(r'其中圖 (\d+) 為十類', t):
        if (int(m.group(1)) - 8) % 7 != 3: badr.append(('其中', m.group(1)))
chk_old('R4 「圖 a 至圖 b」皆為 4 張、「其中圖 N 為十類」位於各地點第 4 張', not badr, str(badr))
chk_old('R5 「各格依圖 N 圖例」指向貢寮十類圖（圖 11）', any('各格依圖 11 圖例' in t for t in alltext))

# ---------------------------------------------------------------- 成果文字可追溯
bt = [(pstyle(p), ptext(p)) for p in kidsB]
E = [('虛擬質點穿越流場範圍邊界或抵達海岸時停止追蹤；貢寮之', '貢寮之'),
     ('情境矩陣由 5 處研究地點、10 類代表物件、20 個受體（5 個水平位置 × 4 個深度）、50 個到達時間與 10 個系集成員交叉而成，每顆', '情境設計如（一）所述，另以 10 個系集成員交叉，每顆'),
     ('綜合說明與整體結論置於章末', '綜合說明與整體結論置於最後'),
     (V5E[1], V5E[2])]
H3 = {'逐地點成果': '逐地點成果：', '綜合說明與結論': '綜合說明與結論：', '整體結論': '整體結論：'}   # v6：不編號
# ---------------------------------------------------------------- v8：成果段轉換（登記）——主檔文字 → 候選文字
V8_RENUM = {8: 8, 9: 9, 10: 10, 15: 19, 16: 20, 17: 21, 22: 30, 23: 31, 24: 32, 29: 41, 30: 42, 31: 43, 36: 52, 37: 53, 38: 54}   # 保留之 15 張圖：舊號 → 新號
V8_TAIL = [   # 軌跡導引段結尾：與圖（十類來源方向）→ 與圖（全年邊界）及圖（全年海岸）；新竹外海僅邊界圖（全年無集中海岸段）
    ('圖 8 為隨機抽取之 24 條', '與圖 11 之主要來源方向一致。', '與圖 11 及圖 12 之主要來源一致。'),
    ('圖 15 為龜山島', '與圖 18 之主要來源方向一致。', '與圖 22 及圖 23 之主要來源一致。'),
    ('圖 22 為新竹外海', '與圖 25 之主要來源方向一致。', '與圖 33 之主要來源一致。'),
    ('圖 29 為南灣', '與圖 32 之主要來源方向一致。', '與圖 44 及圖 45 之主要來源一致。'),
    ('圖 36 為連江', '與圖 39 之主要來源方向一致。', '與圖 55 及圖 56 之主要來源一致。')]
V8_POLAR = [('貢寮', '圖 11 至圖 14 為貢寮之圓形來源方位圖', '圖14 貢寮來源方向機率（依大小潮）'), ('龜山島', '圖 18 至圖 21 為龜山島之圓形來源方位圖', '圖21 龜山島來源方向機率（依大小潮）'),
              ('新竹外海', '圖 25 至圖 28 為新竹外海之圓形來源方位圖', '圖28 新竹外海來源方向機率（依大小潮）'), ('南灣', '圖 32 至圖 35 為南灣之圓形來源方位圖', '圖35 南灣來源方向機率（依大小潮）'),
              ('連江', '圖 39 至圖 42 為連江之圓形來源方位圖', '圖42 連江來源方向機率（依大小潮）')]
V8_REW = [('本節呈現', 'p33'), ('各研究地點依序以七張圖說明', 'p34'), ('各研究地點十類之間', 'p148_summary')] + [('（%d）%s：' % (k + 1, s_), 'conclusion_%d' % (k + 1)) for k, s_ in enumerate(['貢寮', '龜山島', '新竹外海', '南灣', '連江'])] + [('（6）綜合五個研究地點', 'conclusion_6')]
def v8_transform(ex):
    log = json.load(open(os.path.join(HERE, 'v10_build_log.json' if V10 else ('v9_build_log.json' if V9 else 'v8_build_log.json')), encoding='utf-8'))
    newt = {}
    for c in log['changes']:
        if c['kind'] == 'new': newt.setdefault(c['site'], []).append(c['new'])
    rew = {c['role']: c['new'] for c in log['changes'] if c['kind'] == 'rewrite' and c.get('role')}
    out = []
    for t in ex:
        for a_, o_, n_ in V8_TAIL:
            if t.startswith(a_):
                assert t.endswith(o_), a_; t = t[:-len(o_)] + n_
        mm = re.match(r'^圖( ?)(\d+)', t)
        if mm and int(mm.group(2)) in V8_RENUM and not t.startswith('圖 11 至圖') and not any(t.startswith(pp[1]) or t == pp[2] for pp in V8_POLAR) and not re.match(r'^圖 ?(11|12|13|14|18|19|20|21|25|26|27|28|32|33|34|35|39|40|41|42)\b', t):
            t = '圖%s%d' % (mm.group(1), V8_RENUM[int(mm.group(2))]) + t[mm.end():]
        for pre_, role in V8_REW:
            if t.startswith(pre_) and len(t) > 30: t = rew[role]
        out.append(t)
    # 圓形來源方位圖（含圖說）→ 來源分段機率圖
    res2 = []; i = 0
    while i < len(out):
        hit = next((pp for pp in V8_POLAR if out[i].startswith(pp[1])), None)
        if hit:
            j = next(k for k in range(i, len(out)) if out[k] == hit[2])
            res2 += newt[{'貢寮': 'gongliao', '龜山島': 'guishan', '新竹外海': 'hsinchu', '南灣': 'nanwan', '連江': 'lienchiang'}[hit[0]]]; i = j + 1
        else: res2.append(out[i]); i += 1
    return res2

expt = []; site = 0; cnt_edit = [0, 0, 0, 0]; cnt_v6 = [0]
for idx, (st, t) in enumerate(bt):
    if figs_in(kidsB[idx]): continue
    if idx == 0: expt.append('（二）分析成果：'); continue
    if st == '-': expt.append(H3[t]); continue
    if st == '2': site += 1; expt.append('%s：' % t); continue   # v6：研究地點小標不編號
    if re.match(r'^圖\d+ ', t):
        n = int(re.match(r'^圖(\d+)', t).group(1)) + 7; expt.append('圖%d %s' % (n, t.split(' ', 1)[1])); continue
    if t.startswith('　　'): t = t[2:]
    for k, (o, nw) in enumerate(E):
        if o in t: t = t.replace(o, nw); cnt_edit[k] += 1
    t = re.sub(r'圖( ?)(\d+)', lambda m: '圖%s%d' % (m.group(1), int(m.group(2)) + 7), t)
    t = re.sub(r'^(\d)、', r'（\1）', t) if (idx >= 120 and re.match(r'^\d、', t)) else t
    for _a, _o, _n in V6_TRAJ:
        if t.startswith(_a): t = t.replace(_o, _n); cnt_v6[0] += 1
    if t.startswith(V6D[0]): t = t.replace(V6D[1], V6D[2]); cnt_v6[0] += 100
    if t.startswith(V6E_R[0]): assert V6E_R[1] in t; t = t.replace(V6E_R[1], V6E_R[2]); cnt_v6[0] += 1000
    expt.append(t)
expt = v8_transform(expt) if V8 else expt
got = [ptext(p) for p in ins if p.tag == q('p') and not figs_in(p)]
def _t1_old(t):   # v10 登記之新文字還原為舊文字（新文字之逐字內容由 V10-12、V10-13 斷言）
    t = t.replace(SENT_E, '')
    t = t.replace('為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。', '為全部 100,000 顆虛擬質點於回溯期間內仍在追蹤之比例。')
    t = re.sub('全部 100,000 顆虛擬質點合計，(?=(?:貢寮|龜山島|新竹外海|南灣|連江)仍在追蹤之比例於回溯第)', '', t)
    t = re.sub('(?<=降至 0%。)十類曲線相近(?:，其中不可回收塑膠略慢)?。', '', t)
    t = t.replace('（十類）；小圖為回溯前 20 天之放大', '')
    return t.replace('呈現十類虛擬質點各自於回溯期間內仍在追蹤之比例', '呈現全部虛擬質點於回溯期間內仍在追蹤之比例')
if V10: got = [_t1_old(t) for t in got]
chk('T1 成果文字 = 執行成果原文 + 已登記修改（獨立重建後逐段相等，%d 段）' % len(expt), got == expt,
    next((('第%d段不同：%r / %r' % (n, a[:40], b[:40])) for n, (a, b) in enumerate(zip(got, expt)) if a != b), ''))
chk('T2 v6 五段水平軌跡導引段補句各恰發生一次；v6d 貢寮圖 9 導引段刪虛線說明恰一次；v6e 綜合說明刪「無回溯達 180 天者」恰一次', cnt_v6 == [1105], str(cnt_v6))
chk('T2 四項文字刪改各恰發生一次（v5e 增：貢寮圖 13 導引段沉底時刻定義改為隨機決定）', cnt_edit == [1, 1, 1, 1], str(cnt_edit))
chk('T3 成果段無行首全形空格', not any(t.startswith('　') for t in got))
n343 = ''.join(ptext(e) for e in kids[:ia + 1] if e.tag == q('p'))
chk('T4 導言之系集成員、顆數敘述保留（3-4-3 無此述）', any('10 個系集成員' in t and '500,000' in t and '100,000' in t for t in got))
chk('T5 3-4-3 原文未提 100,000／500,000／系集成員', '100,000' not in n343 and '500,000' not in n343 and '系集成員' not in n343)

# ---------------------------------------------------------------- 版面格式（成果段，與 3-4-3 參照段；左縮依使用者手改 1260）
def ref_para(prefix): return next(e for e in kidsA if e.tag == q('p') and ptext(e).startswith(prefix))
def np_(p):
    ps = p.find(q('pPr')); return strip_keep(clean(ps)) if ps is not None else ''
R_L2 = np_(ref_para('（一）')); R_L3 = np_(ref_para('4、溯源')); R_BODY = np_(ref_para('每個水平位置再隨機取'))
R_CAP = np_(ref_para('圖3 ')); R_FIG = np_(kidsA[14])
L4 = R_L3.replace('w:left="980" w:hangingChars="150" w:hanging="420"', 'w:left="1260" w:hangingChars="250" w:hanging="700"')
L3N = R_L3.replace('w:left="980" w:hangingChars="150" w:hanging="420"', 'w:left="560"')   # v6：不編號（使用者於 v6b 後存檔時 Word 去掉 leftChars），起於 560、無懸掛
L4N = R_L3.replace('w:left="980" w:hangingChars="150" w:hanging="420"', 'w:leftChars="354" w:left="991"')   # v6：研究地點小標與內文左緣（991）對齊
okfmt = {k: 0 for k in ('L2', 'L3', 'L4', 'ITEM', 'BODY', 'CAP', 'FIG')}; badf = []
for p in ins:
    t = ptext(p); pp = np_(p)
    if figs_in(p): k = 'FIG'; ok = pp == R_FIG
    elif re.match(r'^圖\d+ ', t): k = 'CAP'; ok = pp == R_CAP
    elif t == '（二）分析成果：': k = 'L2'; ok = pp == R_L2
    elif t in ('逐地點成果：', '綜合說明與結論：', '整體結論：'): k = 'L3'; ok = pp == L3N
    elif re.match(r'^(貢寮|龜山島|新竹外海|南灣|連江)：$', t): k = 'L4'; ok = pp == L4N
    elif re.match(r'^（\d）', t): k = 'ITEM'; ok = pp == L4
    else: k = 'BODY'; ok = pp == R_BODY
    okfmt[k] += ok
    if not ok: badf.append((k, t[:20]))
chk('V1 各層級段落格式與 3-4-3 對應段落一致（條列左縮 1260 為使用者手改；v6：L3 小標左 560 無懸掛、研究地點小標左 991 與內文對齊）', not badf, str(badf[:3]))
chk_old('V2 各類段落數：L2×1、L3×3、L4×5、條列×6、圖×35、圖說×35', (okfmt['L2'], okfmt['L3'], okfmt['L4'], okfmt['ITEM'], okfmt['FIG'], okfmt['CAP']) == (1, 3, 5, 6, 35, 35), str(okfmt))
allowed = {'<w:rPr><w:szCs w:val="28"/></w:rPr>', '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:szCs w:val="28"/></w:rPr>',
           '<w:rPr><w:color w:val="000000"/><w:szCs w:val="28"/></w:rPr>', '<w:rPr><w:noProof/></w:rPr>'}
rp = set()
for p in ins:
    for r in p.iter(q('r')):
        x = r.find(q('rPr'))
        if x is not None: rp.add(clean(x))
chk('V3 成果段 run 格式僅限 3-4-3 所用之 4 種', rp <= allowed, str(rp - allowed))
insxml = ''.join(etree.tostring(p, encoding='unicode') for p in ins)
chk('V4 成果段僅用 afe 或無樣式', set(re.findall(r'<w:pStyle w:val="([^"]*)"', insxml)) <= {'afe'})
chk('V5 成果段無 w:sz／w:b／eastAsia 直接格式', not re.search(r'<w:sz |<w:b/>|<w:b |<w:i/>|w:eastAsia=', insxml))
chk('V6 圖說：「圖」＋「編號＋半形空格」＋說明', all(re.match(r'^圖\d+ \S', t) for _, t in caps))

# ---------------------------------------------------------------- 數值一致（成果段；3-4-3 已不含結果數字）
PUB = {'貢寮': ('73.40', '26.60'), '龜山島': ('78.35', '21.65'), '新竹外海': ('97.77', '2.23'), '南灣': ('68.91', '31.09'), '連江': ('56.06', '43.94')}
sites = ['貢寮', '龜山島', '新竹外海', '南灣', '連江']
blocks = {s: [ptext(p) for p in kids[a:b] if p.tag == q('p')] for s, a, b in zip(sites, sitepos, sitepos[1:] + [pos[13]])}
dres = {}; lastday = {}; top = {}; sub = {}; area = {}; depth = {}
for s, bl in blocks.items():
    jt = '\n'.join(bl)
    m = re.search(r'穿越比例為 ([\d.]+)%，海岸來源比例為 ([\d.]+)%', jt) or re.search(r'穿越流場範圍邊界者占 ([\d.]+)%，稱為穿越比例，抵達海岸者占 ([\d.]+)%', jt)
    dres[s] = (m.group(1), m.group(2))
    lastday[s] = re.search(r'至第 ([\d.]+) 天降至 0%', jt).group(1)
    top[s] = None if V8 else re.search(r'全年（十類合併）之主要來源方向為(.+?)，機率為 ([\d.]+)%，其次為(\S+?) ([\d.]+)%', jt).groups()
    m = re.search(r'海岸來源（([\d.]+)%）中，([^。]+)。', jt)
    if m: sub[s] = (m.group(1), re.findall(r'占 ([\d.]+)%', m.group(2)))
    area[s] = re.search(r'高密度區間面積分別為 ([\d,]+)、', jt).group(1)
    m = re.search(r'停留時數之 ([\d.]+)% 位於水深淺於 200 m', jt); depth[s] = m.group(1) if m else None
chk('N1 5 地點穿越／海岸來源比例（成果各地點段）= 已公布值', dres == PUB, str(dres))
for s in sites:
    a, b = Decimal(dres[s][0]), Decimal(dres[s][1])
    chk('N2 %s：穿越＋海岸來源%s = 100.00' % (s, '（另 1 顆觸底 0.001%）' if s == '南灣' else ''), abs(a + b - 100) <= Decimal('0.0100') and (a + b == 100 or s == '南灣'), str(a + b))
chk('N3 「33.4 天」：連江最末降至 0% 日 = 全部地點最大值 = 綜合說明之 33.4 天', lastday['連江'] == '33.4' and max(Decimal(v) for v in lastday.values()) == Decimal('33.4') and any('皆於 33.4 天內停止追蹤' in t for t in got), str(lastday))
cr = [Decimal(dres[s][0]) for s in sites]; co = [Decimal(dres[s][1]) for s in sites]
rngt = '穿越比例為 %s%%–%s%%' % (min(cr), max(cr)); rngc = '海岸來源比例為 %s%%–%s%%' % (min(co), max(co))
chk('N4 綜合說明之比例範圍 = 五地點最小最大', any(rngt in t and rngc in t for t in got), rngt + ' ' + rngc)
chk('N5 整體結論第 6 條範圍與綜合說明相同', any(t.startswith('（6）') and rngc in t for t in got))
concl = {s: next(t for t in got if t.startswith('（%d）%s：' % (k + 1, s)) and len(t) > 40) for k, s in enumerate(sites)}
okc = True; det = []
for s in sites:
    m = re.search(r'全年主要來源方向為(.+?)，機率為 ([\d.]+)%，其次為(\S+?) ([\d.]+)%', concl[s])
    if not m or m.groups() != top[s]: okc = False; det.append(s)
    m2 = re.search(r'海岸來源占 ([\d.]+)%', concl[s])
    if not m2 or m2.group(1) != dres[s][1]: okc = False; det.append(s + '海岸來源')
chk_old('N6 整體結論各地點：主要／次要來源方向與機率、海岸來源 = 各地點段', okc, str(det))
mx = [Decimal(0)] if V8 else [Decimal(top[s][1]) for s in sites]
chk_old('N7 整體結論第 6 條最主要來源方向機率範圍 = 五地點最小最大', any(t.startswith('（6）') and '機率為 %s%%–%s%%' % (min(mx), max(mx)) in t for t in got), '%s–%s' % (min(mx), max(mx)))
def _sub_ok(s, tot, parts):
    if Decimal(tot) != Decimal(dres[s][1]): return False
    if V8 and s == '龜山島':   # v8b：不再寫其他島嶼之數值（「其餘為其他島嶼」）；餘數須落在主檔原值 0.13 之進位誤差內（0.10–0.15）
        return Decimal('0.10') <= Decimal(tot) - sum(Decimal(x) for x in parts) <= Decimal('0.15') and any('龜山島占 9.57%，其餘為其他島嶼' in t for t in got)
    return sum(Decimal(x) for x in parts) == Decimal(tot)
bs = [s for s, (tot, parts) in sub.items() if not _sub_ok(s, tot, parts)]
chk('N8 海岸來源細項加總 = 該地點海岸來源比例（%d 地點）' % len(sub), not bs and len(sub) == 4, str(bs))
ar = sorted(Decimal(a.replace(',', '')) for a in area.values())
chk('N9 綜合說明 50% 高密度區間面積範圍 = 各地點最小最大', any('面積為 %s–%s km²' % (f'{int(ar[0]):,}', f'{int(ar[-1]):,}') in t for t in got), str(area))
dv = [Decimal(v) for v in depth.values() if v is not None] + [Decimal('100.00')] * sum(1 for v in depth.values() if v is None)
chk('N10 水深淺於 200 m 之比例範圍 = 各地點最小最大', any('%s%%–100.00%%' % min(dv) in t for t in got), str(depth))
chk('N11 顆數：20 × 50 × 10 = 10,000（每類）；× 10 類 = 100,000；× 5 地點 = 500,000，與 3-4-3 每類 10,000 顆一致', 20 * 50 * 10 == 10000 and '每類各有 10,000 顆' in n343 and '100,000' in got[1] and '500,000' in got[1])
chk('N12 3-4-3 敘述之 20 種到達地點、50 種到達時間、10 類、五個研究地點確實存在', all(k in n343 for k in ['設定 20 種不同', '設定 50 種不同', '10 類', '五個研究地點']))

# ---------------------------------------------------------------- v3：停止邊界段不含結果數字、「上升」處理、成果段定義齊全
item4 = next(t for t in alltext if t.startswith('4、溯源停止邊界'))
item1 = next(t for t in alltext if t.startswith('1、沉降動力參數'))
para3 = next(t for t in alltext if t.startswith('十種項目對應'))
chk('X1 停止邊界段不含任何結果數字（無百分比、無 33.4、無各地點比例）', not re.search(r'\d+(\.\d+)?%|33\.4|穿越比例|海岸來源比例|貢寮|龜山島|新竹外海|南灣|連江', item4), item4[-80:])
chk('X2 停止邊界段保留停止條件定義（穿越流場範圍邊界或抵達海岸即停止追蹤；稱為海岸來源）；使用者於 v6b 後手刪「回溯期間上限為 180 天」，段末為「即停止追蹤。」',
    '虛擬質點穿越流場範圍邊界或抵達海岸（視為找到來源，稱為海岸來源）即停止追蹤' in item4 and item4.rstrip().endswith('即停止追蹤。') and '180' not in item4, item4[-60:])
chk('X3 第 1 項標題已無「上升」（使用者手改）、「假設」改「設定 10 種不同的沉降速度」（v5b）', '設定 10 種不同的沉降速度' in item1 and '假設' not in item1 and '上升' not in item1)
chk('X4 第 1 項「各類速度均為沉降速度。」且無「不含上升情境」', '各類速度均為沉降速度。每類由' in para3 and '上升情境' not in para3)
up = [(t[max(0, m.start() - 2):m.end() + 4]) for t in alltext for m in re.finditer('上升|上浮|浮升|上漂', t)]
chk('X5 全文「上升」等字眼僅餘兩處「潮位上升」（潮汐描述，與沉降速度無關）', len(up) == 2 and all(x.startswith('潮位上升') for x in up), str(up))
rt = '\n'.join(got)
chk('X6 成果部分有「穿越比例」「海岸來源比例」之定義（貢寮圖 9 導引段）', '稱為穿越比例' in rt and '稱為海岸來源比例' in rt and '貢寮之虛擬質點穿越流場範圍邊界者占 73.40%' in rt)
chk('X7 成果部分有 33.4 天（連江圖 37；綜合說明）與各地點比例（五地點圖導引段）', '至第 33.4 天降至 0%' in rt and '皆於 33.4 天內停止追蹤' in rt and all(('穿越比例為 %s%%' % PUB[s][0]) in rt or s == '貢寮' for s in sites))

# ---------------------------------------------------------------- 百分比、術語
pct_bad = []
for t in got:
    for m in re.finditer(r'(\d+(?:\.\d+)?)%', t):
        v = m.group(1)
        if '.' in v:
            if not (len(v.split('.')[1]) == 2 or v == '0.001'): pct_bad.append(v)
        elif v not in ('0', '1', '10', '50', '75', '90', '100'): pct_bad.append(v)
chk('E1 成果段百分比小數位數：2 位（0.001% 例外）；整數僅限 0、1、10、50、75、90、100', not pct_bad, str(pct_bad[:5]))
forb = ['各地點', '到達時刻', '沉底時間', '流場邊界', '穿越點', '反射', '取樣誤差', '逐時取樣', '接收點', '錨定點', '海岸源', '觸地', '著底', '受點', '研究站', '研究點']
fl = [f for f in forb if any(f in t.replace('各研究地點', '') for t in got)]
chk('E2 成果段無禁用／異形用語', not fl, str(fl))
fl3 = [f for f in forb if any(f in t.replace('各研究地點', '') for t in [ptext(e) for e in head if e.tag == q('p')])]
chk('E3 3-4-3 情境段亦無上列異形用語', not fl3, str(fl3))
req = ['到達時間', '沉底時刻', '流場範圍邊界', '海岸來源', '觸底', '受體', '研究地點', '錨點']
allres = '\n'.join(got + n343.split('\n'))
chk('E4 八個統一術語皆出現於合併文件', all(r in allres for r in req), str([r for r in req if r not in allres]))
chk('E5 成果段不得含「A/A2/D 定義」或「反射海岸」', not re.search(r'反射|A2|方法 ?[ABD]', rt))
chk('E6 無表情符號', not re.search('[\U0001F300-\U0001FAFF☀-➿]', '\n'.join(alltext)))
chk('E7 「臺灣本島」用語', not any(('台灣本島' in t or '臺灣島' in t) for t in alltext))

# ---------------------------------------------------------------- 分頁保持與大綱層級
def rpr_has(p, name):
    ps = p.find(q('pPr')); return ps is not None and ps.find(q(name)) is not None
def outl(p):
    ps = p.find(q('pPr')); e = ps.find(q('outlineLvl')) if ps is not None else None
    return int(e.get(q('val'))) if e is not None else None
L0 = {'Lagrangian系集逆向溯源'}; L1 = {'（一）情境矩陣與停止條件設計：', '（二）分析成果：'}
L2 = {'逐地點成果：', '綜合說明與結論：', '整體結論：', '推導流程'}
paras = list(enumerate(kids))
apx_n = next(n for n, e in paras if e.tag == q('p') and ptext(e) == '附錄　十類海底廢棄物終端沉降速度之推導')
exp_lvl = {}
for n, e in paras:
    if e.tag != q('p') or figs_in(e): continue
    t = ptext(e)
    if n == 0 and t in L0: exp_lvl[n] = (0, True)
    elif t in L1: exp_lvl[n] = (1, True)
    elif t in L2: exp_lvl[n] = (2, True)
    elif n > apx_n and re.match(r'^([一二三四五六七八]、\S{2,20}|補充[一二]、\S{2,25})$', t): exp_lvl[n] = (2, True)
    elif re.match(r'^(貢寮|龜山島|新竹外海|南灣|連江)：$', t) and n < apx_n: exp_lvl[n] = (3, True)
    elif n > apx_n and re.match(r'^步驟[一二三四五]　\S.{2,40}$', t): exp_lvl[n] = (3, True)
    elif t in ('參考資料', '附錄　十類海底廢棄物終端沉降速度之推導'): exp_lvl[n] = (1, False)
chk('K0 標題段共 29 個', len(exp_lvl) == 29, str(len(exp_lvl)))
chk('K1 全部標題段皆有 keepNext 與 keepLines', all(rpr_has(kids[n], 'keepNext') and rpr_has(kids[n], 'keepLines') for n in exp_lvl))
chk('O1 直接大綱層級值與預期相符（0/1/2/3；27 個直接設定）', all(outl(kids[n]) == (lv if d else None) for n, (lv, d) in exp_lvl.items()) and sum(1 for n, (lv, d) in exp_lvl.items() if d) == 27)
stz = etree.fromstring(zm.read('word/styles.xml'))
st21 = next(s for s in stz.findall(q('style')) if s.get(q('styleId')) == '21')
chk('O2 參考資料與附錄維持 Heading 2 樣式（outlineLvl=1），未加直接層級', st21.find(q('pPr')).find(q('outlineLvl')).get(q('val')) == '1' and all(pstyle(kids[n]) == '21' for n, (lv, d) in exp_lvl.items() if not d))
styled = {s.get(q('styleId')) for s in stz.findall(q('style')) if s.find(q('pPr')) is not None and s.find(q('pPr')).find(q('outlineLvl')) is not None}
others = [ptext(e)[:15] for n, e in paras if e.tag == q('p') and n not in exp_lvl and ptext(e).strip() and (outl(e) is not None or pstyle(e) in styled)]
chk('O3 除上列標題外，無其他有文字之段落有大綱層級', not others, str(others[:5]))
seq = [(exp_lvl[n][0], ptext(kids[n])) for n in sorted(exp_lvl)]
chk('O4 大綱層級序列不跳級', all(b[0] <= a[0] + 1 for a, b in zip(seq, seq[1:])) and seq[0][0] == 0)
open(os.path.join(HERE, 'outline_tree.txt'), 'w', encoding='utf-8').write('\n'.join('%s%s' % ('    ' * lv, t) for lv, t in seq) + '\n')
tcn = [n for n, e in paras if e.tag == q('p') and re.match(r'^表(A?\d+) ', ptext(e)) and n + 1 < len(kids) and kids[n + 1].tag == q('tbl')]
chk('K2 表標題（緊接表格者）皆有 keepNext 與 keepLines（14 個）', len(tcn) == 14 and all(rpr_has(kids[n], 'keepNext') and rpr_has(kids[n], 'keepLines') for n in tcn))
fgn = [n for n, e in paras if e.tag == q('p') and figs_in(e)]
chk_old('K3 全部 42 個圖段皆有 keepNext', len(fgn) == 42 and all(rpr_has(kids[n], 'keepNext') for n in fgn))
LIM = (16838 - 2880) / 20.0 - 17.0
def nl(text, wpt):
    ln, w = 1, 0.0
    for ch in text:
        cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
        if w + cw > wpt: ln += 1; w = cw
        else: w += cw
    return ln
intro_n, fitflags = [], {}
for n in fgn:
    j2 = n - 1; pe = kids[j2]
    if pe.tag != q('p') or figs_in(pe) or j2 in exp_lvl or re.match(r'^(圖|表)(A?\d+) ', ptext(pe)) or not ptext(pe).strip(): continue
    chain = []; h = j2 - 1
    while h >= 0 and h in exp_lvl: chain.append(h); h -= 1
    capn = kids[n + 1] if n + 1 < len(kids) and kids[n + 1].tag == q('p') and re.match(r'^圖\d+ ', ptext(kids[n + 1])) else None
    bw = (11906 - 2880 - 991) / 20.0; cw_ = (11906 - 2880) / 20.0
    tot = sum(nl(ptext(kids[x]), bw) * 22 + 18 for x in chain) + nl(ptext(pe), bw) * 22 + 18 + 12 + figs_in(kids[n])[0][1] / 12700.0 + ((2 + nl(ptext(capn), cw_) * 27 + 8) if capn is not None else 0)
    intro_n.append(j2); fitflags[j2] = tot <= LIM
chk_old('K4 圖前導引段共 36 個皆有 keepLines', len(intro_n) == 36 and all(rpr_has(kids[x], 'keepLines') for x in intro_n), str(len(intro_n)))
chk_old('K5 導引段 keepNext：放得下者有，放不下者無（22 有／14 無）', all(rpr_has(kids[x], 'keepNext') == fitflags[x] for x in intro_n) and sum(fitflags.values()) == 22)
plan = json.load(open(os.path.join(HERE, 'keep_plan.json'), encoding='utf-8'))
chk_old('K6 與 keep_plan.json 逐項一致', [x['fits'] for x in plan['intros']] == [fitflags[x] for x in intro_n])

# ---------------------------------------------------------------- v4：用語統一、刪減句、格式
def full_text():
    out = []
    for e in kids:
        out.append(''.join(t.text or '' for t in e.iter(q('t'))))
    return '\n'.join(out)
FT = full_text()
chk('W1 全文（含表格）無「垂向」「代理」', not any(w in FT for w in ('垂向', '代理')), str([w for w in ('垂向', '代理') if w in FT]))
hai = [FT[max(0, m.start() - 8):m.end() + 6] for m in re.finditer('海廢', FT)]
chk('W2 「海廢」僅餘 2 處（指引所列「十個海廢分類名稱」之原名；第 4 項「精確鎖定海廢移入」照計畫書原文，為 #4 之例外）；「海底廢」後皆接「棄物」；「海漂底廢棄物調查指引」保留 2 處（第 1 項段與參考資料）',
    len(hai) == 2 and any('海廢分類名稱' in h for h in hai) and any('鎖定海廢移入' in h for h in hai) and all(FT[m.end():m.end() + 2] == '棄物' for m in re.finditer('海底廢', FT)) and FT.count('海漂底廢棄物調查指引') == 2, str(hai))
chk('W3 「海底廢棄物」共 7 處；v6c：全文無「本研究」、「本計畫」共 21 處（含第 4 項、附錄、表）；v6e：刪 3 處（（一）1 末段、附錄三「參照其兩極限」、附錄五「尚無實測，故以」），餘 18 處', FT.count('海底廢棄物') == 7 and FT.count('本研究') == 0 and FT.count('本計畫') == 18, '%d %d %d' % (FT.count('海底廢棄物'), FT.count('本研究'), FT.count('本計畫')))
chk('W17 v6c：全文（含表格、圖說、附錄）無「本研究」；本文 w:t 中亦無跨 run 殘留', '本研究' not in FT and '本研究' not in ''.join(t.text or '' for t in rootM.iter(q('t'))))
f9 = next(t for t in alltext if t.startswith('圖 9 為全部 100,000 顆虛擬質點') or (V10 and t.startswith('圖 9 為十類虛擬質點')))   # v10：使用者核定改為「十類虛擬質點（每類 10,000 顆）各自」（V10-13 逐項斷言）
chk('W18 v6d：貢寮圖 9 導引段「回溯期間內仍在追蹤之比例。貢寮之虛擬質點穿越流場範圍邊界者占 73.40%」連續；全文無「垂直虛線」「標示 90 天」；「虛線」僅餘圖 2 圖說 1 處，v6e：全文已無「180 天」',
    (f9.startswith('圖 9 為全部 100,000 顆虛擬質點於回溯期間內仍在追蹤之比例。貢寮之虛擬質點穿越流場範圍邊界者占 73.40%，') or (V10 and f9.startswith('圖 9 為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。貢寮之虛擬質點穿越流場範圍邊界者占 73.40%，'))) and '垂直虛線' not in FT and '標示 90 天' not in FT and '180 天' not in FT and '180天' not in FT and '無回溯達' not in FT and FT.count('虛線') == 1, f9[:60])
chk('W19 v6e：五處改後句——（一）1 末段「各類 10 個速度係以受力平衡求解，」、附錄三「參照其兩極限相加之概念，黏滯項改採 Schiller–Naumann 式」「本計畫採用之薄板，依教科書（表 11-2）有限矩形板式為 1.14–1.18」、附錄五「尚無實測，故以表 A4 之有效密度作為設定值。」、綜合說明「南灣另有 1 顆觸底）。穿越比例為」',
    all(x in FT for x in ('各類 10 個速度係以受力平衡求解，', '。參照其兩極限相加之概念，黏滯項改採 Schiller–Naumann 式', '本計畫採用之薄板，依教科書（表 11-2）有限矩形板式為 1.14–1.18', '尚無實測，故以表 A4 之有效密度作為設定值。', '南灣另有 1 顆觸底）。穿越比例為')) and not any(x in FT for x in ('由本計畫以受力平衡', '本計畫參照其兩極限', '本計畫薄板', '本計畫以表 A4')))
lg = re.findall(r'Lagrangian(.)', FT)
chk('W4 「Lagrangian」與中文之間：3 處皆無空格（標題、第 2 項、成果導言），已一致，未改', len(lg) == 3 and all('\u4e00' <= c <= '\u9fff' for c in lg), str(lg))
item2 = next(t for t in alltext if t.startswith('共由「5 個水平位置'))
chk('W5 受體選取段：保留半徑 10 km、方位×距離各 5 等面積區間、區塊內隨機抽取；已無「濕潤」「逐時空間支援檢查」「遞補」',
    '受體位置限於全部到達時間皆有流場與波浪資料之海域；符合條件者再依方位及距研究地點錨點之距離' in item2 and '半徑 10 km 範圍內之候選海域等分為面積相同之 5 個區間' in item2 and item2.endswith('內隨機抽取。')
    and not any(w in FT for w in ('濕潤', '逐時空間支援檢查', '遞補')))
chk('W6 「4 個深度抽樣」；第 3 段「這 4 個深度的編號只是抽籤順序，不代表表層、中層或底層。」保留', '4 個深度抽樣」構成' in item2 and '這 4 個深度的編號只是抽籤順序，不代表表層、中層或底層。' in FT)
item3 = next(t for t in alltext if t.startswith('50 個到達時間由 48 個分層時刻'))
chk('W7 到達時間段：「三種潮相」「以潮位資料判定」；已刪「日間銜接處」句與「相互重疊」句',
    '四季 × 大潮／小潮 × 三種潮相」共 24 格' in item3 and '低於者為小潮。三種潮相以潮位資料判定，分別為潮位上升最快、潮位下降最快' in item3 and '日間銜接' not in FT and '相互重疊' not in FT)
t1 = next(e for e in kids if e.tag == q('tbl'))
okr = 0
for p_ in t1.iter(q('p')):
    rs = list(p_.iter(q('r')))
    for k, r_ in enumerate(rs):
        if ''.join(t.text or '' for t in r_.iter(q('t'))) == '至':
            if rs[k - 1].find(q('br')) is not None and ''.join(t.text or '' for t in rs[k + 1].iter(q('t'))).startswith(' '): okr += 1
chk('W8 表 1 速度範圍 10 列：「至」前為手動換行、後為半形空格（版面上兩側皆有分隔；v8 登記：使用者於 Word 重存後「至」後之 run 與數字合併為「 a.bc×10」，故改檢「至」後文字以半形空格起首），未改；表 A 系列之「至」皆前後半形空格', okr == 10 and not re.search(r'\d%至|至\+|\S至 [+−]', ''.join(ptext(e) for e in kids[ia:] if e.tag == q('tbl'))), str(okr))
chk('W10 #9：已無「未必涵蓋」，新句「用以納入分層設計以外之極端動力狀態」；#13 還原（v6b）：第 4 項為計畫書原文「藉此精確鎖定海廢移入該海域之潛在外部來源與邊界通量路徑。」',
    '未必涵蓋' not in FT and '用以納入分層設計以外之極端動力狀態' in FT and '藉此精確鎖定海廢移入該海域之潛在外部來源與邊界通量路徑。' in FT and '辨識海底廢棄物' not in FT)
chk('W11 #12 還原（依使用者決定，沉底時刻屬實作細節）：（一）3 不含「沉底」；全文「沉底年齡」恰 1 處（在成果段貢寮圖 13 導引段），無「0–90」（v5f 刪）',
    '沉底' not in item3 and item3.endswith('各潮相皆有代表時刻。') and FT.count('沉底年齡') == 1 and '0–90' not in FT)
r13 = next((t for t in alltext if t.startswith('圖 13 為依沉底時刻劃分之貢寮四季曲線') or t.startswith('圖 15 為依沉底時刻劃分之貢寮四季邊界')), '')
chk_old('W12 貢寮圖 13 導引段（v5e）：沉底時刻定義為隨機決定（每個系集成員各自隨機抽取沉底年齡，不寫天數範圍），其後「季節依沉底時刻所在月份劃分，冬季為 12–2 月…」維持，無「定義見（一）3」；不含「分層」「種子」',
    '圖 13 為依沉底時刻劃分之貢寮四季曲線。沉底時刻為廢棄物沉降至海床之時刻，亦即虛擬質點回溯之起點；由於廢棄物實際何時沉底未知，沉底時刻由所屬到達時間往前推一段隨機抽取之沉底年齡決定，每個系集成員各自抽取；季節依沉底時刻所在月份劃分，冬季為 12–2 月、春季為 3–5 月、夏季為 6–8 月、秋季為 9–11 月。' in r13 and '定義見（一）3' not in FT and '分層' not in r13 and '種子' not in r13)
_gc = [int(x.get(q('w'))) for x in t1.find(q('tblGrid')).findall(q('gridCol'))]
chk('W13 #17／v7：表 1 欄寬 %s（總寬 %d 不變）；第 4 欄 %d twips（可用 %.1f pt）放得下第二行「至 1.08×10⁻¹」（約 75 pt）但放不下一行「8.97×10⁻⁴ 至 1.08×10⁻¹」，故兩行、保留「至」前手動換行' % (_gc, sum(_gc), _gc[3], (_gc[3] - 240) / 20.0), _gc == GRID_T1 and sum(_gc) == 8457 and 75 < (_gc[3] - 240) / 20.0 < 130)
V6F_OLD = [('0.000897', '0.108'), ('0.00661', '0.162'), ('0.00527', '0.187'), ('0.00795', '0.223'), ('0.0293', '0.134'),
           ('0.0135', '0.303'), ('0.0342', '0.345'), ('0.0367', '0.175'), ('0.107', '0.427'), ('0.195', '0.920')]   # 改前十列（v6e 後之值）
_rows = t1.findall(q('tr'))[1:]; _bad = []; _fmt = set()
for _r, (_o1, _o2) in zip(_rows, V6F_OLD):
    _p = list(_r.findall(q('tc'))[3].iter(q('p')))[0]; _parts = []; _br = 0
    for _run in _p.findall(q('r')):
        _sup = _run.find(q('rPr')).find(q('vertAlign')) is not None and _run.find(q('rPr')).find(q('vertAlign')).get(q('val')) == 'superscript'
        if _run.find(q('br')) is not None: _br += 1; _parts.append(('br', '')); continue
        for _t in _run.findall(q('t')): _parts.append(('sup' if _sup else 'n', _t.text or ''))
    _mg = []
    for k, t in _parts:
        if _mg and _mg[-1][0] == k and k == 'sup': _mg[-1] = (k, _mg[-1][1] + t)
        else: _mg.append((k, t))
    _txt = ''.join(('^{%s}' % t) if k == 'sup' else ('|' if k == 'br' else t) for k, t in _mg)
    _m = re.fullmatch(r'(\d(?:\.\d+)?)×10\^\{(−?\d)\}\|至 (\d(?:\.\d+)?)×10\^\{(−?\d)\}', _txt)
    if not _m or _br != 1: _bad.append(('格式', _txt)); continue
    _fmt.add(re.sub(r'\d', '9', _txt))
    for (_mt, _ex), _o in ((_m.group(1, 2), _o1), (_m.group(3, 4), _o2)):
        if Decimal(_mt + 'E' + _ex.replace('−', '-')) != Decimal(_o) or len(_mt.replace('.', '')) != len(_o.replace('.', '').lstrip('0')): _bad.append(('數值', _txt, _o))
        if not (1 <= Decimal(_mt) < 10): _bad.append(('尾數', _mt))
chk('W20 v6f／v7：表 1 十列科學記號（v7 起不寫負號）——尾數×10 與上標指數（run 之 vertAlign=superscript，非 Unicode 上標字元），換算後逐一等於原值（Decimal），有效位數照原值（含 0.920→9.20），「至」前保留手動換行，十列格式一致',
    len(_rows) == 10 and not _bad and len(_fmt) <= 3 and not any(c in ''.join(ptext(e) for e in [t1]) for c in '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'), str(_bad) + str(_fmt))
chk('W14 （二）內小標已不編號（無「1、逐地點成果」「（1）貢寮：」等），整體結論之條列（1）–（6）保留編號；（一）項目 1–4 與附錄編號不動',
    not any(w in FT for w in ('1、逐地點成果', '2、綜合說明與結論', '3、整體結論')) and not any(('\n' + '（%d）%s：\n' % (i + 1, s_)) in FT for i, s_ in enumerate(sites))
    and all(any(t.startswith('（%d）%s：' % (i + 1, s_)) and len(t) > 40 for t in alltext) for i, s_ in enumerate(sites)) and any(t.startswith('（6）綜合五個研究地點') for t in alltext)
    and all(any(t.startswith('%d、' % k) for t in alltext) for k in (1, 2, 3, 4)))
_fig_intro = [ptext(kids[n - 1]) for n in fgn if n > ia and kids[n - 1].tag == q('p') and ptext(kids[n - 1]).startswith('圖 ')]
_res = [bool(re.search(r'\d+(\.\d+)?%|相近|一致', t)) for t in _fig_intro]
chk_old('W15 （二）35 張圖之導引段皆含結果句（百分比、「相近」或「一致」）', len(_fig_intro) == 35 and all(_res), '%d %d' % (len(_fig_intro), sum(_res)))
_traj = [t for t in _fig_intro if '隨機抽取之 24 條' in t or '隨機抽取之24條' in t or re.match(r'^圖 \d+ 為.*隨機抽取之 24 條', t)]
chk_old('W16 五張軌跡圖導引段各有一句結果（延伸／穿越方向、抵達海岸，並呼應各地點十類來源方向圖：圖 11、18、25、32、39 之主要來源方向）；不含條數',
    len(_traj) == 5 and all(('與圖 %d 之主要來源方向' % k) in t for t, k in zip(_traj, (11, 18, 25, 32, 39))) and not any(re.search(r'\d+ 條中|其中 \d+ 條', t) for t in _traj) and all('延伸' in t and '一致' in t for t in _traj), str(len(_traj)))
bk = os.path.join(BASE, '成果報告_run09_合併版.docx.bak03-20261025')
if os.path.exists(bk):
    zb3 = zipfile.ZipFile(bk)
    chk_old('W9 與 v3 備份（bak03）相比：除 word/document.xml 與使用者 Word 重存所改之 settings.xml、docProps/core.xml、docProps/app.xml 外，全部 zip 成員逐位元組相同（%d 個）' % (len(zm.namelist()) - 4), zm.namelist() == zb3.namelist() and all(zm.read(n) == zb3.read(n) for n in zm.namelist() if n not in ('word/document.xml', 'word/settings.xml', 'docProps/core.xml', 'docProps/app.xml')))

# ---------------------------------------------------------------- v7 新增核對
BAK16 = os.path.join(BASE, '成果報告_run09_合併版.docx.bak16-20261027')
if os.path.exists(BAK16):
    _zb, _rb, _kb, _rl = load(BAK16)
    def marked(e):
        out = []
        for p_ in e.iter(q('p')):
            buf = []
            for r_ in p_.findall(q('r')):
                rp_ = r_.find(q('rPr')); va_ = rp_.find(q('vertAlign')) if rp_ is not None else None
                sup_ = va_ is not None and va_.get(q('val')) == 'superscript'
                for c_ in r_:
                    if c_.tag == q('t'): buf.append(('s' if sup_ else 'n', c_.text or ''))
                    elif c_.tag == q('br'): buf.append(('n', '⏎'))
            mg = []
            for k_, t_ in buf:
                if mg and mg[-1][0] == k_ == 's': mg[-1] = (k_, mg[-1][1] + t_)
                else: mg.append((k_, t_))
            out.append(''.join('^{%s}' % t_ if k_ == 's' else t_ for k_, t_ in mg))
        return '\n'.join(out)
    def rev(t): return re.sub(r'(\d(?:\.\d+)?)×10\^\{−(\d+)\}', lambda m: format(Decimal(m.group(1) + 'E-' + m.group(2)), 'f'), t)
    diffs = []
    chk_old('W21a v7：區塊數與 v6f 相同（%d）' % len(_kb), len(_kb) == len(kids))
    for n_, (eb, em) in enumerate(zip(_kb, kids)):
        mb, mm = marked(eb), marked(em)
        nb, nm = rev(mb), rev(mm)
        if n_ == i_tbl1: nb = re.sub(r'−(?=0\.)', '', nb)
        for anchor, dele, allowed, n_exp in V7_P:
            if dele is not None and nb.startswith(anchor): nb = nb.replace(dele, '')
        if nb != nm: diffs.append((n_, nb[:50], nm[:50]))
    nsup_m = len(re.findall(r'×10\^\{−\d+\}', ''.join(marked(e) for e in kids))); nsup_b = len(re.findall(r'×10\^\{−\d+\}', ''.join(marked(e) for e in _kb)))
    mant = [Decimal(m_) for m_ in re.findall(r'(\d(?:\.\d+)?)×10\^\{−\d+\}', ''.join(marked(e) for e in kids))]
    chk_old('W21 v7：還原（尾數×10^−n → 定點小數，位數照寫法）後，全文逐區塊等於 v6f（僅差表 1 去負號與兩句符號說明之刪除）；新增科學記號 %d 處（表 A1 20、A4 100、A9 10、A10 14、A11 4、A12 10、正文 11）、上標 run 總數 %d；尾數皆在 [1,10)' % (nsup_m - nsup_b, nsup_m),
        not diffs and nsup_m - nsup_b == 169 and nsup_b == 20 and all(1 <= m_ < 10 for m_ in mant), str(diffs[:3]))
else:
    chk('W21 v7（缺 bak16，略過）', True)
negpat = [r'[−\-]\s?0\.\d+(?![\d%])', r'[−\-]\s?\d(?:\.\d+)?×10', '取負值', '均為負值', '垂直座標向上為正', '向上為正']
chk('W22 v7：全文（含表格；數學式內不計）無帶負號之沉降速度（−0.xxx、−a.bc×10⁻ⁿ），無「取負值」「均為負值」「垂直座標向上為正」', not any(re.search(pp, FT) for pp in negpat), str([pp for pp in negpat if re.search(pp, FT)]))
_t10 = next(e for n_, e in enumerate(kids) if e.tag == q('tbl') and ptext(kids[n_ - 1]).startswith('表A10 '))
def _grid(t_): return [int(x.get(q('w'))) for x in t_.find(q('tblGrid')).findall(q('gridCol'))]
def _tcw_ok(t_): return all([int(tc.find(q('tcPr')).find(q('tcW')).get(q('w'))) for tc in tr.findall(q('tc'))] == _grid(t_) for tr in t_.findall(q('tr')))
chk('W23 v7：表 1 欄寬 %s、表 A10 欄寬 %s（各自總寬 8457 不變）；各列 tcW 與 tblGrid 一致；表 1 十個資料列皆 cantSplit（列不跨頁）' % (_grid(t1), _grid(_t10)),
    _grid(t1) == GRID_T1 and _grid(_t10) == GRID_A10 and sum(_grid(t1)) == sum(_grid(_t10)) == 8457 and _tcw_ok(t1) and _tcw_ok(_t10)
    and sum(1 for tr in t1.findall(q('tr'))[1:] if tr.find(q('trPr')) is not None and tr.find(q('trPr')).find(q('cantSplit')) is not None) == 10)

# ---------------------------------------------------------------- PDF
if PDF and os.path.exists(PDF):
    pages = subprocess.run(['pdfinfo', PDF], capture_output=True, text=True).stdout
    npg = int(re.search(r'Pages:\s+(\d+)', pages).group(1))
    def pg_of(s_, start=1):
        for p_ in range(start, npg + 1):
            if s_ in subprocess.run(['pdftotext', '-f', str(p_), '-l', str(p_), '-layout', PDF, '-'], capture_output=True, text=True).stdout.replace(' ', ''): return p_
    pa, pb, pc, pd = pg_of('（二）分析成果'), pg_of('整體結論：'), pg_of('參考資料', 40), pg_of('附錄十類海底廢棄物', 40)
    chk('D1 PDF 頁序：（二）< 整體結論 < 參考資料 < 附錄（%s<%s<%s<%s，共 %d 頁）' % (pa, pb, pc, pd, npg), None not in (pa, pb, pc, pd) and pa < pb < pc < pd)
    txt = subprocess.run(['pdftotext', '-layout', PDF, '-'], capture_output=True, text=True).stdout
    _p13 = subprocess.run(['pdftotext', '-f', '1', '-l', '3', '-layout', PDF, '-'], capture_output=True, text=True).stdout
    _pdfr = [l_ for l_ in _p13.split('\n')]
    chk('D4 v7：PDF 表 1 速度欄每列兩行——「a.bc×10⁻ⁿ」共 20 處、其中「至 a.bc×10⁻ⁿ」10 處（第二行），無單獨成行之「至」',
        len(re.findall(r'(?<![\d.])\d\.\d{2}×10−\d', _p13)) == 20 and len(re.findall(r'至\s+\d\.\d{2}×10−\d', _p13)) == 10 and not any(l_.strip() == '至' for l_ in _pdfr))
    chk_old('D2 PDF 圖說 圖1–圖42 皆出現', all(re.search(r'圖 ?%d ' % n, txt) for n in range(1, 43)))
    st4 = txt.replace('\n', '').replace(' ', '')
    chk('D3 PDF 中停止邊界段結尾為「即停止追蹤。」且其後無「33.4」', '即停止追蹤。' in st4 and '即停止追蹤。33.4' not in st4)

# ---------------------------------------------------------------- v8 新增與重新登記之檢查
if V8:
    import csv, struct
    SEADEF_M = '沉底時刻為廢棄物沉降至海床之時刻，亦即虛擬質點回溯之起點；由於廢棄物實際何時沉底未知，沉底時刻由所屬到達時間往前推一段隨機抽取之沉底年齡決定，每個系集成員各自抽取；季節依沉底時刻所在月份劃分，冬季為 12–2 月、春季為 3–5 月、夏季為 6–8 月、秋季為 9–11 月。'
    LOG = json.load(open(os.path.join(HERE, 'v10_build_log.json' if V10 else ('v9_build_log.json' if V9 else 'v8_build_log.json')), encoding='utf-8'))
    SEG = os.path.join(BASE, '_verify', 'preview', 'segprob_v3')
    MAN = json.load(open(os.path.join(SEG, 'manifest.json'), encoding='utf-8'))
    SITEK = ['gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang']
    FINAL9 = os.path.join(BASE, '_verify', 'preview', 'segprob_fullpage', 'final')
    FIGK = ['boundary_all', 'coast_all', 'boundary_class', 'coast_class', 'boundary_season', 'coast_season', 'boundary_tide', 'coast_tide']
    KEYS8 = ['all_b', 'all_c', 'cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c']
    BASE8 = [8, 19, 30, 41, 52]
    NEWFIG = {b_ + 3 + j_ for b_ in BASE8 for j_ in range(8)}
    rd = lambda f: list(csv.DictReader(open(os.path.join(SEG, 'tables', f), encoding='utf-8-sig')))
    AS = rd('annotated_stretches.csv'); GTR = {(r['site'], r['group']): r for r in rd('group_totals.csv')}
    VER = list(csv.DictReader(open(os.path.join(BASE, '_verify', 'preview', 'segprob_v2', 'tables', 'diff_verdicts_final.csv'), encoding='utf-8-sig')))
    # ---- M3 候選 = 主檔 ＋ 登記之 v8 變更（保留區塊逐字、逐格式、逐圖片相同；新增區塊以下列檢查把關）
    sgC = [sig(e, zm, relsM) for e in kids]
    POLB = [(46, 58, 25), (69, 80, 24), (91, 102, 24), (113, 124, 24), (135, 146, 24)]   # 主檔圓形圖區段（含頭尾）、候選新增區塊數（貢寮含方法段）
    REWR = {33, 34, 148, 152, 153, 154, 155, 156, 157}
    TRIP = ((37, 8, 8), (60, 15, 19), (82, 22, 30), (104, 29, 41), (126, 36, 52))
    def v8_expect(i, sg):
        for t0, o0, n0 in TRIP:
            for j in range(3):
                if o0 != n0 and i == t0 + 3 * j: sg = (sg[0], sg[1], mod_chars(sg[2], '圖 %d ' % (o0 + j), '圖 %d ' % (n0 + j)), sg[3])
                if o0 != n0 and i == t0 + 3 * j + 2: sg = (sg[0], sg[1], mod_chars(sg[2], '圖%d ' % (o0 + j), '圖%d ' % (n0 + j)), sg[3])
            if i == t0:
                for a_, o_, n_ in V8_TAIL:
                    if ptext(kmast[i]).startswith(a_): sg = (sg[0], sg[1], mod_chars(sg[2], o_, n_), sg[3])
        return sg
    skip = set(); starts = {}
    for a_, b_, n_ in POLB: skip |= set(range(a_, b_ + 1)); starts[a_] = n_
    exp8 = []
    for i, sg in enumerate(sgM):
        if i in starts: exp8 += [None] * starts[i]
        if i in skip: continue
        exp8.append(('P*', sg[1]) if i in REWR else v8_expect(i, sg))
    CURVE_IDX = {41, 76, 110, 144, 178}   # v10：5 張仍在追蹤比例圖（圖檔已由 B4 新圖取代，圖之內容另由 V10-13 查）
    def _m3_bad(j, e_, c_):
        if e_ is None: return False
        if e_[0] == 'P*': return c_[0] != 'p' or c_[1] != e_[1]
        if V10 and j in CURVE_IDX: return not (c_[0] == 'p' and e_[0] == 'p' and c_[1] == e_[1] and c_[2] == e_[2] and len(c_[3]) == 1 and len(e_[3]) == 1)
        return e_ != c_
    bad8 = [j for j, (e_, c_) in enumerate(zip(exp8, sgC)) if _m3_bad(j, e_, c_)]
    chk('M3 v8：候選 ＝ 主檔 ＋ 登記之變更（移除 61 區塊〔圓形圖 20 張及其導引段、圖說、貢寮方法段〕、新增 121 區塊、保留圖與軌跡導引段改號、改寫 9 段）；其餘 %d 個區塊逐字、逐格式、逐圖片相同' % sum(1 for e_ in exp8 if e_ is not None and e_[0] != 'P*'),
        len(exp8) == len(sgC) == 359 and not bad8, str(bad8[:8]))
    # ---- 圖檔
    kept_idx = [(7 + 11 * s_ + j_, 7 * s_ + j_) for s_ in range(5) for j_ in range(3) if not (V10 and LOG['v10']['curves'] and j_ == 1)]   # v10：選項 B 後，5 張仍在追蹤比例圖（j_ = 1）改由 V10 專項檢查
    chk('F4 v8：保留之 15 張圖（v10 若已換入選項 B 曲線圖，則為 10 張）（各地點軌跡、仍在追蹤、足跡）圖檔 sha 與執行成果相同', len(imgsB) == 35 and len(kept_idx) in (10, 15) and all(sha(imgsM[a_][1]) == sha(imgsB[b_][1]) for a_, b_ in kept_idx))
    chk('F5 v8：保留之 15 張顯示尺寸（v10 同上）與執行成果相同', all(imgsM[a_][0][:2] == imgsB[b_][0][:2] for a_, b_ in kept_idx))
    okF = []; detF = []
    for s_ in range(5):
        for kk, fk in enumerate(FIGK):
            f_, data_ = imgsM[7 + 11 * s_ + 3 + kk]; ent = MAN['figures'][SITEK[s_]][fk]
            w_, h_ = struct.unpack('>II', data_[16:24])
            if V9 and fk not in ('boundary_all', 'coast_all'):   # v9：30 張多格圖取自 segprob_fullpage/final（核准版面）
                disk = open(os.path.join(FINAL9, SITEK[s_], fk + '.png'), 'rb').read(); ent_sha = sha(disk)
            else:
                disk = open(os.path.join(SEG, SITEK[s_], ent['png']), 'rb').read(); ent_sha = ent['sha256']
            ok_ = data_[:8] == b'\x89PNG\r\n\x1a\n' and sha(data_) == ent_sha == sha(disk) and f_[0] == 5733288 and f_[1] == round(h_ / 600.0 * 914400) and w_ == 3762
            okF.append(ok_)
            if not ok_: detF.append((SITEK[s_], fk))
    chk('F10 v8：新增 40 張圖之 sha 等於 segprob_v3 manifest 與磁碟檔（來源分段機率圖），寬 6.27 in（5733288 EMU、3762 px/600 dpi），高依像素等比，未縮放；順序：全年邊界、全年海岸、十類邊界、十類海岸、季節邊界、季節海岸、大小潮邊界、大小潮海岸', len(okF) == 40 and all(okF), str(detF[:4]))
    nmbad = []
    for n, p in enumerate(kids):
        if p.tag == q('p') and figs_in(p) and n > ia:
            dp = next(p.iter('{%s}docPr' % WP)); cn = int(re.match(r'^圖(\d+) ', ptext(kids[n + 1])).group(1))
            if not (re.match(r'^fig%d\.(jpg|png)$' % cn, dp.get('name')) and dp.get('descr').startswith('圖%d ' % cn)): nmbad.append((cn, dp.get('name')))
    chk('F11 v8：成果段每張圖（55 張）之 docPr 名稱與說明與圖說編號一致', not nmbad, str(nmbad[:3]))
    chk('F1 v8：圖說編號 1–62 連續且依序', capnums == list(range(1, 63)), str(capnums))
    chk('H3 v8：每個研究地點各 11 張圖', nf == [11] * 5, str(nf))
    chk('P9 v8：docPr id 全文唯一（62 張圖）', len(ids) == len(set(ids)) == 62)
    chk('P11 v8：62 張圖各引用一個圖檔，無共用', len({relsM[f[2]] for _, f in allfig}) == 62)
    chk('P12 v8：zip 無孤立之圖檔（word/media 內檔案 = 關係所引用者）、被移除之 20 張舊圖檔已不在包內', {n for n in zm.namelist() if n.startswith('word/media/')} == {'word/' + t for t in relsM.values() if t.startswith('media/')} and (V10 or not (set(LOG['dropped_media']) & set(zm.namelist()))))
    mzs = [n for n in zm.namelist() if not n.startswith('word/media/') and n not in ('word/document.xml', 'word/_rels/document.xml.rels')]
    chk('W9 v8：與主檔相比，除 word/document.xml、document.xml.rels 與 word/media（移除 20、新增 40）外，其餘 zip 成員逐位元組相同（%d 個）' % len(mzs), all(zm.read(n) == zmast.read(n) for n in mzs) and set(mzs) == {n for n in zmast.namelist() if not n.startswith('word/media/') and n not in ('word/document.xml', 'word/_rels/document.xml.rels')}) if not V10 else chk('W9 v10：與現行主檔（sha 20d83708…）相比，除 word/document.xml、document.xml.rels 與 word/media 外，其餘 zip 成員逐位元組相同（%d 個）；v8 之主檔（bak17）已被使用者 Word 重存取代，不再作為 zip 比對基準' % len(mzs), M10P is not None and all(zm.read(n) == zb10.read(n) for n in mzs) and set(mzs) == {n for n in zb10.namelist() if not n.startswith('word/media/') and n not in ('word/document.xml', 'word/_rels/document.xml.rels')})
    # ---- 圖號引用
    chk('R2 v8：成果段內圖號引用皆在 8–62', not refs_bad(ins, 8, 62))
    badr = []; nfound = 0
    for t in [ptext(p) for p in ins if p.tag == q('p')]:
        for m in re.finditer(r'圖 (\d+) 至圖 (\d+)', t):
            nfound += 1; a_, b_ = int(m.group(1)), int(m.group(2))
            if b_ - a_ != 7 or capnums[a_ - 1] != a_ or capnums[b_ - 1] != b_ or a_ != 11: badr.append((a_, b_))
    chk('R4 v8：「圖 a 至圖 b」僅貢寮方法段 1 處（圖 11 至圖 18，共 8 張）', not badr and nfound == 1, str(badr))
    chk('R5 v8：十類圖不再引用「圖例」順序（各格標題標明類別）', not any('圖例' in t for t in got))
    rbad = []
    for s_, nm in enumerate(sites):
        lo, hi = BASE8[s_], BASE8[s_] + 10
        for t in blocks[nm]:
            if re.match(r'^圖\d+ ', t): continue
            for m in re.finditer(r'圖 (\d+)', t):
                if not (lo <= int(m.group(1)) <= hi): rbad.append((nm, m.group(1)))
    chk('R6 v8：各地點段落之圖號引用皆在該地點之 11 張圖內（圖 %s）' % '、'.join('%d–%d' % (b_, b_ + 10) for b_ in BASE8), not rbad, str(rbad[:4]))
    chk('V2 v8：各類段落數：L2×1、L3×3、L4×5、條列×6、圖×55、圖說×55', (okfmt['L2'], okfmt['L3'], okfmt['L4'], okfmt['ITEM'], okfmt['FIG'], okfmt['CAP']) == (1, 3, 5, 6, 55, 55), str(okfmt))
    # ---- 分頁保持
    chk('K3 v8：全部 62 個圖段皆有 keepNext', len(fgn) == 62 and all(rpr_has(kids[n], 'keepNext') for n in fgn))
    chk('K4 v8：圖前導引段共 56 個（到達時間 1 + 55）皆有 keepLines', len(intro_n) == 56 and all(rpr_has(kids[x], 'keepLines') for x in intro_n), str(len(intro_n)))
    fl = [fitflags[x] for x in intro_n]
    chk('K5 v8：導引段 keepNext：放得下者有，放不下者無（有 %d／無 %d）' % (sum(fl), len(fl) - sum(fl)), all(rpr_has(kids[x], 'keepNext') == fitflags[x] for x in intro_n))
    kp = [fl[0] == plan['intros'][0]['fits']] + [fl[1 + 11 * s_ + j_] == plan['intros'][1 + 7 * s_ + j_]['fits'] for s_ in range(5) for j_ in range(3)]
    nw = [fl[1 + 11 * s_ + 3 + kk] == LOG['fig_map'][8 * s_ + kk]['fit'] for s_ in range(5) for kk in range(8)]
    chk('K6 v8：與舊 keep_plan.json（保留之 16 項）及 v8_build_log（新增 40 項）逐項一致', all(kp) and all(nw) and len(LOG['fig_map']) == 40, '%d %d' % (sum(kp), sum(nw)))
    # ---- 文字：圖說、方法、數值、判定
    texts_new = [c['new'] for c in LOG['changes'] if c['kind'] in ('new', 'rewrite')]
    GUI = {(c['site'], c['role'].split(':')[1]): c['new'] for c in LOG['changes'] if c['kind'] == 'new' and c['role'].startswith('guide:')}
    CAPT = {(c['site'], c['role'].split(':')[1]): c['new'] for c in LOG['changes'] if c['kind'] == 'new' and c['role'].startswith('caption:')}
    CAPREQ = {'all_b': ['邊界穿越來源分段機率（全年，十類合併）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例', '黑色括號標示機率集中之連續邊界段', '灰色括號標示其餘連續非零邊界段'],
              'all_c': ['海岸來源分段機率（全年，十類合併）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例'],
              'cls_b': ['邊界穿越來源分段機率（十類）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例'],
              'cls_c': ['海岸來源分段機率（十類）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例'],
              'sea_b': ['（依沉底時刻之季節）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例', '括號標示機率集中之連續邊界段', '圓圈數字', '「其餘」'],
              'sea_c': ['（依沉底時刻之季節）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例', '圓圈數字'],
              'tid_b': ['（依大小潮）', '邊界每 10 km 一段', '色階為各段占全部虛擬質點之比例', '括號標示機率集中之連續邊界段', '圓圈數字', '「其餘」'],
              'tid_c': ['（依大小潮）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例', '圓圈數字']}
    if V9 and not V10:
        for k_ in ('cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c'): CAPREQ[k_] = CAPREQ[k_] + ['數值框']
        CAPREQ['cls_b'] = CAPREQ['cls_b'] + ['括號標示機率集中之連續邊界段', '圓圈數字', '「其餘」']
    if V10:   # v10：圖說不寫位置、不寫「數值框」；多格圖之圈號用語依實際圈號（圖 38、40）由 V10-7 逐圖把關
        CAPREQ['cls_b'] = CAPREQ['cls_b'] + ['括號標示機率集中之連續邊界段', '圓圈數字', '「其餘」']
        CAPREQ['sea_c'] = ['（依沉底時刻之季節）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例']
        CAPREQ['tid_c'] = ['（依大小潮）', '海岸每 5 km 一段', '僅計流場範圍內海岸', '色階為各段占全部虛擬質點之比例']
    cbad = []
    for s_, nm in enumerate(sites):
        cs = [t for t in blocks[nm] if re.match(r'^圖\d+ ', t)]
        if len(cs) != 11: cbad.append((nm, len(cs))); continue
        for kk, k_ in enumerate(KEYS8):
            for ph in CAPREQ[k_]:
                if ph not in cs[3 + kk].replace(' ', ''): 
                    if ph.replace(' ', '') not in cs[3 + kk].replace(' ', ''): cbad.append((nm, k_, ph))
            if nm not in cs[3 + kk]: cbad.append((nm, k_, '地點名'))
    chk('C1 v8：新增 40 張圖之圖說含規定用語（邊界每 10 km 一段；海岸每 5 km 一段、僅計流場範圍內海岸；色階為各段占全部虛擬質點之比例；單張邊界圖之黑、灰色括號句；多格圖之圓圈數字與「其餘」說明），並標明研究地點', not cbad, str(cbad[:4]))
    thr = [t[:20] for t in texts_new if re.search(r'[≥≧≤]|門檻|閾值|至少 ?\d|(?<![\d.])[13]%|2 段|1%以上|3%以上', t)]
    chk('C2 v8：新增／改寫文字不寫括號門檻（1%、3%、≥2 段、≥1%）', not thr, str(thr[:3]))
    meth = LOG['changes'][0]['new']
    chk('C3 v8b：方法段含：邊界每 10 km／海岸每 5 km 分段、依穿越位置與首次接觸位置歸段、歸入邊界段／海岸段／觸底三者之一且只計入一次（回應重複計算之疑慮）、機率合計 100%；已刪『組間差異之判讀』說明（使用者先前明確移除；僅留備查）——不含「相近或差異明顯」「稱為相近」「抽樣而產生」「差異範圍」；無 360°、10° 一格、1/36、高於平均',
        all(x in meth for x in ['邊界每 10 km', '海岸每 5 km', '穿越位置', '首次接觸海岸之位置', '只歸入邊界段、海岸段與觸底三者之一，且只計入一次', '互不重疊', '合計為 100%']) and not re.search(r'360|10°|1/36|高於平均|平均值|相近|差異明顯|抽樣而產生|差異範圍|稱為相近|稱為差異', meth))
    # 數值：每個百分比必須出自 segprob_v3 之表（含地點內任一分組之括號值、分組合計、母版海岸子分項）
    nbad = []; gbad = []
    for s_, nm in enumerate(sites):
        sk = SITEK[s_]
        allowed_ = set()
        for r in AS:
            if r['site'] == sk: allowed_.add('%.2f' % float(r['pct']))
        for (ss, g_), r in GTR.items():
            if ss == sk: allowed_ |= {'%.2f' % float(r['crossed_pct']), '%.2f' % float(r['coast_pct'])}
        for fk in FIGK:
            f_ = MAN['figures'][sk][fk]
            for t_ in f_.get('panel_titles', []): allowed_ |= set('%.2f' % float(x) for x in re.findall(r'([\d.]+)%', t_))
            anns = f_['annotations']; anns = [anns] if isinstance(anns, dict) else anns
            for a_ in anns:
                for st in a_['stretches']: allowed_.add('%.2f' % float(st.get('shown_pct', st.get('pct'))))
            if f_.get('inset'):
                for st in f_['inset']['annotations']['stretches']: allowed_.add('%.2f' % float(st['pct']))
        allowed_ |= set(sub[nm][1]) if nm in sub else set()
        allowed_ |= {'100.00', '0.00'}
        for k_ in KEYS8:
            for v_ in re.findall(r'(\d+\.\d+)%', GUI[(sk, k_)].replace('<0.01%', '')):
                if v_ not in allowed_: nbad.append((nm, k_, v_))
        # 完整性：全年邊界之主要與其餘括號、全年海岸括號、各分組編號括號之機率皆須出現於對應導引段
        need = {'all_b': ['%.2f' % float(r['pct']) for r in AS if r['site'] == sk and r['kind'] == 'boundary' and r['group'] == 'all'] + ['%.2f' % float(st['shown_pct']) for st in MAN['figures'][sk]['boundary_all']['annotations']['stretches']],
                'all_c': ['%.2f' % float(r['pct']) for r in AS if r['site'] == sk and r['kind'] == 'coast' and r['group'] == 'all']}
        for kk, (k_, kd, gp) in enumerate([('sea_b', 'boundary', 'season'), ('sea_c', 'coast', 'season'), ('tid_b', 'boundary', 'tide'), ('tid_c', 'coast', 'tide')]):
            need[k_] = ['%.2f' % float(x) for t_ in MAN['figures'][sk][FIGK[KEYS8.index(k_)]]['panel_titles'] for x in re.findall(r'①\s*([\d.]+)%', t_)]   # 多格圖：每格只寫第一名（①），數值取自面板標題（最大餘數法，合計對齊）
        for k_, lst_ in need.items():
            miss = [v_ for v_ in lst_ if (v_ + '%') not in GUI[(sk, k_)]]
            if miss: gbad.append((nm, k_, miss))
        # 分組合計（穿越／海岸來源比例）須與 group_totals 相同
        for k_, gp, fld, lab in (('sea_b', 'season', 'crossed_pct', '穿越'), ('sea_c', 'season', 'coast_pct', '海岸來源'), ('tid_b', 'tide', 'crossed_pct', '穿越'), ('tid_c', 'tide', 'coast_pct', '海岸來源')):
            for m in re.finditer(r'(冬季|春季|夏季|秋季|大潮|小潮)%s比例為 ([\d.]+)%%' % lab, GUI[(sk, k_)]):
                gk = {'冬季': 'season_DJF', '春季': 'season_MAM', '夏季': 'season_JJA', '秋季': 'season_SON', '大潮': 'tide_spring_proxy', '小潮': 'tide_neap_proxy'}[m.group(1)]
                if '%.2f' % float(GTR[(sk, gk)][fld]) != m.group(2): nbad.append((nm, k_, m.group(1), m.group(2)))
        # 零值：顆數恰為 0 者寫『無海岸來源』；顆數不為 0 而四捨五入為 0.00% 者寫『<0.01%』；不得出現『海岸來源比例為 0.00%』
        GK_ = {'冬季': 'season_DJF', '春季': 'season_MAM', '夏季': 'season_JJA', '秋季': 'season_SON', '大潮': 'tide_spring_proxy', '小潮': 'tide_neap_proxy'}
        for k_, labs_ in (('sea_c', ['冬季', '春季', '夏季', '秋季']), ('tid_c', ['大潮', '小潮'])):
            tx_ = GUI[(sk, k_)]
            named0 = set(); namedt = set(re.findall(r'(冬季|春季|夏季|秋季|大潮|小潮)海岸來源比例為 <0\.01%', tx_))
            for m in re.finditer(r'((?:冬|春|夏|秋)(?:、(?:冬|春|夏|秋))*季|大潮|小潮)無海岸來源', tx_):
                named0 |= set(x + '季' for x in re.findall(r'[冬春夏秋]', m.group(1))) if m.group(1).endswith('季') else {m.group(1)}
            z0 = {l_ for l_ in labs_ if int(GTR[(sk, GK_[l_])]['coast_n']) == 0}
            zt = {l_ for l_ in labs_ if int(GTR[(sk, GK_[l_])]['coast_n']) > 0 and '%.2f' % float(GTR[(sk, GK_[l_])]['coast_pct']) == '0.00'}
            if named0 != z0 or namedt != zt or re.search(r'海岸來源比例為 0\.00%', tx_): nbad.append((nm, k_, '零值', sorted(named0), sorted(z0), sorted(namedt), sorted(zt)))
        # 十類範圍
        cl = [r for (ss, g_), r in GTR.items() if ss == sk and g_.startswith('class_')]
        for k_, fld in (('cls_b', 'crossed_pct'), ('cls_c', 'coast_pct')):
            vv = [float(r[fld]) for r in cl]
            if ('介於 %.2f%% 至 %.2f%%' % (min(vv), max(vv))) not in GUI[(sk, k_)]: nbad.append((nm, k_, '十類範圍'))
    chk('N13 v8：新增導引段之每個百分比皆出自 segprob_v3 之表（括號值、分組合計、母版海岸子分項）；各分組編號括號之機率、全年邊界與海岸括號之機率皆出現於對應導引段', not nbad and not gbad, str(nbad[:4]) + str(gbad[:3]))
    nbad2 = []
    for s_, nm in enumerate(sites):
        sk = SITEK[s_]; cc = concl[nm]
        mains = ['%.2f' % float(r['pct']) for r in AS if r['site'] == sk and r['kind'] == 'boundary' and r['group'] == 'all']
        coasts = ['%.2f' % float(r['pct']) for r in AS if r['site'] == sk and r['kind'] == 'coast' and r['group'] == 'all']
        for v_ in mains + coasts:
            if v_ + '%' not in cc: nbad2.append((nm, v_))
        if ('全年穿越比例為 %s%%' % PUB[nm][0]) not in cc or ('海岸來源占 %s%%' % PUB[nm][1]) not in cc: nbad2.append((nm, '總量'))
        if nm == '南灣' and '0.001%' not in cc: nbad2.append((nm, '觸底'))
        if nm == '連江' and not ('29.68%' in cc and '14.26%' in cc): nbad2.append((nm, '子分項'))
    chk('N6 v8b：整體結論（1）–（5）：全年穿越比例與海岸來源比例 = 已公布值（73.40/26.60、78.35/21.65、97.77/2.23、68.91/31.09 ＋觸底 0.001%、56.06/43.94）；全年邊界與海岸括號值取自 segprob_v3；連江子分項 29.68／14.26 維持；季節部分為定性（依各季節最集中邊界段所在邊界分組），不含季節百分比', not nbad2, str(nbad2[:4]))
    topm = [max(float(r['pct']) for r in AS if r['site'] == sk and r['kind'] == 'boundary' and r['group'] == 'all') for sk in SITEK]
    chk('N7 v8：整體結論第 6 條「全年穿越最集中邊界段之機率」範圍 = 五地點最大括號值之最小最大（%.2f%%–%.2f%%）' % (min(topm), max(topm)), any(t.startswith('（6）') and ('機率為 %.2f%%–%.2f%%' % (min(topm), max(topm))) in t for t in got))
    # 判定：以 diff_verdicts_final.csv 逐項核對（獨立於產生程式）
    vbad = []
    for s_, nm in enumerate(sites):
        sk = SITEK[s_]; rows = [r for r in VER if r['site'] == sk]
        sea = [r for r in rows if r['kind'] == 'season']; sim = [(r['group_a'], r['group_b']) for r in sea if r['verdict_final'] == '相近']
        if len(sea) != 6: vbad.append((nm, 'season rows'))
        txt_ = GUI[(sk, 'sea_c')]; cc = concl[nm]
        if not sim: ok1 = '四季之分段分佈兩兩差異明顯' in txt_ and '四季之分段分佈兩兩差異明顯' in cc
        else: ok1 = sim == [('DJF', 'MAM')] and '冬、春兩季之分段分佈相近，其餘季節兩兩之間差異明顯' in txt_ and '冬、春兩季之分段分佈相近，其餘季節兩兩之間差異明顯' in cc
        if not ok1: vbad.append((nm, 'season'))
        if not all(r['verdict_final'] == '相近' for r in rows if r['kind'] == 'tide') or '大小潮兩組相近' not in GUI[(sk, 'tid_c')]: vbad.append((nm, 'tide'))
        if not all(r['verdict_final'] == '相近' for r in rows if r['kind'] == 'class') or '十類之間相近' not in GUI[(sk, 'cls_c')]: vbad.append((nm, 'class'))
        if not all(r['verdict_final'] == '相近' for r in rows if r['kind'] == 'velocity_within_class') or '同類十個沉降速度之間亦相近' not in GUI[(sk, 'cls_c')]: vbad.append((nm, 'velocity'))
    s148 = next(t for t in got if t.startswith('各研究地點十類之間'))
    chk('N14 v8：季節、大小潮、十類（含十速度）之「相近／差異明顯」判定 = diff_verdicts_final.csv（貢寮、新竹外海、南灣、連江四季兩兩差異明顯；龜山島僅冬–春相近；大小潮、十類、同類十速度皆相近），導引段、整體結論與綜合說明一致', not vbad and '貢寮、新竹外海、南灣與連江四季兩兩之間差異明顯；龜山島僅冬、春兩季相近' in s148, str(vbad[:4]))
    # ---- v8b 新增檢查：簡化段落、定性結論、龜山島 9.56/9.57、位置詞
    qbad = []; lbad = []; sbad = []; wbad = []
    for s_, nm in enumerate(sites):
        sk = SITEK[s_]
        # 季節／大小潮／十類導引段：不寫經緯度；每格只寫第一名（無「其次」）；零值不寫 0.00%
        for k_ in ('cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c'):
            g_ = GUI[(sk, k_)]
            if re.search(r'°|其次|[②③]', g_): lbad.append((nm, k_))
        for k_, fk in (('sea_b', 'boundary_season'), ('tid_b', 'boundary_tide'), ('sea_c', 'coast_season'), ('tid_c', 'coast_tide')):
            npan = len(MAN['figures'][sk][fk]['panel_titles'])
            nconc = GUI[(sk, k_)].count('集中於')
            ntop = sum(1 for t_ in MAN['figures'][sk][fk]['panel_titles'] if '①' in t_)
            if nconc != ntop: qbad.append((nm, k_, nconc, ntop))
        # 位置詞：邊界（南側邊界中段、北側邊界西段、西南角…）、海岸（研究地點某側臺灣本島海岸、龜山島沿岸、馬祖北竿…）
        for k_ in ('sea_b', 'tid_b'):
            for pos_ in re.findall(r'集中於([^（]+)（', GUI[(sk, k_)]):
                if not re.fullmatch(r'(?:[東西南北]側邊界(?:[西中東南北]段)?|[東西][南北]角)', pos_): wbad.append((nm, k_, pos_))
        for k_ in ('sea_c', 'tid_c'):
            for pos_ in re.findall(r'集中於([^（]+)（', GUI[(sk, k_)]):
                if not re.fullmatch(r'(?:研究地點(?:[東西南北]|東北|東南|西北|西南)側臺灣本島海岸|研究地點周邊(?:臺灣本島|恆春半島南端)?海岸|龜山島沿岸|馬祖北竿|馬祖南竿|福建沿岸(?:近岸島嶼)?|馬祖各島)', pos_): wbad.append((nm, k_, pos_))
        # 整體結論：季節部分定性，不含季節百分比；全年範圍保留；各季所屬邊界 = 該季最集中（①）括號所在邊界（獨立由 boundary_segments 表推得）
        cc = concl[nm]; parts_ = cc.split('；')
        sp_ = [x for x in parts_ if re.search(r'季.*集中於|四季穿越皆集中於', x) and '主要集中於' not in x]
        if len(sp_) != 1 or re.search(r'\d', sp_[0]): sbad.append((nm, '季節定性句', [x[:30] for x in parts_], sp_[:1])); continue
        if '°' not in parts_[0]: sbad.append((nm, '全年範圍'))
        bsr = {r['segment_id']: r for r in rd('boundary_segments_%s.csv' % sk)}
        SIDEN = {'N': '北側', 'S': '南側', 'E': '東側', 'W': '西側'}; CORN = {frozenset('NE'): '東北角', frozenset('NW'): '西北角', frozenset('SW'): '西南角', frozenset('SE'): '東南角'}
        for lab_, pn_ in zip('冬春夏秋', MAN['figures'][sk]['boundary_season']['annotations']):
            st_ = [x for x in pn_['stretches'] if x['marker'] == '①'][0]
            sides_ = {}
            for sid in st_['segments']:
                r_ = bsr[sid]; sd_ = r_['side']; v_ = [float(r_['start_lon']), float(r_['end_lon'])] if sd_ in 'NS' else [float(r_['start_lat']), float(r_['end_lat'])]
                sides_.setdefault(sd_, []).extend(v_)
            big_ = [k for k, v in sides_.items() if max(v) - min(v) > 0.12]
            if len(big_) == 1 and len(sides_) > 1: sides_ = {big_[0]: sides_[big_[0]]}
            nmx = CORN[frozenset(''.join(sides_))] if len(sides_) == 2 else SIDEN[list(sides_)[0]] + '邊界'
            clause = next((x for x in re.split(r'，', sp_[0]) if re.search(r'^[冬春夏秋、]*%s' % lab_, x) or re.search(r'[冬春夏秋、]*%s[冬春夏秋、]*季' % lab_, x)), '')
            if not clause.endswith(nmx) and not ('四季穿越皆集中於' + nmx) in sp_[0]: sbad.append((nm, lab_, nmx, clause))
    chk('N15 v8b：季節、大小潮、十類導引段不寫經緯度、每格只寫第一名（無「其次」「②③」），「集中於」之數目 = 有括號之面板數', not lbad and not qbad, str(lbad[:3]) + str(qbad[:3]))
    chk('N16 v8b：季節、大小潮導引段所用位置詞限於：邊界（X 側邊界、X 側邊界西／中／東（南／北）段、四個角）、海岸（研究地點某側臺灣本島海岸、研究地點周邊恆春半島南端海岸、龜山島沿岸、馬祖北竿、馬祖南竿、福建沿岸、福建沿岸近岸島嶼、馬祖各島）', not wbad, str(wbad[:4]))
    chk('N17 v8b：整體結論 (1)–(5) 之季節部分為定性（冬、春季穿越集中於…，夏、秋季集中於…），不含季節百分比，且保留全年穿越範圍（經緯度）', not sbad, str(sbad[:4]))
    # v8c：海岸括號跨研究地點兩側者不稱方位（獨立重算）：括號含同一海岸環上距研究地點最近之海岸段且該段不在括號端點，或兩端點方位夾角 ≥ 150°；且括號最近處距研究地點 ≤ 10 km
    import math as _m
    ANC_ = {'gongliao': (121.92, 25.10), 'guishan': (121.94, 24.85), 'hsinchu': (120.45, 24.75), 'nanwan': (120.77, 21.92), 'lienchiang': (119.95, 26.20)}
    def _two(sk, ids):
        cs_ = {r['segment_id']: r for r in rd('coast_segments_%s.csv' % sk)}; ax, ay = ANC_[sk]
        km_ = lambda lo, la: ((lo - ax) * 111.32 * _m.cos(_m.radians(ay)), (la - ay) * 110.57)
        def dd(i):
            r = cs_[i]; A = km_(float(r['start_lon']), float(r['start_lat'])); B = km_(float(r['end_lon']), float(r['end_lat']))
            dx, dy = B[0] - A[0], B[1] - A[1]; L = dx * dx + dy * dy; t = max(0.0, min(1.0, -(A[0] * dx + A[1] * dy) / L)) if L else 0.0
            return _m.hypot(A[0] + t * dx, A[1] + t * dy)
        ids = sorted(ids); ring = ids[0].split('-')[0]; near = min([i for i in cs_ if i.startswith(ring + '-')], key=dd)
        if min(dd(i) for i in ids) > 10.0: return False
        e1 = km_(float(cs_[ids[0]]['start_lon']), float(cs_[ids[0]]['start_lat'])); e2 = km_(float(cs_[ids[-1]]['end_lon']), float(cs_[ids[-1]]['end_lat']))
        b1 = _m.degrees(_m.atan2(*e1)); b2 = _m.degrees(_m.atan2(*e2))
        return (near in ids and 0 < ids.index(near) < len(ids) - 1) or abs((b1 - b2 + 180) % 360 - 180) >= 150
    pbad = []; pchg = []
    for s_, nm in enumerate(sites):
        sk = SITEK[s_]
        for k_, fk in (('sea_c', 'coast_season'), ('tid_c', 'coast_tide')):
            f_ = MAN['figures'][sk][fk]; anns_ = [a_ for a_ in (f_['annotations'] if isinstance(f_['annotations'], list) else [f_['annotations']]) if a_['stretches']]
            tops = [t_ for t_ in f_['panel_titles'] if '①' in t_]
            labs_ = re.findall(r'集中於([^（]+)（([\d.]+)%', GUI[(sk, k_)])
            if len(anns_) != len(tops) or len(labs_) != len(tops): pbad.append((nm, k_, len(anns_), len(tops), len(labs_))); continue
            for a_, t_, (lab_, pc_) in zip(anns_, tops, labs_):
                st_ = [x for x in a_['stretches'] if x['marker'] == '①'][0]
                if not (lab_.startswith('研究地點') and (lab_.endswith('側臺灣本島海岸') or lab_ == '研究地點周邊臺灣本島海岸')): continue
                exp_ = _two(sk, st_['segments'])
                if exp_ != (lab_ == '研究地點周邊臺灣本島海岸'): pbad.append((nm, k_, lab_, pc_, exp_))
                if lab_ == '研究地點周邊臺灣本島海岸': pchg.append((nm, k_, pc_))
    chk('N19 v8c：季節、大小潮海岸括號若跨研究地點兩側（含最近海岸段且不在端點，或兩端夾角 ≥ 150°；10 km 內）則稱「研究地點周邊臺灣本島海岸」，否則用方位詞（獨立重算；周邊者 %d 處：%s）' % (len(pchg), '、'.join('%s%s%s%%' % (a_, '季節' if b_ == 'sea_c' else '大小潮', c_) for a_, b_, c_ in pchg)), not pbad, str(pbad[:4]))
    cap_new = [kids[n_ + 1] for n_, p_ in enumerate(kids) if p_.tag == q('p') and figs_in(p_) and n_ > ia and int(re.match(r'^圖(\d+) ', ptext(kids[n_ + 1])).group(1)) in NEWFIG]
    capm_ = [p_ for p_ in kmast if p_.tag == q('p') and re.match(r'^圖\d+ ', ptext(p_))]
    chk('K7 v8c：新增 40 個圖說皆有 keepLines（圖說不跨頁）；主檔既有 %d 個圖說皆無 keepLines（保留之 22 個圖說維持不變）' % len(capm_), len(cap_new) == 40 and all(rpr_has(p_, 'keepLines') for p_ in cap_new) and not any(rpr_has(p_, 'keepLines') for p_ in capm_))
    # 龜山島 9.56／9.57 與 0.13
    g_c = GUI[('guishan', 'all_c')]
    gs_ = [t for t in got if '龜山島' in t and re.search(r'9\.5[67]%|0\.13%', t)]
    chk('N18 v8b：龜山島海岸來源子分項寫「臺灣本島占 11.96%，龜山島占 9.57%，其餘為其他島嶼」；同段 9.57% 只出現一次；全部新增／改寫文字無 9.56%、0.13%', '臺灣本島占 11.96%，龜山島占 9.57%，其餘為其他島嶼' in g_c and g_c.count('9.57%') == 1 and not [t[:20] for t in texts_new if '9.56%' in t or '0.13%' in t] and all(t.count('9.57%') <= 1 for t in texts_new))
    # 殘留用語
    resid = ['圓形', '來源方位', '扇區', '玫瑰', '方向機率', '主要來源方向', '來源方向', '方向區間', '正北', '順時針', '1/36', '2.78%', '十速度', 'BayTrace', 'Baytrace', 'baytrace', '反射', '本研究', '曲線之間']
    rb = [w for w in resid if w in FT] + (['360'] if re.search(r'(?<![\d.])360(?![\d.])', FT) else [])
    chk('T6 v8：全文已無圓形來源方位圖用語（圓形、來源方位、扇區、玫瑰、方向機率、主要來源方向、來源方向、方向區間、正北、順時針、1/36、2.78%、360）、無「十速度」圖或段落、無 BayTrace、無「反射」、無「本研究」', not rb, str(rb))
    ng = [t[:15] for t in got if re.search(r'^圖 ?\d+ ', t) and re.search(r'十個沉降速度之曲線|十類各十個', t)]
    chk('T7 v8：十速度之圖與導引段已移除，僅保留「同類十個沉降速度之間亦相近」一句（五地點各 1 處）', not ng and sum(t.count('同類十個沉降速度之間亦相近') for t in got) == 5, str(ng))
    PLC = {'龜山島', '臺灣本島', '福建沿岸', '馬祖北竿', '馬祖南竿', '馬祖各島', '馬祖列島', '恆春半島', '小琉球', '蘭嶼', '近岸島嶼', '其他島嶼', '各島'}
    found = set()
    for t in texts_new: found |= set(re.findall(r'[一-鿿]{0,3}(?:本島|各島|列島|半島|竿|沿岸|小琉球|蘭嶼|島嶼)', t))
    unk = {x for x in found if not any(x.endswith(p_) or p_.endswith(x) or p_ in x for p_ in PLC)}
    chk('T8 v8：新增文字所用之地名限於已登記者（龜山島、臺灣本島、福建沿岸、馬祖北竿、馬祖南竿、馬祖各島、馬祖列島、恆春半島、小琉球、蘭嶼）', not unk, str(unk))
    chk('T9 v8b：新增文字無「本研究」、無待確認／推測語（可能、或許、大約、似乎、尚待、尚未、未知、不確定）；唯一例外為使用者核定之子句「由於廢棄物實際何時沉底未知，」（貢寮圖 15，逐字保留）', not [t[:20] for t in texts_new if re.search(r'本研究|可能|或許|似乎|尚待|尚未|未知|不確定|應該', t.replace('由於廢棄物實際何時沉底未知，', ''))])
    chk('T10 v8：每個新增導引段皆含結果句（百分比）', all(re.search(r'\d+\.\d\d%', GUI[(sk, k_)]) for sk in SITEK for k_ in KEYS8 if not (sk == 'hsinchu' and k_ == 'all_c')) and re.search(r'\d+\.\d\d%', GUI[('hsinchu', 'all_c')]))
    chk('W12 v8b：貢寮圖 15 導引段：沉底時刻定義逐字等於使用者核定之主檔文字（含「由於廢棄物實際何時沉底未知，」），其後「季節依沉底時刻所在月份劃分，冬季為 12–2 月…」維持；不含「分層」「種子」',
        '沉底時刻為廢棄物沉降至海床之時刻，亦即虛擬質點回溯之起點；由於廢棄物實際何時沉底未知，沉底時刻由所屬到達時間往前推一段隨機抽取之沉底年齡決定，每個系集成員各自抽取；季節依沉底時刻所在月份劃分，冬季為 12–2 月、春季為 3–5 月、夏季為 6–8 月、秋季為 9–11 月。' in r13 and (SEADEF_M in ptext(kmast[53])) and '分層' not in r13 and '種子' not in r13 and r13.startswith('圖 15 為依沉底時刻劃分之貢寮四季邊界穿越來源分段機率圖。'))
    chk('W15 v8：（二）55 張圖之導引段皆含結果句（百分比、「相近」或「一致」）', len(_fig_intro) == 55 and all(_res), '%d %d' % (len(_fig_intro), sum(_res)))
    TT = [(11, 12), (22, 23), (33,), (44, 45), (55, 56)]
    chk('W16 v8：五張軌跡圖導引段各有一句結果（延伸／穿越方向、抵達海岸），並呼應各地點全年邊界與海岸來源分段機率圖（圖 11 及 12、22 及 23、33、44 及 45、55 及 56 之主要來源）；不含條數',
        len(_traj) == 5 and all(t.endswith('與圖 %s 之主要來源一致。' % ' 及圖 '.join(str(x) for x in k_)) for t, k_ in zip(_traj, TT)) and not any(re.search(r'\d+ 條中|其中 \d+ 條', t) for t in _traj) and all('延伸' in t and '一致' in t for t in _traj), str(len(_traj)))
    if PDF and os.path.exists(PDF):
        chk('D2 v8：PDF 圖說 圖1–圖62 皆出現', all(re.search(r'圖 ?%d ' % n, txt) for n in range(1, 63)))
    # ---- 使用者手改（v7 之後、v8 之前）之登記與其餘舊檢查之改版說明
    bk3 = os.path.join(BASE, '成果報告_run09_合併版.docx.bak03-20261025')
    if os.path.exists(bk3):
        zb3_ = zipfile.ZipFile(bk3)
        chk('W9a 主檔與 v3 備份（bak03）相比：除 word/document.xml 與使用者 Word 重存所改之 settings.xml、docProps/core.xml、docProps/app.xml 外，全部 zip 成員逐位元組相同', zmast.namelist() == zb3_.namelist() and all(zmast.read(n) == zb3_.read(n) for n in zmast.namelist() if n not in ('word/document.xml', 'word/settings.xml', 'docProps/core.xml', 'docProps/app.xml')))
    if os.path.exists(BAK16):
        dd = []
        for n_, eb in enumerate(_kb):
            if 33 <= n_ < 158: continue            # 成果段（v8 改寫範圍）不在此比對；其內之附錄／表格無涉
            em = kids[n_ + (DELTA if n_ >= 158 else 0)]
            nb, nm_ = rev(marked(eb)), rev(marked(em))
            if n_ == i_tbl1: nb = re.sub(r'−(?=0\.)', '', nb).replace('半壓扁至嚴重壓扁', '半壓扁至重壓扁')   # 使用者手改（登記）
            if n_ == i_item1: nb = nb.replace('參數： 設定', '參數：設定')                                      # 使用者手改（登記）
            nm_ = nm_.replace(SENT_E, '')   # v10 登記：區塊 6 末之核定句（V10-12 逐字斷言）
            nb = nb.replace('嚴重壓扁', '重壓扁').replace('到達時間驗證圖，每個研究地點 1 張全年總覽圖，', '到達時間驗證圖，')   # v10 登記：使用者於 v9 升格後在 Word 手改（全文「嚴重壓扁」→「重壓扁」；區塊 20 刪「每個研究地點 1 張全年總覽圖，」）
            for anchor, dele, allowed, n_exp in V7_P:
                if dele is not None and nb.startswith(anchor): nb = nb.replace(dele, '')
            if nb != nm_: dd.append((n_, nb[:40], nm_[:40]))
        chk('W21 v8：v7 之科學記號與去負號結果仍成立——還原後，候選之首段（區塊 0–32）與附錄、參考資料（主檔區塊 158 起）逐區塊等於 v6f（bak16），僅差登記之使用者手改 2 處；成果段（33–157）由 M3、N13 等另行把關', not dd and len(_kb) == 299, str(dd[:3]))
    chk('S4 v8：已由 v8 版取代之舊檢查：%s（舊式程式碼保留於本檔供對照，結果不計）' % '、'.join(SUPERSEDED), len(SUPERSEDED) >= 20)

# ---------------------------------------------------------------- v9 新增檢查：候選 ＝ v8 主檔（sha 74d288ed…）＋ 登記之 v9 變更
if V9 and not V10:
    import glob, struct, csv as _csv
    _cands = [os.path.join(BASE, '成果報告_run09_合併版.docx')] + sorted(glob.glob(os.path.join(BASE, '成果報告_run09_合併版.docx.bak*')))
    M8P = next((p_ for p_ in _cands if os.path.exists(p_) and sha(open(p_, 'rb').read()).startswith('74d288ed1dc16b94')), None)
    chk('V9-1 基準主檔：v8 主檔（sha256 以 74d288ed1dc16b94 起首、359 區塊）存在且未被動過（%s）' % (os.path.basename(M8P) if M8P else '找不到'), M8P is not None)
    z8, root8, k8, rels8 = load(M8P)
    chk('V9-2 候選區塊數 = v8 主檔 = 359（不增不減）', len(k8) == 359 == len(kids), '%d %d' % (len(k8), len(kids)))
    FIGN = {(s_, k_): BASE8[i_] + 3 + KEYS8.index(k_) for i_, s_ in enumerate(SITEK) for k_ in KEYS8}
    CHG = ['cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c']
    def fig_idx(kk, n_):
        for i_, p_ in enumerate(kk):
            if p_.tag == q('p'):
                for inl in p_.iter('{%s}inline' % WP):
                    if inl.find('{%s}docPr' % WP).get('name') == 'fig%d.png' % n_: return i_
        return None
    FM9 = {(f['site'], f['key']): f for f in LOG['fig_map']}
    sd9 = {}
    bad_img, bad_cap, bad_gd, bad_ot = [], [], [], []
    chg_i = {}
    for (s_, k_), n_ in FIGN.items():
        ic, i8 = fig_idx(kids, n_), fig_idx(k8, n_)
        if ic is None or ic != i8: bad_ot.append(('位置', s_, k_, ic, i8)); continue
        if k_ in CHG: chg_i[(s_, k_)] = ic
    chk('V9-3 圖號 8–62 之 40 張圖在候選與 v8 主檔之區塊位置相同；其中 30 張為變更對象', not bad_ot and len(chg_i) == 30, str(bad_ot[:3]))
    imgset = {i_ for i_ in chg_i.values()}; capset = {i_ + 1 for i_ in imgset}; gdset = {i_ - 1 for i_ in imgset}
    OLDSET = {}
    for i_, (e8, ec) in enumerate(zip(k8, kids)):
        s8 = sig(e8, z8, rels8); sc = sig(ec, zm, relsM)
        if i_ in imgset:
            ok_ = sc[0] == 'p' and sc[1] == s8[1] and sc[2] == s8[2] and len(sc[3]) == 1 == len(s8[3]) and sc[3][0][0] == s8[3][0][0] == 5733288 and sc[3][0][2] != s8[3][0][2]
            if not ok_: bad_img.append(i_)
        elif i_ in capset:
            cs8 = {r_ for r_, _ in s8[2]}; csc = {r_ for r_, _ in sc[2]}
            ok_ = sc[0] == 'p' and sc[1] == s8[1] and cs8 == csc and ''.join(c_ for _, c_ in sc[2]) != ''.join(c_ for _, c_ in s8[2])
            if not ok_: bad_cap.append(i_)
        elif i_ in gdset:
            if sig(e8, z8, rels8, nokeep=True) != sig(ec, zm, relsM, nokeep=True): bad_gd.append(i_)
        elif s8 != sc: bad_ot.append(('區塊', i_))
    chk('V9-4 其餘 %d 個區塊（含全年 10 張圖、各導引段文字、全部其他段落與表格）與 v8 主檔逐字、逐格式、逐圖片相同；30 張圖之段落格式與前後文字不變、僅圖檔與高度不同；30 個圖說僅文字不同（run 格式同）；30 個導引段除 keepNext 外相同' % (359 - 90),
        not bad_img and not bad_cap and not bad_gd and not [x for x in bad_ot if x[0] == '區塊'] and len(imgset) == len(capset) == len(gdset) == 30, '%s %s %s %s' % (bad_img[:3], bad_cap[:3], bad_gd[:3], bad_ot[:3]))
    nm8 = z8.namelist(); nmc = zm.namelist()
    media30 = {'word/' + relsM[figs_in(kids[i_])[0][2]] for i_ in imgset}
    others = [n_ for n_ in nmc if n_ != 'word/document.xml' and n_ not in media30]
    chk('V9-5 zip 成員：名稱與次序同 v8 主檔；除 word/document.xml 與被替換之 30 個圖檔外，其餘 %d 個成員（含 10 張全年圖、其他圖檔、styles、numbering、rels、footer…）逐位元組相同；30 個圖檔皆不同' % len(others),
        nm8 == nmc and len(media30) == 30 and all(zm.read(n_) == z8.read(n_) for n_ in others) and all(zm.read(n_) != z8.read(n_) for n_ in media30))
    ibad = []
    for (s_, k_), i_ in chg_i.items():
        f_ = figs_in(kids[i_])[0]; data_ = zm.read('word/' + relsM[f_[2]]); w_, h_ = struct.unpack('>II', data_[16:24])
        disk_ = open(os.path.join(FINAL9, s_, FIGK[KEYS8.index(k_)] + '.png'), 'rb').read()
        mf_ = json.load(open(os.path.join(FINAL9, s_, 'manifest.json'), encoding='utf-8'))[FIGK[KEYS8.index(k_)]]
        e_ = FM9[(s_, k_)]
        if not (data_ == disk_ and sha(data_) == e_['png_sha256'] and w_ == 3762 and f_[0] == 5733288 and f_[1] == round(h_ / 600.0 * 914400) == e_['cy'] and h_ == e_['h_px']
                and abs(mf_['canvas_in'][0] - 6.27) < 0.005 and abs(mf_['canvas_in'][1] - h_ / 600.0) < 0.002):
            ibad.append((s_, k_))
    chk('V9-6 30 張替換圖：圖檔與 segprob_fullpage/final 之磁碟檔逐位元組相同（sha 同建置紀錄）、寬 6.27 in（5733288 EMU、3762 px／600 dpi）、高依像素等比（cy = 高 px／600 × 914400）、未縮放', not ibad and len(chg_i) == 30, str(ibad[:4]))
    # 圖說
    POSW9 = {'UR': '右上', 'UL': '左上', 'LR': '右下', 'LL': '左下'}
    cbad9 = []; thr9 = []
    for (s_, k_), i_ in chg_i.items():
        t_ = ptext(kids[i_ + 1]); mf_ = json.load(open(os.path.join(FINAL9, s_, 'manifest.json'), encoding='utf-8'))[FIGK[KEYS8.index(k_)]]
        pw = POSW9[mf_['box_corner']]
        need_ = ['各格%s數值框列出' % pw]
        if k_ == 'cls_b': need_ += ['括號標示機率集中之連續邊界段', '圓圈數字依機率由大至小編號', '該類之穿越比例、各括號段之機率（圈號）與其餘', '「其餘」為未編號邊界段之機率合計']
        if k_ in ('sea_b', 'tid_b'): need_ += ['括號標示機率集中之連續邊界段', '圓圈數字依機率由大至小編號', '穿越比例、各括號段之機率（圈號）與其餘', '「其餘」為未編號邊界段之機率合計']
        if k_ == 'cls_c': need_ += ['該類之海岸來源比例']
        if k_ in ('sea_c', 'tid_c'): need_ += ['圓圈數字依機率由大至小標示機率集中之海岸段', '海岸來源比例及各圈號海岸段之機率']
        nk_ = re.sub(r' ', '', t_)
        for ph in need_:
            if ph.replace(' ', '') not in nk_: cbad9.append((s_, k_, ph))
        if re.search(r'各格標題|其後數值|標題', t_): cbad9.append((s_, k_, '殘留標題用語'))
        if re.search(r'[≥≧≤]|門檻|閾值|至少 ?\d|(?<![\d.])[13]%|2 段|1%以上|3%以上', t_): thr9.append((s_, k_))
        if sum(1 for pw_ in POSW9.values() if pw_ + '數值框' in t_) != 1: cbad9.append((s_, k_, '位置詞不唯一'))
        if not rpr_has(kids[i_ + 1], 'keepLines'): cbad9.append((s_, k_, 'keepLines'))
        if t_ != '圖%d %s' % (FIGN[(s_, k_)], FM9[(s_, k_)]['caption']): cbad9.append((s_, k_, '與建置紀錄不同'))
    chk('V9-7 30 個圖說：改述為「各格○○數值框列出…」（位置詞 = 該圖數值框實際角落〔final manifest box_corner〕，每個圖說恰一個位置詞）；十類邊界與季節、大小潮邊界含括號、圓圈數字（依機率由大至小）、穿越比例、各括號段之機率（圈號）與其餘；海岸含圈號海岸段之機率；無「各格標題」「其後數值」；不寫門檻值；保留 keepLines；與建置紀錄相同',
        not cbad9 and not thr9, str(cbad9[:4]) + str(thr9[:2]))
    chk('V9-8 全文已無「各格標題」「其後數值為該段機率」之殘留指稱', not [i_ for i_, p_ in enumerate(kids) if p_.tag == q('p') and re.search(r'各格標題|其後數值為該段機率', ptext(p_))])
    # 導引段 keepNext 之變動（由圖高改變所致）須與建置紀錄一致
    flips_c = [(s_, k_, rpr_has(kids[i_ - 1], 'keepNext'), rpr_has(k8[i_ - 1], 'keepNext')) for (s_, k_), i_ in chg_i.items() if rpr_has(kids[i_ - 1], 'keepNext') != rpr_has(k8[i_ - 1], 'keepNext')]
    fl9 = {(f_['site'], f_['key']) for f_ in LOG['flips']}
    chk('V9-9 導引段 keepNext 與 v8 主檔不同者 %d 處，與建置紀錄之 keep_flips 相同，且皆為圖高改變之 30 張圖之導引段；導引段文字不變' % len(flips_c), {(a_, b_) for a_, b_, _, _ in flips_c} == fl9 and all(ptext(kids[i_ - 1]) == ptext(k8[i_ - 1]) for i_ in imgset), str(flips_c[:4]))
    # final 之檢查結果
    rbad = []; nchk9 = 0; ncol9 = 0
    for s_ in SITEK:
        rj = json.load(open(os.path.join(FINAL9, s_, 'result.json'), encoding='utf-8'))
        if rj['fails']: rbad.append((s_, rj['fails'][:2]))
        mf_ = json.load(open(os.path.join(FINAL9, s_, 'manifest.json'), encoding='utf-8'))
        for fk_ in FIGK[2:]:
            ncol9 += mf_[fk_]['collision_count']
        rows_ = list(_csv.DictReader(open(os.path.join(FINAL9, s_, 'checks.csv'), encoding='utf-8-sig'))); nchk9 += len(rows_)
        rbad += [(s_, r_['figure'], r_['panel'], r_['check']) for r_ in rows_ if r_['result'] == 'FAIL']
    chk('V9-10 五地點 30 張圖之版面與數值檢查（final/<site>/result.json 與 checks.csv，共 %d 列）：失敗 0；數值框、括號、圈號、標籤之碰撞 0；括號範圍與集中段一致；標籤距離合格；各面板數值與 v8 標題逐面板相同（圈號、其餘加總 = 總計）' % nchk9, not rbad and ncol9 == 0 and nchk9 > 0, str(rbad[:3]) + ' coll=%d' % ncol9)

# ---------------------------------------------------------------- v10 新增檢查：候選 ＝ 現行主檔（sha 20d83708…）＋ 登記之 v10 變更
if V10:
    import struct, csv as _csv
    V10L = LOG['v10']
    chk('V10-1 基準主檔：現行主檔（使用者於 v9 升格後再存；sha256 以 20d8370880154b53 起首、359 區塊）存在且未被動過（%s）' % (os.path.basename(M10P) if M10P else '找不到'), M10P is not None and len(kb10) == 359 and V10L['base_sha_prefix'] == '20d8370880154b53')
    chk('V10-2 候選區塊數 = 現行主檔 = 359（不增不減）', len(kb10) == 359 == len(kids), '%d %d' % (len(kb10), len(kids)))
    FIGN10 = {(s_, k_): BASE8[i_] + 3 + KEYS8.index(k_) for i_, s_ in enumerate(SITEK) for k_ in KEYS8}
    def fig_map_idx(kk):
        d_ = {}
        for i_, p_ in enumerate(kk):
            if p_.tag == q('p'):
                for inl in p_.iter('{%s}inline' % WP):
                    m_ = re.match(r'^fig(\d+)\.(png|jpg)$', inl.find('{%s}docPr' % WP).get('name'))
                    if m_: d_[int(m_.group(1))] = i_
        return d_
    FMC, FMB = fig_map_idx(kids), fig_map_idx(kb10)
    chk('V10-3 成果段 55 張圖（圖 8–62，docPr 名稱 figN）在候選與現行主檔之區塊位置、編號完全相同；40 個來源分段機率圖（圖 11–18、22–29、33–40、44–51、55–62）之圖說、導引段位置相同', FMC == FMB and sorted(FMC) == list(range(8, 63)) and all(n_ in FMC for n_ in FIGN10.values()), str([n_ for n_ in FMB if FMC.get(n_) != FMB[n_]][:5]) + str((len(FMC), len(FMB), sorted(set(range(1, 63)) - set(FMC))[:10])))
    CHGN = {c_['fig'] for c_ in V10L['changes'] if c_['kind'] == 'v10_caption'}
    B6 = V10L['block6']; SENT6 = B6['added']
    CURV = V10L['curves']; CURN = {c_['fig'] for c_ in CURV}
    CT = {c_['fig']: c_ for c_ in V10L.get('curve_text', [])}; B34 = V10L.get('block34')
    curcap = {FMB[n_] + 1: n_ for n_ in CT}; curgd = {FMB[n_] - 1: n_ for n_ in CT}
    capset = {FMB[n_] + 1 for n_ in FIGN10.values()}; gdset = {FMB[n_] - 1 for n_ in FIGN10.values()}; curset = {FMB[n_] for n_ in CURN}
    cap_i2n = {FMB[n_] + 1: n_ for n_ in FIGN10.values()}
    bad_cap, bad_gd, bad_cur, bad_oth = [], [], [], []
    for i_, (eb, ec) in enumerate(zip(kb10, kids)):
        sb = sig(eb, zb10, relsb10); sc = sig(ec, zm, relsM)
        if i_ in capset:
            n_ = cap_i2n[i_]
            same_fmt = sc[0] == 'p' and sc[1] == sb[1] and {r_ for r_, _ in sc[2]} == {r_ for r_, _ in sb[2]} and sc[3] == sb[3] == ()
            txt_chg = ''.join(c_ for _, c_ in sc[2]) != ''.join(c_ for _, c_ in sb[2])
            if not (same_fmt and txt_chg == (n_ in CHGN)): bad_cap.append((i_, n_))
        elif i_ in gdset:
            if sig(eb, zb10, relsb10, nokeep=True) != sig(ec, zm, relsM, nokeep=True): bad_gd.append(i_)
        elif i_ in curcap or i_ in curgd:   # 使用者核定之文字（已於 sig() 正規化為舊文字）：格式與其餘內容與主檔相同；新文字逐字由 V10-13 斷言
            a_ = sb if i_ in curcap else sig(eb, zb10, relsb10, nokeep=True); b_ = sc if i_ in curcap else sig(ec, zm, relsM, nokeep=True)
            if a_ != b_ or (i_ in curgd and rpr_has(ec, 'keepNext') != CT[curgd[i_]]['guide_keepNext']): bad_oth.append(i_)
        elif i_ == 34 and B34:
            if sb != sc or ptext(eb) != B34['old']: bad_oth.append(i_)
        elif i_ in curset:
            ok_ = sc[0] == 'p' and sc[1] == sb[1] and sc[2] == sb[2] and len(sc[3]) == 1 == len(sb[3]) and sc[3][0][0] == sb[3][0][0] == 5733288 and sc[3][0][2] != sb[3][0][2]
            if not ok_: bad_cur.append(i_)
        elif sb != sc: bad_oth.append(i_)
    chk('V10-4 其餘 %d 個區塊（含全部圖、其他段落與表格；區塊 6 之核定句已於比對時正規化、另由 V10-12 逐字斷言其存在；使用者手改之第 20 區塊與 10 個「重壓扁」區塊已隨基準一併帶入，雙方皆為現行主檔文字）與現行主檔逐字、逐格式、逐圖片相同；40 個圖說僅文字不同（run 格式同；改寫 %d、不變 %d）；40 個導引段除 keepNext 外相同；曲線圖 %d 張僅圖檔與高度不同' % (359 - len(capset) - len(gdset) - len(curset) - len(CT) * 2 - (1 if B34 else 0), len(CHGN), 40 - len(CHGN), len(curset)),
        not bad_cap and not bad_gd and not bad_cur and not bad_oth and len(capset) == len(gdset) == 40 and len(curset) == len(CURV), '%s %s %s %s' % (bad_cap[:3], bad_gd[:3], bad_cur[:3], bad_oth[:5]))
    nmc, nmb = zm.namelist(), zb10.namelist()
    drop10 = {c_['old_member'] for c_ in CURV}; add10 = {c_['new_member']: c_ for c_ in CURV}
    exp_names = [n_ for n_ in nmb if n_ not in drop10] + [c_['new_member'] for c_ in CURV]
    oth10 = [n_ for n_ in nmc if n_ not in ('word/document.xml', 'word/_rels/document.xml.rels') and n_ not in add10]
    ra_b = {r_.get('Id'): r_.get('Target') for r_ in etree.fromstring(zb10.read('word/_rels/document.xml.rels'))}; ra_c = relsM
    rels_diff = {k_ for k_ in set(ra_b) | set(ra_c) if ra_b.get(k_) != ra_c.get(k_)}
    chk('V10-5 zip 成員：名稱與次序同現行主檔（曲線圖取代者除外：移除 %d、新增 %d）；除 word/document.xml、document.xml.rels 與曲線圖外，其餘 %d 個成員逐位元組相同；document.xml.rels 僅差曲線圖 %d 個關係' % (len(drop10), len(add10), len(oth10), len(CURV)),
        nmc == exp_names and all(zm.read(n_) == zb10.read(n_) for n_ in oth10) and all(sha(zm.read(n_)) == c_['png_sha256'] for n_, c_ in add10.items()) and rels_diff == {c_['rid'] for c_ in CURV}, str((nmc == exp_names, rels_diff)))
    # 圖說
    cbad, pbad, oldbad = [], [], []
    CAPNEW = {n_: next(c_['new'] for c_ in LOG['changes'] if c_.get('fig') == n_ and c_['kind'] == 'new' and c_['role'].startswith('caption:')) for n_ in FIGN10.values()}
    FMlog = {(f_['site'], f_['key']): f_ for f_ in LOG['fig_map']}
    capt = {}
    for (s_, k_), n_ in FIGN10.items():
        t_ = ptext(kids[FMC[n_] + 1]); capt[n_] = t_
        if t_ != '圖%d %s' % (n_, FMlog[(s_, k_)]['caption']) or t_ != CAPNEW[n_]: cbad.append((n_, '與建置紀錄不同'))
        if not rpr_has(kids[FMC[n_] + 1], 'keepLines'): cbad.append((n_, 'keepLines'))
        if sites[SITEK.index(s_)] not in t_: cbad.append((n_, '地點名'))
        if re.search(r'數值框|[左右][上下]|角落|各格標題|其後數值|標題', t_): pbad.append((n_, re.findall(r'數值框|[左右][上下]|角落|各格標題|其後數值|標題', t_)))
        if re.search(r'[≥≧≤]|門檻|閾值|至少 ?\d|(?<![\d.])[13]%|2 段|1%以上|3%以上', t_): cbad.append((n_, '門檻'))
        o_ = FMlog[(s_, k_)].get('v9_caption')
        if o_ is not None:   # 數字（km、段數等）與 v9 圖說相同，只動位置與數值框用語
            if re.findall(r'\d+(?:\.\d+)?', o_) != re.findall(r'\d+(?:\.\d+)?', FMlog[(s_, k_)]['caption']): oldbad.append(n_)
    chk('V10-6 40 個圖說：文字 ＝ 建置紀錄（圖號 ＋ 新句）、保留 keepLines、含研究地點名、不寫門檻值；與 v9 圖說相比，數字（10 km、5 km 等）完全相同', not cbad and not oldbad, str(cbad[:4]) + str(oldbad[:3]))
    allp = [ptext(p_) for p_ in kids if p_.tag == q('p')] + [ptext(p_) for t_ in kids if t_.tag == q('tbl') for p_ in t_.iter(q('p'))]
    chk('V10-7 全文（含表格）已無「數值框」「右上／右下／左上／左下」位置詞（圖說）與「各格標題」「其後數值為該段機率」之殘留；40 個圖說無位置詞', not pbad and not [t_ for t_ in allp if re.search(r'數值框|各格標題|其後數值為該段機率', t_)], str(pbad[:3]))
    # 依圖面實際內容的用語（final manifest 為真）
    rbad = []; n_unnum = {}
    for (s_, k_), n_ in FIGN10.items():
        t_ = capt[n_]
        if k_ == 'all_b':
            if not (t_.endswith('，括號旁數值為該段之機率') and '黑色括號標示機率集中之連續邊界段' in t_ and '灰色括號標示其餘連續非零邊界段' in t_): rbad.append((n_, '全年邊界'))
            continue
        if k_ == 'all_c':
            if t_ != '圖%d %s' % (n_, FMlog[(s_, k_)]['v9_caption']) or '引線旁數值為機率集中之海岸段之機率' not in t_: rbad.append((n_, '全年海岸（應與 v9 圖說相同）'))
            continue
        mf_ = json.load(open(os.path.join(FINAL9, s_, 'manifest.json'), encoding='utf-8'))[FIGK[KEYS8.index(k_)]]
        anns_ = mf_['annotations']; anns_ = [anns_] if isinstance(anns_, dict) else anns_
        if k_ in ('cls_b', 'sea_b', 'tid_b'):
            pre = '各格內數值依序為' + ('該類之' if k_ == 'cls_b' else '') + '穿越比例、各括號段之機率（圈號）與其餘'
            unnum = any(st_.get('marker', '') == '' for a_ in anns_ for st_ in a_['stretches']); n_unnum[n_] = unnum
            tail = '「其餘」為未編號邊界段（含未編號之括號段）之機率合計' if unnum else '「其餘」為未編號邊界段之機率合計'
            if not (pre in t_ and t_.endswith(tail) and '括號標示機率集中之連續邊界段' in t_ and '圓圈數字依機率由大至小編號' in t_): rbad.append((n_, '邊界', unnum))
            if (not unnum) and '未編號之括號段' in t_: rbad.append((n_, '不應有未編號括號段'))
        if k_ == 'cls_c':
            if not (t_.endswith('各格內數值為該類之海岸來源比例') and '圓圈' not in t_ and '括號' not in t_): rbad.append((n_, '十類海岸'))
        if k_ in ('sea_c', 'tid_c'):
            has_ = [any(re.search(r'[①-⑳]', ln_) for ln_ in bl_) for bl_ in mf_['box_lines']]; nP = len(has_)
            if sum(has_) == 0:
                if not (t_.endswith('各格內數值為海岸來源比例') and '圓圈' not in t_): rbad.append((n_, '海岸無圈號'))
            elif sum(has_) == nP:
                if not ('圓圈數字依機率由大至小標示機率集中之海岸段' in t_ and t_.endswith('各格內數值依序為海岸來源比例及各圈號海岸段之機率')): rbad.append((n_, '海岸每格皆有圈號'))
            else:
                names_ = [mf_['panel_titles'][j_][:2] for j_, h_ in enumerate(has_) if h_]
                if not (('僅%s格有機率集中之海岸段，以圓圈數字標示' % '、'.join(names_)) in t_ and t_.endswith('各格內數值為海岸來源比例，%s格另列圈號海岸段之機率' % '、'.join(names_))): rbad.append((n_, '海岸部分格有圈號', names_, has_))
    UNN = sorted(n_ for n_, u_ in n_unnum.items() if u_)
    chk('V10-8 40 個圖說之用語與圖面實際內容相符（依 final manifest）：全年邊界「括號旁數值為該段之機率」；十類、季節、大小潮邊界「各格內數值依序為…穿越比例、各括號段之機率（圈號）與其餘」，有未編號括號段者（%s）另註「含未編號之括號段」；海岸圖依各格實有圈號者寫（圖 38 僅秋季格有、圖 40 全無）' % '、'.join('圖 %d' % n_ for n_ in UNN),
        not rbad and UNN == [15, 48], str(rbad[:4]) + str(UNN))
    # 使用者手改之登記
    n_ya = sum(ptext(p_).count('重壓扁') for p_ in kids if p_.tag == q('p')) + sum(ptext(p_).count('重壓扁') for t_ in kids if t_.tag == q('tbl') for p_ in t_.iter(q('p')))
    chk('V10-H1 使用者手改（全文「嚴重壓扁」→「重壓扁」）：候選已無「嚴重壓扁」，「重壓扁」共 %d 處（現行主檔 %d 處）；候選之 10 個區塊（260、294、302、308、312、318、319、331、332、356）與首段區塊 5 之用語皆為「重壓扁」' % (n_ya, sum(ptext(p_).count('重壓扁') for p_ in kb10 if p_.tag == q('p')) + sum(ptext(p_).count('重壓扁') for t_ in kb10 if t_.tag == q('tbl') for p_ in t_.iter(q('p')))),
        not any('嚴重壓扁' in t_ for t_ in allp) and n_ya == 18 and all('重壓扁' in ptext(kids[i_]) or any('重壓扁' in ptext(p_) for p_ in kids[i_].iter(q('p'))) for i_ in (5, 260, 294, 302, 308, 312, 318, 319, 331, 332, 356)), str(n_ya))
    t6 = ptext(kids[6]); t4 = ptext(kids[4])
    chk('V10-12 表 1 後第一段（區塊 6）末加使用者核定之一句，逐字：「各類 10 個速度之跨距較大，表 1 以科學記號列其範圍；模擬時 10 個速度各自直接使用，不取平均值或對數平均。」；原有文字（受力平衡求解、詳見附錄）在前、不變；表 1 標題（區塊 4）與表（區塊 5）不變；未新增區塊（仍為 359）',
        SENT6 == '各類 10 個速度之跨距較大，表 1 以科學記號列其範圍；模擬時 10 個速度各自直接使用，不取平均值或對數平均。' and t6 == ptext(kb10[6]) + SENT6 and t6.startswith('各類 10 個速度係以受力平衡求解，完整推導過程、代表物件之尺寸與輸入參數、實測量級對照與敏感度分析詳見附錄。各類 10 個速度之跨距較大')
        and t6.endswith('不取平均值或對數平均。') and t4 == ptext(kb10[4]) == '表1 十類海底廢棄物之代表形狀與終端沉降速度' and sig(kids[5], zm, relsM) == sig(kb10[5], zb10, relsb10) and len(kids) == 359 and B6['new'] == t6,
        t6[-60:])
    t20 = ptext(kids[20])
    chk('V10-H2 使用者手改（區塊 20）：到達時間驗證圖導引段已刪「每個研究地點 1 張全年總覽圖，」，其餘文字不變（逐字：「以下列出各研究地點之到達時間驗證圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7。…」）',
        t20.startswith('以下列出各研究地點之到達時間驗證圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7。') and '全年總覽' not in t20 and ptext(kb10[20]) == t20)
    # 導引段 keepNext
    flips_c = [n_ for n_ in list(FIGN10.values()) + list(CT) if rpr_has(kids[FMC[n_] - 1], 'keepNext') != rpr_has(kb10[FMB[n_] - 1], 'keepNext')]
    fl10 = {f_['fig'] for f_ in V10L['keep_flips']}
    chk('V10-9 導引段 keepNext 與現行主檔不同者 %d 處，與建置紀錄之 keep_flips 相同；40 個導引段文字不變；圖段皆有 keepNext' % len(flips_c), set(flips_c) == fl10 and all(ptext(kids[FMC[n_] - 1]) == ptext(kb10[FMB[n_] - 1]) for n_ in FIGN10.values()) and all(rpr_has(kids[FMC[n_]], 'keepNext') for n_ in FMC), str((flips_c, fl10)))
    # 曲線圖（選項 B）
    cu_bad = []
    for c_ in CURV:
        i_ = FMC[c_['fig']]; f_ = figs_in(kids[i_])[0]; data_ = zm.read('word/' + relsM[f_[2]]); w_, h_ = struct.unpack('>II', data_[16:24])
        sd_ = c_['site']
        if not (sha(data_) == c_['png_sha256'] and w_ == 3762 and f_[0] == 5733288 and f_[1] == round(h_ / 600.0 * 914400) == c_['new_cy'] and h_ == c_['h_px']): cu_bad.append(c_['fig'])
    chk('V10-10 5 張仍在追蹤比例圖（圖 9、20、31、42、53）：%s' % ('尚未換入選項 B（待使用者確認預覽）——圖檔與現行主檔相同' if not CURV else '已換入選項 B 曲線圖，圖檔 sha 同建置紀錄、寬 6.27 in（5733288 EMU、3762 px／600 dpi）、高依像素等比、未縮放'),
        (len(CURV) == 0 and all(zm.read('word/' + relsM[figs_in(kids[FMC[n_]])[0][2]]) == zb10.read('word/' + relsb10[figs_in(kb10[FMB[n_]])[0][2]]) for n_ in (9, 20, 31, 42, 53))) or (len(CURV) == 5 and not cu_bad and CURN == {9, 20, 31, 42, 53}), str(cu_bad))
    # 5 張仍在追蹤比例圖之文字（使用者核定）
    if CT:
        zhn = {'gongliao': '貢寮', 'guishan': '龜山島', 'hsinchu': '新竹外海', 'nanwan': '南灣', 'lienchiang': '連江'}
        tbad = []
        for n_, c_ in CT.items():
            zh_ = zhn[c_['site']]; g_ = ptext(kids[FMC[n_] - 1]); cp_ = ptext(kids[FMC[n_] + 1]); go_ = ptext(kb10[FMB[n_] - 1])
            slow_ = c_['site'] in ('gongliao', 'guishan', 'nanwan')
            if not g_.startswith('圖 %d 為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。' % n_): tbad.append((n_, '首句'))
            if '全部 100,000 顆虛擬質點合計，%s仍在追蹤之比例於回溯第' % zh_ not in g_ or '為全部 100,000' in g_: tbad.append((n_, '前綴'))
            if re.findall(r'\d+(?:\.\d+)?%?', g_.replace('每類 10,000 顆', '').replace('全部 100,000 顆', '')) != re.findall(r'\d+(?:\.\d+)?%?', go_.replace('全部 100,000 顆', '')): tbad.append((n_, '數字'))
            if not g_.endswith('降至 0%。十類曲線相近，其中不可回收塑膠略慢。' if slow_ else '降至 0%。十類曲線相近。'): tbad.append((n_, '結果句', slow_))
            if '小圖' in g_ or '全部類別' in g_ or '黑' in g_: tbad.append((n_, '導引段不應提小圖或全部線'))
            if cp_ != '圖%d %s回溯期間仍在追蹤之虛擬質點比例（十類）；小圖為回溯前 20 天之放大' % (n_, zh_) or rpr_has(kids[FMC[n_] + 1], 'keepLines') != rpr_has(kb10[FMB[n_] + 1], 'keepLines'): tbad.append((n_, '圖說'))
        b34_ = ptext(kids[34])
        chk('V10-13 5 張仍在追蹤比例圖之文字（使用者核定）：導引段首句「圖 N 為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。」；50%／10%／1% 句加前綴「全部 100,000 顆虛擬質點合計，」、數字不變；句末加「十類曲線相近」（貢寮、龜山島、南灣另加「，其中不可回收塑膠略慢」）；導引段不再提小圖；圖說加「（十類）；小圖為回溯前 20 天之放大」；區塊 34 改為「呈現十類虛擬質點各自於回溯期間內仍在追蹤之比例」',
            not tbad and '呈現十類虛擬質點各自於回溯期間內仍在追蹤之比例' in b34_ and '全部虛擬質點於回溯期間內仍在追蹤' not in b34_ and len(CT) == 5, str(tbad[:4]))
    # 參考資料（使用者要求依期中報告用法排序：中文在前、英文依第一作者姓氏／機構名 A→Z；內文為作者—年份括號式）
    ri0 = next(i_ for i_, p_ in enumerate(kids) if p_.tag == q('p') and ptext(p_) == '參考資料')
    ri1 = next(i_ for i_ in range(ri0 + 1, len(kids)) if kids[i_].tag == q('p') and ptext(kids[i_]).startswith('附錄　'))
    refs_ = [ptext(kids[i_]) for i_ in range(ri0 + 1, ri1)]
    def _k(t):
        m_ = re.match(r'^(.*?)[,.(]\s', t) or re.match(r'^(.*?)\(', t)
        nm_ = unicodedata.normalize('NFKD', t.split(',')[0] if ', ' in t.split('(')[0] and not re.match(r'^[A-Z][A-Za-z ]+\. \(', t) else t.split('. (')[0]).lower()
        nm_ = ''.join(c_ for c_ in nm_ if c_.isalpha() and c_.isascii())
        yr_ = re.search(r'\((\d{4})\)', t).group(1)
        return (nm_, yr_)
    zh_ = [t_ for t_ in refs_ if re.search(r'[\u4e00-\u9fff]', t_[:6])]; en_ = [t_ for t_ in refs_ if t_ not in zh_]
    order_ok = refs_[:len(zh_)] == zh_ and [_k(t_) for t_ in en_] == sorted(_k(t_) for t_ in en_)
    CITE = ['海洋委員會海洋保育署, 2025', 'Bagheri & Bonadonna, 2016', 'Cambridge University Engineering Department, 2003', 'Çengel & Cimbala, 2018', 'Duis & Coors, 2016', 'Ferguson & Church, 2004', 'Forest Products Laboratory, 2010', 'GESAMP, 2015', 'GESAMP, 2019', 'Goral et al., 2023', 'Ioakeimidis et al., 2014', 'ITTC, 2011', 'Kooi et al., 2017', 'Liu et al., 2022', 'MSFD Technical Subgroup on Marine Litter, 2013', 'Nikolaev, 2022', 'Prado & Dremière, 1990', 'Thompson & Taylor, 2008', 'Xu et al., 2024']
    body_ = '\n'.join(ptext(p_) for i_, p_ in enumerate(kids) if p_.tag == q('p') and not (ri0 <= i_ < ri1)) + '\n' + '\n'.join(ptext(p_) for i_, t_ in enumerate(kids) if t_.tag == q('tbl') for p_ in t_.iter(q('p')))
    miss_ = [c_ for c_ in CITE if c_ not in body_]
    pair_ = [(re.match(r'^(.*?)[,(]', t_).group(1).strip(' .'), re.search(r'\((\d{4})\)', t_).group(1)) for t_ in refs_]
    listed_ = {x_ for x_ in ['海洋委員會海洋保育署2025', 'Bagheri2016', 'Cambridge University Engineering Department2003', 'Çengel2018', 'Duis2016', 'Ferguson2004', 'Forest Products Laboratory2010', 'GESAMP2015', 'GESAMP2019', 'Goral2023', 'Ioakeimidis2014', 'ITTC2011', 'Kooi2017', 'Liu2022', 'MSFD Technical Subgroup on Marine Litter2013', 'Nikolaev2022', 'Prado1990', 'Thompson2008', 'Xu2024']}
    got_ = {a_ + b_ for a_, b_ in pair_}
    num_ok = all(kids[i_].find(q('pPr')).find(q('numPr')) is not None for i_ in range(ri0 + 1, ri1))
    sameas = all(sig(kids[i_], zm, relsM) == sig(kb10[i_], zb10, relsb10) for i_ in range(ri0, ri1))
    chk('V10-R 參考資料（使用者要求依期中報告用法）：共 %d 筆，中文 %d 筆置前、英文 %d 筆依第一作者姓氏／機構名 A→Z（同一作者依年份），已編號且懸掛縮排（沿用主檔段落格式）；內文引用為作者—年份括號式（「Xu et al., 2024」「Goral et al., 2023」「Çengel & Cimbala, 2018」等 19 種）且逐一對應清單、清單每筆皆在本文或附錄被引用（無遺漏、無多列）；本版未改動參考資料段（與主檔逐字、逐格式相同）' % (len(refs_), len(zh_), len(en_)),
        len(refs_) == 19 and len(zh_) == 1 and order_ok and not miss_ and got_ == listed_ and num_ok and sameas, str((order_ok, miss_, got_ ^ listed_, num_ok, sameas)))
    # final 之檢查結果沿用 v9（30 張圖未動）
    rbad2 = []; nchk10 = 0
    for s_ in SITEK:
        rj = json.load(open(os.path.join(FINAL9, s_, 'result.json'), encoding='utf-8'))
        if rj['fails']: rbad2.append((s_, rj['fails'][:2]))
        rows_ = list(_csv.DictReader(open(os.path.join(FINAL9, s_, 'checks.csv'), encoding='utf-8-sig'))); nchk10 += len(rows_)
        rbad2 += [(s_, r_['figure'], r_['panel'], r_['check']) for r_ in rows_ if r_['result'] == 'FAIL']
    imgd = [n_ for (s_, k_), n_ in FIGN10.items() if k_ not in ('all_b', 'all_c') and zm.read('word/' + relsM[figs_in(kids[FMC[n_]])[0][2]]) != zb10.read('word/' + relsb10[figs_in(kb10[FMB[n_]])[0][2]])]
    chk('V10-11 30 張多格來源分段機率圖之圖檔與現行主檔逐位元組相同（v10 不動圖）；final 之版面與數值檢查（共 %d 列）失敗 0' % nchk10, not imgd and not rbad2, str(imgd[:3]) + str(rbad2[:3]))

import shutil; shutil.rmtree(tmpd, ignore_errors=True)   # 清除 v2 重建之暫存目錄（約 17 MB）；v9 起每次執行後自行清理
fails = [r for r in res if not r[0]]
out = ['%s %s%s' % ('OK  ' if ok else 'FAIL', n, ('  -- ' + d) if (not ok and d) else '') for ok, n, d in res]
out += ['', '檢查對象：%s（sha256 %s）' % (os.path.basename(MERGED), sha(open(MERGED, 'rb').read())), '%s：%d 項，失敗 %d' % ('全部通過' if not fails else '有失敗', len(res), len(fails))]
open(os.path.join(HERE, 'check_merged_run09.report.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('\n'.join(out))
sys.exit(1 if fails else 0)
