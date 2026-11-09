#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v11 候選：研究地點順序改為 龜山島、貢寮、新竹、南灣、連江（依期中報告圖 2-17 紅圈 1–5），並「新竹外海」→「新竹」。
以「現行主檔」（v10 升格後使用者再存：標題去冒號、新竹標題、刪（6）綜合段；sha256 a0f99778…）為基準，就地修改 word/document.xml；
圖 1、圖 2（地點字母 (a)–(e) 隨順序改變）之 word/media 兩檔改為重畫版，其餘 zip 成員逐位元組保留。

操作（皆整區塊搬移，不重畫圖）：
 (1) 表 2：貢寮列與龜山島列對調；圖 2 圖說／替代文字之字母對應 (a) 龜山島、(b) 貢寮；到達時間驗證圖導引句與圖 3、圖 4 區塊對調；
 (2) （二）逐地點成果：貢寮區塊（區塊 36–70）與龜山島區塊（71–104）對調；圖號 3↔4、8–18↔19–29（全文所有「圖 N」同步，含替代文字）；
 (3) 第一個地點才有之說明改隨新的第一個地點（龜山島）：線色說明、穿越比例／海岸來源比例定義、停留時數密度與高密度區間定義、來源分段取法段、沉底時刻與季節定義、大小潮定義句；
     貢寮改為其餘地點之簡短寫法；
 (4) 整體結論（1）（2）對調並重編號；綜合說明「季節之影響明顯」句依新順序；
 (5) 全文「新竹外海」→「新竹」（含圖說、替代文字、docPr 名稱）；(5b) 龜山島作為海岸來源之 9 處（「龜山島沿岸」7、「龜山島占 x%」2）→「龜山島周圍海岸」；
 (6) 受影響導引段之 keepNext 依 v8 起之估算式（fits）重算。
用法：edit_v11_inplace.py <現行主檔> <新圖1.png> <新圖2.png> <輸出docx> <log.json>
前提：主檔 sha256 須以 a0f9977871d44f6c 起首且區塊數 357（含 sectPr），否則停止。主檔不被修改。"""
import sys, os, re, zipfile, hashlib, json, unicodedata
from lxml import etree

PRE = 'a0f9977871d44f6c'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
XML = 'http://www.w3.org/XML/1998/namespace'
q = lambda t: '{%s}%s' % (W, t)
LIM = (16838 - 2880) / 20.0 - 17.0
BW = (11906 - 2880 - 991) / 20.0
CW = (11906 - 2880) / 20.0


def nl(text, wpt):
    ln, w = 1, 0.0
    for ch in text:
        cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
        if w + cw > wpt:
            ln += 1; w = cw
        else:
            w += cw
    return ln


def fits(text, fig_cy, cap_text):
    tot = nl(text, BW) * 22 + 18 + 12 + fig_cy / 12700.0 + (2 + nl(cap_text, CW) * 27 + 8)
    return tot <= LIM, tot


def fmap(n):
    """舊圖號 → 新圖號（龜山島與貢寮互換；圖 3、圖 4 互換）"""
    if n == 3: return 4
    if n == 4: return 3
    if 8 <= n <= 18: return n + 11
    if 19 <= n <= 29: return n - 11
    return n


def ptx(p):
    return ''.join(t.text or '' for t in p.iter(q('t')))


def dtx(p):
    """直屬 run 之文字串接（rs 之座標系）"""
    return ''.join(t.text or '' for r in p.findall(q('r')) for t in r.findall(q('t')))


def rs(p, spans):
    """於段落（直屬 run）以 (a, b, new) 取代字元區間；由右至左施行。新文字落在區間首字所在 run（沿用其格式），區間內其餘字元自各 run 刪除。"""
    for a, b, new in sorted(spans, key=lambda s: -s[0]):
        assert a < b, (a, b)
        ts = [t for r in p.findall(q('r')) for t in r.findall(q('t'))]
        pos = 0; first = True
        for t in ts:
            tx = t.text or ''; lo, hi = pos, pos + len(tx); pos = hi
            if hi <= a or lo >= b: continue
            o0 = max(a, lo) - lo; o1 = min(b, hi) - lo
            t.text = (tx[:o0] + new + tx[o1:]) if first else (tx[:o0] + tx[o1:]); first = False
            t.set('{%s}space' % XML, 'preserve')
        assert not first, (a, b)


def sub1(p, old, new):
    """唯一出現之 old → new"""
    s = dtx(p); assert s.count(old) == 1, (old, s.count(old), s[:60])
    a = s.index(old); rs(p, [(a, a + len(old), new)])


def main(src, fig1, fig2, dst, logp):
    raw = open(src, 'rb').read()
    assert hashlib.sha256(raw).hexdigest().startswith(PRE), '主檔 sha 與預期（a0f99778…）不同，停止'
    zin = zipfile.ZipFile(src)
    root = etree.fromstring(zin.read('word/document.xml')); body = root.find(q('body')); kids = list(body)
    assert len(kids) == 357, len(kids)
    T = lambda i: ptx(kids[i])
    log = dict(base_sha_prefix=PRE, paras=[], fig_map={}, keep_flips=[], notes=[])
    old_text = {i: ptx(k) for i, k in enumerate(kids) if k.tag == q('p')}

    # ------------------------------------------------------------ 前提斷言
    for i, nm in ((36, '貢寮'), (71, '龜山島'), (105, '新竹'), (139, '南灣'), (173, '連江')): assert T(i) == nm, (i, T(i))
    assert T(46).startswith('圖 11 至圖 18 為貢寮之來源分段機率圖')
    assert T(81).startswith('圖 22 為龜山島全年（十類合併）')
    assert T(212).startswith('（1）貢寮：') and T(213).startswith('（2）龜山島：')
    assert T(217) == '參考資料'
    assert T(20).startswith('以下列出各研究地點之到達時間驗證圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5')
    assert T(17).endswith('(a) 貢寮、(b) 龜山島、(c) 新竹外海、(d) 南灣、(e) 連江')
    assert T(22) == '圖3 貢寮到達時間驗證圖' and T(24) == '圖4 龜山島到達時間驗證圖'
    skip_f = {id(kids[46]), id(kids[20])}      # 搬移段與到達時間導引句另行處理

    # ------------------------------------------------------------ (1) 表 2 列對調
    tbl = kids[13]; trs = tbl.findall(q('tr'))
    assert ''.join(trs[1].itertext()).startswith('貢寮') and ''.join(trs[2].itertext()).startswith('龜山島')
    trs[1].addprevious(trs[2])
    assert ''.join(tbl.findall(q('tr'))[1].itertext()).startswith('龜山島')

    # ------------------------------------------------------------ 圖號與名稱：全文逐段處理（不含搬移段、到達時間導引句）
    n_f = 0
    for p in body.iter(q('p')):
        if id(p) in skip_f: continue
        s = dtx(p); sp = []
        for m in re.finditer(r'圖 ?(\d+)', s):
            n = int(m.group(1)); nn = fmap(n)
            if nn != n: sp.append((m.start(1), m.end(1), str(nn)))
        if sp: rs(p, sp); n_f += len(sp)
    log['n_fig_ref_renumbered'] = n_f

    # ------------------------------------------------------------ 區塊重排
    # 到達時間驗證圖：[21,22] 貢寮、[23,24] 龜山島
    kids[21].addprevious(kids[23]); kids[21].addprevious(kids[24])
    # （二）：A=貢寮 36–70，B=龜山島 71–104；搬移段 46 自 A 移到 B 之「圖 22→11 導引段（原 81）」前
    A = kids[36:71]; B = kids[71:105]
    p46 = kids[46]; A.remove(p46); B.insert(B.index(kids[81]), p46)
    anchor = kids[105]
    for e in B + A: anchor.addprevious(e)
    # 整體結論 (1)(2)
    kids[212].addprevious(kids[213])
    c1, c2 = kids[213], kids[212]       # c1 = 龜山島（新 (1)），c2 = 貢寮（新 (2)）
    sub1(c1, '（2）龜山島：', '（1）龜山島：'); sub1(c2, '（1）貢寮：', '（2）貢寮：')

    # ------------------------------------------------------------ 到達時間導引句、圖 2 圖說
    p20 = kids[20]
    sub1(p20, '貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5', '龜山島見圖 3，貢寮見圖 4，新竹見圖 5')
    sub1(kids[17], '(a) 貢寮、(b) 龜山島、(c) 新竹外海', '(a) 龜山島、(b) 貢寮、(c) 新竹')

    # ------------------------------------------------------------ (3) 第一個地點才有之說明
    Bn = {i: p for i, p in enumerate(B)}   # 僅為取用方便
    def find(par_list, prefix):
        r = [p for p in par_list if ptx(p).startswith(prefix)]; assert len(r) == 1, (prefix, len(r)); return r[0]
    # 新龜山島（B 區塊，現為圖 8–18）、新貢寮（A 區塊，現為圖 19–29）
    # --- 軌跡導引段
    bt = find(B, '圖 8 為龜山島隨機抽取之 24 條代表虛擬質點回溯軌跡。')
    sub1(bt, '回溯軌跡。', '回溯軌跡，線色依起始深度由淺至深。')
    at = find(A, '圖 19 為隨機抽取之 24 條代表虛擬質點回溯軌跡，線色依起始深度由淺至深。')
    sub1(at, '為隨機抽取之 24 條代表虛擬質點回溯軌跡，線色依起始深度由淺至深。', '為貢寮隨機抽取之 24 條代表虛擬質點回溯軌跡。')
    # --- 仍在追蹤比例導引段（定義穿越比例與海岸來源比例）
    bc = find(B, '圖 9 為十類虛擬質點'); ac = find(A, '圖 20 為十類虛擬質點')
    sub1(bc, '龜山島之穿越比例為 78.35%，海岸來源比例為 21.65%。', '龜山島之虛擬質點穿越流場範圍邊界者占 78.35%，稱為穿越比例，抵達海岸者占 21.65%，稱為海岸來源比例。')
    sub1(ac, '貢寮之虛擬質點穿越流場範圍邊界者占 73.40%，稱為穿越比例，抵達海岸者占 26.60%，稱為海岸來源比例。', '貢寮之穿越比例為 73.40%，海岸來源比例為 26.60%。')
    # --- 停留時數密度導引段（圖 10／21）
    bf = find(B, '圖 10 中，龜山島之 50%'); af = find(A, '圖 21 之底色為停留時數密度：')
    a_s = dtx(af); k = a_s.index('貢寮之 50%、75%、90% 高密度區間面積分別為')
    a_def = a_s[:k]; a_tail = a_s[k:]
    assert a_def.startswith('圖 21 之底色為停留時數密度：') and a_def.endswith('表示分布於較廣之海域。'), a_def[-30:]
    b_def = '圖 10 之底色' + a_def[len('圖 21 之底色'):]     # 新龜山島之圖 10
    b_s = dtx(bf); assert b_s.startswith('圖 10 中，龜山島之 50%')
    # 新貢寮：「圖 21 中，」+ 貢寮數據；新龜山島：定義 + 龜山島數據（去「圖 21 中，」）
    rs(af, [(0, k, '圖 21 中，')])
    rs(bf, [(0, len('圖 10 中，'), b_def)])
    # --- 來源分段取法段：移至新龜山島
    sub1(p46, '為貢寮之來源分段機率圖', '為龜山島之來源分段機率圖')
    # --- 沉底時刻、季節定義（圖 15／26）
    bs = find(B, '圖 15 為依沉底時刻劃分之龜山島四季邊界穿越來源分段機率圖。'); as_ = find(A, '圖 26 為依沉底時刻劃分之貢寮四季邊界穿越來源分段機率圖。')
    s_a = dtx(as_); i0 = s_a.index('沉底時刻為廢棄物沉降至海床之時刻'); i1 = s_a.index('各季節之穿越比例及穿越最集中之邊界段為：')
    defn = s_a[i0:i1]; assert defn.endswith('秋季為 9–11 月。') and '由於廢棄物實際何時沉底未知' in defn, defn
    rs(as_, [(i0, i1, '')])
    sub1(bs, '四季邊界穿越來源分段機率圖。', '四季邊界穿越來源分段機率圖。' + defn)
    # --- 大小潮定義句（圖 17、圖 18／28、29）
    LONG = '依沉底時刻之潮位資料劃分為大潮與小潮之'; SHORT = '依大小潮劃分之'
    a17 = find(A, '圖 28 為' + LONG + '貢寮邊界穿越來源分段機率圖。'); a18 = find(A, '圖 29 為' + LONG + '貢寮海岸來源分段機率圖。')
    b17 = find(B, '圖 17 為' + SHORT + '龜山島邊界穿越來源分段機率圖。'); b18 = find(B, '圖 18 為' + SHORT + '龜山島海岸來源分段機率圖。')
    sub1(a17, LONG, SHORT); sub1(a18, LONG, SHORT); sub1(b17, SHORT, LONG); sub1(b18, SHORT, LONG)

    # ------------------------------------------------------------ 綜合說明
    p208 = kids[208]
    sub1(p208, '貢寮、新竹外海、南灣與連江四季兩兩之間差異明顯；龜山島僅冬、春兩季相近，其餘季節兩兩之間差異明顯。',
         '龜山島僅冬、春兩季相近，其餘季節兩兩之間差異明顯；貢寮、新竹、南灣與連江四季兩兩之間差異明顯。')

    # ------------------------------------------------------------ 「新竹外海」→「新竹」
    n_ren = 0
    for p in body.iter(q('p')):
        s = dtx(p); sp = [(m.start(), m.end(), '新竹') for m in re.finditer('新竹外海', s)]
        if sp: rs(p, sp); n_ren += len(sp)
    log['n_hsinchu_renamed_text'] = n_ren
    # ------------------------------------------------------------ (5b) 龜山島作為海岸來源（非研究地點）→「龜山島周圍海岸」（使用者核定）
    n_gs = 0; n_gz = 0; gs_hits = []
    for p in body.iter(q('p')):
        s = dtx(p); sp = []
        for m in re.finditer('龜山島沿岸', s): sp.append((m.start(), m.end(), '龜山島周圍海岸')); n_gs += 1
        for m in re.finditer(r'龜山島占(?= \d)', s): sp.append((m.start(), m.end(), '龜山島周圍海岸占')); n_gz += 1
        if sp:
            gs_hits.append(s[:12]); rs(p, sp)
    assert (n_gs, n_gz) == (7, 2), (n_gs, n_gz)
    log['n_guishan_coast_rename'] = dict(yanan=n_gs, zhan=n_gz, paras=gs_hits)
    assert '龜山島沿岸' not in ''.join(t.text or '' for t in root.iter(q('t')))
    # 屬性：圖號與名稱
    n_attr = 0
    for tag in ('{%s}docPr' % WP, '{%s}cNvPr' % PIC):
        for e in body.iter(tag):
            for a in ('name', 'descr'):
                v = e.get(a)
                if v is None: continue
                nv = v
                if a == 'name':
                    m = re.match(r'^fig(\d+)\.(png|jpg)$', v)
                    if m: nv = 'fig%d.%s' % (fmap(int(m.group(1))), m.group(2))
                else:
                    nv = re.sub(r'圖(\d+)', lambda m: '圖%d' % fmap(int(m.group(1))), v)
                nv = nv.replace('新竹外海', '新竹')
                nv = nv.replace('(a) 貢寮、(b) 龜山島', '(a) 龜山島、(b) 貢寮')
                if nv != v: e.set(a, nv); n_attr += 1
    log['n_attr_changed'] = n_attr
    assert '新竹外海' not in etree.tostring(root, encoding='unicode')

    # ------------------------------------------------------------ (6) keepNext 重算（只動導引段文字有變者）
    flips = []
    for i, p in enumerate(list(body)):
        if p.tag != q('p'): continue
        for inl in p.iter('{%s}inline' % WP):
            m = re.match(r'^fig(\d+)\.(?:png|jpg)$', inl.find('{%s}docPr' % WP).get('name'))
            if not m: continue
            n = int(m.group(1)); g = list(body)[i - 1]; cp = list(body)[i + 1]
            cy = int(inl.find('{%s}extent' % WP).get('cy'))
            ppr = g.find(q('pPr')); kn = ppr.find(q('keepNext'))
            fit, tot = fits(ptx(g), cy, ptx(cp))
            if fit != (kn is not None):
                flips.append(dict(fig=n, old=kn is not None, new=fit, tot=round(tot, 1)))
                if kn is not None: ppr.remove(kn)
                else: ppr.insert(1 if ppr.find(q('pStyle')) is not None else 0, etree.Element(q('keepNext')))
    log['keep_flips'] = flips

    # ------------------------------------------------------------ 輸出
    kk = list(body)
    for i, p in enumerate(kk):
        if p.tag == q('p'):
            t = ptx(p)
            log['paras'].append(t)
    log['n_blocks'] = len(kk)
    xml_out = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    new_media = {'word/media/image1.png': open(fig1, 'rb').read(), 'word/media/image2.png': open(fig2, 'rb').read()}
    for k_, d_ in new_media.items(): assert d_[:8] == b'\x89PNG\r\n\x1a\n'
    zout = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
    for zi in zin.infolist():
        data = zin.read(zi.filename)
        if zi.filename == 'word/document.xml': data = xml_out
        elif zi.filename in new_media: data = new_media[zi.filename]
        ni = zipfile.ZipInfo(zi.filename, date_time=zi.date_time); ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
        zout.writestr(ni, data)
    zout.close()
    log['media_sha256'] = {k_: hashlib.sha256(d_).hexdigest() for k_, d_ in new_media.items()}
    json.dump(log, open(logp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('ok', dst, 'blocks', len(kk), 'figrefs', n_f, 'renamed', n_ren, 'attrs', n_attr, 'flips', flips)


if __name__ == '__main__':
    main(*sys.argv[1:6])
