#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【2026-10-25 起合併版為主檔，使用者已在 Word 手改；本腳本只用於重現 v2 基準（check_merged_run09.py 以子程序執行並比對），不可用來覆寫主檔。後續修改走 edit_v3_inplace.py 類之就地修改。】

合併 3-4-3_run09.docx（情境設計）與 執行成果_run09_含斯托克斯漂流.docx（執行成果）。

作法：以 3-4-3 為底稿，原 zip 成員逐位元組保留（僅 document.xml、
document.xml.rels、[Content_Types].xml 三個成員改寫，另新增 35 張圖片）；
document.xml 以字串切接，於「4、溯源停止邊界」段落之後插入「（二）分析成果」，
參考資料與附錄因此自然移到最後。成果段落以 3-4-3 之版面格式重建（不沿用成果檔之樣式 ID）。

用法：python3 merge_run09.py <輸出docx路徑>   （輸出路徑已存在則拒絕覆寫）
"""
import sys, os, re, zipfile, hashlib, json
from html import escape
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))          # .../成果報告_run09
SRC_A = os.path.join(BASE, '3-4-3_run09.docx')
SRC_B = os.path.join(BASE, '執行成果_run09_含斯托克斯漂流.docx')
SHA_A = '2266588cfaac1140250ef4989e8c7ef1207154bc153a2977c7487f3555d415ad'
SHA_B = 'a22d0be9ed895158dfe96ddde34ae2d5b9762f55eacae96b939d19b1734e1055'
SHIFT = 7            # 3-4-3 已有圖 1–7；成果圖 1–35 → 8–42
ANCHOR_PARAID = '016A1432'   # 3-4-3「4、溯源停止邊界」段，成果插在其後

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
q = lambda t: '{%s}%s' % (W, t)

def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

# ---------------------------------------------------------------- 3-4-3 版面格式（取自 3-4-3 之參考段落）
RPR_BODY = '<w:rPr><w:szCs w:val="28"/></w:rPr>'                                   # 3-4-3 內文段（如 para 9）
RPR_ITEM = '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:szCs w:val="28"/></w:rPr>'  # 3-4-3「1、」項目
RPR_CAP = '<w:rPr><w:color w:val="000000"/><w:szCs w:val="28"/></w:rPr>'           # 3-4-3 圖說
SP_BODY = '<w:spacing w:before="240" w:after="120" w:line="440" w:lineRule="exact"/>'

PPR_L2 = ('<w:pPr><w:keepNext/>' + SP_BODY + '<w:jc w:val="both"/><w:rPr><w:szCs w:val="28"/></w:rPr></w:pPr>')   # （一）同式
PPR_L3 = ('<w:pPr><w:pStyle w:val="afe"/><w:keepNext/>' + SP_BODY +
          '<w:ind w:left="980" w:hangingChars="150" w:hanging="420"/><w:jc w:val="both"/>'
          '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:szCs w:val="28"/></w:rPr></w:pPr>')  # 「1、」同式
PPR_L4 = ('<w:pPr><w:pStyle w:val="afe"/><w:keepNext/>' + SP_BODY +
          '<w:ind w:left="1680" w:hangingChars="250" w:hanging="700"/><w:jc w:val="both"/>'
          '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:szCs w:val="28"/></w:rPr></w:pPr>')  # 「（1）」：首行起於 980，與內文左緣對齊
PPR_ITEM = PPR_L4.replace('<w:keepNext/>', '')                                    # 整體結論各條（（1）…（6））
PPR_BODYP = ('<w:pPr>' + SP_BODY + '<w:ind w:leftChars="354" w:left="991"/><w:jc w:val="both"/>'
             '<w:rPr><w:szCs w:val="28"/></w:rPr></w:pPr>')                       # 內文段同 3-4-3 para 9
PPR_FIG = '<w:pPr><w:keepNext/><w:spacing w:before="240"/><w:jc w:val="center"/></w:pPr>'   # 3-4-3 圖片段
PPR_CAP = '<w:pPr><w:spacing w:before="40" w:after="160" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'  # 3-4-3 圖說段

def run(rpr, text):
    return '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (rpr, escape(text, quote=False))

def para(ppr, runs):
    return '<w:p>' + ppr + ''.join(runs) + '</w:p>'

DRAWING = ('<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
           '<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
           '<wp:docPr id="{did}" name="{name}" descr="{descr}"/>'
           '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
           '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
           '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{did}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
           '<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
           '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
           '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')

# ---------------------------------------------------------------- 文字轉換規則（全部登記，檢查程式據此驗證）
def shift_figs(s):
    return re.sub(r'圖( ?)(\d+)', lambda m: '圖%s%d' % (m.group(1), int(m.group(2)) + SHIFT), s)

TEXT_EDITS = [   # (標籤, 舊, 新)  皆須在成果檔原文中恰出現一次（作用於已去除行首全形空格之文字）
    ('E2 導言：情境矩陣改指回（一）',
     '情境矩陣由 5 處研究地點、10 類代表物件、20 個受體（5 個水平位置 × 4 個深度）、50 個到達時間與 10 個系集成員交叉而成，每顆',
     '情境設計如（一）所述，另以 10 個系集成員交叉，每顆'),
    ('E3 導言：章末→最後', '綜合說明與整體結論置於章末', '綜合說明與整體結論置於最後'),
    ('E5 貢寮：刪與前文重複之停止條件句', '虛擬質點穿越流場範圍邊界或抵達海岸時停止追蹤；貢寮之', '貢寮之'),
]
H3_MAP = {'逐地點成果': ('1', '逐地點成果：'), '綜合說明與結論': ('2', '綜合說明與結論：'), '整體結論': ('3', '整體結論：')}



# ================================================================ v2：分頁保持與大綱層級（後處理，只改段落屬性 pPr）
import unicodedata
from xml.sax.saxutils import unescape

PPR_ORDER = ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl', 'numPr', 'suppressLineNumbers',
             'pBdr', 'shd', 'tabs', 'suppressAutoHyphens', 'kinsoku', 'wordWrap', 'overflowPunct', 'topLinePunct', 'autoSpaceDE',
             'autoSpaceDN', 'bidi', 'adjustRightInd', 'snapToGrid', 'spacing', 'ind', 'contextualSpacing', 'mirrorIndents',
             'suppressOverlap', 'jc', 'textDirection', 'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle',
             'rPr', 'sectPr', 'pPrChange']
USABLE_PT = (16838 - 2 * 1440) / 20.0      # A4 扣上下邊界之可用高度（pt）
SAFETY_PT = 17.0                           # 估算誤差之保留量（約 0.24 in）
BODY_W_PT = (11906 - 2 * 1440 - 991) / 20.0   # 內文寬（左縮 991）
CAP_W_PT = (11906 - 2 * 1440) / 20.0

# 整段即為標題者：全文 → 大綱層級（w:outlineLvl 之 val：0 = 第 1 層）
L0 = {'Lagrangian系集逆向溯源'}
L1 = {'（一）情境矩陣與停止條件設計：', '（二）分析成果：'}
L2 = {'1、逐地點成果：', '2、綜合說明與結論：', '3、整體結論：', '推導流程'}
RE_L2_APX = re.compile(r'^([一二三四五六七八]、\S{2,20}|補充[一二]、\S{2,25})$')
RE_L3_SITE = re.compile(r'^（[1-5]）(貢寮|龜山島|新竹外海|南灣|連江)：$')
RE_L3_APX = re.compile(r'^步驟[一二三四五]　\S.{2,40}$')
STYLE_HEADS = {'參考資料', '附錄　十類海廢終端沉降速度之推導'}      # 樣式為 Heading 2（大綱層級由樣式決定，不動）


def _blocks(doc):
    toks = re.compile(r'<w:tbl>|<w:tbl |</w:tbl>|<w:p[ >/]|</w:p>')
    out, depth, tstart, pstart = [], 0, None, None
    for m in toks.finditer(doc):
        t = m.group(0)
        if t.startswith('<w:tbl'):
            if depth == 0: tstart = m.start()
            depth += 1
        elif t == '</w:tbl>':
            depth -= 1
            if depth == 0: out.append({'k': 'tbl', 'a': tstart, 'b': m.end()})
        elif depth == 0:
            if t.startswith('<w:p'):
                e = doc.index('>', m.start())
                if doc[e - 1] == '/': out.append({'k': 'p', 'a': m.start(), 'b': e + 1})
                else: pstart = m.start()
            else:
                out.append({'k': 'p', 'a': pstart, 'b': m.end()})
    for blk in out:
        if blk['k'] != 'p': continue
        s = doc[blk['a']:blk['b']]
        blk['text'] = ''.join(unescape(x) for x in re.findall(r'<w:t(?: [^>]*)?>([^<]*)</w:t>', s))
        blk['fig'] = '<w:drawing>' in s
        m2 = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"', s)
        blk['cy'] = int(m2.group(2)) if m2 else 0
        ms = re.match(r'<w:p(?: [^>]*)?>(?:<w:pPr>.*?<w:pStyle w:val="([^"]*)")?', s, re.S)
        blk['style'] = ms.group(1) if ms and ms.group(1) else None
    return out


def _lines(text, width_pt):
    lines, w = 1, 0.0
    for ch in text:
        cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
        if w + cw > width_pt: lines += 1; w = cw
        else: w += cw
    return lines


def _patch_ppr(s, adds):
    m = re.match(r'(<w:p(?: [^>]*)?>)(<w:pPr>(.*?)</w:pPr>)?', s, re.S)
    head, inner = m.group(1), (m.group(3) or '')
    frag = etree.fromstring('<w:pPr xmlns:w="%s">%s</w:pPr>' % (W, inner))
    names = [c.tag.split('}')[1] for c in frag]
    changed = False
    for nm, val in adds:
        if nm in names: continue
        el = etree.Element(q(nm))
        if val is not None: el.set(q('val'), val)
        idx, pos = PPR_ORDER.index(nm), len(frag)
        for k, c in enumerate(frag):
            if PPR_ORDER.index(c.tag.split('}')[1]) > idx: pos = k; break
        frag.insert(pos, el); names.insert(pos, nm); changed = True
    if not changed: return s
    newinner = ''.join(etree.tostring(c, encoding='unicode', with_tail=False) for c in frag).replace(' xmlns:w="%s"' % W, '')
    end = m.end()
    return head + '<w:pPr>' + newinner + '</w:pPr>' + s[end:]


def postprocess(doc):
    """依規則補 keepNext／keepLines／outlineLvl；回傳（新 document.xml, 計畫 dict）。"""
    blocks = _blocks(doc)
    apx = None
    for i, b in enumerate(blocks):
        if b['k'] == 'p' and b['text'] == '附錄　十類海廢終端沉降速度之推導': apx = i
    assert apx is not None
    plan = {'headings': [], 'tablecaps': [], 'figs': 0, 'intros': [], 'unmarked_runin': []}
    edits = {}      # block index → adds
    is_head = {}
    for i, b in enumerate(blocks):
        if b['k'] != 'p' or b['fig']: continue
        t = b['text']; lvl = None; direct = True
        if i == 0 and t in L0: lvl = 0
        elif t in L1: lvl = 1
        elif t in L2: lvl = 2
        elif i > apx and RE_L2_APX.match(t): lvl = 2
        elif RE_L3_SITE.match(t): lvl = 3
        elif i > apx and RE_L3_APX.match(t): lvl = 3
        elif t in STYLE_HEADS: lvl = 1; direct = False
        if lvl is not None:
            is_head[i] = lvl
            adds = [('keepNext', None), ('keepLines', None)] + ([('outlineLvl', str(lvl))] if direct else [])
            edits[i] = adds
            plan['headings'].append({'level': lvl, 'text': t, 'direct_outline': direct})
    for i, b in enumerate(blocks):
        if b['k'] == 'p' and re.match(r'^表(A?\d+) ', b['text']) and i + 1 < len(blocks) and blocks[i + 1]['k'] == 'tbl':
            edits[i] = [('keepNext', None), ('keepLines', None)]
            plan['tablecaps'].append(b['text'][:30])
    for i, b in enumerate(blocks):
        if b['k'] == 'p' and b['fig']:
            edits.setdefault(i, []).append(('keepNext', None)); plan['figs'] += 1
    lim = USABLE_PT - SAFETY_PT
    for i, b in enumerate(blocks):
        if not (b['k'] == 'p' and b['fig']) or i == 0: continue
        j = i - 1; p = blocks[j]
        if p['k'] != 'p' or p['fig'] or j in is_head or re.match(r'^(圖|表)(A?\d+) ', p['text']) or not p['text'].strip(): continue
        chain = []; h = j - 1
        while h >= 0 and h in is_head: chain.append(h); h -= 1
        cap = blocks[i + 1] if i + 1 < len(blocks) and blocks[i + 1]['k'] == 'p' and re.match(r'^圖\d+ ', blocks[i + 1]['text']) else None
        hh = sum(_lines(blocks[c]['text'], BODY_W_PT) * 22 + 18 for c in chain)
        ih = _lines(p['text'], BODY_W_PT) * 22 + 18
        fh = 12 + b['cy'] / 12700.0
        ch = (2 + _lines(cap['text'], CAP_W_PT) * 27 + 8) if cap else 0
        tot = hh + ih + fh + ch
        fits = tot <= lim
        adds = [('keepLines', None)] + ([('keepNext', None)] if fits else [])
        edits.setdefault(j, []).extend(adds)
        plan['intros'].append({'text': p['text'][:24], 'chain': [blocks[c]['text'] for c in chain], 'heads_pt': hh, 'intro_pt': ih,
                               'fig_pt': round(fh, 1), 'cap_pt': ch, 'total_pt': round(tot, 1), 'limit_pt': round(lim, 1), 'fits': fits})
    for i, b in enumerate(blocks):
        t = b['text'] if b['k'] == 'p' else ''
        if b['k'] == 'p' and i not in is_head and not b['fig'] and re.match(r'^(\d、|（[一二三四五六七八九十]）|步驟 ?[\d一二三四五]|補充[一二]、)', t) and len(t) > 12:
            plan['unmarked_runin'].append(t[:28])
    out, last = [], 0
    for i in sorted(edits):
        a, b_ = blocks[i]['a'], blocks[i]['b']
        out.append(doc[last:a]); out.append(_patch_ppr(doc[a:b_], edits[i])); last = b_
    out.append(doc[last:])
    plan['counts'] = {'headings': len(plan['headings']), 'with_outlineLvl_direct': sum(1 for h in plan['headings'] if h['direct_outline']),
                      'tablecaps': len(plan['tablecaps']), 'figs_keepNext': plan['figs'], 'intro_keepLines': len(plan['intros']),
                      'intro_keepNext': sum(1 for x in plan['intros'] if x['fits']), 'intro_no_fit': [x['text'] for x in plan['intros'] if not x['fits']],
                      'paragraphs_touched': len(edits)}
    return ''.join(out), plan


def main(out):
    assert sha(SRC_A) == SHA_A, '3-4-3 之 sha 與預期不同（可能被手改），停止'
    assert sha(SRC_B) == SHA_B, '執行成果之 sha 與預期不同（可能被手改），停止'
    if os.path.exists(out):
        sys.exit('輸出已存在，拒絕覆寫：' + out)

    za, zb = zipfile.ZipFile(SRC_A), zipfile.ZipFile(SRC_B)
    broot = etree.fromstring(zb.read('word/document.xml'))
    body = broot.find(q('body'))
    rels_b = {r.get('Id'): r.get('Target') for r in etree.fromstring(zb.read('word/_rels/document.xml.rels'))}

    # ---- 解析成果檔
    items = []
    for idx, p in enumerate(body):
        if p.tag != q('p'):
            continue
        ps = p.find(q('pPr'))
        st = ps.find(q('pStyle')).get(q('val')) if ps is not None and ps.find(q('pStyle')) is not None else None
        text = ''.join(t.text or '' for t in p.iter(q('t')))
        dr = list(p.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline'))
        d = {'src_idx': idx, 'style': st, 'text': text}
        if dr:
            ext = dr[0].find('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent')
            blip = next(p.iter('{%s}blip' % A_NS))
            d.update(kind='fig', cx=int(ext.get('cx')), cy=int(ext.get('cy')), target=rels_b[blip.get('{%s}embed' % R_NS)])
        elif idx == 0:
            d['kind'] = 'h1'
        elif st == '-':
            d['kind'] = 'h3'
        elif st == '2':
            d['kind'] = 'site'
        elif re.match(r'^圖\d+ ', text):
            d['kind'] = 'cap'
        elif ps is not None and ps.find(q('ind')) is not None and ps.find(q('ind')).get(q('hanging')):
            d['kind'] = 'item'
        else:
            d['kind'] = 'body'
        items.append(d)
    kinds = [d['kind'] for d in items]
    assert kinds.count('h1') == 1 and kinds.count('h3') == 3 and kinds.count('site') == 5
    assert kinds.count('fig') == 35 and kinds.count('cap') == 35 and kinds.count('item') == 6, {k: kinds.count(k) for k in set(kinds)}

    # ---- 轉換
    parts, trace, rels_add, media_add = [], [], [], []
    used_edits = {e[0]: 0 for e in TEXT_EDITS}
    site_no = 0
    figpend = None
    for ii, d in enumerate(items):
        k, txt0 = d['kind'], d['text']
        txt = txt0[2:] if txt0.startswith('　　') else txt0       # 去行首兩個全形空格（改由版面縮排）
        for lab, old, new in TEXT_EDITS:
            if old in txt:
                assert txt.count(old) == 1, lab
                txt = txt.replace(old, new); used_edits[lab] += 1
        rec = {'src_idx': d['src_idx'], 'kind': k, 'orig': txt0}
        if k == 'h1':
            xml = para(PPR_L2, [run(RPR_BODY, '（二）'), run(RPR_BODY, '分析成果：')])
            new = '（二）分析成果：'; rec['edit'] = 'E1 原標題「Lagrangian系集逆向溯源分析成果」降階為（二）'
        elif k == 'h3':
            n, t = H3_MAP[txt]
            xml = para(PPR_L3, [run(RPR_ITEM, n), run(RPR_ITEM, '、'), run(RPR_ITEM, t)])
            new = n + '、' + t; rec['edit'] = 'E4 小標改為 3-4-3 之「1、」層級'
        elif k == 'site':
            site_no += 1
            xml = para(PPR_L4, [run(RPR_ITEM, '（%d）' % site_no), run(RPR_ITEM, txt + '：')])
            new = '（%d）%s：' % (site_no, txt); rec['edit'] = 'E4 研究地點小標改為「（1）」層級'
        elif k == 'item':
            m = re.match(r'^(\d)、', txt); assert m
            body_t = txt[m.end():]
            xml = para(PPR_ITEM, [run(RPR_ITEM, '（%s）' % m.group(1)), run(RPR_ITEM, body_t)])
            new = '（%s）%s' % (m.group(1), body_t); rec['edit'] = 'E4 條列編號「1、」改「（1）」以免與上層「1、」混淆'
        elif k == 'body':
            new = shift_figs(txt)
            xml = para(PPR_BODYP, [run(RPR_BODY, new)])
        elif k == 'fig':
            figpend = d
            xml = None
            new = ''
        elif k == 'cap':
            m = re.match(r'^圖(\d+) (.*)$', txt, re.S); assert m
            n_old = int(m.group(1)); n_new = n_old + SHIFT
            assert figpend is not None
            rid = 'rId%d' % (100 + n_new)
            src_media = figpend['target']                       # media/figN.jpg
            dst_media = 'media/fig%d.jpg' % n_new
            rels_add.append((rid, dst_media)); media_add.append(('word/' + dst_media, 'word/' + src_media))
            capt = '圖%d %s' % (n_new, m.group(2))
            figxml = para(PPR_FIG, [DRAWING.format(cx=figpend['cx'], cy=figpend['cy'], did=9700 + n_new,
                                                    name='fig%d.jpg' % n_new, descr=escape(capt, quote=True), rid=rid)])
            capxml = para(PPR_CAP, [run(RPR_CAP, '圖'), run(RPR_CAP, '%d ' % n_new), run(RPR_CAP, m.group(2))])
            parts.append(figxml)
            trace.append({'src_idx': figpend['src_idx'], 'kind': 'fig', 'orig': '(image %s, %dx%d EMU)' % (figpend['target'], figpend['cx'], figpend['cy']),
                          'new': '(image %s, %dx%d EMU)' % (dst_media, figpend['cx'], figpend['cy']), 'edit': '圖號 %d→%d' % (n_old, n_new)})
            xml = capxml; new = capt; rec['edit'] = '圖號 %d→%d' % (n_old, n_new)
            figpend = None
        if xml is None:
            continue
        rec['new'] = new
        if k == 'body':
            labs = [l for l, o, n_ in TEXT_EDITS if o in txt0]
            if shift_figs(txt) != txt:
                labs.append('圖號順延 +%d' % SHIFT)
            if labs:
                rec['edit'] = '；'.join(labs)
        parts.append(xml); trace.append(rec)
    assert all(v == 1 for v in used_edits.values()), used_edits
    new_xml = ''.join(parts)

    # ---- 切接 document.xml
    doc = za.read('word/document.xml').decode('utf-8')
    i = doc.index('<w:p w14:paraId="%s"' % ANCHOR_PARAID)
    j = doc.index('</w:p>', i) + len('</w:p>')
    assert '溯源停止邊界' in doc[i:j]
    doc2 = doc[:j] + new_xml + doc[j:]
    doc2, plan = postprocess(doc2)

    rels = za.read('word/_rels/document.xml.rels').decode('utf-8')
    add = ''.join('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="%s"/>' % (r, t) for r, t in rels_add)
    assert all(('Id="%s"' % r) not in rels for r, _ in rels_add)
    rels2 = rels.replace('</Relationships>', add + '</Relationships>')
    ct = za.read('[Content_Types].xml').decode('utf-8')
    assert 'Extension="jpg"' not in ct
    ct2 = ct.replace('<Default Extension="png" ContentType="image/png"/>', '<Default Extension="png" ContentType="image/png"/><Default Extension="jpg" ContentType="image/jpeg"/>')
    assert ct2 != ct
    repl = {'word/document.xml': doc2.encode('utf-8'), 'word/_rels/document.xml.rels': rels2.encode('utf-8'), '[Content_Types].xml': ct2.encode('utf-8')}

    # ---- 寫出 zip（原成員逐位元組保留；ZipInfo 沿用）
    tmp = out + '.partial'
    with zipfile.ZipFile(tmp, 'w') as zo:
        for info in za.infolist():
            zi = zipfile.ZipInfo(info.filename, info.date_time)
            zi.compress_type = info.compress_type; zi.external_attr = info.external_attr; zi.create_system = info.create_system
            zo.writestr(zi, repl.get(info.filename, za.read(info.filename)))
        for dst, src in media_add:
            zi = zipfile.ZipInfo(dst, (1980, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 25165824; zi.create_system = 0
            zo.writestr(zi, zb.read(src))
    os.replace(tmp, out)
    json.dump(trace, open(os.path.join(HERE, 'merge_trace.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(plan, open(os.path.join(HERE, 'keep_plan.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('plan:', json.dumps(plan['counts'], ensure_ascii=False))
    print('OK', out, os.path.getsize(out), 'bytes; sha256', sha(out))
    print('inserted paragraphs:', len(parts), ' images:', len(media_add), ' trace rows:', len(trace))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
