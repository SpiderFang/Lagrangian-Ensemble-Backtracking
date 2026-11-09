#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_slides_v20261106.py - 由「Lagrangian逆向溯源分析_精簡版.docx」產生 16:9 簡報 (pptx)。
用法：python3 build_slides_v20261106.py
可用環境變數覆寫：LBT_DOCX（報告 docx 路徑）、LBT_OUT（輸出 pptx 路徑）、SLIDEWORK（暫存資料夾）。
簡報文字均取自報告（以 Q() 逐字核對；圖表編號、附件編號改寫為頁面用語，改寫處以 N() 標示）。
"""
import os, re, io, math, sys, tempfile, zipfile, shutil, json
from PIL import Image
import docx
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn as pqn
from lxml import etree

HOME = os.path.expanduser('~')
WS = None
for base in (os.path.join(HOME, 'mnt', 'Workspace'), os.path.join(HOME, 'Workspace')):
    if os.path.isdir(base):
        WS = base
        break
DOCX = os.environ.get('LBT_DOCX') or os.path.join(WS, '未命名檔案夾', '精簡版', 'Lagrangian逆向溯源分析_精簡版.docx')
OUT_DIR = os.path.join(WS, 'Lagrangian-Ensemble-Backtracking', 'outputs', '成果報告_run09', '簡報') if WS else '.'
OUT = os.environ.get('LBT_OUT') or os.path.join(OUT_DIR, '成果報告_run09_簡報_v20261106.pptx')
WORK = os.environ.get('SLIDEWORK') or os.path.join(tempfile.gettempdir(), 'slidework_v20261106')
os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---------- 色彩與字型 ----------
BG, TITLE, BODY, MUTED, ACC = 'F7F5EF', '111827', '374151', '64748B', '155DA8'
RED, GREEN, GRAY, BLUE, LINE = 'B91C1C', '0F766E', 'F3F4F6', 'EAF5FB', 'C8D0DB'
FONT = 'Noto Sans TC'
SW, SH = 13.333, 7.5
MX, CW = 0.6, 12.133          # 左邊界與內容寬
TOP, BOT = 1.32, 6.95         # 內容區上下界
WARN = []
NOTQ = []                     # 非逐字（改寫／連接）文字，供人工複核

# ---------- 讀報告：全文（含 OMML 與欄位文字） ----------
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
_doc = docx.Document(DOCX)
_body = _doc.element.body
def ptext(p):
    return ''.join((n.text or '') for n in p.iter() if n.tag in (qn('w:t'), M + 't'))
PARAS = [ptext(p) for p in _body.iter(qn('w:p'))]
FULL = '\n'.join(PARAS)

def Q(s):
    """回傳 s；若 s 不是報告原文的逐字片段就中止。"""
    if s not in FULL:
        raise SystemExit('NOT IN REPORT: ' + s)
    return s

def N(s):
    """非逐字（連接詞、標題、改寫圖表編號）文字；記錄供複核。"""
    NOTQ.append(s)
    return s

# ---------- 由 docx 依序抽出原始圖片（不裁切、不重新編碼） ----------
def extract_images():
    d = os.path.join(WORK, 'img'); os.makedirs(d, exist_ok=True)
    els = list(_body.iterchildren()); rels = _doc.part.rels
    BL = '{http://schemas.openxmlformats.org/drawingml/2006/main}blip'
    found = {}; pend = []
    for el in els:
        if el.tag != qn('w:p'): continue
        bl = list(el.iter(BL))
        for b in bl: pend.append(rels[b.get(qn('r:embed'))].target_part)
        t = ''.join(x.text or '' for x in el.iter(qn('w:t')))
        m = re.match(r'圖\s*(附3-)?(\d+)', t)
        if m and not bl:
            key = ('A' if m.group(1) else 'F') + m.group(2)
            for k, part in enumerate(pend):
                ext = os.path.splitext(part.partname)[1]
                fn = os.path.join(d, '%s_%d%s' % (key, k, ext))
                open(fn, 'wb').write(part.blob)
                found['%s_%d' % (key, k)] = fn
            pend = []
    return found
IMG = extract_images()

# ---------- 文字寬度估計（避免溢出） ----------
def _cw(c):
    o = ord(c)
    if o < 0x80:
        if c == ' ': return 0.28
        if c in '.,:;|!\'`': return 0.3
        if c in '()[]/-': return 0.4
        if c.isdigit(): return 0.58
        if c.isupper(): return 0.66
        return 0.55
    if o in (0xB0, 0xD7, 0xB1, 0x2013, 0x2014, 0x2212): return 0.6
    return 1.0

def est_lines(text, size, width_in):
    em = sum(_cw(c) for c in text) * size / 72.0
    return max(1, math.ceil(em / max(width_in, 0.1) * 1.07))

# ---------- 樣式輔助 ----------
def rgb(h): return RGBColor.from_string(h)

def style_run(r, size, color=BODY, bold=False, sup=False, sub=False):
    f = r.font
    f.size = Pt(size); f.bold = bool(bold); f.color.rgb = rgb(color)
    rPr = r._r.get_or_add_rPr()
    rPr.set('lang', 'zh-TW')
    if sup: rPr.set('baseline', '30000')
    if sub: rPr.set('baseline', '-25000')
    for tag in ('a:latin', 'a:ea', 'a:cs'):
        for e in rPr.findall(pqn(tag)): rPr.remove(e)
        etree.SubElement(rPr, pqn(tag)).set('typeface', FONT)

def _norm_runs(t):
    if isinstance(t, str): return [(t, {})]
    out = []
    for x in t:
        out.append((x, {}) if isinstance(x, str) else x)
    return out

def fill_tf(tf, paras, size=16, color=BODY, label='', box=None):
    """paras: list of dict(t=str|runs, size, color, bold, bullet, after, before, align, line)"""
    total_h = 0.0
    for i, pd in enumerate(paras):
        if isinstance(pd, str): pd = {'t': pd}
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sz = pd.get('size', size); col = pd.get('color', color); bd = pd.get('bold', False)
        line = pd.get('line', 1.0)
        p.alignment = {'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}.get(pd.get('align'), PP_ALIGN.LEFT)
        p.line_spacing = line
        p.space_after = Pt(pd.get('after', 0)); p.space_before = Pt(pd.get('before', 0))
        plain = ''
        for txt, o in _norm_runs(pd['t']):
            r = p.add_run(); r.text = txt; plain += txt
            style_run(r, o.get('size', sz), o.get('color', col), o.get('bold', bd), o.get('sup', False), o.get('sub', False))
        indent = 0.0
        if pd.get('bullet'):
            indent = pd.get('indent', 0.26)
            pPr = p._p.get_or_add_pPr()
            pPr.set('marL', str(int(Inches(indent)))); pPr.set('indent', str(-int(Inches(indent))))
            c = etree.SubElement(pPr, pqn('a:buClr')); etree.SubElement(c, pqn('a:srgbClr')).set('val', pd.get('bucolor', ACC))
            etree.SubElement(pPr, pqn('a:buFont')).set('typeface', 'Arial')
            etree.SubElement(pPr, pqn('a:buChar')).set('char', '•')
        else:
            etree.SubElement(p._p.get_or_add_pPr(), pqn('a:buNone'))
        if box:
            wi = box[0] - indent
            total_h += est_lines(plain, sz, wi) * sz * 1.45 * line / 72.0 + (pd.get('after', 0) + pd.get('before', 0)) / 72.0
    if box and total_h > box[1] + 0.02:
        WARN.append('OVERFLOW? %s est %.2f > %.2f in' % (label, total_h, box[1]))
    return total_h

def fit_size(texts, w, h, sizes, after, hdr_h=0.0, indent=0.26):
    for sz in sizes:
        tot = hdr_h + sum(est_lines(t, sz, w - indent - 0.1) * sz * 1.45 / 72.0 + after / 72.0 for t in texts)
        if tot <= h: return sz
    return sizes[-1]

CLOSERS = set('）)」』】》。，、；：！？%')
OPENERS = set('（(「『【《')
def _tokens(text):
    return re.findall(r'[A-Za-z0-9.,%°–\-:/<>×+=_~]+|\s|.', text)

def has_bad_break(text, size, width_in):
    """模擬貪婪斷行；若結束括號/句號等落在行首，或開括號落在行尾，回傳 True。"""
    avail = width_in * 72.0 / size
    line = 0.0; prev = None
    for t in _tokens(text):
        w = sum(_cw(c) for c in t)
        if t.isspace():
            line += w; continue
        if line + w > avail + 1e-6 and line > 0:
            if t[0] in CLOSERS: return True
            if prev and prev[-1] in OPENERS: return True
            line = 0.0
        line += w; prev = t
    return False

def pick_extra(paras, w, margin):
    items = []
    for pd in paras:
        if isinstance(pd, str): pd = {'t': pd}
        plain = ''.join(x if isinstance(x, str) else x[0] for x in (pd['t'] if not isinstance(pd['t'], str) else [pd['t']]))
        items.append((plain, pd.get('size', 16), (pd.get('indent', 0.26) if pd.get('bullet') else 0.0)))
    base = w - margin[0] - margin[1]
    for k in range(0, 16):
        ex = k * 0.04
        if not any(has_bad_break(t, sz, base - ex - ind) for t, sz, ind in items): return ex
    return 0.0

def set_frame(tf, ml=0.1, mr=0.1, mt=0.05, mb=0.05, anchor='t'):
    tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left, tf.margin_right, tf.margin_top, tf.margin_bottom = Inches(ml), Inches(mr), Inches(mt), Inches(mb)
    tf.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]

def tbox(slide, x, y, w, h, paras, size=16, color=BODY, anchor='t', margin=(0.05, 0.05, 0.03, 0.03), label=''):
    margin = (margin[0], margin[1] + pick_extra(paras, w, margin), margin[2], margin[3])
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_frame(tb.text_frame, *margin, anchor=anchor)
    fill_tf(tb.text_frame, paras, size, color, label=label or str(paras)[:30], box=(w - margin[0] - margin[1], h - margin[2] - margin[3]))
    return tb

def _noshadow(shape):
    spPr = shape._element.spPr
    if spPr.find(pqn('a:effectLst')) is None: etree.SubElement(spPr, pqn('a:effectLst'))
    st = shape._element.find(pqn('p:style'))
    if st is not None: shape._element.remove(st)

def card(slide, x, y, w, h, paras=None, fill=GRAY, border=LINE, size=15, color=BODY, margin=(0.16, 0.16, 0.1, 0.08),
         anchor='t', radius=0.05, label='', bw=0.75):
    if paras: margin = (margin[0], margin[1] + pick_extra(paras, w, margin), margin[2], margin[3])
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = radius * 1.0 / min(w, h) * 1.0 if min(w, h) > 0 else 0.05
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    if border: sh.line.color.rgb = rgb(border); sh.line.width = Pt(bw)
    else: sh.line.fill.background()
    _noshadow(sh)
    set_frame(sh.text_frame, *margin, anchor=anchor)
    if paras:
        fill_tf(sh.text_frame, paras, size, color, label=label or str(paras)[:30], box=(w - margin[0] - margin[1], h - margin[2] - margin[3]))
    return sh

def rect(slide, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    if line: sh.line.color.rgb = rgb(line)
    else: sh.line.fill.background()
    _noshadow(sh); return sh

def hline(slide, x, y, w, color=LINE, wt=1.0):
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.line.color.rgb = rgb(color); ln.line.width = Pt(wt)
    st = ln._element.find(pqn('p:style'))
    if st is not None: ln._element.remove(st)
    return ln

def place_img(slide, path, x, y, w, h, descr='', align='c'):
    im = Image.open(path); r = im.width / im.height
    pw, ph = (w, w / r) if w / h < r else (h * r, h)
    px = x + (w - pw) / 2 if align == 'c' else x
    pic = slide.shapes.add_picture(path, Inches(px), Inches(y), Inches(pw), Inches(ph))
    if descr: pic._element.nvPicPr.cNvPr.set('descr', descr)
    return pic, (px, y, pw, ph)

# ---------- 表格 ----------
def style_cell(cell, fill=None, bottom=(LINE, 0.75), top=None, ml=0.08, mr=0.08, mt=0.03, mb=0.03, anchor='m'):
    tcPr = cell._tc.get_or_add_tcPr()
    for ch in list(tcPr): tcPr.remove(ch)
    def ln(tag, spec):
        e = etree.SubElement(tcPr, pqn(tag))
        if spec is None:
            e.set('w', '0'); etree.SubElement(e, pqn('a:noFill'))
        else:
            col, wt = spec
            e.set('w', str(int(wt * 12700))); e.set('cap', 'flat'); e.set('cmpd', 'sng'); e.set('algn', 'ctr')
            etree.SubElement(etree.SubElement(e, pqn('a:solidFill')), pqn('a:srgbClr')).set('val', col)
            etree.SubElement(e, pqn('a:prstDash')).set('val', 'solid')
    ln('a:lnL', None); ln('a:lnR', None); ln('a:lnT', top); ln('a:lnB', bottom)
    if fill: etree.SubElement(etree.SubElement(tcPr, pqn('a:solidFill')), pqn('a:srgbClr')).set('val', fill)
    else: etree.SubElement(tcPr, pqn('a:noFill'))
    cell.margin_left, cell.margin_right = Inches(ml), Inches(mr)
    cell.margin_top, cell.margin_bottom = Inches(mt), Inches(mb)
    cell.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE}[anchor]

def add_table(slide, x, y, col_w, row_h, data, size=14, header_size=None, aligns=None, bold_first_col=False,
              header_fill=BLUE, label='table', pad=(0.08, 0.03)):
    nr, nc = len(data), len(col_w)
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(sum(row_h)))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set('firstRow', '1'); tblPr.set('bandRow', '0')
    sid = tblPr.find(pqn('a:tableStyleId'))
    if sid is None: sid = etree.SubElement(tblPr, pqn('a:tableStyleId'))
    sid.text = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'
    for j, w in enumerate(col_w): tbl.columns[j].width = Inches(w)
    for i, h in enumerate(row_h): tbl.rows[i].height = Inches(h)
    hs = header_size or size
    est = 0
    for i in range(nr):
        rowmax = 1
        for j in range(nc):
            c = tbl.cell(i, j); v = data[i][j]
            hdr = (i == 0)
            style_cell(c, fill=header_fill if hdr else None, bottom=(ACC, 1.25) if hdr else (LINE, 0.75), ml=pad[0], mr=pad[0], mt=pad[1], mb=pad[1])
            tf = c.text_frame; tf.word_wrap = True
            al = (aligns[j] if aligns else 'l')
            runs = v if isinstance(v, list) else [(v, {})]
            paras = [{'t': runs, 'size': hs if hdr else size, 'bold': hdr or (bold_first_col and j == 0),
                      'color': TITLE if hdr or (bold_first_col and j == 0) else BODY, 'align': al}]
            fill_tf(tf, paras)
            plain = ''.join(t if isinstance(t, str) else t[0] for t in runs)
            plain = ''.join((r if isinstance(r, str) else r[0]) for r in runs)
            rowmax = max(rowmax, est_lines(plain, hs if hdr else size, col_w[j] - 0.16))
        need = rowmax * (hs if i == 0 else size) * 1.45 / 72 + 2 * pad[1]
        est += max(row_h[i], need)
        if need > row_h[i] + 0.02: WARN.append('%s row %d grows: need %.2f > %.2f' % (label, i, need, row_h[i]))
    return gf, est

# ---------- 投影片骨架 ----------
prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
# 母片與版面配置由 4:3 縮放為 16:9
def _scale_x(shapes):
    for s in shapes:
        try:
            if s.left is not None: s.left = int(s.left * 4 / 3)
            if s.width is not None: s.width = int(s.width * 4 / 3)
        except Exception: pass
_scale_x(prs.slide_master.shapes)
for lo in prs.slide_layouts: _scale_x(lo.shapes)
prs.slide_master.background.fill.solid(); prs.slide_master.background.fill.fore_color.rgb = rgb(BG)
LAYOUT = prs.slide_layouts[5]   # Title Only
SLIDES = []

def new_slide(title, tag=None, number=True, title_w=None):
    s = prs.slides.add_slide(LAYOUT)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    t = s.shapes.title
    tw = title_w or (CW - (2.0 if tag else 0))
    t.left, t.top, t.width, t.height = Inches(MX), Inches(0.3), Inches(tw), Inches(0.72)
    tf = t.text_frame; set_frame(tf, 0.0, 0.0, 0.02, 0.02, anchor='m')
    fill_tf(tf, [{'t': title, 'size': 28, 'bold': True, 'color': TITLE}], label='TITLE ' + title, box=(tw, 0.68))
    hline(s, MX, 1.1, CW)
    if tag:
        card(s, MX + CW - 1.78, 0.45, 1.78, 0.42, [{'t': tag, 'size': 14, 'bold': True, 'color': ACC, 'align': 'c'}],
             fill=BLUE, margin=(0.05, 0.05, 0.02, 0.02), anchor='m', radius=0.2, label='tag')
    SLIDES.append(s)
    if number: slide_number(s, len(SLIDES))
    return s

def slide_number(s, n):
    tb = s.shapes.add_textbox(Inches(11.9), Inches(7.04), Inches(0.83), Inches(0.3))
    set_frame(tb.text_frame, 0, 0, 0, 0, anchor='m')
    p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, pqn('a:fld')); fld.set('id', '{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}'); fld.set('type', 'slidenum')
    rPr = etree.SubElement(fld, pqn('a:rPr')); rPr.set('lang', 'zh-TW'); rPr.set('sz', '1100')
    etree.SubElement(etree.SubElement(rPr, pqn('a:solidFill')), pqn('a:srgbClr')).set('val', TITLE)
    for tag in ('a:latin', 'a:ea', 'a:cs'): etree.SubElement(rPr, pqn(tag)).set('typeface', FONT)
    etree.SubElement(fld, pqn('a:t')).text = str(n)

def bullets(items, size=16, after=7, **kw):
    out = []
    for it in items:
        d = {'t': it, 'size': size, 'bullet': True, 'after': after}; d.update(kw); out.append(d)
    return out

def cap(slide, x, y, w, text, h=0.3, size=12, align='c'):
    return tbox(slide, x, y, w, h, [{'t': text, 'size': size, 'color': MUTED, 'align': align}], label='cap ' + text[:12])
# ======================================================================
# 共用元件（覆寫／擴充）
# ======================================================================
def R(s, **base):
    """簡易標記：_{下標}、^{上標}，回傳 runs。"""
    out = []
    for part in re.split(r'(_\{[^}]*\}|\^\{[^}]*\})', s):
        if not part: continue
        if part.startswith('_{'): d = dict(base); d['sub'] = True; out.append((part[2:-1], d))
        elif part.startswith('^{'): d = dict(base); d['sup'] = True; out.append((part[2:-1], d))
        else: out.append((part, dict(base)))
    return out

def acc(t, **o):
    d = {'color': ACC, 'bold': True}; d.update(o); return (t, d)

def table_bad_breaks(text, size, col_w, pad):
    return has_bad_break(text, size, col_w - 2 * pad - 0.02)

def add_table(slide, x, y, col_w, row_h, data, size=14, header_size=None, aligns=None, bold_first_col=False,
              header_fill=BLUE, label='table', pad=(0.08, 0.03), merges=None):
    """data 儲存格可為 str 或 runs list；merges=[(r0,c0,r1,c1)] 垂直／水平合併（文字置於左上角）。"""
    nr, nc = len(data), len(col_w)
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(sum(row_h)))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set('firstRow', '1'); tblPr.set('bandRow', '0')
    sid = tblPr.find(pqn('a:tableStyleId'))
    if sid is None: sid = etree.SubElement(tblPr, pqn('a:tableStyleId'))
    sid.text = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'
    for j, w in enumerate(col_w): tbl.columns[j].width = Inches(w)
    for i, h in enumerate(row_h): tbl.rows[i].height = Inches(h)
    hs = header_size or size
    est = 0
    for i in range(nr):
        rowmax = 1
        for j in range(nc):
            c = tbl.cell(i, j); v = data[i][j]
            hdr = (i == 0)
            style_cell(c, fill=header_fill if hdr else None, bottom=(ACC, 1.25) if hdr else (LINE, 0.75), ml=pad[0], mr=pad[0], mt=pad[1], mb=pad[1])
            tf = c.text_frame; tf.word_wrap = True
            al = (aligns[j] if aligns else 'l')
            runs = v if isinstance(v, list) else [(v, {})]
            paras = [{'t': runs, 'size': hs if hdr else size, 'bold': hdr or (bold_first_col and j == 0),
                      'color': TITLE if hdr or (bold_first_col and j == 0) else BODY, 'align': al}]
            fill_tf(tf, paras)
            plain = ''.join((r if isinstance(r, str) else r[0]) for r in runs)
            sz = hs if hdr else size
            rowmax = max(rowmax, est_lines(plain, sz, col_w[j] - 2 * pad[0]))
            if plain and table_bad_breaks(plain, sz, col_w[j], pad[0]):
                WARN.append('%s 括號換行 r%d c%d: %s' % (label, i, j, plain))
        need = rowmax * (hs if i == 0 else size) * 1.45 / 72 + 2 * pad[1]
        est += max(row_h[i], need)
        if need > row_h[i] + 0.02: WARN.append('%s row %d grows: need %.2f > %.2f' % (label, i, need, row_h[i]))
    for (r0, c0, r1, c1) in (merges or []):
        tbl.cell(r0, c0).merge(tbl.cell(r1, c1))
    return gf, est

def pic_box(slide, key, x, y, w, h, descr='', align='l', frame=True):
    """原圖等比縮放放入 (w,h) 框；底襯白色矩形，不裁切。回傳 (px,py,pw,ph)。"""
    path = IMG[key]; im = Image.open(path); r = im.width / im.height
    pw, ph = (w, w / r) if w / h < r else (h * r, h)
    px = x + (w - pw) / 2 if align == 'c' else x
    bk = rect(slide, px, y, pw, ph, 'FFFFFF', LINE if frame else None)
    pic = slide.shapes.add_picture(path, Inches(px), Inches(y), Inches(pw), Inches(ph))
    if descr: pic._element.nvPicPr.cNvPr.set('descr', descr)
    return (px, y, pw, ph)

def stat(slide, x, y, w, h, label, value, fill=BLUE, vsize=28):
    card(slide, x, y, w, h, [{'t': label, 'size': 13, 'color': MUTED, 'after': 0},
                             {'t': value, 'size': vsize, 'bold': True, 'color': ACC}],
         fill=fill, margin=(0.14, 0.1, 0.06, 0.04), anchor='m', label='stat ' + label)

def plain_of(it):
    return it if isinstance(it, str) else ''.join(x if isinstance(x, str) else x[0] for x in it)

def fb(items, w, h, sizes=(20, 19, 18, 17, 16, 15, 14, 13), after=8, hdr=0.0, **kw):
    """依可用寬 w、高 h 自動挑選最大字級之條列（w 為文字區內寬）。"""
    sz = fit_size([plain_of(i) for i in items], w, h, sizes, after, hdr_h=hdr)
    return bullets(items, size=sz, after=after, **kw)

def B(items, size=15, after=8, **kw):
    """items: str 或 runs；回傳條列段落。"""
    return bullets(items, size=size, after=after, **kw)

# ======================================================================
# 由報告取得各地點段落（結論 (1)–(5)、附件各地點補充）
# ======================================================================
SITE_NAMES = ['龜山島', '貢寮', '新竹', '南灣', '連江']
def find_para(prefix, start=0):
    for i in range(start, len(PARAS)):
        if PARAS[i].startswith(prefix): return i
    raise SystemExit('PARA NOT FOUND: ' + prefix)

def clean(s):
    """去除圖表／附件編號用語（意思不變），並將「抵達海岸」改為「接觸海岸」。"""
    s0 = s
    s = re.sub(r'由表 3 可知，', '', s)
    s = re.sub(r'由圖 \d\([ab]\) 可見，', '', s)
    s = re.sub(r'如表 4 所示，', '', s)
    s = s.replace('由附件 3 之代表回溯軌跡可見', '代表回溯軌跡可見').replace('附件 3 之代表軌跡', '代表軌跡')
    s = re.sub(r'，如圖 附3-\d+ 所示', '', s)
    s = re.sub(r'，如圖 附3-\d+ 所示，', '，', s)
    s = s.replace('推導如附件 3 所示', '推導見補充資料').replace('由附件 3 之十類與大小潮來源分段機率圖可見', '由十類與大小潮來源分段機率圖可見').replace('如第二節所述', '如結果所述')
    s = s.replace('抵達海岸', '接觸海岸')
    if re.search(r'[圖表]\s*\d|附件|附3|圖\s*附', s): raise SystemExit('REF LEFT: ' + s)
    if s != s0: NOTQ.append('[改寫] ' + s0 + '  =>  ' + s)
    return s

def sents(p):
    return [x for x in p.split('。') if x]

SITE = []
_pc = find_para('（1）龜山島：')
_ap = find_para('三、各研究地點補充結果')
for k, nm in enumerate(SITE_NAMES):
    ci = find_para('（%d）%s：' % (k + 1, nm), _pc)
    cs = sents(PARAS[ci]); cs[0] = re.sub(r'^（\d）[^：]+：', '', cs[0])
    ai = find_para(nm + '之代表回溯軌跡', _ap)
    a1 = sents(PARAS[ai]); a2 = sents(PARAS[ai + 1])
    m = re.match(r'域外來源占 ([\d.]+%)、海岸來源占 ([\d.]+%)', clean(cs[0]))
    d = dict(name=nm, k=k, concl=[clean(x) for x in cs], cross=m.group(1), coast=m.group(2),
             traj=[clean(x) for x in a1], tenp=[clean(x) for x in a2[:1]], tide=clean(a2[1] if len(a2) > 1 else ''),
             tideclose=clean(a2[2]) if len(a2) > 2 else '')
    SITE.append(d)
# 各地點補充結果之 sentence 結構檢核
for S in SITE:
    assert S['tenp'][0].startswith('十類之間'), S['name']
    assert S['tide'].startswith('依大小潮區分'), S['name']
    assert S['tideclose'].startswith('兩組之分段分布相近'), (S['name'], S['tideclose'])

# ======================================================================
# 封面、大綱
# ======================================================================
N_SITE0, N_RES, N_CONC, N_SUP = 18, 16, 48, 50
SITE_P = [(N_SITE0 + 6 * i, N_SITE0 + 6 * i + 5) for i in range(5)]

s = prs.slides.add_slide(LAYOUT)
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
SLIDES.append(s)
t = s.shapes.title
t.left, t.top, t.width, t.height = Inches(0.9), Inches(2.35), Inches(10.6), Inches(1.0)
set_frame(t.text_frame, 0, 0, 0, 0, anchor='m')
fill_tf(t.text_frame, [{'t': Q('Lagrangian 系集逆向溯源分析成果'), 'size': 40, 'bold': True, 'color': TITLE}], label='cover title', box=(10.6, 1.0))
tbox(s, 0.9, 1.8, 6, 0.4, [{'t': Q('3-3-3'), 'size': 18, 'bold': True, 'color': ACC}], margin=(0, 0, 0, 0))
tbox(s, 0.9, 3.5, 10.6, 0.9, [
    {'t': N('推估各研究地點海底廢棄物之來源及其機率'), 'size': 22, 'color': BODY, 'after': 6},
    {'t': N('研究地點：龜山島、貢寮、新竹、南灣、連江'), 'size': 16, 'color': MUTED}], margin=(0, 0, 0, 0))
Q('推估龜山島、貢寮、新竹、南灣及連江五個研究地點海底廢棄物之來源及其機率')
rect(s, 12.2, 1.85, 0.3, 3.8, ACC)

s = new_slide('簡報大綱')
rows = [
    ('01', '資料與方法', '研究地點、情境矩陣、終端沉降速度、受體位置、發現時刻與回溯起始時刻、逆向追蹤方法、溯源停止條件、來源分段方式、發現時刻驗證圖', '第 3–15 頁', None),
    ('02', '結果總覽', '域外來源比例與海岸來源比例、主要海岸來源段及其沿岸地物', '第 %d–%d 頁' % (N_RES, N_RES + 1), None),
    ('03', '各研究地點結果', '來源分段機率（全年、季節、十類、大小潮）、代表虛擬質點水平軌跡總覽、本地點結論',
     '第 %d–%d 頁' % (N_SITE0, N_SITE0 + 29), '龜山島 %d–%d　｜　貢寮 %d–%d　｜　新竹 %d–%d　｜　南灣 %d–%d　｜　連江 %d–%d' % tuple(x for p in SITE_P for x in p)),
    ('04', '綜合結論', '綜合結論與結果之限制', '第 %d–%d 頁' % (N_CONC, N_CONC + 1), None),
    ('05', '補充資料', '十類沉降速度推導重點', '第 %d–%d 頁' % (N_SUP, N_SUP + 2), None),
]
y = 1.4
for num, ttl, desc, pg, extra in rows:
    h = 1.3 if extra else (1.15 if len(desc) > 60 else 0.95)
    tbox(s, MX, y, 1.0, h, [{'t': num, 'size': 32, 'bold': True, 'color': ACC}], anchor='m', margin=(0, 0, 0, 0))
    paras = [{'t': ttl, 'size': 21, 'bold': True, 'color': TITLE, 'after': 2}, {'t': desc, 'size': 14, 'color': BODY, 'after': 2}]
    if extra: paras.append({'t': extra, 'size': 13, 'color': MUTED})
    tbox(s, MX + 1.1, y, 8.9, h, paras, anchor='m', margin=(0, 0, 0, 0), label='outline ' + ttl)
    tbox(s, MX + 10.1, y, 2.03, h, [{'t': pg, 'size': 15, 'color': MUTED, 'align': 'r'}], anchor='m', margin=(0, 0, 0, 0))
    y += h + 0.08
    hline(s, MX, y - 0.04, CW)

# ======================================================================
# 資料與方法
# ======================================================================
# ---- 3 研究地點 ----
s = new_slide('研究地點：四區流場範圍與五個地點')
px, py, pw, ph = pic_box(s, 'F1_0', MX, 1.4, 3.6, 5.5, descr='四區流場範圍與各研究地點抽樣範圍位置')
rx = MX + pw + 0.5; rw = MX + CW - rx
card(s, rx, 1.4, rw, 1.15, [{'t': Q('本計畫以 Lagrangian 系集逆向追蹤推估龜山島、貢寮、新竹、南灣及連江五個研究地點海底廢棄物之來源及其機率'), 'size': 17}],
     fill=BLUE, anchor='m', label='lead sites')
tbox(s, rx, 2.8, rw, 0.4, [{'t': N('圖中字母對應之研究地點'), 'size': 15, 'color': MUTED}], anchor='m', margin=(0, 0, 0, 0))
chips = ['(a) 龜山島', '(b) 貢寮', '(c) 新竹', '(d) 南灣', '(e) 連江']
Q('(a) 龜山島、(b) 貢寮、(c) 新竹、(d) 南灣、(e) 連江')
cwid = (rw - 4 * 0.15) / 5
for i, ch in enumerate(chips):
    card(s, rx + i * (cwid + 0.15), 3.25, cwid, 0.7, [{'t': ch, 'size': 17, 'bold': True, 'color': ACC, 'align': 'c'}], fill=GRAY, anchor='m', margin=(0.05, 0.05, 0.02, 0.02), radius=0.2, label='chip ' + ch)
tbox(s, rx, 4.3, rw, 2.6, B([
    [N('關注區域即四區流場範圍')],
    [Q('各研究地點以錨點為圓心、半徑 10 km 之圓為抽樣範圍')],
    [Q('陸地上灰色虛線為縣市界')],
], size=20, after=14), label='sites bullets')

# ---- 4 情境矩陣 ----
s = new_slide('情境矩陣')
card(s, MX, 1.4, CW, 0.8, [{'t': Q('情境矩陣由終端沉降速度、受體位置、發現時刻三個維度及 10 個系集成員組成'), 'size': 20, 'align': 'c'}], fill=BLUE, anchor='m', label='lead matrix')
cw4 = (CW - 3 * 0.2) / 4
dims = [('終端沉降速度', '10 類', Q('每類設 5 種代表物件並分別代入阻力係數之上、下界，得 10 個終端沉降速度')),
        ('受體位置', '20 個', Q('5 個水平位置 × 每個位置 4 個深度')),
        ('發現時刻', '50 個', Q('由 48 個分層時刻與 2 個事件時刻組成')),
        ('系集成員', '10 個', Q('各系集成員分別抽取，同一發現時刻對應 10 個不同之回溯起始時刻'))]
for i, (nm, big, desc) in enumerate(dims):
    cx = MX + i * (cw4 + 0.2)
    card(s, cx, 2.45, cw4, 2.75, [{'t': nm, 'size': 19, 'bold': True, 'color': TITLE, 'after': 2},
                                {'t': big, 'size': 38, 'bold': True, 'color': ACC, 'after': 8},
                                {'t': desc, 'size': 16.5}], label='dim ' + nm)
card(s, MX, 5.45, CW, 1.35, [
    {'t': Q('各研究地點虛擬質點數為 10 類 × 20 個受體位置 × 50 個發現時刻 × 10 個系集成員，共 100,000 顆，五地合計 500,000 顆'), 'size': 21, 'align': 'c'}],
     fill=GRAY, anchor='m', label='matrix total')

# ---- 5 表1 ----
s = new_slide('十類海底廢棄物之代表形狀與終端沉降速度')
tb1 = _doc.tables[0]
def cell_runs(c):
    out = []
    for p in c.paragraphs:
        for r in p.runs:
            txt = r.text.replace('\n', '')
            if not txt: continue
            out.append((txt, {'sup': True}) if r.font.superscript else (txt, {}))
    return out
data = [[Q('項目（材質）'), Q('代表形狀與狀態'), Q('終端沉降速度範圍') + ' (m/s)']]
for r in tb1.rows[1:]:
    cs_ = r.cells
    runs = cell_runs(cs_[2])
    # 「至」前後保持與報告相同：…10^-4 至 …
    runs = [(('' if i == 0 else '') + t, o) for i, (t, o) in enumerate(runs)]
    runs2 = []
    for t_, o_ in runs:
        if t_ == '至': runs2.append((' 至', o_))
        else: runs2.append((t_, o_))
    data.append([cs_[0].text.strip(), cs_[1].text.strip(), runs2])
add_table(s, MX, 1.4, [4.3, 4.8, 3.033], [0.46] + [0.42] * 10, data, size=14, bold_first_col=True, label='表1')
tbox(s, MX, 6.25, CW, 0.65, [{'t': [Q('模擬時各類之 10 個速度各自直接使用、不取平均'), '；', Q('每個研究地點每類 10,000 顆虛擬質點中，各速度分配 1,000 顆')], 'size': 15}], label='note t1')

# ---- 6 受體位置 ----
s = new_slide('受體位置')
px, py, pw, ph = pic_box(s, 'F2_0', MX, 1.4, 3.7, 4.95, descr='各研究地點受體抽樣範圍與受體水平位置')
cap(s, MX, 1.4 + ph + 0.05, pw, Q('各研究地點受體抽樣範圍（虛線圓）與 5 個受體水平位置（空心圓）'), h=0.5, size=11)
rx = MX + pw + 0.4; rw = MX + CW - rx
tb2 = _doc.tables[1]
t2 = [[Q('研究地點'), Q('錨點經度') + ' (°E)', Q('錨點緯度') + ' (°N)', Q('抽樣半徑') + ' (km)']]
for i, r in enumerate(tb2.rows[1:]):
    c = [x.text.strip() for x in r.cells]
    t2.append([c[0], c[1], c[2], '10' if i == 0 else ''])
add_table(s, rx, 1.4, [rw * 0.25, rw * 0.25, rw * 0.25, rw * 0.25], [0.4] + [0.36] * 5, t2, size=14, aligns=['l', 'c', 'c', 'c'],
          bold_first_col=True, label='表2', merges=[(1, 3, 5, 3)])
tbox(s, rx, 1.4 + 0.4 + 5 * 0.36 + 0.05, rw, 0.32, [{'t': [Q('抽樣範圍為以錨點為圓心之圓，如'), N('左圖'), Q('虛線圓所示')], 'size': 12, 'color': MUTED}], label='note t2')
tbox(s, rx, 4.05, rw, 2.9, B([
    [Q('受體位置為假設廢棄物被發現之地點，共 20 個（5 個水平位置 × 每個位置 4 個深度），除底床外亦可位於水體任意深度')],
    [Q('候選海域再依方位及距錨點之距離分別等分為面積相同之 5 個區間，使每一方位區間與每一距離區間各恰有 1 個受體水平位置，並於所屬區塊（方位區間與距離區間之交會處）內隨機抽取')],
    [Q('每個水平位置於海面至海床之水深內均勻隨機抽出 4 個互不相同之相對深度（0 為海面、1 為海床），實際深度依各發現時刻之潮位與該處地形水深換算'), '；', Q('深度編號僅為抽籤順序，不代表表層、中層或底層')],
], size=14, after=7), label='receptor bullets')

# ---- 7 發現時刻與回溯起始時刻 ----
s = new_slide('發現時刻與回溯起始時刻')
hw = (CW - 0.3) / 2
card(s, MX, 1.4, hw, 4.15, [{'t': '發現時刻', 'size': 18, 'bold': True, 'color': ACC, 'after': 5}] + B([
    [Q('發現時刻為假設廢棄物於受體位置被發現之時刻，共 50 個，由 48 個分層時刻與 2 個事件時刻組成')],
    [Q('48 個分層時刻由「四季 × 大潮／小潮 × 三種潮相」共 24 格組成，每格選 2 個時刻，故每季含 12 個分層時刻（大潮、小潮各 6 個）')],
    [Q('2 個事件時刻分別對應高浪事件與強流事件')],
    [Q('大潮與小潮以各時刻前後共 25 小時（約一個潮日）之潮汐強度（該時段海面高度之均方根）與該季中位數比較，高於者為大潮、低於者為小潮')],
    [Q('三種潮相依潮位資料判定，為潮位上升最快、下降最快及最接近漲退潮交替（近平潮）之時刻')],
], size=15, after=6), fill=GRAY, label='found card')
card(s, MX + hw + 0.3, 1.4, hw, 4.15, [{'t': '回溯起始時刻（逆向追蹤之起點）', 'size': 18, 'bold': True, 'color': ACC, 'after': 5}] + B([
    [acc(Q('發現時刻並非逆向追蹤之起點'), color=TITLE)],
    [Q('因廢棄物實際沉降時間未知，另定義回溯起始時刻（廢棄物沉降抵達受體位置之時刻）作為起點，由所屬發現時刻往前推 0–90 天間均勻隨機抽取，各系集成員分別抽取，同一發現時刻對應 10 個不同之回溯起始時刻')],
    [Q('經 0–90 天抽樣後，各研究地點之 500 個回溯起始時刻仍涵蓋四季與大潮、小潮：各季占 17.6%–31.8%，大潮占 48.6%–55.4%、小潮占 44.6%–51.4%')],
    [Q('季節與大小潮之分組以回溯起始時刻（即假設之沉降時刻）為準，反映廢棄物沉降當時之海況')],
], size=15, after=6), fill=BLUE, label='backstart card')
# 時間軸示意
ay = 6.3
hline(s, 2.6, ay, 8.3, color=ACC, wt=2.0)
for dx in (2.6, 10.9):
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(dx - 0.09), Inches(ay - 0.09), Inches(0.18), Inches(0.18))
    d.fill.solid(); d.fill.fore_color.rgb = rgb(ACC); d.line.fill.background(); _noshadow(d)
tbox(s, 4.6, ay - 0.42, 4.3, 0.34, [{'t': Q('由所屬發現時刻往前推 0–90 天間均勻隨機抽取'), 'size': 12, 'color': MUTED, 'align': 'c'}], anchor='m', margin=(0, 0, 0, 0))
tbox(s, 0.6, ay + 0.14, 4.4, 0.55, [{'t': [acc('回溯起始時刻'), '：', Q('廢棄物沉降抵達受體位置之時刻')], 'size': 12.5, 'align': 'c'}], margin=(0, 0, 0, 0), label='axis L')
tbox(s, 8.7, ay + 0.14, 4.4, 0.55, [{'t': [acc('發現時刻'), '：', Q('假設廢棄物於受體位置被發現之時刻')], 'size': 12.5, 'align': 'c'}], margin=(0, 0, 0, 0), label='axis R')
tbox(s, 0.6, ay - 0.5, 2.6, 0.36, [{'t': Q('逆向追蹤將時間倒轉'), 'size': 12, 'color': MUTED, 'align': 'l'}], anchor='m', margin=(0, 0, 0, 0))

# ---- 8 逆向追蹤方法 ----
s = new_slide('逆向追蹤方法')
cw4 = (CW - 3 * 0.2) / 4
steps = [('起點', [Q('逆向追蹤將時間倒轉，虛擬質點自受體位置於回溯起始時刻')]),
         ('水平與垂直輸送', [Q('依流場資料之三維海流（含垂直流速）及由波浪資料計算之斯托克斯漂流反向移動')]),
         ('垂直沉降', [Q('垂直方向並以所屬類別之終端沉降速度'), Q('反向回溯')]),
         ('數值方法', [Q('積分採四階 Runge-Kutta 法，時間步長依流速與網格大小調整、上限 30 秒'), '；', Q('以 Smagorinsky 方式計算之水平擴散係數及固定之垂直擴散係數加入隨機擴散位移')])]
for i, (nm, runs) in enumerate(steps):
    cx = MX + i * (cw4 + 0.2)
    card(s, cx, 1.6, cw4, 3.3, [{'t': '%d　%s' % (i + 1, nm), 'size': 19, 'bold': True, 'color': ACC, 'after': 8}, {'t': runs, 'size': 16}], label='step ' + nm)
card(s, MX, 5.25, CW, 1.2, [{'t': [Q('推算廢棄物沉降前之移動路徑，直至符合'), Q('停止條件為止')], 'size': 21, 'align': 'c'}], fill=BLUE, anchor='m', label='trace strip')

# ---- 9 溯源停止條件 ----
s = new_slide('溯源停止條件')
card(s, MX, 1.4, CW, 1.4, [{'t': [Q('虛擬質點抵達關注區域'), N('（即四區流場範圍）'), Q('邊界（以下簡稱邊界）或海岸時停止追蹤'), '，', Q('以辨識海廢移入各研究海域之潛在外部來源與邊界通量路徑')], 'size': 21}],
     fill=BLUE, anchor='m', label='stop lead')
cw3 = (CW - 2 * 0.25) / 3
stops = [('抵達邊界', Q('抵達邊界者代表廢棄物由流場範圍外之海域經該邊界段移入，稱為域外來源')),
         ('首次接觸海岸', [N('首次接觸海岸者'), Q('代表廢棄物來自陸地海岸，稱為海岸來源')]),
         ('觸及海床', Q('回溯途中觸及海床者亦停止追蹤並另計為觸底'))]
for i, (nm, txt) in enumerate(stops):
    cx = MX + i * (cw3 + 0.25)
    runs = txt if isinstance(txt, list) else [txt]
    card(s, cx, 3.05, cw3, 2.7, [{'t': nm, 'size': 22, 'bold': True, 'color': ACC, 'after': 8}, {'t': runs, 'size': 21}], label='stop ' + nm)
tbox(s, MX, 6.0, CW, 0.5, [{'t': Q('邊界為開放海域之邊界、不含陸地'), 'size': 18, 'color': MUTED}], margin=(0, 0, 0, 0))

# ---- 10 來源分段與來源比例 ----
s = new_slide('來源分段方式與來源比例')
lw = 6.6
tbox(s, MX, 1.4, lw, 5.4, B([
    [Q('來源以邊界與海岸各分段之機率表示')],
    [acc(Q('邊界每 10 km 為一段，海岸每 5 km 為一段'), color=TITLE), Q('（僅計流場範圍內之海岸）')],
    [Q('虛擬質點依其抵達邊界或首次接觸海岸之位置歸入所在之邊界段或海岸段，觸底者另計，故邊界各段、海岸各段與觸底之機率合計為 100%')],
    [Q('各段機率為占該組全部虛擬質點之比例')],
], size=21, after=20), label='seg bullets')
rx = MX + lw + 0.3; rw = MX + CW - rx
card(s, rx, 1.4, rw, 2.2, [{'t': '域外來源', 'size': 19, 'bold': True, 'color': ACC, 'after': 6},
                           {'t': Q('抵達邊界者代表廢棄物由流場範圍外之海域經該邊界段移入，稱為域外來源'), 'size': 19}], fill=BLUE, label='ext card')
card(s, rx, 3.8, rw, 2.2, [{'t': '海岸來源', 'size': 19, 'bold': True, 'color': ACC, 'after': 6},
                           {'t': [N('首次接觸海岸者'), Q('代表廢棄物來自陸地海岸，稱為海岸來源')], 'size': 19}], fill=GRAY, label='coast card')
tbox(s, rx, 6.15, rw, 0.7, [{'t': Q('其比例分別稱為域外來源比例與海岸來源比例'), 'size': 16, 'color': MUTED}], margin=(0, 0, 0, 0))

# ---- 11 集中來源段／分散來源段 ----
s = new_slide('集中來源段與分散來源段')
hw = (CW - 0.3) / 2
card(s, MX, 1.4, hw, 2.75, [{'t': '集中來源段', 'size': 20, 'bold': True, 'color': ACC, 'after': 6},
     {'t': Q('位置相連之各段，若每段機率換算為每 10 km 長度皆達 1% 以上（海岸段長 5 km，即每段達 0.5% 以上），且合計達 3% 以上，合稱一個集中來源段'), 'size': 21}], fill=BLUE, label='conc card')
card(s, MX + hw + 0.3, 1.4, hw, 2.75, [{'t': '分散來源段', 'size': 20, 'bold': True, 'color': ACC, 'after': 6},
     {'t': Q('邊界上集中來源段以外，機率大於 0 之連續兩段以上、合計達 1% 以上者，合稱一個分散來源段，其各段機率多不足 1%，合計可能高於某些集中來源段'), 'size': 21}], fill=GRAY, label='disp card')
card(s, MX, 4.4, CW, 2.4, [{'t': '圖示方式', 'size': 19, 'bold': True, 'color': ACC, 'after': 6}] + B([
    [Q('以黑色括號標示邊界集中來源段、灰色括號標示邊界分散來源段、引線標示海岸集中來源段，各至多標示機率最大之 3 段')],
    [Q('十類圖僅標示邊界上之集中來源段，季節與大小潮圖不標示分散來源段')],
], size=20, after=12), fill=GRAY, label='legend card')

# ---- 12 分組與差異判定 ----
s = new_slide('分組比較與差異判定')
card(s, MX, 1.4, hw, 2.75, [{'t': '分組方式', 'size': 19, 'bold': True, 'color': ACC, 'after': 6}] + B([
    [Q('依回溯起始時刻所在月份分季節（冬季 12–2 月、春季 3–5 月、夏季 6–8 月、秋季 9–11 月）')],
    [Q('十類圖與大小潮圖則分別依 10 類海底廢棄物與大小潮（依回溯起始時刻之潮位資料劃分）分組')],
], size=18.5, after=10), fill=GRAY, label='group card')
card(s, MX + hw + 0.3, 1.4, hw, 2.75, [{'t': '代表軌跡', 'size': 19, 'bold': True, 'color': ACC, 'after': 6},
     {'t': [N('各研究地點另列'), Q('隨機抽取之 24 條代表虛擬質點水平回溯軌跡（以下稱代表軌跡；線色依起始深度由淺至深）')], 'size': 18.5}], fill=GRAY, label='traj card')
card(s, MX, 4.4, CW, 2.4, [{'t': '差異判定', 'size': 19, 'bold': True, 'color': ACC, 'after': 6}] + B([
    [Q('為判斷各圖中兩組之來源分布是否真有不同，而非僅是抽樣之隨機差異，以總變異距離（各段機率差之絕對值加總的一半，0 為完全相同，1 為完全不同）衡量差異')],
    [Q('以同組虛擬質點依發現時刻隨機分成兩半所得距離之第 99 百分位為抽樣變異上限：未超過者視為分布相近，超過者表示差異大於抽樣變異，稱差異明顯')],
], size=19, after=12), fill=BLUE, label='diff card')

# ---- 13–15 發現時刻驗證圖 ----
VCAP = Q('上為潮汐強度與 50 個發現時刻之位置，中為有效波高，下為流速')
VKEY = ['F3_0', 'A1_0', 'A2_0', 'A3_0', 'A4_0']
for pair in ((0, 1), (2, 3), (4,)):
    names = [SITE_NAMES[i] for i in pair]
    s = new_slide('發現時刻驗證圖｜' + '、'.join(names))
    ih = 4.95; iw = ih * Image.open(IMG[VKEY[0]]).width / Image.open(IMG[VKEY[0]]).height
    if len(pair) == 2:
        gap = 0.7; x0 = MX + (CW - 2 * iw - gap) / 2
        for j, i in enumerate(pair):
            bx = x0 + j * (iw + gap)
            tbox(s, bx, 1.32, iw, 0.38, [{'t': SITE_NAMES[i], 'size': 17, 'bold': True, 'color': ACC, 'align': 'c'}], anchor='m', margin=(0, 0, 0, 0))
            pic_box(s, VKEY[i], bx, 1.75, iw, ih, descr=SITE_NAMES[i] + '發現時刻驗證圖（全年）')
        cap(s, MX, 6.75, CW, VCAP, h=0.26, size=12)
    else:
        bx = MX + 0.3
        tbox(s, bx, 1.32, iw, 0.38, [{'t': SITE_NAMES[4], 'size': 17, 'bold': True, 'color': ACC, 'align': 'c'}], anchor='m', margin=(0, 0, 0, 0))
        pic_box(s, VKEY[4], bx, 1.75, iw, ih, descr='連江發現時刻驗證圖（全年）')
        cap(s, bx - 0.3, 6.75, iw + 0.6, VCAP, h=0.26, size=12)
        rx = bx + iw + 0.5
        card(s, rx, 1.75, MX + CW - rx, 4.95, [{'t': '驗證說明', 'size': 18, 'bold': True, 'color': ACC, 'after': 8}] + B([
            [Q('五個研究地點均依上述同一方法挑選發現時刻（各為 50 個，分層方式與每格時刻數相同）')],
            [Q('48 個分層時刻於四季均勻分布，每季大潮、小潮各 6 個')],
            [Q('2 個事件時刻分別位於有效波高與流速之高值處')],
        ], size=19, after=14), fill=BLUE, label='verify card')
assert len(SLIDES) == N_RES - 1, len(SLIDES)

# ======================================================================
# 結果總覽
# ======================================================================
s = new_slide('域外來源比例與海岸來源比例')
tb3 = _doc.tables[2]
t3 = [[Q('研究地點'), Q('域外來源比例') + ' (%)', Q('海岸來源比例') + ' (%)']]
for r in tb3.rows[1:]:
    t3.append([c.text.strip() for c in r.cells])
tbox(s, MX, 1.3, 7.2, 0.4, [{'t': Q('五個研究地點之域外來源比例與海岸來源比例'), 'size': 15, 'color': MUTED}], anchor='m', margin=(0, 0, 0, 0))
add_table(s, MX, 1.8, [2.4, 2.4, 2.4], [0.6] + [0.8] * 5, t3, size=20, header_size=16, aligns=['l', 'c', 'c'], bold_first_col=True, label='表3', pad=(0.12, 0.04))
rx = MX + 7.5; rw = MX + CW - rx
card(s, rx, 1.8, rw, 4.6, B([
    [Q('虛擬質點於回溯初期即快速停止追蹤，各研究地點半數虛擬質點於回溯 3.3 天內、全部於 33.4 天內停止追蹤')],
    [Q('南灣另有 1 顆（0.001%）觸底')],
    [Q('各研究地點之物件種類、沉降速度與大小潮影響皆不明顯')],
], size=19, after=16), fill=GRAY, label='t3 notes')

s = new_slide('主要海岸來源段及其沿岸地物')
tb4 = _doc.tables[3]
t4 = [[Q('研究地點'), Q('海岸來源段'), Q('機率') + ' (%)', Q('港口、漁港及河口'), Q('遊憩據點')]]
cur = ''
for r in tb4.rows[1:]:
    c = [x.text.strip() for x in r.cells]
    t4.append(c)
# 合併研究地點欄（相同地點之連續列）
merges = []; i0 = 1
for i in range(2, len(t4) + 1):
    if i == len(t4) or t4[i][0] != t4[i0][0]:
        if i - 1 > i0: merges.append((i0, 0, i - 1, 0))
        i0 = i
_orig0 = [r_[0] for r_ in t4]
for i in range(2, len(t4)):
    if _orig0[i] == _orig0[i - 1]: t4[i][0] = ''
cw_ = [0.95, 3.2, 1.0, 3.95, 3.033]
tbox(s, MX, 1.28, CW, 0.36, [{'t': Q('各研究地點之主要海岸來源段及其沿岸人為活動地物'), 'size': 14, 'color': MUTED}], anchor='m', margin=(0, 0, 0, 0))
add_table(s, MX, 1.68, cw_, [0.4] + [0.46] * 3 + [0.58] + [0.46] * 5, t4, size=12.5, header_size=12.5, aligns=['l', 'l', 'c', 'l', 'l'], bold_first_col=True, label='表4', merges=merges)
tbox(s, MX, 6.45, CW, 0.5, [{'t': Q('地物依 OpenStreetMap 之港區、河道與海灘等幾何量測其與海岸來源段之最短距離，0.5 km 以內列入，0.5–2 km 者加註（鄰）。新竹無集中來源段，所列為海岸來源分布之主要沿岸。'), 'size': 11.5, 'color': MUTED}], margin=(0, 0, 0, 0), label='note t4')
assert len(SLIDES) == N_SITE0 - 1, len(SLIDES)

# ======================================================================
# 各研究地點（每地點 6 頁）
# ======================================================================
def split_semi(s): return [x for x in s.split('；') if x]
def drop_cmp(pieces):
    """略去跨地點排序之子句。"""
    return [p for p in pieces if not p.startswith('海岸來源比例最高')]

NOTE_A = [Q('以黑、灰括號分別標示邊界上之集中來源段與分散來源段'), '；', Q('引線標示海岸上之集中來源段，旁註數值為各段機率')]
NOTE_A_LJ = [Q('以黑、灰括號分別標示邊界上之集中來源段與分散來源段'), '；', Q('引線標示海岸上之集中來源段，旁註數值為各段機率，插圖為馬祖列島之放大')]
NOTE_B = [Q('依回溯起始時刻之季節'), '；', Q('僅以黑括號標示邊界上之集中來源段'), '；', Q('引線標示海岸上之集中來源段，旁註數值為各段機率'), '，', Q('格上方為該季之域外來源與海岸來源比例')]
NOTE_C = Q('黑色括號標示邊界上之集中來源段，格上方為該類之域外來源與海岸來源比例，括號旁數值為各段機率')
NOTE_D = Q('黑色括號與引線分別標示邊界上與海岸上之集中來源段，格上方為域外來源與海岸來源比例，旁註數值為各段機率')
NOTE_T = [Q('隨機抽取之 24 條代表虛擬質點水平回溯軌跡'), '，', Q('線色依起始深度由淺至深')]

WA, HA = 8.05, 5.4      # 全年／軌跡圖框
WB, HB = 6.85, 5.4      # 季節圖框
WC, HC = 5.3, 5.5       # 十類圖框
WD, HD = 9.0, 4.0       # 大小潮圖框
IY = 1.4

def site_slides(S):
    k = S['k']; nm = S['name']; tag = '地點 %d／5' % (k + 1)
    Fa, Fb = 'F%d_0' % (4 + k), 'F%d_1' % (4 + k)
    At, Ac, Ad = 'A%d_0' % (5 + 3 * k), 'A%d_0' % (6 + 3 * k), 'A%d_0' % (7 + 3 * k)
    c = S['concl']
    CM = (0.16, 0.16, 0.1, 0.08)

    # ---- 1 全年 ----
    s = new_slide(nm + '｜來源分段機率（全年）', tag)
    px, py, pw, ph = pic_box(s, Fa, MX, IY, WA, HA, descr=nm + '來源分段機率（全年，十類合併）')
    rx = MX + WA + 0.3; rw = MX + CW - rx
    hw2 = (rw - 0.15) / 2
    stat(s, rx, IY, hw2, 1.0, '域外來源', S['cross'], vsize=24)
    stat(s, rx + hw2 + 0.15, IY, hw2, 1.0, '海岸來源', S['coast'], fill=GRAY, vsize=24)
    items = split_semi(c[1]) + [c[2]]
    bh = 2.85
    card(s, rx, IY + 1.15, rw, bh, fb(items, rw - 0.32 - 0.05, bh - 0.2, sizes=(17, 16, 15, 14.5, 14, 13), after=6), fill=GRAY, margin=CM, label=nm + ' a bullets')
    ny = IY + 1.15 + bh + 0.15
    card(s, rx, ny, rw, IY + ph - ny, [{'t': (NOTE_A_LJ if nm == '連江' else NOTE_A), 'size': 12, 'color': MUTED}], fill=BG, border=LINE, margin=CM, label=nm + ' a note')

    # ---- 2 季節 ----
    s = new_slide(nm + '｜來源分段機率（依回溯起始時刻之季節）', tag)
    px, py, pw, ph = pic_box(s, Fb, MX, IY, WB, HB, descr=nm + '來源分段機率（依回溯起始時刻之季節）')
    rx = MX + WB + 0.3; rw = MX + CW - rx
    items = drop_cmp(split_semi(c[3]))
    bh = ph - 1.3
    card(s, rx, IY, rw, bh, fb(items, rw - 0.37, bh - 0.2, sizes=(18, 17, 16, 15, 14.5, 14, 13), after=9), fill=BLUE, margin=CM, label=nm + ' b bullets')
    card(s, rx, IY + bh + 0.15, rw, ph - bh - 0.15, [{'t': NOTE_B, 'size': 12, 'color': MUTED}], fill=BG, border=LINE, margin=CM, label=nm + ' b note')

    # ---- 3 代表軌跡 ----
    s = new_slide(nm + '｜代表虛擬質點水平軌跡總覽', tag)
    px, py, pw, ph = pic_box(s, At, MX, IY, WA, HA, descr=nm + '代表虛擬質點水平軌跡總覽')
    rx = MX + WA + 0.3; rw = MX + CW - rx
    bh = ph - 1.45
    card(s, rx, IY, rw, bh, fb(S['traj'], rw - 0.37, bh - 0.2, sizes=(19, 18, 17, 16, 15, 14), after=12), fill=BLUE, margin=CM, label=nm + ' c bullets')
    card(s, rx, IY + bh + 0.15, rw, ph - bh - 0.15, [{'t': NOTE_T, 'size': 12.5, 'color': MUTED}], fill=BG, border=LINE, margin=CM, label=nm + ' c note')

    # ---- 4 十類 ----
    s = new_slide(nm + '｜來源分段機率（十類）', tag)
    px, py, pw, ph = pic_box(s, Ac, MX, IY, WC, HC, descr=nm + '來源分段機率（十類）')
    rx = MX + WC + 0.3; rw = MX + CW - rx
    tp = S['tenp'][0]
    m = re.match(r'十類之間，域外來源比例介於 ([\d.–]+%)、海岸來源比例介於 ([\d.–]+%)，邊界與海岸分段分布相近，(.*)', tp)
    hw2 = (rw - 0.2) / 2
    stat(s, rx, IY, hw2, 1.35, '十類之域外來源比例', m.group(1), vsize=28)
    stat(s, rx + hw2 + 0.2, IY, hw2, 1.35, '十類之海岸來源比例', m.group(2), fill=GRAY, vsize=28)
    p1 = '十類之間，域外來源比例介於 %s、海岸來源比例介於 %s' % (m.group(1), m.group(2))
    p2 = '邊界與海岸分段分布相近，' + m.group(3)
    bh = 2.55
    card(s, rx, IY + 1.55, rw, bh, fb([p1, p2], rw - 0.37, bh - 0.2, sizes=(20, 19, 18, 17, 16), after=12), fill=GRAY, margin=CM, label=nm + ' d bullets')
    ny = IY + 1.55 + bh + 0.2
    card(s, rx, ny, rw, IY + ph - ny, [{'t': NOTE_C, 'size': 13, 'color': MUTED}], fill=BG, border=LINE, margin=CM, label=nm + ' d note')

    # ---- 5 大小潮 ----
    s = new_slide(nm + '｜來源分段機率（依大小潮）', tag)
    px, py, pw, ph = pic_box(s, Ad, MX, IY, WD, HD, descr=nm + '來源分段機率（依大小潮）')
    rx = MX + WD + 0.25; rw = MX + CW - rx
    card(s, rx, IY, rw, ph, [{'t': S['tideclose'], 'size': 16, 'bold': True, 'color': ACC, 'after': 12},
                             {'t': NOTE_D, 'size': 12, 'color': MUTED}], fill=BLUE, margin=CM, label=nm + ' e side')
    parts = split_semi(S['tide'])
    assert len(parts) == 2, parts
    bodies = []
    for tname, ptxt in zip(('大潮', '小潮'), parts):
        ptxt = re.sub(r'^依大小潮區分，', '', ptxt)
        assert ptxt.startswith(tname), ptxt
        b_ = ptxt[2:]
        if b_.startswith('之'): b_ = b_[1:]
        bodies.append((tname, b_))
    hw = (CW - 0.25) / 2
    cy = IY + ph + 0.2
    for i, (tname, b_) in enumerate(bodies):
        fs = fit_size([b_], hw - 0.95, BOT - cy - 0.55, (18, 17, 16, 15.5, 15, 14.5, 14), 0)
        card(s, MX + i * (hw + 0.25), cy, hw, BOT - cy, [{'t': tname, 'size': 17, 'bold': True, 'color': ACC, 'after': 3}, {'t': b_, 'size': fs}],
             fill=GRAY, margin=(0.16, 0.16, 0.08, 0.05), label=nm + ' e ' + tname)

    # ---- 6 結論 ----
    s = new_slide(nm + '｜本地點結論', tag)
    hwc = (CW - 0.3) / 2
    left = [c[0]] + split_semi(c[1]) + [c[2]]
    right = []
    for ct in (c[3], c[4]): right += drop_cmp(split_semi(ct))
    ch = BOT - 1.4
    szl = fit_size(left, hwc - 0.4, ch - 0.75, (21, 20, 19, 18, 17, 16, 15), 10)
    szr = fit_size(right, hwc - 0.4, ch - 0.75, (21, 20, 19, 18, 17, 16, 15), 10)
    sz = min(szl, szr)
    pl = B(left, size=sz, after=10); pl[0]['bold'] = True; pl[0]['color'] = TITLE
    card(s, MX, 1.4, hwc, ch, [{'t': '全年來源', 'size': 19, 'bold': True, 'color': ACC, 'after': 8}] + pl, fill=GRAY, margin=(0.2, 0.2, 0.15, 0.1), label=nm + ' f left')
    card(s, MX + hwc + 0.3, 1.4, hwc, ch, [{'t': '季節與流場', 'size': 19, 'bold': True, 'color': ACC, 'after': 8}] + B(right, size=sz, after=10), fill=BLUE, margin=(0.2, 0.2, 0.15, 0.1), label=nm + ' f right')

for S in SITE:
    site_slides(S)
assert len(SLIDES) == N_CONC - 1, len(SLIDES)

# ======================================================================
# 綜合結論
# ======================================================================
i6 = find_para('（6）域外來源之主要方向因地而異')
c6 = [clean(x) for x in sents(PARAS[i6])]
c6[0] = re.sub(r'^（6）', '', c6[0])
s = new_slide('綜合結論')
items = [c6[0]]
for piece in split_semi(c6[1]): items.append(piece)
items.append(c6[2])
fs = fit_size(items, CW, BOT - 1.4 - 0.2, (26, 24, 22, 21, 20, 19, 18), 18)
paras = B(items, size=fs, after=18)
paras[-1]['bold'] = True; paras[-1]['color'] = ACC
tbox(s, MX, 1.5, CW, BOT - 1.5, paras, label='綜合結論')

i7 = find_para('（7）結果之限制')
c7 = [clean(x) for x in sents(PARAS[i7])]
c7[0] = re.sub(r'^（7）結果之限制：', '', c7[0])
s = new_slide('結果之限制')
items = []
for t_ in c7: items += split_semi(t_)
fs = fit_size(items, CW, BOT - 1.4 - 0.2, (26, 24, 22, 21, 20, 19, 18), 18)
tbox(s, MX, 1.5, CW, BOT - 1.5, B(items, size=fs, after=18), label='結果之限制')
assert len(SLIDES) == N_SUP - 1, len(SLIDES)

# ======================================================================
# 補充資料：十類沉降速度推導重點
# ======================================================================
def fcard(slide, x, y, w, h, label, formula, size=17, fill=GRAY, note=None):
    paras = [{'t': R(label), 'size': 13, 'color': MUTED, 'after': 2}, {'t': R(formula), 'size': size, 'bold': True, 'color': TITLE}]
    if note: paras.append({'t': note, 'size': 12, 'color': MUTED, 'before': 2})
    card(slide, x, y, w, h, paras, fill=fill, anchor='m', label='formula ' + label)

s = new_slide('沉降速度推導｜計算原理與阻力係數')
lw = 6.55
card(s, MX, 1.4, lw, 1.25, [{'t': Q('物件於水中下沉，速度增加至重力扣除浮力恰等於水之阻力時即不再加速，此時之速度為終端沉降速度'), 'size': 15}], fill=BLUE, anchor='m', label='sup lead')
fcard(s, MX, 2.8, lw, 1.15, N('受力平衡（重力扣除浮力 ＝ 流體阻力）'), 'V g (ρ_{p} − ρ_{w}) = ½ ρ_{w} C_{d} A_{proj} w^{2}')
fcard(s, MX, 4.1, lw, 1.15, N('終端沉降速度（淹沒比重 R ≡ ρ_{p}/ρ_{w} − 1）'), 'w = √( 2 V g R / (C_{d} A_{proj}) )')
fcard(s, MX, 5.4, lw, 1.4, N('阻力係數（隨雷諾數 Re 改變；黏滯項採 Schiller–Naumann 式）'), 'C_{d}(Re) = 24/Re × (1 + 0.15 Re^{0.687}) + C_{d,∞}，Re = w d_{A} / ν', size=15)
rx = MX + lw + 0.3; rw = MX + CW - rx
tbox(s, rx, 1.4, rw, 5.5, B([
    R(N('V 為物件體積（m³，決定重力與浮力）、A_{proj} 為垂直於運動方向之投影面積（m²，決定阻力）、C_{d} 為阻力係數（無因次）、w 為終端沉降速度（m/s）')),
    [Q('Cd 代表物件形狀所造成之阻力大小，並隨雷諾數 Re 改變：物件越小或越慢，Cd 越大'), '；', Q('因 Re 又取決於 w，故 Cd 與 w 須同時求解')],
    R(N('C_{d,∞} 為教科書所列各形狀於 Re>10^{4} 之阻力係數；d_{A} 為與 A_{proj} 等面積之圓直徑（鈍體即 D_{eq}）')),
    R(N('海水密度 ρ_{w} = 1025 kg/m³、運動黏滯係數 ν = 1.05×10^{−6} m²/s、重力加速度 g = 9.81 m/s²')),
], size=16.5, after=12), label='sup1 bullets')

# ---- 補充 2：代表物件與幾何模型 ----
def find_table(first):
    for t_ in _doc.tables:
        if t_.rows and t_.rows[0].cells[0].text.strip() == first: return t_
    raise SystemExit('TABLE NOT FOUND ' + first)
t_cd = find_table('形態')
t_geo = find_table('幾何模型')
s = new_slide('沉降速度推導｜代表物件與幾何模型')
geo = [[Q('幾何模型'), Q('適用情境'), Q('尺寸符號'), R('體積 V'), R('投影面積 A_{proj}'), R('特徵長度 d_{A}')],
       ['薄板', t_geo.rows[1].cells[1].text.strip(), '長 L、寬 W、厚 t', 'L⋅W⋅t', 'L⋅W', R('4A_{proj}/π')],
       ['圓柱', t_geo.rows[2].cells[1].text.strip(), '直徑 d、長 L', R('(π/4)d^{2}L'), 'd⋅L', R('4A_{proj}/π')],
       ['鈍體', t_geo.rows[3].cells[1].text.strip(), R('外包絡體積 V_{env}，等效球徑 D_{eq}=(6V_{env}/π)^{1/3}'), R('V_{env}'), R('(π/4)D_{eq}^{2}'), R('D_{eq}')]]
for r_ in geo[1:]: Q(r_[1]); 
for t_, n_ in ((geo[1][0], 0), (geo[2][0], 1), (geo[3][0], 2)): Q(t_)
add_table(s, MX, 1.4, [1.1, 3.1, 3.55, 1.5, 1.6, 1.283], [0.45, 0.5, 0.5, 0.75], geo, size=13.5, header_size=13.5, bold_first_col=True, label='geo table', aligns=['l', 'l', 'l', 'c', 'c', 'c'])
cd = [[Q('形態'), R('本計畫 C_{d,∞}')]]
for r_ in t_cd.rows[1:]:
    cd.append([r_.cells[0].text.strip(), r_.cells[1].text.strip()])
add_table(s, MX, 3.95, [1.6, 2.0], [0.42, 0.42, 0.42, 0.42], cd, size=14.5, bold_first_col=True, aligns=['l', 'c'], label='cd table')
tbox(s, MX, 5.75, 3.7, 1.1, [{'t': [Q('阻力係數之形狀項以流體力學教科書值為基準設定上、下界'), '；', Q('兩界分別代入，得各代表物件速度之上界與下界')], 'size': 13, 'color': MUTED}], label='cd note')
rx = MX + 4.3; rw = MX + CW - rx
tbox(s, rx, 3.95, rw, 3.0, B([
    [Q('各類海底廢棄物設定 5 種代表物件，共 50 組'), '，', Q('各組分別代入阻力係數上下界，得 100 個解')],
    [Q('瓶、罐等薄壁中空容器不直接假設密度，而由殼體質量與進水體積推算')],
    [Q('以 600 mL 寶特瓶為例，PET 殼重 25 g、材質密度 1380 kg/m³，材料僅佔全瓶體積 2.9%，完全進水後有效密度約 1035 kg/m³')],
    [Q('片狀與柱狀物件自由沉降時傾向以最大投影面積垂直於運動方向'), '，', Q('故薄板取寬面迎流、圓柱取側面迎流')],
], size=16.5, after=12), label='sup2 bullets')

# ---- 補充 3：求解流程與結果 ----
s = new_slide('沉降速度推導｜求解流程與結果')
lw = 6.0
card(s, MX, 1.4, lw, 5.4, [{'t': '求解流程', 'size': 18, 'bold': True, 'color': ACC, 'after': 6},
    {'t': Q('逐一處理 10 類中每一類（c）之 5 種代表物件（j）').replace('（c）', '').replace('（j）', ''), 'size': 15.5, 'bold': True, 'color': TITLE, 'after': 3}] + B([
    [Q('計算 V、Aproj、dA')],
    R(N('計算淹沒比重 R = ρ_{p}/ρ_{w} − 1、N = 2VgR/A_{proj}')),
    [Q('Cd,∞ 分別取下界與上界，各執行一次步驟 2')],
], size=16, after=4) + [
    {'t': N('迭代求終端沉降速度'), 'size': 15.5, 'bold': True, 'color': TITLE, 'before': 8, 'after': 3}] + B([
    R(N('設初值 w ← 1×10^{−4} m/s')),
    R(N('計算 Re = w d_{A}/ν、C_{d}、w* = N/C_{d}')),
    R(N('若 |w* − w| < 10^{−12} w*，輸出 w*；否則令 w ← 0.5w + 0.5w*，重複計算，上限 500 次')),
], size=16, after=4) + [
    {'t': N('用於模擬'), 'size': 15.5, 'bold': True, 'color': TITLE, 'before': 8, 'after': 3}] + B([
    [N('各類 10 個解之最小值與最大值即速度範圍之下界與上界')],
    [Q('各研究地點每類之 10,000 顆虛擬質點，10 個解各分配給其中 1,000 顆')],
], size=16, after=4), fill=GRAY, label='sup3 steps')
rx = MX + lw + 0.3; rw = MX + CW - rx
card(s, rx, 1.4, rw, 3.0, [{'t': '結果', 'size': 18, 'bold': True, 'color': ACC, 'after': 6}] + B([
    [Q('50 組代表物件分別代入阻力係數上、下界所得之 100 個解，依類別取最小值與最大值')],
    [Q('保麗龍、竹木、廢紙、不可回收塑膠、廢漁網漁具與可回收塑膠六類之最大值為最小值的 10 倍以上')],
    [Q('寶特瓶、鋁罐、鐵罐之半壓扁與重壓扁兩種代表物件速度相同'), '，', Q('故這三類各有 8 個相異速度')],
], size=16.5, after=8), fill=BLUE, label='sup3 results')
card(s, rx, 4.55, rw, 2.25, [{'t': '條件', 'size': 18, 'bold': True, 'color': ACC, 'after': 6},
    {'t': Q('靜止、均質之無界水體（不計紊流、密度分層與壁面效應）；單一物件獨立沉降；物件為剛體，形狀與密度不隨時間改變；加速至終端速度之時間與距離忽略不計'), 'size': 16}], fill=GRAY, label='sup3 cond')
assert len(SLIDES) == N_SUP + 2, len(SLIDES)

# ======================================================================
# 儲存
# ======================================================================
cp = prs.core_properties
cp.title = 'Lagrangian 系集逆向溯源分析成果'; cp.author = ''; cp.last_modified_by = ''; cp.subject = ''; cp.comments = ''
tmp = os.path.join(WORK, 'tmp_build.pptx')
prs.save(tmp)
zin = zipfile.ZipFile(tmp)
zout = zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data_ = zin.read(item.filename)
    if re.match(r'ppt/theme/theme\d+\.xml', item.filename):
        x = data_.decode('utf8')
        x = re.sub(r'<a:latin typeface="[^"]*"', '<a:latin typeface="%s"' % FONT, x)
        x = re.sub(r'<a:ea typeface="[^"]*"', '<a:ea typeface="%s"' % FONT, x)
        x = re.sub(r'<a:cs typeface="[^"]*"', '<a:cs typeface="%s"' % FONT, x)
        data_ = x.encode('utf8')
    zout.writestr(item, data_)
zout.close(); zin.close(); os.remove(tmp)
open(os.path.join(WORK, 'notq.txt'), 'w', encoding='utf8').write('\n'.join(NOTQ))
print('slides:', len(SLIDES), 'size MB: %.2f' % (os.path.getsize(OUT) / 1e6))
print('\n'.join(WARN) if WARN else 'no fit warnings')
