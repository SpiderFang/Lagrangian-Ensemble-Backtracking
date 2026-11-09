#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_slides.py - 由成果報告 (docx) 產生 16:9 簡報 (pptx)。
用法：SLIDEWORK=<暫存資料夾> python3 build_slides.py
"""
import os, re, io, math, sys, tempfile, zipfile, shutil
from PIL import Image, ImageChops
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
for base in (os.path.join(HOME, 'mnt', 'Workspace'), os.path.join(HOME, 'Workspace')):
    if os.path.isdir(base):
        WS = base
        break
REPORT_DIR = os.path.join(WS, 'Lagrangian-Ensemble-Backtracking', 'outputs', '成果報告_run09')
DOCX = os.path.join(REPORT_DIR, '成果報告_run09_合併版.docx')
OUT_DIR = os.path.join(REPORT_DIR, '簡報')
OUT = os.path.join(OUT_DIR, '成果報告_run09_簡報.pptx')
WORK = os.environ.get('SLIDEWORK') or os.path.join(tempfile.gettempdir(), 'slidework_build')
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 色彩與字型 ----------
BG, TITLE, BODY, MUTED, ACC = 'F7F5EF', '111827', '374151', '64748B', '155DA8'
RED, GREEN, GRAY, BLUE, LINE = 'B91C1C', '0F766E', 'F3F4F6', 'EAF5FB', 'C8D0DB'
FONT = 'Noto Sans TC'
SW, SH = 13.333, 7.5
MX, CW = 0.6, 12.133          # 左邊界與內容寬
TOP, BOT = 1.32, 6.95         # 內容區上下界

WARN = []

# ---------- 讀報告：全文（供 Q() 逐字核對）與圖片 ----------
_doc = docx.Document(DOCX)
_body = _doc.element.body
FULL = '\n'.join(''.join(t.text or '' for t in p.iter(qn('w:t'))) for p in _body.iter(qn('w:p')))

def Q(s):
    """回傳 s；若 s 不是報告原文的逐字片段就中止（確保內容取自報告）。"""
    if s not in FULL:
        raise SystemExit('NOT IN REPORT: ' + s)
    return s

def extract_images():
    d = os.path.join(WORK, 'img'); os.makedirs(d, exist_ok=True)
    els = list(_body.iterchildren()); rels = _doc.part.rels
    found = {}
    for i, el in enumerate(els):
        if el.tag != qn('w:p'): continue
        for b in el.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
            part = rels[b.get(qn('r:embed'))].target_part
            cap = ''.join(t.text or '' for t in els[i + 1].iter(qn('w:t')))
            m = re.match(r'圖(\d+)', cap)
            n = int(m.group(1))
            ext = os.path.splitext(part.partname)[1]
            fn = os.path.join(d, 'fig%02d%s' % (n, ext))
            if not os.path.exists(fn):
                open(fn, 'wb').write(part.blob)
            found[n] = (fn, cap)
    return found
IMGS = extract_images()

def _bbox(im, thr=12):
    rgb = im.convert('RGBA')
    bg = Image.new('RGBA', rgb.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, rgb).convert('RGB')
    diff = ImageChops.difference(comp, Image.new('RGB', comp.size, (255, 255, 255))).convert('L').point(lambda p: 255 if p > thr else 0)
    return diff.getbbox(), comp

def prep(nums, maxpx=1600, pad=16, trim=True):
    """裁掉空白邊（群組共用同一裁切框）、縮到長邊 maxpx，PNG/JPEG 擇小者。回傳 {num: path}。"""
    if isinstance(nums, int): nums = [nums]
    d = os.path.join(WORK, 'prep'); os.makedirs(d, exist_ok=True)
    comps = {}; box = None
    for n in nums:
        im = Image.open(IMGS[n][0]); bb, comp = _bbox(im); comps[n] = comp
        if box is None: box = list(bb)
        else: box = [min(box[0], bb[0]), min(box[1], bb[1]), max(box[2], bb[2]), max(box[3], bb[3])]
    out = {}
    for n in nums:
        comp = comps[n]
        if trim:
            b = (max(0, box[0] - pad), max(0, box[1] - pad), min(comp.width, box[2] + pad), min(comp.height, box[3] + pad))
            comp = comp.crop(b)
        sc = maxpx / max(comp.size)
        if sc < 1: comp = comp.resize((round(comp.width * sc), round(comp.height * sc)), Image.LANCZOS)
        bj, bp = io.BytesIO(), io.BytesIO()
        comp.save(bj, 'JPEG', quality=88, optimize=True, progressive=True)
        comp.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(bp, 'PNG', optimize=True)
        if bp.tell() <= 150 * 1024 or bp.tell() <= 1.1 * bj.tell():
            fn, data = os.path.join(d, 'fig%02d_%d.png' % (n, maxpx)), bp.getvalue()
        else:
            fn, data = os.path.join(d, 'fig%02d_%d.jpg' % (n, maxpx)), bj.getvalue()
        open(fn, 'wb').write(data); out[n] = fn
    return out

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

def style_run(r, size, color=BODY, bold=False, sup=False):
    f = r.font
    f.size = Pt(size); f.bold = bool(bold); f.color.rgb = rgb(color)
    rPr = r._r.get_or_add_rPr()
    rPr.set('lang', 'zh-TW')
    if sup: rPr.set('baseline', '30000')
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
            style_run(r, o.get('size', sz), o.get('color', col), o.get('bold', bd), o.get('sup', False))
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
# ================= 各地點資料（字串皆以 Q() 逐字核對報告原文） =================
SEAS = ['冬', '春', '夏', '秋']
def srow(prefix, season, pct, seg):
    Q('%s季%s為 %s，集中於%s' % (season, prefix, pct, seg))
    return (season, pct, seg)

SITES = []
def add_site(**kw): SITES.append(kw)

add_site(name='龜山島',
  traj=Q('軌跡向南至東南延伸者居多，其次向北延伸；部分抵達海岸'),
  cross='78.35%', coast='21.65%',
  track=Q('第 2.1 天降至 50% 以下、第 6.0 天降至 10% 以下、第 11.7 天降至 1% 以下，至第 26.0 天降至 0%'),
  tracknote=Q('十類曲線相近，其中不可回收塑膠略慢'),
  areas=('801', '2,277', '4,516'),
  area_note=None,
  depth=Q('停留時數之 62.53% 位於水深淺於 200 m 處，多屬大陸棚與近岸海域'),
  pos=Q('50% 高密度區間位於受體周圍之近岸海域，西側緊鄰臺灣本島東北部海岸，向北延伸至受體北側，向南延伸至流場範圍邊界'),
  bd=Q('穿越主要集中於流場範圍南側邊界約 121.8–122.4°E 段（56.46%），其次為北側邊界約 121.7–122.2°E 段（13.95%）'),
  cs=[Q('臺灣本島占 11.96%，龜山島周圍海岸占 9.57%，其餘為其他島嶼'),
      Q('海岸來源主要集中於龜山島周圍海岸，其次為研究地點西南側臺灣本島海岸約 24.60–24.80°N、121.82–121.88°E 段（6.29%）')],
  sb=[srow('穿越比例', '冬', '91.97%', '南側邊界中段（91.33%）'), srow('穿越比例', '春', '80.51%', '南側邊界中段（75.06%）'),
      srow('穿越比例', '夏', '67.31%', '南側邊界中段（35.19%）'), srow('穿越比例', '秋', '77.30%', '南側邊界中段（36.62%）')],
  sc=[srow('海岸來源比例', '冬', '8.03%', '龜山島周圍海岸（5.54%）'), srow('海岸來源比例', '春', '19.49%', '龜山島周圍海岸（8.99%）'),
      srow('海岸來源比例', '夏', '32.69%', '研究地點西南側臺灣本島海岸（16.18%）'), srow('海岸來源比例', '秋', '22.70%', '龜山島周圍海岸（11.65%）')],
  sdiff=Q('冬、春兩季之分段分佈相近，其餘季節兩兩之間差異明顯'),
  tide=[('大潮', '穿越比例為 77.70%，集中於南側邊界中段（54.40%）', '海岸來源比例為 22.30%，集中於龜山島周圍海岸（8.78%）'),
        ('小潮', '穿越比例為 78.99%，集中於南側邊界中段（57.61%）', '海岸來源比例為 21.01%，集中於龜山島周圍海岸（10.33%）')],
  concl=[Q('全年穿越比例為 78.35%，穿越主要集中於南側邊界約 121.8–122.4°E 段（56.46%），其次為北側邊界約 121.7–122.2°E 段（13.95%）'),
         Q('海岸來源占 21.65%，主要集中於龜山島周圍海岸（9.57%），其次為研究地點西南側臺灣本島海岸（6.29%）'),
         Q('四季穿越皆集中於南側邊界'),
         Q('冬、春兩季之分段分佈相近，其餘季節兩兩之間差異明顯')])

add_site(name='貢寮',
  traj=Q('軌跡多向南延伸並穿越流場範圍南界，其次向北延伸；少數抵達海岸'),
  cross='73.40%', coast='26.60%',
  track=Q('第 1.9 天降至 50% 以下、第 5.4 天降至 10% 以下、第 10.5 天降至 1% 以下，至第 26.7 天降至 0%'),
  tracknote=Q('十類曲線相近，其中不可回收塑膠略慢'),
  areas=('1,327', '3,278', '5,585'),
  area_note=None,
  depth=Q('停留時數之 63.52% 位於水深淺於 200 m 處，多屬大陸棚與近岸海域'),
  pos=Q('50% 高密度區間沿臺灣本島東北部海岸外側呈南北向帶狀，通過受體並向南北兩側延伸'),
  bd=Q('穿越主要集中於流場範圍南側邊界約 121.9–122.5°E 段（43.97%），其次為北側邊界約 121.5–122.2°E 段（20.16%）'),
  cs=[Q('臺灣本島占 24.50%，龜山島周圍海岸占 0.88%，其他島嶼占 1.22%'),
      Q('海岸來源主要集中於研究地點西北側臺灣本島海岸約 25.07–25.15°N、121.81–121.92°E 段（16.90%），其次為研究地點東南側臺灣本島海岸約 24.98–25.05°N、121.93–122.00°E 段（6.10%）')],
  sb=[srow('穿越比例', '冬', '81.59%', '南側邊界中段（80.79%）'), srow('穿越比例', '春', '72.85%', '南側邊界中段（51.98%）'),
      srow('穿越比例', '夏', '62.19%', '北側邊界中段（28.90%）'), srow('穿越比例', '秋', '77.65%', '北側邊界中段（39.57%）')],
  sc=[srow('海岸來源比例', '冬', '18.41%', '研究地點西北側臺灣本島海岸（10.97%）'), srow('海岸來源比例', '春', '27.15%', '研究地點周邊臺灣本島海岸（24.08%）'),
      srow('海岸來源比例', '夏', '37.81%', '研究地點周邊臺灣本島海岸（32.21%）'), srow('海岸來源比例', '秋', '22.35%', '研究地點西北側臺灣本島海岸（15.17%）')],
  sdiff=Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯'),
  tide=[('大潮', '穿越比例為 74.21%，集中於南側邊界中段（43.34%）', '海岸來源比例為 25.79%，集中於研究地點西北側臺灣本島海岸（15.76%）'),
        ('小潮', '穿越比例為 72.61%，集中於南側邊界中段（45.84%）', '海岸來源比例為 27.39%，集中於研究地點西北側臺灣本島海岸（17.55%）')],
  concl=[Q('全年穿越比例為 73.40%，穿越主要集中於南側邊界約 121.9–122.5°E 段（43.97%），其次為北側邊界約 121.5–122.2°E 段（20.16%）'),
         Q('海岸來源占 26.60%，主要集中於研究地點西北側臺灣本島海岸（16.90%），其次為研究地點東南側臺灣本島海岸（6.10%）'),
         Q('冬、春季穿越集中於南側邊界，夏、秋季集中於北側邊界'),
         Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯')])

add_site(name='新竹',
  traj=Q('軌跡向西南至西方與東北方延伸，穿越流場範圍西界者較多；僅極少數抵達海岸'),
  cross='97.77%', coast='2.23%',
  track=Q('第 2.7 天降至 50% 以下、第 6.3 天降至 10% 以下、第 11.8 天降至 1% 以下，至第 18.9 天降至 0%'),
  tracknote=Q('十類曲線相近'),
  areas=('1,745', '3,678', '5,898'),
  area_note=None,
  depth=Q('停留時數之 100.00% 位於水深淺於 200 m 處，皆屬大陸棚與近岸海域'),
  pos=Q('50% 高密度區間沿臺灣本島西部海岸外側呈西南至東北向帶狀，通過受體並向東北與西南兩側延伸'),
  bd=Q('穿越主要集中於流場範圍西南角（南側邊界約 119.7–120.0°E 與西側邊界約 24.3–24.9°N，56.04%），其次為北側邊界約 120.6–121.2°E 段（36.58%）'),
  cs=[Q('新竹之海岸來源（2.23%）皆位於臺灣本島'), Q('各海岸段之機率分散於臺灣本島海岸，無機率集中之連續海岸段')],
  sb=[srow('穿越比例', '冬', '100.00%', '北側邊界東段（95.33%）'), srow('穿越比例', '春', '100.00%', '西南角（82.37%）'),
      srow('穿越比例', '夏', '100.00%', '西南角（97.38%）'), srow('穿越比例', '秋', '92.19%', '西側邊界（45.82%）')],
  sc=[('冬', '—', '無海岸來源'), ('春', '—', '無海岸來源'), ('夏', '—', '無海岸來源'),
      srow('海岸來源比例', '秋', '7.81%', '研究地點東南側臺灣本島海岸（5.79%）')],
  sdiff=Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯'),
  tide=[('大潮', '穿越比例為 95.97%，集中於西南角（57.68%）', '海岸來源比例為 4.03%，分散於各海岸段，無機率集中之連續海岸段'),
        ('小潮', '穿越比例為 100.00%，集中於西南角（55.59%）', '海岸來源比例為 <0.01%')],
  concl=[Q('全年穿越比例為 97.77%，穿越主要集中於西南角（南側邊界約 119.7–120.0°E 與西側邊界約 24.3–24.9°N，56.04%），其次為北側邊界約 120.6–121.2°E 段（36.58%）'),
         Q('海岸來源占 2.23%，皆位於臺灣本島，分散於各海岸段'),
         Q('冬季穿越集中於北側邊界，春、夏季集中於西南角，秋季集中於西側邊界'),
         Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯')])

add_site(name='南灣',
  traj=Q('軌跡多向西北延伸並穿越流場範圍西界，少數向東延伸；部分抵達海岸'),
  cross='68.91%', coast='31.09%',
  track=Q('第 1.9 天降至 50% 以下、第 5.0 天降至 10% 以下、第 11.2 天降至 1% 以下，至第 28.2 天降至 0%'),
  tracknote=Q('十類曲線相近，其中不可回收塑膠略慢'),
  areas=('1,836', '4,123', '7,262'),
  area_note=None,
  depth=Q('停留時數之 23.26% 位於水深淺於 200 m 處，多屬水深 200 m 以深之外海'),
  pos=Q('50% 高密度區間自受體沿恆春半島西側向西北延伸，另有一段沿其東側向東北延伸'),
  bd=Q('穿越主要集中於流場範圍西北角（西側邊界約 21.8–22.4°N 與北側邊界約 120.2–120.4°E，50.53%），其次為南側邊界約 121.1–121.4°E 段（4.29%）'),
  cs=[Q('臺灣本島占 29.94%，小琉球占 0.58%，蘭嶼占 0.54%，其他島嶼占 0.03%'),
      Q('海岸來源主要集中於研究地點周邊恆春半島南端海岸約 21.90–22.03°N、120.69–120.86°E 段（27.23%）')],
  sb=[srow('穿越比例', '冬', '54.69%', '南側邊界東段（19.98%）'), srow('穿越比例', '春', '61.56%', '西北角（52.03%）'),
      srow('穿越比例', '夏', '86.62%', '西側邊界（85.64%）'), srow('穿越比例', '秋', '71.39%', '西北角（56.39%）')],
  sc=[srow('海岸來源比例', '冬', '45.31%', '研究地點東側臺灣本島海岸（29.28%）'), srow('海岸來源比例', '春', '38.44%', '研究地點周邊臺灣本島海岸（34.88%）'),
      srow('海岸來源比例', '夏', '13.38%', '研究地點東側臺灣本島海岸（7.02%）'), srow('海岸來源比例', '秋', '28.61%', '研究地點東側臺灣本島海岸（21.47%）')],
  sdiff=Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯'),
  tide=[('大潮', '穿越比例為 68.31%，集中於西北角（50.27%）', '海岸來源比例為 31.69%，集中於研究地點周邊恆春半島南端海岸（27.91%）'),
        ('小潮', '穿越比例為 69.49%，集中於西北角（50.78%）', '海岸來源比例為 30.51%，集中於研究地點東側臺灣本島海岸（19.05%）')],
  concl=[Q('全年穿越比例為 68.91%，穿越主要集中於西北角（西側邊界約 21.8–22.4°N 與北側邊界約 120.2–120.4°E，50.53%），其次為南側邊界約 121.1–121.4°E 段（4.29%）'),
         Q('海岸來源占 31.09%，主要集中於研究地點周邊恆春半島南端海岸（27.23%）'),
         Q('冬季穿越集中於南側邊界，春、秋季集中於西北角，夏季集中於西側邊界'),
         Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯'),
         Q('另有 1 顆（0.001%）於回溯途中觸底而停止追蹤')])

add_site(name='連江',
  traj=Q('軌跡多向東北延伸並穿越流場範圍北界，另有向西南延伸者；多條抵達福建沿岸與馬祖各島'),
  cross='56.06%', coast='43.94%',
  track=Q('第 3.3 天降至 50% 以下、第 7.8 天降至 10% 以下、第 14.5 天降至 1% 以下，至第 33.4 天降至 0%'),
  tracknote=Q('十類曲線相近'),
  areas=('1,029', '2,318', '4,098'),
  area_note=None,
  depth=Q('停留時數之 100.00% 位於水深淺於 200 m 處，皆屬大陸棚與近岸海域'),
  pos=Q('50% 高密度區間自受體向東北延伸，另有一獨立區塊緊鄰福建沿岸'),
  bd=Q('穿越主要集中於流場範圍北側邊界約 120.2–120.6°E 段（30.48%），其次為南側邊界約 119.6–119.9°E 段（12.55%）與南側邊界約 120.4–120.6°E 段（3.04%）'),
  cs=[Q('福建沿岸（含近岸島嶼）占 29.68%，馬祖各島占 14.26%'),
      Q('海岸來源主要集中於馬祖北竿（7.99%），其次為福建沿岸約 25.94–26.04°N、119.62–119.71°E 段（6.26%）與馬祖南竿（4.56%）')],
  sb=[srow('穿越比例', '冬', '85.01%', '北側邊界東段（83.41%）'), srow('穿越比例', '春', '33.86%', '北側邊界東段（17.91%）'),
      srow('穿越比例', '夏', '49.62%', '南側邊界（45.22%）'), srow('穿越比例', '秋', '65.09%', '北側邊界東段（41.64%）')],
  sc=[srow('海岸來源比例', '冬', '14.99%', '馬祖北竿（8.99%）'), srow('海岸來源比例', '春', '66.14%', '馬祖北竿（10.67%）'),
      srow('海岸來源比例', '夏', '50.38%', '福建沿岸（11.31%）'), srow('海岸來源比例', '秋', '34.91%', '馬祖北竿（7.44%）')],
  sdiff=Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯'),
  tide=[('大潮', '穿越比例為 56.70%，集中於北側邊界東段（29.99%）', '海岸來源比例為 43.30%，集中於馬祖北竿（9.42%）'),
        ('小潮', '穿越比例為 55.41%，集中於北側邊界東段（30.97%）', '海岸來源比例為 44.59%，集中於福建沿岸（7.37%）')],
  concl=[Q('全年穿越比例為 56.06%，穿越主要集中於北側邊界約 120.2–120.6°E 段（30.48%），其次為南側邊界約 119.6–119.9°E 段（12.55%）與南側邊界約 120.4–120.6°E 段（3.04%）'),
         Q('海岸來源占 43.94%，其中福建沿岸（含近岸島嶼）占 29.68%，馬祖各島占 14.26%，主要集中於馬祖北竿（7.99%），其次為福建沿岸（6.26%）與馬祖南竿（4.56%）'),
         Q('冬、春、秋季穿越集中於北側邊界，夏季集中於南側邊界'),
         Q('四季之分段分佈兩兩差異明顯，季節對來源之影響明顯')])

# 大小潮句逐字核對
for s in SITES:
    for tname, a, b in s['tide']:
        Q(tname + a); Q(tname + b)
    Q(s['name'] + '仍在追蹤之比例於回溯' + s['track'])
    Q(s['name'] + '之 50%、75%、90% 高密度區間面積分別為 {}、{} 與 {} km²'.format(*s['areas']))
CLOSE_TIDE = Q('大小潮兩組相近，大小潮對來源之影響不明顯')
CLOSE_ALL = Q('物件種類、沉降速度與大小潮對來源之影響不明顯')
# ================= 前段：封面、大綱、方法 =================
def acc(t, **o): d = {'color': ACC, 'bold': True}; d.update(o); return (t, d)

# ---- 1 封面 ----
s = prs.slides.add_slide(LAYOUT)
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
SLIDES.append(s)
t = s.shapes.title
t.left, t.top, t.width, t.height = Inches(0.9), Inches(2.35), Inches(10.6), Inches(1.0)
set_frame(t.text_frame, 0, 0, 0, 0, anchor='m')
fill_tf(t.text_frame, [{'t': 'Lagrangian系集逆向溯源分析成果', 'size': 40, 'bold': True, 'color': TITLE}], label='cover title', box=(10.6, 1.0))
tbox(s, 0.9, 1.8, 6, 0.4, [{'t': '成果報告', 'size': 18, 'bold': True, 'color': ACC}], margin=(0, 0, 0, 0))
tbox(s, 0.9, 3.5, 10.6, 0.5, [{'t': '含斯托克斯漂流案例', 'size': 24, 'bold': True, 'color': BODY}], margin=(0, 0, 0, 0))
tbox(s, 0.9, 4.15, 10.6, 0.9, [
    {'t': '推估各研究地點海底廢棄物之來源及其機率', 'size': 18, 'color': BODY, 'after': 4},
    {'t': '研究地點：龜山島、貢寮、新竹、南灣、連江', 'size': 16, 'color': MUTED}], margin=(0, 0, 0, 0))
rect(s, 12.2, 1.85, 0.3, 3.8, ACC)

# 頁碼規劃（固定版面）
P_METHOD, P_SITE0, P_CONCL, P_APP = 3, 11, 41, 43
SITE_P = [(P_SITE0 + 6 * i, P_SITE0 + 6 * i + 5) for i in range(5)]

# ---- 2 簡報大綱 ----
s = new_slide('簡報大綱')
rows = [
    ('01', '研究方法與設定', '逆向追蹤概念、十類代表形狀與沉降速度、研究地點與受體、五個地點之到達時間驗證圖', '第 3–10 頁', None),
    ('02', '逐地點成果', '軌跡與追蹤比例、來源足跡、來源分段機率、季節、大小潮、本地點結論',
     '第 11–40 頁', '龜山島 %d–%d　｜　貢寮 %d–%d　｜　新竹 %d–%d　｜　南灣 %d–%d　｜　連江 %d–%d' % tuple(x for p in SITE_P for x in p)),
    ('03', '綜合結論', '各因素對來源之影響、各地點結論重點', '第 41–42 頁', None),
    ('04', '附錄', '十類海底廢棄物終端沉降速度之推導重點', '第 43 頁', None),
]
y = 1.45
for num, ttl, desc, pg, extra in rows:
    h = 1.5 if extra else 1.1
    tbox(s, MX, y, 1.0, h, [{'t': num, 'size': 34, 'bold': True, 'color': ACC}], anchor='m', margin=(0, 0, 0, 0))
    paras = [{'t': ttl, 'size': 22, 'bold': True, 'color': TITLE, 'after': 3}, {'t': desc, 'size': 15, 'color': BODY, 'after': 3}]
    if extra: paras.append({'t': extra, 'size': 14, 'color': MUTED})
    tbox(s, MX + 1.1, y, 8.6, h, paras, anchor='m', margin=(0, 0, 0, 0), label='outline ' + ttl)
    tbox(s, MX + 9.9, y, 2.23, h, [{'t': pg, 'size': 16, 'color': MUTED, 'align': 'r'}], anchor='m', margin=(0, 0, 0, 0))
    y += h + 0.1
    hline(s, MX, y - 0.05, CW)

# ---- 3 逆向追蹤概念與設定重點 ----
s = new_slide('逆向追蹤概念與設定重點')
card(s, MX, TOP, CW, 0.82, [{'t': [acc('Lagrangian系集逆向溯源含斯托克斯漂流案例'), '：由到達（被發現）地點逆向追蹤虛擬質點，推估各研究地點海底廢棄物之來源及其機率'], 'size': 16}],
     fill=BLUE, anchor='m', label='lead')
cw2 = (CW - 0.25) / 2
cards = [
    ('1　沉降動力參數', [['10 類海底廢棄物，各類 10 個沉降速度'], ['每類由 5 種代表物件、各代入阻力係數之上下界求得'], ['各類 10 個速度各分配 1,000 顆虛擬質點']]),
    ('2　到達地點（Receptor locations）', [['20 種到達／被發現地點'], ['5 個水平位置 × 每個位置 4 個深度抽樣'], ['抽樣範圍為以錨點為圓心、半徑 10 km 之圓']]),
    ('3　到達時間（Arrival times）', [['50 種到達時間，涵蓋四季與大／小潮'], ['48 個分層時刻：四季 × 大潮／小潮 × 三種潮相，每格 2 個'], ['2 個事件時刻：局地大浪、局地強流']]),
    ('4　溯源停止邊界（Boundary condition）', [['持續追蹤至離開「關注區域」邊界為止'], ['穿越流場範圍邊界即停止追蹤'], ['抵達海岸（視為找到來源，稱為海岸來源）即停止追蹤']]),
]
for i, (hd, items) in enumerate(cards):
    cx = MX + (i % 2) * (cw2 + 0.25); cy = 2.27 + (i // 2) * 2.05
    paras = [{'t': hd, 'size': 18, 'bold': True, 'color': ACC, 'after': 5}] + bullets([x[0] for x in items], size=16, after=4)
    card(s, cx, cy, cw2, 1.95, paras, fill=GRAY, label='card ' + hd)
card(s, MX, 6.4, CW, 0.55, [{'t': ['另以 10 個系集成員交叉：每個研究地點 ', acc('100,000'), ' 顆、五個研究地點合計 ', acc('500,000'), ' 顆虛擬質點'], 'size': 16, 'align': 'c'}],
     fill=BLUE, anchor='m', label='strip')
for x in ['10 類', '20 種', '50 種', '1,000 顆', '半徑 10 km', '48 個分層時刻', '局地大浪', '局地強流']:
    pass
Q('設定 10 種不同的沉降速度（10 類，各類 10 個速度）'); Q('設定 20 種不同的到達／被發現地點'); Q('設定 50 種不同的到達時間')
Q('每類由 5 種代表物件、各代入阻力係數之上下界，求得 10 個終端沉降速度'); Q('該類 10 個速度各分配給其中 1,000 顆')
Q('另以 10 個系集成員交叉'); Q('每個研究地點 100,000 顆、五個研究地點合計 500,000 顆虛擬質點')

# ---- 4 表1 ----
s = new_slide('十類海底廢棄物之代表形狀與終端沉降速度')
tb1 = _doc.tables[0]
def cell_runs(c):
    out = []
    for p in c.paragraphs:
        for r in p.runs:
            txt = r.text.replace('\n', ' ')
            if not txt: continue
            out.append((txt, {'sup': True}) if r.font.superscript else (txt, {}))
    # 合併「至」前後空白
    return out
data = [['項目', '材質', '代表形狀與狀態', '終端沉降速度範圍 (m/s)']]
for r in tb1.rows[1:]:
    cs_ = r.cells
    data.append([cs_[0].text.strip(), cs_[1].text.strip(), cs_[2].text.strip(), cell_runs(cs_[3])])
gf, th = add_table(s, MX, TOP + 0.05, [1.75, 2.95, 4.6, 2.83], [0.46] + [0.43] * 10, data, size=14, bold_first_col=True, label='表1')
tbox(s, MX, 6.28, CW, 0.65, [{'t': ['各類 10 個速度係以受力平衡求解；', acc('模擬時 10 個速度各自直接使用，不取平均值或對數平均'), '。'], 'size': 15, 'color': BODY}], label='note t1')
Q('模擬時 10 個速度各自直接使用，不取平均值或對數平均')

# ---- 5 研究地點 ----
s = new_slide('研究地點：四區流場範圍與五個地點')
f1 = prep(1, 1500)[1]
pic, (px, py, pw, ph) = place_img(s, f1, MX, TOP, 3.55, 5.0, descr='四區流場範圍與各研究地點抽樣範圍位置', align='l')
rx = MX + pw + 0.45; rw = MX + CW - rx
tb2 = _doc.tables[1]
data = [['研究地點', '經度 (°E)', '緯度 (°N)', '抽樣範圍經度 (°E)', '抽樣範圍緯度 (°N)']]
letters = 'abcde'
for i, r in enumerate(tb2.rows[1:]):
    c = [x.text.strip() for x in r.cells]
    data.append(['(%s) %s' % (letters[i], c[0]), c[1], c[2], c[3], c[4]])
cwid = [1.55, 1.2, 1.2, 1.9, 1.9]; scale = rw / sum(cwid); cwid = [w * scale for w in cwid]
add_table(s, rx, TOP + 0.05, cwid, [0.46] + [0.46] * 5, data, size=14, header_size=13, aligns=['l', 'c', 'c', 'c', 'c'], bold_first_col=True, label='表2')
tbox(s, rx, 4.3, rw, 2.6, bullets([
    ['流場範圍共四區；', acc('五個研究地點'), '依序為龜山島、貢寮、新竹、南灣、連江'],
    ['抽樣範圍：以各研究地點錨點為圓心、', acc('半徑 10 km'), ' 之圓'],
    ['關注區域即四區流場範圍；虛擬質點穿越流場範圍邊界或抵達海岸即停止追蹤']], size=17, after=10), label='sites bullets')
Q('五個研究地點之錨點座標與抽樣範圍見表 2'); Q('抽樣範圍為以各研究地點錨點為圓心、半徑 10 km 之圓'); Q('本計畫之關注區域即圖 1 之四區流場範圍')
cap(s, MX, TOP + ph + 0.03, 3.55, '圖中字母對應右表 (a)–(e)', size=12, align='l')

# ---- 6 受體抽樣 ----
s = new_slide('受體抽樣：5 個水平位置 × 4 個深度')
f2 = prep(2, 1500)[2]
pic, (px, py, pw, ph) = place_img(s, f2, MX, TOP, 4.4, 5.1, descr='各研究地點受體抽樣範圍與水平位置', align='l')
cap(s, MX, TOP + ph + 0.03, 6, '虛線圓為抽樣範圍，空心圓為 5 個受體水平位置', size=12, align='l')
rx = MX + pw + 0.4; rw = MX + CW - rx
tbox(s, rx, TOP, rw, 5.5, bullets([
    ['共 ', acc('20'), ' 個受體 ＝ 5 個水平位置 × 每個位置 4 個深度抽樣；受體可分布於三維水體中任意懸浮深度'],
    ['抽樣範圍與邊界保持緩衝，避免虛擬質點甫釋放即抵達邊界'],
    ['水平位置：將圓內候選海域依方位及距錨點之距離，各等分為面積相同之 5 個區間，每一方位區間與每一距離區間各恰有 1 個水平位置，於所屬區塊內隨機抽取'],
    ['深度：於海面至海床之整段水深內，均勻隨機抽出 4 個互不相同的比例（0 為海面、1 為海床）；實際深度隨各到達時間之潮位而異'],
    ['4 個深度的編號只是抽籤順序，不代表表層、中層或底層']], size=17, after=10), label='receptor bullets')
for q in ['避免虛擬質點甫釋放即抵達邊界', '均勻隨機抽出 4 個互不相同的比例（0 為海面、1 為海床）', '這 4 個深度的編號只是抽籤順序，不代表表層、中層或底層',
          '於所屬區塊（方位區間與距離區間之交會處）內隨機抽取', '該受體節點亦可廣泛分布於三維水體中的任意懸浮深度']:
    Q(q)

# ---- 7 到達時間說明 ----
s = new_slide('到達時間：50 個到達時間之設計')
lw7 = 7.7
tbox(s, MX, TOP, lw7, 5.6, bullets([
    [acc('50'), ' 個到達時間 ＝ 48 個分層時刻 ＋ 2 個事件時刻'],
    ['分層時刻：四季 × 大潮／小潮 × 三種潮相，共 24 格，每格選出 2 個時刻'],
    ['大潮／小潮：以各時刻前後共 25 小時（約一個潮日）之潮汐強度與該季中位數比較，高於者為大潮，低於者為小潮'],
    ['三種潮相：潮位上升最快、潮位下降最快、最接近漲退潮交替（近平潮）'],
    ['事件時刻：局地大浪事件與局地強流事件'],
    ['每個研究地點涵蓋四季，每季 12 個分層時刻，大潮、小潮各 6 個']], size=17, after=10), label='arrival bullets')
for q in ['50 個到達時間由 48 個分層時刻與 2 個事件時刻組成', '每格選出 2 個時刻', '高於中位數者為大潮，低於者為小潮', '2 個事件時刻則分別對應局地大浪事件與局地強流事件',
          '每季 12 個分層時刻，大潮、小潮各 6 個', '潮位上升最快、潮位下降最快，以及最接近漲退潮交替之時刻']:
    Q(q)
card(s, MX + lw7 + 0.3, TOP, CW - lw7 - 0.3, 3.9, [
    {'t': '驗證圖之圖例說明', 'size': 17, 'bold': True, 'color': ACC, 'after': 8},
    {'t': Q('潮汐強度即前述各時刻前後 25 小時之水位起伏大小，以海面高度之均方根計算') + '。', 'size': 15, 'after': 8},
    {'t': Q('季內中位數即大潮與小潮之分界') + '。', 'size': 15, 'after': 8},
    {'t': Q('近平潮即最接近漲退潮交替之時刻') + '。', 'size': 15, 'after': 8},
    {'t': Q('2 個事件時刻另標示於有效波高與流速曲線上') + '。', 'size': 15}], fill=BLUE, label='legend card')

# ---- 8–10 到達時間驗證圖 ----
AT = prep([3, 4, 5, 6, 7], 1600)
AT_SITES = ['龜山島', '貢寮', '新竹', '南灣', '連江']
for pair in ((0, 1), (2, 3), (4,)):
    names = [AT_SITES[i] for i in pair]
    s = new_slide('到達時間驗證圖｜' + '、'.join(names))
    for j, i in enumerate(pair):
        bx = MX + (CW - 5.95) / 2 if len(pair) == 1 else (MX if j == 0 else MX + CW - 5.95)
        place_img(s, AT[3 + i], bx, TOP, 5.95, 5.15, descr=AT_SITES[i] + '到達時間驗證圖')
        cap(s, bx, TOP + 5.2, 5.95, AT_SITES[i] + '到達時間驗證圖', h=0.32, size=14)
assert len(SLIDES) == P_SITE0 - 1, len(SLIDES)
# ================= 逐地點成果（每地點 5 頁） =================
def cap_text(n):
    c = IMGS[n][1]
    c = re.sub(r'^圖\d+\s*', '', c)
    return c.split('；')[0]

def hdr(t, color=ACC, size=14, after=3): return {'t': t, 'size': size, 'bold': True, 'color': color, 'after': after}

def site_slides(k, S):
    name = S['name']; b = 8 + 11 * k; tag = '地點 %d／5' % (k + 1)
    hw = 5.95; lx = MX; rx = MX + CW - hw
    IH = 3.55; CY = TOP + IH + 0.4; CH = BOT - CY
    CM = (0.16, 0.16, 0.07, 0.05)

    # ---- a 軌跡與追蹤比例 ----
    s = new_slide('%s｜水平軌跡與追蹤比例' % name, tag)
    fa = prep(b, 1500)[b]; fb = prep(b + 1, 1500)[b + 1]
    place_img(s, fa, lx, TOP, hw, IH, descr=cap_text(b))
    place_img(s, fb, rx, TOP, hw, IH, descr=cap_text(b + 1))
    cap(s, lx, TOP + IH + 0.03, hw, cap_text(b), h=0.3)
    cap(s, rx, TOP + IH + 0.03, hw, cap_text(b + 1).replace('；小圖為回溯前 20 天之放大', ''), h=0.3)
    card(s, lx, CY, hw, CH, [hdr('軌跡特徵'),
         {'t': '隨機抽取之 24 條代表虛擬質點回溯軌跡', 'size': 13, 'color': MUTED, 'after': 3},
         {'t': S['traj'] + '。', 'size': 15}], margin=CM, label=name + ' a-left')
    body = '全部 100,000 顆虛擬質點合計，仍在追蹤之比例於回溯' + S['track'] + '。' + S['tracknote'] + '。'
    if name == '南灣':
        body = '其餘 1 顆（0.001%）於回溯途中觸底而停止追蹤。' + body
        Q('其餘 1 顆（0.001%）於回溯途中觸底而停止追蹤')
    Q('全部 100,000 顆虛擬質點合計，' + name + '仍在追蹤之比例於回溯')
    card(s, rx, CY, hw, CH, [{'t': [('穿越比例 ', {}), acc(S['cross']), ('　海岸來源比例 ', {}), acc(S['coast'])], 'size': 15, 'bold': True, 'color': TITLE, 'after': 2},
         {'t': body, 'size': 13.5}], fill=BLUE, margin=CM, label=name + ' a-right')

    # ---- b 條件式來源足跡 ----
    s = new_slide('%s｜條件式來源足跡（停留時數密度）' % name, tag)
    ff = prep(b + 2, 1800)[b + 2]
    pic, (px, py, pw, ph) = place_img(s, ff, MX, TOP, 8.2, 5.0, descr=cap_text(b + 2), align='l')
    tbox(s, MX, TOP + ph + 0.04, pw, 0.62, [{'t': '底色為停留時數密度：虛擬質點每在流場範圍內漂流 1 小時，即在其所在之 1 km × 1 km 網格計 1 小時停留時數（抵達海岸後不再計入）；高密度區間為包含該比例停留時數之最小面積區域。', 'size': 12, 'color': MUTED}], label=name + ' b-cap')
    Q('虛擬質點每在流場範圍內漂流 1 小時，即在其所在之 1 km × 1 km 網格計 1 小時停留時數'); Q('虛擬質點抵達海岸後不再計入'); Q('即包含該比例停留時數之最小面積區域')
    cx = px + pw + 0.3; cwid = MX + CW - cx
    a50, a75, a90 = S['areas']
    paras = [hdr('高密度區間面積', size=16, after=4)]
    for lab, v in (('50%', a50), ('75%', a75), ('90%', a90)):
        paras.append({'t': [(lab + '　', {'size': 16, 'bold': True, 'color': MUTED}), (v, {'size': 26, 'bold': True, 'color': ACC}), (' km²', {'size': 16})], 'after': 1})
    card(s, cx, TOP, cwid, 2.3, paras, fill=BLUE, label=name + ' b-areas')
    p2 = []
    if S['area_note']: p2.append({'t': S['area_note'] + '。', 'size': 16, 'after': 8, 'bold': True, 'color': TITLE})
    p2.append({'t': S['depth'] + '。', 'size': 16, 'after': 8})
    p2.append({'t': S['pos'] + '。', 'size': 16})
    card(s, cx, TOP + 2.45, cwid, BOT - TOP - 2.45, p2, label=name + ' b-text')

    # ---- c 邊界穿越與海岸來源（全年） ----
    s = new_slide('%s｜邊界穿越與海岸來源（全年）' % name, tag)
    g = prep([b + 3, b + 4], 1500)
    place_img(s, g[b + 3], lx, TOP, hw, IH, descr=cap_text(b + 3))
    place_img(s, g[b + 4], rx, TOP, hw, IH, descr=cap_text(b + 4))
    cap(s, lx, TOP + IH + 0.03, hw, cap_text(b + 3), h=0.3)
    cap(s, rx, TOP + IH + 0.03, hw, cap_text(b + 4), h=0.3)
    card(s, lx, CY, hw, CH, [{'t': [('邊界穿越來源　穿越比例 ', {}), acc(S['cross'])], 'size': 15, 'bold': True, 'color': TITLE, 'after': 2},
         {'t': S['bd'] + '。', 'size': 14}], margin=CM, label=name + ' c-left')
    cs = S['cs']
    cbody = cs[0] + '。' + cs[1] + '。'
    card(s, rx, CY, hw, CH, [{'t': [('海岸來源　海岸來源比例 ', {}), acc(S['coast'])], 'size': 15, 'bold': True, 'color': TITLE, 'after': 2},
         {'t': cbody, 'size': 13.5}], fill=BLUE, margin=CM, label=name + ' c-right')

    # ---- d 季節 ----
    s = new_slide('%s｜依沉底時刻之季節分析' % name, tag)
    g = prep([b + 7, b + 8], 1500)
    DH = 3.72
    place_img(s, g[b + 7], lx, TOP, hw, DH, descr=cap_text(b + 7))
    place_img(s, g[b + 8], rx, TOP, hw, DH, descr=cap_text(b + 8))
    cap(s, lx, TOP + DH + 0.02, hw, cap_text(b + 7), h=0.27)
    cap(s, rx, TOP + DH + 0.02, hw, cap_text(b + 8), h=0.27)
    ty = TOP + DH + 0.34
    rh = [0.28] + [0.26] * 4
    add_table(s, lx, ty, [0.7, 1.25, 4.0], rh, [['季節', '穿越比例', '最集中邊界段']] + [[a + '季', p, q] for a, p, q in S['sb']],
              size=12.5, header_size=12.5, bold_first_col=True, pad=(0.08, 0.005), label=name + ' d-left')
    add_table(s, rx, ty, [0.7, 1.45, 3.8], rh, [['季節', '海岸來源比例', '最集中海岸段']] + [[a + '季', p, q] for a, p, q in S['sc']],
              size=12.5, header_size=12.5, bold_first_col=True, pad=(0.08, 0.005), label=name + ' d-right')
    tbox(s, lx, ty + sum(rh) + 0.05, 6.4, 0.3, [{'t': S['sdiff'] + '。', 'size': 14, 'bold': True, 'color': ACC}], anchor='m', label=name + ' d-strip')
    tbox(s, rx, ty + sum(rh) + 0.05, hw, 0.3, [{'t': '沉底時刻為廢棄物沉降至海床之時刻，亦即虛擬質點回溯之起點', 'size': 12, 'color': MUTED, 'align': 'r'}], anchor='m', label=name + ' d-note')
    Q('沉底時刻為廢棄物沉降至海床之時刻，亦即虛擬質點回溯之起點')

    # ---- e 大小潮 ----
    s = new_slide('%s｜依大小潮分析' % name, tag)
    g = prep([b + 9, b + 10], 1500)
    TH = 3.3
    _, (ax_, ay_, aw_, ah_) = place_img(s, g[b + 9], lx, TOP, hw, TH, descr=cap_text(b + 9))
    place_img(s, g[b + 10], rx, TOP, hw, TH, descr=cap_text(b + 10))
    cap(s, lx, TOP + ah_ + 0.04, hw, cap_text(b + 9), h=0.28)
    cap(s, rx, TOP + ah_ + 0.04, hw, cap_text(b + 10), h=0.28)
    def parse(sx):
        m = re.match(r'.*?比例為 ([^，]+)(?:，(?:集中於)?(.*))?$', sx)
        return m.group(1), (m.group(2) or '—')
    rows_b = [['大小潮', '穿越比例', '最集中邊界段']]; rows_c = [['大小潮', '海岸來源比例', '最集中海岸段']]
    for tname, a_, b_ in S['tide']:
        r1, s1 = parse(a_); r2, s2 = parse(b_)
        rows_b.append([tname, r1, s1]); rows_c.append([tname, r2, s2])
    ty = TOP + ah_ + 0.5
    rhh = [0.42, 0.62, 0.62]
    add_table(s, lx, ty, [0.95, 1.3, 3.7], rhh, rows_b, size=16, header_size=15, bold_first_col=True, pad=(0.08, 0.03), label=name + ' e-left')
    add_table(s, rx, ty, [0.95, 1.55, 3.45], rhh, rows_c, size=16, header_size=15, bold_first_col=True, pad=(0.08, 0.03), label=name + ' e-right')
    tbox(s, lx, ty + sum(rhh) + 0.3, CW, 0.45, [{'t': CLOSE_TIDE + '。', 'size': 18, 'bold': True, 'color': ACC}], anchor='m', label=name + ' e-strip')

    # ---- f 本地點結論 ----
    s = new_slide('%s｜本地點結論' % name, tag)
    items = list(S['concl']) + [CLOSE_ALL]
    fs = fit_size(items, CW, BOT - TOP - 0.1, (21, 20, 19, 18, 17), 16)
    tbox(s, MX, TOP + 0.05, CW, BOT - TOP - 0.05, bullets(items, size=fs, after=16), label=name + ' f-concl')

for k, S in enumerate(SITES):
    site_slides(k, S)
assert len(SLIDES) == P_CONCL - 1, len(SLIDES)

# ================= 綜合結論 =================
s = new_slide('綜合結論｜各因素對來源之影響')
cw2 = (CW - 0.25) / 2
card(s, MX, TOP, cw2, 5.0, [{'t': '物件種類、沉降速度與大小潮', 'size': 22, 'bold': True, 'color': TITLE, 'after': 2},
     {'t': '對來源之影響不明顯', 'size': 22, 'bold': True, 'color': MUTED, 'after': 14}] +
     bullets(['十類之間，分段分佈（邊界與海岸合計）相近', '同類十個沉降速度之間，分段分佈相近', '大潮與小潮之間，分段分佈相近'], size=20, after=14) +
     [{'t': '各地點皆如此', 'size': 16, 'color': MUTED, 'before': 8}], label='concl L')
Q('各研究地點十類之間、同類十個沉降速度之間與大小潮之間，分段分佈（邊界與海岸合計）相近')
sd = [{'t': '季節', 'size': 22, 'bold': True, 'color': TITLE, 'after': 2}, {'t': '對來源之影響明顯', 'size': 22, 'bold': True, 'color': ACC, 'after': 14}]
for S in SITES:
    sd.append({'t': S['name'], 'size': 16, 'bold': True, 'color': TITLE, 'after': 1})
    sd.append({'t': S['sdiff'], 'size': 16, 'after': 9})
card(s, MX + cw2 + 0.25, TOP, cw2, 5.0, sd, fill=BLUE, label='concl R')

s = new_slide('綜合結論｜各地點結論重點')
data = [['研究地點', '全年穿越比例', '穿越主要集中處', '全年海岸來源比例', '海岸來源主要集中處'],
        ['龜山島', '78.35%', '南側邊界約 121.8–122.4°E 段（56.46%）', '21.65%', '龜山島周圍海岸（9.57%）'],
        ['貢寮', '73.40%', '南側邊界約 121.9–122.5°E 段（43.97%）', '26.60%', '研究地點西北側臺灣本島海岸（16.90%）'],
        ['新竹', '97.77%', '西南角（56.04%）', '2.23%', '皆位於臺灣本島，分散於各海岸段'],
        ['南灣', '68.91%', '西北角（50.53%）', '31.09%', '研究地點周邊恆春半島南端海岸（27.23%）'],
        ['連江', '56.06%', '北側邊界約 120.2–120.6°E 段（30.48%）', '43.94%', '馬祖北竿（7.99%）']]
add_table(s, MX, TOP + 0.1, [1.2, 1.45, 4.1, 1.6, 3.78], [0.55] + [0.9] * 5, data, size=15, header_size=14, bold_first_col=True, aligns=['l', 'c', 'l', 'c', 'l'], label='結論表')

# ================= 附錄 =================
s = new_slide('附錄｜十類沉降速度之推導重點')
lw = 6.35
tbox(s, MX, TOP, lw, 5.6, bullets([
    ['依受力平衡，由 5 種代表物件之體積、投影面積、密度與阻力係數計算終端沉降速度'],
    ['阻力係數之形狀項取流體力學教科書之標準值上下界；各類設定 5 種代表物件，共 50 組，每類得 10 個解'],
    ['各類 10 個解之最小值與最大值即為速度範圍之下界與上界'],
    ['條件：靜止、均質之無界水體；單一物件獨立沉降；物件為剛體，加速至終端速度之時間與距離忽略不計'],
    ['保麗龍、竹木、廢紙、不可回收塑膠、廢漁網漁具與可回收塑膠六類之最大值為最小值的 10 倍以上'],
    ['寶特瓶、鋁罐、鐵罐之半壓扁與重壓扁速度相同，這三類各有 8 個相異速度']], size=16, after=10), label='appendix bullets')
for q in ['依受力平衡，由 5 種代表物件之體積、投影面積、密度與阻力係數計算終端沉降速度', '阻力係數之形狀項取流體力學教科書之標準值', '各類海底廢棄物設定 5 種代表物件，共 50 組',
          '各類 10 個解之最小值與最大值即第一節之下界與上界', '靜止、均質之無界水體（不計紊流、密度分層與壁面效應）；單一物件獨立沉降；物件為剛體',
          '加速至終端速度之時間與距離忽略不計', '保麗龍、竹木、廢紙、不可回收塑膠、廢漁網漁具與可回收塑膠六類之最大值為最小值的 10 倍以上',
          '寶特瓶、鋁罐、鐵罐之半壓扁與重壓扁兩種代表物件速度相同', '這三類各有 8 個相異速度']:
    Q(q)
ta1 = _doc.tables[2]
data = [['類別', '下界 (m/s)', '上界 (m/s)']]
for r in ta1.rows[1:]:
    cs_ = r.cells
    data.append([cs_[0].text.strip(), cell_runs(cs_[1]), cell_runs(cs_[2])])
rx = MX + lw + 0.3; rw = MX + CW - rx
cwid = [rw * 0.50, rw * 0.25, rw * 0.25]
add_table(s, rx, TOP + 0.05, cwid, [0.46] + [0.43] * 10, data, size=13, header_size=13, aligns=['l', 'c', 'c'], bold_first_col=True, label='表A1')
cap(s, rx, TOP + 5.0, rw, '表 A1　十類海底廢棄物終端沉降速度之下界與上界', size=12, align='l', h=0.3)

# ================= 儲存與後處理 =================
cp = prs.core_properties
cp.title = 'Lagrangian系集逆向溯源分析成果'; cp.author = ''; cp.last_modified_by = ''; cp.subject = '含斯托克斯漂流案例'; cp.comments = ''
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
print('slides:', len(SLIDES), 'size MB: %.2f' % (os.path.getsize(OUT) / 1e6))
print('\n'.join(WARN) if WARN else 'no fit warnings')
