# -*- coding: utf-8 -*-
import copy, re, sys
sys.path.insert(0, '/tmp/w')
from lxml import etree
from lib import q, ns, W, M, ptext, lin, show

XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'


def is_text_run(r):
    if r.tag != q('w:r'):
        return False
    has_t = False
    for ch in r:
        if ch.tag == q('w:rPr') or ch.tag == q('w:lastRenderedPageBreak'):
            continue
        if ch.tag == q('w:t'):
            has_t = True
            continue
        return False
    return has_t


def rtext(r):
    return ''.join(t.text or '' for t in r.findall(q('w:t')))


def set_rtext(r, s):
    ts = r.findall(q('w:t'))
    for t in ts[1:]:
        r.remove(t)
    t = ts[0]
    t.text = s
    if s != s.strip() or '  ' in s:
        t.set(XML_SPACE, 'preserve')
    elif XML_SPACE in t.attrib:
        del t.attrib[XML_SPACE]


def groups(p):
    gs, cur = [], []
    for ch in p:
        if ch.tag == q('w:pPr'):
            continue
        if is_text_run(ch):
            cur.append(ch)
        else:
            if cur:
                gs.append(cur)
            cur = []
    if cur:
        gs.append(cur)
    return gs


def replace_in_p(p, old, new, count=1, check_fmt=True):
    """在段落文字（跨相鄰純文字 run）中取代 old -> new；old 須恰出現 count 次。數學式不動。"""
    hits = []
    for g in groups(p):
        txt = ''.join(rtext(r) for r in g)
        pos = 0
        while True:
            i = txt.find(old, pos)
            if i < 0:
                break
            hits.append((g, i))
            pos = i + len(old)
    if len(hits) != count:
        raise ValueError('replace_in_p: %r found %d times (expected %d) in %r' % (old, len(hits), count, ptext(p)[:60]))
    # 由後往前處理，避免位置位移
    for g, i in reversed(hits):
        starts = []
        acc = 0
        for r in g:
            starts.append(acc)
            acc += len(rtext(r))
        # 起訖 run
        j0 = max(k for k, s in enumerate(starts) if s <= i)
        end = i + len(old)
        j1 = max(k for k, s in enumerate(starts) if s < end) if len(old) else j0
        if check_fmt and j1 > j0:
            f0 = etree.tostring(g[j0].find(q('w:rPr'))) if g[j0].find(q('w:rPr')) is not None else b''
            for k in range(j0 + 1, j1 + 1):
                fk = etree.tostring(g[k].find(q('w:rPr'))) if g[k].find(q('w:rPr')) is not None else b''
                if fk != f0:
                    print('  [warn] 取代跨不同格式 run:', old[:30], file=sys.stderr)
        t0 = rtext(g[j0])
        if j0 == j1:
            set_rtext(g[j0], t0[:i - starts[j0]] + new + t0[end - starts[j0]:])
        else:
            set_rtext(g[j0], t0[:i - starts[j0]] + new)
            for k in range(j0 + 1, j1):
                set_rtext(g[k], '')
            t1 = rtext(g[j1])
            set_rtext(g[j1], t1[end - starts[j1]:])
        # 移除變空的 run
        for k in range(j0, j1 + 1):
            if rtext(g[k]) == '' and g[k].getparent() is not None:
                g[k].getparent().remove(g[k])


def find_p(container_iter, prefix, nth=0):
    c = 0
    for p in container_iter:
        if ptext(p).startswith(prefix):
            if c == nth:
                return p
            c += 1
    raise ValueError('paragraph not found: ' + prefix)


# ---- 建立 run / 數學式 ----
def mk(tag, **attrs):
    el = etree.Element(q(tag))
    for k, v in attrs.items():
        el.set(q('w:' + k) if not k.startswith('m_') else q('m:' + k[2:]), v)
    return el


def rpr_from(sz=None, szcs='28', bold=False, extra=None):
    rp = etree.Element(q('w:rPr'))
    if bold:
        rp.append(mk('w:b')); rp.append(mk('w:bCs'))
    if sz:
        e = etree.SubElement(rp, q('w:sz')); e.set(q('w:val'), str(sz))
    e = etree.SubElement(rp, q('w:szCs')); e.set(q('w:val'), str(szcs))
    return rp


def text_run(s, rpr=None):
    r = etree.Element(q('w:r'))
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = etree.SubElement(r, q('w:t'))
    t.text = s
    if s != s.strip():
        t.set(XML_SPACE, 'preserve')
    return r


class MathFactory:
    """依文件既有 OMML 寫法產生行內數學式。sz=None 表示只設 szCs（正文），sz='18' 表示表格 9 pt。"""
    def __init__(self, sz=None, szcs='28'):
        self.sz, self.szcs = sz, szcs

    def _rpr(self, plain=False):
        rp = etree.Element(q('w:rPr'))
        f = etree.SubElement(rp, q('w:rFonts')); f.set(q('w:ascii'), 'Cambria Math'); f.set(q('w:hAnsi'), 'Cambria Math')
        if self.sz:
            e = etree.SubElement(rp, q('w:sz')); e.set(q('w:val'), self.sz)
        e = etree.SubElement(rp, q('w:szCs')); e.set(q('w:val'), self.szcs)
        return rp

    def r(self, text, plain=False):
        mr = etree.Element(q('m:r'))
        if plain:
            mrp = etree.SubElement(mr, q('m:rPr'))
            sty = etree.SubElement(mrp, q('m:sty')); sty.set(q('m:val'), 'p')
        mr.append(self._rpr())
        t = etree.SubElement(mr, q('m:t')); t.text = text
        if text != text.strip():
            t.set(XML_SPACE, 'preserve')
        return mr

    def _ctrl(self, parent, name):
        pr = etree.SubElement(parent, q('m:' + name))
        c = etree.SubElement(pr, q('m:ctrlPr'))
        rp = etree.SubElement(c, q('w:rPr'))
        f = etree.SubElement(rp, q('w:rFonts')); f.set(q('w:ascii'), 'Cambria Math'); f.set(q('w:hAnsi'), 'Cambria Math')

    def sub(self, base, sub, subplain=False):
        s = etree.Element(q('m:sSub'))
        self._ctrl(s, 'sSubPr')
        e = etree.SubElement(s, q('m:e')); e.append(self.r(base))
        b = etree.SubElement(s, q('m:sub')); b.append(self.r(sub, plain=subplain))
        return s

    def sup(self, base, sup):
        s = etree.Element(q('m:sSup'))
        self._ctrl(s, 'sSupPr')
        e = etree.SubElement(s, q('m:e')); e.append(self.r(base))
        b = etree.SubElement(s, q('m:sup')); b.append(self.r(sup))
        return s

    def sci(self, mant, exp, prefix=''):
        """prefix + mant×10^exp 的 OMML 片段（回傳元素清單）"""
        return [self.r('%s%s×' % (prefix, mant)), self.sup('10', str(exp))]

    def omath(self, *parts):
        om = etree.Element(q('m:oMath'))
        for p in parts:
            if isinstance(p, (list, tuple)):
                for x in p:
                    om.append(x)
            else:
                om.append(p)
        return om


def insert_after(ref, new):
    ref.addnext(new)
    return new


def set_cell_text(tc, text, keep_run_idx=0):
    """儲存格單一段落：沿用第一個純文字 run 的格式，內容改為 text（保留段落屬性）。"""
    p = tc.find(q('w:p'))
    runs = [r for r in p.findall(q('w:r'))]
    if not runs:
        raise ValueError('cell has no run')
    r0 = runs[keep_run_idx]
    for r in runs:
        if r is not r0:
            p.remove(r)
    for m in p.findall(q('m:oMath')) + p.findall(q('m:oMathPara')):
        p.remove(m)
    set_rtext(r0, text)
