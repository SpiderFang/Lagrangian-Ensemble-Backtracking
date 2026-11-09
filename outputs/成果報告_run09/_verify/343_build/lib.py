from lxml import etree
import re, copy
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
ns={'w':W,'m':M}
def q(t): 
    p,n=t.split(':'); return '{%s}%s'%(ns[p],n)
def load(path):
    return etree.parse(path)
def ptext(p):
    # text of paragraph excluding math
    out=[]
    for t in p.iter(q('w:t')):
        out.append(t.text or '')
    return ''.join(out)

def show(el, maxlen=6000):
    s=etree.tostring(el,encoding='unicode')
    s=re.sub(r'\sxmlns:\w+="[^"]+"','',s)
    s=re.sub(r'\sw14:(paraId|textId)="[^"]+"','',s)
    s=re.sub(r'\sw:rsid\w*="[^"]+"','',s)
    return s[:maxlen]
def body_items(t):
    return list(t.getroot().find(q('w:body')))

def lin(p, runsep='|'):
    """paragraph linear text; w:t runs separated by | ; math as ⟦...⟧"""
    out=[]
    for ch in p:
        tag=ch.tag.split('}')[1]
        if tag=='r':
            ts=''.join((t.text or '') for t in ch.iter(q('w:t')))
            va=ch.find('.//'+q('w:vertAlign'))
            if va is not None: ts='{%s:%s}'%(va.get(q('w:val'))[:3],ts)
            out.append(ts)
        elif tag in ('oMath',):
            out.append('⟦'+''.join((t.text or '') for t in ch.iter(q('m:t')))+'⟧')
        elif tag=='oMathPara':
            out.append('⟦¶'+''.join((t.text or '') for t in ch.iter(q('m:t')))+'⟧')
        elif tag in ('pPr',): pass
        else: out.append('<%s>'%tag)
    return runsep.join(out)
