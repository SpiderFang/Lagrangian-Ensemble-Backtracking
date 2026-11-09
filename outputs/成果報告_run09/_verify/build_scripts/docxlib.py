import re, os, subprocess, tempfile, shutil, zipfile, hashlib, csv
from lxml import etree
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
WNS='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
MNS='http://schemas.openxmlformats.org/officeDocument/2006/math'
NS={'w':WNS,'m':MNS}

def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

# ------------------------------------------------------------ parsing
CAP_RE=re.compile(r'^(\*\*)?(表|圖)\d+ ')
def parse_md(path):
    blocks=[]; lines=open(path,encoding='utf-8').read().split('\n'); i=0
    while i<len(lines):
        l=lines[i]
        if not l.strip(): i+=1; continue
        if l.startswith('#- '): blocks.append(('hsub',l[3:].strip())); i+=1; continue
        if re.match(r'^\d+、',l): m=re.match(r'^(\d+)、(.*)',l); blocks.append(('ni',m.group(1),m.group(2).strip())); i+=1; continue
        if l.startswith('#'):
            m=re.match(r'(#+)\s+(.*)',l); blocks.append(('h',len(m.group(1)),m.group(2))); i+=1; continue
        if l.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                if not re.match(r'^\|[\s\-:|]+\|$',lines[i]):
                    cells=[c.strip() for c in re.split(r'(?<!\\)\|',lines[i].strip())[1:-1]]
                    rows.append(cells)
                i+=1
            blocks.append(('tbl',rows)); continue
        if l.startswith('$$'):
            body=l.strip()[2:-2].strip()
            m=re.search(r'\\qquad\s*\((\d+)\)\s*$',body)
            num=None
            if m: num=m.group(1); body=body[:m.start()].strip()
            blocks.append(('eq',body,num)); i+=1; continue
        m=re.match(r'!\[\]\((.*?)\)\{width=([\d.]+)in\}',l)
        if m: blocks.append(('img',m.group(1),float(m.group(2)))); i+=1; continue
        if CAP_RE.match(l):
            t=l.strip().strip('*')
            blocks.append(('cap_t' if t.startswith('表') else 'cap_f',t)); i+=1; continue
        if l.startswith('- '): blocks.append(('li',l[2:].strip())); i+=1; continue
        if l.startswith('　　'): blocks.append(('p',l)); i+=1; continue
        blocks.append(('note',l)); i+=1
    return blocks

def split_inline(text):
    parts=[]; pos=0
    for m in re.finditer(r'\$([^$]+)\$',text):
        if m.start()>pos: parts.append(('t',text[pos:m.start()]))
        parts.append(('m',m.group(1))); pos=m.end()
    if pos<len(text): parts.append(('t',text[pos:]))
    return parts

def fixlatex(s): return re.sub(r'(?<=\d),(?=\d{3})','{,}',s)

def latex_plain(s):
    s=re.sub(r'\\(?:mathrm|text|mathbf|boldsymbol|mathit)\{([^{}]*)\}',r'\1',s)
    for a,b in [(r'\cdot','·'),(r'\times','×'),(r'\,',''),(r'\;',' '),(r'\ge','≥'),(r'\le','≤'),(r'\approx','≈'),(r'\circ','°'),(r'\pm','±'),(r'\Delta','Δ'),(r'\theta','θ'),(r'\rho','ρ'),(r'\mu','μ'),(r'\eta','η'),(r'\sigma','σ'),(r'\omega','ω'),(r'\phi','φ'),(r'\lambda','λ'),(r'\pi','π'),(r'\left',''),(r'\right','')]:
        s=s.replace(a,b)
    s=re.sub(r'_\{([^{}]*)\}',r'_\1',s); s=re.sub(r'\^\{([^{}]*)\}',r'^\1',s); s=s.replace('{','').replace('}','').replace('\\','')
    return s
def plain_cell(c):
    return re.sub(r'\$([^$]+)\$',lambda m:latex_plain(m.group(1)),c)

# ------------------------------------------------------------ math
class Math:
    def __init__(self): self.frag={}  # (latex,display)->xml
    def collect(self,blocks):
        need=getattr(self,'need',[])
        def add(l,d):
            k=(fixlatex(l),d)
            if k not in self.frag and k not in need: need.append(k)
        for b in blocks:
            if b[0]=='eq': add(b[1],True)
            elif b[0] in('p','li','note','cap_t','cap_f','h'):
                for t,v in split_inline(b[-1] if b[0]!='h' else b[2]):
                    if t=='m': add(v,False)
            elif b[0]=='tbl':
                for r in b[1]:
                    for c in r:
                        for t,v in split_inline(c):
                            if t=='m': add(v,False)
        self.need=need
    def convert(self):
        need=getattr(self,'need',[])
        if not need: return
        md='\n\n'.join(('$$'+l+'$$') if d else ('$'+l+'$') for l,d in need)+'\n'
        tmp=tempfile.mkdtemp()
        open(tmp+'/m.md','w',encoding='utf8').write(md)
        subprocess.check_call(['pandoc',tmp+'/m.md','-f','markdown+tex_math_dollars','-o',tmp+'/m.docx'])
        z=zipfile.ZipFile(tmp+'/m.docx'); root=etree.fromstring(z.read('word/document.xml'))
        paras=root.find('w:body',NS).findall('w:p',NS)
        assert len(paras)==len(need),(len(paras),len(need))
        for (l,d),p in zip(need,paras):
            ms=p.findall('.//m:oMath',NS)
            assert len(ms)==1,(l,len(ms))
            x=etree.tostring(ms[0],encoding='unicode')
            x=re.sub(r' xmlns:\w+="[^"]+"','',x)
            self.frag[(l,d)]=x
        shutil.rmtree(tmp)
    def get(self,latex,display,sz):
        x=self.frag[(fixlatex(latex),display)]
        rp=f'<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>'
        x=re.sub(r'(<m:r>(?:<m:rPr>.*?</m:rPr>)?)(<m:t)',lambda m:m.group(1)+rp+m.group(2),x)
        return x

# ------------------------------------------------------------ profile
class Profile:
    TNR='<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="標楷體" w:cs="Times New Roman"/>'
    def __init__(self,name,sz,total,eq_cols,line_body,jc_body='both'):
        self.name=name; self.sz=sz; self.total=total; self.eq_cols=eq_cols; self.line_body=line_body; self.jc_body=jc_body

def rpr(sz,bold=False,italic=False):
    return f'<w:rPr>{Profile.TNR}<w:b w:val="{1 if bold else 0}"/><w:bCs w:val="{1 if bold else 0}"/><w:i w:val="{1 if italic else 0}"/><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>'

def runs(text,sz,math,bold=False,italic=False):
    out=''
    for t,v in split_inline(text):
        if t=='t':
            if v: out+=f'<w:r>{rpr(sz,bold,italic)}<w:t xml:space="preserve">{esc(v)}</w:t></w:r>'
        else: out+=math.get(v,False,sz)
    return out

def cjk_w(s):
    return sum(2 if ord(c)>0x2e80 else 1 for c in plain_cell(s))

NUMRE=re.compile(r'^[\s\d.,%−\-+–（）()~至／/:×顆\w]*$')
def is_numeric(c):
    c2=plain_cell(c)
    return bool(re.search(r'\d',c2)) and cjk_w(c2)<=14 and not re.search(r'[一-鿿]{3,}',c2)

def table_xml(rows,prof,math,tsz=20,center_all=False):
    ncol=len(rows[0]); total=prof.total
    need=[]
    for j in range(ncol):
        m=max((cjk_w(r[j])*(0.5 if ri==0 else 1) if j<len(r) else 0) for ri,r in enumerate(rows))
        need.append(int(min(m,18)*108+280))
    sn=sum(need)
    if sn<=total:
        extra=total-sn; grid=[int(n+extra*n/sn) for n in need]
    else:
        grid=[int(total*n/sn) for n in need]
    grid[-1]+=total-sum(grid)
    b8='<w:{0} w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
    x='<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'%total
    x+=''.join(b8.format(k) for k in('top','left','bottom','right'))
    x+='<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/></w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="50" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="50" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="0420" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>'
    x+=''.join(f'<w:gridCol w:w="{g}"/>' for g in grid)+'</w:tblGrid>'
    keep = len(rows)<=14
    for ri,r in enumerate(rows):
        hdr=ri==0
        x+='<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if hdr else '')+'</w:trPr>'
        for j in range(ncol):
            c=r[j] if j<len(r) else ''
            jc='center' if (hdr or center_all or is_numeric(c)) else 'left'
            tc=f'<w:tcW w:w="{grid[j]}" w:type="dxa"/>'
            if hdr: tc+='<w:tcBorders><w:bottom w:val="single" w:sz="8" w:space="0" w:color="808080"/></w:tcBorders><w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>'
            tc+='<w:vAlign w:val="center"/>'
            ppr=f'<w:pPr>{"<w:keepNext/>" if (keep and ri<len(rows)-1) else ""}<w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/><w:jc w:val="{jc}"/></w:pPr>'
            x+=f'<w:tc><w:tcPr>{tc}</w:tcPr><w:p>{ppr}{runs(c,tsz,math,bold=hdr)}</w:p></w:tc>'
        x+='</w:tr>'
    return x+'</w:tbl>'

def spacer(): return '<w:p><w:pPr><w:spacing w:before="0" w:after="40" w:line="144" w:lineRule="auto"/></w:pPr></w:p>'

def eq_xml(latex,num,prof,math):
    c=prof.eq_cols; sz=prof.sz
    x='<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/><w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLook w:val="04A0"/></w:tblPr>'%sum(c)
    x+='<w:tblGrid>'+''.join(f'<w:gridCol w:w="{g}"/>' for g in c)+'</w:tblGrid><w:tr><w:trPr><w:cantSplit/></w:trPr>'
    x+=f'<w:tc><w:tcPr><w:tcW w:w="{c[0]}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:p></w:tc>'
    mx=math.get(latex,True,sz)
    x+=f'<w:tc><w:tcPr><w:tcW w:w="{c[1]}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>{mx}</m:oMathPara></w:p></w:tc>'
    n=f'({num})' if num else ''
    x+=f'<w:tc><w:tcPr><w:tcW w:w="{c[2]}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="right"/></w:pPr><w:r>{rpr(sz)}<w:t>{n}</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
    return x

# jpeg cache
def jpeg_for(png,width_in,cache=os.path.expanduser('~/w/jpg')):
    os.makedirs(cache,exist_ok=True)
    st=os.stat(png); key=hashlib.md5((png+str(width_in)+str(st.st_size)+str(int(st.st_mtime))).encode()).hexdigest()[:12]
    out=f'{cache}/{key}.jpg'
    if not os.path.exists(out):
        im=Image.open(png)
        if im.mode in('RGBA','LA','P'):
            im=im.convert('RGBA'); bg=Image.new('RGB',im.size,'white'); bg.paste(im,mask=im.split()[3]); im=bg
        else: im=im.convert('RGB')
        target=int(width_in*380)
        if im.width>target: im=im.resize((target,int(im.height*target/im.width)),Image.LANCZOS)
        im.save(out,'JPEG',quality=84,optimize=True)
    return out

def drawing_xml(rid,cx,cy,did,name):
    return (f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
      f'<wp:docPr id="{did}" name="{esc(name)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
      f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{did}" name="{esc(name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
      f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')

# ------------------------------------------------------------ renderer
def render(blocks,prof,math,figroot,rid_prefix='rIdFig',did0=5000,drop_notes=False):
    out=[]; imgs=[]; prev=None
    sz=prof.sz
    def body_ppr(before=None):
        b=before if before is not None else (120 if prof.name!='res' else 0)
        if prof.name=='impl': return f'<w:pPr><w:spacing w:before="{b}" w:after="120" w:line="360" w:lineRule="auto"/></w:pPr>'
        if prof.name=='res': return f'<w:pPr><w:pStyle w:val="a0"/><w:spacing w:before="{b}" w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
        return f'<w:pPr><w:spacing w:before="{max(b,120)}" w:after="120" w:line="440" w:lineRule="atLeast"/><w:jc w:val="both"/></w:pPr>'
    for b in blocks:
        k=b[0]
        if k=='h':
            lvl,text=b[1],b[2]
            if prof.name=='res':
                st={1:'1',2:'2',3:'3',4:'4'}[lvl]
                out.append(f'<w:p><w:pPr><w:pStyle w:val="{st}"/><w:keepNext/></w:pPr>{runs(text,28,math,bold=True)}</w:p>')
            elif prof.name=='impl':
                cfg={2:(0,160,32),3:(160,160,28),4:(0,80,25)}[lvl]
                out.append(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="{cfg[0]}" w:after="{cfg[1]}" w:line="360" w:lineRule="auto"/></w:pPr>{runs(text,cfg[2],math,bold=True)}</w:p>')
            else:
                # main profile: lvl1/2 plain, lvl3/4 bold
                bold = lvl>=3 if text.startswith('四、') or text.startswith('（一）') or text.startswith('（二）') or text.startswith('（三）') else lvl>=3
                out.append(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120" w:line="440" w:lineRule="atLeast"/><w:jc w:val="both"/></w:pPr>{runs(text,sz,math,bold=bold)}</w:p>')
        elif k=='p':
            before=None
            if prev in('tbl','eq'): before=160 if prof.name=='impl' else 120
            out.append(f'<w:p>{body_ppr(before)}{runs(b[1],sz,math)}</w:p>')
        elif k=='li':
            _ind='<w:ind w:left="480" w:hanging="480"/><w:jc'
            _pp=body_ppr(0).replace('<w:jc',_ind)
            out.append('<w:p>'+_pp+runs('•　'+b[1],sz,math)+'</w:p>')
        elif k=='hsub':
            out.append(f'<w:p><w:pPr><w:pStyle w:val="-"/><w:keepLines/><w:rPr><w:rFonts w:hint="eastAsia"/></w:rPr></w:pPr><w:r><w:t>{esc(b[1])}</w:t></w:r></w:p>')
        elif k=='ni':
            _pp=body_ppr(0).replace('<w:jc','<w:ind w:left="420" w:hanging="420"/><w:jc')
            out.append('<w:p>'+_pp+runs(b[1]+'、'+b[2],sz,math)+'</w:p>')
        elif k=='note':
            if drop_notes: prev=k; continue
            out.append(f'<w:p><w:pPr><w:spacing w:before="0" w:after="120" w:line="360" w:lineRule="auto"/></w:pPr>{runs(b[1],21,math,italic=True)}</w:p>')
        elif k=='cap_t':
            csz = 22 if prof.name!='main' else 24
            out.append(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="60" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{runs(b[1],csz,math,bold=True)}</w:p>')
        elif k=='cap_f':
            csz = 22 if prof.name!='main' else 24
            out.append(f'<w:p><w:pPr><w:spacing w:before="40" w:after="160" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{runs(b[1],csz,math,bold=True)}</w:p>')
        elif k=='tbl':
            out.append(table_xml(b[1],prof,math,center_all=(prof.name=='res'))); out.append(spacer())
        elif k=='eq':
            out.append(eq_xml(b[1],b[2],prof,math)); out.append(spacer())
        elif k=='img':
            wi=min(b[2],6.2) if prof.name=='main' else b[2]
            src=os.path.join(figroot,b[1]); jp=jpeg_for(src,wi)
            im=Image.open(jp); cx=int(wi*914400); cy=int(cx*im.height/im.width)
            rid=f'{rid_prefix}{len(imgs)+1}'; imgs.append((rid,jp))
            out.append(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{drawing_xml(rid,cx,cy,did0+len(imgs),os.path.basename(b[1]))}</w:p>')
        prev=k
    return ''.join(out),imgs

# ------------------------------------------------------------ package
def build_package(template_dir,out_docx,body_xml,imgs,keep_sect=True):
    tmp=tempfile.mkdtemp(); pk=tmp+'/pk'; shutil.copytree(template_dir,pk)
    dp=pk+'/word/document.xml'; x=open(dp,encoding='utf8').read()
    i=x.index('<w:body>')+8
    sect=re.findall(r'<w:sectPr.*?</w:sectPr>',x,re.S)[-1]
    # root must declare wp, a, pic? we declare a/pic inline; ensure wp, r exist
    head=x[:i]
    for ns,uri in (('wp','http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'),('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships'),('m',MNS)):
        if f'xmlns:{ns}=' not in head: head=head.replace('<w:document ',f'<w:document xmlns:{ns}="{uri}" ',1)
    open(dp,'w',encoding='utf8').write(head+body_xml+sect+'</w:body></w:document>')
    # rels
    rp=pk+'/word/_rels/document.xml.rels'; r=open(rp,encoding='utf8').read()
    r=re.sub(r'<Relationship [^>]*relationships/image"[^>]*/>','',r)
    shutil.rmtree(pk+'/word/media',ignore_errors=True); os.makedirs(pk+'/word/media')
    add=''
    for n,(rid,jp) in enumerate(imgs,1):
        shutil.copy(jp,f'{pk}/word/media/fig{n}.jpg')
        add+=f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig{n}.jpg"/>'
    r=r.replace('</Relationships>',add+'</Relationships>'); open(rp,'w',encoding='utf8').write(r)
    ct=pk+'/[Content_Types].xml'; c=open(ct,encoding='utf8').read()
    if 'Extension="jpg"' not in c: c=c.replace('<Default ','<Default Extension="jpg" ContentType="image/jpeg"/><Default ',1)
    open(ct,'w',encoding='utf8').write(c)
    pass
    z=zipfile.ZipFile(out_docx,'w',zipfile.ZIP_DEFLATED)
    z.write(ct,'[Content_Types].xml')
    for root,d,fs in os.walk(pk):
        for f in fs:
            full=os.path.join(root,f); arc=os.path.relpath(full,pk)
            if arc=='[Content_Types].xml': continue
            z.write(full,arc)
    z.close(); shutil.rmtree(tmp)

def export_csv(blocks,prefix_dir):
    os.makedirs(prefix_dir,exist_ok=True); n=0
    for i,b in enumerate(blocks):
        if b[0]=='cap_t':
            nxt=next((blocks[j] for j in range(i+1,len(blocks)) if blocks[j][0]=='tbl'),None)
            m=re.match(r'(表)(\d+)\s+(.*)',b[1]); title=plain_cell(m.group(3))
            title=re.sub(r'[（(].*?[）)]','',title).strip()
            fn=f'{prefix_dir}/{m.group(1)}{int(m.group(2)):02d}_{title}.csv'
            with open(fn,'w',encoding='utf-8-sig',newline='') as f: csv.writer(f).writerows([[plain_cell(c) for c in r] for r in nxt[1]])
            n+=1
    return n
