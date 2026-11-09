import sys
sys.path.insert(0,'/sessions')
from lxml import etree
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
root=etree.parse(sys.argv[1]).getroot()
body=root.find('{%s}body'%W)
for i,el in enumerate(body):
    tag=el.tag.split('}')[1]
    if tag=='p':
        t=''.join((x.text or '') for x in el.iter('{%s}t'%W,'{%s}t'%M))
        extra=''
        if el.find('.//{%s}drawing'%W): extra=' [DRAWING]'
        print(i,'P',len(t),t+extra)
    elif tag=='tbl':
        t=''.join((x.text or '') for x in el.iter('{%s}t'%W,'{%s}t'%M))
        print(i,'TBL',t[:150])
    else: print(i,tag)
