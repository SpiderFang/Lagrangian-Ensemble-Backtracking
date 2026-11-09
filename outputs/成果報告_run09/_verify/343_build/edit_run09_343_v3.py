# -*- coding: utf-8 -*-
"""2026-10-21 第三版修改（圖 3–7 用語改依服務建議書）：在目前之 3-4-3_run09.docx（v2 之後）上就地修改。
只改 word/document.xml 與 word/media/*（5 張到達時間總覽圖），其餘 zip 項目 byte 不動；Word 之 rsid／格式標記保留。
  1. 圖 3–7 置換為 figures-zh-v3（有效波高、全中文圖例、Y 軸垂直標題、日期（年/月/日）；嵌入寬 6.267 in、1:1）
  2. 導引句：顯著波高→有效波高；補潮汐強度白話定義，並說明黑色粗虛線與紅色點線
  3. 圖片替代文字同步
用法：python3 edit_run09_343_v3.py <src.docx> <dst.docx>
"""
import sys, os, zipfile, struct, json, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/tmp/w3')
from lxml import etree
from lib import q, ptext
from edit import replace_in_p

SRC, DST = sys.argv[1], sys.argv[2]
FIGS_ARR = os.path.join(HERE, 'figs_arrival_zh_v3')
NS_WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
SITES = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read('word/document.xml'))
body = root.find(q('w:body'))
ps = [k for k in body if k.tag == q('w:p')]
log = []


def nop(p):
    for e in p.findall(q('w:proofErr')):
        p.remove(e)


def rep(p, old, new):
    replace_in_p(p, old, new)
    log.append((old, new))


g = [p for p in ps if ptext(p).startswith('以下列出各研究地點之到達時間驗證圖')]
assert len(g) == 1
p_guide = g[0]
nop(p_guide)
rep(p_guide, '圖中最上方為 25 小時潮汐強度曲線，標示', '圖中最上方為潮汐強度曲線，標示')
rep(p_guide, '改以顯著波高或流速呈現，高浪事件標示於顯著波高曲線，強流事件標示於流速曲線。各季色塊',
    '改以有效波高或流速呈現，高浪事件標示於有效波高曲線，強流事件標示於流速曲線。'
    '圖中潮汐強度即前述各時刻前後 25 小時之水位起伏大小，以海面高度之均方根計算；'
    '黑色粗虛線為各季潮汐強度之中位數（大潮與小潮之分界），紅色點線為有效波高與流速之全年第 99 百分位。各季色塊')
assert '顯著波高' not in ptext(p_guide) and '示性波高' not in ptext(p_guide)

rels = zin.read('word/_rels/document.xml.rels').decode('utf-8')
rid2target = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="(media/[^"]+)"', rels))
rid2target.update({a: b for b, a in re.findall(r'Target="(media/[^"]+)"[^>]*?Id="(rId\d+)"', rels)})
inls = list(root.iter('{%s}inline' % NS_WP))
assert len(inls) == 7, len(inls)
media_new = {}
fig_map = []
for k, (sid, zh) in enumerate(SITES):
    inl = inls[2 + k]
    blip = inl.find('.//{%s}blip' % NS_A)
    rid = blip.get('{%s}embed' % NS_R)
    b = open(os.path.join(FIGS_ARR, sid, 'wp61_%s_overview_S_notext.png' % sid), 'rb').read()
    w, h = struct.unpack('>II', b[16:24])
    cx = 5730240
    cy = int(round(cx * h / w))
    media_new['word/' + rid2target[rid]] = b
    inl.find('{%s}extent' % NS_WP).set('cx', str(cx)); inl.find('{%s}extent' % NS_WP).set('cy', str(cy))
    ext = inl.find('.//{%s}xfrm/{%s}ext' % (NS_A, NS_A)); ext.set('cx', str(cx)); ext.set('cy', str(cy))
    d = inl.find('{%s}docPr' % NS_WP)
    assert d.get('descr').startswith(zh + '之到達時間驗證圖（全年總覽）'), d.get('descr')
    d.set('descr', '%s之到達時間驗證圖（全年總覽），上為潮汐強度與 50 個到達時間之位置，中為有效波高，下為流速' % zh)
    fig_map.append((3 + k, zh, 'word/' + rid2target[rid], __import__('hashlib').sha256(b).hexdigest(), (cx, cy), round(cy / 914400, 3)))
out_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
zo = zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == 'word/document.xml':
        data = out_xml
    elif it.filename in media_new:
        data = media_new[it.filename]
    ni = zipfile.ZipInfo(it.filename, it.date_time); ni.compress_type = it.compress_type
    ni.external_attr = it.external_attr; ni.create_system = it.create_system
    zo.writestr(ni, data)
zo.close()
json.dump({'text_changes': log, 'fig_map': fig_map}, open(os.path.join(HERE, 'edit_run09_343_v3.log.json'), 'w'), ensure_ascii=False, indent=1)
print('ok', DST, 'figs', len(fig_map), [f[5] for f in fig_map])
