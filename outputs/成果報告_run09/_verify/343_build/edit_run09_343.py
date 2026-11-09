# -*- coding: utf-8 -*-
"""3-4-3_run08.docx（使用者手改主檔，唯讀）→ 3-4-3_run09.docx。
只改 word/document.xml 之內容、置換圖1/圖2 圖檔、新增 10 張到達時間驗證圖及其關聯；其餘部分原封不動。
用法：python3 edit_run09_343.py <src.docx> <dst.docx>
"""
import sys, os, zipfile, copy, struct, random, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lxml import etree
from lib import q, ptext
from edit import replace_in_p, set_cell_text, rtext

SRC, DST = sys.argv[1], sys.argv[2]
FIGS_NEW = os.path.join(HERE, 'figs_new')
FIGS_ARR = os.path.join(HERE, 'figs_arrival_zh')  # 圖內文字全中文版（WP-61 commit 227b3a7）
NS_WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS_PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
NS_W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
NS_WP14 = 'http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing'

zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read('word/document.xml'))
body = root.find(q('w:body'))
items = list(body)
ps = [k for k in items if k.tag == q('w:p')]
log = []  # (原文, 新文)


def nop(p):
    for e in p.findall(q('w:proofErr')):
        p.remove(e)


def findp_in(sub):
    r = [p for p in ps if sub in ptext(p)]
    assert len(r) == 1, (sub, len(r))
    return r[0]


def rep(p, old, new):
    replace_in_p(p, old, new)
    log.append((old, new))


# ---------- 1. 受體段落 ----------
p_rec = findp_in('抽樣範圍為以各研究地點錨點為圓心')
nop(p_rec)
rep(p_rec, '半徑 12.5 km 之圓', '半徑 10 km 之圓')
rep(p_rec,
    '符合條件者再以無放回均勻隨機抽取 5 個位置，抽中後另須通過流場與波浪資料之逐時空間支援檢查，未通過者剔除，並依隨機抽樣之順序遞補。',
    '符合條件者再依方位及距研究地點錨點之距離，分別將半徑 10 km 範圍內之候選海域等分為面積相同之 5 個區間，'
    '使每一方位區間與每一距離區間各恰有 1 個受體水平位置，並於所屬區塊（方位區間與距離區間之交會處）內隨機抽取；'
    '龜山島之西半部配置 3 個、東半部配置 2 個。'
    '抽中後另須通過流場與波浪資料之逐時空間支援檢查，未通過者剔除，並於同一區塊內依隨機順序遞補。')

# ---------- 2. 表 2 南灣錨點 ----------
tb = [k for k in items if k.tag == q('w:tbl') and ''.join(x.text or '' for x in k.iter(q('w:t'))).startswith('研究地點經度')]
assert len(tb) == 1
done = 0
for tr in tb[0].findall(q('w:tr')):
    tcs = tr.findall(q('w:tc'))
    if ''.join(x.text or '' for x in tcs[0].iter(q('w:t'))) == '南灣':
        assert ''.join(x.text or '' for x in tcs[1].iter(q('w:t'))) == '120.76'
        assert ''.join(x.text or '' for x in tcs[2].iter(q('w:t'))) == '21.95'
        set_cell_text(tcs[1], '120.77'); set_cell_text(tcs[2], '21.92'); done += 1
assert done == 1
log.append(('表 2 南灣列：120.76、21.95', '表 2 南灣列：120.77、21.92'))

# ---------- 3. 到達時間段落 ----------
p_arr = findp_in('個到達時間由')
assert ptext(p_arr).startswith('50 個到達時間由 48 個分層時刻')
nop(p_arr)
rep(p_arr, '個時刻。三種潮相代理分別為',
    '個時刻。大潮與小潮之區分，以各時刻前後共 25 小時（約一個潮日）之潮汐強度（該時段水位起伏之大小）與該季之中位數比較，'
    '高於中位數者為大潮，低於者為小潮。挑選時排除流場資料於日間銜接處水位不連續之時刻。三種潮相代理分別為')
rep(p_arr, '分布於全年十二個月，各月 5 至 9 個以確保', '涵蓋四季，每季 12 個分層時刻，大潮、小潮各 6 個，以確保')

# ---------- 3b. 溯源停止邊界段末加一句（數字取自 termination_by_group.csv）----------
import csv
_t = {}
for _r in csv.DictReader(open(os.path.join(HERE, 'src', 'termination_by_group.csv'), encoding='utf-8-sig')):
    if _r['grouping'] == 'site_all':
        _t[_r['site']] = (int(_r['n_total']), int(_r['n_crossed']))
def _pct(k):
    n, c = _t[k]
    return c / n * 100.0
assert all(_t[k][0] == 100000 for k in _t)
assert _pct('gongliao') == _pct('guishan') == _pct('hsinchu') == 100.0
STOP = ('回溯期間上限為 180 天。於此期間內，貢寮、龜山島與新竹外海之全部虛擬質點離開流場範圍邊界，南灣有 %.3f%%，連江有 %.3f%%。'
        % (_pct('nanwan'), _pct('lienchiang')))
p_stop = findp_in('溯源停止邊界（')
nop(p_stop)
rep(p_stop, '潛在外部來源與邊界通量路徑。', '潛在外部來源與邊界通量路徑。' + STOP)

# ---------- 4. 圖檔置換（圖1、圖2）----------
img_bytes = {
    'word/media/image1.png': open(os.path.join(FIGS_NEW, '343_run09_fig1_circle.png'), 'rb').read(),
    'word/media/image2.png': open(os.path.join(FIGS_NEW, '343_run09_fig2_circle.png'), 'rb').read(),
}


def png_size(b):
    assert b[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', b[16:24])


old1, old2 = zin.read('word/media/image1.png'), zin.read('word/media/image2.png')
assert png_size(old1) == png_size(img_bytes['word/media/image1.png']), '圖1 長寬比改變'
assert png_size(old2) == png_size(img_bytes['word/media/image2.png']), '圖2 長寬比改變'

# ---------- 5. 新增到達時間驗證圖 ----------
SITES = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
fig_para = [k for k in ps if k.find('.//' + q('w:drawing')) is not None]
assert len(fig_para) == 2
tmpl_fig = fig_para[1]                      # 圖2 圖片段落（置中、keepNext）
tmpl_cap = tmpl_fig.getnext()               # 圖2 圖題段落
assert ptext(tmpl_cap).startswith('圖2 ')
tmpl_body = p_arr                           # 到達時間段落之版式（導引句用）
CONTENT_W = 5730240                         # 與圖2 同寬（文字區寬）

rid_next = 100
docpr_next = 9603
new_rels, new_media = [], {}
seq = []  # 要插入的段落


def strip_ids(p):
    for a in list(p.attrib):
        if a.startswith('{%s}' % NS_W14) or a.startswith('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rsid'):
            del p.attrib[a]
    for lr in p.iter(q('w:lastRenderedPageBreak')):
        lr.getparent().remove(lr)


def set_paraid(p):
    p.set('{%s}paraId' % NS_W14, '%08X' % random.randint(0x10000000, 0x7FFFFFFF))
    p.set('{%s}textId' % NS_W14, '77777777')


random.seed(20261020)
used = set()


def mk_fig(png_path, media_name, name, descr):
    global rid_next, docpr_next
    b = open(png_path, 'rb').read()
    w, h = png_size(b)
    cx = CONTENT_W
    cy = int(round(cx * h / w))
    p = copy.deepcopy(tmpl_fig)
    strip_ids(p); set_paraid(p)
    inl = p.find('.//{%s}inline' % NS_WP)
    for a in list(inl.attrib):
        if a.startswith('{%s}' % NS_WP14):
            del inl.attrib[a]
    inl.find('{%s}extent' % NS_WP).set('cx', str(cx)); inl.find('{%s}extent' % NS_WP).set('cy', str(cy))
    dp = inl.find('{%s}docPr' % NS_WP); dp.set('id', str(docpr_next)); dp.set('name', name); dp.set('descr', descr)
    cnv = inl.find('.//{%s}cNvPr' % NS_PIC); cnv.set('id', str(docpr_next)); cnv.set('name', name)
    blip = inl.find('.//{%s}blip' % NS_A); rid = 'rId%d' % rid_next; blip.set('{%s}embed' % NS_R, rid)
    ext = inl.find('.//{%s}xfrm/{%s}ext' % (NS_A, NS_A)); ext.set('cx', str(cx)); ext.set('cy', str(cy))
    new_rels.append((rid, 'media/%s' % media_name)); new_media['word/media/%s' % media_name] = b
    rid_next += 1; docpr_next += 1
    return p


def mk_cap(num, title):
    p = copy.deepcopy(tmpl_cap)
    strip_ids(p); set_paraid(p)
    runs = p.findall(q('w:r'))
    rpr = copy.deepcopy(runs[0].find(q('w:rPr')))
    for r in runs:
        p.remove(r)
    for t in ('圖', '%d ' % num, title):
        r = etree.SubElement(p, q('w:r')); r.append(copy.deepcopy(rpr))
        te = etree.SubElement(r, q('w:t')); te.text = t
        if t != t.strip():
            te.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return p


# 導引句
GUIDE = ('以下列出各研究地點之到達時間驗證圖，每個研究地點 2 張：全年總覽圖與逐時刻放大圖。'
         '圖中標示 50 個到達時間在潮汐強度與水位曲線上的位置，橘色標記為大潮、藍色標記為小潮；'
         '三種潮相以標記形狀區分，向上三角為漲潮最快，向下三角為退潮最快，菱形為近平潮（最接近漲退潮交替之時刻）。'
         '2 個事件時刻改以示性波高或流速呈現，高浪事件標示於示性波高曲線，強流事件標示於流速曲線。'
         '貢寮見圖 3、圖 4，龜山島見圖 5、圖 6，新竹外海見圖 7、圖 8，南灣見圖 9、圖 10，連江見圖 11、圖 12。')
gp = copy.deepcopy(tmpl_body)
strip_ids(gp); set_paraid(gp)
for r in gp.findall(q('w:r')) + gp.findall(q('w:proofErr')):
    gp.remove(r)
r0 = etree.SubElement(gp, q('w:r'))
rp = etree.SubElement(r0, q('w:rPr')); sz = etree.SubElement(rp, q('w:szCs')); sz.set(q('w:val'), '28')
t0 = etree.SubElement(r0, q('w:t')); t0.text = GUIDE
seq.append(gp)
log.append(('（新增導引句）', GUIDE))

num = 3
fig_map = []  # (圖號, 地點, 種類, 檔名)
n_img = 3
for sid, zh in SITES:
    for kind, fn_tpl, ktitle, kdesc in (
        ('overview', 'wp61_%s_overview_U_notext.png', '全年總覽', '全年總覽，上為 25 小時潮汐強度曲線，中為示性波高，下為流速，並標示 50 個到達時間'),
        ('zoom', 'wp61_%s_zoom_S_notext.png', '逐時刻放大', '逐時刻放大，每一小圖為一個分層時刻前後之水位曲線，下方為高浪事件之示性波高與強流事件之流速'),
    ):
        fn = fn_tpl % sid
        title = '%s到達時間驗證圖（%s）' % (zh, ktitle)
        descr = '%s之到達時間驗證圖，%s' % (zh, kdesc)
        seq.append(mk_fig(os.path.join(FIGS_ARR, fn), 'image%d.png' % n_img, '到達時間驗證圖_%s_%s.png' % (zh, ktitle), descr))
        seq.append(mk_cap(num, title))
        log.append(('（新增圖題）', '圖%d %s' % (num, title)))
        fig_map.append((num, zh, ktitle, fn))
        num += 1; n_img += 1

anchor = p_arr
for el in seq:
    anchor.addnext(el); anchor = el

# ---------- 6. 輸出 ----------
out_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
rels = zin.read('word/_rels/document.xml.rels').decode('utf-8')
add = ''.join('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="%s"/>' % (i, t) for i, t in new_rels)
assert '</Relationships>' in rels and 'rId100' not in rels
rels = rels.replace('</Relationships>', add + '</Relationships>')
zo = zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == 'word/document.xml':
        data = out_xml
    elif it.filename == 'word/_rels/document.xml.rels':
        data = rels.encode('utf-8')
    elif it.filename in img_bytes:
        data = img_bytes[it.filename]
    ni = zipfile.ZipInfo(it.filename, it.date_time); ni.compress_type = it.compress_type
    ni.external_attr = it.external_attr; ni.create_system = it.create_system
    zo.writestr(ni, data)
for n, b in new_media.items():
    ni = zipfile.ZipInfo(n, (1980, 1, 1, 0, 0, 0)); ni.compress_type = zipfile.ZIP_STORED; ni.external_attr = 0o600 << 16
    zo.writestr(ni, b)
zo.close()
json.dump({'text_changes': log, 'fig_map': fig_map}, open(os.path.join(HERE, 'edit_run09_343.log.json'), 'w'), ensure_ascii=False, indent=1)
print('ok', DST, 'figs', len(fig_map))
