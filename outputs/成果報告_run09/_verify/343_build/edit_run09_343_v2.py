# -*- coding: utf-8 -*-
"""2026-10-21 第二版修改：在使用者已用 Word 修改過的現行 3-4-3_run09.docx 上就地修改（不從 run08 重建）。
只改 word/document.xml、word/_rels/document.xml.rels 與 word/media/*；其餘 zip 項目 byte 不動；Word 留下的 rsid／格式標記全部保留。
  1. 表 2 恢復 5 欄（研究地點、經度、緯度、抽樣範圍經度、抽樣範圍緯度）；圖表題與導引句同步加「與抽樣範圍」
  2. 圖 1、圖 2 置換為 run09 錨點、10 km 圓之重畫版（圖例「抽樣範圍」，與 run08 同樣式）
  3. 到達時間驗證圖：每地點 1 張全年總覽（圖 3–7，S 版、無文字版、中英並列）；刪逐時刻放大圖；導引句與圖說同步
用法：python3 edit_run09_343_v2.py <src.docx> <dst.docx>
"""
import sys, os, zipfile, copy, struct, random, json, math, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/tmp/w3')
from lxml import etree
from lib import q, ptext
from edit import replace_in_p, set_cell_text, rtext, set_rtext

SRC, DST = sys.argv[1], sys.argv[2]
FIGS12 = os.path.join(HERE, 'figs_new_v2')
FIGS_ARR = os.path.join(HERE, 'figs_arrival_zh_v2')
NS_WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS_PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
NS_W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
SITES = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
# run09 母體 release config（.203 release-180d-full_horizon_from_deposition.yaml）之 study_sites[].anchor_lonlat
ANCHORS = {'gongliao': (121.9223889, 25.0964444), 'guishan': (121.939148, 24.850326), 'hsinchu': (120.45, 24.75),
           'nanwan': (120.770675, 21.92041), 'lienchiang': (119.95, 26.2)}
RADIUS_KM = 10.0
EN = '–'


def sampling_range(lon0, lat0):
    dlat = RADIUS_KM / 111.32
    dlon = RADIUS_KM / (111.32 * math.cos(math.radians(lat0)))
    return ('%.2f%s%.2f' % (lon0 - dlon, EN, lon0 + dlon), '%.2f%s%.2f' % (lat0 - dlat, EN, lat0 + dlat))


zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read('word/document.xml'))
body = root.find(q('w:body'))
items = list(body)
ps = [k for k in items if k.tag == q('w:p')]
log = []


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


all_ids = set(re.findall(r'w14:paraId="([0-9A-F]{8})"', zin.read('word/document.xml').decode('utf-8')))
random.seed(20261021)


def new_para_id():
    while True:
        v = '%08X' % random.randint(0x10000000, 0x7FFFFFFF)
        if v not in all_ids:
            all_ids.add(v)
            return v


# ---------- 1. 表 2 ----------
tb = [k for k in items if k.tag == q('w:tbl') and ''.join(x.text or '' for x in k.iter(q('w:t'))).startswith('研究地點經度')]
assert len(tb) == 1
tbl = tb[0]
W5 = [1700, 1350, 1350, 2030, 2027]
grid = tbl.find(q('w:tblGrid'))
cols = grid.findall(q('w:gridCol'))
assert len(cols) == 3
for gc in cols:
    grid.remove(gc)
for w in W5:
    gc = etree.SubElement(grid, q('w:gridCol')); gc.set(q('w:w'), str(w))
site_zh2id = {zh: sid for sid, zh in SITES}
table_rows = []
for ri, tr in enumerate(tbl.findall(q('w:tr'))):
    tcs = tr.findall(q('w:tc'))
    assert len(tcs) == 3
    c_lon, c_lat = copy.deepcopy(tcs[1]), copy.deepcopy(tcs[2])
    for c in (c_lon, c_lat):
        for p in c.iter(q('w:p')):
            p.set('{%s}paraId' % NS_W14, new_para_id())
    if ri == 0:
        assert ''.join(x.text or '' for x in tcs[0].iter(q('w:t'))) == '研究地點'
        r0 = c_lon.find(q('w:p')).findall(q('w:r'))[0]; assert rtext(r0) == '經度'; set_rtext(r0, '抽樣範圍經度')
        r0 = c_lat.find(q('w:p')).findall(q('w:r'))[0]; assert rtext(r0) == '緯度'; set_rtext(r0, '抽樣範圍緯度')
        table_rows.append(('研究地點', '經度 (°E)', '緯度 (°N)', '抽樣範圍經度 (°E)', '抽樣範圍緯度 (°N)'))
    else:
        zh = ''.join(x.text or '' for x in tcs[0].iter(q('w:t')))
        lon0, lat0 = ANCHORS[site_zh2id[zh]]
        # 錨點欄位須與四捨五入後之現值一致
        assert ''.join(x.text or '' for x in tcs[1].iter(q('w:t'))) == '%.2f' % lon0, (zh, lon0)
        assert ''.join(x.text or '' for x in tcs[2].iter(q('w:t'))) == '%.2f' % lat0, (zh, lat0)
        rl, rb = sampling_range(lon0, lat0)
        set_cell_text(c_lon, rl); set_cell_text(c_lat, rb)
        table_rows.append((zh, '%.2f' % lon0, '%.2f' % lat0, rl, rb))
    tcs[2].addnext(c_lon); c_lon.addnext(c_lat)
    for c, w in zip(tr.findall(q('w:tc')), W5):
        c.find(q('w:tcPr')).find(q('w:tcW')).set(q('w:w'), str(w))
log.append(('表 2 由 3 欄恢復為 5 欄', table_rows))

p_intro = findp_in('五個研究地點之錨點座標見表')
nop(p_intro); rep(p_intro, '五個研究地點之錨點座標見表', '五個研究地點之錨點座標與抽樣範圍見表')
p_cap2 = [p for p in ps if ptext(p).replace(' ', '') == '表2五個研究地點之錨點座標']
assert len(p_cap2) == 1
nop(p_cap2[0]); rep(p_cap2[0], '五個研究地點之錨點座標', '五個研究地點之錨點座標與抽樣範圍')

# ---------- 2. 到達時間導引句 ----------
p_guide = findp_in('以下列出各研究地點之到達時間驗證圖')
nop(p_guide)
rep(p_guide, '每個研究地點 2 張：全年總覽圖與逐時刻放大圖。圖中標示 50 個到達時間在潮汐強度與水位曲線上的位置，',
    '每個研究地點 1 張（全年總覽圖）。圖中最上方為 25 小時潮汐強度曲線，標示 50 個到達時間在曲線上的位置，')
rep(p_guide, '事件時刻改以示性波高或流速呈現，高浪事件標示於示性波高曲線，強流事件標示於流速曲線。貢寮見圖 3、圖 4，龜山島見圖 5、圖 6，新竹外海見圖 7、圖 8，南灣見圖 9、圖 10，連江見圖 11、圖 12。',
    '事件時刻改以顯著波高或流速呈現，高浪事件標示於顯著波高曲線，強流事件標示於流速曲線。'
    '各季色塊上方標示該季大潮與小潮之時刻數，圖例括號內為各類時刻數，合計 50 個。'
    '標記位於實際到達時間，同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊。'
    '貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7。')

# ---------- 3. 圖片段落 ----------
rels = zin.read('word/_rels/document.xml.rels').decode('utf-8')
rid2target = dict(re.findall(r'<Relationship Id="(rId\d+)"[^>]*?Target="(media/[^"]+)"', rels))
rid2target.update({a: b for b, a in re.findall(r'Target="(media/[^"]+)"[^>]*?Id="(rId\d+)"', rels)})
fig_paras = [p for p in ps if p.find('.//' + q('w:drawing')) is not None]
assert len(fig_paras) == 12, len(fig_paras)
fig1_p, fig2_p, arr = fig_paras[0], fig_paras[1], fig_paras[2:]


def rid_of(p):
    return p.find('.//{%s}blip' % NS_A).get('{%s}embed' % NS_R)


def png_size(b):
    assert b[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', b[16:24])


media_new = {}
drop_media, drop_rids = [], []
# 圖 1、圖 2
for p, fn in ((fig1_p, '343_run09v2_fig1_circle.png'), (fig2_p, '343_run09v2_fig2_circle.png')):
    b = open(os.path.join(FIGS12, fn), 'rb').read()
    old = zin.read('word/' + rid2target[rid_of(p)])
    assert png_size(b) == png_size(old), fn
    media_new['word/' + rid2target[rid_of(p)]] = b
# 到達時間：奇數（0,2,4,...）為總覽、偶數為逐時刻放大；依圖題確認
fig_map = []
n = 3
for k in range(5):
    pov, pzoom = arr[2 * k], arr[2 * k + 1]
    cap_ov, cap_zm = pov.getnext(), pzoom.getnext()
    assert '全年總' in ptext(cap_ov) and '逐時刻放大' in ptext(cap_zm), (ptext(cap_ov), ptext(cap_zm))
    sid, zh = SITES[k]
    assert zh in ptext(cap_ov) and zh in ptext(cap_zm)
    # 刪放大圖（圖片段落＋圖題）
    drop_rids.append(rid_of(pzoom)); drop_media.append('word/' + rid2target[rid_of(pzoom)])
    body.remove(pzoom); body.remove(cap_zm)
    # 總覽圖：置換圖檔、尺寸、替代文字
    fn = 'wp61_%s_overview_S_notext.png' % sid
    b = open(os.path.join(FIGS_ARR, sid, fn), 'rb').read()
    w, h = png_size(b)
    cx = 5730240; cy = int(round(cx * h / w))
    media_new['word/' + rid2target[rid_of(pov)]] = b
    inl = pov.find('.//{%s}inline' % NS_WP)
    inl.find('{%s}extent' % NS_WP).set('cx', str(cx)); inl.find('{%s}extent' % NS_WP).set('cy', str(cy))
    ext = inl.find('.//{%s}xfrm/{%s}ext' % (NS_A, NS_A)); ext.set('cx', str(cx)); ext.set('cy', str(cy))
    inl.find('{%s}docPr' % NS_WP).set('descr', '%s之到達時間驗證圖（全年總覽），上為 25 小時潮汐強度與 50 個到達時間之位置，中為顯著波高，下為流速；圖內文字中英並列' % zh)
    # 圖號
    nop(cap_ov)
    runs = cap_ov.findall(q('w:r'))
    assert rtext(runs[0]) == '圖' and rtext(runs[1]).strip().isdigit(), [rtext(r) for r in runs]
    set_rtext(runs[1], '%d ' % n)
    log.append(('圖題 %s' % zh, '圖%d %s' % (n, ptext(cap_ov)[len('圖') + len(rtext(runs[1])):])))
    fig_map.append((n, zh, fn, 'sha256:' + __import__('hashlib').sha256(b).hexdigest(), (cx, cy)))
    n += 1

# ---------- 4. 輸出 ----------
out_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
for rid in drop_rids:
    pat = re.compile(r'<Relationship Id="%s"[^>]*/>' % rid)
    assert len(pat.findall(rels)) == 1
    rels = pat.sub('', rels)
zo = zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    if it.filename in drop_media:
        continue
    data = zin.read(it.filename)
    if it.filename == 'word/document.xml':
        data = out_xml
    elif it.filename == 'word/_rels/document.xml.rels':
        data = rels.encode('utf-8')
    elif it.filename in media_new:
        data = media_new[it.filename]
    ni = zipfile.ZipInfo(it.filename, it.date_time); ni.compress_type = it.compress_type
    ni.external_attr = it.external_attr; ni.create_system = it.create_system
    zo.writestr(ni, data)
zo.close()
json.dump({'text_changes': log, 'fig_map': fig_map, 'dropped_media': drop_media, 'dropped_rids': drop_rids,
           'table2': table_rows}, open(os.path.join(HERE, 'edit_run09_343_v2.log.json'), 'w'), ensure_ascii=False, indent=1)
print('ok', DST, 'figs', len(fig_map))
