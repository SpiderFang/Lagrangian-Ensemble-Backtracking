# -*- coding: utf-8 -*-
"""由 report_run08_check.py 產生 report_run09_check.py（逐項取代預期值並新增 run09 檢查）。"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', '..', '..', 'data', 'settling_velocity_derivation_v7', 'report_run08_check.py')
s = open(SRC, encoding='utf-8').read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)


sub('''HERE = os.path.dirname(os.path.abspath(__file__))
DOCX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'outputs', '成果報告_run08', '3-4-3_run08.docx')
V7 = sys.argv[2] if len(sys.argv) > 2 else HERE
OUT = sys.argv[3] if len(sys.argv) > 3 else HERE''',
'''HERE = os.path.dirname(os.path.abspath(__file__))
V7DIR = os.path.join(HERE, '..', '..', '..', '..', 'data', 'settling_velocity_derivation_v7')
DOCX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', '3-4-3_run09.docx')
V7 = sys.argv[2] if len(sys.argv) > 2 else V7DIR
OUT = sys.argv[3] if len(sys.argv) > 3 else V7DIR''')

sub("ok('各研究地點之 50 個到達時間' in fulltext and '五個研究地點之錨點座標見表 2。' in fulltext, '研究地點（到達時間段、座標表前）')",
    "ok('各研究地點之 50 個到達時間' in fulltext and '五個研究地點之錨點座標與抽樣範圍見表 2。' in fulltext and '五個研究地點之錨點座標見表 2。' not in fulltext, '研究地點（到達時間段、座標表前）')")
sub("ok(any(t_ == '表2 五個研究地點之錨點座標' for t_ in paras) and '五個研究地點之錨點座標見表 2。' in fulltext, '表2 標題與導句')",
    "ok(any(t_ == '表2 五個研究地點之錨點座標與抽樣範圍' for t_ in paras) and '五個研究地點之錨點座標與抽樣範圍見表 2。' in fulltext, '表2 標題與導句')")
sub("""ok(len(site_[0][0]) == 3 and site_[0][0][2].startswith('緯度') and all(len(r_) == 3 for r_ in site_[0]), '表2 僅 3 欄（地點、錨點經度、錨點緯度），無抽樣範圍欄：%s' % site_[0][0])
ok(not any('抽樣範圍' in c_ or re.search(r'\\d–\\d', c_) for r_ in site_[0] for c_ in r_), '表2 無範圍欄位內容')""",
    """ok(site_[0][0] == ['研究地點', '經度 (°E)', '緯度 (°N)', '抽樣範圍經度 (°E)', '抽樣範圍緯度 (°N)'] and all(len(r_) == 5 for r_ in site_[0]), '表2 五欄（與 run08 同表頭）：%s' % site_[0][0])
import math as _m
_anc = {'貢寮': (121.9223889, 25.0964444), '龜山島': (121.939148, 24.850326), '新竹外海': (120.45, 24.75), '南灣': (120.770675, 21.92041), '連江': (119.95, 26.2)}
def _rng(lon_, lat_, km_=10.0):
    dlat_ = km_ / 111.32
    dlon_ = km_ / (111.32 * _m.cos(_m.radians(lat_)))
    return ('%.2f–%.2f' % (lon_ - dlon_, lon_ + dlon_), '%.2f–%.2f' % (lat_ - dlat_, lat_ + dlat_))
_want5 = [[k_, '%.2f' % round(v_[0], 2), '%.2f' % round(v_[1], 2)] + list(_rng(v_[0], v_[1])) for k_, v_ in _anc.items()]
ok([r_ for r_ in site_[0][1:]] == _want5, '表2 五欄數值等於由錨點與半徑 10 km 公式（緯 ±10/111.32、經 ±10/(111.32·cosφ)）計算值：%s' % site_[0][1:])
_tblx = [e_ for e_ in body if e_.tag == W + 'tbl' and ''.join(x_.text or '' for x_ in e_.iter(W + 't')).startswith('研究地點經度')]
_grid = [int(g_.get(W + 'w')) for g_ in _tblx[0].find(W + 'tblGrid')]
ok(_grid == [1700, 1350, 1350, 2030, 2027] and len(_tblx) == 1, '表2 欄寬 %s' % _grid)""")

sub("ok('代表物件之尺寸與輸入參數（表 A4）' in fulltext, '表1 後句指向表 A4')",
    "ok('代表物件之尺寸與輸入參數、實測量級對照與敏感度分析詳見附錄' in fulltext and '代表物件之尺寸與輸入參數（表 A4）' not in fulltext, '表1 後句（使用者於 Word 手改，刪去「（表 A4）」，以現況為準）')")
sub('"""核對報告 docx 內的數字', '"""（run09 版，由 make_check_run09.py 自 report_run08_check.py 產生）核對報告 docx 內的數字')
# 圖片替代文字：共 12 張；前兩張（圖1、圖2）含「各研究地點」
sub("ok(len(dsc) == 2 and all('各研究地點' in d.decode('utf8') and '站' not in d.decode('utf8') for d in dsc), '圖片替代文字')",
    "ok(len(dsc) == 7 and all('各研究地點' in d.decode('utf8') for d in dsc[:2]) and all('站' not in d.decode('utf8') and '放大' not in d.decode('utf8') and '半徑' not in d.decode('utf8') for d in dsc), '圖片替代文字（共 7 張，圖1、圖2 含各研究地點，無放大圖、無半徑）')")
# 表2 南灣
sub("ok(gs_[1:] == ['121.94', '24.85'], '表2 龜山島列：%s' % gs_)",
    "ok(gs_[1:3] == ['121.94', '24.85'], '表2 龜山島列：%s' % gs_)\n"
    "ok([r_[1:3] for r_ in rows2_] == [['121.92', '25.10'], ['121.94', '24.85'], ['120.45', '24.75'], ['120.77', '21.92'], ['119.95', '26.20']], '表2 五列座標（南灣已改 120.77、21.92）：%s' % rows2_)\n"
    "ok(['%.2f' % round(x_, 2) for x_ in (121.9223889, 25.0964444, 121.939148, 24.850326, 120.45, 24.75, 120.770675, 21.92041, 119.95, 26.2)] == ['121.92', '25.10', '121.94', '24.85', '120.45', '24.75', '120.77', '21.92', '119.95', '26.20'], '表2 與 run09 release config 錨點進位 0.01° 一致')")
# 圖檔
a = s.index("ok(imgs_.get('word/media/image1.png')")
b = s.index('\n', a)
s = s[:a] + '''import hashlib as _h
import json as _j3
_sha = {n_: _h.sha256(z.read(n_)).hexdigest() for n_ in z.namelist() if n_.startswith('word/media/')}
_ref = {}
for _l in open(os.path.join(HERE, 'figs_new_v2', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_new_v3', 'sha203.txt'), encoding='utf-8'):  # v10：圖 1 改 v3
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_new_v4', 'sha203.txt'), encoding='utf-8'):  # v11：圖 1 改 v4（長寬依期中報告圖 2-17）
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_arrival_zh_v6', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
_ref6 = dict(_ref)   # v9–v12 之圖 3–7（figs_arrival_zh_v6），供新舊對照
for _l in open(os.path.join(HERE, 'figs_new_v5', 'sha203.txt'), encoding='utf-8'):  # v13（WP-74）：圖 1、圖 2 改 v5（8 pt）
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_arrival_zh_v7', 'sha203.txt'), encoding='utf-8'):  # v13（WP-74）：圖 3–7 改 figs_arrival_zh_v7（8 pt）
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
_rels = dict(re.findall(r'Id="(rId\\d+)"[^>]*?Target="(media/[^"]+)"', z.read('word/_rels/document.xml.rels').decode('utf8')))
_rels.update({a_: b_ for b_, a_ in re.findall(r'Target="(media/[^"]+)"[^>]*?Id="(rId\\d+)"', z.read('word/_rels/document.xml.rels').decode('utf8'))})
_docimg = ['word/' + _rels[r_] for r_ in re.findall(r'r:embed="(rId\\d+)"', xml.decode('utf8') if isinstance(xml, bytes) else xml)]
ok(len(_docimg) == 7 and len(set(_docimg)) == 7, '文件內依序引用 7 個不同圖檔：%s' % _docimg)
ok(_sha.get(_docimg[0]) == _ref['343_run09v5_fig1_circle.png'] and _sha.get(_docimg[1]) == _ref['343_run09v5_fig2_circle.png'], '圖1、圖2 圖檔 sha256 等於 .203 重繪輸出 figs_new_v5（圖內文字 8 pt、TRAPs 框線與刻度，v13；長寬依期中報告圖 2-17 之 v4 版面不變）')
ok(_ref['343_run09v3_fig2_circle.png'] == _ref['343_run09v2_fig2_circle.png'] and _ref['343_run09v4_fig2_circle.png'] == _ref['343_run09v2_fig2_circle.png'], 'v3、v4 程式重畫之圖 2 與 v2 逐位元組相同（圖 2 未受圖 1 改版影響）')
_m3 = _j3.load(open(os.path.join(HERE, 'figs_new_v5', '343_run09v5_figs_circle_manifest.json'), encoding='utf-8'))   # v13：與 v4 版面相同
ok(_m3['fig1_extent_lon0_lon1_lat0_lat1'] == [119.0, 123.0, 20.0, 27.0], '圖1 經緯度範圍（manifest）為 119–123°E、20–27°N，與期中報告圖 2-17 相同：%s' % _m3['fig1_extent_lon0_lon1_lat0_lat1'])
_src3 = open(os.path.join(HERE, 'scripts_fig', 'make_343_figs_run09.py'), encoding='utf-8').read()
ok('ax.set_ylim(FIG1_EXTENT[2], FIG1_EXTENT[3])' in _src3 and 'FIG1_EXTENT = (119.0, 123.0, 20.0, 27.0)' in _src3 and 'set_yticks([20, 21, 22, 23, 24, 25, 26, 27])' in _src3, '繪圖程式圖 1 ylim 取 20–27，緯度刻度 20–27')
import struct as _st3
_ps = z.read(_docimg[0])[16:24]
ok(_ps == _st3.pack('>II', 2298, 3779), '圖1 像素尺寸 2298x3779（v11；v10 為 2484x3408）')
# v11：圖 1 長寬依期中報告圖 2-17（量測：整張嵌入 3.7500x6.2067 in；axes 框 3.370x5.893 in）
_xd = z.read('word/document.xml').decode('utf8')
_ext = [tuple(int(v_) for v_ in m_) for m_ in re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', _xd)][0]
_aext = [tuple(int(v_) for v_ in m_) for m_ in re.findall(r'<a:ext cx="(\d+)" cy="(\d+)"/>', _xd)][0]
_disp = (_ext[0] / 914400.0, _ext[1] / 914400.0)
ok(_ext == _aext, '圖1 wp:extent 與 a:ext 相同：%s' % (_ext,))
ok(abs(_disp[0] - 2298 / 600.0) < 0.01 and abs(_disp[1] - 3779 / 600.0) < 0.01, '圖1 嵌入顯示尺寸等於圖檔 600 dpi 原尺寸（不縮放，字級不變）3.830x6.298 in，容差 0.01 in：%.4f x %.4f in' % _disp)
_REF_FIG, _REF_AX = (3.7500, 6.2067), (3.370, 5.893)
ok(abs(_disp[0] / _REF_FIG[0] - 1) < 0.03 and abs(_disp[1] / _REF_FIG[1] - 1) < 0.03, '圖1 嵌入尺寸與期中報告圖 2-17（3.750x6.207 in）相差 3%% 內（字級不縮放所致，不可能更近）：寬 %+.2f%%、高 %+.2f%%' % ((_disp[0] / _REF_FIG[0] - 1) * 100, (_disp[1] / _REF_FIG[1] - 1) * 100))
_axp = _m3['fig1_axes_pt_left_top_right_bottom']
_cvp = _m3['fig1_canvas_pt']
_axin = ((_axp[2] - _axp[0]) / 72.0, (_axp[3] - _axp[1]) / 72.0)
ok(abs(_axin[0] / _REF_AX[0] - 1) < 0.01 and abs(_axin[1] / _REF_AX[1] - 1) < 0.01, '圖1 axes 尺寸（manifest）與期中報告圖 2-17 axes 框（3.370x5.893 in）相差 1%% 內：%.4fx%.4f in，寬 %+.2f%%、高 %+.2f%%' % (_axin[0], _axin[1], (_axin[0] / _REF_AX[0] - 1) * 100, (_axin[1] / _REF_AX[1] - 1) * 100))
_axdisp = (_axin[0] * _disp[0] / (_cvp[0] / 72.0), _axin[1] * _disp[1] / (_cvp[1] / 72.0))
ok(abs(_axdisp[0] / _REF_AX[0] - 1) < 0.01 and abs(_axdisp[1] / _REF_AX[1] - 1) < 0.01, '圖1 axes 在頁面上的實際尺寸（由嵌入尺寸換算）與參考相差 1%% 內：%.4fx%.4f in' % _axdisp)
ok(abs(_axin[0] / _axin[1] - 4.0 / 7.0) < 0.002, '圖1 axes 寬高比 4:7（每度同長，與參考相同）：%.4f' % (_axin[0] / _axin[1]))
ok(min(_m3['fig1_text_clearance_pt_left_top_right_bottom']) >= 0.0, '圖1 文字與刻度外框皆在畫布內（manifest 實測餘裕 pt：%s）' % _m3['fig1_text_clearance_pt_left_top_right_bottom'])
ok(abs(_cvp[0] / 72.0 - _disp[0]) < 0.005 and abs(_cvp[1] / 72.0 - _disp[1]) < 0.005, '圖1 畫布 pt（manifest）與嵌入尺寸一致（無縮放）')
_order = ['wp61_%s_overview_S_notext.png' % _s for _s in ('gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang')]
ok(len(_sha) == 7 and all(_sha.get(_docimg[2 + i_]) == _ref[n_] for i_, n_ in enumerate(_order)) and all(_ref[n_] != _ref6[n_] for n_ in _order), '圖3–圖7 圖檔 sha256 等於 figs_arrival_zh_v7（.203 figures-zh-v6 重繪輸出，圖內文字 8 pt，v13）且已不同於 v6（v9–v12 之 9 pt 版）；媒體檔共 7 個')
import struct as _st0
ok(all(z.read(_docimg[2 + i_])[16:24] == _st0.pack('>II', 1880, 1753) for i_ in range(5)), '圖3–圖7 像素尺寸 1880x1753（與 v3–v12 相同；v13 畫布高固定為前版像素，嵌入尺寸不變）')
try:
    from PIL import Image as _Im
    import numpy as _np
    _al = {}
    _a7b = _j3.load(open(os.path.join(HERE, 'figs_arrival_zh_v7', 'manifest.json'), encoding='utf-8'))['figures']
    for i_, n_ in enumerate(_order):
        _s_ = n_.split('_')[1]
        _im = _Im.open(__import__('io').BytesIO(z.read(_docimg[2 + i_])))
        _g = _np.asarray(_im.convert('L')).astype(int)
        _bx = _a7b['%s/%s' % (_s_, n_)]['axes_boxes_new']   # [速度, 有效波高, 潮汐強度]，英寸（左、下、寬、高）
        _row = int(round((5.845 - (_bx[0][1] + 0.17 * _bx[0][3])) * 300))   # 流速面板內兩個刻度之間的一列（不穿過刻度線）
        _sp = 200 + int(_np.where(_g[_row, 200:240] < 128)[0].min())
        _lg = int(_np.where((_g[1521:1753, :900] < 128).any(0))[0].min())
        _rg = 900 + int(_np.where((_g[1521:1753, 900:] < 200).any(0))[0].min())
        _al[n_] = (_sp, _lg, _rg)
    ok(all(abs(v_[1] - v_[0]) <= 1 for v_ in _al.values()), '圖3–圖7 左欄最左側圖例標記墨跡之 x 與主圖左脊線之 x 相差不超過 1 像素（v9 規則，v13 重驗；各圖 脊線 x／標記 x）：%s' % {k_: v_[:2] for k_, v_ in _al.items()})
    ok(all(abs(v_[2] - 988) <= 2 for v_ in _al.values()), '圖3–圖7 右欄圖例墨跡起點在主圖水平中央（繪圖區 x 222–1754 之中點 988，容差 2 像素）（v9 規則，v13 重驗）：%s' % {k_: v_[2] for k_, v_ in _al.items()})
except ImportError:
    ok(True, '（略過像素檢查：無 PIL／numpy）')
import struct as _st
_ext = [(int(m_.group(1)), int(m_.group(2))) for m_ in re.finditer(r'<wp:extent cx="(\\d+)" cy="(\\d+)"', xml.decode('utf8') if isinstance(xml, bytes) else xml)]
ok(len(_ext) == 7, '文件內圖片 7 張')
ok(abs(_ext[0][0] / 914400 - 3.830) < 0.01 and abs(_ext[0][1] / 914400 - 6.298) < 0.01 and abs(_ext[1][0] / 914400 - 6.267) < 0.01 and abs(_ext[1][1] / 914400 - 8.92) < 0.02, '圖1 嵌入尺寸 3.83x6.30 in（v11，依期中報告圖 2-17 重畫；v10 為 4.13x5.67）、圖2 仍沿用 run08 之 6.27x8.92 in')
for _i in range(2, 7):
    _w, _h = _st.unpack('>II', z.read(_docimg[_i])[16:24])
    _sc = (_ext[_i][0] / 914400) / (_w / 300.0)
    ok(abs(_ext[_i][0] / 914400 - 6.267) < 0.01 and 5.5 <= _ext[_i][1] / 914400 <= 6.01 and _sc >= 0.999, '圖%d 寬 %.3f in、高 %.3f in，嵌入縮放 %.4f（圖內 8 pt 字不縮小）' % (_i + 1, _ext[_i][0] / 914400, _ext[_i][1] / 914400, _sc))
ok(len(re.findall(r'Type="[^"]*/image"', z.read('word/_rels/document.xml.rels').decode('utf8'))) == 7, '圖片關聯 7 個（無放大圖殘留）')
# ---------- v13（WP-74）：圖 1–圖 7 圖內文字 8 pt、Noto Sans CJK TC，與 TRAPs 報告圖規格一致
_m5 = _j3.load(open(os.path.join(HERE, 'figs_new_v5', '343_run09v5_figs_circle_manifest.json'), encoding='utf-8'))
_cmp5 = _j3.load(open(os.path.join(HERE, 'figs_new_v5', 'compare_fig12.json'), encoding='utf-8'))
_a7 = _j3.load(open(os.path.join(HERE, 'figs_arrival_zh_v7', 'manifest.json'), encoding='utf-8'))
_m4 = _j3.load(open(os.path.join(HERE, 'figs_new_v4', '343_run09v4_figs_circle_manifest.json'), encoding='utf-8'))
_FAM = 'Noto Sans CJK TC'
_au = _m5['figures_audit']
_k1, _k2 = '343_run09v5_fig1_circle', '343_run09v5_fig2_circle'
_fa7 = _a7['figures']
ok(all(_au[k_]['text_summary']['sizes_pt'] == [8.0] and _au[k_]['text_summary']['font_families'] == [_FAM] and _au[k_]['text_summary']['n_texts'] > 0 and _au[k_]['text_summary']['mathtext_texts'] == [] for k_ in (_k1, _k2)), '圖1、圖2 圖內全部文字（%d、%d 個：刻度標籤、軸標題、圖例、(a)–(e)）字級皆 8 pt、字型皆 Noto Sans CJK TC（畫完後逐一掃描文字物件）' % (_au[_k1]['text_summary']['n_texts'], _au[_k2]['text_summary']['n_texts']))
ok(len(_fa7) == 5 and all(v_['text_summary_new']['sizes_pt'] == [8.0] and v_['text_summary_new']['font_families'] == [_FAM] and v_['text_summary_new']['n_texts'] > 0 for v_ in _fa7.values()), '圖3–圖7 圖內全部文字（各 %s 個）字級皆 8 pt、字型皆 Noto Sans CJK TC（畫完後逐一掃描文字物件）' % sorted({v_['text_summary_new']['n_texts'] for v_ in _fa7.values()}))
ok(all(v_['svg_text_new']['sizes_px_eq_pt'] == [8.0] for v_ in _fa7.values()), '圖3–圖7 SVG 內 text 元素字級皆 8（SVG 單位 px＝pt）')
ok(all(v_['text_summary_prev']['sizes_pt'] == [9.0] for v_ in _fa7.values()), '對照：改前（前版程式重畫＝現行嵌入圖）圖3–圖7 圖內字級為 9 pt')
def _dpi_png(b_):
    i_ = b_.find(b'pHYs')
    return _st.unpack('>I', b_[i_ + 4:i_ + 8])[0] * 0.0254
_pr = []
for i_ in range(7):
    b_ = z.read(_docimg[i_])
    _pr.append(round(8.0 * (_ext[i_][0] / 914400.0) * _dpi_png(b_) / _st.unpack('>II', b_[16:24])[0], 3))
ok(all(abs(p_ - 8.0) < 0.01 for p_ in _pr), '圖1–圖7 印出字級（圖內 8 pt x 嵌入顯示寬 / 畫布寬，畫布寬由像素與 dpi 換算）皆為 8.00 pt（不小於 8 pt）：%s' % _pr)
ok(_m5['style'] == {'spine_lw': 1.1, 'spine_color': '#222222', 'tick_len': 3.5, 'tick_w': 0.8, 'tick_pad': 2.0, 'label_pad': 2.0, 'font': _FAM}, '圖1、圖2 框線 1.1 pt #222222、刻度朝外長 3.5 寬 0.8 pad 2、labelpad 2（TRAPs 慣例）')
_srcA = open(os.path.join(HERE, 'scripts_fig', 'wp74_343', 'arrival_figs.py'), encoding='utf-8').read()
ok('FS_OV = 8.0' in _srcA and 'TRAPS_SPINE_LW, TRAPS_SPINE_COLOR = 1.1, "#222222"' in _srcA and 'TRAPS_TICK_LEN, TRAPS_TICK_W, TRAPS_TICK_PAD = 3.5, 0.8, 2.0' in _srcA, '圖3–圖7 繪圖程式：圖內字級 FS_OV＝8.0、左下框線 1.1 pt #222222、刻度長 3.5 寬 0.8 pad 2（TRAPs 慣例）')
ok(all(_cmp5[k_]['prev_equals_control'] and _cmp5[k_]['data_fingerprint_equal'] for k_ in ('fig1_circle', 'fig2_circle')), '圖1、圖2：前版（v4）程式重畫＝現行嵌入圖（逐位元組相同，可重現）；新舊兩版存檔前之資料指紋（座標範圍、刻度與標籤、軸標題、線條／多邊形／散點資料座標、圖例與註解文字）相同＝內容不變')
ok(len(_fa7) == 5 and all(v_['prev_equals_control'] and v_['data_fingerprint_equal'] for v_ in _fa7.values()), '圖3–圖7：前版程式重畫＝現行嵌入圖（figs_arrival_zh_v6，逐位元組相同）；新舊資料指紋（軸範圍、刻度標籤、軸標題、各線資料、標記位置、圖例標籤、各季標頭文字）相同＝內容不變')
ok(_m5['fig1_axes_pt_left_top_right_bottom'] == _m4['fig1_axes_pt_left_top_right_bottom'] and _m5['fig1_canvas_pt'] == _m4['fig1_canvas_pt'] and _m5['dpi'] == _m4['dpi'] == 600 and _m5['fig1_extent_lon0_lon1_lat0_lat1'] == _m4['fig1_extent_lon0_lon1_lat0_lat1'], '圖1 畫布與 axes 尺寸、範圍同 v4（版面不變，只改字級、框線與刻度）')
ok(all(v_['png_px_new'] == v_['png_px_prev'] == [1880, 1753] for v_ in _fa7.values()), '圖3–圖7 像素尺寸與前版相同（1880x1753），嵌入尺寸不變')
ok(all(min(v_['clearance_in_new'].values()) > 0 for v_ in _fa7.values()) and all(min(_au[k_]['clearance_pt_left_top_right_bottom']) > 0 for k_ in (_k1, _k2)), '圖1–圖7 文字、刻度與圖例外框皆在畫布內（餘裕皆為正）')
_bak = os.path.join(os.path.dirname(os.path.abspath(DOCX)), '3-4-3_run09.docx.bak08-20261022')
if os.path.exists(_bak):
    zb_ = zipfile.ZipFile(_bak)
    _imgn = ['word/media/image%d.png' % i_ for i_ in range(1, 8)]
    _diff = [n_ for n_ in z.namelist() if z.read(n_) != zb_.read(n_)]
    ok(z.namelist() == zb_.namelist() and sorted(_diff) == _imgn, '與換圖前（.bak08-20261022，v12 版）相比：只有 word/media/image1–7.png 不同，其餘 %d 個 zip 成員（含 document.xml：顯示尺寸、文字、rsid）逐位元組相同' % (len(z.namelist()) - 7))
    ok(all(z.read(n_)[16:24] == zb_.read(n_)[16:24] for n_ in _imgn), '7 張圖換圖前後像素尺寸相同')
''' + s[b:]
sub("ok('12.5 km' in fulltext and '10 km' not in fulltext.replace('12.5 km', ''), '內文有「12.5 km」且無「10 km」')",
    "ok('12.5 km' not in fulltext and fulltext.count('10 km') == 2 and '12.5km' not in fulltext, '內文無「12.5 km」，「10 km」出現 2 次（受體段落兩處）；例外：附錄「+12.5%」為球體檢驗誤差，非抽樣半徑')")
sub("ok('抽樣範圍為以各研究地點錨點為圓心、半徑 12.5 km 之圓' in fulltext, '受體段寫入半徑 12.5 km 之圓')",
    "ok('抽樣範圍為以各研究地點錨點為圓心、半徑 10 km 之圓' in fulltext, '受體段寫入半徑 10 km 之圓')")
sub("ok(any(t_.startswith('圖2 各研究地點受體抽樣範圍（虛線圓）與 5 個受體水平位置（空心圓）：') for t_ in paras) and any(t_.startswith('圖1 ') and '圓旁字母' in t_ for t_ in paras), '圖1、圖2 圖題')",
    "ok(any(t_.startswith('圖2 各研究地點受體抽樣範圍（虛線圓）與 5 個受體水平位置（空心圓）：') for t_ in paras) and any(t_.startswith('圖1 ') and '圓旁字母' in t_ for t_ in paras), '圖1、圖2 圖題')\n"
    "ok(not any('12.5' in t_ for t_ in paras if t_.startswith('圖')) and not any('12.5' in d_.decode('utf8') for d_ in dsc), '圖說與替代文字無 12.5')")
# 新增 run09 檢查
NEW = r'''
# ---------------- run09：新增與變更項目 ----------------
# 1. 受體段落：新方法關鍵句
rec_ = [t_ for t_ in paras if t_.startswith('共由「5 個水平位置')]
ok(len(rec_) == 1, '受體段落唯一')
rec_ = rec_[0]
for k_ in ['半徑 10 km 之圓',
           '分別將半徑 10 km 範圍內之候選海域等分為面積相同之 5 個區間',
           '依方位及距研究地點錨點之距離',
           '使每一方位區間與每一距離區間各恰有 1 個受體水平位置',
           '並於所屬區塊（方位區間與距離區間之交會處）內隨機抽取',
           '並於同一區塊內依隨機順序遞補',
           '候選面須為有效之三角形網格且非陸地，並於全部 50 個到達時間均維持濕潤（不因退潮而乾出）',
           '抽中後另須通過流場與波浪資料之逐時空間支援檢查，未通過者剔除']:
    ok(k_ in rec_, '受體段關鍵句：' + k_)
for k_ in ['無放回', '均勻隨機抽取 5 個位置', '依隨機抽樣之順序遞補', '12.5', '關注海域位於島的西側', '位於島的西側', '故西半部', '西半部', '東半部', '配置 3 個']:
    ok(k_ not in rec_, '受體段不得再有舊方法用語：' + k_)
ok('關注海域位於島的西側' not in fulltext and '龜山島之西半部' not in fulltext, '全文無「關注海域位於島的西側」與「龜山島之西半部」（使用者已於 Word 刪除該句，維持刪除）')
# 2. 到達時間段落
arr_ = [t_ for t_ in paras if t_.startswith('50 個到達時間由 48 個分層時刻')]
ok(len(arr_) == 1, '到達時間段落唯一')
arr_ = arr_[0]
for k_ in ['涵蓋四季，每季 12 個分層時刻，大潮、小潮各 6 個',
           '以各時刻前後共 25 小時（約一個潮日）之潮汐強度（該時段水位起伏之大小）與該季之中位數比較，高於中位數者為大潮，低於者為小潮',
           '挑選時排除流場資料於日間銜接處水位不連續之時刻',
           '每格選出 2 個時刻', '共 24 格']:
    ok(k_ in arr_, '到達時間段關鍵句：' + k_)
ok('各月 5 至 9 個' not in arr_ and '各月' not in arr_ and '全年十二個月' not in arr_ and not re.search(r'5\s*至\s*9', arr_), '到達時間段不得再有「各月 5 至 9 個」「全年十二個月」')
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', arr_)), '到達時間段無英文字母')
# 2b. 溯源停止邊界段（方法 B，v12）：抵達海岸亦停止追蹤；逐地點穿越比例與海岸來源比例（數字自 outcome_by_site.csv 之顆數獨立計算，不手抄）
import csv as _csv, math as _math
from decimal import Decimal as _D, ROUND_HALF_UP as _RHU
_t = {}
for _r in _csv.DictReader(open(os.path.join(HERE, '..', 'src_coastB', 'coastB-tables', 'outcome_by_site.csv'), encoding='utf-8-sig')):
    _t[_r['site']] = _r
ok(set(_t) == {'gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang'}, '停止邊界：csv 五地點齊全')
_p2 = lambda n_, d_: str((_D(int(n_)) * 100 / _D(int(d_))).quantize(_D('0.01'), rounding=_RHU)) + '%'
ok(all(int(_t[k_]['total']) == 100000 and int(_t[k_]['crossed']) + int(_t[k_]['coast']) + int(_t[k_]['touch_bottom']) + int(_t[k_]['max_backtrack']) + int(_t[k_]['other']) == 100000 for k_ in _t), '停止邊界：五地點穿越＋海岸來源＋觸底＋達回溯上限＋其他＝100,000')
ok(all(int(_t[k_]['max_backtrack']) == 0 and int(_t[k_]['other']) == 0 for k_ in _t) and [k_ for k_ in _t if int(_t[k_]['touch_bottom'])] == ['nanwan'] and int(_t['nanwan']['touch_bottom']) == 1, '停止邊界：無達回溯上限與其他；僅南灣 1 顆觸底')
_dmax = '%.1f' % (_math.ceil(max(float(_t[k_]['max_duration_days']) for k_ in _t) * 10 - 1e-9) / 10)
_pp = {k_: (_p2(_t[k_]['crossed'], _t[k_]['total']), _p2(_t[k_]['coast'], _t[k_]['total'])) for k_ in _t}
stop_ = [t_ for t_ in paras if t_.startswith('4、溯源停止邊界')]
ok(len(stop_) == 1, '溯源停止邊界段唯一')
stop_ = stop_[0]
_head = '4、溯源停止邊界（Boundary condition）：逆向追蹤將持續演算，直至虛擬質點完全離開本計畫所設定之「關注區域（Domain of interest）」邊界或抵達海岸為止，藉此精確鎖定海廢移入該海域之潛在外部來源與邊界通量路徑。'
_pre = '本研究之關注區域即圖 1 之四區流場範圍，虛擬質點穿越流場範圍邊界或抵達海岸（視為找到來源，稱為海岸來源）即停止追蹤。'
_want = ('回溯期間上限為 180 天，各研究地點皆於 %s 天內停止追蹤，穿越比例與海岸來源比例依序為貢寮 %s、%s，龜山島 %s、%s，新竹外海 %s、%s，南灣 %s、%s，連江 %s、%s。'
         % ((_dmax,) + _pp['gongliao'] + _pp['guishan'] + _pp['hsinchu'] + _pp['nanwan'] + _pp['lienchiang']))
ok(stop_ == _head + _pre + _want, '停止邊界段全文＝首句（含或抵達海岸）＋停止追蹤條件句＋回溯上限與逐地點比例句（比例自 csv 顆數重算）：' + _want)
ok(stop_.endswith(_want) and stop_.count(_pre) == 1 and stop_.startswith(_head), '停止邊界段：首句、條件句、末句各一次且依序')
ok(_pp == {'gongliao': ('73.40%', '26.60%'), 'guishan': ('78.35%', '21.65%'), 'hsinchu': ('97.77%', '2.23%'), 'nanwan': ('68.91%', '31.09%'), 'lienchiang': ('56.06%', '43.94%')} and _dmax == '33.4', 'csv 值：穿越／海岸來源比例 貢寮 73.40／26.60、龜山島 78.35／21.65、新竹外海 97.77／2.23、南灣 68.91／31.09、連江 56.06／43.94；最長 33.4 天')
# 與成果報告 md 一致（同一組數字）
_md = open(os.path.join(HERE, '..', '..', '執行成果_run09_含斯托克斯漂流.md'), encoding='utf-8').read()
ok(all(('穿越比例為 %s，海岸來源比例為 %s' % _pp[k_]) in _md for k_ in ('guishan', 'hsinchu', 'nanwan', 'lienchiang')) and ('貢寮之虛擬質點穿越流場範圍邊界者占 %s' % _pp['gongliao'][0]) in _md and ('抵達海岸者占 %s' % _pp['gongliao'][1]) in _md and ('33.4 天內') in _md, '停止邊界段各地點比例與成果報告 md 相同（五地點穿越比例與海岸來源比例、33.4 天）')
ok('開放邊界' not in fulltext and '開放邊界' not in stop_ and '流場範圍邊界' in stop_, '停止邊界段用「流場範圍邊界」，全文無「開放邊界」')
ok(stop_.count('海岸來源') == 2 and '穿越比例' in stop_, '停止邊界段用語：海岸來源、穿越比例（與成果報告一致）')
ok('180 天' in stop_ and '180 天' not in re.sub(re.escape('回溯期間上限為 180 天'), '', fulltext), '全文「180 天」僅出現於停止邊界段「回溯期間上限為 180 天」')
ok('離開流場範圍邊界' not in fulltext and '99.999' not in fulltext and '95.857' not in fulltext and '100%' not in fulltext, '全文無舊停止條件句、無舊比例（離開流場範圍邊界、99.999、95.857、100%）')
ok(not re.search(r'較|高於|低於|相比|最|比(?!例)', _want), '停止邊界末句無比較語氣')
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', _want)), '停止邊界末句無英文字母')
ok(not re.search(r'反射|定義 ?[AD]|海岸定義|逐時抽樣|抽樣誤差', fulltext), '全文不提反射式海岸、海岸定義 A／D、逐時抽樣誤差')

# 2c. 圖 3–圖 7 圖內文字（由 SVG 文字取出）：只有專有名詞或符號可有英文（白名單 Hs、m、m/s）、用語依服務建議書（有效波高）、Y 軸標題在 Y 軸位置
import glob as _g, json as _j
_cnt = _j.load(open(os.path.join(HERE, 'src', 'arrival_counts.json'), encoding='utf-8'))
_SITES = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
_svg_all = {}
for _s, _zh in _SITES:
    _f = os.path.join(HERE, 'figs_arrival_zh_v7', _s, 'wp61_%s_overview_S_notext.svg' % _s)
    _sv = open(_f, encoding='utf-8').read()
    _els = []
    for _m in re.finditer(r'<text([^>]*)>(.*?)</text>', _sv, flags=re.S):
        _tx = re.sub(r'<[^>]+>', '', _m.group(2)).strip()
        if _tx:
            _els.append((_tx, _m.group(1)))
    _tx = [x_[0] for x_ in _els]
    _svg_all[_s] = _tx
    _c = _cnt[_s]
    ok(_c['total'] == 50 and sum(_c['class'].values()) == 48 and sum(_c['phase'].values()) == 48 and sum(_c['event'].values()) == 2, '%s：manifest 計數 50 = 48 個分層時刻（大潮 + 小潮；三潮相）+ 2 個事件時刻' % _zh)
    ok(not [x_ for x_ in _tx if re.search(r'[（(]\s*\d+\s*[）)]', x_)], '%s 圖內文字無括號數字（圖例不含各類數量）' % _zh)
    for _w in ['大潮', '小潮', '漲潮最快', '退潮最快', '近平潮', '高浪事件', '強流事件', '季內中位數', '全年第 99 百分位',
               '冬季', '春季', '夏季', '秋季', '日期（年/月/日）']:
        ok(_tx.count(_w) >= 1, '%s 圖內文字含「%s」' % (_zh, _w))
    _dt = [x_ for x_ in _tx if re.fullmatch(r'\d{4}/\d{2}/\d{2}', x_)]
    ok(_dt == ['2025/01/01', '2025/03/01', '2025/05/01', '2025/07/01', '2025/09/01', '2025/11/01', '2026/01/01'], '%s 日期刻度每兩個月：%s' % (_zh, _dt))
    _sn = [x_ for x_ in _tx if x_.startswith('大潮 ') and '／小潮 ' in x_]
    _exp = ['大潮 %d／小潮 %d' % (_c['season_class'][k_ + '|spring_proxy'], _c['season_class'][k_ + '|neap_proxy']) for k_ in ('DJF', 'MAM', 'JJA', 'SON')]
    ok(_sn == _exp, '%s 各季色塊標示大潮／小潮數與 manifest 一致：%s' % (_zh, _sn))
    # 英文：僅白名單
    _lat = [x_ for x_ in _tx if re.search(r'[A-Za-z]', re.sub(r'\bHs\b|\bm/s\b|\bm\b', '', x_))]
    ok(not _lat, '%s 圖內文字無英文（白名單：Hs、m、m/s）：%s' % (_zh, _lat))
    ok(all(not re.search(r'[A-Za-z]', re.sub(r'\bHs\b|\bm/s\b|\bm\b', '', x_)) for x_ in _tx) and not any(w_ in ' '.join(_tx) for w_ in ('UTC', 'Date', 'Spring', 'Neap', 'Winter', 'Summer', 'Autumn', 'Fastest', 'slack', 'median', 'percentile', 'p99', 'Tidal', 'Current', 'wave height')), '%s 無 UTC、Date 及舊英文詞' % _zh)
    ok(not any(w_ in ' '.join(_tx) for w_ in ('顯著波高', '示性波高', '有義波高', '世界時', '半徑', '25 小時海面高度', '均方根')), '%s 無「顯著波高／示性波高／世界時」，定義文字不在圖上' % _zh)
    ok(sum(1 for x_ in _tx if '有效波高' in x_) == 1 and '有效波高 Hs' in _tx, '%s 圖內「有效波高 Hs」（服務建議書用詞）' % _zh)
    ok(set(re.findall(r'font-size: ([0-9.]+)px', _sv)) == {'8'}, '%s 圖內文字皆 8 px（SVG 單位 px＝pt；嵌入縮放 1.0 即印出 8 pt）' % _zh)
    # Y 軸標題：垂直（rotate(-90)）、位於刻度字左側、其垂直位置在對應面板範圍內
    _ytick = [(float(re.search(r'y="([0-9.]+)"', a_).group(1)), x_) for x_, a_ in _els if re.fullmatch(r'\d+(\.\d+)?', x_) and 'text-anchor: end' in a_]
    _grp, _prev = [], None
    for _y, _v in _ytick:
        if _prev is None or float(_v) < _prev:
            _grp.append([])
        _grp[-1].append(_y); _prev = float(_v)
    ok(len(_grp) == 3, '%s 三個面板之縱軸刻度組：%d' % (_zh, len(_grp)))
    _yl = {}
    for _x, _a in _els:
        _mm = re.search(r'translate\(([0-9.]+) ([0-9.]+)\) rotate\(-90\)', _a)
        if _mm:
            _yl.setdefault(_x, []).append((float(_mm.group(1)), float(_mm.group(2))))
    ok(sorted(_yl) == sorted(['潮汐強度', '（m）', '有效波高 Hs', '流速', '（m/s）']) and len(_yl['（m）']) == 2, '%s Y 軸垂直標題文字：%s' % (_zh, sorted(_yl)))
    _rng = [(min(g_), max(g_)) for g_ in _grp]  # 由下而上：流速、有效波高、潮汐強度
    _pos_ok = all(xx_ < 53.0 for v_ in _yl.values() for xx_, _ in v_)
    _in = lambda yy_, r_: r_[0] - 8 <= yy_ <= r_[1] + 8
    _pos_ok = _pos_ok and _in(_yl['潮汐強度'][0][1], _rng[2]) and _in(_yl['有效波高 Hs'][0][1], _rng[1]) and _in(_yl['流速'][0][1], _rng[0])
    _pos_ok = _pos_ok and all(_in(yy_, _rng[2]) or _in(yy_, _rng[1]) or _in(yy_, _rng[0]) for v_ in _yl.values() for _, yy_ in v_)
    ok(_pos_ok, '%s Y 軸標題在 Y 軸位置（x 在繪圖區左側刻度字外、垂直落於各自面板內）：%s' % (_zh, _yl))
    ok(not any('潮汐強度' in x_ and '（' in x_ and 'm）' in x_ and len(x_) > 8 for x_ in _tx) and not any(w_ in ' '.join(_tx) for w_ in ('Significant', 'Tidal strength')), '%s 面板上方無長標題' % _zh)
# 兩條線樣式：由繪圖程式常數與圖例確認（黑色粗長虛線 vs 深紅點線）
_src = open(os.path.join(HERE, 'scripts_fig', 'wp74_343', 'arrival_figs.py'), encoding='utf-8').read()
ok('LINE_MEDIAN = dict(color="black", lw=1.7, ls=(0, (7, 3)))' in _src and 'LINE_P99 = dict(color="#b2182b", lw=1.5, ls=(0, (1, 1.6)))' in _src, '季內中位數為黑色粗長虛線、全年第 99 百分位為深紅點線（程式常數）')
for _s, _zh in _SITES:
    _sv = open(os.path.join(HERE, 'figs_arrival_zh_v7', _s, 'wp61_%s_overview_S_notext.svg' % _s), encoding='utf-8').read()
    ok('stroke-dasharray: 11.9,5.1' in _sv.replace('stroke-dasharray: 11.900000,5.100000', 'stroke-dasharray: 11.9,5.1') or re.search(r'stroke: #000000[^"]*stroke-dasharray: 11\.9', _sv) is not None, '%s 圖中有黑色長虛線（中位數）' % _zh)
    ok(re.search(r'stroke: #b2182b', _sv) is not None, '%s 圖中有深紅線（全年第 99 百分位）' % _zh)
# 圖 1、圖 2 圖例（沿用 v2）

_man = _j.load(open(os.path.join(HERE, 'figs_new_v2', '343_run09v2_figs_circle_manifest.json'), encoding='utf-8'))
ok(_man['circle_legend_label'] == '抽樣範圍' and float(_man['radius_km']) == 10.0, '圖1 圖例文字為「抽樣範圍」，半徑 10 km 僅於本文與表')
_src = open(os.path.join(HERE, 'scripts_fig', 'make_343_figs_run09.py'), encoding='utf-8').read()
_m = re.search(r'def _circle_label.*?return "([^"]*)"', _src, flags=re.S)
ok(_m is not None and _m.group(1) == '抽樣範圍', '繪圖程式 _circle_label 回傳「抽樣範圍」（圖檔文字已轉為路徑，無法由 SVG 讀字；半徑只寫在本文與表 2）')

# 3. 回溯上限：全文不得出現 90 天、2160 小時（正文原無此二詞；180 天亦不在 3-4-3 內）
ok(not re.search(r'90\s*天|2160|2,160|九十天|三個月', fulltext), '全文無「90 天」「2160 小時」')
ok(re.search(r'12\.5\s*km', fulltext) is None, '全文無「12.5 km」')
# 4. 驗證圖：導引句、圖題、編號、引用
guide_ = [t_ for t_ in paras if t_.startswith('以下列出各研究地點之到達時間驗證圖')]
ok(len(guide_) == 1, '導引句唯一')
g_ = guide_[0]
for k_ in ['每個研究地點 1 張全年總覽圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7',
           '圖中潮汐強度即前述各時刻前後 25 小時之水位起伏大小，以海面高度之均方根計算',
           '季內中位數即大潮與小潮之分界；近平潮即最接近漲退潮交替之時刻',
           '2 個事件時刻另標示於有效波高與流速曲線上',
           '同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊']:
    ok(k_ in g_, '導引句：' + k_)
_g6 = ('以下列出各研究地點之到達時間驗證圖，每個研究地點 1 張全年總覽圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7。'
       '圖中潮汐強度即前述各時刻前後 25 小時之水位起伏大小，以海面高度之均方根計算；季內中位數即大潮與小潮之分界；近平潮即最接近漲退潮交替之時刻。'
       '2 個事件時刻另標示於有效波高與流速曲線上。同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊。')
ok(g_ == _g6, '（v6）導引句全文等於精簡版')
ok('橘色標記為大潮' not in fulltext and '圖例括號內為各類時刻數' not in fulltext and '黑色粗虛線' not in fulltext, '（v6）全文無已刪之圖例重複說明（橘色標記為大潮、圖例括號內為各類時刻數、黑色粗虛線）')
_zf = zipfile.ZipFile(DOCX)
_ft = _zf.read('word/footer1.xml').decode('utf-8')
_ftxt = re.sub(r'<[^>]+>', '', re.sub(r'<w:instrText[^>]*>.*?</w:instrText>', '', _ft))
ok('3-' not in _ft and 'PAGE' in _ft and '<w:jc w:val="center"/>' in _ft, '（v6）頁尾無「3-」，仍有 PAGE 欄位且置中')
ok(not re.search(r'[A-Za-z\-]', re.sub(r'\d+', '', _ftxt)) and [n_ for n_ in _zf.namelist() if re.match(r'word/(header|footer)\d*\.xml$', n_)] == ['word/footer1.xml'], '（v6）頁尾僅頁碼欄位結果、無其他頁首頁尾部件')
ok('<w:pgNumType' not in xml.decode('utf-8'), '（v6）節設定無 pgNumType（十進位、由 1 起算）')
ok(not re.search(r'[A-Za-z]', g_) and not re.search(r'可能|或許|大概|似乎|應該|恐怕|推測|也許', g_), '導引句無英文字母、無不確定語氣')
order_ = [t_ for t_ in paras if re.match(r'^圖\d+ ', t_)]
nums_ = [int(re.match(r'^圖(\d+) ', t_).group(1)) for t_ in order_]
ok(nums_ == list(range(1, 8)), '圖號連續 1–7：%s' % nums_)
ok(not re.search('顯著波高|示性波高|有義波高', fulltext) and fulltext.count('有效波高') == 1, '全文無「顯著波高／示性波高／有義波高」，「有效波高」僅於導引句（1 處，依服務建議書）')
ok(not re.search('放大', fulltext) and '逐時刻' not in fulltext, '全文無「放大」「逐時刻」（放大圖與其所有提及已刪）')
want_ = []
for zh_ in ('貢寮', '龜山島', '新竹外海', '南灣', '連江'):
    want_ += ['%s到達時間驗證圖' % zh_]  # 使用者於 Word 手改，圖題刪「（全年總覽）」，以現況為準
ok([t_.split(' ', 1)[1] for t_ in order_[2:]] == want_, '圖3–圖7 圖題含地點名稱且依序為貢寮、龜山島、新竹外海、南灣、連江（僅一張全年總覽圖）')
nonc_ = '\n'.join(t_ for t_ in paras if not re.match(r'^圖\d+ ', t_))
for n_ in range(1, 8):
    ok(re.search(r'圖 %d(?!\d)' % n_, nonc_) is not None, '內文引用 圖 %d' % n_)
ok(re.search(r'(?<![A-Za-z0-9])圖\d', nonc_) is None, '內文引用不得為「圖N」無空格')
# 圖在上、圖題在下（序列檢查）
seq2_ = []
for el_ in body:
    if el_.tag == W + 'p':
        has_ = el_.find('.//' + W + 'drawing') is not None
        tx_ = ptext(el_)
        seq2_.append(('fig' if has_ else ('cap' if re.match(r'^圖\d+ ', tx_) else 'p'), tx_))
fi_ = [i_ for i_, e_ in enumerate(seq2_) if e_[0] == 'fig']
ok(len(fi_) == 7 and all(seq2_[i_ + 1][0] == 'cap' for i_ in fi_), '7 張圖，每張圖下一段皆為圖題（圖在上、圖題在下）')
# 圖1 圖2 引用先於圖；新圖之導引句在圖 3 之前
ok(paras.index(g_) < paras.index(order_[2]), '導引句位於圖3 之前')
ok(paras.index(arr_) < paras.index(g_), '導引句緊接到達時間段落之後')
# 5. 新增文字之禁用字與英文字母
newtext_ = '\n'.join([rec_, arr_, g_, _want] + order_[2:])
for bad_ in ['站', '站點', '粒子', '機率', '扇區', '開放邊界', '本案', 'run0', 'BayTrace', '.py', '.csv', '.json', '.png', 'lbt', 'LBT', 'SCHISM', 'OCM', 'NWW', 'UTC']:
    ok(bad_ not in newtext_, '新增／修改段落不含禁用詞「%s」' % bad_)
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', ''.join([g_] + order_[2:]))), '導引句與圖題無英文字母')
ok(not re.search(r'可能|或許|大概|似乎|應該|恐怕|推測|也許', newtext_), '新增／修改段落無不確定語氣')
# 6. 單位寫法：km 為符號、句中時間寫中文（新增文字內不得出現 h、d 或阿拉伯數字加 hr）
ok(not re.search(r'\d\s*(hr|h\b|d\b|day)', newtext_), '新增文字無英文時間單位')
# 7. 第三輪（v5）：依服務建議書「到達時間」用語，全文不再用「到達時刻」
ok('到達時刻' not in fulltext, '全文無「到達時刻」（用語依建議書「到達時間」）')
ok('實際深度再依各到達時間的潮位與海床高程換算' in fulltext, '深度換算句用「各到達時間的潮位」')
'''
sub("# ---------------- 公式數 ----------------", NEW + "\n# ---------------- 公式數 ----------------")
# 2026-10-22 v12：核對對照使用者於 Word 手改後之現況（正文分配說明段、表 1 簡化短語；以現況為準）
sub("以固定隨機種子之隨機排列決定，不含斯托克斯漂流案例與含斯托克斯漂流案例採用相同之分配。' in fulltext.replace('模擬時，', ''), '正文分配說明段')",
    "以固定隨機種子之隨機排列決定分配。' in fulltext.replace('模擬時，', ''), '正文分配說明段（使用者於 Word 手改，刪「不含斯托克斯漂流案例與含斯托克斯漂流案例採用相同之分配」，以現況為準）')")
sub("short_ = {'不可回收塑膠': '薄膜至帆布片之軟質包材', '廢紙': '吸水紙片至紙箱板', '保麗龍': '吸水或附著之發泡板至箱體側板', '廢漁網漁具': '網線至繩索', '寶特瓶': '瓶身碎片、完整進水瓶、半壓扁至重壓扁瓶身', '竹木': '細竹枝至大型竹木段（水浸飽和）', '可回收塑膠': '硬質塑膠碎片至塑膠籃側板、橡膠塊'",
    "short_ = {'不可回收塑膠': '薄膜、帆布片之軟質包材', '廢紙': '吸水紙片、紙箱板', '保麗龍': '吸水或附著之發泡板、箱體側板', '廢漁網漁具': '網線、繩索', '寶特瓶': '瓶身碎片、完整進水瓶、半壓扁至重壓扁瓶身', '竹木': '細竹枝、大型竹木段（水浸飽和）', '可回收塑膠': '硬質塑膠碎片、塑膠籃側板、橡膠塊'")
open(os.path.join(HERE, 'report_run09_check.py'), 'w', encoding='utf-8').write(s)
print('written')
