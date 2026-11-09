import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, 'make_check_run09.py')
s = open(P, encoding='utf-8').read()

def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)

def sub_line(startswith, new):
    """整行（以 startswith 開頭，含該行至換行）替換。"""
    global s
    i = s.index(startswith); assert s.count(startswith) == 1, startswith[:60]
    j = s.index('\n', i)
    s = s[:i] + new + s[j:]

# A. 參考 sha：保留 v6（供對照）、加入 v5（圖1、圖2）與 v7（圖3–7）
sub("""for _l in open(os.path.join(HERE, 'figs_arrival_zh_v6', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
""", """for _l in open(os.path.join(HERE, 'figs_arrival_zh_v6', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
_ref6 = dict(_ref)   # v9–v12 之圖 3–7（figs_arrival_zh_v6），供新舊對照
for _l in open(os.path.join(HERE, 'figs_new_v5', 'sha203.txt'), encoding='utf-8'):  # v13（WP-74）：圖 1、圖 2 改 v5（8 pt）
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_arrival_zh_v7', 'sha203.txt'), encoding='utf-8'):  # v13（WP-74）：圖 3–7 改 figs_arrival_zh_v7（8 pt）
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
""")
# B. 圖 1、圖 2 sha
sub_line("ok(_sha.get(_docimg[0]) == _ref['343_run09v4_fig1_circle.png']",
 "ok(_sha.get(_docimg[0]) == _ref['343_run09v5_fig1_circle.png'] and _sha.get(_docimg[1]) == _ref['343_run09v5_fig2_circle.png'], '圖1、圖2 圖檔 sha256 等於 .203 重繪輸出 figs_new_v5（圖內文字 8 pt、TRAPs 框線與刻度，v13；長寬依期中報告圖 2-17 之 v4 版面不變）')")
# D. manifest 改讀 v5
sub("_m3 = _j3.load(open(os.path.join(HERE, 'figs_new_v4', '343_run09v4_figs_circle_manifest.json'), encoding='utf-8'))",
    "_m3 = _j3.load(open(os.path.join(HERE, 'figs_new_v5', '343_run09v5_figs_circle_manifest.json'), encoding='utf-8'))   # v13：與 v4 版面相同")
# F. 圖 3–7 sha 文字
sub_line("ok(len(_sha) == 7 and all(_sha.get(_docimg[2 + i_]) == _ref[n_]",
 "ok(len(_sha) == 7 and all(_sha.get(_docimg[2 + i_]) == _ref[n_] for i_, n_ in enumerate(_order)) and all(_ref[n_] != _ref6[n_] for n_ in _order), '圖3–圖7 圖檔 sha256 等於 figs_arrival_zh_v7（.203 figures-zh-v6 重繪輸出，圖內文字 8 pt，v13）且已不同於 v6（v9–v12 之 9 pt 版）；媒體檔共 7 個')")
# G. 刪除 v3／v5 像素差異區塊，改為尺寸與圖例對齊檢查
i = s.index("_v3 = {}\nfor _l in open(os.path.join(HERE, 'figs_arrival_zh_v3', 'sha203.txt')")
j = s.index("except ImportError:\n    ok(True, '（略過像素差異檢查：無 PIL／numpy）')\n")
j2 = j + len("except ImportError:\n    ok(True, '（略過像素差異檢查：無 PIL／numpy）')\n")
new_blk = """import struct as _st0
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
"""
s = s[:i] + new_blk + s[j2:]
# H. 嵌入縮放文字
sub("嵌入縮放 %.4f（圖內 9 pt 字不縮小）", "嵌入縮放 %.4f（圖內 8 pt 字不縮小）")
# I. SVG 讀 v7；字級 8；刻度標籤 x 不寫死
sub("_f = os.path.join(HERE, 'figs_arrival_zh_v6', _s, 'wp61_%s_overview_S_notext.svg' % _s)", "_f = os.path.join(HERE, 'figs_arrival_zh_v7', _s, 'wp61_%s_overview_S_notext.svg' % _s)")
sub("_sv = open(os.path.join(HERE, 'figs_arrival_zh_v6', _s, 'wp61_%s_overview_S_notext.svg' % _s), encoding='utf-8').read()", "_sv = open(os.path.join(HERE, 'figs_arrival_zh_v7', _s, 'wp61_%s_overview_S_notext.svg' % _s), encoding='utf-8').read()")
sub("ok(set(re.findall(r'font-size: ([0-9.]+)px', _sv)) == {'9'}, '%s 圖內文字皆 9 px（嵌入縮放 1.0 即 9 pt）' % _zh)", "ok(set(re.findall(r'font-size: ([0-9.]+)px', _sv)) == {'8'}, '%s 圖內文字皆 8 px（SVG 單位 px＝pt；嵌入縮放 1.0 即印出 8 pt）' % _zh)")
sub("""'x="49.78"' in a_]""", """'text-anchor: end' in a_]""")
sub("_src = open(os.path.join(HERE, 'scripts_fig', 'wp61_v2', 'arrival_figs.py'), encoding='utf-8').read()", "_src = open(os.path.join(HERE, 'scripts_fig', 'wp74_343', 'arrival_figs.py'), encoding='utf-8').read()")
# J. 新增 v13 區塊：接在「圖片關聯 7 個」之後（仍在字串內，故不用反斜線）
anchor = "ok(len(re.findall(r'Type=\"[^\"]*/image\"', z.read('word/_rels/document.xml.rels').decode('utf8'))) == 7, '圖片關聯 7 個（無放大圖殘留）')"
assert s.count(anchor) == 1
NEW = anchor + """
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
"""
s = s.replace(anchor, NEW)
open(P, 'w', encoding='utf-8').write(s)
print('patched')
