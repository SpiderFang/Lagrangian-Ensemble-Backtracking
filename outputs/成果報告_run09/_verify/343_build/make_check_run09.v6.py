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
_sha = {n_: _h.sha256(z.read(n_)).hexdigest() for n_ in z.namelist() if n_.startswith('word/media/')}
_ref = {}
for _l in open(os.path.join(HERE, 'figs_new_v2', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_arrival_zh_v3', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
_rels = dict(re.findall(r'Id="(rId\\d+)"[^>]*?Target="(media/[^"]+)"', z.read('word/_rels/document.xml.rels').decode('utf8')))
_rels.update({a_: b_ for b_, a_ in re.findall(r'Target="(media/[^"]+)"[^>]*?Id="(rId\\d+)"', z.read('word/_rels/document.xml.rels').decode('utf8'))})
_docimg = ['word/' + _rels[r_] for r_ in re.findall(r'r:embed="(rId\\d+)"', xml.decode('utf8') if isinstance(xml, bytes) else xml)]
ok(len(_docimg) == 7 and len(set(_docimg)) == 7, '文件內依序引用 7 個不同圖檔：%s' % _docimg)
ok(_sha.get(_docimg[0]) == _ref['343_run09v2_fig1_circle.png'] and _sha.get(_docimg[1]) == _ref['343_run09v2_fig2_circle.png'], '圖1、圖2 圖檔 sha256 等於 .203 重繪輸出（半徑 10 km 圓、圖例「抽樣範圍」版）')
_order = ['wp61_%s_overview_S_notext.png' % _s for _s in ('gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang')]
ok(len(_sha) == 7 and all(_sha.get(_docimg[2 + i_]) == _ref[n_] for i_, n_ in enumerate(_order)), '圖3–圖7 圖檔 sha256 等於 .203 figures-zh-v3 輸出（全年總覽、各地點座標軸版 S、無文字版，commit b3b0213）；媒體檔共 7 個')
import struct as _st
_ext = [(int(m_.group(1)), int(m_.group(2))) for m_ in re.finditer(r'<wp:extent cx="(\\d+)" cy="(\\d+)"', xml.decode('utf8') if isinstance(xml, bytes) else xml)]
ok(len(_ext) == 7, '文件內圖片 7 張')
ok(abs(_ext[0][0] / 914400 - 4.134) < 0.01 and abs(_ext[0][1] / 914400 - 5.672) < 0.01 and abs(_ext[1][0] / 914400 - 6.267) < 0.01 and abs(_ext[1][1] / 914400 - 8.92) < 0.02, '圖1、圖2 嵌入尺寸沿用 run08（4.13x5.67、6.27x8.92 in）')
for _i in range(2, 7):
    _w, _h = _st.unpack('>II', z.read(_docimg[_i])[16:24])
    _sc = (_ext[_i][0] / 914400) / (_w / 300.0)
    ok(abs(_ext[_i][0] / 914400 - 6.267) < 0.01 and 5.5 <= _ext[_i][1] / 914400 <= 6.01 and _sc >= 0.999, '圖%d 寬 %.3f in、高 %.3f in，嵌入縮放 %.4f（圖內 9 pt 字不縮小）' % (_i + 1, _ext[_i][0] / 914400, _ext[_i][1] / 914400, _sc))
ok(len(re.findall(r'Type="[^"]*/image"', z.read('word/_rels/document.xml.rels').decode('utf8'))) == 7, '圖片關聯 7 個（無放大圖殘留）')''' + s[b:]
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
# 2b. 溯源停止邊界段：回溯上限 180 天與逐地點離開流場範圍比例（數字自 termination_by_group.csv 讀取，不手抄）
import csv as _csv
_t = {}
for _r in _csv.DictReader(open(os.path.join(HERE, 'src', 'termination_by_group.csv'), encoding='utf-8-sig')):
    if _r['grouping'] == 'site_all':
        _t[_r['site']] = (int(_r['n_total']), int(_r['n_crossed']))
ok(set(_t) == {'gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang'}, '停止邊界：csv 五地點齊全')
_pc = {k_: v_[1] / v_[0] * 100.0 for k_, v_ in _t.items()}
stop_ = [t_ for t_ in paras if t_.startswith('4、溯源停止邊界')]
ok(len(stop_) == 1, '溯源停止邊界段唯一')
stop_ = stop_[0]
_want = ('回溯期間上限為 180 天。於此期間內，貢寮、龜山島與新竹外海之全部虛擬質點離開流場範圍邊界，南灣有 %.3f%%，連江有 %.3f%%。'
         % (_pc['nanwan'], _pc['lienchiang']))
ok(stop_.endswith(_want), '停止邊界末句與 csv 一致：' + _want)
_pre = '本研究之關注區域即圖 1 之四區流場範圍，虛擬質點離開流場範圍邊界即停止追蹤。'
ok(stop_.endswith(_pre + _want) and stop_.count(_pre) == 1, '停止邊界段末：先有「本研究之關注區域即圖 1 之四區流場範圍，虛擬質點離開流場範圍邊界即停止追蹤。」再接回溯上限與比例句')
ok(_pc['gongliao'] == _pc['guishan'] == _pc['hsinchu'] == 100.0 and '%.3f' % _pc['nanwan'] == '99.999' and '%.3f' % _pc['lienchiang'] == '95.857', 'csv 值：貢寮、龜山島、新竹外海 100%、南灣 99.999%、連江 95.857%')
ok('開放邊界' not in fulltext and '開放邊界' not in stop_ and '流場範圍邊界' in stop_, '停止邊界段用「流場範圍邊界」，全文無「開放邊界」')
ok('180 天' in stop_ and '180 天' not in re.sub(re.escape('回溯期間上限為 180 天'), '', fulltext), '全文「180 天」僅出現於停止邊界末句')
ok(not re.search(r'較|比|高於|低於|相比|最', _want), '停止邊界末句無比較語氣')
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', _want)), '停止邊界末句無英文字母')

# 2c. 圖 3–圖 7 圖內文字（由 SVG 文字取出）：只有專有名詞或符號可有英文（白名單 Hs、m、m/s）、用語依服務建議書（有效波高）、Y 軸標題在 Y 軸位置
import glob as _g, json as _j
_cnt = _j.load(open(os.path.join(HERE, 'src', 'arrival_counts.json'), encoding='utf-8'))
_SITES = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
_svg_all = {}
for _s, _zh in _SITES:
    _f = os.path.join(HERE, 'figs_arrival_zh_v3', _s, 'wp61_%s_overview_S_notext.svg' % _s)
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
    for _w in ['大潮（%d）' % _c['class']['spring_proxy'], '小潮（%d）' % _c['class']['neap_proxy'],
               '漲潮最快（%d）' % _c['phase']['fastest_rising'], '退潮最快（%d）' % _c['phase']['fastest_falling'],
               '近平潮（%d）' % _c['phase']['slack_proxy'], '高浪事件（%d）' % _c['event']['high_wave_event'],
               '強流事件（%d）' % _c['event']['strong_current_event'], '季內中位數', '全年第 99 百分位',
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
    ok(set(re.findall(r'font-size: ([0-9.]+)px', _sv)) == {'9'}, '%s 圖內文字皆 9 px（嵌入縮放 1.0 即 9 pt）' % _zh)
    # Y 軸標題：垂直（rotate(-90)）、位於刻度字左側、其垂直位置在對應面板範圍內
    _ytick = [(float(re.search(r'y="([0-9.]+)"', a_).group(1)), x_) for x_, a_ in _els if re.fullmatch(r'\d+(\.\d+)?', x_) and 'x="49.78"' in a_]
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
_src = open(os.path.join(HERE, 'scripts_fig', 'wp61_v2', 'arrival_figs.py'), encoding='utf-8').read()
ok('LINE_MEDIAN = dict(color="black", lw=1.7, ls=(0, (7, 3)))' in _src and 'LINE_P99 = dict(color="#b2182b", lw=1.5, ls=(0, (1, 1.6)))' in _src, '季內中位數為黑色粗長虛線、全年第 99 百分位為深紅點線（程式常數）')
for _s, _zh in _SITES:
    _sv = open(os.path.join(HERE, 'figs_arrival_zh_v3', _s, 'wp61_%s_overview_S_notext.svg' % _s), encoding='utf-8').read()
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
    want_ += ['%s到達時間驗證圖（全年總覽）' % zh_]
ok([t_.split(' ', 1)[1] for t_ in order_[2:]] == want_, '圖3–圖7 圖題含地點名稱且依序為貢寮、龜山島、新竹外海、南灣、連江（僅全年總覽）')
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
open(os.path.join(HERE, 'report_run09_check.py'), 'w', encoding='utf-8').write(s)
print('written')
