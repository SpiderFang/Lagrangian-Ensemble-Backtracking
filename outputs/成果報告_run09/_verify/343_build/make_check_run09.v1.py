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
sub('"""核對報告 docx 內的數字', '"""（run09 版，由 make_check_run09.py 自 report_run08_check.py 產生）核對報告 docx 內的數字')
# 圖片替代文字：共 12 張；前兩張（圖1、圖2）含「各研究地點」
sub("ok(len(dsc) == 2 and all('各研究地點' in d.decode('utf8') and '站' not in d.decode('utf8') for d in dsc), '圖片替代文字')",
    "ok(len(dsc) == 12 and all('各研究地點' in d.decode('utf8') for d in dsc[:2]) and all('站' not in d.decode('utf8') for d in dsc), '圖片替代文字（共 12 張，圖1、圖2 含各研究地點）')")
# 表2 南灣
sub("ok(gs_[1:] == ['121.94', '24.85'], '表2 龜山島列：%s' % gs_)",
    "ok(gs_[1:] == ['121.94', '24.85'], '表2 龜山島列：%s' % gs_)\n"
    "ok([r_[1:] for r_ in rows2_] == [['121.92', '25.10'], ['121.94', '24.85'], ['120.45', '24.75'], ['120.77', '21.92'], ['119.95', '26.20']], '表2 五列座標（南灣已改 120.77、21.92）：%s' % rows2_)\n"
    "ok(['%.2f' % round(x_, 2) for x_ in (121.9223889, 25.0964444, 121.939148, 24.850326, 120.45, 24.75, 120.770675, 21.92041, 119.95, 26.2)] == ['121.92', '25.10', '121.94', '24.85', '120.45', '24.75', '120.77', '21.92', '119.95', '26.20'], '表2 與 run09 release config 錨點進位 0.01° 一致')")
# 圖檔
a = s.index("ok(imgs_.get('word/media/image1.png')")
b = s.index('\n', a)
s = s[:a] + '''import hashlib as _h
_sha = {n_: _h.sha256(z.read(n_)).hexdigest() for n_ in z.namelist() if n_.startswith('word/media/')}
_ref = {}
for _l in open(os.path.join(HERE, 'figs_new', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[_p[1]] = _p[0]
for _l in open(os.path.join(HERE, 'figs_arrival_zh', 'sha203.txt'), encoding='utf-8'):
    _p = _l.split()
    _ref[os.path.basename(_p[1])] = _p[0]
ok(_sha.get('word/media/image1.png') == _ref['343_run09_fig1_circle.png'] and _sha.get('word/media/image2.png') == _ref['343_run09_fig2_circle.png'], '圖1、圖2 圖檔 sha256 等於 .203 重繪輸出（半徑 10 km 版）')
_order = []
for _s in ('gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang'):
    _order += ['wp61_%s_overview_U_notext.png' % _s, 'wp61_%s_zoom_S_notext.png' % _s]
ok(len(_sha) == 12 and all(_sha.get('word/media/image%d.png' % (3 + i_)) == _ref[n_] for i_, n_ in enumerate(_order)), '圖3–圖12 圖檔 sha256 等於 .203 run09 驗證圖（全年總覽取 U 版、逐時刻放大取 S 版、無文字版）')''' + s[b:]
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
           '龜山島之西半部配置 3 個、東半部配置 2 個',
           '並於同一區塊內依隨機順序遞補',
           '候選面須為有效之三角形網格且非陸地，並於全部 50 個到達時間均維持濕潤（不因退潮而乾出）',
           '抽中後另須通過流場與波浪資料之逐時空間支援檢查，未通過者剔除']:
    ok(k_ in rec_, '受體段關鍵句：' + k_)
for k_ in ['無放回', '均勻隨機抽取 5 個位置', '依隨機抽樣之順序遞補', '12.5', '關注海域位於島的西側', '位於島的西側', '故西半部']:
    ok(k_ not in rec_, '受體段不得再有舊方法用語：' + k_)
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
ok(_pc['gongliao'] == _pc['guishan'] == _pc['hsinchu'] == 100.0 and '%.3f' % _pc['nanwan'] == '99.999' and '%.3f' % _pc['lienchiang'] == '95.857', 'csv 值：貢寮、龜山島、新竹外海 100%、南灣 99.999%、連江 95.857%')
ok('開放邊界' not in fulltext and '開放邊界' not in stop_ and '流場範圍邊界' in stop_, '停止邊界段用「流場範圍邊界」，全文無「開放邊界」')
ok('180 天' in stop_ and '180 天' not in re.sub(re.escape('回溯期間上限為 180 天'), '', fulltext), '全文「180 天」僅出現於停止邊界末句')
ok(not re.search(r'較|比|高於|低於|相比|最', _want), '停止邊界末句無比較語氣')
ok(not re.search(r'[A-Za-z]', re.sub(r'\s', '', _want)), '停止邊界末句無英文字母')

# 2c. 圖 3–圖 12 圖內文字（由 SVG 文字取出）：不得有英文縮寫、UTC、OCM、Hs、p99，僅允許單位符號 m、m/s、h
import glob as _g
_bad_svg = []
_alltxt = set()
for _f in sorted(_g.glob(os.path.join(HERE, 'figs_arrival_zh', '*_notext.svg'))):
    _s = open(_f, encoding='utf-8').read()
    for _m in re.findall(r'<text[^>]*>(.*?)</text>', _s, flags=re.S):
        _tx = re.sub(r'<[^>]+>', '', _m).strip()
        if _tx:
            _alltxt.add(_tx)
            _rest = re.sub(r'\bm/s\b|\bm\b|\bh\b', '', _tx.replace('&lt;', '<'))
            if re.search(r'[A-Za-z]', _rest):
                _bad_svg.append((os.path.basename(_f), _tx))
ok(len(_g.glob(os.path.join(HERE, 'figs_arrival_zh', '*_notext.svg'))) == 10, '圖內文字 SVG 共 10 個')
ok(not _bad_svg, '圖內文字無英文字母（單位 m、m/s、h 除外）：%s' % _bad_svg)
for _w in ('模式海面水位', '示性波高（m）', '全年第 99 百分位', '全年最大值', '2025 年（世界時，月/日）'):
    ok(_w in _alltxt, '圖內文字含「%s」' % _w)
for _w in ('OCM', 'Hs', 'p99', 'UTC', '年最大', '年 p99'):
    ok(not any(_w in x_ and not x_.startswith('全年') for x_ in _alltxt), '圖內文字不含「%s」' % _w)

# 3. 回溯上限：全文不得出現 90 天、2160 小時（正文原無此二詞；180 天亦不在 3-4-3 內）
ok(not re.search(r'90\s*天|2160|2,160|九十天|三個月', fulltext), '全文無「90 天」「2160 小時」')
ok(re.search(r'12\.5\s*km', fulltext) is None, '全文無「12.5 km」')
# 4. 驗證圖：導引句、圖題、編號、引用
guide_ = [t_ for t_ in paras if t_.startswith('以下列出各研究地點之到達時間驗證圖')]
ok(len(guide_) == 1, '導引句唯一')
g_ = guide_[0]
for k_ in ['圖中標示 50 個到達時間在潮汐強度與水位曲線上的位置', '向上三角為漲潮最快，向下三角為退潮最快，菱形為近平潮（最接近漲退潮交替之時刻）',
           '2 個事件時刻改以示性波高或流速呈現', '橘色標記為大潮、藍色標記為小潮']:
    ok(k_ in g_, '導引句：' + k_)
ok(not re.search(r'[A-Za-z]', g_) and not re.search(r'可能|或許|大概|似乎|應該|恐怕|推測|也許', g_), '導引句無英文字母、無不確定語氣')
order_ = [t_ for t_ in paras if re.match(r'^圖\d+ ', t_)]
nums_ = [int(re.match(r'^圖(\d+) ', t_).group(1)) for t_ in order_]
ok(nums_ == list(range(1, 13)), '圖號連續 1–12：%s' % nums_)
want_ = []
for zh_ in ('貢寮', '龜山島', '新竹外海', '南灣', '連江'):
    want_ += ['%s到達時間驗證圖（全年總覽）' % zh_, '%s到達時間驗證圖（逐時刻放大）' % zh_]
ok([t_.split(' ', 1)[1] for t_ in order_[2:]] == want_, '圖3–圖12 圖題含地點名稱且依序為全年總覽、逐時刻放大')
nonc_ = '\n'.join(t_ for t_ in paras if not re.match(r'^圖\d+ ', t_))
for n_ in range(1, 13):
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
ok(len(fi_) == 12 and all(seq2_[i_ + 1][0] == 'cap' for i_ in fi_), '12 張圖，每張圖下一段皆為圖題（圖在上、圖題在下）')
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
'''
sub("# ---------------- 公式數 ----------------", NEW + "\n# ---------------- 公式數 ----------------")
open(os.path.join(HERE, 'report_run09_check.py'), 'w', encoding='utf-8').write(s)
print('written')
