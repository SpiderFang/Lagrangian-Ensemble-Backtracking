# -*- coding: utf-8 -*-
"""2026-10-22 方法 B 修訂（v12）：「4、溯源停止邊界」段依海岸來源方法 B 改寫（抵達海岸亦停止追蹤，列各地點穿越比例與海岸來源比例）。
只改 word/document.xml 該段三處文字（run 層級、不動 rsid 與格式）；其餘 zip 成員逐位元組相同，成員順序與 ZipInfo 不變。
比例由 _verify/src_coastB/coastB-tables/outcome_by_site.csv 之顆數計算，兩位小數四捨五入（ROUND_HALF_UP）。
用法：python3 edit_run09_343_v12.py <src.docx> <dst.docx>"""
import sys, os, re, csv, zipfile, hashlib, math
from decimal import Decimal, ROUND_HALF_UP
SRC, DST = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'src_coastB', 'coastB-tables', 'outcome_by_site.csv')
rows = {r['site']: r for r in csv.DictReader(open(CSV, encoding='utf-8-sig'))}
ZH = [('gongliao', '貢寮'), ('guishan', '龜山島'), ('hsinchu', '新竹外海'), ('nanwan', '南灣'), ('lienchiang', '連江')]
def pct(n, tot):
    return str((Decimal(n) * 100 / Decimal(tot)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) + '%'
assert all(int(rows[s]['max_backtrack']) == 0 for s, _ in ZH)
assert all(int(rows[s]['crossed']) + int(rows[s]['coast']) + int(rows[s]['touch_bottom']) == int(rows[s]['total']) for s, _ in ZH)
assert [s for s, _ in ZH if int(rows[s]['touch_bottom'])] == ['nanwan'] and int(rows['nanwan']['touch_bottom']) == 1
maxd = max(float(rows[s]['max_duration_days']) for s, _ in ZH)
dmax = '%.1f' % (math.ceil(maxd * 10 - 1e-9) / 10)
assert dmax == '33.4', dmax
P = {s: (pct(rows[s]['crossed'], rows[s]['total']), pct(rows[s]['coast'], rows[s]['total'])) for s, _ in ZH}
TAIL = ('天，各研究地點皆於 %s 天內停止追蹤，穿越比例與海岸來源比例依序為貢寮 %s、%s，龜山島 %s、%s，新竹外海 %s、%s，南灣 %s、%s，連江 %s、%s。'
        % ((dmax,) + P['gongliao'] + P['guishan'] + P['hsinchu'] + P['nanwan'] + P['lienchiang']))
print(TAIL)
zin = zipfile.ZipFile(SRC)
doc = zin.read('word/document.xml').decode('utf8')
ps = [m for m in re.finditer(r'<w:p[ >].*?</w:p>', doc, flags=re.S) if '溯源停止邊界' in re.sub(r'<[^>]+>', '', m.group(0)) and re.sub(r'<[^>]+>', '', m.group(0)).startswith('4、溯源停止邊界')]
assert len(ps) == 1
pm = ps[0]; ptxt = pm.group(0)
runs = list(re.finditer(r'<w:r(?: [^>]*)?>.*?</w:r>', ptxt, flags=re.S))
rt = lambda r: ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', r.group(0)))
texts = [rt(r) for r in runs]
before = ''.join(texts)
def idx(sub):
    h = [i for i, t in enumerate(texts) if sub in t]
    assert len(h) == 1, (sub, h)
    return h[0]
def set_run_text(run_xml, new):
    assert run_xml.count('<w:t') == 1
    return re.sub(r'<w:t(?: [^>]*)?>[^<]*</w:t>', lambda m: '<w:t%s>%s</w:t>' % (' xml:space="preserve"' if (new != new.strip() or '  ' in new) else '', new), run_xml)
new_runs = {}   # index -> new xml or None(delete)
i1 = idx('）」邊界為止，藉此精確鎖定')
new_runs[i1] = set_run_text(runs[i1].group(0), texts[i1].replace('）」邊界為止，', '）」邊界或抵達海岸為止，'))
i2 = idx('之四區流場範圍，虛擬質點離開流場範圍邊界即停止追蹤。回溯期間上限為')
new_runs[i2] = set_run_text(runs[i2].group(0), texts[i2].replace('虛擬質點離開流場範圍邊界即停止追蹤。', '虛擬質點穿越流場範圍邊界或抵達海岸（視為找到來源，稱為海岸來源）即停止追蹤。'))
i3 = idx('天。於此期間內，貢寮、龜山島與新竹外海之全部虛擬質點離開流場範圍邊界，南灣有')
assert [texts[i3 + k] for k in range(1, 5)] == [' 99.999%', '，連江有', ' 95.857%', '。'] and i3 + 4 == len(runs) - 1
rpr = lambda r: re.search(r'<w:rPr>.*?</w:rPr>', r.group(0), flags=re.S).group(0)
assert len({rpr(runs[i3 + k]) for k in range(0, 5)}) == 1
new_runs[i3] = set_run_text(runs[i3].group(0), TAIL)
for k in range(1, 5):
    new_runs[i3 + k] = None
out, pos = [], 0
for i, r in enumerate(runs):
    out.append(ptxt[pos:r.start()])
    if i in new_runs:
        if new_runs[i] is not None:
            out.append(new_runs[i])
    else:
        out.append(r.group(0))
    pos = r.end()
out.append(ptxt[pos:])
newp = ''.join(out)
after = re.sub(r'<[^>]+>', '', newp)
print('BEFORE:', before); print('AFTER :', after)
newdoc = doc[:pm.start()] + newp + doc[pm.end():]
assert newdoc.count(newp) == 1 and len(doc) != len(newdoc)
repl = {'word/document.xml': newdoc.encode('utf8')}
with zipfile.ZipFile(DST, 'w') as zout:
    for zi in zin.infolist():
        data = repl.get(zi.filename, zin.read(zi.filename))
        ni = zipfile.ZipInfo(zi.filename, zi.date_time)
        ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr; ni.create_system = zi.create_system
        ni.comment = zi.comment; ni.extra = zi.extra; ni.flag_bits = zi.flag_bits
        zout.writestr(ni, data)
zo = zipfile.ZipFile(DST)
assert zo.testzip() is None and zo.namelist() == zin.namelist()
for n in zin.namelist():
    assert (zo.read(n) == zin.read(n)) == (n not in repl), n
# 全文其他段落文字不變
ptx = lambda x: [re.sub(r'<[^>]+>', '', m.group(0)) for m in re.finditer(r'<w:p[ >].*?</w:p>', x, flags=re.S)]
a, b = ptx(doc), ptx(newdoc)
assert len(a) == len(b) and [i for i in range(len(a)) if a[i] != b[i]] == [ptx(doc).index(before)]
print('ok', hashlib.sha256(open(DST, 'rb').read()).hexdigest())
