#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v3：就地修改（合併版為主檔，不再由 merge_run09.py 重新合併）。
只改 word/document.xml 兩處文字；其餘 zip 成員逐位元組保留。
  (1) 「溯源停止邊界」段刪除成果數字（「，各研究地點皆於 33.4 天內停止追蹤，穿越比例與海岸來源比例依序為…43.94%」）
  (2) 第 1 項「各類速度均為沉降速度，不含上升情境。」→「各類速度均為沉降速度。」
用法：python3 edit_v3_inplace.py <合併版docx>
前提：目前 sha256 須為 0371e9e1…（使用者手改後之版本），否則停止。"""
import sys, re, zipfile, hashlib, os
PRE = '0371e9e13095e60f78952948'   # 前 24 碼
path = sys.argv[1]
raw = open(path, 'rb').read()
assert hashlib.sha256(raw).hexdigest().startswith(PRE), '合併版 sha 與預期不同，停止（可能又被手改）'
z = zipfile.ZipFile(path)
doc = z.read('word/document.xml').decode('utf-8')

# (2) 上升
old2 = '降速度，不含上升情境。每類由'
assert doc.count(old2) == 1
doc = doc.replace(old2, '降速度。每類由')

# (1) 停止邊界段
i = doc.index('溯源停止邊界（'); a = doc.rfind('<w:p ', 0, i); b = doc.index('</w:p>', i) + 6
p = doc[a:b]
k = p.index('天，各研究地點皆於')
rs = p.rfind('<w:r ', 0, k)                       # 含「天，各研究地點皆於」之 run 起點
assert p[rs:rs + 6] == '<w:r w'
re_first = re.match(r'<w:r [^>]*>(<w:rPr>.*?</w:rPr>)?<w:t[^>]*>天，各研究地點皆於</w:t></w:r>', p[rs:], re.S)
assert re_first, '找不到預期 run 結構'
first = re_first.group(0)
tail = p[rs:]
assert tail.endswith('<w:t>。</w:t></w:r></w:p>'), tail[-120:]
removed_text = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', tail))
assert removed_text.startswith('天，各研究地點皆於 33.4 天內停止追蹤，穿越比例與海岸來源比例依序為貢寮 73.40%、26.60%，龜山島 78.35%、21.65%，新竹外海 97.77%、2.23%，南灣 68.91%、31.09%，連江 56.06%、43.94%。'), removed_text
newrun = first.replace('天，各研究地點皆於', '天。')
p2 = p[:rs] + newrun + '</w:p>'
doc = doc[:a] + p2 + doc[b:]

tmp = path + '.partial'
with zipfile.ZipFile(tmp, 'w') as zo:
    for info in z.infolist():
        zi = zipfile.ZipInfo(info.filename, info.date_time)
        zi.compress_type = info.compress_type; zi.external_attr = info.external_attr; zi.create_system = info.create_system
        zo.writestr(zi, doc.encode('utf-8') if info.filename == 'word/document.xml' else z.read(info.filename))
os.replace(tmp, path)
print('OK', hashlib.sha256(open(path, 'rb').read()).hexdigest(), os.path.getsize(path))
