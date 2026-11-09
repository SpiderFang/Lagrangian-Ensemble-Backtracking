# -*- coding: utf-8 -*-
"""2026-10-21 小修（v6）：(1) 頁尾移除「3-」文字 run，頁碼改為 1,2,3…（保留 PAGE 欄位、置中與段落樣式）；
(2) 「以下列出各研究地點之到達時間驗證圖…」導引段精簡（圖例已有之說明刪除，併入各地點圖號）。
只改 word/footer1.xml 與 word/document.xml 兩個部件，其餘 zip 成員逐位元組相同、順序不變。
用法：python3 edit_run09_343_v6.py <src.docx> <dst.docx>"""
import sys, re, zipfile

SRC, DST = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC)

NEW_TEXT = ('以下列出各研究地點之到達時間驗證圖，每個研究地點 1 張全年總覽圖，貢寮見圖 3，龜山島見圖 4，新竹外海見圖 5，南灣見圖 6，連江見圖 7。'
            '圖中潮汐強度即前述各時刻前後 25 小時之水位起伏大小，以海面高度之均方根計算；季內中位數即大潮與小潮之分界；近平潮即最接近漲退潮交替之時刻。'
            '2 個事件時刻另標示於有效波高與流速曲線上。同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊。')

# ---------- 1. 頁尾 ----------
ft = zin.read('word/footer1.xml').decode('utf-8')
OLD_RUN = '<w:r><w:rPr><w:rFonts w:hint="eastAsia"/></w:rPr><w:t>3-</w:t></w:r>'
assert ft.count(OLD_RUN) == 1 and ft.count('<w:t>3-</w:t>') == 1
assert ft.count('PAGE') == 1 and '<w:jc w:val="center"/>' in ft
ft2 = ft.replace(OLD_RUN, '')
assert '3-' not in re.sub(r'<[^>]+>', '', ft2)
# 其他頁首／頁尾部件與節
parts = [n for n in zin.namelist() if re.match(r'word/(header|footer)\d*\.xml$', n)]
assert parts == ['word/footer1.xml'], parts

# ---------- 2. 內文導引段 ----------
doc = zin.read('word/document.xml').decode('utf-8')
assert doc.count('<w:sectPr') == 1 and '<w:pgNumType' not in doc    # 節設定無 pgNumType（預設十進位、續編）
assert 'REF ' not in doc and 'fldSimple' not in doc               # 圖號為純文字，非交叉引用欄位
i = doc.index('以下列出各研究地點之到達時間驗證圖')
assert doc.count('以下列出各研究地點之到達時間驗證圖') == 1
s = max(doc.rfind('<w:p ', 0, i), doc.rfind('<w:p>', 0, i))
e = doc.index('</w:p>', i) + len('</w:p>')
para = doc[s:e]
m = re.match(r'(<w:p [^>]*>)(<w:pPr>.*?</w:pPr>)(<w:r>)(<w:rPr>.*?</w:rPr>)', para, flags=re.S)
assert m, para[:300]
open_tag, ppr, _, rpr = m.groups()
assert '<w:fldChar' not in para
newp = open_tag + ppr + '<w:r>' + rpr + '<w:t xml:space="preserve">' + NEW_TEXT + '</w:t></w:r></w:p>'
doc2 = doc[:s] + newp + doc[e:]
assert doc2.count(NEW_TEXT) == 1 and '橘色標記為大潮' not in doc2 and '圖例括號內為各類時刻數' not in doc2

# ---------- 寫出（保留部件順序與 ZipInfo） ----------
new = {'word/footer1.xml': ft2.encode('utf-8'), 'word/document.xml': doc2.encode('utf-8')}
zo = zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = new.get(it.filename, zin.read(it.filename))
    ni = zipfile.ZipInfo(it.filename, it.date_time); ni.compress_type = it.compress_type
    ni.external_attr = it.external_attr; ni.create_system = it.create_system
    zo.writestr(ni, data)
zo.close()
print('ok')
