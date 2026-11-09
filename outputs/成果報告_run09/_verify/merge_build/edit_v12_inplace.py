#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v12 候選：（一）2 受體選取段之一句，「海床高程」→「該處地形水深」（使用者核定；其餘字不動，「潮位」保留）。
基準：使用者於 v11 升格後再存之主檔（sha256 以 9808fcad1dfc285f 起首；相對 v11 候選之手改：標題加「分析成果」、表 1 三格用語）。
做法：只在 word/document.xml 的原始位元組中把唯一一處「海床高程」（單一 run，rsid、格式不動）換成「該處地形水深」；其餘 zip 成員逐位元組保留。
用法：edit_v12_inplace.py <現行主檔> <輸出docx>"""
import sys, zipfile, hashlib
PRE = '9808fcad1dfc285f'
OLD, NEW = '海床高程', '該處地形水深'
def main(src, dst):
    raw = open(src, 'rb').read(); assert hashlib.sha256(raw).hexdigest().startswith(PRE), '主檔 sha 與預期不同，停止'
    zin = zipfile.ZipFile(src); xml = zin.read('word/document.xml')
    assert xml.count(OLD.encode('utf-8')) == 1, xml.count(OLD.encode('utf-8'))
    ctx = b'<w:t>' + OLD.encode('utf-8') + b'</w:t>'; assert xml.count(ctx) == 1   # 唯一，且為獨立 run 之全文
    xml2 = xml.replace(ctx, b'<w:t>' + NEW.encode('utf-8') + b'</w:t>')
    zout = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
    for zi in zin.infolist():
        data = xml2 if zi.filename == 'word/document.xml' else zin.read(zi.filename)
        ni = zipfile.ZipInfo(zi.filename, date_time=zi.date_time); ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
        zout.writestr(ni, data)
    zout.close(); print('ok', dst, hashlib.sha256(open(dst, 'rb').read()).hexdigest())
if __name__ == '__main__': main(*sys.argv[1:3])
