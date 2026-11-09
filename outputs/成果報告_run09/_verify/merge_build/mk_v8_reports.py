#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 v8_build_log.json 與主檔／候選產生 v8_change_list.md、v8_place_names.md（不改任何 docx）。"""
import sys, os, re, json, zipfile, collections
from lxml import etree
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'; q = lambda t: '{%s}%s' % (W, t)
def blocks(path):
    root = etree.fromstring(zipfile.ZipFile(path).read('word/document.xml')); out = []
    for e in root.find(q('body')):
        out.append(''.join(t.text or '' for t in e.iter(q('t'))) if e.tag in (q('p'), q('tbl')) else '')
    return out
MP = os.path.join(BASE, '成果報告_run09_合併版.docx'); CP = os.path.join(HERE, '成果報告_run09_合併版_v8候選.docx')
M = blocks(MP); Cn = blocks(CP)
LOG = json.load(open(os.path.join(HERE, 'v8_build_log.json'), encoding='utf-8'))
SN = {'gongliao': '貢寮', 'guishan': '龜山島', 'hsinchu': '新竹外海', 'nanwan': '南灣', 'lienchiang': '連江'}
BASEN = {'gongliao': 8, 'guishan': 19, 'hsinchu': 30, 'nanwan': 41, 'lienchiang': 52}; OLDN = {'gongliao': 8, 'guishan': 15, 'hsinchu': 22, 'nanwan': 29, 'lienchiang': 36}
KEYN = ['全年邊界', '全年海岸', '十類邊界', '十類海岸', '季節邊界', '季節海岸', '大小潮邊界', '大小潮海岸']
out = []
w = out.append
w('# v8 候選變更清單（來源分段機率圖取代圓形來源方位圖）\n')
w('候選：`_verify/merge_build/成果報告_run09_合併版_v8候選.docx`；主檔 `成果報告_run09_合併版.docx`（sha256 45f8d015…，未動）。產生程式：`edit_v8_inplace.py`（文字由 `v8_content.py` 依 segprob_v3 之 manifest 與表產生）。\n')
w('## 一、圖號對照（舊 → 新）\n')
w('圖 1–7 不變（位於（一））。（二）各地點原 7 張圖，現 11 張（總計 42 → 62 張）。\n')
w('| 地點 | 舊圖號 | 舊內容 | 新圖號 | 處理 |\n|---|---|---|---|---|')
for s in SN:
    o, n = OLDN[s], BASEN[s]
    if o == n:
        w('| %s | %d–%d | 軌跡、仍在追蹤、足跡 | %d–%d | 保留，號不變 |' % (SN[s], o, o + 2, n, n + 2))
    else:
        w('| %s | %d–%d | 軌跡、仍在追蹤、足跡 | %d–%d | 保留，改號 |' % (SN[s], o, o + 2, n, n + 2))
    w('| %s | %d | 十類來源方位（圓形） | %d、%d | 移除；改為十類邊界（%d）、十類海岸（%d） |' % (SN[s], o + 3, n + 5, n + 6, n + 5, n + 6))
    w('| %s | （新增） | 全年（十類合併）來源分段機率 | %d、%d | 新增：全年邊界（%d）、全年海岸（%d） |' % (SN[s], n + 3, n + 4, n + 3, n + 4))
    w('| %s | %d | 十類各十個沉降速度 | — | 移除（圖與導引段）；十類圖導引段末保留一句「同類十個沉降速度之間亦相近」 |' % (SN[s], o + 4))
    w('| %s | %d | 季節（四季） | %d–%d | 移除；改為季節邊界（%d）、季節海岸（%d） |' % (SN[s], o + 5, n + 7, n + 8, n + 7, n + 8))
    w('| %s | %d | 大小潮 | %d–%d | 移除；改為大小潮邊界（%d）、大小潮海岸（%d） |' % (SN[s], o + 6, n + 9, n + 10, n + 9, n + 10))
w('\n新增圖（每地點 8 張，共 40 張；順序為全年邊界、全年海岸、十類邊界、十類海岸、季節邊界、季節海岸、大小潮邊界、大小潮海岸）：\n')
w('| 地點 | 新圖號 | 內容 | 檔案（segprob_v3） | 高 (px) | 嵌入檔 |\n|---|---|---|---|---|---|')
FM = LOG['fig_map']; fn = {'all_b': 'boundary_all', 'all_c': 'coast_all', 'cls_b': 'boundary_class', 'cls_c': 'coast_class', 'sea_b': 'boundary_season', 'sea_c': 'coast_season', 'tid_b': 'boundary_tide', 'tid_c': 'coast_tide'}
for r in FM:
    w('| %s | %d | %s | %s/%s.png | %d | word/media/%s |' % (SN[r['site']], r['n'], KEYN[list(fn).index(r['key'])], r['site'], fn[r['key']], r['h_px'], r['media']))
# 圖號引用逐項
w('\n## 二、圖號引用逐項（舊 → 新；靜態文字，非欄位）\n')
w('| 主檔區塊 | 段落開頭 | 舊引用 | 新引用 | 處理 |\n|---|---|---|---|---|')
removed = {int(c['old_block']) for c in LOG['changes'] if c['kind'] == 'removed'}
renum = {int(o): int(n) for o, n in LOG['renum']}
rewritten = {int(c['old_block']): c for c in LOG['changes'] if c['kind'] == 'rewrite' and 'old_block' in c}
for i in range(33, 158):
    t = M[i]
    if not t: continue
    refs = re.findall(r'圖 ?\d+(?: 至圖 ?\d+)?', t)
    if not refs: continue
    if i in removed: w('| %d | %s | %s | — | 該段移除（圓形圖段，由新圖及其導引段取代） |' % (i, t[:14], '、'.join(refs))); continue
    if i in rewritten:
        nt = rewritten[i]['new']; nr = re.findall(r'圖 ?\d+(?: 至圖 ?\d+)?', nt)
        w('| %d | %s | %s | %s | 改寫 |' % (i, t[:14], '、'.join(refs), '、'.join(nr) or '（無）')); continue
    m = re.match(r'^圖( ?)(\d+)', t)
    if m and int(m.group(2)) in renum:
        o = int(m.group(2)); w('| %d | %s | %s | 圖%s%d | %s |' % (i, t[:14], refs[0], m.group(1), renum[o], '改號' if renum[o] != o else '不變')); continue
    w('| %d | %s | %s | %s | 不變 |' % (i, t[:14], '、'.join(refs), '、'.join(refs)))
w('\n新段落中之圖號引用（皆指向同地點之 11 張圖，見檢查 R3、R4、R6）：方法段「圖 11 至圖 18」；每個導引段首之「圖 N」＝其圖說編號；十類、季節、大小潮之判定句引用同組邊界圖與海岸圖（如「依圖 13 與圖 14」）；各地點軌跡導引段結尾「與圖 A 及圖 B 之主要來源一致」（新竹外海僅圖 33）。\n')
w('## 三、逐段變更（舊 → 新文字）\n')
for c in LOG['changes']:
    if c['kind'] == 'rewrite':
        w('### 區塊 %s（%s）\n\n- 舊：%s\n- 新：%s\n' % (c['old_block'], SN.get(c.get('site'), '全文') + '／' + c['role'], c['old'], c['new']))
w('### 移除之段落（圓形來源方位圖及其導引段、圖說）\n')
for c in LOG['changes']:
    if c['kind'] == 'removed': w('- 主檔區塊 %d（%s）：%s' % (int(c['old_block']), SN[c['site']], c['old'][:120]))
w('\n### 新增之段落（導引段與圖說；數字取自 segprob_v3）\n')
for c in LOG['changes']:
    if c['kind'] == 'new': w('- [%s／%s%s] %s' % (SN[c['site']], c['role'], ('／圖%d' % c['fig']) if 'fig' in c else '', c['new']))
# (f)
w('\n## 四、（f）關鍵字逐項處理（主檔出現處 → 候選處理）\n')
kw = ['圓', '扇區', '方向機率', '玫瑰', '十速度', '主要來源方向', '360', '來源方向', '方位', '正北', '10°']
w('| 關鍵字 | 主檔出現區塊 | 處理 | 候選殘留 |\n|---|---|---|---|')
for k in kw:
    hits = [i for i, t in enumerate(M) if k in t]
    cand = [i for i, t in enumerate(Cn) if k in t]
    inside = [i for i in hits if 33 <= i <= 157]
    outside = [i for i in hits if not (33 <= i <= 157)]
    res = []
    if inside: res.append('（二）內 %d 段（區塊 %s）：圓形圖段已移除或改寫' % (len(inside), ','.join(map(str, inside[:30])) + ('…' if len(inside) > 30 else '')))
    if outside: res.append('（二）外 %d 段（區塊 %s）：%s' % (len(outside), ','.join(map(str, outside[:20])), '與圓形來源圖無關，維持' if k in ('圓', '方位') else '維持'))
    w('| %s | %s | %s | %s |' % (k, ','.join(map(str, hits[:12])) + ('…' if len(hits) > 12 else '') or '—', '；'.join(res) or '—', '無' if not cand else '候選區塊 %s' % ','.join(map(str, cand[:12]))))
dg = collections.Counter(); angle = []
for i, t in enumerate(M):
    for m in re.finditer('度', t): dg[t[m.start() - 1] if m.start() else ''] += 1
    for m in re.finditer(r'\d+ ?°(?![CNE])', t): angle.append((i, t[max(0, m.start() - 6):m.end() + 4]))
w('\n「度」於主檔共 %d 次，前字分佈：%s；皆為速度、深度、密度、程度、角度（幾何）等詞，非「方位角度」之義；以「數字＋°」表角度者於主檔僅出現 %s，皆在已移除之方法段（10°）。' % (sum(dg.values()), '、'.join('%s×%d' % kv for kv in dg.most_common(12)), angle[:5]))
cd = collections.Counter();
for t in Cn:
    for m in re.finditer(r'°(.)', t): cd[m.group(1)] += 1
w('候選中「°」之後字元：%s（°N、°E 為經緯度；°C 為附錄海水溫度）。' % dict(cd))
w('候選全文殘留檢查（檢查 T6）：圓形、來源方位、扇區、玫瑰、方向機率、主要來源方向、來源方向、方向區間、正北、順時針、1/36、2.78%、360、十速度、BayTrace、反射、本研究——皆無。\n')

# ---- 第二輪（v8b）：相對第一版候選（v8a，sha f814b93f…）之逐段差異
A = blocks('/tmp/v8a_cand.docx')
assert len(A) == len(Cn)
w('\n## 五、第二輪修訂（v8b；相對第一版候選 v8a）逐段舊→新\n')
w('依協調者審閱意見：(1) 還原使用者核定之「由於廢棄物實際何時沉底未知，」；(2) 刪方法段「組間差異之判讀」句；(3) 龜山島海岸子分項改寫為「臺灣本島占 11.96%，龜山島占 9.57%，其餘為其他島嶼」並避免同段重複 9.57%；(4) 季節、大小潮、十類導引段不寫經緯度，以位置詞（南側邊界中段、西南角、研究地點西北側臺灣本島海岸、龜山島沿岸、馬祖北竿…）命名，每格只寫穿越／海岸來源比例與第一名之值；(5) 零值：顆數恰為 0 寫「無海岸來源」，不為 0 而低於 0.005% 寫「<0.01%」（新竹外海：冬、春、夏季無海岸來源；小潮 <0.01%，1 顆／44,600 顆）；(6) 整體結論 (1)–(5) 季節部分改為定性。\n')
n_ch = 0
for i, (a, b) in enumerate(zip(A, Cn)):
    if a != b:
        n_ch += 1; w('### 候選區塊 %d\n\n- v8a：%s\n- v8b：%s\n' % (i, a, b))
w('共 %d 個區塊有變動；其餘區塊與 v8a 逐字相同。\n' % n_ch)
w('## 六、9.56／0.13 全文出現處（主檔與候選，含表格、附錄、頁首頁尾）\n')
w('主檔：僅 1 處，位於正文（非表格）之圓形圖導引段「龜山島之海岸來源（21.65%）中，臺灣本島占 11.96%，龜山島占 9.56%，其他島嶼占 0.13%」（主檔區塊 69，龜山島舊圓形來源方位圖之方法與十類導引段；該段已移除）。主檔之表格、附錄、頁首頁尾皆無 9.56 或 0.13，故無「三值加總」之表。候選：無 9.56、無 0.13；龜山島 9.57% 僅出現於圖 23 導引段（1 次）與整體結論 (2)（1 次），各段只出現一次。\n')

B8 = blocks('/tmp/v8b_cand.docx')
w('\n## 七、第三輪修訂（v8c；相對 v8b，sha ef26da12…）\n')
w('1. 海岸位置詞：括號跨研究地點兩側者不稱方位，改稱「研究地點周邊臺灣本島海岸」。判定：括號含同一海岸環上距研究地點最近之海岸段且該段不在括號端點（兩端分居其兩側），或兩端點方位夾角 ≥ 150°；且括號最近處距研究地點 ≤ 10 km。套用於全年、季節、大小潮之全部海岸括號與整體結論，檢查 N19 以獨立程式重算。')
w('   變更共 3 處（全年括號與整體結論無變動：貢寮 16.90%、6.10% 之括號各在研究地點一側，南灣全年已為「研究地點周邊恆春半島南端海岸」）：')
w('   - 貢寮 圖 16 春季（24.08%）：研究地點南側臺灣本島海岸 → 研究地點周邊臺灣本島海岸')
w('   - 貢寮 圖 16 夏季（32.21%）：研究地點西南側臺灣本島海岸 → 研究地點周邊臺灣本島海岸')
w('   - 南灣 圖 49 春季（34.88%）：研究地點北側臺灣本島海岸 → 研究地點周邊臺灣本島海岸')
w('   未變者：新竹外海 秋季 5.79%（含最近海岸段但距研究地點約 32 km，超過 10 km，維持「研究地點東南側臺灣本島海岸」）；龜山島臺灣本島括號距研究地點 ≥ 13 km 且不跨兩側，維持西南側；其餘皆為單側括號或地名。\n')
w('2. 新增 40 個圖說段加 keepLines（避免圖說跨頁）。主檔既有 42 個圖說段皆無 keepLines（2 個表格圖說有 keepLines 與 keepNext）；保留之 22 個圖說維持與主檔相同，未加。檢查 K7。LO 渲染：55 個（二）圖說皆未跨頁。\n')
n7 = 0
for i, (a, b) in enumerate(zip(B8, Cn)):
    if a != b:
        n7 += 1; w('### 候選區塊 %d\n\n- v8b：%s\n- v8c：%s\n' % (i, a, b))
w('v8b→v8c 文字有變動之區塊共 %d 個（圖說之 keepLines 為格式變更，不在此列）。\n' % n7)
open(os.path.join(HERE, 'v8_change_list.md'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('ok', len(out))
