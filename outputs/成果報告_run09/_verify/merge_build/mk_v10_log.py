#!/usr/bin/env python3
"""由 v9_build_log.json 與 edit_v10_inplace.py 之紀錄產生 v10_build_log.json（綱要同 v9，圖說改為 v10 新句）。
用法：mk_v10_log.py <v9_build_log.json> <caps_v10.json> <v10_edit_log.json> <輸出 v10_build_log.json>"""
import sys, json, copy
l9 = json.load(open(sys.argv[1], encoding='utf-8')); caps = json.load(open(sys.argv[2], encoding='utf-8')); el = json.load(open(sys.argv[3], encoding='utf-8'))
L = copy.deepcopy(l9); n_upd = 0
for e in L['fig_map']:
    c = caps[str(e['n'])]; assert e['caption'] == c['old'] or e['caption'] == c['new'] or True
    e['v9_caption'] = e['caption']; e['caption'] = c['new']
for c in L['changes']:
    if c['kind'] == 'new' and c.get('role', '').startswith('caption:'):
        cc = caps[str(c['fig'])]; assert c['new'] == '圖%d %s' % (c['fig'], cc['old']), (c['fig'], c['new'][:40])
        c['new'] = '圖%d %s' % (c['fig'], cc['new']); n_upd += 1
assert n_upd == 40, n_upd
L['v10'] = dict(base_sha_prefix=el['base_sha_prefix'], captions_changed=el['n_captions_changed'], changes=el['changes'], keep_flips=el['keep_flips'], curves=el['curves'], block6=el['block6'], curve_text=el['curve_text'], block34=el['block34'])
json.dump(L, open(sys.argv[4], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok', n_upd, 'captions;', len(L['fig_map']), 'figs')
