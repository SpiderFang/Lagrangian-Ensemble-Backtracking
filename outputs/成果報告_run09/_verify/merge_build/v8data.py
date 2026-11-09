# -*- coding: utf-8 -*-
import json, csv, re, math
import os
B = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preview', 'segprob_v3') + os.sep
m = json.load(open(B + 'manifest.json'))
SITES = ['gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang']
SN = {'gongliao': '貢寮', 'guishan': '龜山島', 'hsinchu': '新竹外海', 'nanwan': '南灣', 'lienchiang': '連江'}
ANCHOR = {'gongliao': (121.92, 25.10), 'guishan': (121.94, 24.85), 'hsinchu': (120.45, 24.75), 'nanwan': (120.77, 21.92), 'lienchiang': (119.95, 26.20)}
SEASONS = ['冬季', '春季', '夏季', '秋季']
CIRC = '①②③④⑤'
seg = {s: {r['segment_id']: r for r in csv.DictReader(open(B + 'tables/boundary_segments_%s.csv' % s, encoding='utf-8-sig'))} for s in SITES}
cseg = {s: {r['segment_id']: r for r in csv.DictReader(open(B + 'tables/coast_segments_%s.csv' % s, encoding='utf-8-sig'))} for s in SITES}
SIDE = {'N': '北側', 'S': '南側', 'E': '東側', 'W': '西側'}
CORNER = {frozenset('NE'): '東北角', frozenset('NW'): '西北角', frozenset('SW'): '西南角', frozenset('SE'): '東南角'}

def pieces(s, ids):
    out = []
    cur = None
    for i in ids:
        r = seg[s][i]; sd = r['side']
        xs = [float(r['start_lon']), float(r['end_lon'])]; ys = [float(r['start_lat']), float(r['end_lat'])]
        if cur and cur[0] == sd:
            cur[1] += xs; cur[2] += ys
        else:
            cur = [sd, list(xs), list(ys)]; out.append(cur)
    res = []
    for sd, xs, ys in out:
        v = (min(xs), max(xs)) if sd in 'NS' else (min(ys), max(ys))
        res.append((sd, v[0], v[1]))
    return res

def bdesc(s, ids):
    """邊界括號之位置描述（1 位小數）。"""
    p = pieces(s, ids)
    big = [x for x in p if x[2] - x[1] > 0.12]
    if len(big) == 1 and len(p) > 1:
        p = big   # 轉角處之極短一段併入（不單獨描述）
    def one(x):
        sd, lo, hi = x
        return '%s邊界約 %.1f–%.1f°%s 段' % (SIDE[sd], lo, hi, 'E' if sd in 'NS' else 'N')
    if len(p) == 1:
        return one(p[0])
    if len(p) == 2:
        c = CORNER[frozenset(p[0][0] + p[1][0])]
        return '%s（%s與%s）' % (c, one(p[0])[:-2], one(p[1])[:-2])
    raise ValueError((s, ids, p))

def panel_values(title):
    lines = title.split('\n')
    head = lines[0]
    tot = float(re.search(r'([\d.]+)%', lines[1]).group(1))
    items = {}
    rest = None
    for ln in lines[2:]:
        for mk, v in re.findall(r'([①②③④⑤])\s*([\d.]+)%', ln): items[mk] = float(v)
        r = re.search(r'其餘\s*([\d.]+)%', ln)
        if r: rest = float(r.group(1))
    return head, tot, items, rest

def fig_panels(s, fig):
    f = m['figures'][s][fig]
    anns = f['annotations']; anns = [anns] if isinstance(anns, dict) else anns
    titles = f['panel_titles']
    pv = [panel_values(t) for t in titles]
    ann_iter = iter([a for a in anns if a['stretches']])
    out = []
    for (head, tot, items, rest) in pv:
        br = []
        if items:
            a = next(ann_iter)
            sts = [x for x in a['stretches'] if x['marker']]
            sts.sort(key=lambda x: CIRC.index(x['marker']))
            br = sts
            assert len(sts) == len(items), (s, fig, head, len(sts), len(items))
        out.append({'head': head, 'total': tot, 'items': items, 'rest': rest, 'brackets': br})
    return out

def bearing8(dx, dy):
    ang = math.degrees(math.atan2(dx, dy)) % 360
    names = ['北', '東北', '東', '東南', '南', '西南', '西', '西北']
    return names[int((ang + 22.5) // 45) % 8]

def cdesc(s, st):
    """海岸括號：位置（相對研究地點錨點）與座標範圍。"""
    ids = st['segments']
    pts = []
    for i in ids:
        r = cseg[s][i]; pts += [(float(r['start_lon']), float(r['start_lat'])), (float(r['end_lon']), float(r['end_lat']))]
    lo = (min(p[0] for p in pts), max(p[0] for p in pts)); la = (min(p[1] for p in pts), max(p[1] for p in pts))
    ax, ay = ANCHOR[s]
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    dkm = math.hypot((cx - ax) * 111.32 * math.cos(math.radians(ay)), (cy - ay) * 110.57)
    return {'lon': lo, 'lat': la, 'dir': bearing8(cx - ax, cy - ay), 'dkm': dkm, 'cls': cseg[s][ids[0]]['class'], 'island': cseg[s][ids[0]]['island_name']}

if __name__ == '__main__':
    for s in SITES:
        print('=====', SN[s])
        for fig in ['boundary_all']:
            a = m['figures'][s][fig]['annotations']['stretches']
            for st in a: print(fig, st['style'], st['shown_pct'], bdesc(s, st['segments']))
        for fig in ['boundary_season', 'boundary_tide']:
            for p in fig_panels(s, fig):
                print(fig, p['head'], p['total'], p['rest'], [(k, v, bdesc(s, b['segments'])) for (k, v), b in zip(p['items'].items(), p['brackets'])])
        for fig in ['coast_all', 'coast_season', 'coast_tide']:
            if fig == 'coast_all':
                a = m['figures'][s][fig]['annotations']['stretches']
                for st in a: print(fig, st['pct'], cdesc(s, st))
                inset = m['figures'][s][fig].get('inset')
                if inset:
                    for st in inset['annotations']['stretches']: print(fig, 'inset', st['pct'], cdesc(s, st))
            else:
                for p in fig_panels(s, fig):
                    print(fig, p['head'], p['total'], [(k, v, cdesc(s, b)['dir'], round(cdesc(s, b)['dkm'], 1), cdesc(s, b)['cls'], cdesc(s, b)['island']) for (k, v), b in zip(p['items'].items(), p['brackets'])])
