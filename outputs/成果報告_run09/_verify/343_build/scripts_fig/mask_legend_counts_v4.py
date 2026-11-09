#!/usr/bin/env python3
"""mask_legend_counts_v4.py  (2026-10-21)
由 figs_arrival_zh_v3 的 wp61_<site>_overview_S_notext.png 以像素遮罩方式，
去掉圖例標籤中全形括號及其內計數（例如「大潮（24）」→「大潮」），輸出到 figs_arrival_zh_v4。
.203 上的 arrival_figs.py 仍會寫出計數；日後重繪需同步修改該腳本。

定位方式：圖例字型 9 pt、300 dpi -> 1 em = 9*300/72 = 37.5 px；
標籤起點 x_label（SVG: 28.98 pt 左欄 / 249.7812 pt 右欄）後第 N 個 em 起為「（」。
填色範圍：x 從 ceil(x_label + N*em) 到該列最後一個非白像素 +2；
y 為該列在此 x 範圍內非白像素之上下界 ±1。填入 (255,255,255,255)。
"""
import sys, os, math, hashlib
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

BASE = os.path.dirname(os.path.abspath(__file__))
B = os.path.normpath(os.path.join(BASE, '..'))
SRC = os.path.join(B, 'figs_arrival_zh_v3')
DST = os.path.join(B, 'figs_arrival_zh_v4')
SITES = ['gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang']

S = 300 / 72.0          # px per pt
EM = 9 * S              # 37.5 px
PITCH = 10.62 * S       # 44.25 px
Y0 = 371.25 * S         # first-row centre (marker y in SVG)
# (label x [pt], row index, number of label characters before 「（」, label)
ROWS = [
    (28.98, 0, 2, '大潮（24）'), (28.98, 1, 2, '小潮（24）'),
    (28.98, 2, 4, '漲潮最快（16）'), (28.98, 3, 4, '退潮最快（16）'),
    (28.98, 4, 3, '近平潮（16）'),
    (249.7812, 0, 4, '高浪事件（1）'), (249.7812, 1, 4, '強流事件（1）'),
]
XMAX_LEFT = 900         # left column text never extends beyond this


def nonwhite(a):
    return (a[:, :, :3] != 255).any(axis=2)


def process(site):
    src = os.path.join(SRC, site, f'wp61_{site}_overview_S_notext.png')
    dst = os.path.join(DST, site, f'wp61_{site}_overview_S_notext.png')
    im = Image.open(src)
    assert im.mode == 'RGBA'
    a = np.array(im)
    assert (a[1500:, :, 3] == 255).all()
    nw = nonwhite(a)
    out = a.copy()
    boxes = []
    for x_pt, r, n, label in ROWS:
        xs = x_pt * S
        x0 = int(math.ceil(xs + n * EM))
        c = Y0 + r * PITCH
        ya, yb = int(c - PITCH / 2 + 1), int(c + PITCH / 2 - 1)
        xr = XMAX_LEFT if xs < 500 else a.shape[1]
        sub = nw[ya:yb, x0:xr]
        cols = np.where(sub.any(axis=0))[0]
        assert cols.size, (site, label, 'no ink found')
        x1 = x0 + int(cols.max()) + 2 + 1          # exclusive, +2 px margin
        sub2 = nw[ya:yb, x0:x1]
        rows = np.where(sub2.any(axis=1))[0]
        y_top, y_bot = ya + int(rows.min()) - 1, ya + int(rows.max()) + 2   # exclusive
        # sanity: ink must start at least ~0.3 em after x0 (left bearing of 「（」)
        assert cols.min() > 0.2 * EM, (site, label, 'paren ink starts too early', cols.min())
        # sanity: the character ink immediately left of x0 must not be touched
        out[y_top:y_bot, x0:x1] = 255
        boxes.append((x0, y_top, x1, y_bot, label))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    info = PngInfo()
    for k, v in im.text.items() if hasattr(im, 'text') else []:
        info.add_text(k, v)
    Image.fromarray(out, 'RGBA').save(dst, pnginfo=info, dpi=im.info.get('dpi'))
    # ---- verification ----
    b = np.array(Image.open(dst))
    assert b.shape == a.shape
    diff = (a != b).any(axis=2)
    allowed = np.zeros_like(diff)
    for x0, y0, x1, y1, _ in boxes:
        allowed[y0:y1, x0:x1] = True
    assert not (diff & ~allowed).any(), 'diff outside boxes'
    for x0, y0, x1, y1, lab in boxes:
        assert diff[y0:y1, x0:x1].any(), (lab, 'nothing changed')
        # no ink must remain right of x0 in the row band
        assert not nonwhite(b)[y0:y1, x0:XMAX_LEFT if x0 < 500 else a.shape[1]].any(), (lab, 'remnant')
    im2 = Image.open(dst)
    assert im2.size == im.size and im2.info.get('dpi') == im.info.get('dpi')
    return dst, boxes, int(diff.sum())


if __name__ == '__main__':
    lines = []
    for s in SITES:
        dst, boxes, nd = process(s)
        sha = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
        print(s, 'changed px', nd, sha)
        for bx in boxes:
            print('   box x[%d,%d) y[%d,%d)  %s' % (bx[0], bx[2], bx[1], bx[3], bx[4]))
        lines.append(f'{sha}  {s}/wp61_{s}_overview_S_notext.png')
    open(os.path.join(DST, 'sha256.txt'), 'w').write('\n'.join(lines) + '\n')
