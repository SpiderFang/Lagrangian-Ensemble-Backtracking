import sys
sys.path.insert(0, '.')
import numpy as np
import fp_final as F, fp_lib4 as L4, fp_lib as L0
from PIL import Image
site, kind, name = sys.argv[1:4]
S = L0.prep_site(site)
cnts, bw, bh = F.group_info(S, kind, name)
w = F.W_REF[name]; h = w / F.RATIO
st = L4.static_occ(S, kind, w, h); d = L4.seg_dist(S, kind, cnts, w, h, name)
nx, ny = L4._dims(w, h)
xs, ys = (np.arange(nx) + .5) * L4.GRID, (np.arange(ny) + .5) * L4.GRID
X, Y = np.meshgrid(xs, ys)
frame = (X < .04) | (X > w - .04) | (Y < .04) | (Y > h - .04)
for m in (0.035, 0.028, 0.022, 0.0):
    occ = st | frame | (d < m)
    # 最大可放寬度（對各高度）
    best = []
    for hh in (0.3, 0.4, 0.5, 0.56, 0.68):
        for ww in np.arange(1.2, 0.2, -0.01):
            ok, _, _ = L4.free_positions(occ, ww, hh)
            if ok.any(): best.append((hh, round(float(ww), 2))); break
    print(site, kind, name, 'm', m, 'box', round(bw,2), round(bh,2), 'max width by height', best)
img = np.zeros((ny, nx, 3), np.uint8) + 255
img[st] = (200, 200, 200); img[d < 0.1] = (255, 150, 0); img[d < 0.03] = (200, 0, 0)
Image.fromarray(img[::-1]).resize((nx * 4, ny * 4), Image.NEAREST).save(f'dbg_{site}_{kind}_{name}.png')
