import sys
sys.path.insert(0, '.')
import numpy as np
import fp_final as F, fp_lib4 as L4, fp_lib as L0, fp_run as FR
site, kind, name = sys.argv[1:4]; HH = float(sys.argv[4])
S = L0.prep_site(site); FR.FONT = S.font.get_file()
orig = L4.place_box
def pb(S_, kind_, cnts, w, h, bw, bh, corner=None, margins=L4.MARGINS, extra=None, tag="", corners=None):
    r0 = orig(S_, kind_, cnts, w, h, bw, bh, corner=corner, margins=margins, extra=None, tag=tag)
    r1 = orig(S_, kind_, cnts, w, h, bw, bh, corner=corner, margins=margins, extra=extra, tag=tag)
    print(f"  place_box w={w:.3f} h={h:.3f} bw={bw:.3f} bh={bh:.3f} corner={corner} noextra={r0} extra={r1} extra_frac={None if extra is None else float(extra.mean()):.3f}", flush=True)
    if r1 is None and corner == 'UR' and abs(w-1.743)<0.01:
        nx, ny = L4._dims(w, h)
        xs, ys = (np.arange(nx) + .5) * L4.GRID, (np.arange(ny) + .5) * L4.GRID
        X, Y = np.meshgrid(xs, ys)
        frame = (X < .04) | (X > w - .04) | (Y < .04) | (Y > h - .04)
        dsg = L4.seg_dist(S_, kind_, cnts, w, h, tag)
        from PIL import Image
        stt = L4.static_occ(S_, kind_, w, h, None)
        im = np.full((ny, nx, 3), 255, np.uint8)
        im[stt] = (200, 200, 200); im[extra] = (120, 120, 255); im[dsg < 0.012] = (255, 0, 0); im[frame] = (0, 0, 0)
        Image.fromarray(im[::-1]).resize((nx * 3, ny * 3), Image.NEAREST).save('dbg_occ_%s_%s_%s.png' % (S_.sd.site_id, kind_, tag))
        for tier, lma in enumerate((None, L4.LAND_MIN_AREA)):
            base = L4.static_occ(S_, kind_, w, h, lma) | frame | extra
            for m in (0.035, 0.022, 0.012):
                occ = base | (dsg < m)
                fr = []
                for hh in (0.2, 0.28, 0.36, 0.40, 0.45, 0.5, 0.6):
                    for ww in np.arange(1.5, 0.2, -0.01):
                        ok, _, _ = L4.free_positions(occ, ww, hh)
                        if ok.any(): fr.append((hh, round(float(ww), 2))); break
                print('   FR tier', tier, 'm', m, fr, flush=True)
    return r1
L4.place_box = pb
nr, nc, mode, sh = F.LAYOUT[name]
for c in ('UR',):
    try:
        L4.render_fit3(S, kind, name, f"/tmp/dbg5_{c}.png", nrow=nr, ncol=nc, H=HH, mode=mode, corner=c, class_brackets=(kind=="boundary" and name=="class"), shared_titles=sh)
        print(c, "ok")
    except RuntimeError as e:
        print(c, "ERR", e)
