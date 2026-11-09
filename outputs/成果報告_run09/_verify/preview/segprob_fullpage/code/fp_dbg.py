import sys, time
sys.path.insert(0, '.')
import fp_final as F
import fp_lib as L0
for site in sys.argv[1:]:
    t0 = time.time()
    S = L0.prep_site(site)
    for kind in ('boundary', 'coast'):
        c, rows = F.select_corner(S, kind)
        print(site, kind, 'best', c, f'{time.time()-t0:.0f}s')
        for r in rows: print('  ', r)
        for name in ('class','season','tide'):
            cnts, bw, bh = F.group_info(S, kind, name)
            print('   box', name, round(bw,2), round(bh,2))
