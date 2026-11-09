import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from docxlib import *
R = os.path.expanduser('~/mnt/Lagrangian-Ensemble-Backtracking/outputs/成果報告_run09')
name = '執行成果_run09_含斯托克斯漂流'
br = parse_md(R + f'/{name}.md')
m = Math(); m.collect(br); m.convert()
res = Profile('res', 28, 9026, [700, 7626, 700], 360)
body, imgs = render(br, res, m, R)
build_package(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res_template'), R + f'/{name}.docx', body, imgs)
print('figs', len(imgs), name)
