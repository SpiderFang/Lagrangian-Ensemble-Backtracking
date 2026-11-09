"""WP-71：每個圖種五地點並排對照圖（預覽用，不進報告）。輸出 _verify/preview/20261022d/compare_<圖種>.jpg（3 欄 × 2 列，左上起依貢寮、龜山島、新竹外海、南灣、連江）。"""
import os, sys
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
R = os.path.expanduser('~/mnt/Lagrangian-Ensemble-Backtracking/outputs/成果報告_run09')
OUTD = sys.argv[1] if len(sys.argv) > 1 else f'{R}/_verify/preview/20261022d'
BASE = sys.argv[2] if len(sys.argv) > 2 else f'{R}/figures/finite_depth_stokes/無文字'
TAG = sys.argv[3] if len(sys.argv) > 3 else 'compare'
os.makedirs(OUTD, exist_ok=True)
SITES = ['gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang']
STEMS = ['horizontal_overview', 'domain_fraction_vs_backtrack_time', 'source_footprint_residence', 'polar_source_v2_by_class', 'polar_source_v2_by_season', 'polar_source_v2_by_tide', 'polar_source_v2_velocity_panels']
for st in STEMS:
    ims = [Image.open(f'{BASE}/{s}/{st}.png').convert('RGB') for s in SITES]
    w = 900 if st in ('horizontal_overview', 'source_footprint_residence', 'domain_fraction_vs_backtrack_time') else 640
    tiles = []
    for s, im in zip(SITES, ims):
        h = round(im.height * w / im.width); t = im.resize((w, h), Image.LANCZOS)
        c = Image.new('RGB', (w + 8, h + 26), 'white'); c.paste(t, (4, 22)); ImageDraw.Draw(c).text((6, 4), s + f'  {im.width}x{im.height}', fill='black'); tiles.append(c)
    th = max(t.height for t in tiles); tw = tiles[0].width
    sheet = Image.new('RGB', (tw * 3, th * 2), (200, 200, 200))
    for i, t in enumerate(tiles): sheet.paste(t, ((i % 3) * tw, (i // 3) * th))
    sheet.save(f'{OUTD}/{TAG}_{st}.jpg', quality=88)
print('done')
