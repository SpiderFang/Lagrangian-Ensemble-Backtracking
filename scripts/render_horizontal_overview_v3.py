"""run09 水平軌跡總覽 v3 版面（WP-71：座標軸統一、不畫格線）重畫腳本。

只重畫 horizontal_overview（--figure-text none），其餘輸入與參數與
finite_depth_stokes/figures-notext 的命令相同；輸出到全新目錄，不覆寫舊圖。
用法：python scripts/render_horizontal_overview_v3.py <輸出目錄>
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from lagrangian_backtracking.native_runner import figures as F

HOME = Path.home()
RUN = HOME / "Workspace/lbt-runs/run09-h180-m10-v8"
RELEASE = (
    HOME
    / "Workspace/lbt-inputs/formal/v8-run09/20261019_final-sites_h180_common_m10_v8"
    / "release-configs/release-180d-full_horizon_from_deposition.yaml"
)


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    config = F.NativeFiguresConfig(
        postprocess_dir=RUN / "finite_depth_stokes/postprocess",
        coastline_path=HOME / "Workspace/lbt-inputs/coastline-taiwan.geojson",
        output_dir=out,
        max_trajectories_per_site=24,
        dpi=600,
        figure_text="none",
        bathymetry_path=RUN / "isobaths.geojson",
        receptor_config_path=RELEASE,
    )
    command = shlex.join(
        [
            "python",
            "scripts/render_horizontal_overview_v3.py",
            str(out),
            "(--postprocess-dir",
            str(config.postprocess_dir),
            "--coastline",
            str(config.coastline_path),
            "--bathymetry",
            str(config.bathymetry_path),
            "--config",
            str(RELEASE),
            "--figure-text none --dpi 600 --max-trajectories-per-site 24)",
        ]
    )
    F.build_horizontal_overview_v2(config, command=command, layout="v3")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
