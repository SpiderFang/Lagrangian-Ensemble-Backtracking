"""WP-73：報告圖字級與版面與 TRAPs 一致（fs8）——軌跡總覽、條件式來源足跡的重畫腳本（方法 B 資料、全新輸出目錄）。

用法：python scripts/wp73_deploy/render_fs8.py traj|fp|traj_old|fp_old <輸出目錄>
  traj_old／fp_old：以舊版面畫同一組圖，只供「數值未變」的資料指紋比對（不是報告用圖）。
  traj：水平軌跡總覽（layout v4）；fp：條件式來源足跡圖（停留時數密度，1 km 網格、色階 B1、座標軸統一）。
其餘參數與 wp72 的 coastB 版相同（同一份 postprocess-coastB、同一份海岸線與等深線）。
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
    kind, out = argv[0], Path(argv[1])
    config = F.NativeFiguresConfig(
        postprocess_dir=RUN / "finite_depth_stokes/postprocess-coastB",
        coastline_path=HOME / "Workspace/lbt-inputs/coastline-taiwan.geojson",
        output_dir=out,
        max_trajectories_per_site=24,
        dpi=100 if kind.endswith("_old") else 600,
        figure_text="none",
        bathymetry_path=RUN / "isobaths.geojson",
        receptor_config_path=RELEASE,
        **(
            {"footprint_colorbar_range": (1e-3, 1e3), "footprint_colormap_scheme": "nonuniform", "axes_unified": True}
            if kind in ("fp", "fp_old")
            else {}
        ),
    )
    command = shlex.join(
        ["python", "scripts/wp73_deploy/render_fs8.py", kind, str(out), "(--postprocess-dir", str(config.postprocess_dir),
         "--coastline", str(config.coastline_path), "--bathymetry", str(config.bathymetry_path), "--config", str(RELEASE),
         f"--figure-text none --dpi {config.dpi}" + (" --footprint-colorbar-range 1e-3 1e3 --footprint-colormap-scheme nonuniform --axes-unified)" if kind in ("fp", "fp_old") else " --max-trajectories-per-site 24)")]
    )
    if kind == "traj":
        F.build_horizontal_overview_v2(config, command=command, layout="v4")
    elif kind == "traj_old":
        F.build_horizontal_overview_v2(config, command=command, layout="v3")
    elif kind == "fp":
        F.build_footprint_fs8(config, command=command)
    elif kind == "fp_old":
        F.build_footprint_fs8(config, command=command, report_layout=False)
    else:
        raise SystemExit("kind 必須是 traj、fp、traj_old 或 fp_old")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
