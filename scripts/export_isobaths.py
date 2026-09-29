#!/usr/bin/env python
"""WP-38：由 SCHISM／OCM 標準化子網格匯出等深線 GeoJSON，供 ``native-figures --bathymetry`` 使用。

水深來源是各 flow domain 快取目錄下 ``grid/`` 的節點水深（``source_depth_m.npy``）與三角形
拓撲，以三角網格線性內插後等值化（預設 50／100／200／1000 公尺）。輸出為 CRS84
``FeatureCollection``，每個 feature 是某站某水深的 ``MultiLineString``，屬性為
``site_id`` 與 ``depth_m``；該水深沒有出現在某站網格範圍內時不輸出（例如新竹外海最深
約 105 公尺，沒有 200 公尺等深線）。

指定網格的兩種方式（擇一）：

1. ``--site-grid 站點=網格目錄``（可重複）；
2. ``--identity-table`` ＋ ``--region-flow-domain-map`` ＋ ``--grid-root``：由粒子身分表的
   站點與分析區，經「分析區 -> flow domain 快取目錄」對照求出每站的 ``<grid-root>/<目錄>/grid``。

只讀網格陣列，不改寫任何輸入；輸出檔已存在時拒絕覆寫。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lagrangian_backtracking.native_runner import bathymetry


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="由 SCHISM／OCM 網格匯出等深線 GeoJSON")
    parser.add_argument(
        "--site-grid", action="append", default=[], metavar="SITE=DIR", help="站點=網格目錄（可重複）"
    )
    parser.add_argument("--identity-table", type=Path, default=None, help="粒子身分表 npz（求站點與分析區）")
    parser.add_argument(
        "--region-flow-domain-map", type=Path, default=None, help="分析區 -> flow domain 目錄名稱 JSON"
    )
    parser.add_argument("--grid-root", type=Path, default=None, help="各 flow domain 快取目錄所在的上層目錄")
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=list(bathymetry.DEFAULT_ISOBATH_LEVELS_M),
        help="等深線水深（公尺），預設 50 100 200 1000",
    )
    parser.add_argument("--output", type=Path, required=True, help="輸出 GeoJSON 路徑（必須尚不存在）")
    return parser.parse_args(argv)


def _site_grid_dirs(args: argparse.Namespace) -> dict[str, Path]:
    if args.site_grid:
        result: dict[str, Path] = {}
        for item in args.site_grid:
            site, sep, directory = item.partition("=")
            if not sep or not site or not directory:
                raise SystemExit(f"--site-grid 格式必須是 站點=目錄，收到：{item!r}")
            result[site] = Path(directory)
        return result
    if args.identity_table is None or args.region_flow_domain_map is None or args.grid_root is None:
        raise SystemExit(
            "請提供 --site-grid，或同時提供 --identity-table、--region-flow-domain-map、--grid-root"
        )
    from lagrangian_backtracking.native_runner.particle_input import load_particle_identity_table_npz

    identity = load_particle_identity_table_npz(args.identity_table)
    site_to_region = dict(
        zip(identity.study_site_id.tolist(), identity.analysis_region_id.tolist(), strict=True)
    )
    region_map = json.loads(args.region_flow_domain_map.read_text(encoding="utf-8"))
    return bathymetry.resolve_site_grid_dirs(site_to_region, region_map, args.grid_root)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"輸出檔已存在，拒絕覆寫：{args.output}")
    grid_dirs = _site_grid_dirs(args)
    loaded: dict[Path, bathymetry.BathymetryField] = {}
    fields: dict[str, bathymetry.BathymetryField] = {}
    for site, directory in sorted(grid_dirs.items()):
        if directory not in loaded:
            loaded[directory] = bathymetry.load_ocm_grid_bathymetry(directory)
        fields[site] = loaded[directory]
    payload = bathymetry.build_isobath_geojson(fields, args.levels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bathymetry.dump_geojson(payload, args.output)
    counts: dict[str, int] = {}
    for feature in payload["features"]:
        key = f"{feature['properties']['site_id']}:{feature['properties']['depth_m']:g}"
        counts[key] = len(feature["geometry"]["coordinates"])
    print(
        json.dumps(
            {"output": str(args.output), "features": len(payload["features"]), "segments": counts},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
