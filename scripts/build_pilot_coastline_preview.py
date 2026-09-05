"""由已完成的 B 區 r2 preview 獨立重繪 legacy 或新版成果圖及來源清單。

預設 ``legacy`` 只輸出原契約的兩張水平圖；指定 ``--style baytrace`` 才增加區域／
局部水平、垂向高度及停止原因四張新版圖，並各自保存對應的繁中 README 與 manifest。

本模組只讀既有 preview 目錄中的
``manifest.json``、``summary.json``、``particles.csv`` 與 ``observations.csv``，
以及呼叫端明示的已驗收 B 區 domain/open-boundary manifest 和 CRS84 海岸
FeatureCollection。它不讀海流或波浪陣列、不重新取樣、不重新積分，也不改寫任何
輸入檔。回傳資料仍保留 CSV 的全部欄位與列順序，供後續繪圖階段使用原始 20 顆
粒子、203 筆觀測及 summary 的 H1--H5 順序。

domain/open manifest 的語意雜湊沿用專案既有 ``manifests._read_json``；B 區 domain
與 ``hsinchu_cache_v3`` flow open owner 必須同時符合 preview summary 的
``geometry_canonical_hashes``。海岸資料只作地理參考線，不被當成流場有效網格或
粒子遮罩。命令列只建立全新輸出目錄；來源前後 SHA-256 必須相同，PNG、README
與 JSON 清單皆以排他建立方式落檔。失敗保留新目錄供診斷，不清理或覆寫原始結果。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, MultiPolygon, Polygon, shape

from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.manifests import _read_json
from lagrangian_backtracking.outputs import sha256_file


def load_plot_inputs(
    preview_dir: str | Path,
    domain_path: str | Path,
    open_path: str | Path,
    coastline_path: str | Path,
) -> dict[str, Any]:
    """讀取並核對海岸版圖面輸入，回傳不含重新取樣資料的唯讀組合。

    ``preview_dir`` 必須是既有 ``pilot-preview-v1`` 目錄；manifest 宣告的 summary、
    particles 與 observations bytes 會在讀 CSV 前逐一核對。CSV 不做排序、插值或
    欄位刪減，列中的文字值也原樣保留，讓後續繪圖仍可追溯到原 preview。

    ``domain_path`` 與 ``open_path`` 必須是 schema 1.0.0、approved 的 JSON。只選取
    analysis region ``B`` 的 domain record 及其 ``flow_domain`` open record，並以
    summary 的 canonical geometry hash 綁定；open line 還必須與 domain polygon
    exterior 相等，避免把 A 區或 local record 誤畫成 B 區外框。海岸 GeoJSON 必須
    明示 OGC CRS84（座標順序為 longitude、latitude），每個 land feature 轉為
    Shapely Polygon/MultiPolygon。載入器不繪圖；繪圖函式可填灰色背景，但不以陸地
    判定粒子是否有效。

    回傳包含原始 payload、CSV 全列、B 區 Shapely 幾何、以 summary 中心建立的
    ``DomainProjection`` 與原始／語意 SHA。CSV 附 bytes 計數，完整檔案大小由 build
    wrapper 記錄。此函式只讀檔；缺檔拋出 ``OSError``，內容或來源綁定不符拋出
    ``ValueError``。
    """

    preview = Path(preview_dir)
    manifest, manifest_sha, manifest_canonical, _ = _read_json(preview / "manifest.json")
    if manifest.get("artifact_kind") != "pilot-preview-v1":
        raise ValueError("preview artifact_kind 不符")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("preview manifest 缺少 files")

    source_files: dict[str, dict[str, Any]] = {}
    for name in ("summary.json", "particles.csv", "observations.csv"):
        contract = files.get(name)
        if not isinstance(contract, dict):
            raise ValueError(f"preview manifest 缺少 {name}")
        path = preview / name
        raw_size = path.stat().st_size
        raw_sha = sha256_file(path)
        if raw_size != contract.get("size_bytes") or raw_sha != contract.get("sha256"):
            raise ValueError(f"preview {name} bytes/checksum 不符")
        source_files[name] = {"size_bytes": raw_size, "sha256": raw_sha}

    summary, _, _, _ = _read_json(preview / "summary.json")
    if summary.get("artifact_kind") != "pilot-preview-v1" or summary.get("run_id") != manifest.get("run_id"):
        raise ValueError("preview summary 與 manifest 不符")
    if summary.get("particle_count") != 20 or summary.get("observation_count") != 203:
        raise ValueError("preview 必須是 20 粒子／203 觀測的既有 r2 結果")
    projection_info = summary.get("projection")
    hashes = projection_info.get("geometry_canonical_hashes") if isinstance(projection_info, dict) else None
    if not isinstance(hashes, dict):
        raise ValueError("preview summary 缺少 geometry canonical hashes")

    def read_csv(name: str) -> list[dict[str, str]]:
        """保留既有 CSV 全欄位與列順序，不讓補圖階段改變軌跡資料。"""

        with (preview / name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return rows

    particles = read_csv("particles.csv")
    observations = read_csv("observations.csv")
    if len(particles) != summary["particle_count"] or len(observations) != summary["observation_count"]:
        raise ValueError("preview CSV 計數與 summary 不符")

    domain, domain_sha, domain_canonical, _ = _read_json(domain_path)
    open_boundary, open_sha, open_canonical, _ = _read_json(open_path)
    if domain_canonical != hashes.get("domain") or open_canonical != hashes.get("open_boundary"):
        raise ValueError("domain/open canonical hash 與 preview summary 不符")
    if (
        domain.get("manifest_kind") != "domain_geometry_manifest"
        or domain.get("schema_version") != "1.0.0"
        or domain.get("coordinate_reference") != "EPSG:4326"
        or domain.get("status") != "approved"
    ):
        raise ValueError("domain manifest 不是 approved geometry")
    if (
        open_boundary.get("manifest_kind") != "open_boundary_manifest"
        or open_boundary.get("schema_version") != "1.0.0"
        or open_boundary.get("coordinate_reference") != "EPSG:4326"
        or open_boundary.get("status") != "approved"
    ):
        raise ValueError("open manifest 不是 approved geometry")

    region = projection_info.get("analysis_region_id")
    center = projection_info.get("center_lonlat")
    if region != "B" or not isinstance(center, list) or len(center) != 2:
        raise ValueError("preview projection 不是 B 區 AEQD")
    domain_rows = [row for row in domain.get("records", []) if row.get("analysis_region_id") == "B"]
    if len(domain_rows) != 1:
        raise ValueError("domain manifest 的 B record 不唯一")
    domain_record = domain_rows[0]
    flow_domain_id = domain_record.get("flow_domain_id")
    open_rows = [
        row
        for row in open_boundary.get("records", [])
        if row.get("owner_kind") == "flow_domain"
        and row.get("owner_id") == flow_domain_id
        and row.get("analysis_region_id") == "B"
    ]
    if len(open_rows) != 1:
        raise ValueError("open manifest 的 B flow owner 不唯一")
    open_record = open_rows[0]
    domain_geometry = shape(domain_record["geometry"])
    open_geometry = shape(open_record["geometry"])
    if not isinstance(domain_geometry, Polygon) or not domain_geometry.is_valid:
        raise ValueError("B domain geometry 不是有效 Polygon")
    if not isinstance(open_geometry, LineString) or not open_geometry.is_valid:
        raise ValueError("B flow open geometry 不是有效 LineString")
    if not open_geometry.equals(domain_geometry.exterior):
        raise ValueError("B flow open line 未與 domain exterior 相等")

    coastline, coastline_sha, coastline_canonical, _ = _read_json(coastline_path)
    # 本入口專用於使用者已確認的 r2 海岸檔，不以相同 feature 數量接受另一份地圖。
    if coastline_sha != "9e2e0ac9bc527aca87d89332cd428fdcb776eefbf94a85dd70f887f729b95fdd":
        raise ValueError("海岸檔 SHA-256 與本次已確認來源不符")
    crs_name = coastline.get("crs", {}).get("properties", {}).get("name")
    if coastline.get("type") != "FeatureCollection" or crs_name != "urn:ogc:def:crs:OGC:1.3:CRS84":
        raise ValueError("海岸 GeoJSON 必須明示 CRS84 FeatureCollection")
    land_geometries = [shape(feature["geometry"]) for feature in coastline.get("features", [])]
    if (
        len(land_geometries) != 1905
        or not all(isinstance(geometry, (Polygon, MultiPolygon)) for geometry in land_geometries)
        or not all(geometry.is_valid for geometry in land_geometries)
    ):
        raise ValueError("海岸 GeoJSON features 不符合既有 1905 筆有效 land polygons")

    return {
        "preview_manifest": manifest,
        "summary": summary,
        "particles": particles,
        "observations": observations,
        "domain_manifest": domain,
        "open_boundary_manifest": open_boundary,
        "domain_record": domain_record,
        "open_boundary_record": open_record,
        "domain_geometry": domain_geometry,
        "open_geometry": open_geometry,
        "land_geometries": land_geometries,
        "projection": DomainProjection(float(center[0]), float(center[1])),
        "source": {
            "preview_manifest": {"sha256": manifest_sha, "canonical_sha256": manifest_canonical},
            "preview_files": source_files,
            "domain": {"sha256": domain_sha, "canonical_sha256": domain_canonical},
            "open_boundary": {"sha256": open_sha, "canonical_sha256": open_canonical},
            "coastline": {"sha256": coastline_sha, "canonical_sha256": coastline_canonical},
        },
    }


_VERTICAL_COLORS = {
    "near_bed": "#2166ac",
    "mid_lower_water_column": "#f28e2b",
    "mid_upper_water_column": "#31a354",
    "upper_water_column": "#d73027",
}
"""四個既有垂向層的固定圖色；顏色只表示層別，不表示來源機率。"""

_STATUS_MARKERS = {
    "max_age": "^",
    "flow_domain_open_exit": "s",
    "numerical_failure": "X",
}
"""三種本次試跑終止狀態的幾何標記，避免把終點混稱為來源。"""

_VERTICAL_LABELS = {
    "near_bed": "近海床",
    "mid_lower_water_column": "中下水層",
    "mid_upper_water_column": "中上水層",
    "upper_water_column": "上水層",
}
"""初始水層的中文圖例，與 CSV 層別及固定顏色一一對應。"""

_BAYTRACE_STYLE_VERSION = "1.0.0"
"""新版圖面呈現契約版本；legacy 分支不使用此版本欄位。"""

_BAYTRACE_VERTICAL_COLORS = {
    "near_bed": "#377eb8",
    "mid_lower_water_column": "#e69f00",
    "mid_upper_water_column": "#984ea3",
    "upper_water_column": "#00a6a6",
}
"""新版固定水層線色；刻意避開綠色回溯起點與紅色停止位置。"""

_BAYTRACE_VERTICAL_LABELS = {
    "near_bed": "近床水層",
    "mid_lower_water_column": "中下水層",
    "mid_upper_water_column": "中上水層",
    "upper_water_column": "上水層",
}
"""新版不以 L1--L4 縮寫呈現的完整水層名稱。"""

_BAYTRACE_VERTICAL_DEFINITIONS = {
    "near_bed": "最低有效 OCM 水層中心（不是海床面）",
    "mid_lower_water_column": "自海面向下總水深 70% 設定目標",
    "mid_upper_water_column": "自海面向下總水深 40% 設定目標",
    "upper_water_column": "自海面向下總水深 10% 設定目標",
}
"""由已核對的 normalized_config 語意整理出的新版水層說明；不在補圖時讀 run。"""

_BAYTRACE_INITIAL_COLOR = "#1a9850"
_BAYTRACE_FINAL_COLOR = "#d73027"
"""新版端點固定使用綠色起點與紅色終點，避免與水層線色混淆。"""

_TERMINAL_STATUSES = (
    "flow_domain_open_exit",
    "coast_contact",
    "surface_regime_exit",
    "deposited",
    "forcing_start",
    "data_gap",
    "max_age",
    "numerical_failure",
)
"""停止圖的固定八狀態順序；即使計數為零也必須保留。"""

_BAYTRACE_STATUS_LABELS = {
    "flow_domain_open_exit": "離開計算範圍",
    "coast_contact": "接觸海岸",
    "surface_regime_exit": "離開表面適用範圍",
    "deposited": "沉積",
    "forcing_start": "到達驅動資料起點",
    "data_gap": "資料缺口",
    "max_age": "完成1小時回溯",
    "numerical_failure": "數值停止",
}
"""新版停止中文名稱；數值失敗只有在 CSV／summary 雙重核對後才改成取樣失敗停止。"""

_BAYTRACE_STATUS_COLORS = {
    "flow_domain_open_exit": "#4c78a8",
    "coast_contact": "#f58518",
    "surface_regime_exit": "#e45756",
    "deposited": "#72b7b2",
    "forcing_start": "#b279a2",
    "data_gap": "#ff9da6",
    "max_age": "#54a24b",
    "numerical_failure": "#79706e",
}
"""停止圖分段色；此色只表示分類，終點標記仍一律使用紅色。"""


def _display_font():
    """沿用原預覽字型 helper，優先使用本機既有 Arial Unicode 正常筆重。"""

    from lagrangian_backtracking.pilot_preview import _plot_font

    candidate = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font, _, _ = _plot_font(candidate if candidate.is_file() else None)
    font.set_weight("normal")
    font.set_size(11)
    return font


def _subtitle(summary: dict[str, Any]) -> str:
    """由來源到達時刻與回溯上限組合副題；不把模型保存紀錄稱為實測觀測。"""

    arrival = datetime.fromisoformat(summary["arrival_utc"])
    horizon = float(summary["settings"]["horizon_seconds"])
    end = arrival - timedelta(seconds=horizon)
    return (
        f"{arrival:%Y-%m-%d %H:%M} → {end:%H:%M} UTC｜"
        f"回溯{horizon / 3600:g}小時｜{summary['particle_count']}顆粒子"
    )


def _legend_handles(levels: list[str], *, include_land: bool = False) -> list:
    """總覽與局部圖共用中文水層、起點、停止形狀及同一登錄外框圖例。"""

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=_VERTICAL_COLORS[level], linewidth=2, label=_VERTICAL_LABELS[level])
        for level in levels
    ]
    handles.append(
        Line2D(
            [], [], marker="o", color="#222222", markerfacecolor="white", linestyle="None", label="回溯起點"
        )
    )
    for status, label in (
        ("max_age", "達回溯 1 小時"),
        ("flow_domain_open_exit", "開放邊界離域"),
        ("numerical_failure", "數值停止"),
    ):
        handles.append(
            Line2D(
                [],
                [],
                marker=_STATUS_MARKERS[status],
                color="#222222",
                markerfacecolor="white",
                linestyle="None",
                label=label,
            )
        )
    handles.append(Line2D([], [], color="#174a70", linewidth=1.4, linestyle="--", label="登錄開放邊界"))
    if include_land:
        handles.append(Patch(facecolor="#c9c9c9", edgecolor="#858585", label="陸地"))
    return handles


def render_overview(data: dict[str, Any], output_path: str | Path) -> Path:
    """以已載入的 20 顆粒子與 203 筆觀測繪製單張經緯度總覽 PNG。

    每條線只連接同一 ``particle_id`` 的保存觀測；位置由既有
    ``DomainProjection.unproject`` 將 AEQD 公尺轉回經度、緯度，沒有重新取樣或
    重新積分。海岸 FeatureCollection 交給 Shapely 的 ``plot_polygon`` 處理，因而
    保留 Polygon/MultiPolygon 的孔洞；B 開放邊界只畫一次。土地置於底層，軌跡、起點
    圓點與依停止狀態區分的終點標記置於上層，圖面不以陸地遮蔽資料。

    ``output_path`` 必須是尚不存在的 PNG 路徑；函式只建立其父目錄並拒絕覆寫，回傳
    實際輸出路徑；包含失效符號連結也拒絕。PNG 以 ``open('xb')`` 排他開檔，
    交由 Matplotlib 寫入同一檔案代號，避免存在檢查與存圖之間發生覆寫競爭。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_line, plot_polygon

    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"總覽圖已存在，拒絕覆寫：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = data["summary"]
    projection = data["projection"]
    particles = data["particles"]
    observations = data["observations"]
    curves: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        curves[row["particle_id"]].append(row)
    for rows in curves.values():
        # age_seconds 是從到達時刻向回溯方向增加；排序只決定繪線方向，不改資料列。
        rows.sort(key=lambda row: float(row["age_seconds"]))

    font = _display_font()
    levels = list(summary["vertical_order"])
    colors = {level: _VERTICAL_COLORS.get(level, "#666666") for level in levels}
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    try:
        ax.set_facecolor("#eaf3f8")
        # 先畫真實海岸 land polygons；Shapely 會處理 MultiPolygon 與孔洞路徑。
        for land in data["land_geometries"]:
            plot_polygon(
                land,
                ax=ax,
                add_points=False,
                facecolor="#c9c9c9",
                edgecolor="#858585",
                linewidth=0.25,
                alpha=0.85,
                zorder=1,
            )
        # domain exterior 與 open line 已經在 loader 驗證相等，故只呈現 open line 一次。
        plot_line(
            data["open_geometry"],
            ax=ax,
            add_points=False,
            color="#174a70",
            linewidth=1.4,
            linestyle="--",
            zorder=2,
        )

        for particle in particles:
            rows = curves.get(particle["particle_id"], [])
            color = colors.get(particle["vertical_id"], "#666666")
            if rows:
                x_m = [float(row["x_m"]) for row in rows]
                y_m = [float(row["y_m"]) for row in rows]
                longitude, latitude = projection.unproject(x_m, y_m)
                # 線條只代表實際保存觀測之間的連接，沒有補入未保存的中間點。
                ax.plot(
                    longitude,
                    latitude,
                    color=color,
                    linewidth=0.9,
                    alpha=0.78,
                    zorder=4,
                )
                # age 最小的保存觀測是回溯起點；四層可能重合，仍逐粒子保留圓點。
                ax.plot(
                    longitude[0],
                    latitude[0],
                    marker="o",
                    markersize=4.8,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    linestyle="None",
                    zorder=5,
                )
            marker = _STATUS_MARKERS.get(particle["status"], "D")
            ax.plot(
                float(particle["longitude"]),
                float(particle["latitude"]),
                marker=marker,
                markersize=6.2,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.7,
                linestyle="None",
                zorder=6,
            )

        # 使用 summary 的原始水平受體順序，避免由座標排序重新定義 H1--H5。
        for panel in summary["horizontal_panels"]:
            ax.annotate(
                f"H{panel['panel']}",
                (float(panel["receptor_lon"]), float(panel["receptor_lat"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontproperties=font,
                fontsize=9,
                color="#17324d",
                zorder=7,
            )

        min_lon, min_lat, max_lon, max_lat = data["domain_geometry"].bounds
        lon_margin = max((max_lon - min_lon) * 0.04, 0.02)
        lat_margin = max((max_lat - min_lat) * 0.04, 0.02)
        ax.set_xlim(min_lon - lon_margin, max_lon + lon_margin)
        ax.set_ylim(min_lat - lat_margin, max_lat + lat_margin)
        center_lat = float(summary["projection"]["center_lonlat"][1])
        # 經緯度資料仍保留原數值；此 aspect 只補償同一緯度的一度經度較短的實距離。
        ax.set_aspect(1.0 / math.cos(math.radians(center_lat)), adjustable="box")
        ax.set_xlabel("經度 longitude (°)", fontproperties=font)
        ax.set_ylabel("緯度 latitude (°)", fontproperties=font)
        ax.grid(color="white", linewidth=0.7, alpha=0.8)

        ax.set_title("B 區沉降粒子反向追蹤", fontproperties=font, fontsize=18, pad=36)
        ax.text(
            0.5,
            1.025,
            _subtitle(summary),
            transform=ax.transAxes,
            ha="center",
            fontproperties=font,
            fontsize=11,
        )
        ax.legend(
            handles=_legend_handles(levels, include_land=True),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=3,
            prop=font,
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output


def render_local(data: dict[str, Any], output_path: str | Path) -> Path:
    """以原 AEQD 公尺座標繪製五個局部面板，第六格放共用圖例與簡短圖說。

    H1--H5 沿用 summary 的水平分組，每組四粒子。各組以第一顆粒子的初始保存 XY
    作共同原點，只平移座標；線段、終點及已投影的 B 外框接受同量平移，不重新取樣、
    旋轉或縮放位移。各面板 x/y 等比例，範圍由保存軌跡及終點決定，再留出閱讀邊距。
    外框只畫與該範圍相交的線段，不為遠處外框擴張局部圖。第六格的圖例與圖說使用
    分離子面板，繪製後再檢查文字框不相交。輸出含失效連結均拒絕，以 ``xb`` 寫 PNG。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from shapely.affinity import translate
    from shapely.geometry import box
    from shapely.plotting import plot_line

    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"局部圖已存在，拒絕覆寫：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = data["summary"]
    curves: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in data["observations"]:
        curves[row["particle_id"]].append(row)
    for rows in curves.values():
        rows.sort(key=lambda row: float(row["age_seconds"]))
    boundary = data["projection"].project_geometry(data["open_geometry"])
    font = _display_font()
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    try:
        for ax, panel in zip(axes.flat, summary["horizontal_panels"], strict=False):
            selected = [p for p in data["particles"] if int(p["horizontal_panel"]) == panel["panel"]]
            if len(selected) != 4:
                raise ValueError("局部面板必須保留原始四個水層粒子")
            origin = curves[selected[0]["particle_id"]][0]
            x0, y0 = float(origin["x_m"]), float(origin["y_m"])
            points = [(0.0, 0.0)]
            for particle in selected:
                points.extend(
                    (float(row["x_m"]) - x0, float(row["y_m"]) - y0)
                    for row in curves[particle["particle_id"]]
                )
                points.append((float(particle["x_m"]) - x0, float(particle["y_m"]) - y0))
            # 兩軸採相同公尺跨度；邊距涵蓋邊界終點，避免以浮點 covers 判定漏畫外框。
            xs, ys = zip(*points, strict=True)
            span = max(max(xs) - min(xs), max(ys) - min(ys), 100.0) * 1.28
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            bounds = (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)
            clipped = translate(boundary, xoff=-x0, yoff=-y0).intersection(box(*bounds))
            ax.set_facecolor("#eaf3f8")
            if not clipped.is_empty:
                plot_line(
                    clipped, ax=ax, add_points=False, color="#174a70", linewidth=1.4, linestyle="--", zorder=2
                )
            for particle in selected:
                rows = curves[particle["particle_id"]]
                x = [float(row["x_m"]) - x0 for row in rows]
                y = [float(row["y_m"]) - y0 for row in rows]
                color = _VERTICAL_COLORS[particle["vertical_id"]]
                ax.plot(x, y, color=color, linewidth=1.7, alpha=0.9, zorder=4)
                ax.plot(
                    x[0],
                    y[0],
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    linestyle="None",
                    zorder=5,
                )
                ax.plot(
                    float(particle["x_m"]) - x0,
                    float(particle["y_m"]) - y0,
                    marker=_STATUS_MARKERS[particle["status"]],
                    markersize=8,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    linestyle="None",
                    zorder=6,
                )
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"H{panel['panel']}｜4 顆粒子", fontproperties=font, fontsize=14)
            ax.set_xlabel("東向位移 (m)", fontproperties=font)
            ax.set_ylabel("北向位移 (m)", fontproperties=font)
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.grid(color="white", linewidth=0.8)

        # 第六格分為獨立上下兩區，讓雙欄圖例與圖說不共用同一文字配置空間。
        legend_slot = axes.flat[5].get_subplotspec().subgridspec(2, 1, height_ratios=(1.15, 1), hspace=0.18)
        axes.flat[5].remove()
        legend_ax = fig.add_subplot(legend_slot[0])
        caption_ax = fig.add_subplot(legend_slot[1])
        legend_ax.axis("off")
        caption_ax.axis("off")
        legend_font = font.copy()
        legend_font.set_size(10)
        legend = legend_ax.legend(
            handles=_legend_handles(list(summary["vertical_order"])),
            loc="upper left",
            prop=legend_font,
            frameon=False,
            ncol=2,
            labelspacing=0.7,
            columnspacing=1.5,
            borderaxespad=0,
        )
        speed = -float(summary["settling_velocity_mps"]) * 1000
        caption = caption_ax.text(
            0.0,
            0.95,
            "各面板以起點為原點；距離單位為公尺\n"
            f"正向沉降速度{speed:g}毫米/秒\n"
            f"每個初始條件{summary['members_per_scenario']}顆粒子（含隨機擴散）",
            transform=caption_ax.transAxes,
            fontproperties=font,
            fontsize=10,
            va="top",
            linespacing=1.8,
        )
        fig.suptitle("B 區沉降粒子反向追蹤", fontproperties=font, fontsize=20, y=0.985)
        fig.text(0.5, 0.955, _subtitle(summary), ha="center", fontproperties=font, fontsize=12)
        fig.tight_layout(rect=(0.02, 0.01, 0.98, 0.93), h_pad=3.0, w_pad=2.5)
        fig.canvas.draw()
        if legend.get_window_extent().overlaps(caption.get_window_extent()):
            raise ValueError("局部圖的圖例與圖說重疊，停止輸出")
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output


def _prepare_baytrace_output(output_path: str | Path, description: str) -> Path:
    """準備新版單一 PNG 的排他輸出路徑，不接受既有檔案或任何符號連結。

    這個 helper 與 legacy renderer 分開，讓新版的圖面可以共用相同的拒覆寫保護，
    同時不改變上一輪兩張舊版圖的輸出流程。父目錄只在確定目標檔案不存在後建立；
    真正寫入仍使用 ``open('xb')``，以處理檢查與寫檔之間的競爭。
    """

    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{description}已存在，拒絕覆寫：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _ordered_baytrace_curves(data: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """依回溯年齡整理每顆粒子的既有觀測列，不新增或插值任何位置。

    ``age_seconds`` 以到達時刻為零，沿回溯方向增加；排序只決定線段繪製方向，
    不會改寫回傳資料或改變 CSV 的原始列順序。每一列仍來自既有 preview 的
    ``observations.csv``，因此缺失的海面／海床值可以在後續以空白中斷線段。
    """

    curves: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in data["observations"]:
        curves[row["particle_id"]].append(row)
    for rows in curves.values():
        rows.sort(key=lambda row: float(row["age_seconds"]))
    return curves


def _plot_value(value: Any) -> float:
    """將圖面欄位轉為有限浮點；空白、null、非數值與非有限值一律保留為 NaN。

    這是垂向診斷圖的缺值政策：NaN 交給 Matplotlib 形成斷線，絕不把沒有取樣的
    ``eta_m`` 或 ``bed_z_m`` 變成零。原始 CSV 仍保持不變，這個轉換只存在於繪圖
    所需的暫時序列。
    """

    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _plot_series(rows: list[dict[str, str]], field: str) -> list[float]:
    """依既有觀測列抽取一個圖面序列，維持欄位順序與缺值斷線語意。"""

    return [_plot_value(row.get(field)) for row in rows]


def _missing_series_count(rows: list[dict[str, str]], field: str) -> int:
    """計算某環境欄位無法使用的觀測列數，供 README 誠實報告缺值數量。"""

    return sum(not math.isfinite(value) for value in _plot_series(rows, field))


def _integer_value(value: Any) -> int | None:
    """讀取 CSV 字串或 summary 整數欄位；不接受布林、浮點近似或空白。"""

    if type(value) is int:
        return value
    if isinstance(value, str):
        token = value.strip()
        if token and (token.isdigit() or (token.startswith("-") and token[1:].isdigit())):
            return int(token)
    return None


def _sampling_failure_is_verified(data: dict[str, Any]) -> bool:
    """只有 CSV 與 summary 的同一批失敗事件一致時，才使用「取樣失敗停止」標籤。

    真實 r2 的三筆 ``numerical_failure`` 必須同時在 ``particles.csv`` 與
    ``summary.json`` 的 ``failure_details``／彙總診斷中標為
    ``invalid_velocity_sample`` 且 ``qc_flags=16``。若任何欄位缺少、數量不符、
    粒子身分不一致或原因不同，呼叫端就退回較保守的「數值停止」，避免把一個
    特定 pilot 的診斷原因泛化到所有數值失敗。
    """

    summary = data["summary"]
    rows = [row for row in data["particles"] if row.get("status") == "numerical_failure"]
    terminal_counts = summary.get("terminal_counts")
    if not rows or not isinstance(terminal_counts, dict):
        return False
    expected_count = _integer_value(terminal_counts.get("numerical_failure"))
    if expected_count != len(rows):
        return False

    details = summary.get("failure_details")
    if not isinstance(details, list) or len(details) != len(rows):
        return False
    detail_by_id = {
        detail.get("particle_id"): detail
        for detail in details
        if isinstance(detail, dict) and isinstance(detail.get("particle_id"), str)
    }
    particle_ids = {row.get("particle_id") for row in rows}
    if len(detail_by_id) != len(details) or set(detail_by_id) != particle_ids:
        return False

    for row in rows:
        if (
            row.get("failure_reason") != "invalid_velocity_sample"
            or _integer_value(row.get("qc_flags")) != 16
        ):
            return False
        detail = detail_by_id[row["particle_id"]]
        if (
            detail.get("status") != "numerical_failure"
            or detail.get("failure_reason") != "invalid_velocity_sample"
            or _integer_value(detail.get("qc_flags")) != 16
        ):
            return False

    diagnostics = summary.get("diagnostics")
    if isinstance(diagnostics, list):
        diagnostic = next(
            (
                item
                for item in diagnostics
                if isinstance(item, dict) and item.get("status") == "numerical_failure"
            ),
            None,
        )
        if diagnostic is None or (
            _integer_value(diagnostic.get("count")) != len(rows)
            or diagnostic.get("failure_reason") != "invalid_velocity_sample"
            or _integer_value(diagnostic.get("qc_flags")) != 16
        ):
            return False
    return True


def _baytrace_status_labels(data: dict[str, Any]) -> dict[str, str]:
    """依資料核對結果建立新版顯示名稱，底層 status 值與計數永遠不被改寫。"""

    labels = dict(_BAYTRACE_STATUS_LABELS)
    if _sampling_failure_is_verified(data):
        labels["numerical_failure"] = "取樣失敗停止"
    return labels


def _baytrace_layer_label(level: str) -> str:
    """回傳不含內部層代碼的完整中文水層名稱，未知值使用保守通用名稱。"""

    return _BAYTRACE_VERTICAL_LABELS.get(level, "其他初始水層")


def _baytrace_layer_definition(level: str) -> str:
    """回傳水層的設定目標或近床物理定義，避免由圖面反推固定公尺高度。"""

    return _BAYTRACE_VERTICAL_DEFINITIONS.get(level, "來源設定未提供可核對的水層定義")


def _baytrace_metadata(summary: dict[str, Any]) -> str:
    """組合四張新版圖共用的研究情境摘要，不顯示 run／scenario 等內部識別碼。

    時間只把既有 arrival 與 horizon 轉成易讀的 UTC 時段；不從圖面推算新的
    垂向高度或粒子統計。``M`` 的含義由「同一組粒子」與 README 另行說明，避免
    將成員數誤讀為已完成回溯的數量。
    """

    arrival = datetime.fromisoformat(summary["arrival_utc"])
    horizon = float(summary["settings"]["horizon_seconds"])
    start = arrival - timedelta(seconds=horizon)
    region = summary.get("projection", {}).get("analysis_region_id", "B")
    site = {"hsinchu": "新竹外海"}.get(summary.get("study_site_id"), "研究站點")
    return (
        f"{region}區／{site}｜{arrival:%Y-%m-%d}｜"
        f"同一組{summary['particle_count']}顆粒子｜"
        f"{len(summary['horizontal_panels'])}個初始位置×{len(summary['vertical_order'])}個初始水層｜"
        f"單一沉降材質｜{arrival:%H:%M}→{start:%H:%M} UTC｜"
        f"回溯上限{horizon / 3600:g}小時"
    )


def _baytrace_layer_handles(levels: list[str]):
    """建立「線色：初始水層」的中文圖例，讓線色與水層角色分組呈現。"""

    from matplotlib.lines import Line2D

    return [
        Line2D(
            [],
            [],
            color=_BAYTRACE_VERTICAL_COLORS.get(level, "#666666"),
            linewidth=2.0,
            label=_baytrace_layer_label(level),
        )
        for level in levels
    ]


def _baytrace_endpoint_handles(status_labels: dict[str, str], *, include_boundary: bool = True):
    """建立「標記：端點／停止狀態」圖例；三種終點形狀全部固定為紅色。"""

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=_BAYTRACE_INITIAL_COLOR,
            markerfacecolor=_BAYTRACE_INITIAL_COLOR,
            markeredgecolor=_BAYTRACE_INITIAL_COLOR,
            linestyle="None",
            markersize=7,
            label="初始位置（回溯起點）",
        )
    ]
    for status in ("max_age", "flow_domain_open_exit", "numerical_failure"):
        handles.append(
            Line2D(
                [],
                [],
                marker=_STATUS_MARKERS[status],
                color=_BAYTRACE_FINAL_COLOR,
                markerfacecolor=_BAYTRACE_FINAL_COLOR,
                markeredgecolor=_BAYTRACE_FINAL_COLOR,
                linestyle="None",
                markersize=7,
                label=f"最終位置｜{status_labels[status]}",
            )
        )
    if include_boundary:
        handles.append(
            Line2D(
                [],
                [],
                color="#174a70",
                linewidth=1.4,
                linestyle="--",
                label="本次計算範圍邊界",
            )
        )
    return handles


def _save_baytrace_figure(fig, output: Path) -> None:
    """將新版 Matplotlib 圖以排他模式保存，不在 PNG metadata 寫入外部來源警語。"""

    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=180, bbox_inches="tight")


def render_baytrace_overview(data: dict[str, Any], output_path: str | Path) -> Path:
    """繪製新版經緯度區域總覽，第一眼說明平面回溯問題與端點語意。

    軌跡仍逐粒子連接既有保存觀測；總覽用原 AEQD 的反投影顯示經度／緯度，海岸
    僅作已確認的地理參照。四層線色固定使用藍、橙、紫、青，綠色只表示回溯起點，
    紅色只表示追蹤停止位置；``^``、``s``、``X`` 分別保留 max_age、開放邊界離域、
    numerical_failure 的停止形狀。這張圖不把終點解讀成已知污染來源。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_line, plot_polygon

    output = _prepare_baytrace_output(output_path, "新版水平總覽圖")
    summary = data["summary"]
    projection = data["projection"]
    curves = _ordered_baytrace_curves(data)
    levels = list(summary["vertical_order"])
    status_labels = _baytrace_status_labels(data)
    font = _display_font()
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    try:
        ax.set_facecolor("#eaf3f8")
        for land in data["land_geometries"]:
            plot_polygon(
                land,
                ax=ax,
                add_points=False,
                facecolor="#c9c9c9",
                edgecolor="#858585",
                linewidth=0.25,
                alpha=0.85,
                zorder=1,
            )
        plot_line(
            data["open_geometry"],
            ax=ax,
            add_points=False,
            color="#174a70",
            linewidth=1.4,
            linestyle="--",
            zorder=2,
        )

        for particle in data["particles"]:
            rows = curves.get(particle["particle_id"], [])
            color = _BAYTRACE_VERTICAL_COLORS.get(particle["vertical_id"], "#666666")
            if rows:
                longitude, latitude = projection.unproject(
                    [float(row["x_m"]) for row in rows], [float(row["y_m"]) for row in rows]
                )
                ax.plot(longitude, latitude, color=color, linewidth=1.05, alpha=0.82, zorder=4)
                ax.plot(
                    longitude[0],
                    latitude[0],
                    marker="o",
                    markersize=5.6,
                    markerfacecolor=_BAYTRACE_INITIAL_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    linestyle="None",
                    zorder=6,
                )
            ax.plot(
                float(particle["longitude"]),
                float(particle["latitude"]),
                marker=_STATUS_MARKERS.get(particle["status"], "D"),
                markersize=7.1,
                markerfacecolor=_BAYTRACE_FINAL_COLOR,
                markeredgecolor=_BAYTRACE_FINAL_COLOR,
                markeredgewidth=0.8,
                color=_BAYTRACE_FINAL_COLOR,
                linestyle="None",
                zorder=7,
            )

        for panel in summary["horizontal_panels"]:
            ax.annotate(
                f"位置{panel['panel']}",
                (float(panel["receptor_lon"]), float(panel["receptor_lat"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontproperties=font,
                fontsize=9,
                color="#17324d",
                zorder=8,
            )

        min_lon, min_lat, max_lon, max_lat = data["domain_geometry"].bounds
        lon_margin = max((max_lon - min_lon) * 0.04, 0.02)
        lat_margin = max((max_lat - min_lat) * 0.04, 0.02)
        ax.set_xlim(min_lon - lon_margin, max_lon + lon_margin)
        ax.set_ylim(min_lat - lat_margin, max_lat + lat_margin)
        center_lat = float(summary["projection"]["center_lonlat"][1])
        ax.set_aspect(1.0 / math.cos(math.radians(center_lat)), adjustable="box")
        ax.set_xlabel("經度（°E）", fontproperties=font)
        ax.set_ylabel("緯度（°N）", fontproperties=font)
        ax.grid(color="white", linewidth=0.7, alpha=0.8)

        fig.suptitle(
            "水平軌跡｜回溯一小時內的平面移動路徑",
            fontproperties=font,
            fontsize=18,
            y=0.995,
            va="top",
        )
        fig.text(
            0.5,
            0.962,
            "從指定位置往前回溯，粒子經過哪些位置？",
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=11,
        )
        fig.text(
            0.5,
            0.937,
            _baytrace_metadata(summary),
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=10,
        )

        layer_legend = fig.legend(
            handles=_baytrace_layer_handles(levels),
            title="線色：初始水層",
            loc="lower center",
            # 把兩組圖例放在軸區以下的不同列；提高水層列可避免外框彼此壓住，
            # 也不讓圖例侵入經度／緯度軸標籤。輸出仍由同一張圖保存，不增加分析內容。
            bbox_to_anchor=(0.5, 0.145),
            ncol=4,
            prop=font,
            frameon=True,
            columnspacing=1.4,
        )
        layer_legend.get_title().set_fontproperties(font)
        endpoint_legend = fig.legend(
            handles=_baytrace_endpoint_handles(status_labels),
            title="標記：端點／停止狀態",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=3,
            prop=font,
            frameon=True,
            columnspacing=1.0,
        )
        endpoint_legend.get_title().set_fontproperties(font)
        fig.tight_layout(rect=(0.01, 0.20, 0.99, 0.90))
        _save_baytrace_figure(fig, output)
    finally:
        plt.close(fig)
    return output


def render_baytrace_local(data: dict[str, Any], output_path: str | Path) -> Path:
    """繪製新版五位置局部圖，只平移原 AEQD 公尺座標且各位置獨立設範圍。

    ``位置1`` 至 ``位置5`` 各自包含四個初始水層，線段與端點使用和區域總覽相同的
    顏色／標記語意。平移原點只為便於閱讀，不移除、放大、抖動或重新積分任何座標；
    五個面板可以有不同範圍，因此不能以印刷長度跨面板比較移動量。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from shapely.affinity import translate
    from shapely.geometry import box
    from shapely.plotting import plot_line

    output = _prepare_baytrace_output(output_path, "新版水平局部圖")
    summary = data["summary"]
    curves = _ordered_baytrace_curves(data)
    levels = list(summary["vertical_order"])
    status_labels = _baytrace_status_labels(data)
    boundary = data["projection"].project_geometry(data["open_geometry"])
    font = _display_font()
    # 五個資料面板維持原來的 3×2 版面；右下空格再切成三個獨立子格，
    # 讓水層圖例、端點圖例與兩行圖說各自有真實的版面空間，不靠同一座標軸
    # 的相對 y 值互相避讓。這也使最後的 bbox 檢查能對應三個獨立 artist。
    fig = plt.figure(figsize=(12, 16.5), dpi=150)
    grid = fig.add_gridspec(3, 2)
    axes = [[fig.add_subplot(grid[row, column]) for column in range(2)] for row in range(3)]
    panel_axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1], axes[2][0]]
    fig.delaxes(axes[2][1])
    legend_grid = grid[2, 1].subgridspec(
        3,
        1,
        # 第一列需容納圖例標題加四個水層，第二列需容納單欄五個端點項目；
        # 列高按實際內容預留，第三列獨立放兩行 caption，避免 artist 跨列溢出。
        height_ratios=(4.0, 5.0, 2.0),
        # 子格之間保留明確的空白帶；這個間距是版面保護的一部分，避免
        # 圖例外框因字型實際高度跨過相鄰列，即使文字內容變長也不相接。
        hspace=0.35,
    )
    layer_legend_ax, endpoint_legend_ax, caption_ax = [
        fig.add_subplot(legend_grid[index, 0]) for index in range(3)
    ]
    try:
        for ax, panel in zip(panel_axes, summary["horizontal_panels"], strict=True):
            selected = [
                particle
                for particle in data["particles"]
                if int(particle["horizontal_panel"]) == panel["panel"]
            ]
            if len(selected) != 4:
                raise ValueError("新版局部圖每個位置必須保留四個初始水層粒子")
            first_rows = curves.get(selected[0]["particle_id"], [])
            if first_rows:
                x0, y0 = float(first_rows[0]["x_m"]), float(first_rows[0]["y_m"])
            else:
                x0, y0 = float(selected[0]["x_m"]), float(selected[0]["y_m"])
            points = [(0.0, 0.0)]
            for particle in selected:
                points.extend(
                    (float(row["x_m"]) - x0, float(row["y_m"]) - y0)
                    for row in curves.get(particle["particle_id"], [])
                )
                points.append((float(particle["x_m"]) - x0, float(particle["y_m"]) - y0))
            xs, ys = zip(*points, strict=True)
            span = max(max(xs) - min(xs), max(ys) - min(ys), 100.0) * 1.28
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            bounds = (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)
            clipped = translate(boundary, xoff=-x0, yoff=-y0).intersection(box(*bounds))
            ax.set_facecolor("#eaf3f8")
            if not clipped.is_empty:
                plot_line(
                    clipped,
                    ax=ax,
                    add_points=False,
                    color="#174a70",
                    linewidth=1.4,
                    linestyle="--",
                    zorder=2,
                )
            for particle in selected:
                rows = curves.get(particle["particle_id"], [])
                color = _BAYTRACE_VERTICAL_COLORS.get(particle["vertical_id"], "#666666")
                if rows:
                    x = [float(row["x_m"]) - x0 for row in rows]
                    y = [float(row["y_m"]) - y0 for row in rows]
                    ax.plot(x, y, color=color, linewidth=1.7, alpha=0.9, zorder=4)
                    ax.plot(
                        x[0],
                        y[0],
                        marker="o",
                        markersize=6.5,
                        markerfacecolor=_BAYTRACE_INITIAL_COLOR,
                        markeredgecolor="white",
                        markeredgewidth=0.7,
                        linestyle="None",
                        zorder=6,
                    )
                ax.plot(
                    float(particle["x_m"]) - x0,
                    float(particle["y_m"]) - y0,
                    marker=_STATUS_MARKERS.get(particle["status"], "D"),
                    markersize=8.2,
                    markerfacecolor=_BAYTRACE_FINAL_COLOR,
                    markeredgecolor=_BAYTRACE_FINAL_COLOR,
                    markeredgewidth=0.8,
                    color=_BAYTRACE_FINAL_COLOR,
                    linestyle="None",
                    zorder=7,
                )
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"位置{panel['panel']}｜粒子數：{len(selected)}", fontproperties=font, fontsize=14)
            ax.set_xlabel("東向位移（公尺）", fontproperties=font)
            ax.set_ylabel("北向位移（公尺）", fontproperties=font)
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.grid(color="white", linewidth=0.8)

        for legend_axis in (layer_legend_ax, endpoint_legend_ax, caption_ax):
            legend_axis.axis("off")
        layer_legend = layer_legend_ax.legend(
            handles=_baytrace_layer_handles(levels),
            title="線色：初始水層",
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            ncol=1,
            prop=font,
            frameon=False,
            labelspacing=0.8,
        )
        layer_legend.get_title().set_fontproperties(font)
        endpoint_legend = endpoint_legend_ax.legend(
            handles=_baytrace_endpoint_handles(status_labels),
            title="標記：端點／停止狀態",
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            # 端點圖例獨佔右下格的中列；單欄避免向頁面外側撐開，
            # 並保留下列 caption 的獨立版面空間。
            ncol=1,
            prop=font,
            frameon=False,
            labelspacing=0.7,
        )
        endpoint_legend.get_title().set_fontproperties(font)
        caption = caption_ax.text(
            0.0,
            0.5,
            "各位置以初始位置為原點（0,0）。\n各面板尺度不同，請依公尺刻度比較。",
            transform=caption_ax.transAxes,
            fontproperties=font,
            fontsize=10,
            va="center",
        )

        fig.suptitle(
            "水平軌跡｜回溯一小時內的平面移動路徑",
            fontproperties=font,
            fontsize=20,
            y=0.995,
            va="top",
        )
        fig.text(
            0.5,
            0.962,
            "從指定位置往前回溯，粒子經過哪些位置？",
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=11,
        )
        fig.text(
            0.5,
            0.937,
            _baytrace_metadata(summary),
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=10,
        )
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.04, top=0.90, hspace=0.45, wspace=0.25)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if (
            layer_legend.get_window_extent(renderer).overlaps(endpoint_legend.get_window_extent(renderer))
            or endpoint_legend.get_window_extent(renderer).overlaps(caption.get_window_extent(renderer))
            or layer_legend.get_window_extent(renderer).overlaps(caption.get_window_extent(renderer))
        ):
            raise ValueError("新版局部圖的圖例與兩行圖說重疊，停止輸出")
        _save_baytrace_figure(fig, output)
    finally:
        plt.close(fig)
    return output


def render_baytrace_depth(data: dict[str, Any], output_path: str | Path) -> Path:
    """繪製四個初始水層的高度—回溯時間診斷圖，所有面板統一 0--60 分鐘。

    粒子高度 ``z_m``、海面 ``eta_m`` 與海床 ``bed_z_m`` 直接取自既有觀測 CSV；
    空白欄位轉為 NaN 只讓圖線留空，不補零、不從 normalized_config 硬算統一公尺
    高度，也不重新讀 forcing 或重跑取樣。水層名稱使用完整中文，並明確註明海床
    只是參考線；近床水層是最低有效 OCM 水層中心，不是海床面。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = _prepare_baytrace_output(output_path, "新版垂向診斷圖")
    summary = data["summary"]
    curves = _ordered_baytrace_curves(data)
    levels = list(summary["vertical_order"])
    status_labels = _baytrace_status_labels(data)
    font = _display_font()
    fig, axes = plt.subplots(len(levels), 1, figsize=(11, 12), squeeze=False, sharex=True)
    try:
        for index, level in enumerate(levels):
            ax = axes[index, 0]
            selected = [particle for particle in data["particles"] if particle["vertical_id"] == level]
            layer_color = _BAYTRACE_VERTICAL_COLORS.get(level, "#666666")
            for particle in selected:
                rows = curves.get(particle["particle_id"], [])
                if not rows:
                    continue
                age_minutes = [_plot_value(row.get("age_seconds")) / 60.0 for row in rows]
                z_values = _plot_series(rows, "z_m")
                eta_values = _plot_series(rows, "eta_m")
                bed_values = _plot_series(rows, "bed_z_m")
                ax.plot(age_minutes, z_values, color=layer_color, linewidth=1.2, alpha=0.72, zorder=3)
                ax.plot(
                    age_minutes,
                    eta_values,
                    color="#1b4f72",
                    linestyle="--",
                    linewidth=0.75,
                    alpha=0.55,
                    zorder=2,
                )
                ax.plot(
                    age_minutes,
                    bed_values,
                    color="#8c510a",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.58,
                    zorder=2,
                )
                if age_minutes and math.isfinite(z_values[0]):
                    ax.plot(
                        age_minutes[0],
                        z_values[0],
                        marker="o",
                        markersize=4.7,
                        markerfacecolor=_BAYTRACE_INITIAL_COLOR,
                        markeredgecolor=_BAYTRACE_INITIAL_COLOR,
                        color=_BAYTRACE_INITIAL_COLOR,
                        linestyle="None",
                        clip_on=False,
                        zorder=5,
                    )
                terminal_index = next(
                    (
                        position
                        for position in range(len(z_values) - 1, -1, -1)
                        if math.isfinite(z_values[position])
                    ),
                    None,
                )
                if terminal_index is not None:
                    ax.plot(
                        age_minutes[terminal_index],
                        z_values[terminal_index],
                        marker=_STATUS_MARKERS.get(particle["status"], "D"),
                        markersize=5.7,
                        markerfacecolor=_BAYTRACE_FINAL_COLOR,
                        markeredgecolor=_BAYTRACE_FINAL_COLOR,
                        color=_BAYTRACE_FINAL_COLOR,
                        linestyle="None",
                        clip_on=False,
                        zorder=6,
                    )
            ax.set_xlim(0.0, 60.0)
            ax.set_title(
                f"{_baytrace_layer_label(level)}（{_baytrace_layer_definition(level)}）｜粒子數：{len(selected)}",
                fontproperties=font,
                fontsize=12,
            )
            ax.set_ylabel("高度（公尺；向上為正）", fontproperties=font)
            ax.grid(color="#d9e2e8", linewidth=0.7)
            ax.set_xticks((0, 15, 30, 45, 60))
        axes[-1, 0].set_xlabel("回溯時間（分鐘；0 為到達時刻）", fontproperties=font)
        legend_handles = [
            Line2D([], [], color="#555555", linewidth=1.4, label="粒子高度"),
            Line2D([], [], color="#1b4f72", linestyle="--", linewidth=1.0, label="海面高度"),
            Line2D([], [], color="#8c510a", linestyle=":", linewidth=1.0, label="海床高度"),
        ]
        legend_handles.extend(_baytrace_endpoint_handles(status_labels, include_boundary=False))
        legend = fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=4,
            prop=font,
            frameon=True,
        )
        legend.get_title().set_fontproperties(font)
        fig.suptitle(
            "垂向軌跡｜回溯過程中的粒子高度變化",
            fontproperties=font,
            fontsize=18,
            y=0.995,
            va="top",
        )
        fig.text(
            0.5,
            0.962,
            "回溯過程中，粒子在水中的高度如何變化？海面與海床提供垂向參考。",
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=11,
        )
        fig.text(
            0.5,
            0.937,
            _baytrace_metadata(summary),
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=10,
        )
        fig.tight_layout(rect=(0.08, 0.095, 0.99, 0.89), h_pad=1.5)
        _save_baytrace_figure(fig, output)
    finally:
        plt.close(fig)
    return output


def _baytrace_terminal_table(data: dict[str, Any]) -> dict[str, dict[str, int]]:
    """讀取 summary 的四水層八狀態計數，保留零值並驗證每層分母為五。"""

    summary = data["summary"]
    if len(summary["vertical_order"]) != 4:
        raise ValueError("新版停止圖必須有四個初始水層")
    raw = summary.get("terminal_counts_by_vertical")
    if not isinstance(raw, dict):
        raise ValueError("新版停止圖缺少 terminal_counts_by_vertical")
    csv_counts = Counter(
        (particle.get("vertical_id"), particle.get("status")) for particle in data["particles"]
    )
    table: dict[str, dict[str, int]] = {}
    for level in summary["vertical_order"]:
        source = raw.get(level)
        if not isinstance(source, dict):
            raise ValueError(f"新版停止圖缺少水層計數：{_baytrace_layer_label(level)}")
        row: dict[str, int] = {}
        for status in _TERMINAL_STATUSES:
            if status not in source:
                raise ValueError(f"新版停止圖缺少停止狀態：{status}")
            count = _integer_value(source[status])
            if count is None or count < 0:
                raise ValueError(f"新版停止圖的停止計數無效：{status}")
            if count != csv_counts[(level, status)]:
                raise ValueError(
                    f"新版停止圖 summary 與 CSV 逐格計數不符：{_baytrace_layer_label(level)}／{status}"
                )
            row[status] = count
        if sum(row.values()) != 5:
            raise ValueError(f"新版停止圖每個初始水層必須是五顆粒子：{_baytrace_layer_label(level)}")
        table[level] = row
    return table


def render_baytrace_terminal(data: dict[str, Any], output_path: str | Path) -> Path:
    """繪製四初始水層的八狀態停止原因圖，保留全部粒子與零計數狀態。

    圖面回答「有多少粒子完成回溯，其餘為何提前停止？」；``max_age`` 以
    「完成1小時回溯」顯示，開放邊界離域與已核對的取樣失敗則分別顯示。底層
    status、分母及 9/8/3 計數不變；這是既有 preview 的補充診斷，不是新的來源分析。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output = _prepare_baytrace_output(output_path, "新版停止原因圖")
    summary = data["summary"]
    counts = summary["terminal_counts"]
    levels = list(summary["vertical_order"])
    table = _baytrace_terminal_table(data)
    status_labels = _baytrace_status_labels(data)
    font = _display_font()
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=150)
    try:
        y_positions = list(range(len(levels)))
        left = [0] * len(levels)
        for status in _TERMINAL_STATUSES:
            values = [table[level][status] for level in levels]
            ax.barh(
                y_positions,
                values,
                left=left,
                height=0.58,
                color=_BAYTRACE_STATUS_COLORS[status],
                edgecolor="white",
                linewidth=0.6,
                label=status_labels[status],
            )
            for position, count, start in zip(y_positions, values, left, strict=True):
                if count:
                    ax.text(
                        start + count / 2,
                        position,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if status not in {"data_gap", "numerical_failure"} else "#222222",
                    )
            left = [start + count for start, count in zip(left, values, strict=True)]
        for position in y_positions:
            ax.text(5.08, position, "總數 5", va="center", fontproperties=font, fontsize=10, color="#17324d")
        ax.set_yticks(y_positions, [_baytrace_layer_label(level) for level in levels], fontproperties=font)
        ax.invert_yaxis()
        ax.set_xlim(0, 5.55)
        ax.set_xlabel("粒子數（每個初始水層共5顆）", fontproperties=font)
        ax.set_ylabel("初始水層", fontproperties=font)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(axis="x", color="#d9e2e8", linewidth=0.7)
        legend_handles = [
            Patch(facecolor=_BAYTRACE_STATUS_COLORS[status], edgecolor="white", label=status_labels[status])
            for status in _TERMINAL_STATUSES
        ]
        legend = ax.legend(
            handles=legend_handles,
            title="全部停止狀態（含0計數）",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=4,
            prop=font,
            frameon=True,
        )
        legend.get_title().set_fontproperties(font)
        fig.suptitle(
            "停止原因｜完成回溯與提前停止的粒子數",
            fontproperties=font,
            fontsize=18,
            y=0.995,
            va="top",
        )
        fig.text(
            0.5,
            0.958,
            f"{counts.get('max_age', 0)}顆完成回溯、{counts.get('flow_domain_open_exit', 0)}顆離開計算範圍、"
            f"{counts.get('numerical_failure', 0)}顆{status_labels['numerical_failure']}。",
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=11,
        )
        fig.text(
            0.5,
            0.932,
            _baytrace_metadata(summary),
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=10,
        )
        # 研究問題已在 README 閱讀順序說明；圖面上只保留標題、動態 9/8/3
        # 導讀與 metadata 三列，讓 header 不與繪圖軸或彼此重疊。
        fig.tight_layout(rect=(0.08, 0.22, 0.98, 0.90))
        _save_baytrace_figure(fig, output)
    finally:
        plt.close(fig)
    return output


def _source_snapshot(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """只為七個實際繪圖輸入記錄本機路徑、bytes 與 SHA；不掃描 run 或驅動陣列。"""

    return {
        name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _output_readme(data: dict[str, Any], sources: dict, rebuild_command: str) -> str:
    """產生 PI 可閱讀的繁中圖說、來源事實與重建命令，技術細節不擠入 PNG。"""

    summary = data["summary"]
    center = summary["projection"]["center_lonlat"]
    counts = summary["terminal_counts"]
    lines = [
        "# B 區沉降粒子反向追蹤",
        "",
        f"Run：`{summary['run_id']}`；站點：B／hsinchu／{data['domain_record']['flow_domain_id']}。",
        _subtitle(summary) + "。",
        "",
        "- [區域總覽](horizontal_overview.png)：經緯度、陸地及本次試跑登錄外框。",
        "- [五個局部面板](horizontal_local.png)：沿用原圖 H1–H5，每組四個初始水層。",
        "",
        "## 圖面與計數",
        "",
        "保留全部 20 顆粒子與 203 筆模型保存紀錄；203 不是實測觀測數。",
        "線條依回溯時間連接保存位置，未重新取樣。圓點為回溯起點；終點取自 particles CSV。",
        "初始水層固定為近海床（藍）、中下水層（橙）、中上水層（綠）、上水層（紅）。",
        f"停止計數：達 1 小時 {counts['max_age']}、開放邊界離域 {counts['flow_domain_open_exit']}、"
        f"數值停止 {counts['numerical_failure']}；資料缺口 {counts['data_gap']}、"
        f"沉積 {counts['deposited']}。",
        "三顆數值停止均屬上水層；未刪除或以其他狀態替代。",
        "",
        f"材質代理：`{summary['material_id']}`；正向沉降速度 "
        f"{summary['settling_velocity_mps']} m/s（向下 2 毫米／秒）。",
        f"每個初始條件 M={summary['members_per_scenario']}，仍包含隨機擴散；"
        "這些反向路徑不是來源機率，也未經系集收斂或獨立觀測驗證。",
        "逆向時間的深度變淺不代表正向材料上浮；原 depth 圖與無底圖版均保持不變。",
        "",
        "## 座標與圖層",
        "",
        f"公尺座標沿用原等距方位投影（AEQD），中心為經度 {center[0]}、緯度 {center[1]}。",
        "總覽以原 DomainProjection 反投影為經緯度；緯度增加方向向上為北。",
        "總覽以中心緯度補償經緯度顯示比例，並非處處等距量測圖；距離請讀局部公尺座標。",
        "各局部圖以共同初始 XY 為原點，只平移原 AEQD x/y；圖中東／北向指原投影軸。",
        "每個面板 x/y 等比例，但五個面板的範圍各自設定，不能直接比較印刷線段長度。",
        "外框與軌跡接受相同座標轉換／平移；局部僅顯示外框與視窗的實際交集。",
        "土地與外框在底層，保存軌跡及起訖標記在上層；不以陸地填色遮蔽或刪除粒子。",
        "",
        "## 來源與限制",
        "",
        "海岸使用已確認的 `taiwan_exact_coastline.geojson`：CRS84（經度、緯度），"
        "1905 個有效 Polygon／MultiPolygon；原始供應者、年代與授權未附。",
        "本次用於內部 PI 地理參照，不宣稱為官方精確岸線，亦不套用其他資料的授權。",
        f"B 外框來源方法：`{data['domain_manifest']['provenance']['method_id']}`。",
        "這是 accepted manifest 已登錄的加密經緯度矩形幾何，不是完整濕網格或有效海水邊界；"
        "外框內不保證全部都是有效水域。",
        "B 的 flow_domain 開放邊界與 polygon 外環相等，因此只畫一條登錄開放邊界線。",
        "domain/open 的語意 SHA 已與原 summary 綁定；各檔原始 bytes 與 SHA 如下。",
        "",
        "| 輸入 | bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {name} | {record['size_bytes']} | `{record['sha256']}` |" for name, record in sources.items()
    )
    lines += [
        "",
        "## 本機重建",
        "",
        "在專案根目錄執行下列命令；輸出目錄必須不存在。重建目錄若已存在，請另取新名稱。",
        "",
        "```bash",
        rebuild_command,
        "```",
        "",
        "本入口只讀 preview JSON/CSV 與明示幾何，不讀海流／波浪大陣列，不重新積分。",
        "來源七檔的前後 bytes/SHA 必須一致；manifest 保存其核對結果、程式與字型 SHA，"
        "以及兩張 PNG 和本 README 的 SHA，不對 manifest 本身做循環雜湊。",
        "失敗保留新輸出目錄供診斷；完成狀態須以 manifest 存在及各輸出校驗匹配為準。",
        "原 G2-report.md、preview/README.md、run 與 audit 均未修改；"
        "本補圖不新增連續碰撞驗證、來源歸因或模擬有效性結論。",
    ]
    return "\n".join(lines) + "\n"


def _baytrace_readme(
    data: dict[str, Any],
    sources_before: dict[str, dict[str, Any]],
    sources_after: dict[str, dict[str, Any]],
    status_labels: dict[str, str],
) -> str:
    """產生新版四圖的 PI 閱讀說明、呈現對照、水層定義及來源核對摘要。

    README 只使用可移植的檔名與 shell 變數，不把個人絕對路徑寫入可追蹤文件。
    圖面意義先於程式細節：先說明三類問題，再列出本次 20 顆粒子、四水層八狀態、
    缺值與取樣失敗的資料事實。新版深度／停止圖被明確標為專案補充診斷，與 default
    legacy 的兩張水平圖分開，避免把呈現參照誤寫成數值或資料契約。
    """

    summary = data["summary"]
    counts = summary["terminal_counts"]
    terminal_table = _baytrace_terminal_table(data)
    missing_eta = _missing_series_count(data["observations"], "eta_m")
    missing_bed = _missing_series_count(data["observations"], "bed_z_m")
    sampling_verified = _sampling_failure_is_verified(data)
    numerical_rows = [row for row in data["particles"] if row.get("status") == "numerical_failure"]
    numerical_ages = sorted(float(row["age_seconds"]) for row in numerical_rows if row.get("age_seconds"))
    numerical_age_text = "、".join(f"{age:g}" for age in numerical_ages) if numerical_ages else "未提供"
    center = summary["projection"]["center_lonlat"]
    lines = [
        "# B 區新版回溯成果圖",
        "",
        "本目錄是由已完成 preview 獨立重繪的新版圖面；不讀 forcing、不重新取樣、不重新積分，"
        "也不改變原始 CSV、summary 或舊成果。輸出只包含兩張水平圖、一張垂向診斷圖及一張停止原因圖，"
        "沒有新增直方圖或其他分析。",
        f"圖上共通情境：{_baytrace_metadata(summary)}。",
        "",
        "## 給 PI 的閱讀方式",
        "",
        "共同研究問題是：從指定到達位置往前回溯，粒子在平面上經過哪裡、在水中的高度如何改變，"
        "以及哪些粒子完成回溯、哪些粒子提前停止。這些圖把一組 20 顆粒子（5 個初始位置 × "
        "4 個初始水層）放在同一個單一沉降材質情境中；它們是工程先導的條件式路徑診斷，"
        "不是已知污染來源的證明。",
        "",
        "建議閱讀順序：",
        "",
        "1. [水平區域總覽](horizontal_overview.png)：先回答「從指定位置往前回溯，粒子經過哪些位置？」；"
        "經緯度與已確認海岸讓整體位置關係可直接判讀。",
        "2. [水平局部圖](horizontal_local.png)：再看位置1–5 的原 AEQD 公尺座標，確認各位置的路徑細節；"
        "五個面板各自等比例，但範圍不同。",
        "3. [垂向軌跡](depth_age.png)：回答「回溯過程中，粒子在水中的高度如何變化？」；"
        "z、海面 eta、海床 bed 都只讀保存觀測。",
        "4. [停止原因](terminal_counts.png)：最後回答「有多少粒子完成回溯，其餘為何提前停止？」；"
        "每個初始水層的五顆粒子與八種狀態都保留，包含零計數。",
        "",
        "這樣的順序先交代平面路徑，再補充垂向變化，最後用停止原因解釋為何不能把全部 20 顆都視為追滿一小時。"
        "終點是追蹤停止位置，不是污染來源；M=1 仍含隨機擴散，不能稱為來源機率。",
        "",
        "## 呈現參照與本專案界線",
        "",
        "呈現參照來源為 BayTrace v4.7.2 套件內 `scripts/analyze_cases.py` 的 `plot_case`；"
        "本專案只整理其軌跡、初始／最終位置、經緯度與粒子數等呈現方式，不複製其程式。",
        "",
        "下表只整理可採用的圖面呈現方式與標籤，不表示整套新版圖面都由同一方法定義，"
        "數值與 CSV 契約仍完全沿用本專案既有 preview。",
        "",
        "| 呈現參照 | 本專案採用 | 必要差異 |",
        "|---|---|---|",
        "| 軌跡 | 水平軌跡，逐粒子連接已保存位置 | 本專案是逆時間回溯，線段不是正向預測路徑 |",
        "| 初始 | 初始位置（回溯起點）（綠色圓點） | 指到達時刻的保存位置，不是污染源位置 |",
        "| 最終 | 最終位置｜追蹤停止狀態（紅色 ^／s／X） | 代表停止狀態；不是已知污染來源 |",
        "| 經度／緯度 | 區域總覽使用經度、緯度 | 局部圖改用原 AEQD 平移公尺座標，未移動或放大資料 |",
        "| 粒子數 | 圖上明示同一組 20 顆粒子、5 個初始位置 × 4 個初始水層 | "
        "20 顆中只有完成狀態才追滿 1 小時，M=1 且含隨機擴散 |",
        "| 垂向軌跡 | 專案補充診斷圖：粒子高度、海面、海床 | "
        "呈現參照沒有直接對應，z／eta／bed 均取既有 CSV |",
        "| 停止原因 | 專案補充診斷圖：完成與提前停止的粒子數 | 呈現參照沒有直接對應，八狀態及零計數均保留 |",
        "",
        "新版紅色端點圖例依資料核對結果顯示「最終位置｜"
        f"{status_labels['max_age']}」、「最終位置｜{status_labels['flow_domain_open_exit']}」與"
        f"「最終位置｜{status_labels['numerical_failure']}」；三者均不是已知污染來源。",
        "新版與 default `legacy` 分開：legacy 仍只產生原有兩張水平 PNG 及其舊契約；"
        "不要把 legacy 的 H1–H5、線色或任意終點語意與本新版混用。新版圖面統一使用「位置1–5」、"
        "完整中文水層名、藍／橙／紫／青線色、綠色回溯起點及紅色追蹤停止位置。",
        "",
        "## 水層定義與停止統計",
        "",
        "上／中上／中下水層的百分比是設定目標，不把到達時刻資料硬算成四個統一公尺高度；"
        "近床水層則是最低有效 OCM 水層中心，不是海床面。",
        "",
        "| 新版圖面水層名稱 | 設定方式 |",
        "|---|---|",
    ]
    lines.extend(
        f"| {_baytrace_layer_label(level)} | {_baytrace_layer_definition(level)} |"
        for level in summary["vertical_order"]
    )
    lines += [
        "",
        "停止圖固定保留八種狀態，且每個初始水層分母均為 5：",
        "",
        "| 停止狀態 | 總數 | "
        + " | ".join(_baytrace_layer_label(level) for level in summary["vertical_order"])
        + " |",
        "|---|---:|" + "---:|" * len(summary["vertical_order"]),
    ]
    for status in _TERMINAL_STATUSES:
        columns = " | ".join(str(terminal_table[level][status]) for level in summary["vertical_order"])
        lines.append(f"| {status_labels[status]} | {counts.get(status, 0)} | {columns} |")
    # 位置對照必須保留 summary 的原始面板順序與經緯度，避免新版顯示名稱
    # 取代或重排既有 H1--H5，讓 PI 能逐列回查原 preview 契約。
    position_rows = [
        f"| 位置{int(panel['panel'])} | 原 H{int(panel['panel'])} | "
        f"{float(panel['receptor_lon']):.8f} | {float(panel['receptor_lat']):.8f} |"
        for panel in summary["horizontal_panels"]
    ]
    position_insert_at = lines.index("## 水層定義與停止統計")
    lines[position_insert_at:position_insert_at] = [
        "",
        "### 位置對照（新版位置與原 H 標籤）",
        "",
        "下表直接取自 `summary.horizontal_panels`，讓局部圖的「位置1–5」可回對原 preview 的 H1–H5 與經緯度。",
        "",
        "| 新版位置 | 原 preview 標籤 | 經度（°E） | 緯度（°N） |",
        "|---|---|---:|---:|",
        *position_rows,
        "",
    ]
    lines += [
        "",
        f"本次圖面共保留 {len(data['particles'])} 顆粒子與 {len(data['observations'])} 筆模型保存紀錄；"
        f"完成1小時回溯 {counts.get('max_age', 0)} 顆、"
        f"離開計算範圍 {counts.get('flow_domain_open_exit', 0)} 顆。",
        f"垂向資料缺值原樣留空：eta_m 空白 {missing_eta} 筆、bed_z_m 空白 {missing_bed} 筆，"
        "不補成零；高度軸標示為「高度（公尺；向上為正）」，時間軸統一 0–60 分鐘。",
        "",
        "## 取樣失敗診斷界線",
        "",
    ]
    if sampling_verified:
        lines.append(
            f"本次 {len(numerical_rows)} 顆數值失敗只有在 particles.csv 與 summary.json 雙重"
            "核對後才標為「取樣失敗停止」："
            f"每筆 failure_reason=invalid_velocity_sample、qc_flags=16，停止 age 為 {numerical_age_text} 秒。"
        )
        lines.append("這只描述本次保存的停步原因，不暗示取樣問題已修復，也不能泛化到其他 numerical_failure。")
    else:
        lines.append(
            f"本次 {len(numerical_rows)} 顆數值失敗未通過特定取樣原因的 CSV／summary 雙重"
            "核對，因此圖面只標「數值停止」，"
            "不把 numerical_failure 泛化為取樣失敗。"
        )
    lines += [
        "",
        "## 座標、來源與限制",
        "",
        f"區域總覽使用經緯度；局部圖沿用 AEQD（等距方位投影）公尺座標並只作平移，中心為經度 {center[0]}、"
        f"緯度 {center[1]}。五個局部面板各自等比例且範圍不同；不移除、放大或抖動任何粒子座標。",
        "海岸沿用已確認的 CRS84 `taiwan_exact_coastline.geojson`（1905 個有效 Polygon／MultiPolygon）；"
        "它只作地理參照，本次 PNG 不加入資料來源授權註記或警語。",
        "本次停止圖與垂向圖是本專案補充診斷，不能替代正式 report 的收斂、獨立觀測驗證或來源歸因。"
        "逆向路徑變淺不表示正向材料上浮；這組 M=1 路徑不代表絕對來源機率。",
        "",
        "### 原始七檔前後 SHA-256",
        "",
        "| 輸入 | bytes | before SHA-256 | after SHA-256 |",
        "|---|---:|---|---|",
    ]
    for name, before in sources_before.items():
        after = sources_after[name]
        lines.append(f"| {name} | {before['size_bytes']} | `{before['sha256']}` | `{after['sha256']}` |")
    lines += [
        "",
        "## 本機重建",
        "",
        "請在專案根目錄以自己的路徑變數執行；輸出目錄必須是全新目錄。",
        "",
        "```bash",
        "UV_CACHE_DIR=work/uv-cache MPLCONFIGDIR=work/matplotlib-cache \\",
        "uv run python3 scripts/build_pilot_coastline_preview.py \\",
        "  --style baytrace \\",
        "  --preview-dir \"$PILOT_PREVIEW\" --domain \"$DOMAIN_GEOMETRY\" \\",
        "  --open-boundary \"$OPEN_BOUNDARY_GEOMETRY\" --coastline \"$COASTLINE_GEOJSON\" \\",
        "  --output-dir \"$BAYTRACE_STYLE_PREVIEW\"",
        "```",
        "",
        "manifest.json 保存 style、style_version、四張 PNG 與 README SHA、原始七檔 before／after SHA、"
        "script SHA、實際命令、重建命令、幾何綁定及停止標籤核對結果；manifest 本身不做循環雜湊。",
        "失敗保留新輸出目錄供診斷，既有輸出、輸入檔、原無底圖 preview 與 legacy 成果均不覆寫。",
    ]
    return "\n".join(lines) + "\n"


def build_coastline_preview(
    preview_dir: str | Path,
    domain_path: str | Path,
    open_path: str | Path,
    coastline_path: str | Path,
    output_dir: str | Path,
    *,
    style: str = "legacy",
) -> dict[str, Any]:
    """在全新目錄建立 legacy 或新版四圖，回傳完成的 JSON 內容。

    ``legacy``（預設）保留上一輪兩張水平圖、H1--H5 標籤及舊 manifest 契約；
    ``baytrace`` 只在全新目錄寫入水平區域／局部、垂向高度、停止原因四張 PNG，並
    保存新版 style/version 與來源前後 SHA。兩個分支都先拒絕既有目錄或失效連結，
    再讀取並核對既有輸入；核對失敗均保留新目錄供診斷，不清理、不覆寫輸入或舊成果。
    新版僅使用 preview 的 CSV／summary，不讀 forcing、run 或重新積分。
    """

    if style not in {"legacy", "baytrace"}:
        raise ValueError("style 必須是 legacy 或 baytrace")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"輸出目錄已存在，拒絕覆寫：{output}")
    preview = Path(preview_dir)
    paths = {
        name: preview / name
        for name in ("manifest.json", "summary.json", "particles.csv", "observations.csv")
    }
    paths.update(domain=Path(domain_path), open_boundary=Path(open_path), coastline=Path(coastline_path))
    before = _source_snapshot(paths)
    data = load_plot_inputs(preview, domain_path, open_path, coastline_path)
    summary = data["summary"]
    counts = dict(Counter(particle["status"] for particle in data["particles"]))
    expected = {"max_age": 9, "flow_domain_open_exit": 8, "numerical_failure": 3}
    if summary["run_id"] != "b-fishinggear-m1-1h-r2" or counts != expected:
        raise ValueError("必須使用本次已驗收的 B 區 r2 run 與 9/8/3 停止計數")
    if counts != {status: count for status, count in summary["terminal_counts"].items() if count}:
        raise ValueError("CSV 停止計數與 summary 不符")

    args = [
        "uv",
        "run",
        "python3",
        "scripts/build_pilot_coastline_preview.py",
        "--preview-dir",
        str(preview),
        "--domain",
        str(domain_path),
        "--open-boundary",
        str(open_path),
        "--coastline",
        str(coastline_path),
        "--output-dir",
        str(output),
    ]
    if style == "baytrace":
        args.extend(["--style", "baytrace"])
    prefix = "UV_CACHE_DIR=work/uv-cache MPLCONFIGDIR=work/matplotlib-cache "
    invocation = prefix + shlex.join(args)
    output_argument_index = args.index("--output-dir") + 1
    args[output_argument_index] = str(output.with_name(output.name + "-rebuild"))
    rebuild = prefix + shlex.join(args)
    output.mkdir(parents=True, exist_ok=False)
    if style == "legacy":
        # 預設分支完整保留上一輪兩圖與 manifest 欄位，讓舊成果仍可由同一命令重建。
        render_overview(data, output / "horizontal_overview.png")
        render_local(data, output / "horizontal_local.png")
        with (output / "README.md").open("x", encoding="utf-8") as stream:
            stream.write(_output_readme(data, before, rebuild))
        after = _source_snapshot(paths)
        if before != after:
            raise ValueError("來源前後 bytes/SHA 不一致；保留新目錄供診斷，不建立完成清單")

        font_path = Path(_display_font().get_file())
        manifest = {
            "artifact_kind": "pilot-coastline-preview-v1",
            "schema_version": "1.0.0",
            "run_id": summary["run_id"],
            "particle_count": len(data["particles"]),
            "observation_count": len(data["observations"]),
            "terminal_counts": summary["terminal_counts"],
            "horizontal_panels": summary["horizontal_panels"],
            "projection": {key: summary["projection"][key] for key in ("kind", "units", "center_lonlat")},
            "sources": before,
            "source_hashes_after": {name: record["sha256"] for name, record in after.items()},
            "source_unchanged": True,
            "geometry_canonical_hashes": {
                name: data["source"][name]["canonical_sha256"] for name in ("domain", "open_boundary")
            },
            "geometry_owner": {
                "analysis_region_id": "B",
                "flow_domain_id": data["domain_record"]["flow_domain_id"],
            },
            "coastline_provenance": {
                "provider": None,
                "vintage": None,
                "license": None,
                "use": "internal_PI_geographic_reference",
            },
            "script_sha256": sha256_file(Path(__file__)),
            "font": {"filename": font_path.name, "sha256": sha256_file(font_path)},
            "invocation_command": invocation,
            "rebuild_command": rebuild,
            "files": {
                name: {"size_bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
                for name in ("horizontal_overview.png", "horizontal_local.png", "README.md")
            },
        }
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return manifest

    # 新版只增加四張圖與新版說明，不回讀 forcing 或改動既有 preview 資料。
    render_baytrace_overview(data, output / "horizontal_overview.png")
    render_baytrace_local(data, output / "horizontal_local.png")
    render_baytrace_depth(data, output / "depth_age.png")
    render_baytrace_terminal(data, output / "terminal_counts.png")
    after = _source_snapshot(paths)
    if before != after:
        raise ValueError("來源前後 bytes/SHA 不一致；保留新目錄供診斷，不建立完成清單")
    status_labels = _baytrace_status_labels(data)
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(_baytrace_readme(data, before, after, status_labels))
    # README 落檔不會改動七個來源，但仍在建立 manifest 前再核對一次，縮小競爭窗口。
    after = _source_snapshot(paths)
    if before != after:
        raise ValueError("來源前後 bytes/SHA 不一致；保留新目錄供診斷，不建立完成清單")

    font_path = Path(_display_font().get_file())
    output_names = (
        "horizontal_overview.png",
        "horizontal_local.png",
        "depth_age.png",
        "terminal_counts.png",
        "README.md",
    )
    terminal_table = _baytrace_terminal_table(data)
    manifest = {
        "artifact_kind": "pilot-coastline-preview-v1",
        "schema_version": "1.0.0",
        "style": "baytrace",
        "style_version": _BAYTRACE_STYLE_VERSION,
        "run_id": summary["run_id"],
        "particle_count": len(data["particles"]),
        "observation_count": len(data["observations"]),
        "terminal_counts": summary["terminal_counts"],
        "terminal_counts_by_vertical": terminal_table,
        "terminal_status_labels": status_labels,
        "sampling_failure_verified": _sampling_failure_is_verified(data),
        "horizontal_panels": summary["horizontal_panels"],
        "projection": {key: summary["projection"][key] for key in ("kind", "units", "center_lonlat")},
        "water_layer_definitions": {
            level: {"name_zh": _baytrace_layer_label(level), "definition": _baytrace_layer_definition(level)}
            for level in summary["vertical_order"]
        },
        "sources_before": before,
        "sources_after": after,
        "source_hashes_after": {name: record["sha256"] for name, record in after.items()},
        "source_unchanged": True,
        "geometry_canonical_hashes": {
            name: data["source"][name]["canonical_sha256"] for name in ("domain", "open_boundary")
        },
        "geometry_owner": {
            "analysis_region_id": "B",
            "flow_domain_id": data["domain_record"]["flow_domain_id"],
        },
        "coastline_provenance": {
            "provider": None,
            "vintage": None,
            "license": None,
            "use": "internal_PI_geographic_reference",
        },
        "script_sha256": sha256_file(Path(__file__)),
        "font": {"filename": font_path.name, "sha256": sha256_file(font_path)},
        "invocation_command": invocation,
        "rebuild_command": rebuild,
        "files": {
            name: {"size_bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
            for name in output_names
        },
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """解析明示本機路徑與圖面 style；成功回傳 0，拒絕回傳 2 並保留診斷目錄。"""

    parser = argparse.ArgumentParser(description="由 B 區已完成 r2 preview 獨立重繪水平與診斷圖")
    parser.add_argument("--preview-dir", required=True, type=Path)
    parser.add_argument("--domain", required=True, type=Path)
    parser.add_argument("--open-boundary", required=True, type=Path)
    parser.add_argument("--coastline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--style", choices=("legacy", "baytrace"), default="legacy")
    args = parser.parse_args(argv)
    try:
        manifest = build_coastline_preview(
            args.preview_dir,
            args.domain,
            args.open_boundary,
            args.coastline,
            args.output_dir,
            style=args.style,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "run_id": manifest["run_id"],
                "particle_count": manifest["particle_count"],
                "observation_count": manifest["observation_count"],
                "style": manifest.get("style", "legacy"),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "load_plot_inputs",
    "render_overview",
    "render_local",
    "render_baytrace_overview",
    "render_baytrace_local",
    "render_baytrace_depth",
    "render_baytrace_terminal",
    "build_coastline_preview",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
