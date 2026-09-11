"""由已完成的 A／B／C／D 四區五站 preview 獨立重繪 legacy 或新版成果圖及來源清單。

預設 ``legacy`` 只輸出原契約的兩張水平圖；指定 ``--style baytrace`` 才增加區域／
局部水平、垂向高度及停止原因四張新版圖，並各自保存對應的繁中 README 與 manifest。

本模組只讀既有 preview 目錄中的
``manifest.json``、``summary.json``、``particles.csv`` 與 ``observations.csv``，
以及呼叫端明示的已驗收站點 domain/open-boundary manifest 和 CRS84 海岸
FeatureCollection。它不讀海流或波浪陣列、不重新取樣、不重新積分，也不改寫任何
輸入檔。回傳資料仍保留 CSV 的全部欄位與列順序，供後續繪圖階段使用原始粒子、
模型保存紀錄及 summary 的原始水平面板順序。

domain/open manifest 的語意雜湊沿用專案既有 ``manifests._read_json``；各區 domain
與各自 flow open owner 必須同時符合 preview summary 的
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
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, MultiPolygon, Polygon, shape

from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.manifests import _read_json
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.pilot_preview import (
    _acquire_nfs_publish_lock,
    _canonical_json_bytes,
    _publish_nfs_marker_preview,
    _release_nfs_publish_lock,
    _safe_path,
    _validate_staged_preview,
    _verified_storage_gate_evidence,
    validate_published_artifact,
    validate_published_preview,
)


# 預覽成果會保存可重建的 shell 命令，但不能把本機或 SERVER 的實際絕對路徑寫入
# README／manifest。這四個 LBT_* 根目錄必須先由 SERVER 儲存政策閘門驗證；命令執行時
# 再由 shell 將它們分別提供給 uv、Matplotlib、XDG 快取與暫存檔，且在變數未設定時
# 立即停止，避免重建時回退到未驗證的專案相對路徑或主機暫存區。
def _verified_server_cache_prefix() -> str:
    """建立使用已驗證 NFS 根目錄的 fail-closed shell 環境前綴。

    README 與 manifest 只能保存環境變數名稱，不能保存產製者主機的實際路徑。
    shell ``:?`` 展開會在任一根目錄未設定時中止重建命令，確保操作端先完成
    SERVER 儲存政策閘門，再把同一組四個根目錄提供給 uv、Matplotlib、XDG
    快取與暫存檔。``PYTHONDONTWRITEBYTECODE`` 則避免預覽重建把未登錄的
    Python bytecode 寫入結果或程式目錄。
    """

    return (
        'UV_CACHE_DIR="${LBT_UV_CACHE_ROOT:?set verified NFS root}" '
        'MPLCONFIGDIR="${LBT_MPL_CACHE_ROOT:?set verified NFS root}" '
        'XDG_CACHE_HOME="${LBT_XDG_CACHE_ROOT:?set verified NFS root}" '
        'TMPDIR="${LBT_TMP_ROOT:?set verified NFS root}" '
        "PYTHONDONTWRITEBYTECODE=1 "
    )


def _validate_coastline_coverage(domain_geometry: Polygon, land_geometries: Sequence[Any]) -> None:
    """驗證海岸資料的整體 bounds 涵蓋 domain，且至少有一個 land feature 相交。

    海岸 GeoJSON 只作經緯度底圖參照，不能被當成 forcing mask；這個 gate 的目的只是
    防止誤傳另一區或截短的海岸檔，讓圖面外框與底圖空間範圍可被解讀。先以所有有效
    feature 的 bounds 組成涵蓋範圍，再要求至少一個 feature 與 domain 相交；不以
    feature 數量或單一 polygon 的大小代替五站各自的 domain 綁定。
    """

    if not land_geometries:
        raise ValueError("海岸 GeoJSON 不得為空")
    coastline_min_x = min(float(geometry.bounds[0]) for geometry in land_geometries)
    coastline_min_y = min(float(geometry.bounds[1]) for geometry in land_geometries)
    coastline_max_x = max(float(geometry.bounds[2]) for geometry in land_geometries)
    coastline_max_y = max(float(geometry.bounds[3]) for geometry in land_geometries)
    domain_min_x, domain_min_y, domain_max_x, domain_max_y = domain_geometry.bounds
    if not (
        coastline_min_x <= domain_min_x
        and coastline_min_y <= domain_min_y
        and coastline_max_x >= domain_max_x
        and coastline_max_y >= domain_max_y
    ):
        raise ValueError("海岸資料 bounds 未涵蓋站點 domain")
    if not any(geometry.intersects(domain_geometry) for geometry in land_geometries):
        raise ValueError("海岸資料沒有 land feature 與站點 domain 相交")


def load_plot_inputs(
    preview_dir: str | Path,
    domain_path: str | Path,
    open_path: str | Path,
    coastline_path: str | Path,
    *,
    storage_gate_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """讀取並核對海岸版圖面輸入，回傳不含重新取樣資料的唯讀組合。

    ``preview_dir`` 必須是既有 ``pilot-preview-v1`` 目錄；manifest 宣告的 summary、
    particles 與 observations bytes 會在讀 CSV 前逐一核對。CSV 不做排序、插值或
    欄位刪減，列中的文字值也原樣保留，讓後續繪圖仍可追溯到原 preview。

    ``domain_path`` 與 ``open_path`` 必須是 schema 1.0.0、approved 的 JSON。只選取
    summary 指定 analysis region 的 domain record 及其 ``flow_domain`` open record，並以
    summary 的 canonical geometry hash 綁定；open line 還必須與 domain polygon
    exterior 相等，避免把其他區或 local record 誤畫成目前站點外框。海岸 GeoJSON 必須
    明示 OGC CRS84（座標順序為 longitude、latitude），每個 land feature 轉為
    Shapely Polygon/MultiPolygon。載入器不繪圖；繪圖函式可填灰色背景，但不以陸地
    判定粒子是否有效。

    若 preview 由 NFS completion-marker 協定發布，必須先驗證 ``.complete``、manifest
    SHA、Git／dirty provenance 與 storage gate binding；未完成目錄不會進入 CSV reader。
    回傳包含原始 payload、CSV 全列、站點 Shapely 幾何、以 summary 中心建立的
    ``DomainProjection`` 與原始／語意 SHA。CSV 附 bytes 計數，完整檔案大小由 build
    wrapper 記錄。此函式只讀檔；缺檔拋出 ``OSError``，內容或來源綁定不符拋出
    ``ValueError``。
    """

    preview = Path(preview_dir)
    validate_published_preview(preview, storage_gate_evidence=storage_gate_evidence)
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
    particle_count = _integer_value(summary.get("particle_count"))
    observation_count = _integer_value(summary.get("observation_count"))
    if particle_count is None or particle_count <= 0:
        raise ValueError("preview particle_count 必須是正整數")
    if observation_count is None or observation_count < 0:
        raise ValueError("preview observation_count 必須是非負整數")
    _horizon_seconds(summary)
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
    if len(particles) != particle_count or len(observations) != observation_count:
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
    site_region, _, _, expected_flow_domain = _site_context(summary)
    if region != site_region or not isinstance(center, list) or len(center) != 2:
        raise ValueError("preview projection 不是已登錄站點 AEQD")
    domain_rows = [
        row for row in domain.get("records", []) if row.get("analysis_region_id") == site_region
    ]
    if len(domain_rows) != 1:
        raise ValueError("domain manifest 的站點 record 不唯一")
    domain_record = domain_rows[0]
    flow_domain_id = domain_record.get("flow_domain_id")
    if flow_domain_id != expected_flow_domain:
        raise ValueError("domain manifest 的 flow_domain_id 與五站契約不符")
    open_rows = [
        row
        for row in open_boundary.get("records", [])
        if row.get("owner_kind") == "flow_domain"
        and row.get("owner_id") == flow_domain_id
        and row.get("analysis_region_id") == site_region
    ]
    if len(open_rows) != 1:
        raise ValueError("open manifest 的站點 flow owner 不唯一")
    open_record = open_rows[0]
    domain_geometry = shape(domain_record["geometry"])
    open_geometry = shape(open_record["geometry"])
    if not isinstance(domain_geometry, Polygon) or not domain_geometry.is_valid:
        raise ValueError("站點 domain geometry 不是有效 Polygon")
    if not isinstance(open_geometry, LineString) or not open_geometry.is_valid:
        raise ValueError("站點 flow open geometry 不是有效 LineString")
    if not open_geometry.equals(domain_geometry.exterior):
        raise ValueError("站點 flow open line 未與 domain exterior 相等")

    coastline, coastline_sha, coastline_canonical, _ = _read_json(coastline_path)
    # 本入口專用於使用者已確認的海岸檔，不以相同 feature 數量接受另一份地圖。
    if coastline_sha != "9e2e0ac9bc527aca87d89332cd428fdcb776eefbf94a85dd70f887f729b95fdd":
        raise ValueError("海岸檔 SHA-256 與本次已確認來源不符")
    crs_name = coastline.get("crs", {}).get("properties", {}).get("name")
    if coastline.get("type") != "FeatureCollection" or crs_name != "urn:ogc:def:crs:OGC:1.3:CRS84":
        raise ValueError("海岸 GeoJSON 必須明示 CRS84 FeatureCollection")
    land_geometries = [shape(feature["geometry"]) for feature in coastline.get("features", [])]
    if (
        len(land_geometries) != 1905
        or not all(isinstance(geometry, (Polygon, MultiPolygon)) for geometry in land_geometries)
        or not all(geometry.is_valid and not geometry.is_empty for geometry in land_geometries)
    ):
        raise ValueError("海岸 GeoJSON features 不符合既有 1905 筆有效 land polygons")
    _validate_coastline_coverage(domain_geometry, land_geometries)

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
"""既有垂向層的固定圖色；顏色只表示層別，不表示來源機率。"""

_STATUS_MARKERS = {
    "flow_domain_open_exit": "s",
    "coast_contact": "P",
    "surface_regime_exit": "v",
    "deposited": "d",
    "forcing_start": "D",
    "data_gap": "h",
    "max_age": "^",
    "numerical_failure": "X",
}
"""八種已登錄終止狀態各自使用固定幾何標記，避免終點被混稱為來源。

``forcing_start`` 的菱形（``D``）與 ``max_age`` 的三角形、開放邊界離域的方形及
``numerical_failure`` 的紅色叉號分開；這只是在圖面保留狀態差異，不代表任何狀態
本身就是科學來源判定。
"""

_VERTICAL_LABELS = {
    "near_bed": "近海床",
    "mid_lower_water_column": "中下水層",
    "mid_upper_water_column": "中上水層",
    "upper_water_column": "上水層",
}
"""初始水層的中文圖例，與 CSV 層別及固定顏色一一對應。"""

_BAYTRACE_STYLE_VERSION = "1.2.0"
"""新版圖面呈現契約版本；1.2.0 分開設定回溯上限與資料窗端點文字。"""

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
_BAYTRACE_FORCING_START_COLOR = "#7b3294"
_BAYTRACE_FORCING_START_EARLY_COLOR = "#d95f02"
_FORCING_START_AGE_TOLERANCE_SECONDS = 1e-4
"""新版端點顏色與 forcing_start age 核對容差；容差為固定 0.1 毫秒。

本容差比本次真資料約 1.6e-5 秒的誤差大一個數量級，但仍遠小於 300 秒保存／輸出
間隔，不會把真正提前數百秒或數秒的 forcing_start 停止吞成「完成設定時長」。紫色
表示在容差內抵達設定 horizon，橙色表示明顯提前；兩者都不是紅色數值失敗叉號。
"""

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

_SITE_BY_REGION = {
    "A": frozenset({"gongliao", "guishan"}),
    "B": frozenset({"hsinchu"}),
    "C": frozenset({"houwan"}),
    "D": frozenset({"lienchiang"}),
}
_SITE_LABEL_ZH = {
    "gongliao": "貢寮",
    "guishan": "龜山島",
    "hsinchu": "新竹外海",
    "houwan": "後灣",
    "lienchiang": "連江",
}
_FLOW_DOMAIN_BY_SITE = {
    "gongliao": "northeast_taiwan_common_cache_v3",
    "guishan": "northeast_taiwan_common_cache_v3",
    "hsinchu": "hsinchu_cache_v3",
    "houwan": "houwan_nmmba_cache_v3",
    "lienchiang": "lienchiang_common_cache_v3",
}
"""五站固定的 analysis-region、study-site 與 flow-domain 來源契約。"""
_BAYTRACE_CANDIDATE_POLICY_ID = "houwan_red_frame_two_subregions_anchor_first_maximin_2plus3_v1"
"""C 區候選子區政策識別碼；只作 provenance 核對，不將 polygon 當 forcing mask。"""


def _site_context(summary: dict[str, Any]) -> tuple[str, str, str, str]:
    """由 preview summary 核對 region／site／flow-domain，回傳圖面標題語境。

    A 區含貢寮與龜山島兩個獨立站點；它們可共用 A 的 flow-domain geometry，但
    preview、受體、粒子分母與成果目錄必須由各自 summary 綁定。此 allow-list 只防止
    未登錄站點借用其他區圖面契約；flow-domain 也必須符合五站已核定的來源 ID，
    不重建或推測 forcing／海岸資料。
    """

    projection = summary.get("projection")
    region = projection.get("analysis_region_id") if isinstance(projection, dict) else None
    site = summary.get("study_site_id")
    if (
        not isinstance(region, str)
        or not isinstance(site, str)
        or site not in _SITE_BY_REGION.get(region, frozenset())
    ):
        raise ValueError("preview study site 與 analysis region 繫結不符")
    expected_flow_domain = _FLOW_DOMAIN_BY_SITE[site]
    declared_flow_domain = projection.get("flow_domain_id") if isinstance(projection, dict) else None
    if declared_flow_domain is not None and declared_flow_domain != expected_flow_domain:
        raise ValueError("preview flow domain 與 study site 繫結不符")
    return region, site, _SITE_LABEL_ZH[site], expected_flow_domain


def _site_title(summary: dict[str, Any]) -> str:
    """建立站點化圖面標題，不把回溯終點誤稱為來源。"""

    region, _, site_label, _ = _site_context(summary)
    return f"{region}區{site_label}沉降粒子反向追蹤"

_BAYTRACE_STATUS_LABELS = {
    "flow_domain_open_exit": "離開計算範圍",
    "coast_contact": "接觸海岸",
    "surface_regime_exit": "離開表面適用範圍",
    "deposited": "沉積",
    "forcing_start": "到達驅動資料起點",
    "data_gap": "資料缺口",
    "max_age": "完成回溯",
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
"""停止圖分段色；只表示分類數量，不取代水平／垂向端點的狀態標記。"""


def _display_font():
    """沿用原預覽字型 helper，優先使用本機既有 Arial Unicode 正常筆重。"""

    from lagrangian_backtracking.pilot_preview import _plot_font

    candidate = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font, _, _ = _plot_font(candidate if candidate.is_file() else None)
    font.set_weight("normal")
    font.set_size(11)
    return font


def _subtitle(summary: dict[str, Any]) -> str:
    """建立共用時間副題，完整列出到達時刻與回溯至的 UTC 日期時間。

    ``arrival_utc`` 是受體／指定位置的到達時刻，回溯至時間由已核對的
    ``settings.horizon_seconds`` 計算。完整顯示兩個日期可避免跨日回溯被誤讀成同一
    日期內的時間循環；此文字只描述資料時間窗，不把模型保存紀錄稱為實測觀測。
    """

    horizon = _horizon_seconds(summary)
    return (
        f"{_time_window_label(summary)}｜"
        f"回溯{horizon / 3600:g}小時｜{summary['particle_count']}顆粒子"
    )


def _time_window(summary: dict[str, Any]) -> tuple[datetime, datetime]:
    """回傳以 UTC 表示的到達與回溯至時刻，拒絕沒有時區的時間字串。

    preview 的 ``arrival_utc`` 是資料交換用的 ISO-8601 到達時間；本函式只依回溯
    設定計算顯示端點，不改寫 summary，也不從粒子 ``age_seconds`` 反推事件。forcing
    起點的標籤是否適用仍由輸入的 ``status=forcing_start`` 決定，不能因時間相減吻合
    就把其他終止狀態改成 forcing_start。
    """

    try:
        arrival = datetime.fromisoformat(str(summary["arrival_utc"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("preview arrival_utc 必須是含時區的 ISO-8601 時間") from None
    if arrival.tzinfo is None:
        raise ValueError("preview arrival_utc 必須明示時區")
    arrival_utc = arrival.astimezone(UTC)
    return arrival_utc, arrival_utc - timedelta(seconds=_horizon_seconds(summary))


def _time_window_label(summary: dict[str, Any]) -> str:
    """產生明示完整日期的 UTC 時間列，避免跨日回溯只剩 ``HH:MM``。

    這個顯示 helper 不代表資料中每一顆粒子都一定抵達回溯至時刻；是否完成本次
    forcing 時間窗，必須仍以各粒子的原始終止狀態判定。
    """

    arrival, start = _time_window(summary)
    return f"到達：{arrival:%Y-%m-%d %H:%M} UTC；回溯至：{start:%Y-%m-%d %H:%M} UTC"


def _horizon_seconds(summary: dict[str, Any]) -> float:
    """讀取正的回溯秒數，讓所有圖面共用同一個時間設定且拒絕無效軸範圍。

    ``summary.settings.horizon_seconds`` 是模擬設定的回溯上限，單位為秒；此處只做
    顯示前的型別與有限值核對，不依據圖面資料推算時間，也不改寫 summary。零或負值
    會讓完成狀態及深度軸失去意義，因此直接拒絕產圖。
    """

    settings = summary.get("settings")
    value = settings.get("horizon_seconds") if isinstance(settings, dict) else None
    try:
        horizon = float(value)
    except (TypeError, ValueError):
        raise ValueError("preview horizon_seconds 必須是有限正數") from None
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("preview horizon_seconds 必須是有限正數")
    return horizon


def _horizon_hours_text(summary: dict[str, Any]) -> str:
    """把設定中的回溯秒數轉成不失真的小時文字，供圖例與 README 共用。"""

    return f"{_horizon_seconds(summary) / 3600:g}"


def _max_age_label(summary: dict[str, Any]) -> str:
    """由回溯設定建立 max-age 的中文標籤，避免把固定時長寫死。"""

    return f"完成{_horizon_hours_text(summary)}小時回溯"


def _baytrace_max_age_label(summary: dict[str, Any]) -> str:
    """建立 BayTrace 專用的 max_age 標籤，明確表示它是設定上限而非資料窗端點。

    legacy 圖面仍沿用舊的 ``完成N小時回溯`` 文字；新版 BayTrace 必須與
    ``forcing_start`` 的「到達本次資料時間窗起點（完成N小時回溯）」分開，避免
    讀者把數值上限事件誤讀成抵達 forcing 資料窗起點。小時數直接由 summary
    的 ``settings.horizon_seconds`` 計算，不把本次 24 小時案例寫死。
    """

    return f"到達設定回溯時間上限（{_horizon_hours_text(summary)}小時）"


def _forcing_start_age_state(age_seconds: Any, horizon_seconds: float) -> str:
    """依單顆粒子 age 與設定 horizon 判定 forcing_start 的顯示狀態。

    ``completed`` 只在 ``abs(age_seconds - horizon_seconds)`` 不超過固定
    ``1e-4`` 秒時成立；``early`` 表示明顯早於 horizon，其他值則保守標為
    ``overrun`` 或 ``unverified``。這個判定只控制圖面文字與顏色，絕不修改粒子的
    原始 ``status=forcing_start``，也不把抵達 forcing 資料窗起點推論成污染來源。
    """

    try:
        age = float(age_seconds)
    except (TypeError, ValueError):
        return "unverified"
    if not math.isfinite(age) or not math.isfinite(horizon_seconds):
        return "unverified"
    delta = age - horizon_seconds
    if abs(delta) <= _FORCING_START_AGE_TOLERANCE_SECONDS:
        return "completed"
    if delta < -_FORCING_START_AGE_TOLERANCE_SECONDS:
        return "early"
    return "overrun"


def _forcing_start_age_states(data: dict[str, Any]) -> list[str]:
    """逐列讀取 forcing_start 粒子的 age，回傳每顆粒子的保守顯示分類。

    只有原始 status 已是 ``forcing_start`` 的粒子會進入清單；其他終止狀態即使
    age 恰好接近 horizon，也不會被標成資料窗端點。清單順序沿用 particles.csv，
    方便 manifest 與測試追溯每顆粒子的判定數量。
    """

    horizon = _horizon_seconds(data["summary"])
    return [
        _forcing_start_age_state(particle.get("age_seconds"), horizon)
        for particle in data["particles"]
        if particle.get("status") == "forcing_start"
    ]


def _forcing_start_state_label(summary: dict[str, Any], state: str) -> str:
    """回傳單一 forcing_start age 分類的中文端點標籤。"""

    if state == "completed":
        return f"到達本次資料時間窗起點（完成{_horizon_hours_text(summary)}小時回溯）"
    if state == "early":
        return "提前到達驅動資料起點"
    if state == "overrun":
        return "到達驅動資料起點（回溯時長超過設定）"
    return "到達驅動資料起點（回溯時長未核對）"


def _forcing_start_label(data: dict[str, Any]) -> str:
    """由每顆 forcing_start 粒子的 age 建立整批安全的圖例／摘要標籤。

    全部粒子在容差內時才使用「完成設定時長」；全部明顯提前時使用明確的「提前」
    文案。若同一批資料混合兩種狀態，則使用不宣稱全批完成的保守文字；水平／垂向
    renderer 仍會依每顆粒子的 age 類別使用對應顏色，避免 aggregate label 掩蓋個體
    差異。原始 status 與 terminal_counts 永遠不被改寫。
    """

    summary = data["summary"]
    states = set(_forcing_start_age_states(data))
    if states == {"completed"}:
        return _forcing_start_state_label(summary, "completed")
    if states == {"early"}:
        return _forcing_start_state_label(summary, "early")
    if "early" in states:
        return "到達驅動資料起點（含提前到達；各粒子依 age 判讀）"
    if "overrun" in states:
        return "到達驅動資料起點（回溯時長未全部符合設定）"
    return "到達驅動資料起點（回溯時長未核對）"


def _position_order_note(summary: dict[str, Any]) -> str:
    """說明水平總覽的位置編號是受體／指定位置，不是粒子時間序列。

    編號沿用 summary 的原始水平面板順序，避免讀者把位置1到位置N誤看成同一粒子
    隨時間移動的先後順序；目前真實案例的 N=5，因此輸出「位置1–5是五個受體／指定
    位置，不是時間順序」。
    """

    count = len(summary["horizontal_panels"])
    count_text = {1: "一個", 2: "兩個", 3: "三個", 4: "四個", 5: "五個"}.get(
        count, f"{count}個"
    )
    return f"位置1–{count}是{count_text}受體／指定位置，不是時間順序。"


def _depth_axis_limits_and_ticks(summary: dict[str, Any]) -> tuple[float, list[float]]:
    """回傳垂向圖的分鐘上限與四等分刻度，確保軸範圍跟隨實際回溯設定。

    圖面資料的 ``age_seconds`` 仍直接來自 preview；此 helper 只把已核對的
    ``horizon_seconds`` 轉成分鐘，並在 0 與回溯上限間建立五個等距刻度，讓不同
    回溯長度的圖都能使用同一資料契約。
    """

    horizon_minutes = _horizon_seconds(summary) / 60.0
    quarter = horizon_minutes / 4.0
    return horizon_minutes, [index * quarter for index in range(5)]


def _legend_handles(summary: dict[str, Any], levels: list[str], *, include_land: bool = False) -> list:
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
        ("max_age", _max_age_label(summary)),
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
    """以已載入的粒子與模型保存紀錄繪製單張經緯度總覽 PNG。

    此函式只由 ``build_coastline_preview(style="legacy")`` 呼叫，用來維持既有
    兩圖相容成果；新版 BayTrace 圖面使用 ``render_baytrace_overview``，不共用
    舊版輸出契約。

    每條線只連接同一 ``particle_id`` 的保存觀測；位置由既有
    ``DomainProjection.unproject`` 將 AEQD 公尺轉回經度、緯度，沒有重新取樣或
    重新積分。海岸 FeatureCollection 交給 Shapely 的 ``plot_polygon`` 處理，因而
    保留 Polygon/MultiPolygon 的孔洞；各站 flow open boundary 只畫一次。土地置於底層，軌跡、起點
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

        # 使用 summary 的原始水平受體順序，避免由座標排序重新定義面板標籤。
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

        ax.set_title(_site_title(summary), fontproperties=font, fontsize=18, pad=36)
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
            handles=_legend_handles(summary, levels, include_land=True),
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
    """以原 AEQD 公尺座標繪製既有局部面板，第六格放共用圖例與簡短圖說。

    此函式只由 ``build_coastline_preview(style="legacy")`` 呼叫，用來保留舊成果
    的兩張水平圖；新版 BayTrace 局部圖由 ``render_baytrace_local`` 獨立處理。

    summary 的水平分組與垂向層別決定各面板內容。各組以第一顆粒子的初始保存 XY
    作共同原點，只平移座標；線段、終點及已投影的站點外框接受同量平移，不重新取樣、
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
            expected_layer_count = len(summary["vertical_order"])
            if len(selected) != expected_layer_count:
                raise ValueError(f"局部面板必須保留原始 {expected_layer_count} 個水層粒子")
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
            ax.set_title(f"H{panel['panel']}｜{len(selected)} 顆粒子", fontproperties=font, fontsize=14)
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
            handles=_legend_handles(summary, list(summary["vertical_order"])),
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
        fig.suptitle(_site_title(summary), fontproperties=font, fontsize=20, y=0.985)
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


def _baytrace_candidate_regions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """驗證設定檔保存的候選子區，供 README／manifest provenance 使用。

    候選 polygon 是 input derivation 從設定檔保存的 WGS84 GeoJSON；這裡只核對其
    CRS、有效性與 C 區 2+3 配額，不使用粒子座標反推框線，也不把候選邊界當成海陸
    或 forcing mask。此 helper 不繪圖；缺少候選欄位的 A／B／D 與舊 B preview 回傳
    空清單，維持各站既有圖面契約。
    """

    raw_regions = summary.get("receptor_candidate_regions")
    raw_policy = summary.get("receptor_candidate_selection")
    if raw_regions is None and raw_policy is None:
        return []
    if not isinstance(raw_regions, list) or not isinstance(raw_policy, dict):
        raise ValueError("preview candidate regions／selection 必須同時存在")
    if raw_policy.get("policy_id") != _BAYTRACE_CANDIDATE_POLICY_ID:
        raise ValueError("preview candidate region policy 未登錄")
    if raw_policy.get("require_each_region") is not True or raw_policy.get("total_horizontal_count") != 5:
        raise ValueError("preview candidate region 必須要求兩子區各自完成且總數為 5")
    if len(raw_regions) != 2:
        raise ValueError("preview C 區 candidate regions 必須恰有兩個子區")
    regions: list[dict[str, Any]] = []
    allocation_total = 0
    seen_ids: set[str] = set()
    for index, raw_region in enumerate(raw_regions):
        if not isinstance(raw_region, dict):
            raise ValueError(f"preview candidate region[{index}] 必須是 mapping")
        region_id = raw_region.get("region_id")
        allocation = raw_region.get("allocation_count")
        if (
            not isinstance(region_id, str)
            or not region_id.strip()
            or region_id in seen_ids
            or type(allocation) is not int
            or allocation < 1
        ):
            raise ValueError(f"preview candidate region[{index}] 的 id／配額無效")
        if raw_region.get("coordinate_reference") != "EPSG:4326":
            raise ValueError(f"preview candidate region[{index}] 必須是 EPSG:4326")
        try:
            geometry = shape(raw_region["geometry"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"preview candidate region[{index}] geometry 無法解析") from exc
        if (
            not isinstance(geometry, Polygon)
            or geometry.is_empty
            or not geometry.is_valid
            or geometry.area <= 0.0
        ):
            raise ValueError(f"preview candidate region[{index}] geometry 無效")
        regions.append({**raw_region, "geometry": geometry})
        allocation_total += allocation
        seen_ids.add(region_id)
    if allocation_total != 5:
        raise ValueError(f"preview candidate region 配額總數不符：{allocation_total} != 5")
    return regions


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

    每筆 ``numerical_failure`` 必須同時在 ``particles.csv`` 與
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
    """依資料核對結果建立新版顯示名稱，底層 status 值與計數永遠不被改寫。

    ``forcing_start`` 的文字由每顆粒子的 age 與本次回溯設定動態產生，讓圖例同時
    交代資料窗端點與本案例的回溯時長；它只反映輸入事件的顯示語意，不會把 status
    轉成 ``max_age``。
    """

    labels = dict(_BAYTRACE_STATUS_LABELS)
    labels["max_age"] = _baytrace_max_age_label(data["summary"])
    labels["forcing_start"] = _forcing_start_label(data)
    if _sampling_failure_is_verified(data):
        labels["numerical_failure"] = "取樣失敗停止"
    return labels


def _baytrace_endpoint_style(status: str, forcing_age_state: str | None = None) -> tuple[str, str]:
    """回傳水平／垂向終點的 marker 與顏色，保留每一種 status 的視覺差異。

    所有已登錄狀態均從固定表取得形狀；``forcing_start`` 仍固定使用非紅色菱形，
    但依單顆粒子的 age 分類選紫色（容差內完成）、橙色（明顯提前）或保守色（其他
    未核對情況），以明確區分資料窗端點與紅色 ``X`` 的 ``numerical_failure``。未知
    值只提供保守 fallback 給直接呼叫的繪圖 helper，正式 build 仍會在資料核對階段
    拒絕未登錄狀態。
    """

    marker = _STATUS_MARKERS.get(status, "D")
    if status != "forcing_start":
        color = _BAYTRACE_FINAL_COLOR
    elif forcing_age_state == "early":
        color = _BAYTRACE_FORCING_START_EARLY_COLOR
    elif forcing_age_state in {"overrun", "unverified"}:
        color = "#756bb1"
    else:
        color = _BAYTRACE_FORCING_START_COLOR
    return marker, color


def _baytrace_terminal_summary(data: dict[str, Any]) -> str:
    """建立停止圖頂端摘要，區分資料窗端點、一般回溯上限、離域與數值失敗。

    ``forcing_start`` 只有在原始粒子 status 確實如此時才列入「到達本次資料時間窗
    起點」；本函式同時讀取每顆粒子的 age，保留提前與完成設定時長的差異，不改寫
    資料層。當本次 24 小時案例的 20 顆粒子全為容差內的 forcing_start 時，摘要會
    寫出 20 顆到達資料窗起點，而不會寫成「0 顆完成回溯」。此顯示判讀仍只限於已選
    forcing window 的端點，不是科學來源或因果判定。
    """

    summary = data["summary"]
    counts = summary["terminal_counts"]
    forcing_count = int(counts.get("forcing_start", 0))
    max_age_count = int(counts.get("max_age", 0))
    completion_parts: list[str] = []
    if forcing_count:
        completion_parts.append(f"{forcing_count} 顆{_forcing_start_label(data)}")
    if max_age_count:
        completion_parts.append(f"{max_age_count} 顆{_baytrace_max_age_label(summary)}")
    if not completion_parts:
        completion_parts.append("0 顆到達本次資料時間窗起點")
    completion_parts.extend(
        (
            f"{int(counts.get('flow_domain_open_exit', 0))} 顆離域",
            f"{int(counts.get('numerical_failure', 0))} 顆數值失敗",
        )
    )
    return "、".join(completion_parts) + "。"


def _baytrace_layer_label(level: str) -> str:
    """回傳不含內部層代碼的完整中文水層名稱，未知值使用保守通用名稱。"""

    return _BAYTRACE_VERTICAL_LABELS.get(level, "其他初始水層")


def _baytrace_layer_definition(level: str) -> str:
    """回傳水層的設定目標或近床物理定義，避免由圖面反推固定公尺高度。"""

    return _BAYTRACE_VERTICAL_DEFINITIONS.get(level, "來源設定未提供可核對的水層定義")


def _baytrace_depth_particle_note(particle_count: int) -> str:
    """建立垂向子圖的逐粒子實線說明，並明示該子圖實際粒子數。

    垂向圖的彩色實線使用初始水層顏色，但每一條線仍是單一粒子的 ``z_m`` 保存
    軌跡；若只在全圖底部放一條泛稱圖例，容易讓讀者以為存在未繪出的灰色資料線。
    因此說明放在各子圖標題附近，數量直接取該水層的粒子列數，讓其他粒子數的
    preview 也能沿用同一個 renderer。此文字只描述已保存軌跡的顯示方式，不代表
    粒子彼此相同或具有統計獨立性。
    """

    if type(particle_count) is not int or particle_count < 0:
        raise ValueError("垂向子圖粒子數必須是非負整數")
    return f"每條彩色實線代表 1 顆粒子的高度軌跡（共 {particle_count} 顆）"


def _baytrace_depth_bed_label(particle_counts: Sequence[int]) -> str:
    """依四個水層子圖的粒子數建立海床點線圖例文字。

    ``bed_z_m`` 是每顆粒子所在水平位置的海床高度，不是五個水層或一條共用海床
    曲線；同一子圖保留每顆粒子的點線，才能呈現其水平路徑下方地形的差異。當所有
    子圖粒子數相同時，圖例可安全地精確標示「每圖 N 條」；數量不一致時則不把某一
    個子圖的數量誤套到全圖，改由各子圖標題的實際數字說明。這個標籤只改變呈現
    語意，不會篩選、合併或改寫 ``bed_z_m`` 資料。
    """

    counts = tuple(particle_counts)
    if not counts:
        return "各粒子所在位置的海床高度"
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("垂向圖粒子數必須是非負整數")
    if len(set(counts)) == 1:
        return f"各粒子所在位置的海床高度（每圖 {counts[0]} 條）"
    return "各粒子所在位置的海床高度（各圖數量依子圖標示）"


def _baytrace_metadata(summary: dict[str, Any]) -> str:
    """組合四張新版圖共用的研究情境摘要，不顯示 run／scenario 等內部識別碼。

    時間只把既有 arrival 與 horizon 轉成完整日期的 UTC 時間列；不從圖面推算新的
    垂向高度或粒子統計。``M`` 的含義由「同一組粒子」與 README 另行說明，避免
    將成員數誤讀為已完成回溯的數量。
    """

    arrival, _ = _time_window(summary)
    horizon = _horizon_seconds(summary)
    region, _, site, _ = _site_context(summary)
    return (
        f"{region}區／{site}｜{arrival:%Y-%m-%d}｜"
        f"同一組{summary['particle_count']}顆粒子｜"
        f"{len(summary['horizontal_panels'])}個初始位置×{len(summary['vertical_order'])}個初始水層｜"
        f"單一沉降材質｜{_time_window_label(summary)}｜"
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


def _baytrace_endpoint_handles(
    status_labels: dict[str, str],
    *,
    include_boundary: bool = True,
    forcing_start_states: set[str] | None = None,
    forcing_start_summary: dict[str, Any] | None = None,
    statuses: set[str] | None = None,
):
    """建立「標記：端點／停止狀態」圖例，保留八種終止狀態的獨立標記。

    ``forcing_start`` 在容差內完成時使用紫色菱形，明顯提前時使用橙色菱形並明示
    「提前到達」；只有真正的 ``numerical_failure`` 使用紅色 ``X``。其他狀態仍保留
    各自 marker 與中文名稱，不因本次資料恰好只有 forcing_start 就合併成同一類。
    ``forcing_start_states`` 由呼叫端以每顆粒子的 age 建立；未提供時沿用傳入的
    aggregate label，方便單元測試與其他相容呼叫。若要繪製多種 forcing_start age
    狀態，呼叫端也必須傳入 summary，讓每個圖例項目使用同一 horizon 設定。
    ``statuses`` 讓水平／垂向圖例只列本批實際出現的終止狀態，避免零計數項目擠壓
    圖面；停止原因柱狀圖仍固定列出全部八種狀態與零計數。
    """

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
    selected_statuses = (
        _TERMINAL_STATUSES
        if statuses is None
        else tuple(status for status in _TERMINAL_STATUSES if status in statuses)
    )
    for status in selected_statuses:
        if status == "forcing_start" and forcing_start_states:
            forcing_labels = [
                state
                for state in ("completed", "early", "overrun", "unverified")
                if state in forcing_start_states
            ]
        else:
            forcing_labels = []
        if forcing_labels:
            if forcing_start_summary is None:
                raise ValueError("forcing_start 分類圖例缺少 summary horizon")
            for state in forcing_labels:
                marker, color = _baytrace_endpoint_style(status, state)
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker=marker,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        linestyle="None",
                        markersize=7,
                        label=f"最終位置｜{_forcing_start_state_label(forcing_start_summary, state)}",
                    )
                )
            continue
        marker, color = _baytrace_endpoint_style(status)
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
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
    僅作已確認的地理參照。四層線色固定使用藍、橙、紫、青，綠色只表示回溯起點；
    終點依八種 status 使用各自形狀，forcing_start 另依每顆 age 以紫／橙菱形區分
    容差內完成與提前抵達，numerical_failure 才使用紅色 X。這張圖不把終點解讀成
    已知污染來源。
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
    forcing_start_states = set(_forcing_start_age_states(data))
    terminal_statuses = {particle["status"] for particle in data["particles"]}
    horizon_seconds = _horizon_seconds(summary)
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
            marker, endpoint_color = _baytrace_endpoint_style(
                particle["status"],
                _forcing_start_age_state(
                    particle.get("age_seconds"), horizon_seconds
                )
                if particle.get("status") == "forcing_start"
                else None,
            )
            ax.plot(
                float(particle["longitude"]),
                float(particle["latitude"]),
                marker=marker,
                markersize=7.1,
                markerfacecolor=endpoint_color,
                markeredgecolor=endpoint_color,
                markeredgewidth=0.8,
                color=endpoint_color,
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
            f"水平軌跡｜回溯{_horizon_hours_text(summary)}小時內的平面移動路徑",
            fontproperties=font,
            fontsize=18,
            y=0.995,
            va="top",
        )
        fig.text(
            0.5,
            0.962,
            f"從指定位置往前回溯，粒子經過哪些位置？{_position_order_note(summary)}",
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
            handles=_baytrace_endpoint_handles(
                status_labels,
                forcing_start_states=forcing_start_states,
                forcing_start_summary=summary,
                statuses=terminal_statuses,
            ),
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
    """繪製新版各位置局部圖，只平移原 AEQD 公尺座標且各位置獨立設範圍。

    各位置依 summary 的水平面板與垂向層別分組，線段與端點使用和區域總覽相同的
    顏色／標記語意。平移原點只為便於閱讀，不移除、放大、抖動或重新積分任何座標；
    面板可以有不同範圍，因此不能以印刷長度跨面板比較移動量。
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
    forcing_start_states = set(_forcing_start_age_states(data))
    terminal_statuses = {particle["status"] for particle in data["particles"]}
    horizon_seconds = _horizon_seconds(summary)
    boundary = data["projection"].project_geometry(data["open_geometry"])
    font = _display_font()
    endpoint_handles = _baytrace_endpoint_handles(
        status_labels,
        forcing_start_states=forcing_start_states,
        forcing_start_summary=summary,
        statuses=terminal_statuses,
    )
    # 終止狀態依站點的實際 CSV 而變動；例如龜山島同時有離岸、離域、資料窗起點與
    # 回溯上限，端點圖例會比只有兩種狀態的站點高。依圖例項目數增加右下說明列，讓
    # 繪圖保留獨立的「水層圖例／端點圖例／兩行圖說」區域，而不是把長中文壓到下一列。
    # 這只改變印刷版面，不改變任何軌跡、狀態、座標或分母；最後仍以 renderer 的
    # 實際 bounding box 交疊檢查作為發布前的 fail-closed 閘門。
    bottom_grid_ratio = 1.35 + max(0, len(endpoint_handles) - 5) * 0.35
    # 資料面板維持既有 3×2 版面；右下空格再切成三個獨立子格，
    # 讓水層圖例、端點圖例與兩行圖說各自有真實的版面空間，不靠同一座標軸
    # 的相對 y 值互相避讓。這也使最後的 bbox 檢查能對應三個獨立 artist。
    fig = plt.figure(figsize=(12, 16.5), dpi=150)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, bottom_grid_ratio))
    axes = [[fig.add_subplot(grid[row, column]) for column in range(2)] for row in range(3)]
    panel_axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1], axes[2][0]]
    fig.delaxes(axes[2][1])
    legend_grid = grid[2, 1].subgridspec(
        3,
        1,
        # 第一列需容納圖例標題與水層，第二列需容納單欄端點項目；
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
            expected_layer_count = len(levels)
            if len(selected) != expected_layer_count:
                raise ValueError(f"新版局部圖每個位置必須保留 {expected_layer_count} 個初始水層粒子")
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
                marker, endpoint_color = _baytrace_endpoint_style(
                    particle["status"],
                    _forcing_start_age_state(
                        particle.get("age_seconds"), horizon_seconds
                    )
                    if particle.get("status") == "forcing_start"
                    else None,
                )
                ax.plot(
                    float(particle["x_m"]) - x0,
                    float(particle["y_m"]) - y0,
                    marker=marker,
                    markersize=8.2,
                    markerfacecolor=endpoint_color,
                    markeredgecolor=endpoint_color,
                    markeredgewidth=0.8,
                    color=endpoint_color,
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
            handles=endpoint_handles,
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
            "各位置以本身回溯起點為 (0,0)。\n各面板尺度不同，請依公尺刻度比較。",
            transform=caption_ax.transAxes,
            fontproperties=font,
            fontsize=10,
            va="center",
        )

        fig.suptitle(
            f"水平軌跡｜回溯{_horizon_hours_text(summary)}小時內的平面移動路徑",
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
    """繪製各初始水層的高度—回溯時間診斷圖，時間軸跟隨 summary 回溯上限。

    粒子高度 ``z_m``、海面 ``eta_m`` 與海床 ``bed_z_m`` 直接取自既有觀測 CSV；
    空白欄位轉為 NaN 只讓圖線留空，不補零、不從 normalized_config 硬算統一公尺
    高度，也不重新讀 forcing 或重跑取樣。水層名稱使用完整中文，並明確註明海床
    只是參考線；近床水層是最低有效 OCM 水層中心，不是海床面。每條彩色實線是
    一顆粒子的高度軌跡，棕色點線則是該粒子所在水平位置取出的海床高度；兩者都
    保留逐粒子資料，不把海床線合併成範圍帶，也不把點線誤稱為不同水層。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = _prepare_baytrace_output(output_path, "新版垂向診斷圖")
    summary = data["summary"]
    curves = _ordered_baytrace_curves(data)
    levels = list(summary["vertical_order"])
    horizon_minutes, age_ticks = _depth_axis_limits_and_ticks(summary)
    status_labels = _baytrace_status_labels(data)
    forcing_start_states = set(_forcing_start_age_states(data))
    terminal_statuses = {particle["status"] for particle in data["particles"]}
    horizon_seconds = _horizon_seconds(summary)
    font = _display_font()
    selected_by_level = {
        level: [particle for particle in data["particles"] if particle["vertical_id"] == level]
        for level in levels
    }
    particle_counts = [len(selected_by_level[level]) for level in levels]
    fig, axes = plt.subplots(len(levels), 1, figsize=(11, 12), squeeze=False, sharex=True)
    try:
        for index, level in enumerate(levels):
            ax = axes[index, 0]
            selected = selected_by_level[level]
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
                    marker, endpoint_color = _baytrace_endpoint_style(
                        particle["status"],
                        _forcing_start_age_state(
                            particle.get("age_seconds"), horizon_seconds
                        )
                        if particle.get("status") == "forcing_start"
                        else None,
                    )
                    ax.plot(
                        age_minutes[terminal_index],
                        z_values[terminal_index],
                        marker=marker,
                        markersize=5.7,
                        markerfacecolor=endpoint_color,
                        markeredgecolor=endpoint_color,
                        color=endpoint_color,
                        linestyle="None",
                        clip_on=False,
                        zorder=6,
                    )
            ax.set_xlim(0.0, horizon_minutes)
            ax.set_title(
                f"{_baytrace_layer_label(level)}（{_baytrace_layer_definition(level)}）｜粒子數：{len(selected)}\n"
                f"{_baytrace_depth_particle_note(len(selected))}",
                fontproperties=font,
                fontsize=11,
            )
            ax.set_ylabel("高度（公尺；向上為正）", fontproperties=font)
            ax.grid(color="#d9e2e8", linewidth=0.7)
            ax.set_xticks(age_ticks)
        axes[-1, 0].set_xlabel("回溯時間（分鐘；0 為到達時刻）", fontproperties=font)
        legend_handles = [
            Line2D([], [], color="#1b4f72", linestyle="--", linewidth=1.0, label="海面高度"),
            Line2D(
                [],
                [],
                color="#8c510a",
                linestyle=":",
                linewidth=1.0,
                label=_baytrace_depth_bed_label(particle_counts),
            ),
        ]
        legend_handles.extend(
            _baytrace_endpoint_handles(
                status_labels,
                include_boundary=False,
                forcing_start_states=forcing_start_states,
                forcing_start_summary=summary,
                statuses=terminal_statuses,
            )
        )
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
    """讀取 summary 的各水層八狀態計數，保留零值並以 CSV 分母逐層核對。"""

    summary = data["summary"]
    levels = summary.get("vertical_order")
    if not isinstance(levels, list) or not levels:
        raise ValueError("新版停止圖必須有至少一個初始水層")
    raw = summary.get("terminal_counts_by_vertical")
    if not isinstance(raw, dict):
        raise ValueError("新版停止圖缺少 terminal_counts_by_vertical")
    csv_counts = Counter(
        (particle.get("vertical_id"), particle.get("status")) for particle in data["particles"]
    )
    table: dict[str, dict[str, int]] = {}
    for level in levels:
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
        expected_count = sum(count for (vertical_id, _), count in csv_counts.items() if vertical_id == level)
        if sum(row.values()) != expected_count:
            raise ValueError(
                f"新版停止圖水層分母與 CSV 不符：{_baytrace_layer_label(level)}"
            )
        table[level] = row
    return table


def render_baytrace_terminal(data: dict[str, Any], output_path: str | Path) -> Path:
    """繪製各初始水層的八狀態停止原因圖，保留全部粒子與零計數狀態。

    圖面回答「各粒子到達哪一種終止狀態？」；``forcing_start`` 是否可寫成完成設定
    時長，另依每顆 particles CSV 的 ``age_seconds`` 與 summary horizon 以保守容差
    核對。開放邊界離域與已核對的取樣失敗則分別顯示。底層 status、分母及各狀態計數
    均直接取自輸入；這是 preview 的補充診斷，不是新的來源分析。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output = _prepare_baytrace_output(output_path, "新版停止原因圖")
    summary = data["summary"]
    levels = list(summary["vertical_order"])
    table = _baytrace_terminal_table(data)
    status_labels = _baytrace_status_labels(data)
    layer_counts = {level: sum(table[level].values()) for level in levels}
    maximum_layer_count = max(layer_counts.values(), default=0)
    count_label_offset = max(0.08, maximum_layer_count * 0.016)
    axis_right = maximum_layer_count + max(0.55, maximum_layer_count * 0.11)
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
        for position, level in zip(y_positions, levels, strict=True):
            ax.text(
                maximum_layer_count + count_label_offset,
                position,
                f"總數 {layer_counts[level]}",
                va="center",
                fontproperties=font,
                fontsize=10,
                color="#17324d",
            )
        ax.set_yticks(y_positions, [_baytrace_layer_label(level) for level in levels], fontproperties=font)
        ax.invert_yaxis()
        ax.set_xlim(0, axis_right)
        ax.set_xlabel("粒子數（每個初始水層分母依資料）", fontproperties=font)
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
            "停止原因｜各終止狀態的粒子數",
            fontproperties=font,
            fontsize=18,
            y=0.995,
            va="top",
        )
        fig.text(
            0.5,
            0.958,
            _baytrace_terminal_summary(data),
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
        # 研究問題已在 README 閱讀順序說明；圖面上只保留資料驅動的停止摘要
        # 與 metadata 三列，讓 header 不與繪圖軸或彼此重疊。
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
    """產生 legacy 兩圖成果的繁中圖說、來源事實與重建命令。

    此函式只供 ``style="legacy"`` 相容分支使用；BayTrace 四圖使用
    ``_baytrace_readme``，因此兩種成果的文件契約不會互相覆蓋。
    """

    summary = data["summary"]
    center = summary["projection"]["center_lonlat"]
    counts = summary["terminal_counts"]
    particle_count = len(data["particles"])
    observation_count = len(data["observations"])
    panel_count = len(summary["horizontal_panels"])
    level_count = len(summary["vertical_order"])
    horizon_label = _max_age_label(summary)
    horizon_minutes, _ = _depth_axis_limits_and_ticks(summary)
    region, _, site_label, _ = _site_context(summary)
    lines = [
        f"# {region}區{site_label}沉降粒子反向追蹤",
        "",
        f"Run：`{summary['run_id']}`；站點：{region}／{site_label}／{data['domain_record']['flow_domain_id']}。",
        _subtitle(summary) + "。",
        "",
        "- [區域總覽](horizontal_overview.png)：經緯度、陸地及本次試跑登錄外框。",
        f"- [{panel_count} 個局部面板](horizontal_local.png)：沿用原圖水平面板，"
        f"每組 {level_count} 個初始水層。",
        "",
        "## 圖面與計數",
        "",
        f"保留全部 {particle_count} 顆粒子與 {observation_count} 筆模型保存紀錄；"
        "這些不是實測觀測數。",
        "線條依回溯時間連接保存位置，未重新取樣。圓點為回溯起點；終點取自 particles CSV。",
        "初始水層固定為近海床（藍）、中下水層（橙）、中上水層（綠）、上水層（紅）。",
        f"停止計數：{horizon_label} {counts['max_age']}、開放邊界離域 {counts['flow_domain_open_exit']}、"
        f"數值停止 {counts['numerical_failure']}；資料缺口 {counts['data_gap']}、"
        f"沉積 {counts['deposited']}。",
        f"數值停止共 {counts['numerical_failure']} 顆；未刪除或以其他狀態替代。",
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
        f"每個面板 x/y 等比例，但 {panel_count} 個面板的範圍各自設定，不能直接比較印刷線段長度。",
        "外框與軌跡接受相同座標轉換／平移；局部僅顯示外框與視窗的實際交集。",
        "土地與外框在底層，保存軌跡及起訖標記在上層；不以陸地填色遮蔽或刪除粒子。",
        "",
        "## 來源與限制",
        "",
        "海岸使用已確認的 `taiwan_exact_coastline.geojson`：CRS84（經度、緯度），"
        "1905 個有效 Polygon／MultiPolygon；原始供應者、年代與授權未附。",
        "本次用於內部 PI 地理參照，不宣稱為官方精確岸線，亦不套用其他資料的授權。",
        f"{region} 外框來源方法：`{data['domain_manifest']['provenance']['method_id']}`。",
        "這是 accepted manifest 已登錄的加密經緯度矩形幾何，不是完整濕網格或有效海水邊界；"
        "外框內不保證全部都是有效水域。",
        f"{region} 的 flow_domain 開放邊界與 polygon 外環相等，因此只畫一條登錄開放邊界線。",
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
    圖面意義先於程式細節：先說明三類問題，再列出本次粒子、水層與八狀態、
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
    particle_count = len(data["particles"])
    panel_count = len(summary["horizontal_panels"])
    level_count = len(summary["vertical_order"])
    horizon_label = status_labels["max_age"]
    forcing_start_states = Counter(_forcing_start_age_states(data))
    horizon_minutes, _ = _depth_axis_limits_and_ticks(summary)
    layer_totals = {level: sum(terminal_table[level].values()) for level in summary["vertical_order"]}
    depth_particle_counts = [
        sum(1 for particle in data["particles"] if particle.get("vertical_id") == level)
        for level in summary["vertical_order"]
    ]
    depth_bed_label = _baytrace_depth_bed_label(depth_particle_counts)
    layer_total_text = "、".join(
        f"{_baytrace_layer_label(level)} {layer_totals[level]} 顆" for level in summary["vertical_order"]
    )
    region, _, site_label, _ = _site_context(summary)
    candidate_regions = _baytrace_candidate_regions(summary)
    candidate_note_lines: list[str] = []
    if candidate_regions:
        candidate_note_lines = [
            "",
            "本次 C 區研究候選來自 receptor manifest 的明示 EPSG:4326 polygon；候選",
            "polygon 只作科學 provenance，視覺圖面不呈現候選區域。兩個子區各自通過",
            "受體選取所需的支援閘門後，固定配置如下：",
            "",
            "| 候選子區 | 水平位置配額 |",
            "|---|---:|",
            *[
                f"| {region['name_zh']}（`{region['region_id']}`） | {region['allocation_count']} |"
                for region in candidate_regions
            ],
            "",
            "候選不足時整體 fail closed，不跨區補點；polygon、影像 hash 與重建資訊另保存在",
            "`summary.json`、`manifest.json` 的 candidate region 欄位。",
        ]
    lines = [
        f"# {region}區{site_label}新版回溯成果圖",
        "",
        "本目錄是由已完成 preview 獨立重繪的新版圖面；不讀 forcing、不重新取樣、不重新積分，"
        "也不改變原始 CSV、summary 或舊成果。輸出只包含兩張水平圖、一張垂向診斷圖及一張停止原因圖，"
        "沒有新增直方圖或其他分析。",
        f"圖上共通情境：{_baytrace_metadata(summary)}。",
        *candidate_note_lines,
        "",
        "## 給 PI 的閱讀方式",
        "",
        "共同研究問題是：從指定到達位置往前回溯，粒子在平面上經過哪裡、在水中的高度如何改變，"
        f"以及各粒子到達哪一種終止狀態。這些圖把一組 {particle_count} 顆粒子（"
        f"{panel_count} 個初始位置 × {level_count} 個初始水層）放在同一個單一沉降材質情境中；"
        "它們是工程先導的條件式路徑診斷，"
        "不是已知污染來源的證明。",
        "",
        "建議閱讀順序：",
        "",
        "1. [水平區域總覽](horizontal_overview.png)：先回答「從指定位置往前回溯，粒子經過哪些位置？」；"
        f"{_position_order_note(summary)}經緯度與已確認海岸讓整體位置關係可直接判讀。",
        f"2. [水平局部圖](horizontal_local.png)：再看位置1–{panel_count} 的原 AEQD 公尺座標，"
        "確認各位置的路徑細節；"
        f"{panel_count} 個面板各自等比例，但範圍不同。",
        "3. [垂向軌跡](depth_age.png)：回答「回溯過程中，粒子在水中的高度如何變化？」；"
        "z、海面 eta、海床 bed 都只讀保存觀測。",
        "4. [停止原因](terminal_counts.png)：最後回答「各粒子到達哪一種終止狀態？」；"
        f"每個初始水層的粒子分母依逐格資料核對（{layer_total_text}），八種狀態都保留，包含零計數。",
        "",
        f"這樣的順序先交代平面路徑，再補充垂向變化，最後用停止原因說明：{_baytrace_terminal_summary(data)}"
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
        "| 最終 | 最終位置｜各終止狀態（forcing_start 紫／橙色 D；numerical_failure 紅色 X） | "
        "代表停止狀態；不是已知污染來源 |",
        "| 經度／緯度 | 區域總覽使用經度、緯度 | 局部圖改用原 AEQD 平移公尺座標，未移動或放大資料 |",
        f"| 粒子數 | 圖上明示同一組 {particle_count} 顆粒子、{panel_count} 個初始位置 × "
        f"{level_count} 個初始水層 | "
        f"是否追滿{_horizon_hours_text(summary)}小時依每顆 age_seconds 與 horizon 核對；M=1 且含隨機擴散 |",
        "| 垂向軌跡 | 專案補充診斷圖：逐粒子高度軌跡、海面、各粒子所在位置的海床高度 | "
        "每條彩色實線代表一顆粒子；棕色點線依各子圖實際粒子數保留，z／eta／bed 均取既有 CSV |",
        "| 停止原因 | 專案補充診斷圖：八種終止狀態的粒子數 | 呈現參照沒有直接對應，八狀態及零計數均保留 |",
        "",
        "新版端點圖例依資料核對結果區分八種狀態；forcing_start 依每顆 particles.csv 的"
        f" age_seconds 與 horizon_seconds 判定，本批標籤為「{status_labels['forcing_start']}」。"
        f"固定容差為 {_FORCING_START_AGE_TOLERANCE_SECONDS:g} 秒，不能以 300 秒 output interval 取代；"
        "圖例仍分開列出「最終位置｜"
        f"{status_labels['max_age']}」、「最終位置｜{status_labels['flow_domain_open_exit']}」與"
        f"「最終位置｜{status_labels['numerical_failure']}」；只有 numerical_failure 使用紅色 X，"
        "所有終點均不是已知污染來源。",
        f"forcing_start age 分類計數：{dict(forcing_start_states)}；資料層 status 仍完整"
        "保留為 forcing_start。",
        "新版與 default `legacy` 分開：legacy 仍只產生原有兩張水平 PNG 及其舊契約；"
        f"不要把 legacy 的 H1–H{panel_count}、線色或任意終點語意與本新版混用。"
        f"新版圖面統一使用「位置1–{panel_count}」、"
        "完整中文水層名、藍／橙／紫／青線色、綠色回溯起點及依狀態區分的終點標記。",
        "",
        "## 水層定義與停止統計",
        "",
        "上／中上／中下水層的百分比是設定目標，不把到達時刻資料硬算成固定公尺高度；"
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
        f"停止圖固定保留八種狀態；各初始水層分母依資料核對，本次為：{layer_total_text}。",
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
    # 取代或重排既有水平面板名稱，讓 PI 能逐列回查原 preview 契約。
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
        "下表直接取自 `summary.horizontal_panels`，讓局部圖的位置名稱可回對原 preview 標籤與經緯度。",
        "",
        "| 新版位置 | 原 preview 標籤 | 經度（°E） | 緯度（°N） |",
        "|---|---|---:|---:|",
        *position_rows,
        "",
    ]
    lines += [
        "",
        f"本次圖面共保留 {len(data['particles'])} 顆粒子與 {len(data['observations'])} 筆模型保存紀錄；"
        f"停止摘要：{_baytrace_terminal_summary(data)}",
        "垂向圖每條彩色實線代表 1 顆粒子的高度軌跡；棕色點線代表該粒子所在位置的海床高度，"
        f"不是五個水層；海床圖例項目為「{depth_bed_label}」。"
        "圖例只有在各子圖粒子數相同時才標示「每圖 N 條」，數量不同時以各子圖標題為準。",
        f"垂向資料缺值原樣留空：eta_m 空白 {missing_eta} 筆、bed_z_m 空白 {missing_bed} 筆，"
        f"不補成零；高度軸標示為「高度（公尺；向上為正）」，時間軸涵蓋 0–{horizon_minutes:g} 分鐘，"
        "刻度由回溯上限等分。",
        "",
        "## 取樣失敗診斷界線",
        "",
    ]
    if counts.get("max_age", 0):
        lines.append(f"另有 {horizon_label} {counts['max_age']} 顆；此狀態與 forcing_start 分開保存。")
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
        f"緯度 {center[1]}。各局部面板各自等比例且範圍不同；不移除、放大或抖動任何粒子座標。",
        "各位置以本身回溯起點為 (0,0)；水平總覽的位置編號是受體／指定位置，不是時間順序。",
        "forcing_start 只有在每顆 age_seconds 與 horizon_seconds 差值不超過固定容差時，"
        "才可寫成完成設定時長；提前者標為「提前到達驅動資料起點」，不能泛化成科學來源判定。",
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
        f"{_verified_server_cache_prefix()}\\",
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
    storage_gate_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """在全新目錄建立 legacy 或新版四圖，回傳完成的 JSON 內容。

    ``legacy``（預設）限定用於既有舊成果相容性，保留兩張水平圖、原水平標籤及舊
    manifest 契約；``baytrace`` 只在全新目錄寫入水平區域／局部、垂向高度、停止原因四張 PNG，並
    保存新版 style/version 與來源前後 SHA。兩個分支都先拒絕既有目錄或失效連結，
    再讀取並核對既有輸入；核對失敗均保留新目錄供診斷，不清理、不覆寫輸入或舊成果。
    新版僅使用 preview 的 CSV／summary，不讀 forcing、run 或重新積分。讀取 preview 前
    先驗證 completion marker；若 caller 明示 storage gate evidence，缺 marker 或 gate
    binding 不符即拒絕，避免將逐檔發布中的 final 誤當完整圖面來源。
    """

    if style not in {"legacy", "baytrace"}:
        raise ValueError("style 必須是 legacy 或 baytrace")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"輸出目錄已存在，拒絕覆寫：{output}")
    preview = Path(preview_dir)
    preview_marker = validate_published_preview(preview, storage_gate_evidence=storage_gate_evidence)
    marker_publish = storage_gate_evidence is not None
    gate = None
    provenance = None
    if marker_publish:
        if not isinstance(preview_marker, dict):
            raise ValueError("NFS 圖面發布需要已完成的 preview marker")
        gate = _verified_storage_gate_evidence(storage_gate_evidence)
        if (
            gate["sha256"] != preview_marker.get("storage_gate_snapshot_sha256")
            or gate["source_token_hash"] != preview_marker.get("storage_root_source_token_hash")
        ):
            raise ValueError("preview marker 的 storage gate binding 不符")
        provenance = {
            key: preview_marker[key]
            for key in ("base_commit", "base_tree", "dirty_files", "diff_sha256")
        }
    paths = {
        name: preview / name
        for name in ("manifest.json", "summary.json", "particles.csv", "observations.csv")
    }
    paths.update(domain=Path(domain_path), open_boundary=Path(open_path), coastline=Path(coastline_path))
    before = _source_snapshot(paths)
    data = load_plot_inputs(
        preview,
        domain_path,
        open_path,
        coastline_path,
        storage_gate_evidence=storage_gate_evidence,
    )
    summary = data["summary"]
    counts = dict(Counter(particle["status"] for particle in data["particles"]))
    if not isinstance(summary.get("run_id"), str) or not summary["run_id"]:
        raise ValueError("preview run_id 必須是非空字串")
    unknown_statuses = sorted(set(counts) - set(_TERMINAL_STATUSES))
    if unknown_statuses:
        raise ValueError(f"CSV 含未登錄停止狀態：{', '.join(unknown_statuses)}")
    summary_counts = summary.get("terminal_counts")
    if not isinstance(summary_counts, dict):
        raise ValueError("preview 缺少 terminal_counts")
    for status in _TERMINAL_STATUSES:
        expected_count = _integer_value(summary_counts.get(status))
        if expected_count is None or expected_count < 0 or expected_count != counts.get(status, 0):
            raise ValueError(f"CSV 停止計數與 summary 不符：{status}")
    if sum(counts.values()) != len(data["particles"]):
        raise ValueError("CSV 停止計數未涵蓋全部粒子")

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
    if storage_gate_evidence is not None:
        args.extend(["--storage-gate-evidence", str(storage_gate_evidence)])
    # manifest 中的 invocation/rebuild 只保存已驗證根目錄的環境變數名稱，讓
    # 讀取成果的人可以在同一 SERVER 儲存政策下重建，而不會誤用產生者的實際路徑。
    prefix = _verified_server_cache_prefix()
    invocation = prefix + shlex.join(args)
    output_argument_index = args.index("--output-dir") + 1
    args[output_argument_index] = str(output.with_name(output.name + "-rebuild"))
    rebuild = prefix + shlex.join(args)
    render_output = output
    parent_identity = None
    if marker_publish:
        parent = _safe_path(output).parent
        parent_stat = parent.stat()
        if not parent_stat or not parent.is_dir():
            raise ValueError("圖面輸出 parent 必須是既有普通目錄")
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        render_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=parent))
    else:
        output.mkdir(parents=True, exist_ok=False)
    if style == "legacy":
        # 預設分支完整保留上一輪兩圖與 manifest 欄位，讓舊成果仍可由同一命令重建。
        render_overview(data, render_output / "horizontal_overview.png")
        render_local(data, render_output / "horizontal_local.png")
        with (render_output / "README.md").open("x", encoding="utf-8") as stream:
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
                "analysis_region_id": _site_context(summary)[0],
                "study_site_id": _site_context(summary)[1],
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
                name: {
                    "size_bytes": (render_output / name).stat().st_size,
                    "sha256": sha256_file(render_output / name),
                }
                for name in ("horizontal_overview.png", "horizontal_local.png", "README.md")
            },
        }
        if marker_publish:
            manifest["publish_protocol"] = "nfs_completion_marker_v1"
            manifest["completion_marker"] = ".complete"
            manifest["release_provenance"] = {
                **provenance,
                "storage_gate_snapshot_sha256": gate["sha256"],
                "storage_root_source_token_hash": gate["source_token_hash"],
            }
        with (render_output / "manifest.json").open("xb") as stream:
            stream.write(_canonical_json_bytes(manifest))
        if marker_publish:
            _validate_staged_preview(render_output, manifest)
            marker = {
                "schema_version": "1.0.0",
                "artifact_kind": manifest["artifact_kind"],
                "publish_protocol": "nfs_completion_marker_v1",
                "manifest_sha256": sha256_file(render_output / "manifest.json"),
                **provenance,
                "storage_gate_snapshot_sha256": gate["sha256"],
                "storage_root_source_token_hash": gate["source_token_hash"],
            }
            gate_after = _verified_storage_gate_evidence(storage_gate_evidence)
            if gate_after != gate:
                raise ValueError("storage gate 在圖面發布前變動")
            lock_descriptor = _acquire_nfs_publish_lock(parent, output)
            try:
                if before != _source_snapshot(paths):
                    raise ValueError("來源前後 bytes/SHA 不一致；拒絕發布圖面")
                _publish_nfs_marker_preview(
                    render_output,
                    output,
                    parent_identity=parent_identity,
                    marker=marker,
                )
                validate_published_artifact(
                    output,
                    expected_artifact_kind=manifest["artifact_kind"],
                    storage_gate_evidence=storage_gate_evidence,
                )
            finally:
                _release_nfs_publish_lock(lock_descriptor)
        return manifest

    # 新版只增加四張圖與新版說明，不回讀 forcing 或改動既有 preview 資料。
    render_baytrace_overview(data, render_output / "horizontal_overview.png")
    render_baytrace_local(data, render_output / "horizontal_local.png")
    render_baytrace_depth(data, render_output / "depth_age.png")
    render_baytrace_terminal(data, render_output / "terminal_counts.png")
    after = _source_snapshot(paths)
    if before != after:
        raise ValueError("來源前後 bytes/SHA 不一致；保留新目錄供診斷，不建立完成清單")
    status_labels = _baytrace_status_labels(data)
    with (render_output / "README.md").open("x", encoding="utf-8") as stream:
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
    site_region, site_id, _, _ = _site_context(summary)
    # C 區候選子區只以已驗證的 summary provenance 寫入 manifest；繪圖函式不把
    # polygon 當 forcing mask，也不在此階段重新選點。沒有候選欄位的其他站點維持
    # 原有 manifest 形狀，避免把空值誤解成「候選已驗證」。
    candidate_regions = _baytrace_candidate_regions(summary)
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
        "forcing_start_age_tolerance_seconds": _FORCING_START_AGE_TOLERANCE_SECONDS,
        "forcing_start_age_state_counts": dict(Counter(_forcing_start_age_states(data))),
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
            "analysis_region_id": site_region,
            "study_site_id": site_id,
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
            name: {
                "size_bytes": (render_output / name).stat().st_size,
                "sha256": sha256_file(render_output / name),
            }
            for name in output_names
        },
    }
    if candidate_regions:
        for key in (
            "receptor_candidate_selection",
            "receptor_candidate_regions",
            "receptor_candidate_regions_provenance",
        ):
            if key in summary:
                manifest[key] = summary[key]
    if marker_publish:
        manifest["publish_protocol"] = "nfs_completion_marker_v1"
        manifest["completion_marker"] = ".complete"
        manifest["release_provenance"] = {
            **provenance,
            "storage_gate_snapshot_sha256": gate["sha256"],
            "storage_root_source_token_hash": gate["source_token_hash"],
        }
    with (render_output / "manifest.json").open("xb") as stream:
        stream.write(_canonical_json_bytes(manifest))
    if marker_publish:
        _validate_staged_preview(render_output, manifest)
        marker = {
            "schema_version": "1.0.0",
            "artifact_kind": manifest["artifact_kind"],
            "publish_protocol": "nfs_completion_marker_v1",
            "manifest_sha256": sha256_file(render_output / "manifest.json"),
            **provenance,
            "storage_gate_snapshot_sha256": gate["sha256"],
            "storage_root_source_token_hash": gate["source_token_hash"],
        }
        gate_after = _verified_storage_gate_evidence(storage_gate_evidence)
        if gate_after != gate:
            raise ValueError("storage gate 在圖面發布前變動")
        lock_descriptor = _acquire_nfs_publish_lock(parent, output)
        try:
            if before != _source_snapshot(paths):
                raise ValueError("來源前後 bytes/SHA 不一致；拒絕發布圖面")
            _publish_nfs_marker_preview(
                render_output,
                output,
                parent_identity=parent_identity,
                marker=marker,
            )
            validate_published_artifact(
                output,
                expected_artifact_kind=manifest["artifact_kind"],
                storage_gate_evidence=storage_gate_evidence,
            )
        finally:
            _release_nfs_publish_lock(lock_descriptor)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """解析明示本機路徑與圖面 style；成功回傳 0，拒絕回傳 2 並保留診斷目錄。"""

    parser = argparse.ArgumentParser(description="由已完成 A／B／C／D 四區五站 preview 獨立重繪水平與診斷圖")
    parser.add_argument("--preview-dir", required=True, type=Path)
    parser.add_argument("--domain", required=True, type=Path)
    parser.add_argument("--open-boundary", required=True, type=Path)
    parser.add_argument("--coastline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--style", choices=("legacy", "baytrace"), default="legacy")
    parser.add_argument(
        "--storage-gate-evidence",
        type=Path,
        help="若 preview 使用 NFS marker，重新驗證同一份 storage gate evidence",
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_coastline_preview(
            args.preview_dir,
            args.domain,
            args.open_boundary,
            args.coastline,
            args.output_dir,
            style=args.style,
            storage_gate_evidence=args.storage_gate_evidence,
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
