"""WP-51 C：龜山島西側站錨點由陸地移到海上的設定契約與選點規則測試。

真實網格上的推導由 ``scripts/derive_guishan_anchor.py`` 在伺服器執行；本檔以兩種方式
守住結果：（一）設定檔、``config.py`` 常數與推導紀錄互相一致，且錨點落在走廊內、舊錨點
（落在島上）已被取代；（二）以合成網格驗證「走廊內最近的 persistent-wet 面中心」規則本身，
包含乾點、邊界距離與無候選時的行為。
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from shapely.geometry import Point, Polygon, box

import lagrangian_backtracking.config as config_module
from lagrangian_backtracking.config import (
    FORMAL_STUDY_SITE_ANCHORS_LON_LAT,
    GUISHAN_ANCHOR_POLICY_ID,
    GUISHAN_SOFT_PRIORITY_POLYGON_LON_LAT,
    ProjectConfig,
    load_config,
)
from lagrangian_backtracking.geometry import DomainProjection, densified_bbox_polygon
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.receptors import (
    prepare_horizontal_receptor_candidates,
    rank_anchor_snap_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "lagrangian_backtracking.formal_h30_h60_h90_m10.yaml"
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"

# 搬移前的原錨點：落在龜山島陸地內，只作為推導起點與 provenance 保留，不得再是正式錨點。
ORIGINAL_ANCHOR = (121.951606, 24.843127)
# A 區 flow domain 外框（與設定檔 domains[A].bbox_lon_lat 相同）與 AEQD 投影中心。
A_BBOX_LON_LAT = (121.306315, 122.793685, 24.600844, 25.499156)
A_CENTER_LON_LAT = (122.05, 25.05)
# 伺服器真實 OCM 網格上量得的三角形邊長中位數（m）；安全距離以其兩倍（設定的
# minimum_flow_domain_margin_local_grid_scales）為下限。
MEDIAN_TRIANGLE_EDGE_M = 30.155216835865733


def _guishan(path: Path):
    config = load_config(path)
    return next(site for site in config.study_sites if site.study_site_id == "guishan"), config


@pytest.mark.parametrize("config_path", [FORMAL_CONFIG, EXAMPLE_CONFIG])
def test_guishan_anchor_is_registered_in_both_configs(config_path: Path) -> None:
    """正式範本與範例設定都使用同一個海上錨點與推導規則識別碼。"""

    site, _ = _guishan(config_path)
    assert tuple(site.anchor_lonlat) == FORMAL_STUDY_SITE_ANCHORS_LON_LAT["guishan"]
    assert tuple(site.anchor_lonlat) != ORIGINAL_ANCHOR
    extra = site.model_extra or {}
    assert extra["anchor_policy"] == GUISHAN_ANCHOR_POLICY_ID
    provenance = extra["anchor_snap_provenance"]
    assert tuple(provenance["original_anchor_lonlat"]) == ORIGINAL_ANCHOR
    assert provenance["original_anchor_status"] == "on_land_guishan_island_per_reviewed_coastline"
    assert provenance["derivation_script"] == "scripts/derive_guishan_anchor.py"
    # 寫死座標是推導出的面中心四捨五入到小數 6 位：差距必須小於約 0.1 m。
    full_lon, full_lat = provenance["snapped_face_center_lonlat_full_precision"]
    assert abs(site.anchor_lonlat[0] - full_lon) <= 5.0e-7
    assert abs(site.anchor_lonlat[1] - full_lat) <= 5.0e-7
    assert provenance["boundary_margin_m"] == pytest.approx(
        float(extra["minimum_flow_domain_margin_local_grid_scales"]) * MEDIAN_TRIANGLE_EDGE_M,
        abs=0.1,
    )


def test_guishan_anchor_is_inside_priority_corridor_and_core_pair_is_intact() -> None:
    """錨點在走廊內（含邊界以內），且受體核心半徑／local 半徑沒有被改動。"""

    site, config = _guishan(FORMAL_CONFIG)
    corridor = Polygon(GUISHAN_SOFT_PRIORITY_POLYGON_LON_LAT)
    assert corridor.covers(Point(*site.anchor_lonlat))
    assert site.receptor_core_radius_m == 12_500
    assert site.local_domain_baseline_radius_m == 20_000
    assert (site.model_extra or {})["minimum_flow_domain_margin_local_grid_scales"] == 2
    # 貢寮錨點與幾何參數完全不受影響（兩站共用 A 區 forcing，但幾何獨立）。
    gongliao = next(item for item in config.study_sites if item.study_site_id == "gongliao")
    assert tuple(gongliao.anchor_lonlat) == (121.9223889, 25.0964444)
    assert gongliao.receptor_core_radius_m == 12_500
    assert gongliao.local_domain_baseline_radius_m == 20_000
    assert "anchor_policy" not in (gongliao.model_extra or {})


def test_guishan_core_and_local_domain_keep_safe_distance_from_flow_domain() -> None:
    """以新錨點為圓心的 12.5 km 核心與 20 km local 圓完整落在 A 區 flow domain 內，
    核心外緣到 flow domain 邊界的距離遠大於兩倍局部網格尺度。"""

    site, _ = _guishan(FORMAL_CONFIG)
    projection = DomainProjection(*A_CENTER_LON_LAT)
    flow_metric = projection.project_geometry(densified_bbox_polygon(A_BBOX_LON_LAT))
    anchor = Point(*(float(v) for v in projection.project(*site.anchor_lonlat)))
    core = anchor.buffer(float(site.receptor_core_radius_m), quad_segs=64)
    local = anchor.buffer(float(site.local_domain_baseline_radius_m), quad_segs=64)
    assert flow_metric.covers(core)
    assert flow_metric.covers(local)  # input-build 的 v3 policy 也要求 local 圓完整在 flow bbox 內
    required_margin_m = 2.0 * MEDIAN_TRIANGLE_EDGE_M
    core_gap_m = core.exterior.distance(flow_metric.exterior)
    local_gap_m = local.exterior.distance(flow_metric.exterior)
    assert core_gap_m > required_margin_m
    assert local_gap_m > required_margin_m
    # 具體數量級：最近的是南緣，核心外緣離邊界仍有 14 km 以上、local 圓外緣 7 km 以上。
    assert core_gap_m > 14_000.0
    assert local_gap_m > 7_000.0


def test_guishan_anchor_moved_from_land_by_about_one_point_three_km() -> None:
    """新舊錨點的距離符合推導紀錄（約 1,288 m），確認沒有誤用其他候選面。"""

    site, _ = _guishan(FORMAL_CONFIG)
    projection = DomainProjection(*A_CENTER_LON_LAT)
    old = np.asarray(projection.project(*ORIGINAL_ANCHOR), dtype=float)
    new = np.asarray(projection.project(*site.anchor_lonlat), dtype=float)
    recorded = (site.model_extra or {})["anchor_snap_provenance"]["distance_from_original_anchor_m"]
    assert float(np.hypot(*(new - old))) == pytest.approx(recorded, abs=1.0)


def _payload() -> dict:
    return yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))


def _site(payload: dict) -> dict:
    return next(site for site in payload["study_sites"] if site["study_site_id"] == "guishan")


def test_current_design_rejects_old_land_anchor() -> None:
    """current 設計不接受舊的陸地錨點（FORMAL_STUDY_SITE_ANCHORS 為 exact 契約）。"""

    payload = _payload()
    _site(payload)["anchor_lonlat"] = list(ORIGINAL_ANCHOR)
    with pytest.raises(ValueError, match="anchor"):
        ProjectConfig.model_validate(payload)


def test_current_design_requires_anchor_policy_and_corridor_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺 anchor_policy、policy 不符，或錨點被移出走廊都應在 schema 層 fail closed。"""

    payload = _payload()
    del _site(payload)["anchor_policy"]
    with pytest.raises(ValueError, match="anchor_policy"):
        ProjectConfig.model_validate(payload)

    payload = _payload()
    _site(payload)["anchor_policy"] = "flow_domain_center_snap_to_persistent_wet_mesh"
    with pytest.raises(ValueError, match="anchor_policy"):
        ProjectConfig.model_validate(payload)

    # 走廊外的點即使通過 anchor exact 檢核（此處同步改寫常數表以隔離這一項），仍須被擋。
    outside = (121.99, 24.845304)
    payload = _payload()
    _site(payload)["anchor_lonlat"] = list(outside)
    monkeypatch.setattr(
        config_module,
        "FORMAL_STUDY_SITE_ANCHORS_LON_LAT",
        {**config_module.FORMAL_STUDY_SITE_ANCHORS_LON_LAT, "guishan": outside},
    )
    with pytest.raises(ValueError, match="走廊"):
        ProjectConfig.model_validate(payload)


def test_formal_template_guishan_block_matches_example() -> None:
    """正式範本與範例的龜山島錨點相關欄位逐項相同，避免兩份設定漂移。"""

    formal = yaml.safe_load(FORMAL_CONFIG.read_text(encoding="utf-8"))
    example = _payload()
    keys = (
        "anchor_lonlat",
        "anchor_policy",
        "anchor_snap_provenance",
        "receptor_core_radius_m",
        "local_domain_baseline_radius_m",
        "receptor_priority_polygon",
        "receptor_priority_selection",
    )
    formal_site = _site(formal)
    example_site = _site(example)
    for key in keys:
        assert formal_site[key] == example_site[key], key


# ── 合成網格：走廊內最近的 persistent-wet 面中心規則 ───────────────────────────


def _row_mesh(face_x: list[float]) -> NativeMesh:
    """沿 x 軸排列、互不重疊的小三角形；面中心 x 約為 ``face_x + 5/3``，y 約 5/3 m。"""

    coordinates: list[list[float]] = []
    faces: list[list[int]] = []
    for x in face_x:
        base = len(coordinates)
        coordinates.extend([[x, 0.0], [x + 5.0, 0.0], [x, 5.0]])
        faces.append([base, base + 1, base + 2, -1])
    xy = np.asarray(coordinates)
    count = len(face_x)
    return NativeMesh(
        node_lon=120.0 + xy[:, 0] * 1e-5,
        node_lat=24.0 + xy[:, 1] * 1e-5,
        node_xy=xy,
        source_depth_m=np.full(xy.shape[0], 20.0),
        source_node_bottom_index=np.zeros(xy.shape[0], dtype=np.int64),
        face_nodes_local=np.asarray(faces),
        face_node_count=np.full(count, 3),
        source_face_global_index=np.arange(1000, 1000 + count),
        bin_size_m=10.0,
    )


def test_rank_anchor_snap_candidates_picks_nearest_persistent_wet_face_in_polygon() -> None:
    """排序由近到遠；乾點、走廊外與太靠邊界的面都不會成為候選。"""

    # 面中心 x ≈ 21.7, 41.7, 61.7, 81.7, 101.7, 121.7（y≈1.7），全域面編號 1000..1005；
    # 原錨點在 x=0（模擬陸地側）。
    mesh = _row_mesh([20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    wetdry = np.zeros((50, 6))
    wetdry[17, 2] = 1.0  # 1002（x≈61.7）在第 17 個到達時刻乾掉 -> persistent-wet 判定剔除
    # 走廊只涵蓋 x∈[30, 110]：1000（x≈21.7）與 1005（x≈121.7）在走廊外；margin=10 m 使
    # 1004（x≈101.7，距 x=110 邊界 8.3 m）被剔除，1001（距 x=30 邊界 11.7 m）保留。
    region = box(30.0, -100.0, 110.0, 100.0)
    ranked = rank_anchor_snap_candidates(
        study_site_id="guishan",
        mesh=mesh,
        region_polygon_metric=region,
        original_anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
        boundary_margin_m=10.0,
    )
    assert [item.source_face_global_index for item in ranked] == [1001, 1003]
    distances = [item.distance_to_original_anchor_m for item in ranked]
    assert distances == sorted(distances)
    assert ranked[0].distance_to_original_anchor_m == pytest.approx(
        math.hypot(41.0 + 2.0 / 3.0, 5.0 / 3.0), abs=0.01
    )
    # 候選的經緯度就是節點經緯度平均，與正式受體候選池一致（錨點是候選池成員）。
    pool = prepare_horizontal_receptor_candidates(
        study_site_id="guishan",
        mesh=mesh,
        candidate_polygon_metric=region,
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
        boundary_margin_m=10.0,
    )
    assert sorted(item.source_face_local_index for item in ranked) == sorted(
        int(i) for i in pool.candidate_face_local_indices
    )
    for item in ranked:
        position = list(pool.candidate_face_local_indices).index(item.source_face_local_index)
        assert item.lon == pytest.approx(float(pool.candidate_lon[position]))
        assert item.lat == pytest.approx(float(pool.candidate_lat[position]))


def test_rank_anchor_snap_candidates_tie_break_and_empty_result() -> None:
    """等距時以經度、緯度、全域面編號決定順序；沒有候選時回傳空串列而不補值。"""

    mesh = _row_mesh([-40.0, 40.0])  # 面中心 x ≈ -38.3 與 41.7：到 x=5/3 錨點等距
    ranked = rank_anchor_snap_candidates(
        study_site_id="guishan",
        mesh=mesh,
        region_polygon_metric=box(-100.0, -10.0, 100.0, 10.0),
        original_anchor_xy=(5.0 / 3.0, 5.0 / 3.0),
        wetdry_at_arrivals=np.zeros((50, 2)),
    )
    assert [item.source_face_global_index for item in ranked] == [1000, 1001]
    assert ranked[0].distance_to_original_anchor_m == pytest.approx(
        ranked[1].distance_to_original_anchor_m, abs=1e-6
    )
    assert rank_anchor_snap_candidates(
        study_site_id="guishan",
        mesh=mesh,
        region_polygon_metric=box(500.0, 500.0, 600.0, 600.0),
        original_anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=np.zeros((50, 2)),
    ) == []


def test_rank_anchor_snap_candidates_all_dry_gives_no_candidates() -> None:
    """全部在某個到達時刻變乾的面都不得被選為錨點（persistent-wet 需 50 個時刻皆濕）。"""

    mesh = _row_mesh([20.0, 40.0])
    wetdry = np.zeros((50, 2))
    wetdry[3, :] = 1.0
    assert (
        rank_anchor_snap_candidates(
            study_site_id="guishan",
            mesh=mesh,
            region_polygon_metric=box(0.0, -10.0, 100.0, 10.0),
            original_anchor_xy=(0.0, 0.0),
            wetdry_at_arrivals=wetdry,
        )
        == []
    )


def test_guishan_anchor_provenance_is_json_serializable() -> None:
    """設定的錨點推導紀錄可被 JSON 序列化（input-build 會原樣寫入受體 provenance）。"""

    site, _ = _guishan(FORMAL_CONFIG)
    provenance = (site.model_extra or {})["anchor_snap_provenance"]
    encoded = json.dumps(copy.deepcopy(provenance), sort_keys=True)
    assert json.loads(encoded) == provenance
