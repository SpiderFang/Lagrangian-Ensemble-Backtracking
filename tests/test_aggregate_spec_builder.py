"""測試由 BoundaryGeometry 建立彙整規格的公開 builder API。

本檔只測試 :func:`write_aggregate_spec_from_boundaries` 的公開輸入與已發布
JSON；不呼叫 production module 的私有 helper，也不以人工重建內部資料結構
取代 loader。測試中的座標是公尺制的最小幾何 fixture，僅用來驗證網格向外
對齊、邊界段識別碼、投影中心與原子發布契約，不代表正式研究區域。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import LineString, Polygon, box

import lagrangian_backtracking.aggregate_spec as aggregate_spec_module
from lagrangian_backtracking.aggregate_spec import (
    load_aggregate_spec,
    write_aggregate_spec_from_boundaries,
)
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.manifests import BoundaryGeometryBundle

_MISSING = object()


def _geometry(
    *,
    site_id: str,
    flow_domain: Any,
    flow_line: Any,
    local_line: Any,
    local_equals_flow: bool = False,
    flow_segment_id: str | None = None,
    local_segment_id: str | None = None,
) -> BoundaryGeometry:
    """建立單站測試幾何，保留 production BoundaryGeometry 的公開語意。

    ``flow_domain``、``flow_line`` 與 ``local_line`` 可以刻意傳入非法測試物件，
    讓 builder 自己驗證真正要消費的資料；``own_local_domain`` 仍使用合法
    polygon，避免錯誤來源混入不相關的 fixture 欄位。開放邊界長度均以幾何的
    公尺座標計算，測試不把經緯度當作距離。
    """

    valid_domain = box(-100.0, -100.0, 100.0, 100.0)
    return BoundaryGeometry(
        own_local_domain=valid_domain,
        flow_domain=flow_domain,
        foreign_local_domains={},
        own_local_open_boundary=local_line,
        flow_open_boundary=flow_line,
        local_equals_flow=local_equals_flow,
        own_local_boundary_segment_id=local_segment_id or f"{site_id}-local",
        flow_boundary_segment_id=flow_segment_id or f"{site_id}-flow",
    )


def _valid_geometry(
    *,
    site_id: str,
    flow_domain: Any = _MISSING,
    flow_line: Any = _MISSING,
    local_line: Any = _MISSING,
    local_equals_flow: bool = False,
    flow_segment_id: str | None = None,
    local_segment_id: str | None = None,
) -> BoundaryGeometry:
    """建立具有明確 flow/local 線段的合法單站 fixture。"""

    actual_flow_domain = (
        box(0.1, 0.1, 19.9, 29.9)
        if flow_domain is _MISSING
        else flow_domain
    )
    default_flow_line = LineString([(0.0, 0.0), (10.0, 0.0)])
    default_local_line = LineString([(0.0, 0.0), (0.0, 4.0)])
    return _geometry(
        site_id=site_id,
        flow_domain=actual_flow_domain,
        flow_line=(
            default_flow_line if flow_line is _MISSING else flow_line
        ),
        local_line=(
            None
            if local_equals_flow
            else default_local_line if local_line is _MISSING else local_line
        ),
        local_equals_flow=local_equals_flow,
        flow_segment_id=flow_segment_id,
        local_segment_id=local_segment_id,
    )


def _bundle(
    geometries: dict[str, BoundaryGeometry],
    *,
    projection_site_ids: tuple[str, ...] | None = None,
) -> BoundaryGeometryBundle:
    """以公開 bundle 類別建立可供 builder 消費的 projection 對應。

    ``projection_site_ids`` 可刻意與 geometry site set 不同，用於驗證 bundle
    不會靜默遺漏或新增站點。file/canonical hash 在本測試不參與 builder 的
    計算，因此使用空 mapping；正式 loader 仍會保存上游 manifest hash。
    """

    site_ids = (
        tuple(geometries)
        if projection_site_ids is None
        else projection_site_ids
    )
    projections = {
        site_id: DomainProjection(121.0 + index, 24.0 + index)
        for index, site_id in enumerate(site_ids)
    }
    return BoundaryGeometryBundle(
        geometries=geometries,
        projections=projections,
        file_sha256={},
        canonical_component_hashes={},
    )


def _centers(site_ids: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """建立與站點集合完全相同的 WGS84 投影中心 mapping。"""

    return {
        site_id: (121.0 + index, 24.0 + index)
        for index, site_id in enumerate(site_ids)
    }


def _write(
    target: Path,
    boundaries: BoundaryGeometryBundle | dict[str, BoundaryGeometry],
    centers: dict[str, tuple[float, float]],
    *,
    grid_cell_size_m: float = 10.0,
) -> Path:
    """以固定且合法的統計參數呼叫 production public builder。"""

    return write_aggregate_spec_from_boundaries(
        target,
        boundaries,
        run_id="builder-test-run",
        site_metric_centers_deg=centers,
        grid_cell_size_m=grid_cell_size_m,
        boundary_bin_size_m=5.0,
        kde_bandwidths_m=(5.0, 10.0, 20.0),
        age_bin_edges_seconds=(0.0, 1.0, 2.0),
        bootstrap_replicates=10,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=123,
    )


def test_two_site_output_bytes_and_hash_are_deterministic(tmp_path: Path) -> None:
    """兩站輸入即使改變 mapping 順序，也必須產生完全相同的 bytes 與 hash。"""

    geometries = {
        "beta": _valid_geometry(
            site_id="beta",
            flow_domain=box(101.2, -3.2, 140.0, 4.3),
            flow_line=LineString([(0.0, 0.0), (6.0, 0.0)]),
            local_line=LineString([(0.0, 0.0), (0.0, 3.0)]),
            flow_segment_id="beta-flow",
            local_segment_id="beta-local",
        ),
        "alpha": _valid_geometry(
            site_id="alpha",
            flow_domain=box(-10.1, -20.0, 30.1, 40.1),
            flow_line=LineString([(0.0, 0.0), (8.0, 0.0)]),
            local_line=LineString([(0.0, 0.0), (0.0, 2.0)]),
            flow_segment_id="alpha-flow",
            local_segment_id="alpha-local",
        ),
    }
    reversed_geometries = {
        site_id: geometries[site_id] for site_id in reversed(tuple(geometries))
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    centers = _centers(("alpha", "beta"))
    _write(first, _bundle(geometries), dict(reversed(centers.items())))
    _write(second, _bundle(reversed_geometries), centers)

    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    assert first_bytes == second_bytes
    first_hash = hashlib.sha256(first_bytes).hexdigest()
    second_hash = hashlib.sha256(second_bytes).hexdigest()
    assert first_hash == second_hash

    first_spec = load_aggregate_spec(first)
    second_spec = load_aggregate_spec(second)
    assert first_spec.source_sha256 == first_hash
    assert second_spec.source_sha256 == second_hash
    assert first_spec.source_sha256 == second_spec.source_sha256
    assert first_spec.canonical_sha256 == second_spec.canonical_sha256
    assert set(first_spec.site_grids) == {"alpha", "beta"}


def test_flow_bounds_use_outward_floor_ceil_with_negative_coordinates(
    tmp_path: Path,
) -> None:
    """flow polygon 四軸須以 grid 向外 floor/ceil，不能因負座標向內裁切。"""

    boundaries = {
        "negative-site": _valid_geometry(
            site_id="negative-site",
            flow_domain=box(-10.1, -20.0, 30.1, 40.1),
        )
    }
    output = tmp_path / "negative.json"

    _write(output, boundaries, _centers(("negative-site",)))
    grid = load_aggregate_spec(output).site_grids["negative-site"]

    assert (grid.x_min_m, grid.x_max_m, grid.y_min_m, grid.y_max_m) == (
        -20.0,
        40.0,
        -20.0,
        50.0,
    )


def test_general_site_uses_separate_local_and_outer_segments_and_lengths(
    tmp_path: Path,
) -> None:
    """一般站 local/outer 應引用不同 segment，且長度須來自各自 open line。"""

    geometry = _valid_geometry(
        site_id="general",
        flow_line=LineString([(0.0, 0.0), (12.0, 0.0)]),
        local_line=LineString([(0.0, 0.0), (0.0, 5.0)]),
        flow_segment_id="general-flow",
        local_segment_id="general-local",
    )
    output = tmp_path / "general.json"

    _write(output, {"general": geometry}, _centers(("general",)))
    spec = load_aggregate_spec(output)
    segments = spec.site_boundary_segment_ids["general"]

    assert segments.local == ("general-local",)
    assert segments.outer == ("general-flow",)
    assert spec.boundary_segment_lengths_m["general-local"] == pytest.approx(5.0)
    assert spec.boundary_segment_lengths_m["general-flow"] == pytest.approx(12.0)


def test_local_equals_flow_references_one_flow_segment_for_both_roles(
    tmp_path: Path,
) -> None:
    """local_equals_flow 時兩個角色必須精確使用同一 flow segment ID。"""

    geometry = _valid_geometry(
        site_id="same-boundary",
        local_equals_flow=True,
        flow_line=LineString([(0.0, 0.0), (7.0, 0.0)]),
        flow_segment_id="same-boundary-flow",
        local_segment_id="unused-local-id",
    )
    output = tmp_path / "same-boundary.json"

    _write(output, {"same-boundary": geometry}, _centers(("same-boundary",)))
    spec = load_aggregate_spec(output)
    segments = spec.site_boundary_segment_ids["same-boundary"]

    assert segments.local == segments.outer == ("same-boundary-flow",)
    assert set(spec.boundary_segment_lengths_m) == {"same-boundary-flow"}
    assert spec.boundary_segment_lengths_m["same-boundary-flow"] == pytest.approx(7.0)


def test_bundle_geometry_and_projection_site_sets_must_match(tmp_path: Path) -> None:
    """BoundaryGeometryBundle 不得以不一致的 geometry/projection site set 發布。"""

    geometry = _valid_geometry(site_id="geometry-site")
    bundle = _bundle(
        {"geometry-site": geometry},
        projection_site_ids=("projection-site",),
    )
    output = tmp_path / "mismatch.json"

    with pytest.raises(ValueError):
        _write(output, bundle, {"geometry-site": (121.0, 24.0)})
    assert not output.exists()


@pytest.mark.parametrize(
    "centers",
    [
        {"other-site": (121.0, 24.0)},
        {
            "center-site": (121.0, 24.0),
            "extra-site": (122.0, 25.0),
        },
    ],
)
def test_metric_center_site_set_must_match(
    tmp_path: Path,
    centers: dict[str, tuple[float, float]],
) -> None:
    """metric center mapping 必須與 geometry site set 完全相同。"""

    boundaries = {"center-site": _valid_geometry(site_id="center-site")}

    with pytest.raises(ValueError):
        _write(tmp_path / f"center-set-{len(centers)}.json", boundaries, centers)


@pytest.mark.parametrize(
    "center",
    [
        (180.000001, 24.0),
        (-180.000001, 24.0),
        (121.0, 90.0),
        (121.0, -90.0),
        (float("nan"), 24.0),
        (121.0, float("inf")),
    ],
)
def test_metric_center_must_be_within_wgs84_finite_range(
    tmp_path: Path,
    center: tuple[float, float],
) -> None:
    """AEQD 中心經緯度須有限且符合 WGS84 經緯範圍。"""

    boundaries = {"center-site": _valid_geometry(site_id="center-site")}
    output = tmp_path / f"invalid-center-{len(list(tmp_path.iterdir()))}.json"

    with pytest.raises(ValueError):
        _write(output, boundaries, {"center-site": center})
    assert not output.exists()


@pytest.mark.parametrize(
    "flow_domain",
    [
        Polygon(),
        Polygon([(0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)]),
    ],
)
def test_empty_or_invalid_flow_polygon_is_rejected(
    tmp_path: Path,
    flow_domain: Polygon,
) -> None:
    """空 polygon 與 self-intersection polygon 都不可建立 metric grid。"""

    boundaries = {
        "bad-polygon": _valid_geometry(
            site_id="bad-polygon",
            flow_domain=flow_domain,
        )
    }

    with pytest.raises(ValueError):
        _write(
            tmp_path / f"bad-polygon-{len(list(tmp_path.iterdir()))}.json",
            boundaries,
            _centers(("bad-polygon",)),
        )


@pytest.mark.parametrize(
    "flow_line",
    [
        None,
        LineString([(0.0, 0.0), (float("nan"), 1.0)]),
        LineString([(0.0, 0.0), (0.0, 0.0)]),
    ],
)
def test_none_invalid_or_zero_length_flow_open_line_is_rejected(
    tmp_path: Path,
    flow_line: Any,
) -> None:
    """flow open line 的 None、invalid 或零長輸入都必須在 builder 邊界失敗。"""

    boundaries = {
        "bad-line": _valid_geometry(
            site_id="bad-line",
            flow_line=flow_line,
            local_equals_flow=True,
        )
    }

    with pytest.raises(ValueError):
        _write(
            tmp_path / f"bad-line-{len(list(tmp_path.iterdir()))}.json",
            boundaries,
            _centers(("bad-line",)),
        )


def test_same_segment_id_with_different_lengths_is_rejected(tmp_path: Path) -> None:
    """跨站共用 segment ID 時，弧長不一致不得被靜默覆蓋。"""

    boundaries = {
        "short": _valid_geometry(
            site_id="short",
            flow_line=LineString([(0.0, 0.0), (4.0, 0.0)]),
            flow_segment_id="shared-flow",
        ),
        "long": _valid_geometry(
            site_id="long",
            flow_line=LineString([(0.0, 0.0), (8.0, 0.0)]),
            flow_segment_id="shared-flow",
        ),
    }
    output = tmp_path / "different-length.json"

    with pytest.raises(ValueError):
        _write(output, boundaries, _centers(("short", "long")))
    assert not output.exists()


def test_existing_target_is_not_overwritten(tmp_path: Path) -> None:
    """既有目標檔案必須保留原始 bytes，builder 不得覆寫成果。"""

    boundaries = {"existing": _valid_geometry(site_id="existing")}
    centers = _centers(("existing",))
    output = tmp_path / "existing.json"
    _write(output, boundaries, centers)
    original_bytes = output.read_bytes()

    with pytest.raises(FileExistsError):
        _write(output, boundaries, centers)

    assert output.read_bytes() == original_bytes
    assert not list(tmp_path.glob(f".{output.name}.partial-*"))


def test_target_symlink_is_not_overwritten(tmp_path: Path) -> None:
    """即使 symlink 目標存在，builder 也必須拒絕發布並保護其指向內容。"""

    boundaries = {"symlink": _valid_geometry(site_id="symlink")}
    centers = _centers(("symlink",))
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_bytes(b"sentinel-bytes")
    target = tmp_path / "symlink.json"
    target.symlink_to(sentinel)

    with pytest.raises(FileExistsError):
        _write(target, boundaries, centers)

    assert target.is_symlink()
    assert sentinel.read_bytes() == b"sentinel-bytes"
    assert not list(tmp_path.glob(f".{target.name}.partial-*"))


def test_failed_partial_validation_is_not_published_or_left_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """partial JSON 驗證失敗時不得發布 target，且同父目錄暫存檔必須清理。"""

    boundaries = {"partial": _valid_geometry(site_id="partial")}
    target = tmp_path / "partial.json"

    def reject_partial(_: Path) -> Any:
        """模擬 loader 在發布前拒絕 partial；只用於驗證清理契約。"""

        raise ValueError("刻意模擬 partial 驗證失敗")

    monkeypatch.setattr(aggregate_spec_module, "load_aggregate_spec", reject_partial)

    with pytest.raises(ValueError, match="刻意模擬"):
        _write(target, boundaries, _centers(("partial",)))

    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.partial-*"))


def test_input_mappings_are_snapshotted_after_write(tmp_path: Path) -> None:
    """寫檔後修改呼叫端 mapping 不得改變已發布 JSON 或其 loader 結果。"""

    boundaries = {"snapshot": _valid_geometry(site_id="snapshot")}
    centers = {"snapshot": (121.0, 24.0)}
    output = tmp_path / "snapshot.json"
    _write(output, boundaries, centers)
    original_bytes = output.read_bytes()

    boundaries.clear()
    centers["snapshot"] = (122.0, 25.0)
    centers["new-site"] = (123.0, 26.0)

    assert output.read_bytes() == original_bytes
    spec = load_aggregate_spec(output)
    assert spec.site_metric_crs["snapshot"].center_lon_deg == 121.0
    assert spec.site_metric_crs["snapshot"].center_lat_deg == 24.0
