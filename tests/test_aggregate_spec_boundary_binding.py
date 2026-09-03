"""AggregateSpec 與已驗證邊界幾何 binding 的公開契約測試。

本檔以 ``test_aggregate_spec_builder`` 提供的最小公尺制 geometry fixture 建立
規格，再用公開 validator 驗證 caller 提供的 ``AggregateSpec`` 確實由同一組
flow bounds、open-water 線段、local/outer role 與明示的 WGS84 投影中心導出。
測試中的 Polygon、LineString、AEQD 中心與網格值都是 synthetic 工程資料；它們
只驗證 m、degree 與 mapping 的資料契約，不是 OCM/NWW 實際資料或科學成果。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import test_aggregate_spec_builder as builder_fixture
from shapely.geometry import LineString, box

from lagrangian_backtracking.aggregate_spec import (
    AggregateSpec,
    load_aggregate_spec,
    validate_aggregate_spec_against_boundaries,
)
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.manifests import BoundaryGeometryBundle


def _generated_case(
    tmp_path: Path,
    *,
    as_bundle: bool,
) -> tuple[
    AggregateSpec,
    BoundaryGeometryBundle | dict[str, BoundaryGeometry],
    dict[str, tuple[float, float]],
    Path,
]:
    """以既有 builder fixture 產生一份載入後 spec 與對應的 mapping／Bundle。

    writer 仍負責建立 JSON、source hash 與 canonical hash；本 helper 只把 validator
    要重驗的 in-memory inputs 與其輸出放在同一個測試案例中，避免手工複製 writer
    的幾何算法。``as_bundle`` 只控制是否附帶 projection site set metadata。
    """

    geometry = builder_fixture._valid_geometry(
        site_id="binding-site",
        flow_domain=box(0.1, 0.1, 19.9, 29.9),
        flow_line=LineString([(0.0, 0.0), (10.0, 0.0)]),
        local_line=LineString([(0.0, 0.0), (0.0, 4.0)]),
    )
    boundaries: BoundaryGeometryBundle | dict[str, BoundaryGeometry]
    if as_bundle:
        boundaries = builder_fixture._bundle({"binding-site": geometry})
    else:
        boundaries = {"binding-site": geometry}
    centers = builder_fixture._centers(("binding-site",))
    target = tmp_path / "generated" / "aggregate-spec.json"
    builder_fixture._write(target, boundaries, centers)
    return load_aggregate_spec(target), boundaries, centers, target


def _input_snapshot(
    boundaries: BoundaryGeometryBundle | dict[str, BoundaryGeometry],
    centers: dict[str, tuple[float, float]],
) -> tuple[Any, ...]:
    """擷取 validator 讀取的 geometry、Bundle site set 與中心 mapping snapshot。"""

    geometry_mapping = (
        boundaries.by_study_site
        if isinstance(boundaries, BoundaryGeometryBundle)
        else boundaries
    )

    def geometry_snapshot(geometry: BoundaryGeometry) -> tuple[Any, ...]:
        """以 WKB bytes 與 scalar metadata 表示 geometry 的不可變測試快照。"""

        return (
            geometry.own_local_domain.wkb,
            geometry.flow_domain.wkb,
            None if geometry.own_local_open_boundary is None else geometry.own_local_open_boundary.wkb,
            None if geometry.flow_open_boundary is None else geometry.flow_open_boundary.wkb,
            tuple(sorted((key, value.wkb) for key, value in geometry.foreign_local_domains.items())),
            geometry.local_equals_flow,
            geometry.own_local_boundary_segment_id,
            geometry.flow_boundary_segment_id,
        )

    projection_snapshot = (
        tuple(sorted((site_id, id(projection)) for site_id, projection in boundaries.projections.items()))
        if isinstance(boundaries, BoundaryGeometryBundle)
        else ()
    )
    return (
        tuple(
            sorted(
                (site_id, geometry_snapshot(geometry))
                for site_id, geometry in geometry_mapping.items()
            )
        ),
        projection_snapshot,
        tuple(sorted((site_id, tuple(center)) for site_id, center in centers.items())),
    )


def _assert_binding_error(operation: Callable[[], Any], tmp_path: Path) -> None:
    """固定 validator 失敗為 ValueError，且錯誤不得洩漏測試暫存路徑。"""

    with pytest.raises(ValueError) as exc_info:
        operation()
    assert str(tmp_path) not in str(exc_info.value)


@pytest.mark.parametrize("as_bundle", [False, True], ids=["mapping", "bundle"])
def test_loaded_writer_spec_binds_to_mapping_or_bundle_without_mutation(
    tmp_path: Path,
    as_bundle: bool,
) -> None:
    """writer 產生並 load 的 spec，對 mapping 與 Bundle 都應驗證成功且輸入不變。"""

    spec, boundaries, centers, target = _generated_case(tmp_path, as_bundle=as_bundle)
    input_before = _input_snapshot(boundaries, centers)
    file_before = target.read_bytes()
    tree_before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))

    assert validate_aggregate_spec_against_boundaries(
        spec,
        boundaries,
        site_metric_centers_deg=centers,
    ) is None

    assert _input_snapshot(boundaries, centers) == input_before
    assert target.read_bytes() == file_before
    assert tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))) == tree_before


def test_grid_bounds_derived_from_changed_flow_geometry_are_rejected(tmp_path: Path) -> None:
    """flow polygon bounds 改變後，即使 open line 與中心不變，也不得沿用舊 grid。"""

    spec, _, centers, _ = _generated_case(tmp_path, as_bundle=False)
    changed = {
        "binding-site": builder_fixture._valid_geometry(
            site_id="binding-site",
            flow_domain=box(0.1, 0.1, 29.9, 29.9),
        )
    }

    _assert_binding_error(
        lambda: validate_aggregate_spec_against_boundaries(
            spec,
            changed,
            site_metric_centers_deg=centers,
        ),
        tmp_path,
    )


def test_changed_explicit_metric_center_is_rejected(tmp_path: Path) -> None:
    """AEQD 中心是 caller 明示的度數輸入，改變中心不得被 validator 忽略。"""

    spec, boundaries, _, _ = _generated_case(tmp_path, as_bundle=False)
    changed_centers = {"binding-site": (122.0, 24.0)}

    _assert_binding_error(
        lambda: validate_aggregate_spec_against_boundaries(
            spec,
            boundaries,
            site_metric_centers_deg=changed_centers,
        ),
        tmp_path,
    )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("projection_method", "different_projection_method"),
        ("linear_unit", "degree"),
        ("axis_order", "y_north_x_east"),
    ],
    ids=["projection-method", "linear-unit", "axis-order"],
)
def test_tampered_metric_projection_contract_fields_are_rejected(
    tmp_path: Path,
    field: str,
    tampered_value: str,
) -> None:
    """投影 method、單位與軸序任一被竄改時，validator 都須拒絕資料契約不一致。

    ``SiteMetricCRSSpec`` 的 constructor 本身會拒絕非固定值，因此先以
    ``dataclasses.replace`` 建立 detached value，再用測試專用的 frozen-object
    tamper 模擬「已載入 spec 在記憶體中被竄改」的狀況；這樣確認的是公開
    validator 對 projection_method、linear_unit、axis_order 三個投影資料契約
    欄位的逐項比對，而非只測 nested constructor 的 schema gate。
    """

    spec, boundaries, centers, _ = _generated_case(tmp_path, as_bundle=False)
    detached_metric = replace(spec.site_metric_crs["binding-site"])
    object.__setattr__(detached_metric, field, tampered_value)
    detached_spec = replace(
        spec,
        site_metric_crs={"binding-site": detached_metric},
    )

    _assert_binding_error(
        lambda: validate_aggregate_spec_against_boundaries(
            detached_spec,
            boundaries,
            site_metric_centers_deg=centers,
        ),
        tmp_path,
    )


@pytest.mark.parametrize(
    "change",
    ["length", "id", "role"],
    ids=["segment-length", "segment-id", "segment-role"],
)
def test_changed_segment_length_id_or_local_outer_role_is_rejected(
    tmp_path: Path,
    change: str,
) -> None:
    """open segment 的公尺弧長、識別碼及 local/outer role 改變都必須拒絕。"""

    spec, _, centers, _ = _generated_case(tmp_path, as_bundle=False)
    if change == "length":
        changed_geometry = builder_fixture._valid_geometry(
            site_id="binding-site",
            flow_line=LineString([(0.0, 0.0), (11.0, 0.0)]),
        )
    elif change == "id":
        changed_geometry = builder_fixture._valid_geometry(
            site_id="binding-site",
            flow_segment_id="binding-flow-changed",
        )
    else:
        # local_equals_flow 會讓 flow segment 同時成為 local 與 outer；原規格則有
        # 獨立的 own-local line/ID，兩者 role mapping 不可視為同義。
        changed_geometry = builder_fixture._valid_geometry(
            site_id="binding-site",
            local_equals_flow=True,
        )
    changed = {"binding-site": changed_geometry}

    _assert_binding_error(
        lambda: validate_aggregate_spec_against_boundaries(
            spec,
            changed,
            site_metric_centers_deg=centers,
        ),
        tmp_path,
    )


@pytest.mark.parametrize("site_change", ["missing", "extra"], ids=["site-missing", "site-extra"])
def test_boundary_site_set_must_match_loaded_spec(tmp_path: Path, site_change: str) -> None:
    """邊界 geometry 的站點集合少站或多站時，不得與既有 spec 靜默配對。"""

    spec, _, _, _ = _generated_case(tmp_path, as_bundle=False)
    other_geometry = builder_fixture._valid_geometry(site_id="other-site")
    if site_change == "missing":
        boundaries = {"other-site": other_geometry}
        centers = {"other-site": (121.0, 24.0)}
    else:
        original_geometry = builder_fixture._valid_geometry(site_id="binding-site")
        boundaries = {
            "binding-site": original_geometry,
            "other-site": other_geometry,
        }
        centers = {
            "binding-site": (121.0, 24.0),
            "other-site": (122.0, 25.0),
        }

    _assert_binding_error(
        lambda: validate_aggregate_spec_against_boundaries(
            spec,
            boundaries,
            site_metric_centers_deg=centers,
        ),
        tmp_path,
    )


def test_bundle_projection_site_set_must_match_geometry_site_set(tmp_path: Path) -> None:
    """Bundle 的 projections site set 不一致時，validator 必須在 geometry binding 前拒絕。"""

    spec, _, centers, _ = _generated_case(tmp_path, as_bundle=False)
    geometry = builder_fixture._valid_geometry(site_id="binding-site")
    invalid_bundle = builder_fixture._bundle(
        {"binding-site": geometry},
        projection_site_ids=("projection-only-site",),
    )

    _assert_binding_error(
        lambda: validate_aggregate_spec_against_boundaries(
            spec,
            invalid_bundle,
            site_metric_centers_deg=centers,
        ),
        tmp_path,
    )


def test_validator_requires_exact_aggregate_spec_instance(tmp_path: Path) -> None:
    """validator 不接受 AggregateSpec subclass 或只提供相似欄位的 duck type。"""

    spec, boundaries, centers, _ = _generated_case(tmp_path, as_bundle=False)

    class AggregateSpecChild(AggregateSpec):
        """只供測試 exact-type gate 的 subclass，不代表另一種規格 schema。"""

    subclass_instance = object.__new__(AggregateSpecChild)
    duck_instance = type(
        "AggregateSpecDuck",
        (),
        {
            "grid_cell_size_m": spec.grid_cell_size_m,
            "site_grids": spec.site_grids,
            "site_metric_crs": spec.site_metric_crs,
            "boundary_segment_lengths_m": spec.boundary_segment_lengths_m,
            "site_boundary_segment_ids": spec.site_boundary_segment_ids,
        },
    )()

    for candidate in (subclass_instance, duck_instance):
        _assert_binding_error(
            lambda candidate=candidate: validate_aggregate_spec_against_boundaries(
                candidate,
                boundaries,
                site_metric_centers_deg=centers,
            ),
            tmp_path,
        )


def test_validator_does_not_validate_hashes_or_guess_run_id(tmp_path: Path) -> None:
    """hash bytes 真實性與 run identity 分屬 loader／pipeline，非 geometry validator 責任。"""

    spec, boundaries, centers, _ = _generated_case(tmp_path, as_bundle=False)
    detached_spec = replace(
        spec,
        run_id="different-run-identity",
        source_sha256="1" * 64,
        canonical_sha256="2" * 64,
    )

    assert validate_aggregate_spec_against_boundaries(
        detached_spec,
        boundaries,
        site_metric_centers_deg=centers,
    ) is None


def test_validator_writes_no_files_and_binding_errors_have_no_tmp_path(tmp_path: Path) -> None:
    """validator 只讀 in-memory inputs；成功與失敗都不得新增檔案或洩漏 tmp path。"""

    spec, boundaries, centers, target = _generated_case(tmp_path, as_bundle=False)
    files_before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    invalid_centers = {"binding-site": (123.0, 24.0)}
    _assert_binding_error(
        lambda: validate_aggregate_spec_against_boundaries(
            spec,
            boundaries,
            site_metric_centers_deg=invalid_centers,
        ),
        tmp_path,
    )
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == files_before
    assert target.exists()
