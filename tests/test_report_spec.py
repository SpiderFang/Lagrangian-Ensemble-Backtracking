"""正式 report spec 的資料契約、aggregate binding 與原子 I/O 測試。

本檔只在 pytest 暫存目錄建立小型 JSON 規格檔與假 AggregateSpec，不讀取或寫入任何
報告圖表、表格、軌跡、OCM schema 3 或 NWW3 schema 1 產品。所有數值、SHA-256 與
路徑都是 synthetic engineering fixture，用來驗證 immutable schema、可重建的 renderer
policy、aggregate binding 及檔案安全性；測試通過不代表真實 OCM／NWW 科學成果、條件式
來源足跡、相對來源權重、絕對來源機率或觀測驗證已經成立。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import lagrangian_backtracking.report_spec as report_spec_module
from lagrangian_backtracking.aggregate_spec import (
    AGGREGATE_SPEC_SCHEMA_VERSION,
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
)
from lagrangian_backtracking.report_spec import (
    REPORT_SPEC_SCHEMA_VERSION,
    ReportSpec,
    load_report_spec,
    validate_report_spec_against_aggregate_spec,
    write_report_spec,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64
_THIRD_HASH = "c" * 64
_REPORT_POLICY = "stable_hash_core_season_tide_v1"
_TRAVEL_AGE_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_PATHWAY_QUANTILES = (0.25, 0.5, 0.75)
_VERTICAL_DEPTH_EDGES = (0.0, 5.0, 10.0)
_FIGURE_FORMATS = ("png", "svg", "pdf")
_REPORT_STYLE = "academic_zh_tw_v1"
_REPORT_LANGUAGE = "zh-TW"
_WRITE_FAILURE = "report spec 寫入或驗證失敗"

# 這份 root key set 是測試對 public JSON schema 的獨立描述，不讀取 production private
# constant。vertical depth bin edges 是公尺制（m）深度軸；其他 quantile 是秒制（s）
# travel-age／first-passage 軸。兩個 hash 是 loader/writer 衍生 metadata，不可由輸入
# JSON caller 直接提供。
_REPORT_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "aggregate_spec_canonical_sha256",
        "primary_kde_bandwidth_m",
        "minimum_kde_raw_count",
        "low_sample_min_member_count",
        "vertical_depth_bin_edges_m",
        "representative_trajectory_count_per_site",
        "representative_selection_policy",
        "representative_selection_seed",
        "travel_age_quantiles",
        "pathway_first_passage_quantiles",
        "figure_formats",
        "raster_dpi",
        "renderer_style_version",
        "language",
    }
)


@pytest.fixture(scope="module")
def aggregate_spec() -> AggregateSpec:
    """建立一站一 segment 的最小 AggregateSpec synthetic fixture。

    網格與邊界長度均使用公尺，KDE 頻寬固定為 100／200／300 m；source/canonical hash
    是格式正確的假 provenance。此物件只供 report binding 測試，並不代表任何真實海域
    或 OCM／NWW 輸入資料。
    """

    return AggregateSpec(
        schema_version=AGGREGATE_SPEC_SCHEMA_VERSION,
        run_id="synthetic-report-spec-run",
        grid_cell_size_m=100,
        site_grids={
            "synthetic-site": SiteGridSpec(
                x_min_m=0,
                x_max_m=1000,
                y_min_m=0,
                y_max_m=1000,
            )
        },
        site_metric_crs={
            "synthetic-site": SiteMetricCRSSpec(
                projection_method="azimuthal_equidistant_wgs84",
                center_lon_deg=121.0,
                center_lat_deg=24.0,
                linear_unit="m",
                axis_order="x_east_y_north",
            )
        },
        boundary_bin_size_m=100,
        boundary_segment_lengths_m={"synthetic-segment": 1000},
        site_boundary_segment_ids={
            "synthetic-site": SiteBoundarySegments(
                local_segment_ids=("synthetic-segment",),
                outer_segment_ids=("synthetic-segment",),
            )
        },
        kde_bandwidths_m=(100, 200, 300),
        hdr_levels=(0.5, 0.75, 0.9),
        age_bin_edges_seconds=(0, 10),
        bootstrap_replicates=10,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_and_numerical_failure_v1",
        source_sha256=_HASH,
        canonical_sha256=_OTHER_HASH,
    )


def _report_kwargs(aggregate: AggregateSpec, **overrides: object) -> dict[str, object]:
    """回傳一份欄位各自合法的 ReportSpec constructor kwargs。

    representative count 預設為 8，因為它代表 4 個 season × 2 個 spring/neap strata
    的一次串流等額取樣下限；vertical depth edges 以 list 傳入的案例會由 constructor
    轉成獨立的 float tuple。
    """

    values: dict[str, object] = {
        "schema_version": REPORT_SPEC_SCHEMA_VERSION,
        "run_id": aggregate.run_id,
        "aggregate_spec_canonical_sha256": aggregate.canonical_sha256,
        "primary_kde_bandwidth_m": 100,
        "minimum_kde_raw_count": 1,
        "low_sample_min_member_count": 1,
        "vertical_depth_bin_edges_m": _VERTICAL_DEPTH_EDGES,
        "representative_trajectory_count_per_site": 8,
        "representative_selection_policy": _REPORT_POLICY,
        "representative_selection_seed": 0,
        "travel_age_quantiles": _TRAVEL_AGE_QUANTILES,
        "pathway_first_passage_quantiles": _PATHWAY_QUANTILES,
        "figure_formats": _FIGURE_FORMATS,
        "raster_dpi": 300,
        "renderer_style_version": _REPORT_STYLE,
        "language": _REPORT_LANGUAGE,
        "source_sha256": _HASH,
        "canonical_sha256": _OTHER_HASH,
    }
    values.update(overrides)
    return values


def _report_document(aggregate: AggregateSpec, **overrides: object) -> dict[str, object]:
    """建立不含兩個衍生 hash 的 ReportSpec JSON input fixture。"""

    document: dict[str, object] = {
        "schema_version": REPORT_SPEC_SCHEMA_VERSION,
        "run_id": aggregate.run_id,
        "aggregate_spec_canonical_sha256": aggregate.canonical_sha256,
        "primary_kde_bandwidth_m": 100,
        "minimum_kde_raw_count": 1,
        "low_sample_min_member_count": 1,
        "vertical_depth_bin_edges_m": [0.0, 5.0, 10.0],
        "representative_trajectory_count_per_site": 8,
        "representative_selection_policy": _REPORT_POLICY,
        "representative_selection_seed": 0,
        "travel_age_quantiles": list(_TRAVEL_AGE_QUANTILES),
        "pathway_first_passage_quantiles": list(_PATHWAY_QUANTILES),
        "figure_formats": list(_FIGURE_FORMATS),
        "raster_dpi": 300,
        "renderer_style_version": _REPORT_STYLE,
        "language": _REPORT_LANGUAGE,
    }
    document.update(overrides)
    return document


def _json_bytes(
    document: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    allow_nan: bool = False,
    separators: tuple[str, str] | None = None,
) -> bytes:
    """以指定排版寫入 synthetic JSON bytes，保留 parser 邊界案例的控制權。"""

    options: dict[str, object] = {
        "ensure_ascii": False,
        "indent": indent,
        "sort_keys": sort_keys,
        "allow_nan": allow_nan,
    }
    if separators is not None:
        options["separators"] = separators
    return json.dumps(document, **options).encode("utf-8")


def _write_json(path: Path, document: object, **kwargs: object) -> None:
    """只在 pytest tmp_path 寫入小型 report spec fixture，不建立實際報告成果。"""

    path.write_bytes(_json_bytes(document, **kwargs))


def _canonical_input_bytes(document: dict[str, object]) -> bytes:
    """計算 loader 應得到的 canonical bytes，並將 depth edges 正規化成 float。

    ReportSpec 對 vertical depth bin edges 的 canonical contract 是 float tuple；因此
    JSON 輸入的 ``0`` 與 ``0.0`` 具有相同 canonical 語意，但 raw source bytes 仍可不同。
    此 helper 不讀取 production private function，避免測試與實作共用同一錯誤。
    """

    payload = dict(document)
    payload["vertical_depth_bin_edges_m"] = [
        float(value) for value in payload["vertical_depth_bin_edges_m"]  # type: ignore[union-attr]
    ]
    return _json_bytes(
        payload,
        indent=None,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _assert_load_failure(operation: Callable[[], object], path: Path) -> None:
    """確認 loader 對所有輸入錯誤統一成不洩漏 absolute path 的 ValueError。"""

    with pytest.raises(ValueError) as error_info:
        operation()
    assert type(error_info.value) is ValueError
    assert str(path) not in str(error_info.value)
    assert str(path) not in repr(error_info.value)


def _assert_write_failure(operation: Callable[[], object], path: Path) -> None:
    """確認 writer 一般失敗使用固定訊息且不攜帶 tmp／SERVER 路徑。"""

    with pytest.raises(ValueError) as error_info:
        operation()
    assert type(error_info.value) is ValueError
    assert str(error_info.value) == _WRITE_FAILURE
    assert error_info.value.__cause__ is None
    assert str(path) not in str(error_info.value)
    assert str(path) not in repr(error_info.value)


def _write_report_spec(
    target: Path,
    aggregate: AggregateSpec,
    **overrides: object,
) -> Path:
    """以所有研究參數明示的方式呼叫 public writer。"""

    values: dict[str, object] = {
        "primary_kde_bandwidth_m": 100,
        "minimum_kde_raw_count": 1,
        "low_sample_min_member_count": 1,
        "vertical_depth_bin_edges_m": _VERTICAL_DEPTH_EDGES,
        "representative_trajectory_count_per_site": 8,
        "representative_selection_seed": 0,
    }
    values.update(overrides)
    return write_report_spec(target, aggregate_spec=aggregate, **values)


def _partial_paths(target: Path) -> list[Path]:
    """列出 target 同父目錄下 writer 命名規則的所有 partial，含手工 marker。"""

    return sorted(target.parent.glob(f".{target.name}.partial-*"))


def _prepare_existing_target(target: Path, kind: str) -> tuple[bytes | None, str | None]:
    """建立 file/dir/symlink/broken symlink target，回傳可比較的既有狀態。"""

    if kind == "file":
        content = b"protected-final-bytes"
        target.write_bytes(content)
        return content, None
    if kind == "dir":
        target.mkdir()
        (target / "marker").write_bytes(b"protected-directory")
        return None, None
    if kind == "symlink":
        referent = target.parent / "referent.json"
        referent.write_bytes(b"protected-referent")
        os.symlink(referent, target)
        return None, os.readlink(target)
    if kind == "broken-symlink":
        referent = target.parent / "missing-referent.json"
        os.symlink(referent, target)
        return None, os.readlink(target)
    raise AssertionError(f"unknown target kind: {kind}")


def _report_spec_kwargs_from_instance(spec: ReportSpec) -> dict[str, object]:
    """複製 ReportSpec 欄位供 exact-subclass binding 測試，不依賴 repr。"""

    return {
        "schema_version": spec.schema_version,
        "run_id": spec.run_id,
        "aggregate_spec_canonical_sha256": spec.aggregate_spec_canonical_sha256,
        "primary_kde_bandwidth_m": spec.primary_kde_bandwidth_m,
        "minimum_kde_raw_count": spec.minimum_kde_raw_count,
        "low_sample_min_member_count": spec.low_sample_min_member_count,
        "vertical_depth_bin_edges_m": spec.vertical_depth_bin_edges_m,
        "representative_trajectory_count_per_site": spec.representative_trajectory_count_per_site,
        "representative_selection_policy": spec.representative_selection_policy,
        "representative_selection_seed": spec.representative_selection_seed,
        "travel_age_quantiles": spec.travel_age_quantiles,
        "pathway_first_passage_quantiles": spec.pathway_first_passage_quantiles,
        "figure_formats": spec.figure_formats,
        "raster_dpi": spec.raster_dpi,
        "renderer_style_version": spec.renderer_style_version,
        "language": spec.language,
        "source_sha256": spec.source_sha256,
        "canonical_sha256": spec.canonical_sha256,
    }


def _aggregate_spec_kwargs_from_instance(aggregate: AggregateSpec) -> dict[str, object]:
    """複製最小 AggregateSpec 欄位供 exact-subclass binding 測試。"""

    return {
        "schema_version": aggregate.schema_version,
        "run_id": aggregate.run_id,
        "grid_cell_size_m": aggregate.grid_cell_size_m,
        "site_grids": aggregate.site_grids,
        "site_metric_crs": aggregate.site_metric_crs,
        "boundary_bin_size_m": aggregate.boundary_bin_size_m,
        "boundary_segment_lengths_m": aggregate.boundary_segment_lengths_m,
        "site_boundary_segment_ids": aggregate.site_boundary_segment_ids,
        "kde_bandwidths_m": aggregate.kde_bandwidths_m,
        "hdr_levels": aggregate.hdr_levels,
        "age_bin_edges_seconds": aggregate.age_bin_edges_seconds,
        "bootstrap_replicates": aggregate.bootstrap_replicates,
        "bootstrap_confidence_level": aggregate.bootstrap_confidence_level,
        "bootstrap_seed": aggregate.bootstrap_seed,
        "denominator_policy": aggregate.denominator_policy,
        "source_sha256": aggregate.source_sha256,
        "canonical_sha256": aggregate.canonical_sha256,
    }


def test_report_spec_public_api_constants_and_no_heavy_science_imports() -> None:
    """public surface 必須固定，report_spec source 不得依賴大型科學／繪圖套件。"""

    assert REPORT_SPEC_SCHEMA_VERSION == "1.0.0"
    assert report_spec_module.__all__ == [
        "REPORT_SPEC_SCHEMA_VERSION",
        "ReportSpec",
        "load_report_spec",
        "validate_report_spec_against_aggregate_spec",
        "write_report_spec",
    ]

    source_path = Path(report_spec_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    forbidden = {"numpy", "matplotlib", "pyarrow", "scipy"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(forbidden)


def test_report_spec_constructor_to_dict_is_defensive_json_ready_and_frozen(
    aggregate_spec: AggregateSpec,
) -> None:
    """合法 ReportSpec 應保存 policy、hash 與 float depth tuple，且隔離 caller mutation。"""

    depth_edges = [0, 5, 10]
    spec = ReportSpec(
        **_report_kwargs(
            aggregate_spec,
            vertical_depth_bin_edges_m=depth_edges,
        )
    )
    depth_edges.append(999)
    assert spec.vertical_depth_bin_edges_m == _VERTICAL_DEPTH_EDGES
    assert isinstance(spec.vertical_depth_bin_edges_m, tuple)
    assert all(type(edge) is float for edge in spec.vertical_depth_bin_edges_m)
    assert spec.representative_selection_policy == _REPORT_POLICY
    assert spec.source_sha256 == _HASH
    assert spec.canonical_sha256 == _OTHER_HASH

    encoded = spec.to_dict()
    assert set(encoded) == _REPORT_ROOT_KEYS | {"source_sha256", "canonical_sha256"}
    assert type(encoded["vertical_depth_bin_edges_m"]) is list
    encoded["vertical_depth_bin_edges_m"].append(123.0)  # type: ignore[union-attr]
    assert spec.vertical_depth_bin_edges_m == _VERTICAL_DEPTH_EDGES
    assert json.loads(json.dumps(encoded, ensure_ascii=False, sort_keys=True)) == encoded
    with pytest.raises(AttributeError):
        spec.primary_kde_bandwidth_m = 200  # type: ignore[misc]


@pytest.mark.parametrize("representative_count", (8, 16))
def test_representative_trajectory_count_accepts_eight_and_sixteen(
    aggregate_spec: AggregateSpec,
    representative_count: int,
) -> None:
    """4 season × 2 spring/neap strata 的等額取樣允許 8 與 16。"""

    spec = ReportSpec(
        **_report_kwargs(
            aggregate_spec,
            representative_trajectory_count_per_site=representative_count,
        )
    )
    assert spec.representative_trajectory_count_per_site == representative_count


@pytest.mark.parametrize("vertical_edges", ((0, 5), (0.0, 5.0, 10.0), (0, 5, 10, 15)))
def test_vertical_depth_edges_accept_two_or_more_strictly_increasing_native_numbers(
    aggregate_spec: AggregateSpec,
    vertical_edges: tuple[int | float, ...],
) -> None:
    """深度軸至少兩點、首值精確 0，輸出固定為 defensive float tuple。"""

    spec = ReportSpec(
        **_report_kwargs(
            aggregate_spec,
            vertical_depth_bin_edges_m=vertical_edges,
        )
    )
    assert spec.vertical_depth_bin_edges_m == tuple(float(edge) for edge in vertical_edges)
    assert spec.to_dict()["vertical_depth_bin_edges_m"] == [
        float(edge) for edge in vertical_edges
    ]


@pytest.mark.parametrize(
    "field, value",
    (
        ("schema_version", "2.0.0"),
        ("run_id", "../unsafe"),
        ("aggregate_spec_canonical_sha256", "A" * 64),
        ("primary_kde_bandwidth_m", 0),
        ("primary_kde_bandwidth_m", -1),
        ("primary_kde_bandwidth_m", True),
        ("primary_kde_bandwidth_m", np.int64(100)),
        ("primary_kde_bandwidth_m", np.float64(100.0)),
        ("primary_kde_bandwidth_m", float("nan")),
        ("primary_kde_bandwidth_m", float("inf")),
        ("minimum_kde_raw_count", 0),
        ("minimum_kde_raw_count", -1),
        ("minimum_kde_raw_count", True),
        ("minimum_kde_raw_count", np.int64(1)),
        ("minimum_kde_raw_count", 1.0),
        ("low_sample_min_member_count", 0),
        ("low_sample_min_member_count", -1),
        ("low_sample_min_member_count", True),
        ("low_sample_min_member_count", np.int64(1)),
        ("low_sample_min_member_count", 1.0),
        ("representative_trajectory_count_per_site", 1),
        ("representative_trajectory_count_per_site", 7),
        ("representative_trajectory_count_per_site", 9),
        ("representative_trajectory_count_per_site", 12),
        ("representative_trajectory_count_per_site", 0),
        ("representative_trajectory_count_per_site", -1),
        ("representative_trajectory_count_per_site", True),
        ("representative_trajectory_count_per_site", np.int64(8)),
        ("representative_trajectory_count_per_site", 8.0),
        ("representative_selection_policy", "stable_hash_stratified_quantile_v1"),
        ("representative_selection_seed", -1),
        ("representative_selection_seed", 2**128),
        ("representative_selection_seed", True),
        ("representative_selection_seed", np.int64(0)),
        ("representative_selection_seed", 0.0),
        ("representative_selection_seed", float("nan")),
        ("representative_selection_seed", float("inf")),
        ("travel_age_quantiles", (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
        ("travel_age_quantiles", (0.05, 0.25, float("nan"), 0.75, 0.95)),
        ("travel_age_quantiles", (0.05, 0.25, np.float64(0.5), 0.75, 0.95)),
        ("pathway_first_passage_quantiles", (0.25, 0.5)),
        ("pathway_first_passage_quantiles", (0.25, np.float64(0.5), 0.75)),
        ("figure_formats", ("png", "svg", "jpg")),
        ("raster_dpi", 299),
        ("raster_dpi", True),
        ("raster_dpi", np.int64(300)),
        ("raster_dpi", 300.0),
        ("renderer_style_version", "default"),
        ("language", "en-US"),
        ("source_sha256", "A" * 64),
        ("source_sha256", "short"),
        ("source_sha256", np.str_("a" * 64)),
        ("canonical_sha256", "g" * 64),
        ("canonical_sha256", b"a" * 64),
    ),
)
def test_report_spec_constructor_rejects_each_illegal_value_in_isolation(
    aggregate_spec: AggregateSpec,
    field: str,
    value: object,
) -> None:
    """每個非法 scalar／policy／hash 案例只竄改一欄，避免雙重非法造成假綠。"""

    kwargs = _report_kwargs(aggregate_spec)
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        ReportSpec(**kwargs)


@pytest.mark.parametrize(
    "vertical_edges",
    (
        (),
        (0.0,),
        (1.0, 5.0),
        (0.0, 5.0, 5.0),
        (0.0, 10.0, 5.0),
        (0.0, float("nan"), 10.0),
        (0.0, float("inf"), 10.0),
        (0.0, -float("inf"), 10.0),
        (True, 5.0),
        (0.0, np.float64(5.0), 10.0),
        (0.0, "5.0", 10.0),
        (0.0, {"edge": 5.0}, 10.0),
        {"first": 0.0, "second": 5.0},
    ),
)
def test_vertical_depth_edges_reject_each_shape_type_order_or_finite_violation(
    aggregate_spec: AggregateSpec,
    vertical_edges: object,
) -> None:
    """vertical depth edges 不得接受少於兩點、非零起點、非遞增或非 native numbers。"""

    kwargs = _report_kwargs(aggregate_spec, vertical_depth_bin_edges_m=vertical_edges)
    with pytest.raises((TypeError, ValueError)):
        ReportSpec(**kwargs)


def test_load_report_spec_validates_fixture_hashes_keys_and_vertical_depth_axis(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
) -> None:
    """loader 應嚴格讀取 JSON，並由 raw bytes／normalized canonical payload 推導 hash。"""

    document = _report_document(aggregate_spec)
    path = tmp_path / "report_spec.json"
    raw = _json_bytes(document)
    path.write_bytes(raw)
    spec = load_report_spec(path)

    assert isinstance(spec, ReportSpec)
    assert spec.run_id == aggregate_spec.run_id
    assert spec.aggregate_spec_canonical_sha256 == aggregate_spec.canonical_sha256
    assert spec.primary_kde_bandwidth_m == 100
    assert spec.vertical_depth_bin_edges_m == _VERTICAL_DEPTH_EDGES
    assert type(spec.vertical_depth_bin_edges_m) is tuple
    assert spec.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert spec.canonical_sha256 == hashlib.sha256(_canonical_input_bytes(document)).hexdigest()
    assert set(spec.to_dict()) == _REPORT_ROOT_KEYS | {"source_sha256", "canonical_sha256"}


@pytest.mark.parametrize("case", ("unknown", "missing", "source-hash", "canonical-hash"))
def test_load_report_spec_requires_exact_root_keys(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    case: str,
) -> None:
    """root key 必須 exact；unknown、missing 與 caller supplied derived hash 都拒絕。"""

    document = _report_document(aggregate_spec)
    if case == "unknown":
        document["unexpected"] = 1
    elif case == "missing":
        del document["vertical_depth_bin_edges_m"]
    elif case == "source-hash":
        document["source_sha256"] = _HASH
    else:
        document["canonical_sha256"] = _OTHER_HASH
    path = tmp_path / f"{case}.json"
    _write_json(path, document)
    _assert_load_failure(lambda: load_report_spec(path), path)


def test_load_report_spec_rejects_duplicate_root_keys(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
) -> None:
    """JSON object 的 duplicate key 不得由 decoder 靜默採用最後一個值。"""

    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        encoding="utf-8",
    )
    _assert_load_failure(lambda: load_report_spec(path), path)


@pytest.mark.parametrize(
    "field, bad_value",
    (
        ("travel_age_quantiles", 1),
        ("pathway_first_passage_quantiles", {"q": 0.5}),
        ("figure_formats", "png"),
        ("vertical_depth_bin_edges_m", 5),
    ),
)
def test_load_report_spec_rejects_array_as_scalar(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    field: str,
    bad_value: object,
) -> None:
    """四組 sequence 欄位必須是 JSON array，不得把 scalar/object 當成可迭代設定。"""

    document = _report_document(aggregate_spec, **{field: bad_value})
    path = tmp_path / f"bad-array-{field}.json"
    _write_json(path, document)
    _assert_load_failure(lambda: load_report_spec(path), path)


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity", "1e999"))
def test_load_report_spec_rejects_nonfinite_json_numbers(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    token: str,
) -> None:
    """JSON NaN／Infinity 與解析後成為 infinity 的 1e999 都不得進入 typed spec。"""

    document = _report_document(aggregate_spec)
    placeholder = dict(document)
    placeholder["primary_kde_bandwidth_m"] = "__NONFINITE_TOKEN__"
    raw = _json_bytes(placeholder).replace(
        b'"__NONFINITE_TOKEN__"',
        token.encode("ascii"),
        1,
    )
    path = tmp_path / f"nonfinite-{token.replace('-', 'negative-')}.json"
    path.write_bytes(raw)
    _assert_load_failure(lambda: load_report_spec(path), path)


@pytest.mark.parametrize("raw", (b"[1, 2, 3]", b"null", b"true", b"not-json"))
def test_load_report_spec_rejects_invalid_root_or_json_bytes(
    tmp_path: Path,
    raw: bytes,
) -> None:
    """root 必須是 object，語法與嚴格 UTF-8 失敗也不得將 path 帶進公開錯誤。"""

    path = tmp_path / "invalid-root.json"
    path.write_bytes(raw)
    _assert_load_failure(lambda: load_report_spec(path), path)


def test_load_report_spec_rejects_strict_utf8_failure(tmp_path: Path) -> None:
    """非 UTF-8 bytes 必須在 JSON schema 驗證前拒絕。"""

    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(b"{\xff}")
    _assert_load_failure(lambda: load_report_spec(path), path)


@pytest.mark.parametrize("kind", ("symlink", "broken-symlink", "directory", "missing"))
def test_load_report_spec_requires_existing_non_symlink_regular_file(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    kind: str,
) -> None:
    """loader 只接受既有普通檔；symlink、broken symlink、directory、missing 都拒絕。"""

    valid = tmp_path / "valid.json"
    _write_json(valid, _report_document(aggregate_spec))
    if kind == "symlink":
        path = tmp_path / "link.json"
        os.symlink(valid, path)
    elif kind == "broken-symlink":
        path = tmp_path / "broken.json"
        os.symlink(tmp_path / "not-there.json", path)
    elif kind == "directory":
        path = tmp_path / "directory.json"
        path.mkdir()
    else:
        path = tmp_path / "missing.json"
    _assert_load_failure(lambda: load_report_spec(path), path)


def test_load_report_spec_source_hash_changes_but_canonical_hash_does_not(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
) -> None:
    """同語意不同排版／key 順序的 input，raw source hash 不同但 canonical hash 相同。"""

    document = _report_document(aggregate_spec)
    first_path = tmp_path / "pretty.json"
    second_path = tmp_path / "compact-reordered.json"
    _write_json(first_path, document, indent=2, sort_keys=False)
    reordered = {key: document[key] for key in reversed(tuple(document))}
    second_path.write_bytes(_json_bytes(reordered, indent=None, sort_keys=True))

    first = load_report_spec(first_path)
    second = load_report_spec(second_path)
    assert first.source_sha256 != second.source_sha256
    assert first.canonical_sha256 == second.canonical_sha256
    assert first.vertical_depth_bin_edges_m == second.vertical_depth_bin_edges_m


def test_validate_report_spec_against_aggregate_spec_accepts_numeric_equal_binding_without_mutation(
    aggregate_spec: AggregateSpec,
) -> None:
    """validator 允許 100 與 100.0 的 numeric equality，但不改動兩個 immutable input。"""

    report_spec = ReportSpec(
        **_report_kwargs(
            aggregate_spec,
            primary_kde_bandwidth_m=100.0,
        )
    )
    report_before = report_spec.to_dict()
    aggregate_before = aggregate_spec.to_dict()
    assert validate_report_spec_against_aggregate_spec(report_spec, aggregate_spec) is None
    assert report_spec.to_dict() == report_before
    assert aggregate_spec.to_dict() == aggregate_before


@pytest.mark.parametrize(
    "field, value",
    (
        ("run_id", "other-run"),
        ("aggregate_spec_canonical_sha256", _THIRD_HASH),
        ("primary_kde_bandwidth_m", 150),
    ),
)
def test_validate_report_spec_against_aggregate_spec_rejects_each_binding_tamper(
    aggregate_spec: AggregateSpec,
    field: str,
    value: object,
) -> None:
    """run_id、aggregate canonical hash、primary bandwidth 各自脫綁時必須獨立失敗。"""

    report_spec = ReportSpec(**_report_kwargs(aggregate_spec, **{field: value}))
    with pytest.raises(ValueError):
        validate_report_spec_against_aggregate_spec(report_spec, aggregate_spec)


def test_validate_report_spec_against_aggregate_spec_requires_exact_classes(
    aggregate_spec: AggregateSpec,
) -> None:
    """binding API 不接受 subclass 或 duck object，避免未驗證物件偽裝成 spec。"""

    report_spec = ReportSpec(**_report_kwargs(aggregate_spec))

    class ReportSpecSubclass(ReportSpec):
        """只供測試 exact-class gate 的 synthetic subclass。"""

    class AggregateSpecSubclass(AggregateSpec):
        """只供測試 exact-class gate 的 synthetic subclass。"""

    report_subclass = ReportSpecSubclass(**_report_spec_kwargs_from_instance(report_spec))
    aggregate_subclass = AggregateSpecSubclass(
        **_aggregate_spec_kwargs_from_instance(aggregate_spec)
    )
    with pytest.raises(TypeError):
        validate_report_spec_against_aggregate_spec(report_subclass, aggregate_spec)
    with pytest.raises(TypeError):
        validate_report_spec_against_aggregate_spec(report_spec, aggregate_subclass)
    with pytest.raises(TypeError):
        validate_report_spec_against_aggregate_spec(object(), aggregate_spec)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_report_spec_against_aggregate_spec(report_spec, object())  # type: ignore[arg-type]


def test_write_report_spec_atomically_publishes_valid_spec_without_hashes_in_input(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
) -> None:
    """合法 writer 應回傳 Path、原子發布、可 load/validate，且輸入 JSON 不含衍生 hash。"""

    parent = tmp_path / "report-spec-output"
    parent.mkdir()
    target = parent / "report_spec.json"
    returned = _write_report_spec(target, aggregate_spec)

    assert returned == target
    assert isinstance(returned, Path)
    assert stat.S_ISREG(target.lstat().st_mode)
    assert not target.is_symlink()
    raw = target.read_bytes()
    document = json.loads(raw)
    assert set(document) == _REPORT_ROOT_KEYS
    assert "source_sha256" not in document
    assert "canonical_sha256" not in document
    loaded = load_report_spec(target)
    validate_report_spec_against_aggregate_spec(loaded, aggregate_spec)
    assert loaded.vertical_depth_bin_edges_m == _VERTICAL_DEPTH_EDGES
    assert loaded.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert loaded.canonical_sha256 == loaded.source_sha256
    assert _partial_paths(target) == []


@pytest.mark.parametrize("kind", ("file", "dir", "symlink", "broken-symlink"))
def test_write_report_spec_rejects_existing_target_without_overwrite_or_cleanup(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    kind: str,
) -> None:
    """既有 ordinary file/dir/symlink/broken symlink 都不得被 writer 覆寫或刪除。"""

    parent = tmp_path / f"existing-{kind}"
    parent.mkdir()
    target = parent / "report_spec.json"
    old_bytes, old_link = _prepare_existing_target(target, kind)

    with pytest.raises(FileExistsError):
        _write_report_spec(target, aggregate_spec)
    assert _partial_paths(target) == []
    if kind == "file":
        assert target.read_bytes() == old_bytes
    elif kind == "dir":
        assert (target / "marker").read_bytes() == b"protected-directory"
    else:
        assert target.is_symlink()
        assert os.readlink(target) == old_link


@pytest.mark.parametrize("parent_kind", ("missing", "symlink", "file"))
def test_write_report_spec_rejects_unsafe_parent_without_mkdir(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    parent_kind: str,
) -> None:
    """target parent 必須預先存在且為普通 non-symlink directory，writer 不得 mkdir。"""

    if parent_kind == "missing":
        parent = tmp_path / "not-created"
    elif parent_kind == "file":
        parent = tmp_path / "parent-file"
        parent.write_bytes(b"parent-file")
    else:
        referent = tmp_path / "outside-parent"
        referent.mkdir()
        parent = tmp_path / "parent-link"
        os.symlink(referent, parent)
    target = parent / "report_spec.json"

    _assert_write_failure(lambda: _write_report_spec(target, aggregate_spec), tmp_path)
    assert not target.exists()
    assert not (parent / "report_spec.json").exists() if parent.exists() else True
    if parent_kind == "missing":
        assert not parent.exists()


@pytest.mark.parametrize(
    "field, value",
    (
        ("primary_kde_bandwidth_m", 150),
        ("minimum_kde_raw_count", 0),
        ("low_sample_min_member_count", True),
        ("vertical_depth_bin_edges_m", (0.0, 5.0, 5.0)),
        ("representative_trajectory_count_per_site", 9),
        ("representative_selection_seed", 2**128),
    ),
)
def test_write_report_spec_rejects_invalid_input_before_partial_creation(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    field: str,
    value: object,
) -> None:
    """每個 writer input tamper 都只改一欄，且在 partial 建立前固定失敗。"""

    parent = tmp_path / "invalid-input"
    parent.mkdir()
    target = parent / "report_spec.json"
    _assert_write_failure(
        lambda: _write_report_spec(target, aggregate_spec, **{field: value}),
        tmp_path,
    )
    assert not target.exists()
    assert _partial_paths(target) == []


def test_write_report_spec_rejects_nonexact_aggregate_spec_before_partial_creation(
    tmp_path: Path,
) -> None:
    """writer 的 aggregate binding 只接受 exact AggregateSpec，不能以 duck object 取代。"""

    parent = tmp_path / "invalid-aggregate"
    parent.mkdir()
    target = parent / "report_spec.json"
    _assert_write_failure(
        lambda: _write_report_spec(target, object()),  # type: ignore[arg-type]
        tmp_path,
    )
    assert _partial_paths(target) == []


@pytest.mark.parametrize("failure_point", ("load", "validate", "replace"))
def test_write_report_spec_failure_cleans_only_owned_partial_and_preserves_existing_bytes(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """load/validate/replace 失敗只清本次 partial，不能碰 unrelated partial 或既有 bytes。"""

    parent = tmp_path / f"failure-{failure_point}"
    parent.mkdir()
    target = parent / "report_spec.json"
    protected_final = parent / "existing-final.json"
    protected_bytes = b"existing-final-must-survive"
    protected_final.write_bytes(protected_bytes)
    manual_partial = parent / f".{target.name}.partial-manual"
    manual_bytes = b"unrelated-partial-must-survive"
    manual_partial.write_bytes(manual_bytes)
    called = {"value": False}

    if failure_point == "load":

        def fail_load(path: object) -> object:
            """在 production 已寫入 partial 後模擬 loader 失敗。"""

            del path
            called["value"] = True
            raise OSError("synthetic load failure")

        monkeypatch.setattr(report_spec_module, "load_report_spec", fail_load)
    elif failure_point == "validate":

        def fail_validate(report: object, aggregate: object) -> None:
            """在 partial load 成功後模擬 aggregate binding 失敗。"""

            del report, aggregate
            called["value"] = True
            raise ValueError("synthetic validate failure")

        monkeypatch.setattr(
            report_spec_module,
            "validate_report_spec_against_aggregate_spec",
            fail_validate,
        )
    else:

        def fail_replace(source: object, destination: object) -> None:
            """在第二次 target absence check 後模擬 rename I/O 失敗。"""

            del source, destination
            called["value"] = True
            raise OSError("synthetic replace failure")

        monkeypatch.setattr(report_spec_module.os, "replace", fail_replace)

    _assert_write_failure(lambda: _write_report_spec(target, aggregate_spec), tmp_path)
    assert called["value"] is True
    assert not target.exists()
    assert protected_final.read_bytes() == protected_bytes
    assert manual_partial.read_bytes() == manual_bytes
    assert manual_partial in _partial_paths(target)
    assert [path for path in _partial_paths(target) if path != manual_partial] == []


def test_write_report_spec_second_target_absence_check_fails_closed_on_race(
    tmp_path: Path,
    aggregate_spec: AggregateSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """partial 驗證後若 target 被他者搶先建立，writer 必須保留 race target 並清理自己 partial。"""

    parent = tmp_path / "target-race"
    parent.mkdir()
    target = parent / "report_spec.json"
    race_bytes = b"race-created-final"
    original_require_absent = report_spec_module._require_target_absent
    call_count = {"value": 0}

    def race_target(path: Path) -> None:
        """只在第二次 absence check 建立 target，模擬 rename 前的競爭者。"""

        call_count["value"] += 1
        if call_count["value"] == 2:
            path.write_bytes(race_bytes)
        original_require_absent(path)

    monkeypatch.setattr(report_spec_module, "_require_target_absent", race_target)
    with pytest.raises(FileExistsError):
        _write_report_spec(target, aggregate_spec)
    assert call_count["value"] == 2
    assert target.read_bytes() == race_bytes
    assert _partial_paths(target) == []
