"""聚合規格 JSON schema、正規化、雜湊與防禦性邊界測試。

本測試檔只透過 ``lagrangian_backtracking.aggregate_spec`` 的公開常數、資料類別、
``to_dict`` 與 loader 驗證契約；測試資料是公尺制的最小合法 fixture，不代表正式研究
資料的實際空間範圍。所有檔案均建立在 pytest 的暫存目錄，避免把測試輸出寫入專案。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from lagrangian_backtracking.aggregate_spec import (
    AGGREGATE_SPEC_SCHEMA_VERSION,
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
    load_aggregate_spec,
)

_ROOT_KEYS = {
    "schema_version",
    "run_id",
    "grid_cell_size_m",
    "site_grids",
    "site_metric_crs",
    "site_boundary_segment_ids",
    "boundary_bin_size_m",
    "boundary_segment_lengths_m",
    "kde_bandwidths_m",
    "hdr_levels",
    "age_bin_edges_seconds",
    "bootstrap_replicates",
    "bootstrap_confidence_level",
    "bootstrap_seed",
    "denominator_policy",
}
_HASH_KEYS = {"source_sha256", "canonical_sha256"}
_DENOMINATOR_POLICY = "exclude_data_gap_and_numerical_failure_v1"


def _valid_payload() -> dict[str, Any]:
    """建立題目指定的最小合法 JSON fixture，並在每次呼叫時回傳獨立副本。

    ``site_grids`` 的 key 是站點 slug，站點內容保存公尺制 x/y 軸最小與最大邊界；
    ``site_metric_crs`` 逐站保存固定本地區域方位等距投影（AEQD）的中心、單位
    與 x/y 軸序，
    ``site_boundary_segment_ids`` 分別保存各站 local／outer 所引用的線段 ID，
    ``boundary_segment_lengths_m`` 則以線段 slug 對應其公尺長度。其餘數值欄位描述
    聚合格網、邊界分箱、KDE 頻寬、HDR 門檻、共用 travel-age 秒數邊界與 bootstrap
    統計設定；這是最小可重建 fixture，不代表正式研究資料的投影或時間範圍。
    """

    return {
        "schema_version": "1.0.0",
        "run_id": "run-001",
        "grid_cell_size_m": 100,
        "site_grids": {
            "gongliao": {
                "x_min_m": 0,
                "x_max_m": 1000,
                "y_min_m": 0,
                "y_max_m": 2000,
            }
        },
        "site_metric_crs": {
            "gongliao": {
                "projection_method": "azimuthal_equidistant_wgs84",
                "center_lon_deg": 121.9,
                "center_lat_deg": 25.0,
                "linear_unit": "m",
                "axis_order": "x_east_y_north",
            }
        },
        "site_boundary_segment_ids": {
            "gongliao": {
                "local": ["gongliao-local"],
                "outer": ["gongliao-local"],
            }
        },
        "boundary_bin_size_m": 50,
        "boundary_segment_lengths_m": {"gongliao-local": 1000},
        "kde_bandwidths_m": [100, 200, 400],
        "hdr_levels": [0.5, 0.75, 0.9],
        "age_bin_edges_seconds": [0, 3600, 21600, 86400],
        "bootstrap_replicates": 1000,
        "bootstrap_confidence_level": 0.95,
        "bootstrap_seed": 123,
        "denominator_policy": _DENOMINATOR_POLICY,
    }


def _guishan_grid() -> dict[str, int]:
    """建立第二站且與 100 公尺 cell 對齊的矩形格網，供跨站線段契約測試使用。"""

    return {
        "x_min_m": 1000,
        "x_max_m": 2000,
        "y_min_m": 0,
        "y_max_m": 2000,
    }


def _guishan_metric_crs() -> dict[str, Any]:
    """建立第二站的固定本地區域方位等距（AEQD）設定，刻意與第一站共用投影中心。

    A 區允許不同 site 使用相同中心；這裡保留獨立的 dict 副本，確認 schema
    要求的是每站明示一份可重建設定，而不是透過 mapping alias 隱含繼承。
    """

    return {
        "projection_method": "azimuthal_equidistant_wgs84",
        "center_lon_deg": 121.9,
        "center_lat_deg": 25.0,
        "linear_unit": "m",
        "axis_order": "x_east_y_north",
    }


def _write_json(path: Path, payload: Any, *, indent: int | None = 2, sort_keys: bool = False) -> None:
    """以 UTF-8 寫入測試 JSON；保留非有限數值選項供 parser rejection 測試使用。"""

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def _write_raw(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """直接寫入指定 bytes 編碼的測試內容，以涵蓋 JSON 解碼前的檔案邊界。"""

    path.write_bytes(content.encode(encoding))


def _replace_value(payload: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    """沿著測試 payload 的巢狀 key 路徑替換單一欄位，不改變其他契約條件。"""

    cursor: Any = payload
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _assert_value_error_without_path(operation: Callable[[], Any], path: Path) -> None:
    """確認非法輸入統一回傳 ValueError，且錯誤訊息不洩漏暫存目錄絕對路徑。"""

    with pytest.raises(ValueError) as exc_info:
        operation()
    assert str(path) not in str(exc_info.value)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """依 loader 正規化後的 JSON 語意計算預期 canonical hash。

    age edges 在 ``AggregateSpec`` 建構時固定為 float tuple，所以測試預期也
    先將該欄位轉成 float；其餘欄位保留輸入數值型別，對應目前 schema 的
    to_dict 正規化契約。最後使用排序 key、緊湊分隔符與 UTF-8 bytes，隔離
    JSON 排版與 object key 順序造成的差異。
    """

    canonical_payload = dict(payload)
    canonical_payload["age_bin_edges_seconds"] = [
        float(edge) for edge in payload["age_bin_edges_seconds"]
    ]

    canonical = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _input_fields(value: dict[str, Any]) -> dict[str, Any]:
    """從 loader 的完整輸出快照取出可重新作為輸入的 schema 欄位。"""

    return {key: value[key] for key in _ROOT_KEYS}


def test_loads_valid_fixture_into_typed_aggregate_spec(tmp_path: Path) -> None:
    """合法 fixture 應載入為 AggregateSpec，且保留所有公尺制與統計設定的語意值。"""

    payload = _valid_payload()
    path = tmp_path / "aggregate.json"
    _write_json(path, payload)

    spec = load_aggregate_spec(path)
    site = spec.site_grids["gongliao"]
    metric_crs = spec.site_metric_crs["gongliao"]
    site_segments = spec.site_boundary_segment_ids["gongliao"]

    assert AGGREGATE_SPEC_SCHEMA_VERSION == "1.0.0"
    assert isinstance(spec, AggregateSpec)
    assert isinstance(site, SiteGridSpec)
    assert isinstance(metric_crs, SiteMetricCRSSpec)
    assert isinstance(site_segments, SiteBoundarySegments)
    assert spec.schema_version == "1.0.0"
    assert spec.run_id == "run-001"
    assert spec.grid_cell_size_m == 100
    assert (site.x_min_m, site.x_max_m, site.y_min_m, site.y_max_m) == (0.0, 1000.0, 0.0, 2000.0)
    assert metric_crs.projection_method == "azimuthal_equidistant_wgs84"
    assert metric_crs.center_lon_deg == 121.9
    assert metric_crs.center_lat_deg == 25.0
    assert metric_crs.linear_unit == "m"
    assert metric_crs.axis_order == "x_east_y_north"
    assert site_segments.local == ("gongliao-local",)
    assert site_segments.outer == ("gongliao-local",)
    assert spec.boundary_bin_size_m == 50
    assert spec.boundary_segment_lengths_m["gongliao-local"] == 1000.0
    assert spec.kde_bandwidths_m == (100.0, 200.0, 400.0)
    assert spec.hdr_levels == (0.5, 0.75, 0.9)
    assert spec.age_bin_edges_seconds == (0.0, 3600.0, 21600.0, 86400.0)
    assert all(type(edge) is float for edge in spec.age_bin_edges_seconds)
    assert spec.bootstrap_replicates == 1000
    assert spec.bootstrap_confidence_level == 0.95
    assert spec.bootstrap_seed == 123
    assert spec.denominator_policy == _DENOMINATOR_POLICY


def test_to_dict_returns_exact_normalized_json_shape(tmp_path: Path) -> None:
    """to_dict 應把 tuple 與唯讀 mapping 正規化為可序列化的設定與 provenance 快照。"""

    payload = _valid_payload()
    path = tmp_path / "aggregate.json"
    _write_json(path, payload)

    normalized = load_aggregate_spec(path).to_dict()

    assert set(normalized) == _ROOT_KEYS | _HASH_KEYS
    assert _input_fields(normalized) == payload
    assert normalized["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert normalized["canonical_sha256"] == _canonical_sha256(payload)
    assert type(normalized["site_grids"]) is dict
    assert type(normalized["site_metric_crs"]) is dict
    assert type(normalized["site_boundary_segment_ids"]) is dict
    assert type(normalized["boundary_segment_lengths_m"]) is dict
    assert type(normalized["kde_bandwidths_m"]) is list
    assert type(normalized["hdr_levels"]) is list
    assert type(normalized["age_bin_edges_seconds"]) is list
    assert all(type(edge) is float for edge in normalized["age_bin_edges_seconds"])
    assert type(normalized["site_grids"]["gongliao"]) is dict
    assert type(normalized["site_metric_crs"]["gongliao"]) is dict
    assert set(normalized["site_metric_crs"]["gongliao"]) == {
        "projection_method",
        "center_lon_deg",
        "center_lat_deg",
        "linear_unit",
        "axis_order",
    }
    assert type(normalized["site_boundary_segment_ids"]["gongliao"]) is dict
    assert type(normalized["site_boundary_segment_ids"]["gongliao"]["local"]) is list
    assert type(normalized["site_boundary_segment_ids"]["gongliao"]["outer"]) is list


def test_source_and_canonical_sha256_are_reproducible(tmp_path: Path) -> None:
    """相同 bytes 應得到可重現 source hash，而 canonical hash 應符合固定 JSON 編碼規則。"""

    payload = _valid_payload()
    first_path = tmp_path / "first.json"
    _write_json(first_path, payload, indent=2, sort_keys=False)
    first = load_aggregate_spec(first_path)
    repeated = load_aggregate_spec(first_path)

    assert first.source_sha256 == hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert first.canonical_sha256 == _canonical_sha256(payload)
    assert first.source_sha256 == repeated.source_sha256
    assert first.canonical_sha256 == repeated.canonical_sha256


def test_canonical_sha256_ignores_formatting_and_key_order(tmp_path: Path) -> None:
    """只改變排版與各層 object key 順序時，source hash 應改變而 canonical hash 不變。"""

    payload = _valid_payload()
    first_path = tmp_path / "formatted.json"
    second_path = tmp_path / "reordered.json"
    _write_json(first_path, payload, indent=2, sort_keys=False)

    reordered = {
        key: value
        for key, value in reversed(list(payload.items()))
    }
    reordered["site_grids"] = {
        "gongliao": {
            key: value
            for key, value in reversed(list(payload["site_grids"]["gongliao"].items()))
        }
    }
    reordered["site_metric_crs"] = {
        "gongliao": {
            key: value
            for key, value in reversed(
                list(payload["site_metric_crs"]["gongliao"].items())
            )
        }
    }
    reordered["site_boundary_segment_ids"] = {
        "gongliao": {
            key: value
            for key, value in reversed(
                list(payload["site_boundary_segment_ids"]["gongliao"].items())
            )
        }
    }
    _write_json(second_path, reordered, indent=None, sort_keys=False)

    first = load_aggregate_spec(first_path)
    second = load_aggregate_spec(second_path)

    assert first.source_sha256 != second.source_sha256
    assert first.canonical_sha256 == second.canonical_sha256
    assert _input_fields(first.to_dict()) == _input_fields(second.to_dict())


def test_site_and_segment_mappings_are_read_only_and_defensive(tmp_path: Path) -> None:
    """回傳 mapping 不可直接改寫，to_dict 的巢狀副本也不可回頭污染已載入規格。"""

    payload = _valid_payload()
    path = tmp_path / "aggregate.json"
    _write_json(path, payload)
    spec = load_aggregate_spec(path)

    with pytest.raises(TypeError):
        spec.site_grids["other"] = spec.site_grids["gongliao"]
    with pytest.raises(TypeError):
        spec.site_metric_crs["other"] = spec.site_metric_crs["gongliao"]
    with pytest.raises(TypeError):
        spec.boundary_segment_lengths_m["gongliao-local"] = 999.0

    exported = spec.to_dict()
    exported["site_grids"]["gongliao"]["x_min_m"] = 999.0
    exported["site_metric_crs"]["gongliao"]["center_lon_deg"] = 0.0
    exported["boundary_segment_lengths_m"]["gongliao-local"] = 999.0
    exported["age_bin_edges_seconds"][0] = 999.0

    assert spec.site_grids["gongliao"].x_min_m == 0.0
    assert spec.site_metric_crs["gongliao"].center_lon_deg == 121.9
    assert spec.boundary_segment_lengths_m["gongliao-local"] == 1000.0
    assert spec.age_bin_edges_seconds[0] == 0.0
    assert _input_fields(spec.to_dict()) == payload


def test_site_metric_crs_mapping_is_read_only_and_value_is_frozen(
    tmp_path: Path,
) -> None:
    """每站 metric CRS mapping 與其 frozen value 都不可在載入後改寫。"""

    payload = _valid_payload()
    path = tmp_path / "metric-crs-immutable.json"
    _write_json(path, payload)
    spec = load_aggregate_spec(path)
    metric_crs = spec.site_metric_crs["gongliao"]

    assert type(spec.site_metric_crs) is MappingProxyType
    with pytest.raises(TypeError):
        spec.site_metric_crs["other"] = metric_crs
    with pytest.raises(FrozenInstanceError):
        metric_crs.center_lat_deg = 0.0

    exported = spec.to_dict()
    exported["site_metric_crs"]["gongliao"]["axis_order"] = "y_north_x_east"
    assert spec.site_metric_crs["gongliao"].axis_order == "x_east_y_north"


@pytest.mark.parametrize(
    "invalid_site_value",
    [
        {
            "projection_method": "azimuthal_equidistant_wgs84",
            "center_lon_deg": 121.9,
            "center_lat_deg": 25.0,
            "linear_unit": "m",
        },
        {
            "projection_method": "azimuthal_equidistant_wgs84",
            "center_lon_deg": 121.9,
            "center_lat_deg": 25.0,
            "linear_unit": "m",
            "axis_order": "x_east_y_north",
            "unexpected": 1,
        },
    ],
)
def test_site_metric_crs_value_requires_exact_keys(
    tmp_path: Path, invalid_site_value: dict[str, Any]
) -> None:
    """每站投影描述必須恰含五個固定欄位，避免漏存重建所需的軸或單位。"""

    payload = _valid_payload()
    payload["site_metric_crs"]["gongliao"] = invalid_site_value
    path = tmp_path / "invalid-metric-crs-keys.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_value", [None, True, [], "metric-crs"])
def test_site_metric_crs_root_requires_json_object(
    tmp_path: Path, invalid_value: Any
) -> None:
    """site_metric_crs 必須是 JSON object，不可由 scalar 或 array 猜測站點投影。"""

    payload = _valid_payload()
    payload["site_metric_crs"] = invalid_value
    path = tmp_path / "invalid-metric-crs-root.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("mismatch", ["grid-only", "crs-only"])
def test_site_metric_crs_site_set_must_equal_site_grid_set(
    tmp_path: Path, mismatch: str
) -> None:
    """每個 site 都必須同時有格網與投影基準，不可缺站或多站。"""

    payload = _valid_payload()
    if mismatch == "grid-only":
        payload["site_grids"]["guishan"] = _guishan_grid()
    else:
        payload["site_metric_crs"]["guishan"] = _guishan_metric_crs()
    path = tmp_path / "metric-crs-site-set-mismatch.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("projection_method", "aeqd"),
        ("projection_method", True),
        ("linear_unit", "degree"),
        ("linear_unit", True),
        ("axis_order", "lon_lat"),
        ("axis_order", True),
    ],
)
def test_site_metric_crs_fixed_strings_are_exact(
    tmp_path: Path, field: str, invalid_value: Any
) -> None:
    """投影方法、長度單位與軸序只能採用已登錄字串，避免下游解讀分歧。"""

    payload = _valid_payload()
    payload["site_metric_crs"]["gongliao"][field] = invalid_value
    path = tmp_path / "invalid-metric-crs-string.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        "121.9",
        None,
        math.nan,
        math.inf,
        -math.inf,
        -180.000001,
        180.000001,
    ],
)
def test_site_metric_crs_longitude_requires_finite_number_inclusive_range(
    tmp_path: Path, invalid_value: Any
) -> None:
    """經度必須是有限原生數字，且允許 -180 與 180 兩個邊界。"""

    payload = _valid_payload()
    payload["site_metric_crs"]["gongliao"]["center_lon_deg"] = invalid_value
    path = tmp_path / "invalid-metric-crs-longitude.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        "25.0",
        None,
        math.nan,
        math.inf,
        -math.inf,
        -90,
        90,
        -90.000001,
        90.000001,
    ],
)
def test_site_metric_crs_latitude_requires_finite_number_open_range(
    tmp_path: Path, invalid_value: Any
) -> None:
    """緯度必須是有限原生數字，且極點 -90 與 90 不在允許的開區間內。"""

    payload = _valid_payload()
    payload["site_metric_crs"]["gongliao"]["center_lat_deg"] = invalid_value
    path = tmp_path / "invalid-metric-crs-latitude.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    ("field", "valid_value"),
    [
        ("center_lon_deg", -180),
        ("center_lon_deg", 180),
        ("center_lat_deg", -89.999999),
        ("center_lat_deg", 89.999999),
    ],
)
def test_site_metric_crs_valid_boundary_values_are_accepted(
    tmp_path: Path, field: str, valid_value: int | float
) -> None:
    """投影中心接受經度閉邊界與接近極點但仍在緯度開區間內的合法值。"""

    payload = _valid_payload()
    payload["site_metric_crs"]["gongliao"][field] = valid_value
    path = tmp_path / "valid-metric-crs-boundary.json"
    _write_json(path, payload)

    spec = load_aggregate_spec(path)
    assert getattr(spec.site_metric_crs["gongliao"], field) == valid_value


def test_age_bin_edges_are_float_tuple_and_last_edge_is_preserved(
    tmp_path: Path,
) -> None:
    """年齡邊界會固定成 float tuple，最後邊界保留供最後閉區間重建。"""

    payload = _valid_payload()
    payload["age_bin_edges_seconds"] = [0, 1, 2.5]
    path = tmp_path / "valid-age-edges.json"
    _write_json(path, payload)

    spec = load_aggregate_spec(path)

    assert spec.age_bin_edges_seconds == (0.0, 1.0, 2.5)
    assert type(spec.age_bin_edges_seconds) is tuple
    assert all(type(edge) is float for edge in spec.age_bin_edges_seconds)
    assert spec.to_dict()["age_bin_edges_seconds"] == [0.0, 1.0, 2.5]


@pytest.mark.parametrize("invalid_value", [None, True, "0,1", {"edges": [0, 1]}, 1])
def test_age_bin_edges_require_json_array(tmp_path: Path, invalid_value: Any) -> None:
    """共用 travel-age 軸必須是明確的 JSON array，不可由其他容器猜測。"""

    payload = _valid_payload()
    payload["age_bin_edges_seconds"] = invalid_value
    path = tmp_path / "invalid-age-edges-type.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_edges", [[], [0]])
def test_age_bin_edges_require_at_least_two_points(
    tmp_path: Path, invalid_edges: list[Any]
) -> None:
    """沒有至少兩個邊界就無法形成任何 travel-age histogram 箱。"""

    payload = _valid_payload()
    payload["age_bin_edges_seconds"] = invalid_edges
    path = tmp_path / "invalid-age-edges-count.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_edges",
    [[True, 1], [0, True], [0, "1"], [0, None], [0, [1]]],
)
def test_age_bin_edges_reject_non_numeric_or_bool_values(
    tmp_path: Path, invalid_edges: list[Any]
) -> None:
    """每個年齡邊界都必須是原生有限 int/float，bool 與文字不可冒充秒數。"""

    payload = _valid_payload()
    payload["age_bin_edges_seconds"] = invalid_edges
    path = tmp_path / "invalid-age-edges-number.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_edges",
    [[0, math.nan], [0, math.inf], [0, -math.inf]],
)
def test_age_bin_edges_reject_nonfinite_values(
    tmp_path: Path, invalid_edges: list[float]
) -> None:
    """年齡軸不接受 NaN 或正負 Infinity，避免直方圖索引不可重建。"""

    payload = _valid_payload()
    payload["age_bin_edges_seconds"] = invalid_edges
    path = tmp_path / "invalid-age-edges-nonfinite.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_edges", [[1, 2], [-1, 0], [0.1, 1]])
def test_age_bin_edges_must_start_at_exact_zero(
    tmp_path: Path, invalid_edges: list[int | float]
) -> None:
    """travel age 的共同時間原點必須精確是 0 秒，不可任意平移。"""

    payload = _valid_payload()
    payload["age_bin_edges_seconds"] = invalid_edges
    path = tmp_path / "invalid-age-edges-origin.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_edges",
    [[0, 1, 1], [0, 2, 1], [0, 2, 1.5]],
)
def test_age_bin_edges_must_be_strictly_increasing(
    tmp_path: Path, invalid_edges: list[int | float]
) -> None:
    """重複或倒退的邊界會造成重疊／反向箱，必須在 schema 層拒絕。"""

    payload = _valid_payload()
    payload["age_bin_edges_seconds"] = invalid_edges
    path = tmp_path / "invalid-age-edges-order.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


def test_site_boundary_mapping_and_tuples_are_read_only_and_defensive(tmp_path: Path) -> None:
    """站點線段 mapping 與 local／outer tuple 不可改寫，輸出副本也不得污染原規格。"""

    payload = _valid_payload()
    path = tmp_path / "aggregate.json"
    _write_json(path, payload)
    spec = load_aggregate_spec(path)
    site_segments = spec.site_boundary_segment_ids["gongliao"]

    assert isinstance(site_segments, SiteBoundarySegments)
    assert type(site_segments.local) is tuple
    assert type(site_segments.outer) is tuple
    with pytest.raises(TypeError):
        spec.site_boundary_segment_ids["other"] = site_segments
    with pytest.raises(TypeError):
        site_segments.local[0] = "other-segment"
    with pytest.raises(TypeError):
        site_segments.outer[0] = "other-segment"

    exported = spec.to_dict()
    exported["site_boundary_segment_ids"]["gongliao"]["local"].append("other-segment")
    exported["site_boundary_segment_ids"]["gongliao"]["outer"][0] = "other-segment"

    assert site_segments.local == ("gongliao-local",)
    assert site_segments.outer == ("gongliao-local",)
    assert _input_fields(spec.to_dict()) == payload


@pytest.mark.parametrize("mismatch", ["grid-only", "boundary-only"])
def test_site_boundary_site_set_must_equal_site_grid_set(tmp_path: Path, mismatch: str) -> None:
    """站點線段索引與站點格網必須涵蓋完全相同的 site set，不可缺站或多站。"""

    payload = _valid_payload()
    if mismatch == "grid-only":
        payload["site_grids"]["guishan"] = _guishan_grid()
        payload["site_metric_crs"]["guishan"] = _guishan_metric_crs()
    else:
        payload["site_boundary_segment_ids"]["guishan"] = {
            "local": ["gongliao-local"],
            "outer": ["gongliao-local"],
        }
    path = tmp_path / "site-set-mismatch.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_site_value",
    [
        {"local": ["gongliao-local"]},
        {"outer": ["gongliao-local"]},
        {
            "local": ["gongliao-local"],
            "outer": ["gongliao-local"],
            "unexpected": ["gongliao-local"],
        },
    ],
)
def test_site_boundary_value_requires_exact_local_and_outer_keys(
    tmp_path: Path, invalid_site_value: dict[str, Any]
) -> None:
    """每站線段設定必須恰含 local 與 outer，缺少或加入其他欄位都不得載入。"""

    payload = _valid_payload()
    payload["site_boundary_segment_ids"]["gongliao"] = invalid_site_value
    path = tmp_path / "invalid-site-boundary-keys.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("boundary_kind", ["local", "outer"])
@pytest.mark.parametrize("invalid_value", [None, True, "gongliao-local", {"id": "gongliao-local"}])
def test_site_boundary_local_and_outer_require_json_arrays(
    tmp_path: Path, boundary_kind: str, invalid_value: Any
) -> None:
    """local 與 outer 都必須是 JSON array，不可由 null、bool、文字或 object 猜測。"""

    payload = _valid_payload()
    payload["site_boundary_segment_ids"]["gongliao"][boundary_kind] = invalid_value
    path = tmp_path / "invalid-site-boundary-array.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("boundary_kind", ["local", "outer"])
def test_site_boundary_local_and_outer_reject_empty_arrays(
    tmp_path: Path, boundary_kind: str
) -> None:
    """每站 local 與 outer 各自至少要引用一條線段，空陣列沒有可聚合的邊界語意。"""

    payload = _valid_payload()
    payload["site_boundary_segment_ids"]["gongliao"][boundary_kind] = []
    path = tmp_path / "empty-site-boundary-array.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("boundary_kind", ["local", "outer"])
def test_site_boundary_local_and_outer_reject_duplicate_ids(
    tmp_path: Path, boundary_kind: str
) -> None:
    """同一站同一邊界類型不可重複線段 ID，避免同一長度被重複計數。"""

    payload = _valid_payload()
    payload["site_boundary_segment_ids"]["gongliao"][boundary_kind] = [
        "gongliao-local",
        "gongliao-local",
    ]
    path = tmp_path / "duplicate-site-boundary-id.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("boundary_kind", ["local", "outer"])
@pytest.mark.parametrize("invalid_segment_id", ["", "invalid segment", "../invalid-segment"])
def test_site_boundary_local_and_outer_reject_invalid_segment_slugs(
    tmp_path: Path, boundary_kind: str, invalid_segment_id: str
) -> None:
    """local／outer 陣列內每個線段 ID 都必須是安全 slug，不得含空白或 traversal。"""

    payload = _valid_payload()
    payload["site_boundary_segment_ids"]["gongliao"][boundary_kind] = [invalid_segment_id]
    payload["boundary_segment_lengths_m"] = {invalid_segment_id: 1000}
    path = tmp_path / "invalid-site-boundary-slug.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("boundary_kind", ["local", "outer"])
def test_site_boundary_local_and_outer_reject_unknown_segments(
    tmp_path: Path, boundary_kind: str
) -> None:
    """local／outer 引用的每個線段都必須存在於公尺長度 mapping，不可留下懸空 ID。"""

    payload = _valid_payload()
    payload["site_boundary_segment_ids"]["gongliao"][boundary_kind] = ["unknown-segment"]
    path = tmp_path / "unknown-site-boundary-segment.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


def test_boundary_length_mapping_rejects_unreferenced_segment(tmp_path: Path) -> None:
    """長度 mapping 不可含任何未被站點 local 或 outer 引用的線段，避免孤立設定漂移。"""

    payload = _valid_payload()
    payload["boundary_segment_lengths_m"]["unused-segment"] = 500
    path = tmp_path / "unreferenced-boundary-segment.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


def test_boundary_segment_can_be_shared_across_sites(tmp_path: Path) -> None:
    """同一具名邊界線段可由兩站共同引用，長度 mapping 仍只需保存一份定義。"""

    payload = _valid_payload()
    payload["site_grids"]["guishan"] = _guishan_grid()
    payload["site_metric_crs"]["guishan"] = _guishan_metric_crs()
    payload["site_boundary_segment_ids"]["guishan"] = {
        "local": ["gongliao-local"],
        "outer": ["gongliao-local"],
    }
    path = tmp_path / "shared-boundary-segment.json"
    _write_json(path, payload)

    spec = load_aggregate_spec(path)

    assert set(spec.site_boundary_segment_ids) == {"gongliao", "guishan"}
    assert spec.site_boundary_segment_ids["gongliao"].local == ("gongliao-local",)
    assert spec.site_boundary_segment_ids["guishan"].outer == ("gongliao-local",)
    assert spec.boundary_segment_lengths_m == {"gongliao-local": 1000}


def test_same_segment_can_be_local_and_outer_for_one_site(tmp_path: Path) -> None:
    """同站 local 與 outer 可合法引用同一線段，兩個分類間不套用互斥限制。"""

    payload = _valid_payload()
    path = tmp_path / "same-local-outer-segment.json"
    _write_json(path, payload)

    site_segments = load_aggregate_spec(path).site_boundary_segment_ids["gongliao"]

    assert site_segments.local == ("gongliao-local",)
    assert site_segments.outer == ("gongliao-local",)


def test_missing_file_is_value_error_without_absolute_path(tmp_path: Path) -> None:
    """不存在的 loader 目標必須以不洩漏暫存絕對路徑的 ValueError fail closed。"""

    path = tmp_path / "missing.json"
    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


def test_symlink_file_is_rejected_without_absolute_path(tmp_path: Path) -> None:
    """symlink 不可繞過普通檔案邊界，避免規格內容來自未登錄的外部路徑。"""

    target = tmp_path / "target.json"
    link = tmp_path / "aggregate.json"
    _write_json(target, _valid_payload())
    link.symlink_to(target)

    _assert_value_error_without_path(lambda: load_aggregate_spec(link), link)


def test_non_utf8_file_is_rejected_without_absolute_path(tmp_path: Path) -> None:
    """不是 UTF-8 的 bytes 不得被替換字元容錯後當成合法規格。"""

    path = tmp_path / "non-utf8.json"
    path.write_bytes(b"{\xff\xfe")

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


def test_duplicate_json_key_is_rejected_without_absolute_path(tmp_path: Path) -> None:
    """JSON object 的重複欄位必須報錯，不可靜默採用最後一個值。"""

    path = tmp_path / "duplicate.json"
    raw = json.dumps(_valid_payload(), separators=(",", ":"))
    raw = raw.replace('"run_id":"run-001"', '"run_id":"run-001","run_id":"run-002"', 1)
    _write_raw(path, raw)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_json_number_is_rejected_without_absolute_path(tmp_path: Path, token: str) -> None:
    """JSON 常數與極大指數造成的 NaN／Infinity 都不得進入公尺或統計欄位。"""

    path = tmp_path / f"nonfinite-{token.replace('-', 'negative-')}.json"
    raw = json.dumps(_valid_payload(), indent=2)
    raw = raw.replace('"grid_cell_size_m": 100', f'"grid_cell_size_m": {token}', 1)
    _write_raw(path, raw)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("root", [[], "text", None, 1, True])
def test_non_object_root_is_rejected_without_absolute_path(tmp_path: Path, root: Any) -> None:
    """root 必須是 JSON object；array、scalar 與 null 不可被當成設定文件。"""

    path = tmp_path / "non-object.json"
    _write_json(path, root)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("missing_key", sorted(_ROOT_KEYS))
def test_missing_root_key_is_rejected_without_absolute_path(tmp_path: Path, missing_key: str) -> None:
    """根物件缺少任一資料契約欄位時都必須拒絕，避免使用隱含預設值。"""

    payload = _valid_payload()
    del payload[missing_key]
    path = tmp_path / f"missing-{missing_key}.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


def test_unknown_root_key_is_rejected_without_absolute_path(tmp_path: Path) -> None:
    """根物件含未登錄欄位時必須 extra-forbid，避免 schema 漂移被靜默忽略。"""

    payload = _valid_payload()
    payload["unexpected"] = 1
    path = tmp_path / "unknown-key.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("schema_version", ["0.0.0", "1.0", "1.0.1", True, 1])
def test_schema_version_must_equal_fixed_version(tmp_path: Path, schema_version: Any) -> None:
    """schema_version 只能精確等於 1.0.0，避免不同格式被誤當成相容產品。"""

    payload = _valid_payload()
    payload["schema_version"] = schema_version
    path = tmp_path / "schema-version.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    ("identity_kind", "invalid_value"),
    [
        ("run", ""),
        ("run", "run 001"),
        ("run", "run/001"),
        ("run", "../run-001"),
        ("run", "-run-001"),
        ("site", ""),
        ("site", "gongliao local"),
        ("site", "gongliao/local"),
        ("site", "../gongliao"),
        ("segment", ""),
        ("segment", "gongliao local"),
        ("segment", "gongliao/local"),
        ("segment", "../gongliao-local"),
    ],
)
def test_run_site_and_segment_ids_must_be_safe_slugs(
    tmp_path: Path, identity_kind: str, invalid_value: str
) -> None:
    """run、site 與 segment 識別碼只能是安全單一路徑 slug，不得含空白或 traversal。"""

    payload = _valid_payload()
    if identity_kind == "run":
        payload["run_id"] = invalid_value
    elif identity_kind == "site":
        payload["site_grids"] = {invalid_value: payload["site_grids"]["gongliao"]}
        payload["site_boundary_segment_ids"] = {
            invalid_value: payload["site_boundary_segment_ids"]["gongliao"]
        }
    else:
        payload["boundary_segment_lengths_m"] = {invalid_value: 1000}
        payload["site_boundary_segment_ids"]["gongliao"] = {
            "local": [invalid_value],
            "outer": [invalid_value],
        }
    path = tmp_path / f"invalid-{identity_kind}.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("grid_cell_size_m",), True),
        (("grid_cell_size_m",), "100"),
        (("grid_cell_size_m",), [100]),
        (("boundary_bin_size_m",), True),
        (("boundary_bin_size_m",), "50"),
        (("boundary_bin_size_m",), [50]),
        (("bootstrap_replicates",), True),
        (("bootstrap_replicates",), "1000"),
        (("bootstrap_replicates",), [1000]),
        (("bootstrap_confidence_level",), True),
        (("bootstrap_confidence_level",), "0.95"),
        (("bootstrap_confidence_level",), [0.95]),
        (("bootstrap_seed",), True),
        (("bootstrap_seed",), "123"),
        (("bootstrap_seed",), [123]),
        (("site_grids", "gongliao", "x_min_m"), True),
        (("site_grids", "gongliao", "x_min_m"), "0"),
        (("site_grids", "gongliao", "x_min_m"), [0]),
        (("site_grids", "gongliao", "x_max_m"), True),
        (("site_grids", "gongliao", "x_max_m"), "1000"),
        (("site_grids", "gongliao", "x_max_m"), [1000]),
        (("site_grids", "gongliao", "y_min_m"), True),
        (("site_grids", "gongliao", "y_min_m"), "0"),
        (("site_grids", "gongliao", "y_min_m"), [0]),
        (("site_grids", "gongliao", "y_max_m"), True),
        (("site_grids", "gongliao", "y_max_m"), "2000"),
        (("site_grids", "gongliao", "y_max_m"), [2000]),
        (("boundary_segment_lengths_m", "gongliao-local"), True),
        (("boundary_segment_lengths_m", "gongliao-local"), "1000"),
        (("boundary_segment_lengths_m", "gongliao-local"), [1000]),
    ],
)
def test_scalar_numeric_fields_reject_bool_string_and_list(
    tmp_path: Path, field_path: tuple[str, ...], invalid_value: Any
) -> None:
    """所有 scalar 數值欄位都必須維持 JSON number 語意，不可接受 bool、文字或 array。"""

    payload = _valid_payload()
    _replace_value(payload, field_path, invalid_value)
    path = tmp_path / "invalid-scalar.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "field_path",
    [
        ("grid_cell_size_m",),
        ("boundary_bin_size_m",),
        ("bootstrap_replicates",),
        ("bootstrap_confidence_level",),
        ("bootstrap_seed",),
        ("site_grids", "gongliao", "x_min_m"),
        ("site_grids", "gongliao", "x_max_m"),
        ("site_grids", "gongliao", "y_min_m"),
        ("site_grids", "gongliao", "y_max_m"),
        ("boundary_segment_lengths_m", "gongliao-local"),
    ],
)
@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_scalar_numeric_fields_reject_nonfinite_values(
    tmp_path: Path, field_path: tuple[str, ...], invalid_value: float
) -> None:
    """公尺邊界、格距與 bootstrap scalar 不得含 NaN 或正負 Infinity。"""

    payload = _valid_payload()
    _replace_value(payload, field_path, invalid_value)
    path = tmp_path / "invalid-nonfinite.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_cell", [0, -1, 100.5])
def test_grid_cell_size_requires_positive_integer(tmp_path: Path, invalid_cell: Any) -> None:
    """聚合格網 cell size 必須是可用於整數格索引的正整數，不能為零、負值或浮點數。"""

    payload = _valid_payload()
    payload["grid_cell_size_m"] = invalid_cell
    path = tmp_path / "invalid-cell.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("site_grids", "gongliao", "x_min_m"), 1000),
        (("site_grids", "gongliao", "x_max_m"), 0),
        (("site_grids", "gongliao", "y_min_m"), 2000),
        (("site_grids", "gongliao", "y_max_m"), 0),
    ],
)
def test_grid_bounds_require_strictly_increasing_min_and_max(
    tmp_path: Path, field_path: tuple[str, ...], invalid_value: int
) -> None:
    """x/y 最小與最大邊界必須形成正面積矩形，不接受相等或反向座標。"""

    payload = _valid_payload()
    _replace_value(payload, field_path, invalid_value)
    path = tmp_path / "invalid-bounds.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("site_grids", {}),
        ("site_grids", []),
        ("site_grids", True),
        ("boundary_segment_lengths_m", {}),
        ("boundary_segment_lengths_m", []),
        ("boundary_segment_lengths_m", True),
    ],
)
def test_site_grids_and_boundary_segments_must_be_nonempty_mappings(
    tmp_path: Path, field: str, invalid_value: Any
) -> None:
    """至少一個站點格網與一條邊界線段是聚合規格的必要索引範圍。"""

    payload = _valid_payload()
    payload[field] = invalid_value
    path = tmp_path / "invalid-empty-mapping.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_value", [True, "100", 100, {"bandwidth": 100}])
def test_kde_bandwidths_must_be_a_list(tmp_path: Path, invalid_value: Any) -> None:
    """KDE bandwidth 設定必須是明確的 JSON list，不能由 scalar 或 object 猜測。"""

    payload = _valid_payload()
    payload["kde_bandwidths_m"] = invalid_value
    path = tmp_path / "invalid-bandwidth-type.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_bandwidths", [[], [100], [100, 200], [100, 200, 400, 800]])
def test_kde_bandwidths_must_contain_exactly_three_values(
    tmp_path: Path, invalid_bandwidths: list[Any]
) -> None:
    """KDE 必須固定比較三個頻寬，少於或多於三個都不能建立規格。"""

    payload = _valid_payload()
    payload["kde_bandwidths_m"] = invalid_bandwidths
    path = tmp_path / "invalid-bandwidth-count.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_bandwidths",
    [
        [0, 200, 400],
        [-1, 200, 400],
        [100, 0, 400],
        [100, 200, -400],
        [100, True, 400],
        [100, "200", 400],
        [100, [200], 400],
        [100, math.nan, 400],
        [100, math.inf, 400],
    ],
)
def test_kde_bandwidths_must_be_finite_positive_numbers(
    tmp_path: Path, invalid_bandwidths: list[Any]
) -> None:
    """每個 KDE 頻寬都必須是有限正數，避免非法單位或非有限核尺度進入分析。"""

    payload = _valid_payload()
    payload["kde_bandwidths_m"] = invalid_bandwidths
    path = tmp_path / "invalid-bandwidth-value.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_bandwidths", [[200, 100, 400], [100, 400, 200], [100, 200, 200]])
def test_kde_bandwidths_must_be_strictly_increasing(
    tmp_path: Path, invalid_bandwidths: list[int]
) -> None:
    """三個頻寬的順序必須嚴格遞增，確保敏感度產品的索引語意固定。"""

    payload = _valid_payload()
    payload["kde_bandwidths_m"] = invalid_bandwidths
    path = tmp_path / "invalid-bandwidth-order.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("invalid_hdr", [True, "0.5", 0.5, {"levels": [0.5, 0.75, 0.9]}])
def test_hdr_levels_must_be_a_list(tmp_path: Path, invalid_hdr: Any) -> None:
    """HDR 設定必須是 JSON list，不能把單一門檻或 object 當作三段產品設定。"""

    payload = _valid_payload()
    payload["hdr_levels"] = invalid_hdr
    path = tmp_path / "invalid-hdr-type.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    "invalid_hdr",
    [
        [],
        [0.5],
        [0.5, 0.75],
        [0.5, 0.75, 0.9, 0.95],
        [0.5, 0.9, 0.75],
        [0.75, 0.5, 0.9],
        [0.4, 0.75, 0.9],
        [0.5, 0.7, 0.9],
        [0.5, 0.75, 0.95],
        [0, 0.75, 0.9],
        [0.5, 1, 0.9],
        [0.5, True, 0.9],
        [0.5, "0.75", 0.9],
        [0.5, [0.75], 0.9],
        [0.5, math.nan, 0.9],
        [0.5, math.inf, 0.9],
    ],
)
def test_hdr_levels_must_match_exact_order_and_values(
    tmp_path: Path, invalid_hdr: list[Any]
) -> None:
    """HDR 只能精確使用 0.5、0.75、0.9 的遞增門檻，型別與數值偏差都必須拒絕。"""

    payload = _valid_payload()
    payload["hdr_levels"] = invalid_hdr
    path = tmp_path / "invalid-hdr-value.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("replicates", [0, -1, 1000.0, True, "1000"])
def test_bootstrap_replicates_require_positive_integer(
    tmp_path: Path, replicates: Any
) -> None:
    """bootstrap replicate 數必須是正的真正整數，避免零次、負值或 bool 混入。"""

    payload = _valid_payload()
    payload["bootstrap_replicates"] = replicates
    path = tmp_path / "invalid-bootstrap-replicates.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("confidence", [-0.01, 0.0, 1.0, 1.01, True, "0.95"])
def test_bootstrap_confidence_requires_finite_number_strictly_between_zero_and_one(
    tmp_path: Path, confidence: Any
) -> None:
    """bootstrap confidence level 必須嚴格位於 0 與 1 之間，且不得是 bool 或文字。"""

    payload = _valid_payload()
    payload["bootstrap_confidence_level"] = confidence
    path = tmp_path / "invalid-bootstrap-confidence.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize("seed", [-1, 2**128, 1.0, True, "123"])
def test_bootstrap_seed_requires_nonnegative_integer_and_rejects_bool(
    tmp_path: Path, seed: Any
) -> None:
    """bootstrap seed 必須是可重現用的非負整數，且 Python bool 不可冒充 seed 0/1。"""

    payload = _valid_payload()
    payload["bootstrap_seed"] = seed
    path = tmp_path / "invalid-bootstrap-seed.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap_replicates", 1),
        ("bootstrap_replicates", 1000),
        ("bootstrap_confidence_level", 0.0001),
        ("bootstrap_confidence_level", 0.9999),
        ("bootstrap_seed", 0),
        ("bootstrap_seed", 2**128 - 1),
    ],
)
def test_bootstrap_boundary_values_are_accepted(tmp_path: Path, field: str, value: Any) -> None:
    """契約允許的 bootstrap 下界與信賴區間內部值應可正常載入，不應過度收窄範圍。"""

    payload = _valid_payload()
    payload[field] = value
    path = tmp_path / "valid-bootstrap-boundary.json"
    _write_json(path, payload)

    spec = load_aggregate_spec(path)
    assert getattr(spec, field) == value


@pytest.mark.parametrize("policy", ["include_all", "", "exclude_data_gap", True, ["exclude"]])
def test_denominator_policy_must_equal_fixed_policy(tmp_path: Path, policy: Any) -> None:
    """分母政策是跨產品的固定版本識別，不可由輸入檔改成其他文字或型別。"""

    payload = _valid_payload()
    payload["denominator_policy"] = policy
    path = tmp_path / "invalid-denominator-policy.json"
    _write_json(path, payload)

    _assert_value_error_without_path(lambda: load_aggregate_spec(path), path)
