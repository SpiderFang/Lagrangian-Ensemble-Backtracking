"""OCM surface arrival scalar 的取值等價與布局測試。

這些測試使用小型 synthetic surface cache，專門鎖定 input derivation 的資料契約：
規則格網四角的靜態／逐時有效性、有限值檢查、最近格點 scalar、重複 UTC 的後者優先
規則，以及逐月陣列的 shape 驗證。測試中的 C-order、Fortran-order 與非連續 ndarray
分別代表列優先、欄優先與帶有間隔步長的陣列布局；不代表任何 OCM 科學資料或正式
SERVER 結果。
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import lagrangian_backtracking.input_derivation as input_derivation_module


def _layout_array(array: np.ndarray, layout: str) -> np.ndarray:
    """以指定布局回傳同一組邏輯值，讓測試能隔離索引與記憶體連續性。

    ``C`` 與 ``F`` 分別代表列優先順序（C-order）與欄優先順序（Fortran-order）；
    ``noncontiguous`` 使用帶有間隔欄位的陣列檢視，確保每個逐時網格不能依賴連續記憶體。
    三者的 shape、dtype 與元素值完全相同，故比較結果只反映取值方法是否依賴整張網格
    的 ``ravel`` 複製。
    """

    if layout == "C":
        return np.ascontiguousarray(array)
    if layout == "F":
        return np.asfortranarray(array)
    if layout == "noncontiguous":
        padded = np.empty((*array.shape[:-1], array.shape[-1] * 2), dtype=array.dtype)
        padded[..., ::2] = array
        # 間隔欄位不會被取值；用 dtype 可表達的零填入，避免 uint8 fixture 溢位。
        padded[..., 1::2] = np.zeros((), dtype=array.dtype)
        view = padded[..., ::2]
        assert not view.flags.c_contiguous
        assert not view.flags.f_contiguous
        return view
    raise AssertionError(f"未知測試布局：{layout}")


def _build_surface_product(
    tmp_path: Any,
    *,
    layout: str,
) -> tuple[Any, dict[Any, np.ndarray], tuple[int, ...]]:
    """建立兩個 synthetic 月份與 fake loader 對照表。

    兩個月份刻意共享一個 UTC，第二月份的值可用來檢查既有 dict assignment 的
    prefer-last precedence。回傳的 product 只提供被 private helper 讀取的
    ``grid_dir`` 與 ``months`` 欄位，避免測試引入完整輸入發布流程。
    """

    root = tmp_path / "surface"
    grid_dir = root / "grid"
    grid_dir.mkdir(parents=True)
    values: dict[Any, np.ndarray] = {
        grid_dir / "lon.npy": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        grid_dir / "lat.npy": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        grid_dir / "mask_static.npy": np.ones((3, 4), dtype=np.uint8),
    }
    months: list[Any] = []
    for month_number, times in ((1, (100, 200)), (2, (200, 300))):
        directory = root / f"2024{month_number:02d}"
        directory.mkdir()
        base = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        offset = np.float32(100.0 * month_number)
        numeric = {
            "u_surface_mps.npy": base + offset,
            "v_surface_mps.npy": base * np.float32(0.1) + offset + np.float32(1.0),
            "surface_z.npy": base * np.float32(0.01) - np.float32(5.0) + offset,
            "eta_m.npy": base * np.float32(0.02) + offset,
            "valid_mask_surface.npy": np.ones(base.shape, dtype=np.uint8),
            "qc_flags.npy": np.zeros(base.shape, dtype=np.int16),
        }
        for filename, array in numeric.items():
            values[directory / filename] = _layout_array(array, layout)
        months.append(
            SimpleNamespace(
                label=f"2024{month_number:02d}",
                directory=directory,
                time_ns=np.asarray(times, dtype=np.int64),
            )
        )
    product = SimpleNamespace(grid_dir=grid_dir, months=tuple(months))
    return product, values, (200, 300)


def _legacy_surface_series_for_test(
    product: Any,
    *,
    lon: float,
    lat: float,
    loader: Callable[[Any], np.ndarray],
) -> tuple[dict[int, float], dict[int, float]]:
    """保留修改前整張 frame 轉型路徑，作為輸出等價性的測試 oracle。

    這段只在小型 synthetic 陣列上執行，並不作為產品實作；它完整保留原本依列優先
    順序攤平、四角有限值 gate、最近點速度與重複 UTC 覆蓋規則，讓效能修改有可檢驗的
    行為基準。
    """

    spatial_shape, spatial_indices, flat_index, static_valid = (
        input_derivation_module._grid_spatial_support(
            grid_dir=product.grid_dir,
            product_label="OCM surface",
            lon=lon,
            lat=lat,
        )
    )
    elevation: dict[int, float] = {}
    speed: dict[int, float] = {}
    for month in product.months:
        u_surface = loader(month.directory / "u_surface_mps.npy")
        v_surface = loader(month.directory / "v_surface_mps.npy")
        surface_z = loader(month.directory / "surface_z.npy")
        eta = loader(month.directory / "eta_m.npy")
        valid_surface = loader(month.directory / "valid_mask_surface.npy")
        qc_flags = loader(month.directory / "qc_flags.npy")
        arrays = (u_surface, v_surface, surface_z, eta, valid_surface, qc_flags)
        if any(
            array.shape[0] != month.time_ns.size or tuple(array.shape[1:]) != spatial_shape
            for array in arrays
        ):
            raise input_derivation_module.InputDerivationError(
                f"OCM surface {month.label} 陣列時間／空間 shape 不符"
            )
        for local, time_ns in enumerate(month.time_ns):
            u_frame = np.asarray(u_surface[local], dtype=np.float64).ravel()
            v_frame = np.asarray(v_surface[local], dtype=np.float64).ravel()
            surface_z_frame = np.asarray(surface_z[local], dtype=np.float64).ravel()
            eta_frame = np.asarray(eta[local], dtype=np.float64).ravel()
            valid_frame = np.asarray(valid_surface[local], dtype=bool).ravel()
            support = list(spatial_indices)
            available = static_valid and all(bool(valid_frame[index]) for index in support)
            if available:
                support_values = np.concatenate(
                    (
                        u_frame[support],
                        v_frame[support],
                        surface_z_frame[support],
                        eta_frame[support],
                    )
                )
                available = bool(np.all(np.isfinite(support_values)))
            if available:
                elevation[int(time_ns)] = float(eta_frame[flat_index])
                speed[int(time_ns)] = float(np.hypot(u_frame[flat_index], v_frame[flat_index]))
            else:
                elevation[int(time_ns)] = float("nan")
                speed[int(time_ns)] = float("nan")
    return elevation, speed


def _assert_series_equal(
    actual: tuple[dict[int, float], dict[int, float]],
    expected: tuple[dict[int, float], dict[int, float]],
) -> None:
    """逐元素精確比較含 NaN 的 UTC scalar mapping，避免近似值掩蓋布局差異。"""

    assert actual[0].keys() == expected[0].keys()
    assert actual[1].keys() == expected[1].keys()
    np.testing.assert_array_equal(
        np.asarray(tuple(actual[0].values()), dtype=np.float64),
        np.asarray(tuple(expected[0].values()), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(tuple(actual[1].values()), dtype=np.float64),
        np.asarray(tuple(expected[1].values()), dtype=np.float64),
    )


@pytest.mark.parametrize("layout", ["C", "F", "noncontiguous"])
def test_surface_series_matches_legacy_for_supported_array_layouts(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    """surface scalar 對 C/F/非連續布局都必須與原本輸出逐值相同。"""

    product, values, duplicate_times = _build_surface_product(tmp_path, layout=layout)

    def fake_load_npy(path: Any, *, dtype: Any = None) -> np.ndarray:
        return values[path]

    monkeypatch.setattr(input_derivation_module, "_load_npy", fake_load_npy)
    location = {"lon": 1.25, "lat": 0.75}
    expected = _legacy_surface_series_for_test(product, **location, loader=fake_load_npy)
    actual = input_derivation_module._surface_series_for_location(product, **location)

    _assert_series_equal(actual, expected)
    # 第二月份含有與第一月份相同的 UTC；後者值必須覆蓋前者，不能因優化改成前者優先。
    duplicate_time = duplicate_times[0]
    nearest = np.unravel_index(
        input_derivation_module._grid_spatial_support(
            grid_dir=product.grid_dir,
            product_label="OCM surface",
            **location,
        )[2],
        (3, 4),
        order="C",
    )
    assert nearest == (1, 1)
    # 位置 (1, 1) 的 C-order flatten index 是 5；第二月份 local=0 的 synthetic 值
    # 直接由 base=5 與 offset=200 組成，這個手算檢查可獨立確認「最近點」沒有被誤改成四角平均。
    expected_eta = np.float64(np.float32(5.0) * np.float32(0.02) + np.float32(200.0))
    expected_speed = np.hypot(
        np.float64(np.float32(5.0) + np.float32(200.0)),
        np.float64(np.float32(5.0) * np.float32(0.1) + np.float32(201.0)),
    )
    assert actual[0][duplicate_time] == float(expected_eta)
    assert actual[1][duplicate_time] == float(expected_speed)


def test_surface_series_preserves_nan_dynamic_mask_and_static_four_corner_gate(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN、逐時 mask 與靜態四角 mask 必須各自維持原本的 invalid 語意。"""

    product, values, _ = _build_surface_product(tmp_path, layout="noncontiguous")

    def fake_load_npy(path: Any, *, dtype: Any = None) -> np.ndarray:
        return values[path]

    monkeypatch.setattr(input_derivation_module, "_load_npy", fake_load_npy)
    location = {"lon": 1.25, "lat": 0.75}
    _, support_indices, _, _ = input_derivation_module._grid_spatial_support(
        grid_dir=product.grid_dir,
        product_label="OCM surface",
        **location,
    )
    support = np.unravel_index(np.asarray(support_indices, dtype=np.intp), (3, 4), order="C")

    # 第一月份第一個 UTC 的四角之一為 NaN，只影響該 UTC；不能搜尋其他最近有效格點。
    first_u = values[product.months[0].directory / "u_surface_mps.npy"]
    first_u[(0, *support)] = np.nan
    expected = _legacy_surface_series_for_test(product, **location, loader=fake_load_npy)
    actual = input_derivation_module._surface_series_for_location(product, **location)
    _assert_series_equal(actual, expected)
    assert np.isnan(actual[0][100])
    assert np.isfinite(actual[0][200])

    # 第二月份第二個 UTC 的 dynamic mask 失效，重複 UTC 的第一筆仍由第二月份有效值覆蓋。
    second_valid = values[product.months[1].directory / "valid_mask_surface.npy"]
    second_valid[(1, *support)] = 0
    expected = _legacy_surface_series_for_test(product, **location, loader=fake_load_npy)
    actual = input_derivation_module._surface_series_for_location(product, **location)
    _assert_series_equal(actual, expected)
    assert np.isnan(actual[0][300])

    # 靜態 mask 是所有月份共用的四角 gate；任一角失效時所有 UTC 都必須失效。
    static_mask = values[product.grid_dir / "mask_static.npy"]
    static_mask[support[0][0], support[1][0]] = 0
    expected = _legacy_surface_series_for_test(product, **location, loader=fake_load_npy)
    actual = input_derivation_module._surface_series_for_location(product, **location)
    _assert_series_equal(actual, expected)
    assert all(np.isnan(value) for value in actual[0].values())
    assert all(np.isnan(value) for value in actual[1].values())


def test_surface_series_keeps_time_spatial_shape_contract(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任何 surface 欄位的時間或空間 shape 不符時仍須 fail closed。"""

    product, values, _ = _build_surface_product(tmp_path, layout="C")

    def fake_load_npy(path: Any, *, dtype: Any = None) -> np.ndarray:
        return values[path]

    monkeypatch.setattr(input_derivation_module, "_load_npy", fake_load_npy)
    bad_path = product.months[0].directory / "surface_z.npy"
    values[bad_path] = values[bad_path][..., :-1]
    with pytest.raises(input_derivation_module.InputDerivationError, match="shape 不符"):
        input_derivation_module._surface_series_for_location(product, lon=1.25, lat=0.75)
