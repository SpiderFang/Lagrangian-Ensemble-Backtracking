"""lazy forcing window 的月份選擇、LRU、NWW lazy loading 與 hint 傳遞測試。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from lagrangian_backtracking.diffusion import SmagorinskySettings
from lagrangian_backtracking.forcing import NWWAnalysisMonth, OCMNativeMonth
from lagrangian_backtracking.forcing_window import (
    ForcingWindowManager,
    MissingForcingMonth,
)
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.models import SampleQC


def _mesh() -> NativeMesh:
    """建立可容納測試粒子的單一三角形 native mesh。"""

    return NativeMesh(
        node_lon=np.array([121.0, 121.001, 121.0]),
        node_lat=np.array([25.0, 25.0, 25.001]),
        node_xy=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        source_depth_m=np.full(3, 10.0),
        source_node_bottom_index=np.zeros(3, dtype=np.int64),
        face_nodes_local=np.array([[0, 1, 2, -1]]),
        face_node_count=np.array([3]),
        source_face_global_index=np.array([99]),
        bin_size_m=20.0,
    )


def _month_start_ns(month_id: str) -> int:
    """將 YYYYMM 轉成 UTC 月初奈秒，避免測試依賴本機時區。"""

    start = datetime.strptime(month_id, "%Y%m").replace(tzinfo=UTC)
    return int(start.timestamp()) * 1_000_000_000


def _ocm(month_id: str, mesh: NativeMesh, *, eastward_mps: float = 1.0) -> OCMNativeMonth:
    """建立兩個逐時支撐點的有效 OCM month；所有面保持 wet。"""

    times = np.array(
        [_month_start_ns(month_id), _month_start_ns(month_id) + 3_600_000_000_000], dtype=np.int64
    )
    z_levels = np.array([-10.0, 0.0])
    zcor = np.broadcast_to(z_levels, (2, 3, 2)).copy()
    hvel = np.zeros((2, 3, 2, 2), dtype=np.float64)
    hvel[..., 0] = eastward_mps
    vertical_velocity = np.zeros((2, 3, 2), dtype=np.float64)
    diffusivity = np.full((2, 3, 2), 0.01, dtype=np.float64)
    return OCMNativeMonth(
        month_id=month_id,
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=np.zeros((2, 3)),
        wetdry_elem=np.zeros((2, 1)),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=7_200.0,
    )


def _nww(month_id: str) -> NWWAnalysisMonth:
    """建立覆蓋測試位置的有效 NWW analysis month。"""

    shape = (2, 2, 2)
    return NWWAnalysisMonth(
        month_id=month_id,
        lon=np.array([120.0, 122.0]),
        lat=np.array([24.0, 26.0]),
        time_utc_ns=np.array(
            [_month_start_ns(month_id), _month_start_ns(month_id) + 3_600_000_000_000],
            dtype=np.int64,
        ),
        significant_wave_height=np.ones(shape),
        peak_frequency=np.full(shape, 0.1),
        peak_direction_raw_deg=np.zeros(shape),
        valid_mask_wave=np.ones(shape, dtype=bool),
        qc_flags=np.zeros(shape, dtype=np.uint16),
        maximum_time_gap_seconds=7_200.0,
    )


def _manager(
    ocm_by_month: dict[str, OCMNativeMonth],
    nww_by_month: dict[str, NWWAnalysisMonth] | None = None,
    *,
    max_resident_months: int = 2,
) -> tuple[ForcingWindowManager, Mock, Mock]:
    """建立注入式 manager 並回傳 loader mocks，方便檢查 lazy load 次數。"""

    ocm_loader = Mock(side_effect=lambda month_id: ocm_by_month.get(month_id))
    nww_loader = Mock(
        side_effect=lambda month_id: None if nww_by_month is None else nww_by_month.get(month_id)
    )
    manager = ForcingWindowManager(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        mesh=_mesh(),
        ocm_loader=ocm_loader,
        nww_loader=nww_loader,
        max_resident_months=max_resident_months,
    )
    return manager, ocm_loader, nww_loader


def _sample_time(month_id: str, *, offset_seconds: int = 1_800) -> int:
    """回傳月份內的 RK stage 測試時間。"""

    return _month_start_ns(month_id) + offset_seconds * 1_000_000_000


def test_same_month_material_facades_share_ocm_and_no_stokes_never_loads_nww() -> None:
    """不同沉降速度只建立輕量合併 facade，同月 OCM 及 no-Stokes NWW 都只讀一次／零次。"""

    month = "197001"
    manager, ocm_loader, nww_loader = _manager({month: _ocm(month, _mesh())})
    first = manager.provider(-0.1, include_stokes=False)
    second = manager.provider(-0.2, include_stokes=False)
    time_ns = _sample_time(month)
    assert first.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    assert second.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    assert ocm_loader.call_count == 1
    assert nww_loader.call_count == 0
    stats = manager.cache_stats
    assert stats.ocm_load_count == 1
    assert stats.ocm_cache_miss_count == 1
    assert stats.ocm_cache_hit_count >= 1
    assert stats.resident_ndarray_bytes > 0


def test_stokes_loads_nww_lazily_once_after_no_stokes() -> None:
    """先使用 no-Stokes 不碰 NWW；同月後續 Stokes 只觸發一次 NWW load。"""

    month = "197001"
    manager, ocm_loader, nww_loader = _manager(
        {month: _ocm(month, _mesh())}, {month: _nww(month)}
    )
    time_ns = _sample_time(month)
    manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, time_ns)
    assert nww_loader.call_count == 0
    stokes = manager.provider(-0.1, True)
    result = stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
    assert result.valid
    stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
    assert ocm_loader.call_count == 1
    assert nww_loader.call_count == 1
    assert manager.cache_stats.nww_load_count == 1


def test_missing_ocm_and_missing_nww_have_distinct_qc() -> None:
    """OCM 月份缺失是 OUTSIDE_TIME_RANGE，OCM 存在但 NWW 缺月是 WAVE_UNSUPPORTED。"""

    manager, _, nww_loader = _manager({})
    missing_ocm = manager.provider(-0.1, False).sample(
        1.0, 1.0, -5.0, _sample_time("197001")
    )
    assert missing_ocm.qc == SampleQC.OUTSIDE_TIME_RANGE
    assert nww_loader.call_count == 0

    month = "197002"
    manager, _, nww_loader = _manager({month: _ocm(month, _mesh())}, {})
    missing_nww = manager.provider(-0.1, True).sample(
        1.0, 1.0, -5.0, _sample_time(month)
    )
    assert missing_nww.qc == SampleQC.WAVE_UNSUPPORTED
    assert nww_loader.call_count == 1


def test_lru_eviction_reloads_month_and_preload_uses_same_contract() -> None:
    """容量一個時跨月會淘汰並重新載入；preload 仍沿用同一 LRU，不繞過月份驗證。"""

    jan, feb = "197001", "197002"
    manager, ocm_loader, _ = _manager(
        {jan: _ocm(jan, _mesh()), feb: _ocm(feb, _mesh())}, max_resident_months=1
    )
    provider = manager.provider(-0.1, False)
    for month in (jan, feb, jan):
        assert provider.sample(1.0, 1.0, -5.0, _sample_time(month)).valid
    manager.preload([feb])
    assert ocm_loader.call_count == 4
    assert manager.cache_stats.resident_month_ids == (feb,)
    assert manager.cache_stats.eviction_count == 3


def test_loader_error_for_existing_month_is_not_converted_to_qc() -> None:
    """注入 loader 的產品錯誤必須上拋，不能與整月不存在混為 OUTSIDE_TIME_RANGE。"""

    def broken_loader(month_id: str):
        """模擬已存在月份的 schema/array 讀取錯誤。"""

        del month_id
        raise ValueError("schema damaged")

    manager = ForcingWindowManager(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        mesh=_mesh(),
        ocm_loader=broken_loader,
    )
    with pytest.raises(ValueError, match="schema damaged"):
        manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, _sample_time("197001"))


def test_hint_is_forwarded_through_managed_facade() -> None:
    """managed facade 的 triangle_hint 應完整抵達 OCM sampler，不改變 forcing physics。"""

    month = "197001"
    ocm = _ocm(month, _mesh())
    spy = Mock(wraps=ocm.sample)
    ocm.sample = spy  # type: ignore[method-assign]
    manager, _, _ = _manager({month: ocm})
    result = manager.provider(-0.1, False).sample(
        1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0
    )
    assert result.valid
    assert spy.call_args.kwargs["triangle_hint"] == 0


def test_smagorinsky_facade_shares_ocm_lru_and_never_loads_nww() -> None:
    """Smagorinsky facade 應共用 OCM LRU，且擴散取樣不建立 NWW／Combined forcing。"""

    month = "197001"
    manager, ocm_loader, nww_loader = _manager({month: _ocm(month, _mesh())})
    settings = SmagorinskySettings(
        coefficient_cs=0.10,
        floor_m2ps=0.01,
        cap_m2ps=10.0,
        constant_kz_m2ps=0.02,
    )
    first = manager.smagorinsky_provider(settings)
    second = manager.smagorinsky_provider(settings)
    first_result = first.sample(1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0)
    second_result = second.sample(1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0)
    assert first_result.valid
    assert second_result.valid
    assert first.manager is manager
    assert first.settings is settings
    assert ocm_loader.call_count == 1
    assert nww_loader.call_count == 0
    assert manager.cache_stats.nww_cache_miss_count == 0


def test_smagorinsky_facade_forwards_triangle_hint_and_missing_ocm_is_explicit_qc() -> None:
    """擴散 hint 必須到達 OCM；缺少整月 OCM 時需保留方法／月份／缺失診斷。"""

    month = "197001"
    ocm = _ocm(month, _mesh())
    spy = Mock(wraps=ocm.sample_smagorinsky_diffusion)
    ocm.sample_smagorinsky_diffusion = spy  # type: ignore[method-assign]
    manager, _, nww_loader = _manager({month: ocm})
    settings = SmagorinskySettings(0.15, 0.01, 10.0, 0.02)
    result = manager.smagorinsky_provider(settings).sample(
        1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0
    )
    assert result.valid
    assert spy.call_args.kwargs["triangle_hint"] == 0
    assert spy.call_args.args[4] is settings
    assert nww_loader.call_count == 0

    missing_manager, _, missing_nww_loader = _manager({})
    missing = missing_manager.smagorinsky_provider(settings).sample(
        1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0
    )
    assert not missing.valid
    assert missing.qc == SampleQC.OUTSIDE_TIME_RANGE
    assert missing.diagnostics["method"] == "smagorinsky_native_mesh_p1_nodal"
    assert missing.diagnostics["forcing_month_id"] == month
    assert missing.diagnostics["ocm_month_missing"] is True
    assert missing_nww_loader.call_count == 0


def test_smagorinsky_four_argument_call_keeps_existing_provider_shape() -> None:
    """facade 的四參數 ``__call__`` 介面不應偷偷需要 triangle hint。"""

    month = "197001"
    manager, _, _ = _manager({month: _ocm(month, _mesh())})
    provider = manager.smagorinsky_provider(SmagorinskySettings(0.10, 0.01, 10.0, 0.02))
    result = provider(1.0, 1.0, -5.0, _sample_time(month))
    assert result.valid


def _save_npy(path: Path, name: str, value: np.ndarray) -> None:
    """將測試 array 寫成 production reader 使用的 .npy 檔。"""

    np.save(path / name, value)


def _write_production_fixture(root: Path, month_id: str) -> None:
    """建立正式 root layout 的最小 OCM/NWW 陣列，驗證 from_roots 的檔案路徑契約。"""

    ocm_grid = root / "ocm" / "synthetic_domain" / "grid"
    ocm_month = root / "ocm" / "synthetic_domain" / "months" / month_id
    nww_grid = root / "nww" / "synthetic_domain" / "grid"
    nww_month = root / "nww" / "synthetic_domain" / "months" / month_id
    for directory in (ocm_grid, ocm_month, nww_grid, nww_month):
        directory.mkdir(parents=True)
    _save_npy(ocm_grid, "source_lon.npy", np.array([121.0, 121.001, 121.0]))
    _save_npy(ocm_grid, "source_lat.npy", np.array([25.0, 25.0, 25.001]))
    _save_npy(ocm_grid, "source_depth_m.npy", np.full(3, 10.0))
    _save_npy(ocm_grid, "source_node_bottom_index.npy", np.zeros(3, dtype=np.int64))
    _save_npy(ocm_grid, "source_face_nodes_local.npy", np.array([[0, 1, 2, -1]], dtype=np.int64))
    _save_npy(ocm_grid, "source_face_node_count.npy", np.array([3], dtype=np.int64))
    _save_npy(ocm_grid, "source_face_global_index.npy", np.array([99], dtype=np.int64))
    month_data = _ocm(month_id, _mesh())
    for name in (
        "time_utc_ns",
        "hvel",
        "vertical_velocity",
        "zcor",
        "elev",
        "wetdry_elem",
        "diffusivity",
    ):
        _save_npy(ocm_month, f"{name}.npy", getattr(month_data, name))
    nww_data = _nww(month_id)
    for name in (
        "time_utc_ns",
        "significant_wave_height",
        "peak_frequency",
        "peak_direction_raw_deg",
        "valid_mask_wave",
        "qc_flags",
    ):
        _save_npy(nww_month, f"{name}.npy", getattr(nww_data, name))
    _save_npy(nww_grid, "lon.npy", nww_data.lon)
    _save_npy(nww_grid, "lat.npy", nww_data.lat)


def test_from_roots_loads_mesh_once_and_production_missing_month_is_qc(tmp_path: Path) -> None:
    """from_roots 使用固定 root layout；已存在月份可取樣，缺月份回時間範圍 QC。"""

    month = "197001"
    _write_production_fixture(tmp_path, month)
    manager = ForcingWindowManager.from_roots(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        ocm_root=tmp_path / "ocm",
        nww_root=tmp_path / "nww",
    )
    assert manager.provider(-0.1, True).sample(1.0, 1.0, -5.0, _sample_time(month)).valid
    missing = manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, _sample_time("197002"))
    assert missing.qc == SampleQC.OUTSIDE_TIME_RANGE
    assert manager.cache_stats.ocm_load_count == 1


def test_missing_month_exception_is_supported_by_injected_loader() -> None:
    """注入 loader 可用 MissingForcingMonth 明確標示整個月份目錄缺失。"""

    def missing_loader(month_id: str):
        """模擬不存在的 YYYYMM directory。"""

        raise MissingForcingMonth(month_id)

    manager = ForcingWindowManager(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        mesh=_mesh(),
        ocm_loader=missing_loader,
    )
    assert (
        manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, _sample_time("197001")).qc
        == SampleQC.OUTSIDE_TIME_RANGE
    )
