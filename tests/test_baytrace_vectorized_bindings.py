"""建場向量化與舊逐節點參考的位元級／拒絕理由對照，不改物理或PI核心。"""

import importlib.util
from dataclasses import fields
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lagrangian_backtracking.baytrace_native_cpu import BayTraceGridArrays
from lagrangian_backtracking.baytrace_ocm_adapter import (
    OCMPreparedFrameResolver,
    _FramePreparationFallback,
)

SPEC = importlib.util.spec_from_file_location(
    'scalar_reference', Path(__file__).with_name('baytrace_scalar_reference.py'))
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def fixture(seed=17, dtypes=(np.float64,) * 5):
    """建立隨機48節點、7層兩端場；底部未使用層刻意含NaN／Inf。

    五種dtype依序控制z、eta、dp、水平與垂向速度；不同端點各自隨機值。
    sigma留在合法範圍內，避免資料製作本身的捨入跨越邊界。
    """
    rng = np.random.default_rng(seed)
    nnode, nvrt = 48, 7
    ztype, etype, dtype, htype, vtype = dtypes
    xy = np.tile([[0., 0.], [10., 0.], [0., 10.]], (nnode // 3, 1))
    triangles = np.arange(nnode).reshape(-1, 3)
    mesh = SimpleNamespace(node_xy=xy, source_depth_m=rng.uniform(10, 200, nnode).astype(dtype),
                           triangle_nodes=triangles, triangle_neighbors=np.full_like(triangles, -1))
    first = rng.integers(0, nvrt - 1, nnode)
    first[0] = 0
    rows = {}
    for endpoint in ('before', 'after'):
        eta = rng.uniform(-.5, .5, nnode).astype(etype)
        z = np.empty((nnode, nvrt), dtype=ztype)
        for node in range(nnode):
            z[node, :first[node]] = np.nan
            sigma = np.linspace(-.98, -.02, nvrt - first[node])
            z[node, first[node]:] = eta[node] + sigma * (mesh.source_depth_m[node] + eta[node])
        horizontal = rng.normal(size=(nnode, nvrt, 3)).astype(htype)
        vertical = rng.normal(size=(nnode, nvrt)).astype(vtype)
        unused = np.arange(nvrt)[None, :] < first[:, None]
        horizontal[unused] = np.nan
        vertical[unused] = np.inf
        rows.update({f'z_{endpoint}': z, f'eta_{endpoint}': eta,
                     f'hvel_{endpoint}': horizontal, f'vertical_{endpoint}': vertical})
    return mesh, rows


@pytest.mark.parametrize('dtypes', list(product((np.float32, np.float64), repeat=5)))
@pytest.mark.parametrize('seed', [17, 41])
def test_all_arrays_bitwise_and_inputs_unchanged(dtypes, seed):
    """32種混合dtype逐欄對照bytes；包含未使用底層、沉降相加及sigma弱純量捨入。"""
    mesh, rows = fixture(seed, dtypes)
    before = {key: value.tobytes() for key, value in rows.items()}
    expected = reference.reference_build_grid_and_fields(None, mesh, rows, settling_velocity_mps=-.02)
    actual = OCMPreparedFrameResolver._build_grid_and_fields(None, mesh, rows, settling_velocity_mps=-.02)
    for left, right in zip(expected, actual, strict=True):
        for field in fields(left):
            old, new = getattr(left, field.name), getattr(right, field.name)
            if isinstance(old, np.ndarray):
                assert old.dtype == new.dtype
                assert old.shape == new.shape
                assert old.tobytes() == new.tobytes(), field.name
            else:
                assert old == new
    assert before == {key: value.tobytes() for key, value in rows.items()}


@pytest.mark.parametrize('case', [
    'shape', 'depth', 'eta', 'empty', 'mismatch', 'hole', 'denominator', 'sigma_finite',
    'vertical_order', 'sigma_order', 'range', 'reconstruction', 'hvel_before', 'hvel_after',
    'vertical_before', 'vertical_after', 'earliest_node', 'index',
])
def test_rejection_reason_and_detail_equal(case):
    """每個原gate及跨節點衝突均保留第一個拒絕原因與欄位細節，不能放寬契約。"""
    mesh, rows = fixture()
    if case == 'shape':
        rows['vertical_before'] = rows['vertical_before'][:, :-1]
    elif case == 'depth':
        mesh.source_depth_m[0] = 0
    elif case == 'eta':
        rows['eta_after'][0] = np.nan
    elif case == 'empty':
        rows['z_before'][0] = np.nan
    elif case == 'mismatch':
        rows['z_after'][0, 0] = np.nan
    elif case == 'hole':
        rows['z_before'][0, 3] = np.nan
    elif case == 'denominator':
        rows['eta_before'][0] = -mesh.source_depth_m[0]
    elif case == 'sigma_finite':
        mesh.source_depth_m[0] = 1e-300
        rows['eta_before'][0] = 0
        rows['z_before'][0] = -1e100
    elif case == 'vertical_order':
        rows['z_before'][0, 2] = rows['z_before'][0, 1] - 1
    elif case == 'sigma_order':
        mesh.source_depth_m[0] = .001
        for end in ('before', 'after'):
            rows[f'eta_{end}'][0] = 0
            rows[f'z_{end}'][0] = np.linspace(-.0009, -.0001, 7)
        rows['z_before'][0, 1] = rows['z_before'][0, 0] - 1e-8
    elif case == 'range':
        rows['z_before'][0, 0] = -2 * mesh.source_depth_m[0]
    elif case == 'reconstruction':
        mesh.source_depth_m[0] = 1e20
        rows['eta_before'][0] = 1e20
        rows['z_before'][0] = np.arange(1, 8)
    elif case == 'index':
        mesh.triangle_nodes[0, 0] = 9999
    elif case == 'earliest_node':
        rows['hvel_after'][0, 0, 0] = np.nan
        rows['z_before'][1] = np.nan
    else:
        rows[case][0, 0] = np.nan
    messages = []
    for build in (reference.reference_build_grid_and_fields,
                  OCMPreparedFrameResolver._build_grid_and_fields):
        with np.errstate(all='ignore'), pytest.raises(_FramePreparationFallback) as error:
            build(None, mesh, rows, settling_velocity_mps=-.02)
        messages.append((error.value.reason, error.value.detail, error.value.qc))
    assert messages[0] == messages[1]


def grid_input():
    """正向三角形與四邊形混合，第二三角形必須獨立檢核。"""
    return dict(i34=np.array([3, 4]), elnode=np.array([[0, 1, 2, -1], [0, 1, 3, 2]]),
                grid_x=np.array([0., 2., 0., 2.]), grid_y=np.array([0., 0., 2., 2.]),
                ic3=np.full((4, 2), -1), nvrt=3, kbp=np.zeros(4, dtype=int),
                kbe=np.zeros(2, dtype=int), sigma_lcl=np.tile([[-1.], [-.5], [0.]], (1, 4)),
                dp=np.full(4, 10.))


@pytest.mark.parametrize('case', ['valid', 'index', 'unused', 'first', 'second', 'zero',
                                 'earliest_element', 'finite', 'neighbor', 'kbp', 'sigma_shape'])
def test_topology_preserves_order_and_quad_second_triangle(case):
    """逐元素參考與向量化檢核完全一致，含索引、面積閾值與錯誤優先順序。"""
    values = grid_input()
    if case == 'index':
        values['elnode'][1, 3] = 999
    elif case == 'unused':
        values['elnode'][0, 3] = 0
    elif case in ('first', 'earliest_element'):
        values['elnode'][0, :3] = [0, 2, 1]
        if case == 'earliest_element':
            values['elnode'][1, 3] = -5
    elif case == 'second':
        values['elnode'][1] = [0, 1, 3, 1]
    elif case == 'zero':
        values['grid_x'][:] = 0
    elif case == 'finite':
        values['grid_y'][0] = np.inf
    elif case == 'neighbor':
        values['ic3'][0, 0] = 9
    elif case == 'kbp':
        values['kbp'][0] = 3
    elif case == 'sigma_shape':
        values['sigma_lcl'] = values['sigma_lcl'][:, :3]
    if case == 'valid':
        old, new = reference.BayTraceGridArrays(**values), BayTraceGridArrays(**values)
        for field in fields(old):
            np.testing.assert_array_equal(getattr(old, field.name), getattr(new, field.name))
    else:
        errors = []
        for cls in (reference.BayTraceGridArrays, BayTraceGridArrays):
            with pytest.raises(ValueError) as error:
                cls(**values)
            errors.append(str(error.value))
        assert errors[0] == errors[1]
