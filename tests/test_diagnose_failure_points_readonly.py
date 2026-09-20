"""固定一至五點唯讀診斷契約：保留綁定、原精確定位、幾何量且禁止步進。"""

import ast
import gzip
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

PATH = Path(__file__).parents[1] / 'scripts/diagnose_failure_points_readonly.py'
SPEC = importlib.util.spec_from_file_location('point_diagnostic', PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_point_comparison_uses_existing_methods_without_extrapolation():
    """模擬索引查無但全面候選成功；輸出權重完全來自既有精確方法。"""
    calls = []

    @dataclass
    class Sample:
        """保留無效樣本與品質旗標，不將失敗填成零速度。"""
        qc: int = 1
        valid: bool = False
        u_mps: float = float('nan')

    def locate(x, y, **kwargs):
        """記錄提示與無提示路徑，刻意讓索引查無。"""
        calls.append(('locate', kwargs))
        return None

    def exact(point, triangle_id, *, tolerance):
        """原精確方法替身，測試不得自行重算權重。"""
        calls.append(('exact', triangle_id, tolerance))
        return {'triangle_id': triangle_id}, np.array([0.2, 0.3, 0.5])

    def sample(*args, **kwargs):
        """樣本呼叫必須保存原點、原UTC及第62代提示。"""
        calls.append(('sample', args, kwargs))
        return Sample()

    mesh = SimpleNamespace(triangle_bbox_min=np.array([[0, 0], [10, 10]]),
                           triangle_bbox_max=np.array([[2, 2], [12, 12]]),
                           locate=locate, _location_from_triangle=exact)
    manager = SimpleNamespace(mesh=mesh, projection=SimpleNamespace(
        unproject=lambda x, y: (120.5, 22.5)))
    request = SimpleNamespace(velocity=SimpleNamespace(manager=manager, sample=sample),
                              boundaries=SimpleNamespace(flow_domain=box(0, 0, 3, 3),
                                                         own_local_domain=box(0, 0, 2, 2),
                                                         flow_open_boundary=None))
    event = dict(particle_id='p', x_m=1., y_m=1., z_m=-2., time_utc_ns=123)
    result = runner.compare_point(request, event, 7)
    assert calls[0] == ('sample', (1., 1., -2., 123), {'triangle_hint': 7})
    assert ('exact', 0, 1e-10) in calls
    assert ('locate', {'triangle_hint': 7}) in calls and ('locate', {}) in calls
    assert result['bbox_candidates'] == [dict(triangle_id=0, location={'triangle_id': 0},
                                              weights=[0.2, 0.3, 0.5])]
    assert result['sample']['u_mps'] == 'nan'
    assert result['flow_domain_covers'] and result['own_local_domain_covers']
    assert result['flow_open_boundary_distance_m'] is None


def test_entrypoint_cannot_advance_restore_or_draw_rng():
    """入口不得呼叫任何批次步進、原run寫入或亂數抽樣方法，硬限固定180秒。"""
    tree = ast.parse(PATH.read_text())
    forbidden = {'advance', 'run_shard', 'restore', 'reconcile', 'ProductionBatch',
                 'default_rng', 'normal', 'random', 'from_checkpoint'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, 'attr', getattr(node.func, 'id', None))
            assert name not in forbidden
    assert 'process.wait(timeout=180)' in PATH.read_text()
    assert 'start_new_session=True' in PATH.read_text()
    assert 'stop_owned(process)' in PATH.read_text()


def point_fixture(tmp_path, count=2, sequence=302, legacy=False):
    """建立十粒子原分片的小型固定代fixture；事件只含指定失敗子集，不建模型。"""
    cp = tmp_path / 'checkpoint'
    cp.mkdir()
    records = [dict(identity={'particle_id': f'p{i}'},
                    state=dict(x_m=float(i), y_m=2., z_m=-3., time_utc_ns=123 + i),
                    triangle_hint=i) for i in range(10)]
    raw = gzip.compress(json.dumps({'records': records}).encode())
    (cp / 'compact_state.json.gz').write_bytes(raw)
    descriptor = dict(sequence=sequence, binding={'shard_id': 'shard'},
                      files={'compact_state.json.gz': {'sha256': hashlib.sha256(raw).hexdigest()}})
    (cp / 'checkpoint.json').write_text(json.dumps(descriptor))
    events = tmp_path / 'events.json'
    events.write_text(json.dumps({'events': [dict(event=dict(particle_id=f'p{i}', **r['state']))
                                           for i, r in enumerate(records[:count])]}))
    spec = dict(shard_id='shard', events=str(events))
    spec.update({'checkpoint62': str(cp)} if legacy else
                dict(checkpoint_dir=str(cp), expected_sequence=sequence, expected_point_count=count))
    return spec, {f'p{i}': object() for i in range(10)}, cp, events


@pytest.mark.parametrize('count,sequence,legacy', [(2, 302, False), (5, 62, True)])
def test_fixed_scope_accepts_new_and_legacy(tmp_path, count, sequence, legacy):
    """新兩點302代與舊五點62代皆保留原事件、提示和分片身分。"""
    spec, requests, cp, _ = point_fixture(tmp_path, count, sequence, legacy)
    actual_cp, _, events, states = runner.bound_failure_points(spec, requests)
    assert actual_cp == cp and len(events) == count and states['p1']['triangle_hint'] == 1


@pytest.mark.parametrize('field,value', [
    ('expected_sequence', True), ('expected_sequence', 0), ('expected_sequence', -1),
    ('expected_sequence', '302'), ('expected_sequence', 302.0),
    ('expected_point_count', True), ('expected_point_count', False),
    ('expected_point_count', 0), ('expected_point_count', 6),
    ('expected_point_count', -1), ('expected_point_count', '2'), ('expected_point_count', 2.0)])
def test_scope_rejects_invalid_integer(field, value):
    """布林、文字、浮點與超界數值不能被悄悄轉成合法讀取範圍。"""
    spec = dict(checkpoint_dir='x', expected_sequence=302, expected_point_count=2)
    spec[field] = value
    with pytest.raises(ValueError):
        runner.checkpoint_scope(spec)


@pytest.mark.parametrize('missing', ['checkpoint_dir', 'expected_sequence', 'expected_point_count'])
def test_scope_rejects_partial_fields(missing):
    """新格式缺任一欄即拒絕，不借舊格式預設補齊。"""
    spec = dict(checkpoint_dir='x', expected_sequence=302, expected_point_count=2)
    del spec[missing]
    with pytest.raises(ValueError):
        runner.checkpoint_scope(spec)


def test_scope_rejects_mixed_or_missing():
    """新舊同時指定或完全未指定均屬歧義，必須在setup前拒絕。"""
    for spec in ({}, dict(checkpoint62='old', checkpoint_dir='new',
                         expected_sequence=302, expected_point_count=2)):
        with pytest.raises(ValueError):
            runner.checkpoint_scope(spec)


@pytest.mark.parametrize('fault,reason', [
    ('sequence', '序號'), ('shard', '分片'), ('sha', 'SHA'),
    ('duplicate', '重複'), ('foreign', '不屬於'), ('missing_request', '不屬於'),
    ('missing_state', '不屬於'), ('count', '點數'),
    ('x_m', '位置'), ('y_m', '位置'), ('z_m', '位置'), ('time_utc_ns', '時間')])
def test_bound_points_reject_wrong_binding(tmp_path, fault, reason):
    """逐一破壞代號、分片、SHA、粒子身分或精確狀態，所有錯誤均fail-closed。"""
    spec, requests, cp, events_path = point_fixture(tmp_path)
    descriptor_path = cp / 'checkpoint.json'
    d = json.loads(descriptor_path.read_text())
    e = json.loads(events_path.read_text())
    if fault == 'sequence':
        d['sequence'] = 303
    elif fault == 'shard':
        d['binding']['shard_id'] = 'other'
    elif fault == 'sha':
        d['files']['compact_state.json.gz']['sha256'] = 'bad'
    elif fault == 'duplicate':
        e['events'][1] = e['events'][0]
    elif fault == 'foreign':
        e['events'][0]['event']['particle_id'] = 'other'
    elif fault == 'missing_request':
        del requests['p0']
    elif fault == 'missing_state':
        state_path = cp / 'compact_state.json.gz'
        state = json.loads(gzip.decompress(state_path.read_bytes()))
        state['records'] = state['records'][1:]
        raw = gzip.compress(json.dumps(state).encode())
        state_path.write_bytes(raw)
        d['files']['compact_state.json.gz']['sha256'] = hashlib.sha256(raw).hexdigest()
    elif fault == 'count':
        e['events'].pop()
    else:
        e['events'][0]['event'][fault] += 1
    descriptor_path.write_text(json.dumps(d))
    events_path.write_text(json.dumps(e))
    with pytest.raises(ValueError, match=reason):
        runner.bound_failure_points(spec, requests)
