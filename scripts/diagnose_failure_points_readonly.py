"""只讀一至五個既有失敗點，比對原取樣器、網格定位及公尺制幾何；不建立執行批次。

輸入是明示來源規格、原始失敗事件及固定代狀態；不推進、不建亂數產生器、不發布
任何原run檔案。外層只建立一個診斷子程序，以180秒硬限清理自己的程序群組。
"""

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    """讀取明示來源檔的SHA，不遍歷檢查點歷史。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """保留原始數值；非有限數以文字明示，避免JSON誤當零或有效速度。"""
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def checkpoint_scope(spec):
    """解析固定代與點數，避免新舊欄位混用後悄悄查到其他狀態。

    新格式三欄必須成套，序號為正整數、點數為一至五的整數，布林值不視為數字。
    僅舊格式保留第62代／五點契約；這是診斷讀取範圍，不改模型檢查點規格。
    """
    keys = {'checkpoint_dir', 'expected_sequence', 'expected_point_count'}
    present = keys.intersection(spec)
    if present:
        if present != keys or 'checkpoint62' in spec:
            raise ValueError('新三欄必須成套，且不能混用checkpoint62')
        sequence, count = spec['expected_sequence'], spec['expected_point_count']
        if type(sequence) is not int or sequence <= 0:
            raise ValueError('expected_sequence必須是正整數且不可為bool')
        if type(count) is not int or not 1 <= count <= 5:
            raise ValueError('expected_point_count必須是1..5整數且不可為bool')
        return Path(spec['checkpoint_dir']), sequence, count
    if 'checkpoint62' not in spec:
        raise ValueError('缺少成套檢查點欄位或舊checkpoint62')
    return Path(spec['checkpoint62']), 62, 5


def bound_failure_points(spec, requests):
    """核對指定代描述檔、壓縮狀態SHA與事件身分後，回傳原事件及狀態。

    只讀這一代compact，不讀歷史鏈；每點須唯一且同時屬於十粒子request與狀態，
    精確位置／UTC必須等於保存值，不能以相近座標或其他分片粒子代替。
    """
    cp, sequence, count = checkpoint_scope(spec)
    descriptor = json.loads((cp / 'checkpoint.json').read_text())
    if (descriptor['sequence'] != sequence
            or descriptor['binding']['shard_id'] != spec['shard_id']):
        raise ValueError('檢查點序號或分片不符指定綁定')
    state_path = cp / 'compact_state.json.gz'
    raw = state_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor['files']['compact_state.json.gz']['sha256']:
        raise ValueError('固定代狀態SHA不符')
    records = json.loads(gzip.decompress(raw))['records']
    states = {r['identity']['particle_id']: r for r in records}
    if len(states) != len(records):
        raise ValueError('狀態粒子身分重複')
    events = json.loads(Path(spec['events']).read_text())['events']
    ids = [e['event']['particle_id'] for e in events]
    if len(events) != count or len(set(ids)) != count:
        raise ValueError('失敗點數不符或粒子身分重複')
    for entry in events:
        event = entry['event']
        pid = event['particle_id']
        if pid not in requests or pid not in states:
            raise ValueError('事件粒子不屬於指定分片request或狀態')
        if any(event[k] != states[pid]['state'][k]
               for k in ('x_m', 'y_m', 'z_m', 'time_utc_ns')):
            raise ValueError('事件位置／時間與固定代失敗狀態不一致')
    return cp, state_path, events, states


def compare_point(request, event, hint):
    """以原投影公尺點比較三條定位路徑，不重寫重心公式、不使用最近點外插。

    全面搜尋只對所有三角形包圍盒做一次向量遮罩，再呼叫原精確判定方法；每個
    候選的原始定位回傳及權重都保存。幾何距離僅為量測，不據此分類沿岸或域外。
    """
    import numpy as np
    from shapely.geometry import Point

    provider = request.velocity
    manager = provider.manager
    mesh = manager.mesh
    x, y, z, utc = (event[k] for k in ('x_m', 'y_m', 'z_m', 'time_utc_ns'))
    point = np.array([x, y], dtype=np.float64)
    sample = provider.sample(x, y, z, utc, triangle_hint=hint)
    candidates = np.flatnonzero(
        np.all(point >= mesh.triangle_bbox_min - 1e-10, axis=1)
        & np.all(point <= mesh.triangle_bbox_max + 1e-10, axis=1))
    exact = []
    for triangle_id in candidates:
        location, weights = mesh._location_from_triangle(
            point, int(triangle_id), tolerance=1e-10)
        exact.append(dict(triangle_id=int(triangle_id), location=plain(location),
                          weights=plain(weights)))
    boundary = request.boundaries
    shape = Point(x, y)
    lon, lat = manager.projection.unproject(x, y)
    return plain(dict(
        particle_id=event['particle_id'], point=dict(x_m=x, y_m=y, z_m=z, time_utc_ns=utc),
        triangle_hint=hint, sample=sample, sample_valid=sample.valid,
        locate_with_hint=mesh.locate(x, y, triangle_hint=hint),
        locate_without_hint=mesh.locate(x, y), bbox_candidates=exact,
        flow_domain_covers=boundary.flow_domain.covers(shape),
        own_local_domain_covers=boundary.own_local_domain.covers(shape),
        flow_boundary_distance_m=boundary.flow_domain.boundary.distance(shape),
        local_boundary_distance_m=boundary.own_local_domain.boundary.distance(shape),
        flow_open_boundary_distance_m=(None if boundary.flow_open_boundary is None
                                       else boundary.flow_open_boundary.distance(shape)),
        lon=float(lon), lat=float(lat)))


def worker(spec, output):
    """只建立十個既有request，再查明示點數；不呼叫批次執行或恢復方法。"""
    sys.path.insert(0, str(Path(spec['project_root']) / 'src'))
    from lagrangian_backtracking import cli, runtime
    from lagrangian_backtracking import mesh as mesh_module
    from lagrangian_backtracking.config import load_config
    from lagrangian_backtracking.manifests import resolve_manifest_path
    from lagrangian_backtracking.runner import iter_run_units

    started = time.monotonic()
    sources = {}
    for module in (cli, runtime, mesh_module):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path(spec['project_root']).resolve() / 'src'):
            raise ValueError('模型模組不在指定乾淨版本')
        sources[str(path)] = digest(path)
    print('setup_start', flush=True)
    controller = cli.open_worker_controller(
        Path(spec['workspace']), config_path=Path(spec['config']),
        ocm_native_root=Path(spec['ocm_native_root']),
        ocm_reconstruction_root=Path(spec['ocm_reconstruction_root']),
        checkpoint_root=Path(spec['checkpoint_root']), continuous_time_fallback=True,
        _defer_full_validation=True,
        _private_dynamic_bootstrap=runtime._DEFERRED_VALIDATION_PRIVATE_TOKEN)
    shard = controller._shard(spec['shard_id'])
    units = tuple(iter_run_units(shard, master_seed=controller.plan['master_seed'],
                                random_stream_id=controller.plan.get('random_stream_id')))
    if len(units) != 10:
        raise ValueError('只允許指定十粒子的原分片')
    requests = {u.particle_id: controller.request_factory(u) for u in units}
    cp, state_path, events, states = bound_failure_points(spec, requests)
    result = []
    for entry in events:
        event = entry['event']
        pid = event['particle_id']
        row = compare_point(requests[pid], event, states[pid]['triangle_hint'])
        result.append(row)
        print(json.dumps(dict(stage='point', **row)), flush=True)
    manager = requests[events[0]['event']['particle_id']].velocity.manager
    if any(r.velocity.manager is not manager for r in requests.values()):
        raise ValueError('十個request未共用同一原流場管理器')
    # 來源僅列明示設定、網格小型靜態檔及幾何manifest；不雜湊整批月份陣列。
    config = load_config(Path(spec['config']), formal_release=False)
    for ref in config.geometry.values():
        if isinstance(ref, str):
            path = resolve_manifest_path(Path(spec['config']), ref)
            if path.is_file():
                sources[str(path)] = digest(path)
    grid = Path(spec['ocm_native_root']) / manager.flow_domain_id / 'grid'
    for path in grid.iterdir():
        if path.is_file():
            sources[str(path)] = digest(path)
    for path in (Path(spec['config']), Path(spec['workspace']) / 'input_inventory.json',
                 Path(spec['workspace']) / 'normalized_config.json', Path(spec['events']),
                 cp / 'checkpoint.json', state_path):
        sources[str(path)] = digest(path)
    projection_wkt = manager.projection._forward.target_crs.to_wkt()
    payload = dict(formal_result=False, no_advance=True, no_rng_consumption=True,
                   diagnostic_script_commit=spec.get('diagnostic_script_commit'),
                   diagnostic_script_sha256=digest(__file__),
                   expected_sequence=checkpoint_scope(spec)[1], expected_point_count=len(events),
                   setup_validation='startup_bindings_only', actual_commit=spec['expected_commit'],
                   run_id=controller.plan['run_id'], binding=plain(controller._binding(shard)),
                   node_count=len(manager.mesh.node_xy),
                   triangle_count=len(manager.mesh.triangle_nodes),
                   flow_domain_id=manager.flow_domain_id, source_sha256=sources,
                   projection_wkt=projection_wkt,
                   projection_sha256=hashlib.sha256(projection_wkt.encode()).hexdigest(),
                   flow_domain_wkb_sha256=hashlib.sha256(requests[events[0]['event']['particle_id']]
                                                       .boundaries.flow_domain.wkb).hexdigest(),
                   points=result, elapsed_seconds=time.monotonic() - started)
    (output / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def main():
    """檢查RAM與版本後，用既有安全清理helper限制單一新診斷程序180秒。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--internal-worker', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    checkpoint_scope(spec)  # 啟動子程序前先拒絕歧義或超界範圍，避免白做流場初始化。
    if args.internal_worker:
        worker(spec, args.output)
        return
    root = Path(spec['project_root'])
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
    if actual != spec['expected_commit'] or dirty.strip():
        raise ValueError('拒絕非明示乾淨模型版本')
    output = args.output.resolve()
    if not output.is_relative_to('/data/LBT/operations') or any(
        output.is_relative_to(Path(spec[k]).resolve())
        for k in ('project_root', 'workspace', 'checkpoint_root')):
        raise ValueError('診斷只能寫獨立NFS操作目錄')
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))) * 1024
    if available < 30 * 1024**3:
        raise RuntimeError('可用RAM低於30GiB，不啟動')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    for key in ('TMPDIR', 'TMP', 'TEMP', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
        path = output / key.lower()
        path.mkdir()
        env[key] = str(path)
    sys.path.insert(0, str(root / 'scripts'))
    from diagnose_batch_readonly import stop_owned
    plan = Path(spec['workspace']) / 'run_plan.json'
    before = digest(plan)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--spec', str(args.spec),
               '--output', str(output), '--internal-worker']
    with (output / 'stdout.log').open('x') as stream:
        process = subprocess.Popen(command, env=env, cwd=root, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        (output / 'launch.json').write_text(json.dumps(dict(
            pid=process.pid, command=command, script_sha256=digest(__file__),
            spec=spec, plan_sha256=before, mem_available_bytes=available), indent=2))
        try:
            process.wait(timeout=180)
        finally:
            stop_owned(process)
            (output / 'exit.json').write_text(json.dumps(dict(
                returncode=process.returncode, plan_sha256_after=digest(plan))))
    if process.returncode != 0 or digest(plan) != before:
        raise RuntimeError('診斷未成功或原計畫雜湊改變，請讀原始log')


if __name__ == '__main__':
    main()
