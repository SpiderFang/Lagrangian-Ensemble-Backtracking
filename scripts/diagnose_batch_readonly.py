"""在獨立程序記憶體重現單片步進；不發布輸出、不寫檢查點或原執行進度。

此入口只供有界診斷，不是正式成果。數值模組必須來自明示的乾淨版本；原種子、
亂數狀態、流場、步長與原版批次平流器均保留。逾時只終止本入口建立的程序群組。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path


def digest(path):
    """計算明示小型來源檔的位元組雜湊；不展開整批歷史軌跡。"""
    return sha256(Path(path).read_bytes()).hexdigest()


def emit(stage, **values):
    """即時輸出階段與世界協調時間，避免卡住時緩衝區吞掉定位證據。"""
    print(json.dumps(dict(stage=stage, utc=datetime.now(UTC).isoformat(), **values),
                     ensure_ascii=False), flush=True)


def build_batch(controller, shard_id, checkpoint, batch_class):
    """只建構記憶體批次；續接時由既有載入器驗證指定檢查點及恢復原亂數。

    不使用控制器的 run_shard、restore 或 reconcile，也不取得發布用鎖。
    平流器不可省略，否則重現會錯誤回到逐粒子舊核心。
    """
    shard = controller._shard(shard_id)
    if controller.advection_batcher is None:
        raise ValueError("診斷必須注入原執行的批次平流器")
    binding = controller._binding(shard)
    options = dict(shard=shard, master_seed=controller.plan['master_seed'],
                   random_stream_id=controller.plan.get('random_stream_id'),
                   request_factory=controller.request_factory,
                   active_chunk_size=controller.plan.get('active_chunk_size'),
                   advection_batcher=controller.advection_batcher)
    emit('binding', binding=asdict(binding), checkpoint=str(checkpoint) if checkpoint else None)
    if checkpoint is not None:
        return batch_class.from_checkpoint(checkpoint, expected_binding=binding, **options)
    return batch_class(**options)


def summarize(batch, sweep, started, report=None, setup_validation='startup'):
    """彙總步數／年齡及程序累計快取、原版批次計數，供辨識建場或快取反覆淘汰。

    快取計數是此次診斷程序累計值，不當作正式run進度；不輸出位置或整份軌跡。
    """
    executions = [runtime.execution for runtime in batch.runtimes]
    steps = [item.step_count for item in executions]
    ages = [item.state.age_seconds for item in executions]
    times = [item.state.time_utc_ns for item in executions]
    report = report or {}
    cache = report.get('resolver_report', {})
    emit('advance', setup_validation=setup_validation,
         sweep=sweep, elapsed_seconds=time.monotonic() - started,
         particle_steps=sum(steps), step_min=min(steps), step_max=max(steps),
         age_seconds_min=min(ages), age_seconds_max=max(ages),
         model_utc_ns_min=min(times), model_utc_ns_max=max(times), terminal=batch.terminal,
         cache={key: cache.get(key) for key in (
             'cache_hits', 'cache_misses', 'frame_builds', 'frame_evictions', 'cached_bytes')},
         native_completed=report.get('native_completed_particle_steps'),
         fallback=report.get('fallback_particle_steps'), max_group_size=report.get('max_group_size'),
         group_size_histogram=report.get('group_size_histogram'))


def worker(args):
    """開啟唯讀控制器並最多逐次推進指定掃描數；結果只留在診斷目錄。

    預設startup以既有私有能力執行啟動繫結驗證，保留不可變輸入、設定、種子
    與路徑檢核；full則不傳私有能力，讓控制器完整驗證歷史，以比較前置耗時。
    兩種模式均只在記憶體推進，明示檢查點恢復仍驗該片鏈，不能發布成果。
    靜態驗證、建構／載入與實際步進分開計時。由父層py-spy以祖先權限非停止式
    採樣；不啟用faulthandler或cProfile，避免混用診斷器。不得附加原計算程序。
    """
    sys.path.insert(0, str(args.project_root / 'src'))
    import lagrangian_backtracking.cli as cli
    import lagrangian_backtracking.production as production

    mode = getattr(args, 'setup_validation', 'startup')
    if mode not in {'full', 'startup'}:
        raise ValueError('setup_validation只允許full或startup')

    with (args.output / 'numerical-pid.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp()), stream)
    for module in (cli, production):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to((args.project_root / 'src').resolve()):
            raise ValueError("數值模組來源不是明示版本")
        emit('module_source', setup_validation=mode,
             module=module.__name__, path=str(path), sha256=digest(path))
    start = time.monotonic()
    emit('setup_start', setup_validation=mode, diagnostic_startup_bindings_only=mode == 'startup',
         full_history_validation_requested=mode == 'full', formal_result=False)
    # full不傳任何私有參數，保留公開控制器預設完整static驗證；不另呼叫run_shard。
    # startup能力只限本受控、完全不發布結果的重現，並非正式入口的通用略過旗標。
    validation_kwargs = {}
    if mode == 'startup':
        from lagrangian_backtracking.runtime import _DEFERRED_VALIDATION_PRIVATE_TOKEN

        validation_kwargs = dict(_defer_full_validation=True,
                                 _private_dynamic_bootstrap=_DEFERRED_VALIDATION_PRIVATE_TOKEN)
    controller = cli.open_worker_controller(
        args.workspace, config_path=args.config, ocm_native_root=args.ocm_native_root,
        ocm_reconstruction_root=args.ocm_reconstruction_root,
        checkpoint_root=args.checkpoint_root, continuous_time_fallback=True,
        **validation_kwargs)
    emit('setup_complete', setup_validation=mode, elapsed_seconds=time.monotonic() - start,
         full_history_validation_not_performed=mode == 'startup')
    restore_start = time.monotonic()
    emit('restore_start' if args.checkpoint else 'initialization_start', setup_validation=mode)
    batch = build_batch(controller, args.shard_id, args.checkpoint, production.ProductionBatch)
    emit('batch_ready', setup_validation=mode, restore_seconds=time.monotonic() - restore_start,
         from_checkpoint=args.checkpoint is not None)
    advance_start = time.monotonic()
    summarize(batch, 0, advance_start, controller.advection_reporter(), mode)
    for sweep in range(1, args.max_sweeps + 1):
        if batch.terminal:
            break
        batch.advance(1)
        if sweep % 10 == 0 or batch.terminal:
            summarize(batch, sweep, advance_start, controller.advection_reporter(), mode)
    emit('diagnostic_return', setup_validation=mode, elapsed_seconds=time.monotonic() - start,
         native_report=controller.advection_reporter(), formal_result=False)
    return 0


def stop_owned(process, child_pid=None):
    """硬逾時只清理由 Popen 新建且群組編號等於自身PID的診斷子程序。

    不搜尋模型程序。py-spy正常完成指定秒數採樣後，可能留下仍在計算的診斷子程序；
    此時僅接受本次獨立目錄記錄的child_pid，且再次確認它仍在新建群組內。
    """
    if process.poll() is None:
        if os.getpgid(process.pid) != process.pid or process.pid == os.getpgrp():
            raise RuntimeError("診斷程序群組身分不符，拒絕送訊號")
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    elif child_pid is not None:
        with suppress(ProcessLookupError):
            if os.getpgid(child_pid) != process.pid or process.pid == os.getpgrp():
                raise RuntimeError("診斷子程序不在自建群組，拒絕清理")
            os.killpg(process.pid, signal.SIGKILL)


def profiler_command(py_spy, child_command, output, profiling_seconds):
    """組成祖先非停止式採樣命令；僅呼叫新子程序，不附加任何既有模型PID。"""
    return [str(py_spy.resolve(strict=True)), 'record', '--nonblocking',
            '--rate', '5', '--format', 'raw', '--full-filenames',
            '--duration', str(profiling_seconds),
            '--output', str(output / 'profile.raw'), '--', *child_command]


def main(argv=None):
    """建立唯一NFS診斷目錄及單一子程序，保存来源證據並限制總時長120–180秒。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'config', 'project-root', 'checkpoint-root', 'ocm-native-root',
                 'ocm-reconstruction-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--shard-id', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--setup-validation', choices=('full', 'startup'), default='startup',
                        help='僅診斷控制器前置驗證：full完整歷史；startup啟動繫結（預設）')
    parser.add_argument('--py-spy', required=True, type=Path,
                        help='獨立工具環境的py-spy絕對路徑；只由它啟動新診斷子程序')
    parser.add_argument('--max-sweeps', type=int, default=1000)
    parser.add_argument('--timeout-seconds', type=int, default=150)
    parser.add_argument('--profiling-seconds', type=int, default=60,
                        help='採樣30–150整數秒，須比硬上限至少少20秒；預設60秒')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_sweeps <= 1000 or not 120 <= args.timeout_seconds <= 180:
        raise ValueError("診斷只允許1–1000次掃描及120–180秒硬上限")
    # 預留至少20秒讓採樣器寫出證據及父層清理；不得靠延長採樣解除180秒上限。
    if not 30 <= args.profiling_seconds <= 150 or args.profiling_seconds > args.timeout_seconds - 20:
        raise ValueError('採樣秒數須為30–150且不超過硬上限減20秒')
    if args.internal_worker:
        return worker(args)
    output = args.output.resolve()
    # 操作證據不能放進受根目錄白名單約束的原run、檢查點或計算checkout。
    if not output.is_relative_to(Path('/data/LBT/operations')):
        raise ValueError("診斷輸出必須在NFS operations下")
    for protected in (args.workspace, args.checkpoint_root, args.project_root):
        if output.is_relative_to(protected.resolve()):
            raise ValueError("不得將診斷輸出放進原執行資料或程式")
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.project_root,
                                     text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=args.project_root,
                                    text=True).strip()
    if actual != args.expected_commit or dirty:
        raise ValueError("診斷來源必須為核准乾淨版本")
    plan_path = args.workspace / 'run_plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['experiment_case_id'] != 'no_stokes':
        raise ValueError("本入口限此次純流診斷")
    if args.checkpoint and not args.checkpoint.resolve().is_relative_to(args.checkpoint_root.resolve()):
        raise ValueError("檢查點不在明示唯讀根目錄")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(args.project_root / 'src'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    for key in ('TMPDIR', 'TMP', 'TEMP', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR',
                'XDG_CACHE_HOME', 'UV_CACHE_DIR'):
        path = output / key.lower()
        path.mkdir()
        env[key] = str(path)
    child_command = [sys.executable, '-u', str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:]), '--internal-worker']
    # 預設60秒，自訂時以經核對秒數自動落盤；外層仍有150秒預設／180秒最大硬上限。
    # 不使用--pid附加原worker，py-spy是新診斷Python的直接父程序。
    command = profiler_command(args.py_spy, child_command, output, args.profiling_seconds)
    evidence = dict(policy='READ_ONLY_IN_MEMORY_DIAGNOSTIC_NOT_RESULTS', actual_commit=actual,
                    setup_validation=args.setup_validation,
                    diagnostic_startup_bindings_only=args.setup_validation == 'startup',
                    full_history_validation_requested=args.setup_validation == 'full',
                    full_history_validation_not_performed=(
                        True if args.setup_validation == 'startup' else None),
                    formal_result=False,
                    profiler='py-spy nonblocking ancestor sampling; no faulthandler/cProfile',
                    profiling_seconds=args.profiling_seconds, profiling_rate_hz=5,
                    project_root=str(args.project_root), script_path=str(Path(__file__).resolve()),
                    script_sha256=digest(__file__),
                    plan_sha256=digest(plan_path), run_id=plan['run_id'], shard_id=args.shard_id,
                    checkpoint=str(args.checkpoint) if args.checkpoint else None,
                    checkpoint_sha256=(digest(args.checkpoint / 'checkpoint.json')
                                       if args.checkpoint else None),
                    command=command)
    with (output / 'provenance.json').open('x') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    process = None
    started = time.monotonic()
    timed_out = False
    try:
        with (output / 'stdout.jsonl').open('x') as stdout, (output / 'stacks.stderr').open('x') as stderr:
            process = subprocess.Popen(command, cwd=args.project_root, env=env,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            emit('diagnostic_started', pid=process.pid, output=str(output))
            with (output / 'pid.json').open('x') as stream:
                json.dump(dict(pid=process.pid, pgid=process.pid), stream)
            try:
                process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        if process is not None:
            child_path = output / 'numerical-pid.json'
            child_pid = json.loads(child_path.read_text())['pid'] if child_path.exists() else None
            stop_owned(process, child_pid=child_pid)
    with (output / 'result.json').open('x') as stream:
        json.dump(dict(timed_out=timed_out, returncode=process.returncode,
                       elapsed_seconds=time.monotonic() - started,
                       plan_sha256_after=digest(plan_path), formal_result=False), stream)
    return 124 if timed_out else process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
