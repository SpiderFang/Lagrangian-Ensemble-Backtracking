"""建立 aggregate release payload 的唯讀、單次串流正式聚合管線。

本模組只負責在已完成的 pilot/formal run 上建立記憶體中的
``AggregateReleasePayload``；它不建立 partial/final release 目錄、不寫 progress，
也不讀取 OCM 或 NWW3 大型 forcing array。整個 static binding、source JSON
SHA-256、trajectory shard 讀取、事件與 pathway 累加，以及 payload constructor 都
在同一個 run gate exclusive lock 內完成，避免 run output 在聚合期間被替換。

空間資料的 x/y 格線與邊界長度一律使用公尺（m），旅行年齡與停留時間使用秒（s），
檔案時間與粒子時間使用世界協調時間（UTC）奈秒；事件網格軸為 ``(y_cell, x_cell)``，
pathway 首次進入直方圖軸為 ``(y_cell, x_cell, age_bin)``。缺值、dry/land、資料
時間缺口與數值失敗由上游 run／trajectory 契約保留，這裡不補零、不重新平流、不用
trajectory extent 猜測或裁切 AggregateSpec。

回傳結果只能保存指定條件下的條件式來源足跡或相對來源權重原始統計；尚未建立先驗、
似然與觀測驗證前，不得解讀為絕對來源機率或因果歸因。測試所用 synthetic fixture
只代表工程資料流驗證，不是真實 OCM／NWW 科學成果；正式科學結果仍須在具備已驗收
OCM schema 3 與 NWW3 schema 1 產品的 SERVER 上執行。
"""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import numpy as np

from .aggregate_release_payload import (
    AGGREGATE_RELEASE_SCHEMA_VERSION,
    AggregateReleasePayload,
)
from .aggregate_release_records import (
    AggregateShardBinding,
    ScenarioStratum,
    scenario_inputs_to_strata,
)
from .aggregate_spec import (
    AggregateSpec,
    validate_aggregate_spec_against_boundaries,
)
from .engine import ParticleResult
from .event_aggregation import EventAggregateAccumulator, aggregate_result_events
from .run_locking import RunLockBusyError, acquire_run_lock
from .run_validation import iter_complete_run_trajectory_shards
from .runtime import load_validated_run_static_inputs
from .streaming_aggregation import (
    StreamingPathwayAccumulator,
    stream_pathway_first_passage,
)

__all__ = ["build_aggregate_release_payload"]


_RUN_GATE_DIRECTORY_NAME: Final[str] = "locks"
_RUN_GATE_FILE_NAME: Final[str] = "run_gate.lock"
_SOURCE_FILE_ORDER: Final[tuple[str, ...]] = (
    "run_plan.json",
    "run_progress.json",
    "normalized_config.json",
    "input_inventory.json",
)
_READ_CHUNK_BYTES: Final[int] = 1024 * 1024
_RUN_KINDS: Final[frozenset[str]] = frozenset({"pilot", "formal"})


def _require_ordinary_directory(path: Path, *, label: str) -> None:
    """以 ``lstat`` 確認目錄節點本身是普通非 symlink directory。

    source root 與其 ``locks`` 目錄是 run gate 的安全邊界；這裡只檢查最後一個
    path node，不把祖先路徑的系統 alias 或合法 mount 解析差異誤當成 run topology
    錯誤。真正的例外會在公開入口統一隱藏，避免 SERVER 絕對路徑進入錯誤訊息。
    """

    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} 必須是普通非 symlink 目錄")


def _require_ordinary_file(path: Path, *, label: str) -> None:
    """以 ``lstat`` 確認固定輸入檔或 run lock 是普通非 symlink 檔案。"""

    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} 必須是普通非 symlink 檔案")


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    """比較檔案在開啟前後的 device/inode identity，不比較易變的 mtime。"""

    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _hash_source_file(source_root: Path, file_name: str) -> str:
    """以 no-follow descriptor 串流計算一份 run source JSON 的 SHA-256。

    ``run_plan.json``、``run_progress.json``、``normalized_config.json`` 與
    ``input_inventory.json`` 都是 static binding 的 exact bytes。先以 ``lstat``
    拒絕 symlink，再以可用的 ``O_NOFOLLOW`` 開啟並以 ``fstat`` 核對 device/inode；
    讀取後再次核對 descriptor identity，避免檔案在 open 前後被替換。資料不整份載入
    記憶體，只保留固定大小讀取 buffer，且不把 path 寫入 payload。
    """

    path = source_root / file_name
    before_open = os.lstat(path)
    if stat.S_ISLNK(before_open.st_mode) or not stat.S_ISREG(before_open.st_mode):
        raise ValueError("run source file 必須是普通非 symlink 檔案")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(opened.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or not _same_file_identity(before_open, opened)
        ):
            raise ValueError("run source file 在開啟時 identity 改變")

        digest = sha256()
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)

        after_read = os.fstat(descriptor)
        if (
            stat.S_ISLNK(after_read.st_mode)
            or not stat.S_ISREG(after_read.st_mode)
            or not _same_file_identity(opened, after_read)
        ):
            raise ValueError("run source file 在讀取後 identity 改變")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _hash_source_files(source_root: Path) -> dict[str, str]:
    """依固定順序產生四份 source JSON 的 bytes digest mapping。"""

    return {
        file_name: _hash_source_file(source_root, file_name)
        for file_name in _SOURCE_FILE_ORDER
    }


def _build_site_metric_centers(
    config: Any,
    geometries: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """由 config 的 study-site→region→domain 關係建立明示 WGS84 投影中心。

    ``AggregateSpec`` 的 boundary validator 需要每站 ``(lon_deg, lat_deg)`` 中心，
    但 pipeline 不得從 ``DomainProjection`` 私有狀態反推。這裡先確認
    ``study_sites`` 的站點識別唯一，再以該站的 ``analysis_region_id`` 唯一找到
    ``config.domains``，中心只取自正式設定的 ``center_lonlat``；經緯度只作投影
    定義與資料交換，後續統計仍在公尺制座標執行。
    """

    site_to_region: dict[str, str] = {}
    for site in config.study_sites:
        site_id = site.study_site_id
        if site_id in site_to_region:
            raise ValueError("config study_sites 的 study_site_id 不得重複")
        site_to_region[site_id] = site.analysis_region_id

    domains_by_region: dict[str, Any] = {}
    for domain in config.domains:
        region_id = domain.analysis_region_id
        if region_id in domains_by_region:
            raise ValueError("config domains 的 analysis_region_id 不得重複")
        domains_by_region[region_id] = domain

    centers: dict[str, tuple[float, float]] = {}
    for site_id in geometries:
        region_id = site_to_region.get(site_id)
        if region_id is None:
            raise ValueError("geometry site 必須能由 config study_sites 唯一解析")
        domain = domains_by_region.get(region_id)
        if domain is None:
            raise ValueError("study site 的 analysis_region_id 必須對應唯一 config domain")
        center = domain.center_lonlat
        if len(center) != 2:
            raise ValueError("config domain center_lonlat 必須是二元素經緯度")
        centers[site_id] = (float(center[0]), float(center[1]))
    return centers


def _age_edges_from_spec(spec: AggregateSpec) -> np.ndarray:
    """只由 AggregateSpec 建立共享的有限、嚴格遞增 float64 age edges。"""

    edges = np.array(spec.age_bin_edges_seconds, dtype=np.float64, copy=True)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("spec age edges 必須是至少兩點的一維陣列")
    if not np.all(np.isfinite(edges)) or edges[0] != 0.0:
        raise ValueError("spec age edges 必須有限且從 0 秒開始")
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("spec age edges 必須嚴格遞增")
    edges.setflags(write=False)
    return edges


def _grid_cell_count(spec: AggregateSpec, site_id: str, axis: str) -> int:
    """依 AggregateSpec 既有 alignment 語意取得單站 x 或 y cell 數。"""

    grid = spec.site_grids[site_id]
    if axis == "x":
        lower, upper = grid.x_min_m, grid.x_max_m
    else:
        lower, upper = grid.y_min_m, grid.y_max_m
    cell_size = float(spec.grid_cell_size_m)
    # 與 AggregateSpec／payload constructor 使用同一套原始數值相減與 round 語意；
    # 不能先把端點各自轉換後再另創一套 cell 對齊規則，否則尾端格線可能和公開
    # payload 驗證器產生不同的 shape。真正的有限性與嚴格遞增會在格線重建時檢查。
    width = upper - lower
    ratio = width / cell_size
    count = int(round(ratio))
    if count <= 0 or not math.isclose(ratio, count, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"site {site_id} 的 {axis} axis 無法依 spec alignment 建立 cell")
    return count


def _canonical_metric_edges(
    minimum_m: int | float,
    maximum_m: int | float,
    cell_size_m: int | float,
    cell_count: int,
    *,
    label: str,
) -> np.ndarray:
    """以 ``min + arange*cell`` 重建公尺制格線並固定最後一點為 max。"""

    minimum = float(minimum_m)
    maximum = float(maximum_m)
    cell_size = float(cell_size_m)
    if not all(math.isfinite(value) for value in (minimum, maximum, cell_size)):
        raise ValueError(f"{label} 必須是有限公尺數值")
    if cell_count <= 0 or cell_size <= 0.0:
        raise ValueError(f"{label} 的 cell count 與 cell size 必須為正")
    edges = minimum + np.arange(cell_count + 1, dtype=np.float64) * cell_size
    if edges.ndim != 1 or edges.shape != (cell_count + 1,):
        raise ValueError(f"{label} 格線 shape 不符合 spec cell 數")
    edges[-1] = maximum
    if not np.all(np.isfinite(edges)) or not np.all(edges[1:] > edges[:-1]):
        raise ValueError(f"{label} 必須是有限且嚴格遞增的公尺格線")
    edges.setflags(write=False)
    return edges


def _canonical_pathway_edges(spec: AggregateSpec) -> dict[str, dict[str, np.ndarray]]:
    """依每站 spec 矩形建立 pathway 使用的 canonical x/y 公尺格線。"""

    result: dict[str, dict[str, np.ndarray]] = {}
    for site_id, grid in spec.site_grids.items():
        x_count = _grid_cell_count(spec, site_id, "x")
        y_count = _grid_cell_count(spec, site_id, "y")
        result[site_id] = {
            "x_edges_m": _canonical_metric_edges(
                grid.x_min_m,
                grid.x_max_m,
                spec.grid_cell_size_m,
                x_count,
                label=f"site {site_id} x",
            ),
            "y_edges_m": _canonical_metric_edges(
                grid.y_min_m,
                grid.y_max_m,
                spec.grid_cell_size_m,
                y_count,
                label=f"site {site_id} y",
            ),
        }
    return result


def _validate_plan_strata_binding(
    static_scenarios: tuple[Any, ...],
    strata: tuple[ScenarioStratum, ...],
) -> None:
    """確認 strata 的數量與 scenario ID 逐列等於 static plan execution order。"""

    expected_ids = tuple(scenario.scenario_id for scenario in static_scenarios)
    actual_ids = tuple(stratum.scenario_id for stratum in strata)
    if len(actual_ids) != len(expected_ids) or actual_ids != expected_ids:
        raise ValueError("scenario_strata 必須逐列等於 static scenario execution order")


def _binding_from_record(
    record: Any,
    plan_row: Mapping[str, object],
    *,
    shard_index: int,
    static_scenarios: tuple[Any, ...],
) -> AggregateShardBinding:
    """核對單一 trajectory record 與 plan shard row，建立小型 binding record。

    iterator 已完成 run-level validation，但 aggregate payload 仍在同一個 lock 內重核
    對 plan row、scenario 半開區間與 record.scenarios；這個第二層 binding 可避免
    downstream writer 把錯誤的 trajectory record 綁到另一個 shard。只回傳九欄小型
    record，不保存該 shard 的 ParticleResult 或完整 scenario mapping。
    """

    required_plan_fields = (
        "shard_id",
        "scenario_start_index",
        "scenario_stop_index",
        "scenario_count",
        "particle_count",
    )
    if any(field not in plan_row for field in required_plan_fields):
        raise ValueError("run plan shard row 缺少 aggregate binding 欄位")
    expected_shard_id = plan_row["shard_id"]
    start = plan_row["scenario_start_index"]
    stop = plan_row["scenario_stop_index"]
    scenario_count = plan_row["scenario_count"]
    particle_count = plan_row["particle_count"]
    if type(expected_shard_id) is not str:
        raise ValueError("run plan shard_id 必須是原生字串")
    if any(type(value) is not int for value in (start, stop, scenario_count, particle_count)):
        raise ValueError("run plan shard range/count 必須是原生整數")
    if stop - start != scenario_count or start < 0 or stop <= start:
        raise ValueError("run plan shard range/count 不一致")
    if record.shard_id != expected_shard_id:
        raise ValueError(f"trajectory shard_id 與 plan row 不一致：index={shard_index}")
    if record.scenario_start_index != start or record.scenario_stop_index != stop:
        raise ValueError("trajectory scenario range 與 plan 不一致")
    if record.particle_count != particle_count:
        raise ValueError("trajectory particle_count 與 plan 不一致")

    expected_scenarios = static_scenarios[start:stop]
    try:
        record_scenarios = record.scenarios
        if type(record_scenarios) is not tuple or record_scenarios != expected_scenarios:
            raise ValueError("trajectory record.scenarios 與 plan slice 不一致")
    except AttributeError as error:
        raise ValueError("trajectory record 缺少 scenarios") from error

    return AggregateShardBinding(
        shard_id=record.shard_id,
        scenario_start_index=record.scenario_start_index,
        scenario_stop_index=record.scenario_stop_index,
        output_relative_path=record.output_relative_path,
        trajectory_manifest_sha256=record.trajectory_manifest_sha256,
        particle_count=record.particle_count,
        observation_count=record.observation_count,
        event_count=record.event_count,
    )


def _scenarios_by_id(record_scenarios: tuple[Any, ...]) -> dict[str, Any]:
    """只由目前 shard 的 scenarios 建立 event validator 所需的小型 ID mapping。"""

    mapping: dict[str, Any] = {}
    for scenario in record_scenarios:
        scenario_id = scenario.scenario_id
        if scenario_id in mapping:
            raise ValueError("單一 trajectory shard 的 scenario_id 不得重複")
        mapping[scenario_id] = scenario
    return mapping


def _group_results_by_site(
    results: tuple[ParticleResult, ...],
    *,
    scenarios_by_id: Mapping[str, Any],
    site_ids: frozenset[str],
) -> dict[str, list[ParticleResult]]:
    """依當前 shard 每個 result 的 final-state site 分組並即時核對 identity。"""

    grouped: dict[str, list[ParticleResult]] = {}
    for result in results:
        if not isinstance(result, ParticleResult):
            raise ValueError("trajectory result 必須是 ParticleResult")
        state = result.final_state
        site_id = state.study_site_id
        if site_id not in site_ids:
            raise ValueError("trajectory result 引用未知 study site")
        scenario = scenarios_by_id.get(state.scenario_id)
        if scenario is None:
            raise ValueError("trajectory result 引用未知 scenario identity")
        if (
            state.study_site_id != scenario.study_site_id
            or state.analysis_region_id != scenario.analysis_region_id
            or state.receptor_id != scenario.receptor_id
        ):
            raise ValueError("trajectory final_state identity 與 scenario 不一致")
        grouped.setdefault(site_id, []).append(result)
    return grouped


def _build_payload_locked(
    *,
    source_root: Path,
    config_path: str | Path,
    spec: AggregateSpec,
    checkpoint_root: str | Path | None,
) -> AggregateReleasePayload:
    """在 caller 已持有 source run exclusive gate 時完成所有 payload 建構。"""

    # static helper 是 controller 與 aggregate pipeline 共用的唯一 static binding 入口；
    # 它本身不取 lock，因此這個呼叫必須留在本函式的 run gate 內，且不得在此重新建立
    # forcing manager、讀取 OCM/NWW array 或修改 progress/checkpoint。
    static_inputs = load_validated_run_static_inputs(
        source_root,
        config_path=config_path,
        checkpoint_root=checkpoint_root,
        require_complete=True,
    )
    plan = static_inputs.plan
    run_kind = plan["run_kind"]
    if type(run_kind) is not str or run_kind not in _RUN_KINDS:
        raise ValueError("aggregate payload 只接受 static helper 已驗證的 pilot/formal run")
    if spec.run_id != plan["run_id"]:
        raise ValueError("AggregateSpec.run_id 必須 exact 等於 static plan run_id")

    centers = _build_site_metric_centers(
        static_inputs.config,
        static_inputs.geometries,
    )
    validate_aggregate_spec_against_boundaries(
        spec,
        static_inputs.geometries,
        site_metric_centers_deg=centers,
    )

    strata = scenario_inputs_to_strata(
        static_inputs.scenario_inputs,
        formal=run_kind == "formal",
    )
    _validate_plan_strata_binding(static_inputs.scenario_inputs.scenarios, strata)

    # 先建立 source bytes 的 immutable provenance；第二次 hash 會在 payload constructor
    # 完成後執行，將「準備開始聚合」到「準備回傳」封閉在同一個 exclusive gate 裡。
    source_hashes = _hash_source_files(source_root)

    age_edges = _age_edges_from_spec(spec)
    pathway_edges = _canonical_pathway_edges(spec)
    site_ids = tuple(spec.site_grids)
    site_id_set = frozenset(site_ids)
    event_accumulator = EventAggregateAccumulator()
    pathway_accumulators = {
        site_id: StreamingPathwayAccumulator() for site_id in site_ids
    }
    shard_rows = plan["shards"]
    if not isinstance(shard_rows, (tuple, list)):
        raise ValueError("static plan shards 必須是固定順序序列")

    bindings: list[AggregateShardBinding] = []
    expected_start = 0
    total_particle_count = 0
    for shard_index, record in enumerate(
        iter_complete_run_trajectory_shards(
            source_root,
            checkpoint_root=checkpoint_root,
        )
    ):
        if shard_index >= len(shard_rows):
            raise ValueError("trajectory shard 數量超過 plan shard_count")
        plan_row = shard_rows[shard_index]
        if not isinstance(plan_row, Mapping):
            raise ValueError("static plan shard row 必須是 mapping")
        binding = _binding_from_record(
            record,
            plan_row,
            shard_index=shard_index,
            static_scenarios=static_inputs.scenario_inputs.scenarios,
        )
        if binding.scenario_start_index != expected_start:
            raise ValueError("trajectory shard range 未依 plan 順序連續")
        bindings.append(binding)
        expected_start = binding.scenario_stop_index
        total_particle_count += binding.particle_count

        scenarios_by_id = _scenarios_by_id(record.scenarios)
        event_chunk = aggregate_result_events(
            record.results,
            scenarios_by_id=scenarios_by_id,
            spec=spec,
            age_bin_edges_seconds=age_edges,
        )
        # event accumulator 立即複製並累加目前 chunk；不保存過往 shard 的 chunk 或
        # result，故 peak memory 只包括固定 aggregate topology 與當前 trajectory shard。
        event_accumulator.add(event_chunk)

        results_by_site = _group_results_by_site(
            record.results,
            scenarios_by_id=scenarios_by_id,
            site_ids=site_id_set,
        )
        for site_id in site_ids:
            current_results = results_by_site.get(site_id)
            if not current_results:
                continue
            pathway_chunk = stream_pathway_first_passage(
                current_results,
                x_edges_m=pathway_edges[site_id]["x_edges_m"],
                y_edges_m=pathway_edges[site_id]["y_edges_m"],
                age_bin_edges_seconds=age_edges,
            )
            pathway_accumulators[site_id].add(pathway_chunk)

    plan_shard_count = plan["shard_count"]
    scenario_count = plan["scenario_count"]
    particle_count = plan["particle_count"]
    if type(plan_shard_count) is not int or len(bindings) != plan_shard_count:
        raise ValueError("trajectory shard 數量必須 exact 等於 plan shard_count")
    if type(scenario_count) is not int or expected_start != scenario_count:
        raise ValueError("trajectory shard ranges 必須從 0 連續覆蓋 plan scenario_count")
    if type(particle_count) is not int or total_particle_count != particle_count:
        raise ValueError("trajectory particle_count 總和必須 exact 等於 plan particle_count")
    if event_accumulator.chunk_count < 1:
        raise ValueError("event aggregate 不得在零 shard 時 finalize")
    if any(
        pathway_accumulators[site_id].chunk_count < 1
        for site_id in site_ids
    ):
        raise ValueError("每個 spec site 的 pathway aggregate 都必須至少有一個 shard chunk")

    event_aggregate = event_accumulator.finalize()
    pathway_by_site = {
        site_id: pathway_accumulators[site_id].finalize() for site_id in site_ids
    }
    payload = AggregateReleasePayload(
        schema_version=AGGREGATE_RELEASE_SCHEMA_VERSION,
        run_id=plan["run_id"],
        run_kind=run_kind,
        experiment_case_id=plan["experiment_case_id"],
        members_per_scenario=plan["members_per_scenario"],
        config_hash=plan["config_hash"],
        checkpoint_input_binding_hash=plan["checkpoint_input_binding_hash"],
        source_run_plan_sha256=source_hashes["run_plan.json"],
        source_run_progress_sha256=source_hashes["run_progress.json"],
        source_normalized_config_sha256=source_hashes["normalized_config.json"],
        source_input_inventory_sha256=source_hashes["input_inventory.json"],
        aggregate_spec=spec,
        shard_bindings=tuple(bindings),
        scenario_strata=tuple(strata),
        event_aggregate=event_aggregate,
        pathway_by_site=pathway_by_site,
    )

    final_source_hashes = _hash_source_files(source_root)
    if final_source_hashes != source_hashes:
        raise ValueError("run source JSON 在 aggregate payload 建構期間改變")
    return payload


def build_aggregate_release_payload(
    *,
    source_run_root: str | Path,
    config_path: str | Path,
    aggregate_spec: AggregateSpec,
    checkpoint_root: str | Path | None = None,
) -> AggregateReleasePayload:
    """在 exclusive run gate 內串流建立不可變 aggregate release payload。

    Args:
        source_run_root: 已完成 pilot/formal run 的 workspace 目錄。最後一個目錄節點
            必須以 ``lstat`` 是普通非 symlink directory；函式只讀取其中的 immutable
            run 文件與已驗收 trajectory shard，不把此絕對位置保存到 payload。
        config_path: static loader 使用的已驗證專案設定路徑；設定、geometry、inventory
            與 dynamic initial-condition provenance 由共用 static binding helper 驗證。
        aggregate_spec: exact ``AggregateSpec``。其每站公尺格網、邊界 topology 與秒制
            age 軸決定所有下游 reducer 的固定拓撲，不從 trajectory extent 反推。
        checkpoint_root: 可選的 external checkpoint 根目錄，原樣傳給 static loader 與
            complete trajectory iterator；不在 payload 內保存 path，也不修改 checkpoint。

    Returns:
        已通過 static run binding、AggregateSpec/geometry binding、逐 shard identity、
        source SHA-256 二次確認與 payload cross-product constructor 的
        ``AggregateReleasePayload``。事件陣列軸為 ``(y, x)``，pathway histogram 軸為
        ``(y, x, age)``；x/y 距離使用公尺，age／停留時間使用秒。

    Raises:
        RunLockBusyError: ``locks/run_gate.lock`` 被其他程序持有時原樣傳出，讓 CLI
            能區分 contention 與資料錯誤。
        ValueError: 任何 source topology、static binding、spec／geometry、trajectory
            identity、計數守恆、reducer 或 payload constructor 錯誤；公開錯誤固定為
            ``aggregate release payload 建構失敗``，不攜帶 path 或底層 cause。

    Notes:
        exclusive run gate 必須涵蓋從 static binding 到 source hash、trajectory reader、
        reducer finalize 與二次 source hash 的完整生命週期；後續 writer 仍須在同一
        run gate 內重新驗證 current trajectory manifest，因為本函式回傳的 payload
        只是一個未發布的記憶體 snapshot。這個函式不建立 RuntimeRequestFactory 或
        ForcingWindowManager、不讀 OCM/NWW array、不重跑平流，也不因 synthetic 本機
        測試通過而宣稱真實 OCM/NWW 科學成果。
    """

    try:
        if type(aggregate_spec) is not AggregateSpec:
            raise ValueError("aggregate_spec 必須是 exact AggregateSpec")
        source_root = Path(source_run_root)
        _require_ordinary_directory(source_root, label="source run root")
        locks_root = source_root / _RUN_GATE_DIRECTORY_NAME
        _require_ordinary_directory(locks_root, label="run locks")
        lock_path = locks_root / _RUN_GATE_FILE_NAME
        _require_ordinary_file(lock_path, label="run gate lock")
        with acquire_run_lock(lock_path, mode="exclusive", blocking=False):
            return _build_payload_locked(
                source_root=source_root,
                config_path=config_path,
                spec=aggregate_spec,
                checkpoint_root=checkpoint_root,
            )
    except RunLockBusyError:
        # lock contention 是正常的併發控制結果；保留原始型別，不能與資料契約錯誤
        # 混成一般 ValueError，否則 caller 無法採取 retry/backoff 策略。
        raise
    except Exception:
        # 所有其他例外都可能攜帶 SERVER absolute path、JSON parser context 或底層
        # filesystem 詳情；公開 API 固定錯誤文字，避免部署資訊進入日志／artifact。
        raise ValueError("aggregate release payload 建構失敗") from None
