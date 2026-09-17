"""persistent-wet 受體候選池與可重現的隨機受體選樣。

正式母體先在每站建立通過 static-ocean、persistent-wet、核心／local 與 forcing
支援閘門的 source-face 候選池，再以獨立 seeded random stream 無放回抽取五個水平
face；每個水平 face 再以另一個獨立 stream 從完整有效水柱的 normalized fraction 開放
區間 ``(0, 1)`` 抽四個垂向位置。隨機抽樣的 rank 只是可重現的識別順序，不代表固定的
物理水層。舊版 deterministic maximin、固定座標與 10/40/70/near-bed helper 仍保留給
歷史／唯讀 loader 及相容測試，current formal validator 不允許以它們產生正式母體。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from shapely.geometry import Point, Polygon

from .config import (
    HORIZONTAL_RECEPTOR_SELECTION_POLICY_ID,
    RANDOM_VERTICAL_RECEPTOR_ID_PREFIX,
    RECEPTOR_SELECTION_SEED_POLICY_ID,
    VERTICAL_RECEPTOR_SELECTION_POLICY_ID,
)
from .geometry import deterministic_maximin
from .mesh import NativeMesh
from .scenarios import stable_identifier


@dataclass(frozen=True, slots=True)
class HorizontalReceptor:
    """一個站點的水平 receptor 位置與 OCM face provenance。"""

    horizontal_receptor_id: str
    study_site_id: str
    x_m: float
    y_m: float
    lon: float
    lat: float
    source_face_local_index: int
    source_face_global_index: int
    anchor_snap_distance_m: float | None


@dataclass(frozen=True, slots=True)
class HorizontalReceptorCandidatePool:
    """單站一次建立、可供多輪 deterministic 重選的水平候選池。

    candidate pool 已完成 persistent-wet、candidate polygon 與 boundary margin gate；
    陣列分別保存 source face local/global index、公尺制中心、經緯度與 tie-break key。
    所有 numpy 欄位都是 defensive copy 且設為唯讀，後續 NWW／OCM 垂向 blacklist 只能透過
    ``excluded_face_indices`` 傳入 selector，不能改寫原始候選池或重新掃描 Shapely
    geometry。經緯度只作 receptor 資料交換與 NWW 空間 gate，maximin 距離仍在公尺制
    ``candidate_xy_m`` 與 ``anchor_xy`` 上計算。
    """

    study_site_id: str
    anchor_xy: tuple[float, float]
    maximum_anchor_snap_distance_m: float | None
    candidate_face_local_indices: np.ndarray
    candidate_face_global_indices: np.ndarray
    candidate_xy_m: np.ndarray
    candidate_lon: np.ndarray
    candidate_lat: np.ndarray
    tie_break_keys: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True, slots=True)
class VerticalTarget:
    """單一 arrival/水平位置的實際 positive-up z 與 HAB。"""

    vertical_id: str
    target_fraction_below_surface: float | None
    z_m_positive_up: float
    height_above_bed_m: float
    bracket_span_m: float


@dataclass(frozen=True, slots=True)
class RandomVerticalDraw:
    """單一水平 face 的 seeded random 垂向抽樣結果。

    ``normalized_fraction_below_surface`` 是瞬時海面到海床之間的無因次比例，嚴格落在
    ``(0, 1)``；它不是 OCM 固定 layer 編號，也不應在報告中被解讀為上／中／近床物理
    層。``draw_order`` 只用來保存 deterministic stream 的輸出順序；實際 positive-up
    z 仍須在每個 arrival 由該時刻的 ``eta``、bed 與 zcor 雙側 bracket 換算。
    """

    vertical_id: str
    draw_order: int
    normalized_fraction_below_surface: float
    derived_seed_hex: str
    seed_derivation_sha256: str


def _validate_seed_text(value: str, field_name: str) -> str:
    """確認 seed 派生所需的識別文字非空，避免不同 stream 靜默共用。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必須是非空字串")
    return value


def derive_receptor_selection_seed(
    *,
    master_seed: int,
    study_site_id: str,
    design_version: str,
    selection_policy_id: str,
    stream_scope: str,
) -> tuple[int, str]:
    """以版本化欄位派生 128-bit PCG64DXSM seed 與完整 SHA-256。

    派生材料明確包含正式設定的 ``master_seed``、站點、設計版本、選樣 policy 與
    stream scope。水平 stream 以站點為 scope；垂向 stream 再加入 source face 的
    local/global identity，因此五站與每個 face 不會因共用一個全域 RNG 而互相消耗
    狀態。回傳的整數取 SHA-256 前 128 bit，並由呼叫端把完整 digest 與十六進位 seed
    寫入 manifest，讓相同設定可 byte-stable 重建、不同 seed 或 face binding 可稽核。
    """

    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(
        master_seed, (int, np.integer)
    ):
        raise ValueError("master_seed 必須是整數")
    if int(master_seed) < 0:
        raise ValueError("master_seed 不可為負數")
    material = {
        "design_version": _validate_seed_text(design_version, "design_version"),
        "master_seed": int(master_seed),
        "selection_policy_id": _validate_seed_text(selection_policy_id, "selection_policy_id"),
        "stream_scope": _validate_seed_text(stream_scope, "stream_scope"),
        "study_site_id": _validate_seed_text(study_site_id, "study_site_id"),
        "seed_policy": RECEPTOR_SELECTION_SEED_POLICY_ID,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256(encoded).hexdigest()
    return int(digest[:32], 16), digest


def candidate_pool_fingerprint(pool: HorizontalReceptorCandidatePool) -> str:
    """計算候選 face pool 的內容指紋，防止只保存 seed 卻遺失 pool 變更。

    fingerprint 使用 source face local/global index、投影中心與 mesh 經緯度的精確
    hexadecimal 浮點表示；它不保存大型 wetdry 陣列，但能在 manifest 中辨識同一個
    accepted OCM mesh 的候選集合是否被增刪或重新排序。候選排序依 source face identity
    固定，故同一 pool 即使由不同 caller 傳入也會產生相同指紋。
    """

    if not isinstance(pool, HorizontalReceptorCandidatePool):
        raise ValueError("pool 必須是 HorizontalReceptorCandidatePool")
    entries = []
    for local, global_index, xy, lon, lat in zip(
        pool.candidate_face_local_indices,
        pool.candidate_face_global_indices,
        pool.candidate_xy_m,
        pool.candidate_lon,
        pool.candidate_lat,
        strict=True,
    ):
        entries.append(
            {
                "local": int(local),
                "global": int(global_index),
                "x_hex": float(xy[0]).hex(),
                "y_hex": float(xy[1]).hex(),
                "lon_hex": float(lon).hex(),
                "lat_hex": float(lat).hex(),
            }
        )
    entries.sort(key=lambda item: (item["global"], item["local"]))
    encoded = json.dumps(
        {
            "study_site_id": pool.study_site_id,
            "candidate_faces": entries,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _excluded_face_set(excluded_face_indices: Iterable[int]) -> frozenset[int]:
    """將 exclusion 轉成不可變整數集合，並拒絕布林或非整數 face index。"""

    try:
        values = tuple(excluded_face_indices)
    except TypeError as exc:
        raise ValueError("excluded_face_indices 必須是整數 iterable") from exc
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError("excluded_face_indices 必須是整數 iterable")
    if any(int(value) < 0 for value in values):
        raise ValueError("excluded_face_indices 不可為負數")
    return frozenset(int(value) for value in values)


def select_horizontal_receptors_random_from_pool(
    pool: HorizontalReceptorCandidatePool,
    *,
    master_seed: int,
    design_version: str,
    selection_policy_id: str = HORIZONTAL_RECEPTOR_SELECTION_POLICY_ID,
    stream_scope: str = "horizontal",
    count: int = 5,
    excluded_face_indices: Iterable[int] = (),
) -> list[HorizontalReceptor]:
    """在已通過資料閘門的 pool 中以獨立 RNG 無放回抽取水平受體。

    候選先依 source face global/local identity 排序，再由 PCG64DXSM 依派生 seed 抽樣；
    因此同一設定與 pool 的結果和輸入 dictionary／呼叫順序無關。selector 不做 anchor
    first、maximin、最近點或補點；排除後少於 ``count`` 就直接 ``ValueError``，由
    input-build 將其轉為 fail-closed。回傳順序是 RNG draw order，metadata 應另存 draw
    order 與 pool fingerprint，不能把此順序誤當空間優先序。
    """

    if not isinstance(pool, HorizontalReceptorCandidatePool):
        raise ValueError("pool 必須是 HorizontalReceptorCandidatePool")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("count 必須為正整數")
    count_value = int(count)
    if count_value < 1:
        raise ValueError("count 必須為正整數")
    excluded = _excluded_face_set(excluded_face_indices)
    eligible_positions = [
        index
        for index, face_index in enumerate(pool.candidate_face_local_indices)
        if int(face_index) not in excluded
    ]
    eligible_positions.sort(
        key=lambda index: (
            int(pool.candidate_face_global_indices[index]),
            int(pool.candidate_face_local_indices[index]),
        )
    )
    if len(eligible_positions) < count_value:
        raise ValueError(
            f"{pool.study_site_id} seeded random 候選不足：需要 {count_value}，"
            f"實際 {len(eligible_positions)}"
        )
    seed, _digest = derive_receptor_selection_seed(
        master_seed=master_seed,
        study_site_id=pool.study_site_id.split(":", 1)[0],
        design_version=design_version,
        selection_policy_id=selection_policy_id,
        stream_scope=stream_scope,
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    selected_positions = np.atleast_1d(
        np.asarray(
            rng.choice(len(eligible_positions), size=count_value, replace=False), dtype=np.int64
        )
    )
    selected: list[HorizontalReceptor] = []
    for selected_position in selected_positions:
        position = eligible_positions[int(selected_position)]
        x_m, y_m = pool.candidate_xy_m[position]
        selected.append(
            HorizontalReceptor(
                horizontal_receptor_id=stable_identifier(
                    "hr",
                    [
                        pool.study_site_id.split(":", 1)[0],
                        str(int(pool.candidate_face_global_indices[position])),
                        f"{x_m:.6f}",
                        f"{y_m:.6f}",
                    ],
                ),
                study_site_id=pool.study_site_id.split(":", 1)[0],
                x_m=float(x_m),
                y_m=float(y_m),
                lon=float(pool.candidate_lon[position]),
                lat=float(pool.candidate_lat[position]),
                source_face_local_index=int(pool.candidate_face_local_indices[position]),
                source_face_global_index=int(pool.candidate_face_global_indices[position]),
                anchor_snap_distance_m=None,
            )
        )
    return selected


def validate_random_vertical_fractions(
    fractions: Sequence[float], *, count: int = 4
) -> tuple[float, ...]:
    """驗證垂向 random draw 僅落於完整水柱開放區間且彼此不重複。

    ``fractions`` 代表從瞬時海面往海床方向的 normalized fraction，不接受 0、1、NaN、
    無限值或重複值。這個獨立 validator 也供測試與 manifest loader 使用，讓 endpoint、
    duplicate 或非有限資料不會在後續 zcor 內插才以不明原因失敗。
    """

    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("垂向 random draw count 必須是正整數")
    count_value = int(count)
    try:
        raw_values = tuple(fractions)
    except TypeError as exc:
        raise ValueError(f"垂向 random draw 必須恰有 {count_value} 個 fraction") from exc
    if count_value < 1 or len(raw_values) != count_value:
        raise ValueError(f"垂向 random draw 必須恰有 {count_value} 個 fraction")
    try:
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("垂向 random fraction 必須是數值") from exc
    if not all(np.isfinite(value) and 0.0 < value < 1.0 for value in values):
        raise ValueError("垂向 random fraction 必須是 (0, 1) 內的有限值")
    if len(set(values)) != len(values):
        raise ValueError("垂向 random fraction 不可重複")
    return values


def sample_random_vertical_draws(
    *,
    master_seed: int,
    study_site_id: str,
    design_version: str,
    source_face_local_index: int,
    source_face_global_index: int,
    count: int = 4,
    selection_policy_id: str = VERTICAL_RECEPTOR_SELECTION_POLICY_ID,
) -> tuple[RandomVerticalDraw, ...]:
    """為一個水平 face 產生四個可重建的 random 垂向抽樣。

    每個 face 使用獨立 stream scope，並以 PCG64DXSM 均勻抽取完整水柱的開放區間；
    fractions 的檢查先於建立 draw identity。因為 actual z 會在每個 arrival 重新依
    ``eta``／bed／zcor bracket 計算，這裡只保存 normalized draw 及其 seed provenance，
    不預先把某一 arrival 的 z 值當成所有時次的固定深度。
    """

    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("垂向 random draw count 必須是正整數")
    count_value = int(count)
    if count_value < 1:
        raise ValueError("垂向 random draw count 必須是正整數")
    for value, field_name in (
        (source_face_local_index, "source_face_local_index"),
        (source_face_global_index, "source_face_global_index"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{field_name} 必須是整數")
        if int(value) < 0:
            raise ValueError(f"{field_name} 不可為負數")
    stream_scope = f"vertical:{int(source_face_global_index)}:{int(source_face_local_index)}"
    seed, digest = derive_receptor_selection_seed(
        master_seed=master_seed,
        study_site_id=study_site_id,
        design_version=design_version,
        selection_policy_id=selection_policy_id,
        stream_scope=stream_scope,
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    fractions = validate_random_vertical_fractions(rng.random(count_value), count=count_value)
    seed_hex = f"{seed:032x}"
    return tuple(
        RandomVerticalDraw(
            vertical_id=f"{RANDOM_VERTICAL_RECEPTOR_ID_PREFIX}_{order}",
            draw_order=order,
            normalized_fraction_below_surface=fraction,
            derived_seed_hex=seed_hex,
            seed_derivation_sha256=digest,
        )
        for order, fraction in enumerate(fractions)
    )


def _face_centers(mesh: NativeMesh) -> np.ndarray:
    """依每個原生 face 的 3/4 個有效 node 計算公尺中心。"""

    centers = np.empty((mesh.face_nodes_local.shape[0], 2), dtype=np.float64)
    for face_index, count in enumerate(mesh.face_node_count):
        nodes = mesh.face_nodes_local[face_index, : int(count)]
        centers[face_index] = mesh.node_xy[nodes].mean(axis=0)
    return centers


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    """建立候選池專用的唯讀 array copy，避免 excluded 狀態污染 immutable pool。"""

    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def prepare_horizontal_receptor_candidates(
    *,
    study_site_id: str,
    mesh: NativeMesh,
    candidate_polygon_metric: Polygon,
    anchor_xy: tuple[float, float],
    wetdry_at_arrivals: np.ndarray,
    boundary_margin_m: float = 0.0,
    wet_value: float = 0.0,
    maximum_anchor_snap_distance_m: float | None = None,
) -> HorizontalReceptorCandidatePool:
    """一次掃描 geometry，建立單站 persistent-wet 的 immutable 水平候選池。

    ``wetdry_at_arrivals`` shape 為 ``(arrival,source_face)``；50 個時次任一非有限或非 wet
    即剔除。``boundary_margin_m`` 同時避開 candidate polygon 外界與被 ocean clipping
    形成的岸線。這個階段不接受 count，也不進行 maximin；候選數量與 NWW／垂向
    blacklist 由 ``select_horizontal_receptors_from_pool`` 在不重做 geometry scan 的
    情況下處理。pool 保存的座標與 tie-break key 已經是原 wrapper 的同一份資料，故
    不會因重選輪次或 excluded set 的輸入順序改變結果。
    """

    wetdry = np.asarray(wetdry_at_arrivals, dtype=np.float64)
    if wetdry.ndim != 2 or wetdry.shape[1] != mesh.face_nodes_local.shape[0]:
        raise ValueError("wetdry_at_arrivals 必須是 (arrival,source_face)")
    if boundary_margin_m < 0:
        raise ValueError("boundary margin 不可為負")
    persistent_wet = np.all(np.isfinite(wetdry) & np.isclose(wetdry, wet_value, atol=0.1), axis=0)
    centers = _face_centers(mesh)
    candidate_indices: list[int] = []
    for face_index, (x_m, y_m) in enumerate(centers):
        point = Point(float(x_m), float(y_m))
        if not persistent_wet[face_index] or not candidate_polygon_metric.covers(point):
            continue
        if boundary_margin_m > 0 and point.distance(candidate_polygon_metric.boundary) < boundary_margin_m:
            continue
        candidate_indices.append(face_index)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_xy = centers[indices]
    lon = np.empty(indices.size, dtype=np.float64)
    lat = np.empty(indices.size, dtype=np.float64)
    for position, face_index in enumerate(indices):
        node_count = int(mesh.face_node_count[face_index])
        nodes = mesh.face_nodes_local[face_index, :node_count]
        lon[position] = float(mesh.node_lon[nodes].mean())
        lat[position] = float(mesh.node_lat[nodes].mean())
    tie_keys = [
        (float(lon[index]), float(lat[index]), int(mesh.source_face_global_index[face_index]))
        for index, face_index in enumerate(indices)
    ]
    return HorizontalReceptorCandidatePool(
        study_site_id=study_site_id,
        anchor_xy=(float(anchor_xy[0]), float(anchor_xy[1])),
        maximum_anchor_snap_distance_m=(
            float(maximum_anchor_snap_distance_m)
            if maximum_anchor_snap_distance_m is not None
            else None
        ),
        candidate_face_local_indices=_readonly_copy(indices),
        candidate_face_global_indices=_readonly_copy(mesh.source_face_global_index[indices]),
        candidate_xy_m=_readonly_copy(candidate_xy),
        candidate_lon=_readonly_copy(lon),
        candidate_lat=_readonly_copy(lat),
        tie_break_keys=tuple(tie_keys),
    )


def select_horizontal_receptors_from_pool(
    pool: HorizontalReceptorCandidatePool,
    *,
    count: int = 5,
    excluded_face_indices: Iterable[int] = (),
) -> list[HorizontalReceptor]:
    """在既有 immutable candidate pool 上重跑歷史 anchor-first deterministic maximin。

    ``excluded_face_indices`` 是目前 study site 已被 NWW 或 OCM 垂向 gate 淘汰的 source
    face local index 集合；selector 只建立一份布林選取 mask，不修改 pool 的任何 array。
    因此每輪只做數值 maximin，不重新建立 Shapely Point、重新判斷 polygon 或讀取 wetdry。
    候選不足仍以 ``ValueError`` fail closed；未被排除的候選、tie-break、anchor snap 與
    receptor ID 格式均保持原 ``select_horizontal_receptors`` 的結果。current formal 不可
    呼叫此固定／maximin helper，而應使用 ``select_horizontal_receptors_random_from_pool``；
    保留本函式是為了讀取歷史 pilot 與測試舊 artifact。
    """

    if not isinstance(pool, HorizontalReceptorCandidatePool):
        raise ValueError("pool 必須是 HorizontalReceptorCandidatePool")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("count 必須為正整數")
    count_value = int(count)
    if count_value < 1:
        raise ValueError("count 必須為正整數")
    try:
        excluded_values = tuple(excluded_face_indices)
    except TypeError as exc:
        raise ValueError("excluded_face_indices 必須是整數 iterable") from exc
    if any(
        isinstance(face_index, (bool, np.bool_))
        or not isinstance(face_index, (int, np.integer))
        for face_index in excluded_values
    ):
        raise ValueError("excluded_face_indices 必須是整數 iterable")
    excluded = frozenset(int(face_index) for face_index in excluded_values)
    include = np.asarray(
        [int(face_index) not in excluded for face_index in pool.candidate_face_local_indices],
        dtype=bool,
    )
    remaining_count = int(np.count_nonzero(include))
    if remaining_count < count_value:
        raise ValueError(
            f"{pool.study_site_id} persistent-wet/margin 候選不足：需要 {count_value}，實際 {remaining_count}"
        )
    candidate_xy = pool.candidate_xy_m[include]
    candidate_lon = pool.candidate_lon[include]
    candidate_lat = pool.candidate_lat[include]
    candidate_local_indices = pool.candidate_face_local_indices[include]
    candidate_global_indices = pool.candidate_face_global_indices[include]
    candidate_tie_keys = tuple(
        key for key, included in zip(pool.tie_break_keys, include, strict=True) if included
    )
    local_selected = deterministic_maximin(
        candidate_xy,
        count=count_value,
        anchor_xy=pool.anchor_xy,
        tie_break_keys=candidate_tie_keys,
    )
    selected: list[HorizontalReceptor] = []
    anchor = np.asarray(pool.anchor_xy, dtype=np.float64)
    for order, local_index in enumerate(local_selected):
        index = int(local_index)
        x_m, y_m = candidate_xy[index]
        snap_distance = float(np.linalg.norm(candidate_xy[index] - anchor)) if order == 0 else None
        if (
            order == 0
            and pool.maximum_anchor_snap_distance_m is not None
            and snap_distance > pool.maximum_anchor_snap_distance_m
        ):
            raise ValueError(
                f"{pool.study_site_id} anchor snap {snap_distance:.3f} m "
                f"超過 {pool.maximum_anchor_snap_distance_m:.3f} m"
            )
        selected.append(
            HorizontalReceptor(
                horizontal_receptor_id=stable_identifier(
                    "hr",
                    [
                        pool.study_site_id,
                        str(int(candidate_global_indices[index])),
                        f"{x_m:.6f}",
                        f"{y_m:.6f}",
                    ],
                ),
                study_site_id=pool.study_site_id,
                x_m=float(x_m),
                y_m=float(y_m),
                lon=float(candidate_lon[index]),
                lat=float(candidate_lat[index]),
                source_face_local_index=int(candidate_local_indices[index]),
                source_face_global_index=int(candidate_global_indices[index]),
                anchor_snap_distance_m=snap_distance,
            )
        )
    return selected


def select_horizontal_receptors_from_coordinates(
    pool: HorizontalReceptorCandidatePool,
    *,
    coordinates_lonlat: Sequence[tuple[float, float]],
    coordinates_xy: Sequence[tuple[float, float]],
    tolerance_m: float = 1.0,
) -> list[HorizontalReceptor]:
    """依已核定的 WGS84 順序逐點映射 persistent-wet 原生 OCM face。

    ``coordinates_lonlat`` 是研究者已核定的資料交換座標，``coordinates_xy`` 是同一批
    座標依 flow-domain 的固定公尺制投影所得的位置；兩者只用來尋找 face，不會取代
    OCM 的垂向、NWW 或 arrival 支援檢查。``pool`` 已先完成 candidate polygon、邊界
    margin 與所有 arrival 的 persistent-wet 篩選，因此每個宣告點都必須在該 immutable
    pool 中找到唯一 source face。此函式保留宣告順序，拒絕重複 face、超過公尺制公差
    或不在候選池的座標；它不會退回 maximin、最近值或其他未核定位置。

    Args:
        pool: 同一份 OCM 原生 mesh 建立的 persistent-wet candidate pool。
        coordinates_lonlat: 五個固定水平受體的 WGS84 經度／緯度，順序不可改變。
        coordinates_xy: 上述座標在相同 flow-domain projection 下的公尺制位置。
        tolerance_m: 宣告座標與 source-face 中心允許的公尺制差異；必須為有限正數。

    Returns:
        依宣告順序排列的五個 ``HorizontalReceptor``，其 lon/lat 與 source face
        provenance 取自原生 mesh，而非由輸入座標臨時插值。

    Raises:
        ValueError: 固定點數量、投影座標、匹配公差或唯一 face 契約不成立時。
    """

    if not isinstance(tolerance_m, (int, float)) or isinstance(tolerance_m, bool):
        raise ValueError("固定水平受體 tolerance_m 必須是數值")
    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("固定水平受體 tolerance_m 必須是有限正數")
    if len(coordinates_lonlat) != len(coordinates_xy) or len(coordinates_lonlat) != 5:
        raise ValueError("固定水平受體必須恰有順序一致的五個 lon/lat 與公尺制座標")
    declared_lonlat = np.asarray(coordinates_lonlat, dtype=np.float64)
    declared_xy = np.asarray(coordinates_xy, dtype=np.float64)
    if declared_lonlat.shape != (5, 2) or declared_xy.shape != (5, 2):
        raise ValueError("固定水平受體座標必須是 (5, 2) 數值陣列")
    if not np.all(np.isfinite(declared_lonlat)) or not np.all(np.isfinite(declared_xy)):
        raise ValueError("固定水平受體座標不可含非有限值")
    if np.any(declared_lonlat[:, 0] < -180.0) or np.any(declared_lonlat[:, 0] > 180.0):
        raise ValueError("固定水平受體經度超出 WGS84 bounds")
    if np.any(declared_lonlat[:, 1] < -90.0) or np.any(declared_lonlat[:, 1] > 90.0):
        raise ValueError("固定水平受體緯度超出 WGS84 bounds")

    candidate_xy = np.asarray(pool.candidate_xy_m, dtype=np.float64)
    if candidate_xy.ndim != 2 or candidate_xy.shape[1] != 2:
        raise ValueError("candidate pool 公尺制座標形狀不符")
    selected: list[HorizontalReceptor] = []
    used_faces: set[int] = set()
    anchor = np.asarray(pool.anchor_xy, dtype=np.float64)
    for order, (_lonlat, target_xy) in enumerate(zip(declared_lonlat, declared_xy, strict=True)):
        distances = np.linalg.norm(candidate_xy - target_xy, axis=1)
        if distances.size == 0:
            raise ValueError(f"{pool.study_site_id} 固定受體候選池為空")
        candidate_position = int(np.argmin(distances))
        distance_m = float(distances[candidate_position])
        if distance_m > tolerance:
            raise ValueError(
                f"{pool.study_site_id} 固定受體第 {order + 1} 點無法映射到同一 OCM mesh face："
                f"distance={distance_m:.6f} m > tolerance={tolerance:.6f} m"
            )
        local_index = int(pool.candidate_face_local_indices[candidate_position])
        if local_index in used_faces:
            raise ValueError(
                f"{pool.study_site_id} 固定受體第 {order + 1} 點與既有點映射到重複 OCM face："
                f"{local_index}"
            )
        used_faces.add(local_index)
        x_m, y_m = candidate_xy[candidate_position]
        snap_distance = (
            float(np.linalg.norm(candidate_xy[candidate_position] - anchor))
            if order == 0
            else None
        )
        selected.append(
            HorizontalReceptor(
                horizontal_receptor_id=stable_identifier(
                    "hr",
                    [
                        pool.study_site_id,
                        str(int(pool.candidate_face_global_indices[candidate_position])),
                        f"{x_m:.6f}",
                        f"{y_m:.6f}",
                    ],
                ),
                study_site_id=pool.study_site_id,
                x_m=float(x_m),
                y_m=float(y_m),
                lon=float(pool.candidate_lon[candidate_position]),
                lat=float(pool.candidate_lat[candidate_position]),
                source_face_local_index=local_index,
                source_face_global_index=int(pool.candidate_face_global_indices[candidate_position]),
                anchor_snap_distance_m=snap_distance,
            )
        )
    return selected


def select_horizontal_receptors(
    *,
    study_site_id: str,
    mesh: NativeMesh,
    candidate_polygon_metric: Polygon,
    anchor_xy: tuple[float, float],
    wetdry_at_arrivals: np.ndarray,
    count: int = 5,
    boundary_margin_m: float = 0.0,
    wet_value: float = 0.0,
    maximum_anchor_snap_distance_m: float | None = None,
) -> list[HorizontalReceptor]:
    """相容既有 API：prepare 一次候選池後執行歷史 deterministic selector。

    wrapper 保持原參數與回傳型別，讓 legacy caller 不需知道 pool 內部拆分；候選 pool 的
    geometry、persistent-wet、anchor-first、tie-break、boundary margin 與 snap distance
    語意完全由既有兩階段 API 共用。current formal input derivation 不使用此入口，改由
    ``select_horizontal_receptors_random_from_pool`` 依版本化 seed 契約選樣。
    """

    pool = prepare_horizontal_receptor_candidates(
        study_site_id=study_site_id,
        mesh=mesh,
        candidate_polygon_metric=candidate_polygon_metric,
        anchor_xy=anchor_xy,
        wetdry_at_arrivals=wetdry_at_arrivals,
        boundary_margin_m=boundary_margin_m,
        wet_value=wet_value,
        maximum_anchor_snap_distance_m=maximum_anchor_snap_distance_m,
    )
    return select_horizontal_receptors_from_pool(pool, count=count)


def build_vertical_targets(
    *,
    surface_z_m: float,
    bed_z_m: float,
    valid_layer_z_m: np.ndarray,
) -> list[VerticalTarget]:
    """建立歷史 10/40/70% depth 與 near-bed 四個可包夾、互異垂向目標。

    前三個目標保持連續物理 z，不硬 snap 到 layer；``bracket_span_m`` 保存實際上下層
    間距。near-bed 使用最低兩個有效 full levels 的中點，代表最低 layer center。若水柱
    太淺或有效層不足以形成四個互異目標，水平位置應由上游淘汰重選。current formal
    不使用本 helper；正式垂向受體改由 ``sample_random_vertical_draws`` 建立。
    """

    layers = np.unique(np.asarray(valid_layer_z_m, dtype=np.float64))
    layers = np.sort(layers[np.isfinite(layers)])
    if not np.isfinite(surface_z_m) or not np.isfinite(bed_z_m) or surface_z_m <= bed_z_m:
        raise ValueError("surface/bed z 必須有限且 surface 高於 bed")
    layers = layers[(layers >= bed_z_m - 1e-9) & (layers <= surface_z_m + 1e-9)]
    if layers.size < 4:
        raise ValueError("有效 z layers 不足以建立四個垂向 receptor")

    def bracket_span(target: float) -> float:
        """回傳目標 z 最近上下有效層距；無雙側支撐時拒絕垂向外插。"""

        below = layers[layers <= target]
        above = layers[layers >= target]
        if below.size == 0 or above.size == 0:
            raise ValueError("垂向目標無上下包夾")
        return float(above.min() - below.max())

    depth = surface_z_m - bed_z_m
    specs = [
        ("upper_water_column", 0.10),
        ("mid_upper_water_column", 0.40),
        ("mid_lower_water_column", 0.70),
    ]
    targets = []
    for vertical_id, fraction in specs:
        target = surface_z_m - fraction * depth
        targets.append(
            VerticalTarget(
                vertical_id,
                fraction,
                float(target),
                float(target - bed_z_m),
                bracket_span(target),
            )
        )
    near_bed = float((layers[0] + layers[1]) * 0.5)
    targets.append(
        VerticalTarget(
            "near_bed",
            None,
            near_bed,
            near_bed - bed_z_m,
            float(layers[1] - layers[0]),
        )
    )
    if len({round(item.z_m_positive_up, 9) for item in targets}) != 4:
        raise ValueError("四個垂向目標不互異，需淘汰此水平候選")
    return targets
