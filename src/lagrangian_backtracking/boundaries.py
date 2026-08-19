"""巢狀 local、foreign-local、共用 outer 與垂向障壁事件解析。

水平事件以步內線段與 polygon exterior 的第一交點計算，不以步末位置替代。每個
polygon boundary 必須再以顯式 open-water 線段區分開放邊界與海岸：穿越 open-water
才是 local/flow exit，穿越被海岸裁切出的其餘邊界則是 coast contact。貢寮／龜山島的
own-local exit 只記錄後繼續；foreign-local crossing 永不改變 site、scenario、seed 或
停止狀態；共同 A outer exit 才停止。B-D 的 local/flow 重合時只寫一筆 outer event，
並以 attribute 保存同時具有 local-first-exit 語意。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from .geometry import SegmentCrossing, first_polygon_crossing, polygon_crossings
from .models import BoundaryEvent, EventType, ParticleState, ParticleStatus, VelocitySample


@dataclass(frozen=True, slots=True)
class BoundaryGeometry:
    """單一 study site 執行時所需的固定公尺制邊界集合。

    ``own_local_open_boundary`` 與 ``flow_open_boundary`` 是 polygon exterior 中允許
    水體交換的 LineString/MultiLineString 子集合；交點不在子集合上即視為海岸。
    ``None`` 只保留給合成測試與舊 manifest 相容，代表整圈皆為開放邊界；正式發布
    的 geometry manifest 必須提供兩者，避免把海岸誤算成來源入口。
    """

    own_local_domain: Polygon
    flow_domain: Polygon
    foreign_local_domains: dict[str, Polygon]
    own_local_open_boundary: BaseGeometry | None = None
    flow_open_boundary: BaseGeometry | None = None
    local_equals_flow: bool = False
    own_local_boundary_segment_id: str = "own_local_open_boundary"
    flow_boundary_segment_id: str = "flow_domain_open_boundary"
    coastline_segment_id: str = "coastline"
    boundary_match_tolerance_m: float = 1.0e-6

    def __post_init__(self) -> None:
        """拒絕負 tolerance 與非線狀 open-boundary 幾何。

        tolerance 只吸收 GEOS overlay 的浮點殘差，不得以大型 buffer 取代正確的岸線
        拓撲。Polygon 若誤傳為 open boundary 會讓內部點距離為零，因此在此 fail-fast。
        """

        if self.boundary_match_tolerance_m < 0:
            raise ValueError("boundary_match_tolerance_m 不可為負")
        for name, value in (
            ("own_local_open_boundary", self.own_local_open_boundary),
            ("flow_open_boundary", self.flow_open_boundary),
        ):
            if value is not None and value.geom_type not in {"LineString", "MultiLineString"}:
                raise ValueError(f"{name} 必須是 LineString 或 MultiLineString")


def _open_boundary_crossing(
    crossing: SegmentCrossing,
    open_boundary: BaseGeometry | None,
    *,
    tolerance_m: float,
) -> SegmentCrossing | None:
    """判斷 polygon crossing 是否落在命名的 open-water 線段。

    回傳值的 ``boundary_s_m`` 改以 open-boundary 線段本身的弧長座標表示，使不同
    local/flow polygon 的海岸段不會佔據 KDE 的一維座標。無顯式線段時沿用 polygon
    exterior 弧長，此模式僅供合成測試及舊資料相容。
    """

    if open_boundary is None:
        return crossing
    point = Point(crossing.x_m, crossing.y_m)
    if point.distance(open_boundary) > tolerance_m:
        return None
    return replace(crossing, boundary_s_m=float(open_boundary.project(point)))


def _coast_event(
    previous: ParticleState,
    proposed: ParticleState,
    *,
    crossing: SegmentCrossing,
    geometry: BoundaryGeometry,
) -> BoundaryEvent:
    """建立不可穿越的海岸接觸事件；海岸不納入任何 open-boundary density。"""

    return _event_at_crossing(
        previous,
        proposed,
        crossing=replace(crossing, boundary_s_m=0.0),
        event_type=EventType.COAST_CONTACT,
        boundary_segment_id=geometry.coastline_segment_id,
        attributes={"baseline_policy": "stop_at_first_contact"},
    )


def _event_at_crossing(
    previous: ParticleState,
    proposed: ParticleState,
    *,
    crossing: SegmentCrossing,
    event_type: EventType,
    related_study_site_id: str | None = None,
    boundary_segment_id: str | None = None,
    attributes: dict[str, bool | float | int | str] | None = None,
) -> BoundaryEvent:
    """以同一 crossing fraction 內插 UTC 與 z，避免事件欄位彼此錯位。"""

    fraction = crossing.fraction
    time_ns = previous.time_utc_ns + int(round(fraction * (proposed.time_utc_ns - previous.time_utc_ns)))
    z_m = previous.z_m + fraction * (proposed.z_m - previous.z_m)
    return BoundaryEvent(
        particle_id=previous.particle_id,
        scenario_id=previous.scenario_id,
        member_id=previous.member_id,
        study_site_id=previous.study_site_id,
        analysis_region_id=previous.analysis_region_id,
        receptor_id=previous.receptor_id,
        event_type=event_type,
        time_utc_ns=time_ns,
        x_m=crossing.x_m,
        y_m=crossing.y_m,
        z_m=float(z_m),
        fraction=fraction,
        related_study_site_id=related_study_site_id,
        boundary_segment_id=boundary_segment_id,
        boundary_s_m=crossing.boundary_s_m,
        attributes=attributes or {},
    )


def resolve_horizontal_boundaries(
    previous: ParticleState, proposed: ParticleState, geometry: BoundaryGeometry
) -> tuple[ParticleState, list[BoundaryEvent]]:
    """依沿步進方向的 fraction 排序並套用水平事件。

    同一步可能先離開 own local、再穿過 foreign local、最後離開 flow domain；函式保留
    outer stop 以前的所有事件，outer 後事件不寫入。邊界上的 ``covers`` 語意可避免
    浮點小誤差把步首誤判為域外。
    """

    start_xy = (previous.x_m, previous.y_m)
    end_xy = (proposed.x_m, proposed.y_m)
    start_point = Point(*start_xy)
    end_point = Point(*end_xy)
    candidates: list[tuple[float, str, BoundaryEvent]] = []

    own_start = geometry.own_local_domain.covers(start_point)
    own_end = geometry.own_local_domain.covers(end_point)
    if own_start and not own_end and not previous.own_local_exit_recorded:
        crossing = first_polygon_crossing(start_xy, end_xy, geometry.own_local_domain)
        if crossing is not None:
            open_geometry = (
                geometry.flow_open_boundary
                if geometry.local_equals_flow
                else geometry.own_local_open_boundary
            )
            open_crossing = _open_boundary_crossing(
                crossing,
                open_geometry,
                tolerance_m=geometry.boundary_match_tolerance_m,
            )
            if open_crossing is None:
                event = _coast_event(previous, proposed, crossing=crossing, geometry=geometry)
                candidates.append((crossing.fraction, "coast_stop", event))
            elif geometry.local_equals_flow:
                event = _event_at_crossing(
                    previous,
                    proposed,
                    crossing=open_crossing,
                    event_type=EventType.FLOW_DOMAIN_OPEN_EXIT,
                    boundary_segment_id=geometry.flow_boundary_segment_id,
                    attributes={"also_local_domain_first_exit": True},
                )
                candidates.append((crossing.fraction, "flow_stop", event))
            else:
                event = _event_at_crossing(
                    previous,
                    proposed,
                    crossing=open_crossing,
                    event_type=EventType.LOCAL_DOMAIN_FIRST_EXIT,
                    boundary_segment_id=geometry.own_local_boundary_segment_id,
                )
                candidates.append((crossing.fraction, "own_local", event))

    for foreign_site_id, polygon in sorted(geometry.foreign_local_domains.items()):
        inside_start = polygon.covers(start_point)
        crossings = polygon_crossings(start_xy, end_xy, polygon)
        inside = inside_start
        for crossing in crossings:
            # foreign local 是非終止診斷；步末 fraction=1 已記錄抵達邊界時，下一步會再
            # 以 fraction=0 看見同一交點。跳過步首交點可避免 enter 後立刻產生假 exit。
            # own/flow terminal crossing 不採此規則，因從邊界向外的 fraction=0 是必要
            # 停止語意。正式 receptor 不允許初始化在 foreign boundary 上。
            if crossing.fraction <= 1.0e-12:
                continue
            event_type = (
                EventType.OTHER_SITE_LOCAL_DOMAIN_EXIT if inside else EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER
            )
            event = _event_at_crossing(
                previous,
                proposed,
                crossing=crossing,
                event_type=event_type,
                related_study_site_id=foreign_site_id,
                boundary_segment_id=f"{foreign_site_id}_local_boundary",
            )
            candidates.append((crossing.fraction, "foreign", event))
            inside = not inside

    flow_start = geometry.flow_domain.covers(start_point)
    flow_end = geometry.flow_domain.covers(end_point)
    if not geometry.local_equals_flow and flow_start and not flow_end:
        crossing = first_polygon_crossing(start_xy, end_xy, geometry.flow_domain)
        if crossing is not None:
            open_crossing = _open_boundary_crossing(
                crossing,
                geometry.flow_open_boundary,
                tolerance_m=geometry.boundary_match_tolerance_m,
            )
            if open_crossing is None:
                event = _coast_event(previous, proposed, crossing=crossing, geometry=geometry)
                candidates.append((crossing.fraction, "coast_stop", event))
            else:
                event = _event_at_crossing(
                    previous,
                    proposed,
                    crossing=open_crossing,
                    event_type=EventType.FLOW_DOMAIN_OPEN_EXIT,
                    boundary_segment_id=geometry.flow_boundary_segment_id,
                )
                candidates.append((crossing.fraction, "flow_stop", event))

    events: list[BoundaryEvent] = []
    state = proposed
    for _, kind, event in sorted(candidates, key=lambda item: (item[0], item[1])):
        events.append(event)
        if kind == "own_local":
            state = replace(state, own_local_exit_recorded=True)
        if kind in {"coast_stop", "flow_stop"}:
            elapsed_fraction = event.fraction
            status = ParticleStatus.COAST_CONTACT if kind == "coast_stop" else ParticleStatus.FLOW_DOMAIN_EXIT
            state = replace(
                state,
                x_m=event.x_m,
                y_m=event.y_m,
                z_m=event.z_m,
                time_utc_ns=event.time_utc_ns,
                age_seconds=previous.age_seconds
                + elapsed_fraction * (proposed.age_seconds - previous.age_seconds),
                status=status,
                own_local_exit_recorded=(
                    state.own_local_exit_recorded or (kind == "flow_stop" and geometry.local_equals_flow)
                ),
            )
            break
    return state, events


def resolve_vertical_boundaries(
    previous: ParticleState,
    proposed: ParticleState,
    *,
    reference_sample: VelocitySample,
    behavior_class: str,
) -> tuple[ParticleState, list[BoundaryEvent]]:
    """套用海面／海床 first contact、反射或適用範圍退出。

    reference surface/bed 取自步首有效 forcing sample；adaptive dt 會限制垂向位移，正式
    驗收另以 dt 減半確認此線性化。rising 到海面停止；sinking 到海床沉積停止；其他
    類別鏡射並記 contact。若單步同時跨兩側，取沿線最早 crossing。
    """

    if not reference_sample.valid:
        raise ValueError("垂向邊界需要有效 reference forcing sample")
    dz = proposed.z_m - previous.z_m
    if abs(dz) <= np.finfo(np.float64).eps:
        return proposed, []
    candidates: list[tuple[float, str, float]] = []
    if proposed.z_m > reference_sample.eta_m and previous.z_m <= reference_sample.eta_m:
        fraction = (reference_sample.eta_m - previous.z_m) / dz
        candidates.append((float(fraction), "surface", reference_sample.eta_m))
    if proposed.z_m < reference_sample.bed_z_m and previous.z_m >= reference_sample.bed_z_m:
        fraction = (reference_sample.bed_z_m - previous.z_m) / dz
        candidates.append((float(fraction), "bed", reference_sample.bed_z_m))
    if not candidates:
        return proposed, []
    fraction, boundary_kind, boundary_z = min(candidates, key=lambda item: item[0])
    time_ns = previous.time_utc_ns + int(round(fraction * (proposed.time_utc_ns - previous.time_utc_ns)))
    x_m = previous.x_m + fraction * (proposed.x_m - previous.x_m)
    y_m = previous.y_m + fraction * (proposed.y_m - previous.y_m)
    terminal = (boundary_kind == "surface" and behavior_class == "rising") or (
        boundary_kind == "bed" and behavior_class in {"sinking", "near_bed"}
    )
    event_type = (
        EventType.SURFACE_REGIME_EXIT
        if boundary_kind == "surface" and terminal
        else EventType.DEPOSITED
        if boundary_kind == "bed" and terminal
        else EventType.SURFACE_CONTACT
        if boundary_kind == "surface"
        else EventType.BED_CONTACT
    )
    event = BoundaryEvent(
        particle_id=previous.particle_id,
        scenario_id=previous.scenario_id,
        member_id=previous.member_id,
        study_site_id=previous.study_site_id,
        analysis_region_id=previous.analysis_region_id,
        receptor_id=previous.receptor_id,
        event_type=event_type,
        time_utc_ns=time_ns,
        x_m=float(x_m),
        y_m=float(y_m),
        z_m=boundary_z,
        fraction=fraction,
        attributes={"behavior_class": behavior_class, "boundary_kind": boundary_kind},
    )
    if terminal:
        status = (
            ParticleStatus.SURFACE_REGIME_EXIT if boundary_kind == "surface" else ParticleStatus.DEPOSITED
        )
        stopped = replace(
            proposed,
            x_m=float(x_m),
            y_m=float(y_m),
            z_m=boundary_z,
            time_utc_ns=time_ns,
            age_seconds=previous.age_seconds + fraction * (proposed.age_seconds - previous.age_seconds),
            status=status,
        )
        return stopped, [event]
    reflected_z = 2.0 * boundary_z - proposed.z_m
    return replace(proposed, z_m=float(reflected_z)), [event]
