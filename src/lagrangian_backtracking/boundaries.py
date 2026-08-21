"""判定局部分析區、共同流場範圍、海岸及垂向邊界的穿越事件。

水平事件以粒子在單一步內走過的線段與多邊形邊界的第一個交點計算，不能只看步末位置。
每個多邊形邊界還要明確標示可與外海交換水體的線段：穿越這些線段才是離開局部分析區
或流場範圍；其餘被海岸切出的邊界都視為撞岸。貢寮與龜山島各自離開局部分析區時只記錄
事件、不中止回溯；穿越另一研究區的局部範圍也不改變粒子的歸屬。只有離開共同 A 區流場
範圍才停止。B 至 D 區若局部範圍與流場範圍相同，只寫一筆離開流場事件，並註明它同時
也是第一次離開局部分析區。
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
    """執行單一研究區時需要的公尺制邊界資料。

    ``own_local_open_boundary`` 與 ``flow_open_boundary`` 分別是局部分析區及流場範圍
    的邊界中，允許水體進出的線段；交點不在這些線段上就代表海岸。值為 ``None`` 只供
    自動測試和舊版設定檔相容，意義是整個外框都可進出。正式分析必須提供兩組線段，否則
    可能把海岸誤判為潛在來源的入口。
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
        """拒絕不合理的容許誤差與非線狀的可進出邊界。

        容許誤差只用來吸收幾何運算的極小浮點誤差，不能用很大的緩衝距離掩蓋錯誤的岸線
        資料。若誤把面積多邊形當成可進出線段，面內任何點到它的距離都可能為零，因此在
        建立設定時立刻報錯，避免分析結果被悄悄污染。
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
    """判斷穿越點是否落在允許進出水體的邊界線段。

    回傳資料中的 ``boundary_s_m`` 會改成沿可進出線段量得的距離。如此一來，後續估計
    邊界來源密度時，海岸線不會占用一維座標。若沒有提供線段，才暫時使用整個多邊形外框
    的距離；這個相容模式只供測試或讀取舊資料，不能用於正式成果。
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
    """建立不可穿越的撞岸事件；海岸不列入外海入口的來源統計。"""

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
    """依穿越位置在本步所占比例，同步內插時刻與深度，避免欄位彼此錯位。"""

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
    """依粒子行進順序判定並套用水平邊界事件。

    單一步內可能先離開自己的局部分析區、再穿越另一研究區的局部範圍，最後離開共同
    流場範圍。本函式會保留真正停止前的所有事件，停止後的事件不再寫入。使用包含邊界
    的幾何判定，可避免極小浮點誤差把剛好在邊界上的步首位置誤認成範圍外。
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
