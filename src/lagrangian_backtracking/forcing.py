"""讀取海流、波浪、表面漂移與浮沉資料，並在指定時空位置提供速度。

海流資料使用海洋模式（OCM）的原始三維網格及時間軸；波浪資料使用 NWW3 分析資料。
每個讀取器一次只以唯讀方式開啟一個月份，避免把完整兩年資料載入記憶體。跨月時由
``MonthlyCombinedForcing`` 依世界協調時間（UTC）選取正確月份，絕不借用最近月份資料。
海流依序在節點垂向、三角形平面與時間上內插；移動海面造成固定 target z 高於某一
端點最高 ``zcor`` 時，該端點可明確使用最高有效層的表層控制體值，但最後仍以 query-time
``eta``／海床做幾何 gate。若三個支撐節點任一處無法取得雙側支撐或表層控制體必要值、
網格面乾涸或時間間隔過大，會回傳明確的品質檢查旗標。波浪只有在周圍四個格點及前後
兩個時刻的資料都有效時，才做空間與時間內插。
Smagorinsky reference 則只使用 OCM native current，在公尺制 triangle shape function
上建立先套 floor/cap 的 P1 nodal Kh 與 grad(K)；它不讀取 Stokes、settling 或空變 Kz，
也尚未代表正式 runtime baseline。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .accelerated import interpolate_ocm_support_numba
from .diffusion import (
    DiffusionCoefficients,
    DiffusionSample,
    SmagorinskySettings,
    smagorinsky_horizontal_diffusivity,
)
from .geometry import DomainProjection
from .mesh import MeshLocation, NativeMesh
from .models import (
    SURFACE_BOUNDARY_TOLERANCE_M,
    VERTICAL_BOUNDARY_TOLERANCE_M,
    SampleQC,
    VelocityComponents,
    VelocitySample,
    clamp_query_z_to_surface,
)
from .stokes import finite_depth_stokes


def _vertical_support_indices(
    physical_z: np.ndarray,
    usable: np.ndarray,
    target_z_m: float,
) -> tuple[int, int] | None:
    """依 OCM 單柱建立垂向支援層索引，包含保守的移動海面表層控制體政策。

    ``physical_z`` 是單一時間、單一 node 的 OCM ``zcor``，單位為 m 且海面向上為正；
    ``usable`` 表示該 layer 所需物理量是否全部有限。一般水柱仍要求 target z 有
    有效的下側與上側 layer，並以兩層做線性內插。唯一例外是 target z 高於該端點最高
    有限 ``zcor``（含海面 ``SURFACE_BOUNDARY_TOLERANCE_M`` 的數值容差）：此時只能使用
    最高 ``zcor`` layer 本身，且該 layer 的必要物理量缺值就直接失敗，不能退回更深層
    或把它描述為任意最近值外插。這是 OCM 最上層控制體的端點支援，不是海面邊界判定；
    query-time 的 ``eta``／海床範圍仍由外層最後檢查。底部不採對稱 hold，故海床下方
    或沒有下側支撐時維持 fail closed。

    回傳值是 ``(lower_index, upper_index)``；兩索引相同表示表層控制體 hold，呼叫端
    應將垂向跨度記為零。``None`` 代表 z、``zcor`` 或必要物理量不足以形成合法支援。
    """

    if (
        physical_z.ndim != 1
        or usable.shape != physical_z.shape
        or not np.isfinite(target_z_m)
    ):
        return None
    finite_z = np.isfinite(physical_z)
    finite_indices = np.flatnonzero(finite_z)
    if finite_indices.size == 0:
        return None
    top_index = int(finite_indices[np.argmax(physical_z[finite_indices])])
    top_z = float(physical_z[top_index])
    if target_z_m >= top_z - SURFACE_BOUNDARY_TOLERANCE_M:
        if not bool(usable[top_index]):
            return None
        return top_index, top_index
    below = np.flatnonzero(usable & (physical_z <= target_z_m))
    above = np.flatnonzero(usable & (physical_z >= target_z_m))
    if below.size == 0 or above.size == 0:
        return None
    lower = int(below[np.argmax(physical_z[below])])
    upper = int(above[np.argmin(physical_z[above])])
    return lower, upper


def _query_z_within_geometric_bounds(
    z_m: float, geometric_bounds: tuple[float, float] | None
) -> bool:
    """判定 query-time z 是否仍在有限海床／海面水柱內。

    ``geometric_bounds`` 已由同一 before/after eta、固定 native mesh 水深與 barycentric
    權重建立；此 helper 只允許公尺制數值誤差內的邊界點。它不負責修正 z，也不會讓
    endpoint surface hold 取代 query-time 海面；非有限幾何一律回傳 False。
    """

    if geometric_bounds is None or not np.isfinite(z_m):
        return False
    eta_m, bed_z_m = geometric_bounds
    if (
        not np.isfinite(eta_m)
        or not np.isfinite(bed_z_m)
        or bed_z_m > eta_m + VERTICAL_BOUNDARY_TOLERANCE_M
    ):
        return False
    return (
        z_m >= bed_z_m - VERTICAL_BOUNDARY_TOLERANCE_M
        and clamp_query_z_to_surface(z_m, eta_m) is not None
    )


def _query_z_for_vertical_support(
    z_m: float, geometric_bounds: tuple[float, float] | None
) -> float:
    """為 OCM 垂向支援提供已套用海面容許帶的 query z。

    ``geometric_bounds`` 是 query-time 的 ``(eta_m, bed_z_m)``；若海面幾何可用，
    只把容許帶內的微小上越夾回 ``eta_m``，讓 NumPy 與 Numba 看到完全相同的 target。
    超過容許帶時保留原始 z，讓後續的幾何 gate 回傳 ``VERTICAL_UNSUPPORTED``，而不把
    真實海面上越誤變成表層速度。幾何缺值時也保留原始 z，因為缺值不能取得邊界特權。
    """

    if geometric_bounds is None:
        return z_m
    clamped = clamp_query_z_to_surface(z_m, geometric_bounds[0])
    return z_m if clamped is None else clamped


def _time_bracket(
    times_ns: np.ndarray, target_ns: int, *, maximum_gap_ns: int
) -> tuple[int, int, float, SampleQC]:
    """找出目標時刻前後的資料索引與內插比例，並分開標記各種時間問題。"""

    if times_ns.ndim != 1 or times_ns.size < 2:
        return 0, 0, 0.0, SampleQC.OUTSIDE_TIME_RANGE
    position = int(np.searchsorted(times_ns, target_ns, side="left"))
    if position < times_ns.size and int(times_ns[position]) == target_ns:
        return position, position, 0.0, SampleQC.OK
    if position == 0 or position >= times_ns.size:
        return 0, 0, 0.0, SampleQC.OUTSIDE_TIME_RANGE
    before = position - 1
    after = position
    span = int(times_ns[after]) - int(times_ns[before])
    if span <= 0:
        return before, after, 0.0, SampleQC.NUMERICAL_FAILURE
    if span > maximum_gap_ns:
        return before, after, 0.0, SampleQC.TIME_GAP
    alpha = (target_ns - int(times_ns[before])) / span
    return before, after, float(alpha), SampleQC.OK


def _interpolate_optional_float(first: float, second: float, alpha: float) -> float:
    """在失敗診斷需要時內插兩端有限幾何值，否則保留 ``NaN`` 缺值語意。"""

    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(first + alpha * (second - first))


@dataclass(frozen=True, slots=True)
class WaveSample:
    """一筆 NWW3 波浪摘要資料；保留波向圓向量供跨月原語意內插。

    ``significant_wave_height_m``、``peak_frequency_hz`` 與
    ``peak_direction_raw_deg`` 是四角空間內插後的波浪摘要。NWW3 的方向不是普通
    線性角度；``direction_x`` 與 ``direction_y`` 保存空間內插階段實際累積的
    ``sin(direction)``／``cos(direction)`` 加權值，讓跨月份時能先對原始波浪欄位與
    圓向量做時間內插，再轉回角度。這兩個欄位對既有 caller 保持 optional；舊的
    synthetic callback 若只建立五欄 ``WaveSample``，仍維持原本資料格式，但跨月
    合成會拒絕缺少圓向量的波浪端點，避免用已轉回的角度猜回向量權重。
    """

    significant_wave_height_m: float
    peak_frequency_hz: float
    peak_direction_raw_deg: float
    qc_flags: int
    qc: SampleQC = SampleQC.OK
    direction_x: float | None = None
    direction_y: float | None = None

    @property
    def valid(self) -> bool:
        """只有資料遮罩與物理數值都通過檢查時，才能計算波浪造成的表面漂移。"""

        return self.qc == SampleQC.OK


@dataclass(frozen=True, slots=True)
class _SmagorinskyTimeSlice:
    """一個 OCM time slice 的 P1 nodal Kh 與 gradient 中間結果。

    ``nodal_kh_m2ps`` 只保存目前 particle triangle 三個節點的連續 nodal Kh；其值已
    由所有 valid incident triangle 的面積加權 candidate 組成，並非直接保存某一面
    的 piecewise-constant Kh。其餘欄位用於時間內插與 JSON-safe diagnostics，最後才由
    ``sample_smagorinsky_diffusion`` 組成公開的 ``DiffusionSample``。
    """

    nodal_kh_m2ps: tuple[float, float, float]
    particle_kh_m2ps: float
    gradient_x_mps: float
    gradient_y_mps: float
    raw_current_triangle_kh_m2ps: float
    floor_hit: bool
    cap_hit: bool
    valid_incident_triangle_count: int
    excluded_incident_triangle_count: int


def _smagorinsky_method_name() -> str:
    """回傳固定 method token，讓 diagnostics 可在 JSON 與報告中辨識演算法版本。"""

    return "smagorinsky_native_mesh_p1_nodal"


def _invalid_smagorinsky_sample(
    settings: SmagorinskySettings,
    qc: SampleQC,
    *,
    triangle_id: int | None,
    valid_incident_triangle_count: int = 0,
    excluded_incident_triangle_count: int = 0,
) -> DiffusionSample:
    """建立有明確非零 QC 的 Smagorinsky 失敗樣本，不以零 K 偽裝有效結果。

    當位置、時間、乾面或固定 z 垂向支撐失效時，水平 K 與梯度沒有科學值；此 helper
    仍保存方法、設定、triangle ID 與支撐計數，讓上層可稽核失敗來源。``None`` 是
    JSON-safe 缺值，與有效的數值零明確區分；Kz 只保留設定值作為型別欄位，不會因
    ``qc`` 非零而進入粒子積分。
    """

    return DiffusionSample(
        coefficients=DiffusionCoefficients(0.0, 0.0, settings.constant_kz_m2ps),
        diffusivity_divergence_mps=(0.0, 0.0, 0.0),
        qc=qc,
        diagnostics={
            "method": _smagorinsky_method_name(),
            "coefficient_cs": settings.coefficient_cs,
            "kh_m2ps": None,
            "d_kh_dx_mps": None,
            "d_kh_dy_mps": None,
            "raw_current_triangle_kh_m2ps": None,
            "floor_hit": False,
            "cap_hit": False,
            "valid_incident_triangle_count": int(valid_incident_triangle_count),
            "excluded_incident_triangle_count": int(excluded_incident_triangle_count),
            "triangle_id": triangle_id,
        },
    )


class OCMNativeMonth:
    """讀取一個月份 OCM 原始網格的速度與擴散資料。

    資料包含時間、水平位置與垂向深度三個方向；網格拓撲沿用前處理結果，不自行重建。
    ``sample_smagorinsky_diffusion`` 是同一月份物理資料上的 NumPy reference 方法，
    只用 native current 算空間 Kh；一般 ``sample`` 的 OCM vertical velocity 與
    diffusivity 取樣契約不會被它改寫。
    """

    def __init__(
        self,
        *,
        month_id: str,
        mesh: NativeMesh,
        time_utc_ns: np.ndarray,
        hvel: np.ndarray,
        vertical_velocity: np.ndarray,
        zcor: np.ndarray,
        elev: np.ndarray,
        wetdry_elem: np.ndarray,
        diffusivity: np.ndarray,
        maximum_time_gap_seconds: float = 5_400.0,
        wet_value: float = 0.0,
        use_numba_kernel: bool = False,
    ) -> None:
        """保留唯讀的大型陣列並檢查各欄位的時間、節點與深度維度一致。"""

        self.month_id = month_id
        self.mesh = mesh
        self.time_utc_ns = np.asarray(time_utc_ns)
        self.hvel = hvel
        self.vertical_velocity = vertical_velocity
        self.zcor = zcor
        self.elev = elev
        self.wetdry_elem = wetdry_elem
        self.diffusivity = diffusivity
        self.maximum_time_gap_ns = int(round(maximum_time_gap_seconds * 1_000_000_000))
        self.wet_value = float(wet_value)
        self.use_numba_kernel = bool(use_numba_kernel)
        expected_tn = (self.time_utc_ns.size, self.mesh.node_xy.shape[0])
        if self.hvel.shape[:2] != expected_tn or self.hvel.ndim != 4 or self.hvel.shape[-1] < 2:
            raise ValueError("hvel shape 必須是 (time,node,layer,component>=2)")
        expected_tnl = self.hvel.shape[:3]
        if (
            self.vertical_velocity.shape != expected_tnl
            or self.zcor.shape != expected_tnl
            or self.diffusivity.shape != expected_tnl
        ):
            raise ValueError("vertical_velocity/zcor/diffusivity 必須與 hvel time/node/layer 對齊")
        if self.elev.shape != expected_tn or self.wetdry_elem.shape != (
            self.time_utc_ns.size,
            self.mesh.source_face_global_index.size,
        ):
            raise ValueError("elev 或 wetdry_elem shape 與 mesh/time 不符")
        if self.time_utc_ns.dtype != np.int64 or np.any(np.diff(self.time_utc_ns) <= 0):
            raise ValueError("OCM time_utc_ns 必須是嚴格遞增 int64")

    @classmethod
    def from_directory(cls, month_dir: str | Path, *, mesh: NativeMesh) -> OCMNativeMonth:
        """從月份資料夾以唯讀方式開啟陣列，不允許載入任意序列化物件。"""

        root = Path(month_dir)

        def load(name: str) -> np.ndarray:
            """唯讀開啟一個月份陣列，避免複製龐大的時間、節點與深度資料。"""

            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"缺少 OCM month array：{path}")
            return np.load(path, mmap_mode="r", allow_pickle=False)

        return cls(
            month_id=root.name,
            mesh=mesh,
            time_utc_ns=load("time_utc_ns.npy"),
            hvel=load("hvel.npy"),
            vertical_velocity=load("vertical_velocity.npy"),
            zcor=load("zcor.npy"),
            elev=load("elev.npy"),
            wetdry_elem=load("wetdry_elem.npy"),
            diffusivity=load("diffusivity.npy"),
        )

    def _vertical_node_sample(
        self, *, time_index: int, node_index: int, z_m: float
    ) -> tuple[np.ndarray, float] | None:
        """在單一節點的水柱中，以指定深度取樣速度與垂向擴散。

        回傳的四個值依序是東向、北向、垂向速度與垂向擴散係數，另附兩層間的深度差。
        不假設資料層號已由淺到深排序；一般位置任一必要數值缺漏時該層不可用，必須由
        上下兩個有效 layer 夾住後線性內插。移動海面使固定 target z 在某一時間端點
        高於最高有限 ``zcor`` 時，則使用該端點最高 layer 的值並回傳零垂向跨度，這是
        OCM 最上層控制體的 surface hold，不是任意最近值外插，也不代表粒子已在海面上。
        真正的 query-time ``eta``／海床 gate 在 ``sample`` 完成 before/after 時間內插後
        執行；底層沒有對稱 hold，缺少最高層必要物理量仍回傳 ``None``。
        """

        physical_z = np.asarray(self.zcor[time_index, node_index], dtype=np.float64)
        values = np.column_stack(
            (
                np.asarray(self.hvel[time_index, node_index, :, 0], dtype=np.float64),
                np.asarray(self.hvel[time_index, node_index, :, 1], dtype=np.float64),
                np.asarray(self.vertical_velocity[time_index, node_index], dtype=np.float64),
                np.asarray(self.diffusivity[time_index, node_index], dtype=np.float64),
            )
        )
        usable = np.isfinite(physical_z) & np.all(np.isfinite(values), axis=1)
        support = _vertical_support_indices(physical_z, usable, z_m)
        if support is None:
            return None
        lower, upper = support
        span = float(physical_z[upper] - physical_z[lower])
        if abs(span) <= np.finfo(np.float64).eps * 16.0:
            return values[lower], 0.0
        alpha = (z_m - float(physical_z[lower])) / span
        return values[lower] + alpha * (values[upper] - values[lower]), span

    def _vertical_horizontal_velocity_node_sample(
        self, *, time_index: int, node_index: int, z_m: float
    ) -> tuple[np.ndarray, float] | None:
        """只以 OCM 水平 current 在固定 z 取樣，供 Smagorinsky 梯度使用。

        這裡刻意不讀 ``vertical_velocity`` 或 OCM ``diffusivity``：Slice 2B1 的 Kh
        定義只使用 native current，Kz 則由 ``SmagorinskySettings.constant_kz_m2ps``
        提供。垂向支援索引沿用一般速度取樣的上下夾層與 surface hold 政策：端點固定
        z 高於最高有限 ``zcor`` 時只能使用該端點最高 layer 的有限 ``hvel``，不做底層
        對稱 hold 或任意最近值外插。這個 helper 本身不判斷 query-time 海面，該 gate
        由 ``sample_smagorinsky_diffusion`` 在 before/after 幾何內插後統一執行。回傳的
        兩個水平分量單位為 m/s，第二項是實際夾層深度差（m），只供資料品質與垂向支援
        診斷。
        """

        if not np.isfinite(z_m):
            return None
        physical_z = np.asarray(self.zcor[time_index, node_index], dtype=np.float64)
        horizontal_velocity = np.asarray(
            self.hvel[time_index, node_index, :, :2], dtype=np.float64
        )
        usable = np.isfinite(physical_z) & np.all(np.isfinite(horizontal_velocity), axis=1)
        support = _vertical_support_indices(physical_z, usable, z_m)
        if support is None:
            return None
        lower, upper = support
        span = float(physical_z[upper] - physical_z[lower])
        if abs(span) <= np.finfo(np.float64).eps * 16.0:
            sampled = horizontal_velocity[lower]
        else:
            alpha = (z_m - float(physical_z[lower])) / span
            sampled = horizontal_velocity[lower] + alpha * (
                horizontal_velocity[upper] - horizontal_velocity[lower]
            )
        if not np.all(np.isfinite(sampled)):
            return None
        return np.asarray(sampled, dtype=np.float64), span

    def _smagorinsky_time_slice(
        self,
        location: MeshLocation,
        *,
        time_index: int,
        z_m: float,
        settings: SmagorinskySettings,
    ) -> tuple[_SmagorinskyTimeSlice | None, SampleQC, int, int]:
        """計算一個時間片的 incident-support area weighting 與 P1 Kh gradient。

        先以目前 triangle 三個節點的 adjacency 建立 deterministic 支撐集合，再在同一
        ``time_index`` 內用 ``node_cache`` 保存每個 unique node 的固定 z 水平 current。
        每個 incident triangle 的速度梯度由 native mesh 的 shape functions 計算；該面
        的 raw Kh 先套 floor/cap，才以 triangle area 做 nodal Kh 加權。乾面、固定 z
        無法夾層的 incident triangle 只增加 excluded count；但目前 particle triangle
        無 valid candidate，或目前任一節點沒有 valid support，便回傳相應 QC，禁止零值
        或最近值補洞。
        """

        triangle_id = location.triangle_id
        incident_triangle_ids = sorted(
            {
                incident_id
                for node in location.node_indices
                for incident_id in self.mesh.incident_triangles(node)
            }
        )
        node_cache: dict[int, tuple[np.ndarray, float] | None] = {}
        bounded_by_triangle: dict[int, float] = {}
        raw_by_triangle: dict[int, float] = {}
        floor_hit_by_triangle: dict[int, bool] = {}
        cap_hit_by_triangle: dict[int, bool] = {}
        excluded_count = 0

        for incident_id in incident_triangle_ids:
            face = int(self.mesh.triangle_face_local[incident_id])
            wetdry = float(self.wetdry_elem[time_index, face])
            if not np.isfinite(wetdry) or not np.isclose(wetdry, self.wet_value, atol=0.1):
                excluded_count += 1
                continue
            node_values: list[np.ndarray] = []
            unsupported = False
            for node_value in self.mesh.triangle_nodes[incident_id]:
                node = int(node_value)
                if node not in node_cache:
                    node_cache[node] = self._vertical_horizontal_velocity_node_sample(
                        time_index=time_index,
                        node_index=node,
                        z_m=z_m,
                    )
                sampled = node_cache[node]
                if sampled is None:
                    unsupported = True
                    break
                node_values.append(sampled[0])
            if unsupported:
                excluded_count += 1
                continue
            horizontal_values = np.asarray(node_values, dtype=np.float64)
            try:
                du_dx, du_dy = self.mesh.triangle_linear_gradient(
                    incident_id, horizontal_values[:, 0]
                )
                dv_dx, dv_dy = self.mesh.triangle_linear_gradient(
                    incident_id, horizontal_values[:, 1]
                )
                raw_kh, _, _ = smagorinsky_horizontal_diffusivity(
                    du_dx_per_s=du_dx,
                    du_dy_per_s=du_dy,
                    dv_dx_per_s=dv_dx,
                    dv_dy_per_s=dv_dy,
                    triangle_area_m2=float(self.mesh.triangle_area_m2[incident_id]),
                    coefficient_cs=settings.coefficient_cs,
                )
                bounded_kh, floor_hit, cap_hit = smagorinsky_horizontal_diffusivity(
                    du_dx_per_s=du_dx,
                    du_dy_per_s=du_dy,
                    dv_dx_per_s=dv_dx,
                    dv_dy_per_s=dv_dy,
                    triangle_area_m2=float(self.mesh.triangle_area_m2[incident_id]),
                    coefficient_cs=settings.coefficient_cs,
                    floor_m2ps=settings.floor_m2ps,
                    cap_m2ps=settings.cap_m2ps,
                )
            except (TypeError, ValueError, FloatingPointError) as error:
                # 靜態 mesh 與有限 node current 已通過前置閘門；若公式仍不能產生有限
                # Kh，這是數值失敗而非可安全排除的乾面，必須 fail closed。
                del error
                return (
                    None,
                    SampleQC.NUMERICAL_FAILURE,
                    len(bounded_by_triangle),
                    len(incident_triangle_ids) - len(bounded_by_triangle),
                )
            bounded_by_triangle[incident_id] = float(bounded_kh)
            raw_by_triangle[incident_id] = float(raw_kh)
            floor_hit_by_triangle[incident_id] = bool(floor_hit)
            cap_hit_by_triangle[incident_id] = bool(cap_hit)

        valid_count = len(bounded_by_triangle)
        if triangle_id not in bounded_by_triangle:
            current_face = int(self.mesh.triangle_face_local[triangle_id])
            current_wetdry = float(self.wetdry_elem[time_index, current_face])
            if not np.isfinite(current_wetdry) or not np.isclose(
                current_wetdry, self.wet_value, atol=0.1
            ):
                qc = SampleQC.DRY_FACE
            else:
                qc = SampleQC.VERTICAL_UNSUPPORTED
            return None, qc, valid_count, len(incident_triangle_ids) - valid_count

        nodal_kh: list[float] = []
        for node_value in location.node_indices:
            node = int(node_value)
            support_ids = [
                incident_id
                for incident_id in self.mesh.incident_triangles(node)
                if incident_id in bounded_by_triangle
            ]
            if not support_ids:
                return (
                    None,
                    SampleQC.VERTICAL_UNSUPPORTED,
                    valid_count,
                    len(incident_triangle_ids) - valid_count,
                )
            areas = np.asarray(
                [self.mesh.triangle_area_m2[incident_id] for incident_id in support_ids],
                dtype=np.float64,
            )
            area_total = float(np.sum(areas))
            if not np.all(np.isfinite(areas)) or area_total <= 0:
                return (
                    None,
                    SampleQC.NUMERICAL_FAILURE,
                    valid_count,
                    len(incident_triangle_ids) - valid_count,
                )
            weighted = float(
                np.dot(
                    areas,
                    np.asarray([bounded_by_triangle[incident_id] for incident_id in support_ids]),
                )
                / area_total
            )
            if not np.isfinite(weighted) or weighted < 0:
                return (
                    None,
                    SampleQC.NUMERICAL_FAILURE,
                    valid_count,
                    len(incident_triangle_ids) - valid_count,
                )
            nodal_kh.append(weighted)

        nodal_values = (nodal_kh[0], nodal_kh[1], nodal_kh[2])
        weights = np.asarray(location.barycentric_weights, dtype=np.float64)
        if weights.shape != (3,) or not np.all(np.isfinite(weights)):
            return (
                None,
                SampleQC.NUMERICAL_FAILURE,
                valid_count,
                len(incident_triangle_ids) - valid_count,
            )
        particle_kh = float(weights @ np.asarray(nodal_values, dtype=np.float64))
        try:
            gradient_x, gradient_y = self.mesh.triangle_linear_gradient(triangle_id, nodal_values)
        except (TypeError, ValueError, IndexError):
            return (
                None,
                SampleQC.NUMERICAL_FAILURE,
                valid_count,
                len(incident_triangle_ids) - valid_count,
            )
        if not np.isfinite(particle_kh) or particle_kh < 0:
            return (
                None,
                SampleQC.NUMERICAL_FAILURE,
                valid_count,
                len(incident_triangle_ids) - valid_count,
            )
        return (
            _SmagorinskyTimeSlice(
                nodal_kh_m2ps=nodal_values,
                particle_kh_m2ps=particle_kh,
                gradient_x_mps=float(gradient_x),
                gradient_y_mps=float(gradient_y),
                raw_current_triangle_kh_m2ps=float(raw_by_triangle[triangle_id]),
                floor_hit=floor_hit_by_triangle[triangle_id],
                cap_hit=cap_hit_by_triangle[triangle_id],
                valid_incident_triangle_count=valid_count,
                excluded_incident_triangle_count=len(incident_triangle_ids) - valid_count,
            ),
            SampleQC.OK,
            valid_count,
            len(incident_triangle_ids) - valid_count,
        )

    def sample_smagorinsky_diffusion(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        settings: SmagorinskySettings,
        *,
        triangle_hint: int | None = None,
    ) -> DiffusionSample:
        """以 OCM native current 取樣 P1 nodal Smagorinsky 空間變擴散。

        取樣流程固定為：先以既有 ``mesh.locate`` 取得 particle triangle，再用既有
        ``_time_bracket`` 找 UTC 前後時間片；每個時間片先在三個節點固定 ``z_m`` 做
        水平 current 垂向內插，接著對每個 incident triangle 以 native 公尺座標的線性
        shape function 計算速度梯度與 equation 10 的 triangle Kh。每個 candidate 先套
        ``settings.floor_m2ps``／``cap_m2ps``，再按 triangle area 對目前三個節點加權，
        形成連續 P1 nodal Kh；最後以 particle triangle 的 barycentric weights 內插
        particle Kh，並由同三個 nodal 值計算公尺制 ``(dKh/dx,dKh/dy)``。

        before／after 時間片各自完成上述非線性 Kh 計算後，再對 particle Kh、梯度與
        raw current-triangle Kh 線性內插；Kx=Ky 為內插 Kh，Kz 固定取設定值，垂向梯度
        暫設為零。每個時間片內 unique node 的固定 z 取樣由 helper cache，避免同一
        node column 因多個 incident triangle 重複讀取。乾面或 z unsupported 的非目前
        incident triangle 只排除並記數；目前 triangle 在任一必要時間片失效時回傳非零
        QC，不做最近值、跨 gap 或域外外插。``use_numba_kernel`` 不參與此方法，因為
        Slice 2B1 的 Smagorinsky reference algorithm 必須在 NumPy 與 Numba velocity
        switch 下得到相同數值；這不代表已具備正式 runtime 或通過 well-mixed/PDE 驗證。
        """

        if not isinstance(settings, SmagorinskySettings):
            raise TypeError("settings 必須是 SmagorinskySettings")
        settings.validate()
        location = self.mesh.locate(x_m, y_m, triangle_hint=triangle_hint)
        if location is None:
            return _invalid_smagorinsky_sample(
                settings,
                SampleQC.OUTSIDE_HORIZONTAL_DOMAIN,
                triangle_id=None,
            )
        before, after, alpha, time_qc = _time_bracket(
            self.time_utc_ns,
            time_utc_ns,
            maximum_gap_ns=self.maximum_time_gap_ns,
        )
        if time_qc != SampleQC.OK:
            return _invalid_smagorinsky_sample(
                settings,
                time_qc,
                triangle_id=location.triangle_id,
            )
        # Smagorinsky 共用 endpoint surface hold 時，不能只因兩端都有水平 current
        # 就放寬粒子垂向範圍。先以同一 query-time eta／native bed 建立幾何 gate，
        # 讓 z>eta、z<bed、eta 缺值與不相容水柱在進入 incident-triangle 計算前
        # 便明確失敗；這與一般 OCM velocity sample 的最後 gate 完全一致。
        geometric_bounds = self._geometric_bounds_at_time(
            location,
            before=before,
            after=after,
            alpha=alpha,
        )
        if not _query_z_within_geometric_bounds(z_m, geometric_bounds):
            return _invalid_smagorinsky_sample(
                settings,
                SampleQC.VERTICAL_UNSUPPORTED,
                triangle_id=location.triangle_id,
            )
        support_z_m = _query_z_for_vertical_support(z_m, geometric_bounds)

        first, first_qc, first_valid_count, first_excluded_count = self._smagorinsky_time_slice(
            location,
            time_index=before,
            z_m=support_z_m,
            settings=settings,
        )
        if first is None or first_qc != SampleQC.OK:
            return _invalid_smagorinsky_sample(
                settings,
                first_qc,
                triangle_id=location.triangle_id,
                valid_incident_triangle_count=first_valid_count,
                excluded_incident_triangle_count=first_excluded_count,
            )
        if after == before:
            second = first
            second_qc = SampleQC.OK
            second_valid_count = first_valid_count
            second_excluded_count = first_excluded_count
        else:
            second, second_qc, second_valid_count, second_excluded_count = (
                self._smagorinsky_time_slice(
                    location,
                    time_index=after,
                    z_m=support_z_m,
                    settings=settings,
                )
            )
        if second is None or second_qc != SampleQC.OK:
            return _invalid_smagorinsky_sample(
                settings,
                first_qc | second_qc,
                triangle_id=location.triangle_id,
                valid_incident_triangle_count=second_valid_count,
                excluded_incident_triangle_count=second_excluded_count,
            )

        interpolation = float(alpha)
        first_nodal = np.asarray(first.nodal_kh_m2ps, dtype=np.float64)
        second_nodal = np.asarray(second.nodal_kh_m2ps, dtype=np.float64)
        nodal_kh = first_nodal + interpolation * (second_nodal - first_nodal)
        weights = np.asarray(location.barycentric_weights, dtype=np.float64)
        particle_kh = float(weights @ nodal_kh)
        gradient_x, gradient_y = self.mesh.triangle_linear_gradient(
            location.triangle_id,
            nodal_kh,
        )
        raw_current_triangle_kh = first.raw_current_triangle_kh_m2ps + interpolation * (
            second.raw_current_triangle_kh_m2ps - first.raw_current_triangle_kh_m2ps
        )
        floor_hit = bool(first.floor_hit or second.floor_hit)
        cap_hit = bool(first.cap_hit or second.cap_hit)
        valid_count = min(first.valid_incident_triangle_count, second.valid_incident_triangle_count)
        excluded_count = max(
            first.excluded_incident_triangle_count,
            second.excluded_incident_triangle_count,
        )
        diagnostics: dict[str, bool | float | int | str] = {
            "method": _smagorinsky_method_name(),
            "coefficient_cs": settings.coefficient_cs,
            "kh_m2ps": particle_kh,
            "d_kh_dx_mps": float(gradient_x),
            "d_kh_dy_mps": float(gradient_y),
            "raw_current_triangle_kh_m2ps": float(raw_current_triangle_kh),
            "floor_hit": floor_hit,
            "cap_hit": cap_hit,
            "valid_incident_triangle_count": int(valid_count),
            "excluded_incident_triangle_count": int(excluded_count),
            "valid_incident_triangle_count_before": int(first.valid_incident_triangle_count),
            "valid_incident_triangle_count_after": int(second.valid_incident_triangle_count),
            "excluded_incident_triangle_count_before": int(
                first.excluded_incident_triangle_count
            ),
            "excluded_incident_triangle_count_after": int(
                second.excluded_incident_triangle_count
            ),
            "triangle_id": int(location.triangle_id),
        }
        return DiffusionSample(
            coefficients=DiffusionCoefficients(
                particle_kh,
                particle_kh,
                settings.constant_kz_m2ps,
            ),
            diffusivity_divergence_mps=(float(gradient_x), float(gradient_y), 0.0),
            qc=SampleQC.OK,
            diagnostics=diagnostics,
        )

    def sample_smagorinsky_diffusion_at_time_index(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_index: int,
        settings: SmagorinskySettings,
        *,
        triangle_hint: int | None = None,
        support_z_m: float | None = None,
        enforce_geometric_bounds: bool = True,
    ) -> DiffusionSample:
        """在單一 OCM 時間列計算非線性 Smagorinsky endpoint。

        這是跨月份時間軸的明示 endpoint 介面。它先在該時間列完成每個 incident
        triangle 的速度梯度、面積加權 nodal ``Kh`` 以及 floor/cap，再回傳該 endpoint
        的粒子 ``Kh``、梯度與診斷；manager 之後才對兩個 endpoint 的結果做時間線性
        內插。因此不會把已經合成的總速度或不同月份的 Kh candidate 先相加，並且與
        將兩端列放入同一個連續 provider 後由原方法分別計算再內插的順序一致。

        ``support_z_m`` 和 ``enforce_geometric_bounds`` 的用途與
        ``sample_at_time_index`` 相同：跨月 caller 以兩端海面內插得到共同 query-time
        垂向支援值，端點只驗證 OCM layer／遮罩，最後再由 caller 做一次共同幾何 gate。
        省略時則使用該 endpoint 的海面政策，保持直接 caller 的獨立取樣語意。
        """

        if not isinstance(settings, SmagorinskySettings):
            raise TypeError("settings 必須是 SmagorinskySettings")
        settings.validate()
        index = self._validate_time_index(time_index)
        location = self.mesh.locate(x_m, y_m, triangle_hint=triangle_hint)
        if location is None:
            return _invalid_smagorinsky_sample(
                settings,
                SampleQC.OUTSIDE_HORIZONTAL_DOMAIN,
                triangle_id=None,
            )
        geometric_bounds = self._geometric_bounds_at_time(
            location,
            before=index,
            after=index,
            alpha=0.0,
        )
        if enforce_geometric_bounds and not _query_z_within_geometric_bounds(z_m, geometric_bounds):
            return _invalid_smagorinsky_sample(
                settings,
                SampleQC.VERTICAL_UNSUPPORTED,
                triangle_id=location.triangle_id,
            )
        target_z = z_m if support_z_m is None else support_z_m
        if not np.isfinite(target_z):
            return _invalid_smagorinsky_sample(
                settings,
                SampleQC.VERTICAL_UNSUPPORTED,
                triangle_id=location.triangle_id,
            )
        endpoint, endpoint_qc, valid_count, excluded_count = self._smagorinsky_time_slice(
            location,
            time_index=index,
            z_m=target_z,
            settings=settings,
        )
        if endpoint is None or endpoint_qc != SampleQC.OK:
            return _invalid_smagorinsky_sample(
                settings,
                endpoint_qc,
                triangle_id=location.triangle_id,
                valid_incident_triangle_count=valid_count,
                excluded_incident_triangle_count=excluded_count,
            )
        nodal_kh = np.asarray(endpoint.nodal_kh_m2ps, dtype=np.float64)
        weights = np.asarray(location.barycentric_weights, dtype=np.float64)
        particle_kh = float(weights @ nodal_kh)
        try:
            gradient_x, gradient_y = self.mesh.triangle_linear_gradient(
                location.triangle_id,
                nodal_kh,
            )
        except (TypeError, ValueError, IndexError):
            return _invalid_smagorinsky_sample(
                settings,
                SampleQC.NUMERICAL_FAILURE,
                triangle_id=location.triangle_id,
                valid_incident_triangle_count=valid_count,
                excluded_incident_triangle_count=excluded_count,
            )
        diagnostics: dict[str, bool | float | int | str] = {
            "method": _smagorinsky_method_name(),
            "coefficient_cs": settings.coefficient_cs,
            "kh_m2ps": particle_kh,
            "d_kh_dx_mps": float(gradient_x),
            "d_kh_dy_mps": float(gradient_y),
            "raw_current_triangle_kh_m2ps": float(endpoint.raw_current_triangle_kh_m2ps),
            "floor_hit": bool(endpoint.floor_hit),
            "cap_hit": bool(endpoint.cap_hit),
            "valid_incident_triangle_count": int(valid_count),
            "excluded_incident_triangle_count": int(excluded_count),
            "valid_incident_triangle_count_before": int(valid_count),
            "valid_incident_triangle_count_after": int(valid_count),
            "excluded_incident_triangle_count_before": int(excluded_count),
            "excluded_incident_triangle_count_after": int(excluded_count),
            "triangle_id": int(location.triangle_id),
        }
        return DiffusionSample(
            coefficients=DiffusionCoefficients(
                particle_kh,
                particle_kh,
                settings.constant_kz_m2ps,
            ),
            diffusivity_divergence_mps=(float(gradient_x), float(gradient_y), 0.0),
            qc=SampleQC.OK,
            diagnostics=diagnostics,
        )

    def _spatial_at_time(
        self, location: MeshLocation, *, time_index: int, z_m: float
    ) -> tuple[tuple[np.ndarray, float, float] | None, SampleQC]:
        """在三個三角形節點完成垂向及水平內插，並區分乾涸與深度資料不足。"""

        face = location.source_face_local_index
        wetdry = float(self.wetdry_elem[time_index, face])
        if not np.isfinite(wetdry) or not np.isclose(wetdry, self.wet_value, atol=0.1):
            return None, SampleQC.DRY_FACE
        node_values: list[np.ndarray] = []
        spans: list[float] = []
        for node in location.node_indices:
            sampled = self._vertical_node_sample(time_index=time_index, node_index=node, z_m=z_m)
            if sampled is None:
                return None, SampleQC.VERTICAL_UNSUPPORTED
            values, span = sampled
            node_values.append(values)
            spans.append(span)
        weights = np.asarray(location.barycentric_weights, dtype=np.float64)
        combined = weights @ np.asarray(node_values)
        eta_nodes = np.asarray(self.elev[time_index, list(location.node_indices)], dtype=np.float64)
        if not np.all(np.isfinite(eta_nodes)):
            return None, SampleQC.VERTICAL_UNSUPPORTED
        eta = float(weights @ eta_nodes)
        vertical_scale = max(min((value for value in spans if value > 0), default=0.1), 0.1)
        return (combined, eta, vertical_scale), SampleQC.OK

    def _geometric_bounds_at_time(
        self,
        location: MeshLocation,
        *,
        before: int,
        after: int,
        alpha: float,
    ) -> tuple[float, float] | None:
        """在速度垂向夾層失敗時，獨立整理仍可取得的海面／海床幾何上下文。

        ``elev`` 是每個 OCM 節點的瞬時海面高程，``source_depth_m`` 是 native mesh
        的靜態水深；兩者都使用公尺、海面向上為正。這裡只做同一個已通過時間
        bracket 的前後時間片與目前 triangle 權重內插，不讀取速度、不做最近層外插，
        也不把失敗速度轉成零值。回傳 ``None`` 代表海面、海床或 triangle 權重本身
        不足以證明垂向邊界，呼叫端必須保留原本的 fail-closed 狀態。

        這個幾何 context 與 ``_spatial_at_time`` 的速度支撐刻意分開：粒子可能只因
        已經越過海面而無法由上下兩層夾住 ``z_m``，但仍可由有限 ``elev`` 與 mesh
        水深證明「應套用海面反射」；反之，乾面、時間缺口及未知幾何不會因本 helper
        而被放寬。
        """

        nodes = np.asarray(location.node_indices, dtype=np.int64)
        weights = np.asarray(location.barycentric_weights, dtype=np.float64)
        if nodes.ndim != 1 or nodes.size != 3 or weights.shape != (3,):
            return None
        if not np.all(np.isfinite(weights)):
            return None
        eta_before = np.asarray(self.elev[before, nodes], dtype=np.float64)
        eta_after = np.asarray(self.elev[after, nodes], dtype=np.float64)
        source_depth = np.asarray(self.mesh.source_depth_m[nodes], dtype=np.float64)
        if not (
            np.all(np.isfinite(eta_before))
            and np.all(np.isfinite(eta_after))
            and np.all(np.isfinite(source_depth))
        ):
            return None
        eta_before_value = float(weights @ eta_before)
        eta_after_value = float(weights @ eta_after)
        eta = eta_before_value + float(alpha) * (eta_after_value - eta_before_value)
        bed = -float(weights @ source_depth)
        if (
            not np.isfinite(eta)
            or not np.isfinite(bed)
            or bed > eta + VERTICAL_BOUNDARY_TOLERANCE_M
        ):
            return None
        return eta, bed

    def _validate_time_index(self, time_index: int) -> int:
        """驗證單一 endpoint 的時間索引，避免跨月 adapter 讀錯列。

        endpoint API 只接受這個月份自身 ``time_utc_ns`` 的整數列索引。索引不是
        UTC 時刻，因此不能由 caller 以負值、浮點或布林值偷偷繞過月份時間軸檢查；
        超出範圍也直接上拋，讓月份管理器把真正的產品／程式錯誤與可回傳的時間 QC
        分開。這個驗證不修改任何 memory-map 陣列。
        """

        if isinstance(time_index, bool) or not isinstance(time_index, (int, np.integer)):
            raise TypeError("time_index 必須是真正的 integer，不可為 bool")
        normalized = int(time_index)
        if normalized < 0 or normalized >= self.time_utc_ns.size:
            raise IndexError(f"time_index 超出 OCM 月份範圍：{normalized}")
        return normalized

    def geometry_at_time_index(
        self,
        x_m: float,
        y_m: float,
        time_index: int,
        *,
        triangle_hint: int | None = None,
    ) -> tuple[MeshLocation | None, tuple[float, float] | None]:
        """回傳單一 OCM endpoint 的 mesh location 與 ``(eta, bed)`` 幾何上下界。

        跨月份 manager 需要先以兩端的海面／海床建立 query-time 幾何，再決定固定
        ``z_m`` 應送入哪一個 endpoint 的垂向層。此方法只讀同一月份的 ``elev``、native
        靜態水深與 mesh 權重，不讀取或合成速度，也不對缺值做任何補值。水平域外會
        回傳 ``(None, None)``；合法索引但幾何缺值則保留 location 並回傳 ``None``，
        讓呼叫端維持原本 ``VERTICAL_UNSUPPORTED`` 的 fail-closed 語意。
        """

        index = self._validate_time_index(time_index)
        location = self.mesh.locate(x_m, y_m, triangle_hint=triangle_hint)
        if location is None:
            return None, None
        return location, self._geometric_bounds_at_time(
            location,
            before=index,
            after=index,
            alpha=0.0,
        )

    def sample_at_time_index(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_index: int,
        *,
        triangle_hint: int | None = None,
        support_z_m: float | None = None,
        enforce_geometric_bounds: bool = True,
    ) -> VelocitySample:
        """在一個 OCM 時間列取出原始 current／Kz endpoint，不進行時間內插。

        ``time_index`` 指向這個月份實際的 UTC 時間列；跨月流程會先從兩個月份
        建立全域 before／after，再用本方法各取一次。水平三角形、垂向 ``zcor``、
        ``wetdry``、OCM current、垂向速度、擴散係數與海面幾何沿用一般
        ``sample`` 的支援規則，故不會把缺值、乾面或海床下方轉成零速度。

        ``support_z_m`` 是 manager 依兩端內插後的 query-time 海面所決定的固定
        垂向查詢值；省略時才由此 endpoint 自己的海面容許帶推導。跨月 caller
        應傳入前者，並將 ``enforce_geometric_bounds=False``，最後在兩端原始值
        內插後用 query-time ``eta``／海床做一次共同 gate。這個例外只避免 endpoint
        surface hold 依月份各自判定，並不放寬垂向層支援或資料品質。
        """

        index = self._validate_time_index(time_index)
        location = self.mesh.locate(x_m, y_m, triangle_hint=triangle_hint)
        if location is None:
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                SampleQC.OUTSIDE_HORIZONTAL_DOMAIN,
                forcing_month_id=self.month_id,
            )
        geometric_bounds = self._geometric_bounds_at_time(
            location,
            before=index,
            after=index,
            alpha=0.0,
        )
        target_z = z_m if support_z_m is None else support_z_m
        if not np.isfinite(target_z):
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                *(geometric_bounds or (np.nan, np.nan)),
                np.sqrt(location.triangle_area_m2),
                np.nan,
                SampleQC.VERTICAL_UNSUPPORTED,
                source_face_id=location.source_face_global_index,
                triangle_id=location.triangle_id,
                forcing_month_id=self.month_id,
            )
        if self.use_numba_kernel:
            face = location.source_face_local_index
            wet_value = float(self.wetdry_elem[index, face])
            if not np.isfinite(wet_value) or not np.isclose(wet_value, self.wet_value, atol=0.1):
                values: np.ndarray | None = None
                vertical_scale = np.nan
                endpoint_qc = SampleQC.DRY_FACE
            elif geometric_bounds is None:
                values = None
                vertical_scale = np.nan
                endpoint_qc = SampleQC.VERTICAL_UNSUPPORTED
            else:
                values, vertical_scale, valid = interpolate_ocm_support_numba(
                    self.hvel,
                    self.vertical_velocity,
                    self.zcor,
                    self.diffusivity,
                    index,
                    index,
                    0.0,
                    np.asarray(location.node_indices, dtype=np.int64),
                    np.asarray(location.barycentric_weights, dtype=np.float64),
                    target_z,
                    SURFACE_BOUNDARY_TOLERANCE_M,
                )
                endpoint_qc = SampleQC.OK if valid else SampleQC.VERTICAL_UNSUPPORTED
        else:
            sampled, endpoint_qc = self._spatial_at_time(
                location,
                time_index=index,
                z_m=target_z,
            )
            if sampled is None:
                values = None
                vertical_scale = np.nan
            else:
                values, _, vertical_scale = sampled
        eta, bed_z = geometric_bounds if geometric_bounds is not None else (np.nan, np.nan)
        if values is None or endpoint_qc != SampleQC.OK:
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                eta,
                bed_z,
                np.sqrt(location.triangle_area_m2),
                float(vertical_scale),
                endpoint_qc,
                source_face_id=location.source_face_global_index,
                triangle_id=location.triangle_id,
                forcing_month_id=self.month_id,
            )
        if enforce_geometric_bounds and not _query_z_within_geometric_bounds(z_m, geometric_bounds):
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                eta,
                bed_z,
                np.sqrt(location.triangle_area_m2),
                float(vertical_scale),
                SampleQC.VERTICAL_UNSUPPORTED,
                source_face_id=location.source_face_global_index,
                triangle_id=location.triangle_id,
                forcing_month_id=self.month_id,
            )
        return VelocitySample(
            u_mps=float(values[0]),
            v_mps=float(values[1]),
            w_mps=float(values[2]),
            eta_m=eta,
            bed_z_m=bed_z,
            horizontal_scale_m=float(np.sqrt(location.triangle_area_m2)),
            vertical_scale_m=float(vertical_scale),
            qc=SampleQC.OK,
            source_face_id=location.source_face_global_index,
            triangle_id=location.triangle_id,
            forcing_month_id=self.month_id,
            diagnostics={"kz_m2ps": float(values[3])},
        )

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        triangle_hint: int | None = None,
    ) -> VelocitySample:
        """取得 OCM 東、北、垂向速度、擴散係數、海面高度與水深。

        取樣失敗時仍回傳品質檢查旗標，讓呼叫端知道是超出範圍、時間缺口、乾涸或深度
        資料不足，而不是只得到沒有原因的空值。端點 surface hold 只解決移動海面造成的
        固定 z 支援洞；完成端點速度／物理量時間內插後，仍以 query-time ``eta`` 與 native
        海床共同檢查 ``z_m``，所以真正越過海面或海床的 query 不會被 hold 放寬。端點
        eta、海床或必要 layer 缺值仍 fail closed。``triangle_hint`` 只作 native mesh
        定位的效能提示，不是物理資料；失效或過期提示由 mesh locator 自動回退原本的
        uniform-bin 候選搜尋，因此不會改變三角形 provenance、遮罩或數值結果。
        """

        location = self.mesh.locate(x_m, y_m, triangle_hint=triangle_hint)
        if location is None:
            return VelocitySample(
                0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, SampleQC.OUTSIDE_HORIZONTAL_DOMAIN
            )
        before, after, alpha, time_qc = _time_bracket(
            self.time_utc_ns, time_utc_ns, maximum_gap_ns=self.maximum_time_gap_ns
        )
        if time_qc != SampleQC.OK:
            return VelocitySample(0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, time_qc)
        # 先保存只依賴 elev 與 native 水深的幾何上下文。若後續速度的垂向夾層失敗，
        # 這些有限值仍可讓 engine 判斷是否為已證明的海面穿越；若幾何本身缺值，
        # helper 會回傳 None，失敗樣本仍以 NaN 表達未知，不會因此取得邊界特權。
        geometric_bounds = self._geometric_bounds_at_time(
            location,
            before=before,
            after=after,
            alpha=alpha,
        )
        support_z_m = _query_z_for_vertical_support(z_m, geometric_bounds)
        if self.use_numba_kernel:
            face = location.source_face_local_index
            wet_values = np.asarray(self.wetdry_elem[[before, after], face], dtype=np.float64)
            if not np.all(np.isfinite(wet_values)) or not np.allclose(wet_values, self.wet_value, atol=0.1):
                first = second = None
                first_qc = second_qc = SampleQC.DRY_FACE
            elif geometric_bounds is None:
                first = second = None
                first_qc = second_qc = SampleQC.VERTICAL_UNSUPPORTED
            else:
                values, vertical_scale, valid = interpolate_ocm_support_numba(
                    self.hvel,
                    self.vertical_velocity,
                    self.zcor,
                    self.diffusivity,
                    before,
                    after,
                    alpha,
                    np.asarray(location.node_indices, dtype=np.int64),
                    np.asarray(location.barycentric_weights, dtype=np.float64),
                    support_z_m,
                    SURFACE_BOUNDARY_TOLERANCE_M,
                )
                if valid:
                    eta_first = float(
                        np.asarray(location.barycentric_weights)
                        @ np.asarray(self.elev[before, list(location.node_indices)], dtype=np.float64)
                    )
                    eta_second = float(
                        np.asarray(location.barycentric_weights)
                        @ np.asarray(self.elev[after, list(location.node_indices)], dtype=np.float64)
                    )
                    eta_combined = eta_first + alpha * (eta_second - eta_first)
                    first = (values, eta_combined, vertical_scale)
                    second = first
                    first_qc = second_qc = SampleQC.OK
                else:
                    first = second = None
                    first_qc = second_qc = SampleQC.VERTICAL_UNSUPPORTED
        else:
            first, first_qc = self._spatial_at_time(location, time_index=before, z_m=support_z_m)
            if after == before:
                second, second_qc = first, first_qc
            else:
                second, second_qc = self._spatial_at_time(location, time_index=after, z_m=support_z_m)
        if first is None or second is None:
            eta, bed_z = geometric_bounds if geometric_bounds is not None else (np.nan, np.nan)
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                eta,
                bed_z,
                np.sqrt(location.triangle_area_m2),
                np.nan,
                first_qc | second_qc,
                source_face_id=location.source_face_global_index,
                triangle_id=location.triangle_id,
                forcing_month_id=self.month_id,
            )
        values = first[0] + alpha * (second[0] - first[0])
        vertical_scale = min(first[2], second[2])
        # ``geometric_bounds`` 是 query-time 的單一權威上下界；不使用 endpoint hold
        # 後的 layer z 來代替實際海面，也不重新以粒子位置猜測水深。
        eta, bed_z = geometric_bounds if geometric_bounds is not None else (np.nan, np.nan)
        if not _query_z_within_geometric_bounds(z_m, geometric_bounds):
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                eta,
                bed_z,
                np.sqrt(location.triangle_area_m2),
                vertical_scale,
                SampleQC.VERTICAL_UNSUPPORTED,
                source_face_id=location.source_face_global_index,
                triangle_id=location.triangle_id,
                forcing_month_id=self.month_id,
            )
        return VelocitySample(
            u_mps=float(values[0]),
            v_mps=float(values[1]),
            w_mps=float(values[2]),
            eta_m=eta,
            bed_z_m=bed_z,
            horizontal_scale_m=float(np.sqrt(location.triangle_area_m2)),
            vertical_scale_m=vertical_scale,
            source_face_id=location.source_face_global_index,
            triangle_id=location.triangle_id,
            forcing_month_id=self.month_id,
            diagnostics={"kz_m2ps": float(values[3])},
        )


class NWWAnalysisMonth:
    """一個 NWW3 analysis 月份的保守規則格網取樣器。"""

    def __init__(
        self,
        *,
        month_id: str,
        lon: np.ndarray,
        lat: np.ndarray,
        time_utc_ns: np.ndarray,
        significant_wave_height: np.ndarray,
        peak_frequency: np.ndarray,
        peak_direction_raw_deg: np.ndarray,
        valid_mask_wave: np.ndarray,
        qc_flags: np.ndarray,
        maximum_time_gap_seconds: float = 5_400.0,
    ) -> None:
        """檢查時間、緯度、經度三個軸的資料格式，並保留唯讀大型陣列。"""

        self.month_id = month_id
        self.lon = np.asarray(lon)
        self.lat = np.asarray(lat)
        self.time_utc_ns = np.asarray(time_utc_ns)
        self.significant_wave_height = significant_wave_height
        self.peak_frequency = peak_frequency
        self.peak_direction_raw_deg = peak_direction_raw_deg
        self.valid_mask_wave = valid_mask_wave
        self.qc_flags = qc_flags
        self.maximum_time_gap_ns = int(round(maximum_time_gap_seconds * 1_000_000_000))
        expected = (self.time_utc_ns.size, self.lat.size, self.lon.size)
        arrays = (
            self.significant_wave_height,
            self.peak_frequency,
            self.peak_direction_raw_deg,
            self.valid_mask_wave,
            self.qc_flags,
        )
        if any(item.shape != expected for item in arrays):
            raise ValueError("NWW 欄位必須全部是 (time,lat,lon)")
        if self.valid_mask_wave.dtype != np.bool_ or self.qc_flags.dtype != np.uint16:
            raise ValueError("NWW valid mask 必須 bool，qc_flags 必須 uint16")
        if (
            np.any(np.diff(self.lon) <= 0)
            or np.any(np.diff(self.lat) <= 0)
            or np.any(np.diff(self.time_utc_ns) <= 0)
        ):
            raise ValueError("NWW lon/lat/time 軸必須嚴格遞增")

    @classmethod
    def from_directories(cls, grid_dir: str | Path, month_dir: str | Path) -> NWWAnalysisMonth:
        """從波浪格網與月份資料夾建立唯讀取樣器。"""

        grid = Path(grid_dir)
        month = Path(month_dir)

        def load(root: Path, name: str) -> np.ndarray:
            """從格網或月份資料夾唯讀開啟 NWW 陣列，不載入任意序列化物件，也不猜測缺檔。"""

            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"缺少 NWW array：{path}")
            return np.load(path, mmap_mode="r", allow_pickle=False)

        return cls(
            month_id=month.name,
            lon=load(grid, "lon.npy"),
            lat=load(grid, "lat.npy"),
            time_utc_ns=load(month, "time_utc_ns.npy"),
            significant_wave_height=load(month, "significant_wave_height.npy"),
            peak_frequency=load(month, "peak_frequency.npy"),
            peak_direction_raw_deg=load(month, "peak_direction_raw_deg.npy"),
            valid_mask_wave=load(month, "valid_mask_wave.npy"),
            qc_flags=load(month, "qc_flags.npy"),
        )

    def _sample_spatial_at_time(self, lon: float, lat: float, time_index: int) -> WaveSample:
        """對單一 NWW3 時間列做四角空間內插，保留尚未正規化的方向圓向量。

        這個 helper 不碰時間軸；它只讀 ``time_index`` 的四個水平角點，並沿用原有
        valid mask、QC 聯集、有限值、波高／頻率與方向物理檢查。方向向量的大小不能
        被丟掉，因為跨月時間內插必須和單一連續月份 provider 的
        ``direction_x``／``direction_y`` 累加順序一致。
        """

        if lon < self.lon[0] or lon > self.lon[-1] or lat < self.lat[0] or lat > self.lat[-1]:
            return WaveSample(np.nan, np.nan, np.nan, 0, SampleQC.OUTSIDE_HORIZONTAL_DOMAIN)
        x_after = min(max(int(np.searchsorted(self.lon, lon, side="right")), 1), self.lon.size - 1)
        y_after = min(max(int(np.searchsorted(self.lat, lat, side="right")), 1), self.lat.size - 1)
        x0, x1 = x_after - 1, x_after
        y0, y1 = y_after - 1, y_after
        wx = float((lon - self.lon[x0]) / (self.lon[x1] - self.lon[x0]))
        wy = float((lat - self.lat[y0]) / (self.lat[y1] - self.lat[y0]))
        corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
        spatial_weights = np.array(
            [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx],
            dtype=np.float64,
        )
        mask = np.array([self.valid_mask_wave[time_index, y, x] for y, x in corners], dtype=bool)
        qc_union = int(
            np.bitwise_or.reduce([self.qc_flags[time_index, y, x] for y, x in corners])
        )
        if not np.all(mask):
            return WaveSample(np.nan, np.nan, np.nan, qc_union, SampleQC.WAVE_UNSUPPORTED)
        hs_values = np.array(
            [self.significant_wave_height[time_index, y, x] for y, x in corners],
            dtype=np.float64,
        )
        fp_values = np.array(
            [self.peak_frequency[time_index, y, x] for y, x in corners],
            dtype=np.float64,
        )
        directions = np.deg2rad(
            np.array(
                [self.peak_direction_raw_deg[time_index, y, x] for y, x in corners],
                dtype=np.float64,
            )
        )
        if not (
            np.all(np.isfinite(hs_values))
            and np.all(np.isfinite(fp_values))
            and np.all(np.isfinite(directions))
        ):
            return WaveSample(np.nan, np.nan, np.nan, qc_union, SampleQC.WAVE_UNSUPPORTED)
        hs = float(spatial_weights @ hs_values)
        fp = float(spatial_weights @ fp_values)
        direction_x = float(spatial_weights @ np.sin(directions))
        direction_y = float(spatial_weights @ np.cos(directions))
        if hs < 0 or fp <= 0 or abs(direction_x) + abs(direction_y) <= 1e-15:
            return WaveSample(
                hs,
                fp,
                np.nan,
                qc_union,
                SampleQC.INVALID_PHYSICS,
                direction_x=direction_x,
                direction_y=direction_y,
            )
        raw_direction = float(np.degrees(np.arctan2(direction_x, direction_y)) % 360.0)
        return WaveSample(
            hs,
            fp,
            raw_direction,
            qc_union,
            SampleQC.OK,
            direction_x=direction_x,
            direction_y=direction_y,
        )

    def _validate_time_index(self, time_index: int) -> int:
        """驗證 NWW3 單一 endpoint 時間列，拒絕隱式負索引與非整數。"""

        if isinstance(time_index, bool) or not isinstance(time_index, (int, np.integer)):
            raise TypeError("time_index 必須是真正的 integer，不可為 bool")
        normalized = int(time_index)
        if normalized < 0 or normalized >= self.time_utc_ns.size:
            raise IndexError(f"time_index 超出 NWW 月份範圍：{normalized}")
        return normalized

    def sample_at_time_index(self, lon: float, lat: float, time_index: int) -> WaveSample:
        """回傳單一 NWW3 endpoint 的四角空間內插結果，不進行時間內插。

        跨月份 manager 以兩個月份的實際時間列建立全域 before／after，再呼叫本方法。
        返回值保留方向圓向量，呼叫端可先內插 ``Hs``、``fp`` 與向量後再交給有限水深
        Stokes 公式；缺角點、遮罩、非有限值與物理失敗維持原本非零 QC。
        """

        index = self._validate_time_index(time_index)
        return self._sample_spatial_at_time(lon, lat, index)

    def sample(self, lon: float, lat: float, time_utc_ns: int) -> WaveSample:
        """僅在四個周圍格點都有效時做空間與時間內插；方向以圓向量處理。

        時間內插先對每個 endpoint 的 ``Hs``、``fp`` 與未正規化方向向量做線性運算，
        再把向量轉回角度。這個順序也供跨月 manager 使用，避免在 0／360 度邊界把
        已轉成角度的數值直接平均。
        """

        before, after, alpha, time_qc = _time_bracket(
            self.time_utc_ns,
            time_utc_ns,
            maximum_gap_ns=self.maximum_time_gap_ns,
        )
        if time_qc != SampleQC.OK:
            return WaveSample(np.nan, np.nan, np.nan, 0, time_qc)
        first = self.sample_at_time_index(lon, lat, before)
        if not first.valid:
            return first
        if after == before:
            return first
        second = self.sample_at_time_index(lon, lat, after)
        if not second.valid:
            # 舊版逐 endpoint 迴圈在第二端失敗前已累積第一端的 QC flags；保留第二端
            # 的非零 SampleQC，同時把兩端空間四角的原始旗標聯集，避免跨時間失敗時
            # 遺失已讀取 endpoint 的資料品質 provenance。
            return replace(second, qc_flags=int(first.qc_flags | second.qc_flags))
        if first.direction_x is None or first.direction_y is None:
            return WaveSample(np.nan, np.nan, np.nan, first.qc_flags, SampleQC.WAVE_UNSUPPORTED)
        if second.direction_x is None or second.direction_y is None:
            return WaveSample(np.nan, np.nan, np.nan, second.qc_flags, SampleQC.WAVE_UNSUPPORTED)
        interpolation = float(alpha)
        hs = first.significant_wave_height_m + interpolation * (
            second.significant_wave_height_m - first.significant_wave_height_m
        )
        fp = first.peak_frequency_hz + interpolation * (
            second.peak_frequency_hz - first.peak_frequency_hz
        )
        direction_x = first.direction_x + interpolation * (second.direction_x - first.direction_x)
        direction_y = first.direction_y + interpolation * (second.direction_y - first.direction_y)
        qc_union = int(first.qc_flags | second.qc_flags)
        if hs < 0 or fp <= 0 or abs(direction_x) + abs(direction_y) <= 1e-15:
            return WaveSample(
                hs,
                fp,
                np.nan,
                qc_union,
                SampleQC.INVALID_PHYSICS,
                direction_x=direction_x,
                direction_y=direction_y,
            )
        raw_direction = float(np.degrees(np.arctan2(direction_x, direction_y)) % 360.0)
        return WaveSample(
            hs,
            fp,
            raw_direction,
            qc_union,
            SampleQC.OK,
            direction_x=direction_x,
            direction_y=direction_y,
        )


class CombinedMonthForcing:
    """合併同月海流、波浪、座標投影與粒子浮沉速度的取樣器。

    有效回傳值除了既有總速度，也保留同一次 OCM／NWW3 取樣實際使用的分項：OCM
    東、北、向上 current、Stokes 水平東／北速度，以及以向上為正的垂向沉降速度。
    Stokes 垂向與沉降水平不在公式中，因此不新增欄位或查詢。關閉 Stokes 時，Stokes
    水平分項是明知的 0；NWW3 月份缺失時則回傳非零 ``WAVE_UNSUPPORTED``，不把缺值
    轉成有效的零 Stokes。
    """

    def __init__(
        self,
        *,
        ocm: OCMNativeMonth,
        nww: NWWAnalysisMonth | None,
        projection: DomainProjection,
        settling_velocity_mps: float,
        include_stokes: bool,
    ) -> None:
        """不納入波浪表面漂移的案例可不讀波浪資料；納入時則必須提供 NWW 資料。"""

        if include_stokes and nww is None:
            raise ValueError("include_stokes=True 時必須提供 NWW month")
        self.ocm = ocm
        self.nww = nww
        self.projection = projection
        self.settling_velocity_mps = float(settling_velocity_mps)
        self.include_stokes = include_stokes

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        triangle_hint: int | None = None,
    ) -> VelocitySample:
        """合成海流、有限水深波浪表面漂移與粒子浮沉後回傳物理時間往後的速度。

        mesh hint 只傳給同月 OCM locator；NWW 的規則經緯度格網不使用此提示。其餘
        Stokes、沉降、時間、乾點、垂向遮罩與品質檢查流程保持原本順序與語意。
        """

        current = self.ocm.sample(
            x_m, y_m, z_m, time_utc_ns, triangle_hint=triangle_hint
        )
        if not current.valid:
            return current
        stokes_u = 0.0
        stokes_v = 0.0
        diagnostics = dict(current.diagnostics)
        if self.include_stokes:
            assert self.nww is not None
            lon, lat = self.projection.unproject(x_m, y_m)
            wave = self.nww.sample(float(lon), float(lat), time_utc_ns)
            if not wave.valid:
                return VelocitySample(
                    current.u_mps,
                    current.v_mps,
                    current.w_mps + self.settling_velocity_mps,
                    current.eta_m,
                    current.bed_z_m,
                    current.horizontal_scale_m,
                    current.vertical_scale_m,
                    wave.qc,
                    current.source_face_id,
                    current.triangle_id,
                    current.forcing_month_id,
                    {**diagnostics, "nww_qc_flags": wave.qc_flags},
                )
            try:
                stokes = finite_depth_stokes(
                    significant_wave_height_m=wave.significant_wave_height_m,
                    peak_frequency_hz=wave.peak_frequency_hz,
                    direction_raw_deg=wave.peak_direction_raw_deg,
                    particle_z_m=z_m,
                    surface_z_m=current.eta_m,
                    bed_z_m=current.bed_z_m,
                )
            except ValueError:
                return VelocitySample(
                    current.u_mps,
                    current.v_mps,
                    current.w_mps + self.settling_velocity_mps,
                    current.eta_m,
                    current.bed_z_m,
                    current.horizontal_scale_m,
                    current.vertical_scale_m,
                    SampleQC.INVALID_PHYSICS,
                    current.source_face_id,
                    current.triangle_id,
                    current.forcing_month_id,
                    diagnostics,
                )
            stokes_u, stokes_v = stokes.u_mps, stokes.v_mps
            diagnostics.update(
                {
                    "stokes_u_mps": stokes_u,
                    "stokes_v_mps": stokes_v,
                    "stokes_kh": stokes.kh,
                    "wave_steepness_ka": stokes.steepness_ka,
                }
            )
        return VelocitySample(
            u_mps=current.u_mps + stokes_u,
            v_mps=current.v_mps + stokes_v,
            w_mps=current.w_mps + self.settling_velocity_mps,
            eta_m=current.eta_m,
            bed_z_m=current.bed_z_m,
            horizontal_scale_m=current.horizontal_scale_m,
            vertical_scale_m=current.vertical_scale_m,
            qc=current.qc,
            source_face_id=current.source_face_id,
            triangle_id=current.triangle_id,
            forcing_month_id=current.forcing_month_id,
            diagnostics=diagnostics,
            components=VelocityComponents(
                total_u_mps=current.u_mps + stokes_u,
                total_v_mps=current.v_mps + stokes_v,
                total_w_mps=current.w_mps + self.settling_velocity_mps,
                ocm_u_mps=current.u_mps,
                ocm_v_mps=current.v_mps,
                ocm_w_mps=current.w_mps,
                stokes_u_mps=stokes_u,
                stokes_v_mps=stokes_v,
                settling_w_mps=self.settling_velocity_mps,
            ),
        )

    def sample_from_endpoints(
        self,
        *,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        ocm_before: VelocitySample,
        ocm_after: VelocitySample,
        ocm_alpha: float,
        nww_before: WaveSample | None = None,
        nww_after: WaveSample | None = None,
        nww_alpha: float = 0.0,
        forcing_month_id: str | None = None,
        endpoint_month_before: str | None = None,
        endpoint_month_after: str | None = None,
        endpoint_nww_month_before: str | None = None,
        endpoint_nww_month_after: str | None = None,
    ) -> VelocitySample:
        """以兩個實際時間列合成跨月份速度，保留原始物理運算順序。

        ``ocm_before``／``ocm_after`` 已由各自月份的 endpoint API 在固定座標與垂向
        支援值取樣；本方法只對 OCM 原始 ``u/v/w/Kz`` 與幾何 ``eta/bed`` 做時間線性
        內插。若啟用 Stokes，NWW 端點的 ``Hs``、峰值頻率與未正規化方向圓向量另以
        ``nww_alpha`` 內插，之後才在查詢時刻對內插後的波浪與 query-time 水柱套用
        一次 ``finite_depth_stokes``。所以結果等同把相同兩端點放進單一連續月份
        provider，不會對已合成的 total velocity 做時間內插。

        端點任何一項不是有效樣本時，保留其原有 QC；幾何、triangle 或方向圓向量
        不一致則 fail closed。``forcing_month_id`` 必須由 manager 傳入單一合法
        ``YYYYMM``，以符合 observation/checkpoint 輸出欄位。有效樣本的兩端實際月份
        不重複寫入 diagnostics，應由這個欄位搭配輸入 manifest 重建；失敗樣本才保留
        endpoint identity 診斷，避免把混合來源誤標成單一資料檔。
        """

        if not isinstance(ocm_before, VelocitySample) or not isinstance(ocm_after, VelocitySample):
            raise TypeError("ocm_before/ocm_after 必須是 VelocitySample")
        if not isinstance(nww_alpha, (int, float, np.integer, np.floating)) or isinstance(
            nww_alpha, (bool, np.bool_)
        ):
            raise TypeError("nww_alpha 必須是有限數值")
        if not isinstance(ocm_alpha, (int, float, np.integer, np.floating)) or isinstance(
            ocm_alpha, (bool, np.bool_)
        ):
            raise TypeError("ocm_alpha 必須是有限數值")
        ocm_weight = float(ocm_alpha)
        nww_weight = float(nww_alpha)
        if not np.isfinite(ocm_weight) or not 0.0 <= ocm_weight <= 1.0:
            raise ValueError("ocm_alpha 必須位於 0 與 1 之間")
        if not np.isfinite(nww_weight) or not 0.0 <= nww_weight <= 1.0:
            raise ValueError("nww_alpha 必須位於 0 與 1 之間")
        if forcing_month_id is None:
            forcing_month_id = endpoint_month_after or endpoint_month_before
        if forcing_month_id is not None and (
            len(forcing_month_id) != 6 or not forcing_month_id.isdigit()
        ):
            raise ValueError("forcing_month_id 必須是 YYYYMM 或 None")
        source_face_id = (
            ocm_before.source_face_id
            if ocm_before.source_face_id == ocm_after.source_face_id
            else None
        )
        triangle_id = (
            ocm_before.triangle_id if ocm_before.triangle_id == ocm_after.triangle_id else None
        )
        diagnostics: dict[str, bool | float | int | str] = {
            "cross_month_endpoint": 1,
            "endpoint_month_before": endpoint_month_before or "",
            "endpoint_month_after": endpoint_month_after or "",
        }
        if endpoint_nww_month_before is not None:
            diagnostics["endpoint_nww_month_before"] = endpoint_nww_month_before
        if endpoint_nww_month_after is not None:
            diagnostics["endpoint_nww_month_after"] = endpoint_nww_month_after
        if source_face_id is None or triangle_id is None:
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                SampleQC.NUMERICAL_FAILURE,
                forcing_month_id=forcing_month_id,
                diagnostics={**diagnostics, "endpoint_identity_mismatch": 1},
            )
        if not ocm_before.valid or not ocm_after.valid:
            qc = ocm_before.qc | ocm_after.qc
            eta = _interpolate_optional_float(ocm_before.eta_m, ocm_after.eta_m, ocm_weight)
            bed = _interpolate_optional_float(ocm_before.bed_z_m, ocm_after.bed_z_m, ocm_weight)
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                eta,
                bed,
                min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
                np.nan,
                qc,
                source_face_id=source_face_id,
                triangle_id=triangle_id,
                forcing_month_id=forcing_month_id,
                diagnostics=diagnostics,
            )

        u = ocm_before.u_mps + ocm_weight * (ocm_after.u_mps - ocm_before.u_mps)
        v = ocm_before.v_mps + ocm_weight * (ocm_after.v_mps - ocm_before.v_mps)
        w = ocm_before.w_mps + ocm_weight * (ocm_after.w_mps - ocm_before.w_mps)
        eta = ocm_before.eta_m + ocm_weight * (ocm_after.eta_m - ocm_before.eta_m)
        bed = ocm_before.bed_z_m + ocm_weight * (ocm_after.bed_z_m - ocm_before.bed_z_m)
        vertical_scale = min(ocm_before.vertical_scale_m, ocm_after.vertical_scale_m)
        if not (
            np.all(np.isfinite((u, v, w, eta, bed, vertical_scale)))
            and bed <= eta + VERTICAL_BOUNDARY_TOLERANCE_M
            and _query_z_within_geometric_bounds(z_m, (eta, bed))
        ):
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                eta,
                bed,
                min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
                vertical_scale,
                SampleQC.VERTICAL_UNSUPPORTED,
                source_face_id=source_face_id,
                triangle_id=triangle_id,
                forcing_month_id=forcing_month_id,
                diagnostics=diagnostics,
            )
        # 有效樣本的 diagnostics 必須與既有 CombinedMonthForcing 完全相同，避免把
        # endpoint provenance 欄位誤當成速度物理量。兩端來源仍由 forcing_month_id 與
        # manager 的輸入 manifest 追蹤；只有失敗樣本才保留上面的 identity 診斷。
        diagnostics = {}
        kz_before = ocm_before.diagnostics.get("kz_m2ps")
        kz_after = ocm_after.diagnostics.get("kz_m2ps")
        if kz_before is not None and kz_after is not None:
            try:
                diagnostics["kz_m2ps"] = float(kz_before) + ocm_weight * (
                    float(kz_after) - float(kz_before)
                )
            except (TypeError, ValueError):
                diagnostics["kz_m2ps"] = np.nan
        stokes_u = 0.0
        stokes_v = 0.0
        if self.include_stokes:
            if nww_before is None or nww_after is None:
                return VelocitySample(
                    u,
                    v,
                    w + self.settling_velocity_mps,
                    eta,
                    bed,
                    min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
                    vertical_scale,
                    SampleQC.WAVE_UNSUPPORTED,
                    source_face_id=source_face_id,
                    triangle_id=triangle_id,
                    forcing_month_id=forcing_month_id,
                    diagnostics={**diagnostics, "nww_month_missing": 1},
                )
            if not nww_before.valid or not nww_after.valid:
                return VelocitySample(
                    u,
                    v,
                    w + self.settling_velocity_mps,
                    eta,
                    bed,
                    min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
                    vertical_scale,
                    nww_before.qc | nww_after.qc,
                    source_face_id=source_face_id,
                    triangle_id=triangle_id,
                    forcing_month_id=forcing_month_id,
                    diagnostics={
                        **diagnostics,
                        "nww_qc_flags": int(nww_before.qc_flags | nww_after.qc_flags),
                    },
                )
            if (
                nww_before.direction_x is None
                or nww_before.direction_y is None
                or nww_after.direction_x is None
                or nww_after.direction_y is None
            ):
                return VelocitySample(
                    u,
                    v,
                    w + self.settling_velocity_mps,
                    eta,
                    bed,
                    min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
                    vertical_scale,
                    SampleQC.WAVE_UNSUPPORTED,
                    source_face_id=source_face_id,
                    triangle_id=triangle_id,
                    forcing_month_id=forcing_month_id,
                    diagnostics={**diagnostics, "nww_direction_vector_missing": 1},
                )
            hs = nww_before.significant_wave_height_m + nww_weight * (
                nww_after.significant_wave_height_m - nww_before.significant_wave_height_m
            )
            fp = nww_before.peak_frequency_hz + nww_weight * (
                nww_after.peak_frequency_hz - nww_before.peak_frequency_hz
            )
            direction_x = nww_before.direction_x + nww_weight * (
                nww_after.direction_x - nww_before.direction_x
            )
            direction_y = nww_before.direction_y + nww_weight * (
                nww_after.direction_y - nww_before.direction_y
            )
            if hs < 0 or fp <= 0 or abs(direction_x) + abs(direction_y) <= 1e-15:
                return VelocitySample(
                    u,
                    v,
                    w + self.settling_velocity_mps,
                    eta,
                    bed,
                    min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
                    vertical_scale,
                    SampleQC.INVALID_PHYSICS,
                    source_face_id=source_face_id,
                    triangle_id=triangle_id,
                    forcing_month_id=forcing_month_id,
                    diagnostics=diagnostics,
                )
            raw_direction = float(np.degrees(np.arctan2(direction_x, direction_y)) % 360.0)
            try:
                stokes = finite_depth_stokes(
                    significant_wave_height_m=float(hs),
                    peak_frequency_hz=float(fp),
                    direction_raw_deg=raw_direction,
                    particle_z_m=z_m,
                    surface_z_m=eta,
                    bed_z_m=bed,
                )
            except ValueError:
                return VelocitySample(
                    u,
                    v,
                    w + self.settling_velocity_mps,
                    eta,
                    bed,
                    min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
                    vertical_scale,
                    SampleQC.INVALID_PHYSICS,
                    source_face_id=source_face_id,
                    triangle_id=triangle_id,
                    forcing_month_id=forcing_month_id,
                    diagnostics=diagnostics,
                )
            stokes_u, stokes_v = stokes.u_mps, stokes.v_mps
            diagnostics.update(
                {
                    "stokes_u_mps": stokes_u,
                    "stokes_v_mps": stokes_v,
                    "stokes_kh": stokes.kh,
                    "wave_steepness_ka": stokes.steepness_ka,
                }
            )
        total_u = u + stokes_u
        total_v = v + stokes_v
        total_w = w + self.settling_velocity_mps
        return VelocitySample(
            u_mps=total_u,
            v_mps=total_v,
            w_mps=total_w,
            eta_m=eta,
            bed_z_m=bed,
            horizontal_scale_m=min(ocm_before.horizontal_scale_m, ocm_after.horizontal_scale_m),
            vertical_scale_m=vertical_scale,
            qc=SampleQC.OK,
            source_face_id=source_face_id,
            triangle_id=triangle_id,
            forcing_month_id=forcing_month_id,
            diagnostics=diagnostics,
            components=VelocityComponents(
                total_u_mps=total_u,
                total_v_mps=total_v,
                total_w_mps=total_w,
                ocm_u_mps=u,
                ocm_v_mps=v,
                ocm_w_mps=w,
                stokes_u_mps=stokes_u,
                stokes_v_mps=stokes_v,
                settling_w_mps=self.settling_velocity_mps,
            ),
        )

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """以既有四參數 ``VelocityProvider`` 介面取樣，且不攜帶 mesh hint。"""

        return self.sample(x_m, y_m, z_m, time_utc_ns)


class MonthlyCombinedForcing:
    """依每個 RK stage UTC 選月份的唯讀 provider，不在缺月時外插。"""

    def __init__(self, months: Mapping[str, CombinedMonthForcing]) -> None:
        """月份 key 必須是唯一 YYYYMM；空 mapping 無法積分。"""

        if not months or any(len(key) != 6 or not key.isdigit() for key in months):
            raise ValueError("months 必須是非空 YYYYMM -> forcing mapping")
        self.months = dict(months)

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        triangle_hint: int | None = None,
    ) -> VelocitySample:
        """以 UTC 月份選取 adapter 並傳遞 mesh hint；缺月明確回傳範圍外狀態。"""

        month_id = datetime.fromtimestamp(time_utc_ns / 1_000_000_000, tz=UTC).strftime("%Y%m")
        provider = self.months.get(month_id)
        if provider is None:
            return VelocitySample(0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, SampleQC.OUTSIDE_TIME_RANGE)
        return provider.sample(
            x_m, y_m, z_m, time_utc_ns, triangle_hint=triangle_hint
        )

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """以既有四參數 provider 介面選月取樣，且不攜帶 mesh hint。"""

        return self.sample(x_m, y_m, z_m, time_utc_ns)
