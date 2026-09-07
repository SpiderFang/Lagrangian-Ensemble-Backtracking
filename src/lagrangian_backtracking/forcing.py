"""讀取海流、波浪、表面漂移與浮沉資料，並在指定時空位置提供速度。

海流資料使用海洋模式（OCM）的原始三維網格及時間軸；波浪資料使用 NWW3 分析資料。
每個讀取器一次只以唯讀方式開啟一個月份，避免把完整兩年資料載入記憶體。跨月時由
``MonthlyCombinedForcing`` 依世界協調時間（UTC）選取正確月份，絕不借用最近月份資料。
海流依序在節點垂向、三角形平面與時間上內插；若三個支撐節點任一處無法夾住指定深度、
網格面乾涸或時間間隔過大，會回傳明確的品質檢查旗標。波浪只有在周圍四個格點及前後
兩個時刻的資料都有效時，才做空間與時間內插。
Smagorinsky reference 則只使用 OCM native current，在公尺制 triangle shape function
上建立先套 floor/cap 的 P1 nodal Kh 與 grad(K)；它不讀取 Stokes、settling 或空變 Kz，
也尚未代表正式 runtime baseline。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
from .models import SampleQC, VelocityComponents, VelocitySample
from .stokes import finite_depth_stokes


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


@dataclass(frozen=True, slots=True)
class WaveSample:
    """一筆 NWW3 波浪摘要資料；原始波向尚未轉為傳播方向向量。"""

    significant_wave_height_m: float
    peak_frequency_hz: float
    peak_direction_raw_deg: float
    qc_flags: int
    qc: SampleQC = SampleQC.OK

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
        """在單一節點的水柱中，以指定深度上下兩層內插速度與垂向擴散。

        回傳的四個值依序是東向、北向、垂向速度與垂向擴散係數，另附兩層間的深度差。
        不假設資料層號已由淺到深排序；任一必要數值缺漏時該層不可用。只用一側最近層的
        外插會製造不可靠速度，因此粒子在海床以下或海面以上時回傳 ``None``。
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
        below = np.flatnonzero(usable & (physical_z <= z_m))
        above = np.flatnonzero(usable & (physical_z >= z_m))
        if below.size == 0 or above.size == 0:
            return None
        lower = int(below[np.argmax(physical_z[below])])
        upper = int(above[np.argmin(physical_z[above])])
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
        提供。仍沿用與一般速度取樣相同的上下夾層政策，故海面以上、海床以下、zcor
        缺值或 hvel 缺值都回傳 ``None``，不做單側最近層外插。回傳的兩個水平分量單位
        為 m/s，第二項是實際夾層深度差（m），只供資料品質與垂向支撐診斷。
        """

        if not np.isfinite(z_m):
            return None
        physical_z = np.asarray(self.zcor[time_index, node_index], dtype=np.float64)
        horizontal_velocity = np.asarray(
            self.hvel[time_index, node_index, :, :2], dtype=np.float64
        )
        usable = np.isfinite(physical_z) & np.all(np.isfinite(horizontal_velocity), axis=1)
        below = np.flatnonzero(usable & (physical_z <= z_m))
        above = np.flatnonzero(usable & (physical_z >= z_m))
        if below.size == 0 or above.size == 0:
            return None
        lower = int(below[np.argmax(physical_z[below])])
        upper = int(above[np.argmin(physical_z[above])])
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

        first, first_qc, first_valid_count, first_excluded_count = self._smagorinsky_time_slice(
            location,
            time_index=before,
            z_m=z_m,
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
                    z_m=z_m,
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
        if not np.isfinite(eta) or not np.isfinite(bed) or bed > eta + 1.0e-6:
            return None
        return eta, bed

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
        資料不足，而不是只得到沒有原因的空值。``triangle_hint`` 只作 native mesh
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
        if self.use_numba_kernel:
            face = location.source_face_local_index
            wet_values = np.asarray(self.wetdry_elem[[before, after], face], dtype=np.float64)
            if not np.all(np.isfinite(wet_values)) or not np.allclose(wet_values, self.wet_value, atol=0.1):
                first = second = None
                first_qc = second_qc = SampleQC.DRY_FACE
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
                    z_m,
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
            first, first_qc = self._spatial_at_time(location, time_index=before, z_m=z_m)
            if after == before:
                second, second_qc = first, first_qc
            else:
                second, second_qc = self._spatial_at_time(location, time_index=after, z_m=z_m)
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
        eta = first[1] + alpha * (second[1] - first[1])
        vertical_scale = min(first[2], second[2])
        nodes = np.asarray(location.node_indices, dtype=np.int64)
        depth = float(
            np.asarray(location.barycentric_weights, dtype=np.float64) @ self.mesh.source_depth_m[nodes]
        )
        bed_z = -depth
        if z_m < bed_z - 1e-6 or z_m > eta + 1e-6:
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

    def sample(self, lon: float, lat: float, time_utc_ns: int) -> WaveSample:
        """僅在四個周圍格點都有效時做空間與時間內插；波向以正弦、餘弦避免 0/360 度斷點。"""

        if lon < self.lon[0] or lon > self.lon[-1] or lat < self.lat[0] or lat > self.lat[-1]:
            return WaveSample(np.nan, np.nan, np.nan, 0, SampleQC.OUTSIDE_HORIZONTAL_DOMAIN)
        x_after = min(max(int(np.searchsorted(self.lon, lon, side="right")), 1), self.lon.size - 1)
        y_after = min(max(int(np.searchsorted(self.lat, lat, side="right")), 1), self.lat.size - 1)
        x0, x1 = x_after - 1, x_after
        y0, y1 = y_after - 1, y_after
        wx = float((lon - self.lon[x0]) / (self.lon[x1] - self.lon[x0]))
        wy = float((lat - self.lat[y0]) / (self.lat[y1] - self.lat[y0]))
        before, after, alpha, time_qc = _time_bracket(
            self.time_utc_ns, time_utc_ns, maximum_gap_ns=self.maximum_time_gap_ns
        )
        if time_qc != SampleQC.OK:
            return WaveSample(np.nan, np.nan, np.nan, 0, time_qc)
        corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
        spatial_weights = np.array([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
        time_indices = [before] if before == after else [before, after]
        time_weights = [1.0] if before == after else [1.0 - alpha, alpha]
        hs = 0.0
        fp = 0.0
        direction_x = 0.0
        direction_y = 0.0
        qc_union = 0
        for time_index, time_weight in zip(time_indices, time_weights, strict=True):
            mask = np.array([self.valid_mask_wave[time_index, y, x] for y, x in corners], dtype=bool)
            qc_union |= int(np.bitwise_or.reduce([self.qc_flags[time_index, y, x] for y, x in corners]))
            if not np.all(mask):
                return WaveSample(np.nan, np.nan, np.nan, qc_union, SampleQC.WAVE_UNSUPPORTED)
            hs_values = np.array([self.significant_wave_height[time_index, y, x] for y, x in corners])
            fp_values = np.array([self.peak_frequency[time_index, y, x] for y, x in corners])
            directions = np.deg2rad(
                np.array([self.peak_direction_raw_deg[time_index, y, x] for y, x in corners])
            )
            if not (
                np.all(np.isfinite(hs_values))
                and np.all(np.isfinite(fp_values))
                and np.all(np.isfinite(directions))
            ):
                return WaveSample(np.nan, np.nan, np.nan, qc_union, SampleQC.WAVE_UNSUPPORTED)
            hs += time_weight * float(spatial_weights @ hs_values)
            fp += time_weight * float(spatial_weights @ fp_values)
            direction_x += time_weight * float(spatial_weights @ np.sin(directions))
            direction_y += time_weight * float(spatial_weights @ np.cos(directions))
        if hs < 0 or fp <= 0 or abs(direction_x) + abs(direction_y) <= 1e-15:
            return WaveSample(hs, fp, np.nan, qc_union, SampleQC.INVALID_PHYSICS)
        raw_direction = float(np.degrees(np.arctan2(direction_x, direction_y)) % 360.0)
        return WaveSample(hs, fp, raw_direction, qc_union)


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
