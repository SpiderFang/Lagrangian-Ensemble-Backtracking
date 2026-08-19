"""OCM native 4D、NWW3 analysis 3D 與 Stokes/浮沉的唯讀 forcing adapters。

每個 adapter 只 memory-map 一個月份。跨月切換由 ``MonthlyCombinedForcing`` 依 stage
UTC 選擇，禁止最近月份外插。OCM 依「node 垂向 → triangle barycentric → 時間線性」
順序取樣；三個支撐 node 任一無法包夾 z、face 乾或時次缺口時，回傳明確 QC bit。
NWW3 只在四角及前後時次的 ``valid_mask_wave`` 全部有效時做保守雙線性／時間插值。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .accelerated import interpolate_ocm_support_numba
from .geometry import DomainProjection
from .mesh import MeshLocation, NativeMesh
from .models import SampleQC, VelocitySample
from .stokes import finite_depth_stokes


def _time_bracket(
    times_ns: np.ndarray, target_ns: int, *, maximum_gap_ns: int
) -> tuple[int, int, float, SampleQC]:
    """回傳前後 index 與線性權重；域外、重複或過大 gap 分開標記。"""

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
    """NWW3 bulk 波況；raw direction 尚未轉成傳播向量。"""

    significant_wave_height_m: float
    peak_frequency_hz: float
    peak_direction_raw_deg: float
    qc_flags: int
    qc: SampleQC = SampleQC.OK

    @property
    def valid(self) -> bool:
        """只有共同有效遮罩與物理值都通過才可計算 Stokes。"""

        return self.qc == SampleQC.OK


class OCMNativeMonth:
    """一個 OCM schema 3 月份的 4D 原生網格取樣器。"""

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
        """保留 memory-map 並驗證跨陣列前導維度，不複製完整月份。"""

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
        """由月份目錄以禁止 pickle 的 read-only mmap 建立取樣器。"""

        root = Path(month_dir)

        def load(name: str) -> np.ndarray:
            """以唯讀 mmap 載入單一 OCM 月陣列，不複製大型 time×node×layer 資料。"""

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
        """在一個 node 柱中包夾 z，回傳 ``[u,v,w,kz]`` 與 bracket span。

        不假設 layer index 排序；任一必要欄位缺值時該 layer 不可用。單側最近層不作
        外插，粒子在海床以下或海面以上會由 caller 收到 ``None``。
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

    def _spatial_at_time(
        self, location: MeshLocation, *, time_index: int, z_m: float
    ) -> tuple[tuple[np.ndarray, float, float] | None, SampleQC]:
        """對三個 node 完成垂向與水平內插，並分開回報乾點或垂向無支撐。"""

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

    def sample(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """取樣 OCM ``u/v/w/Kz/eta/bed``，失敗以 QC 回傳而非丟失原因。"""

        location = self.mesh.locate(x_m, y_m)
        if location is None:
            return VelocitySample(
                0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, SampleQC.OUTSIDE_HORIZONTAL_DOMAIN
            )
        before, after, alpha, time_qc = _time_bracket(
            self.time_utc_ns, time_utc_ns, maximum_gap_ns=self.maximum_time_gap_ns
        )
        if time_qc != SampleQC.OK:
            return VelocitySample(0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, time_qc)
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
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                np.nan,
                np.nan,
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
        """驗證 ``(time,lat,lon)`` 契約並保留 read-only memory maps。"""

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
        """由 analysis grid 與月份目錄建立 mmap 取樣器。"""

        grid = Path(grid_dir)
        month = Path(month_dir)

        def load(root: Path, name: str) -> np.ndarray:
            """由 grid 或 month root 唯讀 mmap NWW 陣列，禁止 pickle 與缺檔猜測。"""

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
        """以四角全有效政策做空間／時間插值，方向用 cos/sin 避免 0/360 斷點。"""

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
    """同一月份 OCM、NWW、投影與粒子浮沉的速度 provider。"""

    def __init__(
        self,
        *,
        ocm: OCMNativeMonth,
        nww: NWWAnalysisMonth | None,
        projection: DomainProjection,
        settling_velocity_mps: float,
        include_stokes: bool,
    ) -> None:
        """no-Stokes case 可不提供 NWW，含 Stokes case 則強制要求。"""

        if include_stokes and nww is None:
            raise ValueError("include_stokes=True 時必須提供 NWW month")
        self.ocm = ocm
        self.nww = nww
        self.projection = projection
        self.settling_velocity_mps = float(settling_velocity_mps)
        self.include_stokes = include_stokes

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """合成 ``OCM current + finite-depth Stokes + settling`` 的 forward-time 速度。"""

        current = self.ocm.sample(x_m, y_m, z_m, time_utc_ns)
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
        )


class MonthlyCombinedForcing:
    """依每個 RK stage UTC 選月份的唯讀 provider，不在缺月時外插。"""

    def __init__(self, months: Mapping[str, CombinedMonthForcing]) -> None:
        """月份 key 必須是唯一 YYYYMM；空 mapping 無法積分。"""

        if not months or any(len(key) != 6 or not key.isdigit() for key in months):
            raise ValueError("months 必須是非空 YYYYMM -> forcing mapping")
        self.months = dict(months)

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """以 UTC calendar month 選 adapter；缺月明確回傳 OUTSIDE_TIME_RANGE。"""

        month_id = datetime.fromtimestamp(time_utc_ns / 1_000_000_000, tz=UTC).strftime("%Y%m")
        provider = self.months.get(month_id)
        if provider is None:
            return VelocitySample(0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, SampleQC.OUTSIDE_TIME_RANGE)
        return provider(x_m, y_m, z_m, time_utc_ns)
