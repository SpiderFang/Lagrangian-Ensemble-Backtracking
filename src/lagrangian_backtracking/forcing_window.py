"""以 UTC 月份為邊界的 lazy forcing window 與 bounded LRU cache。

``ForcingWindowManager`` 將單一 flow domain 的 OCM native mesh、OCM 月份與可選的 NWW3
analysis 月份接到既有 ``CombinedMonthForcing``。它只在粒子 RK stage 實際要求某個
``YYYYMM`` 時載入該月，並以固定容量的最近最少使用（LRU）快取限制常駐記憶體。不同
material settling velocity、Stokes 開關、triangle hint 或 provider facade 都不會複製
OCM 月份陣列；velocity facade 只保存 manager 與小型參數，月份淘汰時底層
``CombinedMonthForcing`` 也一併釋放其引用。Smagorinsky 空間擴散另由
``ManagedSpatialDiffusionProvider`` 走 OCM-only route，仍共用同一個 OCM LRU，但不會
建立 combined forcing 或觸發 NWW lazy loader。

正式 constructor 只允許讀取已驗收的 ``<root>/<flow_domain_id>/{grid,months/YYYYMM}``
目錄，不會回讀 raw NetCDF 或用最近月份補值。整個 manager 預期由單一 process 使用，
不是 thread-safe；多 process worker 應各自建立一個 manager。cache stats 刻意暴露給
Phase 3B benchmark，以量測月份 I/O、命中率、淘汰與 resident ndarray bytes。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .diffusion import DiffusionCoefficients, DiffusionSample, SmagorinskySettings
from .forcing import CombinedMonthForcing, NWWAnalysisMonth, OCMNativeMonth
from .geometry import DomainProjection
from .mesh import NativeMesh
from .models import SampleQC, VelocitySample


class MissingForcingMonth(Exception):
    """表示整個 ``YYYYMM`` 月份目錄不存在，可轉成明確的時間範圍 QC。

    loader 若已看到目錄存在但缺少 array、schema 或 shape 損壞，應讓原始例外繼續上拋；
    只有「月份目錄本身不存在」才使用此例外或回傳 ``None``。如此可區分正常的時間支撐
    範圍外與輸入產品損毀。
    """


def _validate_month_id(month_id: str) -> str:
    """驗證可由 UTC stage 選出的六位數 YYYYMM 月份識別碼。"""

    if not isinstance(month_id, str) or len(month_id) != 6 or not month_id.isdigit():
        raise ValueError(f"month_id 必須是 YYYYMM：{month_id!r}")
    try:
        datetime.strptime(month_id, "%Y%m")
    except ValueError as exc:
        raise ValueError(f"month_id 不是有效月份：{month_id!r}") from exc
    return month_id


def _month_from_utc_ns(time_utc_ns: int) -> str:
    """由真正的 UTC 奈秒整數選出月份，不使用浮點秒數避免月界線誤差。"""

    if isinstance(time_utc_ns, bool) or not isinstance(time_utc_ns, (int, np.integer)):
        raise TypeError("time_utc_ns 必須是真正的 integer，不可為 bool")
    seconds, _ = divmod(int(time_utc_ns), 1_000_000_000)
    try:
        value = datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("time_utc_ns 無法轉成 UTC datetime") from exc
    return value.strftime("%Y%m")


def _finite_settling(value: float) -> float:
    """驗證 facade 的通用沉降速度有限；正式負值 gate 仍由 manifest/config 負責。"""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("settling_velocity_mps 必須是有限數值且不可為 bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("settling_velocity_mps 不可為 NaN 或 Infinity")
    return result


@dataclass(frozen=True, slots=True)
class ForcingCacheStats:
    """某一時間點的 immutable forcing cache 使用統計。

    ``resident_month_ids`` 依 LRU 從最舊到最新排列；``resident_ndarray_bytes`` 是已載入
    OCM/NWW 月份欄位的 ``nbytes`` 總和，不包含固定 mesh 的靜態陣列，也不是作業系統 RSS。
    因此它適合作為不同 cache 容量的相對 benchmark 指標，但不能取代 SERVER 實測記憶體。
    """

    ocm_load_count: int
    nww_load_count: int
    ocm_cache_hit_count: int
    ocm_cache_miss_count: int
    nww_cache_hit_count: int
    nww_cache_miss_count: int
    eviction_count: int
    resident_month_ids: tuple[str, ...]
    resident_ndarray_bytes: int

    @property
    def cache_hit_count(self) -> int:
        """回傳 OCM 與 NWW 合計 cache hits。"""

        return self.ocm_cache_hit_count + self.nww_cache_hit_count

    @property
    def cache_miss_count(self) -> int:
        """回傳 OCM 與 NWW 合計 cache misses。"""

        return self.ocm_cache_miss_count + self.nww_cache_miss_count


@dataclass(slots=True)
class _MonthEntry:
    """單一 resident month 的 OCM/NWW 物件與 material-specific lightweight cache。"""

    month_id: str
    ocm: OCMNativeMonth | None
    nww: NWWAnalysisMonth | None = None
    nww_attempted: bool = False
    combined: dict[tuple[float, bool], CombinedMonthForcing] | None = None

    def __post_init__(self) -> None:
        """以獨立 dictionary 保存同月不同 material facade 的合併取樣器。"""

        if self.combined is None:
            self.combined = {}


class ManagedForcingProvider:
    """只保存 manager 與 material 參數的輕量 provider facade。

    facade 不直接持有 ``OCMNativeMonth``、``NWWAnalysisMonth`` 或 ``CombinedMonthForcing``
    引用；每次 sample 都回到 manager 查詢目前 UTC 月份。這是 LRU 真正能釋放月份陣列的
    關鍵，也讓同一 manager 建立多個 material／Stokes 組合時不重複讀取 OCM。
    """

    def __init__(
        self, manager: ForcingWindowManager, settling_velocity_mps: float, include_stokes: bool
    ) -> None:
        """建立單一 material 的 facade；``include_stokes`` 僅控制 NWW lazy loading。"""

        self.manager = manager
        self.settling_velocity_mps = _finite_settling(settling_velocity_mps)
        if not isinstance(include_stokes, bool):
            raise TypeError("include_stokes 必須是 boolean")
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
        """取樣並完整轉交 triangle hint，供 ``HintTrackingVelocityProvider`` 使用。"""

        return self.manager.sample(
            x_m,
            y_m,
            z_m,
            time_utc_ns,
            settling_velocity_mps=self.settling_velocity_mps,
            include_stokes=self.include_stokes,
            triangle_hint=triangle_hint,
        )

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """提供既有四參數 velocity provider 介面；無 hint 時由 manager 正常取樣。"""

        return self.sample(x_m, y_m, z_m, time_utc_ns)


@dataclass(frozen=True, slots=True)
class ManagedSpatialDiffusionProvider:
    """由月份 manager 管理的輕量空間擴散 provider facade。

    facade 只保存 ``ForcingWindowManager`` 與 immutable 的
    ``SmagorinskySettings``，不保存任何 OCM 月份、網格 array 或合併 forcing 物件。每次
    ``sample`` 都把 UTC 時刻、絕對公尺座標與上一個成功 triangle hint 交回 manager，讓
    manager 依同一個 OCM-only LRU 選月並在需要時載入資料。這個設計使多個 member 能共用
    同一個 provider，而月份淘汰後不會因 facade 的隱藏參照阻止大型 array 釋放。

    這個 facade 的資料來源只限 OCM native current 與 native mesh 的
    ``sample_smagorinsky_diffusion``；它不會要求 NWW3，也不會建立
    ``CombinedMonthForcing``。但使用 Smagorinsky 案例的 velocity provider 仍可另外啟用
    有限水深 Stokes，因此「擴散 provider 不讀 NWW」不等於整個 velocity request 不需要
    NWW root。
    """

    manager: ForcingWindowManager
    settings: SmagorinskySettings

    def __post_init__(self) -> None:
        """在 facade 建立時鎖定物理設定，避免取樣途中被 caller 改寫。"""

        if not isinstance(self.settings, SmagorinskySettings):
            raise TypeError("settings 必須是 SmagorinskySettings")
        self.settings.validate()

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        triangle_hint: int | None = None,
    ) -> DiffusionSample:
        """取樣 OCM-only 空間擴散，並完整轉交 native triangle hint。

        輸入座標是公尺制運算座標，時間是 UTC 奈秒；輸出包含 ``Kx/Ky/Kz``、
        pseudo-time 散度漂移與品質旗標。hint 只改善 native mesh 定位 locality，不得改變
        公式或資料結果；缺少 OCM 月份時由 manager 回傳非零
        ``SampleQC.OUTSIDE_TIME_RANGE``，不以零係數表示有效靜水。
        """

        return self.manager.sample_smagorinsky_diffusion(
            x_m,
            y_m,
            z_m,
            time_utc_ns,
            settings=self.settings,
            triangle_hint=triangle_hint,
        )

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> DiffusionSample:
        """提供與既有 provider 一致的四參數呼叫形式；未提供 hint 時正常定位。"""

        return self.sample(x_m, y_m, z_m, time_utc_ns)


class ForcingWindowManager:
    """單一 flow domain 的月份 lazy loader、LRU cache 與 material provider factory。

    manager 的 OCM loader 必須回傳指定 ``month_id`` 的 ``OCMNativeMonth``，NWW loader
    可回傳 ``None`` 表示整個月份目錄不存在；兩者若遇到已存在但損壞的資料，例外不可
    被 manager 吃掉。因 manager 內部沒有 lock，請勿在同一 instance 上跨 thread 取樣；
    multiprocessing worker 也應各自建立獨立 instance。
    """

    def __init__(
        self,
        *,
        flow_domain_id: str,
        projection: DomainProjection,
        mesh: NativeMesh,
        ocm_loader: Callable[[str], OCMNativeMonth | None],
        nww_loader: Callable[[str], NWWAnalysisMonth | None] | None = None,
        max_resident_months: int = 2,
    ) -> None:
        """建立注入式 manager，測試可用小型 loader，production 則由 ``from_roots`` 建立。"""

        if not isinstance(flow_domain_id, str) or not flow_domain_id.strip():
            raise ValueError("flow_domain_id 不可為空白")
        if not isinstance(projection, DomainProjection):
            raise TypeError("projection 必須是 DomainProjection")
        if not isinstance(mesh, NativeMesh):
            raise TypeError("mesh 必須是 NativeMesh")
        if not callable(ocm_loader):
            raise TypeError("ocm_loader 必須是 callable")
        if nww_loader is not None and not callable(nww_loader):
            raise TypeError("nww_loader 必須是 callable 或 None")
        if isinstance(max_resident_months, bool) or not isinstance(max_resident_months, int):
            raise TypeError("max_resident_months 必須是正整數")
        if max_resident_months < 1:
            raise ValueError("max_resident_months 必須是正整數")
        self.flow_domain_id = flow_domain_id
        self.projection = projection
        self.mesh = mesh
        self._ocm_loader = ocm_loader
        self._nww_loader = nww_loader
        self.max_resident_months = max_resident_months
        self._months: OrderedDict[str, _MonthEntry] = OrderedDict()
        self._ocm_load_count = 0
        self._nww_load_count = 0
        self._ocm_cache_hit_count = 0
        self._ocm_cache_miss_count = 0
        self._nww_cache_hit_count = 0
        self._nww_cache_miss_count = 0
        self._eviction_count = 0

    @classmethod
    def from_roots(
        cls,
        *,
        flow_domain_id: str,
        projection: DomainProjection,
        ocm_root: str | Path,
        nww_root: str | Path | None,
        max_resident_months: int = 2,
    ) -> ForcingWindowManager:
        """從正式 root layout 建立 manager，且 OCM mesh 只在此處載入一次。

        OCM 讀取位置為 ``<ocm_root>/<flow_domain_id>/grid`` 與
        ``months/YYYYMM``；NWW 讀取位置為 ``<nww_root>/<flow_domain_id>/grid`` 與
        ``months/YYYYMM``。缺少整個月份目錄回傳 ``MissingForcingMonth``，但已存在目錄
        的必要檔案或 schema 錯誤由既有 reader 原樣上拋。
        """

        ocm_domain_root = Path(ocm_root) / flow_domain_id
        nww_domain_root = Path(nww_root) / flow_domain_id if nww_root is not None else None
        grid_dir = ocm_domain_root / "grid"
        if not grid_dir.is_dir():
            raise FileNotFoundError(f"缺少 OCM flow-domain grid directory：{grid_dir}")
        mesh = NativeMesh.from_directory(grid_dir, projection=projection)
        nww_grid_dir = nww_domain_root / "grid" if nww_domain_root is not None else None

        def load_ocm(month_id: str) -> OCMNativeMonth:
            """檢查月份目錄後使用既有 OCM reader，避免把缺檔誤轉成時間缺月。"""

            month_dir = ocm_domain_root / "months" / _validate_month_id(month_id)
            if not month_dir.is_dir():
                raise MissingForcingMonth(str(month_dir))
            return OCMNativeMonth.from_directory(month_dir, mesh=mesh)

        def load_nww(month_id: str) -> NWWAnalysisMonth | None:
            """僅在 Stokes facade 需要時開啟 NWW 月份；整個月不存在回傳 None。"""

            if nww_domain_root is None or nww_grid_dir is None:
                return None
            month_dir = nww_domain_root / "months" / _validate_month_id(month_id)
            if not month_dir.is_dir():
                return None
            if not nww_grid_dir.is_dir():
                raise FileNotFoundError(f"缺少 NWW flow-domain grid directory：{nww_grid_dir}")
            return NWWAnalysisMonth.from_directories(nww_grid_dir, month_dir)

        return cls(
            flow_domain_id=flow_domain_id,
            projection=projection,
            mesh=mesh,
            ocm_loader=load_ocm,
            nww_loader=load_nww,
            max_resident_months=max_resident_months,
        )

    def _evict_if_needed(self) -> None:
        """淘汰最舊月份；entry 一併丟棄 combined cache，避免 facade 保留大型陣列。"""

        while len(self._months) > self.max_resident_months:
            self._months.popitem(last=False)
            self._eviction_count += 1

    def _load_ocm(self, month_id: str) -> OCMNativeMonth | None:
        """載入一個 OCM 月份，只有整個月份不存在才轉為 None。"""

        try:
            result = self._ocm_loader(month_id)
        except MissingForcingMonth:
            return None
        if result is None:
            return None
        if not isinstance(result, OCMNativeMonth):
            raise TypeError("ocm_loader 必須回傳 OCMNativeMonth、None 或 MissingForcingMonth")
        if result.month_id != month_id:
            raise ValueError(f"OCM loader month_id 不一致：{result.month_id} != {month_id}")
        self._ocm_load_count += 1
        return result

    def _get_month(self, month_id: str) -> _MonthEntry:
        """取得或建立 resident month，並更新 OCM LRU hit/miss 統計。"""

        if month_id in self._months:
            self._ocm_cache_hit_count += 1
            entry = self._months.pop(month_id)
            self._months[month_id] = entry
            return entry
        self._ocm_cache_miss_count += 1
        entry = _MonthEntry(month_id=month_id, ocm=self._load_ocm(month_id))
        self._months[month_id] = entry
        self._evict_if_needed()
        return entry

    def _ensure_nww(self, entry: _MonthEntry) -> NWWAnalysisMonth | None:
        """lazy 載入同月 NWW；no-Stokes path 不會進入此函式。"""

        if entry.nww_attempted:
            self._nww_cache_hit_count += 1
            return entry.nww
        self._nww_cache_miss_count += 1
        if self._nww_loader is None:
            entry.nww = None
            entry.nww_attempted = True
            return None
        result = self._nww_loader(entry.month_id)
        if result is not None:
            if not isinstance(result, NWWAnalysisMonth):
                raise TypeError("nww_loader 必須回傳 NWWAnalysisMonth 或 None")
            if result.month_id != entry.month_id:
                raise ValueError(
                    f"NWW loader month_id 不一致：{result.month_id} != {entry.month_id}"
                )
            self._nww_load_count += 1
        entry.nww = result
        entry.nww_attempted = True
        return result

    def _combined(
        self, entry: _MonthEntry, *, settling_velocity_mps: float, include_stokes: bool
    ) -> CombinedMonthForcing:
        """建立或重用同月同參數的既有 CombinedMonthForcing。"""

        if entry.ocm is None:
            raise RuntimeError("缺少 OCM 月份時不應建立 CombinedMonthForcing")
        key = (settling_velocity_mps, include_stokes)
        assert entry.combined is not None
        cached = entry.combined.get(key)
        if cached is not None:
            return cached
        nww = entry.nww if include_stokes and entry.nww_attempted else None
        if include_stokes and not entry.nww_attempted:
            nww = self._ensure_nww(entry)
        if include_stokes and nww is None:
            raise RuntimeError("缺少 NWW 月份時不應建立 Stokes CombinedMonthForcing")
        value = CombinedMonthForcing(
            ocm=entry.ocm,
            nww=nww,
            projection=self.projection,
            settling_velocity_mps=settling_velocity_mps,
            include_stokes=include_stokes,
        )
        entry.combined[key] = value
        return value

    @staticmethod
    def _missing_ocm_sample(month_id: str) -> VelocitySample:
        """建立 OCM 月份不存在時的明確 sample，不以零速度代表物理靜止。"""

        return VelocitySample(
            0.0,
            0.0,
            0.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            SampleQC.OUTSIDE_TIME_RANGE,
            forcing_month_id=month_id,
        )

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        settling_velocity_mps: float,
        include_stokes: bool,
        triangle_hint: int | None = None,
    ) -> VelocitySample:
        """依每個 RK stage 的 UTC 月份取樣，嚴禁最近月、跨月外插與預設零速度。

        OCM 整月缺失回 ``OUTSIDE_TIME_RANGE``；若 OCM 存在而 Stokes 所需 NWW 整月缺失，
        回 ``WAVE_UNSUPPORTED``。已存在月份的 array、schema、shape 或 physics 錯誤不會
        被轉成 QC，而是直接讓例外上拋，方便 SERVER input gate 發現產品損壞。
        """

        month_id = _month_from_utc_ns(time_utc_ns)
        settling = _finite_settling(settling_velocity_mps)
        if not isinstance(include_stokes, bool):
            raise TypeError("include_stokes 必須是 boolean")
        entry = self._get_month(month_id)
        if entry.ocm is None:
            return self._missing_ocm_sample(month_id)
        if include_stokes and self._ensure_nww(entry) is None:
            current = entry.ocm.sample(x_m, y_m, z_m, time_utc_ns, triangle_hint=triangle_hint)
            if not current.valid:
                return current
            return replace(
                current,
                w_mps=current.w_mps + settling,
                qc=SampleQC.WAVE_UNSUPPORTED,
                diagnostics={**current.diagnostics, "nww_month_missing": 1},
            )
        combined = self._combined(
            entry, settling_velocity_mps=settling, include_stokes=include_stokes
        )
        return combined.sample(
            x_m,
            y_m,
            z_m,
            time_utc_ns,
            triangle_hint=triangle_hint,
        )

    def provider(self, settling_velocity_mps: float, include_stokes: bool) -> ManagedForcingProvider:
        """回傳可直接交給 reference request 或 hint wrapper 的輕量 provider。"""

        return ManagedForcingProvider(self, settling_velocity_mps, include_stokes)

    @staticmethod
    def _missing_smagorinsky_sample(
        month_id: str,
        settings: SmagorinskySettings,
        *,
        flow_domain_id: str,
    ) -> DiffusionSample:
        """建立 OCM 月份缺失時的明確擴散失敗樣本。

        失敗樣本保留設定中的垂向係數只作診斷輪廓，水平係數與散度不宣稱任何物理值；
        ``qc`` 明確設為 ``OUTSIDE_TIME_RANGE``，所以 engine 不會把它送進選步長或
        Brownian 位移。診斷固定保存方法、UTC 月份、flow domain 與缺失原因，方便把
        「時間支撐不存在」和「有效的零擴散」分開稽核。
        """

        return DiffusionSample(
            coefficients=DiffusionCoefficients(0.0, 0.0, settings.constant_kz_m2ps),
            diffusivity_divergence_mps=(np.nan, np.nan, np.nan),
            qc=SampleQC.OUTSIDE_TIME_RANGE,
            diagnostics={
                "method": "smagorinsky_native_mesh_p1_nodal",
                "forcing_month_id": month_id,
                "flow_domain_id": flow_domain_id,
                "missing": True,
                "ocm_month_missing": True,
                "missing_reason": "ocm_month_directory_unavailable",
            },
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
        """依 UTC 月份從 OCM native route 取樣 Smagorinsky 空間擴散。

        月份選擇與 velocity path 共用 ``_month_from_utc_ns``、``_get_month`` 及同一個
        OCM LRU，因此 request 建構仍是 lazy，跨月份時也會使用相同淘汰／統計契約。若
        OCM 月份存在，直接呼叫該月 ``OCMNativeMonth.sample_smagorinsky_diffusion``，
        將座標、時間、immutable settings 與 triangle hint 原樣傳遞；這條路徑刻意不呼叫
        ``_ensure_nww``、``nww_loader`` 或 ``_combined``，因為 Smagorinsky 擴散只需要
        OCM native current、mesh 梯度與垂向欄位。整個 OCM 月份不存在時回傳明確的
        ``OUTSIDE_TIME_RANGE``，不做最近月份、零係數或跨資料缺口外插。

        所有計算仍在公尺制座標進行，輸出的擴散係數單位為 m²/s、散度漂移單位為 m/s；
        OCM reader 已驗收但內容損壞時，原始例外繼續上拋，不把輸入產品錯誤偽裝成時間
        缺值。
        """

        if not isinstance(settings, SmagorinskySettings):
            raise TypeError("settings 必須是 SmagorinskySettings")
        settings.validate()
        month_id = _month_from_utc_ns(time_utc_ns)
        entry = self._get_month(month_id)
        if entry.ocm is None:
            return self._missing_smagorinsky_sample(
                month_id,
                settings,
                flow_domain_id=self.flow_domain_id,
            )
        return entry.ocm.sample_smagorinsky_diffusion(
            x_m,
            y_m,
            z_m,
            time_utc_ns,
            settings,
            triangle_hint=triangle_hint,
        )

    def smagorinsky_provider(
        self, settings: SmagorinskySettings
    ) -> ManagedSpatialDiffusionProvider:
        """建立只保存 manager／settings 的 Smagorinsky facade，不預載任何月份。

        ``settings`` 會在 facade 建立邊界再次驗證並以 frozen dataclass 保存；此 factory
        本身不碰 OCM loader、NWW loader、mesh month array 或時間取樣，因此可安全用於
        runtime request 的 lazy construction。真正取樣時才由 manager 選取 UTC 月份。
        """

        return ManagedSpatialDiffusionProvider(self, settings)

    def preload(self, month_ids: Iterable[str], *, include_stokes: bool = False) -> None:
        """按指定月份預熱同一 LRU cache；不改變缺月與損壞資料的錯誤契約。"""

        if not isinstance(include_stokes, bool):
            raise TypeError("include_stokes 必須是 boolean")
        for raw_month_id in month_ids:
            month_id = _validate_month_id(raw_month_id)
            entry = self._get_month(month_id)
            if include_stokes and entry.ocm is not None:
                self._ensure_nww(entry)

    @staticmethod
    def _array_bytes(value: object) -> int:
        """回傳 ndarray/memmap 的 nbytes；非陣列欄位不計入 resident month bytes。"""

        return int(getattr(value, "nbytes", 0))

    def _resident_bytes(self) -> int:
        """估計 resident OCM/NWW month arrays 的記憶體映射大小，且每個 object 只計一次。"""

        total = 0
        seen: set[int] = set()
        for entry in self._months.values():
            objects: list[object] = []
            if entry.ocm is not None:
                objects.extend(
                    [
                        entry.ocm.time_utc_ns,
                        entry.ocm.hvel,
                        entry.ocm.vertical_velocity,
                        entry.ocm.zcor,
                        entry.ocm.elev,
                        entry.ocm.wetdry_elem,
                        entry.ocm.diffusivity,
                    ]
                )
            if entry.nww is not None:
                objects.extend(
                    [
                        entry.nww.lon,
                        entry.nww.lat,
                        entry.nww.time_utc_ns,
                        entry.nww.significant_wave_height,
                        entry.nww.peak_frequency,
                        entry.nww.peak_direction_raw_deg,
                        entry.nww.valid_mask_wave,
                        entry.nww.qc_flags,
                    ]
                )
            for value in objects:
                identity = id(value)
                if identity not in seen:
                    seen.add(identity)
                    total += self._array_bytes(value)
        return total

    @property
    def cache_stats(self) -> ForcingCacheStats:
        """以 immutable snapshot 回傳 cache 命中、載入、淘汰與 resident bytes。"""

        return ForcingCacheStats(
            ocm_load_count=self._ocm_load_count,
            nww_load_count=self._nww_load_count,
            ocm_cache_hit_count=self._ocm_cache_hit_count,
            ocm_cache_miss_count=self._ocm_cache_miss_count,
            nww_cache_hit_count=self._nww_cache_hit_count,
            nww_cache_miss_count=self._nww_cache_miss_count,
            eviction_count=self._eviction_count,
            resident_month_ids=tuple(self._months),
            resident_ndarray_bytes=self._resident_bytes(),
        )

    def get_cache_stats(self) -> ForcingCacheStats:
        """提供 method 形式的 cache stats alias，方便 benchmark adapter 注入。"""

        return self.cache_stats

    @property
    def stats(self) -> ForcingCacheStats:
        """提供簡短的 immutable stats property alias。"""

        return self.cache_stats


__all__ = [
    "ForcingCacheStats",
    "ForcingWindowManager",
    "ManagedForcingProvider",
    "ManagedSpatialDiffusionProvider",
    "MissingForcingMonth",
]
