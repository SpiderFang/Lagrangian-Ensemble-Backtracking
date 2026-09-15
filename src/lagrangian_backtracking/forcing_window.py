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
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from .accelerated import validate_physics_kernel_backend
from .diffusion import DiffusionCoefficients, DiffusionSample, SmagorinskySettings
from .forcing import (
    CombinedMonthForcing,
    NWWAnalysisMonth,
    OCMNativeMonth,
    _query_z_within_geometric_bounds,
)
from .geometry import DomainProjection
from .integrators import StepStartSampleReuseProvider
from .mesh import NativeMesh
from .models import SURFACE_BOUNDARY_TOLERANCE_M, SampleQC, VelocitySample

_MONTH_METADATA_CACHE_SIZE = 512


@dataclass(frozen=True, slots=True)
class _MonthMetadata:
    """保存一個已驗證 ``YYYYMM`` 的小型曆法資訊。

    月份解析只影響目錄選擇與月界 halo 判斷，並不代表該月份資料一定存在；因此這裡
    只快取 UTC 曆法值，不快取 OCM/NWW 陣列，也不把缺月份轉成任何數值。metadata
    快取由 ``lru_cache`` 限制為固定 512 個月份鍵，屬於單一 Python process 的狀態；
    不同 multiprocessing worker 各自擁有一份小型快取，且 manager 本身仍維持原本
    的非 thread-safe 契約。``month_start_utc`` 只保存解析後的 aware datetime；奈秒
    轉換仍在原本的 ``_month_start_ns`` 呼叫點進行，以保留 datetime 上下界與奈秒邊界
    的既有語意。
    """

    month_id: str
    year: int
    month: int
    month_start_utc: datetime


@lru_cache(maxsize=_MONTH_METADATA_CACHE_SIZE)
def _cached_month_metadata(month_id: str) -> _MonthMetadata:
    """以有限容量快取有效月份的 ``strptime`` 結果與 UTC 月初 datetime。

    呼叫端必須先完成 ``_validate_month_id`` 的型別、長度與數字檢查；因此不可雜湊的
    非字串輸入仍會由既有驗證路徑回傳 ``ValueError``，不會變成 ``lru_cache`` 的
    ``TypeError``。``lru_cache`` 不保存例外，非法月份也不會佔用快取；每個不同有效
    ``YYYYMM`` 最多解析一次，重複的 RK stage 查詢直接重用同一份 immutable metadata。
    """

    parsed = datetime.strptime(month_id, "%Y%m")
    month_start = parsed.replace(tzinfo=UTC)
    return _MonthMetadata(
        month_id=month_id,
        year=parsed.year,
        month=parsed.month,
        month_start_utc=month_start,
    )


class MissingForcingMonth(Exception):
    """表示整個 ``YYYYMM`` 月份目錄不存在，可轉成明確的時間範圍 QC。

    loader 若已看到目錄存在但缺少 array、schema 或 shape 損壞，應讓原始例外繼續上拋；
    只有「月份目錄本身不存在」才使用此例外或回傳 ``None``。如此可區分正常的時間支撐
    範圍外與輸入產品損毀。
    """


def _validated_month_metadata(month_id: str) -> _MonthMetadata:
    """驗證月份並只取一次其快取 metadata，避免 caller 先驗證再重複查快取。

    ``month_id`` 的外部契約仍是六位數字字串；型別與格式錯誤在進入
    ``lru_cache`` 前處理，維持原本的 ``ValueError`` 訊息。有效月份由
    ``_cached_month_metadata`` 解析並保存，讓同一熱路徑同時需要「相鄰月份識別碼」與
    「下一月月初」時共用同一個小型 immutable 物件，而不必在單次呼叫中重複驗證或解析。
    """

    if not isinstance(month_id, str) or len(month_id) != 6 or not month_id.isdigit():
        raise ValueError(f"month_id 必須是 YYYYMM：{month_id!r}")
    try:
        # 先保留既有的外部錯誤訊息與型別行為，再把有效月份交給有限容量快取；
        # ``lru_cache`` 不會快取例外，所以 202413 等非法月份不會污染快取內容。
        return _cached_month_metadata(month_id)
    except ValueError as exc:
        raise ValueError(f"month_id 不是有效月份：{month_id!r}") from exc


def _validate_month_id(month_id: str) -> str:
    """驗證可由 UTC stage 選出的六位數 YYYYMM 月份識別碼。"""

    _validated_month_metadata(month_id)
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


def _adjacent_month_id(month_id: str, offset: int) -> str:
    """以 UTC 曆月計算相鄰月份，避免跨年時用整數加減產生非法 ``YYYY00``。"""

    metadata = _validated_month_metadata(month_id)
    serial = metadata.year * 12 + metadata.month - 1 + int(offset)
    year, month_zero_based = divmod(serial, 12)
    if year < 1 or year > 9999:
        raise ValueError(f"相鄰月份超出 datetime 支援範圍：{month_id!r} offset={offset}")
    return f"{year:04d}{month_zero_based + 1:02d}"


def _month_start_ns(month_id: str) -> int:
    """回傳 UTC 月初奈秒，供判斷月份末列與相鄰 leading halo 的距離。"""

    metadata = _validated_month_metadata(month_id)
    # timestamp 仍在此處計算，讓月初奈秒的例外與原本 ``_month_start_ns`` 完全相同；
    # 快取只重用解析後的 aware datetime，不會改變 datetime 上下界的錯誤契約。
    return int(metadata.month_start_utc.timestamp()) * 1_000_000_000


def _cross_month_support_z(
    z_m: float, geometric_bounds: tuple[float, float] | None
) -> float:
    """依跨月 query-time 海面建立兩端共用的固定垂向支援值。

    endpoint 的 ``zcor`` 仍由各自資料列決定；這裡只處理移動海面容許帶內的數值夾回，
    使跨月結果與單一連續 provider 先內插 ``eta``、再交給兩端 layer support 的順序
    一致。超過海面容許帶或幾何缺值時保留原始 z，後續 endpoint／共同 geometry gate
    會回傳原本的 ``VERTICAL_UNSUPPORTED``，不以夾回掩蓋真實越界。
    """

    if geometric_bounds is None:
        return z_m
    eta_m = geometric_bounds[0]
    try:
        eta_value = float(eta_m)
        z_value = float(z_m)
    except (TypeError, ValueError, OverflowError):
        return z_m
    if not np.isfinite(eta_value) or not np.isfinite(z_value):
        return z_m
    if z_value <= eta_value:
        return z_value
    if z_value <= eta_value + SURFACE_BOUNDARY_TOLERANCE_M:
        return eta_value
    return z_value


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


@dataclass(frozen=True, slots=True)
class _GlobalTimeBracket:
    """跨月份時間軸的兩端來源與內插比例。

    ``before``／``after`` 以 ``(_MonthEntry, local_index, utc_ns)`` 保存實際產品來源，
    不是依檔名猜測的月份。相同 UTC 若由兩個月份 halo 同時提供，建立 bracket 時已按
    月份順序採後者，遵守 canonical prefer-last；只有真正沒有兩側資料，或兩端間距
    超過任一端允許的 maximum gap，才回傳非零 ``qc``。
    """

    before: tuple[_MonthEntry, int, int] | None
    after: tuple[_MonthEntry, int, int] | None
    alpha: float
    qc: SampleQC


class ManagedForcingProvider(StepStartSampleReuseProvider):
    """只保存月份快取管理器與材質參數的輕量速度取樣介面。

    這個介面不直接持有 ``OCMNativeMonth``、``NWWAnalysisMonth`` 或 ``CombinedMonthForcing``
    引用；每次取樣都回到月份快取管理器查詢目前 UTC 月份。這是最近最少使用（LRU）快取
    能真正釋放月份陣列的關鍵，也讓同一管理器建立多個材質／Stokes 組合時不重複讀取 OCM。

    這個介面明示允許步首樣本供 RK4 的 k1 重用：OCM／NWW3 陣列以唯讀方式開啟，
    相同座標、深度、UTC 與材質參數的樣本結果不依呼叫次數改變；月份快取管理器的
    最近最少使用快取與三角形搜尋提示只影響快取／搜尋狀態，不改變速度、QC 或邊界資料。
    若未來速度取樣器引入會改變物理結果的可變狀態，必須移除此標記，不能只為了減少查詢
    而沿用它。
    """

    def __init__(
        self, manager: ForcingWindowManager, settling_velocity_mps: float, include_stokes: bool
    ) -> None:
        """建立單一材質的速度取樣介面；``include_stokes`` 僅控制 NWW 延後載入。"""

        self.manager = manager
        self.settling_velocity_mps = _finite_settling(settling_velocity_mps)
        if not isinstance(include_stokes, bool):
            raise TypeError("include_stokes 必須是 boolean")
        self.include_stokes = include_stokes

    @property
    def step_start_sample_reuse_safe(self) -> bool:
        """回傳唯讀月份快取管理器介面的明示 k1 重用承諾。"""

        return True

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        triangle_hint: int | None = None,
    ) -> VelocitySample:
        """取樣並完整轉交三角形搜尋提示，供 ``HintTrackingVelocityProvider`` 使用。"""

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
        """提供既有四參數速度取樣器介面；無搜尋提示時由月份快取管理器正常取樣。"""

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
        physics_kernel_backend: str = "numpy_v1",
    ) -> None:
        """建立注入式 manager，測試可用小型 loader，production 則由 ``from_roots`` 建立。

        ``physics_kernel_backend`` 固定此 manager 所有材質 provider 共用的數值 primitive
        版本；這樣同月 combined cache 不必在每次 RK stage 依 request 改寫，也不會讓同一
        run 中有些粒子走不同 Stokes 方程實作。舊 caller 省略時使用 NumPy reference。
        """

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
        self.physics_kernel_backend = validate_physics_kernel_backend(physics_kernel_backend)
        self._months: OrderedDict[str, _MonthEntry] = OrderedDict()
        self._ocm_load_count = 0
        self._nww_load_count = 0
        self._ocm_cache_hit_count = 0
        self._ocm_cache_miss_count = 0
        self._nww_cache_hit_count = 0
        self._nww_cache_miss_count = 0
        self._eviction_count = 0
        # 跨月 bracket 在容量 1 的 cache 也必須同時保留兩個 endpoint。此集合只在
        # 單次 sample 的明確 scope 內使用；scope 結束後立即恢復設定容量，故長期
        # resident stats 仍反映真正可回收的月份，而不是靠隱藏 facade 永久留住陣列。
        self._pinned_month_ids: set[str] = set()
        # NWW 的 lon／lat 是跨月份共用的規則分析格網；保留第一份不可變簽章，讓
        # 注入式 loader 也遵守正式產品的 grid identity 契約。簽章只保存 shape、dtype
        # 與 bytes，不持有已淘汰月份的大型波浪欄位，也不改變 resident bytes 統計。
        self._nww_grid_signature: (
            tuple[tuple[int, ...], str, bytes, tuple[int, ...], str, bytes] | None
        ) = None

    @classmethod
    def from_roots(
        cls,
        *,
        flow_domain_id: str,
        projection: DomainProjection,
        ocm_root: str | Path,
        nww_root: str | Path | None,
        max_resident_months: int = 2,
        use_numba_kernel: bool = False,
        physics_kernel_backend: str = "numpy_v1",
    ) -> ForcingWindowManager:
        """從正式 root layout 建立 manager，且 OCM mesh 只在此處載入一次。

        OCM 讀取位置為 ``<ocm_root>/<flow_domain_id>/grid`` 與
        ``months/YYYYMM``；NWW 讀取位置為 ``<nww_root>/<flow_domain_id>/grid`` 與
        ``months/YYYYMM``。缺少整個月份目錄回傳 ``MissingForcingMonth``，但已存在目錄
        的必要檔案或 schema 錯誤由既有 reader 原樣上拋。``use_numba_kernel`` 只傳給
        ``OCMNativeMonth.from_directory`` 的 OCM 內層插值 kernel；
        ``physics_kernel_backend`` 另決定 Stokes、步長、RK4 最後純量更新及 Brownian 位移
        的數值 primitive。月份選擇、NWW 載入、缺值／遮罩、QC 與邊界判定仍由 Python 控制層
        負責。兩個參數都使用舊 caller 相容預設，不會因省略而切換到 Numba。
        """

        if type(use_numba_kernel) is not bool:
            raise TypeError("use_numba_kernel 必須是 bool")
        backend = validate_physics_kernel_backend(physics_kernel_backend)

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
            return OCMNativeMonth.from_directory(
                month_dir,
                mesh=mesh,
                use_numba_kernel=use_numba_kernel,
            )

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
            physics_kernel_backend=backend,
        )

    def _evict_if_needed(self) -> None:
        """淘汰最舊且未被當前 endpoint 使用的月份。

        平常這是普通 LRU；跨月取樣會暫時 pin before／after entry。若容量設定為 1，
        兩個 endpoint 可能使 cache 短暫超過 1，但不會在第二端載入時先丟掉第一端的
        memory-map。pin scope 結束才執行這裡的淘汰，``cache_stats`` 因而只把實際仍
        被 manager 持有的陣列列入 resident bytes。
        """

        while len(self._months) > self.max_resident_months:
            evictable = next(
                (month_id for month_id in self._months if month_id not in self._pinned_month_ids),
                None,
            )
            if evictable is None:
                return
            del self._months[evictable]
            self._eviction_count += 1

    @contextmanager
    def _pin_months(self, month_ids: Iterable[str]):
        """在一個取樣 scope 內保留指定月份，離開時恢復 LRU 容量。

        pin 只保護 manager cache 中的 entry，不改變 loader、資料陣列或物理結果。呼叫端
        可以在 scope 內先把日後需要的相鄰月份加入 ``_pinned_month_ids``，再呼叫
        ``_get_month``；若 loader 讀取中途拋出例外，``finally`` 仍會解除 pin 並清理
        可淘汰項目，避免下一次 sample 繼承半套狀態。
        """

        previous = set(self._pinned_month_ids)
        self._pinned_month_ids.update(month_ids)
        try:
            yield
        finally:
            self._pinned_month_ids = previous
            self._evict_if_needed()

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
        if not self._meshes_match(result.mesh, self.mesh):
            raise ValueError(f"OCM loader mesh identity 不一致：{month_id}")
        self._ocm_load_count += 1
        return result

    @staticmethod
    def _meshes_match(first: NativeMesh, second: NativeMesh) -> bool:
        """以靜態 node／face／深度資料核對月份 mesh，拒絕跨月換網格取樣。

        注入式測試 loader 可能為每個月份建立不同 Python object，但正式 physics 仍要求
        拓撲、投影後座標、native 水深與 global face ID 完全相同；因此不以 object identity
        判定，而比較會影響 locator、垂向海床與三角形梯度的 immutable 欄位。大型 forcing
        array 不在這裡複製，``np.array_equal`` 只掃描固定 mesh。
        """

        if not isinstance(first, NativeMesh) or not isinstance(second, NativeMesh):
            return False
        fields = (
            "node_lon",
            "node_lat",
            "node_xy",
            "source_depth_m",
            "source_node_bottom_index",
            "face_nodes_local",
            "face_node_count",
            "source_face_global_index",
            "triangle_nodes",
            "triangle_face_local",
        )
        return all(np.array_equal(getattr(first, field), getattr(second, field)) for field in fields)

    @staticmethod
    def _nww_grid_signature_for(
        nww: NWWAnalysisMonth,
    ) -> tuple[tuple[int, ...], str, bytes, tuple[int, ...], str, bytes]:
        """建立 NWW 規則格網的 immutable identity 簽章。

        NWW 月份資料只應變更時間列與波浪欄位，``lon``／``lat`` 必須保持同一分析格網。
        將 shape、dtype 與連續 bytes 固定下來，可在某月份被 LRU 淘汰後仍驗證後續
        月份；簽章不保存波浪資料，因此不會把大型 memory-map 變成永久 resident。
        """

        lon = np.ascontiguousarray(np.asarray(nww.lon))
        lat = np.ascontiguousarray(np.asarray(nww.lat))
        return (
            tuple(lon.shape),
            lon.dtype.str,
            lon.tobytes(order="C"),
            tuple(lat.shape),
            lat.dtype.str,
            lat.tobytes(order="C"),
        )

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

    def _pin_and_get_month(self, month_id: str) -> _MonthEntry:
        """先 pin 再取得月份，確保它不會在同一 bracket 內被第二端淘汰。"""

        normalized = _validate_month_id(month_id)
        self._pinned_month_ids.add(normalized)
        return self._get_month(normalized)

    def _time_array_for_entry(
        self,
        entry: _MonthEntry,
        *,
        source: str,
    ) -> np.ndarray | None:
        """取得 entry 的 OCM 或 NWW UTC 軸；NWW 只在 Stokes path lazy 開啟。"""

        if source == "ocm":
            return None if entry.ocm is None else entry.ocm.time_utc_ns
        if source == "nww":
            nww = self._ensure_nww(entry)
            return None if nww is None else nww.time_utc_ns
        raise ValueError(f"未知 forcing source：{source}")

    def _prepare_time_entries(
        self,
        month_id: str,
        target_ns: int,
        *,
        source: str,
        adjacent_if_missing: bool = True,
        required_month_ids: Iterable[str] = (),
    ) -> list[_MonthEntry]:
        """按實際月份時間軸選取當月及必要相鄰月份，維持 lazy 載入。

        月份目錄名稱只用來找到候選檔案；真正的 before／after 由每個產品自己的
        ``time_utc_ns`` 決定。當目標早於當月第一列、晚於最後一列、當月整月不存在，
        或恰好落在最後列需要檢查跨月 duplicate 時，才 pin 並載入前／後相鄰月份。
        這能處理產品把月界 halo 放在前一個月份（例如前月最後列為下月 00:00）的
        情況，而不會為每次一般月內取樣預載整套資料。
        """

        current = self._pin_and_get_month(month_id)
        entries = [current]
        current_times = self._time_array_for_entry(current, source=source)
        neighbors: list[int] = []
        if current_times is None or current_times.size == 0:
            # NWW 缺月通常只需回傳 WAVE_UNSUPPORTED；只有 OCM bracket 已證明需要
            # 某個相鄰月份時，caller 才以 required_month_ids 明示探查，避免每次
            # 一般月內取樣把三個 NWW loader 都叫醒。OCM path 保留雙向探查，因為
            # 它是速度積分的必要時間支援來源。
            neighbors = [-1, 1] if adjacent_if_missing else []
        elif target_ns < int(current_times[0]):
            neighbors = [-1]
        elif target_ns > int(current_times[-1]):
            neighbors = [1]
        elif target_ns == int(current_times[-1]):
            # 後月份的同 UTC 列是 canonical prefer-last；沒有 duplicate 時也只是一次
            # 明確的相鄰時間軸探查，下一次會由 LRU 命中，不會複製 forcing array。
            neighbors = [1]
        else:
            # 下一月份可能把第一筆 leading halo 放在「本月最後一筆」以前，而不一定
            # 落在下一曆月月初。例如 current=[22:00,23:00]、next=[23:00,00:00]
            # 時，22:30 的 after 端點也必須採 next 的 23:00，才能符合全期
            # sort-and-deduplicate-prefer-last。只用 next calendar month start 判斷會
            # 漏掉這種重複，因此允許 current 末列在下一曆月起點前一個 maximum gap
            # 內；只有同時接近 current 末列時才探查下一月份，避免一般月內查詢多載入。
            next_month = _adjacent_month_id(month_id, 1)
            next_start_ns = _month_start_ns(next_month)
            limit = (
                int(current.ocm.maximum_time_gap_ns)
                if source == "ocm" and current.ocm is not None
                else None
            )
            if source == "nww":
                nww_current = self._ensure_nww(current)
                limit = int(nww_current.maximum_time_gap_ns) if nww_current is not None else None
            if (
                limit is not None
                and int(current_times[-1]) >= next_start_ns - limit
                and target_ns >= int(current_times[-1]) - limit
            ):
                neighbors = [1]
        candidate_ids = [_adjacent_month_id(month_id, offset) for offset in neighbors]
        # caller 可能用 set 傳入 OCM bracket 月份；排序後固定 loader／LRU 順序，避免
        # Python hash iteration 只改變快取統計而影響任何可重建的 provenance。
        candidate_ids.extend(
            sorted({_validate_month_id(value) for value in required_month_ids})
        )
        seen_ids = {item.month_id for item in entries}
        for neighbor_id in candidate_ids:
            if neighbor_id in seen_ids:
                continue
            neighbor = self._pin_and_get_month(neighbor_id)
            entries.append(neighbor)
            seen_ids.add(neighbor.month_id)
        return entries

    def _find_time_bracket(
        self,
        entries: Iterable[_MonthEntry],
        target_ns: int,
        *,
        source: str,
    ) -> _GlobalTimeBracket:
        """在已載入月份的實際 UTC 聯集中建立 deterministic before／after。

        先處理 exact，再處理前後端點；同 UTC duplicate 依月份順序採後者，與
        input/preflight 的 canonical prefer-last 一致。非 exact 時刻若兩端間距超過
        兩端產品中較嚴格的 maximum gap，就回 ``TIME_GAP``；沒有兩側資料則回
        ``OUTSIDE_TIME_RANGE``。此 helper 絕不建立最近值或跨缺口外插。
        """

        records: list[tuple[_MonthEntry, np.ndarray]] = []
        for entry in entries:
            times = self._time_array_for_entry(entry, source=source)
            if times is not None and times.size:
                records.append((entry, times))
        if not records:
            return _GlobalTimeBracket(None, None, 0.0, SampleQC.OUTSIDE_TIME_RANGE)

        exact: list[tuple[_MonthEntry, int, int]] = []
        before_candidates: list[tuple[_MonthEntry, int, int]] = []
        after_candidates: list[tuple[_MonthEntry, int, int]] = []
        for entry, times in records:
            # 產品 constructor 已保證時間軸嚴格遞增；單次 searchsorted 同時取得 exact、
            # before、after，避免每個 RK stage 另以 times == target_ns 線性掃描整個月份。
            # 跨月份 exact duplicate 仍會收集每個 entry 的單一候選，最後由月份排序採
            # canonical prefer-last；資料列內若違反嚴格遞增則由 reader 先行拒絕。
            position = int(np.searchsorted(times, target_ns, side="left"))
            if position < times.size and int(times[position]) == target_ns:
                exact.append((entry, position, target_ns))
                continue
            if position > 0:
                index = position - 1
                before_candidates.append((entry, index, int(times[index])))
            if position < times.size:
                index = position
                after_candidates.append((entry, index, int(times[index])))
        if exact:
            entry, index, timestamp = max(exact, key=lambda item: (item[0].month_id, item[1]))
            return _GlobalTimeBracket(
                (entry, index, timestamp),
                (entry, index, timestamp),
                0.0,
                SampleQC.OK,
            )
        if not before_candidates or not after_candidates:
            return _GlobalTimeBracket(None, None, 0.0, SampleQC.OUTSIDE_TIME_RANGE)

        latest_time = max(item[2] for item in before_candidates)
        before_same_time = [item for item in before_candidates if item[2] == latest_time]
        before = max(before_same_time, key=lambda item: (item[0].month_id, item[1]))
        earliest_time = min(item[2] for item in after_candidates)
        after_same_time = [item for item in after_candidates if item[2] == earliest_time]
        after = max(after_same_time, key=lambda item: (item[0].month_id, item[1]))
        span = after[2] - before[2]
        if span <= 0:
            return _GlobalTimeBracket(before, after, 0.0, SampleQC.NUMERICAL_FAILURE)
        objects = []
        if source == "ocm":
            objects = [item[0].ocm for item in (before, after)]
        else:
            objects = [self._ensure_nww(item[0]) for item in (before, after)]
        limits = [
            int(value.maximum_time_gap_ns)
            for value in objects
            if value is not None
        ]
        if not limits or span > min(limits):
            return _GlobalTimeBracket(before, after, 0.0, SampleQC.TIME_GAP)
        alpha = (target_ns - before[2]) / span
        return _GlobalTimeBracket(before, after, float(alpha), SampleQC.OK)

    @staticmethod
    def _is_same_month_hot_path_safe(
        month_id: str,
        target_ns: int,
        forcing: OCMNativeMonth | NWWAnalysisMonth | None,
    ) -> bool:
        """判斷是否能安全沿用單月取樣器，避免建立全域時間 bracket。

        ``forcing.time_utc_ns`` 是該產品實際保存的嚴格遞增 UTC 奈秒軸；月份目錄名稱
        只用來推算下一個曆月的起點，不代表資料一定從月初到月底完整覆蓋。這個熱路徑
        只讀時間軸首列、末列與產品的 ``maximum_time_gap_ns``，不執行二分搜尋，也不
        預載相鄰月份。查詢必須嚴格位於首列與末列之間，因為末列可能與下一月重複，月尾
        查詢需由全域 bracket 執行 canonical prefer-last；若末列是下一曆月 halo，靠近
        該末列的最大允許時間間隔也必須交給全域路徑確認相鄰月份。通過此判定只表示
        「不需跨月找端點」，同月內部仍可能有真正時間缺口，最後交由既有單月 ``sample``
        回傳 ``TIME_GAP``，不以最近值、零值或外插補齊。

        參數 ``forcing`` 可是 OCM 或 NWW3 產品；傳入 ``None``、空時間軸或不符合首末
        範圍時回傳 ``False``，讓呼叫端走原本的 lazy 相鄰月份與全域 endpoint 流程。
        """

        if forcing is None:
            return False
        times = forcing.time_utc_ns
        if times.ndim != 1 or times.size == 0:
            return False
        first_ns = int(times[0])
        last_ns = int(times[-1])
        # 等於末列時保守走全域 path：既有流程會探查下一月，才能維持月界
        # exact duplicate 的 canonical prefer-last；大於末列則顯然需要 after endpoint。
        if target_ns < first_ns or target_ns >= last_ns:
            return False
        # 下一月份的 leading halo 可能早於下一曆月起點，甚至與 current 的末列重疊；
        # 只看「末列是否已跨入下一曆月」會把 22:30 的 after 錯留在 current。先確認
        # current 末列位於下一曆月起點前一個 maximum gap 內，再將 current 尾端的
        # maximum-gap window 交給 global bracket，由實際相鄰時間軸決定是否真的使用鄰月。
        next_month_start_ns = _month_start_ns(_adjacent_month_id(month_id, 1))
        maximum_gap_ns = int(forcing.maximum_time_gap_ns)
        return not (
            last_ns >= next_month_start_ns - maximum_gap_ns
            and target_ns >= last_ns - maximum_gap_ns
        )

    def _try_same_month_hot_sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        month_id: str,
        settling_velocity_mps: float,
        include_stokes: bool,
        triangle_hint: int | None,
    ) -> VelocitySample | None:
        """在可由首末時間軸證明安全時，以一次 OCM 查詢完成同月速度取樣。

        這個 helper 先以一次 ``_get_month`` 取得 query 曆月，並在 Stokes 開啟時只對
        同一 entry 做一次 lazy NWW 嘗試。通過 OCM 與 NWW 的首末／halo 檢查後，直接
        重用已快取的 ``CombinedMonthForcing.sample``，所以暖機後每次 no-Stokes sample
        只產生一次 OCM cache hit，Stokes sample 再產生一次 NWW cache hit。若任何產品
        可能需要相鄰月份，回傳 ``None`` 交給既有全域 bracket；若 OCM 安全但 NWW 整月
        缺失，則保留原本「單次 NWW loader 後回 ``WAVE_UNSUPPORTED``」的行為。單月
        ``CombinedMonthForcing.sample`` 仍自行判定同月內部缺口、遮罩、垂向支援與物理
        QC，因此本快取最佳化不會把時間範圍檢查放寬。
        """

        entry = self._get_month(month_id)
        if not self._is_same_month_hot_path_safe(month_id, time_utc_ns, entry.ocm):
            return None
        assert entry.ocm is not None
        if include_stokes:
            nww = self._ensure_nww(entry)
            if nww is None:
                current = entry.ocm.sample(
                    x_m,
                    y_m,
                    z_m,
                    time_utc_ns,
                    triangle_hint=triangle_hint,
                )
                if not current.valid:
                    return current
                return replace(
                    current,
                    w_mps=current.w_mps + settling_velocity_mps,
                    qc=SampleQC.WAVE_UNSUPPORTED,
                    diagnostics={**current.diagnostics, "nww_month_missing": 1},
                )
            if not self._is_same_month_hot_path_safe(month_id, time_utc_ns, nww):
                return None
        combined = self._combined(
            entry,
            settling_velocity_mps=settling_velocity_mps,
            include_stokes=include_stokes,
        )
        return combined.sample(
            x_m,
            y_m,
            z_m,
            time_utc_ns,
            triangle_hint=triangle_hint,
        )

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
            signature = self._nww_grid_signature_for(result)
            if self._nww_grid_signature is None:
                self._nww_grid_signature = signature
            elif signature != self._nww_grid_signature:
                raise ValueError(f"NWW loader grid identity 不一致：{entry.month_id}")
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
            physics_kernel_backend=self.physics_kernel_backend,
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

    @staticmethod
    def _invalid_time_bracket_sample(
        month_id: str,
        qc: SampleQC,
        *,
        before_month: str | None = None,
        after_month: str | None = None,
    ) -> VelocitySample:
        """建立跨月時間 bracket 失敗樣本，明確區分缺時與整月不存在。"""

        diagnostics: dict[str, bool | float | int | str] = {
            "cross_month_endpoint": 1,
            "time_bracket_qc": int(qc),
        }
        if before_month is not None:
            diagnostics["endpoint_month_before"] = before_month
        if after_month is not None:
            diagnostics["endpoint_month_after"] = after_month
        return VelocitySample(
            0.0,
            0.0,
            0.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            qc,
            forcing_month_id=month_id,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _interpolated_geometry(
        before: tuple[float, float] | None,
        after: tuple[float, float] | None,
        alpha: float,
    ) -> tuple[float, float] | None:
        """以 OCM 兩 endpoint 的海面／海床建立 query-time 幾何上下界。"""

        if before is None or after is None:
            return None
        eta_before, bed_before = before
        eta_after, bed_after = after
        values = (eta_before, bed_before, eta_after, bed_after)
        if not np.all(np.isfinite(values)):
            return None
        eta = float(eta_before + alpha * (eta_after - eta_before))
        bed = float(bed_before + alpha * (bed_after - bed_before))
        return (eta, bed) if np.isfinite(eta) and np.isfinite(bed) else None

    def _cross_velocity_sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        settling_velocity_mps: float,
        include_stokes: bool,
        ocm_bracket: _GlobalTimeBracket,
        nww_bracket: _GlobalTimeBracket | None,
        query_month_id: str,
        triangle_hint: int | None,
    ) -> VelocitySample:
        """組合跨月份 OCM／NWW endpoint，並讓 CombinedForcing 完成 Stokes 合成。"""

        if ocm_bracket.before is None or ocm_bracket.after is None:
            return self._invalid_time_bracket_sample(query_month_id, ocm_bracket.qc)
        ocm_before_entry, ocm_before_index, _ = ocm_bracket.before
        ocm_after_entry, ocm_after_index, _ = ocm_bracket.after
        if ocm_before_entry.ocm is None or ocm_after_entry.ocm is None:
            return self._invalid_time_bracket_sample(query_month_id, SampleQC.OUTSIDE_TIME_RANGE)
        before_location, before_geometry = ocm_before_entry.ocm.geometry_at_time_index(
            x_m,
            y_m,
            ocm_before_index,
            triangle_hint=triangle_hint,
        )
        after_location, after_geometry = ocm_after_entry.ocm.geometry_at_time_index(
            x_m,
            y_m,
            ocm_after_index,
            triangle_hint=triangle_hint,
        )
        if before_location is None or after_location is None:
            return self._invalid_time_bracket_sample(
                query_month_id,
                SampleQC.OUTSIDE_HORIZONTAL_DOMAIN,
                before_month=ocm_before_entry.month_id,
                after_month=ocm_after_entry.month_id,
            )
        if (
            before_location.triangle_id != after_location.triangle_id
            or before_location.source_face_global_index != after_location.source_face_global_index
        ):
            return self._invalid_time_bracket_sample(
                query_month_id,
                SampleQC.NUMERICAL_FAILURE,
                before_month=ocm_before_entry.month_id,
                after_month=ocm_after_entry.month_id,
            )
        geometry = self._interpolated_geometry(before_geometry, after_geometry, ocm_bracket.alpha)
        support_z = _cross_month_support_z(z_m, geometry)
        if geometry is None or not _query_z_within_geometric_bounds(z_m, geometry):
            return self._invalid_time_bracket_sample(
                query_month_id,
                SampleQC.VERTICAL_UNSUPPORTED,
                before_month=ocm_before_entry.month_id,
                after_month=ocm_after_entry.month_id,
            )
        ocm_before = ocm_before_entry.ocm.sample_at_time_index(
            x_m,
            y_m,
            z_m,
            ocm_before_index,
            triangle_hint=triangle_hint,
            support_z_m=support_z,
            enforce_geometric_bounds=False,
        )
        ocm_after = ocm_after_entry.ocm.sample_at_time_index(
            x_m,
            y_m,
            z_m,
            ocm_after_index,
            triangle_hint=triangle_hint,
            support_z_m=support_z,
            enforce_geometric_bounds=False,
        )
        nww_before_sample = None
        nww_after_sample = None
        nww_before_month = None
        nww_after_month = None
        nww_qc = SampleQC.OK
        nww_missing = False
        if include_stokes:
            if nww_bracket is None or nww_bracket.before is None or nww_bracket.after is None:
                nww_missing = True
                nww_qc = (
                    SampleQC.WAVE_UNSUPPORTED
                    if nww_bracket is None
                    else nww_bracket.qc
                )
            elif nww_bracket.qc != SampleQC.OK:
                nww_qc = nww_bracket.qc
            else:
                nww_before_entry, nww_before_index, _ = nww_bracket.before
                nww_after_entry, nww_after_index, _ = nww_bracket.after
                nww_before = self._ensure_nww(nww_before_entry)
                nww_after = self._ensure_nww(nww_after_entry)
                nww_before_month = nww_before_entry.month_id
                nww_after_month = nww_after_entry.month_id
                if nww_before is None or nww_after is None:
                    nww_missing = True
                    nww_qc = SampleQC.WAVE_UNSUPPORTED
                else:
                    lon, lat = self.projection.unproject(x_m, y_m)
                    nww_before_sample = nww_before.sample_at_time_index(
                        float(lon),
                        float(lat),
                        nww_before_index,
                    )
                    nww_after_sample = nww_after.sample_at_time_index(
                        float(lon),
                        float(lat),
                        nww_after_index,
                    )
        forcing_month_id = ocm_after_entry.month_id
        if not include_stokes or (nww_bracket is not None and nww_bracket.qc == SampleQC.OK):
            combined = CombinedMonthForcing(
                ocm=ocm_before_entry.ocm,
                nww=(
                    self._ensure_nww(nww_bracket.before[0])
                    if include_stokes and nww_bracket is not None and nww_bracket.before is not None
                    else None
                ),
                projection=self.projection,
                settling_velocity_mps=settling_velocity_mps,
                include_stokes=include_stokes,
                physics_kernel_backend=self.physics_kernel_backend,
            )
            return combined.sample_from_endpoints(
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                time_utc_ns=time_utc_ns,
                ocm_before=ocm_before,
                ocm_after=ocm_after,
                ocm_alpha=ocm_bracket.alpha,
                nww_before=nww_before_sample,
                nww_after=nww_after_sample,
                nww_alpha=0.0 if nww_bracket is None else nww_bracket.alpha,
                forcing_month_id=forcing_month_id,
                endpoint_month_before=ocm_before_entry.month_id,
                endpoint_month_after=ocm_after_entry.month_id,
                endpoint_nww_month_before=nww_before_month,
                endpoint_nww_month_after=nww_after_month,
            )
        combined = CombinedMonthForcing(
            ocm=ocm_before_entry.ocm,
            nww=None,
            projection=self.projection,
            settling_velocity_mps=settling_velocity_mps,
            include_stokes=False,
            physics_kernel_backend=self.physics_kernel_backend,
        )
        current = combined.sample_from_endpoints(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            time_utc_ns=time_utc_ns,
            ocm_before=ocm_before,
            ocm_after=ocm_after,
            ocm_alpha=ocm_bracket.alpha,
            forcing_month_id=forcing_month_id,
            endpoint_month_before=ocm_before_entry.month_id,
            endpoint_month_after=ocm_after_entry.month_id,
        )
        if not current.valid:
            return current
        diagnostics = {
            **current.diagnostics,
            "nww_month_missing": int(nww_missing),
            "nww_time_bracket_qc": int(nww_qc),
        }
        return replace(
            current,
            qc=SampleQC.WAVE_UNSUPPORTED if nww_missing else nww_qc,
            components=None,
            diagnostics=diagnostics,
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
        """依產品實際 UTC 時間軸取樣，必要時跨相鄰月份拼接兩個 endpoint。

        月份目錄只決定 lazy loader 的初始候選；真正時間支援由 OCM（以及啟用 Stokes
        時的 NWW3）各自的 ``time_utc_ns`` 建立。月界 halo 的 exact duplicate 遵守
        canonical prefer-last；非 exact query 若兩端 span 超過原產品的 maximum gap
        仍回 ``TIME_GAP``，沒有兩側資料則回 ``OUTSIDE_TIME_RANGE``。跨月 scope 會
        暫時 pin 兩端，避免 ``max_resident_months=1`` 在第二端載入時淘汰仍在使用的
        第一端；資料陣列、遮罩、垂向支援與 Stokes／沉降物理都由原 endpoint／combined
        API 保持，不以最近值、零值或合成 total velocity 補洞。
        """

        month_id = _month_from_utc_ns(time_utc_ns)
        settling = _finite_settling(settling_velocity_mps)
        if not isinstance(include_stokes, bool):
            raise TypeError("include_stokes 必須是 boolean")
        hot_sample = self._try_same_month_hot_sample(
            x_m,
            y_m,
            z_m,
            time_utc_ns,
            month_id=month_id,
            settling_velocity_mps=settling,
            include_stokes=include_stokes,
            triangle_hint=triangle_hint,
        )
        if hot_sample is not None:
            return hot_sample
        with self._pin_months((month_id,)):
            ocm_entries = self._prepare_time_entries(month_id, time_utc_ns, source="ocm")
            ocm_bracket = self._find_time_bracket(
                ocm_entries,
                time_utc_ns,
                source="ocm",
            )
            if ocm_bracket.qc != SampleQC.OK:
                return self._invalid_time_bracket_sample(month_id, ocm_bracket.qc)
            nww_bracket = None
            if include_stokes:
                ocm_months_for_nww = {
                    item[0].month_id
                    for item in (ocm_bracket.before, ocm_bracket.after)
                    if item is not None
                }
                nww_entries = self._prepare_time_entries(
                    month_id,
                    time_utc_ns,
                    source="nww",
                    adjacent_if_missing=False,
                    required_month_ids=ocm_months_for_nww,
                )
                nww_bracket = self._find_time_bracket(
                    nww_entries,
                    time_utc_ns,
                    source="nww",
                )
            ocm_before = ocm_bracket.before
            ocm_after = ocm_bracket.after
            same_ocm = (
                ocm_before is not None
                and ocm_after is not None
                and ocm_before[0].month_id == ocm_after[0].month_id
            )
            same_nww = nww_bracket is None or (
                nww_bracket.before is not None
                and nww_bracket.after is not None
                and nww_bracket.before[0].month_id == nww_bracket.after[0].month_id
            )
            # Stokes fast path 必須確定 OCM 與 NWW 的 before／after 實際來自同一月份。
            # 月份目錄只是 lazy lookup hint；例如 OCM 前月 halo 與 NWW 當月首列可能
            # 共享 exact UTC。若只看各自 bracket「內部同月」，會把 OCM entry 的 NWW
            # 誤當成 exact 波浪端點，故此處要求兩套 endpoint 月份逐一相等。
            same_product_endpoint_months = not include_stokes
            if include_stokes and nww_bracket is not None:
                same_product_endpoint_months = (
                    ocm_before is not None
                    and ocm_after is not None
                    and nww_bracket.before is not None
                    and nww_bracket.after is not None
                    and ocm_before[0].month_id == nww_bracket.before[0].month_id
                    and ocm_after[0].month_id == nww_bracket.after[0].month_id
                )
            if (
                same_ocm
                and same_nww
                and same_product_endpoint_months
                and ocm_before is not None
            ):
                entry = ocm_before[0]
                if entry.ocm is None:
                    return self._missing_ocm_sample(month_id)
                if include_stokes and self._ensure_nww(entry) is None:
                    current = entry.ocm.sample(
                        x_m,
                        y_m,
                        z_m,
                        time_utc_ns,
                        triangle_hint=triangle_hint,
                    )
                    if not current.valid:
                        return current
                    return replace(
                        current,
                        w_mps=current.w_mps + settling,
                        qc=SampleQC.WAVE_UNSUPPORTED,
                        diagnostics={**current.diagnostics, "nww_month_missing": 1},
                    )
                combined = self._combined(
                    entry,
                    settling_velocity_mps=settling,
                    include_stokes=include_stokes,
                )
                return combined.sample(
                    x_m,
                    y_m,
                    z_m,
                    time_utc_ns,
                    triangle_hint=triangle_hint,
                )
            return self._cross_velocity_sample(
                x_m,
                y_m,
                z_m,
                time_utc_ns,
                settling_velocity_mps=settling,
                include_stokes=include_stokes,
                ocm_bracket=ocm_bracket,
                nww_bracket=nww_bracket,
                query_month_id=month_id,
                triangle_hint=triangle_hint,
            )

    def provider(self, settling_velocity_mps: float, include_stokes: bool) -> ManagedForcingProvider:
        """回傳可直接交給 reference request 或 hint wrapper 的輕量 provider。"""

        return ManagedForcingProvider(self, settling_velocity_mps, include_stokes)

    def _cross_smagorinsky_sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        settings: SmagorinskySettings,
        *,
        bracket: _GlobalTimeBracket,
        query_month_id: str,
        triangle_hint: int | None,
    ) -> DiffusionSample:
        """以兩個 OCM endpoint 先各自算非線性 Kh，再做時間內插。

        每個端點都由 ``OCMNativeMonth.sample_smagorinsky_diffusion_at_time_index`` 完成
        incident triangle 遮罩、垂向支援、floor/cap 與梯度計算；這裡只內插已完成的
        P1 nodal 結果所對應的粒子 Kh／梯度，保留原 Smagorinsky 非線性運算先於時間
        內插的順序。兩端 mesh location 或 query-time eta／海床無法一致時回非零 QC，
        不以零擴散取代缺資料。
        """

        if bracket.before is None or bracket.after is None:
            return self._missing_smagorinsky_sample(
                query_month_id,
                settings,
                flow_domain_id=self.flow_domain_id,
            )
        before_entry, before_index, _ = bracket.before
        after_entry, after_index, _ = bracket.after
        if before_entry.ocm is None or after_entry.ocm is None:
            return self._missing_smagorinsky_sample(
                query_month_id,
                settings,
                flow_domain_id=self.flow_domain_id,
            )
        before_location, before_geometry = before_entry.ocm.geometry_at_time_index(
            x_m,
            y_m,
            before_index,
            triangle_hint=triangle_hint,
        )
        after_location, after_geometry = after_entry.ocm.geometry_at_time_index(
            x_m,
            y_m,
            after_index,
            triangle_hint=triangle_hint,
        )
        if before_location is None or after_location is None:
            return DiffusionSample(
                coefficients=DiffusionCoefficients(0.0, 0.0, settings.constant_kz_m2ps),
                diffusivity_divergence_mps=(np.nan, np.nan, np.nan),
                qc=SampleQC.OUTSIDE_HORIZONTAL_DOMAIN,
                diagnostics={
                    "method": "smagorinsky_native_mesh_p1_nodal",
                    "forcing_month_id": query_month_id,
                    "cross_month_endpoint": 1,
                },
            )
        if (
            before_location.triangle_id != after_location.triangle_id
            or before_location.source_face_global_index != after_location.source_face_global_index
        ):
            return DiffusionSample(
                coefficients=DiffusionCoefficients(0.0, 0.0, settings.constant_kz_m2ps),
                diffusivity_divergence_mps=(np.nan, np.nan, np.nan),
                qc=SampleQC.NUMERICAL_FAILURE,
                diagnostics={
                    "method": "smagorinsky_native_mesh_p1_nodal",
                    "forcing_month_id": query_month_id,
                    "cross_month_endpoint": 1,
                    "endpoint_identity_mismatch": 1,
                },
            )
        geometry = self._interpolated_geometry(before_geometry, after_geometry, bracket.alpha)
        if geometry is None or not _query_z_within_geometric_bounds(z_m, geometry):
            return DiffusionSample(
                coefficients=DiffusionCoefficients(0.0, 0.0, settings.constant_kz_m2ps),
                diffusivity_divergence_mps=(np.nan, np.nan, np.nan),
                qc=SampleQC.VERTICAL_UNSUPPORTED,
                diagnostics={
                    "method": "smagorinsky_native_mesh_p1_nodal",
                    "forcing_month_id": query_month_id,
                    "cross_month_endpoint": 1,
                },
            )
        support_z = _cross_month_support_z(z_m, geometry)
        before_sample = before_entry.ocm.sample_smagorinsky_diffusion_at_time_index(
            x_m,
            y_m,
            z_m,
            before_index,
            settings,
            triangle_hint=triangle_hint,
            support_z_m=support_z,
            enforce_geometric_bounds=False,
        )
        after_sample = after_entry.ocm.sample_smagorinsky_diffusion_at_time_index(
            x_m,
            y_m,
            z_m,
            after_index,
            settings,
            triangle_hint=triangle_hint,
            support_z_m=support_z,
            enforce_geometric_bounds=False,
        )
        if not before_sample.valid or not after_sample.valid:
            return DiffusionSample(
                coefficients=DiffusionCoefficients(0.0, 0.0, settings.constant_kz_m2ps),
                diffusivity_divergence_mps=(np.nan, np.nan, np.nan),
                qc=before_sample.qc | after_sample.qc,
                diagnostics={
                    "method": "smagorinsky_native_mesh_p1_nodal",
                    "forcing_month_id": query_month_id,
                    "cross_month_endpoint": 1,
                    "endpoint_month_before": before_entry.month_id,
                    "endpoint_month_after": after_entry.month_id,
                },
            )
        weight = float(bracket.alpha)
        kh = before_sample.coefficients.kx_m2ps + weight * (
            after_sample.coefficients.kx_m2ps - before_sample.coefficients.kx_m2ps
        )
        grad_x = before_sample.diffusivity_divergence_mps[0] + weight * (
            after_sample.diffusivity_divergence_mps[0]
            - before_sample.diffusivity_divergence_mps[0]
        )
        grad_y = before_sample.diffusivity_divergence_mps[1] + weight * (
            after_sample.diffusivity_divergence_mps[1]
            - before_sample.diffusivity_divergence_mps[1]
        )
        raw_before = before_sample.diagnostics.get("raw_current_triangle_kh_m2ps")
        raw_after = after_sample.diagnostics.get("raw_current_triangle_kh_m2ps")
        diagnostics: dict[str, bool | float | int | str] = {
            "method": "smagorinsky_native_mesh_p1_nodal",
            "coefficient_cs": settings.coefficient_cs,
            "kh_m2ps": float(kh),
            "d_kh_dx_mps": float(grad_x),
            "d_kh_dy_mps": float(grad_y),
            "floor_hit": bool(
                before_sample.diagnostics.get("floor_hit", False)
                or after_sample.diagnostics.get("floor_hit", False)
            ),
            "cap_hit": bool(
                before_sample.diagnostics.get("cap_hit", False)
                or after_sample.diagnostics.get("cap_hit", False)
            ),
            "valid_incident_triangle_count": min(
                int(before_sample.diagnostics.get("valid_incident_triangle_count", 0)),
                int(after_sample.diagnostics.get("valid_incident_triangle_count", 0)),
            ),
            "excluded_incident_triangle_count": max(
                int(before_sample.diagnostics.get("excluded_incident_triangle_count", 0)),
                int(after_sample.diagnostics.get("excluded_incident_triangle_count", 0)),
            ),
            "valid_incident_triangle_count_before": int(
                before_sample.diagnostics.get("valid_incident_triangle_count", 0)
            ),
            "valid_incident_triangle_count_after": int(
                after_sample.diagnostics.get("valid_incident_triangle_count", 0)
            ),
            "excluded_incident_triangle_count_before": int(
                before_sample.diagnostics.get("excluded_incident_triangle_count", 0)
            ),
            "excluded_incident_triangle_count_after": int(
                after_sample.diagnostics.get("excluded_incident_triangle_count", 0)
            ),
            "triangle_id": int(before_location.triangle_id),
        }
        if raw_before is not None and raw_after is not None:
            diagnostics["raw_current_triangle_kh_m2ps"] = float(raw_before) + weight * (
                float(raw_after) - float(raw_before)
            )
        return DiffusionSample(
            coefficients=DiffusionCoefficients(float(kh), float(kh), settings.constant_kz_m2ps),
            diffusivity_divergence_mps=(float(grad_x), float(grad_y), 0.0),
            qc=SampleQC.OK,
            diagnostics=diagnostics,
        )

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
        """依 OCM 實際 UTC 軸取樣 Smagorinsky，必要時跨月份計算兩個非線性 endpoint。

        manager 仍只在取樣時 lazy 載入當月／相鄰月份，並沿用同一 OCM LRU。單月內部
        查詢維持既有 ``OCMNativeMonth.sample_smagorinsky_diffusion``；跨月時先在各端
        完成 native mesh 的 Kh candidate、floor/cap、nodal area weighting 與 gradient，
        再依全域 UTC bracket 線性內插。這條路徑不呼叫 NWW 或 Combined forcing；真正
        的資料缺時仍回 ``TIME_GAP``／``OUTSIDE_TIME_RANGE``，不改用最近月或零擴散。
        """

        if not isinstance(settings, SmagorinskySettings):
            raise TypeError("settings 必須是 SmagorinskySettings")
        settings.validate()
        month_id = _month_from_utc_ns(time_utc_ns)
        with self._pin_months((month_id,)):
            entries = self._prepare_time_entries(month_id, time_utc_ns, source="ocm")
            bracket = self._find_time_bracket(entries, time_utc_ns, source="ocm")
            if bracket.qc != SampleQC.OK:
                return self._missing_smagorinsky_sample(
                    month_id,
                    settings,
                    flow_domain_id=self.flow_domain_id,
                ) if bracket.qc == SampleQC.OUTSIDE_TIME_RANGE else DiffusionSample(
                    coefficients=DiffusionCoefficients(0.0, 0.0, settings.constant_kz_m2ps),
                    diffusivity_divergence_mps=(np.nan, np.nan, np.nan),
                    qc=bracket.qc,
                    diagnostics={
                        "method": "smagorinsky_native_mesh_p1_nodal",
                        "forcing_month_id": month_id,
                        "time_bracket_qc": int(bracket.qc),
                    },
                )
            before = bracket.before
            after = bracket.after
            if before is None or after is None:
                return self._missing_smagorinsky_sample(
                    month_id,
                    settings,
                    flow_domain_id=self.flow_domain_id,
                )
            if before[0].month_id == after[0].month_id:
                entry = before[0]
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
            return self._cross_smagorinsky_sample(
                x_m,
                y_m,
                z_m,
                settings,
                bracket=bracket,
                query_month_id=month_id,
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
