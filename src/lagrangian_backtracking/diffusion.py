"""定義空間擴散取樣、隨機位移與安全時間步長的基準介面。

擴散係數的單位是平方公尺每秒，粒子座標與擴散梯度都在公尺制運算座標中處理。常數
係數仍沿用既有 ``DiffusionCoefficients`` 與 ``brownian_displacement`` 行為；空間變化
係數則由 ``SpatialDiffusionProvider`` 在步首回傳一次 ``DiffusionSample``，以
``+div(K)|dt|`` 的 Euler--Maruyama operator split 漂移搭配同一個 Brownian 增量。這是
逆向回溯的 pseudo-time generator 約定，不是宣稱已完成嚴格的 reversed-time SDE 推導。
隨機增量永遠在完整四階 Runge--Kutta（RK4）確定性步驟之後加入，不能插入 RK4 stage。

Smagorinsky 方法在本模組只產生候選水平擴散係數與上下限命中紀錄；正式科學基準仍須
另以 well-mixed、擴散障壁及解析統計測試驗證，不能因介面已存在就把它視為 OCM 正式
擴散產品。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from .models import SampleQC


@dataclass(frozen=True, slots=True)
class DiffusionCoefficients:
    """東向、北向與垂向三個方向的擴散係數，單位為平方公尺每秒。"""

    kx_m2ps: float
    ky_m2ps: float
    kz_m2ps: float

    def validate(self) -> None:
        """負值或非有限擴散係數代表模型不合理，必須在產生亂數前拒絕。"""

        values = (self.kx_m2ps, self.ky_m2ps, self.kz_m2ps)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Kx/Ky/Kz 必須是有限非負 m²/s")


@dataclass(frozen=True, slots=True)
class SmagorinskySettings:
    """P1 nodal Smagorinsky 取樣所需的固定物理參數與上下限。

    ``coefficient_cs`` 是無因次 Smagorinsky 常數，必須嚴格大於零；``floor_m2ps`` 與
    ``cap_m2ps`` 是每個 triangle candidate 在面積加權前套用的 Kh 下／上限，單位為
    m²/s；``constant_kz_m2ps`` 是本 Slice 2B1 暫時沿用的常數垂向係數，亦為 m²/s。
    這個設定只描述取樣演算法，不代表已核定正式 experiment case 或已完成 OCM
    well-mixed/PDE 驗證。
    """

    coefficient_cs: float
    floor_m2ps: float
    cap_m2ps: float
    constant_kz_m2ps: float

    def __post_init__(self) -> None:
        """將數值固定成原生有限 float，並在 provider 建立前拒絕不可能的物理設定。"""

        normalized = {
            "coefficient_cs": _finite_parameter(
                self.coefficient_cs, label="coefficient_cs", strictly_positive=True
            ),
            "floor_m2ps": _finite_parameter(self.floor_m2ps, label="floor_m2ps"),
            "cap_m2ps": _finite_parameter(self.cap_m2ps, label="cap_m2ps"),
            "constant_kz_m2ps": _finite_parameter(
                self.constant_kz_m2ps, label="constant_kz_m2ps"
            ),
        }
        if normalized["floor_m2ps"] > normalized["cap_m2ps"]:
            raise ValueError("floor_m2ps 不可大於 cap_m2ps")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def validate(self) -> None:
        """再次驗證 immutable settings，供外部建立資料管線時使用。"""

        # constructor 已完成相同檢查；保留公開方法與 DiffusionCoefficients.validate 的
        # 介面對稱，讓 provider 可在進入大型陣列取樣前明確執行設定閘門。
        _finite_parameter(self.coefficient_cs, label="coefficient_cs", strictly_positive=True)
        floor = _finite_parameter(self.floor_m2ps, label="floor_m2ps")
        cap = _finite_parameter(self.cap_m2ps, label="cap_m2ps")
        _finite_parameter(self.constant_kz_m2ps, label="constant_kz_m2ps")
        if floor > cap:
            raise ValueError("floor_m2ps 不可大於 cap_m2ps")


def _finite_parameter(value: object, *, label: str, strictly_positive: bool = False) -> float:
    """驗證 Smagorinsky scalar 是有限非負（或嚴格正）值並轉成 Python float。"""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} 不可為 bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 必須是有限數值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必須是有限數值")
    if strictly_positive and normalized <= 0:
        raise ValueError(f"{label} 必須大於 0")
    if not strictly_positive and normalized < 0:
        raise ValueError(f"{label} 必須是非負值")
    return normalized


@dataclass(frozen=True, slots=True)
class DiffusionSample:
    """單一粒子步首的擴散係數、梯度漂移與品質狀態。

    ``coefficients`` 是對角擴散張量的三個對角元素 ``Kx/Ky/Kz``，單位為 m²/s。
    ``diffusivity_divergence_mps`` 是同一公尺制座標下的三維向量
    ``[∂Kx/∂x, ∂Ky/∂y, ∂Kz/∂z]``，單位為 m/s；它只代表本 Slice 2A 已裁決的
    pseudo-time drift 項，不是完整張量散度或其他尚未驗證的高階校正。``qc=OK`` 才表示樣本
    可進入步長、RK4 split 與亂數計算；任何非零 ``qc`` 都會保留 provider 的失敗原因，
    不得以三個零係數偽裝成有效靜水。

    這個資料類別不可重新賦值，且將梯度固定成 tuple、診斷資料包成唯讀 mapping，避免
    取樣後 caller 修改同一個物件而造成「步長使用的 K」與「split 使用的 K」不一致。帶有
    非零品質旗標的失敗樣本可以保留 NaN 或負值作為診斷；但 ``qc=OK`` 建構時會立即
    拒絕非有限或負值，讓無效資料在消耗亂數前被截斷。
    """

    coefficients: DiffusionCoefficients
    diffusivity_divergence_mps: tuple[float, float, float] | list[float] | np.ndarray
    qc: SampleQC = SampleQC.OK
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """canonicalize immutable 欄位並在有效樣本入口執行數值閘門。

        provider 可能使用 list 或 NumPy 一維陣列回傳梯度，因此先複製成固定三元素
        tuple；只有數值可轉成浮點數才接受，避免 object array 或字串在後續位移計算中
        延遲爆炸。失敗樣本仍允許非有限／負值以保存原始診斷，但一定要有非零 ``qc``。
        診斷 mapping 只保存 shallow snapshot；其值應是可序列化 scalar，複雜陣列不能當作
        正式輸出契約的一部分。
        """

        if not isinstance(self.coefficients, DiffusionCoefficients):
            raise TypeError("coefficients 必須是 DiffusionCoefficients")
        normalized_qc = _normalize_diffusion_qc(self.qc)
        normalized_divergence = _normalize_divergence(
            self.diffusivity_divergence_mps,
            allow_nonfinite=normalized_qc != SampleQC.OK,
        )
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics 必須是 mapping")
        if any(not isinstance(key, str) for key in self.diagnostics):
            raise TypeError("diagnostics 的 key 必須是字串")
        if normalized_qc == SampleQC.OK:
            try:
                self.coefficients.validate()
            except (TypeError, ValueError) as error:
                raise ValueError("qc=OK 的 DiffusionSample 必須含有限非負 K") from error
        object.__setattr__(self, "diffusivity_divergence_mps", normalized_divergence)
        object.__setattr__(self, "qc", normalized_qc)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def valid(self) -> bool:
        """回傳是否可安全用於選步長與 Euler--Maruyama 位移。"""

        if self.qc != SampleQC.OK:
            return False
        try:
            self.coefficients.validate()
        except (TypeError, ValueError):
            return False
        return all(math.isfinite(value) for value in self.diffusivity_divergence_mps)

    def validate(self) -> None:
        """嚴格驗證有效樣本；失敗樣本只可被 engine 記錄，不能直接積分。"""

        if not self.valid:
            raise ValueError(f"DiffusionSample 無法用於積分：qc={int(self.qc)}")


@runtime_checkable
class SpatialDiffusionProvider(Protocol):
    """依粒子步首位置、UTC 時刻與網格提示回傳空間擴散樣本。

    所有輸入座標均為公尺，``time_utc_ns`` 是 UTC 奈秒整數，``triangle_hint`` 是前一個
    成功速度取樣得到的 native mesh 三角形 ID；hint 只用來改善搜尋 locality，不得改變
    provider 的物理結果。provider 每個 engine step 只會被呼叫一次，回傳資料由 engine
    同時供 choose-time-step 與 RK4 後的 diffusion split 使用。
    """

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        triangle_hint: int | None = None,
    ) -> DiffusionSample:
        """回傳指定步首位置與時間的擴散樣本。"""


DiffusionModel: TypeAlias = DiffusionCoefficients | SpatialDiffusionProvider


def _normalize_diffusion_qc(value: SampleQC | int) -> SampleQC:
    """把品質旗標固定成 ``SampleQC``，拒絕 bool 或浮點數的隱式轉換。"""

    if isinstance(value, bool) or not isinstance(value, (int, SampleQC)):
        raise TypeError("DiffusionSample.qc 必須是 SampleQC 或整數旗標")
    try:
        return SampleQC(int(value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("DiffusionSample.qc 無法轉成 SampleQC") from error


def _normalize_divergence(
    value: tuple[float, float, float] | list[float] | np.ndarray,
    *,
    allow_nonfinite: bool,
) -> tuple[float, float, float]:
    """把三維擴散梯度整理為 immutable tuple，並檢查維度與有限性政策。"""

    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("diffusivity_divergence_mps 必須是三元素序列") from error
    if values.shape != (3,):
        raise ValueError("diffusivity_divergence_mps 必須是長度 3 的一維向量")
    normalized: list[float] = []
    for item in values.tolist():
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("diffusivity_divergence_mps 不可含 bool")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("diffusivity_divergence_mps 必須含數值") from error
        if not allow_nonfinite and not math.isfinite(number):
            raise ValueError("qc=OK 的 diffusivity divergence 必須有限")
        normalized.append(number)
    return (normalized[0], normalized[1], normalized[2])


def resolve_diffusion_sample(
    diffusion: DiffusionModel,
    x_m: float,
    y_m: float,
    z_m: float,
    time_utc_ns: int,
    triangle_hint: int | None = None,
) -> DiffusionSample:
    """集中解析常數或空間 provider 的單一步首擴散樣本。

    常數係數不需要額外取樣，會建立零梯度且 ``qc=OK`` 的有效樣本；這保留舊版常數
    Brownian 流程，同時讓 engine 使用統一的資料型別。provider 則只呼叫一次，並嚴格
    要求回傳 ``DiffusionSample``；若 provider 已回傳非零品質旗標，函式原樣保留，讓
    engine 以既有 ``SamplingError``／終止事件政策停止，而不是偷偷改成零擴散繼續。

    ``triangle_hint`` 直接傳給 provider，因為它是 mesh 搜尋提示而不是物理計算輸入；
    本函式不消耗粒子 RNG，也不把座標從經緯度轉換成公尺制。
    """

    if isinstance(diffusion, DiffusionCoefficients):
        diffusion.validate()
        return DiffusionSample(
            coefficients=diffusion,
            diffusivity_divergence_mps=(0.0, 0.0, 0.0),
            qc=SampleQC.OK,
            diagnostics={"source": "constant"},
        )
    sample_method = getattr(diffusion, "sample", None)
    if not callable(sample_method):
        raise TypeError("diffusion 必須是 DiffusionCoefficients 或具有 sample 方法的 provider")
    sample = sample_method(x_m, y_m, z_m, time_utc_ns, triangle_hint=triangle_hint)
    if not isinstance(sample, DiffusionSample):
        raise TypeError("SpatialDiffusionProvider.sample 必須回傳 DiffusionSample")
    # 有效樣本在 constructor 已做一次閘門；這裡再驗證是為了防止 subclass 或低階
    # object.__setattr__ 繞過 constructor，確保 engine 取得的 sample 真能進入數值流程。
    if sample.qc == SampleQC.OK:
        sample.validate()
    return sample


def diffusion_displacement(
    sample: DiffusionSample,
    dt_seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """以 ``+div(K)|dt|`` 加 Brownian 增量產生三軸擴散位移。

    這是本專案 backward pseudo-time baseline 的 Euler--Maruyama split：確定性梯度漂移
    與隨機項都使用 ``abs(dt_seconds)``，所以同一個 RNG state 下正、負時間步長得到完全
    相同的擴散位移。函式先驗證樣本再呼叫亂數，無效樣本因此不會消耗 RNG；完整 RK4
    順序由 ``split_rk4_brownian_step`` 保證，而不是在這裡重新取樣速度。
    """

    if not isinstance(sample, DiffusionSample):
        raise TypeError("sample 必須是 DiffusionSample")
    sample.validate()
    if not math.isfinite(dt_seconds) or dt_seconds == 0:
        raise ValueError("dt_seconds 必須是有限非零值")
    absolute_dt = abs(dt_seconds)
    divergence = np.asarray(sample.diffusivity_divergence_mps, dtype=np.float64)
    # 零梯度走既有 Brownian helper，刻意保留原本的浮點運算與亂數消耗順序，讓常數
    # DiffusionCoefficients 的固定 seed 結果維持 bit-for-bit 相容。
    if not np.any(divergence):
        return brownian_displacement(sample.coefficients, dt_seconds=dt_seconds, rng=rng)
    return divergence * absolute_dt + brownian_displacement(
        sample.coefficients,
        dt_seconds=dt_seconds,
        rng=rng,
    )


@dataclass(frozen=True, slots=True)
class TimeStepDecision:
    """自動選出的時間步長絕對秒數與限制來源；回溯或正向方向由呼叫端另行指定。"""

    seconds: float
    limiting_reason: str


def brownian_displacement(
    coefficients: DiffusionCoefficients, *, dt_seconds: float, rng: np.random.Generator
) -> np.ndarray:
    """產生東、北、垂向彼此獨立的隨機擴散位移，回傳三個公尺值。"""

    coefficients.validate()
    if not math.isfinite(dt_seconds) or dt_seconds == 0:
        raise ValueError("dt_seconds 必須是有限非零值")
    diffusivity = np.array(
        [coefficients.kx_m2ps, coefficients.ky_m2ps, coefficients.kz_m2ps], dtype=np.float64
    )
    return rng.normal(size=3) * np.sqrt(2.0 * diffusivity * abs(dt_seconds))


def smagorinsky_horizontal_diffusivity(
    *,
    du_dx_per_s: float,
    du_dy_per_s: float,
    dv_dx_per_s: float,
    dv_dy_per_s: float,
    triangle_area_m2: float,
    coefficient_cs: float,
    floor_m2ps: float | None = None,
    cap_m2ps: float | None = None,
) -> tuple[float, bool, bool]:
    """依文件公式（10）計算三角形內的候選水平擴散係數，並回報是否碰到上下限。"""

    values = [du_dx_per_s, du_dy_per_s, dv_dx_per_s, dv_dy_per_s, triangle_area_m2, coefficient_cs]
    if not all(math.isfinite(value) for value in values) or triangle_area_m2 <= 0 or coefficient_cs < 0:
        raise ValueError("速度梯度需有限、triangle area 正值且 Cs 非負")
    for label, bound in (("floor_m2ps", floor_m2ps), ("cap_m2ps", cap_m2ps)):
        if bound is None:
            continue
        try:
            is_invalid_bound = (
                isinstance(bound, (bool, np.bool_)) or not math.isfinite(bound) or bound < 0
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} 必須是有限非負 m²/s 或 None") from error
        if is_invalid_bound:
            raise ValueError(f"{label} 必須是有限非負 m²/s 或 None")
    if floor_m2ps is not None and cap_m2ps is not None and floor_m2ps > cap_m2ps:
        raise ValueError("floor_m2ps 不可大於 cap_m2ps")
    delta_m = math.sqrt(triangle_area_m2)
    strain = math.sqrt((du_dx_per_s - dv_dy_per_s) ** 2 + (dv_dx_per_s + du_dy_per_s) ** 2)
    value = (coefficient_cs * delta_m) ** 2 * strain
    hit_floor = floor_m2ps is not None and value < floor_m2ps
    hit_cap = cap_m2ps is not None and value > cap_m2ps
    if floor_m2ps is not None:
        value = max(value, floor_m2ps)
    if cap_m2ps is not None:
        value = min(value, cap_m2ps)
    return value, hit_floor, hit_cap


def choose_time_step(
    *,
    speed_horizontal_mps: float,
    speed_vertical_mps: float,
    horizontal_scale_m: float,
    vertical_scale_m: float,
    coefficients: DiffusionCoefficients,
    dt_min_seconds: float,
    dt_max_seconds: float,
    seconds_to_forcing_boundary: float | None = None,
    advective_fraction: float = 0.25,
    vertical_fraction: float = 0.25,
    diffusive_fraction: float = 0.25,
) -> TimeStepDecision:
    """從水平移動、垂向移動、擴散與資料時間邊界中選擇最小且安全的步長。

    各軸 Brownian 位移的方差分別是 ``2*K_axis*dt``，所以 diffusion 限制必須將
    水平尺度與 ``max(Kx, Ky)`` 配對、將垂向尺度與 ``Kz`` 配對。禁止把三軸最大 K
    與最小尺度交叉配對，因為那會用水平大擴散係數不當限制垂向細層，或用垂向係數
    不當限制水平步長。公式中的尺度單位是 m、擴散係數是 m²/s、步長是 s；只有對應
    軸的 K 嚴格大於零時才建立該軸 diffusion candidate，K=0 表示該軸沒有 Brownian
    位移限制，而不是以零值製造有限步長。

    ``seconds_to_forcing_boundary`` 是沿目前積分方向走到下一個資料時刻邊界的正秒數。
    所有候選仍與既有水平／垂向 advection、forcing boundary 及 ``dt_max`` 一起取最小；
    若所需步長小於設定最小值，函式仍回傳最小值並標記此情況。粒子引擎必須累計
    ``minimum_clamp`` 發生次數，超過核定上限時以數值計算失敗停止，而不是在本函式
    改變 minimum clamp 政策或無限縮小步長。本函式只選步長，不改 Brownian displacement、
    RK4 stage 或 operator-split 的執行順序。
    """

    coefficients.validate()
    numeric = [
        speed_horizontal_mps,
        speed_vertical_mps,
        horizontal_scale_m,
        vertical_scale_m,
        dt_min_seconds,
        dt_max_seconds,
    ]
    if not all(math.isfinite(value) and value >= 0 for value in numeric):
        raise ValueError("time-step 輸入必須有限非負")
    if (
        horizontal_scale_m <= 0
        or vertical_scale_m <= 0
        or dt_min_seconds <= 0
        or dt_max_seconds < dt_min_seconds
    ):
        raise ValueError("尺度與 dt 範圍無效")
    candidates: list[tuple[float, str]] = [(dt_max_seconds, "maximum")]
    if speed_horizontal_mps > 0:
        candidates.append(
            (advective_fraction * horizontal_scale_m / speed_horizontal_mps, "horizontal_advection")
        )
    if speed_vertical_mps > 0:
        candidates.append((vertical_fraction * vertical_scale_m / speed_vertical_mps, "vertical_advection"))
    # Brownian 三軸的方差彼此獨立；水平候選只能由水平尺度與 Kx/Ky 決定，避免
    # 垂向小層厚和水平大 Kh 形成沒有物理意義的跨軸限制。Kx=Ky=0 時，水平
    # Brownian 位移為零，因此不加入水平 diffusion candidate。
    maximum_horizontal_k = max(coefficients.kx_m2ps, coefficients.ky_m2ps)
    if maximum_horizontal_k > 0:
        candidates.append(
            (
                (diffusive_fraction * horizontal_scale_m) ** 2
                / (2.0 * maximum_horizontal_k),
                "horizontal_diffusion",
            )
        )
    # 垂向候選只看 Kz 與垂向尺度；Kz=0 只移除垂向 diffusion 限制，不影響水平
    # diffusion、advection、forcing boundary 或 dt_max 候選。
    if coefficients.kz_m2ps > 0:
        candidates.append(
            (
                (diffusive_fraction * vertical_scale_m) ** 2
                / (2.0 * coefficients.kz_m2ps),
                "vertical_diffusion",
            )
        )
    if seconds_to_forcing_boundary is not None:
        if not math.isfinite(seconds_to_forcing_boundary) or seconds_to_forcing_boundary <= 0:
            raise ValueError("seconds_to_forcing_boundary 必須是有限正值")
        candidates.append((seconds_to_forcing_boundary, "forcing_boundary"))
    seconds, reason = min(candidates, key=lambda item: item[0])
    if seconds < dt_min_seconds:
        return TimeStepDecision(dt_min_seconds, "minimum_clamp")
    return TimeStepDecision(seconds, reason)
