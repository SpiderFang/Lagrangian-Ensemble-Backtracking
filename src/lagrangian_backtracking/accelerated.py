"""Numba production primitives 與純 NumPy/Python 參考介面的等價層。

此模組只加速已由解析測試固定的內層迴圈，不在 Numba 路徑重新定義物理。OCM kernel
處理三個支撐節點、前後兩時次的垂向包夾與重心／時間內插；獨立的純量 kernel 則處理
有限水深色散求根與 Stokes 公式、時間步候選比較、四階 Runge-Kutta 最後向量組合／時間
更新，以及把已抽樣常態數轉成擴散位移。這些數值核心只接收已由 Python 驗證的有限
輸入；月份選擇、缺值、品質旗標（QC）、邊界、粒子狀態與亂數抽樣仍由 Python 控制層
決定，不會在 Numba 中重新定義科學規則。

移動海面時，若固定查詢深度超過某一端點最高有限 ``zcor``，OCM kernel 只暫用該端點
最高層的有限物理量作為表層控制體；查詢時刻的海面／海床最終幾何判斷仍由可稽核的
Python 控制層執行，不在 kernel 放寬海面邊界。所有 dispatcher 均設 ``cache=False``，
目前不使用磁碟編譯快取；因此即使設定 ``NUMBA_CACHE_DIR``，本模組也不會默認將快取
寫進正式程式目錄或 ``/home``。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# 後端識別碼同時進入設定雜湊與每條軌跡的 EngineSettings；集中在數值核心模組，
# 避免設定驗證、forcing 取樣與粒子積分各自拼出不同字串。Numba kernel 均關閉磁碟快取，
# 因此編譯結果只留在目前程序記憶體，不會默認寫進正式 SERVER 的程式目錄。
PHYSICS_KERNEL_BACKEND_NUMPY_V1 = "numpy_v1"
PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1 = "numba_cpu_v1"
PHYSICS_KERNEL_BACKENDS = frozenset(
    {PHYSICS_KERNEL_BACKEND_NUMPY_V1, PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1}
)


def validate_physics_kernel_backend(value: object) -> str:
    """驗證物理數值核心後端識別碼，禁止未知值靜默退回參考實作。

    設定可選純 NumPy 參考核心或以 Numba 編譯的 CPU primitive。這個檢查不建立 kernel、
    不讀取 forcing，也不改變粒子或亂數狀態；呼叫端應在進入長時間運算前執行，以免拼錯
    的版本字串被誤當成成功啟用加速。
    """

    if type(value) is not str:
        raise TypeError("physics_kernel_backend 必須是字串")
    if value not in PHYSICS_KERNEL_BACKENDS:
        raise ValueError(f"不支援的 physics_kernel_backend：{value}")
    return value


@njit(cache=False, fastmath=False)
def solve_wave_number_numba_kernel(
    angular_frequency_radps: float,
    water_depth_m: float,
    gravity_mps2: float,
) -> tuple[float, int]:
    """以有界二分法求有限水深波數，回傳波數及明確的數值狀態碼。

    輸入分別是角頻率（rad/s）、正水深（m）及重力加速度（m/s²）。色散殘差
    ``g*k*tanh(k*h)-omega²`` 對正波數嚴格遞增，因此先沿用參考路徑的深水／淺水
    初始上界並逐次加倍建立異號括界，再於最多 100 次二分內將波數誤差縮至
    ``1e-13 + 1e-13*abs(k)``。狀態碼 0 表示成功，1 表示無法建立括界，2 表示
    二分未達容差，3 表示最後殘差超過既有 ``max(1e-10, omega²*1e-10)`` 門檻，
    4 表示直接呼叫時輸入不合法。函式不丟 Python 例外，讓外層保留既有例外契約。
    """

    if (
        not math.isfinite(angular_frequency_radps)
        or angular_frequency_radps <= 0.0
        or not math.isfinite(water_depth_m)
        or water_depth_m <= 0.0
        or not math.isfinite(gravity_mps2)
        or gravity_mps2 <= 0.0
    ):
        return np.nan, 4
    omega2 = angular_frequency_radps**2
    upper = max(
        omega2 / gravity_mps2,
        angular_frequency_radps / math.sqrt(gravity_mps2 * water_depth_m),
    )
    upper = max(upper * 2.0, 1e-12)
    upper_residual = gravity_mps2 * upper * math.tanh(upper * water_depth_m) - omega2
    bracket_expansions = 0
    while upper_residual <= 0.0:
        upper *= 2.0
        bracket_expansions += 1
        # 這個限制同時保留參考路徑的 1e6 m⁻¹ 最大搜尋尺度，並保證 kernel 有限結束。
        if upper > 1e6 or bracket_expansions >= 64:
            return np.nan, 1
        upper_residual = gravity_mps2 * upper * math.tanh(upper * water_depth_m) - omega2

    lower = 0.0
    root = np.nan
    converged = False
    for _ in range(100):
        midpoint = lower + 0.5 * (upper - lower)
        residual = gravity_mps2 * midpoint * math.tanh(midpoint * water_depth_m) - omega2
        if residual == 0.0:
            root = midpoint
            converged = True
            break
        if residual < 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if 0.5 * (upper - lower) <= 1e-13 + 1e-13 * abs(midpoint):
            root = lower + 0.5 * (upper - lower)
            converged = True
            break
    if not converged:
        return np.nan, 2
    final_residual = gravity_mps2 * root * math.tanh(root * water_depth_m) - omega2
    if abs(final_residual) > max(1e-10, omega2 * 1e-10):
        return root, 3
    return root, 0


@njit(cache=False, fastmath=False)
def finite_depth_stokes_numba_kernel(
    significant_wave_height_m: float,
    peak_frequency_hz: float,
    direction_raw_deg: float,
    relative_z_m: float,
    water_depth_m: float,
    gravity_mps2: float,
) -> tuple[float, float, float, float, float, float, float, int]:
    """計算已由 Python 控制層驗證的有限水深 Stokes 速度及診斷量。

    ``relative_z_m`` 是粒子相對海面的垂向位置（海面向上為正，單位 m），水深以正值 m
    傳入；波高 m、頻率 Hz、波向為「來向」角度（正北起順時針）、重力 m/s²。kernel
    只執行既有色散與 Stokes 公式，不判斷粒子是否在水柱內，也不解讀 NWW 品質旗標；
    對 ``kh>20`` 使用和參考實作相同的深水穩定極限。回傳東／北速度、波數、波長、
    ``kh``、陡峭度與相對深度，以及色散 solver 狀態碼。
    """

    omega = 2.0 * math.pi * peak_frequency_hz
    wave_number, status = solve_wave_number_numba_kernel(
        omega, water_depth_m, gravity_mps2
    )
    if status != 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, status
    amplitude = significant_wave_height_m / 2.0
    kh = wave_number * water_depth_m
    if significant_wave_height_m == 0.0:
        speed = 0.0
    elif kh > 20.0:
        speed = amplitude**2 * omega * wave_number * math.exp(2.0 * wave_number * relative_z_m)
    else:
        numerator = (
            amplitude**2
            * omega
            * wave_number
            * math.cosh(2.0 * wave_number * (relative_z_m + water_depth_m))
        )
        speed = numerator / (2.0 * math.sinh(kh) ** 2)
    direction_to_rad = math.radians((direction_raw_deg + 180.0) % 360.0)
    east = math.sin(direction_to_rad)
    north = math.cos(direction_to_rad)
    return (
        speed * east,
        speed * north,
        wave_number,
        2.0 * math.pi / wave_number,
        kh,
        wave_number * amplitude,
        -relative_z_m / water_depth_m,
        0,
    )


@njit(cache=False, fastmath=False)
def rk4_finalize_numba_kernel(
    x_m: float,
    y_m: float,
    z_m: float,
    dt_seconds: float,
    k1_x: float,
    k1_y: float,
    k1_z: float,
    k2_x: float,
    k2_y: float,
    k2_z: float,
    k3_x: float,
    k3_y: float,
    k3_z: float,
    k4_x: float,
    k4_y: float,
    k4_z: float,
) -> tuple[float, float, float]:
    """以純 scalar 運算完成 RK4 最終加權與座標更新，避免多個短陣列暫存。

    輸入位置為公尺、四個速度 stage 為 m/s、時間步長為 signed 秒。加權的括號與舊
    NumPy 表達式 ``position + dt*(k1+2*k2+2*k3+k4)/6`` 保持左結合順序；不使用
    fast-math 或重新排序，避免改變最終浮點捨入與粒子狀態。中間 stage 取樣、失敗例外
    與邊界處理仍留在 Python 控制層。
    """

    weighted_x = ((k1_x + 2.0 * k2_x) + 2.0 * k3_x) + k4_x
    weighted_y = ((k1_y + 2.0 * k2_y) + 2.0 * k3_y) + k4_y
    weighted_z = ((k1_z + 2.0 * k2_z) + 2.0 * k3_z) + k4_z
    return (
        x_m + dt_seconds * weighted_x / 6.0,
        y_m + dt_seconds * weighted_y / 6.0,
        z_m + dt_seconds * weighted_z / 6.0,
    )


@njit(cache=False, fastmath=False)
def rk4_time_age_numba_kernel(
    time_utc_ns: int,
    dt_ns: int,
    age_seconds: float,
    absolute_dt_seconds: float,
) -> tuple[int, float]:
    """以有號 64 位 UTC 奈秒及秒制年齡更新 RK4 狀態時間。

    呼叫端先確認加法落在有號 64 位範圍內，因 Numba 的整數溢位語意不同於 Python
    任意精度整數；超出範圍時由 Python 原路徑完成加法。時間步長的捨入仍由 caller
    使用既有 ``round(dt_seconds*1e9)``，此 kernel 不重新量化物理時間。
    """

    return time_utc_ns + dt_ns, age_seconds + absolute_dt_seconds


@njit(cache=False, fastmath=False)
def choose_time_step_numba_kernel(
    speed_horizontal_mps: float,
    speed_vertical_mps: float,
    horizontal_scale_m: float,
    vertical_scale_m: float,
    kx_m2ps: float,
    ky_m2ps: float,
    kz_m2ps: float,
    dt_min_seconds: float,
    dt_max_seconds: float,
    advective_fraction: float,
    vertical_fraction: float,
    diffusive_fraction: float,
    seconds_to_forcing_boundary: float,
    has_forcing_boundary: bool,
) -> tuple[float, int]:
    """比較既有六種時間步長候選，回傳最小秒數及固定原因代碼。

    尺度單位為 m、速度 m/s、擴散係數 m²/s、步長 s。輸入有限性、正值與擴散係數
    合法性由 Python wrapper 驗證；kernel 只照既有候選建立順序計算，平手時保留較早
    的原因，並保留 dt_min 最小步長夾制。原因代碼依序為 maximum、horizontal_advection、
    vertical_advection、horizontal_diffusion、vertical_diffusion、forcing_boundary、
    minimum_clamp。
    """

    chosen = dt_max_seconds
    reason = 0
    if speed_horizontal_mps > 0.0:
        candidate = advective_fraction * horizontal_scale_m / speed_horizontal_mps
        if candidate < chosen:
            chosen = candidate
            reason = 1
    if speed_vertical_mps > 0.0:
        candidate = vertical_fraction * vertical_scale_m / speed_vertical_mps
        if candidate < chosen:
            chosen = candidate
            reason = 2
    maximum_horizontal_k = max(kx_m2ps, ky_m2ps)
    if maximum_horizontal_k > 0.0:
        candidate = (diffusive_fraction * horizontal_scale_m) ** 2 / (2.0 * maximum_horizontal_k)
        if candidate < chosen:
            chosen = candidate
            reason = 3
    if kz_m2ps > 0.0:
        candidate = (diffusive_fraction * vertical_scale_m) ** 2 / (2.0 * kz_m2ps)
        if candidate < chosen:
            chosen = candidate
            reason = 4
    if has_forcing_boundary and seconds_to_forcing_boundary < chosen:
        chosen = seconds_to_forcing_boundary
        reason = 5
    if chosen < dt_min_seconds:
        return dt_min_seconds, 6
    return chosen, reason


@njit(cache=False, fastmath=False)
def diffusion_displacement_numba_kernel(
    normals: np.ndarray,
    kx_m2ps: float,
    ky_m2ps: float,
    kz_m2ps: float,
    absolute_dt_seconds: float,
    divergence_x_mps: float,
    divergence_y_mps: float,
    divergence_z_mps: float,
    include_divergence_drift: bool,
) -> tuple[float, float, float]:
    """將已由粒子 RNG 產生的三個標準常態數轉為公尺制擴散位移。

    隨機抽樣刻意不在 kernel 內執行，以保留每粒子既有 ``normal(size=3)`` 呼叫次數與
    順序；輸入擴散係數為 m²/s、絕對步長為 s，輸出為東、北、垂向 m。散度漂移只在
    空間擴散樣本明示需要時加入，常數擴散路徑不做額外零值加法，維持舊結果的位元值。
    """

    x = normals[0] * math.sqrt((2.0 * kx_m2ps) * absolute_dt_seconds)
    y = normals[1] * math.sqrt((2.0 * ky_m2ps) * absolute_dt_seconds)
    z = normals[2] * math.sqrt((2.0 * kz_m2ps) * absolute_dt_seconds)
    if include_divergence_drift:
        x = divergence_x_mps * absolute_dt_seconds + x
        y = divergence_y_mps * absolute_dt_seconds + y
        z = divergence_z_mps * absolute_dt_seconds + z
    return x, y, z


def warmup_numba_backend() -> dict[str, object]:
    """以小型合成輸入編譯 CPU kernels，供正式 worker 啟動前明確檢查環境。

    此 warmup 不開啟 OCM／NWW 產品、不呼叫亂數產生器，也不建立大於三個網格節點、
    一層垂向與單一時間列的陣列。所有 dispatcher 使用 ``cache=False``，所以函式不會
    寫出 `.nbc`／`.nbi` 檔案或在 `/home` 建立隱含快取。回傳值只含字串、布林值與小型
    mapping，能直接寫入 JSON 啟動紀錄；任何 kernel 編譯失敗都會拋回原例外，不把失敗
    偽裝成已預熱。
    """

    solve_wave_number_numba_kernel(1.0, 10.0, 9.80665)
    finite_depth_stokes_numba_kernel(1.0, 0.1, 0.0, -1.0, 10.0, 9.80665)
    choose_time_step_numba_kernel(
        0.1,
        0.01,
        100.0,
        10.0,
        1.0,
        1.0,
        0.01,
        0.1,
        100.0,
        0.25,
        0.25,
        0.25,
        0.0,
        False,
    )
    rk4_finalize_numba_kernel(
        0.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    rk4_time_age_numba_kernel(0, -1, 0.0, 1e-9)
    diffusion_displacement_numba_kernel(
        np.zeros(3, dtype=np.float64), 1.0, 1.0, 0.01, 1.0, 0.0, 0.0, 0.0, False
    )
    # 已有 OCM kernel 也是 ``numba_cpu_v1`` 部署鏈可用的部分；三點、一層、單時次 fixture
    # 用來提早驗證 Numba 對唯讀 forcing 陣列型別與支援拓樸的編譯能力。
    hvel = np.zeros((1, 3, 1, 2), dtype=np.float64)
    scalar_field = np.zeros((1, 3, 1), dtype=np.float64)
    nodes = np.arange(3, dtype=np.int64)
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    interpolate_ocm_support_numba(
        hvel,
        scalar_field,
        scalar_field,
        scalar_field,
        0,
        0,
        0.0,
        nodes,
        weights,
        0.0,
        5e-6,
    )
    kernel_names = (
        "solve_wave_number_numba_kernel",
        "finite_depth_stokes_numba_kernel",
        "choose_time_step_numba_kernel",
        "rk4_finalize_numba_kernel",
        "rk4_time_age_numba_kernel",
        "diffusion_displacement_numba_kernel",
        "interpolate_ocm_support_numba",
    )
    dispatchers = (
        solve_wave_number_numba_kernel,
        finite_depth_stokes_numba_kernel,
        choose_time_step_numba_kernel,
        rk4_finalize_numba_kernel,
        rk4_time_age_numba_kernel,
        diffusion_displacement_numba_kernel,
        interpolate_ocm_support_numba,
    )
    return {
        "backend": PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
        "status": "compiled",
        "disk_cache_enabled": False,
        "kernels": {
            name: "compiled" if dispatcher.signatures else "not_compiled"
            for name, dispatcher in zip(kernel_names, dispatchers, strict=True)
        },
    }


@njit(cache=False)
def _vertical_column_kernel(
    hvel: np.ndarray,
    vertical_velocity: np.ndarray,
    zcor: np.ndarray,
    diffusivity: np.ndarray,
    time_index: int,
    node_index: int,
    target_z_m: float,
    vertical_boundary_tolerance_m: float,
) -> tuple[np.ndarray, float, bool]:
    """在單柱搜尋上下支撐層，並實作與 NumPy reference 相同的表層 hold。

    ``target_z_m`` 是固定的 query z；它可以高於某一端點的最高 ``zcor``，因為移動
    海面會讓同一固定 z 在 before／after 端點落在不同的垂向支援區間。此時只要該端點
    的最高有限 ``zcor`` layer 所需的 ``hvel``、``vertical_velocity`` 與 Kz 都有限，便
    回傳該層值與零垂向夾層跨度；真正是否位於 query-time 水柱內由外層共同幾何 gate
    決定。底層沒有對稱 hold，任何必要 layer 缺值仍回傳 ``valid=False``。
    """

    layer_count = zcor.shape[2]
    usable = np.zeros(layer_count, dtype=np.bool_)
    top_index = -1
    top_z = -np.inf
    lower_index = -1
    upper_index = -1
    lower_z = -np.inf
    upper_z = np.inf
    for layer in range(layer_count):
        z_value = zcor[time_index, node_index, layer]
        u_value = hvel[time_index, node_index, layer, 0]
        v_value = hvel[time_index, node_index, layer, 1]
        w_value = vertical_velocity[time_index, node_index, layer]
        k_value = diffusivity[time_index, node_index, layer]
        if np.isfinite(z_value) and z_value > top_z:
            top_z = z_value
            top_index = layer
        usable[layer] = (
            np.isfinite(z_value)
            and np.isfinite(u_value)
            and np.isfinite(v_value)
            and np.isfinite(w_value)
            and np.isfinite(k_value)
        )
    if top_index < 0 or not np.isfinite(target_z_m):
        return np.zeros(4, dtype=np.float64), 0.0, False
    # 端點最高層以上的固定 z 不是任意最近值外插，而是明確代表 OCM 最上層控制體。
    # 最高 zcor 的必要物理量若缺值，不能退回更深層冒充表層支援。
    if target_z_m >= top_z - vertical_boundary_tolerance_m:
        if not usable[top_index]:
            return np.zeros(4, dtype=np.float64), 0.0, False
        result = np.array(
            [
                hvel[time_index, node_index, top_index, 0],
                hvel[time_index, node_index, top_index, 1],
                vertical_velocity[time_index, node_index, top_index],
                diffusivity[time_index, node_index, top_index],
            ],
            dtype=np.float64,
        )
        return result, 0.0, True
    for layer in range(layer_count):
        z_value = zcor[time_index, node_index, layer]
        if not usable[layer]:
            continue
        if z_value <= target_z_m and z_value > lower_z:
            lower_z = z_value
            lower_index = layer
        if z_value >= target_z_m and z_value < upper_z:
            upper_z = z_value
            upper_index = layer
    result = np.zeros(4, dtype=np.float64)
    if lower_index < 0 or upper_index < 0:
        return result, 0.0, False
    lower_values = np.array(
        [
            hvel[time_index, node_index, lower_index, 0],
            hvel[time_index, node_index, lower_index, 1],
            vertical_velocity[time_index, node_index, lower_index],
            diffusivity[time_index, node_index, lower_index],
        ],
        dtype=np.float64,
    )
    span = upper_z - lower_z
    if abs(span) <= np.finfo(np.float64).eps * 16.0:
        return lower_values, 0.0, True
    upper_values = np.array(
        [
            hvel[time_index, node_index, upper_index, 0],
            hvel[time_index, node_index, upper_index, 1],
            vertical_velocity[time_index, node_index, upper_index],
            diffusivity[time_index, node_index, upper_index],
        ],
        dtype=np.float64,
    )
    alpha = (target_z_m - lower_z) / span
    return lower_values + alpha * (upper_values - lower_values), span, True


@njit(cache=False)
def interpolate_ocm_support_numba(
    hvel: np.ndarray,
    vertical_velocity: np.ndarray,
    zcor: np.ndarray,
    diffusivity: np.ndarray,
    before_time_index: int,
    after_time_index: int,
    time_alpha: float,
    node_indices: np.ndarray,
    barycentric_weights: np.ndarray,
    target_z_m: float,
    vertical_boundary_tolerance_m: float,
) -> tuple[np.ndarray, float, bool]:
    """加速兩時次 × 三 node 的 OCM 垂向、水平與時間內插。

    回傳 ``([u,v,w,kz], minimum_positive_bracket_span, valid)``。端點 target z 高於
    最高有效 ``zcor`` 時沿用最高層控制體值；底層不做對稱 hold。任一支撐柱沒有
    雙側包夾或缺少其最高層必要物理量即 ``valid=False``，不重新正規化其餘 node 權重；
    query-time ``eta``／海床的最後範圍檢查由 Python caller 與 NumPy reference 共用，
    因此這個 kernel 不會把端點 hold 誤當成海面以上的有效速度。
    """

    first = np.zeros(4, dtype=np.float64)
    second = np.zeros(4, dtype=np.float64)
    # NumPy reference 會先在每個 endpoint 內忽略零跨度的 exact-layer／surface-hold，
    # 若該 endpoint 沒有正跨度才使用 0.1 m 的保守尺度，最後再取 before／after 較小者。
    # 分開保存兩端，避免 after 全部採 surface hold 時錯誤沿用 before 的大跨度。
    first_minimum_span = np.inf
    second_minimum_span = np.inf
    for support_index in range(3):
        node = int(node_indices[support_index])
        weight = barycentric_weights[support_index]
        values, span, valid = _vertical_column_kernel(
            hvel,
            vertical_velocity,
            zcor,
            diffusivity,
            before_time_index,
            node,
            target_z_m,
            vertical_boundary_tolerance_m,
        )
        if not valid:
            return first, 0.0, False
        first += weight * values
        if span > 0.0 and span < first_minimum_span:
            first_minimum_span = span
        if after_time_index == before_time_index:
            second += weight * values
        else:
            values, span, valid = _vertical_column_kernel(
                hvel,
                vertical_velocity,
                zcor,
                diffusivity,
                after_time_index,
                node,
                target_z_m,
                vertical_boundary_tolerance_m,
            )
            if not valid:
                return first, 0.0, False
            second += weight * values
            if span > 0.0 and span < second_minimum_span:
                second_minimum_span = span
    if after_time_index == before_time_index:
        second_minimum_span = first_minimum_span
    first_vertical_scale = 0.1
    second_vertical_scale = 0.1
    if np.isfinite(first_minimum_span):
        first_vertical_scale = max(first_minimum_span, 0.1)
    if np.isfinite(second_minimum_span):
        second_vertical_scale = max(second_minimum_span, 0.1)
    vertical_scale = min(first_vertical_scale, second_vertical_scale)
    return first + time_alpha * (second - first), vertical_scale, True
