"""BayTrace 原始 CPU 被動平流核心的實驗性直接介面。

本模組只包裝 PI 提供的 BayTrace v4.7.2 原始
``batch_interpolate_velocity`` 與 ``batch_move_rk4``，不重新實作內插或
四階 Runge--Kutta 數學。它的用途是讓比較程式以明確的陣列契約呼叫原始
CPU kernel；這不是 LBT 正式追蹤器，也沒有移植 BayTrace 的完整生命週期。

原 tracker 在 MOVE 後仍會執行 SEARCH、``clamp_z`` 與 INTERPOLATE；本
adapter 僅執行「先以原始內插取得當前速度，再以原始 RK4 被動平流一次」。
它不更新粒子的元素邊界生命週期，測試資料必須保證粒子全程留在同一面內。
對照紀錄必須標註四個 stage 共用同一 ``trat``、原始 ``eta1`` 是 LSC
垂向座標而非 LBT 的 moving-zcor，以及 out-of-face 搜尋失敗會退回 hint。
因此輸出只能作 direct-kernel comparison，不得宣稱與 production tracker
或 LBT 正式結果等價。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import threading
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# BayTrace config.py 的 h0 初值為 0.0；step_native 的公開介面刻意不另增
# h0 參數，故只能忠實採用原始設定初值，而不是在 adapter 內發明科學預設。
BAYTRACE_DEFAULT_H0 = 0.0

_MODULE_NAMES = ("geometry", "vertical", "random", "batch_interp")
_LOAD_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class BayTraceCPUProvenance:
    """記錄本次原始 kernel 載入所使用的來源路徑與 SHA-256。

    ``dependency_paths`` 與 ``dependency_sha256`` 的 key 是受控載入的四個
    BayTrace 原始模組名稱。hash 是讀取當下的原始檔案內容指紋，供比較報告
    追溯 artifact；它不代表該來源已通過 LBT 正式科學驗證。
    """

    source_root: Path
    package_dir: Path
    dependency_paths: Mapping[str, Path]
    dependency_sha256: Mapping[str, str]
    cache_dir: Path

    @property
    def source_sha256(self) -> Mapping[str, str]:
        """回傳原始依賴檔案的 SHA-256 映射，作為較直觀的追溯別名。"""

        return self.dependency_sha256


@dataclass(frozen=True, slots=True)
class BayTraceCPUKernels:
    """保留原始 Numba CPU dispatcher 與其來源證據。"""

    batch_interpolate_velocity: Callable[..., Any]
    batch_move_rk4: Callable[..., Any]
    provenance: BayTraceCPUProvenance


@dataclass(frozen=True, slots=True)
class BayTraceGridArrays:
    """BayTrace 被動平流 kernel 所需的已驗證網格陣列。

    座標 ``grid_x/grid_y`` 與深度 ``dp`` 使用公尺制；``i34`` 表示每個
    元素是三角形或四邊形，``elnode`` 是零起始節點索引，``ic3`` 必須採
    BayTrace 原始的 ``(4, ne)`` 鄰接布局，邊界鄰居以 ``-1`` 表示。LSC
    垂向資料以 ``sigma_lcl(nvrt, nnode)``、``kbp`` 和元素底層 ``kbe``
    表示。所有完整網格掃描與有限值檢查只在建構時執行，避免每一步重掃。

    這個資料容器只保護輸入形狀與索引安全，不執行邊界搜尋、越界修正或
    海陸判斷；那些行為不在本 comparison adapter 的範圍內。
    """

    i34: np.ndarray
    elnode: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    ic3: np.ndarray
    nvrt: int
    kbp: np.ndarray
    kbe: np.ndarray
    sigma_lcl: np.ndarray
    dp: np.ndarray

    def __post_init__(self) -> None:
        """轉成連續陣列並批次檢核索引／方向／面積，保留原首個錯誤語意。

        向量運算僅移除逐元素Python迴圈；三角與四邊形每個必要三角形都檢查，
        不改原始BayTrace核心、有效層、索引界線、有限值或形狀契約。
        """

        i34 = _int_array(self.i34, "i34", 1)
        elnode = _int_array(self.elnode, "elnode", 2)
        grid_x = _float_array(self.grid_x, "grid_x", 1)
        grid_y = _float_array(self.grid_y, "grid_y", 1)
        ic3 = _int_array(self.ic3, "ic3", 2)
        kbp = _int_array(self.kbp, "kbp", 1)
        kbe = _int_array(self.kbe, "kbe", 1)
        sigma_lcl = _float_array(self.sigma_lcl, "sigma_lcl", 2)
        dp = _float_array(self.dp, "dp", 1)
        nvrt = _positive_int(self.nvrt, "nvrt")

        ne = i34.size
        nnode = grid_x.size
        if ne == 0:
            raise ValueError("i34 不得為空")
        if grid_y.shape != grid_x.shape:
            raise ValueError("grid_x 與 grid_y shape 必須一致")
        if elnode.shape != (ne, 4):
            raise ValueError("elnode 必須是 (元素數, 4) 的零起始節點索引")
        if ic3.shape != (4, ne):
            raise ValueError("ic3 必須是 BayTrace 原始布局 (4, 元素數)")
        if kbp.shape != (nnode,):
            raise ValueError("kbp shape 必須等於節點數")
        if kbe.shape != (ne,):
            raise ValueError("kbe shape 必須等於元素數")
        if sigma_lcl.shape != (nvrt, nnode):
            raise ValueError("sigma_lcl shape 必須是 (nvrt, 節點數)")
        if dp.shape != (nnode,):
            raise ValueError("dp shape 必須等於節點數")
        if np.any(~np.isin(i34, (3, 4))):
            raise ValueError("i34 只能包含 3 或 4")
        if np.any(kbp < 0) or np.any(kbp >= nvrt) or np.any(kbe < 0) or np.any(kbe >= nvrt):
            raise ValueError("kbp/kbe 含超出垂向層數的索引")
        if np.any(dp <= 0.0):
            raise ValueError("dp 必須是有限的正水深")
        if np.any((ic3 < -1) | (ic3 >= ne)):
            raise ValueError("ic3 只能使用 -1 或有效元素索引")

        # 一次建立元素×四欄有效索引遮罩，避免每個面各自呼叫np.any。
        # 暫時夾限只供安全計算面積；原索引仍完整檢核，從不送修補索引進kernel。
        used = np.arange(4)[None, :] < i34[:, None]
        bad_indices = np.any(used & ((elnode < 0) | (elnode >= nnode)), axis=1)
        if nnode == 0:
            raise ValueError("elnode[0] 含無效節點索引")
        safe_nodes = np.clip(elnode, 0, nnode - 1)
        n0, n1, n2, n3 = safe_nodes.T
        # 有向面積逐項沿用 _signed_triangle_area 的減、乘、減、除順序，
        # 四邊形第二三角形仍是[0,2,3]，不以總多邊形面積取代其獨立檢查。
        first_area = ((grid_x[n0] - grid_x[n2]) * (grid_y[n1] - grid_y[n2])
                      - (grid_x[n1] - grid_x[n2]) * (grid_y[n0] - grid_y[n2])) / 2.0
        second_area = ((grid_x[n0] - grid_x[n3]) * (grid_y[n2] - grid_y[n3])
                       - (grid_x[n2] - grid_x[n3]) * (grid_y[n0] - grid_y[n3])) / 2.0
        failures = np.select(
            [bad_indices, (i34 == 3) & (elnode[:, 3] != -1),
             first_area <= 1.0e-12, (i34 == 4) & (second_area <= 1.0e-12)],
            [1, 2, 3, 4], default=0)
        bad_elements = np.flatnonzero(failures)
        if bad_elements.size:
            # 必須先找最早元素，再依該元素原檢查順序回報，不能讓後面的
            # 索引錯誤蓋掉前面元素的面積錯誤。
            element = int(bad_elements[0])
            code = failures[element]
            if code == 1:
                raise ValueError(f"elnode[{element}] 含無效節點索引")
            if code == 2:
                raise ValueError("三角形 elnode 的未使用第四欄必須是 -1")
            which = "第一個" if code == 3 else "第二個"
            raise ValueError(f"元素 {element} 的{which}三角形方向或面積無效")

        for name, array in (
            ("grid_x", grid_x),
            ("grid_y", grid_y),
            ("sigma_lcl", sigma_lcl),
            ("dp", dp),
        ):
            _require_finite(array, name)

        object.__setattr__(self, "i34", i34)
        object.__setattr__(self, "elnode", elnode)
        object.__setattr__(self, "grid_x", grid_x)
        object.__setattr__(self, "grid_y", grid_y)
        object.__setattr__(self, "ic3", ic3)
        object.__setattr__(self, "nvrt", nvrt)
        object.__setattr__(self, "kbp", kbp)
        object.__setattr__(self, "kbe", kbe)
        object.__setattr__(self, "sigma_lcl", sigma_lcl)
        object.__setattr__(self, "dp", dp)


@dataclass(frozen=True, slots=True)
class BayTraceFieldPair:
    """兩個時間層的三維原始速度場與第一時間層海面高程。

    ``u1/v1/w1`` 和 ``u2/v2/w2`` 的 shape 都是
    ``(nvrt, nnode)``，單位是 m/s；``eta1`` 是第一時間層節點海面高程，
    shape 是 ``(nnode,)``、單位是 m。BayTrace 原始函式只接收 eta1，因而
    不在 adapter 內偷偷增加 eta2 或移動水面模型。
    """

    u1: np.ndarray
    v1: np.ndarray
    w1: np.ndarray
    eta1: np.ndarray
    u2: np.ndarray
    v2: np.ndarray
    w2: np.ndarray

    def __post_init__(self) -> None:
        """一次性驗證七個場的維度、有限值與連續布局。"""

        arrays = {
            name: _float_array(getattr(self, name), name, 2 if name != "eta1" else 1)
            for name in ("u1", "v1", "w1", "eta1", "u2", "v2", "w2")
        }
        shape = arrays["u1"].shape
        if len(shape) != 2 or shape[0] < 2 or shape[1] == 0:
            raise ValueError("速度場必須是非空的 (nvrt >= 2, 節點數) 陣列")
        for name in ("v1", "w1", "u2", "v2", "w2"):
            if arrays[name].shape != shape:
                raise ValueError(f"{name} shape 必須與 u1 一致")
        if arrays["eta1"].shape != (shape[1],):
            raise ValueError("eta1 shape 必須等於速度場的節點數")
        for name, array in arrays.items():
            _require_finite(array, name)
            object.__setattr__(self, name, array)


@dataclass(slots=True)
class BayTraceParticleArrays:
    """擁有資料的粒子陣列與每步 comparison snapshot。

    ``x/y/z`` 是目前位置（m），``elem`` 是零起始元素 hint，``status=1``
    代表原始 kernel 會處理、``status=0`` 代表停用。``x0/y0/z0`` 保存
    每次 ``step_native`` 呼叫前的步首位置；``x_prev/y_prev/z_prev`` 與
    ``u_prev/v_prev/w_prev`` 保存前一次位置/速度快照，``u/v/w`` 則是本步
    原始內插後供 RK4 使用的速度。所有浮點陣列都是獨立擁有的 float64
    陣列，避免比較程序不知不覺改寫呼叫端的輸入 view。

    這個容器不負責元素 SEARCH 或 boundary lifecycle；``elem`` 在一次
    comparison fixture 中應由呼叫端維持為有效且同面的 hint。
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    elem: np.ndarray
    status: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    x_prev: np.ndarray
    y_prev: np.ndarray
    z_prev: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    u_prev: np.ndarray
    v_prev: np.ndarray
    w_prev: np.ndarray

    def __post_init__(self) -> None:
        """驗證直接建構的容器，並將所有陣列轉為獨立連續儲存。"""

        float_names = (
            "x", "y", "z", "x0", "y0", "z0", "x_prev", "y_prev", "z_prev",
            "u", "v", "w", "u_prev", "v_prev", "w_prev",
        )
        arrays = {name: _owned_float_array(getattr(self, name), name) for name in float_names}
        elem = _owned_int_array(self.elem, "elem", 1)
        status = _owned_int_array(self.status, "status", 1, dtype=np.int32)
        n = arrays["x"].size
        for name, array in arrays.items():
            if array.shape != (n,):
                raise ValueError(f"{name} 必須是一維且長度等於粒子數")
            _require_finite(array, name)
        if elem.shape != (n,) or status.shape != (n,):
            raise ValueError("elem/status shape 必須等於粒子數")
        if np.any(elem < -1):
            raise ValueError("elem 只能是 -1 或非負元素索引")
        if np.any(~np.isin(status, (0, 1))):
            raise ValueError("status 只能是 0（停用）或 1（作用中）")
        if np.any((status == 1) & (elem < 0)):
            raise ValueError("作用中粒子必須有非負 elem")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "elem", elem)
        object.__setattr__(self, "status", status)

    @classmethod
    def from_positions(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        elem: np.ndarray,
        status: np.ndarray | None = None,
    ) -> BayTraceParticleArrays:
        """由步首位置建立可獨立擁有的粒子容器。

        ``x/y`` 是公尺制水平座標，``z`` 是向上為正的公尺深度；三者必須
        同長度且有限。未提供 ``status`` 時，所有粒子預設為原始 BayTrace
        的 active code ``1``。本方法只建立資料，不搜尋元素，也不推導
        ``elem``；元素 hint 必須由呼叫端提供。
        """

        x_array = _owned_float_array(x, "x")
        y_array = _owned_float_array(y, "y")
        z_array = _owned_float_array(z, "z")
        elem_array = _owned_int_array(elem, "elem", 1)
        if x_array.shape != y_array.shape or x_array.shape != z_array.shape:
            raise ValueError("x/y/z shape 必須一致")
        if elem_array.shape != x_array.shape:
            raise ValueError("elem shape 必須等於 x/y/z")
        if status is None:
            status_array = np.ones(x_array.size, dtype=np.int32)
        else:
            status_array = _owned_int_array(status, "status", 1, dtype=np.int32)
        if status_array.shape != x_array.shape:
            raise ValueError("status shape 必須等於 x/y/z")

        zeros = np.zeros(x_array.size, dtype=np.float64)
        return cls(
            x=x_array,
            y=y_array,
            z=z_array,
            elem=elem_array,
            status=status_array,
            x0=x_array.copy(),
            y0=y_array.copy(),
            z0=z_array.copy(),
            x_prev=x_array.copy(),
            y_prev=y_array.copy(),
            z_prev=z_array.copy(),
            u=zeros.copy(),
            v=zeros.copy(),
            w=zeros.copy(),
            u_prev=zeros.copy(),
            v_prev=zeros.copy(),
            w_prev=zeros.copy(),
        )

    @property
    def n_particles(self) -> int:
        """回傳粒子數；此屬性只讀取 shape，不進行網格掃描。"""

        return self.x.size

    # 下面的別名讓呼叫端可對照 BayTrace 原始 p_* 命名，但不另建 view。
    @property
    def px(self) -> np.ndarray:
        """目前 x 位置陣列的 BayTrace 對照別名。"""

        return self.x

    @property
    def py(self) -> np.ndarray:
        """目前 y 位置陣列的 BayTrace 對照別名。"""

        return self.y

    @property
    def pz(self) -> np.ndarray:
        """目前 z 位置陣列的 BayTrace 對照別名。"""

        return self.z


def load_original_cpu(source_root: Path, cache_dir: Path) -> BayTraceCPUKernels:
    """受控載入原始 BayTrace CPU kernel，完全跳過 ``baytrace.__init__``。

    ``source_root`` 必須明確指向含 ``baytrace/`` 的 BayTrace artifact 根目錄，
    或直接指向該 package 目錄；不從工作目錄、Python path 或環境變數猜來源。
    載入只允許 ``geometry.py``、``vertical.py``、``random.py`` 與
    ``batch_interp.py``，並將原始 Numba dispatcher 保留在同一個來源 package
    module 中供其 JIT 全域函式使用。若 ``sys.modules`` 已有不同來源的
    ``baytrace`` 或子模組，直接拒絕，避免同名 package 冒名。

    Numba cache 只在載入期間指向外部 ``cache_dir``，完成後還原環境變數與
    ``numba.config.CACHE_DIR``；原始 source 檔不會被寫入或加入 Git。回傳的
    provenance 會保存四個依賴檔案的絕對路徑與 SHA-256。
    """

    source_root = Path(source_root).expanduser().resolve(strict=True)
    package_dir = _resolve_package_dir(source_root)
    cache_dir = Path(cache_dir).expanduser().resolve()
    if package_dir == cache_dir or package_dir in cache_dir.parents:
        raise ValueError("cache_dir 不得位於原始 BayTrace source 目錄內")
    # 必須先完成 source/cache 邊界檢查，再建立外部 Numba cache 目錄；否則
    # 呼叫者誤把 cache_dir 指到原始 BayTrace 內時，失敗前仍會在 source tree
    # 留下新目錄，破壞原始 artifact 的唯讀與 hash 證據。
    cache_dir.mkdir(parents=True, exist_ok=True)

    dependency_paths = {
        name: package_dir / f"{name}.py" for name in _MODULE_NAMES
    }
    for name, path in dependency_paths.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"缺少或不允許的 BayTrace 原始檔案: {name}.py")
    dependency_sha256 = {
        name: _sha256(path) for name, path in dependency_paths.items()
    }

    with _LOAD_LOCK:
        _check_existing_modules(package_dir, dependency_paths, dependency_sha256)
        previous_env = os.environ.get("NUMBA_CACHE_DIR")
        numba_module = importlib.import_module("numba")
        previous_numba_cache = numba_module.config.CACHE_DIR
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        numba_module.config.CACHE_DIR = str(cache_dir)
        try:
            modules = _load_controlled_modules(package_dir, dependency_paths, dependency_sha256)
            interpolate = getattr(modules["batch_interp"], "batch_interpolate_velocity", None)
            move = getattr(modules["batch_interp"], "batch_move_rk4", None)
            _require_numba_cpu_dispatcher(interpolate, "batch_interpolate_velocity")
            _require_numba_cpu_dispatcher(move, "batch_move_rk4")
        finally:
            numba_module.config.CACHE_DIR = previous_numba_cache
            if previous_env is None:
                os.environ.pop("NUMBA_CACHE_DIR", None)
            else:
                os.environ["NUMBA_CACHE_DIR"] = previous_env

    provenance = BayTraceCPUProvenance(
        source_root=source_root,
        package_dir=package_dir,
        dependency_paths={key: value for key, value in dependency_paths.items()},
        dependency_sha256={key: value for key, value in dependency_sha256.items()},
        cache_dir=cache_dir,
    )
    return BayTraceCPUKernels(interpolate, move, provenance)


def step_native(
    kernels: BayTraceCPUKernels,
    grid: BayTraceGridArrays,
    fields: BayTraceFieldPair,
    particles: BayTraceParticleArrays,
    *,
    dt_seconds: float,
    trat: float,
    ibf: int,
) -> None:
    """執行一次原始 BayTrace passive-advection kernel comparison step。

    每次呼叫先以 ``batch_interpolate_velocity`` 在目前位置取得速度，再以
    ``batch_move_rk4`` 使用同一 ``trat`` 的四個 stage 移動；``dt_seconds``
    必須是正秒數，方向只由 ``ibf`` 的 ``+1/-1`` 決定。步首位置會先保存
    到 ``particles.x0/y0/z0``，速度也會保存到 ``u_prev/v_prev/w_prev``。

    原始 ``_find_element_at`` 找不到候選面時會回傳原 ``elem_hint``，
    ``find_vlevel`` 會把垂向層夾在合法範圍；本函式不替它補做 SEARCH、
    clamp_z、INTERPOLATE，也不更新 ``particles.elem`` 的 boundary lifecycle。
    ``fields.eta1`` 也只代表原始 LSC 垂向座標輸入，不能解讀為 LBT 的
    moving-zcor；對照報告應保留這個差異與 out-of-face hint degradation。
    因此 comparison fixture 必須保證整步留在同一面內，輸出不能解讀為
    完整 tracker 或 production equivalence。
    """

    dt_seconds = _positive_float(dt_seconds, "dt_seconds")
    trat = float(trat)
    if not np.isfinite(trat) or not 0.0 <= trat <= 1.0:
        raise ValueError("trat 必須是 [0, 1] 內的有限時間內插比例")
    if type(ibf) is not int or ibf not in (-1, 1):
        raise ValueError("ibf 必須是整數 +1 或 -1")
    _validate_step_shapes(grid, fields, particles)

    # 只檢查粒子目前狀態；完整網格有限值與索引掃描已在 grid/field
    # constructor 完成，避免每一步重新掃過大型網格。
    _require_finite(particles.x, "particles.x")
    _require_finite(particles.y, "particles.y")
    _require_finite(particles.z, "particles.z")
    if np.any((particles.status == 1) & ((particles.elem < 0) | (particles.elem >= grid.i34.size))):
        raise ValueError("作用中粒子的 elem 超出 grid 元素範圍")

    np.copyto(particles.x0, particles.x)
    np.copyto(particles.y0, particles.y)
    np.copyto(particles.z0, particles.z)
    np.copyto(particles.x_prev, particles.x0)
    np.copyto(particles.y_prev, particles.y0)
    np.copyto(particles.z_prev, particles.z0)
    np.copyto(particles.u_prev, particles.u)
    np.copyto(particles.v_prev, particles.v)
    np.copyto(particles.w_prev, particles.w)

    kernels.batch_interpolate_velocity(
        particles.x,
        particles.y,
        particles.z,
        particles.elem,
        particles.status,
        grid.i34,
        grid.elnode,
        grid.grid_x,
        grid.grid_y,
        grid.nvrt,
        grid.kbp,
        grid.kbe,
        grid.sigma_lcl,
        grid.dp,
        fields.u1,
        fields.v1,
        fields.w1,
        fields.eta1,
        fields.u2,
        fields.v2,
        fields.w2,
        trat,
        BAYTRACE_DEFAULT_H0,
        particles.u,
        particles.v,
        particles.w,
    )
    kernels.batch_move_rk4(
        particles.x,
        particles.y,
        particles.z,
        particles.u,
        particles.v,
        particles.w,
        particles.x_prev,
        particles.y_prev,
        particles.elem,
        particles.status,
        particles.x0,
        particles.y0,
        particles.z0,
        grid.i34,
        grid.elnode,
        grid.grid_x,
        grid.grid_y,
        grid.ic3,
        grid.nvrt,
        grid.kbp,
        grid.kbe,
        grid.sigma_lcl,
        grid.dp,
        fields.u1,
        fields.v1,
        fields.w1,
        fields.eta1,
        fields.u2,
        fields.v2,
        fields.w2,
        ibf,
        dt_seconds,
        trat,
    )


def _resolve_package_dir(source_root: Path) -> Path:
    """解析明確來源根目錄，不從其他位置猜 BayTrace。"""

    package_dir = source_root if source_root.name == "baytrace" else source_root / "baytrace"
    package_dir = package_dir.resolve(strict=True)
    if not package_dir.is_dir() or package_dir.name != "baytrace":
        raise FileNotFoundError("source_root 必須包含 baytrace package 目錄")
    if not (package_dir / "__init__.py").is_file():
        raise FileNotFoundError("BayTrace package 缺少 __init__.py")
    return package_dir


def _sha256(path: Path) -> str:
    """以串流方式計算原始檔案 hash，避免把 source 整檔載入記憶體。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_existing_modules(
    package_dir: Path,
    dependency_paths: Mapping[str, Path],
    dependency_sha256: Mapping[str, str],
) -> None:
    """拒絕不同來源或非本 adapter 建立的同名 BayTrace modules。"""

    package = sys.modules.get("baytrace")
    if "baytrace" in sys.modules:
        if package is None:
            raise ImportError("sys.modules 的 baytrace sentinel 阻擋了明確來源載入")
        if getattr(package, "__lbt_controlled_source__", None) != str(package_dir):
            raise ImportError("sys.modules 已存在非本來源的 baytrace package")
        if getattr(package, "__lbt_dependency_sha256__", None) != dict(dependency_sha256):
            raise ImportError("同名 baytrace package 的來源 hash 與目前 artifact 不一致")
    for name, path in dependency_paths.items():
        full_name = f"baytrace.{name}"
        if full_name not in sys.modules:
            continue
        module = sys.modules[full_name]
        if module is None:
            raise ImportError(f"sys.modules 的 {full_name} sentinel 阻擋了明確來源載入")
        if getattr(module, "__lbt_controlled_source__", None) != str(path):
            raise ImportError(f"sys.modules 已存在非本來源的 baytrace.{name}")
        if Path(module.__file__).resolve() != path.resolve():
            raise ImportError(f"baytrace.{name} __file__ 與明確 source 不一致")


def _load_controlled_modules(
    package_dir: Path,
    dependency_paths: Mapping[str, Path],
    dependency_sha256: Mapping[str, str],
) -> dict[str, types.ModuleType]:
    """建立不執行 ``__init__`` 的 package，並依固定順序載入四個模組。"""

    package = sys.modules.get("baytrace")
    if "baytrace" not in sys.modules:
        package = types.ModuleType("baytrace")
        package.__file__ = str(package_dir / "__init__.py")
        package.__package__ = "baytrace"
        package.__path__ = [str(package_dir)]
        package.__lbt_controlled_source__ = str(package_dir)
        package.__lbt_dependency_sha256__ = dict(dependency_sha256)
        sys.modules["baytrace"] = package

    loaded: dict[str, types.ModuleType] = {}
    for name in _MODULE_NAMES:
        full_name = f"baytrace.{name}"
        module = sys.modules.get(full_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(full_name, dependency_paths[name])
            if spec is None or spec.loader is None:
                raise ImportError(f"無法建立 baytrace.{name} 的明確 source loader")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            module.__lbt_controlled_source__ = str(dependency_paths[name])
            setattr(package, name, module)
        loaded[name] = module
    return loaded


def _require_numba_cpu_dispatcher(function: Any, name: str) -> None:
    """確認原始函式具 Numba dispatcher 的 signatures/py_func 證據。"""

    if function is None or not hasattr(function, "signatures") or not hasattr(function, "py_func"):
        raise TypeError(f"原始 {name} 不是可驗證的 Numba dispatcher")
    target_name = getattr(getattr(function, "targetdescr", None), "target_name", "cpu")
    if target_name != "cpu":
        raise TypeError(f"原始 {name} 不是 CPU dispatcher")


def _validate_step_shapes(
    grid: BayTraceGridArrays,
    fields: BayTraceFieldPair,
    particles: BayTraceParticleArrays,
) -> None:
    """在進入 compiled code 前做不掃描網格內容的 shape/dtype 閘門。"""

    if fields.u1.shape != (grid.nvrt, grid.grid_x.size):
        raise ValueError("fields 與 grid 的 (nvrt, 節點數) 不一致")
    if fields.eta1.shape != (grid.grid_x.size,):
        raise ValueError("fields.eta1 與 grid 節點數不一致")
    n = particles.n_particles
    float_names = (
        "x", "y", "z", "x0", "y0", "z0", "x_prev", "y_prev", "z_prev",
        "u", "v", "w", "u_prev", "v_prev", "w_prev",
    )
    for name in float_names:
        array = getattr(particles, name)
        if array.shape != (n,) or array.dtype != np.float64 or not array.flags.c_contiguous:
            raise ValueError(f"particles.{name} 的 dtype/shape/layout 不符合 native 契約")
    if particles.elem.shape != (n,) or particles.elem.dtype != np.int64:
        raise ValueError("particles.elem 的 dtype/shape 不符合 native 契約")
    if particles.status.shape != (n,) or particles.status.dtype != np.int32:
        raise ValueError("particles.status 的 dtype/shape 不符合 native 契約")


def _signed_triangle_area(x: np.ndarray, y: np.ndarray, nodes: np.ndarray) -> float:
    """計算原始 geometry 使用的正向三角形有向面積。"""

    n0, n1, n2 = nodes
    return float(((x[n0] - x[n2]) * (y[n1] - y[n2]) - (x[n1] - x[n2]) * (y[n0] - y[n2])) / 2.0)


def _positive_int(value: Any, name: str) -> int:
    """驗證非 bool 的正整數 scalar。"""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} 必須是正整數")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    """驗證有限正浮點 scalar。"""

    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必須是有限正數")
    return result


def _float_array(value: Any, name: str, ndim: int) -> np.ndarray:
    """轉成連續 float64 陣列但不保證擁有底層記憶體。"""

    raw = np.asarray(value)
    if raw.ndim != ndim or np.iscomplexobj(raw):
        raise ValueError(f"{name} 維度或數值型別不符合要求")
    try:
        array = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 無法轉成 float64") from exc
    _require_finite(array, name)
    return array


def _int_array(value: Any, name: str, ndim: int) -> np.ndarray:
    """拒絕浮點索引並轉成連續 int64 陣列。"""

    raw = np.asarray(value)
    if raw.ndim != ndim or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} 必須是整數索引陣列且維度為 {ndim}")
    return np.ascontiguousarray(raw, dtype=np.int64)


def _owned_float_array(value: Any, name: str) -> np.ndarray:
    """建立粒子專用、連續且擁有資料的 float64 一維陣列。"""

    array = _float_array(value, name, 1)
    return np.array(array, dtype=np.float64, order="C", copy=True)


def _owned_int_array(
    value: Any,
    name: str,
    ndim: int,
    *,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """建立粒子專用、連續且擁有資料的整數陣列。"""

    array = _int_array(value, name, ndim)
    target_dtype = np.dtype(np.int64) if dtype is None else dtype
    return np.array(array, dtype=target_dtype, order="C", copy=True)


def _require_finite(array: np.ndarray, name: str) -> None:
    """拒絕 NaN/無限值，避免它們進入 Numba native 計算。"""

    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} 含 NaN 或無限值")


__all__ = [
    "BAYTRACE_DEFAULT_H0",
    "BayTraceCPUProvenance",
    "BayTraceCPUKernels",
    "BayTraceGridArrays",
    "BayTraceFieldPair",
    "BayTraceParticleArrays",
    "load_original_cpu",
    "step_native",
]
