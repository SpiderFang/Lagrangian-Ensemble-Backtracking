"""靜態網格橋接：把 LBT 的 ``NativeMesh`` 與單一參考時刻的 OCM 垂向資訊，
整個 run 只轉換一次，變成可直接餵進 BayTrace 原版 ``batch_interp`` njit 批次
kernel（例如 ``batch_interpolate_velocity``）的網格陣列。

為什麼要有這個模組
--------------------
逆向追蹤的規模是 500,000 顆粒子、回溯 30 天；既有 ``baytrace_ocm_adapter.py``
的 ``_build_grid_and_fields`` 會在「每一次流場切換（每個 bracket 換月／換
時刻）」重新掃描整個網格、重新反推 ``sigma_lcl`` 與 ``kbp``。這對逐粒子路徑
是可接受的，但對「每流場只驅動一次原版批次迴圈」的新架構而言，若把這件事
放在更內層重複做，就會變成違反「任何逐粒子或逐步的重複工作都是致命」原則
的隱藏成本來源。本模組把「網格拓撲＋垂向 sigma 反推」這件事獨立出來，讓呼叫
端可以自行決定要在整個 run 只做一次（見下方「科學限制」的誠實警告），還是
在每次流場 bracket 更新時重做一次（呼叫端目前的作法，也是本模組建議的
折衷）。

轉換內容
--------
1. 三角化拓撲（``i34``/``elnode``/``ic3``）：直接沿用 ``NativeMesh`` 已經
   完成一次性三角化與鄰接建構的 ``triangle_nodes``／``triangle_neighbors``，
   純屬座標系統與陣列 layout 轉換，不重新三角化、不重新計算鄰接。這部分是
   真正意義上的「靜態」，不隨時間、流場或海面高度改變。
2. 垂向資訊（``kbp``/``kbe``/``sigma_lcl``）：OCM ``ocm_native`` 只提供每個
   時刻節點各層的實際物理 ``z``（``zcor``）與海面高程（``elev``），並未提供
   BayTrace 原版期待的、per-node 固定分率垂向座標（SCHISM LSC2 的
   ``sigma_lcl``）。本模組因此比照既有 adapter 的作法，從單一參考時刻的
   ``zcor``／``eta`` 反推 ``sigma = (z - eta) / (dp + eta)``，再把結果當成
   BayTrace 原版 kernel 全程使用的固定 ``sigma_lcl``。**這一步只在數學上等
   價於「OCM 的垂向座標本身就是不隨時間變動的固定分率（純 LSC2）」時才精確
   成立**；請務必先閱讀下方「科學限制與量化檢核」一節，不要假設它必然成立。

單位與座標系
------------
輸入輸出座標一律是公尺制（``NativeMesh.node_xy`` 已投影過），深度 ``dp``
是正值水深（向下為正），``sigma_lcl``、``eta`` 沿用 BayTrace／SCHISM 慣例
（sigma 由海床 ``-1`` 到海面 ``0``，``eta`` 正值代表水位高於平均基準面）。

缺值與乾點政策
--------------
任何節點在參考時刻找不到有限 ``zcor``（全乾或缺值）都會讓建構失敗並丟出
``NativeGridBridgeError``，不會用 0 或最近鄰頂替；呼叫端必須先以 OCM
schema 3 的濕遮罩排除這類節點，或改用「濕支援永遠涵蓋追蹤網格」的子網格。
海床以下（``k < kbp``）的層改複製 ``kbp`` 當層的 sigma 值，這與 BayTrace
原版 ``compute_z_levels_lsc`` 對海床以下的 clamp 語意完全一致（該函式本來
就只讀 ``sigma_lcl[kbp]`` 來計算海床以下的 z，並不會真的用到
``sigma_lcl[:kbp]`` 的內容）；填入相同值只是為了滿足陣列「必須全部有限」
的 kernel 輸入契約，不是真正的物理資料，也因此不能解讀成「海床以下也有
獨立觀測」。

科學限制與量化檢核（務必先讀）
--------------------------------
BayTrace 原版的假設是 SCHISM LSC2：每個節點的 ``sigma_lcl`` 是**固定分率**，
不隨海面高度或時間改變，任何時刻的 z 都精確等於
``eta(t) + sigma_lcl * (dp + eta(t))``。若 OCM 上游模式真的是 LSC2 並且
``zcor`` 產品是該模式的直接輸出，則從任一時刻反推出的 ``sigma_lcl`` 理論上
在其他時刻仍然精確成立，唯一誤差來源是儲存精度（例如 ``zcor`` 若以
float32 落地，量級落在毫米級）。

但本專案能看到的上游契約（``vertical_interpolation.py`` 參考實作）刻意
**不假設**這件事：它逐時刻、逐節點以「數值落在哪兩層之間」搜尋內插，
既不假設層索引已排序，也不假設任何固定分率公式；文件也明講「不假設 layer
index 已排序」。這代表 OCM 產品的資料契約本身並未承諾垂向座標是純 LSC2
固定分率——它有可能是含 Z 層的混合座標（SCHISM ``ivcor=2`` SZ 系統，近底
層是固定絕對深度、只有近表層才跟著海面伸縮）、也可能因前處理或數值截斷而
偏離理想 LSC2。這兩種情況都會讓「單一參考時刻反推、之後永遠固定」的
``sigma_lcl`` 在海面變化較大的時刻產生**非顯微量級**的重建誤差，尤其是在
固定 Z 層與伸縮層的轉換帶附近。

因此本模組刻意不假裝這件事沒有風險：``evaluate_static_sigma_error`` 讓呼叫
端拿一段時間、多個節點的真實 ``zcor``／``eta`` 序列，量化「用某一參考時刻
反推出的 sigma_lcl」套用到其他時刻時的最大 z 重建誤差與最大 sigma 誤差，
並回報發生的節點與時刻，供正式 run 前的 QC 判斷取捨；不得只因為測試資料
剛好通過就放寬容差或跳過這一步。

若在真實 OCM 產品上跑出的誤差不可接受，建議的替代方案是「每次流場 bracket
更新時重算一次 sigma_lcl／kbp」（就是既有 ``baytrace_ocm_adapter.py`` 現在
的作法），而不是整個 30 天 run 只反推一次。這個折衷的效能代價遠低於逐粒子
或逐步重算：bracket 更新的頻率是「每次前後兩個 OCM 時間格點切換」，對 30
天回溯、通常以小時級解析度儲存的 ``ocm_native`` 產品而言大約是數十到數百次
量級，每次重算只需對 ``O(nnode × nvrt)`` 的陣列做一次向量化 NumPy 運算
（如 ``_build_grid_and_fields`` 目前的做法），相較 500,000 顆粒子在 30 天內
的總步數（數量級是粒子數 × 步數，遠大於 bracket 次數），這個成本可以忽略；
真正必須避免的是「每個粒子」或「每一個 RK4 sub-step」重算，而不是「每次流
場更新」重算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baytrace_native_cpu import BayTraceGridArrays
from ..mesh import NativeMesh

# 與既有 ``baytrace_ocm_adapter.py`` 反推 sigma 時使用的容差同量級，避免同一
# 個科學判斷（sigma 落在 [-1, 0] 附近、隨層遞增）在不同模組使用不一致的標準。
_DEFAULT_SIGMA_TOLERANCE = 1.0e-6


class NativeGridBridgeError(ValueError):
    """靜態網格橋接的輸入不符資料契約或無法安全反推垂向座標時提出。

    刻意獨立於 ``ValueError`` 命名，讓呼叫端可以精確攔截「橋接失敗」而不
    誤吞其他來源的 ``ValueError``；仍繼承 ``ValueError`` 以相容既有防呆。
    """


@dataclass(frozen=True, slots=True)
class NativeGridArrays:
    """可直接交給 BayTrace 原版 ``batch_interp`` kernel 的靜態網格陣列。

    欄位語意與 shape 完全比照 BayTrace 原版 kernel 實際索引方式（見
    ``baytrace_native_cpu.BayTraceGridArrays`` 與
    ``baytrace-root/baytrace/batch_interp.py`` 的逐元素索引），而不是只照
    變數名稱猜測：

    - ``i34`` ``(ne,)``：每個元素是三角形（3）或四邊形（4）。本橋接只從
      ``NativeMesh`` 既有的三角化結果建立，因此永遠是 3。
    - ``elnode`` ``(ne, 4)``：零起始節點索引；三角形的第 4 欄固定為 ``-1``，
      對應 kernel 用 ``i34`` 決定實際讀取幾欄，不會誤讀 ``-1``。
    - ``grid_x``/``grid_y`` ``(nnode,)``：節點座標，公尺制（``NativeMesh``
      的投影座標），不是經緯度。
    - ``ic3`` ``(4, ne)``：BayTrace 原版慣用的「元素在前、依對邊局部索引在
      前」布局是反過來的——kernel 實際是以 ``ic3[local_edge, elem]`` 索引
      （見 ``batch_interp.batch_check_boundaries`` 與
      ``grid.py.build_side_structure``），所以本類別採 ``(4, ne)`` 而非
      ``(ne, 4)``。局部邊 ``j`` 對應「跨過與局部頂點 ``j`` 相對的那條邊」
      的鄰居元素；沒有鄰居（外邊界或非流形共邊）一律填 ``-1``。
    - ``nvrt``：垂向層數，純量。
    - ``kbp`` ``(nnode,)``：每個節點第一個「有效」垂向層索引（往上數）；
      索引以下的層代表海床以下或缺值，clamp 到 ``kbp`` 當層，不是零值。
    - ``kbe`` ``(ne,)``：元素層級底層 ``= min(kbp[三個節點])``，比照 BayTrace
      原版 ``grid.py`` 的 ``kbe`` 定義，讓 kernel 垂向搜尋不會讀到任何節點
      ``kbp`` 以下的資料。
    - ``sigma_lcl`` ``(nvrt, nnode)``：反推自參考時刻的固定分率垂向座標
      （見模組 docstring 的科學限制）。
    - ``dp`` ``(nnode,)``：節點正值水深，公尺制。

    所有陣列在建構後都會是 C-contiguous 的 ``int64``／``float64``；驗證邏輯
    直接重用 ``baytrace_native_cpu.BayTraceGridArrays`` 已經過測試的索引、
    方向與有限值檢查，避免在這裡重寫一份容易與原版行為分歧的複本。
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
        """委由 ``BayTraceGridArrays`` 完成索引界線、方向、面積與有限值檢查。

        驗證失敗時直接讓底層例外訊息透傳，因為那些訊息已經指出是哪個元素
        或哪個節點的哪一種問題（例如「元素 12 的第一個三角形方向或面積
        無效」），比在這裡重新包裝一次更利於除錯。
        """

        validated = BayTraceGridArrays(
            i34=self.i34,
            elnode=self.elnode,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            ic3=self.ic3,
            nvrt=self.nvrt,
            kbp=self.kbp,
            kbe=self.kbe,
            sigma_lcl=self.sigma_lcl,
            dp=self.dp,
        )
        for name in (
            "i34", "elnode", "grid_x", "grid_y", "ic3",
            "nvrt", "kbp", "kbe", "sigma_lcl", "dp",
        ):
            object.__setattr__(self, name, getattr(validated, name))

    def as_baytrace_grid_arrays(self) -> BayTraceGridArrays:
        """回傳一個獨立的 ``BayTraceGridArrays``，供直接呼叫原版 kernel 使用。

        ``NativeGridArrays`` 與 ``BayTraceGridArrays`` 欄位完全一致（本類別
        建構時已經跑過同一套驗證），這裡只是重新包一次以取得型別正確的
        物件；呼叫端不需要（也不應該）繞過 ``native_runner`` 直接建構
        ``BayTraceGridArrays``，以免將來橋接邏輯改變時分成兩份要同步維護。
        """

        return BayTraceGridArrays(
            i34=self.i34,
            elnode=self.elnode,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            ic3=self.ic3,
            nvrt=self.nvrt,
            kbp=self.kbp,
            kbe=self.kbe,
            sigma_lcl=self.sigma_lcl,
            dp=self.dp,
        )


def build_native_grid_arrays(
    mesh: NativeMesh,
    *,
    zcor_reference: np.ndarray,
    eta_reference: np.ndarray,
    sigma_tolerance: float = _DEFAULT_SIGMA_TOLERANCE,
) -> NativeGridArrays:
    """把 ``NativeMesh`` 與單一參考時刻的 OCM 垂向資訊轉成 ``NativeGridArrays``。

    參數
    ------
    mesh:
        已完成公尺制投影與三角化的 ``NativeMesh``；本函式只讀取其
        ``node_xy``、``source_depth_m``、``triangle_nodes``、
        ``triangle_neighbors``，不重新三角化、不修改傳入物件。
    zcor_reference:
        shape ``(節點數, 垂向層數)`` 的參考時刻 ``zcor``（公尺），節點順序
        必須與 ``mesh.node_xy`` 一致——也就是 OCM ``ocm_native`` 對單一時刻
        的原生節點/層排列（``zcor.npy`` 的 ``(time, node, layer)`` 拿掉
        ``time`` 軸後的樣子），不是 BayTrace 的 ``(nvrt, nnode)`` 轉置。
    eta_reference:
        shape ``(節點數,)`` 的同一參考時刻海面高程（公尺）。
    sigma_tolerance:
        判定反推 sigma 是否落在 ``[-1, 0]`` 合理範圍、是否沿層遞增的容差；
        預設與既有 ``baytrace_ocm_adapter.py`` 反推 sigma 時使用的容差同
        量級，避免同一個判斷標準在不同模組各自漂移。

    回傳
    ------
    ``NativeGridArrays``，可直接呼叫 ``.as_baytrace_grid_arrays()`` 餵給
    BayTrace 原版 ``batch_interp`` kernel。

    缺值與乾點
    ----------
    任何節點若在參考時刻沒有任何有限 ``zcor`` 層、水深加海面高程非正、或
    反推的 sigma 超出合理範圍／非遞增，都會讓本函式丟出
    ``NativeGridBridgeError`` 並指出第一個出問題的節點，不會用 0 或最近鄰
    頂替；呼叫端必須先用濕遮罩排除這些節點，或改選其他參考時刻。

    重要限制
    --------
    本函式只用「一個」參考時刻反推 ``sigma_lcl``，之後視為整個 run 固定不
    變。這是否精確等價於原始 OCM 垂向座標，取決於上游模式是否真的是固定
    分率的 LSC2；本函式本身**不會**替你驗證這件事是否成立，請另外呼叫
    ``evaluate_static_sigma_error`` 對照多個時刻的真實 ``zcor``／``eta``，
    量化「固定 sigma」在其他時刻的重建誤差後再決定是否採用（詳見模組
    docstring 的「科學限制與量化檢核」一節）。
    """

    node_xy = np.asarray(mesh.node_xy, dtype=np.float64)
    dp = np.asarray(mesh.source_depth_m, dtype=np.float64)
    triangle_nodes = np.asarray(mesh.triangle_nodes, dtype=np.int64)
    neighbors = np.asarray(mesh.triangle_neighbors, dtype=np.int64)
    nnode = int(node_xy.shape[0])
    ne = int(triangle_nodes.shape[0])

    if node_xy.ndim != 2 or node_xy.shape != (nnode, 2) or dp.shape != (nnode,):
        raise NativeGridBridgeError("mesh 節點座標／水深 shape 不一致")
    if (
        triangle_nodes.ndim != 2
        or triangle_nodes.shape[1] != 3
        or neighbors.shape != triangle_nodes.shape
    ):
        raise NativeGridBridgeError("mesh 三角化拓撲 shape 不符 (ne, 3)")
    if np.any(triangle_nodes < 0) or np.any(triangle_nodes >= nnode):
        raise NativeGridBridgeError("mesh.triangle_nodes 含超出節點數的索引")

    zcor = np.asarray(zcor_reference, dtype=np.float64)
    eta = np.asarray(eta_reference, dtype=np.float64)
    if zcor.ndim != 2 or zcor.shape[0] != nnode:
        raise NativeGridBridgeError("zcor_reference 必須是 (節點數, 垂向層數) 且節點數與 mesh 一致")
    nvrt = int(zcor.shape[1])
    if nvrt < 2:
        raise NativeGridBridgeError("zcor_reference 垂向層數必須至少為 2，才有上下夾層可內插")
    if eta.shape != (nnode,):
        raise NativeGridBridgeError("eta_reference 必須是 (節點數,) 且與 mesh 節點數一致")

    if np.any(~np.isfinite(dp)) or np.any(dp <= 0.0):
        raise NativeGridBridgeError(
            "source_depth_m 必須是有限正水深；陸地或缺值節點須在前處理排除，不能以零值頂替"
        )
    if np.any(~np.isfinite(eta)):
        raise NativeGridBridgeError("eta_reference 必須全部有限；乾點或缺值不可混入固定分率反推")

    finite_zcor = np.isfinite(zcor)
    has_support = finite_zcor.any(axis=1)
    if not np.all(has_support):
        missing = np.flatnonzero(~has_support)
        raise NativeGridBridgeError(
            f"{missing.size} 個節點在參考時刻沒有任何有限 zcor 層（可能全乾或缺值），"
            f"第一個是節點 {int(missing[0])}；靜態網格橋接無法為其建立 sigma_lcl，"
            "請先用濕遮罩排除或改選其他參考時刻"
        )
    kbp = np.argmax(finite_zcor, axis=1).astype(np.int64)

    layers = np.arange(nvrt)[None, :]
    used = layers >= kbp[:, None]
    if np.any(used & ~finite_zcor):
        first_bad = int(np.flatnonzero(np.any(used & ~finite_zcor, axis=1))[0])
        raise NativeGridBridgeError(
            f"節點 {first_bad} 在 kbp 以上出現缺值層，違反「有限層之間才可內插、"
            "不得外插」的垂向契約，無法反推固定 sigma"
        )

    depth_plus_eta = dp + eta
    if np.any(depth_plus_eta <= 0.0):
        first_bad = int(np.flatnonzero(depth_plus_eta <= 0.0)[0])
        raise NativeGridBridgeError(
            f"節點 {first_bad} 在參考時刻的水深加海面高程非正（可能為乾點），"
            "無法反推 sigma；請改選濕的參考時刻或先排除該節點"
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_actual = (zcor - eta[:, None]) / depth_plus_eta[:, None]
    out_of_range = used & (
        ~np.isfinite(sigma_actual)
        | (sigma_actual < -1.0 - sigma_tolerance)
        | (sigma_actual > sigma_tolerance)
    )
    if np.any(out_of_range):
        first_bad = int(np.flatnonzero(np.any(out_of_range, axis=1))[0])
        raise NativeGridBridgeError(
            f"節點 {first_bad} 反推的 sigma 超出 [-1, 0]（含容差）合理範圍，"
            "參考時刻資料可能不符合 LSC 固定分率假設，不應直接當成靜態 sigma_lcl"
        )
    pair_used = used[:, :-1]
    non_monotonic = pair_used & (np.diff(sigma_actual, axis=1) < -sigma_tolerance)
    if np.any(non_monotonic):
        first_bad = int(np.flatnonzero(np.any(non_monotonic, axis=1))[0])
        raise NativeGridBridgeError(
            f"節點 {first_bad} 反推的 sigma 未沿垂向層遞增，"
            "參考時刻資料可能不是單調的固定分率垂向網格"
        )

    # 海床以下（k < kbp）沒有物理意義的 sigma；比照 BayTrace 原版
    # compute_z_levels_lsc 對海床以下的 clamp 語意，複製 kbp 當層的值，
    # 只是為了讓陣列滿足「必須全部有限」的 kernel 輸入契約。
    clamped_layers = np.maximum(layers, kbp[:, None])
    sigma_lcl = np.ascontiguousarray(
        np.take_along_axis(sigma_actual, clamped_layers, axis=1).T, dtype=np.float64
    )

    kbe = np.min(kbp[triangle_nodes], axis=1).astype(np.int64)

    i34 = np.full(ne, 3, dtype=np.int64)
    elnode = np.full((ne, 4), -1, dtype=np.int64)
    elnode[:, :3] = triangle_nodes
    ic3 = np.full((4, ne), -1, dtype=np.int64)
    ic3[:3, :] = neighbors.T

    return NativeGridArrays(
        i34=i34,
        elnode=elnode,
        grid_x=np.ascontiguousarray(node_xy[:, 0], dtype=np.float64),
        grid_y=np.ascontiguousarray(node_xy[:, 1], dtype=np.float64),
        ic3=ic3,
        nvrt=nvrt,
        kbp=kbp,
        kbe=kbe,
        sigma_lcl=sigma_lcl,
        dp=dp,
    )


@dataclass(frozen=True, slots=True)
class SigmaStaticizationReport:
    """量化「單一參考時刻反推的固定 sigma_lcl」相對其他時刻真實垂向座標的誤差。

    這是一份 QC 診斷結果，不是科學結論；判斷「這個誤差是否可接受」必須由
    呼叫端依照本次 run 的精度需求（例如與擴散、RK4 積分誤差的量級比較）
    決定，本報告只負責誠實量測，不自行放寬或美化。

    欄位
    ------
    max_abs_z_error_m:
        在所有可評估的 (時間, 節點, 垂向層) 組合中，「用固定 sigma_lcl 重建
        的 z」與「該時刻真實 zcor」之間的最大絕對誤差，單位公尺。
    max_abs_sigma_error:
        同樣組合中，「固定 sigma_lcl」與「該時刻由真實 zcor／eta 反推出的
        sigma」之間的最大絕對誤差（sigma 本身無單位）。
    worst_time_index / worst_node_index / worst_layer_index:
        ``max_abs_z_error_m`` 發生時對應的時間、節點、垂向層索引，供進一步
        追查是哪一種情境（例如特定淺水節點或特定極端潮位時刻）造成偏差。
    evaluated_point_count:
        實際參與比較的 (時間, 節點, 垂向層) 組合數。
    skipped_point_count:
        因缺值、乾點（水深加海面高程非正）或海床以下（低於該節點 ``kbp``）
        而未參與比較的組合數；這些位置不是「誤差為零」，而是「本來就沒有
        可比較的觀測」，因此明確排除在統計之外，不得誤讀成通過檢核。
    """

    max_abs_z_error_m: float
    max_abs_sigma_error: float
    worst_time_index: int
    worst_node_index: int
    worst_layer_index: int
    evaluated_point_count: int
    skipped_point_count: int


def evaluate_static_sigma_error(
    grid: NativeGridArrays,
    *,
    zcor_series: np.ndarray,
    eta_series: np.ndarray,
) -> SigmaStaticizationReport:
    """量化 ``grid.sigma_lcl``（固定分率）套用到一段時間序列時的重建誤差。

    參數
    ------
    grid:
        由 ``build_native_grid_arrays`` 建立的靜態網格，內含目前打算整個
        run 沿用的 ``sigma_lcl``、``kbp``、``dp``。
    zcor_series:
        shape ``(時間, 節點數, 垂向層數)`` 的真實 ``zcor``（公尺），節點與
        層數必須與 ``grid`` 一致；時間軸可以是任意想檢查的時刻集合，例如
        整個 30 天回溯視窗涵蓋的所有 OCM 時間格點。
    eta_series:
        shape ``(時間, 節點數)`` 的對應真實海面高程（公尺）。

    回傳
    ------
    ``SigmaStaticizationReport``，見其 docstring。

    比較方式
    --------
    對每個 (時間, 節點, 層) 組合，只在滿足以下條件時才計入比較：
    層索引不低於該節點 ``kbp``（海床以下沒有物理意義）、當時 ``zcor``／
    ``eta`` 皆有限、水深加海面高程為正（不是乾點）。用
    ``z_hat = eta + sigma_lcl * (dp + eta)`` 重建 z 並與真實 ``zcor`` 比較；
    同時也直接比較反推的 sigma 本身，避免「z 誤差恰巧在深水區被稀釋」掩蓋
    sigma 本身已經偏移的情況。

    這個函式刻意不接受任何「容差」參數自動判定通過與否——是否可接受必須
    由呼叫端對照本次 run 的科學需求自行判斷，不能靠這裡的預設值悄悄放行。
    """

    zcor = np.asarray(zcor_series, dtype=np.float64)
    eta = np.asarray(eta_series, dtype=np.float64)
    nnode = int(grid.grid_x.shape[0])
    nvrt = int(grid.nvrt)
    if zcor.ndim != 3 or zcor.shape[1:] != (nnode, nvrt):
        raise NativeGridBridgeError(
            "zcor_series 必須是 (時間, 節點數, 垂向層數)，且節點數／層數須與 grid 一致"
        )
    ntime = int(zcor.shape[0])
    if eta.shape != (ntime, nnode):
        raise NativeGridBridgeError("eta_series 必須是 (時間, 節點數) 且時間軸與 zcor_series 一致")
    if ntime == 0:
        raise NativeGridBridgeError("zcor_series 至少要有一個時刻才能量化誤差")

    dp = grid.dp[None, :, None]
    eta_b = eta[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_plus_eta = dp + eta_b
        sigma_lcl_full = np.broadcast_to(grid.sigma_lcl.T[None, :, :], zcor.shape)
        z_reconstructed = eta_b + sigma_lcl_full * depth_plus_eta
        sigma_actual = (zcor - eta_b) / depth_plus_eta

    layers = np.arange(nvrt)[None, None, :]
    used = layers >= grid.kbp[None, :, None]
    evaluable = (
        used
        & np.isfinite(zcor)
        & np.isfinite(eta_b)
        & np.isfinite(depth_plus_eta)
        & (depth_plus_eta > 0.0)
    )
    evaluated_point_count = int(np.count_nonzero(evaluable))
    if evaluated_point_count == 0:
        raise NativeGridBridgeError(
            "沒有任何 (時間, 節點, 層) 組合可用來量化 sigma 靜態化誤差；"
            "請確認 zcor_series/eta_series 是否全為缺值或乾點"
        )

    abs_z_error = np.where(evaluable, np.abs(z_reconstructed - zcor), np.nan)
    abs_sigma_error = np.where(evaluable, np.abs(sigma_lcl_full - sigma_actual), np.nan)

    worst_flat = int(np.nanargmax(abs_z_error))
    worst_time, worst_node, worst_layer = np.unravel_index(worst_flat, abs_z_error.shape)

    return SigmaStaticizationReport(
        max_abs_z_error_m=float(np.nanmax(abs_z_error)),
        max_abs_sigma_error=float(np.nanmax(abs_sigma_error)),
        worst_time_index=int(worst_time),
        worst_node_index=int(worst_node),
        worst_layer_index=int(worst_layer),
        evaluated_point_count=evaluated_point_count,
        skipped_point_count=int(zcor.size - evaluated_point_count),
    )


__all__ = [
    "NativeGridArrays",
    "NativeGridBridgeError",
    "SigmaStaticizationReport",
    "build_native_grid_arrays",
    "evaluate_static_sigma_error",
]
