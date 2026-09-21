"""把 ``ocm_native`` 逐月 memmap 陣列，轉成 BayTrace 原版 njit 批次主迴圈要的
雙時次流場欄位（``u_field1/v_field1/w_field1/eta_field1`` 與
``u_field2/v_field2/w_field2``）。

為什麼要有這個模組
--------------------
BayTrace 原版批次主迴圈（``tracker.py`` 的 ``_read_into_buffer``／
``read_schism_output_sliding``）原本直接讀 ``schout_*.nc``；本專案的資料契約
（見 ``AGENTS.md``）規定正式流程只能讀已驗收的 ``ocm_native`` 產品，不得回讀
raw NetCDF。本模組因此扮演「換掉讀檔層、保留原版 kernel 呼叫介面」的橋接角色：
呼叫端仍舊只需要 ``(nvrt, nnode)`` 的 ``u/v/w`` 與 ``(nnode,)`` 的 ``eta``
陣列，本模組負責把 ``ocm_native`` 的 ``(node, layer, component)`` 陣列轉成
這個形狀，並處理跨月、缺口與逆向追蹤的時間語意。

效能設計：以「流場時刻」為快取單位
------------------------------------
500,000 顆粒子、30 天回溯的規模下，任何「每粒子」層級的場解析都會讓 I/O 與
轉置成本乘上粒子數而失控。因此本模組把「同一個 UTC 小時的欄位」快取成單一
不可變快照（見 ``_HourSnapshot``），同一小時內所有粒子共用同一份陣列；往前
（或依 ``ibf`` 往「追蹤方向」）滑動一小時時，原本的 field2 快照直接變成新的
field1，不重新讀取、不重新轉置——見 ``NativeForcingBridgeReader._get_snapshot``
的小型 LRU（大小 2）與 ``read_counts`` 讀取次數統計。

軸轉置與 C-contiguous 契約
----------------------------
``hvel[t, :, :, 0]`` 是 ``(node, layer)``；BayTrace 原版 njit kernel（見
``baytrace/batch_interp.py``）要的是 ``(layer, node)`` 且必須是 C-contiguous
（否則 Numba 走非最佳化路徑，甚至某些版本會直接要求連續記憶體配置）。轉置與
``np.ascontiguousarray`` 只在快照第一次建立時做一次，快取命中時完全不重算。

缺值與乾點分開保存，不得無聲補零
------------------------------------
每個快照除了 ``u/v/w/eta/diffusivity`` 之外，另外保留：

* ``missing_mask``：``(nvrt, nnode)`` 布林，只要 ``hvel``／``zcor``／
  ``vertical_velocity``／``diffusivity`` 任一者在該 (層, 節點) 非有限，就記
  為 ``True``。這是「資料缺值」的唯一權威判定，本模組**不會**把這些位置填成
  0 或任何替代值——``u/v/w/eta/diffusivity`` 保留原生 ``NaN``。停用 OCM
  擴散係數讀取（見下方「``load_diffusivity`` 開關」）時，``diffusivity`` 不
  再是這個判定的一部分——缺值只由 ``hvel``／``zcor``／``vertical_velocity``
  決定。

``load_diffusivity`` 開關：本專案逆向驅動層不讀 OCM 擴散係數
-------------------------------------------------------------------
BayTrace 原版逆向驅動層（``native_runner.backward_driver``／
``native_runner.hour_major``）的水平擴散係數（Kh）一律由
``backward_driver.compute_smagorinsky_kh_field`` 以 Smagorinsky 方法（工作書
式 (10)）從本次運算合成後的流速直接算出，再交給未修改的
``handles.batch_interpolate_diffusivity`` 取樣；驅動層從未讀取、也從未傳入
``HourlyFieldPair.diffusivity_field1``／``diffusivity_field2``——這兩個欄位
只在本模組的資料契約檢查（例如缺值遮罩）中使用。這件事已經以程式碼搜尋
確認（``diffusivity_field1``／``diffusivity_field2`` 只出現在本模組與測試
fixture，未出現在 ``backward_driver.py``／``hour_major.py`` 的任何 kernel
呼叫參數中）。

由於正式運算伺服器讀 NAS 頻寬有限（約 11.7 MB/s），已驗收 OCM 產品正在被
一次性複製到本機，且**只複製逆向運算需要的變數**，本機月份副本刻意不含
``diffusivity.npy``。``NativeForcingBridgeReader`` 因此提供選用參數
``load_diffusivity``（預設 ``True``，與既有行為完全相同：仍會開啟
``diffusivity.npy``，缺檔一樣報錯）。呼叫端明確傳入
``load_diffusivity=False`` 時：

* 不會嘗試開啟任何月份的 ``diffusivity.npy``（該檔案存在與否都不影響）。
* ``_HourSnapshot.diffusivity``／``HourlyFieldPair.diffusivity_field1``／
  ``diffusivity_field2`` 一律為 ``None``。
* 缺值遮罩不再把 diffusivity 的有限性計入判定。
* 跨月合併時，重建 patch 的 diffusivity 列也一併忽略（不比對 shape、不讀取
  資料），即使 patch 月份本身仍保留完整的 diffusivity 陣列。

這個模式**只適用於本身不使用 OCM 擴散係數的驅動層**（即本專案的逆向
``native_runner``）；若有其他呼叫端仍需要 OCM 原生擴散係數（例如舊版
``forcing.OCMNativeMonth`` 為其他管線提供的取樣路徑），不得對它使用
``load_diffusivity=False``。
* ``dry_face_flag``：直接透傳 ``wetdry_elem`` 該小時的原始列（``0=wet,
  1=dry``，重建 patch 另有 ``NaN=not_persistent``），不做任何節點投影或再
  解讀。

BayTrace 原版真正決定「哪個節點乾涸該歸零」的邏輯是 ``tracker.py`` 的
``_compute_dry_flags``／``_zero_dry_velocities``，只需要 ``dp``（來自
``native_runner.grid_bridge``）、我們提供的 ``eta`` 與 ``h0``；因此本模組刻意
**不**自行重新發明一套乾濕判定去清洗 ``u/v/w``——避免兩套判定互相打架，也
避免在還不知道真正 ``dp``／``h0`` 之前，錯誤地把「其實是缺值」的 ``NaN``
偷偷清成看似正常的 0。作為替代，本模組提供 ``HourlyFieldPair.
raise_if_unresolved_gap``：呼叫端算出 BayTrace 自己的節點乾濕狀態後，可以用
它確認「濕節點上真的沒有殘留缺值」，缺值就必須 fail-closed，不能被默默吃掉。
另提供 ``project_face_dryness_to_nodes`` 作為選用的交叉檢查（用 OCM 自己的
``wetdry_elem`` 概略投影到節點），純供 QC 參考，不涉入主要資料路徑。

逆向追蹤的 field1/field2 指派與 ``trat`` 推導（容易出錯的地方）
-------------------------------------------------------------------
BayTrace 原版 kernel 的時間內插固定是
``value = field1 * (1 - trat) + field2 * trat``（見
``batch_interp.batch_interpolate_velocity`` 內部逐行運算），也就是
``trat=0`` 對應 field1、``trat=1`` 對應 field2，且 field1 永遠代表「這次
substep 插值的起點」。``tracker.py`` 主迴圈第 1519-1523 行的
``trat = idt/ndeltp``（``ibf=1``）與 ``trat = (idt-1)/ndeltp``（``ibf=-1``）
只是「substep 計數器」版本的同一件事：兩者都保證 trat 從 0 開始，隨 substep
推進逼近（或如 ``ibf=-1`` 時逼近但不含 1 的） 1；它們**不會**告訴你哪個真實
UTC 小時對應 field1、哪個對應 field2，那件事完全由「這次 hourly slide 讀到
的是哪個方向」決定，而這正是本模組要接手的部分（原本由 ``schout_*.nc``
檔案本身有沒有先反向排序決定）。

推導如下。設兩個相鄰、涵蓋查詢時刻 ``target`` 的 canonical 小時
``lo < hi``（真實時間序，非追蹤方向），定義
``natural_alpha = (target - lo) / (hi - lo) ∈ [0, 1]``（0 在 lo、1 在 hi，
與 ``forcing.py`` 既有 ``_time_bracket`` 的 ``alpha`` 同義）：

* ``ibf == 1``（正向）：時間前進，這次 substep 的「起點」自然是較早的
  ``lo``，「終點」是較晚的 ``hi``。所以 ``field1 = lo``、``field2 = hi``，
  ``trat = natural_alpha``——與 ``tracker.py`` 的 ``idt/ndeltp`` 方向一致
  （trat 隨真實時間前進而增加）。
* ``ibf == -1``（逆向）：時間倒退，這次 substep 的「起點」是較晚的 ``hi``
  （因為我們是從晚往早走），「終點」是較早的 ``lo``。所以
  ``field1 = hi``、``field2 = lo``，``trat = 1 - natural_alpha``——當
  ``target = hi``（逆向剛開始這一小時）時 ``trat = 0``（純 field1）；當
  ``target = lo``（逆向走到這一小時尾端）時 ``trat = 1``（純 field2）。這與
  ``tracker.py`` 的 ``(idt-1)/ndeltp`` 同樣「trat 從 0 開始、隨 substep 推進
  遠離起點」的方向一致。

兩種方向都保持「trat 在起點為 0、終點為 1」的統一語意，差別只在於「起點」
在真實時間上是較早（正向）還是較晚（逆向）的那個 canonical 小時。這個推導
由 ``_resolve_field_assignment`` 實作，並有獨立單元測試涵蓋兩個方向。

**已知限制（誠實揭露，供 WP-5 確認）**：本模組的 ``trat`` 是「目標 UTC 時刻
在兩個 canonical 小時之間的實際經過時間比例」，而 ``tracker.py`` 內部
substep 迴圈的 ``trat`` 是「``idt``／``ndeltp`` 的計數比例」。兩者只有在
「``ndeltp`` 個 substep 均勻切分整個小時、且呼叫端傳入 ``resolve`` 的
``target_utc_ns`` 確實對齊該 substep 的真實時刻」時才會數值相同。若 WP-5
選擇完全比照原版主迴圈、以固定 ``dt/ndeltp`` 驅動 substep（而不是用任意
``target_utc_ns`` 查詢), 則本模組回傳的 ``trat`` 只在「field1/field2 該指派
給哪個 UTC 小時」這件事上具權威性；substep 內的實際 ``trat`` 數值究竟要用
我們算的，還是重新用 ``idt/ndeltp`` 算一次，屬於 WP-5 的整合決策，本模組不
代為決定，只保證兩者在均勻切分的假設下相等。

缺口政策：沿用而非另創
------------------------
本模組不自行發明缺口分類，而是：

1. 每個月份先套用 ``forcing.OCMReconstructionPatchMonth`` 已驗證過的「觀測
   ＋重建 patch 合併時間軸」演算法（與 ``forcing.OCMNativeMonth.__init__``
   同一套邏輯，這裡重新實作是因為本模組刻意不依賴 ``NativeMesh``，詳見下方
   ``_merge_month_with_patch`` 的說明），得到「觀測與已核准重建都算數」的
   月內 canonical 時間軸。
2. 跨月合併後，呼叫 ``ocm_reconstruction.discover_hourly_gaps`` 找出「仍然
   缺少」的逐時 UTC——因為已核准 patch 已經併入時間軸，這裡找到的缺口保證
   是「未登錄」的缺口。
3. 任何查詢跨越這種缺口，或 ``inventory_forcing_gaps`` 掃到這種缺口，一律
   ``fail-closed`` 丟出例外並列出缺失的 UTC 時刻；不做最近值、不做零值、不
   做跨缺口的線性外插。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..forcing import OCMReconstructionPatchMonth
from ..mesh import NativeMesh
from ..ocm_reconstruction import HOURLY_NS, GapSpec, discover_hourly_gaps

__all__ = [
    "ForcingBridgeError",
    "UnregisteredForcingGapError",
    "UnresolvedForcingGapError",
    "HourlyFieldPair",
    "ForcingGapInventoryReport",
    "NativeForcingBridgeReader",
    "inventory_forcing_gaps",
    "project_face_dryness_to_nodes",
]


class ForcingBridgeError(RuntimeError):
    """forcing_bridge 讀取、合併或時間解析階段的一般性資料契約錯誤。"""


class UnregisteredForcingGapError(ForcingBridgeError):
    """查詢時刻落在一段既有 ``ocm_native`` 觀測與已核准重建都未涵蓋的缺口。

    依 AGENTS.md 的資料契約，未登錄缺口一律拒絕，不得以最近值、零值或跨缺口
    線性外插頂替；本例外訊息固定包含缺口兩端與缺失的 UTC 時刻，方便直接定位
    需要哪一段 OCM 重建或前處理補齊。
    """


class UnresolvedForcingGapError(ForcingBridgeError):
    """濕節點上仍殘留缺值，且未被任何已核准機制解釋，不得靜默補零。"""


def _iso(utc_ns: int) -> str:
    """把 UTC 奈秒轉成人類可讀的 ISO 8601 字串，只用於錯誤訊息與報告。"""

    return datetime.fromtimestamp(int(utc_ns) / 1_000_000_000, tz=UTC).isoformat()


# ── 月份層級：唯讀 memmap 載入與觀測／重建合併 ────────────────────────────


@dataclass(slots=True)
class _RawMonth:
    """一個月份 ``ocm_native`` 觀測陣列的唯讀 memory-map 集合。

    所有陣列都以 ``mmap_mode="r"`` 開啟，建構時只讀取小型 header／``time_utc_ns``
    做形狀檢查，不會把數十 GiB 的 ``hvel``／``zcor`` 實際讀進記憶體；真正的資料
    頁只在後續切出單一小時、單一欄位時才由作業系統依需要載入。
    """

    month_id: str
    time_utc_ns: np.ndarray
    hvel: np.ndarray
    vertical_velocity: np.ndarray
    zcor: np.ndarray
    elev: np.ndarray
    wetdry_elem: np.ndarray
    diffusivity: np.ndarray | None


def _require(condition: bool, message: str) -> None:
    """以一致例外型別檢查資料契約，避免用看似合理的資料繼續往下算。"""

    if not condition:
        raise ForcingBridgeError(message)


def _load_month_time_axis(month_dir: str | Path) -> np.ndarray:
    """只開啟一個月份的 ``time_utc_ns.npy``，供缺口盤點使用。

    缺口盤點必須能在「正式跑之前」快速掃過兩年資料，若連這一步都讀取
    ``hvel``／``zcor`` 等大型陣列，前置檢查本身就會變成效能瓶頸。因此這裡
    刻意只讀時間軸，語意對照 ``native_reader_excerpt.py`` 的
    ``_build_timeline``：先確認時間骨架，再決定要不要 materialize 動態欄位。
    """

    path = Path(month_dir) / "time_utc_ns.npy"
    if not path.is_file():
        raise ForcingBridgeError(f"缺少 OCM month 時間軸：{path}")
    times = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.int64)
    _require(
        times.ndim == 1 and times.size > 0 and np.all(np.diff(times) > 0),
        f"{path} 的 time_utc_ns 必須是非空、嚴格遞增的一維 int64 軸",
    )
    return times


def _load_raw_month(month_dir: str | Path, *, load_diffusivity: bool = True) -> _RawMonth:
    """唯讀開啟一個月份的 ``ocm_native`` 動態欄位，並檢查彼此形狀一致。

    載入型態與檢查項目對照 ``forcing.py`` 的 ``OCMNativeMonth.from_directory``
    ／``__init__``（約 722-766 行、597-615 行），但刻意不要求呼叫端提供
    ``NativeMesh``——本模組只負責把資料轉成 BayTrace 需要的 ``(layer, node)``
    陣列，節點數是否與某個特定追蹤網格一致，交由使用這些陣列的呼叫端（例如
    ``native_runner.grid_bridge``）驗證，避免這裡重複耦合一份不相干的網格
    拓撲相依。

    ``load_diffusivity=False``（見模組 docstring「``load_diffusivity`` 開關」
    一節）時完全不嘗試開啟 ``diffusivity.npy``——本機一次性複製只保留逆向
    驅動層需要的變數，該檔案可能根本不存在；``_RawMonth.diffusivity`` 此時
    為 ``None``，且不納入形狀檢查。
    """

    root = Path(month_dir)

    def load(name: str) -> np.ndarray:
        path = root / name
        if not path.is_file():
            raise ForcingBridgeError(f"缺少 OCM month 陣列：{path}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    time_utc_ns = _load_month_time_axis(root)
    hvel = load("hvel.npy")
    vertical_velocity = load("vertical_velocity.npy")
    zcor = load("zcor.npy")
    elev = load("elev.npy")
    wetdry_elem = load("wetdry_elem.npy")
    diffusivity = load("diffusivity.npy") if load_diffusivity else None

    time_count = int(time_utc_ns.size)
    _require(
        hvel.ndim == 4 and hvel.shape[0] == time_count and hvel.shape[-1] >= 2
        and np.issubdtype(hvel.dtype, np.floating),
        f"{root} 的 hvel 必須是浮點 (time,node,layer,component>=2) 且與 time_utc_ns 對齊",
    )
    expected_tnl = hvel.shape[:3]
    required_fields = [("vertical_velocity", vertical_velocity), ("zcor", zcor)]
    if load_diffusivity:
        required_fields.append(("diffusivity", diffusivity))
    for name, array in required_fields:
        _require(
            array.shape == expected_tnl and np.issubdtype(array.dtype, np.floating),
            f"{root} 的 {name} 必須是浮點且與 hvel 的 time/node/layer 對齊",
        )
    _require(
        elev.shape == expected_tnl[:2] and np.issubdtype(elev.dtype, np.floating),
        f"{root} 的 elev 必須是浮點 (time,node) 且與 hvel 對齊",
    )
    _require(
        wetdry_elem.ndim == 2 and wetdry_elem.shape[0] == time_count,
        f"{root} 的 wetdry_elem 必須是 (time,face) 且與 time_utc_ns 對齊",
    )
    return _RawMonth(
        month_id=root.name,
        time_utc_ns=time_utc_ns,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=elev,
        wetdry_elem=wetdry_elem,
        diffusivity=diffusivity,
    )


@dataclass(slots=True)
class _MergedMonth:
    """一個月份「觀測＋已核准重建 patch」合併後的小型 UTC 時間索引。

    大型陣列仍分別留在 ``raw``／``patch`` 的唯讀 memmap；這裡只保存合併後的
    ``time_utc_ns`` 與逐列來源（0=觀測、1=重建），語意與 ``forcing.py`` 的
    ``OCMNativeMonth._time_source_kind``／``_time_source_index`` 完全對應。
    """

    month_id: str
    raw: _RawMonth
    patch: OCMReconstructionPatchMonth | None
    time_utc_ns: np.ndarray
    source_kind: np.ndarray
    source_index: np.ndarray


def _merge_month_with_patch(
    month_id: str,
    raw: _RawMonth,
    patch: OCMReconstructionPatchMonth | None,
    *,
    load_diffusivity: bool = True,
) -> _MergedMonth:
    """合併一個月份的觀測時間軸與（若有）已核准重建 patch。

    這裡重新實作而非直接呼叫 ``forcing.OCMNativeMonth``，是因為該類別的
    建構子強制要求一個與 ``hvel`` 節點數一致的 ``NativeMesh``（見其
    ``__init__`` 的 ``expected_tn`` 檢查），但那是給「逐粒子重心＋垂向取樣」
    用的座標查找結構，與本模組「整格網一次轉置」的用途無關；硬塞一個無關
    的 ``NativeMesh`` 只會多一層不必要的耦合與測試複雜度。因此這裡只複製
    該類別合併時間軸的演算法本身（串接、``argsort(kind="stable")``、
    ``source_kind``／``source_index``），並比照其 shape 與不重疊檢查，確保
    行為與既有已驗收流程一致。

    ``load_diffusivity=False`` 時不比對 diffusivity 的 shape——``raw.
    diffusivity`` 此時是 ``None``，即使 ``patch``（補值月份）仍保留完整的
    diffusivity 陣列（本機一次性複製時補值 patch 是完整檔案），也一併忽略，
    對稱於「驅動層本來就不讀 diffusivity」的設計（見模組 docstring）。
    """

    if patch is None:
        source_kind = np.zeros(raw.time_utc_ns.size, dtype=np.int8)
        source_index = np.arange(raw.time_utc_ns.size, dtype=np.int64)
        return _MergedMonth(
            month_id=month_id,
            raw=raw,
            patch=None,
            time_utc_ns=raw.time_utc_ns,
            source_kind=source_kind,
            source_index=source_index,
        )

    _require(patch.month_id == month_id, f"重建 patch month_id 與月份不符：{patch.month_id} != {month_id}")
    patch_times = np.asarray(patch.time_utc_ns, dtype=np.int64)
    _require(
        np.intersect1d(raw.time_utc_ns, patch_times).size == 0,
        f"{month_id} 的重建 patch 與觀測 UTC 重疊，不得覆寫觀測資料",
    )
    shape_check_fields = ["hvel", "vertical_velocity", "zcor"]
    if load_diffusivity:
        shape_check_fields.append("diffusivity")
    for name in shape_check_fields:
        _require(
            np.asarray(getattr(patch, name)).shape[1:] == getattr(raw, name).shape[1:],
            f"{month_id} 重建 patch 的 {name} 網格 shape 與觀測不一致",
        )
    _require(
        np.asarray(patch.elev).shape[1:] == raw.elev.shape[1:],
        f"{month_id} 重建 patch 的 elev 網格 shape 與觀測不一致",
    )
    _require(
        np.asarray(patch.wetdry_elem).shape[1:] == raw.wetdry_elem.shape[1:],
        f"{month_id} 重建 patch 的 wetdry_elem 網格 shape 與觀測不一致",
    )

    merged_times = np.concatenate((raw.time_utc_ns, patch_times))
    order = np.argsort(merged_times, kind="stable")
    source_kind = np.concatenate(
        (np.zeros(raw.time_utc_ns.size, dtype=np.int8), np.ones(patch_times.size, dtype=np.int8))
    )[order]
    source_index = np.concatenate(
        (np.arange(raw.time_utc_ns.size, dtype=np.int64), np.arange(patch_times.size, dtype=np.int64))
    )[order]
    return _MergedMonth(
        month_id=month_id,
        raw=raw,
        patch=patch,
        time_utc_ns=merged_times[order],
        source_kind=source_kind,
        source_index=source_index,
    )


def _merged_row(merged: _MergedMonth, local_index: int, field_name: str) -> np.ndarray:
    """依合併後的月內索引取出單一 UTC 列，透明地選觀測或重建 patch 來源。"""

    kind = int(merged.source_kind[local_index])
    index = int(merged.source_index[local_index])
    if kind == 0:
        return np.asarray(getattr(merged.raw, field_name)[index])
    if merged.patch is None:  # pragma: no cover - 合併時已保證一致，防禦性檢查
        raise ForcingBridgeError(f"{merged.month_id} 合併索引指向不存在的重建 patch")
    return np.asarray(getattr(merged.patch, field_name)[index])


# ── 小時快照：轉置、缺值遮罩，只在快取未命中時計算一次 ─────────────────────


@dataclass(frozen=True, slots=True)
class _HourSnapshot:
    """單一 UTC 小時、已轉成 BayTrace 陣列配置的原生流場快照。

    ``u/v/w/diffusivity`` 為 ``(nvrt, nnode)`` C-contiguous float64，``eta``
    為 ``(nnode,)``；缺值（來源 ``hvel``／``zcor``／``vertical_velocity``／
    ``diffusivity`` 任一非有限）保留原生 ``NaN``，不做任何填補，另以
    ``missing_mask`` 明確標記同一位置。``dry_face_flag`` 是 ``wetdry_elem``
    當小時的原始列，未經任何節點投影。``diffusivity`` 在
    ``load_diffusivity=False``（見模組 docstring）時為 ``None``，且不計入
    ``missing_mask``。
    """

    utc_ns: int
    month_id: str
    time_origin: str
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    eta: np.ndarray
    diffusivity: np.ndarray | None
    missing_mask: np.ndarray
    dry_face_flag: np.ndarray


def _build_snapshot(
    merged: _MergedMonth, local_index: int, *, load_diffusivity: bool = True
) -> _HourSnapshot:
    """讀取並轉置單一 UTC 小時的全部欄位；呼叫端必須確保同一小時只呼叫一次。

    轉置與缺值遮罩都只在這裡計算一次（每小時一次，不是每粒子一次）：
    ``hvel``／``zcor``／``vertical_velocity``／``diffusivity`` 原生排列是
    ``(node, layer[, component])``，BayTrace 原版 kernel 要 ``(layer, node)``；
    ``np.ascontiguousarray`` 確保轉置後的陣列是 njit 友善的 C-contiguous
    記憶體配置，而不是原陣列的 strided view。``load_diffusivity=False`` 時
    完全不讀取、不轉置 diffusivity 列，``missing_mask`` 也不再把它計入缺值
    判定（見模組 docstring「``load_diffusivity`` 開關」）。
    """

    utc_ns = int(merged.time_utc_ns[local_index])
    hvel_row = _merged_row(merged, local_index, "hvel")
    zcor_row = _merged_row(merged, local_index, "zcor")
    w_row = _merged_row(merged, local_index, "vertical_velocity")
    elev_row = _merged_row(merged, local_index, "elev")
    wetdry_row = _merged_row(merged, local_index, "wetdry_elem")

    u = np.ascontiguousarray(np.asarray(hvel_row[:, :, 0], dtype=np.float64).T)
    v = np.ascontiguousarray(np.asarray(hvel_row[:, :, 1], dtype=np.float64).T)
    w = np.ascontiguousarray(np.asarray(w_row, dtype=np.float64).T)
    zcor = np.asarray(zcor_row, dtype=np.float64).T
    finite_terms = np.isfinite(u) & np.isfinite(v) & np.isfinite(w) & np.isfinite(zcor)
    diffusivity: np.ndarray | None = None
    if load_diffusivity:
        diff_row = _merged_row(merged, local_index, "diffusivity")
        diffusivity = np.ascontiguousarray(np.asarray(diff_row, dtype=np.float64).T)
        finite_terms = finite_terms & np.isfinite(diffusivity)
    missing_mask = ~finite_terms
    eta = np.ascontiguousarray(np.asarray(elev_row, dtype=np.float64))
    dry_face_flag = np.ascontiguousarray(np.asarray(wetdry_row, dtype=np.float64))
    kind = int(merged.source_kind[local_index])

    return _HourSnapshot(
        utc_ns=utc_ns,
        month_id=merged.month_id,
        time_origin="observed" if kind == 0 else "reconstructed",
        u=u,
        v=v,
        w=w,
        eta=eta,
        diffusivity=diffusivity,
        missing_mask=missing_mask,
        dry_face_flag=dry_face_flag,
    )


# ── 跨月 canonical 時間軸：邊界重疊採 prefer-last，語意對照 ocm_reconstruction ──


def _build_canonical_index(
    months: Mapping[str, _MergedMonth]
) -> tuple[np.ndarray, dict[int, tuple[str, int]]]:
    """建立跨月份的 canonical UTC 小時軸，月界重疊時保留後一個月份的列。

    「prefer-last」政策與 ``ocm_reconstruction.NpyDomainSource._locate`` 的
    月界 halo 慣例一致：依月份目錄名稱排序後，遇到相同 UTC 採後一個月，而
    非任意選第一筆或視為重複資料錯誤。
    """

    location_by_time: dict[int, tuple[str, int]] = {}
    for month_id in sorted(months):
        merged = months[month_id]
        for local_index, utc_ns in enumerate(merged.time_utc_ns):
            location_by_time[int(utc_ns)] = (month_id, local_index)
    times = np.asarray(sorted(location_by_time), dtype=np.int64)
    return times, location_by_time


# ── field1/field2 指派與 trat（模組 docstring 已完整推導） ─────────────────


def _resolve_field_assignment(
    lo_ns: int, hi_ns: int, target_ns: int, ibf: int
) -> tuple[int, int, float]:
    """依 ``ibf`` 把 (lo, hi) 指派成 (field1 時刻, field2 時刻, trat)。

    完整推導見模組 docstring「逆向追蹤的 field1/field2 指派與 trat 推導」。
    ``lo_ns <= target_ns <= hi_ns`` 由呼叫端保證；``lo_ns == hi_ns`` 是查詢
    剛好落在某個 canonical 小時上的退化情形，此時 field1 與 field2 指向同一
    個快照、``trat`` 定義為 0（兩者數值相同，取何值都不影響內插結果）。
    """

    if ibf not in (1, -1):
        raise ForcingBridgeError(f"ibf 只能是 1（正向）或 -1（逆向），收到 {ibf}")
    if lo_ns == hi_ns:
        return lo_ns, hi_ns, 0.0
    if not (lo_ns <= target_ns <= hi_ns):
        raise ForcingBridgeError(f"target={target_ns} 必須落在 lo={lo_ns} 與 hi={hi_ns} 之間（含端點）")
    natural_alpha = (target_ns - lo_ns) / (hi_ns - lo_ns)
    if ibf == 1:
        return lo_ns, hi_ns, float(natural_alpha)
    return hi_ns, lo_ns, float(1.0 - natural_alpha)


# ── 對外資料格式：一對相鄰時刻的欄位組 ──────────────────────────────────


@dataclass(frozen=True, slots=True)
class HourlyFieldPair:
    """一對相鄰 UTC 小時的 BayTrace 相容流場欄位（field1=前、field2=後）。

    「前」「後」是 BayTrace 原版 kernel 的內插方向（``field1*(1-trat) +
    field2*trat``），不是真實時間的先後——``ibf=-1``（本專案的逆向追蹤）時
    ``time_before_utc_ns`` 對應真實時間較晚的那一小時，``time_after_utc_ns``
    對應較早的那一小時；推導見模組 docstring。所有速度／海面欄位保留原生
    ``NaN`` 於缺值位置，``missing_maskN`` 明確標記同一批位置，呼叫端在套用
    BayTrace 自己的乾濕判定（``dp``/``eta``/``h0``）之後，應呼叫
    ``raise_if_unresolved_gap`` 確認濕節點沒有殘留缺值。

    ``diffusivity_field1``／``diffusivity_field2`` 在
    ``NativeForcingBridgeReader(load_diffusivity=False)``（見模組 docstring
    「``load_diffusivity`` 開關」）時為 ``None``——本專案逆向驅動層本來就
    不讀這兩個欄位，水平擴散改由 Smagorinsky 方法從流速直接算出。
    """

    time_before_utc_ns: int
    time_after_utc_ns: int
    trat: float
    ibf: int

    u_field1: np.ndarray
    v_field1: np.ndarray
    w_field1: np.ndarray
    eta_field1: np.ndarray
    diffusivity_field1: np.ndarray | None
    missing_mask1: np.ndarray
    dry_face_flag1: np.ndarray
    month_id1: str
    time_origin1: str

    u_field2: np.ndarray
    v_field2: np.ndarray
    w_field2: np.ndarray
    eta_field2: np.ndarray
    diffusivity_field2: np.ndarray | None
    missing_mask2: np.ndarray
    dry_face_flag2: np.ndarray
    month_id2: str
    time_origin2: str

    def __post_init__(self) -> None:
        """檢查兩個時刻欄位彼此形狀一致、``trat`` 有限且落在 ``[0, 1]``。"""

        if not (0.0 <= float(self.trat) <= 1.0) or not np.isfinite(self.trat):
            raise ForcingBridgeError(f"trat 必須是 [0,1] 內的有限值，收到 {self.trat}")
        if (self.diffusivity_field1 is None) != (self.diffusivity_field2 is None):
            raise ForcingBridgeError(
                "diffusivity_field1 與 diffusivity_field2 必須同時為 None 或同時有值"
            )
        shape = self.u_field1.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ForcingBridgeError("速度場必須是非空的 (nvrt, nnode) 陣列")
        shape_check_names = [
            "v_field1", "w_field1", "missing_mask1",
            "u_field2", "v_field2", "w_field2", "missing_mask2",
        ]
        contiguous_check_names = [
            "u_field1", "v_field1", "w_field1", "eta_field1",
            "u_field2", "v_field2", "w_field2", "eta_field2",
        ]
        if self.diffusivity_field1 is not None:
            shape_check_names += ["diffusivity_field1", "diffusivity_field2"]
            contiguous_check_names += ["diffusivity_field1", "diffusivity_field2"]
        for name in shape_check_names:
            array = getattr(self, name)
            if array.shape != shape:
                raise ForcingBridgeError(f"{name} shape 必須與 u_field1 一致：{array.shape} != {shape}")
        for name in ("eta_field1", "eta_field2"):
            array = getattr(self, name)
            if array.shape != (shape[1],):
                raise ForcingBridgeError(f"{name} shape 必須等於節點數：{array.shape} != {(shape[1],)}")
        for name in contiguous_check_names:
            array = getattr(self, name)
            if not array.flags["C_CONTIGUOUS"]:
                raise ForcingBridgeError(f"{name} 必須是 C-contiguous，供 njit kernel 安全讀取")

    def raise_if_unresolved_gap(
        self,
        *,
        node_is_dry1: np.ndarray | None = None,
        node_is_dry2: np.ndarray | None = None,
    ) -> None:
        """確認缺值遮罩沒有殘留在（呼叫端認定為）濕的節點上。

        ``node_is_dryN`` 是呼叫端用 BayTrace 自己的 ``dp``／``eta``／``h0``
        算出的節點乾濕狀態（``(nnode,)`` 布林，``True``=乾）；未提供時視為
        「沒有任何節點已知為乾」，也就是任何缺值都會被視為未解釋而報錯。
        這個方法只做檢查、不修改任何陣列，也不會把缺值改寫成 0。
        """

        self._raise_if_unresolved(self.missing_mask1, node_is_dry1, self.time_before_utc_ns, "field1")
        self._raise_if_unresolved(self.missing_mask2, node_is_dry2, self.time_after_utc_ns, "field2")

    @staticmethod
    def _raise_if_unresolved(
        missing_mask: np.ndarray, node_is_dry: np.ndarray | None, utc_ns: int, label: str
    ) -> None:
        if node_is_dry is None:
            remaining = missing_mask
        else:
            dry = np.asarray(node_is_dry, dtype=bool)
            if dry.shape != (missing_mask.shape[1],):
                raise ForcingBridgeError(f"node_is_dry shape 必須等於節點數：{dry.shape}")
            remaining = missing_mask & ~dry[None, :]
        if np.any(remaining):
            bad_nodes = np.flatnonzero(np.any(remaining, axis=0))
            raise UnresolvedForcingGapError(
                f"{label}（UTC {_iso(utc_ns)}）在濕節點上仍有 {bad_nodes.size} 個缺值節點，"
                f"第一個是節點 {int(bad_nodes[0])}；缺值不得以零值或最近值頂替，"
                "請確認是否需要 OCM 缺口重建 patch 或排除該節點"
            )


# ── 選用交叉檢查：把 wetdry_elem 概略投影到節點，純供 QC 參考 ──────────────


def project_face_dryness_to_nodes(
    mesh: NativeMesh, dry_face_flag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """把原生 face（元素）乾濕列概略投影到節點層級，供 QC 交叉檢查。

    語意比照 BayTrace ``tracker._compute_dry_flags`` 的「節點濕若任一相鄰
    元素濕」規則：只要有一個觸及該節點的 face 是濕的（``dry_face_flag==0``），
    該節點就視為濕；否則（所有觸及的 face 都乾）視為乾。這只是「概略」投影，
    因為 BayTrace 真正的節點乾濕仍以其自己的 ``dp``／``eta``／``h0`` 計算為
    準（見模組 docstring）；本函式純粹用 OCM 自己的 ``wetdry_elem`` 做交叉
    檢查或診斷，不寫入任何 forcing 陣列。

    回傳
    ------
    ``(node_is_dry, node_is_known)``：兩者皆為 ``(nnode,)`` 布林。沒有任何
    face 觸及、或觸及的 face 的 ``wetdry_elem`` 值非有限（例如重建 patch 的
    ``NaN=not_persistent``）的節點，``node_is_known`` 為 ``False``，此時
    ``node_is_dry`` 固定為 ``False``（fail-closed：未知狀態不得被誤判為
    「已知為乾」而放行缺值）。
    """

    dry = np.asarray(dry_face_flag, dtype=np.float64)
    if dry.shape != (mesh.face_nodes_local.shape[0],):
        raise ForcingBridgeError("dry_face_flag shape 必須等於 mesh 的原生 face 數")
    node_count = int(mesh.node_xy.shape[0])
    any_wet = np.zeros(node_count, dtype=bool)
    any_known = np.zeros(node_count, dtype=bool)
    for face_index in range(dry.shape[0]):
        state = dry[face_index]
        if not np.isfinite(state):
            continue
        count = int(mesh.face_node_count[face_index])
        nodes = np.asarray(mesh.face_nodes_local[face_index, :count], dtype=np.int64)
        any_known[nodes] = True
        if state == 0.0:
            any_wet[nodes] = True
    node_is_dry = any_known & ~any_wet
    return node_is_dry, any_known


# ── 缺口盤點：正式跑之前的第一道前置檢查 ────────────────────────────────


@dataclass(frozen=True, slots=True)
class ForcingGapInventoryReport:
    """一段指定期間內，``ocm_native`` 逐時完整性的前置檢查報告。

    只由各月份的 ``time_utc_ns`` 與（若提供）重建 patch 的時間軸算出，不讀取
    ``hvel``／``zcor`` 等大型欄位，因此可以在正式跑之前快速掃過整個 2024–2025
    期間；這是背景需求特別強調的「資料提供者到 2026-09-14 才補齊原始 nc 檔，
    已驗收產品很可能尚未重建」情境下的第一道關卡。
    """

    flow_id: str
    expected_start_utc_ns: int
    expected_end_utc_ns: int
    expected_hour_count: int
    observed_hour_count: int
    reconstructed_hour_count: int
    unregistered_gaps: tuple[GapSpec, ...]

    @property
    def is_run_ready(self) -> bool:
        """沒有任何未登錄缺口時，才適合直接驅動正式批次追蹤。"""

        return len(self.unregistered_gaps) == 0

    @property
    def unregistered_hour_count(self) -> int:
        """所有未登錄缺口涵蓋的逐時 UTC 總數。"""

        return int(sum(gap.missing_times_ns.size for gap in self.unregistered_gaps))

    def describe(self) -> str:
        """回傳可直接列印／寫入前置檢查記錄的多行摘要，逐一列出缺口起訖。"""

        lines = [
            f"flow_id={self.flow_id} 期間 [{_iso(self.expected_start_utc_ns)}, "
            f"{_iso(self.expected_end_utc_ns)}]：預期 {self.expected_hour_count} 筆、"
            f"觀測 {self.observed_hour_count} 筆、已核准重建 {self.reconstructed_hour_count} 筆、"
            f"未登錄缺口 {self.unregistered_hour_count} 筆",
        ]
        for gap in self.unregistered_gaps:
            lines.append(
                f"  - 缺口 {gap.gap_id}：{_iso(int(gap.missing_times_ns[0]))} 到 "
                f"{_iso(int(gap.missing_times_ns[-1]))}（共 {gap.missing_times_ns.size} 筆，"
                f"兩端支援 {_iso(gap.left_time_ns)} / {_iso(gap.right_time_ns)}）"
            )
        return "\n".join(lines)


def inventory_forcing_gaps(
    month_dirs: Mapping[str, str | Path],
    *,
    expected_start_utc_ns: int,
    expected_end_utc_ns: int,
    patches: Mapping[str, OCMReconstructionPatchMonth] | None = None,
    flow_id: str = "lbt_native",
) -> ForcingGapInventoryReport:
    """掃過指定期間的 ``ocm_native`` 月份，回報逐時完整性，供正式跑前檢查。

    參數
    ------
    month_dirs:
        ``{month_id: 月份資料夾路徑}``；只需要涵蓋 ``[expected_start_utc_ns,
        expected_end_utc_ns]`` 的月份，通常是 2024、2025 兩年裡實際要用到的
        月份子集。
    patches:
        ``{month_id: 已核准的 OCMReconstructionPatchMonth}``；只有出現在這裡
        的重建才會被視為「已登錄」，其餘缺口一律回報為未登錄。
    flow_id:
        寫入 ``GapSpec.gap_id`` 的 flow 識別碼，只影響報告可讀性。

    本函式只讀取每個月份的 ``time_utc_ns.npy``（與 patch 的時間軸），不觸碰
    任何大型動態欄位，因此即使涵蓋兩年份月份也應該在數秒內完成。
    """

    if expected_end_utc_ns <= expected_start_utc_ns:
        raise ForcingBridgeError("expected_end_utc_ns 必須晚於 expected_start_utc_ns")
    if not month_dirs:
        raise ForcingBridgeError("month_dirs 不得為空，缺口盤點至少需要一個月份")

    patches = patches or {}
    merged_time_axes: dict[str, np.ndarray] = {}
    observed_hour_count = 0
    reconstructed_hour_count = 0
    for month_id, path in month_dirs.items():
        observed_times = _load_month_time_axis(path)
        observed_hour_count += int(observed_times.size)
        patch = patches.get(month_id)
        if patch is None:
            merged_time_axes[month_id] = observed_times
            continue
        patch_times = np.asarray(patch.time_utc_ns, dtype=np.int64)
        _require(
            np.intersect1d(observed_times, patch_times).size == 0,
            f"{month_id} 的重建 patch 與觀測 UTC 重疊",
        )
        reconstructed_hour_count += int(patch_times.size)
        merged_time_axes[month_id] = np.sort(np.concatenate((observed_times, patch_times)))

    canonical_times = np.unique(np.concatenate(list(merged_time_axes.values())))
    if canonical_times.size < 2:
        raise ForcingBridgeError("可用時間格點不足兩筆，無法進行缺口盤點")

    gaps = discover_hourly_gaps(
        canonical_times,
        flow_id=flow_id,
        expected_start_ns=int(expected_start_utc_ns),
        expected_end_ns=int(expected_end_utc_ns),
    )
    expected_hour_count = int(
        (int(expected_end_utc_ns) - int(expected_start_utc_ns)) // int(HOURLY_NS)
    ) + 1
    return ForcingGapInventoryReport(
        flow_id=flow_id,
        expected_start_utc_ns=int(expected_start_utc_ns),
        expected_end_utc_ns=int(expected_end_utc_ns),
        expected_hour_count=expected_hour_count,
        observed_hour_count=observed_hour_count,
        reconstructed_hour_count=reconstructed_hour_count,
        unregistered_gaps=gaps,
    )


# ── 主要讀取器：以流場時刻為單位快取，支援滑動前進 ──────────────────────


class NativeForcingBridgeReader:
    """把多個月份的 ``ocm_native`` memmap 轉成 BayTrace 相容的雙時次欄位。

    建構時只載入各月份的動態陣列 memmap（不讀取實際資料頁）並合併觀測／
    重建的月內時間軸；真正的欄位轉置延後到 ``resolve`` 第一次需要某個小時
    時才發生，並以「目前查詢用得到的兩個小時」為快取範圍——同一小時只轉置
    一次，往前（或依 ``ibf`` 往追蹤方向）滑動一小時時，原本的 field2 快照
    直接變成新的 field1。``read_counts`` 記錄每個 UTC 小時實際被讀取（轉置）
    的次數，供效能測試驗證「沒有重複讀取」。

    ``load_diffusivity``（預設 ``True``，行為與既有完全相同）：``False`` 時
    完全不開啟任何月份的 ``diffusivity.npy``，回傳的 ``HourlyFieldPair.
    diffusivity_field1``／``diffusivity_field2`` 為 ``None``，缺值遮罩也不
    再把 diffusivity 計入。這個模式只適用於本身不讀 OCM 擴散係數的呼叫端
    （本專案逆向驅動層，見模組 docstring「``load_diffusivity`` 開關」）。
    """

    def __init__(
        self,
        month_dirs: Mapping[str, str | Path],
        *,
        ibf: int,
        patches: Mapping[str, OCMReconstructionPatchMonth] | None = None,
        load_diffusivity: bool = True,
    ) -> None:
        if ibf not in (1, -1):
            raise ForcingBridgeError(f"ibf 只能是 1（正向）或 -1（逆向），收到 {ibf}")
        if not month_dirs:
            raise ForcingBridgeError("month_dirs 不得為空")
        self.ibf = int(ibf)
        self._load_diffusivity = bool(load_diffusivity)
        patches = patches or {}
        self._months: dict[str, _MergedMonth] = {}
        for month_id, path in month_dirs.items():
            raw = _load_raw_month(path, load_diffusivity=self._load_diffusivity)
            self._months[month_id] = _merge_month_with_patch(
                month_id, raw, patches.get(month_id), load_diffusivity=self._load_diffusivity
            )
        self._times, self._location_by_time = _build_canonical_index(self._months)
        if self._times.size < 2:
            raise ForcingBridgeError("可用時間格點不足兩筆，無法建立雙時次欄位")
        self._snapshot_cache: dict[int, _HourSnapshot] = {}
        self.read_counts: dict[int, int] = {}

    @property
    def available_time_range_utc_ns(self) -> tuple[int, int]:
        """回傳目前已載入月份實際涵蓋的 UTC 起訖（含端點）。"""

        return int(self._times[0]), int(self._times[-1])

    def _get_snapshot(self, utc_ns: int) -> _HourSnapshot:
        """回傳指定小時的快照；快取命中時完全不重新讀取或轉置。"""

        cached = self._snapshot_cache.get(utc_ns)
        if cached is not None:
            return cached
        location = self._location_by_time.get(utc_ns)
        if location is None:  # pragma: no cover - _bracket 已保證存在，防禦性檢查
            raise ForcingBridgeError(f"canonical 時間軸不含 {_iso(utc_ns)}")
        month_id, local_index = location
        snapshot = _build_snapshot(
            self._months[month_id], local_index, load_diffusivity=self._load_diffusivity
        )
        self._snapshot_cache[utc_ns] = snapshot
        self.read_counts[utc_ns] = self.read_counts.get(utc_ns, 0) + 1
        return snapshot

    def _bracket(self, target_utc_ns: int) -> tuple[int, int]:
        """找出涵蓋 ``target_utc_ns`` 的相鄰 canonical 小時，未登錄缺口一律拒絕。

        查詢時刻剛好命中某個 canonical 小時時，仍必須依 ``ibf`` 往「追蹤方向」
        找出真正要配對的下一個小時，而不是直接回傳退化的自我配對——否則正向
        追蹤從整點出發時，會漏掉這一小時到下一小時之間本該存在的內插視窗。
        只有當追蹤方向已經沒有更多資料（已載入範圍的端點）時，才回退成
        ``field1 == field2`` 的退化配對。非整點查詢維持原本「找左右相鄰
        canonical 小時」的邏輯，與 ``ibf`` 無關（方向只影響哪個是 field1、
        哪個是 field2，由 ``_resolve_field_assignment`` 決定）。
        """

        times = self._times
        target = int(target_utc_ns)
        position = int(np.searchsorted(times, target, side="left"))
        exact = position < times.size and int(times[position]) == target

        if not exact:
            if position == 0 or position >= times.size:
                lo, hi = int(times[0]), int(times[-1])
                raise ForcingBridgeError(
                    f"目標時間 {_iso(target)} 超出已載入原生資料涵蓋範圍 [{_iso(lo)}, {_iso(hi)}]"
                )
            lo_pos, hi_pos = position - 1, position
        elif self.ibf == 1:
            lo_pos = position
            hi_pos = position + 1 if position + 1 < times.size else position
        else:
            hi_pos = position
            lo_pos = position - 1 if position - 1 >= 0 else position

        lo_ns, hi_ns = int(times[lo_pos]), int(times[hi_pos])
        if lo_ns != hi_ns and hi_ns - lo_ns != int(HOURLY_NS):
            missing_times = list(range(lo_ns + int(HOURLY_NS), hi_ns, int(HOURLY_NS)))
            preview = ", ".join(_iso(value) for value in missing_times[:5])
            suffix = "" if len(missing_times) <= 5 else f" 等共 {len(missing_times)} 筆"
            raise UnregisteredForcingGapError(
                f"目標時間 {_iso(target)} 落在未登錄缺口內：{_iso(lo_ns)} 到 {_iso(hi_ns)} "
                f"之間缺少 {len(missing_times)} 個逐時資料點（{preview}{suffix}）；"
                "請先以核准的 OCM 重建 patch 補齊，或改用 gap-safe arrival window，"
                "不得以最近值或零值頂替"
            )
        return lo_ns, hi_ns

    def resolve(self, target_utc_ns: int) -> HourlyFieldPair:
        """回傳涵蓋 ``target_utc_ns`` 的雙時次欄位；自動沿用可重用的快取快照。

        ``ibf`` 決定 field1／field2 的真實時間指派方向與 ``trat`` 定義，完整
        推導見模組 docstring。回傳後會清除快取中不再需要的快照，讓常駐記憶體
        維持在「最多兩個小時」，即使呼叫端在很長的回溯視窗中持續呼叫。
        """

        lo_ns, hi_ns = self._bracket(target_utc_ns)
        lo_snapshot = self._get_snapshot(lo_ns)
        hi_snapshot = self._get_snapshot(hi_ns)
        for cached_ns in tuple(self._snapshot_cache):
            if cached_ns not in (lo_ns, hi_ns):
                del self._snapshot_cache[cached_ns]

        before_ns, after_ns, trat = _resolve_field_assignment(lo_ns, hi_ns, int(target_utc_ns), self.ibf)
        before_snapshot = lo_snapshot if before_ns == lo_ns else hi_snapshot
        after_snapshot = hi_snapshot if after_ns == hi_ns else lo_snapshot

        return HourlyFieldPair(
            time_before_utc_ns=before_ns,
            time_after_utc_ns=after_ns,
            trat=trat,
            ibf=self.ibf,
            u_field1=before_snapshot.u,
            v_field1=before_snapshot.v,
            w_field1=before_snapshot.w,
            eta_field1=before_snapshot.eta,
            diffusivity_field1=before_snapshot.diffusivity,
            missing_mask1=before_snapshot.missing_mask,
            dry_face_flag1=before_snapshot.dry_face_flag,
            month_id1=before_snapshot.month_id,
            time_origin1=before_snapshot.time_origin,
            u_field2=after_snapshot.u,
            v_field2=after_snapshot.v,
            w_field2=after_snapshot.w,
            eta_field2=after_snapshot.eta,
            diffusivity_field2=after_snapshot.diffusivity,
            missing_mask2=after_snapshot.missing_mask,
            dry_face_flag2=after_snapshot.dry_face_flag,
            month_id2=after_snapshot.month_id,
            time_origin2=after_snapshot.time_origin,
        )
