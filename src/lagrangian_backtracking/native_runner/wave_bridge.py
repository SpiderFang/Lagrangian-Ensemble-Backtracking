"""WP-5 補充：把已驗收的 NWW3 波浪逐月產品（``nww3_analysis`` schema 1）內插到
追蹤網格節點，補上 ``backward_driver.run_backward_batch`` 的 ``wave_field_resolver``
介面缺口（見 ``backward_driver`` 模組 docstring「已知限制」第 2 點）。

為什麼需要獨立的橋接模組
--------------------------
``backward_driver.compose_total_velocity_fields`` 需要的是「已經內插到追蹤網格
每個節點」的 ``WaveFieldAtNodes``（``Hs``／``fp``／來向／``valid_mask``／
``qc_flags``，皆為 ``(nnode,)``），但 ``nww3_analysis`` 產品本身是「規則經緯度
網格 × 逐月時間軸」的原生格式（見 ``forcing.NWWAnalysisMonth``）。既有
``baytrace_wave_frame.py`` 的 ``_node_wave_samples`` 已經示範過一種轉換方式，
但它用 ``for node in range(count): manager.sample_at_time_index(...)`` 逐節點
呼叫 Python 函式——先前的效能量測顯示這是「已知的效能災難」（一個節點一次
Python 函式呼叫，數萬節點 × 數百個流場小時會直接讓橋接本身變成瓶頸，而不是
真正的物理計算）。本模組刻意採取完全不同的實作策略：

1. **開機時只算一次**空間內插的四角索引與雙線性權重（
   ``build_nww3_node_interpolation_weights``），因為追蹤網格節點的經緯度與
   NWW3 規則網格在整個 run 期間都不會變動——這與 ``grid_bridge.py`` 只做一次
   網格拓樸轉換是同一種設計哲學（靜態幾何前處理，動態資料才逐步重算）。
2. **每小時只做一次向量化收集＋加權**（``gather_nww3_hour_at_nodes``）：對
   NWW3 該小時的 2 維 ``(lat, lon)`` 陣列，一次用四組整數索引陣列做花式索引
   （fancy indexing）取出全部節點的四個角點值，再用預先算好的權重加權平均，
   全程沒有任何逐節點的 Python 迴圈或函式呼叫。

資料來源與職責邊界
--------------------
本模組只讀取 ``forcing.NWWAnalysisMonth``（``nww3_analysis`` schema 1 的月份
唯讀 memmap 容器，已由 ``forcing.py`` 驗證過欄位 shape 與 dtype），直接讀取其
公開屬性（``lon``／``lat``／``time_utc_ns``／``significant_wave_height``／
``peak_frequency``／``peak_direction_raw_deg``／``valid_mask_wave``／
``qc_flags``），不呼叫也不修改其任何方法（尤其不呼叫其
``sample``／``sample_at_time_index``——那是逐點介面，用在這裡就會退化成
``baytrace_wave_frame.py`` 那種逐節點迴圈）；本模組同樣不讀取任何原始 NetCDF
或 transfer archive，遵守 ``AGENTS.md`` 的資料契約。

單位與慣例
------------
``significant_wave_height_m``（有效波高，m）、``peak_frequency_hz``（尖峰
頻率，Hz）與 ``direction_raw_deg``（原始尖峰波向，度）三者的物理定義完全
沿用 ``forcing.py``／``stokes_batch.py`` 既有慣例，本模組不重新定義：波向是
「正北起順時針的來向」（不是去向），與 ``stokes_batch.batch_finite_depth_stokes``
的輸入慣例一致，可以直接餵給它，不需要呼叫端再轉換。

空間內插：雙線性、四角索引慣例與 ``forcing.py`` 逐點內插完全一致
------------------------------------------------------------------
``build_nww3_node_interpolation_weights`` 對每個追蹤網格節點，用
``numpy.searchsorted(..., side="right")`` 找到 NWW3 經緯度軸上「夾住」該節點
的相鄰格點（``x0<x1``、``y0<y1``），角點順序固定是
``(y0,x0), (y0,x1), (y1,x0), (y1,x1)``，權重固定是
``(1-wy)(1-wx), (1-wy)wx, wy(1-wx), wy*wx``——這與
``forcing.NWWAnalysisMonth._sample_spatial_at_time`` 逐點內插使用的慣例逐項
相同（同一套 bracket／權重公式），只是這裡對全部節點一次算完、快取索引，而
不是每次查詢時重新搜尋。

遮罩政策：要求四個角點全部有效，不接受「部分角點頂替」
----------------------------------------------------------
沿用 ``forcing.NWWAnalysisMonth._sample_spatial_at_time`` 的既有政策：四個
角點只要有任何一個 ``valid_mask_wave`` 為假，這個節點這一小時就整體判定為
無效（``WaveFieldAtNodes.valid_mask_wave=False``），不會拿其餘三個角點做
「退化雙線性」或補值——那樣會在遮罩邊界附近（陸地／域外交界）悄悄引入偏差，
且違反 AGENTS.md「缺值不得以最近值或零值頂替」的原則。四個角點的
``qc_flags`` 一律取聯集（bitwise OR）保留下來，即使節點被判定為無效，這個
聯集仍然有 QC 稽核價值（可用來分辨「純遮罩問題」還是「遮罩通過但品質有
疑慮」）。

除了四角遮罩外，本模組另外套用與 ``forcing.py`` 逐點內插相同的兩層物理防呆：

1. 四個角點的 ``Hs``／``fp``／方向任一非有限（``NaN``／``inf``），節點視為
   無效——理論上不該發生在已驗收的 ``valid_mask_wave=True`` 資料上，但防禦性
   地與 ``forcing.py`` 保持一致，不假設上游資料絕對乾淨。
2. 內插後的波高為負、頻率非正，或方向圓向量退化（``|sin 加權和|+|cos 加權和|``
   趨近於零，代表四個角點的波向彼此抵銷、內插結果沒有明確方向）——這三種
   情況與 ``forcing.py`` 的 ``SampleQC.INVALID_PHYSICS`` 判斷同義，這裡同樣
   判定節點無效。

方向的空間平均：先取 sin／cos 分量再合成角度，不能直接對角度線性平均
--------------------------------------------------------------------------
角度是圓週期量，用四個角點的角度直接做加權算術平均在跨越 0°/360° 邊界時會
得到完全錯誤的結果（例如 350° 與 10° 兩個角點各半加權，正確答案應該是
0°／360°附近，但直接線性平均會得到 180°，方向整個反過來）。本模組沿用
``forcing.py`` 1946-1958 行、``stokes_batch.py``「波向慣例」一節記載的既有
專案作法：先把每個角點的角度轉成單位圓上的東／北分量
（``sin(角度)``／``cos(角度)``，角度以正北為 0、順時針為正），對這兩個分量
分別做雙線性加權平均，最後才用 ``atan2(東分量, 北分量) % 360`` 轉回角度。
這個「先分量後合成」的順序，即使跨越 0°/360° 邊界也能得到正確、連續的加權
平均方向（見本模組測試對跨邊界情境的專門驗證）。

時間：要求逐小時精確對齊，缺時刻明確拒絕，不做時間內插
----------------------------------------------------------
與 ``forcing_bridge.py``（OCM 海流／海面）不同，本模組**不**在 NWW3 時間軸上
做任何內插或最近值搜尋：``NativeWaveBridgeReader.resolve(target_utc_ns)``
只接受在已載入月份的 ``time_utc_ns`` 陣列裡能找到**精確**（位元相等）UTC
奈秒值的查詢，找不到就丟出 ``WaveBridgeError`` 並中止，不嘗試找最接近的
NWW3 時間列頂替。這是刻意的設計取捨：

* 正式資料的 ``nww3_analysis`` 逐月產品被要求是逐時、與 ``ocm_native``
  同一組 UTC 格點，因此「精確對齊」在正常情況下應該永遠成立；一旦不成立
  （NWW3 缺某個小時、或兩個產品的時間格點意外沒有對齊），代表資料契約本身
  出了問題，應該在 QC 階段就被攔下，而不是被本模組用內插悄悄掩蓋。
* 呼叫端（``run_backward_batch``）呼叫 ``wave_field_resolver`` 時傳入的正是
  ``HourlyFieldPair.time_before_utc_ns``／``time_after_utc_ns``——這兩個值
  本來就是 OCM 逐時 canonical 時刻，不是任意的浮點時刻，因此「要求精確對齊」
  在整合到 ``run_backward_batch`` 之後不會造成額外限制。

空間涵蓋缺口：近岸節點可能落在 NWW3 網格範圍外
--------------------------------------------------
NWW3 全球或區域分析網格的解析度、外插邊界不一定涵蓋追蹤網格全部節點，尤其是
非常靠近陸地、NWW3 網格因陸地遮罩而沒有格點的近岸區域。
``build_nww3_node_interpolation_weights`` 會明確標記這種節點
（``node_covered_by_nww3=False``：節點經緯度落在 NWW3 lon／lat 軸範圍
之外），並讓這種節點在**任何小時**都回傳 ``valid_mask_wave=False``（見
``gather_nww3_hour_at_nodes``），不會用邊界外插或最近格點頂替。

**這對驅動層物理意義的後果，呼叫端務必理解**：``compose_total_velocity_fields``
在 ``valid_mask_wave=False`` 的節點上完全不加 Stokes 貢獻，只保留純海流＋
沉降；這代表「該節點的 Stokes 漂流是缺值、未知」，而不是「該節點確實沒有
波浪、Stokes 為零」。如果近岸關注區域有相當比例的節點落在 NWW3 涵蓋範圍外，
正式報告必須揭露這個比例（可用
``NativeWaveBridgeReader.weights.node_covered_by_nww3`` 直接統計），因為
「近岸粒子完全沒有 Stokes 修正」與「近岸粒子的 Stokes 剛好是零」是兩個科學
意義完全不同的陳述，不應該被本模組的 ``NaN``／``False`` 靜默混為一談，也
不應該被呼叫端不假思索地讀成 0。

快取與效能
------------
``NativeWaveBridgeReader`` 對已經算好的 ``WaveFieldAtNodes`` 結果做一個大小
為 2 的小型快取（依插入順序淘汰最舊的一筆），語意比照
``forcing_bridge.NativeForcingBridgeReader`` 的「同一流場小時只轉置一次」：
逆向主迴圈每小時會用同一個 ``field_pair.time_before_utc_ns``／
``time_after_utc_ns`` 各呼叫一次 ``wave_field_resolver``，且相鄰小時之間
這兩個時刻其中一個會被下一小時重用（滑動視窗），因此快取大小 2 就能讓
「同一小時同一端點只內插一次」，不需要對每個 UTC 各自建一個無上限成長的
字典。``read_counts`` 記錄每個 UTC 小時實際被計算（而非快取命中）的次數，
供效能測試驗證「沒有重複計算」。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..forcing import NWWAnalysisMonth
from .backward_driver import WaveFieldAtNodes

__all__ = [
    "WaveBridgeError",
    "NWW3NodeInterpolationWeights",
    "build_nww3_node_interpolation_weights",
    "gather_nww3_hour_at_nodes",
    "NativeWaveBridgeReader",
]


class WaveBridgeError(RuntimeError):
    """NWW3→節點波浪橋接階段的一般性資料契約錯誤（載入、網格不一致、缺時刻）。"""


def _iso(utc_ns: int) -> str:
    """把 UTC 奈秒轉成人類可讀的 ISO 8601 字串，只用於錯誤訊息，風格比照
    ``forcing_bridge._iso``（獨立重寫避免跨模組耦合一個純字串工具函式）。
    """

    return datetime.fromtimestamp(int(utc_ns) / 1_000_000_000, tz=UTC).isoformat()


# ── 開機時計算一次：追蹤網格節點對 NWW3 規則網格的雙線性內插索引與權重 ──────


@dataclass(frozen=True, slots=True)
class NWW3NodeInterpolationWeights:
    """追蹤網格每個節點對 NWW3 規則經緯度網格預先算好的雙線性內插索引與權重。

    所有欄位皆為 ``(nnode,)``，節點順序必須與呼叫
    ``build_nww3_node_interpolation_weights`` 時傳入的 ``node_lon``／
    ``node_lat``（也就是 ``NativeMesh.node_lon``／``node_lat``，與
    ``grid_bridge.NativeGridArrays`` 的節點順序相同）逐一對應。

    四個角點的順序固定是 ``(y0,x0), (y0,x1), (y1,x0), (y1,x1)``，對應權重
    ``weight_00, weight_01, weight_10, weight_11``；這與
    ``forcing.NWWAnalysisMonth._sample_spatial_at_time`` 的角點與權重慣例
    逐項相同（見模組 docstring）。``node_covered_by_nww3`` 為假的節點，其
    索引與權重仍然是「安全但無意義」的值（已 clip 到網格範圍內，不會造成
    陣列越界），真正的無效判定由 ``gather_nww3_hour_at_nodes`` 讀取這個旗標
    後套用，見模組 docstring「空間涵蓋缺口」一節。
    """

    node_y0: np.ndarray
    node_y1: np.ndarray
    node_x0: np.ndarray
    node_x1: np.ndarray
    weight_00: np.ndarray
    weight_01: np.ndarray
    weight_10: np.ndarray
    weight_11: np.ndarray
    node_covered_by_nww3: np.ndarray

    def __post_init__(self) -> None:
        """檢查全部欄位是同長度的一維陣列，避免呼叫端傳入形狀不一致的自製權重。"""

        shape = self.node_y0.shape
        if len(shape) != 1:
            raise WaveBridgeError("NWW3NodeInterpolationWeights 的欄位必須是一維陣列")
        for name in (
            "node_y1", "node_x0", "node_x1",
            "weight_00", "weight_01", "weight_10", "weight_11",
            "node_covered_by_nww3",
        ):
            if getattr(self, name).shape != shape:
                raise WaveBridgeError(f"NWW3NodeInterpolationWeights 欄位 {name} shape 不一致")


def build_nww3_node_interpolation_weights(
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    nww3_lon: np.ndarray,
    nww3_lat: np.ndarray,
) -> NWW3NodeInterpolationWeights:
    """對每個追蹤網格節點，預先算好 NWW3 規則網格雙線性內插的四角索引與權重。

    這是本模組「開機只算一次」的靜態幾何前處理（見模組 docstring）；呼叫端
    通常只需要呼叫一次（``NativeWaveBridgeReader`` 建構時已經自動呼叫），
    不需要每小時重算。

    Args:
        node_lon, node_lat: ``(nnode,)`` 追蹤網格節點的經緯度（度），即
            ``NativeMesh.node_lon``／``node_lat``。
        nww3_lon, nww3_lat: NWW3 規則網格的經度／緯度軸（度），必須嚴格
            遞增（``forcing.NWWAnalysisMonth`` 已經驗證過這件事，這裡對
            呼叫端直接傳入原始 ``lon.npy``／``lat.npy`` 的情況重複驗證，
            不假設呼叫端一定先過一手 ``NWWAnalysisMonth``）。

    Returns:
        ``NWW3NodeInterpolationWeights``。

    Raises:
        WaveBridgeError: 輸入 shape 不符，或 NWW3 經緯度軸不是嚴格遞增、
            長度不足兩個格點（雙線性內插至少需要頭尾兩個格點才能夾出一段
            區間）。
    """

    lon = np.asarray(node_lon, dtype=np.float64)
    lat = np.asarray(node_lat, dtype=np.float64)
    if lon.ndim != 1 or lat.shape != lon.shape:
        raise WaveBridgeError("node_lon／node_lat 必須是同長度的一維陣列")

    grid_lon = np.asarray(nww3_lon, dtype=np.float64)
    grid_lat = np.asarray(nww3_lat, dtype=np.float64)
    if grid_lon.ndim != 1 or grid_lat.ndim != 1:
        raise WaveBridgeError("nww3_lon／nww3_lat 必須是一維陣列")
    if grid_lon.size < 2 or grid_lat.size < 2:
        raise WaveBridgeError("NWW3 經緯度軸至少需要兩個格點才能雙線性內插")
    if np.any(np.diff(grid_lon) <= 0) or np.any(np.diff(grid_lat) <= 0):
        raise WaveBridgeError("NWW3 經緯度軸必須嚴格遞增")

    finite_node = np.isfinite(lon) & np.isfinite(lat)
    covered = (
        finite_node
        & (lon >= grid_lon[0]) & (lon <= grid_lon[-1])
        & (lat >= grid_lat[0]) & (lat <= grid_lat[-1])
    )

    # 對域外（或非有限）節點也算出一組「安全」的 clip 後座標，只是為了讓
    # searchsorted／索引不會出界或遇到 NaN 比較的未定義行為；這些節點的
    # 實際內插數值永遠不會被使用，因為 gather_nww3_hour_at_nodes 一律先看
    # node_covered_by_nww3，見該函式與模組 docstring「空間涵蓋缺口」一節。
    safe_lon = np.where(finite_node, lon, grid_lon[0])
    safe_lat = np.where(finite_node, lat, grid_lat[0])
    clipped_lon = np.clip(safe_lon, grid_lon[0], grid_lon[-1])
    clipped_lat = np.clip(safe_lat, grid_lat[0], grid_lat[-1])

    x_after = np.clip(np.searchsorted(grid_lon, clipped_lon, side="right"), 1, grid_lon.size - 1)
    y_after = np.clip(np.searchsorted(grid_lat, clipped_lat, side="right"), 1, grid_lat.size - 1)
    x0, x1 = x_after - 1, x_after
    y0, y1 = y_after - 1, y_after

    wx = (clipped_lon - grid_lon[x0]) / (grid_lon[x1] - grid_lon[x0])
    wy = (clipped_lat - grid_lat[y0]) / (grid_lat[y1] - grid_lat[y0])

    return NWW3NodeInterpolationWeights(
        node_y0=y0.astype(np.int64),
        node_y1=y1.astype(np.int64),
        node_x0=x0.astype(np.int64),
        node_x1=x1.astype(np.int64),
        weight_00=(1.0 - wy) * (1.0 - wx),
        weight_01=(1.0 - wy) * wx,
        weight_10=wy * (1.0 - wx),
        weight_11=wy * wx,
        node_covered_by_nww3=covered,
    )


# ── 每小時：向量化收集四角資料＋加權，全程沒有逐節點 Python 迴圈 ───────────


def gather_nww3_hour_at_nodes(
    month: NWWAnalysisMonth,
    time_index: int,
    weights: NWW3NodeInterpolationWeights,
) -> WaveFieldAtNodes:
    """對 NWW3 月份的單一時間列，向量化內插出全部節點的波浪場。

    這是本模組「每小時只做一次向量化收集＋加權」的核心（見模組 docstring）：
    對 ``month`` 的四個原始欄位分別用 ``weights`` 裡的整數索引陣列做花式
    索引（一次取出全部節點的四個角點值，不逐節點迴圈），再套用固定權重與
    「四角全部有效」遮罩政策、方向圓平均與物理防呆（見模組 docstring對應
    章節），組成 ``WaveFieldAtNodes``。

    Args:
        month: 已載入的 ``NWWAnalysisMonth``；本函式只讀取其
            ``significant_wave_height``／``peak_frequency``／
            ``peak_direction_raw_deg``／``valid_mask_wave``／``qc_flags``
            五個公開陣列屬性，不呼叫它的任何方法。
        time_index: ``month.time_utc_ns`` 裡的整數列索引（呼叫端已確認精確
            對齊目標 UTC，見 ``NativeWaveBridgeReader.resolve``），本函式
            不重新驗證這個索引對應哪個 UTC。
        weights: ``build_nww3_node_interpolation_weights`` 的輸出，節點順序
            必須與呼叫端期待的追蹤網格節點順序一致。

    Returns:
        ``WaveFieldAtNodes``，``(nnode,)``。無效節點的
        ``significant_wave_height_m``／``peak_frequency_hz``／
        ``direction_raw_deg`` 固定為 ``NaN``（不是 0），``valid_mask_wave``
        為假；``qc_flags`` 一律回傳四角聯集，即使節點本身被判定無效（供
        QC 稽核，見模組 docstring）。
    """

    y0, y1 = weights.node_y0, weights.node_y1
    x0, x1 = weights.node_x0, weights.node_x1
    w00, w01, w10, w11 = weights.weight_00, weights.weight_01, weights.weight_10, weights.weight_11

    hs_hour = month.significant_wave_height[time_index]
    fp_hour = month.peak_frequency[time_index]
    dir_hour = month.peak_direction_raw_deg[time_index]
    valid_hour = month.valid_mask_wave[time_index]
    qc_hour = month.qc_flags[time_index]

    def _corners(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return field[y0, x0], field[y0, x1], field[y1, x0], field[y1, x1]

    hs00, hs01, hs10, hs11 = (np.asarray(v, dtype=np.float64) for v in _corners(hs_hour))
    fp00, fp01, fp10, fp11 = (np.asarray(v, dtype=np.float64) for v in _corners(fp_hour))
    dir00, dir01, dir10, dir11 = (np.asarray(v, dtype=np.float64) for v in _corners(dir_hour))
    valid00, valid01, valid10, valid11 = (np.asarray(v, dtype=bool) for v in _corners(valid_hour))
    qc00, qc01, qc10, qc11 = (np.asarray(v, dtype=np.int64) for v in _corners(qc_hour))

    corner_valid = valid00 & valid01 & valid10 & valid11
    finite_ok = (
        np.isfinite(hs00) & np.isfinite(hs01) & np.isfinite(hs10) & np.isfinite(hs11)
        & np.isfinite(fp00) & np.isfinite(fp01) & np.isfinite(fp10) & np.isfinite(fp11)
        & np.isfinite(dir00) & np.isfinite(dir01) & np.isfinite(dir10) & np.isfinite(dir11)
    )

    hs_interp = w00 * hs00 + w01 * hs01 + w10 * hs10 + w11 * hs11
    fp_interp = w00 * fp00 + w01 * fp01 + w10 * fp10 + w11 * fp11

    # 方向：先轉成單位圓上的東／北分量再加權平均，最後才轉回角度，避免在
    # 0/360 度邊界附近直接對角度做線性平均（見模組 docstring）。角度慣例
    # 與 forcing.py 1946-1958 行、stokes_batch.py 完全一致：sin↔東分量、
    # cos↔北分量、atan2(東,北) 取回「正北為 0、順時針為正」的角度。
    theta00 = np.deg2rad(dir00)
    theta01 = np.deg2rad(dir01)
    theta10 = np.deg2rad(dir10)
    theta11 = np.deg2rad(dir11)
    direction_x = (
        w00 * np.sin(theta00) + w01 * np.sin(theta01) + w10 * np.sin(theta10) + w11 * np.sin(theta11)
    )
    direction_y = (
        w00 * np.cos(theta00) + w01 * np.cos(theta01) + w10 * np.cos(theta10) + w11 * np.cos(theta11)
    )
    degenerate_direction = (np.abs(direction_x) + np.abs(direction_y)) <= 1.0e-15

    physics_ok = (hs_interp >= 0.0) & (fp_interp > 0.0) & ~degenerate_direction
    node_valid = weights.node_covered_by_nww3 & corner_valid & finite_ok & physics_ok

    with np.errstate(invalid="ignore"):
        raw_direction_deg = np.degrees(np.arctan2(direction_x, direction_y)) % 360.0

    qc_union = qc00 | qc01 | qc10 | qc11

    return WaveFieldAtNodes(
        significant_wave_height_m=np.where(node_valid, hs_interp, np.nan),
        peak_frequency_hz=np.where(node_valid, fp_interp, np.nan),
        direction_raw_deg=np.where(node_valid, raw_direction_deg, np.nan),
        valid_mask_wave=node_valid,
        qc_flags=qc_union,
    )


# ── 主要讀取器：以流場小時為快取單位，供 run_backward_batch 直接使用 ───────


def _build_canonical_time_index(
    months: Mapping[str, NWWAnalysisMonth]
) -> tuple[np.ndarray, dict[int, tuple[str, int]]]:
    """建立跨月份的 canonical UTC 時間索引，月界重疊時保留後一個月份的列。

    「prefer-last」政策與 ``forcing_bridge._build_canonical_index`` 一致
    （依月份目錄名稱排序後，遇到相同 UTC 採後一個月），這裡獨立重寫是因為
    這兩個模組讀的容器型別不同（``NWWAnalysisMonth`` vs. 內部
    ``_MergedMonth``），沒有共用基礎可以直接呼叫。
    """

    location_by_time: dict[int, tuple[str, int]] = {}
    for month_id in sorted(months):
        month = months[month_id]
        for local_index, utc_ns in enumerate(np.asarray(month.time_utc_ns)):
            location_by_time[int(utc_ns)] = (month_id, int(local_index))
    times = np.asarray(sorted(location_by_time), dtype=np.int64)
    return times, location_by_time


class NativeWaveBridgeReader:
    """把多個月份的 ``nww3_analysis`` memmap 轉成追蹤網格節點的波浪場。

    建構時載入每個月份的 ``NWWAnalysisMonth``（唯讀 memmap，不搬動大型資料）
    並確認全部月份共用同一個 NWW3 經緯度網格（``lon``／``lat`` 完全相同）
    ——這是「開機只算一次內插權重」能夠成立的前提，若不同月份用不同網格
    （理論上不應該發生在同一個 ``nww3_analysis`` 產品線內），本類別會直接
    拒絕建構，而不是靜默地只用第一個月份的網格套用到全部月份。

    Args:
        grid_dir: NWW3 規則網格目錄，含共用的 ``lon.npy``／``lat.npy``（見
            ``forcing.NWWAnalysisMonth.from_directories``）。
        month_dirs: ``{month_id: 月份資料夾路徑}``，每個資料夾含
            ``time_utc_ns.npy`` 與四個逐時波浪欄位（見模組 docstring）。
        node_lon, node_lat: 追蹤網格節點的經緯度（度），即
            ``NativeMesh.node_lon``／``node_lat``；回傳的 ``WaveFieldAtNodes``
            節點順序與這裡傳入的順序一致。

    Raises:
        WaveBridgeError: ``month_dirs`` 為空、任一月份載入失敗，或各月份的
            NWW3 網格彼此不一致。
    """

    def __init__(
        self,
        *,
        grid_dir: str | Path,
        month_dirs: Mapping[str, str | Path],
        node_lon: np.ndarray,
        node_lat: np.ndarray,
        cache_size: int = 2,
    ) -> None:
        if not month_dirs:
            raise WaveBridgeError("month_dirs 不得為空")
        if cache_size < 1:
            raise WaveBridgeError("cache_size 必須是正整數")

        self._months: dict[str, NWWAnalysisMonth] = {}
        for month_id, path in month_dirs.items():
            self._months[month_id] = NWWAnalysisMonth.from_directories(grid_dir, path)

        first_id = next(iter(self._months))
        first = self._months[first_id]
        for month_id, month in self._months.items():
            if not (np.array_equal(month.lon, first.lon) and np.array_equal(month.lat, first.lat)):
                raise WaveBridgeError(
                    f"月份 {month_id} 的 NWW3 經緯度網格與 {first_id} 不同，"
                    "無法沿用同一組預先計算的內插權重"
                )

        self._weights = build_nww3_node_interpolation_weights(node_lon, node_lat, first.lon, first.lat)
        self._times, self._location_by_time = _build_canonical_time_index(self._months)
        if self._times.size == 0:
            raise WaveBridgeError("已載入的 NWW3 月份沒有任何時間格點")

        self._cache_size = int(cache_size)
        self._cache: OrderedDict[int, WaveFieldAtNodes] = OrderedDict()
        self.read_counts: dict[int, int] = {}

    @property
    def weights(self) -> NWW3NodeInterpolationWeights:
        """暴露預先算好的內插權重，供呼叫端／QC 稽核（例如統計涵蓋率）。"""

        return self._weights

    @property
    def available_time_range_utc_ns(self) -> tuple[int, int]:
        """回傳目前已載入月份實際涵蓋的 UTC 起訖（含端點）。"""

        return int(self._times[0]), int(self._times[-1])

    def resolve(self, target_utc_ns: int) -> WaveFieldAtNodes:
        """回傳 ``target_utc_ns`` 這個精確時刻、已內插到追蹤網格節點的波浪場。

        只接受在已載入月份時間軸上能找到位元相等值的查詢；找不到就直接拒絕
        （見模組 docstring「時間」一節），不嘗試找最接近的時刻頂替。

        Raises:
            WaveBridgeError: ``target_utc_ns`` 不是任何已載入月份的整點列。
        """

        target = int(target_utc_ns)
        cached = self._cache.get(target)
        if cached is not None:
            self._cache.move_to_end(target)
            return cached

        location = self._location_by_time.get(target)
        if location is None:
            lo, hi = self.available_time_range_utc_ns
            raise WaveBridgeError(
                f"目標時間 {_iso(target)} 不是任何已載入 NWW3 月份的精確時間列；"
                f"已載入範圍 [{_iso(lo)}, {_iso(hi)}]（範圍內仍可能有缺時刻）。"
                "wave_bridge 要求逐小時精確對齊，不做時間內插或最近值頂替"
            )
        month_id, local_index = location
        result = gather_nww3_hour_at_nodes(self._months[month_id], local_index, self._weights)

        self._cache[target] = result
        self._cache.move_to_end(target)
        self.read_counts[target] = self.read_counts.get(target, 0) + 1
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    def as_wave_field_resolver(self) -> Callable[[int], WaveFieldAtNodes]:
        """回傳可直接傳給 ``run_backward_batch(wave_field_resolver=...)`` 的可呼叫物件。

        就是 ``self.resolve`` 本身（bound method）；獨立提供這個方法只是讓
        呼叫端的整合程式碼意圖更明確（「這是要接到 wave_field_resolver 的
        那個東西」），不代表額外包了一層邏輯。
        """

        return self.resolve
