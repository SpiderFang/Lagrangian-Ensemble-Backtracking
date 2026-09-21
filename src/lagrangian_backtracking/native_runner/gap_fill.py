"""WP-10：OCM／NWW3 長缺口的「相鄰太陰日位移混合」保險填補。

為什麼需要這個模組
--------------------
``ocm_reconstruction.py`` 的正式重建核心只嚴格支援兩種缺口長度：1 小時（雙側線性）
與 23--49 小時（EOF／諧波／AR(2) 狀態空間）。伺服器上游雖然已在補齊 2025 年缺口
（見 ``reports/WP-10.md`` 記載的驗收現況），但正式管線仍可能遇到：

* 上游重建後殘留的、長度落在 1 與 23--49 小時契約之外的 OCM 缺口（例如數天到
  一整個月）。
* NWW3 波浪分析產品的缺時刻（``wave_bridge.py`` 目前遇缺時刻一律直接拒絕，見該
  模組「時間」一節）。

本模組是「保險」：目標是讓正式逆向追蹤運算在遇到這類缺口時**不停止**，而不是
提供比核心重建更精確的科學結果。所有由本模組填補的資料列都會清楚標記方法、
來源時刻與權重，讓下游可以把「觀測」「核心重建」「相鄰日位移混合」三種來源分開
稽核，不假裝任何一種是獨立觀測。

方法：相鄰太陰日位移（25 小時）混合，而非簡單複製前一日
----------------------------------------------------------------
本海域以潮流為主，主要分潮週期（小時）：M2＝12.4206、S2＝12.0、K1＝23.9345、
O1＝25.8193。一個太陰日（潮汐相位重複的自然週期）約 24.84 小時，而不是曆法的
24 小時；若以「同一時刻複製前一個曆法日」（位移 24 小時）填補，每天會使 M2 相位
偏移約 24.0/12.4206 個週期的小數部分 ×360°，換算時間誤差約 50 分鐘／天，且逐日
累積。改以位移 25 小時取樣，每個分潮的相位誤差（｜25 對週期取模｜換算成時間，
再取與 0／週期的較近距離）遠小於 24 小時位移，詳細數字見下表（由
``lunar_day_shift_phase_error_report`` 產生，同一組數字也寫在
``reports/WP-10.md``）：

===== ===== ============ ============ ==================== ====================
分潮  週期(h) 24h 相位誤差(h) 25h 相位誤差(h) 24h 相位誤差(度) 25h 相位誤差(度)
===== ===== ============ ============ ==================== ====================
M2    12.4206   0.8412       0.1588        24.38°              4.60°
S2    12.0000   0.0000       1.0000         0.00°              30.00°
K1    23.9345   0.0655       1.0655         0.99°              16.03°
O1    25.8193   1.8193       0.8193        25.37°              11.42°
===== ===== ============ ============ ==================== ====================

（上表由 ``lunar_day_shift_phase_error_report()`` 直接計算得出，非手工估算；
可重新執行該函式核對。）S2／O1 在單一分潮上不一定是 25 小時嚴格佔優（S2 的週期
恰好是 24 小時的整數倍之半，複製前一天在 S2 上誤差為 0）；但本專案潮汐能量以
M2 為主導分潮，且 25 小時位移在「四個主要分潮相位誤差的均方根」上明顯優於 24
小時位移（25 小時位移 0.841 小時 vs. 24 小時位移 1.003 小時，見
``lunar_day_shift_phase_error_report`` 的 ``rms_phase_error_hours`` 欄位與
``validate_gap_fill_blocked`` 的 synthetic 交叉驗證），這是本模組選擇 25 小時、
而非逐分潮各自最佳位移的理由：後者需要對每個分潮分別位移再合成，複雜度與
「保險、求快」的設計目標衝突。

雙側來源與漸變接合
--------------------
缺口內每個小時 ``t``，各自從缺口前方找「最小的 k，使 ``t - k*25h`` 落在缺口外
且確實是觀測時刻」、從缺口後方找「最小的 k，使 ``t + k*25h`` 落在缺口外且確實是
觀測時刻」（``_search_shift_candidates``）。若只有一側找得到來源，只用那一側並
標記 ``QUALITY_SINGLE_SIDED_BLEND``。兩側都找到時，依 ``t`` 在缺口內的相對位置
做連續權重混合（越靠近缺口前緣，前方來源權重越高；越靠近後緣，後方權重越高），
避免缺口兩端與觀測資料銜接處出現流場跳變。

與既有讀取端的相容策略（重要限制，誠實記錄）
------------------------------------------------
``forcing.OCMReconstructionPatchMonth.__post_init__`` 目前寫死
``origin_code`` 只能是 1 或 2（見該模組原始碼），且本工作包**不得修改**
``forcing.py``。因此本模組**沒有**依原始需求新增第三個 ``origin_code`` 數值，
而是：

* 相鄰日位移混合列一律回填 ``origin_code=ORIGIN_RECONSTRUCTED_STATE_SPACE``
  （2，語意最接近的既有分類：多來源、含不確定度的預測，而非精確雙側線性）。
* 額外在 ``quality_flags`` 加入本模組專屬、目前未被任何既有程式碼使用的新位元
  ``QUALITY_LUNAR_DAY_BLEND``(128) 與 ``QUALITY_SINGLE_SIDED_BLEND``(256)，
  讓下游可以用 bitwise AND 分辨「這列其實是相鄰日位移混合，不是真正的狀態空間
  模型」。``forcing.OCMReconstructionPatchMonth.row_is_valid`` 只檢查 bit 64，
  新位元不影響既有 QC fail-closed 邏輯。
* 每個月 patch 的 ``metadata.json`` 在既有欄位之外新增 ``gap_fill`` 物件，逐列
  記錄實際來源時刻、位移倍數 k 與混合權重；新增鍵不影響
  ``forcing_window._ReconstructionPatchStore``／
  ``ocm_reconstruction.validate_reconstruction_patch`` 的既有檢查（兩者都只用
  ``.get()`` 讀取已知欄位，或只檢查 ``arrays`` 子集合），checksum 也已同步更新
  （見 ``_augment_month_metadata``）。

NWW3 沒有 sparse patch 概念：輸出「補齊後的完整月份」
----------------------------------------------------------
``wave_bridge.NativeWaveBridgeReader``／``forcing.NWWAnalysisMonth`` 目前**不**
支援 sparse patch，遇缺時刻直接拒絕（見 ``wave_bridge.py`` 模組文件「時間」一節），
且本工作包不得修改 ``wave_bridge.py``。因此 ``build_wave_gap_fill_patches`` 改
輸出一份「與原始 ``nww3_analysis`` schema 完全相同格式的完整替代月份」：有缺口
的月份重寫成含原始列＋新填補列的完整逐時月檔，其餘月份與 ``grid`` 目錄以符號連結
指向原始資料（不複製大型陣列）。呼叫端只需要把 ``NWW_ANALYSIS_ROOT``（或等效
設定）指向本函式的 ``output_root``，``NWWAnalysisMonth.from_directories`` 完全
不需要修改就能讀取（符號連結對 ``numpy.load(..., mmap_mode="r")`` 透明）。

**關鍵限制（務必注意，已在此明確記錄）**：``NWWAnalysisMonth`` 的
``qc_flags`` 陣列會被 ``native_runner.stokes_batch.batch_finite_depth_stokes``
直接檢查——只要 ``qc_flags[i] != 0`` 就會判定為
``STOKES_BATCH_STATUS_WAVE_QC_FLAGGED`` 而捨棄該筆 Stokes 貢獻（見
``stokes_batch.py`` 原始碼），**不是**單純的診斷位元。因此本模組**不會**在
``qc_flags`` 裡加註「這是相鄰日位移混合」的旗標——那樣會讓所有補值列在下游被
當成無效資料，違背「不停止」的設計目標。``qc_flags`` 只保留來源列本身既有的
QC 聯集（乾淨來源聯集出來就是 0）；「這一列是怎麼來的」改記錄在本模組寫出的
``metadata.json``（不影響 ``NWWAnalysisMonth`` 讀取，因為它只認欄位陣列檔名）。

單位、缺值與物理假設
----------------------
所有流速、垂向座標、擴散係數與 NWW3 欄位延用來源產品既有單位與慣例（公尺制、
m²/s、度），本模組不重新定義。混合結果的 NaN／缺值語意與 ``ocm_reconstruction``
一致：局部格點缺值以 ``NaN`` 保存，不得以 0 或最近值頂替；``wetdry_elem`` 不做
內插，只取權重較大一側的實際觀測值（沿用 ``ocm_reconstruction._enforce_constraints``
的乾濕與 ``zcor`` 垂向嚴格遞增語意，兩者共用同一段既有程式碼，本模組不重寫）。
本模組產生的「條件式來源足跡」／相對權重僅供工程保險使用，不構成獨立科學驗證；
真正的準確度評估見 ``validate_gap_fill_blocked`` 與 ``reports/WP-10.md``。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..ocm_reconstruction import (
    CONTINUOUS_FIELD_NAMES,
    HOURLY_NS,
    ORIGIN_RECONSTRUCTED_STATE_SPACE,
    QUALITY_NONFINITE_CELL,
    RECONSTRUCTION_ARRAY_NAMES,
    ArraySequenceSource,
    GapSpec,
    NpyDomainSource,
    ReconstructionConfig,
    ReconstructionError,
    SequenceSource,
    discover_hourly_gaps,
    sha256_file,
    validate_reconstruction_patch,
)
from ..ocm_reconstruction import _baseline_predictions as _core_baseline_predictions
from ..ocm_reconstruction import _canonical_json_bytes as _core_canonical_json_bytes
from ..ocm_reconstruction import _corrcoef as _core_corrcoef
from ..ocm_reconstruction import _enforce_constraints as _core_enforce_constraints
from ..ocm_reconstruction import _month_for_ns as _core_month_for_ns
from ..ocm_reconstruction import _reconstruct_gap as _core_reconstruct_gap
from ..ocm_reconstruction import _write_atomic_month as _core_write_atomic_month
from ..ocm_reconstruction import _write_domain_manifest as _core_write_domain_manifest

__all__ = [
    "GAP_FILL_METHOD_ID",
    "GAP_FILL_MIXED_METHOD_ID",
    "NWW3_GAP_FILL_METHOD_ID",
    "QUALITY_LUNAR_DAY_BLEND",
    "QUALITY_SINGLE_SIDED_BLEND",
    "GapFillConfig",
    "GapFillReport",
    "GapFillValidationReport",
    "WaveGapFillReport",
    "build_gap_fill_patches",
    "build_wave_gap_fill_patches",
    "lunar_day_shift_phase_error_report",
    "validate_gap_fill_blocked",
    "write_gap_fill_root_index",
]

# ── 常數：方法識別碼與本模組專屬的 quality_flags／origin_code 相容策略 ──────

GAP_FILL_METHOD_ID = "ocm_lunar_day_shift_blend_v1"
GAP_FILL_MIXED_METHOD_ID = "ocm_gap_fill_mixed_v1"
NWW3_GAP_FILL_METHOD_ID = "nww3_lunar_day_shift_blend_v1"
NWW3_GAP_FILL_SCHEMA_VERSION = "nww3_gap_fill_month_v1"

# 見模組 docstring「與既有讀取端的相容策略」：origin_code 被 forcing.py 寫死只能
# 是 1 或 2，因此相鄰日混合列固定回填 2，真正方法別由下列新位元在 quality_flags
# 區分；兩個新位元刻意選在既有核心模組（1/2/4/8/16/32/64）之外，不會與任何既有
# 判斷式衝突（既有程式碼只檢查 bit 64，見 forcing.OCMReconstructionPatchMonth）。
ORIGIN_CODE_FOR_LUNAR_DAY_BLEND = ORIGIN_RECONSTRUCTED_STATE_SPACE
QUALITY_LUNAR_DAY_BLEND = np.uint16(128)
QUALITY_SINGLE_SIDED_BLEND = np.uint16(256)

_EXISTING_METHOD_SHORT_LENGTH = 1
_EXISTING_METHOD_STATE_SPACE_MIN = 23
_EXISTING_METHOD_STATE_SPACE_MAX = 49

# 四個主要分潮週期（小時）：M2、S2、K1、O1，與 ``ocm_reconstruction.
# ReconstructionConfig.harmonic_periods_hours`` 預設值完全一致，供相位誤差報表
# 與模組 docstring 的說明表共用同一組數字。
_MAJOR_TIDAL_CONSTITUENTS_HOURS: Mapping[str, float] = {
    "M2": 12.4206,
    "S2": 12.0,
    "K1": 23.9345,
    "O1": 25.8193,
}


# ── 相位誤差報表：證明 25 小時位移優於 24 小時位移（供 docstring 與報告引用） ──


def lunar_day_shift_phase_error_report(
    shift_hours_candidates: Sequence[float] = (24.0, 25.0),
) -> dict[str, Any]:
    """計算每個主要分潮在不同「位移小時數」下的相位誤差，證明 25 小時優於 24 小時。

    一個位移量 ``shift`` 對週期為 ``period`` 的分潮造成的相位誤差，是
    ``shift`` 對 ``period`` 取模後，與「完整週期（相位誤差為 0）」或「0（沒有
    位移）」的最短時間距離：``min(shift mod period, period - shift mod period)``。
    這個值愈小，代表複製／混合該位移量之後的分潮相位愈接近「完全沒有位移」，
    即科學上愈合理。本函式回傳每個候選位移、每個分潮的小時與角度誤差，以及
    四個分潮的均方根（RMS）相位誤差，供呼叫端或 ``reports/WP-10.md`` 直接引用，
    不需要重新計算或憑空列表。
    """

    rows: list[dict[str, Any]] = []
    for shift_hours in shift_hours_candidates:
        squared_errors: list[float] = []
        per_constituent: dict[str, dict[str, float]] = {}
        for name, period_hours in _MAJOR_TIDAL_CONSTITUENTS_HOURS.items():
            remainder = float(shift_hours) % period_hours
            error_hours = min(remainder, period_hours - remainder)
            error_degrees = 360.0 * error_hours / period_hours
            per_constituent[name] = {
                "period_hours": period_hours,
                "phase_error_hours": error_hours,
                "phase_error_degrees": error_degrees,
            }
            squared_errors.append(error_hours * error_hours)
        rows.append(
            {
                "shift_hours": float(shift_hours),
                "constituents": per_constituent,
                "rms_phase_error_hours": float(np.sqrt(np.mean(squared_errors))),
            }
        )
    return {"schema_version": "ocm_gap_fill_phase_error_report_v1", "shift_candidates": rows}


# ── 設定與報告 dataclass ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GapFillConfig:
    """控制相鄰太陰日位移混合的固定設定。

    ``shift_hours`` 必須是整數小時，才能落在既有逐時格點上（見模組 docstring
    的相位誤差表，預設 25）。``max_shift_multiples`` 是每一側最多嘗試的位移
    倍數上限，只是安全上限（實際搜尋一旦跨出來源時間軸範圍就會提早停止，不會
    真的把兩年資料掃過一輪）；``force_lunar_day_blend_only`` 對應使用者要求的
    「全部用相鄰日方法以求最快」選項——啟用時連 1 小時與 23--49 小時缺口都不
    呼叫既有核心方法，直接用位移混合（犧牲一點精度換取單一、簡單、可預期的
    執行路徑）。``reconstruction_config`` 是委派給既有 1 小時／23--49 小時方法
    （``ocm_reconstruction._reconstruct_gap``）與 diffusivity 物理範圍門檻的
    設定，本模組不重新定義這些既有語意。
    """

    shift_hours: int = 25
    max_shift_multiples: int = 2000
    force_lunar_day_blend_only: bool = False
    reconstruction_config: ReconstructionConfig = field(default_factory=ReconstructionConfig)

    def __post_init__(self) -> None:
        """在開始讀取來源前拒絕無法定義穩定搜尋的設定。"""

        if int(self.shift_hours) < 1:
            raise ValueError("shift_hours 必須是正整數小時")
        if int(self.max_shift_multiples) < 1:
            raise ValueError("max_shift_multiples 必須為正整數")


_DEFAULT_GAP_FILL_CONFIG = GapFillConfig()
"""模組層級單例，供三個公開函式的 ``config=`` 預設值共用，避免在函式簽章裡直接
呼叫 ``GapFillConfig()``；``GapFillConfig`` 是 frozen dataclass，跨呼叫共用同一個
實例是安全的，不會有 mutable-default 的副作用風險。"""


@dataclass(frozen=True, slots=True)
class GapFillReport:
    """``build_gap_fill_patches`` 的執行摘要，供 CLI 進度輸出與正式 run manifest 使用。"""

    flow_id: str
    domain_manifest_path: str
    month_ids: tuple[str, ...]
    existing_method_gap_ids: tuple[str, ...]
    blend_method_gap_ids: tuple[str, ...]
    fallback_to_blend_gap_ids: tuple[str, ...]
    single_sided_gap_ids: tuple[str, ...]
    unfilled_time_utc_ns: tuple[int, ...]
    row_count: int


@dataclass(frozen=True, slots=True)
class WaveGapFillReport:
    """``build_wave_gap_fill_patches`` 的執行摘要。"""

    flow_id: str
    output_domain_root: str
    filled_month_ids: tuple[str, ...]
    symlinked_month_ids: tuple[str, ...]
    blended_row_count: int
    single_sided_row_count: int
    unfilled_time_utc_ns: tuple[int, ...]
    manifest_path: str


@dataclass(frozen=True, slots=True)
class GapFillValidationReport:
    """``validate_gap_fill_blocked`` 的遮蔽交叉驗證輸出；只衡量 synthetic 遮蔽表現。"""

    schema_version: str
    block_lengths_hours: tuple[int, ...]
    n_blocks: int
    seed: int
    results: tuple[Mapping[str, Any], ...]
    acceptance: Mapping[str, Any]


# ── 共用工具：位移搜尋、canonical JSON、月份 patch 組裝 ──────────────────


def _search_shift_candidates(
    missing_times_ns: np.ndarray,
    *,
    sign: int,
    observed_sorted: np.ndarray,
    shift_ns: np.int64,
    max_multiples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化搜尋每個缺口時刻沿指定方向、最小位移倍數 k 的可用觀測來源時刻。

    ``sign`` 為 -1 表示往缺口前方（更早）找、+1 表示往缺口後方（更晚）找。對每個
    候選位移量 ``k``（由 1 遞增），檢查候選時刻是否確實是某一列觀測時刻（不是
    另一個缺口或域外時刻）。候選時刻只會沿搜尋方向單調遠離起點：``sign=+1`` 時
    只會愈搜尋愈晚，``sign=-1`` 時只會愈搜尋愈早；因此只有在候選時刻**永久**
    超出觀測時間軸「搜尋方向那一端」的邊界時才提早放棄（例如 ``sign=+1`` 時
    超過觀測軸最晚時刻 ``hi``），不能因為候選時刻暫時還沒追上觀測軸另一端
    （例如 ``sign=+1`` 時還沒到達 ``lo``，這在候選時刻仍落在本缺口或域外「起點
    之前」的區段時很常見）就放棄——那樣會誤判本來在較大 ``k`` 就能找到的來源
    為「找不到」。找不到時對應位置回傳 ``-1``（來源時刻）與 ``0``（位移倍數）。
    """

    missing = np.asarray(missing_times_ns, dtype=np.int64)
    found_time = np.full(missing.shape, -1, dtype=np.int64)
    found_k = np.zeros(missing.shape, dtype=np.int64)
    if missing.size == 0:
        return found_time, found_k
    lo = int(observed_sorted[0])
    hi = int(observed_sorted[-1])
    remaining = np.ones(missing.shape, dtype=bool)
    for k in range(1, int(max_multiples) + 1):
        if not np.any(remaining):
            break
        candidate = missing + np.int64(sign) * np.int64(k) * shift_ns
        within_bounds = (candidate >= lo) & (candidate <= hi)
        check = remaining & within_bounds
        if np.any(check):
            idx = np.flatnonzero(check)
            present = np.isin(candidate[idx], observed_sorted)
            resolved = idx[present]
            if resolved.size:
                found_time[resolved] = candidate[resolved]
                found_k[resolved] = k
        permanently_overshot = (candidate > hi) if sign > 0 else (candidate < lo)
        remaining &= ~permanently_overshot
        remaining &= found_time < 0
    return found_time, found_k


def _continuous_blend_weights(row_count: int) -> np.ndarray:
    """回傳缺口內每列「前方權重」的連續斜坡，靠近前緣趨近 1、靠近後緣趨近 0。

    位置以 ``(index+1)/(row_count+1)`` 表示，永遠落在開區間 ``(0,1)``（不含端點
    本身），確保缺口邊界不會出現權重的不連續跳變；``1 - weight_before`` 即後方
    權重。
    """

    position = (np.arange(row_count, dtype=np.float64) + 1.0) / (row_count + 1.0)
    return 1.0 - position


def _month_for_time(time_ns: int) -> str:
    """薄包裝 ``ocm_reconstruction._month_for_ns``，讓本模組其他函式呼叫風格一致。"""

    return _core_month_for_ns(int(time_ns))


def write_gap_fill_root_index(output_root: str | Path, flow_ids: Sequence[str]) -> dict[str, Any]:
    """把已發布的多個 flow domain manifest 組成一份 root index。

    格式沿用 ``scripts/build_ocm_reconstruction.py`` 的 ``_write_index``（本函式
    不匯入該腳本，避免耦合命令列參數解析，但採用完全相同的 JSON schema），讓
    ``forcing_window._ReconstructionPatchStore._validate_root_index`` 不需修改就
    能接受。呼叫端通常在對每個 flow domain 呼叫過 ``build_gap_fill_patches``
    之後，一次呼叫本函式發布共用 index。
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    domains: list[dict[str, Any]] = []
    for flow_id in sorted(set(flow_ids)):
        manifest_path = root / flow_id / "reconstruction-manifest.json"
        if not manifest_path.is_file():
            raise ReconstructionError(f"flow_id={flow_id} 缺少已發布的 domain manifest：{manifest_path}")
        domains.append(
            {
                "flow_id": flow_id,
                "manifest_path": f"{flow_id}/reconstruction-manifest.json",
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    document = {"schema_version": "ocm_reconstruction_index_v1", "domains": domains}
    data = _core_canonical_json_bytes(document)
    index_path = root / "reconstruction-index.json"
    index_path.write_bytes(data)
    (root / "reconstruction-index.json.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n", encoding="ascii"
    )
    return document


def _augment_month_metadata(
    month_dir: Path,
    metadata: Mapping[str, Any],
    *,
    row_provenance: Sequence[Mapping[str, Any]],
    shift_hours: int,
) -> dict[str, Any]:
    """在既有 ``_write_atomic_month`` 已發布的月份上，補寫逐列 provenance。

    只新增 ``gap_fill`` 這個頂層鍵，不刪改任何既有欄位；重寫 ``metadata.json``
    與其 sha256 sidecar，讓兩者與回傳值保持位元一致，供
    ``_write_domain_manifest``／``validate_reconstruction_patch`` 之後重新計算
    checksum 時仍然吻合（見模組 docstring「與既有讀取端的相容策略」）。
    """

    augmented = dict(metadata)
    augmented["gap_fill"] = {
        "schema_version": "ocm_gap_fill_row_provenance_v1",
        "shift_hours": int(shift_hours),
        "row_provenance": list(row_provenance),
    }
    metadata_bytes = _core_canonical_json_bytes(augmented)
    (month_dir / "metadata.json").write_bytes(metadata_bytes)
    (month_dir / "metadata.json.sha256").write_text(
        hashlib.sha256(metadata_bytes).hexdigest() + "\n", encoding="ascii"
    )
    return augmented


# ── OCM：相鄰太陰日位移混合核心 ───────────────────────────────────────────


def _lunar_day_shift_blend_gap(
    source: SequenceSource,
    gap: GapSpec,
    config: GapFillConfig,
    *,
    observed_sorted: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """對一段任意長度的缺口，以相鄰太陰日位移混合重建全部 patch 陣列。

    回傳值第一項與 ``ocm_reconstruction._reconstruct_gap`` 完全同型的 arrays
    dict（缺少任何一側來源的列會被整列剔除，不勉強發布不可信的 row），第二項
    是逐列 provenance（來源時刻、位移倍數、權重），供呼叫端寫入
    ``metadata.json``。
    """

    missing = np.asarray(gap.missing_times_ns, dtype=np.int64)
    shift_ns = np.int64(int(config.shift_hours)) * HOURLY_NS
    left_time, left_k = _search_shift_candidates(
        missing, sign=-1, observed_sorted=observed_sorted, shift_ns=shift_ns,
        max_multiples=config.max_shift_multiples,
    )
    right_time, right_k = _search_shift_candidates(
        missing, sign=1, observed_sorted=observed_sorted, shift_ns=shift_ns,
        max_multiples=config.max_shift_multiples,
    )
    have_left = left_time >= 0
    have_right = right_time >= 0
    keep = have_left | have_right
    if not np.any(keep):
        return (
            {name: np.empty((0,), dtype=np.int64) if name == "time_utc_ns" else np.empty((0,))
             for name in RECONSTRUCTION_ARRAY_NAMES},
            [],
        )
    kept_missing = missing[keep]
    kept_have_left = have_left[keep]
    kept_have_right = have_right[keep]
    kept_left_time = left_time[keep]
    kept_right_time = right_time[keep]
    kept_left_k = left_k[keep]
    kept_right_k = right_k[keep]

    weight_before_full = _continuous_blend_weights(missing.size)[keep]
    both = kept_have_left & kept_have_right
    weight_left = np.where(both, weight_before_full, np.where(kept_have_left, 1.0, 0.0))
    weight_right = np.where(both, 1.0 - weight_before_full, np.where(kept_have_right, 1.0, 0.0))

    left_source_times = {int(v) for v in kept_left_time[kept_have_left]}
    right_source_times = {int(v) for v in kept_right_time[kept_have_right]}
    unique_times = np.asarray(sorted(left_source_times | right_source_times), dtype=np.int64)
    fallback_time = int(unique_times[0])
    left_pos = np.searchsorted(unique_times, np.where(kept_have_left, kept_left_time, fallback_time))
    right_pos = np.searchsorted(unique_times, np.where(kept_have_right, kept_right_time, fallback_time))

    kept_n = int(kept_missing.size)
    arrays: dict[str, np.ndarray] = {
        "time_utc_ns": kept_missing.copy(),
        "origin_code": np.full(kept_n, ORIGIN_CODE_FOR_LUNAR_DAY_BLEND, dtype=np.uint8),
    }
    quality = np.full(kept_n, QUALITY_LUNAR_DAY_BLEND, dtype=np.uint16)
    single_sided = kept_have_left ^ kept_have_right
    quality = np.where(single_sided, quality | QUALITY_SINGLE_SIDED_BLEND, quality).astype(np.uint16)

    for field_name in CONTINUOUS_FIELD_NAMES:
        shape = source.field_shapes[field_name]
        feature_count = int(np.prod(shape, dtype=np.int64))
        batch = source.read_times(unique_times, field_name).astype(np.float64)
        left_vals = np.where(kept_have_left[:, None], batch[left_pos], 0.0)
        right_vals = np.where(kept_have_right[:, None], batch[right_pos], 0.0)
        blended = weight_left[:, None] * left_vals + weight_right[:, None] * right_vals
        shaped = blended.reshape((kept_n, *shape))
        if field_name == "diffusivity":
            empty_training = np.empty((0, feature_count))
            left_training = (
                batch[np.unique(left_pos[kept_have_left])] if np.any(kept_have_left) else empty_training
            )
            right_training = (
                batch[np.unique(right_pos[kept_have_right])] if np.any(kept_have_right) else empty_training
            )
            training = np.concatenate((left_training, right_training), axis=0)
            flat, constraint_flags = _core_enforce_constraints(
                field_name, blended, source, config.reconstruction_config,
                flat_start=0, flat_stop=feature_count, training_values=training,
            )
            shaped = flat.reshape((kept_n, *shape))
            quality |= constraint_flags
        elif field_name == "zcor":
            shaped, constraint_flags = _core_enforce_constraints(
                field_name, shaped, source, config.reconstruction_config,
                flat_start=0, flat_stop=feature_count,
            )
            quality |= constraint_flags
        else:
            if not np.isfinite(shaped).all():
                quality |= QUALITY_NONFINITE_CELL
        arrays[field_name] = shaped.astype(source.field_dtypes[field_name], copy=False)

    wet_shape = source.field_shapes["wetdry_elem"]
    batch_wet = source.read_times(unique_times, "wetdry_elem").astype(np.float64)
    pick_left = kept_have_left & (weight_left >= weight_right)
    selected_pos = np.where(pick_left, left_pos, right_pos)
    wet_flat = batch_wet[selected_pos].astype(np.float32)
    arrays["wetdry_elem"] = wet_flat.reshape((kept_n, *wet_shape))
    arrays["quality_flags"] = quality

    row_provenance: list[dict[str, Any]] = []
    for i in range(kept_n):
        row_provenance.append(
            {
                "time_utc_ns": int(kept_missing[i]),
                "method": GAP_FILL_METHOD_ID,
                "left_source_time_utc_ns": int(kept_left_time[i]) if kept_have_left[i] else None,
                "right_source_time_utc_ns": int(kept_right_time[i]) if kept_have_right[i] else None,
                "left_shift_multiples": int(kept_left_k[i]) if kept_have_left[i] else None,
                "right_shift_multiples": int(kept_right_k[i]) if kept_have_right[i] else None,
                "left_weight": float(weight_left[i]),
                "right_weight": float(weight_right[i]),
                "single_sided": bool(single_sided[i]),
            }
        )
    return arrays, row_provenance


def build_gap_fill_patches(
    ocm_native_domain_root: str | Path,
    output_root: str | Path,
    *,
    gaps: Sequence[GapSpec] | None = None,
    start_utc_ns: int | None = None,
    end_utc_ns: int | None = None,
    config: GapFillConfig = _DEFAULT_GAP_FILL_CONFIG,
) -> GapFillReport:
    """建立一個 OCM flow domain 的保險缺口填補 patch，輸出格式與核心重建 patch 相容。

    缺口長度為 1 小時或 23--49 小時時優先呼叫既有 ``ocm_reconstruction.
    _reconstruct_gap``（不重寫核心演算法）；若既有方法因雙側支援不足而失敗，
    或 ``config.force_lunar_day_blend_only`` 為真，或長度落在既有契約之外，一律
    改用相鄰太陰日位移混合。單一月份若同時含兩種方法的列，仍只發布一份合併後
    的 immutable patch（一個月只能有一個 ``OCMReconstructionPatchMonth``，見
    ``forcing_bridge`` 的 ``patches: Mapping[str, OCMReconstructionPatchMonth]``
    契約）。任一來源都找不到的列不會被勉強寫入 patch，保留為真正的未登錄缺口，
    由呼叫端另行走 gap-safe arrival window（見 AGENTS.md 資料契約）。

    Args:
        ocm_native_domain_root: 已驗收 ``ocm_native`` 單一 flow domain 根目錄
            （``<flow_id>/``，其下有 ``grid``／``months``），與
            ``ocm_reconstruction.NpyDomainSource`` 的輸入相同。
        output_root: patch 輸出根目錄；本函式只在
            ``output_root/<flow_id>/months/YYYYMM`` 尚未存在時才會寫入。
        gaps: 已知缺口列表；未提供時以 ``discover_hourly_gaps`` 搭配
            ``start_utc_ns``／``end_utc_ns`` 自動掃描。
        start_utc_ns, end_utc_ns: 這個 flow domain 應涵蓋的正式時間範圍（UTC
            奈秒），用來偵測資料起訖的 edge gap；語意與
            ``ocm_reconstruction.discover_hourly_gaps`` 的
            ``expected_start_ns``／``expected_end_ns`` 完全相同。

    Returns:
        ``GapFillReport``。

    Raises:
        ReconstructionError: 來源網格／時間軸不合法、輸出目的地已存在，或全部
            缺口都完全沒有可用的雙向來源。
    """

    root = Path(output_root)
    source = NpyDomainSource(ocm_native_domain_root)
    flow_id = source.flow_id
    observed_sorted = source.times_utc_ns

    resolved_gaps = (
        tuple(gaps)
        if gaps is not None
        else discover_hourly_gaps(
            observed_sorted, flow_id=flow_id, expected_start_ns=start_utc_ns, expected_end_ns=end_utc_ns
        )
    )
    if not resolved_gaps:
        return GapFillReport(
            flow_id=flow_id, domain_manifest_path="", month_ids=(), existing_method_gap_ids=(),
            blend_method_gap_ids=(), fallback_to_blend_gap_ids=(), single_sided_gap_ids=(),
            unfilled_time_utc_ns=(), row_count=0,
        )

    months_needed = sorted({_month_for_time(t) for gap in resolved_gaps for t in gap.missing_times_ns})
    for month in months_needed:
        destination = root / flow_id / "months" / month
        if destination.exists():
            raise ReconstructionError(f"gap-fill patch 目的地已存在，拒絕覆寫：{destination}")

    by_month: dict[str, dict[str, list[Any]]] = {}
    by_month_methods: dict[str, set[str]] = {}
    provenance_by_time: dict[int, dict[str, Any]] = {}
    existing_ids: set[str] = set()
    blend_ids: set[str] = set()
    fallback_ids: set[str] = set()
    single_sided_ids: set[str] = set()
    unfilled_times: list[int] = []

    for gap in resolved_gaps:
        length = int(gap.missing_times_ns.size)
        qualifies_for_existing = (
            length == _EXISTING_METHOD_SHORT_LENGTH
            or _EXISTING_METHOD_STATE_SPACE_MIN <= length <= _EXISTING_METHOD_STATE_SPACE_MAX
        )
        arrays: dict[str, np.ndarray] | None = None
        method_tag = ""
        if qualifies_for_existing and not config.force_lunar_day_blend_only:
            try:
                arrays = _core_reconstruct_gap(source, gap, config.reconstruction_config)
                method_tag = "existing"
                existing_ids.add(gap.gap_id)
                for t in np.asarray(arrays["time_utc_ns"], dtype=np.int64):
                    provenance_by_time[int(t)] = {
                        "time_utc_ns": int(t),
                        "method": config.reconstruction_config.method_id,
                    }
            except ReconstructionError:
                fallback_ids.add(gap.gap_id)
        if arrays is None:
            arrays, row_provenance = _lunar_day_shift_blend_gap(
                source, gap, config, observed_sorted=observed_sorted
            )
            kept_times = {int(t) for t in np.asarray(arrays["time_utc_ns"], dtype=np.int64)}
            unfilled_times.extend(
                int(t) for t in gap.missing_times_ns if int(t) not in kept_times
            )
            if arrays["time_utc_ns"].size == 0:
                continue
            method_tag = "blend"
            blend_ids.add(gap.gap_id)
            for entry in row_provenance:
                provenance_by_time[entry["time_utc_ns"]] = entry
                if entry["single_sided"]:
                    single_sided_ids.add(gap.gap_id)

        for row, time in enumerate(np.asarray(arrays["time_utc_ns"], dtype=np.int64)):
            month = _month_for_time(int(time))
            record = by_month.setdefault(month, {name: [] for name in RECONSTRUCTION_ARRAY_NAMES})
            for name in RECONSTRUCTION_ARRAY_NAMES:
                record[name].append(np.asarray(arrays[name])[row])
            record.setdefault("_gap_ids", []).append(gap.gap_id)
            by_month_methods.setdefault(month, set()).add(method_tag)

    if not by_month:
        raise ReconstructionError(
            f"flow_id={flow_id} 的全部缺口在相鄰太陰日方法下都完全沒有可用來源，未產生任何 patch row"
        )

    month_metadata: list[Mapping[str, Any]] = []
    total_rows = 0
    for month in sorted(by_month):
        record = by_month[month]
        arrays = {
            name: (
                np.asarray(record[name], dtype=np.int64)
                if name == "time_utc_ns"
                else np.stack(record[name], axis=0)
            )
            for name in RECONSTRUCTION_ARRAY_NAMES
        }
        order = np.argsort(arrays["time_utc_ns"], kind="stable")
        arrays = {name: value[order] for name, value in arrays.items()}
        methods_used = by_month_methods[month]
        if methods_used == {"existing"}:
            month_config = config.reconstruction_config
        elif methods_used == {"blend"}:
            month_config = replace(config.reconstruction_config, method_id=GAP_FILL_METHOD_ID)
        else:
            month_config = replace(config.reconstruction_config, method_id=GAP_FILL_MIXED_METHOD_ID)
        metadata = _core_write_atomic_month(
            root, flow_id, month, arrays,
            gap_ids=sorted(set(record["_gap_ids"])),
            config=month_config,
            source_fingerprint=source.source_fingerprint,
        )
        row_provenance_list = [
            provenance_by_time.get(int(t), {"time_utc_ns": int(t), "method": "unknown"})
            for t in arrays["time_utc_ns"]
        ]
        month_dir = root / flow_id / "months" / month
        augmented = _augment_month_metadata(
            month_dir, metadata, row_provenance=row_provenance_list, shift_hours=config.shift_hours
        )
        month_metadata.append(augmented)
        total_rows += int(arrays["time_utc_ns"].size)

    overall_methods = {tag for methods in by_month_methods.values() for tag in methods}
    if overall_methods == {"existing"}:
        domain_config = config.reconstruction_config
    elif overall_methods == {"blend"}:
        domain_config = replace(config.reconstruction_config, method_id=GAP_FILL_METHOD_ID)
    else:
        domain_config = replace(config.reconstruction_config, method_id=GAP_FILL_MIXED_METHOD_ID)
    _core_write_domain_manifest(
        root, flow_id, month_metadata,
        config=domain_config,
        source_fingerprint=source.source_fingerprint,
        mesh_fingerprint=source.mesh_fingerprint,
    )
    validate_reconstruction_patch(root / flow_id)

    return GapFillReport(
        flow_id=flow_id,
        domain_manifest_path=str(root / flow_id / "reconstruction-manifest.json"),
        month_ids=tuple(sorted(by_month)),
        existing_method_gap_ids=tuple(sorted(existing_ids)),
        blend_method_gap_ids=tuple(sorted(blend_ids)),
        fallback_to_blend_gap_ids=tuple(sorted(fallback_ids)),
        single_sided_gap_ids=tuple(sorted(single_sided_ids)),
        unfilled_time_utc_ns=tuple(sorted(set(unfilled_times))),
        row_count=total_rows,
    )


# ── NWW3：完整替代月份 ────────────────────────────────────────────────────

_NWW3_FIELD_NAMES = (
    "significant_wave_height",
    "peak_frequency",
    "peak_direction_raw_deg",
    "valid_mask_wave",
    "qc_flags",
)


def _scan_nww3_domain(domain_root: Path) -> tuple[tuple[str, ...], np.ndarray, dict[int, tuple[str, int]]]:
    """掃描 NWW3 domain 的全部月份，建立跨月份 canonical UTC 索引（prefer-last）。"""

    months_root = domain_root / "months"
    if not months_root.is_dir():
        raise ReconstructionError(f"找不到 NWW3 months 目錄：{months_root}")
    month_ids = sorted(p.name for p in months_root.iterdir() if p.is_dir())
    if not month_ids:
        raise ReconstructionError(f"NWW3 domain 沒有任何月份：{months_root}")
    location_by_time: dict[int, tuple[str, int]] = {}
    for month_id in month_ids:
        time_path = months_root / month_id / "time_utc_ns.npy"
        if not time_path.is_file():
            raise ReconstructionError(f"NWW3 月份缺少 time_utc_ns：{time_path}")
        times = np.asarray(np.load(time_path, mmap_mode="r"), dtype=np.int64)
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0):
            raise ReconstructionError(f"NWW3 月份 UTC 軸不合法：{month_id}")
        for row, t in enumerate(times):
            location_by_time[int(t)] = (month_id, int(row))
    times_sorted = np.asarray(sorted(location_by_time), dtype=np.int64)
    return tuple(month_ids), times_sorted, location_by_time


class _NWW3MonthCache:
    """月份陣列的小型 lazy memmap 快取，避免同一個月份被重複 ``np.load``。"""

    def __init__(self, months_root: Path) -> None:
        self._months_root = months_root
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def arrays(self, month_id: str) -> dict[str, np.ndarray]:
        cached = self._cache.get(month_id)
        if cached is None:
            directory = self._months_root / month_id
            cached = {
                name: np.load(directory / f"{name}.npy", mmap_mode="r")
                for name in ("time_utc_ns", *_NWW3_FIELD_NAMES)
            }
            self._cache[month_id] = cached
        return cached


def _blend_wave_hour(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    weight_left: float,
    weight_right: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """對單一缺時刻的整個 NWW3 規則網格做向量化混合。

    每個 tuple 是 ``(hs, fp, dir_deg, valid, qc)``，形狀皆為 ``(nlat, nlon)``。
    波向先轉 sin／cos 分量加權，再用 ``atan2`` 合成，沿用 ``wave_bridge.
    gather_nww3_hour_at_nodes`` 的既有慣例（正北為 0、順時針為正）。有效遮罩在
    雙側都可用時取交集（任一側無效即無效，即「無效遮罩取聯集」）；只有單側
    可用時直接沿用該側遮罩。``qc_flags`` 一律是來源列的聯集，不額外加註本模組
    的旗標（見模組 docstring 的「關鍵限制」一節）。
    """

    if left is not None and right is not None:
        hs_l, fp_l, dir_l, valid_l, qc_l = left
        hs_r, fp_r, dir_r, valid_r, qc_r = right
        hs = weight_left * hs_l + weight_right * hs_r
        fp = weight_left * fp_l + weight_right * fp_r
        theta_l = np.deg2rad(dir_l)
        theta_r = np.deg2rad(dir_r)
        dx = weight_left * np.sin(theta_l) + weight_right * np.sin(theta_r)
        dy = weight_left * np.cos(theta_l) + weight_right * np.cos(theta_r)
        direction = np.degrees(np.arctan2(dx, dy)) % 360.0
        valid = valid_l & valid_r
        qc = (qc_l | qc_r).astype(np.uint16)
        return hs, fp, direction, valid, qc
    single = left if left is not None else right
    if single is None:
        raise ReconstructionError("_blend_wave_hour 需要至少一側來源")
    return single


def build_wave_gap_fill_patches(
    nww3_domain_root: str | Path,
    output_root: str | Path,
    *,
    gaps: Sequence[GapSpec] | None = None,
    start_utc_ns: int | None = None,
    end_utc_ns: int | None = None,
    config: GapFillConfig = _DEFAULT_GAP_FILL_CONFIG,
) -> WaveGapFillReport:
    """建立一個 NWW3 flow domain 的保險缺口填補，輸出「完整替代月份」。

    ``wave_bridge.py``／``forcing.NWWAnalysisMonth`` 不支援 sparse patch（見
    模組 docstring），因此本函式對含缺口的月份重寫完整逐時月檔（原始列＋相鄰
    太陰日位移混合填補列），其餘月份與 ``grid`` 目錄一律用符號連結指向原始
    資料，不複製大型陣列。呼叫端把 NWW3 根目錄改指向 ``output_root`` 即可，
    不需修改 ``wave_bridge.py``／``forcing.py``。

    Args:
        nww3_domain_root: 已驗收 ``nww3_analysis`` 單一 flow domain 根目錄
            （``<flow_id>/``，其下有 ``grid``／``months``）。
        output_root: 完整替代 domain 的輸出根目錄。
        gaps: 已知缺口列表；未提供時以跨月份 canonical UTC 軸與
            ``discover_hourly_gaps`` 自動掃描。
        start_utc_ns, end_utc_ns: 語意與 ``build_gap_fill_patches`` 相同。

    Returns:
        ``WaveGapFillReport``。
    """

    domain_root = Path(nww3_domain_root)
    root = Path(output_root)
    flow_id = domain_root.name
    month_ids, times_sorted, location_by_time = _scan_nww3_domain(domain_root)

    resolved_gaps = (
        tuple(gaps)
        if gaps is not None
        else discover_hourly_gaps(
            times_sorted, flow_id=flow_id, expected_start_ns=start_utc_ns, expected_end_ns=end_utc_ns
        )
    )

    output_domain_root = root / flow_id
    output_months_root = output_domain_root / "months"
    output_months_root.mkdir(parents=True, exist_ok=True)
    output_grid_dir = output_domain_root / "grid"
    if not output_grid_dir.exists():
        os.symlink(domain_root / "grid", output_grid_dir, target_is_directory=True)

    if not resolved_gaps:
        for month_id in month_ids:
            link = output_months_root / month_id
            if not link.exists():
                os.symlink(domain_root / "months" / month_id, link, target_is_directory=True)
        manifest_path = _write_wave_gap_fill_manifest(
            output_domain_root, flow_id=flow_id, filled_months=(), symlinked_months=month_ids,
            blended_row_count=0, single_sided_row_count=0, unfilled_time_utc_ns=(),
        )
        return WaveGapFillReport(
            flow_id=flow_id, output_domain_root=str(output_domain_root), filled_month_ids=(),
            symlinked_month_ids=month_ids, blended_row_count=0, single_sided_row_count=0,
            unfilled_time_utc_ns=(), manifest_path=str(manifest_path),
        )

    cache = _NWW3MonthCache(domain_root / "months")
    shift_ns = np.int64(int(config.shift_hours)) * HOURLY_NS

    def _fetch(time_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        month_id, row = location_by_time[int(time_ns)]
        arrays = cache.arrays(month_id)
        return (
            np.asarray(arrays["significant_wave_height"][row], dtype=np.float64),
            np.asarray(arrays["peak_frequency"][row], dtype=np.float64),
            np.asarray(arrays["peak_direction_raw_deg"][row], dtype=np.float64),
            np.asarray(arrays["valid_mask_wave"][row], dtype=bool),
            np.asarray(arrays["qc_flags"][row], dtype=np.uint16),
        )

    filled_rows_by_month: dict[str, dict[str, list[np.ndarray]]] = {}
    blended_row_count = 0
    single_sided_row_count = 0
    unfilled_times: list[int] = []

    for gap in resolved_gaps:
        missing = np.asarray(gap.missing_times_ns, dtype=np.int64)
        left_time, _ = _search_shift_candidates(
            missing, sign=-1, observed_sorted=times_sorted, shift_ns=shift_ns,
            max_multiples=config.max_shift_multiples,
        )
        right_time, _ = _search_shift_candidates(
            missing, sign=1, observed_sorted=times_sorted, shift_ns=shift_ns,
            max_multiples=config.max_shift_multiples,
        )
        weight_before = _continuous_blend_weights(missing.size)
        for i, t in enumerate(missing):
            has_left = bool(left_time[i] >= 0)
            has_right = bool(right_time[i] >= 0)
            if not has_left and not has_right:
                unfilled_times.append(int(t))
                continue
            left_data = _fetch(int(left_time[i])) if has_left else None
            right_data = _fetch(int(right_time[i])) if has_right else None
            if has_left and has_right:
                w_left, w_right = float(weight_before[i]), float(1.0 - weight_before[i])
            elif has_left:
                w_left, w_right = 1.0, 0.0
            else:
                w_left, w_right = 0.0, 1.0
                single_sided_row_count += 1
            if has_left != has_right:
                single_sided_row_count += 0 if (has_left and has_right) else 1
            hs, fp, direction, valid, qc = _blend_wave_hour(left_data, right_data, w_left, w_right)
            month_id = _month_for_time(int(t))
            record = filled_rows_by_month.setdefault(
                month_id, {"time_utc_ns": [], **{name: [] for name in _NWW3_FIELD_NAMES}}
            )
            record["time_utc_ns"].append(np.int64(t))
            record["significant_wave_height"].append(hs)
            record["peak_frequency"].append(fp)
            record["peak_direction_raw_deg"].append(direction)
            record["valid_mask_wave"].append(valid)
            record["qc_flags"].append(qc)
            blended_row_count += 1

    filled_months = tuple(sorted(filled_rows_by_month))
    for month_id in filled_months:
        _write_merged_nww3_month(
            domain_root=domain_root,
            output_months_root=output_months_root,
            month_id=month_id,
            filled=filled_rows_by_month[month_id],
        )
    symlinked_months = tuple(month_id for month_id in month_ids if month_id not in filled_rows_by_month)
    for month_id in symlinked_months:
        link = output_months_root / month_id
        if not link.exists():
            os.symlink(domain_root / "months" / month_id, link, target_is_directory=True)

    manifest_path = _write_wave_gap_fill_manifest(
        output_domain_root, flow_id=flow_id, filled_months=filled_months, symlinked_months=symlinked_months,
        blended_row_count=blended_row_count, single_sided_row_count=single_sided_row_count,
        unfilled_time_utc_ns=tuple(sorted(set(unfilled_times))),
    )

    return WaveGapFillReport(
        flow_id=flow_id, output_domain_root=str(output_domain_root), filled_month_ids=filled_months,
        symlinked_month_ids=symlinked_months, blended_row_count=blended_row_count,
        single_sided_row_count=single_sided_row_count,
        unfilled_time_utc_ns=tuple(sorted(set(unfilled_times))), manifest_path=str(manifest_path),
    )


def _write_merged_nww3_month(
    *, domain_root: Path, output_months_root: Path, month_id: str, filled: Mapping[str, list[np.ndarray]]
) -> None:
    """把原始月份與新填補列合併、排序後寫成完整替代月份（不是 sparse patch）。"""

    original_dir = domain_root / "months" / month_id
    original_times = np.asarray(np.load(original_dir / "time_utc_ns.npy", mmap_mode="r"), dtype=np.int64)
    combined_times = np.concatenate((original_times, np.asarray(filled["time_utc_ns"], dtype=np.int64)))
    order = np.argsort(combined_times, kind="stable")
    destination = output_months_root / month_id
    partial = output_months_root / f".{month_id}.partial-gap-fill"
    if partial.exists():
        for child in partial.iterdir():
            child.unlink()
        partial.rmdir()
    partial.mkdir()
    try:
        np.save(partial / "time_utc_ns.npy", combined_times[order])
        for name in _NWW3_FIELD_NAMES:
            original_field = np.asarray(np.load(original_dir / f"{name}.npy", mmap_mode="r"))
            if filled[name]:
                new_rows = np.stack(filled[name], axis=0).astype(original_field.dtype)
            else:
                new_rows = np.empty((0, *original_field.shape[1:]), dtype=original_field.dtype)
            combined = np.concatenate((original_field, new_rows), axis=0)
            np.save(partial / f"{name}.npy", combined[order])
        os.replace(partial, destination)
    except Exception:
        import shutil

        shutil.rmtree(partial, ignore_errors=True)
        raise


def _write_wave_gap_fill_manifest(
    output_domain_root: Path,
    *,
    flow_id: str,
    filled_months: Sequence[str],
    symlinked_months: Sequence[str],
    blended_row_count: int,
    single_sided_row_count: int,
    unfilled_time_utc_ns: Sequence[int],
) -> Path:
    """寫出本模組自訂的 NWW3 gap-fill 稽核 manifest（不影響 ``NWWAnalysisMonth`` 讀取）。"""

    document = {
        "schema_version": NWW3_GAP_FILL_SCHEMA_VERSION,
        "flow_id": flow_id,
        "method": NWW3_GAP_FILL_METHOD_ID,
        "filled_months": list(filled_months),
        "symlinked_months": list(symlinked_months),
        "blended_row_count": int(blended_row_count),
        "single_sided_row_count": int(single_sided_row_count),
        "unfilled_time_utc_ns": [int(v) for v in unfilled_time_utc_ns],
    }
    data = _core_canonical_json_bytes(document)
    path = output_domain_root / "nww3-gap-fill-manifest.json"
    path.write_bytes(data)
    (output_domain_root / "nww3-gap-fill-manifest.json.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n", encoding="ascii"
    )
    return path


# ── 遮蔽交叉驗證：重用 ``validate_blocked_masks`` 架構 ────────────────────


def validate_gap_fill_blocked(
    ocm_native_domain_root: str | Path,
    *,
    block_lengths_hours: Sequence[int] = (24, 49, 72, 120, 168),
    n_blocks: int = 3,
    seed: int = 0,
    config: GapFillConfig = _DEFAULT_GAP_FILL_CONFIG,
) -> GapFillValidationReport:
    """在既有觀測時段人為遮蔽，比較相鄰太陰日方法與既有 baseline 的表現。

    架構沿用 ``ocm_reconstruction.validate_blocked_masks``：對一段完整觀測序列
    的中段人為遮蔽固定長度，分別以 persistence、兩端線性、24 小時位移（naive、
    未混合，作為「複製前一天」的對照組）與本模組方法（缺口長度在既有契約內時
    使用 ``_reconstruct_gap``，否則使用相鄰太陰日位移混合）重建，計算表層／中層
    水平流速 RMSE 與向量相關係數。與核心函式不同的是：本函式從磁碟
    ``NpyDomainSource`` 的 domain root 出發，只在 ``n_blocks`` 個隨機（seed
    固定）中心點附近截取足以支援本模組與既有方法的小窗口做遮蔽驗證，不會把
    整個 flow domain 常駐記憶體。

    Returns:
        ``GapFillValidationReport``；``acceptance.formal_scientific_claim``
        固定為 ``False``（與核心模組一致，只是 synthetic 遮蔽驗證，不是正式
        科學驗證）。
    """

    lengths = tuple(int(v) for v in block_lengths_hours)
    if not lengths or any(v < 1 for v in lengths):
        raise ValueError("block_lengths_hours 必須為正整數")
    if n_blocks < 1:
        raise ValueError("n_blocks 必須為正整數")

    source = NpyDomainSource(ocm_native_domain_root)
    observed = source.times_utc_ns
    context_hours = config.reconstruction_config.context_hours
    margin = max(lengths) + 2 * context_hours + 2 * config.shift_hours * 4 + 8
    if observed.size < 2 * margin + 4:
        raise ReconstructionError("來源 domain 觀測時段太短，無法支援 blocked 驗證窗口")

    rng = np.random.default_rng(seed)
    low = margin
    high = observed.size - margin
    centers = rng.integers(low=low, high=high, size=int(n_blocks))

    field_name = "hvel"
    per_length_trials: dict[int, list[dict[str, Any]]] = {length: [] for length in lengths}

    for center in centers:
        window_start = int(center) - margin
        window_stop = int(center) + margin
        window_times = observed[window_start:window_stop]
        fields = {
            name: source.read_times(window_times, name).reshape(
                (window_times.size, *source.field_shapes[name])
            )
            for name in CONTINUOUS_FIELD_NAMES
        }
        fields["wetdry_elem"] = source.read_times(window_times, "wetdry_elem").reshape(
            (window_times.size, *source.field_shapes["wetdry_elem"])
        )
        window_source = ArraySequenceSource(window_times, fields)
        window_center = window_times.size // 2

        for length in lengths:
            start_index = window_center - length // 2
            missing = window_times[start_index : start_index + length]
            masked = window_source.masked(missing)
            gap = GapSpec(
                gap_id=f"gap_fill_blocked_{length}h",
                missing_times_ns=missing,
                left_time_ns=int(missing[0]) - int(HOURLY_NS),
                right_time_ns=int(missing[-1]) + int(HOURLY_NS),
            )
            qualifies = length == 1 or 23 <= length <= 49
            if qualifies and not config.force_lunar_day_blend_only:
                predicted = _core_reconstruct_gap(masked, gap, config.reconstruction_config)[field_name]
                method_used = "existing"
            else:
                arrays, _ = _lunar_day_shift_blend_gap(
                    masked, gap, config, observed_sorted=masked.times_utc_ns
                )
                if arrays["time_utc_ns"].size != missing.size:
                    continue
                predicted = arrays[field_name]
                method_used = "blend"
            predicted = predicted.reshape(length, -1)
            truth = window_source.read_times(missing, field_name)

            twenty_four_hour = _naive_shift_baseline(window_source, missing, field_name, shift_hours=24)

            metrics: dict[str, Any] = {}
            persistence_baseline = _core_baseline_predictions(
                window_source, missing, field_name, kind="persistence"
            )
            endpoint_linear_baseline = _core_baseline_predictions(
                window_source, missing, field_name, kind="endpoint_linear"
            )
            for name, baseline in (
                ("persistence", persistence_baseline),
                ("endpoint_linear", endpoint_linear_baseline),
                ("shift_24h_naive", twenty_four_hour),
            ):
                error = baseline.astype(np.float64) - truth.astype(np.float64)
                metrics[name] = {
                    "rmse": float(np.sqrt(np.nanmean(np.square(error)))),
                    "correlation": _core_corrcoef(truth.ravel(), baseline.ravel()),
                }
            error = predicted.astype(np.float64) - truth.astype(np.float64)
            metrics["gap_fill"] = {
                "method": method_used,
                "rmse": float(np.sqrt(np.nanmean(np.square(error)))),
                "correlation": _core_corrcoef(truth.ravel(), predicted.ravel()),
            }
            per_length_trials[length].append(metrics)

    results: list[Mapping[str, Any]] = []
    for length in lengths:
        trials = per_length_trials[length]
        if not trials:
            results.append({"block_length_hours": length, "trial_count": 0, "trials": [], "accepted": False})
            continue
        mean_rmse = {
            name: float(np.mean([trial[name]["rmse"] for trial in trials]))
            for name in ("persistence", "endpoint_linear", "shift_24h_naive", "gap_fill")
        }
        beats_persistence = mean_rmse["gap_fill"] < mean_rmse["persistence"]
        beats_naive_shift = mean_rmse["gap_fill"] < mean_rmse["shift_24h_naive"]
        accepted = bool(beats_persistence and beats_naive_shift)
        results.append(
            {
                "block_length_hours": length,
                "trial_count": len(trials),
                "mean_rmse": mean_rmse,
                "trials": trials,
                "accepted": accepted,
            }
        )

    acceptance = {
        "all_blocks_pass": bool(all(item["accepted"] for item in results)),
        "formal_scientific_claim": False,
        "basis": "synthetic_blocked_mask_window_from_domain",
    }
    return GapFillValidationReport(
        schema_version="ocm_gap_fill_blocked_validation_v1",
        block_lengths_hours=lengths,
        n_blocks=int(n_blocks),
        seed=int(seed),
        results=tuple(results),
        acceptance=acceptance,
    )


def _naive_shift_baseline(
    full_source: ArraySequenceSource, missing_times: np.ndarray, field_name: str, *, shift_hours: int
) -> np.ndarray:
    """「複製前一個曆法日同一時刻」的 naive baseline，供與 25 小時混合方法對照。

    每個缺口時刻直接取 ``t - shift_hours`` 小時的觀測值，不做任何混合或雙側
    支援；只有在該來源時刻確實是觀測列時才有效，否則退回同一時刻加
    ``shift_hours`` 小時（僅供 synthetic 驗證使用，不代表正式管線行為）。
    """

    shift_ns = np.int64(int(shift_hours)) * HOURLY_NS
    observed = full_source.times_utc_ns
    observed_set = {int(v) for v in observed}
    output = np.empty((missing_times.size, *full_source.field_shapes[field_name]), dtype=np.float64)
    for i, t in enumerate(missing_times):
        candidate = int(t) - int(shift_ns)
        if candidate not in observed_set:
            candidate = int(t) + int(shift_ns)
        value = full_source.read_times(np.asarray([candidate], dtype=np.int64), field_name)
        output[i] = value.reshape(full_source.field_shapes[field_name])
    return output.reshape(missing_times.size, -1)
