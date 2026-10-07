"""WP-51 B2：把沉底時刻標籤寫進 per_member 沉底 arrival 的 metadata（input-build 階段）。

為什麼在 input-build 算、不在 postprocess 算
--------------------------------------------
標籤需要三條**整段研究期**的站點序列：OCM 表層潮位（算潮汐強度與潮位導數）、OCM 表層流速
（強流事件）與 NWW 的 Hs（高浪事件）。這三條序列在 ``input_derivation.build`` 的 arrival
selector 階段已經以 ``_SiteArrivalSelectionContext`` 讀好、放在記憶體；postprocess
（``native-postprocess``）只讀 ``*.driver_result.npz`` 與粒子身分表 npz，**讀不到**任何原始流場
或波浪產品，若要在那裡算就得重新掃描 OCM／NWW，既慢又讓 postprocess 多一條會出錯的資料相依。
在 input-build 算有四個好處：

1. 零額外 I/O：序列已在記憶體，算 ``250 × M`` 筆標籤只是 O(1) 查表。
2. native-run 不受影響：標籤只是 arrival metadata，隨 ``arrival.json`` 與粒子身分表 npz 流到
   後處理，沒有任何執行期成本。
3. 可稽核：標籤與其模型參數、門檻、各（年, 季）格的潮汐強度中位數都寫進 arrival manifest 的
   provenance（受 manifest hash 保護），事後可重算、可比對。
4. 與 per_member 抽樣同一步完成，arrival ID 與初始條件一旦定案就不會再有「標籤版本」分叉。

本模組只負責「把 :mod:`deposition_labels` 的結果寫進 metadata 並產出 provenance」，不含任何
標註規則；規則與預設參數見 :mod:`deposition_labels`。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .deposition_labels import (
    DEPOSITION_LABEL_METADATA_KEYS,
    DEPOSITION_LABEL_POLICY_ID,
    TIDE_CLASS_VALUES,
    TIDE_PHASE_VALUES,
    DepositionLabelModel,
    build_deposition_label_model,
    labels_digest,
)
from .scenarios import ArrivalTime

__all__ = [
    "DEPOSITION_LABEL_BOOL_KEYS",
    "DEPOSITION_LABEL_PROVENANCE_KEY",
    "attach_deposition_labels",
    "validate_deposition_label_metadata",
]

DEPOSITION_LABEL_BOOL_KEYS = frozenset(
    {"deposition_high_wave", "deposition_strong_current", "deposition_wave_available"}
)
"""arrival metadata 中值型別必須是 bool 的標籤鍵（其餘為字串）。``manifests._metadata`` 對一般
metadata 只接受有限 float／int／非空 string，這三個鍵是唯一例外。"""

DEPOSITION_LABEL_PROVENANCE_KEY = "deposition_time_labels"
"""arrival manifest root provenance 保存標註模型描述的鍵。"""


def attach_deposition_labels(
    arrivals_by_site: Mapping[str, Sequence[ArrivalTime]],
    contexts_by_site: Mapping[str, Any],
    *,
    tidal_strength_window_hours: float | None = None,
) -> dict[str, Any]:
    """依各站 OCM／NWW 序列，把沉底時刻標籤原地寫進每筆沉底 arrival 的 metadata。

    Args:
        arrivals_by_site: ``{站點: 該站全部沉底 ArrivalTime}``；``time_utc_ns`` 必須是沉底 UTC
            （per_member 的輸出）。
        contexts_by_site: ``{站點: _SiteArrivalSelectionContext}``；只用 ``elevation``、``speed``、
            ``nww_series``、``nww_valid`` 四個欄位（各為 ``{UTC ns: 值}``）。每個站都必須有，
            缺站即 fail closed。
        tidal_strength_window_hours: 潮汐強度時窗；``None`` 沿用 ``deposition_labels`` 預設
            （與 v2 selector 預設相同）。呼叫端應傳入 config 的 ``arrival_time_selection`` 設定，
            讓觀測錨點與沉底標籤用同一個時窗。

    Returns:
        可寫進 arrival manifest provenance 的 dict（政策 ID、各站模型 provenance、其 SHA-256）。

    Raises:
        ValueError: 缺站、序列無法建模、沉底時刻不在序列軸上，或標籤季節與 ``ArrivalTime.season``
            不一致時。不以觀測標籤或預設值頂替。
    """

    models: dict[str, DepositionLabelModel] = {}
    for site_id in sorted(arrivals_by_site):
        context = contexts_by_site.get(site_id)
        if context is None:
            raise ValueError(f"缺少站點 {site_id!r} 的 OCM／NWW 序列，無法計算沉底時刻標籤")
        kwargs: dict[str, Any] = {}
        if tidal_strength_window_hours is not None:
            kwargs["tidal_strength_window_hours"] = float(tidal_strength_window_hours)
        model = build_deposition_label_model(
            study_site_id=site_id,
            elevation=context.elevation,
            significant_wave_height=context.nww_series,
            current_speed=context.speed,
            wave_valid=context.nww_valid,
            **kwargs,
        )
        models[site_id] = model
        for arrival in arrivals_by_site[site_id]:
            if arrival.study_site_id != site_id:
                raise ValueError(f"arrival {arrival.arrival_time_id} 的站點與 mapping key {site_id!r} 不一致")
            labels = model.labels_at(int(arrival.time_utc_ns))
            if labels.season != arrival.season:
                # ArrivalTime.season 已由沉底 UTC 重算；兩者不同代表時間軸或對照表出錯。
                raise ValueError(
                    f"arrival {arrival.arrival_time_id} 的沉底季節 {arrival.season!r} 與標籤模型"
                    f"季節 {labels.season!r} 不一致"
                )
            arrival.metadata.update(
                {
                    "deposition_tide_class": labels.tide_class,
                    "deposition_tide_phase": labels.tide_phase,
                    "deposition_high_wave": labels.high_wave,
                    "deposition_strong_current": labels.strong_current,
                    "deposition_wave_available": labels.wave_available,
                    "deposition_label_policy": DEPOSITION_LABEL_POLICY_ID,
                }
            )
    ordered = [models[site_id] for site_id in sorted(models)]
    return {
        "policy_id": DEPOSITION_LABEL_POLICY_ID,
        "labels_sha256": labels_digest(ordered),
        "sites": [model.provenance() for model in ordered],
    }


def validate_deposition_label_metadata(metadata: Mapping[str, Any], *, label: str) -> bool:
    """檢查單筆 arrival metadata 的沉底標籤；回傳這筆是否帶有標籤。

    標籤是選用的（舊 per_member 產物沒有），但**全有或全無**：只要出現任何一個
    ``deposition_*`` 標籤鍵，其餘必須齊全、型別正確、值在允許集合內，政策 ID 也必須是目前版本。
    """

    present = [key for key in DEPOSITION_LABEL_METADATA_KEYS if key in metadata]
    if not present:
        return False
    missing = [key for key in DEPOSITION_LABEL_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"{label} 沉底標籤不完整，缺少 {missing}")
    for key in DEPOSITION_LABEL_BOOL_KEYS:
        if type(metadata[key]) is not bool:
            raise ValueError(f"{label}.{key} 必須是 bool")
    if metadata["deposition_tide_class"] not in TIDE_CLASS_VALUES:
        raise ValueError(f"{label}.deposition_tide_class 不合法：{metadata['deposition_tide_class']!r}")
    if metadata["deposition_tide_phase"] not in TIDE_PHASE_VALUES:
        raise ValueError(f"{label}.deposition_tide_phase 不合法：{metadata['deposition_tide_phase']!r}")
    if metadata["deposition_label_policy"] != DEPOSITION_LABEL_POLICY_ID:
        raise ValueError(f"{label}.deposition_label_policy 與目前版本 {DEPOSITION_LABEL_POLICY_ID} 不符")
    if metadata["deposition_high_wave"] and not metadata["deposition_wave_available"]:
        raise ValueError(f"{label} 的 NWW 無有效值卻標記 high_wave")
    return True
