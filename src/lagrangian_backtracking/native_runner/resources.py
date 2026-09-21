"""逆向驅動所需「每個流場區域」資源的共用契約。

為什麼要有這個模組
------------------
正式接線（從設定檔與已驗收產品建立網格、關注區域、流場與波浪解析器）與排程
（決定哪些粒子在哪個小時前進）是兩件獨立的工作。兩者只透過本模組定義的
``DomainResources`` 交換資料：接線端負責「建出正確的資源」，排程端只「使用資源」，
測試時排程端可以用合成資源替代，不必讀真實資料。

一個流場區域（flow domain）是共用同一套 OCM 子網格與流場的範圍；例如 A 區兩站共用
一個流場區域，其餘各站各自一個。同一區域內所有粒子讀同一組逐時流場，因此每個區域
每個流場小時只需要合成一次總速度場。

座標一律是公尺制運算座標（與 ``NativeMesh.node_xy`` 同一投影）；經緯度只用於
資料交換與圖面。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DomainResources",
    "DomainResourcesFactory",
]


@dataclass(frozen=True)
class DomainResources:
    """一個流場區域執行逆向追蹤所需的全部唯讀資源。

    欄位
    ----
    domain_id:
        流場區域識別碼，與粒子身分表的區域欄位一致。
    mesh:
        ``lagrangian_backtracking.mesh.NativeMesh``；已投影成公尺座標的 OCM 子網格。
    driver_grid:
        ``backward_driver.NativeBackwardDriverGrid``（或同欄位物件）；由 ``mesh`` 與靜態
        垂向資訊一次建立，交給 BayTrace 核心使用。
    interest_regions:
        「測站識別碼 → 關注區域多邊形（Shapely ``Polygon``，公尺座標）」。工作書規定逆向
        追蹤持續到粒子完全離開其所屬測站的關注區域為止；同一區域的不同測站有各自的
        多邊形，所以驅動層必須依每顆粒子的測站選用對應多邊形。
    forcing_resolver_factory:
        無參數函式，回傳「給定 UTC 奈秒 → ``forcing_bridge.HourlyFieldPair``」的解析器，
        且必須是逆向（``ibf=-1``）設定。每個工作程序各自呼叫一次，取得自己的解析器與
        滑動快取（解析器內部持有記憶體映射檔案，不可跨程序共用）。
    wave_resolver_factory:
        同上，回傳「UTC 奈秒 → ``backward_driver.WaveFieldAtNodes``」的解析器；
        ``None`` 表示此區域沒有可用波浪資料。工作書要求 Stokes，正式運算時不應為
        ``None``；若為 ``None``，驅動層必須在成果中明確標示 Stokes 未啟用，不得靜默。
    available_utc_range_ns:
        此區域已驗收流場涵蓋的（最早, 最晚）UTC 奈秒，供排程檢查回溯窗。
    provenance:
        產生這些資源的來源紀錄（設定檔雜湊、產品路徑與 manifest 雜湊、網格雜湊等），
        會寫進成果 manifest。
    """

    domain_id: str
    mesh: Any
    driver_grid: Any
    interest_regions: Mapping[str, Any]
    forcing_resolver_factory: Callable[[], Callable[[int], Any]]
    wave_resolver_factory: Callable[[], Callable[[int], Any]] | None
    available_utc_range_ns: tuple[int, int]
    provenance: Mapping[str, Any] = field(default_factory=dict)


# 排程端接受的工廠型別：給定區域識別碼，回傳該區域的資源。
# 必須是模組層級、可被 pickle 的函式，才能在 spawn 啟動的工作程序中呼叫。
DomainResourcesFactory = Callable[[str], DomainResources]
