"""WP-8：把「已核定設定＋已驗收產品」接成排程端可以直接使用的 ``DomainResources``。

為什麼需要這個模組
--------------------
``native_runner`` 底下每個模組（``mesh``／``grid_bridge``／``forcing_bridge``／
``wave_bridge``／``backward_driver``／``particle_input``／``orchestrator``）各自
都已經個別完成、個別測試過；``resources.py`` 也已經定義好排程端需要的
``DomainResources`` 契約。但「從一份正式 YAML 設定與一批已驗收的
``ocm_native``／``nww3_analysis`` 產品目錄，實際把這些既有模組串起來、生出一份
``DomainResources``」——本工作包背景所稱的「正式接線」——在本模組出現以前完全
沒有人做過。本模組**不新增任何科學邏輯**，只負責這條「設定 → 資源」與
「設定 → 粒子身分表」的裝配路徑，並在裝配前後做 fail-closed 的資料契約檢查。

本模組刻意不做的事（重用原則）
--------------------------------
1. 不重新計算網格拓撲、垂向 sigma、流場內插、波浪內插或粒子身分——全部呼叫
   既有 ``mesh.NativeMesh``、``grid_bridge.build_native_grid_arrays``、
   ``backward_driver.build_native_backward_driver_grid``、
   ``forcing_bridge.NativeForcingBridgeReader``、``wave_bridge.NativeWaveBridgeReader``、
   ``manifests.load_scenario_inputs``、``runner.plan_scenario_shards``、
   ``particle_input.build_particle_identity_table``。
2. 不重新定義「關注區域」是什麼——直接重用既有
   ``manifests.load_boundary_geometries`` 算出的
   ``BoundaryGeometry.own_local_domain``（見下方「關注區域定義」一節的完整說明
   與理由）。
3. 不重新發明缺口分類或到達時刻重選規則——``preflight_native_inputs`` 只呼叫既有
   ``forcing_bridge.inventory_forcing_gaps`` 與
   ``orchestrator.derive_backward_window_available``。

環境變數與路徑（絕不硬編伺服器路徑）
--------------------------------------
本模組全部路徑都經由三個環境變數解析，變數「名稱」由設定檔的
``inputs.ocm_native_root_env``／``inputs.ocm_reconstruction_root_env``／
``inputs.nww_analysis_root_env`` 指定，實際「路徑值」在呼叫端環境中設定，本模組
只在呼叫當下用 ``env.get(name)`` 讀取，不快取、不寫回、不記錄實際路徑以外的
任何主機資訊。``LBT_BAYTRACE_ROOT``（BayTrace 原版原始碼根目錄）比照既有
``backward_driver.load_baytrace_modules`` 與既有測試慣例，同樣由呼叫端環境提供。

目錄慣例（本模組唯一「自行認定」、需要對照真實伺服器資料核對的假設）
--------------------------------------------------------------------------
本模組假設一份已驗收產品 root 下的佈局是::

    <root>/<flow_domain_id>/grid/...             # 靜態網格陣列
    <root>/<flow_domain_id>/months/<YYYYMM>/...  # 逐月動態陣列

這個慣例不是本模組發明的，而是從三個既有、彼此獨立的程式碼路徑逐一核對一致
後採用：``forcing_window.ForcingWindowManager.from_roots``、
``runtime.py`` 的 ``_formal_month_path_token``，以及既有測試（
``tests/test_native_forcing_bridge.py``、``tests/test_native_wave_bridge.py``）
的合成資料寫入方式。**誠實揭露**：本模組尚未在真實伺服器資料上驗證過這個慣例；
若正式資料目錄佈局不同（例如月份資料夾不是 ``YYYYMM``、或 grid/months 不在同一個
``<flow_domain_id>`` 之下），``preflight_native_inputs`` 與 ``build_domain_resources``
都會在對應目錄找不到時丟出清楚指出路徑的 ``WiringError``，而不會嘗試用其他佈局
猜測，也不會用空資料頂替。

關注區域（``interest_regions``）定義的選擇與理由
----------------------------------------------------
``resources.DomainResources.interest_regions`` 要求「測站識別碼 → 關注區域多邊形」，
且工作書規定「逆向追蹤持續到粒子完全離開其所屬測站的關注區域為止」。專案中唯一
已經定義、已經驗證過、且與「測站的局部作業範圍」語意一致的既有多邊形，是
``manifests.load_boundary_geometries`` 回傳的
``boundaries.BoundaryGeometry.own_local_domain``——它是 local-domain manifest（
``geometry.local_domain_manifest``）中每個測站自己的多邊形，投影到該測站所屬流場
的 AEQD 公尺座標。該多邊形本身可能是測站 12.5 km 受體核心衍生的 local domain，
也可能等於整個 flow domain（``local_equals_flow=true`` 的站點），依各站在既有
local-domain manifest 中的既定定義而異；本模組不重新推導這個定義，只原樣重用它，
理由是：(1) 它已經被既有正向追蹤 runtime（``runtime.py``／``boundaries.py``）用
作「粒子離開這個範圍即視為離開研究站點作業範圍」的同一個語意；(2) 工作書要求
「完全離開所屬測站的關注區域」，而不是「離開整個流場區域」——若改用
``flow_domain``（``BoundaryGeometry.flow_domain``）會讓貢寮／龜山島共用同一個
過大的 A 區關注區域，稀釋掉「各站各自何時離開」這個工作書要求的判定粒度。
**誠實揭露**：本模組沒有另外驗證「local domain 就是工作書撰寫時心裡想的『關注
區域』」這件事在科學上是否恰當——這是既有幾何 pipeline 的既定定義，本工作包
只負責重用，若研究團隊認為關注區域應該是另一個尚未存在的多邊形，需要另外決策
並可能修改 ``manifests.py`` 或提供新的 manifest，不是本模組能片面決定的事。

月份選擇政策：「磁碟上找得到什麼就用什麼」，完整度檢查交給 preflight
----------------------------------------------------------------------
``build_domain_resources`` 只依 ``inputs.years``（目前是 2024、2025）過濾
``months`` 資料夾下的年份，不要求剛好 24 個月都存在；這是刻意的設計取捨，
理由記在 ``_year_window_months`` 的 docstring。「是否完整涵蓋、是否有缺口、
缺口是否影響到達時刻回溯窗」全部是 ``preflight_native_inputs`` 的職責——這讓
兩個函式的失敗模式清楚分工：``build_domain_resources`` 只在「完全沒有任何可用
月份」時才拒絕，其餘完整度問題一律留給 preflight 在正式開跑前先攔下。

已知限制與尚未被本工作包測試涵蓋的路徑（誠實揭露，不要求呼叫端自行發現）
------------------------------------------------------------------------------
1. ``OCM_RECONSTRUCTION_ROOT`` 的整合（見 ``_load_reconstruction_patches``）
   直接重用 ``forcing_window._ReconstructionPatchStore``——這是刻意匯入一個
   私有類別，而不是重新實作一份 checksum／root index／mesh fingerprint 驗證
   邏輯，因為「不重新推導既有已驗收程式」的原則優先於避免使用底線開頭的名稱。
   但本模組的測試**沒有**建置真正的重建 manifest／root index／checksum sidecar
   fixture（成本很高），只驗證了「完全沒有設定 reconstruction root」與
   「root 存在但這個流場區域沒有 manifest」兩條路徑；「manifest 確實存在且
   成功讀到至少一個月份 patch」這條路徑未被驗證，需要 WP-5／WP-9 整合時另外
   對真實或至少完整的合成重建 manifest 驗證一次。
2. ``build_formal_identity_table`` 的測試改用 ``tests/test_manifests.py`` 既有的
   ``_receptor_payload``／``_arrival_payload``／``_initial_condition_payload``
   等 fixture，直接以 ``formal=True`` 建出完整五站 50,000 情境（``M=1``）與
   500,000 情境（``M=10``）母體，實際套用 ``validate_baseline_coverage``；
   沒有另外用 ``formal=False`` 迴避這個檢查。
3. ``preflight_native_inputs`` 的到達時刻缺口安全檢查是「盡力而為」：只有在
   ``config.scenarios.arrival_time_manifest`` 已設定、且該 manifest 能被
   ``load_scenario_inputs`` 成功讀取時才會執行；讀不到時該檢查會被跳過（在
   報告中以 ``arrival_total_count=None`` 呈現，不是假裝檢查通過）。
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ProjectConfig, load_config, resolve_flow_domain_id
from ..forcing import OCMReconstructionPatchMonth
from ..forcing_window import _ReconstructionPatchStore
from ..geometry import DomainProjection
from ..manifests import load_boundary_geometries, load_scenario_inputs
from ..mesh import NativeMesh
from ..runner import RunUnit, plan_scenario_shards
from .backward_driver import build_native_backward_driver_grid, load_baytrace_modules
from .forcing_bridge import (
    ForcingGapInventoryReport,
    NativeForcingBridgeReader,
    inventory_forcing_gaps,
)
from .grid_bridge import build_native_grid_arrays
from .orchestrator import derive_backward_window_available
from .particle_input import ParticleIdentityTable, build_particle_identity_table
from .resources import DomainResources, DomainResourcesFactory
from .wave_bridge import NativeWaveBridgeReader, build_nww3_node_interpolation_weights

__all__ = [
    "WiringError",
    "build_domain_resources",
    "make_domain_resources_factory",
    "build_formal_identity_table",
    "DomainPreflightReport",
    "PreflightReport",
    "preflight_native_inputs",
    "ReconstructionPatchTimeAxis",
    "load_domain_reconstruction_patch_time_axes",
]


class WiringError(RuntimeError):
    """正式接線（resources／identity table／preflight）階段的一般性資料契約錯誤。

    獨立於單純 ``ValueError``／``RuntimeError`` 命名，方便呼叫端（CLI、排程器）
    精確攔截「接線失敗」而不誤吞其他來源的例外；本模組的 fail-closed 檢查
    （缺目錄、缺環境變數、節點數不一致等）一律丟這個型別。
    """


# ── 共用小工具：設定載入、環境變數、月份目錄探索、雜湊 ─────────────────────


def _resolve_config(
    config_path_or_config: str | Path | ProjectConfig,
    *,
    config_path: str | Path | None,
) -> tuple[ProjectConfig, str | None]:
    """接受設定檔路徑或已載入的 ``ProjectConfig``，統一回傳兩者與可用的路徑字串。

    允許直接傳入已載入的 ``ProjectConfig`` 是為了讓呼叫端（例如已經跑過一次
    ``load_config`` 的 CLI）不必重複解析同一份 YAML；此時 ``config_path`` 仍
    可另外提供，供相對 manifest 路徑解析使用（``manifests.py`` 的
    ``resolve_manifest_path`` 需要 config YAML 所在目錄當基準）。刻意不在這裡
    呼叫 ``formal_release=True``：``ProjectConfig.assert_formal_release_ready``
    要求 ``config_status=="approved"`` 等一整組尚待研究團隊核定的欄位，屬於
    CLI／正式發布閘門的職責，不是「接線」本身要擋的事——接線只在乎能不能把
    既有已驗收產品組成資源，不重複判斷設定是否已核定發布。
    """

    if isinstance(config_path_or_config, ProjectConfig):
        resolved_path = str(config_path) if config_path is not None else None
        return config_path_or_config, resolved_path
    path_text = str(config_path_or_config)
    config = load_config(path_text, formal_release=False)
    return config, path_text


def _require_env(env: Mapping[str, str], name: str, *, purpose: str) -> str:
    """讀取一個必要環境變數，缺少或空白時丟出清楚指名途徑與用途的錯誤。"""

    value = env.get(name)
    if value is None or not str(value).strip():
        raise WiringError(
            f"環境變數 {name} 未設定或為空白；{purpose} 必須由此環境變數指定實際路徑，"
            "本模組不接受硬編伺服器路徑，也不會用預設路徑頂替"
        )
    return str(value)


def _year_window_months(months_root: Path, years: Sequence[int]) -> list[str]:
    """列出 ``months_root`` 下、月份代碼（``YYYYMM``）年份落在 ``years`` 內的子資料夾。

    只依資料夾名稱判斷年份，不開啟任何檔案，因此即使 ``months_root`` 下有數十個
    月份也在毫秒等級完成。刻意不要求「剛好涵蓋整個年份範圍」：磁碟上目前有哪些
    月份是接線當下的事實，「是否完整、有沒有缺口」屬於 ``preflight_native_inputs``
    的職責（見模組 docstring「月份選擇政策」一節），這裡只誠實列出找得到的月份，
    一個都找不到才視為接線本身無法進行下去而拒絕。
    """

    year_set = {int(year) for year in years}
    if not months_root.is_dir():
        raise WiringError(f"缺少月份資料夾：{months_root}")
    months: list[str] = []
    for entry in sorted(months_root.iterdir()):
        if not entry.is_dir() or len(entry.name) != 6 or not entry.name.isdigit():
            continue
        if int(entry.name[:4]) in year_set:
            months.append(entry.name)
    if not months:
        raise WiringError(f"{months_root} 下找不到任何屬於 {sorted(year_set)} 年份的月份資料")
    return months


def _hash_mesh(mesh: NativeMesh) -> str:
    """對網格的靜態陣列計算 SHA-256，供 provenance 使用。

    刻意不呼叫 ``forcing_window._mesh_fingerprint``（那是重建 patch 綁定用的
    私有指紋，欄位子集與用途不同），改用本模組自己的欄位清單獨立計算，避免
    provenance 的 mesh 雜湊語意跟著另一個模組的私有實作細節漂移。
    """

    digest = sha256()
    for name in (
        "node_lon",
        "node_lat",
        "source_depth_m",
        "source_node_bottom_index",
        "face_nodes_local",
        "face_node_count",
        "source_face_global_index",
    ):
        array = np.ascontiguousarray(np.asarray(getattr(mesh, name)))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _site_projection_map(config: ProjectConfig) -> dict[str, DomainProjection]:
    """建立「研究站點 → 所屬流場區域投影中心」的公尺制投影對照表。

    水平受體座標（``scenarios.Receptor.lon/lat``）是 WGS84 經緯度，必須先投影
    成公尺才能交給 BayTrace kernel；投影中心固定沿用 config 的 domain
    ``center_lonlat``，與 ``manifests.load_boundary_geometries`` 建立
    ``own_local_domain`` 用的是完全相同的 ``DomainProjection`` 建構方式（見該
    模組 ``DomainProjection(*domain_config.center_lonlat)`` 的用法），不重新
    推導或另選投影中心，確保受體位置與關注區域多邊形在同一個公尺座標系統下。
    """

    domain_by_region = {domain.analysis_region_id: domain for domain in config.domains}
    projections: dict[str, DomainProjection] = {}
    for site in config.study_sites:
        domain_config = domain_by_region.get(site.analysis_region_id)
        if domain_config is None:
            raise WiringError(
                f"站點 {site.study_site_id!r} 的 analysis_region_id="
                f"{site.analysis_region_id!r} 在 config.domains 中找不到對應流場區域"
            )
        projections[site.study_site_id] = DomainProjection(*domain_config.center_lonlat)
    return projections


def _load_reconstruction_patches(
    reconstruction_root: str | None,
    *,
    flow_domain_id: str,
    months: Sequence[str],
    mesh: NativeMesh,
) -> tuple[dict[str, OCMReconstructionPatchMonth], bool]:
    """嘗試連接已核准的 OCM 缺口重建 patch；此流場區域沒有已核准重建視為未啟用。

    ``forcing_window._ReconstructionPatchStore`` 是既有已驗收的重建 patch 讀取
    與驗證邏輯（checksum、root index、mesh fingerprint 全部沿用，本函式不重新
    實作），本函式只決定「要不要嘗試連上它」。``OCM_RECONSTRUCTION_ROOT`` 是
    全案共用的環境變數，但不是每個流場區域都保證有已核准的重建：若
    ``reconstruction_root`` 有值，但這個 ``flow_domain_id`` 底下找不到 domain
    manifest 或 root index（``FileNotFoundError``），視為「這個區域目前沒有
    已核准重建」而不是設定錯誤；manifest 存在但 checksum、mesh fingerprint 或
    月份資料本身有問題，則不吞掉例外，直接讓呼叫端 fail-closed（維持
    ``forcing_bridge`` 模組文件要求的「缺值不得以最近值或零值頂替」原則）。

    回傳 ``(patches, attempted)``；``attempted`` 只用於 provenance 記錄「有沒有
    嘗試連接」，不代表連接成功。**已知限制**：本函式呼叫的私有類別在本工作包
    的測試中，沒有實際的重建 manifest／checksum sidecar fixture 可用，見模組
    docstring「已知限制」第 1 點。
    """

    if not reconstruction_root:
        return {}, False
    try:
        store = _ReconstructionPatchStore(
            root=reconstruction_root, flow_domain_id=flow_domain_id, mesh=mesh
        )
    except FileNotFoundError:
        # 只有「此流場區域根本沒有 domain manifest」才視為未啟用重建；manifest 在、
        # 但 root index 或其他必要檔案缺漏，代表重建產品不完整（例如本機副本漏複製
        # root index），必須 fail-closed，否則補值會被靜默忽略，運算到缺口時才失敗。
        domain_manifest = Path(reconstruction_root) / flow_domain_id / "reconstruction-manifest.json"
        if domain_manifest.is_file():
            raise
        return {}, True
    patches: dict[str, OCMReconstructionPatchMonth] = {}
    for month_id in months:
        patch = store.load_month(month_id)
        if patch is not None:
            patches[month_id] = patch
    return patches, True


# ── 一、DomainResources 接線 ────────────────────────────────────────────────

#: 一個 OCM 原生月份資料夾在「不讀 diffusivity」模式下仍必須存在的動態陣列
#: 檔名（刻意不含 ``diffusivity.npy``——見 ``forcing_bridge`` 模組文件
#: 「``load_diffusivity`` 開關」一節：本機一次性複製只保留逆向驅動層需要的
#: 變數，擴散係數改由 Smagorinsky 方法從流速算出，OCM 的 diffusivity 不再是
#: 必要檔案）。``metadata.json``／``quality_report.json`` 是背景描述的隨附
#: 檔案，本模組現有讀取路徑（``forcing_bridge``／本檔）都不會打開它們，因此
#: 這裡不列入「會被實際讀取」的必要清單，只檢查真正會被載入的動態陣列。
_REQUIRED_OCM_MONTH_ARRAYS: tuple[str, ...] = (
    "hvel.npy",
    "vertical_velocity.npy",
    "zcor.npy",
    "elev.npy",
    "wetdry_elem.npy",
    "time_utc_ns.npy",
)


def _check_required_ocm_month_files(flow_root: Path, months: Sequence[str]) -> list[str]:
    """檢查每個月份資料夾是否具備 ``_REQUIRED_OCM_MONTH_ARRAYS`` 全部檔案。

    回傳問題訊息清單（缺一個檔案就是一條訊息，明確列出月份與檔名），不
    fail-fast——讓呼叫端（``preflight_native_inputs``）一次收集完整缺漏，
    不必修一個、重跑一次才發現下一個。刻意不檢查 ``diffusivity.npy``：本
    模組正式運算一律以 ``load_diffusivity=False`` 呼叫
    ``NativeForcingBridgeReader``（見 ``build_domain_resources``），該檔案
    是否存在不影響是否可以開跑。
    """

    problems: list[str] = []
    for month_id in months:
        month_dir = flow_root / "months" / month_id
        for file_name in _REQUIRED_OCM_MONTH_ARRAYS:
            if not (month_dir / file_name).is_file():
                problems.append(f"OCM 月份 {month_id} 缺少必要檔案：{month_dir / file_name}")
    return problems


def _load_reference_month_zcor_elev(month_dir: Path, *, mesh: NativeMesh) -> tuple[np.ndarray, np.ndarray]:
    """只讀參考月份 ``zcor``／``elev`` 的第 0 個時間列，供反推固定分率垂向座標用。

    ``build_domain_resources`` 需要的只是 ``grid_bridge.build_native_grid_
    arrays`` 要求的單一參考時刻 ``zcor``／``eta``（見該函式模組文件「單一
    參考時刻」設計），刻意不透過 ``forcing.OCMNativeMonth``——那個類別的
    建構子固定要求 ``diffusivity.npy`` 存在，因為它服務的是另一條仍需要
    OCM 原生擴散係數逐節點取樣的既有管線（``forcing_window``／其他非
    native_runner 呼叫端），本模組（逆向驅動層的水平擴散改用 Smagorinsky
    方法，見 ``forcing_bridge`` 模組文件「``load_diffusivity`` 開關」）不
    應該為了讀兩個陣列的第 0 列，反過來要求本機一次性複製（刻意不含
    diffusivity）補一個它用不到的檔案。因此改用唯讀 memmap 直接開檔，只讀
    ``hvel`` 的 shape（不觸資料本體，確認節點軸與 ``mesh`` 一致）與
    ``zcor``／``elev`` 的第 0 列，行為與 ``OCMNativeMonth.__init__`` 對這兩
    個欄位原本做的形狀檢查等價。
    """

    def load(name: str) -> np.ndarray:
        path = month_dir / name
        if not path.is_file():
            raise WiringError(f"缺少 OCM month 陣列：{path}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    hvel_shape = load("hvel.npy").shape
    zcor = load("zcor.npy")
    elev = load("elev.npy")
    node_count = int(mesh.node_xy.shape[0])
    if len(hvel_shape) != 4 or hvel_shape[1] != node_count:
        found = hvel_shape[1] if len(hvel_shape) > 1 else None
        raise WiringError(f"{month_dir} 的 hvel 節點軸 {found} 與網格節點數 {node_count} 不一致")
    if zcor.shape[:2] != hvel_shape[:2]:
        raise WiringError(f"{month_dir} 的 zcor shape 與 hvel 的 time/node 軸不一致")
    if elev.shape != hvel_shape[:2]:
        raise WiringError(f"{month_dir} 的 elev shape 與 hvel 的 time/node 軸不一致")
    return np.asarray(zcor[0], dtype=np.float64), np.asarray(elev[0], dtype=np.float64)


def build_domain_resources(
    config_path_or_config: str | Path | ProjectConfig,
    domain_id: str,
    *,
    env: Mapping[str, str] = os.environ,
    config_path: str | Path | None = None,
    formal: bool = True,
) -> DomainResources:
    """依正式設定與已驗收產品，建出一個流場區域逆向追蹤所需的全部唯讀資源。

    ``domain_id`` 是 ``analysis_region_id``（例如 ``"A"``），必須與
    ``resources.DomainResources.domain_id`` 的既有契約（「與粒子身分表的區域
    欄位一致」，即 ``ParticleIdentityTable.analysis_region_id``）一致——不是
    ``flow_domain_id``；同一個 ``flow_domain_id`` 內部才會依此再解析。

    重用清單（本函式不重新實作以下任何邏輯，只依序呼叫）：

    - ``config.resolve_flow_domain_id``：pilot／formal 的上游產品識別碼解析。
    - ``mesh.NativeMesh.from_directory``：靜態網格與投影。
    - ``native_runner.backward_driver.load_baytrace_modules``／
      ``build_native_backward_driver_grid``：受控載入 BayTrace 原始碼、建立
      驅動層網格。
    - ``_load_reference_month_zcor_elev``（本模組自有，WP-12 新增）：讀取
      單一參考月份的 ``zcor``／``elev``，供 ``grid_bridge.build_native_grid_
      arrays`` 反推固定分率垂向座標（只用該月份第 0 個時間列當參考時刻，與
      既有 ``grid_bridge`` 模組文件的「單一參考時刻」設計一致，不代表其餘
      時刻的垂向結構完全相同——這是既有 ``grid_bridge`` 模組已經記載並驗證
      過的科學限制，本函式只是選擇「哪一個參考時刻」，不改變限制本身）。
      WP-12 之前這裡呼叫 ``forcing.OCMNativeMonth.from_directory``，但該類
      別固定要求 ``diffusivity.npy`` 存在；本專案逆向驅動層本來就不讀 OCM
      擴散係數（水平擴散改用 Smagorinsky 方法，見 ``forcing_bridge`` 模組
      文件「``load_diffusivity`` 開關」），且本機一次性複製的月份副本刻意
      不含 ``diffusivity.npy``，因此改用只讀 ``zcor``／``elev``／``hvel``
      形狀的最小讀取，不再耦合這個用不到的檔案契約。
    - ``native_runner.forcing_bridge.NativeForcingBridgeReader``（``ibf=-1``
      逆向）／``native_runner.wave_bridge.NativeWaveBridgeReader``：分別包成
      無參數工廠，每個工作程序各自呼叫一次取得自己的解析器（見
      ``resources.DomainResources`` 的欄位文件）。
    - ``manifests.load_boundary_geometries``：取得每個測站的
      ``BoundaryGeometry.own_local_domain`` 當關注區域（見模組 docstring
      「關注區域定義的選擇與理由」一節的完整討論）。

    Args:
        config_path_or_config: 設定 YAML 路徑，或已載入的 ``ProjectConfig``。
        domain_id: ``analysis_region_id``（例如 ``"A"``／``"B"``／``"C"``／
            ``"D"``）。
        env: 用於解析 ``OCM_NATIVE_ROOT``／``OCM_RECONSTRUCTION_ROOT``／
            ``NWW_ANALYSIS_ROOT``（實際變數名稱由設定檔決定）與
            ``LBT_BAYTRACE_ROOT`` 的環境變數來源；預設 ``os.environ``，測試或
            排程器可傳入自訂 mapping。
        config_path: ``config_path_or_config`` 是 ``ProjectConfig`` 時，供相對
            geometry manifest 路徑解析用的 YAML 路徑；``config_path_or_config``
            本身是路徑時忽略此參數（一律以它為準）。
        formal: 是否使用正式（而非 pilot）上游產品識別碼與完整站點覆蓋要求，
            透傳給 ``resolve_flow_domain_id``／``load_boundary_geometries``。

    Returns:
        ``resources.DomainResources``。

    Raises:
        WiringError: 環境變數缺漏、目錄不存在、找不到對應站點或關注區域、
            OCM／NWW3 可用時間範圍沒有交集等資料契約問題。
    """

    if not isinstance(domain_id, str) or not domain_id.strip():
        raise WiringError("domain_id 不可為空白")
    config, resolved_config_path = _resolve_config(config_path_or_config, config_path=config_path)

    domain_matches = [item for item in config.domains if item.analysis_region_id == domain_id]
    if len(domain_matches) != 1:
        raise WiringError(f"config.domains 必須恰有一個 analysis_region_id={domain_id!r} 的流場區域")
    domain_config = domain_matches[0]
    flow_domain_id = resolve_flow_domain_id(config, domain_id, formal=formal)
    projection = DomainProjection(*domain_config.center_lonlat)

    ocm_native_root = Path(
        _require_env(env, config.inputs.ocm_native_root_env, purpose="OCM 原生產品 root")
    )
    flow_root = ocm_native_root / flow_domain_id
    grid_dir = flow_root / "grid"
    if not grid_dir.is_dir():
        raise WiringError(f"缺少 OCM 原生網格資料夾：{grid_dir}")
    mesh = NativeMesh.from_directory(grid_dir, projection=projection)

    baytrace_root = _require_env(env, "LBT_BAYTRACE_ROOT", purpose="BayTrace 原版原始碼根目錄")
    handles = load_baytrace_modules(baytrace_root)

    ocm_months = _year_window_months(flow_root / "months", config.inputs.years)
    reference_month_id = ocm_months[0]
    zcor_reference, eta_reference = _load_reference_month_zcor_elev(
        flow_root / "months" / reference_month_id, mesh=mesh
    )
    grid_arrays = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    driver_grid = build_native_backward_driver_grid(grid_arrays, handles)

    reconstruction_root_env = config.inputs.ocm_reconstruction_root_env
    reconstruction_root = env.get(reconstruction_root_env) if reconstruction_root_env else None
    patches, reconstruction_attempted = _load_reconstruction_patches(
        reconstruction_root, flow_domain_id=flow_domain_id, months=ocm_months, mesh=mesh
    )

    forcing_month_dirs = {month_id: str(flow_root / "months" / month_id) for month_id in ocm_months}

    def forcing_resolver_factory() -> Callable[[int], Any]:
        """每個工作程序各自呼叫一次，取得自己的逆向（``ibf=-1``）流場解析器。

        ``load_diffusivity=False``：本專案逆向驅動層不讀 OCM 原生擴散係數
        （水平擴散改用 Smagorinsky 方法直接從流速算出，見 ``forcing_bridge``
        模組文件「``load_diffusivity`` 開關」一節），本機一次性複製的月份
        副本也刻意不含 ``diffusivity.npy``；固定傳 ``False`` 讓正式資源不
        依賴這個檔案是否存在。
        """

        reader = NativeForcingBridgeReader(
            forcing_month_dirs, ibf=-1, patches=patches, load_diffusivity=False
        )
        return reader.resolve

    probe_forcing_reader = NativeForcingBridgeReader(
        forcing_month_dirs, ibf=-1, patches=patches, load_diffusivity=False
    )
    available_lo_ns, available_hi_ns = probe_forcing_reader.available_time_range_utc_ns

    nww_native_root = Path(
        _require_env(env, config.inputs.nww_analysis_root_env, purpose="NWW3 分析產品 root")
    )
    nww_flow_root = nww_native_root / flow_domain_id
    nww_grid_dir = nww_flow_root / "grid"
    wave_resolver_factory: Callable[[], Callable[[int], Any]] | None = None
    nww_months: list[str] = []
    if nww_grid_dir.is_dir() and (nww_flow_root / "months").is_dir():
        try:
            nww_months = _year_window_months(nww_flow_root / "months", config.inputs.years)
        except WiringError:
            nww_months = []
    if nww_months:
        nww_month_dirs = {month_id: str(nww_flow_root / "months" / month_id) for month_id in nww_months}

        def wave_resolver_factory() -> Callable[[int], Any]:
            """每個工作程序各自呼叫一次，取得自己的波浪場解析器。"""

            reader = NativeWaveBridgeReader(
                grid_dir=nww_grid_dir,
                month_dirs=nww_month_dirs,
                node_lon=mesh.node_lon,
                node_lat=mesh.node_lat,
            )
            return reader.as_wave_field_resolver()

        probe_wave_reader = NativeWaveBridgeReader(
            grid_dir=nww_grid_dir,
            month_dirs=nww_month_dirs,
            node_lon=mesh.node_lon,
            node_lat=mesh.node_lat,
        )
        wave_lo_ns, wave_hi_ns = probe_wave_reader.available_time_range_utc_ns
        available_lo_ns = max(available_lo_ns, wave_lo_ns)
        available_hi_ns = min(available_hi_ns, wave_hi_ns)
        if available_lo_ns > available_hi_ns:
            raise WiringError(
                f"{flow_domain_id} 的 OCM 與 NWW3 可用時間範圍沒有交集：OCM="
                f"[{probe_forcing_reader.available_time_range_utc_ns}]，NWW3=[{wave_lo_ns},{wave_hi_ns}]"
            )

    boundary_bundle = load_boundary_geometries(config, config_path=resolved_config_path, formal=formal)
    sites_in_domain = [site for site in config.study_sites if site.analysis_region_id == domain_id]
    if not sites_in_domain:
        raise WiringError(f"config.study_sites 沒有任何站點屬於分析區域 {domain_id!r}")
    interest_regions: dict[str, Any] = {}
    for site in sites_in_domain:
        if site.study_site_id not in boundary_bundle:
            raise WiringError(
                f"boundary geometry bundle 缺少站點 {site.study_site_id!r} 的關注區域多邊形；"
                "請確認 local domain manifest 已涵蓋此站"
            )
        interest_regions[site.study_site_id] = boundary_bundle[site.study_site_id].own_local_domain

    provenance: dict[str, Any] = {
        "config_sha256": config.config_hash(),
        "config_path": resolved_config_path,
        "domain_id": domain_id,
        "flow_domain_id": flow_domain_id,
        "formal": formal,
        "mesh_sha256": _hash_mesh(mesh),
        "baytrace_source_root": str(handles.source_root),
        "baytrace_dependency_sha256": dict(handles.dependency_sha256),
        "ocm_native_root": str(ocm_native_root),
        "ocm_months_used": tuple(ocm_months),
        "ocm_reference_month_for_sigma": reference_month_id,
        "ocm_reconstruction_root": reconstruction_root,
        "ocm_reconstruction_attempted": reconstruction_attempted,
        "ocm_reconstruction_patch_months_used": tuple(sorted(patches)),
        "ocm_diffusivity_used": False,
        "horizontal_diffusion_method": (
            "本次運算不使用 OCM diffusivity，水平擴散採 Smagorinsky 方法（工作書式 (10)），"
            "由 backward_driver.compute_smagorinsky_kh_field 從本次運算合成後的流速直接算出；"
            "NativeForcingBridgeReader 以 load_diffusivity=False 建立，不讀取 diffusivity.npy"
        ),
        "nww_analysis_root": str(nww_native_root),
        "nww_months_used": tuple(nww_months),
        "wave_data_available": wave_resolver_factory is not None,
        "interest_region_definition": (
            "boundaries.BoundaryGeometry.own_local_domain（local-domain manifest 投影後的"
            "站點 local polygon），與既有正向追蹤 runtime 共用同一份幾何定義"
        ),
        "geometry_manifest_file_sha256": dict(boundary_bundle.file_sha256),
        "geometry_manifest_canonical_sha256": dict(boundary_bundle.canonical_component_hashes),
    }

    return DomainResources(
        domain_id=domain_id,
        mesh=mesh,
        driver_grid=driver_grid,
        interest_regions=interest_regions,
        forcing_resolver_factory=forcing_resolver_factory,
        wave_resolver_factory=wave_resolver_factory,
        available_utc_range_ns=(available_lo_ns, available_hi_ns),
        provenance=provenance,
    )


def _domain_resources_factory_entrypoint(
    config_path: str, env: dict[str, str], formal: bool, domain_id: str
) -> DomainResources:
    """``make_domain_resources_factory`` 綁定後、實際被排程器呼叫的模組層級函式。

    獨立成模組層級函式（而非在 ``make_domain_resources_factory`` 內定義閉包），
    是因為 spawn 啟動的工作程序需要對這個 callable 做 pickle：閉包無法被
    pickle，但對一個模組層級函式配合純字串／字典／布林引數的
    ``functools.partial`` 可以（``functools.partial`` 物件 pickle 時只保存函式的
    模組限定名稱與已綁定引數，載入端從各自的 ``sys.path`` 重新 import 這個
    名稱，因此工作程序不需要與主程序共享任何記憶體）。
    """

    return build_domain_resources(config_path, domain_id, env=env, formal=formal)


@dataclass(frozen=True, slots=True)
class ReconstructionPatchTimeAxis:
    """規劃階段用的重建 patch「時間軸」替身：只含 ``time_utc_ns``，不含任何流場陣列。"""

    month_id: str
    time_utc_ns: np.ndarray


def load_domain_reconstruction_patch_time_axes(
    reconstruction_root: str | Path | None,
    flow_domain_id: str,
    *,
    months: Sequence[str],
) -> dict[str, ReconstructionPatchTimeAxis]:
    """讀取某流場區域已核准重建 patch 在指定月份補上的 UTC 小時，供 ``native-plan`` 缺口盤點。

    ``inventory_forcing_gaps`` 只使用 patch 的時間軸，因此規劃階段不必載入並逐檔
    驗證整份 patch（NE 區單月可達數 GB，經 100 Mb/s 網路磁碟驗證雜湊要數十分鐘）；
    這裡只讀 domain manifest 是否存在與各月份的 ``time_utc_ns.npy``。**完整驗證
    （checksum、root index、網格指紋）仍在正式執行時由 ``build_domain_resources`` →
    ``_ReconstructionPatchStore`` 執行**，驗證失敗會讓該區域 fail-closed，不會因為
    規劃階段較寬鬆而使用未驗證的補值。``reconstruction_root`` 為空或此區域沒有
    domain manifest 時回傳空字典（等同未啟用重建）。
    """

    if not reconstruction_root:
        return {}
    domain_root = Path(reconstruction_root) / flow_domain_id
    if not (domain_root / "reconstruction-manifest.json").is_file():
        return {}
    axes: dict[str, ReconstructionPatchTimeAxis] = {}
    for month_id in months:
        time_path = domain_root / "months" / month_id / "time_utc_ns.npy"
        if time_path.is_file():
            axes[month_id] = ReconstructionPatchTimeAxis(
                month_id=month_id, time_utc_ns=np.asarray(np.load(time_path), dtype=np.int64)
            )
    return axes


def make_domain_resources_factory(
    config_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    formal: bool = True,
) -> DomainResourcesFactory:
    """回傳可直接交給排程器（``--resource-provider``）使用、可 pickle 的資源工廠。

    回傳值符合 ``resources.DomainResourcesFactory``（``Callable[[str],
    DomainResources]``）：呼叫 ``factory(domain_id)`` 等同呼叫
    ``build_domain_resources(config_path, domain_id, env=env, formal=formal)``。
    刻意要求 ``config_path`` 是路徑而非已載入的 ``ProjectConfig``——後者雖然
    技術上也可能被 pickle，但那等於把整份設定的記憶體副本複製給每個工作
    程序，且無法保證使用者手上的 ``ProjectConfig`` 執行個體在 pickle／unpickle
    往返後仍與磁碟上的 YAML 一致；要求路徑則讓每個工作程序各自重新
    ``load_config``，來源永遠是同一份可稽核的 YAML 檔案。

    Args:
        config_path: 設定 YAML 路徑（不是 ``ProjectConfig``）。
        env: 環境變數 mapping；``None`` 時在呼叫當下複製一份
            ``os.environ``（複製而非直接引用，確保之後修改
            ``os.environ`` 不會影響已建立的工廠）。
        formal: 透傳給每次 ``build_domain_resources`` 呼叫。

    Returns:
        ``resources.DomainResourcesFactory``；可安全地在 ``multiprocessing``
        ``spawn`` context 下傳遞給子程序。
    """

    if isinstance(config_path, ProjectConfig):
        raise WiringError(
            "make_domain_resources_factory 需要設定檔路徑，不接受已載入的 ProjectConfig"
            "（無法安全 pickle 給 spawn 工作程序，見本函式 docstring）"
        )
    env_dict = dict(env) if env is not None else dict(os.environ)
    return functools.partial(_domain_resources_factory_entrypoint, str(config_path), env_dict, formal)


# ── 二、正式粒子身分表 ────────────────────────────────────────────────────


def build_formal_identity_table(
    config_path_or_config: str | Path | ProjectConfig,
    *,
    members_per_scenario: int,
    env: Mapping[str, str] = os.environ,
    config_path: str | Path | None = None,
    formal: bool = True,
    shard_scenario_count: int | None = None,
    experiment_case_id: str | None = None,
    random_stream_id: str | None = None,
) -> ParticleIdentityTable:
    """由既有 inputs-build 產物建出全案粒子身分表，``members_per_scenario`` 可為 1 或 10。

    重用清單（本函式不重新推導任何科學或亂數邏輯，只依序呼叫）：

    - ``manifests.load_scenario_inputs``：讀取材料、受體、到達時刻與（若已有）
      receptor×arrival 正式三維初始條件 manifest，並用既有
      ``scenarios.build_scenarios`` 建出完整交叉組合。``formal=True``（預設）時
      會套用既有 ``scenarios.validate_baseline_coverage``（五站各 10,000、
      A 區 20,000、全案 50,000），與正式母體契約完全一致；本函式不放寬或
      重寫這個檢查。
    - ``runner.plan_scenario_shards``：把情境切成不重疊的批次。
    - ``native_runner.particle_input.build_particle_identity_table``：把批次
      展開成粒子身分表，``members_per_scenario`` 直接透傳（``M=1`` 時
      50,000 顆，``M=10`` 時 500,000 顆，五站合計）。

    本函式唯一新增的邏輯，是 ``resolve_initial_position``：把
    ``scenarios.Receptor``（水平位置模板，WGS84）與
    ``scenarios.ReceptorArrivalInitialCondition``（該 receptor×arrival pair
    由 OCM 在到達時刻推導出的實際初始深度）合併查詢成
    ``particle_input.InitialPositionResolver`` 要求的公尺制三維座標；水平座標
    的投影中心與 ``manifests.load_boundary_geometries`` 建立關注區域用的是
    同一個 ``DomainProjection`` 建構方式（見 ``_site_projection_map``）。

    Args:
        config_path_or_config: 設定 YAML 路徑，或已載入的 ``ProjectConfig``。
        members_per_scenario: 每個基礎情境的獨立隨機系集成員數（正式設計為
            10；``M=1`` 供收斂測試或工程試跑）。
        env: 目前未被本函式使用（身分表建構不需要讀取 OCM／NWW3 產品），保留
            這個參數只是讓函式簽章與 ``build_domain_resources``、
            ``preflight_native_inputs`` 一致，方便呼叫端統一傳遞同一組環境
            變數 mapping。
        config_path: ``config_path_or_config`` 是 ``ProjectConfig`` 時，供相對
            manifest 路徑解析用的 YAML 路徑。
        formal: 是否套用 ``validate_baseline_coverage`` 與要求完整
            receptor×arrival 初始條件 manifest（見 ``load_scenario_inputs``
            docstring）。**測試專用的已知妥協**：本模組測試以
            ``formal=False`` 對一個縮小的單站設定驗證數量與內容正確性，避免
            必須建置全尺寸五站 50,000 情境合成資料；正式呼叫必須維持預設
            ``formal=True``。
        shard_scenario_count: 批次大小；``None`` 時依序回退到
            ``config.execution.shard_scenario_count``，兩者都缺時退回
            「全部情境當成單一批次」。批次大小只影響 I/O 與 checkpoint
            粒度，不影響粒子身分、seed 或科學結果，因此這個退回值是安全的
            工程預設，不是科學決策。
        experiment_case_id: 傳給 ``plan_scenario_shards``／
            ``iter_run_units`` 的物理案例識別碼；``None`` 時使用
            ``config.design_version``。
        random_stream_id: 透傳給 ``build_particle_identity_table``。

    Returns:
        ``particle_input.ParticleIdentityTable``。

    Raises:
        WiringError: ``members_per_scenario`` 非正整數、``master_seed`` 尚未
            核定、找不到受體或初始條件，或站點缺少對應流場區域投影。
    """

    del env  # 見上方 docstring：目前保留參數只為簽章一致，本函式尚未用到它。
    if type(members_per_scenario) is not int or members_per_scenario < 1:
        raise WiringError("members_per_scenario 必須是正整數")
    config, resolved_config_path = _resolve_config(config_path_or_config, config_path=config_path)

    # 先檢查 master_seed（便宜、只讀設定本身），再讀取情境 manifest（可能是大型
    # JSON 檔案的 I/O）——讓「master_seed 尚未核定」這種設定層面的錯誤，不必等到
    # manifest 都讀完才被發現，呼叫端也不會被一個實際上與 manifest 內容無關的
    # FileNotFoundError 誤導。
    if config.scenarios.master_seed is None or config.scenarios.master_seed < 0:
        raise WiringError("config.scenarios.master_seed 尚未核定，無法建立正式粒子身分表")
    master_seed = int(config.scenarios.master_seed)

    scenario_inputs = load_scenario_inputs(
        config,
        config_path=resolved_config_path,
        require_dynamic_initial_conditions=True,
        formal=formal,
    )

    case_id = experiment_case_id or config.design_version
    resolved_shard_count = (
        shard_scenario_count
        or config.execution.shard_scenario_count
        or len(scenario_inputs.scenarios)
    )
    shards = plan_scenario_shards(
        list(scenario_inputs.scenarios),
        members_per_scenario=members_per_scenario,
        shard_scenario_count=resolved_shard_count,
        experiment_case_id=case_id,
    )

    site_projection = _site_projection_map(config)
    receptors_by_id = {receptor.receptor_id: receptor for receptor in scenario_inputs.receptors}
    initial_conditions_by_pair = scenario_inputs.initial_conditions_by_pair

    def resolve_initial_position(unit: RunUnit) -> tuple[float, float, float]:
        """合併受體水平位置模板與 receptor×arrival 正式初始條件的實際初始深度。"""

        scenario = unit.scenario
        receptor = receptors_by_id.get(scenario.receptor_id)
        if receptor is None:
            raise WiringError(f"找不到受體 {scenario.receptor_id!r} 的水平位置模板")
        condition = initial_conditions_by_pair.get((scenario.receptor_id, scenario.arrival_time_id))
        if condition is None:
            raise WiringError(
                f"找不到 receptor×arrival ({scenario.receptor_id!r}, "
                f"{scenario.arrival_time_id!r}) 的正式三維初始條件"
            )
        projection = site_projection.get(scenario.study_site_id)
        if projection is None:
            raise WiringError(f"找不到站點 {scenario.study_site_id!r} 的公尺制投影中心")
        x_m, y_m = projection.project(receptor.lon, receptor.lat)
        return float(x_m), float(y_m), float(condition.z_m_positive_up)

    return build_particle_identity_table(
        shards,
        master_seed=master_seed,
        resolve_initial_position=resolve_initial_position,
        random_stream_id=random_stream_id,
    )


# ── 三、正式跑前置檢查 ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DomainPreflightReport:
    """單一流場區域的正式資料前置檢查結果。

    每個欄位在對應檢查因故無法執行時保持 ``None``（而不是假造一個看似通過的
    值），``describe()`` 會如實反映哪些檢查有跑、哪些沒有。
    """

    domain_id: str
    flow_domain_id: str
    problems: tuple[str, ...]
    ocm_months_found: tuple[str, ...]
    nww_months_found: tuple[str, ...]
    node_count_grid: int | None
    node_count_hvel_sample: int | None
    wetdry_elem_values_ok: bool | None
    nww3_node_coverage_fraction: float | None
    forcing_gap_report: ForcingGapInventoryReport | None
    arrival_unsafe_count: int | None
    arrival_total_count: int | None

    @property
    def is_run_ready(self) -> bool:
        """沒有 fail-closed 問題、缺口盤點（若已算出）就緒、到達窗未受影響時才視為可執行。"""

        if self.problems:
            return False
        if self.forcing_gap_report is not None and not self.forcing_gap_report.is_run_ready:
            return False
        return not self.arrival_unsafe_count

    def describe(self) -> str:
        """回傳可直接列印的多行人類可讀摘要。"""

        lines = [f"=== 流場區域 {self.domain_id}（flow_domain_id={self.flow_domain_id}） ==="]
        if self.problems:
            lines.append("發現以下問題：")
            lines.extend(f"  - {problem}" for problem in self.problems)
        else:
            lines.append("基本資料契約檢查通過（目錄、節點數、wetdry 編碼）。")
        ocm_list = ", ".join(self.ocm_months_found) or "無"
        nww_list = ", ".join(self.nww_months_found) or "無"
        lines.append(f"OCM 月份（{len(self.ocm_months_found)} 個）：{ocm_list}")
        lines.append(f"NWW3 月份（{len(self.nww_months_found)} 個）：{nww_list}")
        if self.node_count_grid is not None:
            lines.append(
                f"網格節點數={self.node_count_grid}，OCM hvel 樣本節點數="
                f"{self.node_count_hvel_sample}"
            )
        if self.wetdry_elem_values_ok is not None:
            lines.append(f"wetdry_elem 編碼是否符合 {{0,1,NaN}} 契約：{self.wetdry_elem_values_ok}")
        if self.nww3_node_coverage_fraction is not None:
            lines.append(f"NWW3 對 OCM 節點的空間涵蓋率：{self.nww3_node_coverage_fraction:.4f}")
        if self.forcing_gap_report is not None:
            lines.append(self.forcing_gap_report.describe())
        if self.arrival_total_count is not None:
            safe_count = self.arrival_total_count - (self.arrival_unsafe_count or 0)
            lines.append(
                f"到達時刻回溯窗缺口安全性：{safe_count}/{self.arrival_total_count} "
                "個安全（30 天回溯窗未落在未登錄缺口內）"
            )
        else:
            lines.append("到達時刻回溯窗缺口安全性：未檢查（缺少 arrival manifest 或無法讀取）")
        lines.append(f"整體判定：{'可執行' if self.is_run_ready else '尚不可執行'}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """全案（可能多個流場區域）的正式資料前置檢查彙整。"""

    domain_reports: tuple[DomainPreflightReport, ...]

    @property
    def is_run_ready(self) -> bool:
        """所有流場區域都判定可執行時才整體可執行。"""

        return all(report.is_run_ready for report in self.domain_reports)

    def describe(self) -> str:
        """回傳可直接列印／寫入前置檢查記錄的多行摘要，逐區域列出。"""

        return "\n\n".join(report.describe() for report in self.domain_reports)


def _try_load_arrival_times_by_site(
    config: ProjectConfig, config_path: str | None
) -> Mapping[str, tuple[Any, ...]] | None:
    """盡力載入各站到達時刻清單，供缺口安全性檢查；讀不到時回傳 ``None``。

    刻意不視為 fail-closed：``preflight_native_inputs`` 的核心職責是檢查
    forcing／wave 產品本身是否就緒，到達時刻缺口安全性是「已有 arrival
    manifest」時的加值檢查，不應該讓一個還沒產出 arrival manifest 的階段性
    設定，連基本的目錄／節點數檢查都做不了。呼叫端（``DomainPreflightReport``）
    會誠實呈現「這個檢查根本沒有跑」而不是「跑了而且通過」。
    """

    if not config.scenarios.arrival_time_manifest:
        return None
    try:
        scenario_inputs = load_scenario_inputs(
            config,
            config_path=config_path,
            require_dynamic_initial_conditions=False,
            formal=False,
        )
    except (ValueError, FileNotFoundError):
        return None
    by_site: dict[str, list[Any]] = {}
    for arrival in scenario_inputs.arrival_times:
        by_site.setdefault(arrival.study_site_id, []).append(arrival)
    return {site_id: tuple(items) for site_id, items in by_site.items()}


def _preflight_one_domain(
    config: ProjectConfig,
    domain_id: str,
    *,
    env: Mapping[str, str],
    formal: bool,
    max_backtrack_seconds: float,
    arrival_times_by_site: Mapping[str, tuple[Any, ...]] | None,
) -> DomainPreflightReport:
    """對單一流場區域執行全部前置檢查，任何一步失敗都記錄成 ``problems`` 繼續檢查其餘項目。

    與 ``build_domain_resources`` 不同：這裡刻意不在第一個問題發生時就中止，
    而是盡量收集完整報告，讓操作者一次看到這個區域所有問題，不必逐一修正、
    逐一重跑才發現下一個問題。
    """

    problems: list[str] = []
    domain_matches = [item for item in config.domains if item.analysis_region_id == domain_id]
    if len(domain_matches) != 1:
        return DomainPreflightReport(
            domain_id=domain_id,
            flow_domain_id="<unresolved>",
            problems=(f"config.domains 找不到恰一個 analysis_region_id={domain_id!r} 的區域",),
            ocm_months_found=(),
            nww_months_found=(),
            node_count_grid=None,
            node_count_hvel_sample=None,
            wetdry_elem_values_ok=None,
            nww3_node_coverage_fraction=None,
            forcing_gap_report=None,
            arrival_unsafe_count=None,
            arrival_total_count=None,
        )
    domain_config = domain_matches[0]
    try:
        flow_domain_id = resolve_flow_domain_id(config, domain_id, formal=formal)
    except ValueError as exc:
        return DomainPreflightReport(
            domain_id=domain_id,
            flow_domain_id="<unresolved>",
            problems=(str(exc),),
            ocm_months_found=(),
            nww_months_found=(),
            node_count_grid=None,
            node_count_hvel_sample=None,
            wetdry_elem_values_ok=None,
            nww3_node_coverage_fraction=None,
            forcing_gap_report=None,
            arrival_unsafe_count=None,
            arrival_total_count=None,
        )

    ocm_months: tuple[str, ...] = ()
    nww_months: tuple[str, ...] = ()
    node_count_grid: int | None = None
    node_count_hvel_sample: int | None = None
    wetdry_ok: bool | None = None
    coverage_fraction: float | None = None
    gap_report: ForcingGapInventoryReport | None = None
    mesh: NativeMesh | None = None

    ocm_native_root: Path | None = None
    try:
        ocm_native_root = Path(
            _require_env(env, config.inputs.ocm_native_root_env, purpose="OCM 原生產品 root")
        )
    except WiringError as exc:
        problems.append(str(exc))

    if ocm_native_root is not None:
        flow_root = ocm_native_root / flow_domain_id
        grid_dir = flow_root / "grid"
        if not grid_dir.is_dir():
            problems.append(f"缺少 OCM 原生網格資料夾：{grid_dir}")
        else:
            try:
                projection = DomainProjection(*domain_config.center_lonlat)
                mesh = NativeMesh.from_directory(grid_dir, projection=projection)
                node_count_grid = int(mesh.node_xy.shape[0])
            except (OSError, ValueError) as exc:
                problems.append(f"OCM 原生網格載入失敗：{exc}")

        try:
            ocm_months = tuple(_year_window_months(flow_root / "months", config.inputs.years))
        except WiringError as exc:
            problems.append(str(exc))

        if ocm_months:
            # 逐月檢查必要動態陣列都存在（刻意不含 diffusivity.npy——本專案
            # 逆向驅動層不讀 OCM 原生擴散係數，見 build_domain_resources 與
            # forcing_bridge 模組文件「load_diffusivity 開關」一節）。這裡
            # 獨立於下面的 hvel/wetdry 抽樣檢查先跑一次，即使只抽查參考月份
            # 也能一次列出「哪個月份缺哪個檔案」，不必等 inventory_forcing_
            # gaps 在缺 time_utc_ns.npy 時才間接暴露問題。
            problems.extend(_check_required_ocm_month_files(flow_root, ocm_months))

        if mesh is not None and ocm_months:
            sample_month_dir = flow_root / "months" / ocm_months[0]
            # 先只窺視 hvel 的節點軸（不需要完整讀取 hvel 陣列），因為節點數
            # 若與 mesh 不符，後續的 wetdry 抽樣檢查也沒有意義——先用
            # ``np.load`` 讀 shape（不讀資料本體，記憶體映射，成本可忽略）
            # 比對節點數，節點數若不符就不再嘗試往下讀，直接報告；節點數
            # 相符才繼續走 wetdry 檢查。刻意不透過 ``forcing.OCMNativeMonth``
            # ——該類別固定要求 diffusivity.npy 存在，本機一次性複製的月份
            # 副本刻意不含這個檔案（見 build_domain_resources 同一段說明），
            # 因此這裡與它一樣改成只讀 wetdry_elem.npy 的最小讀取。
            try:
                sample_hvel_shape = np.load(
                    sample_month_dir / "hvel.npy", mmap_mode="r", allow_pickle=False
                ).shape
            except OSError as exc:
                problems.append(f"OCM 樣本月份 {ocm_months[0]} 的 hvel.npy 無法讀取：{exc}")
                sample_hvel_shape = None
            if sample_hvel_shape is not None:
                node_count_hvel_sample = int(sample_hvel_shape[1]) if len(sample_hvel_shape) > 1 else None
                if node_count_hvel_sample != node_count_grid:
                    problems.append(
                        f"OCM 樣本月份 {ocm_months[0]} 的 hvel 節點軸 "
                        f"{node_count_hvel_sample} 與網格節點數 {node_count_grid} 不一致"
                    )
                else:
                    try:
                        wetdry_path = sample_month_dir / "wetdry_elem.npy"
                        wetdry_sample = np.asarray(
                            np.load(wetdry_path, mmap_mode="r", allow_pickle=False)[0], dtype=np.float64
                        )
                        finite_values = wetdry_sample[np.isfinite(wetdry_sample)]
                        wetdry_ok = bool(np.all(np.isin(finite_values, (0.0, 1.0))))
                        if not wetdry_ok:
                            problems.append(
                                f"OCM 樣本月份 {ocm_months[0]} 的 wetdry_elem 出現非 {{0,1,NaN}} 的值"
                            )
                    except OSError as exc:
                        problems.append(f"OCM 樣本月份 {ocm_months[0]} 載入失敗：{exc}")

        if ocm_months:
            years = sorted({int(year) for year in config.inputs.years})
            expected_start_ns = int(datetime(years[0], 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
            expected_end_ns = int(
                datetime(years[-1], 12, 31, 23, 0, tzinfo=UTC).timestamp() * 1_000_000_000
            )
            month_dirs = {month_id: str(flow_root / "months" / month_id) for month_id in ocm_months}
            reconstruction_root_env = config.inputs.ocm_reconstruction_root_env
            reconstruction_root = env.get(reconstruction_root_env) if reconstruction_root_env else None
            patches: dict[str, OCMReconstructionPatchMonth] = {}
            if reconstruction_root and mesh is not None:
                patches, _ = _load_reconstruction_patches(
                    reconstruction_root, flow_domain_id=flow_domain_id, months=ocm_months, mesh=mesh
                )
            try:
                gap_report = inventory_forcing_gaps(
                    month_dirs,
                    expected_start_utc_ns=expected_start_ns,
                    expected_end_utc_ns=expected_end_ns,
                    patches=patches,
                    flow_id=flow_domain_id,
                )
            except (OSError, ValueError) as exc:
                problems.append(f"缺口盤點失敗：{exc}")

    nww_native_root: Path | None = None
    try:
        nww_native_root = Path(
            _require_env(env, config.inputs.nww_analysis_root_env, purpose="NWW3 分析產品 root")
        )
    except WiringError as exc:
        problems.append(str(exc))

    if nww_native_root is not None:
        nww_flow_root = nww_native_root / flow_domain_id
        nww_grid_dir = nww_flow_root / "grid"
        if not nww_grid_dir.is_dir():
            problems.append(f"缺少 NWW3 網格資料夾：{nww_grid_dir}")
        else:
            try:
                nww_months = tuple(_year_window_months(nww_flow_root / "months", config.inputs.years))
            except WiringError as exc:
                problems.append(str(exc))
            if mesh is not None:
                try:
                    nww_lon = np.asarray(
                        np.load(nww_grid_dir / "lon.npy", mmap_mode="r", allow_pickle=False)
                    )
                    nww_lat = np.asarray(
                        np.load(nww_grid_dir / "lat.npy", mmap_mode="r", allow_pickle=False)
                    )
                    weights = build_nww3_node_interpolation_weights(
                        mesh.node_lon, mesh.node_lat, nww_lon, nww_lat
                    )
                    coverage_fraction = float(np.mean(weights.node_covered_by_nww3))
                except (OSError, ValueError) as exc:
                    problems.append(f"NWW3 涵蓋率計算失敗：{exc}")

    arrival_unsafe_count: int | None = None
    arrival_total_count: int | None = None
    if gap_report is not None and arrival_times_by_site:
        site_ids = {
            site.study_site_id for site in config.study_sites if site.analysis_region_id == domain_id
        }
        relevant_arrivals = [
            arrival for site_id in site_ids for arrival in arrival_times_by_site.get(site_id, ())
        ]
        if relevant_arrivals:
            times = np.asarray([item.time_utc_ns for item in relevant_arrivals], dtype=np.int64)
            safe_mask = derive_backward_window_available(
                times, gap_report, max_backtrack_seconds=max_backtrack_seconds
            )
            arrival_total_count = int(times.size)
            arrival_unsafe_count = int(np.count_nonzero(~safe_mask))
            if arrival_unsafe_count:
                problems.append(
                    f"{arrival_unsafe_count}/{arrival_total_count} 個到達時刻的 30 天回溯窗"
                    "落在未登錄缺口內，需要重選或補齊重建"
                )

    return DomainPreflightReport(
        domain_id=domain_id,
        flow_domain_id=flow_domain_id,
        problems=tuple(problems),
        ocm_months_found=ocm_months,
        nww_months_found=nww_months,
        node_count_grid=node_count_grid,
        node_count_hvel_sample=node_count_hvel_sample,
        wetdry_elem_values_ok=wetdry_ok,
        nww3_node_coverage_fraction=coverage_fraction,
        forcing_gap_report=gap_report,
        arrival_unsafe_count=arrival_unsafe_count,
        arrival_total_count=arrival_total_count,
    )


def preflight_native_inputs(
    config_path_or_config: str | Path | ProjectConfig,
    *,
    env: Mapping[str, str] = os.environ,
    config_path: str | Path | None = None,
    domain_ids: Sequence[str] | None = None,
    formal: bool = True,
    max_backtrack_seconds: float = 30.0 * 86400.0,
) -> PreflightReport:
    """正式開跑前的第一道前置檢查：目錄、節點數、wetdry 編碼、NWW3 涵蓋率、缺口與到達窗。

    這是背景需求明確要求的「伺服器上第一步」：在花任何時間建立網格、驅動層或
    真正呼叫 BayTrace kernel 之前，先用秒級／分鐘級的檢查找出資料契約問題。
    每個流場區域的檢查彼此獨立、互不中止——即使某個區域完全找不到 OCM 目錄，
    其餘區域仍會照常檢查並回報。

    重用清單：``forcing_bridge.inventory_forcing_gaps``（缺口盤點，只讀時間軸）、
    ``orchestrator.derive_backward_window_available``（到達時刻回溯窗是否受
    缺口影響）、``wave_bridge.build_nww3_node_interpolation_weights``（NWW3
    空間涵蓋率）；本函式不重新定義任何一項判定規則。

    Args:
        config_path_or_config: 設定 YAML 路徑，或已載入的 ``ProjectConfig``。
        env: 環境變數 mapping，語意與 ``build_domain_resources`` 相同。
        config_path: ``config_path_or_config`` 是 ``ProjectConfig`` 時的相對
            路徑解析基準。
        domain_ids: 要檢查的 ``analysis_region_id`` 清單；``None`` 時檢查
            ``config.domains`` 全部區域。
        formal: 透傳給 ``resolve_flow_domain_id``。
        max_backtrack_seconds: 到達時刻回溯窗長度，預設 30 天，與工作書一致。

    Returns:
        ``PreflightReport``；呼叫 ``.describe()`` 取得人類可讀報告，
        ``.is_run_ready`` 取得單一布林總結。
    """

    config, resolved_config_path = _resolve_config(config_path_or_config, config_path=config_path)
    target_domain_ids = (
        list(domain_ids) if domain_ids is not None else [item.analysis_region_id for item in config.domains]
    )
    arrival_times_by_site = _try_load_arrival_times_by_site(config, resolved_config_path)

    reports = tuple(
        _preflight_one_domain(
            config,
            domain_id,
            env=env,
            formal=formal,
            max_backtrack_seconds=max_backtrack_seconds,
            arrival_times_by_site=arrival_times_by_site,
        )
        for domain_id in target_domain_ids
    )
    return PreflightReport(domain_reports=reports)
