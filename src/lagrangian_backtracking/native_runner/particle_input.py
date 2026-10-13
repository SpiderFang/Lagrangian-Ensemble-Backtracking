"""把 LBT 既有的情境矩陣與受體機制，轉成 BayTrace 原版批次核心要的粒子輸入。

背景與邊界
----------
工作書的情境矩陣是「每站 10 種沉降速度 × 20 個受體位置 × 50 個到達時刻 =
10,000 個基礎情境」，五站共 50,000 個基礎情境；每個基礎情境外面再配置 ``M``
個獨立隨機系集成員（正式設計 ``members_per_scenario=10``），因此全案軌跡數是
``50,000 × M``（``M=1`` 時 50,000 顆，``M=10`` 時 500,000 顆）。這些數字、
``scenario_id``／``particle_id``／亂數種子的推導規則，全部已經由既有
``scenarios.py``（``build_scenarios``、``derive_member_seed``、
``stable_identifier``）與 ``runner.py``（``ScenarioShard``、``iter_run_units``）
定義並測試過；本模組**完全重用**這兩個模組的輸出，不重新實作任何科學或亂數
邏輯，只負責「把它們的輸出，轉成 BayTrace 原版批次核心可以直接吃的
struct-of-arrays（SoA）格式」，並提供一張不可變的「粒子身分表」讓後處理
（工作包 WP-6）可以把軌跡結果對回情境。

受體數量與工作書一致性
------------------------
既有受體機制（``receptors.py`` 的
``select_horizontal_receptors_random_from_pool``／
``sample_random_vertical_draws``，以及 ``input_derivation.py`` 的
``EXPECTED_RECEPTOR_COUNT = 100``）是「每站 5 個水平位置 × 4 個垂向位置 =
20 個受體」，五站合計 100 個受體。這與工作書「每站 20 個受體位置」完全一致
——工作書的「20 種受體」不是另一組獨立設計，而正是「5 水平 × 4 垂向」交叉
出的 20 個三維受體位置。``configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml``
的 ``horizontal_receptor_count_per_site: 5``／
``vertical_receptor_count_per_horizontal_location: 4`` 與此一致，本模組不
另外定義或假設不同的受體計數。

本模組刻意不做的事
--------------------
1. 不重新計算 receptor 的實際三維初始位置。水平位置（公尺制 ``x_m``／``y_m``）
   來自 ``receptors.HorizontalReceptor``，垂向位置（``z_m_positive_up``）來自
   ``scenarios.ReceptorArrivalInitialCondition``（每個 receptor×arrival pair 由
   OCM 當時的 ``eta``、``zcor``、wet/dry 狀態推導的實際初始深度）；這兩者的
   合併查詢由呼叫端以 ``InitialPositionResolver`` 注入，本模組只消費結果，
   不假設任何特定的 OCM 查詢方式，也不在找不到位置時以 0 或最近值頂替。
2. 不呼叫 BayTrace 的 ``find_initial_elements``（``tracker.py`` 約 437 行）。
   該函式需要完整的 ``NativeGridArrays``（見 ``native_runner.grid_bridge``），
   屬於驅動層（WP-5）在取得網格後才能做的事；本模組產出的 ``elem`` 一律是
   ``-1``「待定位」哨兵值，與 BayTrace 原版對「尚未定位成功」的慣例一致
   （``tracker.py`` 的 ``elem_id = np.full(n_particles, -1, ...)``）。
3. 不呼叫或修改 BayTrace 任何原始碼；``BayTraceParticleSoA`` 只是資料容器，
   真正把陣列填入 ``baytrace.particle_arrays.ParticleArrays`` 或呼叫任何
   ``batch_interp``／``tracker`` 函式，都是 WP-5 driver 的工作。

沉降速度與 BayTrace RK4 的相容性（重要科學結論，見下方詳述）
--------------------------------------------------------------
BayTrace 原版沒有「逐粒子垂向速度」欄位；``ParticleArrays`` 的 ``w`` 只是
「目前速度快取」，由 ``batch_interpolate_velocity`` 對 OCM 流場內插得到。
本模組因此在 SoA 輸出中額外提供 ``settling_velocity_mps``（逐粒子沉降速度，
不是 BayTrace 原生欄位名稱），供 WP-5 driver 依此分組後，用「把沉降速度疊加
進共用的 ``w_field1``／``w_field2`` 網格陣列副本」的方式注入，而不是疊加進
``pa.w``。理由見模組尾端 ``評估：沉降速度疊加進 pa.w 是否正確`` 一節；結論是
**疊加進 ``pa.w`` 不正確**，只會讓 RK4 的第一階（k1）看到沉降貢獻，第二至
四階會被 ``batch_move_rk4`` 內部呼叫的 ``_interp_velocity_at`` 用純 OCM 流場
覆寫掉，等同於把沉降速度的權重從應有的 6/6 削減成 1/6。正確做法是在呼叫
``batch_move_rk4`` 之前，對該次呼叫要用的 ``w_field1``／``w_field2`` 陣列先各自
加上「這一組粒子共用的沉降速度」常數（沉降速度不隨空間或時間變化，加常數後
仍是合法的 ``(nvrt, nnode)`` 陣列），因為 ``_interp_velocity_at`` 在全部四個
RK4 階段都是直接讀這兩個陣列，這樣沉降速度才會正確出現在每一階，與 LBT 自己
既有 NumPy 參考引擎（``integrators.py`` 的 ``VelocityProvider``，其回傳的
``w_mps`` 本來就要求已經是「海流＋沉降合成後」的總速度，見該模組
docstring）的物理契約一致。這個做法要求同一次 ``batch_move_rk4`` 呼叫內的
粒子共用同一個沉降速度，因此「依沉降速度分組」不只是效能考量，而是物理
正確性的必要條件——詳見下方分批策略一節。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np

from ..runner import RunUnit, ScenarioShard, iter_run_units

__all__ = [
    "ArrivalTimeResolver",
    "LabelResolver",
    "InitialPositionResolver",
    "ParticleIdentityTable",
    "ParticleInputError",
    "BayTraceParticleSoA",
    "BatchGroup",
    "BatchLayoutReport",
    "build_particle_identity_table",
    "build_baytrace_particle_soa",
    "group_indices_by_settling_velocity",
    "group_indices_by_arrival_time",
    "group_indices_by_settling_velocity_and_arrival_time",
    "summarize_batch_layout",
    "validate_batch_partition",
    "validate_arrival_time_homogeneous_batches",
    "validate_settling_velocity_homogeneous_batches",
    "save_particle_identity_table_npz",
    "load_particle_identity_table_npz",
]


class ParticleInputError(ValueError):
    """粒子輸入建構或分批驗證階段的一般性資料契約錯誤。

    刻意獨立於單純 ``ValueError``，讓呼叫端可以精確攔截「粒子輸入橋接失敗」
    而不誤吞其他來源的 ``ValueError``；仍繼承 ``ValueError`` 以相容既有防呆。
    """


InitialPositionResolver = Callable[[RunUnit], tuple[float, float, float]]
"""給定一個既有 ``RunUnit``（已含固定 ``scenario``、``member_id``、
``particle_id``、``seed``），回傳這條粒子逆向追蹤起點的公尺制三維位置
``(x_m, y_m, z_m_positive_up)``。

``x_m``／``y_m`` 應來自 ``receptors.HorizontalReceptor`` 對應
``unit.scenario.receptor_id`` 的水平位置；``z_m_positive_up`` 應來自
``scenarios.ReceptorArrivalInitialCondition`` 對應
``(receptor_id, arrival_time_id)`` 這個 pair 在該到達時刻由 OCM 推導出的實際
初始深度（見該類別 docstring）。本模組不規定呼叫端如何建立這個查詢（可以是
一次性建好的 dict、也可以是延遲查詢的 callable），只要求：

1. 找不到位置時必須丟出例外（fail-closed），不得回傳 0 或任何頂替值；
2. 回傳的三個數值必須是有限實數；
3. 同一個 ``(study_site_id, receptor_id, arrival_time_id)`` 組合每次呼叫都
   回傳相同結果（可重現性），因為本模組只呼叫一次並把結果寫入不可變身分表。
"""


ArrivalTimeResolver = Callable[[RunUnit], int]
"""WP-51：逐粒子解析「起算（沉底）UTC 奈秒」的選用 callable。

per_observation 時每個情境只有一個起算時刻，直接取 ``unit.scenario.arrival_time_utc_ns``，
不需要這個 resolver。per_member 時同一個情境的 M 個成員各自有不同的沉底時刻，
由 ``(unit.scenario.arrival_time_id, unit.member_id)`` 查得；回傳值必須是落在整點的
正整數 UTC 奈秒，找不到時必須丟出例外（fail-closed），不得回傳情境的觀測時刻頂替。
"""

LabelResolver = Callable[[RunUnit], "dict[str, object]"]
"""WP-51：逐粒子解析附加標籤欄位（例如沉底時刻的季節／大小潮／潮相／事件旗標）。

回傳 mapping 的鍵集合在所有粒子上必須完全相同；值只能是 ``str``、``bool`` 或
``int``，同一個鍵的值型別必須一致（字串→固定寬度 unicode、布林→bool、整數→int64）。
標籤只進入身分表的選用 ``labels`` 欄位，且只在存在時才進入內容雜湊，因此沒有標籤的
既有流程（per_observation）雜湊與先前位元相同。
"""

_LABEL_STR_DTYPE = "<U32"


def _labels_to_arrays(rows: Sequence[dict[str, object]]) -> dict[str, np.ndarray]:
    """把逐粒子標籤 dict 序列轉為「欄名 → 一維陣列」，並檢查鍵集合與型別一致。"""

    if not rows:
        raise ParticleInputError("標籤列不可為空")
    keys = sorted(rows[0])
    if not keys:
        raise ParticleInputError("標籤 resolver 回傳空 mapping")
    key_set = set(keys)
    if any(row.keys() != key_set for row in rows):
        raise ParticleInputError("標籤欄位的鍵集合在粒子間不一致")
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        column = [row[key] for row in rows]
        first = column[0]
        if isinstance(first, bool):
            if any(not isinstance(value, bool) for value in column):
                raise ParticleInputError(f"標籤 {key!r} 的值型別在粒子間不一致（應為 bool）")
            arrays[key] = np.asarray(column, dtype=np.bool_)
        elif isinstance(first, int):
            if any(isinstance(value, bool) or not isinstance(value, int) for value in column):
                raise ParticleInputError(f"標籤 {key!r} 的值型別在粒子間不一致（應為 int）")
            arrays[key] = np.asarray(column, dtype=np.int64)
        elif isinstance(first, str):
            if any(not isinstance(value, str) or not value for value in column):
                raise ParticleInputError(f"標籤 {key!r} 必須是非空字串")
            if max(len(value) for value in column) > 32:
                raise ParticleInputError(f"標籤 {key!r} 超過 {_LABEL_STR_DTYPE} 寬度")
            arrays[key] = np.asarray(column, dtype=_LABEL_STR_DTYPE)
        else:
            raise ParticleInputError(f"標籤 {key!r} 的值型別不支援：{type(first).__name__}")
    return arrays


# ── 粒子身分表：固定欄位、內容雜湊、可序列化保存 ──────────────────────────

# 各字串欄位的固定寬度 numpy dtype。使用固定寬度 unicode 字串（而非 object
# dtype）才能以 ``np.savez`` 無 pickle 地序列化，並維持 C-contiguous 記憶體
# 配置；寬度留有餘裕（既有 stable_identifier 預設輸出約 27-28 字元，
# material_id／arrival_time_id 等語意字串可能更長），但刻意不無限放寬——numpy
# 的固定寬度 unicode 字元一律以 4 位元組（UCS4）儲存，寬度每加倍，這七個字串
# 欄位在 50 萬列規模下的記憶體就多一份數百 MB，因此寬度是「安全但不浪費」的
# 折衷，超出寬度一律在 ``_as_id_array`` fail-closed，不會靜默截斷。
_ID_DTYPE = "<U64"
_ID_WIDTH = 64
_SEED_HEX_DTYPE = "<U32"  # derive_member_seed 是 128-bit，恰好 32 個十六進位字元


def _as_id_array(values: Sequence[str]) -> np.ndarray:
    """把字串序列轉成固定寬度 unicode 陣列，並拒絕超出寬度或非字串內容。

    刻意避免「轉成 numpy 陣列後逐一 ``str()`` 解碼回來比對」的作法：那個寫法
    在 50 萬列規模下，等於對每個字串欄位多做一次完整的 numpy→Python 字串
    往返轉換，實測是本模組建構耗時的主要來源之一。改成只計算每個字串的
    ``len()``（不做任何解碼或往返比較）就足以偵測「超出固定寬度會被 numpy
    靜默截斷」的唯一風險，同時仍然保留 fail-closed 語意：找到任何非字串、
    空字串或超長字串都會在真正配置陣列前直接拒絕。
    """

    lengths = [len(value) if isinstance(value, str) else -1 for value in values]
    for index, length in enumerate(lengths):
        if length <= 0:
            raise ParticleInputError(f"識別碼欄位第 {index} 列必須是非空字串")
    max_length = max(lengths)
    if max_length > _ID_WIDTH:
        raise ParticleInputError(
            f"識別碼長度 {max_length} 超出固定欄位寬度 {_ID_DTYPE}，"
            "請放寬 _ID_DTYPE 或檢查上游是否產生異常長字串"
        )
    return np.asarray(values, dtype=_ID_DTYPE)


@dataclass(frozen=True, slots=True)
class ParticleIdentityTable:
    """全案粒子的不可變身分表：一列一顆粒子，是後處理對回情境的唯一依據。

    欄位設計沿用既有 ``models.ParticleState``、``scenarios.Scenario`` 與
    ``runner.RunUnit`` 已定義的身分欄位語意，不重新命名或改變其意義：
    ``particle_id``／``scenario_id``／``member_id``／``study_site_id``／
    ``analysis_region_id``／``receptor_id`` 與 ``ParticleState`` 完全對應；
    ``material_id``／``settling_velocity_mps``／``arrival_time_id``／
    ``arrival_time_utc_ns`` 與 ``Scenario`` 完全對應；``seed_hex`` 是
    ``RunUnit.seed``（128-bit 整數）以 32 位十六進位字串保存，格式與
    ``receptors.RandomVerticalDraw.derived_seed_hex`` 一致，避免把超過 64 位元
    的種子塞進 numpy 原生整數 dtype 而截斷。

    ``x0_m``／``y0_m``／``z0_m`` 是逆向追蹤的公尺制起點（``z0_m`` 海面為零、
    向上為正，與 ``models.py`` 全案慣例一致）；這是本表**唯一**新增、既有
    ``RunUnit`` 沒有的資訊，由建構時注入的 ``InitialPositionResolver`` 提供。

    ``content_sha256`` 在建構完成後才計算（見 ``__post_init__``），涵蓋所有
    陣列欄位的原始位元組與識別性中繼資料（``design_version`` 由呼叫端經
    ``experiment_case_id`` 間接保證一致，見 ``_derive_content_hash``）；同一組
    輸入（情境清單、``members_per_scenario``、``master_seed``、
    ``random_stream_id``、位置解析結果）重建時必須得到完全相同的雜湊，這是
    「建構時一次驗證＋內容雜湊取代每步驗證」政策的核心：下游只需比對這個
    雜湊，不必像 ``production.py`` 的 ``_validate_alignment`` 那樣每個 sweep
    都逐列核對。

    ``bed_no_deposition``（WP-26 新增、WP-28 正名接線，見
    ``native_runner.backward_driver.BackwardDriverConfig.bed_no_deposition``
    docstring）**刻意不**是本表的欄位、也不參與 ``content_sha256`` 的計算：
    這個開關決定的是「粒子軌跡回溯到接觸海床時要不要終止」，屬於回溯動力學
    層級的執行時設定，與本表描述的「粒子從哪裡出發、屬於哪個實驗案例（是否
    含 Stokes，即 ``experiment_case_id``）」完全無關——同一批粒子起點與種子
    本來就應該同時餵給 ``bed_no_deposition=False``（基線）與 ``True``（敏感
    度案例）兩次執行，才能在完全相同的粒子部署下比較「是否關掉沉積終止判定」
    這一個變因；若把它納入內容雜湊，會強迫兩種案例各自重建一份內容相同、只
    是雜湊不同的粒子身分表，徒增管理負擔且沒有任何科學上的必要性。兩種案例
    的結果差異改由 ``native_runner.postprocess.PostprocessManifest.config``
    的 ``runtime_bed_no_deposition`` 欄位（來自 ``native-plan`` 計畫檔的
    ``bed_no_deposition`` 區塊）與各自獨立的 ``--output`` 目錄辨識，不依賴
    粒子身分表本身的雜湊差異。
    """

    particle_id: np.ndarray
    scenario_id: np.ndarray
    member_id: np.ndarray
    study_site_id: np.ndarray
    analysis_region_id: np.ndarray
    receptor_id: np.ndarray
    material_id: np.ndarray
    arrival_time_id: np.ndarray
    settling_velocity_mps: np.ndarray
    arrival_time_utc_ns: np.ndarray
    seed_hex: np.ndarray
    x0_m: np.ndarray
    y0_m: np.ndarray
    z0_m: np.ndarray
    experiment_case_id: str
    master_seed: int
    random_stream_id: str | None
    content_sha256: str = field(init=False, default="")
    # WP-51：選用的逐粒子標籤欄位（沉底時刻的 season／tide_class／潮相／事件旗標與
    # 觀測時刻的舊標籤）。None 表示沒有標籤（per_observation 既有流程）；有標籤時每個
    # 陣列都必須是長度 N 的一維陣列，並納入內容雜湊與 .npz。刻意放在 content_sha256
    # 之後的預設值欄位，讓既有以位置或關鍵字建構的呼叫端完全不受影響。
    labels: dict[str, np.ndarray] | None = None
    # WP-54（v7 ``class_candidates_balanced``）：選用的逐粒子實際沉降速度。``settling_velocity_mps``
    # 仍是「類別代表速度」，維持工作單元規劃、分組與後處理依類別分組的語意；這三個欄位
    # 只在 v7 抽樣模式下才有值：
    #   particle_settling_velocity_mps：每顆粒子實際使用的沉降速度（m/s，嚴格為負）；
    #   settling_candidate_index：該速度在類別 10 個候選值中的序號（0 為最慢、9 為最快）；
    #   settling_sampling：抽樣中繼資料（模式、種子、各站各類別計數等，可 JSON 序列化）。
    # 三者必須同時提供或同時省略；省略時內容雜湊、.npz 格式與 WP-54 之前逐位元相同。
    particle_settling_velocity_mps: np.ndarray | None = None
    settling_candidate_index: np.ndarray | None = None
    settling_sampling: dict | None = None

    def __post_init__(self) -> None:
        """一次性驗證欄位形狀、型別、唯一性與有限性，並計算內容雜湊。

        驗證只在建構時執行一次（``O(N)`` 或 ``O(N log N)``），不像
        ``ProductionBatch._validate_alignment`` 在每個 sweep 前後都重新逐列
        比對；這對 50 萬顆粒子規模而言是必要的效能取捨，但代價是驗證失敗必須
        非常明確地指出哪個欄位、第一個出錯的位置，避免「建構時看起來正確、
        後面才發現對錯位」的情況真的發生。
        """

        n = self.particle_id.shape[0]
        if n < 1:
            raise ParticleInputError("粒子身分表不可為空")

        array_fields = (
            "particle_id", "scenario_id", "member_id", "study_site_id",
            "analysis_region_id", "receptor_id", "material_id",
            "arrival_time_id", "settling_velocity_mps", "arrival_time_utc_ns",
            "seed_hex", "x0_m", "y0_m", "z0_m",
        )
        for name in array_fields:
            array = getattr(self, name)
            if not isinstance(array, np.ndarray) or array.ndim != 1 or array.shape[0] != n:
                raise ParticleInputError(f"{name} 必須是長度 {n} 的一維陣列，實際 {array!r}")
            if not array.flags["C_CONTIGUOUS"]:
                object.__setattr__(self, name, np.ascontiguousarray(array))

        expected_dtypes = {
            "member_id": np.int32,
            "settling_velocity_mps": np.float64,
            "arrival_time_utc_ns": np.int64,
            "x0_m": np.float64,
            "y0_m": np.float64,
            "z0_m": np.float64,
        }
        for name, dtype in expected_dtypes.items():
            array = getattr(self, name)
            if array.dtype != dtype:
                raise ParticleInputError(f"{name} dtype 必須是 {dtype}，實際 {array.dtype}")
        for name in ("particle_id", "scenario_id", "study_site_id", "analysis_region_id",
                     "receptor_id", "material_id", "arrival_time_id"):
            array = getattr(self, name)
            if array.dtype.kind != "U":
                raise ParticleInputError(f"{name} 必須是 unicode 字串陣列，實際 {array.dtype}")
        if self.seed_hex.dtype.kind != "U":
            raise ParticleInputError("seed_hex 必須是 unicode 字串陣列")

        if np.unique(self.particle_id).size != n:
            raise ParticleInputError("particle_id 必須全表唯一，發現重複識別碼")
        if np.any(self.member_id < 0):
            raise ParticleInputError("member_id 不可為負")
        for name in ("settling_velocity_mps", "x0_m", "y0_m", "z0_m"):
            array = getattr(self, name)
            if not np.all(np.isfinite(array)):
                bad = int(np.flatnonzero(~np.isfinite(array))[0])
                raise ParticleInputError(f"{name} 第 {bad} 列含非有限值")
        if np.any(self.arrival_time_utc_ns <= 0):
            raise ParticleInputError("arrival_time_utc_ns 必須是正值 UTC 奈秒")
        seed_hex_lengths = np.char.str_len(self.seed_hex)
        if np.any(seed_hex_lengths != 32):
            raise ParticleInputError("seed_hex 必須恰為 32 個十六進位字元")

        if self.labels is not None:
            if not isinstance(self.labels, dict) or not self.labels:
                raise ParticleInputError("labels 必須是非空 dict 或 None")
            for name, array in self.labels.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(array, np.ndarray)
                    or array.ndim != 1
                    or array.shape[0] != n
                ):
                    raise ParticleInputError(f"labels[{name!r}] 必須是長度 {n} 的一維陣列")
                if array.dtype.kind not in {"U", "b", "i"}:
                    raise ParticleInputError(f"labels[{name!r}] dtype 不支援：{array.dtype}")
                if not array.flags["C_CONTIGUOUS"]:
                    self.labels[name] = np.ascontiguousarray(array)
        per_particle = (
            self.particle_settling_velocity_mps,
            self.settling_candidate_index,
            self.settling_sampling,
        )
        if any(item is not None for item in per_particle):
            if any(item is None for item in per_particle):
                raise ParticleInputError(
                    "particle_settling_velocity_mps／settling_candidate_index／settling_sampling "
                    "必須同時提供或同時省略"
                )
            velocities = self.particle_settling_velocity_mps
            indices = self.settling_candidate_index
            for name, array in (
                ("particle_settling_velocity_mps", velocities),
                ("settling_candidate_index", indices),
            ):
                if not isinstance(array, np.ndarray) or array.ndim != 1 or array.shape[0] != n:
                    raise ParticleInputError(f"{name} 必須是長度 {n} 的一維陣列")
                if not array.flags["C_CONTIGUOUS"]:
                    object.__setattr__(self, name, np.ascontiguousarray(array))
            velocities = self.particle_settling_velocity_mps
            indices = self.settling_candidate_index
            if velocities.dtype != np.float64:
                raise ParticleInputError("particle_settling_velocity_mps dtype 必須是 float64")
            if indices.dtype != np.int16:
                raise ParticleInputError("settling_candidate_index dtype 必須是 int16")
            if not np.all(np.isfinite(velocities)) or np.any(velocities >= 0.0):
                raise ParticleInputError("particle_settling_velocity_mps 必須全部是嚴格小於 0 的有限值")
            if np.any(indices < 0):
                raise ParticleInputError("settling_candidate_index 不可為負")
            if not isinstance(self.settling_sampling, dict) or not self.settling_sampling:
                raise ParticleInputError("settling_sampling 必須是非空 dict")
            try:
                json.dumps(self.settling_sampling, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ParticleInputError(f"settling_sampling 必須可序列化成 JSON：{error}") from error
        if not isinstance(self.experiment_case_id, str) or not self.experiment_case_id.strip():
            raise ParticleInputError("experiment_case_id 不可空白")
        seed = self.master_seed
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ParticleInputError("master_seed 必須是非負整數")
        if self.random_stream_id is not None and (
            not isinstance(self.random_stream_id, str) or not self.random_stream_id.strip()
        ):
            raise ParticleInputError("random_stream_id 必須是非空字串或 None")

        object.__setattr__(self, "content_sha256", _compute_identity_table_hash(self))

    def __len__(self) -> int:
        """回傳粒子總數，方便 ``len(table)`` 之類的直覺用法。"""

        return int(self.particle_id.shape[0])


def _compute_identity_table_hash(table: ParticleIdentityTable) -> str:
    """對身分表所有陣列欄位與識別性中繼資料計算穩定的 SHA-256 雜湊。

    雜湊材料依固定欄位順序串接每個陣列的 dtype 字串、shape 與原始位元組
    （``tobytes()``；呼叫前已在 ``__post_init__`` 確保 C-contiguous），並在最後
    附上 ``experiment_case_id``／``master_seed``／``random_stream_id``／粒子數
    這些不在陣列裡、但同樣會改變「這批粒子代表哪個 run」的識別性欄位。
    """

    digest = sha256()
    array_fields = (
        "particle_id", "scenario_id", "member_id", "study_site_id",
        "analysis_region_id", "receptor_id", "material_id",
        "arrival_time_id", "settling_velocity_mps", "arrival_time_utc_ns",
        "seed_hex", "x0_m", "y0_m", "z0_m",
    )
    for name in array_fields:
        array = getattr(table, name)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    if table.labels is not None:
        # 只有存在標籤時才納入雜湊；沒有標籤的身分表雜湊與 WP-51 之前逐位元相同。
        for name in sorted(table.labels):
            array = table.labels[name]
            digest.update(b"label:" + name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
    metadata = {
        "experiment_case_id": table.experiment_case_id,
        "master_seed": int(table.master_seed),
        "random_stream_id": table.random_stream_id,
        "particle_count": int(table.particle_id.shape[0]),
    }
    if table.particle_settling_velocity_mps is not None:
        # WP-54：只有 v7 逐粒子沉降速度存在時才納入雜湊；沒有時雜湊與先前逐位元相同。
        for name in ("particle_settling_velocity_mps", "settling_candidate_index"):
            array = getattr(table, name)
            digest.update(b"per_particle:" + name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        metadata["settling_sampling"] = table.settling_sampling
    canonical_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest.update(canonical_metadata.encode("utf-8"))
    return digest.hexdigest()


def build_particle_identity_table(
    shards: Sequence[ScenarioShard],
    *,
    master_seed: int,
    resolve_initial_position: InitialPositionResolver,
    random_stream_id: str | None = None,
    resolve_arrival_time_utc_ns: ArrivalTimeResolver | None = None,
    resolve_labels: LabelResolver | None = None,
) -> ParticleIdentityTable:
    """把一或多個既有 ``ScenarioShard`` 展開成全案粒子身分表。

    WP-51 新增兩個選用 resolver（預設 None，行為與先前完全相同）：
    ``resolve_arrival_time_utc_ns`` 讓每顆粒子自己的起算（沉底）時刻取代情境的
    ``arrival_time_utc_ns``（per_member 用）；``resolve_labels`` 提供逐粒子標籤欄位。
    ``resolve_initial_position`` 的介面仍是 ``(RunUnit) -> (x, y, z)``，RunUnit 已含
    ``member_id``，因此 per_member 的逐成員水平位置（延後項目）只需改 resolver 內容，
    不必改這個介面。

    這是本模組唯一的建構入口，刻意只接受既有 ``runner.ScenarioShard``（而不是
    重新接受原始 ``Scenario`` 清單），因為 shard 切分、排序與批次識別碼已經是
    既有正式流程（``runner.plan_scenario_shards``）的產物；本函式只需要依
    ``iter_run_units`` 既定順序逐一展開，不重新排序、不重新分組。

    參數
    ------
    shards:
        依 ``plan_scenario_shards`` 或等價方式建立的批次序列；函式依傳入順序
        逐一處理，不去重、不排序。所有 shard 必須共用同一個
        ``experiment_case_id``（正式流程本來就是同一個物理案例切成多個
        shard），否則視為呼叫端誤把不同案例混在一次建構中，直接拒絕。
    master_seed:
        與 ``iter_run_units`` 相同的主種子，直接透傳，不在本函式做任何轉換。
    resolve_initial_position:
        見 ``InitialPositionResolver``；每個 ``RunUnit`` 只呼叫一次。
    random_stream_id:
        與 ``iter_run_units`` 相同的可選共同亂數流命名空間，直接透傳。

    回傳
    ------
    ``ParticleIdentityTable``，列數恰為 ``sum(shard.particle_count for shard
    in shards)``；若非如此會在下方直接以 ``RuntimeError`` 攔截，因為那代表
    ``iter_run_units`` 產生的數量與 shard 自己宣稱的 ``particle_count`` 不一致
    ——這種不一致理論上不該發生（``ProductionBatch`` 也有相同的防禦性檢查），
    但一旦發生就是嚴重的資料契約錯誤，不能只靠事後的雜湊比對發現。
    """

    if not shards:
        raise ParticleInputError("shards 不可為空")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ParticleInputError("master_seed 必須是非負整數")
    if not callable(resolve_initial_position):
        raise ParticleInputError("resolve_initial_position 必須是 callable")

    experiment_case_id = shards[0].experiment_case_id
    for shard in shards:
        if shard.experiment_case_id != experiment_case_id:
            raise ParticleInputError(
                "所有 shard 必須共用同一個 experiment_case_id，"
                f"發現 {shard.experiment_case_id!r} != {experiment_case_id!r}；"
                "不同物理案例請分別建立粒子身分表，不要混在同一次建構"
            )

    expected_total = sum(shard.particle_count for shard in shards)

    particle_id: list[str] = []
    scenario_id: list[str] = []
    member_id: list[int] = []
    study_site_id: list[str] = []
    analysis_region_id: list[str] = []
    receptor_id: list[str] = []
    material_id: list[str] = []
    arrival_time_id: list[str] = []
    settling_velocity_mps: list[float] = []
    arrival_time_utc_ns: list[int] = []
    label_rows: list[dict[str, object]] = []
    seed_hex: list[str] = []
    x0_m: list[float] = []
    y0_m: list[float] = []
    z0_m: list[float] = []

    for shard in shards:
        for unit in iter_run_units(shard, master_seed=master_seed, random_stream_id=random_stream_id):
            scenario = unit.scenario
            x_value, y_value, z_value = resolve_initial_position(unit)
            x_value, y_value, z_value = float(x_value), float(y_value), float(z_value)
            if not (np.isfinite(x_value) and np.isfinite(y_value) and np.isfinite(z_value)):
                raise ParticleInputError(
                    f"粒子 {unit.particle_id} 的 resolve_initial_position 回傳非有限位置"
                )
            particle_id.append(unit.particle_id)
            scenario_id.append(scenario.scenario_id)
            member_id.append(unit.member_id)
            study_site_id.append(scenario.study_site_id)
            analysis_region_id.append(scenario.analysis_region_id)
            receptor_id.append(scenario.receptor_id)
            material_id.append(scenario.material_id)
            arrival_time_id.append(scenario.arrival_time_id)
            settling_velocity_mps.append(scenario.settling_velocity_mps)
            if resolve_arrival_time_utc_ns is None:
                arrival_time_utc_ns.append(scenario.arrival_time_utc_ns)
            else:
                resolved_utc = resolve_arrival_time_utc_ns(unit)
                if isinstance(resolved_utc, bool) or not isinstance(resolved_utc, (int, np.integer)):
                    raise ParticleInputError(
                        f"粒子 {unit.particle_id} 的 resolve_arrival_time_utc_ns 必須回傳整數"
                    )
                arrival_time_utc_ns.append(int(resolved_utc))
            if resolve_labels is not None:
                label_rows.append(dict(resolve_labels(unit)))
            seed_hex.append(f"{unit.seed:032x}")
            x0_m.append(x_value)
            y0_m.append(y_value)
            z0_m.append(z_value)

    if len(particle_id) != expected_total:
        raise RuntimeError(
            f"iter_run_units 產生 {len(particle_id)} 顆粒子，與 shard.particle_count 總和 "
            f"{expected_total} 不一致，這是嚴重的資料契約錯誤"
        )

    return ParticleIdentityTable(
        particle_id=_as_id_array(particle_id),
        scenario_id=_as_id_array(scenario_id),
        member_id=np.asarray(member_id, dtype=np.int32),
        study_site_id=_as_id_array(study_site_id),
        analysis_region_id=_as_id_array(analysis_region_id),
        receptor_id=_as_id_array(receptor_id),
        material_id=_as_id_array(material_id),
        arrival_time_id=_as_id_array(arrival_time_id),
        settling_velocity_mps=np.asarray(settling_velocity_mps, dtype=np.float64),
        arrival_time_utc_ns=np.asarray(arrival_time_utc_ns, dtype=np.int64),
        seed_hex=np.asarray(seed_hex, dtype=_SEED_HEX_DTYPE),
        x0_m=np.asarray(x0_m, dtype=np.float64),
        y0_m=np.asarray(y0_m, dtype=np.float64),
        z0_m=np.asarray(z0_m, dtype=np.float64),
        experiment_case_id=experiment_case_id,
        master_seed=master_seed,
        random_stream_id=random_stream_id,
        labels=_labels_to_arrays(label_rows) if resolve_labels is not None else None,
    )


# ── 序列化：.npz 保存與重建，重建後即時核對內容雜湊 ────────────────────────


def save_particle_identity_table_npz(table: ParticleIdentityTable, path: str | Path) -> None:
    """把身分表保存成單一 ``.npz`` 檔，內含所有陣列欄位與可稽核中繼資料。

    使用 ``np.savez``（不壓縮）而非 ``savez_compressed``：50 萬列規模下所有
    陣列合計約數十 MB，壓縮省下的磁碟空間相對於正式 run 的其他輸出微不足道，
    但壓縮／解壓縮會拖慢每次重跑驗證雜湊的速度，不符合本表「快速重建、快速
    比對雜湊」的用途。中繼資料（``experiment_case_id``、``master_seed``、
    ``random_stream_id``、已計算的 ``content_sha256``）以 JSON 字串存進一個
    額外的 0 維陣列 ``__metadata__``，讀回時只需字串解析，不需要另外的
    schema 檔案。
    """

    metadata = {
        "experiment_case_id": table.experiment_case_id,
        "master_seed": int(table.master_seed),
        "random_stream_id": table.random_stream_id,
        "content_sha256": table.content_sha256,
        "particle_count": len(table),
    }
    label_arrays: dict[str, np.ndarray] = {}
    if table.labels is not None:
        # 標籤欄位以 ``label__`` 前綴存放；只在有標籤時才增加這些項目與 metadata 鍵，
        # 沒有標籤的檔案格式與先前完全相同。
        metadata["label_names"] = sorted(table.labels)
        label_arrays = {f"label__{name}": array for name, array in table.labels.items()}
    if table.particle_settling_velocity_mps is not None:
        # WP-54：逐粒子沉降速度與抽樣中繼資料；沒有時檔案格式與先前相同。
        metadata["settling_sampling"] = table.settling_sampling
        label_arrays["particle_settling_velocity_mps"] = table.particle_settling_velocity_mps
        label_arrays["settling_candidate_index"] = table.settling_candidate_index
    np.savez(
        str(path),
        particle_id=table.particle_id,
        scenario_id=table.scenario_id,
        member_id=table.member_id,
        study_site_id=table.study_site_id,
        analysis_region_id=table.analysis_region_id,
        receptor_id=table.receptor_id,
        material_id=table.material_id,
        arrival_time_id=table.arrival_time_id,
        settling_velocity_mps=table.settling_velocity_mps,
        arrival_time_utc_ns=table.arrival_time_utc_ns,
        seed_hex=table.seed_hex,
        x0_m=table.x0_m,
        y0_m=table.y0_m,
        z0_m=table.z0_m,
        __metadata__=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        **label_arrays,
    )


def load_particle_identity_table_npz(path: str | Path) -> ParticleIdentityTable:
    """讀回 ``save_particle_identity_table_npz`` 保存的身分表，並核對內容雜湊。

    重建時會依原始欄位重新跑一次 ``ParticleIdentityTable.__post_init__``
    （包含唯一性、有限性與 dtype 檢查），並把重新計算出的 ``content_sha256``
    與保存當時記錄的值比對；兩者不一致代表檔案在保存與讀回之間被竄改、被
    不同版本的建構邏輯覆寫，或是磁碟／傳輸過程損壞，一律 fail-closed 直接
    拒絕，不得假設「形狀對得上就還原成功」。
    """

    with np.load(str(path), allow_pickle=False) as npz:
        metadata = json.loads(str(npz["__metadata__"]))
        table = ParticleIdentityTable(
            particle_id=npz["particle_id"],
            scenario_id=npz["scenario_id"],
            member_id=npz["member_id"],
            study_site_id=npz["study_site_id"],
            analysis_region_id=npz["analysis_region_id"],
            receptor_id=npz["receptor_id"],
            material_id=npz["material_id"],
            arrival_time_id=npz["arrival_time_id"],
            settling_velocity_mps=npz["settling_velocity_mps"],
            arrival_time_utc_ns=npz["arrival_time_utc_ns"],
            seed_hex=npz["seed_hex"],
            x0_m=npz["x0_m"],
            y0_m=npz["y0_m"],
            z0_m=npz["z0_m"],
            experiment_case_id=metadata["experiment_case_id"],
            master_seed=int(metadata["master_seed"]),
            random_stream_id=metadata["random_stream_id"],
            labels=(
                {name: npz[f"label__{name}"] for name in metadata["label_names"]}
                if "label_names" in metadata
                else None
            ),
            particle_settling_velocity_mps=(
                npz["particle_settling_velocity_mps"] if "settling_sampling" in metadata else None
            ),
            settling_candidate_index=(
                npz["settling_candidate_index"] if "settling_sampling" in metadata else None
            ),
            settling_sampling=metadata.get("settling_sampling"),
        )
    if table.content_sha256 != metadata["content_sha256"]:
        raise ParticleInputError(
            "重建的 content_sha256 與保存時記錄的值不一致，"
            f"保存值={metadata['content_sha256']}, 重建值={table.content_sha256}；"
            "檔案可能已損壞或被不同版本的建構邏輯覆寫，拒絕使用"
        )
    return table


# ── BayTrace SoA 轉換：只填 WP-4 能安全確定的欄位 ──────────────────────────


@dataclass(frozen=True, slots=True)
class BayTraceParticleSoA:
    """可直接寫入 ``baytrace.particle_arrays.ParticleArrays`` 對應欄位的陣列組。

    只包含 WP-4 能在不接觸網格與流場的情況下安全確定的欄位：位置（``x``／
    ``y``／``z``／``z0``）、``elem``（``-1`` 表示尚未定位，語意與 BayTrace 原版
    ``find_initial_elements`` 對「還沒找到」的慣例一致）與 ``status``
    （``0``，BayTrace 原版對「尚未啟動追蹤」粒子的慣例值，讓 WP-5 在跑完
    ``find_initial_elements`` 並確認找到元素後才轉成 ``1``）。

    ``settling_velocity_mps`` 不是 ``ParticleArrays`` 的原生欄位；這裡刻意
    保留它，是因為呼叫端必須依這個值把粒子分組，並在呼叫
    ``batch_move_rk4`` 前把它加進**該組專用**的 ``w_field1``／``w_field2``
    複本（見模組 docstring「沉降速度與 BayTrace RK4 的相容性」），而不是寫進
    ``pa.w``；把它保留在 SoA 輸出中，可以避免 WP-5 還要另外從身分表重新
    join 一次。

    ``row_particle_index`` 是 ``0..n-1`` 的列位置，供 WP-5 需要一個數值型
    識別碼時使用（例如 BayTrace 原版除錯用的 ``BAYTRACE_DEBUG_PIDS`` 機制
    實際上會對 ``pa.ids[i]`` 呼叫 ``int()``，無法直接放入本專案雜湊型
    ``particle_id`` 字串；若 WP-5 需要接上該除錯機制，應該用這個整數列位置
    當作 ``pa.ids``，再透過同一列位置回頭查身分表取得真正的
    ``particle_id``，不要把雜湊字串硬塞進 ``pa.ids``）。``start_time``（每個
    粒子在 BayTrace 全域時鐘上的釋放時刻）、``u``／``v``／``w`` 初始值等其餘
    ``ParticleArrays`` 欄位，都需要依賴 WP-5 選定的模型時間原點與網格，不在
    本類別範圍內，見模組尾端「分批策略」一節的討論。
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    z0: np.ndarray
    elem: np.ndarray
    status: np.ndarray
    settling_velocity_mps: np.ndarray
    row_particle_index: np.ndarray

    def __post_init__(self) -> None:
        """驗證 shape 一致、dtype 與 BayTrace kernel 期待一致、全部 C-contiguous。

        dtype 依據 ``baytrace/particle_arrays.py``：``x``／``y``／``z``／``z0``
        屬於 ``_FLOAT_FIELDS``（``float64``），``element``／``status`` 屬於
        ``_INT64_FIELDS``（``int64``，因為 ``batch_interp.py`` 的 njit kernel
        直接以 int64 陣列呼叫，見 ``tracker.py`` 的 ``elem_i64 = pa.element``／
        ``status_i64 = pa.status``）。
        """

        n = self.x.shape[0]
        float_fields = ("x", "y", "z", "z0", "settling_velocity_mps")
        int_fields = ("elem", "status", "row_particle_index")
        for name in (*float_fields, *int_fields):
            array = getattr(self, name)
            if not isinstance(array, np.ndarray) or array.ndim != 1 or array.shape[0] != n:
                raise ParticleInputError(f"{name} 必須是長度 {n} 的一維陣列")
            if not array.flags["C_CONTIGUOUS"]:
                raise ParticleInputError(f"{name} 必須是 C-contiguous，供 njit kernel 安全讀取")
        for name in float_fields:
            if getattr(self, name).dtype != np.float64:
                raise ParticleInputError(f"{name} dtype 必須是 float64")
        for name in int_fields:
            if getattr(self, name).dtype != np.int64:
                raise ParticleInputError(f"{name} dtype 必須是 int64")
        if np.any(self.elem != -1):
            raise ParticleInputError("elem 建構時必須全部是 -1（待定位），不得預先假設任何元素")
        if np.any(self.status != 0):
            raise ParticleInputError("status 建構時必須全部是 0（尚未啟動），啟動時機由 WP-5 決定")

    def __len__(self) -> int:
        """回傳粒子總數。"""

        return int(self.x.shape[0])


def build_baytrace_particle_soa(table: ParticleIdentityTable) -> BayTraceParticleSoA:
    """把不可變身分表轉成 BayTrace 相容的 struct-of-arrays。

    位置直接複製 ``x0_m``／``y0_m``／``z0_m``（正式初始位置，經
    ``InitialPositionResolver`` 解析），``elem`` 一律 ``-1``、``status`` 一律
    ``0``；兩者的啟動時機（跑 ``find_initial_elements`` 找到元素後轉
    ``status=1``，或依到達時刻延後啟動）屬於 WP-5 driver 的職責，本函式不
    代為決定。回傳陣列與 ``table`` 列序完全一致，因此 WP-6 後處理只要保留同一
    列序（或連同 ``row_particle_index`` 一起攜帶），就能把 BayTrace 輸出的
    每一列軌跡對回身分表的對應列，不需要額外的 join 鍵。
    """

    n = len(table)
    return BayTraceParticleSoA(
        x=np.ascontiguousarray(table.x0_m, dtype=np.float64),
        y=np.ascontiguousarray(table.y0_m, dtype=np.float64),
        z=np.ascontiguousarray(table.z0_m, dtype=np.float64),
        z0=np.ascontiguousarray(table.z0_m, dtype=np.float64),
        elem=np.full(n, -1, dtype=np.int64),
        status=np.zeros(n, dtype=np.int64),
        settling_velocity_mps=np.ascontiguousarray(table.settling_velocity_mps, dtype=np.float64),
        row_particle_index=np.arange(n, dtype=np.int64),
    )


# ── 分批：依到達時刻與（正確性必要的）沉降速度分組 ────────────────────────


@dataclass(frozen=True, slots=True)
class BatchGroup:
    """一組共用某個分批鍵值的粒子列位置。

    ``key`` 是分批依據欄位與其值的具名 tuple（例如
    ``(("settling_velocity_mps", -0.001),)``），保留具名而非裸值，是為了讓
    ``validate_batch_partition`` 與呼叫端的除錯訊息不需要另外猜測欄位對應。
    ``indices`` 是遞增排序的 ``int64`` 陣列，內容是該組粒子在
    ``ParticleIdentityTable``／``BayTraceParticleSoA`` 中的列位置，可直接用於
    ``table.x0_m[group.indices]`` 或 ``pa.x[group.indices]`` 這類 fancy
    indexing，也可以配合 ``ParticleArrays.slice_view`` 的用法先排序整個陣列
    再切連續區間（若 WP-5 決定實體重排陣列以避免 fancy indexing 的額外複製，
    ``indices`` 也可以直接當作那個重排的 permutation）。
    """

    key: tuple[tuple[str, object], ...]
    indices: np.ndarray

    def __post_init__(self) -> None:
        """驗證 ``indices`` 是非空、遞增排序、無重複的 int64 一維陣列。"""

        if not isinstance(self.indices, np.ndarray) or self.indices.ndim != 1:
            raise ParticleInputError("BatchGroup.indices 必須是一維陣列")
        if self.indices.dtype != np.int64:
            object.__setattr__(self, "indices", self.indices.astype(np.int64))
        if self.indices.size == 0:
            raise ParticleInputError("BatchGroup 不可為空批次")
        if np.any(np.diff(self.indices) <= 0):
            raise ParticleInputError("BatchGroup.indices 必須是嚴格遞增排序、無重複的列位置")


def _group_by_columns(table: ParticleIdentityTable, columns: tuple[str, ...]) -> tuple[BatchGroup, ...]:
    """依一或多個欄位的相同值分組，回傳依鍵值排序的 ``BatchGroup`` 序列。

    使用 ``np.lexsort`` 一次排序整張表，而不是對每個唯一值各掃描一次陣列
    （那是 ``O(唯一值數 × N)``）；lexsort 排序後同鍵值的列必然相鄰，只需一次
    ``O(N log N)`` 排序＋一次線性掃描即可切出所有分組，在 50 萬列規模下這個
    差異很明顯。
    """

    n = len(table)
    values = [getattr(table, name) for name in columns]
    order = np.lexsort(tuple(reversed(values)))
    sorted_values = [value[order] for value in values]
    # 一列是新分組的起點，只要「任一」分組欄位與前一列不同即成立（多欄位分組
    # 是邏輯 OR，不是 AND）；欄位鍵值本身在同一分組內每欄仍然完全相同，因為
    # lexsort 已把所有欄位都排序過，任何一欄改變就代表進入新的鍵值組合。
    boundaries = np.zeros(n, dtype=bool)
    boundaries[0] = True
    for value in sorted_values:
        boundaries[1:] |= value[1:] != value[:-1]
    boundary_positions = np.flatnonzero(boundaries)
    group_starts = boundary_positions
    group_stops = np.append(boundary_positions[1:], n)

    groups: list[BatchGroup] = []
    for start, stop in zip(group_starts, group_stops, strict=True):
        key = tuple(
            (name, _scalar_for_key(sorted_values[column_index][start]))
            for column_index, name in enumerate(columns)
        )
        indices = np.sort(order[start:stop]).astype(np.int64)
        groups.append(BatchGroup(key=key, indices=indices))
    groups.sort(key=lambda group: group.key)
    return tuple(groups)


def _scalar_for_key(value: np.generic) -> object:
    """把 numpy 純量轉成適合放進分批鍵值與錯誤訊息的原生 Python 型別。"""

    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return str(value)


def group_indices_by_settling_velocity(table: ParticleIdentityTable) -> tuple[BatchGroup, ...]:
    """依沉降速度分組；正確套用沉降速度到 RK4 全部四階的**必要**分組維度。

    十種基線代理各自對應一個嚴格互異的 ``settling_velocity_mps``（見
    ``scenarios.BASELINE_BEHAVIORS``），因此正式全案最多分成 10 組。這個分組
    不是效能上的選擇，而是模組 docstring 已論證的物理正確性要求：同一次
    ``batch_move_rk4`` 呼叫只能對應一組共用的 ``w_field1``／``w_field2``，
    混合不同沉降速度的粒子會讓其中至少一部分粒子在四個 RK4 階段中，有三個
    階段用到錯誤（別組）的沉降速度貢獻。
    """

    return _group_by_columns(table, ("settling_velocity_mps",))


def group_indices_by_arrival_time(table: ParticleIdentityTable) -> tuple[BatchGroup, ...]:
    """依到達時刻（UTC 奈秒）分組；這是最保守、與 BayTrace 全域同步時鐘直接
    相容的基準分批方案（見模組尾端「分批策略」一節）。

    正式設計每站 50 個到達時刻，五站合計最多 250 組（若不同站點剛好選到
    完全相同的到達 UTC，實際組數會更少，因為到達時刻本身已跨站點合併分組，
    而不是每站各自獨立計數）。這個方案的優點是「同批粒子的 BayTrace 全域時鐘
    從一開始就完全同步、不需要任何粒子啟動延遲機制」，缺點是分批數量遠多於
    ``group_indices_by_settling_velocity``、每批粒子數遠少，NumPy／njit 批次
    呼叫的向量化效益較差；詳見模組尾端分析。
    """

    return _group_by_columns(table, ("arrival_time_utc_ns",))


def group_indices_by_settling_velocity_and_arrival_time(
    table: ParticleIdentityTable,
) -> tuple[BatchGroup, ...]:
    """同時依沉降速度與到達時刻分組，是兩個限制條件的最小公倍分割。

    這是「兩個限制條件同時成立」時唯一保證正確又不需要額外機制的分批方式：
    同批粒子既共用沉降速度（RK4 正確性必要），也共用到達時刻（迴避
    ``ParticleArrays.start_time`` 分階段啟動機制是否可用的不確定性，見模組
    尾端「分批策略」一節）。組數是前兩者的乘積上限（最多 10 × 250 =
    2,500 組），每組粒子數 = 該到達時刻的受體數 × 每情境成員數
    （``20 × M``），規模測試與報告見測試檔與回報。
    """

    return _group_by_columns(table, ("settling_velocity_mps", "arrival_time_utc_ns"))


def validate_batch_partition(table: ParticleIdentityTable, groups: Sequence[BatchGroup]) -> None:
    """確認一組分批彼此不重疊、聯集等於全體粒子，無遺漏也無重複。

    這是分批策略的通用防呆，不論分批依據是哪個或哪些欄位都適用；呼叫端在
    正式把身分表交給 WP-5 driver 之前應該先跑這個檢查，確保「分批」本身沒有
    因為排序、去重或欄位型別轉換而漏掉或重複計入任何一顆粒子。
    """

    n = len(table)
    covered = np.zeros(n, dtype=bool)
    for group in groups:
        if np.any(covered[group.indices]):
            raise ParticleInputError("分批之間有重疊的粒子列位置")
        covered[group.indices] = True
    if not np.all(covered):
        missing = int(np.count_nonzero(~covered))
        raise ParticleInputError(f"分批未涵蓋全部粒子，遺漏 {missing} 列")


def validate_arrival_time_homogeneous_batches(
    table: ParticleIdentityTable, groups: Sequence[BatchGroup]
) -> None:
    """確認每一批粒子的 ``arrival_time_utc_ns`` 完全相同。

    供任何聲稱「已依到達時刻分批」的分組結果做事後檢查，包括
    ``group_indices_by_arrival_time`` 與
    ``group_indices_by_settling_velocity_and_arrival_time`` 的輸出，也適用於
    WP-5 若自行合併或再切分批次後的結果。
    """

    for group in groups:
        values = table.arrival_time_utc_ns[group.indices]
        if np.any(values != values[0]):
            raise ParticleInputError(f"批次 {group.key} 內含不同到達時刻，違反同批同步時鐘假設")


def validate_settling_velocity_homogeneous_batches(
    table: ParticleIdentityTable, groups: Sequence[BatchGroup]
) -> None:
    """確認每一批粒子的 ``settling_velocity_mps`` 完全相同。

    供任何要把沉降速度疊加進共用 ``w_field`` 的呼叫端，在真正呼叫
    ``batch_move_rk4`` 之前做最後一道防呆；一旦這個檢查通過，該批唯一的
    沉降速度值就可以安全地當成要加進 ``w_field1``／``w_field2`` 的常數。
    """

    for group in groups:
        values = table.settling_velocity_mps[group.indices]
        if np.any(values != values[0]):
            raise ParticleInputError(f"批次 {group.key} 內含不同沉降速度，違反 RK4 疊加前提")


@dataclass(frozen=True, slots=True)
class BatchLayoutReport:
    """一種分批方案的粒子數分佈摘要，供效能與平行化規劃參考。"""

    scheme_id: str
    batch_count: int
    particle_count_total: int
    min_particles_per_batch: int
    max_particles_per_batch: int
    mean_particles_per_batch: float


def summarize_batch_layout(scheme_id: str, groups: Sequence[BatchGroup]) -> BatchLayoutReport:
    """統計一組分批結果的批次數與每批粒子數分佈，不重新掃描身分表。"""

    if not groups:
        raise ParticleInputError("groups 不可為空，無法統計分批分佈")
    counts = [int(group.indices.size) for group in groups]
    total = sum(counts)
    return BatchLayoutReport(
        scheme_id=scheme_id,
        batch_count=len(groups),
        particle_count_total=total,
        min_particles_per_batch=min(counts),
        max_particles_per_batch=max(counts),
        mean_particles_per_batch=total / len(counts),
    )
