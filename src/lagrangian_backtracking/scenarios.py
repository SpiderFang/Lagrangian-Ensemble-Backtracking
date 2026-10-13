"""建立十種非上浮海廢代理、五站受體與到達時刻的完整情境清單。

情境識別碼只由站點、材料行為、受體、到達時刻及設計版本決定；實驗案例與隨機系集成員
是情境之外的維度。這可避免「不考慮波浪表面漂移」或不同成員數被誤算為計畫書的基礎
情境，也讓批次切分、工作程序數量與中途續跑不會改變亂數序列。十種代理採海洋保育署
iOcean 類別作名稱銜接，但速度是待校準的敏感度格點，不是官方清除統計推得的物性。

2026-09-23 對 run01（H30、M=1）真實成果的診斷進一步發現：最快四檔（-0.1、-0.05、
-0.02、-0.01 m/s）彼此的水平來源足跡差異低於純抽樣雜訊，統計上不可分辨（完整數據見
``claude/diagnosis-20260923-十檔沉降速度可區辨性與M的下限.md`` 與
``docs/foundation/03_scientific_method_and_validation.md`` 第 11.1 節）。因此下游報告
與圖表必須以 ``settling_velocity_mps`` 數值或極慢／慢／中／快分群為主詞呈現，
``oca_category_zh`` 只能作附註，不得單獨作為「材質來自某方向」之類表述的主詞；本模組
仍保留十個 iOcean 類別名稱一對一綁定，是為了保持可稽核的情境識別碼與既有 manifest
位元相同，不代表這十個名稱在目前 M=1 的成果中彼此可分辨。2026-09-24／25 改版後的
新值仍全數落在該診斷所指的快沉降區間（十值皆 ≥ 0.01 m/s），因此上述「彼此統計上不可
分辨」的警語在改版後依然成立，不因換算新的力平衡推導值而自動失效。

2026-10-13（WP-54，v7）再把「每類一個速度」擴充為「每類 10 個候選速度」：推導來源
（``data/settling_velocity_derivation``）每類改為 5 種代表物件（由小到大）× 阻力係數上下界
＝ 10 個沉降速度解，這 10 個解寫進 ``Behavior.settling_velocity_candidates_mps``，正式計算時
每顆粒子在所屬類別的 10 個候選速度中平衡抽樣取其一（見
``native_runner/settling_sampling.py``）。``Behavior.settling_velocity_mps`` 因此降為
「類別代表值」（10 個解的幾何平均，取三位有效數字）：它仍是類別的分組鍵（情境識別碼、
工作單元識別碼、後處理與圖表的分組都以它代表「這是哪一類」），在
``representative`` 抽樣模式下也仍是粒子實際使用的速度（完全重現 v6 以前每類一個速度的
行為）；在 ``class_candidates_balanced`` 模式下，粒子實際使用的速度另存於粒子身分表的
``particle_settling_velocity_mps`` 欄位，不再等於類別代表值。注意上一段「十值皆 ≥ 0.01 m/s」
只適用於 2026-09-24／25 的代表值；v7 的候選速度跨度更大（例如其他／不可回收最慢約
0.0009 m/s），其可分辨性必須以 v7 成果重新檢視，不能沿用上段結論。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

SETTLING_CANDIDATE_COUNT = 10
"""每個類別的候選沉降速度個數（5 種代表物件 × 阻力係數上下界）。"""

SETTLING_VELOCITY_SAMPLING_REPRESENTATIVE = "representative"
"""每個類別所有粒子都使用該類別的代表速度（v6 以前的行為，逐位重現）。"""

SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED = "class_candidates_balanced"
"""同一站、同一類別的粒子在該類別的 10 個候選速度中平衡抽樣，各候選分得相同顆數。"""

SETTLING_VELOCITY_SAMPLING_MODES: tuple[str, ...] = (
    SETTLING_VELOCITY_SAMPLING_REPRESENTATIVE,
    SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED,
)


def validate_settling_velocity_sampling_mode(mode: object) -> str:
    """確認沉降速度抽樣模式是已登錄的字串，否則丟出 ``ValueError``（不做任何自動修正）。"""

    if type(mode) is not str or mode not in SETTLING_VELOCITY_SAMPLING_MODES:
        raise ValueError(
            f"沉降速度抽樣模式 {mode!r} 不合法，只接受 {SETTLING_VELOCITY_SAMPLING_MODES}"
        )
    return mode


@dataclass(frozen=True, slots=True)
class Behavior:
    """一種具名海廢材質／形狀條件及其沉降速度。

    ``oca_category_zh`` 只保存 iOcean 清除統計使用的分類名稱；官方頁面沒有單體密度、
    尺寸、形狀因子或終端速度，因此 ``settling_velocity_mps`` 仍是研究設計的負值敏感度
    格點。``applicability_condition_zh`` 用來防止把保麗龍、木材或密閉瓶罐等原本可能浮水
    的物件無條件納入：只有已吸水、進水、生物附著或其他原因使其完全沉沒且向下運動時，
    才符合本基線。輸出 material manifest 時會完整保留這些限制與證據等級。

    2026-09-24／25 改版（見
    ``claude/decision-20260924-十檔沉降速度定案與重算交辦.md``）新增三個欄位，記錄
    ``settling_velocity_mps``（v7 起為該類 10 個解的幾何平均代表值）在推導來源中實際涵蓋的
    速度範圍：

    * ``settling_velocity_slow_bound_mps``：該類推導區間中絕對值最小者（沉得最慢），
      仍是負值。
    * ``settling_velocity_fast_bound_mps``：該類推導區間中絕對值最大者（沉得最快），
      仍是負值。
    * ``derivation_source``：可稽核推導來源的相對路徑（固定為
      ``data/settling_velocity_derivation/results.csv``）。

    這三個欄位刻意不用 ``min``／``max`` 命名，因為速度是負值：絕對值最大（沉得最快）
    在數值上反而最小。若照數值大小稱呼 ``min``／``max``，任何依「min <= max」直覺寫的
    檢查都會把 ``fast_bound``（數值較小）誤判成下界、``slow_bound``（數值較大）誤判成
    上界，方向恰好相反。正確的數值關係固定是
    ``fast_bound <= settling_velocity_mps <= slow_bound``（三者皆為負值）。

    2026-10-13（WP-54，v7）新增 ``settling_velocity_candidates_mps``：該類 5 種代表物件 ×
    阻力係數上下界的 10 個沉降速度解（單位 m/s、z 軸向上為正，因此皆為嚴格負值），依
    **絕對值由小到大**（最慢到最快）排列，位置序號 0–9 即「候選序號」。``slow_bound`` 恆為
    序號 0 的值、``fast_bound`` 恆為序號 9 的值；``settling_velocity_mps`` 為這 10 個解絕對值的
    幾何平均、取三位有效數字（只作類別代表值與 ``representative`` 模式使用）。三個類別
    （寶特瓶、鋁罐、鐵罐）的「壓扁」與「嚴重壓扁」兩種物件，殼重與外包絡體積使有效密度與
    厚度的乘積相同，因此兩者的 ``w_lo``／``w_hi`` 逐位相同，候選值中各有 2 組數值重複
    （物理必然，不是錯誤）；候選序號仍各自獨立，平衡抽樣以序號分配顆數。
    預設空 tuple 僅為讀取 v6 以前 manifest 的相容性而保留（此時只能使用 ``representative``
    模式），``BASELINE_BEHAVIORS`` 一律有 10 個候選值。
    """

    material_id: str
    oca_category_zh: str
    material_family_zh: str
    representative_shape_zh: str
    settling_velocity_mps: float
    behavior_class: str
    applicability_condition_zh: str
    calibration_status: str
    evidence_grade: str
    settling_velocity_slow_bound_mps: float
    settling_velocity_fast_bound_mps: float
    derivation_source: str
    settling_velocity_candidates_mps: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Receptor:
    """指定站點的一個三維終端受體。

    ``z_m_positive_up`` 是水平位置與 ``vertical_id`` 的模板代表／候選深度，單位為公尺、
    海面為零且向上為正；它不能宣稱是所有 arrival UTC 的正式實際初始深度。正式 runtime
    必須查詢 receptor×arrival 的 ``ReceptorArrivalInitialCondition``，使用該筆由 OCM
    eta、zcor 與 wet/dry 推導的 actual z。經緯度只作 WGS84 資料交換與受體定位，不直接
    進入公尺制物理運算；目標水深比例、距海床高度、深度調整原因與 OCM 網格來源可放入
    ``metadata`` 或 pair manifest 的來源欄位。
    """

    receptor_id: str
    study_site_id: str
    analysis_region_id: str
    lon: float
    lat: float
    z_m_positive_up: float
    vertical_id: str
    metadata: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ArrivalTime:
    """站點的到達世界協調時間（UTC）與季節、潮況、事件分層資料。"""

    arrival_time_id: str
    study_site_id: str
    time_utc_ns: int
    year: int
    season: str
    tide_class: str
    phase_or_event: str
    metadata: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ReceptorArrivalInitialCondition:
    """一個 receptor×arrival pair 在到達時刻的正式三維初始條件。

    ``z_m_positive_up`` 是 OCM 在該 arrival UTC 由海面高程（eta）、垂向網格（zcor）與
    wet/dry 狀態推導出的實際初始深度，單位為公尺，海面向上為正；它不是 ``Receptor``
    模板上的代表深度。``zcor_lower_m_positive_up``、``zcor_upper_m_positive_up`` 是
    OCM 垂向 bracket 的兩個端點，``vertical_bracket_alpha`` 是無因次線性內插權重。
    經緯度不在此資料類別中，也不參與物理計算；source face、月份與來源時間索引則保留
    可重現的 OCM 定位證據。``wetdry_elem_value=0`` 才表示可供正式／pilot 執行的濕元素，
    乾元素不以零值或替代深度偷偷通過。
    """

    receptor_id: str
    arrival_time_id: str
    study_site_id: str
    analysis_region_id: str
    flow_domain_id: str
    time_utc_ns: int
    vertical_id: str
    z_m_positive_up: float
    eta_m_positive_up: float
    bed_z_m_positive_up: float
    water_column_height_m: float
    height_above_bed_m: float
    zcor_lower_m_positive_up: float
    zcor_upper_m_positive_up: float
    vertical_bracket_alpha: float
    source_face_local_index: int
    source_face_global_index: int
    wetdry_elem_value: int
    wetdry_semantics_id: str
    ocm_month_yyyymm: str
    ocm_source_time_index: int
    ocm_time_origin: str


@dataclass(frozen=True, slots=True)
class Scenario:
    """一組固定行為、受體與到達時間的基礎情境。"""

    scenario_id: str
    study_site_id: str
    analysis_region_id: str
    material_id: str
    receptor_id: str
    arrival_time_id: str
    settling_velocity_mps: float
    arrival_time_utc_ns: int
    design_version: str


BASELINE_BEHAVIORS: tuple[Behavior, ...] = (
    Behavior(
        "oca_styrofoam_porous_fragment",
        "保麗龍",
        "發泡聚苯乙烯（EPS）",
        "吸水或生物附著之多孔不規則碎塊",
        -0.0482,
        "sinking",
        "限已吸水或生物附著、整體仍完全沉沒且呈負浮力者",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.005274,
        -0.187049,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.005274, -0.006292, -0.030493, -0.038718, -0.053909,
            -0.069663, -0.114480, -0.141766, -0.150064, -0.187049,
        ),
    ),
    Behavior(
        "oca_wood_waterlogged_elongated",
        "竹木",
        "竹材或木材",
        "水浸飽和之細長枝條或片狀木屑",
        -0.0885,
        "sinking",
        "限水浸後整體密度已高於周圍海水者",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.013503,
        -0.303349,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.013503, -0.015205, -0.056465, -0.065310, -0.097057,
            -0.113218, -0.196277, -0.231153, -0.256543, -0.303349,
        ),
    ),
    Behavior(
        "oca_wastepaper_folded_fiber",
        "廢紙",
        "吸水纖維材料",
        "濕潤摺疊紙片或纖維團",
        -0.0406,
        "sinking",
        "固定形狀與速度，不模擬持續解體、溶散或團聚",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.006610,
        -0.162364,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.006610, -0.007942, -0.020341, -0.025649, -0.048532,
            -0.062636, -0.074154, -0.097400, -0.123786, -0.162364,
        ),
    ),
    Behavior(
        "oca_nonrecyclable_flexible_sheet",
        "其他/不可回收",
        "混合塑膠或複合包材",
        "進水薄膜、軟片或皺摺包裝片",
        -0.0114,
        "sinking",
        "只作異質類別的條件代理，不解讀為類別平均物性",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.000897,
        -0.107722,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.000897, -0.001064, -0.001405, -0.001724, -0.027185,
            -0.035293, -0.038879, -0.050796, -0.081720, -0.107722,
        ),
    ),
    Behavior(
        "oca_fishinggear_open_mesh_bundle",
        "廢漁網漁具",
        "尼龍、聚乙烯或混合漁具材料",
        "網片、繩索或繩結纖維束",
        -0.0504,
        "sinking",
        "不顯式解析網目開孔、纏繞、展開與姿態變化",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.007950,
        -0.223412,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.007950, -0.008676, -0.029639, -0.033541, -0.044512,
            -0.051101, -0.116965, -0.136675, -0.189684, -0.223412,
        ),
    ),
    Behavior(
        "oca_pet_waterfilled_bottle",
        "寶特瓶",
        "聚對苯二甲酸乙二酯（PET）",
        "進水或壓扁之中空瓶體",
        -0.0712,
        "sinking",
        "只納入已進水且向下運動者；密閉含氣瓶體必須排除",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.029300,
        -0.133527,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.029300, -0.037568, -0.056303, -0.056303, -0.073531,
            -0.073531, -0.099252, -0.099889, -0.133143, -0.133527,
        ),
    ),
    Behavior(
        "oca_other_recyclable_irregular_fragment",
        "其他/可回收",
        "混合可回收塑膠、橡膠或複合材料",
        "不規則片塊或零組件",
        -0.0959,
        "sinking",
        "只作異質類別代理；正式校準前應再依材質與形狀分群",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.034223,
        -0.344737,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.034223, -0.043001, -0.043534, -0.055415, -0.088343,
            -0.116511, -0.123786, -0.162364, -0.260350, -0.344737,
        ),
    ),
    Behavior(
        "oca_aluminum_crushed_cylinder",
        "鋁罐",
        "鋁合金",
        "進水壓扁之薄壁中空圓筒",
        -0.108,
        "sinking",
        "只納入已進水或壓扁者；密閉含氣罐體必須排除",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.036715,
        -0.175479,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.036715, -0.047057, -0.107871, -0.107871, -0.130788,
            -0.130897, -0.141024, -0.141024, -0.175279, -0.175479,
        ),
    ),
    Behavior(
        "oca_steel_rigid_cylinder",
        "鐵罐",
        "鋼鐵或鍍錫鋼板",
        "進水之剛性圓筒或金屬片",
        -0.257,
        "sinking",
        "姿態、開口、塗層與腐蝕效應尚未顯式解析",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.107413,
        -0.427352,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.107413, -0.139479, -0.237796, -0.237796, -0.288447,
            -0.313089, -0.313089, -0.315346, -0.389421, -0.427352,
        ),
    ),
    Behavior(
        "oca_glass_bottle_or_fragment",
        "玻璃瓶",
        "玻璃",
        "進水瓶體或緻密銳角碎片",
        -0.48,
        "sinking",
        "完整瓶與碎片幾何差異留待樣本量測校準",
        "derived_uncalibrated",
        "B_force_balance_derived_5_objects_x_drag_bounds_10_candidates",
        -0.195421,
        -0.920397,
        "data/settling_velocity_derivation/results.csv",
        (
            -0.195421, -0.255255, -0.355430, -0.466427, -0.509251,
            -0.517654, -0.672576, -0.691764, -0.693677, -0.920397,
        ),
    ),
)


def validate_non_rising_behaviors(behaviors: Sequence[Behavior]) -> None:
    """拒絕零速、上浮或缺少材質／形狀追溯資訊的基線代理。

    本檢查屬於情境建表前的科學閘門。座標採海面向上為正，因此只有嚴格負值可表示
    物理時間向前的沉降；零值會重新引入 PI 已取消的中性懸浮情境，正值則是上浮情境。
    iOcean 類別必須一對一且不可空白，避免十個速度雖然不同、實際卻沒有對應到十種可
    稽核海廢條件。函式只驗證設計一致性，不主張暫定速度已經完成現地校準。
    """

    if not behaviors:
        raise ValueError("material behaviors 不可為空")
    material_ids = [item.material_id for item in behaviors]
    categories = [item.oca_category_zh for item in behaviors]
    if len(set(material_ids)) != len(material_ids):
        raise ValueError("material_id 必須唯一")
    if len(set(categories)) != len(categories):
        raise ValueError("iOcean 類別必須一對一且不可重複")
    for item in behaviors:
        required_text = (
            item.material_id,
            item.oca_category_zh,
            item.material_family_zh,
            item.representative_shape_zh,
            item.applicability_condition_zh,
            item.calibration_status,
            item.evidence_grade,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError(f"{item.material_id or '<missing>'} 缺少材質、形狀或證據欄位")
        if item.behavior_class != "sinking" or item.settling_velocity_mps >= 0.0:
            raise ValueError(f"{item.material_id} 必須是 sinking 且 settling_velocity_mps 嚴格小於 0")
        _validate_candidate_velocities(item)


def _validate_candidate_velocities(item: Behavior) -> None:
    """檢查候選速度：空 tuple（舊 manifest）略過；否則必須 10 個嚴格負值、由慢到快、與界限一致。

    只驗證設計一致性，不重算力平衡（重算由 ``tests/test_scenarios.py`` 對照
    ``data/settling_velocity_derivation/results.csv`` 完成）。慢界必須等於序號 0、快界必須
    等於序號 9，代表值必須落在兩界之間，避免候選值與界限各自漂移。
    """

    candidates = item.settling_velocity_candidates_mps
    if len(candidates) == 0:
        return
    if not isinstance(candidates, tuple):
        raise ValueError(f"{item.material_id} 的 settling_velocity_candidates_mps 必須是 tuple")
    if len(candidates) != SETTLING_CANDIDATE_COUNT:
        raise ValueError(
            f"{item.material_id} 的 settling_velocity_candidates_mps 必須恰有 "
            f"{SETTLING_CANDIDATE_COUNT} 個，實際 {len(candidates)} 個"
        )
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, float) or value != value or abs(value) == float("inf"):
            raise ValueError(f"{item.material_id} 的候選沉降速度必須是有限浮點數")
        if value >= 0.0:
            raise ValueError(f"{item.material_id} 的候選沉降速度必須嚴格小於 0")
    magnitudes = [abs(value) for value in candidates]
    if magnitudes != sorted(magnitudes):
        raise ValueError(f"{item.material_id} 的候選沉降速度必須依絕對值由小到大排列")
    if item.settling_velocity_slow_bound_mps != candidates[0] or item.settling_velocity_fast_bound_mps != candidates[-1]:
        raise ValueError(f"{item.material_id} 的慢／快界必須等於候選序號 0 與 9 的值")
    if not (candidates[-1] <= item.settling_velocity_mps <= candidates[0]):
        raise ValueError(f"{item.material_id} 的代表速度必須落在候選速度的慢界與快界之間")


def stable_identifier(namespace: str, fields: Sequence[str], *, length: int = 24) -> str:
    """用欄位長度與文字內容建立穩定識別碼，避免不同欄位組合混淆。

    例如 ``["ab", "c"]`` 與 ``["a", "bc"]`` 若直接接成一串文字會相同；先記錄每段
    長度即可保留欄位界線。回傳值是在指定類別前綴後加上 SHA-256 雜湊的前段文字；正式
    情境清單仍須保存所有原始欄位，不能只保存識別碼。
    """

    if not namespace or length < 16 or length > 64:
        raise ValueError("namespace 不可空白，hash length 必須介於 16 與 64")
    encoded = b"".join(
        len(value.encode("utf-8")).to_bytes(4, "big") + value.encode("utf-8") for value in fields
    )
    return f"{namespace}_{sha256(encoded).hexdigest()[:length]}"


def build_scenarios(
    *,
    behaviors: Sequence[Behavior],
    receptors: Sequence[Receptor],
    arrival_times: Sequence[ArrivalTime],
    design_version: str,
) -> list[Scenario]:
    """依站點建立材料行為、受體與到達時刻的完整交叉組合。

    每個受體與到達時刻都先依站點分組；函式拒絕遺漏站點或重複識別碼，也不會把 A 區的
    貢寮與龜山島合併後才交叉。正式五站清單應再用 ``validate_baseline_coverage`` 確認
    每站恰為 10,000 個、全案恰為 50,000 個基礎情境。
    """

    if not design_version:
        raise ValueError("design_version 不可空白")
    validate_non_rising_behaviors(behaviors)
    if len({item.receptor_id for item in receptors}) != len(receptors):
        raise ValueError("receptor_id 必須全案唯一")
    if len({item.arrival_time_id for item in arrival_times}) != len(arrival_times):
        raise ValueError("arrival_time_id 必須全案唯一")
    receptors_by_site: dict[str, list[Receptor]] = {}
    arrivals_by_site: dict[str, list[ArrivalTime]] = {}
    for receptor in receptors:
        receptors_by_site.setdefault(receptor.study_site_id, []).append(receptor)
    for arrival in arrival_times:
        arrivals_by_site.setdefault(arrival.study_site_id, []).append(arrival)
    if set(receptors_by_site) != set(arrivals_by_site):
        raise ValueError("receptor 與 arrival 的 study_site_id 集合不一致")

    result: list[Scenario] = []
    for site_id in sorted(receptors_by_site):
        for behavior in sorted(behaviors, key=lambda item: item.material_id):
            for receptor in sorted(receptors_by_site[site_id], key=lambda item: item.receptor_id):
                for arrival in sorted(arrivals_by_site[site_id], key=lambda item: item.arrival_time_id):
                    fields = [
                        site_id,
                        behavior.material_id,
                        receptor.receptor_id,
                        arrival.arrival_time_id,
                        design_version,
                    ]
                    result.append(
                        Scenario(
                            scenario_id=stable_identifier("scn", fields),
                            study_site_id=site_id,
                            analysis_region_id=receptor.analysis_region_id,
                            material_id=behavior.material_id,
                            receptor_id=receptor.receptor_id,
                            arrival_time_id=arrival.arrival_time_id,
                            settling_velocity_mps=behavior.settling_velocity_mps,
                            arrival_time_utc_ns=arrival.time_utc_ns,
                            design_version=design_version,
                        )
                    )
    if len({item.scenario_id for item in result}) != len(result):
        raise RuntimeError("scenario hash 發生碰撞")
    return result


def validate_baseline_coverage(scenarios: Sequence[Scenario]) -> dict[str, int]:
    """確認五站各 10,000 個、A 區 20,000 個、全案 50,000 個基礎情境。

    現行 ``CURRENT_DESIGN_VERSION`` 的 C 區正式站點是 ``nanwan``；若整批情境仍
    使用已登錄 legacy design，則暫時接受歷史 ``houwan`` 集合，供唯讀結果與相容
    工具驗證。這個分流讓歷史 artifact 不被改寫，同時使目前正式設計無法悄悄把
    houwan 混入南灣母體。
    """

    counts: dict[str, int] = {}
    for scenario in scenarios:
        counts[scenario.study_site_id] = counts.get(scenario.study_site_id, 0) + 1
    # 這裡不能反向 import config，因為 config 會載入 bed-residence，而 bed-residence
    # 又依賴 Scenario 型別。以實際 C 區 site ID 判斷目前／歷史集合即可：正式 current
    # builder 一定含 nanwan；只有完全沒有 nanwan 的舊 artifact 才可保留 houwan。
    expected_sites = (
        {"gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"}
        if "nanwan" in counts
        else {"gongliao", "guishan", "hsinchu", "houwan", "lienchiang"}
    )
    if set(counts) != expected_sites or any(value != 10_000 for value in counts.values()):
        raise ValueError(f"baseline coverage 不符：{counts}")
    if counts["gongliao"] + counts["guishan"] != 20_000 or len(scenarios) != 50_000:
        raise ValueError("A 區或全案 scenario count 不符")
    return counts


def validate_random_stream_id(value: object, *, allow_none: bool = True) -> str | None:
    """驗證可選的共同亂數流識別碼，並保留呼叫端提供的原文字串。

    ``random_stream_id`` 是工程配對的亂數命名空間，不是物理實驗案例；例如
    ``no_stokes`` 與 ``finite_depth_stokes`` 仍保留各自的 ``experiment_case_id``，但可
    明示同一個 stream 讓兩組案例對相同的 scenario／member 使用同一個初始亂數狀態。這個
    欄位只接受非空白原生字串；不做自動 trim 或替換，避免在 plan、seed table 與
    checkpoint binding 中留下與 operator 看到的識別碼不同的值。省略時回傳 ``None``，
    讓既有 seed 導出規則完全維持以 experiment case 為命名空間的行為。

    Args:
        value: 呼叫端提供的 stream ID，或代表未啟用配對的 ``None``。
        allow_none: 是否允許 ``None``；公開 seed API 使用預設值，schema 驗證可要求
            欄位必須存在時關閉此選項。

    Returns:
        原樣保留的非空字串，或在允許省略且輸入為 ``None`` 時回傳 ``None``。

    Raises:
        TypeError: 輸入不是原生字串或 ``None``。
        ValueError: 字串去除前後空白後沒有內容，或不允許省略卻傳入 ``None``。
    """

    if value is None:
        if allow_none:
            return None
        raise ValueError("random_stream_id 不可省略")
    if type(value) is not str:
        raise TypeError("random_stream_id 必須是字串或 None")
    if not value.strip():
        raise ValueError("random_stream_id 不可為空白")
    return value


def derive_member_seed(
    *,
    master_seed: int,
    scenario_id: str,
    experiment_case_id: str,
    member_id: int,
    random_stream_id: str | None = None,
) -> int:
    """產生不受工作程序、批次切分與中途續跑影響的 NumPy 128 位元亂數種子。

    不使用 Python 內建雜湊，因它在不同程序可能加入隨機值。未啟用共同亂數流時，主種子、
    情境、物理實驗案例與成員編號都輸入 SHA-256，完整保留舊版 seed 規則；若明示
    ``random_stream_id``，則只把這個配對識別碼替換「案例身分」輸入，讓不同物理案例在
    相同主種子、scenario 與 member 下得到同一 seed，而不改變 ``particle_id`` 或任何
    物理案例欄位。兩種模式都取 SHA-256 前 16 位元組作為非負整數，交給 NumPy
    PCG64DXSM 亂數產生器。
    """

    if master_seed < 0 or member_id < 0 or not scenario_id or not experiment_case_id:
        raise ValueError("seed 欄位必須非負且識別碼不可空白")
    stream_id = validate_random_stream_id(random_stream_id)
    case_identity = experiment_case_id if stream_id is None else stream_id
    fields = [str(master_seed), scenario_id, case_identity, str(member_id)]
    encoded = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(sha256(encoded).digest()[:16], "big", signed=False)


def records_as_dicts(records: Iterable[Behavior | Receptor | ArrivalTime | Scenario]) -> list[dict]:
    """將資料類別轉為可寫入 JSON 或 Parquet 的字典清單，不建立效能較差的物件陣列。"""

    return [asdict(item) for item in records]
