"""聚合 release 使用的不可變資料列與 shard 綁定記錄。

本模組只定義聚合 release 的資料契約，不負責讀取檔案、執行粒子追蹤或計算統計量。
``AggregateShardBinding`` 將一個 trajectory shard 的情境範圍、輸出相對路徑、SHA-256
及計數綁在一起，讓後續 release writer 能確認輸入沒有被悄悄替換。``ScenarioStratum``
則把每個基礎 scenario 所需的材料、受體、到達時間與（可選的）OCM-derived 動態初始
條件攤平成一列；這些欄位是分層統計的識別與分組資料，不是絕對來源機率。

所有長度、位置與速度欄位採公尺制或公尺／秒；經緯度僅作 WGS84 資料交換與圖面定位。
動態初始條件若缺值，必須整組缺值；不能用單一零值代替乾點、缺時或未知資料。正式
release 會要求完整動態條件，pilot 才可在整個 bundle 都沒有初始條件時保留空值。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from .manifests import ScenarioInputs

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# OCM 月份識別碼只允許四位數年份與 01 至 12 月，避免把不存在的月份當成資料來源。
_YYYYMM_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{4}(?:0[1-9]|1[0-2])$")


def _require_text(value: object, label: str) -> str:
    """驗證資料契約中的文字識別或描述欄位。

    release record 要可穩定序列化與分組，因此不接受非文字、空字串或首尾藏有空白的
    值。函式不會 trim 後默默改寫資料；輸入若含首尾空白會直接失敗，避免同一個
    scientific stratum 在不同輸入來源產生不同字串語意。
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_native_int(value: object, label: str) -> int:
    """驗證原生 Python ``int``，明確排除 ``bool`` 與 NumPy 整數代理。

    這些 index、年份與 UTC 奈秒值會跨 JSON、Parquet 與 checkpoint 邊界傳遞；在資料列
    邊界保留原生整數型別，可避免布林被 Python 的整數子類別規則誤收，也讓序列化契約
    不依賴 NumPy scalar 的隱式轉換。
    """

    if type(value) is not int:
        raise ValueError(f"{label} 必須是原生 int，且不可是 bool")
    return value


def _require_finite_float(value: object, label: str) -> float:
    """驗證可有限表示的原生 Python ``int`` 或 ``float``，並 canonical 成 ``float``。

    浮點欄位的輸入可來自已驗證 manifest 的原生整數或浮點數，但不可接受 ``bool``、
    NumPy scalar 或其他可轉型物件。轉換後一律保存為 Python ``float``；NaN、Infinity
    及無法以有限 Python ``float`` 表示的巨大整數都拒絕。這裡只處理型別與有限性，
    不把缺值、乾點或時間缺口改成零值。
    """

    if type(value) not in (int, float):
        raise ValueError(f"{label} 必須是有限的原生 int 或 float，且不可是 bool")
    try:
        canonical = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} 無法 canonical 成有限的 Python float") from exc
    if not math.isfinite(canonical):
        raise ValueError(f"{label} 必須是有限的原生 int 或 float")
    return canonical


def _require_optional_finite_float(value: object, label: str) -> float | None:
    """驗證可整組缺值的有限浮點欄位。"""

    if value is None:
        return None
    return _require_finite_float(value, label)


def _require_optional_native_int(value: object, label: str) -> int | None:
    """驗證可整組缺值的原生整數欄位。"""

    if value is None:
        return None
    return _require_native_int(value, label)


def _require_optional_text(value: object, label: str) -> str | None:
    """驗證可整組缺值的文字欄位，拒絕空白字串冒充缺值。"""

    if value is None:
        return None
    return _require_text(value, label)


@dataclass(frozen=True, slots=True)
class AggregateShardBinding:
    """一個 trajectory shard 的不可變輸入綁定與完整性摘要。

    ``scenario_start_index`` 與 ``scenario_stop_index`` 是半開區間
    ``[start, stop)``，對應 immutable run plan 中的基礎 scenario 順序，而不是粒子
    或檔案內 row 的任意位置。``output_relative_path`` 必須是相對於 run workspace 的
    POSIX 路徑；禁止絕對路徑、父層 ``..`` 與反斜線，避免 release manifest 跨出核准
    workspace；每一層名稱都必須非空且不可為 ``.`` 或 ``..``。``shard_id`` 最長 128
    字元，允許英數字、句點、底線與連字號，但第一字元必須是英數字。三個 count 是已驗證
    trajectory 輸出的 Python 原生整數，其中 observation 與 event 至少各包含每個 particle
    的基礎資料列。
    """

    shard_id: str
    scenario_start_index: int
    scenario_stop_index: int
    output_relative_path: str
    trajectory_manifest_sha256: str
    particle_count: int
    observation_count: int
    event_count: int

    def __post_init__(self) -> None:
        """在資料列建立時完成路徑、hash、索引與計數的 fail-fast 驗證。"""

        shard_id = _require_text(self.shard_id, "shard_id")
        if _SLUG_RE.fullmatch(shard_id) is None:
            raise ValueError(
                "shard_id 必須是長度不超過 128 的 slug，首字元為英數字，"
                "其餘只能含英數字、句點、底線或連字號"
            )
        start = _require_native_int(self.scenario_start_index, "scenario_start_index")
        stop = _require_native_int(self.scenario_stop_index, "scenario_stop_index")
        if start < 0 or stop <= start:
            raise ValueError("scenario index 必須滿足 0 <= start < stop")

        path = _require_text(self.output_relative_path, "output_relative_path")
        path_tokens = path.split("/")
        if any(token in {"", ".", ".."} for token in path_tokens):
            raise ValueError("output_relative_path 每一層都必須非空，且不可是 . 或 ..")
        if "\\" in path:
            raise ValueError("output_relative_path 不可含反斜線")
        pure_path = PurePosixPath(path)
        if pure_path.is_absolute():
            raise ValueError("output_relative_path 必須是安全相對路徑，不可為 absolute path")

        digest = _require_text(
            self.trajectory_manifest_sha256, "trajectory_manifest_sha256"
        )
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("trajectory_manifest_sha256 必須是 64 位小寫 SHA-256")

        particle_count = _require_native_int(self.particle_count, "particle_count")
        observation_count = _require_native_int(self.observation_count, "observation_count")
        event_count = _require_native_int(self.event_count, "event_count")
        if particle_count <= 0:
            raise ValueError("particle_count 必須大於 0")
        if observation_count < particle_count:
            raise ValueError("observation_count 不可小於 particle_count")
        if event_count < particle_count:
            raise ValueError("event_count 不可小於 particle_count")

        # frozen dataclass 只阻止重新賦值；這裡以驗證後的同值欄位明確固定契約，並保留
        # 未來若型別註記擴充時的單一檢查入口。所有輸入本來就要求原生型別，不做轉型。
        object.__setattr__(self, "shard_id", shard_id)
        object.__setattr__(self, "scenario_start_index", start)
        object.__setattr__(self, "scenario_stop_index", stop)
        object.__setattr__(self, "output_relative_path", path)
        object.__setattr__(self, "trajectory_manifest_sha256", digest)
        object.__setattr__(self, "particle_count", particle_count)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "event_count", event_count)


@dataclass(frozen=True, slots=True)
class ScenarioStratum:
    """一筆可重現的 scenario 分層資料列。

    前半部欄位來自已驗證的材料、受體、arrival manifest 與 immutable ``Scenario``；
    ``settling_velocity_mps`` 為條件式材料代理速度，不能單獨解讀為官方物性。經緯度
    是 WGS84 交換座標，``receptor_template_z_m_positive_up`` 是受體模板代表深度；
    實際平流仍在公尺制 flow-domain 座標執行。

    後半部欄位是 receptor×arrival 的 OCM-derived 動態初始條件。它們要嘛全部為
    ``None``（代表此 bundle 沒有可用的動態初始條件），要嘛全部有值；不允許只填一部
    分，也不把 0、乾點或未知值當作替代缺值。``has_dynamic_initial_condition`` 只表示
    這一列是否具備完整初始條件，不表示來源的絕對機率或因果歸因；正式統計應稱為
    條件式來源足跡或相對來源權重。
    """

    scenario_id: str
    study_site_id: str
    analysis_region_id: str
    material_id: str
    material_category_zh: str
    material_family_zh: str
    representative_shape_zh: str
    behavior_class: str
    settling_velocity_mps: float
    applicability_condition_zh: str
    calibration_status: str
    evidence_grade: str
    receptor_id: str
    receptor_lon_deg: float
    receptor_lat_deg: float
    receptor_template_z_m_positive_up: float
    vertical_id: str
    arrival_time_id: str
    arrival_time_utc_ns: int
    arrival_year: int
    season: str
    tide_class: str
    phase_or_event: str
    design_version: str
    initial_z_m_positive_up: float | None
    initial_eta_m_positive_up: float | None
    initial_bed_z_m_positive_up: float | None
    initial_water_column_height_m: float | None
    initial_height_above_bed_m: float | None
    initial_zcor_lower_m_positive_up: float | None
    initial_zcor_upper_m_positive_up: float | None
    initial_vertical_bracket_alpha: float | None
    initial_source_face_local_index: int | None
    initial_source_face_global_index: int | None
    initial_wetdry_elem_value: int | None
    initial_wetdry_semantics_id: str | None
    initial_ocm_month_yyyymm: str | None
    initial_ocm_source_time_index: int | None
    initial_ocm_time_origin: str | None

    def __post_init__(self) -> None:
        """驗證文字、座標、時間、物理量及動態條件的整組缺值政策。"""

        text_fields = (
            "scenario_id",
            "study_site_id",
            "analysis_region_id",
            "material_id",
            "material_category_zh",
            "material_family_zh",
            "representative_shape_zh",
            "behavior_class",
            "applicability_condition_zh",
            "calibration_status",
            "evidence_grade",
            "receptor_id",
            "vertical_id",
            "arrival_time_id",
            "season",
            "tide_class",
            "phase_or_event",
            "design_version",
        )
        for field_name in text_fields:
            _require_text(getattr(self, field_name), field_name)

        validated_float_values: dict[str, float] = {}
        for field_name in (
            "settling_velocity_mps",
            "receptor_lon_deg",
            "receptor_lat_deg",
            "receptor_template_z_m_positive_up",
        ):
            validated_float_values[field_name] = _require_finite_float(
                getattr(self, field_name), field_name
            )
        if validated_float_values["settling_velocity_mps"] >= 0.0:
            raise ValueError("settling_velocity_mps 必須嚴格小於 0")
        if not -180.0 <= validated_float_values["receptor_lon_deg"] <= 180.0:
            raise ValueError("receptor_lon_deg 必須位於 [-180, 180]")
        if not -90.0 <= validated_float_values["receptor_lat_deg"] <= 90.0:
            raise ValueError("receptor_lat_deg 必須位於 [-90, 90]")

        for field_name in ("arrival_time_utc_ns", "arrival_year"):
            _require_native_int(getattr(self, field_name), field_name)
        if self.arrival_year < 1:
            raise ValueError("arrival_year 必須是正整數年份")

        dynamic_float_fields = (
            "initial_z_m_positive_up",
            "initial_eta_m_positive_up",
            "initial_bed_z_m_positive_up",
            "initial_water_column_height_m",
            "initial_height_above_bed_m",
            "initial_zcor_lower_m_positive_up",
            "initial_zcor_upper_m_positive_up",
            "initial_vertical_bracket_alpha",
        )
        dynamic_int_fields = (
            "initial_source_face_local_index",
            "initial_source_face_global_index",
            "initial_wetdry_elem_value",
            "initial_ocm_source_time_index",
        )
        dynamic_text_fields = (
            "initial_wetdry_semantics_id",
            "initial_ocm_month_yyyymm",
            "initial_ocm_time_origin",
        )
        dynamic_fields = dynamic_float_fields + dynamic_int_fields + dynamic_text_fields
        dynamic_values = [getattr(self, field_name) for field_name in dynamic_fields]
        dynamic_present = [value is not None for value in dynamic_values]
        if any(dynamic_present) and not all(dynamic_present):
            raise ValueError("initial_* 動態初始條件必須全部有值或全部為 None")

        validated_dynamic_float_values = {
            field_name: _require_optional_finite_float(getattr(self, field_name), field_name)
            for field_name in dynamic_float_fields
        }
        validated_dynamic_int_values = {
            field_name: _require_optional_native_int(getattr(self, field_name), field_name)
            for field_name in dynamic_int_fields
        }
        validated_dynamic_text_values = {
            field_name: _require_optional_text(getattr(self, field_name), field_name)
            for field_name in dynamic_text_fields
        }

        water_column_height = validated_dynamic_float_values["initial_water_column_height_m"]
        height_above_bed = validated_dynamic_float_values["initial_height_above_bed_m"]
        if water_column_height is not None and water_column_height < 0.0:
            raise ValueError("initial_water_column_height_m 不可為負")
        if height_above_bed is not None and height_above_bed < 0.0:
            raise ValueError("initial_height_above_bed_m 不可為負")
        alpha = validated_dynamic_float_values["initial_vertical_bracket_alpha"]
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("initial_vertical_bracket_alpha 必須位於 [0, 1]")

        if all(
            validated_dynamic_float_values[field_name] is not None
            for field_name in (
                "initial_z_m_positive_up",
                "initial_zcor_lower_m_positive_up",
                "initial_zcor_upper_m_positive_up",
            )
        ):
            initial_z = validated_dynamic_float_values["initial_z_m_positive_up"]
            zcor_lower = validated_dynamic_float_values["initial_zcor_lower_m_positive_up"]
            zcor_upper = validated_dynamic_float_values["initial_zcor_upper_m_positive_up"]
            assert initial_z is not None
            assert zcor_lower is not None
            assert zcor_upper is not None
            if not zcor_lower <= initial_z <= zcor_upper:
                raise ValueError(
                    "initial_zcor_lower_m_positive_up 必須小於或等於 initial_z_m_positive_up，"
                    "且 initial_z_m_positive_up 必須小於或等於 initial_zcor_upper_m_positive_up"
                )

        for field_name in (
            "initial_source_face_local_index",
            "initial_source_face_global_index",
            "initial_ocm_source_time_index",
        ):
            index_value = validated_dynamic_int_values[field_name]
            if index_value is not None and index_value < 0:
                raise ValueError(f"{field_name} 必須大於或等於 0")

        wetdry_value = validated_dynamic_int_values["initial_wetdry_elem_value"]
        if wetdry_value is not None and wetdry_value != 0:
            raise ValueError("initial_wetdry_elem_value 必須等於 0")

        month_value = validated_dynamic_text_values["initial_ocm_month_yyyymm"]
        if month_value is not None and _YYYYMM_RE.fullmatch(month_value) is None:
            raise ValueError("initial_ocm_month_yyyymm 必須符合 YYYYMM，且月份必須是 01–12")

        # 將所有有限浮點欄位固定成 Python float；這是對外序列化前的唯一 canonical 邊界，
        # 不改寫文字、整數或 None，也不把來源缺值與乾點狀態折疊成數值零值。
        for field_name, value in validated_float_values.items():
            object.__setattr__(self, field_name, value)
        for field_name, value in validated_dynamic_float_values.items():
            object.__setattr__(self, field_name, value)

    @property
    def has_dynamic_initial_condition(self) -> bool:
        """回傳此列是否保有完整的 receptor×arrival 動態初始條件。"""

        return self.initial_z_m_positive_up is not None


def scenario_inputs_to_strata(
    inputs: ScenarioInputs, *, formal: bool
) -> tuple[ScenarioStratum, ...]:
    """依 ``ScenarioInputs.scenarios`` 原順序建立不可變分層資料列。

    此函式先為材料、受體、到達時間及受體×到達時間動態初始條件建立唯一索引，再以
    ``Scenario`` 保存的識別碼逐列做完全相等連接。連接過程會交叉核對站點、分析區域、
    世界協調時間（UTC）奈秒、垂向設定、沉降速度及設計版本；任一識別碼缺漏、重複或
    契約欄位不一致都立即拒絕，不採最近時間、替代受體或其他推測值。

    材料沉降速度沿用公尺／秒，經緯度沿用 WGS84 度數，垂向位置沿用海面向上為正的
    公尺值。動態欄位來自已驗收的海洋環流模式（OCM）受體×到達時間初始條件；若整份
    ``initial_conditions`` 為空，僅非正式模式可將十五個 ``initial_*`` 欄位整組保存為
    ``None``。只要輸入含任一動態初始條件，每個情境都必須有精確配對；正式模式則每列
    一律要求完整動態資料，不能以零值表示缺時、乾點或未知狀態。

    函式只建立新的 tuple 與資料列，不排序、不讀取檔案，也不修改 ``inputs`` 或其成員。
    輸出供條件式來源足跡與相對來源權重的分層聚合使用，不在此處執行物理計算。

    Args:
        inputs: 已載入並驗證的唯一 ``ScenarioInputs`` 實例。
        formal: 是否套用正式輸出的完整動態初始條件限制；必須是原生 ``bool``。

    Returns:
        與 ``inputs.scenarios`` 長度及順序完全一致的 ``ScenarioStratum`` tuple。

    Raises:
        TypeError: ``inputs`` 或 ``formal`` 不是契約指定的原生型別。
        ValueError: 識別碼重複或缺漏、連接欄位不一致，或動態條件完整性不符。
    """

    if type(inputs) is not ScenarioInputs:
        raise TypeError("inputs 必須是 exact ScenarioInputs 實例")
    if type(formal) is not bool:
        raise TypeError("formal 必須是原生 bool")

    design_version = _require_text(inputs.design_version, "inputs.design_version")

    # 每個 component 都先逐筆檢查重複後才寫入索引，避免 dict 建構式把較早資料靜默覆蓋。
    materials_by_id = {}
    for position, material in enumerate(inputs.materials):
        material_id = _require_text(
            material.material_id, f"inputs.materials[{position}].material_id"
        )
        if material_id in materials_by_id:
            raise ValueError(f"inputs.materials 的 material_id 重複：{material_id}")
        materials_by_id[material_id] = material

    receptors_by_id = {}
    for position, receptor in enumerate(inputs.receptors):
        receptor_id = _require_text(
            receptor.receptor_id, f"inputs.receptors[{position}].receptor_id"
        )
        if receptor_id in receptors_by_id:
            raise ValueError(f"inputs.receptors 的 receptor_id 重複：{receptor_id}")
        receptors_by_id[receptor_id] = receptor

    arrivals_by_id = {}
    for position, arrival in enumerate(inputs.arrival_times):
        arrival_time_id = _require_text(
            arrival.arrival_time_id,
            f"inputs.arrival_times[{position}].arrival_time_id",
        )
        if arrival_time_id in arrivals_by_id:
            raise ValueError(
                f"inputs.arrival_times 的 arrival_time_id 重複：{arrival_time_id}"
            )
        arrivals_by_id[arrival_time_id] = arrival

    # 動態初始條件以 receptor×arrival pair 唯一識別；同時驗證每筆來源紀錄確實對應
    # 已索引的受體與到達時間，避免未被 scenario 使用的額外紀錄藏有不一致 provenance。
    initial_conditions_by_pair = {}
    for position, initial in enumerate(inputs.initial_conditions):
        receptor_id = _require_text(
            initial.receptor_id,
            f"inputs.initial_conditions[{position}].receptor_id",
        )
        arrival_time_id = _require_text(
            initial.arrival_time_id,
            f"inputs.initial_conditions[{position}].arrival_time_id",
        )
        pair = (receptor_id, arrival_time_id)
        if pair in initial_conditions_by_pair:
            raise ValueError(
                "inputs.initial_conditions 的 receptor×arrival pair 重複："
                f"{receptor_id}×{arrival_time_id}"
            )

        receptor = receptors_by_id.get(receptor_id)
        if receptor is None:
            raise ValueError(
                f"inputs.initial_conditions[{position}] 找不到 receptor_id：{receptor_id}"
            )
        arrival = arrivals_by_id.get(arrival_time_id)
        if arrival is None:
            raise ValueError(
                "inputs.initial_conditions"
                f"[{position}] 找不到 arrival_time_id：{arrival_time_id}"
            )
        if initial.receptor_id != receptor.receptor_id:
            raise ValueError(
                f"inputs.initial_conditions[{position}] receptor_id 與 joined receptor 不一致"
            )
        if initial.arrival_time_id != arrival.arrival_time_id:
            raise ValueError(
                "inputs.initial_conditions"
                f"[{position}] arrival_time_id 與 joined arrival 不一致"
            )
        if (
            initial.study_site_id != receptor.study_site_id
            or initial.study_site_id != arrival.study_site_id
        ):
            raise ValueError(
                f"inputs.initial_conditions[{position}] study_site_id 與 joined records 不一致"
            )
        if initial.analysis_region_id != receptor.analysis_region_id:
            raise ValueError(
                "inputs.initial_conditions"
                f"[{position}] analysis_region_id 與 joined receptor 不一致"
            )
        if initial.time_utc_ns != arrival.time_utc_ns:
            raise ValueError(
                f"inputs.initial_conditions[{position}] UTC 奈秒與 joined arrival 不一致"
            )
        if initial.vertical_id != receptor.vertical_id:
            raise ValueError(
                f"inputs.initial_conditions[{position}] vertical_id 與 joined receptor 不一致"
            )
        initial_conditions_by_pair[pair] = initial

    has_initial_conditions = bool(inputs.initial_conditions)
    if formal and not has_initial_conditions:
        raise ValueError("formal=True 時 inputs.initial_conditions 不可為空")

    # 依 scenario 原順序建立輸出；set 只用於檢查 scenario_id，不參與排序或重新分組。
    seen_scenario_ids: set[str] = set()
    strata: list[ScenarioStratum] = []
    for position, scenario in enumerate(inputs.scenarios):
        scenario_id = _require_text(
            scenario.scenario_id, f"inputs.scenarios[{position}].scenario_id"
        )
        if scenario_id in seen_scenario_ids:
            raise ValueError(f"inputs.scenarios 的 scenario_id 重複：{scenario_id}")
        seen_scenario_ids.add(scenario_id)

        material = materials_by_id.get(scenario.material_id)
        if material is None:
            raise ValueError(
                f"inputs.scenarios[{position}] 找不到 material_id：{scenario.material_id}"
            )
        receptor = receptors_by_id.get(scenario.receptor_id)
        if receptor is None:
            raise ValueError(
                f"inputs.scenarios[{position}] 找不到 receptor_id：{scenario.receptor_id}"
            )
        arrival = arrivals_by_id.get(scenario.arrival_time_id)
        if arrival is None:
            raise ValueError(
                "inputs.scenarios"
                f"[{position}] 找不到 arrival_time_id：{scenario.arrival_time_id}"
            )

        if scenario.material_id != material.material_id:
            raise ValueError(
                f"inputs.scenarios[{position}] material_id 與 joined material 不一致"
            )
        if scenario.receptor_id != receptor.receptor_id:
            raise ValueError(
                f"inputs.scenarios[{position}] receptor_id 與 joined receptor 不一致"
            )
        if scenario.arrival_time_id != arrival.arrival_time_id:
            raise ValueError(
                f"inputs.scenarios[{position}] arrival_time_id 與 joined arrival 不一致"
            )
        if (
            scenario.study_site_id != receptor.study_site_id
            or scenario.study_site_id != arrival.study_site_id
        ):
            raise ValueError(
                f"inputs.scenarios[{position}] study_site_id 與 receptor/arrival 不一致"
            )
        if scenario.analysis_region_id != receptor.analysis_region_id:
            raise ValueError(
                f"inputs.scenarios[{position}] analysis_region_id 與 receptor 不一致"
            )
        if scenario.arrival_time_utc_ns != arrival.time_utc_ns:
            raise ValueError(
                f"inputs.scenarios[{position}] arrival_time_utc_ns 與 ArrivalTime 不一致"
            )
        if scenario.settling_velocity_mps != material.settling_velocity_mps:
            raise ValueError(
                f"inputs.scenarios[{position}] settling_velocity_mps 與 Behavior 不一致"
            )
        if scenario.design_version != design_version:
            raise ValueError(
                f"inputs.scenarios[{position}] design_version 與 inputs.design_version 不一致"
            )

        pair = (scenario.receptor_id, scenario.arrival_time_id)
        initial = initial_conditions_by_pair.get(pair)
        if has_initial_conditions and initial is None:
            raise ValueError(
                "inputs.scenarios"
                f"[{position}] 找不到 receptor×arrival 動態初始條件：{pair[0]}×{pair[1]}"
            )
        if formal and initial is None:
            raise ValueError(f"inputs.scenarios[{position}] 正式模式必須有動態初始條件")

        # Pilot 僅在整份 initial_conditions 為空時保留整組 None；有來源紀錄時則逐欄原值
        # 映射，單位與正向約定交由 ScenarioStratum 的資料契約再次驗證並 canonical 化。
        if initial is None:
            dynamic_values = {
                "initial_z_m_positive_up": None,
                "initial_eta_m_positive_up": None,
                "initial_bed_z_m_positive_up": None,
                "initial_water_column_height_m": None,
                "initial_height_above_bed_m": None,
                "initial_zcor_lower_m_positive_up": None,
                "initial_zcor_upper_m_positive_up": None,
                "initial_vertical_bracket_alpha": None,
                "initial_source_face_local_index": None,
                "initial_source_face_global_index": None,
                "initial_wetdry_elem_value": None,
                "initial_wetdry_semantics_id": None,
                "initial_ocm_month_yyyymm": None,
                "initial_ocm_source_time_index": None,
                "initial_ocm_time_origin": None,
            }
        else:
            dynamic_values = {
                "initial_z_m_positive_up": initial.z_m_positive_up,
                "initial_eta_m_positive_up": initial.eta_m_positive_up,
                "initial_bed_z_m_positive_up": initial.bed_z_m_positive_up,
                "initial_water_column_height_m": initial.water_column_height_m,
                "initial_height_above_bed_m": initial.height_above_bed_m,
                "initial_zcor_lower_m_positive_up": initial.zcor_lower_m_positive_up,
                "initial_zcor_upper_m_positive_up": initial.zcor_upper_m_positive_up,
                "initial_vertical_bracket_alpha": initial.vertical_bracket_alpha,
                "initial_source_face_local_index": initial.source_face_local_index,
                "initial_source_face_global_index": initial.source_face_global_index,
                "initial_wetdry_elem_value": initial.wetdry_elem_value,
                "initial_wetdry_semantics_id": initial.wetdry_semantics_id,
                "initial_ocm_month_yyyymm": initial.ocm_month_yyyymm,
                "initial_ocm_source_time_index": initial.ocm_source_time_index,
                "initial_ocm_time_origin": initial.ocm_time_origin,
            }

        strata.append(
            ScenarioStratum(
                scenario_id=scenario_id,
                study_site_id=scenario.study_site_id,
                analysis_region_id=scenario.analysis_region_id,
                material_id=material.material_id,
                material_category_zh=material.oca_category_zh,
                material_family_zh=material.material_family_zh,
                representative_shape_zh=material.representative_shape_zh,
                behavior_class=material.behavior_class,
                settling_velocity_mps=material.settling_velocity_mps,
                applicability_condition_zh=material.applicability_condition_zh,
                calibration_status=material.calibration_status,
                evidence_grade=material.evidence_grade,
                receptor_id=receptor.receptor_id,
                receptor_lon_deg=receptor.lon,
                receptor_lat_deg=receptor.lat,
                receptor_template_z_m_positive_up=receptor.z_m_positive_up,
                vertical_id=receptor.vertical_id,
                arrival_time_id=arrival.arrival_time_id,
                arrival_time_utc_ns=arrival.time_utc_ns,
                arrival_year=arrival.year,
                season=arrival.season,
                tide_class=arrival.tide_class,
                phase_or_event=arrival.phase_or_event,
                design_version=design_version,
                **dynamic_values,
            )
        )

    return tuple(strata)


__all__ = ["AggregateShardBinding", "ScenarioStratum", "scenario_inputs_to_strata"]
