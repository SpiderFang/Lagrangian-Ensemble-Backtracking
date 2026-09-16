"""建立 pilot/formal run workspace 與單一 ``RunUnit`` 的參考粒子請求。

本模組只負責把已驗證的 scenario、receptor×arrival 動態初始條件、站點幾何與
設定值組合成既有 ``ReferenceParticleRequest``。OCM native 與 NWW3 analysis 的
大型陣列仍由 ``ForcingWindowManager`` 在第一次真正需要某個 flow domain 時才開啟；
因此建立 workspace、建構 factory 或拒絕不一致的 manifest 都不會碰觸 forcing 內容。

``initialize_run`` 與 ``open_run_controller`` 依 immutable plan 明確區分 pilot/formal。
pilot 保留原本的結構與 hash gate；formal 另要求核准設定、manifest、完整月份 topology、
各 flow-domain 時間軸，以及逐 arrival 所屬 flow 的 OCM full-product 或 gap-safe 支援。
兩種模式都把 canonical hash、程式部署 provenance、固定 seed／分片設定交給既有
``initialize_run_workspace``，且不允許未知模式降級。inventory 在入口只讀一次並保留解析後
的 mapping，後續以 mapping 交給 run-control 的 canonical JSON 寫入，避免再次依路徑讀檔
造成替換檔案的 TOCTOU（檢查與使用之間的時間差）問題；absolute input/server path 不會被
加入 run plan。程式可建立 formal workspace 不代表 SERVER 科學批次或來源歸因已完成。

``load_validated_run_static_inputs`` 是 controller 與 aggregate pipeline 共用的唯讀 static
binding 邊界：它只讀 immutable run 文件、設定 manifest、geometry manifest 與 inventory，
回傳不含 workspace／forcing root 的 defensive snapshot。它不建立 forcing manager、不讀取
OCM/NWW array，也不把本機 synthetic fixture 當成正式 OCM／NWW 科學成果。

``EXPERIMENT_CASE_SPECS`` 是 runtime physics case 的唯一 registry：兩個常數擴散案例維持
既有 ``DiffusionCoefficients`` 數值路徑，三個 ``smagorinsky_cs_*`` 案例則以 immutable
settings 建立 OCM-only 空間擴散 facade；三個 Smagorinsky velocity path 仍啟用有限水深
Stokes，因此 request 仍需 NWW root。registry、設定 parser 與 CLI 只負責明確 wiring，
不表示 Smagorinsky 已通過正式 well-mixed、PDE、收斂與 floor/cap 科學 gate。

此 factory 的生命週期限定為一個 process／worker。``ForcingWindowManager`` 本身
沒有跨執行緒鎖，provider 與 manager cache 也不是 thread-safe；多程序執行時每個
worker 必須各自建立自己的 factory，不能把同一個 instance 分享給其他 thread。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow.parquet as pq
from shapely.geometry import Point

from .bed_residence import BedResidenceTiming, resolve_bed_residence_timing
from .config import (
    OCM_INTERPOLATION_BACKEND_NUMBA_V1,
    ProjectConfig,
    load_config,
    resolve_flow_domain_id,
)
from .diffusion import DiffusionCoefficients, SmagorinskySettings
from .engine import EngineSettings
from .forcing_window import ForcingWindowManager
from .manifests import (
    BoundaryGeometryBundle,
    ScenarioInputs,
    load_boundary_geometries,
    load_scenario_inputs,
    resolve_manifest_path,
)
from .models import ParticleState, ParticleStatus, VelocitySample
from .pilot_selection import (
    apply_scenario_selection,
    build_full_scenario_selection,
    select_exact_pilot_scenarios,
    select_pilot_scenarios,
)
from .preflight import Finding, MonthInventory, TimeAxisInventory, TimeGapInventory
from .provenance import collect_code_provenance
from .run_control import (
    _SCENARIO_COLUMNS,
    RunController,
    RunWorkspace,
    _scenario_from_row,
    _scenario_hash,
    initialize_run_workspace,
    load_run_plan,
)
from .run_validation import validate_run
from .runner import ReferenceParticleRequest, RunUnit, scenario_execution_sort_key
from .scenarios import validate_baseline_coverage


def _forbidden_pre_window_velocity(
    x_m: float, y_m: float, z_m: float, time_utc_ns: int
) -> VelocitySample:
    """讓 pre-window request 保持可序列化，但若錯誤路徑取樣便立即暴露。

    固定日曆窗前已沉底的成員沒有研究窗內漂流期間，runtime 不應為它開啟 forcing
    manager 或建立真實 velocity provider。這個 sentinel 只填滿既有 request 型別；
    terminal initialization 會在任何 velocity call 前短路，故它不代表缺值補零。
    """

    del x_m, y_m, z_m, time_utc_ns
    raise RuntimeError("PRE_WINDOW_DEPOSITION request 不得取樣 velocity 或 forcing")


@dataclass(frozen=True, slots=True)
class ExperimentCaseSpec:
    """單一 runtime experiment case 的 immutable 註冊規格。

    ``experiment_case_id`` 是 run plan／``RunUnit`` 使用的穩定識別碼；``include_stokes``
    決定 velocity path 是否需要 NWW3 有限水深 Stokes 漂移；``diffusion_kind`` 則獨立
    決定 request 的擴散 provider 是常數 ``DiffusionCoefficients`` 或 OCM-only 的
    Smagorinsky facade。Smagorinsky 案例的 ``coefficient_cs`` 是無因次係數，常數案例
    必須保持 ``None``。這個資料類別不保存任何 forcing array，也不代表該案例已通過
    well-mixed、PDE、收斂或正式發布 gate。
    """

    experiment_case_id: str
    include_stokes: bool
    diffusion_kind: str
    coefficient_cs: float | None = None

    def __post_init__(self) -> None:
        """鎖定 registry 的結構約束，避免新增案例時默默改變物理分支。"""

        if type(self.experiment_case_id) is not str or not self.experiment_case_id.strip():
            raise ValueError("experiment_case_id 不可為空白")
        if not isinstance(self.include_stokes, bool):
            raise TypeError("include_stokes 必須是 bool")
        if self.diffusion_kind not in {"constant", "smagorinsky"}:
            raise ValueError("diffusion_kind 只允許 constant 或 smagorinsky")
        if self.diffusion_kind == "constant" and self.coefficient_cs is not None:
            raise ValueError("constant experiment case 不可帶 coefficient_cs")
        if self.diffusion_kind == "smagorinsky":
            if self.coefficient_cs is None or isinstance(self.coefficient_cs, bool):
                raise ValueError("smagorinsky experiment case 必須帶有限 coefficient_cs")
            try:
                coefficient_cs = float(self.coefficient_cs)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "smagorinsky experiment case 必須帶有限 coefficient_cs"
                ) from exc
            if not math.isfinite(coefficient_cs):
                raise ValueError("smagorinsky experiment case 必須帶有限 coefficient_cs")
            if coefficient_cs <= 0:
                raise ValueError("smagorinsky coefficient_cs 必須大於 0")
        normalized_cs = None if self.coefficient_cs is None else float(self.coefficient_cs)
        object.__setattr__(self, "coefficient_cs", normalized_cs)

    @property
    def diffusion_method(self) -> str:
        """回傳較語意化的擴散方法別名，供外部 registry inspection 使用。"""

        return self.diffusion_kind

    @property
    def smagorinsky_cs(self) -> float | None:
        """回傳 Smagorinsky ``Cs`` 別名；常數案例固定為 ``None``。"""

        return self.coefficient_cs


EXPERIMENT_CASE_SPECS = MappingProxyType(
    {
        "finite_depth_stokes": ExperimentCaseSpec(
            "finite_depth_stokes", True, "constant"
        ),
        "no_stokes": ExperimentCaseSpec("no_stokes", False, "constant"),
        "smagorinsky_cs_010": ExperimentCaseSpec(
            "smagorinsky_cs_010", True, "smagorinsky", 0.10
        ),
        "smagorinsky_cs_015": ExperimentCaseSpec(
            "smagorinsky_cs_015", True, "smagorinsky", 0.15
        ),
        "smagorinsky_cs_020": ExperimentCaseSpec(
            "smagorinsky_cs_020", True, "smagorinsky", 0.20
        ),
    }
)
"""五個 runtime experiment case 的唯一、唯讀註冊表。

前兩個案例維持既有常數擴散 reference；後三個是連接到 runtime 的 Smagorinsky
sensitivity case。所有 initializer、static-input gate、factory 與 CLI 都只能由這份
registry 判斷合法性，不得對未知字串猜測 Stokes 或擴散方法；registry 通過只代表 wiring
存在，不能把尚未通過科學驗證的敏感度案例宣稱為正式結果。
"""

EXPERIMENT_CASE_INCLUDE_STOKES = MappingProxyType(
    {case_id: spec.include_stokes for case_id, spec in EXPERIMENT_CASE_SPECS.items()}
)
"""由 ``EXPERIMENT_CASE_SPECS`` 推導的相容唯讀 Stokes 對照表。

舊呼叫端仍可查詢是否需要 NWW3，但不再以這個衍生 mapping 登錄案例；五個案例的
擴散方法與 ``Cs`` 必須回到唯一 registry 取得。
"""


def _experiment_case_spec(experiment_case_id: object) -> ExperimentCaseSpec:
    """由唯一 registry 取得 case spec，未知識別碼一律 fail-closed。"""

    if type(experiment_case_id) is not str:
        raise ValueError(f"未知 experiment_case_id，禁止 fallback：{experiment_case_id!r}")
    try:
        return EXPERIMENT_CASE_SPECS[experiment_case_id]
    except KeyError as exc:
        raise ValueError(f"未知 experiment_case_id，禁止 fallback：{experiment_case_id!r}") from exc

_INVENTORY_KEYS = frozenset(
    {"created_at_utc", "config_hash", "mode", "formal_ready", "inventories", "time_axes", "findings"}
)
"""pilot 與 formal preflight inventory 共用的固定根欄位集合。

這裡只定義磁碟資料契約的根形狀；pilot 仍只做結構與 hash gate，formal 則在相同
strict JSON parser 之上另行驗證產品拓撲、時間軸與 gap-safe 支援。
"""

_PILOT_INVENTORY_KEYS = _INVENTORY_KEYS
"""保留舊私有名稱，讓既有測試與相容程式碼仍能辨識 pilot 根欄位。"""

_FORMAL_INVENTORY_RECORD_KEYS = frozenset(field.name for field in fields(MonthInventory))
"""正式 inventory 的月份列固定欄位，直接沿用 preflight ``MonthInventory`` dataclass。"""

_FORMAL_TIME_AXIS_KEYS = frozenset(field.name for field in fields(TimeAxisInventory))
"""正式 inventory 的跨月時間軸列固定欄位，直接沿用 preflight dataclass。"""

_FORMAL_TIME_GAP_KEYS = frozenset(field.name for field in fields(TimeGapInventory))
"""正式 inventory 的缺口列固定欄位，避免不同模組自行發明 gap schema。"""

_FORMAL_FINDING_KEYS = frozenset(field.name for field in fields(Finding))
"""正式 inventory 的 finding 固定欄位，與 preflight ``Finding`` 保持一致。"""

_RUNTIME_RUN_KINDS = frozenset({"pilot", "formal"})
"""runtime／CLI 允許的執行模式；synthetic 只屬於 run-control 測試資料，不得進入物理 runtime。"""

_PILOT_COMPONENT_HASH_KEYS = (
    "material",
    "receptor",
    "arrival",
    "receptor_arrival_initial_condition",
)
"""pilot scenario component canonical hash 的固定且完整順序。"""

_PILOT_GEOMETRY_HASH_KEYS = ("domain", "local", "open_boundary")
"""pilot geometry canonical hash 的固定且完整順序。"""

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
"""只接受 64 位小寫十六進位 SHA-256 的正規表示式。"""


def _freeze_static_plan_value(value: object) -> object:
    """遞迴封存 run plan，讓回傳 snapshot 不保留可改寫的 JSON 容器 alias。

    run plan 來自 strict JSON，因此實際值只有 mapping、list 與 JSON scalar；mapping
    轉成 ``MappingProxyType``、list 轉成 tuple，scalar 則保持原生型別與值。額外處理
    tuple 是為了讓這個 private 邊界即使被測試或未來的 loader 傳入 tuple，也不會把巢狀
    可變容器原樣帶進公開 snapshot。此函式只做記憶體防禦，不將任何 path 轉成或寫入
    plan，也不改變 run plan 的資料語意。
    """

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_static_plan_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_static_plan_value(item) for item in value)
    return value


def _defensive_scenario_inputs(
    source: ScenarioInputs,
    *,
    scenarios: Sequence[Any] | None = None,
) -> ScenarioInputs:
    """重建 ScenarioInputs，隔離 loader 的巢狀 mapping、pair 與情境資料參照。

    ``ScenarioInputs`` 的 tuple／mapping proxy 只封存最外層；receptor、arrival 與
    dynamic initial-condition record 仍可能包含 metadata dict，``BoundaryGeometry``
    也有 foreign-domain mapping。因此 static binding 回傳前必須 deep-copy 所有資料，
    並可選擇以 plan execution order 取代原始 loader 順序。座標與深度仍遵守上游資料
    契約：經緯度僅是交換欄位，實際 z 與幾何運算欄位的長度單位為公尺，時間欄位為 UTC
    奈秒；此函式不補缺值、不把 gap 轉成零，也不讀取 forcing。
    """

    if not isinstance(source, ScenarioInputs):
        raise TypeError("scenario_inputs 必須是 ScenarioInputs")
    ordered_scenarios = source.scenarios if scenarios is None else tuple(scenarios)
    return ScenarioInputs(
        materials=deepcopy(source.materials),
        receptors=deepcopy(source.receptors),
        arrival_times=deepcopy(source.arrival_times),
        scenarios=deepcopy(ordered_scenarios),
        file_sha256=deepcopy(dict(source.file_sha256)),
        canonical_component_hashes=deepcopy(dict(source.canonical_component_hashes)),
        design_version=source.design_version,
        initial_conditions=deepcopy(source.initial_conditions),
        initial_conditions_by_pair=deepcopy(dict(source.initial_conditions_by_pair)),
    )


def _defensive_geometry_bundle(source: BoundaryGeometryBundle) -> BoundaryGeometryBundle:
    """重建公尺制 geometry bundle，隔離 foreign-domain mapping 與 projection 參照。

    loader 回傳的 bundle 已封存最外層 mapping，但單一幾何資料仍含可變的 foreign local
    domain dict；這裡以 deep-copy 重建，避免 caller 透過回傳物件改動 loader 或其他流程
    持有的資料。projection 只保留公尺制轉換器，不引入 WGS84 array、forcing root 或
    SERVER 路徑；geometry 數值仍由既有 loader 驗證，不在此重新投影或補資料。
    """

    if not isinstance(source, BoundaryGeometryBundle):
        raise TypeError("geometries 必須是 BoundaryGeometryBundle")
    return BoundaryGeometryBundle(
        geometries=deepcopy(dict(source.geometries)),
        projections=deepcopy(dict(source.projections)),
        file_sha256=deepcopy(dict(source.file_sha256)),
        canonical_component_hashes=deepcopy(dict(source.canonical_component_hashes)),
    )


@dataclass(frozen=True, slots=True)
class ValidatedRunStaticInputs:
    """controller 與 aggregate pipeline 共用的已驗證 static-input snapshot。

    ``plan`` 是從磁碟 ``run_plan.json`` 載入後的 recursive immutable mapping；其中
    shard 的半開區間是 scenario execution order 的索引，particle count 是
    ``members_per_scenario`` 倍數。``config`` 保存 deep-copied ProjectConfig；
    ``scenario_inputs`` 保存依 plan 順序重建的 material／receptor／arrival、OCM-derived
    receptor×arrival 初始條件與 hash；``geometries`` 保存已投影至公尺座標的幾何及其 hash。

    此容器刻意不保存 workspace、config 檔案、OCM/NWW root、forcing manager、provider、
    factory 或任何 SERVER absolute path。它只證明 static binding 通過，並不表示已讀取
    OCM/NWW 大型 array，也不把 synthetic 本機測試資料提升為正式 OCM／NWW 科學成果。
    缺值、時間 gap 與 dry/land 狀態由上游 manifest／inventory 維持原狀，這個 snapshot
    不以零值或最近值補齊。
    """

    plan: Mapping[str, object]
    config: ProjectConfig
    scenario_inputs: ScenarioInputs
    geometries: BoundaryGeometryBundle

    def __post_init__(self) -> None:
        """在公開邊界再次封存所有輸入，避免直接建構時繞過 defensive snapshot。"""

        if not isinstance(self.plan, Mapping):
            raise TypeError("plan 必須是 Mapping")
        if not isinstance(self.config, ProjectConfig):
            raise TypeError("config 必須是 ProjectConfig")
        if not isinstance(self.scenario_inputs, ScenarioInputs):
            raise TypeError("scenario_inputs 必須是 ScenarioInputs")
        if not isinstance(self.geometries, BoundaryGeometryBundle):
            raise TypeError("geometries 必須是 BoundaryGeometryBundle")
        immutable_plan = _freeze_static_plan_value(self.plan)
        if not isinstance(immutable_plan, Mapping):
            raise TypeError("plan 必須是 Mapping")
        object.__setattr__(self, "plan", immutable_plan)
        object.__setattr__(self, "config", self.config.model_copy(deep=True))
        object.__setattr__(self, "scenario_inputs", _defensive_scenario_inputs(self.scenario_inputs))
        object.__setattr__(self, "geometries", _defensive_geometry_bundle(self.geometries))


def _unique_index(records: Any, *, key_name: str, label: str) -> Mapping[str, Any]:
    """把已驗證 records 轉成不可改寫的識別碼索引。

    索引只保存 manifest dataclass 的參照，不複製或讀取任何 forcing array。重複識別碼
    會破壞 ``RunUnit`` 的 deterministic lookup，因此不採用最後一筆覆蓋前一筆的作法，
    而是在 factory 建立時直接拒絕。
    """

    indexed: dict[str, Any] = {}
    for record in records:
        key = getattr(record, key_name)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} 的 {key_name} 必須是非空字串")
        if key in indexed:
            raise ValueError(f"{label} 的 {key_name} 不可重複：{key}")
        indexed[key] = record
    return MappingProxyType(indexed)


def _validated_root(root: str | Path, *, label: str) -> Path:
    """確認上游 root 是現有、非符號連結的目錄。

    root 只代表已驗收產品的容器位置；本函式不進入其下的 flow domain、grid 或月份
    目錄，也不讀取檔案。拒絕符號連結可避免實際產品位置在 run 期間悄悄切換，並使
    manifest 中保存的 root 語意保持可稽核。
    """

    try:
        path = Path(root)
    except TypeError as exc:
        raise TypeError(f"{label} 必須是 str 或 Path") from exc
    if path.is_symlink():
        raise ValueError(f"{label} 必須是既有非 symlink 目錄：{path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} 必須是既有目錄：{path}")
    return path


def _finite_scalar(value: Any, *, label: str, minimum: float, allow_zero: bool) -> float:
    """嚴格驗證設定中的有限數值並回傳標準 Python ``float``。

    YAML／Pydantic 可能保存整數或浮點數，但布林值在 Python 中也是整數子類別，不能
    讓 ``True`` 靜默變成一秒或一個係數。所有時間與擴散 scalar 都在這裡拒絕 ``None``、
    非數字、NaN、Infinity 及不合物理範圍的值；實際單位由呼叫端的欄位名稱說明。
    """

    if value is None or isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise TypeError(f"{label} 必須是有限數值")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 不可為 NaN 或 Infinity")
    valid = normalized >= minimum if allow_zero else normalized > minimum
    if not valid:
        relation = f">= {minimum}" if allow_zero else f"> {minimum}"
        raise ValueError(f"{label} 必須 {relation}")
    return normalized


def _positive_integer(value: Any, *, label: str) -> int:
    """驗證不可為布林、零、負數或浮點近似值的正整數設定。"""

    if value is None or isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} 必須是正整數")
    if value < 1:
        raise ValueError(f"{label} 必須是正整數")
    return int(value)


def _backtrack_horizon(max_days: float) -> tuple[float, int]:
    """將設定的回溯天數轉為一致的秒數與整數奈秒，不任意捨入次奈秒時間。

    輸入先驗證為有限正浮點數；一天固定為 86,400 秒、一秒為十億奈秒。舊十進位
    天數規則若已給出整數奈秒，原值完全保留。否則先沿用設定建立器的浮點乘法轉秒，
    再由秒數的最短十進位表示找最近整數奈秒候選。候選僅在以下三項同時成立時接受：
    誤差嚴格小於 0.5 ns、不超過秒數兩個相鄰浮點間距（ULP），且候選轉回秒／天後
    與原輸入 float 完全相同。兩個間距涵蓋乘法與十進位表示誤差，但不能取代往返核對。

    因此 1/24 天及由 10 ns 換算的天數可還原，86.4 ns、10.5 ns 或可區分的鄰近
    天數不得被當成整數奈秒。浮點輸入本來無法區分的更細時間也不能由此恢復；此規則
    不是宣稱任意時間都具奈秒準確度。非正值、非有限秒數及超過有號 64 位奈秒範圍
    均拒絕；不改到達時間、設定檔或物理步長。兩個 runtime 入口必須共用本規則。
    """

    days = _finite_scalar(
        max_days, label="boundaries.max_backtrack_days", minimum=0.0, allow_zero=False,
    )
    seconds = days * 86_400.0
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("max_backtrack_days 換算後的秒數必須有限且為正")
    # 局部精度足以保存 float 最短字串乘單位的所有有效位數，且不受呼叫端 Decimal 設定影響。
    with localcontext() as context:
        context.prec = 50
        scale = Decimal(1_000_000_000)
        horizon = Decimal(str(days)) * Decimal(86_400) * scale
        if horizon != horizon.to_integral_value():
            normalized_ns = Decimal(str(seconds)) * scale
            candidate = normalized_ns.to_integral_value(rounding=ROUND_HALF_EVEN)
            error_ns = abs(candidate - normalized_ns)
            tolerance_ns = Decimal(str(math.ulp(seconds))) * 2 * scale
            candidate_seconds = float(candidate / scale)
            if (error_ns >= Decimal("0.5") or error_ns > tolerance_ns
                    or candidate_seconds / 86_400.0 != days):
                raise ValueError("boundaries.max_backtrack_days 換算後的 horizon ns 必須是可驗證整數")
            horizon = candidate
        if not 1 <= horizon <= (1 << 63) - 1:
            raise ValueError("horizon ns 必須位於正有號 64 位整數範圍")
        return float(horizon / scale), int(horizon)


def _required_mapping_value(container: Any, *, label: str, key: str) -> Any:
    """從巢狀設定 mapping 取出必填欄位，不以預設值掩蓋缺漏。"""

    if not isinstance(container, Mapping) or key not in container:
        raise ValueError(f"缺少 {label}.{key}")
    return container[key]


def _build_smagorinsky_settings(
    horizontal_diffusion: Any,
    *,
    coefficient_cs: float | None,
    constant_kz_m2ps: float,
) -> SmagorinskySettings:
    """解析指定 experiment case 的 Smagorinsky 設定並建立 immutable settings。

    ``horizontal_diffusion.smagorinsky`` 必須是顯式 mapping；其中
    ``cs_sensitivity`` 是用來證明固定 runtime case 已登錄在設定中的非空係數序列，
    每個係數都必須是非 bool、有限、嚴格大於零且不可重複。floor/cap 是每個 native
    triangle candidate 套用的 Kh（m²/s）界線，兩者都必須存在、有限、非負且 floor 不得
    大於 cap；``null`` 會直接 fail-closed，不會改用隱含預設值。Kz 已由 factory 的共同
    vertical diffusion parser 驗證，這裡只把它綁定到同一個 immutable settings，避免
    Smagorinsky case 讀取不同來源的垂向係數。

    ``coefficient_cs`` 來自唯一 experiment registry；只有 registry 指定的 case Cs 被
    ``cs_sensitivity`` 明確列出才可建立 request。這個 gate 只保證 wiring 與參數契約，
    不宣稱 Smagorinsky 已通過 well-mixed、PDE 或收斂驗證。
    """

    if not isinstance(horizontal_diffusion, Mapping):
        raise TypeError("physics.horizontal_diffusion 必須是 mapping")
    smagorinsky = _required_mapping_value(
        horizontal_diffusion,
        label="physics.horizontal_diffusion",
        key="smagorinsky",
    )
    if not isinstance(smagorinsky, Mapping):
        raise TypeError("physics.horizontal_diffusion.smagorinsky 必須是 mapping")

    raw_sensitivity = _required_mapping_value(
        smagorinsky,
        label="physics.horizontal_diffusion.smagorinsky",
        key="cs_sensitivity",
    )
    if isinstance(raw_sensitivity, (str, bytes)) or not isinstance(raw_sensitivity, Sequence):
        raise TypeError("physics.horizontal_diffusion.smagorinsky.cs_sensitivity 必須是 sequence")
    if not raw_sensitivity:
        raise ValueError("physics.horizontal_diffusion.smagorinsky.cs_sensitivity 不可為空")

    sensitivity: list[float] = []
    for index, raw_cs in enumerate(raw_sensitivity):
        value = _finite_scalar(
            raw_cs,
            label=(
                "physics.horizontal_diffusion.smagorinsky"
                f".cs_sensitivity[{index}]"
            ),
            minimum=0.0,
            allow_zero=False,
        )
        if value in sensitivity:
            raise ValueError(
                "physics.horizontal_diffusion.smagorinsky.cs_sensitivity 不可包含重複值"
            )
        sensitivity.append(value)

    if coefficient_cs is None:
        raise ValueError("Smagorinsky experiment case 缺少 registry coefficient_cs")
    if coefficient_cs not in sensitivity:
        raise ValueError(
            "experiment case coefficient_cs 必須列在 "
            "physics.horizontal_diffusion.smagorinsky.cs_sensitivity"
        )
    floor_m2ps = _finite_scalar(
        _required_mapping_value(
            smagorinsky,
            label="physics.horizontal_diffusion.smagorinsky",
            key="kh_floor_m2ps",
        ),
        label="physics.horizontal_diffusion.smagorinsky.kh_floor_m2ps",
        minimum=0.0,
        allow_zero=True,
    )
    cap_m2ps = _finite_scalar(
        _required_mapping_value(
            smagorinsky,
            label="physics.horizontal_diffusion.smagorinsky",
            key="kh_cap_m2ps",
        ),
        label="physics.horizontal_diffusion.smagorinsky.kh_cap_m2ps",
        minimum=0.0,
        allow_zero=True,
    )
    if floor_m2ps > cap_m2ps:
        raise ValueError(
            "physics.horizontal_diffusion.smagorinsky.kh_floor_m2ps 不可大於 kh_cap_m2ps"
        )
    return SmagorinskySettings(
        coefficient_cs=coefficient_cs,
        floor_m2ps=floor_m2ps,
        cap_m2ps=cap_m2ps,
        constant_kz_m2ps=constant_kz_m2ps,
    )


def _nonnegative_integer(value: Any, *, label: str) -> int:
    """驗證不可由布林冒充、且允許零值的原生 Python 非負整數。

    seed 是執行身份的一部分，零是合法且可重現的主種子；其餘型別（包含浮點數、
    NumPy scalar 與 ``bool``）不在這個資料契約內，不能靠隱式轉型放行。回傳原生
    ``int`` 讓後續 seed table 與 JSON plan 的型別保持穩定。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是非負整數，且不可為 bool")
    if value < 0:
        raise ValueError(f"{label} 必須是非負整數")
    return value


def _validated_canonical_hashes(
    value: Any, *, expected_keys: tuple[str, ...], label: str
) -> dict[str, str]:
    """驗證並依固定順序複製 component／geometry canonical hash mapping。

    canonical hash 是上游 JSON 語意內容的 SHA-256，不是檔案路徑或 forcing array 的
    內容。這裡要求 key 集合完全相等，故缺少 dynamic initial condition 或夾帶未登錄
    component 都會在 workspace 發布前 fail-closed；複製後的 dict 會成為傳給
    run-control 的 immutable-input snapshot。
    """

    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        expected = ",".join(expected_keys)
        raise ValueError(f"{label} 必須 exact 包含：{expected}")
    result: dict[str, str] = {}
    for key in expected_keys:
        digest = value[key]
        if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{label}.{key} 必須是 64 位小寫 SHA-256")
        result[key] = digest
    return result


def _reject_inventory_json_constant(value: str) -> None:
    """拒絕 JSON parser 對 ``NaN``／``Infinity`` 的非標準寬鬆解析。"""

    raise ValueError(f"pilot preflight inventory 含非有限 JSON 常數：{value}")


def _assert_finite_inventory_value(value: Any, *, label: str) -> None:
    """遞迴拒絕 JSON 數字溢位後形成的 Infinity。

    ``parse_constant`` 能攔截文字形式的 ``NaN``、``Infinity``，但極大指數如 ``1e400``
    可能先被 Python JSON parser 轉成無限浮點數；因此仍需在解析後走訪 object/list。
    此檢查供 pilot inventory 與 normalized config 共用；兩者都只在 JSON binding 邊界
    保存資料，不在此函式進行額外的科學語意推論，但任何非有限數值都不能進入
    canonical JSON hash 或 run workspace。
    """

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} 不可含 NaN 或 Infinity")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_inventory_value(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_inventory_value(item, label=f"{label}[{index}]")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """讓 strict JSON reader 拒絕重複物件鍵，避免不同原始 bytes 被折疊成同一 mapping。

    Python 的一般 ``json.loads`` 會以最後一個值覆蓋同名鍵；normalized config 是 run
    binding 的輸入，若接受這種覆蓋，檔案原始內容便可能與解析後 mapping 的語意不一致。
    因此此 hook 保留標準 JSON 的 key/value 結構，但在發生重複鍵時立即 fail closed。
    """

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("strict JSON object 不可含重複 key")
        result[key] = value
    return result


def _read_strict_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    """只讀解析一個非 symlink 的 UTF-8 strict JSON object。

    檔案必須是既有的普通檔案；讀取使用 strict UTF-8，不以替代字元掩蓋損壞 bytes。
    JSON parser 會拒絕重複鍵、``NaN``、``Infinity`` 及其他非標準常數，解析後再檢查
    極大指數造成的無限浮點數。回傳的 mapping 僅代表磁碟 snapshot，呼叫端仍必須做
    exact binding 比對；本函式不寫檔、不 reconcile，也不讀取 forcing。
    """

    try:
        json_path = Path(path)
    except TypeError as exc:
        raise TypeError(f"{label} path 必須是 str 或 Path") from exc
    if json_path.is_symlink() or not json_path.is_file():
        raise ValueError(f"{label} 必須是既有、非 symlink 的普通檔案")
    try:
        raw = json_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} 無法讀取") from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_inventory_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} 必須是 UTF-8 JSON") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} JSON 無法通過 strict parser") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root 必須是 object")
    _assert_finite_inventory_value(payload, label=label)
    return payload


def _inventory_utc_datetime(value: Any, *, label: str) -> datetime:
    """解析 inventory 內明示 ``Z`` 的 UTC 時間字串。

    preflight 的時間欄位代表 forcing 的實際 UTC 支援，而不是本地時區顯示文字；因此
    只接受字串與 ``Z`` 尾碼，不讓 ``+08:00``、日期物件或其他可隱式轉換型別進入
    formal coverage 計算。回傳值只用於 gate 比較，不會被寫回輸入 manifest。
    """

    if type(value) is not str or not value.strip() or not value.endswith("Z"):
        raise ValueError(f"{label} 必須是以 Z 結尾的 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} 不是可解析的 UTC timestamp") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} 必須明示 UTC")
    return parsed


def _inventory_epoch_ns(value: Any, *, label: str) -> int:
    """把 inventory UTC 字串轉成整數 epoch nanoseconds，拒絕浮點近似。"""

    parsed = _inventory_utc_datetime(value, label=label)
    return int(parsed.timestamp() * 1_000_000_000)


def _strict_nonnegative_int(value: Any, *, label: str) -> int:
    """驗證 JSON 盤點欄位是原生、非負且不可由 bool 冒充的整數。"""

    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必須是非負整數")
    return value


def _strict_positive_int(value: Any, *, label: str) -> int:
    """驗證 JSON 盤點欄位是原生正整數。"""

    result = _strict_nonnegative_int(value, label=label)
    if result < 1:
        raise ValueError(f"{label} 必須是正整數")
    return result


def _strict_finite_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    """驗證 inventory 的浮點比例／間隔，保留有限值與物理下界。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必須是 JSON 數字")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{label} 必須是大於等於 {minimum} 的有限數值")
    return normalized


def _validate_inventory_root(
    payload: Mapping[str, Any], *, expected_config_hash: str, formal: bool
) -> None:
    """驗證 pilot/formal 共用的 inventory 根欄位與 strict 基本型別。

    root schema 由 preflight 的 ``PreflightReport.to_dict`` 固定為七個欄位。pilot 只在
    此層確認 hash、UTC 與 list/object 形狀；formal 再由下方 semantic gate 驗證完整產品
    topology 與時間支援，避免把「能解析 JSON」誤當成正式 forcing 可用。
    """

    if set(payload) != _INVENTORY_KEYS:
        missing = sorted(_INVENTORY_KEYS - set(payload))
        unknown = sorted(set(payload) - _INVENTORY_KEYS)
        raise ValueError(f"input inventory 根欄位不符：missing={missing}, unknown={unknown}")
    _inventory_utc_datetime(payload["created_at_utc"], label="input_inventory.created_at_utc")
    if type(payload["config_hash"]) is not str or payload["config_hash"] != expected_config_hash:
        raise ValueError("input_inventory.config_hash 必須 exact 等於 config.config_hash()")
    if formal:
        if payload["mode"] != "formal_release":
            raise ValueError("formal input inventory.mode 必須 exact 等於 formal_release")
        if payload["formal_ready"] is not True:
            raise ValueError("formal input inventory.formal_ready 必須 exact 等於 true")
    else:
        if type(payload["mode"]) is not str or not payload["mode"].strip():
            raise ValueError("input_inventory.mode 必須是非空字串")
        if type(payload["formal_ready"]) is not bool:
            raise TypeError("input_inventory.formal_ready 必須是 bool")
    for key in ("inventories", "time_axes", "findings"):
        records = payload[key]
        if not isinstance(records, list):
            raise TypeError(f"input_inventory.{key} 必須是 list")
        if any(not isinstance(item, dict) for item in records):
            raise TypeError(f"input_inventory.{key} 的每個元素必須是 object")


def _validated_pilot_inventory(
    input_inventory_path: str | Path, *, expected_config_hash: str
) -> dict[str, Any]:
    """只讀一次並驗證 pilot inventory，保留 pilot 對資料完整性的寬鬆邊界。

    strict reader 會拒絕 symlink、損壞 UTF-8、重複 key、非標準 JSON 常數與溢位數值；
    之後只套用 pilot 原本的 root/hash/shape gate，不把 ``formal_ready`` 或產品列的
    值升格成正式 coverage 證據。解析後 mapping 直接交給 run-control canonicalization，
    因而不會再次讀取可能已被替換的原始檔案。
    """

    payload = _read_strict_json_object(input_inventory_path, label="input inventory")
    _validate_inventory_root(
        payload,
        expected_config_hash=expected_config_hash,
        formal=False,
    )
    return payload


def _validate_declared_support_release(
    config: ProjectConfig,
    *,
    config_path: str | Path,
    formal: bool,
) -> None:
    """在 runtime 讀取 scenario 前驗證明示支援窗的 immutable release binding。

    新設定若明示 ``inputs.backtrack_support_days``，該整日數只能代表母體的要求，
    不能取代 artifact 的實際 evidence。因此必須由 config 內的 artifact index 參照
    找到同一份 release binding，再交給 input derivation 的既有 validator 比對 source
    config hash、所有 component hash、gap-safe 母體與 requested horizon。這個入口只
    讀小型 JSON metadata，不會載入 OCM/NWW 大型陣列；同時保留每個 runtime config
    自己的 input inventory exact ``config_hash`` gate，7 日與 30 日不能共用 inventory。
    舊 YAML 未明示新欄位時跳過這層新契約，讓既有 pilot／runtime 行為維持相容。
    """

    if "backtrack_support_days" not in config.inputs.model_fields_set:
        return
    extra = config.inputs.model_extra or {}
    artifact_index_ref = extra.get("derived_input_artifact_index")
    if type(artifact_index_ref) is not str or not artifact_index_ref.strip():
        raise ValueError(
            "明示 inputs.backtrack_support_days 的 runtime config 必須綁定 derived_input_artifact_index"
        )
    artifact_index_path = resolve_manifest_path(config_path, artifact_index_ref)
    if artifact_index_path.name != "artifact_index.json":
        raise ValueError("derived_input_artifact_index 必須指向 artifact_index.json")
    from .input_derivation import validate_release_config

    result = validate_release_config(
        config_path,
        input_directory=artifact_index_path.parent,
        formal=formal,
    )
    if result.get("valid") is not True:
        raw_errors = result.get("errors")
        errors = (
            [item for item in raw_errors if type(item) is str]
            if isinstance(raw_errors, list)
            else ["release_validator_invalid"]
        )
        raise ValueError("runtime release binding validation failed: " + "; ".join(errors))


def _formal_months(config: ProjectConfig) -> tuple[str, ...]:
    """依正式研究期展開固定順序的 24 個 ``YYYYMM``。

    本專案正式研究期固定為 2024 至 2025 年，且設定檔中的年份序列必須逐項
    exact 等於 ``[2024, 2025]``。不能接受重排、缺年、額外年份或跨年洞；這個
    gate 讓月份 inventory、完整逐時時間軸與 arrival 回溯視窗共享同一個研究期
    定義，而不會因設定檔的排序方式產生不同的正式資料範圍。
    """

    years = config.inputs.years
    if years != [2024, 2025]:
        raise ValueError(
            "formal config.inputs.years 必須 exact 等於 [2024, 2025]；"
            "不接受重排、缺年、額外年份或跨年洞"
        )
    return tuple(f"{year}{month:02d}" for year in years for month in range(1, 13))


def _formal_flow_domain_ids(config: ProjectConfig) -> frozenset[str]:
    """解析 formal 執行模式的全部 flow-domain ID，確保 expanded domain 不被遺漏。"""

    resolved = tuple(
        resolve_flow_domain_id(config, domain.analysis_region_id, formal=True)
        for domain in config.domains
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError("formal config resolved flow-domain ID 必須唯一")
    return frozenset(resolved)


def _formal_product_contract(
    config: ProjectConfig,
    product: str,
) -> tuple[int, frozenset[str], frozenset[str]]:
    """取得 OCM/NWW 的正式 schema major、status 與 cache kind 契約。"""

    if product == "ocm_native":
        contract = config.inputs.ocm_contract
        default_statuses = [contract.get("required_status", "ready")]
    elif product == "nww3_analysis":
        contract = config.inputs.nww_contract
        default_statuses = [contract.get("required_status", "ready")]
    else:
        raise ValueError(f"不支援的 formal inventory product：{product}")
    if not isinstance(contract, Mapping):
        raise ValueError(f"config.inputs.{product} contract 必須是 mapping")
    try:
        schema_major = int(contract["required_schema_major"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"config.inputs.{product} 缺少 required_schema_major") from exc
    statuses = contract.get("accepted_statuses", default_statuses)
    cache_kinds = contract.get("accepted_cache_kinds", [])
    if not isinstance(statuses, list) or not statuses or any(type(item) is not str for item in statuses):
        raise ValueError(f"config.inputs.{product}.accepted_statuses 契約錯誤")
    if not isinstance(cache_kinds, list) or any(type(item) is not str for item in cache_kinds):
        raise ValueError(f"config.inputs.{product}.accepted_cache_kinds 契約錯誤")
    return schema_major, frozenset(statuses), frozenset(cache_kinds)


def _schema_major_text(value: Any, *, label: str) -> int:
    """從 schema version 字串取得 major，拒絕空白或非數字版本。"""

    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} 必須是 schema version 字串")
    head = value.split(".", 1)[0]
    if not head.isdigit():
        raise ValueError(f"{label} major 必須是整數")
    return int(head)


def _formal_month_path_token(
    config: ProjectConfig,
    *,
    product: str,
    flow_id: str,
    month: str,
) -> str:
    """建立正式月份列唯一允許的環境根目錄 provenance token。

    inventory 不保存 SERVER 絕對路徑，而是保存與設定檔環境變數名稱綁定的邏輯
    位置。OCM 與 NWW3 使用不同 root token，但兩者都必須精確包含 resolved
    ``flow_domain_id``、``months`` 目錄及六位數 UTC 月份；集中建立字串可避免
    loader 與 gate 各自拼出不同的資料契約。呼叫端已先驗證 product、flow 與 month，
    因此這裡只負責選取對應 root 並回傳 expected token。
    """

    if product == "ocm_native":
        root_env = config.inputs.ocm_native_root_env
    elif product == "nww3_analysis":
        root_env = config.inputs.nww_analysis_root_env
    else:
        raise ValueError(f"不支援的 formal inventory product：{product}")
    return f"${root_env}/{flow_id}/months/{month}"


def _validate_formal_month_record(
    row: Mapping[str, Any],
    *,
    config: ProjectConfig,
    flow_ids: frozenset[str],
    months: frozenset[str],
    seen: set[tuple[str, str, str]],
    index: int,
) -> None:
    """逐列驗證 ``MonthInventory`` 固定欄位與 config product contract。"""

    label = f"input_inventory.inventories[{index}]"
    if set(row) != _FORMAL_INVENTORY_RECORD_KEYS:
        raise ValueError(f"{label} 欄位集合必須 exact 等於 MonthInventory")
    product = row["product"]
    if type(product) is not str or product not in {"ocm_native", "nww3_analysis"}:
        raise ValueError(f"{label}.product 不合法")
    flow_id = row["flow_domain_id"]
    if type(flow_id) is not str or flow_id not in flow_ids:
        raise ValueError(f"{label}.flow_domain_id 必須是 formal resolved flow domain")
    month = row["month"]
    if type(month) is not str or month not in months or len(month) != 6 or not month.isdigit():
        raise ValueError(f"{label}.month 必須是 config 年份內的 YYYYMM")
    key = (product, flow_id, month)
    if key in seen:
        raise ValueError(f"formal inventory topology 不可重複：{key}")
    seen.add(key)
    schema_major, statuses, cache_kinds = _formal_product_contract(config, product)
    if _schema_major_text(row["schema_version"], label=f"{label}.schema_version") != schema_major:
        raise ValueError(f"{label}.schema_version major 與 config contract 不一致")
    if type(row["status"]) is not str or row["status"] not in statuses:
        raise ValueError(f"{label}.status 不符合 config contract")
    if type(row["cache_kind"]) is not str or row["cache_kind"] not in cache_kinds:
        raise ValueError(f"{label}.cache_kind 不符合 config contract")
    if row["required_arrays_present"] is not True:
        raise ValueError(f"{label}.required_arrays_present 必須 exact 等於 true")
    time_count = _strict_positive_int(row["time_count"], label=f"{label}.time_count")
    start_ns = _inventory_epoch_ns(row["time_start_utc"], label=f"{label}.time_start_utc")
    end_ns = _inventory_epoch_ns(row["time_end_utc"], label=f"{label}.time_end_utc")
    if end_ns < start_ns:
        raise ValueError(f"{label}.time_start_utc 不得晚於 time_end_utc")
    if time_count < 1:
        raise ValueError(f"{label}.time_count 必須為正")
    _strict_finite_number(row["maximum_gap_seconds"], label=f"{label}.maximum_gap_seconds")
    expected_path_token = _formal_month_path_token(
        config,
        product=product,
        flow_id=flow_id,
        month=month,
    )
    if type(row["path_token"]) is not str or row["path_token"] != expected_path_token:
        raise ValueError(
            f"{label}.path_token 必須 exact 等於 {expected_path_token}；"
            "禁止絕對路徑、錯誤 root token、flow 或 month"
        )
    for field_name in (
        "source_missing_day_count",
        "source_timestamp_repair_file_count",
        "source_zero_kept_file_count",
        "source_skipped_overlap_time_step_count",
    ):
        value = row[field_name]
        if value is not None:
            _strict_nonnegative_int(value, label=f"{label}.{field_name}")


def _validate_formal_gap(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_step_ns: int,
    period_start_ns: int,
    period_end_ns: int,
    seen: set[tuple[str, str]],
) -> tuple[int, int, int]:
    """驗證單一 ``TimeGapInventory`` 並回傳 inclusive 端點與缺時數。

    preflight 的研究期起點缺口只有右側 ``after_utc``，終點缺口只有左側
    ``before_utc``，內部缺口則必須同時有兩側支撐。端點必須恰好相隔一個設定時距，
    不能只檢查大小關係，否則矛盾的 boundary evidence 仍可能通過正式 release gate。
    """

    if set(row) != _FORMAL_TIME_GAP_KEYS:
        raise ValueError(f"{label} 欄位集合必須 exact 等於 TimeGapInventory")
    start_ns = _inventory_epoch_ns(row["missing_start_utc"], label=f"{label}.missing_start_utc")
    end_ns = _inventory_epoch_ns(row["missing_end_utc"], label=f"{label}.missing_end_utc")
    if end_ns < start_ns or (end_ns - start_ns) % expected_step_ns != 0:
        raise ValueError(f"{label} missing UTC 區間與 expected timestep 不一致")
    count = _strict_positive_int(row["missing_step_count"], label=f"{label}.missing_step_count")
    if end_ns - start_ns != (count - 1) * expected_step_ns:
        raise ValueError(f"{label}.missing_step_count 與 inclusive UTC 區間不一致")
    if start_ns < period_start_ns or end_ns > period_end_ns:
        raise ValueError(f"{label} 必須完整位於 config years 研究期")
    pair = (str(row["missing_start_utc"]), str(row["missing_end_utc"]))
    if pair in seen:
        raise ValueError(f"formal time axis gap 不可重複：{pair}")
    seen.add(pair)
    before_value = row["before_utc"]
    after_value = row["after_utc"]
    before_ns = (
        None
        if before_value is None
        else _inventory_epoch_ns(before_value, label=f"{label}.before_utc")
    )
    after_ns = (
        None
        if after_value is None
        else _inventory_epoch_ns(after_value, label=f"{label}.after_utc")
    )
    at_period_start = start_ns == period_start_ns
    at_period_end = end_ns == period_end_ns
    if at_period_start and at_period_end:
        if before_ns is not None or after_ns is not None or row["gap_hours"] is not None:
            raise ValueError(f"{label} 全期 boundary gap 不得宣告外部支撐端點")
    elif at_period_start:
        if (
            before_ns is not None
            or after_ns != end_ns + expected_step_ns
            or row["gap_hours"] is not None
        ):
            raise ValueError(f"{label} 起點 boundary gap 只能有相鄰 after_utc")
    elif at_period_end:
        if (
            after_ns is not None
            or before_ns != start_ns - expected_step_ns
            or row["gap_hours"] is not None
        ):
            raise ValueError(f"{label} 終點 boundary gap 只能有相鄰 before_utc")
    else:
        if (
            before_ns != start_ns - expected_step_ns
            or after_ns != end_ns + expected_step_ns
        ):
            raise ValueError(f"{label} internal gap 必須有相鄰 before_utc 與 after_utc")
        gap_hours = _strict_finite_number(row["gap_hours"], label=f"{label}.gap_hours", minimum=0.0)
        if not math.isclose(
            gap_hours,
            (after_ns - before_ns) / 3_600_000_000_000,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label}.gap_hours 與 before/after 不一致")
    return start_ns, end_ns, count


def _formal_period_contract(
    config: ProjectConfig,
    *,
    expected_step_ns: int,
) -> tuple[int, int, int]:
    """由 config 年份與固定時距計算研究期邊界及應有時次數。

    計數逐一展開 config 宣告的曆月，與 preflight 產製參考軸的方式一致。正式產品要求
    每月長度可被固定時距整除；目前 2024–2025 逐時契約因此恰為 17,544 個時次。
    回傳起訖均為 inclusive UTC nanoseconds，供 boundary gap 與 arrival horizon 共用。
    """

    if expected_step_ns <= 0:
        raise ValueError("formal expected timestep 必須形成正整數 nanoseconds step")
    month_starts: list[tuple[int, int]] = []
    expected_count = 0
    for month_text in _formal_months(config):
        year = int(month_text[:4])
        month = int(month_text[4:])
        start = datetime(year, month, 1, tzinfo=UTC)
        next_month = (
            datetime(year + 1, 1, 1, tzinfo=UTC)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=UTC)
        )
        start_ns = int(start.timestamp() * 1_000_000_000)
        next_month_ns = int(next_month.timestamp() * 1_000_000_000)
        duration_ns = next_month_ns - start_ns
        month_count, remainder = divmod(duration_ns, expected_step_ns)
        if remainder != 0:
            raise ValueError("formal expected timestep 必須整除每個 config 曆月")
        month_starts.append((start_ns, month_count))
        expected_count += month_count
    period_start_ns = month_starts[0][0]
    last_start_ns, last_count = month_starts[-1]
    period_end_ns = last_start_ns + (last_count - 1) * expected_step_ns
    return period_start_ns, period_end_ns, expected_count


def _validate_formal_time_axis(
    row: Mapping[str, Any],
    *,
    config: ProjectConfig,
    flow_ids: frozenset[str],
    seen: set[tuple[str, str]],
    index: int,
) -> tuple[str, str, int, tuple[tuple[int, int], ...]]:
    """逐列驗證 ``TimeAxisInventory`` 與其缺口列的完整型別和計數關係。"""

    label = f"input_inventory.time_axes[{index}]"
    if set(row) != _FORMAL_TIME_AXIS_KEYS:
        raise ValueError(f"{label} 欄位集合必須 exact 等於 TimeAxisInventory")
    product = row["product"]
    flow_id = row["flow_domain_id"]
    if type(product) is not str or product not in {"ocm_native", "nww3_analysis"}:
        raise ValueError(f"{label}.product 不合法")
    if type(flow_id) is not str or flow_id not in flow_ids:
        raise ValueError(f"{label}.flow_domain_id 必須是 formal resolved flow domain")
    key = (product, flow_id)
    if key in seen:
        raise ValueError(f"formal time_axes topology 不可重複：{key}")
    seen.add(key)
    policy = config.inputs.time_axis_contract.get("canonicalization_policy")
    expected_hours_value = config.inputs.time_axis_contract.get("expected_timestep_hours")
    if type(row["policy"]) is not str or row["policy"] != policy:
        raise ValueError(f"{label}.policy 與 config 不一致")
    expected_hours = _strict_finite_number(
        row["expected_timestep_hours"], label=f"{label}.expected_timestep_hours", minimum=0.0
    )
    if expected_hours != float(expected_hours_value):
        raise ValueError(f"{label}.expected_timestep_hours 與 config 不一致")
    expected_step_ns = int(round(expected_hours * 3_600_000_000_000))
    if expected_step_ns <= 0:
        raise ValueError(f"{label}.expected_timestep_hours 必須形成正整數 nanoseconds step")
    period_start_ns, period_end_ns, expected_period_count = _formal_period_contract(
        config,
        expected_step_ns=expected_step_ns,
    )
    count_fields = (
        "input_time_count",
        "canonical_time_count",
        "reordered_time_step_count",
        "dropped_duplicate_time_step_count",
        "expected_period_time_count",
        "available_period_time_count",
        "missing_period_time_count",
        "extra_halo_time_count",
    )
    counts = {
        field_name: _strict_nonnegative_int(row[field_name], label=f"{label}.{field_name}")
        for field_name in count_fields
    }
    if counts["expected_period_time_count"] != expected_period_count:
        raise ValueError(f"{label}.expected_period_time_count 與 config 研究期不一致")
    if (
        counts["input_time_count"] - counts["dropped_duplicate_time_step_count"]
        != counts["canonical_time_count"]
    ):
        raise ValueError(f"{label}.input/dropped/canonical time count 不一致")
    if counts["canonical_time_count"] != (
        counts["available_period_time_count"] + counts["extra_halo_time_count"]
    ):
        raise ValueError(f"{label}.canonical/available/extra-halo time count 不一致")
    if counts["available_period_time_count"] > counts["expected_period_time_count"]:
        raise ValueError(f"{label}.available_period_time_count 不得大於 expected_period_time_count")
    if counts["missing_period_time_count"] != (
        counts["expected_period_time_count"] - counts["available_period_time_count"]
    ):
        raise ValueError(f"{label}.missing_period_time_count 與 coverage count 不一致")
    coverage = _strict_finite_number(row["coverage_fraction"], label=f"{label}.coverage_fraction")
    if coverage > 1.0 or counts["expected_period_time_count"] == 0:
        raise ValueError(f"{label}.coverage_fraction/expected_period_time_count 不合法")
    expected_coverage = (
        counts["available_period_time_count"] / counts["expected_period_time_count"]
    )
    if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}.coverage_fraction 與 time count 不一致")
    start_ns = _inventory_epoch_ns(row["time_start_utc"], label=f"{label}.time_start_utc")
    end_ns = _inventory_epoch_ns(row["time_end_utc"], label=f"{label}.time_end_utc")
    if end_ns < start_ns:
        raise ValueError(f"{label}.time_start_utc 不得晚於 time_end_utc")
    _strict_finite_number(
        row["maximum_internal_gap_hours"],
        label=f"{label}.maximum_internal_gap_hours",
    )
    gaps = row["gaps"]
    if not isinstance(gaps, list):
        raise ValueError(f"{label}.gaps 必須是 list")
    gap_seen: set[tuple[str, str]] = set()
    validated_gaps: list[tuple[int, int, int]] = []
    for gap_index, gap in enumerate(gaps):
        if not isinstance(gap, dict):
            raise ValueError(f"{label}.gaps[{gap_index}] 必須是 object")
        validated_gaps.append(
            _validate_formal_gap(
                gap,
                label=f"{label}.gaps[{gap_index}]",
                expected_step_ns=expected_step_ns,
                period_start_ns=period_start_ns,
                period_end_ns=period_end_ns,
                seen=gap_seen,
            )
        )
    intervals = [(start_ns, end_ns) for start_ns, end_ns, _ in validated_gaps]
    if any(intervals[index][1] >= intervals[index + 1][0] for index in range(len(intervals) - 1)):
        raise ValueError(f"{label}.gaps 不得重疊或未排序")
    gap_missing_count = sum(count for _, _, count in validated_gaps)
    if counts["missing_period_time_count"] != gap_missing_count:
        raise ValueError(f"{label}.missing_period_time_count 與 gaps 明細不一致")
    if (
        product == "nww3_analysis"
        and (counts["missing_period_time_count"] != 0 or coverage != 1.0 or intervals)
    ):
        raise ValueError("formal NWW time axis 必須完整、coverage=1 且無 gaps")
    return product, flow_id, expected_step_ns, tuple(intervals)


def _validate_formal_ocm_gap_support(
    config: ProjectConfig,
    scenario_inputs: ScenarioInputs,
    *,
    ocm_axes: Sequence[tuple[str, str, int, tuple[tuple[int, int], ...]]],
) -> None:
    """驗證 OCM 全覆蓋、逐 arrival gap-safe 或明示截尾母體。

    legacy／一般 gap-safe 每個 arrival 的合法支援窗是
    ``[arrival - max_backtrack_days, arrival]``，兩端都算入檢查。明示
    ``observed_gap_censored_stop_at_first_gap_v1`` 時，長窗與缺口相交是可預期的
    data-gap exposure，不阻擋啟動；本 gate 仍要求 deposition exact-hour 在研究期內、
    不位於 gap、manifest reference 存在且 ``stop_at_data_gap=true``。這裡只比較
    inventory UTC 摘要與 arrival metadata，不讀 OCM 大型陣列，也不執行缺口重建。
    """

    # 這個政策只改變「長窗與缺口相交」的 gate；起點 exact-hour／finite／不在 gap
    # 仍必須逐筆通過。常數與 legacy config 不會讀取此分支，維持既有 full-window
    # gap-safe 語意。
    from .input_derivation import _gap_censoring_enabled

    gap_censoring_enabled = _gap_censoring_enabled(config)
    site_flow_ids: dict[str, str] = {}
    for site in config.study_sites:
        site_id = site.study_site_id
        if type(site_id) is not str or not site_id.strip() or site_id in site_flow_ids:
            raise ValueError("formal config study_site_id 必須是唯一非空字串")
        site_flow_ids[site_id] = resolve_flow_domain_id(
            config,
            site.analysis_region_id,
            formal=True,
        )

    gaps_by_flow: dict[str, tuple[tuple[int, int], ...]] = {}
    step_by_flow: dict[str, int] = {}
    for _, flow_id, expected_step_ns, axis_gaps in ocm_axes:
        if flow_id in gaps_by_flow:
            raise ValueError(f"formal OCM time axis 不可重複 flow-domain：{flow_id}")
        gaps_by_flow[flow_id] = axis_gaps
        step_by_flow[flow_id] = expected_step_ns
    for site_id, flow_id in site_flow_ids.items():
        if flow_id not in gaps_by_flow:
            raise ValueError(f"formal site {site_id} 缺少所屬 OCM time axis")

    arrival_flows: list[tuple[Any, str]] = []
    for arrival in scenario_inputs.arrival_times:
        flow_id = site_flow_ids.get(arrival.study_site_id)
        if flow_id is None:
            raise ValueError(f"arrival {arrival.arrival_time_id} 的 study_site_id 未登錄")
        arrival_flows.append((arrival, flow_id))
    has_residual_gaps = any(gaps_by_flow.values())
    if gap_censoring_enabled:
        safe_manifest = config.inputs.ocm_gap_safe_arrival_manifest
        if type(safe_manifest) is not str or not safe_manifest.strip():
            raise ValueError(
                "gap-censored formal runtime 必須明示 ocm_gap_safe_arrival_manifest"
            )
        if config.boundaries.stop_at_data_gap is not True:
            raise ValueError("gap-censored formal runtime 必須設定 stop_at_data_gap=true")
        # input manifest 的完整 hash／path closure 由
        # _validate_declared_support_release 驗證；這裡再確認其欄位確實存在，避免只
        # 依 runtime 開關便放行一份未登錄的截尾政策。
        if not isinstance(safe_manifest, str) or Path(safe_manifest).is_absolute():
            raise ValueError("ocm_gap_safe_arrival_manifest 必須是相對 manifest reference")
        for arrival, flow_id in arrival_flows:
            expected_step_ns = step_by_flow[flow_id]
            period_start_ns, period_end_ns, _ = _formal_period_contract(
                config,
                expected_step_ns=expected_step_ns,
            )
            arrival_ns = arrival.time_utc_ns
            if (arrival_ns - period_start_ns) % expected_step_ns != 0:
                raise ValueError(
                    f"gap-censored deposition 起點非 exact UTC hour：{arrival.arrival_time_id}"
                )
            if arrival_ns < period_start_ns or arrival_ns > period_end_ns:
                raise ValueError(
                    f"gap-censored deposition 起點超出 config years：{arrival.arrival_time_id}"
                )
            if any(gap_start <= arrival_ns <= gap_end for gap_start, gap_end in gaps_by_flow[flow_id]):
                raise ValueError(
                    f"gap-censored deposition 起點落在 OCM gap：{arrival.arrival_time_id}"
                )
            metadata = getattr(arrival, "metadata", None)
            if not isinstance(metadata, Mapping):
                raise ValueError(f"gap-censored arrival 缺少 metadata：{arrival.arrival_time_id}")
            observation_ns = metadata.get("observation_time_utc_ns")
            if observation_ns is not None and (
                isinstance(observation_ns, bool) or not isinstance(observation_ns, int)
            ):
                raise ValueError(
                    f"gap-censored observation anchor UTC 型別無效：{arrival.arrival_time_id}"
                )
        return
    support_declared = "backtrack_support_days" in config.inputs.model_fields_set
    if not has_residual_gaps and not support_declared:
        # 舊設定沒有獨立的母體支援窗；維持原本「全覆蓋時不再檢查 gap-safe window」
        # 的行為，避免新契約無意間改變既有 pilot／runtime gate。明示新欄位後，即使
        # inventory 沒有 gap，也必須檢查 requested window 是否落在研究期內。
        return
    safe_manifest = config.inputs.ocm_gap_safe_arrival_manifest
    if has_residual_gaps and not safe_manifest:
        raise ValueError(
            "OCM inventory 仍有 gaps；僅宣告 reconstruction manifest 或未宣告 gap-safe "
            "arrival manifest，formal runtime 必須拒絕"
        )
    _, horizon_ns = _backtrack_horizon(config.boundaries.max_backtrack_days)
    if not arrival_flows:
        raise ValueError("formal gap-safe inventory 必須有 arrival records")
    for arrival, flow_id in arrival_flows:
        period_start_ns, period_end_ns, _ = _formal_period_contract(
            config,
            expected_step_ns=step_by_flow[flow_id],
        )
        arrival_ns = arrival.time_utc_ns
        window_start = arrival_ns - horizon_ns
        window_end = arrival_ns
        if window_start < period_start_ns or window_end > period_end_ns:
            raise ValueError(
                f"arrival {arrival.arrival_time_id} 的 gap-safe window 超出 config years 研究期"
            )
        if not has_residual_gaps:
            continue
        for gap_start, gap_end in gaps_by_flow[flow_id]:
            if max(window_start, gap_start) <= min(window_end, gap_end):
                raise ValueError(
                    f"arrival {arrival.arrival_time_id} 的 gap-safe window 與 OCM gap 相交"
                )


def _validated_formal_inventory(
    input_inventory_path: str | Path,
    *,
    expected_config_hash: str,
    config: ProjectConfig,
    scenario_inputs: ScenarioInputs,
) -> dict[str, Any]:
    """以共用 strict parser 載入並完成 formal inventory semantic gate。"""

    payload = _read_strict_json_object(input_inventory_path, label="formal input inventory")
    _validate_inventory_root(
        payload,
        expected_config_hash=expected_config_hash,
        formal=True,
    )
    flow_ids = _formal_flow_domain_ids(config)
    months = frozenset(_formal_months(config))
    expected_inventory_count = len(flow_ids) * len(months) * 2
    if len(payload["inventories"]) != expected_inventory_count:
        raise ValueError(
            "formal inventories 必須 exact 涵蓋 resolved flow-domain × config years × two products"
        )
    seen_inventory: set[tuple[str, str, str]] = set()
    for index, row in enumerate(payload["inventories"]):
        _validate_formal_month_record(
            row,
            config=config,
            flow_ids=flow_ids,
            months=months,
            seen=seen_inventory,
            index=index,
        )
    expected_inventory_keys = {
        (product, flow_id, month)
        for product in ("ocm_native", "nww3_analysis")
        for flow_id in flow_ids
        for month in months
    }
    if seen_inventory != expected_inventory_keys:
        missing = sorted(expected_inventory_keys - seen_inventory)
        extra = sorted(seen_inventory - expected_inventory_keys)
        raise ValueError(f"formal inventories topology 不完整：missing={missing}, extra={extra}")

    findings = payload["findings"]
    for index, row in enumerate(findings):
        label = f"input_inventory.findings[{index}]"
        if set(row) != _FORMAL_FINDING_KEYS:
            raise ValueError(f"{label} 欄位集合必須 exact 等於 Finding")
        for field_name in _FORMAL_FINDING_KEYS:
            if type(row[field_name]) is not str or not row[field_name].strip():
                raise ValueError(f"{label}.{field_name} 必須是非空字串")
        if row["severity"] not in {"info", "warning", "error"}:
            raise ValueError(f"{label}.severity 不合法")
        if row["severity"] == "error":
            raise ValueError("formal input inventory.formal_ready=true 不得含 error finding")

    expected_axis_count = len(flow_ids) * 2
    if len(payload["time_axes"]) != expected_axis_count:
        raise ValueError("formal time_axes 必須 exact 涵蓋每個 resolved flow-domain × two products")
    seen_axis: set[tuple[str, str]] = set()
    axis_records: list[tuple[str, str, int, tuple[tuple[int, int], ...]]] = []
    for index, row in enumerate(payload["time_axes"]):
        axis_records.append(
            _validate_formal_time_axis(
                row,
                config=config,
                flow_ids=flow_ids,
                seen=seen_axis,
                index=index,
            )
        )
    expected_axis_keys = {
        (product, flow_id)
        for product in ("ocm_native", "nww3_analysis")
        for flow_id in flow_ids
    }
    if seen_axis != expected_axis_keys:
        missing = sorted(expected_axis_keys - seen_axis)
        extra = sorted(seen_axis - expected_axis_keys)
        raise ValueError(f"formal time_axes topology 不完整：missing={missing}, extra={extra}")
    ocm_axes = [item for item in axis_records if item[0] == "ocm_native"]
    _validate_formal_ocm_gap_support(config, scenario_inputs, ocm_axes=ocm_axes)
    return payload


def _validated_pilot_execution_config(
    config: ProjectConfig,
) -> tuple[int, int, int, int, str, str, int | None]:
    """驗證 pilot run 必需的 seed、分片、checkpoint 與 backend 設定。

    這些欄位會影響 scenario ordering、seed table 與 worker 的執行邊界，因此不能由
    Pydantic 的寬鬆 coercion 或 run-control 的預設值補齊。``max_resident_forcing_months``
    雖然不寫入固定 run plan，仍是稍後 forcing runtime 必須遵守的 cache 上限，所以在
    workspace 發布前一併檢查；本函式不驗證或讀取任何 forcing 內容。
    """

    members_per_scenario = _positive_integer(
        config.scenarios.members_per_scenario,
        label="scenarios.members_per_scenario",
    )
    master_seed = _nonnegative_integer(config.scenarios.master_seed, label="scenarios.master_seed")
    shard_scenario_count = _positive_integer(
        config.execution.shard_scenario_count,
        label="execution.shard_scenario_count",
    )
    checkpoint_interval_sweeps = _positive_integer(
        config.execution.checkpoint_interval_sweeps,
        label="execution.checkpoint_interval_sweeps",
    )
    if config.scenarios.seed_policy != "sha256_v1_pcg64dxsm":
        raise ValueError("scenarios.seed_policy 必須 exact 等於 sha256_v1_pcg64dxsm")
    if config.execution.production_backend != "numpy_reference":
        raise ValueError("execution.production_backend 必須 exact 等於 numpy_reference")
    active_chunk_size = config.execution.active_chunk_size
    if active_chunk_size is not None:
        active_chunk_size = _positive_integer(
            active_chunk_size,
            label="execution.active_chunk_size",
        )
    _positive_integer(
        config.execution.max_resident_forcing_months,
        label="execution.max_resident_forcing_months",
    )
    return (
        members_per_scenario,
        master_seed,
        shard_scenario_count,
        checkpoint_interval_sweeps,
        config.scenarios.seed_policy,
        config.execution.production_backend,
        active_chunk_size,
    )


def _load_plan_ordered_scenarios(
    root: Path,
    plan: Mapping[str, Any],
    scenario_inputs: ScenarioInputs,
) -> tuple[Any, ...]:
    """將 loader 的 current scenarios 綁定到 immutable plan 的 execution order。

    ``ScenarioInputs`` loader 依 manifest 建立的是目前設定下的 deterministic cross-product；
    run plan 另外以 ``scenario_table.parquet`` 保存已發布的順序與每個 shard 的 hash。本函式
    先依既有 ``scenario_execution_sort_key`` 排序 current scenarios，再逐列以既有
    ``_scenario_hash`` 與完整 dataclass 欄位對照磁碟 scenario table；接著不重排 plan
    ``shards``，逐列驗證半開 range、scenario count、``M`` 倍 particle count、連續覆蓋與
    shard hash。scenario 的到達時間單位是 UTC 奈秒、沉降速度單位是公尺／秒，particle
    count 是每個 scenario 的 ensemble member 數量乘積；這裡不讀取 forcing、不補缺值，
    也不把 synthetic 測試資料轉成正式科學結果。
    """

    scenario_table_path = root / "scenario_table.parquet"
    if scenario_table_path.is_symlink() or not scenario_table_path.is_file():
        raise ValueError("scenario_table.parquet 缺失或不是普通檔案")
    try:
        table = pq.read_table(scenario_table_path)
    except Exception as exc:  # noqa: BLE001 - static binding 要統一拒絕壞 parquet
        raise ValueError("scenario_table.parquet 無法讀取") from exc
    if tuple(table.column_names) != _SCENARIO_COLUMNS:
        raise ValueError("scenario_table.parquet 欄位順序不符")
    try:
        plan_scenarios = tuple(
            _scenario_from_row(row, label=f"scenario_table[{index}]")
            for index, row in enumerate(table.to_pylist())
        )
    except Exception as exc:  # noqa: BLE001 - row schema 錯誤不能進入 runtime
        raise ValueError("scenario_table.parquet scenario row 不合法") from exc

    scenario_count = _positive_integer(plan["scenario_count"], label="run_plan.scenario_count")
    particle_count = _positive_integer(plan["particle_count"], label="run_plan.particle_count")
    members_per_scenario = _positive_integer(
        plan["members_per_scenario"],
        label="run_plan.members_per_scenario",
    )
    if len(plan_scenarios) != scenario_count:
        raise ValueError("scenario table row count 與 run plan scenario_count 不一致")
    if tuple(sorted(plan_scenarios, key=scenario_execution_sort_key)) != plan_scenarios:
        raise ValueError("scenario table 未依 scenario execution ordering policy 排序")

    try:
        ordered_current = tuple(
            sorted(scenario_inputs.scenarios, key=scenario_execution_sort_key)
        )
    except Exception as exc:  # noqa: BLE001 - current manifest ordering 必須 fail closed
        raise ValueError("current scenario execution ordering 不合法") from exc
    if len(ordered_current) != scenario_count:
        raise ValueError("current scenario 數量與 run plan scenario_count 不一致")
    if len({item.scenario_id for item in ordered_current}) != len(ordered_current):
        raise ValueError("current scenario_id 不可重複")

    # 逐 scenario 同時做既有 canonical hash 與 dataclass exact equality；hash 綁定保留
    # run-control 的唯一算法，欄位 equality 則讓 hash collision 或欄位遺漏不會掩蓋設定漂移。
    for index, (current, planned) in enumerate(
        zip(ordered_current, plan_scenarios, strict=True)
    ):
        if _scenario_hash((current,)) != _scenario_hash((planned,)):
            raise ValueError(f"current scenario hash 與 plan 不一致：{index}")
        if current != planned:
            raise ValueError(f"current scenario 欄位與 plan 不一致：{index}")

    shard_rows = plan["shards"]
    if not isinstance(shard_rows, list):
        raise ValueError("run plan shards 必須是 list")
    shard_count = _positive_integer(plan["shard_count"], label="run_plan.shard_count")
    if len(shard_rows) != shard_count:
        raise ValueError("run plan shard_count 與 shards 長度不一致")
    expected_start = 0
    total_particles = 0
    for index, row in enumerate(shard_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"run_plan.shards[{index}] 必須是 object")
        try:
            start = row["scenario_start_index"]
            stop = row["scenario_stop_index"]
            count = row["scenario_count"]
            particles = row["particle_count"]
            declared_hash = row["scenario_hash"]
        except KeyError as exc:
            raise ValueError(f"run_plan.shards[{index}] 缺少固定欄位") from exc
        if any(type(value) is not int for value in (start, stop, count, particles)):
            raise ValueError(f"run_plan.shards[{index}] range/count 必須是原生整數")
        if (
            start < 0
            or stop <= start
            or stop - start != count
            or start != expected_start
            or count < 1
        ):
            raise ValueError(f"run plan shard range 不連續：{index}")
        if particles != count * members_per_scenario:
            raise ValueError(f"run plan shard particle_count 不一致：{index}")
        if type(declared_hash) is not str or _SHA256_RE.fullmatch(declared_hash) is None:
            raise ValueError(f"run plan shard scenario_hash 不合法：{index}")
        current_subset = ordered_current[start:stop]
        planned_subset = plan_scenarios[start:stop]
        if len(current_subset) != count or len(planned_subset) != count:
            raise ValueError(f"run plan shard range 超出 scenario table：{index}")
        if _scenario_hash(current_subset) != declared_hash:
            raise ValueError(f"current shard scenario hash 與 plan 不一致：{index}")
        if _scenario_hash(planned_subset) != declared_hash:
            raise ValueError(f"scenario table shard hash 與 plan 不一致：{index}")
        expected_start = stop
        total_particles += particles
    if expected_start != scenario_count:
        raise ValueError("run plan shards 未完整覆蓋 scenario")
    if total_particles != particle_count or particle_count != scenario_count * members_per_scenario:
        raise ValueError("run plan particle_count 與 scenario/member binding 不一致")
    return ordered_current


def _select_current_plan_scenarios(
    plan: Mapping[str, Any],
    config: ProjectConfig,
    scenario_inputs: ScenarioInputs,
    *,
    run_kind: str,
) -> tuple[Any, ...]:
    """由完整 current manifest 重算 plan 選擇，將 legacy 2.0 視為 full。

    2.1 plan 的 pilot 分層與精確選擇繫結，均以設定中的 ``scenario_count`` 作完整來源數
    gate；full binding 則以 plan 自身的 source count 驗證，因為既有 synthetic／小型
    engineering fixture 可能刻意只提供一筆完整 fixture。無論版本或模式，實際 selected
    tuple 都由 ``apply_scenario_selection`` 重跑，不信任 scenario table 以外的舊列或
    caller 傳入的 in-memory plan。這個函式不讀 forcing、不修改 workspace，也不縮減
    materials、receptors、arrival 或 dynamic initial-condition records。
    """

    if "scenario_selection" not in plan:
        # schema 2.0 沒有 selection 欄位；依相容契約把當時的完整 scenario table 解讀為
        # full，並建立只存在於本次驗證流程的 implicit binding。
        binding = build_full_scenario_selection(scenario_inputs.scenarios)
        expected_source_count = len(scenario_inputs.scenarios)
    else:
        binding = plan["scenario_selection"]
        if not isinstance(binding, Mapping):
            raise ValueError("run plan scenario_selection 必須是 mapping")
        if binding["mode"] in {"pilot_stratified", "pilot_exact"}:
            if binding["mode"] == "pilot_exact":
                # 精確先導仍從正式五萬基礎情境出發，不以 run plan 的小樣本數代替母體。
                validate_baseline_coverage(scenario_inputs.scenarios)
            configured_source_count = _positive_integer(
                config.scenarios.scenario_count,
                label="config.scenarios.scenario_count",
            )
            if configured_source_count != binding["source_scenario_count"]:
                raise ValueError("scenario_selection source count 與 config.scenarios.scenario_count 不一致")
            expected_source_count = configured_source_count
        else:
            expected_source_count = binding["source_scenario_count"]
    return apply_scenario_selection(
        binding,
        scenario_inputs.scenarios,
        scenario_inputs.receptors,
        expected_source_count,
        run_kind,
    )


def load_validated_run_static_inputs(
    workspace: str | Path | RunWorkspace,
    *,
    config_path: str | Path,
    checkpoint_root: str | Path | None = None,
    require_complete: bool = False,
    expected_run_kind: str | None = None,
) -> ValidatedRunStaticInputs:
    """載入並驗證 controller／aggregate 共用的 run static inputs。

    ``workspace`` 只用來定位 immutable run；即使 caller 傳入 ``RunWorkspace``，仍會從磁碟
    重新讀取 ``run_plan.json``，不信任其中附帶的 in-memory plan。驗證順序固定為：先確認
    plan 的 ``run_kind``（只接受 pilot/formal）、expected kind 與已登錄 experiment case，
    再呼叫 read-only ``validate_run``；驗證失敗只把 validator 的 JSON-safe error codes
    傳入例外，不攜帶 workspace 或 checkpoint 絕對路徑。其後才載入已驗收的 ProjectConfig、
    component／dynamic initial-condition manifests、公尺制 geometry 與 pilot/formal
    inventory，並 exact 比對 config／component／geometry hash、normalized config、seed、
    ensemble M、shard、checkpoint cadence、scenario selection、scenario count 與 particle count。

    回傳的 ``ValidatedRunStaticInputs`` 是 frozen defensive snapshot：plan 的巢狀 mapping
    與 list 會變成唯讀 proxy／tuple，config 以 ``model_copy(deep=True)`` 隔離 loader 參照，
    ScenarioInputs 會先由完整 current manifest 重算 plan selection，再依既有 execution sort
    key 重建成 plan 順序；geometry 也會重建以隔離 foreign-domain mapping。所有時間仍是 UTC
    奈秒、z／幾何距離仍是公尺；缺值、dry/land
    狀態與時間 gap 不以零值或最近值補齊。這個 helper 不取得 run lock、不建立
    ``RuntimeRequestFactory``、不處理 OCM/NWW root 或 array、不修改 progress/reconcile/
    checkpoint；呼叫成功只代表 static binding 完成，不代表本機 synthetic fixture 是真實
    OCM／NWW 科學成果。
    """

    if type(require_complete) is not bool:
        raise TypeError("require_complete 必須是 bool")
    if expected_run_kind is not None and (
        type(expected_run_kind) is not str or expected_run_kind not in _RUNTIME_RUN_KINDS
    ):
        raise ValueError("expected_run_kind 只允許 None、pilot 或 formal")

    root = workspace.path if isinstance(workspace, RunWorkspace) else Path(workspace)
    # plan 必須從磁碟重新讀取；RunWorkspace.plan 可能是建立時的舊 in-memory view，不能
    # 讓它繞過 immutable 文件、scenario range 或 progress 的現場驗證。
    plan = load_run_plan(root)
    run_kind = plan["run_kind"]
    if type(run_kind) is not str or run_kind not in _RUNTIME_RUN_KINDS:
        raise ValueError("run plan run_kind 只允許 pilot 或 formal")
    if expected_run_kind is not None and run_kind != expected_run_kind:
        raise ValueError(
            f"run plan run_kind 必須 exact 等於 {expected_run_kind}，實際為 {run_kind}"
        )
    experiment_case_id = plan["experiment_case_id"]
    try:
        _experiment_case_spec(experiment_case_id)
    except ValueError as exc:
        raise ValueError(f"run plan experiment_case_id 未登錄：{experiment_case_id!r}") from exc

    validation = validate_run(
        root,
        require_complete=require_complete,
        checkpoint_root=checkpoint_root,
    )
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raw_errors = validation.get("errors") if isinstance(validation, Mapping) else None
        error_codes = (
            [error for error in raw_errors if type(error) is str]
            if isinstance(raw_errors, list)
            else []
        )
        if not error_codes:
            error_codes = ["validator_invalid"]
        raise ValueError("run workspace validation failed: " + "; ".join(error_codes))

    formal = run_kind == "formal"
    # 到此才讀取設定與 manifest；這些 loader 只讀已驗收 JSON／geometry 描述，不開啟
    # forcing。dynamic initial condition 是 OCM-derived 的 receptor×arrival actual z 與
    # face provenance，不能退回 receptor 模板深度或以缺值零填補。
    config = load_config(config_path, formal_release=formal)
    _validate_declared_support_release(config, config_path=config_path, formal=formal)
    (
        members_per_scenario,
        master_seed,
        shard_scenario_count,
        checkpoint_interval_sweeps,
        seed_policy,
        production_backend,
        active_chunk_size,
    ) = _validated_pilot_execution_config(config)
    del production_backend
    scenario_inputs = load_scenario_inputs(
        config,
        config_path=config_path,
        require_dynamic_initial_conditions=True,
        formal=formal,
    )
    geometries = load_boundary_geometries(config, config_path=config_path, formal=formal)
    component_hashes = _validated_canonical_hashes(
        scenario_inputs.canonical_component_hashes,
        expected_keys=_PILOT_COMPONENT_HASH_KEYS,
        label=f"{run_kind} scenario_inputs.canonical_component_hashes",
    )
    geometry_hashes = _validated_canonical_hashes(
        geometries.canonical_component_hashes,
        expected_keys=_PILOT_GEOMETRY_HASH_KEYS,
        label=f"{run_kind} geometries.canonical_component_hashes",
    )

    inventory_path = root / "input_inventory.json"
    if formal:
        _validated_formal_inventory(
            inventory_path,
            expected_config_hash=config.config_hash(),
            config=config,
            scenario_inputs=scenario_inputs,
        )
    else:
        _validated_pilot_inventory(
            inventory_path,
            expected_config_hash=config.config_hash(),
        )

    normalized_config = _read_strict_json_object(
        root / "normalized_config.json",
        label="normalized_config",
    )
    current_normalized_config = config.normalized_payload()
    current_config_hash = config.config_hash()
    if plan["config_hash"] != current_config_hash:
        raise ValueError("run plan config_hash 與目前 config 不一致")
    if plan["component_canonical_hashes"] != component_hashes:
        raise ValueError("run plan component_canonical_hashes 與目前 scenario 不一致")
    if plan["geometry_canonical_hashes"] != geometry_hashes:
        raise ValueError("run plan geometry_canonical_hashes 與目前 geometry 不一致")
    if plan["master_seed"] != master_seed:
        raise ValueError("run plan master_seed 與目前 config 不一致")
    if plan["seed_policy"] != seed_policy:
        raise ValueError("run plan seed_policy 與目前 config 不一致")
    if plan["members_per_scenario"] != members_per_scenario:
        raise ValueError("run plan members_per_scenario 與目前 config 不一致")
    if plan["shard_scenario_count"] != shard_scenario_count:
        raise ValueError("run plan shard_scenario_count 與目前 config 不一致")
    if plan["checkpoint_interval_sweeps"] != checkpoint_interval_sweeps:
        raise ValueError("run plan checkpoint_interval_sweeps 與目前 config 不一致")
    if plan["active_chunk_size"] != active_chunk_size:
        raise ValueError("run plan active_chunk_size 與目前 config 不一致")
    if normalized_config != current_normalized_config:
        raise ValueError("workspace normalized_config 與目前 config 不一致")

    # 先以完整 current manifest 重算 2.1 selection（或建立 2.0 的 implicit full binding），
    # 再把 selected tuple 交給既有 plan-order／shard/hash gate。這個順序確保 plan 不能只
    # 依賴自己保存的 selected scenario table，也不會遺失未被抽中的 materials、receptors、
    # arrival 或 dynamic initial-condition records。
    selected_scenarios = _select_current_plan_scenarios(
        plan,
        config,
        scenario_inputs,
        run_kind=run_kind,
    )
    selected_inputs = _defensive_scenario_inputs(
        scenario_inputs,
        scenarios=selected_scenarios,
    )
    # 將 selector 的 deterministic 順序轉成 immutable plan 的 execution 順序，再由公開
    # 容器進行第二層 defensive copy。controller 與 aggregate 因而共用同一個 index／shard
    # 解讀，而不會各自猜測輸入列順序。
    ordered_scenarios = _load_plan_ordered_scenarios(root, plan, selected_inputs)
    ordered_inputs = _defensive_scenario_inputs(
        scenario_inputs,
        scenarios=ordered_scenarios,
    )
    return ValidatedRunStaticInputs(
        plan=plan,
        config=config,
        scenario_inputs=ordered_inputs,
        geometries=geometries,
    )


def initialize_run(
    *,
    config_path: str | Path,
    input_inventory_path: str | Path,
    destination: str | Path,
    run_id: str,
    experiment_case_id: str,
    project_root: str | Path,
    run_kind: str,
    declared_git_commit: str | None = None,
    pilot_scenarios_per_stratum: int | None = None,
    pilot_study_site_id: str | None = None,
    pilot_arrival_id: str | None = None,
    pilot_material_id: str | None = None,
) -> RunWorkspace:
    """依 ``run_kind`` 初始化 pilot 或 formal run workspace。

    ``pilot`` 與 ``formal`` 共用同一套 scenario、geometry、seed、checkpoint 與
    immutable plan 發布流程，但 formal 會以 ``formal_release=True`` 載入設定、以
    formal coverage loader 讀取五站／四域 manifest，並對 preflight inventory 進行
    strict topology／時間支援／gap-safe gate。兩種模式都只讀 inventory 一次，不建立
    forcing manager、不讀取 OCM/NWW 大型陣列；所有失敗都發生在 workspace 原子發布前。
    ``run_kind`` 只接受 ``pilot`` 或 ``formal``，未知值不得降級成任何既有模式。
    ``pilot_scenarios_per_stratum`` 只供 pilot 工程 sanity／benchmark；指定時會先從
    完整 source scenarios 依站點×receptor vertical 分層選樣，formal 即使傳入數值也會在
    任何 manifest／forcing I/O 前拒絕。省略時 pilot 與 formal 都保存 full selection。
    三個 pilot_*_id 必須一起明示，且不可與分層 N 混用；其精確選擇在完整來源清單、
    幾何、輸入盤點及執行設定驗證後才執行。arrival_id 對應 arrival_time_id 識別碼，
    不接受 UTC 或列索引。選中站的全部受體原樣保留，三個識別碼只存入獨立版本的
    scenario_selection，不改 config 或校準繫結。正式／合成模式及不完整參數均拒絕。
    """

    if type(run_kind) is not str or run_kind not in _RUNTIME_RUN_KINDS:
        raise ValueError("run_kind 只允許 pilot 或 formal；未知模式禁止 fallback")
    if pilot_scenarios_per_stratum is not None and (
        type(pilot_scenarios_per_stratum) is not int or pilot_scenarios_per_stratum < 1
    ):
        raise ValueError("pilot_scenarios_per_stratum 必須是正整數或 None")
    formal = run_kind == "formal"
    if formal and pilot_scenarios_per_stratum is not None:
        raise ValueError("formal run 禁止 pilot_scenarios_per_stratum")
    exact_ids = (pilot_study_site_id, pilot_arrival_id, pilot_material_id)
    exact_requested = any(value is not None for value in exact_ids)
    if exact_requested:
        if formal:
            raise ValueError("formal run 禁止 pilot_exact")
        if pilot_scenarios_per_stratum is not None:
            raise ValueError("pilot_exact 不可與 pilot_scenarios_per_stratum 混用")
        if any(type(value) is not str or not value or value != value.strip() for value in exact_ids):
            raise ValueError("pilot_exact 必須明示完整且無首尾空白的三個識別碼")
    # experiment case 是 run identity 與 physics 分支的唯一 registry key；先在讀取 config
    # 與 manifest 前拒絕未知值，避免 caller 以任意字串觸發未定義的 fallback 路徑。
    _experiment_case_spec(experiment_case_id)
    config = load_config(config_path, formal_release=formal)
    _validate_declared_support_release(config, config_path=config_path, formal=formal)

    scenario_inputs = load_scenario_inputs(
        config,
        config_path=config_path,
        require_dynamic_initial_conditions=True,
        formal=formal,
    )
    geometries = load_boundary_geometries(config, config_path=config_path, formal=formal)
    component_hashes = _validated_canonical_hashes(
        scenario_inputs.canonical_component_hashes,
        expected_keys=_PILOT_COMPONENT_HASH_KEYS,
        label=f"{run_kind} scenario_inputs.canonical_component_hashes",
    )
    geometry_hashes = _validated_canonical_hashes(
        geometries.canonical_component_hashes,
        expected_keys=_PILOT_GEOMETRY_HASH_KEYS,
        label=f"{run_kind} geometries.canonical_component_hashes",
    )
    # 精確選擇延至下方完整輸入盤點及執行設定檢查後，既有完整／分層路徑維持原順序。
    if not exact_requested and pilot_scenarios_per_stratum is None:
        selected_scenarios = tuple(scenario_inputs.scenarios)
        scenario_selection = build_full_scenario_selection(selected_scenarios)
    elif not exact_requested:
        expected_source_count = _positive_integer(
            config.scenarios.scenario_count,
            label="config.scenarios.scenario_count",
        )
        selected_scenarios, scenario_selection = select_pilot_scenarios(
            scenario_inputs.scenarios,
            scenario_inputs.receptors,
            pilot_scenarios_per_stratum,
            expected_source_count,
        )
    config_hash = config.config_hash()
    inventory = (
        _validated_formal_inventory(
            input_inventory_path,
            expected_config_hash=config_hash,
            config=config,
            scenario_inputs=scenario_inputs,
        )
        if formal
        else _validated_pilot_inventory(
            input_inventory_path,
            expected_config_hash=config_hash,
        )
    )
    (
        members_per_scenario,
        master_seed,
        shard_scenario_count,
        checkpoint_interval_sweeps,
        seed_policy,
        production_backend,
        active_chunk_size,
    ) = _validated_pilot_execution_config(config)
    # production_backend 已在 strict gate 驗證；既有 initialize_run_workspace 的固定
    # plan schema 不保存這個欄位，避免在 Slice 3B2a 擴張自訂 plan 欄位。
    del production_backend

    if exact_requested:
        validate_baseline_coverage(scenario_inputs.scenarios)
        selected_scenarios, scenario_selection = select_exact_pilot_scenarios(
            scenario_inputs.scenarios,
            scenario_inputs.receptors,
            _positive_integer(config.scenarios.scenario_count, label="config.scenarios.scenario_count"),
            study_site_id=pilot_study_site_id,
            arrival_id=pilot_arrival_id,
            material_id=pilot_material_id,
            run_kind=run_kind,
        )

    provenance = collect_code_provenance(
        project_root,
        declared_git_commit=declared_git_commit,
        formal=formal,
    )
    return initialize_run_workspace(
        destination,
        run_id=run_id,
        scenarios=selected_scenarios,
        normalized_config=config.normalized_payload(),
        config_hash=config_hash,
        input_inventory_file=inventory,
        component_canonical_hashes=component_hashes,
        geometry_canonical_hashes=geometry_hashes,
        provenance=provenance,
        experiment_case_id=experiment_case_id,
        master_seed=master_seed,
        seed_policy=seed_policy,
        members_per_scenario=members_per_scenario,
        shard_scenario_count=shard_scenario_count,
        checkpoint_interval_sweeps=checkpoint_interval_sweeps,
        active_chunk_size=active_chunk_size,
        run_kind=run_kind,
        scenario_selection=scenario_selection,
    )


def initialize_pilot_run(
    *,
    config_path: str | Path,
    input_inventory_path: str | Path,
    destination: str | Path,
    run_id: str,
    experiment_case_id: str,
    project_root: str | Path,
    declared_git_commit: str | None = None,
    pilot_scenarios_per_stratum: int | None = None,
    pilot_study_site_id: str | None = None,
    pilot_arrival_id: str | None = None,
    pilot_material_id: str | None = None,
) -> RunWorkspace:
    """建立工程先導執行，可選分層 N 或單站／到達時間／材質的全部受體。

    三個精確識別碼須同時提供，與 pilot_scenarios_per_stratum 互斥；皆省略時維持
    舊完整選擇。驗證與拒絕條件同 initialize_run；不修改來源清單、設定或 seed。
    """

    return initialize_run(
        config_path=config_path,
        input_inventory_path=input_inventory_path,
        destination=destination,
        run_id=run_id,
        experiment_case_id=experiment_case_id,
        project_root=project_root,
        run_kind="pilot",
        declared_git_commit=declared_git_commit,
        pilot_scenarios_per_stratum=pilot_scenarios_per_stratum,
        pilot_study_site_id=pilot_study_site_id,
        pilot_arrival_id=pilot_arrival_id,
        pilot_material_id=pilot_material_id,
    )


def initialize_formal_run(
    *,
    config_path: str | Path,
    input_inventory_path: str | Path,
    destination: str | Path,
    run_id: str,
    experiment_case_id: str,
    project_root: str | Path,
    declared_git_commit: str | None = None,
) -> RunWorkspace:
    """建立 formal run workspace；設定、manifest、inventory 任一 gate 失敗即不發布。"""

    return initialize_run(
        config_path=config_path,
        input_inventory_path=input_inventory_path,
        destination=destination,
        run_id=run_id,
        experiment_case_id=experiment_case_id,
        project_root=project_root,
        run_kind="formal",
        declared_git_commit=declared_git_commit,
    )


class RuntimeRequestFactory:
    """建立 pilot/formal ``RunUnit`` 所需的 immutable identity 與 lazy forcing facade。

    一個 instance 僅供一個 process／worker 使用，且不是 thread-safe。建構期間只建立
    scenario、material、receptor、arrival、pair、site、幾何與設定索引；不建立
    ``ForcingWindowManager``、不讀取網格、不載入 OCM/NWW 月份，也不呼叫 velocity
    provider。只有一筆 ``RunUnit`` 通過完整的 dataclass identity 與 dynamic pair
    provenance 核對後，才會為其 flow domain 建立且重用唯一 manager。

    「來源」在此仍是條件式來源足跡的運算輸入，不是絕對來源機率或因果歸因。factory
    僅組合資料，不替缺失月份、缺失 pair、域外位置或 face provenance 不一致的資料
    做最近值、零值或其他 fallback。
    """

    def __init__(
        self,
        *,
        config: ProjectConfig,
        scenario_inputs: ScenarioInputs,
        geometries: BoundaryGeometryBundle,
        ocm_native_root: str | Path,
        nww_analysis_root: str | Path | None,
        experiment_case_id: str,
        run_kind: str = "pilot",
    ) -> None:
        """驗證 pilot/formal 建立條件並保存不會改寫的 lookup；不開啟 forcing 資料。

        ``run_kind`` 只接受 ``pilot`` 或 ``formal``。正式 inventory gate 已在 runtime
        initializer/open controller 完成；此 constructor 仍以相同的 immutable scenario、
        geometry、resolved flow ID 與 lazy root 邊界建立 factory，不提供 synthetic、
        production alias 或任何缺口／路徑 fallback。NWW root 只有在有限水深 Stokes 案例
        中才會驗證；三個 Smagorinsky case 使用獨立的 OCM-only diffusion facade，但因
        registry 將它們的 velocity 設為 include-Stokes，仍會驗證 NWW root；``no_stokes``
        則完全忽略傳入的 NWW 參數，連路徑是否存在都不查。
        """

        if type(run_kind) is not str or run_kind not in _RUNTIME_RUN_KINDS:
            raise ValueError("run_kind 只允許 pilot 或 formal；未知模式禁止 fallback")
        self._run_kind = run_kind
        case_spec = _experiment_case_spec(experiment_case_id)
        include_stokes = case_spec.include_stokes
        self._experiment_case_id = experiment_case_id
        self._experiment_case_spec = case_spec

        # root gate 只檢查容器本身；no-Stokes 分支刻意不把傳入的 NWW 物件轉成 Path，
        # 以保證錯誤路徑、缺失路徑或會觸發 __fspath__ 的 sentinel 都不會被探查。
        self._ocm_root = _validated_root(ocm_native_root, label="ocm_native_root")
        if include_stokes:
            self._nww_root = _validated_root(nww_analysis_root, label="nww_analysis_root")
        else:
            self._nww_root = None

        # 這些 mapping proxy 是 constructor 階段唯一建立的資料索引；manager/provider
        # cache 另以空 dictionary 保存，直到第一筆完整合法請求才會增加資源物件。
        self._scenarios_by_id = _unique_index(
            scenario_inputs.scenarios, key_name="scenario_id", label="scenarios"
        )
        self._materials_by_id = _unique_index(
            scenario_inputs.materials, key_name="material_id", label="materials"
        )
        self._receptors_by_id = _unique_index(
            scenario_inputs.receptors, key_name="receptor_id", label="receptors"
        )
        self._arrivals_by_id = _unique_index(
            scenario_inputs.arrival_times, key_name="arrival_time_id", label="arrival_times"
        )
        self._pairs_by_key = MappingProxyType(
            {
                key: value
                for key, value in scenario_inputs.initial_conditions_by_pair.items()
            }
        )
        if len(self._pairs_by_key) != len(scenario_inputs.initial_conditions_by_pair):
            raise ValueError("initial_conditions_by_pair 的 pair key 不可重複")

        self._sites_by_id = _unique_index(
            config.study_sites, key_name="study_site_id", label="study_sites"
        )
        self._flow_by_site = MappingProxyType(
            {
                site_id: resolve_flow_domain_id(
                    config, site.analysis_region_id, formal=run_kind == "formal"
                )
                for site_id, site in self._sites_by_id.items()
            }
        )

        geometry_values = geometries.geometries
        projection_values = geometries.projections
        self._geometries = MappingProxyType(dict(geometry_values))
        self._projections = MappingProxyType(dict(projection_values))

        # max resident months 是 manager 建立時必須固定的 cache 上限；其餘 scalar 也在
        # 此處先驗證並封存，讓後續 request 不會把 bool、None 或非有限值帶進引擎。
        self._max_resident_months = _positive_integer(
            config.execution.max_resident_forcing_months,
            label="execution.max_resident_forcing_months",
        )
        # OCM 插值和物理 scalar kernels 是兩個獨立、各自版本化的選項。manager 的布林值
        # 僅控制既有 OCM 垂向／水平／時間內插；physics token 則同時進入 EngineSettings
        # 與 finite-depth Stokes provider，讓 Python 控制層固定處理 stage、RNG、QC、事件
        # 與邊界。舊 YAML 的 default 由 ExecutionConfig 補入，不在此猜測或靜默 fallback。
        self._ocm_use_numba_kernel = (
            config.execution.ocm_interpolation_backend == OCM_INTERPOLATION_BACKEND_NUMBA_V1
        )
        self._physics_kernel_backend = config.execution.physics_kernel_backend
        self._dt_min_seconds = _finite_scalar(
            config.integration.dt_min_seconds,
            label="integration.dt_min_seconds",
            minimum=0.0,
            allow_zero=False,
        )
        self._dt_max_seconds = _finite_scalar(
            config.integration.dt_max_seconds,
            label="integration.dt_max_seconds",
            minimum=0.0,
            allow_zero=False,
        )
        if self._dt_max_seconds < self._dt_min_seconds:
            raise ValueError("integration.dt_max_seconds 必須大於或等於 dt_min_seconds")
        self._output_interval_seconds = _finite_scalar(
            config.integration.output_interval_seconds,
            label="integration.output_interval_seconds",
            minimum=0.0,
            allow_zero=False,
        )
        self._max_backtrack_days = _finite_scalar(
            config.boundaries.max_backtrack_days,
            label="boundaries.max_backtrack_days",
            minimum=0.0,
            allow_zero=False,
        )
        self._bed_residence_config = config.scenarios.bed_residence_time
        if self._bed_residence_config is not None:
            support_days = self._bed_residence_config.runtime_horizon_support_days
            if support_days is None:
                raise ValueError(
                    "bed_residence_time template 的 runtime_horizon_support_days 為 null，"
                    "不可直接建立 runtime"
                )
            if self._max_backtrack_days > support_days:
                raise ValueError(
                    "boundaries.max_backtrack_days 超出 bed_residence_time "
                    "runtime_horizon_support_days"
                )
        self._maximum_step_count = _positive_integer(
            config.boundaries.maximum_step_count,
            label="boundaries.maximum_step_count",
        )

        horizontal_diffusion = _required_mapping_value(
            config.physics, label="physics", key="horizontal_diffusion"
        )
        vertical_diffusion = _required_mapping_value(
            config.physics, label="physics", key="vertical_diffusion"
        )
        self._kz_m2ps = _finite_scalar(
            _required_mapping_value(
                vertical_diffusion,
                label="physics.vertical_diffusion",
                key="constant_kz_m2ps",
            ),
            label="physics.vertical_diffusion.constant_kz_m2ps",
            minimum=0.0,
            allow_zero=True,
        )
        if case_spec.diffusion_kind == "constant":
            # 常數案例維持 Slice 2B1 原本的 scalar 解析與 DiffusionCoefficients 數值路徑；
            # 因此既有 baseline 的 Kh/Kz bit-compatible 契約不會因新增 Smagorinsky registry
            # 而被改寫。常數案例不需要讀取或驗證 smagorinsky 子 mapping。
            self._kh_m2ps = _finite_scalar(
                _required_mapping_value(
                    horizontal_diffusion,
                    label="physics.horizontal_diffusion",
                    key="constant_kh_m2ps",
                ),
                label="physics.horizontal_diffusion.constant_kh_m2ps",
                minimum=0.0,
                allow_zero=True,
            )
            self._smagorinsky_settings: SmagorinskySettings | None = None
        else:
            # Smagorinsky case 的水平 K 不是 config 裡的常數；只在此分支解析其完整
            # sensitivity／floor／cap 契約，讓 constants 即使保留 null 的 placeholder 也能
            # 正常建立。Kz 已在上方由共同 vertical_diffusion scalar 驗證並注入 settings。
            self._kh_m2ps = None
            self._smagorinsky_settings = _build_smagorinsky_settings(
                horizontal_diffusion,
                coefficient_cs=case_spec.coefficient_cs,
                constant_kz_m2ps=self._kz_m2ps,
            )

        integration_extra = getattr(config.integration, "model_extra", None)
        if isinstance(integration_extra, Mapping) and "maximum_minimum_clamps" in integration_extra:
            self._maximum_minimum_clamps = _positive_integer(
                integration_extra["maximum_minimum_clamps"],
                label="integration.maximum_minimum_clamps",
            )
        else:
            self._maximum_minimum_clamps = 100

        self._include_stokes = include_stokes
        self._managers: dict[str, ForcingWindowManager] = {}
        self._providers: dict[tuple[str, float, bool], Any] = {}
        self._diffusion_providers: dict[
            tuple[str, SmagorinskySettings], Any
        ] = {}

    def __call__(self, unit: RunUnit) -> ReferenceParticleRequest:
        """將一個 ``RunUnit`` 轉成可交給 reference engine 的 request。

        驗證順序先由 scenario ID 找到 manifest record，再做 dataclass exact equality；
        其後才核對 material、receptor、arrival 與 dynamic pair 的跨表欄位。所有身分、
        site／region／時間／垂向層位／pilot flow 與 face provenance 都一致後，才建立
        manager、投影 receptor、定位網格面及取得 provider。這裡不 sample、不 preload，
        因而建立 request 不會載入月份 array 或消耗 forcing cache。
        """

        supplied_scenario = unit.scenario
        scenario_id = supplied_scenario.scenario_id
        manifest_scenario = self._scenarios_by_id.get(scenario_id)
        if manifest_scenario is None:
            raise ValueError(f"未知 scenario_id，禁止 fallback：{scenario_id!r}")
        if supplied_scenario != manifest_scenario:
            raise ValueError("RunUnit scenario 與 manifest scenario 必須 dataclass exact equality")
        scenario = manifest_scenario
        if unit.experiment_case_id != self._experiment_case_id:
            raise ValueError("RunUnit experiment_case_id 與 factory experiment_case_id 不一致")

        material = self._materials_by_id.get(scenario.material_id)
        if material is None:
            raise ValueError(f"scenario material_id 未在 manifest 中找到：{scenario.material_id}")
        if material.material_id != scenario.material_id:
            raise ValueError("material_id 與 scenario 不一致")
        if material.settling_velocity_mps != scenario.settling_velocity_mps:
            raise ValueError("material settling_velocity_mps 與 scenario 不一致")

        receptor = self._receptors_by_id.get(scenario.receptor_id)
        if receptor is None:
            raise ValueError(f"scenario receptor_id 未在 manifest 中找到：{scenario.receptor_id}")
        if receptor.receptor_id != scenario.receptor_id:
            raise ValueError("receptor_id 與 scenario 不一致")
        if receptor.study_site_id != scenario.study_site_id:
            raise ValueError("receptor study_site_id 與 scenario 不一致")
        if receptor.analysis_region_id != scenario.analysis_region_id:
            raise ValueError("receptor analysis_region_id 與 scenario 不一致")

        arrival = self._arrivals_by_id.get(scenario.arrival_time_id)
        if arrival is None:
            raise ValueError(f"scenario arrival_time_id 未在 manifest 中找到：{scenario.arrival_time_id}")
        if arrival.arrival_time_id != scenario.arrival_time_id:
            raise ValueError("arrival_time_id 與 scenario 不一致")
        if arrival.study_site_id != scenario.study_site_id:
            raise ValueError("arrival study_site_id 與 scenario 不一致")
        if arrival.time_utc_ns != scenario.arrival_time_utc_ns:
            raise ValueError("arrival time_utc_ns 與 scenario 不一致")

        pair_key = (scenario.receptor_id, scenario.arrival_time_id)
        pair = self._pairs_by_key.get(pair_key)
        if pair is None:
            raise ValueError(f"缺少 receptor×arrival dynamic pair：{pair_key}")
        if pair.receptor_id != receptor.receptor_id or pair.arrival_time_id != arrival.arrival_time_id:
            raise ValueError("dynamic pair id 與 receptor/arrival 不一致")
        if pair.study_site_id != scenario.study_site_id:
            raise ValueError("dynamic pair study_site_id 與 scenario 不一致")
        if pair.analysis_region_id != scenario.analysis_region_id:
            raise ValueError("dynamic pair analysis_region_id 與 scenario 不一致")
        if pair.time_utc_ns != scenario.arrival_time_utc_ns:
            raise ValueError("dynamic pair time_utc_ns 與 scenario 不一致")
        if pair.vertical_id != receptor.vertical_id:
            raise ValueError("dynamic pair vertical_id 與 receptor 不一致")

        site = self._sites_by_id.get(scenario.study_site_id)
        if site is None:
            raise ValueError(f"未知 study_site_id，禁止 fallback：{scenario.study_site_id}")
        if site.analysis_region_id != scenario.analysis_region_id:
            raise ValueError("site analysis_region_id 與 scenario 不一致")
        flow_id = self._flow_by_site.get(scenario.study_site_id)
        if flow_id is None:
            raise ValueError(
                f"study_site 缺少 resolved {self._run_kind} flow：{scenario.study_site_id}"
            )
        if pair.flow_domain_id != flow_id:
            raise ValueError(f"dynamic pair flow_domain_id 與 resolved {self._run_kind} flow 不一致")

        # 回溯上限與正式 gap-safe 共用精度有界的天／秒／奈秒轉換；只修正可往返的浮點
        # 表示，不偷改到達時間。先建立設定，讓不合法次奈秒或溢位在 forcing manager 前停止。
        bed_timing: BedResidenceTiming | None = None
        if self._bed_residence_config is not None:
            residence = self._bed_residence_config
            bed_timing = resolve_bed_residence_timing(
                arrival,
                backtrack_mode=residence.backtrack_mode,
                requested_horizon_days=self._max_backtrack_days,
                maximum_age_days=residence.maximum_age_days,
                sampling_seed=residence.sampling_seed,
            )
        settings = self._build_settings(
            scenario.arrival_time_utc_ns,
            bed_timing=bed_timing,
        )

        geometry = self._geometries.get(scenario.study_site_id)
        projection = self._projections.get(scenario.study_site_id)
        if geometry is None or projection is None:
            raise ValueError(f"study_site 缺少已驗證 geometry/projection：{scenario.study_site_id}")

        if bed_timing is not None and bed_timing.pre_window_deposition:
            # pre-window 不開 forcing manager，但 request 仍需帶有與正式 receptor 一致的
            # 公尺制起點。先完成幾何 gate，避免 terminal outcome 掩蓋無效 receptor。
            projected_x, projected_y = projection.project(receptor.lon, receptor.lat)
            x_m = float(projected_x)
            y_m = float(projected_y)
            point = Point(x_m, y_m)
            if not geometry.own_local_domain.covers(point) or not geometry.flow_domain.covers(point):
                raise ValueError("receptor 投影點必須同時被 own_local_domain 與 flow_domain covers")
            initial_state = ParticleState(
                particle_id=unit.particle_id,
                scenario_id=scenario.scenario_id,
                member_id=unit.member_id,
                study_site_id=scenario.study_site_id,
                analysis_region_id=scenario.analysis_region_id,
                receptor_id=scenario.receptor_id,
                x_m=x_m,
                y_m=y_m,
                z_m=pair.z_m_positive_up,
                time_utc_ns=scenario.arrival_time_utc_ns,
                status=ParticleStatus.PRE_WINDOW_DEPOSITION,
            )
            # 研究窗前已沉底的成員只保存合法 terminal request；不開 forcing manager、
            # 不定位 forcing mesh、不建立 velocity provider，也不建立空間擴散 provider。
            return ReferenceParticleRequest(
                initial_state=initial_state,
                velocity=_forbidden_pre_window_velocity,
                boundaries=geometry,
                behavior_class=material.behavior_class,
                diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
                settings=settings,
            )

        # ACTIVE legacy 路徑維持原有順序：先建立／取得 manager，再做投影、幾何與 mesh
        # face provenance 驗證。如此不改變既有 active request 的副作用與錯誤時序。
        manager = self._managers.get(flow_id)
        if manager is None:
            manager = ForcingWindowManager.from_roots(
                flow_domain_id=flow_id,
                projection=projection,
                ocm_root=self._ocm_root,
                nww_root=self._nww_root,
                max_resident_months=self._max_resident_months,
                use_numba_kernel=self._ocm_use_numba_kernel,
                physics_kernel_backend=self._physics_kernel_backend,
            )
            self._managers[flow_id] = manager

        projected_x, projected_y = projection.project(receptor.lon, receptor.lat)
        x_m = float(projected_x)
        y_m = float(projected_y)
        point = Point(x_m, y_m)
        if not geometry.own_local_domain.covers(point) or not geometry.flow_domain.covers(point):
            raise ValueError("receptor 投影點必須同時被 own_local_domain 與 flow_domain covers")

        location = manager.mesh.locate(x_m, y_m)
        if location is None:
            raise ValueError("receptor 投影點無法定位到 native mesh face")
        if location.source_face_local_index != pair.source_face_local_index:
            raise ValueError("mesh source_face_local_index 與 dynamic pair provenance 不一致")
        if location.source_face_global_index != pair.source_face_global_index:
            raise ValueError("mesh source_face_global_index 與 dynamic pair provenance 不一致")

        # key 固定包含 flow、scenario settling 與 Stokes 開關；同一 flow 的不同 member
        # 只共用輕量 velocity facade，不把任何 month array 預先掛在 provider 上。Smagorinsky
        # 的擴散 facade 另以獨立 cache 管理，避免把 velocity 的 settling／Stokes 組合誤當成
        # 空間擴散參數，也確保同 flow、同 settings 的 members 共用同一個 immutable facade。
        provider_key = (flow_id, scenario.settling_velocity_mps, self._include_stokes)
        provider = self._providers.get(provider_key)
        if provider is None:
            provider = manager.provider(scenario.settling_velocity_mps, self._include_stokes)
            self._providers[provider_key] = provider

        # identity 與時間只能從 RunUnit/scenario 取得；z 特別使用動態 pair 的實際 OCM
        # 初始深度，絕不回退到 receptor 的模板 z_m_positive_up。
        initial_state = ParticleState(
            particle_id=unit.particle_id,
            scenario_id=scenario.scenario_id,
            member_id=unit.member_id,
            study_site_id=scenario.study_site_id,
            analysis_region_id=scenario.analysis_region_id,
            receptor_id=scenario.receptor_id,
            x_m=x_m,
            y_m=y_m,
            z_m=pair.z_m_positive_up,
            time_utc_ns=scenario.arrival_time_utc_ns,
        )

        if self._experiment_case_spec.diffusion_kind == "constant":
            # 常數 reference 必須維持既有三方向係數與 validate 行為，讓 no-Stokes 與
            # finite-depth-stokes 的數值輸入不受新增敏感度案例影響。
            assert self._kh_m2ps is not None
            diffusion = DiffusionCoefficients(self._kh_m2ps, self._kh_m2ps, self._kz_m2ps)
            diffusion.validate()
        else:
            # Smagorinsky provider 只在 request 第一次實際需要時建立；factory method 不會
            # sample 或 preload，所以這裡雖然完成 provider wiring，仍不會載入任何月份。
            assert self._smagorinsky_settings is not None
            diffusion_key = (flow_id, self._smagorinsky_settings)
            diffusion = self._diffusion_providers.get(diffusion_key)
            if diffusion is None:
                diffusion = manager.smagorinsky_provider(self._smagorinsky_settings)
                self._diffusion_providers[diffusion_key] = diffusion
        return ReferenceParticleRequest(
            initial_state=initial_state,
            velocity=provider,
            boundaries=geometry,
            behavior_class=material.behavior_class,
            diffusion=diffusion,
            settings=settings,
        )

    def _build_settings(
        self,
        arrival_time_utc_ns: int,
        *,
        bed_timing: BedResidenceTiming | None = None,
    ) -> EngineSettings:
        """建立秒制引擎設定；沉底模式使用 resolver 的有效期間與 UTC 起點。

        legacy config 未啟用沉底時間時沿用舊 arrival 減 H 行為。新契約則直接使用已核對
        的整數奈秒 resolver 結果；pre-window request 仍以原本正 H 填入引擎設定，讓設定
        schema 保持有效，而 terminal initial state 會保證不執行任何積分步。
        """

        if bed_timing is None:
            max_backtrack_seconds, horizon_ns = _backtrack_horizon(
                self._max_backtrack_days
            )
            earliest_ns = arrival_time_utc_ns - horizon_ns
        else:
            earliest_ns = bed_timing.earliest_forcing_time_utc_ns
            if bed_timing.pre_window_deposition:
                max_backtrack_seconds, _ = _backtrack_horizon(
                    self._max_backtrack_days
                )
            else:
                max_backtrack_seconds = bed_timing.effective_horizon_seconds
        if not -(1 << 63) <= earliest_ns < (1 << 63):
            raise ValueError("earliest_forcing_time_utc_ns 超出有號 64 位整數範圍")
        return EngineSettings(
            dt_min_seconds=self._dt_min_seconds,
            dt_max_seconds=self._dt_max_seconds,
            output_interval_seconds=self._output_interval_seconds,
            max_backtrack_seconds=max_backtrack_seconds,
            maximum_step_count=self._maximum_step_count,
            earliest_forcing_time_utc_ns=earliest_ns,
            maximum_minimum_clamps=self._maximum_minimum_clamps,
            physics_kernel_backend=self._physics_kernel_backend,
        )

    def resource_stats(self) -> dict[str, int]:
        """回傳目前 manager cache 的扁平整數資源統計。

        ``loads``、``hits``、``misses`` 與 ``evictions`` 是目前 process 內各 manager
        的累計事件 counter；``manager_count`` 與 ``resident_bytes`` 是取樣當下的 gauge，
        不是整台 SERVER 的連續峰值。``resident_bytes`` 只使用既有 cache snapshot 的
        resident ndarray bytes，不把 path、月份名稱、provider 或巢狀 stats 物件暴露出去。
        controller 會在每個 shard invocation 開始時取 baseline，再以同一 baseline 計算
        counter 增量並對 gauge 取樣本最大值。即使底層 stats 使用 NumPy 整數，也在此轉
        成原生 Python ``int``，方便 manifest／監控序列化；這些欄位只作工程資源規劃，
        不代表物理結果或整機 I/O 峰值。
        """

        loads = 0
        hits = 0
        misses = 0
        evictions = 0
        resident_bytes = 0
        for manager in self._managers.values():
            stats = manager.cache_stats
            loads += int(stats.ocm_load_count) + int(stats.nww_load_count)
            if hasattr(stats, "cache_hit_count"):
                hits += int(stats.cache_hit_count)
            else:
                hits += int(stats.ocm_cache_hit_count) + int(stats.nww_cache_hit_count)
            if hasattr(stats, "cache_miss_count"):
                misses += int(stats.cache_miss_count)
            else:
                misses += int(stats.ocm_cache_miss_count) + int(stats.nww_cache_miss_count)
            evictions += int(stats.eviction_count)
            resident_bytes += int(stats.resident_ndarray_bytes)
        return {
            "manager_count": int(len(self._managers)),
            "loads": int(loads),
            "hits": int(hits),
            "misses": int(misses),
            "evictions": int(evictions),
            "resident_bytes": int(resident_bytes),
        }


def _open_run_controller(
    workspace: str | Path | RunWorkspace,
    *,
    config_path: str | Path,
    ocm_native_root: str | Path,
    nww_analysis_root: str | Path | None = None,
    resume: bool = False,
    checkpoint_root: str | Path | None = None,
    expected_run_kind: str | None = None,
) -> RunController:
    """依共用 static binding snapshot 開啟 pilot/formal controller；不啟動 shard。

    ``resume`` 仍在任何後續工作前嚴格驗證；其餘 plan、progress、config、scenario、
    dynamic initial condition、geometry、inventory、hash、順序與 shard binding 全部委派
    給 ``load_validated_run_static_inputs``。因此 RuntimeRequestFactory 只有在 static
    inputs 完整通過後才會接觸 OCM/NWW root；controller 仍只載入自身的 progress topology，
    不執行 shard、reconcile 或 forcing array。public open_* signature 與 resume、checkpoint
    行為維持相容。
    """

    if type(resume) is not bool:
        raise TypeError("resume 必須是 bool")

    static_inputs = load_validated_run_static_inputs(
        workspace,
        config_path=config_path,
        checkpoint_root=checkpoint_root,
        require_complete=False,
        expected_run_kind=expected_run_kind,
    )
    plan = static_inputs.plan
    config = static_inputs.config
    scenario_inputs = static_inputs.scenario_inputs
    geometries = static_inputs.geometries
    run_kind = plan["run_kind"]
    experiment_case_id = plan["experiment_case_id"]
    root = workspace.path if isinstance(workspace, RunWorkspace) else Path(workspace)

    # 所有 immutable binding 均已通過後才建立 factory；其 constructor 只建立 lookup，並
    # 不建立 ForcingWindowManager。controller 也只載入 plan/progress topology，不會自動
    # 執行 shard 或 reconcile，forcing 必須等 caller 明確執行後才由 request lazy 開啟。
    factory = RuntimeRequestFactory(
        config=config,
        scenario_inputs=scenario_inputs,
        geometries=geometries,
        ocm_native_root=ocm_native_root,
        nww_analysis_root=nww_analysis_root,
        experiment_case_id=experiment_case_id,
        run_kind=run_kind,
    )
    return RunController(
        root,
        request_factory=factory,
        resume=resume,
        checkpoint_root=checkpoint_root,
        resource_reporter=factory.resource_stats,
    )


def open_run_controller(
    workspace: str | Path | RunWorkspace,
    *,
    config_path: str | Path,
    ocm_native_root: str | Path,
    nww_analysis_root: str | Path | None = None,
    resume: bool = False,
    checkpoint_root: str | Path | None = None,
) -> RunController:
    """依 immutable run plan 自動開啟 pilot 或 formal ``RunController``。

    ``run_kind`` 不由 caller 另外指定，而是從已驗證的 run plan 讀取；因此 pilot 與
    formal 不能被 CLI 參數互換。函式只做 read-only validation、binding 與 lazy factory
    建立，不會自行執行 shard 或 reconcile。
    """

    return _open_run_controller(
        workspace,
        config_path=config_path,
        ocm_native_root=ocm_native_root,
        nww_analysis_root=nww_analysis_root,
        resume=resume,
        checkpoint_root=checkpoint_root,
    )


def open_pilot_run_controller(
    workspace: str | Path | RunWorkspace,
    *,
    config_path: str | Path,
    ocm_native_root: str | Path,
    nww_analysis_root: str | Path | None = None,
    resume: bool = False,
    checkpoint_root: str | Path | None = None,
) -> RunController:
    """相容既有 API，且只允許開啟 ``run_kind=pilot`` 的 workspace。"""

    return _open_run_controller(
        workspace,
        config_path=config_path,
        ocm_native_root=ocm_native_root,
        nww_analysis_root=nww_analysis_root,
        resume=resume,
        checkpoint_root=checkpoint_root,
        expected_run_kind="pilot",
    )


def open_formal_run_controller(
    workspace: str | Path | RunWorkspace,
    *,
    config_path: str | Path,
    ocm_native_root: str | Path,
    nww_analysis_root: str | Path | None = None,
    resume: bool = False,
    checkpoint_root: str | Path | None = None,
) -> RunController:
    """開啟 formal workspace，並拒絕 pilot 或未登錄 run kind。"""

    return _open_run_controller(
        workspace,
        config_path=config_path,
        ocm_native_root=ocm_native_root,
        nww_analysis_root=nww_analysis_root,
        resume=resume,
        checkpoint_root=checkpoint_root,
        expected_run_kind="formal",
    )


__all__ = [
    "EXPERIMENT_CASE_INCLUDE_STOKES",
    "EXPERIMENT_CASE_SPECS",
    "ExperimentCaseSpec",
    "RuntimeRequestFactory",
    "ValidatedRunStaticInputs",
    "initialize_run",
    "initialize_formal_run",
    "initialize_pilot_run",
    "load_validated_run_static_inputs",
    "open_run_controller",
    "open_formal_run_controller",
    "open_pilot_run_controller",
]
