"""報告代表軌跡的固定分層、成員閘門與可重建 identity 摘要。

本模組只處理報告層需要的純 Python identity 資料，不讀取檔案、不重新計算軌跡，也不
把抽出的代表軌跡解讀成絕對來源機率或因果歸因。核心分層固定為四季與兩種潮差代理；
成員有效性則沿用事件與 pathway 聚合的分母政策，資料缺口、數值失敗及固定日曆窗前已沉底的
成員不能進入有效報告
成員。完整 identity 包含 ``ParticleState`` 的六個原生欄位，以及 material、arrival、
season、tide 四個 report strata 欄位；它們與選樣 seed 及版本化 policy 一起寫入排序後的
緊湊 JSON，再以 UTF-8 計算 SHA-256，讓不同輸入順序不會改變優先序，同時讓任一 identity
欄位變動都能留下可稽核的摘要差異。

``RepresentativeTrajectory`` 只保存 immutable identity、完整 64 碼優先序摘要與至少
兩筆 ``Observation``。觀測序列會在建構時複製成 tuple，避免呼叫端之後修改原 list 影響
已選出的報告案例；它仍然保存原始觀測順序、環境 context、同點逐步速度與公尺、秒、UTC
奈秒欄位，不在此模組補值、裁切或改寫物理資料。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Final

from .engine import EnvironmentSampleStatus, Observation, ParticleResult
from .models import ParticleState, ParticleStatus, VelocitySampleStatus

__all__ = [
    "CORE_SEASONS",
    "CORE_TIDE_CLASSES",
    "REPRESENTATIVE_SELECTION_POLICY",
    "RepresentativeTrajectory",
    "is_valid_report_member",
    "representative_priority_digest",
]


# 四季與兩種潮差代理是報告核心 4×2 分層的固定順序；tuple 讓 caller 不能在執行期間
# 改寫分層順序，並使它可直接作為後續排序或驗證的 immutable contract。
CORE_SEASONS: Final[tuple[str, ...]] = ("DJF", "MAM", "JJA", "SON")
CORE_TIDE_CLASSES: Final[tuple[str, ...]] = ("spring_proxy", "neap_proxy")

# policy 名稱會進入優先序摘要的 canonical payload。未來若改變欄位、序列化或選樣邏輯，
# 必須建立新版本名稱，避免不同演算法產生相同 identity/seed 卻被誤認為可比較。
REPRESENTATIVE_SELECTION_POLICY: Final[str] = "stable_hash_core_season_tide_v1"

_MAX_UINT128: Final[int] = 2**128 - 1
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELD_NAMES: Final[tuple[str, ...]] = (
    "particle_id",
    "scenario_id",
    "member_id",
    "study_site_id",
    "analysis_region_id",
    "receptor_id",
    "material_id",
    "arrival_time_id",
    "season",
    "tide_class",
)


def _require_uint128(value: object, *, label: str) -> int:
    """驗證選樣 seed 是不含 ``bool``／NumPy scalar 的原生 128 位元無號整數。

    seed 對外仍是原生 Python 整數，但進入 canonical JSON 時會由 caller-visible 的合法
    整數轉成十進位文字，與 exact bootstrap 的跨語言 128 位元契約一致。``0`` 與
    ``2^128-1`` 都是合法邊界，超出範圍時立即失敗，避免不同實作各自截斷高位元。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if not 0 <= value <= _MAX_UINT128:
        raise ValueError(f"{label} 必須介於 0 與 2^128-1 之間")
    return value


def _require_identity_text(value: object, *, label: str) -> str:
    """驗證 identity 文字為原生、非空且沒有首尾空白的 ``str``。

    identity 會成為跨 shard、分層與報告選樣的 join key；不做 trim 或隱式轉型，可避免
    ``"site-a"`` 與 ``" site-a"`` 在不同上游處理器中被悄悄視為同一粒子。內部空白及
    非 ASCII Unicode 仍可保留，因為 canonical JSON 會以 UTF-8 保存其原始語意。
    """

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_member_id(value: object, *, label: str) -> int:
    """驗證系集成員編號為非負的原生 Python ``int``。

    member 編號是粒子 identity 的數值欄位，不是可四捨五入或可由 NumPy scalar 代替的
    近似數值；負值也沒有目前 run plan 的成員語意，因此在產生摘要前拒絕。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if value < 0:
        raise ValueError(f"{label} 必須是非負整數")
    return value


def _require_core_stratum(
    value: object,
    *,
    label: str,
    allowed: tuple[str, ...],
) -> str:
    """驗證 season／tide 是原生非空文字，且精確落在報告核心分層集合。

    這兩個欄位不是自由描述文字：season 必須是四季代碼，tide 必須是既定的
    spring/neap proxy。拒絕其他標籤可防止 event arrival 或未登錄潮況混入 F03/F08 的
    八個核心 strata；文字先經過共同 identity gate，因而不會以空白或可轉型物件繞過限制。
    """

    normalized = _require_identity_text(value, label=label)
    if normalized not in allowed:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"{label} 必須精確屬於固定集合：{allowed_values}")
    return normalized


def _validated_identity(
    *,
    particle_id: object,
    scenario_id: object,
    member_id: object,
    study_site_id: object,
    analysis_region_id: object,
    receptor_id: object,
    material_id: object,
    arrival_time_id: object,
    season: object,
    tide_class: object,
) -> dict[str, str | int]:
    """以固定欄位集合建立 canonical JSON 可用的 identity snapshot。

    前六欄分別是粒子、情境、系集成員、研究站點、分析區域及受體，正是既有
    ``ParticleState``／production identity 核對所使用的欄位；後四欄是 material、arrival、
    season、tide 的 report strata identity，補足 F03/F08/F09 不能從粒子狀態反推的分類。
    函式逐欄驗證並建立新 dict，不保留 caller 的 mapping alias，也不把缺少欄位的部分
    identity 靜默補成空字串。
    """

    return {
        "particle_id": _require_identity_text(particle_id, label="particle_id"),
        "scenario_id": _require_identity_text(scenario_id, label="scenario_id"),
        "member_id": _require_member_id(member_id, label="member_id"),
        "study_site_id": _require_identity_text(study_site_id, label="study_site_id"),
        "analysis_region_id": _require_identity_text(
            analysis_region_id,
            label="analysis_region_id",
        ),
        "receptor_id": _require_identity_text(receptor_id, label="receptor_id"),
        "material_id": _require_identity_text(material_id, label="material_id"),
        "arrival_time_id": _require_identity_text(
            arrival_time_id,
            label="arrival_time_id",
        ),
        "season": _require_core_stratum(
            season,
            label="season",
            allowed=CORE_SEASONS,
        ),
        "tide_class": _require_core_stratum(
            tide_class,
            label="tide_class",
            allowed=CORE_TIDE_CLASSES,
        ),
    }


def _require_sha256(value: object, *, label: str) -> str:
    """要求完整、全小寫的 64 碼 SHA-256 十六進位摘要。"""

    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是 64 碼小寫 SHA-256")
    return value


def _require_observation_time(value: object, *, label: str) -> int:
    """驗證觀測時間是原生 Python 整數，保留 UTC 奈秒的精確值。

    時間是逆向軌跡的排序軸；這裡拒絕 ``bool``、NumPy scalar 及其他可轉型物件，避免
    caller 在正式報告邊界以隱式轉換改變 UTC 奈秒。數值本身不限制正負，因為合法的
    UTC 時間可能位於 Unix epoch 之前；相鄰觀測的嚴格遞減關係由 snapshot 函式統一檢查。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 Python int，且不可是 bool 或 NumPy scalar")
    return value


def _require_observation_float(value: object, *, label: str) -> float:
    """驗證位置或回溯年齡是原生、有限的 Python ``float``。

    ``x_m``、``y_m``、``z_m`` 使用公尺，``age_seconds`` 使用秒；報告 representative
    snapshot 不接受整數、布林值、NumPy scalar、NaN 或無限值，避免繪圖與統計在不同
    backend 產生無法重現的座標或時間軸。這裡只做資料型別與有限性驗證，年齡的非負及
    相鄰順序由呼叫端依 backtracking 契約檢查。
    """

    if type(value) is not float:
        raise TypeError(f"{label} 必須是原生 Python float，且不可是 int、bool 或 NumPy scalar")
    if not math.isfinite(value):
        raise ValueError(f"{label} 必須是有限數值")
    return value


def _snapshot_observations(value: object, *, particle_id: str) -> tuple[Observation, ...]:
    """驗證並重建可供報告使用的 immutable 逆向觀測序列。

    ``ParticleResult.observations`` 在既有引擎中是可變 list；代表案例一旦進入 report
    product 就不應再受到 runtime 或 caller 追加、刪除資料的影響。因此這裡先消費 iterable
    成新 tuple，再逐項要求 exact ``Observation``，以原生 canonical 值重建新的 frozen
    ``Observation``；輸出 tuple 與輸入 list／observation object 都沒有 alias。

    觀測必須是完整的逆向 backtracking 序列：``time_utc_ns`` 以 UTC 奈秒嚴格遞減，
    ``age_seconds`` 以秒嚴格遞增，``x_m``／``y_m``／``z_m`` 是有限公尺座標且年齡不得為負。
    除最後一筆外每筆狀態都必須是 ``ACTIVE``，最後一筆必須是 ``ParticleStatus`` 的正式
    非 ``ACTIVE`` 終止狀態；因此中途 terminal、最後仍 active 或半途結果都不能被畫成
    代表軌跡。每筆的環境與速度 context 會原樣交給既有 ``Observation`` constructor
    再驗證並 canonicalize；這裡只保存已提供的 context，不補值、不把 ``NOT_SAMPLED``
    改成有效。正式 v2／v3 的 context completeness 由後續 trajectory stream gate 負責。

    至少兩筆資料才能表達一段可繪製的完整軌跡，而不是只有單一端點。
    """

    if isinstance(value, (str, bytes, bytearray, dict)):
        raise TypeError("observations 必須是 Observation 的 iterable")
    try:
        observations = tuple(iter(value))  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("observations 必須是 Observation 的 iterable") from error
    if len(observations) < 2:
        raise ValueError("observations 至少需要兩筆 Observation")

    canonical_observations: list[Observation] = []
    previous_time_utc_ns: int | None = None
    previous_age_seconds: float | None = None
    for index, observation in enumerate(observations):
        if type(observation) is not Observation:
            raise TypeError(f"observations[{index}] 必須是 exact Observation")

        observed_particle_id = _require_identity_text(
            observation.particle_id,
            label=f"observations[{index}].particle_id",
        )
        if observed_particle_id != particle_id:
            raise ValueError("所有 observation 的 particle_id 必須與代表軌跡 identity 一致")

        time_utc_ns = _require_observation_time(
            observation.time_utc_ns,
            label=f"observations[{index}].time_utc_ns",
        )
        age_seconds = _require_observation_float(
            observation.age_seconds,
            label=f"observations[{index}].age_seconds",
        )
        if age_seconds < 0.0:
            raise ValueError(f"observations[{index}].age_seconds 不得為負值")
        x_m = _require_observation_float(
            observation.x_m,
            label=f"observations[{index}].x_m",
        )
        y_m = _require_observation_float(
            observation.y_m,
            label=f"observations[{index}].y_m",
        )
        z_m = _require_observation_float(
            observation.z_m,
            label=f"observations[{index}].z_m",
        )

        if previous_time_utc_ns is not None and time_utc_ns >= previous_time_utc_ns:
            raise ValueError("逆向軌跡的 time_utc_ns 必須嚴格遞減")
        if previous_age_seconds is not None and age_seconds <= previous_age_seconds:
            raise ValueError("逆向軌跡的 age_seconds 必須嚴格遞增")

        status = observation.status
        if type(status) is not ParticleStatus:
            raise TypeError(f"observations[{index}].status 必須是 exact ParticleStatus")
        if index < len(observations) - 1:
            if status is not ParticleStatus.ACTIVE:
                raise ValueError("除最後一筆外，代表軌跡 observation.status 必須是 ACTIVE")
        elif status is ParticleStatus.ACTIVE:
            raise ValueError("代表軌跡最後一筆 observation.status 必須是正式終止狀態")

        environment_sample_status = observation.environment_sample_status
        if type(environment_sample_status) is not EnvironmentSampleStatus:
            raise TypeError(
                f"observations[{index}].environment_sample_status "
                "必須是 exact EnvironmentSampleStatus"
            )
        velocity_sample_status = observation.velocity_sample_status
        if type(velocity_sample_status) is not VelocitySampleStatus:
            raise TypeError(
                f"observations[{index}].velocity_sample_status "
                "必須是 exact VelocitySampleStatus"
            )

        # Observation constructor 是環境與速度 context 的既有唯一驗證入口；以已驗證的
        # 核心原生值重建新物件，同時讓 eta／bed／月份／品質旗標、九個公尺/秒速度欄位、
        # 狀態與缺值語意重新通過資料契約。這也確保不會把 caller 的 frozen object
        # reference 直接帶進正式 representative record。
        canonical_observations.append(
            Observation(
                particle_id=observed_particle_id,
                time_utc_ns=time_utc_ns,
                age_seconds=age_seconds,
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                status=status,
                environment_sample_status=environment_sample_status,
                eta_m=observation.eta_m,
                bed_z_m=observation.bed_z_m,
                forcing_month_id=observation.forcing_month_id,
                environment_qc_flags=observation.environment_qc_flags,
                velocity_sample_status=velocity_sample_status,
                total_u_mps=observation.total_u_mps,
                total_v_mps=observation.total_v_mps,
                total_w_mps=observation.total_w_mps,
                ocm_u_mps=observation.ocm_u_mps,
                ocm_v_mps=observation.ocm_v_mps,
                ocm_w_mps=observation.ocm_w_mps,
                stokes_u_mps=observation.stokes_u_mps,
                stokes_v_mps=observation.stokes_v_mps,
                settling_w_mps=observation.settling_w_mps,
                velocity_qc_flags=observation.velocity_qc_flags,
            )
        )
        previous_time_utc_ns = time_utc_ns
        previous_age_seconds = age_seconds

    return tuple(canonical_observations)


def is_valid_report_member(result: ParticleResult) -> bool:
    """依終止狀態判定一筆 ``ParticleResult`` 是否能進入報告有效成員分母。

    Args:
        result: 必須是 exact ``ParticleResult``，其 ``final_state`` 也必須是 exact
            ``ParticleState``。狀態代表整條逆向軌跡的終止原因，而非最後一筆觀測的
            顯示標籤。

    Returns:
        False 代表 DATA_GAP、NUMERICAL_FAILURE 或 PRE_WINDOW_DEPOSITION。資料／數值失敗
        無法支持有效 pathway；PRE_WINDOW_DEPOSITION 則表示研究窗內沒有漂流歷程。其餘
        非 ACTIVE 的正式終止狀態回傳 True。

    Raises:
        TypeError: 外層結果、final state 或 status 不是 exact 資料契約型別。
        ValueError: 軌跡仍為 ``ACTIVE``，表示 caller 在結果尚未終止時就嘗試產生報告。
    """

    if type(result) is not ParticleResult:
        raise TypeError("result 必須是 exact ParticleResult")
    if type(result.final_state) is not ParticleState:
        raise TypeError("result.final_state 必須是 exact ParticleState")
    status = result.final_state.status
    if type(status) is not ParticleStatus:
        raise TypeError("result.final_state.status 必須是 ParticleStatus")
    if status is ParticleStatus.ACTIVE:
        raise ValueError("ACTIVE ParticleResult 尚未終止，不可進入報告")
    return status not in {
        ParticleStatus.DATA_GAP,
        ParticleStatus.NUMERICAL_FAILURE,
        ParticleStatus.PRE_WINDOW_DEPOSITION,
    }


def representative_priority_digest(
    selection_seed: int,
    particle_id: str,
    scenario_id: str,
    member_id: int,
    study_site_id: str,
    analysis_region_id: str,
    receptor_id: str,
    material_id: str,
    arrival_time_id: str,
    season: str,
    tide_class: str,
) -> str:
    """計算代表軌跡的 deterministic SHA-256 最小優先序摘要。

    Args:
        selection_seed: 報告選樣專用的原生 Python ``int``，範圍為 ``0`` 到 ``2^128-1``。
            它與物理 run seed 分開保存；驗證後以十進位文字進入 canonical JSON，避免
            不同語言對 128 位元 JSON number 的解析差異。
        particle_id: 粒子唯一識別碼；必須是原生、非空且無首尾空白的文字。
        scenario_id: 基礎情境識別碼；必須是原生、非空且無首尾空白的文字。
        member_id: 情境內系集成員的非負原生 Python ``int``。
        study_site_id: 研究站點識別碼；必須是原生、非空且無首尾空白的文字。
        analysis_region_id: 分析區域識別碼；必須是原生、非空且無首尾空白的文字。
        receptor_id: 受體識別碼；必須是原生、非空且無首尾空白的文字。
        material_id: 材質識別碼；必須是原生、非空且無首尾空白的文字。
        arrival_time_id: 到達時刻識別碼；必須是原生、非空且無首尾空白的文字。
        season: 報告核心季節，必須精確屬於 ``CORE_SEASONS``。
        tide_class: 報告核心潮況，必須精確屬於 ``CORE_TIDE_CLASSES``。

    Returns:
        完整 64 碼小寫 SHA-256。摘要輸入是含 policy、十進位文字 seed 及上述十欄
        identity 的 canonical JSON：key 排序、無多餘空白、保留 Unicode 後以 UTF-8 編碼。

    Raises:
        TypeError: seed、member 或任一 identity 欄位不是要求的原生型別。
        ValueError: seed 超出 128 位元範圍、member 為負值，或文字欄位為空／含首尾空白。
    """

    normalized_seed = _require_uint128(selection_seed, label="selection_seed")
    identity = _validated_identity(
        particle_id=particle_id,
        scenario_id=scenario_id,
        member_id=member_id,
        study_site_id=study_site_id,
        analysis_region_id=analysis_region_id,
        receptor_id=receptor_id,
        material_id=material_id,
        arrival_time_id=arrival_time_id,
        season=season,
        tide_class=tide_class,
    )
    payload = {
        "identity": identity,
        "selection_policy": REPRESENTATIVE_SELECTION_POLICY,
        # 128 位元 seed 以十進位文字進入 JSON；這是跨語言 canonicalization 的固定
        # 契約，避免某些 JSON parser 將超過 64 位元的 number 轉成不精確浮點數。
        "selection_seed": str(normalized_seed),
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


@dataclass(frozen=True, slots=True)
class RepresentativeTrajectory:
    """報告層保留的一條 immutable 代表軌跡。

    ``particle_id``、``scenario_id``、``member_id``、``study_site_id``、
    ``analysis_region_id`` 與 ``receptor_id`` 是粒子本體 identity；``material_id``、
    ``arrival_time_id``、``season`` 與 ``tide_class`` 是不可省略的 report strata identity。
    後四欄使 F03/F08 的八層與 F09 的十種材質能從 record 本身重新核對，而不必從可能不
    完整的外部索引反推。``priority_digest`` 是由同一 identity 與選樣 seed/policy 產生的
    完整 64 碼摘要；本類別只驗證摘要格式，seed 不在此 record 重複保存。``observations``
    會從 caller iterable 複製成至少兩筆的 tuple，保留每筆 observation 的公尺制位置、秒制
    age、UTC 奈秒與環境 context；frozen dataclass 加上 immutable tuple 可避免建構後改寫
    欄位或透過原始 list alias 污染報告案例。
    """

    particle_id: str
    scenario_id: str
    member_id: int
    study_site_id: str
    analysis_region_id: str
    receptor_id: str
    material_id: str
    arrival_time_id: str
    season: str
    tide_class: str
    priority_digest: str
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        """在封存代表案例前完成 identity、digest 與 observation tuple 的 fail-fast 驗證。"""

        identity = _validated_identity(
            particle_id=self.particle_id,
            scenario_id=self.scenario_id,
            member_id=self.member_id,
            study_site_id=self.study_site_id,
            analysis_region_id=self.analysis_region_id,
            receptor_id=self.receptor_id,
            material_id=self.material_id,
            arrival_time_id=self.arrival_time_id,
            season=self.season,
            tide_class=self.tide_class,
        )
        observations = _snapshot_observations(
            self.observations,
            particle_id=identity["particle_id"],
        )
        digest = _require_sha256(self.priority_digest, label="priority_digest")
        for field_name in _IDENTITY_FIELD_NAMES:
            object.__setattr__(self, field_name, identity[field_name])
        object.__setattr__(self, "priority_digest", digest)
        object.__setattr__(self, "observations", observations)

    @property
    def digest(self) -> str:
        """提供簡短的摘要別名，仍回傳已驗證的完整優先序 SHA-256。"""

        return self.priority_digest
