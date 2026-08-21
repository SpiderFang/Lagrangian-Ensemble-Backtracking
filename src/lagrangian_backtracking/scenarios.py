"""建立十種行為、五站受體與到達時刻的完整情境清單。

情境識別碼只由站點、材料行為、受體、到達時刻及設計版本決定；實驗案例與隨機系集成員
是情境之外的維度。這可避免「不考慮波浪表面漂移」或不同成員數被誤算為計畫書的基礎
情境，也讓批次切分、工作程序數量與中途續跑不會改變亂數序列。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class Behavior:
    """尚未對應特定廢棄物材質時，用於敏感度分析的垂向移動類別。"""

    material_id: str
    settling_velocity_mps: float
    behavior_class: str


@dataclass(frozen=True, slots=True)
class Receptor:
    """指定站點的一個三維終端受體。

    ``z_m_positive_up`` 是該到達條件使用的實際深度，海面為零且向上為正。正式清單還應
    保存目標水深比例、距海床高度、深度調整原因與 OCM 網格面來源；額外欄位可放入
    ``metadata``。
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
    Behavior("sink_100mmps", -0.100, "sinking"),
    Behavior("sink_030mmps", -0.030, "sinking"),
    Behavior("sink_010mmps", -0.010, "sinking"),
    Behavior("sink_003mmps", -0.003, "sinking"),
    Behavior("sink_001mmps", -0.001, "sinking"),
    Behavior("neutral_000mmps", 0.000, "suspended"),
    Behavior("rise_001mmps", 0.001, "rising"),
    Behavior("rise_003mmps", 0.003, "rising"),
    Behavior("rise_010mmps", 0.010, "rising"),
    Behavior("rise_030mmps", 0.030, "rising"),
)


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
    if len({item.material_id for item in behaviors}) != len(behaviors):
        raise ValueError("material_id 必須唯一")
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
    """確認五站各 10,000 個、A 區 20,000 個、全案 50,000 個基礎情境。"""

    counts: dict[str, int] = {}
    for scenario in scenarios:
        counts[scenario.study_site_id] = counts.get(scenario.study_site_id, 0) + 1
    expected_sites = {"gongliao", "guishan", "hsinchu", "houwan", "lienchiang"}
    if set(counts) != expected_sites or any(value != 10_000 for value in counts.values()):
        raise ValueError(f"baseline coverage 不符：{counts}")
    if counts["gongliao"] + counts["guishan"] != 20_000 or len(scenarios) != 50_000:
        raise ValueError("A 區或全案 scenario count 不符")
    return counts


def derive_member_seed(*, master_seed: int, scenario_id: str, experiment_case_id: str, member_id: int) -> int:
    """產生不受工作程序、批次切分與中途續跑影響的 NumPy 128 位元亂數種子。

    不使用 Python 內建雜湊，因它在不同程序可能加入隨機值。主種子、情境、實驗案例與
    成員編號都輸入 SHA-256，取前 16 位元組轉為非負整數，可直接交給 NumPy 亂數產生器。
    """

    if master_seed < 0 or member_id < 0 or not scenario_id or not experiment_case_id:
        raise ValueError("seed 欄位必須非負且識別碼不可空白")
    fields = [str(master_seed), scenario_id, experiment_case_id, str(member_id)]
    encoded = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(sha256(encoded).digest()[:16], "big", signed=False)


def records_as_dicts(records: Iterable[Behavior | Receptor | ArrivalTime | Scenario]) -> list[dict]:
    """將資料類別轉為可寫入 JSON 或 Parquet 的字典清單，不建立效能較差的物件陣列。"""

    return [asdict(item) for item in records]
