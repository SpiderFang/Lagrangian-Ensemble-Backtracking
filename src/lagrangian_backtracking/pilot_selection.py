"""建立可重算的工程 pilot scenario 子集。

本模組只處理已由完整 manifest 解析出的 ``Scenario`` 與 ``Receptor`` 記錄，不讀取
forcing、server 路徑或任何軌跡結果。pilot 的目的只是工程 sanity／benchmark；正式
執行仍必須使用完整 scenario coverage。selector 的 binding 只保存版本、分層規則、
計數與 SHA-256 證據，不保存一長串 scenario ID，因此 static loader 可以在執行前由
目前完整 manifest 重算同一子集，再與 immutable run plan 做逐欄位比對。
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from typing import Any, Final

from .scenarios import Receptor, Scenario

PILOT_SCENARIO_SELECTION_SCHEMA_VERSION: Final[str] = "1.0.0"
"""scenario selection binding 的固定 schema 版本。"""

PILOT_SCENARIO_SELECTION_RANKING_POLICY: Final[str] = "pilot_site_vertical_sha256_rank_v1"
"""pilot 分層抽樣使用的版本化 deterministic ranking policy 識別碼。"""

PILOT_SCENARIO_SELECTION_POLICY: Final[str] = PILOT_SCENARIO_SELECTION_RANKING_POLICY
"""``PILOT_SCENARIO_SELECTION_RANKING_POLICY`` 的相容語意別名。"""

PILOT_SCENARIO_SELECTION_STRATUM_FIELDS: Final[tuple[str, str]] = (
    "study_site_id",
    "receptor.vertical_id",
)
"""pilot stratified selector 的 immutable 固定分層欄位順序。

公開常數刻意使用 tuple，避免外部呼叫端以 list mutation 改變 selector 的資料契約；
寫入 selection binding 時才轉成 JSON 相容的 list。
"""

_SELECTION_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "mode",
        "ranking_policy",
        "stratum_fields",
        "samples_per_stratum",
        "source_scenario_count",
        "selected_scenario_count",
        "source_scenario_ids_sha256",
        "selected_scenario_ids_sha256",
        "strata",
    }
)
_STRATUM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "study_site_id",
        "vertical_id",
        "source_count",
        "selected_count",
        "selected_scenario_ids_sha256",
    }
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_RUN_KINDS: Final[frozenset[str]] = frozenset({"formal", "pilot", "synthetic"})


def _strict_nonempty_text(value: object, *, label: str) -> str:
    """驗證識別碼是非空白的原生 Python 字串，避免以其他型別穿透 binding。"""

    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} 必須是非空白字串")
    return value


def _strict_positive_integer(value: object, *, label: str) -> int:
    """驗證計數是原生 Python 正整數，明確拒絕 ``bool`` 與浮點近似值。"""

    if type(value) is not int or value < 1:
        raise ValueError(f"{label} 必須是正整數")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    """驗證 binding 中的雜湊為 64 位小寫 SHA-256 文字。"""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是 64 位小寫 SHA-256")
    return value


def _length_prefixed_bytes(fields: Sequence[str]) -> bytes:
    """以欄位數與 UTF-8 byte 長度保留字串邊界，供 deterministic SHA-256 使用。

    直接串接字串會讓 ``["ab", "c"]`` 與 ``["a", "bc"]`` 產生同一輸入；每段先寫入
    八位元組長度即可避免這種欄位界線碰撞。長度與內容都在 bytes 層處理，因此中文
    識別碼與 ASCII 識別碼具有相同的明確規則。
    """

    encoded = bytearray()
    encoded.extend(len(fields).to_bytes(8, "big", signed=False))
    for index, value in enumerate(fields):
        text = _strict_nonempty_text(value, label=f"hash fields[{index}]")
        raw = text.encode("utf-8")
        encoded.extend(len(raw).to_bytes(8, "big", signed=False))
        encoded.extend(raw)
    return bytes(encoded)


def scenario_ids_sha256(scenario_ids: Iterable[str | Scenario]) -> str:
    """回傳 order-independent 的 scenario ID 集合 SHA-256。

    輸入可為 scenario ID 字串，或直接為 ``Scenario`` 物件；兩者都會先轉成唯一的
    scenario ID、排序，再以長度前綴 canonical bytes 計算 SHA-256。重複或空白 ID
    直接拒絕，因為 selection binding 代表的是 immutable scenario identity 集合，不能
    把重複列或 Python 內建 hash 的程序相依性帶入 run plan。
    """

    values: list[str] = []
    for index, item in enumerate(scenario_ids):
        value = item.scenario_id if isinstance(item, Scenario) else item
        values.append(_strict_nonempty_text(value, label=f"scenario_ids[{index}]"))
    if len(set(values)) != len(values):
        raise ValueError("scenario_id 不可重複")
    return sha256(_length_prefixed_bytes(tuple(sorted(values)))).hexdigest()


def canonical_scenario_ids_sha256(scenario_ids: Iterable[str | Scenario]) -> str:
    """``scenario_ids_sha256`` 的語意化公開別名，供外部 binding 檢查程式使用。"""

    return scenario_ids_sha256(scenario_ids)


def _validated_scenarios(
    scenarios: Sequence[Scenario], *, expected_count: int | None = None
) -> tuple[Scenario, ...]:
    """驗證 scenario tuple 的 identity 欄位與可重算數量。"""

    if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
        raise ValueError("scenarios 必須是 Scenario sequence")
    values = tuple(scenarios)
    if not values:
        raise ValueError("scenarios 不可為空")
    if expected_count is not None:
        _strict_positive_integer(expected_count, label="expected_source_scenario_count")
        if len(values) != expected_count:
            raise ValueError("完整 current scenarios 數量與 expected_source_scenario_count 不一致")
    seen: set[str] = set()
    for index, scenario in enumerate(values):
        if not isinstance(scenario, Scenario):
            raise ValueError(f"scenarios[{index}] 必須是 Scenario")
        _strict_nonempty_text(scenario.scenario_id, label=f"scenarios[{index}].scenario_id")
        _strict_nonempty_text(scenario.study_site_id, label=f"scenarios[{index}].study_site_id")
        _strict_nonempty_text(scenario.receptor_id, label=f"scenarios[{index}].receptor_id")
        if scenario.scenario_id in seen:
            raise ValueError(f"scenario_id 重複：{scenario.scenario_id}")
        seen.add(scenario.scenario_id)
    return values


def _validated_receptors(receptors: Sequence[Receptor]) -> tuple[Receptor, ...]:
    """驗證 receptor ID、站點與垂向識別碼，保留輸入順序以外的 deterministic 語意。"""

    if isinstance(receptors, (str, bytes)) or not isinstance(receptors, Sequence):
        raise ValueError("receptors 必須是 Receptor sequence")
    values = tuple(receptors)
    if not values:
        raise ValueError("receptors 不可為空")
    seen: set[str] = set()
    for index, receptor in enumerate(values):
        if not isinstance(receptor, Receptor):
            raise ValueError(f"receptors[{index}] 必須是 Receptor")
        _strict_nonempty_text(receptor.receptor_id, label=f"receptors[{index}].receptor_id")
        _strict_nonempty_text(receptor.study_site_id, label=f"receptors[{index}].study_site_id")
        _strict_nonempty_text(receptor.vertical_id, label=f"receptors[{index}].vertical_id")
        if receptor.receptor_id in seen:
            raise ValueError(f"receptor_id 不可重複：{receptor.receptor_id}")
        seen.add(receptor.receptor_id)
    return values


def _scenario_context(
    scenarios: Sequence[Scenario], receptors: Sequence[Receptor]
) -> tuple[tuple[Scenario, Receptor, tuple[str, str]], ...]:
    """解析每個 scenario 的 receptor 與 exact ``(site, vertical)`` 分層。"""

    scenario_values = _validated_scenarios(scenarios)
    receptor_values = _validated_receptors(receptors)
    receptors_by_id = {receptor.receptor_id: receptor for receptor in receptor_values}
    result: list[tuple[Scenario, Receptor, tuple[str, str]]] = []
    for index, scenario in enumerate(scenario_values):
        receptor = receptors_by_id.get(scenario.receptor_id)
        if receptor is None:
            raise ValueError(
                f"scenarios[{index}].receptor_id 找不到對應 receptor：{scenario.receptor_id}"
            )
        if receptor.study_site_id != scenario.study_site_id:
            raise ValueError(f"scenario/receptor study_site_id 不一致：{scenario.scenario_id}")
        result.append(
            (
                scenario,
                receptor,
                (scenario.study_site_id, receptor.vertical_id),
            )
        )
    scenario_strata = {item[2] for item in result}
    receptor_strata = {(item.study_site_id, item.vertical_id) for item in receptor_values}
    if scenario_strata != receptor_strata:
        raise ValueError("scenario 與 receptor 的 site/vertical strata 不完整一致")
    return tuple(result)


def _ranking_key(scenario: Scenario, vertical_id: str) -> tuple[bytes, str]:
    """依版本化 policy、站點、垂向與 scenario ID 建立 deterministic 排序鍵。"""

    payload = _length_prefixed_bytes(
        (
            PILOT_SCENARIO_SELECTION_RANKING_POLICY,
            scenario.study_site_id,
            vertical_id,
            scenario.scenario_id,
        )
    )
    return sha256(payload).digest(), scenario.scenario_id


def _selection_binding(
    *,
    mode: str,
    samples_per_stratum: int | None,
    source_scenarios: Sequence[Scenario],
    selected_scenarios: Sequence[Scenario],
    strata: list[dict[str, Any]],
) -> dict[str, Any]:
    """建立 root binding；呼叫端已先完成所有 scenario/receptor 語意 gate。"""

    return {
        "schema_version": PILOT_SCENARIO_SELECTION_SCHEMA_VERSION,
        "mode": mode,
        "ranking_policy": PILOT_SCENARIO_SELECTION_RANKING_POLICY,
        "stratum_fields": list(PILOT_SCENARIO_SELECTION_STRATUM_FIELDS)
        if mode == "pilot_stratified"
        else [],
        "samples_per_stratum": samples_per_stratum,
        "source_scenario_count": len(source_scenarios),
        "selected_scenario_count": len(selected_scenarios),
        "source_scenario_ids_sha256": scenario_ids_sha256(source_scenarios),
        "selected_scenario_ids_sha256": scenario_ids_sha256(selected_scenarios),
        "strata": strata,
    }


def build_full_scenario_selection(scenarios: Sequence[Scenario]) -> dict[str, Any]:
    """建立代表完整 scenario coverage 的 full selection binding。

    full mode 不做抽樣、不需要 receptor mapping，source 與 selected 皆為傳入的完整
    scenario tuple。回傳值是可直接寫入 run plan 的 ordinary dict；scenario 本身不被
    複製或修改，binding 只保存 order-independent count/hash 證據。
    """

    source = _validated_scenarios(scenarios)
    return _selection_binding(
        mode="full",
        samples_per_stratum=None,
        source_scenarios=source,
        selected_scenarios=source,
        strata=[],
    )


def select_pilot_scenarios(
    scenarios: Sequence[Scenario],
    receptors: Sequence[Receptor],
    samples_per_stratum: int,
    expected_source_scenario_count: int,
) -> tuple[tuple[Scenario, ...], dict[str, Any]]:
    """按 ``study_site_id × receptor.vertical_id`` 每層選出 N 筆 pilot 情境。

    完整 source scenario 先以 ``expected_source_scenario_count`` 做 count gate，再以
    receptor ID 解析垂向層位。每層候選依長度前綴 SHA-256 bytes 排序，scenario ID 作
    tie-break；因此輸入列順序不影響結果，且 N=1 的集合必然是 N=2 集合的子集。若
    任一 strata 不足 N、receptor 缺漏、scenario 無法解析或 ID 重複，函式不產出部分
    結果而直接拒絕。
    """

    n = _strict_positive_integer(samples_per_stratum, label="samples_per_stratum")
    source = _validated_scenarios(
        scenarios,
        expected_count=expected_source_scenario_count,
    )
    context = _scenario_context(source, receptors)
    grouped: dict[tuple[str, str], list[tuple[Scenario, Receptor]]] = defaultdict(list)
    for scenario, receptor, stratum in context:
        grouped[stratum].append((scenario, receptor))

    selected: list[Scenario] = []
    strata: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        candidates = grouped[stratum]
        if len(candidates) < n:
            raise ValueError(
                f"stratum {stratum[0]}/{stratum[1]} 情境不足 samples_per_stratum={n}"
            )
        ranked = sorted(
            candidates,
            key=lambda item: _ranking_key(item[0], item[1].vertical_id),
        )
        chosen = tuple(item[0] for item in ranked[:n])
        selected.extend(chosen)
        strata.append(
            {
                "study_site_id": stratum[0],
                "vertical_id": stratum[1],
                "source_count": len(candidates),
                "selected_count": len(chosen),
                "selected_scenario_ids_sha256": scenario_ids_sha256(chosen),
            }
        )

    selected_tuple = tuple(selected)
    binding = _selection_binding(
        mode="pilot_stratified",
        samples_per_stratum=n,
        source_scenarios=source,
        selected_scenarios=selected_tuple,
        strata=strata,
    )
    return selected_tuple, binding


def _strict_equal(expected: object, actual: object, *, label: str) -> None:
    """遞迴要求 binding 的容器型別與每個 scalar 都 exact 相同。"""

    if type(expected) is not type(actual):
        raise ValueError(f"{label} 型別與重算 binding 不一致")
    if isinstance(expected, dict):
        if set(expected) != set(actual):  # type: ignore[arg-type]
            raise ValueError(f"{label} key 與重算 binding 不一致")
        for key in expected:
            _strict_equal(expected[key], actual[key], label=f"{label}.{key}")  # type: ignore[index]
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):  # type: ignore[arg-type]
            raise ValueError(f"{label} 長度與重算 binding 不一致")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):  # type: ignore[arg-type]
            _strict_equal(left, right, label=f"{label}[{index}]")
        return
    if expected != actual:
        raise ValueError(f"{label} 與重算 binding 不一致")


def validate_scenario_selection_binding_shape(
    binding: Mapping[str, Any],
    run_kind: str,
    selected_scenario_count: int,
) -> None:
    """驗證 selection binding 的 exact topology、型別與 run-kind policy。

    這個 public shape validator 不讀 source scenario，因此只檢查可以由 binding 自身
    證明的欄位；source/selected ID hash 與 strata 內容的真實性由
    ``apply_scenario_selection`` 以目前完整 manifests 重算後 exact 比對。formal 只能
    使用 full；pilot 可使用 full 或 pilot_stratified；synthetic 只為既有工程 fixture
    保留 full 相容，不進入 runtime physical initializer。
    """

    if type(binding) is not dict:
        raise ValueError("scenario_selection 必須是 ordinary dict")
    if set(binding) != _SELECTION_BINDING_KEYS:
        raise ValueError("scenario_selection root 欄位集合不符")
    if type(run_kind) is not str or run_kind not in _ALLOWED_RUN_KINDS:
        raise ValueError("scenario_selection run_kind 不支援")
    expected_count = _strict_positive_integer(
        selected_scenario_count,
        label="scenario_selection selected_scenario_count argument",
    )
    if binding["schema_version"] != PILOT_SCENARIO_SELECTION_SCHEMA_VERSION:
        raise ValueError("scenario_selection schema_version 不支援")
    if binding["ranking_policy"] != PILOT_SCENARIO_SELECTION_RANKING_POLICY:
        raise ValueError("scenario_selection ranking_policy 不支援")
    mode = binding["mode"]
    if type(mode) is not str or mode not in {"full", "pilot_stratified"}:
        raise ValueError("scenario_selection mode 不支援")
    if run_kind in {"formal", "synthetic"} and mode != "full":
        raise ValueError(f"{run_kind} run 只允許 full scenario selection")
    stratum_fields = binding["stratum_fields"]
    samples = binding["samples_per_stratum"]
    if type(stratum_fields) is not list or any(type(item) is not str for item in stratum_fields):
        raise ValueError("scenario_selection stratum_fields 必須是字串 list")
    if mode == "full":
        if stratum_fields != [] or samples is not None:
            raise ValueError("full scenario selection 不得帶 stratified 欄位")
    else:
        if stratum_fields != list(PILOT_SCENARIO_SELECTION_STRATUM_FIELDS):
            raise ValueError("pilot_stratified stratum_fields 不符")
        _strict_positive_integer(samples, label="scenario_selection.samples_per_stratum")
    source_count = _strict_positive_integer(
        binding["source_scenario_count"],
        label="scenario_selection.source_scenario_count",
    )
    selected_count = _strict_positive_integer(
        binding["selected_scenario_count"],
        label="scenario_selection.selected_scenario_count",
    )
    if selected_count != expected_count:
        raise ValueError("scenario_selection selected count 與 plan scenario_count 不一致")
    if selected_count > source_count:
        raise ValueError("scenario_selection selected count 不得大於 source count")
    _strict_sha256(
        binding["source_scenario_ids_sha256"],
        label="scenario_selection.source_scenario_ids_sha256",
    )
    _strict_sha256(
        binding["selected_scenario_ids_sha256"],
        label="scenario_selection.selected_scenario_ids_sha256",
    )
    strata = binding["strata"]
    if type(strata) is not list:
        raise ValueError("scenario_selection.strata 必須是 list")
    if mode == "full":
        if source_count != selected_count or strata != []:
            raise ValueError("full scenario selection 必須 source/selected 相等且無 strata")
        if binding["source_scenario_ids_sha256"] != binding["selected_scenario_ids_sha256"]:
            raise ValueError("full scenario selection source/selected hash 必須相等")
        return

    if not strata:
        raise ValueError("pilot_stratified scenario selection 不可沒有 strata")
    seen_strata: set[tuple[str, str]] = set()
    total_source = 0
    total_selected = 0
    for index, row in enumerate(strata):
        if type(row) is not dict or set(row) != _STRATUM_KEYS:
            raise ValueError(f"scenario_selection.strata[{index}] 欄位集合不符")
        site = _strict_nonempty_text(row["study_site_id"], label=f"strata[{index}].study_site_id")
        vertical = _strict_nonempty_text(row["vertical_id"], label=f"strata[{index}].vertical_id")
        key = (site, vertical)
        if key in seen_strata:
            raise ValueError(f"scenario_selection strata 重複：{key}")
        seen_strata.add(key)
        source_stratum_count = _strict_positive_integer(
            row["source_count"], label=f"strata[{index}].source_count"
        )
        selected_stratum_count = _strict_positive_integer(
            row["selected_count"], label=f"strata[{index}].selected_count"
        )
        if selected_stratum_count != samples or selected_stratum_count > source_stratum_count:
            raise ValueError(f"scenario_selection.strata[{index}] count 不符")
        _strict_sha256(
            row["selected_scenario_ids_sha256"],
            label=f"strata[{index}].selected_scenario_ids_sha256",
        )
        total_source += source_stratum_count
        total_selected += selected_stratum_count
    strata_keys = [
        (row["study_site_id"], row["vertical_id"])
        for row in strata
    ]
    if strata_keys != sorted(strata_keys):
        raise ValueError("scenario_selection strata 必須按 site、vertical 排序")
    if total_selected != selected_count:
        raise ValueError("scenario_selection strata selected count 總和不一致")
    if total_source != source_count:
        raise ValueError("scenario_selection strata source count 總和不一致")


def apply_scenario_selection(
    binding: Mapping[str, Any],
    scenarios: Sequence[Scenario],
    receptors: Sequence[Receptor],
    expected_source_scenario_count: int,
    run_kind: str,
) -> tuple[Scenario, ...]:
    """由目前完整 scenarios/receptors 重算並 exact 比對 binding 後回傳 selected tuple。

    static loader 不信任 plan 保存的 selected scenario table，也不使用 binding 內未保存的
    ID 清單。它會先驗證 binding shape，再驗證目前 source count、receptor mapping 與
    scenario ID hash，最後重跑 full 或 stratified selector；任何 count、垂向 mapping、
    ranking metadata、strata、hash 或 scalar 型別差異都 fail closed。回傳 tuple 的順序是
    selector 定義的 deterministic strata/ranking 順序，後續 run-control 仍會套用既有
    execution ordering policy。
    """

    if type(binding) is not dict:
        raise ValueError("scenario_selection 必須是 ordinary dict")
    selected_count = binding.get("selected_scenario_count")
    _strict_positive_integer(
        expected_source_scenario_count,
        label="expected_source_scenario_count",
    )
    _strict_positive_integer(selected_count, label="scenario_selection.selected_scenario_count")
    validate_scenario_selection_binding_shape(binding, run_kind, selected_count)
    source = _validated_scenarios(
        scenarios,
        expected_count=expected_source_scenario_count,
    )
    # 即使是 full mode 也驗證 receptor 解析，讓 current manifest 的基本 identity 不會
    # 因 selector 未使用垂向分層而完全跳過；full 本身沒有垂向 hash 欄位，故垂向改變的
    # reject 只可能由 stratified binding 的重算差異觸發。
    _scenario_context(source, receptors)
    if binding["mode"] == "full":
        selected = source
        expected_binding = build_full_scenario_selection(source)
    else:
        selected, expected_binding = select_pilot_scenarios(
            source,
            receptors,
            binding["samples_per_stratum"],
            expected_source_scenario_count,
        )
    _strict_equal(expected_binding, binding, label="scenario_selection")
    if len(selected) != selected_count:
        raise ValueError("重算 selected scenario count 不一致")
    return selected


__all__ = [
    "PILOT_SCENARIO_SELECTION_POLICY",
    "PILOT_SCENARIO_SELECTION_RANKING_POLICY",
    "PILOT_SCENARIO_SELECTION_SCHEMA_VERSION",
    "PILOT_SCENARIO_SELECTION_STRATUM_FIELDS",
    "apply_scenario_selection",
    "build_full_scenario_selection",
    "canonical_scenario_ids_sha256",
    "scenario_ids_sha256",
    "select_pilot_scenarios",
    "validate_scenario_selection_binding_shape",
]
