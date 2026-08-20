"""設定 schema、五站計數與正式發布閘門測試。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from lagrangian_backtracking.config import ProjectConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"


def _payload() -> dict:
    """讀取範例 YAML mapping，供單一契約破壞測試使用。"""

    value = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_example_config_has_fixed_scientific_counts() -> None:
    """範例設定必須保留 4 domains、5 sites 與全案 50,000 基礎情境。"""

    config = load_config(EXAMPLE_CONFIG)
    assert len(config.domains) == 4
    assert len(config.study_sites) == 5
    assert config.scenarios.expected_receptor_count == 100
    assert config.scenarios.scenario_count == 50_000


def test_config_hash_is_independent_of_mapping_order() -> None:
    """canonical hash 不應因 YAML key 順序改變，避免無科學差異卻產生新 run。"""

    first = ProjectConfig.model_validate(_payload())
    reordered = dict(reversed(list(_payload().items())))
    second = ProjectConfig.model_validate(reordered)
    assert first.config_hash() == second.config_hash()


def test_rejects_region_a_site_merging() -> None:
    """A 區必須同時保留貢寮與龜山島兩個獨立 study_site_id。"""

    payload = deepcopy(_payload())
    payload["study_sites"] = [site for site in payload["study_sites"] if site["study_site_id"] != "guishan"]
    payload["study_area"]["expected_study_site_count"] = 4
    with pytest.raises(ValueError, match="A 區必須恰含"):
        ProjectConfig.model_validate(payload)


def test_rejects_reduced_scenario_count() -> None:
    """不得以修改設定把全案完整交叉靜默降回 10,000 或每站 1,000。"""

    payload = _payload()
    payload["scenarios"]["scenario_count"] = 10_000
    with pytest.raises(ValueError, match="情境契約"):
        ProjectConfig.model_validate(payload)


def test_example_is_intentionally_blocked_for_formal_release() -> None:
    """design example 尚缺 M、時步、horizon 與 expanded A ID，正式模式必須 fail closed。"""

    with pytest.raises(ValueError, match="approved reconstruction 或 gap-safe"):
        load_config(EXAMPLE_CONFIG, formal_release=True)


def test_formal_time_gate_accepts_either_ocm_support_manifest_name() -> None:
    """重建未過門檻時可採 gap-safe baseline，設定 gate 不得強迫偽造重建成功。"""

    payload = _payload()
    payload["inputs"]["ocm_gap_safe_arrival_manifest"] = "manifests/ocm-gap-safe.json"
    config = ProjectConfig.model_validate(payload)
    with pytest.raises(ValueError) as exc_info:
        config.assert_formal_release_ready()
    assert "approved reconstruction 或 gap-safe" not in str(exc_info.value)


def test_yaml_member_field_is_not_silently_ignored() -> None:
    """schema 欄位必須直接對應 YAML 的 members_per_scenario，避免正式 M 永遠讀成 None。"""

    payload = _payload()
    payload["scenarios"]["members_per_scenario"] = 8
    config = ProjectConfig.model_validate(payload)
    assert config.scenarios.members_per_scenario == 8
