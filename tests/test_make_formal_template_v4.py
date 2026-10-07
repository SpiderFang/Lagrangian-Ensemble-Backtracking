"""WP-51 整合：v4 正式範本產生器（scripts/make_formal_template_v4.py）的自足測試。

伺服器上的 v3 範本不在 repo 內，所以這裡把 repo 的正式設定「逆向還原」成 v3 範本的寫法
（龜山島舊錨點、舊 Kz、舊 integration 欄位），再用產生器轉回 v4，驗證結構結果與 repo 正式設定
相同、只多出 ``age_sampling_scope: per_member``；並守住起訖標記不因 repo YAML 改版而悄悄失效。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs" / "lagrangian_backtracking.formal_h30_h60_h90_m10.yaml"


def _module():
    path = ROOT / "scripts" / "make_formal_template_v4.py"
    spec = importlib.util.spec_from_file_location("make_formal_template_v4_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _as_v3_text(module, repo_text: str) -> str:
    """把 repo 正式設定改寫成 v3 範本的相關寫法（只含產生器會動到的四處）。"""

    guishan = module._span(repo_text, module._GUISHAN_START, module._GUISHAN_END, "guishan", include_end=False)
    kz = module._span(repo_text, module._KZ_START, module._KZ_END, "kz", include_end=True)
    integration = module._span(
        repo_text, module._INTEGRATION_START, module._CFL_END, "integration", include_end=True
    )
    lines = integration.splitlines(keepends=True)
    interval_index = next(i for i, line in enumerate(lines) if line == module._OUTPUT_INTERVAL_V4)
    cfl_index = next(i for i, line in enumerate(lines) if line.startswith("  # 實際子步上限"))
    interval_text = "".join(lines[: interval_index + 1])
    middle_text = "".join(lines[interval_index + 1 : cfl_index])
    cfl_text = "".join(lines[cfl_index:])
    text = repo_text.replace(guishan, module._ANCHOR_V3)
    text = text.replace(kz, module._KZ_V3)
    text = text.replace(interval_text + middle_text + cfl_text, module._OUTPUT_INTERVAL_V3 + middle_text + module._CFL_V3)
    return text


def test_generator_roundtrip_matches_repo_formal_plus_per_member() -> None:
    module = _module()
    repo_text = FORMAL.read_text(encoding="utf-8")
    base_text = _as_v3_text(module, repo_text)
    assert base_text != repo_text
    base = yaml.safe_load(base_text)
    assert base["physics"]["vertical_diffusion"]["constant_kz_m2ps"] != 0.0003
    assert base["integration"]["output_interval_seconds"] == 300

    generated = module.generate(base_text, repo_text)
    payload = yaml.safe_load(generated)
    expected = yaml.safe_load(repo_text)
    expected["scenarios"]["bed_residence_time"]["age_sampling_scope"] = "per_member"
    assert payload == expected

    differing = module._check_diff(base_text, generated)
    assert "scenarios.bed_residence_time.age_sampling_scope" in differing
    assert "physics.vertical_diffusion.constant_kz_m2ps" in differing
    assert "study_sites[guishan].anchor_lonlat" in differing


def test_generator_rejects_base_that_is_not_the_expected_v3_shape() -> None:
    """範本改版（找不到或出現多次的舊文字）時必須中止，不得靜默產生半套設定。"""

    module = _module()
    repo_text = FORMAL.read_text(encoding="utf-8")
    base_text = _as_v3_text(module, repo_text)
    with pytest.raises(SystemExit):
        module.generate(base_text.replace(module._KZ_V3, "    constant_kz_m2ps: 0.001\n"), repo_text)
    with pytest.raises(SystemExit):
        module.generate(base_text + module._ANCHOR_V3, repo_text)
