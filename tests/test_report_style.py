"""正式 Matplotlib renderer style 的 fail-closed 與 reproducibility 測試。

本檔只建立小型 ``ReportSpec``、暫存普通目錄與假的 ``CJKFontSelection``；不讀取
OCM／NWW3、不載入真實主機 CJK 字型、不繪製正式圖，也不下載任何資料。Matplotlib
若已被其他測試載入，測試仍只檢查 ``report_style`` 自身 globals 沒有頂層依賴，並
透過 monkeypatch 驗證 gate、Agg、rc restoration 與固定 hash 的順序與結果。
"""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from lagrangian_backtracking import report_style
from lagrangian_backtracking.report_spec import ReportSpec

_MPLCONFIGDIR_ERROR = "MPLCONFIGDIR 必須明示為既有、可寫、非符號連結的專用目錄"


def _make_report_spec(canonical_sha256: str = "a" * 64) -> ReportSpec:
    """建立不含任何海洋資料的最小合法 renderer spec fixture。"""

    return ReportSpec(
        schema_version="1.0.0",
        run_id="synthetic-style-test",
        aggregate_spec_canonical_sha256="b" * 64,
        primary_kde_bandwidth_m=100.0,
        minimum_kde_raw_count=10,
        low_sample_min_member_count=2,
        vertical_depth_bin_edges_m=(0.0, 10.0),
        representative_trajectory_count_per_site=8,
        representative_selection_policy="stable_hash_core_season_tide_v1",
        representative_selection_seed=7,
        travel_age_quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        pathway_first_passage_quantiles=(0.25, 0.5, 0.75),
        figure_formats=("png", "svg", "pdf"),
        raster_dpi=300,
        renderer_style_version="academic_zh_tw_v1",
        language="zh-TW",
        source_sha256="c" * 64,
        canonical_sha256=canonical_sha256,
    )


def _fake_font_selection() -> object:
    """建立只含 provenance 的假字型 record，不依賴主機字型檔。"""

    # 延遲 import，讓本檔的 lazy-import 測試仍能在 report_style import 後先觀察
    # module globals；這個實際 CJKFontSelection 只承載假的 bytes hash。
    from lagrangian_backtracking import report_font

    return report_font.CJKFontSelection(
        family=report_font.CJK_FONT_CANDIDATES[0],
        filename="fake-report-font.ttf",
        file_sha256="d" * 64,
        required_glyph_count=len(report_font.REQUIRED_REPORT_GLYPHS),
    )


def _configure_fake_renderer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    canonical_sha256: str = "a" * 64,
) -> tuple[ReportSpec, object]:
    """設定普通 MPL cache 與 fake resolver，回傳 spec 及預期字型 record。"""

    mplconfigdir = tmp_path / "mplconfig"
    mplconfigdir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mplconfigdir))
    selection = _fake_font_selection()
    report_font = importlib.import_module("lagrangian_backtracking.report_font")
    monkeypatch.setattr(report_font, "resolve_cjk_font", lambda: selection)
    return _make_report_spec(canonical_sha256), selection


def test_import_is_lazy_even_when_matplotlib_is_already_loaded() -> None:
    """report_style module import 不應在 globals 建立 Matplotlib 或 report_font 名稱。"""

    module = importlib.reload(report_style)
    assert "matplotlib" not in module.__dict__
    assert "pyplot" not in module.__dict__
    assert "report_font" not in module.__dict__


def test_validate_mplconfigdir_accepts_only_existing_ordinary_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """普通既有目錄可通過，且 gate 不建立額外檔案。"""

    directory = tmp_path / "ordinary"
    directory.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(directory))
    before = sorted(path.name for path in directory.iterdir())

    assert report_style.validate_mplconfigdir() == directory
    assert sorted(path.name for path in directory.iterdir()) == before


@pytest.mark.parametrize("value", [None, "", "relative/mplconfig", "."])
def test_validate_mplconfigdir_rejects_missing_empty_or_relative_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    """缺少、空字串與相對路徑都必須以固定錯誤拒絕。"""

    if value is None:
        monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    else:
        monkeypatch.setenv("MPLCONFIGDIR", value)
    with pytest.raises(ValueError, match=f"^{_MPLCONFIGDIR_ERROR}$") as error:
        report_style.validate_mplconfigdir()
    assert "/" not in str(error.value)


def test_validate_mplconfigdir_rejects_symlink_file_missing_and_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """symlink、普通檔案、不存在與 access gate 失敗均不得放行。"""

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    regular_file = tmp_path / "regular-file"
    regular_file.write_bytes(b"not a directory")
    missing = tmp_path / "missing"

    for candidate in (symlink, regular_file, missing):
        monkeypatch.setenv("MPLCONFIGDIR", str(candidate))
        with pytest.raises(ValueError, match=f"^{_MPLCONFIGDIR_ERROR}$"):
            report_style.validate_mplconfigdir()

    monkeypatch.setenv("MPLCONFIGDIR", str(target))
    real_access = os.access
    monkeypatch.setattr(report_style.os, "access", lambda _path, _mode: False)
    with pytest.raises(ValueError, match=f"^{_MPLCONFIGDIR_ERROR}$"):
        report_style.validate_mplconfigdir()
    assert real_access(target, os.W_OK | os.X_OK) in {True, False}


def test_validate_mplconfigdir_rejects_home_default_and_canonical_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """$HOME/.matplotlib 及指向同一 canonical 位置的 alias 都必須拒絕。"""

    home = tmp_path / "fake-home"
    default = home / ".matplotlib"
    default.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    for candidate in (default, home / "child" / ".." / ".matplotlib"):
        monkeypatch.setenv("MPLCONFIGDIR", str(candidate))
        with pytest.raises(ValueError, match=f"^{_MPLCONFIGDIR_ERROR}$") as error:
            report_style.validate_mplconfigdir()
        assert str(home) not in str(error.value)


def test_resolve_uses_gate_then_agg_then_font_and_fixed_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """resolve 順序、Agg backend、字型 provenance 與 salt 計算必須固定。"""

    spec, selection = _configure_fake_renderer(monkeypatch, tmp_path)
    matplotlib = importlib.import_module("matplotlib")
    report_font = importlib.import_module("lagrangian_backtracking.report_font")
    calls: list[str] = []
    original_validate = report_style.validate_mplconfigdir
    original_use = matplotlib.use

    def recording_validate() -> Path:
        calls.append("gate")
        return original_validate()

    def recording_use(backend: str, *, force: bool = False) -> None:
        calls.append(f"use:{backend}:{force}")
        original_use(backend, force=force)

    def recording_resolve() -> object:
        calls.append("font")
        return selection

    monkeypatch.setattr(report_style, "validate_mplconfigdir", recording_validate)
    monkeypatch.setattr(matplotlib, "use", recording_use)
    monkeypatch.setattr(report_font, "resolve_cjk_font", recording_resolve)

    style = report_style.resolve_report_render_style(spec)

    expected_salt = hashlib.sha256(
        f"academic_zh_tw_v1|{spec.canonical_sha256}".encode()
    ).hexdigest()
    assert calls == ["gate", "use:Agg:True", "font"]
    assert matplotlib.get_backend().lower() == "agg"
    assert style.font_selection == selection
    assert style.svg_hashsalt == expected_salt
    assert len(style.svg_hashsalt) == 64
    assert style.svg_hashsalt == style.svg_hashsalt.lower()
    assert "backend" not in style.rc_params
    assert style.rc_params["font.family"] == (selection.family,)
    assert style.rc_params["svg.hashsalt"] == expected_salt


def test_hash_changes_only_with_canonical_sha_and_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """相同 canonical hash 產生相同 salt，不同 canonical hash 必須改變 salt。"""

    spec, selection = _configure_fake_renderer(monkeypatch, tmp_path)
    style_a = report_style.resolve_report_render_style(spec)
    monkeypatch.setattr(
        importlib.import_module("lagrangian_backtracking.report_font"),
        "resolve_cjk_font",
        lambda: selection,
    )
    style_a_repeat = report_style.resolve_report_render_style(spec)

    mplconfigdir = Path(os.environ["MPLCONFIGDIR"])
    assert mplconfigdir.is_dir()
    style_b = report_style.resolve_report_render_style(
        _make_report_spec("e" * 64),
    )
    assert style_a.svg_hashsalt == style_a_repeat.svg_hashsalt
    assert style_a.svg_hashsalt != style_b.svg_hashsalt


def test_report_render_style_constructor_defensively_snapshots_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """constructor 必須拒絕 rc/path/hash tamper 並封存 mapping。"""

    spec, selection = _configure_fake_renderer(monkeypatch, tmp_path)
    style = report_style.resolve_report_render_style(spec)
    supplied = dict(style.rc_params)
    constructed = report_style.ReportRenderStyle(
        renderer_style_version=style.renderer_style_version,
        language=style.language,
        raster_dpi=style.raster_dpi,
        font_selection=selection,
        svg_hashsalt=style.svg_hashsalt,
        rc_params=supplied,
    )
    supplied["figure.dpi"] = 999
    assert isinstance(constructed.rc_params, MappingProxyType)
    assert constructed.rc_params["figure.dpi"] == 100
    assert "MPLCONFIGDIR" not in repr(constructed)
    assert str(Path(os.environ["MPLCONFIGDIR"])) not in repr(constructed)
    with pytest.raises(FrozenInstanceError):
        constructed.language = "en-US"  # type: ignore[misc]

    tampered_values = [
        {**style.rc_params, "figure.dpi": 101},
        {**style.rc_params, "unexpected": True},
        {key: value for key, value in style.rc_params.items() if key != "grid.alpha"},
        {**style.rc_params, "svg.hashsalt": "f" * 64},
        {**style.rc_params, "font.family": ("/tmp/private-font.ttf",)},
    ]
    for rc_params in tampered_values:
        with pytest.raises((TypeError, ValueError)):
            report_style.ReportRenderStyle(
                renderer_style_version=style.renderer_style_version,
                language=style.language,
                raster_dpi=style.raster_dpi,
                font_selection=selection,
                svg_hashsalt=style.svg_hashsalt,
                rc_params=rc_params,
            )

    with pytest.raises(ValueError):
        report_style.ReportRenderStyle(
            renderer_style_version=style.renderer_style_version,
            language=style.language,
            raster_dpi=style.raster_dpi,
            font_selection=selection,
            svg_hashsalt="A" * 64,
            rc_params=style.rc_params,
        )


def test_report_render_style_context_applies_and_restores_rc_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """context 內套用固定 rc，離開後 rcParams 必須逐項回復。"""

    spec, _selection = _configure_fake_renderer(monkeypatch, tmp_path)
    matplotlib = importlib.import_module("matplotlib")
    tracked_keys = (
        "font.family",
        "axes.unicode_minus",
        "svg.hashsalt",
        "savefig.dpi",
        "figure.dpi",
        "text.usetex",
        "image.origin",
    )
    before = {key: matplotlib.rcParams[key] for key in tracked_keys}

    with report_style.report_render_style_context(spec) as style:
        assert matplotlib.get_backend().lower() == "agg"
        assert matplotlib.rcParams["font.family"] == [style.font_selection.family]
        assert matplotlib.rcParams["svg.hashsalt"] == style.svg_hashsalt
        assert matplotlib.rcParams["savefig.dpi"] == 300.0
        assert matplotlib.rcParams["figure.dpi"] == 100.0
        assert matplotlib.rcParams["text.usetex"] is False
        assert matplotlib.rcParams["image.origin"] == "lower"

    after = {key: matplotlib.rcParams[key] for key in tracked_keys}
    assert after == before
