"""WP-56：沉降速度細分圖（``settling_velocity_breakdown``）的類別名稱與逐粒子速度範圍標示測試。

背景
----
v7（run08）每類沉降速度有 10 個候選值、每顆粒子抽一個，``departures.parquet`` 因此多了
``particle_settling_velocity_mps``（粒子實際速度）與 ``material_id``（類別識別）兩欄；
舊版（run07 以前）每類只有一個速度，沒有這兩欄。本檔用合成後處理目錄（沿用
``test_native_figures`` 的造資料函式）涵蓋兩種情況：

* 沒有逐粒子速度欄位：縱軸刻度、軸名、圖說與 README 一律維持舊版寫法；
* 有逐粒子速度欄位：縱軸刻度為「類別顯示名稱 + 實際速度範圍」，圖說與 README 改寫成
  「每類 10 個候選沉降速度」，manifest 多一個 ``settling_velocity_breakdown_labels`` 區塊。

另外檢查終止類別（``terminal_counts``）只畫實際出現的類別、「沉底」有中文標籤。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_native_figures import _write_synthetic_coastline, _write_synthetic_postprocess_dir

from lagrangian_backtracking.native_runner import figures as native_figures
from lagrangian_backtracking.native_runner.backward_driver import (
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_LEFT_INTEREST_REGION,
)

# 合成資料的三個類別代表速度（與 test_native_figures._SETTLING_VELOCITIES_MPS 相同）對應的類別識別碼。
_MATERIAL_BY_REPRESENTATIVE = {
    -0.001: "oca_nonrecyclable_flexible_sheet",
    -0.01: "oca_wood_waterlogged_elongated",
    -0.1: "oca_other_recyclable_irregular_fragment",
}


def _make_inputs(tmp_path: Path, *, particle_velocity: str | None) -> tuple[Path, Path]:
    """寫合成後處理目錄；``particle_velocity`` 為 ``None`` 表示舊版（不加兩個可選欄位）。

    ``"spread"``：每類粒子速度為代表速度的 1、1.5、2 倍輪流（範圍 = 1～2 倍）；倍數依 i//3 輪替，
    因合成資料的類別依 i%3 輪替，兩者互不相關，所以每類都會出現三種倍數；
    ``"constant"``：每顆粒子速度等於代表速度（每類只有一個速度，但欄位存在）；
    ``"nan"``：第一顆粒子速度為缺值，用來確認 fail-closed。
    """

    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path)
    coastline_path = _write_synthetic_coastline(tmp_path)
    if particle_velocity is None:
        return postprocess_dir, coastline_path
    path = postprocess_dir / "departures.parquet"
    table = pq.read_table(path)
    rep = np.asarray(table.column("settling_velocity_mps").to_pylist(), dtype=np.float64)
    material = [_MATERIAL_BY_REPRESENTATIVE[round(float(v), 6)] for v in rep]
    factor = np.array([1.0, 1.5, 2.0])[(np.arange(rep.size) // 3) % 3]
    if particle_velocity == "constant":
        factor = np.ones(rep.size)
    speeds = (rep * factor).tolist()
    if particle_velocity == "nan":
        speeds[0] = None
    table = table.append_column("material_id", pa.array(material, type=pa.string()))
    table = table.append_column("particle_settling_velocity_mps", pa.array(speeds, type=pa.float64()))
    pq.write_table(table, path)
    return postprocess_dir, coastline_path


def _run(
    tmp_path: Path,
    postprocess_dir: Path,
    coastline_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    figure_text: str = "full",
) -> tuple[Path, dict[str, dict[str, list[str]]]]:
    """出圖並攔截每站沉降速度細分圖與停止類別圖的座標軸文字。

    回傳 ``(輸出目錄, {"<站>/<圖種>": {"ylabel": ..., "bar_ticks": [...], "heat_ticks": [...]}})``。
    """

    captured: dict[str, dict[str, list[str]]] = {}
    original_save = native_figures._save_exclusive

    def _spy(fig, path, *, dpi, **kw):
        key = f"{path.parent.name}/{path.stem}"
        if path.stem == "settling_velocity_breakdown":
            ax_bar, ax_heat = fig.axes[0], fig.axes[1]
            captured[key] = {
                "ylabel": [ax_bar.get_ylabel()],
                "bar_ticks": [t.get_text() for t in ax_bar.get_yticklabels()],
                "heat_ticks": [t.get_text() for t in ax_heat.get_yticklabels()],
            }
        return original_save(fig, path, dpi=dpi, **kw)

    monkeypatch.setattr(native_figures, "_save_exclusive", _spy)
    output_dir = tmp_path / f"out-{figure_text}"
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir,
        coastline_path=coastline_path,
        output_dir=output_dir,
        figure_text=figure_text,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")
    return output_dir, captured


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _caption(output_dir: Path, site_id: str, stem: str) -> str:
    """從 captions.md 取出某站某圖種的圖說（站標題為「## 站名（站代號）」，圖標題為「### `圖種.png`」）。"""

    text = (output_dir / "captions.md").read_text(encoding="utf-8")
    heading = re.search(rf"^## .*（{re.escape(site_id)}）$", text, re.MULTILINE)
    assert heading is not None, f"captions.md 找不到站 {site_id}"
    block = text[heading.end():].split("\n## ")[0]
    return block.split(f"### `{stem}.png`")[1].split("\n### ")[0].strip()


# ── 沒有逐粒子速度欄位：舊版寫法完全不變 ─────────────────────────────────────


def test_legacy_departures_without_particle_velocity_keep_old_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    postprocess_dir, coastline_path = _make_inputs(tmp_path, particle_velocity=None)
    output_dir, captured = _run(tmp_path, postprocess_dir, coastline_path, monkeypatch)
    manifest = _manifest(output_dir)
    cjk = manifest["cjk_labels_used"]

    for site_id in ("site_a", "site_b"):
        got = captured[f"{site_id}/settling_velocity_breakdown"]
        assert got["bar_ticks"] == ["-0.1", "-0.01", "-0.001"]
        assert got["heat_ticks"] == ["-0.1", "-0.01", "-0.001"]
        expected = (
            "沉降速度（m/s，負值為下沉）" if cjk else "Settling velocity (m/s, negative = sinking)"
        )
        assert got["ylabel"] == [expected]
        caption = _caption(output_dir, site_id, "settling_velocity_breakdown")
        assert caption.startswith("十個未校準沉降速度敏感度格點各自的穿越開放邊界比例長條圖")
        assert "particle_settling_velocity_mps" not in caption

    assert "settling_velocity_breakdown_labels" not in manifest
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert (
        "8. `settling_velocity_breakdown.png`：十種沉降速度（`settling_velocity_mps`）各\n"
        "   自的穿越開放邊界比例（長條圖）與各沉降速度分組內的方位扇區相對來源權重\n"
        "   （熱圖）。資料來源：`departures.parquet`（逐沉降速度重新分組聚合）。\n"
    ) in readme


# ── 有逐粒子速度欄位：類別顯示名稱 + 實際速度範圍 ────────────────────────────


@pytest.mark.parametrize("figure_text", ["full", "none"])
def test_particle_velocity_ticks_use_display_names_and_speed_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, figure_text: str
) -> None:
    postprocess_dir, coastline_path = _make_inputs(tmp_path, particle_velocity="spread")
    output_dir, captured = _run(
        tmp_path, postprocess_dir, coastline_path, monkeypatch, figure_text=figure_text
    )
    manifest = _manifest(output_dir)
    if not manifest["cjk_labels_used"]:
        pytest.skip("測試環境沒有合格繁體中文字型，中文標籤由英文退場測試另行涵蓋")

    for site_id in ("site_a", "site_b"):
        got = captured[f"{site_id}/settling_velocity_breakdown"]
        # 排序仍依類別代表速度：-0.1、-0.01、-0.001（與舊版列順序相同）。
        assert got["bar_ticks"] == [
            "可回收塑膠 0.100–0.200",
            "竹木 0.0100–0.0200",
            "不可回收塑膠 0.00100–0.00200",
        ]
        assert got["heat_ticks"] == ["可回收塑膠", "竹木", "不可回收塑膠"]
        assert got["ylabel"] == ["沉降速度（m/s）"]
        assert not any("其他" in label for label in got["bar_ticks"] + got["heat_ticks"])


def test_particle_velocity_captions_readme_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    postprocess_dir, coastline_path = _make_inputs(tmp_path, particle_velocity="spread")
    output_dir, _captured = _run(tmp_path, postprocess_dir, coastline_path, monkeypatch)
    manifest = _manifest(output_dir)

    caption = _caption(output_dir, "site_a", "settling_velocity_breakdown")
    assert caption.startswith("3個沉降速度類別各自的穿越開放邊界比例長條圖")
    assert "每類有 10 個候選沉降速度" in caption
    assert "十分之一" in caption
    assert "「不可回收塑膠」對應材質清冊的「其他/不可回收」" in caption
    assert "十個未校準" not in caption
    assert "particle_settling_velocity_mps" in caption
    # manifest 內的圖說與 captions.md 同一份文字。
    record = manifest["output_png_sha256"]["site_a/settling_velocity_breakdown.png"]
    assert record["caption"].startswith(caption.split("。")[0])

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "每類 10 個候選沉降速度" in readme
    assert "十種沉降速度（`settling_velocity_mps`）" not in readme
    assert "particle_settling_velocity_mps" in readme

    labels = manifest["settling_velocity_breakdown_labels"]
    rows = labels["per_site"]["site_a"]
    assert [r["material_id"] for r in rows] == [
        "oca_other_recyclable_irregular_fragment",
        "oca_wood_waterlogged_elongated",
        "oca_nonrecyclable_flexible_sheet",
    ]
    assert [r["display_name_zh"] for r in rows] == ["可回收塑膠", "竹木", "不可回收塑膠"]
    assert rows[0]["particle_abs_min_mps"] == pytest.approx(0.1)
    assert rows[0]["particle_abs_max_mps"] == pytest.approx(0.2)
    assert rows[0]["distinct_velocity_count"] == 3
    assert rows[2]["tick_label_zh"] == "不可回收塑膠 0.00100–0.00200"


def test_particle_velocity_constant_per_class_prints_single_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    postprocess_dir, coastline_path = _make_inputs(tmp_path, particle_velocity="constant")
    output_dir, captured = _run(tmp_path, postprocess_dir, coastline_path, monkeypatch)
    if not _manifest(output_dir)["cjk_labels_used"]:
        pytest.skip("測試環境沒有合格繁體中文字型")
    got = captured["site_a/settling_velocity_breakdown"]
    assert got["bar_ticks"] == ["可回收塑膠 0.100", "竹木 0.0100", "不可回收塑膠 0.00100"]


def test_particle_velocity_english_fallback_uses_material_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_no_font() -> None:
        raise ValueError("找不到涵蓋報告必要字元的合格繁體中文字型")

    monkeypatch.setattr(native_figures, "resolve_cjk_font", _raise_no_font)
    postprocess_dir, coastline_path = _make_inputs(tmp_path, particle_velocity="spread")
    output_dir, captured = _run(tmp_path, postprocess_dir, coastline_path, monkeypatch)
    assert _manifest(output_dir)["cjk_labels_used"] is False
    got = captured["site_a/settling_velocity_breakdown"]
    assert got["bar_ticks"][0] == "oca_other_recyclable_irregular_fragment 0.100–0.200"
    assert got["heat_ticks"] == ["", "", ""]
    assert got["ylabel"] == ["Settling-velocity class (speed range, m/s)"]


def test_particle_velocity_with_missing_value_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    postprocess_dir, coastline_path = _make_inputs(tmp_path, particle_velocity="nan")
    with pytest.raises(native_figures.FiguresError, match="particle_settling_velocity_mps"):
        _run(tmp_path, postprocess_dir, coastline_path, monkeypatch)


def test_class_velocity_rows_helper_is_none_without_column() -> None:
    departures = {"settling_velocity_mps": np.array([-0.1, -0.1])}
    assert native_figures._class_velocity_rows(departures, [-0.1]) is None


def test_speed_text_has_three_significant_digits_and_keeps_trailing_zero() -> None:
    assert native_figures._three_sig(0.000897) == "0.000897"
    assert native_figures._three_sig(0.108) == "0.108"
    assert native_figures._three_sig(0.920397) == "0.920"
    assert native_figures._three_sig(0.0114) == "0.0114"


# ── 終止類別：只畫實際出現者，「沉底」有中文標籤 ─────────────────────────────


def test_terminal_status_present_lists_only_observed_codes_and_deposited_has_zh_label() -> None:
    codes = np.array([NATIVE_STATUS_LEFT_INTEREST_REGION] * 98 + [NATIVE_STATUS_DEPOSITED] * 2)
    present = native_figures._status_codes_present({"final_status_code": codes})
    assert present == (NATIVE_STATUS_LEFT_INTEREST_REGION, NATIVE_STATUS_DEPOSITED)
    assert native_figures._STATUS_LABELS_ZH[NATIVE_STATUS_DEPOSITED] == "觸底"
    assert native_figures._STATUS_LABELS_EN[NATIVE_STATUS_DEPOSITED] == "Deposited"
