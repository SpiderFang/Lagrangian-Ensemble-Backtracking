"""source-pathway-v1 成果包的 synthetic 工程契約測試。

本檔不讀取 OCM schema 3、NWW3 schema 1、SERVER forcing 或 trajectory shard；輸入是
既有 report-statistics 的小型 immutable payload，並以 monkeypatch 模擬「已通過
aggregate reader」的入口，讓測試集中檢查 source-pathway 自身的圖檔、三張 sidecar、
manifest provenance、沉降速度 fail-closed 與 validator tamper 行為。測試通過只代表
資料拓撲與 renderer 工程邊界成立，不代表正式五站科學結果、來源機率或沉積質量已被
驗證。人工圖面 QA 可用測試建立的 final 目錄直接檢查 PNG；正式 CLI 仍要求真實或由
aggregate release writer 產出的完整 release。
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest

import lagrangian_backtracking.cli as cli_module
import lagrangian_backtracking.report_font as report_font
import lagrangian_backtracking.source_pathway_release as source_pathway_module
from lagrangian_backtracking.models import ParticleStatus


def _load_statistics_fixture_module() -> ModuleType:
    """載入既有 exact payload fixture，不把 production code 複製到本測試。

    report-statistics fixture 本身只建構兩站 1×2 公尺格網與完整停止／事件產品，沒有
    讀取外部資料。使用獨立 module name 避免干擾 pytest 已載入的測試模組；取用
    ``pytest.fixture`` 包裹函式的 ``__wrapped__`` 是測試資料組合，不是 production API。
    """

    fixture_path = Path(__file__).with_name("test_report_statistics.py")
    spec = importlib.util.spec_from_file_location("source_pathway_report_fixture", fixture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("無法載入 synthetic report fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exact_payload_and_report_spec() -> tuple[object, object]:
    """回傳兩站小型向下沉降 aggregate／ReportSpec synthetic fixture。"""

    module = _load_statistics_fixture_module()
    return module.exact_payload_and_report_spec.__wrapped__()


@pytest.fixture
def mplconfigdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """準備 renderer 所需的既有 cache 目錄與不依賴主機字型檔的 resolver。

    report style 仍由 production ``report_render_style_context`` 執行 Agg、MPLCONFIGDIR
    與 metadata gate；假 resolver 只隔離測試環境的字型安裝差異，不改變 renderer 的
    CJK family／provenance 欄位契約。
    """

    directory = tmp_path / "mplconfig"
    directory.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(directory))
    selection = report_font.CJKFontSelection(
        family=report_font.CJK_FONT_CANDIDATES[0],
        filename="synthetic-report-font.ttf",
        file_sha256="d" * 64,
        required_glyph_count=len(report_font.REQUIRED_REPORT_GLYPHS),
    )
    monkeypatch.setattr(report_font, "resolve_cjk_font", lambda: selection)
    return directory


def _build_with_fake_reader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: object,
    report_spec: object,
    aggregate_root: Path,
    destination: Path,
    mplconfigdir: Path,
) -> Path:
    """以 synthetic reader binding 建立一份完整 source-pathway final。"""

    aggregate_root.mkdir()
    # source-pathway provenance 要雜湊 aggregate release 實際 manifest bytes；reader
    # 在這個 bounded test 中以 monkeypatch 提供 payload，因此仍建立固定檔案讓 hash
    # 綁定可被 validator 與 tamper assertion 檢查。
    (aggregate_root / "aggregate_manifest.json").write_bytes(b'{"synthetic":true}\n')
    report_path = aggregate_root.parent / "report-spec.json"
    report_path.write_bytes(b"synthetic report spec placeholder\n")
    monkeypatch.setattr(source_pathway_module, "read_aggregate_release", lambda _: payload)
    monkeypatch.setattr(source_pathway_module, "load_report_spec", lambda _: report_spec)
    # synthetic font selection 只保存 provenance，不提供真正 CJK glyph；Matplotlib
    # 因而會針對每一個中文 code point 發出 UserWarning。這裡以區域性的 warning
    # context 壓掉該工程 fixture 的已知缺 glyph 噪音，並保留 production style gate
    # 與正式字型解析不變；若 renderer 產生其他類型警告，測試仍會照常顯示。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font\(s\) .*\.",
            category=UserWarning,
        )
        return source_pathway_module.build_source_pathway_release(
            aggregate_release=aggregate_root,
            report_spec=report_path,
            destination=destination,
            mplconfigdir=mplconfigdir,
        )


def _payload_with_invalid_velocity(payload: object, velocity: float) -> object:
    """只在測試邊界繞過 record constructor，模擬竄改後的已讀 payload。

    ``ScenarioStratum`` production constructor 本身已拒絕零／正沉降；若只測正常
    constructor 會測不到 source-pathway 的第二道防線。這裡使用 dataclass low-level
    snapshot 刻意建立不合法 immutable record，對應 manifest／記憶體竄改情境；production
    builder 必須在任何統計或檔案建立前再次檢查並 fail closed。
    """

    original_stratum = payload.scenario_strata[0]
    invalid_stratum = object.__new__(type(original_stratum))
    for field in fields(original_stratum):
        object.__setattr__(invalid_stratum, field.name, getattr(original_stratum, field.name))
    object.__setattr__(invalid_stratum, "settling_velocity_mps", velocity)

    invalid_payload = object.__new__(type(payload))
    for field in fields(payload):
        object.__setattr__(invalid_payload, field.name, getattr(payload, field.name))
    object.__setattr__(
        invalid_payload,
        "scenario_strata",
        (invalid_stratum, *payload.scenario_strata[1:]),
    )
    return invalid_payload


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """建立 metadata gate 測試所需的最小 PNG 區塊。

    Production parser 在 Matplotlib 已完成編碼後只讀區塊長度、type 與文字 keyword，
    不重算 CRC；因此 fixture 使用固定四個零位元組占位，讓測試直接聚焦「IDAT 偶然
    出現 date 不得誤判、tEXt 的 Date keyword 必須拒絕」這項契約。
    """

    return len(payload).to_bytes(4, byteorder="big") + chunk_type + payload + (b"\x00" * 4)


def test_png_wall_clock_gate_only_reads_text_metadata_keywords() -> None:
    """確認壓縮像素中的 date 不誤判，文字 metadata 的 Date 仍 fail closed。"""

    signature = b"\x89PNG\r\n\x1a\n"
    image_only = signature + _png_chunk(b"IDAT", b"compressed-date-bytes") + _png_chunk(b"IEND", b"")
    source_pathway_module._assert_no_wall_clock_metadata(image_only, file_format="png")

    dated = signature + _png_chunk(b"tEXt", b"Date\x002026-09-08") + _png_chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="wall-clock date metadata"):
        source_pathway_module._assert_no_wall_clock_metadata(dated, file_format="png")


def test_pre_window_outcome_has_explicit_release_status_label() -> None:
    """outcome sidecar 與圖軸不可把 pre-window 狀態退回小寫 enum raw value。"""

    assert (
        source_pathway_module._status_label(ParticleStatus.PRE_WINDOW_DEPOSITION.value)
        == "PRE_WINDOW_DEPOSITION"
    )


def test_build_validate_sidecars_and_figures_round_trip(
    exact_payload_and_report_spec: tuple[object, object],
    mplconfigdir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """建立兩站六面板圖與 sidecar，並確認 validator／manifest provenance 完整。

    這個 bounded test 同時覆蓋 1×2 格網的低樣本 hatch／小網格 contour skip、PNG／SVG／
    PDF bytes、grid 的首次／重複底床接觸欄位、boundary／outcome 分母以及 aggregate
    manifest hash。它不是視覺學或海洋物理驗證；PNG 的教授級版面仍需人工開圖 QA。
    """

    payload, report_spec = exact_payload_and_report_spec
    destination = tmp_path / "synthetic.source-pathway-v1"
    final = _build_with_fake_reader(
        monkeypatch,
        payload=payload,
        report_spec=report_spec,
        aggregate_root=tmp_path / "aggregate.source-v1",
        destination=destination,
        mplconfigdir=mplconfigdir,
    )
    validation = source_pathway_module.validate_source_pathway_release(final)
    assert validation == {
        "valid": True,
        "errors": [],
        "summary": {
            "run_id": "report-statistics-integration",
            "run_kind": "synthetic",
            "experiment_case_id": "report-statistics-case",
            "members_per_scenario": 2,
            "site_count": 2,
            "file_count": 16,
        },
    }
    for site_id in ("site-a", "site-b"):
        site_root = final / "sites" / site_id
        assert {path.name for path in site_root.iterdir()} == {
            "figure.png",
            "figure.svg",
            "figure.pdf",
            "caption.json",
            "grid.parquet",
            "boundary.parquet",
            "outcomes.parquet",
        }
        grid = pq.read_table(site_root / "grid.parquet")
        assert {
            "bed_first_contact_count",
            "bed_repeated_contact_count",
            "primary_kde_status",
            "low_sample",
        }.issubset(grid.column_names)
        caption = json.loads((site_root / "caption.json").read_text(encoding="utf-8"))
        caption_by_panel = {row["panel"]: row for row in caption["panel_semantics"]}
        assert (
            "逆向 local_first_exit 在條件式解讀下對應正向潛在移入入口" in caption_by_panel["C"]["description"]
        )
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["provenance"]["aggregate_manifest_sha256"]) == 64
    assert manifest["pooling"]["design_balanced_claim"] is False
    assert "條件於本次情境設計與有效成員" in manifest["pooling"]["claim"]


def test_available_kde_and_positive_or_zero_settling_fail_closed(
    exact_payload_and_report_spec: tuple[object, object],
    mplconfigdir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """確認 KDE available 仍可產圖，且零／正沉降速度在 partial 前拒絕。

    測試把 synthetic ReportSpec 的 KDE raw-count 門檻降為 1，只為覆蓋 available layer
    的 renderer 分支；它不宣稱這個門檻是正式研究設計。第二段把一個 immutable stratum
    改成零沉降，source-pathway 應在任何輸出目錄建立前 fail closed；同一防線對正值由
    參數化案例覆蓋。
    """

    payload, report_spec = exact_payload_and_report_spec
    available_spec = replace(report_spec, minimum_kde_raw_count=1)
    final = _build_with_fake_reader(
        monkeypatch,
        payload=payload,
        report_spec=available_spec,
        aggregate_root=tmp_path / "aggregate-available.source-v1",
        destination=tmp_path / "available.source-pathway-v1",
        mplconfigdir=mplconfigdir,
    )
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert all(row["primary_kde_status"] == "available" for row in manifest["sites"])

    for velocity in (0.0, 0.0001):
        bad_payload = _payload_with_invalid_velocity(payload, velocity)
        with (
            patch.object(source_pathway_module, "read_aggregate_release", return_value=bad_payload),
            patch.object(source_pathway_module, "load_report_spec", return_value=report_spec),
            pytest.raises(ValueError, match="settling_velocity_mps < 0"),
        ):
            source_pathway_module.build_source_pathway_release(
                aggregate_release=tmp_path / f"bad-{velocity}",
                report_spec=tmp_path / "report.json",
                destination=tmp_path / f"bad-{velocity}.source-pathway-v1",
                mplconfigdir=mplconfigdir,
            )


def test_validator_rejects_tampered_hash_and_cli_routes(
    exact_payload_and_report_spec: tuple[object, object],
    mplconfigdir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """竄改 sidecar bytes 會失敗，兩個 CLI route 則輸出 JSON-safe 結果。"""

    payload, report_spec = exact_payload_and_report_spec
    final = _build_with_fake_reader(
        monkeypatch,
        payload=payload,
        report_spec=report_spec,
        aggregate_root=tmp_path / "aggregate-tamper.source-v1",
        destination=tmp_path / "tamper.source-pathway-v1",
        mplconfigdir=mplconfigdir,
    )
    manifest_path = final / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest.decode("utf-8"))
    # 以 canonical bytes 重寫，模擬攻擊者保留 JSON 格式與所有 file hash、只改 manifest
    # 語意；validator 不能只依檔案 inventory 通過，必須檢查 run_kind／evidence exact map。
    manifest["evidence_class"] = "formal_aggregate_evidence"
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )
    assert source_pathway_module.validate_source_pathway_release(final)["valid"] is False
    manifest_path.write_bytes(original_manifest)
    caption = final / "sites" / "site-a" / "caption.json"
    caption.write_bytes(caption.read_bytes() + b" ")
    invalid = source_pathway_module.validate_source_pathway_release(final)
    assert invalid["valid"] is False
    assert invalid["errors"]

    # CLI route 直接接到 production build／validator；只有 aggregate reader 與
    # ReportSpec loader 由 bounded fixture monkeypatch，因而會真的建立一份完整
    # source-pathway final，而不是以回傳固定路徑的 stub 取代成果包建置。
    monkeypatch.setattr(
        cli_module,
        "build_source_pathway_release",
        source_pathway_module.build_source_pathway_release,
    )
    cli_destination = tmp_path / "cli.source-pathway-v1"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font\(s\) .*\.",
            category=UserWarning,
        )
        assert (
            cli_module.main(
                [
                    "source-pathway-build",
                    "--aggregate-release",
                    str(tmp_path / "aggregate-tamper.source-v1"),
                    "--report-spec",
                    str(tmp_path / "report.json"),
                    "--destination",
                    str(cli_destination),
                    "--mplconfigdir",
                    str(mplconfigdir),
                ]
            )
            == 0
        )
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["valid"] is True
    assert cli_destination.is_dir()
    assert source_pathway_module.validate_source_pathway_release(cli_destination)["valid"] is True
