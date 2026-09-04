"""報告 renderer staging foundation 的可重現輸出契約測試。

測試只建立 synthetic ReportSpec、PyArrow Table、Matplotlib 小圖與 task-specific
MPLCONFIGDIR；不讀取 OCM、NWW3、trajectory 或 aggregate release，也不下載字型。
圖檔只驗證共同 renderer 邊界，例如三種 media bytes、300 dpi、日期 metadata、
caption provenance、exclusive staging 與 checksum；這些通過不代表任何 F01–F12
科學計算或正式 report release 已完成。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet
import pytest

from lagrangian_backtracking.report_render import (
    RenderedArtifact,
    ReportStagingRenderer,
)
from lagrangian_backtracking.report_spec import ReportSpec

_HASH = "a" * 64
_OTHER_HASH = "b" * 64


def _report_spec() -> ReportSpec:
    """建立固定 synthetic renderer spec，避免測試依賴任何資料來源。"""

    return ReportSpec(
        schema_version="1.0.0",
        run_id="synthetic-render-run",
        aggregate_spec_canonical_sha256=_HASH,
        primary_kde_bandwidth_m=100.0,
        minimum_kde_raw_count=1,
        low_sample_min_member_count=1,
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
        source_sha256=_OTHER_HASH,
        canonical_sha256="c" * 64,
    )


def _table() -> pa.Table:
    """建立含 typed nullable columns 與固定 row order 的 Arrow table。"""

    schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("label", pa.string(), nullable=True),
            pa.field("value_m", pa.float64(), nullable=True),
        ],
        metadata={b"source": b"synthetic"},
    )
    return pa.Table.from_arrays(
        [
            pa.array([2, 1], type=pa.int64()),
            pa.array(["second", None], type=pa.string()),
            pa.array([2.5, None], type=pa.float64()),
        ],
        schema=schema,
    )


def _record_metadata() -> dict[str, object]:
    """提供每個 available record 所需的 caller 明示 provenance。"""

    return {
        "title_zh": "合成 renderer 測試",
        "evidence_class": "synthetic_engineering_evidence",
        "input_sha256": {"source": _HASH, "spec": _OTHER_HASH},
        "raw_sample_count": 2,
        "denominator_name": "valid_member_count",
        "denominator_count": 2,
        "units": {"row_id": "1", "value_m": "m"},
        "crs_by_site": {"synthetic-site": "local_metric_test"},
        "limitations": ("僅供 renderer contract 測試",),
        "component_status": {"primary": "available"},
    }


@pytest.fixture
def fake_style_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    """設定既有 style 的 task-specific cache 與 fake font resolver。

    fake font 只提供既有 style contract 所需的 family、basename、檔案摘要與 glyph
    計數，不讀取或下載真實字型；ASCII 圖面足以測試輸出格式，而 caption 仍完整保存
    fake font provenance，讓測試與主機字型環境解耦。
    """

    mplconfigdir = tmp_path / "mplconfig"
    mplconfigdir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mplconfigdir))
    from lagrangian_backtracking import report_font

    selection = report_font.CJKFontSelection(
        family=report_font.CJK_FONT_CANDIDATES[0],
        filename="synthetic-font.ttf",
        file_sha256="d" * 64,
        required_glyph_count=len(report_font.REQUIRED_REPORT_GLYPHS),
    )
    monkeypatch.setattr(report_font, "resolve_cjk_font", lambda: selection)
    return SimpleNamespace(selection=selection, mplconfigdir=mplconfigdir)


def _render_kwargs() -> dict[str, object]:
    """建立 render API 的共同 explicit metadata，不讓 renderer 猜測統計欄位。"""

    return _record_metadata()


def _figure_builder(calls: list[int]) -> Callable[[], object]:
    """建立只畫 ASCII 線段的 callback，供測試 callback single-call contract。"""

    def build() -> object:
        calls[0] += 1
        import matplotlib.pyplot as pyplot

        figure, axis = pyplot.subplots(figsize=(2.0, 1.5))
        axis.plot([0.0, 1.0], [1.0, 2.0])
        axis.set_title("synthetic F01")
        axis.set_xlabel("x m")
        return figure

    return build


def _canonical_json(value: object) -> bytes:
    """重建 production canonical JSON bytes，驗證 sidecar 沒有格式漂移。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _render_figure(
    renderer: ReportStagingRenderer,
    *,
    artifact_id: str = "F01",
    placement: str = "main",
    calls: list[int] | None = None,
    data_table: pa.Table | object | None = None,
) -> RenderedArtifact:
    """以固定測試 metadata 建立一項圖面 artifact。"""

    call_counter = [0] if calls is None else calls
    metadata = _render_kwargs()
    return renderer.render_figure(
        artifact_id,
        placement,
        _figure_builder(call_counter),
        _table() if data_table is None else data_table,
        **metadata,
        caption_metadata={
            "caption": "固定 canonical metadata",
            "nullable_value": None,
            "display_unit": "m",
        },
    )


def _render_table(
    renderer: ReportStagingRenderer,
    *,
    artifact_id: str = "T01",
    table: pa.Table | object | None = None,
) -> RenderedArtifact:
    """以固定測試 metadata 建立一項表格 artifact。"""

    metadata = _render_kwargs()
    return renderer.render_table(
        artifact_id,
        _table() if table is None else table,
        **metadata,
        metadata={
            "table_purpose": "deterministic round trip",
            "nullable_value": None,
            "zero_is_real_value": 0,
        },
    )


def test_constructor_requires_empty_ordinary_root_and_creates_fixed_topology(
    tmp_path: Path,
) -> None:
    """root 必須是空 ordinary directory，成功後只出現固定 staging topology。"""

    root = tmp_path / "empty"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)

    assert renderer.state == "open"
    assert renderer.staging_root == root
    assert tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    ) == (
        "caption_sidecars",
        "data_sidecars",
        "figures",
        "figures/main",
        "figures/supplement",
        "tables",
    )
    for path in root.rglob("*"):
        metadata = os.lstat(path)
        assert stat.S_ISDIR(metadata.st_mode)
        assert not stat.S_ISLNK(metadata.st_mode)
    assert renderer.artifact_records == {}
    assert isinstance(renderer.artifact_records, MappingProxyType)


@pytest.mark.parametrize("root_kind", ["nonempty", "symlink", "regular_file"])
def test_constructor_rejects_nonempty_symlink_or_non_directory_root(
    tmp_path: Path,
    root_kind: str,
) -> None:
    """staging root gate 不會追隨 symlink，也不會清除或覆寫既有內容。"""

    root = tmp_path / "root"
    if root_kind == "nonempty":
        root.mkdir()
        (root / "existing").write_text("keep", encoding="utf-8")
    elif root_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    else:
        root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError):
        ReportStagingRenderer(_report_spec(), root)
    if root_kind == "nonempty":
        assert (root / "existing").read_text(encoding="utf-8") == "keep"


def test_rendered_figure_has_fixed_roles_readable_formats_and_real_checksums(
    fake_style_environment: SimpleNamespace,
    tmp_path: Path,
) -> None:
    """圖面同時輸出固定五項 products，格式可讀且 record checksum 來自實際 bytes。"""

    root = tmp_path / "figure-stage"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    calls = [0]
    artifact = _render_figure(renderer, calls=calls)

    assert calls == [1]
    assert artifact.record.artifact_id == "F01"
    assert artifact.record.artifact_kind == "figure"
    assert artifact.record.status == "available"
    assert tuple(product.role for product in artifact.record.products) == (
        "figure_png",
        "figure_svg",
        "figure_pdf",
        "caption_sidecar_json",
        "data_sidecar_parquet",
    )
    expected_paths = {
        "figures/main/F01.png",
        "figures/main/F01.svg",
        "figures/main/F01.pdf",
        "caption_sidecars/F01.json",
        "data_sidecars/F01.parquet",
    }
    assert set(artifact.staging_paths) == expected_paths
    assert set(artifact.staging_paths) == {
        product.relative_path for product in artifact.record.products
    }

    products_by_path = {
        product.relative_path: product for product in artifact.record.products
    }
    for relative_path, path in artifact.staging_paths.items():
        metadata = os.lstat(path)
        assert stat.S_ISREG(metadata.st_mode)
        assert not stat.S_ISLNK(metadata.st_mode)
        raw_bytes = path.read_bytes()
        product = products_by_path[relative_path]
        assert product.size_bytes == len(raw_bytes)
        assert product.sha256 == hashlib.sha256(raw_bytes).hexdigest()

    from PIL import Image

    with Image.open(artifact.staging_paths["figures/main/F01.png"]) as image:
        assert image.format == "PNG"
        assert image.info["dpi"][0] == pytest.approx(300.0, abs=0.5)
    assert artifact.staging_paths["figures/main/F01.svg"].read_bytes().startswith(
        b"<?xml"
    )
    assert artifact.staging_paths["figures/main/F01.pdf"].read_bytes().startswith(b"%PDF")
    assert pa_parquet.read_table(
        artifact.staging_paths["data_sidecars/F01.parquet"]
    ).equals(_table())

    caption_path = artifact.staging_paths["caption_sidecars/F01.json"]
    caption_document = json.loads(caption_path.read_text(encoding="utf-8"))
    assert caption_path.read_bytes() == _canonical_json(caption_document)
    assert caption_document["renderer_provenance"]["raster_dpi"] == 300
    assert (
        caption_document["renderer_provenance"]["font_selection"]["file_sha256"]
        == "d" * 64
    )
    assert caption_document["caller_metadata"]["nullable_value"] is None
    assert str(root) not in caption_path.read_text(encoding="utf-8")
    assert str(root) not in repr(artifact.record.to_dict())

    svg_bytes = artifact.staging_paths["figures/main/F01.svg"].read_bytes().lower()
    pdf_bytes = artifact.staging_paths["figures/main/F01.pdf"].read_bytes().lower()
    png_bytes = artifact.staging_paths["figures/main/F01.png"].read_bytes().lower()
    assert b"<dc:date" not in svg_bytes
    assert b"creationdate" not in svg_bytes
    assert b"/creationdate" not in pdf_bytes
    assert b"/moddate" not in pdf_bytes
    assert b"date" not in png_bytes
    assert "MPLCONFIGDIR" not in repr(artifact.record.to_dict())


def test_render_figure_style_callback_is_called_once_and_failure_closes_renderer(
    fake_style_environment: SimpleNamespace,
    tmp_path: Path,
) -> None:
    """callback 只呼叫一次；callback 錯誤後 renderer 永久 failed 且不留 partial files。"""

    root = tmp_path / "callback-stage"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    calls = [0]

    def failing_builder() -> object:
        calls[0] += 1
        raise RuntimeError("synthetic callback failure")

    metadata = _render_kwargs()
    with pytest.raises(RuntimeError, match="synthetic callback failure"):
        renderer.render_figure(
            "F01",
            "main",
            failing_builder,
            _table(),
            **metadata,
        )
    assert calls == [1]
    assert renderer.state == "failed"
    assert tuple(path for path in root.rglob("*") if path.is_file()) == ()
    with pytest.raises(ValueError, match="永久關閉"):
        _render_table(renderer)


def test_rendered_artifact_mapping_is_defensive_and_rechecks_tampering(
    tmp_path: Path,
) -> None:
    """staging mapping 是唯讀 snapshot，任何 missing/key/checksum/symlink 都 fail closed。"""

    root = tmp_path / "mapping-stage"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    artifact = _render_table(renderer)

    supplied_paths = dict(artifact.staging_paths)
    copied = RenderedArtifact(artifact.record, supplied_paths)
    supplied_paths.clear()
    assert set(copied.staging_paths) == set(artifact.staging_paths)
    assert isinstance(copied.staging_paths, MappingProxyType)
    with pytest.raises(TypeError):
        copied.staging_paths["extra"] = root / "extra"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        copied.record = artifact.record  # type: ignore[misc]

    with pytest.raises(ValueError, match="paths"):
        RenderedArtifact(artifact.record, {})

    tampered = artifact.staging_paths["tables/T01.csv"]
    original = tampered.read_bytes()
    tampered.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="size_bytes|sha256"):
        RenderedArtifact(artifact.record, artifact.staging_paths)

    tampered.write_bytes(original)
    tampered.unlink()
    tampered.symlink_to(root / "tables/T01.parquet")
    with pytest.raises(ValueError, match="ordinary"):
        RenderedArtifact(artifact.record, artifact.staging_paths)


def test_render_table_parquet_csv_roundtrip_preserves_null_and_deterministic_order(
    tmp_path: Path,
) -> None:
    """表格輸出保留 Arrow schema/nullability；CSV 維持 caller row order 且只作交換。"""

    root = tmp_path / "table-stage"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    artifact = _render_table(renderer)

    assert tuple(product.role for product in artifact.record.products) == (
        "table_parquet",
        "table_csv",
        "metadata_sidecar_json",
    )
    parquet_path = artifact.staging_paths["tables/T01.parquet"]
    parquet_table = pa_parquet.read_table(parquet_path)
    assert parquet_table.column_names == ["row_id", "label", "value_m"]
    assert parquet_table["row_id"].to_pylist() == [2, 1]
    assert parquet_table["label"].to_pylist() == ["second", None]
    assert parquet_table["value_m"].to_pylist() == [2.5, None]
    assert parquet_table.schema.field("label").nullable is True
    assert parquet_table.schema.field("value_m").nullable is True

    csv_path = artifact.staging_paths["tables/T01.csv"]
    csv_bytes = csv_path.read_bytes()
    assert csv_bytes.index(b"2") < csv_bytes.index(b"1")
    exchanged = pa_csv.read_csv(csv_path)
    assert exchanged["row_id"].to_pylist() == [2, 1]

    metadata_path = artifact.staging_paths["data_sidecars/T01.json"]
    metadata_document = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_path.read_bytes() == _canonical_json(metadata_document)
    columns = metadata_document["table_schema"]["columns"]
    assert columns == [
        {
            "name": "row_id",
            "type": "int64",
            "nullable": False,
            "metadata": {},
        },
        {
            "name": "label",
            "type": "string",
            "nullable": True,
            "metadata": {},
        },
        {
            "name": "value_m",
            "type": "double",
            "nullable": True,
            "metadata": {},
        },
    ]
    assert metadata_document["units"] == {"row_id": "1", "value_m": "m"}
    assert metadata_document["caller_metadata"]["nullable_value"] is None
    assert metadata_document["caller_metadata"]["zero_is_real_value"] == 0
    assert metadata_document["null_policy"]["parquet"] == (
        "authoritative_preserve_arrow_nullability"
    )
    assert str(root) not in metadata_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("kind", "bad_id"),
    [
        ("figure", "T01"),
        ("table", "F01"),
    ],
)
def test_renderer_rejects_wrong_ft_ids_and_non_available_status(
    tmp_path: Path,
    kind: str,
    bad_id: str,
) -> None:
    """figure/table ID 與 unavailable status 不得被 renderer 模糊接受。"""

    root = tmp_path / f"bad-{kind}"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    metadata = _render_kwargs()
    if kind == "figure":
        with pytest.raises(ValueError, match="artifact_id"):
            renderer.render_figure(
                bad_id,
                "main",
                lambda: None,
                _table(),
                **metadata,
            )
    else:
        with pytest.raises(ValueError, match="artifact_id"):
            renderer.render_table(bad_id, _table(), **metadata)
    assert renderer.state == "failed"

    second_root = tmp_path / f"status-{kind}"
    second_root.mkdir()
    second = ReportStagingRenderer(_report_spec(), second_root)
    if kind == "figure":
        with pytest.raises(ValueError, match="available"):
            second.render_figure(
                "F01",
                "main",
                lambda: None,
                _table(),
                **metadata,
                status="unavailable_missing_comparison",
            )
    else:
        with pytest.raises(ValueError, match="available"):
            second.render_table(
                "T01",
                _table(),
                **metadata,
                status="unavailable_missing_validation_evidence",
            )
    assert second.state == "failed"


def test_existing_target_duplicate_id_and_overwrite_are_fail_closed(
    tmp_path: Path,
) -> None:
    """既有 target、重複 artifact ID 與前次成功 path 都不得覆寫。"""

    root = tmp_path / "overwrite-stage"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    first = _render_table(renderer)
    first_bytes = first.staging_paths["tables/T01.parquet"].read_bytes()

    with pytest.raises(ValueError, match="重複"):
        _render_table(renderer, artifact_id="T01")
    assert renderer.state == "failed"
    assert first.staging_paths["tables/T01.parquet"].read_bytes() == first_bytes
    with pytest.raises(ValueError, match="永久關閉"):
        _render_table(renderer, artifact_id="T02")

    preexisting_root = tmp_path / "preexisting-target-stage"
    preexisting_root.mkdir()
    preexisting = ReportStagingRenderer(_report_spec(), preexisting_root)
    target = preexisting_root / "tables" / "T01.parquet"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        _render_table(preexisting)
    assert target.read_bytes() == b"keep"
    assert preexisting.state == "failed"


@pytest.mark.parametrize(
    ("kind", "artifact_id"),
    [
        ("figure", "F11"),
        ("figure", "F12"),
        ("table", "T05"),
        ("table", "T06"),
    ],
)
def test_comparison_validation_zero_row_placeholder_is_not_created(
    tmp_path: Path,
    fake_style_environment: SimpleNamespace,
    kind: str,
    artifact_id: str,
) -> None:
    """comparison/validation 缺證不能藉 zero-row figure/table 冒充可用成果。"""

    root = tmp_path / f"zero-{kind}-{artifact_id}"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    empty_table = pa.table({"placeholder": pa.array([], type=pa.int64())})
    metadata = _render_kwargs()
    if kind == "figure":
        calls = [0]
        with pytest.raises(ValueError, match="zero-row"):
            renderer.render_figure(
                artifact_id,
                "supplement",
                _figure_builder(calls),
                empty_table,
                **metadata,
            )
        assert calls == [0]
    else:
        with pytest.raises(ValueError, match="zero-row"):
            renderer.render_table(artifact_id, empty_table, **metadata)
    assert renderer.state == "failed"
    assert tuple(path for path in root.rglob("*") if path.is_file()) == ()


def test_table_rejects_non_arrow_input_and_failed_renderer_cannot_continue(
    tmp_path: Path,
) -> None:
    """exact PyArrow Table gate 與永久 failed 狀態不能被 duck typing 繞過。"""

    root = tmp_path / "type-stage"
    root.mkdir()
    renderer = ReportStagingRenderer(_report_spec(), root)
    with pytest.raises(TypeError, match="exact PyArrow Table"):
        _render_table(renderer, table={"row_id": [1]})
    assert renderer.state == "failed"
    with pytest.raises(ValueError, match="永久關閉"):
        _render_table(renderer, artifact_id="T02")
