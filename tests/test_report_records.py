"""報告成果 records 的 immutable contract 測試。

本檔只建立記憶體中的假產品 metadata，不讀取或寫入任何報告、圖表、表格或
SERVER 檔案。產品 path、大小、媒體型別與 SHA-256 都是 synthetic engineering
fixture；它們用來驗證 ``report_records`` 的型別、閉包、缺證與 provenance binding，
不是 OCM schema 3、NWW3 schema 1 的科學成果，也不是條件式來源足跡、相對來源權重
或絕對來源機率。
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from lagrangian_backtracking.report_records import (
    REPORT_COMPARISON_ARTIFACT_IDS,
    REPORT_CORE_ARTIFACT_IDS,
    REPORT_FIGURE_IDS,
    REPORT_RELEASE_SCHEMA_VERSION,
    REPORT_TABLE_IDS,
    REPORT_VALIDATION_ARTIFACT_IDS,
    ReportArtifactRecord,
    ReportProductRef,
    ReportRegistry,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64

# 這份 mapping 刻意在測試中重述 public product contract，而不是讀取 production
# private constant。每個 role 都固定相對資料夾、副檔名與 media type；假產品沒有內容
# bytes，size/hash 僅代表 caller 將來必須提供的已驗證 I/O metadata。
_ROLE_CONTRACTS: dict[str, tuple[str, str, str]] = {
    "figure_png": ("figures/", ".png", "image/png"),
    "figure_svg": ("figures/", ".svg", "image/svg+xml"),
    "figure_pdf": ("figures/", ".pdf", "application/pdf"),
    "table_parquet": ("tables/", ".parquet", "application/vnd.apache.parquet"),
    "table_csv": ("tables/", ".csv", "text/csv"),
    "caption_sidecar_json": ("caption_sidecars/", ".json", "application/json"),
    "data_sidecar_parquet": (
        "data_sidecars/",
        ".parquet",
        "application/vnd.apache.parquet",
    ),
    "data_sidecar_npy": ("data_sidecars/", ".npy", "application/x-npy"),
    "metadata_sidecar_json": ("data_sidecars/", ".json", "application/json"),
}


def _product(role: str, stem: str, *, digest: str = _HASH) -> ReportProductRef:
    """建立唯一 synthetic product reference，不接觸實際檔案。

    ``stem`` 會包含 artifact ID，讓 registry 測試可分辨「同一 artifact 內 role
    unique」與「整個 registry 內 relative_path globally unique」兩層契約。所有大小
    都是正的原生 bytes 計數；digest 只是固定格式的假 SHA-256。
    """

    prefix, suffix, media_type = _ROLE_CONTRACTS[role]
    return ReportProductRef(
        relative_path=f"{prefix}{stem}{suffix}",
        role=role,
        media_type=media_type,
        size_bytes=1,
        sha256=digest,
    )


def _figure_products(artifact_id: str, *, data_role: str = "data_sidecar_parquet") -> tuple[
    ReportProductRef, ...
]:
    """建立 figure 所需的 PNG/SVG/PDF/caption 與一個 parquet 或 NPY sidecar。"""

    return (
        _product("figure_png", f"{artifact_id}-figure"),
        _product("figure_svg", f"{artifact_id}-figure"),
        _product("figure_pdf", f"{artifact_id}-figure"),
        _product("caption_sidecar_json", f"{artifact_id}-caption"),
        _product(data_role, f"{artifact_id}-data"),
    )


def _table_products(artifact_id: str) -> tuple[ReportProductRef, ...]:
    """建立 table 所需的 Parquet、CSV 與 metadata JSON 三件產品。"""

    return (
        _product("table_parquet", f"{artifact_id}-table"),
        _product("table_csv", f"{artifact_id}-table"),
        _product("metadata_sidecar_json", f"{artifact_id}-metadata"),
    )


def _available_record(
    artifact_id: str,
    *,
    evidence_class: str = "synthetic_engineering_evidence",
    data_role: str = "data_sidecar_parquet",
    products: tuple[ReportProductRef, ...] | None = None,
    input_sha256: dict[str, str] | None = None,
    raw_sample_count: int | None = 12,
    denominator_name: str | None = "synthetic_particles",
    denominator_count: int | None = 12,
    units: dict[str, str] | None = None,
) -> ReportArtifactRecord:
    """建立一列具備完整 provenance 的 available synthetic artifact。

    ``raw_sample_count``、denominator 與 units 都明示保存工程統計的語意；它們不是由
    測試假設成真實觀測樣本。不同 artifact 的 product path 由 ID 區分，避免 helper
    本身無意中破壞 registry 的全域 path closure。
    """

    kind = "figure" if artifact_id.startswith("F") else "table"
    if products is None:
        products = (
            _figure_products(artifact_id, data_role=data_role)
            if kind == "figure"
            else _table_products(artifact_id)
        )
    return ReportArtifactRecord(
        artifact_id=artifact_id,
        artifact_kind=kind,
        title_zh=f"synthetic {artifact_id}",
        status="available",
        evidence_class=evidence_class,
        products=products,
        input_sha256={"source": _HASH} if input_sha256 is None else input_sha256,
        raw_sample_count=raw_sample_count,
        denominator_name=denominator_name,
        denominator_count=denominator_count,
        units={"count": "1"} if units is None else units,
        crs_by_site={"synthetic-site": "EPSG:32651"},
        limitations=("僅供 synthetic engineering contract test",),
        unavailable_reason_code=None,
        unavailable_reason_zh=None,
        component_status={"primary": "available"},
    )


def _unavailable_record(
    artifact_id: str,
    *,
    status: str,
    reason_code: str,
    evidence_class: str = "synthetic_engineering_evidence",
) -> ReportArtifactRecord:
    """建立沒有 products、樣本／denominator 均缺值且帶固定 reason 的缺證列。

    缺證 row 仍須和整個 registry 使用同一 evidence class；這讓 formal baseline 的
    成對缺證與 synthetic engineering 缺證各自保留 provenance 邊界，而不是把缺證
    row 偷標成另一種證據再讓 registry 放行。
    """

    kind = "figure" if artifact_id.startswith("F") else "table"
    return ReportArtifactRecord(
        artifact_id=artifact_id,
        artifact_kind=kind,
        title_zh=f"synthetic {artifact_id} unavailable",
        status=status,
        evidence_class=evidence_class,
        products=(),
        input_sha256={},
        raw_sample_count=None,
        denominator_name=None,
        denominator_count=None,
        units={},
        crs_by_site={"synthetic-site": "EPSG:32651"},
        limitations=(f"缺少 {evidence_class} evidence",),
        unavailable_reason_code=reason_code,
        unavailable_reason_zh="本 synthetic 測試刻意保留缺證狀態",
        component_status={"primary": "not_applicable_by_registered_design"},
    )


def _record_kwargs(record: ReportArtifactRecord) -> dict[str, object]:
    """複製 record 欄位供 subclass fixture 使用，避免依賴 dataclass repr。"""

    return {
        "artifact_id": record.artifact_id,
        "artifact_kind": record.artifact_kind,
        "title_zh": record.title_zh,
        "status": record.status,
        "evidence_class": record.evidence_class,
        "products": record.products,
        "input_sha256": record.input_sha256,
        "raw_sample_count": record.raw_sample_count,
        "denominator_name": record.denominator_name,
        "denominator_count": record.denominator_count,
        "units": record.units,
        "crs_by_site": record.crs_by_site,
        "limitations": record.limitations,
        "unavailable_reason_code": record.unavailable_reason_code,
        "unavailable_reason_zh": record.unavailable_reason_zh,
        "component_status": record.component_status,
    }


def _registry_records(
    *,
    evidence_class: str = "synthetic_engineering_evidence",
    missing_comparison: bool = False,
    missing_validation: bool = False,
    figure_overrides: dict[str, ReportArtifactRecord] | None = None,
    table_overrides: dict[str, ReportArtifactRecord] | None = None,
) -> tuple[tuple[ReportArtifactRecord, ...], tuple[ReportArtifactRecord, ...]]:
    """建立固定 F/T closure，並只在指定 optional pair 注入缺證 row。"""

    figure_overrides = {} if figure_overrides is None else figure_overrides
    table_overrides = {} if table_overrides is None else table_overrides
    figures: list[ReportArtifactRecord] = []
    tables: list[ReportArtifactRecord] = []
    for artifact_id in REPORT_FIGURE_IDS:
        if artifact_id in figure_overrides:
            record = figure_overrides[artifact_id]
        elif missing_comparison and artifact_id == "F11":
            record = _unavailable_record(
                artifact_id,
                status="unavailable_missing_comparison",
                reason_code="missing_comparison",
                evidence_class=evidence_class,
            )
        elif missing_validation and artifact_id == "F12":
            record = _unavailable_record(
                artifact_id,
                status="unavailable_missing_validation_evidence",
                reason_code="missing_validation_evidence",
                evidence_class=evidence_class,
            )
        else:
            record = _available_record(artifact_id, evidence_class=evidence_class)
        figures.append(record)
    for artifact_id in REPORT_TABLE_IDS:
        if artifact_id in table_overrides:
            record = table_overrides[artifact_id]
        elif missing_comparison and artifact_id == "T05":
            record = _unavailable_record(
                artifact_id,
                status="unavailable_missing_comparison",
                reason_code="missing_comparison",
                evidence_class=evidence_class,
            )
        elif missing_validation and artifact_id == "T06":
            record = _unavailable_record(
                artifact_id,
                status="unavailable_missing_validation_evidence",
                reason_code="missing_validation_evidence",
                evidence_class=evidence_class,
            )
        else:
            record = _available_record(artifact_id, evidence_class=evidence_class)
        tables.append(record)
    return tuple(figures), tuple(tables)


def _registry(
    *,
    schema_version: str = REPORT_RELEASE_SCHEMA_VERSION,
    run_id: str = "synthetic-report-run",
    run_kind: str = "pilot",
    evidence_class: str = "synthetic_engineering_evidence",
    aggregate_manifest_sha256: str = _HASH,
    source_run_plan_sha256: str = _HASH,
    source_run_progress_sha256: str = _HASH,
    config_hash: str = _HASH,
    checkpoint_input_binding_hash: str = _HASH,
    allow_missing_comparison: bool = False,
    allow_missing_validation_evidence: bool = False,
    missing_comparison: bool = False,
    missing_validation: bool = False,
    figures: tuple[ReportArtifactRecord, ...] | None = None,
    tables: tuple[ReportArtifactRecord, ...] | None = None,
    figure_overrides: dict[str, ReportArtifactRecord] | None = None,
    table_overrides: dict[str, ReportArtifactRecord] | None = None,
) -> ReportRegistry:
    """建立完整 registry fixture；所有 digest 都是格式合法但無科學含意的假值。"""

    if figures is None or tables is None:
        default_figures, default_tables = _registry_records(
            evidence_class=evidence_class,
            missing_comparison=missing_comparison,
            missing_validation=missing_validation,
            figure_overrides=figure_overrides,
            table_overrides=table_overrides,
        )
        figures = default_figures if figures is None else figures
        tables = default_tables if tables is None else tables
    return ReportRegistry(
        schema_version=schema_version,
        run_id=run_id,
        run_kind=run_kind,
        experiment_case_id="synthetic-case",
        evidence_class=evidence_class,
        aggregate_manifest_sha256=aggregate_manifest_sha256,
        source_run_plan_sha256=source_run_plan_sha256,
        source_run_progress_sha256=source_run_progress_sha256,
        config_hash=config_hash,
        checkpoint_input_binding_hash=checkpoint_input_binding_hash,
        allow_missing_comparison=allow_missing_comparison,
        allow_missing_validation_evidence=allow_missing_validation_evidence,
        figures=figures,
        tables=tables,
    )


@pytest.mark.parametrize("role, prefix_suffix_media", tuple(_ROLE_CONTRACTS.items()))
def test_report_product_ref_accepts_every_public_role_contract(
    role: str,
    prefix_suffix_media: tuple[str, str, str],
) -> None:
    """九種 public role 都必須綁定正確 prefix、suffix、media type 與 checksum 格式。"""

    prefix, suffix, media_type = prefix_suffix_media
    product = ReportProductRef(
        relative_path=f"{prefix}accepted{suffix}",
        role=role,
        media_type=media_type,
        size_bytes=1,
        sha256=_HASH,
    )

    assert product.to_dict() == {
        "relative_path": f"{prefix}accepted{suffix}",
        "role": role,
        "media_type": media_type,
        "size_bytes": 1,
        "sha256": _HASH,
    }
    assert json.dumps(product.to_dict(), ensure_ascii=False, sort_keys=True)


@pytest.mark.parametrize(
    "relative_path",
    (
        "/figures/absolute.png",
        "figures\\backslash.png",
        "figures/control\x01.png",
        "figures/./dot.png",
        "figures/../parent.png",
        "figures//empty.png",
    ),
)
def test_report_product_ref_rejects_unsafe_relative_path_components(relative_path: str) -> None:
    """產品 path 不得把 absolute、Windows separator、控制字元或 traversal 帶進 release。"""

    with pytest.raises((TypeError, ValueError)):
        ReportProductRef(
            relative_path=relative_path,
            role="figure_png",
            media_type="image/png",
            size_bytes=1,
            sha256=_HASH,
        )


@pytest.mark.parametrize(
    "relative_path, role, media_type",
    (
        ("figures/wrong.svg", "figure_png", "image/png"),
        ("figures/right.png", "figure_png", "image/jpeg"),
        ("figures/right.jpg", "figure_jpg", "image/jpeg"),
        ("figures/right.png", "unknown_role", "image/png"),
        ("tables/right.csv", "table_parquet", "application/vnd.apache.parquet"),
    ),
)
def test_report_product_ref_rejects_wrong_role_suffix_media_or_unknown_role(
    relative_path: str,
    role: str,
    media_type: str,
) -> None:
    """role、suffix 與 media type 是同一產品 contract，不得各自宣告互相矛盾的值。"""

    with pytest.raises((TypeError, ValueError)):
        ReportProductRef(
            relative_path=relative_path,
            role=role,
            media_type=media_type,
            size_bytes=1,
            sha256=_HASH,
        )


@pytest.mark.parametrize("size_bytes", (0, True, np.int64(1), np.float64(1.0)))
def test_report_product_ref_rejects_nonpositive_or_nonpython_size(size_bytes: object) -> None:
    """size 必須是正的 Python int，不能用零、bool 或 NumPy scalar 代表 bytes。"""

    with pytest.raises((TypeError, ValueError)):
        ReportProductRef(
            relative_path="figures/invalid.png",
            role="figure_png",
            media_type="image/png",
            size_bytes=size_bytes,
            sha256=_HASH,
        )


@pytest.mark.parametrize(
    "sha256",
    (
        "A" * 64,
        "a" * 63,
        "g" * 64,
        "a" * 65,
        b"a" * 64,
    ),
)
def test_report_product_ref_rejects_bad_sha256(sha256: object) -> None:
    """sha 必須是完整小寫 64 碼十六進位文字，不能混入大小寫或其他型別。"""

    with pytest.raises((TypeError, ValueError)):
        ReportProductRef(
            relative_path="figures/invalid-sha.png",
            role="figure_png",
            media_type="image/png",
            size_bytes=1,
            sha256=sha256,
        )


def test_report_product_ref_is_frozen_and_to_dict_is_plain_json() -> None:
    """product record 不得被 caller 改寫，序列化結果也不得暴露 immutable wrapper。"""

    product = _product("figure_png", "frozen")
    with pytest.raises(AttributeError):
        product.size_bytes = 2  # type: ignore[misc]
    encoded = product.to_dict()
    assert type(encoded) is dict
    assert json.loads(json.dumps(encoded)) == encoded


@pytest.mark.parametrize(
    "artifact_id, expected_kind",
    (("F01", "figure"), ("T01", "table")),
)
def test_report_artifact_record_enforces_exact_f_or_t_kind(
    artifact_id: str,
    expected_kind: str,
) -> None:
    """F/T ID 與 artifact_kind 必須一一對應，避免 renderer 以錯誤拓樸解讀產品。"""

    record = _available_record(artifact_id)
    assert record.artifact_kind == expected_kind

    wrong_kind = "table" if expected_kind == "figure" else "figure"
    with pytest.raises(ValueError):
        replace(record, artifact_kind=wrong_kind)


@pytest.mark.parametrize("data_role", ("data_sidecar_parquet", "data_sidecar_npy"))
@pytest.mark.parametrize(
    "missing_role",
    ("figure_png", "figure_svg", "figure_pdf", "caption_sidecar_json"),
)
def test_available_figure_requires_all_visual_products_caption_and_data_sidecar(
    data_role: str,
    missing_role: str,
) -> None:
    """figure available 必須具備三種視覺格式、caption JSON 與 parquet/NPY data sidecar。"""

    record = _available_record("F01", data_role=data_role)
    assert {product.role for product in record.products} == {
        "figure_png",
        "figure_svg",
        "figure_pdf",
        "caption_sidecar_json",
        data_role,
    }

    products = [product for product in record.products if product.role != missing_role]
    with pytest.raises(ValueError):
        replace(record, products=products)

    products = [product for product in record.products if product.role != data_role]
    with pytest.raises(ValueError):
        replace(record, products=products)


@pytest.mark.parametrize("missing_role", ("table_parquet", "table_csv", "metadata_sidecar_json"))
def test_available_table_requires_parquet_csv_and_metadata_json(missing_role: str) -> None:
    """table available 必須同時留下可重算的 Parquet、CSV 與 schema metadata JSON。"""

    record = _available_record("T01")
    assert {product.role for product in record.products} == {
        "table_parquet",
        "table_csv",
        "metadata_sidecar_json",
    }
    with pytest.raises(ValueError):
        replace(
            record,
            products=tuple(product for product in record.products if product.role != missing_role),
        )


@pytest.mark.parametrize(
    "tampered_products",
    (
        lambda products: products + (_product("data_sidecar_parquet", "F01-extra"),),
        lambda products: products + (products[0],),
    ),
)
def test_report_artifact_record_rejects_duplicate_product_role_or_path(
    tampered_products: Any,
) -> None:
    """單一 artifact 的 role/path 都必須 unique，避免同一產品被多重解讀或覆寫。"""

    record = _available_record("F01")
    products = tuple(tampered_products(record.products))
    with pytest.raises((TypeError, ValueError)):
        replace(record, products=products)


@pytest.mark.parametrize(
    "field, value",
    (
        ("input_sha256", {}),
        ("raw_sample_count", None),
        ("denominator_name", None),
        ("denominator_count", None),
        ("units", {}),
    ),
)
def test_available_artifact_requires_nonempty_input_sample_denominator_and_units(
    field: str,
    value: object,
) -> None:
    """available 列不可省略輸入雜湊、樣本計數、denominator 或單位描述。"""

    record = _available_record("F01")
    kwargs: dict[str, object] = {field: value}
    if field in {"denominator_name", "denominator_count"}:
        # denominator 是成對欄位；任一缺值都應被視為不可用的 available provenance。
        kwargs["denominator_name"] = None
        kwargs["denominator_count"] = None
    with pytest.raises((TypeError, ValueError)):
        replace(record, **kwargs)


@pytest.mark.parametrize(
    "field, value",
    (
        ("raw_sample_count", True),
        ("raw_sample_count", np.int64(1)),
        ("denominator_count", True),
        ("denominator_count", np.int64(1)),
        ("input_sha256", {"source": np.str_("a" * 64)}),
        ("units", {"count": np.str_("1")}),
    ),
)
def test_report_artifact_record_rejects_bool_numpy_scalars_and_tampered_field_types(
    field: str,
    value: object,
) -> None:
    """計數與文字 mapping 不得用 bool／NumPy scalar 偷渡非 canonical JSON 型別。"""

    record = _available_record("F01")
    kwargs: dict[str, object] = {field: value}
    if field == "denominator_count":
        kwargs["denominator_name"] = "synthetic_particles"
    with pytest.raises((TypeError, ValueError)):
        replace(record, **kwargs)


def test_unavailable_artifact_has_no_products_or_sample_denominator_and_has_reason() -> None:
    """缺證列必須以空 products、三個 None 與 reason 明示缺失，不得偽裝為空結果。"""

    record = _unavailable_record(
        "F11",
        status="unavailable_missing_comparison",
        reason_code="missing_comparison",
    )
    assert record.products == ()
    assert record.raw_sample_count is None
    assert record.denominator_name is None
    assert record.denominator_count is None
    assert record.unavailable_reason_code == "missing_comparison"
    assert record.unavailable_reason_zh

    with pytest.raises(ValueError):
        replace(record, unavailable_reason_zh=None)
    with pytest.raises(ValueError):
        replace(record, products=_table_products("F11"))

    for field_kwargs in (
        {"raw_sample_count": 1},
        {"denominator_name": "synthetic_particles", "denominator_count": 1},
    ):
        with pytest.raises(ValueError):
            replace(record, **field_kwargs)


@pytest.mark.parametrize(
    "artifact_id, status, reason_code",
    (
        ("F01", "unavailable_missing_comparison", "missing_comparison"),
        ("T01", "unavailable_missing_validation_evidence", "missing_validation_evidence"),
    ),
)
def test_comparison_and_validation_missing_statuses_only_belong_to_fixed_ids(
    artifact_id: str,
    status: str,
    reason_code: str,
) -> None:
    """comparison／validation 缺證狀態只能出現在固定 F11/T05 與 F12/T06。"""

    with pytest.raises(ValueError):
        _unavailable_record(artifact_id, status=status, reason_code=reason_code)


def test_report_artifact_record_defensively_snapshots_nested_inputs_and_serializes() -> None:
    """record 必須複製 mapping/list，對外只提供 MappingProxy、tuple 與 plain JSON。"""

    products = list(_figure_products("F01"))
    input_sha256 = {"source": _HASH}
    units = {"count": "1"}
    crs_by_site = {"synthetic-site": "EPSG:32651"}
    limitations = ["synthetic only"]
    component_status = {"primary": "available"}
    record = ReportArtifactRecord(
        artifact_id="F01",
        artifact_kind="figure",
        title_zh="synthetic defensive snapshot",
        status="available",
        evidence_class="synthetic_engineering_evidence",
        products=products,
        input_sha256=input_sha256,
        raw_sample_count=12,
        denominator_name="synthetic_particles",
        denominator_count=12,
        units=units,
        crs_by_site=crs_by_site,
        limitations=limitations,
        unavailable_reason_code=None,
        unavailable_reason_zh=None,
        component_status=component_status,
    )
    products.clear()
    input_sha256["new"] = _OTHER_HASH
    units["mass"] = "kg"
    crs_by_site["other"] = "EPSG:4326"
    limitations.append("caller mutation")
    component_status["primary"] = "not_applicable_by_registered_design"

    assert len(record.products) == 5
    assert isinstance(record.input_sha256, MappingProxyType)
    assert isinstance(record.units, MappingProxyType)
    assert isinstance(record.crs_by_site, MappingProxyType)
    assert isinstance(record.component_status, MappingProxyType)
    assert isinstance(record.products, tuple)
    assert isinstance(record.limitations, tuple)
    assert dict(record.input_sha256) == {"source": _HASH}
    assert dict(record.units) == {"count": "1"}
    assert record.limitations == ("synthetic only",)

    encoded = record.to_dict()
    assert type(encoded) is dict
    json_text = json.dumps(encoded, ensure_ascii=False, sort_keys=True)
    assert "MappingProxyType" not in json_text
    assert json.loads(json_text) == encoded


def test_report_artifact_record_rejects_product_subclass_and_duck_object() -> None:
    """products 必須是 exact ReportProductRef，不能用 subclass 或形似 record 的 duck object。"""

    class ProductSubclass(ReportProductRef):
        """只供測試 nominal type gate 的 synthetic subclass。"""

    subclass_product = ProductSubclass(
        relative_path="data_sidecars/subclass.parquet",
        role="data_sidecar_parquet",
        media_type="application/vnd.apache.parquet",
        size_bytes=1,
        sha256=_HASH,
    )
    products = list(_figure_products("F01"))
    products[-1] = subclass_product
    with pytest.raises(TypeError):
        _available_record("F01", products=tuple(products))

    class ProductDuck:
        """具有相似屬性但不是 contract class 的測試物件。"""

        role = "data_sidecar_parquet"
        relative_path = "data_sidecars/duck.parquet"

    products[-1] = ProductDuck()  # type: ignore[assignment]
    with pytest.raises(TypeError):
        _available_record("F01", products=tuple(products))


def test_report_registry_validates_exact_closure_order_evidence_and_json_boundary() -> None:
    """合法 synthetic registry 必須固定 F01-F12/T01-T06 順序、證據與 JSON 邊界。"""

    registry = _registry()
    assert tuple(record.artifact_id for record in registry.figures) == REPORT_FIGURE_IDS
    assert tuple(record.artifact_id for record in registry.tables) == REPORT_TABLE_IDS
    assert REPORT_FIGURE_IDS[:10] + REPORT_TABLE_IDS[:4] == REPORT_CORE_ARTIFACT_IDS
    assert all(record.status == "available" for record in registry.figures[:10])
    assert all(record.status == "available" for record in registry.tables[:4])
    assert all(
        record.evidence_class == "synthetic_engineering_evidence"
        for record in (*registry.figures, *registry.tables)
    )
    encoded = registry.to_dict()
    assert type(encoded) is dict
    json_text = json.dumps(encoded, ensure_ascii=False, sort_keys=True)
    assert "MappingProxyType" not in json_text
    assert json.loads(json_text) == encoded
    assert "Path(" not in json_text


def test_report_registry_rejects_nonavailable_core_artifact_and_wrong_closure_order() -> None:
    """F01-F10/T01-T04 不能缺證，且 F/T 序列不可交換、遺漏或重複。"""

    figures, tables = _registry_records()
    missing_core = _unavailable_record(
        "F01",
        status="not_applicable_by_registered_design",
        reason_code="registered_design",
    )
    with pytest.raises(ValueError):
        _registry(figures=(missing_core, *figures[1:]), tables=tables)

    with pytest.raises(ValueError):
        _registry(figures=(figures[1], figures[0], *figures[2:]), tables=tables)
    with pytest.raises(ValueError):
        _registry(figures=figures[:-1], tables=tables)
    with pytest.raises(ValueError):
        _registry(figures=figures, tables=(tables[1], tables[0], *tables[2:]))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"missing_comparison": True, "allow_missing_comparison": True},
        {"missing_validation": True, "allow_missing_validation_evidence": True},
        {
            "missing_comparison": True,
            "missing_validation": True,
            "allow_missing_comparison": True,
            "allow_missing_validation_evidence": True,
        },
    ),
)
def test_report_registry_allows_only_paired_optional_missing_evidence(
    kwargs: dict[str, object],
) -> None:
    """F11/T05 與 F12/T06 缺證必須成對，且只在 caller 開啟對應 allow flag 時接受。"""

    registry = _registry(**kwargs)
    records = {record.artifact_id: record for record in (*registry.figures, *registry.tables)}
    if kwargs.get("missing_comparison"):
        assert records["F11"].status == records["T05"].status == (
            "unavailable_missing_comparison"
        )
    if kwargs.get("missing_validation"):
        assert records["F12"].status == records["T06"].status == (
            "unavailable_missing_validation_evidence"
        )


@pytest.mark.parametrize(
    "allow_flag, missing_kwargs",
    (
        ("allow_missing_comparison", {"missing_comparison": True}),
        ("allow_missing_validation_evidence", {"missing_validation": True}),
    ),
)
def test_report_registry_rejects_optional_missing_without_allow_flag(
    allow_flag: str,
    missing_kwargs: dict[str, object],
) -> None:
    """optional comparison／validation 缺證不可在未明示 allow flag 時悄悄通過。"""

    del allow_flag
    with pytest.raises(ValueError):
        _registry(**missing_kwargs)


@pytest.mark.parametrize(
    "figure_id, table_id, missing_status, allow_flag, missing_side",
    (
        (
            "F11",
            "T05",
            "unavailable_missing_comparison",
            "allow_missing_comparison",
            "figure",
        ),
        (
            "F11",
            "T05",
            "unavailable_missing_comparison",
            "allow_missing_comparison",
            "table",
        ),
        (
            "F12",
            "T06",
            "unavailable_missing_validation_evidence",
            "allow_missing_validation_evidence",
            "figure",
        ),
        (
            "F12",
            "T06",
            "unavailable_missing_validation_evidence",
            "allow_missing_validation_evidence",
            "table",
        ),
    ),
)
def test_report_registry_rejects_mixed_optional_pair_statuses(
    figure_id: str,
    table_id: str,
    missing_status: str,
    allow_flag: str,
    missing_side: str,
) -> None:
    """allow=True 也不得讓四種 figure/table 方向中的任一 pair 混合 available/unavailable。"""

    missing = _unavailable_record(
        figure_id if missing_side == "figure" else table_id,
        status=missing_status,
        reason_code="missing_comparison"
        if missing_status == "unavailable_missing_comparison"
        else "missing_validation_evidence",
    )
    if missing_side == "figure":
        figure_overrides = {figure_id: missing}
        table_overrides: dict[str, ReportArtifactRecord] = {}
    else:
        figure_overrides = {}
        table_overrides = {table_id: missing}
    with pytest.raises(ValueError):
        _registry(
            allow_missing_comparison=allow_flag == "allow_missing_comparison",
            allow_missing_validation_evidence=allow_flag
            == "allow_missing_validation_evidence",
            figure_overrides=figure_overrides,
            table_overrides=table_overrides,
        )


def test_report_registry_rejects_global_duplicate_product_relative_path() -> None:
    """同一 release 內的所有 product relative_path 必須全域唯一，避免 publish overwrite。"""

    figures, tables = _registry_records()
    tampered_f02 = replace(
        figures[1],
        products=(figures[0].products[0], *figures[1].products[1:]),
    )
    tampered_figures = (figures[0], tampered_f02, *figures[2:])
    with pytest.raises(ValueError):
        _registry(figures=tampered_figures, tables=tables)


@pytest.mark.parametrize(
    "run_kind, evidence_class, allow_missing_comparison, allow_missing_validation_evidence",
    (
        ("pilot", "server_scientific_evidence", False, False),
        ("formal", "server_pilot_evidence", False, False),
        ("formal", "server_scientific_evidence", True, False),
        ("formal", "server_scientific_evidence", False, True),
    ),
)
def test_server_evidence_class_has_run_kind_and_completeness_gates(
    run_kind: str,
    evidence_class: str,
    allow_missing_comparison: bool,
    allow_missing_validation_evidence: bool,
) -> None:
    """SERVER scientific 只能是完整 formal，SERVER pilot 只能是 pilot。"""

    with pytest.raises(ValueError):
        _registry(
            run_kind=run_kind,
            evidence_class=evidence_class,
            allow_missing_comparison=allow_missing_comparison,
            allow_missing_validation_evidence=allow_missing_validation_evidence,
        )

    if evidence_class == "server_scientific_evidence":
        valid = _registry(run_kind="formal", evidence_class=evidence_class)
    else:
        valid = _registry(run_kind="pilot", evidence_class=evidence_class)
    assert valid.evidence_class == evidence_class


@pytest.mark.parametrize(
    "missing_kwargs",
    (
        {},
        {"missing_comparison": True, "allow_missing_comparison": True},
        {"missing_validation": True, "allow_missing_validation_evidence": True},
        {
            "missing_comparison": True,
            "missing_validation": True,
            "allow_missing_comparison": True,
            "allow_missing_validation_evidence": True,
        },
    ),
)
def test_server_formal_baseline_evidence_accepts_formal_complete_or_paired_missing(
    missing_kwargs: dict[str, object],
) -> None:
    """formal baseline 可完整發布，或依 allow flag 成對保留 F11/T05、F12/T06 缺證。

    本案例仍只使用記憶體中的假 product metadata；``server_formal_baseline_evidence``
    只是要驗證 evidence-class／run-kind／缺證 closure 的 binding，不把 synthetic fixture
    宣稱為 SERVER 的 OCM／NWW 科學成果。
    """

    registry = _registry(
        run_kind="formal",
        evidence_class="server_formal_baseline_evidence",
        **missing_kwargs,
    )
    assert registry.run_kind == "formal"
    assert registry.evidence_class == "server_formal_baseline_evidence"
    assert all(
        record.evidence_class == "server_formal_baseline_evidence"
        for record in (*registry.figures, *registry.tables)
    )


@pytest.mark.parametrize("run_kind", ("pilot", "unknown"))
def test_server_formal_baseline_evidence_rejects_nonformal_run_kind(run_kind: str) -> None:
    """formal baseline evidence 不得被 pilot 或未知 run kind 冒用。"""

    with pytest.raises(ValueError):
        _registry(
            run_kind=run_kind,
            evidence_class="server_formal_baseline_evidence",
        )


def test_synthetic_registry_requires_synthetic_evidence_on_every_artifact() -> None:
    """本機 synthetic fixture 只能標成 synthetic evidence，不得混入 SERVER evidence。"""

    figures, tables = _registry_records()
    tampered = replace(figures[0], evidence_class="server_pilot_evidence")
    with pytest.raises(ValueError):
        _registry(figures=(tampered, *figures[1:]), tables=tables)

    registry = _registry(run_kind="formal", evidence_class="synthetic_engineering_evidence")
    assert registry.run_kind == "formal"
    assert registry.evidence_class == "synthetic_engineering_evidence"


@pytest.mark.parametrize(
    "field, value",
    (
        ("schema_version", "2.0.0"),
        ("run_id", "../unsafe"),
        ("run_kind", "synthetic"),
        ("evidence_class", "unknown_evidence"),
        ("aggregate_manifest_sha256", "A" * 64),
        ("source_run_plan_sha256", "short"),
        ("source_run_progress_sha256", np.str_("a" * 64)),
        ("config_hash", "g" * 64),
        ("checkpoint_input_binding_hash", b"a" * 64),
        ("allow_missing_comparison", np.bool_(False)),
        ("allow_missing_validation_evidence", 0),
    ),
)
def test_report_registry_rejects_schema_slug_run_kind_evidence_hash_and_bool_tamper(
    field: str,
    value: object,
) -> None:
    """registry top-level binding 只接受固定 schema、slug、證據、hash 與原生 bool。"""

    with pytest.raises((TypeError, ValueError)):
        _registry(**{field: value})


def test_report_registry_rejects_artifact_subclass_and_duck_record() -> None:
    """registry closure 也要求每列是 exact ReportArtifactRecord，避免未驗證物件混入。"""

    class ArtifactSubclass(ReportArtifactRecord):
        """只供測試 registry nominal type gate 的 synthetic subclass。"""

    base = _available_record("F01")
    subclass = ArtifactSubclass(**_record_kwargs(base))
    figures, tables = _registry_records()
    with pytest.raises(TypeError):
        _registry(figures=(subclass, *figures[1:]), tables=tables)

    class ArtifactDuck:
        """具有 artifact_id 屬性但不是 validated record 的 duck object。"""

        artifact_id = "F01"
        artifact_kind = "figure"

    with pytest.raises(TypeError):
        _registry(figures=(ArtifactDuck(), *figures[1:]), tables=tables)  # type: ignore[arg-type]


def test_report_registry_defensively_snapshots_sequences_and_mappings() -> None:
    """registry 應複製 caller 的 list，並以 tuple 保存兩組 closure。"""

    figures, tables = _registry_records()
    figure_list = list(figures)
    table_list = list(tables)
    registry = _registry(figures=figure_list, tables=table_list)  # type: ignore[arg-type]
    figure_list.clear()
    table_list.clear()
    assert isinstance(registry.figures, tuple)
    assert isinstance(registry.tables, tuple)
    assert len(registry.figures) == 12
    assert len(registry.tables) == 6
    with pytest.raises(AttributeError):
        registry.run_id = "tampered"  # type: ignore[misc]


def test_report_registry_optional_pair_constants_are_fixed_and_complete() -> None:
    """測試 fixture 也明示兩組 optional pair 的 ID closure，避免只測到單一 F row。"""

    assert REPORT_COMPARISON_ARTIFACT_IDS == ("F11", "T05")
    assert REPORT_VALIDATION_ARTIFACT_IDS == ("F12", "T06")
    registry = _registry(
        missing_comparison=True,
        missing_validation=True,
        allow_missing_comparison=True,
        allow_missing_validation_evidence=True,
    )
    records = {record.artifact_id: record for record in (*registry.figures, *registry.tables)}
    assert all(
        records[artifact_id].status == "unavailable_missing_comparison"
        for artifact_id in REPORT_COMPARISON_ARTIFACT_IDS
    )
    assert all(
        records[artifact_id].status == "unavailable_missing_validation_evidence"
        for artifact_id in REPORT_VALIDATION_ARTIFACT_IDS
    )
