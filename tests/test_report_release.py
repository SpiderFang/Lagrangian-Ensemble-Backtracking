"""report-v1 writer、reader 與 validator 的最小 synthetic engineering 驗收。

本檔只建立暫存的合成工程資料：source snapshot 以可追溯 JSON bytes 表示，aggregate
spec／report spec 使用專案既有 immutable schema；圖表、表格與 sidecar 則是非空的
最小 bytes，不宣稱具有 OCM schema 3、NWW3 schema 1 或科學結果意義。測試刻意把
每一個產品 path 明示放入 registry 與 staging tree，驗證 report release 不會掃描或
採用未登錄的檔案；所有 release 目錄都由 production writer 建立與驗證。
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import lagrangian_backtracking.report_release as report_release_module
from lagrangian_backtracking.aggregate_spec import (
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
)
from lagrangian_backtracking.report_records import (
    REPORT_FIGURE_IDS,
    REPORT_RELEASE_SCHEMA_VERSION,
    REPORT_TABLE_IDS,
    ReportArtifactRecord,
    ReportProductRef,
    ReportRegistry,
)
from lagrangian_backtracking.report_release import (
    read_report_registry,
    read_report_release,
    validate_report_release,
    write_report_release,
)
from lagrangian_backtracking.report_spec import (
    REPORT_SPEC_SCHEMA_VERSION,
    ReportSpec,
)

_RUN_ID = "synthetic-report-run"
_CASE_ID = "synthetic-case"
_HASH = "a" * 64
_CHECKPOINT_HASH = "b" * 64
_SOURCE_NAMES = (
    "aggregate_manifest.json",
    "aggregate_spec.json",
    "report_spec.json",
    "run_plan.json",
    "run_progress.json",
    "normalized_config.json",
    "input_inventory.json",
)


def _canonical_json(document: object, *, newline: bool = False) -> bytes:
    """建立測試用 deterministic UTF-8 JSON；newline 只由呼叫端明示。"""

    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(raw_bytes: bytes) -> str:
    """回傳 synthetic fixture bytes 的小寫 SHA-256。"""

    return hashlib.sha256(raw_bytes).hexdigest()


def _aggregate_spec_document() -> tuple[dict[str, object], str]:
    """建立一站一邊界的有效 AggregateSpec JSON 與 semantic canonical hash。"""

    spec = AggregateSpec(
        schema_version="1.0.0",
        run_id=_RUN_ID,
        grid_cell_size_m=10,
        site_grids={
            "site-a": SiteGridSpec(
                x_min_m=0,
                x_max_m=20,
                y_min_m=0,
                y_max_m=10,
            )
        },
        site_metric_crs={
            "site-a": SiteMetricCRSSpec(
                projection_method="azimuthal_equidistant_wgs84",
                center_lon_deg=121.0,
                center_lat_deg=25.0,
                linear_unit="m",
                axis_order="x_east_y_north",
            )
        },
        boundary_bin_size_m=10,
        boundary_segment_lengths_m={"segment-a": 20},
        site_boundary_segment_ids={
            "site-a": SiteBoundarySegments(
                local_segment_ids=("segment-a",),
                outer_segment_ids=("segment-a",),
            )
        },
        kde_bandwidths_m=(10, 20, 30),
        hdr_levels=(0.5, 0.75, 0.9),
        age_bin_edges_seconds=(0, 10),
        bootstrap_replicates=1,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_numerical_failure_and_pre_window_deposition_v1",
        source_sha256=_HASH,
        canonical_sha256=_HASH,
    )
    document = spec.to_dict()
    semantic_payload = dict(document)
    semantic_payload.pop("source_sha256")
    semantic_payload.pop("canonical_sha256")
    # AggregateSpec loader 的輸入契約不允許 caller 把兩個衍生 hash 寫回 root；測試
    # 仍保留 semantic hash，供 report spec 與 aggregate manifest 做 source binding。
    return semantic_payload, _sha256(_canonical_json(semantic_payload))


def _report_spec_document(aggregate_canonical_hash: str) -> bytes:
    """建立有效 ReportSpec 輸入 JSON；兩個衍生 hash 不由 caller 寫入。"""

    spec = ReportSpec(
        schema_version=REPORT_SPEC_SCHEMA_VERSION,
        run_id=_RUN_ID,
        aggregate_spec_canonical_sha256=aggregate_canonical_hash,
        primary_kde_bandwidth_m=10,
        minimum_kde_raw_count=1,
        low_sample_min_member_count=1,
        vertical_depth_bin_edges_m=(0.0, 5.0, 10.0),
        representative_trajectory_count_per_site=8,
        representative_selection_policy="stable_hash_core_season_tide_v1",
        representative_selection_seed=0,
        travel_age_quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        pathway_first_passage_quantiles=(0.25, 0.5, 0.75),
        figure_formats=("png", "svg", "pdf"),
        raster_dpi=300,
        renderer_style_version="academic_zh_tw_v1",
        language="zh-TW",
        source_sha256=_HASH,
        canonical_sha256=_HASH,
    )
    document = spec.to_dict()
    document.pop("source_sha256")
    document.pop("canonical_sha256")
    return _canonical_json(document)


def _write_product(path: Path, content: bytes) -> ReportProductRef:
    """寫入單一明示 staging file，並由實際 bytes 建立 product reference。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    role_by_suffix = {
        ".png": ("figure_png", "image/png"),
        ".svg": ("figure_svg", "image/svg+xml"),
        ".pdf": ("figure_pdf", "application/pdf"),
        ".parquet": (
            "data_sidecar_parquet" if "data_sidecars" in path.parts else "table_parquet",
            "application/vnd.apache.parquet",
        ),
        ".csv": ("table_csv", "text/csv"),
        ".json": (
            "metadata_sidecar_json" if "data_sidecars" in path.parts else "caption_sidecar_json",
            "application/json",
        ),
    }
    role, media_type = role_by_suffix[path.suffix]
    raw_bytes = path.read_bytes()
    return ReportProductRef(
        relative_path=path.relative_to(path.parents[3]).as_posix(),
        role=role,
        media_type=media_type,
        size_bytes=len(raw_bytes),
        sha256=_sha256(raw_bytes),
    )


def _artifact(
    artifact_id: str,
    staging_root: Path,
) -> ReportArtifactRecord:
    """建立一列 available figure/table registry，所有 products 均實際存在於 staging。"""

    if artifact_id.startswith("F"):
        files = (
            (f"figures/main/{artifact_id}.png", b"png synthetic"),
            (f"figures/main/{artifact_id}.svg", b"svg synthetic"),
            (f"figures/main/{artifact_id}.pdf", b"pdf synthetic"),
            (f"caption_sidecars/{artifact_id}.json", b'{"caption":"synthetic"}'),
            (f"data_sidecars/{artifact_id}.parquet", b"parquet synthetic"),
        )
    else:
        files = (
            (f"tables/{artifact_id}.parquet", b"parquet synthetic"),
            (f"tables/{artifact_id}.csv", b"csv synthetic"),
            (f"data_sidecars/{artifact_id}.json", b'{"columns":[]}'),
        )
    products: list[ReportProductRef] = []
    for relative_path, content in files:
        target = staging_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        role_media = {
            ".png": ("figure_png", "image/png"),
            ".svg": ("figure_svg", "image/svg+xml"),
            ".pdf": ("figure_pdf", "application/pdf"),
            ".parquet": (
                "data_sidecar_parquet" if relative_path.startswith("data_sidecars/") else "table_parquet",
                "application/vnd.apache.parquet",
            ),
            ".csv": ("table_csv", "text/csv"),
            ".json": (
                (
                    "metadata_sidecar_json"
                    if relative_path.startswith("data_sidecars/")
                    else "caption_sidecar_json"
                ),
                "application/json",
            ),
        }
        role, media_type = role_media[target.suffix]
        products.append(
            ReportProductRef(
                relative_path=relative_path,
                role=role,
                media_type=media_type,
                size_bytes=len(content),
                sha256=_sha256(content),
            )
        )
    return ReportArtifactRecord(
        artifact_id=artifact_id,
        artifact_kind="figure" if artifact_id.startswith("F") else "table",
        title_zh=f"synthetic {artifact_id}",
        status="available",
        evidence_class="synthetic_engineering_evidence",
        products=tuple(products),
        input_sha256={"aggregate": _HASH},
        raw_sample_count=1,
        denominator_name="synthetic_members",
        denominator_count=1,
        units={"count": "1"},
        crs_by_site={"site-a": "EPSG:32651"},
        limitations=("僅供 synthetic engineering contract test",),
        unavailable_reason_code=None,
        unavailable_reason_zh=None,
        component_status={"primary": "available"},
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    """建立 writer 所需的 source run、aggregate source 與明示產品 staging。"""

    source_root = tmp_path / _RUN_ID
    aggregate_root = tmp_path / f"{_RUN_ID}.aggregate-v1"
    staging_root = tmp_path / "staging"
    source_root.mkdir()
    aggregate_root.mkdir()
    staging_root.mkdir()

    aggregate_spec_document, aggregate_canonical_hash = _aggregate_spec_document()
    aggregate_spec_bytes = _canonical_json(aggregate_spec_document)
    normalized_config_document = {"run_id": _RUN_ID, "members_per_scenario": 1}
    normalized_config_bytes = _canonical_json(normalized_config_document)
    config_hash = _sha256(normalized_config_bytes)
    run_plan_document = {
        "run_id": _RUN_ID,
        "run_kind": "pilot",
        "experiment_case_id": _CASE_ID,
        "config_hash": config_hash,
        "checkpoint_input_binding_hash": _CHECKPOINT_HASH,
        "files": {
            "normalized_config.json": {
                "size_bytes": len(normalized_config_bytes),
                "sha256": _sha256(normalized_config_bytes),
            },
            "input_inventory.json": {"size_bytes": 2, "sha256": _sha256(b"{}")},
        },
    }
    run_plan_bytes = _canonical_json(run_plan_document)
    run_progress_bytes = _canonical_json({"run_id": _RUN_ID, "run_lifecycle": "COMPLETE"})
    input_inventory_bytes = b"{}"
    aggregate_manifest_document = {
        "schema_version": "1.0.0",
        "metadata": {
            "run_id": _RUN_ID,
            "run_kind": "pilot",
            "experiment_case_id": _CASE_ID,
            "config_hash": config_hash,
            "checkpoint_input_binding_hash": _CHECKPOINT_HASH,
            "source_run_plan_sha256": _sha256(run_plan_bytes),
            "source_run_progress_sha256": _sha256(run_progress_bytes),
            "source_normalized_config_sha256": _sha256(normalized_config_bytes),
            "source_input_inventory_sha256": _sha256(input_inventory_bytes),
            "aggregate_spec_source_sha256": _sha256(aggregate_spec_bytes),
            "aggregate_spec_canonical_sha256": aggregate_canonical_hash,
        },
        "files": {
            "aggregate_spec.json": {
                "size_bytes": len(aggregate_spec_bytes),
                "sha256": _sha256(aggregate_spec_bytes),
            },
            "source_run_plan.json": {
                "size_bytes": len(run_plan_bytes),
                "sha256": _sha256(run_plan_bytes),
            },
            "source_run_progress.json": {
                "size_bytes": len(run_progress_bytes),
                "sha256": _sha256(run_progress_bytes),
            },
            "source_normalized_config.json": {
                "size_bytes": len(normalized_config_bytes),
                "sha256": _sha256(normalized_config_bytes),
            },
            "source_input_inventory.json": {
                "size_bytes": len(input_inventory_bytes),
                "sha256": _sha256(input_inventory_bytes),
            },
        },
    }
    aggregate_manifest_bytes = _canonical_json(aggregate_manifest_document, newline=True)

    source_bytes = {
        "aggregate_manifest.json": aggregate_manifest_bytes,
        "aggregate_spec.json": aggregate_spec_bytes,
        "report_spec.json": _report_spec_document(aggregate_canonical_hash),
        "run_plan.json": run_plan_bytes,
        "run_progress.json": run_progress_bytes,
        "normalized_config.json": normalized_config_bytes,
        "input_inventory.json": input_inventory_bytes,
    }
    for name, content in {
        "run_plan.json": run_plan_bytes,
        "run_progress.json": run_progress_bytes,
        "normalized_config.json": normalized_config_bytes,
        "input_inventory.json": input_inventory_bytes,
    }.items():
        (source_root / name).write_bytes(content)
    for name, content in {
        "aggregate_manifest.json": aggregate_manifest_bytes,
        "aggregate_spec.json": aggregate_spec_bytes,
        "source_run_plan.json": run_plan_bytes,
        "source_run_progress.json": run_progress_bytes,
        "source_normalized_config.json": normalized_config_bytes,
        "source_input_inventory.json": input_inventory_bytes,
    }.items():
        (aggregate_root / name).write_bytes(content)
    report_spec_path = tmp_path / "report_spec.json"
    report_spec_path.write_bytes(source_bytes["report_spec.json"])

    figures = tuple(_artifact(artifact_id, staging_root) for artifact_id in REPORT_FIGURE_IDS)
    tables = tuple(_artifact(artifact_id, staging_root) for artifact_id in REPORT_TABLE_IDS)
    registry = ReportRegistry(
        schema_version=REPORT_RELEASE_SCHEMA_VERSION,
        run_id=_RUN_ID,
        run_kind="pilot",
        experiment_case_id=_CASE_ID,
        evidence_class="synthetic_engineering_evidence",
        aggregate_manifest_sha256=_sha256(aggregate_manifest_bytes),
        source_run_plan_sha256=_sha256(run_plan_bytes),
        source_run_progress_sha256=_sha256(run_progress_bytes),
        config_hash=config_hash,
        checkpoint_input_binding_hash=_CHECKPOINT_HASH,
        allow_missing_comparison=False,
        allow_missing_validation_evidence=False,
        figures=figures,
        tables=tables,
    )
    return {
        "source_root": source_root,
        "aggregate_root": aggregate_root,
        "report_spec_path": report_spec_path,
        "staging_root": staging_root,
        "registry": registry,
        "final_path": tmp_path / f"{_RUN_ID}.report-v1",
    }


def _write_fixture(inputs: dict[str, object]) -> Path:
    """以固定 source/staging inputs 寫入一份 report-v1 final。"""

    return write_report_release(
        source_run_root=inputs["source_root"],  # type: ignore[arg-type]
        aggregate_release_root=inputs["aggregate_root"],  # type: ignore[arg-type]
        report_spec_path=inputs["report_spec_path"],  # type: ignore[arg-type]
        staging_root=inputs["staging_root"],  # type: ignore[arg-type]
        registry=inputs["registry"],  # type: ignore[arg-type]
        destination=inputs["final_path"],  # type: ignore[arg-type]
    )


def _rewrite_manifest_contract(
    final_path: Path,
    relative_path: str,
    raw_bytes: bytes,
) -> None:
    """同步測試竄改檔案的 manifest contract，讓驗證能進入後續 schema/binding gate。

    正常 writer 會一次產生 manifest；這個 helper 只在測試中模擬攻擊者同時竄改
    inventory 宣告，區分「單純 bytes checksum 失敗」與「完整 inventory 後仍被
    registry/source 語意拒絕」兩條 validator 路徑。
    """

    manifest_path = final_path / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["files"][relative_path]
    contract["size_bytes"] = len(raw_bytes)
    contract["sha256"] = _sha256(raw_bytes)
    if relative_path.startswith("source/"):
        source_name = relative_path.removeprefix("source/")
        source_contract = manifest["source"][source_name]
        source_contract["size_bytes"] = len(raw_bytes)
        source_contract["sha256"] = _sha256(raw_bytes)
    elif relative_path in {"figure_registry.json", "table_registry.json"}:
        registry_contract = manifest["registry"][relative_path]
        registry_contract["size_bytes"] = len(raw_bytes)
        registry_contract["sha256"] = _sha256(raw_bytes)
    manifest_path.write_bytes(_canonical_json(manifest, newline=True))


def _rewrite_registry_file(final_path: Path, registry_name: str, mutate: object) -> None:
    """以 canonical bytes 寫回一份測試用 registry 竄改，並同步其 manifest hash。"""

    registry_path = final_path / registry_name
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    mutate(document)
    raw_bytes = _canonical_json(document, newline=True)
    registry_path.write_bytes(raw_bytes)
    _rewrite_manifest_contract(final_path, registry_name, raw_bytes)


def test_report_release_success_round_trip(tmp_path: Path) -> None:
    """synthetic release 可原子建立、完整驗證，且 reader 能還原 registry/source bytes。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)

    validation = validate_report_release(final_path)
    assert validation["valid"] is True
    assert validation["errors"] == []
    release = read_report_release(final_path)
    assert release.registry == inputs["registry"]
    assert read_report_registry(final_path) == inputs["registry"]
    assert set(release.source_bytes) == set(_SOURCE_NAMES)
    assert release.source_bytes["input_inventory.json"] == b"{}"


def test_report_release_tamper_is_reported_without_absolute_path(tmp_path: Path) -> None:
    """final product bytes 竄改後只能得到固定 checksum stage，且錯誤不洩漏暫存路徑。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)
    tampered = final_path / "figures" / "main" / "F01.png"
    tampered.write_bytes(b"tampered")

    validation = validate_report_release(final_path)
    assert validation["valid"] is False
    assert validation["errors"] == [{"stage": "checksum", "reason": "hash_mismatch"}]
    assert str(tmp_path) not in json.dumps(validation, ensure_ascii=False)
    with pytest.raises(ValueError, match="report release 驗證失敗"):
        read_report_release(final_path)


def test_report_release_existing_final_is_never_overwritten(tmp_path: Path) -> None:
    """同一 final 第二次發布必須失敗，原始 release bytes 完全保留。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)
    before = (final_path / "report_manifest.json").read_bytes()

    with pytest.raises(FileExistsError):
        _write_fixture(inputs)

    assert (final_path / "report_manifest.json").read_bytes() == before
    assert validate_report_release(final_path)["valid"] is True


@pytest.mark.parametrize("racing_kind", ("directory", "file", "broken_symlink"))
def test_report_release_writer_rejects_racing_existing_final_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    racing_kind: str,
) -> None:
    """最後一次 lstat 後注入各類 final 節點，必須保留 inode、內容或 link target。"""

    inputs = _fixture(tmp_path)
    final_path = inputs["final_path"]  # type: ignore[assignment]
    real_rename = report_release_module._atomic_exclusive_directory_rename
    racing_identity: tuple[int, int] | None = None
    racing_file_bytes = b"racing-final-file"
    racing_link_target: str | None = None

    def create_racing_final(
        source: Path,
        destination: Path,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> None:
        """在 production backend 呼叫前注入另一程序建立的 final 節點。"""

        nonlocal racing_identity, racing_link_target
        if racing_kind == "directory":
            destination.mkdir()
        elif racing_kind == "file":
            destination.write_bytes(racing_file_bytes)
        else:
            broken_target = tmp_path / "missing-racing-final-target"
            destination.symlink_to(broken_target)
        node = os.lstat(destination)
        racing_identity = (node.st_dev, node.st_ino)
        if racing_kind == "broken_symlink":
            racing_link_target = os.readlink(destination)
        real_rename(
            source,
            destination,
            expected_parent_identity=expected_parent_identity,
        )

    monkeypatch.setattr(
        report_release_module,
        "_atomic_exclusive_directory_rename",
        create_racing_final,
    )
    with pytest.raises(FileExistsError) as error:
        _write_fixture(inputs)

    assert str(tmp_path) not in str(error.value)
    assert racing_identity is not None
    final_node = os.lstat(final_path)
    assert (final_node.st_dev, final_node.st_ino) == racing_identity
    if racing_kind == "directory":
        assert final_path.is_dir()
        assert list(final_path.iterdir()) == []
    elif racing_kind == "file":
        assert final_path.read_bytes() == racing_file_bytes
    else:
        assert final_path.is_symlink()
        assert racing_link_target is not None
        assert os.readlink(final_path) == racing_link_target
        assert not (tmp_path / "missing-racing-final-target").exists()
    assert not any(
        path.name.startswith(f".{final_path.name}.partial-")
        for path in final_path.parent.iterdir()
    )


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="host atomic rename backend 僅在 Linux/Darwin 提供",
)
def test_report_release_exclusive_directory_rename_uses_real_host_backend(
    tmp_path: Path,
) -> None:
    """本機實際後端（backend）必須以同一 inode 完成目錄重新命名且不覆寫目的名稱。"""

    source = tmp_path / ".partial"
    destination = tmp_path / "final"
    source.mkdir()
    (source / "marker").write_bytes(b"host-backend")
    source_node = os.lstat(source)
    parent_node = os.lstat(tmp_path)

    report_release_module._atomic_exclusive_directory_rename(
        source,
        destination,
        expected_parent_identity=(parent_node.st_dev, parent_node.st_ino),
    )

    destination_node = os.lstat(destination)
    assert (destination_node.st_dev, destination_node.st_ino) == (
        source_node.st_dev,
        source_node.st_ino,
    )
    assert not source.exists()
    assert (destination / "marker").read_bytes() == b"host-backend"


def test_report_release_linux_backend_dispatch_configures_libc_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux 平台分派（dispatch）必須選 renameat2 並設定完整 C 函式庫型別。"""

    def fake_renameat2(*_arguments: object) -> int:
        """模擬 libc renameat2 function pointer，供 ABI 設定驗證。"""

        return 0

    class FakeLibc:
        """提供單一 Linux 原子拒覆寫函式符號的測試 C 標準函式庫。"""

        renameat2 = staticmethod(fake_renameat2)

    calls: list[tuple[object, bool]] = []

    def fake_cdll(library: object, *, use_errno: bool) -> FakeLibc:
        """記錄 CDLL(None, use_errno=True) 的 native errno 設定。"""

        calls.append((library, use_errno))
        return FakeLibc()

    monkeypatch.setattr(report_release_module.sys, "platform", "linux")
    monkeypatch.setattr(report_release_module.ctypes, "CDLL", fake_cdll)

    function, flags = report_release_module._load_exclusive_rename_backend()

    assert function is fake_renameat2
    assert flags == report_release_module._LINUX_RENAME_NOREPLACE
    assert function.argtypes == list(report_release_module._RENAMEAT_ARGTYPES)
    assert function.restype is ctypes.c_int
    assert calls == [(None, True)]


def test_report_release_darwin_backend_dispatch_configures_libc_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Darwin 平台分派（dispatch）必須選 SDK 的 renameatx_np 與 RENAME_EXCL。"""

    def fake_renameatx_np(*_arguments: object) -> int:
        """模擬 Darwin renameatx_np function pointer，供 ABI 設定驗證。"""

        return 0

    class FakeLibc:
        """提供單一 Darwin 原子拒覆寫函式符號的測試 C 標準函式庫。"""

        renameatx_np = staticmethod(fake_renameatx_np)

    monkeypatch.setattr(report_release_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        report_release_module.ctypes,
        "CDLL",
        lambda _library, *, use_errno: FakeLibc(),
    )

    function, flags = report_release_module._load_exclusive_rename_backend()

    assert function is fake_renameatx_np
    assert flags == report_release_module._DARWIN_RENAME_EXCL
    assert function.argtypes == list(report_release_module._RENAMEAT_ARGTYPES)
    assert function.restype is ctypes.c_int


def test_report_release_missing_libc_symbol_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """支援平台若 CDLL 缺少原子拒覆寫函式符號，必須拒絕載入且不走替代路徑。"""

    class FakeLibcWithoutRename:
        """模擬可載入但未提供平台原子拒覆寫函式的 C 標準函式庫。"""

    monkeypatch.setattr(report_release_module.sys, "platform", "linux")
    monkeypatch.setattr(
        report_release_module.ctypes,
        "CDLL",
        lambda _library, *, use_errno: FakeLibcWithoutRename(),
    )

    with pytest.raises(report_release_module._ReportReleaseAtomicRenameUnsupported):
        report_release_module._load_exclusive_rename_backend()


@pytest.mark.parametrize(
    ("error_number", "expected_error"),
    (
        (errno.EEXIST, report_release_module._ReportReleaseFinalExistsError),
        (errno.EINVAL, report_release_module._ReportReleaseAtomicRenameUnsupported),
        (errno.ENOSYS, report_release_module._ReportReleaseAtomicRenameUnsupported),
        (errno.EOPNOTSUPP, report_release_module._ReportReleaseAtomicRenameUnsupported),
        (errno.EIO, report_release_module._ReportReleaseAtomicRenameFailure),
    ),
)
def test_report_release_exclusive_rename_errno_is_fail_closed(
    error_number: int,
    expected_error: type[Exception],
) -> None:
    """C 標準函式庫錯誤碼（errno）只允許 EEXIST 成為衝突，其餘均不得覆寫。"""

    def fake_rename(*_arguments: object) -> int:
        """回傳 native-style -1 並設定本執行緒 errno。"""

        ctypes.set_errno(error_number)
        return -1

    with pytest.raises(expected_error):
        report_release_module._call_exclusive_rename(
            fake_rename,
            parent_descriptor=3,
            source_name=".partial",
            destination_name="final.report-v1",
            rename_flags=1,
        )


def test_report_release_unknown_platform_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未支援平台不得以任何 Python 重新命名替代路徑發布。"""

    monkeypatch.setattr(report_release_module.sys, "platform", "win32")

    with pytest.raises(report_release_module._ReportReleaseAtomicRenameUnsupported):
        report_release_module._load_exclusive_rename_backend()


@pytest.mark.parametrize(
    "backend_error",
    (
        report_release_module._ReportReleaseAtomicRenameUnsupported,
        report_release_module._ReportReleaseAtomicRenameFailure,
    ),
)
def test_report_release_writer_cleans_owned_partial_on_backend_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_error: type[RuntimeError],
) -> None:
    """不支援或後端（backend）失敗只能清理自有 partial，且不得留下 final。"""

    inputs = _fixture(tmp_path)

    def fail_closed_backend() -> None:
        """注入不支援／後端失敗，確認 writer 沒有覆寫 fallback。"""

        raise backend_error("synthetic backend failure")

    monkeypatch.setattr(
        report_release_module,
        "_load_exclusive_rename_backend",
        fail_closed_backend,
    )

    with pytest.raises(ValueError, match="^report release 寫入失敗$") as error:
        _write_fixture(inputs)

    assert str(tmp_path) not in str(error.value)
    assert not inputs["final_path"].exists()  # type: ignore[union-attr]
    assert not any(
        path.name.startswith(f".{inputs['final_path'].name}.partial-")  # type: ignore[index]
        for path in tmp_path.iterdir()
    )


def test_report_release_writer_preserves_final_after_parent_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rename 已成功但 parent fsync 失敗時，final 必須保留並回報 durability error。"""

    inputs = _fixture(tmp_path)
    real_fsync = report_release_module._fsync_directory

    def fail_after_publish(
        path: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        """只讓 final publish 後的 parent fsync 失敗，保留 partial 建置 fsync。"""

        if expected_identity is not None:
            raise OSError("synthetic parent fsync failure")
        real_fsync(path, expected_identity=expected_identity)

    monkeypatch.setattr(report_release_module, "_fsync_directory", fail_after_publish)

    with pytest.raises(RuntimeError, match="^report release 已發布但 parent durability 未確認$"):
        _write_fixture(inputs)

    assert inputs["final_path"].is_dir()  # type: ignore[union-attr]
    assert not any(
        path.name.startswith(f".{inputs['final_path'].name}.partial-")  # type: ignore[index]
        for path in tmp_path.iterdir()
    )


def test_report_release_rejects_symlink_staging_product(tmp_path: Path) -> None:
    """registry 指向的 staging product 若為 symlink，writer 必須 fail closed 且不發布。"""

    inputs = _fixture(tmp_path)
    staging_product = inputs["staging_root"] / "figures" / "main" / "F01.png"  # type: ignore[operator]
    original = staging_product.read_bytes()
    staging_product.unlink()
    staging_product.symlink_to(tmp_path / "elsewhere.png")
    (tmp_path / "elsewhere.png").write_bytes(original)

    with pytest.raises(ValueError, match="report release 寫入失敗"):
        _write_fixture(inputs)
    assert not inputs["final_path"].exists()  # type: ignore[union-attr]


def test_report_release_rejects_direct_figure_product_path(tmp_path: Path) -> None:
    """release 層不接受基礎 record 允許、但固定 topology 禁止的 direct figure path。"""

    inputs = _fixture(tmp_path)
    registry = inputs["registry"]  # type: ignore[assignment]
    first_figure = registry.figures[0]
    direct_product = replace(first_figure.products[0], relative_path="figures/F01.png")
    direct_figure = replace(
        first_figure,
        products=(direct_product, *first_figure.products[1:]),
    )
    direct_registry = replace(
        registry,
        figures=(direct_figure, *registry.figures[1:]),
    )
    inputs["registry"] = direct_registry

    with pytest.raises(ValueError, match="report release 寫入失敗") as error:
        _write_fixture(inputs)

    assert str(tmp_path) not in str(error.value)
    assert not inputs["final_path"].exists()  # type: ignore[union-attr]
    assert not any(
        path.name.startswith(f".{inputs['final_path'].name}.partial-")  # type: ignore[index]
        for path in tmp_path.iterdir()
    )


def test_report_release_requires_both_figure_layers(tmp_path: Path) -> None:
    """即使 supplement 沒有產品，刪除其固定空目錄仍必須使 release 失效。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)
    (final_path / "figures" / "supplement").rmdir()

    assert validate_report_release(final_path) == {
        "valid": False,
        "errors": [{"stage": "topology", "reason": "fixed_node_set_mismatch"}],
        "summary": {},
    }


@pytest.mark.parametrize(
    ("relative_path", "is_directory", "expected_error"),
    (
        (
            "figures/main/unregistered.png",
            False,
            {"stage": "manifest", "reason": "manifest_inventory_mismatch"},
        ),
        (
            "tables/unregistered",
            True,
            {"stage": "topology", "reason": "symbolic_link_or_non_regular_node"},
        ),
    ),
)
def test_report_release_rejects_extra_file_or_directory(
    tmp_path: Path,
    relative_path: str,
    is_directory: bool,
    expected_error: dict[str, str],
) -> None:
    """exact inventory 不得被額外普通檔或未登錄子目錄擴張。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)
    extra_path = final_path / relative_path
    if is_directory:
        extra_path.mkdir()
    else:
        extra_path.write_bytes(b"unregistered")

    validation = validate_report_release(final_path)
    assert validation["valid"] is False
    assert validation["errors"] == [expected_error]
    assert str(tmp_path) not in json.dumps(validation, ensure_ascii=False)


def test_report_release_rejects_deleted_inventory_file(tmp_path: Path) -> None:
    """刪除 registry 已登錄的產品後，manifest 與 actual file set 必須不再閉合。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)
    (final_path / "tables" / "T01.csv").unlink()

    assert validate_report_release(final_path)["errors"] == [
        {"stage": "manifest", "reason": "manifest_inventory_mismatch"}
    ]


@pytest.mark.parametrize("mutation", ("artifact_extra", "product_extra", "product_missing"))
def test_report_release_registry_rows_require_exact_keys(
    tmp_path: Path,
    mutation: str,
) -> None:
    """artifact 與 product row 的 unknown/missing key 都必須由 reader 拒絕。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)

    def mutate(document: dict[str, object]) -> None:
        """只改動 registry JSON 結構，不替測試繞過 release inventory gate。"""

        artifact = document["artifacts"][0]
        if mutation == "artifact_extra":
            artifact["unknown_artifact_field"] = "must-fail"  # type: ignore[index]
            return
        product = artifact["products"][0]  # type: ignore[index]
        if mutation == "product_extra":
            product["unknown_product_field"] = "must-fail"  # type: ignore[index]
        else:
            product.pop("sha256")  # type: ignore[union-attr]

    _rewrite_registry_file(final_path, "figure_registry.json", mutate)
    validation = validate_report_release(final_path)
    assert validation["valid"] is False
    assert validation["errors"] == [
        {"stage": "registry", "reason": "registry_schema_mismatch"}
    ]


def test_report_release_rejects_source_binding_tamper_after_inventory_update(
    tmp_path: Path,
) -> None:
    """攻擊者同步改 source bytes 與 manifest hash，仍不得繞過 aggregate/source binding。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)
    source_path = final_path / "source" / "run_progress.json"
    tampered_document = json.loads(source_path.read_text(encoding="utf-8"))
    tampered_document["run_lifecycle"] = "FAILED"
    tampered_bytes = _canonical_json(tampered_document)
    source_path.write_bytes(tampered_bytes)
    _rewrite_manifest_contract(final_path, "source/run_progress.json", tampered_bytes)

    validation = validate_report_release(final_path)
    assert validation["valid"] is False
    assert validation["errors"] == [
        {"stage": "source", "reason": "source_binding_mismatch"}
    ]
    assert str(tmp_path) not in json.dumps(validation, ensure_ascii=False)


def test_report_release_rejects_broken_symlink_without_following_it(tmp_path: Path) -> None:
    """final product 變成 broken symlink 時，validator 只能回報 topology，不得追隨目標。"""

    inputs = _fixture(tmp_path)
    final_path = _write_fixture(inputs)
    product_path = final_path / "figures" / "main" / "F01.png"
    product_path.unlink()
    product_path.symlink_to(tmp_path / "missing-product-target.png")

    validation = validate_report_release(final_path)
    assert validation["valid"] is False
    assert validation["errors"] == [
        {"stage": "topology", "reason": "symbolic_link_or_non_regular_node"}
    ]
    assert str(tmp_path) not in json.dumps(validation, ensure_ascii=False)


def test_report_release_writer_cleans_owned_partial_but_not_other_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rename 前失敗只清本次 partial，預先存在的他人 partial 必須原樣保留。"""

    inputs = _fixture(tmp_path)
    final_path = inputs["final_path"]  # type: ignore[assignment]
    other_partial = final_path.parent / f".{final_path.name}.partial-other-owner"
    other_partial.mkdir()
    marker = other_partial / "owner-marker"
    marker.write_bytes(b"other-owner")
    real_writer = report_release_module._write_prepared_partial

    def fail_after_write(partial_root: Path, prepared: object) -> None:
        """讓完整 partial 建好後故意失敗，以驗證 ownership-scoped cleanup。"""

        real_writer(partial_root, prepared)  # type: ignore[arg-type]
        raise RuntimeError("synthetic pre-rename failure")

    monkeypatch.setattr(report_release_module, "_write_prepared_partial", fail_after_write)
    with pytest.raises(ValueError) as error:
        _write_fixture(inputs)

    assert str(tmp_path) not in str(error.value)
    assert not final_path.exists()
    partial_names = {
        path.name
        for path in final_path.parent.iterdir()
        if path.name.startswith(f".{final_path.name}.partial-")
    }
    assert partial_names == {other_partial.name}
    assert marker.read_bytes() == b"other-owner"


def test_report_release_writer_preserves_partial_after_ownership_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """partial 被另一 owner 置換後，cleanup 必須以 device/inode 保護新節點。"""

    inputs = _fixture(tmp_path)
    final_path = inputs["final_path"]  # type: ignore[assignment]
    real_writer = report_release_module._write_prepared_partial
    displaced_paths: list[Path] = []

    def take_over_partial(partial_root: Path, prepared: object) -> None:
        """模擬另一程序把原 partial 移走，再以不同 inode 佔回原名稱。"""

        real_writer(partial_root, prepared)  # type: ignore[arg-type]
        displaced = partial_root.with_name(f"{partial_root.name}-other-owner")
        partial_root.rename(displaced)
        partial_root.mkdir()
        (partial_root / "new-owner-marker").write_bytes(b"new-owner")
        displaced_paths.append(displaced)
        raise RuntimeError("synthetic ownership change")

    monkeypatch.setattr(report_release_module, "_write_prepared_partial", take_over_partial)
    with pytest.raises(ValueError, match="report release 寫入失敗"):
        _write_fixture(inputs)

    assert not final_path.exists()
    assert displaced_paths and displaced_paths[0].is_dir()
    replacement_paths = [
        path
        for path in final_path.parent.iterdir()
        if path.name.startswith(f".{final_path.name}.partial-")
    ]
    assert len(replacement_paths) == 2
    assert any(
        (path / "new-owner-marker").exists()
        and (path / "new-owner-marker").read_bytes() == b"new-owner"
        for path in replacement_paths
    )


def test_report_release_writer_self_validates_partial_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """writer 必須在 os.replace 前以 require_final_name=False 驗證 UUID partial。"""

    inputs = _fixture(tmp_path)
    calls: list[tuple[str, bool]] = []
    real_inspector = report_release_module._inspect_report_release

    def inspect_and_record(path: str | Path, *, require_final_name: bool) -> object:
        """保留真實自驗證結果，同時記錄被驗證的是 partial 還是 final。"""

        calls.append((Path(path).name, require_final_name))
        return real_inspector(path, require_final_name=require_final_name)

    monkeypatch.setattr(report_release_module, "_inspect_report_release", inspect_and_record)
    final_path = _write_fixture(inputs)

    assert final_path.is_dir()
    assert any(name.startswith(f".{final_path.name}.partial-") and not required for name, required in calls)


@pytest.mark.parametrize("existing_kind", ("file", "directory", "symlink"))
def test_report_release_writer_never_overwrites_existing_final(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    """既有 final file、directory、symlink（含 broken target）都必須保持原狀。"""

    inputs = _fixture(tmp_path)
    final_path = inputs["final_path"]  # type: ignore[assignment]
    if existing_kind == "file":
        final_path.write_bytes(b"pre-existing-file")
    elif existing_kind == "directory":
        final_path.mkdir()
        (final_path / "marker").write_bytes(b"pre-existing-directory")
    else:
        final_path.symlink_to(tmp_path / "broken-final-target")
    before = os.lstat(final_path)

    with pytest.raises(FileExistsError):
        _write_fixture(inputs)

    after = os.lstat(final_path)
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )
    if existing_kind == "file":
        assert final_path.read_bytes() == b"pre-existing-file"
    elif existing_kind == "directory":
        assert (final_path / "marker").read_bytes() == b"pre-existing-directory"
    else:
        assert final_path.is_symlink()
    assert not any(
        path.name.startswith(f".{final_path.name}.partial-")
        for path in final_path.parent.iterdir()
    )


def test_report_release_public_errors_are_json_safe_and_path_free(tmp_path: Path) -> None:
    """公開 reader/validator 僅回傳固定 stage/reason 或固定 ValueError，不洩漏絕對 path。"""

    missing_root = tmp_path / "missing-report-release"
    validation = validate_report_release(missing_root)
    serialized = json.dumps(validation, ensure_ascii=False, sort_keys=True)
    assert validation == {
        "valid": False,
        "errors": [{"stage": "topology", "reason": "fixed_node_set_mismatch"}],
        "summary": {},
    }
    assert str(tmp_path) not in serialized
    with pytest.raises(ValueError, match="^report release 驗證失敗$") as error:
        read_report_release(missing_root)
    assert str(tmp_path) not in str(error.value)
