"""validation evidence schema、來源綁定與安全 I/O 的完整 synthetic 驗收測試。

本檔只建立小型 synthetic JSON source snapshot，不讀取 OCM schema 3、NWW3 schema 1、
軌跡或 SERVER 資料。測試中的七類 metric、時間步／成員收斂曲線、SHA-256 與檔案
內容，僅用來驗證 F12/T06 evidence 的工程資料契約：數值欄位必須有限、比較結果由
``value``／``comparison``／``threshold`` 推導，來源摘要必須由實際 snapshot bytes
計算，且 reader／writer 不得因 path、symlink、覆寫或 partial ownership 而放寬驗證。
通過本檔不代表任何真實海洋資料已完成科學驗證或取得絕對來源機率／因果歸因。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import lagrangian_backtracking.report_validation_evidence as evidence_module
from lagrangian_backtracking.report_validation_evidence import (
    VALIDATION_EVIDENCE_CATEGORIES,
    VALIDATION_EVIDENCE_SCHEMA_VERSION,
    VALIDATION_METRIC_CATEGORIES,
    ValidationEvidence,
    ValidationMetric,
    load_validation_evidence,
    validate_validation_evidence,
    write_validation_evidence,
)

_RUN_ID = "synthetic-validation-run"
_SOURCE_NAMES = (
    "aggregate_spec",
    "report_spec",
    "source_run_plan",
    "threshold_document",
)
_EXPECTED_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "aggregate_spec_canonical_sha256",
        "report_spec_canonical_sha256",
        "source_run_plan_sha256",
        "threshold_document_sha256",
        "source_raw_sha256",
        "source_canonical_sha256",
        "source_snapshots",
        "metrics",
        "all_passed",
    }
)


def _canonical_json_bytes(document: object) -> bytes:
    """以獨立於 production helper 的固定規則產生 canonical JSON bytes。

    evidence 的 canonical bytes 使用 UTF-8、排序鍵、compact separators 與單一尾端
    換行；測試自行重述這個公開格式，避免和主檔共用同一個錯誤。source snapshot 的
    canonical digest 也使用同一 JSON 語意，但 raw digest 仍保留原始排版與位元組。
    """

    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _source_snapshots(run_id: str = _RUN_ID) -> dict[str, str]:
    """建立四份帶有 run identity 的 synthetic source raw text。

    每份文件都是合法 JSON object，但刻意使用縮排與尾端換行，讓測試可以區分 raw
    SHA-256 與 canonical SHA-256。文件的 ``run_id`` 只表達 engineering binding，並
    不模擬真實 run plan、AggregateSpec 或 ReportSpec 的完整 schema。
    """

    return {
        name: json.dumps(
            {
                "run_id": run_id,
                "document": name,
                "revision": 1,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
        for name in _SOURCE_NAMES
    }


def _metrics(*, failed_forward: bool = False) -> tuple[ValidationMetric, ...]:
    """建立涵蓋七類 category 的最小定量 metric 集合。

    ``dt_seconds`` 與 ``member_count`` 各有兩個正且不同的 curve point；其餘五類各
    有一個 metric。``forward_validation`` 可選擇性設為 failed，藉此驗證合法的
    failed metric 仍可保存，而 ``all_passed`` 必須由實際比較結果推導。
    """

    return (
        ValidationMetric(
            category="analytic_solution",
            metric_id="analytic-error",
            value=0.1,
            unit="m",
            sample_count=12,
            comparison="<=",
            threshold=0.2,
        ),
        ValidationMetric(
            category="timestep_convergence",
            metric_id="dt-error",
            value=0.2,
            unit="m",
            sample_count=12,
            comparison="<=",
            threshold=0.3,
            independent_name="dt_seconds",
            independent_value=1.0,
            independent_unit="s",
        ),
        ValidationMetric(
            category="timestep_convergence",
            metric_id="dt-error",
            value=0.1,
            unit="m",
            sample_count=12,
            comparison="<=",
            threshold=0.3,
            independent_name="dt_seconds",
            independent_value=0.5,
            independent_unit="s",
        ),
        ValidationMetric(
            category="member_convergence",
            metric_id="member-error",
            value=0.2,
            unit="m",
            sample_count=12,
            comparison="<=",
            threshold=0.3,
            independent_name="member_count",
            independent_value=10,
            independent_unit="members",
        ),
        ValidationMetric(
            category="member_convergence",
            metric_id="member-error",
            value=0.1,
            unit="m",
            sample_count=12,
            comparison="<=",
            threshold=0.3,
            independent_name="member_count",
            independent_value=20,
            independent_unit="members",
        ),
        ValidationMetric(
            category="known_source_synthetic",
            metric_id="source-distance-error",
            value=0.05,
            unit="m",
            sample_count=8,
            comparison="<=",
            threshold=0.1,
        ),
        ValidationMetric(
            category="checkpoint_restart",
            metric_id="restart-error",
            value=1e-6,
            unit="m",
            sample_count=8,
            comparison="<=",
            threshold=1e-5,
        ),
        ValidationMetric(
            category="numpy_numba_consistency",
            metric_id="backend-error",
            value=1e-8,
            unit="m",
            sample_count=8,
            comparison="<=",
            threshold=1e-6,
        ),
        ValidationMetric(
            category="forward_validation",
            metric_id="forward-score",
            value=0.5 if failed_forward else 0.8,
            unit="fraction",
            sample_count=16,
            comparison=">=",
            threshold=0.75,
        ),
    )


def _evidence(*, failed_forward: bool = False) -> ValidationEvidence:
    """建立一份完整的 immutable synthetic evidence fixture。"""

    return ValidationEvidence(
        schema_version=VALIDATION_EVIDENCE_SCHEMA_VERSION,
        run_id=_RUN_ID,
        metrics=_metrics(failed_forward=failed_forward),
        source_snapshots=_source_snapshots(),
    )


def _write_source_files(root: Path, *, run_id: str = _RUN_ID) -> dict[str, Path]:
    """在 pytest 暫存目錄建立四份 ordinary source JSON 檔並回傳明示 path mapping。"""

    root.mkdir()
    snapshots = _source_snapshots(run_id)
    paths: dict[str, Path] = {}
    for name in _SOURCE_NAMES:
        path = root / f"{name}.json"
        path.write_text(snapshots[name], encoding="utf-8")
        paths[name] = path
    return paths


def _partial_paths(destination: Path) -> list[Path]:
    """列出指定 writer 命名規則下的 partial，包含其他 ownership 的 foreign 檔。"""

    return sorted(destination.parent.glob(f".{destination.name}.*.partial"))


def _document_copy(evidence: ValidationEvidence) -> dict[str, object]:
    """以 JSON round-trip 複製 plain document，避免測試誤改 immutable fixture。"""

    return json.loads(json.dumps(evidence.to_dict(), ensure_ascii=False))


def _write_document(path: Path, document: object) -> None:
    """只在 pytest 暫存目錄寫入指定 bytes，讓 loader 邊界測試可控制排版。"""

    path.write_bytes(_canonical_json_bytes(document))


def _assert_path_safe_error(error: BaseException, path: Path) -> None:
    """確認公開 exception 不攜帶 caller 提供的 absolute path 或底層 repr。"""

    assert str(path) not in str(error)
    assert str(path) not in repr(error)


def test_validation_evidence_roundtrip_and_source_hashes_are_derived_from_snapshots(
    tmp_path: Path,
) -> None:
    """constructor、writer、strict loader 與 validator 應完整 round-trip 並保留摘要語意。"""

    source_paths = _write_source_files(tmp_path / "sources")
    evidence = ValidationEvidence.from_source_paths(
        run_id=_RUN_ID,
        metrics=_metrics(),
        aggregate_spec_path=source_paths["aggregate_spec"],
        report_spec_path=source_paths["report_spec"],
        source_run_plan_path=source_paths["source_run_plan"],
        threshold_document_path=source_paths["threshold_document"],
    )

    assert VALIDATION_EVIDENCE_CATEGORIES == VALIDATION_METRIC_CATEGORIES
    assert set(evidence.to_dict()) == _EXPECTED_ROOT_KEYS
    assert evidence.schema_version == "1.0.0"
    assert evidence.all_passed is True
    assert len(evidence.metrics) == 9
    for name in _SOURCE_NAMES:
        raw = evidence.source_snapshots[name].encode("utf-8")
        source_document = json.loads(raw)
        expected_raw = hashlib.sha256(raw).hexdigest()
        expected_canonical = hashlib.sha256(_canonical_json_bytes(source_document)).hexdigest()
        assert evidence.source_raw_sha256[name] == expected_raw
        assert evidence.source_canonical_sha256[name] == expected_canonical
    assert evidence.aggregate_spec_canonical_sha256 == evidence.source_canonical_sha256[
        "aggregate_spec"
    ]
    assert evidence.report_spec_canonical_sha256 == evidence.source_canonical_sha256["report_spec"]
    assert evidence.source_run_plan_sha256 == evidence.source_raw_sha256["source_run_plan"]
    assert evidence.threshold_document_sha256 == evidence.source_raw_sha256["threshold_document"]

    destination = tmp_path / "output" / "validation_evidence.json"
    destination.parent.mkdir()
    returned = write_validation_evidence(
        evidence=evidence,
        destination=destination,
        source_paths=source_paths,
    )
    loaded = load_validation_evidence(returned)
    validation = validate_validation_evidence(returned)

    assert returned == destination
    assert loaded == evidence
    assert loaded.to_dict() == evidence.to_dict()
    assert validation == {
        "valid": True,
        "errors": [],
        "summary": {
            "schema_version": "1.0.0",
            "run_id": _RUN_ID,
            "metric_count": 9,
            "category_count": 7,
            "source_document_count": 4,
            "failed_metric_count": 0,
            "all_passed": True,
        },
    }
    assert destination.lstat().st_nlink == 1
    assert _partial_paths(destination) == []


def test_validation_metric_and_evidence_are_defensive_and_immutable() -> None:
    """frozen record、tuple、mapping proxy 與 to_dict defensive copy 不得被 caller 改寫。"""

    mutable_metrics = list(_metrics())
    mutable_sources = _source_snapshots()
    evidence = ValidationEvidence(
        run_id=_RUN_ID,
        metrics=mutable_metrics,
        source_snapshots=mutable_sources,
    )
    metric = evidence.metrics[0]
    metric_snapshot = metric.to_dict()
    evidence_snapshot = evidence.to_dict()

    mutable_metrics.clear()
    mutable_sources["aggregate_spec"] = '{"run_id":"tampered"}'
    assert len(evidence.metrics) == 9
    assert evidence.source_snapshots["aggregate_spec"] != mutable_sources["aggregate_spec"]

    with pytest.raises(FrozenInstanceError):
        metric.value = 999  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metric.passed = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        evidence.run_id = "other-run"  # type: ignore[misc]
    with pytest.raises(TypeError):
        evidence.source_snapshots["aggregate_spec"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        evidence.source_raw_sha256["aggregate_spec"] = "f" * 64  # type: ignore[index]

    metric_snapshot["value"] = 999
    evidence_snapshot["metrics"][0]["value"] = 999  # type: ignore[index]
    assert metric.value != 999
    assert evidence.metrics[0].value != 999
    assert isinstance(evidence.metrics, tuple)
    assert type(evidence.source_snapshots).__name__ == "mappingproxy"
    assert metric.to_dict() != metric_snapshot


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("value", True),
        ("value", math.nan),
        ("threshold", math.inf),
        ("sample_count", True),
        ("sample_count", 0),
        ("comparison", "=="),
        ("independent_value", 0),
    ),
)
def test_validation_metric_rejects_nonfinite_boolean_invalid_or_nonpositive_values(
    field: str,
    value: object,
) -> None:
    """定量欄位不可以 bool、非有限數或零取代有效 scientific validation value。"""

    kwargs: dict[str, object] = {
        "category": "analytic_solution",
        "metric_id": "analytic-error",
        "value": 0.1,
        "unit": "m",
        "sample_count": 1,
        "comparison": "<=",
        "threshold": 0.2,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        ValidationMetric(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("category", "independent_name", "independent_value"),
    (
        ("timestep_convergence", "dt_seconds", 0),
        ("timestep_convergence", "member_count", 1.0),
        ("member_convergence", "member_count", 0),
        ("member_convergence", "member_count", 1.5),
    ),
)
def test_curve_points_require_the_registered_name_and_positive_native_type(
    category: str,
    independent_name: str,
    independent_value: int | float,
) -> None:
    """兩種收斂曲線的獨立變數名稱、單位與正值型別必須符合固定資料契約。"""

    with pytest.raises(ValueError):
        ValidationMetric(
            category=category,
            metric_id="curve-error",
            value=0.1,
            unit="m",
            sample_count=1,
            comparison="<=",
            threshold=0.2,
            independent_name=independent_name,
            independent_value=independent_value,
            independent_unit="s",
        )


def test_evidence_preserves_a_failed_metric_but_derives_all_passed_false() -> None:
    """未達門檻的定量 metric 仍是合法 evidence，且 all_passed 不可由 caller 指定。"""

    evidence = _evidence(failed_forward=True)
    forward = next(metric for metric in evidence.metrics if metric.category == "forward_validation")

    assert forward.passed is False
    assert evidence.all_passed is False
    assert sum(not metric.passed for metric in evidence.metrics) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param(lambda document: document.update(unexpected_key=True), id="unknown-root-key"),
        pytest.param(
            lambda document: document["metrics"].pop(),
            id="missing-required-category",
        ),
    ),
)
def test_loader_rejects_unknown_or_missing_root_contract_keys(
    tmp_path: Path,
    mutation: object,
) -> None:
    """root object 必須 exact key set，未知欄位與缺少必要欄位都要 fail closed。"""

    evidence = _evidence()
    document = _document_copy(evidence)
    mutation(document)  # type: ignore[operator]
    path = tmp_path / "unknown-or-missing.json"
    _write_document(path, document)

    with pytest.raises(ValueError) as error_info:
        load_validation_evidence(path)
    _assert_path_safe_error(error_info.value, path)
    result = validate_validation_evidence(path)
    assert result["valid"] is False
    assert isinstance(result["errors"], list)
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_loader_rejects_duplicate_json_key_at_root(tmp_path: Path) -> None:
    """strict JSON parser 不得採用 duplicate key 的最後一個值。"""

    evidence = _evidence()
    canonical = _canonical_json_bytes(evidence.to_dict())
    duplicate = canonical[:-2] + b',"all_passed":true}\n'
    path = tmp_path / "duplicate.json"
    path.write_bytes(duplicate)

    with pytest.raises(ValueError) as error_info:
        load_validation_evidence(path)
    _assert_path_safe_error(error_info.value, path)
    result = validate_validation_evidence(path)
    assert result["valid"] is False
    assert result["errors"] == [{"stage": "schema", "reason": "invalid_json"}]


def test_loader_rejects_noncanonical_outer_json(tmp_path: Path) -> None:
    """即使 JSON 語意相同，非固定排序／compact／newline 的 outer bytes 也不得載入。"""

    evidence = _evidence()
    path = tmp_path / "pretty.json"
    path.write_bytes(
        json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")
    )

    with pytest.raises(ValueError):
        load_validation_evidence(path)
    assert validate_validation_evidence(path)["errors"] == [
        {"stage": "schema", "reason": "not_canonical_json"}
    ]


def test_loader_and_constructor_reject_nonfinite_json_or_numeric_values(tmp_path: Path) -> None:
    """NaN、Infinity 與 -Infinity 不得進入 metric 或 JSON parser 的數值欄位。"""

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            ValidationMetric(
                category="analytic_solution",
                metric_id="nonfinite",
                value=value,
                unit="m",
                sample_count=1,
                comparison="<=",
                threshold=0.2,
            )

    document = _document_copy(_evidence())
    canonical = _canonical_json_bytes(document)
    nonfinite = canonical.replace(b'"value":0.1', b'"value":NaN', 1)
    path = tmp_path / "nonfinite.json"
    path.write_bytes(nonfinite)
    with pytest.raises(ValueError):
        load_validation_evidence(path)
    result = validate_validation_evidence(path)
    assert result["valid"] is False
    assert result["errors"] == [{"stage": "schema", "reason": "invalid_json"}]
    json.dumps(result, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        (
            "aggregate_spec_canonical_sha256",
            lambda document: document.__setitem__("aggregate_spec_canonical_sha256", "f" * 64),
        ),
        (
            "source_run_plan_sha256",
            lambda document: document.__setitem__("source_run_plan_sha256", "f" * 64),
        ),
        (
            "source_snapshot",
            lambda document: document["source_snapshots"].__setitem__(
                "threshold_document", '{"document":"tampered","run_id":"synthetic-validation-run"}'
            ),
        ),
    ),
)
def test_loader_rejects_hash_or_source_snapshot_tampering(
    tmp_path: Path,
    field: str,
    mutate: object,
) -> None:
    """任何 digest 或實際 source snapshot 脫綁，都必須在 strict loader 被偵測。"""

    document = _document_copy(_evidence())
    mutate(document)  # type: ignore[operator]
    path = tmp_path / f"tampered-{field}.json"
    _write_document(path, document)

    with pytest.raises(ValueError) as error_info:
        load_validation_evidence(path)
    _assert_path_safe_error(error_info.value, path)
    result = validate_validation_evidence(path)
    assert result["valid"] is False
    assert result["errors"][0]["stage"] == "source"  # type: ignore[index]


@pytest.mark.parametrize("field", ("passed", "all_passed"))
def test_loader_rejects_tampered_derived_pass_flags(tmp_path: Path, field: str) -> None:
    """passed 與 all_passed 都是 derived 欄位，文件內手動竄改必須失敗。"""

    document = _document_copy(_evidence())
    if field == "passed":
        analytic = next(
            item for item in document["metrics"] if item["category"] == "analytic_solution"
        )
        analytic["passed"] = False
    else:
        document["all_passed"] = False
    path = tmp_path / f"tampered-{field}.json"
    _write_document(path, document)

    with pytest.raises(ValueError):
        load_validation_evidence(path)
    result = validate_validation_evidence(path)
    assert result["valid"] is False
    assert result["errors"][0]["reason"] in {"passed_mismatch", "all_passed_mismatch"}  # type: ignore[index]


@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param(
            lambda metrics: metrics.pop(0),
            id="category-closure",
        ),
        pytest.param(
            lambda metrics: metrics.pop(1),
            id="timestep-curve-needs-two-points",
        ),
        pytest.param(
            lambda metrics: metrics.pop(2),
            id="member-curve-needs-two-points",
        ),
        pytest.param(
            lambda metrics: metrics.append(metrics[0]),
            id="duplicate-metric-identity",
        ),
    ),
)
def test_constructor_enforces_category_and_curve_closure(mutation: object) -> None:
    """七類 closure、metric identity 唯一性與兩種曲線最低點數不可被繞過。"""

    metrics = list(_metrics())
    mutation(metrics)  # type: ignore[operator]
    with pytest.raises(ValueError):
        ValidationEvidence(
            run_id=_RUN_ID,
            metrics=metrics,
            source_snapshots=_source_snapshots(),
        )


def test_loader_rejects_unknown_metric_category(tmp_path: Path) -> None:
    """文件中的 metric category 只能來自固定七類，不接受 caller 擴充欄位語意。"""

    document = _document_copy(_evidence())
    document["metrics"][0]["category"] = "future_category"  # type: ignore[index]
    path = tmp_path / "unknown-category.json"
    _write_document(path, document)

    with pytest.raises(ValueError):
        load_validation_evidence(path)
    assert validate_validation_evidence(path)["errors"][0]["stage"] == "metric"  # type: ignore[index]


@pytest.mark.parametrize("kind", ("file", "directory", "symlink", "broken-symlink"))
def test_writer_is_non_overwrite_for_every_existing_final_node(
    tmp_path: Path,
    kind: str,
) -> None:
    """既有 ordinary file、directory、symlink 或 broken symlink 都不得覆寫或清理。"""

    source_paths = _write_source_files(tmp_path / f"sources-{kind}")
    parent = tmp_path / f"output-{kind}"
    parent.mkdir()
    destination = parent / "validation_evidence.json"
    referent = parent / "protected.json"
    if kind == "file":
        original = b"protected-final"
        destination.write_bytes(original)
    elif kind == "directory":
        destination.mkdir()
        (destination / "marker").write_bytes(b"protected-directory")
    elif kind == "symlink":
        referent.write_bytes(b"protected-target")
        os.symlink(referent, destination)
        original = os.readlink(destination)
    else:
        os.symlink(parent / "missing.json", destination)
        original = os.readlink(destination)

    with pytest.raises(FileExistsError) as error_info:
        write_validation_evidence(
            run_id=_RUN_ID,
            metrics=_metrics(),
            destination=destination,
            source_paths=source_paths,
        )
    _assert_path_safe_error(error_info.value, destination)
    assert _partial_paths(destination) == []
    if kind == "file":
        assert destination.read_bytes() == original
    elif kind == "directory":
        assert (destination / "marker").read_bytes() == b"protected-directory"
    else:
        assert destination.is_symlink()
        assert os.readlink(destination) == original


@pytest.mark.parametrize("kind", ("symlink", "broken-symlink", "directory"))
def test_loader_requires_non_symlink_ordinary_file(tmp_path: Path, kind: str) -> None:
    """loader 不得追隨 symlink，也不得把 directory 當成 evidence JSON。"""

    source = tmp_path / "valid.json"
    source.write_bytes(_canonical_json_bytes(_evidence().to_dict()))
    if kind == "symlink":
        path = tmp_path / "link.json"
        os.symlink(source, path)
    elif kind == "broken-symlink":
        path = tmp_path / "broken.json"
        os.symlink(tmp_path / "missing.json", path)
    else:
        path = tmp_path / "directory.json"
        path.mkdir()

    with pytest.raises(ValueError) as error_info:
        load_validation_evidence(path)
    _assert_path_safe_error(error_info.value, path)
    result = validate_validation_evidence(path)
    assert result == {
        "valid": False,
        "errors": [{"stage": "topology", "reason": "not_regular_file"}],
        "summary": {},
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_writer_rejects_symlink_source_and_symlink_parent_without_partial(
    tmp_path: Path,
) -> None:
    """source 與 destination parent 都必須是明示且 ordinary 的 I/O 節點。"""

    source_paths = _write_source_files(tmp_path / "sources")
    source_link = tmp_path / "aggregate-link.json"
    os.symlink(source_paths["aggregate_spec"], source_link)
    source_paths["aggregate_spec"] = source_link
    output = tmp_path / "output"
    output.mkdir()
    destination = output / "validation_evidence.json"

    with pytest.raises(ValueError) as error_info:
        write_validation_evidence(
            run_id=_RUN_ID,
            metrics=_metrics(),
            destination=destination,
            source_paths=source_paths,
        )
    _assert_path_safe_error(error_info.value, destination)
    assert not destination.exists()
    assert _partial_paths(destination) == []

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    os.symlink(real_parent, linked_parent, target_is_directory=True)
    linked_destination = linked_parent / "validation_evidence.json"
    valid_source_paths = _write_source_files(tmp_path / "valid-sources")
    with pytest.raises(ValueError) as error_info:
        write_validation_evidence(
            run_id=_RUN_ID,
            metrics=_metrics(),
            destination=linked_destination,
            source_paths=valid_source_paths,
        )
    _assert_path_safe_error(error_info.value, linked_destination)
    assert not linked_destination.exists()
    assert _partial_paths(linked_destination) == []


def test_writer_source_tamper_is_rejected_without_publishing_or_partial(
    tmp_path: Path,
) -> None:
    """evidence 建立後若明示 source bytes 改變，writer 必須拒絕重新綁定。"""

    source_paths = _write_source_files(tmp_path / "sources")
    evidence = ValidationEvidence.from_source_paths(
        run_id=_RUN_ID,
        metrics=_metrics(),
        aggregate_spec_path=source_paths["aggregate_spec"],
        report_spec_path=source_paths["report_spec"],
        source_run_plan_path=source_paths["source_run_plan"],
        threshold_document_path=source_paths["threshold_document"],
    )
    source_paths["threshold_document"].write_text(
        '{"document":"threshold_document","revision":2,"run_id":"synthetic-validation-run"}\n',
        encoding="utf-8",
    )
    destination = tmp_path / "output" / "validation_evidence.json"
    destination.parent.mkdir()

    with pytest.raises(ValueError) as error_info:
        write_validation_evidence(
            evidence=evidence,
            destination=destination,
            source_paths=source_paths,
        )
    _assert_path_safe_error(error_info.value, destination)
    assert not destination.exists()
    assert _partial_paths(destination) == []


def test_writer_cleans_owned_partial_after_first_write_failure_and_keeps_foreign_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """partial 建立後立即失敗時只清理本 writer 的檔案，不碰 foreign partial。"""

    source_paths = _write_source_files(tmp_path / "sources")
    parent = tmp_path / "output"
    parent.mkdir()
    destination = parent / "validation_evidence.json"
    foreign = parent / f".{destination.name}.foreign.partial"
    foreign.write_bytes(b"foreign-owner")
    original_write = evidence_module.os.write
    calls = 0

    def fail_first_write(file_descriptor: int, data: memoryview) -> int:
        """模擬 partial 建立後的第一個 bytes write 失敗。"""

        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic write failure")
        return original_write(file_descriptor, data)

    monkeypatch.setattr(evidence_module.os, "write", fail_first_write)
    with pytest.raises(ValueError) as error_info:
        write_validation_evidence(
            run_id=_RUN_ID,
            metrics=_metrics(),
            destination=destination,
            source_paths=source_paths,
        )

    _assert_path_safe_error(error_info.value, destination)
    assert calls == 1
    assert not destination.exists()
    assert _partial_paths(destination) == [foreign]
    assert foreign.read_bytes() == b"foreign-owner"


def test_writer_does_not_delete_replaced_foreign_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """partial path 若在 self-validation 前被外部換 inode，清理不得刪除 foreign file。"""

    source_paths = _write_source_files(tmp_path / "sources")
    parent = tmp_path / "output"
    parent.mkdir()
    destination = parent / "validation_evidence.json"
    foreign_bytes = b"foreign-partial-owner"
    original_loader = evidence_module._load_validation_evidence

    def replace_before_load(path: Path) -> ValidationEvidence:
        """將本 writer partial 替換成不同 inode 的 foreign bytes 後交回原 loader。"""

        path.unlink()
        path.write_bytes(foreign_bytes)
        return original_loader(path)

    monkeypatch.setattr(evidence_module, "_load_validation_evidence", replace_before_load)
    with pytest.raises(ValueError) as error_info:
        write_validation_evidence(
            run_id=_RUN_ID,
            metrics=_metrics(),
            destination=destination,
            source_paths=source_paths,
        )

    _assert_path_safe_error(error_info.value, destination)
    partials = _partial_paths(destination)
    assert len(partials) == 1
    assert partials[0].read_bytes() == foreign_bytes
    assert not destination.exists()


def test_public_errors_and_validator_result_are_path_safe_for_missing_path(tmp_path: Path) -> None:
    """missing path、writer 失敗與 validator 結果都不得洩漏 absolute path 或非 JSON 值。"""

    missing = tmp_path / "private" / "missing-validation-evidence.json"
    with pytest.raises(ValueError) as error_info:
        load_validation_evidence(missing)
    _assert_path_safe_error(error_info.value, missing)

    validation = validate_validation_evidence(missing)
    assert validation == {
        "valid": False,
        "errors": [{"stage": "topology", "reason": "not_regular_file"}],
        "summary": {},
    }
    json.dumps(validation, ensure_ascii=False, allow_nan=False)

    destination = tmp_path / "private" / "write-failure.json"
    with pytest.raises(ValueError) as error_info:
        write_validation_evidence(
            run_id=_RUN_ID,
            metrics=_metrics(),
            destination=destination,
            source_paths={},
        )
    _assert_path_safe_error(error_info.value, destination)
