"""report pipeline 唯讀建置前閘門的核心契約測試。

測試以小型 synthetic 記憶體 payload stub 隔離完整 aggregate products，並實際建立
run／aggregate／MPLCONFIGDIR 的普通檔案拓撲。這樣可以在不產生圖表、不建立 report
partial、也不讀取 raw NetCDF 的前提下，驗證 preflight 的 identity、hash、schema、
output ownership 與 evidence policy；正式科學 run 的 trajectory 測試只放置 manifest，
確認 v2/v3 schema gate 不會呼叫 trajectory reader 或第二次 materialize 全部軌跡。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import lagrangian_backtracking.report_pipeline as pipeline
from lagrangian_backtracking.report_spec import ReportSpec
from lagrangian_backtracking.report_validation_evidence import (
    ValidationEvidence,
    ValidationMetric,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64
_CHECKPOINT_HASH = "c" * 64
_SCHEMA = "1.0.0"
_ERROR = "report build preflight 驗證失敗"


def _report_spec(run_id: str, aggregate_hash: str, report_hash: str) -> ReportSpec:
    """建立完整 ReportSpec；數值只代表測試用公尺／秒規格，不代表實際研究設定。"""

    return ReportSpec(
        schema_version=_SCHEMA,
        run_id=run_id,
        aggregate_spec_canonical_sha256=aggregate_hash,
        primary_kde_bandwidth_m=100,
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
        source_sha256=_OTHER_HASH,
        canonical_sha256=report_hash,
    )


def _metrics(*, fail: bool = False) -> tuple[ValidationMetric, ...]:
    """建立七類 evidence metric，並以兩個正 dt／member 點封閉曲線最低契約。"""

    value = 2.0 if fail else 0.5
    return (
        ValidationMetric("analytic_solution", "analytic-error", value, "m", 4, "<=", 1.0),
        ValidationMetric(
            "timestep_convergence",
            "dt-error",
            value,
            "m",
            4,
            "<=",
            1.0,
            "dt_seconds",
            1.0,
            "s",
        ),
        ValidationMetric(
            "timestep_convergence",
            "dt-error",
            value,
            "m",
            4,
            "<=",
            1.0,
            "dt_seconds",
            2.0,
            "s",
        ),
        ValidationMetric(
            "member_convergence",
            "member-error",
            value,
            "m",
            4,
            "<=",
            1.0,
            "member_count",
            2,
            "member",
        ),
        ValidationMetric(
            "member_convergence",
            "member-error",
            value,
            "m",
            4,
            "<=",
            1.0,
            "member_count",
            4,
            "member",
        ),
        ValidationMetric("known_source_synthetic", "coverage", 1.0, "fraction", 4, ">=", 0.9),
        ValidationMetric("checkpoint_restart", "restart-error", value, "m", 4, "<=", 1.0),
        ValidationMetric("numpy_numba_consistency", "backend-error", value, "m", 4, "<=", 1.0),
        ValidationMetric("forward_validation", "forward-error", value, "m", 4, "<=", 1.0),
    )


def _evidence(run_id: str, plan_text: str, *, fail: bool = False) -> ValidationEvidence:
    """以實際 source snapshot 建立 immutable evidence；hash 不由測試手抄。"""

    snapshots = {
        "aggregate_spec": json.dumps({"run_id": run_id, "kind": "aggregate"}),
        "report_spec": json.dumps({"run_id": run_id, "kind": "report"}),
        "source_run_plan": plan_text,
        "threshold_document": json.dumps({"run_id": run_id, "threshold": 1}),
    }
    return ValidationEvidence(
        run_id=run_id,
        metrics=_metrics(fail=fail),
        source_snapshots=snapshots,
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_kind: str = "synthetic",
    evidence: ValidationEvidence | None = None,
    comparison: bool = False,
    manifest_schema: str = "2.0.0",
) -> dict[str, object]:
    """建立不含圖表的最小 source topology，並替換下游 reader 為 typed fixture。"""

    tmp_path.mkdir(parents=True, exist_ok=True)
    run_id = "preflight-run"
    source = tmp_path / run_id
    aggregate_root = tmp_path / f"{run_id}.aggregate-v1"
    cache = tmp_path / "mpl-cache"
    source.mkdir()
    aggregate_root.mkdir()
    cache.mkdir()
    (aggregate_root / "aggregate_manifest.json").write_text("{}", encoding="utf-8")
    spec_path = tmp_path / "report_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    plan_document = {
        "run_id": run_id,
        "run_kind": run_kind,
        "experiment_case_id": "case-a",
        "config_hash": _HASH,
        "checkpoint_input_binding_hash": _CHECKPOINT_HASH,
        "shards": [],
    }
    plan_text = json.dumps(plan_document, separators=(",", ":"))
    plan_bytes = plan_text.encode("utf-8")
    source_files = {
        "run_plan.json": plan_bytes,
        "run_progress.json": json.dumps({"run_id": run_id}).encode("utf-8"),
        "normalized_config.json": b'{"settings":{}}',
        "input_inventory.json": b'{"files":[]}',
    }
    for name, raw in source_files.items():
        (source / name).write_bytes(raw)

    actual_evidence = evidence
    aggregate_hash = _HASH
    report_hash = _OTHER_HASH
    if actual_evidence is not None:
        aggregate_hash = actual_evidence.aggregate_spec_canonical_sha256
        report_hash = actual_evidence.report_spec_canonical_sha256
        evidence_path = tmp_path / "validation-evidence.json"
        evidence_path.write_text("{}", encoding="utf-8")
    else:
        evidence_path = None
    report_spec = _report_spec(run_id, aggregate_hash, report_hash)
    aggregate_spec = SimpleNamespace(
        schema_version=_SCHEMA,
        canonical_sha256=aggregate_hash,
        source_sha256=_OTHER_HASH,
        kde_bandwidths_m=(100,),
    )
    source_hashes = {
        field: hashlib.sha256(source_files[name]).hexdigest()
        for field, name in {
            "source_run_plan_sha256": "run_plan.json",
            "source_run_progress_sha256": "run_progress.json",
            "source_normalized_config_sha256": "normalized_config.json",
            "source_input_inventory_sha256": "input_inventory.json",
        }.items()
    }
    payload = SimpleNamespace(
        schema_version=_SCHEMA,
        run_id=run_id,
        run_kind=run_kind,
        experiment_case_id="case-a",
        config_hash=_HASH,
        checkpoint_input_binding_hash=_CHECKPOINT_HASH,
        aggregate_spec=aggregate_spec,
        **source_hashes,
    )
    comparison_payload = SimpleNamespace(
        schema_version=_SCHEMA,
        run_id="comparison-run",
        run_kind=run_kind,
        experiment_case_id="case-a",
        aggregate_spec=SimpleNamespace(
            schema_version=_SCHEMA,
            canonical_sha256="d" * 64,
            source_sha256="e" * 64,
        ),
    )
    if comparison:
        comparison_root = tmp_path / "comparison-run.aggregate-v1"
        comparison_root.mkdir()
        (comparison_root / "aggregate_manifest.json").write_text("{}", encoding="utf-8")
    else:
        comparison_root = None

    if run_kind == "formal":
        plan_document["shards"] = [{"shard_id": "s0"}]
        plan_bytes = json.dumps(plan_document, separators=(",", ":")).encode("utf-8")
        (source / "run_plan.json").write_bytes(plan_bytes)
        source_files["run_plan.json"] = plan_bytes
        payload.source_run_plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        (source / "run_progress.json").write_text(
            json.dumps({
                "run_id": run_id,
                "shards": {
                    "s0": {"lifecycle": "COMPLETE", "output_relative_path": "shards/s0"}
                },
            }),
            encoding="utf-8",
        )
        source_files["run_progress.json"] = (source / "run_progress.json").read_bytes()
        payload.source_run_progress_sha256 = hashlib.sha256(source_files["run_progress.json"]).hexdigest()
        shard = source / "shards" / "s0"
        shard.mkdir(parents=True)
        (shard / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": manifest_schema,
                    "run_metadata": {"run_id": run_id, "shard_id": "s0"},
                }
            ),
            encoding="utf-8",
        )
        plan_document = dict(plan_document)

    progress_document = {"run_id": run_id, "shards": {}}
    if run_kind == "formal":
        progress_document = {
            "run_id": run_id,
            "shards": {
                "s0": {
                    "lifecycle": "COMPLETE",
                    "output_relative_path": "shards/s0",
                }
            },
        }

    def fake_validate_run(
        path: Path,
        *,
        require_complete: bool,
        checkpoint_root: Path | None = None,
    ) -> dict[str, object]:
        """讓 policy 測試聚焦 preflight；真實 run validator 仍由 production 呼叫。"""

        del path, checkpoint_root
        return {"valid": require_complete, "errors": [], "summary": {"run_id": run_id}}

    def fake_read_aggregate(path: Path) -> object:
        """依 final root 選擇 primary 或 comparison 的已解碼 payload。"""

        return comparison_payload if comparison_root is not None and path == comparison_root else payload

    monkeypatch.setattr(pipeline, "validate_run", fake_validate_run)
    monkeypatch.setattr(pipeline, "load_run_plan", lambda path: plan_document)
    monkeypatch.setattr(pipeline, "load_run_progress", lambda path: progress_document)
    monkeypatch.setattr(pipeline, "read_aggregate_release", fake_read_aggregate)
    monkeypatch.setattr(pipeline, "load_report_spec", lambda path: report_spec)
    monkeypatch.setattr(
        pipeline,
        "validate_report_spec_against_aggregate_spec",
        lambda report, aggregate: None,
    )
    if actual_evidence is not None:
        monkeypatch.setattr(pipeline, "load_validation_evidence", lambda path: actual_evidence)

    return {
        "source": source,
        "aggregate": aggregate_root,
        "cache": cache,
        "spec": spec_path,
        "output": tmp_path / f"{run_id}.report-v1",
        "comparison": comparison_root,
        "evidence_path": evidence_path,
        "payload": payload,
        "report_spec": report_spec,
        "plan": plan_document,
    }


def _call(fixture: dict[str, object], **kwargs: object) -> pipeline.ReportBuildPreflight:
    """以固定 fixture 參數呼叫公開 API，避免每個 policy case 重複組路徑。"""

    arguments = {
        "source_run_root": fixture["source"],
        "aggregate_release_root": fixture["aggregate"],
        "report_spec_path": fixture["spec"],
        "output": fixture["output"],
        "evidence_class": "synthetic_engineering_evidence",
        "mplconfigdir": fixture["cache"],
    }
    arguments.update(kwargs)
    return pipeline.preflight_report_build(**arguments)  # type: ignore[arg-type]


def test_happy_path_is_immutable_and_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """synthetic engineering gate 回傳絕對 Path frozen view，且不建立 output/partial。"""

    fixture = _fixture(tmp_path, monkeypatch)
    before = frozenset(path.name for path in tmp_path.iterdir())
    result = _call(fixture)

    assert isinstance(result, pipeline.ReportBuildPreflight)
    assert result.run_id == "preflight-run"
    assert result.output == Path(fixture["output"]).absolute()
    assert result.output.name == "preflight-run.report-v1"
    assert result.output.is_absolute()
    assert not result.output.exists()
    assert frozenset(path.name for path in tmp_path.iterdir()) == before
    with pytest.raises(FrozenInstanceError):
        result.run_id = "tampered"  # type: ignore[misc]


def test_formal_accepts_single_v2_or_v3_without_reader_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """formal 只讀 manifest schema；單一 v2/v3 通過、v1 被拒絕，且不讀軌跡 payload。"""

    fixture = _fixture(tmp_path, monkeypatch, run_kind="formal")
    result = _call(
        fixture,
        evidence_class="server_formal_baseline_evidence",
        allow_missing_comparison=True,
        allow_missing_validation_evidence=True,
    )
    assert result.trajectory_schema_version == "2.0.0"

    fixture = _fixture(
        tmp_path / "v3",
        monkeypatch,
        run_kind="formal",
        manifest_schema=pipeline.TRAJECTORY_SHARD_SCHEMA_VERSION,
    )
    result = _call(
        fixture,
        evidence_class="server_formal_baseline_evidence",
        allow_missing_comparison=True,
        allow_missing_validation_evidence=True,
    )
    assert result.trajectory_schema_version == pipeline.TRAJECTORY_SHARD_SCHEMA_VERSION

    fixture = _fixture(tmp_path / "legacy", monkeypatch, run_kind="formal", manifest_schema="1.0.0")
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(
            fixture,
            evidence_class="server_formal_baseline_evidence",
            allow_missing_comparison=True,
            allow_missing_validation_evidence=True,
        )


def test_formal_rejects_mixed_v2_v3_manifests(tmp_path: Path) -> None:
    """正式 manifest 只允許全 run 單一版本，不可混用 v2 與 v3。"""

    root = tmp_path / "mixed-run"
    (root / "shards" / "s0").mkdir(parents=True)
    (root / "shards" / "s1").mkdir(parents=True)
    versions = ("2.0.0", pipeline.TRAJECTORY_SHARD_SCHEMA_VERSION)
    plan = {"run_id": "mixed-run", "shards": [{"shard_id": "s0"}, {"shard_id": "s1"}]}
    progress = {
        "shards": {
            f"s{i}": {"lifecycle": "COMPLETE", "output_relative_path": f"shards/s{i}"}
            for i in range(2)
        }
    }
    for index, version in enumerate(versions):
        (root / "shards" / f"s{index}" / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": version,
                    "run_metadata": {"run_id": "mixed-run", "shard_id": f"s{index}"},
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="mixed trajectory schema"):
        pipeline._validate_formal_manifests(root, plan, progress)


@pytest.mark.parametrize(
    ("evidence_class", "run_kind"),
    [
        ("synthetic_engineering_evidence", "pilot"),
        ("server_pilot_evidence", "synthetic"),
        ("server_formal_baseline_evidence", "pilot"),
        ("server_scientific_evidence", "synthetic"),
        ("unknown_evidence", "synthetic"),
    ],
)
def test_evidence_class_cannot_masquerade_as_another_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_class: str,
    run_kind: str,
) -> None:
    """固定 evidence class 與 run kind 的身份界線不能靠 caller 旗標繞過。"""

    fixture = _fixture(tmp_path, monkeypatch, run_kind=run_kind)
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture, evidence_class=evidence_class)


def test_scientific_requires_comparison_and_passed_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scientific 禁用 allow flags，且必須有不同 run 的 comparison 與 all_passed evidence。"""

    fixture = _fixture(tmp_path, monkeypatch, run_kind="formal")
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture, evidence_class="server_scientific_evidence")

    formal_plan_text = json.dumps(
        {
            "run_id": "preflight-run",
            "run_kind": "formal",
            "experiment_case_id": "case-a",
            "config_hash": _HASH,
            "checkpoint_input_binding_hash": _CHECKPOINT_HASH,
            "shards": [{"shard_id": "s0"}],
        },
        separators=(",", ":"),
    )
    passed = _evidence("preflight-run", formal_plan_text)
    fixture = _fixture(tmp_path / "passed", monkeypatch, run_kind="formal", evidence=passed, comparison=True)
    result = _call(
        fixture,
        evidence_class="server_scientific_evidence",
        validation_evidence=fixture["evidence_path"],
        comparison_release=fixture["comparison"],
    )
    assert result.validation_all_passed is True
    assert result.comparison_run_id == "comparison-run"

    failed = _evidence("preflight-run", formal_plan_text, fail=True)
    fixture = _fixture(tmp_path / "failed", monkeypatch, run_kind="formal", evidence=failed, comparison=True)
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(
            fixture,
            evidence_class="server_scientific_evidence",
            validation_evidence=fixture["evidence_path"],
            comparison_release=fixture["comparison"],
        )


@pytest.mark.parametrize(
    ("comparison", "validation", "allow_comparison", "allow_validation", "valid"),
    [
        (False, False, False, False, False),
        (False, False, True, False, False),
        (False, False, False, True, False),
        (False, False, True, True, True),
        (True, False, False, False, False),
        (False, True, False, False, False),
    ],
)
def test_formal_baseline_missing_evidence_requires_relative_allow_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comparison: bool,
    validation: bool,
    allow_comparison: bool,
    allow_validation: bool,
    valid: bool,
) -> None:
    """formal baseline 的兩類 optional input 各自受相對 allow flag 控制。"""

    evidence = _evidence("preflight-run", json.dumps({"run_id": "preflight-run"})) if validation else None
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        run_kind="formal",
        evidence=evidence,
        comparison=comparison,
    )
    kwargs = {
        "evidence_class": "server_formal_baseline_evidence",
        "allow_missing_comparison": allow_comparison,
        "allow_missing_validation_evidence": allow_validation,
    }
    if comparison:
        kwargs["comparison_release"] = fixture["comparison"]
    if validation:
        kwargs["validation_evidence"] = fixture["evidence_path"]
    if valid:
        assert _call(fixture, **kwargs).run_kind == "formal"
    else:
        with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
            _call(fixture, **kwargs)


def test_output_name_existing_and_symlink_are_read_only_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """output 必須是固定 sibling 且不存在；regular directory、symlink、broken symlink 都拒絕。"""

    fixture = _fixture(tmp_path, monkeypatch)
    output = Path(fixture["output"])
    output.mkdir()
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture)

    output.rmdir()
    output.symlink_to(Path(fixture["source"]))
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture)

    output.unlink()
    output.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture)

    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture, output=tmp_path / "arbitrary.report-v1")


def test_validation_binding_and_comparison_identity_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validation canonical binding、source plan hash 與 comparison 同 run tamper 都被阻擋。"""

    evidence = _evidence("preflight-run", json.dumps({"run_id": "preflight-run"}))
    fixture = _fixture(tmp_path, monkeypatch, evidence=evidence)
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(
            fixture,
            validation_evidence=fixture["evidence_path"],
            evidence_class="synthetic_engineering_evidence",
        )

    payload = fixture["payload"]
    payload.source_run_plan_sha256 = "f" * 64
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture)

    fixture = _fixture(tmp_path / "same", monkeypatch, comparison=True)
    comparison = fixture["comparison"]
    payload = fixture["payload"]
    original = pipeline.read_aggregate_release

    def same_run_reader(path: Path) -> object:
        """把 comparison stub 竄改成 primary run，確認不同 run_id 是必要條件。"""

        value = original(path)
        if path == comparison:
            value.run_id = payload.run_id
        return value

    monkeypatch.setattr(pipeline, "read_aggregate_release", same_run_reader)
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(fixture, comparison_release=comparison)


def test_mplconfigdir_and_public_errors_do_not_leak_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPLCONFIGDIR 必須為既有絕對 ordinary directory，錯誤文字不得含 fixture path。"""

    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$") as error:
        _call(fixture, mplconfigdir=tmp_path / "relative" / ".." / "not-created")
    assert str(tmp_path) not in str(error.value)

    bad_cache = tmp_path / "bad-cache"
    bad_cache.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$") as error:
        _call(fixture, mplconfigdir=bad_cache)
    assert str(tmp_path) not in str(error.value)


def test_formal_manifest_duplicate_and_nonfinite_json_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """manifest schema reader 同時拒絕 duplicate key 與非有限 JSON，避免寬鬆 parser 繞過。"""

    fixture = _fixture(tmp_path, monkeypatch, run_kind="formal")
    manifest = Path(fixture["source"]) / "shards" / "s0" / "manifest.json"
    manifest.write_text(
        '{"schema_version":"2.0.0","schema_version":"2.0.0",'
        '"run_metadata":{"run_id":"preflight-run","shard_id":"s0"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(
            fixture,
            evidence_class="server_formal_baseline_evidence",
            allow_missing_comparison=True,
            allow_missing_validation_evidence=True,
        )

    manifest.write_text(
        '{"schema_version":"2.0.0","particle_count":NaN,'
        '"run_metadata":{"run_id":"preflight-run","shard_id":"s0"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="^report build preflight 驗證失敗$"):
        _call(
            fixture,
            evidence_class="server_formal_baseline_evidence",
            allow_missing_comparison=True,
            allow_missing_validation_evidence=True,
        )
