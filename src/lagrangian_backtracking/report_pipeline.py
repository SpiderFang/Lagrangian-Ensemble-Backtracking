"""報告建置前的唯讀完整性與證據資格閘門。

本模組只負責在任何 report partial／staging 目錄建立以前，讀取並核對已發布的
run workspace、aggregate release、ReportSpec 與可選的 validation evidence。它不
建立目錄、不寫 JSON、不讀取 raw NetCDF，也不產生圖表或 final report release。
成功結果是不可變的記憶體快照，供未來 writer 在取得自己的 ownership 後繼續工作；
通過這個閘門只表示輸入資料契約與發布政策相容，不代表 F01--F12/T01--T06 的
科學結果已產生或已通過觀測驗證。

座標、距離與報告統計的單位沿用下游 AggregateSpec：公尺（m）為運算距離、秒（s）
為 travel age。這裡不重算任何軌跡、事件或統計，只保存 run／aggregate／spec／
evidence 的 identity、schema 與 SHA-256 摘要；所有公開失敗都收斂成不含輸入路徑的
固定錯誤。
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from .aggregate_release import read_aggregate_release
from .outputs import TRAJECTORY_SHARD_SCHEMA_VERSION
from .report_spec import (
    ReportSpec,
    load_report_spec,
    validate_report_spec_against_aggregate_spec,
)
from .report_validation_evidence import ValidationEvidence, load_validation_evidence
from .run_control import load_run_plan, load_run_progress
from .run_validation import validate_run

__all__ = ["ReportBuildPreflight", "preflight_report_build"]


# 報告 final 目錄名稱是固定資料路由；閘門不接受 caller 自訂 suffix，以免未來 writer
# 把同一 run 的不同產品誤寫到另一種產品的目錄或覆蓋既有 evidence。
_AGGREGATE_RELEASE_SUFFIX: Final[str] = ".aggregate-v1"
_REPORT_RELEASE_SUFFIX: Final[str] = ".report-v1"
_AGGREGATE_MANIFEST_NAME: Final[str] = "aggregate_manifest.json"
_PREFLIGHT_ERROR: Final[str] = "report build preflight 驗證失敗"
_MPLCONFIGDIR_ERROR: Final[str] = (
    "MPLCONFIGDIR 必須明示為既有、可寫、非符號連結的專用目錄"
)

# 這四個類別與本 pipeline 的公開語意一致；每個類別只可搭配一種 run kind，避免
# caller 以 evidence class 文字把 synthetic／pilot／baseline 輸入冒充成另一種研究階段。
_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "synthetic_engineering_evidence",
        "server_pilot_evidence",
        "server_formal_baseline_evidence",
        "server_scientific_evidence",
    }
)


def _absolute_path(value: str | Path) -> Path:
    """以不解析 symlink 的方式建立絕對 lexical Path。

    ``Path.resolve`` 會追蹤 symlink，可能讓 caller 以別名繞過 final node gate；這裡只
    正規化 ``.``／``..`` 並補上目前工作目錄，實際是否為 symlink 另由 lstat 檢查。
    輸入只在記憶體中轉換，不會建立任何檔案或目錄。
    """

    if not isinstance(value, (str, Path)):
        raise ValueError("path type")
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("path value")
    return Path(os.path.abspath(os.path.normpath(raw)))


def _require_directory(path: Path) -> None:
    """要求輸入根目錄是既有 ordinary directory，且最後節點不是 symlink。"""

    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("directory gate")


def _read_regular_file_bytes(path: Path) -> bytes:
    """以 no-follow descriptor 讀取普通檔案的 exact bytes。

    source SHA-256 必須由目前實際讀到的 snapshot 推導，因此不能只使用檔名或 caller
    提供的 digest。``O_NOFOLLOW`` 防止 check-then-open 期間把固定來源替換成 symlink；
    失敗只回傳給內部 gate，不把作業系統訊息暴露給公開 API。
    """

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("regular file gate")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            return handle.read()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha256_file(path: Path) -> str:
    """從 no-follow 讀取的 exact file bytes 計算小寫 SHA-256。"""

    return sha256(_read_regular_file_bytes(path)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[object, object]]) -> dict[str, object]:
    """拒絕 manifest 內任何層級的 duplicate key，避免採用最後一個值。"""

    document: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in document:
            raise ValueError("duplicate JSON key")
        document[key] = value
    return document


def _reject_json_constant(token: str) -> None:
    """拒絕 JSON decoder 額外接受的 NaN、Infinity 與 -Infinity。"""

    del token
    raise ValueError("non-finite JSON constant")


def _reject_nonfinite(value: object) -> None:
    """遞迴拒絕 manifest 中解析後才顯現的非有限浮點數。"""

    if type(value) is float and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if type(value) is dict:
        for nested in value.values():
            _reject_nonfinite(nested)
    elif type(value) is list:
        for nested in value:
            _reject_nonfinite(nested)


def _read_json_object(path: Path) -> dict[str, object]:
    """讀取一個 strict UTF-8 JSON object；只供 manifest/schema 的唯讀檢查。"""

    document = json.loads(
        _read_regular_file_bytes(path).decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    _reject_nonfinite(document)
    if type(document) is not dict:
        raise ValueError("JSON root")
    return document


def _safe_relative(root: Path, token: object) -> Path:
    """依既有 run token 找到 root 內的 ordinary path，不接受 traversal 或 symlink。"""

    if type(token) is not str or not token or "\\" in token:
        raise ValueError("relative path")
    relative = Path(token)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("relative path")
    result = root.joinpath(*relative.parts)
    if result.is_symlink():
        raise ValueError("relative symlink")
    return result


def _validate_mplconfigdir(value: str | Path | None) -> Path:
    """執行與 report renderer 相同的 explicit MPLCONFIGDIR gate，但不修改環境。

    pipeline 將 cache 目錄視為 caller 明示的輸入，而非透過暫時修改 process environment
    呼叫 renderer。目錄必須已存在、可寫可搜尋、最後節點不是 symlink，且不可退回
    ``$HOME/.matplotlib``；整個檢查不建立 probe file，因此保持 preflight 唯讀。
    """

    if value is None or not isinstance(value, (str, Path)):
        raise ValueError(_MPLCONFIGDIR_ERROR)
    try:
        raw_path = Path(os.fspath(value))
        if not raw_path.is_absolute():
            raise ValueError
        path = _absolute_path(value)
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError
        home = os.path.expanduser("~")
        if home != "~" and os.path.realpath(path) == os.path.realpath(Path(home) / ".matplotlib"):
            raise ValueError
        if not os.access(path, os.W_OK | os.X_OK):
            raise ValueError
    except Exception:
        raise ValueError(_MPLCONFIGDIR_ERROR) from None
    return path


def _select_mplconfigdir(
    mplconfigdir: str | Path | None,
    MPLCONFIGDIR: str | Path | None,
) -> Path:
    """解析 Python 風格與環境變數大小寫兩種呼叫拼法，不讓兩者互相矛盾。"""

    if mplconfigdir is not None and MPLCONFIGDIR is not None:
        first = _validate_mplconfigdir(mplconfigdir)
        second = _validate_mplconfigdir(MPLCONFIGDIR)
        if first != second:
            raise ValueError(_MPLCONFIGDIR_ERROR)
        return first
    explicit = mplconfigdir if mplconfigdir is not None else MPLCONFIGDIR
    if explicit is None:
        # 允許與既有 renderer 一致地由明示 process environment 提供，但不把環境值寫回
        # 或暫時修改；這也讓 CLI 未來可以直接把 operator 已核定的環境當作 gate input。
        explicit = os.environ.get("MPLCONFIGDIR")
    return _validate_mplconfigdir(explicit)


def _validate_formal_manifests(
    root: Path,
    plan: dict[str, Any],
    progress: dict[str, Any],
) -> str:
    """只讀所有正式 shard 的 manifest，精確關閉 trajectory schema 2.0.0。

    既有一般 reader 為工程相容性可接受 v1；正式報告則必須使用含完整 environment
    context 的 v2 payload。這裡只解析 manifest、確認 path／identity／schema，不呼叫
    ``read_trajectory_shard``，所以不會第二次 materialize 50,000×M 的軌跡結果。
    """

    plan_shards = plan.get("shards")
    progress_shards = progress.get("shards")
    if type(plan_shards) is not list or type(progress_shards) is not dict:
        raise ValueError("shard documents")
    for plan_row in plan_shards:
        if type(plan_row) is not dict:
            raise ValueError("plan shard")
        shard_id = plan_row.get("shard_id")
        progress_row = progress_shards.get(shard_id)
        if type(shard_id) is not str or type(progress_row) is not dict:
            raise ValueError("shard identity")
        if progress_row.get("lifecycle") != "COMPLETE":
            raise ValueError("shard incomplete")
        expected_token = f"shards/{shard_id}"
        if progress_row.get("output_relative_path") != expected_token:
            raise ValueError("shard output binding")
        shard_root = _safe_relative(root, expected_token)
        metadata = os.lstat(shard_root)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("shard directory")
        manifest = _read_json_object(shard_root / "manifest.json")
        if manifest.get("schema_version") != TRAJECTORY_SHARD_SCHEMA_VERSION:
            raise ValueError("trajectory schema")
        # 這兩個欄位已由 validate_run 做完整 payload binding；在只讀 schema gate 再次
        # 比對可防止未來上游 validator 放寬時，manifest 被誤綁到另一個 run/shard。
        if manifest.get("run_metadata", {}).get("run_id") != plan.get("run_id"):
            raise ValueError("trajectory run identity")
        if manifest.get("run_metadata", {}).get("shard_id") != shard_id:
            raise ValueError("trajectory shard identity")
    return TRAJECTORY_SHARD_SCHEMA_VERSION


def _validate_source_identity(
    root: Path,
    plan: dict[str, Any],
    progress: dict[str, Any],
    payload: Any,
) -> dict[str, str]:
    """由 source run snapshot 推導 digest，並核對 aggregate payload provenance。

    aggregate reader 已經驗證 release 內的 source copies；這裡再直接讀 primary run 的
    immutable plan/progress/config/inventory bytes，確保 preflight 的身份鎖點來自目前
    實際 source snapshot，而不是只相信 release metadata 或目錄名稱。
    """

    required = {
        "source_run_plan_sha256": root / "run_plan.json",
        "source_run_progress_sha256": root / "run_progress.json",
        "source_normalized_config_sha256": root / "normalized_config.json",
        "source_input_inventory_sha256": root / "input_inventory.json",
    }
    hashes = {field: _sha256_file(path) for field, path in required.items()}
    for field, digest in hashes.items():
        if getattr(payload, field, None) != digest:
            raise ValueError("aggregate source digest")
    if plan.get("run_id") != getattr(payload, "run_id", None):
        raise ValueError("run identity")
    if plan.get("run_kind") != getattr(payload, "run_kind", None):
        raise ValueError("run kind")
    if plan.get("experiment_case_id") != getattr(payload, "experiment_case_id", None):
        raise ValueError("experiment identity")
    if plan.get("config_hash") != getattr(payload, "config_hash", None):
        raise ValueError("config identity")
    if plan.get("checkpoint_input_binding_hash") != getattr(
        payload,
        "checkpoint_input_binding_hash",
        None,
    ):
        raise ValueError("checkpoint identity")
    if progress.get("run_id") != plan.get("run_id"):
        raise ValueError("progress identity")
    return hashes


def _validate_evidence_policy(
    *,
    evidence_class: str,
    run_kind: str,
    comparison: Any,
    validation: ValidationEvidence | None,
    allow_missing_comparison: bool,
    allow_missing_validation_evidence: bool,
) -> None:
    """執行 evidence class、run kind、allow flag 與 optional input truth table。

    scientific 必須是 formal run，兩個 allow flag 必須關閉，且 comparison 與通過的
    validation evidence 都實際存在。formal baseline 可在相對 allow flag 明示時缺少
    對應產品；synthetic／pilot engineering evidence 不會被提升成 scientific，且可在
    尚未有 optional comparison／validation 的工程階段先通過此建置前閘門。
    """

    if type(evidence_class) is not str or evidence_class not in _EVIDENCE_CLASSES:
        raise ValueError("evidence class")
    if type(run_kind) is not str or run_kind not in {"synthetic", "pilot", "formal"}:
        raise ValueError("run kind")
    if type(allow_missing_comparison) is not bool or type(allow_missing_validation_evidence) is not bool:
        raise ValueError("allow flag type")

    if evidence_class == "synthetic_engineering_evidence" and run_kind != "synthetic":
        raise ValueError("synthetic evidence identity")
    if evidence_class == "server_pilot_evidence" and run_kind != "pilot":
        raise ValueError("pilot evidence identity")
    if (
        evidence_class in {"server_formal_baseline_evidence", "server_scientific_evidence"}
        and run_kind != "formal"
    ):
        raise ValueError("formal evidence identity")
    if evidence_class == "server_scientific_evidence":
        if allow_missing_comparison or allow_missing_validation_evidence:
            raise ValueError("scientific allow policy")
        if comparison is None or validation is None or validation.all_passed is not True:
            raise ValueError("scientific evidence incomplete")
    elif evidence_class == "server_formal_baseline_evidence":
        if comparison is None and not allow_missing_comparison:
            raise ValueError("baseline comparison missing")
        if validation is None and not allow_missing_validation_evidence:
            raise ValueError("baseline validation missing")


@dataclass(frozen=True, slots=True)
class ReportBuildPreflight:
    """報告 writer 可消費的 immutable、唯讀 preflight memory view。

    這個容器只保存絕對 lexical Path、run／case identity、schema 版本、實際來源 SHA-256
    與 evidence policy flags；不保存 aggregate 陣列、trajectory 結果、JSON 文件或任何
    可供 writer 直接發佈的 mutable registry。``frozen=True`` 與 tuple/scalar 欄位讓
    下游不能透過此 view 改寫 preflight 結果；這仍不是 report release，也不代表科學
    validation 已完成。
    """

    source_run_root: Path
    aggregate_release_root: Path
    report_spec_path: Path
    output: Path
    mplconfigdir: Path
    checkpoint_root: Path | None
    run_id: str
    run_kind: str
    experiment_case_id: str
    aggregate_release_schema_version: str
    aggregate_manifest_sha256: str
    aggregate_spec_schema_version: str
    aggregate_spec_canonical_sha256: str
    aggregate_spec_source_sha256: str
    report_spec_schema_version: str
    report_spec_canonical_sha256: str
    report_spec_source_sha256: str
    source_run_plan_sha256: str
    source_run_progress_sha256: str
    source_normalized_config_sha256: str
    source_input_inventory_sha256: str
    trajectory_schema_version: str | None
    evidence_class: str
    allow_missing_comparison: bool
    allow_missing_validation_evidence: bool
    comparison_release_root: Path | None
    comparison_run_id: str | None
    comparison_release_schema_version: str | None
    comparison_aggregate_spec_canonical_sha256: str | None
    comparison_aggregate_manifest_sha256: str | None
    validation_evidence_path: Path | None
    validation_evidence_schema_version: str | None
    validation_evidence_run_id: str | None
    validation_evidence_aggregate_spec_canonical_sha256: str | None
    validation_evidence_report_spec_canonical_sha256: str | None
    validation_evidence_source_run_plan_sha256: str | None
    validation_evidence_threshold_document_sha256: str | None
    validation_all_passed: bool | None


def preflight_report_build(
    *,
    source_run_root: str | Path,
    aggregate_release_root: str | Path,
    report_spec_path: str | Path,
    output: str | Path,
    evidence_class: str,
    mplconfigdir: str | Path | None = None,
    MPLCONFIGDIR: str | Path | None = None,
    comparison_release: str | Path | None = None,
    validation_evidence: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    allow_missing_comparison: bool = False,
    allow_missing_validation_evidence: bool = False,
) -> ReportBuildPreflight:
    """在 report writer 建立任何 partial 前完成完整、唯讀的 build gate。

    Args:
        source_run_root: 已完成 run workspace；會以 ``validate_run(require_complete=True)``
            驗證，若有 external checkpoint root 也會傳入同一個唯讀 validator。
        aggregate_release_root: final ``<run_id>.aggregate-v1``；使用完整 aggregate
            reader，不接受 partial 或只含 manifest 的目錄。
        report_spec_path: strict ReportSpec JSON 普通檔；loader 會從實際 bytes 推導
            source/canonical SHA-256，再與 aggregate spec 綁定。
        output: 預計由後續 writer 建立的 final 路徑；必須是 source/aggregate 同父目錄
            下不存在的 ``<run_id>.report-v1``，本函式絕不建立它。
        evidence_class: 四種固定 report evidence class 之一；class 與 run kind 及
            optional comparison/validation policy 會重新核對。
        mplconfigdir: Python 慣例的小寫參數；需指向既有 task-specific cache 目錄。
        MPLCONFIGDIR: 同一 gate 的環境變數拼法，供 CLI／既有呼叫端傳入；不可與
            ``mplconfigdir`` 指向不同目錄。兩者皆省略時只讀 process environment。
        comparison_release: 可選的另一份 final aggregate release；只做完整 reader 與
            不同 run_id 檢查，精確 compatibility matrix 留給 comparison statistics。
        validation_evidence: 可選的 F12/T06 immutable quantitative evidence JSON。
        checkpoint_root: 可選 external checkpoint 根目錄，只傳給 run validator 並保存為
            絕對 Path；不讀 raw NetCDF 或建立 checkpoint。
        allow_missing_comparison: formal baseline 缺 comparison 時的明示政策旗標。
        allow_missing_validation_evidence: formal baseline 缺 validation evidence 時的
            明示政策旗標。

    Returns:
        只含 typed identity/hash/schema/flags 與絕對 Path 的 frozen memory view。

    Raises:
        ValueError: 任一輸入、schema、identity、hash、manifest、MPLCONFIGDIR 或 policy
            不符；公開訊息固定為 ``report build preflight 驗證失敗``，不含 path。

    Note:
        這個 API 不是 ``build_report_release``，不產生圖表、JSON、partial 或 final
        release；成功也只代表後續 writer 可以安全開始自己的建置流程。
    """

    try:
        source_root = _absolute_path(source_run_root)
        aggregate_root = _absolute_path(aggregate_release_root)
        spec_path = _absolute_path(report_spec_path)
        output_path = _absolute_path(output)
        checkpoint_path = _absolute_path(checkpoint_root) if checkpoint_root is not None else None
        comparison_path = _absolute_path(comparison_release) if comparison_release is not None else None
        evidence_path = _absolute_path(validation_evidence) if validation_evidence is not None else None

        _require_directory(source_root)
        _require_directory(aggregate_root)
        _require_directory(output_path.parent)
        if output_path.parent != source_root.parent or output_path.parent != aggregate_root.parent:
            raise ValueError("output parent")

        if checkpoint_path is None:
            run_validation = validate_run(source_root, require_complete=True)
        else:
            run_validation = validate_run(
                source_root,
                require_complete=True,
                checkpoint_root=checkpoint_path,
            )
        if type(run_validation) is not dict or run_validation.get("valid") is not True:
            raise ValueError("run validation")
        plan = load_run_plan(source_root)
        progress = load_run_progress(source_root)
        run_id = plan["run_id"]
        run_kind = plan["run_kind"]
        experiment_case_id = plan["experiment_case_id"]
        if type(run_id) is not str or type(run_kind) is not str or type(experiment_case_id) is not str:
            raise ValueError("run identity type")
        if source_root.name != run_id or aggregate_root.name != f"{run_id}{_AGGREGATE_RELEASE_SUFFIX}":
            raise ValueError("release identity name")

        # output 的 final node 以 lstat 檢查，連 broken symlink 也不能被當成不存在；這是
        # 後續 atomic writer 的 ownership 邊界，preflight 只讀取 metadata，不 reserve target。
        if output_path.name != f"{run_id}{_REPORT_RELEASE_SUFFIX}":
            raise ValueError("output name")
        try:
            os.lstat(output_path)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("output exists")

        mpl_path = _select_mplconfigdir(mplconfigdir, MPLCONFIGDIR)
        aggregate = read_aggregate_release(aggregate_root)
        if aggregate.run_id != run_id or aggregate.run_kind != run_kind:
            raise ValueError("aggregate identity")
        if aggregate.experiment_case_id != experiment_case_id:
            raise ValueError("aggregate case identity")
        source_hashes = _validate_source_identity(source_root, plan, progress, aggregate)
        aggregate_manifest_sha256 = _sha256_file(aggregate_root / _AGGREGATE_MANIFEST_NAME)
        aggregate_spec = aggregate.aggregate_spec
        report_spec = load_report_spec(spec_path)
        if type(report_spec) is not ReportSpec:
            raise ValueError("report spec type")
        validate_report_spec_against_aggregate_spec(report_spec, aggregate_spec)
        if report_spec.run_id != run_id:
            raise ValueError("report spec run identity")

        trajectory_schema = None
        if run_kind == "formal":
            trajectory_schema = _validate_formal_manifests(source_root, plan, progress)

        comparison = None
        comparison_manifest_sha256 = None
        if comparison_path is not None:
            _require_directory(comparison_path)
            comparison = read_aggregate_release(comparison_path)
            if comparison.run_id == run_id:
                raise ValueError("comparison run identity")
            comparison_manifest_sha256 = _sha256_file(
                comparison_path / _AGGREGATE_MANIFEST_NAME
            )

        evidence = None
        if evidence_path is not None:
            evidence = load_validation_evidence(evidence_path)
            if type(evidence) is not ValidationEvidence:
                raise ValueError("validation evidence type")
            if evidence.run_id != run_id:
                raise ValueError("validation evidence run identity")
            if evidence.aggregate_spec_canonical_sha256 != aggregate_spec.canonical_sha256:
                raise ValueError("validation aggregate binding")
            if evidence.report_spec_canonical_sha256 != report_spec.canonical_sha256:
                raise ValueError("validation report binding")
            if evidence.source_run_plan_sha256 != source_hashes["source_run_plan_sha256"]:
                raise ValueError("validation source plan binding")

        _validate_evidence_policy(
            evidence_class=evidence_class,
            run_kind=run_kind,
            comparison=comparison,
            validation=evidence,
            allow_missing_comparison=allow_missing_comparison,
            allow_missing_validation_evidence=allow_missing_validation_evidence,
        )

        return ReportBuildPreflight(
            source_run_root=source_root,
            aggregate_release_root=aggregate_root,
            report_spec_path=spec_path,
            output=output_path,
            mplconfigdir=mpl_path,
            checkpoint_root=checkpoint_path,
            run_id=run_id,
            run_kind=run_kind,
            experiment_case_id=experiment_case_id,
            aggregate_release_schema_version=aggregate.schema_version,
            aggregate_manifest_sha256=aggregate_manifest_sha256,
            aggregate_spec_schema_version=aggregate_spec.schema_version,
            aggregate_spec_canonical_sha256=aggregate_spec.canonical_sha256,
            aggregate_spec_source_sha256=aggregate_spec.source_sha256,
            report_spec_schema_version=report_spec.schema_version,
            report_spec_canonical_sha256=report_spec.canonical_sha256,
            report_spec_source_sha256=report_spec.source_sha256,
            source_run_plan_sha256=source_hashes["source_run_plan_sha256"],
            source_run_progress_sha256=source_hashes["source_run_progress_sha256"],
            source_normalized_config_sha256=source_hashes["source_normalized_config_sha256"],
            source_input_inventory_sha256=source_hashes["source_input_inventory_sha256"],
            trajectory_schema_version=trajectory_schema,
            evidence_class=evidence_class,
            allow_missing_comparison=allow_missing_comparison,
            allow_missing_validation_evidence=allow_missing_validation_evidence,
            comparison_release_root=comparison_path,
            comparison_run_id=None if comparison is None else comparison.run_id,
            comparison_release_schema_version=None if comparison is None else comparison.schema_version,
            comparison_aggregate_spec_canonical_sha256=(
                None if comparison is None else comparison.aggregate_spec.canonical_sha256
            ),
            comparison_aggregate_manifest_sha256=comparison_manifest_sha256,
            validation_evidence_path=evidence_path,
            validation_evidence_schema_version=None if evidence is None else evidence.schema_version,
            validation_evidence_run_id=None if evidence is None else evidence.run_id,
            validation_evidence_aggregate_spec_canonical_sha256=(
                None if evidence is None else evidence.aggregate_spec_canonical_sha256
            ),
            validation_evidence_report_spec_canonical_sha256=(
                None if evidence is None else evidence.report_spec_canonical_sha256
            ),
            validation_evidence_source_run_plan_sha256=(
                None if evidence is None else evidence.source_run_plan_sha256
            ),
            validation_evidence_threshold_document_sha256=(
                None if evidence is None else evidence.threshold_document_sha256
            ),
            validation_all_passed=None if evidence is None else evidence.all_passed,
        )
    except Exception:
        # 下層 validator、JSON、Arrow、Path 或作業系統例外可能攜帶 caller/server path；
        # preflight 的公開邊界只保留固定錯誤，讓 CLI 與測試可安全比對且不洩漏部署資訊。
        raise ValueError(_PREFLIGHT_ERROR) from None
