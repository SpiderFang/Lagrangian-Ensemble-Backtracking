"""F12/T06 版本化定量 validation evidence 契約。

本模組把解析解、時間步與系集成員收斂、known-source synthetic、checkpoint/restart、
NumPy/Numba 一致性及正向驗證所需的數值指標封裝成一份 immutable JSON evidence。它
不把「pytest 通過」或其他布林文字當成定量證據，也不執行數值模擬；呼叫端必須先在
自己的 scientific validation 流程中產生有限數值、單位、樣本數、比較符號與預先登錄
門檻。這些指標可以保存 failed metric 供診斷，但 ``all_passed`` 仍由全部指標的實際
比較結果推導，後續 report pipeline 可據此阻擋 F12/T06 scientific release。

四份來源文件（AggregateSpec、ReportSpec、source run plan 與預先登錄門檻文件）會以
UTF-8 JSON 的 exact raw text snapshot 寫入 evidence。writer 與 loader 都從 snapshot
重新計算 raw/canonical SHA-256，並且只接受 ordinary non-symlink files；JSON 內的 hash
欄位只是可稽核輸出，不能成為未驗證的信任來源。資料根目錄、SERVER 絕對路徑與底層
例外不會進入公開 validator report。synthetic evidence 通過此工程契約，仍不等於真實
OCM schema 3／NWW3 schema 1 驗證或科學成果。
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final, NoReturn
from uuid import uuid4

__all__ = [
    "VALIDATION_EVIDENCE_CATEGORIES",
    "VALIDATION_EVIDENCE_SCHEMA_VERSION",
    "VALIDATION_METRIC_CATEGORIES",
    "ValidationEvidence",
    "ValidationMetric",
    "load_validation_evidence",
    "validate_validation_evidence",
    "write_validation_evidence",
]


VALIDATION_EVIDENCE_SCHEMA_VERSION: Final[str] = "1.0.0"
"""F12/T06 evidence 的唯一 schema 版本；未知版本不得由 reader 猜測欄位語意。"""

VALIDATION_METRIC_CATEGORIES: Final[tuple[str, ...]] = (
    "analytic_solution",
    "timestep_convergence",
    "member_convergence",
    "known_source_synthetic",
    "checkpoint_restart",
    "numpy_numba_consistency",
    "forward_validation",
)
"""七個固定 validation 類別；每一類至少要有一個定量 metric。"""

# 以較長的語意名稱保留公開相容別名，讓後續 pipeline 不必依賴私有常數。
VALIDATION_EVIDENCE_CATEGORIES: Final[tuple[str, ...]] = VALIDATION_METRIC_CATEGORIES

_SOURCE_DOCUMENT_KEYS: Final[tuple[str, ...]] = (
    "aggregate_spec",
    "report_spec",
    "source_run_plan",
    "threshold_document",
)
_METRIC_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "category",
        "metric_id",
        "value",
        "unit",
        "sample_count",
        "comparison",
        "threshold",
        "passed",
        "independent_name",
        "independent_value",
        "independent_unit",
    }
)
_EVIDENCE_ROOT_KEYS: Final[frozenset[str]] = frozenset(
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
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}$")
_LOAD_ERROR = "validation evidence 驗證失敗"
_WRITE_ERROR = "validation evidence 寫入失敗"
_FINAL_EXISTS_ERROR = "validation evidence final 已存在"
_DURABILITY_ERROR = "validation evidence 已發布但 parent durability 未確認"


class _ValidationEvidenceError(ValueError):
    """內部 stage/reason 例外；公開 loader／writer 不直接暴露其內容。"""

    __slots__ = ("stage", "reason")

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


def _fail(stage: str, reason: str) -> NoReturn:
    """以固定代碼中止內部流程，避免把路徑或第三方例外帶入公開回報。"""

    raise _ValidationEvidenceError(stage, reason)


def _require_exact_text(value: object, *, label: str) -> str:
    """要求非空、沒有首尾空白與 NUL 的原生文字。"""

    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} 必須是安全非空文字")
    return value


def _require_token(value: object, *, label: str) -> str:
    """要求可穩定作為 run／metric identity 的安全 ASCII token。"""

    text = _require_exact_text(value, label=label)
    if _SAFE_TOKEN_RE.fullmatch(text) is None:
        raise ValueError(f"{label} 必須是安全 token")
    return text


def _require_sha256(value: object, *, label: str) -> str:
    """要求完整、全小寫的 SHA-256 十六進位摘要。"""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是 64 碼小寫 SHA-256")
    return value


def _require_native_number(value: object, *, label: str) -> int | float:
    """只接受有限的原生 Python int／float，拒絕 bool、NumPy scalar 與非有限值。"""

    if type(value) not in (int, float):
        raise ValueError(f"{label} 必須是原生 int 或 float")
    try:
        finite_value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{label} 必須是有限數值") from None
    if not math.isfinite(finite_value):
        raise ValueError(f"{label} 必須是有限數值")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    """要求不含 bool 的正原生整數，供 sample_count 與 member convergence point 使用。"""

    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必須是正的原生 int")
    return value


def _require_positive_number(value: object, *, label: str) -> int | float:
    """要求有限且嚴格大於零的原生數值。"""

    number = _require_native_number(value, label=label)
    if number <= 0:
        raise ValueError(f"{label} 必須大於 0")
    return number


def _reject_duplicate_json_keys(pairs: list[tuple[object, object]]) -> dict[str, object]:
    """拒絕同一 JSON object 內的 duplicate key，避免 parser 靜默採用最後一列。"""

    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise ValueError("JSON object key 必須是文字")
        if key in result:
            raise ValueError("JSON object 不可有 duplicate key")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> NoReturn:
    """拒絕 JSON decoder 額外接受的 NaN、Infinity 與 -Infinity。"""

    del token
    raise ValueError("JSON 不允許非有限常數")


def _reject_nonfinite_json_numbers(value: object) -> None:
    """遞迴拒絕 JSON 解析後才顯現的非有限浮點數。"""

    if type(value) is float and not math.isfinite(value):
        raise ValueError("JSON number 必須有限")
    if type(value) is dict:
        for nested in value.values():
            _reject_nonfinite_json_numbers(nested)
    elif type(value) is list:
        for nested in value:
            _reject_nonfinite_json_numbers(nested)


def _strict_json_object(raw_bytes: bytes) -> dict[str, object]:
    """把 exact UTF-8 bytes 解析為拒絕 duplicate/nonfinite 的 JSON object。"""

    if type(raw_bytes) is not bytes:
        raise ValueError("JSON 輸入必須是 bytes")
    document = json.loads(
        raw_bytes.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )
    _reject_nonfinite_json_numbers(document)
    if type(document) is not dict:
        raise ValueError("JSON root 必須是 object")
    return document


def _canonical_json_bytes(document: object) -> bytes:
    """依本專案固定規則產生 compact、sorted、UTF-8 JSON 與單一尾端換行。"""

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


def _read_source_document_text(path: Path) -> str:
    """以 no-follow ordinary-file I/O 讀取來源 JSON 的 exact UTF-8 raw text。"""

    raw_bytes = _read_regular_file_bytes(path)
    raw_bytes.decode("utf-8", errors="strict")
    # 先驗證來源本身是 finite、無 duplicate 的 object；canonical hash 必須有明確語意。
    _strict_json_object(raw_bytes)
    return raw_bytes.decode("utf-8", errors="strict")


def _derive_source_fingerprints(
    source_snapshots: Mapping[str, str],
    *,
    run_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """從四份 raw source snapshot 推導 raw/canonical SHA-256，完全不信任 caller digest。"""

    raw_hashes: dict[str, str] = {}
    canonical_hashes: dict[str, str] = {}
    for name in _SOURCE_DOCUMENT_KEYS:
        raw_text = source_snapshots[name]
        try:
            raw_bytes = raw_text.encode("utf-8", errors="strict")
            document = _strict_json_object(raw_bytes)
            canonical_bytes = _canonical_json_bytes(document)
        except Exception:
            _fail("source", "source_document_invalid")
        raw_hashes[name] = sha256(raw_bytes).hexdigest()
        canonical_hashes[name] = sha256(canonical_bytes).hexdigest()
        declared_run_id = document.get("run_id")
        if declared_run_id is not None and declared_run_id != run_id:
            _fail("source", "source_run_id_mismatch")
    return raw_hashes, canonical_hashes


def _normalize_source_snapshots(value: object) -> dict[str, str]:
    """建立四份來源 snapshot 的 defensive JSON-ready mapping。"""

    if not isinstance(value, Mapping) or set(value) != set(_SOURCE_DOCUMENT_KEYS):
        raise ValueError("source_snapshots key set 不符")
    normalized: dict[str, str] = {}
    for name in _SOURCE_DOCUMENT_KEYS:
        raw_value = value[name]
        if type(raw_value) is bytes:
            try:
                raw_text = raw_value.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise ValueError("source snapshot 必須是 UTF-8") from None
        elif type(raw_value) is str:
            raw_text = raw_value
        else:
            raise ValueError("source snapshot 必須是 str 或 bytes")
        normalized[name] = raw_text
    return normalized


def _normalize_metrics(value: object) -> tuple[ValidationMetric, ...]:
    """複製、排序並拒絕重複 identity 的 metric sequence。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Iterable):
        raise ValueError("metrics 必須是非字串 iterable")
    try:
        metrics = tuple(value)
    except Exception:
        raise ValueError("metrics 無法建立 defensive tuple") from None
    if any(type(metric) is not ValidationMetric for metric in metrics):
        raise ValueError("metrics 必須全部是 exact ValidationMetric")

    identities: set[tuple[str, str, int | float | None]] = set()
    for metric in metrics:
        identity = (metric.category, metric.metric_id, metric.independent_value)
        if identity in identities:
            raise ValueError("category×metric_id×independent value 不可重複")
        identities.add(identity)

    def sort_key(metric: ValidationMetric) -> tuple[object, ...]:
        independent = metric.independent_value
        if independent is None:
            independent_key: tuple[object, ...] = (0, 0.0)
        else:
            independent_key = (1, float(independent))
        return (metric.category, metric.metric_id, *independent_key, metric.unit)

    return tuple(sorted(metrics, key=sort_key))


def _validate_metric_collection(metrics: tuple[ValidationMetric, ...]) -> None:
    """核對七類 closure 與 timestep/member 曲線點的最低數量契約。"""

    categories = {metric.category for metric in metrics}
    if categories != set(VALIDATION_METRIC_CATEGORIES):
        raise ValueError("metrics 必須恰好涵蓋七個 validation category")
    timestep_values = {
        metric.independent_value
        for metric in metrics
        if metric.category == "timestep_convergence"
    }
    member_values = {
        metric.independent_value
        for metric in metrics
        if metric.category == "member_convergence"
    }
    if len(timestep_values) < 2:
        raise ValueError("timestep_convergence 必須至少有兩個 distinct dt_seconds point")
    if len(member_values) < 2:
        raise ValueError("member_convergence 必須至少有兩個 distinct member_count point")


@dataclass(frozen=True, slots=True)
class ValidationMetric:
    """一個可重現、可比較且不含布林替代值的定量 validation metric。

    ``value`` 與 ``threshold`` 是有限原生 Python 數值；``comparison`` 只允許 ``<=``
    或 ``>=``，因此 ``passed`` 可由 constructor 唯一推導，caller 不能傳入或覆寫。
    ``sample_count`` 是實際參與此 metric 的原始樣本數，不能以 0 代表遺失樣本。曲線
    metric 以三個 optional 欄位同時保存獨立變數名稱、有限值與單位；時間步收斂固定
    使用 ``dt_seconds``，成員收斂固定使用正原生整數 ``member_count``。
    """

    category: str
    metric_id: str
    value: int | float
    unit: str
    sample_count: int
    comparison: str
    threshold: int | float
    independent_name: str | None = None
    independent_value: int | float | None = None
    independent_unit: str | None = None
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        """逐欄驗證 metric 並計算不可竄改的 pass/fail 結果。"""

        if type(self.category) is not str or self.category not in VALIDATION_METRIC_CATEGORIES:
            raise ValueError("category 不在固定 validation category set")
        metric_id = _require_token(self.metric_id, label="metric_id")
        unit = _require_exact_text(self.unit, label="unit")
        value = _require_native_number(self.value, label="value")
        threshold = _require_native_number(self.threshold, label="threshold")
        sample_count = _require_positive_int(self.sample_count, label="sample_count")
        if self.comparison not in {"<=", ">="}:
            raise ValueError("comparison 只允許 <= 或 >=")

        optional_presence = (
            self.independent_name is not None,
            self.independent_value is not None,
            self.independent_unit is not None,
        )
        if any(optional_presence) and not all(optional_presence):
            raise ValueError("independent_name/value/unit 必須同時提供或同時省略")
        independent_value = self.independent_value
        if all(optional_presence):
            independent_name = _require_exact_text(
                self.independent_name,
                label="independent_name",
            )
            if _SAFE_NAME_RE.fullmatch(independent_name) is None:
                raise ValueError("independent_name 必須是安全名稱")
            _require_exact_text(
                self.independent_unit,
                label="independent_unit",
            )
            independent_value = _require_native_number(
                independent_value,
                label="independent_value",
            )
            if self.category == "timestep_convergence":
                if independent_name != "dt_seconds" or independent_value <= 0:
                    raise ValueError("timestep_convergence point 必須是正 dt_seconds")
            elif self.category == "member_convergence":
                if independent_name != "member_count" or type(independent_value) is not int:
                    raise ValueError("member_convergence point 必須是正原生 member_count")
                if independent_value <= 0:
                    raise ValueError("member_convergence point 必須大於 0")
        elif self.category in {"timestep_convergence", "member_convergence"}:
            raise ValueError("收斂 metric 必須提供 independent curve point")

        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "independent_value", independent_value)
        passed = value <= threshold if self.comparison == "<=" else value >= threshold
        object.__setattr__(self, "passed", passed)

    @property
    def operator(self) -> str:
        """回傳 comparison 的語意化別名；JSON schema 仍固定使用 ``comparison``。"""

        return self.comparison

    def to_dict(self) -> dict[str, object]:
        """回傳完全由原生 scalar 組成的 JSON-ready metric snapshot。"""

        return {
            "category": self.category,
            "metric_id": self.metric_id,
            "value": self.value,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "passed": self.passed,
            "independent_name": self.independent_name,
            "independent_value": self.independent_value,
            "independent_unit": self.independent_unit,
        }


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """F12/T06 的 immutable quantitative evidence 與推導式 source binding。

    constructor 只接受 run identity、metrics 與四份 source raw snapshot；所有 raw／
    canonical SHA-256、顯式 canonical binding 及 ``all_passed`` 都在 constructor 計算，
    因此不能由 caller 直接傳入 digest 或 pass/fail 布林值。nested mapping 會包成
    ``MappingProxyType``，metrics 會排序成 tuple，``to_dict`` 則輸出可直接序列化的
    defensive plain dict。schema version 固定為 ``1.0.0``。
    """

    run_id: str
    metrics: Iterable[ValidationMetric]
    source_snapshots: Mapping[str, str | bytes]
    schema_version: str = VALIDATION_EVIDENCE_SCHEMA_VERSION
    aggregate_spec_canonical_sha256: str = field(init=False)
    report_spec_canonical_sha256: str = field(init=False)
    source_run_plan_sha256: str = field(init=False)
    threshold_document_sha256: str = field(init=False)
    source_raw_sha256: Mapping[str, str] = field(init=False)
    source_canonical_sha256: Mapping[str, str] = field(init=False)
    all_passed: bool = field(init=False)

    def __post_init__(self) -> None:
        """建立 defensive snapshot、推導 source hashes 並核對全部 category closure。"""

        if self.schema_version != VALIDATION_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("schema_version 不支援")
        run_id = _require_token(self.run_id, label="run_id")
        metrics = _normalize_metrics(self.metrics)
        _validate_metric_collection(metrics)
        source_snapshots = _normalize_source_snapshots(self.source_snapshots)
        raw_hashes, canonical_hashes = _derive_source_fingerprints(
            source_snapshots,
            run_id=run_id,
        )

        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "source_snapshots", MappingProxyType(source_snapshots))
        object.__setattr__(self, "source_raw_sha256", MappingProxyType(raw_hashes))
        object.__setattr__(self, "source_canonical_sha256", MappingProxyType(canonical_hashes))
        object.__setattr__(
            self,
            "aggregate_spec_canonical_sha256",
            canonical_hashes["aggregate_spec"],
        )
        object.__setattr__(
            self,
            "report_spec_canonical_sha256",
            canonical_hashes["report_spec"],
        )
        object.__setattr__(self, "source_run_plan_sha256", raw_hashes["source_run_plan"])
        object.__setattr__(self, "threshold_document_sha256", raw_hashes["threshold_document"])
        object.__setattr__(self, "all_passed", all(metric.passed for metric in metrics))

    @classmethod
    def from_source_paths(
        cls,
        *,
        run_id: str,
        metrics: Iterable[ValidationMetric],
        aggregate_spec_path: str | Path,
        report_spec_path: str | Path,
        source_run_plan_path: str | Path,
        threshold_document_path: str | Path,
    ) -> ValidationEvidence:
        """從四個 caller 明示的 ordinary source files 建立 evidence。

        這個 factory 讓後續 pipeline 不必先手抄 hash；每個檔案都在此以 no-follow
        ordinary-file I/O 讀取，raw/canonical digest 仍由 constructor 從實際內容推導。
        source JSON 若含 ``run_id``，其值必須與 evidence identity 一致；不會由檔名猜測。
        """

        source_paths = {
            "aggregate_spec": Path(aggregate_spec_path),
            "report_spec": Path(report_spec_path),
            "source_run_plan": Path(source_run_plan_path),
            "threshold_document": Path(threshold_document_path),
        }
        source_snapshots = {
            name: _read_source_document_text(path) for name, path in source_paths.items()
        }
        return cls(run_id=run_id, metrics=metrics, source_snapshots=source_snapshots)

    def to_dict(self) -> dict[str, object]:
        """回傳 canonical writer／reader 使用的 JSON-ready plain dict。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "aggregate_spec_canonical_sha256": self.aggregate_spec_canonical_sha256,
            "report_spec_canonical_sha256": self.report_spec_canonical_sha256,
            "source_run_plan_sha256": self.source_run_plan_sha256,
            "threshold_document_sha256": self.threshold_document_sha256,
            "source_raw_sha256": {
                name: self.source_raw_sha256[name] for name in _SOURCE_DOCUMENT_KEYS
            },
            "source_canonical_sha256": {
                name: self.source_canonical_sha256[name] for name in _SOURCE_DOCUMENT_KEYS
            },
            "source_snapshots": {
                name: self.source_snapshots[name] for name in _SOURCE_DOCUMENT_KEYS
            },
            "metrics": [metric.to_dict() for metric in self.metrics],
            "all_passed": self.all_passed,
        }


def _require_exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict[str, object]:
    """核對 JSON object 的 exact key set，unknown/missing key 都 fail closed。"""

    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} key set 不符")
    return value


def _metric_from_document(document: object) -> ValidationMetric:
    """從 exact metric object 建立 metric，並核對文件內 declared passed。"""

    try:
        metric_document = _require_exact_keys(document, _METRIC_ROOT_KEYS, label="metric")
        declared_passed = metric_document["passed"]
        if type(declared_passed) is not bool:
            _fail("metric", "passed_schema_mismatch")
        metric = ValidationMetric(
            category=metric_document["category"],
            metric_id=metric_document["metric_id"],
            value=metric_document["value"],
            unit=metric_document["unit"],
            sample_count=metric_document["sample_count"],
            comparison=metric_document["comparison"],
            threshold=metric_document["threshold"],
            independent_name=metric_document["independent_name"],
            independent_value=metric_document["independent_value"],
            independent_unit=metric_document["independent_unit"],
        )
    except _ValidationEvidenceError:
        raise
    except Exception:
        _fail("metric", "metric_schema_mismatch")
    if metric.passed is not declared_passed:
        _fail("metric", "passed_mismatch")
    return metric


def _require_hash_mapping(value: object, *, label: str) -> dict[str, str]:
    """核對 source hash mapping 的 exact key set 與每一項摘要格式。"""

    mapping = _require_exact_keys(value, frozenset(_SOURCE_DOCUMENT_KEYS), label=label)
    result: dict[str, str] = {}
    for name in _SOURCE_DOCUMENT_KEYS:
        try:
            result[name] = _require_sha256(mapping[name], label=f"{label}.{name}")
        except Exception:
            _fail("source", "source_hash_schema_mismatch")
    return result


def _evidence_from_document(document: object) -> ValidationEvidence:
    """由 parsed document 建立 evidence，並用 derived hashes 逐項反驗 JSON 欄位。"""

    root = _require_exact_keys(document, _EVIDENCE_ROOT_KEYS, label="evidence")
    source_snapshots = _require_exact_keys(
        root["source_snapshots"],
        frozenset(_SOURCE_DOCUMENT_KEYS),
        label="source_snapshots",
    )
    if type(root["metrics"]) is not list:
        _fail("metric", "metrics_schema_mismatch")
    try:
        metrics = tuple(_metric_from_document(item) for item in root["metrics"])
    except _ValidationEvidenceError:
        raise
    except Exception:
        _fail("metric", "metric_schema_mismatch")

    try:
        evidence = ValidationEvidence(
            schema_version=root["schema_version"],
            run_id=root["run_id"],
            metrics=metrics,
            source_snapshots=source_snapshots,
        )
    except _ValidationEvidenceError:
        raise
    except Exception:
        _fail("metric", "metric_contract_violation")

    declared_raw = _require_hash_mapping(root["source_raw_sha256"], label="source_raw_sha256")
    declared_canonical = _require_hash_mapping(
        root["source_canonical_sha256"],
        label="source_canonical_sha256",
    )
    try:
        declared_aggregate = _require_sha256(
            root["aggregate_spec_canonical_sha256"],
            label="aggregate_spec_canonical_sha256",
        )
        declared_report = _require_sha256(
            root["report_spec_canonical_sha256"],
            label="report_spec_canonical_sha256",
        )
        declared_plan = _require_sha256(
            root["source_run_plan_sha256"],
            label="source_run_plan_sha256",
        )
        declared_threshold = _require_sha256(
            root["threshold_document_sha256"],
            label="threshold_document_sha256",
        )
    except Exception:
        _fail("source", "source_hash_schema_mismatch")

    if dict(evidence.source_raw_sha256) != declared_raw:
        _fail("source", "source_raw_hash_mismatch")
    if dict(evidence.source_canonical_sha256) != declared_canonical:
        _fail("source", "source_canonical_hash_mismatch")
    if evidence.aggregate_spec_canonical_sha256 != declared_aggregate:
        _fail("source", "aggregate_spec_canonical_hash_mismatch")
    if evidence.report_spec_canonical_sha256 != declared_report:
        _fail("source", "report_spec_canonical_hash_mismatch")
    if evidence.source_run_plan_sha256 != declared_plan:
        _fail("source", "source_run_plan_hash_mismatch")
    if evidence.threshold_document_sha256 != declared_threshold:
        _fail("source", "threshold_document_hash_mismatch")
    if type(root["all_passed"]) is not bool:
        _fail("metric", "all_passed_schema_mismatch")
    if evidence.all_passed is not root["all_passed"]:
        _fail("metric", "all_passed_mismatch")
    return evidence


def _open_regular_file_descriptor(path: Path) -> int:
    """以 lstat、O_NOFOLLOW 與 fstat 三重檢查開啟 ordinary file。"""

    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise ValueError("固定節點必須是 ordinary file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("固定節點不是 ordinary file")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_file_bytes(path: Path) -> bytes:
    """從 ordinary file descriptor 讀取完整 bytes，不追隨 symbolic link。"""

    descriptor = _open_regular_file_descriptor(path)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _require_regular_directory(path: Path) -> None:
    """要求 writer parent 是既有 ordinary non-symlink directory。"""

    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise ValueError("writer parent 必須是 ordinary directory")


def _fsync_directory(path: Path) -> None:
    """將同父目錄的 directory entry 變更同步到檔案系統。"""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_validation_evidence(path: str | Path) -> ValidationEvidence:
    """內部 strict reader，保留固定 stage/reason 供 public validator 使用。"""

    try:
        raw_bytes = _read_regular_file_bytes(Path(path))
    except Exception:
        _fail("topology", "not_regular_file")
    try:
        document = _strict_json_object(raw_bytes)
    except Exception:
        _fail("schema", "invalid_json")
    try:
        if raw_bytes != _canonical_json_bytes(document):
            _fail("schema", "not_canonical_json")
    except _ValidationEvidenceError:
        raise
    except Exception:
        _fail("schema", "not_canonical_json")
    try:
        return _evidence_from_document(document)
    except _ValidationEvidenceError:
        raise
    except Exception:
        _fail("schema", "schema_mismatch")


def load_validation_evidence(path: str | Path) -> ValidationEvidence:
    """strict 讀取一份 validation evidence，失敗時只拋固定且不含 path 的錯誤。

    reader 會拒絕 symlink、broken symlink、目錄、unknown/duplicate key、非 canonical JSON、
    非有限數值、來源 snapshot hash 不一致與 derived ``passed``／``all_passed`` 不一致。
    需要 stage/reason JSON-safe 結果時，請使用 :func:`validate_validation_evidence`。
    """

    try:
        return _load_validation_evidence(path)
    except Exception:
        raise ValueError(_LOAD_ERROR) from None


def _resolve_source_paths(
    *,
    source_paths: Mapping[str, str | Path] | None,
    aggregate_spec_path: str | Path | None,
    report_spec_path: str | Path | None,
    source_run_plan_path: str | Path | None,
    threshold_document_path: str | Path | None,
) -> dict[str, Path]:
    """解析 writer 的明示 source path，拒絕缺檔、猜路徑與混用兩種輸入形式。"""

    explicit = (
        aggregate_spec_path,
        report_spec_path,
        source_run_plan_path,
        threshold_document_path,
    )
    if source_paths is not None:
        if any(item is not None for item in explicit):
            raise ValueError("source_paths 不可與個別 source path 同時提供")
        if not isinstance(source_paths, Mapping) or set(source_paths) != set(_SOURCE_DOCUMENT_KEYS):
            raise ValueError("source_paths key set 不符")
        values = tuple(source_paths[name] for name in _SOURCE_DOCUMENT_KEYS)
    else:
        if any(item is None for item in explicit):
            raise ValueError("四個 source path 都必須明示")
        values = explicit
    result: dict[str, Path] = {}
    for name, value in zip(_SOURCE_DOCUMENT_KEYS, values, strict=True):
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{name} path 型別不符")
        result[name] = Path(value)
    return result


def _cleanup_owned_partial(
    partial_path: Path | None,
    partial_identity: tuple[int, int] | None,
) -> None:
    """只刪除本次 writer 以 device/inode 識別出的 ordinary partial。"""

    if partial_path is None or partial_identity is None:
        return
    try:
        node = os.lstat(partial_path)
    except FileNotFoundError:
        return
    except OSError:
        return
    if (
        stat.S_ISLNK(node.st_mode)
        or not stat.S_ISREG(node.st_mode)
        or (node.st_dev, node.st_ino) != partial_identity
    ):
        return
    try:
        os.unlink(partial_path)
    except FileNotFoundError:
        return
    except OSError:
        return


def write_validation_evidence(
    evidence: ValidationEvidence | None = None,
    destination: str | Path | None = None,
    *,
    run_id: str | None = None,
    metrics: Iterable[ValidationMetric] | None = None,
    source_paths: Mapping[str, str | Path] | None = None,
    aggregate_spec_path: str | Path | None = None,
    report_spec_path: str | Path | None = None,
    source_run_plan_path: str | Path | None = None,
    threshold_document_path: str | Path | None = None,
) -> Path:
    """以 atomic、non-overwrite 方式寫入一份 canonical validation evidence。

    可直接提供 ``run_id``／``metrics``，或提供一份由 constructor 建立的 ``evidence``；
    四個 source path 必須始終由 caller 明示，writer 會重新讀取並推導所有 source hash。
    destination parent 必須已存在且是 ordinary non-symlink directory。writer 先在同父
    目錄以 UUID 建立 exclusive hidden partial、寫入後 fsync 並用 strict reader 自驗證，
    再以不覆寫既有 entry 的 atomic hard-link 發布；既有 regular file、directory、symlink、
    broken symlink 或其他 partial 都不會被覆寫或清理。成功只代表 evidence engineering
    contract 完整，不代表 synthetic evidence 已成為真實資料科學驗證。
    """

    partial_path: Path | None = None
    partial_identity: tuple[int, int] | None = None
    published = False
    try:
        if destination is None:
            raise ValueError("destination 必須明示")
        destination_path = Path(destination)
        parent = destination_path.parent
        _require_regular_directory(parent)
        try:
            os.lstat(destination_path)
        except FileNotFoundError:
            pass
        else:
            raise _ValidationEvidenceError("topology", "final_exists")

        resolved_source_paths = _resolve_source_paths(
            source_paths=source_paths,
            aggregate_spec_path=aggregate_spec_path,
            report_spec_path=report_spec_path,
            source_run_plan_path=source_run_plan_path,
            threshold_document_path=threshold_document_path,
        )
        source_snapshots = {
            name: _read_source_document_text(path)
            for name, path in resolved_source_paths.items()
        }
        if evidence is not None:
            if type(evidence) is not ValidationEvidence:
                raise ValueError("evidence 必須是 exact ValidationEvidence")
            if run_id is not None or metrics is not None:
                raise ValueError("evidence 不可與 run_id／metrics 同時提供")
            prepared = ValidationEvidence(
                schema_version=evidence.schema_version,
                run_id=evidence.run_id,
                metrics=evidence.metrics,
                source_snapshots=source_snapshots,
            )
            if (
                prepared.run_id != evidence.run_id
                or prepared.metrics != evidence.metrics
                or prepared.source_raw_sha256 != evidence.source_raw_sha256
                or prepared.source_canonical_sha256 != evidence.source_canonical_sha256
            ):
                raise ValueError("evidence source binding 與明示 source 不一致")
        else:
            if run_id is None or metrics is None:
                raise ValueError("evidence 或 run_id／metrics 必須明示")
            prepared = ValidationEvidence(
                run_id=run_id,
                metrics=metrics,
                source_snapshots=source_snapshots,
            )

        payload = _canonical_json_bytes(prepared.to_dict())
        partial_path = parent / f".{destination_path.name}.{uuid4().hex}.partial"
        descriptor = os.open(
            partial_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            # partial 一旦由本 writer 建立，就立即保存 device/inode；即使第一個 write
            # 便失敗，清理流程仍只能刪除這個 writer 自己建立的 ordinary file。
            partial_status = os.fstat(descriptor)
            if not stat.S_ISREG(partial_status.st_mode):
                raise OSError("partial 必須是 ordinary file")
            partial_identity = (partial_status.st_dev, partial_status.st_ino)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("partial write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        inspected = _load_validation_evidence(partial_path)
        if inspected.to_dict() != prepared.to_dict():
            raise ValueError("partial self validation snapshot mismatch")

        # hard-link 建立 final directory entry 是 atomic 且不覆寫既有 entry；與 os.replace
        # 不同，即使另一個 process 在最後 lstat 後搶先建立 final，也只會失敗而不破壞它。
        os.link(partial_path, destination_path, follow_symlinks=False)
        published = True
        _fsync_directory(parent)
        os.unlink(partial_path)
        partial_path = None
        _fsync_directory(parent)
        return destination_path
    except _ValidationEvidenceError as error:
        _cleanup_owned_partial(partial_path, partial_identity)
        if error.reason == "final_exists":
            raise FileExistsError(_FINAL_EXISTS_ERROR) from None
        raise ValueError(_WRITE_ERROR) from None
    except FileExistsError:
        _cleanup_owned_partial(partial_path, partial_identity)
        raise FileExistsError(_FINAL_EXISTS_ERROR) from None
    except Exception:
        _cleanup_owned_partial(partial_path, partial_identity)
        if published:
            raise RuntimeError(_DURABILITY_ERROR) from None
        raise ValueError(_WRITE_ERROR) from None


def _validation_summary(evidence: ValidationEvidence) -> dict[str, object]:
    """建立不攜帶 path、raw source 或底層 exception 的 validator summary。"""

    return {
        "schema_version": evidence.schema_version,
        "run_id": evidence.run_id,
        "metric_count": len(evidence.metrics),
        "category_count": len(VALIDATION_METRIC_CATEGORIES),
        "source_document_count": len(_SOURCE_DOCUMENT_KEYS),
        "failed_metric_count": sum(not metric.passed for metric in evidence.metrics),
        "all_passed": evidence.all_passed,
    }


def validate_validation_evidence(path: str | Path) -> dict[str, object]:
    """驗證 evidence 並回傳固定 JSON-safe ``valid/errors/summary``。

    ``valid=True`` 表示 evidence 的 schema、七類 closure、定量 metric、source snapshot
    與 checksum contract 都完整；``summary.all_passed`` 另表示所有 metric 是否達到預先
    登錄門檻。故合法但含 failed metric 的 evidence 仍可被讀取供診斷，而下游 scientific
    pipeline 必須另外以 ``all_passed`` 阻擋未通過的 F12/T06 發布。失敗回報只含固定
    stage/reason code，不含 absolute path、檔名或底層例外文字。
    """

    try:
        evidence = _load_validation_evidence(path)
    except _ValidationEvidenceError as error:
        return {
            "valid": False,
            "errors": [{"stage": error.stage, "reason": error.reason}],
            "summary": {},
        }
    except Exception:
        return {
            "valid": False,
            "errors": [{"stage": "decoder", "reason": "unexpected_failure"}],
            "summary": {},
        }
    return {"valid": True, "errors": [], "summary": _validation_summary(evidence)}
