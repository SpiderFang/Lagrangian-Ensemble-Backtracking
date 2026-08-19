"""上游 OCM/NWW3 月份產品的唯讀、低記憶體 preflight。

preflight 只讀 metadata 與 ``time_utc_ns.npy`` 的 memory-map，不掃描完整四維速度或
波浪陣列。它驗證 schema major、月份、status、必要檔案、metadata shape 與時間軸，
並以 root token 取代報告中的實際絕對路徑。正式數值 QC 仍需抽樣讀取實值陣列；本模組
的職責是先阻擋明確不相容或未驗收的產品。
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig


@dataclass(frozen=True, slots=True)
class Finding:
    """單一 preflight 發現；``error`` 會阻擋正式批次，``warning`` 只允許 pilot。"""

    severity: str
    code: str
    location: str
    message: str


@dataclass(slots=True)
class MonthInventory:
    """單一產品、domain 與月份的低記憶體契約摘要。"""

    product: str
    flow_domain_id: str
    month: str
    status: str | None
    schema_version: str | None
    cache_kind: str | None
    time_count: int | None
    time_start_utc: str | None
    time_end_utc: str | None
    maximum_gap_seconds: float | None
    required_arrays_present: bool
    path_token: str


@dataclass(slots=True)
class PreflightReport:
    """可直接寫入 JSON manifest 的完整 preflight 結果。"""

    created_at_utc: str
    config_hash: str
    mode: str
    inventories: list[MonthInventory] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def formal_ready(self) -> bool:
        """沒有 error 才表示輸入契約可進正式批次。"""

        return not any(item.severity == "error" for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """轉為無 NaN、可排序寫入 JSON 的資料。"""

        return {
            "created_at_utc": self.created_at_utc,
            "config_hash": self.config_hash,
            "mode": self.mode,
            "formal_ready": self.formal_ready,
            "inventories": [asdict(item) for item in self.inventories],
            "findings": [asdict(item) for item in self.findings],
        }

    def write_json(self, path: str | Path) -> None:
        """在目標目錄內建立暫存檔後原子改名，避免中斷留下半份證據。"""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, destination)


def _read_json(path: Path) -> dict[str, Any]:
    """讀取 JSON object；格式或根型別錯誤交由 caller 記為 finding。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON 根節點不是 object")
    return payload


def _schema_major(value: object) -> int | None:
    """由 ``3.0.0`` 取 major；缺值或非整數開頭回傳 ``None``。"""

    try:
        return int(str(value).split(".", maxsplit=1)[0])
    except (TypeError, ValueError):
        return None


def _utc_from_ns(value: int) -> str:
    """把 epoch nanoseconds 轉成固定 UTC ISO 字串。"""

    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC).isoformat().replace("+00:00", "Z")


def _tokenized_path(root_token: str, relative_path: Path) -> str:
    """只在報告保存環境變數 token 與相對路徑，不洩漏 SERVER 絕對目錄。"""

    return f"${root_token}/{relative_path.as_posix()}"


def _array_contract(metadata: dict[str, Any], name: str) -> tuple[tuple[int, ...] | None, str | None]:
    """從 metadata 的 arrays mapping 取得 shape/dtype，不開啟大型資料。"""

    arrays = metadata.get("arrays")
    item = arrays.get(name) if isinstance(arrays, dict) else None
    if not isinstance(item, dict):
        return None, None
    raw_shape = item.get("shape")
    shape = tuple(int(value) for value in raw_shape) if isinstance(raw_shape, list) else None
    dtype = str(item.get("dtype")) if item.get("dtype") is not None else None
    return shape, dtype


def _inspect_grid(
    *,
    product: str,
    root: Path,
    root_token: str,
    flow_domain_id: str,
    required_arrays: tuple[str, ...],
    expected_schema_major: int,
    findings: list[Finding],
) -> None:
    """檢查靜態 grid metadata、識別碼與必要陣列存在性。

    OCM grid metadata 使用 ``cache_schema_version`` 且 domain ID 位於 ``domain`` object；
    NWW grid 使用 ``schema_version`` 與頂層 ``flow_domain_id``。只要二者與設定不一致，
    即使月份 shape 看似可讀也不能假設是同一空間格網。
    """

    relative = Path(flow_domain_id) / "grid"
    grid_dir = root / relative
    location = _tokenized_path(root_token, relative)
    try:
        metadata = _read_json(grid_dir / "metadata.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", "GRID_METADATA_UNREADABLE", location, str(exc)))
        return
    schema_value = (
        metadata.get("cache_schema_version") if product == "ocm_native" else metadata.get("schema_version")
    )
    if _schema_major(schema_value) != expected_schema_major:
        findings.append(
            Finding(
                "error",
                "GRID_SCHEMA_MAJOR_MISMATCH",
                location,
                f"預期 major={expected_schema_major}，實際={schema_value}",
            )
        )
    actual_domain = (
        metadata.get("domain", {}).get("domain_id")
        if product == "ocm_native" and isinstance(metadata.get("domain"), dict)
        else metadata.get("flow_domain_id")
    )
    if actual_domain != flow_domain_id:
        findings.append(
            Finding(
                "error",
                "GRID_DOMAIN_ID_MISMATCH",
                location,
                f"預期 flow_domain_id={flow_domain_id}，實際={actual_domain}",
            )
        )
    for name in required_arrays:
        if not (grid_dir / name).is_file():
            findings.append(
                Finding("error", "GRID_ARRAY_MISSING", f"{location}/{name}", "缺少必要靜態格網陣列")
            )


def _compare_month_time_axes(
    *,
    ocm_root: Path,
    nww_root: Path,
    ocm_root_token: str,
    nww_root_token: str,
    flow_domain_id: str,
    month: str,
    maximum_nww_gap_hours: float,
    findings: list[Finding],
) -> None:
    """確認 OCM 與 NWW target UTC 逐值一致，並檢查 NWW 最大時間缺口。

    每月時間軸只有約數百個 int64，逐值比較不會載入大型物理陣列。不同時間軸不可由
    下游猜測 nearest time，因那會使 RK stage 的 current 與 wave 不同步。
    """

    relative = Path(flow_domain_id) / "months" / month / "time_utc_ns.npy"
    ocm_path = ocm_root / relative
    nww_path = nww_root / relative
    if not ocm_path.is_file() or not nww_path.is_file():
        return
    try:
        ocm_time = np.load(ocm_path, mmap_mode="r", allow_pickle=False)
        nww_time = np.load(nww_path, mmap_mode="r", allow_pickle=False)
        if ocm_time.shape != nww_time.shape or not np.array_equal(ocm_time, nww_time):
            findings.append(
                Finding(
                    "error",
                    "OCM_NWW_TIME_MISMATCH",
                    f"${ocm_root_token}/{relative.as_posix()} ↔ ${nww_root_token}/{relative.as_posix()}",
                    "OCM 與 NWW target time 軸不是逐值一致",
                )
            )
        if nww_time.size >= 2:
            maximum_gap_hours = float(np.max(np.diff(nww_time)) / 3_600_000_000_000)
            if maximum_gap_hours > maximum_nww_gap_hours + 1e-12:
                findings.append(
                    Finding(
                        "error",
                        "NWW_TIME_GAP_EXCEEDED",
                        f"${nww_root_token}/{relative.as_posix()}",
                        f"最大 gap={maximum_gap_hours:.6g} h，限制={maximum_nww_gap_hours:.6g} h",
                    )
                )
    except (OSError, TypeError, ValueError) as exc:
        findings.append(Finding("error", "TIME_ALIGNMENT_UNREADABLE", f"{flow_domain_id}/{month}", str(exc)))


def _compare_cross_month_time_boundaries(
    *,
    nww_root: Path,
    nww_root_token: str,
    flow_domain_id: str,
    months: Iterable[str],
    maximum_gap_hours: float,
    findings: list[Finding],
) -> None:
    """以曆月切換 UTC 檢查相鄰 cache 的共同時間支撐。

    OCM/NWW 每月時間軸已由 ``_compare_month_time_axes`` 逐值比對，因此只需讀一份 NWW
    軸即可代表共同 forcing boundary。上游 monthly cache 可合法包含前後數日 halo，不能
    用整個陣列的 first/last 判斷為 overlap error；函式改取前月在曆月界以前的最後支撐與
    次月在曆月界以後的第一支撐。缺檔由月份 inventory 另報，這裡不重複 missing finding。
    """

    ordered = sorted(months)
    for previous_month, current_month in zip(ordered[:-1], ordered[1:], strict=True):
        previous_relative = Path(flow_domain_id) / "months" / previous_month / "time_utc_ns.npy"
        current_relative = Path(flow_domain_id) / "months" / current_month / "time_utc_ns.npy"
        previous_path = nww_root / previous_relative
        current_path = nww_root / current_relative
        if not previous_path.is_file() or not current_path.is_file():
            continue
        try:
            previous = np.load(previous_path, mmap_mode="r", allow_pickle=False)
            current = np.load(current_path, mmap_mode="r", allow_pickle=False)
            if previous.ndim != 1 or current.ndim != 1 or previous.size == 0 or current.size == 0:
                continue
            boundary = datetime(
                int(current_month[:4]),
                int(current_month[4:]),
                1,
                tzinfo=UTC,
            )
            boundary_ns = int(boundary.timestamp() * 1_000_000_000)
            previous_support = previous[previous <= boundary_ns]
            current_support = current[current >= boundary_ns]
            location = (
                f"${nww_root_token}/{previous_relative.as_posix()} ↔ "
                f"${nww_root_token}/{current_relative.as_posix()}"
            )
            if previous_support.size == 0 or current_support.size == 0:
                findings.append(
                    Finding(
                        "error",
                        "CROSS_MONTH_BOUNDARY_UNSUPPORTED",
                        location,
                        f"曆月界 {boundary.isoformat()} 前後缺少可用時間支撐",
                    )
                )
                continue
            gap_hours = (int(current_support[0]) - int(previous_support[-1])) / 3_600_000_000_000
            if gap_hours > maximum_gap_hours + 1.0e-12:
                findings.append(
                    Finding(
                        "error",
                        "CROSS_MONTH_TIME_GAP_EXCEEDED",
                        location,
                        f"跨月 gap={gap_hours:.6g} h，限制={maximum_gap_hours:.6g} h",
                    )
                )
        except (OSError, TypeError, ValueError) as exc:
            findings.append(
                Finding(
                    "error",
                    "CROSS_MONTH_TIME_UNREADABLE",
                    f"{flow_domain_id}/{previous_month}-{current_month}",
                    str(exc),
                )
            )


def _inspect_month(
    *,
    product: str,
    root: Path,
    root_token: str,
    flow_domain_id: str,
    month: str,
    required_schema_major: int,
    required_status: str,
    required_arrays: Iterable[str],
    accepted_cache_kinds: set[str] | None,
    findings: list[Finding],
) -> MonthInventory:
    """檢查一個月份的 metadata、必要檔案與時間軸。

    所有錯誤都累積進 ``findings``，讓一次 preflight 能列出完整阻擋清單；只有無法安全
    讀取的檔案才使該 inventory 欄位保持 ``None``。時間軸以 memory-map 檢查遞增與最大
    gap，資料內容不載入 RAM。
    """

    relative = Path(flow_domain_id) / "months" / month
    month_dir = root / relative
    location = _tokenized_path(root_token, relative)
    metadata_path = month_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    try:
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", "METADATA_UNREADABLE", location, str(exc)))

    schema_value = (
        metadata.get("cache_schema_version") if product == "ocm_native" else metadata.get("schema_version")
    )
    status = str(metadata.get("status")) if metadata.get("status") is not None else None
    cache_kind = str(metadata.get("cache_kind")) if metadata.get("cache_kind") is not None else None
    if _schema_major(schema_value) != required_schema_major:
        findings.append(
            Finding(
                "error",
                "SCHEMA_MAJOR_MISMATCH",
                location,
                f"預期 major={required_schema_major}，實際={schema_value}",
            )
        )
    if status != required_status:
        severity = "error" if required_status == "ready" else "warning"
        findings.append(
            Finding(severity, "STATUS_NOT_READY", location, f"預期 status={required_status}，實際={status}")
        )
    if accepted_cache_kinds is not None and cache_kind not in accepted_cache_kinds:
        findings.append(
            Finding(
                "error",
                "CACHE_KIND_REJECTED",
                location,
                f"cache_kind={cache_kind} 不在 {sorted(accepted_cache_kinds)}",
            )
        )

    arrays_present = True
    time_count: int | None = None
    start: str | None = None
    end: str | None = None
    maximum_gap_seconds: float | None = None
    for name in required_arrays:
        path = month_dir / name
        expected_shape, expected_dtype = _array_contract(metadata, name)
        if not path.is_file() or expected_shape is None or expected_dtype is None:
            arrays_present = False
            findings.append(
                Finding("error", "REQUIRED_ARRAY_MISSING", f"{location}/{name}", "檔案或 metadata 契約缺失")
            )
            continue
        try:
            # mmap 只解析 NPY header 並建立虛擬映射，不會把 hvel 等大型四維陣列載入 RAM。
            # 必須比對實際 header，否則 metadata 與被截斷/替換檔案不一致時仍會誤通過。
            actual = np.load(path, mmap_mode="r", allow_pickle=False)
            if tuple(actual.shape) != expected_shape or str(actual.dtype) != expected_dtype:
                arrays_present = False
                findings.append(
                    Finding(
                        "error",
                        "ARRAY_CONTRACT_MISMATCH",
                        f"{location}/{name}",
                        (
                            f"metadata shape/dtype={expected_shape}/{expected_dtype}，"
                            f"實際={tuple(actual.shape)}/{actual.dtype}"
                        ),
                    )
                )
            del actual
        except (OSError, TypeError, ValueError) as exc:
            arrays_present = False
            findings.append(Finding("error", "ARRAY_HEADER_UNREADABLE", f"{location}/{name}", str(exc)))

    time_path = month_dir / "time_utc_ns.npy"
    if time_path.is_file():
        try:
            time_values = np.load(time_path, mmap_mode="r", allow_pickle=False)
            expected_shape, expected_dtype = _array_contract(metadata, "time_utc_ns.npy")
            if time_values.ndim != 1 or time_values.dtype != np.dtype("int64"):
                raise ValueError("time_utc_ns 必須是一維 int64")
            if expected_shape != tuple(time_values.shape) or expected_dtype != str(time_values.dtype):
                raise ValueError("time_utc_ns 實際 shape/dtype 與 metadata 不符")
            time_count = int(time_values.size)
            if time_count < 2:
                raise ValueError("time_utc_ns 至少需要兩個時次")
            differences = np.diff(time_values)
            if np.any(differences <= 0):
                raise ValueError("time_utc_ns 必須嚴格遞增且唯一")
            start = _utc_from_ns(int(time_values[0]))
            end = _utc_from_ns(int(time_values[-1]))
            maximum_gap_seconds = float(np.max(differences) / 1_000_000_000)
        except (OSError, TypeError, ValueError) as exc:
            findings.append(Finding("error", "TIME_AXIS_INVALID", f"{location}/time_utc_ns.npy", str(exc)))

    return MonthInventory(
        product=product,
        flow_domain_id=flow_domain_id,
        month=month,
        status=status,
        schema_version=str(schema_value) if schema_value is not None else None,
        cache_kind=cache_kind,
        time_count=time_count,
        time_start_utc=start,
        time_end_utc=end,
        maximum_gap_seconds=maximum_gap_seconds,
        required_arrays_present=arrays_present,
        path_token=location,
    )


def run_preflight(
    config: ProjectConfig,
    *,
    ocm_native_root: str | Path,
    nww_analysis_root: str | Path,
    months: Iterable[str] | None = None,
    formal_release: bool = False,
) -> PreflightReport:
    """對所有設定 domain 執行 OCM/NWW 月份低記憶體 inventory。

    若未指定 ``months``，依設定年份產生 24 個 ``YYYYMM``。開發模式仍使用正式 status
    要求產生 error，但 caller 可保存報告並繼續合成/pilot；正式模式除輸入 error 外，
    還會先驗證設定衍生閘門並於失敗時直接回報例外。
    """

    if formal_release:
        config.assert_formal_release_ready()
    month_labels = (
        list(months)
        if months is not None
        else [f"{year}{month:02d}" for year in config.inputs.years for month in range(1, 13)]
    )
    if len(set(month_labels)) != len(month_labels) or any(
        len(item) != 6 or not item.isdigit() for item in month_labels
    ):
        raise ValueError("months 必須是唯一的 YYYYMM 字串")

    report = PreflightReport(
        created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        config_hash=config.config_hash(),
        mode="formal_release" if formal_release else "development",
    )
    ocm_contract = config.inputs.ocm_contract
    nww_contract = config.inputs.nww_contract
    ocm_required = tuple(str(name) for name in ocm_contract["required_month_arrays"])
    nww_required = tuple(str(name) for name in nww_contract["required_arrays"])
    accepted_kinds = {str(value) for value in ocm_contract.get("accepted_cache_kinds", [])}
    for domain in config.domains:
        resolved_domain_id = (
            domain.formal_release_flow_domain_id
            if formal_release and domain.formal_release_flow_domain_id
            else domain.flow_domain_id
        )
        _inspect_grid(
            product="ocm_native",
            root=Path(ocm_native_root),
            root_token=config.inputs.ocm_native_root_env,
            flow_domain_id=resolved_domain_id,
            required_arrays=tuple(str(name) for name in ocm_contract["required_grid_arrays"]),
            expected_schema_major=int(ocm_contract["required_schema_major"]),
            findings=report.findings,
        )
        _inspect_grid(
            product="nww3_analysis",
            root=Path(nww_analysis_root),
            root_token=config.inputs.nww_analysis_root_env,
            flow_domain_id=resolved_domain_id,
            required_arrays=("lon.npy", "lat.npy", "mask_static.npy"),
            expected_schema_major=int(nww_contract["required_schema_major"]),
            findings=report.findings,
        )
        for month in month_labels:
            report.inventories.append(
                _inspect_month(
                    product="ocm_native",
                    root=Path(ocm_native_root),
                    root_token=config.inputs.ocm_native_root_env,
                    flow_domain_id=resolved_domain_id,
                    month=month,
                    required_schema_major=int(ocm_contract["required_schema_major"]),
                    required_status=str(ocm_contract["required_status"]),
                    required_arrays=ocm_required,
                    accepted_cache_kinds=accepted_kinds,
                    findings=report.findings,
                )
            )
            report.inventories.append(
                _inspect_month(
                    product="nww3_analysis",
                    root=Path(nww_analysis_root),
                    root_token=config.inputs.nww_analysis_root_env,
                    flow_domain_id=resolved_domain_id,
                    month=month,
                    required_schema_major=int(nww_contract["required_schema_major"]),
                    required_status=str(nww_contract["required_status"]),
                    required_arrays=nww_required,
                    accepted_cache_kinds=None,
                    findings=report.findings,
                )
            )
            _compare_month_time_axes(
                ocm_root=Path(ocm_native_root),
                nww_root=Path(nww_analysis_root),
                ocm_root_token=config.inputs.ocm_native_root_env,
                nww_root_token=config.inputs.nww_analysis_root_env,
                flow_domain_id=resolved_domain_id,
                month=month,
                maximum_nww_gap_hours=float(nww_contract["maximum_time_gap_hours"]),
                findings=report.findings,
            )
        _compare_cross_month_time_boundaries(
            nww_root=Path(nww_analysis_root),
            nww_root_token=config.inputs.nww_analysis_root_env,
            flow_domain_id=resolved_domain_id,
            months=month_labels,
            maximum_gap_hours=float(nww_contract["maximum_time_gap_hours"]),
            findings=report.findings,
        )
    return report
