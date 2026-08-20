"""上游 OCM/NWW3 月份產品的唯讀、低記憶體 preflight。

preflight 只讀 metadata 與 ``time_utc_ns.npy`` 的 memory-map，不掃描完整四維速度或
波浪陣列。它驗證 schema major、月份、status、必要檔案與 metadata shape；時間則先
跨月份 stable sort、對重複 UTC 採 prefer-last，再以完整研究期間的逐時參考軸盤點缺口。
這可避免把月份 halo、跨月倒序與同一缺口重複列成多個錯誤。報告以 root token 取代實際
絕對路徑；正式數值 QC、缺口重建 skill 與軌跡敏感度仍由後續 manifest 驗證。
"""

from __future__ import annotations

import json
import os
import tempfile
from calendar import monthrange
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .time_axis import CanonicalTimeAxis, TimeChunk, canonicalize_time_chunks


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


@dataclass(frozen=True, slots=True)
class TimeGapInventory:
    """研究期間逐時參考軸上的一段連續缺時。

    內部缺口有左右可用端點，``gap_hours`` 是兩端時距；研究期間最前或最後的 boundary
    缺時沒有雙側支撐，因此 ``gap_hours`` 為 ``None``，不能誤送進雙向重建器。
    """

    missing_start_utc: str
    missing_end_utc: str
    missing_step_count: int
    before_utc: str | None
    after_utc: str | None
    gap_hours: float | None


@dataclass(slots=True)
class TimeAxisInventory:
    """單一產品/domain 的跨月 canonical 時間軸與可用率摘要。"""

    product: str
    flow_domain_id: str
    policy: str
    expected_timestep_hours: float
    input_time_count: int
    canonical_time_count: int
    reordered_time_step_count: int
    dropped_duplicate_time_step_count: int
    expected_period_time_count: int
    available_period_time_count: int
    missing_period_time_count: int
    extra_halo_time_count: int
    coverage_fraction: float
    time_start_utc: str
    time_end_utc: str
    maximum_internal_gap_hours: float
    gaps: list[TimeGapInventory] = field(default_factory=list)


@dataclass(slots=True)
class PreflightReport:
    """可直接寫入 JSON manifest 的完整 preflight 結果。"""

    created_at_utc: str
    config_hash: str
    mode: str
    inventories: list[MonthInventory] = field(default_factory=list)
    time_axes: list[TimeAxisInventory] = field(default_factory=list)
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
            "time_axes": [asdict(item) for item in self.time_axes],
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


def _expected_hourly_axis(months: Iterable[str], *, expected_timestep_hours: float) -> np.ndarray:
    """建立所選曆月聯集的 UTC 參考軸，不把月份 cache halo 誤算為研究樣本。

    正式設定目前為逐時資料。函式仍依設定步長建立每月 ``[month_start, next_month_start)``
    軸，使單月 preflight、完整兩年 preflight 與未來不同固定步長使用同一計數定義。
    """

    step_ns = int(round(expected_timestep_hours * 3_600_000_000_000))
    if step_ns <= 0:
        raise ValueError("expected_timestep_hours 必須為正值")
    pieces: list[np.ndarray] = []
    for month in sorted(months):
        year = int(month[:4])
        month_number = int(month[4:])
        start = datetime(year, month_number, 1, tzinfo=UTC)
        day_count = monthrange(year, month_number)[1]
        start_ns = int(start.timestamp() * 1_000_000_000)
        end_ns = start_ns + day_count * 24 * 3_600_000_000_000
        pieces.append(np.arange(start_ns, end_ns, step_ns, dtype=np.int64))
    if not pieces:
        raise ValueError("至少需要一個 YYYYMM 才能建立參考時間軸")
    return np.concatenate(pieces)


def _missing_runs(expected: np.ndarray, available: np.ndarray) -> list[TimeGapInventory]:
    """把參考軸缺時轉成連續區段，保留雙側端點與 boundary 差異。

    ``available`` 可含月份 halo；只要 UTC 出現在 ``expected`` 即視為該研究時次有來源。
    缺口區段以參考軸索引判斷，不會因跨月目錄分片而重複列出同一事件。
    """

    present = np.isin(expected, available, assume_unique=False)
    runs: list[TimeGapInventory] = []
    start = 0
    while start < present.size:
        if present[start]:
            start += 1
            continue
        stop = start + 1
        while stop < present.size and not present[stop]:
            stop += 1
        before_ns = int(expected[start - 1]) if start > 0 and present[start - 1] else None
        after_ns = int(expected[stop]) if stop < present.size and present[stop] else None
        gap_hours = (
            (after_ns - before_ns) / 3_600_000_000_000
            if before_ns is not None and after_ns is not None
            else None
        )
        runs.append(
            TimeGapInventory(
                missing_start_utc=_utc_from_ns(int(expected[start])),
                missing_end_utc=_utc_from_ns(int(expected[stop - 1])),
                missing_step_count=stop - start,
                before_utc=_utc_from_ns(before_ns) if before_ns is not None else None,
                after_utc=_utc_from_ns(after_ns) if after_ns is not None else None,
                gap_hours=float(gap_hours) if gap_hours is not None else None,
            )
        )
        start = stop
    return runs


def _inspect_canonical_time_axis(
    *,
    product: str,
    root: Path,
    root_token: str,
    flow_domain_id: str,
    months: Iterable[str],
    policy: str,
    expected_timestep_hours: float,
    findings: list[Finding],
) -> tuple[CanonicalTimeAxis, TimeAxisInventory] | None:
    """載入小型月時間軸，建立跨月索引並量化完整期間 coverage。

    物理陣列不在此載入。若任一月時間檔缺失或不可讀，月份 inventory 已會指出檔案問題；
    本函式另以單一 canonical error 說明無法形成完整來源索引，避免輸出誤導的 gap 統計。
    """

    chunks: list[TimeChunk] = []
    ordered_months = sorted(months)
    for month in ordered_months:
        relative = Path(flow_domain_id) / "months" / month / "time_utc_ns.npy"
        path = root / relative
        try:
            chunks.append(TimeChunk(month, np.load(path, mmap_mode="r", allow_pickle=False)))
        except (OSError, TypeError, ValueError) as exc:
            findings.append(
                Finding(
                    "error",
                    "CANONICAL_TIME_AXIS_UNREADABLE",
                    _tokenized_path(root_token, relative),
                    str(exc),
                )
            )
            return None
    try:
        canonical = canonicalize_time_chunks(
            chunks,
            policy=policy,  # type: ignore[arg-type]
            expected_timestep_hours=expected_timestep_hours,
        )
        expected = _expected_hourly_axis(
            ordered_months,
            expected_timestep_hours=expected_timestep_hours,
        )
    except ValueError as exc:
        findings.append(
            Finding(
                "error",
                "CANONICAL_TIME_AXIS_INVALID",
                f"${root_token}/{flow_domain_id}/months",
                str(exc),
            )
        )
        return None

    available_mask = np.isin(expected, canonical.time_utc_ns, assume_unique=False)
    available_count = int(np.count_nonzero(available_mask))
    gaps = _missing_runs(expected, canonical.time_utc_ns)
    internal_gap_hours = [item.gap_hours for item in gaps if item.gap_hours is not None]
    inventory = TimeAxisInventory(
        product=product,
        flow_domain_id=flow_domain_id,
        policy=canonical.policy,
        expected_timestep_hours=expected_timestep_hours,
        input_time_count=canonical.input_time_count,
        canonical_time_count=int(canonical.time_utc_ns.size),
        reordered_time_step_count=canonical.reordered_time_step_count,
        dropped_duplicate_time_step_count=canonical.dropped_duplicate_time_step_count,
        expected_period_time_count=int(expected.size),
        available_period_time_count=available_count,
        missing_period_time_count=int(expected.size - available_count),
        extra_halo_time_count=int(np.count_nonzero(~np.isin(canonical.time_utc_ns, expected))),
        coverage_fraction=available_count / int(expected.size),
        time_start_utc=_utc_from_ns(int(canonical.time_utc_ns[0])),
        time_end_utc=_utc_from_ns(int(canonical.time_utc_ns[-1])),
        maximum_internal_gap_hours=max(internal_gap_hours, default=expected_timestep_hours),
        gaps=gaps,
    )
    location = f"${root_token}/{flow_domain_id}/months"
    if canonical.dropped_duplicate_time_step_count:
        findings.append(
            Finding(
                "info",
                "TIME_DUPLICATES_CANONICALIZED",
                location,
                (
                    f"依 {canonical.policy} 去除 {canonical.dropped_duplicate_time_step_count} 筆重複 UTC；"
                    f"重排計數={canonical.reordered_time_step_count}"
                ),
            )
        )
    if inventory.missing_period_time_count:
        code = (
            "OCM_TIME_RECONSTRUCTION_REQUIRED"
            if product == "ocm_native"
            else "NWW_ANALYSIS_FULL_HOURLY_REBUILD_REQUIRED"
        )
        findings.append(
            Finding(
                "warning",
                code,
                location,
                (
                    f"全部可得資料契約下缺 {inventory.missing_period_time_count}/"
                    f"{inventory.expected_period_time_count} 個規則時次，連續缺口 {len(gaps)} 段，"
                    f"最大內部端點時距 {inventory.maximum_internal_gap_hours:.6g} h；"
                    "此 finding 要求重建／gap-safe manifest，不代表等待供應者補件"
                ),
            )
        )
    return canonical, inventory


def _compare_canonical_time_support(
    *,
    ocm: CanonicalTimeAxis,
    nww: CanonicalTimeAxis,
    flow_domain_id: str,
    findings: list[Finding],
) -> None:
    """確認 NWW 支援所有 OCM UTC；允許完整 hourly NWW 比 gappy OCM 多出時次。

    NWW native 在 SERVER 上具有完整 17,544 個逐時時次，因此正式 analysis 將以靜態 OCM
    grid 重建完整 hourly 產品。要求兩軸逐值相等反而會阻止這項修正；真正的耦合條件是
    每一筆觀測或重建 OCM 時次都必須可取得同 UTC 波浪，不能以 nearest time 猜測。
    """

    unsupported = np.setdiff1d(ocm.time_utc_ns, nww.time_utc_ns, assume_unique=True)
    if unsupported.size:
        findings.append(
            Finding(
                "error",
                "NWW_MISSING_OCM_TIME_SUPPORT",
                flow_domain_id,
                f"NWW canonical 軸缺少 {unsupported.size} 個 OCM UTC 支撐",
            )
        )
    extra = np.setdiff1d(nww.time_utc_ns, ocm.time_utc_ns, assume_unique=True)
    if extra.size:
        findings.append(
            Finding(
                "info",
                "NWW_EXTRA_HOURLY_SUPPORT",
                flow_domain_id,
                f"NWW 比原始 OCM 多 {extra.size} 個逐時時次，可供 OCM 缺口重建後同時取樣",
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
    accepted_statuses: set[str],
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
    if status not in accepted_statuses:
        findings.append(
            Finding(
                "error",
                "STATUS_REJECTED",
                location,
                f"status={status} 不在全部可得資料契約 {sorted(accepted_statuses)}",
            )
        )
    elif status == "trial_ready":
        # 上游名稱只表示 forecast-cycle 優先序不宣稱為最佳預報。研究團隊已決定資料
        # 提供者不可考，故 LBT 以固定 lexical selection 與完整 audit 接受它；仍在報告留下
        # info，防止成果文字把它誤寫成 provider-confirmed analysis cycle。
        findings.append(
            Finding(
                "info",
                "AVAILABLE_SAMPLE_STATUS_ACCEPTED",
                location,
                "接受 trial_ready 的全部可得樣本；不等待供應者確認，也不宣稱為最佳 forecast cycle",
            )
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
    elif cache_kind == "standard_partial_month":
        findings.append(
            Finding(
                "info",
                "AVAILABLE_PARTIAL_MONTH_ACCEPTED",
                location,
                "依全部可得 2024–2025 契約納入 partial month；缺時由跨月 canonical coverage 統一處理",
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

    若未指定 ``months``，依設定年份產生 24 個 ``YYYYMM``。月份 status/cache kind 依
    ``available_2024_2025`` 契約判定；partial 或 ``trial_ready`` 名稱本身不再是正式阻擋，
    但仍以 info 保存其限制。正式模式除輸入 error 外，還會先驗證重建、幾何與收斂等
    衍生 manifest，避免把「接受全部可得資料」誤解為「可略過科學驗證」。
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
    ocm_accepted_kinds = {str(value) for value in ocm_contract.get("accepted_cache_kinds", [])}
    nww_accepted_kinds = {str(value) for value in nww_contract.get("accepted_cache_kinds", [])}
    ocm_accepted_statuses = {
        str(value)
        for value in ocm_contract.get(
            "accepted_statuses",
            [ocm_contract.get("required_status", "ready")],
        )
    }
    nww_accepted_statuses = {
        str(value)
        for value in nww_contract.get(
            "accepted_statuses",
            [nww_contract.get("required_status", "ready")],
        )
    }
    time_contract = config.inputs.time_axis_contract
    policy = str(time_contract["canonicalization_policy"])
    expected_timestep_hours = float(time_contract["expected_timestep_hours"])
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
                    accepted_statuses=ocm_accepted_statuses,
                    required_arrays=ocm_required,
                    accepted_cache_kinds=ocm_accepted_kinds,
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
                    accepted_statuses=nww_accepted_statuses,
                    required_arrays=nww_required,
                    accepted_cache_kinds=nww_accepted_kinds or None,
                    findings=report.findings,
                )
            )
        ocm_axis_result = _inspect_canonical_time_axis(
            product="ocm_native",
            root=Path(ocm_native_root),
            root_token=config.inputs.ocm_native_root_env,
            flow_domain_id=resolved_domain_id,
            months=month_labels,
            policy=policy,
            expected_timestep_hours=expected_timestep_hours,
            findings=report.findings,
        )
        nww_axis_result = _inspect_canonical_time_axis(
            product="nww3_analysis",
            root=Path(nww_analysis_root),
            root_token=config.inputs.nww_analysis_root_env,
            flow_domain_id=resolved_domain_id,
            months=month_labels,
            policy=policy,
            expected_timestep_hours=expected_timestep_hours,
            findings=report.findings,
        )
        if ocm_axis_result is not None:
            report.time_axes.append(ocm_axis_result[1])
        if nww_axis_result is not None:
            report.time_axes.append(nww_axis_result[1])
        if ocm_axis_result is not None and nww_axis_result is not None:
            _compare_canonical_time_support(
                ocm=ocm_axis_result[0],
                nww=nww_axis_result[0],
                flow_domain_id=resolved_domain_id,
                findings=report.findings,
            )
    return report
