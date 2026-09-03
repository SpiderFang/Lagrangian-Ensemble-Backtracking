"""正式報告成果 registry 的 immutable typed foundation。

本模組只保存報告圖表、表格、caption 與 data sidecar 的資料契約，不執行檔案讀寫、
圖表繪製、統計計算或資料來源推論，因此故意不匯入 NumPy、PyArrow、Matplotlib 或
其他大型科學／I/O 套件。每個 product path 都是相對於未來 report release 根目錄的
安全 POSIX 路徑；大小使用 bytes，距離與格網沿用公尺（m），travel age 與時間沿用
秒（s），經緯度／投影識別則只作資料交換與顯示座標。

registry 的證據類別刻意把本機 synthetic engineering evidence、SERVER pilot evidence、
SERVER formal baseline evidence 與 SERVER scientific evidence 分開。formal baseline
可以在兩組 allow policy 明示且成對缺 comparison／validation 時保存工程基準，但不因此
取得 scientific evidence 資格。通過 constructor 只代表 registry 的 immutable
工程契約、來源雜湊、產品角色與缺證政策一致；它不把本機合成測試變成真實 OCM schema 3
或 NWW3 schema 1 科學成果，也不允許將條件式來源足跡／相對來源權重稱為絕對來源機率
或因果歸因。後續 writer、renderer 與 validator 會在各自的 I/O 邊界再次驗證這些 records。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

__all__ = [
    "REPORT_RELEASE_SCHEMA_VERSION",
    "REPORT_FIGURE_IDS",
    "REPORT_TABLE_IDS",
    "REPORT_CORE_ARTIFACT_IDS",
    "REPORT_COMPARISON_ARTIFACT_IDS",
    "REPORT_VALIDATION_ARTIFACT_IDS",
    "ReportProductRef",
    "ReportArtifactRecord",
    "ReportRegistry",
]


# 報告 schema 與 F/T artifact closure 都是 exact contract；未知版本不可由 caller 猜測
# 欄位語意後繼續發布，避免 renderer 與 validator 對同一列產生不同解讀。
REPORT_RELEASE_SCHEMA_VERSION: Final[str] = "1.0.0"
REPORT_FIGURE_IDS: Final[tuple[str, ...]] = tuple(f"F{index:02d}" for index in range(1, 13))
REPORT_TABLE_IDS: Final[tuple[str, ...]] = tuple(f"T{index:02d}" for index in range(1, 7))
REPORT_CORE_ARTIFACT_IDS: Final[tuple[str, ...]] = (
    REPORT_FIGURE_IDS[:10] + REPORT_TABLE_IDS[:4]
)
REPORT_COMPARISON_ARTIFACT_IDS: Final[tuple[str, ...]] = ("F11", "T05")
REPORT_VALIDATION_ARTIFACT_IDS: Final[tuple[str, ...]] = ("F12", "T06")

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_ALLOWED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "available",
        "unavailable_missing_comparison",
        "unavailable_missing_validation_evidence",
        "not_applicable_by_registered_design",
    }
)
_ALLOWED_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "synthetic_engineering_evidence",
        "server_pilot_evidence",
        "server_formal_baseline_evidence",
        "server_scientific_evidence",
    }
)
_ALLOWED_COMPONENT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "available",
        "unavailable_missing_provenance_subpanel",
        "not_applicable_by_registered_design",
    }
)

# role 是產品資料型別的唯一來源。prefix 既固定 release topology，suffix 與 media type
# 又共同防止把任意檔案冒充成另一種可繪製／可重繪產品；metadata sidecar 保留在
# data_sidecars/，因為它描述 table schema，而不是 figure caption。
_PRODUCT_ROLE_CONTRACTS: Final[dict[str, tuple[str, str, str]]] = {
    "figure_png": ("figures/", ".png", "image/png"),
    "figure_svg": ("figures/", ".svg", "image/svg+xml"),
    "figure_pdf": ("figures/", ".pdf", "application/pdf"),
    "table_parquet": ("tables/", ".parquet", "application/vnd.apache.parquet"),
    "table_csv": ("tables/", ".csv", "text/csv"),
    "caption_sidecar_json": ("caption_sidecars/", ".json", "application/json"),
    "data_sidecar_parquet": ("data_sidecars/", ".parquet", "application/vnd.apache.parquet"),
    "data_sidecar_npy": ("data_sidecars/", ".npy", "application/x-npy"),
    "metadata_sidecar_json": ("data_sidecars/", ".json", "application/json"),
}

_FIGURE_REQUIRED_ROLES: Final[frozenset[str]] = frozenset(
    {"figure_png", "figure_svg", "figure_pdf", "caption_sidecar_json"}
)
_FIGURE_ALLOWED_ROLES: Final[frozenset[str]] = _FIGURE_REQUIRED_ROLES | frozenset(
    {"data_sidecar_parquet", "data_sidecar_npy"}
)
_TABLE_REQUIRED_ROLES: Final[frozenset[str]] = frozenset(
    {"table_parquet", "table_csv", "metadata_sidecar_json"}
)
_TABLE_ALLOWED_ROLES: Final[frozenset[str]] = _TABLE_REQUIRED_ROLES | frozenset(
    {"data_sidecar_parquet", "data_sidecar_npy"}
)
_DATA_SIDECAR_ROLES: Final[frozenset[str]] = frozenset(
    {"data_sidecar_parquet", "data_sidecar_npy"}
)


def _require_exact_text(value: object, *, label: str) -> str:
    """要求原生、非空且沒有首尾空白的文字。

    report manifest 會把這些欄位直接交給 JSON／檔名與分組鍵；不在 record layer 自動
    trim 或把其他型別轉成文字，才能讓 caller 清楚知道來源資料哪一項沒有符合契約。
    """

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_token(value: object, *, label: str) -> str:
    """要求可安全放在 mapping key 或 reason code 的非空 ASCII token。"""

    text = _require_exact_text(value, label=label)
    if _TOKEN_RE.fullmatch(text) is None:
        raise ValueError(f"{label} 必須是安全非空 token")
    return text


def _require_slug(value: object, *, label: str) -> str:
    """要求 run／experiment case 使用不含路徑語意的固定 ASCII slug。"""

    text = _require_exact_text(value, label=label)
    if _SLUG_RE.fullmatch(text) is None:
        raise ValueError(f"{label} 必須是安全 ASCII slug")
    return text


def _require_sha256(value: object, *, label: str) -> str:
    """要求完整、全小寫、沒有前綴的 SHA-256 十六進位摘要。"""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是 64 碼小寫 SHA-256")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    """要求排除 bool 與 NumPy scalar 的非負原生 Python ``int``。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool")
    if value < 0:
        raise ValueError(f"{label} 不得為負數")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    """要求檔案大小是正的原生 bytes 計數。"""

    number = _require_nonnegative_int(value, label=label)
    if number == 0:
        raise ValueError(f"{label} 必須大於 0")
    return number


def _snapshot_tuple(value: object, *, label: str) -> tuple[object, ...]:
    """把 caller 的 iterable 複製成 tuple，拒絕字串與 mapping 這類歧義容器。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"{label} 必須是可 materialize 的非字串序列")
    try:
        return tuple(value)
    except Exception as error:
        raise ValueError(f"{label} 無法建立 defensive tuple") from error


def _snapshot_text_mapping(
    value: object,
    *,
    label: str,
    hash_values: bool = False,
    allowed_values: frozenset[str] | None = None,
) -> Mapping[str, str]:
    """建立 key/value 都是字串的 defensive ``MappingProxyType`` 快照。

    mapping 的 key 會成為 JSON 欄位、站點索引或 component 名稱，所以只接受安全 token；
    value 在一般欄位要求非空文字，input hash 則進一步要求完整小寫 SHA-256。逐項複製
    而不是直接包住 caller mapping，避免 caller 後續修改原 dict 影響已封存 registry。
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    try:
        items = tuple(value.items())
    except Exception as error:
        raise ValueError(f"{label} 無法讀取 mapping items") from error

    copied: dict[str, str] = {}
    for raw_key, raw_value in items:
        key = _require_token(raw_key, label=f"{label} key")
        if key in copied:
            raise ValueError(f"{label} 不得有重複 key")
        if hash_values:
            normalized_value = _require_sha256(raw_value, label=f"{label}[{key}]")
        else:
            normalized_value = _require_exact_text(raw_value, label=f"{label}[{key}]")
        if allowed_values is not None and normalized_value not in allowed_values:
            raise ValueError(f"{label}[{key}] 含不允許的狀態值")
        copied[key] = normalized_value
    return MappingProxyType(copied)


def _validate_relative_product_path(relative_path: object, *, role: str) -> str:
    """驗證產品是安全 POSIX relative path，且符合 role 的固定目錄與副檔名。"""

    path = _require_exact_text(relative_path, label="relative_path")
    if path.startswith("/") or "\\" in path:
        raise ValueError("relative_path 必須是安全 POSIX relative path")
    if "\x00" in path or any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise ValueError("relative_path 不得含控制字元")
    components = path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("relative_path 每層不得為空、. 或 ..")

    if role not in _PRODUCT_ROLE_CONTRACTS:
        raise ValueError("role 不在固定 report product role contract")
    prefix, suffix, _ = _PRODUCT_ROLE_CONTRACTS[role]
    if not path.startswith(prefix) or len(path) <= len(prefix) or not path.endswith(suffix):
        raise ValueError("relative_path 與 role 的 prefix/suffix 不一致")
    return path


def _validate_artifact_product_roles(
    *,
    artifact_kind: str,
    products: tuple[ReportProductRef, ...],
) -> None:
    """驗證 available figure/table 的最小產品閉包與不混 primary role 政策。"""

    roles = {product.role for product in products}
    if artifact_kind == "figure":
        if not _FIGURE_REQUIRED_ROLES.issubset(roles):
            raise ValueError("figure available 必須含 PNG/SVG/PDF、caption 與 data sidecar")
        if not roles.issubset(_FIGURE_ALLOWED_ROLES):
            raise ValueError("figure products 不得混入 table 或 metadata primary role")
        if not roles.intersection(_DATA_SIDECAR_ROLES):
            raise ValueError("figure available 至少需要一個 data sidecar")
        return

    if not _TABLE_REQUIRED_ROLES.issubset(roles):
        raise ValueError("table available 必須含 Parquet、CSV 與 metadata sidecar")
    if not roles.issubset(_TABLE_ALLOWED_ROLES):
        raise ValueError("table products 不得混入 figure 或 caption primary role")


@dataclass(frozen=True, slots=True)
class ReportProductRef:
    """一個 report product 的不可變相對路徑與 content contract。

    ``relative_path`` 只允許 release root 下的 POSIX relative path，並由 ``role`` 決定
    預期資料夾、副檔名與 media type。``size_bytes`` 是正的檔案大小（bytes），``sha256``
    是內容的完整小寫摘要；這個 record 不讀取檔案，後續 writer／validator 必須用 no-follow
    I/O 重新核對這兩項。它的 frozen slots 只保護欄位本身，所有欄位都已是 scalar，因此
    不存在可由 caller 改動的 nested mutable alias。
    """

    relative_path: str
    role: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """建立 record 前完成 path、role、media type、大小與 checksum 的 fail-fast 驗證。"""

        role = _require_exact_text(self.role, label="role")
        relative_path = _validate_relative_product_path(self.relative_path, role=role)
        _, _, expected_media_type = _PRODUCT_ROLE_CONTRACTS[role]
        media_type = _require_exact_text(self.media_type, label="media_type")
        if media_type != expected_media_type:
            raise ValueError("media_type 與 role contract 不一致")
        size_bytes = _require_positive_int(self.size_bytes, label="size_bytes")
        sha256 = _require_sha256(self.sha256, label="sha256")

        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "sha256", sha256)

    def to_dict(self) -> dict[str, object]:
        """回傳新的 JSON-ready plain dict，不暴露任何 immutable wrapper。"""

        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReportArtifactRecord:
    """F01–F12／T01–T06 單項報告成果的 immutable registry row。

    figure 必須同時登錄 PNG、SVG、PDF、caption JSON 與至少一個 data sidecar；table
    必須登錄 Parquet、CSV、metadata JSON，可另外登錄 data sidecar。``unavailable`` row
    不得夾帶產品，而要以固定狀態、reason code 與繁中理由說明缺少 comparison、validation
    evidence 或 registered design。available row 必須有非空 input SHA-256 mapping、raw
    sample count 與完整 denominator；這些欄位即使是 0 也代表已觀測的非負計數，不等於
    缺值。unavailable row 的 raw sample count、denominator name/count 則必須全部是
    ``None``，避免缺證列冒充有樣本統計。units 的數值描述欄位要在 available row 非空，
    不能把缺值或資料缺口偽裝成 0。

    ``input_sha256``、``units``、``crs_by_site`` 與 ``component_status`` 都會複製成
    ``MappingProxyType``，sequence 則複製成 tuple；因此 caller 後續修改原始 dict/list
    不會污染 registry。所有 evidence 都只描述工程證據類別，不能跨類別挪用 synthetic
    或 pilot／formal baseline 產品作為 SERVER scientific evidence。
    """

    artifact_id: str
    artifact_kind: str
    title_zh: str
    status: str
    evidence_class: str
    products: tuple[ReportProductRef, ...]
    input_sha256: Mapping[str, str]
    raw_sample_count: int | None
    denominator_name: str | None
    denominator_count: int | None
    units: Mapping[str, str]
    crs_by_site: Mapping[str, str]
    limitations: tuple[str, ...]
    unavailable_reason_code: str | None
    unavailable_reason_zh: str | None
    component_status: Mapping[str, str]

    def __post_init__(self) -> None:
        """驗證單項 artifact 的 exact ID/kind、產品閉包、計數與缺證政策。"""

        artifact_id = _require_exact_text(self.artifact_id, label="artifact_id")
        if artifact_id in REPORT_FIGURE_IDS:
            expected_kind = "figure"
        elif artifact_id in REPORT_TABLE_IDS:
            expected_kind = "table"
        else:
            raise ValueError("artifact_id 必須是固定 F01-F12 或 T01-T06")

        artifact_kind = _require_exact_text(self.artifact_kind, label="artifact_kind")
        if artifact_kind not in {"figure", "table"} or artifact_kind != expected_kind:
            raise ValueError("artifact_kind 必須與 artifact_id 的 F/T 類型一致")
        title_zh = _require_exact_text(self.title_zh, label="title_zh")
        status = _require_exact_text(self.status, label="status")
        if status not in _ALLOWED_STATUSES:
            raise ValueError("status 不在固定 report artifact status set")
        evidence_class = _require_exact_text(self.evidence_class, label="evidence_class")
        if evidence_class not in _ALLOWED_EVIDENCE_CLASSES:
            raise ValueError("evidence_class 不在固定 report evidence class set")

        raw_products = _snapshot_tuple(self.products, label="products")
        products: tuple[ReportProductRef, ...] = tuple()
        seen_roles: set[str] = set()
        seen_paths: set[str] = set()
        product_list: list[ReportProductRef] = []
        for index, product in enumerate(raw_products):
            if type(product) is not ReportProductRef:
                raise TypeError(f"products[{index}] 必須是 exact ReportProductRef")
            if product.role in seen_roles:
                raise ValueError("products 的 role 不得重複")
            if product.relative_path in seen_paths:
                raise ValueError("products 的 relative_path 不得重複")
            seen_roles.add(product.role)
            seen_paths.add(product.relative_path)
            product_list.append(product)
        products = tuple(product_list)

        input_sha256 = _snapshot_text_mapping(
            self.input_sha256,
            label="input_sha256",
            hash_values=True,
        )
        raw_sample_count = (
            None
            if self.raw_sample_count is None
            else _require_nonnegative_int(self.raw_sample_count, label="raw_sample_count")
        )
        denominator_name = (
            None
            if self.denominator_name is None
            else _require_exact_text(self.denominator_name, label="denominator_name")
        )
        denominator_count = (
            None
            if self.denominator_count is None
            else _require_nonnegative_int(self.denominator_count, label="denominator_count")
        )
        if (denominator_name is None) != (denominator_count is None):
            raise ValueError("denominator_name 與 denominator_count 必須成對出現或同時缺值")

        units = _snapshot_text_mapping(self.units, label="units")
        crs_by_site = _snapshot_text_mapping(self.crs_by_site, label="crs_by_site")
        raw_limitations = _snapshot_tuple(self.limitations, label="limitations")
        limitations: list[str] = []
        seen_limitations: set[str] = set()
        for index, limitation in enumerate(raw_limitations):
            text = _require_exact_text(limitation, label=f"limitations[{index}]")
            if text in seen_limitations:
                raise ValueError("limitations 必須 unique")
            seen_limitations.add(text)
            limitations.append(text)

        if self.unavailable_reason_code is None:
            unavailable_reason_code = None
        else:
            unavailable_reason_code = _require_token(
                self.unavailable_reason_code,
                label="unavailable_reason_code",
            )
        if self.unavailable_reason_zh is None:
            unavailable_reason_zh = None
        else:
            unavailable_reason_zh = _require_exact_text(
                self.unavailable_reason_zh,
                label="unavailable_reason_zh",
            )

        component_status = _snapshot_text_mapping(
            self.component_status,
            label="component_status",
            allowed_values=_ALLOWED_COMPONENT_STATUSES,
        )

        if status == "available":
            if not products:
                raise ValueError("available artifact 必須有 products")
            _validate_artifact_product_roles(artifact_kind=artifact_kind, products=products)
            if not input_sha256:
                raise ValueError("available artifact 的 input_sha256 不得為空")
            if raw_sample_count is None:
                raise ValueError("available artifact 必須提供 raw_sample_count")
            if denominator_name is None or denominator_count is None:
                raise ValueError("available artifact 必須提供完整 denominator")
            if not units:
                raise ValueError("available artifact 的 units 不得為空")
            if unavailable_reason_code is not None or unavailable_reason_zh is not None:
                raise ValueError("available artifact 不得有 unavailable reason")
        else:
            if products:
                raise ValueError("unavailable artifact 的 products 必須為空")
            if unavailable_reason_code is None or unavailable_reason_zh is None:
                raise ValueError("unavailable artifact 必須同時提供 reason code 與繁中理由")
            if (
                raw_sample_count is not None
                or denominator_name is not None
                or denominator_count is not None
            ):
                raise ValueError("unavailable artifact 的 sample/denominator 欄位必須全部缺值")

        # 這兩個 missing 狀態只能出現在其對應的 comparison／validation ID；registered
        # design 的 not-applicable 狀態保留給 registry policy 之後明示判斷，不在單列層
        # 猜測研究設計，避免 records layer 偷替 caller 決定是否適用。
        if status == "unavailable_missing_comparison" and artifact_id not in REPORT_COMPARISON_ARTIFACT_IDS:
            raise ValueError("unavailable_missing_comparison 只能用於 F11 或 T05")
        if (
            status == "unavailable_missing_validation_evidence"
            and artifact_id not in REPORT_VALIDATION_ARTIFACT_IDS
        ):
            raise ValueError("unavailable_missing_validation_evidence 只能用於 F12 或 T06")

        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "artifact_kind", artifact_kind)
        object.__setattr__(self, "title_zh", title_zh)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "raw_sample_count", raw_sample_count)
        object.__setattr__(self, "denominator_name", denominator_name)
        object.__setattr__(self, "denominator_count", denominator_count)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "crs_by_site", crs_by_site)
        object.__setattr__(self, "limitations", tuple(limitations))
        object.__setattr__(self, "unavailable_reason_code", unavailable_reason_code)
        object.__setattr__(self, "unavailable_reason_zh", unavailable_reason_zh)
        object.__setattr__(self, "component_status", component_status)

    def to_dict(self) -> dict[str, object]:
        """轉成固定欄位的 plain JSON-ready dict，所有 nested container 都重新建立。"""

        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "title_zh": self.title_zh,
            "status": self.status,
            "evidence_class": self.evidence_class,
            "products": [product.to_dict() for product in self.products],
            "input_sha256": dict(self.input_sha256),
            "raw_sample_count": self.raw_sample_count,
            "denominator_name": self.denominator_name,
            "denominator_count": self.denominator_count,
            "units": dict(self.units),
            "crs_by_site": dict(self.crs_by_site),
            "limitations": list(self.limitations),
            "unavailable_reason_code": self.unavailable_reason_code,
            "unavailable_reason_zh": self.unavailable_reason_zh,
            "component_status": dict(self.component_status),
        }


def _validate_optional_artifact_group(
    records_by_id: Mapping[str, ReportArtifactRecord],
    *,
    artifact_ids: tuple[str, str],
    allow_missing: bool,
    missing_status: str,
    label: str,
) -> None:
    """執行 comparison／validation 的明示 allow policy 與成對狀態 closure。

    allow flag 關閉時兩列都必須 available；開啟時也不允許一列 available、另一列
    missing，而只能兩列都 available 或兩列都為同一個對應 missing status。這樣下游
    不會把半份 comparison matrix 或半份 validation panel 誤當成完整證據。
    """

    statuses = tuple(records_by_id[artifact_id].status for artifact_id in artifact_ids)
    if not allow_missing:
        if statuses != ("available", "available"):
            raise ValueError(f"{label} 未允許缺證時兩項都必須 available")
        return

    allowed_statuses = {"available", missing_status}
    if any(status not in allowed_statuses for status in statuses):
        raise ValueError(f"{label} allow flag 只允許 available 或對應 unavailable 狀態")
    if statuses not in {
        ("available", "available"),
        (missing_status, missing_status),
    }:
        # comparison／validation 是成對產品；不能只讓其中一列缺證，否則 downstream
        # 會得到不完整的矩陣／圖表卻看見 allow flag 已開啟的錯誤安全感。
        raise ValueError(f"{label} allow flag 開啟時兩項狀態必須成對一致")


@dataclass(frozen=True, slots=True)
class ReportRegistry:
    """完整 F01–F12／T01–T06 report registry 的 immutable closure。

    registry 永遠要求 F01–F10 與 T01–T04 available；F11/T05 的 comparison 與 F12/T06
    的 validation evidence 是否可缺，必須由 caller 明示 allow flag，且缺少時 row 自身要
    保存固定 reason。``server_scientific_evidence`` 進一步禁止兩個 allow flag，並只能
    搭配 formal run；``server_pilot_evidence`` 只能搭配 pilot run；
    ``server_formal_baseline_evidence`` 只能搭配 formal run，且可依兩個 allow flag
    保存成對缺少的 comparison／validation rows。synthetic evidence 可用於 pilot/formal
    的本機工程 smoke，但不會因 schema 完整而取得正式科學證據資格。

    所有輸入 mapping、figure/table sequence 都會被複製；同一個 product relative path
    在整個 registry 內也只能出現一次，避免不同 F/T row 互相覆蓋或錯綁檔案。
    comparison F11/T05 與 validation F12/T06 的狀態也必須成對一致：allow flag 開啟時
    只能兩項都 available，或兩項都為各自對應的 missing 狀態。``to_dict`` 只輸出原生
    scalar、list 與 dict，適合後續 canonical JSON writer。此 foundation 不保存 Path、讀取
    檔案或驗證 checksum bytes；hash 的實際內容核對由後續 report release writer/validator 負責。
    """

    schema_version: str
    run_id: str
    run_kind: str
    experiment_case_id: str
    evidence_class: str
    aggregate_manifest_sha256: str
    source_run_plan_sha256: str
    source_run_progress_sha256: str
    config_hash: str
    checkpoint_input_binding_hash: str
    allow_missing_comparison: bool
    allow_missing_validation_evidence: bool
    figures: tuple[ReportArtifactRecord, ...]
    tables: tuple[ReportArtifactRecord, ...]

    def __post_init__(self) -> None:
        """驗證 schema、provenance、record closure、證據類別與缺證 policy。"""

        schema_version = _require_exact_text(self.schema_version, label="schema_version")
        if schema_version != REPORT_RELEASE_SCHEMA_VERSION:
            raise ValueError("schema_version 不受支援")
        run_id = _require_slug(self.run_id, label="run_id")
        run_kind = _require_exact_text(self.run_kind, label="run_kind")
        if run_kind not in {"pilot", "formal"}:
            raise ValueError("run_kind 只允許 pilot 或 formal")
        experiment_case_id = _require_slug(
            self.experiment_case_id,
            label="experiment_case_id",
        )
        evidence_class = _require_exact_text(self.evidence_class, label="evidence_class")
        if evidence_class not in _ALLOWED_EVIDENCE_CLASSES:
            raise ValueError("evidence_class 不在固定 report evidence class set")

        digests = {
            field_name: _require_sha256(getattr(self, field_name), label=field_name)
            for field_name in (
                "aggregate_manifest_sha256",
                "source_run_plan_sha256",
                "source_run_progress_sha256",
                "config_hash",
                "checkpoint_input_binding_hash",
            )
        }
        if type(self.allow_missing_comparison) is not bool:
            raise TypeError("allow_missing_comparison 必須是原生 bool")
        if type(self.allow_missing_validation_evidence) is not bool:
            raise TypeError("allow_missing_validation_evidence 必須是原生 bool")

        raw_figures = _snapshot_tuple(self.figures, label="figures")
        raw_tables = _snapshot_tuple(self.tables, label="tables")
        figures: list[ReportArtifactRecord] = []
        tables: list[ReportArtifactRecord] = []
        for index, record in enumerate(raw_figures):
            if type(record) is not ReportArtifactRecord:
                raise TypeError(f"figures[{index}] 必須是 exact ReportArtifactRecord")
            if record.artifact_kind != "figure":
                raise ValueError("figures 只能包含 figure artifact")
            figures.append(record)
        for index, record in enumerate(raw_tables):
            if type(record) is not ReportArtifactRecord:
                raise TypeError(f"tables[{index}] 必須是 exact ReportArtifactRecord")
            if record.artifact_kind != "table":
                raise ValueError("tables 只能包含 table artifact")
            tables.append(record)
        figures_tuple = tuple(figures)
        tables_tuple = tuple(tables)

        figure_ids = tuple(record.artifact_id for record in figures_tuple)
        table_ids = tuple(record.artifact_id for record in tables_tuple)
        if figure_ids != REPORT_FIGURE_IDS:
            raise ValueError("figures 必須精確包含 F01-F12 且維持固定順序")
        if table_ids != REPORT_TABLE_IDS:
            raise ValueError("tables 必須精確包含 T01-T06 且維持固定順序")
        all_records = figures_tuple + tables_tuple
        if any(record.evidence_class != evidence_class for record in all_records):
            raise ValueError("每個 artifact 的 evidence_class 必須等於 registry")
        records_by_id = {record.artifact_id: record for record in all_records}

        # 單列已檢查 role/path 不重複；這裡再跨 artifact 做全域 closure 檢查，因為同一個
        # figures/ 或 sidecar 檔案若被兩個 ID 共用，writer 無法判斷哪一列是真正產品，
        # validator 也可能把一份 bytes 誤當成兩項獨立證據。
        seen_product_paths: set[str] = set()
        for record in all_records:
            for product in record.products:
                if product.relative_path in seen_product_paths:
                    raise ValueError("registry 內 product relative_path 必須 globally unique")
                seen_product_paths.add(product.relative_path)

        for artifact_id in REPORT_CORE_ARTIFACT_IDS:
            if records_by_id[artifact_id].status != "available":
                raise ValueError("core F01-F10/T01-T04 必須全部 available")

        _validate_optional_artifact_group(
            records_by_id,
            artifact_ids=REPORT_COMPARISON_ARTIFACT_IDS,
            allow_missing=self.allow_missing_comparison,
            missing_status="unavailable_missing_comparison",
            label="comparison F11/T05",
        )
        _validate_optional_artifact_group(
            records_by_id,
            artifact_ids=REPORT_VALIDATION_ARTIFACT_IDS,
            allow_missing=self.allow_missing_validation_evidence,
            missing_status="unavailable_missing_validation_evidence",
            label="validation F12/T06",
        )

        # ReportArtifactRecord 已在單列限制兩種 missing status 的 ID；這裡再明示整個
        # closure 不接受未登錄設計的 not_applicable，因為本版 18 個 F/T ID 都已有固定
        # core、comparison 或 validation 責任，不能用自由狀態繞過上述 gate。
        if any(
            record.status == "not_applicable_by_registered_design" for record in all_records
        ):
            raise ValueError("本版固定 F/T closure 沒有可套用 not_applicable 的 artifact ID")

        if evidence_class == "server_scientific_evidence":
            if run_kind != "formal":
                raise ValueError("server_scientific_evidence 必須搭配 formal run")
            if self.allow_missing_comparison or self.allow_missing_validation_evidence:
                raise ValueError("server_scientific_evidence 不得允許 comparison/validation 缺證")
        elif evidence_class == "server_pilot_evidence" and run_kind != "pilot":
            raise ValueError("server_pilot_evidence 必須搭配 pilot run")
        elif evidence_class == "server_formal_baseline_evidence" and run_kind != "formal":
            raise ValueError("server_formal_baseline_evidence 必須搭配 formal run")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "run_kind", run_kind)
        object.__setattr__(self, "experiment_case_id", experiment_case_id)
        object.__setattr__(self, "evidence_class", evidence_class)
        for field_name, digest in digests.items():
            object.__setattr__(self, field_name, digest)
        object.__setattr__(self, "figures", figures_tuple)
        object.__setattr__(self, "tables", tables_tuple)

    def to_dict(self) -> dict[str, object]:
        """轉成固定欄位、可 canonical JSON 序列化的 plain dict。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "experiment_case_id": self.experiment_case_id,
            "evidence_class": self.evidence_class,
            "aggregate_manifest_sha256": self.aggregate_manifest_sha256,
            "source_run_plan_sha256": self.source_run_plan_sha256,
            "source_run_progress_sha256": self.source_run_progress_sha256,
            "config_hash": self.config_hash,
            "checkpoint_input_binding_hash": self.checkpoint_input_binding_hash,
            "allow_missing_comparison": self.allow_missing_comparison,
            "allow_missing_validation_evidence": self.allow_missing_validation_evidence,
            "figures": [record.to_dict() for record in self.figures],
            "tables": [record.to_dict() for record in self.tables],
        }
