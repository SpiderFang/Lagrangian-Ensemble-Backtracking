"""正式報告的可重現 renderer staging foundation。

本模組只負責把 caller 已經準備好的圖面 callback 或 Apache Arrow 表格寫入一個
明示、既有且空白的 staging root；它不讀取 trajectory、aggregate release 或任何
大型科學輸入，也不決定 final report 路徑，更不呼叫 report_release writer。
因此 F01–F12 與 T01–T06 的科學計算、分母、統計、版面內容仍由上游 pipeline 與
各自 renderer 負責，本模組只守住共同的輸出檔案、metadata、checksum 與 registry
record 邊界。

圖面 staging 固定產生 300 dpi PNG、SVG、PDF、caption JSON 與最小 Parquet data
sidecar；表格 staging 固定產生 Parquet、CSV 與欄位 metadata JSON。Parquet 是
保留 Apache Arrow 型別與 nullable 語意的權威交換產品，CSV 只作跨工具交換，不能
用空字串或零值猜測原始 null。所有 JSON 都以 UTF-8、排序 key、compact separators
及拒絕非有限數值的 canonical 規則產生；圖檔 metadata 不寫入 wall-clock 時間。

RenderedArtifact 的 staging path 只存在記憶體 view，不會進入
ReportArtifactRecord.to_dict() 或未來 registry JSON。成功 artifact 以前的輸出
可以留在 staging；任何新錯誤都會讓 renderer 永久 failed，且不回傳當次 partial
artifact。這些工程條件成立不代表真實 OCM schema 3、NWW3 schema 1 科學驗證、
條件式來源足跡或相對來源權重已完成。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet

from . import report_style
from .report_records import (
    REPORT_FIGURE_IDS,
    REPORT_TABLE_IDS,
    ReportArtifactRecord,
    ReportProductRef,
)
from .report_spec import ReportSpec

__all__ = [
    "FigureBuilder",
    "RenderedArtifact",
    "ReportStagingRenderer",
    "report_render_style_context",
]


# 這份角色表重述 report_records 的公開 role contract，讓 renderer 能在寫檔前建立
# 固定 relative path 與 media type；不匯入 records 的 private constant，避免 renderer
# 與 registry 實作細節形成不可見耦合。真正的 record constructor 仍會再做一次驗證。
_PRODUCT_ROLE_CONTRACTS: Final[dict[str, tuple[str, str, str]]] = {
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
    "metadata_sidecar_json": ("data_sidecars/", ".json", "application/json"),
}

_PLACEMENTS: Final[frozenset[str]] = frozenset({"main", "supplement"})
_RENDERER_NAME: Final[str] = "lagrangian_backtracking.report_render"
_CAPTION_SCHEMA_VERSION: Final[str] = "report_caption_metadata_v1"
_TABLE_METADATA_SCHEMA_VERSION: Final[str] = "report_table_metadata_v1"
_PARQUET_WRITER_POLICY: Final[str] = "pyarrow_parquet_uncompressed_dictionary_disabled_v1"
_CSV_WRITER_POLICY: Final[str] = "pyarrow_csv_exchange_only_v1"

FigureBuilder: TypeAlias = Callable[[], object]


def report_render_style_context(
    report_spec: ReportSpec,
) -> AbstractContextManager[object]:
    """轉接既有可重現 style context，保留測試與 renderer 的單一公開入口。

    真正的 style resolver 仍由 report_style 負責：它會驗證 task-specific
    MPLCONFIGDIR、固定 Agg backend、解析繁體中文字型並派生 SVG hashsalt。這個薄
    wrapper 不新增 fallback、不讀取資料，也讓 caller／測試能在不改動 render 流程
    的情況下注入已驗證的 style context。
    """

    return report_style.report_render_style_context(report_spec)


def _as_absolute_path(value: object, *, label: str) -> Path:
    """把 staging root 轉成明示的絕對 Path，拒絕隱式目前工作目錄。

    staging root 代表 caller 已核准的中間輸出位置；若允許相對路徑，renderer 會受
    process 啟動目錄影響，產生同一 spec 對應不同輸出位置的不可追溯差異。因此只接受
    path-like 值轉出的絕對路徑，且不在這裡建立目錄或解析 symlink。
    """

    try:
        raw_path = os.fspath(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是 path-like 絕對路徑") from error
    if not isinstance(raw_path, (str, bytes)):
        raise TypeError(f"{label} 必須是 path-like 絕對路徑")
    try:
        path = Path(raw_path)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是可解析的檔案系統路徑") from error
    if not path.is_absolute():
        raise ValueError(f"{label} 必須是絕對路徑")
    return path


def _require_ordinary_directory(path: Path, *, label: str) -> None:
    """以 lstat 要求現有最後節點是普通 non-symlink directory。"""

    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} 必須是既有普通 non-symlink directory") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} 必須是既有普通 non-symlink directory")


def _require_empty_directory(path: Path, *, label: str) -> None:
    """在建立固定 staging topology 前確認 root 沒有任何既有 children。"""

    _require_ordinary_directory(path, label=label)
    try:
        next(path.iterdir())
    except StopIteration:
        return
    except OSError as error:
        raise ValueError(f"{label} 無法檢查是否為空目錄") from error
    raise ValueError(f"{label} 必須是空目錄")


def _create_ordinary_directory(path: Path, *, label: str) -> None:
    """以 exist_ok=False 建立固定子目錄並立即驗證最後節點。"""

    try:
        path.mkdir(mode=0o700, exist_ok=False)
    except OSError as error:
        raise ValueError(f"{label} 無法建立固定 staging directory") from error
    _require_ordinary_directory(path, label=label)


def _snapshot_json_value(value: object, *, label: str) -> object:
    """建立可 canonical JSON 序列化的 defensive snapshot。

    caller metadata 是報告 provenance 的延伸欄位，不能接受 NumPy scalar、Path、bytes、
    set 或自訂物件等會依執行環境改變序列化結果的值。None 保持為真正 JSON null；
    不用空字串或 0 猜測缺失意義。mapping key 也逐項要求原生文字，避免 JSON encoder
    的隱式 key 轉型。
    """

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} 不可包含 NaN 或 Infinity")
        return value
    if type(value) in {list, tuple}:
        return [
            _snapshot_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        try:
            items = tuple(value.items())
        except Exception as error:
            raise TypeError(f"{label} mapping 無法建立 snapshot") from error
        for index, (raw_key, raw_value) in enumerate(items):
            if type(raw_key) is not str:
                raise TypeError(f"{label} key[{index}] 必須是原生 str")
            if raw_key in copied:
                raise ValueError(f"{label} 不可包含重複 key")
            copied[raw_key] = _snapshot_json_value(
                raw_value,
                label=f"{label}[{raw_key!r}]",
            )
        return copied
    raise TypeError(f"{label} 必須是 JSON-compatible scalar、sequence 或 mapping")


def _canonical_json_bytes(value: Mapping[str, object], *, label: str) -> bytes:
    """以固定 UTF-8 compact JSON 建立 deterministic bytes。"""

    snapshot = _snapshot_json_value(value, label=label)
    if not isinstance(snapshot, dict):
        raise TypeError(f"{label} 必須是 mapping")
    try:
        return json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(f"{label} 無法建立 canonical JSON") from error


def _arrow_metadata_snapshot(
    value: Mapping[bytes, bytes] | None,
    *,
    label: str,
) -> Mapping[str, str]:
    """將 Arrow schema bytes metadata 以 base64 保存，避免隱式解碼破壞 bytes。"""

    if value is None:
        return {}
    copied: dict[str, str] = {}
    for index, (key, raw_value) in enumerate(value.items()):
        if type(key) is not bytes or type(raw_value) is not bytes:
            raise TypeError(f"{label}[{index}] 必須是 bytes key/value")
        encoded_key = base64.b64encode(key).decode("ascii")
        if encoded_key in copied:
            raise ValueError(f"{label} 不可包含重複 key")
        copied[encoded_key] = base64.b64encode(raw_value).decode("ascii")
    return copied


def _arrow_schema_metadata(table: pa.Table) -> dict[str, object]:
    """建立表格欄位型別、nullability 與 schema metadata 的 JSON snapshot。"""

    columns: list[dict[str, object]] = []
    for index, field in enumerate(table.schema):
        if type(field.name) is not str:
            raise TypeError(f"table schema field[{index}].name 必須是 str")
        columns.append(
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(field.nullable),
                "metadata": _arrow_metadata_snapshot(
                    field.metadata,
                    label=f"table schema field[{index}].metadata",
                ),
            }
        )
    return {
        "num_rows": int(table.num_rows),
        "num_columns": int(table.num_columns),
        "columns": columns,
        "schema_metadata": _arrow_metadata_snapshot(
            table.schema.metadata,
            label="table schema.metadata",
        ),
    }


def _require_arrow_table(value: object, *, label: str) -> pa.Table:
    """要求 exact pyarrow.Table，不接受 pandas 或 duck-typed table。"""

    if type(value) is not pa.Table:
        raise TypeError(f"{label} 必須是 exact PyArrow Table")
    return value


def _require_artifact_id(value: object, *, kind: str) -> str:
    """要求 artifact ID 精確落在固定 F01–F12 或 T01–T06 集合。"""

    if type(value) is not str:
        raise TypeError("artifact_id 必須是原生 str")
    allowed = REPORT_FIGURE_IDS if kind == "figure" else REPORT_TABLE_IDS
    if value not in allowed:
        raise ValueError(f"{kind} artifact_id 必須精確落在固定 F/T ID 集合")
    return value


def _require_placement(value: object) -> str:
    """要求圖面只能明示放入 main 或 supplement topology。"""

    if type(value) is not str or value not in _PLACEMENTS:
        raise ValueError("figure placement 必須精確是 main 或 supplement")
    return value


def _require_callable(value: object, *, label: str) -> FigureBuilder:
    """要求圖面 callback 可被呼叫；callback 不由 renderer 隱式重試。"""

    if not callable(value):
        raise TypeError(f"{label} 必須是 callable")
    return value  # type: ignore[return-value]


def _require_available_status(value: object) -> str:
    """renderer 只建立 available row，拒絕用 placeholder 冒充缺證產品。"""

    if type(value) is not str or value != "available":
        raise ValueError(
            "renderer 只建立 available artifact；缺 comparison/validation 請由 "
            "pipeline/registry 建立 unavailable row"
        )
    return value


def _product_relative_path(path: Path, *, relative_path: str, role: str) -> None:
    """確認固定 output path 仍對應 role 的公開 prefix/suffix contract。"""

    prefix, suffix, _ = _PRODUCT_ROLE_CONTRACTS[role]
    if (
        relative_path != relative_path.strip()
        or relative_path.startswith("/")
        or "\\" in relative_path
        or any(component in {"", ".", ".."} for component in relative_path.split("/"))
        or not relative_path.startswith(prefix)
    ):
        raise ValueError("renderer relative path 與 product role 不一致")
    if not relative_path.endswith(suffix):
        raise ValueError("renderer relative path suffix 與 product role 不一致")
    if not path.is_absolute() or path.name in {".", ".."}:
        raise ValueError("renderer output path 必須是 staging 絕對檔案")


def _lstat_absent(path: Path, *, label: str) -> None:
    """拒絕既有檔案、目錄、symlink 與 broken symlink，保證不覆寫。"""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(f"{label} 無法檢查既有 target") from error
    raise FileExistsError(f"{label} 已存在，renderer 不覆寫")


def _read_regular_file_bytes(path: Path, *, label: str) -> tuple[bytes, tuple[int, int]]:
    """以 no-follow I/O 讀取 ordinary file並回傳 bytes 與 inode identity。

    lstat 先拒絕 symlink，O_NOFOLLOW 防止 open 時跟隨替換連結，fstat 再確認 descriptor
    仍指向同一個 ordinary regular file。此邊界讓 RenderedArtifact 不會只相信 renderer
    內部剛寫入的 bytes，而會實際核對 staging filesystem。
    """

    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} 必須是 ordinary non-symlink file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(opened.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"{label} file identity 在讀取期間改變")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), (opened.st_dev, opened.st_ino)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"{label} ordinary file 無法讀取") from error
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _write_exclusive_bytes(path: Path, data: bytes, *, label: str) -> tuple[int, int]:
    """以 exclusive regular-file create 寫入完整 bytes，不跟隨既有 symlink。"""

    if type(data) is not bytes:
        raise TypeError(f"{label} bytes 必須是原生 bytes")
    parent = path.parent
    _require_ordinary_directory(parent, label=f"{label} parent")
    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("zero-byte write")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} 不是 ordinary regular file")
        return (metadata.st_dev, metadata.st_ino)
    except FileExistsError:
        raise
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} 無法以 exclusive mode 寫入") from error
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _cleanup_created_files(created: Sequence[tuple[Path, tuple[int, int]]]) -> None:
    """只清除本次 renderer 建立且 inode 未變動的 ordinary file。"""

    for path, identity in reversed(tuple(created)):
        try:
            metadata = os.lstat(path)
        except OSError:
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == identity
        ):
            with suppress(OSError):
                os.unlink(path)


def _sha256_bytes(data: bytes) -> str:
    """由實際 product bytes 計算完整小寫 SHA-256。"""

    return hashlib.sha256(data).hexdigest()


def _product_ref(
    path: Path,
    *,
    relative_path: str,
    role: str,
) -> ReportProductRef:
    """讀取已寫入檔案並由實際 size/SHA 建立 exact ReportProductRef。"""

    _product_relative_path(path, relative_path=relative_path, role=role)
    raw_bytes, _ = _read_regular_file_bytes(path, label=f"product {relative_path}")
    _, _, media_type = _PRODUCT_ROLE_CONTRACTS[role]
    return ReportProductRef(
        relative_path=relative_path,
        role=role,
        media_type=media_type,
        size_bytes=len(raw_bytes),
        sha256=_sha256_bytes(raw_bytes),
    )


def _report_spec_provenance(report_spec: ReportSpec) -> dict[str, object]:
    """建立不含 path 或時間的完整 report spec provenance snapshot。"""

    return {
        "report_spec": report_spec.to_dict(),
        "report_spec_schema_version": report_spec.schema_version,
        "report_spec_source_sha256": report_spec.source_sha256,
        "report_spec_canonical_sha256": report_spec.canonical_sha256,
    }


def _style_provenance(style: object) -> dict[str, object]:
    """抽取 style/font provenance，不保存 MPLCONFIGDIR 或字型絕對路徑。"""

    try:
        font_selection = style.font_selection
        font = {
            "family": font_selection.family,
            "filename": font_selection.filename,
            "file_sha256": font_selection.file_sha256,
            "required_glyph_count": font_selection.required_glyph_count,
        }
        provenance = {
            "renderer": _RENDERER_NAME,
            "renderer_style_version": style.renderer_style_version,
            "language": style.language,
            "raster_dpi": style.raster_dpi,
            "svg_hashsalt": style.svg_hashsalt,
            "font_selection": font,
        }
    except AttributeError as error:
        raise TypeError("report_render_style_context 必須提供完整 style/font provenance") from error
    snapshot = _snapshot_json_value(provenance, label="renderer style provenance")
    if not isinstance(snapshot, dict):
        raise TypeError("renderer style provenance 必須是 mapping")
    return snapshot


def _table_renderer_provenance(report_spec: ReportSpec) -> dict[str, object]:
    """建立不需要載入 Matplotlib 的表格 renderer provenance。"""

    return {
        "renderer": _RENDERER_NAME,
        "renderer_style_version": report_spec.renderer_style_version,
        "language": report_spec.language,
        "raster_dpi": report_spec.raster_dpi,
        "table_serialization_policy": _PARQUET_WRITER_POLICY,
        "csv_serialization_policy": _CSV_WRITER_POLICY,
    }


def _stable_figure_file_metadata(
    *,
    artifact_id: str,
    title_zh: str,
    file_format: str,
) -> dict[str, object]:
    """建立不含 wall-clock 欄位的 Matplotlib metadata。

    PDF backend 對 CreationDate/ModDate 使用 None 明示禁止日期；SVG backend 對 Date
    也採同一策略。PNG 只寫固定作者／標題，不傳入作業系統時間、process id、temporary
    path 或目前工作目錄。
    """

    if file_format == "svg":
        # SVG writer 的 metadata whitelist 不接受 PDF 的 Subject；Date=None 會
        # 明示省略 backend 預設的現在時間。
        return {
            "Creator": _RENDERER_NAME,
            "Title": f"{artifact_id} {title_zh}",
            "Date": None,
        }
    if file_format == "pdf":
        return {
            "Creator": _RENDERER_NAME,
            "Title": f"{artifact_id} {title_zh}",
            "Subject": "reproducible report artifact",
            "CreationDate": None,
            "ModDate": None,
        }
    # PNG 使用文字 chunk 保存固定 provenance；不要傳入 Date、CreationDate 或
    # 任何由 backend 依當前時間產生的欄位。
    return {
        "Software": _RENDERER_NAME,
        "Title": f"{artifact_id} {title_zh}",
        "Description": "reproducible report artifact",
    }


def _assert_no_wall_clock_metadata(raw_bytes: bytes, *, file_format: str) -> None:
    """檢查三種圖檔 bytes 沒有 backend 偷加的日期 metadata。"""

    lowered = raw_bytes.lower()
    if file_format == "svg":
        if any(token in lowered for token in (b"<dc:date", b"<date", b"creationdate", b"moddate")):
            raise ValueError("SVG 不得含 wall-clock date metadata")
    elif file_format == "pdf":
        if b"/creationdate" in lowered or b"/moddate" in lowered:
            raise ValueError("PDF 不得含 wall-clock date metadata")
    elif file_format == "png" and any(
        token in lowered for token in (b"date", b"creationdate", b"moddate")
    ):
        raise ValueError("PNG 不得含 wall-clock date metadata")


def _write_parquet_bytes(table: pa.Table) -> bytes:
    """以固定 writer options 序列化 Arrow Table，保留型別與 nullable schema。"""

    sink = io.BytesIO()
    try:
        pa_parquet.write_table(
            table,
            sink,
            compression="NONE",
            use_dictionary=False,
            write_statistics=False,
        )
    except Exception as error:
        raise ValueError("PyArrow Table 無法序列化為 Parquet") from error
    return sink.getvalue()


def _write_csv_bytes(table: pa.Table) -> bytes:
    """建立只供交換的 Arrow CSV bytes，不排序欄位或列。"""

    sink = io.BytesIO()
    try:
        pa_csv.write_csv(table, sink)
    except Exception as error:
        raise ValueError("PyArrow Table 無法序列化為 CSV") from error
    return sink.getvalue()


def _record(
    *,
    artifact_id: str,
    artifact_kind: str,
    title_zh: str,
    evidence_class: str,
    products: tuple[ReportProductRef, ...],
    input_sha256: Mapping[str, str],
    raw_sample_count: int,
    denominator_name: str,
    denominator_count: int,
    units: Mapping[str, str],
    crs_by_site: Mapping[str, str],
    limitations: Sequence[str],
    component_status: Mapping[str, str],
    status: str,
) -> ReportArtifactRecord:
    """以 caller 明示的完整 metadata 建立 available registry row。

    renderer 不自行重算 numerator、denominator 或 scientific status；這裡只把 caller
    提供的計數、單位、CRS（座標參考系統）、限制與 component status 交給既有
    ReportArtifactRecord 做 exact validation。缺證項目不在這裡建立 unavailable row。
    """

    return ReportArtifactRecord(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        title_zh=title_zh,
        status=status,
        evidence_class=evidence_class,
        products=products,
        input_sha256=input_sha256,
        raw_sample_count=raw_sample_count,
        denominator_name=denominator_name,
        denominator_count=denominator_count,
        units=units,
        crs_by_site=crs_by_site,
        limitations=limitations,
        unavailable_reason_code=None,
        unavailable_reason_zh=None,
        component_status=component_status,
    )


@dataclass(frozen=True, slots=True)
class RenderedArtifact:
    """一項已完成 bytes 驗證的 immutable report artifact staging view。

    record 必須是 exact ReportArtifactRecord；staging_paths 是 relative_path 到 Path
    的記憶體 mapping，key 必須與 record products 的 path 集合完全相等。constructor
    會重新使用 no-follow I/O 確認每個 target 是 ordinary non-symlink file，並以實際
    bytes 比對 ReportProductRef.size_bytes 與 SHA-256。path 不會被寫入 record 或
    registry JSON；它只讓同一個 process 後續把 staging view 交給 release writer。
    """

    record: ReportArtifactRecord
    staging_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        """防禦性複製 mapping 並封閉 record product/path/checksum 對應。"""

        if type(self.record) is not ReportArtifactRecord:
            raise TypeError("record 必須是 exact ReportArtifactRecord")
        if not isinstance(self.staging_paths, Mapping):
            raise TypeError("staging_paths 必須是 relative_path 到 Path 的 mapping")
        copied: dict[str, Path] = {}
        try:
            items = tuple(self.staging_paths.items())
        except Exception as error:
            raise TypeError("staging_paths mapping 無法讀取") from error
        for index, (relative_path, raw_path) in enumerate(items):
            if type(relative_path) is not str:
                raise TypeError(f"staging_paths key[{index}] 必須是原生 str")
            if not isinstance(raw_path, Path):
                raise TypeError(f"staging_paths[{relative_path!r}] 必須是 pathlib.Path")
            if relative_path in copied:
                raise ValueError("staging_paths 不可包含重複 relative_path")
            copied[relative_path] = raw_path

        products_by_path = {product.relative_path: product for product in self.record.products}
        if set(copied) != set(products_by_path):
            raise ValueError("staging_paths keys 必須精確等於 record.products paths")
        for relative_path, product in products_by_path.items():
            raw_bytes, _ = _read_regular_file_bytes(
                copied[relative_path],
                label=f"staging product {relative_path}",
            )
            if len(raw_bytes) != product.size_bytes:
                raise ValueError(f"staging product {relative_path} size_bytes 不符")
            if _sha256_bytes(raw_bytes) != product.sha256:
                raise ValueError(f"staging product {relative_path} sha256 不符")
        object.__setattr__(self, "staging_paths", MappingProxyType(dict(copied)))

    @property
    def paths(self) -> Mapping[str, Path]:
        """回傳 staging path mapping 的相容唯讀別名。"""

        return self.staging_paths

    @property
    def products(self) -> tuple[ReportProductRef, ...]:
        """回傳 exact record 的 products tuple，不暴露 staging root。"""

        return self.record.products


class ReportStagingRenderer:
    """建立共同 F/T artifact staging topology 的 fail-closed renderer foundation。

    Args:
        report_spec: exact ReportSpec。它提供 run、renderer style、語言與 300 dpi
            的固定 provenance；本類別不從其他資料推測或改寫規格。
        staging_root: caller 明示、已存在、絕對、ordinary non-symlink 且空白的目錄。
            constructor 只在此 root 內建立固定 staging 子目錄，不決定 final report path。

    render_figure 的 callback 只負責建立 Matplotlib Figure；renderer 會在
    report_render_style_context(report_spec) 內只呼叫一次 callback，再以固定 metadata
    輸出三種圖檔。render_table 只處理 exact PyArrow Table，維持 caller 的欄位與
    row order，不執行分母、統計或排序。兩者都建立 available ReportArtifactRecord；
    F11/T05、F12/T06 缺證時應由 pipeline／registry 建立 unavailable row，本類別不畫
    placeholder，也不把 zero-row 假表當成缺證產品。

    任一 public render call 的錯誤都會把 renderer 永久標記 failed；先前成功 artifact
    可留在 staging，但 failed instance 不可再使用。exclusive create 與
    RenderedArtifact 的 filesystem recheck 共同防止既有 target、symlink 或 checksum
    不一致被回報成成功。
    """

    _OPEN: Final[str] = "open"
    _FAILED: Final[str] = "failed"

    __slots__ = (
        "_report_spec",
        "_staging_root",
        "_state",
        "_records_by_id",
        "_paths_by_relative",
    )

    def __init__(self, report_spec: ReportSpec, staging_root: str | Path) -> None:
        """驗證 exact spec 與空白 root，並建立固定共同 staging directories。"""

        if type(report_spec) is not ReportSpec:
            raise TypeError("report_spec 必須是 exact ReportSpec")
        root = _as_absolute_path(staging_root, label="staging_root")
        _require_empty_directory(root, label="staging_root")

        # 固定 topology 讓後續 registry path 不依賴 caller 自訂資料夾；所有 mkdir 都
        # 使用 exist_ok=False，若 root 在檢查後被其他程序改動便直接失敗，不覆寫外部檔案。
        figures = root / "figures"
        _create_ordinary_directory(figures, label="staging_root/figures")
        _create_ordinary_directory(
            figures / "main",
            label="staging_root/figures/main",
        )
        _create_ordinary_directory(
            figures / "supplement",
            label="staging_root/figures/supplement",
        )
        for directory_name in ("tables", "caption_sidecars", "data_sidecars"):
            _create_ordinary_directory(
                root / directory_name,
                label=f"staging_root/{directory_name}",
            )

        self._report_spec = report_spec
        self._staging_root = root
        self._state = self._OPEN
        self._records_by_id: dict[str, ReportArtifactRecord] = {}
        self._paths_by_relative: dict[str, Path] = {}

    @property
    def report_spec(self) -> ReportSpec:
        """回傳已綁定的 exact ReportSpec。"""

        return self._report_spec

    @property
    def staging_root(self) -> Path:
        """回傳 staging root 的記憶體路徑，不會寫入任何 registry record。"""

        return self._staging_root

    @property
    def state(self) -> str:
        """回傳 renderer 的 open 或永久 failed 狀態。"""

        return self._state

    @property
    def artifact_records(self) -> Mapping[str, ReportArtifactRecord]:
        """回傳目前成功 artifact records 的 defensive read-only mapping。"""

        return MappingProxyType(dict(self._records_by_id))

    def _ensure_open(self) -> None:
        """拒絕 failed renderer，避免 caller 忽略先前 partial/metadata 錯誤。"""

        if self._state != self._OPEN:
            raise ValueError("ReportStagingRenderer 已永久關閉")

    def _mark_failed(self) -> None:
        """將任一 render 失敗固定為不可恢復狀態。"""

        self._state = self._FAILED

    def _preflight_artifact(
        self,
        *,
        artifact_id: str,
        relative_roles: Sequence[tuple[str, str]],
    ) -> dict[str, Path]:
        """檢查 ID/path unique、targets absent，並建立固定 staging paths。"""

        if artifact_id in self._records_by_id:
            raise ValueError("artifact_id 不可重複 render")
        resolved: dict[str, Path] = {}
        for relative_path, role in relative_roles:
            if relative_path in resolved or relative_path in self._paths_by_relative:
                raise ValueError("artifact relative_path 不可重複 render")
            if role not in _PRODUCT_ROLE_CONTRACTS:
                raise ValueError("未知 report product role")
            path = self._staging_root / relative_path
            _product_relative_path(
                path,
                relative_path=relative_path,
                role=role,
            )
            _lstat_absent(path, label=f"product {relative_path}")
            resolved[relative_path] = path
        return resolved

    def _commit(
        self,
        *,
        artifact_id: str,
        artifact_kind: str,
        title_zh: str,
        evidence_class: str,
        relative_roles: Sequence[tuple[str, str]],
        product_bytes: Mapping[str, bytes],
        input_sha256: Mapping[str, str],
        raw_sample_count: int,
        denominator_name: str,
        denominator_count: int,
        units: Mapping[str, str],
        crs_by_site: Mapping[str, str],
        limitations: Sequence[str],
        component_status: Mapping[str, str],
        status: str,
    ) -> RenderedArtifact:
        """完成 exclusive writes、實際 checksum、record 與 immutable staging view。

        所有 serial bytes 在進入這個函式前已準備完畢；因此若 metadata、target 或
        filesystem 任一步失敗，當次 created files 會依 inode 只清理自己建立的檔案，
        不影響先前 artifact。這不是跨 artifact transaction；parent renderer 仍在
        public caller 中被永久標記 failed。
        """

        paths = self._preflight_artifact(
            artifact_id=artifact_id,
            relative_roles=relative_roles,
        )
        created: list[tuple[Path, tuple[int, int]]] = []
        try:
            for relative_path, _role in relative_roles:
                if relative_path not in product_bytes:
                    raise ValueError(f"product bytes 缺少 {relative_path}")
                identity = _write_exclusive_bytes(
                    paths[relative_path],
                    product_bytes[relative_path],
                    label=f"product {relative_path}",
                )
                created.append((paths[relative_path], identity))

            products = tuple(
                _product_ref(
                    paths[relative_path],
                    relative_path=relative_path,
                    role=role,
                )
                for relative_path, role in relative_roles
            )
            record = _record(
                artifact_id=artifact_id,
                artifact_kind=artifact_kind,
                title_zh=title_zh,
                evidence_class=evidence_class,
                products=products,
                input_sha256=input_sha256,
                raw_sample_count=raw_sample_count,
                denominator_name=denominator_name,
                denominator_count=denominator_count,
                units=units,
                crs_by_site=crs_by_site,
                limitations=limitations,
                component_status=component_status,
                status=status,
            )
            staged = RenderedArtifact(
                record=record,
                staging_paths={
                    relative_path: paths[relative_path]
                    for relative_path, _role in relative_roles
                },
            )
        except Exception:
            _cleanup_created_files(created)
            raise

        self._records_by_id[artifact_id] = record
        self._paths_by_relative.update(staged.staging_paths)
        return staged

    def render_figure(
        self,
        artifact_id: str,
        placement: str,
        figure_builder: FigureBuilder,
        data_table: pa.Table,
        *,
        title_zh: str,
        evidence_class: str,
        input_sha256: Mapping[str, str],
        raw_sample_count: int,
        denominator_name: str,
        denominator_count: int,
        units: Mapping[str, str],
        crs_by_site: Mapping[str, str],
        limitations: Sequence[str],
        component_status: Mapping[str, str],
        caption_metadata: Mapping[str, object] | None = None,
        status: str = "available",
    ) -> RenderedArtifact:
        """建立一項 F01–F12 圖面 artifact，callback 只呼叫一次。

        figure_builder 不接收 trajectory／aggregate，也不應在 callback 內偷偷重算
        分母；它只需建立並回傳 Matplotlib Figure。data_table 是 exact PyArrow Table，
        作為最小可重繪 Parquet sidecar；caption metadata 則以 nested caller_metadata
        保存 caller 明示欄位。所有 record 欄位由 caller 明示，renderer 不自行推測
        scientific numerator、denominator、units 或 CRS。

        F11/F12 若缺 comparison／validation evidence，caller 不應呼叫本方法；傳入
        非 available status 會直接拒絕，renderer 不產生 placeholder 或 unavailable
        record。輸出相對路徑固定為 figures/<placement>/<Fxx>.<ext>、
        caption_sidecars/<Fxx>.json 與 data_sidecars/<Fxx>.parquet。
        """

        self._ensure_open()
        try:
            artifact_id = _require_artifact_id(artifact_id, kind="figure")
            placement = _require_placement(placement)
            builder = _require_callable(figure_builder, label="figure_builder")
            table = _require_arrow_table(data_table, label="data_table")
            status = _require_available_status(status)
            if artifact_id in {"F11", "F12"} and table.num_rows == 0:
                raise ValueError(
                    "comparison/validation artifact 不得以 zero-row placeholder render"
                )
            caller_metadata = (
                {}
                if caption_metadata is None
                else _snapshot_json_value(caption_metadata, label="caption_metadata")
            )
            if not isinstance(caller_metadata, dict):
                raise TypeError("caption_metadata 必須是 mapping")

            relative_roles = (
                (f"figures/{placement}/{artifact_id}.png", "figure_png"),
                (f"figures/{placement}/{artifact_id}.svg", "figure_svg"),
                (f"figures/{placement}/{artifact_id}.pdf", "figure_pdf"),
                (f"caption_sidecars/{artifact_id}.json", "caption_sidecar_json"),
                (f"data_sidecars/{artifact_id}.parquet", "data_sidecar_parquet"),
            )
            self._preflight_artifact(
                artifact_id=artifact_id,
                relative_roles=relative_roles,
            )

            data_bytes = _write_parquet_bytes(table)
            schema_metadata = _arrow_schema_metadata(table)
            figure_bytes: dict[str, bytes] = {}
            figure = None
            try:
                # report_render_style_context 會驗證 MPLCONFIGDIR、Agg 與 CJK 字型；
                # callback 在 with block 內建立 Figure，確保 rcParams、font 與 SVG
                # hashsalt 對三種輸出一致；callback 絕不被 renderer 重試。
                with report_render_style_context(self._report_spec) as style:
                    figure = builder()
                    from matplotlib.figure import Figure

                    if not isinstance(figure, Figure):
                        raise TypeError("figure_builder 必須回傳 Matplotlib Figure")
                    for file_format in ("png", "svg", "pdf"):
                        sink = io.BytesIO()
                        figure.savefig(
                            sink,
                            format=file_format,
                            dpi=style.raster_dpi,
                            metadata=_stable_figure_file_metadata(
                                artifact_id=artifact_id,
                                title_zh=title_zh,
                                file_format=file_format,
                            ),
                        )
                        raw_bytes = sink.getvalue()
                        if not raw_bytes:
                            raise ValueError(f"{file_format} figure bytes 不可為空")
                        _assert_no_wall_clock_metadata(
                            raw_bytes,
                            file_format=file_format,
                        )
                        figure_bytes[
                            f"figures/{placement}/{artifact_id}.{file_format}"
                        ] = raw_bytes
                    caption_payload: dict[str, object] = {
                        "schema_version": _CAPTION_SCHEMA_VERSION,
                        "artifact_id": artifact_id,
                        "artifact_kind": "figure",
                        "title_zh": title_zh,
                        "status": status,
                        "evidence_class": evidence_class,
                        **_report_spec_provenance(self._report_spec),
                        "renderer_provenance": _style_provenance(style),
                        "input_sha256": input_sha256,
                        "raw_sample_count": raw_sample_count,
                        "denominator": {
                            "name": denominator_name,
                            "count": denominator_count,
                        },
                        "units": units,
                        "crs_by_site": crs_by_site,
                        "limitations": limitations,
                        "component_status": component_status,
                        "data_sidecar_schema": schema_metadata,
                        "caller_metadata": caller_metadata,
                    }
                    caption_bytes = _canonical_json_bytes(
                        caption_payload,
                        label=f"{artifact_id} caption metadata",
                    )
            finally:
                if figure is not None:
                    try:
                        import matplotlib.pyplot as pyplot

                        pyplot.close(figure)
                    except Exception:
                        # close 失敗不應讓已序列化的 bytes 變成另一種錯誤；真正的
                        # output validation 仍由 _commit 與 RenderedArtifact 完成。
                        pass

            product_bytes = {
                **figure_bytes,
                f"caption_sidecars/{artifact_id}.json": caption_bytes,
                f"data_sidecars/{artifact_id}.parquet": data_bytes,
            }
            return self._commit(
                artifact_id=artifact_id,
                artifact_kind="figure",
                title_zh=title_zh,
                evidence_class=evidence_class,
                relative_roles=relative_roles,
                product_bytes=product_bytes,
                input_sha256=input_sha256,
                raw_sample_count=raw_sample_count,
                denominator_name=denominator_name,
                denominator_count=denominator_count,
                units=units,
                crs_by_site=crs_by_site,
                limitations=limitations,
                component_status=component_status,
                status=status,
            )
        except Exception:
            self._mark_failed()
            raise

    def render_table(
        self,
        artifact_id: str,
        table: pa.Table,
        *,
        title_zh: str,
        evidence_class: str,
        input_sha256: Mapping[str, str],
        raw_sample_count: int,
        denominator_name: str,
        denominator_count: int,
        units: Mapping[str, str],
        crs_by_site: Mapping[str, str],
        limitations: Sequence[str],
        component_status: Mapping[str, str],
        metadata: Mapping[str, object] | None = None,
        status: str = "available",
    ) -> RenderedArtifact:
        """建立一項 T01–T06 table artifact，保留 caller table schema、null 與 row order。

        Parquet 使用固定 writer policy 保存 exact Arrow schema；CSV 由 Arrow writer
        建立且只作交換，不能取代 Parquet 的 nullable/type semantics。metadata JSON
        會保存每個欄位的 Arrow type、nullability、units、ReportSpec、renderer
        provenance 與 caller 明示 metadata；metadata 不會用空字串或 0 取代 None。
        renderer 不排序列，也不重新計算表格統計或 denominator。
        """

        self._ensure_open()
        try:
            artifact_id = _require_artifact_id(artifact_id, kind="table")
            table = _require_arrow_table(table, label="table")
            status = _require_available_status(status)
            if artifact_id in {"T05", "T06"} and table.num_rows == 0:
                raise ValueError(
                    "comparison/validation artifact 不得以 zero-row placeholder render"
                )
            caller_metadata = (
                {}
                if metadata is None
                else _snapshot_json_value(metadata, label="table metadata")
            )
            if not isinstance(caller_metadata, dict):
                raise TypeError("metadata 必須是 mapping")

            relative_roles = (
                (f"tables/{artifact_id}.parquet", "table_parquet"),
                (f"tables/{artifact_id}.csv", "table_csv"),
                (f"data_sidecars/{artifact_id}.json", "metadata_sidecar_json"),
            )
            parquet_bytes = _write_parquet_bytes(table)
            csv_bytes = _write_csv_bytes(table)
            metadata_payload: dict[str, object] = {
                "schema_version": _TABLE_METADATA_SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "artifact_kind": "table",
                "title_zh": title_zh,
                "status": status,
                "evidence_class": evidence_class,
                **_report_spec_provenance(self._report_spec),
                "renderer_provenance": _table_renderer_provenance(self._report_spec),
                "input_sha256": input_sha256,
                "raw_sample_count": raw_sample_count,
                "denominator": {
                    "name": denominator_name,
                    "count": denominator_count,
                },
                "units": units,
                "crs_by_site": crs_by_site,
                "limitations": limitations,
                "component_status": component_status,
                "table_schema": _arrow_schema_metadata(table),
                "null_policy": {
                    "parquet": "authoritative_preserve_arrow_nullability",
                    "csv": "exchange_only",
                },
                "caller_metadata": caller_metadata,
            }
            metadata_bytes = _canonical_json_bytes(
                metadata_payload,
                label=f"{artifact_id} table metadata",
            )
            return self._commit(
                artifact_id=artifact_id,
                artifact_kind="table",
                title_zh=title_zh,
                evidence_class=evidence_class,
                relative_roles=relative_roles,
                product_bytes={
                    f"tables/{artifact_id}.parquet": parquet_bytes,
                    f"tables/{artifact_id}.csv": csv_bytes,
                    f"data_sidecars/{artifact_id}.json": metadata_bytes,
                },
                input_sha256=input_sha256,
                raw_sample_count=raw_sample_count,
                denominator_name=denominator_name,
                denominator_count=denominator_count,
                units=units,
                crs_by_site=crs_by_site,
                limitations=limitations,
                component_status=component_status,
                status=status,
            )
        except Exception:
            self._mark_failed()
            raise
