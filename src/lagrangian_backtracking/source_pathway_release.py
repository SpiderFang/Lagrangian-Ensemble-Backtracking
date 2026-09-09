"""向下沉降粒子的來源路徑圖與獨立成果包。

本模組把已驗證的 aggregate release 與同一份 ``ReportSpec`` 轉成可重建的
``source-pathway-v1`` 成果目錄。它只接受嚴格小於零的沉降速度；零沉降與向上
浮升情境會在統計建置前拒絕，因此圖面不會把不同垂向物理混在同一個 pooled
footprint。統計量一律沿用 ``build_report_statistics`` 已建立的 pathway、KDE/HDR
與 outcome products，繪圖層不重新決定分母、分位數、頻寬或補零政策。

每站成果包含一張六面板圖，以及 ``grid.parquet``、``boundary.parquet``、
``outcomes.parquet`` 和 caption。格網所有距離使用 local 公尺座標；停留時間與
首次通過年齡先保存秒，再提供小時／日的顯示欄位。訪格 numerator 是每粒子每格
只計一次的有效成員數，停留秒數則保留重複迴游造成的時間，因此兩者不合併成同一
個濃度量。KDE 的格網機率欄位只代表目前格網內的相對權重；未建立來源先驗、似然
與觀測驗證前，成果稱為「條件式來源足跡」或「相對來源權重」，不是絕對來源機率、
沉積質量、沉積濃度或因果歸因。

成果包是獨立於完整 report-v1 F01--F12/T01--T06 registry 的 schema 1.0.0 產品。
writer 以 sibling partial 目錄與 atomic rename 發布，validator 只回傳 JSON-safe
摘要，不洩漏 caller 的絕對路徑。即使 synthetic fixture 通過工程驗證，也不代表
正式 OCM schema 3／NWW3 schema 1 科學成果。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .aggregate_release import read_aggregate_release
from .aggregate_release_payload import AggregateReleasePayload
from .event_aggregation import BoundaryAggregateKey
from .models import ParticleStatus
from .report_spec import (
    ReportSpec,
    load_report_spec,
    validate_report_spec_against_aggregate_spec,
)
from .report_statistics import ReportStatistics, build_report_statistics
from .report_style import report_render_style_context

__all__ = [
    "SOURCE_PATHWAY_RELEASE_SCHEMA_VERSION",
    "SOURCE_PATHWAY_FINAL_SUFFIX",
    "build_source_pathway_release",
    "read_source_pathway_release",
    "validate_source_pathway_release",
]


SOURCE_PATHWAY_RELEASE_SCHEMA_VERSION = "1.0.0"
SOURCE_PATHWAY_FINAL_SUFFIX = ".source-pathway-v1"
_MANIFEST_FILE_NAME = "manifest.json"
_README_FILE_NAME = "README.md"
_SITE_DIR_NAME = "sites"
_DAY_SECONDS = 86_400.0
_HOUR_SECONDS = 3_600.0
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_TEXT_CHUNK_TYPES = frozenset({b"tEXt", b"zTXt", b"iTXt"})
_EVIDENCE_CLASS_BY_RUN_KIND = {
    "synthetic": "synthetic_engineering_evidence",
    "pilot": "pilot_aggregate_evidence",
    "formal": "formal_aggregate_evidence",
}
_KDE_STATUS_VALUES = frozenset({"available", "zero_raw_count", "below_minimum_raw_count"})
_EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "release_kind",
        "run_kind",
        "experiment_case_id",
        "run_id",
        "members_per_scenario",
        "evidence_class",
        "semantics",
        "provenance",
        "materials",
        "settling_velocity_range_mps",
        "vertical_ids",
        "arrival_ids",
        "scenario_strata",
        "pooling",
        "sites",
        "files",
    }
)
_GRID_COLUMNS = frozenset(
    {
        "cell_y_index",
        "cell_x_index",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "visit_numerator",
        "visit_fraction",
        "residence_time_seconds",
        "residence_hours_per_member",
        "first_passage_q25_seconds",
        "first_passage_q25_days",
        "first_passage_q50_seconds",
        "first_passage_q50_days",
        "first_passage_q75_seconds",
        "first_passage_q75_days",
        "low_sample",
        "local_first_exit_count",
        "outer_first_exit_count",
        "bed_first_contact_count",
        "bed_repeated_contact_count",
        "data_gap_failure_count",
        "numerical_failure_count",
        "primary_kde_status",
        "primary_kde_cell_probability",
        "primary_kde_density_per_m2",
    }
)
_BOUNDARY_COLUMNS = frozenset(
    {
        "boundary_kind",
        "boundary_segment_id",
        "bin_index",
        "s_lower_m",
        "s_upper_m",
        "raw_count",
        "fraction_valid_members",
        "fraction_within_kind",
        "travel_age_median_seconds",
        "travel_age_median_days",
    }
)
_OUTCOME_COLUMNS = frozenset(
    {
        "status",
        "status_label_zh",
        "raw_count",
        "total_member_denominator",
        "total_denominator_fraction",
        "valid_member_denominator",
        "valid_denominator_fraction",
    }
)
_CITATIONS = (
    {
        "authors": "van Sebille et al.",
        "year": 2018,
        "doi": "10.1016/j.ocemod.2017.11.008",
        "reason": "訪格一次計數與 residence time 分離",
    },
    {
        "authors": "Rypina et al.",
        "year": 2014,
        "doi": "10.1002/2014JC010306",
        "reason": "訪格比例與 first-arrival age 成對呈現",
    },
    {
        "authors": "Cedarholm et al.",
        "year": 2019,
        "doi": "10.1029/2019GL082500",
        "reason": "成功路徑的訪格與到達時間；代表線只作輔助",
    },
    {
        "authors": "Abascal et al.",
        "year": 2012,
        "doi": "10.1007/s10236-012-0546-4",
        "reason": "逆向終點 KDE 與高密度區輪廓",
    },
    {
        "authors": "Carlson et al.",
        "year": 2017,
        "doi": "10.3389/fmars.2017.00078",
        "reason": "邊界分段連通與停止結果",
    },
    {
        "authors": "Baudena et al.",
        "year": 2023,
        "doi": "10.1021/acs.est.2c08873",
        "reason": "離開表層、海床終點與 connectivity 診斷分開",
    },
    {
        "authors": "Nooteboom et al.",
        "year": 2020,
        "doi": "10.1371/journal.pone.0238650",
        "reason": "由海底位置逆推的系集分布、橫向位移與覆蓋面積呈現",
    },
)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """以固定 UTF-8 compact JSON 序列化 manifest／caption。

    成果包的 JSON 是 provenance 輸出，故禁止 ``NaN``、``Infinity`` 與自訂型別；
    排序鍵名及固定尾端換行讓 bytes 可由 validator 直接重建，比較結果不依賴
    dictionary insertion order。
    """

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(raw: bytes) -> str:
    """計算實際輸出 bytes 的完整小寫 SHA-256。"""

    return hashlib.sha256(raw).hexdigest()


def _aggregate_manifest_sha256(aggregate_release_path: Path) -> str:
    """讀取已驗證 aggregate release 的固定 manifest bytes 並計算摘要。

    aggregate reader 會先驗證 release topology 與所有 checksum；這裡只再讀固定檔名
    的 manifest，將 source release 本身綁進 source-pathway provenance，不把 caller
    提供的絕對位置寫入成果包。
    """

    manifest_path = aggregate_release_path / "aggregate_manifest.json"
    try:
        metadata = os.lstat(manifest_path)
    except OSError as error:
        raise ValueError("aggregate release manifest 無法讀取") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("aggregate release manifest 必須是普通檔案")
    return _sha256_bytes(manifest_path.read_bytes())


def _require_regular_directory(path: Path, *, label: str) -> None:
    """要求最後節點是現有 ordinary directory，避免沿用 symlink。"""

    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} 必須是既有普通目錄") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} 必須是既有普通目錄")


def _require_absent(path: Path, *, label: str) -> None:
    """拒絕任何既有 final／partial 節點，保證 writer 不覆寫。"""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(f"{label} 無法檢查") from error
    raise FileExistsError(f"{label} 已存在")


def _write_exclusive(path: Path, raw: bytes, *, label: str) -> None:
    """以 exclusive create 寫入並 fsync 一個普通檔案。"""

    if type(raw) is not bytes:
        raise TypeError(f"{label} 必須是 bytes")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("zero-byte write")
            offset += written
        os.fsync(descriptor)
    except FileExistsError:
        raise
    except OSError as error:
        raise ValueError(f"{label} 寫入失敗") from error
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _safe_json_load(raw: bytes, *, label: str) -> object:
    """以 duplicate-key／非有限數字拒絕策略讀取 JSON。"""

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} 含非有限 JSON 數值：{token}")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} 含重複 JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} JSON 無法嚴格讀取") from error
    return value


def _json_safe(value: object, *, label: str = "value") -> object:
    """把 manifest／caption 用值檢查為 JSON-safe，拒絕 Path、NumPy 與非有限 float。"""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} 不可包含 NaN 或 Infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} key 必須是原生 str")
            copied[key] = _json_safe(item, label=f"{label}[{key!r}]")
        return copied
    raise TypeError(f"{label} 含不支援的 JSON 型別")


def _contains_absolute_path(value: object) -> bool:
    """遞迴檢查 manifest 是否誤含以 slash 開頭的絕對位置。"""

    if type(value) is str:
        return value.startswith("/") or value.startswith("~/")
    if isinstance(value, Mapping):
        return any(
            _contains_absolute_path(key) or _contains_absolute_path(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _require_safe_component(value: object, *, label: str) -> str:
    """驗證會被拼入成果相對路徑的識別碼。

    AggregateSpec 本身已有站點 slug 閘門，但 validator 也必須獨立防禦被竄改的
    manifest；若直接把 ``../`` 或 slash 放入 site id，預期檔案清單可能跳出 release
    root。此函式只接受既有資料契約使用的 ASCII slug，不改寫輸入也不以 basename
    靜默修剪，確保 sidecar 路徑與 manifest 的識別碼一一對應。
    """

    if not isinstance(value, str) or _SAFE_COMPONENT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是安全的 ASCII slug")
    return value


def _none_if_nan(value: object) -> float | None:
    """把統計產品的 NaN 轉成 Parquet nullable null，不把不可用誤寫成零。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fraction(numerator: int, denominator: int) -> float | None:
    """依明示分母計算比例；零分母保留 null。"""

    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _histogram_midpoint_quantile(
    counts: np.ndarray,
    edges: np.ndarray,
    quantile: float = 0.5,
) -> float | None:
    """從既有 age histogram 以固定 bin midpoint 規則取得分位數秒數。

    聚合 release 只保存分箱計數，不保存每個粒子的未分箱年齡；因此回傳值是
    canonical age-bin midpoint 近似。空 histogram 回傳 ``None``，讓 boundary
    sidecar 與圖面都能區分沒有事件和真正的零秒事件。
    """

    if counts.ndim != 1 or edges.ndim != 1 or edges.size != counts.size + 1:
        raise ValueError("travel age histogram 與 age edges 形狀不一致")
    total = int(sum(int(item) for item in counts.flat))
    if total <= 0:
        return None
    target = float(quantile) * total
    cumulative = 0
    selected_index = 0
    for candidate_index, item in enumerate(counts):
        cumulative += int(item)
        if cumulative >= target:
            selected_index = candidate_index
            break
    return float((edges[selected_index] + edges[selected_index + 1]) / 2.0)


def _downward_material_snapshot(payload: AggregateReleasePayload) -> tuple[dict[str, object], ...]:
    """驗證所有 scenario 的沉降速度為負，並建立 material provenance snapshot。"""

    materials: dict[str, float] = {}
    for stratum in payload.scenario_strata:
        velocity = float(stratum.settling_velocity_mps)
        if not math.isfinite(velocity) or velocity >= 0.0:
            raise ValueError("source pathway 只接受 settling_velocity_mps < 0 的向下沉降粒子")
        previous = materials.get(stratum.material_id)
        if previous is not None and previous != velocity:
            raise ValueError("同一 material_id 的 settling velocity 不一致")
        materials[stratum.material_id] = velocity
    return tuple(
        {"material_id": material_id, "settling_velocity_mps": velocity}
        for material_id, velocity in sorted(materials.items())
    )


def _site_path(site_id: str, *, filename: str) -> str:
    """建立 site sidecar 的固定 POSIX relative path。"""

    return f"{_SITE_DIR_NAME}/{site_id}/{filename}"


def _parquet_bytes(table: pa.Table) -> bytes:
    """以無壓縮、停用 dictionary 的固定 Arrow writer policy 序列化表格。"""

    import io

    sink = io.BytesIO()
    pq.write_table(table, sink, compression="NONE", use_dictionary=False, write_statistics=False)
    return sink.getvalue()


def _build_grid_table(
    *,
    payload: AggregateReleasePayload,
    site_id: str,
    statistics: ReportStatistics,
) -> pa.Table:
    """由既有 pathway／KDE／事件 products 建立單站格網 sidecar。

    這裡只做欄位對齊與單位轉換：訪格比例、first-passage quantiles、停留時間與
    KDE 都來自 facade，底床接觸／失敗網格則來自同一個 EventAggregateChunk。低樣本
    格與零分母不清除原始 count，也不以 0 取代 null。
    """

    pathway = statistics.pathway_statistics_by_site[site_id]
    grid_counts = payload.event_aggregate.site_grid_counts[site_id]
    kde_product = statistics.kde_statistics_by_site[site_id]
    kde_layer = kde_product.layers[kde_product.primary_bandwidth_m]
    shape = pathway.visit_numerator.shape
    quantiles = pathway.first_passage_quantiles
    q25 = quantiles.get(0.25)
    q50 = quantiles.get(0.5)
    q75 = quantiles.get(0.75)
    if q25 is None or q50 is None or q75 is None:
        raise ValueError("ReportSpec pathway quantiles 必須包含 0.25、0.5、0.75")

    denominator = pathway.valid_member_denominator
    primary_probability = None if kde_layer.grid is None else kde_layer.grid.cell_probability
    primary_density = None if kde_layer.grid is None else kde_layer.grid.density_per_m2
    status = kde_layer.status.value
    x_edges = pathway.x_edges_m
    y_edges = pathway.y_edges_m
    data: dict[str, list[object]] = {name: [] for name in _GRID_COLUMNS}
    for y_index in range(shape[0]):
        for x_index in range(shape[1]):
            visit = int(pathway.visit_numerator[y_index, x_index])
            residence_seconds = float(pathway.residence_time_seconds[y_index, x_index])
            data["cell_y_index"].append(y_index)
            data["cell_x_index"].append(x_index)
            data["x_min_m"].append(float(x_edges[x_index]))
            data["x_max_m"].append(float(x_edges[x_index + 1]))
            data["y_min_m"].append(float(y_edges[y_index]))
            data["y_max_m"].append(float(y_edges[y_index + 1]))
            data["visit_numerator"].append(visit)
            data["visit_fraction"].append(_none_if_nan(pathway.visit_fraction[y_index, x_index]))
            data["residence_time_seconds"].append(residence_seconds)
            data["residence_hours_per_member"].append(
                None if denominator == 0 else residence_seconds / _HOUR_SECONDS / denominator
            )
            for name, values, divisor in (
                ("first_passage_q25_seconds", q25, 1.0),
                ("first_passage_q50_seconds", q50, 1.0),
                ("first_passage_q75_seconds", q75, 1.0),
            ):
                seconds = _none_if_nan(values[y_index, x_index])
                data[name].append(seconds)
                days_name = name.replace("_seconds", "_days")
                data[days_name].append(None if seconds is None else seconds / _DAY_SECONDS / divisor)
            data["low_sample"].append(bool(pathway.low_sample_mask[y_index, x_index]))
            data["local_first_exit_count"].append(int(grid_counts.local_first_exit_count[y_index, x_index]))
            data["outer_first_exit_count"].append(int(grid_counts.outer_first_exit_count[y_index, x_index]))
            data["bed_first_contact_count"].append(int(grid_counts.bed_first_contact_count[y_index, x_index]))
            data["bed_repeated_contact_count"].append(
                int(grid_counts.bed_repeated_contact_count[y_index, x_index])
            )
            data["data_gap_failure_count"].append(int(grid_counts.data_gap_failure_count[y_index, x_index]))
            data["numerical_failure_count"].append(int(grid_counts.numerical_failure_count[y_index, x_index]))
            data["primary_kde_status"].append(status)
            data["primary_kde_cell_probability"].append(
                None if primary_probability is None else float(primary_probability[y_index, x_index])
            )
            data["primary_kde_density_per_m2"].append(
                None if primary_density is None else float(primary_density[y_index, x_index])
            )

    return pa.table(data)


def _build_boundary_table(
    *,
    payload: AggregateReleasePayload,
    site_id: str,
    valid_denominator: int,
) -> pa.Table:
    """由事件 aggregate 的完整 local／outer boundary bins 建立 sidecar。"""

    event = payload.event_aggregate
    spec = payload.aggregate_spec
    data: dict[str, list[object]] = {name: [] for name in _BOUNDARY_COLUMNS}
    keys_by_kind: dict[str, list[BoundaryAggregateKey]] = {"local": [], "outer": []}
    site_segments = spec.site_boundary_segment_ids[site_id]
    for kind, segment_ids in (
        ("local", site_segments.local_segment_ids),
        ("outer", site_segments.outer_segment_ids),
    ):
        for segment_id in segment_ids:
            keys_by_kind[kind].append(
                BoundaryAggregateKey(
                    study_site_id=site_id,
                    boundary_kind=kind,
                    boundary_segment_id=segment_id,
                )
            )
    totals_by_kind = {
        kind: sum(
            int(sum(int(item) for item in event.boundary_arclength_raw_count[key].flat)) for key in keys
        )
        for kind, keys in keys_by_kind.items()
    }
    for kind in ("local", "outer"):
        for key in keys_by_kind[kind]:
            edges = event.boundary_bin_edges_m[key]
            raw_counts = event.boundary_arclength_raw_count[key]
            travel_histogram = event.boundary_travel_age_histogram[key]
            for bin_index, raw_value in enumerate(raw_counts):
                count = int(raw_value)
                median_seconds = _histogram_midpoint_quantile(
                    travel_histogram[bin_index],
                    event.age_bin_edges_seconds,
                )
                data["boundary_kind"].append(kind)
                data["boundary_segment_id"].append(key.boundary_segment_id)
                data["bin_index"].append(bin_index)
                data["s_lower_m"].append(float(edges[bin_index]))
                data["s_upper_m"].append(float(edges[bin_index + 1]))
                data["raw_count"].append(count)
                data["fraction_valid_members"].append(_fraction(count, valid_denominator))
                data["fraction_within_kind"].append(_fraction(count, totals_by_kind[kind]))
                data["travel_age_median_seconds"].append(median_seconds)
                data["travel_age_median_days"].append(
                    None if median_seconds is None else median_seconds / _DAY_SECONDS
                )
    return pa.table(data)


def _status_label(status: str) -> str:
    """提供圖面用中文狀態名稱，保留 ``deposited`` 的 BED_DEPOSITED 標示。"""

    labels = {
        ParticleStatus.FLOW_DOMAIN_EXIT.value: "FLOW_DOMAIN_OPEN_EXIT",
        ParticleStatus.COAST_CONTACT.value: "COAST_CONTACT",
        ParticleStatus.SURFACE_REGIME_EXIT.value: "SURFACE_REGIME_EXIT",
        ParticleStatus.DEPOSITED.value: "BED_DEPOSITED",
        ParticleStatus.FORCING_START.value: "FORCING_START",
        ParticleStatus.DATA_GAP.value: "DATA_GAP",
        ParticleStatus.MAX_AGE.value: "MAX_AGE",
        ParticleStatus.NUMERICAL_FAILURE.value: "NUMERICAL_FAILURE",
    }
    return labels.get(status, status)


def _build_outcome_table(
    *,
    statistics: ReportStatistics,
    site_id: str,
) -> pa.Table:
    """建立保存 total 與 valid 分母的停止結果表。"""

    outcomes = statistics.outcome_statistics
    total = outcomes.total_member_denominator_by_site[site_id]
    valid = outcomes.valid_member_denominator_by_site[site_id]
    data: dict[str, list[object]] = {name: [] for name in _OUTCOME_COLUMNS}
    for status, raw_value in outcomes.outcome_count_by_site[site_id].items():
        count = int(raw_value)
        data["status"].append(status)
        data["status_label_zh"].append(_status_label(status))
        data["raw_count"].append(count)
        data["total_member_denominator"].append(total)
        data["total_denominator_fraction"].append(_fraction(count, total))
        data["valid_member_denominator"].append(valid)
        data["valid_denominator_fraction"].append(_fraction(count, valid))
    return pa.table(data)


def _draw_hatch(ax: object, x_edges: np.ndarray, y_edges: np.ndarray, mask: np.ndarray) -> None:
    """在低樣本格上加斜線遮罩，保留 raw fraction 且不插值補值。"""

    if not np.any(mask):
        return
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        # contourf 對 1×1／1×N 格網沒有可形成的等值線；以逐格 Rectangle
        # 保留斜線語意，避免小型 synthetic fixture 被 matplotlib 維度限制阻斷。
        from matplotlib.patches import Rectangle

        for y_index, x_index in zip(*np.nonzero(mask), strict=True):
            ax.add_patch(
                Rectangle(
                    (float(x_edges[x_index]), float(y_edges[y_index])),
                    float(x_edges[x_index + 1] - x_edges[x_index]),
                    float(y_edges[y_index + 1] - y_edges[y_index]),
                    fill=False,
                    hatch="///",
                    edgecolor="#333333",
                    linewidth=0.0,
                )
            )
        return
    # contourf 只負責視覺標記，不改變資料陣列；以透明 fill 加斜線避免把低樣本
    # 格看成平滑連續場；透明 fill 只保留斜線標記，不改變 sidecar 的數值。
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    masked = np.where(mask, 1.0, np.nan)
    collection = ax.contourf(x_centers, y_centers, masked, levels=[0.5, 1.5], colors="none", hatches=["///"])
    collection.set_edgecolor("#333333")
    collection.set_linewidth(0.0)


def _png_text_metadata_keywords(raw: bytes) -> tuple[bytes, ...]:
    """解析 PNG 文字 metadata 的 keyword，不掃描壓縮像素資料。

    PNG 的影像資料放在 ``IDAT`` 壓縮區塊，任意位元組序列都可能偶然包含 ``date``；
    直接搜尋整份檔案會因此把正常圖檔誤判為含建置時間。這個小型 parser 只走 PNG 的
    length／type／data／CRC 區塊結構，並取出 ``tEXt``、``zTXt``、``iTXt`` 在第一個
    NUL 前的 keyword。輸入是 Matplotlib 剛輸出的記憶體 bytes，CRC 與完整影像解碼
    由 writer／後續讀檔驗證負責；此處只驗證足以安全定位 metadata 的結構邊界。
    """

    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("PNG signature 無效")
    keywords: list[bytes] = []
    offset = len(_PNG_SIGNATURE)
    saw_end = False
    while offset < len(raw):
        # 每個區塊至少包含 4-byte 長度、4-byte type 與 4-byte CRC；先驗證邊界，
        # 避免截斷檔案讓長度欄位導向 PNG bytes 之外。
        if len(raw) - offset < 12:
            raise ValueError("PNG chunk 結構遭截斷")
        data_length = int.from_bytes(raw[offset : offset + 4], byteorder="big")
        chunk_type = raw[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + data_length
        next_offset = data_end + 4
        if next_offset > len(raw):
            raise ValueError("PNG chunk 長度超出檔案邊界")
        if chunk_type in _PNG_TEXT_CHUNK_TYPES:
            keyword, separator, _ = raw[data_start:data_end].partition(b"\x00")
            if not separator:
                raise ValueError("PNG 文字 metadata 缺少 keyword 分隔符")
            keywords.append(keyword.strip().lower())
        offset = next_offset
        if chunk_type == b"IEND":
            saw_end = True
            break
    if not saw_end:
        raise ValueError("PNG 缺少 IEND 區塊")
    return tuple(keywords)


def _assert_no_wall_clock_metadata(raw: bytes, *, file_format: str) -> None:
    """確認 backend 沒有偷偷把目前時間寫入圖檔 metadata。

    source-pathway-v1 的 manifest 只保存實際 bytes hash；若 Matplotlib backend 自動
    寫入當下日期，相同 aggregate／ReportSpec 每次建置都會得到不同圖檔。三種格式
    的合法 metadata key 由 ``_draw_site_figure`` 分開指定，此處再檢查不應出現的日期
    token，避免格式間的 backend 預設值破壞可重現性。
    """

    lowered = raw.lower()
    if file_format == "svg" and any(
        token in lowered for token in (b"<dc:date", b"<date", b"creationdate", b"moddate")
    ):
        raise ValueError("SVG 不得含 wall-clock date metadata")
    if file_format == "pdf" and (b"/creationdate" in lowered or b"/moddate" in lowered):
        raise ValueError("PDF 不得含 wall-clock date metadata")
    if file_format == "png":
        date_tokens = (b"date", b"creationdate", b"moddate")
        if any(token in keyword for keyword in _png_text_metadata_keywords(raw) for token in date_tokens):
            raise ValueError("PNG 不得含 wall-clock date metadata")


def _draw_site_figure(
    *,
    payload: AggregateReleasePayload,
    report_spec: ReportSpec,
    statistics: ReportStatistics,
    site_id: str,
) -> dict[str, bytes]:
    """建立六面板站點圖並以固定 metadata 輸出 PNG／SVG／PDF bytes。

    A、B、C、D 都共用同一個公尺制 extent 與等比例座標。A 與 B 是成對的每格訪格
    numerator／first-passage age；D 單獨呈現 residence hours/member，避免重複迴游
    時間被誤讀成 visit fraction。A／B／D 的無樣本遮罩依 pathway visit-once 分母，
    C 則依 local first-exit／KDE 自身樣本呈現，讓 KDE 平滑不被 pathway 遮罩截斷。
    C 在 KDE unavailable 時顯示 raw count 與狀態文字，不以低樣本平滑或零值冒充可估計密度；
    D 以底床首次接觸 count 的輪廓作診斷，不把輪廓稱為沉積質量或濃度。
    """

    import matplotlib.pyplot as plt

    pathway = statistics.pathway_statistics_by_site[site_id]
    event_grid = payload.event_aggregate.site_grid_counts[site_id]
    kde = statistics.kde_statistics_by_site[site_id]
    primary_layer = kde.layers[kde.primary_bandwidth_m]
    x_edges = pathway.x_edges_m
    y_edges = pathway.y_edges_m
    extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
    no_sample_mask = pathway.visit_numerator <= 0
    low_sample_hatch = pathway.low_sample_mask & ~no_sample_mask
    q50 = pathway.first_passage_quantiles[0.5] / _DAY_SECONDS
    if pathway.valid_member_denominator == 0:
        residence = np.full(pathway.residence_time_seconds.shape, np.nan, dtype=float)
    else:
        residence = pathway.residence_time_seconds / _HOUR_SECONDS / pathway.valid_member_denominator
    local_counts = event_grid.local_first_exit_count
    if primary_layer.grid is not None:
        c_values = primary_layer.grid.cell_probability
        c_title = "C  潛在移入入口（逆向首次離開）"
        c_label = "格網內相對權重"
    else:
        c_values = local_counts.astype(float)
        c_title = "C  潛在移入入口（逆向首次離開）"
        c_label = "原始計數"

    figure = plt.figure(figsize=(15.5, 9.6), constrained_layout=True)
    axes = figure.subplot_mosaic(
        [["A", "B", "C"], ["D", "E", "F"]],
        gridspec_kw={"height_ratios": [1.0, 0.9]},
    )
    panels = (
        ("A", pathway.visit_fraction, "A  訪格比例（條件式來源足跡）", "訪格比例", "viridis"),
        ("B", q50, "B  中位首次通過年齡（first-passage age）", "日", "magma"),
        ("C", c_values, c_title, c_label, "Blues"),
        ("D", residence, "D  每有效成員停留時數", "小時／成員", "YlGnBu"),
    )
    for key, values, title, colorbar_label, cmap in panels:
        axis = axes[key]
        array = np.asarray(values, dtype=float)
        if key in {"A", "B", "D"}:
            # A／B／D 的可估計樣本語意由 pathway visit-once 分母決定；沒有任何
            # 首次訪問的 cell 不代表可估計的零值，圖面留白，sidecar 仍保留 raw
            # count／fraction 與 low_sample 狀態。C 的 local first-exit/KDE 有自己
            # 的樣本支持，KDE 平滑可延伸到沒有 pathway visit 的 cell，故不套用這個 mask。
            array = np.where(no_sample_mask, np.nan, array)
        finite = array[np.isfinite(array)]
        image_kwargs: dict[str, object] = {"cmap": cmap, "shading": "flat"}
        if finite.size and np.nanmax(finite) > np.nanmin(finite):
            mesh = axis.pcolormesh(x_edges, y_edges, array, **image_kwargs)
        else:
            mesh = axis.pcolormesh(x_edges, y_edges, array, **image_kwargs)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("x（m）")
        axis.set_ylabel("y（m）")
        colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label(colorbar_label)
        if key in {"A", "B", "D"}:
            _draw_hatch(axis, x_edges, y_edges, low_sample_hatch)
    axes["A"].text(
        0.02,
        0.98,
        (
            f"有效成員 n={pathway.valid_member_denominator}\n"
            f"斜線：低樣本（0<訪格數<{pathway.low_sample_min_member_count}）；空白：無樣本"
        ),
        transform=axes["A"].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
    )
    if primary_layer.grid is not None:
        contour_axis = axes["C"]
        for level, color in zip(kde.hdr_levels, ("#ffffff", "#ffcc00", "#ff4d4d"), strict=True):
            mask = primary_layer.grid.hdr_masks[level].astype(float)
            if np.any(mask) and mask.shape[0] >= 2 and mask.shape[1] >= 2:
                contour_axis.contour(
                    (x_edges[:-1] + x_edges[1:]) / 2.0,
                    (y_edges[:-1] + y_edges[1:]) / 2.0,
                    mask,
                    levels=[0.5],
                    colors=[color],
                    linewidths=1.1,
                )
        hdr_note = (
            f"KDE 狀態：{primary_layer.status.value}；原始樣本={primary_layer.raw_point_count}\n"
            "HDR 50／75／90%"
        )
        if primary_layer.grid.hdr_masks[0.5].shape[0] < 2 or primary_layer.grid.hdr_masks[0.5].shape[1] < 2:
            hdr_note += "\n1×N 不繪輪廓\n遮罩見 sidecar"
        axes["C"].text(
            0.02,
            0.02,
            hdr_note,
            transform=axes["C"].transAxes,
            fontsize=8,
            color="#222222",
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
        )
    else:
        axes["C"].text(
            0.02,
            0.02,
            f"KDE 狀態：{primary_layer.status.value}；保留 local first-exit 原始計數",
            transform=axes["C"].transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
        )

    bed_mask = event_grid.bed_first_contact_count > 0
    if np.any(bed_mask) and bed_mask.shape[0] >= 2 and bed_mask.shape[1] >= 2:
        axes["D"].contour(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
            event_grid.bed_first_contact_count,
            levels=[0.5],
            colors=["#d7191c"],
            linewidths=1.4,
        )
        axes["D"].text(
            0.02,
            0.02,
            "紅線：bed_first_contact_count≥1（底床邊界接觸診斷）",
            transform=axes["D"].transAxes,
            fontsize=8,
            color="#a50026",
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
        )
    elif np.any(bed_mask):
        axes["D"].text(
            0.02,
            0.02,
            "bed_first_contact_count≥1（1×N 格網，sidecar 保存診斷）",
            transform=axes["D"].transAxes,
            fontsize=8,
            color="#a50026",
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
        )

    boundary = _build_boundary_table(
        payload=payload,
        site_id=site_id,
        valid_denominator=pathway.valid_member_denominator,
    )
    boundary_dict = boundary.to_pydict()
    local_rows = [index for index, kind in enumerate(boundary_dict["boundary_kind"]) if kind == "local"]
    e_axis = axes["E"]
    if local_rows:
        positions = np.arange(len(local_rows), dtype=float)
        heights = np.asarray([boundary_dict["raw_count"][index] for index in local_rows], dtype=float)
        fractions = np.asarray(
            [
                boundary_dict["fraction_within_kind"][index]
                if boundary_dict["fraction_within_kind"][index] is not None
                else np.nan
                for index in local_rows
            ],
            dtype=float,
        )
        e_axis.bar(positions, heights, color="#2A9D8F", alpha=0.85, label="原始計數")
        fraction_axis = e_axis.twinx()
        fraction_axis.plot(
            positions, fractions, color="#E76F51", marker="o", linewidth=1.2, label="局部類別內相對比例"
        )
        fraction_axis.set_ylabel("局部類別內相對比例")
        e_axis.set_xticks(positions)
        e_axis.set_xticklabels(
            [
                f"{boundary_dict['boundary_segment_id'][index]}\n{boundary_dict['bin_index'][index]}"
                for index in local_rows
            ],
            rotation=45,
            ha="right",
        )
        e_axis.set_xlabel("局部邊界分段／弧長分箱（s-bin）")
        e_axis.set_ylabel("弧長原始計數")
        e_axis.legend(loc="upper left")
        fraction_axis.legend(loc="upper right")
    else:
        e_axis.text(0.5, 0.5, "沒有 local boundary bin", ha="center", va="center")
        e_axis.set_axis_off()
    e_axis.set_title("E  潛在移入邊界區段")

    outcome = _build_outcome_table(statistics=statistics, site_id=site_id).to_pydict()
    f_axis = axes["F"]
    positions = np.arange(len(outcome["status"]), dtype=float)
    counts = np.asarray(outcome["raw_count"], dtype=float)
    fractions = np.asarray(
        [value if value is not None else np.nan for value in outcome["total_denominator_fraction"]],
        dtype=float,
    )
    f_axis.bar(positions, counts, color="#6A4C93", alpha=0.85)
    f_fraction_axis = f_axis.twinx()
    f_fraction_axis.plot(positions, fractions, color="#F4A261", marker="o", linewidth=1.2)
    f_axis.set_xticks(positions)
    f_axis.set_xticklabels(outcome["status_label_zh"], rotation=45, ha="right")
    f_axis.set_ylabel("停止原始計數")
    f_fraction_axis.set_ylabel("總成員分母比例")
    f_axis.set_title("F  停止結果與品質檢查")
    f_axis.text(
        0.01,
        0.98,
        (
            f"total={outcome['total_member_denominator'][0] if outcome['total_member_denominator'] else 0}；"
            f"valid={outcome['valid_member_denominator'][0] if outcome['valid_member_denominator'] else 0}"
        ),
        transform=f_axis.transAxes,
        va="top",
        fontsize=8,
    )
    f_axis.text(
        0.01,
        0.91,
        "失敗與截尾皆納入分母",
        transform=f_axis.transAxes,
        va="top",
        fontsize=8,
    )

    figure.suptitle(
        f"{site_id}｜向下沉降粒子移入關注海域的條件式來源足跡",
        fontsize=14,
    )
    outputs: dict[str, bytes] = {}
    import io

    for file_format in ("png", "svg", "pdf"):
        stream = io.BytesIO()
        if file_format == "svg":
            metadata = {
                "Creator": "lagrangian_backtracking.source_pathway_release",
                "Title": f"{site_id} source pathway",
                "Date": None,
            }
        elif file_format == "pdf":
            metadata = {
                "Creator": "lagrangian_backtracking.source_pathway_release",
                "Title": f"{site_id} source pathway",
                "Subject": "downward-settling conditional source footprint",
                "CreationDate": None,
                "ModDate": None,
            }
        else:
            metadata = {
                "Software": "lagrangian_backtracking.source_pathway_release",
                "Title": f"{site_id} source pathway",
                "Description": "reproducible downward-settling source pathway artifact",
            }
        figure.savefig(stream, format=file_format, dpi=report_spec.raster_dpi, metadata=metadata)
        raw = stream.getvalue()
        if not raw:
            raise RuntimeError("source pathway figure bytes 為空")
        _assert_no_wall_clock_metadata(raw, file_format=file_format)
        outputs[file_format] = raw
    plt.close(figure)
    return outputs


def _caption_document(
    *,
    payload: AggregateReleasePayload,
    report_spec: ReportSpec,
    site_id: str,
    statistics: ReportStatistics,
) -> dict[str, object]:
    """建立可追溯的站點 caption／方法與限制說明。"""

    pathway = statistics.pathway_statistics_by_site[site_id]
    kde = statistics.kde_statistics_by_site[site_id]
    layer = kde.layers[kde.primary_bandwidth_m]
    site_strata = tuple(stratum for stratum in payload.scenario_strata if stratum.study_site_id == site_id)
    if not site_strata:
        raise ValueError("caption site strata 不得為空")
    velocities = tuple(float(stratum.settling_velocity_mps) for stratum in site_strata)
    evidence_statement = (
        "本圖為指定受體、到達時刻、向下沉降材質與流場條件下的條件式來源足跡／"
        "相對來源權重，不是絕對來源機率、沉積質量、沉積濃度或因果歸因。"
    )
    first_passage_description = "首次通過年齡由秒制 histogram 的固定 bin midpoint 近似；與訪格比例成對讀取。"
    kde_description = (
        "達門檻時顯示 primary KDE cell_probability 與 50／75／90% HDR 輪廓，"
        "逆向 local_first_exit 在條件式解讀下對應正向潛在移入入口；這是邊界事件診斷，"
        "不是確定來源，且不套用 pathway visit mask；"
        "否則明示 zero/below-minimum 並保留 raw count。"
    )
    residence_description = (
        "以線段切格後累積的停留時間；與 A 的 visit-once numerator 分圖。"
        "紅線只表示 bed_first_contact_count≥1 的底床邊界接觸診斷。"
    )
    boundary_description = (
        "local boundary 分段與弧長 bin 的入口事件，保存 valid denominator與 kind 內相對比例。"
    )
    outcome_description = (
        "完整列出 BED_DEPOSITED、DATA_GAP、NUMERICAL_FAILURE、outer exit、MAX_AGE 與其他停止狀態。"
    )
    transfer_limit = (
        "文獻中的 surface／forward／observed-source 或 source release assumptions "
        "不直接移植；本成果沒有來源質量先驗、表層釋放通量或再懸浮模型，"
        "只呈現向下沉降受體的逆向條件式產品。"
    )
    return {
        "schema_version": "source_pathway_caption_v1",
        "site_id": site_id,
        "title_zh": f"{site_id} 向下沉降粒子條件式來源足跡",
        "evidence_statement": evidence_statement,
        "material_scope": "僅納入 settling_velocity_mps < 0；零沉降懸浮材質與向上浮升材質不納入。",
        "scenario_strata": [
            {
                "scenario_id": stratum.scenario_id,
                "material_id": stratum.material_id,
                "settling_velocity_mps": float(stratum.settling_velocity_mps),
                "vertical_id": stratum.vertical_id,
                "arrival_time_id": stratum.arrival_time_id,
                "total_member_denominator": payload.members_per_scenario,
                "valid_member_denominator": None,
                "valid_member_denominator_scope": "site/receptor pooled in aggregate release",
            }
            for stratum in site_strata
        ],
        "settling_velocity_range_mps": [min(velocities), max(velocities)],
        "pooling_semantics": {
            "weighting": "executed_member_weighted",
            "claim": "條件於本次情境設計與有效成員",
            "design_balanced_claim": False,
            "natural_material_fraction_inferred": False,
        },
        "panel_semantics": [
            {
                "panel": "A",
                "quantity": "visit_fraction",
                "unit": "fraction",
                "denominator": "valid_member_denominator",
                "description": "每個有效成員在每格只計首次訪問；低樣本以斜線標示，沒有樣本保留空白。",
            },
            {
                "panel": "B",
                "quantity": "median_first_passage_age",
                "unit": "day",
                "description": first_passage_description,
            },
            {
                "panel": "C",
                "quantity": "local_first_exit",
                "unit": "raw count or grid relative weight",
                "description": kde_description,
            },
            {
                "panel": "D",
                "quantity": "residence_time_per_valid_member",
                "unit": "hour/member",
                "description": residence_description,
            },
            {
                "panel": "E",
                "quantity": "local_boundary_arclength",
                "unit": "raw count and fraction",
                "description": boundary_description,
            },
            {
                "panel": "F",
                "quantity": "outcomes_and_qc",
                "unit": "raw count and total-member fraction",
                "description": outcome_description,
            },
        ],
        "kde_status": layer.status.value,
        "primary_kde_bandwidth_m": float(kde.primary_bandwidth_m),
        "valid_member_denominator": pathway.valid_member_denominator,
        "references": [dict(item) for item in _CITATIONS],
        "transfer_limit": transfer_limit,
        "report_spec_canonical_sha256": report_spec.canonical_sha256,
        "aggregate_spec_canonical_sha256": payload.aggregate_spec.canonical_sha256,
    }


def _release_readme(
    *,
    payload: AggregateReleasePayload,
    report_spec: ReportSpec,
    materials: tuple[dict[str, object], ...],
    site_ids: tuple[str, ...],
) -> bytes:
    """建立成果包根目錄的繁體中文 README。"""

    material_lines = "\n".join(
        f"- `{item['material_id']}`：settling_velocity_mps={item['settling_velocity_mps']} m/s"
        for item in materials
    )
    reference_lines = "\n".join(
        f"- {item['authors']} ({item['year']}), DOI `{item['doi']}`：{item['reason']}。"
        for item in _CITATIONS
    )
    text = f"""# source-pathway-v1

本成果包針對向下沉降粒子（`settling_velocity_mps < 0`），以已驗證 aggregate release
和相同 `ReportSpec` 建立每站六面板圖。結果是指定受體、到達時刻、材質與流場條件下
的條件式來源足跡／相對來源權重，不是絕對來源機率、沉積質量、沉積濃度或因果歸因。

## 內容

- `sites/<site>/figure.png|svg|pdf`：300 dpi PNG、SVG 與 PDF 六面板圖。
- `sites/<site>/grid.parquet`：公尺制格網、訪格／first-passage／停留／KDE／底床接觸／失敗診斷。
- `sites/<site>/boundary.parquet`：local／outer boundary 弧長 bins、raw count、分母比例與旅行年齡中位數。
- `sites/<site>/outcomes.parquet`：完整停止分類與 total／valid 分母。
- `sites/<site>/caption.json`：圖面量綱、分母、低樣本政策、BED_DEPOSITED 與文獻依據。
- `manifest.json`：schema 1.0.0、來源 hash、材質／垂向／到達識別與每個檔案的 size／SHA-256。

有效成員的訪格比例採每粒子每格只計一次；停留時間保留重複迴游造成的秒數，兩者
分圖。沒有樣本的格子保留 null／空白，低樣本格以斜線標示；KDE 未達門檻時保留
local first-exit raw count，並在圖面標出 zero 或 below-minimum。紅色輪廓是
`bed_first_contact_count` 的底床邊界接觸診斷，不能解讀成沉積質量或沉積濃度。
C 面板的逆向 `local_first_exit` 在條件式解讀下對應正向潛在移入入口；E 面板呈現
潛在移入邊界區段。兩者都是邊界事件診斷，不能直接稱為確定來源。

多情境資料依本次已執行的有效成員數加權 pooled；manifest 列出每個 scenario
stratum 的 material、沉降速度、vertical／arrival 與 total member。aggregate release
沒有保存 scenario-level valid denominator，因此該欄保留 null，圖面只宣稱「條件於
本次情境設計與有效成員」，不推論材料自然比例，也不宣稱 design-balanced。

## 本次材料

{material_lines}

## 站點

{", ".join(f"`{site_id}`" for site_id in site_ids)}

## 方法依據與移植限制

{reference_lines}

上述文獻的 surface／forward／observed-source assumptions 不直接移植到本專案的
向下沉降受體條件式逆推；本成果沒有來源質量先驗、表層釋放通量或再懸浮模型，
因此不輸出 concentration、mass flux 或來源國貢獻圖。
"""
    return text.encode("utf-8")


def _manifest_document(
    *,
    payload: AggregateReleasePayload,
    report_spec: ReportSpec,
    aggregate_release_path: Path,
    aggregate_manifest_sha256: str,
    materials: tuple[dict[str, object], ...],
    vertical_ids: tuple[str, ...],
    arrival_ids: tuple[str, ...],
    scenario_strata: tuple[dict[str, object], ...],
    site_summaries: tuple[dict[str, object], ...],
    files: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """建立不含絕對位置且可自行驗證的 root manifest。"""

    document: dict[str, object] = {
        "schema_version": SOURCE_PATHWAY_RELEASE_SCHEMA_VERSION,
        "release_kind": "source_pathway_v1",
        "run_kind": payload.run_kind,
        "experiment_case_id": payload.experiment_case_id,
        "run_id": payload.run_id,
        "members_per_scenario": payload.members_per_scenario,
        "evidence_class": _EVIDENCE_CLASS_BY_RUN_KIND[payload.run_kind],
        "semantics": {
            "particle_scope": "settling_velocity_mps < 0",
            "source_measure": "conditional_source_footprint_or_relative_source_weight",
            "absolute_probability": False,
            "bed_contact_is_diagnostic": True,
            "bed_contact_quantity": "bed_first_contact_count",
            "bed_deposited_status": ParticleStatus.DEPOSITED.value,
        },
        "provenance": {
            "aggregate_release_name": aggregate_release_path.name,
            "aggregate_manifest_sha256": aggregate_manifest_sha256,
            "aggregate_spec_source_sha256": payload.aggregate_spec.source_sha256,
            "aggregate_spec_canonical_sha256": payload.aggregate_spec.canonical_sha256,
            "report_spec_source_sha256": report_spec.source_sha256,
            "report_spec_canonical_sha256": report_spec.canonical_sha256,
        },
        "materials": [dict(item) for item in materials],
        "settling_velocity_range_mps": [
            float(min(item["settling_velocity_mps"] for item in materials)),
            float(max(item["settling_velocity_mps"] for item in materials)),
        ],
        "vertical_ids": list(vertical_ids),
        "arrival_ids": list(arrival_ids),
        "scenario_strata": [dict(item) for item in scenario_strata],
        "pooling": {
            "weighting": "executed_member_weighted",
            "claim": "條件於本次情境設計與有效成員",
            "design_balanced_claim": False,
            "natural_material_fraction_inferred": False,
            "valid_denominator_scope": (
                "aggregate release 保存 site/receptor pooled 分母；未推導 scenario-level valid denominator"
            ),
        },
        "sites": [dict(item) for item in site_summaries],
        "files": {key: dict(value) for key, value in sorted(files.items())},
    }
    snapshot = _json_safe(document, label="manifest")
    if not isinstance(snapshot, dict) or _contains_absolute_path(snapshot):
        raise ValueError("source pathway manifest 不得含絕對路徑")
    return snapshot


def _build_release_bytes(
    *,
    aggregate_release_path: Path,
    payload: AggregateReleasePayload,
    report_spec: ReportSpec,
    statistics: ReportStatistics,
) -> dict[str, bytes]:
    """建立成果包所有檔案 bytes；manifest 最後依實際 bytes 補入。"""

    materials = _downward_material_snapshot(payload)
    site_ids = tuple(payload.aggregate_spec.site_grids)
    vertical_ids = tuple(sorted({stratum.vertical_id for stratum in payload.scenario_strata}))
    arrival_ids = tuple(sorted({stratum.arrival_time_id for stratum in payload.scenario_strata}))
    scenario_strata = tuple(
        {
            "scenario_id": stratum.scenario_id,
            "study_site_id": stratum.study_site_id,
            "receptor_id": stratum.receptor_id,
            "material_id": stratum.material_id,
            "settling_velocity_mps": float(stratum.settling_velocity_mps),
            "vertical_id": stratum.vertical_id,
            "arrival_time_id": stratum.arrival_time_id,
            "total_member_denominator": payload.members_per_scenario,
            # AggregateReleasePayload 的 valid member 只保存 site 與 receptor 層級，
            # 並沒有 scenario×arrival×material 的逐列有效數；這裡保留 null，避免
            # 把 pooled 分母錯標成單一 stratum 的有效成員數。
            "valid_member_denominator": None,
            "valid_member_denominator_scope": "site/receptor pooled in aggregate release",
        }
        for stratum in payload.scenario_strata
    )
    result: dict[str, bytes] = {
        _README_FILE_NAME: _release_readme(
            payload=payload,
            report_spec=report_spec,
            materials=materials,
            site_ids=site_ids,
        )
    }
    site_summaries: list[dict[str, object]] = []
    for site_id in site_ids:
        figure_bytes = _draw_site_figure(
            payload=payload,
            report_spec=report_spec,
            statistics=statistics,
            site_id=site_id,
        )
        for file_format, raw in figure_bytes.items():
            result[_site_path(site_id, filename=f"figure.{file_format}")] = raw
        caption = _caption_document(
            payload=payload,
            report_spec=report_spec,
            site_id=site_id,
            statistics=statistics,
        )
        result[_site_path(site_id, filename="caption.json")] = _canonical_json_bytes(caption)
        grid_table = _build_grid_table(payload=payload, site_id=site_id, statistics=statistics)
        boundary_table = _build_boundary_table(
            payload=payload,
            site_id=site_id,
            valid_denominator=statistics.outcome_statistics.valid_member_denominator_by_site[site_id],
        )
        outcome_table = _build_outcome_table(statistics=statistics, site_id=site_id)
        result[_site_path(site_id, filename="grid.parquet")] = _parquet_bytes(grid_table)
        result[_site_path(site_id, filename="boundary.parquet")] = _parquet_bytes(boundary_table)
        result[_site_path(site_id, filename="outcomes.parquet")] = _parquet_bytes(outcome_table)
        pathway = statistics.pathway_statistics_by_site[site_id]
        kde = statistics.kde_statistics_by_site[site_id]
        layer = kde.layers[kde.primary_bandwidth_m]
        outcomes = statistics.outcome_statistics
        site_summaries.append(
            {
                "site_id": site_id,
                "total_member_denominator": outcomes.total_member_denominator_by_site[site_id],
                "valid_member_denominator": pathway.valid_member_denominator,
                "grid_cell_count": int(pathway.visit_numerator.size),
                "local_first_exit_raw_count": int(
                    sum(
                        int(item)
                        for item in payload.event_aggregate.site_grid_counts[
                            site_id
                        ].local_first_exit_count.flat
                    )
                ),
                "outer_first_exit_raw_count": int(
                    sum(
                        int(item)
                        for item in payload.event_aggregate.site_grid_counts[
                            site_id
                        ].outer_first_exit_count.flat
                    )
                ),
                "primary_kde_status": layer.status.value,
                "primary_kde_raw_point_count": layer.raw_point_count,
            }
        )
    files = {
        relative_path: {"relative_path": relative_path, "size_bytes": len(raw), "sha256": _sha256_bytes(raw)}
        for relative_path, raw in result.items()
    }
    manifest = _manifest_document(
        payload=payload,
        report_spec=report_spec,
        aggregate_release_path=aggregate_release_path,
        aggregate_manifest_sha256=_aggregate_manifest_sha256(aggregate_release_path),
        materials=materials,
        vertical_ids=vertical_ids,
        arrival_ids=arrival_ids,
        scenario_strata=scenario_strata,
        site_summaries=tuple(site_summaries),
        files=files,
    )
    result[_MANIFEST_FILE_NAME] = _canonical_json_bytes(manifest)
    return result


def _validate_table(path: Path, required_columns: frozenset[str], *, label: str) -> None:
    """讀取 Parquet 並確認必要欄位存在；資料型別交由 Arrow 保留。"""

    try:
        table = pq.read_table(path)
    except Exception as error:
        raise ValueError(f"{label} Parquet 無法讀取") from error
    if not required_columns.issubset(set(table.column_names)):
        raise ValueError(f"{label} 缺少必要欄位")


def _iter_nodes(root: Path) -> tuple[set[str], set[str]]:
    """以 no-follow lstat 遞迴取得普通檔案／目錄相對集合。"""

    files: set[str] = set()
    directories: set[str] = set()
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        if relative_current != Path("."):
            metadata = os.lstat(current_path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("release topology 含非普通目錄")
            directories.add(relative_current.as_posix())
        for name in dirnames:
            candidate = current_path / name
            metadata = os.lstat(candidate)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("release topology 含 symbolic link 或非普通目錄")
        for name in filenames:
            candidate = current_path / name
            metadata = os.lstat(candidate)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("release topology 含 symbolic link 或非普通檔案")
            files.add(candidate.relative_to(root).as_posix())
    return files, directories


def _validate_release_root(path: Path, *, require_final_name: bool) -> dict[str, object]:
    """執行成果包 topology、manifest、checksum 與 sidecar schema 驗證。"""

    if not path.is_absolute():
        raise ValueError("release root 必須是絕對位置")
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("release root 不是普通目錄")
    if require_final_name and not path.name.endswith(SOURCE_PATHWAY_FINAL_SUFFIX):
        raise ValueError("release final basename 不符合 source-pathway-v1")
    files, directories = _iter_nodes(path)
    manifest_relative = _MANIFEST_FILE_NAME
    if manifest_relative not in files:
        raise ValueError("manifest 缺失")
    manifest_raw = (path / manifest_relative).read_bytes()
    document = _safe_json_load(manifest_raw, label="manifest")
    if not isinstance(document, dict) or set(document) != _EXPECTED_MANIFEST_KEYS:
        raise ValueError("manifest schema mismatch")
    if _canonical_json_bytes(document) != manifest_raw:
        raise ValueError("manifest 非 canonical JSON")
    if _contains_absolute_path(document):
        raise ValueError("manifest 含絕對路徑")
    if (
        document.get("schema_version") != SOURCE_PATHWAY_RELEASE_SCHEMA_VERSION
        or document.get("release_kind") != "source_pathway_v1"
    ):
        raise ValueError("manifest schema/version 不符")
    run_kind = document.get("run_kind")
    evidence_class = document.get("evidence_class")
    if run_kind not in _EVIDENCE_CLASS_BY_RUN_KIND or evidence_class != _EVIDENCE_CLASS_BY_RUN_KIND.get(
        run_kind
    ):
        raise ValueError("manifest run/evidence class 不符")
    _require_safe_component(document.get("run_id"), label="manifest.run_id")
    _require_safe_component(document.get("experiment_case_id"), label="manifest.experiment_case_id")
    if type(document.get("members_per_scenario")) is not int or document["members_per_scenario"] <= 0:
        raise ValueError("manifest members_per_scenario 不符")
    semantics = document.get("semantics")
    if not isinstance(semantics, dict) or set(semantics) != {
        "particle_scope",
        "source_measure",
        "absolute_probability",
        "bed_contact_is_diagnostic",
        "bed_contact_quantity",
        "bed_deposited_status",
    }:
        raise ValueError("manifest semantics schema mismatch")
    if (
        semantics.get("particle_scope") != "settling_velocity_mps < 0"
        or semantics.get("source_measure") != "conditional_source_footprint_or_relative_source_weight"
        or semantics.get("absolute_probability") is not False
        or semantics.get("bed_contact_is_diagnostic") is not True
        or semantics.get("bed_contact_quantity") != "bed_first_contact_count"
        or semantics.get("bed_deposited_status") != ParticleStatus.DEPOSITED.value
    ):
        raise ValueError("manifest semantics value mismatch")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "aggregate_release_name",
        "aggregate_manifest_sha256",
        "aggregate_spec_source_sha256",
        "aggregate_spec_canonical_sha256",
        "report_spec_source_sha256",
        "report_spec_canonical_sha256",
    }:
        raise ValueError("manifest provenance schema mismatch")
    for name in (
        "aggregate_manifest_sha256",
        "aggregate_spec_source_sha256",
        "aggregate_spec_canonical_sha256",
        "report_spec_source_sha256",
        "report_spec_canonical_sha256",
    ):
        digest = provenance.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("manifest provenance hash mismatch")
    aggregate_release_name = provenance.get("aggregate_release_name")
    if (
        not isinstance(aggregate_release_name, str)
        or not aggregate_release_name
        or Path(aggregate_release_name).name != aggregate_release_name
        or aggregate_release_name in {".", ".."}
    ):
        raise ValueError("manifest aggregate release name 不符")
    materials = document.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("manifest materials 必須為非空陣列")
    material_velocities: dict[str, float] = {}
    for material in materials:
        if not isinstance(material, dict) or set(material) != {"material_id", "settling_velocity_mps"}:
            raise ValueError("manifest material schema mismatch")
        material_id = _require_safe_component(material.get("material_id"), label="manifest.material_id")
        if material_id in material_velocities:
            raise ValueError("manifest material id 不得重複")
        velocity = material.get("settling_velocity_mps")
        if type(velocity) not in {int, float} or not math.isfinite(float(velocity)) or float(velocity) >= 0.0:
            raise ValueError("manifest material 必須是向下沉降")
        material_velocities[material_id] = float(velocity)
    velocity_range = document.get("settling_velocity_range_mps")
    if (
        not isinstance(velocity_range, list)
        or len(velocity_range) != 2
        or any(
            type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) >= 0.0
            for value in velocity_range
        )
        or velocity_range != [min(material_velocities.values()), max(material_velocities.values())]
    ):
        raise ValueError("manifest settling velocity range 不符")
    vertical_ids = document.get("vertical_ids")
    arrival_ids = document.get("arrival_ids")
    for values, label in ((vertical_ids, "vertical_ids"), (arrival_ids, "arrival_ids")):
        if not isinstance(values, list) or not values:
            raise ValueError(f"manifest {label} 必須為非空陣列")
        safe_values = [_require_safe_component(value, label=f"manifest.{label}") for value in values]
        if len(set(safe_values)) != len(safe_values):
            raise ValueError(f"manifest {label} 不得重複")
        if safe_values != values:
            raise ValueError(f"manifest {label} 型別不符")
    vertical_id_set = set(vertical_ids)
    arrival_id_set = set(arrival_ids)
    scenario_strata = document.get("scenario_strata")
    if not isinstance(scenario_strata, list) or not scenario_strata:
        raise ValueError("manifest scenario_strata 必須為非空陣列")
    required_stratum_keys = {
        "scenario_id",
        "study_site_id",
        "receptor_id",
        "material_id",
        "settling_velocity_mps",
        "vertical_id",
        "arrival_time_id",
        "total_member_denominator",
        "valid_member_denominator",
        "valid_member_denominator_scope",
    }
    scenario_ids: set[str] = set()
    scenario_site_ids: set[str] = set()
    scenario_count_by_site: dict[str, int] = {}
    for stratum in scenario_strata:
        if not isinstance(stratum, dict) or set(stratum) != required_stratum_keys:
            raise ValueError("manifest scenario stratum schema mismatch")
        for key in (
            "scenario_id",
            "study_site_id",
            "receptor_id",
            "material_id",
            "vertical_id",
            "arrival_time_id",
        ):
            _require_safe_component(stratum[key], label=f"scenario_stratum.{key}")
        scenario_id = stratum["scenario_id"]
        if scenario_id in scenario_ids:
            raise ValueError("manifest scenario_id 不得重複")
        scenario_ids.add(scenario_id)
        study_site_id = stratum["study_site_id"]
        scenario_site_ids.add(study_site_id)
        scenario_count_by_site[study_site_id] = scenario_count_by_site.get(study_site_id, 0) + 1
        material_id = stratum["material_id"]
        if material_id not in material_velocities:
            raise ValueError("manifest scenario material_id 未登錄")
        if stratum["vertical_id"] not in vertical_id_set:
            raise ValueError("manifest scenario vertical_id 未登錄")
        if stratum["arrival_time_id"] not in arrival_id_set:
            raise ValueError("manifest scenario arrival_time_id 未登錄")
        velocity = stratum["settling_velocity_mps"]
        if (
            type(velocity) not in {int, float}
            or not math.isfinite(float(velocity))
            or float(velocity) >= 0.0
            or float(velocity) != material_velocities[material_id]
        ):
            raise ValueError("manifest scenario settling velocity 不符")
        if (
            type(stratum["total_member_denominator"]) is not int
            or stratum["total_member_denominator"] <= 0
            or stratum["total_member_denominator"] != document["members_per_scenario"]
        ):
            raise ValueError("manifest stratum total denominator 不符")
        if stratum["valid_member_denominator"] is not None and (
            type(stratum["valid_member_denominator"]) is not int
            or stratum["valid_member_denominator"] < 0
            or stratum["valid_member_denominator"] > stratum["total_member_denominator"]
        ):
            raise ValueError("manifest stratum valid denominator 不符")
        if stratum["valid_member_denominator_scope"] != "site/receptor pooled in aggregate release":
            raise ValueError("manifest stratum valid denominator scope 不符")
    pooling = document.get("pooling")
    if (
        not isinstance(pooling, dict)
        or set(pooling)
        != {
            "weighting",
            "claim",
            "design_balanced_claim",
            "natural_material_fraction_inferred",
            "valid_denominator_scope",
        }
        or pooling.get("weighting") != "executed_member_weighted"
        or pooling.get("claim") != "條件於本次情境設計與有效成員"
        or pooling.get("design_balanced_claim") is not False
        or pooling.get("natural_material_fraction_inferred") is not False
        or pooling.get("valid_denominator_scope")
        != "aggregate release 保存 site/receptor pooled 分母；未推導 scenario-level valid denominator"
    ):
        raise ValueError("manifest pooling semantics 不符")
    sites = document.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("manifest sites 必須為非空陣列")
    expected_site_summary_keys = {
        "site_id",
        "total_member_denominator",
        "valid_member_denominator",
        "grid_cell_count",
        "local_first_exit_raw_count",
        "outer_first_exit_raw_count",
        "primary_kde_status",
        "primary_kde_raw_point_count",
    }
    site_ids: list[str] = []
    site_summaries_by_id: dict[str, dict[str, object]] = {}
    for site in sites:
        if not isinstance(site, dict) or set(site) != expected_site_summary_keys:
            raise ValueError("site summary 不符")
        site_id = _require_safe_component(site.get("site_id"), label="site summary.site_id")
        site_ids.append(site_id)
        if site_id in site_summaries_by_id:
            raise ValueError("site summary 不得重複")
        site_summaries_by_id[site_id] = site
        total_denominator = site["total_member_denominator"]
        valid_denominator = site["valid_member_denominator"]
        if type(total_denominator) is not int or total_denominator <= 0:
            raise ValueError("site summary total denominator 不符")
        if (
            type(valid_denominator) is not int
            or valid_denominator < 0
            or valid_denominator > total_denominator
        ):
            raise ValueError("site summary valid denominator 不符")
        for key in (
            "grid_cell_count",
            "local_first_exit_raw_count",
            "outer_first_exit_raw_count",
            "primary_kde_raw_point_count",
        ):
            if type(site[key]) is not int or site[key] < 0:
                raise ValueError(f"site summary {key} 不符")
        if site["grid_cell_count"] <= 0:
            raise ValueError("site summary grid cell count 不符")
        if site["primary_kde_status"] not in _KDE_STATUS_VALUES:
            raise ValueError("site summary KDE status 不符")
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("site summary 不得重複")
    if set(scenario_site_ids) != set(site_ids) or any(
        site_id not in scenario_count_by_site for site_id in site_ids
    ):
        raise ValueError("scenario strata 與 site summary site set 不一致")
    for site_id, summary in site_summaries_by_id.items():
        expected_total = scenario_count_by_site[site_id] * document["members_per_scenario"]
        if summary["total_member_denominator"] != expected_total:
            raise ValueError("site summary total denominator 與 strata 不一致")
    expected_files = {_README_FILE_NAME, _MANIFEST_FILE_NAME}
    expected_dirs = {_SITE_DIR_NAME}
    for site_id in site_ids:
        expected_dirs.add(f"{_SITE_DIR_NAME}/{site_id}")
        expected_files.update(
            {
                _site_path(site_id, filename="figure.png"),
                _site_path(site_id, filename="figure.svg"),
                _site_path(site_id, filename="figure.pdf"),
                _site_path(site_id, filename="caption.json"),
                _site_path(site_id, filename="grid.parquet"),
                _site_path(site_id, filename="boundary.parquet"),
                _site_path(site_id, filename="outcomes.parquet"),
            }
        )
    if files != expected_files or directories != expected_dirs:
        raise ValueError("release topology inventory mismatch")
    file_records = document.get("files")
    if not isinstance(file_records, dict) or set(file_records) != expected_files - {_MANIFEST_FILE_NAME}:
        raise ValueError("manifest files inventory mismatch")
    for relative_path, record in file_records.items():
        if not isinstance(record, dict) or set(record) != {"relative_path", "size_bytes", "sha256"}:
            raise ValueError("manifest file record schema mismatch")
        if (
            record.get("relative_path") != relative_path
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("manifest file record value mismatch")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("manifest file hash mismatch")
        actual = path / relative_path
        raw = actual.read_bytes()
        if len(raw) != record["size_bytes"] or _sha256_bytes(raw) != digest:
            raise ValueError("file size or hash mismatch")
    for site_id in site_ids:
        caption_path = path / _site_path(site_id, filename="caption.json")
        caption = _safe_json_load(caption_path.read_bytes(), label="caption")
        if (
            not isinstance(caption, dict)
            or caption.get("schema_version") != "source_pathway_caption_v1"
            or caption.get("site_id") != site_id
        ):
            raise ValueError("caption schema mismatch")
        caption_strata = caption.get("scenario_strata")
        if not isinstance(caption_strata, list) or not caption_strata:
            raise ValueError("caption scenario strata schema mismatch")
        manifest_site_strata = {
            stratum["scenario_id"]: stratum
            for stratum in scenario_strata
            if stratum["study_site_id"] == site_id
        }
        if not manifest_site_strata or {
            stratum.get("scenario_id") for stratum in caption_strata if isinstance(stratum, dict)
        } != set(manifest_site_strata):
            raise ValueError("caption scenario strata 與 manifest 不一致")
        expected_caption_stratum_keys = {
            "scenario_id",
            "material_id",
            "settling_velocity_mps",
            "vertical_id",
            "arrival_time_id",
            "total_member_denominator",
            "valid_member_denominator",
            "valid_member_denominator_scope",
        }
        caption_velocities: list[float] = []
        for stratum in caption_strata:
            if not isinstance(stratum, dict) or set(stratum) != expected_caption_stratum_keys:
                raise ValueError("caption scenario stratum schema mismatch")
            for key in ("scenario_id", "material_id", "vertical_id", "arrival_time_id"):
                _require_safe_component(stratum[key], label=f"caption scenario_stratum.{key}")
            velocity = stratum["settling_velocity_mps"]
            if (
                type(velocity) not in {int, float}
                or not math.isfinite(float(velocity))
                or float(velocity) >= 0.0
            ):
                raise ValueError("caption scenario stratum 必須是向下沉降")
            caption_velocities.append(float(velocity))
            total_denominator = stratum["total_member_denominator"]
            if type(total_denominator) is not int or total_denominator <= 0:
                raise ValueError("caption scenario total denominator 不符")
            valid_denominator = stratum["valid_member_denominator"]
            if valid_denominator is not None and (
                type(valid_denominator) is not int or valid_denominator < 0
            ):
                raise ValueError("caption scenario valid denominator 不符")
            manifest_stratum = manifest_site_strata[stratum["scenario_id"]]
            if any(
                stratum[key] != manifest_stratum[key]
                for key in (
                    "scenario_id",
                    "material_id",
                    "settling_velocity_mps",
                    "vertical_id",
                    "arrival_time_id",
                    "total_member_denominator",
                    "valid_member_denominator",
                    "valid_member_denominator_scope",
                )
            ):
                raise ValueError("caption scenario stratum 與 manifest 欄位不一致")
        velocity_range = caption.get("settling_velocity_range_mps")
        if (
            not isinstance(velocity_range, list)
            or len(velocity_range) != 2
            or any(
                type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) >= 0.0
                for value in velocity_range
            )
            or velocity_range != [min(caption_velocities), max(caption_velocities)]
        ):
            raise ValueError("caption settling velocity range 不符")
        pooling_semantics = caption.get("pooling_semantics")
        if (
            not isinstance(pooling_semantics, dict)
            or set(pooling_semantics)
            != {
                "weighting",
                "claim",
                "design_balanced_claim",
                "natural_material_fraction_inferred",
            }
            or pooling_semantics.get("weighting") != "executed_member_weighted"
            or pooling_semantics.get("claim") != "條件於本次情境設計與有效成員"
            or pooling_semantics.get("design_balanced_claim") is not False
            or pooling_semantics.get("natural_material_fraction_inferred") is not False
        ):
            raise ValueError("caption pooling semantics 不符")
        if (
            caption.get("valid_member_denominator")
            != site_summaries_by_id[site_id]["valid_member_denominator"]
        ):
            raise ValueError("caption valid denominator 與 site summary 不一致")
        _validate_table(path / _site_path(site_id, filename="grid.parquet"), _GRID_COLUMNS, label="grid")
        _validate_table(
            path / _site_path(site_id, filename="boundary.parquet"), _BOUNDARY_COLUMNS, label="boundary"
        )
        _validate_table(
            path / _site_path(site_id, filename="outcomes.parquet"), _OUTCOME_COLUMNS, label="outcomes"
        )
    return {
        "run_id": document["run_id"],
        "run_kind": document["run_kind"],
        "experiment_case_id": document["experiment_case_id"],
        "members_per_scenario": document["members_per_scenario"],
        "site_count": len(site_ids),
        "file_count": len(files),
    }


def validate_source_pathway_release(path: str | Path) -> dict[str, object]:
    """回傳 source-pathway-v1 的固定 JSON-safe 驗證摘要。

    ``valid=True`` 只代表成果包拓撲、manifest、checksum 與三張 sidecar Parquet 的
    工程契約成立；validator 不讀取 trajectory、不驗證 OCM／NWW3 科學內容，也不把
    synthetic fixture 提升為正式成果。失敗回傳固定 error token，不包含絕對路徑。
    """

    try:
        report = _validate_release_root(Path(path), require_final_name=True)
    except FileNotFoundError:
        return {"valid": False, "errors": ["root_missing"], "summary": {}}
    except FileExistsError:
        return {"valid": False, "errors": ["topology"], "summary": {}}
    except Exception:
        return {"valid": False, "errors": ["release_invalid"], "summary": {}}
    return {"valid": True, "errors": [], "summary": report}


def read_source_pathway_release(path: str | Path) -> Mapping[str, object]:
    """嚴格讀取 final release manifest，成功後回傳唯讀 mapping snapshot。"""

    root = Path(path)
    _validate_release_root(root, require_final_name=True)
    document = _safe_json_load((root / _MANIFEST_FILE_NAME).read_bytes(), label="manifest")
    if not isinstance(document, dict):
        raise ValueError("source pathway manifest 讀取失敗")
    return MappingProxyType(dict(document))


def build_source_pathway_release(
    *,
    aggregate_release: str | Path,
    report_spec: str | Path,
    destination: str | Path,
    mplconfigdir: str | Path | None = None,
) -> Path:
    """建立向下沉降粒子 source-pathway-v1 成果包並以 atomic rename 發布。

    Args:
        aggregate_release: 已驗證、final basename 的 aggregate release 根目錄。
        report_spec: 與 aggregate spec canonical hash 綁定的 ReportSpec JSON。
        destination: 新的絕對 final 目錄，basename 必須以 ``.source-pathway-v1`` 結尾。
        mplconfigdir: 可選的既有、可寫、非 symbolic link Matplotlib cache 目錄。
            提供時只在 renderer context 內暫時設定 ``MPLCONFIGDIR``；省略時沿用
            caller 已明示的環境變數，仍由 ``report_render_style_context`` 驗證。

    Returns:
        已發布的 destination ``Path``。

    Raises:
        ValueError: 來源、ReportSpec、沉降速度、輸出拓撲或圖面／sidecar 建置不符。
        FileExistsError: destination 已存在，writer 不覆寫既有成果。
    """

    aggregate_path = Path(aggregate_release)
    report_path = Path(report_spec)
    destination_path = Path(destination)
    if not destination_path.is_absolute() or not destination_path.name.endswith(SOURCE_PATHWAY_FINAL_SUFFIX):
        raise ValueError("destination 必須是絕對路徑且 basename 以 .source-pathway-v1 結尾")
    _require_regular_directory(destination_path.parent, label="destination parent")
    _require_absent(destination_path, label="destination")
    payload = read_aggregate_release(aggregate_path)
    loaded_spec = load_report_spec(report_path)
    validate_report_spec_against_aggregate_spec(loaded_spec, payload.aggregate_spec)
    _downward_material_snapshot(payload)
    statistics = build_report_statistics(payload, report_spec=loaded_spec)
    partial = destination_path.parent / f".{destination_path.name}.partial-{uuid4().hex}"
    partial.mkdir(mode=0o700, exist_ok=False)
    _require_regular_directory(partial, label="partial")
    try:

        @contextmanager
        def _mplconfigdir_context() -> object:
            """只在既有 style context 期間暫時套用 caller 指定的 cache 目錄。"""

            if mplconfigdir is None:
                yield
                return
            raw = os.fspath(mplconfigdir)
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError("mplconfigdir 必須是絕對路徑")
            previous = os.environ.get("MPLCONFIGDIR")
            os.environ["MPLCONFIGDIR"] = raw
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("MPLCONFIGDIR", None)
                else:
                    os.environ["MPLCONFIGDIR"] = previous

        with _mplconfigdir_context(), report_render_style_context(loaded_spec):
            files = _build_release_bytes(
                aggregate_release_path=aggregate_path,
                payload=payload,
                report_spec=loaded_spec,
                statistics=statistics,
            )
        # 先寫所有圖、sidecar 與 README，最後才寫 manifest；即使 operator 在
        # sibling partial 目錄觀察到中間狀態，也不會先看到宣稱完整 inventory 的索引。
        for relative_path, raw in sorted(files.items()):
            if relative_path == _MANIFEST_FILE_NAME:
                continue
            target = partial / relative_path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _require_regular_directory(target.parent, label="product parent")
            _write_exclusive(target, raw, label=relative_path)
        manifest_target = partial / _MANIFEST_FILE_NAME
        _write_exclusive(manifest_target, files[_MANIFEST_FILE_NAME], label=_MANIFEST_FILE_NAME)
        _validate_release_root(partial, require_final_name=False)
        os.replace(partial, destination_path)
        try:
            descriptor = os.open(destination_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise RuntimeError("source pathway 已發布但 parent durability 未確認") from error
    except Exception:
        if partial.exists():
            for child in sorted(partial.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                except OSError:
                    pass
            with suppress(OSError):
                partial.rmdir()
        raise
    validation = validate_source_pathway_release(destination_path)
    if validation.get("valid") is not True:
        raise RuntimeError("source pathway release 驗證失敗")
    return destination_path
