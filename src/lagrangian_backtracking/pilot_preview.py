"""由已完成且來源綁定的精確先導產生獨立工程預覽，不建立正式 report-v1。

只讀既有 trajectory schema 2／3 的軌跡、環境、速度與事件，不開啟海流／波浪陣列、不重新積分。全部成員
皆進入停止統計及 CSV；曲線太多時依明示的穩定成員順序限量並揭露分母。公尺制位置、
UTC 奈秒及回溯秒數原樣保留，缺環境或診斷不補零。來源與產品的 SHA-256 可供重建核對。
合成測試只驗工程，預覽不是收斂、觀測驗證、絕對來源機率或來源歸因證據。
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import runtime
from .models import ParticleStatus
from .outputs import TRAJECTORY_SHARD_SCHEMA_VERSION, sha256_file
from .report_release import (
    _atomic_exclusive_directory_rename,
    _fsync_directory,
    _write_exclusive_durable_bytes,
)
from .report_style import validate_mplconfigdir
from .run_control import load_run_plan, load_run_progress
from .run_validation import iter_complete_run_trajectory_shards
from .scenarios import derive_member_seed, stable_identifier

PILOT_PREVIEW_SCHEMA_VERSION = "1.0.0"
"""獨立先導預覽的版本，不沿用正式報告固定圖表清單。"""

# 預覽只接受已定義且有環境欄位的 trajectory schema 2 或目前 writer 發布的 schema 3。
# schema 1 沒有環境證據，不能在這個需要輸出環境／速度 CSV 的 consumer 中被假裝成完整資料。
_PREVIEW_TRAJECTORY_SCHEMA_VERSIONS = frozenset({"2.0.0", TRAJECTORY_SHARD_SCHEMA_VERSION})

_TERMINAL_STATUSES = tuple(item.value for item in ParticleStatus if item is not ParticleStatus.ACTIVE)
_VERTICAL_PRIORITY = ("near_bed", "mid_lower_water_column", "mid_upper_water_column", "upper_water_column")
_LEVEL_LABELS = {
    "near_bed": ("近海床", "Near bed"),
    "mid_lower_water_column": ("中下水層", "Lower-mid"),
    "mid_upper_water_column": ("中上水層", "Upper-mid"),
    "upper_water_column": ("上水層", "Upper water"),
}
_STOP_LABELS_ZH = {
    "flow_domain_open_exit": "流場域開放邊界離域",
    "coast_contact": "接觸海岸",
    "surface_regime_exit": "離開表面適用範圍",
    "deposited": "沉積",
    "forcing_start": "到達驅動資料起點",
    "data_gap": "資料缺口",
    "max_age": "達回溯時間上限",
    "numerical_failure": "數值失敗",
}
"""顯示名稱只解釋既有垂向／停止分類，不合併分母，也不推定特定失敗原因。"""
_DIAGNOSTIC_TEXT = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_LIMITS = {"max_particles": 10_000, "max_observations": 2_000_000, "max_curves_per_vertical": 100}
"""硬上限防止將先導工具當完整批次載入器；預設容量另見公開建立函式。"""
_METADATA_BYTE_LIMIT = 8 * 1024 * 1024
"""早期拒絕只讀小型 JSON，單檔至多 8 MiB；此檢查不能當成來源已驗收。"""
_DIAGNOSTIC_INTS = (
    "diagnostic_version",
    "qc_flags",
    "step_count",
    "minimum_clamp_count",
    "maximum_step_count",
    "maximum_minimum_clamps",
    "sample_time_utc_ns",
)
_DIAGNOSTIC_FLOATS = (
    "dt_min_seconds",
    "dt_max_seconds",
    "attempted_dt_seconds",
    "sample_x_m",
    "sample_y_m",
    "sample_z_m",
    "sample_eta_m",
    "sample_bed_z_m",
)
_DIAGNOSTIC_AVAILABILITY = (
    "attempted_dt_available",
    "qc_available",
    "sampling_context_available",
    "sample_eta_available",
    "sample_bed_available",
)
"""失敗事件版本 1 的實際欄名；取樣位置不等於粒子的最後位置，時間保持整數奈秒。"""


class PilotPreviewError(ValueError):
    """不含私有路徑的預覽拒絕或發布錯誤；不得因此刪除來源或已發布成果。"""


def _plain(value: Any) -> Any:
    """將已驗證的唯讀對照表轉成 JSON 容器；不加入 Path 或猜測缺值。"""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _safe_path(value: str | Path) -> Path:
    """拒絕路徑各層符號連結及上移；僅接受系統固定 /tmp、/var 別名。

    不建立任何目錄。輸出父層必須由操作端預先準備，避免工具擴大寫入範圍。
    """

    path = Path(os.path.abspath(value))
    if ".." in Path(value).parts:
        raise PilotPreviewError("preview 路徑不得含上移片段")
    for part in (*reversed(path.parents), path):
        if part.is_symlink() and (
            str(part) not in {"/tmp", "/var"} or str(part.resolve()) != "/private" + str(part)
        ):
            raise PilotPreviewError("preview 路徑不得經由符號連結")
    return path.resolve()


# 網路檔案系統（NFS）不一定提供 Linux ``renameat2(RENAME_NOREPLACE)`` 或 Darwin
# ``renameatx_np(RENAME_EXCL)``；因此預覽另提供明示啟用的完成標記協定。這不是把
# 逐檔搬移誤稱為整體原子操作，而是以同一父目錄的 cooperative flock、完整 staging
# 自我驗證及最後排他建立的完成標記，定義下游 reader 可以接受的交付邊界。
# 未通過已驗證的 SERVER 儲存閘門時，這條路徑永遠不會被隱含啟用。
_NFS_MARKER_PROTOCOL = "nfs_completion_marker_v1"
_COMPLETION_MARKER_NAME = ".complete"
_STORAGE_GATE_SCHEMA_VERSION = "1.0.0"
_STORAGE_ROOT_LABELS = frozenset(
    {
        "result_nfs_root",
        "execution_package_root",
        "output_root",
        "scratch_root",
        "checkpoint_root",
        "uv_cache_root",
        "mpl_cache_root",
        "xdg_cache_root",
        "tmp_root",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _finite(value: Any) -> float | None:
    """僅將已有有限數值轉為 Python float；缺值與非有限值維持不可用，不補零。"""

    if value is None or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _small_json(path: Path) -> dict:
    """僅讀普通小型 JSON；大小超限或連結在任何軌跡讀取前拒絕。"""

    path = _safe_path(path)
    if not path.is_file() or path.stat().st_size > _METADATA_BYTE_LIMIT:
        raise PilotPreviewError("preview 來源中繼文件不是普通小型檔案")
    with path.open("rb") as stream:
        raw = stream.read(_METADATA_BYTE_LIMIT + 1)
    if len(raw) > _METADATA_BYTE_LIMIT:
        raise PilotPreviewError("preview 來源中繼文件超過容量上限")
    return json.loads(raw)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """以預覽共用的固定 JSON 排版產生可驗證位元組。

    完成標記與 manifest 是跨主機傳遞的稽核資料；固定 UTF-8、排序欄位及換行，讓
    reader 能辨識內容遭截斷或改寫，而不把「能被 JSON parser 讀取」誤當成完整發布。
    此函式不放入動態絕對路徑，避免成果把 SERVER 私有位置帶出。
    """

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _verified_storage_gate_evidence(path: str | Path) -> dict[str, str]:
    """驗證已保存的 SERVER 儲存閘門，回傳完成標記所需的安全摘要。

    完成標記只可在操作端明示提供儲存閘門證據時啟用。此驗證要求九個結果／執行根目錄
    都是已通過的 NFS、共享同一 source token，且同主機檔案鎖（``flock``）測試明示
    PASS；缺少任一項便停止，不以本機可寫性猜測替代。回傳只含閘門版本、證據檔 SHA-256
    與共同 token，不保存證據檔的絕對路徑。
    """

    evidence = _safe_path(path)
    if not evidence.is_file() or evidence.stat().st_size > _METADATA_BYTE_LIMIT:
        raise PilotPreviewError("preview storage gate evidence 不是普通小型檔案")
    try:
        document = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PilotPreviewError("preview storage gate evidence 無法解析") from None
    if not isinstance(document, dict):
        raise PilotPreviewError("preview storage gate evidence root 無效")
    if document.get("schema_version") != _STORAGE_GATE_SCHEMA_VERSION:
        raise PilotPreviewError("preview storage gate schema 不符")
    if document.get("gate_status") != "PASS" or document.get("issues") != []:
        raise PilotPreviewError("preview storage gate 未通過")
    probes = document.get("probes")
    if not isinstance(probes, list):
        raise PilotPreviewError("preview storage gate 缺少 probes")
    probe_status = {
        row.get("label"): row.get("gate_status")
        for row in probes
        if isinstance(row, dict)
    }
    if probe_status.get("write_probe") != "PASS":
        raise PilotPreviewError("preview storage gate write_probe 未通過")
    if probe_status.get("same_host_flock_probe") != "PASS":
        raise PilotPreviewError("preview storage gate same_host_flock_probe 未通過")
    roots = document.get("roots")
    if not isinstance(roots, list) or len(roots) != len(_STORAGE_ROOT_LABELS):
        raise PilotPreviewError("preview storage gate roots 不完整")
    root_labels = {row.get("label") for row in roots if isinstance(row, dict)}
    if root_labels != _STORAGE_ROOT_LABELS:
        raise PilotPreviewError("preview storage gate roots label 不符")
    minimum_free = document.get("minimum_free_bytes")
    if type(minimum_free) is not int or minimum_free <= 0:
        raise PilotPreviewError("preview storage gate minimum free bytes 無效")
    source_tokens: set[str] = set()
    for row in roots:
        if not isinstance(row, dict):
            raise PilotPreviewError("preview storage gate root record 無效")
        if row.get("gate_status") != "PASS" or str(row.get("fstype", "")).lower() not in {"nfs", "nfs4"}:
            raise PilotPreviewError("preview storage gate root 未通過 NFS 條件")
        token = row.get("source_token_hash")
        free_bytes = row.get("free_bytes")
        if not isinstance(token, str) or _HEX64.fullmatch(token) is None:
            raise PilotPreviewError("preview storage gate source token 無效")
        if type(free_bytes) is not int or free_bytes < minimum_free:
            raise PilotPreviewError("preview storage gate free bytes 不足")
        source_tokens.add(token)
    if len(source_tokens) != 1:
        raise PilotPreviewError("preview storage gate roots source token 不一致")
    return {
        "schema_version": _STORAGE_GATE_SCHEMA_VERSION,
        "sha256": sha256_file(evidence),
        "source_token_hash": next(iter(source_tokens)),
    }


def _verified_release_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """核對完成標記必須保存的基準 commit、Git tree、dirty 清單與 diff 摘要。

    這些值由執行包的明示 code provenance 產生，不能在預覽內從任意工作目錄猜 Git 狀態。
    ``base_tree`` 是 40 位 Git tree object，不是 deployment tree 的 SHA-256。dirty file
    只允許 repository-relative 名稱；完成標記因此可供稽核而不洩漏 SERVER 絕對路徑。
    """

    if not isinstance(value, Mapping):
        raise PilotPreviewError("preview marker 缺少 release provenance")
    base_commit = value.get("base_commit")
    base_tree = value.get("base_tree")
    diff_sha = value.get("diff_sha256")
    dirty_files = value.get("dirty_files")
    if not isinstance(base_commit, str) or _GIT_COMMIT.fullmatch(base_commit) is None:
        raise PilotPreviewError("preview marker base commit 無效")
    if not isinstance(base_tree, str) or _GIT_COMMIT.fullmatch(base_tree) is None:
        raise PilotPreviewError("preview marker base tree 無效")
    if not isinstance(diff_sha, str) or _HEX64.fullmatch(diff_sha) is None:
        raise PilotPreviewError("preview marker diff hash 無效")
    if not isinstance(dirty_files, (list, tuple)):
        raise PilotPreviewError("preview marker dirty files 無效")
    normalized_files = []
    for item in dirty_files:
        if (
            not isinstance(item, str)
            or not item
            or Path(item).is_absolute()
            or ".." in Path(item).parts
            or "\\" in item
        ):
            raise PilotPreviewError("preview marker dirty file 非 repository-relative")
        normalized_files.append(item)
    if normalized_files != sorted(set(normalized_files)):
        raise PilotPreviewError("preview marker dirty files 必須排序且不可重複")
    return {
        "base_commit": base_commit,
        "base_tree": base_tree,
        "dirty_files": normalized_files,
        "diff_sha256": diff_sha,
    }


def _regular_preview_file(path: Path) -> None:
    """要求節點是普通檔，拒絕完成標記／成果路徑的 symbolic link。"""

    try:
        node = path.lstat()
    except OSError:
        raise PilotPreviewError("preview 目錄缺少普通檔") from None
    if not stat.S_ISREG(node.st_mode):
        raise PilotPreviewError("preview 目錄含非普通檔節點")


def _validate_completion_marker(
    preview: Path,
    *,
    expected_artifact_kind: str = "pilot-preview-v1",
    storage_gate_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """在 reader 使用預覽任何資料前驗證完成標記及其 manifest binding。

    完成標記是 NFS 逐檔發布的唯一可讀宣告；它必須是 canonical JSON、和 manifest
    SHA-256 相符，並攜帶程式／dirty／儲存閘門指紋。若 caller 再提供閘門證據，會重新
    驗證其 PASS、同主機 ``flock`` 與共同 source token，防止只竄改標記文字就繞過環境閘門。
    """

    marker = _safe_path(preview / _COMPLETION_MARKER_NAME)
    _regular_preview_file(marker)
    if marker.stat().st_size > 64 * 1024:
        raise PilotPreviewError("preview completion marker 超過容量上限")
    try:
        raw = marker.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PilotPreviewError("preview completion marker 無法解析") from None
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "publish_protocol",
        "manifest_sha256",
        "base_commit",
        "base_tree",
        "dirty_files",
        "diff_sha256",
        "storage_gate_snapshot_sha256",
        "storage_root_source_token_hash",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise PilotPreviewError("preview completion marker 欄位不符")
    if raw != _canonical_json_bytes(document):
        raise PilotPreviewError("preview completion marker 非 canonical bytes")
    if document["schema_version"] != _STORAGE_GATE_SCHEMA_VERSION:
        raise PilotPreviewError("preview completion marker schema 不符")
    if (
        document["artifact_kind"] != expected_artifact_kind
        or document["publish_protocol"] != _NFS_MARKER_PROTOCOL
    ):
        raise PilotPreviewError("preview completion marker artifact 不符")
    if _HEX64.fullmatch(document["manifest_sha256"]) is None:
        raise PilotPreviewError("preview completion marker manifest hash 無效")
    if _HEX64.fullmatch(document["storage_gate_snapshot_sha256"]) is None:
        raise PilotPreviewError("preview completion marker storage hash 無效")
    if _HEX64.fullmatch(document["storage_root_source_token_hash"]) is None:
        raise PilotPreviewError("preview completion marker source token 無效")
    provenance = _verified_release_provenance(document)
    manifest = _safe_path(preview / "manifest.json")
    _regular_preview_file(manifest)
    if sha256_file(manifest) != document["manifest_sha256"]:
        raise PilotPreviewError("preview completion marker manifest binding 不符")
    manifest_document = _small_json(manifest)
    if (
        manifest_document.get("artifact_kind") != document["artifact_kind"]
        or manifest_document.get("publish_protocol") != document["publish_protocol"]
        or manifest_document.get("release_provenance")
        != {
            **provenance,
            "storage_gate_snapshot_sha256": document["storage_gate_snapshot_sha256"],
            "storage_root_source_token_hash": document["storage_root_source_token_hash"],
        }
    ):
        raise PilotPreviewError("preview completion marker provenance binding 不符")
    # marker 只宣告整批 artifact 可讀；reader 仍須重新核對發布時寫入的完整 inventory
    # 與每個產品 hash。這能拒絕發布後新增的檔案、遺失的檔案、替換的 symbolic link
    # 或只改檔案內容卻未同步 manifest 的竄改，避免下游把「marker 存在」誤當成資料完整。
    products = manifest_document.get("files")
    if not isinstance(products, Mapping):
        raise PilotPreviewError("preview completion marker manifest files 無效")
    expected_names = set(products) | {"manifest.json", _COMPLETION_MARKER_NAME}
    try:
        actual_names = {path.name for path in preview.iterdir()}
    except OSError:
        raise PilotPreviewError("preview completion marker 目錄無法列舉") from None
    if actual_names != expected_names:
        raise PilotPreviewError("preview completion marker inventory 不符")
    for name, contract in products.items():
        if not isinstance(name, str) or Path(name).name != name or name.startswith("."):
            raise PilotPreviewError("preview completion marker product name 無效")
        if not isinstance(contract, Mapping):
            raise PilotPreviewError("preview completion marker product contract 無效")
        product = _safe_path(preview / name)
        _regular_preview_file(product)
        if (
            type(contract.get("size_bytes")) is not int
            or contract["size_bytes"] < 0
            or product.stat().st_size != contract["size_bytes"]
            or not isinstance(contract.get("sha256"), str)
            or _HEX64.fullmatch(contract["sha256"]) is None
            or sha256_file(product) != contract["sha256"]
        ):
            raise PilotPreviewError("preview completion marker product bytes/hash 不符")
    if storage_gate_evidence is not None:
        gate = _verified_storage_gate_evidence(storage_gate_evidence)
        if (
            gate["sha256"] != document["storage_gate_snapshot_sha256"]
            or gate["source_token_hash"] != document["storage_root_source_token_hash"]
        ):
            raise PilotPreviewError("preview completion marker storage gate binding 不符")
    return document


def validate_published_preview(
    preview: str | Path,
    *,
    storage_gate_evidence: str | Path | None = None,
) -> dict[str, Any] | None:
    """驗證預覽的發布協定，供 CSV reader 與 BayTrace wrapper 共用。

    新的 NFS marker artifact 必須先通過 ``.complete``；未帶 marker 的既有成果仍保留
    exclusive-directory-rename 相容性。若 caller 明示 gate evidence，則要求預覽使用
    marker protocol，避免把閘門驗證誤套在 legacy 目錄或直接寫入的半成品上。
    """

    return validate_published_artifact(
        preview,
        expected_artifact_kind="pilot-preview-v1",
        storage_gate_evidence=storage_gate_evidence,
    )


def validate_published_artifact(
    preview: str | Path,
    *,
    expected_artifact_kind: str,
    storage_gate_evidence: str | Path | None = None,
) -> dict[str, Any] | None:
    """驗證任一 marker-enabled 預覽／圖面 artifact，供 BayTrace reader 共用。

    ``expected_artifact_kind`` 將 pilot 原始預覽與 coastline 四圖分開綁定；兩者都必須
    先通過相同 marker、manifest、程式 provenance 與儲存閘門檢查。未宣告 marker 的舊
    exclusive release 只在沒有 gate evidence 時保留相容性，不能被拿來冒充本次 NFS 產物。
    """

    root = _safe_path(preview)
    marker = root / _COMPLETION_MARKER_NAME
    if marker.exists() or marker.is_symlink():
        return _validate_completion_marker(
            root,
            expected_artifact_kind=expected_artifact_kind,
            storage_gate_evidence=storage_gate_evidence,
        )
    try:
        manifest = _small_json(root / "manifest.json")
    except PilotPreviewError:
        # 逐檔發布若在 manifest 搬入前中止，final 可能只剩部分產品；不能以缺少
        # manifest 猜測它是舊版成果，統一視為沒有完成宣告的 invalid release。
        raise PilotPreviewError("preview completion marker 缺失") from None
    if manifest.get("publish_protocol") == _NFS_MARKER_PROTOCOL:
        raise PilotPreviewError("preview completion marker 缺失")
    if storage_gate_evidence is not None:
        raise PilotPreviewError("preview storage gate evidence 需要 completion marker")
    return None


def _validate_staged_preview(staging: Path, manifest: Mapping[str, Any]) -> None:
    """在發布前核對 staging 的完整 ordinary-file inventory 與每個產品雜湊。

    逐檔 marker 協定允許 final 目錄在發布途中暫時存在，因此自我驗證必須先封閉所有
    檔案與 manifest contract；任何額外節點、symbolic link、bytes 或大小差異都停止發布。
    此處不刪除 staging，讓操作端能保留失敗證據。
    """

    products = manifest.get("files")
    if not isinstance(products, Mapping):
        raise PilotPreviewError("preview staging manifest files 無效")
    expected = set(products) | {"manifest.json"}
    actual = {path.name for path in staging.iterdir()}
    if actual != expected:
        raise PilotPreviewError("preview staging inventory 不符")
    for path in staging.iterdir():
        _regular_preview_file(path)
    manifest_path = staging / "manifest.json"
    if manifest_path.read_bytes() != _canonical_json_bytes(manifest):
        raise PilotPreviewError("preview staging manifest bytes 不符")
    for name, contract in products.items():
        if not isinstance(name, str) or Path(name).name != name or name.startswith("."):
            raise PilotPreviewError("preview staging product name 無效")
        if not isinstance(contract, Mapping):
            raise PilotPreviewError("preview staging product contract 無效")
        path = staging / name
        if path.stat().st_size != contract.get("size_bytes") or sha256_file(path) != contract.get("sha256"):
            raise PilotPreviewError("preview staging product bytes 不符")


def _acquire_nfs_publish_lock(parent: Path, destination: Path) -> int:
    """在成果父目錄取得同主機 cooperative flock，拒絕鎖競爭與符號連結。

    儲存閘門已先驗證目前 NFS 的跨程序鎖定能力；這個 lock descriptor 會從 staging
    自我驗證前後一直持有到 final marker 與 parent fsync 完成。鎖檔保留在 parent 作為
    固定 cooperative protocol 名稱，避免成功後刪除／重建鎖檔造成不同程序各持有不同
    inode；它不是成果 manifest 的資料檔，也不會被搬入 final 目錄。
    """

    if not hasattr(fcntl, "flock"):
        raise PilotPreviewError("preview NFS marker 缺少 flock 支援")
    lock_path = parent / f".{destination.name}.publish.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if descriptor is not None:
            os.close(descriptor)
        raise PilotPreviewError("preview 發布鎖忙碌") from None
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise PilotPreviewError("preview 發布鎖無法建立") from None
    if descriptor is None:
        raise PilotPreviewError("preview 發布鎖 descriptor 無效")
    return descriptor


def _release_nfs_publish_lock(descriptor: int) -> None:
    """釋放預覽 cooperative flock；釋放失敗不掩蓋已保存的發布例外。"""

    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _fsync_preview_file(path: Path) -> None:
    """以 no-follow descriptor 同步已搬入 final 的普通檔位元組。"""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PilotPreviewError("preview final 含非普通檔")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_nfs_marker_preview(
    staging: Path,
    destination: Path,
    *,
    parent_identity: tuple[int, int],
    marker: Mapping[str, Any],
) -> None:
    """以逐檔 durable move、最後 exclusive marker 發布 NFS preview。

    final 目錄先以 ``mkdir`` 做 exclusive claim，之後才把已驗證 staging 的普通檔逐一
    搬入並 fsync；這不是整體原子 rename。任何中途錯誤都保留無 ``.complete`` 的 invalid
    final，reader 因而無法消費，也不以 copy/delete 模擬原子性。只有全部檔案、final
    directory 與 marker bytes 都完成 fsync，且 parent identity 未變，才回傳成功。
    """

    destination.mkdir(mode=0o750, exist_ok=False)
    for source in sorted(staging.iterdir(), key=lambda item: item.name):
        _regular_preview_file(source)
        target = destination / source.name
        # 同父 cooperative flock 已序列化本協定 writer；這個明示檢查仍拒絕任何
        # 非預期的目錄／符號連結碰撞，避免 os.rename 在異常狀態下覆寫既有節點。
        if target.exists() or target.is_symlink():
            raise FileExistsError("preview final product collision")
        os.rename(source, target)
        _fsync_preview_file(target)
    _fsync_directory(destination)
    manifest_path = destination / "manifest.json"
    if sha256_file(manifest_path) != marker["manifest_sha256"]:
        raise PilotPreviewError("preview final manifest hash 不符")
    marker_path = destination / _COMPLETION_MARKER_NAME
    _write_exclusive_durable_bytes(marker_path, _canonical_json_bytes(marker))
    _fsync_directory(destination)
    _fsync_directory(destination.parent, expected_identity=parent_identity)
    with contextlib.suppress(OSError):
        staging.rmdir()


def _trajectory_metadata(root: Path, plan: Mapping) -> dict:
    """由小型進度／分片清單取得宣告計數，供早期拒絕及前後指紋比對。

    此處不讀粒子表、觀測陣列或 forcing；宣告數量不能證明內容正確，後續仍需完整驗證。
    """

    _small_json(root / "run_progress.json")
    progress = load_run_progress(root)
    shards = {}
    for row in plan["shards"]:
        shard_id = row["shard_id"]
        entry = progress["shards"][shard_id]
        if entry["lifecycle"] != "COMPLETE":
            raise PilotPreviewError("preview 需要全部分片完成")
        path = _safe_path(root / entry["output_relative_path"] / "manifest.json")
        if not path.is_relative_to(root):
            raise PilotPreviewError("trajectory manifest 不屬於來源 run")
        manifest = _small_json(path)
        schema_version = manifest.get("schema_version")
        if type(schema_version) is not str or schema_version not in _PREVIEW_TRAJECTORY_SCHEMA_VERSIONS:
            raise PilotPreviewError("preview 只接受 trajectory schema 2.0.0 或 3.0.0；schema 1 缺少環境欄位")
        counts = {key: manifest[key] for key in ("particle_count", "observation_count", "event_count")}
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise PilotPreviewError("trajectory manifest 計數無效")
        shards[shard_id] = {**counts, "trajectory_manifest_sha256": sha256_file(path)}
    return shards


def _early_reject(root: Path, *, max_particles: int, max_observations: int) -> tuple[dict, dict]:
    """在通用驗證器讀軌跡前拒絕非精確先導或超量輸入，不產生驗收成功結論。"""

    _small_json(root / "run_plan.json")
    plan = load_run_plan(root)
    if plan["run_kind"] != "pilot" or plan.get("scenario_selection", {}).get("mode") != "pilot_exact":
        raise PilotPreviewError("preview 只接受 pilot_exact 的 pilot run")
    if plan["particle_count"] > max_particles:
        raise PilotPreviewError("preview 粒子數超過上限")
    shards = _trajectory_metadata(root, plan)
    if sum(row["particle_count"] for row in shards.values()) > max_particles:
        raise PilotPreviewError("preview 分片粒子數超過上限")
    if sum(row["observation_count"] for row in shards.values()) > max_observations:
        raise PilotPreviewError("preview 觀測數超過上限")
    return plan, shards


def _source_snapshot(root: Path, config: Path, plan: Mapping) -> tuple[dict, dict]:
    """完整來源驗證後保存實際文件位元組與分片指紋，發布前再驗一次。

    上層已通過完整 static 驗證；此處不替代它，只保存前後可比的實際 bytes 雜湊。
    對外只用邏輯名稱、相對分片 ID 及摘要，不保存 config 或 run 的絕對路徑。
    """

    names = (
        "run_plan.json",
        "run_progress.json",
        "normalized_config.json",
        "input_inventory.json",
        "scenario_table.parquet",
        "seed_table.parquet",
    )
    hashes = {name: sha256_file(_safe_path(root / name)) for name in names}
    hashes["caller_config_sha256"] = sha256_file(config)
    if _small_json(root / "run_plan.json") != _plain(plan):
        raise PilotPreviewError("preview 計畫在來源驗證期間變動")
    if any(hashes[name] != contract["sha256"] for name, contract in plan["files"].items()):
        raise PilotPreviewError("preview 來源文件與計畫指紋不符")
    return hashes, _trajectory_metadata(root, plan)


def _diagnostic(result) -> dict:
    """只取終止事件已保存的標準診斷；舊空 attributes 明示 unknown／null。

    失敗原因（failure_reason）、階段（failure_stage）只接受不含路徑的代碼，其他文字
    標為 unknown。品質位元（qc_flags）缺少時保留 null，實際的零仍原樣保留。
    來源可用性（availability）另列，不以省略
    推定數值零、成功或特定失敗原因；診斷版本亦不自行升級。
    """

    events = [event for event in result.events if event.event_type.value == result.final_state.status.value]
    attributes = events[-1].attributes if events else {}
    text_values = {
        key: value
        if type(value := attributes.get(key)) is str and _DIAGNOSTIC_TEXT.fullmatch(value)
        else "unknown"
        for key in ("failure_reason", "failure_stage")
    }
    return {
        **text_values,
        **{key: value if type(value := attributes.get(key)) is int else None for key in _DIAGNOSTIC_INTS},
        **{key: _finite(attributes.get(key)) for key in _DIAGNOSTIC_FLOATS},
        **{
            key: value if type(value := attributes.get(key)) is bool else None
            for key in _DIAGNOSTIC_AVAILABILITY
        },
    }


def _collect(
    root: Path,
    static,
    shard_sources: dict,
    *,
    max_particles: int,
    max_observations: int,
    max_curves_per_vertical: int,
    checkpoint_root: Path | None,
) -> tuple:
    """逐分片核對全部成員與來源軸，回傳有界逐粒子／觀測表及統計。

    不按停止原因刪除任何資料；只有畫線集合依 member_id、receptor_id、scenario_id
    固定排序限量。所有成員都保留於表格及分母。UTC 與回溯秒數誤差只容許奈秒量化
    引起的 1 微秒絕對或 1e-9 相對誤差，不重新推算或修正軌跡。
    """

    plan = static.plan
    scenarios = static.scenario_inputs.scenarios
    selection = plan["scenario_selection"]
    if plan["run_kind"] != "pilot" or selection["mode"] != "pilot_exact":
        raise PilotPreviewError("preview 只接受 pilot_exact 的 pilot run")
    identities = {(item.study_site_id, item.arrival_time_id, item.material_id) for item in scenarios}
    if len(identities) != 1 or next(iter(identities)) != (
        selection["study_site_id"],
        selection["arrival_time_id"],
        selection["material_id"],
    ):
        raise PilotPreviewError("preview 必須是單站／同到達／同材質並吻合選擇繫結")
    site, arrival_id, material = next(iter(identities))
    receptors = {
        item.receptor_id: item for item in static.scenario_inputs.receptors if item.study_site_id == site
    }
    if len(scenarios) != len(receptors) or {item.receptor_id for item in scenarios} != set(receptors):
        raise PilotPreviewError("preview 必須保留本站全部受體，不缺不多")
    if len({item.arrival_time_utc_ns for item in scenarios}) != 1:
        raise PilotPreviewError("preview 到達 UTC 不一致")
    speeds = {item.settling_velocity_mps for item in scenarios}
    if len(speeds) != 1 or not math.isfinite(next(iter(speeds))) or next(iter(speeds)) >= 0:
        raise PilotPreviewError("preview 必須保留同一有限負沉降速度")
    members = plan["members_per_scenario"]
    if plan["particle_count"] > max_particles or plan["particle_count"] != len(scenarios) * members:
        raise PilotPreviewError("preview 粒子數超過上限或情境／M 分母不符")
    if sum(row["observation_count"] for row in shard_sources.values()) > max_observations:
        raise PilotPreviewError("preview 觀測數超過上限")
    expected = {}
    for scenario in scenarios:
        for member_id in range(members):
            particle_id = stable_identifier(
                "prt", [scenario.scenario_id, plan["experiment_case_id"], str(member_id)]
            )
            expected[particle_id] = (scenario, member_id, receptors[scenario.receptor_id].vertical_id)
    levels = sorted(
        {item.vertical_id for item in receptors.values()},
        key=lambda value: (
            _VERTICAL_PRIORITY.index(value) if value in _VERTICAL_PRIORITY else len(_VERTICAL_PRIORITY),
            value,
        ),
    )
    # 水平分組直接取來源受體的經緯度配對，不從受體 ID 字串猜位置，也不依位移長短分組。
    positions = sorted({(item.lon, item.lat) for item in receptors.values()})
    horizontal = {item.receptor_id: positions.index((item.lon, item.lat)) + 1 for item in receptors.values()}
    if max_curves_per_vertical < max(
        sum(item.vertical_id == level for item in receptors.values()) for level in levels
    ):
        raise PilotPreviewError("preview 曲線上限須至少保留每個水平受體的各垂向一條曲線")
    drawn = set()
    for level in levels:
        candidates = [pid for pid, (_, _, vertical) in expected.items() if vertical == level]
        candidates.sort(
            key=lambda pid: (expected[pid][1], expected[pid][0].receptor_id, expected[pid][0].scenario_id)
        )
        drawn.update(candidates[:max_curves_per_vertical])
    counts = {level: dict.fromkeys(_TERMINAL_STATUSES, 0) for level in levels}
    particles, observations, seen, seen_shards = [], [], set(), set()
    projection = static.geometries.projections[site]
    for shard in iter_complete_run_trajectory_shards(root, checkpoint_root=checkpoint_root):
        if shard.shard_id in seen_shards or shard.shard_id not in shard_sources:
            raise PilotPreviewError("preview 分片重複或與來源計畫不符")
        seen_shards.add(shard.shard_id)
        if (
            type(shard.trajectory_schema_version) is not str
            or shard.trajectory_schema_version not in _PREVIEW_TRAJECTORY_SCHEMA_VERSIONS
        ):
            raise PilotPreviewError("preview 只接受 trajectory schema 2.0.0 或 3.0.0；拒絕 schema 1")
        if shard.trajectory_manifest_sha256 != shard_sources[shard.shard_id]["trajectory_manifest_sha256"]:
            raise PilotPreviewError("preview trajectory manifest 在讀取期間變動")
        for result in shard.results:
            state = result.final_state
            if state.particle_id in seen or state.particle_id not in expected:
                raise PilotPreviewError("preview 粒子身分重複或超出來源選擇")
            scenario, member_id, vertical = expected[state.particle_id]
            if (
                state.scenario_id,
                state.member_id,
                state.study_site_id,
                state.receptor_id,
                state.analysis_region_id,
            ) != (
                scenario.scenario_id,
                member_id,
                site,
                scenario.receptor_id,
                scenario.analysis_region_id,
            ) or state.status.value not in _TERMINAL_STATUSES:
                raise PilotPreviewError("preview 粒子身分或終止狀態不符")
            seen.add(state.particle_id)
            counts[vertical][state.status.value] += 1
            diagnostic = _diagnostic(result)
            lon, lat = projection.unproject(state.x_m, state.y_m)
            particles.append(
                {
                    "particle_id": state.particle_id,
                    "scenario_id": scenario.scenario_id,
                    "member_id": member_id,
                    "study_site_id": site,
                    "arrival_time_id": arrival_id,
                    "material_id": material,
                    "receptor_id": scenario.receptor_id,
                    "vertical_id": vertical,
                    "horizontal_panel": horizontal[scenario.receptor_id],
                    "seed": derive_member_seed(
                        master_seed=plan["master_seed"],
                        scenario_id=scenario.scenario_id,
                        experiment_case_id=plan["experiment_case_id"],
                        member_id=member_id,
                        # paired plan 的 seed 命名空間由 immutable plan 指定；預覽只讀
                        # 結果但仍須輸出與 ProductionBatch／seed table 相同的 seed，不能
                        # 因為 particle identity 保留物理案例而回退到 experiment case。
                        random_stream_id=plan.get("random_stream_id"),
                    ),
                    "status": state.status.value,
                    "time_utc_ns": state.time_utc_ns,
                    "age_seconds": state.age_seconds,
                    "x_m": _finite(state.x_m),
                    "y_m": _finite(state.y_m),
                    "z_m": _finite(state.z_m),
                    "longitude": _finite(lon),
                    "latitude": _finite(lat),
                    "result_step_count": result.step_count,
                    "result_minimum_clamp_count": result.minimum_clamp_count,
                    "observation_count": len(result.observations),
                    "curve_drawn": state.particle_id in drawn,
                    **diagnostic,
                }
            )
            previous_age = -1.0
            for observation in result.observations:
                if len(observations) >= max_observations:
                    raise PilotPreviewError("preview 實際觀測數超過上限")
                if observation.particle_id != state.particle_id or observation.age_seconds < previous_age:
                    raise PilotPreviewError("preview 觀測身分或回溯順序不符")
                age_from_utc = (scenario.arrival_time_utc_ns - observation.time_utc_ns) / 1e9
                if not math.isclose(observation.age_seconds, age_from_utc, rel_tol=1e-9, abs_tol=1e-6):
                    raise PilotPreviewError("preview UTC 奈秒與回溯秒數不符")
                previous_age = observation.age_seconds
                observations.append(
                    {
                        "particle_id": state.particle_id,
                        "vertical_id": vertical,
                        "time_utc_ns": observation.time_utc_ns,
                        "age_seconds": observation.age_seconds,
                        "x_m": _finite(observation.x_m),
                        "y_m": _finite(observation.y_m),
                        "z_m": _finite(observation.z_m),
                        "eta_m": observation.eta_m,
                        "bed_z_m": observation.bed_z_m,
                        "environment_sample_status": observation.environment_sample_status.value,
                        "environment_qc_flags": observation.environment_qc_flags,
                        "forcing_month_id": observation.forcing_month_id,
                        "velocity_sample_status": observation.velocity_sample_status.value,
                        "total_u_mps": observation.total_u_mps,
                        "total_v_mps": observation.total_v_mps,
                        "total_w_mps": observation.total_w_mps,
                        "ocm_u_mps": observation.ocm_u_mps,
                        "ocm_v_mps": observation.ocm_v_mps,
                        "ocm_w_mps": observation.ocm_w_mps,
                        "stokes_u_mps": observation.stokes_u_mps,
                        "stokes_v_mps": observation.stokes_v_mps,
                        "settling_w_mps": observation.settling_w_mps,
                        "velocity_qc_flags": observation.velocity_qc_flags,
                    }
                )
    if seen != set(expected) or seen_shards != set(shard_sources):
        raise PilotPreviewError("preview 未取得全部選中成員或分片")
    if len(observations) != sum(row["observation_count"] for row in shard_sources.values()):
        raise PilotPreviewError("preview 觀測總數與來源不一致")
    particles.sort(key=lambda row: (levels.index(row["vertical_id"]), row["member_id"], row["receptor_id"]))
    totals = {status: sum(row[status] for row in counts.values()) for status in _TERMINAL_STATUSES}
    diagnostics = Counter(
        (row["status"], row["failure_reason"], row["failure_stage"], row["qc_flags"]) for row in particles
    )
    summary = {
        "artifact_kind": "pilot-preview-v1",
        "schema_version": PILOT_PREVIEW_SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "run_kind": "pilot",
        "selection_mode": "pilot_exact",
        "study_site_id": site,
        "arrival_time_id": arrival_id,
        "arrival_time_utc_ns": scenarios[0].arrival_time_utc_ns,
        "arrival_utc": datetime.fromtimestamp(
            scenarios[0].arrival_time_utc_ns // 1_000_000_000, UTC
        ).isoformat(),
        "material_id": material,
        "settling_velocity_mps": next(iter(speeds)),
        "scenario_count": len(scenarios),
        "receptor_count": len(receptors),
        "members_per_scenario": members,
        "particle_count": len(particles),
        "observation_count": len(observations),
        "terminal_position_sample_count": sum(
            all(row[key] is not None for key in ("x_m", "y_m", "z_m")) for row in particles
        ),
        "terminal_position_sample_policy": "finite_xyz_all_terminal_statuses_including_failures",
        "failure_particle_count": totals["data_gap"] + totals["numerical_failure"],
        "terminal_counts": totals,
        "terminal_counts_by_vertical": counts,
        "vertical_order": levels,
        "diagnostics": [
            {
                "status": key[0],
                "failure_reason": key[1],
                "failure_stage": key[2],
                "qc_flags": key[3],
                "count": count,
            }
            for key, count in sorted(diagnostics.items(), key=lambda item: repr(item[0]))
        ],
        "failure_details": [
            {
                key: row[key]
                for key in (
                    "particle_id",
                    "scenario_id",
                    "member_id",
                    "status",
                    "failure_reason",
                    "failure_stage",
                    *_DIAGNOSTIC_INTS,
                    *_DIAGNOSTIC_FLOATS,
                    *_DIAGNOSTIC_AVAILABILITY,
                )
            }
            for row in particles
            if row["status"] in {"numerical_failure", "data_gap"}
        ],
        "horizontal_panels": [
            {"panel": index + 1, "receptor_lon": lon, "receptor_lat": lat}
            for index, (lon, lat) in enumerate(positions)
        ],
        "environment_counts": dict(Counter(row["environment_sample_status"] for row in observations)),
        "curve_selection_policy": "member_id_receptor_id_scenario_id_v1",
        "drawn_particle_count": len(drawn),
        "drawn_particle_ids": sorted(drawn),
        "drawn_count_by_vertical": {
            level: sum(expected[pid][2] == level for pid in drawn) for level in levels
        },
        "engineering_only": True,
        "convergence_validated": False,
        "observation_validated": False,
    }
    return particles, observations, summary


_ZH_LABELS = (
    "水平回溯軌跡",
    "深度與回溯時間",
    "全部成員停止原因",
    "回溯時間（秒）",
    "公尺",
    "工程先導未經收斂與觀測驗證",
)
_EN_LABELS = (
    "Horizontal backtracking",
    "Depth and backward age",
    "All-member termination counts",
    "Backward age (s)",
    "m",
    "Engineering pilot; no convergence or observational validation",
)


def _plot_font(font_path: Path | None) -> tuple[Any, dict, tuple[str, ...]]:
    """在專用字型快取驗證後挑選字型；缺中文字碼時只用英文圖面，不下載字型。

    自訂路徑必須是普通字型檔。回傳 FontProperties、無路徑的字型摘要及固定語言標籤；
    所需中文字碼與全部 ASCII 均檢查，避免圖上缺字方框。繁中 Markdown 不受圖面語言影響。
    """

    from matplotlib import font_manager, get_data_path
    from matplotlib.ft2font import FT2Font

    candidate = font_path
    if candidate is None:
        from .report_font import resolve_cjk_font

        try:
            selected = resolve_cjk_font()
            candidate = Path(
                font_manager.findfont(
                    font_manager.FontProperties(family=[selected.family]), fallback_to_default=False
                )
            )
        except ValueError:
            candidate = None
    required = set("".join(_ZH_LABELS) + "來源水層" + "".join(row[0] for row in _LEVEL_LABELS.values()))
    required |= {chr(value) for value in range(32, 127)}
    if candidate is not None:
        candidate = _safe_path(candidate)
        charmap = FT2Font(str(candidate)).get_charmap()
        if all(ord(char) in charmap for char in required):
            return (
                font_manager.FontProperties(fname=candidate),
                {
                    "filename": candidate.name,
                    "sha256": sha256_file(candidate),
                    "plot_language": "zh-TW",
                },
                _ZH_LABELS,
            )
    fallback = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return (
        font_manager.FontProperties(fname=fallback),
        {
            "filename": fallback.name,
            "sha256": sha256_file(fallback),
            "plot_language": "en",
        },
        _EN_LABELS,
    )


def _render(
    staging: Path, particles: list, observations: list, summary: dict, font_path: Path | None
) -> dict:
    """畫三張有界 PNG；x/y 等比例公尺軸、z 正向上，缺 eta／bed 留空不補零。

    無底圖也無海岸推測；投影來源由 summary 保存。深度面板依來源垂向組別完整呈現，
    near_bed 優先。停止圖計入全部成員，與限量畫線分母明確分開。
    """

    validate_mplconfigdir()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    font, font_record, labels = _plot_font(font_path)
    levels = summary["vertical_order"]
    curves = {pid: [] for pid in summary["drawn_particle_ids"]}
    for row in observations:
        if row["particle_id"] in curves:
            curves[row["particle_id"]].append(row)
    grouped = {level: [row for row in particles if row["vertical_id"] == level] for level in levels}
    language_index = 0 if font_record["plot_language"] == "zh-TW" else 1
    level_labels = {
        level: f"L{index + 1} " + _LEVEL_LABELS.get(level, ("來源水層", "Source level"))[language_index]
        for index, level in enumerate(levels)
    }
    legend_font = font.copy()
    legend_font.set_size(8)

    def save(fig, name: str, title: str) -> None:
        """把完整 PNG 先寫記憶體再排他落檔，確保自有暫存目錄內也不覆寫節點。"""
        try:
            fig.suptitle(
                f"{title}\n{summary['study_site_id']} | arrival UTC {summary['arrival_utc']} | "
                f"M={summary['members_per_scenario']} | "
                f"horizon={summary['settings']['horizon_seconds']:g} s\n"
                f"{labels[5]} | curves {summary['drawn_particle_count']}/{summary['particle_count']}",
                fontproperties=font,
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=150, metadata={"Software": "pilot-preview-v1"})
            _write_exclusive_durable_bytes(staging / name, buffer.getvalue())
        finally:
            plt.close(fig)

    with matplotlib.rc_context({"axes.unicode_minus": False}):
        panel_count = len(summary["horizontal_panels"]) + 1
        columns = 2
        fig, axes = plt.subplots(
            math.ceil(panel_count / columns),
            columns,
            figsize=(13, 4.4 * math.ceil(panel_count / columns)),
            squeeze=False,
        )
        for panel_index, ax in enumerate(axes.flat):
            if panel_index >= panel_count:
                ax.set_visible(False)
                continue
            selected = (
                particles
                if panel_index == 0
                else [row for row in particles if row["horizontal_panel"] == panel_index]
            )
            plotted = [row for row in selected if row["particle_id"] in curves]
            for index, level in enumerate(levels):
                for particle in plotted:
                    if particle["vertical_id"] != level:
                        continue
                    rows = curves[particle["particle_id"]]
                    if rows:
                        ax.plot(
                            [row["x_m"] for row in rows],
                            [row["y_m"] for row in rows],
                            color=f"C{index % 10}",
                            alpha=0.65,
                            linewidth=1,
                        )
                        # 起訖標記也保留無位移或僅一筆觀測的失敗成員，不偽造最小位移。
                        ax.plot(rows[0]["x_m"], rows[0]["y_m"], "o", color=f"C{index % 10}", markersize=4)
                        ax.plot(rows[-1]["x_m"], rows[-1]["y_m"], "x", color=f"C{index % 10}", markersize=5)
                n = sum(row["vertical_id"] == level for row in plotted)
                total = sum(row["vertical_id"] == level for row in selected)
                ax.plot([], [], color=f"C{index % 10}", label=f"{level_labels[level]}: {n}/{total}")
            ax.set_aspect("equal", adjustable="datalim")
            title = "Overview (no basemap)" if panel_index == 0 else f"H{panel_index} local extent"
            ax.set_title(f"{title} | drawn/all {len(plotted)}/{len(selected)}", fontsize=10)
            ax.set_xlabel("Source AEQD x (m)")
            ax.set_ylabel("Source AEQD y (m)")
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.tick_params(labelsize=8)
            ax.legend(prop=legend_font, ncols=2)
            ax.grid(alpha=0.3)
            if panel_index:
                summary["horizontal_panels"][panel_index - 1].update(
                    drawn_particle_count=len(plotted),
                    particle_count=len(selected),
                )
        save(fig, "horizontal.png", labels[0])

        fig, axes = plt.subplots(len(levels), 1, figsize=(10, max(3, 2.6 * len(levels))), squeeze=False)
        for index, level in enumerate(levels):
            ax = axes[index, 0]
            for particle in grouped[level]:
                rows = curves.get(particle["particle_id"])
                if not rows:
                    continue
                age = [row["age_seconds"] for row in rows]
                ax.plot(age, [row["z_m"] for row in rows], color=f"C{index % 10}", alpha=0.7)
                for name, color, style in (("eta_m", "green", "--"), ("bed_z_m", "black", ":")):
                    ax.plot(
                        age,
                        [np.nan if row[name] is None else row[name] for row in rows],
                        color=color,
                        linestyle=style,
                        linewidth=0.5,
                        alpha=0.5,
                    )
            ax.set_title(
                f"{level_labels[level]}  curves "
                f"{summary['drawn_count_by_vertical'][level]}/{len(grouped[level])}",
                fontproperties=font,
            )
            ax.set_ylabel("z (m, positive up)")
            ax.set_xlabel(labels[3], fontproperties=font)
            ax.grid(alpha=0.3)
        axes[0, 0].plot([], [], color="C0", label="particle z (panel color)")
        axes[0, 0].plot([], [], color="green", linestyle="--", label="eta (missing = gap)")
        axes[0, 0].plot([], [], color="black", linestyle=":", label="bed (missing = gap)")
        axes[0, 0].legend(fontsize=8)
        save(fig, "depth_age.png", labels[1])

        fig, axes = plt.subplots(len(levels), 1, figsize=(10, max(3, 2.7 * len(levels))), squeeze=False)
        for index, level in enumerate(levels):
            ax = axes[index, 0]
            ax.barh(
                _TERMINAL_STATUSES,
                [summary["terminal_counts_by_vertical"][level][key] for key in _TERMINAL_STATUSES],
            )
            ax.set_title(f"{level_labels[level]}  all members N={len(grouped[level])}", fontproperties=font)
            ax.set_xlabel("Particle count (includes failures)")
            ax.xaxis.get_major_locator().set_params(integer=True)
        save(fig, "terminal_counts.png", labels[2])
    return font_record


def _csv_bytes(rows: list[dict]) -> bytes:
    """輸出每列同欄位的 UTF-8 CSV；None 留空，原始零值仍寫為 0。"""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _caption(summary: dict) -> str:
    """產生 PI 可審查的繁中說明，明列設計、分母、缺值、失敗及科學限制。"""

    settings = summary["settings"]
    lines = [
        "# 單站沉降工程先導預覽",
        "",
        "本成果為 pilot-preview-v1，不是正式 report-v1 或全期研究結論。",
        "",
        f"研究站點：{summary['study_site_id']}；到達 UTC：{summary['arrival_utc']}。",
        f"材質：{summary['material_id']}；正向沉降速度：{summary['settling_velocity_mps']} m/s。",
        f"情境 n={summary['scenario_count']}；受體 n={summary['receptor_count']}；"
        f"每情境 M={summary['members_per_scenario']}；全部粒子 N={summary['particle_count']}；"
        f"有限終止位置樣本 n={summary['terminal_position_sample_count']}（含失敗）。",
        f"畫線數={summary['drawn_particle_count']}/{summary['particle_count']}，"
        "依成員、受體、情境順序固定選取；停止圖及 CSV 未刪成員。",
        f"設定：dt_min/dt_max={settings['dt_min_seconds']}/{settings['dt_max_seconds']} 秒；"
        f"horizon={settings['horizon_seconds']} 秒；master seed={settings['master_seed']}。",
        f"擴散模式：{settings['diffusion_kind']}；"
        f"Kh={settings['diffusion']['horizontal'].get('constant_kh_m2ps', '未提供')}、"
        f"Kz={settings['diffusion']['vertical'].get('constant_kz_m2ps', '未提供')} 公尺平方／秒。",
        "上述為設定係數，非沿途重算的有效擴散；完整設定請參閱 summary.json。",
        "",
        "水平圖採來源幾何的 AEQD 公尺投影，x/y 等比例，未繪底圖或推測海岸。",
        "總覽及各水平受體局部面板分別自動設定座標範圍，均保留真實公尺刻度，不放大位移。",
        "圓點為曲線起點，叉號為終點；各面板揭露全部／畫出成員數，各垂向固定同色。零位移仍保留。",
        "深度 z、海面 eta 與海床 bed 均使用來源垂直基準、公尺正向上；缺失環境留空，不補成零。",
        "逆向時間曲線變淺不表示材料在正向時間上浮。near_bed 優先呈現，其餘垂向層完整保留。",
        "",
        "## 垂向面板及全部停止原因",
        "",
    ]
    for index, level in enumerate(summary["vertical_order"]):
        lines.append(f"L{index + 1}：{_LEVEL_LABELS.get(level, (level, level))[0]}（{level}）。")
    lines += [
        "",
        "| 停止原因 | 全部粒子數 | "
        + " | ".join(f"L{index + 1}" for index in range(len(summary["vertical_order"])))
        + " |",
        "|---|---:|" + "---:|" * len(summary["vertical_order"]),
    ]
    for status, total in summary["terminal_counts"].items():
        columns = " | ".join(
            str(summary["terminal_counts_by_vertical"][level][status]) for level in summary["vertical_order"]
        )
        lines.append(f"| {_STOP_LABELS_ZH.get(status, status)} | {total} | {columns} |")
    lines += [
        "",
        "## 診斷與驗證限制",
        "",
        "數值失敗及資料缺口皆計入全部成員分母；"
        "缺失敗原因／階段（failure_reason／failure_stage）時標 unknown。",
        "品質位元（qc_flags）、失敗查詢公尺座標／UTC 奈秒、海面／海床、有方向的嘗試步長及計數上限逐欄保存。",
        "未提供的數值為 JSON null／CSV 空欄，可用性旗標保留來源值；"
        "不補零，也不拿最後粒子位置猜失敗查詢位置。",
        f"失敗粒子共 {summary['failure_particle_count']}/{summary['particle_count']}；"
        "逐筆診斷、完整原因／階段計數及設定保存在 summary.json，不從缺值推論原因。",
        "",
        "本工程試跑未經 M／dt 收斂與獨立觀測驗證，不宣稱全期代表性、絕對來源機率或因果來源歸因。",
        "summary.json 保存全部設定、來源及程式指紋；manifest.json 列各輸出 SHA-256（自身不做循環自我校驗）。",
        "particles.csv 每成員一列；observations.csv 保存全部已記錄觀測，"
        "時間為 UTC 奈秒、age 為秒，不重新取樣。",
    ]
    return "\n".join(lines) + "\n"


def build_pilot_preview(
    run: str | Path,
    *,
    config_path: str | Path,
    output: str | Path,
    checkpoint_root: str | Path | None = None,
    font_path: str | Path | None = None,
    max_particles: int = 2000,
    max_observations: int = 250_000,
    max_curves_per_vertical: int = 20,
    storage_gate_evidence: str | Path | None = None,
    nfs_marker_protocol: bool = False,
    release_provenance: Mapping[str, Any] | None = None,
) -> dict:
    """以已驗證完整先導建立新目錄，回傳無私有路徑的來源／輸出校驗清單。

    先只讀小型計畫／分片清單做容量與模式的早期拒絕；通過不代表來源已驗收。隨後必須
    完整驗證靜態來源（pilot、require_complete=True），再串流 schema 2／3 結果，三 ID 與
    全部受體／M 必須一致。容量上限在通用驗證器讀觀測前以清單宣告數量檢查，
    讀回後再驗實數；單次最多一個分片加有界表格，不載入 forcing。MPLCONFIGDIR 必須
    由 caller 明示且已存在；字型可指定，無合格中文字碼時圖用英文，繁中說明仍保留。

    來源驗證與讀取成功後才建自有同父暫存目錄，預設使用平台提供的原子拒覆寫發布。
    若明示 ``nfs_marker_protocol`` 與已驗證的 ``storage_gate_evidence``／release provenance，
    則改用同父 cooperative flock、逐檔 durable move 與最後 exclusive ``.complete``；沒有
    marker 的 final 永遠不被 reader 接受。目的已存在時拋出 FileExistsError，其他拒絕拋出
    PilotPreviewError，皆不洩漏底層私有路徑。失敗僅清理身分仍吻合且尚未搬移的 staging；
    marker 發布途中失敗會保留沒有 marker 的 invalid final 供稽核。
    """

    staging = None
    staging_identity = None
    parent_identity = None
    published = False
    storage_gate = None
    normalized_provenance = None
    phase = "source"
    try:
        root, config = _safe_path(run), _safe_path(config_path)
        if Path(output).exists() or Path(output).is_symlink():
            raise FileExistsError("preview 目的已存在")
        destination = _safe_path(output)
        for key, value in (
            ("max_particles", max_particles),
            ("max_observations", max_observations),
            ("max_curves_per_vertical", max_curves_per_vertical),
        ):
            if type(value) is not int or not 1 <= value <= _LIMITS[key]:
                raise PilotPreviewError(f"preview {key} 超出允許範圍")
        early_plan, early_shards = _early_reject(
            root, max_particles=max_particles, max_observations=max_observations
        )
        config_sha_before = sha256_file(config)
        static = runtime.load_validated_run_static_inputs(
            root,
            config_path=config,
            checkpoint_root=checkpoint_root,
            expected_run_kind="pilot",
            require_complete=True,
        )
        if static.plan["run_kind"] != "pilot" or static.plan["scenario_selection"]["mode"] != "pilot_exact":
            raise PilotPreviewError("preview 只接受 pilot_exact 的 pilot run")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("preview 目的已存在")
        if destination.is_relative_to(root) or root.is_relative_to(destination):
            raise PilotPreviewError("preview 輸出不得位於來源 run 內或取代其父層")
        parent = destination.parent
        parent_stat = parent.stat()
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise PilotPreviewError("preview 父層必須為既有普通目錄")
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        if nfs_marker_protocol:
            if storage_gate_evidence is None:
                raise PilotPreviewError("preview NFS marker 需要明示 storage gate evidence")
            storage_gate = _verified_storage_gate_evidence(storage_gate_evidence)
            normalized_provenance = _verified_release_provenance(release_provenance)
        elif storage_gate_evidence is not None or release_provenance is not None:
            raise PilotPreviewError("preview storage gate evidence 僅可搭配 NFS marker")
        source_hashes, shard_sources = _source_snapshot(root, config, static.plan)
        if (
            early_plan != _plain(static.plan)
            or early_shards != shard_sources
            or config_sha_before != source_hashes["caller_config_sha256"]
        ):
            raise PilotPreviewError("preview 來源在完整驗證期間變動")
        particles, observations, summary = _collect(
            root,
            static,
            shard_sources,
            max_particles=max_particles,
            max_observations=max_observations,
            max_curves_per_vertical=max_curves_per_vertical,
            checkpoint_root=checkpoint_root,
        )
        if not observations:
            raise PilotPreviewError("preview 缺少已記錄觀測，不能產製軌跡圖")
        config_data = static.config
        case = runtime.EXPERIMENT_CASE_SPECS[static.plan["experiment_case_id"]]
        summary["settings"] = {
            "experiment_case_id": static.plan["experiment_case_id"],
            "include_stokes": case.include_stokes,
            "diffusion_kind": case.diffusion_kind,
            "diffusion": _plain(
                {
                    "horizontal": config_data.physics["horizontal_diffusion"],
                    "vertical": config_data.physics["vertical_diffusion"],
                }
            ),
            "dt_min_seconds": config_data.integration.dt_min_seconds,
            "dt_max_seconds": config_data.integration.dt_max_seconds,
            "output_interval_seconds": config_data.integration.output_interval_seconds,
            "horizon_seconds": config_data.boundaries.max_backtrack_days * 86400,
            "master_seed": static.plan["master_seed"],
            "seed_policy": static.plan["seed_policy"],
        }
        domain = next(
            item
            for item in config_data.domains
            if item.analysis_region_id == static.scenario_inputs.scenarios[0].analysis_region_id
        )
        summary["projection"] = {
            "kind": "source_geometry_AEQD",
            "units": "m",
            "coastline_shown": False,
            "center_lonlat": list(domain.center_lonlat),
            "analysis_region_id": domain.analysis_region_id,
            "flow_domain_id": domain.flow_domain_id,
            "geometry_canonical_hashes": _plain(static.plan["geometry_canonical_hashes"]),
        }
        # 候選子區是輸入衍生階段從設定檔帶入的 receptor provenance；只複寫已驗證的
        # 小型設定資料，讓下游圖面能保存數位化 polygon、配額與來源影像雜湊。這些欄位
        # 只描述受體候選，不是 OCM／NWW forcing 支援，也不在本模組重畫或重新判定。
        study_site_id = str(summary["study_site_id"])
        site_config = next(
            item for item in config_data.study_sites if item.study_site_id == study_site_id
        )
        candidate_extras = site_config.model_extra or {}
        candidate_values = {
            "receptor_candidate_selection": site_config.receptor_candidate_selection,
            "receptor_candidate_regions": site_config.receptor_candidate_regions,
            "receptor_candidate_regions_provenance": site_config.receptor_candidate_regions_provenance,
        }
        for key, value in candidate_values.items():
            if value is not None:
                summary[key] = _plain(value)
            elif key in candidate_extras:
                # 舊版 in-memory model 可能把新增欄位留在 extra；相容讀取仍保留
                # 明示 provenance，但不為缺少設定的站點建立空候選資料。
                summary[key] = _plain(candidate_extras[key])
        summary["source"] = {
            "files": source_hashes,
            "trajectory_shards": shard_sources,
            "config_hash": static.plan["config_hash"],
            "component_canonical_hashes": _plain(static.plan["component_canonical_hashes"]),
            "scenario_selection": _plain(static.plan["scenario_selection"]),
            "run_code_provenance": _plain(static.plan["code_provenance"]),
            "preview_module_sha256": sha256_file(Path(__file__)),
            "preview_cli_sha256": sha256_file(
                Path(__file__).parents[2] / "scripts" / "build_pilot_preview.py"
            ),
        }
        validate_mplconfigdir()
        phase = "render"
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=parent))
        status = staging.stat()
        staging_identity = (status.st_dev, status.st_ino)
        summary["font"] = _render(
            staging, particles, observations, summary, None if font_path is None else Path(font_path)
        )
        for name, raw in (
            ("particles.csv", _csv_bytes(particles)),
            ("observations.csv", _csv_bytes(observations)),
            (
                "summary.json",
                (
                    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode(),
            ),
            ("README.md", _caption(summary).encode()),
        ):
            _write_exclusive_durable_bytes(staging / name, raw)
        products = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        }
        manifest = {
            "artifact_kind": "pilot-preview-v1",
            "schema_version": PILOT_PREVIEW_SCHEMA_VERSION,
            "run_id": summary["run_id"],
            "source": summary["source"],
            "files": products,
        }
        if nfs_marker_protocol:
            manifest["publish_protocol"] = _NFS_MARKER_PROTOCOL
            manifest["completion_marker"] = _COMPLETION_MARKER_NAME
            manifest["release_provenance"] = {
                **normalized_provenance,
                "storage_gate_snapshot_sha256": storage_gate["sha256"],
                "storage_root_source_token_hash": storage_gate["source_token_hash"],
            }
        else:
            # 標示原有平台原子目錄發布；舊 manifest 沒有此欄位時 reader 仍維持相容性。
            manifest["publish_protocol"] = "exclusive_directory_rename_v1"
        _write_exclusive_durable_bytes(
            staging / "manifest.json",
            _canonical_json_bytes(manifest),
        )
        _validate_staged_preview(staging, manifest)
        if _source_snapshot(root, config, static.plan) != (source_hashes, shard_sources):
            raise PilotPreviewError("preview 來源在產製期間變動，拒絕發布")
        phase = "publish"
        _safe_path(parent)
        _fsync_directory(staging)
        if nfs_marker_protocol:
            gate_after = _verified_storage_gate_evidence(storage_gate_evidence)
            if gate_after != storage_gate:
                raise PilotPreviewError("preview storage gate 在發布前變動")
            lock_descriptor = _acquire_nfs_publish_lock(parent, destination)
            try:
                if _source_snapshot(root, config, static.plan) != (source_hashes, shard_sources):
                    raise PilotPreviewError("preview 來源在鎖定後變動，拒絕發布")
                marker = {
                    "schema_version": _STORAGE_GATE_SCHEMA_VERSION,
                    "artifact_kind": manifest["artifact_kind"],
                    "publish_protocol": _NFS_MARKER_PROTOCOL,
                    "manifest_sha256": sha256_file(staging / "manifest.json"),
                    **normalized_provenance,
                    "storage_gate_snapshot_sha256": storage_gate["sha256"],
                    "storage_root_source_token_hash": storage_gate["source_token_hash"],
                }
                _publish_nfs_marker_preview(
                    staging,
                    destination,
                    parent_identity=parent_identity,
                    marker=marker,
                )
            finally:
                _release_nfs_publish_lock(lock_descriptor)
        else:
            _atomic_exclusive_directory_rename(staging, destination, expected_parent_identity=parent_identity)
        published = True
        try:
            _fsync_directory(parent, expected_identity=parent_identity)
        except Exception:
            raise PilotPreviewError("preview 已發布，但耐久性未確認；保留 final，請勿原地重建") from None
        return manifest
    except FileExistsError:
        raise FileExistsError("preview 目的已存在；未覆寫對方節點") from None
    except PilotPreviewError:
        raise
    except Exception as error:
        raise PilotPreviewError(f"preview {phase} 失敗：{type(error).__name__}") from None
    finally:
        if staging is not None and not published:
            try:
                current_parent = _safe_path(staging.parent).stat()
                current_stage = staging.lstat()
                if (
                    (current_parent.st_dev, current_parent.st_ino) == parent_identity
                    and stat.S_ISDIR(current_stage.st_mode)
                    and (current_stage.st_dev, current_stage.st_ino) == staging_identity
                ):
                    shutil.rmtree(staging)
            except (OSError, PilotPreviewError):
                pass


__all__ = [
    "PILOT_PREVIEW_SCHEMA_VERSION",
    "PilotPreviewError",
    "build_pilot_preview",
    "validate_published_artifact",
    "validate_published_preview",
]
