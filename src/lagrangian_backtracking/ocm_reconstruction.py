"""OCM 已知時間缺口的版本化、稀疏重建核心。

本模組只處理已驗收的 OCM schema 3 月資料，不讀取 raw NetCDF，也不覆寫上游
cache。重建結果是「只包含缺少 UTC rows」的 immutable patch；runtime 可以在讀取
原始月份時以時間鍵查找 patch row，而不需要複製整個 OCM 月檔案。

重建方法分成兩個嚴格邊界：

* 一小時缺口使用缺口兩側的同一 component 做精確雙側線性內插。
* 23--49 小時缺口使用多變量縮放快照低秩基底、M2/S2/K1/O1 諧波、正則化
  AR(2) companion state，以及缺口兩端預測的不確定度／距離融合。模型只讀取
  缺口附近的時間區塊；不把兩年或整月資料一次載入記憶體。

此模組不把任何重建結果宣稱為獨立觀測或正式驗證。metadata 會保留來源 fingerprint、
方法設定、每個 patch row 的 provenance 與品質旗標，方便下游將 observed 與
reconstructed 結果分開統計。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

HOURLY_NS = np.int64(3_600_000_000_000)
RECONSTRUCTION_SCHEMA_VERSION = "ocm_reconstruction_patch_v1"
RECONSTRUCTION_METHOD_ID = "ocm_multivariate_eof_harmonic_state_space_smoother_v1"
ORIGIN_RECONSTRUCTED_SHORT = np.uint8(1)
ORIGIN_RECONSTRUCTED_STATE_SPACE = np.uint8(2)

# quality_flags 是 row-level uint16；連續場內無法合法重建的 cell 會另外以 NaN
# 保存，因此不會把「整列可用」誤寫成「每個 cell 都通過」。
QUALITY_OK = np.uint16(0)
QUALITY_SHORT_LINEAR = np.uint16(1)
QUALITY_STATE_SPACE = np.uint16(2)
QUALITY_WETDRY_UNSUPPORTED = np.uint16(4)
QUALITY_NONFINITE_CELL = np.uint16(8)
QUALITY_ZCOR_ORDER = np.uint16(16)
QUALITY_DIFFUSIVITY_RANGE = np.uint16(32)
QUALITY_NO_TWO_SIDED_SUPPORT = np.uint16(64)

RECONSTRUCTION_ARRAY_NAMES = (
    "time_utc_ns",
    "hvel",
    "vertical_velocity",
    "zcor",
    "elev",
    "wetdry_elem",
    "diffusivity",
    "origin_code",
    "quality_flags",
)
CONTINUOUS_FIELD_NAMES = (
    "hvel",
    "vertical_velocity",
    "zcor",
    "elev",
    "diffusivity",
)


class ReconstructionError(RuntimeError):
    """重建輸入、支援、物理約束或 immutable 發布失敗。"""


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    """控制單一缺口重建的固定設定。

    ``context_hours`` 是每一側最多取用的鄰近觀測時數，不是把整月載入的要求；
    ``feature_block_size`` 限制每次處理的展平特徵數，讓大型節點／層陣列仍以區塊
    流式讀取。``randomized_oversampling`` 與 ``randomized_power_iterations`` 控制只求前
    幾個 EOF modes 的 deterministic randomized SVD；兩者進入 manifest，不能在重啟時
    暗中變更。諧波週期以小時表示，對應半日與主要日潮週期；AR(2) 的 regularization
    只用於穩定狀態係數，不替代缺失資料來源。

    ``diffusivity_max_m2ps`` 可指定已驗收物理上限；未指定時由每個訓練區塊的有限值
    上界決定。缺少雙側支援、訓練值不足或約束無法滿足時，呼叫端應拒絕發布 patch。
    """

    context_hours: int = 96
    min_state_history: int = 8
    feature_block_size: int = 4096
    eof_rank: int = 8
    randomized_oversampling: int = 4
    randomized_power_iterations: int = 1
    ar_order: int = 2
    ridge_lambda: float = 1.0e-4
    harmonic_periods_hours: tuple[float, ...] = (12.4206, 12.0, 23.9345, 25.8193)
    diffusivity_max_m2ps: float | None = None
    source_fingerprint: str = "unspecified"
    method_id: str = RECONSTRUCTION_METHOD_ID

    def __post_init__(self) -> None:
        """在開始讀取大型來源前拒絕無法產生穩定模型的設定。"""

        if self.context_hours < 2 or self.min_state_history < 4:
            raise ValueError("context_hours/min_state_history 必須足以支援雙側 AR 模型")
        if self.feature_block_size < 1 or self.eof_rank < 1:
            raise ValueError("feature_block_size/eof_rank 必須為正整數")
        if self.randomized_oversampling < 2 or self.randomized_power_iterations < 0:
            raise ValueError("randomized EOF oversampling 至少為 2，power iterations 不得為負")
        if self.ar_order != 2:
            raise ValueError("目前版本固定使用二階 AR companion state")
        if self.ridge_lambda <= 0 or not np.isfinite(self.ridge_lambda):
            raise ValueError("ridge_lambda 必須為有限正數")
        if not self.harmonic_periods_hours:
            raise ValueError("至少需要一個潮汐週期")


@dataclass(frozen=True, slots=True)
class GapSpec:
    """一段由逐時 observed time axis 推得的缺口。

    ``missing_times_ns`` 僅包含原始月份真正缺少的 UTC rows；gap 兩側的 observed
    時間必須存在，否則重建會 fail closed。``gap_id`` 寫入每月 metadata 與 domain
    manifest，讓同一時間列的 provenance 不依賴目錄排序。
    """

    gap_id: str
    missing_times_ns: np.ndarray
    left_time_ns: int
    right_time_ns: int

    def __post_init__(self) -> None:
        """確認缺口時間嚴格遞增且兩端相隔整數小時。"""

        times = np.asarray(self.missing_times_ns, dtype=np.int64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("missing_times_ns 必須是非空一維陣列")
        if np.any(np.diff(times) != HOURLY_NS):
            raise ValueError("缺口時間必須以一小時嚴格遞增")
        if int(times[0]) != int(self.left_time_ns) + int(HOURLY_NS):
            raise ValueError("left_time_ns 不是缺口前一個逐時節點")
        if int(times[-1]) != int(self.right_time_ns) - int(HOURLY_NS):
            raise ValueError("right_time_ns 不是缺口後一個逐時節點")


@runtime_checkable
class SequenceSource(Protocol):
    """重建核心需要的最小唯讀、按時間／特徵區塊讀取介面。"""

    @property
    def times_utc_ns(self) -> np.ndarray: ...

    @property
    def field_shapes(self) -> Mapping[str, tuple[int, ...]]: ...

    @property
    def field_dtypes(self) -> Mapping[str, np.dtype[Any]]: ...

    def read_times(
        self,
        times_utc_ns: Sequence[int] | np.ndarray,
        field_name: str,
        *,
        flat_start: int | None = None,
        flat_stop: int | None = None,
    ) -> np.ndarray: ...


def _canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    """以固定 UTF-8 bytes 產生 manifest，避免不同 JSON 排版產生不同 hash。"""

    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """以區塊讀取計算檔案 SHA-256，不把大型 NPY 一次載入記憶體。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SOURCE_GRID_ARRAYS = {
    "node_lon": "source_lon.npy",
    "node_lat": "source_lat.npy",
    "source_depth_m": "source_depth_m.npy",
    "source_node_bottom_index": "source_node_bottom_index.npy",
    "face_nodes_local": "source_face_nodes_local.npy",
    "face_node_count": "source_face_node_count.npy",
    "source_face_global_index": "source_face_global_index.npy",
}
_SOURCE_GRID_DTYPES = {
    "node_lon": np.float64,
    "node_lat": np.float64,
    "source_depth_m": np.float64,
    "source_node_bottom_index": np.int64,
    "face_nodes_local": np.int64,
    "face_node_count": np.int64,
    "source_face_global_index": np.int64,
}


def _source_mesh_fingerprint(grid_root: Path) -> str:
    """由原生經緯度、深度與拓撲建立投影無關的 OCM 網格身分。

    同一個 flow domain 可能服務兩個不同受體投影；因此 fingerprint 不可包含由站點
    中央經線推得的 ``node_xy``。這裡直接雜湊 schema 3 的原生 grid arrays，與 runtime
    對 ``NativeMesh`` 原生欄位所做的計算一致，讓同一份 A 區 patch 可供兩站共用。
    """

    digest = hashlib.sha256()
    for field_name, file_name in _SOURCE_GRID_ARRAYS.items():
        path = grid_root / file_name
        if not path.is_file():
            raise ReconstructionError(f"缺少 OCM grid array：{path}")
        value = np.ascontiguousarray(
            np.asarray(
                np.load(path, mmap_mode="r", allow_pickle=False),
                dtype=_SOURCE_GRID_DTYPES[field_name],
            )
        )
        digest.update(field_name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _source_domain_fingerprint(domain_root: Path, mesh_fingerprint: str) -> str:
    """建立正式來源版本指紋，不讀取數十 GiB 物理陣列的全部 payload。

    指紋納入原生網格、逐月時間軸、月份 metadata bytes，以及各物理 NPY 的相對路徑、
    大小與 header 所記錄的 shape/dtype。若上游 metadata 保存內容 checksum，其 bytes
    也會自然進入本指紋；此設計讓重建來源可稽核，同時避免重複掃描全部流場資料。
    """

    digest = hashlib.sha256()
    digest.update(mesh_fingerprint.encode("ascii"))
    months_root = domain_root / "months"
    for directory in sorted(path for path in months_root.iterdir() if path.is_dir()):
        digest.update(directory.name.encode("ascii"))
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            digest.update(metadata_path.read_bytes())
        for name in ("time_utc_ns", *CONTINUOUS_FIELD_NAMES, "wetdry_elem"):
            path = directory / f"{name}.npy"
            if not path.is_file():
                raise ReconstructionError(f"月份缺少 OCM 欄位：{path}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            digest.update(str(path.relative_to(domain_root)).encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            if name == "time_utc_ns":
                digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


class ArraySequenceSource:
    """以記憶體陣列或 NPY memory-map 提供逐時、展平特徵區塊。

    這個 adapter 只建立時間索引與 shape metadata；``read_times`` 才讀取呼叫端要求的
    rows/features。測試可直接傳 NumPy array，正式建置則由 ``NpyDomainSource`` 使用
    memory-map 並保留同一介面。
    """

    def __init__(self, times_utc_ns: np.ndarray, fields: Mapping[str, np.ndarray]) -> None:
        """建立唯讀來源並確認所有欄位的第一維與 UTC 軸一致。"""

        times = np.asarray(times_utc_ns, dtype=np.int64)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("times_utc_ns 必須是至少兩筆且嚴格遞增的 int64 UTC 軸")
        self._times = times
        self._fields = {name: np.asarray(value) for name, value in fields.items()}
        required = set(CONTINUOUS_FIELD_NAMES) | {"wetdry_elem"}
        if missing := required - set(self._fields):
            raise ValueError(f"來源缺少欄位：{sorted(missing)}")
        for name, value in self._fields.items():
            if value.ndim < 1 or value.shape[0] != times.size:
                raise ValueError(f"{name} 第一維必須等於 time count")
        self._positions = {int(time): index for index, time in enumerate(times)}
        # 記憶體來源只供單元測試與 blocked-mask 驗證；這個 fingerprint 綁定時間軸與
        # 欄位拓撲，不冒充正式 OCM grid identity。正式建置會改用 NpyDomainSource
        # 對原生 grid arrays 計算的投影無關 fingerprint。
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(times).tobytes(order="C"))
        for name in sorted(self._fields):
            value = self._fields[name]
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(repr(tuple(value.shape[1:])).encode("ascii"))
        synthetic_fingerprint = f"synthetic-{digest.hexdigest()}"
        self.mesh_fingerprint = synthetic_fingerprint
        self.source_fingerprint = synthetic_fingerprint

    @property
    def times_utc_ns(self) -> np.ndarray:
        """回傳唯讀 UTC 軸副本，避免 caller 改寫時間索引。"""

        return self._times.copy()

    @property
    def field_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """回傳不含時間軸的資料 shape。"""

        return {name: tuple(value.shape[1:]) for name, value in self._fields.items()}

    @property
    def field_dtypes(self) -> Mapping[str, np.dtype[Any]]:
        """回傳原始欄位 dtype；patch 的 wetdry 會依缺值語意轉為浮點。"""

        return {name: value.dtype for name, value in self._fields.items()}

    def read_times(
        self,
        times_utc_ns: Sequence[int] | np.ndarray,
        field_name: str,
        *,
        flat_start: int | None = None,
        flat_stop: int | None = None,
    ) -> np.ndarray:
        """只取指定 UTC rows 與展平特徵區塊；缺 row 時明確拒絕而非近似補值。"""

        if field_name not in self._fields:
            raise KeyError(field_name)
        requested = np.asarray(times_utc_ns, dtype=np.int64)
        indices = []
        for time in requested:
            try:
                indices.append(self._positions[int(time)])
            except KeyError as exc:
                raise ReconstructionError(f"來源沒有要求的 UTC row：{int(time)}") from exc
        values = self._fields[field_name][np.asarray(indices, dtype=np.int64)]
        flattened = values.reshape(values.shape[0], -1)
        start = 0 if flat_start is None else int(flat_start)
        stop = flattened.shape[1] if flat_stop is None else int(flat_stop)
        if start < 0 or stop < start or stop > flattened.shape[1]:
            raise ValueError(f"{field_name} 的 flat slice 超出範圍")
        return np.asarray(flattened[:, start:stop])

    def masked(self, missing_times_ns: Sequence[int] | np.ndarray) -> ArraySequenceSource:
        """建立 blocked-mask 驗證來源；原始完整陣列不被修改。"""

        missing = {int(value) for value in np.asarray(missing_times_ns, dtype=np.int64)}
        keep = np.asarray([int(time) not in missing for time in self._times], dtype=bool)
        return ArraySequenceSource(
            self._times[keep],
            {name: value[keep] for name, value in self._fields.items()},
        )


@dataclass(slots=True)
class _NpyMonth:
    """一個月份的 memory-map 與 UTC row 查找表。"""

    directory: Path
    times: np.ndarray
    row_by_time: dict[int, int]
    arrays: dict[str, np.ndarray]


class NpyDomainSource:
    """以月份 memory-map 讀取 OCM schema 3 domain，避免整月／兩年常駐記憶體。

    ``domain_root`` 必須是 ``<flow_id>/``，其下有 ``months/YYYYMM``。每個月的時間軸
    與大型資料陣列只在真正讀取某個缺口的 context block 時開啟；時間 row 索引只有數
    千筆 int64，不會複製 74 GiB 的 OCM payload。
    """

    def __init__(self, domain_root: str | Path) -> None:
        """掃描月份目錄、建立 UTC→(month,row) 索引，並檢查欄位 headers。"""

        root = Path(domain_root)
        months_root = root / "months"
        if not months_root.is_dir():
            raise ReconstructionError(f"找不到 OCM months 目錄：{months_root}")
        self.domain_root = root
        self.flow_id = root.name
        self.mesh_fingerprint = _source_mesh_fingerprint(root / "grid")
        self.source_fingerprint = _source_domain_fingerprint(root, self.mesh_fingerprint)
        self._months: list[_NpyMonth] = []
        self._location_by_time: dict[int, tuple[_NpyMonth, int]] = {}
        names = (*CONTINUOUS_FIELD_NAMES, "wetdry_elem")
        for directory in sorted(path for path in months_root.iterdir() if path.is_dir()):
            time_path = directory / "time_utc_ns.npy"
            if not time_path.is_file():
                continue
            times = np.load(time_path, mmap_mode="r")
            times = np.asarray(times, dtype=np.int64)
            if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0):
                raise ReconstructionError(f"月份 UTC 軸不合法：{time_path}")
            arrays: dict[str, np.ndarray] = {}
            for name in names:
                path = directory / f"{name}.npy"
                if not path.is_file():
                    raise ReconstructionError(f"月份缺少 OCM 欄位：{path}")
                array = np.load(path, mmap_mode="r")
                if array.ndim < 1 or array.shape[0] != times.size:
                    raise ReconstructionError(f"月份 {name} shape 與時間軸不符：{path}")
                arrays[name] = array
            month = _NpyMonth(
                directory=directory,
                times=times,
                row_by_time={int(time): index for index, time in enumerate(times)},
                arrays=arrays,
            )
            self._months.append(month)
            # OCM 月檔在月界線包含重疊 halo。依專案既定 canonical time-axis 契約，
            # 以月份目錄排序後遇到相同 UTC 時採後一個月（prefer-last），而不是把
            # 合法 halo 誤判成重複資料或任意採第一筆。
            for row, time in enumerate(times):
                self._location_by_time[int(time)] = (month, row)
        if not self._months:
            raise ReconstructionError(f"沒有可讀取的 OCM 月份：{months_root}")
        self._times = np.asarray(sorted(self._location_by_time), dtype=np.int64)
        if self._times.size < 2 or np.any(np.diff(self._times) <= 0):
            raise ReconstructionError("跨月份 canonical UTC 軸不足或未排序")
        first = self._months[0]
        self._shapes = {name: tuple(array.shape[1:]) for name, array in first.arrays.items()}
        self._dtypes = {name: array.dtype for name, array in first.arrays.items()}
        for month in self._months[1:]:
            if {name: tuple(array.shape[1:]) for name, array in month.arrays.items()} != self._shapes:
                raise ReconstructionError("跨月份 OCM 欄位 shape 不一致")

    @property
    def times_utc_ns(self) -> np.ndarray:
        """回傳跨月份的 observed UTC 軸，不複製任何大型欄位。"""

        return self._times.copy()

    @property
    def field_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """回傳所有月份一致的非時間維度。"""

        return dict(self._shapes)

    @property
    def field_dtypes(self) -> Mapping[str, np.dtype[Any]]:
        """回傳原始 NPY dtype。"""

        return dict(self._dtypes)

    def _locate(self, time_ns: int) -> tuple[_NpyMonth, int]:
        """依 canonical prefer-last 索引定位單一 UTC row。"""

        location = self._location_by_time.get(int(time_ns))
        if location is None:
            raise ReconstructionError(f"來源沒有要求的 UTC row：{time_ns}")
        return location

    def read_times(
        self,
        times_utc_ns: Sequence[int] | np.ndarray,
        field_name: str,
        *,
        flat_start: int | None = None,
        flat_stop: int | None = None,
    ) -> np.ndarray:
        """分月份、按特徵區塊讀取 rows；不把月份陣列 concatenate 成完整副本。"""

        if field_name not in self._shapes:
            raise KeyError(field_name)
        requested = np.asarray(times_utc_ns, dtype=np.int64)
        total_features = int(np.prod(self._shapes[field_name], dtype=np.int64))
        start = 0 if flat_start is None else int(flat_start)
        stop = total_features if flat_stop is None else int(flat_stop)
        if start < 0 or stop < start or stop > total_features:
            raise ValueError(f"{field_name} 的 flat slice 超出範圍")
        output = np.empty((requested.size, stop - start), dtype=self._dtypes[field_name])
        for output_index, time in enumerate(requested):
            month, row = self._locate(int(time))
            flattened = np.asarray(month.arrays[field_name][row]).reshape(-1)
            output[output_index] = flattened[start:stop]
        return output


def discover_hourly_gaps(
    times_utc_ns: Sequence[int] | np.ndarray,
    *,
    flow_id: str = "flow",
    expected_start_ns: int | None = None,
    expected_end_ns: int | None = None,
) -> tuple[GapSpec, ...]:
    """從 observed UTC 軸找出逐時缺口，並保留沒有雙側支援的 edge gap。

    edge gap 不會被悄悄丟掉；後續 ``build_reconstruction_patch`` 會明確拒絕它，因為
    線性內插及雙向 state-space 都需要真實兩側支援。``expected_start_ns``／``end``
    只應來自已驗收的 canonical time contract，不由本函式猜測兩年外資料。
    """

    times = np.asarray(times_utc_ns, dtype=np.int64)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
        raise ReconstructionError("observed UTC 軸必須嚴格遞增且至少有兩筆")
    gaps: list[GapSpec] = []
    if expected_start_ns is not None and int(expected_start_ns) < int(times[0]):
        missing = np.arange(int(expected_start_ns), int(times[0]), int(HOURLY_NS), dtype=np.int64)
        if missing.size:
            gaps.append(
                GapSpec(
                    gap_id=f"{flow_id}_{int(missing[0])}_{int(missing[-1])}",
                    missing_times_ns=missing,
                    left_time_ns=int(missing[0]) - int(HOURLY_NS),
                    right_time_ns=int(times[0]),
                )
            )
    for before, after in zip(times[:-1], times[1:], strict=True):
        difference = int(after) - int(before)
        if difference <= int(HOURLY_NS):
            continue
        if difference % int(HOURLY_NS):
            raise ReconstructionError(f"UTC 缺口不是整數小時：{int(before)}→{int(after)}")
        missing = np.arange(int(before) + int(HOURLY_NS), int(after), int(HOURLY_NS), dtype=np.int64)
        gaps.append(
            GapSpec(
                gap_id=f"{flow_id}_{int(missing[0])}_{int(missing[-1])}",
                missing_times_ns=missing,
                left_time_ns=int(before),
                right_time_ns=int(after),
            )
        )
    if expected_end_ns is not None and int(expected_end_ns) > int(times[-1]):
        missing = np.arange(
            int(times[-1]) + int(HOURLY_NS),
            int(expected_end_ns) + 1,
            int(HOURLY_NS),
            dtype=np.int64,
        )
        if missing.size:
            gaps.append(
                GapSpec(
                    gap_id=f"{flow_id}_{int(missing[0])}_{int(missing[-1])}",
                    missing_times_ns=missing,
                    left_time_ns=int(times[-1]),
                    right_time_ns=int(missing[-1]) + int(HOURLY_NS),
                )
            )
    return tuple(gaps)


def _context_times(source: SequenceSource, gap: GapSpec, context_hours: int) -> tuple[np.ndarray, np.ndarray]:
    """取得 gap 兩側連續 observed 時間；遇到其他缺口時停止，不跨另一個缺口訓練。"""

    observed = source.times_utc_ns
    positions = {int(time): index for index, time in enumerate(observed)}
    left: list[int] = []
    candidate = int(gap.left_time_ns)
    for _ in range(context_hours):
        if candidate not in positions:
            break
        left.append(candidate)
        candidate -= int(HOURLY_NS)
    right: list[int] = []
    candidate = int(gap.right_time_ns)
    for _ in range(context_hours):
        if candidate not in positions:
            break
        right.append(candidate)
        candidate += int(HOURLY_NS)
    return np.asarray(sorted(left), dtype=np.int64), np.asarray(right, dtype=np.int64)


def _harmonic_design(times_ns: np.ndarray, periods_hours: Sequence[float]) -> np.ndarray:
    """建立固定截距、線性趨勢與四個潮汐週期的設計矩陣。"""

    if times_ns.size == 0:
        return np.empty((0, 2 + 2 * len(periods_hours)), dtype=np.float64)
    hours = (times_ns.astype(np.float64) - float(times_ns[0])) / float(HOURLY_NS)
    columns = [np.ones(hours.size), hours / max(float(hours.size), 1.0)]
    for period in periods_hours:
        angle = 2.0 * np.pi * hours / float(period)
        columns.extend((np.cos(angle), np.sin(angle)))
    return np.column_stack(columns)


def _ridge_fit(design: np.ndarray, values: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """對一個特徵區塊做正則化最小平方法，輸出設計係數。"""

    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += ridge_lambda
    return np.linalg.solve(gram, design.T @ values)


def _ar_coefficients(scores: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """以每個 EOF score 的時間序列估計 AR(2) companion 係數。"""

    if scores.shape[0] <= 2:
        return np.zeros((scores.shape[1], 2), dtype=np.float64)
    coefficients = np.empty((scores.shape[1], 2), dtype=np.float64)
    # 各 EOF mode 是獨立的 scalar companion state；不可把所有 mode 展成同一個
    # 2*rank design 後只取前兩欄，否則第二個以上的 mode 係數會被錯誤丟棄。
    for mode in range(scores.shape[1]):
        design = np.column_stack((scores[1:-1, mode], scores[:-2, mode]))
        target = scores[2:, mode]
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += ridge_lambda
        coefficients[mode] = np.linalg.solve(gram, design.T @ target)
    return coefficients


def _forecast_ar2(history: np.ndarray, coefficients: np.ndarray, steps: int) -> np.ndarray:
    """以 AR(2) companion state 向前預測 score，維持 deterministic row order。"""

    if steps <= 0:
        return np.empty((0, history.shape[1]), dtype=np.float64)
    if history.shape[0] < 2:
        return np.repeat(history[-1:, :], steps, axis=0)
    previous = history[-2:].copy()
    output = np.empty((steps, history.shape[1]), dtype=np.float64)
    for index in range(steps):
        next_value = coefficients[:, 0] * previous[1] + coefficients[:, 1] * previous[0]
        output[index] = next_value
        previous[0], previous[1] = previous[1], next_value
    return output


def _corrcoef(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    """計算有限且有變異的向量相關係數；無法計算時回傳 JSON-safe null。"""

    mask = np.isfinite(reference) & np.isfinite(estimate)
    if int(mask.sum()) < 2:
        return None
    left = reference[mask]
    right = estimate[mask]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _deterministic_truncated_svd(
    residual: np.ndarray,
    *,
    rank: int,
    oversampling: int,
    power_iterations: int,
    seed_material: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """以固定 seed 的 randomized SVD 求低秩 EOF，避免正式網格的完整 SVD 成本。

    真實 feature block 約有 4,096 欄、時間 context 約 192 列；完整 SVD 會為每一缺口
    反覆形成 192 階分解，四區累積成本過高。此處只求發布設定需要的前 ``rank`` modes，
    seed 由欄位與 block 邊界的穩定字串產生，並保留一次 power iteration 以改善潮汐之外
    的較弱模態。小型測試或特徵數接近 rank 時仍走 exact SVD，作為數值參考路徑。
    """

    maximum_rank = min(residual.shape)
    target_rank = min(int(rank), maximum_rank)
    sketch_rank = min(maximum_rank, target_rank + int(oversampling))
    if sketch_rank >= maximum_rank:
        u, singular, vt = np.linalg.svd(residual, full_matrices=False)
        return u[:, :target_rank], singular[:target_rank], vt[:target_rank, :]
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=False,
    )
    generator = np.random.default_rng(seed)
    projection = generator.standard_normal((residual.shape[1], sketch_rank))
    basis, _ = np.linalg.qr(residual @ projection, mode="reduced")
    for _ in range(int(power_iterations)):
        basis, _ = np.linalg.qr(residual @ (residual.T @ basis), mode="reduced")
    compressed = basis.T @ residual
    small_u, singular, vt = np.linalg.svd(compressed, full_matrices=False)
    u = basis @ small_u
    return u[:, :target_rank], singular[:target_rank], vt[:target_rank, :]


def _state_space_block(
    source: SequenceSource,
    field_name: str,
    gap: GapSpec,
    left_times: np.ndarray,
    right_times: np.ndarray,
    config: ReconstructionConfig,
    flat_start: int,
    flat_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """產生 long-gap 預測、row flags 與已讀取的訓練區塊。

    第三個回傳值讓 Kz 物理範圍 gate 重用同一批雙側觀測，不再對 NFS 月檔做第二次
    完全相同的 strided read；其他欄位可直接忽略它。
    """

    left = source.read_times(
        left_times, field_name, flat_start=flat_start, flat_stop=flat_stop
    ).astype(np.float64)
    right = source.read_times(
        right_times, field_name, flat_start=flat_start, flat_stop=flat_stop
    ).astype(np.float64)
    if left.shape[0] < config.min_state_history or right.shape[0] < config.min_state_history:
        raise ReconstructionError(
            f"{field_name} gap={gap.gap_id} 的雙側 state history 不足，不能以單側預測冒充重建"
        )
    training_times = np.concatenate((left_times, right_times))
    training = np.concatenate((left, right), axis=0)
    missing_times = gap.missing_times_ns
    harmonic_training = _harmonic_design(training_times, config.harmonic_periods_hours)
    harmonic_missing = _harmonic_design(
        np.concatenate((training_times[:1], missing_times)), config.harmonic_periods_hours
    )[1:]
    finite = np.isfinite(training)
    # 真實 OCM 某些 cell 可能只在部分時間有效；模型為了保留矩陣形狀會暫以均值
    # 佔位後做 EOF，但這些 feature 仍會被標成 invalid 並輸出 NaN，絕不把佔位值
    # 發布成 reconstructed field。
    counts = finite.sum(axis=0)
    safe_count = np.maximum(counts, 1)
    means = np.where(finite, training, 0.0).sum(axis=0) / safe_count
    filled = np.where(finite, training, means)
    scales = np.std(filled, axis=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    scaled = (filled - means) / scales
    beta = _ridge_fit(harmonic_training, scaled, config.ridge_lambda)
    baseline = harmonic_training @ beta
    missing_baseline = harmonic_missing @ beta
    residual = scaled - baseline
    # 對每一 block 做 snapshot-POD/EOF；U 是時間基底，因此之後同一個 temporal
    # state 可以逐 spatial block 重建，不必建立整個兩年 × 全網格矩陣。
    u, singular, loadings = _deterministic_truncated_svd(
        residual,
        rank=config.eof_rank,
        oversampling=config.randomized_oversampling,
        power_iterations=config.randomized_power_iterations,
        seed_material=f"{field_name}:{flat_start}:{flat_stop}",
    )
    scores = u * singular
    left_scores = scores[: left.shape[0]]
    right_scores = scores[left.shape[0] :]
    left_ar = _ar_coefficients(left_scores, config.ridge_lambda)
    right_ar = _ar_coefficients(right_scores[::-1], config.ridge_lambda)
    left_forecast_scores = _forecast_ar2(left_scores, left_ar, missing_times.size)
    right_forecast_scores_reversed = _forecast_ar2(right_scores[::-1], right_ar, missing_times.size)
    right_forecast_scores = right_forecast_scores_reversed[::-1]
    left_residual = left_forecast_scores @ loadings
    right_residual = right_forecast_scores @ loadings
    uncertainty_left = np.sqrt(np.mean(np.square(left_scores - left_scores.mean(axis=0)), axis=0)).mean()
    uncertainty_right = np.sqrt(np.mean(np.square(right_scores - right_scores.mean(axis=0)), axis=0)).mean()
    left_distance = np.arange(1, missing_times.size + 1, dtype=np.float64)
    right_distance = np.arange(missing_times.size, 0, -1, dtype=np.float64)
    left_weight = 1.0 / (left_distance + uncertainty_left + 1.0e-9)
    right_weight = 1.0 / (right_distance + uncertainty_right + 1.0e-9)
    prediction_scaled = missing_baseline + (
        left_weight[:, None] * left_residual + right_weight[:, None] * right_residual
    ) / (left_weight + right_weight)[:, None]
    prediction = prediction_scaled * scales + means
    invalid = counts < training.shape[0]
    if np.any(invalid):
        prediction[:, invalid] = np.nan
    row_flags = np.zeros(missing_times.size, dtype=np.uint16)
    row_flags |= QUALITY_STATE_SPACE
    if np.any(invalid):
        row_flags |= QUALITY_NONFINITE_CELL
    return prediction, row_flags, training


def _linear_short_block(
    source: SequenceSource,
    field_name: str,
    gap: GapSpec,
    flat_start: int,
    flat_stop: int,
) -> np.ndarray:
    """以缺口前後 exact rows 做 component-wise 1 小時線性內插。"""

    supports = source.read_times(
        np.asarray([gap.left_time_ns, gap.right_time_ns], dtype=np.int64),
        field_name,
        flat_start=flat_start,
        flat_stop=flat_stop,
    ).astype(np.float64)
    alpha = 0.5
    prediction = (supports[0] + alpha * (supports[1] - supports[0]))[None, :]
    valid = np.isfinite(supports).all(axis=0)
    prediction[:, ~valid] = np.nan
    return prediction


def _wetdry_block(
    source: SequenceSource,
    gap: GapSpec,
    flat_start: int,
    flat_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """只重建兩端狀態一致的 wet/dry cell；混合狀態以 NaN 與旗標表示。"""

    supports = source.read_times(
        np.asarray([gap.left_time_ns, gap.right_time_ns], dtype=np.int64),
        "wetdry_elem",
        flat_start=flat_start,
        flat_stop=flat_stop,
    ).astype(np.float64)
    same = np.isfinite(supports).all(axis=0) & (supports[0] == supports[1])
    prediction = np.full((gap.missing_times_ns.size, supports.shape[1]), np.nan, dtype=np.float32)
    prediction[:, same] = supports[0, same].astype(np.float32)
    flags = np.zeros(gap.missing_times_ns.size, dtype=np.uint16)
    if np.any(~same):
        flags |= QUALITY_WETDRY_UNSUPPORTED
    return prediction, flags


def _enforce_constraints(
    field_name: str,
    prediction: np.ndarray,
    source: SequenceSource,
    config: ReconstructionConfig,
    *,
    flat_start: int,
    flat_stop: int,
    training_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """套用 persistent-wet、zcor 垂向順序與 diffusivity 物理範圍 gate。"""

    values = prediction.copy()
    flags = np.zeros(values.shape[0], dtype=np.uint16)
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        flags |= QUALITY_NONFINITE_CELL
    if field_name == "diffusivity":
        # 正式建置由 caller 以同一 feature block 傳入鄰近訓練 rows，避免為了
        # constraint gate 再把全域網格讀進記憶體。若舊 caller 沒提供訓練值，
        # 仍只套用明示的非負／設定上限，而不猜測全域 physical range。
        training = np.asarray(training_values, dtype=np.float64) if training_values is not None else None
        finite_training = (
            training[np.isfinite(training)]
            if training is not None
            else np.empty(0, dtype=np.float64)
        )
        upper = (
            float(config.diffusivity_max_m2ps)
            if config.diffusivity_max_m2ps is not None
            else (
                float(np.max(finite_training))
                if finite_training.size
                else np.inf
            )
        )
        invalid = (
            (values < 0.0) | (values > upper)
            if np.isfinite(upper)
            else (values < 0.0)
        )
        invalid &= np.isfinite(values)
        if np.any(invalid):
            values[invalid] = np.nan
            flags |= QUALITY_DIFFUSIVITY_RANGE
    if field_name == "zcor" and values.ndim == 3 and values.shape[2] > 1:
        # OCM 水柱在各節點 bottom index 以下原本就可為 NaN；只檢查相鄰兩層皆有限
        # 的有效水柱段。若某一節點次序錯誤，只封鎖該節點，不可把整個時刻／全域
        # zcor 清成 NaN，否則遠端單一乾點會讓所有粒子同時停止。
        adjacent_finite = np.isfinite(values[:, :, 1:]) & np.isfinite(values[:, :, :-1])
        ordered_pairs = (~adjacent_finite) | (np.diff(values, axis=2) > 0.0)
        invalid_nodes = ~np.all(ordered_pairs, axis=2)
        if np.any(invalid_nodes):
            values[invalid_nodes] = np.nan
            flags[np.any(invalid_nodes, axis=1)] |= QUALITY_ZCOR_ORDER
    return values, flags


def _reconstruct_gap(
    source: SequenceSource,
    gap: GapSpec,
    config: ReconstructionConfig,
) -> dict[str, np.ndarray]:
    """重建一段 gap 的所有 patch arrays，尚未寫入磁碟。"""

    left_times, right_times = _context_times(source, gap, config.context_hours)
    if left_times.size == 0 or right_times.size == 0:
        raise ReconstructionError(f"gap={gap.gap_id} 沒有雙側 exact observed 支援")
    is_short = gap.missing_times_ns.size == 1
    arrays: dict[str, np.ndarray] = {
        "time_utc_ns": gap.missing_times_ns.copy(),
        "origin_code": np.full(
            gap.missing_times_ns.size,
            ORIGIN_RECONSTRUCTED_SHORT if is_short else ORIGIN_RECONSTRUCTED_STATE_SPACE,
            dtype=np.uint8,
        ),
        "quality_flags": np.zeros(gap.missing_times_ns.size, dtype=np.uint16),
    }
    if not is_short and gap.missing_times_ns.size < 23:
        raise ReconstructionError(f"非短缺口長度不符合 23--49 小時契約：{gap.gap_id}")
    for field_name in CONTINUOUS_FIELD_NAMES:
        shape = source.field_shapes[field_name]
        feature_count = int(np.prod(shape, dtype=np.int64))
        if is_short:
            # 單小時缺口只需兩個 endpoint，最大 A 區兩列 hvel 約 50 MiB，可安全一次
            # 連續讀取。這避免對 NFS 月檔依 4,096 features 重複數千次小讀取；長缺口
            # 仍維持下方 block path，避免把 192 個 context rows 全載入記憶體。
            supports = source.read_times(
                np.asarray([gap.left_time_ns, gap.right_time_ns], dtype=np.int64),
                field_name,
            ).astype(np.float64)
            output = (supports[0] + 0.5 * (supports[1] - supports[0]))[None, :]
            output[:, ~np.isfinite(supports).all(axis=0)] = np.nan
            arrays["quality_flags"] |= QUALITY_SHORT_LINEAR
            if field_name == "diffusivity":
                output, constraint_flags = _enforce_constraints(
                    field_name,
                    output,
                    source,
                    config,
                    flat_start=0,
                    flat_stop=feature_count,
                    training_values=supports,
                )
                arrays["quality_flags"] |= constraint_flags
        else:
            output = np.empty((gap.missing_times_ns.size, feature_count), dtype=np.float64)
            for start in range(0, feature_count, config.feature_block_size):
                stop = min(feature_count, start + config.feature_block_size)
                block, flags, training_block = _state_space_block(
                    source,
                    field_name,
                    gap,
                    left_times,
                    right_times,
                    config,
                    start,
                    stop,
                )
                block = block.reshape(gap.missing_times_ns.size, -1)
                if field_name == "diffusivity":
                    block_reshaped, block_constraint_flags = _enforce_constraints(
                        field_name,
                        block,
                        source,
                        config,
                        flat_start=start,
                        flat_stop=stop,
                        training_values=training_block,
                    )
                    block = block_reshaped.reshape(gap.missing_times_ns.size, -1)
                    arrays["quality_flags"] |= block_constraint_flags
                output[:, start:stop] = block
                arrays["quality_flags"] |= flags
        shaped = output.reshape((gap.missing_times_ns.size, *shape))
        # zcor 的層序檢查需要看到完整 node×layer 軸；其他連續場只需保留 finite
        # provenance。diffusivity 已在 block 階段用鄰近訓練 range 檢查，這裡不重讀全域。
        if field_name == "zcor":
            shaped, constraint_flags = _enforce_constraints(
                field_name,
                shaped,
                source,
                config,
                flat_start=0,
                flat_stop=feature_count,
            )
        else:
            constraint_flags = np.zeros(gap.missing_times_ns.size, dtype=np.uint16)
            if not np.isfinite(shaped).all():
                constraint_flags |= QUALITY_NONFINITE_CELL
        arrays[field_name] = shaped.astype(source.field_dtypes[field_name], copy=False)
        arrays["quality_flags"] |= constraint_flags
    wet_shape = source.field_shapes["wetdry_elem"]
    wet_features = int(np.prod(wet_shape, dtype=np.int64))
    wet_output = np.empty((gap.missing_times_ns.size, wet_features), dtype=np.float32)
    if is_short:
        wet_output, flags = _wetdry_block(source, gap, 0, wet_features)
        arrays["quality_flags"] |= flags
    else:
        for start in range(0, wet_features, config.feature_block_size):
            stop = min(wet_features, start + config.feature_block_size)
            block, flags = _wetdry_block(source, gap, start, stop)
            wet_output[:, start:stop] = block
            arrays["quality_flags"] |= flags
    arrays["wetdry_elem"] = wet_output.reshape((gap.missing_times_ns.size, *wet_shape))
    return arrays


def _array_metadata(path: Path, array: np.ndarray) -> dict[str, Any]:
    """回傳 patch array 的 shape、dtype、bytes 與 checksum。"""

    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": np.dtype(array.dtype).name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_atomic_month(
    output_root: Path,
    flow_id: str,
    month: str,
    arrays: Mapping[str, np.ndarray],
    *,
    gap_ids: Sequence[str],
    config: ReconstructionConfig,
    source_fingerprint: str,
) -> dict[str, Any]:
    """把單月 sparse patch 寫入 hidden partial，再以 atomic rename 發布。"""

    months_root = output_root / flow_id / "months"
    months_root.mkdir(parents=True, exist_ok=True)
    destination = months_root / month
    if destination.exists():
        raise ReconstructionError(f"immutable patch 目的地已存在，拒絕覆寫：{destination}")
    partial = months_root / f".{month}.partial-{uuid.uuid4().hex}"
    partial.mkdir()
    try:
        for name in RECONSTRUCTION_ARRAY_NAMES:
            if name not in arrays:
                raise ReconstructionError(f"patch 缺少固定欄位：{name}")
            np.save(partial / f"{name}.npy", np.asarray(arrays[name]))
        metadata: dict[str, Any] = {
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "flow_id": flow_id,
            "month": month,
            "method": config.method_id,
            "gap_ids": list(gap_ids),
            "source_fingerprint": source_fingerprint,
            "configuration": asdict(config),
            "constraints": {
                "persistent_wet_only": True,
                "wetdry_semantics": "0=wet,1=dry,NaN=not_persistent",
                "zcor_strict_increasing": True,
                "diffusivity_nonnegative": True,
                "observed_arrays_untouched": True,
            },
            "arrays": {},
            "quality_counts": {
                str(int(flag)): int(np.count_nonzero(np.asarray(arrays["quality_flags"]) & flag))
                for flag in (
                    QUALITY_SHORT_LINEAR,
                    QUALITY_STATE_SPACE,
                    QUALITY_WETDRY_UNSUPPORTED,
                    QUALITY_NONFINITE_CELL,
                    QUALITY_ZCOR_ORDER,
                    QUALITY_DIFFUSIVITY_RANGE,
                )
            },
        }
        for name in RECONSTRUCTION_ARRAY_NAMES:
            path = partial / f"{name}.npy"
            metadata["arrays"][name] = _array_metadata(path, np.asarray(arrays[name]))
        metadata_bytes = _canonical_json_bytes(metadata)
        (partial / "metadata.json").write_bytes(metadata_bytes)
        (partial / "metadata.json.sha256").write_text(
            hashlib.sha256(metadata_bytes).hexdigest() + "\n", encoding="ascii"
        )
        # 先 fsync metadata，再將整個月目錄一次發布；若 rename 失敗，partial 會被清掉。
        with (partial / "metadata.json").open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, destination)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return metadata


def _month_for_ns(time_ns: int) -> str:
    """將 UTC epoch nanoseconds 轉為 patch 固定的 YYYYMM 目錄。"""

    return datetime.fromtimestamp(int(time_ns) / 1_000_000_000, tz=UTC).strftime("%Y%m")


def _write_domain_manifest(
    output_root: Path,
    flow_id: str,
    month_metadata: Sequence[Mapping[str, Any]],
    *,
    config: ReconstructionConfig,
    source_fingerprint: str,
    mesh_fingerprint: str,
) -> dict[str, Any]:
    """寫 domain manifest 及其 sidecar hash；manifest 不是可覆寫的結果索引。"""

    domain_root = output_root / flow_id
    path = domain_root / "reconstruction-manifest.json"
    if path.exists():
        raise ReconstructionError(f"domain manifest 已存在，拒絕覆寫：{path}")
    document: dict[str, Any] = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "flow_id": flow_id,
        "method": config.method_id,
        "source_fingerprint": source_fingerprint,
        "mesh_fingerprint": mesh_fingerprint,
        "configuration": asdict(config),
        "months": [
            {
                "month": item["month"],
                "gap_ids": item["gap_ids"],
                "metadata_sha256": hashlib.sha256(_canonical_json_bytes(item)).hexdigest(),
            }
            for item in sorted(month_metadata, key=lambda item: str(item["month"]))
        ],
    }
    data = _canonical_json_bytes(document)
    path.write_bytes(data)
    (domain_root / "reconstruction-manifest.json.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n", encoding="ascii"
    )
    return document


def build_reconstruction_patch(
    source: SequenceSource,
    output_root: str | Path,
    *,
    flow_id: str,
    config: ReconstructionConfig | None = None,
    expected_start_ns: int | None = None,
    expected_end_ns: int | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """建立一個 flow domain 的 sparse immutable OCM reconstruction patch。

    所有 gap 先完成支援與目的地檢查，才開始寫入；每月 patch 只包含缺少的 UTC rows。
    函式若中途失敗會刪除當次 partial 目錄，但不觸碰已存在的 source 或其他月份。
    ``progress_callback`` 只接收 gap／month 的小型 JSON-safe 摘要，供 SERVER log 顯示
    真正進度；它不接收或改寫大型物理陣列，也不影響 deterministic 結果。
    """

    settings = config or ReconstructionConfig()
    source_fingerprint = settings.source_fingerprint
    if source_fingerprint == "unspecified":
        source_fingerprint = str(getattr(source, "source_fingerprint", ""))
    if not source_fingerprint or source_fingerprint == "unspecified":
        raise ReconstructionError("source_fingerprint 不可為 unspecified")
    if settings.source_fingerprint != source_fingerprint:
        settings = replace(settings, source_fingerprint=source_fingerprint)
    mesh_fingerprint = str(getattr(source, "mesh_fingerprint", ""))
    if not mesh_fingerprint:
        raise ReconstructionError("來源缺少 mesh_fingerprint，拒絕發布 patch")
    root = Path(output_root)
    gaps = discover_hourly_gaps(
        source.times_utc_ns,
        flow_id=flow_id,
        expected_start_ns=expected_start_ns,
        expected_end_ns=expected_end_ns,
    )
    if not gaps:
        raise ReconstructionError(f"flow={flow_id} 沒有需要重建的逐時缺口")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "gaps_discovered",
                "flow_id": flow_id,
                "gap_count": len(gaps),
                "missing_row_count": int(sum(gap.missing_times_ns.size for gap in gaps)),
            }
        )
    source_times = {int(value) for value in source.times_utc_ns}
    for gap in gaps:
        if gap.left_time_ns not in source_times or gap.right_time_ns not in source_times:
            raise ReconstructionError(f"gap={gap.gap_id} 缺少雙側 exact support，拒絕重建")
    months = sorted({_month_for_ns(int(time)) for gap in gaps for time in gap.missing_times_ns})
    for month in months:
        destination = root / flow_id / "months" / month
        if destination.exists():
            raise ReconstructionError(f"immutable patch 目的地已存在，拒絕覆寫：{destination}")
    by_month: dict[str, dict[str, list[Any]]] = {}
    for gap_index, gap in enumerate(gaps, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "gap_start",
                    "flow_id": flow_id,
                    "gap_index": gap_index,
                    "gap_count": len(gaps),
                    "gap_id": gap.gap_id,
                    "missing_row_count": int(gap.missing_times_ns.size),
                }
            )
        arrays = _reconstruct_gap(source, gap, settings)
        for row, time in enumerate(np.asarray(arrays["time_utc_ns"], dtype=np.int64)):
            month = _month_for_ns(int(time))
            record = by_month.setdefault(month, {name: [] for name in RECONSTRUCTION_ARRAY_NAMES})
            for name in RECONSTRUCTION_ARRAY_NAMES:
                record[name].append(np.asarray(arrays[name])[row])
            record.setdefault("_gap_ids", []).append(gap.gap_id)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "gap_complete",
                    "flow_id": flow_id,
                    "gap_index": gap_index,
                    "gap_count": len(gaps),
                    "gap_id": gap.gap_id,
                }
            )
    month_metadata: list[Mapping[str, Any]] = []
    for month in sorted(by_month):
        record = by_month[month]
        arrays = {
            name: np.stack(record[name], axis=0)
            if name != "time_utc_ns"
            else np.asarray(record[name], dtype=np.int64)
            for name in RECONSTRUCTION_ARRAY_NAMES
        }
        order = np.argsort(arrays["time_utc_ns"], kind="stable")
        arrays = {name: value[order] for name, value in arrays.items()}
        metadata = _write_atomic_month(
            root,
            flow_id,
            month,
            arrays,
            gap_ids=sorted(set(record["_gap_ids"])),
            config=settings,
            source_fingerprint=source_fingerprint,
        )
        month_metadata.append(metadata)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "month_published",
                    "flow_id": flow_id,
                    "month": month,
                    "row_count": int(arrays["time_utc_ns"].size),
                }
            )
    return _write_domain_manifest(
        root,
        flow_id,
        month_metadata,
        config=settings,
        source_fingerprint=source_fingerprint,
        mesh_fingerprint=mesh_fingerprint,
    )


def validate_reconstruction_patch(domain_root: str | Path) -> dict[str, Any]:
    """驗證 domain manifest、month metadata、array checksum 與固定 patch topology。"""

    root = Path(domain_root)
    manifest_path = root / "reconstruction-manifest.json"
    sidecar = root / "reconstruction-manifest.json.sha256"
    if not manifest_path.is_file() or not sidecar.is_file():
        raise ReconstructionError("domain manifest 或 checksum sidecar 缺少")
    data = manifest_path.read_bytes()
    expected = sidecar.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(data).hexdigest()
    if expected != actual:
        raise ReconstructionError("domain manifest SHA-256 不符")
    document = json.loads(data.decode("utf-8"))
    if document.get("schema_version") != RECONSTRUCTION_SCHEMA_VERSION:
        raise ReconstructionError("不支援的 reconstruction patch schema")
    for item in document.get("months", []):
        month = str(item["month"])
        month_root = root / "months" / month
        metadata_path = month_root / "metadata.json"
        metadata_hash_path = month_root / "metadata.json.sha256"
        if not metadata_path.is_file() or not metadata_hash_path.is_file():
            raise ReconstructionError(f"月份 metadata 缺少：{month}")
        metadata_bytes = metadata_path.read_bytes()
        metadata_hash = metadata_hash_path.read_text(encoding="ascii").strip()
        if hashlib.sha256(metadata_bytes).hexdigest() != metadata_hash:
            raise ReconstructionError(f"月份 metadata SHA-256 不符：{month}")
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        expected_metadata_hash = hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest()
        if expected_metadata_hash != item.get("metadata_sha256"):
            raise ReconstructionError(f"domain manifest 月 metadata hash 不符：{month}")
        expected_names = {f"{name}.npy" for name in RECONSTRUCTION_ARRAY_NAMES}
        actual_names = {
            path.name
            for path in month_root.iterdir()
            if path.is_file() and path.suffix == ".npy"
        }
        if actual_names != expected_names:
            raise ReconstructionError(f"月份 patch 欄位拓撲不符：{month}")
        for name, descriptor in metadata.get("arrays", {}).items():
            path = month_root / f"{name}.npy"
            if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
                raise ReconstructionError(f"月份 array checksum 不符：{month}/{name}")
            loaded = np.load(path, mmap_mode="r")
            if (
                list(loaded.shape) != descriptor.get("shape")
                or np.dtype(loaded.dtype).name != descriptor.get("dtype")
            ):
                raise ReconstructionError(f"月份 array shape/dtype 不符：{month}/{name}")
        if set(metadata.get("arrays", {})) != set(RECONSTRUCTION_ARRAY_NAMES):
            raise ReconstructionError(f"月份 metadata 欄位集合不符：{month}")
        times = np.load(month_root / "time_utc_ns.npy", mmap_mode="r")
        if times.ndim != 1 or np.any(np.diff(times) <= 0):
            raise ReconstructionError(f"月份 patch UTC 軸未嚴格遞增：{month}")
    return document


def _baseline_predictions(
    full_source: ArraySequenceSource,
    missing_times: np.ndarray,
    field_name: str,
    *,
    kind: str,
) -> np.ndarray:
    """blocked-mask 驗證用的 persistence 或兩端線性 baseline。"""

    first = int(missing_times[0])
    left_time = first - int(HOURLY_NS)
    right_time = int(missing_times[-1]) + int(HOURLY_NS)
    left = full_source.read_times(np.asarray([left_time]), field_name)
    right = full_source.read_times(np.asarray([right_time]), field_name)
    if kind == "persistence":
        return np.repeat(left, missing_times.size, axis=0)
    if kind == "endpoint_linear":
        alpha = np.arange(1, missing_times.size + 1, dtype=np.float64)[:, None] / (missing_times.size + 1)
        return left + alpha * (right - left)
    raise ValueError(f"未知 baseline：{kind}")


def validate_blocked_masks(
    full_source: ArraySequenceSource,
    *,
    block_lengths_hours: Sequence[int] = (1, 23, 24, 25, 49),
    config: ReconstructionConfig | None = None,
    lagrangian_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """在 synthetic 完整序列上遮蔽指定長度，輸出 Eulerian 指標與 baseline 比較。

    這是 blocked cross-validation API，不會把 callback 回傳值偽裝成正式 Lagrangian
    驗證。若 caller 提供 ``lagrangian_callback``，其輸入／輸出會原樣放在
    ``lagrangian_validation`` 並標記 ``caller_supplied``；未提供則明確標記
    ``not_run``。
    """

    settings = config or ReconstructionConfig()
    if not isinstance(full_source, ArraySequenceSource):
        raise TypeError("blocked-mask validation 目前需要 ArraySequenceSource 完整 synthetic source")
    lengths = tuple(int(value) for value in block_lengths_hours)
    if not lengths or any(value < 1 for value in lengths):
        raise ValueError("block_lengths_hours 必須為正整數")
    if full_source.times_utc_ns.size < max(lengths) + 2 * settings.min_state_history + 2:
        raise ValueError("synthetic source 太短，無法提供 gap 兩側訓練資料")
    center = full_source.times_utc_ns.size // 2
    results: list[dict[str, Any]] = []
    field_name = "hvel"
    for length in lengths:
        start_index = center - length // 2
        missing = full_source.times_utc_ns[start_index : start_index + length]
        masked = full_source.masked(missing)
        gap = GapSpec(
            gap_id=f"blocked_{length}h",
            missing_times_ns=missing,
            left_time_ns=int(missing[0]) - int(HOURLY_NS),
            right_time_ns=int(missing[-1]) + int(HOURLY_NS),
        )
        if length == 1:
            predicted = _reconstruct_gap(masked, gap, settings)[field_name].reshape(length, -1)
        else:
            predicted = _reconstruct_gap(masked, gap, settings)[field_name].reshape(length, -1)
        truth = full_source.read_times(missing, field_name)
        metrics: dict[str, Any] = {}
        for name, baseline in (
            ("persistence", _baseline_predictions(full_source, missing, field_name, kind="persistence")),
            (
                "endpoint_linear",
                _baseline_predictions(full_source, missing, field_name, kind="endpoint_linear"),
            ),
        ):
            error = baseline.astype(np.float64) - truth.astype(np.float64)
            metrics[name] = {
                "rmse": float(np.sqrt(np.nanmean(np.square(error)))),
                "bias": float(np.nanmean(error)),
                "correlation": _corrcoef(truth.ravel(), baseline.ravel()),
            }
        error = predicted.astype(np.float64) - truth.astype(np.float64)
        metrics["reconstruction"] = {
            "rmse": float(np.sqrt(np.nanmean(np.square(error)))),
            "bias": float(np.nanmean(error)),
            "correlation": _corrcoef(truth.ravel(), predicted.ravel()),
        }
        if length == 1:
            # 一小時契約是「與兩端 component linear 完全相同」，不要求真實 synthetic
            # signal 本身恰好線性；因此 acceptance 比對 endpoint baseline，而不是把
            # curvature 誤判成重建失敗。
            endpoint = _baseline_predictions(full_source, missing, field_name, kind="endpoint_linear")
            accepted = bool(np.allclose(predicted, endpoint, rtol=0.0, atol=1.0e-12))
        else:
            accepted = bool(
                metrics["reconstruction"]["rmse"] < metrics["persistence"]["rmse"]
                and metrics["reconstruction"]["rmse"] < metrics["endpoint_linear"]["rmse"]
            )
        results.append({"block_length_hours": length, "metrics": metrics, "accepted": accepted})
    payload: dict[str, Any] = {
        "schema_version": "ocm_reconstruction_blocked_validation_v1",
        "method": settings.method_id,
        "block_lengths_hours": list(lengths),
        "fields": [field_name],
        "results": results,
        "acceptance": {
            "all_synthetic_blocks_pass": bool(all(item["accepted"] for item in results)),
            "formal_scientific_claim": False,
            "basis": "synthetic_blocked_mask_only",
        },
        "lagrangian_validation": {"status": "not_run"},
    }
    if lagrangian_callback is not None:
        callback_input = {
            "schema_version": "ocm_reconstruction_lagrangian_callback_input_v1",
            "reconstruction_validation": payload,
        }
        payload["lagrangian_validation"] = {
            "status": "caller_supplied",
            "payload": dict(lagrangian_callback(callback_input)),
            "formal_scientific_claim": False,
        }
    return payload


__all__ = [
    "ArraySequenceSource",
    "CONTINUOUS_FIELD_NAMES",
    "GapSpec",
    "HOURLY_NS",
    "NpyDomainSource",
    "ORIGIN_RECONSTRUCTED_SHORT",
    "ORIGIN_RECONSTRUCTED_STATE_SPACE",
    "QUALITY_DIFFUSIVITY_RANGE",
    "QUALITY_NO_TWO_SIDED_SUPPORT",
    "QUALITY_NONFINITE_CELL",
    "QUALITY_SHORT_LINEAR",
    "QUALITY_STATE_SPACE",
    "QUALITY_WETDRY_UNSUPPORTED",
    "QUALITY_ZCOR_ORDER",
    "RECONSTRUCTION_ARRAY_NAMES",
    "RECONSTRUCTION_METHOD_ID",
    "RECONSTRUCTION_SCHEMA_VERSION",
    "ReconstructionConfig",
    "ReconstructionError",
    "SequenceSource",
    "build_reconstruction_patch",
    "discover_hourly_gaps",
    "sha256_file",
    "validate_blocked_masks",
    "validate_reconstruction_patch",
]
