"""Aggregate release 固定檔案拓撲與安全數值容器。

本模組只描述 aggregate release 產品的固定檔名集合，以及封存已編碼表格與數值
陣列的不可變容器；不負責 encode、decode、檔案讀寫、資料列欄位解析或科學計算。
表格列代表 release 中的 shard 綁定、scenario 分層、站點與邊界拓撲、來源—受體
關聯、事件／結果計數或分母資料。陣列的實際 shape 關聯、offset 邊界一致性與
跨檔案語意由 encode/decode 層負責，本容器只保證檔名拓撲與數值資料的安全封存。

其中，``age_bin_edges_seconds.npy``、``boundary_travel_age_histogram.npy`` 與
``source_receptor_travel_age_histogram.npy`` 的時間單位是秒；``boundary_bin_edges_m.npy``
的邊界距離單位是公尺；``pathway_residence_time_seconds.npy`` 的累積停留時間單位是秒。
``site_cell_offsets.npy``、``boundary_edge_offsets.npy`` 與 ``boundary_bin_offsets.npy``
是索引 offset；其餘 ``*_count.npy``、``*_raw_count.npy``、
``pathway_unique_particle_count.npy`` 是非負整數計數。陣列軸與每個 offset／histogram
之間的尺寸對應不在本模組驗證。

這個容器保存的是可供後續 release 編解碼與驗證使用的資料產品內容，不因成功建立
容器就代表資料已通過科學正確性、物理合理性或觀測驗證；結果仍只能依專案資料契約
稱為「條件式來源足跡」或「相對來源權重」，不得直接解讀為絕對來源機率或因果歸因。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

__all__ = [
    "AGGREGATE_RELEASE_ARRAY_FILES",
    "AGGREGATE_RELEASE_TABLE_FILES",
    "EncodedAggregateProducts",
]


# 這九個 Parquet 檔案共同描述 release 的輸入綁定、分層、空間拓撲、關聯統計與分母；
# 固定集合讓 encoder、decoder 與驗證器能拒絕遺漏或額外產物，而不靠檔名猜測用途。
AGGREGATE_RELEASE_TABLE_FILES: Final[frozenset[str]] = frozenset(
    {
        "shard_bindings.parquet",
        "scenario_strata.parquet",
        "site_index.parquet",
        "boundary_index.parquet",
        "source_receptor_index.parquet",
        "cross_site_counts.parquet",
        "outcome_counts.parquet",
        "site_denominators.parquet",
        "receptor_denominators.parquet",
    }
)

# 這十八個 NumPy 檔案固定 release 的時間軸、空間索引、事件計數、路徑統計與邊界／
# 來源—受體直方圖；集合本身不承擔 shape、軸長或 offset 數值的一致性驗證責任。
AGGREGATE_RELEASE_ARRAY_FILES: Final[frozenset[str]] = frozenset(
    {
        "age_bin_edges_seconds.npy",
        "site_cell_offsets.npy",
        "local_first_exit_count.npy",
        "outer_first_exit_count.npy",
        "bed_first_contact_count.npy",
        "bed_repeated_contact_count.npy",
        "data_gap_failure_count.npy",
        "numerical_failure_count.npy",
        "pathway_unique_particle_count.npy",
        "pathway_residence_time_seconds.npy",
        "pathway_first_passage_age_histogram.npy",
        "boundary_edge_offsets.npy",
        "boundary_bin_edges_m.npy",
        "boundary_bin_offsets.npy",
        "boundary_arclength_raw_count.npy",
        "boundary_travel_age_histogram.npy",
        "source_receptor_raw_count.npy",
        "source_receptor_travel_age_histogram.npy",
    }
)


def _copy_exact_mapping(
    value: object,
    *,
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, object]:
    """複製外層 mapping，並以 exact key set 驗證固定 release 拓撲。

    ``tables`` 與 ``arrays`` 都是呼叫端可能仍會變動的 mapping；先 materialize 成獨立
    ``dict``，再比較未知與缺少的 key，可避免驗證期間或建構完成後受外部 mapping
    修改影響。這裡只處理外層檔名，不解析資料列欄位或陣列內容。
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必須是 mapping")
    try:
        copied = dict(value)
    except Exception as error:
        raise ValueError(f"{label} 無法防禦性複製") from error

    actual_keys = frozenset(copied)
    if actual_keys != expected_keys:
        unknown_keys = tuple(key for key in copied if key not in expected_keys)
        missing_keys = tuple(sorted(expected_keys - actual_keys))
        raise ValueError(
            f"{label} 必須恰好包含固定 key set；"
            f"未知 key={unknown_keys!r}；缺少 key={missing_keys!r}"
        )
    return copied


def _snapshot_table(
    value: object,
    *,
    label: str,
    allow_empty: bool,
) -> tuple[Mapping[str, object], ...]:
    """將表格 sequence 逐列複製成唯讀 mapping，並保留原始列順序。

    表格只接受真正的 ``Sequence``，因此不會把 generator、set 或字串誤當成可重現的
    列序列。每一列以 ``dict`` 建立獨立 shallow copy，再由 ``MappingProxyType`` 防止
    透過容器本身新增、刪除或覆寫欄位；外層則固定為 tuple。此層不判斷欄位名稱、欄位
    型別、零事件計數或跨表關聯，因為那些是編解碼層的資料契約。

    ``cross_site_counts.parquet`` 是唯一允許空 tuple 的表格，供單站 synthetic smoke
    保留沒有跨站事件的拓撲；呼叫端必須在此函式之外明確傳入 ``allow_empty=True``，
    以免其他表格意外繞過非空限制。
    """

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} 必須是 sequence of mapping")
    try:
        rows = tuple(value)
    except Exception as error:
        raise ValueError(f"{label} 無法防禦性複製成列序列") from error

    if not rows and not allow_empty:
        raise ValueError(
            f"{label} 不得為空；唯一例外是 cross_site_counts.parquet 可為空 tuple"
        )

    snapshots: list[Mapping[str, object]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}[{row_index}] 必須是 mapping")
        try:
            row_copy = dict(row)
        except Exception as error:
            raise ValueError(f"{label}[{row_index}] 無法防禦性複製") from error
        snapshots.append(MappingProxyType(row_copy))
    return tuple(snapshots)


def _canonicalize_array(value: object, *, label: str) -> np.ndarray:
    """驗證並複製單一數值陣列，統一為唯讀 ``int64`` 或 ``float64``。

    整數陣列的所有元素都代表計數、offset 或其他不可為負的索引型數值，因此負值
    直接拒絕；unsigned 整數若超出 ``int64`` 上限也拒絕，避免轉型時繞回負數。浮點
    陣列先檢查輸入與 canonical 後值都有限，防止 NaN、Infinity 或無法安全縮入
    ``float64`` 的有限大值進入容器。object、structured、字串、複數與 bool dtype
    不接受，即使其中的元素看似可以轉成數字也不做隱式修補。
    """

    if not isinstance(value, np.ndarray):
        raise ValueError(f"{label} 必須是 np.ndarray")

    dtype_kind = value.dtype.kind
    if dtype_kind == "i":
        if bool(np.any(value < 0)):
            raise ValueError(f"{label} 的 integer 元素不可為負")
        canonical = np.array(value, dtype=np.int64, copy=True)
    elif dtype_kind == "u":
        int64_max = np.iinfo(np.int64).max
        if bool(np.any(value > int64_max)):
            raise ValueError(f"{label} 的 unsigned integer 不可超過 int64 上限")
        canonical = np.array(value, dtype=np.int64, copy=True)
    elif dtype_kind == "f":
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{label} 的 float 元素必須全部有限，不可含 NaN 或 Infinity")
        # 某些較高精度浮點 dtype 可保存超出 float64 範圍的有限值；轉型後再次檢查，
        # 確保 canonical 容器不會把這種值靜默變成 Infinity。
        with np.errstate(over="ignore", invalid="ignore"):
            canonical = np.array(value, dtype=np.float64, copy=True)
        if not bool(np.all(np.isfinite(canonical))):
            raise ValueError(f"{label} 無法安全 canonical 成有限的 float64")
    else:
        raise ValueError(
            f"{label} dtype={value.dtype!r} 不允許；只允許非負 integer 或有限 float"
        )

    # canonical 已經是獨立 buffer；此旗標封住 ndarray 本身的元素寫入入口，但不承擔
    # shape、offset 或 histogram 對應關係的驗證，避免把容器層與編解碼層職責混在一起。
    canonical.setflags(write=False)
    return canonical


@dataclass(frozen=True, slots=True)
class EncodedAggregateProducts:
    """封存一份 aggregate release 的固定拓撲產品與安全數值陣列。

    ``tables`` 的 key 必須恰好是 ``AGGREGATE_RELEASE_TABLE_FILES``；每個 value 是
    一串表格列，列內容以 mapping 表示。列的欄位 schema、排序、跨表 join key 與零事件
    時的拓撲語意由編解碼層及上游資料契約負責，本類別只會把列逐列複製並鎖定為
    ``MappingProxyType``，再以 tuple 與外層唯讀 mapping 隔離呼叫端的 dict/list 修改。

    ``arrays`` 的 key 必須恰好是 ``AGGREGATE_RELEASE_ARRAY_FILES``；每個 value 必須
    是 NumPy integer 或 floating array，建立後分別 canonical 成 ``int64`` 或 ``float64``
    的獨立唯讀複本。整數只保存非負值，浮點數只保存有限值；這些限制是為了避免
    release 容器中的計數、offset、距離、時間與直方圖因型別溢位、非有限值或可變
    buffer 而產生不透明的資料狀態。

    陣列的時間軸以秒表示，距離與邊界弧長以公尺表示，offset 是索引位置；事件、失敗、
    路徑與來源—受體資料的空間軸和 histogram 軸仍須由 encode/decode 驗證。建立本
    容器不代表資料就是實際科學結果，也不建立先驗、似然或觀測驗證，因此不得把它
    直接稱為絕對來源機率或因果歸因。
    """

    tables: Mapping[str, tuple[Mapping[str, object], ...]]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        """驗證固定 key、封存表列與 canonical 化數值陣列。

        先複製並驗證 ``tables``／``arrays`` 的 exact key set，再逐項建立不可變快照；
        任一項失敗都立即以 ``ValueError`` 拒絕，不以零值補資料、不推導遺漏檔案，
        也不檢查 offsets、edges、histogram 的 shape 關聯。這個順序確保 dataclass
        完成後的兩個公開 mapping 不再依賴呼叫端的 dict、list 或 NumPy buffer。
        """

        raw_tables = _copy_exact_mapping(
            self.tables,
            expected_keys=AGGREGATE_RELEASE_TABLE_FILES,
            label="tables",
        )
        raw_arrays = _copy_exact_mapping(
            self.arrays,
            expected_keys=AGGREGATE_RELEASE_ARRAY_FILES,
            label="arrays",
        )

        table_snapshots: dict[str, tuple[Mapping[str, object], ...]] = {}
        for file_name in AGGREGATE_RELEASE_TABLE_FILES:
            # 單站 synthetic smoke 可能沒有跨站事件，但仍保留該檔案 key；這是唯一可
            # 省略資料列的表格。其他 outcome、source、denominator 與拓撲表即使計數為
            # 零，也必須有列來表達固定拓撲，不能用空 tuple 取代。
            allow_empty = file_name == "cross_site_counts.parquet"
            table_snapshots[file_name] = _snapshot_table(
                raw_tables[file_name],
                label=f"tables[{file_name!r}]",
                allow_empty=allow_empty,
            )

        array_snapshots: dict[str, np.ndarray] = {}
        for file_name in AGGREGATE_RELEASE_ARRAY_FILES:
            array_snapshots[file_name] = _canonicalize_array(
                raw_arrays[file_name],
                label=f"arrays[{file_name!r}]",
            )

        # frozen dataclass 只保護欄位重新賦值；MappingProxyType、tuple 與唯讀 NumPy
        # buffer 才共同完成本容器需要的外部 mutation isolation。
        object.__setattr__(self, "tables", MappingProxyType(table_snapshots))
        object.__setattr__(self, "arrays", MappingProxyType(array_snapshots))
