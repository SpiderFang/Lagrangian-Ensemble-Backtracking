"""把既有 OCM 時間列轉成原版 BayTrace CPU 一步所需的 frame。

本模組是混合生命週期的轉接層：步首位置、三角形提示、boundary 與 diffusion 仍由
LBT ``ProductionBatch``／engine 管理，這裡只在明確支援的「ManagedForcingProvider、
OCM-only、有限垂向資料」條件下準備原版 BayTrace 的靜態網格與兩端速度場。真正的
內插與四階 Runge--Kutta（RK4）仍由 ``baytrace_native_cpu.step_native`` 呼叫 PI
提供的原始 ``batch_interpolate_velocity``／``batch_move_rk4``；本模組不複製原版公式。

時間端點刻意使用 ``ForcingWindowManager`` 的既有私有流程
``_pin_months``、``_prepare_time_entries`` 與 ``_find_time_bracket(source="ocm")``。
因此跨月 duplicate、continuous-time fallback、merged observed/reconstruction patch
與 canonical endpoint 選擇都保持原版 manager 語意，而不是在 adapter 內自行發明最近
時刻規則。``step_start_sample`` 仍由 LBT 生命週期在步前取得；所以這不是聲稱完整
tracker 已改成 BayTrace，而是保留既有步首參考／步後 boundary 的混合實作。
"""

from __future__ import annotations

import weakref
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .baytrace_batch_advection import PreparedBayTraceFrame
from .baytrace_native_cpu import BayTraceFieldPair, BayTraceGridArrays
from .forcing_window import (
    ManagedForcingProvider,
    _GlobalTimeBracket,
    _month_from_utc_ns,
)
from .models import SampleQC

_NATIVE_INT_BYTES = np.dtype(np.int64).itemsize
_SIGMA_TOLERANCE = 1.0e-6


class _FramePreparationFallback(Exception):
    """表示一個預期內的 native gate 失敗；不可讓單一粒子終止整個 batch。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(slots=True)
class _EndpointSnapshot:
    """以 weak reference 保存最近 bracket，避免把月份 LRU 變成無上限常駐。"""

    entry_ref: weakref.ReferenceType[Any] | None
    # ``_MonthEntry`` 在 forcing_window 使用 slots，部分 Python 版本不能建立 weakref；
    # 此時只保留最近 bracket 最多兩個 entry 的 strong reference，仍是固定容量，不會
    # 隨月份數增長。能 weakref 時則不持有來源陣列，讓 manager LRU 正常回收。
    entry_strong: Any | None
    index: int
    utc_ns: int

    @classmethod
    def from_endpoint(cls, endpoint: tuple[Any, int, int]) -> _EndpointSnapshot:
        """把 manager endpoint 轉成不持有大型 OCM 陣列的快取描述。"""

        entry, index, utc_ns = endpoint
        try:
            entry_ref: weakref.ReferenceType[Any] | None = weakref.ref(entry)
            entry_strong = None
        except TypeError:
            entry_ref = None
            entry_strong = entry
        return cls(entry_ref, entry_strong, int(index), int(utc_ns))

    def resolve(self) -> tuple[Any, int, int] | None:
        """若月份 entry 尚未被 manager 淘汰，還原原始 bracket endpoint。"""

        entry = self.entry_ref() if self.entry_ref is not None else self.entry_strong
        if entry is None:
            return None
        return entry, self.index, self.utc_ns


@dataclass(slots=True)
class _BracketSnapshot:
    """保存單一 manager／UTC 的全域 bracket，最多保留最近一筆。"""

    manager_ref: weakref.ReferenceType[Any]
    target_ns: int
    before: _EndpointSnapshot | None
    after: _EndpointSnapshot | None
    alpha: float
    qc: int
    fallback_used: bool
    fallback_policy: str | None

    @classmethod
    def from_bracket(
        cls,
        manager: Any,
        target_ns: int,
        bracket: _GlobalTimeBracket,
    ) -> _BracketSnapshot:
        """將 manager bracket 保存為 bounded、可回收的最近查詢。"""

        return cls(
            manager_ref=weakref.ref(manager),
            target_ns=int(target_ns),
            before=(
                _EndpointSnapshot.from_endpoint(bracket.before)
                if bracket.before is not None
                else None
            ),
            after=(
                _EndpointSnapshot.from_endpoint(bracket.after)
                if bracket.after is not None
                else None
            ),
            alpha=float(bracket.alpha),
            qc=int(bracket.qc),
            fallback_used=bool(bracket.fallback_used),
            fallback_policy=bracket.fallback_policy,
        )

    def resolve(self, manager: Any, target_ns: int) -> _GlobalTimeBracket | None:
        """只在 manager identity 與 UTC 完全相同且 endpoint 仍存活時命中。"""

        if self.manager_ref() is not manager or self.target_ns != int(target_ns):
            return None
        before = self.before.resolve() if self.before is not None else None
        after = self.after.resolve() if self.after is not None else None
        if self.before is not None and before is None:
            return None
        if self.after is not None and after is None:
            return None
        return _GlobalTimeBracket(
            before,
            after,
            self.alpha,
            SampleQC(self.qc),
            fallback_used=self.fallback_used,
            fallback_policy=self.fallback_policy,
        )


@dataclass(slots=True)
class _FrameCacheRecord:
    """保存可重用 frame 及端點弱引用；不持有已淘汰月份的來源陣列。"""

    frame: PreparedBayTraceFrame
    nbytes: int
    before_ref: weakref.ReferenceType[Any]
    after_ref: weakref.ReferenceType[Any]
    metadata: dict[str, Any]

    def matches(self, before_ocm: Any, after_ocm: Any) -> bool:
        """確認 cache key 對應的來源物件仍是目前 bracket 的兩端。"""

        return self.before_ref() is before_ocm and self.after_ref() is after_ocm


class OCMPreparedFrameResolver:
    """準備可交給原版 BayTrace kernel 的 OCM native frame。

    解析器只接受沿 ``task.velocity.provider`` 逐層 unwrap 後的既有
    ``ManagedForcingProvider``。``include_stokes=True`` 不會被悄悄當成 OCM-only：
    它會以 ``stokes_adapter_pending`` 明確 fallback，保留原 LBT 計算。每個有限 frame
    array cache 以 manager、兩端 OCM object identity、merged row／UTC 及固定沉降值作為
    key；目前查詢的 ``trat`` 只放回傳 frame 的 scalar，不放進 array key，因此相同兩端
    但不同 UTC 比例不會重建整份 grid／field。它不對大型網格或月檔做 hash。超過單 frame
    記憶體上限、無效垂向列、乾面或相鄰面支援不一致時，會以有限容量 negative cache
    記住理由，避免相同端點每粒子重掃全網格；局部乾面 negative key 另含 element ID。
    fallback 只累計在 report，不逐粒子輸出 warning／print。

    ``max_cached_bytes`` 是 resolver 所持有的 grid+field frame 陣列總上限；建立任何
    大型輸出複本前，先依固定 float64/int64 layout 估算 bytes。通常容量滿時淘汰最舊
    frame，只有單一 frame 本身超過上限才 fallback；negative cache 容量固定為
    ``max_cached_frames``，不會隨粒子數或月份數增長。
    """

    def __init__(
        self,
        *,
        max_cached_frames: int = 2,
        max_cached_bytes: int = 512 * 1024**2,
    ) -> None:
        """建立 bounded frame／negative cache；兩個容量都必須是正整數。"""

        if isinstance(max_cached_frames, bool) or not isinstance(max_cached_frames, int):
            raise TypeError("max_cached_frames 必須是正整數")
        if max_cached_frames < 1:
            raise ValueError("max_cached_frames 必須是正整數")
        if isinstance(max_cached_bytes, bool) or not isinstance(max_cached_bytes, int):
            raise TypeError("max_cached_bytes 必須是正整數")
        if max_cached_bytes < 1:
            raise ValueError("max_cached_bytes 必須是正整數")

        self.max_cached_frames = max_cached_frames
        self.max_cached_bytes = max_cached_bytes
        self._frame_cache: OrderedDict[tuple[Any, ...], _FrameCacheRecord] = OrderedDict()
        self._negative_cache: OrderedDict[tuple[Any, ...], str] = OrderedDict()
        self._cached_bytes = 0
        self._last_bracket: _BracketSnapshot | None = None
        self._counters: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._last_frame_metadata: dict[str, Any] | None = None

    def __call__(self, task: Any) -> PreparedBayTraceFrame | None:
        """為一個 Q1 ``AdvectionTask`` 解析 frame；不支援時回傳 ``None``。

        這個 callable 的 ``None`` 是明示的 native unsupported 訊號，會由
        ``NativeBayTraceBatchAdvector`` 交回既有 ``execute_advection_task``，並在其
        ``fallback_particle_steps`` 計數。此處只負責 resolver 自身可查的理由與 cache
        統計，不把 legacy fallback 誤標為原版 BayTrace。
        """

        key: tuple[Any, ...] | None = None
        element_id: int | None = None
        try:
            managed = self._unwrap_managed_provider(task)
            if managed.include_stokes:
                raise _FramePreparationFallback("stokes_adapter_pending")

            manager = managed.manager
            target_ns = self._task_time(task)
            element_id = self._task_element_id(task, manager)
            bracket = self._get_bracket(manager, target_ns)
            if bracket.qc != SampleQC.OK:
                raise _FramePreparationFallback(
                    "ocm_time_bracket_invalid",
                    f"qc={int(bracket.qc)}",
                )
            if bracket.before is None or bracket.after is None:
                raise _FramePreparationFallback("ocm_time_bracket_missing")
            before_entry, before_index, before_ns = bracket.before
            after_entry, after_index, after_ns = bracket.after
            before_ocm = getattr(before_entry, "ocm", None)
            after_ocm = getattr(after_entry, "ocm", None)
            if before_ocm is None or after_ocm is None:
                raise _FramePreparationFallback("ocm_month_missing")

            trat = float(bracket.alpha)
            settling = float(managed.settling_velocity_mps)
            key = self._frame_key(
                manager,
                before_ocm,
                before_index,
                before_ns,
                after_ocm,
                after_index,
                after_ns,
                settling,
            )
            negative_reason = self._negative_lookup(key)
            if negative_reason is not None:
                raise _FramePreparationFallback(negative_reason)

            record = self._frame_cache.get(key)
            if record is not None:
                if record.matches(before_ocm, after_ocm):
                    # 乾面／一環鄰居 gate 依粒子的 element 而定；即使 array frame 命中，
                    # 仍只做這個小型 local check，不重掃整個 mesh。失敗時由 outer catch
                    # 轉成含 element 的 bounded local negative key。
                    self._validate_local_wet_support(
                        manager.mesh,
                        before_ocm,
                        before_index,
                        after_ocm,
                        after_index,
                        element_id,
                    )
                    self._frame_cache.move_to_end(key)
                    self._counters["cache_hits"] += 1
                    query_metadata = self._frame_metadata(manager, bracket, target_ns)
                    query_metadata["frame_cache_reused"] = True
                    query_metadata["frame_build_target_utc_ns"] = record.metadata[
                        "frame_build_target_utc_ns"
                    ]
                    query_metadata["frame_build_trat"] = record.metadata["frame_build_trat"]
                    self._last_frame_metadata = query_metadata
                    return replace(record.frame, trat=trat, element_id=element_id)
                # id() 不應在 manager 仍持有 endpoint 時重用；若 entry 已重建，
                # 先移除 stale record，避免同一 integer identity 意外命中舊資料。
                self._remove_frame(key)
            self._counters["cache_misses"] += 1

            # array cache miss 才檢查跨 mesh 相容性；同一 manager 的正常 endpoint
            # 通常直接共用靜態 mesh object，避免每個粒子呼叫 _meshes_match 掃十個陣列。
            self._validate_endpoint_meshes(manager, before_ocm, after_ocm)
            self._validate_local_wet_support(
                manager.mesh,
                before_ocm,
                before_index,
                after_ocm,
                after_index,
                element_id,
            )

            rows = self._read_endpoint_rows(
                before_ocm,
                before_index,
                after_ocm,
                after_index,
            )
            estimated_bytes = self._estimate_frame_bytes(manager.mesh, rows["z_before"])
            if estimated_bytes > self.max_cached_bytes:
                raise _FramePreparationFallback(
                    "frame_bytes_exceeds_limit",
                    f"estimated={estimated_bytes}, limit={self.max_cached_bytes}",
                )

            grid, fields = self._build_grid_and_fields(
                manager.mesh,
                rows,
                settling_velocity_mps=settling,
            )
            frame = PreparedBayTraceFrame(
                grid=grid,
                fields=fields,
                trat=trat,
                element_id=element_id,
            )
            actual_bytes = self._frame_nbytes(frame)
            if actual_bytes > self.max_cached_bytes:
                # 理論估算與 constructor 後 bytes 不應不同；仍保留第二道 gate，
                # 以防未來陣列 dtype/layout 契約變更時把大複本留在 cache。
                raise _FramePreparationFallback(
                    "frame_bytes_exceeds_limit",
                    f"actual={actual_bytes}, limit={self.max_cached_bytes}",
                )
            metadata = self._frame_metadata(manager, bracket, target_ns)
            metadata["frame_cache_reused"] = False
            metadata["frame_build_target_utc_ns"] = int(target_ns)
            metadata["frame_build_trat"] = float(trat)
            self._store_frame(
                key,
                _FrameCacheRecord(
                    frame=frame,
                    nbytes=actual_bytes,
                    before_ref=weakref.ref(before_ocm),
                    after_ref=weakref.ref(after_ocm),
                    metadata=metadata,
                ),
            )
            self._counters["frame_builds"] += 1
            self._last_frame_metadata = dict(metadata)
            return frame
        except _FramePreparationFallback as error:
            fallback_key = key
            if error.reason in {"dry_face", "adjacent_support_mismatch"} and key is not None:
                # 這兩個 gate 由目前 element 的 local wet/dry 支援決定，不能把一顆
                # dry particle 的理由寫成整個 endpoint frame 的 global negative。
                fallback_key = (*key, "element", element_id)
            return self._record_fallback(error.reason, key=fallback_key, detail=error.detail)
        except (TypeError, ValueError, IndexError, OverflowError) as error:
            # 輸入 task 或 manager 端點不符合 native 契約時，保持批次 fail-closed；
            # unexpected I/O／loader exception 不在此 catch，避免掩蓋產品損壞證據。
            reason = "different_mesh" if "mesh" in str(error).lower() else "resolver_input_invalid"
            return self._record_fallback(
                reason,
                key=key,
                detail=str(error),
            )

    def report(self) -> dict[str, Any]:
        """回傳 JSON-safe 的 cache、時間來源與 fallback reason evidence。"""

        cached_metadata = [dict(record.metadata) for record in self._frame_cache.values()]
        return {
            "backend": "ocm_prepared_frame_resolver",
            "mixed_lifecycle": True,
            "step_start_reference": "LBT_velocity_sample",
            "max_cached_frames": int(self.max_cached_frames),
            "max_cached_bytes": int(self.max_cached_bytes),
            "cached_frames": int(len(self._frame_cache)),
            "cached_bytes": int(self._cached_bytes),
            "negative_cache_entries": int(len(self._negative_cache)),
            "negative_cache_capacity": int(self.max_cached_frames),
            "fallback_count": int(sum(self._fallback_reasons.values())),
            "fallback_reasons": {
                str(reason): int(count)
                for reason, count in sorted(self._fallback_reasons.items())
            },
            "cache_hits": int(self._counters["cache_hits"]),
            "cache_misses": int(self._counters["cache_misses"]),
            "frame_builds": int(self._counters["frame_builds"]),
            "frame_evictions": int(self._counters["frame_evictions"]),
            "negative_cache_hits": int(self._counters["negative_cache_hits"]),
            "negative_cache_stores": int(self._counters["negative_cache_stores"]),
            "negative_cache_evictions": int(self._counters["negative_cache_evictions"]),
            "bracket_cache_hits": int(self._counters["bracket_cache_hits"]),
            "bracket_queries": int(self._counters["bracket_queries"]),
            "time_fallback_count": int(self._counters["time_fallback_count"]),
            "cached_frame_metadata": cached_metadata,
            "last_frame_metadata": (
                dict(self._last_frame_metadata) if self._last_frame_metadata is not None else None
            ),
        }

    def _unwrap_managed_provider(self, task: Any) -> ManagedForcingProvider:
        """只接受 direct Managed 或既有 HintTrackingVelocityProvider 外層。

        任意具有 ``provider`` 屬性的自訂 wrapper 都可能改寫速度、時間或 QC；若讓它
        透過遞迴 unwrap 進入 native，便會漏掉物理差異。因此這裡只承認 production 已
        登錄的 ``HintTrackingVelocityProvider``，且其底層必須直接是
        ``ManagedForcingProvider``。
        """

        current = getattr(task, "velocity", None)
        if isinstance(current, ManagedForcingProvider):
            return current
        from .production import HintTrackingVelocityProvider

        if isinstance(current, HintTrackingVelocityProvider) and isinstance(
            current.provider, ManagedForcingProvider
        ):
            return current.provider
        raise _FramePreparationFallback("unsupported_provider_wrapper")

    @staticmethod
    def _task_time(task: Any) -> int:
        """讀取 Q1 task 的步首 UTC 奈秒，不用浮點秒數選月份。"""

        value = getattr(getattr(task, "state", None), "time_utc_ns", None)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise _FramePreparationFallback("task_time_invalid")
        return int(value)

    @staticmethod
    def _task_element_id(task: Any, manager: Any) -> int:
        """以 LBT 步首 sample 的 triangle_id 選原版 element hint。"""

        sample = getattr(task, "step_start_sample", None)
        if sample is None or getattr(sample, "qc", SampleQC.NUMERICAL_FAILURE) != SampleQC.OK:
            raise _FramePreparationFallback("step_start_sample_invalid")
        value = getattr(sample, "triangle_id", None)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise _FramePreparationFallback("triangle_hint_invalid")
        element_id = int(value)
        triangle_nodes = np.asarray(getattr(manager.mesh, "triangle_nodes", ()))
        if element_id < 0 or element_id >= triangle_nodes.shape[0]:
            raise _FramePreparationFallback("triangle_hint_invalid")
        return element_id

    def _get_bracket(self, manager: Any, target_ns: int) -> _GlobalTimeBracket:
        """優先重用最近同一 manager/UTC bracket，否則呼叫既有全域時間流程。"""

        self._counters["bracket_queries"] += 1
        if self._last_bracket is not None:
            cached = self._last_bracket.resolve(manager, target_ns)
            if cached is not None:
                self._counters["bracket_cache_hits"] += 1
                return cached

        month_id = _month_from_utc_ns(target_ns)
        with manager._pin_months((month_id,)):
            entries = manager._prepare_time_entries(
                month_id,
                target_ns,
                source="ocm",
            )
            bracket = manager._find_time_bracket(entries, target_ns, source="ocm")
            self._last_bracket = _BracketSnapshot.from_bracket(manager, target_ns, bracket)
            if bracket.fallback_used:
                self._counters["time_fallback_count"] += 1
            return bracket

    @staticmethod
    def _frame_key(
        manager: Any,
        before_ocm: Any,
        before_index: int,
        before_ns: int,
        after_ocm: Any,
        after_index: int,
        after_ns: int,
        settling: float,
    ) -> tuple[Any, ...]:
        """建立不掃大型陣列的 endpoint array cache key；trat 不屬於 key。"""

        return (
            id(manager),
            id(before_ocm),
            int(before_index),
            int(before_ns),
            id(after_ocm),
            int(after_index),
            int(after_ns),
            float(settling),
        )

    def _negative_lookup(self, key: tuple[Any, ...]) -> str | None:
        """查 bounded negative cache；命中後只做 counter，不重讀大型 row。"""

        reason = self._negative_cache.get(key)
        if reason is None:
            return None
        self._negative_cache.move_to_end(key)
        self._counters["negative_cache_hits"] += 1
        return reason

    def _record_fallback(
        self,
        reason: str,
        *,
        key: tuple[Any, ...] | None,
        detail: str = "",
    ) -> None:
        """累計可查 fallback reason，並在有端點 key 時保存有限 negative entry。"""

        self._fallback_reasons[reason] += 1
        self._counters["fallback_calls"] += 1
        if key is not None and key not in self._negative_cache:
            self._negative_cache[key] = reason
            self._counters["negative_cache_stores"] += 1
            while len(self._negative_cache) > self.max_cached_frames:
                self._negative_cache.popitem(last=False)
                self._counters["negative_cache_evictions"] += 1
        # detail 僅供偵錯時呼叫者以 debugger 讀取，不寫 stdout，避免粒子級 I/O。
        if detail:
            self._counters[f"detail_{reason}"] += 1
        return None

    def _validate_endpoint_meshes(self, manager: Any, before_ocm: Any, after_ocm: Any) -> None:
        """確認兩端 mesh 與 manager 靜態 mesh 相同，拒絕跨 mesh 硬湊。"""

        # 正常 manager 會把同一個靜態 mesh 物件掛在每個月份；identity 相同已經是
        # 最強且不需掃描十個 mesh array 的證據。此函式只在 array cache miss 呼叫，
        # 因此非同物件的跨網格情況最多驗證一次後才進入 frame 建立或 fallback。
        if before_ocm.mesh is manager.mesh and after_ocm.mesh is manager.mesh:
            return
        matcher = getattr(manager, "_meshes_match", None)
        if not callable(matcher):
            raise _FramePreparationFallback("mesh_matcher_unavailable")
        if before_ocm.mesh is after_ocm.mesh:
            if not matcher(before_ocm.mesh, manager.mesh):
                raise _FramePreparationFallback("different_mesh")
            return
        if not matcher(before_ocm.mesh, manager.mesh) or not matcher(after_ocm.mesh, manager.mesh):
            raise _FramePreparationFallback("different_mesh")

    @staticmethod
    def _read_endpoint_rows(
        before_ocm: Any,
        before_index: int,
        after_ocm: Any,
        after_index: int,
    ) -> dict[str, np.ndarray]:
        """只讀兩端 merged row view，不複製整月 forcing。"""

        rows = {
            "hvel_before": np.asarray(before_ocm._row_array("hvel", before_index)),
            "vertical_before": np.asarray(
                before_ocm._row_array("vertical_velocity", before_index)
            ),
            "z_before": np.asarray(before_ocm._row_array("zcor", before_index)),
            "eta_before": np.asarray(before_ocm._row_array("elev", before_index)),
            "wet_before": np.asarray(before_ocm._row_array("wetdry_elem", before_index)),
            "hvel_after": np.asarray(after_ocm._row_array("hvel", after_index)),
            "vertical_after": np.asarray(
                after_ocm._row_array("vertical_velocity", after_index)
            ),
            "z_after": np.asarray(after_ocm._row_array("zcor", after_index)),
            "eta_after": np.asarray(after_ocm._row_array("elev", after_index)),
            "wet_after": np.asarray(after_ocm._row_array("wetdry_elem", after_index)),
        }
        return rows

    @staticmethod
    def _estimate_frame_bytes(mesh: Any, z_before: np.ndarray) -> int:
        """在任何大型 output allocation 前估算固定 BayTrace frame bytes。"""

        triangle_nodes = np.asarray(mesh.triangle_nodes)
        node_xy = np.asarray(mesh.node_xy)
        if z_before.ndim != 2 or node_xy.ndim != 2 or node_xy.shape[1] != 2:
            raise _FramePreparationFallback("frame_shape_invalid")
        nnode = int(node_xy.shape[0])
        nvrt = int(z_before.shape[1])
        ne = int(triangle_nodes.shape[0])
        if z_before.shape[0] != nnode or nvrt < 2 or ne < 1:
            raise _FramePreparationFallback("frame_shape_invalid")
        # BayTrace bridge constructors固定把 grid index與物理場轉成 int64/float64；
        # 這裡用最終 layout估算，不把 float32 input 的小 bytes 誤當成可用預算。
        grid_elements = ne + 4 * ne + 2 * nnode + 4 * ne + nnode + ne + nnode
        grid_floats = nvrt * nnode + nnode
        field_floats = 6 * nvrt * nnode + nnode
        return int((grid_elements + grid_floats + field_floats) * _NATIVE_INT_BYTES)

    @staticmethod
    def _frame_nbytes(frame: PreparedBayTraceFrame) -> int:
        """以 constructor 後實際持有的連續陣列計算 frame bytes。"""

        arrays = (
            frame.grid.i34,
            frame.grid.elnode,
            frame.grid.grid_x,
            frame.grid.grid_y,
            frame.grid.ic3,
            frame.grid.kbp,
            frame.grid.kbe,
            frame.grid.sigma_lcl,
            frame.grid.dp,
            frame.fields.u1,
            frame.fields.v1,
            frame.fields.w1,
            frame.fields.eta1,
            frame.fields.u2,
            frame.fields.v2,
            frame.fields.w2,
        )
        return int(sum(int(array.nbytes) for array in arrays))

    def _store_frame(self, key: tuple[Any, ...], record: _FrameCacheRecord) -> None:
        """以 frame count 與 bytes 雙容量維持 bounded LRU。"""

        while self._frame_cache and (
            len(self._frame_cache) >= self.max_cached_frames
            or self._cached_bytes + record.nbytes > self.max_cached_bytes
        ):
            _, oldest = self._frame_cache.popitem(last=False)
            self._cached_bytes -= oldest.nbytes
            self._counters["frame_evictions"] += 1
        self._frame_cache[key] = record
        self._cached_bytes += record.nbytes

    def _remove_frame(self, key: tuple[Any, ...]) -> None:
        """移除 stale frame，避免 stale bytes 影響後續容量判定。"""

        record = self._frame_cache.pop(key, None)
        if record is not None:
            self._cached_bytes -= record.nbytes

    @staticmethod
    def _validate_local_wet_support(
        mesh: Any,
        before_ocm: Any,
        before_index: int,
        after_ocm: Any,
        after_index: int,
        element_id: int,
    ) -> None:
        """只檢查 current element 與原版 ic3 一環鄰居，不把 dry face 變 wet。"""

        triangles = np.asarray(mesh.triangle_nodes)
        neighbors = np.asarray(mesh.triangle_neighbors)
        face_local = np.asarray(mesh.triangle_face_local)
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or neighbors.shape != triangles.shape
            or face_local.shape != (triangles.shape[0],)
        ):
            raise _FramePreparationFallback("mesh_topology_invalid")
        current_face = int(face_local[element_id])
        neighbor_ids = [int(value) for value in neighbors[element_id] if int(value) >= 0]
        face_ids = [current_face]
        face_ids.extend(int(face_local[value]) for value in neighbor_ids)
        before_wet = np.asarray(before_ocm._row_array("wetdry_elem", before_index))
        after_wet = np.asarray(after_ocm._row_array("wetdry_elem", after_index))
        for face_index in face_ids:
            if (
                face_index < 0
                or face_index >= before_wet.size
                or face_index >= after_wet.size
                or not np.isfinite(before_wet[face_index])
                or not np.isfinite(after_wet[face_index])
            ):
                raise _FramePreparationFallback("adjacent_support_mismatch")
            before_is_wet = bool(np.isclose(before_wet[face_index], before_ocm.wet_value))
            after_is_wet = bool(np.isclose(after_wet[face_index], after_ocm.wet_value))
            if not before_is_wet or not after_is_wet:
                if face_index == current_face:
                    raise _FramePreparationFallback("dry_face")
                raise _FramePreparationFallback("adjacent_support_mismatch")

    def _build_grid_and_fields(
        self,
        mesh: Any,
        rows: dict[str, np.ndarray],
        *,
        settling_velocity_mps: float,
    ) -> tuple[BayTraceGridArrays, BayTraceFieldPair]:
        """由兩端 row 建立 OCM native grid/field；所有物理軸仍是公尺與 m/s。"""

        node_xy = np.asarray(mesh.node_xy)
        dp = np.asarray(mesh.source_depth_m)
        triangle_nodes = np.asarray(mesh.triangle_nodes)
        neighbors = np.asarray(mesh.triangle_neighbors)
        z_before = np.asarray(rows["z_before"])
        z_after = np.asarray(rows["z_after"])
        eta_before = np.asarray(rows["eta_before"])
        eta_after = np.asarray(rows["eta_after"])
        hvel_before = np.asarray(rows["hvel_before"])
        hvel_after = np.asarray(rows["hvel_after"])
        vertical_before = np.asarray(rows["vertical_before"])
        vertical_after = np.asarray(rows["vertical_after"])

        nnode = int(node_xy.shape[0])
        if (
            node_xy.shape != (nnode, 2)
            or dp.shape != (nnode,)
            or triangle_nodes.ndim != 2
            or triangle_nodes.shape[1] != 3
            or neighbors.shape != triangle_nodes.shape
            or z_before.shape != z_after.shape
            or z_before.shape[0] != nnode
            or eta_before.shape != (nnode,)
            or eta_after.shape != (nnode,)
            or hvel_before.ndim != 3
            or hvel_after.shape != hvel_before.shape
            or hvel_before.shape[:2] != z_before.shape
            or hvel_before.shape[2] < 2
            or vertical_before.shape != z_before.shape
            or vertical_after.shape != z_after.shape
        ):
            raise _FramePreparationFallback("frame_shape_invalid")
        nvrt = int(z_before.shape[1])
        if nvrt < 2:
            raise _FramePreparationFallback("vertical_support_invalid")
        if np.any(~np.isfinite(dp)) or np.any(dp <= 0.0):
            raise _FramePreparationFallback("invalid_depth")
        if np.any(~np.isfinite(eta_before)) or np.any(~np.isfinite(eta_after)):
            raise _FramePreparationFallback("nonfinite_field")

        kbp = np.empty(nnode, dtype=np.int64)
        sigma_lcl = np.empty((nvrt, nnode), dtype=np.float64)
        for node in range(nnode):
            finite_before = np.flatnonzero(np.isfinite(z_before[node]))
            finite_after = np.flatnonzero(np.isfinite(z_after[node]))
            if finite_before.size == 0 or finite_after.size == 0:
                raise _FramePreparationFallback("invalid_vertical_support")
            first = int(finite_before[0])
            if int(finite_after[0]) != first:
                raise _FramePreparationFallback("vertical_support_mismatch")
            # first finite layer 是原版 kbp；其下未使用層只在輸出場中重複此
            # kbp 值，等價於原版 find_vlevel 的 kbp clamp，不以 0 補物理量。
            if not np.all(np.isfinite(z_before[node, first:])) or not np.all(
                np.isfinite(z_after[node, first:])
            ):
                raise _FramePreparationFallback("invalid_vertical_support")
            denominator_before = float(dp[node] + eta_before[node])
            denominator_after = float(dp[node] + eta_after[node])
            if denominator_before <= 0.0 or denominator_after <= 0.0:
                raise _FramePreparationFallback("invalid_depth")
            sigma_before = (z_before[node, first:] - eta_before[node]) / denominator_before
            sigma_after = (z_after[node, first:] - eta_after[node]) / denominator_after
            if not np.all(np.isfinite(sigma_before)) or not np.all(np.isfinite(sigma_after)):
                raise _FramePreparationFallback("invalid_vertical_support")
            if np.any(np.diff(z_before[node, first:]) < -_SIGMA_TOLERANCE) or np.any(
                np.diff(z_after[node, first:]) < -_SIGMA_TOLERANCE
            ):
                raise _FramePreparationFallback("vertical_nonmonotonic")
            if np.any(np.diff(sigma_before) < -_SIGMA_TOLERANCE) or np.any(
                np.diff(sigma_after) < -_SIGMA_TOLERANCE
            ):
                raise _FramePreparationFallback("sigma_nonmonotonic")
            if (
                np.any(sigma_before < -1.0 - _SIGMA_TOLERANCE)
                or np.any(sigma_before > _SIGMA_TOLERANCE)
                or np.any(sigma_after < -1.0 - _SIGMA_TOLERANCE)
                or np.any(sigma_after > _SIGMA_TOLERANCE)
            ):
                raise _FramePreparationFallback("sigma_out_of_range")
            reconstructed_before = eta_before[node] + sigma_before * denominator_before
            reconstructed_after = eta_after[node] + sigma_after * denominator_after
            if not np.allclose(
                reconstructed_before,
                z_before[node, first:],
                rtol=1.0e-6,
                atol=1.0e-6,
            ) or not np.allclose(
                reconstructed_after,
                z_after[node, first:],
                rtol=1.0e-6,
                atol=1.0e-6,
            ):
                raise _FramePreparationFallback("zcor_reconstruction_mismatch")
            kbp[node] = first
            sigma_lcl[first:, node] = sigma_before
            sigma_lcl[:first, node] = sigma_before[0]

            for values, label in (
                (hvel_before[node, first:, :2], "hvel_before"),
                (hvel_after[node, first:, :2], "hvel_after"),
                (vertical_before[node, first:], "vertical_before"),
                (vertical_after[node, first:], "vertical_after"),
            ):
                if not np.all(np.isfinite(values)):
                    raise _FramePreparationFallback("nonfinite_field", label)

        element_nodes = triangle_nodes
        if np.any(element_nodes < 0) or np.any(element_nodes >= nnode):
            raise _FramePreparationFallback("mesh_topology_invalid")
        kbe = np.min(kbp[element_nodes], axis=1).astype(np.int64, copy=False)
        ne = int(triangle_nodes.shape[0])
        i34 = np.full(ne, 3, dtype=np.int64)
        elnode = np.full((ne, 4), -1, dtype=np.int64)
        elnode[:, :3] = triangle_nodes.astype(np.int64, copy=False)
        ic3 = np.full((4, ne), -1, dtype=np.int64)
        ic3[:3, :] = neighbors.T.astype(np.int64, copy=False)

        def transpose_with_kbp(values: np.ndarray, *, add: float = 0.0) -> np.ndarray:
            """轉成 BayTrace (layer,node)，並以 kbp clamp 重複未使用底層。"""

            result = np.empty((nvrt, nnode), dtype=np.float64)
            for node in range(nnode):
                first = int(kbp[node])
                result[first:, node] = np.asarray(values[node, first:], dtype=np.float64) + add
                result[:first, node] = result[first, node]
            if not np.all(np.isfinite(result)):
                raise _FramePreparationFallback("nonfinite_result")
            return result

        u1 = transpose_with_kbp(hvel_before[:, :, 0])
        v1 = transpose_with_kbp(hvel_before[:, :, 1])
        w1 = transpose_with_kbp(vertical_before, add=settling_velocity_mps)
        u2 = transpose_with_kbp(hvel_after[:, :, 0])
        v2 = transpose_with_kbp(hvel_after[:, :, 1])
        w2 = transpose_with_kbp(vertical_after, add=settling_velocity_mps)
        eta1 = np.asarray(eta_before, dtype=np.float64).copy()
        if not np.all(np.isfinite(eta1)):
            raise _FramePreparationFallback("nonfinite_field")

        grid = BayTraceGridArrays(
            i34=i34,
            elnode=elnode,
            grid_x=np.asarray(node_xy[:, 0], dtype=np.float64),
            grid_y=np.asarray(node_xy[:, 1], dtype=np.float64),
            ic3=ic3,
            nvrt=nvrt,
            kbp=kbp,
            kbe=kbe,
            sigma_lcl=sigma_lcl,
            dp=np.asarray(dp, dtype=np.float64),
        )
        fields = BayTraceFieldPair(
            u1=u1,
            v1=v1,
            w1=w1,
            eta1=eta1,
            u2=u2,
            v2=v2,
            w2=w2,
        )
        return grid, fields

    @staticmethod
    def _frame_metadata(
        manager: Any,
        bracket: _GlobalTimeBracket,
        target_ns: int,
    ) -> dict[str, Any]:
        """保存端點 UTC、observed/reconstructed provenance 與時間 fallback 診斷。"""

        if bracket.before is None or bracket.after is None:
            return {"target_utc_ns": int(target_ns), "time_fallback": 0}
        before_entry, before_index, before_ns = bracket.before
        after_entry, after_index, after_ns = bracket.after
        before_ocm = before_entry.ocm
        after_ocm = after_entry.ocm
        before_diag = before_ocm._row_diagnostics(before_index) if before_ocm is not None else {}
        after_diag = after_ocm._row_diagnostics(after_index) if after_ocm is not None else {}
        before_origin = str(before_diag.get("ocm_time_origin", "observed"))
        after_origin = str(after_diag.get("ocm_time_origin", "observed"))
        metadata: dict[str, Any] = {
            "target_utc_ns": int(target_ns),
            "before_utc_ns": int(before_ns),
            "after_utc_ns": int(after_ns),
            "trat": float(bracket.alpha),
            "before_month_id": str(before_entry.month_id),
            "after_month_id": str(after_entry.month_id),
            "before_row_index": int(before_index),
            "after_row_index": int(after_index),
            "before_source": before_origin,
            "after_source": after_origin,
            "data_source": before_origin if before_origin == after_origin else "mixed",
            "time_fallback": int(bool(bracket.fallback_used)),
            "time_fallback_policy": bracket.fallback_policy,
        }
        diagnostics = manager._time_fallback_diagnostics(
            bracket,
            int(target_ns),
            source="ocm",
        )
        for key, value in diagnostics.items():
            if key.startswith("continuous_time_fallback"):
                metadata[key] = value
        if before_diag:
            metadata["before_row_diagnostics"] = dict(before_diag)
        if after_diag:
            metadata["after_row_diagnostics"] = dict(after_diag)
        return metadata


__all__ = ["OCMPreparedFrameResolver"]
