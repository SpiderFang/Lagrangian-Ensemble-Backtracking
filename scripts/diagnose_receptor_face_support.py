#!/usr/bin/env python
"""WP-51 整合：只跑受體候選池 face_support 的診斷（不建母體）。

背景：per_member 母體建置在 ``_receptor_payload`` 的 ``select_supported_from_pool`` 丟出
「垂向支撐淘汰後候選不足：需要 5，實際 0」，但 ``face_support_errors`` 沒有輸出，無法分辨是
NWW 還是 OCM、是哪些沉底整點造成。本腳本以與 input-build 相同的 ``_receptor_payload``（含真實
受體候選池、persistent-wet、邊界距離、隨機順序）逐站執行，只把 ``_nww_exact_hour_samples`` 與
``_build_face_vertical_support`` 包一層記錄器，統計：

* 每站候選池大小、被檢查的面數、通過／失敗面數；
* 失敗原因分類（NWW 空間靜態四角無效、NWW 特定時刻無效、OCM 垂向支撐）與訊息彙總；
* NWW 失敗時刻：各時刻在多少個被檢查面上無效、是否在 NWW 時間軸內、是否落在 OCM 整日缺口邊緣
  （距最近缺口的小時數）；
* OCM 失敗：第一個失敗的沉底時刻與訊息分佈。

沉底時刻的來源（``--mode``）：

* ``per_member``：以 v3 母體的 50 個觀測時刻與各站 OCM canonical 可用軸，呼叫與正式流程相同的
  ``sample_bed_residence_member_ages_conditioned_on_availability``（設定檔的種子與 M），每站 500 個；
* ``per_observation``：直接用 v3 母體 arrival.json 的 50 個沉底時刻（對照組，證明同一函式在 v3
  情形可通過，排除 B2 標籤接線的影響——受體階段本來就不讀標籤）。

``--exclude-times-file`` 可提供「每站要排除的整點」（JSON：{site: [utc_ns,...]}），用來模擬把
不可支撐整點納入沉底年齡可用性條件之後的結果。只讀輸入，不寫母體。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lagrangian_backtracking import input_derivation as idv
from lagrangian_backtracking.bed_residence import (
    sample_bed_residence_member_ages_conditioned_on_availability as sample_member,
)
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.scenarios import ArrivalTime

HOUR_NS = 3_600 * 10**9


class _StopDiagnosis(Exception):
    """已檢查足夠面數，提早結束本站診斷。"""


class _Proxy:
    """只暴露一個站點的 config 代理（其餘屬性原樣轉發）。"""

    def __init__(self, config: Any, site_id: str) -> None:
        self._config = config
        self.study_sites = [s for s in config.study_sites if s.study_site_id == site_id]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


def _gap_distance_hours(time_ns: int, available: set[int]) -> int:
    """距最近一個「不在可用軸」的整點幾小時（可用軸上的點才有意義；不可用回傳 0）。"""

    if time_ns not in available:
        return 0
    for k in range(1, 200):
        if (time_ns - k * HOUR_NS) not in available or (time_ns + k * HOUR_NS) not in available:
            return k
    return 200


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="v4 範本或 release config")
    ap.add_argument("--population", type=Path, required=True, help="v3 母體目錄（含 common-input/arrival.json）")
    ap.add_argument("--ocm-native-root", type=Path, required=True)
    ap.add_argument("--nww-analysis-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["per_member", "per_observation"], default="per_member")
    ap.add_argument("--sites", nargs="+", default=None)
    ap.add_argument("--max-faces", type=int, default=400, help="每站最多檢查幾個候選面（預設 400；0 不限）")
    ap.add_argument("--exclude-times-file", type=Path, default=None,
                    help="（事後排除，不重新抽樣）每站要從沉底時刻中刪除的整點 JSON")
    ap.add_argument("--nww-conditioned", action="store_true",
                    help="把沉底年齡可用性條件加上「NWW 在該站 metric location 的 exact-hour 四角支撐有效」，"
                         "即正式修正的做法（重新抽樣，不是事後刪除）")
    ap.add_argument("--output", type=Path, required=True, help="輸出 JSON（必須尚不存在）")
    args = ap.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"拒絕覆寫：{args.output}")

    config = load_config(args.config, formal_release=False)
    months = idv._months_for_config(config)
    native_root = idv._env_root(args.ocm_native_root, config.inputs.ocm_native_root_env)
    nww_root = idv._env_root(args.nww_analysis_root, config.inputs.nww_analysis_root_env)
    bed = config.scenarios.bed_residence_time
    member_count = int(config.scenarios.members_per_scenario)

    ocm_by_region: dict[str, Any] = {}
    nww_by_region: dict[str, Any] = {}
    for domain in sorted(config.domains, key=lambda d: d.analysis_region_id):
        region = domain.analysis_region_id
        flow_id = idv._resolve_flow_product_id(config, region, native_root, formal=True)
        ocm_by_region[region] = idv._load_product(
            product="ocm_native", root=native_root, root_token=config.inputs.ocm_native_root_env,
            flow_domain_id=flow_id, months=months, config=config,
        )
        nww_by_region[region] = idv._load_product(
            product="nww3_analysis", root=nww_root, root_token=config.inputs.nww_analysis_root_env,
            flow_domain_id=idv._resolve_flow_product_id(config, region, nww_root, formal=True),
            months=months, config=config,
        )
    nww_caches = {r: idv._load_nww_runtime_cache(nww_by_region[r]) for r in sorted(nww_by_region)}
    pair_caches = {r: idv._load_ocm_pair_cache(ocm_by_region[r]) for r in sorted(ocm_by_region)}
    source_hashes = idv._source_hashes_for_products(list(ocm_by_region.values()) + list(nww_by_region.values()))
    _d, _l, _o, flow_polygons, local_polygons = idv._geometry_payloads(
        config=config, products_by_region=ocm_by_region, source_hashes=dict(source_hashes), strict=True
    )

    arrival_doc = json.load(open(args.population / "common-input" / "arrival.json"))
    obs: dict[str, list[int]] = defaultdict(list)
    v3_dep: dict[str, list[int]] = defaultdict(list)
    for rec in arrival_doc["records"]:
        obs[rec["study_site_id"]].append(int(rec["metadata"]["observation_time_utc_ns"]))
        v3_dep[rec["study_site_id"]].append(int(rec["time_utc_ns"]))
    exclude: dict[str, set[int]] = {}
    if args.exclude_times_file is not None:
        exclude = {k: {int(v) for v in vals} for k, vals in json.load(open(args.exclude_times_file)).items()}

    site_by_region = {s.study_site_id: s.analysis_region_id for s in config.study_sites}
    available = {
        sid: {int(v) for v in ocm_by_region[reg].canonical.time_utc_ns} for sid, reg in site_by_region.items()
    }
    nww_axis = {
        reg: {int(v) for month in nww_caches[reg].months for v in month.source.time_ns} for reg in nww_caches
    }

    # 各站 NWW metric location（取自 v3 arrival.json；錨點優先、否則核心內最近的支撐 cell 中心）
    metric_lonlat: dict[str, tuple[float, float]] = {}
    for rec in arrival_doc["records"]:
        md = rec["metadata"]
        metric_lonlat.setdefault(
            rec["study_site_id"], (float(md["metric_location_lon"]), float(md["metric_location_lat"]))
        )
    # 錨點型站點改用目前設定檔的 anchor（龜山島 WP-51 C 移到海上後，v3 的舊 anchor 已不適用）
    anchor_by_site = {s.study_site_id: s.anchor_lonlat for s in config.study_sites}
    kind_by_site = {
        rec["study_site_id"]: rec["metadata"].get("metric_location_kind") for rec in arrival_doc["records"]
    }
    for sid, kind in kind_by_site.items():
        anchor = anchor_by_site.get(sid)
        if kind == "anchor" and anchor is not None:
            metric_lonlat[sid] = (float(anchor[0]), float(anchor[1]))
    metric_valid: dict[str, set[int]] = {}
    for sid, (mlon, mlat) in metric_lonlat.items():
        _hs, valid = idv._nww_exact_hour_samples(nww_caches[site_by_region[sid]], lon=mlon, lat=mlat)
        metric_valid[sid] = {t for t, ok in valid.items() if ok}
    # 沉底時刻
    deposition: dict[str, list[int]] = {}
    if args.mode == "per_member":
        sorted_obs = {s: sorted(v) for s, v in obs.items()}
        res = sample_member(
            observation_times_by_site=sorted_obs,
            available_time_ns_by_site={
                s: (available[s] & metric_valid[s] if args.nww_conditioned else available[s]) for s in sorted_obs
            },
            maximum_age_days=int(bed.maximum_age_days),
            sample_count=int(bed.sample_count_per_site),
            member_count=member_count,
            seed=int(bed.effective_sampling_seed),
        )
        age = np.asarray(res.age_hours)
        for sid, series in sorted_obs.items():
            deposition[sid] = [
                int(series[r]) - int(age[r, m]) * HOUR_NS for r in range(age.shape[0]) for m in range(age.shape[1])
            ]
    else:
        deposition = {s: sorted(v) for s, v in v3_dep.items()}

    report: dict[str, Any] = {
        "mode": args.mode,
        "config": str(args.config),
        "member_count": member_count,
        "sampling_seed": int(bed.effective_sampling_seed),
        "nww_conditioned": bool(args.nww_conditioned),
        "metric_location_nww_invalid_hours": {
            sid: sum(1 for t in available[sid] if t not in metric_valid[sid]) for sid in sorted(metric_lonlat)
        },
        "sites": {},
    }
    original_nww = idv._nww_exact_hour_samples
    original_support = idv._build_face_vertical_support

    for site_id in sorted(args.sites or deposition):
        region = site_by_region[site_id]
        times = [t for t in deposition[site_id] if t not in exclude.get(site_id, set())]
        arrivals = [
            ArrivalTime(
                arrival_time_id=f"diag-{site_id}-{i:04d}", study_site_id=site_id, time_utc_ns=int(t),
                year=2025, season="winter", tide_class="spring", phase_or_event="slack", metadata={},
            )
            for i, t in enumerate(sorted(times))
        ]
        order = [a.time_utc_ns for a in arrivals]
        stat: dict[str, Any] = {
            "deposition_time_count": len(order),
            "excluded_time_count": len(deposition[site_id]) - len(order),
            "faces": 0, "nww_static_invalid_faces": 0, "nww_time_invalid_faces": 0, "ocm_failed_faces": 0,
            "passed_faces": 0,
        }
        invalid_by_time: Counter[int] = Counter()
        ocm_first_fail_time: Counter[int] = Counter()
        ocm_messages: Counter[str] = Counter()
        state = {"ocm_ok_since_face": 0, "face_open": False, "limit_hit": False}

        def close_face(_ok: bool, stat: dict[str, Any] = stat, state: dict[str, Any] = state) -> None:
            state["face_open"] = False

        def nww_wrapper(cache: Any, *, lon: float, lat: float, requested_times: Any = None,
                        stat: dict[str, Any] = stat, state: dict[str, Any] = state,
                        invalid_by_time: Counter[int] = invalid_by_time, order: list[int] = order) -> Any:
            if requested_times is not None and tuple(requested_times) == tuple(order):
                if args.max_faces and stat["faces"] >= args.max_faces:
                    state["limit_hit"] = True
                    raise _StopDiagnosis()
                stat["faces"] += 1
                state["ocm_ok_since_face"] = 0
            result = original_nww(cache, lon=lon, lat=lat, requested_times=requested_times)
            if requested_times is not None and tuple(requested_times) == tuple(order):
                valid = result[1]
                bad = [t for t in order if not valid.get(t, False)]
                if len(bad) == len(order):
                    stat["nww_static_invalid_faces"] += 1
                elif bad:
                    stat["nww_time_invalid_faces"] += 1
                    for t in bad:
                        invalid_by_time[t] += 1
            return result

        def support_wrapper(*a: Any, state: dict[str, Any] = state, stat: dict[str, Any] = stat,
                            ocm_first_fail_time: Counter[int] = ocm_first_fail_time,
                            ocm_messages: Counter[str] = ocm_messages, order: list[int] = order,
                            **kw: Any) -> Any:
            try:
                out = original_support(*a, **kw)
            except idv.InputDerivationError as exc:
                stat["ocm_failed_faces"] += 1
                index = state["ocm_ok_since_face"]
                if index < len(order):
                    ocm_first_fail_time[order[index]] += 1
                ocm_messages[str(exc)[:160]] += 1
                raise
            state["ocm_ok_since_face"] += 1
            return out

        idv._nww_exact_hour_samples = nww_wrapper
        idv._build_face_vertical_support = support_wrapper
        outcome = "ok"
        try:
            idv._receptor_payload(
                config=_Proxy(config, site_id), products_by_region=ocm_by_region,
                nww_products_by_region=nww_by_region, flow_polygons=flow_polygons,
                local_polygons=local_polygons, arrivals_by_site={site_id: arrivals},
                nww_runtime_caches=nww_caches, ocm_pair_caches=pair_caches, vertical_support_cache={},
                source_hashes=source_hashes, strict=True,
            )
        except _StopDiagnosis:
            outcome = f"已檢查 {args.max_faces} 個面仍未結束（診斷中止）"
        except idv.InputDerivationError as exc:
            outcome = f"InputDerivationError：{str(exc)[:300]}"
        finally:
            idv._nww_exact_hour_samples = original_nww
            idv._build_face_vertical_support = original_support
        stat["passed_faces"] = stat["faces"] - stat["nww_static_invalid_faces"] - stat["nww_time_invalid_faces"] - stat["ocm_failed_faces"]
        stat["outcome"] = outcome
        stat["ocm_failure_messages"] = dict(ocm_messages.most_common(10))

        def describe(t: int) -> dict[str, Any]:
            return {
                "utc": np.datetime64(int(t), "ns").astype("datetime64[s]").astype(str) + "Z",
                "utc_ns": int(t),
                "in_nww_time_axis": int(t) in nww_axis[region],
                "in_ocm_available_axis": int(t) in available[site_id],
                "hours_to_nearest_ocm_gap": _gap_distance_hours(int(t), available[site_id]),
            }

        stat["nww_invalid_times"] = [
            {**describe(t), "invalid_faces": n} for t, n in invalid_by_time.most_common(40)
        ]
        stat["nww_invalid_time_count"] = len(invalid_by_time)
        stat["ocm_first_fail_times"] = [
            {**describe(t), "faces": n} for t, n in ocm_first_fail_time.most_common(40)
        ]
        # 全部面都失敗的時刻 = 該時刻對所有被檢查且 NWW 靜態有效的面都不可支撐
        spatially_ok = stat["faces"] - stat["nww_static_invalid_faces"]
        stat["times_invalid_for_all_spatially_valid_faces"] = [
            int(t) for t, n in invalid_by_time.items() if spatially_ok > 0 and n >= spatially_ok
        ]
        stat["nww_invalid_times_also_invalid_at_metric_location"] = sum(
            1 for t in invalid_by_time if t not in metric_valid[site_id]
        )
        report["sites"][site_id] = stat
        print(f"[{site_id}] 面 {stat['faces']}，通過 {stat['passed_faces']}，NWW 靜態無效 {stat['nww_static_invalid_faces']}，"
              f"NWW 時刻無效 {stat['nww_time_invalid_faces']}，OCM 失敗 {stat['ocm_failed_faces']}；{outcome}", flush=True)

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
