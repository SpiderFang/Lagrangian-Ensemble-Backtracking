# ruff: noqa: E501, E402, SIM115
"""WP-61：50 個到達（觀測）時刻的潮位／波浪／流速驗證——資料重現、獨立重算名次、統計表與 summary 數字。

只讀。序列由 ``stage1.py`` 以 ``input_derivation`` 的同一批函式（``_surface_series_for_location``、
``_nww_series_for_location``、``_select_arrivals_for_site``）重現後存成 ``data/<site>.npz``；
本模組讀取 manifest（run08 native-plan 指定的 ``common-input/arrival.json``）的 ``observation_*`` 欄位，
以另寫的排名程式重新挑選，並算各項核對數字。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

W = Path("/home/mustlab/Workspace")
INP = W / "lbt-inputs/formal/v7-class-velocity/20261013_final-sites_h30-h60-h90_common_m10_v7"
DATA = W / "lbt-wp61/data"
SITES = ("gongliao", "guishan", "hsinchu", "lienchiang", "nanwan")
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_OF_MONTH = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
PHASES = ("fastest_rising", "fastest_falling", "slack_proxy")
CLASSES = ("spring_proxy", "neap_proxy")
HOUR_NS = 3_600_000_000_000
YEAR = 2025


@dataclass
class SiteSeries:
    site: str
    times: np.ndarray
    eta: np.ndarray
    speed: np.ndarray
    hs: np.ndarray
    strength: np.ndarray
    deriv: np.ndarray
    valid: np.ndarray
    anchor: tuple[float, float]
    metric: tuple[float, float]
    year_of: np.ndarray
    month_of: np.ndarray
    season_of: np.ndarray


def load_series(site: str) -> SiteSeries:
    z = np.load(DATA / f"{site}.npz")
    t = z["times"]
    dt = t.astype("datetime64[ns]")
    years = dt.astype("datetime64[Y]").astype(int) + 1970
    months = dt.astype("datetime64[M]").astype(int) % 12 + 1
    seasons = np.array([SEASON_OF_MONTH[int(m)] for m in months])
    return SiteSeries(
        site,
        t,
        z["eta"],
        z["speed"],
        z["hs"],
        z["strength"],
        z["deriv"],
        z["valid"].astype(bool),
        tuple(z["anchor"]),
        tuple(z["metric"]),
        years,
        months,
        seasons,
    )


def load_manifest_records() -> dict[str, list[dict]]:
    """每地點 50 筆觀測時刻（由 2,500 筆沉底紀錄的 ``observation_*`` 欄位去重而得）。"""

    arr = json.load(open(INP / "common-input/arrival.json"))
    by: dict[str, dict[str, dict]] = {}
    for r in arr["records"]:
        m = r["metadata"]
        by.setdefault(r["study_site_id"], {}).setdefault(m["observation_arrival_time_id"], m)
    out = {}
    for site, d in by.items():
        rows = sorted(d.values(), key=lambda m: m["observation_time_utc_ns"])
        out[site] = rows
    return out


def independent_selection(s: SiteSeries, year: int = YEAR) -> dict:
    """另寫的名次重算：同一份序列、同樣規則（季節內潮汐強度中位數分大潮／小潮；各潮況內依
    上升最快／下降最快／最接近平潮取前兩名；全域不重複；事件取其餘候選中 Hs／流速最大者）。"""

    selected: list[int] = []
    picks: dict[tuple[str, str, str], list[int]] = {}
    thresholds: dict[str, float] = {}
    for season in SEASONS:
        idx = np.flatnonzero(
            s.valid & (s.year_of == year) & (s.season_of == season) & np.isfinite(s.strength)
        )
        thr = float(np.median(s.strength[idx]))
        thresholds[season] = thr
        groups = {"spring_proxy": idx[s.strength[idx] >= thr], "neap_proxy": idx[s.strength[idx] < thr]}
        for cls in CLASSES:
            ci = groups[cls]
            for phase, key in (
                ("fastest_rising", lambda i: -s.deriv[i]),
                ("fastest_falling", lambda i: s.deriv[i]),
                ("slack_proxy", lambda i: abs(s.deriv[i])),
            ):
                cand = [int(i) for i in ci if int(i) not in selected and np.isfinite(s.deriv[i])]
                cand.sort(key=lambda i: (key(i), int(s.times[i])))
                picks[(season, cls, phase)] = cand[:2]
                selected += cand[:2]
    remaining = [int(i) for i in np.flatnonzero(s.valid & (s.year_of == year)) if int(i) not in selected]
    events = {}
    for name, score in (("high_wave_event", s.hs), ("strong_current_event", s.speed)):
        best = min(remaining, key=lambda i: (-float(score[i]), int(s.times[i])))
        events[name] = best
        remaining.remove(best)
    return {"picks": picks, "events": events, "thresholds": thresholds}


def zero_crossings(s: SiteSeries) -> np.ndarray:
    """dη/dt 變號點（高潮、低潮）的時刻（小時序號，含線性內插）；跨缺口的相鄰點不內插。"""

    d = s.deriv
    t_h = s.times.astype(np.float64) / HOUR_NS
    ok = np.isfinite(d[:-1]) & np.isfinite(d[1:]) & (np.diff(s.times) == HOUR_NS)
    sign = (d[:-1] * d[1:] < 0) & ok
    i = np.flatnonzero(sign)
    return t_h[i] + (t_h[i + 1] - t_h[i]) * d[i] / (d[i] - d[i + 1])


def nearest_extreme_hours(s: SiteSeries, idx: int, zc: np.ndarray) -> float:
    t = float(s.times[idx]) / HOUR_NS
    j = int(np.searchsorted(zc, t))
    c = [abs(zc[k] - t) for k in (j - 1, j) if 0 <= k < zc.size]
    return float(min(c)) if c else float("nan")


def pctl_rank(series: np.ndarray, mask: np.ndarray, value: float) -> float:
    v = series[mask & np.isfinite(series)]
    return float(100.0 * np.mean(v <= value))


def analyze_site(site: str, recs: list[dict], s: SiteSeries, ref: SiteSeries | None = None) -> dict:
    """單站核對。``ref`` 為借用其挑選時刻的來源站序列（龜山島借貢寮），其餘為 None。"""

    ind = independent_selection(s)
    zc = zero_crossings(s)
    year_mask = s.valid & (s.year_of == YEAR)
    idx_of = {int(t): i for i, t in enumerate(s.times)}
    # 若為借用站，額外取來源站（挑選依據）的獨立名次，作為「manifest 排名依據」
    ind_ref = independent_selection(ref) if ref is not None else None
    zero_crossings(ref) if ref is not None else None
    manifest_picks: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    rows = []
    for m in recs:
        t = int(m["observation_time_utc_ns"])
        i = idx_of[t]
        cls, ph, season = (
            m["observation_tide_class"],
            m["observation_phase_or_event"],
            m["observation_season"],
        )
        row = {
            "site": site,
            "utc": m["observation_time_utc"],
            "month": int(s.month_of[i]),
            "season": season,
            "tide_class": cls,
            "phase_or_event": ph,
            "replicate_rank_manifest": m.get("replicate_rank_one_based", ""),
            "eta_m": float(s.eta[i]),
            "deta_dt_mm_per_min": float(s.deriv[i] * 1000 * 60),
            "tidal_strength_m": float(s.strength[i]),
            "season_threshold_m": ind["thresholds"].get(season, float("nan")) if cls != "event" else "",
            "hs_m": float(s.hs[i]),
            "speed_mps": float(s.speed[i]),
            "month_season_consistent": SEASON_OF_MONTH[int(s.month_of[i])] == season,
            "year": int(s.year_of[i]),
        }
        if cls in CLASSES:
            thr = ind["thresholds"][season]
            own_cls = "spring_proxy" if s.strength[i] >= thr else "neap_proxy"
            row["class_rule_consistent"] = own_cls == cls
            row["sign_consistent"] = bool(
                (s.deriv[i] > 0)
                if ph == "fastest_rising"
                else ((s.deriv[i] < 0) if ph == "fastest_falling" else True)
            )
            row["hours_to_nearest_extreme"] = nearest_extreme_hours(s, i, zc)
            key = (season, cls, ph)
            manifest_picks.setdefault(key, []).append((int(m.get("replicate_rank", 0)), i))
        else:
            row["class_rule_consistent"] = ""
            row["sign_consistent"] = ""
            row["hours_to_nearest_extreme"] = ""
            score = s.hs if ph == "high_wave_event" else s.speed
            row["event_value"] = float(score[i])
            row["event_year_max"] = float(np.nanmax(np.where(year_mask, score, np.nan)))
            row["event_year_p99"] = float(np.nanpercentile(np.where(year_mask, score, np.nan), 99))
            row["event_percentile_rank"] = pctl_rank(score, year_mask, float(score[i]))
            row["event_is_year_max_of_remaining"] = int(ind["events"][ph]) == i
        rows.append(row)

    # 名次比對（以本站序列重算；借用站則另以來源站序列重算作為 manifest 的實際挑選依據）
    def compare(indsel: dict, label: str):
        exact = 0
        setmatch = 0
        n = 0
        per = {}
        for key, got in indsel["picks"].items():
            want = sorted(manifest_picks.get(key, []), key=lambda x: x[0])
            want_idx = [times_idx for _, times_idx in want]
            n += 1
            per[key] = (got == want_idx, set(got) == set(want_idx))
            exact += got == want_idx
            setmatch += set(got) == set(want_idx)
        return {"strata": n, "exact": exact, "set": setmatch, "per": per}

    cmp_own = compare(ind, "own")
    cmp_ref = compare(ind_ref, "ref") if ind_ref is not None else None
    # 個別時刻是否出現在重算的前兩名（時刻層級）
    {i for v in ind["picks"].values() for i in v}
    {i for v in ind_ref["picks"].values() for i in v} if ind_ref is not None else None
    for row, m in zip(rows, recs, strict=False):
        i = idx_of[int(m["observation_time_utc_ns"])]
        if m["observation_tide_class"] in CLASSES:
            key = (m["observation_season"], m["observation_tide_class"], m["observation_phase_or_event"])
            row["independent_top2_contains"] = i in ind["picks"][key]
            row["independent_rank"] = (ind["picks"][key].index(i) + 1) if i in ind["picks"][key] else ""
            row["independent_rank_equals_manifest_rank"] = row["independent_rank"] == m.get(
                "replicate_rank_one_based"
            )
            if ind_ref is not None:
                row["source_series_top2_contains"] = i in ind_ref["picks"][key]
                row["source_series_rank"] = (
                    (ind_ref["picks"][key].index(i) + 1) if i in ind_ref["picks"][key] else ""
                )
        else:
            row["independent_top2_contains"] = int(ind["events"][m["observation_phase_or_event"]]) == i
            row["independent_rank"] = ""
            row["independent_rank_equals_manifest_rank"] = ""
    return {
        "rows": rows,
        "ind": ind,
        "cmp_own": cmp_own,
        "cmp_ref": cmp_ref,
        "zc": zc,
        "idx_of": idx_of,
        "ind_ref": ind_ref,
    }


def write_site_csv(path: Path, rows: list[dict]) -> None:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
