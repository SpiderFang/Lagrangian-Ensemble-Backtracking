"""WP-63：回溯時間內「仍在流場區域內」的質點比例曲線（每地點線性與對數縱軸各一張，含文字版與無文字版）。

定義
----
對每一顆虛擬質點，回溯結束狀態 ``final_status_code``：

* 1＝已離開流場區域（flow_domain）開放邊界；``backtrack_duration_seconds`` 就是離開的回溯時間；
* 其他（2＝達回溯上限仍在域內、4＝沉底停留等）＝整段回溯期間都沒有離開開放邊界。

回溯時間 τ 時仍在流場區域內的比例 ``inside(τ) = 1 - (離開時間 <= τ 的顆數) / 總顆數``。
曲線是階梯函式的精確值（不分箱），在 0 ～ ``max_days`` 天的格點上取值；90 天與 180 天（可設定）畫垂直標線。
每個地點畫「全部」一條粗線與「十類」十條細線（類別顏色與圓形來源方位圖 B 圖的 viridis 取色一致）。

圖面規則：不寫地點名稱，也不寫站碼，圖上沒有任何座標；站碼只出現在目錄與檔名。含中文處一律傳入字型物件。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..outputs import sha256_file
from .display_names import material_display_name_zh
from .figures import FiguresError, _Labeler, _resolve_font, _tick_font_kwargs

__all__ = [
    "DomainFractionError",
    "EXIT_STATUS_CODE",
    "SiteCurves",
    "inside_fraction",
    "load_site_curves",
    "render_site_figure",
    "build_domain_fraction",
]

EXIT_STATUS_CODE = 1
SECONDS_PER_DAY = 86400.0
CANVAS_IN = (8.0, 5.2)
_GRID_STEP_DAYS = 0.25


class DomainFractionError(RuntimeError):
    """輸入不合法或輸出已存在時 fail-closed。"""


def inside_fraction(
    duration_s: np.ndarray, status: np.ndarray, grid_days: np.ndarray, *, exit_code: int = EXIT_STATUS_CODE
) -> np.ndarray:
    """``inside(τ)``：``grid_days`` 各點（天）上仍在流場區域內的比例（0～1）。純函式。"""

    dur = np.asarray(duration_s, dtype=np.float64)
    st = np.asarray(status)
    if dur.shape != st.shape or dur.ndim != 1:
        raise DomainFractionError("duration_s 與 status 必須是同長度一維陣列")
    n = int(dur.size)
    if n == 0:
        raise DomainFractionError("沒有質點")
    if not np.all(np.isfinite(dur)) or np.any(dur < 0.0):
        raise DomainFractionError("backtrack_duration_seconds 必須是有限非負數")
    grid = np.asarray(grid_days, dtype=np.float64)
    exits = np.sort(dur[st == exit_code] / SECONDS_PER_DAY)
    exited = np.searchsorted(exits, grid, side="right")
    return 1.0 - exited.astype(np.float64) / float(n)


@dataclass(frozen=True, slots=True)
class SiteCurves:
    site_id: str
    grid_days: np.ndarray
    n_particles: int
    all_curve: np.ndarray
    class_ids: tuple[str, ...]
    class_counts: tuple[int, ...]
    class_curves: tuple[np.ndarray, ...]
    max_duration_days: float


def load_site_curves(
    departures: Path, *, max_days: float, sites: Sequence[str] | None = None
) -> dict[str, SiteCurves]:
    """讀 ``departures.parquet``，每地點算「全部」與各類別的曲線。"""

    import pyarrow.parquet as pq

    table = pq.read_table(
        departures,
        columns=["study_site_id", "material_id", "final_status_code", "backtrack_duration_seconds"],
    )
    site = np.asarray(table["study_site_id"].to_pylist(), dtype=object)
    mat = np.asarray(table["material_id"].to_pylist(), dtype=object)
    status = np.asarray(table["final_status_code"])
    dur = np.asarray(table["backtrack_duration_seconds"], dtype=np.float64)
    grid = np.arange(0.0, float(max_days) + 1e-9, _GRID_STEP_DAYS)
    out: dict[str, SiteCurves] = {}
    for sid in sorted(set(site.tolist())):
        if sites is not None and sid not in sites:
            continue
        m = site == sid
        longest = float(dur[m].max()) / SECONDS_PER_DAY
        if longest > float(max_days) + 1e-6:
            raise DomainFractionError(f"站 {sid} 的最長回溯 {longest:.3f} 天超過圖的上限 {max_days} 天")
        classes = sorted(set(mat[m].tolist()))
        curves = []
        counts = []
        for cid in classes:
            mm = m & (mat == cid)
            curves.append(inside_fraction(dur[mm], status[mm], grid))
            counts.append(int(mm.sum()))
        out[sid] = SiteCurves(
            site_id=sid,
            grid_days=grid,
            n_particles=int(m.sum()),
            all_curve=inside_fraction(dur[m], status[m], grid),
            class_ids=tuple(classes),
            class_counts=tuple(counts),
            class_curves=tuple(curves),
            max_duration_days=longest,
        )
    if not out:
        raise DomainFractionError("沒有任何地點可畫")
    return out


def value_at(curve: np.ndarray, grid_days: np.ndarray, day: float) -> float:
    idx = int(np.argmin(np.abs(grid_days - float(day))))
    if abs(float(grid_days[idx]) - float(day)) > 1e-6:
        raise DomainFractionError(f"格點上沒有 {day} 天")
    return float(curve[idx])


def render_site_figure(
    curves: SiteCurves,
    path: Path,
    *,
    mode: str,
    font: Any,
    t: _Labeler,
    marks_days: Sequence[float] = (90.0, 180.0),
    dpi: int = 300,
    case_text: str = "",
    yscale: str = "linear",
) -> None:
    """畫單張圖並排他落檔（PNG 與 SVG）。``mode`` 為 ``full``（含標題、標註）或 ``none``（無文字）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import FuncFormatter

    if mode not in ("full", "none"):
        raise DomainFractionError("mode 只接受 full 或 none")
    if yscale not in ("linear", "log"):
        raise DomainFractionError("yscale 只接受 linear 或 log")
    # 對數軸：比例為 0 的點無法畫，改以「半顆質點」的比例當顯示下限（只影響顯示，不改表與標註的數值）。
    floor_pct = 100.0 * 0.5 / float(curves.n_particles)

    def shown(cur: np.ndarray) -> np.ndarray:
        v = 100.0 * cur
        return np.maximum(v, floor_pct) if yscale == "log" else v

    fig = plt.figure(figsize=CANVAS_IN, dpi=150)
    ax = fig.add_axes((0.10, 0.12, 0.66, 0.74 if mode == "full" else 0.84))
    grid = curves.grid_days
    k = len(curves.class_ids)
    for i, (cid, cur) in enumerate(zip(curves.class_ids, curves.class_curves, strict=True)):
        ax.plot(grid, shown(cur), color=cm.viridis(0.88 * i / max(k - 1, 1)), linewidth=1.1,
                label=material_display_name_zh(cid), zorder=3)
    ax.plot(grid, shown(curves.all_curve), color="black", linewidth=2.4, label=t.t("全部類別", "All classes"), zorder=4)
    top = float(grid[-1])
    for d in marks_days:
        ax.axvline(float(d), color="#B22222", linewidth=1.0, linestyle=(0, (5, 3)), zorder=2)
    ax.set_xlim(0.0, top)
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(min(0.01, floor_pct), 100.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    else:
        ax.set_ylim(0.0, 100.0)
        ax.set_yticks(np.arange(0.0, 100.0 + 1e-9, 20.0))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
    ax.set_xticks(np.arange(0.0, top + 1e-9, 30.0))
    ax.grid(True, color="#DDDDDD", linewidth=0.5, zorder=1)
    tick_kw = _tick_font_kwargs(t, font)
    ax.tick_params(labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if tick_kw:
            lab.set_fontproperties(font)
        lab.set_fontsize(10)
    ax.set_xlabel(t.t("回溯時間（天）", "Backtrack time (days)"), fontproperties=font, fontsize=11)
    ax.set_ylabel(t.t("仍在流場區域內的比例（%）", "Fraction still inside flow domain (%)"), fontproperties=font, fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              prop=font, fontsize=9)
    if mode == "full":
        fig.suptitle(t.t("回溯時間內仍留在流場區域的比例", "Fraction still inside the flow domain vs backtrack time"),
                     fontproperties=font, fontsize=15, y=0.975)
        sub = t.t(f"虛擬質點 {curves.n_particles:,} 顆；紅色虛線為 ", f"{curves.n_particles:,} particles; dashed red at ")
        sub += "、".join(f"{d:g}" for d in marks_days) + t.t(" 天", " d")
        if case_text:
            sub = case_text + "；" + sub
        fig.text(0.5, 0.925, sub, ha="center", va="top", fontproperties=font, fontsize=10)
        for d in marks_days:
            v = 100.0 * value_at(curves.all_curve, grid, d)
            ax.annotate(f"{v:.2g}%" if yscale == "log" else f"{v:.1f}%", xy=(float(d), max(v, floor_pct) if yscale == "log" else v), xytext=(-6, 8),
                        textcoords="offset points", ha="right", fontproperties=font, fontsize=9, color="#B22222",
                        zorder=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=dpi)
    with path.with_suffix(".svg").open("xb") as stream:
        fig.savefig(stream, format="svg", dpi=dpi)
    plt.close(fig)


def write_table(curves_by_site: Mapping[str, SiteCurves], path: Path, marks_days: Sequence[float]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(["study_site_id", "group", "particles"] + [f"inside_pct_at_{d:g}d" for d in marks_days])
        for sid, c in curves_by_site.items():
            w.writerow([sid, "all", c.n_particles] + [f"{100.0 * value_at(c.all_curve, c.grid_days, d):.4f}" for d in marks_days])
            for cid, n, cur in zip(c.class_ids, c.class_counts, c.class_curves, strict=True):
                w.writerow([sid, cid, n] + [f"{100.0 * value_at(cur, c.grid_days, d):.4f}" for d in marks_days])


def build_domain_fraction(
    postprocess_dir: Path,
    out_full: Path,
    out_notext: Path,
    *,
    max_days: float = 180.0,
    marks_days: Sequence[float] = (90.0, 180.0),
    dpi: int = 300,
    font_path: Path | None = None,
    case_text: str = "",
    sites: Sequence[str] | None = None,
) -> list[Path]:
    """產出整個案例的曲線圖（含文字版與無文字版）、表與 manifest；輸出目錄必須是全新目錄。"""

    for d in (out_full, out_notext):
        if d.exists():
            raise DomainFractionError(f"輸出目錄必須是尚不存在的全新目錄：{d}")
    dep = Path(postprocess_dir) / "departures.parquet"
    if not dep.is_file():
        raise DomainFractionError(f"找不到 {dep}")
    before = sha256_file(dep)
    curves = load_site_curves(dep, max_days=max_days, sites=sites)
    try:
        font_res = _resolve_font(font_path)
    except FiguresError as exc:
        raise DomainFractionError(str(exc)) from exc
    t = _Labeler(font_res.cjk_available)
    written: list[Path] = []
    out_full.mkdir(parents=True)
    out_notext.mkdir(parents=True)
    for sid, c in curves.items():
        for mode, base in (("full", out_full), ("none", out_notext)):
            for ys, suffix in (("linear", ""), ("log", "_logy")):
                p = base / sid / f"domain_fraction_vs_backtrack_time{suffix}.png"
                render_site_figure(c, p, mode=mode, font=font_res.font_properties, t=t, marks_days=marks_days,
                                   dpi=dpi, case_text=case_text, yscale=ys)
                written.append(p)
    write_table(curves, out_full / "inside_fraction_summary.csv", marks_days)
    if sha256_file(dep) != before:
        raise DomainFractionError("輸入檔在繪圖過程中被改動，fail-closed")
    manifest = {
        "kind": "domain_fraction_v1",
        "departures_sha256": before,
        "max_days": float(max_days),
        "marks_days": [float(d) for d in marks_days],
        "exit_status_code": EXIT_STATUS_CODE,
        "sites": {sid: {"particles": c.n_particles, "max_backtrack_days": c.max_duration_days} for sid, c in curves.items()},
        "outputs": {str(p.relative_to(out_full.parent)): sha256_file(p) for p in written},
    }
    with (out_full / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
    return written
