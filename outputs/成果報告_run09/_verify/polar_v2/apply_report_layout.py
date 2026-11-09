"""在 lbt-repo-wp63 套用報告版面修改（polar_source_figures.py 與 cli.py）。用法：python apply_report_layout.py <patch.py> <repo_root>"""
import sys
from pathlib import Path

patch = Path(sys.argv[1]).read_text(encoding="utf-8")
root = Path(sys.argv[2])
pf = root / "src/lagrangian_backtracking/native_runner/polar_source_figures.py"
cl = root / "src/lagrangian_backtracking/cli.py"
s = pf.read_text(encoding="utf-8")


def rep(text, old, new, count=1):
    assert text.count(old) == count, (old[:60], text.count(old))
    return text.replace(old, new)


# 1. 報告版面程式碼
marker = "# ── 子行程工作 ───"
s = rep(s, marker, patch.rstrip("\n") + "\n\n\n" + marker)

# 2. 設定欄位與驗證
s = rep(s, "    notext_crop_in: tuple[float, float, float, float] | None = None\n\n    def __post_init__(self) -> None:\n        if self.envelope_km is not None and self.envelope_auto:",
        "    notext_crop_in: tuple[float, float, float, float] | None = None\n    report_layout: bool = False\n\n    def __post_init__(self) -> None:\n"
        "        if self.report_layout:\n"
        "            if self.modes != (\"none\",):\n"
        "                raise PolarSourceError(\"report_layout 只輸出無文字版：modes 必須是 ('none',)\")\n"
        "            if self.envelope_km is not None or self.envelope_auto or self.notext_crop_in is not None:\n"
        "                raise PolarSourceError(\"report_layout 不使用包絡與裁切框設定\")\n"
        "            if abs(self.radius_km - 25.0) > 1e-9:\n"
        "                raise PolarSourceError(\"report_layout 的半徑固定為 25 km\")\n"
        "        if self.envelope_km is not None and self.envelope_auto:")
s = rep(s, "``envelope_km``（四個方向的公里數）明示包絡；", "``report_layout``：報告版面（畫布寬 6.27 英寸、圖例在主圖下方、A 類十圖併成一張多格圖，見 :func:`render_report_overview`），只輸出無文字版；\n    ``envelope_km``（四個方向的公里數）明示包絡；")

# 3. build_polar_source
old = """    if config.envelope_km is not None:
        envelope, envelope_source = {k: float(config.envelope_km[k]) for k in ENVELOPE_SIDES}, "explicit"
    elif config.envelope_auto:"""
new = """    if config.report_layout:
        envelope, envelope_source = dict(_ENVELOPE_KM), "not_applicable_report_layout"
    elif config.envelope_km is not None:
        envelope, envelope_source = {k: float(config.envelope_km[k]) for k in ENVELOPE_SIDES}, "explicit"
    elif config.envelope_auto:"""
s = rep(s, old, new)
old = """    if config.notext_crop_in is not None:
        layout_crop, crop_source = tuple(config.notext_crop_in), "explicit\""""
new = """    if config.report_layout:
        layout_crop, crop_source = NOTEXT_CROP_IN, "not_applicable_report_layout"
    elif config.notext_crop_in is not None:
        layout_crop, crop_source = tuple(config.notext_crop_in), "explicit\""""
s = rep(s, old, new)
old = """    figure_layout_record = {
        "envelope_km_at_radius_25": layout.envelope(),"""
new = """    if config.report_layout:
        figure_layout_record = {
            "report_layout": True,
            "canvas_in": {"overview": [REPORT_W_IN, REPORT_OVERVIEW_H], "panels": [REPORT_W_IN, REPORT_PANELS_H]},
            "window_half_km": config.radius_km * REPORT_WINDOW_FACTOR,
            "overview_frame_in": _RO_FRAME,
            "panel_frame_in": _RG_P,
            "text_pt": REPORT_TEXT_PT,
            "stems": dict(REPORT_STEMS),
        }
    else:
        figure_layout_record = {
        "envelope_km_at_radius_25": layout.envelope(),"""
s = rep(s, old, new)
# 把 legacy record 的其餘行縮排
old = """        "envelope_source": envelope_source,
        "notext_crop_in": list(layout.notext_crop_in) if layout.notext_crop_in is not None else None,
        "notext_crop_source": crop_source,
        "canvas_in": [CANVAS_W_IN, CANVAS_H_IN],
        "in_per_km_at_radius_25": layout.in_per_km_25,
        "anchor_position_in": [layout.anchor_x_in, layout.anchor_y_in],
    }"""
new = """        "envelope_source": envelope_source,
        "notext_crop_in": list(layout.notext_crop_in) if layout.notext_crop_in is not None else None,
        "notext_crop_source": crop_source,
        "canvas_in": [CANVAS_W_IN, CANVAS_H_IN],
        "in_per_km_at_radius_25": layout.in_per_km_25,
        "anchor_position_in": [layout.anchor_x_in, layout.anchor_y_in],
        }"""
s = rep(s, old, new)
# render jobs: 報告版面用另一個 job 函式
old = """    written: list[str] = []
    if config.workers <= 1 or len(render_jobs) <= 1:
        for job in render_jobs:
            written.extend(_render_job(job))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=min(config.workers, len(render_jobs)), mp_context=mp.get_context("fork")) as pool:
            for out in pool.map(_render_job, render_jobs):
                written.extend(out)"""
new = """    written: list[str] = []
    job_fn = _render_report_job if config.report_layout else _render_job
    if config.workers <= 1 or len(render_jobs) <= 1:
        for job in render_jobs:
            written.extend(job_fn(job))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=min(config.workers, len(render_jobs)), mp_context=mp.get_context("fork")) as pool:
            for out in pool.map(job_fn, render_jobs):
                written.extend(out)"""
s = rep(s, old, new)
# manifest 目錄需存在
old = """        _write_json_exclusive(case.text_dir / "manifest.json", manifest)"""
new = """        case.text_dir.mkdir(parents=True, exist_ok=True)
        manifest["report_layout"] = bool(config.report_layout)
        _write_json_exclusive(case.text_dir / "manifest.json", manifest)"""
s = rep(s, old, new)
old = """        _write_text_exclusive(
            case.text_dir / "README.md",
            "# 圓形來源方位圖（WP-60）\\n\\n\""""
new = """        if config.report_layout:
            _write_text_exclusive(
                case.text_dir / "README.md",
                "# 圓形來源方位圖：報告版面（WP-65）\\n\\n"
                f"案例：`{case.case_id}`。只改版面、不改數值：畫布寬 6.27 英寸（＝報告插入寬度），字級 8 pt，圖例在主圖正下方；"
                "每地點 4 張圖（十類總覽、十類各自十個沉降速度之多格圖、季節、大小潮），圖檔在同層 `figures-polar-v2-notext/<地點>/`。"
                "統計表、summary.md 與 captions.md 沿用 `figures-polar/`（數值相同）。\\n\\n"
                f"重建命令：\\n\\n```\\n{command}\\n```\\n",
            )
            outputs.append(case.text_dir)
            continue
        _write_text_exclusive(
            case.text_dir / "README.md",
            "# 圓形來源方位圖（WP-60）\\n\\n\""""
s = rep(s, old, new)
# captions 在 report 模式也寫（已在 README 前），保持；jobs 的 mode 與 layout 欄位沿用
pf.write_text(s, encoding="utf-8")

# 4. CLI
c = cl.read_text(encoding="utf-8")
old = """    parser.add_argument(
        "--notext-crop-in", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"), default=None,"""
new = """    parser.add_argument(
        "--report-layout", action="store_true",
        help="報告版面：畫布寬 6.27 英寸（＝報告插入寬度）、字級 8 pt、圖例在主圖下方、A 類十圖併成一張多格圖；只輸出無文字版（需 --modes none）",
    )
    parser.add_argument(
        "--notext-crop-in", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"), default=None,"""
c = rep(c, old, new)
old = """        notext_crop_in=tuple(args.notext_crop_in) if args.notext_crop_in is not None else None,
    )
    output_dirs = native_polar_source_figures.build_polar_source"""
new = """        notext_crop_in=tuple(args.notext_crop_in) if args.notext_crop_in is not None else None,
        report_layout=bool(args.report_layout),
    )
    output_dirs = native_polar_source_figures.build_polar_source"""
c = rep(c, old, new)
cl.write_text(c, encoding="utf-8")
print("ok")
