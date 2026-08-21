"""產生程式碼導覽文件第 3、4 節的可列印流程圖。

本腳本把 ``docs/11_source_code_guide_and_plan_traceability.md`` 中的兩張 Mermaid 概念圖
轉為獨立的 A4 橫式 PDF。輸出刻意只呈現模組責任、資料流與一條粒子的處理順序，不把
尚未產生的實值科學結果畫成已完成成果。PDF 適合報告與交接文件引用；同一 PDF 的每頁
可再用 Poppler 轉為高解析度 PNG，供簡報或文件嵌入。

輸入不讀取大型 OCM、NWW3 或軌跡資料；所有文字均來自已版本化的文件 11，因此此腳本
不會接觸 SERVER 資料，也不會改變科學計算結果。輸出位置由命令列指定，預設為
``output/pdf/source_code_flow_diagrams.pdf``。若日後模組責任或流程改變，必須先更新文件
11，再同步更新本腳本的方塊與箭頭文字，避免圖與文件不一致。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

# A4 橫式頁面可在一頁內清楚放入四個程式群組；所有座標均以 points 表示。
PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
MARGIN = 30.0
TITLE_FONT = "SourceGuideChinese"
BODY_FONT = "SourceGuideChinese"
CODE_FONT = "Helvetica"
# 使用本機繁體中文字型的第一個子字型，ReportLab 會將實際用到的字元嵌入 PDF，讓讀者端
# 不必另安裝中文字型。路徑是本機 macOS 的系統字型；成品 PDF 已嵌入必要字元。
CHINESE_FONT_PATH = Path("/System/Library/Fonts/STHeiti Light.ttc")

# 每一群組使用固定但低飽和度的顏色，讓列印成灰階時仍可依框線及位置辨識。
GROUP_COLORS = {
    "data": HexColor("#DCEAF7"),
    "physics": HexColor("#DDEFE1"),
    "execution": HexColor("#F9E8C7"),
    "output": HexColor("#E9E0F3"),
    "neutral": HexColor("#EDEFF2"),
}
GROUP_BORDERS = {
    "data": HexColor("#477DAA"),
    "physics": HexColor("#4C8960"),
    "execution": HexColor("#AE7520"),
    "output": HexColor("#7D5E9A"),
    "neutral": HexColor("#5E6874"),
}
TEXT_COLOR = HexColor("#17212B")
ARROW_COLOR = HexColor("#455A6F")


def register_fonts() -> None:
    """註冊並嵌入繁體中文字型，避免 PDF 讀者缺少 CMap 時出現空白字。

    先前若只用 PDF 的 CID 參照字型，部分 Poppler 環境沒有對應的中文字碼表，轉成 PNG
    時會遺失文字。此處改以 TrueType Collection 的第一個中文字型子集嵌入輸出 PDF。
    系統中找不到該字型時立即失敗，避免產生看似成功但文字不可讀的圖檔。程式碼與函式
    名稱仍改用 Helvetica，避免中文字型對底線、括號等程式符號的字距造成閱讀困難。
    """

    if not CHINESE_FONT_PATH.is_file():
        raise FileNotFoundError(f"缺少繁體中文字型，無法安全產製流程圖：{CHINESE_FONT_PATH}")
    pdfmetrics.registerFont(TTFont(TITLE_FONT, str(CHINESE_FONT_PATH), subfontIndex=0))


def draw_text_lines(
    canvas: Canvas,
    lines: Iterable[str],
    *,
    x: float,
    y_top: float,
    font_name: str,
    font_size: float,
    leading: float,
    color: Color = TEXT_COLOR,
) -> None:
    """由上而下繪製固定行距文字。

    流程圖中的文字都已在呼叫端手動分行。這種作法比依字數自動換行更可預期，特別是
    中英文混排和函式名稱並存時，可避免 PDF 在不同字型替代環境中出現截斷或重疊。
    """

    canvas.setFillColor(color)
    canvas.setFont(font_name, font_size)
    for line_index, line in enumerate(lines):
        canvas.drawString(x, y_top - line_index * leading, line)


def draw_box(
    canvas: Canvas,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title_lines: tuple[str, ...],
    body_lines: tuple[str, ...] = (),
    style: str = "neutral",
    title_is_code: bool = False,
) -> None:
    """繪製一個可讀的流程方塊。

    方塊上半部放檔案或函式名稱，下半部用白話中文說明責任。輸入 ``x, y`` 是左下角，
    因為 ReportLab 的頁面座標由左下方起算；此函式統一處理留白與字型，避免各圖元素
    因手動座標誤差而重疊。``style`` 只負責視覺分群，不代表科學完成狀態。
    """

    fill = GROUP_COLORS[style]
    border = GROUP_BORDERS[style]
    canvas.setFillColor(fill)
    canvas.setStrokeColor(border)
    canvas.setLineWidth(1.1)
    canvas.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    text_x = x + 8
    title_font = CODE_FONT if title_is_code else BODY_FONT
    title_size = 8.4 if title_is_code else 9.1
    draw_text_lines(
        canvas,
        title_lines,
        x=text_x,
        y_top=y + height - 13,
        font_name=title_font,
        font_size=title_size,
        leading=10.5,
    )
    if body_lines:
        separator_y = y + height - 27 - (len(title_lines) - 1) * 10.5
        canvas.setStrokeColor(border)
        canvas.setLineWidth(0.45)
        canvas.line(text_x, separator_y, x + width - 8, separator_y)
        draw_text_lines(
            canvas,
            body_lines,
            x=text_x,
            y_top=separator_y - 11,
            font_name=BODY_FONT,
            font_size=7.5,
            leading=9.0,
        )


def draw_group_panel(
    canvas: Canvas,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    style: str,
) -> None:
    """繪製四大程式群組的淡色外框與標題。

    外框只標示「資料與幾何、物理與邊界、情境與執行、輸出與聚合」四個閱讀層次，內部
    小方塊才對應實際檔案。這可讓接手者先理解責任邊界，再追查個別函式細節。
    """

    canvas.setFillColor(GROUP_COLORS[style])
    canvas.setStrokeColor(GROUP_BORDERS[style])
    canvas.setLineWidth(1.3)
    canvas.roundRect(x, y, width, height, 10, fill=1, stroke=1)
    canvas.setFillColor(GROUP_BORDERS[style])
    canvas.roundRect(x + 8, y + height - 24, 104, 17, 5, fill=1, stroke=0)
    draw_text_lines(
        canvas,
        (title,),
        x=x + 14,
        y_top=y + height - 12,
        font_name=BODY_FONT,
        font_size=8.5,
        leading=9,
        color=white,
    )


def draw_arrow(
    canvas: Canvas,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """繪製含箭頭的資料流，必要時加上短中文標籤。

    箭頭僅用來說明資料依賴或控制流程，並非 Python 的直接呼叫關係。標籤放在中點附近，
    避免接手者把「設定供給幾何」誤讀為兩個模組必然互相 import。
    """

    start_x, start_y = start
    end_x, end_y = end
    canvas.setStrokeColor(ARROW_COLOR)
    canvas.setFillColor(ARROW_COLOR)
    canvas.setLineWidth(1.0)
    canvas.line(start_x, start_y, end_x, end_y)
    vector_x = end_x - start_x
    vector_y = end_y - start_y
    length = max((vector_x**2 + vector_y**2) ** 0.5, 1.0)
    unit_x = vector_x / length
    unit_y = vector_y / length
    # 兩條短斜線形成箭頭，長度固定以避免短箭頭失真。
    head = 6.0
    side_x = -unit_y * 3.0
    side_y = unit_x * 3.0
    canvas.line(end_x, end_y, end_x - unit_x * head + side_x, end_y - unit_y * head + side_y)
    canvas.line(end_x, end_y, end_x - unit_x * head - side_x, end_y - unit_y * head - side_y)
    if label:
        midpoint_x = (start_x + end_x) * 0.5 + label_offset[0]
        midpoint_y = (start_y + end_y) * 0.5 + label_offset[1]
        canvas.setFillColor(white)
        canvas.roundRect(midpoint_x - 18, midpoint_y - 5, 36, 10, 2, fill=1, stroke=0)
        draw_text_lines(
            canvas,
            (label,),
            x=midpoint_x - 15,
            y_top=midpoint_y + 2,
            font_name=BODY_FONT,
            font_size=6.8,
            leading=7.5,
            color=ARROW_COLOR,
        )


def draw_page_title(canvas: Canvas, title: str, subtitle: str, page_number: int) -> None:
    """繪製全頁共用的標題、用途與頁碼。

    標題說明此圖對應文件 11 的哪一節；頁腳加入「概念圖，不代表正式成果已完成」，使
    圖檔被單獨轉貼到簡報時仍不會誤導讀者對專案完成狀態的判讀。
    """

    draw_text_lines(
        canvas,
        (title,),
        x=MARGIN,
        y_top=PAGE_HEIGHT - 24,
        font_name=TITLE_FONT,
        font_size=16,
        leading=18,
    )
    draw_text_lines(
        canvas,
        (subtitle,),
        x=MARGIN,
        y_top=PAGE_HEIGHT - 42,
        font_name=BODY_FONT,
        font_size=8.5,
        leading=10,
        color=HexColor("#506070"),
    )
    canvas.setStrokeColor(HexColor("#9AA8B5"))
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_HEIGHT - 51, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 51)
    draw_text_lines(
        canvas,
        ("Lagrangian Ensemble Backtracking | 概念導覽圖，非正式成果圖",),
        x=MARGIN,
        y_top=16,
        font_name=BODY_FONT,
        font_size=7.2,
        leading=8,
        color=HexColor("#506070"),
    )
    draw_text_lines(
        canvas,
        (f"{page_number} / 2",),
        x=PAGE_WIDTH - MARGIN - 25,
        y_top=16,
        font_name=CODE_FONT,
        font_size=7.2,
        leading=8,
        color=HexColor("#506070"),
    )


def draw_module_relationship_page(canvas: Canvas) -> None:
    """繪製文件第 3 節的四群組關係圖。

    第一頁由左至右安排資料、物理、執行與輸出，使箭頭主要往右移動。局部上下連線補充
    設定與幾何的共同輸入關係，目的是提供讀碼路徑，不追求表達每個 Python import。
    """

    draw_page_title(
        canvas,
        "圖 1. src 模組關係與資料流",
        "對應文件 11 第 3 節；方塊是責任分工，箭頭是資料依賴。",
        1,
    )
    top_y = 290
    bottom_y = 52
    group_width = 185
    group_height = 220
    left_x = 36
    right_x = PAGE_WIDTH - 36 - group_width

    draw_group_panel(
        canvas, x=left_x, y=top_y, width=group_width, height=group_height, title="資料與幾何", style="data"
    )
    draw_group_panel(
        canvas,
        x=right_x,
        y=top_y,
        width=group_width,
        height=group_height,
        title="物理與邊界",
        style="physics",
    )
    draw_group_panel(
        canvas,
        x=left_x,
        y=bottom_y,
        width=group_width,
        height=group_height,
        title="情境與執行",
        style="execution",
    )
    draw_group_panel(
        canvas,
        x=right_x,
        y=bottom_y,
        width=group_width,
        height=group_height,
        title="輸出與聚合",
        style="output",
    )

    box_width = 165
    box_height = 36
    for index, (title, body) in enumerate(
        (
            (("config.py",), ("科學計數與正式發布閘門",)),
            (("preflight.py + time_axis.py",), ("上游月份、UTC 與缺時盤點",)),
            (("geometry.py + mesh.py",), ("公尺座標、範圍、原始網格定位",)),
            (("receptors.py + arrival_times.py",), ("20 個受體與 50 個時刻的選取核心",)),
        )
    ):
        draw_box(
            canvas,
            x=left_x + 10,
            y=top_y + 22 + (3 - index) * 43,
            width=box_width,
            height=box_height,
            title_lines=title,
            body_lines=body,
            style="data",
            title_is_code=True,
        )

    for index, (title, body) in enumerate(
        (
            (("models.py",), ("共用粒子、速度、事件與品質型別",)),
            (("forcing.py + accelerated.py",), ("OCM、NWW3 時空取樣",)),
            (("stokes.py + diffusion.py + integrators.py",), ("總速度、擴散、逆時間積分",)),
            (("boundaries.py + engine.py",), ("邊界事件與單粒子停止控制",)),
        )
    ):
        draw_box(
            canvas,
            x=right_x + 10,
            y=top_y + 22 + (3 - index) * 43,
            width=box_width,
            height=box_height,
            title_lines=title,
            body_lines=body,
            style="physics",
            title_is_code=True,
        )

    for index, (title, body) in enumerate(
        (
            (("scenarios.py + runner.py",), ("10 x 20 x 50 與 M 個成員批次",)),
            (("checkpoint.py",), ("中途狀態的安全續跑",)),
            (("cli.py",), ("設定、盤點、合成試算與驗證入口",)),
        )
    ):
        draw_box(
            canvas,
            x=left_x + 10,
            y=bottom_y + 28 + (2 - index) * 52,
            width=box_width,
            height=44,
            title_lines=title,
            body_lines=body,
            style="execution",
            title_is_code=True,
        )

    for index, (title, body) in enumerate(
        (
            (("outputs.py",), ("軌跡、事件、檢查資料的原子發布",)),
            (("aggregation.py",), ("入口密度、足跡、路徑、停留時間",)),
            (("圖表與成果報告",), ("需由已發布 aggregate 產製",)),
        )
    ):
        draw_box(
            canvas,
            x=right_x + 10,
            y=bottom_y + 28 + (2 - index) * 52,
            width=box_width,
            height=44,
            title_lines=title,
            body_lines=body,
            style="output",
            title_is_code=index < 2,
        )

    # 主要跨群組資料流：設定與幾何供應物理取樣，物理結果與情境共同形成軌跡，最後聚合。
    draw_arrow(
        canvas, start=(left_x + group_width, top_y + 130), end=(right_x, top_y + 130), label="資料契約"
    )
    draw_arrow(
        canvas, start=(left_x + 92, top_y), end=(left_x + 92, bottom_y + group_height), label="受體與時刻"
    )
    draw_arrow(
        canvas, start=(right_x + 92, top_y), end=(right_x + 92, bottom_y + group_height), label="粒子結果"
    )
    draw_arrow(
        canvas,
        start=(left_x + group_width, bottom_y + 130),
        end=(right_x, bottom_y + 130),
        label="軌跡與事件",
    )
    draw_arrow(canvas, start=(right_x + 92, bottom_y + 92), end=(right_x + 92, bottom_y + 72), label="聚合")
    canvas.showPage()


def draw_particle_flow_page(canvas: Canvas) -> None:
    """繪製文件第 4 節的一條粒子處理流程圖。

    第二頁依時間順序由左至右呈現。OCM 與 NWW3 兩個資料來源在中段合成總速度，再經
    四階積分、擴散與邊界判定。停止與未停止分支分開畫出，避免把「離開其他站 local
    domain」誤讀為所有情況都會中止計算。
    """

    draw_page_title(
        canvas,
        "圖 2. 單一粒子逆向溯源的處理流程",
        "對應文件 11 第 4 節；每條軌跡維持原始 study_site_id，不因跨站事件改變歸屬。",
        2,
    )
    y_main = 335
    y_source = 210
    x_positions = [35, 160, 285, 410, 535, 660]
    main_width = 112
    main_height = 68

    draw_box(
        canvas,
        x=x_positions[0],
        y=y_main,
        width=main_width,
        height=main_height,
        title_lines=("Scenario",),
        body_lines=("站點、行為、受體、", "到達 UTC 的基礎情境"),
        style="execution",
        title_is_code=True,
    )
    draw_box(
        canvas,
        x=x_positions[1],
        y=y_main,
        width=main_width,
        height=main_height,
        title_lines=("RunUnit",),
        body_lines=("加入 member、粒子 ID", "與可重現亂數種子"),
        style="execution",
        title_is_code=True,
    )
    draw_box(
        canvas,
        x=x_positions[2],
        y=y_main,
        width=main_width,
        height=main_height,
        title_lines=("run_particle",),
        body_lines=("初始位置、邊界、", "擴散與停止設定"),
        style="physics",
        title_is_code=True,
    )
    draw_box(
        canvas,
        x=x_positions[3],
        y=y_main,
        width=main_width,
        height=main_height,
        title_lines=("總速度",),
        body_lines=("海流 + Stokes + 浮沉", "物理時間往後的速度"),
        style="physics",
    )
    draw_box(
        canvas,
        x=x_positions[4],
        y=y_main,
        width=main_width,
        height=main_height,
        title_lines=("RK4 + 擴散",),
        body_lines=("負時間步回溯；", "再加入一次隨機位移"),
        style="physics",
    )
    draw_box(
        canvas,
        x=x_positions[5],
        y=y_main,
        width=main_width,
        height=main_height,
        title_lines=("邊界與停止",),
        body_lines=("海岸、local、outer、", "海面、海床與資料條件"),
        style="physics",
    )
    for start_x, end_x, label in zip(
        x_positions[:-1], x_positions[1:], ("情境 x M", "建立狀態", "取樣", "積分", "判定"), strict=True
    ):
        draw_arrow(
            canvas,
            start=(start_x + main_width, y_main + main_height * 0.5),
            end=(end_x, y_main + main_height * 0.5),
            label=label,
            label_offset=(0, 11),
        )

    # OCM 與 NWW3 來源分開畫出，強調總速度不是單一資料集直接給定的欄位。
    draw_box(
        canvas,
        x=370,
        y=y_source,
        width=122,
        height=58,
        title_lines=("OCMNativeMonth.sample",),
        body_lines=("三維海流、海面、海床、Kz",),
        style="data",
        title_is_code=True,
    )
    draw_box(
        canvas,
        x=505,
        y=y_source,
        width=122,
        height=58,
        title_lines=("NWWAnalysisMonth.sample",),
        body_lines=("波高、頻率、波向與有效遮罩",),
        style="data",
        title_is_code=True,
    )
    draw_box(
        canvas,
        x=505,
        y=118,
        width=122,
        height=52,
        title_lines=("finite_depth_stokes",),
        body_lines=("有限水深波浪表面漂移",),
        style="physics",
        title_is_code=True,
    )
    draw_arrow(canvas, start=(431, y_source + 58), end=(466, y_main), label="海流")
    draw_arrow(canvas, start=(566, y_source + 58), end=(566, y_main), label="波浪")
    draw_arrow(canvas, start=(566, y_source), end=(566, 170), label="計算")
    draw_arrow(canvas, start=(566, 170), end=(500, y_main), label="Stokes", label_offset=(7, 0))

    # 停止分支使用不同顏色，讓閱覽者快速看到何時會輸出一條完整軌跡。
    draw_box(
        canvas,
        x=640,
        y=185,
        width=137,
        height=57,
        title_lines=("仍可回溯",),
        body_lines=("返回下一個時間步；", "不改變 study_site_id"),
        style="neutral",
    )
    draw_box(
        canvas,
        x=640,
        y=92,
        width=137,
        height=70,
        title_lines=("ParticleResult",),
        body_lines=("最終狀態、固定間隔軌跡、", "邊界事件與停止原因"),
        style="output",
        title_is_code=True,
    )
    draw_arrow(canvas, start=(716, y_main), end=(716, 242), label="未停止")
    draw_arrow(canvas, start=(716, y_main), end=(716, 162), label="停止")
    draw_arrow(canvas, start=(640, 213), end=(590, 287), label="下一步", label_offset=(0, -7))

    draw_box(
        canvas,
        x=470,
        y=48,
        width=135,
        height=46,
        title_lines=("write_trajectory_shard",),
        body_lines=("原子寫出軌跡、事件與檢查資料",),
        style="output",
        title_is_code=True,
    )
    draw_box(
        canvas,
        x=304,
        y=48,
        width=145,
        height=46,
        title_lines=("aggregation.py",),
        body_lines=("入口密度、足跡、路徑、停留時間",),
        style="output",
        title_is_code=True,
    )
    draw_arrow(canvas, start=(690, 92), end=(605, 71), label="發布")
    draw_arrow(canvas, start=(470, 71), end=(449, 71), label="聚合")
    canvas.showPage()


def parse_arguments() -> argparse.Namespace:
    """讀取輸出 PDF 路徑，預設位置符合專案產物目錄慣例。"""

    parser = argparse.ArgumentParser(description="產生 src 程式碼導覽的兩頁流程圖 PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/source_code_flow_diagrams.pdf"),
        help="輸出 PDF；預設為 output/pdf/source_code_flow_diagrams.pdf",
    )
    return parser.parse_args()


def main() -> None:
    """建立兩頁 PDF，並在寫入前確保輸出資料夾存在。

    只建立指定的輸出檔，不覆寫任何上游資料或正式 run 結果。ReportLab 在 ``save`` 前會
    暫存頁面內容；若程式中途失敗，輸出資料夾可能存在但不會產生可誤用的完整 PDF。
    """

    arguments = parse_arguments()
    output_path = arguments.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()
    canvas = Canvas(str(output_path), pagesize=landscape(A4), pageCompression=1)
    canvas.setTitle("Lagrangian 系集逆向溯源程式流程圖")
    canvas.setAuthor("Lagrangian Ensemble Backtracking")
    canvas.setSubject("文件 11 第 3、4 節：程式模組關係與單一粒子流程")
    draw_module_relationship_page(canvas)
    draw_particle_flow_page(canvas)
    canvas.save()


if __name__ == "__main__":
    main()
