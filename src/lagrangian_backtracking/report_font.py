"""正式報告所需的繁體中文字型解析與可重現選擇紀錄。

本模組只接受 Matplotlib 字型管理器已能在執行環境中找到的系統字型，並依固定的
候選家族順序逐一檢查。每一個候選字型都必須由 FreeType 字碼表（charmap）涵蓋報告
會使用的繁體中文與 ASCII 數字；通過後才讀取一般檔案 bytes 計算 SHA-256。回傳的
``CJKFontSelection`` 只保存家族名稱、檔名 basename、檔案摘要與必要字元數量，不
保存絕對路徑，因此可以安全地寫入報告 provenance。這裡不下載字型，也不把英文或
其他預設字型當作缺件時的替代品；若沒有合格候選，所有內部例外都會收斂為固定的
``ValueError``，避免把本機路徑或底層錯誤訊息帶到公開邊界。
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from matplotlib import font_manager
from matplotlib.ft2font import FT2Font

__all__ = [
    "CJK_FONT_CANDIDATES",
    "REQUIRED_REPORT_GLYPHS",
    "CJKFontSelection",
    "resolve_cjk_font",
]


# 候選順序是報告重現性契約的一部分；前面的字型只要完整涵蓋必要字元，就不能被
# 後面的系統字型取代。名稱必須保留原始 family label，不能由作業系統自動排序。
CJK_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Source Han Sans TC",
    "PingFang TC",
    "Heiti TC",
    "Arial Unicode MS",
)


# 這些固定片語對應正式 F01-F12／T01-T06 報告的規劃標題、圖例、表格欄名與限制
# 說明。以可讀的片語分段保存 inventory，讓新增正式面板時能逐項審查，而不是只
# 維護一串無法追溯用途的單字元清單；重複字元最後仍只計一次。
_PLANNED_SCIENTIFIC_ZH_TW_TEXT: Final[tuple[str, ...]] = (
    "報告",
    "條件式來源足跡",
    "相對來源權重",
    "實驗設定",
    "資料完整性",
    "時間涵蓋",
    "單位",
    "公尺",
    "秒",
    "海流",
    "波浪",
    "受體",
    "到達",
    "五站",
    "月份",
    "季節",
    "潮況",
    "季節潮況材質停止失敗",
    "物性",
    "粒徑",
    "密度",
    "沉降速度",
    "逆向追蹤",
    "軌跡",
    "深度",
    "年齡",
    "海面",
    "海面海床深度",
    "海床接觸",
    "邊界",
    "停止原因",
    "數值驗證",
    "解析解",
    "收斂",
    "誤差",
    "比例",
    "分母",
    "樣本不足",
    "不可估計",
    "核密度",
    "高密度區",
    "頻寬敏感度",
    "連通性",
    "旅行時間",
    "停留時間",
    "首次通過",
    "觀測",
    "重建",
    "缺口",
    "不確定性",
    "基線比較",
    "科學驗證",
    "正式",
    "合成工程證據",
    "圖表附錄",
    # source-pathway-v1 六面板圖的可見標題與品質標註；先在字型 gate 登錄，
    # 才能讓正式 renderer 在真正寫檔前拒絕缺少這些新增字元的字型。
    "向下沉降粒子移入關注海域",
    "訪格比例",
    "中位首次通過年齡",
    "局部邊界首次離開端點",
    "每有效成員停留時數",
    "原始計數",
    "品質檢查",
    "斜線",
    "無樣本",
    "低樣本",
    "底床邊界接觸診斷",
    # source-pathway-v1 renderer 的 colorbar、軸標籤與狀態註記；這些是實際會
    # 出現在 PNG／SVG／PDF 圖面的字串，不能只依賴一般報告文字的間接涵蓋。
    "格網內相對權重",
    "小時／成員",
    "空白",
    "KDE 狀態",
    "格網內累積權重輪廓",
    "原始樣本",
    "局部類別內相對比例",
    "局部邊界分段／弧長分箱",
    "停止原始計數",
    "總成員分母比例",
    "紅線",
    "等值線",
    "完整保留",
    "HDR 50／75／90%",
    "1×N 不繪輪廓",
    "遮罩見 sidecar",
    "失敗與截尾皆納入分母",
    "潛在移入入口",
    "逆向首次離開",
    "潛在移入邊界區段",
)

# 正式圖表會使用面板／表格識別碼、英文縮寫、ASCII 數值與常用單位／數學符號；
# 明確登錄大小寫、數字及符號可避免 renderer 在特定平台才首次觸發缺 glyph。
_PLANNED_ASCII_AND_SYMBOL_GLYPHS: Final[str] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%()[]-–—_/,.:;=+×≤≥<>°²μ"
    "\u0020\u2212（）：；，｜／"
)

# 以 frozenset 封存 Unicode code point inventory，防止 caller 改寫必要字元；
# ``required_glyph_count`` 因而代表所有固定片語、ASCII 與符號去重後的檢查數量。
REQUIRED_REPORT_GLYPHS: Final[frozenset[str]] = frozenset(
    "".join(_PLANNED_SCIENTIFIC_ZH_TW_TEXT) + _PLANNED_ASCII_AND_SYMBOL_GLYPHS
)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_NO_VALID_FONT_MESSAGE: Final[str] = "找不到涵蓋報告必要字元的合格繁體中文字型"


@dataclass(frozen=True, slots=True)
class CJKFontSelection:
    """已通過必要字元檢查的繁體中文字型 immutable provenance record。

    ``family`` 是固定候選清單中的 family label；``filename`` 只允許保存 basename，
    不包含目錄或絕對路徑；``file_sha256`` 是從該字型檔案的原始 bytes 計算出的 64
    碼小寫 SHA-256；``required_glyph_count`` 則是本模組必要 Unicode code point 的
    數量。這個 record 供報告 metadata／caption 保存，不足以單獨描述作業系統的完整
    字型搜尋環境，且不代表任何海洋科學結果的有效性。
    """

    family: str
    filename: str
    file_sha256: str
    required_glyph_count: int

    def __post_init__(self) -> None:
        """在建立 record 時阻擋錯誤型別、路徑欄位與非 canonical 摘要。

        resolver 本身只會傳入已驗證的值，但公開 dataclass 仍可能被其他呼叫端直接
        建構；這裡再次守住資料契約，避免日後把絕對路徑、大小寫混雜的摘要或 bool
        計數寫進報告。只驗證 record 形狀，不重新讀檔或重新解析字型。
        """

        if type(self.family) is not str or not self.family:
            raise TypeError("family 必須是非空原生 str")
        if self.family not in CJK_FONT_CANDIDATES:
            raise ValueError("family 必須精確屬於 CJK_FONT_CANDIDATES")
        if type(self.filename) is not str or not self.filename:
            raise TypeError("filename 必須是非空原生 str basename")
        if (
            self.filename in {".", ".."}
            or "/" in self.filename
            or "\\" in self.filename
            or Path(self.filename).is_absolute()
        ):
            raise ValueError("filename 必須是不含目錄的 basename")
        if type(self.file_sha256) is not str or _SHA256_PATTERN.fullmatch(self.file_sha256) is None:
            raise ValueError("file_sha256 必須是 64 碼小寫 SHA-256")
        if type(self.required_glyph_count) is not int:
            raise TypeError("required_glyph_count 必須是原生 int，且不可是 bool")
        if self.required_glyph_count != len(REQUIRED_REPORT_GLYPHS):
            raise ValueError("required_glyph_count 必須精確等於必要字元數量")


def _is_ordinary_font_file(filename: Path) -> bool:
    """以 ``lstat`` 驗證候選實際目標是非符號連結的普通檔案。

    ``os.lstat`` 不追隨符號連結，因此即使 symlink 指向內容完整的合法字型，也會
    以 link 本身的檔案型別被拒絕。directory、FIFO、socket、missing 或其他無法
    取得 metadata 的目標同樣不合格；這個檢查必須先於 FreeType 開檔與 bytes 讀取，
    避免 resolver 對非預期 I/O 目標產生副作用。底層例外不向公開邊界傳遞。
    """

    try:
        file_stat = os.lstat(filename)
    except OSError:
        return False
    return stat.S_ISREG(file_stat.st_mode) and not stat.S_ISLNK(file_stat.st_mode)


def _font_covers_required_glyphs(filename: Path) -> bool:
    """以 FreeType charmap 判斷字型是否涵蓋每一個必要 Unicode code point。

    ``filename`` 僅在內部傳給 FreeType，因為解析字型必須知道實際檔案位置；這個
    私有函式不會把路徑放進例外或回傳值。charmap 的 key 是 Unicode code point，
    因此必須對每個必要 glyph 使用 ``ord`` 精確比對，而不能只檢查字型名稱或以
    Matplotlib 的 fallback 字型結果代替缺少的字元。
    """

    charmap = FT2Font(str(filename)).get_charmap()
    return all(ord(glyph) in charmap for glyph in REQUIRED_REPORT_GLYPHS)


def _sha256_file_bytes(filename: Path) -> str:
    """讀取普通檔案 bytes 並計算 canonical 的小寫 SHA-256 摘要。

    報告 provenance 需要的是實際被選中字型檔的內容摘要，而不是檔名、mtime 或
    FreeType 解析後的部分 metadata；因此這裡明確使用 ``Path.read_bytes`` 的完整
    bytes 作為雜湊輸入。
    """

    return hashlib.sha256(filename.read_bytes()).hexdigest()


def resolve_cjk_font() -> CJKFontSelection:
    """依固定候選順序解析可支援報告字元的繁體中文字型。

    每輪使用 Matplotlib ``font_manager`` 的 family matching，並明確關閉預設字型
    fallback；找到檔案後再以 FreeType charmap 驗證全部必要 glyph，最後以原始檔案
    bytes 產生摘要。候選順序、字元集合與錯誤訊息都是 deterministic contract。若
    所有候選都不存在、無法解析、缺少任一字元或無法讀取，函式只拋出不含路徑的固定
    ``ValueError``；不會下載字型，也不會回傳英文 fallback。
    """

    for family in CJK_FONT_CANDIDATES:
        try:
            # ``fallback_to_default=False`` 是必要的安全閘門；否則找不到 CJK family
            # 時，Matplotlib 可能交回預設西文字型，讓報告以 tofu 或英文掩蓋缺件。
            filename = Path(
                font_manager.findfont(
                    font_manager.FontProperties(family=[family]),
                    fallback_to_default=False,
                )
            )
            if not _is_ordinary_font_file(filename):
                continue
            if not _font_covers_required_glyphs(filename):
                continue
            file_sha256 = _sha256_file_bytes(filename)
            return CJKFontSelection(
                family=family,
                filename=filename.name,
                file_sha256=file_sha256,
                required_glyph_count=len(REQUIRED_REPORT_GLYPHS),
            )
        # font_manager、FreeType 與檔案 I/O 的底層例外可能包含絕對路徑；候選只需
        # 視為不合格並繼續，最終由公開邊界統一拋出固定訊息，避免洩漏環境資訊。
        except Exception:
            continue

    raise ValueError(_NO_VALID_FONT_MESSAGE)
