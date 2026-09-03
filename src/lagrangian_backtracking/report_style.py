"""正式報告 renderer 的可重現 Matplotlib 樣式與環境閘門。

本模組只保存 renderer 的固定設定，不讀取 OCM schema 3／NWW3 schema 1、不讀取
軌跡、不繪製圖面，也不會下載字型。模組頂層刻意不載入 Matplotlib、pyplot 或
``report_font``；一般 CLI 只要 import 套件就不會因此觸發 Matplotlib cache。真正
建立報告樣式時，呼叫端必須先提供既有、可寫且非符號連結的專用
``MPLCONFIGDIR``，之後函式才以非互動 Agg backend lazy import Matplotlib 與字型
解析器。

這裡的 reproducibility 只描述圖面工程設定，例如 dpi、色盤、字型 provenance 與
SVG hash salt；它不構成海洋科學結果。即使 synthetic 測試能建立樣式與圖檔，也不
代表真實 OCM／NWW3 資料已載入、已通過科學驗證或已產生條件式來源足跡。
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from .report_spec import ReportSpec

if TYPE_CHECKING:
    from .report_font import CJKFontSelection

__all__ = [
    "ReportRenderStyle",
    "report_render_style_context",
    "resolve_report_render_style",
    "validate_mplconfigdir",
]


# 這些值與 ReportSpec 的固定 renderer policy 必須完全一致。style constructor 仍會
# 再驗證一次，避免 caller 以 object.__setattr__ 或外部反序列化資料繞過 ReportSpec
# 原本的 dataclass 驗證後，把不同樣式混入同一份報告 release。
_RENDERER_STYLE_VERSION: Final[str] = "academic_zh_tw_v1"
_LANGUAGE: Final[str] = "zh-TW"
_RASTER_DPI: Final[int] = 300

# 公開錯誤必須固定，不能把 MPLCONFIGDIR、HOME 或底層 lstat/access 例外中的絕對路徑
# 洩漏到 log、CLI 或報告 metadata。所有環境閘門失敗都收斂到這一個訊息。
_MPLCONFIGDIR_ERROR: Final[str] = (
    "MPLCONFIGDIR 必須明示為既有、可寫、非符號連結的專用目錄"
)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

# SVG salt 的 payload 契約固定使用 ASCII pipe 作為唯一分隔符，不使用 Python hash、
# 時間、主機路徑或字型路徑。實際 bytes 精確為：
# ``b"academic_zh_tw_v1|" + report_spec.canonical_sha256.encode("utf-8")``。
_SVG_HASHSALT_SEPARATOR: Final[str] = "|"

# 所有 renderer 都從這個 immutable 色盤字串建立同一個 Matplotlib cycler。使用
# Matplotlib 可解析的文字而不是 ``cycler.Cycler`` 物件，使 style record 的每個值都
# 能以 JSON-like scalar／tuple 表示，並可安全被 snapshot、檢查與序列化。
_COLOR_CYCLE: Final[str] = (
    "cycler('color', "
    "['#1B4965', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#6A4C93'])"
)

# 這是本模組唯一允許的 rc key 集合；backend 不在其中，因為 backend 必須由
# ``matplotlib.use('Agg', force=True)`` 控制，不能被 rc_context 或 caller 的 dict
# 偷渡覆蓋。動態欄位只有 font.family 與 svg.hashsalt，其餘值固定在此表。
_RC_PARAM_KEYS: Final[tuple[str, ...]] = (
    "font.family",
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "figure.titlesize",
    "axes.linewidth",
    "lines.linewidth",
    "lines.markersize",
    "grid.linewidth",
    "grid.alpha",
    "grid.color",
    "grid.linestyle",
    "axes.grid",
    "axes.prop_cycle",
    "axes.unicode_minus",
    "axes.formatter.use_locale",
    "text.usetex",
    "svg.hashsalt",
    "svg.fonttype",
    "pdf.fonttype",
    "ps.fonttype",
    "figure.dpi",
    "savefig.dpi",
    "savefig.bbox",
    "savefig.pad_inches",
    "figure.facecolor",
    "axes.facecolor",
    "savefig.facecolor",
    "image.interpolation",
    "image.cmap",
    "image.origin",
)

_FIXED_RC_VALUES: Final[dict[str, object]] = {
    "font.size": 10.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "figure.titlesize": 13.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "grid.color": "#7A7A7A",
    "grid.linestyle": "-",
    "axes.grid": False,
    "axes.prop_cycle": _COLOR_CYCLE,
    "axes.unicode_minus": True,
    "axes.formatter.use_locale": False,
    "text.usetex": False,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 100,
    "savefig.dpi": _RASTER_DPI,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.1,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "image.interpolation": "nearest",
    "image.cmap": "viridis",
    "image.origin": "lower",
}


def _raise_mplconfigdir_error() -> None:
    """以固定訊息拒絕所有不合格的 Matplotlib cache 目錄。"""

    raise ValueError(_MPLCONFIGDIR_ERROR)


def _is_home_matplotlib_directory(config_path: Path) -> bool:
    """判斷候選目錄是否等同於 HOME 下的預設 Matplotlib 目錄。

    比較使用 canonical path，是為了連 ``./``、父目錄 alias 或 symlink parent
    形成的同一個 ``$HOME/.matplotlib`` 也拒絕；候選自身的最後一節仍會由 caller
    先以 lstat 驗證為非 symlink。此 helper 不會建立目錄或檔案。
    """

    try:
        home = os.path.expanduser("~")
        if type(home) is not str or not home or home == "~":
            return False
        config_canonical = os.path.realpath(os.fspath(config_path))
        home_default = os.path.realpath(os.path.join(home, ".matplotlib"))
        return config_canonical == home_default
    except Exception:
        # HOME 解析失敗時不能因為檢查本身造成 cache gate 放行；由外層統一拒絕。
        raise ValueError(_MPLCONFIGDIR_ERROR) from None


def validate_mplconfigdir() -> Path:
    """驗證正式 renderer 的 Matplotlib cache 目錄並回傳其原始絕對路徑。

    ``MPLCONFIGDIR`` 必須在 process environment 中明示為原生、非空 ``str``，且
    必須是已存在的絕對路徑；最後節點以 ``lstat`` 確認為 ordinary directory、不是
    symbolic link，並以 ``os.access(W_OK | X_OK)`` 確認目前執行者可寫入及搜尋。另
    外禁止 canonical path 等於 ``$HOME/.matplotlib``，因為那是 Matplotlib 的一般
    fallback 位置而不是本專案 task-specific cache。

    這個 gate 絕不 mkdir、建立 probe file、建立暫存 fallback 或下載字型。任何失敗
    都只拋出完全固定且不含 path 的 ``ValueError``：
    ``MPLCONFIGDIR 必須明示為既有、可寫、非符號連結的專用目錄``。
    """

    try:
        config_value = os.environ.get("MPLCONFIGDIR")
        if type(config_value) is not str or not config_value:
            _raise_mplconfigdir_error()

        config_path = Path(config_value)
        if not config_path.is_absolute():
            _raise_mplconfigdir_error()

        metadata = os.lstat(config_path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _raise_mplconfigdir_error()
        if _is_home_matplotlib_directory(config_path):
            _raise_mplconfigdir_error()
        if not os.access(config_path, os.W_OK | os.X_OK):
            _raise_mplconfigdir_error()
        return config_path
    except ValueError:
        # 將 helper 自己產生的固定錯誤重新建立，避免未來有人修改 helper 訊息後
        # 不小心讓公開 gate 出現多套文字。
        raise ValueError(_MPLCONFIGDIR_ERROR) from None
    except Exception:
        # lstat、Path、environment、access 的底層錯誤都可能攜帶絕對路徑；不得向
        # renderer 或 operator 轉送，所有失敗維持同一個 fail-closed 邊界。
        raise ValueError(_MPLCONFIGDIR_ERROR) from None


def _validate_report_spec_for_renderer(report_spec: ReportSpec) -> None:
    """在載入任何繪圖依賴前重驗 renderer 所需的 ReportSpec 固定欄位。"""

    if type(report_spec) is not ReportSpec:
        raise TypeError("report_spec 必須是 exact ReportSpec")
    if (
        type(report_spec.renderer_style_version) is not str
        or report_spec.renderer_style_version != _RENDERER_STYLE_VERSION
    ):
        raise ValueError("renderer_style_version 不受支援")
    if (
        type(report_spec.language) is not str
        or report_spec.language != _LANGUAGE
    ):
        raise ValueError("language 必須精確為 zh-TW")
    if type(report_spec.raster_dpi) is not int or report_spec.raster_dpi != _RASTER_DPI:
        raise ValueError("raster_dpi 必須精確為原生 int 300")
    if (
        type(report_spec.canonical_sha256) is not str
        or _SHA256_PATTERN.fullmatch(report_spec.canonical_sha256) is None
    ):
        raise ValueError("canonical_sha256 必須是 64 碼小寫 SHA-256")


def _validate_font_selection(font_selection: object) -> None:
    """驗證字型 provenance 的最小資料契約，不載入字型檔或保存其路徑。

    resolver 的正式回傳型別是 ``CJKFontSelection``；這裡採用欄位級 fail-closed
    驗證，讓測試可以用等價 immutable fake selection 注入而不必依賴主機字型。實際
    report font resolver 仍在 gate 後執行完整的 FreeType glyph coverage 檢查。style
    只會保存四個欄位：family、filename basename、file SHA-256 與 glyph count。
    """

    try:
        family = font_selection.family  # type: ignore[attr-defined]
        filename = font_selection.filename  # type: ignore[attr-defined]
        file_sha256 = font_selection.file_sha256  # type: ignore[attr-defined]
        required_glyph_count = font_selection.required_glyph_count  # type: ignore[attr-defined]
    except Exception:
        raise TypeError("font_selection 必須符合 CJKFontSelection 資料契約") from None

    if type(family) is not str or not family:
        raise TypeError("font_selection.family 必須是非空原生 str")
    if type(filename) is not str or not filename:
        raise TypeError("font_selection.filename 必須是非空 basename")
    if (
        filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
    ):
        raise ValueError("font_selection.filename 必須是不含目錄的 basename")
    if type(file_sha256) is not str or _SHA256_PATTERN.fullmatch(file_sha256) is None:
        raise ValueError("font_selection.file_sha256 必須是 64 碼小寫 SHA-256")
    if type(required_glyph_count) is not int or required_glyph_count <= 0:
        raise ValueError("font_selection.required_glyph_count 必須是正的原生 int")


def _expected_rc_params(font_selection: object, svg_hashsalt: str) -> dict[str, object]:
    """由 style 的兩個動態欄位建立唯一 rc snapshot。"""

    params = dict(_FIXED_RC_VALUES)
    params["font.family"] = (font_selection.family,)  # type: ignore[attr-defined]
    params["svg.hashsalt"] = svg_hashsalt
    # 這個 assert 同時保護開發者新增 key 時必須更新固定契約；不把 assert 當成
    # caller validation，真正的 constructor 還會以明確例外再次檢查。
    if set(params) != set(_RC_PARAM_KEYS):
        raise RuntimeError("renderer rc 參數固定集合未同步")
    return params


def _is_json_like(value: object) -> bool:
    """判斷 rc value 是否只由 JSON-like scalar 與 tuple 組成。"""

    if value is None or type(value) in {str, int, float, bool}:
        return True
    if type(value) is tuple:
        return all(_is_json_like(item) for item in value)
    return False


def _same_json_like_value(left: object, right: object) -> bool:
    """以不接受 NumPy／自訂 equality 的方式比對固定 rc value。"""

    if type(left) is not type(right):
        return False
    if type(left) is tuple and type(right) is tuple:
        return len(left) == len(right) and all(
            _same_json_like_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _validate_rc_params(
    supplied: object,
    *,
    font_selection: object,
    svg_hashsalt: str,
) -> MappingProxyType[str, object]:
    """驗證 caller rc mapping 並建立 immutable defensive snapshot。"""

    if not isinstance(supplied, Mapping):
        raise TypeError("rc_params 必須是 Mapping")
    try:
        copied = dict(supplied)
    except Exception:
        raise TypeError("rc_params 無法建立 defensive snapshot") from None

    expected = _expected_rc_params(font_selection, svg_hashsalt)
    if set(copied) != set(_RC_PARAM_KEYS):
        raise ValueError("rc_params keys 不符合固定 renderer contract")
    for key in _RC_PARAM_KEYS:
        value = copied[key]
        if not _is_json_like(value) or not _same_json_like_value(value, expected[key]):
            raise ValueError(f"rc_params[{key}] 不符合固定 renderer contract")
    return MappingProxyType(dict(copied))


def _svg_hashsalt(report_spec: ReportSpec) -> str:
    """以固定 compact UTF-8 payload 產生 deterministic SVG hash salt。

    exact payload 是 ``academic_zh_tw_v1|<canonical_sha256>``，其中 ``|`` 是單一
    ASCII U+007C 分隔符，``<canonical_sha256>`` 是已驗證的 64 碼小寫十六進位文字。
    因此實際 hash input 為上述 payload 的 compact UTF-8 bytes；沒有 newline、時間、
    Python process hash、MPLCONFIGDIR、字型絕對路徑或其他執行環境值。
    """

    payload = (
        _RENDERER_STYLE_VERSION
        + _SVG_HASHSALT_SEPARATOR
        + report_spec.canonical_sha256
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReportRenderStyle:
    """正式 report renderer 使用的 immutable、可重現樣式 record。

    ``font_selection`` 只允許保存繁體中文字型的 family、basename、原始檔案
    SHA-256 與必要 glyph 數量；record 不保存 font path 或 ``MPLCONFIGDIR``。原始
    ``rc_params`` 會先 defensive copy，再以 ``MappingProxyType`` 封存，且 constructor
    會拒絕缺 key、額外 key、錯誤值及非 JSON-like value。這個 record 只描述圖面設定，
    不包含 OCM／NWW3 資料，也不能作為科學結果或來源機率的證明。
    """

    renderer_style_version: str
    language: str
    raster_dpi: int
    font_selection: CJKFontSelection
    svg_hashsalt: str
    rc_params: Mapping[str, object]

    def __post_init__(self) -> None:
        """交叉驗證固定 renderer 欄位、字型 provenance、hash 与 rc snapshot。"""

        if (
            type(self.renderer_style_version) is not str
            or self.renderer_style_version != _RENDERER_STYLE_VERSION
        ):
            raise ValueError("renderer_style_version 不受支援")
        if type(self.language) is not str or self.language != _LANGUAGE:
            raise ValueError("language 必須精確為 zh-TW")
        if type(self.raster_dpi) is not int or self.raster_dpi != _RASTER_DPI:
            raise ValueError("raster_dpi 必須精確為原生 int 300")
        _validate_font_selection(self.font_selection)
        if type(self.svg_hashsalt) is not str or _SHA256_PATTERN.fullmatch(self.svg_hashsalt) is None:
            raise ValueError("svg_hashsalt 必須是 64 碼小寫 SHA-256")
        snapshot = _validate_rc_params(
            self.rc_params,
            font_selection=self.font_selection,
            svg_hashsalt=self.svg_hashsalt,
        )
        object.__setattr__(self, "rc_params", snapshot)


def resolve_report_render_style(report_spec: ReportSpec) -> ReportRenderStyle:
    """在環境 gate 後解析字型並建立正式 renderer style。

    執行順序是固定契約：先 exact type／固定欄位驗證 ``ReportSpec``，再驗證
    ``MPLCONFIGDIR``；只有兩者成功後才在函式內 lazy import Matplotlib，呼叫
    ``matplotlib.use('Agg', force=True)``，最後 lazy import 並呼叫
    ``report_font.resolve_cjk_font()``。這個順序避免一般 import 或不合格環境先寫入
    HOME cache，也讓缺少完整繁中字型時 fail closed。函式不讀取 OCM／NWW3、不繪圖，
    local synthetic 成功只代表 renderer 工程契約成立。
    """

    # 這四行必須位於任何依賴 import 之前；ReportSpec 本身在 module import 時已是
    # 無 Matplotlib 的純資料契約，故此處可安全先封閉輸入。
    _validate_report_spec_for_renderer(report_spec)
    validate_mplconfigdir()

    # 兩個 import 都刻意留在 gate 之後。尤其 report_font 會載入 Matplotlib
    # font_manager／FreeType，不能在不明示 MPLCONFIGDIR 時讓它碰到 HOME cache。
    import matplotlib

    matplotlib.use("Agg", force=True)
    from . import report_font

    font_selection = report_font.resolve_cjk_font()
    salt = _svg_hashsalt(report_spec)
    rc_params = _expected_rc_params(font_selection, salt)
    return ReportRenderStyle(
        renderer_style_version=_RENDERER_STYLE_VERSION,
        language=_LANGUAGE,
        raster_dpi=_RASTER_DPI,
        font_selection=font_selection,
        svg_hashsalt=salt,
        rc_params=rc_params,
    )


@contextmanager
def report_render_style_context(report_spec: ReportSpec) -> Iterator[ReportRenderStyle]:
    """以 temporary ``matplotlib.rc_context`` 套用正式 style 並在離開時復原 rc。

    style resolution 先執行完整的 ReportSpec／MPLCONFIGDIR／字型 gate；進入 context
    後只把 immutable rc snapshot 的 plain dict 複製品交給 Matplotlib。backend 由
    resolver 固定為 Agg，但不放入 rc mapping；``rc_context`` 離開後所有 rcParams
    回到進入前的值，不會把字型、色盤或 dpi 永久污染同一個 process 的其他工作。
    """

    style = resolve_report_render_style(report_spec)
    # resolve 已在 gate 後載入並設定 backend；此處再次以函式內 lazy import 取得同一
    # module，避免 report_style 的 module globals 出現 Matplotlib 依賴。
    import matplotlib

    backend = str(matplotlib.get_backend()).lower()
    if backend != "agg":
        raise RuntimeError("renderer backend 必須是 Agg")
    with matplotlib.rc_context(rc=dict(style.rc_params)):
        yield style
