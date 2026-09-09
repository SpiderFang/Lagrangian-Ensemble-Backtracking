"""報告繁體中文字型解析的 deterministic contract 測試。

本檔完全以 monkeypatch、暫存 bytes 與假的 FreeType 字碼表模擬字型，不依賴測試主機
實際安裝的字型。測試涵蓋固定候選順序、全部必要 glyph、原始 bytes SHA-256、basename
path policy、frozen record 型別與無合格字型時不洩漏本機路徑的固定 ValueError。
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lagrangian_backtracking import report_font


class _FakeFT2Font:
    """用 filename 對應預先指定 charmap 的 FreeType 替身。

    測試只需要驗證 resolver 是否檢查 Unicode code point；不應載入真實 TTF／OTF
    二進位格式，因此這個替身把每個測試檔案的 basename 映射到假的 glyph 集合。
    """

    charmaps_by_filename: dict[str, set[int]] = {}
    opened_filenames: list[str] = []

    def __init__(self, filename: str) -> None:
        self.filename = Path(filename).name
        self.opened_filenames.append(self.filename)

    def get_charmap(self) -> dict[int, int]:
        """回傳與 FreeType ``get_charmap`` 相同語意的 code point mapping。"""

        return {codepoint: index for index, codepoint in enumerate(self.charmaps_by_filename[self.filename])}


def _patch_fake_fonts(
    monkeypatch: pytest.MonkeyPatch,
    paths_by_family: dict[str, Path],
    glyphs_by_filename: dict[str, set[str]],
) -> list[tuple[str, bool]]:
    """以假的 font_manager／FT2Font 安裝可觀測的測試字型環境。"""

    calls: list[tuple[str, bool]] = []

    def fake_findfont(properties: object, *, fallback_to_default: bool) -> str:
        """記錄 family 與 fallback policy，再回傳測試用的普通檔案。"""

        family = properties.get_family()[0]  # type: ignore[attr-defined]
        calls.append((family, fallback_to_default))
        return str(paths_by_family[family])

    charmaps = {filename: {ord(glyph) for glyph in glyphs} for filename, glyphs in glyphs_by_filename.items()}
    _FakeFT2Font.charmaps_by_filename = charmaps
    _FakeFT2Font.opened_filenames = []
    monkeypatch.setattr(report_font.font_manager, "findfont", fake_findfont)
    monkeypatch.setattr(report_font, "FT2Font", _FakeFT2Font)
    return calls


def test_constants_are_fixed_and_required_text_is_covered() -> None:
    """候選 family 順序與報告所需中文、狀態文字及 ASCII 數字必須固定存在。"""

    assert report_font.CJK_FONT_CANDIDATES == (
        "Noto Sans CJK TC",
        "Noto Sans TC",
        "Source Han Sans TC",
        "PingFang TC",
        "Heiti TC",
        "Arial Unicode MS",
    )
    required_phrases = (
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
    for phrase in required_phrases:
        assert set(phrase) <= report_font.REQUIRED_REPORT_GLYPHS
    assert set(string.ascii_letters + string.digits) <= report_font.REQUIRED_REPORT_GLYPHS
    assert set("%()[]-–—_/,.:;=+×≤≥<>°²μ（）：；，｜／") <= report_font.REQUIRED_REPORT_GLYPHS
    assert "\u0020" in report_font.REQUIRED_REPORT_GLYPHS
    assert "\u2212" in report_font.REQUIRED_REPORT_GLYPHS


def test_selection_direct_constructor_fails_closed_for_family_and_glyph_count() -> None:
    """直接建構 record 時，family 與必要字元數量都必須完全符合公開契約。"""

    valid_kwargs = {
        "filename": "report-font.ttf",
        "file_sha256": "a" * 64,
        "required_glyph_count": len(report_font.REQUIRED_REPORT_GLYPHS),
    }
    with pytest.raises(ValueError):
        report_font.CJKFontSelection(
            family="Noto Sans CJK SC",
            **valid_kwargs,
        )

    wrong_counts = (
        0,
        len(report_font.REQUIRED_REPORT_GLYPHS) - 1,
        len(report_font.REQUIRED_REPORT_GLYPHS) + 1,
    )
    for wrong_count in wrong_counts:
        with pytest.raises(ValueError):
            report_font.CJKFontSelection(
                family=report_font.CJK_FONT_CANDIDATES[0],
                **{**valid_kwargs, "required_glyph_count": wrong_count},
            )


def test_resolver_uses_fixed_order_all_glyphs_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """缺 glyph 的前一候選必須被跳過，並依固定順序選取下一個完整字型。"""

    first_path = tmp_path / "first-fake.ttf"
    selected_path = tmp_path / "selected-fake.ttf"
    first_path.write_bytes(b"first fake font bytes")
    selected_path.write_bytes(b"selected fake font bytes")

    incomplete_glyphs = set(report_font.REQUIRED_REPORT_GLYPHS)
    incomplete_glyphs.remove("失")
    calls = _patch_fake_fonts(
        monkeypatch,
        {
            report_font.CJK_FONT_CANDIDATES[0]: first_path,
            report_font.CJK_FONT_CANDIDATES[1]: selected_path,
        },
        {
            first_path.name: incomplete_glyphs,
            selected_path.name: set(report_font.REQUIRED_REPORT_GLYPHS),
        },
    )

    selection = report_font.resolve_cjk_font()

    assert calls == [
        (report_font.CJK_FONT_CANDIDATES[0], False),
        (report_font.CJK_FONT_CANDIDATES[1], False),
    ]
    assert selection.family == report_font.CJK_FONT_CANDIDATES[1]
    assert selection.filename == selected_path.name
    assert selection.required_glyph_count == len(report_font.REQUIRED_REPORT_GLYPHS)


def test_selection_is_frozen_typed_and_hashes_plain_file_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """成功結果必須是 frozen typed record，且只保存 basename 與完整 bytes 摘要。"""

    font_path = tmp_path / "report-font.otf"
    font_bytes = b"deterministic fake font payload"
    font_path.write_bytes(font_bytes)
    _patch_fake_fonts(
        monkeypatch,
        {report_font.CJK_FONT_CANDIDATES[0]: font_path},
        {font_path.name: set(report_font.REQUIRED_REPORT_GLYPHS)},
    )

    selection = report_font.resolve_cjk_font()

    assert isinstance(selection, report_font.CJKFontSelection)
    assert type(selection.family) is str
    assert type(selection.filename) is str
    assert type(selection.file_sha256) is str
    assert type(selection.required_glyph_count) is int
    assert selection.file_sha256 == hashlib.sha256(font_bytes).hexdigest()
    assert len(selection.file_sha256) == 64
    assert selection.file_sha256 == selection.file_sha256.lower()
    assert selection.filename == font_path.name
    assert "/" not in selection.filename
    assert "\\" not in selection.filename
    assert str(font_path) not in repr(selection)
    assert not hasattr(selection, "path")

    with pytest.raises(FrozenInstanceError):
        selection.family = "other"  # type: ignore[misc]


def test_lstat_rejects_symlink_directory_and_missing_before_fake_ft2font(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """symlink、directory 與 missing target 都須在 FreeType 前被拒絕。"""

    valid_path = tmp_path / "valid-fake.ttf"
    valid_path.write_bytes(b"valid fake font bytes")
    symlink_target = tmp_path / "symlink-target.ttf"
    symlink_target.write_bytes(b"symlink target fake font bytes")
    symlink_path = tmp_path / "symlink-fake.ttf"
    symlink_path.symlink_to(symlink_target)
    directory_path = tmp_path / "directory-fake.ttf"
    directory_path.mkdir()
    missing_path = tmp_path / "missing-fake.ttf"
    calls = _patch_fake_fonts(
        monkeypatch,
        {
            report_font.CJK_FONT_CANDIDATES[0]: symlink_path,
            report_font.CJK_FONT_CANDIDATES[1]: directory_path,
            report_font.CJK_FONT_CANDIDATES[2]: missing_path,
            report_font.CJK_FONT_CANDIDATES[3]: valid_path,
        },
        {valid_path.name: set(report_font.REQUIRED_REPORT_GLYPHS)},
    )

    selection = report_font.resolve_cjk_font()

    assert selection.family == report_font.CJK_FONT_CANDIDATES[3]
    assert calls == [
        (report_font.CJK_FONT_CANDIDATES[0], False),
        (report_font.CJK_FONT_CANDIDATES[1], False),
        (report_font.CJK_FONT_CANDIDATES[2], False),
        (report_font.CJK_FONT_CANDIDATES[3], False),
    ]
    assert _FakeFT2Font.opened_filenames == [valid_path.name]


def test_no_valid_font_raises_fixed_value_error_without_path_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """所有候選失敗時必須嘗試完整順序並只暴露固定、不含本機路徑的 ValueError。"""

    fake_path = tmp_path / "secret-local-font.ttf"
    fake_path.write_bytes(b"not a real font")
    calls = _patch_fake_fonts(
        monkeypatch,
        {family: fake_path for family in report_font.CJK_FONT_CANDIDATES},
        {fake_path.name: set()},
    )

    with pytest.raises(ValueError) as error_info:
        report_font.resolve_cjk_font()

    assert str(error_info.value) == "找不到涵蓋報告必要字元的合格繁體中文字型"
    assert str(tmp_path) not in str(error_info.value)
    assert fake_path.name not in str(error_info.value)
    assert calls == [(family, False) for family in report_font.CJK_FONT_CANDIDATES]


def test_invalid_file_targets_raise_fixed_error_without_path_or_basename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """只有 invalid file target 時，公開錯誤仍不得洩漏任一路徑或 basename。"""

    symlink_target = tmp_path / "private-target.ttf"
    symlink_target.write_bytes(b"private target")
    symlink_path = tmp_path / "private-link.ttf"
    symlink_path.symlink_to(symlink_target)
    directory_path = tmp_path / "private-directory.ttf"
    directory_path.mkdir()
    missing_path = tmp_path / "private-missing.ttf"
    paths = [symlink_path, directory_path, missing_path]
    paths.extend(tmp_path / f"private-{index}.missing" for index in range(3))
    paths_by_family = dict(zip(report_font.CJK_FONT_CANDIDATES, paths, strict=True))
    calls = _patch_fake_fonts(monkeypatch, paths_by_family, {})

    with pytest.raises(ValueError) as error_info:
        report_font.resolve_cjk_font()

    message = str(error_info.value)
    assert message == "找不到涵蓋報告必要字元的合格繁體中文字型"
    assert str(tmp_path) not in message
    assert all(path.name not in message for path in paths)
    assert calls == [(family, False) for family in report_font.CJK_FONT_CANDIDATES]
    assert _FakeFT2Font.opened_filenames == []
