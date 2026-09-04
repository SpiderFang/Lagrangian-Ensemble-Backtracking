"""``report-validate`` 公開命令的 bounded CLI contract tests。

本檔刻意把 unit test 與一個真實 validator smoke test 分開：前者以 monkeypatch
確認整合 parser、dispatch、JSON 序列化與 shell exit code 不改寫 validator report；
後者只使用不存在的 synthetic 路徑，確認真正的 report-v1 validator 能在輸入缺失時
回傳固定 JSON-safe 失敗報告。測試不建立 OCM schema 3、NWW3 schema 1、trajectory、
aggregate 或 report 科學產品；即使 CLI 成功驗證 synthetic engineering contract，
也不代表任何正式科學成果、條件式來源足跡、相對來源權重或觀測驗證成立。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lagrangian_backtracking.cli as cli
from lagrangian_backtracking.report_release import validate_report_release


def _directory_names(directory: Path) -> tuple[str, ...]:
    """記錄測試暫存目錄的直接 children，作為 CLI 無寫入副作用的證據。

    ``report-validate`` 的契約是唯讀；只比較名稱即可捕捉建立 partial、final、
    sidecar 或其他暫存檔的行為，同時避免把測試 fixture 的檔案內容納入本測試責任。
    """

    return tuple(sorted(entry.name for entry in directory.iterdir()))


def _pretty_json(report: dict[str, object]) -> str:
    """以公開 CLI 約定產生預期 stdout，保留非 ASCII 文字與固定 key 順序。"""

    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_report_validate_parser_requires_exactly_one_release_path(tmp_path: Path) -> None:
    """parser 接受唯一明示 release，缺參數、額外參數與未知 option 均維持 argparse 2。"""

    release = tmp_path / "synthetic.report-v1"
    parser = cli._report_validate_parser()
    assert parser.parse_args([str(release)]).release == release

    for argv in ([], [str(release), "extra"], ["--unknown", str(release)]):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(argv)
        assert error.value.code == 2


def test_report_validate_success_dispatch_prints_exact_json_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """valid=True 應由 main dispatch、原樣輸出 pretty JSON 並回傳 0。"""

    release = tmp_path / "synthetic.report-v1"
    expected = {
        "valid": True,
        "errors": [],
        "summary": {"說明": "工程驗證，不是科學成果"},
    }
    calls: list[Path] = []

    def fake_validator(path: Path) -> dict[str, object]:
        """攔截 validator 以隔離 CLI contract，並確認 caller path 未被猜測或改寫。"""

        calls.append(path)
        return expected

    monkeypatch.setattr(cli, "validate_report_release", fake_validator)
    before = _directory_names(tmp_path)

    assert cli.main(["report-validate", str(release)]) == 0

    captured = capsys.readouterr()
    assert captured.out == _pretty_json(expected)
    assert json.loads(captured.out) == expected
    assert captured.err == ""
    assert calls == [release]
    assert _directory_names(tmp_path) == before


def test_report_validate_failure_dispatch_prints_exact_json_and_returns_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """valid 非 True 的固定 validator report 應保持原樣並回傳 2。"""

    release = tmp_path / "tampered.report-v1"
    expected = {
        "valid": False,
        "errors": [{"stage": "manifest", "reason": "sha256 mismatch"}],
        "summary": {},
    }

    monkeypatch.setattr(cli, "validate_report_release", lambda path: expected)
    before = _directory_names(tmp_path)

    assert cli.main(["report-validate", str(release)]) == 2

    captured = capsys.readouterr()
    assert captured.out == _pretty_json(expected)
    assert json.loads(captured.out) == expected
    assert captured.err == ""
    assert _directory_names(tmp_path) == before


def test_report_validate_real_missing_release_is_safe_and_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """真正 validator 面對不存在 release 時只輸出固定失敗 JSON，不洩漏絕對路徑。"""

    missing = tmp_path / "does-not-exist.report-v1"
    expected = validate_report_release(missing)
    before = _directory_names(tmp_path)

    assert cli.main(["report-validate", str(missing)]) == 2

    captured = capsys.readouterr()
    assert captured.out == _pretty_json(expected)
    assert json.loads(captured.out) == expected
    assert expected["valid"] is False
    assert str(missing) not in captured.out
    assert str(tmp_path) not in captured.out
    assert str(missing) not in captured.err
    assert _directory_names(tmp_path) == before


def test_report_validate_main_keeps_argparse_exit_two_for_missing_or_unknown_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """整合 main 對 report-validate 的缺參數與未知參數仍交由 argparse 回傳 2。"""

    release = tmp_path / "synthetic.report-v1"
    for argv in (
        ["report-validate"],
        ["report-validate", str(release), "extra"],
        ["report-validate", "--unknown", str(release)],
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 2
        capsys.readouterr()

    assert _directory_names(tmp_path) == ()
