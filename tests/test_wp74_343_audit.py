"""WP-74：fig_audit 之資料指紋對字級、線寬、位置不敏感，對資料與用語敏感；文字稽核逐一回報字級。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wp74_343"))
import fig_audit as fa  # noqa: E402


def _fig(fs: float, lw: float, y=(0.0, 1.0, 0.5), label="流速"):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=50)
    ax.plot([0, 1, 2], list(y), lw=lw, marker="o")
    ax.set_xlabel("日期", fontsize=fs)
    ax.set_ylabel(label, fontsize=fs)
    ax.tick_params(labelsize=fs)
    return fig


def test_fingerprint_ignores_style_but_not_data_or_wording() -> None:
    a, b = _fig(9, 1.0), _fig(8, 2.0)
    c, d = _fig(8, 1.0, y=(0.0, 1.0, 0.6)), _fig(8, 1.0, label="水位")
    fa_, fb, fc, fd = (fa.fingerprint(f) for f in (a, b, c, d))
    assert fa_ == fb
    assert fa_["sha256"] != fc["sha256"] and fa_["sha256"] != fd["sha256"]
    for f in (a, b, c, d):
        plt.close(f)


def test_text_audit_reports_sizes() -> None:
    fig = _fig(8, 1.0)
    audit = fa.text_audit(fig)
    s = fa.text_summary(audit)
    assert s["sizes_pt"] == [8.0] and s["min_size_pt"] == 8.0 and s["n_texts"] >= 4
    assert np.all([a["size_pt"] == 8.0 for a in audit])
    plt.close(fig)
