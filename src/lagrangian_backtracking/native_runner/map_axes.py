"""地圖類圖的經緯度座標軸統一規則（WP-71）。

為什麼需要這個模組
------------------
報告裡五個研究地點的地圖類圖（水平軌跡總覽、條件式來源足跡）原本各自依內容決定座標軸範圍與刻度，
所以五地點的經度跨度、緯度跨度、刻度間隔與刻度小數位數都不同，軸的兩端也不一定有數字。
本模組提供「同一圖種五地點共用」的軸規格：

* 刻度標籤一律一位小數、含方位字母（例如「25.0°N」「121.4°E」）；
* 同一圖種的五個地點，經度跨度（度）相同、緯度跨度（度）相同、刻度間隔相同；
* 每個地點的軸範圍上下限都落在刻度（間隔的整數倍）上，所以軸兩端各有一個數字；
* 軸範圍必須完整包住該地點要呈現的內容（外框、軌跡等），多出來的空間盡量均分在兩側；
* 五地點用同一個等比例係數（一度緯度對一度經度的畫面長度比＝``1 / cos(參考緯度)``），
  所以五地點的主圖框大小完全相同。參考緯度取五地點窗口中心緯度的最小與最大之中點，
  各地點實際的長寬比誤差（相對 ``1 / cos(該地點緯度)``）由 :func:`aspect_error_fraction` 量化並記入 manifest。

本模組只負責幾何與文字格式，不畫圖，不讀任何資料檔。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "AXIS_TITLE_LON_ZH",
    "AXIS_TITLE_LAT_ZH",
    "AxisSpec",
    "AxisWindow",
    "aspect_error_fraction",
    "format_lat",
    "format_lon",
    "plan_axis_spec",
    "window_for_content",
]

AXIS_TITLE_LON_ZH = "經度"
AXIS_TITLE_LAT_ZH = "緯度"
"""座標軸標題（全報告統一；單位與方位已在刻度標籤中，不再重複）。"""

_EPS = 1e-9


def format_lon(value: float, _pos: int | None = None) -> str:
    """經度刻度標籤：一位小數加「°E」，例如 ``121.4°E``。

    刻度必須是 0.1 度的整數倍；不是整數倍的值若被四捨五入成一位小數會誤導讀者，所以直接報錯。
    """

    _require_tenth(value)
    return f"{value:.1f}°E"


def format_lat(value: float, _pos: int | None = None) -> str:
    """緯度刻度標籤：一位小數加「°N」，例如 ``25.0°N``（規則同 :func:`format_lon`）。"""

    _require_tenth(value)
    return f"{value:.1f}°N"


def _require_tenth(value: float) -> None:
    if abs(value * 10.0 - round(value * 10.0)) > 1e-6:
        raise ValueError(f"刻度值 {value!r} 不是 0.1 度的整數倍，不可四捨五入成一位小數標示")


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """同一圖種五地點共用的軸規格。

    ``step_deg``：經緯向共用的刻度間隔（度，須為 0.1 的整數倍）；``lon_span_deg``、``lat_span_deg``：
    經度、緯度跨度（度，皆為 ``step_deg`` 的整數倍）；``ref_lat_deg``：決定畫面長寬比的參考緯度。
    """

    step_deg: float
    lon_span_deg: float
    lat_span_deg: float
    ref_lat_deg: float

    @property
    def aspect(self) -> float:
        """matplotlib ``set_aspect`` 用的比例：一度緯度的畫面長度除以一度經度的畫面長度。"""

        return 1.0 / math.cos(math.radians(self.ref_lat_deg))

    @property
    def frame_width_over_height(self) -> float:
        """主圖框的寬高比（等比例後）。"""

        return self.lon_span_deg / (self.lat_span_deg * self.aspect)


@dataclass(frozen=True, slots=True)
class AxisWindow:
    """一個地點的軸範圍（上下限皆為刻度）與刻度。"""

    lon_lo: float
    lon_hi: float
    lat_lo: float
    lat_hi: float
    step_deg: float

    @property
    def lon_ticks(self) -> list[float]:
        return _ticks(self.lon_lo, self.lon_hi, self.step_deg)

    @property
    def lat_ticks(self) -> list[float]:
        return _ticks(self.lat_lo, self.lat_hi, self.step_deg)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.lon_lo, self.lon_hi, self.lat_lo, self.lat_hi)


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    k0 = round(lo / step)
    k1 = round(hi / step)
    return [round(k * step, 6) for k in range(k0, k1 + 1)]


def _min_units(lo: float, hi: float, step: float) -> tuple[int, int]:
    """包住 ``[lo, hi]`` 的最小刻度格範圍（起點格序號、終點格序號）。"""

    return math.floor(lo / step + _EPS), math.ceil(hi / step - _EPS)


def plan_axis_spec(
    needs: Mapping[str, tuple[float, float, float, float]], *, step_deg: float
) -> AxisSpec:
    """由各地點「必須包住的範圍」決定共用軸規格。

    ``needs``：``{地點: (經度下, 經度上, 緯度下, 緯度上)}``。跨度取各地點所需的最大值（向外擴到刻度整數倍）；
    參考緯度取各地點需求範圍中心緯度的最小與最大之中點，四捨五入到 0.1 度。
    """

    if not needs:
        raise ValueError("needs 不可為空")
    if abs(step_deg * 10.0 - round(step_deg * 10.0)) > 1e-9 or step_deg <= 0:
        raise ValueError("step_deg 必須是 0.1 度的正整數倍")
    lon_units = 0
    lat_units = 0
    centers: list[float] = []
    for lon_lo, lon_hi, lat_lo, lat_hi in needs.values():
        a, b = _min_units(lon_lo, lon_hi, step_deg)
        lon_units = max(lon_units, b - a)
        a, b = _min_units(lat_lo, lat_hi, step_deg)
        lat_units = max(lat_units, b - a)
        centers.append((lat_lo + lat_hi) / 2.0)
    ref = round((min(centers) + max(centers)) / 2.0, 1)
    return AxisSpec(step_deg, round(lon_units * step_deg, 6), round(lat_units * step_deg, 6), ref)


def _place(lo: float, hi: float, units: int, step: float) -> tuple[float, float]:
    """在刻度格上放一個寬 ``units`` 格的窗口，包住 ``[lo, hi]`` 且盡量置中。"""

    a, b = _min_units(lo, hi, step)
    if b - a > units:
        raise ValueError(f"需求範圍 [{lo:.4f}, {hi:.4f}] 需要 {b - a} 格，超過規格的 {units} 格")
    slack = units - (b - a)
    # 兩側各分一半；奇數格時多出的一格給遠離需求中心較需要的一側（取使窗口中心最接近需求中心者）
    best: tuple[float, int] | None = None
    for left in range(slack + 1):
        k0 = a - left
        center = (k0 + units / 2.0) * step
        err = abs(center - (lo + hi) / 2.0)
        if best is None or err < best[0] - 1e-12:
            best = (err, k0)
    assert best is not None
    k0 = best[1]
    return round(k0 * step, 6), round((k0 + units) * step, 6)


def window_for_content(spec: AxisSpec, need: tuple[float, float, float, float]) -> AxisWindow:
    """依共用規格為一個地點決定軸範圍：跨度與間隔同規格，上下限在刻度上，且包住 ``need``。"""

    lon_lo, lon_hi, lat_lo, lat_hi = need
    step = spec.step_deg
    lon_units = round(spec.lon_span_deg / step)
    lat_units = round(spec.lat_span_deg / step)
    x0, x1 = _place(lon_lo, lon_hi, lon_units, step)
    y0, y1 = _place(lat_lo, lat_hi, lat_units, step)
    return AxisWindow(x0, x1, y0, y1, step)


def aspect_error_fraction(spec: AxisSpec, site_lat_deg: float) -> float:
    """共用長寬比相對該地點真實比例的誤差：``cos(參考緯度) / cos(地點緯度) - 1``。

    正值表示東西向相對南北向被拉長；例如 +0.018 表示畫面上的 1 km 東西距離比真實比例長 1.8%。
    """

    return math.cos(math.radians(spec.ref_lat_deg)) / math.cos(math.radians(site_lat_deg)) - 1.0


def window_record(spec: AxisSpec, windows: Mapping[str, AxisWindow], site_lat_deg: Mapping[str, float]) -> dict[str, object]:
    """manifest 用紀錄：共用規格、各地點軸範圍與刻度、長寬比誤差。"""

    return {
        "step_deg": spec.step_deg,
        "lon_span_deg": spec.lon_span_deg,
        "lat_span_deg": spec.lat_span_deg,
        "ref_lat_deg": spec.ref_lat_deg,
        "aspect_dy_over_dx": spec.aspect,
        "sites": {
            site: {
                "lon_lo": w.lon_lo,
                "lon_hi": w.lon_hi,
                "lat_lo": w.lat_lo,
                "lat_hi": w.lat_hi,
                "lon_ticks": w.lon_ticks,
                "lat_ticks": w.lat_ticks,
                "aspect_error_fraction": aspect_error_fraction(spec, site_lat_deg[site]),
            }
            for site, w in windows.items()
        },
    }


def sequence_is_regular(values: Sequence[float], step: float) -> bool:
    """刻度序列是否等間距（測試用）。"""

    return all(abs((b - a) - step) < 1e-6 for a, b in zip(values, values[1:], strict=False))
