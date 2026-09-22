"""由 2024–2025 forcing 衍生每站 48+2 到達時次的決定性 selector。

核心 48 時次按年、季節、spring/neap proxy 與三個潮內相位完整分層。spring/neap 使用
「潮汐強度」（一段時窗內潮位離差的均方根，即 rolling RMS）的季節中位數分組；三相位
分別最大正潮位導數、最小負導數與最小絕對導數。它們是可重現的潮位 proxy，不宣稱為
現場三維最大漲／退潮流。另選全期 local-domain 高波與強流各一時次，並要求與核心
時次互異及具完整 backward window。

潮汐強度時窗長度（``rolling_window_hours``／v2 的 ``tidal_strength_window_hours``）
直接決定 spring/neap 分類是否有意義：半日潮與全日潮的「大小潮週期」（spring-neap
cycle，因太陽與月球引潮力相對相位變化造成潮差週期性增減）約 14.77 日（對應
M2-S2 拍頻）或全日潮回歸週期約 13.66 日（K1-O1 拍頻）。若時窗長度恰好接近整個
大小潮週期（例如舊版 ``ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1``／
``ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1`` 採用的 14 天），該週期分量在時窗
內幾乎被平均掉（可用 sinc 衰減因子估計，14 天窗對 14.77 天週期的衰減係數約
0.054），使 rolling RMS 幾乎無法分辨大小潮，中位數切分退化為接近隨機分組。
``ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2`` 改用約一個潮日（25 小時，涵蓋
半日潮 12.42 小時週期與全日潮 23.93/25.82 小時週期各兩個循環）的短時窗，讓 RMS
序列本身仍保留大小潮週期的振盪，中位數切分才能有效區分大潮／小潮。
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from .scenarios import ArrivalTime, stable_identifier

SEASON_BY_MONTH = {
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

ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1 = "observation_year_stratified_48_plus_2_v1"
"""新版只在明示 observation 年份選錨點的 48+2 分層政策；潮汐強度沿用 14 天長窗
（見模組文件開頭），spring/neap 分類準確度低，僅為相容既有 manifest 保留。"""

ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2 = "observation_year_stratified_48_plus_2_v2"
"""與 v1 分層結構相同（1 個 observation 年份 × 4 季 × 2 潮況 × 3 相位 × 2 replicate
= 48，另加 2 事件），差異只在潮汐強度預設改用約一個潮日（25 小時）短時窗，可由
``tidal_strength_window_hours`` 或 config 的
``arrival_time_selection.tidal_strength_window_hours`` 覆寫；覆寫值與最終採用的
窗長皆會寫入每筆 arrival metadata（``tidal_strength_window_hours``）。"""

ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1 = "two_years_stratified_48_plus_2_v1"
"""未明示新版欄位時保留的兩 forcing 年份相容政策。"""

DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2 = 25.0
"""v2 policy 的潮汐強度時窗預設值（小時），約一個潮日，見模組文件開頭的理由。"""


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    """以 finite convolution 計算去局地平均後 RMS，不以 0 填補缺值。

    selector 進入本函式前已由 valid mask 篩除不支援時次，但長窗仍可能跨資料缺口；
    finite count 不足一半時輸出 NaN，使該時次不能參與 spring/neap 分類。
    """

    finite = np.isfinite(values)
    numeric = np.where(finite, values, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    count = np.convolve(finite.astype(np.float64), kernel, mode="same")
    mean = np.divide(
        np.convolve(numeric, kernel, mode="same"), count, out=np.full(values.shape, np.nan), where=count > 0
    )
    squared = np.where(finite, (values - mean) ** 2, 0.0)
    rms = np.sqrt(
        np.divide(
            np.convolve(squared, kernel, mode="same"),
            count,
            out=np.full(values.shape, np.nan),
            where=count >= max(window // 2, 1),
        )
    )
    return rms


def select_arrival_times(
    *,
    study_site_id: str,
    time_utc_ns: np.ndarray,
    elevation_m: np.ndarray,
    significant_wave_height_m: np.ndarray,
    current_speed_mps: np.ndarray,
    valid_forcing: np.ndarray,
    backward_window_available: np.ndarray,
    design_version: str,
    rolling_window_hours: float = 24.0 * 14.0,
    tidal_strength_window_hours: float | None = None,
    observation_years: tuple[int, ...] | list[int] | None = None,
    replicates: int | None = None,
    selection_policy: str | None = None,
    policy: str | None = None,
    event_supplement_count: int = 2,
) -> list[ArrivalTime]:
    """產生恰好 50 個 arrival records，資料不足立即失敗。

    所有輸入均為同一一維 UTC 軸的站點代表統計；Hs/current 可由 local-domain 有效格點
    的穩健高分位數產生，實際 aggregation 定義必須寫入 metadata。選取順序完全由數值、
    UTC 與固定 tie-break 決定，不使用亂數。未明示新版 ``selection_policy`` 時，函式
    維持舊版「兩個 forcing 年份 × 四季 × spring/neap × 三相位」的 48+2 行為，供
    舊 config／manifest 相容載入；新版 policy 則把兩個 replicate 放在同一個
    ``observation_years × season × tide_class × phase`` stratum 內，2024 只可作
    ``backward_window_available`` 的前置支援，不能成為 observation anchor。

    新版每個 stratum 的兩筆 replicate 依 score（最大值事件或最小值 proxy）再以
    UTC 由早到晚排序，且每筆均從全域已選集合排除，因此同一 stratum 不會重複 UTC。
    缺值、域外與 gap-safe mask 仍由呼叫端先傳入 ``valid_forcing`` 與
    ``backward_window_available``；本函式不補值、不跨缺口外插。

    ``rolling_window_hours`` 是 ``ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1``／
    ``ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1`` 判定潮汐強度的固定時窗（預設
    14 天），為保留既有 manifest 位元相同，此參數對這兩個政策維持原行為，不接受
    ``tidal_strength_window_hours`` 覆寫。``ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2``
    改以 ``tidal_strength_window_hours``（省略時預設
    ``DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2``＝25 小時，約一個潮日）決定潮汐強度時窗；
    只有 v2 才接受這個參數，其他政策明示傳入即視為設定錯誤並立即失敗，避免呼叫端
    誤以為它同時影響 v1／legacy 而在不知情下改變既有 arrival identity。最終採用的
    窗長會原樣寫入每筆 v2 arrival 的 ``metadata["tidal_strength_window_hours"]``。
    """

    time_values = np.asarray(time_utc_ns, dtype=np.int64)
    elevation = np.asarray(elevation_m, dtype=np.float64)
    wave_height = np.asarray(significant_wave_height_m, dtype=np.float64)
    current_speed = np.asarray(current_speed_mps, dtype=np.float64)
    valid = np.asarray(valid_forcing, dtype=bool) & np.asarray(backward_window_available, dtype=bool)
    if time_values.ndim != 1 or time_values.size < 50 or np.any(np.diff(time_values) <= 0):
        raise ValueError("arrival selector time_utc_ns 必須一維、嚴格遞增且至少 50 點")
    if any(item.shape != time_values.shape for item in (elevation, wave_height, current_speed, valid)):
        raise ValueError("arrival selector 所有序列 shape 必須相同")
    finite = np.isfinite(elevation) & np.isfinite(wave_height) & np.isfinite(current_speed)
    valid &= finite
    seconds = time_values.astype(np.float64) / 1_000_000_000
    nominal_seconds = float(np.median(np.diff(seconds)))
    if nominal_seconds <= 0:
        raise ValueError("arrival selector nominal interval 無效")
    derivative = np.gradient(elevation, seconds)

    # ``policy`` 是較短的外部 caller 參數別名；兩者同時出現時不能默默選一個，避免
    # config 與 input builder 產生不同的 arrival identity。這段必須在潮汐強度時窗
    # 決定前完成，因為 v2 的窗長預設值依 policy 而異。
    if policy is not None:
        if selection_policy is not None and selection_policy != policy:
            raise ValueError("arrival selector 的 policy 與 selection_policy 不可互相矛盾")
        selection_policy = policy

    if selection_policy is None and (
        observation_years is not None or (replicates is not None and replicates != 1)
    ):
        raise ValueError("observation_years／replicates 必須搭配明示 selection policy")

    if selection_policy not in {
        None,
        ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1,
        ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
        ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2,
    }:
        raise ValueError(f"不支援的 arrival selection policy：{selection_policy!r}")

    is_v2_policy = selection_policy == ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2
    is_observation_policy = (
        selection_policy == ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1 or is_v2_policy
    )

    # v1／legacy 的潮汐強度時窗必須維持既有預設（14 天）與既有 manifest 位元相同，
    # 因此明示 ``tidal_strength_window_hours`` 卻搭配非 v2 policy 一律 fail closed，
    # 避免呼叫端誤以為這個覆寫同時影響 v1／legacy 而在不知情下改變 arrival identity。
    if is_v2_policy:
        effective_tidal_strength_window_hours = (
            DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2
            if tidal_strength_window_hours is None
            else float(tidal_strength_window_hours)
        )
        if effective_tidal_strength_window_hours <= 0:
            raise ValueError("v2 的 tidal_strength_window_hours 必須為正數")
    else:
        if tidal_strength_window_hours is not None:
            raise ValueError(
                "tidal_strength_window_hours 只有 "
                f"{ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2} 才接受覆寫"
            )
        effective_tidal_strength_window_hours = rolling_window_hours

    window = max(int(round(effective_tidal_strength_window_hours * 3_600.0 / nominal_seconds)), 3)
    tidal_strength = _rolling_rms(elevation, window)
    datetimes = [datetime.fromtimestamp(int(value) / 1_000_000_000, tz=UTC) for value in time_values]
    years = np.array([item.year for item in datetimes])
    seasons = np.array([SEASON_BY_MONTH[item.month] for item in datetimes])
    selected: set[int] = set()
    records: list[ArrivalTime] = []

    def choose(candidates: np.ndarray, score: np.ndarray, *, maximize: bool) -> int:
        """依 score 與較早 UTC tie-break 選尚未使用的有限候選。"""

        available = [
            int(index) for index in candidates if int(index) not in selected and np.isfinite(score[index])
        ]
        if not available:
            raise ValueError("arrival 分層沒有足夠且具完整 backward window 的候選")
        return min(
            available,
            key=lambda index: ((-score[index] if maximize else score[index]), int(time_values[index])),
        )

    def choose_many(
        candidates: np.ndarray,
        score: np.ndarray,
        *,
        maximize: bool,
        count: int,
    ) -> list[int]:
        """依固定 score／UTC 次序取同一 stratum 的多個 replicate。

        排序先依事件分數、再依 UTC；全域 ``selected`` 會排除先前 stratum 已使用的
        時次。這個 helper 不做隨機抽樣，故相同 accepted forcing 與設定一定得到同一
        組 replicate rank；候選不足時直接 fail closed，不以最近值或重複 UTC 補足。
        """

        if count < 1:
            raise ValueError("arrival selector replicate count 必須為正")
        available = [
            int(index)
            for index in candidates
            if int(index) not in selected and np.isfinite(score[index])
        ]
        ordered = sorted(
            available,
            key=lambda index: (
                -float(score[index]) if maximize else float(score[index]),
                int(time_values[index]),
            ),
        )
        if len(ordered) < count:
            raise ValueError("arrival 分層沒有足夠且具完整 backward window 的 replicate 候選")
        return ordered[:count]

    unique_years = sorted(set(int(value) for value in years))
    if is_observation_policy:
        if observation_years is None:
            raise ValueError("新版 observation arrival policy 必須明示 observation_years")
        selected_years = [int(value) for value in observation_years]
        if not selected_years or len(set(selected_years)) != len(selected_years):
            raise ValueError("observation_years 必須是非空且不重複的年份")
        if not set(selected_years).issubset(set(unique_years)):
            raise ValueError("observation_years 必須是 forcing time axis 年份的子集")
        if replicates is not None and (
            isinstance(replicates, bool) or type(replicates) is not int
        ):
            raise ValueError("新版 observation arrival policy 的 replicates 必須是整數")
        replicate_count = 2 if replicates is None else int(replicates)
        if replicate_count != 2:
            raise ValueError("新版 observation arrival policy 的 replicates 必須固定為 2")
        if event_supplement_count != 2:
            raise ValueError("新版 observation arrival policy 的 event_supplement_count 必須固定為 2")
        # 新版固定 4×2×2×3=48；即使未來呼叫端傳入多個 observation 年份，也必須
        # 由設定先明確決定分層總量，不能讓年數增加後悄悄產生超過 50 筆 arrival。
        if len(selected_years) != 1:
            raise ValueError("新版 observation arrival policy 目前必須指定恰好一個 observation 年份")
        # v1 沿用固定字面值等同 selection_policy 字串，metadata／identity 位元不變；
        # v2 則直接使用 selection_policy，讓 identity fields 與 metadata 都能辨識新政策。
        selection_label = selection_policy
        for year in selected_years:
            for season in ("DJF", "MAM", "JJA", "SON"):
                cell = valid & (years == year) & (seasons == season) & np.isfinite(tidal_strength)
                indices = np.flatnonzero(cell)
                if indices.size < 12:
                    raise ValueError(f"{study_site_id} {year}/{season} 有效候選不足 12")
                threshold = float(np.median(tidal_strength[indices]))
                class_masks = {
                    "spring_proxy": indices[tidal_strength[indices] >= threshold],
                    "neap_proxy": indices[tidal_strength[indices] < threshold],
                }
                phase_specs = (
                    ("fastest_rising", derivative, True),
                    ("fastest_falling", derivative, False),
                    ("slack_proxy", np.abs(derivative), False),
                )
                for tide_class, class_indices in class_masks.items():
                    required_candidates = len(phase_specs) * replicate_count
                    if class_indices.size < required_candidates:
                        raise ValueError(
                            f"{study_site_id} {year}/{season}/{tide_class} 有效候選不足 "
                            f"{required_candidates}"
                        )
                    for phase, score, maximize in phase_specs:
                        for replicate_rank, index in enumerate(
                            choose_many(
                                class_indices,
                                score,
                                maximize=maximize,
                                count=replicate_count,
                            )
                        ):
                            selected.add(index)
                            stratum_id = f"{year}/{season}/{tide_class}/{phase}"
                            fields = [
                                study_site_id,
                                str(int(time_values[index])),
                                tide_class,
                                phase,
                                str(replicate_rank),
                                selection_label,
                                design_version,
                            ]
                            records.append(
                                ArrivalTime(
                                    arrival_time_id=stable_identifier("arr", fields),
                                    study_site_id=study_site_id,
                                    time_utc_ns=int(time_values[index]),
                                    year=year,
                                    season=season,
                                    tide_class=tide_class,
                                    phase_or_event=phase,
                                    metadata={
                                        "elevation_m": float(elevation[index]),
                                        "elevation_derivative_mps": float(derivative[index]),
                                        "tidal_strength_proxy_m": float(tidal_strength[index]),
                                        "selection_policy": selection_label,
                                        "observation_year": year,
                                        "replicate_rank": replicate_rank,
                                        "replicate_rank_one_based": replicate_rank + 1,
                                        "stratum_id": stratum_id,
                                        **(
                                            {
                                                "tidal_strength_window_hours": (
                                                    effective_tidal_strength_window_hours
                                                )
                                            }
                                            if is_v2_policy
                                            else {}
                                        ),
                                    },
                                )
                            )
        remaining = np.flatnonzero(
            valid
            & np.isin(years, np.asarray(selected_years, dtype=np.int64))
            & ~np.isin(np.arange(time_values.size), list(selected))
        )
        event_specs = (
            ("high_wave_event", wave_height),
            ("strong_current_event", current_speed),
        )
        for event_name, score in event_specs:
            index = choose(remaining, score, maximize=True)
            selected.add(index)
            remaining = remaining[remaining != index]
            fields = [
                study_site_id,
                str(int(time_values[index])),
                event_name,
                selection_label,
                design_version,
            ]
            records.append(
                ArrivalTime(
                    arrival_time_id=stable_identifier("arr", fields),
                    study_site_id=study_site_id,
                    time_utc_ns=int(time_values[index]),
                    year=int(years[index]),
                    season=str(seasons[index]),
                    tide_class="event",
                    phase_or_event=event_name,
                    metadata={
                        "significant_wave_height_m": float(wave_height[index]),
                        "current_speed_mps": float(current_speed[index]),
                        "selection_policy": selection_label,
                        "observation_year": int(years[index]),
                        "stratum_id": f"event/{event_name}",
                        **(
                            {
                                "tidal_strength_window_hours": (
                                    effective_tidal_strength_window_hours
                                )
                            }
                            if is_v2_policy
                            else {}
                        ),
                    },
                )
            )
    else:
        # 相容路徑保持原本 identity 欄位與選取順序，讓舊 config／manifest hash 與
        # 兩年份 selector 結果不因新增 observation policy 而漂移。
        if len(unique_years) != 2:
            raise ValueError(f"baseline arrival selector 預期恰好兩個年份，實際={unique_years}")
        for year in unique_years:
            for season in ("DJF", "MAM", "JJA", "SON"):
                cell = valid & (years == year) & (seasons == season) & np.isfinite(tidal_strength)
                indices = np.flatnonzero(cell)
                if indices.size < 6:
                    raise ValueError(f"{study_site_id} {year}/{season} 有效候選不足 6")
                threshold = float(np.median(tidal_strength[indices]))
                class_masks = {
                    "spring_proxy": indices[tidal_strength[indices] >= threshold],
                    "neap_proxy": indices[tidal_strength[indices] < threshold],
                }
                for tide_class, class_indices in class_masks.items():
                    phase_specs = (
                        ("fastest_rising", derivative, True),
                        ("fastest_falling", derivative, False),
                        ("slack_proxy", np.abs(derivative), False),
                    )
                    for phase, score, maximize in phase_specs:
                        index = choose(class_indices, score, maximize=maximize)
                        selected.add(index)
                        fields = [
                            study_site_id,
                            str(int(time_values[index])),
                            tide_class,
                            phase,
                            design_version,
                        ]
                        records.append(
                            ArrivalTime(
                                arrival_time_id=stable_identifier("arr", fields),
                                study_site_id=study_site_id,
                                time_utc_ns=int(time_values[index]),
                                year=year,
                                season=season,
                                tide_class=tide_class,
                                phase_or_event=phase,
                                metadata={
                                    "elevation_m": float(elevation[index]),
                                    "elevation_derivative_mps": float(derivative[index]),
                                    "tidal_strength_proxy_m": float(tidal_strength[index]),
                                },
                            )
                        )

        remaining = np.flatnonzero(valid & ~np.isin(np.arange(time_values.size), list(selected)))
        for event_name, score in (("high_wave_event", wave_height), ("strong_current_event", current_speed)):
            index = choose(remaining, score, maximize=True)
            selected.add(index)
            remaining = remaining[remaining != index]
            fields = [study_site_id, str(int(time_values[index])), event_name, design_version]
            records.append(
                ArrivalTime(
                    arrival_time_id=stable_identifier("arr", fields),
                    study_site_id=study_site_id,
                    time_utc_ns=int(time_values[index]),
                    year=int(years[index]),
                    season=str(seasons[index]),
                    tide_class="event",
                    phase_or_event=event_name,
                    metadata={
                        "significant_wave_height_m": float(wave_height[index]),
                        "current_speed_mps": float(current_speed[index]),
                    },
                )
            )
    if len(records) != 50 or len({item.time_utc_ns for item in records}) != 50:
        raise RuntimeError("arrival selector 未產生 50 個唯一 UTC")
    return records
