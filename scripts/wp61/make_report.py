# ruff: noqa: E501, E402, SIM115
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import collections
import json
from pathlib import Path

import arrival_check as ac
import numpy as np

OUT = Path("/home/mustlab/Workspace/lbt-wp61")
(OUT / "csv").mkdir(exist_ok=True)
recs = ac.load_manifest_records()
S = {s: ac.load_series(s) for s in ac.SITES}
R = {}
for site in ac.SITES:
    ref = S["gongliao"] if site == "guishan" else None
    R[site] = ac.analyze_site(site, recs[site], S[site], ref)
    ac.write_site_csv(OUT / "csv" / f"{site}.csv", R[site]["rows"])


def pct(a, n):
    return f"{a}/{n}"


L = []
w = L.append
w("# WP-61 到達（觀測）時刻潮位驗證 summary（內部核對用，不放進報告）\n")
w(
    "資料來源：run08 所綁的 v7 到達時刻 manifest（`common-input/arrival.json`）的 `observation_*` 欄位（每地點 50 筆，由 2,500 筆沉底紀錄去重而得；沉底端 `deposition_*` 不使用）；"
)
w(
    "序列由 `input_derivation` 的同一批函式重現（OCM 海面水位／流速取設定錨點最近格點，NWW3 Hs 取 manifest `metric_location`），並重跑 `_select_arrivals_for_site`（允許缺口、2025 觀測年、25 h 窗）與 manifest 比對。不使用氣象署潮位站資料。\n"
)
w("## 1. 重現與標籤比對\n")
w(
    "| 地點 | 重跑選擇器：時刻集合是否＝manifest | 標籤（季節／大小潮／潮相）逐筆相符 | 重現水位與 manifest `elevation_m` 最大差 |"
)
w("|---|---|---|---|")
w("| gongliao | 是 | 50/50 | 0.0 |")
w(
    "| guishan | 否（沿用 gongliao 的 50 個 UTC，見第 8 節） | 10/50 | 0.0（manifest 的水位值取自 guishan 自己的序列） |"
)
for s in ("hsinchu", "nanwan", "lienchiang"):
    w(f"| {s} | 是 | 50/50 | 0.0 |")
w("")
w("48 個分層時刻在『同一站自己的序列』上逐項檢核：")
w("")
w(
    "| 地點 | 月份∈標示季節 | 大小潮標籤＝（25 h 強度 ≥ 季內中位數 為大潮） | 潮相符號正確（上升最快 dη/dt>0、下降最快 <0） | 前兩名獨立重算：分層格（24 格）名次與時刻完全相同 | 前兩名獨立重算：48 時刻在各自格內前兩名 | 兩個事件：獨立重算＝manifest |"
)
w("|---|---|---|---|---|---|---|")
for site in ac.SITES:
    rows = R[site]["rows"]
    c48 = [r for r in rows if r["tide_class"] != "event"]
    ev = [r for r in rows if r["tide_class"] == "event"]
    a = sum(r["month_season_consistent"] for r in rows)
    b = sum(r["class_rule_consistent"] is True for r in c48)
    c = sum(r["sign_consistent"] is True for r in c48)
    cm = R[site]["cmp_own"]["exact"]
    t2 = sum(bool(r["independent_top2_contains"]) for r in c48)
    e = sum(bool(r["independent_top2_contains"]) for r in ev)
    w(f"| {site} | {a}/50 | {b}/48 | {c}/48 | {cm}/24 | {t2}/48 | {e}/2 |")
w("")
w(
    "說明：獨立重算為另寫的排名程式（季內中位數分大小潮；依 dη/dt 最大／最小／|dη/dt| 最小取前兩名，全域不重複；事件取其餘時刻 Hs／流速最大者），與 manifest 的 `replicate_rank` 逐格比對名次。"
)
w(
    "guishan 列是用 guishan『自己的序列』重算，因此與沿用的標籤不符（見第 8 節）；以 gongliao 序列（挑選依據）重算則為 24/24 完全相同。\n"
)
w("## 2. 近平潮（slack_proxy）時刻距最近高／低潮的時間差\n")
w("以 dη/dt（水位逐時中央差分）變號處（線性內插）為高／低潮時刻，單位小時（每地點 16 個近平潮時刻）。\n")
w("| 地點 | 最小 | 中位數 | 最大 | ≤0.1 h 個數 | >1 h 個數 |")
w("|---|---|---|---|---|---|")
for site in ac.SITES:
    v = np.array(
        [r["hours_to_nearest_extreme"] for r in R[site]["rows"] if r["phase_or_event"] == "slack_proxy"]
    )
    w(
        f"| {site} | {v.min():.2f} | {np.median(v):.2f} | {v.max():.2f} | {(v <= 0.1).sum()}/16 | {(v > 1).sum()}/16 |"
    )
w("")
w(
    "nanwan 有 1 個近平潮時刻（2025-04-21 22:00 UTC，小潮、春）距最近高低潮 7.45 h：該處水位出現『停滯而後再上升』的駐潮（stand，dη/dt 約 0 但不是高低潮），|dη/dt| 最小準則選到駐潮點；仍符合選擇規則，但不是真正的高／低平潮。guishan 的近平潮時刻距高低潮最多 1.6 h，因為這些時刻是依 gongliao 的水位挑的。\n"
)
w("## 3. 大潮組與小潮組的 25 h 潮汐強度\n")
w(
    "| 地點 | 大潮 24 個平均（m） | 小潮 24 個平均（m） | 比值（大潮/小潮） | 大潮最小值（m） | 小潮最大值（m） | 大潮最小是否 ≥ 本季中位數／小潮最大是否 < 本季中位數 |"
)
w("|---|---|---|---|---|---|---|")
for site in ac.SITES:
    rows = [r for r in R[site]["rows"] if r["tide_class"] != "event"]
    sp = [r for r in rows if r["tide_class"] == "spring_proxy"]
    nn = [r for r in rows if r["tide_class"] == "neap_proxy"]
    a = np.mean([r["tidal_strength_m"] for r in sp])
    b = np.mean([r["tidal_strength_m"] for r in nn])
    ok_sp = sum(r["class_rule_consistent"] is True for r in sp)
    ok_nn = sum(r["class_rule_consistent"] is True for r in nn)
    w(
        f"| {site} | {a:.3f} | {b:.3f} | {a / b:.2f} | {min(r['tidal_strength_m'] for r in sp):.3f} | {max(r['tidal_strength_m'] for r in nn):.3f} | 大潮 {ok_sp}/24、小潮 {ok_nn}/24 |"
    )
w(
    "注意：各站『大潮最小值』可低於『小潮最大值』，是因為大小潮分界是各季自己的中位數（季節間強度水準不同），不是全年單一門檻。\n"
)
w("## 4. 各月分層時刻數（48 個，不含 2 個事件）\n")
w("| 地點 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 最小 | 最大 | 落在 5–9 的月數 |")
w("|---|" + "---|" * 15)
for site in ac.SITES:
    mc = collections.Counter(r["month"] for r in R[site]["rows"] if r["tide_class"] != "event")
    v = [mc.get(m, 0) for m in range(1, 13)]
    w(
        f"| {site} | "
        + " | ".join(map(str, v))
        + f" | {min(v)} | {max(v)} | {sum(5 <= x <= 9 for x in v)}/12 |"
    )
w("")
w(
    "注意：48 個時刻平均每月 4 個，『每月 5–9 個』在數學上不可能成立；選擇規則只在『季 × 大小潮 × 潮相』24 格內各取 2 個，沒有每月配額。"
)
w(
    "由於各格取的是極值（上升／下降最快出現在大潮強度最高的時段），同一格的 2 個時刻常落在同一個大潮週期內，所以月分布不均（例如 hsinchu 7 月 0 個、6 月 10 個）。報告若寫『涵蓋全年各月』需改寫為『涵蓋四季』。\n"
)
w("## 5. 事件時刻（高浪、強流）\n")
w(
    "兩個事件取『其餘 2025 有效時刻』中 Hs／流速的年最大值。百分位為該序列 2025 年有效時刻中小於等於該值的比例。\n"
)
w("| 地點 | 事件 | UTC | 值 | 2025 年最大 | 2025 年 p99 | 百分位 | 是年最大 |")
w("|---|---|---|---|---|---|---|---|")
for site in ac.SITES:
    for r in R[site]["rows"]:
        if r["tide_class"] == "event":
            nm = "高浪（Hs, m）" if r["phase_or_event"] == "high_wave_event" else "強流（m/s）"
            w(
                f"| {site} | {nm} | {r['utc']} | {r['event_value']:.3f} | {r['event_year_max']:.3f} | {r['event_year_p99']:.3f} | {r['event_percentile_rank']:.2f} | {'是' if abs(r['event_value'] - r['event_year_max']) < 1e-9 else '否'} |"
            )
w("")
w(
    "gongliao、hsinchu、nanwan、lienchiang 的 8 個事件皆為各自序列 2025 年的最大值（百分位 100，且高於 p99）；guishan 的 2 個事件是沿用 gongliao 的時刻，在 guishan 自己的序列上 Hs 為第 98.7 百分位（2.84 m，低於該年 p99 的 2.93 m、年最大 4.15 m）、流速為第 64 百分位（0.27 m/s，年最大 1.38 m/s），**不是** guishan 的極端事件。\n"
)
w("## 6. 錨點與取樣點\n")
w("| 地點 | 設定錨點（lon, lat） | 波浪取樣點（lon, lat） | 取樣類型 |")
w("|---|---|---|---|")
kinds = {}
arr = json.load(open(ac.INP / "common-input/arrival.json"))
for r in arr["records"]:
    kinds.setdefault(r["study_site_id"], r["metadata"].get("metric_location_kind"))
for site in ac.SITES:
    a = S[site].anchor
    m = S[site].metric
    w(f"| {site} | {a[0]:.5f}, {a[1]:.5f} | {m[0]:.5f}, {m[1]:.5f} | {kinds.get(site)} |")
w("")
w(
    "guishan 的設定錨點已由原點（121.951606, 24.843127；落在龜山島陸地）吸附到持續濕網格（121.939148, 24.850326），位移 1490.5 m（見 common-config 的 `anchor_snap_provenance`）；本核對的 guishan 序列取自吸附後錨點。gongliao 與 nanwan 的波浪取樣點是 NWW3 格心，與海流錨點分別相距約 1.4 km 與 3.0 km（水位與流速仍取錨點最近格點）。\n"
)
w("## 7. 其他資料註記\n")
for site in ac.SITES:
    s = S[site]
    ym = s.year_of == ac.YEAR
    w(
        f"- {site}：2025 年 8,760 小時中有效時刻 {int((s.valid & ym).sum())}；Hs 有限值 {int(np.isfinite(s.hs[ym]).sum())}；水位有限值 {int(np.isfinite(s.eta[ym]).sum())}。"
    )
w(
    "- 高浪事件放大圖（gongliao、hsinchu、nanwan）在事件前約 8–11 小時，Hs 於相鄰一小時內階躍（gongliao 2.59 → 4.66 m、hsinchu 4.73 → 5.80 m、nanwan 2.04 → 5.71 m），為 NWW3 輸入序列本身的特徵（可能為預報週期銜接），本核對未修改；圖中如實呈現。"
)
w("- 水位曲線在少數放大格呈鋸齒（來源 OCM 逐時資料的高頻擾動），不影響標籤判定。\n")
w("## 8. 龜山島（guishan）共用貢寮時刻的結論\n")
gl = {m["observation_time_utc_ns"]: m for m in recs["gongliao"]}
gs = {m["observation_time_utc_ns"]: m for m in recs["guishan"]}
same_utc = set(gl) == set(gs)
same_lab = sum(
    1
    for t in gs
    if t in gl
    and all(
        gs[t][k] == gl[t][k]
        for k in ("observation_tide_class", "observation_phase_or_event", "observation_season")
    )
)
w(
    f"- guishan 與 gongliao 的 50 個觀測 UTC 集合完全相同：{'是' if same_utc else '否'}；季節／大小潮／潮相標籤逐筆相同：{same_lab}/50（標籤是 gongliao 序列的標籤，manifest 標示 `gongliao_paired_utc_reference_v1`、`shared_A_forcing_utc`）。"
)
g = R["guishan"]
rows = g["rows"]
c48 = [r for r in rows if r["tide_class"] != "event"]
n_cls = sum(r["class_rule_consistent"] is True for r in c48)
n_sign = sum(r["sign_consistent"] is True for r in c48)
sl = [r["hours_to_nearest_extreme"] for r in c48 if r["phase_or_event"] == "slack_proxy"]
w(
    "- manifest 內 guishan 的 `elevation_m`、`tidal_strength_proxy_m`、事件 Hs／流速等數值是 guishan 自己序列在該 UTC 的值（重現差 0.0）。"
)
w(
    f"- 把這些 gongliao 標籤放到 guishan 自己的序列上檢核：大小潮標籤與『guishan 本季中位數規則』一致者 {n_cls}/48；上升／下降最快的潮相符號正確 {n_sign}/48；近平潮距高低潮中位數 {np.median(sl):.2f} h、最大 {max(sl):.2f} h。"
)
cr = g["cmp_ref"]
w(
    f"- 以 gongliao 序列（挑選依據）重算：分層格 {cr['exact']}/24 完全相同。以 guishan 自己的序列重跑選擇器：時刻集合與 manifest 不同，標籤相符僅 10/50；獨立重算的分層格完全相同 {g['cmp_own']['exact']}/24（集合相同 {g['cmp_own']['set']}/24）。"
)
sp = np.mean([r["tidal_strength_m"] for r in c48 if r["tide_class"] == "spring_proxy"])
nn = np.mean([r["tidal_strength_m"] for r in c48 if r["tide_class"] == "neap_proxy"])
c = np.corrcoef(
    S["guishan"].strength[np.isfinite(S["guishan"].strength) & np.isfinite(S["gongliao"].strength)],
    S["gongliao"].strength[np.isfinite(S["guishan"].strength) & np.isfinite(S["gongliao"].strength)],
)[0, 1]
w(
    f"- 兩站 25 h 潮汐強度時間序列相關係數 {c:.2f}；guishan 在這些時刻的大潮組／小潮組平均強度比 {sp / nn:.2f}（仍可區分，但小於 gongliao 的 {1.53:.2f}）。"
)
w(
    "- 結論：guishan 的 50 個時刻『屬於貢寮序列的分層』，在 guishan 自己的水位上大致仍落在大小潮週期的相近位置（符號 48/48 正確），但大小潮標籤只有約六成與 guishan 自己的季內中位數規則一致，近平潮最多偏離高低潮 1.6 h，事件時刻在 guishan 並非最大波高／最強流。報告若要說『guishan 的 50 個時刻本身是依其潮汐分層挑選』不成立，宜寫成『沿用貢寮的配對時刻，以利兩地點同時刻比較』。\n"
)
w("## 9. 與預期不符之處（如實列出）\n")
w("1. 每月 5–9 個：不可能且實際不成立（見第 4 節）。")
w("2. nanwan 一個近平潮時刻為駐潮點，距高低潮 7.45 h。")
w("3. guishan 的標籤與時刻是貢寮的（見第 8 節）；其事件不是 guishan 的年極值。")
w(
    "4. 大小潮標籤的物理意義：『大潮／小潮』是以 25 h 滾動 RMS 水位（`proxy`）相對季內中位數分組，並非天文朔望；本核對沒有也不需要天文朔望資料，報告宜稱『大潮代理／小潮代理』或說明定義。"
)
(OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
