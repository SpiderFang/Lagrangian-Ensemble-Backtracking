# ruff: noqa
"""WP-61 追加診斷：上升最快／下降最快時刻是否挑到 OCM 海面高程的跳動。只讀，不改正式資料。"""
import sys, collections, json
sys.path.insert(0, "/home/mustlab/Workspace/lbt-repo-wp61/scripts/wp61")
import numpy as np
import arrival_check as ac

HOUR = ac.HOUR_NS
PER = (12.4206012, 12.0, 23.9344697, 25.8193417)  # M2 S2 K1 O1
SITES = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
R_THR, S_FACTOR = 5.0, 1.5

def fmt(t): return str(np.datetime64(int(t), "ns"))[:16]

def second_diff_abs(eta):
    S = np.full(eta.size, np.nan); S[1:-1] = np.abs(eta[2:] - 2 * eta[1:-1] + eta[:-2]); return S

def local_ratio(S, half=12):
    n = S.size; R = np.full(n, np.nan)
    for k in range(half + 1, n - half - 1):
        w = np.concatenate([S[k - half:k], S[k + 1:k + half + 1]]); m = np.nanmedian(w)
        R[k] = S[k] / m if m > 0 else np.nan
    return R

def find_glitches(s):
    eta = s.eta.astype(float); S = second_diff_abs(eta); R = local_ratio(S)
    y = (s.year_of == 2025)
    p95 = np.nanpercentile(S[y], 95)
    g = y & (R >= R_THR) & (S >= S_FACTOR * p95)
    g4 = y & (R >= 4.0) & (S >= S_FACTOR * p95)
    return g, g4, S, R, p95

def design(t):
    cols = [np.ones_like(t)]; dcols = [np.zeros_like(t)]
    for p in PER:
        w = 2 * np.pi / p
        cols += [np.cos(w * t), np.sin(w * t)]; dcols += [-w * np.sin(w * t), w * np.cos(w * t)]
    return np.stack(cols, 1), np.stack(dcols, 1)

def harmonic(s, i, gmask, half=25):
    """以 i 前後各 25 h 做 M2、S2、K1、O1（含常數）最小平方擬合；擬合時排除 i±1 與視窗內其他跳動時刻±1。"""
    n = s.eta.size
    lo, hi = max(i - half, 0), min(i + half + 1, n)
    t = (s.times[lo:hi] - s.times[i]).astype(float) / HOUR
    y = s.eta[lo:hi].astype(float)
    bad = np.zeros(t.size, bool)
    for j in np.flatnonzero(gmask[lo:hi]):
        bad[max(j - 1, 0):j + 2] = True
    keep = np.isfinite(y) & (np.abs(t) > 1) & ~bad
    A, dA = design(t)
    coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=1e-8)
    c = i - lo
    return float(A[c] @ coef), float(dA[c] @ coef), int(keep.sum())

def ratio_a(eta, i, half=12):
    D = np.full(eta.size, np.nan); D[1:] = np.abs(np.diff(eta))
    here = max(D[i], D[i + 1]); idx = [k for k in range(i - half, i + half + 2) if k not in (i, i + 1)]
    return here / np.nanmedian(D[idx])

def select(s, deriv, strength, cand):
    """與選擇器相同規則（季內中位數分大小潮、各潮相前兩名、全域不重複）；cand 為可選時刻遮罩。回傳 {(season,cls,phase): [index,...]}。"""
    selected, picks = [], {}
    for season in ac.SEASONS:
        idx = np.flatnonzero(s.valid & (s.year_of == 2025) & (s.season_of == season) & np.isfinite(strength))
        thr = float(np.median(strength[idx]))
        groups = {"spring_proxy": idx[strength[idx] >= thr], "neap_proxy": idx[strength[idx] < thr]}
        for cls in ac.CLASSES:
            for phase, key in (("fastest_rising", lambda i: -deriv[i]), ("fastest_falling", lambda i: deriv[i]), ("slack_proxy", lambda i: abs(deriv[i]))):
                c = [int(i) for i in groups[cls] if cand[i] and int(i) not in selected and np.isfinite(deriv[i])]
                c.sort(key=lambda i: (key(i), int(s.times[i])))
                picks[(season, cls, phase)] = c[:2]; selected += c[:2]
    return picks

def main():
    recs = ac.load_manifest_records()
    S_ = {k: ac.load_series(k) for k in SITES}
    G = {k: find_glitches(S_[k]) for k in SITES}
    # 修復時間戳的日檔（metadata：raw_time_axis_zero_kept）
    repaired_days = {"2025-02-10", "2025-02-20", "2025-06-07", "2025-10-29", "2025-11-18", "2025-12-30", "2025-12-31"}
    L = []
    w = L.append
    # ── 整年統計 ──
    w("## 整年跳動統計（2025，二階差分異常：|η[k+1]-2η[k]+η[k-1]| 為前後各 12 h 中位數的 ≥5 倍，且 ≥ 該站 p95 的 1.5 倍）\n")
    w("| 地點 | 跳動時刻數（≥5 倍） | 放寬為 ≥4 倍 | 事件數（相距 ≤3 h 合併） | UTC 時分布（時：次數） |")
    w("|---|---|---|---|---|")
    events = {}
    for k in SITES:
        s = S_[k]; g, g4, S, R, p95 = G[k]
        idx = np.flatnonzero(g)
        hrs = collections.Counter(int((s.times[i] // HOUR) % 24) for i in idx)
        ev = []
        for i in idx:
            if ev and i - ev[-1][-1] <= 3: ev[-1].append(i)
            else: ev.append([i])
        events[k] = ev
        w(f"| {k} | {len(idx)} | {int(g4.sum())} | {len(ev)} | " + "、".join(f"{h:02d}Z:{n}" for h, n in sorted(hrs.items())) + " |")
    w("")
    w("各站事件日期（UTC，事件首個跳動時刻；括號為該事件最大 |二階差分| m）：\n")
    for k in SITES:
        s = S_[k]; S = G[k][2]
        w(f"- {k}：" + "；".join(f"{fmt(s.times[e[0]])[5:]}({max(S[i] for i in e):.2f})" for e in events[k]))
    w("")
    # 事件跨站對照
    dates = collections.defaultdict(dict)
    for k in SITES:
        for e in events[k]:
            dates[fmt(S_[k].times[e[0]])[:10]][k] = fmt(S_[k].times[e[0]])[11:13]
    w("跨站對照（日期：站→首個跳動的 UTC 時）：\n")
    for d in sorted(dates): w(f"- {d}：" + "、".join(f"{k} {h}Z" for k, h in dates[d].items()))
    w("")
    # ── 逐站 32 個時刻 ──
    w("## 32 個『上升最快／下降最快』時刻逐項檢核（龜山島＝貢寮序列與貢寮相同，不另列）\n")
    base_changes = {}
    sus_all = {}
    for site in ("gongliao", "hsinchu", "nanwan", "lienchiang"):
        s = S_[site]; r = ac.analyze_site(site, recs[site], s)
        gm = G[site][0]
        glitch_idx = np.flatnonzero(gm)
        rows = [x for x in r["rows"] if x["phase_or_event"] in ("fastest_rising", "fastest_falling")]
        rows.sort(key=lambda x: (ac.SEASONS.index(x["season"]), x["tide_class"], x["phase_or_event"], x["replicate_rank_manifest"]))
        out = []
        for x in rows:
            i = r["idx_of"][int(np.datetime64(x["utc"].replace("Z", ""), "ns").astype("int64"))]
            ra = ratio_a(s.eta.astype(float), i)
            fe, fd, nk = harmonic(s, i, gm)
            act = (s.eta[i + 1] - s.eta[i - 1]) / 2.0
            res = (act - fd) * 1000 / 60
            near = [int(j) for j in glitch_idx if abs(j - i) <= 1]
            within12 = [int(j) for j in glitch_idx if abs(j - i) <= 12 and abs(j - i) > 1]
            hour = int((s.times[i] // HOUR) % 24)
            seam = hour in (0, 1)
            day = fmt(s.times[i])[:10]
            sus = bool(near) or abs(res) >= 3.0
            out.append(dict(season=x["season"], cls=x["tide_class"], phase=x["phase_or_event"], rank=x["replicate_rank_manifest"], utc=fmt(s.times[i]) + "Z", ratio=ra, act=act * 1000 / 60,
                            fit=fd * 1000 / 60, res=res, eres=float(s.eta[i] - fe), hour=hour, seam=seam, near=bool(near), within12=bool(within12), rep=day in repaired_days, sus=sus, idx=i))
        sus_all[site] = out
        w(f"### {site}\n")
        w("| 季 | 潮況 | 潮相 | 名次 | UTC | |Δη|比值 | 實際 dη/dt (mm/min) | 調和擬合 dη/dt | 殘差 | η殘差 (m) | 接縫（UTC 0/1 時） | 距跳動 ≤1 h | 窗內(2–12 h)有跳動 | 修復日檔 |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for o in out:
            mark = "**" if o["sus"] else ""
            w(f"| {mark}{o['season']}{mark} | {o['cls'][:3]} | {o['phase'][8:]} | {o['rank']} | {mark}{o['utc']}{mark} | {o['ratio']:.2f} | {o['act']:+.1f} | {o['fit']:+.1f} | {o['res']:+.1f} | {o['eres']:+.2f} | {'是' if o['seam'] else ''} | {'是' if o['near'] else ''} | {'是' if o['within12'] else ''} | {'是' if o['rep'] else ''} |")
        w("")
    # 疑似清單
    w("## 疑似挑到跳動的時刻清單（判準：距跳動時刻 ≤1 h，或調和殘差 |res| ≥ 3 mm/min）\n")
    w("| 地點 | 季 | 潮況 | 潮相 | 名次 | UTC | |Δη|比值 | 調和殘差 (mm/min) | η殘差 (m) |")
    w("|---|---|---|---|---|---|---|---|---|")
    cnt = {}
    for site, out in sus_all.items():
        c = 0
        for o in out:
            if o["sus"]:
                c += 1
                w(f"| {site} | {o['season']} | {o['cls'][:3]} | {o['phase'][8:]} | {o['rank']} | {o['utc']} | {o['ratio']:.2f} | {o['res']:+.1f} | {o['eres']:+.2f} |")
        cnt[site] = c
    w("")
    w("疑似個數（32 個中）：" + "、".join(f"{k} {v}" for k, v in cnt.items()) + "；guishan 與 gongliao 相同（沿用同 50 個 UTC）。")
    w("僅 |Δη| 比值 ≥3（無跳動、殘差小）的時刻： " + "、".join(f"{site} {o['utc']}({o['ratio']:.1f})" for site, out in sus_all.items() for o in out if o["ratio"] >= 3 and not o["sus"]) + "\n")
    # 一天中各 UTC 時的異常集中度
    w("## 異常的 UTC 時分布（2025 全年，二階差分為前後 12 h 中位數 ≥3 倍的時刻數，依 UTC 時）\n")
    w("| 地點 | 00Z | 01Z | 02Z | 03Z | 04Z | 05Z | 06–23Z 每小時平均 | 00–01Z 占全部 ≥3 倍時刻的比例 | 00Z／01Z 的|二階差分|中位數相對全日中位數 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k in SITES:
        s = S_[k]; g, g4, S, R, p95 = G[k]; y = (s.year_of == 2025)
        hr = ((s.times // HOUR) % 24).astype(int)
        m3 = y & (R >= 3.0)
        c = [int((m3 & (hr == h)).sum()) for h in range(24)]
        med_all = np.nanmedian(S[y]); med0 = np.nanmedian(S[y & (hr == 0)]); med1 = np.nanmedian(S[y & (hr == 1)])
        w(f"| {k} | " + " | ".join(str(c[h]) for h in range(6)) + f" | {np.mean(c[6:]):.1f} | {(c[0]+c[1])/max(sum(c),1):.0%}（{c[0]+c[1]}/{sum(c)}） | {med0/med_all:.2f}／{med1/med_all:.2f} |")
    w("")
    # 重選
    w("## 排除跳動後重選（只算，不改正式資料）\n")
    w("規則同選擇器（季內中位數分大小潮；各潮相前兩名；全域不重複），強度與季內中位數沿用原值，只改『可選時刻』：")
    w("- V1：排除距『≥5 倍跳動時刻』≤1 h 的時刻；V1b：同，但跳動門檻放寬為 ≥4 倍。")
    w("- V2：排除『實際 dη/dt 與調和擬合 dη/dt 之差 ≥3 mm/min』的時刻（擬合遮掉跳動±1 h）。")
    w("- V3：排除『|Δη| 比值（該時刻前後兩個逐時差較大者／前後 12 h 逐時差中位數）≥3』的時刻；V3b：門檻 ≥4。\n")
    w("表中數字＝正式 48 個分層時刻中被替換掉的個數（上升/下降最快 32＋近平潮 16）。基線（不排除）重算皆與 manifest 相同。\n")
    w("| 地點 | V1 | V1b | V2 | V3（≥3） | V3b（≥4） | 被 V1 替換的時刻 |")
    w("|---|---|---|---|---|---|---|")
    for site in SITES:
        base_site = "gongliao" if site == "guishan" else site
        s = S_[base_site]; gm = G[base_site][0]; gm4 = G[base_site][1]
        r = ac.analyze_site(base_site, recs[base_site], s)
        deriv = s.deriv.astype(float); strength = s.strength.astype(float)
        man = {}
        for x in r["rows"]:
            if x["tide_class"] != "event":
                man.setdefault((x["season"], x["tide_class"], x["phase_or_event"]), []).append((x["replicate_rank_manifest"], r["idx_of"][int(np.datetime64(x["utc"].replace("Z", ""), "ns").astype("int64"))]))
        manp = {k: [i for _, i in sorted(v)] for k, v in man.items()}
        n = s.eta.size
        base = select(s, deriv, strength, np.ones(n, bool))
        okb = all(base[k] == manp[k] for k in manp)
        def mask_near(gmask):
            c = np.ones(n, bool)
            for g in np.flatnonzero(gmask): c[max(g - 1, 0):g + 2] = False
            return c
        c1 = mask_near(gm); c1b = mask_near(gm4 | gm)
        yidx = np.flatnonzero((s.year_of == 2025) & s.valid)
        res = np.full(n, np.nan); ra = np.full(n, np.nan)
        eta = s.eta.astype(float)
        D = np.full(n, np.nan); D[1:] = np.abs(np.diff(eta))
        for i in yidx:
            i = int(i)
            fd = harmonic(s, i, gm)[1]
            res[i] = abs((eta[i + 1] - eta[i - 1]) / 2.0 - fd) * 1000 / 60 if i + 1 < n else np.nan
            if i + 13 < n:
                here = max(D[i], D[i + 1]); idx = [k for k in range(i - 12, i + 14) if k not in (i, i + 1)]
                ra[i] = here / np.nanmedian(D[idx])
        c2 = ~(res >= 3.0); c3 = ~(ra >= 3.0); c3b = ~(ra >= 4.0)
        for cc in (c2, c3, c3b): cc[~np.isfinite(res)] = cc[~np.isfinite(res)]
        def changed(cand):
            new = select(s, deriv, strength, cand)
            rf = sl = 0; lst = []
            for k, v in manp.items():
                miss = set(v) - set(new[k])
                if k[2] == "slack_proxy": sl += len(miss)
                else: rf += len(miss)
                lst += [(k, i) for i in miss]
            return rf, sl, lst
        out = []
        for cand in (c1, c1b, c2, c3, c3b):
            rf, sl, lst = changed(cand); out.append(f"{rf+sl}/48（上升下降 {rf}、近平潮 {sl}）"); 
            if cand is c1: l1 = lst
        w(f"| {site}{'（用貢寮序列）' if site == 'guishan' else ''} | " + " | ".join(out) + " | " + "；".join(f"{k[0]} {k[1][:3]} {k[2][8:12]} {fmt(s.times[i])[5:]}" for k, i in l1) + " |")
    w("")
    # 龜山島自己序列：同樣 32 個時刻是否近跳動
    sg = S_["guishan"]; gg = G["guishan"][0]; ggi = np.flatnonzero(gg)
    r0 = ac.analyze_site("gongliao", recs["gongliao"], S_["gongliao"])
    n_near = 0; lst = []
    for x in r0["rows"]:
        if x["phase_or_event"] in ("fastest_rising", "fastest_falling"):
            i = r0["idx_of"][int(np.datetime64(x["utc"].replace("Z", ""), "ns").astype("int64"))]
            fe, fd, _ = harmonic(sg, i, gg); res = ((sg.eta[i + 1] - sg.eta[i - 1]) / 2.0 - fd) * 1000 / 60
            if any(abs(j - i) <= 1 for j in ggi) or abs(res) >= 3.0: n_near += 1; lst.append(fmt(sg.times[i]))
    w(f"龜山島自己的序列在同 32 個時刻上：距其跳動 ≤1 h 或調和殘差 ≥3 mm/min 者 {n_near}/32 {lst}。\n")
    # 連江該格 ±13 h
    sl_ = S_["lienchiang"]
    i = int(np.flatnonzero(sl_.times == np.datetime64("2025-12-30T00:00", "ns").astype("int64"))[0])
    w("## 連江放大圖第 2 列第 3 格（冬季小潮、退潮最快第 1 名，2025-12-30T00:00Z）±13 h 逐時 η\n")
    w("| 相對時（h） | UTC | η (m) | 逐時差 η[k]-η[k-1] (m) |")
    w("|---|---|---|---|")
    for k in range(-13, 14):
        j = i + k
        w(f"| {k:+d} | {fmt(sl_.times[j])}Z | {sl_.eta[j]:+.4f} | {sl_.eta[j]-sl_.eta[j-1]:+.4f} |")
    w("")
    print("\n".join(L))
    open("/home/mustlab/Workspace/lbt-wp61/diag-jump.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

main()
