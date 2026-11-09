# 十類海廢沉降速度：兩條獨立路線的完整推導與查證

本文件為報告用之完整方法論。十個沉降速度由兩條**互相獨立**的路線建立：路線 A 為受力平衡理論推導，路線 B 為文獻實測對照。兩條路線的原始材料、程式與逐筆中間結果均可重算。

---

## 0. 採用值

| iOcean 類別 | 採用值 (m/s) | 導出區間 w_min–w_max (m/s) |
|---|---:|---|
| 其他／不可回收 | −0.0142 | 0.00090 – 0.1077 |
| 保麗龍 | −0.0359 | 0.00527 – 0.1501 |
| 廢紙 | −0.0384 | 0.00661 – 0.1624 |
| 廢漁網漁具 | −0.0434 | 0.00795 – 0.2234 |
| 竹木 | −0.0684 | 0.01350 – 0.2312 |
| 寶特瓶 | −0.0782 | 0.05630 – 0.1335 |
| 其他／可回收 | −0.128 | 0.04300 – 0.3447 |
| 鋁罐 | −0.132 | 0.10787 – 0.1753 |
| 鐵罐 | −0.301 | 0.23780 – 0.4274 |
| 玻璃瓶 | −0.472 | 0.19542 – 0.9204 |

垂直座標向上為正，故十項均為負值。

**取值規則（僅此一條）**：採用值 = 該類全部 6 個解（3 種尺寸情境 × 阻力係數上下界）之**幾何平均**，取三位有效數字。此即 `derive.py` 執行時「十類彙總表」所印出之 `w_repr` 欄位，無任何人工挑選或格點對齊。

---

## 1. 為什麼用兩條路線

計畫書未提供速度數值，現地實測團隊亦未提供資料。單一來源無法自我檢核，因此本案同時建立兩條獨立路線：

- **路線 A（理論）**：由物件的尺寸、材質密度與形狀，以受力平衡解出終端沉降速度。優點是十類皆可涵蓋、可重算；缺點是未經本案樣本驗證。
- **路線 B（實測）**：蒐集同材質或同形態物件的沉降槽／迴流水槽實測值。優點是真實量測；缺點是公分級完整海廢物件的實測在全球文獻中大量缺漏。

兩條路線在有重疊的類別上互相驗證；在路線 B 空白的類別上，明確標示僅有理論依據。

---

## 2. 路線 A：受力平衡理論推導

### 2.0 本方法在文獻中的地位

本節先回答一個必然會被提出的問題：**這套推導是既有文獻的標準作法，還是本案自創？**

**(a) 以受力平衡加阻力係數求終端沉降速度，是沉降研究的標準方法，且已被直接應用於塑膠與海廢顆粒。**

最直接的對應是 **Goral et al. (2023)**（*Environmental Research* 228:115783）。該研究以沉降槽實測 1,400 餘顆規則與不規則形狀微塑膠，其方法與本案逐項相同：

| 方法元件 | Goral et al. (2023) | 本案 |
|---|---|---|
| 控制方程式 | `C_d = 2g(s−1)V̄ / (w_s² A)` | `V g (ρ_p − ρ_w) = ½ ρ_w C_d w² A_proj`（同式整理） |
| 分形狀定義投影面積 | 球、圓柱／纖維、圓盤／片狀、不規則碎片各有其 `A` 與特徵長度（其 Table 4） | 薄板、圓柱、鈍體各有其 `A_proj` 與 `d_A`（本文 2.3） |
| 阻力關聯式 | `C_d = max[24/Re(1+0.15 Re^0.687), 0.44]` | `C_d = 24/Re(1+0.15 Re^0.687) + C_d,∞` |

換言之，本案的控制方程式、分形狀處理與阻力關聯式，都與一篇 2023 年經同儕審查、且以實測驗證過的塑膠沉降研究一致。

其他採用同一途徑的研究：

- **Khatmullina & Isachenko (2017)** —— 規則形狀微塑膠的沉降速度，建立形狀相關的阻力關係。
- **Francalanci et al. (2021)**（*Environmental Pollution* 290:118068）—— 由代表粒徑、密度與 Corey 形狀因子**預測**塑膠顆粒沉降速度，涵蓋 3D（顆粒）、2D（碎片、圓盤）、1D（纖維）三種形態，方法延伸自 Ferguson & Church (2004)。
- **Chubarenko et al. (2016)** —— 依球體、纖維、薄膜三種形態分別處理海洋微塑膠的動力性質。
- **Waldschläger & Schüttrumpf (2019)** —— 由顆粒物理性質建立沉降與上升速度公式。
- **Dellino et al. (2005)** —— 形狀對終端速度影響的原始方法（火山碎屑），為上述塑膠研究共同引用的方法源頭之一。

**(b) 在 Lagrangian 海漂／海廢輸送模式中，「由物件物理性質計算沉速」本來就是標準選項，而非退而求其次的替代方案。**

廣泛使用的 **TrackMPD** 海洋塑膠追蹤模式（Jalón-Rojas et al. 2019；v3.0, 2025）明載：粒子的沉降或上升速度「可由使用者指定，或依據粒子物理性質以多種經驗式線上計算」，其內建之計算式包括 Dellino et al. (2005)、Waldschläger & Schüttrumpf (2019) 與 Jalón-Rojas et al. (2022)。本案與之屬同一類作法。

**(c) 哪些是文獻方法，哪些是本研究的設定——明確切分。**

| 元件 | 來源 |
|---|---|
| 受力平衡控制方程式 | 標準流體力學；Goral et al. (2023) 用於塑膠 |
| Schiller–Naumann 阻力關聯式 | Schiller & Naumann (1935)；Goral et al. (2023) 用於塑膠 |
| `C_d,∞` 形態常數（薄板／圓柱／鈍體） | 標準阻力係數表 |
| 分形狀之投影面積與特徵長度 | Goral et al. (2023) Table 4；Chubarenko et al. (2016) |
| 薄壁中空容器有效密度 | 質量守恆的直接推算，無自由參數（本文 2.4） |
| 求解器正確性 | 對 Clift & Gauvin (1970) 標準球體阻力曲線驗證（本文 2.6） |
| 30 組物件模板的參數 | **分三級，見 2.7.1** |

模板參數不是單一來源，其中包含可引用的標準物性、可查證的市售規格，以及本研究自行設定的代表性假設三種層級，逐項分級列於 2.7.1。**全部 30 組逐筆明列於 `scenario_parameters.md` 供檢視與替換**；任何人不同意某一組模板，可直接修改該列參數重算，其餘方法不受影響。

**(e) 為何不直接套用現成的塑膠沉降速度經驗公式。**

文獻中已有多套由物件性質預測塑膠沉降速度的經驗公式（Waldschläger & Schüttrumpf 2019、Francalanci et al. 2021、Yu et al. 2022 等，部分並已內建於 TrackMPD）。本案未採用這些公式直接產生十個數值，原因是**適用尺度**：

這些公式全部以**微塑膠**實驗資料擬合，而微塑膠的定義上限為 5 mm；本案十類海廢的代表物件為 50–300 mm，超出其擬合範圍 10 至 60 倍。將擬合型經驗式外推至校準範圍之外，所得數值不具效力——而這正是舊設定值的錯誤型態（把微塑膠尺度的速度階梯套用到巨觀海廢）。

相對地，受力平衡式本身不依賴任何擬合範圍，唯一有適用界線的是阻力關聯式，而該界線**已明確量化並揭露**（2.6：`Re ≲ 1000` 內誤差 1.5% 以內，超出後偏高且方向保守）。這是在本案尺度上唯一站得住的選擇。

Ferguson & Church (2004) 在本文中僅用於 2.6 的求解器驗證（毫米尺度，該式有效範圍內），不用於產生十類速度。

**(d) 本案與文獻作法的差異，以及差異的方向。**

Goral et al. (2023) 的高雷諾數極限取球體值 0.44；本案改取**形態相關**的 `C_d,∞`（薄板 1.1–2.0、圓柱 0.9–1.3、鈍體 0.6–1.2）。對片狀與柱狀物件而言，本案給的阻力**較高**、算出的沉速**較慢**，是較保守的方向。此外本案對每組情境同時給出 `C_d,∞` 上下界對應的速度區間，而非單一值。

### 2.1 控制方程式

物件在靜止均質水體中自由沉降，達終端速度時加速度為零，三力平衡：

```
重力  −  浮力  =  阻力
ρ_p V g  −  ρ_w V g  =  ½ ρ_w C_d w² A_proj
```

整理得

```
V g (ρ_p − ρ_w) = ½ ρ_w C_d w² A_proj
```

兩側同除 ρ_w，令**淹沒比重** `R ≡ ρ_p/ρ_w − 1`：

```
        ┌───────────────┐
        │      2 V g R  │
   w =  │ ─────────────  │ ^(1/2)
        │  C_d A_proj   │
        └───────────────┘
```

其中：`V` 為物件體積（m³，決定浮力與重力）、`A_proj` 為垂直於運動方向之投影面積（m²，決定阻力）、`C_d` 為阻力係數（無因次，與雷諾數及形狀有關）、`w` 為終端沉降速度（m/s）。

因 `C_d` 本身是 `w` 的函數（透過雷諾數），上式為隱式方程，須迭代求解（見 2.5）。

**常數**：海水密度 `ρ_w = 1025 kg/m³`、運動黏滯係數 `ν = 1.05×10⁻⁶ m²/s`、重力加速度 `g = 9.81 m/s²`。

### 2.2 阻力係數閉合式

採用 Schiller–Naumann (1935) 關聯式，並加上形態相關之高雷諾數極限項：

```
C_d(Re) = 24/Re · (1 + 0.15 Re^0.687) + C_d,∞        ,   Re = w · d_A / ν
```

- 第一項為 Schiller–Naumann 之標準球體阻力關聯式，原文適用範圍 `Re < 1000`。低雷諾數時退化為 Stokes 解 `24/Re`。
- 第二項 `C_d,∞` 為高雷諾數（Newton 區）之形態常數，取自標準阻力係數表。各形態取**上下界**分別代入，得到速度的下界與上界：

| 形態 | `C_d,∞` 範圍 | 物理意義 |
|---|---|---|
| 薄板 slab | 1.1 – 2.0 | 下界：穩定寬面下墜；上界：翻滾、飄盪（secondary motion）使有效阻力升高 |
| 圓柱 cyl | 0.9 – 1.3 | 橫流圓柱之典型次臨界值 |
| 鈍體 bluff | 0.6 – 1.2 | 下界：近球形緊實體；上界：不規則姿態 |

範圍並非數值誤差，而是**姿態不確定性**的物理涵蓋：自由沉降的片狀與柱狀物體普遍出現翻滾、飄盪與週期性擺動，實際阻力高於固定姿態值。取上界即代表此效應。

- `d_A` 為計算雷諾數所用之特徵長度，取與投影面積等面積之圓直徑：`d_A = √(4 A_proj / π)`（鈍體情境則直接取等效球徑 `D_eq`）。

### 2.3 三種幾何的處理

十類物件依實際形態分為三種幾何模型。**關鍵原則：薄片與扁平物件不得使用等效球徑法**，否則會把一張 30 公分薄膜折算成直徑數公分的球體，系統性高估沉速十餘倍。

**(a) 薄板 `geom_slab(L, W, t)`** — 寬面（L×W）迎流下墜

```
V = L · W · t          A_proj = L · W          d_A = √(4 A_proj / π)
```

代入控制方程式，`L·W` 完全消去：

```
       ┌──────────────┐
       │   2 g R t    │
  w =  │ ───────────  │ ^(1/2)              ← 薄板沉速只由「厚度」決定
       │     C_d      │
       └──────────────┘
```

此為可獨立驗證之推論，見 2.9。

**(b) 圓柱 `geom_cyl(d, L)`** — 軸向水平、側面迎流

```
V = (π/4) d² L          A_proj = d · L          d_A = √(4 A_proj / π)
```

**(c) 鈍體 `geom_bluff(V_env)`** — 以外包絡體積之等效球徑近似，僅適用於投影面積接近球體投影之**緊實**鈍體（完整罐、完整瓶、緻密碎片）

```
D_eq = (6 V_env / π)^(1/3)          A_proj = (π/4) D_eq²          V = V_env
```

**壓扁後的瓶罐一律改用薄板幾何。** 若誤以緊實鈍體計算，會因低估投影面積而高估沉速約一倍。此為本推導初版曾犯之錯誤，已修正並記錄於程式註解。

### 2.4 薄壁中空容器的有效密度

瓶、罐等薄壁中空容器**不直接假設密度**，而由殼體質量與進水體積推算。設殼體質量 `m_solid`、材質密度 `ρ_solid`、外包絡體積 `V_env`：

```
V_solid = m_solid / ρ_solid                    （殼體材料實佔體積）
V_water = V_env − V_solid                      （其餘視為完全進水）
ρ_eff   = (m_solid + V_water · ρ_w) / V_env
```

保護條件：若 `V_env < V_solid`（殼體材料塞不進該外包絡體積），此近似不成立，直接取 `ρ_solid`。

此步驟是本推導與原設計差異最大之處。以 600 mL 寶特瓶為例，PET 殼重 25 g、材質密度 1380 kg/m³，材料僅佔全瓶體積 2.9%，**完全進水後有效密度僅約 1035 kg/m³**，而非原設計假設的 1050–1150。同理，完整進水鐵罐之有效密度約 1138 kg/m³，而非 1300–3000。

### 2.5 數值解法

控制方程式為隱式（`C_d` 依賴 `w`），以**阻尼固定點迭代**求解：

```
令 num = 2 V g R / A_proj
初值 w⁰ = 1×10⁻⁴ m/s
重複：  Re  = w · d_A / ν
        C_d = 24/Re·(1 + 0.15 Re^0.687) + C_d,∞
        w*  = √(num / C_d)
        w   = 0.5 w + 0.5 w*            （阻尼係數 0.5，確保單調收斂）
直到   |w* − w| < 10⁻¹² · max(w*, 10⁻¹²)  或達 500 次上限
```

若 `R ≤ 0`（物件不重於海水）直接回傳 0，即該物件不下沉。全部 30 組情境均在數十次迭代內收斂。

### 2.6 求解器驗證

以**石英砂球體**（`ρ_p = 2650 kg/m³`、`C_d,∞ = 0`，即標準球體阻力）反算，與兩個可引用之公認關聯式即時比對，不使用任何寫死之期望值：

| 球徑 | Re | 本案解算器 (m/s) | Clift & Gauvin (1970) | 偏差 | Ferguson & Church (2004) | 偏差 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 mm | 0.7 | 0.0074 | 0.0074 | +0.0% | 0.0074 | −0.3% |
| 0.5 mm | 35.7 | 0.0749 | 0.0749 | +0.0% | 0.0903 | −17.1% |
| 1.0 mm | 142.1 | 0.1492 | 0.1488 | +0.2% | 0.1783 | −16.4% |
| 2.0 mm | 521.1 | 0.2736 | 0.2696 | +1.5% | 0.2933 | −6.7% |
| 5.0 mm | 2751.6 | 0.5778 | 0.5136 | +12.5% | 0.4968 | +16.3% |

**主要對照為 Clift & Gauvin (1970)** —— 該式為涵蓋 `Re < 3×10⁵` 之標準球體阻力曲線，與本案求解的是同一個物理量，因此是嚴格的求解器驗證：`Re ≲ 1000` 範圍內偏差在 **1.5% 以內**；`Re > 1000` 後偏高，5 mm 檔 +12.5%，恰為 Schiller–Naumann 關聯式已知的適用範圍上限（原文適用 `Re < 1000`），非程式邏輯錯誤。

**次要對照為 Ferguson & Church (2004)** —— 沉積物領域的獨立經驗式，與本案採用不同的建模途徑。其 ±17% 的差距反映的是各家經驗沉降公式之間的固有離散，而非本求解器誤差；列出用意在於呈現本結果落在既有經驗式的散布範圍內。

### 2.7 輸入參數

30 組情境（十類 × 小／中／大）的完整輸入參數——尺寸、體積、投影面積、密度、密度來源、`C_d,∞`、雷諾數範圍、速度範圍——逐列列於 `data/settling_velocity_derivation/scenario_parameters.md`，並與 `results.csv` 逐列核對一致。該表由程式機械產生，可重算。

### 2.7.1 模板參數的來源分級與可查證性

30 組模板的參數分為三級。這一節的用意，是讓審查者能逐項判斷哪些數字可以直接接受、哪些應該討論。

**第一級：標準物性常數（可引用，非本案設定）**

| 參數 | 數值 | 性質 |
|---|---|---|
| 海水密度 ρ_w | 1025 kg/m³ | 常溫海水標準值 |
| 運動黏滯係數 ν | 1.05×10⁻⁶ m²/s | 常溫海水標準值 |
| PET 密度 | 1380 kg/m³ | 聚合物標準物性 |
| 鋁密度 | 2700 kg/m³ | 金屬標準物性 |
| 鋼密度 | 7850 kg/m³ | 金屬標準物性 |
| 玻璃密度 | 2500 kg/m³ | 標準物性 |
| 尼龍（PA）密度 | 1140 kg/m³ | 聚合物標準物性（用於漁網 M 情境） |

**第二級：可查證之物件型態與市售規格**

各類的小／中／大三檔**物件型態**並非任意選取，而是對應主管機關《海漂底廢棄物調查指引》各類別所列舉的物件項目：

| 類別 | 調查指引所列物件項目 | 本案 S／M／L 對應 |
|---|---|---|
| 保麗龍 | 保麗龍碎片／保麗龍箱／廢棄浮筒、浮球 | 小塊發泡板／箱體殘片／大型浮具 |
| 廢漁網漁具 | 漁網、漁線／粗漁繩 | 單根網線／網片段／繩索 |
| 廢紙 | 紙／紙板／紙箱、利樂包 | 薄紙片／濕紙板碎片／飽和厚紙板 |
| 寶特瓶、玻璃瓶、鐵罐、鋁罐 | 各為單一物件項目 | 完整／半壓扁／嚴重壓扁三種狀態 |

中空容器的殼重與外包絡體積可對照市售規格查證：

| 參數 | 本案模板 | 可查證之實際範圍 | 本案位置與方向 |
|---|---|---|---|
| 330 mL 鋁罐殼重 | 13 g | 13–20 g（依製造商與設計） | 取**最輕端** → 沉速偏慢，保守 |
| 330 mL 鋁罐外包絡體積 | 390 cm³ | 標準罐徑 66 mm × 高 115 mm ≈ 393 cm³ | 相符 |
| 600 mL 寶特瓶殼重 | 25 g | 純水瓶約 12 g；含氣飲料瓶 13–53.6 g | 落於含氣飲料瓶區間中段；**為全部模板中唯一偏快方向之假設**，見下 |

**寶特瓶殼重之敏感度**：薄壁容器沉速 `w ∝ √m_solid`（見 2.9 推論二）。若改取純水瓶之 12 g，本類代表速度將由 0.0782 降至約 0.054 m/s（除以 √(25/12) ≈ 1.44）。本案取 25 g，是因臺灣海廢之寶特瓶以茶飲、運動飲料等厚壁瓶為大宗，而非純水瓶。**經檢視後決定維持 25 g**；此假設連同其影響量級一併揭露於此，若審查認為應改採較輕值，修改該列參數重算即可。

**第三級：本研究設定之代表性假設**

以下為本研究自行設定、無外部出處、亦未經本案現地樣本統計者：

1. **全部物件的尺寸數值**（各情境的 L×W×t、φd×L）。調查指引提供物件型態但不記錄尺寸，故尺寸取各該物件型態之常見量級。
2. **吸水、生物附著或混合材質後的有效密度**：保麗龍 1026／1060／1100、竹木 1030／1090／1150、廢紙 1050／1120／1200、其他不可回收 1030／1200／1400、其他可回收 1100／1200／1400 kg/m³。這些是「已轉為負浮力」狀態下的假設有效密度，取略高於海水至明顯高於海水的區間，用以涵蓋不同吸水與附著程度。

第三級參數的不確定性，正是各類 `w_min`–`w_max` 區間寬度（1.6×–120×）的主要來源；報告中以區間而非單點呈現，即為此故。

### 2.8 結果

| 類別 | w_min | w_repr（採用值） | w_max | 範圍倍數 |
|---|---:|---:|---:|---:|
| 其他／不可回收 | 0.00090 | **0.0142** | 0.10772 | 120× |
| 保麗龍 | 0.00527 | **0.0359** | 0.15006 | 28× |
| 廢紙 | 0.00661 | **0.0384** | 0.16236 | 25× |
| 廢漁網漁具 | 0.00795 | **0.0434** | 0.22341 | 28× |
| 竹木 | 0.01350 | **0.0684** | 0.23115 | 17× |
| 寶特瓶 | 0.05630 | **0.0782** | 0.13353 | 2.4× |
| 其他／可回收 | 0.04300 | **0.128** | 0.34474 | 8.0× |
| 鋁罐 | 0.10787 | **0.132** | 0.17528 | 1.6× |
| 鐵罐 | 0.23780 | **0.301** | 0.42735 | 1.8× |
| 玻璃瓶 | 0.19542 | **0.472** | 0.92040 | 4.7× |

**類別內的速度跨幅（1.6×–120×）大於類別間的跨幅（整體 43×）。** 換言之，同一類海廢因尺寸與破碎程度不同造成的速度差異，超過不同材質之間的差異。此為本推導最重要的結構性結果。

### 2.9 兩個可獨立手算驗證的物理推論

這兩點可讓審查者不必執行程式即可檢核推導的一致性。

**推論一：薄板的沉速只由厚度決定，與長寬無關。**

由 2.3(a)，`w = √(2 g R t / C_d)`，`L`、`W` 已完全消去。故一張 10×10 mm 與一張 300×300 mm 的同材質同厚度薄片，沉速相同——薄片的橫向尺寸只同時放大浮力與阻力，兩者互相抵銷。

以參數表 #3（保麗龍 L，100×100×20 mm，ρ_p = 1100）手算核對：`R = 1100/1025 − 1 = 0.07317`、`t = 0.020 m`，代入得 `w = √(0.028712 / C_d)`。程式輸出 `w_max = 0.1501 m/s`，反解 `C_d = 1.274`；而 `C_d,∞ = 1.1` 加上 `Re = 16127` 時的 Schiller–Naumann 項（0.175）為 1.275，吻合。此一手算同時驗證了薄板公式與阻力閉合式。

**推論二：把瓶罐壓得更扁，不改變其沉降速度。**

對薄壁中空容器，代入 2.4 的有效密度定義，超額質量為

```
(ρ_eff − ρ_w) V_env = m_solid − m_solid · ρ_w/ρ_solid = m_solid (1 − ρ_w/ρ_solid)
```

**此式不含厚度 `t`。** 代回薄板控制方程式：

```
       ┌────────────────────────────────┐
       │  2 g · m_solid (1 − ρ_w/ρ_solid)│
  w =  │ ───────────────────────────────  │ ^(1/2)
       │     ρ_w · A_proj · C_d          │
       └────────────────────────────────┘
```

沉速只取決於**殼體質量**與**投影面積**，與壓扁程度無關：壓得越扁，有效密度越高，但厚度等比例變薄，兩者恰好抵銷。

這解釋了參數表中 #17/#18（寶特瓶 M/L）、#23/#24（鋁罐 M/L）、#26/#27（鐵罐 M/L）三對情境的速度**完全相同**——那不是程式錯誤，而是物理必然。以 #23 半壓扁鋁罐手算核對：`m_solid = 0.013 kg`、`ρ_solid = 2700`、`A_proj = 6.0×10⁻³ m²`，代入上式得 `w = √(0.02573 / C_d)`；程式輸出 `w_hi = 0.1410 m/s` 對應 `C_d = 1.29`，與 `C_d,∞ = 1.1` 加上 `Re ≈ 1.2×10⁴` 時的 Schiller–Naumann 項（約 0.19）一致。

---

## 3. 路線 B：文獻實測對照

### 3.1 檢索範圍與納入準則

檢索對象為海洋／河川塑膠與廢棄物之沉降速度實測研究，涵蓋沉降槽（settling column）與迴流水槽（flume）量測。納入準則：報告終端沉降速度數值、註明物件材質與尺寸、實驗條件可辨識。

**納入時明確區分三個層級**，不得混用：
1. **同類物件實測** —— 直接量測該類海廢本體；
2. **代理物件實測** —— 量測材質或形態相近但非同一物件者（如以瓶蓋代理罐體、以曲殼薄片代理瓶身碎片）；
3. **無可比實測** —— 該類在文獻中完全空白。

### 3.2 逐類文獻證據

| 類別 | 文獻實測值 (m/s) | 證據層級 | 來源 |
|---|---|---|---|
| 保麗龍 | 查無沉降實測；EPS 實測皆為**上升** +0.314 | 無可比實測 | Waldschläger & Schüttrumpf (2019)；Kaiser et al. (2017) |
| 竹木 | 樹葉 0.019–0.023；木屑因無法吸水飽和到下沉而被排除於分析外 | 代理 | Pietsch & Chapman (2023) |
| 廢紙 | 查無任何可比實測 | 無可比實測 | — |
| 其他／不可回收 | PET 薄片沉降 0.029–0.037；PE 薄片**上升** 0.0001–0.004；PP 薄片**上升** 0.002–0.006 | 同類（薄片） | Kuizenga et al. (2022) |
| 廢漁網漁具 | 配重網片 PES 0.20–0.30、PA 0.15–0.25、PE 0.05–0.15（最大 0.40）；單根 PA 纖維 0.0039 | 同類（但含沉子） | Shan et al. (2023)；Waldschläger & Schüttrumpf (2019) |
| 寶特瓶 | PET 碎片 0.018–0.105；瓶片狀曲殼（D = 22–51 mm）0.060–0.070 | 同類（碎片） | Van Melkebeke et al. (2020)；Chan et al. (2021) |
| 其他／可回收 | 環境風化微塑膠與大型塑膠 0.0016–0.0352；0.006–0.091 | 同類 | Waldschläger et al. (2020)；Kowalski et al. (2016) |
| 鋁罐 | 鋁瓶蓋（D_eq 1.03–1.15 cm、密度 1.66–1.97 g/cm³）0.115–0.120 | 代理（瓶蓋） | Nikolaev (2022) |
| 鐵罐 | 鐵瓶蓋（D_eq 0.97 cm、密度 4.43 g/cm³）0.176–0.177 | 代理（瓶蓋） | Nikolaev (2022) |
| 玻璃瓶 | 方解石顆粒（密度 2.7、1.0–2.8 mm）0.32；曲殼（密度比 1.70）0.060–0.070 | 代理 | Kramer et al. (2021)；Chan et al. (2021) |

**方法學警訊（須於報告中一併載明）**：Nikolaev (2022) 之沉降管內徑 5.5 cm，而瓶蓋直徑約 2 cm，`d/D ≈ 0.4`，壁面遲滯效應顯著。其報告值應視為**真實自由沉降速度的下限**。

### 3.3 資料空白

全球文獻中**查無任何公分至數十公分級完整海廢物件**（完整罐、完整瓶、無配重廢網片）的終端沉降速度實測。最接近者為約 1 cm 的瓶蓋與約 5 cm 的曲面薄殼。

這是領域性的資料空白，不是本案的檢索疏漏。保麗龍、竹木、廢紙三類更是連代理實測都付之闕如，僅能以路線 A estimate。

---

## 4. 兩條路線的交叉驗證

| 類別 | 路線 A（理論） | 路線 B（實測） | 判定 |
|---|---:|---|---|
| **鋁罐** | 0.132 | 鋁瓶蓋 0.115–0.120 | **最佳一致**：相差 10–15%，且實測因管壁遲滯偏低、應視為下限，故理論值略高方向正確 |
| 寶特瓶 | 0.0782 | PET 碎片 0.018–0.105；曲殼 0.060–0.070 | **良好**：落在實測區間內 |
| 廢漁網漁具 | 0.0434 | 單根纖維 0.0039；配重網片 0.05–0.40 | **合理**：落在「單線」與「配重網片」兩端之間 |
| 其他／不可回收 | 0.0142 | PET 薄片 0.029–0.037 | 同量級（理論值偏低，因本類含更薄的複合膜片） |
| 鐵罐 | 0.301 | 鐵瓶蓋 0.176–0.177 | 同量級；理論值較快，因罐體等效直徑約為瓶蓋的 10 倍 |
| 其他／可回收 | 0.128 | 環境風化混合 0.0016–0.0352 | 理論值偏高約 3.6×，因實測對象尺寸明顯偏小 |
| 玻璃瓶 | 0.472 | 方解石粒 0.32；曲殼 0.060–0.070 | 同量級，兩條代理由上下夾住 |
| 保麗龍、竹木、廢紙 | 0.0359 / 0.0684 / 0.0384 | 查無同類沉降實測 | 僅有理論依據 |

**交叉驗證的核心結論**：在七個有實測或代理實測可比的類別中，兩條路線皆落在同一數量級；其中鋁罐一項達 10–15% 的定量吻合。這是本案十個速度最強的佐證——兩條方法論完全獨立的路線得到一致結果。

---

## 5. 採用值的不確定度

採用值 `w_repr` 為代表值，`w_min`／`w_max` 為建議的敏感度上下界。不確定度來自三個來源，依影響大小排序：

1. **物件尺寸與破碎程度**（最大）—— 造成類別內 1.6×–120× 的跨幅。
2. **姿態與阻力係數**（中等）—— 以 `C_d,∞` 上下界涵蓋，對單一情境造成約 ±15–30% 的速度差。
3. **材質密度假設**（最小，中空容器除外）—— 對薄壁中空容器，有效密度由殼體質量推算而非假設，不確定度已大幅壓低。

---

## 6. 限制

1. 全球文獻中查無公分級完整海廢物件的終端沉降速度實測。路線 A 為理論估算，**未經本案樣本驗證**。
2. 假設靜止均質水柱、剛體不變形、忽略壁效應與群體沉降。真實物件會翻滾、飄盪、變形，實際速度通常低於理論值；此效應已部分以 `C_d,∞` 上界涵蓋。
3. 30 組情境的尺寸，以及吸水／生物附著後的有效密度，為本研究設定之代表性假設，非本案現地樣本統計（分級詳見 2.7.1 第三級）。寶特瓶殼重 25 g 為唯一偏快方向之假設，其敏感度已於 2.7.1 量化。
4. **保麗龍、竹木、廢紙三類另有「是否下沉」的門檻問題**：未達吸水飽和或未生物附著前根本不下沉（Liu et al. 2022 觀察到 100 μm EPS 顆粒經約兩週生物附著後才開始下沉；Pietsch & Chapman 2023 之木屑因始終無法飽和到下沉而被排除於分析外）。單一速度無法描述此雙峰行為。模式中此三類僅在已轉為負浮力且完全沉沒之條件下納入。
5. `Re > 1000` 之情境（30 組中多數）採用的 Schiller–Naumann 項會額外貢獻約 5–16% 的阻力，使速度略偏低。此偏差方向對本案結論屬**保守**。
6. 模式於整條軌跡中維持固定的形狀與沉降速度，不模擬持續解體、溶散、團聚或生物附著的時間演化。

---

## 7. 參考文獻

**推導方法**

- Clift, R., & Gauvin, W. H. (1970). The motion of particles in turbulent gas streams. *Proceedings of Chemeca '70*, 1, 14–28.
- Clift, R., Grace, J. R., & Weber, M. E. (1978). *Bubbles, Drops, and Particles*. Academic Press.
- Ferguson, R. I., & Church, M. (2004). A simple universal equation for grain settling velocity. *Journal of Sedimentary Research*, 74(6), 933–937. https://doi.org/10.1306/051204740933
- Schiller, L., & Naumann, A. (1935). A drag coefficient correlation. *Zeitschrift des Vereins Deutscher Ingenieure*, 77, 318–320.
- Dietrich, W. E. (1982). Settling velocity of natural particles. *Water Resources Research*, 18(6), 1615–1626. https://doi.org/10.1029/WR018i006p01615

**同方法應用於塑膠／海廢顆粒（方法依據）**

- Chubarenko, I., Bagaev, A., Zobkov, M., & Esiukova, E. (2016). On some physical and dynamical properties of microplastic particles in marine environment. *Marine Pollution Bulletin*, 108(1–2), 105–112. https://doi.org/10.1016/j.marpolbul.2016.04.048
- Dellino, P., Mele, D., Bonasia, R., Braia, G., La Volpe, L., & Sulpizio, R. (2005). The analysis of the influence of pumice shape on its terminal velocity. *Geophysical Research Letters*, 32, L21306. https://doi.org/10.1029/2005GL023954
- Francalanci, S., Paris, E., & Solari, L. (2021). On the prediction of settling velocity for plastic particles of different shapes. *Environmental Pollution*, 290, 118068. https://doi.org/10.1016/j.envpol.2021.118068
- Goral, K. D., Guler, H. G., Larsen, B. E., Carstensen, S., Christensen, E. D., Kerpen, N. B., Schlurmann, T., & Fuhrman, D. R. (2023). Settling velocity of microplastic particles having regular and irregular shapes. *Environmental Research*, 228, 115783. https://doi.org/10.1016/j.envres.2023.115783
- Yu, Z., Yang, G., & Zhang, W. (2022). A new model for the terminal settling velocity of microplastics. *Marine Pollution Bulletin*, 176, 113449. https://doi.org/10.1016/j.marpolbul.2022.113449
- Jalón-Rojas, I., Wang, X. H., & Fredj, E. (2019). A 3D numerical model to Track Marine Plastic Debris (TrackMPD): Sensitivity of microplastic trajectories and fates to particle dynamical properties and physical processes. *Marine Pollution Bulletin*, 141, 256–272. https://doi.org/10.1016/j.marpolbul.2019.02.052

**沉降速度實測**

- Chan, T. T. K., Blay Esteban, L., Huisman, S. G., Shrimpton, J. S., & Ganapathisubramani, B. (2021). Settling behaviour of thin curved particles in quiescent fluid and turbulence. *Journal of Fluid Mechanics*, 922, A30. https://doi.org/10.1017/jfm.2021.520
- Dittmar, S., Ruhl, A. S., Altmann, K., & Jekel, M. (2024). Settling velocities of small microplastic fragments and fibers. *Environmental Science & Technology*, 58, 6359–6369. https://doi.org/10.1021/acs.est.3c09602
- Kaiser, D., Kowalski, N., & Waniek, J. J. (2017). Effects of biofouling on the sinking behavior of microplastics. *Environmental Research Letters*, 12, 124003. https://doi.org/10.1088/1748-9326/aa8e8b
- Khatmullina, L., & Isachenko, I. (2017). Settling velocity of microplastic particles of regular shapes. *Marine Pollution Bulletin*, 114(2), 871–880. https://doi.org/10.1016/j.marpolbul.2016.11.024
- Kowalski, N., Reichardt, A. M., & Waniek, J. J. (2016). Sinking rates of microplastics and potential implications of their alteration by physical, biological, and chemical factors. *Marine Pollution Bulletin*, 109(1), 310–319. https://doi.org/10.1016/j.marpolbul.2016.05.064
- Kramer, O. J. I., et al. (2021). Improvement of the Richardson–Zaki settling velocity equation. *Drinking Water Engineering and Science*, 14, 53–71. https://doi.org/10.5194/dwes-14-53-2021
- Kuizenga, B., van Emmerik, T., Waldschläger, K., & Kooi, M. (2022). Will it float? Rising and settling velocities of common macroplastic foils. *ACS ES&T Water*, 2, 975–981. https://doi.org/10.1021/acsestwater.1c00467
- Liu, S., Huang, Y., Luo, D., Wang, X., Wang, Z., Ji, X., Chen, Z., Dahlgren, R. A., Zhang, M., & Shang, X. (2022). Integrated effects of polymer type, size and shape on the sinking dynamics of biofouled microplastics. *Water Research*, 220, 118656. https://doi.org/10.1016/j.watres.2022.118656
- Nikolaev, A. A. (2022). The physical characterization and terminal velocities of aluminium, iron and plastic bottle caps in a water environment. *Recycling*, 7(3), 28. https://doi.org/10.3390/recycling7030028
- Pietsch, T., & Chapman, C. (2023). Settling velocities of coarse organic solids. *Scientific Reports*, 13, 12760. https://doi.org/10.1038/s41598-023-39645-x
- Shan, X., et al. (2023). Sinking performance of netting panels. *Journal of Marine Science and Engineering*, 11(10), 1972. https://doi.org/10.3390/jmse11101972
- Van Melkebeke, M., Janssen, C., & De Meester, S. (2020). Characteristics and sinking behavior of typical microplastics including the potential effect of biofouling. *Environmental Science & Technology*, 54, 8668–8680. https://doi.org/10.1021/acs.est.9b07378
- Waldschläger, K., & Schüttrumpf, H. (2019). Effects of particle properties on the settling and rise velocities of microplastics in freshwater under laboratory conditions. *Environmental Science & Technology*, 53, 1958–1966. https://doi.org/10.1021/acs.est.8b06794
- Waldschläger, K., Born, M., Cowger, W., Gray, A., & Schüttrumpf, H. (2020). Settling and rising velocities of environmentally weathered micro- and macroplastic particles. *Environmental Research*, 191, 110192. https://doi.org/10.1016/j.envres.2020.110192

**官方文件**

- 海洋委員會海洋保育署。《海漂底廢棄物調查指引》（核定版）。
- 海洋委員會海洋保育署。iOcean 海洋廢棄物管理頁（十類分類名稱來源）。https://iocean.oca.gov.tw/OCA_OceanConservation/PUBLIC/Marine_Litter_v2.aspx

**情境設計比較**

- van der Molen, J., van Leeuwen, S. M., Govers, L. L., van der Heide, T., & Olff, H. (2021). Potential micro-plastics dispersal and accumulation in the North Sea, with application to the MSC Zoe incident. *Frontiers in Marine Science*, 8, 607203. https://doi.org/10.3389/fmars.2021.607203

---

## 附錄　重算方式

推導材料位於 repo 之 `data/settling_velocity_derivation/`，四個檔案均納入版控：

| 檔案 | 內容 |
|---|---|
| `derive.py` | 完整推導程式，僅依賴標準函式庫 |
| `results.csv` | 30 組情境的逐筆結果 |
| `scenario_parameters.md` | 30 組情境的完整輸入參數表 |
| `README.md` | 方法、假設清單、與文獻實測的比對、限制 |

執行 `python3 derive.py` 即可重算全部結果，輸出包含求解器驗證區塊、30 組逐情境結果、十類彙總表，並重寫 `results.csv`。程式為確定性計算，無隨機性與外部相依，重跑結果逐位元相同。
