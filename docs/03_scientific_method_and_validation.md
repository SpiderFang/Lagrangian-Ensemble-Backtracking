# 科學方法與驗證規格

## 1. 座標、狀態與符號

每個 flow domain 在固定 metric CRS 中計算。粒子狀態為：

```text
state = (x_m, y_m, z_m_positive_up, time_utc_ns, triangle_id, status)
```

SCHISM 的 `z` 採正向向上，海面為 `z=eta`，海床近似為 `z=-depth`。此定義與 [SCHISM Physical Formulation](https://schism-dev.github.io/schism/v5.13/schism/physical-formulation.html) 一致；程式不得混用 positive-down depth 與 positive-up z。

完全沉沒粒子的確定性速度定義為：

```text
V_det = (u_ocm + u_stokes, v_ocm + v_stokes, w_ocm + w_b)
```

- `w_b < 0` 表示物理時間向前的沉降，`w_b > 0` 表示上浮。
- Stokes baseline 只有水平分量。
- 完全沉沒 baseline 不含 windage。
- 每一項都以 m/s 表示；任一單位 gate 未通過時不得正式運算。

在沒有特定材質量測的前提下，十個 `w_b` 行為類別固定為 `[-0.100, -0.030, -0.010, -0.003, -0.001, 0, +0.001, +0.003, +0.010, +0.030] m/s`。這是跨三個數量級的敏感度設計，不是對十種具名廢棄物的量測校準；符號、ID 與限制詳見文件 08。

## 2. OCM 四維速度

OCM 速度由 native unstructured mesh 直接取樣：

1. 在每個 source node 的 `zcor` 柱中尋找包夾粒子 z 的兩個有限 layer。
2. 對 `hvel`、`vertical_velocity` 與候選 Kz 作線性垂向內插；禁止單側外插。
3. 使用所在 triangle 的 barycentric weights 作水平內插。
4. 使用前後兩個 OCM 時次作線性時間內插。

若任一必要支撐 node 無法包夾 z，回傳具原因的 invalid sample，不重新正規化剩餘 node 權重。此保守政策避免在陡坡、海床以下或海岸缺值旁製造人工速度。

## 3. 有限水深 bulk Stokes drift

### 3.1 輸入與方向

```text
Hs = significant_wave_height
Tp = 1 / peak_frequency
theta_to = (peak_direction_raw_deg + 180 deg) mod 360 deg
e_to = (sin(theta_to), cos(theta_to))
```

角度由正北起算、順時針增加。上述 `+180°` 來自 NWW 相鄰專案已登錄的 wave-from 推定慣例；正式 run 必須在 config 與 manifest 明示該慣例及其證據版本。

### 3.2 波數與垂向 profile

對 `omega=2*pi/Tp` 解有限水深分散關係：

```text
omega^2 = g k tanh(k h)
```

其中 `h=eta+depth` 為瞬時總水深，`z_r=z-eta` 為相對海面的垂向座標，範圍 `[-h,0]`。以 `a=Hs/2` 建立單色 bulk profile：

```text
U_s(z_r) = [a^2 omega k / (2 sinh^2(kh))]
           cosh(2k(z_r+h)) e_to
```

當 `kh -> infinity` 時，上式回復：

```text
U_s(z_r) = [pi^2 Hs^2 / (L Tp)] exp(4*pi*z_r/L) e_to
```

即附檔式 (7) 的深水形式。有限水深 profile 與深水極限需逐值測試；bulk `Hs/Tp` 近似無法重建完整方向頻譜，文獻也指出 monochromatic／bulk 與 broadband spectral Stokes profile 可有系統差異，因此 no-Stokes、deep-water 與 finite-depth 三案例是必要敏感度，而不是可選美化。方法限制可參考 [Romero, Hypolite and McWilliams (2021)](https://doi.org/10.1016/j.ocemod.2021.101873)。

### 3.3 Stokes QC

- `Hs`、`fp`、方向、h 或 z 無效時不得計算。
- `fp<=0`、粒子在海面上或海床下、dispersion root 未收斂時回傳個別 QC code。
- 保存 `kh`、wave steepness、relative depth 與 surface drift 診斷；超出線性波適用範圍時標示，而非靜默裁切。
- 以 NWW `mean_wavelength` 只作交叉檢查；它是平均波長，不可冒充 peak wavelength。

## 4. 確定性 backward RK4

實作以 signed `dt` 表示時間方向。backward run 使用 `dt<0`，而 `V_det` 始終保存物理時間向前的速度；這能避免 caller 取負一次、forcing 再取負一次的雙重反號。

對 ODE `dX/dt=V_det(X,t)`：

```text
k1 = V(X_n, t_n)
k2 = V(X_n + dt*k1/2, t_n + dt/2)
k3 = V(X_n + dt*k2/2, t_n + dt/2)
k4 = V(X_n + dt*k3,   t_n + dt)
X_adv = X_n + dt*(k1 + 2*k2 + 2*k3 + k4)/6
```

每個 stage 必須重新取樣 OCM、NWW與海面／海床；不能只在步首取得速度後假裝 RK4。

### 4.1 時步控制

基線固定輸出時間、內部 adaptive substep：

- advective displacement 不超過局地水平尺度的 0.25。
- 垂向位移不超過局地最小有效 layer thickness 的 0.25。
- diffusion RMS displacement 不超過對應尺度的 0.25。
- 不跨越下一個 forcing time boundary；跨月由 month window 明確切換。
- `dt_min`、`dt_max` 與步數上限由 config 設定，任何 clamp 都計數。

## 5. 隨機擴散

### 5.1 常數係數參考案例

對常數對角 diffusivity：

```text
delta_X_diff = (sqrt(2Kx*|dt|) Nx,
                sqrt(2Ky*|dt|) Ny,
                sqrt(2Kz*|dt|) Nz)
```

`Nx,Ny,Nz` 為互相獨立的標準常態。使用 `|dt|` 保持 backward ensemble 的變異為正；負 diffusivity 永遠非法。隨機增量在完整 RK4 advection step 前或後以已登錄 split 順序加入。

### 5.2 空變 Smagorinsky Kh

附檔基線在 metric CRS 計算：

```text
Kh = (Cs*Delta)^2 * sqrt((du/dx-dv/dy)^2 + (dv/dx+du/dy)^2)
```

- `Delta` 預定為局地 triangle area 的平方根；另以 1 km target spacing 作敏感度。
- `Cs` 至少比較 0.10、0.15、0.20，並以物理上核定的 Kh floor/cap 防止零擴散與單點爆量；floor/cap 命中率必須輸出。
- node 垂向取樣後，triangle 內的線性 shape functions 可直接得到 `du/dx` 等梯度；不得在經緯度上直接差分。

對空變 K，與 Eulerian advection-diffusion 一致的 Itô SDE 含 diffusivity-gradient drift；[OceanParcels diffusion 方法](https://docs.oceanparcels.org/en/latest/examples/tutorial_diffusion.html)亦明載此項及 Milstein 修正。本專案必須先固定所求的 backward pseudo-time generator，再推導 gradient term 的符號，並通過解析 Fokker-Planck／well-mixed 障壁案例。未通過前，Smagorinsky 只能列為研究敏感度，常數 Kh 為驗證基線。

### 5.3 垂向 Kz

- OCM `diffusivity` 作為 tracer vertical eddy diffusivity 候選，需確認 m²/s、正值範圍、surface／bottom 行為與 NaN 政策。
- 先以常數 Kz 驗證垂向 Brownian variance 及反射／吸收邊界，再加入 OCM Kz。
- 空變 Kz 同樣需要 gradient drift 或經驗證的 Milstein/對應 scheme；不能只套 `sqrt(2Kz dt)`。

### 5.4 backward 結果的解釋

忽略擴散時，逆時間積分是明確的終值 ODE。加入擴散後，time reversal 有多種不同統計定義，結果可能不同；[Gräwe et al. (2011)](https://doi.org/10.1016/j.jmarsys.2011.03.009)專門比較這些作法。因此基線輸出稱為「逆時間平流加正擴散的條件式來源足跡」，不宣稱是唯一的真實歷史路徑或 posterior source probability。

## 6. 邊界與事件

| 邊界／狀態 | 基線 | 必要敏感度或備註 |
|---|---|---|
| 貢寮／龜山島 own local boundary | anchor 半徑 25 km 圓周中連接有效外海的 arc first crossing，記錄 segment、弧長與 backward age 後繼續；岸線不計入 | 20/35 km 半徑敏感度；兩站 local domains 可重疊但主要事件與分母依 `study_site_id` 保存 |
| 另一站 local boundary | 穿越時寫 `other_site_local_domain_enter/exit` 後繼續；不得停止、改變 scenario 所屬或取代 own first-exit | 跨站穿越率、配對 UTC pathway/HDR overlap 與共享傳輸走廊診斷 |
| A 區共用 flow-domain open boundary | 貢寮與龜山島使用同一 outer boundary；計算線段 first crossing，記錄 segment 與弧長後停止 | 擴域前後比較 exit time、HDR 與 ranking；避免用任意站界切斷水動力連通 |
| 海岸／陸地 | 不允許跨越；記錄 coast contact 並停止 | reflect 作敏感度，不混入基準 |
| 海面 | neutral／sinking 類擴散越界時反射；rising 類到達海面以 `surface_regime_exit` 停止 | 完全沉沒 baseline 不加 windage，故不得在海面繼續當表面漂流 |
| 海床 | 記錄 first contact；suspended 反射，sinking／near-bed 首次接觸即 deposit 並停止 | 無再懸浮參數時不宣稱完整底床交換 |
| forcing start | 到 2024-01-01 或實際最早可用時次停止 | 不能環回或外插 |
| data gap | 超過核定間距或必要 forcing 缺值即停止 | no-Stokes 是另一個 physics case |
| max age | 到核定最大回溯期停止 | 與 exit 分開統計 |
| numerical failure | NaN、CFL 無法滿足、定位失敗、step limit | 必須可重現並列入失敗率 |

邊界 crossing 用步內線段與 polygon/segment 求交得到，不以步末位置替代；事件時間以同一步線性或更高階內插估計，並以 dt 減半驗證。

## 7. 情境與 ensemble

貢寮、龜山島、新竹、後灣與連江五個研究站點各自採三因子完整交叉，不再保留每站 1,000 分層抽樣、四區共用 20 個 receptors 或 A 區兩站共用 20 個 receptors 的選項：

\[
N_{\mathrm{base,site}}=N_{\mathrm{material}}N_{\mathrm{receptor,site}}N_{\mathrm{arrival}}
=10\times20\times50=10{,}000.
\]

\[
N_{\mathrm{base,A}}=2\times N_{\mathrm{base,site}}=20{,}000,
\qquad
N_{\mathrm{base,total}}=5\times N_{\mathrm{base,site}}=50{,}000.
\]

此處的「情境」是固定物性、受體與到達時間的一組參數。物理敏感度案例另以 `experiment_case_id` 表示，以免把 no-Stokes 等重跑錯算成計畫書的基礎情境。

每個研究站點的 50 個 arrival-time 條件均由同一套可重現分層設計建立：

- 48 個核心：`2 年 × 4 季 × 2 大／小潮類別 × 3 潮內相位 proxy`；三相位為最快上升、最快下降與 slack。
- 2 個補充：在 forcing 完整前提下，分別選取 local-domain 高波與強流案例。

分層與 deterministic tie-break 已定案；確切 UTC、潮位分類門檻與事件值由 2024-2025 SERVER 資料衍生，不再等待人工任選。貢寮／龜山島在 coverage 允許時使用配對 UTC，但仍各自通過 50 條 coverage。潮位導數相位只是 tide-phase proxy，不宣稱為現場三維最大漲／退潮流；若未來取得真實調查日期，另建立 observation-conditioned experiment，不改寫 baseline。

若同一情境包含隨機擴散、受體位置／深度微擾、forcing ensemble 或其他隨機項，需以不同 seed 產生獨立實現。第 `s` 個情境的 member 數記為 `M_s`，故單一 experiment case 的總軌跡數為：

\[
N_{\mathrm{trajectory,total}}=\sum_{s=1}^{50{,}000}M_s.
\]

只有所有情境採相同 `M_s=M` 時，才是每站點 `10,000×M`、A 區 `20,000×M`、全案 `50,000×M`；完全確定性試驗為 `M=1`。`M` 是本專案為估計 stochastic footprint 所需的實作參數，並非計畫書明列的第四個因子，也不能由「1,000」反推。

正式 baseline 原則上使用一致的 `M`，以使 receptor、material 與 arrival-time 間的 Monte Carlo 誤差可比較。先對代表性情境依序增加 members，觀察 boundary-exit ranking、50/75/90% HDR 面積與重疊、median travel time、pathway density 及 bootstrap interval；只有這些量在預先登錄門檻內穩定後，才固定最小合格 `M`。seed 以 master seed、scenario hash、experiment case 與 member ID 經可重現算法派生；分片、worker 數與 restart 不得改變 seed。

## 8. KDE、HDR 與路徑產品

### 8.1 邊界穿越

保存所有 first-exit points，不只保存 KDE raster。主統計包括：

1. 依 open boundary segment 弧長的 1D KDE／bin density，避免二維 KDE 把質量無意義地抹到 domain 外。
2. 依附檔式 (11) 在 metric plane 計算 2D Gaussian KDE，供地圖呈現與相容比較。
3. 50%、75%、90% highest-density region，以及三種 bandwidth、bootstrap confidence interval。

所有正規化必須寫明分母是全部釋放、有效成員、成功出界、受體、arrival time 或面積／邊界長度。

### 8.2 其他聚合

- `pathway_density`：每格不同 particle count 與累積停留時間分開輸出。
- `residence_time_s`：以實際輸出間隔／步內權重累積，不能只數 observation rows。
- `bottom_contact_density`：first contact 與 repeated contact 分開。
- `failure_density`：data gap、coast、numerical failure 的空間分布，防止把失敗區誤讀為低來源。

## 9. 驗證矩陣

### 9.1 資料與幾何

| 測試 | 通過條件 |
|---|---|
| schema/shape/time | 24 個月份 inventory 可稽核；time 嚴格遞增；OCM/NWW domain 相容 |
| CRS round-trip | WGS84→metric→WGS84 誤差低於預先登錄門檻，domain corner/center 均測 |
| triangle/quad | 面積、方向、對角線、triangle-to-face 與 barycentric sum 通過 |
| mask/wetdry | 合成乾濕 face、海岸 triangle 與真實 snapshot 人工圖面抽查一致 |
| 4D interpolation | 對線性 x/y/z/t 解析場達浮點容許誤差；無外插案例確實失敗 |

### 9.2 物理與積分

| 測試 | 通過條件 |
|---|---|
| constant flow | forward/backward 位移與解析解一致 |
| solid rotation | 閉合軌跡、半徑誤差與 dt 收斂符合 RK4 預期階數 |
| linear shear | 路徑、triangle gradient 與 Smagorinsky 值符合解析解 |
| settling/rising | `z(t)=z0+w_b t`，backward 時方向自然反轉且無雙重取負 |
| Stokes dispersion | deep/shallow 初值均收斂，代回 residual 達門檻 |
| Stokes profile | finite-depth 深水極限回復式 (7)，四個 cardinal DP 方向正確 |
| Brownian statistics | mean 在信賴區間含 0，variance 在統計容許範圍含 `2K t` |
| variable K | diffusivity barrier／well-mixed 與參考 PDE 一致後才啟用 |
| boundary crossing | 解析線段 crossing 位置、時間、segment 與弧長一致 |

### 9.3 系統與科學驗收

| 測試 | 通過條件 |
|---|---|
| NumPy/Numba | 單步 forcing、physics、event 與固定 seed 小案例在容許誤差內一致 |
| checkpoint/restart | 同 config/seed 分片中斷續跑後 row count、ID、event 與 checksum 等價 |
| dt convergence | dt 減半後 exit ranking、90% HDR、median travel time 與 path density 變化低於預先登錄值 |
| ensemble convergence | 隨 members 增加，主要統計與 bootstrap interval 穩定；由曲線決定正式 M |
| domain adequacy | 現行 A v3 僅作 pilot；正式 A 區南界目標不北於約 `24.50°N`，且 OCM native、OCM surface、NWW analysis 對貢寮／龜山島 25/35 km local boundary 均保留至少兩個共同有效格點；擴域前後主要 entry/exit/path/HDR 指標變化預設低於 10%，否則報告尺度依賴性或調整 domain version |
| cross-site semantics | 軌跡穿越 foreign local domain 前後位置、時間與方向可重現；`study_site_id`、scenario、seed、own-local first-exit 與 outer-stop state 不變；跨站比例以原站有效 members 為分母，不把兩站先合併 |
| known-source synthetic | 正向已知來源到受體案例，其來源落入逆向 footprint 的核定 HDR，並量化 coverage |
| forcing ablation | current-only、no-Stokes、deep/finite Stokes、Kh/Kz cases 可比較且命名不混淆 |

## 10. 可宣稱範圍

正式報告可使用：

- 「在指定 OCM/NWW3 forcing、受體、到達時間、物性與擴散假設下的條件式潛在來源足跡」。
- 「相對較常出現的邊界通量方向或傳輸走廊」。
- 「對 Stokes、浮沉、擴散、邊界與情境設計的敏感度」。

除非另有觀測、先驗、likelihood 與驗證，不可使用：

- 「垃圾確定來自某地」。
- 「KDE 值就是實際來源機率」。
- 「單一逆向隨機軌跡重建了真實歷史」。
- 「表層 Stokes 或 TRAP 證明了海底聚集成因」。
