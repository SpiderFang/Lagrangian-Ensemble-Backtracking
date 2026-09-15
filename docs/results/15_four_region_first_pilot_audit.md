# ABCD 四區第一次工程試跑稽核

## 稽核範圍與判讀層級

本紀錄整理 A、B、C、D 四區第一次 24 小時沉降粒子工程試跑的可核對數值、設定差異與
發布狀態。每站均採 20 個情境、4 個 shards、`M=1`；A 區貢寮與龜山島兩站合計 40 個
情境／粒子，B、C、D 各 20 個情境／粒子。輸入衍生 registry 的共同視窗為到達
`2024-01-02T01:00:00Z` 向前回溯 24 小時，含到達時刻共 25 個逐時節點。A 區的輸入入口
是貢寮與龜山島的 exact pair；B、C、D 各自是單站入口。

這些紀錄是 engineering pilot，用來檢查輸入綁定、執行、事件保留、資源與成果讀取鏈路。
它們不是五站正式研究成果，也沒有通過 `M`、時間步長、回溯期收斂、accepted input、
正式 release 或獨立觀測驗證，因此不能推導絕對來源機率、因果來源或沉積質量。

> **2026-09-15 後續裁決。** A 區這批 24 小時、每站 20 情境、`M=1` 產物的用途固定為
> 工程 DEMO，不再補跑相同規模的 finite-depth Stokes 對齊案例。正式 A 區沿用 v3、兩站
> 20 km local domain、12.5 km receptor core 與逐 RK4 階段 forcing 封閉失敗的核心方案，
> 但必須以四區五站完整母體重新建置 30 天輸入與執行；本文件以下的歷史試跑數字不得升格。

## 第一次試跑實測摘要

| 區域／站點 | 執行規模 | 已核對的終止／品質診斷 | 成果發布狀態 |
|---|---|---|---|
| A／貢寮 | 20 粒子、4 shards、`M=1`、24 h | `forcing_start=19`；`numerical_failure=1` | run、reconcile、benchmark、checksum 通過；舊 NFS 不支援 `renameat2(RENAME_NOREPLACE)`，圖面未發布 |
| A／龜山島 | 20 粒子、4 shards、`M=1`、24 h | `forcing_start=11`；`max_age=1`；`flow_domain_open_exit=6`；`coast_contact=2` | run、reconcile、benchmark、checksum 通過；同一舊 NFS 圖面發布限制 |
| B／新竹外海 | 20 粒子、4 shards、`M=1`、24 h | `forcing_start=20`；5,780 筆 observation；825,070 steps | 四張 pilot 圖面可讀；仍為 engineering pilot |
| C／後灣海生館 | 20 粒子、4 shards、`M=1`、24 h | `data_gap=14`；`max_age=5`；`numerical_failure=1`；3,097 筆 observation；`data_gap` 的 `QC=32`；失敗階段 `step_start/k2/k4=1/8/5` | 四張 pilot 圖面可讀；波浪空間支援不足仍保留為 QC／缺口，不補零 |
| D／連江 | 20 粒子、4 shards、`M=1`、24 h | `data_gap=12`；`max_age=8`；4,397 筆 observation；824,794 steps；`data_gap` 的 `QC=32`；失敗階段 `step_start/k2/k4=4/4/4` | 四張 pilot 圖面可讀；波浪空間支援不足仍保留為 QC／缺口，不補零 |

其中 `QC=32` 對應 NWW 波浪空間支援不足。C、D 的 `data_gap` 是取樣資料支援問題，不能
解釋成粒子已成功完成相同的物理回溯；A 的 `numerical_failure` 也必須保留失敗原因與
階段，不能以終止位置或靜水值替代。

## 「相同設定」檢查結果

registry 的時間窗相同，不等於既有 run 的執行設定相同。現場 run plan 顯示：

- A 使用 `no_stokes`。
- B、C、D 使用 `finite_depth_stokes`。
- 各區的 Kh、Kz 與 Smagorinsky cap 可依區域資料校準不同；這是跨區設定比較器明示允許的
  差異，不會把區域校準值誤判成共同常數。

因此第一次四區結果不能寫成「相同設定下的跨區比較」。可在四個 run root 上執行：

```bash
uv run lbt pilot-matrix-validate \
  "$A_RUN_ROOT" "$B_RUN_ROOT" "$C_RUN_ROOT" "$D_RUN_ROOT" \
  > "$PILOT_SCRATCH_ROOT/abcd-first-pilot-matrix.json"
```

`pilot-matrix-validate` 只讀每個 root 的 `run_plan.json` 與 `normalized_config.json`。
它要求 `run_kind`、experiment、`M`／seed、selection、shard／chunk／checkpoint、積分與
邊界、Stokes 與無效波政策、材質／沉降、execution scalar snapshot、程式 tree 與依賴環境
完全一致；研究站點、flow domain、arrival／scenario identity、輸入／幾何 hash／path 及
區域 Kh／Kz／Smagorinsky cap 可不同。輸出是 canonical JSON；通過回傳 `0`，不通過或
輸入不完整回傳 `2`。這個命令不讀 trajectory、forcing 或 checkpoint 大檔，也不代替
`validate-run`。

## NFS 圖面發布與 `.complete` 語意

A 的舊圖面未發布是成果發布協定的限制，不是 A run、reconcile、benchmark 或 checksum
失敗。B、C、D 圖面已可讀，也不能因為有圖就升格為正式成果。新版 NFS
`nfs_completion_marker_v1` 會在同一父目錄 cooperative lock、完整 staging inventory、
逐檔 durable move、manifest／程式 provenance／儲存閘門綁定後，最後建立 `.complete`。

`.complete` 只表示 preview 或 figure artifact 的檔案集合、位元組、SHA-256 與 provenance
已封閉，可交給下游 reader；它不是粒子 run 的生命週期完成旗標。run 是否完成仍以
`run_progress.json`、`run-reconcile` 與 `validate-run --require-complete` 判定。若逐檔發布
中途失敗，沒有 `.complete` 的 final 目錄應保持 invalid 供稽核，不能由目錄存在或部分圖檔
存在推定成果已完成。

## 重新判讀前必須完成的修正與 gate

1. 不再補跑 A 區相同 24 小時／20 情境的工程基準。既有 `no_stokes` run 原樣保留為歷史
   DEMO，不作正式 Stokes 敏感度或跨區可比證據；正式 30 天執行須使用同一份乾淨 deployment
   snapshot，並保存 Python／NumPy／Numba／PyArrow 環境與 lock provenance。
2. C、D 要先補足 NWW 波浪的空間支援，或改用經核准且留下 gap-safe 證據的 arrival window；
   不能把 `QC=32` 的缺口當成有效波浪速度，也不能以最近格點或零值補齊。
3. A 區正式 release 採 `runtime_stage_fail_closed_no_expansion_v1`，不再要求指定 20 km
   邊界的三產品共同兩格餘裕。OCM surface 必須支援完整 arrival 母體；OCM native 與 NWW3
   在每個實際 RK4 階段依位置、深度、UTC 與 mask 驗證，任一必要 forcing 無支援即停止並
   保存 `data_gap`／對應品質狀態，不得補值或降級為 current-only。
4. 四區都還缺 `M`、時間步長與回溯期的收斂證據，以及 accepted input manifest、正式 run
   provenance、aggregate／report release 與逐圖表科學 QC；`M=1` 只能驗證鏈路，不能支持
   系集穩定性或來源路徑的研究結論。
5. A 的 35 km sensitivity exclusion 目前只存在於設定 metadata，尚未有 formal runner
   consumer；本期 formal gate 未開。後續若重新納入，必須由明示且版本化的正式 consumer
   實際執行、保存分母、來源綁定與比較 release，不能僅依 metadata 宣稱已完成敏感度分析。

在上述 gate 完成前，四區圖面最多只能呈現各自的條件式來源足跡工程診斷；不能把 A／B／C／D
第一次試跑合併成正式五站成果，也不能把圖面中的路徑密度直接解讀為絕對來源機率或因果歸因。
