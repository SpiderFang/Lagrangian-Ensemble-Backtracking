# 文件總入口

本專案文件較多，是因為同時保存五種不能互相取代的證據：研究需求與歷史裁決、資料與科學方法、可重建的操作契約、成果發布規格，以及目前程式狀態。把它們全部塞在同一層會讓第一次接手的人分不清「今天能做什麼」與「當時為何這樣決定」，因此本目錄用分類保留原文，只在入口提供閱讀順序與路徑索引。

## 五條閱讀路線

| 目的 | 建議順序 |
|---|---|
| 給 PI 快速看 | 先看[實作狀態](implementation_status.md)，再看[設計基線](foundation/08_design_baseline_and_derived_gates.md)與[成果呈現規格](results/07_results_visualization_plan.md)。 |
| 第一次接手 | 先看[原始碼導覽](development/11_source_code_guide_and_plan_traceability.md)，再回看[架構與資料契約](foundation/02_architecture_and_data_contract.md)、[科學方法](foundation/03_scientific_method_and_validation.md)與[CLI 參考](operations/cli_reference.md)。 |
| SERVER 執行 | 依序看[CLI 參考](operations/cli_reference.md)、[SERVER 執行手冊](operations/06_server_runbook_plan.md)、[輸入衍生契約](operations/14_input_derivation_and_release_contract.md)、[checkpoint 操作契約](operations/18_checkpoint_segment_storage.md)，最後核對[實作狀態](implementation_status.md)。 |
| 成果報告 | 先看[成果呈現規格](results/07_results_visualization_plan.md)，再看[四區試跑稽核](results/15_four_region_first_pilot_audit.md)、[成果 release 管線](results/13_report_release_and_scientific_outputs_plan.md)、[聚合發布計畫](results/12_aggregate_release_and_server_execution_plan.md)與[實作狀態](implementation_status.md)。 |
| 查科學方法 | 依序看[科學方法與驗證](foundation/03_scientific_method_and_validation.md)、[設計基線](foundation/08_design_baseline_and_derived_gates.md)、[資料與時間缺口決策](operations/10_available_data_time_reconstruction_and_a_expansion.md)及[需求追溯](foundation/01_requirements_traceability.md)。 |

## 目錄結構

正式研究與工程試跑的效能工作分開管理，優先順序、結果等價與 30 天回溯的 SERVER 驗收見[效能改善工作線](operations/16_performance_improvement_tracks.md)。

- `foundation/`：需求、資料契約、科學方法與設計基線。
- `operations/`：CLI、SERVER、輸入衍生、資料時間處理、Git 同步與 pilot 操作。
- `results/`：成果圖表、聚合發布與報告發布規格。
- `development/`：決策風險與原始碼交接導覽。
- `archive/`：歷史實作計畫與歷史稽核快照；不代表今天的執行狀態。
- 根層：本索引、[實作狀態](implementation_status.md)、[互動式架構圖](source_code_architecture_map.html)與 `output/` 產生物。

## 原文件路徑對照

下表保留原本 18 份文件的檔名與原路徑，並登錄 3 份新增的 pilot／稽核及效能規劃文件；新增文件沒有
待相容的舊根層路徑，因此 current/original 欄位相同。「狀態」只表示閱讀定位，不把工程
測試通過誤寫成正式科學成果。

| 文件 | 現行路徑 | 一句用途 | 狀態 | 原路徑 |
|---|---|---|---|---|
| 實作狀態與正式驗證邊界 | [implementation_status.md](implementation_status.md) | 集中說明目前程式能力、正式性界線與尚待驗證項目。 | 現行 | `docs/implementation_status.md` |
| 正式運算及單站試跑效能改善 | [operations/16_performance_improvement_tracks.md](operations/16_performance_improvement_tracks.md) | 規範正式研究優先的加速工作線、結果等價與 30 天回溯驗收。 | 現行 | `docs/operations/16_performance_improvement_tracks.md` |
| 需求追溯與範圍裁決 | [foundation/01_requirements_traceability.md](foundation/01_requirements_traceability.md) | 將研究需求、使用者裁決與驗收條件對應到可追查的實作。 | 現行 | `docs/01_requirements_traceability.md` |
| 架構與資料契約 | [foundation/02_architecture_and_data_contract.md](foundation/02_architecture_and_data_contract.md) | 定義上游資料、模組邊界、狀態欄位與輸出拓撲。 | 現行 | `docs/02_architecture_and_data_contract.md` |
| 科學方法與驗證規格 | [foundation/03_scientific_method_and_validation.md](foundation/03_scientific_method_and_validation.md) | 說明座標、公式、時間積分、擴散與科學驗收方法。 | 現行 | `docs/03_scientific_method_and_validation.md` |
| 快速實作計畫 | [archive/04_implementation_plan.md](archive/04_implementation_plan.md) | 保存早期工作拆分、依賴關係與加速原則。 | 歷史 | `docs/04_implementation_plan.md` |
| 決策與風險登錄 | [development/05_decisions_and_risks.md](development/05_decisions_and_risks.md) | 記錄決策狀態、依據、風險與未決限制。 | 現行 | `docs/05_decisions_and_risks.md` |
| SERVER 執行手冊 | [operations/06_server_runbook_plan.md](operations/06_server_runbook_plan.md) | 定義資料盤點、執行、續跑、驗證與發布的操作程序。 | 現行 | `docs/06_server_runbook_plan.md` |
| 成果呈現與學術視覺化規格 | [results/07_results_visualization_plan.md](results/07_results_visualization_plan.md) | 定義成果圖表、分母、限制與報告呈現方式。 | 現行 | `docs/07_results_visualization_plan.md` |
| 五站點情境與巢狀邊界設計基線 | [foundation/08_design_baseline_and_derived_gates.md](foundation/08_design_baseline_and_derived_gates.md) | 固定五站情境、沉降代理、幾何與衍生閘門的設計基線。 | 現行 | `docs/08_design_baseline_and_derived_gates.md` |
| 實作與 SERVER 驗證稽核 | [archive/09_implementation_audit_2026-08-19.md](archive/09_implementation_audit_2026-08-19.md) | 保存 2026-08-19 稽核及其後續更正的歷史證據。 | 歷史 | `docs/09_implementation_audit_2026-08-19.md` |
| 全部可得資料、時間缺口重建與 A 區擴張決策 | [operations/10_available_data_time_reconstruction_and_a_expansion.md](operations/10_available_data_time_reconstruction_and_a_expansion.md) | 說明 2024–2025 資料母體、缺口處理與 A 區擴張判準。 | 現行 | `docs/10_available_data_time_reconstruction_and_a_expansion.md` |
| 程式碼導覽、執行流程與工項計畫書追溯 | [development/11_source_code_guide_and_plan_traceability.md](development/11_source_code_guide_and_plan_traceability.md) | 將程式模組、資料流程、測試與工項要求連成交接地圖。 | 現行 | `docs/11_source_code_guide_and_plan_traceability.md` |
| 聚合發布與 SERVER 正式執行實作計畫 | [results/12_aggregate_release_and_server_execution_plan.md](results/12_aggregate_release_and_server_execution_plan.md) | 規劃軌跡聚合、不可變發布、圖表產製與 SERVER 驗收。 | 現行 | `docs/12_aggregate_release_and_server_execution_plan.md` |
| 正式成果 release 與科學圖表生產管線規劃 | [results/13_report_release_and_scientific_outputs_plan.md](results/13_report_release_and_scientific_outputs_plan.md) | 將成果圖組與驗證證據轉成可重建的發布流程。 | 現行 | `docs/13_report_release_and_scientific_outputs_plan.md` |
| 涵蓋 2024-01-01 的新竹外海 24 小時工程試跑紀錄 | [results/14_hsinchu_2024-01-01_24h_pilot_parameter_record.md](results/14_hsinchu_2024-01-01_24h_pilot_parameter_record.md) | SERVER r5 執行、完整性驗證、後處理與參數／hash 證據；供 pilot 工程追溯，不代表正式研究成果。 | 現行 | `docs/results/14_hsinchu_2024-01-01_24h_pilot_parameter_record.md` |
| ABCD 第一次 24 小時試跑稽核 | [results/15_four_region_first_pilot_audit.md](results/15_four_region_first_pilot_audit.md) | 彙整 A／B／C／D 實測狀態、設定一致性差異、NFS 成果標記語意與正式 gate 缺口；供工程稽核，不代表正式研究成果。 | 現行 | `docs/results/15_four_region_first_pilot_audit.md` |
| 新竹 Numba／平行／checkpoint 測速紀錄 | [results/16_hsinchu_numba_parallel_checkpoint_benchmark.md](results/16_hsinchu_numba_parallel_checkpoint_benchmark.md) | 記錄純流與流加波浪的同站工程測速、NFS checkpoint 容量、輸出差異及正式採用限制。 | 現行 | `docs/results/16_hsinchu_numba_parallel_checkpoint_benchmark.md` |
| Slice 1：SERVER v3 輸入衍生與 release contract | [operations/14_input_derivation_and_release_contract.md](operations/14_input_derivation_and_release_contract.md) | 定義已驗收 OCM／NWW3 產品如何衍生並綁定 runtime 輸入。 | 現行 | `docs/14_input_derivation_and_release_contract.md` |
| CLI 與執行介面參考 | [operations/cli_reference.md](operations/cli_reference.md) | 對照目前可執行命令、參數、輸入輸出與安全限制。 | 現行 | `docs/cli_reference.md` |
| Git 部署與資料同步手冊 | [operations/git_deployment_and_data_sync.md](operations/git_deployment_and_data_sync.md) | 定義本機 Git、SERVER source、資料與部署驗收的邊界。 | 現行 | `docs/git_deployment_and_data_sync.md` |
| 單站沉降先導執行計畫 | [operations/pilot_run_plan.md](operations/pilot_run_plan.md) | 定義單站 pilot 的選取、限制、命令與[獨立海岸底圖重繪](operations/pilot_run_plan.md#獨立海岸底圖重繪)入口。 | 現行 | `docs/pilot_run_plan.md` |
| 單站 30 天工程測速操作契約 | [operations/17_engineering_window_benchmark.md](operations/17_engineering_window_benchmark.md) | 規範工程時窗的輸入、分片、計時、續跑與結果解讀界線。 | 現行 | `docs/operations/17_engineering_window_benchmark.md` |
| Execution checkpoint schema 3.0 操作契約 | [operations/18_checkpoint_segment_storage.md](operations/18_checkpoint_segment_storage.md) | 說明 immutable history segment、compact state、hash chain、容量語意與故障恢復。 | 現行 | `docs/operations/18_checkpoint_segment_storage.md` |

## 產生物與其他入口

| 產物 | 用途與狀態 |
|---|---|
| [source_code_architecture_map.html](source_code_architecture_map.html) | 由 `scripts/render_source_code_architecture_map.py` 產生的離線互動架構圖；狀態為產生物，需由 renderer 重新產製。 |
| `output/` | 保存流程圖等既有產生物；狀態為產生物，本次只更新受文件路徑影響的文字與索引，不重製二進位圖或 PDF。 |
| `data/*/README.md` | 各資料或文獻索引的局部入口；狀態為參考，完整分類仍以本頁五條路線為準。 |

## 狀態判讀

「現行」表示文件仍是目前設計或操作的依據；「參考」表示可用於補充背景但不單獨決定今天狀態；「歷史」表示保留原始快照與決策脈絡；「產生物」表示由腳本重建的 HTML、圖或 PDF。歷史文件若與程式、目前 manifest、測試或[實作狀態](implementation_status.md)不一致，以較新的可驗證證據為準。
