"""資料缺口截尾與正式統計分母的版本化政策常數。

OCM 已知缺口沒有可接受的補值來源，因此正式母體不把缺口內節點當成零值，也不
以最近時刻、線性內插或未登錄的外插方式跨越缺口。本模組只集中保存會進入設定、
manifest、runtime 與報告的識別碼，讓不同模組不會各自拼出近似但不相容的字串。
政策本身不讀取 forcing；是否確實遵守政策，仍由 input、runtime 與統計 validator
依各自資料重新驗證。
"""

from typing import Final

# 每條軌跡遇到逆向第一個未提供 OCM 節點即停止。被截尾的成員保留在總母體分母，
# 但不會被解讀成「沒有穿越邊界」；條件式來源足跡另以有效成員分母計算。
OBSERVED_GAP_CENSORED_STOP_AT_FIRST_GAP_POLICY_ID: Final[str] = (
    "observed_gap_censored_stop_at_first_gap_v1"
)

# 沉底年齡候選只在每個分層中抽取；若任一站點的同 rank 觀測時刻減去候選年齡
# 不在該站 canonical available-time set，便拒絕該候選並記錄稽核，不做最近值補齊。
REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID: Final[str] = (
    "reject_unavailable_deposition_hour_within_stratum_v1"
)

# 統計分母政策與 runtime 的 ParticleStatus.DATA_GAP、NUMERICAL_FAILURE 及
# PRE_WINDOW_DEPOSITION 旗標 exact 綁定；任何報告不得把這些成員誤列為未命中邊界。
EXCLUDE_DATA_GAP_NUMERICAL_FAILURE_AND_PRE_WINDOW_DEPOSITION_DENOMINATOR_POLICY_ID: Final[
    str
] = "exclude_data_gap_numerical_failure_and_pre_window_deposition_v1"

# 沉底 arrival manifest 的語意在加入缺口條件式抽樣稽核後升版。舊 1.1.0 僅能
# 讀取既有未條件式抽樣的 legacy artifact；不能以改寫 schema version 冒充新版。
GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION: Final[str] = "1.2.0"

__all__ = [
    "EXCLUDE_DATA_GAP_NUMERICAL_FAILURE_AND_PRE_WINDOW_DEPOSITION_DENOMINATOR_POLICY_ID",
    "GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION",
    "OBSERVED_GAP_CENSORED_STOP_AT_FIRST_GAP_POLICY_ID",
    "REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID",
]
