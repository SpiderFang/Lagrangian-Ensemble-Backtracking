"""以 BayTrace 原版 njit 批次核心執行逆向追蹤的 LBT 子套件。

為什麼需要這個子套件
--------------------
舊路徑逐粒子以 Python 步進（每粒子步約 5 ms），正式試跑中 BayTrace 原版核心的呼叫
次數為 0。本子套件改為**直接驅動 BayTrace 原版批次核心**：同一批粒子共用全域同步
時鐘，每個子步只呼叫一次未修改的 ``batch_move_rk4``，一次處理整批粒子。

各模組分工（資料流由上而下）
----------------------------
- ``grid_bridge``：靜態網格（``NativeMesh`` 與 ``zcor``）一次轉成 BayTrace 網格陣列。
- ``forcing_bridge``：已驗收 ``ocm_native`` 逐月記憶體映射（memmap）產品轉成
  BayTrace 雙時次流場，每流場小時只建一次；缺值與乾點分開；缺口盤點。
- ``wave_bridge``：已驗收 ``nww3_analysis`` 波浪場以預先計算的雙線性權重內插到網格節點。
- ``stokes_batch``：工作書式(7) 有限水深 Stokes 漂流批次核心。
- ``particle_input``：情境矩陣轉成粒子身分表與 BayTrace 結構陣列，並提供分組。
- ``backward_driver``：逆向驅動層。把「海流＋Stokes＋沉降」合成進流場陣列後呼叫
  未修改的 BayTrace 核心；隨機擴散以算子分裂在四階 Runge-Kutta 法（RK4）之外逐子步施加。
- ``orchestrator``：工作單元切分、長駐多程序平行執行、續跑、缺口安全到達時刻。
- ``postprocess``：串流合併批次成果、離開紀錄、邊界穿越通量與條件式來源足跡。

BayTrace 原始碼一律不修改。
"""

from __future__ import annotations

from . import (
    backward_driver,
    forcing_bridge,
    grid_bridge,
    orchestrator,
    particle_input,
    postprocess,
    stokes_batch,
    wave_bridge,
)
from .backward_driver import *  # noqa: F403
from .forcing_bridge import *  # noqa: F403
from .grid_bridge import *  # noqa: F403
from .orchestrator import *  # noqa: F403
from .particle_input import *  # noqa: F403
from .postprocess import *  # noqa: F403
from .stokes_batch import *  # noqa: F403
from .wave_bridge import *  # noqa: F403

# 公開名稱由各模組自己的 ``__all__`` 決定，避免此處清單與模組實作脫節。
__all__ = [
    *grid_bridge.__all__,
    *forcing_bridge.__all__,
    *wave_bridge.__all__,
    *stokes_batch.__all__,
    *particle_input.__all__,
    *backward_driver.__all__,
    *orchestrator.__all__,
    *postprocess.__all__,
]
