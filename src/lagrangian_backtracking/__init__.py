"""三維 Lagrangian 系集逆向溯源的可重現科學計算套件。

套件只讀取上游已驗收的 OCM schema 3 與 NWW3 analysis schema 1 產品，不讀取
raw NetCDF，也不修改上游快取。公開 API 先提供設定、preflight、科學資料型別與
純 NumPy 參考核心；正式批次必須由 CLI 產生完整 manifest 與 QC 證據。
"""

from .config import ProjectConfig, load_config
from .models import BoundaryEvent, EventType, ParticleState, ParticleStatus, SampleQC, VelocitySample

__all__ = [
    "BoundaryEvent",
    "EventType",
    "ParticleState",
    "ParticleStatus",
    "ProjectConfig",
    "SampleQC",
    "VelocitySample",
    "load_config",
]

__version__ = "0.1.0"
