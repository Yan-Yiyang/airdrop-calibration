"""仅把当前独立仓库加入导入路径，不依赖主仓装置。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
