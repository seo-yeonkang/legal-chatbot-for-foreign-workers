# utils/common.py (새 파일 추천)
import json, config, time
from pathlib import Path
from config import MARKER_FILE

def mark_deployment_ready():
    (config.MODELS_DIR / ".deployment_ready").touch()
