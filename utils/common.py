# utils/common.py (새 파일 추천)
import json, config, time
from pathlib import Path
from config import MARKER_FILE

def mark_deployment_ready():
    """모든 모델·인덱스가 준비된 뒤 호출 → .deployment_ready 기록"""
    MARKER_FILE.write_text('{"deployment_complete": true}')
    data = {
        "deployment_complete": True,
        "timestamp": time.time(),
        "setup_version": "1.0"          # 원하면 버전 증가
    }
    marker.write_text(json.dumps(data))
