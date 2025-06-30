# utils/common.py (새 파일 추천)
import json, config, time
from pathlib import Path

def mark_deployment_ready():
    """모든 모델·인덱스가 준비된 뒤 호출 → .deployment_ready 기록"""
    marker = config.BASE_DIR / ".deployment_ready"
    data = {
        "deployment_complete": True,
        "timestamp": time.time(),
        "setup_version": "1.0"          # 원하면 버전 증가
    }
    marker.write_text(json.dumps(data))
