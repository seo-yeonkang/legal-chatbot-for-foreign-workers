# =============================================================================
# config.py - 설정 파일
# =============================================================================

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 기본 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MARKER_FILE = MODELS_DIR / ".deployment_ready"

# 데이터 경로 (언어별 분리)
CN_LAW_DATA_PATH = DATA_DIR / "cn_law_chunks.jsonl"
VN_LAW_DATA_PATH = DATA_DIR / "vn_law_chunks.jsonl"

# 캐시 경로 (언어별 분리)
CN_FAISS_INDEX_PATH = DATA_DIR / "cn_faiss_index.pkl"
CN_PASSAGES_PATH = DATA_DIR / "cn_passages.pkl"
VN_FAISS_INDEX_PATH = DATA_DIR / "vn_faiss_index.pkl"
VN_PASSAGES_PATH = DATA_DIR / "vn_passages.pkl"

# 모델 설정
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 중국어 모델 (Streamlit Cloud 최적화)
# 메모리 제한으로 인해 더 작은 모델 사용 권장
CHINESE_MODEL_GDRIVE_ID = "1MYuh7dM_w_-292VBHFNaoSD57_mJJWcQ"  # 폴더 ID
CHINESE_MODEL_LOCAL_PATH = MODELS_DIR / "chinese_model"
CHINESE_MODEL_ZIP_PATH = MODELS_DIR / "chinese_model.zip"

# Streamlit Cloud 메모리 제한 고려 (1GB)
# 대용량 모델 대신 경량 모델 사용 옵션
CHINESE_MODEL_FALLBACK = "Helsinki-NLP/opus-mt-en-zh"  # 경량 대체 모델

# 베트남어 모델 (경량화)
VIETNAMESE_MODEL     = "VietAI/vit5-base"
VIETNAMESE_TOKENIZER = "VietAI/vit5-base"

# Streamlit Cloud 메모리 제한 감지
STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') is not None or \
                  'streamlit.io' in os.environ.get('HOSTNAME', '') or \
                  '/mount/src/' in str(Path.cwd())

# 생성 설정 (Streamlit Cloud 최적화)
MAX_RETRIEVED_DOCS = 2 if STREAMLIT_CLOUD else 3  # 메모리 절약
MAX_INPUT_LENGTH = 256 if STREAMLIT_CLOUD else 512  # 메모리 절약
MAX_GENERATION_LENGTH = 256 if STREAMLIT_CLOUD else 512  # 메모리 절약
TEMPERATURE = 0.7
TOP_P = 0.9

# Streamlit 설정
PAGE_TITLE = "외국인 근로자 법률 챗봇"
PAGE_ICON = "⚖️"


