# =============================================================================
# config.py - 설정 파일
# =============================================================================

import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

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

# 중국어 모델 (구글 드라이브에서 다운로드)
# 방법 1: 폴더를 ZIP으로 압축한 경우
CHINESE_MODEL_GDRIVE_ID = "1MYuh7dM_w_-292VBHFNaoSD57_mJWcQ"  # 폴더 ID
CHINESE_MODEL_LOCAL_PATH = MODELS_DIR / "chinese_model"
CHINESE_MODEL_ZIP_PATH = MODELS_DIR / "chinese_model.zip"

# 방법 2: 폴더 직접 다운로드 (gdown으로 폴더 다운로드)
CHINESE_MODEL_FOLDER_ID = "1MYuh7dM_w_-292VBHFNaoSD57_mJWcQ"  # 폴더 ID

# 베트남어 모델 (외부 모델 + 커스텀 토크나이저)
VIETNAMESE_MODEL = "google/mt5-small"
VIETNAMESE_TOKENIZER = "seo-yeonkang/legal-chatbot-for-foreign-workers/models/vietnamese-tokenizer"  # HF Hub의 커스텀 토크나이저

# 생성 설정
MAX_RETRIEVED_DOCS = 3
MAX_INPUT_LENGTH = 512
MAX_GENERATION_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Streamlit 설정
PAGE_TITLE = "외국인 근로자 법률 챗봇"
PAGE_ICON = "⚖️"
