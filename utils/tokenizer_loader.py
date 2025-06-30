# =============================================================================
# utils/tokenizer_loader.py - 토크나이저 로더
# =============================================================================

from transformers import AutoTokenizer
import streamlit as st
import config

@st.cache_resource
def load_tokenizer(language: str):
    """
    언어에 따른 토크나이저 로드
    
    Args:
        language (str): 'zh' (중국어) 또는 'vi' (베트남어)
    
    Returns:
        tokenizer: HuggingFace 토크나이저
    """
    try:
        if language == "zh":
            # 중국어 - 로컬 다운로드된 모델에서 토크나이저 로드
            if not config.CHINESE_MODEL_LOCAL_PATH.exists():
                st.error("중국어 모델이 다운로드되지 않았습니다.")
                return None
                
            tokenizer = AutoTokenizer.from_pretrained(
                str(config.CHINESE_MODEL_LOCAL_PATH),
                local_files_only=True
            )
            
            # mBART의 경우 언어 토큰 설정
            if hasattr(tokenizer, 'src_lang'):
                tokenizer.src_lang = "zh_CN"
                tokenizer.tgt_lang = "zh_CN"
            return tokenizer
            
        elif language == "vi":
            # 베트남어 - HuggingFace Hub에서 커스텀 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(config.VIETNAMESE_TOKENIZER)
            return tokenizer
            
        else:
            raise ValueError(f"지원되지 않는 언어: {language}")
            
    except Exception as e:
        st.error(f"토크나이저 로드 실패 ({language}): {str(e)}")
        return None

def encode_text(tokenizer, text: str, max_length: int = None):
    """
    텍스트를 토큰화
    
    Args:
        tokenizer: 토크나이저
        text (str): 입력 텍스트
        max_length (int): 최대 길이
    
    Returns:
        dict: 토큰화된 입력
    """
    if max_length is None:
        max_length = config.MAX_INPUT_LENGTH
        
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
