# =============================================================================
# utils/generator.py - 답변 생성
# =============================================================================

from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
import torch
import streamlit as st
import config
import gdown
import zipfile
import os
from pathlib import Path
import faiss
import pickle
import json

# SentenceTransformer는 필요할 때만 import (circular import 방지)
def get_sentence_transformer():
    """SentenceTransformer 지연 import"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

READY_MARKER = config.CHINESE_MODEL_LOCAL_PATH / ".ready"   # ✅ 1회 완료표시

def download_chinese_model_from_gdrive():
    """
    • models/chinese_model/ 가 없으면 Drive 폴더(ID)에서 다운로드
    • 완료 후 '.ready' 파일 생성 → 다음 실행부터 다운로드 스킵
    """
    # ① 이미 준비됐으면 즉시 반환
    if READY_MARKER.exists():
        return str(config.CHINESE_MODEL_LOCAL_PATH)

    try:
        st.info("📥 중국어 모델이 없어서 Google Drive에서 다운로드합니다…")

        # models/ 디렉터리
        config.MODELS_DIR.mkdir(exist_ok=True, parents=True)

        # ② Drive 폴더 전체 다운로드 → models/ 하위에
        folder_url = f"https://drive.google.com/drive/folders/{config.CHINESE_MODEL_GDRIVE_ID}"
        gdown.download_folder(
            folder_url,
            output=str(config.CHINESE_MODEL_LOCAL_PATH),  # ✅ 바로 chinese_model 폴더
            quiet=False,
            use_cookies=False
        )

        # ③ 폴더명 정규화 → chinese_model
        if not config.CHINESE_MODEL_LOCAL_PATH.exists():
            for p in config.MODELS_DIR.iterdir():
                if p.is_dir() and p.name != "chinese_model":
                    p.rename(config.CHINESE_MODEL_LOCAL_PATH)
                    break

        # ④ 서브폴더에 숨어있는 가중치(.bin/.safetensors) 끌어올리기
        for root, _, files in os.walk(config.CHINESE_MODEL_LOCAL_PATH):
            for fn in files:
                if fn.endswith((".bin", ".safetensors")):
                    src = Path(root) / fn
                    dst = config.CHINESE_MODEL_LOCAL_PATH / fn
                    if not dst.exists():
                        src.replace(dst)

        # ⑤ 마커 생성 → 이후 재다운로드 없음
        READY_MARKER.touch()
        st.success("✅ 중국어 모델 준비 완료 (캐시됨)!")
        return str(config.CHINESE_MODEL_LOCAL_PATH)

    except Exception as e:
        st.error(f"❌ 모델 다운로드/설치 실패: {e}")
        return None
        

@st.cache_resource
def load_generation_models():
    """생성 모델들 로드 (구글 드라이브 + 외부 모델)"""
    try:
        # 중국어 모델 - 구글 드라이브에서 다운로드
        chinese_model_path = download_chinese_model_from_gdrive()
        if chinese_model_path is None:
            st.error("중국어 모델 로드 실패")
            return None, None
            
        chinese_model = MBartForConditionalGeneration.from_pretrained(
             chinese_model_path,
             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
             device_map="auto" if torch.cuda.is_available() else None,
             local_files_only=True
         )
        
        # 베트남어 모델 - 외부 모델 (HuggingFace Hub)
        vietnamese_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.VIETNAMESE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        st.success("✅ 모든 생성 모델 로드 완료!")
        return chinese_model, vietnamese_model
        
    except Exception as e:
        st.error(f"생성 모델 로드 실패: {str(e)}")
        return None, None

def generate_answer(prompt: str, model, tokenizer, max_length: int = None, temperature: float = None):
    """
    답변 생성
    
    Args:
        prompt (str): 입력 프롬프트
        model: 생성 모델
        tokenizer: 토크나이저
        max_length (int): 최대 생성 길이
        temperature (float): 생성 온도
    
    Returns:
        str: 생성된 답변
    """
    
    if max_length is None:
        max_length = config.MAX_GENERATION_LENGTH
    if temperature is None:
        temperature = config.TEMPERATURE
    
    try:
        # 입력 토큰화
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
            padding=True
        )
        
        inputs.pop("token_type_ids", None)
        
        # GPU 사용 가능하면 GPU로 이동
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 답변 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=config.TOP_P,
                do_sample=True if temperature > 0 else False,
                num_beams=3 if temperature == 0 else 1,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        generated_text = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 프롬프트 부분 제거 (mT5의 경우)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        st.error(f"답변 생성 실패: {str(e)}")
        return "죄송합니다. 답변 생성 중 오류가 발생했습니다."

def generate_streaming_answer(prompt: str, model, tokenizer, max_length: int = None):
    """
    스트리밍 답변 생성 (실시간 텍스트 출력)
    
    Args:
        prompt (str): 입력 프롬프트
        model: 생성 모델
        tokenizer: 토크나이저
        max_length (int): 최대 생성 길이
    
    Yields:
        str: 생성되는 토큰들
    """
    
    if max_length is None:
        max_length = config.MAX_GENERATION_LENGTH
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.MAX_INPUT_LENGTH)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)
        
        with torch.no_grad():
            # 스트리밍을 위한 설정
            generated_ids = inputs["input_ids"]
            
            for _ in range(max_length):
                outputs = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                # 새 토큰 디코딩
                new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                yield new_token
                
                # EOS 토큰이면 중단
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                    
    except Exception as e:
        yield f"스트리밍 생성 오류: {str(e)}"
