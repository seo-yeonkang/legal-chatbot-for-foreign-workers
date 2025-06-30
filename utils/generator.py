# =============================================================================
# utils/generator.py - 답변 생성
# =============================================================================

from transformers import AutoModelForSeq2SeqLM, SentenceTransformer
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

def download_chinese_model_from_gdrive():
    """구글 드라이브에서 중국어 모델 다운로드"""
    try:
        # 이미 모델이 있는지 확인
        if config.CHINESE_MODEL_LOCAL_PATH.exists():
            st.info("🔄 중국어 모델이 이미 로컬에 있습니다.")
            return str(config.CHINESE_MODEL_LOCAL_PATH)
        
        # 모델 디렉토리 생성
        config.MODELS_DIR.mkdir(exist_ok=True)
        
        # 구글 드라이브에서 폴더 다운로드
        st.info("📥 구글 드라이브에서 중국어 모델 폴더를 다운로드하고 있습니다...")
        
        # 방법 1: 폴더를 ZIP으로 압축했다면
        # url = f"https://drive.google.com/uc?id={config.CHINESE_MODEL_GDRIVE_ID}"
        # gdown.download(url, str(config.CHINESE_MODEL_ZIP_PATH), quiet=False)
        
        # 방법 2: 폴더 직접 다운로드 (권장)
        folder_url = f"https://drive.google.com/drive/folders/{config.CHINESE_MODEL_GDRIVE_ID}"
        gdown.download_folder(folder_url, output=str(config.MODELS_DIR), quiet=False)
        
        # 다운로드된 폴더명을 chinese_model로 변경 (필요시)
        downloaded_folder = config.MODELS_DIR / "chinese_model"
        if not downloaded_folder.exists():
            # gdown이 다른 이름으로 폴더를 만들었을 수 있음
            for folder in config.MODELS_DIR.iterdir():
                if folder.is_dir() and folder.name != "chinese_model":
                    folder.rename(downloaded_folder)
                    break
        
        st.success("✅ 중국어 모델 다운로드 완료!")
        return str(config.CHINESE_MODEL_LOCAL_PATH)
        
    except Exception as e:
        st.error(f"❌ 모델 다운로드 실패: {str(e)}")
        st.info("💡 수동 다운로드 방법: 구글 드라이브에서 폴더를 다운로드하고 models/chinese_model 경로에 압축 해제하세요.")
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
            
        chinese_model = AutoModelForSeq2SeqLM.from_pretrained(
            chinese_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only=True  # 로컬 파일만 사용
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
