# =============================================================================
# setup.py - 배포 시점 사전 구축 스크립트
# =============================================================================

import os
import sys
import gdown
import zipfile
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 설정 임포트
import config

def setup_directories():
    """필요한 디렉토리 생성"""
    print("📁 디렉토리 구조 생성 중...")
    config.DATA_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    print("✅ 디렉토리 생성 완료")

def download_chinese_model():
    """구글 드라이브에서 중국어 모델 다운로드"""
    print("📥 중국어 모델 다운로드 중...")
    
    # 이미 있으면 건너뛰기
    if config.CHINESE_MODEL_LOCAL_PATH.exists():
        print("✅ 중국어 모델이 이미 존재합니다.")
        return True
    
    try:
        # 구글 드라이브에서 폴더 다운로드
        folder_url = f"https://drive.google.com/drive/folders/{config.CHINESE_MODEL_GDRIVE_ID}"
        gdown.download_folder(folder_url, output=str(config.MODELS_DIR), quiet=False)
        
        # 폴더명 정리
        downloaded_folder = config.CHINESE_MODEL_LOCAL_PATH
        if not downloaded_folder.exists():
            for folder in config.MODELS_DIR.iterdir():
                if folder.is_dir() and folder.name != "chinese_model" and not folder.name.startswith("."):
                    folder.rename(downloaded_folder)
                    break
        
        # 다운로드 마커 생성
        download_marker = config.MODELS_DIR / ".chinese_model_downloaded"
        download_marker.touch()
        
        print("✅ 중국어 모델 다운로드 완료")
        return True
        
    except Exception as e:
        print(f"❌ 중국어 모델 다운로드 실패: {e}")
        return False

def download_vietnamese_model():
    """베트남어 모델 다운로드"""
    print("📥 베트남어 모델 다운로드 중...")
    
    try:
        # 모델과 토크나이저 다운로드 (캐시에 저장됨)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.VIETNAMESE_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(config.VIETNAMESE_TOKENIZER)
        
        print("✅ 베트남어 모델 다운로드 완료")
        return True
        
    except Exception as e:
        print(f"❌ 베트남어 모델 다운로드 실패: {e}")
        return False

def build_embeddings_and_indexes():
    """임베딩 모델 로드 및 FAISS 인덱스 구축"""
    print("📚 임베딩 시스템 구축 중...")
    
    try:
        # 임베딩 모델 로드
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("✅ 임베딩 모델 로드 완료")
        
        # 중국어 인덱스 구축
        print("🇨🇳 중국어 법률 데이터 인덱싱 중...")
        cn_success = build_language_index(
            embedding_model, 'zh', 
            config.CN_LAW_DATA_PATH,
            config.CN_FAISS_INDEX_PATH,
            config.CN_PASSAGES_PATH
        )
        
        # 베트남어 인덱스 구축  
        print("🇻🇳 베트남어 법률 데이터 인덱싱 중...")
        vn_success = build_language_index(
            embedding_model, 'vi',
            config.VN_LAW_DATA_PATH, 
            config.VN_FAISS_INDEX_PATH,
            config.VN_PASSAGES_PATH
        )
        
        if cn_success and vn_success:
            print("✅ 모든 언어 인덱싱 완료")
            return True
        else:
            print("⚠️ 일부 언어 인덱싱 실패")
            return False
            
    except Exception as e:
        print(f"❌ 임베딩 시스템 구축 실패: {e}")
        return False

def build_language_index(embedding_model, language, jsonl_path, faiss_path, passages_path):
    """특정 언어의 FAISS 인덱스 구축"""
    try:
        # 이미 존재하면 건너뛰기
        if faiss_path.exists() and passages_path.exists():
            print(f"✅ {language} 인덱스가 이미 존재합니다.")
            return True
        
        # JSONL 데이터 로드
        passages, metadata = load_jsonl_data(jsonl_path)
        if not passages:
            print(f"❌ {language} 데이터를 찾을 수 없습니다: {jsonl_path}")
            return False
        
        print(f"📄 {language} 문서 {len(passages)}개 임베딩 생성 중...")
        
        # 임베딩 생성 (배치 처리)
        embeddings = embedding_model.encode(
            passages,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        
        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings.astype('float32'))
        
        # 저장
        faiss.write_index(faiss_index, str(faiss_path))
        
        with open(passages_path, 'wb') as f:
            pickle.dump({
                'passages': passages,
                'metadata': metadata
            }, f)
        
        print(f"✅ {language} 인덱스 저장 완료: {len(passages)}개 문서")
        return True
        
    except Exception as e:
        print(f"❌ {language} 인덱스 구축 실패: {e}")
        return False

def load_jsonl_data(jsonl_path):
    """JSONL 파일에서 데이터 로드"""
    passages = []
    metadata = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if "Trans_Sentence" in data:
                        passages.append(data["Trans_Sentence"])
                        metadata.append(data)
                except json.JSONDecodeError:
                    print(f"⚠️ Line {line_num}: JSON 파싱 오류")
                    continue
        
        return passages, metadata
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {jsonl_path}")
        return [], []

def verify_setup():
    """설정 완료 검증"""
    print("🔍 설정 검증 중...")
    
    checks = []
    
    # 중국어 모델 확인
    chinese_model_ok = config.CHINESE_MODEL_LOCAL_PATH.exists()
    checks.append(("중국어 모델", chinese_model_ok))
    
    # 중국어 인덱스 확인
    cn_index_ok = config.CN_FAISS_INDEX_PATH.exists() and config.CN_PASSAGES_PATH.exists()
    checks.append(("중국어 인덱스", cn_index_ok))
    
    # 베트남어 인덱스 확인
    vn_index_ok = config.VN_FAISS_INDEX_PATH.exists() and config.VN_PASSAGES_PATH.exists()
    checks.append(("베트남어 인덱스", vn_index_ok))
    
    # 결과 출력
    print("\n📋 설정 검증 결과:")
    all_ok = True
    for name, status in checks:
        status_str = "✅" if status else "❌"
        print(f"  {status_str} {name}: {'OK' if status else 'FAIL'}")
        if not status:
            all_ok = False
    
    return all_ok

def create_deployment_marker():
    """배포 완료 마커 생성"""
    marker_file = config.DATA_DIR / ".deployment_ready"
    marker_data = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "version": "1.0",
        "chinese_model": str(config.CHINESE_MODEL_LOCAL_PATH),
        "models_ready": True
    }
    
    with open(marker_file, 'w') as f:
        json.dump(marker_data, f, indent=2)
    
    print("✅ 배포 준비 완료 마커 생성됨")

def main():
    """메인 설정 함수"""
    print("🚀 배포 시점 사전 구축 시작!")
    print("=" * 50)
    
    # 1. 디렉토리 생성
    setup_directories()
    
    # 2. 중국어 모델 다운로드
    if not download_chinese_model():
        print("❌ 중국어 모델 다운로드 실패 - 수동 설정 필요")
        return False
    
    # 3. 베트남어 모델 다운로드
    if not download_vietnamese_model():
        print("❌ 베트남어 모델 다운로드 실패")
        return False
    
    # 4. 임베딩 시스템 구축
    if not build_embeddings_and_indexes():
        print("❌ 임베딩 시스템 구축 실패")
        return False
    
    # 5. 검증
    if verify_setup():
        create_deployment_marker()
        print("\n🎉 배포 사전 구축 완료!")
        print("이제 앱 실행 시 즉시 서비스가 시작됩니다.")
        return True
    else:
        print("\n❌ 일부 구성 요소가 누락되었습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
