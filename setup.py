# =============================================================================
# setup_models.py - 배포시 모든 모델 사전 다운로드 스크립트
# =============================================================================

import os
import sys
from pathlib import Path
import gdown
import zipfile
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import pickle
import argparse

# 현재 디렉토리 설정
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import config
from utils.embedding_index import load_jsonl_data, save_language_index

def setup_directories():
    """필요한 디렉토리 생성"""
    print("📁 디렉토리 구조 생성 중...")
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.DATA_DIR.mkdir(exist_ok=True)
    print("✅ 디렉토리 생성 완료")

def download_chinese_model():
    """중국어 모델 다운로드 및 설치"""
    print("📥 중국어 모델 다운로드 중...")
    
    if config.CHINESE_MODEL_LOCAL_PATH.exists():
        print("✅ 중국어 모델이 이미 존재합니다.")
        return True
    
    try:
        # 구글 드라이브에서 폴더 다운로드
        folder_url = f"https://drive.google.com/drive/folders/{config.CHINESE_MODEL_GDRIVE_ID}"
        print(f"📂 다운로드 URL: {folder_url}")
        
        gdown.download_folder(folder_url, output=str(config.MODELS_DIR), quiet=False)
        
        # 폴더명 정리
        if not config.CHINESE_MODEL_LOCAL_PATH.exists():
            for folder in config.MODELS_DIR.iterdir():
                if folder.is_dir() and folder.name != "chinese_model" and not folder.name.startswith("."):
                    print(f"📂 폴더명 변경: {folder.name} → chinese_model")
                    folder.rename(config.CHINESE_MODEL_LOCAL_PATH)
                    break
        
        # 다운로드 마커 생성
        (config.MODELS_DIR / ".chinese_model_downloaded").touch()
        
        print("✅ 중국어 모델 다운로드 완료")
        return True
        
    except Exception as e:
        print(f"❌ 중국어 모델 다운로드 실패: {e}")
        print("💡 수동 다운로드 방법:")
        print("   1. 구글 드라이브에서 chinese_model 폴더 다운로드")
        print("   2. models/chinese_model/ 경로에 압축 해제")
        return False

def download_vietnamese_model():
    """베트남어 모델 다운로드"""
    print("📥 베트남어 모델 다운로드 중...")
    
    try:
        # HuggingFace에서 모델 다운로드 (캐시됨)
        print(f"🤖 모델 다운로드: {config.VIETNAMESE_MODEL}")
        model = AutoModelForSeq2SeqLM.from_pretrained(config.VIETNAMESE_MODEL)
        
        print(f"🔤 토크나이저 다운로드: {config.VIETNAMESE_TOKENIZER}")
        tokenizer = AutoTokenizer.from_pretrained(config.VIETNAMESE_TOKENIZER)
        
        print("✅ 베트남어 모델 다운로드 완료")
        return True
        
    except Exception as e:
        print(f"❌ 베트남어 모델 다운로드 실패: {e}")
        return False

def build_embedding_indexes():
    """임베딩 인덱스 사전 구축"""
    print("🔍 임베딩 인덱스 구축 중...")
    
    try:
        # 임베딩 모델 로드
        print(f"📊 임베딩 모델 로드: {config.EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # 중국어 인덱스 구축
        if config.CN_LAW_DATA_PATH.exists():
            print("📚 중국어 법률 데이터 인덱싱...")
            passages, metadata = load_jsonl_data(config.CN_LAW_DATA_PATH)
            if passages:
                print(f"   📄 처리할 문서: {len(passages)}개")
                embeddings = embedding_model.encode(passages, show_progress_bar=True, batch_size=16)
                faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
                faiss_index.add(embeddings.astype('float32'))
                save_language_index(faiss_index, passages, metadata, 
                                  config.CN_FAISS_INDEX_PATH, config.CN_PASSAGES_PATH)
                print(f"✅ 중국어 인덱스 완료: {len(passages)}개 문서")
            else:
                print("⚠️ 중국어 법률 데이터가 비어있습니다.")
        else:
            print(f"⚠️ 중국어 법률 데이터 파일 없음: {config.CN_LAW_DATA_PATH}")
        
        # 베트남어 인덱스 구축
        if config.VN_LAW_DATA_PATH.exists():
            print("📚 베트남어 법률 데이터 인덱싱...")
            passages, metadata = load_jsonl_data(config.VN_LAW_DATA_PATH)
            if passages:
                print(f"   📄 처리할 문서: {len(passages)}개")
                embeddings = embedding_model.encode(passages, show_progress_bar=True, batch_size=16)
                faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
                faiss_index.add(embeddings.astype('float32'))
                save_language_index(faiss_index, passages, metadata,
                                  config.VN_FAISS_INDEX_PATH, config.VN_PASSAGES_PATH)
                print(f"✅ 베트남어 인덱스 완료: {len(passages)}개 문서")
            else:
                print("⚠️ 베트남어 법률 데이터가 비어있습니다.")
        else:
            print(f"⚠️ 베트남어 법률 데이터 파일 없음: {config.VN_LAW_DATA_PATH}")
        
        # 완료 마커 생성
        (config.DATA_DIR / ".indexes_built").touch()
        
        print("✅ 모든 임베딩 인덱스 구축 완료")
        return True
        
    except Exception as e:
        print(f"❌ 인덱스 구축 실패: {e}")
        return False

def create_deployment_marker():
    """배포 완료 마커 생성"""
    marker_data = {
        "deployment_complete": True,
        "chinese_model_ready": config.CHINESE_MODEL_LOCAL_PATH.exists(),
        "indexes_built": (config.DATA_DIR / ".indexes_built").exists(),
        "setup_version": "1.0",
        "created_at": str(Path(__file__).stat().st_mtime)
    }
    
    with open(config.BASE_DIR / ".deployment_ready", 'w') as f:
        json.dump(marker_data, f, indent=2)
    
    print("✅ 배포 준비 완료 마커 생성")
    print(f"📁 마커 파일: {config.BASE_DIR / '.deployment_ready'}")

def check_requirements():
    """필수 파일 및 설정 확인"""
    print("🔍 필수 요구사항 확인 중...")
    
    errors = []
    warnings = []
    
    # 데이터 파일 확인
    if not config.CN_LAW_DATA_PATH.exists():
        errors.append(f"중국어 법률 데이터 없음: {config.CN_LAW_DATA_PATH}")
    
    if not config.VN_LAW_DATA_PATH.exists():
        errors.append(f"베트남어 법률 데이터 없음: {config.VN_LAW_DATA_PATH}")
    
    # 구글 드라이브 ID 확인
    if not config.CHINESE_MODEL_GDRIVE_ID or config.CHINESE_MODEL_GDRIVE_ID == "your_google_drive_file_id":
        errors.append("config.py에서 CHINESE_MODEL_GDRIVE_ID 설정 필요")
    
    # 토크나이저 설정 확인
    if not config.VIETNAMESE_TOKENIZER or config.VIETNAMESE_TOKENIZER == "your-username/vietnamese-tokenizer":
        warnings.append("config.py에서 VIETNAMESE_TOKENIZER 설정 권장")
    
    # 결과 출력
    if errors:
        print("❌ 필수 요구사항 오류:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("⚠️ 권장사항:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("✅ 필수 요구사항 확인 완료")
    return True

def main():
    """메인 설정 프로세스"""
    parser = argparse.ArgumentParser(description="법률 챗봇 배포 준비")
    parser.add_argument("--skip-models", action="store_true", help="모델 다운로드 건너뛰기")
    parser.add_argument("--skip-indexes", action="store_true", help="인덱스 구축 건너뛰기")
    parser.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    
    args = parser.parse_args()
    
    print("🚀 법률 챗봇 배포 준비 시작...")
    print("=" * 60)
    
    # 0. 필수 요구사항 확인
    if not check_requirements():
        print("❌ 필수 요구사항을 충족하지 않습니다. 설정을 확인해주세요.")
        return False
    
    success_count = 0
    total_steps = 4
    
    # 1. 디렉토리 생성
    setup_directories()
    success_count += 1
    
    # 2. 중국어 모델 다운로드
    if not args.skip_models:
        if download_chinese_model() or args.force:
            success_count += 1
        else:
            print("⚠️ 중국어 모델 다운로드 실패, 계속 진행...")
    else:
        print("⏭️ 모델 다운로드 건너뛰기")
        success_count += 1
    
    # 3. 베트남어 모델 다운로드
    if not args.skip_models:
        if download_vietnamese_model() or args.force:
            success_count += 1
        else:
            print("⚠️ 베트남어 모델 다운로드 실패, 계속 진행...")
    else:
        print("⏭️ 베트남어 모델 다운로드 건너뛰기")
        success_count += 1
    
    # 4. 임베딩 인덱스 구축
    if not args.skip_indexes:
        if build_embedding_indexes() or args.force:
            success_count += 1
        else:
            print("⚠️ 인덱스 구축 실패, 계속 진행...")
    else:
        print("⏭️ 인덱스 구축 건너뛰기")
        success_count += 1
    
    # 5. 배포 완료 마커 생성
    create_deployment_marker()
    
    print("=" * 60)
    print(f"📊 완료: {success_count}/{total_steps} 단계")
    
    if success_count == total_steps:
        print("🎉 배포 준비 완료!")
        print("✨ 이제 streamlit run app.py 실행시 즉시 시작됩니다.")
        print("")
        print("🚀 프로덕션 모드 테스트:")
        print("   streamlit run app.py")
        print("")
        print("🔄 개발 모드로 되돌리기:")
        print("   rm .deployment_ready")
        return True
    else:
        print("⚠️ 일부 단계가 실패했습니다. 로그를 확인해주세요.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
