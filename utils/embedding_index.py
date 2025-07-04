# =============================================================================
# utils/embedding_index.py - 임베딩 및 FAISS 인덱스
# =============================================================================

from sentence_transformers import SentenceTransformer
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
import faiss
import json
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
import config

def load_prebuilt_models():
    """사전 구축된 모델들을 즉시 로드"""
    try:
        # 중국어 모델 로드 (로컬에서)
        chinese_model = MBartForConditionalGeneration.from_pretrained(
             str(config.CHINESE_MODEL_LOCAL_PATH),
             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
             device_map="auto" if torch.cuda.is_available() else None,
             local_files_only=True
         )
        
        # 베트남어 모델 로드 (캐시에서)
        vietnamese_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.VIETNAMESE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        return chinese_model, vietnamese_model
        
    except Exception as e:
        st.error(f"사전 구축된 모델 로드 실패: {str(e)}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_embeddings_and_index():
    """
    언어별 임베딩 모델과 FAISS 인덱스 로드 (배포 모드 vs 개발 모드)
    
    Returns:
        tuple: (embedding_model, cn_index, cn_passages, cn_metadata, vn_index, vn_passages, vn_metadata)
    """
    
    # 배포 완료 상태 확인
    if is_deployment_ready():
        # 🚀 배포 모드: 사전 구축된 인덱스 즉시 로드
        st.success("⚡ 프로덕션 모드: 사전 구축된 인덱스 즉시 로드")
        return load_prebuilt_indexes()
    else:
        # 🔨 개발 모드: 기존 방식 (인덱스 생성 포함)
        st.info("🔨 개발 모드: 인덱스 생성 및 구축")
        
        # session_state 캐싱 확인
        if 'embedding_system_loaded' in st.session_state and st.session_state.embedding_system_loaded:
            st.info("⚡ 캐시된 임베딩 시스템을 사용합니다.")
            return (st.session_state.embed_model, st.session_state.cn_index, st.session_state.cn_passages, 
                    st.session_state.cn_metadata, st.session_state.vn_index, st.session_state.vn_passages, 
                    st.session_state.vn_metadata)
        
        try:
            # 임베딩 모델 로드 (공통)
            with st.spinner("🔄 임베딩 모델 로드 중..."):
                embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            
            # 중국어 인덱스 로드
            with st.spinner("🔄 중국어 법률 데이터 준비 중..."):
                cn_index, cn_passages, cn_metadata = load_language_index(
                    embedding_model, 
                    'zh',
                    config.CN_LAW_DATA_PATH,
                    config.CN_FAISS_INDEX_PATH,
                    config.CN_PASSAGES_PATH
                )
            
            # 베트남어 인덱스 로드
            with st.spinner("🔄 베트남어 법률 데이터 준비 중..."):
                vn_index, vn_passages, vn_metadata = load_language_index(
                    embedding_model,
                    'vi', 
                    config.VN_LAW_DATA_PATH,
                    config.VN_FAISS_INDEX_PATH,
                    config.VN_PASSAGES_PATH
                )
            
            # session_state에 저장 (이중 캐싱)
            st.session_state.embed_model = embedding_model
            st.session_state.cn_index = cn_index
            st.session_state.cn_passages = cn_passages
            st.session_state.cn_metadata = cn_metadata
            st.session_state.vn_index = vn_index
            st.session_state.vn_passages = vn_passages
            st.session_state.vn_metadata = vn_metadata
            st.session_state.embedding_system_loaded = True
            
            return embedding_model, cn_index, cn_passages, cn_metadata, vn_index, vn_passages, vn_metadata
            
        except Exception as e:
            st.error(f"임베딩 시스템 로드 실패: {str(e)}")
            return None, None, None, None, None, None, None

# is_deployment_ready 함수 추가 (generator.py에서 이동)
def is_deployment_ready():
    """배포 준비가 완료되었는지 확인"""
    """models/.deployment_ready  파일이 존재하기만 하면 True"""
    return (config.MODELS_DIR / ".deployment_ready").exists()

def load_language_index(embedding_model, language, jsonl_path, faiss_path, passages_path):
    """특정 언어의 인덱스 로드"""
    try:
        # 캐시된 인덱스가 있는지 확인
        if faiss_path.exists() and passages_path.exists():
            return load_cached_language_index(embedding_model, language, faiss_path, passages_path)
        
        # 새로 인덱스 생성
        return create_new_language_index(embedding_model, language, jsonl_path, faiss_path, passages_path)
        
    except Exception as e:
        st.warning(f"{language} 인덱스 로드 실패: {str(e)}")
        return None, None, None

def load_cached_language_index(embedding_model, language, faiss_path, passages_path):
    """캐시된 언어별 FAISS 인덱스 로드"""
    try:
        # FAISS 인덱스 로드
        faiss_index = faiss.read_index(str(faiss_path))
        
        # 패시지와 메타데이터 로드
        with open(passages_path, 'rb') as f:
            data = pickle.load(f)
            passages = data['passages']
            metadata = data['metadata']
        
        lang_name = "중국어" if language == "zh" else "베트남어"
        st.success(f"✅ {lang_name} 캐시 로드 완료: {len(passages)}개 문서")
        return faiss_index, passages, metadata
        
    except Exception as e:
        st.warning(f"{language} 캐시 로드 실패, 새로 생성: {str(e)}")
        return None, None, None

def create_new_language_index(embedding_model, language, jsonl_path, faiss_path, passages_path):
    """새로운 언어별 FAISS 인덱스 생성"""
    try:
        # JSONL 파일에서 데이터 로드
        passages, metadata = load_jsonl_data(jsonl_path)
        
        if not passages:
            st.error(f"{language} 법률 데이터를 찾을 수 없습니다: {jsonl_path}")
            return None, None, None
        
        # 임베딩 생성
        lang_name = "중국어" if language == "zh" else "베트남어"
        st.info(f"📚 {lang_name} 문서 임베딩 생성 중...")
        
        embeddings = embedding_model.encode(
            passages, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings.astype('float32'))
        
        # 캐시 저장
        save_language_index(faiss_index, passages, metadata, faiss_path, passages_path)
        
        st.success(f"✅ {lang_name} 새 인덱스 생성 완료: {len(passages)}개 문서")
        return faiss_index, passages, metadata
        
    except Exception as e:
        st.error(f"{language} 인덱스 생성 실패: {str(e)}")
        return None, None, None

def load_jsonl_data(jsonl_path):
    """JSONL 파일에서 데이터 로드"""
    passages = []
    metadata = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # 번역된 문장 사용 (Trans_Sentence 필드)
                    if "Trans_Sentence" in data:
                        passages.append(data["Trans_Sentence"])
                        metadata.append(data)
                    else:
                        st.warning(f"Line {line_num}: 'Trans_Sentence' 필드가 없습니다.")
                except json.JSONDecodeError:
                    st.warning(f"Line {line_num}: JSON 파싱 오류")
                    continue
        
        return passages, metadata
        
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {jsonl_path}")
        return [], []

def save_language_index(faiss_index, passages, metadata, faiss_path, passages_path):
    """언어별 임베딩과 인덱스를 캐시에 저장"""
    try:
        # 디렉토리 생성
        config.DATA_DIR.mkdir(exist_ok=True)
        
        # FAISS 인덱스 저장
        faiss.write_index(faiss_index, str(faiss_path))
        
        # 패시지와 메타데이터 저장
        with open(passages_path, 'wb') as f:
            pickle.dump({
                'passages': passages,
                'metadata': metadata
            }, f)
        
        st.success("📁 인덱스가 캐시에 저장되었습니다.")
        
    except Exception as e:
        st.warning(f"캐시 저장 실패: {str(e)}")

def search_similar_passages(embedding_model, faiss_index, passages, query: str, k: int = None):
    """
    유사한 법률 조문 검색 (언어별 인덱스 사용)
    
    Args:
        embedding_model: 임베딩 모델
        faiss_index: 해당 언어의 FAISS 인덱스
        passages: 해당 언어의 원본 텍스트들
        query (str): 검색 쿼리
        k (int): 반환할 문서 수
    
    Returns:
        list: 검색된 문서들
    """
    if k is None:
        k = config.MAX_RETRIEVED_DOCS
    
    try:
        # 쿼리 임베딩
        query_embedding = embedding_model.encode([query])
        
        # 유사도 검색
        distances, indices = faiss_index.search(query_embedding.astype('float32'), k)
        
        # 결과 반환
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(passages):
                retrieved_docs.append({
                    'text': passages[idx],
                    'score': float(distances[0][i]),
                    'index': int(idx)
                })
        
        return retrieved_docs
        
    except Exception as e:
        st.error(f"검색 실패: {str(e)}")
        return []

def load_prebuilt_indexes():
    """사전 구축된 인덱스들을 즉시 로드"""
    try:
        # 임베딩 모델
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # 중국어 인덱스 로드
        cn_index = faiss.read_index(str(config.CN_FAISS_INDEX_PATH))
        with open(config.CN_PASSAGES_PATH, 'rb') as f:
            cn_data = pickle.load(f)
            cn_passages = cn_data['passages']
            cn_metadata = cn_data['metadata']
        
        # 베트남어 인덱스 로드
        vn_index = faiss.read_index(str(config.VN_FAISS_INDEX_PATH))
        with open(config.VN_PASSAGES_PATH, 'rb') as f:
            vn_data = pickle.load(f)
            vn_passages = vn_data['passages']
            vn_metadata = vn_data['metadata']
        
        return embedding_model, cn_index, cn_passages, cn_metadata, vn_index, vn_passages, vn_metadata
        
    except Exception as e:
        st.error(f"사전 구축된 인덱스 로드 실패: {str(e)}")
        return None, None, None, None, None, None, None
    if k is None:
        k = config.MAX_RETRIEVED_DOCS
    
    try:
        # 쿼리 임베딩
        query_embedding = embedding_model.encode([query])
        
        # 유사도 검색
        distances, indices = faiss_index.search(query_embedding.astype('float32'), k)
        
        # 결과 반환
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(passages):
                retrieved_docs.append({
                    'text': passages[idx],
                    'score': float(distances[0][i]),
                    'index': int(idx)
                })
        
        return retrieved_docs
        
    except Exception as e:
        st.error(f"검색 실패: {str(e)}")
        return []
