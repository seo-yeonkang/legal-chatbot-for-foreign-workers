# =============================================================================
# utils/embedding_index.py - 임베딩 및 FAISS 인덱스
# =============================================================================

from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
import config

@st.cache_resource
def load_embeddings_and_index():
    """
    언어별 임베딩 모델과 FAISS 인덱스 로드
    
    Returns:
        tuple: (embedding_model, cn_index, cn_passages, cn_metadata, vn_index, vn_passages, vn_metadata)
    """
    try:
        # 임베딩 모델 로드 (공통)
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # 중국어 인덱스 로드
        cn_index, cn_passages, cn_metadata = load_language_index(
            embedding_model, 
            'zh',
            config.CN_LAW_DATA_PATH,
            config.CN_FAISS_INDEX_PATH,
            config.CN_PASSAGES_PATH
        )
        
        # 베트남어 인덱스 로드
        vn_index, vn_passages, vn_metadata = load_language_index(
            embedding_model,
            'vi', 
            config.VN_LAW_DATA_PATH,
            config.VN_FAISS_INDEX_PATH,
            config.VN_PASSAGES_PATH
        )
        
        return embedding_model, cn_index, cn_passages, cn_metadata, vn_index, vn_passages, vn_metadata
        
    except Exception as e:
        st.error(f"임베딩 시스템 로드 실패: {str(e)}")
        return None, None, None, None, None, None, None

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
