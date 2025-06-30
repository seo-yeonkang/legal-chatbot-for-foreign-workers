# =============================================================================
# app.py - 메인 Streamlit 앱
# =============================================================================

import streamlit as st
import os
from langdetect import detect, DetectorFactory
from pathlib import Path
import sys
import traceback

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 언어 감지 시드 고정 (일관성을 위해)
DetectorFactory.seed = 0

# 설정 및 유틸리티 임포트
import config
from utils import (
    load_tokenizer,
    load_embeddings_and_index,
    build_prompt,
    generate_answer
)
from utils.embedding_index import search_similar_passages
from utils.generator import load_generation_models

# Streamlit 페이지 설정
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .legal-docs {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown(f"""
<div class="main-header">
    <h1>{config.PAGE_ICON} {config.PAGE_TITLE}</h1>
    <p>중국어와 베트남어를 지원하는 AI 법률 상담 서비스</p>
</div>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.markdown("### 📋 사용 방법")
    st.markdown("""
    1. 중국어 🇨🇳 또는 베트남어 🇻🇳로 질문 입력
    2. 관련 법률 조문 자동 검색
    3. AI가 법률 조문에 기반한 답변 제공
    """)
    
    st.markdown("### ⚠️ 주의사항")
    st.markdown("""
    - 본 서비스는 참고용입니다
    - 중요한 법률 문제는 전문가와 상담하세요
    - 답변의 정확성을 보장하지 않습니다
    """)
    
    st.markdown("### 🌐 지원 언어")
    st.markdown("- 🇨🇳 중국어 (Chinese)")
    st.markdown("- 🇻🇳 베트남어 (Vietnamese)")

# 메인 콘텐츠
def main():
    # 시스템 초기화
    if 'initialized' not in st.session_state:
        with st.spinner("🔄 시스템을 초기화하고 있습니다..."):
            try:
                # 언어별 임베딩 시스템 로드
                (embed_model, cn_index, cn_passages, cn_metadata, 
                 vn_index, vn_passages, vn_metadata) = load_embeddings_and_index()
                
                if embed_model is None:
                    st.error("❌ 임베딩 시스템 로드에 실패했습니다.")
                    st.stop()
                
                # 생성 모델 로드
                chinese_model, vietnamese_model = load_generation_models()
                
                if chinese_model is None or vietnamese_model is None:
                    st.error("❌ 생성 모델 로드에 실패했습니다.")
                    st.stop()
                
                # 세션 상태에 저장 (언어별 분리)
                st.session_state.embed_model = embed_model
                st.session_state.cn_index = cn_index
                st.session_state.cn_passages = cn_passages
                st.session_state.cn_metadata = cn_metadata
                st.session_state.vn_index = vn_index
                st.session_state.vn_passages = vn_passages
                st.session_state.vn_metadata = vn_metadata
                st.session_state.chinese_model = chinese_model
                st.session_state.vietnamese_model = vietnamese_model
                st.session_state.initialized = True
                
                # 로드된 데이터 확인
                cn_count = len(cn_passages) if cn_passages else 0
                vn_count = len(vn_passages) if vn_passages else 0
                st.success(f"✅ 시스템 초기화 완료! (🇨🇳 {cn_count}개, 🇻🇳 {vn_count}개 문서)")
                
            except Exception as e:
                st.error(f"❌ 초기화 실패: {str(e)}")
                st.error(f"상세 오류: {traceback.format_exc()}")
                st.stop()
    
    # 질문 입력
    st.markdown("### 💬 질문을 입력하세요")
    question = st.text_input(
        "",
        placeholder="중국어 또는 베트남어로 법률 관련 질문을 입력하세요...",
        key="question_input"
    )
    
    # 예시 질문들
    st.markdown("#### 📝 예시 질문")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🇨🇳 中文示例**")
        if st.button("我可以在韩国工作多长时间？", key="zh_example"):
            st.session_state.question_input = "我可以在韩国工作多长时间？"
            st.rerun()
    
    with col2:
        st.markdown("**🇻🇳 Tiếng Việt**")
        if st.button("Tôi có thể làm việc ở Hàn Quốc trong bao lâu?", key="vi_example"):
            st.session_state.question_input = "Tôi có thể làm việc ở Hàn Quốc trong bao lâu?"
            st.rerun()
    
    # 질문 처리
    if question and question.strip():
        process_question(question.strip())

def process_question(question: str):
    """질문 처리 및 답변 생성"""
    try:
        # 언어 감지
        detected_lang = detect(question)
        
        # 지원되는 언어 확인
        if detected_lang not in ['zh', 'vi']:
            st.warning("⚠️ 중국어 또는 베트남어로만 질문해주세요.")
            return
        
        # 언어 표시
        lang_display = {"zh": "🇨🇳 중국어", "vi": "🇻🇳 베트남어"}
        st.info(f"감지된 언어: {lang_display.get(detected_lang, detected_lang)}")
        
        # 언어에 따른 인덱스 및 패시지 선택
        if detected_lang == "zh":
            faiss_index = st.session_state.cn_index
            passages = st.session_state.cn_passages
            metadata = st.session_state.cn_metadata
        else:  # detected_lang == "vi"
            faiss_index = st.session_state.vn_index  
            passages = st.session_state.vn_passages
            metadata = st.session_state.vn_metadata
        
        # 해당 언어 데이터가 있는지 확인
        if faiss_index is None or passages is None:
            lang_name = "중국어" if detected_lang == "zh" else "베트남어"
            st.error(f"❌ {lang_name} 법률 데이터가 로드되지 않았습니다.")
            return
        
        # 관련 법률 조문 검색
        with st.spinner("🔍 관련 법률 조문을 검색하고 있습니다..."):
            retrieved_docs = search_similar_passages(
                st.session_state.embed_model,
                faiss_index,
                passages,
                question,
                k=config.MAX_RETRIEVED_DOCS
            )
        
        if not retrieved_docs:
            st.warning("관련 법률 조문을 찾을 수 없습니다.")
            return
        
        # 검색된 법률 조문 표시
        with st.expander("🔍 관련 법률 조문", expanded=True):
            st.markdown('<div class="legal-docs">', unsafe_allow_html=True)
            for i, doc in enumerate(retrieved_docs, 1):
                score = doc.get('score', 0)
                st.markdown(f"**{i}. 법조문** (유사도: {score:.3f})")
                st.markdown(f"{doc['text']}")
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 토크나이저 로드
        tokenizer = load_tokenizer(detected_lang)
        if tokenizer is None:
            st.error("토크나이저 로드에 실패했습니다.")
            return
        
        # 프롬프트 구성
        prompt = build_prompt(question, retrieved_docs, detected_lang)
        
        # 모델 선택
        model = (st.session_state.chinese_model if detected_lang == "zh" 
                else st.session_state.vietnamese_model)
        
        # 답변 생성
        with st.spinner("🤖 답변을 생성하고 있습니다..."):
            answer = generate_answer(prompt, model, tokenizer)
        
        # 답변 표시
        st.markdown("### 💡 AI 법률 상담 답변")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 추가 안내
        st.markdown("---")
        st.markdown("**⚠️ 면책조항:** 본 답변은 AI가 생성한 참고용 정보입니다. 정확한 법률 상담은 전문가와 상의하시기 바랍니다.")
        
    except Exception as e:
        st.error(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
        with st.expander("상세 오류 정보"):
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
