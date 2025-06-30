# =============================================================================
# app.py - 메인 Streamlit 앱
# =============================================================================

import streamlit as st
import os
from langdetect import detect, DetectorFactory
from pathlib import Path
import sys
import traceback
import torch

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
    generate_answer,
    common
)
from utils.common import mark_deployment_ready
from utils.embedding_index import search_similar_passages, is_deployment_ready
from utils.generator import load_generation_models
import pickle
import faiss

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
    
    # 배포 모드 상태 표시
    st.markdown("---")
    st.markdown("### ⚡ 시스템 상태")
    
    # Streamlit Cloud 특별 표시
    if config.STREAMLIT_CLOUD:
        st.info("☁️ Streamlit Cloud")
        st.caption("경량 모델 사용 중")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💾 메모리", "제한됨")
        with col2:
            st.metric("🚀 모드", "경량화")
        
        with st.expander("ℹ️ Streamlit Cloud 정보"):
            st.markdown("""
            - **메모리 제한**: 1GB
            - **경량 모델**: 성능 최적화됨
            - **무료 호스팅**: 24/7 서비스
            """)
            
    elif is_deployment_ready():
        # 프로덕션 모드
        st.success("🚀 프로덕션 모드")
        st.info("⚡ 사전 구축 완료")
        st.metric("🎯 모드", "즉시 시작", help="모든 모델이 사전 구축되어 즉시 시작")
        
        # 배포 정보
        marker_file = config.BASE_DIR / ".deployment_ready"
        if marker_file.exists():
            try:
                import json
                with open(marker_file, 'r') as f:
                    data = json.load(f)
                    setup_version = data.get("setup_version", "unknown")
                    st.caption(f"Setup v{setup_version}")
            except:
                pass
    else:
        # 개발 모드
        st.warning("🔨 개발 모드")
        st.info("📦 런타임 구축")
        
        if 'app_fully_initialized' in st.session_state:
            st.success("✅ 캐시 활성화됨")
            
            # 캐시 초기화 버튼 (개발 모드에서만)
            if st.button("🔄 캐시 초기화", help="문제 발생시에만 사용하세요"):
                # 모든 캐시 클리어
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.cache_resource.clear()
                st.rerun()
        else:
            st.warning("⏳ 초기화 중...")
        
        # 개발 도구
        with st.expander("🛠️ 개발 도구"):
            st.markdown("**프로덕션 모드로 전환하려면:**")
            st.code("python setup_models.py", language="bash")
            st.markdown("실행 후 앱을 재시작하세요.")

    
    # 성능 정보
    st.markdown("### 📊 성능 정보")
    
    if config.STREAMLIT_CLOUD:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⚡ 시작 시간", "< 10초")
        with col2:
            st.metric("🧠 모델", "경량화")
    elif is_deployment_ready():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🚀 시작 시간", "< 5초")
        with col2:
            st.metric("💾 저장 공간", "최적화됨")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if 'app_fully_initialized' in st.session_state:
                st.metric("⚡ 시작 시간", "10-15초")
            else:
                st.metric("⏳ 첫 시작", "2-3분")
        with col2:
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.metric("💻 처리 장치", device)
    
    # 저장 공간 정보
    if 'app_fully_initialized' in st.session_state and not config.STREAMLIT_CLOUD:
        st.markdown("### 📊 데이터 현황")
        model_path = config.CHINESE_MODEL_LOCAL_PATH
        if model_path.exists():
            try:
                size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024*1024)
                st.info(f"💾 모델 크기: {size_mb:.1f}MB")
            except:
                st.info("💾 모델이 저장됨")
        
        # 인덱스 상태
        if (config.CN_FAISS_INDEX_PATH.exists() and config.VN_FAISS_INDEX_PATH.exists()):
            st.info("🔍 검색 인덱스: 준비됨")

# 메인 콘텐츠
def main():
    # 시스템 초기화 (배포 모드 vs 개발 모드)
    if 'app_fully_initialized' not in st.session_state:
        
        # 배포 완료 상태 확인
        if is_deployment_ready():
            # 🚀 프로덕션 모드: 즉시 로드
            with st.spinner("⚡ 프로덕션 모드: 시스템 즉시 로드 중..."):
                
                # 임베딩 시스템 즉시 로드
                (embed_model, cn_index, cn_passages, cn_metadata, 
                 vn_index, vn_passages, vn_metadata) = load_embeddings_and_index()
                
                if embed_model is None:
                    st.error("❌ 임베딩 시스템 로드에 실패했습니다.")
                    st.stop()
                
                # 생성 모델 즉시 로드
                chinese_model, vietnamese_model = load_generation_models()
                
                if chinese_model is None or vietnamese_model is None:
                    st.error("❌ 생성 모델 로드에 실패했습니다.")
                    st.stop()
                
                # 로드된 데이터 확인
                cn_count = len(cn_passages) if cn_passages else 0
                vn_count = len(vn_passages) if vn_passages else 0
                
                # 성공 메시지 (프로덕션 모드)
                st.success(f"""
                🚀 **프로덕션 모드: 즉시 시작!**
                - ⚡ 사전 구축된 모델 로드 완료
                - 🇨🇳 중국어 법률 문서: {cn_count}개
                - 🇻🇳 베트남어 법률 문서: {vn_count}개
                """)

                st.session_state.embeddings_ready  = True
                st.session_state.generation_ready  = True
                st.session_state.chinese_model     = chinese_model
                st.session_state.vietnamese_model  = vietnamese_model
                            
                from utils.common import mark_deployment_ready
                mark_deployment_ready() 
    
        else:
            # 🔨 개발 모드: 기존 방식 (단계별 로드)
            with st.spinner("🔨 개발 모드: 시스템을 단계별로 초기화합니다..."):
                
                # 1단계: 임베딩 시스템 로드
                st.info("📚 1/2 단계: 법률 데이터베이스 준비 중...")
                (embed_model, cn_index, cn_passages, cn_metadata, 
                 vn_index, vn_passages, vn_metadata) = load_embeddings_and_index()
                
                if embed_model is None:
                    st.error("❌ 임베딩 시스템 로드에 실패했습니다.")
                    st.stop()
                
                # 2단계: 생성 모델 로드
                st.info("🤖 2/2 단계: AI 모델 준비 중...")
                chinese_model, vietnamese_model = load_generation_models()
                
                if chinese_model is None or vietnamese_model is None:
                    st.error("❌ 생성 모델 로드에 실패했습니다.")
                    st.stop()
                
                # 로드된 데이터 확인
                cn_count = len(cn_passages) if cn_passages else 0
                vn_count = len(vn_passages) if vn_passages else 0
                
                # 성공 메시지 (개발 모드)
                st.balloons()  # 축하 효과!
                st.success(f"""
                🎉 **개발 모드: 초기화 완료!**
                - 🇨🇳 중국어 법률 문서: {cn_count}개
                - 🇻🇳 베트남어 법률 문서: {vn_count}개
                - 💡 프로덕션 배포시엔 즉시 시작됩니다!
                """)
        
        # 완전 초기화 마크
        st.session_state.app_fully_initialized = True
    
    else:
        # 이미 초기화됨 - 상태 표시만
        if is_deployment_ready():
            st.success("🚀 프로덕션 모드: 법률 챗봇 준비 완료!")
        else:
            st.success("⚡ 개발 모드: 법률 챗봇 준비 완료! (캐시 사용)")
        
        # 간단한 상태 확인
        cn_count = len(st.session_state.cn_passages) if hasattr(st.session_state, 'cn_passages') and st.session_state.cn_passages else 0
        vn_count = len(st.session_state.vn_passages) if hasattr(st.session_state, 'vn_passages') and st.session_state.vn_passages else 0
        
        if cn_count > 0 or vn_count > 0:
            st.info(f"📊 사용 가능: 🇨🇳 {cn_count}개, 🇻🇳 {vn_count}개 법률 문서")
  
    
    
    # 질문 입력
    st.markdown("### 💬 질문을 입력하세요")
    def set_example(text):
        st.session_state.question_input = text      # <- 콜백 내부에서 안전

    st.markdown("#### 📝 예시 질문")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🇨🇳 中文示例**")
        st.button(
            "我可以在韩国工作多长时间？",
            key="zh_example",
            on_click=set_example,
            args=("我可以在韩国工作多长时间？",)
        )
    
    with col2:
        st.markdown("**🇻🇳 Tiếng Việt**")
        st.button(
            "Tôi có thể làm việc ở Hàn Quốc trong bao lâu?",
            key="vi_example",
            on_click=set_example,
            args=("Tôi có thể làm việc ở Hàn Quốc trong bao lâu?",)
        )
    
    # ▶︎ 이제 텍스트 입력 위젯을 **버튼 아래** 또는 위젯 생성 뒤 값 읽기만
    question = st.text_input(
        "질문",
        key="question_input",
        label_visibility="collapsed",
        placeholder="중국어 또는 베트남어로 법률 관련 질문을 입력하세요..."
    )
        
    # 질문 처리
    if st.button("질문하기") and question.strip():
        process_question(question.strip())

from langdetect import detect, DetectorFactory
import regex as re
DetectorFactory.seed = 0      # 이미 있음


def safe_detect(text: str) -> str:
    """짧은·혼합 문장에 대한 보정 포함 중국어/베트남어 감지"""
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    # --- 보정 ① : 중국어 글자 존재하면 강제 zh ---
    if re.search(r"\p{Han}", text):
        return "zh"

    # --- 보정 ② : 베트남어 특수 문자 존재하면 강제 vi ---
    if re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]", text):
        return "vi"

    return lang


def process_question(question: str):
    """질문 처리 및 답변 생성"""
    if 'vn_index' not in st.session_state or 'cn_index' not in st.session_state:
        st.warning("🔄 시스템이 아직 완전히 초기화되지 않았습니다. 잠시만 기다려 주세요.")
    return
    
    try:
        # 언어 감지
        detected_lang = safe_detect(question)
        
        # 지원되는 언어 확인
        if detected_lang not in ['zh', 'vi']:
            st.warning("⚠️ 중국어 또는 베트남어로만 질문해주세요.")
            return
        
        # 언어 표시
        lang_display = {"zh": "🇨🇳 중국어", "vi": "🇻🇳 베트남어"}
        st.info(f"감지된 언어: {lang_display.get(detected_lang, detected_lang)}")
        
        # 언어에 따른 인덱스 및 패시지 선택 (캐시에서 직접 가져오기)
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
        
        # 관련 법률 조문 검색 (더 빠른 검색)
        with st.spinner("🔍 관련 법률 조문 검색 중..."):
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
        
        # 모델 선택 (캐시에서 직접 가져오기)
        model = (st.session_state.chinese_model if detected_lang == "zh" 
                else st.session_state.vietnamese_model)
        
        # 답변 생성 (더 빠른 생성)
        with st.spinner("🤖 AI 답변 생성 중..."):
            answer = generate_answer(prompt, model, tokenizer)
        
        # 답변 표시
        st.markdown("### 💡 AI 법률 상담 답변")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 추가 안내
        st.markdown("---")
        st.markdown("**⚠️ 면책조항:** 본 답변은 AI가 생성한 참고용 정보입니다. 정확한 법률 상담은 전문가와 상의하시기 바랍니다.")
        
        # 성능 정보 (간단하게)
        if 'app_fully_initialized' in st.session_state:
            with st.expander("📊 시스템 성능 정보"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🚀 모드", "고속 캐시")
                with col2:
                    device = "GPU" if torch.cuda.is_available() else "CPU"
                    st.metric("💻 처리 장치", device)
                with col3:
                    docs_count = len(retrieved_docs) if retrieved_docs else 0
                    st.metric("📄 검색 문서", f"{docs_count}개")
        
    except Exception as e:
        st.error(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
        with st.expander("상세 오류 정보"):
            st.text(traceback.format_exc())
        
        # 빠른 해결책 제안
        st.info("💡 **빠른 해결책**: 사이드바의 '캐시 초기화' 버튼을 클릭해보세요.")

if __name__ == "__main__":
    main()
