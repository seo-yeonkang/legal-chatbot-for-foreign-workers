# =============================================================================
# utils/rag_prompt.py - RAG 프롬프트 구성
# =============================================================================

def build_prompt(question: str, retrieved_docs: list, language: str):
    """
    RAG 프롬프트 구성
    
    Args:
        question (str): 사용자 질문
        retrieved_docs (list): 검색된 법률 조문들
        language (str): 언어 ('zh' 또는 'vi')
    
    Returns:
        str: 구성된 프롬프트
    """
    # 문서 텍스트 추출
    if isinstance(retrieved_docs[0], dict):
        doc_texts = [doc['text'] for doc in retrieved_docs]
    else:
        doc_texts = retrieved_docs
    
    # 법률 조문 포맷팅
    formatted_docs = []
    for i, doc in enumerate(doc_texts, 1):
        formatted_docs.append(f"{i}. {doc.strip()}")
    
    joined_docs = "\n".join(formatted_docs)
    
    if language == "zh":
        prompt = f"""请根据以下相关的韩国法律条文，用中文回答用户的法律咨询问题。请提供准确、实用的法律建议。

相关法律条文：
{joined_docs}

用户问题：{question}

请根据上述法律条文提供详细的中文回答："""

    elif language == "vi":
        prompt = f"""Hãy dựa vào các điều luật Hàn Quốc liên quan dưới đây để trả lời câu hỏi tư vấn pháp luật của người dùng bằng tiếng Việt. Vui lòng cung cấp lời khuyên pháp lý chính xác và thực tế.

Các điều luật liên quan:
{joined_docs}

Câu hỏi của người dùng: {question}

Hãy cung cấp câu trả lời chi tiết bằng tiếng Việt dựa trên các điều luật trên:"""

    else:
        # 기본 영어 프롬프트
        prompt = f"""Based on the following relevant Korean legal articles, please answer the user's legal consultation question. Provide accurate and practical legal advice.

Relevant Legal Articles:
{joined_docs}

User Question: {question}

Please provide a detailed answer based on the above legal articles:"""
    
    return prompt

def build_simple_prompt(question: str, language: str):
    """
    단순 질문용 프롬프트 (검색 결과 없을 때)
    
    Args:
        question (str): 사용자 질문
        language (str): 언어
    
    Returns:
        str: 구성된 프롬프트
    """
    if language == "zh":
        return f"""请回答以下法律咨询问题：

问题：{question}

请提供有帮助的法律建议："""

    elif language == "vi":
        return f"""Hãy trả lời câu hỏi tư vấn pháp luật sau:

Câu hỏi: {question}

Vui lòng cung cấp lời khuyên pháp lý hữu ích:"""

    else:
        return f"""Please answer the following legal consultation question:

Question: {question}

Please provide helpful legal advice:"""
