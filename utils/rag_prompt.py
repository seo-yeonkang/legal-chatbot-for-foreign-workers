# =============================================================================
# utils/rag_prompt.py - RAG 프롬프트 구성
# =============================================================================


def build_prompt(question: str, docs: list[dict], lang: str) -> str:
    """
    판례 기반 RAG 프롬프트
    ─────────────────────────────────────────────────────
    • 역할(Role)      : 한국 근로자 대상 노동법 상담 AI
    • 컨텍스트(Context): 유사 판례 1–2건 (사건번호·선고일 포함)
    • 출력(Output)    : ① 상황 요약 → ② 판례 핵심 → ③ 실무적 조언
    """

    # ── 1. 판례 문단 포맷 ─────────────────────────────
    case_lines = []
    for i, d in enumerate(docs[:2], 1):                         # k ≤ 2
        txt = d["text"].strip()
        meta = d.get("metadata", {})                             # 사건번호·선고일 등
        case_tag = meta.get("case_no", "") or meta.get("Unique_num", "")
        date_tag = meta.get("date", "")
        header = f"【판례 {i} {case_tag} {date_tag}】"
        case_lines.append(f"{header}\n{txt[:250]}")              # 250자 이내

    joined_cases = "\n\n".join(case_lines)

    # ── 2. 언어별 틀 ─────────────────────────────
    if lang == "zh":
        role = "你是一位面向在韩工作的劳动者，精通韩国劳动法及相关判例的法律顾问。"
        output_rule = (
            "请遵守以下格式：\n"
            "1. 【问题背景简述】(1–2 句)\n"
            "2. 【相关判例】逐条列出要点，并在括号中写明判例编号\n"
            "3. 【可行建议】给出实际可操作的建议，必要时提示咨询专业律师\n"
            "回答字数约 5–7 句，不要透露内部推理过程。"
        )
    else:  # vi
        role = "Bạn là cố vấn pháp luật chuyên về lao động tại Hàn Quốc, sử dụng các phán quyết làm cơ sở tham khảo."
        output_rule = (
            "Hãy trả lời theo cấu trúc:\n"
            "1. 【Tóm tắt tình huống】(1–2 câu)\n"
            "2. 【Phán quyết liên quan】 liệt kê ngắn gọn, ghi số vụ án trong ngoặc\n"
            "3. 【Khuyến nghị thực tế】 nêu bước tiếp theo; khuyên liên hệ luật sư khi cần\n"
            "Giữ độ dài 5–7 câu, không tiết lộ suy luận nội bộ."
        )

    # ── 3. 최종 프롬프트 ─────────────────────────────
    prompt = (
        f"<system>\n{role}\n</system>\n\n"
        f"<assistant>\n아래는 유사 판례 요약입니다:\n{joined_cases}\n</assistant>\n\n"
        f"<user>\n{question}\n</user>\n\n"
        f"<assistant>\n{output_rule}\n"
    )
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
