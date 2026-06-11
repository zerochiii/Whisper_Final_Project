# rag_query.py
# 從 Chroma 會議知識庫檢索資料，並用 Gemini 回答
# 版本：適合 Streamlit Community Cloud，不需要本機 Ollama

from google import genai

from memory_store import get_vectorstore, get_gemini_api_key


PROMPT_TEMPLATE = """
你是一個專題會議助手，負責根據歷史會議紀錄回答問題。

請遵守以下規則：
1. 只能根據提供的歷史會議資料回答。
2. 不可以自行編造負責人、期限或決議。
3. 如果找不到相關資訊，請回答「這個問題在歷史會議中沒有記錄」。
4. 請使用繁體中文。
5. 回答時盡量附上會議日期、會議標題或依據。

歷史會議資料：
{context}

使用者問題：
{question}

請根據以上資料回答：
"""


DEFAULT_QUESTIONS = [
    "誰負責前端開發？",
    "上次會議決定了什麼？",
    "模型測試的進度如何？",
    "誰負責 Gemini API？",
]


def format_docs(docs) -> str:
    if not docs:
        return ""

    formatted = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        formatted.append(
            f"[資料 {index}]\n"
            f"日期：{metadata.get('date', 'unknown')}\n"
            f"標題：{metadata.get('title', '')}\n"
            f"類型：{metadata.get('type', 'unknown')}\n"
            f"內容：{doc.page_content}"
        )

    return "\n\n".join(formatted)


def query(question: str) -> str:
    if not question.strip():
        return "請先輸入問題。"

    db = get_vectorstore()
    docs = db.similarity_search(question, k=4)

    if not docs:
        return "這個問題在歷史會議中沒有記錄。"

    context = format_docs(docs)

    client = genai.Client(api_key=get_gemini_api_key())

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    print("=== AI 專題會議助手 RAG 查詢 ===")
    for i, q in enumerate(DEFAULT_QUESTIONS, start=1):
        print(f"{i}. {q}")

    while True:
        user_input = input("請輸入問題，或輸入 q 離開：").strip()

        if user_input.lower() == "q":
            break

        if user_input in ("1", "2", "3", "4"):
            user_input = DEFAULT_QUESTIONS[int(user_input) - 1]

        print(query(user_input))
