# memory_store.py
# 將會議 JSON 存入 Chroma 向量資料庫
# 版本：適合 Streamlit Community Cloud，使用 Gemini Embedding，不需要本機 Ollama

import json
import os
import uuid
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "meeting_memory"


def get_gemini_api_key() -> str:
    """本機讀 .env；部署到 Streamlit Cloud 時讀 st.secrets。"""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass

    raise ValueError("找不到 GEMINI_API_KEY，請確認本機 .env 或 Streamlit Secrets 是否設定完成。")


def load_meeting_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_gemini_api_key()
    )


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )


def meeting_to_documents(meeting: dict) -> List[Document]:
    """支援 RAG.zip 原格式，也支援 Gemini MeetingRecord 轉出的格式。"""
    meeting_date = meeting.get("date") or meeting.get("meeting_date") or "unknown"
    title = meeting.get("title") or meeting.get("meeting_title") or "未命名會議"

    docs: List[Document] = []

    # 1. 摘要
    summary = meeting.get("summary") or meeting.get("one_sentence_summary") or ""
    summary_points = meeting.get("summary_points", [])

    summary_parts = []
    if summary:
        summary_parts.append(summary)
    if summary_points:
        summary_parts.append("會議重點：")
        summary_parts.extend([f"- {point}" for point in summary_points])

    if summary_parts:
        docs.append(Document(
            page_content="\n".join(summary_parts),
            metadata={
                "date": meeting_date,
                "title": title,
                "type": "summary"
            }
        ))

    # 2. 會議決議
    for decision in meeting.get("decisions", []):
        decision_text = decision.get("decision", "") if isinstance(decision, dict) else str(decision)
        evidence = decision.get("evidence", "") if isinstance(decision, dict) else ""

        docs.append(Document(
            page_content=f"會議決議：{decision_text}\n依據：{evidence}",
            metadata={
                "date": meeting_date,
                "title": title,
                "type": "decision"
            }
        ))

    # 3. 待辦事項
    for task in meeting.get("tasks", []):
        if isinstance(task, dict):
            owner = task.get("owner", "未指定")
            task_content = task.get("task", "")
            deadline = task.get("deadline") or "未指定"
            status = task.get("status", "待確認")
            evidence = task.get("evidence", "")
        else:
            owner = "未指定"
            task_content = str(task)
            deadline = "未指定"
            status = "待確認"
            evidence = ""

        docs.append(Document(
            page_content=(
                f"負責人：{owner}\n"
                f"任務內容：{task_content}\n"
                f"期限：{deadline}\n"
                f"狀態：{status}\n"
                f"依據：{evidence}"
            ),
            metadata={
                "date": meeting_date,
                "title": title,
                "type": "task",
                "owner": owner,
                "deadline": deadline,
                "status": status
            }
        ))

    # 4. 問題與風險
    for problem in meeting.get("risks_or_problems", []):
        docs.append(Document(
            page_content=f"問題或風險：{problem}",
            metadata={
                "date": meeting_date,
                "title": title,
                "type": "risk"
            }
        ))

    # 5. 下一步行動
    for step in meeting.get("next_steps", []):
        docs.append(Document(
            page_content=f"下一步行動：{step}",
            metadata={
                "date": meeting_date,
                "title": title,
                "type": "next_step"
            }
        ))

    # 6. 逐字稿切塊
    transcript = meeting.get("transcript", "")
    if transcript.strip():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )

        chunks = splitter.create_documents(
            [transcript],
            metadatas=[{
                "date": meeting_date,
                "title": title,
                "type": "transcript"
            }]
        )
        docs.extend(chunks)

    return docs


def add_meeting(json_path: str) -> int:
    """對外接口：傳入 JSON 路徑，自動存入 Chroma 知識庫，回傳新增文件數。"""
    meeting = load_meeting_json(json_path)
    docs = meeting_to_documents(meeting)

    if not docs:
        return 0

    db = get_vectorstore()
    ids = [f"meeting-{uuid.uuid4().hex}" for _ in docs]
    db.add_documents(docs, ids=ids)

    return len(docs)


if __name__ == "__main__":
    count = add_meeting("meetings/meeting_001.json")
    print(f"已存入 {count} 筆文件。")
