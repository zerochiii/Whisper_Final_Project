import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


# =========================
# 1. 定義固定輸出格式
# =========================

class Task(BaseModel):
    owner: str = Field(description="負責人；若無法判斷，填寫「未指定」")
    task: str = Field(description="待辦事項內容")
    deadline: Optional[str] = Field(description="期限；若逐字稿沒有提到，填 null")
    status: str = Field(description="狀態，例如：未開始、進行中、已完成、待確認")
    evidence: str = Field(description="逐字稿中可支持此待辦事項的簡短依據")


class Decision(BaseModel):
    decision: str = Field(description="本次會議做出的決議")
    evidence: str = Field(description="逐字稿中的依據")


class MeetingRecord(BaseModel):
    meeting_title: str = Field(description="會議標題")
    meeting_date: Optional[str] = Field(description="會議日期，格式 YYYY-MM-DD；若無法判斷填 null")
    one_sentence_summary: str = Field(description="一句話總結本次會議")
    summary_points: List[str] = Field(description="會議重點摘要，3 到 6 點")
    discussed_topics: List[str] = Field(description="本次會議討論的主題")
    decisions: List[Decision] = Field(description="本次會議決議")
    tasks: List[Task] = Field(description="待辦事項列表")
    risks_or_problems: List[str] = Field(description="會議中提到的問題、風險或卡關點")
    next_steps: List[str] = Field(description="下一步行動")
    keywords: List[str] = Field(description="方便後續 RAG 檢索的關鍵字")


# =========================
# 2. 讀取逐字稿
# =========================

def read_transcript(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到逐字稿檔案：{file_path}")

    return path.read_text(encoding="utf-8")


# =========================
# 3. 建立 Prompt
# =========================

def build_prompt(transcript: str) -> str:
    return f"""
你是一位專題會議紀錄整理助手，請根據使用者提供的會議逐字稿，整理成固定格式的會議紀錄。

請遵守以下規則：

1. 只根據逐字稿內容整理，不可以自行編造未出現的資訊。
2. 如果逐字稿中沒有明確提到負責人，請填寫「未指定」。
3. 如果逐字稿中沒有明確提到期限，請填寫 null。
4. 請使用繁體中文。
5. 專有名詞請保留原文，例如 Faster-Whisper、Gemini API、RAG、LangChain、Streamlit、Chroma、FAISS。
6. 摘要要簡潔，重點放在本次會議討論內容、決議、進度與後續工作。
7. 待辦事項必須整理出負責人、任務內容、期限、狀態與依據。
8. 狀態請從「未開始、進行中、已完成、待確認」中選擇最適合的一個。
9. 不要輸出 Markdown，不要加上額外說明，只輸出符合指定結構的 JSON。

會議逐字稿如下：

{transcript}
"""


# =========================
# 4. 呼叫 Gemini API
# =========================
def get_gemini_api_key() -> str:
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

def analyze_with_gemini(transcript: str) -> MeetingRecord:
    api_key = get_gemini_api_key()

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(transcript)


    client = genai.Client(api_key=api_key)
    prompt = build_prompt(transcript)

    # 先用品質較好的 flash，失敗時改用 flash-lite
    models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    last_error = None

    for model_name in models:
        for attempt in range(3):
            try:
                print(f"正在使用模型：{model_name}，第 {attempt + 1} 次嘗試")

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": MeetingRecord,
                    }
                )

                if response.parsed:
                    return response.parsed

                return MeetingRecord.model_validate_json(response.text)

            except Exception as e:
                last_error = e
                error_message = str(e)

                # 503 / high demand 通常是暫時性錯誤，等待後重試
                if "503" in error_message or "UNAVAILABLE" in error_message or "high demand" in error_message:
                    wait_seconds = 2 * (attempt + 1)
                    print(f"模型暫時忙碌，等待 {wait_seconds} 秒後重試...")
                    time.sleep(wait_seconds)
                    continue

                # 不是 503 的錯誤就直接丟出
                raise e

    raise RuntimeError(f"Gemini API 多次重試後仍失敗：{last_error}")

# =========================
# 5. 轉成 Markdown 會議紀錄
# =========================

def record_to_markdown(record: MeetingRecord) -> str:
    summary_text = "\n".join(f"- {point}" for point in record.summary_points)
    topics_text = "\n".join(f"- {topic}" for topic in record.discussed_topics)
    problems_text = "\n".join(f"- {problem}" for problem in record.risks_or_problems)
    next_steps_text = "\n".join(f"- {step}" for step in record.next_steps)

    decision_lines = []
    for index, decision in enumerate(record.decisions, start=1):
        decision_lines.append(
            f"{index}. {decision.decision}\n"
            f"   - 依據：{decision.evidence}"
        )

    task_lines = []
    for index, task in enumerate(record.tasks, start=1):
        task_lines.append(
            f"{index}. **{task.owner}**\n"
            f"   - 任務內容：{task.task}\n"
            f"   - 期限：{task.deadline if task.deadline else '未指定'}\n"
            f"   - 狀態：{task.status}\n"
            f"   - 依據：{task.evidence}"
        )

    markdown = f"""# {record.meeting_title}

## 一、會議日期
{record.meeting_date if record.meeting_date else "未指定"}

## 二、一句話摘要
{record.one_sentence_summary}

## 三、會議重點摘要
{summary_text if summary_text else "本次會議未整理出摘要。"}

## 四、討論主題
{topics_text if topics_text else "本次會議未整理出討論主題。"}

## 五、會議決議
{chr(10).join(decision_lines) if decision_lines else "本次會議未明確提到決議。"}

## 六、待辦事項
{chr(10).join(task_lines) if task_lines else "本次會議未明確提到待辦事項。"}

## 七、問題與風險
{problems_text if problems_text else "本次會議未明確提到問題或風險。"}

## 八、下一步行動
{next_steps_text if next_steps_text else "本次會議未明確提到下一步行動。"}

## 九、關鍵字
{", ".join(record.keywords)}
"""

    return markdown


# =========================
# 6. 主流程
# =========================

def main():
    # 如果執行時有指定檔名，就讀取指定檔案
    # 如果沒有指定，就預設讀取 transcript.txt
    if len(sys.argv) > 1:
        transcript_file = sys.argv[1]
    else:
        transcript_file = "transcript.txt"

    transcript_path = Path(transcript_file)
    transcript = read_transcript(transcript_file)

    print(f"已讀取逐字稿：{transcript_file}")
    print("正在呼叫 Gemini API...")

    record = analyze_with_gemini(transcript)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 依照逐字稿檔名產生不同輸出檔，避免覆蓋
    output_name = transcript_path.stem

    json_path = output_dir / f"{output_name}_record.json"
    md_path = output_dir / f"{output_name}_record.md"

    json_path.write_text(
        json.dumps(record.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    md_path.write_text(
        record_to_markdown(record),
        encoding="utf-8"
    )

    print("會議紀錄產生完成！")
    print(f"JSON 檔案：{json_path}")
    print(f"Markdown 檔案：{md_path}")


if __name__ == "__main__":
    main()