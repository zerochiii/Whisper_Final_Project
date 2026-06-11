import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import torch
import tempfile
import os
import traceback
import json
from datetime import datetime
from pathlib import Path

from analyze_meeting import analyze_with_gemini, record_to_markdown


# =========================
# RAG 模組載入
# 需要專案中有 memory_store.py、rag_query.py
# =========================
try:
    from memory_store import add_meeting
    from rag_query import query as rag_query
    RAG_MODULE_READY = True
    RAG_IMPORT_ERROR = ""
except Exception as e:
    RAG_MODULE_READY = False
    RAG_IMPORT_ERROR = str(e)


# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="AI 專題會議助手",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ AI 專題會議助手")
st.write(
    "可使用網頁錄音、上傳音檔或上傳逐字稿，"
    "透過 Faster-Whisper 產生逐字稿、Gemini 產生摘要與待辦事項，"
    "並將會議紀錄存入 RAG 知識庫進行歷史會議查詢。"
)


# =========================
# 工具函式：Faster-Whisper
# =========================
def clean_text(text):
    text = text.strip()
    text = text.replace(",", "，")
    text = text.replace("?", "？")
    text = text.replace("!", "！")
    text = text.replace("  ", " ")
    return text


def get_available_devices():
    if torch.cuda.is_available():
        return ["cuda", "cpu"]
    return ["cpu"]


def get_compute_type(device):
    if device == "cuda":
        return "float16"
    return "int8"


@st.cache_resource
def load_model(model_size, device, compute_type):
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )


def transcribe_audio(audio_path, model_size, device, compute_type, prompt):
    model = load_model(model_size, device, compute_type)

    segments, info = model.transcribe(
        audio_path,
        language="zh",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        initial_prompt=prompt
    )

    transcript = ""

    for segment in segments:
        text = clean_text(segment.text)
        transcript += f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}\n"

    return transcript


# =========================
# 工具函式：RAG JSON 讀取 / 儲存
# =========================
def list_meeting_json_files(folder="meetings"):
    if not os.path.exists(folder):
        return []

    return [
        file for file in os.listdir(folder)
        if file.endswith(".json")
    ]


def load_meeting_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tasks_to_text(tasks):
    if not tasks:
        return ""

    lines = []

    for task in tasks:
        if isinstance(task, dict):
            owner = task.get("owner", "未指定")
            task_content = task.get("task", "")
            deadline = task.get("deadline")
            status = task.get("status")

            extra = []
            if deadline:
                extra.append(f"期限：{deadline}")
            if status:
                extra.append(f"狀態：{status}")

            if extra:
                lines.append(f"{owner}：{task_content}（{'，'.join(extra)}）")
            else:
                lines.append(f"{owner}：{task_content}")
        else:
            lines.append(str(task))

    return "\n".join(lines)


def build_summary_for_rag(record_dict):
    summary_parts = []

    if record_dict.get("one_sentence_summary"):
        summary_parts.append(record_dict["one_sentence_summary"])

    if record_dict.get("summary_points"):
        summary_parts.append("會議重點：")
        summary_parts.extend([f"- {point}" for point in record_dict["summary_points"]])

    if record_dict.get("decisions"):
        summary_parts.append("會議決議：")
        for decision in record_dict["decisions"]:
            summary_parts.append(f"- {decision.get('decision', '')}")

    if record_dict.get("next_steps"):
        summary_parts.append("下一步行動：")
        summary_parts.extend([f"- {step}" for step in record_dict["next_steps"]])

    return "\n".join(summary_parts)


def save_gemini_record_for_rag(record_dict, transcript):
    """
    將 analyze_meeting.py 產生的 MeetingRecord 格式
    轉成 RAG 模組較容易讀取的 meeting JSON 格式：
    {
      "date": "...",
      "title": "...",
      "summary": "...",
      "tasks": [...],
      "transcript": "..."
    }
    """
    os.makedirs("meetings", exist_ok=True)

    meeting_date = record_dict.get("meeting_date") or datetime.now().strftime("%Y-%m-%d")
    title = record_dict.get("meeting_title") or "未命名會議"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join("meetings", f"meeting_{meeting_date}_{timestamp}.json")

    meeting = {
        "date": meeting_date,
        "title": title,
        "summary": build_summary_for_rag(record_dict),
        "tasks": record_dict.get("tasks", []),
        "decisions": record_dict.get("decisions", []),
        "risks_or_problems": record_dict.get("risks_or_problems", []),
        "next_steps": record_dict.get("next_steps", []),
        "keywords": record_dict.get("keywords", []),
        "transcript": transcript.strip()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meeting, f, ensure_ascii=False, indent=2)

    return json_path, meeting


# =========================
# Session State 初始化
# =========================
default_states = {
    "audio_path": None,
    "transcript": "",
    "summary": "",
    "tasks": "",
    "meeting_record": None,
    "meeting_json": "",
    "meeting_md": "",
    "saved_json_path": "",
    "loaded_json_path": "",
    "rag_answer": "",
    "rag_index_count": 0,
}

for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================
# 側邊欄設定
# =========================
st.sidebar.header("⚙️ 模型設定")

model_size = st.sidebar.selectbox(
    "選擇 Faster-Whisper 模型",
    ["base", "small", "medium"],
    index=2
)

available_devices = get_available_devices()

device = st.sidebar.selectbox(
    "執行裝置",
    available_devices,
    index=0
)

compute_type = get_compute_type(device)

st.sidebar.write("目前運算模式：", compute_type)

if torch.cuda.is_available():
    st.sidebar.success("CUDA 可用")
    st.sidebar.write("GPU：", torch.cuda.get_device_name(0))
else:
    st.sidebar.warning("CUDA 不可用，目前使用 CPU")

custom_prompt = st.sidebar.text_area(
    "可選 Prompt",
    value=(
        "以下是一段中文會議或討論錄音。"
        "內容可能包含中文、英文、數字、專有名詞與人名。"
        "請依照實際語音內容轉錄，不要自行摘要或改寫。"
    ),
    height=150
)

st.sidebar.divider()
st.sidebar.header("🧠 RAG 狀態")

if RAG_MODULE_READY:
    st.sidebar.success("RAG 模組已載入")
else:
    st.sidebar.error("RAG 模組載入失敗")
    st.sidebar.code(RAG_IMPORT_ERROR)


# =========================
# 上方分頁
# =========================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📂 會議紀錄",
    "🎙️ 音訊輸入",
    "📝 逐字稿",
    "📌 摘要與待辦",
    "🔍 RAG 查詢"
])


# =========================
# Tab 0：會議紀錄 / JSON 管理
# =========================
with tab0:
    st.header("📂 會議紀錄管理")

    st.write(
        "這裡可以讀取 meetings 資料夾中的 JSON 會議紀錄，"
        "並將它存入 RAG 知識庫。"
    )

    meeting_files = list_meeting_json_files("meetings")

    if not meeting_files:
        st.warning("meetings 資料夾中目前沒有 JSON 檔。你可以先到「摘要與待辦」產生一份會議紀錄。")
    else:
        selected_file = st.selectbox(
            "選擇會議 JSON 檔",
            meeting_files
        )

        json_path = os.path.join("meetings", selected_file)

        col_load, col_store = st.columns(2)

        with col_load:
            if st.button("讀取會議紀錄"):
                try:
                    meeting = load_meeting_json(json_path)

                    st.session_state["transcript"] = meeting.get("transcript", "")
                    st.session_state["summary"] = meeting.get("summary", "")
                    st.session_state["tasks"] = tasks_to_text(meeting.get("tasks", []))
                    st.session_state["saved_json_path"] = json_path
                    st.session_state["loaded_json_path"] = json_path

                    st.success(f"已成功讀取：{selected_file}")

                except Exception as e:
                    st.error("讀取 JSON 失敗")
                    st.code(str(e))
                    st.code(traceback.format_exc())

        with col_store:
            if st.button("將此 JSON 存入 RAG 知識庫"):
                if not RAG_MODULE_READY:
                    st.error("RAG 模組尚未正確載入，無法存入知識庫。")
                    st.code(RAG_IMPORT_ERROR)
                else:
                    try:
                        with st.spinner("正在寫入 RAG 知識庫..."):
                            add_meeting(json_path)
                        st.session_state["rag_index_count"] += 1
                        st.success("已存入 RAG 知識庫")
                    except Exception as e:
                        st.error("存入知識庫失敗")
                        st.code(str(e))
                        st.code(traceback.format_exc())

        if st.session_state.get("loaded_json_path"):
            st.info(f"目前讀取檔案：{st.session_state['loaded_json_path']}")

        if st.session_state.get("transcript"):
            st.subheader("逐字稿")
            st.text_area(
                "Transcript",
                st.session_state["transcript"],
                height=250
            )

        if st.session_state.get("summary"):
            st.subheader("摘要")
            st.text_area(
                "Summary",
                st.session_state["summary"],
                height=150
            )

        if st.session_state.get("tasks"):
            st.subheader("待辦事項")
            st.text_area(
                "Tasks",
                st.session_state["tasks"],
                height=150
            )


# =========================
# Tab 1：音訊輸入
# =========================
with tab1:
    st.header("🎙️ 音訊輸入")

    input_mode = st.radio(
        "選擇輸入方式",
        ["網頁錄音", "上傳音檔"],
        horizontal=True
    )

    if input_mode == "網頁錄音":
        st.subheader("網頁錄音")

        audio = mic_recorder(
            start_prompt="開始錄音",
            stop_prompt="停止錄音",
            just_once=False,
            use_container_width=True,
            key="recorder"
        )

        if audio:
            st.success("錄音完成")

            audio_bytes = audio["bytes"]
            st.audio(audio_bytes, format="audio/wav")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                st.session_state["audio_path"] = tmp_file.name

            st.write("目前錄音檔：", st.session_state["audio_path"])

    elif input_mode == "上傳音檔":
        st.subheader("上傳音檔")

        uploaded_file = st.file_uploader(
            "請上傳會議錄音檔",
            type=["wav", "mp3", "m4a"]
        )

        if uploaded_file is not None:
            st.success("音檔上傳成功")
            st.audio(uploaded_file)

            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                st.session_state["audio_path"] = tmp_file.name

            st.write("目前音檔：", st.session_state["audio_path"])

    if st.session_state["audio_path"]:
        st.info("音訊已準備完成，請切換到「逐字稿」分頁進行轉錄。")
    else:
        st.warning("尚未錄音或上傳音檔。")


# =========================
# Tab 2：逐字稿
# =========================
with tab2:
    st.header("📝 Faster-Whisper 逐字稿")

    if st.session_state["audio_path"] is None:
        st.warning("請先到「音訊輸入」分頁錄音或上傳音檔，或到「摘要與待辦」分頁上傳 .txt 逐字稿。")
    else:
        st.success("已偵測到音檔")
        st.write("音檔路徑：", st.session_state["audio_path"])

        if st.button("開始轉錄"):
            try:
                with st.spinner("Faster-Whisper 轉錄中，請稍候..."):
                    transcript = transcribe_audio(
                        audio_path=st.session_state["audio_path"],
                        model_size=model_size,
                        device=device,
                        compute_type=compute_type,
                        prompt=custom_prompt
                    )

                if transcript.strip() == "":
                    st.warning("沒有辨識到文字，可能是音檔太小聲、沒有語音或錄音品質不佳。")
                else:
                    st.session_state["transcript"] = transcript
                    st.success("轉錄完成")

            except Exception as e:
                st.error("轉錄失敗")
                st.code(str(e))
                st.code(traceback.format_exc())

    if st.session_state["transcript"]:
        st.subheader("逐字稿結果")

        st.text_area(
            "Transcript",
            st.session_state["transcript"],
            height=450
        )

        st.download_button(
            label="下載逐字稿 transcript.txt",
            data=st.session_state["transcript"],
            file_name="transcript.txt",
            mime="text/plain"
        )
    else:
        st.info("尚未產生或讀取逐字稿。")


# =========================
# Tab 3：摘要與待辦
# =========================
with tab3:
    st.header("📌 Gemini 摘要與待辦事項")

    st.info("可以先完成 Faster-Whisper 轉錄，也可以直接上傳 .txt 逐字稿使 Gemini 產生摘要與待辦事項。")

    with st.expander("📄 上傳逐字稿 .txt", expanded=True):
        uploaded_transcript = st.file_uploader(
            "請上傳逐字稿文字檔",
            type=["txt"],
            key="uploaded_transcript_txt"
        )

        if uploaded_transcript is not None:
            try:
                transcript_text = uploaded_transcript.read().decode("utf-8")
            except UnicodeDecodeError:
                uploaded_transcript.seek(0)
                transcript_text = uploaded_transcript.read().decode("big5")

            st.session_state["transcript"] = transcript_text

            # 換新逐字稿時，清空上一筆 Gemini 分析結果
            st.session_state["meeting_record"] = None
            st.session_state["meeting_json"] = ""
            st.session_state["meeting_md"] = ""
            st.session_state["saved_json_path"] = ""

            st.success("逐字稿上傳成功，已設定為目前逐字稿。")

    if not st.session_state["transcript"]:
        st.warning("請先完成逐字稿轉錄，或上傳 .txt 逐字稿。")
    else:
        st.subheader("目前逐字稿")

        st.text_area(
            "逐字稿內容",
            st.session_state["transcript"],
            height=250
        )

        if st.button("使用 Gemini 產生會議紀錄"):
            try:
                with st.spinner("Gemini 正在整理摘要與待辦事項，請稍候..."):
                    record = analyze_with_gemini(
                        st.session_state["transcript"]
                    )

                    record_dict = record.model_dump()

                    meeting_json = json.dumps(
                        record_dict,
                        ensure_ascii=False,
                        indent=2
                    )

                    meeting_md = record_to_markdown(record)

                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    json_path = output_dir / f"meeting_record_{timestamp}.json"
                    md_path = output_dir / f"meeting_record_{timestamp}.md"

                    json_path.write_text(meeting_json, encoding="utf-8")
                    md_path.write_text(meeting_md, encoding="utf-8")

                    # 另存一份給 RAG 使用的 JSON
                    rag_json_path, rag_meeting = save_gemini_record_for_rag(
                        record_dict=record_dict,
                        transcript=st.session_state["transcript"]
                    )

                    # 如果 RAG 模組可用，直接寫入知識庫
                    if RAG_MODULE_READY:
                        add_meeting(rag_json_path)
                        st.session_state["rag_index_count"] += 1

                    st.session_state["meeting_record"] = record
                    st.session_state["meeting_json"] = meeting_json
                    st.session_state["meeting_md"] = meeting_md
                    st.session_state["summary"] = rag_meeting["summary"]
                    st.session_state["tasks"] = tasks_to_text(rag_meeting["tasks"])
                    st.session_state["saved_json_path"] = rag_json_path

                st.success("Gemini 會議紀錄產生完成！")
                st.info(f"已儲存 JSON：{json_path}")
                st.info(f"已儲存 Markdown：{md_path}")
                st.info(f"已儲存 RAG JSON：{st.session_state['saved_json_path']}")

                if RAG_MODULE_READY:
                    st.success("已自動存入 RAG 知識庫")
                else:
                    st.warning("RAG 模組尚未載入，因此尚未存入知識庫。")

            except Exception as e:
                st.error("Gemini 會議紀錄產生失敗")
                st.code(str(e))
                st.code(traceback.format_exc())

        if st.session_state["meeting_record"] is not None:
            record = st.session_state["meeting_record"]

            st.divider()

            st.subheader("一句話摘要")
            st.write(record.one_sentence_summary)

            st.subheader("會議重點摘要")
            for point in record.summary_points:
                st.write(f"- {point}")

            st.subheader("討論主題")
            for topic in record.discussed_topics:
                st.write(f"- {topic}")

            st.subheader("會議決議")
            if record.decisions:
                for decision in record.decisions:
                    st.write(f"- {decision.decision}")
                    st.caption(f"依據：{decision.evidence}")
            else:
                st.info("本次會議未明確整理出決議。")

            st.subheader("待辦事項")

            if record.tasks:
                task_rows = []

                for task in record.tasks:
                    task_rows.append({
                        "負責人": task.owner,
                        "任務內容": task.task,
                        "期限": task.deadline if task.deadline else "未指定",
                        "狀態": task.status,
                        "依據": task.evidence
                    })

                st.dataframe(task_rows, use_container_width=True)
            else:
                st.info("本次會議未明確整理出待辦事項。")

            st.subheader("問題與風險")
            if record.risks_or_problems:
                for problem in record.risks_or_problems:
                    st.write(f"- {problem}")
            else:
                st.info("本次會議未明確提到問題或風險。")

            st.subheader("下一步行動")
            for step in record.next_steps:
                st.write(f"- {step}")

            st.subheader("關鍵字")
            st.write("、".join(record.keywords))

            if st.session_state["saved_json_path"]:
                st.info(f"RAG JSON 檔案：{st.session_state['saved_json_path']}")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="下載會議紀錄 JSON",
                    data=st.session_state["meeting_json"],
                    file_name="meeting_record.json",
                    mime="application/json"
                )

            with col2:
                st.download_button(
                    label="下載會議紀錄 Markdown",
                    data=st.session_state["meeting_md"],
                    file_name="meeting_record.md",
                    mime="text/markdown"
                )


# =========================
# Tab 4：RAG 查詢
# =========================
with tab4:
    st.header("🔍 RAG 歷史會議查詢")

    if not RAG_MODULE_READY:
        st.error("RAG 模組尚未正確載入，請確認 memory_store.py、rag_query.py 與相關套件。")
        st.code(RAG_IMPORT_ERROR)
    else:
        st.info("請先產生會議紀錄，或在「會議紀錄」分頁將 JSON 存入 RAG 知識庫，再進行查詢。")
        st.info(f"目前本次執行期間已加入知識庫的會議數：{st.session_state['rag_index_count']}")

        default_questions = [
            "誰負責前端開發？",
            "上次會議決定了什麼？",
            "模型測試的進度如何？",
            "誰負責 Gemini API？",
            "誰負責 Faster-Whisper？"
        ]

        selected_question = st.selectbox(
            "選擇常用問題",
            ["自訂問題"] + default_questions
        )

        if selected_question == "自訂問題":
            question = st.text_input("請輸入想查詢的會議問題")
        else:
            question = st.text_input(
                "請輸入想查詢的會議問題",
                value=selected_question
            )

        if st.button("開始 RAG 查詢"):
            if question.strip() == "":
                st.warning("請先輸入問題。")
            else:
                try:
                    with st.spinner("RAG 查詢中..."):
                        answer = rag_query(question)

                    st.session_state["rag_answer"] = answer
                    st.success("查詢完成")

                except Exception as e:
                    st.error("RAG 查詢失敗")
                    st.code(str(e))
                    st.code(traceback.format_exc())

        if st.session_state["rag_answer"]:
            st.subheader("系統回答")
            st.write(st.session_state["rag_answer"])
