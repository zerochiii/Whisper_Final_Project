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
# 頁面設定
# =========================
st.set_page_config(
    page_title="AI 專題會議助手",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ AI 專題會議助手")
st.write("可使用網頁錄音或上傳音檔，並透過 Faster-Whisper 產生逐字稿。")


# =========================
# 工具函式
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
# Session State 初始化
# =========================
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""

if "summary" not in st.session_state:
    st.session_state["summary"] = ""

if "tasks" not in st.session_state:
    st.session_state["tasks"] = ""

if "meeting_record" not in st.session_state:
    st.session_state["meeting_record"] = None

if "meeting_json" not in st.session_state:
    st.session_state["meeting_json"] = ""

if "meeting_md" not in st.session_state:
    st.session_state["meeting_md"] = ""


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


# =========================
# 上方分頁
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️ 音訊輸入",
    "📝 逐字稿",
    "📌 摘要與待辦",
    "🔍 RAG 查詢"
])


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
        st.warning("請先到「音訊輸入」分頁錄音或上傳音檔。")
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
            st.info("尚未產生逐字稿。")


# =========================
# Tab 3：摘要與待辦
# =========================
with tab3:
    st.header("📌 Gemini 摘要與待辦事項")

    st.info("你可以先完成 Faster-Whisper 轉錄，也可以直接上傳 .txt 逐字稿使 Gemini 產生摘要與待辦事項。")

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

            st.success("逐字稿上傳成功，已設定為目前逐字稿。")

    if not st.session_state["transcript"]:
        st.warning("請先完成逐字稿轉錄。")
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

                    st.session_state["meeting_record"] = record
                    st.session_state["meeting_json"] = meeting_json
                    st.session_state["meeting_md"] = meeting_md

                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    json_path = output_dir / f"meeting_record_{timestamp}.json"
                    md_path = output_dir / f"meeting_record_{timestamp}.md"

                    json_path.write_text(meeting_json, encoding="utf-8")
                    md_path.write_text(meeting_md, encoding="utf-8")

                st.success("Gemini 會議紀錄產生完成！")
                st.info(f"已儲存 JSON：{json_path}")
                st.info(f"已儲存 Markdown：{md_path}")

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

    if not st.session_state["transcript"]:
        st.warning("請先產生逐字稿，之後才能建立知識庫。")
    else:
        st.info("此區塊預留給 LangChain + Chroma / FAISS。")

        question = st.text_input("請輸入想查詢的會議問題")

        if st.button("查詢（預留 RAG）"):
            if question.strip() == "":
                st.warning("請先輸入問題。")
            else:
                st.write("你的問題：", question)
                st.success("這裡之後會從歷史會議知識庫中檢索答案。")