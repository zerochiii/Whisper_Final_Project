from faster_whisper import WhisperModel
import torch
import os
import traceback


def clean_text(text):
    text = text.strip()
    text = text.replace(",", "，")
    text = text.replace("?", "？")
    text = text.replace("!", "！")
    text = text.replace("  ", " ")
    return text


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_compute_type(device):
    if device == "cuda":
        return "float16"
    return "int8"


def transcribe_audio(
    audio_path="meeting.wav",
    output_path="transcript.txt",
    model_size="medium",
    custom_prompt=None,
    device=None
):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到音檔：{audio_path}")

    if device is None:
        device = get_default_device()

    compute_type = get_compute_type(device)

    print("使用模型：", model_size)
    print("使用裝置：", device)
    print("運算模式：", compute_type)

    default_prompt = (
        "以下是一段中文會議或討論錄音。"
        "內容可能包含中文、英文、數字、專有名詞與人名。"
        "請依照實際語音內容轉錄，不要自行摘要或改寫。"
    )

    prompt = custom_prompt if custom_prompt else default_prompt

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

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
        line = f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}"
        print(line)
        transcript += line + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"逐字稿已輸出：{output_path}")

    return transcript


if __name__ == "__main__":
    try:
        transcribe_audio()
    except Exception as e:
        print("轉錄失敗：", e)
        print(traceback.format_exc())