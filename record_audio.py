import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os


def record_audio(filename="meeting.wav", fs=16000):
    input("按 Enter 開始錄音...")

    print("錄音中，按 Enter 停止錄音。")

    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print("錄音狀態警告：", status)
        audio_data.append(indata.copy())

    stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype="int16",
        callback=callback
    )

    stream.start()
    input()
    stream.stop()
    stream.close()

    if len(audio_data) == 0:
        print("沒有錄到音訊。")
        return

    audio = np.concatenate(audio_data, axis=0)

    write(filename, fs, audio)

    print(f"錄音完成，已儲存為：{os.path.abspath(filename)}")


if __name__ == "__main__":
    record_audio()