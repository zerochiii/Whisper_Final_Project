# 模型辨識準確率不如預期

問題：初期使用 Faster-Whisper 的 base 模型進行中文會議轉錄測試，雖然可以成功輸出逐字稿，但辨識結果出現大量錯字。例如「Gemini API」被辨識成「專門來 API」、「Streamlit」被辨識成「StreamLate」、「逐字稿」被辨識成「足字高」。

排查過程：

1. 測試不同錄音內容。
2. 比較 tiny、base、small 模型辨識結果。
3. 嘗試調整 beam_size 與 language 參數。

解決方法：改用較大的 medium 模型，並加入 initial_prompt 提供語境資訊，同時加入文字後處理機制改善辨識品質。

結果：專有名詞辨識率提升，逐字稿品質較穩定，但所需時間較長。

# 背景聲音影響語音辨識

問題：背景噪音會造成辨識錯字增加。

解法：加入 VAD 過濾無聲片段，並建議使用外接麥克風與安靜環境。

# 環境部屬問題
我們嘗試將 Faster-Whisper 從 CPU 模式改成 CUDA 模式，希望透過 GPU 加速轉錄。但執行時出現 cublas64_12.dll is not found 的錯誤。排查後發現，CUDA 模式不只需要 NVIDIA 顯示卡，也需要完整 CUDA Runtime 與 cuBLAS 函式庫。為了確保展示穩定，最後展示版改回 CPU + int8 模式。

# 已完成功能
1. 語音轉文字(直接錄音或是上傳音檔)
2. 可選faster whisper模型
3. 可選CPU或GPU
4. 可下Prompt
5. 網頁初步設計