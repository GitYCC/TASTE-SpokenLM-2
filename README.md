# TASTE-SpokenLM-2

**TASTE2: Text-Aligned Speech Modeling and Deployment toward Full-Duplex Voice Interaction**

本專案為 TASTE2 語言模型的訓練與推論程式碼庫，將 TASTE 方法擴展為可用於全雙工語音互動的「漸進式對話堆疊」（incremental dialogue stack）。系統能處理即時語音輸入、判斷交流／讓步時機、在使用者打斷時即時停止，同時保留預訓練語言模型的語言能力與音檔的聲學特性。

- 專案首頁：https://gitycc.github.io/TASTE2-Homepage/
- VoiceBot 系統（部署端）：https://github.com/GitYCC/TASTE-Voice-Bot
- 模型檢查點：https://huggingface.co/collections/YC-Chen/taste2
- 資料集：https://huggingface.co/datasets/wilzzzz/paralinguistic_dialogues

## 核心概念

每個文本 token 對應一個連續的音訊潛在表示（audio latent），在保持文本序列長度不變的前提下，讓聲學資訊得以在整個語音對話堆疊中傳遞，避免了異質 token 流交錯的問題。架構主要由三部分組成：

1. **共享詞彙表（Shared Vocabulary）**：統一文本 token 詞彙表，消除詞級平均與語言相關的切分問題
2. **對齊預測（Aligned Prediction）**：語言模型為每個文本 token 預測一個音訊潛在表示，不延長序列長度
3. **串流合成（Streaming Synthesis）**：漸進式的語音解 token 化器產生 S3 unit，再透過 CosyVoice2 合成音訊

詳細方法與實驗結果請參考[論文](https://gitycc.github.io/TASTE2-Homepage/assets/taste2-paper.pdf)。

## 目錄結構

```
taste_speech/       # 核心模型套件（audio encoder/quantizer/segmenter、CosyVoice 模組、TASTE2 SLM 等）
training/           # Stage 1（audio tokenizer/detokenizer）與 Stage 2（Spoken LM）訓練程式碼與設定檔
scripts/            # 推論、資料處理、模型驗證與匯出（ONNX / TensorRT-LLM）腳本
docs/               # 訓練手冊與資料格式說明
docker/             # 訓練/推論用容器環境（cuda124、cuda13）
audio_samples/      # 推論測試用音檔
results/            # 推論輸出結果
```

## 環境安裝

建議使用容器環境，內含所有訓練/推論所需依賴：

```bash
bash docker/cuda13/build.sh   # 建置映像檔（CUDA 13 + TensorRT-LLM）
bash docker/cuda13/run.sh     # 啟動容器
```

亦可參考 `docker/cuda124/` 使用 CUDA 12.4 環境。若在本機開發，依循全域慣例以 `uv venv .venv` 建立虛擬環境並安裝 `docker/*/requirements.txt`。

## 使用方式

### 訓練

分為兩個階段，詳見 [training/docs/TRAINING.md](training/docs/TRAINING.md)：

- **Stage 1**：訓練語音 tokenizer / detokenizer（`training/jobs/run_stage1_*.sh`）
- **Stage 2**：訓練核心 Spoken Language Model（`training/jobs/run_stage2_*.sh`），支援 1B/2B/8B 規模與 LoRA 變體

### 推論（語音生成）

```bash
bash scripts/run_generate_stage2.sh
```

底層呼叫 `scripts/generate_audio.py`，支援 `--stage {1,2,sft}`，可批次處理多個輸入音檔並輸出合成語音。

### 模型匯出

- `scripts/export_flow_estimator_onnx.py` + `scripts/build_flow_estimator_onnx.sh`：匯出 flow estimator 為 ONNX
- `scripts/export_cosyvoice_model.py` + `scripts/build_cosyvoice_trtllm.sh`：匯出並建置 TensorRT-LLM 引擎

## 相關研究成果

| 指標 | 成績 |
|---|---|
| LLaMA-Questions 準確度 | 56.3%（保留文字參考答案 87.6% 的準確度） |
| 使用者打斷後的響應延遲 | 0.060 秒 |
| Full-Duplex-Bench v1.0 | 五項任務中 4 項最佳、1 項並列最佳（727 筆樣本） |
| 首音延遲（經 TensorRT 優化） | 2.701 秒（較未優化快 22%） |

## 作者

Yi-Chang Chen, Chun Wei Chen, Dien-Ruei Wu, Jie Lin, Hung-yi Lee, Da-Shan Shiu
（MediaTek Research / National Taiwan University）
