# CosyVoice 技術文檔

CosyVoice 是一個基於大語言模型的可擴展多語言零樣本文字轉語音合成系統。

## 文檔結構

- **[架構概述](ARCHITECTURE.md)** - 系統整體架構和設計理念
- **[API 參考](API_REFERENCE.md)** - 完整的 API 介面文檔
- **[模組說明](MODULES.md)** - 各核心模組詳細介紹
- **[使用範例](EXAMPLES.md)** - 實用的使用案例和代碼範例
- **[配置說明](CONFIGURATION.md)** - 配置文件和參數詳解
- **[部署指南](DEPLOYMENT.md)** - 生產環境部署方案
- **[故障排除](TROUBLESHOOTING.md)** - 常見問題解決方案

## 快速開始

### 環境準備

1. **安裝 Conda 環境**:
   ```bash
   conda create -n cosyvoice -y python=3.10
   conda activate cosyvoice
   ```

2. **克隆專案並安裝依賴**:
   ```bash
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   cd CosyVoice
   pip install -r requirements.txt
   ```

3. **下載預訓練模型**:
   ```python
   from modelscope import snapshot_download
   snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
   ```

### 基本使用

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 初始化模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# 零樣本語音克隆
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('你好，歡迎使用 CosyVoice', '希望你以後能夠做得更好', prompt_speech_16k)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## 版本資訊

- **CosyVoice 3.0**: 最新版本，包含野外語音生成能力
- **CosyVoice 2.0**: 更精確、穩定、快速的語音生成
- **CosyVoice 1.0**: 基礎版本

## 主要特性

### 🌍 多語言支援
- **支援語言**: 中文、英文、日文、韓文、中文方言（粵語、四川話、上海話、天津話、武漢話等）
- **跨語言合成**: 支援零樣本跨語言語音克隆和程式碼轉換場景
- **混合語言**: 支援多語言混合內容的語音合成

### ⚡ 超低延遲
- **雙向串流支援**: 整合離線和串流建模技術
- **極速響應**: 首包合成延遲低至 150ms，同時保持高品質音訊輸出
- **即時處理**: 支援即時語音合成應用

### 🎯 高精度
- **發音改善**: 相比 CosyVoice 1.0，發音錯誤率降低 30% 到 50%
- **基準表現**: 在 Seed-TTS 評估集困難測試集上達到最低字符錯誤率
- **品質提升**: MOS 評分從 5.4 提升至 5.53

### 🔄 強穩定性
- **音色一致性**: 確保零樣本和跨語言語音合成的音色穩定
- **跨語言改善**: 相比 1.0 版本在跨語言合成方面有顯著改善
- **魯棒性**: 支援各種輸入文本的穩定處理

### 🎭 自然體驗
- **韻律增強**: 改善合成音訊的韻律和音質
- **情感控制**: 支援更細緻的情感控制和口音調整
- **指令式合成**: 支援自然語言指令控制語音特性

### 🔧 多種推理模式
- **SFT 模式**: 使用預訓練音色進行語音合成
- **零樣本克隆**: 僅需 3 秒音訊即可克隆任意音色
- **跨語言合成**: 支援不同語言間的音色遷移
- **語音轉換**: 將一種音色轉換為另一種音色
- **指令式控制**: 使用自然語言描述控制語音特性