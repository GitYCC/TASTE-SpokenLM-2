# CosyVoice 模組詳解

## 核心模組結構

```
cosyvoice/
├── cli/              # 命令行介面
├── dataset/          # 數據集處理
├── flow/             # 流匹配模型
├── hifigan/          # HiFi-GAN 生成器
├── llm/              # 大語言模型
├── tokenizer/        # 分詞器
├── transformer/      # Transformer 架構
├── utils/            # 工具函數
└── vllm/             # vLLM 整合
```

## CLI 模組 (cosyvoice.cli)

### cosyvoice.py
主要的 API 介面，提供 CosyVoice 和 CosyVoice2 類別。

**核心類別**:
- `CosyVoice`: 1.0 版本的主要介面
- `CosyVoice2`: 2.0 版本的增強介面，支援更多功能

**主要方法**:
```python
class CosyVoice:
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1)
    def list_available_spks(self)  # 列出可用音色
    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id)  # 添加零樣本音色
    def save_spkinfo(self)  # 保存音色資訊
    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)
    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)
    def inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)
    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True)
    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0)
```

**關鍵特性**:
- 自動模型下載：支援 ModelScope ID 和本地路徑
- 多種推理模式：SFT、零樣本、跨語言、指令式、語音轉換
- 串流支援：支援即時音頻生成
- 效能優化：JIT、TensorRT、FP16 加速選項

### frontend.py
前端處理模組，負責文字預處理、音頻特徵提取和說話人編碼。

**CosyVoiceFrontEnd 類別**:
```python
class CosyVoiceFrontEnd:
    def __init__(self, get_tokenizer, feat_extractor, campplus_model, speech_tokenizer_model, spk2info='', allowed_special='all')
```

**核心功能**:
1. **文字處理**:
   - `text_normalize()`: 文字標準化（數字轉文字、標點處理等）
   - `tokenize()`: 多語言分詞，基於 Whisper tokenizer
   - 支援特殊控制符號：`[laughter]`, `[breath]`, `<strong></strong>`

2. **音頻處理**:
   - `compute_fbank()`: 計算 Mel 頻譜特徵
   - `compute_spk_embed()`: 提取說話人嵌入（使用 CAM++ 模型）
   - `speech_tokenize()`: 語音 token 化

3. **前端處理方法**:
   - `frontend_sft()`: 為 SFT 推理準備輸入
   - `frontend_zero_shot()`: 為零樣本推理準備輸入
   - `frontend_cross_lingual()`: 為跨語言推理準備輸入
   - `frontend_instruct()`: 為指令式推理準備輸入
   - `frontend_vc()`: 為語音轉換準備輸入

**文字標準化支援**:
- **ttsfrd**: 高性能文字標準化（可選）
- **wetext**: 備用文字標準化工具
- 支援中英日韓等多語言
- 數字、縮寫、標點符號處理

**音頻特徵提取**:
- 16kHz 重採樣
- 80 維 Mel 頻譜
- 256 點 hop size
- 說話人嵌入標準化

### model.py
模型載入和推理管理模組，負責統一管理 LLM、Flow 和 HiFi-GAN 三個組件。

**CosyVoiceModel 類別**:
```python
class CosyVoiceModel:
    def __init__(self, llm, flow, hift, fp16=False)
    def load(self, llm_model, flow_model, hift_model)  # 載入 PyTorch 模型
    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model)  # 載入 JIT 模型
    def load_trt(self, flow_decoder_model, flow_decoder_onnx, trt_concurrent, fp16)  # 載入 TensorRT 模型
    def tts(self, **model_input)  # 統一推理介面
```

**CosyVoice2Model 類別**:
- 繼承自 CosyVoiceModel
- 新增 vLLM 支援
- 支援更高效的批次推理
- 優化的記憶體管理

**模型組件管理**:
1. **LLM 組件**:
   - 文字編碼器：將文字轉換為語義表示
   - LLM 主體：生成語音語義 token
   - 採樣策略：RAS（Repetition Aware Sampling）

2. **Flow 組件**:
   - 編碼器：處理語義 token
   - 解碼器：條件流匹配生成 Mel 頻譜
   - 長度調節器：控制語音時長

3. **HiFi-GAN 組件**:
   - HiFT 生成器：Mel 頻譜轉語音波形
   - F0 預測器：基音頻率預測
   - NSF 聲碼器：神經源濾波器

**推理流程管理**:
- 串流推理：支援分塊處理和即時輸出
- 記憶體優化：自動清理中間結果
- 異常處理：模型載入失敗恢復
- 多 GPU 支援：並行推理能力

## 數據集模組 (cosyvoice.dataset)

### dataset.py
數據集載入和管理模組，支援多種數據格式和處理模式。

**核心功能**:
```python
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, processor_pipeline, mode='train')
    def __len__(self)
    def __getitem__(self, idx)
```

**支援的數據格式**:
- **Parquet 格式**: 高效的列式存儲格式
- **JSON/JSONL**: 文字和音頻路徑對應關係
- **CSV**: 傳統表格格式
- **直接音頻檔案**: WAV, FLAC 等格式

**數據載入特性**:
- 懶載入：節省記憶體使用
- 並行處理：多進程音頻載入
- 動態批次：根據序列長度動態組批
- 錯誤恢復：自動跳過損壞的音頻檔案

**數據集類型**:
1. **預訓練數據集**: 大規模無配對文字-音頻數據
2. **SFT 數據集**: 配對的文字-音頻數據
3. **指令數據集**: 包含指令標籤的數據
4. **評估數據集**: 用於模型評估的測試數據

### processor.py
數據處理器模組，提供豐富的音頻和文字處理功能。

**主要處理器**:

1. **parquet_opener**:
   - 功能：讀取 Parquet 格式數據
   - 支援：並行讀取、內存映射
   - 優化：自動數據類型推斷

2. **tokenize**:
   - 功能：文字分詞和 token 化
   - 支援：多語言、特殊符號
   - 配置：允許的特殊字符、最大長度

3. **filter**:
   ```python
   filter:
     max_length: 40960      # 最大音頻長度（樣本數）
     min_length: 0          # 最小音頻長度
     token_max_length: 200  # 最大文字長度（token 數）
     token_min_length: 1    # 最小文字長度
   ```

4. **resample**:
   - 功能：音頻重採樣到目標採樣率
   - 支援：多種採樣算法
   - 優化：GPU 加速重採樣

5. **truncate**:
   - 功能：截斷音頻到固定長度
   - 要求：長度必須是 hop_size 的倍數
   - 策略：隨機截取、固定截取

6. **compute_fbank**:
   - 功能：計算 Mel 頻譜特徵
   - 參數：n_fft=1024, num_mels=80, hop_size=256
   - 輸出：80 維 Mel 頻譜

7. **compute_f0**:
   - 功能：提取基音頻率
   - 算法：WORLD, DIO, Harvest
   - 後處理：平滑、插值

8. **parse_embedding**:
   - 功能：解析和處理說話人嵌入
   - 標準化：L2 正規化
   - 維度：通常為 192 維

9. **shuffle**:
   - 功能：數據混洗
   - 緩衝區：可配置大小
   - 種子：可重現的隨機性

10. **sort**:
    - 功能：按長度排序（減少填充）
    - 策略：局部排序
    - 效果：提高訓練效率

11. **batch**:
    - 功能：動態批次組建
    - 類型：dynamic（根據總幀數）或 static（固定大小）
    - 參數：max_frames_in_batch

12. **padding**:
    - 功能：序列填充對齊
    - 策略：零填充、重複填充
    - 掩碼：自動生成注意力掩碼

**處理管道配置**:
```yaml
# 基本處理管道
data_pipeline: [
  !ref <parquet_opener>,
  !ref <tokenize>,
  !ref <filter>,
  !ref <resample>,
  !ref <compute_fbank>,
  !ref <parse_embedding>,
  !ref <shuffle>,
  !ref <sort>,
  !ref <batch>,
  !ref <padding>,
]

# GAN 訓練專用管道
data_pipeline_gan: [
  !ref <parquet_opener>,
  !ref <tokenize>,
  !ref <filter>,
  !ref <resample>,
  !ref <truncate>,        # 額外的截斷步驟
  !ref <compute_fbank>,
  !ref <compute_f0>,      # 額外的 F0 提取
  !ref <parse_embedding>,
  !ref <shuffle>,
  !ref <sort>,
  !ref <batch>,
  !ref <padding>,
]
```

**性能優化**:
- **並行處理**: 多進程音頻處理
- **記憶體管理**: 流式處理大數據集
- **快取機制**: 預計算特徵快取
- **GPU 加速**: 支援 GPU 加速的處理步驟

## 流匹配模組 (cosyvoice.flow)

流匹配模組實現了條件流匹配（Conditional Flow Matching）技術，用於高品質的聲學特徵生成。

### flow.py
主要的流匹配模型實現。

**MaskedDiffWithXvec 類別**:
```python
class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self, input_size=512, output_size=80, spk_embed_dim=192, 
                 output_type='mel', vocab_size=4096, input_frame_rate=50)
```

**核心組件**:
1. **編碼器** (ConformerEncoder):
   - 處理語義 token 序列
   - 8 頭注意力，2048 線性單元
   - 6 層 Conformer 結構
   - 支援相對位置編碼

2. **長度調節器** (InterpolateRegulator):
   - 控制語音時長對應關係
   - 支援上採樣比例調整
   - 基於插值的長度調整

3. **條件流匹配解碼器** (ConditionalCFM):
   - 實現條件流匹配算法
   - 支援說話人條件生成
   - 可配置的 CFG（Classifier-Free Guidance）

**流匹配特性**:
- **穩定訓練**: 相比傳統擴散模型更穩定
- **高品質生成**: 更自然的聲學特徵
- **條件控制**: 支援說話人、風格控制
- **推理速度**: 較少的推理步驟

### flow_matching.py
條件流匹配的核心算法實現。

**ConditionalCFM 類別**:
```python
class ConditionalCFM(torch.nn.Module):
    def __init__(self, in_channels, n_spks=1, spk_emb_dim=80, cfm_params=None, estimator=None)
    def forward(self, x, mask, spk_emb=None)
    def inference(self, x, mask, spk_emb=None, solver='euler', cfg_scale=1.0)
```

**算法參數**:
- `sigma_min`: 最小噪聲標準差 (1e-6)
- `solver`: ODE 求解器類型 ('euler', 'heun', 'rk4')
- `t_scheduler`: 時間調度策略 ('linear', 'cosine')
- `training_cfg_rate`: 訓練時無條件比例 (0.2)
- `inference_cfg_rate`: 推理時 CFG 強度 (0.7)

### decoder.py
流匹配解碼器的具體實現。

**ConditionalDecoder 類別**:
```python
class ConditionalDecoder(torch.nn.Module):
    def __init__(self, in_channels=320, out_channels=80, channels=[256, 256], 
                 dropout=0.0, attention_head_dim=64, n_blocks=4, num_mid_blocks=12)
```

**架構特點**:
- **多尺度處理**: 不同解析度的特徵融合
- **注意力機制**: 自注意力和交叉注意力
- **殘差連接**: 梯度流動優化
- **分組歸一化**: 訓練穩定性提升

### length_regulator.py
長度調節器實現，控制語音時序對應關係。

**InterpolateRegulator 類別**:
```python
class InterpolateRegulator(torch.nn.Module):
    def __init__(self, channels=80, sampling_ratios=[1, 1, 1, 1])
    def forward(self, x, x_lengths=None, g=None)
```

**功能特性**:
- **插值上採樣**: 基於插值的特徵上採樣
- **比例控制**: 可配置的上採樣比例
- **長度保持**: 保持序列長度關係
- **可微分**: 支援端到端訓練

## HiFi-GAN 模組 (cosyvoice.hifigan)

HiFi-GAN 模組負責將 Mel 頻譜轉換為高品質的語音波形，是整個系統的最後一環。

### hifigan.py
主要的 HiFi-GAN 模型封裝。

**HiFiGan 類別**:
```python
class HiFiGan(nn.Module):
    def __init__(self, generator, discriminator, mel_spec_transform,
                 multi_mel_spectral_recon_loss_weight=45, 
                 feat_match_loss_weight=2.0, 
                 tpr_loss_weight=1.0, 
                 tpr_loss_tau=0.04)
```

**訓練損失**:
1. **生成器損失**: 欺騙判別器
2. **特徵匹配損失**: 匹配判別器中間特徵
3. **頻譜重建損失**: Mel 頻譜一致性
4. **TPR 損失**: 音高感知損失

### generator.py
HiFT (HiFi-GAN + F0) 生成器實現。

**HiFTGenerator 類別**:
```python
class HiFTGenerator(torch.nn.Module):
    def __init__(self, in_channels=80, base_channels=512, nb_harmonics=8, 
                 sampling_rate=22050, nsf_alpha=0.1, nsf_sigma=0.003)
```

**架構特點**:
1. **神經源濾波器 (NSF)**:
   - 基於可學習的激勵信號生成
   - 諧波建模：nb_harmonics=8
   - 有聲/無聲判別

2. **上採樣網路**:
   - 轉置卷積上採樣：rates=[8, 8]
   - 總上採樣倍數：64 (256 hop → 22050 Hz)
   - 卷積核大小：[16, 16]

3. **殘差塊**:
   - 多尺度卷積核：[3, 7, 11]
   - 擴張卷積：[[1,3,5], [1,3,5], [1,3,5]]
   - LeakyReLU 激活：slope=0.1

4. **ISTFT 後處理**:
   - 短時傅立葉逆變換
   - n_fft=16, hop_len=4
   - 高頻細節恢復

### discriminator.py
多尺度判別器實現。

**MultipleDiscriminator 類別**:
```python
class MultipleDiscriminator(torch.nn.Module):
    def __init__(self):
        self.mpd = MultiPeriodDiscriminator()  # 週期判別器
        self.mrd = MultiResSpecDiscriminator()  # 多解析度頻譜判別器
```

**判別器類型**:
1. **多週期判別器 (MPD)**:
   - 不同週期的子判別器
   - 捕捉語音的週期性特徵
   - 週期：[2, 3, 5, 7, 11]

2. **多解析度頻譜判別器 (MRD)**:
   - 不同 FFT 尺寸的頻譜判別
   - FFT 尺寸：[1024, 2048, 512]
   - 頻域特徵判別

### f0_predictor.py
F0（基音頻率）預測器實現。

**ConvRNNF0Predictor 類別**:
```python
class ConvRNNF0Predictor(torch.nn.Module):
    def __init__(self, num_class=1, in_channels=80, cond_channels=512)
    def forward(self, x, cond=None)
```

**功能特點**:
- **卷積特徵提取**: 處理 Mel 頻譜輸入
- **RNN 時序建模**: LSTM/GRU 捕捉時序依賴
- **條件預測**: 結合語義條件信息
- **連續預測**: 輸出連續 F0 值

**F0 處理流程**:
1. 特徵提取：Mel → 卷積特徵
2. 時序建模：RNN 處理時序信息
3. 條件融合：結合語義特徵
4. F0 預測：回歸連續 F0 值
5. 後處理：平滑、去噪

**NSF 聲碼器集成**:
- F0 驅動的諧波生成
- 可學習的激勵信號
- 有聲/無聲切換
- 高保真度波形合成

**優化特性**:
- **多 GPU 訓練**: 支援分散式訓練
- **混合精度**: FP16 加速訓練
- **梯度檢查點**: 節省顯存使用
- **動態損失縮放**: 避免梯度下溢

## LLM 模組 (cosyvoice.llm)

大語言模型模組是 CosyVoice 的核心，負責將文字轉換為語音語義 token 序列。

### llm.py
主要的大語言模型實現。

**TransformerLM 類別**:
```python
class TransformerLM(torch.nn.Module):
    def __init__(self, text_encoder_input_size=512, llm_input_size=1024, llm_output_size=1024,
                 text_token_size=51866, speech_token_size=4096, 
                 text_encoder, llm, sampling, 
                 length_normalized_loss=True, lsm_weight=0.0, spk_embed_dim=192)
```

**核心組件**:

1. **文字編碼器** (Text Encoder):
   ```python
   # 基於 ConformerEncoder
   text_encoder: ConformerEncoder
     input_size: 512
     output_size: 1024
     attention_heads: 16
     linear_units: 4096
     num_blocks: 6
   ```
   - 功能：將 token 序列編碼為語義表示
   - 架構：Conformer（Convolution + Transformer）
   - 特點：結合卷積和自注意力，適合語音相關任務

2. **LLM 主體** (Language Model):
   ```python
   # 基於 TransformerEncoder
   llm: TransformerEncoder
     input_size: 1024
     output_size: 1024
     attention_heads: 16
     linear_units: 4096
     num_blocks: 14
   ```
   - 功能：生成語音語義 token 序列
   - 架構：標準 Transformer Encoder
   - 特點：深度網路（14 層），強大的語言建模能力

3. **採樣策略** (Sampling):
   ```python
   # RAS (Repetition Aware Sampling)
   sampling: ras_sampling
     top_p: 0.8      # 核採樣概率
     top_k: 25       # 候選 token 數量
     win_size: 10    # RAS 窗口大小
     tau_r: 0.1      # 重複抑制強度
   ```

**推理流程**:
1. **文字預處理**: 分詞、添加特殊標記
2. **文字編碼**: 轉換為語義嵌入
3. **條件融合**: 結合說話人、指令等條件
4. **自回歸生成**: 逐步生成語音 token
5. **採樣控制**: RAS 避免重複，提升自然度

**條件控制機制**:

1. **說話人控制**:
   ```python
   # 說話人嵌入融合
   spk_embed = self.spk_embed_layer(spk_embedding)  # 192 → 1024
   llm_input = text_repr + spk_embed
   ```

2. **指令控制** (僅 Instruct 模型):
   ```python
   # 指令文字編碼
   instruct_repr = self.text_encoder(instruct_tokens)
   # 與主文字拼接
   combined_input = torch.cat([instruct_repr, text_repr], dim=1)
   ```

3. **零樣本控制**:
   ```python
   # 參考語音編碼
   ref_speech_repr = self.speech_encoder(ref_speech)
   # 跨模態對齊
   aligned_repr = self.cross_modal_proj(ref_speech_repr)
   ```

**RAS (Repetition Aware Sampling)**:
```python
def ras_sampling(logits, win_size=10, tau_r=0.1, top_p=0.8, top_k=25):
    # 檢測窗口內重複 token
    repeated_tokens = detect_repetition(prev_tokens, win_size)
    
    # 抑制重複 token 的概率
    for token in repeated_tokens:
        logits[token] -= tau_r
    
    # 標準 top-p/top-k 採樣
    return sample_with_top_p_k(logits, top_p, top_k)
```

**關鍵特性**:
- **多語言支援**: 基於 Whisper tokenizer，支援 100+ 語言
- **零樣本能力**: 透過參考語音實現任意音色克隆
- **指令理解**: 自然語言控制語音特性
- **重複抑制**: RAS 機制避免生成重複內容
- **長度正規化**: 避免長度偏差
- **標籤平滑**: 提升泛化能力

**訓練策略**:
1. **預訓練階段**: 大規模無監督文字-語音對齊
2. **SFT 階段**: 有監督微調，學習特定音色
3. **指令調優**: 學習指令-語音特性映射
4. **多任務學習**: 同時學習多種語音生成任務

**性能優化**:
- **KV 快取**: 加速自回歸生成
- **批次推理**: 並行處理多個序列
- **動態填充**: 減少無效計算
- **梯度檢查點**: 節省訓練記憶體

## 分詞器模組 (cosyvoice.tokenizer)

分詞器模組負責將文字轉換為模型可處理的 token 序列，支援多語言和特殊控制符號。

### tokenizer.py
主要的分詞器實現和管理。

**核心功能**:
```python
def get_tokenizer(multilingual=True, num_languages=100, language='en', task='transcribe'):
    # 返回配置好的 Whisper tokenizer
    return whisper.tokenizer.get_tokenizer(
        multilingual=multilingual,
        num_languages=num_languages, 
        language=language,
        task=task
    )
```

**tokenizer 特性**:

1. **基於 Whisper**:
   - 使用 OpenAI Whisper 的多語言 tokenizer
   - 支援 100+ 語言的統一詞彙表
   - 詞彙表大小：51,866 tokens（標準）或 60,515 tokens（25Hz版本）

2. **多語言支援**:
   - 中文：`<|zh|>` 標記
   - 英文：`<|en|>` 標記
   - 日文：`<|ja|>` 標記  
   - 韓文：`<|ko|>` 標記
   - 粵語：`<|yue|>` 標記
   - 其他語言：對應的 ISO 語言代碼

3. **特殊控制 Token**:
   ```python
   # 語言標記
   '<|zh|>', '<|en|>', '<|ja|>', '<|ko|>', '<|yue|>'
   
   # 任務標記  
   '<|transcribe|>', '<|translate|>'
   
   # 時間標記
   '<|0.00|>', '<|0.02|>', ..., '<|30.00|>'
   
   # 特殊標記
   '<|startoftranscript|>', '<|endoftext|>'
   '<|notimestamps|>', '<|nospeech|>'
   ```

4. **CosyVoice 擴展控制符**:
   ```python
   # 情感控制（v2.0+）
   '[laughter]'     # 笑聲
   '[breath]'       # 呼吸聲
   '<strong></strong>'  # 強調
   '<laughter></laughter>'  # 笑聲段落
   
   # 韻律控制
   '[pause]'        # 停頓
   '[fast]'         # 語速加快
   '[slow]'         # 語速放慢
   ```

### 資源檔案

**multilingual_zh_ja_yue_char_del.tiktoken**:
- BPE（Byte Pair Encoding）詞彙表
- 針對中日粵語言優化
- 字符級別的細粒度分詞
- 支援罕見字和專有名詞

**分詞處理流程**:

1. **文字預處理**:
   ```python
   def preprocess_text(text):
       # 標準化標點符號
       text = normalize_punctuation(text)
       # 處理特殊字符
       text = handle_special_chars(text)
       # 添加語言標記
       text = add_language_tags(text)
       return text
   ```

2. **Token 化**:
   ```python
   def tokenize(text, allowed_special='all'):
       # 編碼為 token ID
       tokens = tokenizer.encode(
           text, 
           allowed_special=allowed_special
       )
       return tokens
   ```

3. **特殊符號處理**:
   ```python
   # 允許所有特殊符號
   allowed_special = 'all'
   
   # 僅允許特定符號
   allowed_special = {'<|zh|>', '<|en|>', '[laughter]'}
   
   # 禁用特殊符號
   allowed_special = set()
   ```

**使用範例**:

```python
# 基本使用
from cosyvoice.tokenizer.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

# 中文分詞
chinese_text = "你好，世界！"
tokens = tokenizer.encode(chinese_text)
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

# 多語言混合
mixed_text = "<|zh|>你好<|en|>Hello<|ja|>こんにちは"
tokens = tokenizer.encode(mixed_text, allowed_special='all')

# 情感控制
emotion_text = "這真是太棒了[laughter]！"
tokens = tokenizer.encode(emotion_text, allowed_special='all')

# 強調控制
emphasis_text = "這是<strong>非常重要</strong>的信息。"
tokens = tokenizer.encode(emphasis_text, allowed_special='all')
```

**詞彙表統計**:
- **總詞彙量**: 51,866 (標準) / 60,515 (25Hz)
- **中文 token**: ~15,000
- **英文 token**: ~25,000  
- **日文 token**: ~8,000
- **韓文 token**: ~2,000
- **特殊標記**: ~500
- **控制符號**: ~100

**優化特性**:
- **高效編碼**: BPE 算法，平衡詞彙量和序列長度
- **語言適應**: 針對東亞語言優化的分詞策略
- **擴展性**: 易於添加新的控制符號和語言支援
- **兼容性**: 與 Whisper 生態系統完全兼容

## Transformer 模組 (cosyvoice.transformer)

Transformer 模組提供了 CosyVoice 所需的各種 Transformer 架構組件，包括編碼器、解碼器、注意力機制等。

### 核心架構文件

**encoder.py** - 編碼器實現:
```python
class TransformerEncoder(torch.nn.Module):
    # 標準 Transformer 編碼器
    def __init__(self, input_size, output_size=None, attention_heads=4, 
                 linear_units=2048, num_blocks=6, dropout_rate=0.1, 
                 input_layer="linear", pos_enc_layer_type="abs_pos", 
                 normalize_before=True, static_chunk_size=0)

class ConformerEncoder(torch.nn.Module):
    # Conformer 編碼器（結合卷積和自注意力）
    def __init__(self, input_size, output_size=None, attention_heads=4,
                 linear_units=2048, num_blocks=6, dropout_rate=0.1,
                 use_cnn_module=True, cnn_module_kernel=31, 
                 macaron_style=False, use_dynamic_chunk=False)
```

**decoder.py** - 解碼器實現:
```python
class TransformerDecoder(torch.nn.Module):
    # 標準 Transformer 解碼器
    def __init__(self, vocab_size, encoder_output_size, attention_heads=4,
                 linear_units=2048, num_blocks=6, dropout_rate=0.1,
                 self_attention_dropout_rate=0.0, src_attention_dropout_rate=0.0)
```

### attention.py
注意力機制實現。

**多頭注意力變體**:

1. **MultiHeadedAttention**:
   ```python
   class MultiHeadedAttention(torch.nn.Module):
       def __init__(self, n_head, n_feat, dropout_rate=0.0)
       def forward(self, query, key, value, mask=None)
   ```

2. **RelPositionMultiHeadedAttention**:
   ```python
   # 相對位置編碼注意力
   class RelPositionMultiHeadedAttention(torch.nn.Module):
       def __init__(self, n_head, n_feat, dropout_rate=0.0, zero_triu=False)
   ```
   - 特點：使用相對位置編碼，適合序列任務
   - 應用：文字編碼器和 LLM 主體

3. **LegacyRelPositionMultiHeadedAttention**:
   - 向後兼容的相對位置注意力實現
   - 用於載入舊版本模型

**注意力優化**:
- **Flash Attention**: 記憶體高效的注意力計算
- **Sparse Attention**: 稀疏注意力模式
- **Linear Attention**: 線性複雜度注意力

### embedding.py
嵌入層實現。

**位置編碼**:
```python
class PositionalEncoding(torch.nn.Module):
    # 絕對位置編碼
    def __init__(self, d_model, dropout_rate, max_len=5000)

class RelPositionalEncoding(torch.nn.Module):
    # 相對位置編碼
    def __init__(self, d_model, dropout_rate, max_len=5000)

class LearnablePositionalEncoding(torch.nn.Module):
    # 可學習位置編碼
    def __init__(self, d_model, max_len=5000)
```

**特殊嵌入**:
```python
class TokenEmbedding(torch.nn.Module):
    # Token 嵌入
    def __init__(self, vocab_size, d_model)

class SpeakerEmbedding(torch.nn.Module):
    # 說話人嵌入
    def __init__(self, n_speakers, d_model)
```

### convolution.py
Conformer 中的卷積模組。

**ConvolutionModule**:
```python
class ConvolutionModule(torch.nn.Module):
    def __init__(self, channels, kernel_size=31, activation=Swish(), 
                 norm="batch_norm", causal=False, bias=True)
    def forward(self, x)
```

**特點**:
- **深度卷積**: 捕捉局部特徵
- **批次歸一化**: 訓練穩定性
- **Swish 激活**: 平滑非線性
- **因果卷積**: 支援串流推理

### encoder_layer.py & decoder_layer.py
Transformer 層級組件。

**EncoderLayer**:
```python
class EncoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate, 
                 normalize_before=True, stochastic_depth_rate=0.0)
    def forward(self, x, mask, cache=None)
```

**ConformerEncoderLayer**:
```python
class ConformerEncoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, feed_forward, feed_forward_macaron,
                 conv_module, dropout_rate, normalize_before=True,
                 stochastic_depth_rate=0.0)
```

**層級架構**:
1. **自注意力子層**: 捕捉長距離依賴
2. **卷積子層**: 捕捉局部模式（Conformer 特有）
3. **前饋子層**: 非線性變換
4. **殘差連接**: 梯度流動
5. **層歸一化**: 訓練穩定

### positionwise_feed_forward.py
位置級前饋網路。

**PositionwiseFeedForward**:
```python
class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate, 
                 activation=torch.nn.ReLU(), normalize_before=True)
    def forward(self, x)
```

**變體**:
- **標準 FFN**: ReLU 激活
- **Swish FFN**: Swish 激活，更平滑
- **GLU FFN**: 門控線性單元

### subsampling.py
子採樣層，用於降低序列長度。

**Conv2dSubsampling**:
```python
class Conv2dSubsampling(torch.nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None)
    def forward(self, x, x_mask)
```

**功能**:
- **降採樣**: 減少計算複雜度
- **特徵提取**: 2D 卷積提取特徵
- **位置編碼**: 自動添加位置信息

### activation.py
激活函數實現。

**支援的激活函數**:
```python
class Swish(torch.nn.Module):
    # x * sigmoid(x)
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(torch.nn.Module):
    # x * tanh(softplus(x))
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class GELU(torch.nn.Module):
    # 高斯誤差線性單元
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
```

### upsample_encoder.py
上採樣編碼器，用於時序對齊。

**UpsampleEncoder**:
```python
class UpsampleEncoder(torch.nn.Module):
    def __init__(self, input_size, output_size, upsample_rate=4)
    def forward(self, x, x_lengths=None)
```

**應用場景**:
- **時序對齊**: 文字和語音特徵對齊
- **解析度匹配**: 不同模組間的特徵匹配
- **串流處理**: 支援即時上採樣

### label_smoothing_loss.py
標籤平滑損失函數。

**LabelSmoothingLoss**:
```python
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, size, padding_idx, smoothing, normalize_length=False)
    def forward(self, x, target)
```

**優勢**:
- **泛化提升**: 避免過擬合
- **置信度調節**: 降低過度自信
- **訓練穩定**: 平滑梯度更新

**模組集成架構**:
```
TransformerLM
├── TextEncoder (ConformerEncoder)
│   ├── ConformerEncoderLayer × 6
│   │   ├── MultiHeadedAttention
│   │   ├── ConvolutionModule  
│   │   └── PositionwiseFeedForward
│   └── RelPositionalEncoding
└── LLM (TransformerEncoder)
    ├── EncoderLayer × 14
    │   ├── RelPositionMultiHeadedAttention
    │   └── PositionwiseFeedForward
    └── TokenEmbedding + PositionalEncoding
```

## 工具模組 (cosyvoice.utils)

工具模組提供了支援整個 CosyVoice 系統運行的各種輔助功能和工具函數。

### common.py
通用工具函數和常量定義。

**核心功能**:

1. **隨機種子控制**:
   ```python
   def set_all_random_seed(seed):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
   ```

2. **RAS 採樣**:
   ```python
   def ras_sampling(logits, win_size=10, tau_r=0.1, top_p=0.8, top_k=25):
       # Repetition Aware Sampling
       # 避免生成重複內容的採樣策略
   ```

3. **音頻後處理**:
   ```python
   def fade_in_out(audio, fade_in_len=1000, fade_out_len=1000):
       # 音頻淡入淡出處理
       
   def normalize_audio(audio, target_peak=0.9):
       # 音頻標準化
   ```

4. **常量定義**:
   ```python
   IGNORE_ID = -1           # 填充標記ID
   EPS = 1e-12             # 數值穩定性常量
   MAX_DECODE_LENGTH = 2048 # 最大解碼長度
   ```

5. **精度計算**:
   ```python
   def th_accuracy(pad_outputs, pad_targets, ignore_label):
       # 計算 token 級別的準確率
   ```

6. **TensorRT 包裝器**:
   ```python
   class TrtContextWrapper:
       # TensorRT 推理上下文管理
       def __init__(self, trt_file, trt_concurrent=1)
       def __call__(self, **kwargs)
   ```

### file_utils.py
檔案和模型管理工具。

**主要功能**:

1. **音頻載入**:
   ```python
   def load_wav(wav_path, target_sample_rate=16000):
       # 載入音頻檔案並重採樣
       audio, sr = torchaudio.load(wav_path)
       if sr != target_sample_rate:
           audio = torchaudio.functional.resample(audio, sr, target_sample_rate)
       return audio.squeeze(0)  # 移除批次維度
   ```

2. **模型轉換**:
   ```python
   def convert_onnx_to_trt(onnx_file, trt_file, fp16=False, max_batch_size=1, 
                          max_workspace_size=1<<30):
       # 將 ONNX 模型轉換為 TensorRT 格式
   ```

3. **vLLM 導出**:
   ```python
   def export_cosyvoice2_vllm(model_dir, output_dir):
       # 導出 CosyVoice2 模型用於 vLLM 推理
   ```

4. **日誌配置**:
   ```python
   import logging
   
   # 配置全局日誌
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

### class_utils.py
類型和模型管理工具。

**功能**:
```python
def get_model_type(configs):
    # 根據配置判斷模型類型
    if 'llm' in configs and hasattr(configs['llm'], '__name__'):
        if 'CosyVoice2' in configs['llm'].__name__:
            return CosyVoice2Model
    return CosyVoiceModel

def instantiate_model(config):
    # 根據配置實例化模型
    model_class = get_model_type(config)
    return model_class(**config)
```

### executor.py
分散式訓練和推理執行器。

**Executor 類別**:
```python
class Executor:
    def __init__(self, rank=0, world_size=1, device='cuda'):
        self.rank = rank
        self.world_size = world_size
        self.device = device
    
    def train(self, model, dataloader, optimizer, scheduler):
        # 訓練循環
    
    def evaluate(self, model, dataloader):
        # 評估循環
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        # 保存檢查點
    
    def load_checkpoint(self, checkpoint_path):
        # 載入檢查點
```

**分散式支援**:
- **DDP**: 分散式資料並行
- **模型並行**: 大模型分片
- **梯度同步**: 跨節點梯度聚合
- **檢查點管理**: 分散式檢查點保存/載入

### frontend_utils.py
前端文字處理工具。

**文字處理函數**:

1. **語言檢測**:
   ```python
   def contains_chinese(text):
       # 檢查文字是否包含中文字符
       return bool(re.search(r'[\u4e00-\u9fff]', text))
   
   def detect_language(text):
       # 自動檢測文字語言
   ```

2. **文字清理**:
   ```python
   def replace_blank(text):
       # 替換空白字符
       return re.sub(r'\s+', ' ', text).strip()
   
   def replace_corner_mark(text):
       # 替換角標符號
   
   def remove_bracket(text):
       # 移除括號內容
   ```

3. **數字處理**:
   ```python
   def spell_out_number(text, lang='zh'):
       # 將數字轉換為文字
       if lang == 'zh':
           return convert_chinese_number(text)
       elif lang == 'en':
           return convert_english_number(text)
   ```

4. **文本分割**:
   ```python
   def split_paragraph(text, max_length=200, lang='zh'):
       # 按語言特性分割長文本
       if lang == 'zh':
           return split_chinese_text(text, max_length)
       else:
           return split_western_text(text, max_length)
   ```

5. **標點處理**:
   ```python
   def is_only_punctuation(text):
       # 檢查是否只含標點符號
       return bool(re.match(r'^[\s\p{P}]+$', text))
   ```

### losses.py
各種損失函數實現。

**損失函數**:

1. **TPR 損失**:
   ```python
   def tpr_loss(y_d_gs, y_d_rs, tau=0.04):
       # Timbre Perturbation Regularization
       # 音色擾動正則化損失
   ```

2. **Mel 頻譜損失**:
   ```python
   def mel_loss(y_true, y_pred, mel_transforms):
       # 多尺度 Mel 頻譜重建損失
       total_loss = 0
       for mel_transform in mel_transforms:
           mel_true = mel_transform(y_true)
           mel_pred = mel_transform(y_pred)
           total_loss += F.l1_loss(mel_pred, mel_true)
       return total_loss
   ```

3. **特徵匹配損失**:
   ```python
   def feature_matching_loss(fmap_r, fmap_g):
       # 判別器特徵匹配損失
       loss = 0
       for dr, dg in zip(fmap_r, fmap_g):
           for rl, gl in zip(dr, dg):
               loss += F.l1_loss(gl, rl.detach())
       return loss
   ```

### mask.py
掩碼生成和處理工具。

**掩碼函數**:
```python
def make_pad_mask(lengths, max_len=None):
    # 生成填充掩碼
    if max_len is None:
        max_len = lengths.max()
    
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, dtype=torch.long, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    
    return seq_range_expand >= seq_length_expand

def make_non_pad_mask(lengths, max_len=None):
    # 生成非填充掩碼
    return ~make_pad_mask(lengths, max_len)

def subsequent_mask(size, device="cpu"):
    # 生成後續位置掩碼（用於解碼器）
    ret = torch.ones(size, size, device=device, dtype=torch.bool)
    return torch.tril(ret)
```

### scheduler.py
學習率調度器。

**調度器類型**:
```python
class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    # 預熱學習率調度器
    def __init__(self, optimizer, warmup_steps=4000, d_model=512)
    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [base_lr * min(step ** -0.5, step * self.warmup_steps ** -1.5) 
                for base_lr in self.base_lrs]

class NoamLR(WarmupLR):
    # Noam 調度器（Transformer 原論文使用）
    pass

class ConstantLR(torch.optim.lr_scheduler._LRScheduler):
    # 恆定學習率
    def get_lr(self):
        return self.base_lrs
```

### train_utils.py
訓練相關工具函數。

**訓練輔助**:
```python
def get_gradient_norm(model):
    # 計算模型梯度範數
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

def clip_gradient(model, clip_value):
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

def count_parameters(model):
    # 統計模型參數量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, optimizer, scheduler, epoch, loss, path):
    # 保存模型檢查點
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)
```

**整合使用範例**:
```python
# 完整的工具模組使用
from cosyvoice.utils.common import set_all_random_seed, ras_sampling
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.frontend_utils import contains_chinese, spell_out_number
from cosyvoice.utils.mask import make_pad_mask

# 設定隨機種子
set_all_random_seed(42)

# 載入音頻
audio = load_wav('path/to/audio.wav', 16000)

# 文字處理
text = "今天溫度是25度"
if contains_chinese(text):
    text = spell_out_number(text, 'zh')

# 生成掩碼
lengths = torch.tensor([10, 8, 12])
mask = make_pad_mask(lengths, max_len=15)

# RAS 採樣
logits = torch.randn(1, 50257)  # 詞彙表大小
sampled_token = ras_sampling(logits, top_p=0.8, top_k=25)
```

## vLLM 整合 (cosyvoice.vllm)

vLLM 整合模組提供了高性能的大語言模型推理加速，專門為 CosyVoice2 優化。

### cosyvoice2.py
CosyVoice2 的 vLLM 適配實現。

**CosyVoice2ForCausalLM 類別**:
```python
class CosyVoice2ForCausalLM(nn.Module):
    def __init__(self, config, cache_config=None, quant_config=None):
        super().__init__()
        self.config = config
        self.llm = LlamaModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.sampler = Sampler()
```

**核心特性**:

1. **高性能推理**:
   - **PagedAttention**: 動態 KV 快取管理
   - **連續批次處理**: 優化 GPU 利用率
   - **張量並行**: 多 GPU 模型並行
   - **動態 Batching**: 自適應批次大小

2. **記憶體優化**:
   - **KV 快取分頁**: 減少記憶體碎片
   - **動態形狀**: 避免固定大小的記憶體浪費
   - **快取重用**: 跨請求共享 KV 快取

3. **並發支援**:
   - **非同步處理**: 並發處理多個請求
   - **請求調度**: 智能請求排程
   - **負載均衡**: 多 GPU 負載分散

**vLLM 配置**:
```python
# vLLM 引擎配置
from vllm import LLM, SamplingParams

# 模型載入
llm = LLM(
    model="pretrained_models/CosyVoice2-0.5B",
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.8,
    tensor_parallel_size=2,  # 使用 2 個 GPU
    dtype="float16"
)

# 採樣參數
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.8,
    top_k=25,
    max_tokens=1024,
    stop_token_ids=[50256]  # EOS token
)
```

**推理流程**:
```python
def vllm_inference(text_tokens, speaker_embedding):
    # 1. 準備輸入
    input_data = {
        'input_ids': text_tokens,
        'speaker_embedding': speaker_embedding
    }
    
    # 2. vLLM 推理
    outputs = llm.generate(
        prompts=[input_data],
        sampling_params=sampling_params,
        use_tqdm=False
    )
    
    # 3. 提取語音 tokens
    speech_tokens = outputs[0].outputs[0].token_ids
    
    return speech_tokens
```

**性能優勢**:

1. **推理速度提升**:
   - 比標準 PyTorch 快 2-5 倍
   - 支援動態批次處理
   - 優化的注意力計算

2. **記憶體效率**:
   - 減少 30-50% 記憶體使用
   - 動態 KV 快取管理
   - 智能記憶體分配

3. **並發能力**:
   - 支援數百個並發請求
   - 自動負載均衡
   - 請求排隊優化

**與 CosyVoice2 整合**:
```python
class CosyVoice2:
    def __init__(self, model_dir, load_vllm=True, **kwargs):
        if load_vllm:
            # 載入 vLLM 加速的 LLM
            self.llm_engine = LLM(
                model=model_dir,
                trust_remote_code=True,
                **kwargs
            )
        else:
            # 載入標準 PyTorch 模型
            self.llm = self._load_standard_model(model_dir)
    
    def inference_zero_shot_vllm(self, text, prompt_text, prompt_speech):
        # 使用 vLLM 加速的零樣本推理
        text_tokens = self.tokenizer.encode(text)
        speaker_emb = self.extract_speaker_embedding(prompt_speech)
        
        speech_tokens = vllm_inference(text_tokens, speaker_emb)
        
        # 後續 Flow 和 HiFiGAN 處理保持不變
        mel_spec = self.flow.inference(speech_tokens, speaker_emb)
        waveform = self.hifigan.inference(mel_spec)
        
        return waveform
```

**配置選項**:

```python
# 高性能配置
vllm_config_high_perf = {
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 256,
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1,
    "block_size": 16,
    "swap_space": 4,  # GB
    "disable_log_stats": True
}

# 記憶體優化配置
vllm_config_memory = {
    "gpu_memory_utilization": 0.7,
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 64,
    "tensor_parallel_size": 1,
    "block_size": 8,
    "swap_space": 8,  # GB
    "enable_prefix_caching": True
}

# 延遲優化配置
vllm_config_latency = {
    "gpu_memory_utilization": 0.8,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 32,
    "tensor_parallel_size": 2,
    "speculative_model": None,
    "num_speculative_tokens": 0,
    "block_size": 32
}
```

**監控和調試**:
```python
# vLLM 統計信息
def get_vllm_stats(llm_engine):
    stats = llm_engine.get_model_config()
    return {
        "num_requests_running": stats.num_requests_running,
        "num_requests_waiting": stats.num_requests_waiting, 
        "gpu_cache_usage": stats.gpu_cache_usage,
        "cpu_cache_usage": stats.cpu_cache_usage,
        "num_preempted": stats.num_preempted
    }

# 性能監控
def benchmark_vllm_vs_pytorch(test_inputs, num_runs=10):
    # vLLM 性能測試
    vllm_times = []
    for _ in range(num_runs):
        start = time.time()
        vllm_output = llm_vllm.generate(test_inputs, sampling_params)
        vllm_times.append(time.time() - start)
    
    # PyTorch 性能測試
    pytorch_times = []
    for _ in range(num_runs):
        start = time.time()
        pytorch_output = llm_pytorch(test_inputs)
        pytorch_times.append(time.time() - start)
    
    return {
        "vllm_avg_time": np.mean(vllm_times),
        "pytorch_avg_time": np.mean(pytorch_times),
        "speedup": np.mean(pytorch_times) / np.mean(vllm_times)
    }
```

**安裝和依賴**:
```bash
# 安裝 vLLM
pip install vllm==0.9.0

# 或從源碼安裝最新版
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# 驗證安裝
python -c "import vllm; print(vllm.__version__)"
```

**注意事項**:
1. **版本兼容**: 需要 vllm==0.9.0 或更高版本
2. **硬體需求**: 需要支援 CUDA 的 GPU
3. **記憶體需求**: 建議至少 16GB GPU 記憶體
4. **模型格式**: 需要轉換為 vLLM 兼容格式
5. **批次推理**: vLLM 在批次推理時效果最佳

**故障排除**:
- **OOM 錯誤**: 降低 `gpu_memory_utilization` 和 `max_num_seqs`
- **速度慢**: 增加 `max_num_batched_tokens` 和批次大小
- **兼容性**: 檢查 CUDA 版本和 PyTorch 版本兼容性