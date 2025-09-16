TasteSLMFusing

定位：把「文字嵌入」與「taste（來自音訊的語音潛表示）」對齊後融合成 LLM 的輸入。
關鍵點：透過 delay 讓 taste 往前（序列開頭補 pad）、text 往後（序列結尾補 pad），確保兩模態在同一時間軸上對齊，再交給你選的 fusion mixer 去做融合。

重要參數

llm_input_size：融合後要餵給 SLM/LLM 的維度。

tokenizer_output_size：taste tokenizer 輸出的維度（若不給，預設等於 llm_input_size，並用線性層 taste_embed_in 投影）。

class_name / fuse_config：從 TTS_INPUT_FUSION_CLASSES 挑一個融合器（例如 weighted_sum / cross_attention 等）與其超參。

主要成員

self.mixer：實際做融合的模組（回傳 (fused, extra)，這裡只用 fused）。

self.pad_taste_embed、self.pad_text_embed：可學的 padding 向量（遇到 delay 需要補位時用）。

self.taste_embed_in：把 taste tokenizer 的輸出維度投影到 llm_input_size。

forward(...)

輸入：

text_token_emb: (1, T_text, D)

taste_token_emb: (1, T_taste, D_or_tokenizer_out)

text_token_len: (1,) 文字長度

delay: int

流程：

把 taste_token_emb 先線性投影到 llm_input_size，再在前面補 delay 個 pad_taste_embed → shifted_taste，長度 T_taste + delay。

把 text_token_emb 在後面補 delay 個 pad_text_embed → shifted_text，長度 T_text + delay。

餵進 self.mixer，對 mixer 報告的有效長度都用 text_token_len + delay（這裡隱含假設 T_taste == T_text；若 taste 長度不等，應改傳各自真實長度）。

回傳 fused_emb，形狀大約是 (1, T_text + delay, D)。

踩雷：batch_size 假設為 1。後續 prepare_lm_input_target 會逐樣本處理，因此整體仍支援 batch，只是「融合本身」這步是逐樣本呼叫的。

TasteSLMOut

定位：把語言模型隱狀態 lm_output 同步做兩件事：
(1) 文字 vocab 分類（CE loss）；(2) taste 潛向量 z 的變分式學習（MSE + 類 KL）。

重要參數

llm_output_size：進來的 hidden 維度

text_vocab_size：字彙表大小（做 CE）

d：taste 潛空間維度

b_logvar_is_linear：若 True，用線性層預測逐位置 logvar；否則用一個全域可學參數（broadcast 成 (B,T,d)）

conduct_reparameterization：訓練時是否走 reparameterization（μ+σ·ε）

主要成員

fc_mu: (llm_output_size → d)，預測潛向量的均值 μ

b_logvar: 線性層或 Parameter，決定 log σ²

ce_loss_module, mse_loss_module

關鍵方法

_calculate_loss_text_ce(logits, labels)：把 (B,T,C) 攤平成 (B·T,C) 對 (B·T,) 做標準 CE，ignore_index = -1 用來忽略不訓練的位置。

_reparameterize(mu, sigma)：標準 VAE reparam。

predict_taste_latent(hidden)：

由 hidden 生 μ、logvar→σ，若開啟 reparam 就取 z = μ + σ·ε，否則用 μ。

回傳 (z, mu, logvar)，形狀皆為 (B,T,d)。

_calculate_loss_taste_mse(z, mu, logvar, target, mask)：

先把 mask 篩出有效位置。

MSE：MSE(z, target) on masked positions。

「KL」：此實作不是「對 N(0,I) 的 KL」，而是用
0.5 * mean( exp(logvar) + (mu - target)^2 * exp(-logvar) - 1 - logvar )
——這較像「對以 target 為均值的高斯」的負對數似然正則。

再用 0.5/0.5 加權合併成 taste_loss。

forward(lm_output, text_logit, lm_text_target, lm_taste_latent_target, lm_taste_mask)：

text_loss = CE(text_logit, lm_text_target)

(z, mu, logvar) = predict_taste_latent(lm_output)

taste_loss = MSE+KL(...)

loss = 0.5*text + 0.5*taste

text_acc = th_accuracy(...)（會忽略 IGNORE_ID）

回傳 dict：{'loss', 'text_acc', 'taste_loss'}

小提醒：註解曾寫會回傳 z，但目前實作沒有把 z 放進回傳 dict（若你需要監看，可加上）。

TasteSLM

定位：主控流程（資料→嵌入→對齊融合→SLM→雙頭輸出），同時兼顧一般訓練與 SFT（supervised fine-tuning）模式，以及逐步解碼的推論（generator）。

重要屬性

taste_stage1：你的一階 taste tokenizer / 編碼器（taste_stage1.taste_tokenizer(...) 會吐出 taste_token_emb 與 taste_latent）

slm：語言模型骨幹，要提供：

forward_embed_tokens(text_ids) → (B,T,D_embed)

__call__(lm_input, lm_input_len) → (lm_output, lm_output_mask)

forward_lm_head(lm_output) → (B,T,V)

forward_one_step(lm_input, masks, cache) → (hidden_pred, cache)

fusing_module：上面的 TasteSLMFusing

out_module：上面的 TasteSLMOut

delay、ignore_id、eos_token_id、text_sampling_callable

核心方法

prepare_lm_input_target(...)（最重要）
逐樣本（unpad→for 迴圈）做：

文字目標 lm_text_target：把原始 text_token 右移一位（丟掉第一顆，補上 EOS），再在尾端補 (delay-1) 個 IGNORE_ID，長度變成 T_text + delay - 1。

taste 目標 lm_taste_latent_target：在前端補 (delay-1) 個全 0 的 d 維向量，再接上原始 taste_latent（對齊策略等同 taste 提前）。

taste mask：前 (delay-1) 位置是 False，其餘與 taste_token_emb 的有效長度對齊為 True。

融合輸入 lm_input：呼叫 fusing_module(text_emb, taste_emb, text_len, delay) 得到 (1, T_text+delay, D)，再砍掉最後一格 → (1, T_text+delay-1, D)，讓輸入長度與 lm_text_target、lm_taste_* 對齊。

最後用 pad_sequence 把 batch 維湊起來，並回傳對應 lm_input_len。

_prepare_for_sft_training(batch, device)
把 batch['data'][i]['messages']（system / user / assistant）在序列維度串接成一個長序列的 text_token / text_token_len / taste_token_emb / taste_latent，讓上面的 prepare_lm_input_target 可以一次處理整段對話。這段程式裡目前有 TODO（我在下一節會給你一個可直接替換的實作）。

forward(batch, device)（訓練路徑）

若 batch['sft_training'] is True → 用 _prepare_for_sft_training 產生四個主張量；
否則就從 taste_stage1.taste_tokenizer(...) 現場把 taste_* 算出來。

text_token_emb = slm.forward_embed_tokens(text_token)

用 prepare_lm_input_target(...) 生 lm_*

lm_output, lm_output_mask = slm(lm_input, lm_input_len)

text_logit = slm.forward_lm_head(lm_output)

outputs = out_module(...) → 回傳 loss / acc / taste_loss

inference(...)（逐步生成器）

先把 text_token_emb 與 taste_token_emb 融合，取 [:-delay] 作為初始 lm_input，並取最後 delay 步的 taste 當作「緩衝」。

在 inference_wrapper(...) 裡逐步：

以 forward_one_step 得當前 hidden_pred

用 forward_lm_head+text_sampling_callable 取下一顆文字 id（必要時忽略 EOS）

產生對應的 taste 嵌入（用 out_module.predict_taste_latent + taste_stage1 的投影器）

以 fusing_module(..., delay=0) 把這一對 (text_emb, taste_emb) 再次融合，作為下一次 single step 的輸入

以 generator 方式 yield (text_id, taste_emb)

小 bug：inference(...) 中 tokenized = self.taste_tokenizer(...) 應該是 self.taste_stage1.taste_tokenizer(...)。記得更正。

2) 每個 class 的各函式細節（含形狀與對齊）

為節省篇幅，上面其實已把每個 method 的 I/O、形狀、邏輯與注意事項寫得很細；這裡補幾個「你在調參與除錯時特別用得到的要點」：

延遲對齊（delay）

taste：前補 delay，等於把 taste 往未來對齊（讓之後某文字位置能「看到」較早出現的語音條件）。

text：後補 delay，確保兩者序列長度一致後再 fuse。

訓練時 lm_input 會少 1 格，目的是讓 t=0..L-1 的 hidden 預測 t=1..L 的目標（文字右移 + EOS）。

目標與 mask

文字：IGNORE_ID 會在 CE 與 th_accuracy 裡被忽略。

taste：前面 (delay-1) 步是無效，因此 lm_taste_mask[:delay-1]=False。

Taste 變分項

現行「KL」其實是對「以 target 為均值」的高斯的 NLL 型式正則，不是標準 VAE 對 N(0,I) 的 KL。這是設計選擇，不是錯誤；但若你原本想要標準 VAE，公式要改。

SFT 打包

若希望只對 assistant 的 token 算 CE，把 user/system 的那段 text_token 位置先改成 IGNORE_ID，再丟進 prepare_lm_input_target（因為它會 shift，一開始的 IGNORE_ID 也會跟著位移）。

若你希望 user 段落也能提供 taste 條件訊號，但不貢獻 CE，taste_* 仍然可以保留、只把文字標成忽略即可。

# TASTE SLM SFT Implementation Plan

## Overview
This document outlines the implementation plan for adding Supervised Fine-Tuning (SFT) capabilities to the TASTE Speech Language Model. The implementation focuses on conversation-style training with special tokens and multimodal alignment.

## Requirements Analysis

### 1. Special Token Integration
- **Goal**: Add `<im_start>` and `<im_end>` tokens to wrap each role message
- **Format**: `<im_start>{role}\n{content}\n<im_end>`
- **Roles**: `system`, `user`, `assistant`
- **Token IDs**: 
  - `<im_start>`: 151644
  - `<im_end>`: 151645
  - Role tokens: `system` (6125), `user` (882), `assistant` (78191)
  - Newline: 198

### 2. Batch Data Processing
- **Current**: Single message processing in `_prepare_for_sft_training`
- **Target**: Message-level tokenization with conversation concatenation
- **Process**: 
  1. Split messages by role
  2. Add special tokens to each message
  3. Concatenate messages within conversation
  4. Batch conversations with padding

### 3. Trainable Silence Tokens
- **Purpose**: Provide learnable representations for special tokens that don't have audio
- **Tokens needing silence**: `<im_start>`, `<im_end>`, role names, newlines
- **Implementation**: Use `fusing_module.pad_taste_embed` as base for silence tokens
- **Shape**: Must match taste embedding dimensions

### 4. Shift and Padding Mechanism
- **Current**: Uses delay-based shifting from pretrain model
- **Target**: Maintain compatibility with existing processor padding
- **Components**:
  - Text token padding (matches existing `prepare_lm_input_target`)
  - Taste token padding (aligned with text)
  - Sequence length alignment after special token insertion

### 5. Latent Space Alignment
- **Requirement**: Latent space must match sequence length after padding
- **Current**: `taste_latent` dimension is `[batch, seq_len, d]` where `d=256`
- **Target**: Ensure latent space includes silence representations for special tokens
- **Shape consistency**: All sequences (text, taste_emb, taste_latent) must have same seq_len

## Implementation Strategy

### Phase 1: Special Token Processing
**File**: `taste_slm.py:472-541` (`_prepare_for_sft_training` method)

**Changes**:
1. Define special token IDs as constants
2. For each message role:
   - Prepend `<im_start>{role}\n`
   - Append `\n<im_end>`
   - Update text_token_len accordingly
3. Handle system messages (no audio) with silence tokens only
4. Handle user/assistant messages (with audio) by combining silence + taste embeddings

**Key Considerations**:
- Maintain tensor shapes consistency
- Ensure proper device placement
- Handle variable message lengths

### Phase 2: Silence Token Integration
**File**: `taste_slm.py:30-124` (`TasteSLMFusing` class)

**Changes**:
1. Add specialized silence token embeddings for roles
2. Create method to generate silence tokens for special tokens
3. Ensure silence tokens are trainable parameters
4. Maintain compatibility with existing `pad_taste_embed`

**New Parameters**:
```python
self.silence_im_start = nn.parameter.Parameter(torch.zeros(llm_input_size))
self.silence_im_end = nn.parameter.Parameter(torch.zeros(llm_input_size))
self.silence_role = nn.parameter.Parameter(torch.zeros(llm_input_size))
```

### Phase 3: Batch Processing Enhancement
**File**: `taste_slm.py:472-541` (continuation of Phase 1)

**Changes**:
1. Modify concatenation logic to handle special tokens
2. Update padding to accommodate variable special token counts
3. Ensure proper mask generation for training
4. Maintain compatibility with `prepare_lm_input_target`

**Sequence Processing**:
```
Original: [text_tokens]
Target:   [<im_start>, role, \n, text_tokens, \n, <im_end>]
```

### Phase 4: Latent Space Synchronization
**File**: `taste_slm.py:418-471` (`prepare_lm_input_target` method)

**Changes**:
1. Update latent target generation to match special token sequences
2. Ensure `lm_taste_latent_target` includes silence representations
3. Maintain mask consistency for valid taste positions
4. Preserve delay mechanism compatibility

**Latent Processing**:
```
taste_latent: [silence, silence, silence, original_latent, silence, silence]
taste_mask:   [False,  False,  False,  True,           False,  False]
```

## Technical Details

### Token ID Mapping
```python
SPECIAL_TOKENS = {
    'im_start': 151644,
    'im_end': 151645,
    'system': 6125,
    'user': 882, 
    'assistant': 78191,
    'newline': 198
}
```

### Sequence Length Calculations
- **Original length**: `original_text_token_len`
- **Special token overhead**: 5 tokens per message (`<im_start>`, role, `\n`, `\n`, `<im_end>`)
- **Final length**: `original_text_token_len + 5`
- **Taste embedding length**: Must match final text token length

### Memory Considerations
- **Silence tokens**: Additional parameters (~896 * 3 floats for im_start/im_end/role)
- **Sequence overhead**: +5 tokens per message increases memory usage
- **Batch processing**: Variable message counts require dynamic padding

## Testing Strategy

### Unit Tests
1. **Token insertion**: Verify correct special token placement
2. **Shape consistency**: Ensure all tensors have matching dimensions
3. **Silence generation**: Test trainable silence token creation
4. **Concatenation**: Verify proper message concatenation

### Integration Tests
1. **SFT data flow**: End-to-end conversation processing
2. **Training compatibility**: Ensure loss computation works correctly
3. **Inference**: Test conversation generation with special tokens
4. **Memory usage**: Monitor memory consumption with special tokens

### Validation Tests
1. **Conversation format**: Verify `<im_start>{role}\n{content}\n<im_end>` format
2. **Multimodal alignment**: Ensure text and taste tokens stay synchronized
3. **Gradient flow**: Test backpropagation through silence tokens
4. **Pretrain compatibility**: Ensure non-SFT training still works

## Risk Assessment

### High Risk
- **Shape mismatch**: Incorrect tensor dimensions could crash training
- **Memory explosion**: Special tokens increase sequence length significantly
- **Gradient issues**: Silence tokens might not learn effectively

### Medium Risk
- **Performance degradation**: Additional tokens increase computational cost
- **Compatibility issues**: Changes might break existing pretrain pipeline
- **Token ID conflicts**: Hardcoded IDs might conflict with tokenizer updates

### Low Risk
- **Code complexity**: Implementation adds complexity but is manageable
- **Debugging difficulty**: Special token processing adds debugging overhead

## Success Criteria

### Functional Requirements
- [ ] Special tokens correctly inserted for all roles
- [ ] Trainable silence tokens for non-audio tokens
- [ ] Proper batch processing with variable message lengths
- [ ] Latent space alignment with padded sequences
- [ ] Backward compatibility with pretrain training

### Performance Requirements
- [ ] Training loss decreases normally
- [ ] Memory usage remains within acceptable limits
- [ ] Training speed degradation < 20%
- [ ] Inference generates proper conversation format

### Quality Requirements
- [ ] Generated conversations follow `<im_start>{role}\n{content}\n<im_end>` format
- [ ] Multimodal alignment preserved (text-audio synchronization)
- [ ] Model learns appropriate silence representations
- [ ] No regression in base model capabilities

## Implementation Timeline

1. **Phase 1** (Special Token Processing): 2-3 hours
2. **Phase 2** (Silence Token Integration): 1-2 hours  
3. **Phase 3** (Batch Processing Enhancement): 1-2 hours
4. **Phase 4** (Latent Space Synchronization): 2-3 hours
5. **Testing & Validation**: 2-3 hours

**Total Estimated Time**: 8-13 hours

## Files to Modify

1. **Primary**: `taste_speech/taste2/taste_slm.py`
   - `TasteSLMFusing.__init__()` - Add silence token parameters
   - `TasteSLMFusing.forward()` - Handle silence token processing
   - `TasteSLM._prepare_for_sft_training()` - Main SFT processing logic
   - `TasteSLM.prepare_lm_input_target()` - Latent space alignment

2. **Secondary** (if needed):
   - `taste_speech/taste2/cosyvoice/dataset/processor.py` - SFT data pipeline
   - Configuration files - Token ID definitions

3. **Testing**:
   - Create test script for SFT functionality
   - Add validation for conversation format
   - Memory profiling script

This plan provides a comprehensive roadmap for implementing SFT capabilities in the TASTE Speech Language Model while maintaining compatibility with existing functionality.



## data format before shuffle
### pretrain
  {
    'utt': 'utterance_001',
    'text': 'Hello, how are you?',
    'text_token': trimmed_text_tokens,     # TRIMMED for alignment
    'speech': resampled_speech_tensor,
    'speech_feat': fbank_features,         # NEW (trimmed for alignment)
    'audio_feature': audio_feat_tensor,
    'audio_feature_len': feat_length,
    'sample_rate': 16000,
    'speech_token': trimmed_speech_tokens  # TRIMMED for alignment
  }

### sft
  {
    'utt': 'conv_001_turn_2',
    'text_list': [...], 
    'text_roles': [...], 
    'text_list_tokens': [...],
    'speech_list': [...], 
    'audio_feature_list': [...], 
                                       #need add audio_feature_length
    'speech_feat_list': [...],
    'audio_timestamps': [...],         # new
    'current_turn_index': 2,           # new
    'current_role': 'assistant',       # new
    'conversation_id': 'conv_001',     # new
    'sample_rate': 16000
  }




  {
      'utt': 'conv_001_turn_2',
      'text_list': [...],  # Original 
      'text_roles': [...], # Original
      'text_list_tokens': [...], # Original
      'speech_list': [...],
      'audio_feature_list': [...],
      'speech_feat_list': [...], # Original
      'audio_timestamps': [...],
      'current_turn_index': 2,
      'current_role': 'assistant',
      'conversation_id': 'conv_001',
      'sample_rate': 16000,

      # NEW FORMATTED FIELDS:
      'text': '<|im_start|>system\n...\n<|im_end|><|im_start|>user\n...\n<|im_end|>...',
      'text_token': [token_id_1, token_id_2, ...], # Single flat list
      'token_message_ids': [0, 0, -1, -1, 1, 1, -1, ...], # Message tracking
      'speech_feat': concatenated_speech_features_tensor  # Single tensor
  }





    data = [
      [  # Batch (list of SFT samples)
          {  # SFT Sample 1
              # Basic identifiers
              'utt': 'conversation_123_turn_2',           # str: unique sample ID
              'conversation_id': 'conversation_123',       # str: conversation identifier
              'current_turn_index': 2,                    # int: which turn in conversation
              'current_role': 'assistant',                # str: role of current turn

              # Multi-segment conversation context (FULL conversation history)
              'text_list': [                              # List[str]: all turns' text
                  "Hello, how are you?",                  # turn 0 text
                  "I'm doing well, thanks!",              # turn 1 text  
                  "What's the weather like?"              # turn 2 text (current)
              ],
              'text_roles': [                             # List[str]: roles for each turn
                  'user', 'assistant', 'user'
              ],
              'audio_data_list': [                        # List[bytes|None]: raw audio for each turn
                  None, b'audio_bytes_turn1', None        # only some turns have audio
              ],
              'speech_list': [                            # List[torch.Tensor|None]: processed speech
                  None,                                   # turn 0: no speech
                  torch.Tensor([1, 16000]),              # turn 1: [1, audio_samples] 
                  None                                    # turn 2: no speech
              ],
              'audio_feature_list': [                     # List[torch.Tensor|None]: audio features
                  None,                                   # turn 0: no features
                  torch.Tensor([200, 512]),              # turn 1: [time_frames, audio_feat_dim]
                  None                                    # turn 2: no features
              ],
              'audio_feature_len_list': [                 # List[int|None]: feature lengths
                  None, 200, None
              ],
              'speech_feat_list': [                       # List[torch.Tensor|None]: mel features  
                  None,                                   # turn 0: no mel features
                  torch.Tensor([150, 80]),               # turn 1: [time_frames, mel_dim]
                  None                                    # turn 2: no mel features
              ],

              # Concatenated/formatted data (after format_and_concatenate_conversation)
              'text': '<|im_start|>user\nHello, how are you?\n<|im_end|><|im_start|>assistant\nI\'m doing well, thanks!\n<|im_end|><|im_start|>user\nWhat\'s the weather like?\n<|im_end|>',
              'text_token': [1234, 5678, 9012, ...],     # List[int]: concatenated token IDs
              'token_message_ids': [-1, -1, 0, 0, -1, -1, 1, 1, -1, -1, 2, 2, -1],  # List[int]: which message each token belongs to (-1 for special tokens)
              'speech_feat': torch.Tensor([150, 80]),     # torch.Tensor: concatenated speech features [total_time_frames, mel_dim]

              # Required fields
              'speech_token': [456, 789, 123, ...],       # List[int]: speech tokens for current turn
              'sample_rate': 22050,                       # int: audio sample rate

              # Optional fields
              'meta': {...},                              # dict: metadata
              'speech_token': [456, 789, ...],           # List[int]: speech tokens
              'reject_speech_token': [111, 222, ...],    # List[int]: for DPO training
          },
          {  # SFT Sample 2 - another conversation turn
              # ... same structure but different conversation/turn
          }
      ]
  ]



  