# CosyVoice 配置说明

本文档详细介绍了 CosyVoice 的各种配置选项，包括模型参数、训练配置、数据处理管道以及推理优化等。

## 主配置文件

### cosyvoice.yaml 结构

CosyVoice 使用 YAML 格式的配置文件来定义模型架构、训练参数和数据处理管道。主配置文件通常位于 `examples/libritts/cosyvoice/conf/cosyvoice.yaml`。

#### 配置文件基本结构

```yaml
# 随机种子设置
__set_seed1: !apply:random.seed [1986]
__set_seed2: !apply:numpy.random.seed [1986]
__set_seed3: !apply:torch.manual_seed [1986]
__set_seed4: !apply:torch.cuda.manual_seed_all [1986]

# 固定参数
sample_rate: 22050
text_encoder_input_size: 512
llm_input_size: 1024
llm_output_size: 1024
spk_embed_dim: 192

# 模型组件
llm: !new:cosyvoice.llm.llm.TransformerLM
flow: !new:cosyvoice.flow.flow.MaskedDiffWithXvec
hift: !new:cosyvoice.hifigan.generator.HiFTGenerator
hifigan: !new:cosyvoice.hifigan.hifigan.HiFiGan

# 数据处理管道
data_pipeline: [...]
data_pipeline_gan: [...]

# 训练配置
train_conf: {...}
train_conf_gan: {...}
```

#### 关键配置参数说明

| 参数名 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `sample_rate` | int | 音频采样率 | 22050 |
| `text_encoder_input_size` | int | 文本编码器输入维度 | 512 |
| `llm_input_size` | int | LLM 输入维度 | 1024 |
| `llm_output_size` | int | LLM 输出维度 | 1024 |
| `spk_embed_dim` | int | 说话人嵌入维度 | 192 |
| `text_token_size` | int | 文本词汇表大小 | 51866 |
| `speech_token_size` | int | 语音 token 大小 | 4096 |

## 模型参数配置

### LLM 配置

```yaml
llm: !new:cosyvoice.llm.llm.TransformerLM
    # 基础参数
    text_encoder_input_size: !ref <text_encoder_input_size>
    llm_input_size: !ref <llm_input_size>
    llm_output_size: !ref <llm_output_size>
    
    # 词汇表大小
    text_token_size: 51866  # 对于 CosyVoice-300M-25Hz，改为 60515
    speech_token_size: 4096
    
    # 训练参数
    length_normalized_loss: True
    lsm_weight: 0  # 标签平滑权重
    spk_embed_dim: !ref <spk_embed_dim>
    
    # 文本编码器
    text_encoder: !new:cosyvoice.transformer.encoder.ConformerEncoder
        input_size: !ref <text_encoder_input_size>
        output_size: 1024
        attention_heads: 16
        linear_units: 4096
        num_blocks: 6
        dropout_rate: 0.1
        positional_dropout_rate: 0.1
        attention_dropout_rate: 0.0
        normalize_before: True
        input_layer: 'linear'
        pos_enc_layer_type: 'rel_pos_espnet'
        selfattention_layer_type: 'rel_selfattn'
        use_cnn_module: False
        macaron_style: False
        use_dynamic_chunk: False
        use_dynamic_left_chunk: False
        static_chunk_size: 1
    
    # LLM 主体
    llm: !new:cosyvoice.transformer.encoder.TransformerEncoder
        input_size: !ref <llm_input_size>
        output_size: !ref <llm_output_size>
        attention_heads: 16
        linear_units: 4096
        num_blocks: 14  # Transformer 层数
        dropout_rate: 0.1
        positional_dropout_rate: 0.1
        attention_dropout_rate: 0.0
        input_layer: 'linear_legacy'
        pos_enc_layer_type: 'rel_pos_espnet'
        selfattention_layer_type: 'rel_selfattn'
        static_chunk_size: 1
    
    # 采样策略
    sampling: !name:cosyvoice.utils.common.ras_sampling
        top_p: 0.8    # Top-p 采样参数
        top_k: 25     # Top-k 采样参数
        win_size: 10  # RAS 窗口大小
        tau_r: 0.1    # RAS 温度参数
```

#### LLM 配置参数详解

- **文本编码器参数**：
  - `attention_heads`: 多头注意力头数，影响模型表达能力
  - `linear_units`: 前馈网络维度，影响模型容量
  - `num_blocks`: 编码器层数，影响模型深度
  - `dropout_rate`: 丢弃率，用于正则化

- **采样策略参数**：
  - `top_p`: 核采样参数，控制生成的随机性
  - `top_k`: 选择概率最高的 k 个token
  - `win_size`: RAS（Repetition Aware Sampling）窗口大小
  - `tau_r`: RAS 温度参数，控制重复抑制强度

### Flow 配置

```yaml
flow: !new:cosyvoice.flow.flow.MaskedDiffWithXvec
    # 基础维度
    input_size: 512
    output_size: 80  # Mel 频谱维度
    spk_embed_dim: !ref <spk_embed_dim>
    output_type: 'mel'
    vocab_size: 4096
    
    # 帧率配置
    input_frame_rate: 50  # 对于 CosyVoice-300M-25Hz，改为 25
    only_mask_loss: True
    
    # Flow 编码器
    encoder: !new:cosyvoice.transformer.encoder.ConformerEncoder
        output_size: 512
        attention_heads: 8
        linear_units: 2048
        num_blocks: 6
        dropout_rate: 0.1
        positional_dropout_rate: 0.1
        attention_dropout_rate: 0.1
        normalize_before: True
        input_layer: 'linear'
        pos_enc_layer_type: 'rel_pos_espnet'
        selfattention_layer_type: 'rel_selfattn'
        input_size: 512
        use_cnn_module: False
        macaron_style: False
    
    # 长度调节器
    length_regulator: !new:cosyvoice.flow.length_regulator.InterpolateRegulator
        channels: 80
        sampling_ratios: [1, 1, 1, 1]  # 上采样比例
    
    # Flow 解码器
    decoder: !new:cosyvoice.flow.flow_matching.ConditionalCFM
        in_channels: 240
        n_spks: 1
        spk_emb_dim: 80
        
        # CFM 参数
        cfm_params: !new:omegaconf.DictConfig
            content:
                sigma_min: 1e-06        # 最小噪声标准差
                solver: 'euler'         # ODE 求解器
                t_scheduler: 'cosine'   # 时间调度器
                training_cfg_rate: 0.2  # 训练时 CFG 比例
                inference_cfg_rate: 0.7 # 推理时 CFG 比例
                reg_loss_type: 'l1'     # 正则化损失类型
        
        # 估计器网络
        estimator: !new:cosyvoice.flow.decoder.ConditionalDecoder
            in_channels: 320
            out_channels: 80
            channels: [256, 256]
            dropout: 0.0
            attention_head_dim: 64
            n_blocks: 4
            num_mid_blocks: 12
            num_heads: 8
            act_fn: 'gelu'
```

#### Flow 配置参数详解

- **帧率相关**：
  - `input_frame_rate`: 输入特征帧率，25Hz 或 50Hz
  - `sampling_ratios`: 上采样比例，影响时间分辨率

- **CFM 参数**：
  - `sigma_min`: 扩散过程的最小噪声水平
  - `solver`: ODE 求解器类型（euler, heun, etc.）
  - `training_cfg_rate`: 训练时无条件生成的比例
  - `inference_cfg_rate`: 推理时 Classifier-Free Guidance 强度

### HiFi-GAN 配置

```yaml
# HiFT (HiFi-GAN + F0) 生成器
hift: !new:cosyvoice.hifigan.generator.HiFTGenerator
    in_channels: 80
    base_channels: 512
    nb_harmonics: 8        # 谐波数量
    sampling_rate: !ref <sample_rate>
    
    # NSF (Neural Source Filter) 参数
    nsf_alpha: 0.1         # NSF alpha 参数
    nsf_sigma: 0.003       # NSF sigma 参数
    nsf_voiced_threshold: 10  # 有声/无声阈值
    
    # 上采样配置
    upsample_rates: [8, 8]           # 上采样倍数
    upsample_kernel_sizes: [16, 16]  # 上采样卷积核大小
    
    # ISTFT 参数
    istft_params:
        n_fft: 16     # FFT 长度
        hop_len: 4    # 跳跃长度
    
    # 残差块配置
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    source_resblock_kernel_sizes: [7, 11]
    source_resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5]]
    lrelu_slope: 0.1
    audio_limit: 0.99  # 音频幅值限制
    
    # F0 预测器
    f0_predictor: !new:cosyvoice.hifigan.f0_predictor.ConvRNNF0Predictor
        num_class: 1
        in_channels: 80
        cond_channels: 512

# Mel 谱变换
mel_spec_transform1: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1024
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 256
    win_size: 1024
    fmin: 0
    fmax: null  # 自动设置为采样率的一半
    center: False

# 完整 HiFiGAN 模型
hifigan: !new:cosyvoice.hifigan.hifigan.HiFiGan
    generator: !ref <hift>
    
    # 判别器
    discriminator: !new:cosyvoice.hifigan.discriminator.MultipleDiscriminator
        mpd: !new:matcha.hifigan.models.MultiPeriodDiscriminator
        mrd: !new:cosyvoice.hifigan.discriminator.MultiResSpecDiscriminator
    
    mel_spec_transform: [!ref <mel_spec_transform1>]
```

#### HiFi-GAN 配置参数详解

- **生成器参数**：
  - `nb_harmonics`: 谐波数量，影响音频质量
  - `nsf_*`: Neural Source Filter 参数，控制激励信号生成
  - `upsample_rates`: 上采样倍数，需满足：∏(rates) = hop_size
  - `audio_limit`: 防止音频削峰的幅值限制

- **频谱参数**：
  - `n_fft`: FFT 窗口大小
  - `num_mels`: Mel 频谱维度
  - `hop_size`: 帧移大小
  - `win_size`: 窗口大小

## 训练配置

### 基本训练配置

```yaml
# LLM + Flow 训练配置
train_conf:
    # 优化器设置
    optim: adam
    optim_conf:
        lr: 0.001          # 预训练学习率，SFT 时改为 1e-5
        betas: [0.9, 0.98]  # Adam beta 参数
        eps: 1e-9          # Adam epsilon
        weight_decay: 0    # 权重衰减
    
    # 学习率调度
    scheduler: warmuplr    # SFT 时改为 constantlr
    scheduler_conf:
        warmup_steps: 2500  # 预热步数
        warmup_rate: 0.0   # 预热起始学习率比例
    
    # 训练参数
    max_epoch: 200      # 最大训练轮数
    grad_clip: 5        # 梯度裁剪
    accum_grad: 2       # 梯度累积步数
    log_interval: 100   # 日志间隔
    save_per_step: -1   # 保存间隔（-1 表示按 epoch 保存）
    
    # 损失函数权重
    loss_weights:
        llm_loss: 1.0
        flow_loss: 1.0
```

#### 训练配置参数详解

- **学习率设置**：
  - 预训练: `lr: 0.001` + `warmuplr`
  - SFT 微调: `lr: 1e-5` + `constantlr`
  - 使用较小的学习率进行 SFT 以避免过拟合

- **梯度优化**：
  - `grad_clip`: 防止梯度爆炸
  - `accum_grad`: 模拟更大的 batch size

### GAN 训练配置

```yaml
# HiFiGAN 训练配置
train_conf_gan:
    # 生成器优化器
    optim: adam
    optim_conf:
        lr: 0.0002        # GAN 训练使用较小学习率
        betas: [0.8, 0.99]
        eps: 1e-9
    scheduler: constantlr
    
    # 判别器优化器
    optim_d: adam
    optim_conf_d:
        lr: 0.0002
        betas: [0.8, 0.99]
        eps: 1e-9
    scheduler_d: constantlr
    
    # GAN 训练参数
    max_epoch: 200
    grad_clip: 5
    accum_grad: 1      # GAN 训练必须为 1
    log_interval: 100
    save_per_step: -1
    
    # GAN 损失权重
    lambda_adv: 1.0    # 对抗损失权重
    lambda_feat: 10.0  # 特征匹配损失权重
    lambda_mel: 45.0   # Mel 谱损失权重
```

#### GAN 训练特殊要求

- **学习率平衡**：生成器和判别器使用相同的小学习率
- **梯度累积**：必须设置为 1，避免影响 GAN 训练稳定性
- **损失平衡**：通过权重控制不同损失的重要性

## 数据处理配置

### 数据处理管道配置

```yaml
# 处理器组件定义
parquet_opener: !name:cosyvoice.dataset.processor.parquet_opener

# 文本分词器
get_tokenizer: !name:whisper.tokenizer.get_tokenizer  # 或使用自定义分词器
    multilingual: True
    num_languages: 100
    language: 'en'
    task: 'transcribe'
allowed_special: 'all'

tokenize: !name:cosyvoice.dataset.processor.tokenize
    get_tokenizer: !ref <get_tokenizer>
    allowed_special: !ref <allowed_special>

# 数据过滤
filter: !name:cosyvoice.dataset.processor.filter
    max_length: 40960     # 最大音频长度（样本数）
    min_length: 0         # 最小音频长度
    token_max_length: 200 # 最大文本长度（token 数）
    token_min_length: 1   # 最小文本长度

# 音频重采样
resample: !name:cosyvoice.dataset.processor.resample
    resample_rate: !ref <sample_rate>

# 音频截断
truncate: !name:cosyvoice.dataset.processor.truncate
    truncate_length: 24576  # 必须是 hop_size 的倍数

# 特征提取
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1024
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 256
    win_size: 1024
    fmin: 0
    fmax: 8000  # 高频截止
    center: False

compute_fbank: !name:cosyvoice.dataset.processor.compute_fbank
    feat_extractor: !ref <feat_extractor>

# F0 计算
compute_f0: !name:cosyvoice.dataset.processor.compute_f0
    sample_rate: !ref <sample_rate>
    hop_size: 256

# 说话人嵌入
parse_embedding: !name:cosyvoice.dataset.processor.parse_embedding
    normalize: True  # 是否标准化嵌入

# 数据混洗和排序
shuffle: !name:cosyvoice.dataset.processor.shuffle
    shuffle_size: 1000  # 混洗缓冲区大小

sort: !name:cosyvoice.dataset.processor.sort
    sort_size: 500  # 排序缓冲区大小（应小于 shuffle_size）

# 批处理
batch: !name:cosyvoice.dataset.processor.batch
    batch_type: 'dynamic'        # 动态批处理
    max_frames_in_batch: 2000    # V100 16G 上 GAN 训练时改为 1400

# 填充
padding: !name:cosyvoice.dataset.processor.padding
    use_spk_embedding: False  # SFT 时改为 True
```

### 数据管道组装

```yaml
# LLM + Flow 训练管道
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

# GAN 训练管道（额外包含 F0 计算和音频截断）
data_pipeline_gan: [
    !ref <parquet_opener>,
    !ref <tokenize>,
    !ref <filter>,
    !ref <resample>,
    !ref <truncate>,      # GAN 训练专用
    !ref <compute_fbank>,
    !ref <compute_f0>,    # GAN 训练专用
    !ref <parse_embedding>,
    !ref <shuffle>,
    !ref <sort>,
    !ref <batch>,
    !ref <padding>,
]
```

#### 数据处理参数调优指南

- **内存优化**：
  - `shuffle_size`: 根据内存大小调整
  - `sort_size`: 通常设为 `shuffle_size` 的一半
  - `max_frames_in_batch`: GPU 内存不足时减小

- **质量控制**：
  - `max_length`/`min_length`: 过滤异常长度音频
  - `token_max_length`: 控制文本长度，避免内存溢出
  - `fmax`: 控制频谱高频成分

## 推理配置

### 性能调优参数

```yaml
# 推理优化配置
inference_conf:
    # 模型加载配置
    load_jit: True     # 启用 JIT 编译加速
    load_trt: True     # 启用 TensorRT 加速
    load_vllm: True    # 启用 vLLM 加速
    fp16: True         # 使用半精度推理
    
    # 并发配置
    trt_concurrent: 4  # TensorRT 并发数
    max_batch_size: 8  # 最大批处理大小
    
    # 流式推理配置
    stream_chunk_size: 1024  # 流式输出块大小
    stream_buffer_size: 4096 # 流式缓冲区大小
    
    # 缓存配置
    enable_kv_cache: True    # 启用 KV 缓存
    cache_size: 1000         # 缓存大小
    
    # 内存优化
    gradient_checkpointing: True  # 梯度检查点
    attention_implementation: "sdpa"  # 使用优化注意力实现
```

### 质量控制参数

```yaml
# 生成质量配置
generation_conf:
    # 采样参数
    top_p: 0.8         # 核采样概率
    top_k: 25          # Top-K 采样
    temperature: 1.0    # 采样温度
    
    # RAS (Repetition Aware Sampling) 参数
    ras_window_size: 10
    ras_tau: 0.1
    
    # CFG (Classifier-Free Guidance) 参数
    cfg_scale: 0.7     # CFG 强度
    cfg_rescale: 0.0   # CFG 重缩放
    
    # Flow 推理参数
    flow_steps: 22     # Flow 推理步数
    flow_solver: 'euler'  # ODE 求解器
    
    # 后处理参数
    normalize_audio: True     # 音频标准化
    trim_silence: True        # 去除静音
    silence_threshold: -40    # 静音阈值（dB）
```

## 环境变量

CosyVoice 支持通过环境变量进行配置：

```bash
# 模型和数据路径
export COSYVOICE_MODEL_DIR="/path/to/models"
export COSYVOICE_DATA_DIR="/path/to/data"
export COSYVOICE_CACHE_DIR="/path/to/cache"

# 设备配置
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export COSYVOICE_DEVICE="cuda"
export COSYVOICE_NUM_WORKERS=4

# 推理优化
export COSYVOICE_ENABLE_JIT=true
export COSYVOICE_ENABLE_TRT=true
export COSYVOICE_FP16=true

# 日志配置
export COSYVOICE_LOG_LEVEL="INFO"
export COSYVOICE_LOG_FILE="/path/to/log"

# ModelScope 配置
export MODELSCOPE_CACHE="/path/to/modelscope/cache"
export MODELSCOPE_MODULES_CACHE="/path/to/modules/cache"

# Python 路径
export PYTHONPATH="${PYTHONPATH}:/path/to/CosyVoice:/path/to/CosyVoice/third_party/Matcha-TTS"
```

## 配置模板

### 开发环境配置

```yaml
# development.yaml - 开发环境推荐配置
sample_rate: 22050

# 较小的模型规模用于快速迭代
llm:
    text_encoder:
        num_blocks: 4      # 减少层数
        linear_units: 2048 # 减少维度
    llm:
        num_blocks: 8      # 减少层数
        linear_units: 2048

flow:
    encoder:
        num_blocks: 4      # 减少层数
    decoder:
        n_blocks: 2        # 减少块数
        num_mid_blocks: 6  # 减少中间块数

# 训练配置
train_conf:
    max_epoch: 50          # 较少的训练轮数
    log_interval: 50       # 更频繁的日志
    accum_grad: 1          # 减少梯度累积

# 数据处理
batch:
    max_frames_in_batch: 1000  # 较小的批处理大小

shuffle:
    shuffle_size: 500      # 较小的混洗缓冲区

sort:
    sort_size: 200         # 较小的排序缓冲区
```

### 生产环境配置

```yaml
# production.yaml - 生产环境推荐配置
sample_rate: 22050

# 完整模型规模
llm:
    text_encoder:
        num_blocks: 6
        linear_units: 4096
    llm:
        num_blocks: 14
        linear_units: 4096
    sampling:
        top_p: 0.8
        top_k: 25
        tau_r: 0.1

flow:
    encoder:
        num_blocks: 6
    decoder:
        n_blocks: 4
        num_mid_blocks: 12
    cfm_params:
        inference_cfg_rate: 0.7  # 较强的引导

hift:
    nb_harmonics: 8
    audio_limit: 0.99

# 推理优化配置
inference_conf:
    load_jit: True
    load_trt: True
    fp16: True
    trt_concurrent: 4
    enable_kv_cache: True
    gradient_checkpointing: True

# 数据处理优化
batch:
    batch_type: 'dynamic'
    max_frames_in_batch: 2000

shuffle:
    shuffle_size: 2000

sort:
    sort_size: 1000

# 质量控制
generation_conf:
    flow_steps: 22         # 充足的推理步数
    cfg_scale: 0.7         # 适中的引导强度
    normalize_audio: True
    trim_silence: True
```

### 高性能配置

```yaml
# high_performance.yaml - 高性能推理配置
# 适用于对速度要求极高的场景

# 推理加速
inference_conf:
    load_jit: True
    load_trt: True
    load_vllm: True        # 启用 vLLM
    fp16: True
    trt_concurrent: 8      # 更高并发
    max_batch_size: 16     # 更大批处理

# 激进的采样参数
generation_conf:
    top_p: 0.9             # 稍微增加随机性
    top_k: 20              # 减少候选数量
    temperature: 0.9       # 稍微降低温度
    flow_steps: 16         # 减少推理步数
    cfg_scale: 0.5         # 较弱的引导

# 内存优化
inference_conf:
    gradient_checkpointing: True
    attention_implementation: "flash_attention_2"
    enable_xformers: True
```

### 高质量配置

```yaml
# high_quality.yaml - 高质量推理配置
# 适用于对音频质量要求极高的场景

# 保守的采样参数
generation_conf:
    top_p: 0.7             # 较低的核采样
    top_k: 30              # 更多候选
    temperature: 0.8       # 较低温度
    flow_steps: 32         # 更多推理步数
    cfg_scale: 0.9         # 较强引导
    cfg_rescale: 0.1       # 启用重缩放

# RAS 参数调优
ras_window_size: 15        # 更大窗口
ras_tau: 0.05              # 更强重复抑制

# 后处理优化
post_processing:
    normalize_audio: True
    trim_silence: True
    silence_threshold: -45  # 更严格的静音检测
    fade_in: 0.01          # 淡入
    fade_out: 0.01         # 淡出
    peak_normalize: True    # 峰值标准化
```

## 配置文件使用示例

```python
# 加载自定义配置
from cosyvoice.cli.cosyvoice import CosyVoice2
import yaml

# 方法1: 使用配置文件
with open('configs/production.yaml', 'r') as f:
    config = yaml.safe_load(f)

cosyvoice = CosyVoice2(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    config=config
)

# 方法2: 直接指定参数
cosyvoice = CosyVoice2(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    load_jit=True,
    load_trt=True,
    fp16=True,
    trt_concurrent=4
)

# 方法3: 运行时修改配置
cosyvoice.update_config({
    'generation_conf': {
        'top_p': 0.9,
        'flow_steps': 20
    }
})
```

通过合理配置这些参数，您可以在不同场景下获得最佳的性能和质量平衡。建议根据具体的硬件环境和应用需求选择合适的配置模板，并根据实际测试结果进行微调。