# Wandb Integration for CosyVoice Training

本文檔說明如何在 CosyVoice 訓練中使用 Weights & Biases (wandb) 進行實驗追蹤和視覺化。

## 安裝 wandb

```bash
pip install wandb
```

## 配置 wandb

首次使用需要登入 wandb：

```bash
wandb login
```

## 記錄器邏輯

### 🎯 **核心設計原則**
- **TensorBoard 總是啟用**: 不論是否使用 wandb，TensorBoard 都會正常工作
- **wandb 是可選加強**: 當啟用 `--use_wandb` 時，會同時記錄到 TensorBoard 和 wandb
- **完全向後相容**: 現有腳本無需任何修改

### 📊 **行為模式**

| 情況 | TensorBoard | wandb | 說明 |
|------|-------------|-------|------|
| 預設 (無 `--use_wandb`) | ✅ 啟用 | ❌ 停用 | 原始行為，完全向後相容 |
| `--use_wandb` + wandb 已安裝 | ✅ 啟用 | ✅ 啟用 | 雙重記錄，最佳體驗 |
| `--use_wandb` + wandb 未安裝 | ✅ 啟用 | ❌ 停用 | 優雅降級到 TensorBoard |

## 使用方法

### 配置文件設定（唯一方式）

wandb 參數已經整合到配置文件中，只需在 `training/conf/taste2_stage1.yaml` 中設定：

```yaml
train_conf:
    # ... other training configs ...
    
    # wandb logging configuration
    use_wandb: true  # set to true to enable wandb logging
    wandb_project: "cosyvoice-taste2"
    wandb_entity: "my-team"  # set to your wandb entity/team name
    wandb_run_name: null  # set to custom run name or leave null for auto-generation
    wandb_tags: "stage1,llm"  # comma-separated tags for organizing experiments
```

然後使用標準訓練命令：

```bash
python CosyVoice/cosyvoice/bin/train.py \
  --model llm \
  --config training/conf/taste2_stage1.yaml \
  --train_data training/data/train.data.list \
  --cv_data training/data/dev.data.list \
  --model_dir training/exp/
```

### 設計理念

- **簡潔性**: 所有 wandb 設定統一在配置文件中管理
- **團隊一致性**: 確保團隊成員使用相同的實驗追蹤設定
- **版本控制友好**: wandb 設定與其他訓練參數一起進行版本控制

### 配置參數說明

在 `train_conf` 區塊中可以設定以下 wandb 參數：

- `use_wandb`: 啟用 wandb 記錄（true/false）
- `wandb_project`: wandb 專案名稱（預設：cosyvoice）
- `wandb_run_name`: 執行名稱（null 表示自動生成）
- `wandb_entity`: wandb 實體/團隊名稱（null 表示個人帳戶）
- `wandb_tags`: 標籤，用逗號分隔（例如：stage1,llm,experiment）

### 完整配置範例

```yaml
train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5
    scheduler: constantlr
    max_epoch: 200
    grad_clip: 5
    accum_grad: 2
    log_interval: 100
    save_per_step: 1000
    
    # wandb logging configuration
    use_wandb: true
    wandb_project: "cosyvoice-taste2"
    wandb_entity: "ntu-taslm"
    wandb_run_name: null  # auto-generated
    wandb_tags: "stage1,llm,baseline"
```

## 記錄的指標

系統會自動記錄以下指標到 wandb：

### 訓練指標
- `train_epoch`: 當前 epoch
- `train_lr`: 學習率
- `train_grad_norm`: 梯度範數
- `train_loss`: 訓練損失
- 其他模型特定的損失項

### 驗證指標
- `cv_epoch`: 驗證 epoch
- `cv_lr`: 學習率
- `cv_loss`: 驗證損失
- 其他驗證損失項

### 配置參數
- 模型類型
- 訓練引擎（torch_ddp/deepspeed）
- 混合精度設定
- DPO 設定
- 所有訓練配置參數

## 技術實現特點

### 🔧 **架構設計**
- **UnifiedLogger 類**: 統一處理 TensorBoard 和 wandb 記錄
- **自動配置記錄**: 訓練參數在 wandb 初始化時自動同步
- **多進程安全**: 只有 rank 0 進程記錄，避免重複寫入
- **優雅錯誤處理**: wandb 相關錯誤不會影響訓練進程

### 📁 **修改的文件**
- `CosyVoice/cosyvoice/bin/train.py`: 新增 wandb 命令行參數
- `CosyVoice/cosyvoice/utils/train_utils.py`: 新增 UnifiedLogger 類和相關邏輯

## 向後相容性

- **100% 向後相容**: 所有現有腳本和工作流程保持不變
- **TensorBoard 保證**: TensorBoard 功能永遠可用，不受 wandb 設定影響
- **優雅降級**: wandb 相關錯誤不會影響訓練進程
- **無需修改配置文件**: 現有的 YAML 配置文件完全不需要改動

## 故障排除

### wandb 未安裝
如果收到 wandb 相關錯誤，請安裝：
```bash
pip install wandb
```

### 網路連線問題
如果無法連接到 wandb 服務，可以使用離線模式：
```bash
export WANDB_MODE=offline
```

### 多 GPU 訓練
系統自動處理多 GPU 環境，只有 rank 0 的進程會記錄到 wandb，避免重複記錄。

## 最佳實踐

1. **專案組織**: 為不同的實驗階段創建不同的 wandb 專案
2. **標籤使用**: 使用有意義的標籤來組織實驗
3. **運行命名**: 使用描述性的運行名稱，包含時間戳
4. **配置追蹤**: 系統會自動記錄所有重要的配置參數

## 範例輸出

在 wandb dashboard 中，您將看到：
- 實時的訓練和驗證損失曲線
- 學習率調度圖
- 梯度範數變化
- 完整的超參數記錄
- 代碼版本追蹤

## 與現有工作流程整合

這個整合設計為無縫添加到現有的訓練流程中：
- 不需要修改現有的配置文件
- 保持與 TensorBoard 的完全相容性
- 支援所有現有的訓練功能（DPO、混合精度等）

## 實際使用案例

### Case 1: 原有工作流程（無變化）
```bash
# 原有的訓練命令完全不變
python CosyVoice/cosyvoice/bin/train.py \
  --model llm \
  --config training/conf/taste2_stage1.yaml \
  --train_data training/data/train.data.list \
  --cv_data training/data/dev.data.list \
  --model_dir training/exp/
# 結果: 只有 TensorBoard 記錄，行為與之前完全相同
```

### Case 2: 配置文件啟用 wandb
```yaml
# 在 training/conf/taste2_stage1.yaml 中設定:
train_conf:
    use_wandb: true
    wandb_project: "taste2-llm-training"
    wandb_entity: "ntu-taslm"
    wandb_tags: "stage1,llm"
```

```bash
# 使用標準訓練命令
python CosyVoice/cosyvoice/bin/train.py \
  --model llm \
  --config training/conf/taste2_stage1.yaml \
  --train_data training/data/train.data.list \
  --cv_data training/data/dev.data.list \
  --model_dir training/exp/
# 結果: TensorBoard + wandb 同時記錄
```

### Case 3: 動態修改配置進行實驗
```bash
# 複製配置文件進行實驗
cp training/conf/taste2_stage1.yaml training/conf/taste2_debug.yaml

# 修改 taste2_debug.yaml 中的 wandb 設定
# use_wandb: true
# wandb_project: "debug-experiments"
# wandb_run_name: "debug-run-$(date +%m%d_%H%M)"

python CosyVoice/cosyvoice/bin/train.py \
  --model llm \
  --config training/conf/taste2_debug.yaml \
  --train_data training/data/train.data.list \
  --cv_data training/data/dev.data.list \
  --model_dir training/exp/
# 結果: 使用不同的 wandb 設定進行實驗
```

## 常見問題 (FAQ)

### Q: 如果我不想用 wandb，會影響原有功能嗎？
A: 完全不會。不添加 `--use_wandb` 參數時，行為與原始版本完全相同。

### Q: TensorBoard 還會正常工作嗎？
A: 是的，TensorBoard 總是啟用，不受 wandb 設定影響。

### Q: 如果 wandb 初始化失敗會怎樣？
A: 系統會自動回退到僅使用 TensorBoard，訓練不會中斷。

### Q: 多 GPU 訓練時會重複記錄嗎？
A: 不會，系統自動確保只有 rank 0 進程記錄到 wandb。

### Q: 可以同時查看 TensorBoard 和 wandb 嗎？
A: 可以，兩者記錄相同的指標，可以同時使用不同的視覺化工具。