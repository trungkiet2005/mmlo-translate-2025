# Architecture and Implementation Details

## 🧠 High-Level Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MMLoSo NMT Pipeline                         │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼───────┐
│ Tokenizer      │                    │ Data Loading   │
│ Extension      │                    │ & Preparation  │
│                │                    │                │
│ - Extract      │                    │ - Load CSV     │
│   tokens       │                    │ - Clean data   │
│ - Filter       │                    │ - Split        │
│ - Add tokens   │                    │ - Tokenize     │
└───────┬────────┘                    └────────┬───────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                ┌───────────▼───────────┐
                │  Model Initialization │
                │                       │
                │ - Load NLLB model     │
                │ - Extend embeddings   │
                │ - Apply LoRA          │
                │ - Freeze base model   │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │  LoRA Fine-tuning     │
                │                       │
                │ - Train adapters      │
                │ - Train new embeddings│
                │ - Evaluate            │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │  Evaluation           │
                │                       │
                │ - BLEU score          │
                │ - chrF score          │
                │ - Final score         │
                └───────────────────────┘
```

## 🧩 Component Details

### 1. Tokenizer Extension

**Purpose**: Add new vocabulary tokens for low-resource languages without destroying existing vocabulary.

**Process**:
```
Base Tokenizer (256k tokens)
    ↓
Load Training Data
    ↓
Extract Token Candidates
    ├── By Script (Devanagari, Ol Chiki, Roman)
    ├── By Frequency
    └── By Word Boundaries
    ↓
Filter Existing Tokens
    ├── Check against existing vocab
    └── Remove already known tokens
    ↓
Select Top N Tokens
    ├── Prioritize rare scripts (Ol Chiki)
    ├── Prioritize high frequency
    └── Balance across scripts
    ↓
Add Tokens (add_tokens())
    ├── Preserves existing vocabulary
    └── Returns new token IDs
    ↓
Extended Tokenizer (256k + N tokens)
```

**Key Features**:
- Preserves existing vocabulary (no retraining)
- Script-aware extraction
- Frequency-based filtering
- Priority for rare scripts

### 2. Embedding Initialization

**Purpose**: Initialize embeddings for new tokens with sensible values.

**Strategies**:
1. **Average** (Recommended): Initialize with average of existing embeddings
   - Pros: Stable, preserves semantic relationships
   - Cons: May be too generic
   
2. **Random**: Initialize with small random values
   - Pros: Allows exploration
   - Cons: May hurt initial performance
   
3. **Zero**: Initialize with zeros
   - Pros: Simple
   - Cons: Not recommended (poor performance)

**Implementation**:
```python
# Get existing embeddings
existing_embeddings = model.get_input_embeddings().weight[:old_vocab_size]

# Compute average
avg_embedding = existing_embeddings.mean(dim=0)

# Initialize new tokens
for token_id in new_token_ids:
    model.get_input_embeddings().weight[token_id] = avg_embedding.clone()
```

### 3. LoRA Architecture

**Purpose**: Fine-tune model with minimal parameters to avoid catastrophic forgetting.

**Architecture**:
```
Base Model (Frozen)
    ├── Embeddings
    │   ├── Existing tokens (Frozen)
    │   └── New tokens (Trainable)
    ├── Encoder (Frozen)
    │   ├── Self-Attention
    │   │   ├── q_proj (LoRA - Trainable)
    │   │   ├── k_proj (LoRA - Trainable)
    │   │   ├── v_proj (LoRA - Trainable)
    │   │   └── o_proj (LoRA - Trainable)
    │   └── Feed-Forward
    │       ├── gate_proj (LoRA - Trainable)
    │       ├── up_proj (LoRA - Trainable)
    │       └── down_proj (LoRA - Trainable)
    └── Decoder (Frozen)
        ├── Self-Attention (LoRA - Trainable)
        ├── Cross-Attention (LoRA - Trainable)
        └── Feed-Forward (LoRA - Trainable)
```

**LoRA Mechanism**:
```
Original: W * x
LoRA:     W * x + (B * A) * x
          │     │   │   │
          │     │   │   └─ LoRA adapter (r × d)
          │     │   └───── LoRA adapter (d × r)
          │     └───────── LoRA scaling (alpha/r)
          └─────────────── Base weight (frozen)
```

**Parameters**:
- `r`: Rank (typically 8, 16, or 32)
- `alpha`: Scaling factor (typically 2× r)
- `dropout`: Dropout rate (typically 0.1)

**Trainable Parameters**:
- LoRA adapters: ~0.1% of total parameters
- New token embeddings: ~0.01% of total parameters
- Total trainable: ~0.11% of total parameters

### 4. Training Process

**Data Flow**:
```
Input (Source Language)
    ↓
Tokenizer (with language code)
    ├── Add language tokens
    └── Tokenize text
    ↓
Encoder (Frozen + LoRA)
    ├── Self-attention (LoRA)
    └── Feed-forward (LoRA)
    ↓
Decoder (Frozen + LoRA)
    ├── Self-attention (LoRA)
    ├── Cross-attention (LoRA)
    └── Feed-forward (LoRA)
    ↓
Output (Target Language)
    ↓
Loss (CrossEntropy)
    ↓
Backpropagation
    ├── Update LoRA adapters
    └── Update new token embeddings
```

**Training Configuration**:
- Learning rate: 5e-5 (low to prevent overfitting)
- Batch size: 8 (adjustable based on GPU memory)
- Gradient accumulation: 4 (effective batch size: 32)
- Epochs: 10 (adjustable based on dataset size)
- Warmup steps: 500
- Weight decay: 0.01

### 5. Evaluation

**Metrics**:
1. **BLEU Score**: N-gram overlap between prediction and reference
   - Range: 0-100
   - Higher is better
   - Good for: General translation quality

2. **chrF Score**: Character-level F-score
   - Range: 0-100
   - Higher is better
   - Good for: Morphologically rich languages

**Final Score (MMLoSo Competition)**:
```
Final = 0.6 * BLEU_component + 0.4 * chrF_component

where:
BLEU_component = 0.6 * BLEU_forward + 0.4 * BLEU_reverse
chrF_component = 0.6 * chrF_forward + 0.4 * chrF_reverse

forward directions:
- Hindi → Bhili
- Hindi → Mundari
- Hindi → Gondi
- English → Santali

reverse directions:
- Bhili → Hindi
- Mundari → Hindi
- Gondi → Hindi
- Santali → English
```

## 🔬 Advanced Techniques

### 1. Avoiding Catastrophic Forgetting

**Strategy**: Freeze base model, train only LoRA adapters and new embeddings.

**Implementation**:
- Base model parameters: `requires_grad = False`
- LoRA adapters: `requires_grad = True`
- New token embeddings: `requires_grad = True`

**Verification**:
```python
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params/total_params*100:.2f}%")
```

### 2. Handling Multiple Scripts

**Scripts**:
- **Devanagari** (Hindi, Bhili, Mundari, Gondi): U+0900 to U+097F
- **Ol Chiki** (Santali): U+1C50 to U+1C7F
- **Roman** (English): Standard ASCII

**Challenges**:
- Different tokenization requirements
- Script-specific normalization
- Handling script variants

**Solutions**:
- Script-aware token extraction
- Priority for rare scripts
- Script-specific preprocessing

### 3. Low-Resource Language Handling

**Challenges**:
- Limited training data
- Rare tokens
- Morphological complexity

**Solutions**:
- Tokenizer extension (add rare tokens)
- LoRA (prevent overfitting)
- Data augmentation (back-translation)
- Multi-task learning (shared representations)

## ⚠️ Common Pitfalls and Solutions

### 1. Tokenizer Extension

**Pitfall**: Retraining SentencePiece from scratch destroys existing vocabulary.

**Solution**: Use `add_tokens()` method which preserves existing tokens.

### 2. Embedding Initialization

**Pitfall**: Random initialization hurts performance.

**Solution**: Initialize with average of existing embeddings.

### 3. Catastrophic Forgetting

**Pitfall**: Full fine-tuning destroys original language capabilities.

**Solution**: Use LoRA instead of full fine-tuning.

### 4. Overfitting

**Pitfall**: Model overfits to small dataset.

**Solution**: 
- Use LoRA dropout (0.1)
- Use low learning rate (5e-5)
- Early stopping
- Data augmentation

### 5. Memory Issues

**Pitfall**: Out of memory during training.

**Solution**:
- Reduce batch size
- Use gradient accumulation
- Use FP16/BF16
- Use gradient checkpointing

## 📊 Performance Considerations

### Memory Usage

- Base model: ~2.3 GB (FP32) or ~1.2 GB (FP16)
- LoRA adapters: ~50 MB
- New token embeddings: ~10 MB
- Training: ~4-6 GB (with batch size 8)

### Training Time

- Tokenizer extension: ~5-10 minutes
- Model initialization: ~1-2 minutes
- Training (10 epochs, ~80k samples): ~2-4 hours (GPU)

### Inference Time

- Single sentence: ~50-100 ms
- Batch (16 sentences): ~200-300 ms

## 🎯 Best Practices

### 1. Tokenizer Extension

- Start with 100-200 new tokens
- Prioritize rare scripts
- Filter by frequency (min_frequency=2)
- Verify tokens are actually new

### 2. LoRA Configuration

- Start with r=16, alpha=32
- Adjust based on dataset size
- Use dropout=0.1
- Target attention and FFN layers

### 3. Training

- Use low learning rate (5e-5)
- Monitor validation loss
- Use early stopping
- Save best model

### 4. Evaluation

- Evaluate on validation set during training
- Evaluate on test set after training
- Compute both BLEU and chrF
- Check for catastrophic forgetting

## 📚 References

- [NLLB Paper](https://arxiv.org/abs/2207.04672)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

