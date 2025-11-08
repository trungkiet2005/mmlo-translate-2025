# MMLoSo NMT Fine-tuning with LoRA

Comprehensive solution for fine-tuning NLLB-200-distilled-600M model for low-resource Indian languages (Bhili, Mundari, Gondi, Santali) using LoRA (Low-Rank Adaptation) and tokenizer extension.

## 🧠 High-level Overview

This project implements a complete pipeline for:

1. **Tokenizer Extension**: Safely add new vocabulary tokens (50-300 tokens) for low-resource languages without destroying existing vocabulary
2. **LoRA Fine-tuning**: Fine-tune the model using Low-Rank Adaptation to avoid catastrophic forgetting
3. **Multi-language Support**: Handle multiple language pairs with different scripts (Devanagari, Ol Chiki, Roman)
4. **Evaluation**: Comprehensive evaluation with BLEU and chrF metrics

## 🧩 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Base NLLB-200 Model                      │
│              (facebook/nllb-200-distilled-600M)             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Embeddings  │  │  Encoder     │  │  Decoder     │     │
│  │  (Extended)  │  │  (Frozen)    │  │  (Frozen)    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│                  ┌─────────▼─────────┐                       │
│                  │   LoRA Adapters   │                       │
│                  │  (Trainable Only) │                       │
│                  │                   │                       │
│                  │  - q_proj         │                       │
│                  │  - k_proj         │                       │
│                  │  - v_proj         │                       │
│                  │  - o_proj         │                       │
│                  │  - gate_proj      │                       │
│                  │  - up_proj        │                       │
│                  │  - down_proj      │                       │
│                  └───────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

Training Flow:
1. Extend Tokenizer → Add new tokens (preserve existing vocab)
2. Initialize Embeddings → Average initialization for new tokens
3. Freeze Base Model → Keep original 200 languages intact
4. Train LoRA + New Embeddings → Only train adapters and new token embeddings
5. Evaluate → BLEU + chrF metrics
```

## 📁 Project Structure

```
MMLOSO/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── tokenizer_utils.py     # Tokenizer extension utilities
│   ├── trainer.py             # LoRA training utilities
│   └── evaluation.py          # Evaluation metrics
├── dataset/                   # Dataset directory
│   ├── bhili-train.csv
│   ├── gondi-train.csv
│   ├── mundari-train.csv
│   ├── santali-train.csv
│   └── test.csv
├── models/                    # Model checkpoints
│   ├── checkpoints/           # Training checkpoints
│   └── tokenizer_extended/    # Extended tokenizer
├── main.py                    # Main training script
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── problems.txt               # Problem description
```

## 🛠 Installation

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; import transformers; import peft; print('All packages installed successfully!')"
```

## 🚀 Quick Start

### 1. Extend Tokenizer

```bash
# Local environment
python main.py --mode extend_tokenizer --local

# Kaggle environment
python main.py --mode extend_tokenizer
```

This will:
- Load training data from all language pairs
- Extract new token candidates (focusing on Ol Chiki and Devanagari scripts)
- Filter out existing tokens
- Add new tokens to tokenizer (preserving existing vocabulary)
- Save extended tokenizer to `./models/tokenizer_extended/`

### 2. Train Model

```bash
# Local environment
python main.py --mode train --local --epochs 10 --batch-size 8 --lora-r 16

# Kaggle environment
python main.py --mode train --epochs 10 --batch-size 8 --lora-r 16
```

### 3. Evaluate

```bash
python main.py --mode eval --local
```

## 📊 Configuration

### LoRA Configuration

**Recommended settings for low-resource languages**:

```python
# Conservative (small dataset, avoid overfitting)
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1

# Balanced (recommended starting point)
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1

# Aggressive (larger dataset, more capacity)
lora_r = 32
lora_alpha = 64
lora_dropout = 0.05
```

### Training Configuration

```python
# For ~20k samples per language pair
num_epochs = 10
batch_size = 8
gradient_accumulation_steps = 4
learning_rate = 5e-5
warmup_steps = 500
```

## 🔬 Key Features

### 1. Safe Tokenizer Extension

- **Preserves existing vocabulary**: Uses `add_tokens()` method which doesn't modify existing tokens
- **Smart token selection**: Prioritizes rare scripts (Ol Chiki, Devanagari variants)
- **Frequency filtering**: Only adds tokens that appear multiple times
- **Script-aware extraction**: Handles different scripts separately

### 2. Embedding Initialization

Three strategies available:

- **Average** (recommended): Initialize new token embeddings with average of existing embeddings
- **Random**: Initialize with small random values
- **Zero**: Initialize with zeros (not recommended)

### 3. LoRA Configuration

**Target modules** (for NLLB):
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- `gate_proj`, `up_proj`, `down_proj` (feed-forward layers)

**Why these modules?**
- Attention layers capture language-specific patterns
- Feed-forward layers handle semantic transformations
- Only ~0.1% of parameters are trainable (prevents overfitting)

### 4. Avoiding Catastrophic Forgetting

- **Freeze base model**: All original parameters remain frozen
- **Train only LoRA**: Only adapter weights are updated
- **Train new embeddings**: New token embeddings are trainable
- **Low learning rate**: Conservative updates (5e-5)

## ⚠️ Common Mistakes and Solutions

### 1. **Tokenizer Extension Destroys Vocabulary**

**Problem**: Retraining SentencePiece from scratch loses existing vocabulary

**Solution**: Use `tokenizer.add_tokens()` which preserves existing tokens

```python
# ✅ Correct
tokenizer.add_tokens(new_tokens, special_tokens=False)

# ❌ Wrong
# Don't retrain SentencePiece from scratch
```

### 2. **New Tokens Not Initialized**

**Problem**: New token embeddings are random, hurting performance

**Solution**: Initialize with average of existing embeddings

```python
# ✅ Correct
avg_embedding = existing_embeddings.mean(dim=0)
new_embedding[token_id] = avg_embedding.clone()
```

### 3. **Catastrophic Forgetting**

**Problem**: Fine-tuning destroys original language capabilities

**Solution**: Use LoRA instead of full fine-tuning

```python
# ✅ Correct - LoRA (only ~0.1% parameters trainable)
peft_config = LoraConfig(r=16, alpha=32, ...)

# ❌ Wrong - Full fine-tuning (all parameters trainable)
# model.train()  # Don't do this!
```

### 4. **Wrong LoRA Target Modules**

**Problem**: Targeting wrong layers reduces effectiveness

**Solution**: Target attention and feed-forward layers

```python
# ✅ Correct
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ❌ Wrong
target_modules = ["embed_tokens"]  # Too limited
```

### 5. **Insufficient Data Augmentation**

**Problem**: Low-resource languages have limited data

**Solution**: Use data augmentation techniques (back-translation, synonym replacement)

## 🔬 Hacks to Improve Translation Quality

### 1. **Data Augmentation**

```python
# Back-translation
# 1. Train a reverse model (HRL → LRL)
# 2. Translate high-resource data to low-resource
# 3. Add to training set

# Synonym replacement
# Replace words with synonyms in source language
```

### 2. **Curriculum Learning**

```python
# Start with easier examples (shorter sentences)
# Gradually increase difficulty
# Helps model learn basic patterns first
```

### 3. **Multi-task Learning**

```python
# Train on multiple language pairs simultaneously
# Shared representations help low-resource languages
# Use language tags to distinguish pairs
```

### 4. **Ensemble Methods**

```python
# Train multiple models with different seeds
# Average predictions at inference
# Improves robustness
```

### 5. **Script Normalization**

```python
# Normalize different script variants
# Handle Romanized vs native script
# Use script conversion libraries
```

### 6. **Few-shot Learning**

```python
# Use in-context learning with examples
# Provide few examples in prompt
# Model learns from context
```

## 📈 Evaluation

### Metrics

1. **BLEU Score**: Measures n-gram overlap
2. **chrF Score**: Character-level F-score (better for morphologically rich languages)

### Evaluation Formula (MMLoSo Competition)

```
Final Score = 0.6 * (0.6 * BLEU_forward + 0.4 * BLEU_reverse)
            + 0.4 * (0.6 * chrF_forward + 0.4 * chrF_reverse)
```

### Language Pairs

- **Forward** (LRL → HRL): Bhili→Hindi, Mundari→Hindi, Gondi→Hindi, Santali→English
- **Reverse** (HRL → LRL): Hindi→Bhili, Hindi→Mundari, Hindi→Gondi, English→Santali

## 🎯 When to Use Different Training Strategies

### 1. **Adapter-Only Training** (Recommended for most cases)

**When**: 
- Limited data (< 50k samples)
- Want to preserve original model capabilities
- Need fast training

**Configuration**:
```python
freeze_embeddings = True  # Freeze all embeddings
train_only_lora = True    # Only train LoRA adapters
```

### 2. **Embedding-Only Training**

**When**:
- Many new tokens (> 500)
- New tokens are critical
- Limited compute

**Configuration**:
```python
freeze_base_model = True  # Freeze all base layers
train_only_embeddings = True  # Only train new token embeddings
```

### 3. **LoRA on Attention Layers**

**When**:
- Medium dataset (50k-200k samples)
- Need balance between capacity and stability
- Want to capture language-specific patterns

**Configuration**:
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_r = 16
lora_alpha = 32
```

### 4. **Full Fine-tuning** (Not Recommended)

**When**:
- Very large dataset (> 500k samples)
- Can afford risk of catastrophic forgetting
- Have resources for full retraining

**Configuration**:
```python
# Don't use LoRA, train all parameters
# High risk of catastrophic forgetting!
```

## 📝 Script Handling Tips

### 1. **Ol Chiki Script (Santali)**

```python
# Ol Chiki Unicode range: U+1C50 to U+1C7F
# Ensure tokenizer can handle these characters
# May need to add special tokens for script markers
```

### 2. **Devanagari Script (Hindi, Bhili, Mundari, Gondi)**

```python
# Devanagari Unicode range: U+0900 to U+097F
# Handle compound characters (conjuncts)
# Normalize different variants
```

### 3. **Roman Script (English)**

```python
# Standard ASCII/Latin characters
# Handle case sensitivity
# Normalize punctuation
```

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**:
- Reduce batch size
- Increase gradient accumulation steps
- Use gradient checkpointing
- Use FP16/BF16

### Issue: Poor Translation Quality

**Solution**:
- Increase LoRA rank (r=32)
- Add more new tokens
- Train for more epochs
- Check data quality

### Issue: Catastrophic Forgetting

**Solution**:
- Reduce learning rate
- Increase LoRA dropout
- Add regularization
- Use smaller LoRA rank

## 📚 References

- [NLLB Paper](https://arxiv.org/abs/2207.04672)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [MMLoSo Workshop](https://mm-loso.github.io/)

## 📄 License

[Your License Here]

## 👥 Contributors

[Your Name/Team]

## 🙏 Acknowledgments

- Facebook AI Research for NLLB model
- HuggingFace for transformers and PEFT libraries
- MMLoSo organizers for the dataset and challenge

