# Detailed Training Guide for MMLoSo NMT Fine-tuning

## 🧠 Step-by-Step Process

### Step 1: Extend Tokenizer

**Goal**: Add new vocabulary tokens for low-resource languages without destroying existing vocabulary.

**Process**:
1. Load base NLLB tokenizer
2. Extract token candidates from training data
3. Filter out tokens that already exist
4. Select top candidates (prioritizing rare scripts)
5. Add tokens using `add_tokens()` method
6. Save extended tokenizer

**Code**:
```bash
python main.py --mode extend_tokenizer --local --new-tokens 200
```

**Key Points**:
- Uses `tokenizer.add_tokens()` which preserves existing vocabulary
- Prioritizes Ol Chiki (Santali) and Devanagari variants
- Filters by frequency (min_frequency=2)
- Saves tokenizer and new tokens list

### Step 2: Prepare Data

**Goal**: Load and preprocess training data for all language pairs.

**Process**:
1. Load CSV files for each language pair
2. Clean data (remove NaN, empty strings)
3. Split into train/val (95/5)
4. Create TranslationDataset for each pair
5. Combine into ConcatDataset

**Key Points**:
- Handles multiple language pairs simultaneously
- Proper language code mapping for NLLB
- Tokenization with language codes (src_lang, tgt_lang)

### Step 3: Initialize Model

**Goal**: Load model and extend embeddings for new tokens.

**Process**:
1. Load base NLLB model
2. Resize token embeddings to match extended tokenizer
3. Initialize new token embeddings (average strategy)
4. Apply LoRA configuration
5. Freeze base model (only LoRA + new embeddings trainable)

**Key Points**:
- New embeddings initialized with average of existing embeddings
- LoRA applied to attention and feed-forward layers
- Base model frozen to prevent catastrophic forgetting

### Step 4: Train Model

**Goal**: Fine-tune model using LoRA.

**Process**:
1. Create data collator
2. Set up training arguments
3. Create trainer
4. Train model
5. Save checkpoints

**Key Points**:
- Only LoRA adapters and new embeddings are trainable
- Low learning rate (5e-5) to prevent overfitting
- Gradient accumulation for effective larger batch size
- Evaluation on validation set

### Step 5: Evaluate

**Goal**: Evaluate model on test set.

**Process**:
1. Load trained model
2. Generate translations for test set
3. Compute BLEU and chrF scores
4. Compute final competition score

## 🧩 Architecture Details

### Tokenizer Extension

```
Base Tokenizer (256k tokens)
    ↓
Extract Candidates (from training data)
    ↓
Filter Existing (remove tokens already in vocab)
    ↓
Select Top N (prioritize rare scripts)
    ↓
Add Tokens (add_tokens() - preserves existing)
    ↓
Extended Tokenizer (256k + N tokens)
```

### LoRA Architecture

```
Base Model (Frozen)
    ├── Embeddings (Extended, Trainable for new tokens)
    ├── Encoder (Frozen)
    │   ├── Attention (LoRA adapters - Trainable)
    │   └── FFN (LoRA adapters - Trainable)
    └── Decoder (Frozen)
        ├── Attention (LoRA adapters - Trainable)
        └── FFN (LoRA adapters - Trainable)
```

### Training Flow

```
Input (Source Language)
    ↓
Tokenizer (with language code)
    ↓
Encoder (Frozen + LoRA)
    ↓
Decoder (Frozen + LoRA)
    ↓
Output (Target Language)
    ↓
Loss (CrossEntropy)
    ↓
Backprop (only LoRA + new embeddings)
```

## ⚠️ Common Issues and Solutions

### Issue 1: Out of Memory

**Symptoms**: CUDA out of memory error

**Solutions**:
- Reduce batch size: `--batch-size 4`
- Increase gradient accumulation: `gradient_accumulation_steps=8`
- Use gradient checkpointing
- Use FP16: `--fp16`

### Issue 2: Poor Translation Quality

**Symptoms**: Low BLEU/chrF scores

**Solutions**:
- Increase LoRA rank: `--lora-r 32`
- Add more new tokens: `--new-tokens 300`
- Train for more epochs: `--epochs 20`
- Check data quality (remove noisy samples)
- Adjust learning rate: `--learning-rate 3e-5`

### Issue 3: Catastrophic Forgetting

**Symptoms**: Model performs worse on original languages

**Solutions**:
- Verify base model is frozen
- Reduce learning rate
- Increase LoRA dropout: `--lora-dropout 0.2`
- Use smaller LoRA rank: `--lora-r 8`

### Issue 4: Tokenizer Extension Fails

**Symptoms**: New tokens not added or errors

**Solutions**:
- Check token frequency (increase min_frequency)
- Verify script detection works
- Check tokenizer compatibility
- Manually specify new tokens if needed

### Issue 5: NLLB Language Codes Not Found

**Symptoms**: Language code errors

**Solutions**:
- Verify language code mapping in `src/utils.py`
- Check NLLB supported languages
- Use fallback language codes
- Handle unsupported languages gracefully

## 🔬 Advanced Techniques

### 1. Data Augmentation

```python
# Back-translation
# 1. Train reverse model (HRL → LRL)
# 2. Translate high-resource data
# 3. Add to training set

# Synonym replacement
# Replace words with synonyms in source
```

### 2. Curriculum Learning

```python
# Start with shorter sentences
# Gradually increase length
# Helps model learn basic patterns first
```

### 3. Multi-task Learning

```python
# Train on multiple language pairs simultaneously
# Shared representations help low-resource languages
# Use language tags to distinguish
```

### 4. Ensemble Methods

```python
# Train multiple models with different seeds
# Average predictions at inference
# Improves robustness
```

### 5. Few-shot Learning

```python
# Use in-context learning with examples
# Provide few examples in prompt
# Model learns from context
```

## 📊 Evaluation Metrics

### BLEU Score

- Measures n-gram overlap between prediction and reference
- Range: 0-100 (higher is better)
- Good for: General translation quality

### chrF Score

- Character-level F-score
- Better for morphologically rich languages
- Range: 0-100 (higher is better)
- Good for: Low-resource languages with complex morphology

### Final Score (MMLoSo Competition)

```
Final = 0.6 * (0.6 * BLEU_forward + 0.4 * BLEU_reverse)
      + 0.4 * (0.6 * chrF_forward + 0.4 * chrF_reverse)
```

## 🎯 Best Practices

### 1. Tokenizer Extension

- Start with 100-200 new tokens
- Prioritize rare scripts (Ol Chiki, Devanagari variants)
- Filter by frequency (min_frequency=2)
- Verify tokens are actually new

### 2. LoRA Configuration

- Start with r=16, alpha=32 (balanced)
- Adjust based on dataset size
- Use dropout=0.1 (prevents overfitting)
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

## 📝 Script-Specific Tips

### Ol Chiki (Santali)

- Unicode range: U+1C50 to U+1C7F
- Ensure tokenizer handles these characters
- May need special tokens for script markers
- Handle Romanized variants

### Devanagari (Hindi, Bhili, Mundari, Gondi)

- Unicode range: U+0900 to U+097F
- Handle compound characters (conjuncts)
- Normalize different variants
- Handle diacritics properly

### Roman (English)

- Standard ASCII/Latin characters
- Handle case sensitivity
- Normalize punctuation
- Handle abbreviations

## 🚀 Quick Reference

### Commands

```bash
# Extend tokenizer
python main.py --mode extend_tokenizer --local --new-tokens 200

# Train model
python main.py --mode train --local --epochs 10 --batch-size 8 --lora-r 16

# Evaluate
python main.py --mode eval --local
```

### Key Parameters

- `--local`: Use local dataset path
- `--new-tokens`: Number of new tokens to add
- `--lora-r`: LoRA rank (8, 16, 32)
- `--lora-alpha`: LoRA alpha (usually 2x r)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate (default: 5e-5)

### File Structure

```
models/
├── checkpoints/          # Training checkpoints
├── tokenizer_extended/   # Extended tokenizer
└── best_model/          # Best model (if saved)
```

## 📚 Additional Resources

- [NLLB Paper](https://arxiv.org/abs/2207.04672)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

