# Project Overview

## 📁 File Structure

```
MMLOSO/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── tokenizer_utils.py       # Tokenizer extension utilities
│   ├── trainer.py               # LoRA training utilities
│   ├── evaluation.py            # Evaluation metrics
│   └── utils.py                 # Utility functions
├── dataset/                      # Dataset directory
│   ├── bhili-train.csv          # Hindi-Bhili training data
│   ├── gondi-train.csv          # Hindi-Gondi training data
│   ├── mundari-train.csv        # Hindi-Mundari training data
│   ├── santali-train.csv        # English-Santali training data
│   └── test.csv                 # Test data
├── models/                       # Model outputs (created during training)
│   ├── tokenizer_extended/      # Extended tokenizer
│   └── checkpoints/             # Training checkpoints
├── main.py                      # Main training script
├── test_setup.py                # Setup test script
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
├── QUICKSTART.md                # Quick start guide
├── TRAINING_GUIDE.md            # Detailed training guide
├── ARCHITECTURE.md              # Architecture details
├── PROJECT_OVERVIEW.md          # This file
└── problems.txt                 # Competition problem description
```

## 📋 File Descriptions

### Core Source Files

#### `src/config.py`
- Configuration management
- Command-line argument parsing
- Configuration classes for model, LoRA, training, tokenizer, and data
- Path switching between local and Kaggle environments

#### `src/data_utils.py`
- Data loading from CSV files
- Data preprocessing and cleaning
- Dataset creation for translation tasks
- Train/validation splitting
- Support for multiple language pairs

#### `src/tokenizer_utils.py`
- Tokenizer extension utilities
- Token extraction from training data
- Script-aware token selection
- Filtering existing tokens
- Embedding initialization for new tokens

#### `src/trainer.py`
- LoRA model creation
- Model freezing utilities
- Training setup and execution
- Data collation for NLLB model

#### `src/evaluation.py`
- BLEU score computation
- chrF score computation
- Model evaluation utilities
- Competition score calculation

#### `src/utils.py`
- Utility functions
- Language code mapping
- File I/O utilities
- Validation functions

### Main Scripts

#### `main.py`
- Main entry point for training
- Orchestrates tokenizer extension, data preparation, training, and evaluation
- Supports multiple modes: `train`, `eval`, `extend_tokenizer`
- Handles path switching between local and Kaggle

#### `test_setup.py`
- Tests package installation
- Verifies data loading
- Tests tokenizer loading
- Validates configuration

### Documentation

#### `README.md`
- Main documentation
- High-level overview
- Architecture diagrams
- Installation instructions
- Usage examples
- Best practices
- Troubleshooting guide

#### `QUICKSTART.md`
- Quick start guide
- Basic usage examples
- Command-line arguments
- Recommended settings
- Troubleshooting tips

#### `TRAINING_GUIDE.md`
- Detailed training guide
- Step-by-step process
- Advanced techniques
- Common issues and solutions
- Performance tips

#### `ARCHITECTURE.md`
- Architecture details
- Component descriptions
- Data flow diagrams
- Performance considerations
- Best practices

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Test setup**:
```bash
python test_setup.py
```

3. **Extend tokenizer**:
```bash
python main.py --mode extend_tokenizer --local --new-tokens 200
```

4. **Train model**:
```bash
python main.py --mode train --local --epochs 10 --batch-size 8
```

## 🔑 Key Features

### 1. Safe Tokenizer Extension
- Preserves existing vocabulary
- Script-aware token selection
- Frequency-based filtering
- Priority for rare scripts

### 2. LoRA Fine-tuning
- Minimal trainable parameters (~0.11%)
- Prevents catastrophic forgetting
- Efficient training
- Configurable rank and alpha

### 3. Multi-language Support
- Handles multiple language pairs
- Different scripts (Devanagari, Ol Chiki, Roman)
- Language code mapping
- Proper tokenization

### 4. Path Switching
- Local environment: `./dataset`
- Kaggle environment: `/kaggle/input/mm-lo-so-2025`
- Automatic switching with `--local` flag

### 5. Comprehensive Evaluation
- BLEU score
- chrF score
- Competition score calculation
- Multi-direction evaluation

## 📊 Workflow

```
1. Extend Tokenizer
   ├── Load base tokenizer
   ├── Extract token candidates
   ├── Filter existing tokens
   ├── Select top N tokens
   └── Add tokens (preserve existing)

2. Prepare Data
   ├── Load CSV files
   ├── Clean data
   ├── Split train/val
   └── Create datasets

3. Initialize Model
   ├── Load base model
   ├── Extend embeddings
   ├── Initialize new tokens
   ├── Apply LoRA
   └── Freeze base model

4. Train Model
   ├── Create data collator
   ├── Set up training arguments
   ├── Create trainer
   ├── Train model
   └── Save checkpoints

5. Evaluate
   ├── Load model
   ├── Generate translations
   ├── Compute metrics
   └── Calculate final score
```

## 🎯 Usage Examples

### Extend Tokenizer
```bash
python main.py --mode extend_tokenizer --local --new-tokens 200
```

### Train Model
```bash
python main.py --mode train --local --epochs 10 --batch-size 8 --lora-r 16
```

### Evaluate
```bash
python main.py --mode eval --local
```

### Kaggle Environment
```bash
python main.py --mode train --epochs 10 --batch-size 8
```

## 🔧 Configuration

### LoRA Configuration
- `r`: Rank (8, 16, or 32)
- `alpha`: Scaling factor (typically 2× r)
- `dropout`: Dropout rate (0.1)
- `target_modules`: Attention and FFN layers

### Training Configuration
- `learning_rate`: 5e-5
- `batch_size`: 8
- `epochs`: 10
- `gradient_accumulation_steps`: 4
- `warmup_steps`: 500

### Tokenizer Configuration
- `new_tokens_count`: 200
- `min_frequency`: 2
- `preserve_existing`: True

## 📈 Performance

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

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size, use FP16
2. **Poor Performance**: Increase LoRA rank, add more tokens
3. **Catastrophic Forgetting**: Verify base model is frozen
4. **Tokenizer Extension Fails**: Check data files, verify encoding
5. **Language Code Errors**: Check language code mapping

### Solutions
- See `TRAINING_GUIDE.md` for detailed solutions
- Check `README.md` for common mistakes
- Review error messages carefully
- Test with smaller dataset first

## 📚 Additional Resources

- [README.md](README.md): Main documentation
- [QUICKSTART.md](QUICKSTART.md): Quick start guide
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): Detailed training guide
- [ARCHITECTURE.md](ARCHITECTURE.md): Architecture details
- [problems.txt](problems.txt): Competition problem description

## 🤝 Contributing

1. Read documentation
2. Test changes
3. Follow code style
4. Update documentation
5. Submit pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Facebook AI Research for NLLB model
- HuggingFace for transformers and PEFT libraries
- MMLoSo organizers for the dataset and challenge

