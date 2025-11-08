# Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Test setup**:
```bash
python test_setup.py
```

## Basic Usage

### 1. Extend Tokenizer (First Time Only)

```bash
# Local environment
python main.py --mode extend_tokenizer --local --new-tokens 200

# Kaggle environment
python main.py --mode extend_tokenizer --new-tokens 200
```

This will:
- Analyze training data
- Extract new token candidates
- Add tokens to tokenizer (preserving existing vocabulary)
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

## Command Line Arguments

### Path Arguments
- `--local`: Use local dataset path (./dataset) instead of Kaggle path
- `--cache-dir`: Cache directory for HuggingFace models

### Model Arguments
- `--model-name`: Base model name (default: facebook/nllb-200-distilled-600M)

### LoRA Arguments
- `--lora-r`: LoRA rank (default: 16)
- `--lora-alpha`: LoRA alpha (default: 32)
- `--lora-dropout`: LoRA dropout (default: 0.1)

### Training Arguments
- `--output-dir`: Output directory for checkpoints (default: ./models/checkpoints)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 8)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--fp16`: Use FP16 training

### Tokenizer Arguments
- `--new-tokens`: Number of new tokens to add (default: 200)

### Mode Arguments
- `--mode`: Mode to run (train, eval, extend_tokenizer)

## Recommended Settings

### Small Dataset (< 10k samples per language pair)
```bash
python main.py --mode train --local \
    --epochs 15 \
    --batch-size 4 \
    --lora-r 8 \
    --lora-alpha 16 \
    --learning-rate 3e-5 \
    --new-tokens 100
```

### Medium Dataset (10k-50k samples per language pair)
```bash
python main.py --mode train --local \
    --epochs 10 \
    --batch-size 8 \
    --lora-r 16 \
    --lora-alpha 32 \
    --learning-rate 5e-5 \
    --new-tokens 200
```

### Large Dataset (> 50k samples per language pair)
```bash
python main.py --mode train --local \
    --epochs 8 \
    --batch-size 16 \
    --lora-r 32 \
    --lora-alpha 64 \
    --learning-rate 5e-5 \
    --new-tokens 300
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 4`
- Use gradient accumulation (edit config.py)
- Use FP16: `--fp16`

### Poor Performance
- Increase LoRA rank: `--lora-r 32`
- Add more tokens: `--new-tokens 300`
- Train longer: `--epochs 20`

### Tokenizer Extension Fails
- Check data files exist
- Verify file encoding (should be UTF-8)
- Check token frequency threshold

## File Structure

After running, you should have:

```
MMLOSO/
├── models/
│   ├── tokenizer_extended/
│   │   ├── tokenizer_config.json
│   │   ├── sentencepiece.bpe.model
│   │   └── new_tokens.json
│   └── checkpoints/
│       ├── checkpoint-1000/
│       ├── checkpoint-2000/
│       └── ...
├── dataset/
│   ├── bhili-train.csv
│   ├── gondi-train.csv
│   ├── mundari-train.csv
│   ├── santali-train.csv
│   └── test.csv
└── ...
```

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for advanced techniques
3. Check `problems.txt` for competition details
4. Experiment with different LoRA configurations
5. Evaluate on test set and submit results

## Support

For issues or questions:
1. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for common issues
2. Review error messages carefully
3. Verify data files are correct
4. Test with smaller dataset first

