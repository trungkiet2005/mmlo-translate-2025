"""
Main training script for MMLoSo NMT fine-tuning
"""
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, parse_args
from src.data_utils import DataLoader, prepare_data_for_tokenizer_extraction
from src.tokenizer_utils import TokenizerExtender
from src.trainer import create_lora_model, freeze_base_model, train_model
from src.evaluation import evaluate_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def extend_tokenizer(config: Config):
    """Extend tokenizer with new vocabulary tokens"""
    print("=" * 80)
    print("STEP 1: Extending Tokenizer")
    print("=" * 80)
    
    # Load base tokenizer
    print(f"Loading base tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        cache_dir=config.cache_dir,
        src_lang="eng_Latn",
        tgt_lang="hin_Deva"
    )
    
    # Load training data
    data_loader = DataLoader(config.path_root)
    
    # Prepare texts for tokenizer extension
    dataset_names = list(config.data.language_pairs.keys())
    training_texts = prepare_data_for_tokenizer_extraction(data_loader, dataset_names)
    
    print(f"Loaded {len(training_texts)} training texts")
    
    # Create tokenizer extender
    extender = TokenizerExtender(
        tokenizer,
        new_tokens_count=config.tokenizer.new_tokens_count,
        min_frequency=config.tokenizer.min_frequency
    )
    
    # Extend tokenizer
    extended_tokenizer = extender.extend_tokenizer(
        training_texts,
        save_path="./models/tokenizer_extended"
    )
    
    # Save new tokens list
    new_tokens = extender.new_tokens
    print(f"New tokens added: {len(new_tokens)}")
    print(f"Sample new tokens: {new_tokens[:10]}")
    
    return extended_tokenizer, new_tokens


def prepare_datasets(config: Config, tokenizer):
    """Prepare training and validation datasets"""
    print("=" * 80)
    print("STEP 2: Preparing Datasets")
    print("=" * 80)
    
    # Load data
    data_loader = DataLoader(config.path_root)
    
    # Prepare datasets
    dataset_names = list(config.data.language_pairs.keys())
    train_dataset, val_dataset = data_loader.prepare_datasets(
        dataset_names=dataset_names,
        tokenizer=tokenizer,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        max_length=config.model.max_length,
        seed=config.data.seed
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def train(config: Config, tokenizer, new_tokens, train_dataset, val_dataset):
    """Train model with LoRA"""
    print("=" * 80)
    print("STEP 3: Training Model with LoRA")
    print("=" * 80)
    
    # Create LoRA model
    lora_config = {
        "r": config.lora.r,
        "alpha": config.lora.alpha,
        "dropout": config.lora.dropout,
        "target_modules": config.lora.target_modules,
        "bias": config.lora.bias
    }
    
    model, tokenizer = create_lora_model(
        model_name=config.model.model_name,
        tokenizer=tokenizer,
        lora_config=lora_config,
        new_tokens=new_tokens,
        cache_dir=config.cache_dir
    )
    
    # Freeze base model (LoRA handles this, but we can also freeze embeddings)
    freeze_base_model(model, freeze_embeddings=False)  # Keep embeddings trainable
    
    # Prepare training arguments
    training_args = {
        "num_train_epochs": config.training.num_train_epochs,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "weight_decay": config.training.weight_decay,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "eval_steps": config.training.eval_steps,
        "save_total_limit": config.training.save_total_limit,
        "load_best_model_at_end": config.training.load_best_model_at_end,
        "metric_for_best_model": config.training.metric_for_best_model,
        "greater_is_better": config.training.greater_is_better,
        "fp16": config.training.fp16,
        "dataloader_num_workers": config.training.dataloader_num_workers,
        "report_to": config.training.report_to,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "logging_dir": f"{config.training.output_dir}/logs"
    }
    
    # Create output directory
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        output_dir=config.training.output_dir
    )
    
    return model, tokenizer, trainer


def evaluate(config: Config, model, tokenizer, val_dataset):
    """Evaluate model"""
    print("=" * 80)
    print("STEP 4: Evaluating Model")
    print("=" * 80)
    
    # Evaluate on validation set
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=val_dataset,
        batch_size=config.training.per_device_eval_batch_size,
        max_length=config.model.max_length
    )
    
    print(f"BLEU Score: {results['bleu']:.4f}")
    print(f"chrF Score: {results['chrf']:.4f}")
    
    return results


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = Config.from_args(args)
    
    print("=" * 80)
    print("MMLoSo NMT Fine-tuning with LoRA")
    print("=" * 80)
    print(f"Model: {config.model.model_name}")
    print(f"Path Root: {config.path_root}")
    print(f"Mode: {args.mode}")
    print(f"Local: {config.is_local}")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == "extend_tokenizer":
        # Only extend tokenizer
        tokenizer, new_tokens = extend_tokenizer(config)
        print("Tokenizer extension complete!")
        return
    
    # Load or extend tokenizer
    tokenizer_path = "./models/tokenizer_extended"
    if Path(tokenizer_path).exists() and Path(tokenizer_path / "new_tokens.json").exists():
        print("Loading existing extended tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        import json
        with open(Path(tokenizer_path) / "new_tokens.json", "r", encoding="utf-8") as f:
            new_tokens = json.load(f)
    else:
        tokenizer, new_tokens = extend_tokenizer(config)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)
    
    # Note: prepare_datasets returns ConcatDataset, which is compatible with Trainer
    
    if args.mode == "train":
        # Train model
        model, tokenizer, trainer = train(config, tokenizer, new_tokens, train_dataset, val_dataset)
        
        # Evaluate
        evaluate(config, model, tokenizer, val_dataset)
        
        print("=" * 80)
        print("Training complete!")
        print(f"Model saved to: {config.training.output_dir}")
        print("=" * 80)
    
    elif args.mode == "eval":
        # Load model for evaluation
        print("Loading model for evaluation...")
        # This would load a trained model
        # For now, we'll skip this
        print("Evaluation mode not yet implemented for standalone evaluation")
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

