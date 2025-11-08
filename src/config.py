"""
Configuration settings for MMLoSo NMT fine-tuning
"""
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "facebook/nllb-200-distilled-600M"
    max_length: int = 512
    max_new_tokens: int = 256


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 16  # Rank
    alpha: int = 32  # LoRA alpha (usually 2x r)
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "TRANSLATION"
    
    def __post_init__(self):
        if self.target_modules is None:
            # For NLLB, target attention and feed-forward layers
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./models/checkpoints"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_bleu"
    greater_is_better: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 4
    report_to: str = "none"  # "tensorboard", "wandb", or "none"


@dataclass
class TokenizerConfig:
    """Tokenizer extension configuration"""
    new_tokens_count: int = 200  # Number of new tokens to add
    min_frequency: int = 2  # Minimum frequency to consider a token
    preserve_existing: bool = True  # Preserve existing vocabulary


@dataclass
class DataConfig:
    """Data configuration"""
    train_split: float = 0.95
    val_split: float = 0.05
    seed: int = 42
    language_pairs: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.language_pairs is None:
            self.language_pairs = {
                "bhili": ["Hindi", "Bhili"],
                "gondi": ["Hindi", "Gondi"],
                "mundari": ["Hindi", "Mundari"],
                "santali": ["English", "Santali"]
            }


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    path_root: str = "./dataset"
    is_local: bool = True
    cache_dir: Optional[str] = None
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create config from command line arguments"""
        # Determine path root
        if args.local:
            path_root = "./dataset"
        else:
            path_root = "/kaggle/input/mm-lo-so-2025"
        
        return cls(
            model=ModelConfig(),
            lora=LoRAConfig(
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout
            ),
            training=TrainingConfig(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                fp16=args.fp16
            ),
            tokenizer=TokenizerConfig(
                new_tokens_count=args.new_tokens
            ),
            data=DataConfig(),
            path_root=path_root,
            is_local=args.local,
            cache_dir=args.cache_dir
        )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MMLoSo NMT Fine-tuning with LoRA"
    )
    
    # Path arguments
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local dataset path (./dataset) instead of Kaggle path"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace models"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Base model name"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training"
    )
    
    # Tokenizer arguments
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=200,
        help="Number of new tokens to add"
    )
    
    # Mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "extend_tokenizer"],
        help="Mode: train, eval, or extend_tokenizer"
    )
    
    return parser.parse_args()

