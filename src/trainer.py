"""
Training utilities for LoRA fine-tuning
"""
import os
import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import numpy as np

from src.tokenizer_utils import initialize_new_token_embeddings
from src.data_utils import TranslationDataset


class NLLBDataCollator:
    """Data collator for NLLB model with language codes"""
    
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        self.tokenizer = tokenizer
        self.model = model
    
    def __call__(self, features):
        """Collate batch of features"""
        # Extract fields
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_lora_model(
    model_name: str,
    tokenizer: AutoTokenizer,
    lora_config: Dict[str, Any],
    new_tokens: Optional[list] = None,
    cache_dir: Optional[str] = None
) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Create LoRA model with extended tokenizer
    
    Args:
        model_name: Base model name
        tokenizer: Extended tokenizer
        lora_config: LoRA configuration
        new_tokens: List of new tokens (for embedding initialization)
        cache_dir: Cache directory for models
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load base model
    print(f"Loading base model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Resize token embeddings if tokenizer was extended
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing token embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
        # Initialize new token embeddings
        if new_tokens:
            initialize_new_token_embeddings(
                model,
                tokenizer,
                new_tokens,
                initialization_strategy="average"
            )
    
    # Create LoRA configuration
    # For Seq2Seq models (like NLLB), we use SEQ_2_SEQ_LM task type
    # Note: PEFT may require different task types depending on version
    try:
        # Try SEQ_2_SEQ_LM first (newer PEFT versions)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", None),
            bias=lora_config.get("bias", "none")
        )
    except (AttributeError, ValueError):
        # Fallback to FEATURE_EXTRACTION for older PEFT versions
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", None),
            bias=lora_config.get("bias", "none")
        )
    
    # Apply LoRA
    print("Applying LoRA to model...")
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def freeze_base_model(model: AutoModelForSeq2SeqLM, freeze_embeddings: bool = False):
    """
    Freeze base model layers, keeping only LoRA and embeddings trainable
    
    Args:
        model: PEFT model
        freeze_embeddings: If True, also freeze embeddings (only train LoRA)
    """
    # PEFT automatically freezes base model, but we can explicitly freeze embeddings
    if freeze_embeddings:
        # Freeze input and output embeddings
        if hasattr(model, "model"):
            if hasattr(model.model, "get_input_embeddings"):
                for param in model.model.get_input_embeddings().parameters():
                    param.requires_grad = False
            if hasattr(model.model, "get_output_embeddings"):
                for param in model.model.get_output_embeddings().parameters():
                    param.requires_grad = False
        else:
            if hasattr(model, "get_input_embeddings"):
                for param in model.get_input_embeddings().parameters():
                    param.requires_grad = False
            if hasattr(model, "get_output_embeddings"):
                for param in model.get_output_embeddings().parameters():
                    param.requires_grad = False
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")


def compute_metrics(eval_pred, tokenizer: AutoTokenizer):
    """Compute BLEU and chrF metrics"""
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # For now, return placeholder metrics
    # In production, use sacrebleu and chrF
    return {
        "bleu": 0.0,  # Will be computed in evaluation script
        "chrf": 0.0
    }


def train_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_dataset: TranslationDataset,
    val_dataset: TranslationDataset,
    training_args: Dict[str, Any],
    output_dir: str
):
    """
    Train model with LoRA
    
    Args:
        model: LoRA model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_args: Training arguments
        output_dir: Output directory
    """
    # Create data collator
    data_collator = NLLBDataCollator(tokenizer, model)
    
    # Create training arguments
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **training_args
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer

