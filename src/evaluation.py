"""
Evaluation utilities for translation tasks
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

try:
    from sacrebleu import BLEU, CHRF
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("Warning: sacrebleu not installed. Install with: pip install sacrebleu")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
    tokenize: str = "13a"
) -> float:
    """
    Compute BLEU score
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        tokenize: Tokenization method
        
    Returns:
        BLEU score
    """
    if not HAS_SACREBLEU:
        return 0.0
    
    bleu = BLEU(tokenize=tokenize)
    score = bleu.corpus_score(predictions, [references])
    return score.score


def compute_chrf_score(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Compute chrF score
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        chrF score
    """
    if not HAS_SACREBLEU:
        return 0.0
    
    chrf = CHRF()
    score = chrf.corpus_score(predictions, [references])
    return score.score


def evaluate_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    source_lang: str = "Hindi",
    target_lang: str = "Bhili",
    batch_size: int = 16,
    max_length: int = 512,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate model on test dataset
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_dataset: Test dataset
        source_lang: Source language
        target_lang: Target language
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # Language code mapping
    lang_code_map = {
        "Hindi": "hin_Deva",
        "Bhili": "bhi_Deva",
        "Mundari": "unr_Deva",
        "Gondi": "gon_Deva",
        "English": "eng_Latn",
        "Santali": "sat_Olck"
    }
    
    source_lang_code = lang_code_map.get(source_lang, "eng_Latn")
    target_lang_code = lang_code_map.get(target_lang, "eng_Latn")
    
    predictions = []
    references = []
    
    # Generate predictions
    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i+batch_size]
            
            # Tokenize source texts
            source_texts = [item["source_text"] for item in batch]
            source_encoded = tokenizer(
                source_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate translations
            generated = model.generate(
                **source_encoded,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code]
            )
            
            # Decode predictions
            batch_predictions = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
            predictions.extend(batch_predictions)
            
            # Get references
            batch_references = [item["target_text"] for item in batch]
            references.extend(batch_references)
    
    # Compute metrics
    bleu_score = compute_bleu_score(predictions, references)
    chrf_score = compute_chrf_score(predictions, references)
    
    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "predictions": predictions,
        "references": references
    }


def evaluate_multiple_directions(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    test_data: Dict[str, pd.DataFrame],
    language_pairs: Dict[str, tuple],
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on multiple translation directions
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_data: Dictionary of test datasets
        language_pairs: Dictionary mapping dataset names to (source_lang, target_lang) tuples
        device: Device to use
        
    Returns:
        Dictionary of metrics for each direction
    """
    results = {}
    
    for dataset_name, (source_lang, target_lang) in language_pairs.items():
        if dataset_name not in test_data:
            continue
        
        print(f"Evaluating {source_lang} -> {target_lang}...")
        
        # Prepare test dataset
        df = test_data[dataset_name]
        # This is a simplified version - you'll need to adapt based on your test data format
        
        # For now, return placeholder
        results[f"{source_lang}->{target_lang}"] = {
            "bleu": 0.0,
            "chrf": 0.0
        }
    
    return results


def compute_final_score(results: Dict[str, Dict[str, float]]) -> float:
    """
    Compute final competition score according to MMLoSo evaluation formula
    
    Formula:
    0.6 * (0.6 * BLEU_avg + 0.4 * BLEU_reverse_avg)
    + 0.4 * (0.6 * chrF_avg + 0.4 * chrF_reverse_avg)
    
    Args:
        results: Dictionary of results for each direction
        
    Returns:
        Final score
    """
    # Separate forward and reverse directions
    forward_directions = [
        "Hindi->Bhili", "Hindi->Mundari", "Hindi->Gondi", "English->Santali"
    ]
    reverse_directions = [
        "Bhili->Hindi", "Mundari->Hindi", "Gondi->Hindi", "Santali->English"
    ]
    
    # Compute averages
    forward_bleu = np.mean([results.get(d, {}).get("bleu", 0.0) for d in forward_directions])
    reverse_bleu = np.mean([results.get(d, {}).get("bleu", 0.0) for d in reverse_directions])
    
    forward_chrf = np.mean([results.get(d, {}).get("chrf", 0.0) for d in forward_directions])
    reverse_chrf = np.mean([results.get(d, {}).get("chrf", 0.0) for d in reverse_directions])
    
    # Compute final score
    bleu_component = 0.6 * forward_bleu + 0.4 * reverse_bleu
    chrf_component = 0.6 * forward_chrf + 0.4 * reverse_chrf
    
    final_score = 0.6 * bleu_component + 0.4 * chrf_component
    
    return final_score

