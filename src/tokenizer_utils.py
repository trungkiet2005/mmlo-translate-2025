"""
Tokenizer extension utilities for adding new vocabulary tokens
"""
import json
import re
from collections import Counter
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer
import torch


class TokenizerExtender:
    """
    Safely extend SentencePiece tokenizer with new vocabulary tokens
    without destroying existing vocabulary.
    """
    
    def __init__(
        self,
        base_tokenizer: AutoTokenizer,
        new_tokens_count: int = 200,
        min_frequency: int = 2
    ):
        """
        Initialize tokenizer extender
        
        Args:
            base_tokenizer: Base HuggingFace tokenizer
            new_tokens_count: Number of new tokens to add
            min_frequency: Minimum frequency to consider a token
        """
        self.base_tokenizer = base_tokenizer
        self.new_tokens_count = new_tokens_count
        self.min_frequency = min_frequency
        self.new_tokens: List[str] = []
        
    def extract_tokens_from_text(
        self,
        texts: List[str],
        script_patterns: Optional[Dict[str, str]] = None
    ) -> Counter:
        """
        Extract potential tokens from text corpus
        
        Args:
            texts: List of text strings
            script_patterns: Optional regex patterns for specific scripts
            
        Returns:
            Counter of token candidates with frequencies
        """
        if script_patterns is None:
            # Common patterns for Indian languages
            script_patterns = {
                "devanagari": r"[\u0900-\u097F]+",
                "ol_chiki": r"[\u1C50-\u1C7F]+",
                "latin": r"[a-zA-Z]+",
                "numbers": r"\d+",
                "punctuation": r"[^\w\s]"
            }
        
        token_candidates = Counter()
        
        for text in texts:
            # Extract tokens by script type
            for script_name, pattern in script_patterns.items():
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) >= 2:  # Minimum token length
                        token_candidates[match] += 1
            
            # Also extract word boundaries (space-separated)
            words = text.split()
            for word in words:
                if len(word) >= 2:
                    token_candidates[word] += 1
        
        return token_candidates
    
    def filter_existing_tokens(
        self,
        candidates: Counter,
        existing_vocab: Optional[Set[str]] = None
    ) -> Counter:
        """
        Filter out tokens that already exist in vocabulary
        
        Args:
            candidates: Counter of token candidates
            existing_vocab: Set of existing vocabulary tokens
            
        Returns:
            Filtered Counter
        """
        if existing_vocab is None:
            # Get existing vocabulary from tokenizer
            existing_vocab = set(self.base_tokenizer.get_vocab().keys())
        
        # Filter candidates
        filtered = Counter()
        for token, freq in candidates.items():
            if token not in existing_vocab and freq >= self.min_frequency:
                # Check if token can be tokenized (should be unknown)
                if self.base_tokenizer.tokenize(token, add_special_tokens=False) != [token]:
                    # This token is split into multiple subwords, good candidate
                    filtered[token] = freq
        
        return filtered
    
    def select_new_tokens(
        self,
        candidates: Counter,
        priority_scripts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select top new tokens to add
        
        Args:
            candidates: Filtered token candidates
            priority_scripts: Scripts to prioritize (e.g., ["ol_chiki", "devanagari"])
            
        Returns:
            List of selected new tokens
        """
        if priority_scripts is None:
            priority_scripts = []
        
        # Separate tokens by script
        script_tokens = {script: [] for script in priority_scripts}
        other_tokens = []
        
        for token, freq in candidates.most_common():
            categorized = False
            for script in priority_scripts:
                if self._matches_script(token, script):
                    script_tokens[script].append((token, freq))
                    categorized = True
                    break
            
            if not categorized:
                other_tokens.append((token, freq))
        
        # Select tokens: prioritize rare scripts, then frequency
        selected = []
        tokens_per_script = self.new_tokens_count // (len(priority_scripts) + 1)
        
        # Add tokens from priority scripts first
        for script in priority_scripts:
            script_selected = [
                token for token, _ in script_tokens[script][:tokens_per_script]
            ]
            selected.extend(script_selected)
        
        # Fill remaining with high-frequency tokens
        remaining = self.new_tokens_count - len(selected)
        other_selected = [token for token, _ in other_tokens[:remaining]]
        selected.extend(other_selected)
        
        # Remove duplicates and limit
        selected = list(dict.fromkeys(selected))[:self.new_tokens_count]
        
        return selected
    
    def _matches_script(self, text: str, script: str) -> bool:
        """Check if text matches a specific script"""
        script_ranges = {
            "devanagari": (0x0900, 0x097F),
            "ol_chiki": (0x1C50, 0x1C7F),
            "latin": (0x0000, 0x007F)
        }
        
        if script not in script_ranges:
            return False
        
        start, end = script_ranges[script]
        return any(start <= ord(char) <= end for char in text)
    
    def extend_tokenizer(
        self,
        training_texts: List[str],
        new_tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> AutoTokenizer:
        """
        Extend tokenizer with new tokens
        
        Args:
            training_texts: Training texts to analyze for new tokens
            new_tokens: Optional pre-selected tokens (if None, will extract)
            save_path: Optional path to save extended tokenizer
            
        Returns:
            Extended tokenizer
        """
        if new_tokens is None:
            # Extract and select new tokens
            print("Extracting token candidates from training data...")
            candidates = self.extract_tokens_from_text(training_texts)
            print(f"Found {len(candidates)} candidate tokens")
            
            print("Filtering existing tokens...")
            filtered = self.filter_existing_tokens(candidates)
            print(f"Found {len(filtered)} new token candidates")
            
            # Priority: Ol Chiki (Santali), then Devanagari variants
            priority_scripts = ["ol_chiki", "devanagari"]
            print("Selecting new tokens...")
            new_tokens = self.select_new_tokens(filtered, priority_scripts)
            print(f"Selected {len(new_tokens)} new tokens")
        
        self.new_tokens = new_tokens
        
        # Add new tokens to tokenizer
        print(f"Adding {len(new_tokens)} new tokens to tokenizer...")
        
        # Use add_tokens method (preserves existing vocab)
        num_added = self.base_tokenizer.add_tokens(new_tokens, special_tokens=False)
        print(f"Successfully added {num_added} new tokens")
        
        # Resize token embeddings (will be handled in model initialization)
        print("Tokenizer extension complete!")
        
        if save_path:
            self.save_tokenizer(save_path)
        
        return self.base_tokenizer
    
    def save_tokenizer(self, save_path: str):
        """Save extended tokenizer"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.base_tokenizer.save_pretrained(save_path)
        
        # Also save new tokens list
        tokens_path = Path(save_path) / "new_tokens.json"
        with open(tokens_path, "w", encoding="utf-8") as f:
            json.dump(self.new_tokens, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {save_path}")
        print(f"New tokens list saved to {tokens_path}")
    
    def load_new_tokens(self, tokens_path: str) -> List[str]:
        """Load new tokens from JSON file"""
        with open(tokens_path, "r", encoding="utf-8") as f:
            return json.load(f)


def initialize_new_token_embeddings(
    model,
    tokenizer: AutoTokenizer,
    new_tokens: List[str],
    initialization_strategy: str = "average"
) -> torch.nn.Embedding:
    """
    Initialize embeddings for new tokens
    
    Args:
        model: Model with embedding layer
        tokenizer: Extended tokenizer
        new_tokens: List of new tokens
        initialization_strategy: How to initialize ("average", "random", "zero")
        
    Returns:
        Updated embedding layer
    """
    # Get embedding layer
    if hasattr(model, "model"):  # For PEFT-wrapped models
        embed_layer = model.model.get_input_embeddings()
    else:
        embed_layer = model.get_input_embeddings()
    
    # Get current embedding size
    old_vocab_size = embed_layer.weight.shape[0]
    new_vocab_size = len(tokenizer)
    embedding_dim = embed_layer.weight.shape[1]
    
    if new_vocab_size <= old_vocab_size:
        print("No new tokens to initialize")
        return embed_layer
    
    # Resize embeddings
    embed_layer.resize_token_embeddings(new_vocab_size)
    
    # Initialize new token embeddings
    new_token_ids = []
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id >= old_vocab_size:
            new_token_ids.append(token_id)
    
    if initialization_strategy == "average":
        # Initialize with average of existing embeddings
        with torch.no_grad():
            existing_embeddings = embed_layer.weight[:old_vocab_size]
            avg_embedding = existing_embeddings.mean(dim=0)
            
            for token_id in new_token_ids:
                embed_layer.weight[token_id] = avg_embedding.clone()
    
    elif initialization_strategy == "random":
        # Initialize with random values (small variance)
        with torch.no_grad():
            for token_id in new_token_ids:
                embed_layer.weight[token_id].normal_(mean=0, std=0.02)
    
    elif initialization_strategy == "zero":
        # Initialize with zeros
        with torch.no_grad():
            for token_id in new_token_ids:
                embed_layer.weight[token_id].zero_()
    
    else:
        raise ValueError(f"Unknown initialization strategy: {initialization_strategy}")
    
    print(f"Initialized {len(new_token_ids)} new token embeddings using '{initialization_strategy}' strategy")
    
    return embed_layer

