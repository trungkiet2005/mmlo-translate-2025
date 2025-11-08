"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """Dataset for translation tasks"""
    
    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_lang: str,
        target_lang: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = str(self.source_texts[idx])
        target_text = str(self.target_texts[idx])
        
        from src.utils import get_language_code
        
        source_lang_code = get_language_code(self.source_lang)
        target_lang_code = get_language_code(self.target_lang)
        
        # For NLLB, we need to tokenize source and target separately
        # NLLB tokenizer requires src_lang and tgt_lang parameters
        try:
            # Try using the NLLB tokenizer's built-in method
            tokenized = self.tokenizer(
                source_text,
                text_target=target_text,
                src_lang=source_lang_code,
                tgt_lang=target_lang_code,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            labels = tokenized.get("labels", tokenized["input_ids"]).squeeze(0)
            
        except Exception as e:
            # Fallback: tokenize separately
            print(f"Warning: NLLB tokenization failed, using fallback: {e}")
            source_encoded = self.tokenizer(
                source_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            target_encoded = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = source_encoded["input_ids"].squeeze(0)
            attention_mask = source_encoded["attention_mask"].squeeze(0)
            labels = target_encoded["input_ids"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_text": source_text,
            "target_text": target_text,
            "source_lang": source_lang_code,
            "target_lang": target_lang_code
        }


class DataLoader:
    """Data loader for MMLoSo dataset"""
    
    def __init__(self, path_root: str):
        """
        Initialize data loader
        
        Args:
            path_root: Root path to dataset (local or Kaggle)
        """
        self.path_root = Path(path_root)
        
        # Dataset file mapping
        self.dataset_files = {
            "bhili": "bhili-train.csv",
            "gondi": "gondi-train.csv",
            "mundari": "mundari-train.csv",
            "santali": "santali-train.csv"
        }
        
        # Language pair mapping
        self.language_pairs = {
            "bhili": ("Hindi", "Bhili"),
            "gondi": ("Hindi", "Gondi"),
            "mundari": ("Hindi", "Mundari"),
            "santali": ("English", "Santali")
        }
    
    def load_dataset(
        self,
        dataset_name: str,
        source_column: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load a dataset file
        
        Args:
            dataset_name: Name of dataset (bhili, gondi, mundari, santali)
            source_column: Source language column name
            target_column: Target language column name
            
        Returns:
            DataFrame with source and target columns
        """
        if dataset_name not in self.dataset_files:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        file_path = self.path_root / self.dataset_files[dataset_name]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path, encoding="utf-8")
        
        # Determine column names
        if source_column is None or target_column is None:
            source_lang, target_lang = self.language_pairs[dataset_name]
            source_column = source_lang
            target_column = target_lang
        
        # Verify columns exist
        if source_column not in df.columns:
            raise ValueError(f"Source column '{source_column}' not found in dataset")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Clean data
        df = df[[source_column, target_column]].copy()
        df = df.dropna()
        df = df[df[source_column].astype(str).str.strip() != ""]
        df = df[df[target_column].astype(str).str.strip() != ""]
        
        return df
    
    def load_all_datasets(
        self,
        combine: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets
        
        Args:
            combine: If True, combine all datasets into one
            
        Returns:
            Dictionary of datasets or single combined DataFrame
        """
        datasets = {}
        
        for dataset_name in self.dataset_files.keys():
            try:
                df = self.load_dataset(dataset_name)
                datasets[dataset_name] = df
                print(f"Loaded {dataset_name}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
        
        if combine:
            # Combine all datasets
            combined = []
            for dataset_name, df in datasets.items():
                source_lang, target_lang = self.language_pairs[dataset_name]
                df_renamed = df.rename(columns={
                    source_lang: "source",
                    target_lang: "target"
                })
                df_renamed["dataset"] = dataset_name
                combined.append(df_renamed)
            
            return pd.concat(combined, ignore_index=True)
        
        return datasets
    
    def prepare_datasets(
        self,
        dataset_names: List[str],
        tokenizer: AutoTokenizer,
        train_split: float = 0.95,
        val_split: float = 0.05,
        max_length: int = 512,
        seed: int = 42
    ) -> Tuple[TranslationDataset, TranslationDataset]:
        """
        Prepare train and validation datasets
        
        Args:
            dataset_names: List of dataset names to use
            tokenizer: Tokenizer for encoding
            train_split: Training split ratio
            val_split: Validation split ratio
            max_length: Maximum sequence length
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        all_source_texts = []
        all_target_texts = []
        all_source_langs = []
        all_target_langs = []
        
        for dataset_name in dataset_names:
            df = self.load_dataset(dataset_name)
            source_lang, target_lang = self.language_pairs[dataset_name]
            
            source_col = source_lang
            target_col = target_lang
            
            source_texts = df[source_col].tolist()
            target_texts = df[target_col].tolist()
            
            all_source_texts.extend(source_texts)
            all_target_texts.extend(target_texts)
            all_source_langs.extend([source_lang] * len(source_texts))
            all_target_langs.extend([target_lang] * len(target_texts))
        
        # Split into train/val
        (
            train_source, val_source,
            train_target, val_target,
            train_src_lang, val_src_lang,
            train_tgt_lang, val_tgt_lang
        ) = train_test_split(
            all_source_texts,
            all_target_texts,
            all_source_langs,
            all_target_langs,
            test_size=val_split,
            random_state=seed
        )
        
        # Create datasets - we need to handle language pairs properly
        # Since we have multiple language pairs, we'll create a list and combine them
        from torch.utils.data import ConcatDataset
        
        # Group by language pair
        lang_pair_datasets = {}
        for i, (src, tgt, src_lang, tgt_lang) in enumerate(zip(
            all_source_texts, all_target_texts, all_source_langs, all_target_langs
        )):
            pair_key = (src_lang, tgt_lang)
            if pair_key not in lang_pair_datasets:
                lang_pair_datasets[pair_key] = {"src": [], "tgt": []}
            lang_pair_datasets[pair_key]["src"].append(src)
            lang_pair_datasets[pair_key]["tgt"].append(tgt)
        
        # Split each language pair separately
        train_datasets = []
        val_datasets = []
        
        for (src_lang, tgt_lang), texts in lang_pair_datasets.items():
            src_texts = texts["src"]
            tgt_texts = texts["tgt"]
            
            # Split
            train_src, val_src, train_tgt, val_tgt = train_test_split(
                src_texts, tgt_texts, test_size=val_split, random_state=seed
            )
            
            # Create datasets
            train_ds = TranslationDataset(train_src, train_tgt, src_lang, tgt_lang, tokenizer, max_length)
            val_ds = TranslationDataset(val_src, val_tgt, src_lang, tgt_lang, tokenizer, max_length)
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            
            print(f"{src_lang}->{tgt_lang}: {len(train_ds)} train, {len(val_ds)} val")
        
        # Combine all datasets
        combined_train = ConcatDataset(train_datasets)
        combined_val = ConcatDataset(val_datasets)
        
        return combined_train, combined_val
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data"""
        test_path = self.path_root / "test.csv"
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        df = pd.read_csv(test_path, encoding="utf-8")
        return df


def prepare_data_for_tokenizer_extraction(
    data_loader: DataLoader,
    dataset_names: List[str]
) -> List[str]:
    """
    Prepare text data for tokenizer extension
    
    Args:
        data_loader: DataLoader instance
        dataset_names: List of dataset names to use
        
    Returns:
        List of all text strings
    """
    all_texts = []
    
    for dataset_name in dataset_names:
        df = data_loader.load_dataset(dataset_name)
        source_lang, target_lang = data_loader.language_pairs[dataset_name]
        
        # Add both source and target texts
        all_texts.extend(df[source_lang].tolist())
        all_texts.extend(df[target_lang].tolist())
    
    return all_texts

