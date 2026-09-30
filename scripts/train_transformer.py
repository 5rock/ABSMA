import os
import sys
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import generate_absa_training_data

class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, texts, aspects, labels, tokenizer, max_length=64):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            text_pair=self.aspects[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def validate_dataset(df):
    """
    Validates dataset before training to ensure data quality and avoid pseudo-labeling errors.
    Stops execution if the dataset is invalid.
    """
    print("\n" + "="*50)
    print(" DATASET VALIDATION REPORT")
    print("="*50)
    
    total_samples = len(df)
    print(f"Total samples: {total_samples}")
    
    if total_samples == 0:
        raise ValueError("Dataset is empty.")
        
    # Check required columns
    required_cols = {'review', 'aspect', 'category', 'sentiment'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found: {df.columns}. Expected: {required_cols}")
        
    # Check missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Dataset contains {missing} missing values. Please clean data before training.")
        
    # Check sentiments
    valid_labels = {0, 1, 2}
    unique_labels = set(df['sentiment'].unique())
    if not unique_labels.issubset(valid_labels):
        raise ValueError(f"Invalid labels found: {unique_labels}. Expected only: {valid_labels}")
        
    print(f"Positive (2): {len(df[df['sentiment'] == 2])}")
    print(f"Negative (0): {len(df[df['sentiment'] == 0])}")
    print(f"Neutral (1):  {len(df[df['sentiment'] == 1])}")
    
    # Check categories
    print("\nCategories:")
    for cat, count in df['category'].value_counts().items():
        print(f"{cat}: {count}")
        
    # Check duplicates
    duplicate_rows = df.duplicated().sum()
    duplicate_pairs = df.duplicated(subset=['review', 'aspect']).sum()
    print(f"\nDuplicate rows: {duplicate_rows}")
    print(f"Duplicate review/aspect pairs: {duplicate_pairs}")
    
    if duplicate_pairs > 0:
        print("[WARNING] Found duplicate review/aspect pairs. This can cause data leakage. Removing duplicates...")
        df = df.drop_duplicates(subset=['review', 'aspect']).reset_index(drop=True)
        print(f"New total samples: {len(df)}")
        
    print("="*50 + "\n")
    return df

def main():
    print("=" * 60)
    print("   ABSMA TRANSFORMER FINE-TUNING (CPU-ONLY)")
    print("=" * 60)
    
    print(f"[System] PyTorch CUDA available: {torch.cuda.is_available()}")
    
    data_path = os.path.join(project_root, "data", "processed", "absa_training.csv")
    
    # Generate an appropriate size dataset if missing or too small
    if not os.path.exists(data_path) or len(pd.read_csv(data_path)) < 24:
        print(f"[INFO] Training data not found or too small at {data_path}.")
        print(f"[WARNING] Using generated development dataset for aspect-level sentiment.")
        print(f"[WARNING] This is strictly DEVELOPMENT DATA, not human-annotated ground truth.")
        df = generate_absa_training_data(output_path=data_path, sample_size=24)
    else:
        print(f"[INFO] Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
    # Map any string sentiments to int just in case
    if df['sentiment'].dtype == object:
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df['sentiment'] = df['sentiment'].map(label_map)
        
    df = validate_dataset(df)
    
    model_name = "bert-base-uncased"
    print(f"[INFO] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['sentiment'])
    
    print(f"[INFO] Train split: {len(train_df)} | Val split: {len(val_df)} | Test split: {len(test_df)}")
    
    test_path = os.path.join(project_root, "data", "processed", "absa_test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"[INFO] Saved held-out test set to {test_path}")
    
    # Tuning params for speed and accuracy on CPU
    max_length = 24
    batch_size = 16
    epochs = 40  # EXACTLY 40 as requested
    lr = 3e-5
    
    train_dataset = ABSADataset(train_df['review'].tolist(), train_df['aspect'].tolist(), train_df['sentiment'].tolist(), tokenizer, max_length)
    val_dataset = ABSADataset(val_df['review'].tolist(), val_df['aspect'].tolist(), val_df['sentiment'].tolist(), tokenizer, max_length)
    
    print(f"[INFO] Initializing model: {model_name} (Labels: 3)")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(project_root, "results", "transformer_absa"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        save_total_limit=1,
        use_cpu=True, 
        disable_tqdm=False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("\n[INFO] Starting EXACTLY 40-epoch training...")
    trainer.train()
    
    print("\n[INFO] Evaluating on validation set...")
    val_results = trainer.evaluate()
    print("Final Validation Results:", val_results)
    
    save_path = os.path.join(project_root, "models", "bert_absa")
    print(f"\n[INFO] Saving BEST model and tokenizer to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("[INFO] Training Complete!")
    print("\n[INFO] Running the final academic evaluation...")
    import evaluate_transformer
    evaluate_transformer.main()

if __name__ == "__main__":
    main()
