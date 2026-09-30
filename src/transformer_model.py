import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransformerSentimentClassifier:
    """
    Transformer-based sentiment classifier for Aspect-Based Sentiment Analysis.
    Takes a sequence formatted as [CLS] text [SEP] aspect [SEP] and predicts sentiment.
    """
    def __init__(self, model_path="models/bert_absa", base_model="bert-base-uncased", num_labels=3):
        # Resolve relative path using the project root
        if not os.path.isabs(model_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = os.path.join(project_root, model_path)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = {0: "negative", 1: "neutral", 2: "positive"}
        
        # The production system MUST use a fine-tuned model.
        if not os.path.exists(model_path):
            print(f"\n[ERROR] Fine-tuned BERT model not found at {model_path}.")
            print(f"[ERROR] Please train the model before using ABSA inference.\n")
            self.model = None
            self.tokenizer = None
        else:
            print(f"[Transformer] Loading fine-tuned model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
                
            self.model.to(self.device)
            self.model.eval()

    def predict(self, text: str, aspect: str = None) -> dict:
        """
        Predict sentiment for a given text and aspect.
        """
        if self.model is None or self.tokenizer is None:
            # Raise an explicit exception instead of mock prediction for academic integrity
            raise RuntimeError("Fine-tuned BERT model not found. Please train the model before using ABSA inference.")

        # If no aspect is provided, just predict on the text
        if aspect is None:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
        else:
            # Pair text and aspect for aspect-based sentiment
            inputs = self.tokenizer(
                text,
                text_pair=aspect,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=-1)
            
        sentiment = self.labels[predicted_class.item()]
        
        return {
            "sentiment": sentiment,
            "confidence": confidence.item()
        }
