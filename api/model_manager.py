import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

class ModelManager:
    def __init__(self):
        self.model_cache_dir = Path("model_cache")
        self.model_cache_dir.mkdir(exist_ok=True)
        self.current_model = None
        self.current_model_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name):
        """Load a model by name."""
        if self.current_model_name == model_name:
            return self.current_model

        if model_name == "mock":
            self.current_model = MockBackend()
        elif model_name == "gpt2":
            self.current_model = GPT2Backend("gpt2", self.model_cache_dir)
        elif model_name == "bert":
            self.current_model = BERTBackend("bert-base-uncased", self.model_cache_dir)
        elif model_name == "distilbert":
            self.current_model = BERTBackend("distilbert-base-uncased", self.model_cache_dir)
        elif model_name == "sentence":
            self.current_model = SentenceTransformerBackend("all-MiniLM-L6-v2", self.model_cache_dir)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.current_model_name = model_name
        return self.current_model

class MockBackend:
    """Mock backend for quick testing."""
    def encode(self, words):
        np.random.seed(42)
        embeddings = np.random.rand(len(words), 16).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def attention(self, text, words):
        np.random.seed(42)
        n = len(words)
        A = np.random.rand(n, n).astype("float32")
        np.fill_diagonal(A, 0)
        row_sums = A.sum(axis=1, keepdims=True)
        return A / (row_sums + 1e-8)

class GPT2Backend:
    """GPT-2 based backend for embeddings and attention."""
    def __init__(self, model_name="gpt2", cache_dir=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = GPT2Model.from_pretrained(model_name, output_attentions=True, cache_dir=cache_dir)
        self.model.eval()

    def encode(self, words):
        embeddings = []
        with torch.no_grad():
            for word in words:
                inputs = self.tokenizer(word, return_tensors="pt")
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(emb)
        return np.array(embeddings)

    def attention(self, text, words):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        attn = torch.stack(outputs.attentions).mean(dim=(0, 1))[0].numpy()
        n = len(words)
        return attn[:n, :n]

class BERTBackend:
    """BERT based backend for embeddings and attention."""
    def __init__(self, model_name, cache_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True, cache_dir=cache_dir)
        self.model.eval()

    def encode(self, words):
        embeddings = []
        with torch.no_grad():
            for word in words:
                inputs = self.tokenizer(word, return_tensors="pt")
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(emb)
        return np.array(embeddings)

    def attention(self, text, words):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        attn = torch.stack(outputs.attentions).mean(dim=(0, 1))[0].numpy()
        n = len(words)
        return attn[:n, :n]

class SentenceTransformerBackend:
    """Sentence-BERT backend for high-quality embeddings."""
    def __init__(self, model_name, cache_dir=None):
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        # For attention, we'll use a lightweight BERT model
        self.attn_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=cache_dir)
        self.attn_model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True, cache_dir=cache_dir)
        self.attn_model.eval()

    def encode(self, words):
        return self.model.encode(words)

    def attention(self, text, words):
        inputs = self.attn_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.attn_model(**inputs)
        attn = torch.stack(outputs.attentions).mean(dim=(0, 1))[0].numpy()
        n = len(words)
        return attn[:n, :n]