import logging
import numpy as np
import json
from typing import List
from sentence_transformers import SentenceTransformer

ALL_LABELS = [
    "Title","Section-header","Text","List-item","Formula",
    "Picture","Table","Caption","Footnote","Page-header","Page-footer"
]

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

def save_config(CONFIG_PATH, **kwargs):
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    config.update(kwargs)
    CONFIG_PATH.write_text(json.dumps(config))

def load_config(CONFIG_PATH):
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}

class Embedder:
    def __init__(self, model_name: str, max_len: int = 256):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_len = max_len

    def encode(self, texts: str):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    def encode_label_texts(self, label_texts: List[List[str]]) -> np.ndarray:
        n_chunks, n_labels, dim = len(label_texts), len(ALL_LABELS), self.model.get_sentence_embedding_dimension()
        E = np.zeros((n_chunks, n_labels, dim), dtype=np.float32)
        for li in range(n_labels):
            texts = [t[li] for t in label_texts]
            embs = self.encode(texts)
            E[:, li, :] = embs
        return E