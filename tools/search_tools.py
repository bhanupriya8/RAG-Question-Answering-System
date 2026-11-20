"""
The two search tools the agent can call: text_search and image_search.

Both tools track which results have already been shown to the agent
*for the current question* (a "seen set"), and skip those on repeat calls -
this is the de-duplication idea from the ProMSA paper: re-searching should
surface NEW evidence, not the same top-k every time.

Usage:
    tools = SearchTools()
    tools.reset()  # call once per new question, to clear the seen-set

    result = tools.text_search("when was the Eiffel Tower built")
    result = tools.image_search("path/to/query_image.jpg")
"""

import json
import os

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")

TEXT_INDEX_PATH = os.path.join(DATA_DIR, "text_index.faiss")
TEXT_META_PATH = os.path.join(DATA_DIR, "text_index_meta.json")
IMAGE_INDEX_PATH = os.path.join(DATA_DIR, "image_index.faiss")
IMAGE_META_PATH = os.path.join(DATA_DIR, "image_index_meta.json")

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


class SearchTools:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

        print("Loading text search index + model...")
        self.text_index = faiss.read_index(TEXT_INDEX_PATH)
        with open(TEXT_META_PATH, "r", encoding="utf-8") as f:
            self.text_meta = json.load(f)
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME)

        print("Loading image search index + model...")
        self.image_index = faiss.read_index(IMAGE_INDEX_PATH)
        with open(IMAGE_META_PATH, "r", encoding="utf-8") as f:
            self.image_meta = json.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        self.clip_model.to(device).eval()

        # de-duplication trackers, reset per question
        self._seen_text_titles = set()
        self._seen_image_titles = set()

    def reset(self):
        """Call this once per new question to clear the de-dup memory."""
        self._seen_text_titles = set()
        self._seen_image_titles = set()

    def text_search(self, query: str):
        query_vec = self.text_model.encode([query], normalize_embeddings=True).astype("float32")
        # over-fetch so we have room to skip already-seen entries
        scores, idxs = self.text_index.search(query_vec, self.top_k + len(self._seen_text_titles))

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            entry = self.text_meta[idx]
            if entry["title"] in self._seen_text_titles:
                continue
            results.append({"title": entry["title"], "extract": entry["extract"], "score": float(score)})
            self._seen_text_titles.add(entry["title"])
            if len(results) >= self.top_k:
                break

        if not results:
            return {"results": [], "note": "No new text results (all top matches already seen)."}
        return {"results": results}

    def image_search(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"results": [], "note": f"Could not open image: {e}"}

        with torch.no_grad():
            tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            feat = self.clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            query_vec = feat.cpu().numpy().astype("float32")

        scores, idxs = self.image_index.search(query_vec, self.top_k + len(self._seen_image_titles))

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            entry = self.image_meta[idx]
            if entry["title"] in self._seen_image_titles:
                continue
            results.append({"title": entry["title"], "extract": entry["extract"], "score": float(score)})
            self._seen_image_titles.add(entry["title"])
            if len(results) >= self.top_k:
                break

        if not results:
            return {"results": [], "note": "No new image results (all top matches already seen)."}
        return {"results": results}
