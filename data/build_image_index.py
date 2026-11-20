"""
Builds a FAISS index over CLIP embeddings of every landmark image, so we can
do reverse-image search later: given a query photo, find the most visually
similar landmarks in our knowledge base.

Run AFTER scrape_wikipedia.py:
    python data/build_image_index.py

Output:
    data/image_index.faiss
    data/image_index_meta.json   -> maps FAISS row -> {title, extract, url, image_path}
"""

import json
import os

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "knowledge_base.json")
INDEX_PATH = os.path.join(HERE, "image_index.faiss")
META_PATH = os.path.join(HERE, "image_index_meta.json")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"  # small, fast, CPU-friendly CLIP checkpoint


def main():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    entries_with_images = [e for e in kb if e.get("image_path")]
    print(f"{len(entries_with_images)} / {len(kb)} entries have images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ({MODEL_NAME}/{PRETRAINED}) on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.to(device).eval()

    vectors = []
    kept_meta = []
    with torch.no_grad():
        for i, entry in enumerate(entries_with_images, 1):
            img_path = os.path.join(HERE, entry["image_path"])
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [skip] {entry['title']} -> {e}")
                continue
            tensor = preprocess(image).unsqueeze(0).to(device)
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize for cosine sim
            vectors.append(feat.cpu().numpy()[0])
            kept_meta.append(entry)
            if i % 10 == 0:
                print(f"  Embedded {i}/{len(entries_with_images)}")

    if not vectors:
        raise RuntimeError(
            "No images were successfully embedded, so there's nothing to index. "
            "This means scrape_wikipedia.py did not download any images - re-run it "
            "and check the [warn] lines it prints for each landmark (likely an HTTP "
            "403 from upload.wikimedia.org, or the images/ folder is empty)."
        )

    vectors = np.array(vectors).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(kept_meta, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS image index ({index.ntotal} vectors, dim={dim}) -> {INDEX_PATH}")
    print(f"Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
