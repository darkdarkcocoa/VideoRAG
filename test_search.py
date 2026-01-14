"""Quick search test"""
import numpy as np
import torch
import open_clip

EMBEDDINGS_FILE = "D:/X/VideoRAG/output/embeddings.npz"

# Load embeddings
data = np.load(EMBEDDINGS_FILE)
embeddings = data['embeddings']
timestamps = data['timestamps']
print(f"Loaded {len(timestamps)} frame embeddings")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def search(query, top_k=5):
    text_tokens = tokenizer([query])
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy().flatten()
    
    similarities = embeddings @ text_emb
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nQuery: '{query}'")
    print("-" * 40)
    for i, idx in enumerate(top_indices, 1):
        ts = timestamps[idx]
        sim = similarities[idx]
        print(f"  {i}. {int(ts//60):02d}:{int(ts%60):02d} (score: {sim:.3f})")

# Test searches
test_queries = [
    "a person",
    "text on screen",
    "outdoor scene",
    "dark scene",
    "bright colorful scene"
]

print("\n" + "=" * 50)
print("SEARCH TEST")
print("=" * 50)

for q in test_queries:
    search(q)
