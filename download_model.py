import open_clip
import torch

print("=" * 50)
print("CLIP Model Download & Test")
print("=" * 50)

# GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
print("\nLoading model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer('ViT-B-32')

print("[OK] Model loaded!")

# Simple test
print("\n" + "=" * 50)
print("Text Embedding Test")
print("=" * 50)

test_texts = ["a dog", "a cat", "a car explosion"]
text_tokens = tokenizer(test_texts)

with torch.no_grad():
    text_features = model.encode_text(text_tokens.to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"\nInput texts: {test_texts}")
print(f"Embedding shape: {text_features.shape}")
print(f"Embedding dimension: {text_features.shape[1]}")

print("\n[OK] Test completed! Model is working.")
