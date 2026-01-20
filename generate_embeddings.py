"""Generate CLIP embeddings + BLIP captions for extracted frames"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import open_clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
FRAMES_DIR = OUTPUT_DIR / "frames"
OUTPUT_FILE = OUTPUT_DIR / "embeddings.npz"

def main():
    print("=" * 60)
    print("CLIP Embedding + BLIP Caption Generator")
    print("=" * 60)

    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    frames = metadata['frames']
    print(f"\nTotal frames to process: {len(frames)}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # Load CLIP model
    # ============================================================
    print("\nLoading CLIP model...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("CLIP model loaded!")

    # ============================================================
    # Load BLIP model
    # ============================================================
    print("\nLoading BLIP model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    blip_model.eval()
    print("BLIP model loaded!")

    # ============================================================
    # Generate embeddings and captions
    # ============================================================
    print("\n" + "-" * 60)
    print("Generating embeddings and captions...")
    start_time = time.time()

    image_embeddings = []
    text_embeddings = []
    captions = []
    timestamps = []

    for i, frame_info in enumerate(frames):
        frame_path = FRAMES_DIR / frame_info['filename']

        # Load image
        image = Image.open(frame_path).convert('RGB')

        # ----------------------------------------------------
        # 1. CLIP image embedding
        # ----------------------------------------------------
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = clip_model.encode_image(image_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        image_embeddings.append(img_emb.cpu().numpy().flatten())

        # ----------------------------------------------------
        # 2. BLIP caption generation
        # ----------------------------------------------------
        with torch.no_grad():
            inputs = blip_processor(image, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_new_tokens=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)

        # ----------------------------------------------------
        # 3. CLIP text embedding (from caption)
        # ----------------------------------------------------
        text_tokens = tokenizer([caption])
        with torch.no_grad():
            txt_emb = clip_model.encode_text(text_tokens.to(device))
            txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
        text_embeddings.append(txt_emb.cpu().numpy().flatten())

        timestamps.append(frame_info['timestamp'])

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            remaining = (len(frames) - i - 1) / speed
            print(f"  [{i+1}/{len(frames)}] {speed:.1f} frames/sec, ~{remaining:.0f}s remaining")
            print(f"      Caption: {caption[:60]}...")

    elapsed = time.time() - start_time
    print(f"\nDone! {elapsed:.1f}s total ({len(frames)/elapsed:.1f} frames/sec)")

    # ============================================================
    # Save embeddings
    # ============================================================
    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)
    timestamps = np.array(timestamps)

    print(f"\nImage embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")

    np.savez(
        OUTPUT_FILE,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        timestamps=timestamps
    )
    print(f"Saved embeddings to {OUTPUT_FILE}")

    # ============================================================
    # Update metadata with captions
    # ============================================================
    for i, caption in enumerate(captions):
        metadata['frames'][i]['caption'] = caption

    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Updated metadata with captions: {METADATA_FILE}")

    # ============================================================
    # Show sample results
    # ============================================================
    print("\n" + "=" * 60)
    print("Sample Captions:")
    print("=" * 60)
    for i in [0, len(frames)//2, len(frames)-1]:
        print(f"  Frame {i} [{frames[i]['time_str']}]: {captions[i]}")

if __name__ == "__main__":
    main()
