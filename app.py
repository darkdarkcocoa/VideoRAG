"""SceneSearch - Gradio Web UI"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
import numpy as np
import torch
import open_clip
import json
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npz"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
FRAMES_DIR = OUTPUT_DIR / "frames"

# Global variables
embeddings = None
frames = None
model = None
tokenizer = None
device = None

def load_resources():
    """Load embeddings, metadata, and CLIP model"""
    global embeddings, frames, model, tokenizer, device

    print("[*] Loading SceneSearch...")

    # Load embeddings
    data = np.load(EMBEDDINGS_FILE)
    embeddings = data['embeddings']

    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    frames = metadata['frames']

    print(f"[+] Loaded {len(frames)} frame embeddings")

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    print("[+] CLIP model loaded")
    print("[!] Ready!\n")

def search(query: str, top_k: int = 8):
    """Search for frames matching the query"""
    if not query.strip():
        return [], ""

    # 1. Encode query (CLIP text encoding)
    t0 = time.perf_counter()
    text_tokens = tokenizer([query])
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy().flatten()
    encode_time = time.perf_counter() - t0

    # 2. Vector search (cosine similarity)
    t1 = time.perf_counter()
    similarities = embeddings @ text_emb
    top_indices = np.argsort(similarities)[::-1][:top_k]
    search_time = time.perf_counter() - t1

    # Build results
    results = []
    for idx in top_indices:
        frame = frames[idx]
        score = similarities[idx]
        image_path = FRAMES_DIR / frame['filename']

        if image_path.exists():
            # Caption with time and score
            caption = f"⏱️ {frame['time_str']} | Score: {score:.3f}"
            results.append((str(image_path), caption))

    time_info = f"🔎 {len(frames):,}개 프레임 | 인코딩: {encode_time*1000:.1f}ms | 검색: {search_time*1000:.2f}ms"

    return results, time_info

def create_app():
    """Create Gradio app"""

    # Custom CSS for clean, modern look
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .search-box {
        font-size: 18px !important;
    }
    .gallery-item {
        border-radius: 12px !important;
    }
    footer {
        display: none !important;
    }
    """

    with gr.Blocks(
        title="SceneSearch",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css=custom_css
    ) as app:
        # Header
        gr.Markdown(
            """
            # 🎬 SceneSearch
            ### 자연어로 영상 속 장면을 검색하세요
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="검색어",
                    placeholder="예: laboratory, Johnny Depp, computer screen, sunset...",
                    lines=1,
                    elem_classes=["search-box"]
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=4,
                    maximum=20,
                    value=8,
                    step=4,
                    label="결과 개수"
                )

        search_btn = gr.Button("🔍 검색", variant="primary", size="lg")

        # Search time info
        search_info = gr.Markdown("")

        # Results gallery
        gallery = gr.Gallery(
            label="검색 결과",
            columns=4,
            rows=2,
            height="auto",
            object_fit="cover",
            show_label=True,
            elem_classes=["gallery-item"]
        )

        # Example queries
        gr.Markdown("### 💡 검색 예시")
        gr.Examples(
            examples=[
                ["Johnny Depp face"],
                ["computer screen with code"],
                ["laboratory equipment"],
                ["outdoor garden scene"],
                ["two people talking"],
                ["dark moody scene"],
                ["explosion or bright light"],
                ["woman with long hair"],
            ],
            inputs=query_input,
            label=""
        )

        # Event handlers
        search_btn.click(
            fn=search,
            inputs=[query_input, top_k_slider],
            outputs=[gallery, search_info]
        )

        query_input.submit(
            fn=search,
            inputs=[query_input, top_k_slider],
            outputs=[gallery, search_info]
        )

        # Footer info
        gr.Markdown(
            """
            ---
            *CLIP ViT-B/32 모델 사용 | Scene Detection으로 추출된 1,125개 프레임*
            """
        )

    return app

if __name__ == "__main__":
    load_resources()
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
