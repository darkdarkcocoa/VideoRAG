"""SceneSearch - Gradio Web UI (Hybrid Search with BLIP Captions)"""
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
image_embeddings = None
text_embeddings = None
frames = None
model = None
tokenizer = None
device = None
use_hybrid = False  # Will be set based on available data

def load_resources():
    """Load embeddings, metadata, and CLIP model"""
    global image_embeddings, text_embeddings, frames, model, tokenizer, device, use_hybrid

    print("[*] Loading SceneSearch...")

    # Load embeddings
    data = np.load(EMBEDDINGS_FILE)
    
    # Check if hybrid embeddings exist (new format)
    if 'image_embeddings' in data:
        image_embeddings = data['image_embeddings']
        text_embeddings = data['text_embeddings']
        use_hybrid = True
        print(f"[+] Loaded hybrid embeddings (image + text)")
    else:
        # Fallback to old format
        image_embeddings = data['embeddings']
        text_embeddings = None
        use_hybrid = False
        print(f"[+] Loaded image embeddings only (legacy mode)")

    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    frames = metadata['frames']

    print(f"[+] Loaded {len(frames)} frames")
    
    # Check if captions exist
    if 'caption' in frames[0]:
        print(f"[+] Captions available")
    else:
        print(f"[!] No captions found (run generate_embeddings.py to add)")

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    print("[+] CLIP model loaded")
    print("[!] Ready!\n")

def search(query: str, top_k: int = 8, image_weight: float = 0.6):
    """Hybrid search for frames matching the query"""
    if not query.strip():
        return [], ""

    # 1. Encode query
    t0 = time.perf_counter()
    text_tokens = tokenizer([query])
    with torch.no_grad():
        query_emb = model.encode_text(text_tokens.to(device))
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy().flatten()
    encode_time = time.perf_counter() - t0

    # 2. Calculate similarities
    t1 = time.perf_counter()
    
    # Image similarity
    img_similarities = image_embeddings @ query_emb
    
    if use_hybrid and text_embeddings is not None:
        # Text (caption) similarity
        txt_similarities = text_embeddings @ query_emb
        
        # Dynamic weight based on query length
        word_count = len(query.split())
        if word_count <= 2:
            w_img, w_txt = 0.7, 0.3  # Short query → visual focus
        else:
            w_img, w_txt = image_weight, 1 - image_weight  # Use slider value
        
        # Combined score
        similarities = img_similarities * w_img + txt_similarities * w_txt
        search_mode = f"Hybrid (img:{w_img:.1f}, txt:{w_txt:.1f})"
    else:
        similarities = img_similarities
        search_mode = "Image only"
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    search_time = time.perf_counter() - t1

    # Build results
    results = []
    for idx in top_indices:
        frame = frames[idx]
        score = similarities[idx]
        image_path = FRAMES_DIR / frame['filename']

        if image_path.exists():
            # Caption with time, score, and BLIP caption
            caption_text = frame.get('caption', '')
            if caption_text:
                caption = f"⏱️ {frame['time_str']} | Score: {score:.3f}\n📝 {caption_text}"
            else:
                caption = f"⏱️ {frame['time_str']} | Score: {score:.3f}"
            results.append((str(image_path), caption))

    time_info = f"🔎 {len(frames):,}개 프레임 | {search_mode} | 인코딩: {encode_time*1000:.1f}ms | 검색: {search_time*1000:.2f}ms"

    return results, time_info

def create_app():
    """Create Gradio app"""

    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .search-box {
        font-size: 18px !important;
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

        # Advanced options (collapsible)
        with gr.Accordion("🔧 고급 설정", open=False):
            image_weight_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.6,
                step=0.1,
                label="이미지 가중치 (낮을수록 캡션 중심 검색)",
                info="짧은 검색어(1-2단어)는 자동으로 이미지 중심(0.7)으로 설정됩니다"
            )

        search_btn = gr.Button("🔍 검색", variant="primary", size="lg")

        # Search info
        search_info = gr.Markdown("")

        # Results gallery
        gallery = gr.Gallery(
            label="검색 결과",
            columns=4,
            rows=2,
            height="auto",
            object_fit="cover",
            show_label=True
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
                ["a man giving presentation"],
                ["explosion or bright light"],
                ["woman with long hair"],
            ],
            inputs=query_input,
            label=""
        )

        # Event handlers
        search_btn.click(
            fn=search,
            inputs=[query_input, top_k_slider, image_weight_slider],
            outputs=[gallery, search_info]
        )

        query_input.submit(
            fn=search,
            inputs=[query_input, top_k_slider, image_weight_slider],
            outputs=[gallery, search_info]
        )

        # Footer
        mode_text = "Hybrid (CLIP + BLIP)" if use_hybrid else "CLIP Only"
        gr.Markdown(
            f"""
            ---
            *{mode_text} | {len(frames) if frames else 0:,}개 프레임*
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
