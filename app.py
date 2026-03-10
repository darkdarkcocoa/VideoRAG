"""SceneSearch - Gradio Web UI with Video Player Integration"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
import numpy as np
import torch
import open_clip
import json
import time
import subprocess
import shutil
import re
import os
import threading
from pathlib import Path
from PIL import Image
import google.generativeai as genai

# Paths
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "video"
OUTPUT_DIR = BASE_DIR / "output"

# Gemini API Setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Global variables
image_embeddings = None
text_embeddings = None
frames = None
clip_model = None
tokenizer = None
device = None
use_hybrid = False
current_movie = None  # Currently loaded movie name
search_results_timestamps = []  # Store timestamps for gallery clicks
stop_event = threading.Event()  # Stop signal for processing
current_ffmpeg_proc = None  # Reference to running ffmpeg process

# ============================================================
# UTILS
# ============================================================
def contains_korean(text):
    """Check if text contains Korean characters"""
    return bool(re.search(r'[가-힣]', text))

def translate_to_english(query):
    """Translate Korean query to English using Gemini for CLIP search
    Returns: (translated_text, error_message or None)
    """
    try:
        prompt = f"""Translate the following Korean text to English.
The translation should be optimized for image search (CLIP model).
Return ONLY the English translation, nothing else.
Keep it concise and descriptive.

Korean: {query}
English:"""
        response = gemini_model.generate_content(prompt)
        translated = response.text.strip()
        print(f"[Gemini] '{query}' → '{translated}'")
        return translated, None
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            error_msg = "API 할당량 초과 (잠시 후 다시 시도)"
        elif "API_KEY" in error_msg:
            error_msg = "API 키가 유효하지 않음"
        else:
            error_msg = f"번역 실패: {error_msg[:50]}"
        print(f"[Gemini Error] {e}")
        return query, error_msg  # Fallback to original + error

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0

def get_movie_name(video_path):
    """Extract movie name from path (without extension)"""
    return Path(video_path).stem

def get_available_movies():
    """Get list of video files in video/ folder"""
    if not VIDEO_DIR.exists():
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        return []

    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm'}
    movies = []
    for f in VIDEO_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            movies.append(f.stem)
    return sorted(movies)

def get_movie_paths(movie_name):
    """Get all paths for a movie"""
    video_path = VIDEO_DIR / f"{movie_name}.mp4"
    # Try other extensions if mp4 doesn't exist
    if not video_path.exists():
        for ext in ['.avi', '.mkv', '.mov', '.wmv', '.webm']:
            alt_path = VIDEO_DIR / f"{movie_name}{ext}"
            if alt_path.exists():
                video_path = alt_path
                break

    movie_output_dir = OUTPUT_DIR / movie_name
    return {
        'video': video_path,
        'output_dir': movie_output_dir,
        'embeddings': movie_output_dir / 'embeddings.npz',
        'metadata': movie_output_dir / 'metadata.json',
        'frames': movie_output_dir / 'frames'
    }

def is_movie_processed(movie_name):
    """Check if a movie has been processed (embeddings exist)"""
    paths = get_movie_paths(movie_name)
    return paths['embeddings'].exists() and paths['metadata'].exists()

def get_movie_status_html(movie_name):
    """Generate HTML status panel for a movie's processing state"""
    if not movie_name:
        return '<div style="color:#718096; text-align:center; padding:20px;">Select a movie to view status</div>'

    paths = get_movie_paths(movie_name)
    video_exists = paths['video'].exists()
    frames_dir = paths['frames']
    frames_exist = frames_dir.exists() and any(frames_dir.glob("frame_*.jpg"))
    frame_count = len(list(frames_dir.glob("frame_*.jpg"))) if frames_exist else 0
    metadata_exists = paths['metadata'].exists()
    embeddings_exists = paths['embeddings'].exists()

    # Get file sizes
    def fmt_size(path):
        if not path.exists():
            return ""
        size = path.stat().st_size
        if size > 1024 * 1024:
            return f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"

    # Caption count from metadata
    caption_count = 0
    if metadata_exists:
        try:
            with open(paths['metadata'], 'r', encoding='utf-8') as f:
                meta = json.load(f)
            caption_count = sum(1 for f in meta.get('frames', []) if f.get('caption'))
        except:
            pass

    def check_row(label, exists, detail=""):
        icon = "✅" if exists else "⬜"
        color = "#48bb78" if exists else "#718096"
        detail_html = f'<span style="color:#a0aec0; font-size:0.8rem; margin-left:8px;">{detail}</span>' if detail else ""
        return f'''
        <div style="display:flex; align-items:center; gap:10px; padding:8px 12px; background:#252a34; border-radius:8px; margin-bottom:6px;">
            <span style="font-size:1.2rem;">{icon}</span>
            <span style="color:{color}; font-weight:500;">{label}</span>
            {detail_html}
        </div>'''

    rows = []
    rows.append(check_row("Video File", video_exists, fmt_size(paths['video']) if video_exists else "not found"))
    rows.append(check_row("Frames Extracted", frames_exist, f"{frame_count} frames" if frames_exist else ""))
    rows.append(check_row("Metadata", metadata_exists, fmt_size(paths['metadata']) if metadata_exists else ""))
    rows.append(check_row("Captions", caption_count > 0, f"{caption_count} captions" if caption_count > 0 else ""))
    rows.append(check_row("Embeddings", embeddings_exists, fmt_size(paths['embeddings']) if embeddings_exists else ""))

    all_done = video_exists and frames_exist and metadata_exists and embeddings_exists and caption_count > 0
    status_badge = '<span style="background:#48bb78; color:#1a1d24; padding:4px 12px; border-radius:12px; font-weight:600; font-size:0.8rem;">Ready for Search</span>' if all_done else '<span style="background:#f59e0b; color:#1a1d24; padding:4px 12px; border-radius:12px; font-weight:600; font-size:0.8rem;">Needs Processing</span>'

    return f'''
    <div style="background:#1a1d24; border-radius:12px; padding:16px; border:1px solid #2d3748;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <span style="color:#e2e8f0; font-weight:600; font-size:1.1rem;">📁 {movie_name}</span>
            {status_badge}
        </div>
        {"".join(rows)}
    </div>
    '''

# ============================================================
# SEARCH ENGINE
# ============================================================
def load_clip_model():
    """Load CLIP model (only once)"""
    global clip_model, tokenizer, device

    if clip_model is not None:
        return True

    device = get_device()
    print(f"[+] Using device: {device}")

    try:
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(device)
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("[+] CLIP model loaded")
        return True
    except Exception as e:
        print(f"[!] Error loading CLIP model: {e}")
        return False


def load_movie_data(movie_name):
    """Load embeddings and metadata for a specific movie"""
    global image_embeddings, text_embeddings, frames, use_hybrid, current_movie

    if not movie_name:
        return False, "No movie selected"

    paths = get_movie_paths(movie_name)

    if not paths['embeddings'].exists():
        return False, f"Movie '{movie_name}' has not been processed yet"

    # Load Embeddings
    try:
        data = np.load(paths['embeddings'])
        if 'image_embeddings' in data:
            image_embeddings = data['image_embeddings']
            text_embeddings = data['text_embeddings']
            use_hybrid = True
        else:
            image_embeddings = data['embeddings']
            text_embeddings = None
            use_hybrid = False
    except Exception as e:
        return False, f"Error loading embeddings: {e}"

    # Load Metadata
    try:
        with open(paths['metadata'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        frames = metadata['frames']
    except Exception as e:
        return False, f"Error loading metadata: {e}"

    current_movie = movie_name
    print(f"[+] Loaded '{movie_name}': {len(frames)} frames")
    return True, f"Loaded {len(frames)} frames"


def load_resources():
    """Initialize - load CLIP model"""
    print("[*] Loading SceneSearch...")
    load_clip_model()
    print("[!] Ready!\n")

def search(query: str, top_k: int, search_mode: str):
    """
    Hybrid search logic
    search_mode: "Visual (Image)", "Hybrid (Smart)", "Conceptual (Text)"
    Returns: (gallery_results, stats_html, timestamps_list)
    """
    global image_embeddings, text_embeddings, frames, use_hybrid, search_results_timestamps

    if image_embeddings is None or frames is None:
        search_results_timestamps = []
        return [], "⚠️ 먼저 '영상 처리' 탭에서 영상을 처리해주세요.", []

    if not query.strip():
        search_results_timestamps = []
        return [], "", []

    # 0. Translate Korean to English if needed
    original_query = query
    translated_query = None
    if contains_korean(query):
        translated_query, translation_error = translate_to_english(query)
        if translation_error:
            error_html = f"""
            <div style="background: linear-gradient(135deg, #5c1a1a 0%, #2d2020 100%); border-radius: 12px; padding: 20px; border: 1px solid #ef4444; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🚫</div>
                <div style="color: #fca5a5; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px;">번역 실패</div>
                <div style="color: #f87171; font-size: 0.9rem; margin-bottom: 12px;">{translation_error}</div>
                <div style="color: #a1a1aa; font-size: 0.8rem; padding: 10px; background: #1f1f1f; border-radius: 8px; font-family: monospace;">
                    Query: "{original_query}"
                </div>
            </div>
            """
            search_results_timestamps = []
            return [], error_html, []
        else:
            query = translated_query

    # 1. Encode Query
    t0 = time.perf_counter()
    text_tokens = tokenizer([query])
    with torch.no_grad():
        query_emb = clip_model.encode_text(text_tokens.to(device))
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy().flatten()
    encode_time = time.perf_counter() - t0

    # 2. Determine Weights based on Mode
    w_img = 0.6
    w_txt = 0.4
    mode_desc = ""

    if search_mode == "Visual":
        w_img, w_txt = 1.0, 0.0
        mode_desc = "🖼️ Visual"
    elif search_mode == "Caption":
        w_img, w_txt = 0.0, 1.0
        mode_desc = "📝 Caption"
    else:  # Hybrid (Smart)
        if use_hybrid and text_embeddings is not None:
            word_count = len(query.split())
            if word_count <= 2:
                w_img, w_txt = 0.8, 0.2
                mode_desc = "⚡ Hybrid (Short Query)"
            else:
                w_img, w_txt = 0.5, 0.5
                mode_desc = "🧠 Hybrid (Balanced)"
        else:
            w_img, w_txt = 1.0, 0.0
            mode_desc = "🖼️ Image Only (No captions avail)"

    # 3. Calculate Similarity
    t1 = time.perf_counter()
    scores = image_embeddings @ query_emb

    if use_hybrid and text_embeddings is not None and w_txt > 0:
        txt_scores = text_embeddings @ query_emb
        scores = scores * w_img + txt_scores * w_txt

    top_indices = np.argsort(scores)[::-1][:top_k]
    search_time = time.perf_counter() - t1

    # 4. Format Results + Collect Timestamps
    results = []
    timestamps = []
    paths = get_movie_paths(current_movie)
    frames_dir = paths['frames']

    for idx in top_indices:
        frame = frames[idx]
        score = scores[idx]
        image_path = frames_dir / frame['filename']

        if image_path.exists():
            caption_text = frame.get('caption', '')
            timestamp = frame.get('timestamp', 0)
            timestamps.append(timestamp)

            # Display caption with click hint
            display_caption = f"⏱️ {frame['time_str']}  |  Score: {score:.3f}"
            if caption_text:
                short_cap = (caption_text[:60] + '..') if len(caption_text) > 60 else caption_text
                display_caption += f"\n📝 {short_cap}"
            display_caption += "\n🎯 Click to jump"

            results.append((str(image_path), display_caption))

    # Store timestamps globally for gallery click handler
    search_results_timestamps = timestamps
    
    # Collect top scores for visualization
    top_scores = [float(scores[idx]) for idx in top_indices]
    total_time = (encode_time + search_time) * 1000
    
    # ===== Generate Scatter Plot Data =====
    all_scores = scores.tolist()
    min_score, max_score = min(all_scores), max(all_scores)
    score_range = max_score - min_score if max_score > min_score else 0.1
    
    # SVG dimensions
    svg_w, svg_h = 500, 120
    pad_l, pad_r, pad_t, pad_b = 35, 10, 10, 25
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b
    
    # Sample frames for performance (max 200 points)
    total_frames_count = len(all_scores)
    if total_frames_count > 200:
        step = total_frames_count // 200
        sample_idx = list(range(0, total_frames_count, step))
    else:
        sample_idx = list(range(total_frames_count))
    
    # Get max timestamp
    max_ts = max(frames[i].get('timestamp', i) for i in sample_idx) if sample_idx else 1
    
    # Scale functions
    def sx(ts): return pad_l + (ts / max_ts) * plot_w if max_ts > 0 else pad_l
    def sy(sc): return pad_t + plot_h - ((sc - min_score) / score_range) * plot_h
    
    # Generate background dots (sampled frames)
    dots = ""
    for i in sample_idx:
        ts = frames[i].get('timestamp', i)
        sc = all_scores[i]
        x, y = sx(ts), sy(sc)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#3d4555" opacity="0.5"><title>{format_time(ts)} | Score: {sc:.3f}</title></circle>'
    
    # Generate top result stars with glow effect
    top_dots = ""
    for rank, idx in enumerate(top_indices):
        ts = frames[idx].get('timestamp', idx)
        sc = all_scores[idx]
        x, y = sx(ts), sy(sc)
        top_dots += f'''
        <circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="#667eea" opacity="0.2" class="pulse-ring"/>
        <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#667eea" stroke="#a78bfa" stroke-width="2" class="top-dot"/>
        <text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" fill="#e2e8f0" font-size="9" font-weight="bold">#{rank+1}</text>
        <title>#{rank+1} | {format_time(ts)} | Score: {sc:.3f}</title>
        '''
    
    # Top results summary
    top_results_html = ""
    for i, (sc, ts) in enumerate(zip(top_scores[:5], timestamps[:5])):
        top_results_html += f'<div style="display:flex; justify-content:space-between; padding:4px 8px; background:#252a34; border-radius:4px; margin-bottom:4px;"><span style="color:#667eea;">#{i+1}</span><span style="color:#a0aec0;">{format_time(ts)}</span><span style="color:#48bb78;">{sc:.3f}</span></div>'

    # Translation info box
    translation_html = ""
    if translated_query:
        translation_html = f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d3748 100%); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; border: 1px solid #3b82f6; display: flex; align-items: center; gap: 14px;">
            <div style="font-size: 1.8rem;">🌐</div>
            <div style="flex: 1;">
                <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 4px;">Translated for search</div>
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <span style="color: #e2e8f0; font-size: 1rem;">{original_query}</span>
                    <span style="color: #3b82f6; font-size: 1.2rem;">→</span>
                    <span style="color: #60a5fa; font-size: 1.05rem; font-weight: 600;">{translated_query}</span>
                </div>
            </div>
        </div>
        """

    # Visualization HTML
    stats = f"""
    {translation_html}
    <style>
        @keyframes cardSlideIn {{
            0% {{ opacity: 0; transform: translateY(20px) scale(0.9); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        @keyframes arrowFade {{
            0% {{ opacity: 0; transform: translateX(-10px); }}
            100% {{ opacity: 1; transform: translateX(0); }}
        }}
        .pipeline-card {{
            background: #2d3748;
            padding: 12px 16px;
            border-radius: 10px;
            text-align: center;
            min-width: 100px;
            opacity: 0;
            animation: cardSlideIn 0.4s ease-out forwards;
        }}
        .pipeline-card.final {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .pipeline-arrow {{
            color: #667eea;
            font-size: 1.2rem;
            opacity: 0;
            animation: arrowFade 0.3s ease-out forwards;
        }}
        .card-0 {{ animation-delay: 0s; }}
        .arrow-0 {{ animation-delay: 0.15s; }}
        .card-1 {{ animation-delay: 0.2s; }}
        .arrow-1 {{ animation-delay: 0.35s; }}
        .card-2 {{ animation-delay: 0.4s; }}
        .arrow-2 {{ animation-delay: 0.55s; }}
        .card-3 {{ animation-delay: 0.6s; }}
        .arrow-3 {{ animation-delay: 0.75s; }}
        .card-4 {{ animation-delay: 0.8s; }}
    </style>
    <div style="background: linear-gradient(135deg, #1a1d24 0%, #252a34 100%); border-radius: 16px; padding: 20px; border: 1px solid #2d3748; margin-bottom: 20px;">

        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 1.2rem; font-weight: 600; color: #e2e8f0;">🔮 Search Pipeline</span>
        </div>

        <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 24px; flex-wrap: wrap;">
            <div class="pipeline-card card-0">
                <div style="font-size: 1.5rem;">{"🌐" if translated_query else "📝"}</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">{"Translated" if translated_query else "Query"}</div>
                <div style="color: #e2e8f0; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">"{query[:15]}{"..." if len(query) > 15 else ""}"</div>
                {f'<div style="color: #667eea; font-size: 0.65rem; margin-top: 2px;">({original_query[:10]}{"..." if len(original_query) > 10 else ""})</div>' if translated_query else ""}
            </div>
            <div class="pipeline-arrow arrow-0">→</div>
            <div class="pipeline-card card-1">
                <div style="font-size: 1.5rem;">🧠</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Image Encode</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">{encode_time*1000:.1f}ms</div>
            </div>
            <div class="pipeline-arrow arrow-1">→</div>
            <div class="pipeline-card card-2">
                <div style="font-size: 1.5rem;">🔢</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Cosine Similarity</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">{len(frames):,} frames</div>
            </div>
            <div class="pipeline-arrow arrow-2">→</div>
            <div class="pipeline-card card-3">
                <div style="font-size: 1.5rem;">📊</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Rank & Select</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">Top {top_k}</div>
            </div>
            <div class="pipeline-arrow arrow-3">→</div>
            <div class="pipeline-card card-4 final">
                <div style="font-size: 1.5rem;">✨</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.75rem; margin-top: 4px;">Results</div>
                <div style="color: white; font-size: 0.85rem; font-weight: 600; margin-top: 2px;">{len(results)} found</div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: center; gap: 24px; margin-bottom: 20px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="color: #a0aec0; font-size: 0.75rem;">Mode</div>
                <div style="color: #667eea; font-weight: 600;">{mode_desc}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #a0aec0; font-size: 0.75rem;">Weights</div>
                <div style="color: #e2e8f0; font-weight: 500;">IMG {w_img:.0%} / TXT {w_txt:.0%}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #a0aec0; font-size: 0.75rem;">Total Time</div>
                <div style="color: #48bb78; font-weight: 600;">{total_time:.1f}ms</div>
            </div>
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="flex: 2; min-width: 280px;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 10px; flex-wrap: wrap;">
                    <span style="color: #a0aec0; font-size: 0.85rem;">📈 Similarity Timeline</span>
                    <div style="display: flex; align-items: center; gap: 12px; background: #252a34; padding: 6px 12px; border-radius: 20px;">
                        <div style="display: flex; align-items: center; gap: 4px;">
                            <span style="display: inline-block; width: 8px; height: 8px; background: #3d4555; border-radius: 50%;"></span>
                            <span style="color: #718096; font-size: 0.7rem;">All frames</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 4px;">
                            <span style="display: inline-block; width: 10px; height: 10px; background: #667eea; border-radius: 50%; box-shadow: 0 0 6px #667eea;"></span>
                            <span style="color: #a78bfa; font-size: 0.7rem;">Top results</span>
                        </div>
                    </div>
                </div>
                <style>
                    .pulse-ring {{ animation: pulse 2s ease-in-out infinite; }}
                    .top-dot {{ filter: drop-shadow(0 0 4px #667eea); }}
                    @keyframes pulse {{ 0%,100% {{ opacity:0.2; r:14; }} 50% {{ opacity:0.4; r:18; }} }}
                </style>
                <svg viewBox="0 0 {svg_w} {svg_h}" style="width:100%; background:#1a1d24; border-radius:8px; overflow:visible;">
                    <line x1="{pad_l}" y1="{svg_h - pad_b}" x2="{svg_w - pad_r}" y2="{svg_h - pad_b}" stroke="#2d3748"/>
                    <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{svg_h - pad_b}" stroke="#2d3748"/>
                    <text x="{pad_l - 3}" y="{pad_t + 4}" text-anchor="end" fill="#718096" font-size="8">{max_score:.2f}</text>
                    <text x="{pad_l - 3}" y="{svg_h - pad_b}" text-anchor="end" fill="#718096" font-size="8">{min_score:.2f}</text>
                    <text x="{pad_l}" y="{svg_h - 8}" fill="#718096" font-size="8">0:00</text>
                    <text x="{svg_w - pad_r}" y="{svg_h - 8}" text-anchor="end" fill="#718096" font-size="8">{format_time(max_ts)}</text>
                    {dots}
                    {top_dots}
                </svg>
            </div>
            <div style="flex: 1; min-width: 180px;">
                <div style="color: #a0aec0; font-size: 0.8rem; margin-bottom: 8px; text-align: center;">
                    🏆 Top Matches
                </div>
                <div style="background: #1a1d24; border-radius: 8px; padding: 8px;">
                    {top_results_html}
                </div>
            </div>
        </div>
        
    </div>
    """

    return results, stats, timestamps


def on_gallery_select(evt: gr.SelectData, timestamps_state):
    """Handle gallery image click - return timestamp to seek"""
    if timestamps_state and evt.index < len(timestamps_state):
        return timestamps_state[evt.index]
    return None

# ============================================================
# PROCESSING ENGINE
# ============================================================
def make_progress_html(phase, current=0, total=0, fps=0, eta=0, caption=""):
    """Generate HTML progress bar for processing stages"""
    hint = ""
    if phase == "extract":
        # current=last_pts_time, total=video_duration, eta=frame_count_found
        pct = (current / total * 100) if total > 0 else 0
        frame_count = int(eta)
        status_icon = "🎬"
        status_text = f"Extracting frames... {frame_count} found"
        if total > 0 and current > 0:
            sub_text = f"Scanning video · {format_time(current)} / {format_time(total)}"
            pulse = False
        else:
            sub_text = "Starting ffmpeg scene detection..."
            pulse = True
        bar_color = "linear-gradient(90deg, #f59e0b, #d97706)"
        hint = f"💡 장면 전환(scene change)이 설정값({caption})을 초과하는 순간만 키프레임으로 추출합니다" if caption else "💡 장면 전환(scene change)이 감지된 순간만 키프레임으로 추출합니다"
    elif phase == "model_load":
        pct = 0
        status_icon = "🧠"
        status_text = "Loading AI models..."
        sub_text = "CLIP + BLIP (first time may take a minute)"
        bar_color = "linear-gradient(90deg, #8b5cf6, #6d28d9)"
        pulse = True
    elif phase == "analyze":
        pct = (current / total * 100) if total > 0 else 0
        status_icon = "⚡"
        status_text = f"Analyzing frames... {current}/{total}"
        eta_str = f"{int(eta//60)}분 {int(eta%60)}초" if eta >= 60 else f"{eta:.0f}초"
        sub_text = f"처리 속도: {fps:.1f} frames/sec · 남은 시간: {eta_str}"
        if caption:
            sub_text += f'<br>📝 "{caption[:60]}..."' if len(caption) > 60 else f'<br>📝 "{caption}"'
        bar_color = "linear-gradient(90deg, #667eea, #764ba2)"
        pulse = False
        hint = "💡 CLIP으로 이미지 임베딩, BLIP으로 캡션 생성 후 캡션도 임베딩합니다"
    elif phase == "save":
        pct = 100
        status_icon = "💾"
        status_text = "Saving data..."
        sub_text = "Writing embeddings and metadata"
        bar_color = "linear-gradient(90deg, #10b981, #059669)"
        pulse = False
    elif phase == "done":
        pct = 100
        status_icon = "✅"
        status_text = f"Complete! {total} frames processed"
        sub_text = "Ready for search"
        bar_color = "linear-gradient(90deg, #10b981, #059669)"
        pulse = False
    elif phase == "stopped":
        pct = (current / total * 100) if total > 0 else 0
        status_icon = "⛔"
        status_text = "Stopped by user"
        sub_text = f"{int(eta)} frames extracted before stop" if eta > 0 else "Processing cancelled"
        bar_color = "linear-gradient(90deg, #ef4444, #b91c1c)"
        pulse = False
    else:
        return ""

    pulse_css = """
        @keyframes barPulse {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
    """ if pulse else ""
    pulse_style = "background-size: 200% 100%; animation: barPulse 2s ease-in-out infinite;" if pulse else ""
    bar_width = "100%" if pulse else f"{pct}%"
    pct_display = f'<div style="color: #667eea; font-size: 1.3rem; font-weight: 700;">{pct:.1f}%</div>' if not pulse else ""

    return f"""
    <style>{pulse_css}</style>
    <div style="background: #1a1d24; border-radius: 16px; padding: 24px; border: 1px solid #2d3748;">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
            <div style="font-size: 2rem;">{status_icon}</div>
            <div style="flex: 1;">
                <div style="color: #e2e8f0; font-size: 1.1rem; font-weight: 600;">{status_text}</div>
                <div style="color: #718096; font-size: 0.85rem; margin-top: 2px;">{sub_text}</div>
            </div>
            {pct_display}
        </div>
        <div style="background: #2d3748; border-radius: 8px; height: 12px; overflow: hidden;">
            <div style="background: {bar_color}; height: 100%; border-radius: 8px; width: {bar_width}; transition: width 0.3s ease; {pulse_style}"></div>
        </div>
        {f'<div style="color:#5a6577; font-size:0.75rem; margin-top:10px; text-align:center;">{hint}</div>' if hint else ''}
    </div>
    """


def process_video(video_file, scene_threshold):
    """Full pipeline: Extract -> Caption -> Embed
    Yields: (log_text, progress_html, sample_gallery)
    """
    global current_ffmpeg_proc

    if video_file is None:
        yield "❌ 영상 파일을 선택해주세요.", "", None
        return

    stop_event.clear()

    video_path = Path(video_file)
    movie_name = video_path.stem
    paths = get_movie_paths(movie_name)
    frames_dir = paths['frames']
    output_dir = paths['output_dir']

    log_messages = []

    def log(msg):
        timestamp = time.strftime("%H:%M:%S")
        log_messages.append(f"[{timestamp}] {msg}")
        return "\n".join(log_messages)

    try:
        # 1. Init - Copy to video/ folder if uploaded from elsewhere
        yield log(f"🚀 Processing: {movie_name}"), "", None

        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        dest_video = VIDEO_DIR / video_path.name
        if not dest_video.exists() or dest_video.resolve() != video_path.resolve():
            yield log(f"📁 Copying video to video/ folder..."), "", None
            shutil.copy2(str(video_path), str(dest_video))
            yield log(f"✅ Saved: {dest_video.name}"), "", None
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. FFmpeg Extraction (with live progress)
        video_duration = get_video_duration(video_path)
        duration_str = format_time(video_duration) if video_duration > 0 else "unknown"
        yield log(f"🎬 Scene detection (threshold: {scene_threshold}) · Duration: {duration_str}"), make_progress_html("extract", 0, video_duration, 0, 0, str(scene_threshold)), None

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
            "-vsync", "vfr",
            str(frames_dir / "frame_%04d.jpg")
        ]

        proc = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='replace'
        )
        current_ffmpeg_proc = proc
        pts_times = []
        last_update = time.time()
        for line in proc.stderr:
            if stop_event.is_set():
                proc.terminate()
                proc.wait()
                current_ffmpeg_proc = None
                last_pts = pts_times[-1] if pts_times else 0
                yield log("⛔ Stopped by user."), make_progress_html("stopped", last_pts, video_duration, 0, len(pts_times)), None
                return
            if "pts_time:" in line:
                match = re.search(r'pts_time:(\d+\.?\d*)', line)
                if match:
                    pts_times.append(float(match.group(1)))
                    # Update UI every 2 seconds
                    now = time.time()
                    if now - last_update > 2:
                        last_update = now
                        last_pts = pts_times[-1]
                        yield log(f"   ▶ Extracting... {len(pts_times)} frames ({format_time(last_pts)})"), make_progress_html("extract", last_pts, video_duration, 0, len(pts_times), str(scene_threshold)), None
        proc.wait()
        current_ffmpeg_proc = None

        frame_files = list(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            yield log("❌ No frames extracted. Try lowering the threshold."), "", None
            return

        yield log(f"✅ Extracted {len(frame_files)} frames"), make_progress_html("extract", video_duration, video_duration, 0, len(frame_files), str(scene_threshold)), None

        # 3. Metadata Structure
        frames_data = []
        for i, pts in enumerate(pts_times):
            if i >= len(frame_files):
                break
            frames_data.append({
                "index": i,
                "filename": f"frame_{i+1:04d}.jpg",
                "timestamp": round(pts, 3),
                "time_str": format_time(pts)
            })

        # 4. AI Processing (CLIP + BLIP)
        yield log("🧠 Loading AI models..."), make_progress_html("model_load"), None

        proc_device = get_device()

        c_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        c_model = c_model.to(proc_device).eval()
        c_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        from transformers import BlipProcessor, BlipForConditionalGeneration
        b_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        b_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(proc_device).eval()

        yield log("✅ Models loaded. Analyzing frames..."), make_progress_html("analyze", 0, len(frames_data)), None

        img_embs = []
        txt_embs = []

        start_time = time.time()
        total = len(frames_data)

        for i, frame_info in enumerate(frames_data):
            if stop_event.is_set():
                yield log(f"⛔ Stopped by user. ({i}/{total} frames processed)"), make_progress_html("stopped", i, total, 0, i), None
                return

            f_path = frames_dir / frame_info['filename']
            image = Image.open(f_path).convert('RGB')

            # CLIP Image Embedding
            img_in = preprocess(image).unsqueeze(0).to(proc_device)
            with torch.no_grad():
                ie = c_model.encode_image(img_in)
                ie /= ie.norm(dim=-1, keepdim=True)
            img_embs.append(ie.cpu().numpy().flatten())

            # BLIP Caption
            with torch.no_grad():
                inputs = b_processor(image, return_tensors="pt").to(proc_device)
                out = b_model.generate(**inputs, max_new_tokens=50)
                caption = b_processor.decode(out[0], skip_special_tokens=True)
            frame_info['caption'] = caption

            # CLIP Text Embedding (of Caption)
            txt_in = c_tokenizer([caption])
            with torch.no_grad():
                te = c_model.encode_text(txt_in.to(proc_device))
                te /= te.norm(dim=-1, keepdim=True)
            txt_embs.append(te.cpu().numpy().flatten())

            if (i + 1) % 5 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                eta = (total - (i + 1)) / fps if fps > 0 else 0
                yield log(f"   ▶ [{i+1}/{total}] {fps:.1f} FPS (ETA: {eta:.0f}s)"), make_progress_html("analyze", i + 1, total, fps, eta, caption), None

        # 5. Save Data
        yield log("💾 Saving data..."), make_progress_html("save"), None

        np.savez(
            paths['embeddings'],
            image_embeddings=np.array(img_embs),
            text_embeddings=np.array(txt_embs),
            timestamps=np.array([f['timestamp'] for f in frames_data])
        )

        metadata = {
            "video_info": {"filename": video_path.name, "movie_name": movie_name},
            "extraction_config": {"method": "scene_detection", "threshold": scene_threshold},
            "total_frames": total,
            "frames": frames_data
        }
        with open(paths['metadata'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        yield log(f"✨ Done! '{movie_name}' is ready for search."), make_progress_html("done", total=total), None

        # Show Samples
        samples = []
        for idx in [0, total // 2, total - 1]:
            if idx < total:
                f = frames_data[idx]
                samples.append((str(frames_dir / f['filename']), f"[{f['time_str']}] {f['caption']}"))

        yield log(""), make_progress_html("done", total=total), samples

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        yield log(f"❌ Error:\n{err}"), make_progress_html("stopped", 0, 0, 0, 0), None

# ============================================================ 
# UI BUILDER
# ============================================================ 
def create_app():
    
    # Custom CSS
    css = """
    /* Main Layout */
    .container { max-width: 1200px; margin: auto; padding-top: 0px; }
    
    /* Dark Theme Base */
    .gradio-container {
        background: #0f1117 !important;
    }
    
    /* Header Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Header */
    .header { text-align: center; margin-bottom: 40px; }
    .header h1 {
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        opacity: 0;
        animation: fadeIn 1.2s ease forwards;
    }
    .header p {
        color: #718096;
        font-size: 1.2em;
        font-weight: 400;
        opacity: 0;
        animation: fadeIn 1s ease forwards;
        animation-delay: 0.5s;
    }
    
    /* Search Section */
    .search-row { 
        background: #1a1d24 !important; 
        padding: 30px; 
        border-radius: 20px; 
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d3748;
        margin-bottom: 30px;
    }
    
    /* Inputs */
    .search-input textarea { 
        font-size: 1.2rem !important; 
        padding: 15px !important; 
        border-radius: 12px !important; 
        border: 2px solid #2d3748 !important;
        background: #0f1117 !important;
        color: #e2e8f0 !important;
        transition: all 0.2s;
    }
    .search-input textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Buttons */
    .primary-btn { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        transition: transform 0.1s !important;
    }
    .primary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Gallery - Dark Theme */
    .result-gallery { 
        border-radius: 16px; 
        overflow: hidden; 
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        padding: 16px !important;
    }
    .result-gallery .grid-wrap { 
        gap: 16px !important; 
        background: #1a1d24 !important;
    }
    .result-gallery .thumbnail-item {
        background: #252a34 !important;
        border-radius: 12px !important;
        border: 1px solid #2d3748 !important;
    }
    .result-gallery .caption {
        background: #252a34 !important;
        color: #a0aec0 !important;
    }
    
    /* Labels & Text */
    .gr-box, .gr-panel {
        background: #1a1d24 !important;
        border-color: #2d3748 !important;
    }
    label, .label-wrap {
        color: #a0aec0 !important;
    }
    
    /* Radio Buttons */
    .gr-radio {
        background: #1a1d24 !important;
    }
    
    /* Slider */
    .gr-slider input[type="range"] {
        background: #2d3748 !important;
    }
    
    /* Tabs */
    .tabs {
        background: transparent !important;
        border: none !important;
    }
    .tab-nav {
        background: #1a1d24 !important;
        border-radius: 12px !important;
        border: 1px solid #2d3748 !important;
    }
    .tab-nav button {
        color: #a0aec0 !important;
    }
    .tab-nav button.selected {
        background: #667eea !important;
        color: white !important;
    }
    
    /* Logs */
    .log-area {
        padding: 8px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    .log-area textarea { 
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        background-color: #0f1117 !important;
        color: #a0aec0 !important;
        border-radius: 12px;
        border: 1px solid #2d3748 !important;
        line-height: 1.5;
        min-height: 240px !important;
        width: 100% !important;
    }
    
    /* Examples */
    .gr-examples {
        background: transparent !important;
    }
    .gr-examples button {
        background: #252a34 !important;
        color: #a0aec0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
    }
    .gr-examples button:hover {
        background: #667eea !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    /* Video Input */
    .gr-video {
        background: #1a1d24 !important;
        border: 2px dashed #2d3748 !important;
        border-radius: 16px !important;
    }
    
    /* Accordion */
    .gr-accordion {
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
    }
    
    /* Gallery - Disable modal/lightbox on click */
    .result-gallery .preview,
    .result-gallery .modal,
    .result-gallery .backdrop,
    .result-gallery .fixed {
        display: none !important;
    }

    /* Footer */
    footer { display: none !important; }
    
    /* Compact file upload area text */
    .compact-upload * {
        font-size: 0.8rem !important;
    }
    .compact-upload label span {
        font-size: 0.85rem !important;
    }
    
    /* Remove hide-container extra padding from section headers */
    .section-header {
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
        height: 20px !important;
        border-width: 0 !important;
        overflow: visible !important;
    }
    .section-header h4 {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    """

    # JavaScript for video seeking
    seek_js = """
    (timestamp) => {
        if (timestamp === null || timestamp === undefined) return timestamp;

        const video = document.querySelector('#main-player video');
        if (video) {
            video.currentTime = timestamp;
            video.pause();
        }

        return timestamp;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", radius_size="lg"), css=css, title="SceneSearch") as app:

        # State for timestamps
        timestamps_state = gr.State([])

        with gr.Column(elem_classes="container"):

            # Header
            gr.HTML("""
                <div class="header">
                    <h1>SceneSearch</h1>
                    <p>Find any moment with natural language</p>
                </div>
            """)

            with gr.Tabs():

                # ---------------------------------------------------------
                # TAB 1: SEARCH + PLAYER
                # ---------------------------------------------------------
                with gr.Tab("🔍 Search", id="search_tab"):

                    # Movie Selection
                    with gr.Row():
                        movie_dropdown = gr.Dropdown(
                            choices=get_available_movies(),
                            label="Select Movie",
                            value=None,
                            scale=3
                        )
                        refresh_btn = gr.Button("🔄", scale=0, min_width=50)

                    # Status message
                    movie_status = gr.HTML(
                        value="<p style='text-align:center; color:#718096;'>Select a movie to start</p>"
                    )

                    # Video Player Section
                    with gr.Column(elem_classes="player-section"):
                        video_player = gr.Video(
                            label=None,
                            show_label=False,
                            elem_id="main-player",
                            height=400,
                            interactive=False
                        )

                    # Search Section
                    with gr.Column(elem_classes="search-row"):
                        query_input = gr.Textbox(
                            show_label=False,
                            placeholder="Describe the scene... (e.g., 'Johnny Depp smiling', 'explosion scene')",
                            lines=1,
                            elem_classes="search-input",
                            autofocus=True
                        )

                        with gr.Row(equal_height=True):
                            search_mode = gr.Radio(
                                choices=["Hybrid", "Visual", "Caption"],
                                value="Hybrid",
                                label="Mode",
                                scale=2
                            )
                            top_k = gr.Slider(4, 20, value=8, step=4, label="Results", scale=1)
                            search_btn = gr.Button("Search", variant="primary", scale=1, elem_classes="primary-btn")

                    # Results
                    stats_output = gr.HTML()
                    gallery = gr.Gallery(
                        label="Click a frame to jump to that moment",
                        columns=4,
                        height="auto",
                        object_fit="cover",
                        elem_classes="result-gallery",
                        show_label=True,
                        interactive=False,
                        preview=False
                    )

                    # Examples
                    gr.Examples(
                        examples=[
                            ["Johnny Depp face"],
                            ["computer screen"],
                            ["two people talking"],
                            ["explosion or fire"],
                            ["outdoor trees"]
                        ],
                        inputs=query_input,
                        label="Try:"
                    )
                    
                    # Hidden component for JS bridge
                    seek_timestamp = gr.Number(value=-1, visible=False)

                # ---------------------------------------------------------
                # TAB 2: PROCESSING
                # ---------------------------------------------------------
                with gr.Tab("⚙️ Process Video", id="process_tab"):

                    # Row 1: Video Source | Progress
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### 🎬 Video Source")
                            with gr.Group(elem_classes="video-source-group"):
                                process_dropdown = gr.Dropdown(
                                    choices=get_available_movies(),
                                    label="Select from video/ folder",
                                    value=None,
                                )
                                video_input = gr.File(
                                    label="Or upload new video",
                                    file_types=["video"],
                                    height=100,
                                    elem_classes="compact-upload",
                                )
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### 📊 Progress", elem_classes="section-header")
                            progress_output = gr.HTML(
                                value='<div style="background:#1a1d24; border-radius:12px; padding:40px; border:1px solid #2d3748; text-align:center; color:#718096; display:flex; align-items:center; justify-content:center;">Waiting to start...</div>'
                            )

                    # Row 2: Settings | Log
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### ⚙️ Settings", elem_classes="section-header")
                            threshold = gr.Slider(
                                0.1, 0.5, value=0.3, step=0.05,
                                label="Scene Sensitivity",
                                info="ffmpeg의 scene change detection 임계값"
                            )
                            gr.HTML('''<div style="color:#8a94a6; font-size:0.72rem; line-height:1.5; padding:0px 4px; margin-top:-6px;">
                                <div style="margin-bottom:3px;">영상의 연속된 두 프레임 간 <b style="color:#a0aec0;">픽셀 변화량(0~1)</b>을 비교해서, 이 값을 초과하면 "장면이 바뀌었다"고 판단하여 해당 프레임을 추출합니다.</div>
                                <div style="display:flex; gap:12px; flex-wrap:wrap;">
                                    <span>🔹 <b style="color:#f59e0b;">0.1</b> 민감 (프레임 많음)</span>
                                    <span>🔹 <b style="color:#48bb78;">0.3</b> 권장</span>
                                    <span>🔹 <b style="color:#667eea;">0.5</b> 둔감 (프레임 적음)</span>
                                </div>
                            </div>''')
                            with gr.Row():
                                process_btn = gr.Button("▶ Start Processing", variant="primary", scale=3)
                                stop_btn = gr.Button("■ Stop", variant="stop", scale=1, visible=False)
                                process_refresh_btn = gr.Button("🔄", scale=0, min_width=40)
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### 📋 Log", elem_classes="section-header")
                            log_output = gr.Textbox(
                                show_label=False,
                                max_lines=8,
                                elem_classes="log-area",
                                interactive=False,
                                autoscroll=True
                            )

                    # Sample Frames (full width)
                    sample_output = gr.Gallery(label="Sample Frames", columns=4, height="auto")

                    # Manage Section
                    gr.Markdown("---")
                    gr.Markdown("#### 📁 Manage Processed Movies")
                    with gr.Row():
                        manage_dropdown = gr.Dropdown(
                            choices=get_available_movies(),
                            label="Select Movie",
                            value=None,
                            scale=4
                        )
                        manage_refresh_btn = gr.Button("🔄 Refresh", scale=0, min_width=100)
                        delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=0, min_width=100)
                    movie_status_html = gr.HTML(
                        value='<div style="color:#718096; text-align:center; padding:16px;">Select a movie to view status</div>'
                    )

        # ---------------------------------------------------------
        # EVENT HANDLERS
        # ---------------------------------------------------------

        # Movie selection -> load video + embeddings
        def on_movie_select(movie_name):
            if not movie_name:
                return None, "<p style='text-align:center; color:#718096;'>Select a movie to start</p>", [], ""

            paths = get_movie_paths(movie_name)

            # Check if processed
            if not is_movie_processed(movie_name):
                return (
                    None,
                    f"<p style='text-align:center; color:#f6ad55;'>⚠️ '{movie_name}' needs processing. Go to Process Video tab.</p>",
                    [],
                    ""
                )

            # Load embeddings
            success, msg = load_movie_data(movie_name)
            if not success:
                return None, f"<p style='text-align:center; color:#fc8181;'>❌ {msg}</p>", [], ""

            # Return video path and success status
            video_path = str(paths['video']) if paths['video'].exists() else None
            status_html = f"<p style='text-align:center; color:#48bb78;'>✓ Loaded: {movie_name} ({len(frames)} frames)</p>"

            return video_path, status_html, [], ""

        movie_dropdown.change(
            fn=on_movie_select,
            inputs=[movie_dropdown],
            outputs=[video_player, movie_status, gallery, stats_output]
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                setTimeout(() => {
                    const video = document.querySelector('#main-player video');
                    if (video && video.src) {
                        video.play().catch(e => console.log('Autoplay blocked:', e));
                    }
                }, 800);
            }
            """
        )

        # Refresh movie list
        def refresh_movies():
            movies = get_available_movies()
            return gr.update(choices=movies)

        refresh_btn.click(
            fn=refresh_movies,
            inputs=[],
            outputs=[movie_dropdown]
        )

        # Search -> update gallery + timestamps
        def do_search(query, k, mode):
            results, stats, ts = search(query, k, mode)
            return results, stats, ts

        search_btn.click(
            fn=do_search,
            inputs=[query_input, top_k, search_mode],
            outputs=[gallery, stats_output, timestamps_state]
        )
        query_input.submit(
            fn=do_search,
            inputs=[query_input, top_k, search_mode],
            outputs=[gallery, stats_output, timestamps_state]
        )

        # Gallery click -> seek video
        gallery.select(
            fn=on_gallery_select,
            inputs=[timestamps_state],
            outputs=[seek_timestamp]
        )
        
        # Trigger seek when timestamp changes
        seek_timestamp.change(
            fn=None,
            inputs=[seek_timestamp],
            outputs=[],
            js=seek_js
        )

        # Process video
        def do_process(selected_movie, uploaded_file, thresh):
            # Determine video path: dropdown takes priority, then upload
            video_path = None
            if selected_movie:
                paths = get_movie_paths(selected_movie)
                video_path = str(paths['video'])
            elif uploaded_file:
                video_path = uploaded_file  # gr.File returns path string
            
            for log_msg, prog_html, samples in process_video(video_path, thresh):
                yield log_msg, prog_html, samples

        def on_process_start():
            """Show stop button, hide start button"""
            return gr.update(visible=False), gr.update(visible=True)

        def on_process_end():
            """Show start button, hide stop button, refresh dropdowns"""
            movies = get_available_movies()
            return (
                gr.update(visible=True),
                gr.update(visible=False, interactive=True, value="■ Stop"),
                gr.update(choices=movies),
                gr.update(choices=movies),
            )

        def on_stop_click():
            """Signal processing to stop"""
            stop_event.set()
            if current_ffmpeg_proc and current_ffmpeg_proc.poll() is None:
                current_ffmpeg_proc.terminate()
            return gr.update(interactive=False, value="■ Stopping...")

        def on_process_refresh():
            return gr.update(choices=get_available_movies())

        process_refresh_btn.click(
            fn=on_process_refresh,
            inputs=[],
            outputs=[process_dropdown]
        )

        process_click_event = process_btn.click(
            fn=on_process_start,
            inputs=[],
            outputs=[process_btn, stop_btn]
        ).then(
            fn=do_process,
            inputs=[process_dropdown, video_input, threshold],
            outputs=[log_output, progress_output, sample_output]
        ).then(
            fn=on_process_end,
            inputs=[],
            outputs=[process_btn, stop_btn, process_dropdown, manage_dropdown]
        )

        stop_btn.click(
            fn=on_stop_click,
            inputs=[],
            outputs=[stop_btn],
            cancels=[process_click_event]
        ).then(
            fn=on_process_end,
            inputs=[],
            outputs=[process_btn, stop_btn, process_dropdown, manage_dropdown]
        )

        # Manage section handlers
        def on_manage_select(movie_name):
            return get_movie_status_html(movie_name)

        def on_manage_refresh():
            movies = get_available_movies()
            return gr.update(choices=movies)

        def delete_movie_data(movie_name):
            if not movie_name:
                return get_movie_status_html(None), gr.update(choices=get_available_movies())
            paths = get_movie_paths(movie_name)
            output_dir = paths['output_dir']
            if output_dir.exists():
                shutil.rmtree(output_dir)
            return get_movie_status_html(movie_name), gr.update(choices=get_available_movies())

        manage_dropdown.change(
            fn=on_manage_select,
            inputs=[manage_dropdown],
            outputs=[movie_status_html]
        )
        manage_refresh_btn.click(
            fn=on_manage_refresh,
            inputs=[],
            outputs=[manage_dropdown]
        )
        delete_btn.click(
            fn=delete_movie_data,
            inputs=[manage_dropdown],
            outputs=[movie_status_html, manage_dropdown]
        )

    return app

if __name__ == "__main__":
    load_resources()
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7880,
        inbrowser=True
    )
