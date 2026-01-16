# 🎬 SceneSearch

**Video Semantic Search** - 자연어로 영상 속 장면을 검색하세요!

영상에서 원하는 장면을 찾고 싶을 때, 일일이 타임라인을 넘기지 않아도 됩니다.
"조니뎁 얼굴", "연구실 장면", "폭발 씬"처럼 자연어로 검색하면 해당 장면의 타임스탬프를 찾아줍니다.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 🎯 **Scene Detection** - ffmpeg 기반 장면 전환 감지로 효율적인 프레임 추출
- 🧠 **CLIP Embeddings** - OpenAI CLIP 모델로 이미지-텍스트 시맨틱 매칭
- 🔍 **Natural Language Search** - 영어 자연어로 원하는 장면 검색
- ⚡ **GPU Accelerated** - CUDA 지원으로 빠른 임베딩 생성

---

## 🛠️ Installation

### Requirements
- Python 3.10+
- CUDA (optional, for GPU acceleration)
- ffmpeg

### Setup
```bash
# Clone repository
git clone https://github.com/darkdarkcocoa/SceneSearch.git
cd SceneSearch

# Install dependencies
pip install torch torchvision open-clip-torch opencv-python pillow numpy
```

---

## 🚀 Usage

### 1. Frame Extraction (ffmpeg scene detection)
```bash
# 장면 전환 30% 이상일 때만 프레임 추출
ffmpeg -i your_video.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfr output/frames/frame_%04d.jpg 2>&1 | grep "pts_time" > output/frame_log.txt
```

### 2. Create Metadata
```bash
python create_metadata.py
```

### 3. Generate Embeddings
```bash
python generate_embeddings.py
```

### 4. Search!
```bash
python search_test.py
```

또는 `prototype.py`로 인터랙티브 검색:
```bash
python prototype.py
```

---

## 📁 Project Structure

```
SceneSearch/
├── app.py                 # Gradio 웹 UI
├── prototype.py           # 올인원 프로토타입 (추출 + 임베딩 + 검색)
├── create_metadata.py     # ffmpeg 로그 → metadata.json 변환
├── generate_embeddings.py # CLIP 임베딩 생성
├── search_test.py         # 검색 테스트 스크립트
└── output/
    ├── frames/            # 추출된 프레임 이미지
    ├── metadata.json      # 프레임 타임스탬프 정보
    └── embeddings.npz     # CLIP 임베딩 벡터
```

---

## 📊 Example Results

**Transcendence (2014)** 영화로 테스트한 결과:

| Query | Top Result | Timestamp |
|-------|------------|-----------|
| "Johnny Depp face" | 조니뎁 정면 얼굴 | 01:02:03 |
| "computer screen" | 컴퓨터 모니터 장면 | 05:41 |
| "laboratory" | 연구실 실험 장면 | 17:07 |
| "outdoor garden" | 야외 정원 | 01:04:50 |
| "explosion" | 폭발/연기 장면 | 10:40 |

### Performance
- **Frame Extraction**: ~1,125 frames from 2hr movie (scene detection)
- **Embedding Speed**: ~80 frames/sec (RTX 4060 Ti)
- **Search Speed**: Instant (cosine similarity)

---

## 🔧 Configuration

### Scene Detection Threshold
`ffmpeg` 명령어에서 `scene` 값 조정:
- `0.1` - 민감 (프레임 많음)
- `0.3` - 적당 (권장)
- `0.5` - 둔감 (프레임 적음)

### CLIP Model
현재 `ViT-B-32` 사용 중. 더 정확한 검색을 원하면:
```python
# generate_embeddings.py에서 변경
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
```

---

## 🗺️ Roadmap

- [ ] 한국어 검색 지원 (multilingual CLIP)
- [ ] 웹 UI 추가
- [ ] 벡터 DB 연동 (대용량 영상)
- [ ] 오디오/자막 통합 검색
- [ ] 실시간 스트리밍 지원

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- [OpenCLIP](https://github.com/mlfoundations/open_clip) - CLIP implementation
- [FFmpeg](https://ffmpeg.org/) - Video processing

---

Made with ❤️ and ☕
