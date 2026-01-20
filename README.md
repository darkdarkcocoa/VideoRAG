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
- 📝 **BLIP Captions** - 각 프레임에 대한 자동 캡션 생성 (NEW!)
- 🔀 **Hybrid Search** - 이미지 + 캡션 결합 검색으로 정확도 향상 (NEW!)
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
pip install torch torchvision open-clip-torch opencv-python pillow numpy transformers
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

### 3. Generate Embeddings + Captions
```bash
python generate_embeddings.py
```
이 단계에서:
- CLIP으로 이미지 임베딩 생성
- BLIP으로 각 프레임 캡션 자동 생성
- 캡션을 CLIP 텍스트 임베딩으로 변환

### 4. Search!
```bash
python search_test.py
```

또는 웹 UI로 검색:
```bash
python app.py
# → http://127.0.0.1:7860
```

---

## 📁 Project Structure

```
SceneSearch/
├── app.py                 # Gradio 웹 UI (Hybrid Search)
├── prototype.py           # 올인원 프로토타입
├── create_metadata.py     # ffmpeg 로그 → metadata.json 변환
├── generate_embeddings.py # CLIP 임베딩 + BLIP 캡션 생성
├── search_test.py         # 검색 테스트 스크립트
└── output/
    ├── frames/            # 추출된 프레임 이미지
    ├── metadata.json      # 프레임 정보 + 캡션
    └── embeddings.npz     # CLIP 임베딩 (이미지 + 텍스트)
```

---

## 🔀 Hybrid Search 원리

```
┌─────────────────────────────────────────────────────────────┐
│  사전 준비                                                   │
├─────────────────────────────────────────────────────────────┤
│  프레임 이미지 → [CLIP Image] → 이미지 벡터 (512d)           │
│              → [BLIP]       → "a man in laboratory"         │
│                                      ↓                       │
│                               [CLIP Text] → 캡션 벡터 (512d) │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  검색 시                                                     │
├─────────────────────────────────────────────────────────────┤
│  "laboratory" → [CLIP Text] → 쿼리 벡터                      │
│                                   ↓                          │
│              쿼리 vs 이미지 벡터 → 유사도 A                   │
│              쿼리 vs 캡션 벡터  → 유사도 B                   │
│                                   ↓                          │
│              최종 점수 = A × 0.6 + B × 0.4                   │
└─────────────────────────────────────────────────────────────┘
```

**동적 가중치**: 
- 짧은 쿼리 (1-2단어): 이미지 70%, 캡션 30% → 시각적 매칭 우선
- 긴 쿼리 (3단어+): 이미지 50-60%, 캡션 40-50% → 의미 매칭 강화

---

## 📊 Example Results

**Transcendence (2014)** 영화로 테스트한 결과:

| Query | Top Result | Timestamp | Caption |
|-------|------------|-----------|---------|
| "Johnny Depp face" | 조니뎁 정면 | 01:02:03 | "a man with glasses looking at camera" |
| "computer screen" | 모니터 장면 | 05:41 | "a computer screen with code" |
| "laboratory" | 연구실 | 17:07 | "a man in white coat in laboratory" |

---

## 🔧 Configuration

### Scene Detection Threshold
`ffmpeg` 명령어에서 `scene` 값 조정:
- `0.1` - 민감 (프레임 많음)
- `0.3` - 적당 (권장)
- `0.5` - 둔감 (프레임 적음)

### Search Weight (웹 UI)
고급 설정에서 이미지 가중치 조절 가능:
- `1.0` - 이미지만 사용
- `0.6` - 이미지 중심 (기본값)
- `0.0` - 캡션만 사용

---

## 🗺️ Roadmap

- [x] BLIP 캡션 생성
- [x] 하이브리드 검색
- [ ] 한국어 검색 지원 (multilingual CLIP)
- [ ] 벡터 DB 연동 (대용량 영상)
- [ ] 오디오/자막 통합 검색
- [ ] 실시간 스트리밍 지원

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- [OpenCLIP](https://github.com/mlfoundations/open_clip) - CLIP implementation
- [BLIP](https://github.com/salesforce/BLIP) - Image captioning
- [FFmpeg](https://ffmpeg.org/) - Video processing

---

Made with ❤️ and ☕
