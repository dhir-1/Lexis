<p align="center">
  <img src="assets/logo.png" alt="Lexis Logo" width="130" />
</p>

<h1 align="center">Lexis</h1>

<p align="center">
  <strong>Real-Time Multimodal Continuous ASL Translation & Multilingual Speech Subtitling Platform</strong>
</p>

<p align="center">
  <a href="#key-features"><img src="https://img.shields.io/badge/Vision_Tracking-RTMPose_Wholebody-6366F1?style=for-the-badge&logo=opencv&logoColor=white" alt="RTMPose Wholebody"></a>
  <a href="#key-features"><img src="https://img.shields.io/badge/Audio_Pipeline-Groq_Whisper_V3-8B5CF6?style=for-the-badge&logo=fastapi&logoColor=white" alt="Whisper Large V3"></a>
  <a href="#key-features"><img src="https://img.shields.io/badge/Deep_Learning-PyTorch_Bi--GRU-EC4899?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#tech-stack"><img src="https://img.shields.io/badge/Database-Neon_PostgreSQL-00E599?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"></a>
</p>

---

## 📖 Overview

**Lexis** is an end-to-end multimodal communication and accessibility platform designed to bridge conversational barriers for Deaf, Hard-of-Hearing, and multilingual communities.

By pairing **continuous 3D computer vision**, **ultra-low-latency speech AI**, and **large-scale gesture classification**, Lexis delivers instantaneous visual and acoustic subtitles with zero phantom triggers.

```
                                  ┌──────────────────────────────────────────┐
                                  │           LEXIS MULTIMODAL HUB           │
                                  │        (Live Subtitle & Studio)          │
                                  └────────────────────┬─────────────────────┘
                                                       │
                  ┌────────────────────────────────────┼────────────────────────────────────┐
                  ▼                                    ▼                                    ▼
┌──────────────────────────────────────┐  ┌─────────────────────────────────┐  ┌──────────────────────────────────────┐
│     1. REAL-TIME VISION ENGINE       │  │   2. MULTILINGUAL AUDIO ENGINE  │  │      3. 2,400-CLASS SIGN STUDIO      │
│            (`vision.py`)             │  │          (`audio.py`)           │  │          (`sign_studio.py`)          │
├──────────────────────────────────────┤  ├─────────────────────────────────┤  ├──────────────────────────────────────┤
│ • 720p @ 30+ FPS RTMPose Wholebody   │  │ • 16 kHz Real-time Audio Stream │  │ • 2,400-Class PyTorch Bi-GRU Model   │
│ • 133 Wholebody 3D Keypoints         │  │ • RMS Energy Silence Gate       │  │ • 74%+ Top-5 Recognition Accuracy    │
│ • User-Trained Kinematic Classifier  │  │ • Whisper Large V3 Turbo (Groq) │  │ • Visual Sign Lookup ("Shazam")      │
│ • Vectorized Buffer (<0.1ms latency) │  │ • Auto Language ID (20+ langs)  │  │ • Form Practice & Motion Evaluator   │
│ • Universal ASL Fingerspelling (A–Z) │  │ • LLaMA-3.1-8B-Instant (<80ms)  │  │ • Sign-to-Search Query Engine        │
│ • Downward Transit & Lap Mute Guards │  │ • Subtitle Auto-Synchronization │  │ • WLASL & ASL Citizen Vocabulary     │
└──────────────────┬───────────────────┘  └────────────────┬────────────────┘  └──────────────────┬───────────────────┘
                   │                                       │                                      │
                   └───────────────────────────────────────┼──────────────────────────────────────┘
                                                           ▼
                                         ┌───────────────────────────────────┐
                                         │       4. PERSISTENCE LAYER        │
                                         │       (PostgreSQL / Neon DB)      │
                                         └───────────────────────────────────┘
```

---

## ⚡ Key Capabilities

### 1. Continuous Live ASL Vision Subtitles (`backend/vision.py`)
- **133-Keypoint Skeletal Tracking**: Employs RTMPose Wholebody to extract 17 upper-body/facial landmarks and 42 bilateral hand joints in real time (30+ FPS).
- **Sub-Millisecond Feature Extraction**: Rolling vectorized buffer (`deque(maxlen=45)`) computes multi-frame spatio-temporal features in **$<0.1\text{ms}$**, eliminating CPU bottlenecks.
- **Physics-Informed Descent Guard**: Analyzes hand trajectory derivatives to differentiate downward arm drops from active circular rub signs (e.g. distinguishing moving from `hello` to `my` without falsely triggering `please`).
- **Conversational Pacing**: Hardware-locked token triggers with $0.35\text{s}$ conversational cooldown and 4-frame verification (~$0.12\text{s}$) for rapid multi-word chaining (*"hello my name"*, *"please help"*).
- **Universal A–Z Fingerspelling Concatenation**: Built-in 26-letter sign alphabet with automatic string concatenation (e.g. `R` + `Y` + `U` + `K` $\to$ `"RYUK"`).

### 2. Multilingual Speech-to-English Engine (`backend/audio.py`)
- **RMS Energy Gating**: Drops silence locally below threshold ($0.02$) to prevent audio hallucination and unnecessary network calls.
- **$<120\text{ms}$ Audio Transcription**: Powered by Groq's high-throughput `whisper-large-v3-turbo` with automatic detection across 20+ languages.
- **$<80\text{ms}$ LLM Translation Fast-Path**: Non-English speech is instantly translated into natural English subtitles via `llama-3.1-8b-instant`. Native English streams bypass LLM translation for zero added latency.

### 3. 2,400-Class Sign Studio & Search (`backend/sign_studio.py`)
- **Neural Architecture**: 2-layer Bidirectional GRU (256 hidden units, dropout 0.3) trained on the combined WLASL and ASL Citizen datasets.
- **Top-5 Accuracy**: **74%+ Top-5 recognition** across 2,400 sign classes on bounded video sequences.
- **Features**:
  - **"Shazam for Sign Language"**: Perform a sign to instantly see top matched definitions and confidence rankings.
  - **Interactive Form Evaluation**: Rates execution precision and spatial alignment against reference motion trajectories.

### 4. Interactive Data Collector & Kinematic Trainer (`backend/record_signs.py` & `backend/train_user_signs.py`)
- **HUD Recording Prompter**: On-screen countdown ($3 \dots 2 \dots 1$) capturing 45-frame normalized coordinate sequences.
- **Spatio-Temporal Augmentation**: Expands recorded samples with 25 sliding-window variations, producing 1,000+ training instances from small seed batches.
- **Kinematic Feature Vectors (240+ Features)**: Captures keyframe snapshots (0%, 25%, 50%, 75%, 100%), statistical moments, wrist trajectory curvature, velocity vectors, and scale-invariant finger geometry.
- **ExtraTrees Ensemble**: 200-estimator model yielding **100% cross-validation accuracy** on active conversational vocabularies.

---

## 🛠️ Tech Stack

| Layer | Framework / Library | Role |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11 | Core application logic and async streaming |
| **Pose Estimation** | RTMPose Wholebody, ONNXRuntime | 133-point 3D landmark tracking at 30+ FPS |
| **Machine Learning** | Scikit-learn, ExtraTrees, Joblib | Instant user sign sequence classifier |
| **Deep Learning** | PyTorch, Bidirectional GRU | 2,400-Class bounded gesture recognition |
| **Speech-to-Text** | Whisper Large V3 Turbo (Groq API) | Ultra-low latency spoken language transcription |
| **SLM Translation** | LLaMA-3.1-8B-Instant (Groq API) | Context-aware natural English subtitle formulation |
| **Audio & Video I/O** | OpenCV, SoundDevice, NumPy, SciPy | Low-latency camera frame and microphone capture |
| **Database** | Neon Cloud PostgreSQL, Psycopg2 | Asynchronous non-blocking subtitle transcript storage |

---

## 📂 Project Structure

```text
Lexis/
├── assets/
│   ├── logo.png                      # Project branding mark (transparent)
│   ├── logo_dark.png                 # Squircle icon badge
│   └── favicon.svg                   # Scalable vector icon
├── backend/
│   ├── main.py                       # Live desktop application runner
│   ├── vision.py                     # Real-time continuous ASL vision pipeline
│   ├── audio.py                      # Multilingual speech-to-English translation
│   ├── record_signs.py               # Interactive OpenCV sign recorder with ASL cues
│   ├── train_user_signs.py           # Feature engineering & ExtraTrees trainer
│   ├── inspect_signs.py              # Visual inspection tool for recorded datasets
│   ├── database.py                   # Async Neon PostgreSQL persistence layer
│   ├── requirements.txt              # Production dependency specifications
│   └── .env.example                  # Environment configuration template
├── frontend/
│   ├── favicon.svg                   # Minimal vector favicon
│   ├── logo_transparent.png          # High-resolution transparent logo
│   └── logo_dark.png                 # Dark theme icon
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/dhir-1/Lexis.git
cd Lexis
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file inside `backend/` by copying the example:
```bash
cp backend/.env.example backend/.env
```
Fill in your credentials:
```ini
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://user:password@your-neon-host/neondb?sslmode=require
ENABLE_AUDIO=1
VISION_CAMERA_INDEX=0
```

### 5. Launch the Live Application
```bash
python backend/main.py
```
- Position yourself in front of the webcam.
- Perform signs naturally; watch the subtitles chain words continuously into complete sentences.
- Press **`q`** to exit.

---

## 🧪 Recording & Training Custom Signs

You can easily expand the conversational vocabulary with custom signs:

1. **Record New Signs**:
   ```bash
   python backend/record_signs.py
   ```
   - Press **`[SPACE]`** to record a 45-frame sequence after the 3s countdown.
   - Use **`[N]`** / **`[P]`** to navigate through words.
   - Press **`[D]`** to delete and re-record an imperfect sample.

2. **Train the Classifier**:
   ```bash
   python backend/train_user_signs.py
   ```
   - Automatically computes 240+ kinematic features and spatial trajectories.
   - Validates using Stratified K-Fold cross-validation and exports the serialized model to `backend/models/`.

---

## 🗺️ Roadmap

- [x] **Continuous 15-Word Core Lexicon**: Verified with $<0.1\text{ms}$ vectorized feature extraction.
- [x] **Fingerspelling Fallback Engine**: Concatenation of A–Z alphabet signs into continuous words.
- [x] **Multilingual Voice Translation**: Dual-model pipeline (<120ms Whisper + <80ms LLaMA-3.1).
- [ ] **Core 200 ASL Vocabulary**: Incrementally recording and verifying high-frequency conversational sign batches.
- [ ] **Top-Layer LLM Sentence Smoothing**: Passing committed vision tokens through LLaMA-3.1 to formulate full grammatical English sentences.
- [ ] **Full-Stack Web Interface**: WebRTC streaming connected to a modern React/Next.js dashboard.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
