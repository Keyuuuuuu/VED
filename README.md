# 🎙️ VED (Voice Emotion Detective)

**State-of-the-Art Speech Emotion Recognition (SER) on CREMA-D Dataset**

## 📖 Introduction

**VED (Voice Emotion Detective)** is a comprehensive research project dedicated to Speech Emotion Recognition (SER). The project systematically explores the path from lightweight, end-to-end deep learning models to large-scale pre-trained transfer learning strategies.

Using the **CREMA-D** dataset (~7,442 samples), we conducted rigorous iterative experiments (CNN → Conformer → Wav2Vec 2.0 → WavLM). The final architecture achieves **SOTA-level performance (Macro F1: 0.7559)**, effectively solving common SER challenges such as distinguishing "Sadness" from "Neutral" and identifying high-frequency "Fear" cues.

## 🏆 Key Results

| Architecture | Input Features | Strategy | Best Macro F1 | Verdict |
| --- | --- | --- | --- | --- |
| **WavLM Ultimate (Stage 3)** | Raw Audio | **Layer-wise Weighted + BiLSTM + Attn + 5-Fold** | **0.7559** | **🏆 SOTA / Final Best** |
| **Conformer V3.1 (Stage 1)** | Log-Mel | Mixup + Label Smoothing | **0.6707** | Best Lightweight Model |
| **Wav2Vec 2.0 (Stage 3)** | Raw Audio | Fine-tuning + BiLSTM | 0.6004 | Strong on Anger, weak on Sadness |
| **CNN / Transformer (Stage 0)** | Log-Mel | Baseline | < 0.57 | Insufficient feature extraction |

---

## 📂 Project Structure

The project is organized into stages representing the evolution of our methodology:

```text
VED/
├── AudioWAV/                  # Dataset Directory (CREMA-D .wav files)
├── Output/                    # Training logs, checkpoints, confusion matrices
├── wav2vec2_local/            # Local pre-trained weights for Wav2Vec 2.0
├── wavlm_base_plus_local/     # Local pre-trained weights for WavLM Base+
│
├── Stage 0/                   # 🟢 Initial Phase: Baselines
│   ├── VED-CNN-Transfomer.ipynb
│   └── voiceEmotionDetective.ipynb  # EDA and initial analysis
│
├── Stage 1/                   # 🔵 Advanced Phase: Conformer Optimization (Log-Mel)
│   ├── ConformerV1.ipynb      # Baseline Conformer
│   └── ConformerV3.1.ipynb    # 🔥 Best Log-Mel Model (Balanced Regularization)
│
├── Stage 2/                   # 🟠 Exploration Phase: Complex Architectures
│   ├── ConformerV3.2.ipynb    # Large Kernel (Optimized for Sadness/Fear)
│   └── DoubelStreamV4.ipynb   # Multi-modal Fusion (Log-Mel + eGeMAPS)
│
├── Stage 3/                   # 🔴 Final Phase: Pre-trained Transfer Learning
│   ├── Wav2Vec2.ipynb         # Wav2Vec 2.0 Fine-tuning
│   └── WavLM_V2.2.ipynb       # 🔥 Final SOTA Model (WavLM + 5-Fold CV)
│
├── requirements.txt           # Project Dependencies
└── README.md                  # Documentation

```

---

## 🧠 Methodology Evolution

### Phase 1: Scratch Training (Lightweight)

We initially explored lightweight architectures using **Log-Mel Spectrograms** as input.

* **Challenge**: The small dataset size (only ~6 hours of audio) caused Transformer models to overfit rapidly.
* **Solution (Conformer V3.1)**:
* Implemented **Mixup** data augmentation, which significantly improved recall for "Fear" and "Disgust".
* Tuned **Label Smoothing (0.05)** and **Dropout (0.3)** to find the perfect balance between fitting and generalization.



### Phase 2: Pre-trained Transfer Learning (Heavyweight)

To break the 0.67 F1 ceiling, we leveraged Hugging Face pre-trained models (Wav2Vec 2.0 / WavLM) with **Raw Waveform** input.

* **The Winner: WavLM Ultimate Architecture**
1. **Backbone**: Used `microsoft/wavlm-base-plus`.
2. **Layer-wise Weighted Sum**: Instead of using only the last layer, we learned a weighted sum of all 12 transformer layers. This recovered **prosody information** often discarded by the final semantic layers.
3. **Head Design**: Introduced **Bi-LSTM** to capture long-term temporal dependencies (solving the low recall for "Sadness") and **Attention Pooling** to focus on emotional peaks.
4. **Loss Function**: Implemented **Weighted Focal Loss** to penalize hard examples and boost recall for "Happy" samples.
5. **Validation**: Employed **5-Fold Cross-Validation** to ensure robust and reliable metrics.



---

## 📊 Performance Visualization

### 1. Confusion Matrix (WavLM 5-Fold)

*Achieved high accuracy across all classes, including the difficult "Sadness" vs. "Neutral" distinction.*

*(Note: Please ensure the image path matches your generated output)*

### 2. Learned Layer Weights

*The model automatically learned to prioritize top layers for emotion recognition, while still utilizing lower-level acoustic features.*

---

## 🚀 Quick Start

### 1. Installation

```bash
# Python 3.8+ recommended
pip install -r requirements.txt

```

*Dependencies include PyTorch (CUDA support), Transformers, Librosa, and Datasets.*

### 2. Data & Model Setup

Due to server network restrictions, this project supports **offline loading** of pre-trained models.

1. Place CREMA-D `.wav` files into the `AudioWAV/` directory.
2. Ensure `wavlm_base_plus_local/` contains the Hugging Face model files (`config.json`, `pytorch_model.bin`, `preprocessor_config.json`).

### 3. Training

**To reproduce the SOTA results (Stage 3):**
Open `Stage 3/WavLM_V2.2.ipynb` (or the final provided script) and run all cells. The script handles:

* Data loading & On-the-fly augmentation
* 5-Fold Cross-Validation training
* Generation of evaluation reports and plots

**To run the lightweight model (Stage 1):**
Open `Stage 1/ConformerV3.1.ipynb` for Log-Mel feature extraction and training.

---

## 📝 Acknowledgments

* **Dataset**: [CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
* **Pre-trained Models**: Microsoft WavLM and Facebook Wav2Vec 2.0 via Hugging Face.

---

**Author**: Elios
**Last Update**: December 2025
