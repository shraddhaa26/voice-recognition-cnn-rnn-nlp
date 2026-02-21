# 🎤 Voice Recognition System — CNN-RNN + NLP

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/voice-recognition-cnn-rnn-nlp/blob/main/notebooks/Voice_Recognition_CNN_RNN_NLP.ipynb)

> A complete voice command recognition system using **CNN-RNN hybrid deep learning** with **NLP post-processing**. No external audio files needed — everything is auto-generated!

---

## ✨ Features

- 🎵 **Auto-Generated Dataset** — 800 synthetic voice samples (no downloads!)
- 🧠 **CNN-RNN Hybrid Model** — Conv2D + Bidirectional LSTM
- 📝 **NLP Post-Processing** — Action mapping, context tracking, response generation
- 📊 **Complete Visualizations** — Waveforms, spectrograms, confusion matrix
- 🎮 **10 Voice Commands** — yes, no, up, down, left, right, stop, go, hello, help

---

## 🚀 Quick Start

### Option 1: Google Colab (Easiest)
Click the "Open in Colab" badge above!

### Option 2: Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/voice-recognition-cnn-rnn-nlp.git
cd voice-recognition-cnn-rnn-nlp
pip install -r requirements.txt
python main.py
```

---

## 🏗️ Architecture

```
Audio (.wav) → MFCC Features → CNN (Conv2D×3) → BiLSTM (×2) → Dense → NLP
```

### Model Details

| Layer | Type | Details |
|-------|------|---------|
| 1-3 | Conv2D | 32→64→128 filters, BatchNorm, MaxPool, Dropout |
| 4-5 | BiLSTM | 128→64 units, dropout=0.3 |
| 6-7 | Dense | 128→64 units, BatchNorm, Dropout |
| 8 | Output | Softmax (10 classes) |

### NLP Post-Processing

| Feature | Description |
|---------|-------------|
| Confidence Filter | Rejects predictions below 40% |
| Action Mapping | `up` → `MOVE_UP` (direction category) |
| Context Tracking | Detects navigation sequences |
| Response Generation | Natural language feedback |

---

## 📊 Dataset

| Command | Samples | Total |
|---------|---------|-------|
| yes, no, up, down, left, right, stop, go, hello, help | 80 each | **800** |

- Format: WAV, 16kHz, 1 second
- Auto-generated with unique spectral signatures per command

---

## 📁 Project Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── main.py                    # Run complete pipeline
├── notebooks/
│   └── Voice_Recognition_CNN_RNN_NLP.ipynb  # Jupyter notebook
├── src/
│   └── __init__.py
├── models/                    # Saved models (auto-generated)
├── data/                      # Dataset (auto-generated)
└── outputs/                   # Plots (auto-generated)
```

---

## 📈 Results

- Training Accuracy: ~95%+
- Test Accuracy: ~85-95%
- 10 voice commands classified with NLP-enhanced output

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** — Deep learning model
- **Librosa** — Audio processing & MFCC extraction
- **NLTK** — Natural language processing
- **Scikit-learn** — Evaluation metrics
- **Matplotlib/Seaborn** — Visualizations

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

⭐ **Star this repo if you find it useful!**
