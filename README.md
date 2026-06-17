# 🎭 Deepfake Detection via Vision Transformers
### A Comparative Study: SVM · CNN · CNN-LSTM · ViT

> **B.Tech Major Project** — Manipal University Jaipur (2025)  
> Authors: [Anoushka Rakheja](https://github.com/anoushka-r) & Rimjhim Sharma  
> Supervisor: Dr. Ghanshyam Raghuwanshi, Associate Professor

---

## 🏆 Results at a Glance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| SVM (baseline) | 36.36% | 0.36 | 0.36 | 0.36 |
| CNN-LSTM | 50.00% | 0.50 | 0.50 | 0.49 |
| CNN (ResNet-18) | 82.08% | 0.77 | 0.91 | 0.83 |
| **ViT (Ours)** | **88.68%** | **1.00** | **0.77** | **0.87** |

> **ViT achieves perfect precision (1.00)** on real-class detection — zero false positives — while outperforming CNN by +6.6% accuracy.

---

## 📌 Overview

Deepfakes — AI-generated synthetic videos that convincingly mimic real people — pose a growing threat to digital trust and media integrity. Existing detection methods struggle to generalise across manipulation types or adapt to evolving generative models.

This project proposes a **Vision Transformer (ViT)-based detection pipeline** that:
- Divides each video frame into fixed-size patches and processes them through transformer encoder blocks
- Uses **global self-attention** to capture facial inconsistencies dispersed across the entire frame — something CNNs with local receptive fields miss
- Aggregates frame-level predictions to produce a **video-level real/fake decision**

---

## 🧠 Why ViT Beats CNN for Deepfake Detection

| Challenge | CNN | ViT (Our Approach) |
|-----------|-----|-------------------|
| Feature scope | Local receptive fields only | Global self-attention across all patches |
| Temporal modelling | Requires LSTM pairing | Transformer handles sequence natively |
| Generalisation | Dataset-specific, overfits to local artifacts | Learns global, adaptive representations |
| Robustness | Degrades under compression/noise | Resilient to adversarial scenarios & resolution variation |

---

## 📂 Repository Structure

```
deepfake-detection-vit/
│
├── vit.ipynb              # 🌟 Proposed model — Vision Transformer pipeline
├── cnn.ipynb              # CNN baseline (ResNet-18)
├── cnn_lstm.ipynb         # CNN-LSTM hybrid baseline
├── basic_svm.ipynb        # SVM baseline with handcrafted features
│
├── paper/
│   └── Deepfake_Detection_ViT_MUJ_2025.pdf   # Full research paper
│
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

We used the **SDFVD (Small-scale Deepfake Forgery Video Dataset)**:
- 106 videos total — 53 real, 53 deepfake
- Diverse contexts, genders, ages, and backgrounds
- Deepfakes generated via face-swapping (Remaker AI)
- Videos pre-cropped to 4–5 seconds, resized to 720p

📥 **Download:** [SDFVD on Mendeley Data](https://data.mendeley.com/datasets/bcmkfgct2s/1)

After downloading, organise the dataset as:
```
dataset/
├── videos_real/     # .mp4 files
└── videos_fake/     # .mp4 files
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/anoushka-r/deepfake-detection-vit.git
cd deepfake-detection-vit

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
torch
torchvision
transformers
opencv-python
scikit-learn
numpy
matplotlib
pandas
tqdm
Pillow
```

---

## 🚀 Run the Models

Open any notebook in Jupyter and update the dataset path to your local directory:

```python
# In each notebook, update this line:
dataset_path = "path/to/your/dataset"
```

Then run all cells top to bottom. The ViT notebook will also prompt you to test on a custom video at the end.

| Notebook | Model | Run time (approx.) |
|----------|-------|--------------------|
| `vit.ipynb` | Vision Transformer | ~15–20 min (GPU) |
| `cnn.ipynb` | CNN (ResNet-18) | ~5 min |
| `cnn_lstm.ipynb` | CNN + LSTM | ~10 min |
| `basic_svm.ipynb` | SVM | ~1–2 min |

---

## 🔬 ViT Architecture Detail

```
Input Video
    ↓
Frame Extraction (uniform sampling)
    ↓
ViTImageProcessor — resize + normalise each frame
    ↓
Patch Embedding — each frame split into 16×16 patches
    ↓
Transformer Encoder — multi-head self-attention × N layers
    ↓
[CLS] Token → Binary Classifier (Real / Fake)
    ↓
Frame-level predictions → aggregated → Video-level decision
```

**Pretrained backbone:** `google/vit-base-patch16-224` (HuggingFace)  
**Fine-tuned on:** SDFVD dataset, 10 epochs, Adam optimiser (lr=2e-5)  
**Hardware:** CUDA GPU

---

## 📊 Confusion Matrix (ViT)

```
              Predicted
              Real   Fake
Actual Real  [ 53     0  ]   ← Perfect — zero false positives
Actual Fake  [ 12    41  ]
```

---

## 📄 Research Paper

> *Harnessing Vision Transformers for Robust Deepfake Detection: A Comparative Perspective*  
> Anoushka Rakheja, Rimjhim Sharma — Manipal University Jaipur, 2025

Full paper available in the `/paper` directory.

---

## 👩‍💻 Authors

**Anoushka Rakheja**  
B.Tech CCE, Manipal University Jaipur  
[LinkedIn](https://www.linkedin.com/in/anoushka-rakheja) · [GitHub](https://github.com/anoushka-r)

**Rimjhim Sharma**  
B.Tech CCE, Manipal University Jaipur

---

## 📜 License

This project is for academic and research purposes.  
Please cite the paper if you use this work.
