# 🍃 Tea Leaf Disease Classification with a CNN from Scratch

A convolutional neural network built from scratch in **PyTorch** to classify tea leaf images from Bangladesh into **4 classes** (three disease/pest conditions and healthy leaves), followed by a set of controlled **hyperparameter experiments** analysing how learning rate, batch size, dropout, data augmentation, and image size affect performance.

> Course project for **CVPR (Computer Vision and Pattern Recognition)**. The model and training code follow the structure of the course's *"Cats vs Dogs — a first CNN with PyTorch"* notebook, adapted to a new dataset.

---

## 📂 Dataset

| | |
|---|---|
| **Name** | Tea Leaf Disease and Pest-Damage Image Dataset (Bangladesh) |
| **Source** | Mendeley Data — [https://data.mendeley.com/datasets/3x42rbj8yv/1](https://data.mendeley.com/datasets/3x42rbj8yv/1) |
| **DOI** | [10.17632/3x42rbj8yv.1](https://doi.org/10.17632/3x42rbj8yv.1) |
| **Paper** | Chowdhury & Arman, *Data in Brief* 69 (2026) 113356 — [doi:10.1016/j.dib.2026.113356](https://doi.org/10.1016/j.dib.2026.113356) |
| **License** | CC BY 4.0 |
| **Total images** | 2,500 (original, no pre-augmentation) |
| **Original size** | ~3.31 GB, full-resolution smartphone photos (up to 4608 × 3456) |

**Description:** Field-collected smartphone images of tea leaves from four tea gardens in Chunarughat, Habiganj, Bangladesh, captured under natural lighting with real backgrounds.

### Classes

| Class | Images | Train | Validation |
|---|---|---|---|
| Blight | 555 | 444 | 111 |
| Healthy_Leaf | 650 | 520 | 130 |
| Helopeltis (pest damage) | 675 | 540 | 135 |
| Red_Rust | 620 | 496 | 124 |
| **Total** | **2,500** | **2,000** | **500** |

### Why this dataset?
- **Recent and original:** published in 2026, containing only original (non-augmented) images, so accuracy results are trustworthy and augmentation experiments are fair.
- **Real-world relevance:** tea is a major crop in Bangladesh, and early disease detection matters to farmers.
- **Meaningful challenge:** field backgrounds, varying light, and visually similar classes (e.g., Blight vs Red_Rust) make it a realistic task for a from-scratch CNN.
- **Fits project requirements:** 2,500 images across 4 nearly balanced classes.

---

## ⚙️ Preprocessing

1. **Stratified 80/20 split** per class (`random.seed(0)`), so every class has the same train/validation ratio.
2. **One-time resize to 256 × 256** and save, to avoid decoding huge photos every epoch and to bring the dataset well under the size limit.
3. **EXIF orientation fix** (`ImageOps.exif_transpose`) so phone photos are upright.
4. **RGB conversion** to guarantee 3 input channels.
5. At load time, images are resized to `IMG_SIZE` (128 by default) and converted to tensors.

**Augmentation (training set only):** random horizontal flip, random rotation (±15°), and colour jitter (brightness and contrast ±0.2). The validation set is never augmented.

---

## 🧠 Model Architecture

```
Input                     3 × 128 × 128
Conv(3→16, 3×3, pad 1) → ReLU → MaxPool(2)   → 16 × 64 × 64
Conv(16→32, 3×3, pad 1) → ReLU → MaxPool(2)  → 32 × 32 × 32
Conv(32→64, 3×3, pad 1) → ReLU → MaxPool(2)  → 64 × 16 × 16
Flatten                                       → 16,384
Dropout(p)
Linear(16,384 → 128) → ReLU
Linear(128 → 4)                               → class scores
```

- **Trainable parameters:** 2,121,380 (≈ 99% in the first fully connected layer)
- **Conv stride:** 1 · **Pool stride:** 2 · **Padding:** 1 (keeps size until pooling)

### Training setup

| Setting | Value |
|---|---|
| Loss | Cross-entropy |
| Optimizer | Adam |
| Learning rate | 0.001 (fixed, no scheduler) |
| Batch size | 32 |
| Epochs | 15 (Round 1), 30 (Round 2 and experiments) |
| Regularization | Dropout 0.5 + data augmentation |
| Hardware | Google Colab, NVIDIA T4 GPU |

---

## 📊 Results

### Round 1 vs Round 2

| | Round 1 (no augmentation, no dropout, 15 epochs) | Round 2 (augmentation + dropout 0.5, 30 epochs) |
|---|---|---|
| Final train accuracy | 99.8% | 87.5% |
| Best validation accuracy | 86.6% | **87.8%** |
| Lowest validation loss | 0.584 (epoch 7, then rising) | **0.399** |
| Overfitting | Strong (train → ~100%, val loss rising) | Controlled (train and val curves close) |

### Best model (Round 2) — per-class performance

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Blight | 0.96 | 0.78 | 0.86 |
| Healthy_Leaf | 0.88 | 0.92 | 0.90 |
| Helopeltis | 0.94 | 0.87 | 0.90 |
| Red_Rust | 0.78 | 0.93 | 0.85 |
| **Accuracy** | | | **0.88** |

**Main confusion:** Blight ↔ Red_Rust, both showing brownish damage. Regularization reduced Red_Rust → Blight errors (19 → 4) but increased Blight → Red_Rust errors (12 → 20), so the confusion shifted direction rather than disappearing.

---

## 🔬 Hyperparameter Experiments

Each experiment changes **one** hyperparameter from the baseline (Round 2 settings) and starts from the same random seed (`torch.manual_seed(0)`).

| # | Experiment | Best val acc | Best epoch | Min val loss | Final train acc | Final gap |
|---|---|---|---|---|---|---|
| 0 | **Baseline** (lr 0.001, batch 32, dropout 0.5, aug, 128 px) | 0.854 | 26 | **0.433** | 0.908 | 0.067 |
| 1 | Learning rate 0.01 | 0.270 | 1 | 1.384 | 0.270 | 0.000 |
| 2 | Learning rate 0.0001 | 0.752 | 29 | 0.613 | 0.739 | 0.007 |
| 3 | Batch size 16 | 0.872 | 29 | 0.468 | 0.897 | 0.037 |
| 4 | Batch size 64 | 0.846 | 25 | 0.445 | 0.858 | 0.038 |
| 5 | Dropout 0.2 | 0.876 | 26 | 0.442 | 0.926 | 0.058 |
| 6 | No augmentation (dropout 0.5) | 0.874 | 20 | 0.484 | 0.983 | **0.127** |
| 7 | Image size 64 | 0.832 | 28 | 0.470 | 0.836 | 0.027 |
| 8 | Image size 192 | 0.864 | 30 | 0.445 | 0.912 | 0.048 |

*Final gap = final train accuracy − final validation accuracy (an overfitting measure).*

> **Note on noise:** Repeating the baseline settings gave best validation accuracies between 85.4% and 87.8%, so differences of about ±2% are within run-to-run variation and should not be treated as real effects.

### Key findings

| Hyperparameter | Effect | Strength |
|---|---|---|
| **Learning rate** | 0.01 collapsed (always predicted the majority class, loss ≈ ln 4); 0.0001 underfit; 0.001 best | **Very strong** |
| **Data augmentation** | Removing it caused strong overfitting (train 98%, val loss rose from 0.484 to 0.778) even with dropout | **Strong** — main defence against overfitting |
| **Image size** | 64 px slightly worse; 192 px no clear gain while making the model 2.2× larger | Small — 128 px is a good balance |
| **Batch size** | 16, 32, and 64 within noise; batch 64 learned more slowly | Small |
| **Dropout** | 0.2 vs 0.5 within noise when augmentation is on | Small |

**Insight:** accuracy alone can hide overfitting. Experiment 6 reached a competitive best accuracy (87.4%), but its validation loss rose sharply, revealing an overconfident, memorizing model.

---

## 🔍 Visualizations in the Notebook

- Sample training images per class
- Loss and accuracy curves for every run
- Learned conv1 filters
- Feature maps from conv1 (edges, colour) and conv3 (abstract, low-resolution)
- Confusion matrices and classification reports (Round 1 and Round 2)
- Same image under 5 random augmentations
- Comparison chart of all experiments with a ±2% noise band

---

## ⚠️ Limitations

- **Small validation set (500 images):** a few predictions changing moves accuracy by 1–2%, making curves noisy.
- **Single run per experiment:** no repeated runs or confidence intervals.
- **No separate test set:** the 20% hold-out set served as both validation and test.
- **Background shortcuts:** feature maps showed strong responses to backgrounds, so the model may partly rely on context rather than symptoms.
- **Simple architecture:** three conv layers with 3 × 3 filters may not exploit fine detail at larger image sizes.

## 🚀 Future Work

- Learning-rate scheduling (e.g., cosine annealing) to stabilise late-epoch fluctuations
- A separate train / validation / test split and repeated runs
- Deeper models or transfer learning (e.g., ResNet, MobileNet)
- Background removal or leaf segmentation to reduce context shortcuts
- Grad-CAM to visualise which leaf regions drive predictions

---

## ▶️ How to Run

1. Download the dataset zip from [Mendeley Data](https://data.mendeley.com/datasets/3x42rbj8yv/1) and upload it to Google Drive.
2. Open `tea-leaf-disease-image.ipynb` in **Google Colab**.
3. Set **Runtime → Change runtime type → T4 GPU**.
4. Update the zip path in the first cell if your Drive folder differs:
   ```python
   !unzip -q "/content/drive/MyDrive/CVPR Assignment/Tea Leaf Disease Image.zip" -d /content/tea_raw
   ```
5. Run all cells from top to bottom.

### Requirements
```
torch
torchvision
numpy
matplotlib
scikit-learn
pandas
Pillow
```
(All pre-installed in Google Colab.)

---

## 📁 Repository Structure

```
├── tea-leaf-disease-image.ipynb   # full notebook: data prep, CNN, training, experiments
└── README.md
```

---

## Acknowledgements

- **Dataset:** Chowdhury & Arman, *A field-acquired tea leaf image dataset for disease and pest-damage classification in Bangladesh*, Data in Brief 69 (2026) 113356. Licensed under CC BY 4.0.
