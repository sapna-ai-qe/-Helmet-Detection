# ⛑️ HelmNet — Helmet Detection using Computer Vision
### CNN | Transfer Learning | TensorFlow | Image Classification | Worker Safety

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Computer%20Vision%20%7C%20Safety-red)

---

## 🏗️ Business Context

Workplace safety in hazardous environments like **construction sites and industrial plants** is critical. One of the most important safety measures is ensuring workers wear safety helmets to prevent head injuries from falling objects and machinery.

**Manual helmet compliance monitoring** is prone to errors, costly at scale, and impossible to implement in real time across large operations.

**SafeGuard Corp** wants to automate this process using AI-powered computer vision to detect helmet compliance instantly and accurately.

---

## 🎯 Objective

> Develop an **image classification model** that automatically classifies images into:
> - ✅ **With Helmet** — Worker wearing a safety helmet
> - ❌ **Without Helmet** — Worker not wearing a safety helmet

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Images | 631 |
| With Helmet | 311 images |
| Without Helmet | 320 images |
| Image Shape | 200 x 200 x 3 (RGB) |
| Environments | Construction sites, factories, industrial settings |
| Conditions | Varied lighting, angles, and worker postures |

**Dataset Characteristics:**
- Diverse real-world environments: construction sites, factories, industrial settings
- Variations in lighting conditions, camera angles, and worker postures
- Workers depicted in different activities: standing, using tools, moving
- Balanced binary classification (near 50-50 split)

---

## 🔬 Approach & Methodology

```
Images → Preprocessing → Augmentation → CNN Architecture → Transfer Learning → Training → Evaluation
```

### 1. Data Loading & Preprocessing
- Loaded 631 RGB images of shape 200x200x3
- Normalized pixel values to [0, 1] range
- Set random seeds for reproducibility across NumPy, TensorFlow backend, and Python
- Train/validation/test split for robust evaluation

### 2. Exploratory Data Analysis
- Visualized random samples from each class
- Verified class distribution (311 vs 320 — near-balanced)
- Analyzed image quality variations across environments

### 3. Data Augmentation
Applied augmentation strategies to improve model robustness and generalization:
- **Rotation** — Random rotations up to 20 degrees
- **Horizontal Flip** — Mirror images for invariance
- **Zoom** — Random zoom in/out
- **Brightness Adjustment** — Handle varied lighting conditions
- **Width/Height Shifts** — Handle off-center subjects

### 4. Model Architecture

**Custom CNN (Baseline):**
```
Input (200x200x3)
→ Conv2D(32) + BatchNorm + MaxPool + Dropout
→ Conv2D(64) + BatchNorm + MaxPool + Dropout
→ Conv2D(128) + BatchNorm + MaxPool + Dropout
→ Flatten
→ Dense(256) + Dropout(0.5)
→ Output Dense(1) + Sigmoid
```

**Transfer Learning (Final Model):**
- Used pre-trained model weights as base (trained on large image datasets)
- Froze base layers to preserve learned features
- Added custom classification head for binary helmet detection
- Fine-tuned top layers on the helmet dataset

### 5. Training
- **Optimizer:** Adam with learning rate scheduling
- **Loss:** Binary Crossentropy
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Monitored both training and validation accuracy to detect overfitting

### 6. Evaluation
- Confusion matrix analysis
- Precision, Recall, F1-Score per class
- ROC curve and AUC score
- Visual analysis of misclassified images

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Test Accuracy | **100%** |
| Model Approach | Transfer Learning + Custom Head |
| Key Challenge | Limited training data (631 images) — solved with data augmentation |

---

## 💡 Business Insights & Recommendations

1. **Helmet Compliance Mapping** — Analyze visual data to identify locations and time slots with consistently low compliance, enabling targeted safety interventions
2. **Real-time Monitoring** — Deploy model on CCTV feeds for instant non-compliance alerts to site supervisors
3. **Safety Reporting** — Use model output to generate automated compliance reports, reducing manual audit effort
4. **High-Risk Zone Focus** — Prioritize monitoring in identified high-risk areas like machinery zones and elevated platforms
5. **Training Feedback** — Use misclassified images to improve safety training materials and identify edge-case compliance scenarios

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| TensorFlow 2.x | Deep learning framework |
| Keras | Neural network building and training |
| OpenCV (cv2) | Image loading and preprocessing |
| NumPy | Numerical operations on image arrays |
| Matplotlib / Seaborn | Visualization and result plotting |
| Google Colab (GPU) | Model training with GPU acceleration |

---

## 📁 Project Structure

```
helmet-detection/
│
├── Sapna_HelmNet_Full_Code.ipynb   # Full notebook: EDA, model, training, evaluation
├── README.md                        # Project documentation
└── data/                            # Dataset (631 images - not included)
    ├── with_helmet/                 # 311 images
    └── without_helmet/              # 320 images
```

---

## 🚀 How to Run

1. Clone this repository
2. Open `Sapna_HelmNet_Full_Code.ipynb` in Google Colab (GPU recommended)
3. Upload dataset images to Google Drive
4. Update the dataset path in the notebook
5. Run all cells sequentially

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn pandas scikit-learn
```

---

## 👩‍💻 Author

**Sapna** | Senior AI Quality Engineer  
Post Graduate in AI/ML — University of Texas at Austin  
GitHub: [@sapna-ai-qe](https://github.com/sapna-ai-qe)

---
*Part of AI/ML Portfolio — UT Austin Post Graduate Program*
