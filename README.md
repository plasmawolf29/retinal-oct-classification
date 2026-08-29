# Retinal OCT Pathology Classification: Classical Ensembles vs. Deep Convolutional Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20(CUDA)-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MedMNIST%20OCTMNIST-brightgreen.svg)](https://medmnist.com/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20GTX%201660%20Ti-76B900.svg)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end biomedical machine learning and computer vision project for multi-class diagnosis of retinal diseases from Optical Coherence Tomography (OCT) B-scans. This repository benchmarks classical statistical machine learning pipelines (PCA dimensionality reduction, 66-D domain feature engineering, soft voting, and multi-stage stacking) against deep regularized convolutional neural networks accelerated with CUDA GPU computing.

---

## 🚀 Engineering Progression & Versioning Story

This project highlights an iterative engineering progression from an initial undergraduate course submission on cloud notebooks to a locally accelerated, production-grade machine learning pipeline:

```
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│   v1.0: Academic Baseline (Colab)      │       │   v2.0: Modernized GPU Pipeline        │
├────────────────────────────────────────┤       ├────────────────────────────────────────┤
│ • Cloud-based execution (Google Colab) │  ───► │ • Local high-performance environment   │
│ • Initial exploration of shallow &     │       │ • PyTorch CUDA GPU acceleration (18x)  │
│   deep models                          │       │ • 66-D Custom Biomedical Feature Extr. │
│ • Flat 784-D raw pixel inputs          │       │ • Fixed loop refitting & namespaces    │
│ • Basic validation set reporting       │       │ • Full 4-class test evaluation suite   │
│ • Baseline test accuracy: ~51%         │       │ • Test Macro ROC AUC: 0.962 (Acc: 73%) │
└────────────────────────────────────────┘       └────────────────────────────────────────┘
```

### Version Comparison in this Repository:
* 📄 [`v1_original_colab_submission.ipynb`](./v1_original_colab_submission.ipynb): Historical baseline notebook as originally developed for the UConn Biomedical Machine Learning course on Google Colab.
* 🚀 [`v2_improved_gpu_pipeline.ipynb`](./v2_improved_gpu_pipeline.ipynb): Modernized, GPU-accelerated pipeline featuring custom feature extraction, bug-free vectorized training, and complete diagnostic evaluations.

---

## 📌 Clinical Problem & Pathologies

Optical Coherence Tomography (OCT) is a non-invasive optical imaging modality that captures cross-sectional microstructural details of the retina. This system classifies scans into four distinct clinical categories from the **MedMNIST OCTMNIST** benchmark (109,309 total scans):

| Class | Pathology | Clinical Manifestation |
| :---: | :--- | :--- |
| **0** | **Choroidal Neovascularization (CNV)** | Neovascular vessel growth beneath the retina causing severe exudation and macular degeneration. |
| **1** | **Diabetic Macular Edema (DME)** | Retinal thickening and fluid accumulation resulting from diabetic retinopathy. |
| **2** | **Drusen (DRUSEN)** | Extracellular lipid/protein deposits beneath the retinal pigment epithelium; precursor to AMD. |
| **3** | **Normal (NORMAL)** | Healthy retinal morphology with intact foveal depression and stratified layers. |

---

## 🏗️ Architectural Taxonomy

### Pipeline 1: Classical Machine Learning (Scikit-Learn)
1. **Design 1A (Soft Voting Ensemble)**: Calibrated soft voting over `HistGradientBoosting`, `RandomForestClassifier`, and `KNeighborsClassifier`.
2. **Design 1B (Dimensionality Reduction + Stacking)**: PCA retaining 90% cumulative variance (21 components) coupled with a `StackingClassifier` (Random Forest + HistGradientBoosting -> Logistic Regression meta-learner).
3. **Design 1C (Custom Engineered Features)**: 66-dimensional domain feature extractor trained with `HistGradientBoostingClassifier`.
4. **Design 1D (Optimized Full-Dataset HGB)**: Tuned gradient boosting on raw pixel vectors.

### Pipeline 2: Deep Artificial Neural Networks (PyTorch on CUDA)
1. **Design 2A (Deep MLP)**: Multi-layer Dense network ($66 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 4$) with Batch Normalization and Dropout.
2. **Design 2B (Baseline CNN1)**: 3-stage Conv2D network ($32 \rightarrow 64 \rightarrow 128$ filters) with MaxPooling and Dropout.
3. **Design 2C (Deep Regularized CNN2 - Optimal)**: Multi-block ConvNet with $3 \times 3$ kernels, Batch Normalization, Dropout ($p=0.25-0.4$), Global Average Pooling, and learning rate scheduling.

---

## 🔬 66-D Biomedical Feature Extractor

To satisfy domain interpretability requirements without deep learning, we engineered 66 morphological and spatial features:
* **16 Regional Grid Means**: Average pixel intensity across $4 \times 4$ sub-regions ($7 \times 7$ blocks) to capture localized fluid pockets.
* **16 Regional Grid Standard Deviations**: Sub-region intensity spread to capture local texture heterogeneity.
* **14 Vertical Depth Projection Profiles**: Mean row intensities capturing layer stratification along the depth axis.
* **14 Horizontal Symmetry Profiles**: Mean column intensities capturing horizontal scan symmetry.
* **6 Global Morphological Descriptors**: Overall mean, contrast (std), central foveal intensity, central foveal std, and horizontal/vertical edge gradient energy.

---

## 📊 Benchmark Results (Balanced Test Set: 1,000 Scans)

| Pipeline | Model Architecture | Feature Representation | Test Accuracy | Balanced Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC AUC (Macro OvR) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline 2** | **Design 2C: Deep CNN2 (Optimal)** | **28x28x1 Image (GPU)** | **73.00%** | **73.00%** | **80.85%** | **73.00%** | **69.06%** | **0.9619** |
| **Pipeline 2** | Design 2B: Baseline CNN1 | 28x28x1 Image (GPU) | 70.80% | 70.80% | 75.70% | 70.80% | 66.84% | 0.9412 |
| **Pipeline 2** | Design 2A: Deep MLP | 66 Custom Features (GPU) | 60.80% | 60.80% | 49.08% | 60.80% | 52.58% | 0.8925 |
| **Pipeline 1** | Design 1D: Tuned HGB (Full) | Flat Pixels (784-D) | 57.30% | 57.30% | 63.82% | 57.30% | 49.23% | 0.8747 |
| **Pipeline 1** | Design 1C: 66 Custom Features HGB | 66 Custom Features | 54.70% | 54.70% | 49.09% | 54.70% | 46.85% | 0.8633 |
| **Pipeline 1** | Design 1A: Soft Voting Ensemble | Flat Pixels (784-D) | 51.30% | 51.30% | 61.23% | 51.30% | 41.16% | 0.8292 |
| **Pipeline 1** | Design 1B: PCA (90%) + Stacking | PCA Components (21-D) | 46.30% | 46.30% | 42.76% | 46.30% | 37.27% | 0.8016 |

---

## ⚡ Hardware Acceleration & Speedup

Training all 97,477 training scans on an **NVIDIA GeForce GTX 1660 Ti** reduced training duration from **~140 seconds per epoch (on CPU) to 7.8 seconds per epoch**, delivering an **18x acceleration** with real-time `tqdm` throughput tracking (~58 it/s).

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Plasmawolf29/retinal-oct-classification.git
cd retinal-oct-classification
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
The dataset is automatically accessible via the `medmnist` library:
```python
import medmnist
from medmnist import OCTMNIST
dataset = OCTMNIST(split='train', download=True)
```
*(Note: Large binary `.npz` files are excluded from git history via `.gitignore` to maintain a lightweight repository).* 

### 5. Launch the Notebook
```bash
jupyter notebook v2_improved_gpu_pipeline.ipynb
```

---

## 📚 References & Citations

1. **Yang, J. et al.** (2023). *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Nature Scientific Data, 10(1), 41.
2. **Kermany, D. S. et al.** (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122-1131.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).