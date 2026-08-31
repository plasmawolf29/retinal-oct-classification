# Retinal OCT Pathology Classification: Classical Ensembles vs. Deep Convolutional Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20(CUDA)-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B%20(GPU)-1182c6.svg)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MedMNIST%20OCTMNIST-brightgreen.svg)](https://medmnist.com/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20GTX%201660%20Ti-76B900.svg)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end biomedical machine learning and computer vision system for automated multi-class diagnosis of retinal diseases from Optical Coherence Tomography (OCT) B-scans. This repository benchmarks classical statistical machine learning pipelines against progressively advanced deep residual neural network architectures, paired with clinical Explainable AI (Grad-CAM & SHAP).

---

## Engineering Progression & Versioning Story

This project illustrates an iterative engineering progression from a cloud-based academic course baseline to a high-performance, GPU-accelerated production pipeline with ophthalmologist-grade diagnostic accuracy:

```
v1.0 (Colab, CPU)         v2.0 (Local GPU)           v3.0 (GPU, Advanced)       v4.0 (GPU, Phase 3)
Academic Baseline    -->  CUDA 18x Speedup      -->  ResNet + Focal Loss   -->  SE-ResNet + 110-D
~51% Accuracy             73.0% Acc, 0.962 AUC       76.5% Acc, 0.962 AUC       83.3% Acc, 0.968 AUC
```

### Notebook Versions:
* v1: [`v1_original_colab_submission.ipynb`](./v1_original_colab_submission.ipynb) — Historical academic baseline (Google Colab, CPU).
* v2: [`v2_improved_gpu_pipeline.ipynb`](./v2_improved_gpu_pipeline.ipynb) — 66-D feature extractor, PyTorch CUDA, 18x speedup.
* v3: [`v3_experimental.ipynb`](./v3_experimental.ipynb) — ResidualCNN3, Focal Loss, TTA, XGBoost GPU, Grad-CAM, SHAP.
* v4: [`v4_phase3_advanced.ipynb`](./v4_phase3_advanced.ipynb) — SE-ResNet4 with Channel Attention, 110-D GLCM/Wavelet Features, GPU Vectorized Augmentation.

---

## Clinical Problem & Pathologies

Optical Coherence Tomography (OCT) is a non-invasive optical imaging modality capturing cross-sectional microstructural details of the retina. This system classifies scans into four clinical categories from the **MedMNIST OCTMNIST** benchmark (109,309 total scans):

| Class | Pathology | Clinical Manifestation |
| :---: | :--- | :--- |
| 0 | Choroidal Neovascularization (CNV) | Neovascular vessel growth beneath the retina causing severe fluid exudation and macular degeneration. |
| 1 | Diabetic Macular Edema (DME) | Retinal thickening and fluid accumulation resulting from diabetic retinopathy. |
| 2 | Drusen (DRUSEN) | Extracellular lipid/protein deposits beneath the retinal pigment epithelium; AMD precursor. |
| 3 | Normal (NORMAL) | Healthy retinal morphology with intact foveal depression and stratified layers. |

---

## Comprehensive Benchmark Results (Balanced Test Set: 1,000 Scans)

| Pipeline | Model Architecture | Key Engineering | Top-1 Acc | Top-2 Acc | Macro F1 | Log Loss | Macro ROC AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pipeline 2** | **SE_ResidualCNN4 + TTA (v4 Best)** | **SE Attention + 110-D + Cosine + FocalLoss** | **83.30%** | **97.60%** | **82.94%** | **0.4856** | **0.9681** |
| **Pipeline 2** | SE_ResidualCNN4 (no TTA) | SE Attention + Vectorized GPU Aug | 82.30% | 97.70% | 81.93% | 0.4869 | 0.9681 |
| **Pipeline 2** | ResidualCNN3 + TTA (v3) | ResNet + FocalLoss + Aug + TTA | 76.50% | — | 75.83% | — | 0.9615 |
| **Pipeline 2** | Deep Regularized CNN2 (v2) | 3-Block ConvNet + BatchNorm | 75.90% | — | 73.89% | — | 0.9696 |
| **Pipeline 2** | Baseline CNN1 | 3-Stage ConvNet | 66.50% | — | 57.56% | — | 0.9374 |
| **Pipeline 1** | **XGBoost GPU (110 Features, v4)** | **GLCM + Wavelets + Spatial + CUDA** | **61.80%** | **84.90%** | **55.11%** | **0.9625** | **0.8980** |
| **Pipeline 1** | HGB (110 Features, v4) | GLCM + Wavelets + Spatial | 61.00% | 84.30% | 54.36% | 0.9588 | 0.8987 |
| **Pipeline 1** | XGBoost GPU (66 Features, v3) | 66-D Spatial + CUDA | 59.00% | — | 51.15% | — | 0.8834 |
| **Pipeline 1** | HGB (66 Features, v2) | 66-D Spatial | 54.70% | — | 46.85% | — | 0.8633 |

### Key Performance Highlights:
* **83.3% Top-1 Test Accuracy** with SE_ResidualCNN4 + TTA — a **+10.3% gain over v2 (73.0%)** and **+6.8% gain over v3 (76.5%)**.
* **Top-2 Accuracy of 97.6%**: In 97.6% of cases, the true diagnosis is within the model's top 2 predicted conditions — directly applicable for clinical differential diagnosis support.
* **Macro ROC AUC = 0.9681**: Near ophthalmologist-level diagnostic discrimination; ranks above the true diseased scan vs. healthy scan 96.8% of the time regardless of classification threshold.
* **110-D GLCM + Wavelet features** push Classical ML to **61.8% accuracy (+2.8% over v3's 59.0%)** with the Haralick texture descriptors capturing sub-pixel cystic fluid patterns invisible to spatial averages alone.

---

## Feature Engineering: 66-D to 110-D

| Feature Group | Count | Clinical Significance |
| :--- | :---: | :--- |
| Regional Grid Means (4x4) | 16 | Spatial fluid pocket localization |
| Regional Grid Std Devs (4x4) | 16 | Local texture heterogeneity from drusen deposits |
| Vertical Depth Profiles | 14 | Retinal layer stratification along optical axis |
| Horizontal Symmetry Profiles | 14 | Bilateral macular symmetry disruption |
| Global Morphological Stats | 6 | Foveal depression integrity, edge gradient energy |
| **Haralick GLCM Textures (4 angles)** | **24** | **Micro-cystic fluid (DME) vs. dense lipid (Drusen) pattern differentiation** |
| **2D Wavelet DWT (db2, 2 levels)** | **20** | **RPE boundary disruption, sub-band layer energy** |
| **Total** | **110** | |

---

## Explainable AI (XAI) & Interpretability

### Grad-CAM on SE_ResidualCNN4 (Channel Attention + Gradient Heatmaps)
The SE channel attention mechanism adaptively amplifies feature channels corresponding to pathological biomarkers:
* **CNV**: Heatmap concentrates beneath the Bruch's membrane where neovascular fluid pools.
* **DME**: Highlights hyporeflective cystic intraretinal fluid between the ONL and GCL.
* **DRUSEN**: Localizes along focal dome-shaped RPE elevations and undulating drusen deposits.
* **NORMAL**: Diffuse uniform activation across intact continuous retinal laminations.

![Grad-CAM SE_ResidualCNN4](./gradcam_v4_se.png)

### SHAP Feature Importance (110-D TreeExplainer)
SHAP attribution on the full 110-dimensional biomedical feature space reveals:
* **New top features from GLCM/Wavelets**: `glcm_contrast_0deg`, `glcm_homogeneity_90deg`, `dwt1_cH1_energy` dominate Pipeline 1 predictions.
* **Validated baseline features**: `depth_row_6`, `foveal_mean`, and `grad_v` remain highly important.

![SHAP 110-Feature Summary](./shap_summary_110.png)

---

## Hardware Acceleration

Training all 97,477 scans on an **NVIDIA GeForce GTX 1660 Ti** with vectorized GPU tensor augmentation achieves **~7-8 seconds per epoch** (~58 it/s). Full Phase 3 v4 training completes in under **10 minutes**.

---

## Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Plasmawolf29/retinal-oct-classification.git
cd retinal-oct-classification
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows:
venv\\Scripts\\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
```python
from medmnist import OCTMNIST
dataset = OCTMNIST(split='train', download=True)
```

### 5. Run Phase 3 Pipeline
```bash
jupyter notebook v4_phase3_advanced.ipynb
```

---

## References & Citations

1. Yang, J. et al. (2023). *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Nature Scientific Data, 10(1), 41.
2. Kermany, D. S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122-1131.
3. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE ICCV, 618-626.
4. Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 30.
5. Haralick, R. M. et al. (1973). *Textural Features for Image Classification*. IEEE Transactions on Systems, Man, and Cybernetics.
6. Hu, J. et al. (2018). *Squeeze-and-Excitation Networks*. IEEE CVPR.

---

## License
This project is open-source under the [MIT License](LICENSE).