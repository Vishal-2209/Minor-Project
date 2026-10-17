# SSMP: Adaptive Self-Supporting Prototype Learning for Remote Sensing (Zero-Shot, 1-Shot, 5-Shot)

Implementation and recreation of the paper:
> **"Adaptive Self-Supporting Prototype Learning for Remote Sensing Few-Shot Semantic Segmentation"**  
> *Weihao Shen, Ailong Ma, Junjue Wang, Zhuo Zheng, Yanfei Zhong*  
> IEEE Transactions on Geoscience and Remote Sensing (TGRS), vol. 62, 2024.  
> DOI: `10.1109/TGRS.2024.3435086`

---

## 📌 Project Overview

This repository provides three standalone, end-to-end Jupyter notebooks reproducing and extending the **SSMP** framework across three shot regimes on the **iSAID-5i** benchmark:

1. **`SSMP_ZeroShot_Recreation.ipynb` (Zero-Shot)**:
   - Uses frozen **CLIP** text embeddings (`openai/clip-vit-base-patch32`) with prompt templates to generate initial semantic prototypes without any support images or masks.
   - Refines prototypes via a learnable `TextPrototypeAdapter` and matches query features using HPR, SSM, and ASBP.

2. **`SSMP_OneShot_Recreation.ipynb` (1-Shot)**:
   - Evaluates the standard 1-way 1-shot setting where 1 support image and binary mask are provided per episode.
   - Computes visual support prototypes via **Masked Average Pooling (MAP)** and clusters support foreground features into hyperprototypes (HPR).

3. **`SSMP_FewShot_Recreation.ipynb` (5-Shot / Multi-Shot)**:
   - Evaluates the multi-shot (5-shot) setting where 5 support exemplars are provided.
   - Performs joint masked average pooling across all $K=5$ support shots and pools support foreground features across exemplars for multi-shot HPR clustering.

---

## 🚀 Key Modules & Architecture

- **Frozen Pretrained Backbone**: ResNet-50 keeping multi-scale spatial feature maps (layer3 + layer4 concatenated to 3072 channels, stride 16).
- **Projection Head**: $1\times 1$ Conv $\to$ GroupNorm $\to$ ReLU $\to$ $1\times 1$ Conv mapping features to a shared 512-dimensional $L_2$-normalized embedding space.
- **HPR (Adaptive Hyperprototype Representation)**:
  - *HPC (Hyperprototype Clustering)*: Clusters foreground features in joint feature + spatial coordinates into multiple representative hyperprototypes.
  - *GPM (Guided Prototype Matching)*: Cosine similarity matching with soft aggregation for pixel-level guided score maps.
- **SSM (Self-Supporting Matching)**: High-confidence pseudo labels from the query image generate self-supporting query prototypes, fused with support prototypes via learnable $\lambda$ coefficients.
- **ASBP (Adaptive Self-Supporting Background Prototype)**: Dynamic affinity matrix between query background features and all query pixels to construct an adaptive, per-pixel background prototype.
- **Prototype Refinement**: Secondary round of pseudo-label extraction and guided matching.

---

## ⚙️ Training Quality Improvements

All three notebooks feature optimized training setups:
- **Epochs**: Increased to **25 epochs** (from initial 8 epochs) for thorough convergence.
- **Learning Rate Schedule**: `torch.optim.lr_scheduler.CosineAnnealingLR` ($5\times 10^{-4} \to 1\times 10^{-5}$) for smooth optimization.
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_` (max norm 1.0) to stabilize updates.
- **Remote Sensing Augmentation**: Random horizontal and vertical flips preserving aerial nadir geometry.
- **Loss Function**: Weighted Cross-Entropy ($W_{fg} = 3.0$) + self-supporting pseudo loss ($L = L_{seg} + 0.2 L_{pseudo}$).

---

## 📊 Benchmark & Evaluation Metrics

- Primary Metric: **mIoU** (mean Intersection-over-Union over unseen classes).
- Secondary Metrics: **FB-IoU** (Foreground-Background IoU), Pixel Accuracy, Dice / F1 Score, and Per-Class IoU.
- Full ablation suite included in each notebook:
  - Config A: Baseline
  - Config B: + HPR
  - Config C: + SSM
  - Config D: + ASBP
  - Config E: + Refine (Full Model)

---

## 💻 How to Run

1. Open any of the notebooks in **Google Colab** (with a free GPU runtime) or run locally with a CUDA GPU:
   - `SSMP_ZeroShot_Recreation.ipynb`
   - `SSMP_OneShot_Recreation.ipynb`
   - `SSMP_FewShot_Recreation.ipynb`
2. Run cells in order: dataset download via `kagglehub`, model training, checkpoint saving, evaluation on novel classes, visualizations, and ablation studies.
