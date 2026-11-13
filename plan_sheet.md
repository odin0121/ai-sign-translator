# AI Sign Language Translator: Agent Implementation Plan Sheet
## Multi-Layered Hybrid Framework with Localized Feature Modeling & Post-Hoc Explainable AI (XAI)

**Authors:** Daniel Biju Jose & Avira S  
**USNs:** P18AJ25S126025 / P18AJ25S126023  
**Benchmark Target:** 99.2% sequence accuracy (target, not a measured result)  
**Latency Budget:** $\le 50\text{ ms}$ per frame  

---

## 1. Executive Directive & Architecture Overview

The primary objective is an autonomous, high-accuracy, interpretable sign language translation pipeline processing real-time video feeds into continuous text and synthesized speech.

```
+---------------------------------------------------------------------------------------+
|                                    INPUT VIDEO FEED                                   |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                       1. COORDINATE TRACKING & NORMALIZATION                          |
|  - MediaPipe 21 Joint Hand Landmark Tracking                                          |
|  - Wrist-Centric Spatial Alignment: P_norm = (P_i - P_wrist) / ||P||                   |
|  - Sliding Sequence Temporal Window: T = 60 frames, D = 63 features (21 x 3)         |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                       2. TEMPORAL CONVOLUTIONAL NETWORK (TCN)                         |
|  - Dilated Causal 1D Convolutions (d in {1, 2, 4}, kernel k=3, channels 64 -> 128)    |
|  - Exponential receptive field capturing kinematic hand transitions & velocity       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                    3. MULTI-HEAD ATTENTION TRANSFORMER BACKBONE                       |
|  - Multi-Head Self-Attention (h=4 heads, d_k=32, d_model=128)                         |
|  - Position-wise Feed Forward Network (FFN: 256 -> 128) + Residual Add & LayerNorm    |
|  - Global Average Pooling 1D -> Latent Embedding Vector z in R^128                    |
+---------------------------------------------------------------------------------------+
                      |                                                 |
                      v                                                 v
+-------------------------------------------+     +-------------------------------------+
|        4. SVM CLASSIFICATION HEAD         |     |       5. POST-HOC XAI EXPLAINER     |
| - Multi-Class SVM (RBF Kernel, C=10.0)    |     | - Gradient Backpropagation Tape     |
| - Max-margin deterministic token decision |     |   A_k = sum_{c} |dz / dp_{k,c}|     |
| - Class posterior probability calibration |     | - 21-joint dynamic saliency overlay |
+-------------------------------------------+     +-------------------------------------+
                      \                                                 /
                       \                                               /
                        v                                             v
+---------------------------------------------------------------------------------------+
|                               6. DJANGO TELEMETRY & TTS                               |
|  - Live MJPEG video stream with dynamic saliency heat skeleton                        |
|  - SQLite translation logging & system performance telemetry                         |
|  - Dual offline/cloud Text-to-Speech (TTS: pyttsx3 / gTTS)                            |
+---------------------------------------------------------------------------------------+
```

---

## 2. Benchmark Datasets & Preprocessing Specification

| Dataset | Modality & Volume | Preprocessing & Uniform Normalization | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **WLASL100 archive** | 1,013 labeled MP4 clips available in the uploaded archive | Use the official train/validation/test split (748/165/100 clips), sample to $T=60$, and extract 21 hand joints. | Train and evaluate the word classifier on the 100 WLASL100 labels |
| **ISL emergency signs** | 8 cropped-video labels, about 50 clips per label and 25–26 signers | Extract 21 hand joints from the cropped AVI clips; split by signer so a person's clips cannot appear in multiple splits. | Train a separate ISL emergency-sign model; do not merge its labels with ASL/WLASL |
| **ASL Alphabet Dataset** | 87,000+ Static RGB Images (26-29 Classes) | Wrist-centric landmark alignment: $\hat{P}_i = (P_i - P_{wrist}) / \max \|P_j - P_{wrist}\|_2$ | Static handshape token grounding & SVM boundary tuning |
| **Uploaded video archive** | 11,980 MP4 files and metadata for 2,000 glosses | The training command uses the official WLASL100 split included in the archive and caches extracted sequences in `weights/wlasl100_landmarks.npz`. | Real labeled video training; the held-out test score is reported by the training command |

---

## 3. Mathematical Foundations

### 3.1 Wrist-Centric Spatial Normalization
For hand landmark positions $P_i = (x_i, y_i, z_i)$ where $i \in \{0, \dots, 20\}$ and $P_0 = P_{wrist}$:
$$\tilde{P}_i = P_i - P_0$$
$$\hat{P} = \frac{\tilde{P}}{\|\tilde{P}\|_2 + 10^{-6}} \in \mathbb{R}^{63}$$

### 3.2 Dilated Causal Convolution (TCN)
Given 1D input sequence $\mathbf{x} \in \mathbb{R}^{T \times D}$ and filter $f: \{0, \dots, k-1\} \to \mathbb{R}$:
$$y(t) = (x *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot x(t - d \cdot i)$$
Where dilation rate $d \in \{1, 2, 4\}$ expands receptive field without recurrent gradient vanishing.

### 3.3 Multi-Head Self-Attention Transformer
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
Where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ with $h=4, d_k=32, d_{model}=128$.

### 3.4 Support Vector Machine (SVM) Classification Head
Given latent representation $\mathbf{z} \in \mathbb{R}^{128}$:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{n} \xi_n \quad \text{s.t.} \quad y_n(\mathbf{w}^T \phi(\mathbf{z}_n) + b) \ge 1 - \xi_n, \quad \xi_n \ge 0$$
Using Radial Basis Function (RBF) kernel:
$$K(\mathbf{z}, \mathbf{z}') = \exp(-\gamma \|\mathbf{z} - \mathbf{z}'\|^2), \quad C=10.0$$

### 3.5 Post-Hoc Explainable AI (XAI) Joint Saliency Attribution
Using TensorFlow `GradientTape` on the backbone latent maximum activation $z^* = \max_m z_m$:
$$A_k = \sum_{c \in \{x,y,z\}} \left| \frac{\partial z^*}{\partial p_{k, c}} \right|$$
Normalized for live skeleton heat rendering:
$$\bar{A}_k = \frac{A_k - \min(A)}{\max(A) - \min(A) + 10^{-6}} \in [0, 1]$$

---

## 4. Verification & Milestone Deliverables

- [x] **Phase 1: Ingestion & Landmarks** (`core_engine/data_loader.py`, `core_engine/landmark_extractor.py`)
- [x] **Phase 2: Hybrid Model Backbone** (`core_engine/hybrid_model.py`)
- [x] **Phase 3: SVM Classification Head** (`core_engine/svm_classifier.py`)
- [x] **Phase 4: Post-Hoc XAI Heatmaps** (`core_engine/xai_explainer.py`)
- [x] **Phase 5: Django Web Integration** (`translator_app/`, `weights/`, `manage.py`)
- [x] **Train from uploaded data and measure held-out accuracy:** `python train_models.py --archive "../archive (2).zip"`
- [x] **Train signer-independent ISL emergency model:** `python train_isl_models.py`

The ASL/WLASL100 model uses landmark feature pipeline version 2 and scored **46.0% top-1 accuracy on 100 held-out clips**. The new, separate ISL emergency model was trained from **410 usable clips across 8 signs**; its signer-independent split is **252 train / 78 validation / 80 test clips**, and its held-out top-1 accuracy is **91.25%**. Its validation-calibrated 0.85 cutoff accepts 32/80 test clips (40% coverage), all 32 correct. That selective metric is a small sample. The UI lets users select the ASL/WLASL100 or ISL emergency model without merging the vocabularies; low-confidence predictions remain `UNCERTAIN`. The ISL dataset contains only two recordings per signer per sign, so additional signers and real-camera testing are still needed before treating this result as production accuracy.

The Learning Studio exposes lessons for all 100 WLASL signs and all 8 ISL emergency signs. Each lesson uses an example clip from the corresponding training split; WLASL clips are served as MP4 and ISL AVI clips are converted to cached animated GIF previews. Learned progress is stored locally in the browser per model profile. Camera practice compares predictions with the selected lesson without adding practice results to the translation sentence. Dataset clips are examples, not authoritative instruction, and signing can vary by signer and region.

The floating Signa assistant is available on both the Live Studio and Analytics pages. It answers common site-use, model, vocabulary, camera, learning, analytics, and translation questions using the selected model profile where relevant. It uses the app's built-in help logic, validates requests, and does not send questions to an external generative-AI service.
