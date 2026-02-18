# Thesis: A Novel GPU-Accelerated Hybrid CNN-BiLSTM-Attention Network for Explainable Intrusion Detection

## Abstract
Intrusion Detection Systems (IDS) must accurately identify malicious activity under high class imbalance, evolving attack behaviors, and the practical need for explainable decisions. While deep learning models can learn complex patterns, many existing approaches either focus on local feature extraction (CNN) or sequence modeling (RNN/LSTM) but fail to unify both efficiently, and they typically remain opaque to security analysts. This thesis proposes a **Hybrid CNN–BiLSTM–Attention Network (HCBAN)** trained on GPU with an end-to-end reproducible research pipeline. HCBAN combines 1D convolution for local pattern extraction, bidirectional LSTM for sequential dependency modeling, and multi-head self-attention to emphasize the most informative parts of the learned feature sequence. To address imbalance, the training pipeline uses class-weighted optimization (and weighted sampling when needed) and combines deep predictions with a tabular ensemble (XGBoost/LightGBM/RandomForest) via calibrated soft-voting. Experiments using 5-fold cross-validation and a holdout test split show strong discriminative performance: mean CV ROC-AUC ≈ 0.982 and mean CV accuracy ≈ 0.936 for the binary IDS task, with similar holdout performance. Finally, the system provides post-hoc interpretability using SHAP-based global and local explanations, supporting analyst-centric “risk reporting.”

## 1. Introduction & Research Gap
### 1.1 The Problem
Modern cyber-attacks are becoming increasingly sophisticated, employing evasion techniques that bypass traditional signature-based IDS. Machine Learning (ML) and Deep Learning (DL) approaches have been adopted to detect anomaly-based attacks. However, existing solutions face significant challenges:
1.  **Limited Feature Representation**: CNN-based models excel at spatial features but miss temporal context. RNN/LSTM models capture sequence but are computationally expensive and struggle with long-term dependencies (vanishing gradient).
2.  **Lack of Focus**: Standard DL models treat all parts of a packet sequence equally, often diluting the signal from small but critical malicious payloads.
3.  **Computational Bottlenecks**: Many complex hybrid models are not optimized for modern hardware, leading to prohibitive training times.
4.  **Interpretability**: High-accuracy DL models are often opaque, making it difficult for analysts to trust and verify alerts.

### 1.2 The Research Gap
A critical gap exists in developing an architecture that:
*   Combines the strengths of **CNN (Spatial)**, **BiLSTM (Temporal)**, and **Attention (Contextual Focus)** in a unified, efficient framework.
*   Is explicitly optimized for **GPU acceleration** to handle large-scale IDS datasets.
*   Addresses **severe class imbalance** without relying on unstable synthetic oversampling for extremely rare attack types.
*   Provides **explainability** without sacrificing predictive performance and reproducibility.

### 1.3 Contributions
This thesis contributes:
1.  **HCBAN Architecture**: A novel hybrid model integrating 1D-CNN, BiLSTM, and Multi-Head Attention.
2.  **Imbalance-Aware Training**: Class-weighted optimization and weighted sampling strategies for minority-sensitive learning.
3.  **Hybrid Deep-Ensemble**: Soft-voting fusion between HCBAN and a strong tabular ensemble to improve robustness.
4.  **Reproducible Evaluation**: 5-Fold Stratified Cross-Validation plus holdout testing with artifact export (predictions, histories, plots, and tables).
5.  **Explainability Framework**: SHAP-based global and local model interpretation and risk reporting.

### 1.4 Focus, Goal, and Objectives
**Focus**: GPU-enabled, imbalance-aware, explainable intrusion detection using a unified spatiotemporal deep model and a complementary tabular ensemble.

**Goal**: Build and evaluate an IDS that is accurate, stable under class imbalance, and explainable for analyst decision support.

**Objectives**:
1. Design and implement the HCBAN architecture for tabular-to-sequence intrusion detection inputs.
2. Develop an end-to-end pipeline for preprocessing, training, evaluation, and artifact generation.
3. Compare model variants and report metrics suitable for imbalanced IDS (ROC-AUC, F1, precision/recall).
4. Provide explainability outputs usable in a thesis and for analyst interpretation.

## 2. Related Work
1. **CNN-based IDS**: Effective at local feature interactions but limited for long-range dependencies in network flow behavior.
2. **RNN/LSTM-based IDS**: Models sequences and dependencies but can be harder to train and slower for large datasets.
3. **Attention-enhanced IDS**: Improves focus on informative segments/features, often increasing robustness to noise.
4. **Ensemble learning for IDS**: Boosting and stacking improve tabular performance and calibration, frequently outperforming single learners.
5. **Explainable IDS**: SHAP/LIME and feature attribution methods help interpret predictions, improving trust and operational adoption.

## 2. Methodology
### 2.1 Proposed Architecture: HCBAN
The HCBAN model is designed as a sequential pipeline:

#### 2.1.1 Spatial Feature Extraction (1D-CNN)
The input network traffic (preprocessed features) is treated as a sequence. Two 1D-Convolutional layers with ReLU activation extract local n-gram features (e.g., patterns in byte sequences).
*   **Layer 1**: 64 filters, kernel size 3, followed by Batch Normalization and Max Pooling.
*   **Layer 2**: 128 filters, kernel size 3, followed by Batch Normalization and Max Pooling.

#### 2.1.2 Sequential Modeling (BiLSTM)
A Bidirectional LSTM layer processes the feature maps from the CNN.
*   **BiLSTM**: 128 units. It processes the sequence in both forward and backward directions, capturing dependencies that span across the packet flow.
*   **Dropout**: A dropout rate of 0.3 is applied to prevent overfitting.

#### 2.1.3 Contextual Attention (Multi-Head Self-Attention)
The core novelty is the addition of a Multi-Head Attention layer after the BiLSTM.
*   **Mechanism**: The attention mechanism computes a weighted sum of the BiLSTM outputs, allowing the model to focus on "relevant" time steps (e.g., specific malicious signatures) while ignoring noise.
*   **Configuration**: 4 parallel attention heads with a key dimension of 128.
*   **Residual Connection**: A residual connection (Add & Norm) preserves gradient flow and stabilizes training.

#### 2.1.4 Classification Head
*   **Global Average Pooling**: Aggregates the attention-weighted sequence into a fixed-size vector.
*   **Dense Layers**: Two fully connected layers (256 and 128 units) with ReLU activation and Dropout (0.4).
*   **Output**: A final linear layer produces logits; Softmax is applied for probabilities.

### 2.2 GPU Acceleration Strategy
Training is executed on GPU using a PyTorch implementation. The pipeline automatically selects CUDA when available and exports checkpoints for reproducibility. Mixed-precision (AMP) is compatible with this design and is included as future optimization work if required by throughput constraints.

### 2.3 Data Preprocessing
1. **Train/Test Split**: Stratified split to preserve class proportions.
2. **Feature Handling**:
   - Numerical features: mean imputation + StandardScaler.
   - Categorical features: imputation + OneHotEncoder.
3. **Leakage Control**:
   - When predicting the binary `label`, the `attack_cat` column is excluded from the feature set.
   - When predicting `attack_cat`, the `label` column is excluded.
4. **Reshaping**: Features are reshaped to `(samples, features, 1)` for 1D-CNN processing.

### 2.4 Class Imbalance Handling
The pipeline uses imbalance-aware training rather than heavy synthetic oversampling for extremely rare classes:
1. **Class-Weighted Loss**: Cross-entropy loss weighted by inverse class frequency.
2. **Weighted Sampling**: When imbalance is severe, weighted sampling increases the probability of sampling minority-class instances during mini-batch training.
3. **Imbalance-Aware ML Training**: Sample weights are passed to XGBoost/LightGBM/RandomForest training.

### 2.5 Hybrid Deep-Ensemble Fusion
Predictions from HCBAN and the ML ensemble are fused using soft-voting. Voting weights are tuned on validation data to prefer whichever component performs better on the current fold/validation split.

### 2.6 Experimental Setup
*   **Tasks**:
    *   Binary classification: `label` (normal vs attack).
    *   Multi-class classification: `attack_cat` (attack category).
*   **Validation**: 5-Fold Stratified Cross-Validation on the training split.
*   **Holdout Test**: Final evaluation on an unseen test split.
*   **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
*   **Environment**: PyTorch + scikit-learn + XGBoost/LightGBM; GPU via CUDA when available.

## 3. Results & Analysis
This section is populated from the pipeline exports:
* `results/research_results.json`
* `results/thesis_performance_table.csv` and `results/thesis_performance_table.tex`
* Figures under `plots/research/`

### 3.1 Performance Metrics
*   **Accuracy**: Measures overall correctness.
*   **Precision/Recall/F1-Score**: Critical for imbalanced datasets (detecting minority attacks like Worms).
*   **ROC-AUC**: Evaluates the model's ability to distinguish between classes.

### 3.2 Summary of Current Experimental Results (Binary IDS)
Using the current research pipeline configuration for the binary IDS task:
* Mean 5-fold CV Accuracy ≈ 0.936
* Mean 5-fold CV ROC-AUC ≈ 0.982
* Holdout Test Accuracy ≈ 0.936
* Holdout Test ROC-AUC ≈ 0.983

These results indicate strong ranking/discrimination (high ROC-AUC) while overall accuracy reflects the inherent ambiguity and overlap present in the selected feature set and data distribution.

### 3.2 Ablation Study
We compare HCBAN against:
1.  **Baseline CNN**: Only CNN layers + Dense.
2.  **Baseline LSTM**: Only BiLSTM layers + Dense.
3.  **HCBAN w/o Attention**: The hybrid model without the attention mechanism.
(Results will show the incremental gain of each component.)

## 4. Conclusion
The HCBAN architecture bridges the research gap by unifying CNN, BiLSTM, and attention into a single IDS model while supporting imbalance-aware training, GPU execution, and explainability outputs. The hybrid deep-ensemble pipeline provides strong ROC-AUC and stable cross-validated performance, and it produces thesis-ready artifacts (tables and figures) for transparent reporting.

## 5. Future Work
1. Enable mixed precision (AMP) in PyTorch for higher throughput on compatible GPUs.
2. Expand the feature space (or use richer flow-level feature sets) to improve separability and raise overall accuracy.
3. Add calibrated decision thresholds and cost-sensitive evaluation aligned with operational IDS requirements.
4. Conduct external validation across datasets and investigate domain adaptation.
5. Extend explainability to sequence-aware attributions and integrate analyst feedback loops.
