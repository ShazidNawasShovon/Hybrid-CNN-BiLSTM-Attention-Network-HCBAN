## Title

Hybrid Deep-Ensemble Network Intrusion Detection with Transformer Backbones and Imbalance-Aware Training

## Abstract

Modern network intrusion detection systems (NIDS) must detect a wide variety of attack types under severe class imbalance and rapidly evolving traffic patterns. This work investigates a hybrid deep-ensemble NIDS that combines a PyTorch backbone (either a temporal transformer-based model, TBGMT, or the hybrid CNN–BiLSTM model HCBAN) with tree-based learners (XGBoost, LightGBM, Random Forest) on top of engineered tabular features. The system operates on a preprocessed, combined benchmark dataset (e.g., UNSW-NB15 combined CSV) with multi-class labels (`attack_cat`) and uses stratified train/test splits, K-fold cross-validation, and a separate holdout evaluation. To address skewed attack distributions, the pipeline integrates imbalance-aware techniques including SMOTE or controlled undersampling, class-weighted cross-entropy, and optional sample weighting in PyTorch. An auxiliary autoencoder-based anomaly feature augments the tabular representation by encoding reconstruction error as an additional signal. The deep backbone and the tree ensemble are fused via probability-level ensembling with data-driven weights. The proposed pipeline is fully configurable via a `.env` file (model choice, AMP, anomaly feature, balancing strategy, batch size) and produces plots and LaTeX-ready tables for thesis reporting. This research aims to quantify the benefits of such a hybrid deep-ensemble architecture for multi-class intrusion detection, compared to single-model baselines, under realistic imbalance and multi-source settings.

## Research Focus

- Develop and evaluate a **hybrid deep-ensemble NIDS** that combines:
  - A configurable PyTorch backbone (TBGMT transformer or HCBAN CNN–BiLSTM),
  - A tabular ML ensemble (XGBoost, LightGBM, Random Forest),
  - An autoencoder-based anomaly feature.
- Target **multi-class intrusion detection** using the `attack_cat` label rather than simple binary (normal vs attack) classification.
- Explicitly handle **severe class imbalance** and heterogeneous traffic sources via resampling, class weighting, and source-wise evaluation.
- Provide a **reproducible research pipeline** (`research_pipeline.py`) with automatic cross-validation, holdout testing, metric aggregation, and exportable plots/tables.

## Problem Statement

Network intrusion detection on modern benchmark datasets (e.g., UNSW-NB15 combined) is challenged by:

- Highly **imbalanced class distributions**, where common benign or frequent attacks dominate rare but critical attack types.
- The need to perform **fine-grained multi-class classification** (multiple attack categories) rather than only binary detection.
- The coexistence of temporal dependencies and tabular feature relationships that are not fully captured by a single model family.
- Limited reproducibility in many works, with ad-hoc splits, incomplete cross-validation, and weak handling of source/domain shifts.

The core problem is:

> How can we design a scalable, reproducible, imbalance-aware hybrid deep-ensemble NIDS that achieves strong multi-class performance across diverse attack types and traffic sources, outperforming single deep or single tree-based models?

## Research Gap

Based on recent literature on CIC/UNSW-style datasets and common practice:

- Many NIDS studies focus on **binary classification** (benign vs attack) or a small number of coarse classes, which inflates accuracy and hides per-attack weaknesses.
- Imbalance handling is often limited to a single technique (e.g., basic oversampling or class weights) without combining **resampling, class-weighted loss**, and **per-class evaluation**.
- Architectures typically use either:
  - One deep model (CNN, LSTM, Transformer), or
  - One classical model (XGBoost, RF),
  but rarely a **systematic hybrid of deep temporal models with strong tabular ensembles**.
- Reproducible research tooling (K-fold cross-validation, holdout validation, per-source evaluation, and automatic table/plot generation) is frequently under-specified.

This work aims to fill these gaps by:

- Operating primarily in **multi-class mode** (`attack_cat`) with detailed per-class metrics.
- Combining **transformer/CNN–BiLSTM backbones** with **XGBoost/LightGBM/RF** in a unified hybrid ensemble.
- Using **configurable imbalance strategies** (SMOTE, strict undersampling, hybrid capping) plus class-weighted loss and optional sample weighting.
- Providing a **single, configurable research pipeline** (`research_pipeline.py`) driven by `.env` for experiment management and reproducible reporting.

## Research Objectives

1. **Design and implement** a configurable hybrid deep-ensemble NIDS pipeline that supports multiple backbone architectures (TBGMT, HCBAN) and a consistent tree-ensemble head.
2. **Handle multi-class imbalance** by integrating SMOTE, controlled undersampling, and class-weighted loss, and evaluate their individual and combined impact on performance.
3. **Incorporate an autoencoder-based anomaly feature** and measure its contribution to detection quality compared to models trained on raw tabular features only.
4. **Evaluate the system under rigorous protocols**, including stratified K-fold cross-validation, train/validation splits, and holdout testing, with macro/micro metrics and per-class scores.
5. **Compare the hybrid ensemble** against single deep backbones and single tree-based baselines to quantify gains in accuracy, F1-score, and robustness to rare attack classes.
6. **Produce thesis-ready artifacts**, including performance tables (CSV and LaTeX) and plots (ROC curves, training histories, metric comparisons) directly from the pipeline.

## Methodology

### 1. Data and Label Configuration

- Use the combined benchmark dataset specified in `dataset_combined/combined_dataset_final.csv` (or equivalent), loaded via `load_raw_data` and `preprocess_data`.
- Select **multi-class classification** by targeting the `attack_cat` column (via `task_choice='2'` in `ResearchPipeline.setup_dataset`).
- Apply **stratified splitting** into train and test sets, optionally with source/holdout strategies controlled by `split_strategy` and `holdout_source`.

### 2. Preprocessing and Class Balancing

- Preprocess features and labels in `preprocess_data`:
  - Encode categorical fields (e.g., service, proto, state),
  - Scale numeric features,
  - Encode multi-class labels into integer indices.
- In `ResearchPipeline._balance_training_data`:
  - Compute class counts and imbalance ratios.
  - Depending on `BALANCE_TYPE` from `.env`:
    - `none`: no resampling,
    - `strict_under`: full undersampling to minority size,
    - `smote`: oversampling using SMOTE (current default in `.env`),
    - `hybrid`: controlled undersampling with capped ratios for multiclass.
- This balancing is applied on the training folds and training subsets, not on validation/test, to preserve realistic evaluation.

### 3. Deep Backbone (PyTorch) Training

- Instantiate the backbone in `_train_hcban` based on `MODEL_NAME`:
  - `HCBAN`: CNN–BiLSTM hybrid model,
  - `TBGMT`: transformer-based temporal model (current default).
- Convert tabular training/validation data to 3D tensors `(samples, features, 1)` and wrap them in `TensorDataset`s.
- Configure `DataLoader`s with:
  - `batch_size` from `.env` (`BATCH_SIZE`),
  - `num_workers` and `pin_memory` optimized for GPU/CPU,
  - Optional `WeightedRandomSampler` when imbalance is extreme.
- Use **class-weighted cross-entropy loss** via `_compute_class_weights` and pass weights to `nn.CrossEntropyLoss`.
- Enable **mixed precision training** with `torch.amp.autocast` and `GradScaler` when `ENABLE_AMP=1`.
- Train for a configurable number of epochs (`epochs` in `ResearchPipeline`) with:
  - Running batch-level logs (loss, AvgLoss, AvgAcc),
  - Epoch-level summaries (Loss, Acc, Val Loss, Val Acc),
  - Early stopping on validation loss.

### 4. Autoencoder-Based Anomaly Feature

- In `_add_anomaly_feature`, train a small MLP autoencoder (`MLPAutoencoder`) on:
  - Normal-class samples only for binary tasks, or
  - The full training set for multiclass (current mode).
- Compute **reconstruction error** for training and evaluation data.
- Concatenate the reconstruction error as an additional feature column to the 2D tabular inputs for the tree ensemble.
- Control training length with `AE_EPOCHS` from `.env`.

### 5. Tree-Based Ensemble and Hybrid Fusion

- Instantiate `HybridEnsemble` with XGBoost, LightGBM, and Random Forest classifiers.
- Fit the ensemble on the augmented, balanced tabular training data (with anomaly feature) using `sample_weight` derived from class weights when available.
- For validation/holdout:
  - Obtain probability outputs from both:
    - The deep backbone (softmax over classes),
    - The ML ensemble (predict_proba).
  - Compute standalone accuracies of deep vs ML models.
  - Derive **data-driven fusion weights** (`w_dl`, `w_ml`) based on validation accuracies with clipping.
  - Combine probabilities: `p_hybrid = w_dl * p_deep + w_ml * p_ensemble`.

### 6. Evaluation Protocol

- **K-Fold Cross-Validation** (`run_kfold_cv`):
  - Use `StratifiedKFold` with `n_splits` (default 2) on training data.
  - For each fold:
    - Apply balancing to the training subset,
    - Train deep backbone + ML ensemble,
    - Compute hybrid predictions on validation subset,
    - Collect accuracy, weighted precision, recall, F1, AUC, and per-class metrics.
  - Aggregate metrics across folds into `self.results['HCBAN_CV']` (historical key name for this hybrid pipeline).

- **Holdout Test** (`run_holdout_test`):
  - Split the full training data into train_sub/val_sub for tuning fusion weights.
  - Train backbone on train_sub and ensemble on balanced full training data.
  - Compute fusion weights from validation.
  - Evaluate hybrid model on the **held-out test set** and log metrics.

- **Source Holdout** (`run_source_holdout_all`, when `RUN_SOURCE_HOLDOUT_ALL=1`):
  - Perform per-source or per-domain evaluations, holding out traffic from specific sources to test generalization.

### 7. Reporting and Visualization

- Save aggregated results to `results/research_results.json` via `save_results`.
- Use `ResearchVisualizer` to:
  - Plot ROC curves (`roc_curve.png`),
  - Plot aggregated training history (`training_history.png`),
  - Plot metric comparison bar charts (`metrics_comparison.png`).
- Use `generate_latex_table` to:
  - Create CSV and LaTeX tables for CV metrics and holdout test performance:
    - `thesis_performance_table.csv` / `.tex`,
    - `thesis_holdout_test_table.csv` / `.tex`.
- These outputs are directly usable in the thesis document.

## Current Configuration Snapshot (from .env)

- Backbone model: **TBGMT** (`MODEL_NAME=tbgmt`)
- Mixed precision: **enabled** (`ENABLE_AMP=1`)
- Anomaly feature: **enabled** (`ENABLE_ANOMALY_FEATURE=1`)
- Autoencoder epochs: **10** (`AE_EPOCHS=10`)
- Source holdout runs: **enabled** (`RUN_SOURCE_HOLDOUT_ALL=1`)
- Class balancing: **SMOTE** (`BALANCE_TYPE=smote`)
- PyTorch batch size: **128** (`BATCH_SIZE=128`)

This configuration emphasizes:

- A transformer-based backbone,
- Strong imbalance handling via SMOTE,
- Rich hybrid features (autoencoder reconstruction error),
- Comprehensive evaluation (CV + holdout + source-wise).

