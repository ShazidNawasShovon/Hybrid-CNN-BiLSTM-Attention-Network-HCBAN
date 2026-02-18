# Hybrid Explainable AI (XAI) Intrusion Detection System (IDS)

## Overview
This project implements a **Hybrid Explainable AI (XAI) Intrusion Detection System (IDS)** with a research pipeline designed for reproducible experiments, GPU training, and thesis-ready outputs (tables + figures).

The core deep model is the **Hybrid CNN-BiLSTM-Attention Network (HCBAN)** (PyTorch), combined with a tabular ML ensemble (XGBoost/LightGBM/RandomForest) using soft-voting to improve robustness.

## Research Gap & Novelty

### The Research Gap
Modern Intrusion Detection Systems (IDS) face several critical challenges:
1.  **Limited Feature Representation**: CNNs excel at spatial features but miss temporal context. RNNs/LSTMs capture sequences but are computationally expensive and struggle with long-term dependencies.
2.  **Lack of Focus**: Standard Deep Learning models treat all parts of a packet sequence equally, often diluting the signal from small but critical malicious payloads.
3.  **Computational Bottlenecks**: Complex hybrid models are often not optimized for modern hardware, leading to prohibitive training times.
4.  **Interpretability**: High-accuracy "Black Box" models are opaque, making it difficult for analysts to trust automated alerts.

### Our Novel Contribution: HCBAN
We propose the **Hybrid CNN-BiLSTM-Attention Network (HCBAN)** to bridge these gaps:
*   **Hybrid Architecture**: Unifies **1D-CNN** (Spatial), **BiLSTM** (Temporal), and **Multi-Head Attention** (Contextual Focus) into a single efficient framework.
*   **Attention Mechanism**: Dynamically weighs the importance of different packet segments, allowing the model to focus on malicious signatures regardless of their position.
*   **GPU Acceleration**: Runs on CUDA-enabled GPUs via PyTorch (`torch.device('cuda')`) for faster training and evaluation.
*   **Explainability**: Integrates SHAP (SHapley Additive exPlanations) to provide transparent, actionable "Risk Reports" for every detection.

## Project Structure
- `dataset/`: Split UNSW-NB15 CSV files (optional mode).
- `dataset_combined/`: Combined dataset file (default mode).
- `processed_data/`: Cached train/test splits after preprocessing.
- `src/`: Source code for the pipeline.
  - `data_preprocessing.py`: Data loading, cleaning, encoding, and normalization.
  - `research/`: **Core Research Module**
    - `hcban_model.py`: PyTorch implementation of the HCBAN architecture.
    - `research_pipeline.py`: End-to-end research runner (preprocess → CV → holdout test → plots/tables).
    - `visualization.py`: Generates publication-ready figures (ROC, Confusion Matrices).
    - `generate_tables.py`: Creates LaTeX/CSV tables for thesis documentation.
  - `explainability.py`: SHAP integration for global and local explanations.
- `results/`: Stores JSON metrics, LaTeX tables, and raw prediction data.
- `plots/`: Stores generated figures (ROC curves, training history).
- `Dockerfile`: Configuration for containerized execution.
- `requirements.txt`: Python dependencies.

## Step-by-Step Execution Guide

### Prerequisites
*   **Python 3.8+**
*   **NVIDIA GPU** (Recommended) with CUDA drivers installed.
*   **Docker** (Optional, for containerized run).

### 1. Installation (Local)

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ShazidNawasShovon/Hybrid-CNN-BiLSTM-Attention-Network-HCBAN.git
    cd Hybrid-CNN-BiLSTM-Attention-Network-HCBAN
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install PyTorch**:
    ```bash
    pip install torch
    ```
    If you want GPU acceleration, install a CUDA-enabled PyTorch build that matches your system CUDA setup.

### 2. Dataset Setup

#### Option A: Combined Dataset (Default)
- Place the file here: `dataset_combined/combined_dataset_final.csv`
- Expected columns include: `attack_cat`, `label`, and `dataset_source`

#### Option B: Split UNSW-NB15 Dataset
- Place these files here:
  - `dataset/UNSW_NB15_training-set.csv`
  - `dataset/UNSW_NB15_testing-set.csv`

### 3. Running the Research Pipeline (Full Process)
This is the main process to train the model, evaluate performance, and generate all thesis materials.

#### Option A: Local Execution (Recommended)
1.  **Run the full research runner**:
    ```bash
    python src/research/research_pipeline.py
    ```
    By default, the script runs:
    - Dataset: Combined (`dataset_combined/combined_dataset_final.csv`)
    - Task: Binary classification (`label`)

    To switch dataset/task, edit the `setup_dataset(...)` call in [research_pipeline.py](file:///d:/Personal/Development/Python/Hybrid%20Explainable%20AI%20Moon/src/research/research_pipeline.py) or run it from Python and call `setup_dataset(choice=None, task_choice=None)` for interactive selection.

    Outputs:
    - Metrics: `results/research_results.json`
    - Per-fold artifacts:
      - `results/fold_<k>_predictions.npz`
      - `results/fold_<k>_history.json`
      - `results/fold_<k>_best_model.pth`
    - Holdout test artifacts:
      - `results/holdout_test_predictions.npz`
      - `results/holdout_history.json`
      - `results/holdout_best_model.pth`
    - Plots (thesis figures): `plots/research/*.png`
    - Tables (thesis tables): `results/thesis_*.csv` and `results/thesis_*.tex`

2.  **(Optional) Regenerate figures only**:
    ```bash
    python src/research/visualization.py
    ```
    Output directory: `plots/research/`
    - `metrics_comparison.png`
    - `roc_curve.png`
    - `training_history.png`
    - `confusion_matrix.png`
    - `holdout_roc_curve.png`

3.  **(Optional) Regenerate tables only**:
    ```bash
    python src/research/generate_tables.py
    ```
    Output directory: `results/`
    - `thesis_performance_table.csv` and `thesis_performance_table.tex`
    - `thesis_holdout_test_table.csv` and `thesis_holdout_test_table.tex`

#### Option B: Docker Execution
Docker support is optional. The current [Dockerfile](file:///d:/Personal/Development/Python/Hybrid%20Explainable%20AI%20Moon/Dockerfile) is legacy and may require updates for the latest PyTorch-based research pipeline. Local execution is recommended.

1.  **Build Image**:
    ```bash
    docker build -t hcban-ids .
    ```

2.  **Run Container**:
    Mounts local folders to save results/plots back to your machine.
    ```bash
    docker run --gpus all -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots hcban-ids
    ```

### 4. Generating Explanations (XAI)
To understand *why* the model flagged a specific packet:

```bash
python main.py
```
*Action*: This runs the `ExplainabilityEngine` which uses SHAP to generate a "Risk Report".
*Output*: Check `plots/risk_report_hybrid.txt` and `plots/shap_summary_hybrid.png`.

## Thesis Resources: Figures & Tables

For your thesis, use the generated assets found in these directories:

### Figures (`plots/research/`)
*   **Figure 1: ROC Analysis** (`roc_curve.png`) - Demonstrates the model's high sensitivity and specificity across all attack classes.
*   **Figure 2: Confusion Matrix** (`confusion_matrix.png`) - Shows classification accuracy per class, highlighting detection rates for minority classes (Worms, Shellcode).
*   **Figure 3: Training Stability** (`training_history.png`) - Proves convergence and lack of overfitting (Loss/Accuracy curves).
*   **Figure 4: Performance Comparison** (`metrics_comparison.png`) - Visual summary of Accuracy, Precision, Recall, F1, and AUC.

### Tables (`results/`)
*   **Table 1: HCBAN Performance** (`thesis_performance_table.tex`) - Comprehensive metrics (Mean ± Std Dev) from 5-Fold CV.
    *   *Includes: Accuracy, Precision, Recall, F1-Score, ROC-AUC with 95% Confidence Intervals.*

## Methodology Summary

1.  **Data Preprocessing**:
    *   **Cleaning**: Imputation for missing values; drop `id` if present.
    *   **Encoding**: OneHotEncoder for categorical columns; StandardScaler for numerical columns.
    *   **Leakage Control**: Drops `attack_cat` when predicting `label` and drops `label` when predicting `attack_cat`.
    *   **Reshaping**: Converts tabular features to `(samples, features, 1)` for the PyTorch CNN input.

2.  **Model Architecture (HCBAN)**:
    *   **CNN Block**: Two 1D-Conv + BatchNorm + MaxPool layers for local pattern extraction.
    *   **BiLSTM Block**: Bidirectional LSTM for sequential dependencies.
    *   **Attention Block**: Multi-Head Self-Attention for contextual focus.
    *   **Head**: Dense layers → logits/probabilities.

3.  **Training Strategy**:
    *   **Optimizer**: Adam.
    *   **Regularization**: Dropout + early stopping.
    *   **Class Imbalance**: Class-weighted loss and sample-weighted ML training; optional capped undersampling for extreme imbalance cases.

## License
MIT
