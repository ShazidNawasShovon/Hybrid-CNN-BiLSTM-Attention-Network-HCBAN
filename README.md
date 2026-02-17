# Hybrid Explainable AI (XAI) Intrusion Detection System (IDS) > 97% Accuracy

## Overview
This project implements a state-of-the-art **Hybrid Explainable AI (XAI) Intrusion Detection System (IDS)** using the **UNSW-NB15** dataset. It addresses the "Black Box" problem of Deep Learning models in security while achieving high accuracy (>97%) through a novel Hybrid Deep-Stacking architecture.

The core of this project is the **Hybrid CNN-BiLSTM-Attention Network (HCBAN)**, which combines the strengths of spatial feature extraction (CNN), temporal sequence modeling (BiLSTM), and dynamic contextual focus (Multi-Head Attention). This architecture is explicitly optimized for **GPU acceleration** using Mixed Precision training.

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
*   **GPU Acceleration**: Implements `mixed_float16` precision to leverage NVIDIA Tensor Cores, reducing memory usage by ~50% and significantly speeding up training.
*   **Explainability**: Integrates SHAP (SHapley Additive exPlanations) to provide transparent, actionable "Risk Reports" for every detection.

## Project Structure
- `dataset/`: Contains the UNSW-NB15 dataset files.
- `src/`: Source code for the pipeline.
  - `data_preprocessing.py`: Data loading, cleaning, encoding, and normalization.
  - `research/`: **Core Research Module**
    - `hcban_model.py`: Implementation of the HCBAN architecture with GPU optimizations.
    - `research_pipeline.py`: Automated 5-Fold Cross-Validation pipeline.
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
*   **NVIDIA GPU** (Recommended) with CUDA 11+ drivers.
*   **Docker** (Optional, for containerized run).

### 1. Installation (Local)

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd Hybrid-Explainable-AI-Moon
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs TensorFlow, XGBoost, LightGBM, SHAP, and other required libraries.*

### 2. Running the Research Pipeline
This is the main process to train the model, evaluate performance, and generate all thesis materials.

#### Option A: Local Execution (Recommended)
1.  **Run the Pipeline**:
    Executes 5-Fold Stratified Cross-Validation on the HCBAN model.
    ```bash
    python src/research/research_pipeline.py
    ```
    *Action*: Trains the model 5 times, saves metrics to `results/research_results.json`, and saves raw predictions to `results/fold_*_predictions.npz`.

2.  **Generate Figures**:
    Creates high-quality plots for your thesis.
    ```bash
    python src/research/visualization.py
    ```
    *Output*: Check `plots/research/` for:
    *   `roc_curve.png`: Multi-class ROC Curves with AUC scores.
    *   `confusion_matrix.png`: Normalized Confusion Matrix heatmap.
    *   `training_history.png`: Accuracy and Loss curves over epochs.
    *   `metrics_comparison.png`: Bar chart of metrics with error bars.

3.  **Generate Tables**:
    Creates formatted tables for your thesis document.
    ```bash
    python src/research/generate_tables.py
    ```
    *Output*: Check `results/` for:
    *   `thesis_performance_table.tex`: LaTeX code to copy-paste into your thesis.
    *   `thesis_performance_table.csv`: Raw CSV data for Excel/Word.

#### Option B: Docker Execution
Use this if you want a consistent, isolated environment.

1.  **Build Image**:
    ```bash
    docker build -t hcban-ids .
    ```

2.  **Run Container**:
    Mounts local folders to save results/plots back to your machine.
    ```bash
    docker run --gpus all -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots hcban-ids
    ```

### 3. Generating Explanations (XAI)
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
    *   **Cleaning**: Handling missing values and dropping irrelevant columns (`id`, `label`).
    *   **Encoding**: One-Hot Encoding for `state` and `service`; Frequency Encoding for `proto`.
    *   **Normalization**: StandardScaler applied to all numerical features.
    *   **Reshaping**: Transforming tabular data into 3D format `(samples, features, 1)` for CNN input.

2.  **Model Architecture (HCBAN)**:
    *   **Input Layer**: Accepts reshaped feature sequences.
    *   **CNN Block**: Two layers of 1D-Conv + BatchNorm + MaxPool to extract local patterns.
    *   **BiLSTM Block**: Bidirectional LSTM (128 units) to capture sequential dependencies.
    *   **Attention Block**: Multi-Head Attention (4 heads) to focus on critical time steps.
    *   **Classification Head**: Global Average Pooling -> Dense (256, 128) -> Softmax.

3.  **Training Strategy**:
    *   **Loss Function**: Sparse Categorical Crossentropy.
    *   **Optimizer**: Adam (learning rate = 0.001) with ReduceLROnPlateau.
    *   **Regularization**: Dropout (0.3 - 0.4) and Early Stopping.
    *   **Acceleration**: Mixed Precision (`mixed_float16`) for GPU efficiency.

## License
MIT
