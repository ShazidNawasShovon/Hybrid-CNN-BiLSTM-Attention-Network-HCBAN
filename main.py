import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.data_preprocessing import load_data, preprocess_data
from src.hybrid_dl_model import HybridDLIDS
from src.explainability import ExplainabilityEngine

def main():
    print("============================================================")
    print("   Hybrid Explainable IDS (Deep Learning + XGBoost) > 97%   ")
    print("============================================================")
    
    # Step 1: Data Preprocessing
    print("\n[Step 1] Data Preprocessing...")
    if not os.path.exists('processed_data/X_train.csv'):
        df = load_data()
        X_train, X_test, y_train, y_test, features, le = preprocess_data(df)
        
        os.makedirs('processed_data', exist_ok=True)
        X_train.to_csv('processed_data/X_train.csv', index=False)
        X_test.to_csv('processed_data/X_test.csv', index=False)
        np.save('processed_data/y_train.npy', y_train)
        np.save('processed_data/y_test.npy', y_test)
        print("Data preprocessing complete and saved.")
    else:
        print("Processed data found. Loading...")
        X_train = pd.read_csv('processed_data/X_train.csv')
        X_test = pd.read_csv('processed_data/X_test.csv')
        y_train = np.load('processed_data/y_train.npy')
        y_test = np.load('processed_data/y_test.npy')
        le = joblib.load('models/label_encoder.joblib')

    # Step 2: Feature Selection (Optional / Analysis)
    # For maximum accuracy (>97%), we use ALL features or a larger subset.
    # The previous 20 features were too restrictive for Deep Learning.
    print("\n[Step 2] Feature Selection (Skipped for Maximum Accuracy)...")
    print(f"Using all {X_train.shape[1]} features for Hybrid Deep Learning Model.")
    selected_features = X_train.columns.tolist()

    # Step 3: Hybrid Model Training
    print("\n[Step 3] Hybrid DL-Stacking Model Training...")
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Initialize Hybrid Model (1D-CNN + BiLSTM + XGBoost + LightGBM)
    hybrid_model = HybridDLIDS(input_shape=(n_features, 1), n_classes=n_classes, epochs=10, batch_size=256)
    
    if not os.path.exists('models/hybrid_dl_model'):
        print("Training Hybrid Model...")
        hybrid_model.fit(X_train, y_train)
        hybrid_model.save_model('models/hybrid_dl_model')
    else:
        print("Loading trained Hybrid Model...")
        hybrid_model.load_model('models/hybrid_dl_model')
        
    # Evaluate
    print("Evaluating Hybrid Model...")
    y_pred = hybrid_model.predict(X_test)
    # y_prob = hybrid_model.predict_proba(X_test) # Optional for ROC-AUC
    
    from sklearn.metrics import classification_report, accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Hybrid Model Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Step 4: Explainability
    print("\n[Step 4] Explainability & Risk Reporting...")
    # Use the hybrid model path
    engine = ExplainabilityEngine(model_path='models/hybrid_dl_model', feature_names=selected_features)
    
    # Compute SHAP values for a sample
    # We use a small sample (100) from test set
    sample_size = 100
    X_sample = X_test.iloc[:sample_size]
    
    # The engine will automatically use the XGBoost component for explanation
    engine.compute_shap_values(X_sample, model_type='hybrid')
    
    os.makedirs('plots', exist_ok=True)
    engine.plot_summary(X_sample, save_path='plots/shap_summary_hybrid.png')
    
    # Find an attack sample to explain
    # Class 'Normal' index
    normal_idx = list(le.classes_).index('Normal')
    attack_indices = [i for i in range(sample_size) if y_test[i] != normal_idx]
    
    if attack_indices:
        target_idx = attack_indices[0]
        true_label = le.classes_[y_test[target_idx]]
        print(f"Generating Risk Report for Sample #{target_idx} (True Label: {true_label})")
        engine.generate_risk_report(target_idx, X_sample, le.classes_, save_path='plots/risk_report_hybrid.txt')
    else:
        print("No attack samples found in the first 100 test samples.")
        
    print("\n============================================================")
    print("   Hybrid Architecture Execution Completed!   ")
    print("============================================================")

if __name__ == "__main__":
    main()
