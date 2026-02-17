import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Redirect stdout/stderr to a log file
# log_file = open("research_pipeline.log", "w")
# sys.stdout = log_file
# sys.stderr = log_file

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.research.hcban_model import HCBAN
from src.data_preprocessing import load_data as load_raw_data, preprocess_data

class ResearchPipeline:
    def __init__(self, data_path='processed_data', n_splits=5, batch_size=256, epochs=10):
        self.data_path = data_path
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.epochs = epochs
        self.results = {}
        
    def setup_dataset(self):
        print("\n--- Dataset Selection ---")
        print("1. Split Dataset (UNSW_NB15_training-set.csv + testing-set.csv)")
        print("2. Combined Dataset (combined_dataset_final.csv)")
        
        choice = input("Select dataset (1 or 2): ").strip()
        
        if choice == '2':
            dataset_type = 'combined'
            combined_path = r'd:\Personal\Development\Python\Hybrid Explainable AI Moon\dataset_combined\combined_dataset_final.csv'
            # Check if file exists, if not ask user
            if not os.path.exists(combined_path):
                print(f"Default path {combined_path} not found.")
                combined_path = input("Enter path to combined_dataset_final.csv: ").strip()
        else:
            dataset_type = 'split'
            combined_path = None
            
        print(f"Selected: {dataset_type}")
        
        # Load and Preprocess
        print("Loading and preprocessing data...")
        df = load_raw_data(dataset_type=dataset_type, combined_path=combined_path)
        X_train, X_test, y_train, y_test, features, le = preprocess_data(df)
        
        # Save processed data (overwriting existing)
        os.makedirs(self.data_path, exist_ok=True)
        X_train.to_csv(os.path.join(self.data_path, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(self.data_path, 'X_test.csv'), index=False)
        np.save(os.path.join(self.data_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.data_path, 'y_test.npy'), y_test)
        print("Data preprocessing complete.")

    def load_data(self):
        print("Loading data for research pipeline...")
        X_train = pd.read_csv(os.path.join(self.data_path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(self.data_path, 'X_test.csv'))
        y_train = np.load(os.path.join(self.data_path, 'y_train.npy'))
        y_test = np.load(os.path.join(self.data_path, 'y_test.npy'))
        
        # Concatenate
        self.X = np.concatenate([X_train.values, X_test.values], axis=0)
        self.y = np.concatenate([y_train, y_test], axis=0)
        
        # Reshape for 1D-CNN (samples, features, 1)
        self.X_reshaped = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))
        self.n_classes = len(np.unique(self.y))
        print(f"Total samples: {len(self.X)}, Features: {self.X.shape[1]}, Classes: {self.n_classes}")

    def run_kfold_cv(self):
        print(f"\nRunning {self.n_splits}-Fold Cross-Validation on HCBAN...")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [], 'training_time': []
        }
        
        fold = 1
        for train_index, val_index in skf.split(self.X_reshaped, self.y):
            print(f"\n[Fold {fold}/{self.n_splits}]")
            X_train_fold, X_val_fold = self.X_reshaped[train_index], self.X_reshaped[val_index]
            y_train_fold, y_val_fold = self.y[train_index], self.y[val_index]
            
            # Initialize HCBAN
            hcban = HCBAN(input_shape=(self.X.shape[1], 1), n_classes=self.n_classes)
            model = hcban.build_model()
            
            # Train
            start_time = time.time()
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1
            )
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred_prob = model.predict(X_val_fold)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # Calculate Metrics
            acc = accuracy_score(y_val_fold, y_pred)
            prec = precision_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            
            try:
                # Need to check if y_val_fold is one-hot or labels
                # y is label encoded (integers)
                # roc_auc_score expects probability for multi_class='ovr'
                auc = roc_auc_score(y_val_fold, y_pred_prob, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Error calculating ROC-AUC: {e}")
                auc = 0.0
                
            print(f"Fold {fold} Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1)
            fold_metrics['auc'].append(auc)
            fold_metrics['training_time'].append(training_time)
            
            # Save Fold ROC Data (FPR, TPR for each class)
            # To save space, we can save just the macro-average or specific class ROC
            # But let's save the y_true and y_pred_prob for this fold to a separate file
            # This allows full flexibility for visualization later.
            # Warning: This can be large. Let's save it to a compressed numpy format.
            np.savez_compressed(
                os.path.join('results', f'fold_{fold}_predictions.npz'),
                y_true=y_val_fold,
                y_pred_prob=y_pred_prob
            )
            
            # Save History
            with open(os.path.join('results', f'fold_{fold}_history.json'), 'w') as f:
                json.dump(history.history, f)
            
            fold += 1
            
        self.results['HCBAN_CV'] = fold_metrics
        self.save_results()
        
    def save_results(self):
        os.makedirs('results', exist_ok=True)
        # Convert numpy floats to python floats for JSON serialization
        serializable_results = {}
        for key, metrics in self.results.items():
            serializable_results[key] = {}
            for k, vals in metrics.items():
                serializable_results[key][k] = [float(v) for v in vals]
            
            # Add mean and std
            serializable_results[key]['summary'] = {}
            for k, vals in metrics.items():
                serializable_results[key]['summary'][f'mean_{k}'] = float(np.mean(vals))
                serializable_results[key]['summary'][f'std_{k}'] = float(np.std(vals))
                
        with open('results/research_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print("Results saved to results/research_results.json")
        
    def generate_report(self):
        if 'HCBAN_CV' not in self.results:
            print("No results to report.")
            return
            
        # Compute mean/std manually
        print("\n=== HCBAN Research Performance Report ===")
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10} | {'95% CI':<15}")
        print("-" * 60)
        
        for metric, values in self.results['HCBAN_CV'].items():
            mean = np.mean(values)
            std = np.std(values)
            ci = 1.96 * std / np.sqrt(len(values)) # 95% Confidence Interval
            print(f"{metric.capitalize():<15} | {mean:.4f}     | {std:.4f}     | +/- {ci:.4f}")

if __name__ == "__main__":
    # Reduced epochs for quick testing, use 20-30 for actual run
    pipeline = ResearchPipeline(epochs=5) 
    pipeline.setup_dataset()
    pipeline.load_data()
    pipeline.run_kfold_cv()
    pipeline.generate_report()
