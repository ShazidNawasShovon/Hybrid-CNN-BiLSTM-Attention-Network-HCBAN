import sys
import os
import time
import json

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

from src.research.hcban_model import HCBAN
from src.research.ensemble_model import HybridEnsemble
from src.data_preprocessing import load_data as load_raw_data, preprocess_data

class ResearchPipeline:
    def __init__(self, data_path='processed_data', n_splits=5, batch_size=256, epochs=10):
        self.data_path = data_path
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.epochs = epochs
        self.results = {}
        self.target_col = None
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def setup_dataset(self, choice=None, task_choice=None):
        print("\n--- Dataset Selection ---")
        print("1. Split Dataset (UNSW_NB15_training-set.csv + testing-set.csv)")
        print("2. Combined Dataset (combined_dataset_final.csv)")
        
        if choice is None:
            # Check if running in non-interactive mode or simply default to input
            try:
                choice = input("Select dataset (1 or 2): ").strip()
            except EOFError:
                print("No input provided, defaulting to Combined Dataset (2)")
                choice = '2'
        
        if choice == '2':
            dataset_type = 'combined'
            combined_path = r'd:\Personal\Development\Python\Hybrid Explainable AI Moon\dataset_combined\combined_dataset_final.csv'
            # Check if file exists, if not ask user or fail gracefully
            if not os.path.exists(combined_path):
                print(f"Default path {combined_path} not found.")
                try:
                    combined_path = input("Enter path to combined_dataset_final.csv: ").strip()
                except EOFError:
                    print("Cannot get path from input. Exiting.")
                    sys.exit(1)
        else:
            dataset_type = 'split'
            combined_path = None
            
        print(f"Selected: {dataset_type}")

        print("\n--- Task Selection ---")
        print("1. Binary Classification (label: normal vs attack)")
        print("2. Multi-class Classification (attack_cat)")
        if task_choice is None:
            try:
                task_choice = input("Select task (1 or 2): ").strip()
            except EOFError:
                task_choice = '1'
        self.target_col = 'attack_cat' if task_choice == '2' else 'label'
        print(f"Selected target: {self.target_col}")
        
        # Load and Preprocess
        print("Loading and preprocessing data...")
        df = load_raw_data(dataset_type=dataset_type, combined_path=combined_path)
        X_train, X_test, y_train, y_test, features, le = preprocess_data(df, target_col=self.target_col)
        
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
        
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train
        self.y_test = y_test

        self.X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        self.n_classes = len(np.unique(self.y_train))
        print(f"Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}, Features: {self.X_train.shape[1]}, Classes: {self.n_classes}")

    def _class_counts(self, y):
        y = np.asarray(y, dtype=int)
        counts = np.bincount(y, minlength=int(y.max()) + 1 if y.size else 0)
        return counts

    def _compute_class_weights(self, y):
        counts = self._class_counts(y)
        present = counts > 0
        n_present = int(present.sum()) if present.any() else 0
        if n_present == 0:
            return np.ones((0,), dtype=np.float32)
        total = float(counts[present].sum())
        weights = np.zeros_like(counts, dtype=np.float32)
        weights[present] = total / (n_present * counts[present].astype(np.float32))
        return weights

    def _compute_sample_weights(self, y, class_weights):
        y = np.asarray(y, dtype=int)
        if class_weights.size == 0:
            return None
        return class_weights[y].astype(np.float32)

    def _compute_auc(self, y_true, y_pred_prob):
        y_true = np.asarray(y_true, dtype=int)
        y_pred_prob = np.asarray(y_pred_prob)
        if self.n_classes == 2:
            if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 2:
                return float(roc_auc_score(y_true, y_pred_prob[:, 1]))
            return float(roc_auc_score(y_true, y_pred_prob))
        return float(roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='weighted'))

    def _print_balance_summary(self, y, title):
        counts = self._class_counts(y)
        present = counts[counts > 0]
        if present.size == 0:
            print(f"{title}: no labels")
            return
        imbalance_ratio = float(present.max() / present.min())
        print(f"{title} class distribution:")
        for idx, c in enumerate(counts):
            if c > 0:
                print(f"  Class {idx}: {int(c)}")
        print(f"{title} imbalance ratio (max/min): {imbalance_ratio:.2f}")

    def _balance_training_data(self, X_train_2d, y_train):
        counts = self._class_counts(y_train)
        present = counts[counts > 0]
        if present.size == 0:
            return X_train_2d, y_train

        imbalance_ratio = float(present.max() / present.min())
        if imbalance_ratio <= 10.0:
            return X_train_2d, y_train

        min_count = int(present.min())
        if min_count < 50:
            return X_train_2d, y_train

        cap_ratio = 10 if self.n_classes == 2 else 20
        target = {}
        for cls_idx, c in enumerate(counts):
            if c > 0:
                target[cls_idx] = int(min(c, min_count * cap_ratio))

        rus = RandomUnderSampler(sampling_strategy=target, random_state=42)
        X_res, y_res = rus.fit_resample(X_train_2d, y_train)
        return X_res, y_res

    def _train_hcban(self, X_train_3d, y_train, X_val_3d, y_val, class_weights, fold_tag):
        train_dataset = TensorDataset(
            torch.tensor(X_train_3d, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_3d, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        sampler = None
        counts = self._class_counts(y_train)
        present = counts[counts > 0]
        if present.size and float(present.max() / present.min()) > 10.0:
            sample_weights = self._compute_sample_weights(y_train, class_weights)
            if sample_weights is not None:
                sampler = WeightedRandomSampler(
                    weights=torch.tensor(sample_weights, dtype=torch.float32),
                    num_samples=len(sample_weights),
                    replacement=True,
                )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=(sampler is None), sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        model = HCBAN(input_channels=1, input_length=X_train_3d.shape[1], n_classes=self.n_classes)
        model.to(self.device)

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device) if class_weights.size else None
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total

            print(
                f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {val_epoch_loss:.4f} - Val Acc: {val_epoch_acc:.4f}"
            )

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_acc)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'results/{fold_tag}_best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        model.load_state_dict(torch.load(f'results/{fold_tag}_best_model.pth'))
        return model, val_loader, history

    def run_kfold_cv(self):
        print(f"\nRunning {self.n_splits}-Fold Cross-Validation on Hybrid Deep-Ensemble (PyTorch HCBAN + XGB/LGBM/RF)...")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [], 'training_time': []
        }
        
        # Ensure results directory exists before starting
        os.makedirs('results', exist_ok=True)

        self._print_balance_summary(self.y_train, "Train")
        
        fold = 1
        for train_index, val_index in skf.split(self.X_train_reshaped, self.y_train):
            print(f"\n[Fold {fold}/{self.n_splits}]")
            X_train_fold, X_val_fold = self.X_train_reshaped[train_index], self.X_train_reshaped[val_index]
            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]
            
            start_time = time.time()

            X_train_2d = X_train_fold.reshape(X_train_fold.shape[0], -1)
            X_train_2d, y_train_fold = self._balance_training_data(X_train_2d, y_train_fold)
            X_train_fold = X_train_2d.reshape(X_train_2d.shape[0], X_train_2d.shape[1], 1)

            class_weights = self._compute_class_weights(y_train_fold)
            sample_weights = self._compute_sample_weights(y_train_fold, class_weights)

            print("Training HCBAN (PyTorch) with class-weighted loss...")
            model, val_loader, history = self._train_hcban(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                class_weights,
                fold_tag=f'fold_{fold}',
            )
            
            # --- 3. Train ML Ensemble (XGBoost, LightGBM, RF) ---
            print("Training ML Ensemble (XGBoost, LightGBM, Random Forest)...")
            ensemble = HybridEnsemble(n_classes=self.n_classes)
            ensemble.fit(X_train_2d, y_train_fold, sample_weight=sample_weights)
            
            training_time = time.time() - start_time
            
            # --- 4. Hybrid Prediction (Ensemble) ---
            print("Generating Hybrid Predictions...")
            # HCBAN Probabilities
            model.eval()
            p_hcban = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    p_hcban.append(probs.cpu().numpy())
            p_hcban = np.concatenate(p_hcban, axis=0)
            
            # ML Ensemble Probabilities (Input needs to be 2D)
            X_val_2d = X_val_fold.reshape(X_val_fold.shape[0], -1)
            p_ensemble = ensemble.predict_proba(X_val_2d)
            
            y_pred_dl = np.argmax(p_hcban, axis=1)
            y_pred_ml = np.argmax(p_ensemble, axis=1)
            acc_dl = float(accuracy_score(y_val_fold, y_pred_dl))
            acc_ml = float(accuracy_score(y_val_fold, y_pred_ml))
            denom = max(acc_dl + acc_ml, 1e-8)
            w_ml = float(np.clip(acc_ml / denom, 0.2, 0.8))
            w_dl = 1.0 - w_ml
            y_pred_prob = (w_dl * p_hcban) + (w_ml * p_ensemble)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # Calculate Metrics
            acc = accuracy_score(y_val_fold, y_pred)
            prec = precision_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            
            try:
                auc = self._compute_auc(y_val_fold, y_pred_prob)
            except Exception as e:
                print(f"Error calculating ROC-AUC: {e}")
                auc = 0.0
                
            print(f"Fold {fold} Hybrid Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1)
            fold_metrics['auc'].append(auc)
            fold_metrics['training_time'].append(training_time)
            
            # Save Fold ROC Data
            os.makedirs('results', exist_ok=True)
            try:
                np.savez_compressed(
                    os.path.join('results', f'fold_{fold}_predictions.npz'),
                    y_true=y_val_fold,
                    y_pred_prob=y_pred_prob
                )
            except Exception as e:
                print(f"Error saving predictions for fold {fold}: {e}")
            
            # Save History
            try:
                with open(os.path.join('results', f'fold_{fold}_history.json'), 'w') as f:
                    json.dump(history, f)
            except Exception as e:
                print(f"Error saving history for fold {fold}: {e}")
            
            fold += 1
            
        # Convert all numpy arrays in fold_metrics to lists before saving
        for key in fold_metrics:
            fold_metrics[key] = [float(x) for x in fold_metrics[key]]
            
        self.results['HCBAN_CV'] = fold_metrics
        self.save_results()

    def run_holdout_test(self):
        print("\nRunning holdout Train/Test evaluation on Hybrid Deep-Ensemble...")
        self._print_balance_summary(self.y_train, "Train")
        self._print_balance_summary(self.y_test, "Test")

        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
            self.X_train_reshaped,
            self.y_train,
            test_size=0.1,
            random_state=42,
            stratify=self.y_train,
        )

        os.makedirs('results', exist_ok=True)

        X_train_sub_2d = X_train_sub.reshape(X_train_sub.shape[0], -1)
        X_train_sub_2d, y_train_sub = self._balance_training_data(X_train_sub_2d, y_train_sub)
        X_train_sub = X_train_sub_2d.reshape(X_train_sub_2d.shape[0], X_train_sub_2d.shape[1], 1)

        class_weights = self._compute_class_weights(y_train_sub)

        print("Training HCBAN (PyTorch) with class-weighted loss for holdout test...")
        model, val_loader, history = self._train_hcban(
            X_train_sub,
            y_train_sub,
            X_val_sub,
            y_val_sub,
            class_weights,
            fold_tag='holdout',
        )

        print("Training ML Ensemble for holdout test...")
        X_train_2d = self.X_train_reshaped.reshape(self.X_train_reshaped.shape[0], -1)
        X_train_2d, y_train_bal = self._balance_training_data(X_train_2d, self.y_train)
        class_weights_full = self._compute_class_weights(y_train_bal)
        sample_weights_full = self._compute_sample_weights(y_train_bal, class_weights_full)

        ensemble = HybridEnsemble(n_classes=self.n_classes)
        ensemble.fit(X_train_2d, y_train_bal, sample_weight=sample_weights_full)

        X_val_2d = X_val_sub.reshape(X_val_sub.shape[0], -1)
        model.eval()
        p_hcban_val = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                p_hcban_val.append(probs.cpu().numpy())
        p_hcban_val = np.concatenate(p_hcban_val, axis=0)
        p_ensemble_val = ensemble.predict_proba(X_val_2d)
        y_pred_dl_val = np.argmax(p_hcban_val, axis=1)
        y_pred_ml_val = np.argmax(p_ensemble_val, axis=1)
        acc_dl_val = float(accuracy_score(y_val_sub, y_pred_dl_val))
        acc_ml_val = float(accuracy_score(y_val_sub, y_pred_ml_val))
        denom = max(acc_dl_val + acc_ml_val, 1e-8)
        w_ml = float(np.clip(acc_ml_val / denom, 0.2, 0.8))
        w_dl = 1.0 - w_ml

        print("Evaluating on test set...")
        test_dataset = TensorDataset(
            torch.tensor(self.X_test_reshaped, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.long),
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model.eval()
        p_hcban = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                p_hcban.append(probs.cpu().numpy())
        p_hcban = np.concatenate(p_hcban, axis=0)

        p_ensemble = ensemble.predict_proba(self.X_test)

        y_pred_prob = (w_dl * p_hcban) + (w_ml * p_ensemble)
        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = float(accuracy_score(self.y_test, y_pred))
        prec = float(precision_score(self.y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(self.y_test, y_pred, average='weighted', zero_division=0))
        try:
            auc = float(self._compute_auc(self.y_test, y_pred_prob))
        except Exception as e:
            print(f"Error calculating ROC-AUC on test set: {e}")
            auc = 0.0

        print(f"Holdout Test Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        self.results['Holdout_Test'] = {
            'accuracy': [acc],
            'precision': [prec],
            'recall': [rec],
            'f1': [f1],
            'auc': [auc],
        }

        try:
            np.savez_compressed(
                os.path.join('results', 'holdout_test_predictions.npz'),
                y_true=self.y_test,
                y_pred=y_pred,
                y_pred_prob=y_pred_prob,
            )
        except Exception as e:
            print(f"Error saving holdout test predictions: {e}")

        try:
            with open(os.path.join('results', 'holdout_history.json'), 'w') as f:
                json.dump(history, f)
        except Exception as e:
            print(f"Error saving holdout history: {e}")
        
        self.save_results()
        
    def save_results(self):
        os.makedirs('results', exist_ok=True)
        # Convert numpy floats to python floats for JSON serialization
        serializable_results = {}
        for key, metrics in self.results.items():
            if isinstance(metrics, dict) and 'summary' in metrics:
                serializable_results[key] = metrics
                continue
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
    # Ensure GPU is visible (PyTorch)
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   Device Count: {torch.cuda.device_count()}")
    else:
        print("⚠️ No GPU detected. Training will be slow on CPU.")

    # Reduced epochs for quick testing, use 20-30 for actual run
    pipeline = ResearchPipeline(epochs=20) 
    # Use Combined Dataset by default for automated runs
    pipeline.setup_dataset(choice='2', task_choice='1')
    pipeline.load_data()
    pipeline.run_kfold_cv()
    pipeline.run_holdout_test()
    pipeline.generate_report()

    try:
        from src.research.visualization import ResearchVisualizer
        from src.research.generate_tables import generate_latex_table

        viz = ResearchVisualizer()
        viz.plot_metrics_comparison()
        label_encoder_path = os.path.join('models', 'label_encoder.joblib')
        classes = None
        if os.path.exists(label_encoder_path):
            import joblib

            le = joblib.load(label_encoder_path)
            classes = [str(x) for x in le.classes_]
        if classes:
            viz.plot_roc_curve(classes)
            viz.plot_training_history()
            viz.plot_holdout_results(classes)

        generate_latex_table()
    except Exception as e:
        print(f"Post-processing (plots/tables) failed: {e}")
