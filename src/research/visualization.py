import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc
import glob
import joblib

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class ResearchVisualizer:
    def __init__(self, results_path='results/research_results.json', output_dir='plots/research'):
        self.results_path = results_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_results()
        
    def load_results(self):
        if os.path.exists(self.results_path):
            with open(self.results_path, 'r') as f:
                self.results = json.load(f)
        else:
            print(f"Results file not found: {self.results_path}")
            self.results = {}

    def plot_roc_curve(self, classes):
        """
        Plots ROC curves for each class (Macro-average across folds if possible, 
        or for the best fold). Here we aggregate or plot the last available fold.
        """
        # Find prediction files
        pred_files = glob.glob(os.path.join('results', 'fold_*_predictions.npz'))
        if not pred_files:
            print("No prediction files found for ROC plotting.")
            return

        # Use the first file for demonstration (or iterate to average)
        # Ideally, we should plot Micro/Macro average over all folds
        
        plt.figure(figsize=(10, 8))
        
        # Aggregate y_true and y_score
        all_y_true = []
        all_y_score = []
        
        for f in pred_files:
            data = np.load(f)
            all_y_true.append(data['y_true'])
            all_y_score.append(data['y_pred_prob'])
            
        y_true = np.concatenate(all_y_true)
        y_score = np.concatenate(all_y_score)
        if y_score.ndim == 1:
            y_score = np.column_stack([1.0 - y_score, y_score])
        elif y_score.ndim == 2 and y_score.shape[1] == 1 and len(classes) == 2:
            y_score = np.column_stack([1.0 - y_score[:, 0], y_score[:, 0]])
        
        n_classes = int(y_score.shape[1]) if y_score.ndim == 2 else int(len(classes))
        if len(classes) != n_classes:
            if len(classes) > n_classes:
                classes = classes[:n_classes]
            else:
                classes = [str(i) for i in range(n_classes)]
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='#4c72b0', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        else:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_true, classes=range(n_classes))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            colors = plt.cm.get_cmap('tab10', n_classes)
            for i, color in zip(range(n_classes), colors.colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]),
                )

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Multi-Class)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        print(f"Saved ROC curve to {self.output_dir}/roc_curve.png")

    def plot_training_history(self):
        """
        Plots training and validation accuracy/loss curves aggregated over folds.
        """
        history_files = glob.glob(os.path.join('results', 'fold_*_history.json'))
        if not history_files:
            print("No history files found.")
            return
            
        # Aggregate metrics
        metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        agg_history = {m: [] for m in metrics}
        
        for f in history_files:
            with open(f, 'r') as file:
                h = json.load(file)
                for m in metrics:
                    if m in h:
                        agg_history[m].append(h[m])

        min_len = None
        for m in metrics:
            if agg_history[m]:
                m_min = min(len(seq) for seq in agg_history[m])
                min_len = m_min if min_len is None else min(min_len, m_min)
        if not min_len or min_len < 1:
            print("Training history files are empty.")
            return

        for m in metrics:
            agg_history[m] = [seq[:min_len] for seq in agg_history[m] if len(seq) >= min_len]
        
        # Plot Mean +/- Std
        epochs = range(1, min_len + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy Plot
        plt.subplot(1, 2, 1)
        for m, label in [('accuracy', 'Training Acc'), ('val_accuracy', 'Validation Acc')]:
            data = np.array(agg_history[m])
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            plt.plot(epochs, mean, label=label)
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)
            
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss Plot
        plt.subplot(1, 2, 2)
        for m, label in [('loss', 'Training Loss'), ('val_loss', 'Validation Loss')]:
            data = np.array(agg_history[m])
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            plt.plot(epochs, mean, label=label)
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)
            
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.close()
        print(f"Saved training history plot to {self.output_dir}/training_history.png")

    def plot_metrics_comparison(self):
        """
        Plots a bar chart comparing Accuracy, Precision, Recall, F1, AUC
        with error bars (Standard Deviation) from K-Fold CV.
        """
        if 'HCBAN_CV' not in self.results:
            return
            
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        means = []
        stds = []
        
        summary = self.results['HCBAN_CV']['summary']
        
        for m in metrics:
            means.append(summary.get(f'mean_{m}', 0))
            stds.append(summary.get(f'std_{m}', 0))
            
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        bars = plt.bar(metrics, means, yerr=stds, capsize=10, color=['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974'])
        
        plt.title('HCBAN Performance Metrics (5-Fold CV)', fontsize=16)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0.0, 1.05)
        
        # Add values on top
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, mean + 0.02, f'{mean:.4f}', 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
        print(f"Saved metrics comparison plot to {self.output_dir}/metrics_comparison.png")

    def plot_source_holdout_accuracy(self):
        keys = [k for k in self.results.keys() if k.startswith('SourceHoldout_')]
        if not keys:
            return

        groups = {}
        for k in keys:
            rem = k.replace('SourceHoldout_', '', 1)
            if rem.startswith('label_'):
                task_name = 'label'
                src = rem.replace('label_', '', 1)
            elif rem.startswith('attack_cat_'):
                task_name = 'attack_cat'
                src = rem.replace('attack_cat_', '', 1)
            else:
                task_name = 'label'
                src = rem
            groups.setdefault(task_name, []).append((src, self.results.get(k, {})))

        for task_name, items in groups.items():
            rows = []
            for src, v in sorted(items, key=lambda x: x[0]):
                acc = v.get('accuracy', [0])
                auc_v = v.get('auc', [0])
                rows.append({
                    'source': src,
                    'accuracy': float(acc[0]) if isinstance(acc, list) and acc else float(acc),
                    'auc': float(auc_v[0]) if isinstance(auc_v, list) and auc_v else float(auc_v),
                })
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            df = df.sort_values('accuracy', ascending=False)

            plt.figure(figsize=(10, 5))
            sns.set_style("whitegrid")
            sns.barplot(data=df, x='source', y='accuracy', color='#4c72b0')
            plt.ylim(0.0, 1.05)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Source-Holdout Accuracy by Dataset Source ({task_name})')
            plt.tight_layout()
            out = os.path.join(self.output_dir, f'source_holdout_accuracy_{task_name}.png')
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"Saved source-holdout accuracy plot to {out}")

    def plot_training_history_single(self, history):
        """
        Plots training and validation accuracy/loss curves.
        Args:
            history: Keras History object or dict.
        """
        # This requires history object, typically passed during training.
        # If we saved history in results, we can plot it.
        # For now, let's assume we pass a dummy history or skip if not available.
        pass

    def plot_holdout_results(self, classes):
        path = os.path.join('results', 'holdout_test_predictions.npz')
        if not os.path.exists(path):
            print("No holdout prediction file found.")
            return
        data = np.load(path)
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_score = data['y_pred_prob']
        if y_score.ndim == 1:
            y_score = np.column_stack([1.0 - y_score, y_score])
        elif y_score.ndim == 2 and y_score.shape[1] == 1 and len(classes) == 2:
            y_score = np.column_stack([1.0 - y_score[:, 0], y_score[:, 0]])
        n_classes = int(y_score.shape[1]) if y_score.ndim == 2 else int(len(classes))
        if len(classes) != n_classes:
            classes = [str(i) for i in range(n_classes)]

        self.plot_confusion_matrix(y_true, y_pred, classes)

        plt.figure(figsize=(10, 8))
        y_unique = np.unique(y_true).size
        if n_classes == 2 and y_unique <= 2:
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='#4c72b0', lw=2, label=f'ROC curve (area = {roc_auc_val:0.2f})')
        else:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_true, classes=range(n_classes))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            colors = plt.cm.get_cmap('tab10', n_classes)
            for i, color in zip(range(n_classes), colors.colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]),
                )
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Holdout Test ROC')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'holdout_roc_curve.png'), dpi=300)
        plt.close()
        print(f"Saved holdout ROC curve to {self.output_dir}/holdout_roc_curve.png")

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """
        Plots a publication-ready confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix (Normalized)', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print(f"Saved confusion matrix to {self.output_dir}/confusion_matrix.png")

if __name__ == "__main__":
    viz = ResearchVisualizer()
    classes = None
    label_encoder_path = os.path.join('models', 'label_encoder.joblib')
    pred_files = glob.glob(os.path.join('results', 'fold_*_predictions.npz'))
    if pred_files:
        data = np.load(pred_files[0])
        n_classes = int(data['y_pred_prob'].shape[1])
        classes = [str(i) for i in range(n_classes)]
    elif os.path.exists(label_encoder_path):
        le = joblib.load(label_encoder_path)
        classes = [str(x) for x in le.classes_]
    else:
        classes = []

    viz.plot_metrics_comparison()
    if classes:
        viz.plot_roc_curve(classes)
        viz.plot_training_history()
        viz.plot_holdout_results(classes)
    viz.plot_source_holdout_accuracy()
