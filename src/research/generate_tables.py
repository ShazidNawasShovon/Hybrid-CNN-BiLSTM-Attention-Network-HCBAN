import json
import pandas as pd
import numpy as np
import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def generate_latex_table(results_path='results/research_results.json'):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)
        
    if 'HCBAN_CV' not in data:
        print("HCBAN_CV results not found.")
        data['HCBAN_CV'] = None
        
    if data['HCBAN_CV'] is not None:
        summary = data['HCBAN_CV']['summary']
    
    metrics = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'auc': 'ROC-AUC'
    }

    if data['HCBAN_CV'] is not None:
        table_data = []
        for key, label in metrics.items():
            mean = summary.get(f'mean_{key}', 0)
            std = summary.get(f'std_{key}', 0)
            ci = 1.96 * std / np.sqrt(5)

            table_data.append({
                'Metric': label,
                'Mean': f"{mean:.4f}",
                'Std Dev': f"{std:.4f}",
                '95% CI': f"± {ci:.4f}"
            })

        df = pd.DataFrame(table_data)

        csv_path = 'results/thesis_performance_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved performance table to {csv_path}")

        try:
            latex_code = df.to_latex(index=False, caption="Performance Metrics of HCBAN (5-Fold CV)", label="tab:hcban_performance")
        except AttributeError:
            latex_code = df.style.to_latex(caption="Performance Metrics of HCBAN (5-Fold CV)", label="tab:hcban_performance")

        latex_path = 'results/thesis_performance_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_code)
        print(f"Saved LaTeX table to {latex_path}")

        print("\n--- Thesis Table (CV) ---")
        print(df.to_string(index=False))

    if 'Holdout_Test' in data:
        holdout = data['Holdout_Test']
        table_data = []
        for key, label in metrics.items():
            val = float(holdout.get(key, [0])[0]) if isinstance(holdout.get(key, 0), list) else float(holdout.get(key, 0))
            table_data.append({'Metric': label, 'Value': f"{val:.4f}"})
        df_test = pd.DataFrame(table_data)

        csv_path = 'results/thesis_holdout_test_table.csv'
        df_test.to_csv(csv_path, index=False)
        print(f"Saved holdout test table to {csv_path}")

        try:
            latex_code = df_test.to_latex(index=False, caption="Performance Metrics on Holdout Test Set", label="tab:holdout_performance")
        except AttributeError:
            latex_code = df_test.style.to_latex(caption="Performance Metrics on Holdout Test Set", label="tab:holdout_performance")

        latex_path = 'results/thesis_holdout_test_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_code)
        print(f"Saved holdout LaTeX table to {latex_path}")

        print("\n--- Thesis Table (Holdout Test) ---")
        print(df_test.to_string(index=False))

if __name__ == "__main__":
    generate_latex_table()
