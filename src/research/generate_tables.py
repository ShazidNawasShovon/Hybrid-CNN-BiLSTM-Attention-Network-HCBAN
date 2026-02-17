import json
import pandas as pd
import numpy as np
import os

def generate_latex_table(results_path='results/research_results.json'):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)
        
    if 'HCBAN_CV' not in data:
        print("HCBAN_CV results not found.")
        return
        
    summary = data['HCBAN_CV']['summary']
    
    # Define metrics to display
    metrics = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'auc': 'ROC-AUC'
    }
    
    # Create DataFrame for Table
    table_data = []
    for key, label in metrics.items():
        mean = summary.get(f'mean_{key}', 0)
        std = summary.get(f'std_{key}', 0)
        # Assuming 5 folds for CI calculation
        ci = 1.96 * std / np.sqrt(5)
        
        table_data.append({
            'Metric': label,
            'Mean': f"{mean:.4f}",
            'Std Dev': f"{std:.4f}",
            '95% CI': f"± {ci:.4f}"
        })
        
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = 'results/thesis_performance_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved performance table to {csv_path}")
    
    # Generate LaTeX code
    # to_latex was deprecated in newer pandas versions, using style.to_latex if available or manual
    try:
        latex_code = df.to_latex(index=False, caption="Performance Metrics of HCBAN (5-Fold CV)", label="tab:hcban_performance")
    except AttributeError:
        latex_code = df.style.to_latex(caption="Performance Metrics of HCBAN (5-Fold CV)", label="tab:hcban_performance")
    
    latex_path = 'results/thesis_performance_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    print(f"Saved LaTeX table to {latex_path}")
    
    print("\n--- Thesis Table ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    generate_latex_table()
