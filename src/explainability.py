import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os

from src.hybrid_dl_model import HybridDLIDS

class ExplainabilityEngine:
    def __init__(self, model_path='models/hybrid_dl_model', feature_names=None):
        self.model_path = model_path
        self.feature_names = feature_names
        self.model = None
        self.explainer = None
        self.shap_values = None
        
    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            # Check if it's a directory (HybridDLIDS) or file
            if os.path.isdir(self.model_path):
                self.model = HybridDLIDS()
                self.model.load_model(self.model_path)
            else:
                self.model = joblib.load(self.model_path)
        else:
            # Try appending .joblib if not found
            if os.path.exists(self.model_path + '.joblib'):
                print(f"Loading model from {self.model_path}.joblib...")
                self.model = joblib.load(self.model_path + '.joblib')
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
    def compute_shap_values(self, X_sample, model_type='hybrid'):
        """
        Compute SHAP values for a sample of data.
        Args:
            X_sample (pd.DataFrame): Data sample to explain.
            model_type (str): 'tree' (RF/DT), 'kernel' (Stacking/SVM), 'hybrid' (HybridDLIDS).
        """
        if self.model is None:
            self.load_model()
            
        print(f"Computing SHAP values using {model_type} explainer...")
        
        if isinstance(self.model, HybridDLIDS):
            # For HybridDLIDS, explaining the whole ensemble is slow (KernelExplainer).
            # We can explain the XGBoost component as a proxy for feature importance
            # because Gradient Boosting usually drives the decision in tabular data.
            # This is a valid "heuristic explanation" for the thesis.
            print("Using XGBoost component of Hybrid Model for fast SHAP explanation...")
            self.explainer = shap.TreeExplainer(self.model.xgb_model)
            self.shap_values = self.explainer.shap_values(X_sample)
            
        elif model_type == 'tree':
            # TreeExplainer is optimized for trees
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample, check_additivity=False)
        else:
            # KernelExplainer for black-box models
            background = shap.kmeans(X_sample, 10)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.shap_values = self.explainer.shap_values(X_sample)
            
        return self.shap_values
    
    def plot_summary(self, X_sample, save_path='plots/shap_summary.png'):
        """
        Generate and save SHAP summary plot.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet.")
            
        plt.figure()
        # For multi-class, shap_values is a list of arrays (one for each class)
        # We plot the summary for the most probable class or all classes
        # Here we plot for the first class or a specific class of interest (e.g., Attack)
        # Or we can plot a summary of all classes if supported.
        # Summary plot usually takes the SHAP values for one class or aggregated.
        
        # If list (multi-class), pick the class with index 1 (Generic Attack?) or average
        # Let's plot for Class 1 (Backdoor?) or Class 9 (Worms?)
        # Or just the overall impact
        
        print("Generating SHAP summary plot...")
        # Plot for class 0 (Analysis) as an example, or loop through important classes
        # shap.summary_plot handles list of shap values for multi-class by color coding classes
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to {save_path}")
        
    def generate_risk_report(self, sample_index, X_sample, class_names, save_path='plots/risk_report.txt'):
        """
        Generate a text-based risk report for a specific sample.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet.")
            
        # Get the true prediction
        prediction_idx = self.model.predict(X_sample.iloc[[sample_index]])[0]
        prediction_name = class_names[prediction_idx]
        
        # Get SHAP values for this sample and the predicted class
        # shap_values is a list of arrays (one per class)
        if isinstance(self.shap_values, list):
            print(f"SHAP values is a list of length {len(self.shap_values)}")
            print(f"Shape of first element: {self.shap_values[0].shape}")
            shap_vals = self.shap_values[prediction_idx][sample_index]
        else:
            print(f"SHAP values is an array of shape {self.shap_values.shape}")
            if len(self.shap_values.shape) == 3:
                # (n_samples, n_features, n_classes)
                shap_vals = self.shap_values[sample_index, :, prediction_idx]
            else:
                shap_vals = self.shap_values[sample_index]
            
        print(f"Shape of shap_vals for report: {shap_vals.shape}")
        print(f"Shape of features: {X_sample.iloc[sample_index].values.shape}")
            
        # Create a DataFrame of feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP Value': shap_vals,
            'Feature Value': X_sample.iloc[sample_index].values
        })
        
        # Sort by absolute SHAP value
        feature_importance['Abs SHAP'] = feature_importance['SHAP Value'].abs()
        feature_importance = feature_importance.sort_values(by='Abs SHAP', ascending=False)
        
        # Generate report
        report = f"--- RISK REPORT ---\n"
        report += f"Sample Index: {sample_index}\n"
        report += f"Predicted Class: {prediction_name}\n"
        report += f"Top Contributing Features:\n"
        
        for i, row in feature_importance.head(5).iterrows():
            impact = "Increases Risk" if row['SHAP Value'] > 0 else "Decreases Risk"
            report += f"- {row['Feature']} = {row['Feature Value']:.4f} ({impact}, SHAP: {row['SHAP Value']:.4f})\n"
            
        print(report)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Risk report saved to {save_path}")
        
        # Also generate a Force Plot for this prediction
        # Force plot requires Javascript, so we save it as HTML
        try:
            force_plot = shap.force_plot(
                self.explainer.expected_value[prediction_idx],
                shap_vals,
                X_sample.iloc[sample_index],
                feature_names=self.feature_names,
                matplotlib=False,
                show=False
            )
            shap.save_html(save_path.replace('.txt', '.html'), force_plot)
            print(f"Force plot saved to {save_path.replace('.txt', '.html')}")
        except Exception as e:
            print(f"Could not generate force plot: {e}")

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    X_test_full = pd.read_csv('processed_data/X_test.csv')
    y_test = np.load('processed_data/y_test.npy')
    
    # Load selected features
    with open('models/selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f]
    
    X_test = X_test_full[selected_features]
    
    # Load Label Encoder classes
    le = joblib.load('models/label_encoder.joblib')
    class_names = le.classes_
    
    # Initialize Engine with Random Forest (TreeExplainer is robust)
    # Stacking can be explained but requires KernelExplainer which is slow.
    # For demonstration, we use RF.
    engine = ExplainabilityEngine(model_path='models/random_forest.joblib', feature_names=selected_features)
    
    # Select a small sample for explanation (e.g., 100 samples)
    # SHAP is computationally expensive
    sample_size = 100
    X_sample = X_test.iloc[:sample_size]
    
    # Compute SHAP values
    engine.compute_shap_values(X_sample, model_type='tree')
    
    # Plot Summary
    os.makedirs('plots', exist_ok=True)
    engine.plot_summary(X_sample)
    
    # Generate Risk Report for a specific attack instance (e.g., finding an attack in the sample)
    # Let's find an index where y_test is NOT 'Normal' (Normal is index 6)
    attack_indices = [i for i in range(sample_size) if y_test[i] != 6]
    if attack_indices:
        target_idx = attack_indices[0] # Pick the first attack found
        print(f"Generating report for attack sample at index {target_idx} (True Label: {class_names[y_test[target_idx]]})")
        engine.generate_risk_report(target_idx, X_sample, class_names)
    else:
        print("No attack samples found in the first 100 samples.")
