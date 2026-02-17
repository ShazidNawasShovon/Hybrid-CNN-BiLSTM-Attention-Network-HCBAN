import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from src.hybrid_dl_model import HybridDLIDS
import joblib
import os
import json
import time

class ModelTrainer:
    def __init__(self, selected_features=None):
        self.selected_features = selected_features
        self.models = {}
        self.results = {}
        
    def train(self, X_train, y_train, X_test, y_test):
        # Define models
        # Use balanced class weights where possible
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        
        # LinearSVC is faster than SVC for large datasets
        # We wrap it in CalibratedClassifierCV to get probabilities for Stacking/ROC-AUC
        svm = CalibratedClassifierCV(LinearSVC(dual=False, random_state=42, class_weight='balanced'))
        
        # Hybrid DL Model (The Novelty)
        # We pass input_shape based on number of features
        # X_train is DataFrame, so shape[1] is n_features
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        hybrid_dl = HybridDLIDS(input_shape=(n_features, 1), n_classes=n_classes, epochs=20, batch_size=256)
        
        models_to_train = {
            'Decision Tree': dt,
            'Random Forest': rf,
            'SVM (Linear)': svm,
            'Hybrid DL-Stacking': hybrid_dl
        }
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            # HybridDLIDS handles splitting internally or we can pass validation set
            # For consistency, we just call fit(X_train, y_train)
            # But HybridDLIDS expects numpy arrays ideally, but it handles DataFrame inside
            model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            print(f"{name} trained in {train_time:.2f} seconds.")
            
            # Save model
            self.models[name] = model
            if hasattr(model, 'save_model'):
                model.save_model(f'models/{name.replace(" ", "_").lower()}')
            else:
                joblib.dump(model, f'models/{name.replace(" ", "_").lower()}.joblib')
            
            # Evaluate
            print(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            # Use weighted average for multi-class
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC-AUC (One-vs-Rest)
            try:
                if y_prob is not None:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                else:
                    roc_auc = "N/A"
            except Exception as e:
                print(f"Error calculating ROC-AUC for {name}: {e}")
                roc_auc = "N/A"
            
            self.results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'Training Time': train_time
            }
            
            print(f"Results for {name}:")
            print(json.dumps(self.results[name], indent=4))
            
            # Classification Report
            report = classification_report(y_test, y_pred, zero_division=0)
            print(f"Classification Report for {name}:\n{report}")
            
        # Save results
        with open('models/evaluation_results_v2.json', 'w') as f:
            json.dump(self.results, f, indent=4)
            
        return self.results

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    X_train_full = pd.read_csv('processed_data/X_train.csv')
    X_test_full = pd.read_csv('processed_data/X_test.csv')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    # Load selected features
    with open('models/selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f]
        
    print(f"Selecting {len(selected_features)} features: {selected_features}")
    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]
    
    trainer = ModelTrainer(selected_features)
    results = trainer.train(X_train, y_train, X_test, y_test)
    
    print("Model training and evaluation complete.")
