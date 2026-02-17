import pandas as pd
import numpy as np
from src.hybrid_dl_model import HybridDLIDS
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def run_experiment():
    print("Loading all processed data...")
    # Load raw processed data (before feature selection)
    X_train = pd.read_csv('processed_data/X_train.csv')
    X_test = pd.read_csv('processed_data/X_test.csv')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    print(f"Training on all {X_train.shape[1]} features to maximize accuracy...")
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Increase complexity
    model = HybridDLIDS(input_shape=(n_features, 1), n_classes=n_classes, epochs=30, batch_size=128)
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating on Test Set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy with All Features: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_experiment()
