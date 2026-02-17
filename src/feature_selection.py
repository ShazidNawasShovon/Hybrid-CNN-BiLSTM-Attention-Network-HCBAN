import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import time

class HybridFeatureSelector:
    def __init__(self, n_features_to_select=20, n_filter_features=50, sample_size=None):
        """
        Initialize the Hybrid Feature Selector.
        
        Args:
            n_features_to_select (int): Final number of features to select using Wrapper method.
            n_filter_features (int): Number of features to select in the Filter step.
            sample_size (float or int): Fraction or number of samples to use for feature selection.
                                      If None, use all data.
        """
        self.n_features_to_select = n_features_to_select
        self.n_filter_features = n_filter_features
        self.sample_size = sample_size
        self.selected_features_ = None
        self.filter_selector_ = None
        self.wrapper_selector_ = None
        
    def fit(self, X, y):
        """
        Fit the feature selector to the data.
        """
        # Subsample if requested
        if self.sample_size:
            if isinstance(self.sample_size, float):
                n_samples = int(len(X) * self.sample_size)
            else:
                n_samples = self.sample_size
            
            # Use random sampling
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
            
        print(f"Starting Feature Selection with {len(X_sample)} samples...")
        start_time = time.time()
        
        # Step 1: Filter Method (Mutual Information)
        print(f"Step 1: Filter Method (Mutual Information) - Selecting top {self.n_filter_features} features...")
        self.filter_selector_ = SelectKBest(score_func=mutual_info_classif, k=self.n_filter_features)
        self.filter_selector_.fit(X_sample, y_sample)
        
        # Get columns selected by filter
        filter_mask = self.filter_selector_.get_support()
        filter_selected_features = X_sample.columns[filter_mask]
        print(f"Filter method selected {len(filter_selected_features)} features.")
        
        # Transform data for Wrapper method
        X_filtered = X_sample[filter_selected_features]
        
        # Step 2: Wrapper Method (RFE with Random Forest)
        print(f"Step 2: Wrapper Method (RFE) - Selecting top {self.n_features_to_select} features...")
        # Use a lighter model for RFE to be faster
        estimator = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        self.wrapper_selector_ = RFE(estimator=estimator, n_features_to_select=self.n_features_to_select, step=0.1)
        self.wrapper_selector_.fit(X_filtered, y_sample)
        
        # Get final selected features
        wrapper_mask = self.wrapper_selector_.get_support()
        self.selected_features_ = filter_selected_features[wrapper_mask]
        
        end_time = time.time()
        print(f"Feature Selection completed in {end_time - start_time:.2f} seconds.")
        print(f"Selected {len(self.selected_features_)} features: {self.selected_features_.tolist()}")
        
        return self
    
    def transform(self, X):
        """
        Transform the dataset to keep only selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("Feature selector has not been fitted yet.")
        return X[self.selected_features_]

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv('processed_data/X_train.csv')
    y_train = np.load('processed_data/y_train.npy')
    
    # Initialize selector
    # We use a sample of 20,000 rows for speed, or full data if feasible.
    # Mutual Info on 200k rows with 195 features can be slow.
    # Let's use 20% sample (~40k rows).
    selector = HybridFeatureSelector(n_features_to_select=20, n_filter_features=50, sample_size=0.2)
    
    # Fit selector
    selector.fit(X_train, y_train)
    
    # Save selector
    os.makedirs('models', exist_ok=True)
    joblib.dump(selector, 'models/feature_selector.joblib')
    
    # Save selected features list
    with open('models/selected_features.txt', 'w') as f:
        for feature in selector.selected_features_:
            f.write(f"{feature}\n")
            
    print("Feature selection model saved.")
