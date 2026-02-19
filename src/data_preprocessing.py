import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

from sklearn.model_selection import train_test_split

def load_data(dataset_type='split', combined_path=None):
    if dataset_type == 'combined':
        if combined_path is None:
            raise ValueError("Path for combined dataset must be provided.")
        print(f"Loading combined dataset from {combined_path}...")
        full_df = pd.read_csv(combined_path)
    else:
        # The file names are often confusing in this dataset.
        # The 'training-set' is smaller than 'testing-set'.
        # We will combine them and re-split to ensure a proper training/testing ratio and distribution.
        train_path = 'dataset/UNSW_NB15_training-set.csv'
        test_path = 'dataset/UNSW_NB15_testing-set.csv'
        
        print(f"Loading split datasets from {train_path} and {test_path}...")
        df1 = pd.read_csv(train_path)
        df2 = pd.read_csv(test_path)
        
        full_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    
    drop_cols = ['id']
    full_df = full_df.drop(columns=[c for c in drop_cols if c in full_df.columns], errors='ignore')
    
    return full_df

def preprocess_data(
    df,
    target_col='attack_cat',
    split_strategy='stratified',
    holdout_source=None,
    source_col='dataset_source',
    include_source_feature=True,
    feature_engineering=True,
):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")

    X = df.drop(target_col, axis=1, errors='ignore')
    if target_col == 'label' and 'attack_cat' in X.columns:
        X = X.drop(columns=['attack_cat'], errors='ignore')
    if target_col == 'attack_cat' and 'label' in X.columns:
        X = X.drop(columns=['label'], errors='ignore')
    if not include_source_feature and source_col in X.columns:
        X = X.drop(columns=[source_col], errors='ignore')
    y = df[target_col]

    if feature_engineering:
        eps = 1e-6
        if 'dbytes' in X.columns and 'dpkts' in X.columns:
            X['dbytes_per_dpkt'] = X['dbytes'] / (X['dpkts'] + eps)
        if 'spkts' in X.columns and 'dpkts' in X.columns:
            X['spkts_per_dpkts'] = X['spkts'] / (X['dpkts'] + eps)
        if 'rate' in X.columns and 'spkts' in X.columns and 'dpkts' in X.columns:
            X['rate_per_pkt'] = X['rate'] / (X['spkts'] + X['dpkts'] + eps)
        if 'dur' in X.columns:
            X['dur_log1p'] = np.log1p(np.maximum(X['dur'].astype(float), 0.0))
        if 'dwin' in X.columns and 'rate' in X.columns:
            X['dwin_x_rate'] = X['dwin'] * X['rate']
    
    if split_strategy == 'source_holdout':
        if source_col not in df.columns:
            raise ValueError(f"Source column '{source_col}' not found for source_holdout split.")
        if holdout_source is None:
            raise ValueError("holdout_source must be provided for source_holdout split.")
        mask = df[source_col].astype(str) == str(holdout_source)
        X_train_raw = X.loc[~mask].copy()
        y_train = y.loc[~mask].copy()
        X_test_raw = X.loc[mask].copy()
        y_test = y.loc[mask].copy()
        if len(X_test_raw) == 0:
            raise ValueError(f"No rows found for holdout_source='{holdout_source}'.")
    else:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Identify categorical and numerical columns dynamically
    # The standard dataset has 'proto', 'service', 'state' as categorical
    # But sometimes they might be missing or named differently
    
    # Auto-detect categorical columns
    cat_cols = [col for col in X_train_raw.columns if X_train_raw[col].dtype == 'object']
    # Ensure specific known categorical columns are included if they exist
    known_cats = ['proto', 'service', 'state']
    for c in known_cats:
        if c in X_train_raw.columns and c not in cat_cols:
            cat_cols.append(c)
            
    num_cols = [col for col in X_train_raw.columns if col not in cat_cols]
    
    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {len(num_cols)}")
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Handle missing values
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle preprocessing
    transformers = [
        ('num', numerical_transformer, num_cols)
    ]
    if cat_cols:
        transformers.append(('cat', categorical_transformer, cat_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit and transform training data
    # X_train_processed is a numpy array
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Get feature names
    feature_names = num_cols.copy()
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            feature_names.extend(list(cat_feature_names))
        except Exception as e:
            print(f"Warning: Could not retrieve feature names from OneHotEncoder: {e}")
            # Fallback: generate dummy names if needed
            # The number of new columns is total columns - num_cols
            n_new = X_train_processed.shape[1] - len(num_cols)
            feature_names.extend([f"cat_{i}" for i in range(n_new)])
            
    # Convert back to DataFrame
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Encode target variable
    le = LabelEncoder()
    y_all = y.astype(str)
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    le.fit(y_all)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Save the encoders and preprocessor
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    
    return X_train_processed_df, X_test_processed_df, y_train_encoded, y_test_encoded, feature_names, le

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Total data shape: {df.shape}")
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, features, le = preprocess_data(df)
    
    print(f"Processed Train Shape: {X_train.shape}")
    print(f"Processed Test Shape: {X_test.shape}")
    print(f"Classes: {le.classes_}")
    
    # Save processed data
    os.makedirs('processed_data', exist_ok=True)
    X_train.to_csv('processed_data/X_train.csv', index=False)
    X_test.to_csv('processed_data/X_test.csv', index=False)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/y_test.npy', y_test)
    
    print("Data preprocessing complete.")
