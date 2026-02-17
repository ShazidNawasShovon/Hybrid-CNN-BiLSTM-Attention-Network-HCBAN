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
    
    # Drop 'id' and 'label' immediately as they are not needed for multi-class
    # We keep 'attack_cat' as the target
    # Check if columns exist before dropping
    drop_cols = ['id', 'label']
    full_df = full_df.drop(columns=[c for c in drop_cols if c in full_df.columns], errors='ignore')
    
    return full_df

def preprocess_data(df):
    # Check if 'attack_cat' exists
    if 'attack_cat' not in df.columns:
        # It might be named differently in the combined dataset or split files
        # Let's print columns to debug if needed, but for now assuming standard UNSW-NB15 format
        # Common variations: 'attack_cat', 'cat'
        pass

    X = df.drop('attack_cat', axis=1, errors='ignore')
    y = df['attack_cat'] if 'attack_cat' in df.columns else df.iloc[:, -1] # Fallback to last column
    
    # Split into train and test
    # Stratify to maintain class distribution
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
            # Fallback: generate dummy names if needed, but usually get_feature_names_out works
            
    # Convert back to DataFrame
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Encode target variable
    le = LabelEncoder()
    # Ensure y is string type for LabelEncoder
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    
    y_train_encoded = le.fit_transform(y_train)
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
