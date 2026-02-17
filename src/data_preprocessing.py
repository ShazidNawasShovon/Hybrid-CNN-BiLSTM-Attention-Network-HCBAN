import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

from sklearn.model_selection import train_test_split

def load_data():
    # The file names are often confusing in this dataset.
    # The 'training-set' is smaller than 'testing-set'.
    # We will combine them and re-split to ensure a proper training/testing ratio and distribution.
    train_path = 'dataset/UNSW_NB15_training-set.csv'
    test_path = 'dataset/UNSW_NB15_testing-set.csv'
    
    df1 = pd.read_csv(train_path)
    df2 = pd.read_csv(test_path)
    
    full_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    
    # Drop 'id' and 'label' immediately as they are not needed for multi-class
    # We keep 'attack_cat' as the target
    full_df = full_df.drop(['id', 'label'], axis=1)
    
    return full_df

def preprocess_data(df):
    X = df.drop('attack_cat', axis=1)
    y = df['attack_cat']
    
    # Split into train and test
    # Stratify to maintain class distribution
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify categorical and numerical columns
    cat_cols = ['proto', 'service', 'state']
    num_cols = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Get feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)
    
    # Convert back to DataFrame
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Encode target variable
    le = LabelEncoder()
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
