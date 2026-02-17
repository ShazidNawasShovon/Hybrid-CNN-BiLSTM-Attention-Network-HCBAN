import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, LSTM, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

from sklearn.model_selection import train_test_split

class HybridDLIDS(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=None, n_classes=None, epochs=20, batch_size=64):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.dl_model = None
        self.xgb_model = None
        self.lgbm_model = None
        self.meta_learner = None
        
    def build_dl_model(self):
        """
        Builds a 1D-CNN + BiLSTM model.
        """
        if self.input_shape is None:
            raise ValueError("Input shape must be defined.")
            
        input_layer = Input(shape=self.input_shape)
        
        # 1D-CNN Layers for Feature Extraction
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # BiLSTM Layer for Temporal/Sequential Patterns
        # return_sequences=False because we want a single vector for classification
        x = Bidirectional(LSTM(128, return_sequences=False))(x)
        x = Dropout(0.5)(x)
        
        # Dense Layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the Hybrid Model components.
        """
        # Ensure X is numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Split for internal validation if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        else:
            X_train, y_train = X, y
            
        # 1. Train Deep Learning Model (CNN-BiLSTM)
        print("Training Deep Learning Branch (1D-CNN + BiLSTM)...")
        # Reshape for CNN: (samples, features, 1)
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        
        if self.input_shape is None:
            self.input_shape = (X_train.shape[1], 1)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
            
        self.dl_model = self.build_dl_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        self.dl_model.fit(
            X_train_reshaped, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val_reshaped, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 2. Train XGBoost
        print("Training XGBoost Branch...")
        self.xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob', # or multi:softmax
            num_class=self.n_classes,
            n_jobs=-1,
            random_state=42,
            tree_method='hist', # Faster
            # device='cuda' if tf.config.list_physical_devices('GPU') else 'cpu' # Let xgboost handle detection or default to cpu
        )
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # 3. Train LightGBM
        print("Training LightGBM Branch...")
        self.lgbm_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            objective='multiclass',
            n_jobs=-1,
            random_state=42,
            verbosity=-1
        )
        self.lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # 4. Train Meta-Learner (Stacking)
        print("Training Meta-Learner (Logistic Regression)...")
        # Use validation set for meta-learner training to avoid overfitting
        # Generate probas on validation set
        dl_probs = self.dl_model.predict(X_val_reshaped)
        xgb_probs = self.xgb_model.predict_proba(X_val)
        lgbm_probs = self.lgbm_model.predict_proba(X_val)
        
        # Stack probabilities: (n_samples, n_classes * 3)
        stacked_features = np.hstack([dl_probs, xgb_probs, lgbm_probs])
        
        self.meta_learner = LogisticRegression(max_iter=1000, C=1.0)
        self.meta_learner.fit(stacked_features, y_val)
        
        print("Hybrid Training Complete.")
        return self

    def predict(self, X):
        """
        Predict class labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        dl_probs = self.dl_model.predict(X_reshaped)
        xgb_probs = self.xgb_model.predict_proba(X)
        lgbm_probs = self.lgbm_model.predict_proba(X)
        
        stacked_features = np.hstack([dl_probs, xgb_probs, lgbm_probs])
        return self.meta_learner.predict(stacked_features)

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        dl_probs = self.dl_model.predict(X_reshaped)
        xgb_probs = self.xgb_model.predict_proba(X)
        lgbm_probs = self.lgbm_model.predict_proba(X)
        
        stacked_features = np.hstack([dl_probs, xgb_probs, lgbm_probs])
        return self.meta_learner.predict_proba(stacked_features)
    
    def save_model(self, path='models/hybrid_dl_model'):
        """
        Custom save method because Pickle doesn't like Keras models in complex objects sometimes.
        """
        os.makedirs(path, exist_ok=True)
        # Save Keras model
        self.dl_model.save(os.path.join(path, 'dl_model.keras'))
        # Save sklearn/xgb models
        joblib.dump(self.xgb_model, os.path.join(path, 'xgb_model.joblib'))
        joblib.dump(self.lgbm_model, os.path.join(path, 'lgbm_model.joblib'))
        joblib.dump(self.meta_learner, os.path.join(path, 'meta_learner.joblib'))
        # Save config
        config = {
            'input_shape': self.input_shape,
            'n_classes': self.n_classes
        }
        joblib.dump(config, os.path.join(path, 'config.joblib'))
        print(f"Model saved to {path}")

    def load_model(self, path='models/hybrid_dl_model'):
        config = joblib.load(os.path.join(path, 'config.joblib'))
        self.input_shape = config['input_shape']
        self.n_classes = config['n_classes']
        
        self.dl_model = tf.keras.models.load_model(os.path.join(path, 'dl_model.keras'))
        self.xgb_model = joblib.load(os.path.join(path, 'xgb_model.joblib'))
        self.lgbm_model = joblib.load(os.path.join(path, 'lgbm_model.joblib'))
        self.meta_learner = joblib.load(os.path.join(path, 'meta_learner.joblib'))
        print(f"Model loaded from {path}")
