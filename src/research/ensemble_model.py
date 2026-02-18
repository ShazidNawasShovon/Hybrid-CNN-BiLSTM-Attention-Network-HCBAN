import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class HybridEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes, use_gpu=True):
        self.n_classes = n_classes
        self.use_gpu = use_gpu
        self.models = {}
        self.init_models()

    def init_models(self):
        # XGBoost
        if self.n_classes == 2:
            xgb_params = {
                'n_estimators': 400,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': -1,
            }
        else:
            xgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 10,
            'objective': 'multi:softprob',
            'num_class': self.n_classes,
            'tree_method': 'hist', # 'gpu_hist' if GPU available, but 'hist' is fast on CPU too
            'random_state': 42,
            'n_jobs': -1
            }
        if self.use_gpu:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        self.models['xgb'] = XGBClassifier(**xgb_params)

        # LightGBM
        if self.n_classes == 2:
            lgbm_params = {
                'n_estimators': 400,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'objective': 'binary',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
        else:
            lgbm_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'objective': 'multiclass',
                'num_class': self.n_classes,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        # LightGBM GPU support is tricky to install, default to CPU is safer unless configured
        # We'll stick to CPU for LGBM for stability, it's very fast anyway.
        
        self.models['lgbm'] = LGBMClassifier(**lgbm_params)

        # Random Forest (for diversity)
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        )

    def fit(self, X, y, sample_weight=None):
        # Fit all models
        print("Training XGBoost...")
        try:
            self.models['xgb'].fit(X, y, sample_weight=sample_weight)
        except XGBoostError as e:
            msg = str(e)
            if "gpu_hist" in msg or "predictor" in msg:
                xgb_params = self.models['xgb'].get_params()
                xgb_params.pop('predictor', None)
                xgb_params['tree_method'] = 'hist'
                self.models['xgb'] = XGBClassifier(**xgb_params)
                self.models['xgb'].fit(X, y, sample_weight=sample_weight)
            else:
                raise
        
        print("Training LightGBM...")
        self.models['lgbm'].fit(X, y, sample_weight=sample_weight)
        
        print("Training Random Forest...")
        self.models['rf'].fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        # Get probabilities from all models
        p_xgb = self.models['xgb'].predict_proba(X)
        p_lgbm = self.models['lgbm'].predict_proba(X)
        p_rf = self.models['rf'].predict_proba(X)
        
        # Soft Voting (Average)
        # Weights can be tuned. XGB/LGBM usually better than RF.
        # Let's give slight higher weight to boosting.
        # HCBAN will be added in the pipeline separately.
        avg_prob = (0.4 * p_xgb) + (0.4 * p_lgbm) + (0.2 * p_rf)
        return avg_prob

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
