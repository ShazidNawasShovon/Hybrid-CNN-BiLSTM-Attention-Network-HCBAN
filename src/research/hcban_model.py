import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
import numpy as np

class HCBAN:
    """
    Hybrid CNN-BiLSTM-Attention Network (HCBAN) for Intrusion Detection.
    Features:
    - 1D-CNN: Extracts local spatial features from packet data.
    - BiLSTM: Captures temporal/sequential dependencies.
    - Multi-Head Self-Attention: Focuses on critical parts of the sequence.
    - Residual Connections & LayerNorm: Improves gradient flow and stability.
    - Mixed Precision: Accelerates training on GPU (FP16).
    """
    def __init__(self, input_shape, n_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.model = None
        
        # Enable Mixed Precision for GPU Acceleration
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("Mixed Precision Policy enabled: mixed_float16")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # --- CNN Block (Feature Extraction) ---
        # Conv1
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Conv2
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # --- BiLSTM Block (Sequence Modeling) ---
        # Return sequences=True to feed into Attention layer
        # MultiHeadAttention expects inputs (query, value, key)
        # Here we use self-attention where Q, K, V are all the same 'x'
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # --- Attention Mechanism (Novelty) ---
        # Multi-Head Attention allows the model to jointly attend to information
        # from different representation subspaces at different positions.
        attention_output = MultiHeadAttention(num_heads=4, key_dim=128)(lstm_out, lstm_out)
        
        # Residual Connection & Normalization (Transformer-style)
        x = Add()([lstm_out, attention_output])
        x = LayerNormalization()(x)
        
        # Global Pooling to flatten
        # GlobalAveragePooling1D aggregates the sequence information
        x = GlobalAveragePooling1D()(x)
        
        # --- Dense Layers (Classification) ---
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x)
        
        # Output Layer (Softmax)
        # dtype='float32' is required for numeric stability in mixed_precision
        outputs = Dense(self.n_classes, activation='softmax', dtype='float32')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="HCBAN")
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])
        return self.model

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet.")
