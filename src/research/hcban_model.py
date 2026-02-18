import torch
import torch.nn as nn
import torch.nn.functional as F

class HCBAN(nn.Module):
    """
    Hybrid CNN-BiLSTM-Attention Network (HCBAN) for Intrusion Detection.
    PyTorch Implementation.
    """
    def __init__(self, input_channels, input_length, n_classes):
        super(HCBAN, self).__init__()
        
        # --- CNN Block ---
        # Input: (Batch, 1, Features)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # --- BiLSTM Block ---
        # Input size to LSTM is channels (128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(0.3)
        
        # --- Attention Block ---
        # Bidirectional LSTM outputs 256 features (128*2)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(256)
        
        # --- Classification Head ---
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # x shape: (Batch, Features, 1) -> (Batch, Length, Channels) in Keras terms
        # PyTorch Conv1d expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1) 
        
        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # LSTM expects (Batch, Seq_Len, Features)
        # Current x: (Batch, Channels, Length) -> Permute to (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)
        
        # Attention
        # MultiheadAttention input: (Batch, Seq_Len, Embed_Dim)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual + Norm
        x = lstm_out + attn_out
        x = self.layer_norm(x)
        
        # Global Average Pooling (over sequence dimension)
        # x shape: (Batch, Seq_Len, Features)
        x = torch.mean(x, dim=1)
        
        # Dense
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) # Logits (CrossEntropyLoss handles softmax)
        
        return x
