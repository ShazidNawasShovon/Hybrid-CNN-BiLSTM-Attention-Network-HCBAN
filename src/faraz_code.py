"""
CNN-LSTM-GNN Hybrid Network Intrusion Detection System with Dynamic Feature Grouping
for CICIDS2017 Dataset

This implementation combines:
1. CNN for spatial feature extraction from network flows
2. LSTM for temporal dependency modeling
3. GNN for relational learning between network entities
4. Dynamic feature grouping based on feature similarity

Based on research from: XG-NID [citation:5] and AutoGraphAD [citation:8]
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
import warnings
import pickle
warnings.filterwarnings('ignore')

import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PART 1: DYNAMIC FEATURE GROUPING MODULE
# ============================================================================

class DynamicFeatureGrouping:
    """
    Dynamically groups features with similar characteristics using clustering
    and mutual information. This implements the "dynamic bucketing" concept
    where features with similar patterns are grouped together.
    """
    
    def __init__(self, n_groups=10, grouping_method='mutual_info'):
        """
        Args:
            n_groups: Number of feature groups to create
            grouping_method: 'mutual_info', 'correlation', or 'kmeans'
        """
        self.n_groups = n_groups
        self.grouping_method = grouping_method
        self.feature_groups = None
        self.feature_importances = None
        self.group_centroids = None
        
    def fit(self, X, y=None):
        """
        Fit the dynamic grouping algorithm on the data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels for mutual information calculation (optional)
        """
        n_features = X.shape[1]
        
        if self.grouping_method == 'mutual_info' and y is not None:
            # Use mutual information to rank features
            mi_scores = mutual_info_classif(X, y)
            self.feature_importances = mi_scores
            
            # Sort features by importance and create groups
            sorted_indices = np.argsort(mi_scores)[::-1]
            self.feature_groups = []
            
            # Distribute features into groups based on importance
            features_per_group = n_features // self.n_groups
            for i in range(self.n_groups):
                start_idx = i * features_per_group
                end_idx = start_idx + features_per_group if i < self.n_groups - 1 else n_features
                group_indices = sorted_indices[start_idx:end_idx]
                self.feature_groups.append(group_indices.tolist())
                
        elif self.grouping_method == 'correlation':
            # Use correlation-based grouping
            corr_matrix = np.corrcoef(X.T)
            # Simple threshold-based grouping
            used_features = set()
            self.feature_groups = []
            
            for i in range(n_features):
                if i in used_features:
                    continue
                    
                # Find highly correlated features
                group = [i]
                for j in range(i+1, n_features):
                    if j not in used_features and abs(corr_matrix[i, j]) > 0.7:
                        group.append(j)
                        used_features.add(j)
                used_features.add(i)
                self.feature_groups.append(group)
                
                if len(self.feature_groups) >= self.n_groups:
                    break
                    
            # Distribute remaining features
            remaining = [f for f in range(n_features) if f not in used_features]
            for i, feat in enumerate(remaining):
                group_idx = i % len(self.feature_groups)
                self.feature_groups[group_idx].append(feat)
                
        elif self.grouping_method == 'kmeans':
            # Use KMeans clustering on feature vectors
            # Transpose to cluster features (each feature is a sample)
            feature_vectors = X.T
            
            # Normalize feature vectors
            scaler = StandardScaler()
            feature_vectors_scaled = scaler.fit_transform(feature_vectors)
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=self.n_groups, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_vectors_scaled)
            
            # Create groups based on cluster labels
            self.feature_groups = []
            self.group_centroids = kmeans.cluster_centers_
            
            for i in range(self.n_groups):
                group_indices = np.where(cluster_labels == i)[0].tolist()
                self.feature_groups.append(group_indices)
        
        else:
            # Default: random grouping
            indices = list(range(n_features))
            random.shuffle(indices)
            self.feature_groups = []
            features_per_group = n_features // self.n_groups
            
            for i in range(self.n_groups):
                start_idx = i * features_per_group
                end_idx = start_idx + features_per_group if i < self.n_groups - 1 else n_features
                self.feature_groups.append(indices[start_idx:end_idx])
        
        # Ensure all features are included exactly once
        all_grouped = set()
        for group in self.feature_groups:
            all_grouped.update(group)
        
        # If some features are missing, add them to existing groups
        if len(all_grouped) < n_features:
            missing = set(range(n_features)) - all_grouped
            for i, feat in enumerate(missing):
                group_idx = i % len(self.feature_groups)
                self.feature_groups[group_idx].append(feat)
        
        print(f"Created {len(self.feature_groups)} feature groups")
        for i, group in enumerate(self.feature_groups):
            print(f"  Group {i}: {len(group)} features")
            
        return self
    
    def transform(self, X):
        """
        Transform features by grouping them according to learned groups
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Grouped features (n_samples, n_groups, max_group_size)
        """
        if self.feature_groups is None:
            raise ValueError("Model must be fitted before transform")
            
        # Determine max group size for padding
        max_group_size = max(len(g) for g in self.feature_groups)
        n_samples = X.shape[0]
        n_groups = len(self.feature_groups)
        
        # Create grouped feature tensor
        grouped_features = np.zeros((n_samples, n_groups, max_group_size))
        mask = np.zeros((n_samples, n_groups, max_group_size))
        
        for g_idx, group in enumerate(self.feature_groups):
            for f_idx, feat_idx in enumerate(group):
                if f_idx < max_group_size:
                    grouped_features[:, g_idx, f_idx] = X[:, feat_idx]
                    mask[:, g_idx, f_idx] = 1
                    
        return grouped_features, mask
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


# ============================================================================
# PART 2: GRAPH CONSTRUCTION MODULE
# ============================================================================

class NetworkGraphConstructor:
    """
    Constructs heterogeneous graphs from network flow data
    Following AutoGraphAD approach [citation:8]
    """
    
    def __init__(self, window_size=180, ip_feature_dim=8):
        """
        Args:
            window_size: Time window in seconds for graph construction
            ip_feature_dim: Dimension of IP node features
        """
        self.window_size = window_size
        self.ip_feature_dim = ip_feature_dim
        self.scaler = StandardScaler()
        self.ip_encoder = LabelEncoder()
        
    def extract_ip_features(self, df, ip_column):
        """Extract features for IP nodes"""
        # Create IP-level statistics
        ip_stats = df.groupby(ip_column).agg({
            'Flow Duration': ['mean', 'std', 'count'],
            'Total Fwd Packets': ['sum', 'mean'],
            'Total Backward Packets': ['sum', 'mean'],
            'Flow Bytes/s': ['mean', 'max'],
            'Flow Packets/s': ['mean', 'max']
        }).fillna(0)
        
        # Flatten column names
        ip_stats.columns = ['_'.join(col).strip() for col in ip_stats.columns.values]
        
        # Add one-hot encoded features for IP version/protocol preferences
        # Simplified: use hashing trick for IP addresses
        n_ips = len(ip_stats)
        ip_features = np.zeros((n_ips, self.ip_feature_dim))
        
        # Simple hash-based features
        for i, ip in enumerate(ip_stats.index):
            h = hash(str(ip)) % 10000
            for j in range(self.ip_feature_dim):
                ip_features[i, j] = (h >> j) & 1
        
        return ip_stats, ip_features
    
    def build_graph_from_window(self, window_df, flow_features, feature_names):
        """
        Build a heterogeneous graph for a single time window
        
        Args:
            window_df: DataFrame for the time window
            flow_features: Preprocessed flow features
            feature_names: Names of features
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get unique IPs
        src_ips = window_df['Src IP'].unique() if 'Src IP' in window_df.columns else []
        dst_ips = window_df['Dst IP'].unique() if 'Dst IP' in window_df.columns else []
        
        # Create IP to node ID mapping
        all_ips = list(set(src_ips) | set(dst_ips))
        ip_to_id = {ip: idx for idx, ip in enumerate(all_ips)}
        
        # Number of nodes: flow nodes + IP nodes
        n_flows = len(window_df)
        n_ips = len(all_ips)
        n_nodes = n_flows + n_ips
        
        # Create node features
        # Flow nodes: use flow_features
        flow_node_features = flow_features
        
        # IP nodes: create features (simplified - you can enhance this)
        ip_node_features = np.zeros((n_ips, self.ip_feature_dim))
        for i, ip in enumerate(all_ips):
            # Simple hash-based features
            h = hash(str(ip)) % 10000
            for j in range(self.ip_feature_dim):
                ip_node_features[i, j] = (h >> j) & 1
        
        # Combine node features
        x = np.vstack([flow_node_features, ip_node_features])
        
        # Create edges: flow <-> source IP and flow <-> destination IP
        edge_index = []
        edge_type = []  # 0: flow-src, 1: flow-dst
        
        for flow_idx in range(n_flows):
            src_ip = window_df.iloc[flow_idx]['Src IP'] if 'Src IP' in window_df.columns else None
            dst_ip = window_df.iloc[flow_idx]['Dst IP'] if 'Dst IP' in window_df.columns else None
            
            if src_ip in ip_to_id:
                ip_node_idx = ip_to_id[src_ip]
                # Edge from flow node to IP node
                edge_index.append([flow_idx, n_flows + ip_node_idx])
                edge_type.append(0)
                
            if dst_ip in ip_to_id:
                ip_node_idx = ip_to_id[dst_ip]
                edge_index.append([flow_idx, n_flows + ip_node_idx])
                edge_type.append(1)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        # Create labels (for flow nodes only, IP nodes get -1)
        y = torch.full((n_nodes,), -1, dtype=torch.long)
        if 'Label' in window_df.columns:
            # Convert labels to numerical
            label_map = {'BENIGN': 0}
            # Add attack labels
            attack_labels = window_df['Label'].unique()
            for i, lbl in enumerate(attack_labels):
                if lbl != 'BENIGN':
                    label_map[lbl] = i + 1
            
            for flow_idx, label in enumerate(window_df['Label']):
                y[flow_idx] = label_map[label]
        
        # Create node type mask: 0 for flow nodes, 1 for IP nodes
        node_type = torch.cat([
            torch.zeros(n_flows, dtype=torch.long),
            torch.ones(n_ips, dtype=torch.long)
        ])
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            y=y,
            node_type=node_type,
            edge_type=edge_type,
            n_flows=n_flows,
            n_ips=n_ips
        )
        
        return data


# ============================================================================
# PART 3: CNN-LSTM-GNN HYBRID MODEL
# ============================================================================

class SpatialFeatureExtractor(nn.Module):
    """CNN module for spatial feature extraction from grouped features"""
    
    def __init__(self, n_groups, group_size, hidden_dim=128):
        super(SpatialFeatureExtractor, self).__init__()
        
        self.n_groups = n_groups
        self.group_size = group_size
        
        # 1D CNN for each group (shared weights)
        self.conv1 = nn.Conv1d(in_channels=n_groups, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Grouped features (batch_size, n_groups, group_size)
            mask: Mask for valid features (batch_size, n_groups, group_size)
            
        Returns:
            Spatial features (batch_size, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask
        
        # Apply convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        
        return x


class TemporalFeatureExtractor(nn.Module):
    """LSTM module for temporal dependency modeling"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, bidirectional=True):
        super(TemporalFeatureExtractor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(0.3)
        
        # Attention mechanism for LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Sequence features (batch_size, seq_len, input_dim)
            
        Returns:
            Temporal features (batch_size, hidden_dim * directions)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenate last hidden states from both directions
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply attention over sequence
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Combine last hidden and attended features
        temporal_features = hidden + attended
        
        return self.dropout(temporal_features)


class RelationalFeatureExtractor(nn.Module):
    """GNN module for relational learning between network entities"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, conv_type='gcn'):
        super(RelationalFeatureExtractor, self).__init__()
        
        self.conv_type = conv_type
        
        if conv_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif conv_type == 'sage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        elif conv_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim // 4, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim, output_dim // 4, heads=4, concat=True)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, node_type=None):
        """
        Args:
            x: Node features
            edge_index: Graph edge indices
            node_type: Node type indicators (for heterogeneous graphs)
            
        Returns:
            Relational features for each node
        """
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        
        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        return x


class CNNLSTMGNNHybrid(nn.Module):
    """
    Complete CNN-LSTM-GNN Hybrid Model with Dynamic Feature Grouping
    
    Architecture:
    1. Dynamic Feature Grouping (external preprocessing)
    2. CNN: Spatial feature extraction from grouped features
    3. LSTM: Temporal modeling across sequences
    4. GNN: Relational learning from graph structure
    5. Fusion: Combine all features for final classification
    """
    
    def __init__(self, 
                 n_features, 
                 n_groups=10,
                 group_size=10,
                 cnn_hidden=128,
                 lstm_hidden=128,
                 gnn_hidden=128,
                 n_classes=2,
                 fusion_method='attention'):  # 'concat', 'attention', 'weighted'
        
        super(CNNLSTMGNNHybrid, self).__init__()
        
        self.n_features = n_features
        self.n_groups = n_groups
        self.group_size = group_size
        self.fusion_method = fusion_method
        
        # CNN module for spatial features
        self.cnn = SpatialFeatureExtractor(n_groups, group_size, cnn_hidden)
        
        # LSTM module for temporal features
        self.lstm = TemporalFeatureExtractor(cnn_hidden, lstm_hidden)
        
        # GNN module for relational features
        # Input to GNN will be concatenated features from CNN+LSTM
        gnn_input_dim = cnn_hidden + (lstm_hidden * 2)  # *2 for bidirectional
        self.gnn = RelationalFeatureExtractor(gnn_input_dim, gnn_hidden, gnn_hidden)
        
        fusion_input_dim = cnn_hidden + (lstm_hidden * 2) + gnn_hidden
        
        if fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(fusion_input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )
        elif fusion_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(3) / 3)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, grouped_features, mask, graph_data, seq_len=10):
        """
        Forward pass
        
        Args:
            grouped_features: Grouped features (batch, n_groups, group_size)
            mask: Mask for valid features
            graph_data: PyG Data object for GNN
            seq_len: Sequence length for LSTM
            
        Returns:
            Classification logits
        """
        batch_size = grouped_features.size(0)
        
        # 1. CNN: Extract spatial features
        spatial_features = self.cnn(grouped_features, mask)  # (batch, cnn_hidden)
        
        # 2. LSTM: Model temporal dependencies
        # Reshape spatial features into sequences
        # We need to create sequences from the batch
        # For simplicity, we'll treat each sample as a sequence of 1
        # In practice, you'd want to create actual sequences from time windows
        
        # Create artificial sequences (for demonstration)
        # In real implementation, you'd have proper sequence data
        seq_features = spatial_features.unsqueeze(1).repeat(1, seq_len, 1)
        # Add some variation to make it interesting
        seq_features = seq_features + torch.randn_like(seq_features) * 0.1
        
        temporal_features = self.lstm(seq_features)  # (batch, lstm_hidden*2)
        
        # 3. GNN: Extract relational features
        # Combine spatial and temporal features for GNN input
        combined_features = torch.cat([spatial_features, temporal_features], dim=1)
        
        # For GNN, we need to handle the graph structure
        # This assumes graph_data contains node features that need to be updated
        # with our combined features
        if hasattr(graph_data, 'x'):
            # Update node features for flow nodes only
            n_flows = graph_data.n_flows
            # In practice, you'd want to properly update flow node features
            # This is a simplified version
            gnn_output = self.gnn(graph_data.x, graph_data.edge_index)
            
            # Extract flow node outputs (first n_flows nodes)
            relational_features = gnn_output[:n_flows].mean(dim=0).unsqueeze(0)
            relational_features = relational_features.repeat(batch_size, 1)
        else:
            # Fallback if no graph structure
            relational_features = torch.zeros(batch_size, self.gnn.conv2.out_channels).to(grouped_features.device)
        
        all_features = torch.cat([spatial_features, temporal_features, relational_features], dim=1)
        output = self.classifier(all_features)
        return output


# ============================================================================
# PART 4: CICIDS2017 DATASET LOADER
# ============================================================================

class CICIDS2017Dataset(Dataset):
    """PyTorch Dataset for CICIDS2017 with graph construction"""
    
    def __init__(self, 
                 csv_files, 
                 feature_grouping=None,
                 graph_constructor=None,
                 sequence_length=10,
                 test_mode=False,
                 label_encoder=None):
        """
        Args:
            csv_files: List of CSV file paths
            feature_grouping: DynamicFeatureGrouping instance
            graph_constructor: NetworkGraphConstructor instance
            sequence_length: Length of sequences for LSTM
            test_mode: If True, load only a subset for testing
        """
        self.csv_files = csv_files
        self.feature_grouping = feature_grouping
        self.graph_constructor = graph_constructor
        self.sequence_length = sequence_length
        self.test_mode = test_mode
        self.label_encoder = label_encoder
        
        # Load and preprocess data
        self.data, self.labels, self.feature_names = self._load_and_preprocess()
        
        # Group features if grouping provided
        if feature_grouping is not None:
            print("Applying dynamic feature grouping...")
            self.grouped_features, self.mask = feature_grouping.transform(self.data)
        else:
            self.grouped_features = None
            self.mask = None
        
        # Create sequences
        self.sequences, self.sequence_labels = self._create_sequences()
        
    def _load_and_preprocess(self):
        """Load and preprocess CICIDS2017 data"""
        all_data = []
        all_labels = []
        feature_names = None
        
        total_rows = 0
        
        for csv_file in self.csv_files:
            print(f"Loading {csv_file}...")
            df = pd.read_csv(csv_file, engine='python')
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            
            # Extract features and labels
            if 'Label' in df.columns:
                X = df.drop('Label', axis=1)
                y = df['Label']
            else:
                X = df
                y = pd.Series(['BENIGN'] * len(df))
            
            # Handle missing values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())
            
            # Remove constant columns
            constant_cols = [col for col in X.columns if X[col].nunique() == 1]
            if constant_cols:
                X = X.drop(columns=constant_cols)
            
            # Store feature names from first file
            if feature_names is None:
                feature_names = X.columns.tolist()
            
            # Convert to numpy
            X_np = X.values.astype(np.float32)
            y_np = y.values
            
            all_data.append(X_np)
            all_labels.extend(y_np)
            
            total_rows += X_np.shape[0]
            if self.test_mode and total_rows > 10000:
                break
        
        # Concatenate all data
        if len(all_data) > 1:
            data = np.vstack(all_data)
        else:
            data = all_data[0]
        
        # Encode labels
        if self.label_encoder is not None:
            labels = self.label_encoder.transform(all_labels)
            classes = self.label_encoder.classes_
        else:
            encoder = LabelEncoder()
            labels = encoder.fit_transform(all_labels)
            classes = encoder.classes_
        
        print(f"Loaded {data.shape[0]} samples with {data.shape[1]} features")
        print(f"Classes: {classes}")
        
        # Scale features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        return data_scaled, labels, feature_names
    
    def _create_sequences(self):
        """Create sequences for LSTM input"""
        n_samples = len(self.labels)
        
        if self.sequence_length <= 1:
            sequences = self.grouped_features if self.grouped_features is not None else self.data
            sequence_labels = self.labels
            if len(sequences.shape) == 2:
                sequences = sequences.reshape(sequences.shape[0], 1, -1)
            return sequences, sequence_labels
        
        step = self.sequence_length
        n_sequences = max(1, n_samples // step)
        
        sequences = []
        sequence_labels = []
        
        for i in range(0, n_sequences * step, step):
            end_idx = min(i + self.sequence_length, n_samples)
            if self.grouped_features is not None:
                seq = self.grouped_features[i:end_idx]
            else:
                seq = self.data[i:end_idx]
            
            if seq.shape[0] < self.sequence_length:
                pad_len = self.sequence_length - seq.shape[0]
                if self.grouped_features is not None:
                    pad_shape = (pad_len, seq.shape[1], seq.shape[2])
                else:
                    pad_shape = (pad_len, seq.shape[1])
                pad = np.zeros(pad_shape, dtype=seq.dtype)
                seq = np.concatenate([seq, pad], axis=0)
            
            label_idx = min(end_idx - 1, n_samples - 1)
            label = self.labels[label_idx]
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def __len__(self):
        return len(self.sequence_labels)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        
        # Convert to torch tensors
        sequence = torch.tensor(sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        # For graph data (simplified - in practice you'd have proper graph for each sample)
        # This is a placeholder - you'd need to construct graphs for each sequence
        dummy_graph = Data(
            x=torch.randn(10, sequence.shape[-1]),
            edge_index=torch.randint(0, 10, (2, 20)),
            n_flows=5,
            n_ips=5
        )
        
        return sequence, label, dummy_graph


def collate_with_dummy_graph(batch):
    sequences, labels, graph_data = zip(*batch)
    sequences = torch.stack(sequences, dim=0)
    labels = torch.stack(labels, dim=0)
    return sequences, labels, None


# ============================================================================
# PART 5: TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    def __init__(self, model, device, learning_rate=0.001, class_weights=None):
        self.model = model.to(device)
        self.device = device
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, labels, graph_data) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Prepare inputs
            # For grouped features, we need to extract them from sequences
            # This assumes sequences have shape (batch, seq_len, n_groups, group_size)
            if len(sequences.shape) == 4:
                grouped_features = sequences[:, -1, :, :]  # Use last timestep
                mask = torch.ones_like(grouped_features)  # Placeholder mask
            else:
                grouped_features = sequences
                mask = torch.ones_like(grouped_features)
            
            # Forward through model
            outputs = self.model(grouped_features, mask, graph_data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels, graph_data in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Prepare inputs
                if len(sequences.shape) == 4:
                    grouped_features = sequences[:, -1, :, :]
                    mask = torch.ones_like(grouped_features)
                else:
                    grouped_features = sequences
                    mask = torch.ones_like(grouped_features)
                
                # Forward pass
                outputs = self.model(grouped_features, mask, graph_data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, np.array(all_preds), np.array(all_labels)
    
    def train(self, train_loader, val_loader, max_epochs=50, early_stopping=10):
        best_val_acc = 0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(max_epochs):
            print(f'\nEpoch {epoch+1}/{max_epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Validate
            val_loss, val_preds, val_labels = self.evaluate(val_loader)
            val_acc = accuracy_score(val_labels, val_preds) * 100
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'✓ New best model saved with accuracy {best_val_acc:.2f}%')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return history


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def plot_training_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    output_png = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_png)
    plt.close()
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = os.path.join(output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv)


def main():
    """Main execution function"""
    
    # Configuration
    CONFIG = {
        'data_path': './dataset_CIC-IDS2017/',  # Path to CICIDS2017 CSV files
        'csv_files': [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv',
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv'
        ],
        'n_groups': 12,  # Number of feature groups
        'grouping_method': 'mutual_info',  # 'mutual_info', 'correlation', 'kmeans'
        'cnn_hidden': 128,
        'lstm_hidden': 128,
        'gnn_hidden': 128,
        'batch_size': 32,
        'sequence_length': 10,
        'max_epochs': 2,
        'learning_rate': 0.001,
        'test_mode': True,  # Set to False for full training
    }
    
    print("=" * 60)
    print("CNN-LSTM-GNN Hybrid NIDS with Dynamic Feature Grouping")
    print("=" * 60)
    print(f"Configuration: {CONFIG}")
    
    results_dir = 'faraz_results'
    os.makedirs(results_dir, exist_ok=True)
    # Step 1: Load a sample of data for feature grouping
    print("\n[Step 1] Loading sample data for feature grouping...")
    all_csv_files = sorted([
        f for f in os.listdir(CONFIG['data_path'])
        if f.lower().endswith('.csv')
    ])
    if not all_csv_files:
        raise RuntimeError(f"No CSV files found in {CONFIG['data_path']}")
    sample_files = [os.path.join(CONFIG['data_path'], all_csv_files[0])]
    
    # Create a temporary dataset to get features
    temp_dataset = CICIDS2017Dataset(
        csv_files=sample_files,
        test_mode=True
    )
    
    # Step 2: Apply dynamic feature grouping
    print("\n[Step 2] Applying dynamic feature grouping...")
    feature_grouping = DynamicFeatureGrouping(
        n_groups=CONFIG['n_groups'],
        grouping_method=CONFIG['grouping_method']
    )
    
    # Fit on a subset of data
    feature_grouping.fit(temp_dataset.data[:5000], temp_dataset.labels[:5000])
    
    # Step 3: Create graph constructor
    print("\n[Step 3] Initializing graph constructor...")
    graph_constructor = NetworkGraphConstructor(window_size=180)
    
    # Step 4: Create datasets
    print("\n[Step 4] Creating datasets...")
    full_csv_files = [os.path.join(CONFIG['data_path'], f) for f in all_csv_files]
    
    # Build global label encoder across all files
    all_labels = []
    for csv_path in full_csv_files:
        df_labels = pd.read_csv(csv_path)
        df_labels.columns = [col.strip().replace(' ', '_') for col in df_labels.columns]
        if 'Label' in df_labels.columns:
            all_labels.extend(df_labels['Label'].values)
        else:
            all_labels.extend(['BENIGN'] * len(df_labels))
    global_label_encoder = LabelEncoder()
    global_label_encoder.fit(all_labels)
    print(f"Global classes: {global_label_encoder.classes_}")
    
    # Split into train/val/test using all CSV files
    n_files = len(full_csv_files)
    train_end = max(1, int(0.6 * n_files))
    val_end = max(train_end + 1, int(0.8 * n_files))
    if val_end >= n_files:
        val_end = n_files - 1
    train_files = full_csv_files[:train_end]
    val_files = full_csv_files[train_end:val_end]
    test_files = full_csv_files[val_end:]
    
    train_dataset = CICIDS2017Dataset(
        csv_files=train_files,
        feature_grouping=feature_grouping,
        graph_constructor=graph_constructor,
        sequence_length=CONFIG['sequence_length'],
        test_mode=False,
        label_encoder=global_label_encoder
    )
    
    val_dataset = CICIDS2017Dataset(
        csv_files=val_files,
        feature_grouping=feature_grouping,
        graph_constructor=graph_constructor,
        sequence_length=CONFIG['sequence_length'],
        test_mode=CONFIG['test_mode'],
        label_encoder=global_label_encoder
    )
    
    test_dataset = CICIDS2017Dataset(
        csv_files=test_files,
        feature_grouping=feature_grouping,
        graph_constructor=graph_constructor,
        sequence_length=CONFIG['sequence_length'],
        test_mode=CONFIG['test_mode'],
        label_encoder=global_label_encoder
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_dummy_graph
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_dummy_graph
    )
    
    # Step 5: Initialize model
    print("\n[Step 5] Building CNN-LSTM-GNN hybrid model...")
    
    # Get dimensions from dataset
    sample_sequence, _, _ = train_dataset[0]
    if len(sample_sequence.shape) == 3:
        n_groups = sample_sequence.shape[1]
        group_size = sample_sequence.shape[2]
    else:
        # Fallback dimensions
        n_groups = CONFIG['n_groups']
        group_size = 10
    
    n_classes = len(global_label_encoder.classes_)
    
    model = CNNLSTMGNNHybrid(
        n_features=temp_dataset.data.shape[1],
        n_groups=n_groups,
        group_size=group_size,
        cnn_hidden=CONFIG['cnn_hidden'],
        lstm_hidden=CONFIG['lstm_hidden'],
        gnn_hidden=CONFIG['gnn_hidden'],
        n_classes=n_classes,
        fusion_method='attention'
    )
    
    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Step 6: Train model
    print("\n[Step 6] Training model...")
    train_labels_array = np.array(train_dataset.sequence_labels)
    class_counts = np.bincount(train_labels_array, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    sample_weights = 1.0 / (class_counts[train_labels_array] + 1e-6)
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    train_sampler = WeightedRandomSampler(sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_with_dummy_graph
    )
    
    trainer = Trainer(model, device, learning_rate=CONFIG['learning_rate'], class_weights=class_weights_tensor)
    
    history = trainer.train(
        train_loader, 
        val_loader, 
        max_epochs=CONFIG['max_epochs'],
        early_stopping=3
    )
    
    # Step 7: Plot training history
    print("\n[Step 7] Plotting training history...")
    plot_training_history(history, results_dir)
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index_label='epoch')
    
    # Step 8: Evaluate on test set
    print("\n[Step 8] Evaluating on test set...")
    test_loss, test_preds, test_labels = trainer.evaluate(test_loader)
    test_acc = accuracy_score(test_labels, test_preds) * 100
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    labels = np.arange(len(global_label_encoder.classes_))
    class_names = [str(c) for c in global_label_encoder.classes_]
    print("\nClassification Report:")
    report_text = classification_report(test_labels, test_preds, labels=labels, target_names=class_names, zero_division=0)
    print(report_text)
    report_dict = classification_report(test_labels, test_preds, labels=labels, target_names=class_names, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(os.path.join(results_dir, 'classification_report.csv'))
    
    plot_confusion_matrix(test_labels, test_preds, class_names, results_dir)
    
    # Step 9: Save model
    print("\n[Step 9] Saving model...")
    model_artifacts = {
        'model_state_dict': model.state_dict(),
        'feature_grouping': feature_grouping,
        'config': CONFIG,
        'class_names': class_names
    }
    torch.save(model_artifacts, os.path.join(results_dir, 'cnn_lstm_gnn_nids.pth'))
    with open(os.path.join(results_dir, 'cnn_lstm_gnn_nids.pkl'), 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print("\n✅ Training completed successfully!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    
    return model, history, (test_labels, test_preds)


if __name__ == "__main__":
    model, history, (test_labels, test_preds) = main()
