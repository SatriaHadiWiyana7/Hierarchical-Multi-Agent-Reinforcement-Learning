import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import PublicAPI

try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data
    from torch_geometric.utils import to_dense_batch
except ImportError:
    print("Peringatan: torch_geometric tidak terinstal. Model GNN 'Lead' tidak akan berfungsi.")
    print("Silakan install: pip install torch_geometric")
    pyg_nn = None

from hmarl_traffic import config

class LeadGNNModel(TorchModelV2, nn.Module):
    """
    Model 'Lead' Manajer Global (GNN).
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if pyg_nn is None:
            raise ImportError("torch_geometric tidak ditemukan. Harap install.")
            
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        node_feature_dim = config.NODE_FEATURE_DIM
        hidden_dim = 64
        
        # Arsitektur GNN
        self.conv1 = pyg_nn.GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        
        # Jaringan untuk Aksi (Logits)
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs) # num_outputs = LEAD_ACTION_SIZE
        )
        
        # Jaringan untuk Value Function (PPO & A2C)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output tunggal untuk value
        )
        
        self._last_graph_embedding = None

    @PublicAPI
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass dari model GNN
        """
        obs = input_dict["obs"]
        
        # Ekstrak data graf dari observasi Dict
        x = obs["node_features"].float()          # Fitur Node [B, MaxNodes, F]
        edge_index = obs["edge_index"].long()     # Struktur Edge [B, 2, MaxEdges]
        node_mask = obs["node_mask"].bool()       # Mask [B, MaxNodes]
        
        batch_size = x.shape[0]
        all_graph_embeddings = []

        # Loop melalui setiap item dalam batch
        for i in range(batch_size):
            # Ambil data untuk item batch ke-i
            x_i = x[i][node_mask[i]] # Hanya ambil node yang valid
            
            # Filter edge_index
            edge_index_i = edge_index[i]
            max_node_idx = torch.sum(node_mask[i]) - 1
            edge_mask_i = (edge_index_i[0] <= max_node_idx) & (edge_index_i[1] <= max_node_idx)
            edge_index_i_filtered = edge_index_i[:, edge_mask_i]

            # GNN Convolutions
            x_conv1 = F.relu(self.conv1(x_i, edge_index_i_filtered))
            x_conv2 = F.relu(self.conv2(x_conv1, edge_index_i_filtered))
            
            # Global Pooling rata-rata semua fitur node yang valid
            if x_conv2.shape[0] == 0:
                 graph_embedding = torch.zeros(self.conv2.out_channels, device=x.device)
            else:
                graph_embedding = torch.mean(x_conv2, dim=0)
            
            all_graph_embeddings.append(graph_embedding)

        batch_graph_embedding = torch.stack(all_graph_embeddings)
        self._last_graph_embedding = batch_graph_embedding
        
        # Dapatkan logits aksi dari embedding graf
        logits = self.action_net(batch_graph_embedding)
        
        return logits, state

    @PublicAPI
    def value_function(self):
        """Mengembalikan nilai dari state graf saat ini"""
        assert self._last_graph_embedding is not None, "Panggil forward() dulu"
        value = self.value_net(self._last_graph_embedding)
        return torch.squeeze(value, -1)