import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from gymnasium.spaces import Box

from hmarl_traffic import config

class AgentTrafficModel(TorchModelV2, nn.Module):
    """
    Model 'Agent' Pekerja Lokal.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Spesifikasi model kustom
        hiddens = model_config.get("custom_model_config", {}).get("hiddens", [64, 64])
        
        # Observasi kita adalah Box flat antrean & fase 
        obs_size = int(config.OBS_SHAPE[0]) 
        
        # Buat jaringan MLP
        self.mlp = TorchFC(
            Box(low=-1.0, high=1.0, shape=(obs_size,)), # Input dummy
            action_space,
            num_outputs,
            model_config,
            name + "_mlp"
        )
        
        # Simpan input observasi terakhir
        self._last_obs = None

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass dari model.
        """
        obs = input_dict["obs_flat"].float()
        self._last_obs = obs
        
        # Kirim observasi ke MLP
        # Output [0] adalah logits, [1] adalah state
        logits, _ = self.mlp({"obs": obs}) 
        
        return logits, state

    def value_function(self):
        """
        Mengembalikan nilai (value) dari state saat ini.
        """
        assert self._last_obs is not None, "Panggil forward() dulu"
        # Kirim obs terakhir ke value branch dari MLP
        value, _ = self.mlp({"obs": self._last_obs})
        return torch.squeeze(value, -1)