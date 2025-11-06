import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import traci
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os
import sys
import json
import torch

# Import konfigurasi
from hmarl_traffic import config

# Tentukan path ke SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Harap deklarasikan environment variable 'SUMO_HOME'")

class HMARLSumoEnv(MultiAgentEnv):
    """
    Lingkungan HMARL SUMO (Tahap 2).
    
    Menyediakan observasi untuk:
    - Agen Pekerja (config.AGENT_IDS): Observasi lokal + Goal dari Lead
    - Agen Manajer (config.LEAD_AGENT_ID): Observasi global (grafik)
    """
    
    def __init__(self, config_dict=None):
        super().__init__()
        env_config = config_dict or {}
        
        self.sumo_config_file = config.SUMO_CONFIG_PATH
        self.use_gui = env_config.get("use_gui", False)
        self.step_length_sec = env_config.get("step_length_sec", config.DEFAULT_STEP_SEC)
        self.port = env_config.get("port", config.RL_PORT) 
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        self.worker_agents = config.AGENT_IDS
        self.lead_agent = config.LEAD_AGENT_ID
        self._agent_ids = set(self.worker_agents + [self.lead_agent])
        self.green_phase_maps = config.GREEN_PHASE_MAPS
        
        # --- Definisi Space ---
        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({
            **{agent_id: config.AGENT_OBS_SPACE for agent_id in self.worker_agents},
            self.lead_agent: config.LEAD_OBS_SPACE
        })
        
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict({
            **{agent_id: config.AGENT_ACTION_SPACE for agent_id in self.worker_agents},
            self.lead_agent: config.LEAD_ACTION_SPACE
        })
        
        self.traci_connected = False
        self.episode_steps = 0
        
        self.incoming_lanes = {}
        self.current_green_phases = {}
        self.global_edge_index = None
        self.agent_id_to_index = {agent_id: i for i, agent_id in enumerate(self.worker_agents)}
        self.agent_index_to_id = {i: agent_id for agent_id, i in self.agent_id_to_index.items()}
        self._load_graph_structure()
        
        # Inisialisasi 'goal' dari Lead (default: 0 = OTOMATIS)
        self.current_lead_goal = 0 
        self.current_lead_goal_onehot = np.zeros(config.LEAD_ACTION_SIZE, dtype=np.float32)
        self.current_lead_goal_onehot[0] = 1.0

    def _load_graph_structure(self):
        """Memuat edge_index dari graph_map.json"""
        try:
            with open(config.GRAPH_MAP_PATH, 'r') as f:
                graph_data = json.load(f)
            
            edges = graph_data["edges"]
            edge_list = []
            for u, v in edges:
                if u in self.agent_id_to_index and v in self.agent_id_to_index:
                    u_idx = self.agent_id_to_index[u]
                    v_idx = self.agent_id_to_index[v]
                    edge_list.append([u_idx, v_idx])
            
            edge_tensor = torch.tensor(edge_list, dtype=torch.int64).t().contiguous()
            
            num_edges = edge_tensor.shape[1]
            pad_size = config.MAX_EDGES_PAD - num_edges
            if pad_size < 0:
                print(f"Peringatan: Jumlah edge ({num_edges}) melebihi MAX_EDGES_PAD ({config.MAX_EDGES_PAD}).")
                edge_tensor = edge_tensor[:, :config.MAX_EDGES_PAD]
            else:
                padding = torch.zeros((2, pad_size), dtype=torch.int64)
                edge_tensor = torch.cat([edge_tensor, padding], dim=1)
            
            self.global_edge_index = edge_tensor.numpy() # Simpan sebagai numpy

        except Exception as e:
            print(f"Error memuat graph_map.json: {e}")
            self.global_edge_index = np.zeros(config.LEAD_OBS_SPACE["edge_index"].shape, dtype=np.int64)

    def _start_sumo(self):
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_config_file,
            "--no-warnings", "true",
            "--step-length", str(self.step_length_sec),
        ]
        try:
            traci.start(sumo_cmd, port=self.port, label=f"sim_{self.port}")
            self.traci_connected = True
        except traci.TraCIException as e:
            print(f"Error memulai SUMO di port {self.port}: {e}")
            sys.exit(1)

    def _cache_lane_data(self):
        for agent_id in self.worker_agents:
            incoming = set()
            for link in traci.trafficlight.getControlledLinks(agent_id):
                incoming.add(link[0][0]) 
            self.incoming_lanes[agent_id] = sorted(list(incoming))
            self.current_green_phases[agent_id] = traci.trafficlight.getPhase(agent_id)

    def reset(self, *, seed=None, options=None):
        if self.traci_connected:
            traci.close()
        self._start_sumo()
        if not self.incoming_lanes:
            self._cache_lane_data()
            
        self.episode_steps = 0
        # Reset goal
        self.current_lead_goal = 0
        self.current_lead_goal_onehot = np.zeros(config.LEAD_ACTION_SIZE, dtype=np.float32)
        self.current_lead_goal_onehot[0] = 1.0
        
        traci.simulationStep()
        obs = self._get_obs()
        return obs, {}

    def step(self, action_dict):
        self.episode_steps += 1
        
        # Ambil aksi 'Lead'
        if self.lead_agent in action_dict:
            self.current_lead_goal = action_dict[self.lead_agent]
            self.current_lead_goal_onehot = np.eye(config.LEAD_ACTION_SIZE)[self.current_lead_goal].astype(np.float32)

        # Terapkan Aksi Agent Pekerja
        for agent_id, action in action_dict.items():
            if agent_id in self.worker_agents:
                current_phase = traci.trafficlight.getPhase(agent_id)
                is_yellow_phase = (current_phase not in self.green_phase_maps[agent_id])
                
                if action == 1 and not is_yellow_phase:
                    next_green_phase = self.green_phase_maps[agent_id][current_phase]
                    traci.trafficlight.setPhase(agent_id, next_green_phase)
                    self.current_green_phases[agent_id] = next_green_phase
        
        traci.simulationStep()
        
        obs = self._get_obs()
        rewards = self._get_reward()
        
        terminated = traci.simulation.getMinExpectedNumber() == 0
        truncated = self.episode_steps > 1500
        
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}
        
        return obs, rewards, terminateds, truncateds, {}

    def _get_obs(self):
        obs_dict = {}
        # List untuk fitur node GNN
        all_node_features = np.zeros((config.MAX_AGENTS_PAD, config.NODE_FEATURE_DIM), dtype=np.float32)
        
        # --- Observasi Agen ---
        for agent_id in self.worker_agents:
            agent_idx = self.agent_id_to_index[agent_id]
            lane_queues = []
            
            if agent_id not in self.incoming_lanes:
                local_obs = np.zeros(config.OBS_SHAPE, dtype=np.float32)
                all_node_features[agent_idx, :] = [0.0, 0.0]
            else:
                total_queue = 0
                for lane_id in self.incoming_lanes[agent_id]:
                    queue = traci.lane.getLastStepHaltingNumber(lane_id)
                    lane_queues.append(queue)
                    total_queue += queue
                
                padded_queues = lane_queues + [0] * (config.MAX_INCOMING_LANES - len(lane_queues))
                current_phase_index = self.current_green_phases.get(agent_id, 0)
                
                local_obs = np.array(padded_queues + [current_phase_index], dtype=np.float32)
                
                # Kumpulkan fitur node untuk GNN [antrean_avg, fase]
                avg_queue = total_queue / len(self.incoming_lanes[agent_id])
                all_node_features[agent_idx, :] = [avg_queue, float(current_phase_index)]

            # Gabungkan observasi lokal dengan goal dari Lead
            obs_dict[agent_id] = np.concatenate([local_obs, self.current_lead_goal_onehot]).astype(np.float32)

        # --- Observasi Manajer ---
        node_mask = np.zeros(config.MAX_AGENTS_PAD, dtype=np.int64)
        node_mask[:len(self.worker_agents)] = 1
        
        obs_dict[self.lead_agent] = {
            "node_features": all_node_features,
            "edge_index": self.global_edge_index,
            "node_mask": node_mask
        }
            
        return obs_dict

    def _get_reward(self):
        reward_dict = {}
        total_global_queue = 0
        
        # --- Ganjaran Agen Pekerja ---
        for agent_id in self.worker_agents:
            # Ganjaran intrinsik: seberapa baik dia mematuhi 'goal'
            if agent_id not in self.incoming_lanes:
                reward_dict[agent_id] = 0.0
                continue
            
            total_queue = 0
            for lane_id in self.incoming_lanes[agent_id]:
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            
            reward_dict[agent_id] = -float(total_queue)
            total_global_queue += total_queue
            
        # --- Ganjaran Manajer (Ekstrinsik/Global) ---
        # Ganjaran global: metrik seluruh kota)
        reward_dict[self.lead_agent] = -float(total_global_queue)
            
        return reward_dict

    def close(self):
        if self.traci_connected:
            traci.close()
            self.traci_connected = False