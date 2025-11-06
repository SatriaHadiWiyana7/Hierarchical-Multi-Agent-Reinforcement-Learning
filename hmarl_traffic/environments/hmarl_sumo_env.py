import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import traci
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os
import sys
from hmarl_traffic import config

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Harap deklarasikan environment variable 'SUMO_HOME'")


class SumoTrafficEnv(MultiAgentEnv):
    """
    Lingkungan Multi-Agent SUMO untuk RLlib Tahap 1: Agent-Only.
    """
    
    def __init__(self, config_dict=None):
        """
        Inisialisasi lingkungan
        """
        super().__init__()
        env_config = config_dict or {}
        
        # --- Konfigurasi SUMO ---
        self.sumo_config_file = config.SUMO_CONFIG_PATH
        self.use_gui = env_config.get("use_gui", False)
        self.step_length_sec = env_config.get("step_length_sec", config.DEFAULT_STEP_SEC)
        self.port = env_config.get("port", config.RL_PORT) 
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        # --- Konfigurasi Agen ---
        self.agents = config.AGENT_IDS
        self._agent_ids = set(self.agents)
        self.green_phase_maps = config.GREEN_PHASE_MAPS
        
        # --- Definisi Space ---
        
        # Tentukan space untuk SATU agen
        single_agent_obs_space = Box(low=0, high=np.inf, shape=config.OBS_SHAPE, dtype=np.float32)
        single_agent_action_space = Discrete(config.ACTION_SIZE)
        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict(
            {agent_id: single_agent_obs_space for agent_id in self.agents}
        )
        
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict(
            {agent_id: single_agent_action_space for agent_id in self.agents}
        )
        
        self.traci_connected = False
        self.episode_steps = 0
        
        self.incoming_lanes = {}
        self.current_green_phases = {}

    def _start_sumo(self):
        """Mulai instance simulasi SUMO dan hubungkan Traci"""
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
        """Mengambil dan menyimpan data lajur masuk untuk setiap agen."""
        for agent_id in self.agents:
            incoming = set()
            for link in traci.trafficlight.getControlledLinks(agent_id):
                incoming.add(link[0][0]) 
            self.incoming_lanes[agent_id] = sorted(list(incoming))
            self.current_green_phases[agent_id] = traci.trafficlight.getPhase(agent_id)

    def reset(self, *, seed=None, options=None):
        """Reset lingkungan untuk memulai episode baru."""
        if self.traci_connected:
            traci.close()
            
        self._start_sumo()
        
        if not self.incoming_lanes:
            self._cache_lane_data()
            
        self.episode_steps = 0
        traci.simulationStep()
        obs = self._get_obs()
        
        return obs, {}

    def step(self, action_dict):
        """Terapkan aksi untuk setiap agen dan jalankan simulasi"""
        self.episode_steps += 1
        
        for agent_id, action in action_dict.items():
            if agent_id not in self.agents:
                continue
                
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
        truncated = self.episode_steps > 1000 
        
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}
        
        return obs, rewards, terminateds, truncateds, {}

    def _get_obs(self):
        """Mengumpulkan observasi untuk setiap agen"""
        obs_dict = {}
        for agent_id in self.agents:
            lane_queues = []
            if agent_id not in self.incoming_lanes:
                obs_dict[agent_id] = np.zeros(config.OBS_SHAPE, dtype=np.float32)
                continue
                
            for lane_id in self.incoming_lanes[agent_id]:
                lane_queues.append(traci.lane.getLastStepHaltingNumber(lane_id))
            
            padded_queues = lane_queues + [0] * (config.MAX_INCOMING_LANES - len(lane_queues))
            current_phase_index = self.current_green_phases.get(agent_id, 0)
            
            obs_vector = np.array(padded_queues + [current_phase_index], dtype=np.float32)
            obs_dict[agent_id] = obs_vector
            
        return obs_dict

    def _get_reward(self):
        """Menghitung ganjaran intrinsik untuk setiap agen"""
        reward_dict = {}
        for agent_id in self.agents:
            if agent_id not in self.incoming_lanes:
                reward_dict[agent_id] = 0.0
                continue
                
            total_queue = 0
            for lane_id in self.incoming_lanes[agent_id]:
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            
            reward_dict[agent_id] = -float(total_queue)
            
        return reward_dict

    def close(self):
        """Menutup koneksi Traci"""
        if self.traci_connected:
            traci.close()
            self.traci_connected = False