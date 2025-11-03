import gymnasium as gym
import numpy as np
import traci
import os
import sys
from hmarl_traffic import config
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("environment variable 'SUMO_HOME'")


class SumoTrafficEnv(MultiAgentEnv):
    
    def __init__(self, config_dict=None):
        super().__init__()
        env_config = config_dict or {}
        
        # --- Konfigurasi SUMO ---
        self.sumo_config_file = config.SUMO_CONFIG_PATH
        self.use_gui = env_config.get("use_gui", False)
        self.step_length_sec = env_config.get("step_length_sec", config.DEFAULT_STEP_SEC)
        # Port untuk parallel rollout workers
        self.port = env_config.get("port", config.RL_PORT) 
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        # --- Konfigurasi Agen ---
        self.agents = config.AGENT_IDS
        self._agent_ids = set(self.agents)
        self.green_phase_maps = config.GREEN_PHASE_MAPS
        
        # Aksi: 0 = EXTEND, 1 = YIELD
        self.action_space = Discrete(config.ACTION_SIZE) 
        # Observasi: Antrean per lajur + Indeks Fase Saat Ini
        self.observation_space = Box(low=0, high=np.inf, shape=config.OBS_SHAPE, dtype=np.float32)
        
        self.traci_connected = False
        self.episode_steps = 0
        
        # Cache untuk data yang sering diakses
        self.incoming_lanes = {}
        self.current_green_phases = {} # Melacak fase hijau

    def _start_sumo(self):
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_config_file,
            "--no-warnings", "true",
            "--step-length", str(self.step_length_sec),
            "--remote-port", str(self.port)
        ]
        
        try:
            traci.start(sumo_cmd, port=self.port, label=f"sim_{self.port}")
            self.traci_connected = True
        except traci.TraCIException as e:
            print(f"Error memulai SUMO di port {self.port}: {e}")
            sys.exit(1)

    def _cache_lane_data(self):
        for agent_id in self.agents:
            controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
            incoming = set()
            # Loop melalui semua koneksi yang dikontrol TL
            for link in traci.trafficlight.getControlledLinks(agent_id):
                incoming.add(link[0][0]) 
            
            # Simpan daftar lajur masuk yang diurutkan
            self.incoming_lanes[agent_id] = sorted(list(incoming)) 
            # Set fase hijau awal
            self.current_green_phases[agent_id] = traci.trafficlight.getPhase(agent_id)

    def reset(self, *, seed=None, options=None):
        if self.traci_connected:
            traci.close()
            
        self._start_sumo() 
        # Cache data lajur
        if not self.incoming_lanes:
            self._cache_lane_data()
            
        self.episode_steps = 0
        
        # Jalankan 1 step agar simulasi dimulai dan data tersedia
        traci.simulationStep()
        
        # Dapatkan observasi awal
        obs = self._get_obs()
        
        # Mengembalikan obs_dict dan info_dict kosong
        return obs, {}

    def step(self, action_dict):
        self.episode_steps += 1
        
        # aksi untuk setiap agen
        for agent_id, action in action_dict.items():
            if agent_id not in self.agents:
                continue
                
            current_phase = traci.trafficlight.getPhase(agent_id)
            
            # Cek apakah saat ini sedang dalam fase kuning/merah transisi
            is_yellow_phase = (current_phase not in self.green_phase_maps[agent_id])
            
            # Aksi YIELD (1) hanya dieksekusi jika sedang dalam fase hijau
            if action == 1 and not is_yellow_phase:
                next_green_phase = self.green_phase_maps[agent_id][current_phase]
                traci.trafficlight.setPhase(agent_id, next_green_phase)
                self.current_green_phases[agent_id] = next_green_phase
        
        # Jalankan simulasi SUMO selama
        traci.simulationStep()
        
        # Kumpulkan status baru setelah step
        obs = self._get_obs()
        rewards = self._get_reward()
        
        # selesai jika mobil = 0 
        terminated = traci.simulation.getMinExpectedNumber() == 0
        
        # hentikan jika melebihi 1500 step sebagai contoh
        truncated = self.episode_steps > 1500 # Contoh batas 1500 step
        
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}
        
        # Info dict
        infos = {}
        
        return obs, rewards, terminateds, truncateds, infos

    def _get_obs(self):
        obs_dict = {}
        for agent_id in self.agents:
            # Dapatkan antrean per lajur
            lane_queues = []
            if agent_id not in self.incoming_lanes:
                 # Jika agen belum siap kirim obs kosong
                obs_dict[agent_id] = np.zeros(config.OBS_SHAPE, dtype=np.float32)
                continue
                
            for lane_id in self.incoming_lanes[agent_id]:
                # 'getLastStepHaltingNumber' -> jumlah mobil yang berhenti (antrean)
                lane_queues.append(traci.lane.getLastStepHaltingNumber(lane_id))
            
            # Padding agar semua vektor observasi punya panjang sama
            padded_queues = lane_queues + [0] * (config.MAX_INCOMING_LANES - len(lane_queues))
            
            # Dapatkan indeks fase hijau saat ini
            current_phase_index = self.current_green_phases.get(agent_id, 0)
            
            # Gabungkan observasi
            obs_vector = np.array(padded_queues + [current_phase_index], dtype=np.float32)
            obs_dict[agent_id] = obs_vector
            
        return obs_dict

    def _get_reward(self):
        reward_dict = {}
        for agent_id in self.agents:
            if agent_id not in self.incoming_lanes:
                reward_dict[agent_id] = 0.0
                continue
                
            # Reward = - total antrean di semua lajur masuk
            total_queue = 0
            for lane_id in self.incoming_lanes[agent_id]:
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            
            # Reward untuk agen agar meminimalkan antrean
            reward_dict[agent_id] = -float(total_queue)
            
        return reward_dict

    def close(self):
        if self.traci_connected:
            traci.close()
            self.traci_connected = False