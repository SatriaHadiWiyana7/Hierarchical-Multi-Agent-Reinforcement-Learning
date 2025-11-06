import os
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np

# --- Path Konfigurasi ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CONFIG_PATH = os.path.join(PROJECT_ROOT, "simulation/sumofiles/main.sumocfg")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GRAPH_MAP_PATH = os.path.join(DATA_DIR, "graph_map.json")
POI_DB_PATH = os.path.join(DATA_DIR, "poi_database.json")

# --- Konfigurasi Agen (Persimpangan) ---
AGENT_IDS = [
    "clusterJ0_J16_J18",
    "clusterJ11_J12_J4_J5",
    "clusterJ13_J14_J18_J19",
    "clusterJ13_J14_J2_J9",
    "clusterJ21_J43_J44",
    "clusterJ22_J24_J25"
]
NUM_AGENTS = len(AGENT_IDS)

# --- Konfigurasi Manajer (Lead) ---
LEAD_AGENT_ID = "lead_manager"
# Aksi Lead: Memberi perintah/goal
# 0: OTOMATIS, 1: PRIORITAS UTARA-SELATAN, 2: PRIORITAS BARAT-TIMUR
LEAD_ACTION_SIZE = 3 

# --- Konfigurasi Observasi & Aksi ---
MAX_INCOMING_LANES = 12 
# Obs Agen: Antrean per lajur + 1 fase
AGENT_OBS_SHAPE = (MAX_INCOMING_LANES + 1,)
AGENT_ACTION_SIZE = 2 # 0: EXTEND, 1: YIELD

# --- Definisi Spaces ---
# 1. Space untuk Agen 
AGENT_OBS_SHAPE_HMARL = (MAX_INCOMING_LANES + 1 + LEAD_ACTION_SIZE,)
AGENT_OBS_SPACE = Box(low=0, high=np.inf, shape=AGENT_OBS_SHAPE_HMARL, dtype=np.float32)
AGENT_ACTION_SPACE = Discrete(AGENT_ACTION_SIZE)

# 2. Space untuk Lead
MAX_AGENTS_PAD = 10
MAX_EDGES_PAD = 30
NODE_FEATURE_DIM = 2 # Fitur Node: [antrean_avg, fase_saat_ini]

LEAD_OBS_SPACE = Dict({
    "node_features": Box(low=-np.inf, high=np.inf, shape=(MAX_AGENTS_PAD, NODE_FEATURE_DIM), dtype=np.float32),
    "edge_index": Box(low=0, high=MAX_AGENTS_PAD, shape=(2, MAX_EDGES_PAD), dtype=np.int64),
    "node_mask": Box(low=0, high=1, shape=(MAX_AGENTS_PAD,), dtype=np.int64) # Mask untuk padding
})
LEAD_ACTION_SPACE = Discrete(LEAD_ACTION_SIZE)


# --- Konfigurasi Pelatihan ---
RL_PORT = 8813 
DEFAULT_STEP_SEC = 5 # Durasi per step simulasi