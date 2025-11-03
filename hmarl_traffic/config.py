import os

# --- Path Konfigurasi ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_CONFIG_PATH = os.path.join(PROJECT_ROOT, "simulation/sumofiles/main.sumocfg")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GRAPH_MAP_PATH = os.path.join(DATA_DIR, "graph_map.json")
POI_DB_PATH = os.path.join(DATA_DIR, "poi_database.json")

# --- Konfigurasi Agen ---
AGENT_IDS = [
    "clusterJ0_J16_J18",
    "clusterJ11_J12_J4_J5",
    "clusterJ13_J14_J18_J19",
    "clusterJ13_J14_J2_J9",
    "clusterJ21_J43_J44",
    "clusterJ22_J24_J25"
]

# Pemetaan Fase Hijau untuk setiap cluster
# aksi YIELD agar mematuhi siklus 
GREEN_PHASE_MAPS = {
    "clusterJ0_J16_J18": {0: 3, 3: 6, 6: 0},
    "clusterJ11_J12_J4_J5": {0: 3, 3: 6, 6: 9, 9: 0},
    "clusterJ13_J14_J18_J19": {0: 3, 3: 6, 6: 0},
    "clusterJ13_J14_J2_J9": {0: 3, 3: 6, 6: 9, 9: 0},
    "clusterJ21_J43_J44": {0: 3, 3: 6, 6: 0},
    "clusterJ22_J24_J25": {0: 3, 3: 6, 6: 0}
}

# --- Konfigurasi Observasi & Aksi ---
MAX_INCOMING_LANES = 12 # Dihitung dari persimpangan terpadat
OBS_SHAPE = (MAX_INCOMING_LANES + 1,) # Antrean per lajur + 1 untuk indeks fase saat ini 
ACTION_SIZE = 2 # 0: EXTEND, 1: YIELD 

# --- Konfigurasi Pelatihan ---
RL_PORT = 8813 # Port Traci
DEFAULT_STEP_SEC = 1 # Durasi per step simulasi