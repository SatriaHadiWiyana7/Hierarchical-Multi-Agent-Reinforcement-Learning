import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
import os
import sys

from hmarl_traffic import config
from hmarl_traffic.environments.hmarl_sumo_env import HMARLSumoEnv
from hmarl_traffic.models.agent_model import AgentTrafficModel
from hmarl_traffic.models.lead_model_gnn import LeadGNNModel

PHASE_1_CHECKPOINT_PATH = "PATH_KE_CHECKPOINT_TAHAP_1_ANDA"  # Lokasi path checkpoint Tahap 1

def setup_phase2_training(use_gui=False):
    # Daftarkan Lingkungan HMARL
    register_env(
        "HMARLSumoEnv-v0", 
        lambda env_config: HMARLSumoEnv(env_config)
    )
    
    # Daftarkan Model
    ModelCatalog.register_custom_model("AgentTrafficModel", AgentTrafficModel)
    ModelCatalog.register_custom_model("LeadGNNModel", LeadGNNModel)
    
    # Konfigurasi Kebijakan 

    # Kebijakan untuk Agent
    agent_policy = PolicySpec(
        policy_class=None,
        observation_space=config.AGENT_OBS_SPACE,
        action_space=config.AGENT_ACTION_SPACE,
        config={
            "model": {
                "custom_model": "AgentTrafficModel",
                # Sesuaikan hiddens jika Anda mengubahnya di Tahap 1
                "custom_model_config": {"hiddens": [64, 64]}, 
            },
        },
    )

    # Kebijakan untuk Manajer Global GNN
    lead_policy = PolicySpec(
        policy_class=None,
        observation_space=config.LEAD_OBS_SPACE,
        action_space=config.LEAD_ACTION_SPACE,
        config={
            "model": {
                "custom_model": "LeadGNNModel",
            },
        },
    )

    # Fungsi pemetaan: Agent_ID -> Policy_ID
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == config.LEAD_AGENT_ID:
            return "lead_policy"
        else:
            return "agent_policy" # Semua agen pekerja berbagi satu kebijakan

    # Konfigurasi PPO
    algo_config = (
        PPOConfig()
        .environment(
            "HMARLSumoEnv-v0",
            env_config={
                "use_gui": use_gui,
                "port": config.RL_PORT,
                "step_length_sec": config.DEFAULT_STEP_SEC
            }
        )
        .framework("torch")
        .env_runners(num_env_runners=1) 
        .multi_agent(
            policies={
                "agent_policy": agent_policy,
                "lead_policy": lead_policy
            },
            policy_mapping_fn=policy_mapping_fn,
            # Kita HANYA melatih 'lead_policy'
            policies_to_train=["lead_policy"]
        )
        .debugging(log_level="INFO")
    )
    
    return algo_config

def main():
    if PHASE_1_CHECKPOINT_PATH == "PATH_KE_CHECKPOINT_TAHAP_1_ANDA":
        print("="*80)
        print("ERROR: Harap atur 'PHASE_1_CHECKPOINT_PATH' di dalam file train.py")
        print("="*80)
        sys.exit(1)

    print("Memulai Inisialisasi Ray...")
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME belum diatur.")
        return
        
    ray.init()
    
    algo_config = setup_phase2_training(use_gui=False) 
    
    storage_path = os.path.abspath("./ray_results_phase2") # Simpan di folder baru
    print(f"Menyimpan hasil ke: {storage_path}")

    print(f"\nMemulai Pelatihan 'Lead' (Tahap 2)...")
    print(f"Memuat bobot 'agent_policy' dari: {PHASE_1_CHECKPOINT_PATH}")
    print("Membekukan 'agent_policy' dan hanya melatih 'lead_policy'.\n")

    tuner = tune.Tuner.restore(
        path=PHASE_1_CHECKPOINT_PATH,
        trainable="PPO",
        param_space=algo_config.to_dict(),
        resume_config=RunConfig( 
            name="PPO_Lead_Training_Phase2",
            stop={"training_iteration": 100},
            checkpoint_config=CheckpointConfig( 
                checkpoint_frequency=10, 
                checkpoint_at_end=True
            ),
            storage_path=storage_path
        )
    )
    
    results = tuner.fit()
    
    print("Pelatihan Tahap 2 selesai")
    
    best_result = results.get_best_result()
    if best_result:
        best_checkpoint = best_result.get_best_checkpoint(
            metric="episode_reward_mean",
            mode="max"
        )
        print(f"Hasil terbaik disimpan di: {best_checkpoint}")
    else:
        print("Pelatihan gagal atau tidak menghasilkan checkpoint")
    
    ray.shutdown()

if __name__ == "__main__":
    main()