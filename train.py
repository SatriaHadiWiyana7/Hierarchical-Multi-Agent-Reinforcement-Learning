import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import os

from hmarl_traffic import config
from hmarl_traffic.environments.hmarl_sumo_env import SumoTrafficEnv
from hmarl_traffic.models.agent_model import AgentTrafficModel

def setup_training(use_gui=False):
    # Daftarkan Lingkungan Kustom
    register_env(
        "SumoTrafficEnv-v0", 
        lambda env_config: SumoTrafficEnv(env_config)
    )
    
    # Daftarkan Model Kustom
    ModelCatalog.register_custom_model("AgentTrafficModel", AgentTrafficModel)
    
    # Konfigurasi Algoritma
    
    # Dapatkan observasi dan ruang aksi dari env
    temp_env = SumoTrafficEnv({"use_gui": False, "port": 9999})
    
    # Ambil space dari agen, karena env.observation_space 
    agent_id_sample = config.AGENT_IDS[0]
    obs_space = temp_env.observation_space[agent_id_sample]
    action_space = temp_env.action_space[agent_id_sample]
    temp_env.close()

    # Kebijakan untuk Agent
    agent_policy = (
        None,  # Gunakan (PPO)
        obs_space,
        action_space,
        {
            "model": {
                "custom_model": "AgentTrafficModel",
                "custom_model_config": {
                    "hiddens": [64, 64] # Ukuran MLP
                },
            },
        },
    )

    # Konfigurasi PPO
    algo_config = (
        PPOConfig()
        .environment(
            "SumoTrafficEnv-v0",
            env_config={
                "use_gui": use_gui,
                "port": config.RL_PORT,
                "step_length_sec": config.DEFAULT_STEP_SEC
            }
        )
        .framework("torch")
        .env_runners(num_env_runners=1)
        .multi_agent(
            # Tahap 1: Latih 'Agent'
            policies={"agent_policy": agent_policy},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: "agent_policy")
        )
        .debugging(log_level="INFO")
    )
    
    return algo_config

def main():
    print("Memulai Inisialisasi Ray...")
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME belum diatur. Set environment variable.")
        return
        
    ray.init()
    
    # konfigurasi algoritma
    algo_config = setup_training(use_gui=False) 
    
    storage_path = os.path.abspath("./ray_results")
    print(f"Menyimpan hasil ke: {storage_path}")

    tuner = tune.Tuner(
        "PPO", # Algoritma
        param_space=algo_config.to_dict(),
        run_config=RunConfig( 
            name="PPO_Agent_Only_Training",
            stop={"training_iteration": 50}, 
            checkpoint_config=CheckpointConfig( 
                checkpoint_frequency=5, 
                checkpoint_at_end=True
            ),
            storage_path=storage_path
        ),
    )
    
    print("Memulai Pelatihan 'Agent' (Tahap 1)...")
    results = tuner.fit()
    
    print("Pelatihan selesai.")

    #API get_best_checkpoint()
    best_result = results.get_best_result()
    
    if best_result:
        best_checkpoint = best_result.get_best_checkpoint(
            metric="episode_reward_mean",
            mode="max"  # reward tertinggi
        )
        print(f"Hasil terbaik disimpan di: {best_checkpoint}")
    else:
        print("Pelatihan gagal atau tidak menghasilkan checkpoint")
    
    ray.shutdown()

if __name__ == "__main__":
    main()