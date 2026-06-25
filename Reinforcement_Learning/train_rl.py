from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from enviroment import enviroment
import sys
from pathlib import Path

# Projektwurzel
ROOT_DIR = Path(__file__).resolve().parent.parent

from Config.config_loader import load_config

# Konfiguration laden
config = load_config()
train_rl_cfg = config["train_rl"]

# Trainingsumgebung erstellen. make_vec_env erzeugt mehrere parallel laufende Umgebungen für schnelleres Training.
env = make_vec_env( lambda: enviroment(train_rl_cfg["map_file"]), n_envs=train_rl_cfg["n_envs"])

# Auswahl des RL-Algorithmus ueber die Konfiguration 
match(train_rl_cfg["rl_algorithm"]):

    # Proximal Policy Optimization
    case "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1
        )
    # Deep Deterministic Policy Gradient
    case "DDPG":
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1
        )
    # Soft Actor-Critic
    case "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1
        )

# Altes Modell zum weiter trainieren laden
if train_rl_cfg["continue_model_training"] == True:
    match(train_rl_cfg["rl_algorithm"]):
        case "PPO":
            model = PPO.load(train_rl_cfg["model_load_path_PPO"], env = env)
        case "DDPG":
            model = DDPG.load(train_rl_cfg["model_load_path_DDPG"], env = env)
        case "SAC":
            model = SAC.load(train_rl_cfg["model_load_path_SAC"], env = env)

# Training bzw. Lernprozess starten
model.learn(total_timesteps=train_rl_cfg["total_timesteps"])

# Modell speichern
match(train_rl_cfg["rl_algorithm"]):
    case "PPO":
        model.save(train_rl_cfg["model_save_path_PPO"])
    case "DDPG":
        model.save(train_rl_cfg["model_save_path_DDPG"])
    case "SAC":
        model.save(train_rl_cfg["model_save_path_SAC"])
print("Training fertig!")