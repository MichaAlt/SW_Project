from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from enviroment import enviroment
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

from Config.config_loader import load_config

config = load_config()
train_rl_cfg = config["train_rl"]

env = make_vec_env( lambda: enviroment(train_rl_cfg["map_file"], render_mode=False), n_envs=train_rl_cfg["n_envs"])

match(train_rl_cfg["rl_algorithm"]):

    case "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1
        )
    case "DDPG":
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1
        )
    case "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1
        )

# Altes Modell zum weiter trainieren laden
if train_rl_cfg["continue_model_training"] == "true":
    model = PPO.load(train_rl_cfg["model_load_path"], env = env) 

model.learn(total_timesteps=train_rl_cfg["total_timesteps"])

model.save(train_rl_cfg["model_save_path"])
print("Training fertig!")