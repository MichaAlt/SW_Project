import sys
from pathlib import Path

from stable_baselines3 import PPO
from rl_env import CarEnv


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config


config = load_config()
rl_cfg = config["rl"]

model_path = ROOT_DIR / rl_cfg["model_save_path"]
model_path.parent.mkdir(parents=True, exist_ok=True)

env = CarEnv(render_mode=None, random_map=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=rl_cfg["learning_rate"],
    n_steps=rl_cfg["n_steps"],
    batch_size=rl_cfg["batch_size"],
    gamma=rl_cfg["gamma"]
)

model.learn(total_timesteps=rl_cfg["total_timesteps"])

model.save(str(model_path))

print("RL model saved to:", model_path)