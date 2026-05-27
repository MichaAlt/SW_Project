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

env = CarEnv(
    render_mode="human",
    random_map=False,
    fixed_map=rl_cfg["map_file_run"]
)
model = PPO.load(str(model_path))

obs, info = env.reset()

running = True
while running:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()