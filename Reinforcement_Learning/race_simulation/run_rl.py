import pygame
from stable_baselines3 import PPO, DDPG, SAC
from enviroment import enviroment
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
run_rl_cfg = config["run_rl"]


# Enviroment initialisieren
env = enviroment(run_rl_cfg["map_file"], render_mode=True)

# Modell laden
model = PPO.load(run_rl_cfg["model_load_path"])

# state initialisieren
state, _ = env.reset()

running = True

# Game-Loop
while running:

    for event in pygame.event.get():

        # Fenster schließen
        if event.type == pygame.QUIT:
            running = False
        
        # Game-Loop mit Taste esc oder q abbrechen
        elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

    # action_prediction
    action, _ = model.predict(state, deterministic=True)


    state, _, terminated, truncated, _ = env.step(action)

    env.render()

    # terminated = Episode natürlich zuende gegangen
    # truncated = Limit erreicht
    if terminated or truncated:
        state, _ = env.reset()