import pygame
from stable_baselines3 import PPO, DDPG, SAC
from enviroment import enviroment
import sys
from pathlib import Path

# Projektwurzel
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

# Konfiguration laden
config = load_config()
run_rl_cfg = config["run_rl"]


# Enviroment initialisieren
env = enviroment(run_rl_cfg["map_file"])

# Modell laden
match(run_rl_cfg["rl_algorithm"]):
     
    case "PPO":
          model = PPO.load(run_rl_cfg["model_load_path_PPO"])

    case "DDPG":
          model = DDPG.load(run_rl_cfg["model_load_path_DDPG"])

    case "SAC":
          model = SAC.load(run_rl_cfg["model_load_path_SAC"])


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

    # state, terminated, truncated pro step abfragen. (reward und info nicht noetig) 
    state, _, terminated, truncated, _ = env.step(action)

    print("state:", state)

    print("state shape:", state.shape)
    # Darstellung der aktuellen Umgebung im Pygame-Fenster
    env.render()

    # terminated = Episode natürlich beendet
    # truncated = Limit erreicht (unbenutzt)
    if terminated or truncated:
        state, _ = env.reset()
        print("state:", state)
        print("state shape:", state.shape)