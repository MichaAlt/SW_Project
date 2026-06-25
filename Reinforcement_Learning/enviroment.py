import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from pathlib import Path

# Projektwurzel
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from car import Car
from map_loader import load_map
from Config.config_loader import load_config

# Konfigurationen laden
config = load_config()
enviroment_cfg = config["enviroment"]

class enviroment(gym.Env):
    def __init__(self, map_path, screen_w=1920, screen_h=1080):
        
        # Gym-Enviroment initialisieren
        super().__init__()

        self.screen = None

        # Maximale Anzahl von Steps pro Episode (unbenutzt)
        self.max_steps = 50000

        # Step-count (unbenutzt)
        self.step_count = 0

        # Karte laden und alle für die Darstellung benoetigten Werte zurueckgeben
        self.screen, self.game_map, self.display_map, self.scale, self.offset_x, self.offset_y = load_map(
            map_path, screen_w, screen_h
        )

        # Car initialisieren
        car_png_path = '../PNG_File/car.png'
        self.car = Car(car_png_path)

        # Pygame-Module initialisieren
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
            
        # State definieren
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([enviroment_cfg["speed_scaling"], 300, 300, 300, 300, 300]),
            shape=(6,),
            dtype=np.float64
        )

        # Action definieren
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64
        )

    # State abfragen
    def get_state(self):
        radar_values = list(self.car.radar_values)

        while len(radar_values) < 5:
            radar_values.append(0.0)

        radar_values = radar_values[:5]

        return np.array([self.car.speed] + radar_values, dtype=np.float32)
    # Umgebung für eine neue Episode zuruecksetzen
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car.reset()
        self.car.update(self.game_map)
        return self.get_state(), {}

    # Ausfuehrung eines Zeitschrittes im Enviroment
    def step(self, action):
        
        # Step counter (unbenutzt)
        self.step_count += 1

        # terminated = Episode natuerlich beendet
        # truncated = Limit erreicht (unbenutzt)
        truncated = False
        terminated = False

        steer = float(action[0])
        speed = float(action[1])

        # Skalierung
        self.car.angle += steer * enviroment_cfg["steer_scaling"] 
        self.car.speed = speed * enviroment_cfg["speed_scaling"] 

        self.car.update(self.game_map)

        obs = self.get_state()

        # Belohnungs-Wert
        reward = 0.0

        # Zusammenstoß mit der Wand bestrafen
        if not self.car.alive:
            reward -= 100

        # Hohe Geschwindigkeit belohnen
        reward += self.car.speed * 0.05

        # Niedrige Geschwindigkeit bestrafen
        if self.car.speed < 0.2:
            reward -= 1
        
        # Starkes lenken bestrafen
        reward -= abs(steer) * 0.01

        # Distanz zur Wand belohnen
        reward += sum(self.car.radar_values) * 0.002

        # Rückwärtsfahren bestrafen
        if self.car.speed < 0:
            reward -= 2
        """
        if self.step_count >= self.max_steps:
            truncated = True
        """
        if not self.car.alive:
            terminated = True
        

        return obs, reward, terminated, truncated, {}

    # Darstellung der aktuellen Umgebung im Pygame-Fenster
    def render(self):
        
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.display_map, (self.offset_x, self.offset_y))
        self.car.draw(self.screen, self.font, self.scale, self.offset_x, self.offset_y)

        pygame.display.flip()
        self.clock.tick(60)