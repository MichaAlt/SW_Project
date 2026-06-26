import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys, math
from pathlib import Path

# Projektwurzel
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from car import Car
from map_loader import load_map
from Config.config_loader import load_config

# Konfigurationen laden
config = load_config()
enviroment_cfg = config["enviroment"]

class enviroment(gym.Env):

    radius = 300
    forward_progress = 0

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
        car_png_path = '../../Car_Img/car.png'
        self.car = Car(car_png_path)

        # Pygame-Module initialisieren
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
            
        # State definieren
        self.observation_space = spaces.Box(

            low=0,

            high=255,

            shape=(84,84,1),

            dtype=np.uint8

        )

        # Action definieren
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64
        )

    # State abfragen
    def get_state(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.display_map, (self.offset_x, self.offset_y))
       # self.car.draw(self.screen, self.font, self.scale, self.offset_x, self.offset_y)
        return self.get_semicircle_img()
    
    # Umgebung für eine neue Episode zuruecksetzen
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.car.reset()
        self.car.update(self.game_map)
        return self.get_state(), {}

    def calculate_reward(self, action, prev_pos):
        steer    = float(action[0])
        reward   = 0.0

        # 1. Crash — stärker bestrafen damit das CNN es früh lernt
        if not self.car.alive:
            return -150.0

        # 2. Geschwindigkeit stärker belohnen (war 0.05, jetzt 0.1)
        reward += self.car.speed * 0.1

        # 3. Stillstand klarer bestrafen (war -0.2, jetzt -1.0)
        if self.car.speed < 0.2:
            reward -= 1.0

        # 4. Lenkstrafe weicher machen — sonst lernt das Auto NIE zu lenken
        #    (war 0.05 + 0.2 bei >0.8, jetzt nur 0.02 ohne harte Grenze)
        reward -= abs(steer) * 0.02

        # 5. NEU: Fortschrittsbonus — echte Bewegung durch die Map belohnen
        #    Das ist das wichtigste Signal für CNN-basiertes RL!
        curr_pos = np.array(self.car.center, dtype=np.float32)
        delta    = np.linalg.norm(curr_pos - prev_pos)
        reward  += delta * 0.1

        # 6. NEU: Wandabstand belohnen (wie in enviroment.py, aber mit CNN-State)
        #    Radar-Werte musst du separat berechnen oder aus car.radar_values lesen
        if hasattr(self.car, 'radar_values') and self.car.radar_values:
            reward += sum(self.car.radar_values) * 0.003

        return reward
 
    # Ausfuehrung eines Zeitschrittes im Enviroment
    def step(self, action):
        self.step_count += 1
        truncated  = False
        terminated = False

        steer    = float(action[0])
        throttle = float(action[1])

        # Vorherige Position speichern (für Fortschrittsbonus)
        prev_pos = np.array(self.car.center, dtype=np.float32)

        self.car.speed += throttle * 0.05
        self.car.speed  = np.clip(self.car.speed, 0.0, enviroment_cfg["speed_scaling"])

        if self.car.speed > 0.2:
            self.car.angle += steer * 1.5

        self.car.update(self.game_map)

        obs    = self.get_state()
        reward = self.calculate_reward(action, prev_pos)

        # NEU: Zeitlimit aktivieren (war auskommentiert!)
        if self.step_count >= self.max_steps:
            truncated = True

        if not self.car.alive:
            terminated = True

        if self.step_count % 5 == 0:
            self.render()
            # Debug-Bild (wie gehabt)
            ...

        return obs, reward, terminated, truncated, {}
    def draw_filled_semicircle(self, screen, center, angle, color):

        points = [center]

        start_angle = angle - math.pi / 2

        end_angle = angle + math.pi / 2

        steps = 20

        for i in range(steps + 1):

            t = start_angle + (end_angle - start_angle) * i / steps

            x = center[0] + math.cos(t) * self.radius

            y = center[1] - math.sin(t) * self.radius

            points.append((x, y))

        pygame.draw.polygon(screen, color, points)
    
    def get_semicircle_img(self):
        angle = math.radians(self.car.angle)

        fx = math.cos(angle)

        fy = -math.sin(angle)

        rx = math.cos(angle + math.pi / 2)

        ry = -math.sin(angle + math.pi / 2)
        center = (
            int(self.car.center[0] * self.scale + self.offset_x),
            int(self.car.center[1] * self.scale + self.offset_y)
        )

        rect_center = (

            center[0] + fx * (self.radius / 2),

            center[1] + fy * (self.radius  / 2)

        )

        hw = self.radius  / 2

        hh = self.radius  / 2

        points = [

            (rect_center[0] - rx * hw - fx * hh, rect_center[1] - ry * hw - fy * hh),

            (rect_center[0] + rx * hw - fx * hh, rect_center[1] + ry * hw - fy * hh),

            (rect_center[0] + rx * hw + fx * hh, rect_center[1] + ry * hw + fy * hh),

            (rect_center[0] - rx * hw + fx * hh, rect_center[1] - ry * hw + fy * hh),

        ]

        rect = pygame.Rect(

            int(min(p[0] for p in points)),

            int(min(p[1] for p in points)),

            int(max(p[0] for p in points) - min(p[0] for p in points)),

            int(max(p[1] for p in points) - min(p[1] for p in points))

        )      


        crop_w = self.radius 
        crop_h = self.radius 


        """
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),   # Rot
            rect,
            2              # Linienbreite
        )
        """
        
        crop = pygame.Surface((crop_w, crop_h))
        crop.fill((255, 255, 255))

        screen_rect = self.screen.get_rect()
        clipped_rect = rect.clip(screen_rect)

        if clipped_rect.width > 0 and clipped_rect.height > 0:
            part = self.screen.subsurface(clipped_rect).copy()

            dest = (
                clipped_rect.x - rect.x,
                clipped_rect.y - rect.y
            )

            crop.blit(part, dest)

        rotated = pygame.transform.rotate(crop, self.car.angle)

        # Create a white background instead of black

        crop = pygame.Surface(rotated.get_size())

        crop.fill((255, 255, 255))

        crop.blit(rotated, (0, 0))
        crop = pygame.transform.scale(crop, (84, 84))

        img = pygame.surfarray.array3d(crop)

        img = np.transpose(img, (1, 0, 2))
        mask = np.all(img > 240, axis=2)

        img = np.zeros((84, 84), dtype=np.uint8)
        img[mask] = 255
        img = img[:, :, None]  # macht aus (84,84) -> (84,84,1)

        return img

    # Darstellung der aktuellen Umgebung im Pygame-Fenster
    # Reihenfolge:
    # - Hintergrund
    # - Map
    # - Auto
    def render(self):
        
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.display_map, (self.offset_x, self.offset_y))
        center = (
            int(self.car.center[0] * self.scale + self.offset_x),
            int(self.car.center[1] * self.scale + self.offset_y)
        )
        crop_w = self.radius *2
        crop_h = self.radius *2
            # Draw a red circle
        #  self.draw_filled_semicircle(self.screen, center, math.radians(self.car.angle), (255,0,0) )
        rect = pygame.Rect(
            center[0] - self.radius,
            center[1] - self.radius,
            crop_w,
            crop_h
        )

        angle = math.radians(self.car.angle)

        fx = math.cos(angle)

        fy = -math.sin(angle)

        rx = math.cos(angle + math.pi / 2)

        ry = -math.sin(angle + math.pi / 2)

        rect_center = (

            center[0] + fx * (self.radius / 2),

            center[1] + fy * (self.radius  / 2)

        )

        hw = self.radius  / 2

        hh = self.radius  / 2

        points = [

            (rect_center[0] - rx * hw - fx * hh, rect_center[1] - ry * hw - fy * hh),

            (rect_center[0] + rx * hw - fx * hh, rect_center[1] + ry * hw - fy * hh),

            (rect_center[0] + rx * hw + fx * hh, rect_center[1] + ry * hw + fy * hh),

            (rect_center[0] - rx * hw + fx * hh, rect_center[1] - ry * hw + fy * hh),

        ]

        pygame.draw.polygon(self.screen, (255, 0, 0), points, 2)

        self.car.draw(self.screen, self.font, self.scale, self.offset_x, self.offset_y)
  
        pygame.display.flip()
        self.clock.tick(60)