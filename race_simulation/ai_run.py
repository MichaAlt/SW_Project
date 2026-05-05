import pygame
import sys
import os
import tensorflow as tf
import numpy as np
from pathlib import Path

from car import Car
from map_loader import load_map

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
ai_cfg = config["ai_run"]
feature_scaling_cfg = config["feature_scaling"]
prediction_cfg = config["prediction"]


def get_screen_size(cfg):
    pygame.init()
    info = pygame.display.Info()

    if cfg["width"] == "auto" or cfg["height"] == "auto":
        return info.current_w, info.current_h

    return cfg["width"], cfg["height"]

WIDTH, HEIGHT = get_screen_size(ai_cfg)

def main():
    os.environ["SDL_RENDER_DRIVER"] = "opengl"
    pygame.init()

    screen, game_map, display_map, scale, offset_x, offset_y = load_map(
        ai_cfg["map_file"], WIDTH, HEIGHT
    )
    pygame.display.set_caption("Car mit Sensoren")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 28)

    car = Car()

    # KI-Modell laden
    if prediction_cfg["prediction"] == "classification":
        model = tf.keras.models.load_model(ai_cfg["model_path_classification"])
    elif prediction_cfg["prediction"] == "regression":
        model = tf.keras.models.load_model(ai_cfg["model_path_regression"])

    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        car.update(game_map)
        car.draw(screen, font_small, scale, offset_x, offset_y)

        # KI-Modell
        if len(car.radar_values) == 5:
            x_input = np.array(car.radar_values, dtype=np.float32).reshape(1, -1) 

            if feature_scaling_cfg["method"] == 1: # Normalisieren
                x_input = x_input/ 298.0
            elif feature_scaling_cfg["method"] == 2: # Standardisieren
                x_input = (x_input - x_input.mean()) / x_input.std() 


            pred = model(x_input, training=False)
            if prediction_cfg["prediction"] == "classification":
                action_index = np.argmax(pred[0])
                mapping = {0: "W", 1: "A", 2: "D", 3: "W+A", 4: "W+D"}
                actions = mapping[action_index].split("+")

            elif prediction_cfg["prediction"] == "regression":
                speed = float(pred[0][0]) # Tensor verursacht bei Regression-Vorhersage Perfomanceprobleme (FPS brechen ein), deshalb casting auf float
                steer = float(pred[0][1])


            
        else:  # Beim Crash ist InputArray leer, dadurch Programmabsturz, fuer die Zeit kein predict
            if prediction_cfg["prediction"] == "classification":
                actions = ["W"]
            elif prediction_cfg["prediction"] == "regression":
                speed = 0
                steer = 0

        # Anzeige der Sensorwerte
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        text_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(text_surface, (50, 50))

        # Steuerung des Autos
        car.speed = 0
        if prediction_cfg["prediction"] == "classification":
            if "W" in actions:
                car.speed = ai_cfg["forward_speed"]
            if "S" in actions:
                car.speed = ai_cfg["backward_speed"]
            if "A" in actions:
                car.angle += ai_cfg["turn_angle"]
            if "D" in actions:
                car.angle -= ai_cfg["turn_angle"]
        
        elif prediction_cfg["prediction"] == "regression":
            car.speed = speed
            car.angle += steer


        pygame.display.flip()
        clock.tick(60)

        # Ereignisbehandlung
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
