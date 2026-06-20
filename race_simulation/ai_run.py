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


def get_screen_size(cfg):
    pygame.init()
    info = pygame.display.Info()

    if cfg["width"] == "auto" or cfg["height"] == "auto":
        return info.current_w, info.current_h

    return cfg["width"], cfg["height"]


WIDTH, HEIGHT = get_screen_size(ai_cfg)


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_input(radar_values):
    x_input = np.array(
        radar_values,
        dtype=np.float32
    ).reshape(1, -1)

    # 5 Sensorwerte normalisieren
    x_input[:, 0:5] = x_input[:, 0:5] / 298.0

    return x_input


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
    model = tf.keras.models.load_model(ai_cfg["model_path"])

    prev_turn_angle = 0.0

    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        if car.alive and len(car.radar_values) == 5:
            x_input = normalize_input(car.radar_values)

            pred = model(x_input, training=False)

            # Modell mit zwei Ausgängen:
            # pred["turn"]  -> [[turn_norm]]
            # pred["speed"] -> [[speed_norm]]
            raw_turn_norm = float(pred["turn"][0][0])
            raw_speed_norm = float(pred["speed"][0][0])

            # turn_norm im Bereich [-1, 1]
            turn_norm = clamp(raw_turn_norm, -1.0, 1.0)

            # speed_norm im Bereich [0, 1]
            speed_norm = clamp(raw_speed_norm, 0.0, 1.0)

            # In einen Winkel zurückrechnen
            pred_turn_angle = turn_norm * 180.0

            # Leichte Glättung
            turn_angle = 0.2 * prev_turn_angle + 0.8 * pred_turn_angle
            prev_turn_angle = turn_angle

            # Geschwindigkeit in reale Geschwindigkeit zurückrechnen
            actual_speed = (
                speed_norm
                * ai_cfg["forward_speed_max"]
                * ai_cfg["speed_multiplier"]
            )
        else:
            turn_angle = 0.0
            speed_norm = 0.0
            actual_speed = 0.0

        # Steuerung zuerst
        car.speed = actual_speed
        max_turn = ai_cfg["max_turn_per_step"]

        if turn_angle > max_turn:
            car.angle += max_turn
        elif turn_angle < -max_turn:
            car.angle -= max_turn
        else:
            car.angle += turn_angle

        # Danach erst Update und Zeichnen
        car.update(game_map)
        car.draw(screen, font_small, scale, offset_x, offset_y)

        # Anzeige
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        sensor_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(sensor_surface, (50, 50))

        turn_text = f"turn_angle: {turn_angle:.3f}"
        turn_surface = font_big.render(turn_text, True, (0, 255, 255))
        screen.blit(turn_surface, (50, 90))

        angle_text = f"car.angle: {car.angle:.2f}, max_turn: {max_turn:.2f}"
        angle_surface = font_big.render(angle_text, True, (0, 255, 255))
        screen.blit(angle_surface, (50, 130))

        speed_text = f"speed_norm: {speed_norm:.3f}, speed/frame: {actual_speed:.3f}"
        speed_surface = font_big.render(speed_text, True, (0, 255, 255))
        screen.blit(speed_surface, (50, 170))

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