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

def normalize_input(radar_values, angle):
    x_input = np.array(
        radar_values + [angle],
        dtype=np.float32
    ).reshape(1, -1)

    # Radarwerte normalisieren
    x_input[:, 0:5] = x_input[:, 0:5] / 298.0

    # Winkel skalieren
    x_input[:, 5] = x_input[:, 5] / 360.0

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

    prev_dx = 0.0
    prev_dy = 0.0

    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        if car.alive and len(car.radar_values) == 5:
            x_input = normalize_input(
                car.radar_values,
                car.angle
            )

            pred = model(x_input, training=False)
            raw_dx = float(pred[0][0])
            raw_dy = float(pred[0][1])

            # Roh-Ausgabe normieren
            norm = (raw_dx ** 2 + raw_dy ** 2) ** 0.5 + 1e-8
            raw_dx = raw_dx / norm
            raw_dy = raw_dy / norm

            # Glättung
            dx = 0.0 * prev_dx + 1.0 * raw_dx
            dy = 0.0 * prev_dy + 1.0 * raw_dy

            # Nach der Glättung erneut normieren
            smooth_norm = (dx ** 2 + dy ** 2) ** 0.5 + 1e-8
            dx = dx / smooth_norm
            dy = dy / smooth_norm

            print(
                f"raw_dx={raw_dx:.3f}, raw_dy={raw_dy:.3f}, "
                f"dx={dx:.3f}, dy={dy:.3f}"
            )

            prev_dx = dx
            prev_dy = dy

            target_angle = (-np.degrees(np.arctan2(dy, dx))) % 360
            angle_diff = (target_angle - car.angle + 180) % 360 - 180
        else:
            dx = prev_dx
            dy = prev_dy
            target_angle = car.angle
            angle_diff = 0.0

        # Steuerung zuerst
        car.speed = ai_cfg["forward_speed"]
        max_turn = ai_cfg["max_turn_per_step"]

        if angle_diff > max_turn:
            car.angle += max_turn
        elif angle_diff < -max_turn:
            car.angle -= max_turn
        else:
            car.angle += angle_diff

        # Danach erst Update und Zeichnen
        car.update(game_map)
        car.draw(screen, font_small, scale, offset_x, offset_y)

        # Anzeige
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        sensor_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(sensor_surface, (50, 50))

        dxdy_text = f"dx: {dx:.3f}, dy: {dy:.3f}"
        dxdy_surface = font_big.render(dxdy_text, True, (0, 255, 255))
        screen.blit(dxdy_surface, (50, 90))

        angle_text = f"Target angle: {target_angle:.2f}, car.angle: {car.angle:.2f}, diff: {angle_diff:.2f}"
        angle_surface = font_big.render(angle_text, True, (0, 255, 255))
        screen.blit(angle_surface, (50, 130))

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