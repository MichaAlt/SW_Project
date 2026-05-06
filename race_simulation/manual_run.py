import pygame
import sys
import os
from collections import deque

from pathlib import Path
from car import Car
from map_loader import load_map

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
manual_cfg = config["manual_run"]

def get_screen_size(cfg):
    pygame.init()
    info = pygame.display.Info()

    if cfg["width"] == "auto" or cfg["height"] == "auto":
        return info.current_w, info.current_h

    return cfg["width"], cfg["height"]

WIDTH, HEIGHT = get_screen_size(manual_cfg)

def save_current_run(file_path, current_run_data):
    if current_run_data:
        with open(file_path, "a") as f:
            for row in current_run_data:
                line = ",".join(map(str, row))
                if len(row) == 10:
                    f.write(line + "\n")
        print(f"Daten gespeichert! Zeilen: {len(current_run_data)}")
    else:
        print("Keine Daten zum Speichern vorhanden!")

def main():
    os.environ["SDL_RENDER_DRIVER"] = "opengl"
    pygame.init()
    
    screen, game_map, display_map, scale, offset_x, offset_y = load_map(
        manual_cfg["map_file"], WIDTH, HEIGHT
    )
    pygame.display.set_caption("Car mit Sensoren")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 28)

    car = Car()

    # Startposition merken
    start_x = car.position[0]
    start_y = car.position[1]

    # Temporäre Daten des aktuellen Durchlaufs
    current_run_data = []

    # Buffer für mehrere Zustände
    state_buffer = deque(maxlen=3)  # aktueller Zustand + 5 Frames Abstand

    # Für weniger Speicherung auf Geraden
    straight_frame_count = 0

    # Für automatische Rundenerkennung
    frames_since_start = 0
    lap_saved = False

    # Dateipfad für die Trainingsdaten
    file_path = os.path.join(
        os.path.dirname(__file__),
        manual_cfg["data_save_path"]
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    running = True
    while running:
        frames_since_start += 1

        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        keys = pygame.key.get_pressed()
        has_input = False

        # Steuerung des Autos
        if keys[pygame.K_w]:
            car.speed = manual_cfg["forward_speed"]
            has_input = True
        elif keys[pygame.K_s]:
            car.speed = manual_cfg["backward_speed"]
            has_input = True
        else:
            if car.alive:
                car.speed = 0

        if keys[pygame.K_a]:
            car.angle += manual_cfg["turn_angle"]
            has_input = True

        if keys[pygame.K_d]:
            car.angle -= manual_cfg["turn_angle"]
            has_input = True

        car.update(game_map)
        car.draw(screen, font_small, scale, offset_x, offset_y)

        # Anzeige der Sensorwerte
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        text_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(text_surface, (50, 50))

        pygame.display.flip()
        clock.tick(60)

        # Speicherrate: Kurven häufiger, Geraden seltener
        is_turning = keys[pygame.K_a] or keys[pygame.K_d]
        should_save = False

        if is_turning:
            should_save = True
            straight_frame_count = 0
        else:
            straight_frame_count += 1
            if straight_frame_count % 5 == 0:
                should_save = True

        # Daten sammeln: Zustand_t -> dx, dy zu Zustand_t+5
        if car.alive and len(car.radar_values) == 5 and has_input and should_save:
            current_state = {
                "radar": car.radar_values.copy(),
                "x": car.position[0],
                "y": car.position[1],
                "angle": car.angle
            }

            state_buffer.append(current_state)

            if len(state_buffer) == 3:
                old_state = state_buffer[0]
                new_state = state_buffer[-1]

                dx = new_state["x"] - old_state["x"]
                dy = new_state["y"] - old_state["y"]

                norm = (dx ** 2 + dy ** 2) ** 0.5 + 1e-8
                dx = dx / norm
                dy = dy / norm

                data = (
                    old_state["radar"]
                    + [old_state["x"], old_state["y"], old_state["angle"], dx, dy]
                )

                if not current_run_data or current_run_data[-1] != data:
                    current_run_data.append(data)

        # Automatische Rundenerkennung und Speicherung
        distance_to_start = ((car.position[0] - start_x) ** 2 + (car.position[1] - start_y) ** 2) ** 0.5

        if (
            car.alive
            and frames_since_start > 300
            and distance_to_start < 50
            and not lap_saved
        ):
            save_current_run(file_path, current_run_data)
            current_run_data.clear()
            state_buffer.clear()
            straight_frame_count = 0
            lap_saved = True

        # Wenn sich das Auto wieder vom Start entfernt, darf die nächste Runde wieder gespeichert werden
        if distance_to_start > 100:
            lap_saved = False

        # Nach einem Crash die temporären Daten dieses Durchlaufs verwerfen
        if not car.alive:
            current_run_data.clear()
            state_buffer.clear()
            straight_frame_count = 0
            frames_since_start = 0
            lap_saved = False

        # Ereignisbehandlung
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

                elif event.key == pygame.K_p:
                    save_current_run(file_path, current_run_data)
                    current_run_data.clear()
                    state_buffer.clear()
                    straight_frame_count = 0
                    frames_since_start = 0
                    lap_saved = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()