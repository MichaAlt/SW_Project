import pygame
import sys
import os

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

def main():
    os.environ["SDL_RENDER_DRIVER"] = "opengl"
    pygame.init()
    
    screen, game_map ,display_map, scale, offset_x, offset_y = load_map(manual_cfg["map_file"], WIDTH, HEIGHT)
    pygame.display.set_caption("Car mit Sensoren")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 28)

    car = Car()

   
    # Temporäre Daten des aktuellen Durchlaufs
    current_run_data = []

    # Dateipfad für die Trainingsdaten
    file_path = os.path.join(
        os.path.dirname(__file__),
        manual_cfg["data_save_path"]
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        car.update(game_map)
        car.draw(screen, font_small, scale, offset_x, offset_y)

        # Anzeige der Sensorwerte
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        text_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(text_surface, (50, 50))

        pygame.display.flip()
        clock.tick(60)

        keys = pygame.key.get_pressed()

        # Steuerung des Autos
        actions = []
        if keys[pygame.K_w]:
            car.speed = manual_cfg["forward_speed"]     # Vorwärts
            actions.append(car.speed)

        elif keys[pygame.K_s]:                          # Rückwärts
            car.speed = manual_cfg["backward_speed"]
            actions.append(car.speed)
        else:
            if car.alive:
                car.speed = 0
                actions.append(car.speed)

        if keys[pygame.K_a] and keys[pygame.K_LSHIFT]: # Links Lenken
            car.angle += manual_cfg["turn_angle"]
            actions.append(+manual_cfg["turn_angle"])
        else:
            if keys[pygame.K_a] and not keys[pygame.K_LSHIFT]: # Sanftes Links Lenken
                car.angle += manual_cfg["turn_angle_soft"]
                actions.append(+manual_cfg["turn_angle_soft"])

        if keys[pygame.K_d] and keys[pygame.K_LSHIFT]: # Rechts Lenken
            car.angle -= manual_cfg["turn_angle"]
            actions.append(-manual_cfg["turn_angle"])
        else:
            if keys[pygame.K_d] and not keys[pygame.K_LSHIFT]: # Sanftes Rechts Lenken
                car.angle -= manual_cfg["turn_angle_soft"]
                actions.append(-manual_cfg["turn_angle_soft"])

        if not keys[pygame.K_a] and not keys[pygame.K_d]: # Nicht Lenken
            actions.append(0)

        # Daten sammeln
        if car.alive and actions:
            # data = car.radar_values + ["+".join(actions)] # Mit Action bzw. Label als String
            data = car.radar_values + actions # Mit Action bzw. Label als Interger
            if not current_run_data or current_run_data[-1] != data:
                current_run_data.append(data)

        # Nach einem Crash die temporären Daten dieses Durchlaufs verwerfen
        if not car.alive:
            current_run_data.clear()

        # Ereignisbehandlung
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key == pygame.K_p:
                    with open(file_path, "a") as f:
                        for row in current_run_data:
                            line = ",".join(map(str, row))

                            if len(row) == 7:
                                f.write(line + "\n")
                    print("Runde gespeichert!")
                    current_run_data.clear()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
