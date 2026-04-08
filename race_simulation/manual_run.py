import pygame
import sys
import os
from car import Car
from map_loader import load_map

WIDTH = 1920
HEIGHT = 1080

def main():
    os.environ["SDL_RENDER_DRIVER"] = "opengl"
    pygame.init()
    
    screen, game_map ,display_map, scale, offset_x, offset_y = load_map("map.png", WIDTH, HEIGHT)
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
        "..",
        "ai",
        "data_file",
        "training_data_map.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        car.update(display_map)
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
            car.speed = 5
            actions.append("W")
        elif keys[pygame.K_s]:
            car.speed = -5
            actions.append("S")
        else:
            if car.alive:
                car.speed = 0

        if keys[pygame.K_a]:
            car.angle += 3
            actions.append("A")
        if keys[pygame.K_d]:
            car.angle -= 3
            actions.append("D")

        # Daten sammeln
        if car.alive and actions:
            data = car.radar_values + ["+".join(actions)]
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
                            f.write(line + "\n")
                    print("Runde gespeichert!")
                    current_run_data.clear()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()