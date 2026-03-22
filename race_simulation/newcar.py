import pygame
import math
import sys
import os

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255)

START_POS = [830, 920]
START_ANGLE = 0

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.reset()

    def reset(self):
        self.position = START_POS.copy()
        self.angle = START_ANGLE
        self.speed = 0
        self.alive = True
        self.radars = []
        self.radar_values = [0] * 5

    def draw(self, screen, font):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radars(screen, font)

    def update(self, game_map):
        if not self.alive:
            self.reset()
            return

        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)

        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        self.center = [
            int(self.position[0] + CAR_SIZE_X / 2),
            int(self.position[1] + CAR_SIZE_Y / 2)
        ]

        x, y = self.center
        if game_map.get_at((x, y))[:3] == BORDER_COLOR:
            self.alive = False
            self.speed = 0
            print("Crash! Auto zurückgesetzt")

        self.radars.clear()
        self.radar_values = []
        for d in range(-90, 120, 45):
            dist = self.check_radar(d, game_map)
            self.radar_values.append(dist)

    def check_radar(self, degree, game_map):
        length = 0
        while length < 300:
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                break

            if game_map.get_at((x, y))[:3] == BORDER_COLOR:
                break

            length += 2

        dist = int(math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2))
        self.radars.append([(x, y), dist])
        return dist

    def draw_radars(self, screen, font):
        for radar in self.radars:
            pygame.draw.line(screen, (0, 255, 0), self.center, radar[0], 2)
            pygame.draw.circle(screen, (0, 255, 0), radar[0], 5)

            text = font.render(str(radar[1]), True, (255, 0, 0))
            screen.blit(text, (radar[0][0] + 5, radar[0][1] - 5))


def main():
    pygame.init()
    os.environ["SDL_RENDER_DRIVER"] = "opengl"

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = screen.get_size()

    pygame.display.set_caption("Car mit Sensoren")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 28)

    car = Car()
    game_map = pygame.image.load('map.png').convert()

    # Temporaere Daten
    current_run_data = []

    # Dateipfad Trainingsdaten
    file_path = os.path.join(os.path.dirname(__file__), "..", "ai", "data_file", "training_data.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    running = True
    while running:
        screen.blit(game_map, (0, 0))

        car.update(game_map)
        car.draw(screen, font_small)

        # Sensoranzeige
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        text_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(text_surface, (50, 50))

        pygame.display.flip()
        clock.tick(60)

        keys = pygame.key.get_pressed()

        # Steuerung
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
            if not current_run_data or current_run_data[-1] != data: # Doppelte Daten hintereinander
                current_run_data.append(data)
                

        # Temporaere Daten verwerfen beim Crash
        if not car.alive:
            current_run_data.clear()

        # Event-Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key == pygame.K_p:  # Daten mit Taste P speichern
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