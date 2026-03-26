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
        self.radar_values = []
        self.rotated_sprite = self.sprite
        self.center = [
            int(self.position[0] + CAR_SIZE_X / 2),
            int(self.position[1] + CAR_SIZE_Y / 2)
        ]

    def draw(self, screen, font, scale, offset_x, offset_y):
        scaled_sprite = pygame.transform.scale(
            self.rotated_sprite,
            (
                max(1, int(self.rotated_sprite.get_width() * scale)),
                max(1, int(self.rotated_sprite.get_height() * scale))
            )
        )

        draw_x = int(self.position[0] * scale + offset_x)
        draw_y = int(self.position[1] * scale + offset_y)

        screen.blit(scaled_sprite, (draw_x, draw_y))
        self.draw_radars(screen, font, scale, offset_x, offset_y)

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

        # Zuerst prüfen, ob das Auto außerhalb der Karte ist, danach auf Kollision mit dem Rand prüfen
        map_w, map_h = game_map.get_size()
        if x < 0 or x >= map_w or y < 0 or y >= map_h:
            self.alive = False
            self.speed = 0
            print("Crash! Auto zurückgesetzt")
            return

        if game_map.get_at((x, y))[:3] == BORDER_COLOR:
            self.alive = False
            self.speed = 0
            print("Crash! Auto zurückgesetzt")
            return

        self.radars.clear()
        self.radar_values = []
        for d in range(-90, 120, 45):
            dist = self.check_radar(d, game_map)
            self.radar_values.append(dist)

    def check_radar(self, degree, game_map):
        length = 0
        map_w, map_h = game_map.get_size()

        while length < 300:
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

            if x < 0 or x >= map_w or y < 0 or y >= map_h:
                break

            if game_map.get_at((x, y))[:3] == BORDER_COLOR:
                break

            length += 2

        dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        self.radars.append([(x, y), dist])
        return dist

    def draw_radars(self, screen, font, scale, offset_x, offset_y):
        center_draw = (
            int(self.center[0] * scale + offset_x),
            int(self.center[1] * scale + offset_y)
        )

        for radar in self.radars:
            radar_draw = (
                int(radar[0][0] * scale + offset_x),
                int(radar[0][1] * scale + offset_y)
            )

            pygame.draw.line(screen, (0, 255, 0), center_draw, radar_draw, 2)
            pygame.draw.circle(screen, (0, 255, 0), radar_draw, max(2, int(5 * scale)))

            text = font.render(str(radar[1]), True, (255, 0, 0))
            screen.blit(text, (radar_draw[0] + 5, radar_draw[1] - 5))


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

    # Ursprüngliche Karte: nur für die Logikberechnung, nicht skalieren
    game_map = pygame.image.load('map5.png').convert()
    map_w, map_h = game_map.get_size()

    # Anzeigekarte: nur für die Darstellung, proportional skaliert
    scale = min(WIDTH / map_w, HEIGHT / map_h)
    display_w = int(map_w * scale)
    display_h = int(map_h * scale)

    display_map = pygame.transform.scale(game_map, (display_w, display_h))
    offset_x = (WIDTH - display_w) // 2
    offset_y = (HEIGHT - display_h) // 2

    # Temporäre Daten des aktuellen Durchlaufs
    current_run_data = []

    # Dateipfad für die Trainingsdaten
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ai",
        "data_file",
        "training_data_map5.csv"
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