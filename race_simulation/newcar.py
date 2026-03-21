import pygame
import math
import sys

# Fullscreen
WIDTH = 1920
HEIGHT = 1080

# Auto-Größe
CAR_SIZE_X = 60
CAR_SIZE_Y = 60

# Farbe, die den Rand der Strecke markiert
BORDER_COLOR = (255, 255, 255, 255)

# Startposition
START_POS = [830, 920]
START_ANGLE = 0

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.reset()

    def reset(self):
        """Auto auf Startlinie zurücksetzen"""
        self.position = START_POS.copy()
        self.angle = START_ANGLE
        self.speed = 0
        self.alive = True
        self.radars = []
        self.radar_values = [0]*5

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radars(screen)

    def update(self, game_map):
        if not self.alive:
            self.reset()
            return

        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed #PyGame-Koordinatensystem:  Y-Achse zeigt nach unten, deshalb korrektur mit 360-self.angle
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        self.center = [int(self.position[0] + CAR_SIZE_X / 2), int(self.position[1] + CAR_SIZE_Y / 2)]

        # Crash prüfen
        x, y = self.center
        if game_map.get_at((x, y)) == BORDER_COLOR:
            self.alive = False
            self.speed = 0
            print("Crash! Auto zurückgesetzt")

        # Radars aktualisieren
        self.radars.clear()
        self.radar_values = []
        for d in range(-90, 120, 45):
            dist = self.check_radar(d, game_map)
            self.radar_values.append(dist)

    def check_radar(self, degree, game_map):
        length = 0
        x, y = self.center
        while True:
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                break
            if game_map.get_at((x, y)) == BORDER_COLOR or length > 300:
                break
            length += 1
        dist = int(math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2))
        self.radars.append([(x, y), dist])
        return dist

    def draw_radars(self, screen):
        for i, radar in enumerate(self.radars):
            pygame.draw.line(screen, (0, 255, 0), self.center, radar[0], 2)
            pygame.draw.circle(screen, (0, 255, 0), radar[0], 5)
            # Sensorwert neben Linie anzeigen
            font = pygame.font.SysFont("Arial", 24)
            text = font.render(str(radar[1]), True, (255, 0, 0))
            screen.blit(text, (radar[0][0]+5, radar[0][1]-5))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Car auf Strecke mit Sensoren")
    clock = pygame.time.Clock()
    car = Car()

    game_map = pygame.image.load('map2.png').convert()

    running = True
    while running:
        screen.blit(game_map, (0, 0))
        car.draw(screen)
        car.update(game_map)

        # Optional: Gesamte Sensorwerte rechts oben
        font = pygame.font.SysFont("Arial", 28)
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        text_surface = font.render(sensor_text, True, (255, 255, 0))
        screen.blit(text_surface, (50, 50))

        pygame.display.flip()
        clock.tick(60)

        # Event-Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

        # Tasteneingaben
        keys = pygame.key.get_pressed()
        key_pressed = []

        if keys[pygame.K_w]:
            car.speed = 5
            key_pressed.append("W")
        elif keys[pygame.K_s]:
            car.speed = -5
            key_pressed.append("S")
        else:
            if car.alive:
                car.speed = 0

        if keys[pygame.K_a]:
            car.angle += 5
            key_pressed.append("A")
        if keys[pygame.K_d]:
            car.angle -= 5
            key_pressed.append("D")

        if key_pressed:
            print("Tasten gedrückt:", "+".join(key_pressed))

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()