import pygame
import math

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