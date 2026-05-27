import pygame

def load_map(map_path, screen_width, screen_height):
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
    # Ursprüngliche Karte: nur für die Logikberechnung, nicht skalieren
    game_map = pygame.image.load(map_path).convert()
    map_w, map_h = game_map.get_size()

    # Anzeigekarte: nur für die Darstellung, proportional skaliert
    scale = min(screen_width / map_w, screen_height / map_h)
    display_w = int(map_w * scale)
    display_h = int(map_h * scale)

    display_map = pygame.transform.scale(game_map, (display_w, display_h))
    offset_x = (screen_width - display_w) // 2
    offset_y = (screen_height - display_h) // 2
    return screen, game_map, display_map, scale, offset_x, offset_y