import pygame
import sys
import os
import tensorflow as tf
import numpy as np
from car import Car
from map_loader import load_map

WIDTH = 1920
HEIGHT = 1080

def main():
    os.environ["SDL_RENDER_DRIVER"] = "opengl"
    pygame.init()
    
    screen, game_map, display_map, scale, offset_x, offset_y = load_map("map.png", WIDTH, HEIGHT)
    pygame.display.set_caption("Car mit Sensoren")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 28)

    car = Car()

    model = tf.keras.models.load_model('../ai/models_file/model.h5')  # KI-Modell laden

    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        car.update(game_map)
        car.draw(screen, font_small, scale, offset_x, offset_y)

        # KI-Modell
        if len(car.radar_values) == 5:
            x_input = np.array(car.radar_values, dtype=np.float32).reshape(1, -1) / 300.0  # reshape der Radarwerte und Normalisierung
          
            #pred = model.predict(x_input, verbose=0) # Keine Predict-Ausgabe in der Konsole # Starke Performanceeinbrueche durch predict, wahrscheinlich durch 60 Aufrufe pro Minute
            pred = model(x_input, training=False)     # Einfacher Forward-Pass, ohne Schleifen ueber Daten, Dataset-Handling etc. 
            action_index = np.argmax(pred[0])         # Tasten-Index der höchsten Wahrscheinlichkeit

            mapping = {0: "W", 1: "A", 2: "S", 3: "D", 4: "W+A", 5: "W+D", 6: "S+A", 7: "S+D"}
            actions = mapping[action_index].split("+")  # Liste mit Aktionstasten, mehrere Tasteneingaben werden gesplittet
        else: #Beim Crash ist InputArray leer, dadurch Programmabsturz, fuer die Zeit kein predict
            actions = ["W"]

        # Anzeige der Sensorwerte
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        text_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(text_surface, (50, 50))

        # Steuerung des Autos
        car.speed = 0
        if "W" in actions:
            car.speed = 10 # Speed bis 20 moeglich
        if "S" in actions:
            car.speed = -5
        if "A" in actions:
            car.angle += 3
        if "D" in actions:
            car.angle -= 3

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