import pygame
import sys
import os
import tensorflow as tf
import numpy as np
from pathlib import Path

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from car import Car
from map_loader import load_map
from Config.config_loader import load_config

# Benoetigte Konfigurationen laden
config = load_config()
mode_cfg = config["mode"]
manual_cfg = config["manual_run"]
ai_cfg = config["ai_run"]
simulation_cfg = config["simulation"]
feature_scaling_cfg = config["feature_scaling"]
prediction_cfg = config["prediction"]

# Funktion zur Abfrage der Bildschirmgroesse
def get_screen_size(cfg):
    pygame.init()
    info = pygame.display.Info()

    if cfg["width"] == "auto" or cfg["height"] == "auto":
        return info.current_w, info.current_h

    return cfg["width"], cfg["height"]

# Bildschirmgroesse ermitteln
WIDTH, HEIGHT = get_screen_size(manual_cfg)

def main():
    os.environ["SDL_RENDER_DRIVER"] = "opengl"
    # Pygame-Module initialisieren
    pygame.init()
    
    # Karte laden und alle für die Darstellung benoetigten Werte zurueckgeben
    screen, game_map ,display_map, scale, offset_x, offset_y = load_map(simulation_cfg["map_file"], WIDTH, HEIGHT)
    pygame.display.set_caption("Auto mit Sensoren")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 28)

    car_png_path = '../../PNG_File/car.png'
    car = Car(car_png_path)

    # "data_collection" übernimmt die Funktion der Datensammlung durch manuelles abfahren der ausgeweahlten Strecke.
    # "ai_run" ist die Funktion, bei dem trainierte KI-Modelle die ausgeweahlte Strecke abfahren.
    if mode_cfg["mode"] == "data_collection":
        # Temporäre Daten des aktuellen Durchlaufs
        current_run_data = []

        # Dateipfad für die Trainingsdaten, abhaengig von Prediction-Methode
        if prediction_cfg["prediction"] == "classification":
            save_path = simulation_cfg["data_save_path_classification"]

        elif prediction_cfg["prediction"] == "regression":
            save_path = simulation_cfg["data_save_path_regression"]

        # Speicherpfad laden
        file_path = ROOT_DIR / "Supervised_Learning" / "ai" / save_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Modell laden, abhaengig von Prediction-Methode
    if mode_cfg["mode"] == "ai_run":
        # KI-Modell laden
        if prediction_cfg["prediction"] == "classification":
            model = tf.keras.models.load_model(ai_cfg["model_path_classification"])
        elif prediction_cfg["prediction"] == "regression":
            model = tf.keras.models.load_model(ai_cfg["model_path_regression"])

    # Simulationsschleife 
    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        car.update(game_map)
        car.draw(screen, font_small, scale, offset_x, offset_y)

        pygame.display.flip()
        clock.tick(60)

        # Anzeige der Sensorwerte
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        text_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(text_surface, (50, 50))

        # Datensammlung durch manuelles fahren
        if mode_cfg["mode"] == "data_collection":
            keys = pygame.key.get_pressed()

            # Steuerung des Autos
            actions = []

            # Steuerung beim Datensammeln, abhaengig von Prediction-Methode
            # Steuerung fuer Mehrklassifikation
            if prediction_cfg["prediction"] == "classification":
                if keys[pygame.K_w]: # Vorwaerts fahren
                    car.speed = manual_cfg["forward_speed"]
                    actions.append("W")
                elif keys[pygame.K_s]: # Rueckwaerts fahren (Nicht verwendet)
                    car.speed = manual_cfg["backward_speed"]
                    actions.append("S")
                else:  # Wenn Auto gecrasht
                    if car.alive:
                        car.speed = 0

                if keys[pygame.K_a]: # Links lenken
                    car.angle += manual_cfg["turn_angle"]
                    actions.append("A")
                if keys[pygame.K_d]: # Rechts Lenken
                    car.angle -= manual_cfg["turn_angle"]
                    actions.append("D")

            # Steuerung fuer Regression
            elif prediction_cfg["prediction"] == "regression":

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
                if prediction_cfg["prediction"] == "classification":
                    data = car.radar_values + ["+".join(actions)] # Mit Action bzw. Label als String
                elif prediction_cfg["prediction"] == "regression":
                    data = car.radar_values + actions # Mit Action bzw. Label als Interger

                if not current_run_data or current_run_data[-1] != data:
                    current_run_data.append(data)

            # Nach einem Crash die temporären Daten dieses Durchlaufs verwerfen
            if not car.alive:
                current_run_data.clear()

        # KI-Modell. Trainiertes Modell faehrt auf der Strecke
        if mode_cfg["mode"] == "ai_run":
            
            if len(car.radar_values) == 5:
                # Umwandlung der Radarwerte in ein NumPy-Array fuer die Prediction
                x_input = np.array(car.radar_values, dtype=np.float32).reshape(1, -1) 

                if feature_scaling_cfg["method"] == 1: # Normalisieren
                    x_input = x_input/ 298.0
                elif feature_scaling_cfg["method"] == 2: # Standardisieren
                    mean = np.load("../ai/data_file/mean.npy")
                    std = np.load("../ai/data_file/std.npy")
                    x_input = (x_input - mean) / std

                # Prediction des trainierten Modells
                pred = model(x_input, training=False)
                if prediction_cfg["prediction"] == "classification":
                    action_index = np.argmax(pred[0])
                    mapping = {0: "W", 1: "A", 2: "D", 3: "W+A", 4: "W+D"}
                    actions = mapping[action_index].split("+")

                elif prediction_cfg["prediction"] == "regression":
                    # Prediction-Tensor verursacht bei Regression-Vorhersage Perfomanceprobleme, deshalb casting auf float
                    speed = float(pred[0][0]) 
                    steer = float(pred[0][1])

            else:  # Nach einem Crash ist das InputArray leer. Deshalb keine Ausfuehrung einer Vorhersage, um Programmabsturz zu vermeiden. 
                if prediction_cfg["prediction"] == "classification":
                    actions = ["W"]
                elif prediction_cfg["prediction"] == "regression":
                    speed = 0
                    steer = 0

            car.speed = 0

            # Ausfuehrung von Actions
            if prediction_cfg["prediction"] == "classification":
                if "W" in actions:
                    car.speed = ai_cfg["forward_speed"]
                if "S" in actions:
                    car.speed = ai_cfg["backward_speed"]
                if "A" in actions:
                    car.angle += ai_cfg["turn_angle"]
                if "D" in actions:
                    car.angle -= ai_cfg["turn_angle"]
            
            elif prediction_cfg["prediction"] == "regression":
                car.speed = speed
                car.angle += steer

        # Ereignisbehandlung
        for event in pygame.event.get():
            
            # Beenden der Simulation beim schließen des Fensters
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                
                # Beenden der Simulation durch Esc- oder Q-Taste
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                
                # Speichern der temporaeren Daten in eine CSV-Datei durch P-Taste
                elif (event.key == pygame.K_p) & (mode_cfg["mode"] == "data_collection"):
                    with open(file_path, "a") as f:
                        for row in current_run_data:
                            line = ",".join(map(str, row))

                            if prediction_cfg["prediction"] == "classification":
                                row_len = 6 
                            elif prediction_cfg["prediction"] == "regression":
                                row_len = 7

                            # Nur speichern der jeweiligen Zeile, wenn Spaltenanzahl der Daten korrekt
                            if len(row) == row_len:
                                f.write(line + "\n")
                    print("Runde gespeichert!")
                    current_run_data.clear()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()