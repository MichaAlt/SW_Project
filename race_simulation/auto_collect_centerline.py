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
auto_cfg = config["auto_collect_centerline"]


def get_screen_size(cfg):
    pygame.init()
    info = pygame.display.Info()

    if cfg["width"] == "auto" or cfg["height"] == "auto":
        return info.current_w, info.current_h

    return cfg["width"], cfg["height"]


WIDTH, HEIGHT = get_screen_size(auto_cfg)


def clamp(value, low, high):
    return max(low, min(high, value))


def centerline_controller(radar_values, cfg):
    """
    radar_values Reihenfolge:
    [r_right, r_right_front, r_front, r_left_front, r_left]
    """
    r_right, r_right_front, r_front, r_left_front, r_left = radar_values

    # 1) 中线保持：左右距离尽量相等
    side_error = r_left - r_right

    # 2) 提前看弯：左右前方差
    front_error = r_left_front - r_right_front

    # 3) 综合转向
    steering = cfg["side_error_gain"] * side_error + cfg["front_error_gain"] * front_error

    # 当前方太近时，更强烈转向
    if r_front < cfg["front_close_threshold"]:
        steering += cfg["front_close_extra_gain"] * front_error

    steering = clamp(steering, -1.0, 1.0)

    # 4) 自动采集时实际驾驶速度（只是为了把车开稳）
    if r_front < cfg["speed_front_very_close"]:
        speed = cfg["speed_very_slow"]
    elif r_front < cfg["speed_front_close"]:
        speed = cfg["speed_slow"]
    elif abs(steering) > cfg["speed_turn_hard"]:
        speed = cfg["speed_slow"]
    elif abs(steering) > cfg["speed_turn_medium"]:
        speed = cfg["speed_medium"]
    else:
        speed = cfg["speed_fast"]

    return steering, speed


def build_training_rows_from_distance(
    states,
    lookahead_distance=8,
    base_speed_max=5.0,
    reference_fps=60.0
):
    rows = []

    for i in range(len(states)):
        start = states[i]
        target_index = None

        # 找到前方直线距离达到 lookahead_distance 的点
        for j in range(i + 1, len(states)):
            dx_step = states[j]["x"] - start["x"]
            dy_step = states[j]["y"] - start["y"]
            dist = (dx_step ** 2 + dy_step ** 2) ** 0.5

            if dist >= lookahead_distance:
                target_index = j
                break

        if target_index is None:
            continue

        target = states[target_index]

        # 先算未来方向 dx, dy
        dx = target["x"] - start["x"]
        dy = target["y"] - start["y"]

        norm = (dx ** 2 + dy ** 2) ** 0.5 + 1e-8
        dx = dx / norm
        dy = dy / norm

        # 再把 dx, dy 转成目标角
        target_angle = (-pygame.math.Vector2(dx, dy).as_polar()[1]) % 360

        # 当前还需要转多少角度（范围 -180 ~ 180）
        turn_angle = (target_angle - start["angle"] + 180) % 360 - 180

        # 速度：用精确时间差 -> 每秒速度 -> 等效每帧速度
        dt = target["time"] - start["time"]
        if dt <= 1e-8:
            continue

        speed_per_second = lookahead_distance / dt
        speed_per_frame = speed_per_second / reference_fps

        speed_norm = speed_per_frame / base_speed_max
        speed_norm = clamp(speed_norm, 0.0, 1.0)

        # 10列:
        # 5 sensor + x + y + angle + turn_angle + speed_norm
        row = (
            start["radar"]
            + [start["x"], start["y"], start["angle"], turn_angle, speed_norm]
        )
        rows.append(row)

    return rows


def save_current_run(file_path, current_states, cfg):
    rows = build_training_rows_from_distance(
        current_states,
        lookahead_distance=cfg["lookahead_distance"],
        base_speed_max=cfg["base_speed_max"],
        reference_fps=cfg["reference_fps"]
    )

    if rows:
        with open(file_path, "a") as f:
            for row in rows:
                line = ",".join(map(str, row))
                if len(row) == 10:
                    f.write(line + "\n")
        print(f"Daten gespeichert! Zeilen: {len(rows)}")
    else:
        print("Keine Daten zum Speichern vorhanden!")


def main():
    os.environ["SDL_RENDER_DRIVER"] = "opengl"
    pygame.init()

    screen, game_map, display_map, scale, offset_x, offset_y = load_map(
        auto_cfg["map_file"], WIDTH, HEIGHT
    )
    pygame.display.set_caption("Auto Centerline Data Collector")
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Arial", 28)

    car = Car()

    start_x = car.position[0]
    start_y = car.position[1]

    current_states = []

    frames_since_start = 0
    lap_saved = False

    file_path = os.path.join(
        os.path.dirname(__file__),
        auto_cfg["data_save_path"]
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    max_turn_per_step = auto_cfg["max_turn_per_step"]

    running = True
    while running:
        frames_since_start += 1

        screen.fill((0, 0, 0))
        screen.blit(display_map, (offset_x, offset_y))

        # 自动控制
        if car.alive and len(car.radar_values) == 5:
            steering, speed = centerline_controller(car.radar_values, auto_cfg)

            car.speed = speed
            car.angle += steering * max_turn_per_step
        else:
            steering, speed = 0.0, 0.0
            car.speed = 0.0

        car.update(game_map)
        car.draw(screen, font_big, scale, offset_x, offset_y)

        # 显示
        sensor_text = "Sensorwerte: " + ", ".join(str(v) for v in car.radar_values)
        sensor_surface = font_big.render(sensor_text, True, (255, 255, 0))
        screen.blit(sensor_surface, (50, 50))

        status_text = f"steering={steering:.2f}, speed={speed:.2f}"
        status_surface = font_big.render(status_text, True, (0, 255, 255))
        screen.blit(status_surface, (50, 90))

        pygame.display.flip()
        clock.tick(60)

        # 记录原始轨迹
        if car.alive and len(car.radar_values) == 5:
            current_time = pygame.time.get_ticks() / 1000.0

            current_state = {
                "radar": car.radar_values.copy(),
                "x": car.position[0],
                "y": car.position[1],
                "angle": car.angle,
                "time": current_time
            }

            if not current_states:
                current_states.append(current_state)
            else:
                last = current_states[-1]
                if (
                    last["radar"] != current_state["radar"]
                    or last["x"] != current_state["x"]
                    or last["y"] != current_state["y"]
                    or last["angle"] != current_state["angle"]
                ):
                    current_states.append(current_state)

        # 自动按圈保存
        distance_to_start = (
            (car.position[0] - start_x) ** 2 + (car.position[1] - start_y) ** 2
        ) ** 0.5

        if (
            car.alive
            and frames_since_start > auto_cfg["lap_min_frames"]
            and distance_to_start < auto_cfg["lap_save_radius"]
            and not lap_saved
        ):
            save_current_run(file_path, current_states, auto_cfg)
            current_states.clear()
            lap_saved = True

        if distance_to_start > auto_cfg["lap_reset_radius"]:
            lap_saved = False

        # 撞墙重来
        if not car.alive:
            current_states.clear()
            frames_since_start = 0
            lap_saved = False

        # 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key == pygame.K_p:
                    save_current_run(file_path, current_states, auto_cfg)
                    current_states.clear()
                    frames_since_start = 0
                    lap_saved = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()