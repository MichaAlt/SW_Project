"""
Car RL Agent — TensorFlow CNN + DQN in Pygame
White background, black map walls. Press R to reset, ESC to quit.
Train the car to navigate without hitting walls using sensor rays.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import random
import collections
import numpy as np
import pygame
import tensorflow as tf
from tensorflow import keras

# ── Hyperparameters ──────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 900, 700
FPS              = 60
NUM_RAYS         = 8          # sensor rays per car
RAY_LEN          = 150        # max ray length
CAR_SPEED        = 3.0
CAR_TURN         = 3.5        # degrees per frame

STATE_SIZE       = NUM_RAYS   # one distance per ray (normalised 0-1)
ACTION_SIZE      = 4          # forward, left, right, brake

GAMMA            = 0.95
LR               = 1e-3
BATCH_SIZE       = 64
MEMORY_SIZE      = 10_000
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 0.995
TARGET_UPDATE    = 200        # steps between target-net syncs
TRAIN_EVERY      = 4          # train every N steps

# ── Colours ──────────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0  )
RED    = (220, 50,  50 )
GREEN  = (50,  200, 80 )
BLUE   = (50,  120, 220)
YELLOW = (255, 220, 0  )
GRAY   = (160, 160, 160)

# ── Maps (list of wall rects as (x,y,w,h)) ───────────────────────────────────
MAPS = [
    # Map 0 — simple oval loop
    [
        (0,   0,   900,  20 ),   # top
        (0,   680, 900,  20 ),   # bottom
        (0,   0,   20,   700),   # left
        (880, 0,   20,   700),   # right
        # inner island
        (200, 150, 500,  20 ),
        (200, 530, 500,  20 ),
        (200, 150, 20,   400),
        (680, 150, 20,   400),
    ],
    # Map 1 — figure-8 style
    [
        (0,   0,   900,  20 ),
        (0,   680, 900,  20 ),
        (0,   0,   20,   700),
        (880, 0,   20,   700),
        (200, 100, 20,   220),
        (200, 100, 260,  20 ),
        (440, 100, 20,   220),
        (200, 300, 260,  20 ),
        (440, 380, 20,   220),
        (440, 380, 260,  20 ),
        (680, 380, 20,   220),
        (440, 580, 260,  20 ),
    ],
    # Map 2 — maze-ish
    [
        (0,   0,   900,  20 ),
        (0,   680, 900,  20 ),
        (0,   0,   20,   700),
        (880, 0,   20,   700),
        (100, 100, 200,  20 ),
        (100, 100, 20,   200),
        (100, 280, 200,  20 ),
        (600, 100, 200,  20 ),
        (780, 100, 20,   200),
        (600, 280, 200,  20 ),
        (100, 400, 20,   200),
        (100, 580, 340,  20 ),
        (780, 400, 20,   200),
        (460, 580, 340,  20 ),
        (300, 400, 300,  20 ),
        (300, 400, 20,   180),
        (580, 400, 20,   180),
        (300, 550, 300,  20 ),
    ],
]

# Starting poses per map: (x, y, angle_deg)
STARTS = [
    (100, 350, 0),
    (100, 350, 0),
    (100, 350, 0),
]

# ── DQN Model ────────────────────────────────────────────────────────────────

def build_model(state_size, action_size):
    inp = keras.Input(shape=(state_size,))
    x = keras.layers.Dense(128, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64,  activation="relu")(x)
    out = keras.layers.Dense(action_size, activation="linear")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                  loss="mse")
    return model


class DQNAgent:
    def __init__(self):
        self.memory   = collections.deque(maxlen=MEMORY_SIZE)
        self.epsilon  = EPS_START
        self.model    = build_model(STATE_SIZE, ACTION_SIZE)
        self.target   = build_model(STATE_SIZE, ACTION_SIZE)
        self.sync_target()
        self.steps    = 0
        self.losses   = []
        self.rewards_ep = []

    def sync_target(self):
        self.target.set_weights(self.model.get_weights())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        q = self.model(np.array([state], dtype=np.float32), training=False)
        return int(np.argmax(q[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)
        s  = np.array(s,  dtype=np.float32)
        s2 = np.array(s2, dtype=np.float32)
        r  = np.array(r,  dtype=np.float32)
        d  = np.array(d,  dtype=np.float32)

        q_next = self.target(s2, training=False).numpy()
        targets = self.model(s, training=False).numpy()
        for i in range(BATCH_SIZE):
            targets[i][a[i]] = r[i] + (1 - d[i]) * GAMMA * np.max(q_next[i])

        h = self.model.fit(s, targets, batch_size=BATCH_SIZE,
                           epochs=1, verbose=0)
        self.losses.append(h.history["loss"][0])
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.sync_target()
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


# ── Car ──────────────────────────────────────────────────────────────────────

class Car:
    W, H = 18, 10

    def __init__(self, x, y, angle, walls):
        self.x, self.y, self.angle = x, y, angle
        self.walls = walls
        self.alive = True
        self.dist_total = 0.0
        self.steps = 0

    def get_corners(self):
        hw, hh = self.W / 2, self.H / 2
        cos_a, sin_a = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return [(self.x + px * cos_a - py * sin_a,
                 self.y + px * sin_a + py * cos_a) for px, py in pts]

    def cast_ray(self, angle_offset):
        ax = math.radians(self.angle + angle_offset)
        dx, dy = math.cos(ax), math.sin(ax)
        for t in range(1, RAY_LEN + 1):
            rx, ry = self.x + dx * t, self.y + dy * t
            for wall in self.walls:
                if wall.collidepoint(rx, ry):
                    return t / RAY_LEN, (rx, ry)
        return 1.0, (self.x + dx * RAY_LEN, self.y + dy * RAY_LEN)

    def get_state(self):
        angles = np.linspace(-90, 90, NUM_RAYS)
        state, endpoints = [], []
        for a in angles:
            dist, ep = self.cast_ray(a)
            state.append(dist)
            endpoints.append(ep)
        return np.array(state, dtype=np.float32), endpoints

    def step(self, action):
        if not self.alive:
            return
        if action == 0:   # forward
            self.x += math.cos(math.radians(self.angle)) * CAR_SPEED
            self.y += math.sin(math.radians(self.angle)) * CAR_SPEED
        elif action == 1: # left
            self.angle -= CAR_TURN
            self.x += math.cos(math.radians(self.angle)) * CAR_SPEED
            self.y += math.sin(math.radians(self.angle)) * CAR_SPEED
        elif action == 2: # right
            self.angle += CAR_TURN
            self.x += math.cos(math.radians(self.angle)) * CAR_SPEED
            self.y += math.sin(math.radians(self.angle)) * CAR_SPEED
        elif action == 3: # brake / drift
            self.x += math.cos(math.radians(self.angle)) * CAR_SPEED * 0.3
            self.y += math.sin(math.radians(self.angle)) * CAR_SPEED * 0.3

        self.dist_total += CAR_SPEED
        self.steps += 1

        # Collision check
        corners = self.get_corners()
        for cx, cy in corners:
            for wall in self.walls:
                if wall.collidepoint(cx, cy):
                    self.alive = False
                    return

    def draw(self, surf):
        corners = self.get_corners()
        color = GREEN if self.alive else RED
        pygame.draw.polygon(surf, color, [(int(c[0]), int(c[1])) for c in corners])
        # Direction indicator
        fx = self.x + math.cos(math.radians(self.angle)) * (self.W / 2 + 3)
        fy = self.y + math.sin(math.radians(self.angle)) * (self.W / 2 + 3)
        pygame.draw.circle(surf, BLUE, (int(fx), int(fy)), 3)


# ── HUD ──────────────────────────────────────────────────────────────────────

def draw_hud(surf, agent, episode, car, map_idx, font, small_font):
    lines = [
        f"Episode : {episode}",
        f"Map     : {map_idx + 1}/{len(MAPS)}",
        f"ε       : {agent.epsilon:.3f}",
        f"Steps   : {agent.steps}",
        f"Memory  : {len(agent.memory)}",
        f"Distance: {car.dist_total:.0f}",
    ]
    if agent.losses:
        lines.append(f"Loss    : {agent.losses[-1]:.4f}")

    panel_w = 200
    pygame.draw.rect(surf, (240, 240, 240), (SCREEN_W - panel_w, 0, panel_w, len(lines) * 22 + 20))
    pygame.draw.rect(surf, BLACK, (SCREEN_W - panel_w, 0, panel_w, len(lines) * 22 + 20), 1)
    for i, line in enumerate(lines):
        txt = small_font.render(line, True, BLACK)
        surf.blit(txt, (SCREEN_W - panel_w + 8, 8 + i * 22))

    # Legend
    pygame.draw.rect(surf, (245, 245, 245), (0, SCREEN_H - 30, 400, 30))
    legend = small_font.render("R=Reset  M=NextMap  SPACE=Pause  ESC=Quit", True, GRAY)
    surf.blit(legend, (8, SCREEN_H - 22))


def draw_rays(surf, car, endpoints, state):
    for ep, dist in zip(endpoints, state):
        # Green when far, red when close
        t = dist  # 0=wall close, 1=far
        color = (int(255 * (1 - t)), int(255 * t), 60)
        pygame.draw.line(surf, color, (int(car.x), int(car.y)),
                         (int(ep[0]), int(ep[1])), 1)


def draw_loss_graph(surf, losses, font):
    if len(losses) < 2:
        return
    gx, gy, gw, gh = 10, SCREEN_H - 130, 180, 100
    pygame.draw.rect(surf, (245, 245, 245), (gx, gy, gw, gh))
    pygame.draw.rect(surf, GRAY, (gx, gy, gw, gh), 1)
    recent = losses[-gw:]
    mn, mx = min(recent), max(recent) + 1e-9
    pts = [(gx + i, gy + gh - int((v - mn) / (mx - mn) * gh))
           for i, v in enumerate(recent)]
    if len(pts) > 1:
        pygame.draw.lines(surf, BLUE, False, pts, 1)
    label = font.render("loss", True, GRAY)
    surf.blit(label, (gx + 2, gy + 2))


# ── Main ─────────────────────────────────────────────────────────────────────

def make_walls(map_idx):
    return [pygame.Rect(*r) for r in MAPS[map_idx]]


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Car RL — TensorFlow DQN")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 11)
    big_f  = pygame.font.SysFont("monospace", 18, bold=True)

    agent   = DQNAgent()
    map_idx = 0
    episode = 0
    paused  = False
    train_step = 0
    show_rays  = True

    def reset(mid=None):
        nonlocal map_idx, episode
        if mid is not None:
            map_idx = mid % len(MAPS)
        episode += 1
        walls = make_walls(map_idx)
        sx, sy, sa = STARTS[map_idx]
        return Car(sx, sy, sa, walls), walls

    car, walls = reset(map_idx)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if ev.key == pygame.K_r:
                    car, walls = reset(map_idx)
                if ev.key == pygame.K_m:
                    car, walls = reset(map_idx + 1)
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_v:
                    show_rays = not show_rays

        if not paused:
            state, endpoints = car.get_state()
            action = agent.act(state)
            car.step(action)
            new_state, new_endpoints = car.get_state()

            # Reward shaping
            min_ray = float(np.min(new_state))
            if not car.alive:
                reward = -100.0
                done   = True
            else:
                # Reward for staying alive + surviving near walls is punished
                reward = 1.0 + min_ray * 2.0
                done   = car.steps > 2000  # max episode length

            agent.remember(state, action, reward, new_state, done)
            train_step += 1
            if train_step % TRAIN_EVERY == 0:
                agent.train()

            if done or not car.alive:
                car, walls = reset(map_idx)
            else:
                endpoints = new_endpoints
                state     = new_state

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(WHITE)

        # Walls
        for wall in walls:
            pygame.draw.rect(screen, BLACK, wall)

        # Rays
        if show_rays:
            s, eps = car.get_state()
            draw_rays(screen, car, eps, s)

        # Car
        car.draw(screen)

        # HUD
        draw_hud(screen, agent, episode, car, map_idx, big_f, font)
        draw_loss_graph(screen, agent.losses, font)

        if paused:
            msg = big_f.render("PAUSED — SPACE to resume", True, RED)
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()