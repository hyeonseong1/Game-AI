"""
Space Invaders RL environment — mirrors space-invaders-core.js physics at 60 fps.

Agent controls the PLAYER ship (shoot aliens, dodge bombs).

Observation (20 floats):
  0   player centre-x / W
  1   bullet alive  (0/1)
  2   bullet x / W       (0 if dead)
  3   bullet y / H       (0 if dead)
  4   n_alive / 55
  5   formation left  / W
  6   formation right / W
  7   formation bottom / H
  8   formation centre-x / W
  9   formation direction  (+1 right / -1 left, normalised to 0/1)
  10-15  nearest 3 bombs: [dx/W, y/H] each (0,0 if absent)
  16  nearest alien column dx / W  (aiming hint)
  17  wave / 10
  18  lives / 3
  19  mystery ship x / W  (0 if absent)

Actions (6):
  0 stay      1 left      2 right
  3 fire      4 fire+left 5 fire+right
"""

import numpy as np

# ── Constants matching CONFIG in space-invaders-core.js ─────────────────────
W  = 224           # LOGICAL_WIDTH
H  = 256           # LOGICAL_HEIGHT
FT = 1000 / 60     # frame time ms (~16.667)

PLAYER_SPEED  = 1
PLAYER_Y      = 216
PLAYER_W      = 13
PLAYER_H      = 8
BULLET_SPEED  = 4
BULLET_W      = 1
BULLET_H      = 4
BOMB_SPEED    = 1
BOMB_W        = 3
BOMB_H        = 7
MAX_BOMBS     = 3
ALIEN_ROWS    = 5
ALIEN_COLS    = 11
ALIEN_SPC_X   = 16
ALIEN_SPC_Y   = 16
ALIEN_START_X = 26
ALIEN_START_Y = 64
ALIEN_STEP_X  = 2
ALIEN_DROP_Y  = 8
ALIEN_SCORES  = [30, 20, 20, 10, 10]    # per row (top→bottom)
ALIEN_WIDTHS  = [8, 11, 11, 12, 12]     # per row (type 0,1,1,2,2)
STARTING_LIVES = 3
MYSTERY_SPEED  = 1
MYSTERY_W      = 16
MYSTERY_SCORE  = 150   # simplified fixed value (real game varies)

# Fire interval formula (ms → frames):  max(18, 72 - (55-alive)*1.2)
FIRE_BASE  = round(1200 / FT)   # 72 frames
FIRE_MIN   = round(300  / FT)   # 18 frames
FIRE_SLOPE = 20 / FT            # per alien removed

MYSTERY_SPAWN = round(25000 / FT)  # 1500 frames

OBS_DIM = 20
ACT_DIM = 6


def _step_interval(alive: int) -> int:
    """Frames between formation marching steps (matches getStepInterval)."""
    ms = (16 if alive <= 1 else 33 if alive <= 2 else 66 if alive <= 5 else
          100 if alive <= 10 else 150 if alive <= 20 else 250 if alive <= 30 else
          350 if alive <= 40 else 500)
    return max(1, round(ms / FT))


def _rects_overlap(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class SpaceInvadersEnv:
    def __init__(self):
        self.reset()

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self):
        self.wave  = 0
        self.lives = STARTING_LIVES
        self.score = 0
        self._start_wave()
        return self._obs()

    def _start_wave(self):
        wave_offset = min(self.wave * 8, 48)          # WAVE_Y_OFFSET * wave
        # Formation
        self.fx = float(ALIEN_START_X)                # formation x offset
        self.fy = float(ALIEN_START_Y + wave_offset)  # formation y offset
        self.fdir = 1                                  # +1=right, -1=left
        # alive[row][col] boolean grid
        self.alive = [[True] * ALIEN_COLS for _ in range(ALIEN_ROWS)]
        self.n_alive = ALIEN_ROWS * ALIEN_COLS         # 55

        self.step_timer = _step_interval(self.n_alive)

        # Player
        self.px = float(W // 2 - PLAYER_W // 2)      # ~105.5
        self.player_alive = True

        # Bullet (player's)
        self.bx = 0.0; self.by = 0.0
        self.bullet_alive = False

        # Bombs (alien)
        self.bombs = []   # list of [x, y]

        # Alien fire timer
        self.fire_timer = FIRE_BASE

        # Mystery ship
        self.mystery_alive = False
        self.mystery_x = 0.0
        self.mystery_dir = 0
        self.mystery_timer = MYSTERY_SPAWN

        self.frames = 0

    # ── Observation ──────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        pcx = self.px + PLAYER_W / 2

        # Formation bounds
        left = right = bottom = -1.0
        cx_sum = 0.0
        if self.n_alive > 0:
            left = W; right = 0; bottom = 0
            for r in range(ALIEN_ROWS):
                for c in range(ALIEN_COLS):
                    if not self.alive[r][c]: continue
                    ax = self.fx + c * ALIEN_SPC_X
                    ay = self.fy + r * ALIEN_SPC_Y
                    aw = ALIEN_WIDTHS[r]
                    if ax < left:     left   = ax
                    if ax+aw > right: right  = ax + aw
                    if ay+8 > bottom: bottom = ay + 8
                    cx_sum += ax + aw / 2
            fcx = cx_sum / self.n_alive
        else:
            left = right = fcx = W / 2; bottom = 0.0

        # Nearest alien column (aiming hint)
        nearest_dx = 0.0
        best_d = W
        for c in range(ALIEN_COLS):
            for r in range(ALIEN_ROWS - 1, -1, -1):
                if self.alive[r][c]:
                    ax = self.fx + c * ALIEN_SPC_X + ALIEN_WIDTHS[r] / 2
                    d = abs(ax - pcx)
                    if d < best_d:
                        best_d = d
                        nearest_dx = ax - pcx
                    break

        # Bombs sorted by y desc (most dangerous = closest to player)
        bs = sorted(self.bombs, key=lambda b: -b[1])[:3]

        obs = [
            pcx / W,
            float(self.bullet_alive),
            self.bx / W if self.bullet_alive else 0.0,
            self.by / H if self.bullet_alive else 0.0,
            self.n_alive / 55.0,
            left  / W,
            right / W,
            bottom / H,
            fcx / W,
            float(self.fdir > 0),
            # bombs
            (bs[0][0] - pcx) / W if len(bs) > 0 else 0.0,
            bs[0][1] / H          if len(bs) > 0 else 0.0,
            (bs[1][0] - pcx) / W if len(bs) > 1 else 0.0,
            bs[1][1] / H          if len(bs) > 1 else 0.0,
            (bs[2][0] - pcx) / W if len(bs) > 2 else 0.0,
            bs[2][1] / H          if len(bs) > 2 else 0.0,
            nearest_dx / W,
            min(self.wave, 10) / 10.0,
            self.lives / float(STARTING_LIVES),
            self.mystery_x / W if self.mystery_alive else 0.0,
        ]
        return np.array(obs, dtype=np.float32)

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        do_left  = action in (1, 4)
        do_right = action in (2, 5)
        do_fire  = action >= 3
        reward   = 0.0

        # ── Player movement ──────────────────────────────────────────────
        if self.player_alive:
            if do_left:  self.px = max(0, self.px - PLAYER_SPEED)
            if do_right: self.px = min(W - PLAYER_W, self.px + PLAYER_SPEED)

            # ── Fire ─────────────────────────────────────────────────────
            if do_fire and not self.bullet_alive:
                self.bx = self.px + PLAYER_W // 2
                self.by = float(PLAYER_Y - BULLET_H)
                self.bullet_alive = True

        # ── Update bullet ────────────────────────────────────────────────
        if self.bullet_alive:
            self.by -= BULLET_SPEED
            if self.by + BULLET_H < 0:
                self.bullet_alive = False

        # ── Formation step ───────────────────────────────────────────────
        self.step_timer -= 1
        if self.step_timer <= 0:
            self._formation_step()
            self.step_timer = _step_interval(self.n_alive)

        # ── Alien fire ───────────────────────────────────────────────────
        self.fire_timer -= 1
        if self.fire_timer <= 0 and len(self.bombs) < MAX_BOMBS:
            self._alien_fire()
            interval = max(FIRE_MIN, FIRE_BASE - round((55 - self.n_alive) * FIRE_SLOPE))
            self.fire_timer = interval + np.random.randint(-10, 11)

        # ── Update bombs ─────────────────────────────────────────────────
        new_bombs = []
        for bom in self.bombs:
            bom[1] += BOMB_SPEED
            if bom[1] < H - 16:
                new_bombs.append(bom)
        self.bombs = new_bombs

        # ── Mystery ship ─────────────────────────────────────────────────
        self.mystery_timer -= 1
        if self.mystery_timer <= 0 and not self.mystery_alive and self.n_alive >= 8:
            self.mystery_alive = True
            self.mystery_dir = np.random.choice([-1, 1])
            self.mystery_x = 0.0 if self.mystery_dir > 0 else float(W)
            self.mystery_timer = MYSTERY_SPAWN
        if self.mystery_alive:
            self.mystery_x += self.mystery_dir * MYSTERY_SPEED
            if (self.mystery_dir > 0 and self.mystery_x > W + MYSTERY_W) or \
               (self.mystery_dir < 0 and self.mystery_x < -MYSTERY_W):
                self.mystery_alive = False

        # ── Collision: bullet ↔ aliens ───────────────────────────────────
        if self.bullet_alive:
            hit = False
            # Mystery ship
            if self.mystery_alive:
                if _rects_overlap(self.bx, self.by, BULLET_W, BULLET_H,
                                  self.mystery_x, 26, MYSTERY_W, 7):
                    self.mystery_alive = False
                    self.bullet_alive  = False
                    self.score += MYSTERY_SCORE
                    reward += MYSTERY_SCORE / 30.0
                    hit = True
            # Alien grid
            if not hit:
                for r in range(ALIEN_ROWS):
                    if hit: break
                    for c in range(ALIEN_COLS):
                        if not self.alive[r][c]: continue
                        ax = self.fx + c * ALIEN_SPC_X
                        ay = self.fy + r * ALIEN_SPC_Y
                        aw = ALIEN_WIDTHS[r]
                        if _rects_overlap(self.bx, self.by, BULLET_W, BULLET_H,
                                          ax, ay, aw, 8):
                            self.alive[r][c] = False
                            self.n_alive -= 1
                            self.bullet_alive = False
                            pts = ALIEN_SCORES[r]
                            self.score += pts
                            reward += pts / 10.0    # scale: top alien=+3, bottom=+1
                            hit = True
                            break

        # ── Collision: bombs ↔ player ─────────────────────────────────────
        if self.player_alive:
            px_rect = (self.px, PLAYER_Y, PLAYER_W, PLAYER_H)
            hit_bombs = []
            for bom in self.bombs:
                if _rects_overlap(bom[0], bom[1], BOMB_W, BOMB_H, *px_rect):
                    hit_bombs.append(bom)
            if hit_bombs:
                for b in hit_bombs:
                    if b in self.bombs: self.bombs.remove(b)
                self.lives -= 1
                reward -= 2.0
                if self.lives <= 0:
                    self.player_alive = False

        # ── Aliens reach bottom ──────────────────────────────────────────
        if self.n_alive > 0:
            bottom_y = max(self.fy + r * ALIEN_SPC_Y + 8
                           for r in range(ALIEN_ROWS)
                           for c in range(ALIEN_COLS)
                           if self.alive[r][c])
            if bottom_y >= PLAYER_Y:
                self.player_alive = False
                self.lives = 0
                reward -= 5.0

        # ── Wave clear ───────────────────────────────────────────────────
        wave_clear = (self.n_alive == 0)
        if wave_clear:
            reward += 10.0
            self.wave += 1
            self._start_wave()

        # ── Survival micro-bonus ─────────────────────────────────────────
        reward += 0.003

        # ── Done ─────────────────────────────────────────────────────────
        done = not self.player_alive and self.lives <= 0
        if done:
            reward -= 5.0

        self.frames += 1
        return self._obs(), float(reward), done, {'score': self.score, 'wave': self.wave}

    # ── Formation step (mirrors AlienFormation.step()) ───────────────────────
    def _formation_step(self):
        # Bounds
        left = W; right = 0
        for r in range(ALIEN_ROWS):
            for c in range(ALIEN_COLS):
                if not self.alive[r][c]: continue
                ax = self.fx + c * ALIEN_SPC_X
                aw = ALIEN_WIDTHS[r]
                if ax < left:     left  = ax
                if ax + aw > right: right = ax + aw

        if self.fdir > 0 and right + ALIEN_STEP_X >= W:
            self.fy += ALIEN_DROP_Y
            self.fdir = -1
        elif self.fdir < 0 and left - ALIEN_STEP_X <= 0:
            self.fy += ALIEN_DROP_Y
            self.fdir = 1
        else:
            self.fx += ALIEN_STEP_X * self.fdir

    # ── Alien fire (mirrors updateAlienFiring) ───────────────────────────────
    def _alien_fire(self):
        candidates = []
        for c in range(ALIEN_COLS):
            for r in range(ALIEN_ROWS - 1, -1, -1):
                if self.alive[r][c]:
                    ax = self.fx + c * ALIEN_SPC_X
                    aw = ALIEN_WIDTHS[r]
                    ay = self.fy + r * ALIEN_SPC_Y
                    candidates.append((ax + aw // 2, ay + 8))
                    break
        if candidates:
            bx, by = candidates[np.random.randint(len(candidates))]
            self.bombs.append([float(bx), float(by)])
