import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


OBS_DIM = 20
ACT_DIM = 6


LOGICAL_WIDTH = 224
LOGICAL_HEIGHT = 256

PLAYER_SPEED = 1
PLAYER_Y = 216
PLAYER_WIDTH = 13

BULLET_SPEED = 4
BULLET_WIDTH = 1
BULLET_HEIGHT = 4

ALIEN_ROWS = 5
ALIEN_COLS = 11
ALIEN_SPACING_X = 16
ALIEN_SPACING_Y = 16
ALIEN_START_X = 26
ALIEN_START_Y = 64
ALIEN_STEP_X = 2
ALIEN_DROP_Y = 8
ALIEN_SCORES = [30, 20, 20, 10, 10]

BOMB_SPEED = 1
MAX_ALIEN_BOMBS = 3
ALIEN_FIRE_BASE_INTERVAL = 1200.0
ALIEN_FIRE_MIN_INTERVAL = 300.0

MYSTERY_Y = 26
MYSTERY_SPEED = 1
MYSTERY_SPAWN_INTERVAL = 25000.0
MYSTERY_MIN_ALIENS = 8
MYSTERY_SCORE_TABLE = [100, 50, 50, 100, 150, 100, 100, 50, 300, 100, 100, 100, 50, 150, 100, 50]

STARTING_LIVES = 3
EXTRA_LIFE_SCORE = 1500
WAVE_Y_OFFSET = 8
MAX_WAVE_OFFSET = 48
FRAME_TIME_MS = 1000.0 / 60.0


def _rects_overlap(ax: float, ay: float, aw: float, ah: float,
                   bx: float, by: float, bw: float, bh: float) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def decode_action(action: int) -> Tuple[bool, bool, bool]:
    action = int(action)
    left = action in (1, 4)
    right = action in (2, 5)
    fire = action >= 3
    return left, right, fire


@dataclass
class Bullet:
    x: float
    y: float
    alive: bool = True

    def update(self) -> None:
        self.y -= BULLET_SPEED
        if self.y < 0:
            self.alive = False


@dataclass
class Bomb:
    x: float
    y: float
    alive: bool = True

    def update(self) -> None:
        self.y += BOMB_SPEED
        if self.y >= LOGICAL_HEIGHT - 16:
            self.alive = False


@dataclass
class MysteryShip:
    direction: int
    alive: bool = True

    def __post_init__(self) -> None:
        self.x = -16.0 if self.direction > 0 else float(LOGICAL_WIDTH)
        self.y = float(MYSTERY_Y)
        self.width = 16

    def update(self) -> None:
        self.x += self.direction * MYSTERY_SPEED
        if (self.direction > 0 and self.x > LOGICAL_WIDTH + 16) or (
            self.direction < 0 and self.x < -16
        ):
            self.alive = False


class Formation:
    def __init__(self, wave: int):
        wave_offset = min(wave * WAVE_Y_OFFSET, MAX_WAVE_OFFSET)
        self.x = float(ALIEN_START_X)
        self.y = float(ALIEN_START_Y + wave_offset)
        self.direction = 1
        self.alive = np.ones((ALIEN_ROWS, ALIEN_COLS), dtype=np.bool_)
        self.step_timer = self.get_step_interval()

    def count_alive(self) -> int:
        return int(self.alive.sum())

    def get_alien_width(self, row: int) -> int:
        return 8 if row == 0 else 11 if row <= 2 else 12

    def get_alien_pos(self, row: int, col: int) -> Tuple[float, float]:
        return self.x + col * ALIEN_SPACING_X, self.y + row * ALIEN_SPACING_Y

    def get_bounds(self) -> Tuple[float, float, float, float]:
        left, right, top, bottom = LOGICAL_WIDTH, 0.0, LOGICAL_HEIGHT, 0.0
        any_alive = False
        for r in range(ALIEN_ROWS):
            for c in range(ALIEN_COLS):
                if not self.alive[r, c]:
                    continue
                any_alive = True
                x, y = self.get_alien_pos(r, c)
                w = self.get_alien_width(r)
                left = min(left, x)
                right = max(right, x + w)
                top = min(top, y)
                bottom = max(bottom, y + 8)
        if not any_alive:
            return 0.0, 0.0, 0.0, 0.0
        return left, right, top, bottom

    def get_step_interval(self) -> float:
        alive = self.count_alive()
        if alive <= 1:
            return 16.0
        if alive <= 2:
            return 33.0
        if alive <= 5:
            return 66.0
        if alive <= 10:
            return 100.0
        if alive <= 20:
            return 150.0
        if alive <= 30:
            return 250.0
        if alive <= 40:
            return 350.0
        return 500.0

    def step_once(self) -> None:
        left, right, _, _ = self.get_bounds()
        if self.direction > 0 and right + ALIEN_STEP_X >= LOGICAL_WIDTH:
            self.y += ALIEN_DROP_Y
            self.direction = -1
        elif self.direction < 0 and left - ALIEN_STEP_X <= 0:
            self.y += ALIEN_DROP_Y
            self.direction = 1
        else:
            self.x += ALIEN_STEP_X * self.direction

    def update(self, dt_ms: float) -> None:
        self.step_timer -= dt_ms
        if self.step_timer <= 0:
            self.step_once()
            self.step_timer = self.get_step_interval()

    def shooter_candidates(self) -> List[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = []
        for c in range(ALIEN_COLS):
            for r in range(ALIEN_ROWS - 1, -1, -1):
                if self.alive[r, c]:
                    candidates.append((r, c))
                    break
        return candidates

    def reached_bottom(self) -> bool:
        _, _, _, bottom = self.get_bounds()
        return bottom >= PLAYER_Y


class SpaceInvadersEnv:
    """Feature-observation Space Invaders environment for browser deployment.

    One environment step repeats the chosen action for ``frame_skip`` game
    frames. The default frame skip is 4, matching the Rainbow Atari training
    protocol where hyperparameters are reported in frames but optimization is
    scheduled every four frames.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        frame_skip: int = 4,
        max_episode_frames: int = 108_000,
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.frame_skip = int(frame_skip)
        self.max_episode_frames = int(max_episode_frames)
        self.reset()

    def reset(self) -> np.ndarray:
        self.player_x = float(LOGICAL_WIDTH // 2 - PLAYER_WIDTH // 2)
        self.player_bullet: Optional[Bullet] = None
        self.bombs: List[Bomb] = []
        self.mystery_ship: Optional[MysteryShip] = None
        self.score = 0
        self.lives = STARTING_LIVES
        self.wave = 0
        self.shot_count = 0
        self.frame = 0
        self.extra_life_awarded = False
        self.mystery_timer = MYSTERY_SPAWN_INTERVAL
        self.alien_fire_timer = ALIEN_FIRE_BASE_INTERVAL
        self.formation = Formation(self.wave)
        return self._obs()

    def _player_center_x(self) -> float:
        return self.player_x + PLAYER_WIDTH / 2

    def _add_score(self, points: int) -> None:
        self.score += int(points)
        if not self.extra_life_awarded and self.score >= EXTRA_LIFE_SCORE:
            self.extra_life_awarded = True
            self.lives += 1

    def _start_next_wave(self) -> None:
        self.wave += 1
        self.formation = Formation(self.wave)
        self.player_bullet = None
        self.bombs.clear()
        self.mystery_ship = None
        self.mystery_timer = MYSTERY_SPAWN_INTERVAL
        self.alien_fire_timer = ALIEN_FIRE_BASE_INTERVAL

    def _update_alien_firing(self, dt_ms: float) -> None:
        self.alien_fire_timer -= dt_ms
        if self.alien_fire_timer > 0 or len(self.bombs) >= MAX_ALIEN_BOMBS:
            return
        candidates = self.formation.shooter_candidates()
        if candidates:
            r, c = self.rng.choice(candidates)
            x, y = self.formation.get_alien_pos(r, c)
            w = self.formation.get_alien_width(r)
            self.bombs.append(Bomb(x + w // 2, y + 8))
        alive = self.formation.count_alive()
        ratio = alive / float(ALIEN_ROWS * ALIEN_COLS)
        self.alien_fire_timer = ALIEN_FIRE_MIN_INTERVAL + (
            ALIEN_FIRE_BASE_INTERVAL - ALIEN_FIRE_MIN_INTERVAL
        ) * ratio

    def _update_mystery(self, dt_ms: float) -> None:
        if self.mystery_ship is not None:
            self.mystery_ship.update()
            if not self.mystery_ship.alive:
                self.mystery_ship = None
            return
        self.mystery_timer -= dt_ms
        if self.mystery_timer <= 0 and self.formation.count_alive() >= MYSTERY_MIN_ALIENS:
            self.mystery_ship = MysteryShip(1 if self.rng.random() < 0.5 else -1)
            self.mystery_timer = MYSTERY_SPAWN_INTERVAL

    def _check_bullet_collisions(self) -> float:
        if self.player_bullet is None or not self.player_bullet.alive:
            return 0.0

        bullet = self.player_bullet
        for r in range(ALIEN_ROWS):
            for c in range(ALIEN_COLS):
                if not self.formation.alive[r, c]:
                    continue
                ax, ay = self.formation.get_alien_pos(r, c)
                aw = self.formation.get_alien_width(r)
                if _rects_overlap(bullet.x, bullet.y, BULLET_WIDTH, BULLET_HEIGHT, ax, ay, aw, 8):
                    self.formation.alive[r, c] = False
                    bullet.alive = False
                    points = ALIEN_SCORES[r]
                    self._add_score(points)
                    return float(points)

        if self.mystery_ship is not None and self.mystery_ship.alive:
            m = self.mystery_ship
            if _rects_overlap(bullet.x, bullet.y, BULLET_WIDTH, BULLET_HEIGHT, m.x, m.y, m.width, 7):
                bullet.alive = False
                points = MYSTERY_SCORE_TABLE[(self.shot_count - 1) % len(MYSTERY_SCORE_TABLE)]
                self._add_score(points)
                self.mystery_ship = None
                return float(points)
        return 0.0

    def _check_bomb_collisions(self) -> Tuple[float, bool]:
        reward = 0.0
        life_lost = False
        px, py, pw, ph = self.player_x, PLAYER_Y, PLAYER_WIDTH, 8
        for bomb in self.bombs:
            if not bomb.alive:
                continue
            if _rects_overlap(bomb.x, bomb.y, 3, 7, px, py, pw, ph):
                bomb.alive = False
                self.lives -= 1
                reward -= 100.0
                life_lost = True
                break
        self.bombs = [b for b in self.bombs if b.alive]
        if life_lost:
            self.player_bullet = None
            self.bombs.clear()
            self.mystery_ship = None
            if self.lives > 0:
                self.player_x = float(LOGICAL_WIDTH // 2 - PLAYER_WIDTH // 2)
        return reward, life_lost

    def _obs(self) -> np.ndarray:
        p_center = self._player_center_x()
        bullet_alive = self.player_bullet is not None and self.player_bullet.alive

        left, right, _, bottom = self.formation.get_bounds()
        n_alive = self.formation.count_alive()
        if n_alive:
            cx_sum = 0.0
            for r in range(ALIEN_ROWS):
                for c in range(ALIEN_COLS):
                    if not self.formation.alive[r, c]:
                        continue
                    ax, _ = self.formation.get_alien_pos(r, c)
                    cx_sum += ax + self.formation.get_alien_width(r) / 2
            formation_cx = cx_sum / n_alive
        else:
            formation_cx = 0.0

        nearest_dx, nearest_d = 0.0, float(LOGICAL_WIDTH)
        for c in range(ALIEN_COLS):
            for r in range(ALIEN_ROWS - 1, -1, -1):
                if not self.formation.alive[r, c]:
                    continue
                ax, _ = self.formation.get_alien_pos(r, c)
                acx = ax + self.formation.get_alien_width(r) / 2
                d = abs(acx - p_center)
                if d < nearest_d:
                    nearest_d = d
                    nearest_dx = acx - p_center
                break

        bombs = sorted([b for b in self.bombs if b.alive], key=lambda b: b.y, reverse=True)[:3]
        bomb_features: List[float] = []
        for i in range(3):
            if i < len(bombs):
                bomb_features.extend([(bombs[i].x - p_center) / LOGICAL_WIDTH, bombs[i].y / LOGICAL_HEIGHT])
            else:
                bomb_features.extend([0.0, 0.0])

        obs = [
            p_center / LOGICAL_WIDTH,
            1.0 if bullet_alive else 0.0,
            self.player_bullet.x / LOGICAL_WIDTH if bullet_alive else 0.0,
            self.player_bullet.y / LOGICAL_HEIGHT if bullet_alive else 0.0,
            n_alive / float(ALIEN_ROWS * ALIEN_COLS),
            left / LOGICAL_WIDTH,
            right / LOGICAL_WIDTH,
            bottom / LOGICAL_HEIGHT,
            formation_cx / LOGICAL_WIDTH,
            1.0 if self.formation.direction > 0 else 0.0,
            *bomb_features,
            nearest_dx / LOGICAL_WIDTH,
            min(self.wave, 10) / 10.0,
            self.lives / float(STARTING_LIVES),
            self.mystery_ship.x / LOGICAL_WIDTH if self.mystery_ship is not None and self.mystery_ship.alive else 0.0,
        ]
        return np.asarray(obs, dtype=np.float32)

    def _game_frame(self, action: int) -> Tuple[float, bool, dict]:
        left, right, fire = decode_action(action)
        if left and not right:
            self.player_x = max(0.0, self.player_x - PLAYER_SPEED)
        elif right and not left:
            self.player_x = min(float(LOGICAL_WIDTH - PLAYER_WIDTH), self.player_x + PLAYER_SPEED)

        if fire and (self.player_bullet is None or not self.player_bullet.alive):
            self.player_bullet = Bullet(
                self.player_x + PLAYER_WIDTH // 2,
                PLAYER_Y - BULLET_HEIGHT,
            )
            self.shot_count += 1

        reward = 0.0
        done = False
        info = {"score": self.score, "lives": self.lives, "wave": self.wave}

        if self.player_bullet is not None and self.player_bullet.alive:
            self.player_bullet.update()
        if self.player_bullet is not None and not self.player_bullet.alive:
            self.player_bullet = None

        self.formation.update(FRAME_TIME_MS)
        self._update_alien_firing(FRAME_TIME_MS)

        for bomb in self.bombs:
            bomb.update()
        self.bombs = [b for b in self.bombs if b.alive]

        self._update_mystery(FRAME_TIME_MS)

        reward += self._check_bullet_collisions()
        death_reward, life_lost = self._check_bomb_collisions()
        reward += death_reward

        if self.formation.count_alive() == 0:
            reward += 50.0
            self._start_next_wave()

        if self.formation.reached_bottom():
            reward -= 100.0
            self.lives = 0
            done = True

        self.frame += 1
        if self.lives <= 0:
            done = True
        if self.frame >= self.max_episode_frames:
            done = True

        info.update({"score": self.score, "lives": self.lives, "wave": self.wave, "life_lost": life_lost})
        return reward, done, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            reward, done, info = self._game_frame(action)
            total_reward += reward
            if done:
                break
        return self._obs(), float(total_reward), done, info


if __name__ == "__main__":
    env = SpaceInvadersEnv(seed=123)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    for _ in range(10):
        obs, reward, done, info = env.step(env.rng.randrange(ACT_DIM))
        assert obs.shape == (OBS_DIM,)
        if done:
            obs = env.reset()
    print("SpaceInvadersEnv smoke test passed")
