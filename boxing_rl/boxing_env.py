"""
Boxing RL environment — mirrors boxing.js physics exactly at 60 fps.

Agent  = right boxer (AI side, starts at x=400)
Opponent = left boxer (rule-based, starts at x=160)

Observation (14 floats, all in [-1,1] or [0,1]):
  0  agent x          (RING centre normalised)
  1  agent y
  2  opp x
  3  opp y
  4  dx  (opp - agent) / RING_W
  5  dy  (opp - agent) / RING_H
  6  distance / MAX_DIST
  7  agent score / MAX_SCORE
  8  opp score   / MAX_SCORE
  9  agent punch_timer / PUNCH_FPS   (0 = can punch)
  10 opp   punch_timer / PUNCH_FPS
  11 agent knock_timer / KNOCK_FPS   (0 = standing)
  12 opp   knock_timer / KNOCK_FPS
  13 time_left / GAME_FPS

Actions (18):
  0-8  : move only  (0=stay 1=L 2=R 3=U 4=D 5=LU 6=LD 7=RU 8=RD)
  9-17 : same move  + punch
"""

import numpy as np

# ── Constants matching boxing.js ────────────────────────────────────────────
RING_X, RING_Y = 40, 40
RING_W, RING_H = 480, 320          # W-80, H-80
PLAYER_R       = 16
PUNCH_RANGE    = 44
HIT_RANGE      = PUNCH_RANGE + PLAYER_R   # 60 px  (JS: d > PUNCH_RANGE + PLAYER_R fails)
PUNCH_FPS      = 13                        # 220 ms @ 60 fps
KNOCK_FPS      = 108                       # 1800 ms @ 60 fps
MAX_SCORE      = 100
GAME_FPS       = 7_200                     # 120 s × 60 fps
SPEED          = 2.2                       # px per frame

X_MIN = RING_X + PLAYER_R                 # 56
X_MAX = RING_X + RING_W - PLAYER_R        # 504
Y_MIN = RING_Y + PLAYER_R                 # 56
Y_MAX = RING_Y + RING_H - PLAYER_R        # 344

CX, CY   = RING_X + RING_W / 2, RING_Y + RING_H / 2   # 280, 200
RW2, RH2 = RING_W / 2, RING_H / 2                       # 240, 160
MAX_DIST = float(np.hypot(RING_W, RING_H))               # 576.9

OBS_DIM = 14
ACT_DIM = 18

# 9 movement deltas (stay + 8 directions)
_DIAG = float(np.sqrt(0.5))
MOVE_DELTA = np.array([
    [ 0,      0    ],   # 0 stay
    [-1,      0    ],   # 1 L
    [+1,      0    ],   # 2 R
    [ 0,     -1    ],   # 3 U
    [ 0,     +1    ],   # 4 D
    [-_DIAG, -_DIAG],   # 5 LU
    [-_DIAG, +_DIAG],   # 6 LD
    [+_DIAG, -_DIAG],   # 7 RU
    [+_DIAG, +_DIAG],   # 8 RD
], dtype=np.float32)

# Opponent presets (match JS AI_CFG + knockdown probs)
OPP_PRESETS = {
    'easy':   dict(speed=1.2, react=54, punch_p=0.012, noise=0.50, knock_p=0.02),
    'medium': dict(speed=1.8, react=30, punch_p=0.022, noise=0.25, knock_p=0.05),
    'hard':   dict(speed=2.4, react=11, punch_p=0.038, noise=0.05, knock_p=0.09),
}


class BoxingEnv:
    def __init__(self, opponent: str = 'medium'):
        self._cfg = OPP_PRESETS.get(opponent, OPP_PRESETS['medium'])
        self.reset()

    # ── Properties ──────────────────────────────────────────────────────────
    @property
    def a_punching(self): return self.a_pt > 0
    @property
    def a_knocked(self):  return self.a_kt > 0
    @property
    def p_punching(self): return self.p_pt > 0
    @property
    def p_knocked(self):  return self.p_kt > 0

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self):
        # Agent (right boxer)
        self.ax, self.ay = RING_X + RING_W * .75, RING_Y + RING_H * .5  # 400, 200
        self.a_score = 0
        self.a_pt = 0   # punch timer
        self.a_kt = 0   # knock timer

        # Opponent (left boxer)
        self.px, self.py = RING_X + RING_W * .25, RING_Y + RING_H * .5  # 160, 200
        self.p_score = 0
        self.p_pt = 0
        self.p_kt = 0
        self._p_react = 0
        self._p_tx, self._p_ty = float(self.ax), float(self.ay)

        self.t = GAME_FPS
        return self._obs()

    # ── Observation ──────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        dx = self.px - self.ax
        dy = self.py - self.ay
        return np.array([
            (self.ax - CX) / RW2,
            (self.ay - CY) / RH2,
            (self.px - CX) / RW2,
            (self.py - CY) / RH2,
            dx / RING_W,
            dy / RING_H,
            float(np.hypot(dx, dy)) / MAX_DIST,
            self.a_score / MAX_SCORE,
            self.p_score / MAX_SCORE,
            self.a_pt / PUNCH_FPS,
            self.p_pt / PUNCH_FPS,
            self.a_kt / KNOCK_FPS,
            self.p_kt / KNOCK_FPS,
            self.t / GAME_FPS,
        ], dtype=np.float32)

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        reward = 0.0
        move_idx = int(action) % 9
        do_punch = int(action) >= 9

        # 1. Agent moves
        if not self.a_knocked:
            dvx, dvy = MOVE_DELTA[move_idx]
            self.ax = float(np.clip(self.ax + dvx * SPEED, X_MIN, X_MAX))
            self.ay = float(np.clip(self.ay + dvy * SPEED, Y_MIN, Y_MAX))

            # Agent punch
            if do_punch and not self.a_punching:
                self.a_pt = PUNCH_FPS
                d = float(np.hypot(self.ax - self.px, self.ay - self.py))
                if d <= HIT_RANGE and not self.p_knocked:
                    self.a_score = min(MAX_SCORE, self.a_score + 1)
                    reward += 1.0
                    if np.random.rand() < 0.04:
                        self.p_kt = KNOCK_FPS
                        reward += 5.0

        # 2. Opponent acts
        opp_hit = self._opp_step()
        if opp_hit:
            reward -= 1.0

        # 3. Tick timers
        self.a_pt = max(0, self.a_pt - 1)
        self.a_kt = max(0, self.a_kt - 1)
        self.p_pt = max(0, self.p_pt - 1)
        self.p_kt = max(0, self.p_kt - 1)
        self.t   -= 1

        # 4. Survival micro-bonus
        reward += 0.002

        # 5. Done
        done = (self.a_score >= MAX_SCORE or
                self.p_score >= MAX_SCORE or
                self.t <= 0)
        if done:
            reward += (self.a_score - self.p_score) * 0.1

        return self._obs(), float(reward), done, {
            'a_score': self.a_score,
            'p_score': self.p_score,
        }

    # ── Rule-based opponent ───────────────────────────────────────────────────
    def _opp_step(self) -> bool:
        """Returns True if opponent landed a punch on the agent."""
        cfg = self._cfg
        if self.p_knocked:
            return False

        # Update target periodically
        self._p_react -= 1
        if self._p_react <= 0:
            self._p_react = cfg['react']
            noise = RING_W * cfg['noise']
            self._p_tx = float(np.clip(self.ax + np.random.uniform(-noise, noise), X_MIN, X_MAX))
            self._p_ty = float(np.clip(self.ay + np.random.uniform(-noise, noise), Y_MIN, Y_MAX))

        dx = self._p_tx - self.px
        dy = self._p_ty - self.py
        d  = max(float(np.hypot(dx, dy)), 1.0)
        if d > HIT_RANGE * 0.7:
            self.px = float(np.clip(self.px + (dx / d) * cfg['speed'], X_MIN, X_MAX))
            self.py = float(np.clip(self.py + (dy / d) * cfg['speed'], Y_MIN, Y_MAX))

        # Punch
        if not self.p_punching and np.random.rand() < cfg['punch_p']:
            d2 = float(np.hypot(self.px - self.ax, self.py - self.ay))
            if d2 <= HIT_RANGE and not self.a_knocked:
                self.p_pt = PUNCH_FPS
                self.p_score = min(MAX_SCORE, self.p_score + 1)
                if np.random.rand() < cfg['knock_p']:
                    self.a_kt = KNOCK_FPS
                return True
        return False
