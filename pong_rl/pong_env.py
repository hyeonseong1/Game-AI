import numpy as np


class PongEnv:
    PADDLE_W = 12
    PADDLE_H = 80
    BALL_R = 8
    PADDLE_SPEED = 7
    LEFT_X = 20
    RIGHT_X_OFFSET = 30

    def __init__(self, width=900, height=600):
        self.W = width
        self.H = height
        self.right_x = self.W - self.RIGHT_X_OFFSET
        self.reset()

    def reset(self):
        self.ball = np.array([self.W / 2, self.H / 2], dtype=float)
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        direction = np.random.choice([-1, 1])
        speed = 6.5
        self.ball_vel = np.array([
            direction * speed * np.cos(angle),
            speed * np.sin(angle),
        ])
        self.left_y = float(self.H / 2)
        self.right_y = float(self.H / 2)
        return self._obs()

    def _obs(self):
        return np.array([
            self.ball[0] / self.W * 2 - 1,
            self.ball[1] / self.H * 2 - 1,
            self.ball_vel[0] / 18,
            self.ball_vel[1] / 18,
            self.right_y / self.H * 2 - 1,   # AI (right) paddle
            self.left_y / self.H * 2 - 1,    # opponent (left) paddle
        ], dtype=np.float32)

    def _clamp(self, y):
        return float(np.clip(y, self.PADDLE_H / 2, self.H - self.PADDLE_H / 2))

    def step(self, right_action: int, left_action: int = 0):
        """
        right_action: controls right paddle (AI / DQN target)
        left_action:  controls left paddle (human or training bot)
        Actions: 0=stay, 1=up, 2=down
        Returns: obs, reward, done, info
          info['scorer'] = 'ai' | 'player'  when done
        """
        if right_action == 1:
            self.right_y -= self.PADDLE_SPEED
        elif right_action == 2:
            self.right_y += self.PADDLE_SPEED
        self.right_y = self._clamp(self.right_y)

        if left_action == 1:
            self.left_y -= self.PADDLE_SPEED
        elif left_action == 2:
            self.left_y += self.PADDLE_SPEED
        self.left_y = self._clamp(self.left_y)

        self.ball += self.ball_vel

        reward = 0.0
        done = False
        info = {}

        # Top / bottom walls
        if self.ball[1] - self.BALL_R <= 0:
            self.ball[1] = self.BALL_R
            self.ball_vel[1] = abs(self.ball_vel[1])
        elif self.ball[1] + self.BALL_R >= self.H:
            self.ball[1] = self.H - self.BALL_R
            self.ball_vel[1] = -abs(self.ball_vel[1])

        # Right paddle collision (AI)
        if (self.ball_vel[0] > 0
                and self.ball[0] + self.BALL_R >= self.right_x
                and self.ball[0] - self.BALL_R <= self.right_x + self.PADDLE_W
                and abs(self.ball[1] - self.right_y) < self.PADDLE_H / 2 + self.BALL_R):
            self.ball[0] = self.right_x - self.BALL_R - 1
            self.ball_vel[0] = -abs(self.ball_vel[0]) * 1.04
            offset = (self.ball[1] - self.right_y) / (self.PADDLE_H / 2)
            self.ball_vel[1] = float(np.clip(offset * 9, -13, 13))
            reward = 1.0

        # Left paddle collision (human / bot)
        if (self.ball_vel[0] < 0
                and self.ball[0] - self.BALL_R <= self.LEFT_X + self.PADDLE_W
                and self.ball[0] + self.BALL_R >= self.LEFT_X
                and abs(self.ball[1] - self.left_y) < self.PADDLE_H / 2 + self.BALL_R):
            self.ball[0] = self.LEFT_X + self.PADDLE_W + self.BALL_R + 1
            self.ball_vel[0] = abs(self.ball_vel[0]) * 1.04
            offset = (self.ball[1] - self.left_y) / (self.PADDLE_H / 2)
            self.ball_vel[1] = float(np.clip(offset * 9, -13, 13))

        # Cap speed
        speed = float(np.linalg.norm(self.ball_vel))
        if speed > 18:
            self.ball_vel = self.ball_vel / speed * 18

        # Scoring
        if self.ball[0] + self.BALL_R < 0:
            # Left player missed → AI (right) scores
            reward = 5.0
            done = True
            info['scorer'] = 'ai'
        elif self.ball[0] - self.BALL_R > self.W:
            # AI (right) missed → left player scores
            reward = -5.0
            done = True
            info['scorer'] = 'player'

        return self._obs(), reward, done, info
