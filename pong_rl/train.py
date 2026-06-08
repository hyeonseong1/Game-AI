"""
SimbaV2 PPO Pong Training Script
================================
Usage:
    python train.py                         # default 10,000,000 steps
    python train.py --steps 10000000
    python train.py --no-self-play          # old fixed-bot training

Saves:
    models/mid_model.pt   — checkpoint at the halfway point
    models/best_model.pt  — checkpoint with best 50-episode avg reward
"""

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from pong_env import PongEnv
from ppo_agent import PPOAgent


# ------------------------------------------------------------------
# Rule opponents (control the left paddle).
# ------------------------------------------------------------------
RuleStyle = Literal['random', 'tracker', 'predictive', 'aggressive']


def _bounced_y(y: float, height: float) -> float:
    span = height * 2
    y = y % span
    return span - y if y > height else y


def _left_intercept_y(env: PongEnv) -> float:
    if env.ball_vel[0] >= -1e-6:
        return float(env.ball[1])
    target_x = env.LEFT_X + env.PADDLE_W + env.BALL_R
    frames = (env.ball[0] - target_x) / max(abs(env.ball_vel[0]), 1e-6)
    return _bounced_y(float(env.ball[1] + env.ball_vel[1] * frames), env.H)


def bot_action(env: PongEnv, style: RuleStyle = 'tracker') -> int:
    if style == 'random':
        if np.random.random() < 0.75:
            return 0
        return np.random.randint(3)

    if style == 'predictive':
        target_y = _left_intercept_y(env)
        mistake_rate = 0.025
        margin = 4.0
    elif style == 'aggressive':
        target_y = _left_intercept_y(env)
        # Aim away from paddle center to create steeper return angles.
        if env.ball_vel[0] < 0:
            target_y += np.random.choice([-1.0, 1.0]) * env.PADDLE_H * 0.22
        mistake_rate = 0.035
        margin = 3.0
    else:
        target_y = float(env.ball[1])
        mistake_rate = 0.05
        margin = 5.0

    if np.random.random() < mistake_rate:
        return np.random.randint(3)
    if target_y < env.left_y - margin:
        return 1   # up
    if target_y > env.left_y + margin:
        return 2   # down
    return 0


def mirror_obs_for_left(obs: np.ndarray) -> np.ndarray:
    """Convert the right-paddle observation into the left-paddle perspective."""
    return np.array([-obs[0], obs[1], -obs[2], obs[3], obs[5], obs[4]], dtype=np.float32)


@dataclass
class Opponent:
    kind: str
    name: str
    style: Optional[RuleStyle] = None
    policy: Optional[torch.nn.Module] = None


class PongOpponentPool:
    def __init__(
        self,
        max_snapshots: int = 8,
        rule_prob: float = 0.35,
        warmup_episodes: int = 25,
        epsilon: float = 0.015,
    ):
        self.max_snapshots = max_snapshots
        self.rule_prob = rule_prob
        self.warmup_episodes = warmup_episodes
        self.epsilon = epsilon
        self.rule_styles: tuple[RuleStyle, ...] = ('tracker', 'predictive', 'aggressive')
        self.snapshots: list[Opponent] = []

    def add_snapshot(self, agent: PPOAgent, step: int, avg_reward: Optional[float] = None):
        policy = copy.deepcopy(agent.policy).to(agent.device)
        policy.eval()
        policy.requires_grad_(False)
        suffix = f"avg{avg_reward:+.2f}" if avg_reward is not None and np.isfinite(avg_reward) else "snapshot"
        self.snapshots.append(Opponent(kind='policy', name=f"step{step}_{suffix}", policy=policy))
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def sample(self, episode: int) -> Opponent:
        use_rule = (
            not self.snapshots
            or episode < self.warmup_episodes
            or np.random.random() < self.rule_prob
        )
        if use_rule:
            style = np.random.choice(self.rule_styles)
            return Opponent(kind='rule', name=f"rule:{style}", style=style)
        return self.snapshots[np.random.randint(len(self.snapshots))]

    def act(self, opponent: Opponent, env: PongEnv, obs: np.ndarray) -> int:
        if opponent.kind == 'rule':
            return bot_action(env, opponent.style or 'tracker')
        if np.random.random() < self.epsilon:
            return int(np.random.randint(3))
        assert opponent.policy is not None
        left_obs = mirror_obs_for_left(obs)
        device = next(opponent.policy.parameters()).device
        with torch.no_grad():
            x = torch.as_tensor(left_obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = opponent.policy(x)[0]
            return int(logits.argmax(-1).item())


def evaluate_agent(agent: PPOAgent, episodes: int = 24, max_frames: int = 2400) -> tuple[float, float]:
    if episodes <= 0:
        return float('nan'), float('nan')
    styles: tuple[RuleStyle, ...] = ('tracker', 'predictive', 'aggressive')
    rewards: list[float] = []
    wins = 0
    for ep in range(episodes):
        env = PongEnv()
        state = env.reset()
        ep_reward = 0.0
        info = {}
        for _ in range(max_frames):
            action = agent.predict(state)
            opp_action = bot_action(env, styles[ep % len(styles)])
            state, reward, done, info = env.step(action, opp_action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        wins += int(info.get('scorer') == 'ai')
    return float(np.mean(rewards)), wins / episodes


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def train(
    total_steps: int = 10_000_000,
    save_dir: str = 'models',
    rollout_size: int = 2048,
    self_play: bool = True,
    max_opponents: int = 8,
    self_play_warmup_episodes: int = 25,
    snapshot_interval_updates: int = 25,
    eval_interval_updates: int = 10,
    eval_episodes: int = 24,
    lr_decay: bool = True,
    min_lr_ratio: float = 0.05,
) -> PPOAgent:
    os.makedirs(save_dir, exist_ok=True)

    env = PongEnv()
    agent = PPOAgent(
        rollout_size=rollout_size,
        value_min=-30.0,
        value_max=30.0,
        total_training_steps=total_steps,
        lr_decay=lr_decay,
        min_lr_ratio=min_lr_ratio,
    )
    opponent_pool = PongOpponentPool(
        max_snapshots=max_opponents,
        warmup_episodes=self_play_warmup_episodes,
    )
    opponent = opponent_pool.sample(episode=0)

    print(f"\n{'='*60}")
    print(f"  SimbaV2 PPO Pong Training")
    print(f"  Device    : {agent.device}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Rollout sz : {agent.rollout_size}")
    print(f"  Opponents  : {'self-play pool' if self_play else 'fixed rule bot'}")
    print(f"  LR decay   : {'linear' if lr_decay else 'off'} "
          f"(base={agent.base_lr:.2e}, min={agent.base_lr * min_lr_ratio:.2e})")
    print(f"  Mid model  : step {total_steps//2:,}")
    print(f"{'='*60}\n")

    state = env.reset()
    ep_reward = 0.0
    ep_count = 0
    episode_rewards: list[float] = []
    episode_wins: list[float] = []

    best_eval = float('-inf')
    mid_saved = False

    t0 = time.time()

    for step in range(1, total_steps + 1):
        action, log_prob, value = agent.act(state)
        opp_action = (
            opponent_pool.act(opponent, env, state)
            if self_play
            else bot_action(env, 'tracker')
        )

        next_state, reward, done, info = env.step(action, opp_action)

        agent.push(state, action, log_prob, reward, value, done)
        ep_reward += reward
        state = next_state

        if done:
            episode_rewards.append(ep_reward)
            episode_wins.append(float(info.get('scorer') == 'ai'))
            ep_count += 1
            ep_reward = 0.0
            state = env.reset()
            opponent = opponent_pool.sample(episode=ep_count)

        # PPO update when buffer is full
        if agent.buffer.ready():
            loss = agent.update(state)
            avg_train = (float(np.mean(episode_rewards[-50:]))
                         if len(episode_rewards) >= 50
                         else float(np.mean(episode_rewards)) if episode_rewards
                         else 0.0)

            # ── mid model ──
            if not mid_saved and step >= total_steps // 2:
                agent.save(os.path.join(save_dir, 'mid_model.pt'))
                mid_saved = True
                print(f"  [SAVED] mid_model.pt  at step {step:,}  "
                      f"(ep {ep_count})")

            # ── opponent pool snapshots ──
            if self_play and snapshot_interval_updates > 0 and agent.updates % snapshot_interval_updates == 0:
                opponent_pool.add_snapshot(agent, step=step, avg_reward=avg_train)

            # ── best model: static rule evaluation keeps the moving self-play curriculum honest ──
            if eval_interval_updates > 0 and agent.updates % eval_interval_updates == 0:
                eval_avg, eval_win_rate = evaluate_agent(agent, episodes=eval_episodes)
                if np.isfinite(eval_avg) and eval_avg > best_eval:
                    best_eval = eval_avg
                    agent.save(os.path.join(save_dir, 'best_model.pt'))
                    if self_play:
                        opponent_pool.add_snapshot(agent, step=step, avg_reward=eval_avg)
                    print(f"  [BEST] eval_avg {eval_avg:+6.2f} | "
                          f"eval_win {100*eval_win_rate:5.1f}% | "
                          f"pool {len(opponent_pool.snapshots)}")

            # ── progress log every 10 updates ──
            if agent.updates % 10 == 0:
                win = (float(np.mean(episode_wins[-50:]))
                       if episode_wins else 0.0)
                elapsed = time.time() - t0
                pct = step / total_steps
                eta = (elapsed / pct - elapsed) if pct > 0 else 0
                print(f"  step {step:>8,}/{total_steps:,} "
                      f"({100*pct:4.1f}%) | "
                      f"ep {ep_count:5d} | "
                      f"avg50 {avg_train:+6.2f} | "
                      f"win50 {100*win:5.1f}% | "
                      f"pool {len(opponent_pool.snapshots):2d} | "
                      f"loss {loss:.4f} | "
                      f"lr {loss.lr:.2e} | "
                      f"ETA {eta/60:.1f}m")

    # ── final saves (safety nets) ──
    if not mid_saved:
        agent.save(os.path.join(save_dir, 'mid_model.pt'))
        print("  [SAVED] mid_model.pt  (end-of-training fallback)")

    if not os.path.exists(os.path.join(save_dir, 'best_model.pt')):
        agent.save(os.path.join(save_dir, 'best_model.pt'))
        print("  [SAVED] best_model.pt (end-of-training fallback)")

    print(f"\nTraining complete!  Best eval reward: {best_eval:.2f}")
    print(f"Models saved in '{save_dir}/'")
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10_000_000,
                        help='Total environment steps (default: 10,000,000)')
    parser.add_argument('--save-dir', default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--rollout-size', type=int, default=2048,
                        help='PPO rollout size before each update')
    parser.add_argument('--no-self-play', action='store_true',
                        help='Train only against the old fixed rule bot')
    parser.add_argument('--max-opponents', type=int, default=8,
                        help='Maximum frozen policy opponents in the self-play pool')
    parser.add_argument('--self-play-warmup-episodes', type=int, default=25,
                        help='Rule-only warmup episodes before sampling policy opponents')
    parser.add_argument('--snapshot-interval-updates', type=int, default=25,
                        help='Add the current policy to the opponent pool every N PPO updates')
    parser.add_argument('--eval-interval-updates', type=int, default=10,
                        help='Run static rule evaluation every N PPO updates')
    parser.add_argument('--eval-episodes', type=int, default=24,
                        help='Number of static rule evaluation episodes')
    parser.add_argument('--no-lr-decay', action='store_true',
                        help='Disable linear learning-rate decay')
    parser.add_argument('--min-lr-ratio', type=float, default=0.05,
                        help='Final LR as a fraction of the base LR')
    args = parser.parse_args()
    train(
        total_steps=args.steps,
        save_dir=args.save_dir,
        rollout_size=args.rollout_size,
        self_play=not args.no_self_play,
        max_opponents=args.max_opponents,
        self_play_warmup_episodes=args.self_play_warmup_episodes,
        snapshot_interval_updates=args.snapshot_interval_updates,
        eval_interval_updates=args.eval_interval_updates,
        eval_episodes=args.eval_episodes,
        lr_decay=not args.no_lr_decay,
        min_lr_ratio=args.min_lr_ratio,
    )
