"""
PPO Pong Training Script
========================
Usage:
    python train.py               # default 1,000,000 steps
    python train.py --steps 500000

Saves:
    models/mid_model.pt   — checkpoint at the halfway point
    models/best_model.pt  — checkpoint with best 50-episode avg reward
"""

import argparse
import os
import time

import numpy as np

from pong_env import PongEnv
from ppo_agent import PPOAgent


# ------------------------------------------------------------------
# Training bot (controls left paddle)
# Medium difficulty: follows the ball but makes occasional mistakes.
# ------------------------------------------------------------------
def bot_action(env: PongEnv) -> int:
    if np.random.random() < 0.04:          # 4% random noise
        return np.random.randint(3)
    if env.ball[1] < env.left_y - 5:
        return 1   # up
    if env.ball[1] > env.left_y + 5:
        return 2   # down
    return 0


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def train(total_steps: int = 1_000_000, save_dir: str = 'models') -> PPOAgent:
    os.makedirs(save_dir, exist_ok=True)

    env = PongEnv()
    agent = PPOAgent(rollout_size=2048)

    print(f"\n{'='*60}")
    print(f"  PPO Pong Training")
    print(f"  Device    : {agent.device}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Rollout sz : {agent.rollout_size}")
    print(f"  Mid model  : step {total_steps//2:,}")
    print(f"{'='*60}\n")

    state = env.reset()
    ep_reward = 0.0
    ep_count = 0
    episode_rewards: list[float] = []

    best_avg = float('-inf')
    mid_saved = False

    t0 = time.time()

    for step in range(1, total_steps + 1):
        action, log_prob, value = agent.act(state)
        opp_action = bot_action(env)

        next_state, reward, done, _ = env.step(action, opp_action)

        agent.push(state, action, log_prob, reward, value, done)
        ep_reward += reward
        state = next_state

        if done:
            episode_rewards.append(ep_reward)
            ep_count += 1
            ep_reward = 0.0
            state = env.reset()

        # PPO update when buffer is full
        if agent.buffer.ready():
            loss = agent.update(state)

            # ── mid model ──
            if not mid_saved and step >= total_steps // 2:
                agent.save(os.path.join(save_dir, 'mid_model.pt'))
                mid_saved = True
                print(f"  [SAVED] mid_model.pt  at step {step:,}  "
                      f"(ep {ep_count})")

            # ── best model ──
            if len(episode_rewards) >= 50:
                avg50 = float(np.mean(episode_rewards[-50:]))
                if avg50 > best_avg:
                    best_avg = avg50
                    agent.save(os.path.join(save_dir, 'best_model.pt'))

            # ── progress log every 10 updates ──
            if agent.updates % 10 == 0:
                avg = (float(np.mean(episode_rewards[-50:]))
                       if len(episode_rewards) >= 50
                       else float(np.mean(episode_rewards)) if episode_rewards
                       else 0.0)
                elapsed = time.time() - t0
                pct = step / total_steps
                eta = (elapsed / pct - elapsed) if pct > 0 else 0
                print(f"  step {step:>8,}/{total_steps:,} "
                      f"({100*pct:4.1f}%) | "
                      f"ep {ep_count:5d} | "
                      f"avg50 {avg:+6.2f} | "
                      f"loss {loss:.4f} | "
                      f"ETA {eta/60:.1f}m")

    # ── final saves (safety nets) ──
    if not mid_saved:
        agent.save(os.path.join(save_dir, 'mid_model.pt'))
        print("  [SAVED] mid_model.pt  (end-of-training fallback)")

    if not os.path.exists(os.path.join(save_dir, 'best_model.pt')):
        agent.save(os.path.join(save_dir, 'best_model.pt'))
        print("  [SAVED] best_model.pt (end-of-training fallback)")

    print(f"\nTraining complete!  Best avg-50 reward: {best_avg:.2f}")
    print(f"Models saved in '{save_dir}/'")
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Total environment steps (default: 1,000,000)')
    parser.add_argument('--save-dir', default='models',
                        help='Directory to save model checkpoints')
    args = parser.parse_args()
    train(total_steps=args.steps, save_dir=args.save_dir)
