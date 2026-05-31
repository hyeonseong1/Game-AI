"""
PPO Space Invaders Training
===========================
Usage:
    python train.py               # default 1,500,000 steps
    python train.py --steps 1000000

Saves:
    models/mid_model.pt   — checkpoint at the halfway point
    models/best_model.pt  — checkpoint with best 50-episode avg reward
"""

import argparse, os, time
import numpy as np

from space_invaders_env import SpaceInvadersEnv, OBS_DIM, ACT_DIM
from ppo_agent import PPOAgent


def train(total_steps: int = 1_500_000, save_dir: str = 'models') -> PPOAgent:
    os.makedirs(save_dir, exist_ok=True)

    env   = SpaceInvadersEnv()
    agent = PPOAgent(state_dim=OBS_DIM, action_dim=ACT_DIM,
                     rollout_size=2048, lr=3e-4)

    print(f"\n{'='*60}")
    print(f"  PPO Space Invaders Training")
    print(f"  Device    : {agent.device}")
    print(f"  State dim : {OBS_DIM}   Action dim: {ACT_DIM}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Mid model  : step {total_steps//2:,}")
    print(f"{'='*60}\n")

    state = env.reset()
    ep_reward   = 0.0
    ep_count    = 0
    ep_rewards: list[float] = []
    ep_scores:  list[float] = []

    best_avg  = float('-inf')
    mid_saved = False
    t0 = time.time()

    for step in range(1, total_steps + 1):
        action, log_prob, value = agent.act(state)

        next_state, reward, done, info = env.step(action)

        agent.push(state, action, log_prob, reward, value, done)
        ep_reward += reward
        state = next_state

        if done:
            ep_rewards.append(ep_reward)
            ep_scores.append(info['score'])
            ep_count += 1
            ep_reward = 0.0
            state = env.reset()

        if agent.buffer.ready():
            loss = agent.update(state)

            # mid model
            if not mid_saved and step >= total_steps // 2:
                agent.save(os.path.join(save_dir, 'mid_model.pt'))
                mid_saved = True
                avg_sc = float(np.mean(ep_scores[-50:])) if len(ep_scores) >= 50 else 0
                print(f"  [SAVED] mid_model.pt  at step {step:,}  "
                      f"(ep {ep_count}, avg_score {avg_sc:.0f})")

            # best model
            if len(ep_rewards) >= 50:
                avg50 = float(np.mean(ep_rewards[-50:]))
                if avg50 > best_avg:
                    best_avg = avg50
                    agent.save(os.path.join(save_dir, 'best_model.pt'))

            # progress log
            if agent.updates % 10 == 0:
                avg = (float(np.mean(ep_rewards[-50:]))
                       if len(ep_rewards) >= 50
                       else float(np.mean(ep_rewards)) if ep_rewards else 0.0)
                sc  = (float(np.mean(ep_scores[-50:]))
                       if len(ep_scores) >= 50
                       else float(np.mean(ep_scores)) if ep_scores else 0.0)
                elapsed = time.time() - t0
                pct = step / total_steps
                eta = (elapsed / pct - elapsed) if pct > 0 else 0
                print(f"  step {step:>8,}/{total_steps:,} "
                      f"({100*pct:4.1f}%) | "
                      f"ep {ep_count:4d} | "
                      f"avg50 {avg:+7.2f} | "
                      f"score {sc:6.0f} | "
                      f"loss {loss:.4f} | "
                      f"ETA {eta/60:.1f}m")

    # safety fallbacks
    if not mid_saved:
        agent.save(os.path.join(save_dir, 'mid_model.pt'))
        print("  [SAVED] mid_model.pt  (fallback)")

    if not os.path.exists(os.path.join(save_dir, 'best_model.pt')):
        agent.save(os.path.join(save_dir, 'best_model.pt'))
        print("  [SAVED] best_model.pt (fallback)")

    print(f"\nTraining complete!  Best avg-50 reward: {best_avg:.2f}")
    return agent


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps',    type=int, default=1_500_000)
    ap.add_argument('--save-dir', default='models')
    args = ap.parse_args()
    train(total_steps=args.steps, save_dir=args.save_dir)
