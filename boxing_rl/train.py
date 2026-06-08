"""
PPO Boxing Training Script
==========================
Usage:
    python train.py               # default 10,000,000 steps
    python train.py --steps 10000000

Saves:
    models/mid_model.pt   — checkpoint at the halfway point
    models/best_model.pt  — checkpoint with best 50-episode avg reward
"""

import argparse, os, time
import numpy as np

from boxing_env import BoxingEnv, OBS_DIM, ACT_DIM
from ppo_agent import PPOAgent


def train(
    total_steps: int = 10_000_000,
    save_dir: str = 'models',
    lr_decay: bool = True,
    min_lr_ratio: float = 0.05,
) -> PPOAgent:
    os.makedirs(save_dir, exist_ok=True)

    opponent_pool = ('medium', 'hard')
    opponent_probs = (0.35, 0.65)
    env   = BoxingEnv(opponent=np.random.choice(opponent_pool, p=opponent_probs))
    agent = PPOAgent(state_dim=OBS_DIM, action_dim=ACT_DIM, rollout_size=2048,
                     value_min=-50.0, value_max=180.0,
                     total_training_steps=total_steps,
                     lr_decay=lr_decay,
                     min_lr_ratio=min_lr_ratio)

    print(f"\n{'='*60}")
    print(f"  SimbaV2 PPO Boxing Training")
    print(f"  Device    : {agent.device}")
    print(f"  State dim : {OBS_DIM}   Action dim: {ACT_DIM}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  LR decay   : {'linear' if lr_decay else 'off'} "
          f"(base={agent.base_lr:.2e}, min={agent.base_lr * min_lr_ratio:.2e})")
    print(f"  Mid model  : step {total_steps//2:,}")
    print(f"{'='*60}\n")

    state = env.reset()
    ep_reward   = 0.0
    ep_count    = 0
    ep_rewards: list[float] = []

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
            ep_count += 1
            ep_reward = 0.0
            env = BoxingEnv(opponent=np.random.choice(opponent_pool, p=opponent_probs))
            state = env.reset()

        if agent.buffer.ready():
            loss = agent.update(state)

            # ── mid model ──
            if not mid_saved and step >= total_steps // 2:
                agent.save(os.path.join(save_dir, 'mid_model.pt'))
                mid_saved = True
                print(f"  [SAVED] mid_model.pt  at step {step:,}  (ep {ep_count})")

            # ── best model ──
            if len(ep_rewards) >= 50:
                avg50 = float(np.mean(ep_rewards[-50:]))
                if avg50 > best_avg:
                    best_avg = avg50
                    agent.save(os.path.join(save_dir, 'best_model.pt'))

            # ── progress ──
            if agent.updates % 10 == 0:
                avg = (float(np.mean(ep_rewards[-50:]))
                       if len(ep_rewards) >= 50
                       else float(np.mean(ep_rewards)) if ep_rewards else 0.0)
                elapsed = time.time() - t0
                pct = step / total_steps
                eta = (elapsed / pct - elapsed) if pct > 0 else 0
                print(f"  step {step:>8,}/{total_steps:,} "
                      f"({100*pct:4.1f}%) | "
                      f"ep {ep_count:5d} | "
                      f"avg50 {avg:+7.3f} | "
                      f"loss {loss:.4f} | "
                      f"lr {loss.lr:.2e} | "
                      f"ETA {eta/60:.1f}m")

    # ── safety fallbacks ──
    if not mid_saved:
        agent.save(os.path.join(save_dir, 'mid_model.pt'))
        print("  [SAVED] mid_model.pt  (fallback)")

    if not os.path.exists(os.path.join(save_dir, 'best_model.pt')):
        agent.save(os.path.join(save_dir, 'best_model.pt'))
        print("  [SAVED] best_model.pt (fallback)")

    print(f"\nTraining complete!  Best avg-50 reward: {best_avg:.3f}")
    return agent


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps',    type=int, default=10_000_000)
    ap.add_argument('--save-dir', default='models')
    ap.add_argument('--no-lr-decay', action='store_true')
    ap.add_argument('--min-lr-ratio', type=float, default=0.05)
    args = ap.parse_args()
    train(
        total_steps=args.steps,
        save_dir=args.save_dir,
        lr_decay=not args.no_lr_decay,
        min_lr_ratio=args.min_lr_ratio,
    )
