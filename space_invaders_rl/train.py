import argparse
import json
import os
import random
import time
from dataclasses import asdict
from typing import Tuple

import numpy as np
import torch

from rainbow_agent import (
    NStepTransitionBuffer,
    PrioritizedReplay,
    RainbowAgent,
    RainbowPaperConfig,
)
from space_invaders_env import ACT_DIM, OBS_DIM, SpaceInvadersEnv


def _clip_reward(reward: float, clip: float) -> float:
    if clip <= 0:
        return float(reward)
    return float(max(min(reward, clip), -clip))


def evaluate(agent: RainbowAgent, episodes: int, seed: int) -> Tuple[float, float]:
    if episodes <= 0:
        return float("nan"), float("nan")
    was_training = agent.online_net.training
    agent.online_net.eval()
    scores = []
    rewards = []
    for ep in range(episodes):
        env = SpaceInvadersEnv(seed=seed + 10_000 + ep, frame_skip=agent.cfg.frame_skip)
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.predict(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
        scores.append(info.get("score", 0))
        rewards.append(ep_reward)
    if was_training:
        agent.online_net.train()
    return float(np.mean(scores)), float(np.mean(rewards))


def train(
    cfg: RainbowPaperConfig,
    save_dir: str = "models",
    disable_cuda: bool = False,
    eval_interval: int = None,
    eval_episodes: int = None,
) -> RainbowAgent:
    os.makedirs(save_dir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cpu" if disable_cuda or not torch.cuda.is_available() else "cuda")
    env = SpaceInvadersEnv(
        seed=cfg.seed,
        frame_skip=cfg.frame_skip,
        max_episode_frames=cfg.max_episode_frames,
    )
    agent = RainbowAgent(OBS_DIM, ACT_DIM, cfg, device=device)
    memory = PrioritizedReplay(
        obs_dim=OBS_DIM,
        capacity=cfg.memory_capacity,
        alpha=cfg.priority_exponent,
        beta=cfg.priority_weight,
    )
    nstep = NStepTransitionBuffer(cfg.multi_step, cfg.discount)

    eval_interval = int(eval_interval if eval_interval is not None else cfg.evaluation_interval)
    eval_episodes = int(eval_episodes if eval_episodes is not None else cfg.evaluation_episodes)
    beta_increase = (1.0 - cfg.priority_weight) / max(cfg.total_steps - cfg.learn_start, 1)

    print()
    print("=" * 72)
    print("  Space Invaders Rainbow DQN Training")
    print(f"  Device          : {device}")
    print(f"  Agent steps     : {cfg.total_steps:,} ({cfg.total_steps * cfg.frame_skip:,} frames)")
    print(f"  Learn start     : {cfg.learn_start:,} steps ({cfg.learn_start * cfg.frame_skip:,} frames)")
    print(f"  Replay capacity : {cfg.memory_capacity:,}")
    print(f"  Target update   : {cfg.target_update:,} steps ({cfg.target_update * cfg.frame_skip:,} frames)")
    print(f"  C51 support     : {cfg.atoms} atoms [{cfg.v_min:g}, {cfg.v_max:g}]")
    print(f"  NoisyNet sigma0 : {cfg.noisy_std:g}, epsilon={cfg.epsilon:g}")
    print(f"  PER alpha/beta  : {cfg.priority_exponent:g}, {cfg.priority_weight:g}->1")
    print(f"  Multi-step n    : {cfg.multi_step}")
    print("=" * 72)
    print()

    state = env.reset()
    episode_reward = 0.0
    episode_score = 0
    episode = 0
    rewards = []
    scores = []
    best_score = float("-inf")
    mid_saved = False
    t0 = time.time()

    for step in range(1, cfg.total_steps + 1):
        if step % cfg.replay_frequency == 0:
            agent.reset_noise()

        action = agent.predict(state)
        next_state, raw_reward, done, info = env.step(action)
        reward = _clip_reward(raw_reward, cfg.reward_clip)

        for transition in nstep.append(state, action, reward, next_state, done):
            memory.append(*transition)

        episode_reward += raw_reward
        episode_score = int(info.get("score", episode_score))
        state = next_state
        agent.total_steps = step

        if done:
            rewards.append(float(episode_reward))
            scores.append(int(episode_score))
            episode += 1
            episode_reward = 0.0
            episode_score = 0
            state = env.reset()

        if step >= cfg.learn_start:
            memory.beta = min(1.0, memory.beta + beta_increase)
            if step % cfg.replay_frequency == 0 and len(memory) >= cfg.batch_size:
                loss = agent.learn(memory)
            else:
                loss = float("nan")

            if step % cfg.target_update == 0:
                agent.update_target()

            if not mid_saved and step >= max(cfg.total_steps // 2, cfg.learn_start):
                agent.save(os.path.join(save_dir, "mid_model.pt"), extra={"kind": "mid"})
                mid_saved = True
                print(f"  [SAVED] mid_model.pt at step {step:,}")

            if eval_interval > 0 and step % eval_interval == 0:
                avg_score, avg_reward = evaluate(agent, eval_episodes, cfg.seed)
                if np.isfinite(avg_score) and avg_score >= best_score:
                    best_score = avg_score
                    agent.save(
                        os.path.join(save_dir, "best_model.pt"),
                        extra={"kind": "best", "eval_avg_score": avg_score, "eval_avg_reward": avg_reward},
                    )
                    print(f"  [BEST] eval_score {avg_score:7.2f} | eval_reward {avg_reward:8.2f}")

            if step % max(cfg.replay_frequency * 250, 1000) == 0:
                elapsed = time.time() - t0
                pct = step / cfg.total_steps
                eta = elapsed / pct - elapsed if pct > 0 else 0.0
                avg_score_20 = float(np.mean(scores[-20:])) if scores else 0.0
                avg_reward_20 = float(np.mean(rewards[-20:])) if rewards else 0.0
                print(
                    f"  step {step:>10,}/{cfg.total_steps:,} "
                    f"({100 * pct:5.2f}%) | ep {episode:5d} | "
                    f"score20 {avg_score_20:7.2f} | reward20 {avg_reward_20:8.2f} | "
                    f"mem {len(memory):7,} | beta {memory.beta:5.3f} | "
                    f"loss {loss:8.4f} | ETA {eta / 60:7.1f}m"
                )

    if not mid_saved:
        agent.save(os.path.join(save_dir, "mid_model.pt"), extra={"kind": "mid_fallback"})
        print("  [SAVED] mid_model.pt (fallback)")
    if not os.path.exists(os.path.join(save_dir, "best_model.pt")):
        agent.save(os.path.join(save_dir, "best_model.pt"), extra={"kind": "best_fallback"})
        print("  [SAVED] best_model.pt (fallback)")

    summary = {
        "algorithm": "Rainbow DQN",
        "paper_locked": cfg.paper_locked(),
        "effective_config": asdict(cfg),
        "episodes": episode,
        "avg_score_last_20": float(np.mean(scores[-20:])) if scores else 0.0,
        "avg_reward_last_20": float(np.mean(rewards[-20:])) if rewards else 0.0,
        "best_eval_score": best_score,
        "actual_evaluation_interval": eval_interval,
        "actual_evaluation_episodes": eval_episodes,
        "total_steps": cfg.total_steps,
        "training_frames": cfg.total_steps * cfg.frame_skip,
    }
    with open(os.path.join(save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Training complete.")
    print(f"Models saved in {save_dir}/")
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=RainbowPaperConfig.total_steps)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--seed", type=int, default=RainbowPaperConfig.seed)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=RainbowPaperConfig.hidden_size)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = RainbowPaperConfig(
        total_steps=args.steps,
        seed=args.seed,
        hidden_size=args.hidden_size,
    )
    train(
        cfg=config,
        save_dir=args.save_dir,
        disable_cuda=args.disable_cuda,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )
