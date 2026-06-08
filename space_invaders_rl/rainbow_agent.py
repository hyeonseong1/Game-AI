import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


@dataclass
class RainbowPaperConfig:
    # Hessel et al. report Atari settings in frames. This project uses
    # frame_skip=4, so step-based fields are the corresponding agent steps.
    total_steps: int = 50_000_000          # 200M frames / 4
    frame_skip: int = 4
    max_episode_frames: int = 108_000
    learn_start: int = 20_000             # 80K frames / 4
    replay_frequency: int = 4
    target_update: int = 8_000            # 32K frames / 4
    memory_capacity: int = 1_000_000
    priority_exponent: float = 0.5
    priority_weight: float = 0.4
    multi_step: int = 3
    discount: float = 0.99
    atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    noisy_std: float = 0.5
    epsilon: float = 0.0
    learning_rate: float = 0.0000625
    adam_eps: float = 1.5e-4
    batch_size: int = 32
    norm_clip: float = 10.0
    reward_clip: float = 1.0
    hidden_size: int = 512
    evaluation_interval: int = 250_000    # 1M frames / 4
    evaluation_episodes: int = 10
    seed: int = 123

    def paper_locked(self) -> Dict[str, object]:
        return {
            "source": "Rainbow: Combining Improvements in Deep Reinforcement Learning, Table 1",
            "frame_skip": self.frame_skip,
            "min_history_to_start_learning_frames": self.learn_start * self.frame_skip,
            "adam_learning_rate": self.learning_rate,
            "exploration_epsilon": self.epsilon,
            "noisy_nets_sigma0": self.noisy_std,
            "target_network_period_frames": self.target_update * self.frame_skip,
            "adam_epsilon": self.adam_eps,
            "prioritization_type": "proportional",
            "prioritization_exponent": self.priority_exponent,
            "importance_sampling_beta": f"{self.priority_weight} -> 1.0",
            "multi_step_returns": self.multi_step,
            "distributional_atoms": self.atoms,
            "distributional_min_max": [self.v_min, self.v_max],
            "configured_training_frames": self.total_steps * self.frame_skip,
            "paper_benchmark_horizon_frames": 200_000_000,
        }


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowFeatureDQN(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: RainbowPaperConfig):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.atoms = int(cfg.atoms)
        self.v_min = float(cfg.v_min)
        self.v_max = float(cfg.v_max)
        self.hidden_size = int(cfg.hidden_size)

        self.fc = NoisyLinear(obs_dim, cfg.hidden_size, cfg.noisy_std)
        self.value_hidden = NoisyLinear(cfg.hidden_size, cfg.hidden_size, cfg.noisy_std)
        self.value_out = NoisyLinear(cfg.hidden_size, cfg.atoms, cfg.noisy_std)
        self.adv_hidden = NoisyLinear(cfg.hidden_size, cfg.hidden_size, cfg.noisy_std)
        self.adv_out = NoisyLinear(cfg.hidden_size, act_dim * cfg.atoms, cfg.noisy_std)

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        x = F.relu(self.fc(x))
        value = F.relu(self.value_hidden(x))
        value = self.value_out(value).view(-1, 1, self.atoms)
        adv = F.relu(self.adv_hidden(x))
        adv = self.adv_out(adv).view(-1, self.act_dim, self.atoms)
        logits = value + adv - adv.mean(dim=1, keepdim=True)
        if log:
            return F.log_softmax(logits, dim=2)
        return F.softmax(logits, dim=2)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        idx = self.write + self.capacity - 1
        self.update(idx, priority)
        data_idx = self.write
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def update(self, idx: int, priority: float) -> None:
        change = float(priority) - self.tree[idx]
        self.tree[idx] = float(priority)
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def repair(self, default_priority: float, priority_cap: float) -> None:
        leaves = self.tree[self.capacity - 1:]
        if self.size < self.capacity:
            active = leaves[:self.size]
            leaves[self.size:] = 0.0
        else:
            active = leaves
        if active.size:
            bad = ~np.isfinite(active) | (active <= 0.0)
            active[bad] = default_priority
            np.clip(active, 0.0, priority_cap, out=active)
        for idx in range(self.capacity - 2, -1, -1):
            self.tree[idx] = self.tree[2 * idx + 1] + self.tree[2 * idx + 2]

    def get(self, value: float) -> Tuple[int, float, int]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                data_idx = idx - self.capacity + 1
                return idx, float(self.tree[idx]), data_idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right


class PrioritizedReplay:
    def __init__(self, obs_dim: int, capacity: int, alpha: float, beta: float, eps: float = 1e-6):
        self.obs_dim = int(obs_dim)
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.priority_cap = 1e6
        self.max_priority = 1.0
        self.tree = SumTree(capacity)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self.tree.size

    def _sanitize_priority(self, priority: float) -> float:
        priority = float(priority)
        if not np.isfinite(priority) or priority <= 0.0:
            priority = self.max_priority if np.isfinite(self.max_priority) and self.max_priority > 0 else 1.0
        return float(min(max(priority, self.eps), self.priority_cap))

    def _repair_if_needed(self) -> float:
        total = self.tree.total()
        max_total = self.priority_cap * max(len(self), 1)
        if np.isfinite(total) and 0.0 < total <= max_total:
            return total
        default_priority = self._sanitize_priority(self.max_priority)
        self.tree.repair(default_priority=default_priority, priority_cap=self.priority_cap)
        total = self.tree.total()
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("Cannot sample from an empty or invalid replay tree")
        return total

    def append(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        data_idx = self.tree.add(self._sanitize_priority(self.max_priority))
        self.states[data_idx] = state
        self.actions[data_idx] = int(action)
        self.rewards[data_idx] = float(reward)
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = bool(done)

    def sample(self, batch_size: int, device: torch.device):
        total = self._repair_if_needed()
        segment = total / batch_size
        idxs = np.empty(batch_size, dtype=np.int64)
        data_idxs = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                total = self._repair_if_needed()
                segment = total / batch_size
                lo = segment * i
                hi = segment * (i + 1)
            sample_value = np.random.uniform(lo, hi)
            idx, priority, data_idx = self.tree.get(sample_value)
            idxs[i] = idx
            data_idxs[i] = data_idx
            priorities[i] = self._sanitize_priority(priority)

        probs = priorities / total
        weights = (len(self) * probs) ** (-self.beta)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights /= max(float(weights.max()), self.eps)

        states = torch.as_tensor(self.states[data_idxs], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[data_idxs], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self.rewards[data_idxs], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self.next_states[data_idxs], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[data_idxs].astype(np.float32), dtype=torch.float32, device=device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
        return idxs, states, actions, rewards, next_states, dones, weights_t

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(idxs, priorities):
            raw_priority = float(abs(priority)) if np.isfinite(priority) else self.priority_cap
            priority = self._sanitize_priority((raw_priority + self.eps) ** self.alpha)
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)


class NStepTransitionBuffer:
    def __init__(self, n: int, discount: float):
        self.n = int(n)
        self.discount = float(discount)
        self.queue: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def _make_transition(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        ret = 0.0
        next_state = self.queue[-1][3]
        done = False
        for i, (_, _, reward, nstate, terminal) in enumerate(self.queue[: self.n]):
            ret += (self.discount ** i) * float(reward)
            next_state = nstate
            if terminal:
                done = True
                break
        state, action = self.queue[0][0], self.queue[0][1]
        return state, action, ret, next_state, done

    def append(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.queue.append((state, int(action), float(reward), next_state, bool(done)))
        out = []
        if len(self.queue) >= self.n:
            out.append(self._make_transition())
            self.queue.pop(0)
        if done:
            while self.queue:
                out.append(self._make_transition())
                self.queue.pop(0)
        return out


class RainbowAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: RainbowPaperConfig, device: Optional[torch.device] = None):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.support = torch.linspace(cfg.v_min, cfg.v_max, cfg.atoms, device=self.device)
        self.delta_z = (cfg.v_max - cfg.v_min) / (cfg.atoms - 1)
        self.online_net = RainbowFeatureDQN(obs_dim, act_dim, cfg).to(self.device)
        self.target_net = RainbowFeatureDQN(obs_dim, act_dim, cfg).to(self.device)
        self.update_target()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
        self.total_steps = 0
        self.updates = 0

    def reset_noise(self) -> None:
        self.online_net.reset_noise()

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.no_grad()
    def predict(self, state: np.ndarray) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.online_net(state_t)
        q = (dist * self.support.view(1, 1, -1)).sum(dim=2)
        return int(q.argmax(dim=1).item())

    def learn(self, memory: PrioritizedReplay) -> float:
        idxs, states, actions, returns, next_states, dones, weights = memory.sample(self.cfg.batch_size, self.device)

        log_dist = self.online_net(states, log=True)
        log_dist_a = log_dist[torch.arange(self.cfg.batch_size, device=self.device), actions]

        with torch.no_grad():
            next_online_dist = self.online_net(next_states)
            next_online_q = (next_online_dist * self.support.view(1, 1, -1)).sum(dim=2)
            next_actions = next_online_q.argmax(dim=1)

            self.target_net.reset_noise()
            next_target_dist = self.target_net(next_states)
            next_dist_a = next_target_dist[torch.arange(self.cfg.batch_size, device=self.device), next_actions]

            tz = returns.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * (
                self.cfg.discount ** self.cfg.multi_step
            ) * self.support.unsqueeze(0)
            tz = tz.clamp(self.cfg.v_min, self.cfg.v_max)
            b = (tz - self.cfg.v_min) / self.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()

            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < self.cfg.atoms - 1) & (lower == upper)] += 1

            projection = torch.zeros(self.cfg.batch_size, self.cfg.atoms, device=self.device)
            offset = (
                torch.arange(self.cfg.batch_size, device=self.device).unsqueeze(1) * self.cfg.atoms
            )
            projection.view(-1).index_add_(
                0,
                (lower + offset).view(-1),
                (next_dist_a * (upper.float() - b)).view(-1),
            )
            projection.view(-1).index_add_(
                0,
                (upper + offset).view(-1),
                (next_dist_a * (b - lower.float())).view(-1),
            )

        per_sample_loss = -(projection * log_dist_a).sum(dim=1)
        loss = (weights * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.online_net.parameters(), self.cfg.norm_clip)
        self.optimizer.step()

        priorities = per_sample_loss.detach().cpu().numpy()
        memory.update_priorities(idxs, priorities)
        self.updates += 1
        return float(loss.item())

    def save(self, path: str, extra: Optional[Dict[str, object]] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "architecture": "rainbow_feature_c51",
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "model_config": asdict(self.cfg),
            "paper_locked": self.cfg.paper_locked(),
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
        }
        if extra:
            ckpt["extra"] = extra
        torch.save(ckpt, path)

    def load(self, path: str, eval_mode: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg_data = ckpt.get("model_config", {})
        self.cfg = RainbowPaperConfig(**{**asdict(self.cfg), **cfg_data})
        self.obs_dim = int(ckpt.get("obs_dim", self.obs_dim))
        self.act_dim = int(ckpt.get("act_dim", self.act_dim))
        self.support = torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.atoms, device=self.device)
        self.delta_z = (self.cfg.v_max - self.cfg.v_min) / (self.cfg.atoms - 1)
        self.online_net = RainbowFeatureDQN(self.obs_dim, self.act_dim, self.cfg).to(self.device)
        self.target_net = RainbowFeatureDQN(self.obs_dim, self.act_dim, self.cfg).to(self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt.get("target_net", ckpt["online_net"]))
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.cfg.learning_rate, eps=self.cfg.adam_eps)
        if not eval_mode and ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.updates = int(ckpt.get("updates", 0))
        if eval_mode:
            self.online_net.eval()
            self.target_net.eval()


def _linear_to_json(layer: NoisyLinear) -> Dict[str, object]:
    return {
        "weight": layer.weight_mu.detach().cpu().tolist(),
        "bias": layer.bias_mu.detach().cpu().tolist(),
    }


def checkpoint_to_rainbow_json(path: str) -> Dict[str, object]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RainbowPaperConfig(**{**asdict(RainbowPaperConfig()), **ckpt.get("model_config", {})})
    obs_dim = int(ckpt.get("obs_dim", 20))
    act_dim = int(ckpt.get("act_dim", 6))
    model = RainbowFeatureDQN(obs_dim, act_dim, cfg)
    model.load_state_dict(ckpt["online_net"])
    model.eval()
    return {
        "architecture": "rainbow_feature_c51",
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "atoms": cfg.atoms,
        "v_min": cfg.v_min,
        "v_max": cfg.v_max,
        "support": np.linspace(cfg.v_min, cfg.v_max, cfg.atoms, dtype=np.float32).tolist(),
        "hidden_size": cfg.hidden_size,
        "layers": {
            "fc": _linear_to_json(model.fc),
            "value_hidden": _linear_to_json(model.value_hidden),
            "value_out": _linear_to_json(model.value_out),
            "adv_hidden": _linear_to_json(model.adv_hidden),
            "adv_out": _linear_to_json(model.adv_out),
        },
        "paper_locked": cfg.paper_locked(),
        "meta": {
            "total_steps": int(ckpt.get("total_steps", 0)),
            "updates": int(ckpt.get("updates", 0)),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "architecture": "rainbow_feature_c51",
            "training_frames": int(ckpt.get("total_steps", 0)) * cfg.frame_skip,
        },
    }
