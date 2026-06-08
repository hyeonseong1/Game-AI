import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


EPS = 1e-8


def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, ord=2, dim=dim, keepdim=True), min=EPS)


class Scaler(nn.Module):
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.forward_scale = float(init) / float(scale)
        self.scale = nn.Parameter(torch.full((dim,), float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.effective_scale()

    def effective_scale(self) -> torch.Tensor:
        return self.scale * self.forward_scale


class HyperLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight())

    def effective_weight(self) -> torch.Tensor:
        return _l2_normalize(self.weight, dim=1)


class HyperMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        normalize_out: bool,
    ):
        super().__init__()
        self.w1 = HyperLinear(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperLinear(hidden_dim, out_dim)
        self.normalize_out = normalize_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        x = F.relu(x) + EPS
        x = self.w2(x)
        if self.normalize_out:
            x = _l2_normalize(x, dim=-1)
        return x


class HyperEmbedder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, scaler_init: float, scaler_scale: float, c_shift: float):
        super().__init__()
        self.c_shift = float(c_shift)
        self.w = HyperLinear(obs_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra = torch.full((*x.shape[:-1], 1), self.c_shift, dtype=x.dtype, device=x.device)
        x = torch.cat([x, extra], dim=-1)
        x = _l2_normalize(x, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        return _l2_normalize(x, dim=-1)


class HyperLERPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int,
    ):
        super().__init__()
        expanded_dim = hidden_dim * expansion
        gain = math.sqrt(expansion)
        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=expanded_dim,
            out_dim=hidden_dim,
            scaler_init=scaler_init / gain,
            scaler_scale=scaler_scale / gain,
            normalize_out=True,
        )
        self.alpha = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        x = x + self.alpha(y - x)
        return _l2_normalize(x, dim=-1)


class HyperCategoricalValue(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.min_v = float(min_v)
        self.max_v = float(max_v)
        self.w1 = HyperLinear(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperLinear(hidden_dim, self.num_bins)
        self.bias = nn.Parameter(torch.zeros(self.num_bins))
        bins = torch.linspace(self.min_v, self.max_v, self.num_bins).view(1, -1)
        self.register_buffer("bins", bins, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.w1(x)
        logits = self.scaler(logits)
        logits = F.relu(logits) + EPS
        logits = self.w2(logits) + self.bias
        probs = F.softmax(logits, dim=-1)
        value = (probs * self.bins.to(device=logits.device, dtype=logits.dtype)).sum(dim=-1)
        return value, logits


class LegacyActorCritic(nn.Module):
    def __init__(self, state_dim: int = 6, action_dim: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(256, action_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_action_and_value(self, x):
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x, actions):
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, None


class SimbaV2ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        expansion: int = 4,
        c_shift: float = 3.0,
        scaler_init: Optional[float] = None,
        scaler_scale: float = 1.0,
        alpha_init: Optional[float] = None,
        alpha_scale: float = 1.0,
        value_bins: int = 101,
        value_min: float = -150.0,
        value_max: float = 200.0,
        obs_clip: float = 5.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.expansion = int(expansion)
        self.c_shift = float(c_shift)
        self.obs_clip = float(obs_clip)
        self.value_bins = int(value_bins)
        self.value_min = float(value_min)
        self.value_max = float(value_max)

        if scaler_init is None:
            scaler_init = 2.0 / math.sqrt(hidden_dim)
        if alpha_init is None:
            alpha_init = 1.0 / (num_blocks + 1.0)
        self.scaler_init = float(scaler_init)
        self.scaler_scale = float(scaler_scale)
        self.alpha_init = float(alpha_init)
        self.alpha_scale = float(alpha_scale)

        self.register_buffer("obs_mean", torch.zeros(state_dim), persistent=True)
        self.register_buffer("obs_var", torch.ones(state_dim), persistent=True)
        self.register_buffer("obs_count", torch.tensor(1e-4), persistent=True)

        self.embedder = HyperEmbedder(state_dim, hidden_dim, self.scaler_init, self.scaler_scale, self.c_shift)
        self.blocks = nn.ModuleList(
            [
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                    expansion=expansion,
                )
                for _ in range(num_blocks)
            ]
        )
        self.actor = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
            normalize_out=False,
        )
        self.actor_bias = nn.Parameter(torch.zeros(action_dim))
        self.critic = HyperCategoricalValue(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_bins=value_bins,
            min_v=value_min,
            max_v=value_max,
        )

    def model_config(self) -> Dict[str, Any]:
        return {
            "architecture": "simba_v2_discrete",
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "expansion": self.expansion,
            "c_shift": self.c_shift,
            "scaler_init": self.scaler_init,
            "scaler_scale": self.scaler_scale,
            "alpha_init": self.alpha_init,
            "alpha_scale": self.alpha_scale,
            "value_bins": self.value_bins,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "obs_clip": self.obs_clip,
        }

    @torch.no_grad()
    def update_obs_stats(self, x: torch.Tensor) -> None:
        x = x.detach()
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.obs_count * batch_count / total_count
        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(torch.clamp(m2 / total_count, min=1e-6))
        self.obs_count.copy_(total_count)

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-6)
        return torch.clamp(x, -self.obs_clip, self.obs_clip)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_obs(x)
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        logits = self.actor(h) + self.actor_bias
        value, value_logits = self.critic(h)
        return logits, value, value_logits

    def get_action_and_value(self, x):
        logits, value, _ = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x, actions):
        logits, value, value_logits = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, value_logits


ActorCritic = SimbaV2ActorCritic


class RolloutBuffer:
    def __init__(self, size: int, state_dim: int):
        self.size = size
        self.ptr = 0
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def push(self, state, action, log_prob, reward, value, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def ready(self):
        return self.ptr >= self.size

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        advs = np.zeros(self.size, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(self.size)):
            mask = 1.0 - self.dones[t]
            next_val = self.values[t + 1] if t < self.size - 1 else last_value
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            advs[t] = last_adv = delta + gamma * lam * mask * last_adv
        returns = advs + self.values
        return advs, returns

    def reset(self):
        self.ptr = 0


@dataclass
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    lr: float

    def __float__(self):
        return float(self.loss)

    def __format__(self, spec: str) -> str:
        return format(float(self.loss), spec)


class PPOAgent:
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        lr: float = 2.5e-4,
        rollout_size: int = 2048,
        architecture: str = "simba_v2_discrete",
        hidden_dim: int = 128,
        num_blocks: int = 4,
        value_bins: int = 101,
        value_min: float = -150.0,
        value_max: float = 200.0,
        total_training_steps: Optional[int] = None,
        lr_decay: bool = True,
        min_lr_ratio: float = 0.05,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rollout_size = int(rollout_size)
        self.architecture = architecture
        self.base_lr = float(lr)
        self.total_training_steps = int(total_training_steps) if total_training_steps else None
        self.lr_decay = bool(lr_decay)
        self.min_lr_ratio = float(min_lr_ratio)
        self.model_kwargs = {
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "value_bins": value_bins,
            "value_min": value_min,
            "value_max": value_max,
        }

        self.policy = self._make_policy(architecture).to(self.device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
        self.buffer = RolloutBuffer(rollout_size, state_dim)

        self.gamma = 0.995
        self.lam = 0.96
        self.clip_eps = 0.18
        self.value_coef = 0.6
        self.entropy_coef = 0.012
        self.entropy_floor = 0.002
        self.n_epochs = 8
        self.batch_size = 256
        self.max_grad_norm = 0.5
        self.target_kl = 0.035

        self.total_steps = 0
        self.updates = 0

    def current_lr(self) -> float:
        if not self.lr_decay or not self.total_training_steps:
            return self.base_lr
        progress = min(max(self.total_steps / max(self.total_training_steps, 1), 0.0), 1.0)
        scale = max(self.min_lr_ratio, 1.0 - progress)
        return self.base_lr * scale

    def _apply_lr_schedule(self) -> float:
        lr = self.current_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _make_policy(self, architecture: str) -> nn.Module:
        if architecture in ("legacy_mlp", "legacy"):
            return LegacyActorCritic(self.state_dim, self.action_dim)
        return SimbaV2ActorCritic(self.state_dim, self.action_dim, **self.model_kwargs)

    def predict(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.policy(s)
            logits = out[0] if isinstance(out, tuple) else out
            return int(logits.argmax(-1).item())

    def act(self, state: np.ndarray):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _, value = self.policy.get_action_and_value(s)
        return action.item(), log_prob.item(), value.item()

    def push(self, state, action, log_prob, reward, value, done):
        self.buffer.push(state, action, log_prob, reward, value, float(done))
        self.total_steps += 1

    def _categorical_value_loss(self, logits: torch.Tensor, returns: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if logits is None:
            return F.smooth_l1_loss(value, returns)
        model = self.policy
        min_v = getattr(model, "value_min", -150.0)
        max_v = getattr(model, "value_max", 200.0)
        num_bins = getattr(model, "value_bins", logits.shape[-1])
        clipped = returns.clamp(min_v, max_v)
        delta = (max_v - min_v) / max(num_bins - 1, 1)
        b = (clipped - min_v) / delta
        lower = torch.floor(b).long().clamp(0, num_bins - 1)
        upper = torch.ceil(b).long().clamp(0, num_bins - 1)
        upper_w = b - lower.float()
        lower_w = 1.0 - upper_w
        target = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device)
        target.scatter_add_(1, lower.unsqueeze(1), lower_w.unsqueeze(1))
        target.scatter_add_(1, upper.unsqueeze(1), upper_w.unsqueeze(1))
        ce = -(target.detach() * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        mse = F.smooth_l1_loss(value, returns)
        return ce + 0.15 * mse

    def update(self, last_state: np.ndarray) -> PPOStats:
        with torch.no_grad():
            s = torch.as_tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.policy(s)
            last_val = out[1].item()

        advs, returns = self.buffer.compute_gae(last_val, self.gamma, self.lam)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        states_t = torch.as_tensor(self.buffer.states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_lp_t = torch.as_tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        advs_t = torch.as_tensor(advs, dtype=torch.float32, device=self.device)
        rets_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        if hasattr(self.policy, "update_obs_stats"):
            self.policy.update_obs_stats(states_t)

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip = 0.0
        n_batches = 0
        lr = self._apply_lr_schedule()

        entropy_coef = max(self.entropy_floor, self.entropy_coef * (0.997 ** self.updates))
        stop_early = False
        for _ in range(self.n_epochs):
            idx = np.random.permutation(self.rollout_size)
            for start in range(0, self.rollout_size, self.batch_size):
                mb = idx[start : start + self.batch_size]

                lp, ent, val, value_logits = self.policy.evaluate(states_t[mb], actions_t[mb])
                logratio = lp - old_lp_t[mb]
                ratio = torch.exp(logratio)
                adv_mb = advs_t[mb]
                p1 = ratio * adv_mb
                p2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(p1, p2).mean()

                value_loss = self._categorical_value_loss(value_logits, rets_t[mb], val)
                entropy = ent.mean()
                loss = policy_loss + self.value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().clamp(min=0)
                    clipfrac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                total_loss += loss.item()
                total_policy += policy_loss.item()
                total_value += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl.item()
                total_clip += clipfrac.item()
                n_batches += 1

                if approx_kl.item() > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.buffer.reset()
        self.updates += 1
        denom = max(n_batches, 1)
        return PPOStats(
            loss=total_loss / denom,
            policy_loss=total_policy / denom,
            value_loss=total_value / denom,
            entropy=total_entropy / denom,
            approx_kl=total_kl / denom,
            clipfrac=total_clip / denom,
            lr=lr,
        )

    def save(self, path: str):
        model_config = self.policy.model_config() if hasattr(self.policy, "model_config") else {"architecture": "legacy_mlp"}
        torch.save(
            {
                "architecture": model_config["architecture"],
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "model_config": model_config,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "updates": self.updates,
                "training_config": {
                    "base_lr": self.base_lr,
                    "total_training_steps": self.total_training_steps,
                    "lr_decay": self.lr_decay,
                    "min_lr_ratio": self.min_lr_ratio,
                },
            },
            path,
        )

    def load(self, path: str, eval_mode: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        architecture = ckpt.get("architecture")
        if architecture is None:
            architecture = "legacy_mlp" if any(k.startswith("shared.") for k in ckpt["policy"]) else "simba_v2_discrete"
        self.state_dim = int(ckpt.get("state_dim", self.state_dim))
        self.action_dim = int(ckpt.get("action_dim", self.action_dim))
        config = ckpt.get("model_config", {})
        if architecture == "simba_v2_discrete":
            self.model_kwargs.update(
                {
                    "hidden_dim": int(config.get("hidden_dim", self.model_kwargs["hidden_dim"])),
                    "num_blocks": int(config.get("num_blocks", self.model_kwargs["num_blocks"])),
                    "value_bins": int(config.get("value_bins", self.model_kwargs["value_bins"])),
                    "value_min": float(config.get("value_min", self.model_kwargs["value_min"])),
                    "value_max": float(config.get("value_max", self.model_kwargs["value_max"])),
                }
            )
        self.architecture = architecture
        training_config = ckpt.get("training_config", {})
        self.base_lr = float(training_config.get("base_lr", self.base_lr))
        self.total_training_steps = training_config.get("total_training_steps", self.total_training_steps)
        self.lr_decay = bool(training_config.get("lr_decay", self.lr_decay))
        self.min_lr_ratio = float(training_config.get("min_lr_ratio", self.min_lr_ratio))
        self.policy = self._make_policy(architecture).to(self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.base_lr, eps=1e-5, weight_decay=1e-4)
        if eval_mode:
            self.policy.eval()
        else:
            opt_state = ckpt.get("optimizer")
            if opt_state:
                try:
                    self.optimizer.load_state_dict(opt_state)
                except ValueError:
                    pass
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates = ckpt.get("updates", 0)


def _tensor_list(x: torch.Tensor):
    return x.detach().cpu().numpy().tolist()


def _export_hyper_mlp(mlp: HyperMLP) -> Dict[str, Any]:
    return {
        "w1": _tensor_list(mlp.w1.effective_weight()),
        "scale": _tensor_list(mlp.scaler.effective_scale()),
        "w2": _tensor_list(mlp.w2.effective_weight()),
        "normalize_out": mlp.normalize_out,
    }


def _export_simba_policy(policy: SimbaV2ActorCritic) -> Dict[str, Any]:
    return {
        "architecture": "simba_v2_discrete",
        "obs_dim": policy.state_dim,
        "act_dim": policy.action_dim,
        "obs_mean": _tensor_list(policy.obs_mean),
        "obs_var": _tensor_list(policy.obs_var),
        "obs_clip": policy.obs_clip,
        "c_shift": policy.c_shift,
        "embed": {
            "w": _tensor_list(policy.embedder.w.effective_weight()),
            "scale": _tensor_list(policy.embedder.scaler.effective_scale()),
        },
        "blocks": [
            {
                "mlp": _export_hyper_mlp(block.mlp),
                "alpha": _tensor_list(block.alpha.effective_scale()),
            }
            for block in policy.blocks
        ],
        "actor": {
            **_export_hyper_mlp(policy.actor),
            "bias": _tensor_list(policy.actor_bias),
        },
    }


def _export_legacy_state(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    def t(key):
        return sd[key].detach().cpu().numpy().tolist()

    return {
        "architecture": "legacy_mlp",
        "shared_0_weight": t("shared.0.weight"),
        "shared_0_bias": t("shared.0.bias"),
        "shared_2_weight": t("shared.2.weight"),
        "shared_2_bias": t("shared.2.bias"),
        "actor_weight": t("actor_head.weight"),
        "actor_bias": t("actor_head.bias"),
    }


def checkpoint_to_json(path: str, state_dim: int, action_dim: int) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["policy"]
    architecture = ckpt.get("architecture")
    if architecture is None:
        architecture = "legacy_mlp" if any(k.startswith("shared.") for k in sd) else "simba_v2_discrete"

    if architecture == "legacy_mlp":
        data = _export_legacy_state(sd)
    else:
        cfg = ckpt.get("model_config", {})
        policy = SimbaV2ActorCritic(
            state_dim=int(ckpt.get("state_dim", state_dim)),
            action_dim=int(ckpt.get("action_dim", action_dim)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            num_blocks=int(cfg.get("num_blocks", 4)),
            expansion=int(cfg.get("expansion", 4)),
            c_shift=float(cfg.get("c_shift", 3.0)),
            scaler_init=float(cfg.get("scaler_init", 2.0 / math.sqrt(float(cfg.get("hidden_dim", 128))))),
            scaler_scale=float(cfg.get("scaler_scale", 1.0)),
            alpha_init=float(cfg.get("alpha_init", 1.0 / (int(cfg.get("num_blocks", 4)) + 1.0))),
            alpha_scale=float(cfg.get("alpha_scale", 1.0)),
            value_bins=int(cfg.get("value_bins", 101)),
            value_min=float(cfg.get("value_min", -150.0)),
            value_max=float(cfg.get("value_max", 200.0)),
            obs_clip=float(cfg.get("obs_clip", 5.0)),
        )
        policy.load_state_dict(sd)
        policy.eval()
        data = _export_simba_policy(policy)

    data["meta"] = {
        "total_steps": int(ckpt.get("total_steps", 0)),
        "updates": int(ckpt.get("updates", 0)),
        "obs_dim": int(ckpt.get("state_dim", state_dim)),
        "act_dim": int(ckpt.get("action_dim", action_dim)),
        "architecture": data["architecture"],
    }
    return data
