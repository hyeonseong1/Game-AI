import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
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
        return dist.log_prob(actions), dist.entropy(), value


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


class PPOAgent:
    def __init__(self, state_dim: int = 6, action_dim: int = 3,
                 lr: float = 3e-4, rollout_size: int = 2048):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.rollout_size = rollout_size

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer(rollout_size, state_dim)

        # PPO hyper-parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.n_epochs = 10
        self.batch_size = 256
        self.max_grad_norm = 0.5

        self.total_steps = 0
        self.updates = 0

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #

    def predict(self, state: np.ndarray) -> int:
        """Greedy action for play (no gradient, no exploration)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.policy(s)
            return logits.argmax(-1).item()

    def act(self, state: np.ndarray):
        """Stochastic action for training. Returns (action, log_prob, value)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.policy.get_action_and_value(s)
        return action.item(), log_prob.item(), value.item()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def push(self, state, action, log_prob, reward, value, done):
        self.buffer.push(state, action, log_prob, reward, value, float(done))
        self.total_steps += 1

    def update(self, last_state: np.ndarray) -> float:
        """Run PPO update over the current rollout buffer."""
        with torch.no_grad():
            s = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            _, last_val = self.policy(s)
            last_val = last_val.item()

        advs, returns = self.buffer.compute_gae(last_val, self.gamma, self.lam)

        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        states_t = torch.FloatTensor(self.buffer.states).to(self.device)
        actions_t = torch.LongTensor(self.buffer.actions).to(self.device)
        old_lp_t = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advs_t = torch.FloatTensor(advs).to(self.device)
        rets_t = torch.FloatTensor(returns).to(self.device)

        total_loss = 0.0
        n_batches = 0
        for _ in range(self.n_epochs):
            idx = np.random.permutation(self.rollout_size)
            for start in range(0, self.rollout_size, self.batch_size):
                mb = idx[start: start + self.batch_size]

                lp, ent, val = self.policy.evaluate(states_t[mb], actions_t[mb])
                ratio = torch.exp(lp - old_lp_t[mb])

                # Clipped policy loss
                adv_mb = advs_t[mb]
                p1 = ratio * adv_mb
                p2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(p1, p2).mean()

                value_loss = nn.functional.mse_loss(val, rets_t[mb])
                entropy_loss = -ent.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        self.buffer.reset()
        self.updates += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'updates': self.updates,
        }, path)

    def load(self, path: str, eval_mode: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        if eval_mode:
            self.policy.eval()
        else:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
        self.updates = ckpt.get('updates', 0)
