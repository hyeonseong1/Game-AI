# AI Arena — Web Game Platform

A web-based platform featuring 6 classic arcade games, each with an AI opponent.
Pong and Boxing use **SimbaV2 PPO reinforcement learning agents**; Space Invaders uses a **paper-locked Rainbow DQN agent**;
the remaining games use well-established classic AI algorithms.

---

## Table of Contents

- [Features](#features)
- [Games](#games)
- [AI Design](#ai-design)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Retraining RL Models](#retraining-rl-models)
- [Tech Stack](#tech-stack)
- [Credits](#credits)

---

## Features

| Feature | Description |
|---------|-------------|
| **6 games** | Pong, Gomoku, Chess, Boxing, Space Invaders, Tetris |
| **3 difficulty levels** | Easy (random/weak AI) → Medium (mid checkpoint) → Hell (best model) |
| **SimbaV2 PPO reinforcement learning** | Hyperspherical-normalized PPO training path for Pong and Boxing |
| **Rainbow DQN** | Space Invaders agent with Double DQN targets, prioritized replay, dueling C51 distributions, NoisyNet exploration, and 3-step returns |
| **Tactical guard rails** | Runtime assist prevents obvious misses, bad boxing spacing, and Space Invaders bomb-lane deaths |
| **Dual-screen battles** | Space Invaders and Tetris run human (left) vs AI (right) simultaneously |
| **In-browser inference** | RL models exported as JSON — no Python server needed at runtime |
| **Win/Lose result modal** | Animated modal at game end with scores, AI comparison %, restart button, and leaderboard shortcut |
| **Personal record tracking** | Detects and highlights new personal bests per user, game, and difficulty; shows previous best vs current score |
| **Leaderboard** | Per-game, per-difficulty score tracking (JSON DB) |
| **Community** | Posts, likes, comments |
| **Guest login** | Play instantly without signing up |

---

## Games

### 1. Pong (Atari, 1972)
Classic table tennis. Player controls the left paddle; AI controls the right.

- **Controls**: `W`/`S` or `↑`/`↓`
- **Win condition**: First to 11 points
- **AI**: PPO neural network (obs: 6-dim, actions: 3)
  - Easy: random policy
  - Medium: `pong_mid.json` (checkpoint at step 501,760)
  - Hell: `pong_best.json` (best reward, step 665,600)

---

### 2. Gomoku (15×15)
Place 5 stones in a row to win on a 15×15 board. Stone color is randomly assigned each game.

- **Controls**: Click to place a stone
- **Win condition**: 5 in a row (horizontal, vertical, or diagonal)
- **Color**: Randomly assigned — if you are White, AI (Black) moves first
- **AI**: Bitboard-based Alpha-Beta Minimax (Web Worker)
  - Easy: greedy heuristic (instant)
  - Medium: 6-ply minimax
  - Hell: 8-ply minimax

---

### 3. Chess
Powered by the Stockfish 18 WASM engine. You play as White.

- **Controls**: Drag and drop pieces
- **AI**: [Stockfish 18](https://github.com/nmrugg/stockfish.js) (Lite Single-threaded WASM, 7 MB)
  - Easy: Skill Level 3, depth 2
  - Medium: Skill Level 10, depth 8
  - Hell: Skill Level 20, depth 15

---

### 4. Boxing (top-view canvas)
Two boxers move and punch in a ring. You are the white boxer on the left.

- **Controls**: `WASD`/arrows (move), `K`/`L`/`Space` (punch)
- **Win condition**: First to 100 points, or higher score after 2 minutes
- **AI**: PPO neural network (obs: 14-dim, actions: 18)
  - Easy: random policy
  - Medium: `boxing_mid.json` (step 501,760)
  - Hell: `boxing_best.json` (best reward, step 342,016)

---

### 5. Space Invaders (Taito, 1978) — **Dual-Screen**
Two game canvases run side by side. Left is the human player; right is the AI.

```
┌──────────────────┬────┬──────────────────┐
│ 🎮 You (Player 1)│ VS │ 🤖 AI (Player 2) │
│   keyboard       │    │ Rainbow DQN agent │
└──────────────────┴────┴──────────────────┘
```

- **Controls**: `←`/`→` (move), `Space` (fire) — left screen only
- **AI**: Rainbow DQN neural network (obs: 20-dim, actions: 6)
  - Easy: random policy
  - Medium: `si_rainbow_mid.json` (20,000 agent steps / 80,000 frames)
  - Hell: `si_rainbow_best.json` (22,000 agent steps / 88,000 frames)

---

### 6. Tetris — **Dual-Screen**
Two game canvases run side by side. Left is the human player; right is the AI.

```
┌──────────────────┬────┬──────────────────┐
│ 🎮 You (Player 1)│ VS │ 🤖 AI (Player 2) │
│   keyboard       │    │   heuristic AI    │
└──────────────────┴────┴──────────────────┘
```

- **Controls**: `←`/`→` (move), `↑`/`Z` (rotate), `↓` (soft drop), `Space` (hard drop) — left screen only
- **AI**: 4-weight heuristic evaluation ([tetrisai](https://github.com/LeeYiyuan/tetrisai) port)
  - Easy: random placement
  - Medium: 1-piece lookahead
  - Hell: 2-piece lookahead (genetically tuned weights) + instant hard drop on alignment

---

## AI Design

### RL Agents (Pong · Boxing · Space Invaders)

Pong and Boxing use a **SimbaV2-style discrete ActorCritic PPO** trainer. Space Invaders now uses a **Rainbow DQN** trainer that follows Hessel et al.'s Rainbow Table 1 settings: Double DQN targets, proportional prioritized replay, dueling C51 return distributions, NoisyNet exploration, 3-step returns, target network updates every 32K frames, and learning after 80K frames. The browser inference engine still supports the older 2-layer MLP and SimbaV2 JSON files, while active Space Invaders play loads the Rainbow JSON files.

#### Network Architecture

```
Pong / Boxing:
Input (obs_dim)
    │
Running observation normalization
    │
HyperEmbedder + L2 feature normalization
    │
4 × HyperLERP residual blocks
    - row-normalized linear weights
    - L2-normalized features
    - learnable residual interpolation
    │
PPO Actor Head          act_dim logits (Pong, Boxing)
PPO Critic Head         distributional value over fixed support

Space Invaders:
Input (20 feature obs)
    │
NoisyLinear feature encoder
    │
Dueling value / advantage NoisyLinear streams
    │
51 C51 atoms over [-10, 10]
    │
Expected Q-values over 6 actions
```

#### Observation and Action Spaces

| Game | obs_dim | act_dim | Observations | Actions |
|------|---------|---------|-------------|---------|
| Pong | 6 | 3 | Ball position & velocity, both paddle positions | stay / up / down |
| Boxing | 14 | 18 | Both boxer positions, scores, punch/knockdown timers, time remaining | 9 directions × punch toggle |
| Space Invaders | 20 | 6 | Player position & bullet, alien formation, 3 bombs, aiming hint, wave & lives | stay·left·right × fire toggle |

#### Bundled Legacy Checkpoints

| Game | Total Steps | Best avg-50 reward | Mid checkpoint |
|------|-------------|-------------------|----------------|
| Pong | 1,000,000 | +8.04 | step 501,760 |
| Boxing | 1,000,000 | +114.156 | step 501,760 |
| Space Invaders | 1,500,000 | +50.47 | step 751,616 |

The old `si_mid.json` and `si_best.json` files are preserved as legacy SimbaV2-style checkpoints. Active Space Invaders play now loads `si_rainbow_mid.json` and `si_rainbow_best.json`, exported with `"architecture": "rainbow_feature_c51"`.

#### In-Browser Inference

No Python or PyTorch required at runtime — inference runs entirely in the browser.

```
Training (Python)                      Deployment (Browser)
─────────────────────────────          ──────────────────────────────
SimbaV2 PPO / Rainbow DQN      →   export_weights.py
PyTorch PPO/Rainbow network    →   model.json  (weight arrays)
                                →   pong-rl-agent.js (JS inference)
                                →   predict(obs) → action
```

The `PongRLAgent` class in `pong-rl-agent.js` is reused by all three games. It detects `"legacy_mlp"`, `"simba_v2_discrete"`, and `"rainbow_feature_c51"` JSON automatically.

---

### Gomoku — Alpha-Beta Minimax (Web Worker)

```
Main thread: board → bitboard conversion → postMessage to Worker
Web Worker:  priority search → Alpha-Beta Pruning → return best move
```

`gomoku-ai-worker.js` is a self-contained port of the `gomoku/` reference project.
Medium searches 6 plies; Hell searches 8 plies. An animated "AI thinking…" indicator is shown during computation.

---

### Chess — Stockfish 18 WASM (UCI Protocol)

```javascript
engine.postMessage('uci');
engine.postMessage('setoption name Skill Level value 20');
engine.postMessage('position startpos moves e2e4 ...');
engine.postMessage('go depth 15');
// → bestmove d7d5
```

`stockfish-18-lite-single.js` + `.wasm` (~7.5 MB total) are served from `/public/`.
No CORS headers required; single-threaded; works in all modern browsers.

---

### Tetris — 4-Weight Heuristic (tetrisai port)

```
score = -h × aggregateHeight + l × lines - o × holes - b × bumpiness
```

| Parameter | Medium | Hell (genetically tuned) |
|-----------|--------|--------------------------|
| h (height penalty) | 0.510 | 0.510 |
| l (lines bonus) | 0.600 | 0.761 |
| o (holes penalty) | 0.450 | 0.357 |
| b (bumpiness penalty) | 0.300 | 0.184 |

Medium uses 1-piece lookahead; Hell uses 2-piece lookahead and hard-drops instantly upon alignment.

---

## Project Structure

```
Game_web/
├── server.js                    # Node.js HTTP server (API + static files)
├── boxing-bridge.js             # Node ↔ Python bridge (legacy, unused)
├── package.json
│
├── public/                      # Web frontend
│   ├── index.html               # Single-page app (SPA)
│   ├── css/
│   │   ├── styles.css
│   │   └── game-icons.css
│   ├── js/
│   │   ├── app.js               # SPA router, auth, leaderboard
│   │   ├── ai-config.js         # Difficulty presets
│   │   ├── game-icons.js        # Game card renderer
│   │   ├── pong-rl-agent.js     # Generic RL JSON inference engine (PPO + Rainbow)
│   │   ├── gomoku-ai-worker.js  # Gomoku Alpha-Beta Worker
│   │   └── games/
│   │       ├── canvas-arena.js  # Canvas game mount helper
│   │       ├── pong-core.js     # Pong game engine
│   │       ├── pong.js          # Pong AI integration (3 PPO modes)
│   │       ├── gomoku.js        # Gomoku game + Worker AI
│   │       ├── chess.js         # Chess + Stockfish UCI integration
│   │       ├── boxing.js        # Boxing game engine + PPO integration
│   │       ├── space-invaders-core.js  # Space Invaders game engine
│   │       ├── space-invaders.js       # Dual-screen + Rainbow integration
│   │       └── tetris.js        # Dual-screen Tetris + heuristic AI (self-contained)
│   ├── models/                  # RL weight JSON files (loaded directly by browser)
│   │   ├── pong_mid.json        # Pong mid  (501,760 steps, 1.5 MB)
│   │   ├── pong_best.json       # Pong best (665,600 steps, 1.5 MB)
│   │   ├── boxing_mid.json      # Boxing mid  (501,760 steps, 1.6 MB)
│   │   ├── boxing_best.json     # Boxing best (342,016 steps, 1.6 MB)
│   │   ├── si_rainbow_mid.json  # Active SI Rainbow mid  (20,000 steps)
│   │   ├── si_rainbow_best.json # Active SI Rainbow best (22,000 steps)
│   │   ├── si_mid.json          # Legacy SI Simba checkpoint
│   │   └── si_best.json         # Legacy SI Simba checkpoint
│   ├── stockfish.js             # Stockfish 18 Lite Single WASM wrapper
│   ├── stockfish.wasm           # Stockfish WASM binary (7 MB)
│   ├── img/chesspieces/wikipedia/  # Chess piece images (local, no CDN)
│   └── jstetris/                # (Legacy) iframe Tetris source — unused
│
├── pong_rl/                     # Pong RL training package
│   ├── pong_env.py              # Game environment (mirrors JS physics)
│   ├── ppo_agent.py             # SimbaV2 PPO agent wrapper
│   ├── train.py                 # Training script
│   ├── export_weights.py        # PyTorch → JSON export
│   ├── play.py                  # Test a trained model
│   ├── models/
│   │   ├── mid_model.pt
│   │   └── best_model.pt
│   └── requirements.txt
│
├── boxing_rl/                   # Boxing RL training package
│   ├── boxing_env.py            # Game environment (mirrors JS physics)
│   ├── ppo_agent.py
│   ├── train.py
│   ├── export_weights.py
│   └── models/
│       ├── mid_model.pt
│       └── best_model.pt
│
├── space_invaders_rl/           # Space Invaders RL training package
│   ├── rainbow_agent.py         # Rainbow DQN: C51, NoisyNet, PER, Double, Dueling
│   ├── space_invaders_env.py    # Feature environment matching browser observations
│   ├── train.py                 # Paper-setting Rainbow training script
│   ├── export_weights.py        # PyTorch -> Rainbow browser JSON export
│   ├── rainbow_models/          # Active Rainbow checkpoints + summary
│   │   ├── mid_model.pt
│   │   ├── best_model.pt
│   │   └── training_summary.json
│   ├── models/                  # Legacy Space Invaders checkpoints
│   │   ├── mid_model.pt
│   │   └── best_model.pt
│   └── Rainbow/                 # Upstream Atari Rainbow reference implementation
│
├── rl_common/                   # Shared PPO implementation (Pong & Boxing)
│   ├── simba_ppo.py             # SimbaV2 ActorCritic network + PPO update
│   └── __init__.py
│
├── services/boxing/             # (Legacy) PettingZoo Boxing service — unused
│   ├── boxing_web_service.py
│   └── requirements.txt
│
└── data/
    └── db.json                  # Leaderboard + community data (JSON)
```

---

## Quick Start

### Requirements

| Item | Version |
|------|---------|
| Node.js | 18+ |
| Python | 3.10+ (only needed for RL retraining) |
| CUDA (optional) | GPU training acceleration |

### Install and Run

```bash
# 1. Install dependencies
npm install

# 2. Start the server
npm start
# or: node server.js
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

> **Note**: All games are playable immediately after starting the server.
> Chess loads the Stockfish WASM (~7 MB) on first play, which takes about 2–3 seconds.

---

## Retraining RL Models

You can retrain the SimbaV2 RL models from scratch or experiment with hyperparameters. The bundled `public/models/*.json` files may still be legacy MLP checkpoints until you run the training and export commands below.

### Python Environment

This project has been smoke-tested with the local `gameai` conda environment:

```bash
conda create -n gameai python=3.10 -y
conda activate gameai
```

If you use a different Python environment, install the dependencies first:

```bash
pip install torch numpy pygame
```

CUDA is used automatically if a compatible GPU is available. Pong and Boxing default to PPO training. Space Invaders defaults to the Rainbow paper horizon: **50,000,000 agent steps**, equivalent to **200,000,000 frames** with frame skip 4. Shorter `--steps` runs keep the Rainbow paper hyperparameters locked, but shorten only the training horizon.

### Pong

Pong uses self-play by default. The learner trains against a mixed opponent pool: rule bots, frozen historical snapshots, and best-evaluation snapshots. Use `--no-self-play` only if you want the older fixed-bot behavior.

```bash
cd pong_rl
python train.py --steps 10000000 --save-dir models \
  --min-lr-ratio 0.05 \
  --max-opponents 8 \
  --self-play-warmup-episodes 25 \
  --snapshot-interval-updates 25 \
  --eval-interval-updates 10 \
  --eval-episodes 24
python export_weights.py --save-dir models --out-dir ../public/models
cd ..
```

### Boxing

```bash
cd boxing_rl
python train.py --steps 10000000 --save-dir models --min-lr-ratio 0.05
python export_weights.py --save-dir models --out-dir ../public/models
cd ..
```

### Space Invaders

Rainbow paper-locked settings used by `space_invaders_rl/train.py`:

| Setting | Value |
|---------|-------|
| Min history before learning | 80,000 frames (`20,000` agent steps) |
| Adam learning rate / epsilon | `6.25e-5` / `1.5e-4` |
| Exploration | NoisyNet greedy policy, `epsilon=0`, `sigma0=0.5` |
| Target network period | 32,000 frames (`8,000` agent steps) |
| Prioritized replay | proportional, exponent `0.5`, beta `0.4 -> 1.0` |
| Multi-step return | `n=3`, `gamma=0.99` |
| Distributional support | 51 atoms over `[-10, 10]` |
| Replay capacity / batch | `1,000,000` / `32` |

Full paper-horizon training:

```bash
cd space_invaders_rl
python train.py --save-dir rainbow_models
python export_weights.py --save-dir rainbow_models --out-dir ../public/models --file-prefix si_rainbow
cd ..
```

Verified short run used for the currently bundled Rainbow browser files:

```bash
cd space_invaders_rl
python train.py --steps 22000 --eval-interval 1000 --eval-episodes 3 --save-dir rainbow_models
python export_weights.py --save-dir rainbow_models --out-dir ../public/models --file-prefix si_rainbow
cd ..
```

Current bundled Rainbow results:

| File | Agent steps | Frames | Notes |
|------|-------------|--------|-------|
| `si_rainbow_mid.json` | 20,000 | 80,000 | Saved at Rainbow learn-start |
| `si_rainbow_best.json` | 22,000 | 88,000 | Best 3-episode eval score: `466.67` |
| `space_invaders_rl/rainbow_models/training_summary.json` | 22,000 | 88,000 | Last-20 train score: `378.5` |

After training, refresh the browser. New exports include `"architecture": "rainbow_feature_c51"` and are loaded by the same in-browser `PongRLAgent`.

### Train All RL Models

From the repository root:

```bash
conda activate gameai

cd pong_rl
python train.py --steps 10000000 --save-dir models \
  --min-lr-ratio 0.05 \
  --max-opponents 8 \
  --self-play-warmup-episodes 25 \
  --snapshot-interval-updates 25 \
  --eval-interval-updates 10 \
  --eval-episodes 24
python export_weights.py --save-dir models --out-dir ../public/models
cd ..

cd boxing_rl
python train.py --steps 10000000 --save-dir models --min-lr-ratio 0.05
python export_weights.py --save-dir models --out-dir ../public/models
cd ..

cd space_invaders_rl
python train.py --save-dir rainbow_models
python export_weights.py --save-dir rainbow_models --out-dir ../public/models --file-prefix si_rainbow
cd ..
```

#### Checkpoint Saving

| File | Saved when |
|------|-----------|
| `mid_model.pt` | At the 50% step mark |
| `best_model.pt` | Whenever the last-50-episode average game score hits a new high |

#### Model JSON Structure

Space Invaders Rainbow exports use this structure:

```json
{
  "architecture": "rainbow_feature_c51",
  "obs_dim": 20,
  "act_dim": 6,
  "atoms": 51,
  "v_min": -10.0,
  "v_max": 10.0,
  "support": [...],
  "layers": {
    "fc": { "weight": [[...]], "bias": [...] },
    "value_hidden": { "weight": [[...]], "bias": [...] },
    "value_out": { "weight": [[...]], "bias": [...] },
    "adv_hidden": { "weight": [[...]], "bias": [...] },
    "adv_out": { "weight": [[...]], "bias": [...] }
  },
  "paper_locked": { "min_history_to_start_learning_frames": 80000 },
  "meta": { "total_steps": 22000, "training_frames": 88000, "architecture": "rainbow_feature_c51" }
}
```

SimbaV2 PPO exports for Pong and Boxing use this structure:

```json
{
  "architecture": "simba_v2_discrete",
  "obs_dim": 14,
  "act_dim": 18,
  "obs_mean": [...],
  "obs_var": [...],
  "embed": { "w": [[...]], "scale": [...] },
  "blocks": [{ "mlp": { "w1": [[...]], "scale": [...], "w2": [[...]] }, "alpha": [...] }],
  "actor": { "w1": [[...]], "scale": [...], "w2": [[...]], "bias": [...] },
  "meta": { "total_steps": 10000000, "updates": 4882, "architecture": "simba_v2_discrete" }
}
```

Legacy bundled checkpoints are still supported and look like this:

```json
{
  "architecture": "legacy_mlp",
  "shared_0_weight": [[...], ...],   // Linear(obs_dim → 256) weights
  "shared_0_bias":   [...],
  "shared_2_weight": [[...], ...],   // Linear(256 → 256) weights
  "shared_2_bias":   [...],
  "actor_weight":    [[...], ...],   // Linear(256 → act_dim) weights
  "actor_bias":      [...],
  "meta": { "total_steps": 10000000, "updates": 4882, "obs_dim": 6, "act_dim": 3 }
}
```

---

## Tech Stack

### Frontend

| Technology | Purpose |
|-----------|---------|
| Vanilla JS (ES2020+) | SPA router, all game logic |
| HTML5 Canvas | Pong, Boxing, Space Invaders, Tetris, Gomoku rendering |
| Web Workers | Gomoku AI async search |
| Fetch API | RL model JSON loading, REST API calls |
| chessboard.js + chess.js | Chess UI and rule validation (CDN) |
| p5.js | Gomoku canvas rendering (CDN) |

### Backend

| Technology | Purpose |
|-----------|---------|
| Node.js (built-in `http`) | HTTP server, static file serving |
| JSON file (`data/db.json`) | Leaderboard and community data storage |

### AI / ML

| Technology | Purpose |
|-----------|---------|
| PyTorch | PPO and Rainbow DQN agent training |
| NumPy | Environment simulation |
| Stockfish 18 WASM | Chess engine (runs in-browser) |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/signup` | Create account |
| POST | `/api/auth/login` | Log in |
| POST | `/api/auth/guest` | Guest login |
| GET | `/api/leaderboard?game=&difficulty=` | Get leaderboard |
| POST | `/api/leaderboard` | Submit score |
| DELETE | `/api/leaderboard` | Clear leaderboard |
| GET | `/api/leaderboard/personal-best?username=&game=&difficulty=` | Get user's personal best score |
| GET | `/api/community/posts` | Get posts |
| POST | `/api/community/posts` | Create post |
| POST | `/api/community/posts/:id/like` | Toggle like |
| POST | `/api/community/posts/:id/comments` | Add comment |

---

## Credits

| Component | Source | License |
|-----------|--------|---------|
| Pong game engine | [juliensimon/browser-games](https://github.com/juliensimon/browser-games) | MIT |
| Space Invaders engine | [juliensimon/browser-games](https://github.com/juliensimon/browser-games) | MIT |
| Tetris AI weights & algorithm | [LeeYiyuan/tetrisai](https://github.com/LeeYiyuan/tetrisai) | MIT |
| Gomoku AI Worker | internal reference project | — |
| Stockfish.js | [nmrugg/stockfish.js](https://github.com/nmrugg/stockfish.js) | GPL-3.0 |
| chess.js | [jhlywa/chess.js](https://github.com/jhlywa/chess.js) | BSD |
| chessboard.js | [oakmac/chessboardjs](https://github.com/oakmac/chessboardjs) | MIT |
| p5.js | [processing/p5.js](https://p5js.org/) | LGPL |

The trademarks of the original arcade games (Pong, Space Invaders, etc.) belong to their respective rights holders.
