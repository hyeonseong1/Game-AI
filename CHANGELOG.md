# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — 2026-05-31

### Major AI Upgrade Release

The core of this release is the introduction of **PPO reinforcement learning agents** and a shift to a **dual-screen battle structure**.
All 6 games received AI replacements or enhancements, and a complete in-browser neural network inference pipeline was built.

---

### Pong — PPO Reinforcement Learning AI

**Before**: Simple rule-based AI adjusting paddle speed via `speedMult` and `mistakeRate` parameters.

**After**: A trained PPO neural network observes the game state each frame and decides the paddle action.

#### Added
- `pong_rl/` package
  - `pong_env.py`: Python simulation mirroring JS game physics (900×600, `PADDLE_SPEED=7`, `BALL_R=8`)
  - `ppo_agent.py`: `ActorCritic` network + `RolloutBuffer` + PPO update loop
  - `train.py`: 1,000,000-step training; auto-saves mid (50%) and best (highest reward) checkpoints
  - `export_weights.py`: Converts PyTorch `.pt` → JSON for browser serving
- `public/js/pong-rl-agent.js`: Generic ActorCritic JS inference engine (`RLAgent` class)
  - Loads model JSON via `fetch`; runs `predict(obs) → action`
  - Pure JS forward pass (Linear + Tanh) — no external dependencies
- `public/models/pong_mid.json` (1.5 MB, step 501,760)
- `public/models/pong_best.json` (1.5 MB, step 665,600)

#### Changed
- `public/js/games/pong-core.js`: Added 3 modes to `AIController`
  - `mode: 'random'` — Easy: random direction each frame
  - `mode: 'ppo'` — Medium/Hell: JS inference engine call
  - `mode: 'rule'` — legacy rule-based fallback (when model fails to load)
  - `_buildObs()`: normalises game coordinates to `[-1, 1]` observation vector
  - `update(ball, paddle2, paddle1)` signature extended with `paddle1` (for opponent paddle observation)
- `public/js/games/pong.js`: Async model loading per difficulty + `window._pongAgents` cache

#### Observation Space (6-dim)

| Index | Meaning | Range |
|-------|---------|-------|
| 0 | Ball X (normalised) | [-1, 1] |
| 1 | Ball Y (normalised) | [-1, 1] |
| 2 | Ball VX / max_speed | [-1, 1] |
| 3 | Ball VY / max_speed | [-1, 1] |
| 4 | AI paddle Y (normalised) | [-1, 1] |
| 5 | Player paddle Y (normalised) | [-1, 1] |

#### Action Space (3)
`0=stay`, `1=up`, `2=down`

---

### Gomoku — Web Worker Alpha-Beta Minimax AI

**Before**: Simple greedy heuristic using `scoreCell()`, running inline on the main thread.

**After**: Bitboard representation + Alpha-Beta Pruning + async Web Worker.

#### Added
- `public/js/gomoku-ai-worker.js`: Self-contained port of the `gomoku/` reference project AI Worker
  - Bitboard representation (eight 32-bit integers = 256 bits for a 15×15 board)
  - Immediate threat detection (`checkImmediateThreat`), open-4 patterns, open-3 patterns
  - Alpha-Beta Minimax: 6-ply for Medium, 8-ply for Hell
  - Message protocol: `FIND_BEST_MOVE` / `BEST_MOVE_FOUND` / `NEW_GAME`

#### Changed
- `public/js/games/gomoku.js`: Full rewrite
  - `board2Bitboards()` inline: converts `board[r][c]` (0/1/2) → `{blackBitboard, whiteBitboard}`
  - Async Worker communication: player move → post Worker message → receive `BEST_MOVE_FOUND` → place AI stone
  - "AI thinking…" animated indicator (animated dots via p5.js `frameCount`)
  - Star points (hoshi) rendered on the board; improved stone highlight rendering
  - Easy uses in-thread greedy (instant); Medium/Hell use Worker
  - **Random color assignment**: each game randomly assigns Black or White to the player; if player is White, AI (Black) moves first
  - **Win detection bug fixed**: removed erroneous `-1` from `checkWin` — 5-in-a-row now correctly returns 5 instead of 4

---

### Chess — Stockfish 18 WASM AI

**Before**: Random move selection or capture-priority random selection.

**After**: Stockfish 18 chess engine via UCI protocol (WASM).

#### Added
- `public/stockfish.js` (21 KB) — Stockfish 18 Lite Single-threaded WASM wrapper
- `public/stockfish.wasm` (7.0 MB) — Stockfish WASM binary
- `public/img/chesspieces/wikipedia/*.png` (12 files) — chess piece images served locally (no CDN dependency)

#### Changed
- `server.js`: Added `.wasm` MIME type (`application/wasm`)
- `public/js/games/chess.js`: Full rewrite
  - `new Worker('/stockfish.js')` initialises Stockfish UCI engine
  - `uci` → `readyok` handshake before play begins
  - UCI command sequence: `setoption name Skill Level value {skill}` → `position startpos moves {history}` → `go depth {depth}` → parse `bestmove {move}`
  - Applies `enginegame.js` pattern (Skill Level · Maximum Error · Probability coordination)
  - Detects checkmate, draw, and stalemate; handles game-over correctly
  - Piece images served from `/img/chesspieces/wikipedia/{piece}.png`

| Difficulty | Skill Level | Depth |
|------------|-------------|-------|
| Easy | 3 | 2 |
| Medium | 10 | 8 |
| Hell | 20 | 15 |

---

### Boxing — PPO Reinforcement Learning AI

**Before**: PettingZoo `boxing_v2` Atari environment via Python backend (`boxing_web_service.py` + `boxing-bridge.js`). In practice, this had already been replaced by a pure Canvas game.

**After**: Pure Canvas game + PPO neural network AI (in-browser inference).

#### Added
- `boxing_rl/` package
  - `boxing_env.py`: Python 60fps simulation mirroring `boxing.js` physics
    - Agent (right boxer) vs rule-based opponent (left boxer)
    - Radius 16px, punch range 60px, knockdown 1800ms
    - Observation: 14-dim (positions, scores, punch/knockdown timers, time remaining)
    - Actions: 18 (9 movement directions × punch toggle)
  - `ppo_agent.py`, `train.py`, `export_weights.py`
- `public/models/boxing_mid.json` (1.6 MB, step 501,760)
- `public/models/boxing_best.json` (1.6 MB, step 342,016; best avg reward +114.156)

#### Changed
- `public/js/games/boxing.js`: PPO integration
  - `loadBoxingAgent(key, url)`: reuses `window.PongRLAgent` (generic inference engine)
  - `buildObs(ai, player, timeLeft)`: builds 14-dim observation vector
  - `decodeAction(action)`: maps action → `{vx, vy, punch}`
  - `aiInputPPO(dt)`: throttled to `PPO_INTERVAL` ms (Hell: 160ms, Medium: 220ms) — previously ran every frame
  - `ppoPunchPending` flag: punch triggers only on new decisions, preventing repeated inputs
  - `aiInputRandom()`: Easy difficulty random policy
  - Shows "Loading PPO model…" screen while fetching model; falls back to rule-based AI on load failure
  - **Bug fix** — `HitEffect.draw()`: clamped `alpha = Math.max(0, life/maxLife)` to prevent negative arc radius

#### Observation Space (14-dim)

| Index | Meaning |
|-------|---------|
| 0–1 | Agent X·Y (normalised to ring centre) |
| 2–3 | Opponent X·Y |
| 4–5 | Relative distance to opponent (dx/RING_W, dy/RING_H) |
| 6 | Euclidean distance between boxers / max distance |
| 7–8 | Agent score / 100, Opponent score / 100 |
| 9–10 | Agent · Opponent punch timer / MAX |
| 11–12 | Agent · Opponent knockdown timer / MAX |
| 13 | Time remaining / total game time |

---

### Space Invaders — PPO AI + Dual-Screen Battle Structure

**Before**: Single canvas, human player only, score compared against a fixed benchmark (Easy: 500, Medium: 1500, Hell: 3500).

**After**: Dual-screen layout (human Player 1 left vs PPO AI Player 2 right), both running simultaneously.

#### Added
- `space_invaders_rl/` package
  - `space_invaders_env.py`: Python 60fps simulation mirroring `space-invaders-core.js` physics
    - Alien formation marching (variable speed), bomb firing (up to 3 at once), player bullet (1 at a time)
    - Observation: 20-dim (player position & bullet, formation bounds & direction, 3 bombs, aiming hint, wave & lives)
    - Actions: 6 (Stay · Left · Right × fire toggle)
  - `ppo_agent.py`, `train.py` (1,500,000 steps), `export_weights.py`
- `public/models/si_mid.json` (1.5 MB, step 751,616)
- `public/models/si_best.json` (1.5 MB, step 1,433,600; best avg reward +50.47)

#### Changed
- `public/js/games/space-invaders-core.js`: Added `_aiInput` hook (3 lines)
  - In `updatePlaying()`, intercepts keyboard input with `ctrl = this._aiInput?.(this.getState())`
  - `ctrl.left · right · fire` directly control the player ship for AI
- `public/js/games/space-invaders.js`: Full rewrite
  - `mountDualSI(container, ctx, aiController)`: creates two `Game` instances side by side
  - `buildSIObs(gameState, game)`: 20-dim observation (formation bounds, bombs, aiming hint)
  - `decodeSIAction(action)`: maps action → `{left, right, fire}`
  - `makeRandomCtrl()`: Easy difficulty random policy controller
  - **AI throttled to every 8 frames (~133ms)** — previously ran every frame
  - Final score comparison calls `ctx.onEnd()` when both games finish

#### Observation Space (20-dim)

| Index | Meaning |
|-------|---------|
| 0 | Player centre X / W |
| 1 | Bullet active (0/1) |
| 2–3 | Bullet X · Y (0 if inactive) |
| 4 | Alive aliens / 55 |
| 5–6 | Formation left · right boundary X / W |
| 7 | Formation bottom Y / H |
| 8 | Formation centre X / W |
| 9 | Formation direction (0=left, 1=right) |
| 10–15 | 3 bombs: [dx/W, y/H] each (0,0 if absent) |
| 16 | dx to nearest alien column / W |
| 17 | Current wave / 10 |
| 18 | Lives remaining / 3 |
| 19 | Mystery ship X / W (0 if absent) |

---

### Tetris — Heuristic AI + Dual-Screen Battle Structure

**Before**: jstetris embed (`public/jstetris/`) loaded via `<iframe>`; single player.

**After**: iframe removed; single Canvas renders both game instances side by side.

#### Added
- `public/js/games/tetris.js`: Full rewrite (self-contained, ~600 lines)
  - `Grid` class: port of `tetrisai/js/grid.js` (aggregateHeight · lines · holes · bumpiness evaluation)
  - `Piece` class: 7 tetrominoes, `rotateCW()`, wall kick, `moveLeft/Right/Down()`
  - `RPG` (Random Piece Generator): 7-bag shuffle randomiser
  - `aiBest()`: recursive search for optimal placement (port of `tetrisai/js/ai.js`)
  - `TetrisEngine`: game state management (gravity · locking · line clearing · level-up)
  - `renderBoard()` / `renderSidePanel()` / `_drawCell()`: single Canvas renderer (3D highlight cells)
  - `mountDualTetris()`: two engine instances rendered in parallel on one 600×480px Canvas
  - Ghost piece (drop preview), DAS/ARR key auto-repeat
  - **AI speed by difficulty**: action interval — Easy 8 frames (~133ms), Medium 5 frames (~83ms), Hell 2 frames (~33ms)
  - **Hell drop speed**: `gravMult = 0.15` (very fast fall during positioning) + instant hard drop on alignment

| Difficulty | Method | Lookahead | Weights (h/l/o/b) | Drop behaviour |
|------------|--------|-----------|-------------------|----------------|
| Easy | Random placement | — | — | Gravity (0.80×) |
| Medium | Heuristic | 1-piece | 0.510/0.600/0.450/0.300 | Gravity (0.65×) |
| Hell | Heuristic | 2-piece | 0.510/0.761/0.357/0.184 | Fast gravity (0.15×) + instant hard drop |

---

### Common Infrastructure Changes

#### Added
- `server.js`: Registered `.wasm` MIME type (`application/wasm`)
- `public/index.html`: Added `<script src="/js/pong-rl-agent.js" defer>`

#### Bug Fixes
- **Boxing** `HitEffect.draw()`: `life` going negative in the same frame as `update()` caused `p.r * alpha < 0`, triggering an invalid arc radius error → fixed with `Math.max(0, life / maxLife)`
- **Gomoku** `checkWin()`: erroneous `-1` caused 5-in-a-row to evaluate as 4, so the game never ended → removed `-1` to correctly evaluate as 5

---

## [1.0.0] — Initial Release

### Added

- **6 classic games** — initial implementation
  - **Pong**: Canvas game based on `pong-core.js` (rule-based AI)
  - **Space Invaders**: Canvas game based on `space-invaders-core.js` (single player; score benchmark)
  - **Tetris**: jstetris iframe embed (`public/jstetris/`)
  - **Boxing**: Pure Canvas top-view boxing (rule-based AI with speedMult · mistakeRate · reactionMs)
  - **Chess**: chess.js + chessboard.js; random/capture-priority AI
  - **Gomoku**: p5.js Canvas; simple greedy heuristic AI

- **SPA infrastructure** (`public/js/app.js`)
  - View router: login → lobby → difficulty selection → game → result
  - Signup · login · guest session

- **Leaderboard** (`GET/POST/DELETE /api/leaderboard`)
  - Per-game and per-difficulty filtering
  - AI comparison percentage calculation

- **Community** (`/api/community/posts`)
  - Create and list posts, toggle likes, add comments

- **Data store**: `data/db.json` (JSON file)

- **Server**: `server.js` — Node.js built-in `http` module (no frameworks)

- **AI difficulty system** (`public/js/ai-config.js`)
  - 3-tier parameter table: Easy / Medium / Hell
  - `getDifficulty(key)`, `shouldAiMistake(difficulty)` helpers

- **Legacy Boxing service** (`services/boxing/`)
  - PettingZoo `boxing_v2` (Atari ROM) Python environment
  - JSON-lines RPC (`boxing_web_service.py` ↔ `boxing-bridge.js`)
  - Not used in current version (replaced by Canvas game)
