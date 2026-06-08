/**
 * Pong (Atari, 1972) — browser-games canvas port
 *
 * AI difficulty modes:
 *   easy   → random policy  (pure random paddle movement)
 *   medium → mid_model.pt   (PPO checkpoint at ~50% training)
 *   hell   → best_model.pt  (PPO checkpoint with highest reward)
 */

// Pre-load RL agents once and cache them globally.
window._pongAgents = window._pongAgents || { mid: null, best: null };

async function _loadPongAgent(key, url) {
  if (window._pongAgents[key]) return window._pongAgents[key];
  try {
    const agent = new window.PongRLAgent();
    await agent.load(url);
    window._pongAgents[key] = agent;
    return agent;
  } catch (e) {
    console.warn(`[Pong] Could not load RL model (${url}):`, e);
    return null;
  }
}

const PongGame = {
  id: "pong",
  title: "Pong",

  mount(container, ctx) {
    const { Game, CONFIG } = window.PongBrowserGame;
    const difficulty = ctx.difficulty || "medium";
    let finished = false;
    let gameHandle = null;

    const mountWithAI = (aiOptions) => {
      gameHandle = CanvasArena.mountSingle(container, {
        title: "You vs AI",
        aspectRatio: "8 / 5",
        createGame(canvas) {
          canvas.width = CONFIG.WIDTH;
          canvas.height = CONFIG.HEIGHT;

          const game = new Game(canvas, {
            autoStart: true,
            ai: aiOptions,
            hooks: {
              onScore(playerScore, aiScore) {
                ctx.onScore(playerScore, aiScore);
              },
              onGameOver(winner, playerScore, aiScore) {
                if (finished) return;
                finished = true;
                ctx.onEnd({
                  playerScore,
                  aiScore,
                  message:
                    winner === 1
                      ? "You win! (Atari Pong, 1972)"
                      : "AI wins the match.",
                });
              },
            },
          });

          const loop = CanvasArena.runLoop(game, () => game.render(), CONFIG);

          return {
            destroy() {
              loop.stop();
              game.destroy();
            },
          };
        },
      });
      this._handle = gameHandle;
    };

    if (difficulty === "easy") {
      // Random policy — no model loading needed
      mountWithAI.call(this, { mode: "random" });
    } else if (difficulty === "medium") {
      _loadPongAgent("mid", "/models/pong_mid.json").then((agent) => {
        mountWithAI.call(this, { mode: agent ? "ppo" : "rule", rlAgent: agent,
          speedMult: 0.8, mistakeRate: 0.15, assistRate: 0.55 });
      });
    } else {
      // hell — best model
      _loadPongAgent("best", "/models/pong_best.json").then((agent) => {
        mountWithAI.call(this, { mode: agent ? "ppo" : "rule", rlAgent: agent,
          speedMult: 1.15, mistakeRate: 0.03, assistRate: 0.92 });
      });
    }
  },

  unmount() {
    this._handle?.unmount();
  },
};
