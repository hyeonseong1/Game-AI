/**
 * Space Invaders (Taito, 1978) — Dual-Screen PvAI
 *
 * Left screen  : Player 1 (human, keyboard)
 * Right screen : Player 2 (PPO AI agent)
 *
 * Both games run simultaneously in real-time.
 *
 * AI difficulty → model:
 *   easy   — random policy
 *   medium — si_mid.json  (PPO ~50% training)
 *   hell   — si_best.json (PPO best reward)
 */

// ── Observation builder (mirrors space_invaders_env.py) ────────────────────
function buildSIObs(gameState, game) {
    const W = 224, H = 256;
    const p = gameState.player;
    if (!p) return null;

    const pcx = p.x + 6.5;
    const bl  = gameState.playerBullet && gameState.playerBullet.alive;

    const f = gameState.formation;
    let left = W, right = 0, bottom = 0, cxSum = 0, nAlive = 0;
    for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 11; c++) {
            const a = f.aliens[r][c];
            if (!a.alive) continue;
            const pos = f.getAlienPos(r, c);
            const aw  = f.getAlienWidth(r);
            if (pos.x < left)       left   = pos.x;
            if (pos.x + aw > right) right  = pos.x + aw;
            if (pos.y + 8 > bottom) bottom = pos.y + 8;
            cxSum += pos.x + aw / 2;
            nAlive++;
        }
    }
    if (nAlive === 0) return null;
    const fcx = cxSum / nAlive;

    let nearestDx = 0, nearestD = W;
    for (let c = 0; c < 11; c++) {
        for (let r = 4; r >= 0; r--) {
            if (!f.aliens[r][c].alive) continue;
            const pos = f.getAlienPos(r, c);
            const acx = pos.x + f.getAlienWidth(r) / 2;
            const d   = Math.abs(acx - pcx);
            if (d < nearestD) { nearestD = d; nearestDx = acx - pcx; }
            break;
        }
    }

    const bombs = (gameState.alienBombs || [])
        .filter(b => b.alive)
        .sort((a, b) => b.y - a.y)
        .slice(0, 3);

    return [
        pcx / W,
        bl ? 1 : 0,
        bl ? gameState.playerBullet.x / W : 0,
        bl ? gameState.playerBullet.y / H : 0,
        nAlive / 55,
        left / W, right / W, bottom / H, fcx / W,
        f.direction > 0 ? 1 : 0,
        bombs[0] ? (bombs[0].x - pcx) / W : 0, bombs[0] ? bombs[0].y / H : 0,
        bombs[1] ? (bombs[1].x - pcx) / W : 0, bombs[1] ? bombs[1].y / H : 0,
        bombs[2] ? (bombs[2].x - pcx) / W : 0, bombs[2] ? bombs[2].y / H : 0,
        nearestDx / W,
        Math.min(game.wave, 10) / 10,
        game.lives / 3,
        gameState.mysteryShip && gameState.mysteryShip.alive
            ? gameState.mysteryShip.x / W : 0,
    ];
}

// action → {left, right, fire}  (0=stay 1=L 2=R 3=fire 4=fire+L 5=fire+R)
function decodeSIAction(action) {
    return { left: action===1||action===4, right: action===2||action===5, fire: action>=3 };
}

// ── Agent cache ────────────────────────────────────────────────────────────
window._siAgents = window._siAgents || {};
async function loadSIAgent(key, url) {
    if (window._siAgents[key]) return window._siAgents[key];
    try {
        const agent = new window.PongRLAgent();
        await agent.load(url);
        window._siAgents[key] = agent;
        return agent;
    } catch (e) { console.warn('[SI]', e); return null; }
}

// Random policy controller
function makeRandomCtrl() {
    let t = 0, act = 0;
    return () => { if (--t <= 0) { t = 8 + Math.floor(Math.random()*12); act = Math.floor(Math.random()*6); } return decodeSIAction(act); };
}

// ── Dual-screen mount helper ───────────────────────────────────────────────
function mountDualSI(container, ctx, aiController) {
    const { Game, CONFIG } = window.SpaceInvadersBrowserGame;
    let p1Score = 0, p2Score = 0;
    let p1Done = false, p2Done = false;

    const checkBothDone = () => {
        if (p1Done && p2Done) {
            ctx.onEnd({
                playerScore: p1Score,
                aiScore:     p2Score,
                message: p1Score > p2Score
                    ? `You win! You ${p1Score} — AI ${p2Score}`
                    : p2Score > p1Score
                    ? `AI wins! AI ${p2Score} — You ${p1Score}`
                    : `Draw! Both scored ${p1Score}`,
            });
        }
    };

    // ── Layout ──────────────────────────────────────────────────────────
    container.innerHTML = `
      <div class="si-dual-wrap">
        <div class="si-panel" id="si-p1">
          <div class="si-label">🎮 You (Player 1)</div>
          <canvas id="si-canvas-p1" class="browser-game-canvas" style="aspect-ratio:7/8;width:100%"></canvas>
        </div>
        <div class="si-divider">VS</div>
        <div class="si-panel" id="si-p2">
          <div class="si-label">🤖 AI (Player 2)</div>
          <canvas id="si-canvas-p2" class="browser-game-canvas" style="aspect-ratio:7/8;width:100%"></canvas>
        </div>
      </div>
      <style>
        .si-dual-wrap{display:flex;align-items:flex-start;gap:8px;width:100%;box-sizing:border-box}
        .si-panel{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center}
        .si-label{font-size:13px;font-weight:600;color:#aaa;margin-bottom:4px;letter-spacing:.5px}
        .si-divider{align-self:center;font-size:18px;font-weight:700;color:#555;padding:0 4px;flex-shrink:0}
      </style>`;

    // ── Player 1 game (human keyboard) ──────────────────────────────────
    const c1 = document.getElementById('si-canvas-p1');
    c1.width = CONFIG.WIDTH; c1.height = CONFIG.HEIGHT;
    const g1 = new Game(c1, {
        autoStart: true,
        hooks: {
            onScore(score) { p1Score = score; ctx.onScore(p1Score, p2Score); },
            onGameOver(score, hi, wave) {
                p1Score = score;
                if (p1Done) return; p1Done = true;
                ctx.onScore(p1Score, p2Score);
                checkBothDone();
            },
        },
    });
    const l1 = CanvasArena.runLoop(g1, () => g1.renderer.render(g1.getState()), CONFIG);

    // ── Player 2 game (AI) ───────────────────────────────────────────────
    const c2 = document.getElementById('si-canvas-p2');
    c2.width = CONFIG.WIDTH; c2.height = CONFIG.HEIGHT;
    const g2 = new Game(c2, {
        autoStart: true,
        hooks: {
            onScore(score) { p2Score = score; ctx.onScore(p1Score, p2Score); },
            onGameOver(score, hi, wave) {
                p2Score = score;
                if (p2Done) return; p2Done = true;
                ctx.onScore(p1Score, p2Score);
                checkBothDone();
            },
        },
    });

    // Attach AI controller to g2
    if (aiController === 'random') {
        const ctrl = makeRandomCtrl();
        g2._aiInput = () => ctrl();
    } else if (aiController) {
        // Throttle: decide every 8 frames (~133 ms at 60 fps)
        let siFrame = 0;
        let siLastCtrl = null;
        g2._aiInput = (state) => {
            siFrame = (siFrame + 1) % 8;
            if (siFrame !== 0) return siLastCtrl;
            const obs = buildSIObs(state, g2);
            if (!obs) return siLastCtrl;
            siLastCtrl = decodeSIAction(aiController.predict(obs));
            return siLastCtrl;
        };
    }

    const l2 = CanvasArena.runLoop(g2, () => g2.renderer.render(g2.getState()), CONFIG);

    ctx.onScore(0, 0);

    return {
        unmount() {
            l1.stop(); l2.stop();
            g1.destroy(); g2.destroy();
            container.innerHTML = '';
        },
    };
}

// ── Public game object ─────────────────────────────────────────────────────
const SpaceInvadersGame = {
    id: "space-invaders",
    title: "Space Invaders",

    mount(container, ctx) {
        const difficulty = ctx.difficulty || 'medium';

        if (difficulty === 'easy') {
            this._runtime = mountDualSI(container, ctx, 'random');
            return;
        }

        const url = difficulty === 'hell' ? '/models/si_best.json' : '/models/si_mid.json';
        container.innerHTML =
            '<div style="text-align:center;padding:60px;color:#aaa;font-size:14px">' +
            'Loading PPO model…</div>';

        loadSIAgent(difficulty, url).then(agent => {
            this._runtime = mountDualSI(container, ctx, agent);
        });
    },

    unmount() {
        this._runtime?.unmount();
        this._runtime = null;
    },
};
