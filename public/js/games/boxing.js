/**
 * Boxing — Pure Canvas browser game
 * Top-view Atari-style boxing ring. Player (white) vs AI (black).
 * Controls: WASD / Arrow keys = move, K / L / Space = punch
 *
 * AI modes by difficulty:
 *   easy   — random policy  (random movement + occasional punch)
 *   medium — PPO mid_model  (boxing_mid.json, ~50% training)
 *   hell   — PPO best_model (boxing_best.json, highest reward)
 */
const BoxingGame = (function () {
  "use strict";

  // ── Constants (must match boxing_env.py exactly) ───────────────────────────
  const W = 560, H = 400;
  const RING = { x: 40, y: 40, w: W - 80, h: H - 80 };
  const PLAYER_R   = 16;
  const PUNCH_RANGE = 44;
  const HIT_RANGE   = PUNCH_RANGE + PLAYER_R;   // 60 px
  const PUNCH_MS    = 220;
  const KNOCKDOWN_MS = 1800;
  const MAX_SCORE   = 100;
  const GAME_SECONDS = 120;
  const HIT_FLASH_MS = 180;

  // Observation normalisation constants (match boxing_env.py)
  const CX = RING.x + RING.w / 2;   // 280
  const CY = RING.y + RING.h / 2;   // 200
  const RW2 = RING.w / 2;            // 240
  const RH2 = RING.h / 2;            // 160
  const MAX_DIST = Math.hypot(RING.w, RING.h);  // 576.9

  // 9 movement directions × 2 (no-punch / punch) = 18 actions
  const MOVE_DELTA = [
    [0,      0     ],   // 0 stay
    [-1,     0     ],   // 1 L
    [+1,     0     ],   // 2 R
    [0,     -1     ],   // 3 U
    [0,     +1     ],   // 4 D
    [-0.707, -0.707],   // 5 LU
    [-0.707, +0.707],   // 6 LD
    [+0.707, -0.707],   // 7 RU
    [+0.707, +0.707],   // 8 RD
  ];
  const SPEED = 2.2;

  // ── Helpers ────────────────────────────────────────────────────────────────
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  function lerp(a, b, t) { return a + (b - a) * t; }

  // ── Build PPO observation (14 floats) ──────────────────────────────────────
  function buildObs(ai, player, timeLeft) {
    const dx = player.x - ai.x;
    const dy = player.y - ai.y;
    const d  = Math.hypot(dx, dy);
    return [
      (ai.x     - CX) / RW2,
      (ai.y     - CY) / RH2,
      (player.x - CX) / RW2,
      (player.y - CY) / RH2,
      dx / RING.w,
      dy / RING.h,
      d  / MAX_DIST,
      ai.score     / MAX_SCORE,
      player.score / MAX_SCORE,
      ai.punching     ? ai.punchTimer     / PUNCH_MS    : 0,
      player.punching ? player.punchTimer / PUNCH_MS    : 0,
      ai.knockedDown  ? ai.knockTimer     / KNOCKDOWN_MS : 0,
      player.knockedDown ? player.knockTimer / KNOCKDOWN_MS : 0,
      timeLeft / (GAME_SECONDS * 1000),
    ];
  }

  // Decode action → { vx, vy, punch }
  function decodeAction(action) {
    const moveIdx = action % 9;
    const punch   = action >= 9;
    const [dx, dy] = MOVE_DELTA[moveIdx];
    return { vx: dx * SPEED, vy: dy * SPEED, punch };
  }

  // ── Boxer class ────────────────────────────────────────────────────────────
  function Boxer(x, y, color, isPlayer) {
    this.x = x; this.y = y;
    this.color = color;
    this.isPlayer = isPlayer;
    this.vx = 0; this.vy = 0;
    this.score = 0;
    this.punching   = false;
    this.punchTimer = 0;
    this.punchAngle = 0;
    this.knockedDown = false;
    this.knockTimer  = 0;
    this.hitFlash    = 0;
    this.faceAngle   = isPlayer ? 0 : Math.PI;
  }

  Boxer.prototype.startPunch = function (targetAngle) {
    if (this.punching || this.knockedDown) return false;
    this.punching   = true;
    this.punchTimer = PUNCH_MS;
    this.punchAngle = targetAngle;
    return true;
  };

  Boxer.prototype.update = function (dt) {
    if (this.knockedDown) {
      this.knockTimer -= dt;
      if (this.knockTimer <= 0) this.knockedDown = false;
      return;
    }
    if (this.punching) {
      this.punchTimer -= dt;
      if (this.punchTimer <= 0) this.punching = false;
    }
    if (this.hitFlash > 0) this.hitFlash -= dt;
    this.x = clamp(this.x + this.vx, RING.x + PLAYER_R, RING.x + RING.w - PLAYER_R);
    this.y = clamp(this.y + this.vy, RING.y + PLAYER_R, RING.y + RING.h - PLAYER_R);
  };

  Boxer.prototype.draw = function (ctx) {
    const t = this.punching ? (1 - this.punchTimer / PUNCH_MS) : 0;
    const knockScale = this.knockedDown ? 0.7 : 1;
    if (this.hitFlash > 0) {
      ctx.save();
      ctx.globalAlpha = (this.hitFlash / HIT_FLASH_MS) * 0.6;
      ctx.fillStyle = "#ff4444";
      ctx.beginPath();
      ctx.arc(this.x, this.y, PLAYER_R * 1.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = "#000";
    ctx.beginPath();
    ctx.ellipse(this.x + 3, this.y + 5, PLAYER_R * knockScale, PLAYER_R * 0.5 * knockScale, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    ctx.save();
    ctx.translate(this.x, this.y);
    if (this.knockedDown) { ctx.rotate(Math.PI / 2); ctx.scale(knockScale, knockScale); }
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.ellipse(0, 2, PLAYER_R - 2, PLAYER_R, 0, 0, Math.PI * 2);
    ctx.fill();
    const headColor = this.color === "#ffffff" ? "#ffe0c0" : "#8B6060";
    ctx.fillStyle = headColor;
    ctx.beginPath();
    ctx.arc(0, -PLAYER_R + 2, PLAYER_R * 0.55, 0, Math.PI * 2);
    ctx.fill();
    const gloveColor = this.isPlayer ? "#e63946" : "#457b9d";
    const gloveR = 7;
    const lAngle = this.faceAngle + Math.PI * 0.6;
    ctx.fillStyle = gloveColor;
    ctx.beginPath();
    ctx.arc(Math.cos(lAngle) * (PLAYER_R - 2), Math.sin(lAngle) * (PLAYER_R - 2), gloveR, 0, Math.PI * 2);
    ctx.fill();
    const baseRAngle = this.faceAngle - Math.PI * 0.6;
    const punchExtend = this.punching ? lerp(0, PUNCH_RANGE - PLAYER_R - 4, Math.sin(t * Math.PI)) : 0;
    const rGx = Math.cos(this.punching ? this.punchAngle : baseRAngle) * (PLAYER_R - 2 + punchExtend);
    const rGy = Math.sin(this.punching ? this.punchAngle : baseRAngle) * (PLAYER_R - 2 + punchExtend);
    ctx.fillStyle = gloveColor;
    ctx.beginPath();
    ctx.arc(rGx, rGy, gloveR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    if (this.knockedDown) {
      const starCount = 3;
      const elapsed = (KNOCKDOWN_MS - this.knockTimer) / KNOCKDOWN_MS;
      for (let i = 0; i < starCount; i++) {
        const a  = (i / starCount) * Math.PI * 2 + elapsed * 6;
        const sx = this.x + Math.cos(a) * 24;
        const sy = this.y - PLAYER_R - 8 + Math.sin(a * 2) * 4;
        ctx.fillStyle = "#FFD700";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("★", sx, sy);
      }
    }
  };

  // ── Hit effect ─────────────────────────────────────────────────────────────
  function HitEffect(x, y) {
    this.x = x; this.y = y;
    this.life = this.maxLife = 320;
    this.particles = Array.from({ length: 8 }, () => ({
      vx: (Math.random() - 0.5) * 5, vy: (Math.random() - 0.5) * 5,
      r: 3 + Math.random() * 4,
    }));
  }
  HitEffect.prototype.update = function (dt) { this.life -= dt; };
  HitEffect.prototype.done   = function ()    { return this.life <= 0; };
  HitEffect.prototype.draw   = function (ctx) {
    const alpha    = Math.max(0, this.life / this.maxLife);
    const progress = 1 - alpha;
    ctx.save();
    ctx.globalAlpha = alpha * 0.9;
    this.particles.forEach((p, i) => {
      ctx.fillStyle = i % 2 === 0 ? "#FFD700" : "#ff4444";
      ctx.beginPath();
      ctx.arc(this.x + p.vx * progress * 30, this.y + p.vy * progress * 30, p.r * alpha, 0, Math.PI * 2);
      ctx.fill();
    });
    if (alpha > 0.5) {
      ctx.globalAlpha = (alpha - 0.5) * 2;
      ctx.fillStyle = "#ff4444";
      ctx.font = `bold ${Math.round(14 + progress * 8)}px sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("POW!", this.x, this.y - 14);
    }
    ctx.restore();
  };

  // ── Hit detection (shared) ─────────────────────────────────────────────────
  function checkHit(attacker, defender, onScore) {
    if (defender.knockedDown) return;
    if (dist(attacker, defender) > HIT_RANGE) return;
    defender.hitFlash = HIT_FLASH_MS;
    attacker.score = Math.min(MAX_SCORE, attacker.score + 1);
    onScore(attacker.score, defender.score);
    const knockProb = attacker.isPlayer ? 0.04 : 0.05;
    if (Math.random() < knockProb) {
      defender.knockedDown = true;
      defender.knockTimer  = KNOCKDOWN_MS;
    }
    return true;
  }

  // ── BoxingEngine ───────────────────────────────────────────────────────────
  function BoxingEngine(container, ctx, difficulty, aiMode) {
    // aiMode: { type: 'random' } | { type: 'rule' } | { type: 'ppo', agent }

    // ── Canvas setup ──
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    canvas.style.cssText = "max-width:100%;display:block;margin:0 auto;border-radius:8px";
    const wrap = document.createElement("div");
    wrap.className = "canvas-game-wrap";
    wrap.appendChild(canvas);
    container.appendChild(wrap);
    const c = canvas.getContext("2d");

    // ── State ──
    const player = new Boxer(RING.x + RING.w * 0.25, RING.y + RING.h / 2, "#ffffff", true);
    const ai     = new Boxer(RING.x + RING.w * 0.75, RING.y + RING.h / 2, "#222222", false);
    player.faceAngle = 0;
    ai.faceAngle = Math.PI;

    const keys = {};
    let timeLeft  = GAME_SECONDS * 1000;
    let gameOver  = false;
    let hitEffects = [];
    let lastTime  = null;
    let rafId     = 0;

    // Rule-based AI state
    let aiReactTimer = 0;
    let aiTargetX    = ai.x, aiTargetY = ai.y;
    const ruleCfg = { reactionMs: 500, speed: 1.8, punchChance: 0.022, mistakeRate: 0.25 };

    // PPO AI throttle state
    let ppoDt = 0;
    let ppoVx = 0, ppoVy = 0, ppoPunchPending = false;
    const PPO_INTERVAL = difficulty === 'hell' ? 160 : 220; // ms between decisions

    // ── Input ──────────────────────────────────────────────────────────────
    function onKey(e, down) {
      keys[e.code] = down;
      if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].includes(e.code)) e.preventDefault();
    }
    window.addEventListener("keydown", e => onKey(e, true));
    window.addEventListener("keyup",   e => onKey(e, false));

    function playerInput() {
      if (player.knockedDown) { player.vx = player.vy = 0; return; }
      let vx = 0, vy = 0;
      if (keys.ArrowLeft  || keys.KeyA) vx -= SPEED;
      if (keys.ArrowRight || keys.KeyD) vx += SPEED;
      if (keys.ArrowUp    || keys.KeyW) vy -= SPEED;
      if (keys.ArrowDown  || keys.KeyS) vy += SPEED;
      if (vx !== 0 && vy !== 0) { vx *= 0.707; vy *= 0.707; }
      player.vx = vx; player.vy = vy;
      const angle = Math.atan2(ai.y - player.y, ai.x - player.x);
      player.faceAngle = angle;
      if ((keys.KeyK || keys.KeyL || keys.Space) && !player.punching) {
        player.startPunch(angle);
        if (checkHit(player, ai, (ps, as) => ctx.onScore(ps, as))) {
          hitEffects.push(new HitEffect((player.x + ai.x) / 2, (player.y + ai.y) / 2));
          if (player.score >= MAX_SCORE) endGame();
        }
      }
    }

    // ── AI input modes ─────────────────────────────────────────────────────

    // Random policy (easy)
    function aiInputRandom(dt) {
      if (ai.knockedDown) { ai.vx = ai.vy = 0; return; }
      aiReactTimer -= dt;
      if (aiReactTimer <= 0) {
        aiReactTimer = 300 + Math.random() * 400;
        const moveIdx = Math.floor(Math.random() * 9);
        const [dx, dy] = MOVE_DELTA[moveIdx];
        ai.vx = dx * (SPEED * 0.8);
        ai.vy = dy * (SPEED * 0.8);
      }
      const angle = Math.atan2(player.y - ai.y, player.x - ai.x);
      ai.faceAngle = angle;
      if (Math.random() < 0.008 && !ai.punching) {
        ai.startPunch(angle);
        if (dist(ai, player) <= HIT_RANGE && checkHit(ai, player, (as, ps) => ctx.onScore(ps, as))) {
          hitEffects.push(new HitEffect((ai.x + player.x) / 2, (ai.y + player.y) / 2));
          if (ai.score >= MAX_SCORE) endGame();
        }
      }
    }

    // Rule-based AI (fallback when model fails to load)
    function aiInputRule(dt) {
      if (ai.knockedDown) { ai.vx = ai.vy = 0; return; }
      aiReactTimer -= dt;
      if (aiReactTimer <= 0) {
        aiReactTimer = ruleCfg.reactionMs;
        const mx = player.x + (Math.random() - 0.5) * RING.w * ruleCfg.mistakeRate;
        const my = player.y + (Math.random() - 0.5) * RING.h * ruleCfg.mistakeRate;
        aiTargetX = clamp(mx, RING.x + PLAYER_R, RING.x + RING.w - PLAYER_R);
        aiTargetY = clamp(my, RING.y + PLAYER_R, RING.y + RING.h - PLAYER_R);
      }
      const dx = aiTargetX - ai.x, dy = aiTargetY - ai.y;
      const d  = Math.hypot(dx, dy) || 1;
      if (d > HIT_RANGE * 0.7) { ai.vx = (dx / d) * ruleCfg.speed; ai.vy = (dy / d) * ruleCfg.speed; }
      else { ai.vx *= 0.85; ai.vy *= 0.85; }
      const angle = Math.atan2(player.y - ai.y, player.x - ai.x);
      ai.faceAngle = angle;
      if (Math.random() < ruleCfg.punchChance && !ai.punching && dist(ai, player) <= HIT_RANGE + 6) {
        ai.startPunch(angle);
        if (checkHit(ai, player, (as, ps) => ctx.onScore(ps, as))) {
          hitEffects.push(new HitEffect((ai.x + player.x) / 2, (ai.y + player.y) / 2));
          if (ai.score >= MAX_SCORE) endGame();
        }
      }
    }

    // PPO inference (medium / hell) — throttled to PPO_INTERVAL ms
    function aiInputPPO(dt) {
      if (ai.knockedDown) { ai.vx = ai.vy = 0; return; }

      ppoDt += dt;
      if (ppoDt >= PPO_INTERVAL) {
        ppoDt = 0;
        const obs    = buildObs(ai, player, timeLeft);
        const action = aiMode.agent.predict(obs);
        const decoded = decodeAction(action);
        ppoVx = decoded.vx;
        ppoVy = decoded.vy;
        if (decoded.punch) ppoPunchPending = true;
      }

      ai.vx = ppoVx; ai.vy = ppoVy;
      const angle = Math.atan2(player.y - ai.y, player.x - ai.x);
      ai.faceAngle = angle;

      if (ppoPunchPending && !ai.punching) {
        ppoPunchPending = false;
        ai.startPunch(angle);
        if (dist(ai, player) <= HIT_RANGE && checkHit(ai, player, (as, ps) => ctx.onScore(ps, as))) {
          hitEffects.push(new HitEffect((ai.x + player.x) / 2, (ai.y + player.y) / 2));
          if (ai.score >= MAX_SCORE) endGame();
        }
      }
    }

    function aiStep(dt) {
      if (aiMode.type === 'random') aiInputRandom(dt);
      else if (aiMode.type === 'ppo') aiInputPPO(dt);
      else aiInputRule(dt);
    }

    // ── Game over ──────────────────────────────────────────────────────────
    function endGame() {
      if (gameOver) return;
      gameOver = true;
      const won  = player.score > ai.score;
      const draw = player.score === ai.score;
      ctx.onEnd({
        playerScore: player.score,
        aiScore:     ai.score,
        message: draw ? "Draw! Evenly matched."
               : won  ? "You Win! 🥊 Great boxing!"
               :        "AI Wins! Better luck next time.",
      });
    }

    // ── Drawing ────────────────────────────────────────────────────────────
    function drawRing() {
      const bgGrad = c.createLinearGradient(0, 0, 0, H);
      bgGrad.addColorStop(0, "#1a0a00"); bgGrad.addColorStop(1, "#2d1200");
      c.fillStyle = bgGrad; c.fillRect(0, 0, W, H);
      const floorGrad = c.createLinearGradient(RING.x, RING.y, RING.x, RING.y + RING.h);
      floorGrad.addColorStop(0, "#c8102e"); floorGrad.addColorStop(0.5, "#a00c24"); floorGrad.addColorStop(1, "#c8102e");
      c.fillStyle = floorGrad; c.fillRect(RING.x, RING.y, RING.w, RING.h);
      c.strokeStyle = "rgba(255,255,255,0.85)"; c.lineWidth = 3;
      c.strokeRect(RING.x + 6, RING.y + 6, RING.w - 12, RING.h - 12);
      c.setLineDash([12, 8]); c.strokeStyle = "rgba(255,255,255,0.3)"; c.lineWidth = 2;
      c.beginPath(); c.moveTo(RING.x + RING.w / 2, RING.y + 8); c.lineTo(RING.x + RING.w / 2, RING.y + RING.h - 8); c.stroke();
      c.setLineDash([]);
      c.strokeStyle = "rgba(255,255,255,0.25)"; c.lineWidth = 2;
      c.beginPath(); c.arc(RING.x + RING.w / 2, RING.y + RING.h / 2, 36, 0, Math.PI * 2); c.stroke();
      [[RING.x,RING.y],[RING.x+RING.w,RING.y],[RING.x,RING.y+RING.h],[RING.x+RING.w,RING.y+RING.h]].forEach(([cx,cy])=>{
        c.fillStyle="#FFD700"; c.beginPath(); c.arc(cx,cy,7,0,Math.PI*2); c.fill();
      });
      ["#FFD700","rgba(255,255,255,0.9)","#FFD700"].forEach((col,i)=>{
        const o=[-12,0,12][i];
        c.strokeStyle=col; c.lineWidth=2.5;
        c.shadowBlur=i===1?6:0; c.shadowColor=col;
        c.strokeRect(RING.x-o*.5,RING.y-o*.5,RING.w+o,RING.h+o);
        c.shadowBlur=0;
      });
    }

    function drawHUD() {
      const hudY = 8;
      c.fillStyle = "rgba(0,0,0,0.5)"; c.beginPath(); c.roundRect(W/2-60,hudY,120,22,6); c.fill();
      const sec = Math.ceil(timeLeft/1000);
      const mm  = String(Math.floor(sec/60)).padStart(2,"0");
      const ss  = String(sec%60).padStart(2,"0");
      c.fillStyle = sec<=10 ? "#ff4444" : "#ffffff"; c.font="bold 14px 'Inter',sans-serif";
      c.textAlign="center"; c.textBaseline="middle";
      c.fillText(`${mm}:${ss}`, W/2, hudY+11);
      const barW=180,barH=22,barY=H-36;
      const pFill=(player.score/MAX_SCORE)*barW;
      c.fillStyle="rgba(0,0,0,0.5)"; c.beginPath(); c.roundRect(20,barY,barW,barH,6); c.fill();
      if(pFill>0){const pg=c.createLinearGradient(20,0,20+pFill,0);pg.addColorStop(0,"#3b82f6");pg.addColorStop(1,"#60a5fa");c.fillStyle=pg;c.beginPath();c.roundRect(20,barY,pFill,barH,6);c.fill();}
      c.fillStyle="#fff"; c.font="bold 12px 'Inter',sans-serif"; c.textAlign="left"; c.textBaseline="middle";
      c.fillText(`YOU  ${player.score}`,26,barY+11);
      const aFill=(ai.score/MAX_SCORE)*barW;
      c.fillStyle="rgba(0,0,0,0.5)"; c.beginPath(); c.roundRect(W-20-barW,barY,barW,barH,6); c.fill();
      if(aFill>0){const ag=c.createLinearGradient(W-20,0,W-20-aFill,0);ag.addColorStop(0,"#ef4444");ag.addColorStop(1,"#f87171");c.fillStyle=ag;c.beginPath();c.roundRect(W-20-aFill,barY,aFill,barH,6);c.fill();}
      c.fillStyle="#fff"; c.textAlign="right"; c.fillText(`AI  ${ai.score}`,W-26,barY+11);
      c.fillStyle="rgba(255,255,255,0.5)"; c.font="11px 'Inter',sans-serif"; c.textAlign="center";
      const modeLabel = aiMode.type === 'ppo' ? `PPO (${difficulty})` : difficulty.toUpperCase();
      c.fillText(modeLabel, W/2, H-27);
    }

    // ── Game loop ──────────────────────────────────────────────────────────
    function loop(ts) {
      if (!lastTime) lastTime = ts;
      const dt = Math.min(ts - lastTime, 60);
      lastTime = ts;
      if (!gameOver) { timeLeft -= dt; if (timeLeft <= 0) { timeLeft = 0; endGame(); } }
      if (!gameOver) { playerInput(); aiStep(dt); }
      player.update(dt); ai.update(dt);
      hitEffects = hitEffects.filter(e => !e.done());
      hitEffects.forEach(e => e.update(dt));
      c.clearRect(0, 0, W, H);
      drawRing();
      player.draw(c); ai.draw(c);
      hitEffects.forEach(e => e.draw(c));
      drawHUD();
      if (!gameOver || timeLeft > 0) rafId = requestAnimationFrame(loop);
    }

    rafId = requestAnimationFrame(loop);

    this.unmount = function () {
      cancelAnimationFrame(rafId);
      window.removeEventListener("keydown", e => onKey(e, true));
      window.removeEventListener("keyup",   e => onKey(e, false));
      container.innerHTML = "";
    };
  }

  // ── Agent cache ────────────────────────────────────────────────────────────
  window._boxingAgents = window._boxingAgents || {};

  async function loadBoxingAgent(key, url) {
    if (window._boxingAgents[key]) return window._boxingAgents[key];
    try {
      const agent = new window.PongRLAgent();   // generic; reused for boxing
      await agent.load(url);
      window._boxingAgents[key] = agent;
      return agent;
    } catch (e) {
      console.warn(`[Boxing] Could not load ${url}:`, e);
      return null;
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  let _engine = null;

  return {
    id: "boxing",
    title: "Boxing",

    mount(container, ctx) {
      const difficulty = ctx.difficulty || "medium";

      const startEngine = (aiMode) => {
        container.innerHTML = "";
        _engine = new BoxingEngine(container, ctx, difficulty, aiMode);
      };

      if (difficulty === "easy") {
        startEngine({ type: "random" });
        ctx.onScore(0, 0);
        return;
      }

      // Show loading state while fetching model
      container.innerHTML =
        '<div style="text-align:center;padding:60px;color:#aaa;font-size:14px">' +
        'Loading PPO model…</div>';

      const url = difficulty === "hell"
        ? "/models/boxing_best.json"
        : "/models/boxing_mid.json";

      loadBoxingAgent(difficulty, url).then((agent) => {
        if (agent) {
          startEngine({ type: "ppo", agent });
        } else {
          console.warn("[Boxing] Falling back to rule-based AI");
          startEngine({ type: "rule" });
        }
        ctx.onScore(0, 0);
      });
    },

    unmount() {
      _engine?.unmount();
      _engine = null;
    },
  };
})();
