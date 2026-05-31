/**
 * Gomoku (Five in a Row)
 *
 * AI modes by difficulty:
 *   easy   — in-thread greedy heuristic (fast, makes occasional mistakes)
 *   medium — Web Worker, 6-ply alpha-beta minimax with bitboards
 *   hell   — Web Worker, 8-ply alpha-beta minimax with bitboards
 */
const GomokuGame = {
  id: "gomoku",
  title: "Gomoku",

  mount(container, ctx) {
    const SIZE = 15;
    const difficulty = ctx.difficulty || "medium";

    const mountEl = document.createElement("div");
    mountEl.className = "gomoku-p5-mount";
    container.innerHTML = "";
    container.appendChild(mountEl);

    // ── Board state ──────────────────────────────────────────────────────
    // 0 = empty, 1 = player (black), 2 = AI (white)
    const board = Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
    let playerTurn = true;
    let over = false;
    let aiThinking = false;
    let playerScore = 0;
    let aiScore = 0;

    // ── Win checker ──────────────────────────────────────────────────────
    const countLine = (r, c, dr, dc, who) => {
      let n = 0, rr = r, cc = c;
      while (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr][cc] === who) {
        n++; rr += dr; cc += dc;
      }
      return n;
    };
    const checkWin = (r, c, who) =>
      [[0,1],[1,0],[1,1],[1,-1]].some(([dr, dc]) =>
        countLine(r, c, dr, dc, who) + countLine(r - dr, c - dc, -dr, -dc, who) - 1 >= 5
      );

    // ── Board → bitboards (for Worker) ───────────────────────────────────
    const board2Bitboards = () => {
      const bb = [0,0,0,0,0,0,0,0], wb = [0,0,0,0,0,0,0,0];
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          const pos = r * SIZE + c;
          const slot = pos >> 5, bit = pos & 31;
          if (board[r][c] === 1) bb[slot] |= 1 << bit;
          else if (board[r][c] === 2) wb[slot] |= 1 << bit;
        }
      }
      return { blackBitboard: bb, whiteBitboard: wb };
    };

    // ── Easy AI: in-thread greedy heuristic ──────────────────────────────
    const scoreCell = (r, c, who) => {
      if (board[r][c]) return -1;
      board[r][c] = who;
      let best = 0;
      [[0,1],[1,0],[1,1],[1,-1]].forEach(([dr, dc]) => {
        const n = countLine(r, c, dr, dc, who) + countLine(r-dr, c-dc, -dr, -dc, who) - 1;
        best = Math.max(best, n);
      });
      board[r][c] = 0;
      return best;
    };

    const easyAiMove = () => {
      let best = -1, picks = [];
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          if (board[r][c]) continue;
          // Win immediately
          if (scoreCell(r, c, 2) >= 4) { picks = [[r, c]]; best = 999; break; }
          // Block player win
          if (scoreCell(r, c, 1) >= 4) { picks = [[r, c]]; best = 998; break; }
          let s = scoreCell(r, c, 2) * 14 + scoreCell(r, c, 1) * 12;
          s *= (0.6 + Math.random() * 0.4); // easy: add noise
          if (s > best) { best = s; picks = [[r, c]]; }
          else if (s === best) picks.push([r, c]);
        }
        if (best >= 998) break;
      }
      if (!picks.length) return;
      const [r, c] = picks[Math.floor(Math.random() * picks.length)];
      placeAI(r, c);
    };

    // ── Place AI stone and check win ──────────────────────────────────────
    const placeAI = (r, c) => {
      board[r][c] = 2;
      aiThinking = false;
      if (checkWin(r, c, 2)) {
        over = true;
        aiScore = 1;
        ctx.onScore(playerScore, aiScore);
        ctx.onEnd({ playerScore, aiScore, message: "AI wins — five in a row!" });
        return;
      }
      playerTurn = true;
    };

    // ── Worker AI (medium / hell) ─────────────────────────────────────────
    let worker = null;
    if (difficulty !== "easy") {
      worker = new Worker("/js/gomoku-ai-worker.js");
      const workerDiff = difficulty === "hell" ? "hard" : "medium";

      worker.onmessage = (e) => {
        if (over) return;
        const { type, move } = e.data;
        if (type === "BEST_MOVE_FOUND" && move) {
          placeAI(move.row, move.col);
        }
      };

      worker.onerror = (err) => {
        console.error("[Gomoku AI]", err);
        aiThinking = false;
        playerTurn = true;
      };

      // Notify worker of new game
      worker.postMessage({ type: "NEW_GAME" });
    }

    const workerAiMove = () => {
      const workerDiff = difficulty === "hell" ? "hard" : "medium";
      const { blackBitboard, whiteBitboard } = board2Bitboards();
      worker.postMessage({
        type: "FIND_BEST_MOVE",
        data: { blackBitboard, whiteBitboard,
                computerPlayer: "white", humanPlayer: "black",
                difficulty: workerDiff },
      });
    };

    const triggerAI = () => {
      if (over) return;
      aiThinking = true;
      playerTurn = false;
      if (difficulty === "easy") {
        // small delay so the move feels natural
        setTimeout(easyAiMove, 160 + Math.random() * 100);
      } else {
        // Worker is async — placeAI called in worker.onmessage
        setTimeout(workerAiMove, 50);
      }
    };

    // ── p5.js rendering ───────────────────────────────────────────────────
    this._p5 = new p5((p) => {
      const pad = 28, cell = 26;
      const boardPx = pad * 2 + cell * (SIZE - 1);

      // dot positions (gomoku star points on 15×15)
      const STARS = [[3,3],[3,7],[3,11],[7,3],[7,7],[7,11],[11,3],[11,7],[11,11]];

      p.setup = () => {
        p.createCanvas(boardPx, boardPx + 24).parent(mountEl);
        p.textFont("Segoe UI, sans-serif");
      };

      p.draw = () => {
        // Board background
        p.background(222, 184, 135);

        // Grid lines
        p.stroke(101, 67, 33);
        p.strokeWeight(1);
        for (let i = 0; i < SIZE; i++) {
          p.line(pad, pad + i * cell, pad + (SIZE - 1) * cell, pad + i * cell);
          p.line(pad + i * cell, pad, pad + i * cell, pad + (SIZE - 1) * cell);
        }

        // Star points
        p.noStroke();
        p.fill(60, 30, 10);
        STARS.forEach(([r, c]) => p.circle(pad + c * cell, pad + r * cell, 5));

        // Stones
        for (let r = 0; r < SIZE; r++) {
          for (let c = 0; c < SIZE; c++) {
            if (!board[r][c]) continue;
            if (board[r][c] === 1) {
              // Player: black with radial gradient-like shadow
              p.fill(30, 30, 30);
              p.noStroke();
              p.circle(pad + c * cell, pad + r * cell, 21);
              p.fill(80, 80, 80);
              p.circle(pad + c * cell - 3, pad + r * cell - 3, 7);
            } else {
              // AI: white stone
              p.stroke(80);
              p.strokeWeight(1);
              p.fill(235, 235, 220);
              p.circle(pad + c * cell, pad + r * cell, 21);
              p.noStroke();
              p.fill(255);
              p.circle(pad + c * cell - 3, pad + r * cell - 3, 7);
            }
          }
        }

        // Status bar
        p.noStroke();
        p.fill(50);
        p.textSize(12);
        p.textAlign(p.CENTER);
        if (over) {
          p.fill(200, 50, 50);
          p.text(aiScore ? "AI wins!" : "You win!", boardPx / 2, boardPx + 16);
        } else if (aiThinking) {
          p.fill(40, 120, 200);
          // Animated dots
          const dots = ".".repeat(1 + Math.floor(p.frameCount / 15) % 3);
          p.text("AI thinking" + dots, boardPx / 2, boardPx + 16);
        } else {
          p.fill(60);
          p.text("Your turn — click to place stone (Black)", boardPx / 2, boardPx + 16);
        }
      };

      p.mousePressed = () => {
        if (!playerTurn || over || aiThinking) return;
        const c = Math.round((p.mouseX - pad) / cell);
        const r = Math.round((p.mouseY - pad) / cell);
        if (r < 0 || r >= SIZE || c < 0 || c >= SIZE || board[r][c]) return;

        board[r][c] = 1;
        if (checkWin(r, c, 1)) {
          over = true;
          playerScore = 1;
          ctx.onScore(playerScore, aiScore);
          ctx.onEnd({ playerScore, aiScore, message: "You win — five in a row!" });
          return;
        }
        triggerAI();
      };
    });

    ctx.onScore(0, 0);
    this._worker = worker;
  },

  unmount() {
    if (this._worker) {
      this._worker.terminate();
      this._worker = null;
    }
    if (this._p5) {
      this._p5.remove();
      this._p5 = null;
    }
  },
};
