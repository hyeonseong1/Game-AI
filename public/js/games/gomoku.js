/**
 * Gomoku (Five in a Row)
 *
 * - Stone colors are randomly assigned each game (Black always goes first)
 * - AI modes by difficulty:
 *     easy   — in-thread greedy heuristic
 *     medium — Web Worker, 6-ply alpha-beta minimax
 *     hell   — Web Worker, 8-ply alpha-beta minimax
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

    // ── Random color assignment ───────────────────────────────────────────
    // board values: 0=empty, 1=black, 2=white
    // Black always moves first; if player is white, AI (black) goes first
    const playerIsBlack  = Math.random() < 0.5;
    const playerStone    = playerIsBlack ? 1 : 2;
    const aiStone        = playerIsBlack ? 2 : 1;
    const playerColor    = playerIsBlack ? "Black" : "White";
    const aiColor        = playerIsBlack ? "white" : "black";   // for Worker
    const playerWorkerColor = playerIsBlack ? "black" : "white";

    // ── Board state ───────────────────────────────────────────────────────
    const board = Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
    let playerTurn = playerIsBlack; // black moves first
    let over       = false;
    let aiThinking = false;
    let playerScore = 0;
    let aiScore     = 0;

    // ── Win checker (fixed: removed erroneous -1) ─────────────────────────
    // countLine: count consecutive stones of 'who' starting at (r,c) in direction (dr,dc), including (r,c) itself
    // checkWin: sum of both directions = actual consecutive count → win if >= 5
    const countLine = (r, c, dr, dc, who) => {
      let n = 0, rr = r, cc = c;
      while (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr][cc] === who) {
        n++; rr += dr; cc += dc;
      }
      return n;
    };
    const checkWin = (r, c, who) =>
      [[0,1],[1,0],[1,1],[1,-1]].some(([dr, dc]) =>
        countLine(r, c, dr, dc, who) + countLine(r - dr, c - dc, -dr, -dc, who) >= 5
      );

    // ── Board → bitboards (for Worker) ───────────────────────────────────
    const board2Bitboards = () => {
      const bb = [0,0,0,0,0,0,0,0], wb = [0,0,0,0,0,0,0,0];
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          const pos = r * SIZE + c;
          const slot = pos >> 5, bit = pos & 31;
          if (board[r][c] === 1) bb[slot] |= 1 << bit;      // black
          else if (board[r][c] === 2) wb[slot] |= 1 << bit; // white
        }
      }
      return { blackBitboard: bb, whiteBitboard: wb };
    };

    // ── Easy AI: in-thread greedy heuristic ──────────────────────────────
    // scoreCell: returns the max consecutive length when 'who' is placed at (r,c)
    // (fixed: -1 removed, so 5-in-a-row correctly returns 5)
    const scoreCell = (r, c, who) => {
      if (board[r][c]) return -1;
      board[r][c] = who;
      let best = 0;
      [[0,1],[1,0],[1,1],[1,-1]].forEach(([dr, dc]) => {
        const n = countLine(r, c, dr, dc, who) + countLine(r-dr, c-dc, -dr, -dc, who);
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
          if (scoreCell(r, c, aiStone) >= 5)     { picks = [[r, c]]; best = 999; break; }
          // Block opponent win
          if (scoreCell(r, c, playerStone) >= 5) { picks = [[r, c]]; best = 998; break; }
          let s = scoreCell(r, c, aiStone) * 14 + scoreCell(r, c, playerStone) * 12;
          s *= (0.6 + Math.random() * 0.4);
          if (s > best) { best = s; picks = [[r, c]]; }
          else if (s === best) picks.push([r, c]);
        }
        if (best >= 998) break;
      }
      if (!picks.length) return;
      const [r, c] = picks[Math.floor(Math.random() * picks.length)];
      placeAI(r, c);
    };

    // ── Place AI stone ────────────────────────────────────────────────────
    const placeAI = (r, c) => {
      board[r][c] = aiStone;
      aiThinking = false;
      if (checkWin(r, c, aiStone)) {
        over = true;
        aiScore = 1;
        ctx.onScore(playerScore, aiScore);
        ctx.onEnd({ playerScore, aiScore, message: "AI wins — five in a row!" });
        return;
      }
      playerTurn = true;
    };

    // ── Worker AI (medium / hell) ──────────────────────────────────────────
    let worker = null;
    if (difficulty !== "easy") {
      worker = new Worker("/js/gomoku-ai-worker.js");

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

      worker.postMessage({ type: "NEW_GAME" });
    }

    const workerAiMove = () => {
      const workerDiff = difficulty === "hell" ? "hard" : "medium";
      const { blackBitboard, whiteBitboard } = board2Bitboards();
      worker.postMessage({
        type: "FIND_BEST_MOVE",
        data: {
          blackBitboard, whiteBitboard,
          computerPlayer: aiColor,
          humanPlayer:    playerWorkerColor,
          difficulty:     workerDiff,
        },
      });
    };

    const triggerAI = () => {
      if (over) return;
      aiThinking = true;
      playerTurn = false;
      if (difficulty === "easy") {
        setTimeout(easyAiMove, 160 + Math.random() * 100);
      } else {
        setTimeout(workerAiMove, 50);
      }
    };

    // If player is white, AI (black) moves first
    if (!playerIsBlack) {
      setTimeout(triggerAI, 400);
    }

    // ── p5.js rendering ───────────────────────────────────────────────────
    this._p5 = new p5((p) => {
      const pad = 28, cell = 26;
      const boardPx = pad * 2 + cell * (SIZE - 1);
      const STARS = [[3,3],[3,7],[3,11],[7,3],[7,7],[7,11],[11,3],[11,7],[11,11]];

      p.setup = () => {
        p.createCanvas(boardPx, boardPx + 28).parent(mountEl);
        p.textFont("Segoe UI, sans-serif");
      };

      p.draw = () => {
        p.background(222, 184, 135);

        // Grid lines
        p.stroke(101, 67, 33); p.strokeWeight(1);
        for (let i = 0; i < SIZE; i++) {
          p.line(pad, pad + i * cell, pad + (SIZE-1)*cell, pad + i*cell);
          p.line(pad + i*cell, pad, pad + i*cell, pad + (SIZE-1)*cell);
        }

        // Star points
        p.noStroke(); p.fill(60, 30, 10);
        STARS.forEach(([r, c]) => p.circle(pad + c*cell, pad + r*cell, 5));

        // Stones (board[r][c]=1 → black, =2 → white)
        for (let r = 0; r < SIZE; r++) {
          for (let c = 0; c < SIZE; c++) {
            if (!board[r][c]) continue;
            const x = pad + c * cell, y = pad + r * cell;
            if (board[r][c] === 1) {
              // Black stone
              p.noStroke();
              p.fill(25, 25, 25);
              p.circle(x, y, 22);
              p.fill(90, 90, 90);
              p.circle(x - 4, y - 4, 7);
            } else {
              // White stone
              p.stroke(100); p.strokeWeight(1);
              p.fill(238, 235, 215);
              p.circle(x, y, 22);
              p.noStroke();
              p.fill(255);
              p.circle(x - 4, y - 4, 7);
            }
          }
        }

        // Status bar
        p.noStroke(); p.textSize(12); p.textAlign(p.CENTER);
        if (over) {
          p.fill(200, 50, 50);
          p.text(aiScore ? "AI wins!" : "You win!", boardPx / 2, boardPx + 16);
        } else if (aiThinking) {
          p.fill(40, 120, 200);
          const dots = ".".repeat(1 + Math.floor(p.frameCount / 15) % 3);
          p.text("AI thinking" + dots, boardPx / 2, boardPx + 16);
        } else if (playerTurn) {
          p.fill(60);
          p.text(`Your turn — click to place stone (${playerColor})`, boardPx / 2, boardPx + 16);
        } else {
          p.fill(100);
          p.text("Waiting...", boardPx / 2, boardPx + 16);
        }
      };

      p.mousePressed = () => {
        if (!playerTurn || over || aiThinking) return;
        const c = Math.round((p.mouseX - pad) / cell);
        const r = Math.round((p.mouseY - pad) / cell);
        if (r < 0 || r >= SIZE || c < 0 || c >= SIZE || board[r][c]) return;

        board[r][c] = playerStone;
        if (checkWin(r, c, playerStone)) {
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
