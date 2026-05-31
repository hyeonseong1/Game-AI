/**
 * Chess — chess.js + chessboard.js + Stockfish WASM AI
 *
 * AI by difficulty:
 *   easy   — Stockfish Skill Level 3, depth 2   (weak, beginner-friendly)
 *   medium — Stockfish Skill Level 10, depth 8  (solid club-level play)
 *   hell   — Stockfish Skill Level 20, depth 15 (near-maximum strength)
 */

// Stockfish UCI skill-level presets per difficulty
const STOCKFISH_PRESET = {
  easy:   { skill: 3,  depth: 2  },
  medium: { skill: 10, depth: 8  },
  hell:   { skill: 20, depth: 15 },
};

const ChessGame = {
  id: "chess",
  title: "Chess",

  mount(container, ctx) {
    const difficulty = ctx.difficulty || "medium";
    const preset = STOCKFISH_PRESET[difficulty] || STOCKFISH_PRESET.medium;

    const uid = "chess-" + Date.now();
    container.innerHTML =
      '<div class="chess-wrap">' +
      `<p class="controls-hint">Drag pieces to move. AI powered by Stockfish 18 (${difficulty})</p>` +
      `<div id="${uid}" style="width:min(420px,100%);margin:0 auto"></div>` +
      `<p id="${uid}-status" style="text-align:center;margin-top:12px;font-size:14px"></p>` +
      '</div>';

    const game   = new Chess();
    const statusEl = document.getElementById(uid + "-status");
    let board  = null;
    let over   = false;
    let engineReady = false;
    let pendingMove = false;

    const setStatus = (msg) => { if (statusEl) statusEl.textContent = msg; };

    // ── Stockfish worker ──────────────────────────────────────────────────
    const engine = new Worker("/stockfish.js");

    const uci = (cmd) => engine.postMessage(cmd);

    const getMoves = () =>
      game.history({ verbose: true })
          .map(m => m.from + m.to + (m.promotion || ""))
          .join(" ");

    const requestEngineMove = () => {
      if (over || !engineReady || pendingMove) return;
      pendingMove = true;
      const moves = getMoves();
      uci("position startpos" + (moves ? " moves " + moves : ""));
      uci("go depth " + preset.depth);
    };

    engine.onmessage = (e) => {
      const line = typeof e === "string" ? e : e.data;

      if (line === "uciok") {
        uci("setoption name Skill Level value " + preset.skill);
        // Error probability tuning (matches enginegame.js pattern)
        const errProb = Math.round(preset.skill * 6.35 + 1);
        const maxErr  = Math.round(preset.skill * -0.5 + 10);
        uci("setoption name Skill Level Maximum Error value " + maxErr);
        uci("setoption name Skill Level Probability value "   + errProb);
        uci("isready");
        return;
      }

      if (line === "readyok") {
        engineReady = true;
        setStatus("Your turn (White)");
        return;
      }

      // bestmove e2e4 [ponder ...]
      const bm = line.match(/^bestmove ([a-h][1-8])([a-h][1-8])([qrbn])?/);
      if (bm) {
        pendingMove = false;
        if (over) return;
        const move = game.move({ from: bm[1], to: bm[2], promotion: bm[3] || "q" });
        if (move) {
          board.position(game.fen());
          afterMove();
        }
      }
    };

    engine.onerror = (err) => {
      console.error("[Stockfish]", err);
      setStatus("Engine error — reload to retry.");
    };

    // Start UCI handshake
    uci("uci");
    uci("ucinewgame");
    setStatus("Loading Stockfish engine…");

    // ── After every move: detect game-over states ────────────────────────
    const afterMove = () => {
      const wTurn = game.turn() === "w";
      ctx.onScore(
        game.in_checkmate() && wTurn ? 1 : 0,
        game.in_checkmate() && !wTurn ? 1 : 0
      );

      if (game.in_checkmate()) {
        over = true;
        const playerWon = wTurn; // it's now white's turn → black just mated white? No—black won
        // After checkmate, turn flips to the player who was mated
        // so if turn is 'b' after the move, WHITE checkmated BLACK → AI won
        const aiWon = !wTurn;
        ctx.onEnd({
          playerScore: aiWon ? 0 : 1,
          aiScore: aiWon ? 1 : 0,
          message: aiWon ? "Checkmate — AI wins." : "Checkmate! You win.",
        });
        setStatus(aiWon ? "AI wins!" : "You win!");
        return;
      }
      if (game.in_draw() || game.in_stalemate() || game.in_threefold_repetition()) {
        over = true;
        ctx.onEnd({ playerScore: 0, aiScore: 0, message: "Draw." });
        setStatus("Draw.");
        return;
      }

      if (game.turn() === "w") {
        setStatus("Your turn (White)");
      } else {
        setStatus("AI thinking…");
        window.setTimeout(requestEngineMove, 100);
      }
    };

    // ── chessboard.js UI ─────────────────────────────────────────────────
    board = Chessboard(uid, {
      position:  "start",
      draggable: true,
      pieceTheme: "/img/chesspieces/wikipedia/{piece}.png",

      onDragStart(source, piece) {
        if (over || game.game_over() || game.turn() !== "w") return false;
        if (/^b/.test(piece)) return false; // can't drag black pieces
      },

      onDrop(source, target) {
        if (!engineReady) return "snapback";
        const move = game.move({ from: source, to: target, promotion: "q" });
        if (!move) return "snapback";
        afterMove();
      },

      onSnapEnd() {
        board.position(game.fen());
      },
    });

    setStatus("Loading Stockfish engine…");
    ctx.onScore(0, 0);
    this._engine = engine;
    this._board  = board;
  },

  unmount() {
    if (this._engine) {
      try { this._engine.postMessage("quit"); } catch (_) {}
      this._engine.terminate();
      this._engine = null;
    }
    if (this._board) {
      this._board.destroy();
      this._board = null;
    }
  },
};
