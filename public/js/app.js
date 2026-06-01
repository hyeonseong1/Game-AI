const API = "";

const GAMES = [
  {
    id: "tetris",
    title: "Tetris",
    desc: "Guideline Tetris (JS-Tetris) — SRS rotation, hold, T-spin scoring.",
    color: "btn-blue",
    engine: TetrisGame,
  },
  {
    id: "pong",
    title: "Pong",
    desc: "Atari Pong (1972) — authentic browser-games canvas recreation.",
    color: "btn-green",
    engine: PongGame,
  },
  {
    id: "boxing",
    title: "Boxing",
    desc: "Top-view canvas boxing. Dodge, jab and KO the AI opponent!",
    color: "btn-red",
    engine: BoxingGame,
  },
  {
    id: "space-invaders",
    title: "Space Invaders",
    desc: "Space Invaders (1978) — Taito-style canvas shooter from browser-games.",
    color: "btn-purple",
    engine: SpaceInvadersGame,
  },
  {
    id: "chess",
    title: "Chess",
    desc: "Play vs AI using chess.js and chessboard.js.",
    color: "btn-gray2",
    engine: ChessGame,
  },
  {
    id: "gomoku",
    title: "Gomoku",
    desc: "Five in a row on a 15x15 board (p5.js canvas).",
    color: "btn-orange",
    engine: GomokuGame,
  },
];

const state = {
  user: null,
  currentGame: null,
  difficulty: "medium",
  activeEngine: null,
};

const GAME_CONTROLS = {
  tetris: {
    title: "Tetris — Controls",
    items: [
      "Arrow Left / Right: move piece sideways",
      "Arrow Down: soft drop",
      "Arrow Up: rotate clockwise",
      "Space: hard drop",
      "X or Ctrl: rotate clockwise; Z: rotate counter-clockwise",
      "Shift or C: hold piece (one swap per lock)",
      "Beat the difficulty score target shown as the AI benchmark",
    ],
  },
  pong: {
    title: "Pong — Controls",
    items: [
      "W / Up or S / Down: move your paddle",
      "The right paddle is controlled by AI",
      "First player to 11 points wins",
    ],
  },
  "space-invaders": {
    title: "Space Invaders — Controls",
    items: [
      "Arrow Left / Right or A / D: move your ship",
      "Space: fire (only one shot on screen at a time)",
      "Difficulty sets the score benchmark to beat",
      "Game over when lives reach 0",
    ],
  },
  boxing: {
    title: "Boxing — Controls",
    items: [
      "WASD or Arrow keys: move your boxer",
      "K / L / Space: punch toward the opponent",
      "You are the white boxer (left side)",
      "Land punches to score points — first to 100 or highest score after 2 min wins",
      "Difficulty affects AI speed, reaction time and knockdown rate",
    ],
  },
};

function showGameControls(gameId) {
  const footer = document.getElementById("game-controls-footer");
  const info = GAME_CONTROLS[gameId];
  if (!info) {
    footer.classList.add("hidden");
    footer.innerHTML = "";
    return;
  }
  footer.classList.remove("hidden");
  footer.innerHTML =
    "<h3 class=\"game-controls-title\">" +
    info.title +
    "</h3><ul class=\"game-controls-list\">" +
    info.items.map((line) => "<li>" + line + "</li>").join("") +
    "</ul>";
}

function hideGameControls() {
  const footer = document.getElementById("game-controls-footer");
  footer.classList.add("hidden");
  footer.innerHTML = "";
}

const views = {
  login: document.getElementById("view-login"),
  signup: document.getElementById("view-signup"),
  games: document.getElementById("view-games"),
  difficulty: document.getElementById("view-difficulty"),
  gameplay: document.getElementById("view-gameplay"),
  leaderboard: document.getElementById("view-leaderboard"),
  community: document.getElementById("view-community"),
};

function showView(name) {
  Object.entries(views).forEach(([key, el]) => {
    el.classList.toggle("hidden", key !== name);
  });
}

function setUser(userId, isGuest = false) {
  state.user = { userId, isGuest };
  document.getElementById("welcome-name").textContent = userId;
  localStorage.setItem("aiArenaUser", JSON.stringify(state.user));
}

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function renderGameGrid() {
  const grid = document.getElementById("game-grid");
  grid.innerHTML = GAMES.map(
    (g) => `
    <article class="game-card">
      <div class="game-icon game-icon--${g.id}">${getGameIcon(g.id)}</div>
      <h3>${g.title}</h3>
      <p>${g.desc}</p>
      <button type="button" class="btn ${g.color} btn-block" data-game="${g.id}">Play Now</button>
    </article>
  `
  ).join("");

  grid.querySelectorAll("[data-game]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.currentGame = GAMES.find((g) => g.id === btn.dataset.game);
      document.getElementById("diff-game-title").textContent = state.currentGame.title;
      showView("difficulty");
    });
  });
}

function closeResultModal() {
  document.getElementById("result-modal-overlay").classList.add("hidden");
}

function showResultModal({ playerScore, aiScore, message, game, difficulty, isNewRecord, prevBest }) {
  const aiPercent = calcAiPercent(playerScore, aiScore);

  const isWin = /win|you win|player win/i.test(message);
  const isLose = /lose|lost|ai win|game over/i.test(message);
  const icon = isWin ? "🏆" : isLose ? "💀" : "🤝";
  const titleClass = isWin ? "win" : isLose ? "lose" : "draw";

  document.getElementById("result-modal-icon").textContent = icon;
  const title = document.getElementById("result-modal-title");
  title.textContent = isWin ? "You Win!" : isLose ? "You Lose!" : message;
  title.className = "result-modal-title " + titleClass;
  document.getElementById("result-player-score").textContent = playerScore;
  document.getElementById("result-ai-score").textContent = aiScore;
  document.getElementById("result-ai-percent").textContent = aiPercent + "%";

  const recordEl = document.getElementById("result-new-record");
  const prevBestEl = document.getElementById("result-prev-best");
  if (isNewRecord) {
    recordEl.classList.remove("hidden");
    prevBestEl.textContent = prevBest !== null
      ? `Previous best: ${prevBest} → ${playerScore}`
      : "First record for this game & difficulty!";
  } else {
    recordEl.classList.add("hidden");
  }

  const saveBtn = document.getElementById("btn-result-save");
  saveBtn.textContent = "Save to Leaderboard";
  saveBtn.disabled = false;
  saveBtn.style.display = state.user?.isGuest ? "none" : "";
  saveBtn.onclick = async () => {
    try {
      await api("/api/leaderboard", {
        method: "POST",
        body: JSON.stringify({
          username: state.user.userId,
          game: game.id,
          difficulty,
          score: playerScore,
          aiPercent,
        }),
      });
      saveBtn.textContent = "Saved!";
      saveBtn.disabled = true;
    } catch (e) {
      alert(e.message);
    }
  };

  document.getElementById("btn-result-restart").onclick = () => {
    closeResultModal();
    startGame(difficulty);
  };

  document.getElementById("btn-result-leaderboard").onclick = () => {
    closeResultModal();
    stopActiveGame();
    loadLeaderboard();
    showView("leaderboard");
  };

  document.getElementById("btn-result-games").onclick = () => {
    closeResultModal();
    stopActiveGame();
    showView("games");
  };

  document.getElementById("result-modal-overlay").classList.remove("hidden");
}

function stopActiveGame() {
  if (state.activeEngine?.unmount) state.activeEngine.unmount();
  state.activeEngine = null;
  hideGameControls();
}

function startGame(difficulty) {
  stopActiveGame();
  state.difficulty = difficulty;
  const game = state.currentGame;
  document.getElementById("play-game-title").textContent = game.title;
  document.getElementById("player-score").textContent = "0";
  document.getElementById("ai-score").textContent = "0";
  const container = document.getElementById("game-container");
  container.innerHTML = "";

  const ctx = {
    difficulty,
    onScore(player, ai) {
      document.getElementById("player-score").textContent = player;
      document.getElementById("ai-score").textContent = ai;
    },
    async onEnd({ playerScore, aiScore, message }) {
      let prevBest = null;
      let isNewRecord = false;

      if (!state.user?.isGuest) {
        try {
          const q = new URLSearchParams({ username: state.user.userId, game: game.id, difficulty });
          const result = await api(`/api/leaderboard/personal-best?${q}`);
          prevBest = result.best?.score ?? null;
          isNewRecord = prevBest === null || playerScore > prevBest;
        } catch {}
      } else {
        const key = `pb_${state.user.userId}_${game.id}_${difficulty}`;
        const stored = localStorage.getItem(key);
        prevBest = stored !== null ? Number(stored) : null;
        isNewRecord = prevBest === null || playerScore > prevBest;
        if (isNewRecord) localStorage.setItem(key, playerScore);
      }

      showResultModal({ playerScore, aiScore, message, game, difficulty, isNewRecord, prevBest });
    },
  };

  if (game.id === "pong" && typeof window.PongBrowserGame === "undefined") {
    alert("Pong is still loading. Please wait and try again.");
    return;
  }
  if (game.id === "space-invaders" && typeof window.SpaceInvadersBrowserGame === "undefined") {
    alert("Space Invaders is still loading. Please wait and try again.");
    return;
  }
  if (game.id === "chess" && (typeof Chess === "undefined" || typeof Chessboard === "undefined")) {
    alert("chess.js / chessboard.js is still loading. Please wait and try again.");
    return;
  }
  if (game.id === "gomoku" && typeof p5 === "undefined") {
    alert("p5.js is still loading. Please wait and try again.");
    return;
  }

  showView("gameplay");
  showGameControls(game.id);
  game.engine.mount(container, ctx);
  state.activeEngine = game.engine;
}

async function loadLeaderboard() {
  const game = document.getElementById("filter-game").value;
  const difficulty = document.getElementById("filter-difficulty").value;
  const q = new URLSearchParams({ game, difficulty });
  const entries = await api(`/api/leaderboard?${q}`);
  const list = document.getElementById("leaderboard-list");
  const medals = MEDAL_ICONS;
  const gameNames = Object.fromEntries(GAMES.map((g) => [g.id, g.title]));

  list.innerHTML =
    entries.length === 0
      ? "<p style='text-align:center;color:var(--muted)'>No entries yet. Play a game!</p>"
      : entries
        .map((e, i) => {
          let pctClass = "practice";
          if (e.aiPercent >= 100) pctClass = "outstanding";
          else if (e.aiPercent >= 70) pctClass = "good";
          const date = new Date(e.createdAt).toLocaleDateString();
          return `
            <div class="lb-entry">
              <span class="lb-rank">${medals[i] || i + 1}</span>
              <div class="lb-info">
                <strong>${e.username}</strong>
                <span>${gameNames[e.game] || e.game} — ${e.difficulty}</span>
              </div>
              <span class="lb-score">${e.score}</span>
              <span class="lb-percent ${pctClass}">${e.aiPercent}%</span>
              <span style="font-size:0.8rem;color:var(--muted)">${date}</span>
            </div>`;
        })
        .join("");
}

async function loadCommunity() {
  const posts = await api("/api/community/posts");
  const feed = document.getElementById("posts-feed");
  feed.innerHTML = posts
    .map(
      (p) => `
    <article class="post-card" data-id="${p.id}">
      <div class="post-header">
        <div class="avatar">${p.avatar}</div>
        <div>
          <strong>${p.username}</strong>
          <div class="post-meta">${new Date(p.createdAt).toLocaleString()}</div>
        </div>
      </div>
      <p>${escapeHtml(p.content)}</p>
      <div class="post-actions">
        <button type="button" class="btn-like">Like (${p.likes})</button>
        <button type="button" class="btn-comment">Comment</button>
      </div>
      <div class="comments-section">
        ${(p.comments || [])
          .map(
            (c) =>
              `<div class="comment"><strong>${c.username}</strong>: ${escapeHtml(c.content)}</div>`
          )
          .join("")}
        <div class="comment-form">
          <input type="text" placeholder="Write a comment..." class="comment-input" />
          <button type="button" class="btn btn-sm btn-purple btn-send-comment">Send</button>
        </div>
      </div>
    </article>
  `
    )
    .join("");

  feed.querySelectorAll(".post-card").forEach((card) => {
    const id = card.dataset.id;
    card.querySelector(".btn-like").addEventListener("click", async () => {
      const post = await api(`/api/community/posts/${id}/like`, {
        method: "POST",
        body: JSON.stringify({ username: state.user?.userId }),
      });
      card.querySelector(".btn-like").textContent = `Like (${post.likes})`;
    });
    card.querySelector(".btn-send-comment").addEventListener("click", async () => {
      const input = card.querySelector(".comment-input");
      const content = input.value.trim();
      if (!content) return;
      await api(`/api/community/posts/${id}/comments`, {
        method: "POST",
        body: JSON.stringify({ username: state.user?.userId, content }),
      });
      input.value = "";
      loadCommunity();
    });
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

document.getElementById("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = document.getElementById("login-error");
  err.classList.add("hidden");
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({
        userId: document.getElementById("login-user").value.trim(),
        password: document.getElementById("login-pass").value,
      }),
    });
    setUser(data.userId, false);
    showView("games");
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove("hidden");
  }
});

document.getElementById("signup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = document.getElementById("signup-error");
  err.classList.add("hidden");
  try {
    await api("/api/auth/signup", {
      method: "POST",
      body: JSON.stringify({
        userId: document.getElementById("signup-user").value.trim(),
        password: document.getElementById("signup-pass").value,
      }),
    });
    setUser(document.getElementById("signup-user").value.trim(), false);
    showView("games");
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove("hidden");
  }
});

document.getElementById("btn-guest").addEventListener("click", async () => {
  try {
    const data = await api("/api/auth/guest", { method: "POST" });
    setUser(data.userId, true);
  } catch {
    const guestNum = Math.floor(1000 + Math.random() * 9000);
    setUser(`Guest${guestNum}`, true);
  }
  showView("games");
});

document.getElementById("go-signup").addEventListener("click", (e) => {
  e.preventDefault();
  showView("signup");
});
document.getElementById("go-login").addEventListener("click", (e) => {
  e.preventDefault();
  showView("login");
});

document.querySelectorAll("[data-nav]").forEach((el) => {
  el.addEventListener("click", () => {
    const target = el.dataset.nav;
    if (target === "games") stopActiveGame();
    if (target === "leaderboard") loadLeaderboard();
    if (target === "community") loadCommunity();
    showView(target);
  });
});

document.getElementById("btn-exit-game").addEventListener("click", () => {
  stopActiveGame();
  showView("games");
});

document.querySelectorAll("[data-difficulty]").forEach((btn) => {
  btn.addEventListener("click", () => startGame(btn.dataset.difficulty));
});

document.getElementById("filter-game").addEventListener("change", loadLeaderboard);
document.getElementById("filter-difficulty").addEventListener("change", loadLeaderboard);

document.getElementById("btn-clear-lb").addEventListener("click", async () => {
  if (!confirm("Clear all leaderboard entries?")) return;
  await api("/api/leaderboard", { method: "DELETE" });
  loadLeaderboard();
});

document.getElementById("btn-post").addEventListener("click", async () => {
  const content = document.getElementById("new-post-content").value.trim();
  if (!content) return;
  await api("/api/community/posts", {
    method: "POST",
    body: JSON.stringify({ username: state.user?.userId || "Guest", content }),
  });
  document.getElementById("new-post-content").value = "";
  loadCommunity();
});

document.getElementById("result-modal-overlay").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) closeResultModal();
});

renderGameGrid();
showView("login");
