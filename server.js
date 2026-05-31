const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { URL } = require("url");

// Boxing은 순수 Canvas 게임으로 재구현 — 서버 API 불필요

const PORT = process.env.PORT || 3000;
const ROOT = path.join(__dirname, "public");
const DB_PATH = path.join(__dirname, "data", "db.json");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".wasm": "application/wasm",
};

function readDb() {
  return JSON.parse(fs.readFileSync(DB_PATH, "utf8"));
}

function writeDb(db) {
  fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2), "utf8");
}

function uid(prefix = "id") {
  return `${prefix}-${crypto.randomBytes(6).toString("hex")}`;
}

function send(res, status, data, type = "application/json") {
  const body = typeof data === "string" ? data : JSON.stringify(data);
  res.writeHead(status, {
    "Content-Type": type,
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  res.end(body);
}

function parseBody(req) {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
    });
    req.on("end", () => {
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        resolve({});
      }
    });
  });
}

function serveStatic(req, res) {
  let filePath = path.join(ROOT, req.url === "/" ? "index.html" : req.url);
  if (!filePath.startsWith(ROOT)) {
    send(res, 403, { error: "Forbidden" });
    return;
  }
  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, "index.html");
  }
  if (!fs.existsSync(filePath)) {
    filePath = path.join(ROOT, "index.html");
  }
  const ext = path.extname(filePath);
  const type = MIME[ext] || "application/octet-stream";
  res.writeHead(200, { "Content-Type": type });
  fs.createReadStream(filePath).pipe(res);
}

async function handleApi(req, res) {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const route = url.pathname;
  const method = req.method;

  if (method === "OPTIONS") {
    send(res, 204, "");
    return;
  }

  const body = method === "GET" || method === "DELETE" ? {} : await parseBody(req);

  if (route === "/api/auth/signup" && method === "POST") {
    const { userId, password } = body;
    if (!userId || !password) return send(res, 400, { error: "User ID and password are required." });
    const db = readDb();
    if (db.users.some((u) => u.userId === userId)) {
      return send(res, 409, { error: "User ID already exists." });
    }
    db.users.push({ userId, password, createdAt: new Date().toISOString() });
    writeDb(db);
    return send(res, 200, { userId, message: "Account created." });
  }

  if (route === "/api/auth/login" && method === "POST") {
    const { userId, password } = body;
    const db = readDb();
    const user = db.users.find((u) => u.userId === userId && u.password === password);
    if (!user) return send(res, 401, { error: "Invalid credentials." });
    return send(res, 200, { userId: user.userId, isGuest: false });
  }

  if (route === "/api/auth/guest" && method === "POST") {
    const guestNum = Math.floor(1000 + Math.random() * 9000);
    return send(res, 200, { userId: `Guest${guestNum}`, isGuest: true });
  }


  if (route === "/api/leaderboard" && method === "GET") {
    const game = url.searchParams.get("game");
    const difficulty = url.searchParams.get("difficulty");
    const db = readDb();
    let entries = [...db.leaderboard];
    if (game && game !== "all") entries = entries.filter((e) => e.game === game);
    if (difficulty && difficulty !== "all") {
      entries = entries.filter((e) => e.difficulty === difficulty);
    }
    entries.sort((a, b) => b.aiPercent - a.aiPercent || b.score - a.score);
    return send(res, 200, entries);
  }

  if (route === "/api/leaderboard" && method === "POST") {
    const { username, game, difficulty, score, aiPercent } = body;
    if (!username || !game || !difficulty || score == null || aiPercent == null) {
      return send(res, 400, { error: "Missing leaderboard fields." });
    }
    const db = readDb();
    const entry = {
      id: uid("lb"),
      username,
      game,
      difficulty,
      score: Number(score),
      aiPercent: Math.round(Number(aiPercent)),
      createdAt: new Date().toISOString(),
    };
    db.leaderboard.push(entry);
    writeDb(db);
    return send(res, 201, entry);
  }

  if (route === "/api/leaderboard" && method === "DELETE") {
    const db = readDb();
    db.leaderboard = [];
    writeDb(db);
    return send(res, 200, { message: "Leaderboard cleared." });
  }

  if (route === "/api/community/posts" && method === "GET") {
    const db = readDb();
    const posts = [...db.community].sort(
      (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
    );
    return send(res, 200, posts);
  }

  if (route === "/api/community/posts" && method === "POST") {
    const { username, content } = body;
    if (!username || !content?.trim()) {
      return send(res, 400, { error: "Username and content required." });
    }
    const db = readDb();
    const post = {
      id: uid("post"),
      username,
      avatar: username.charAt(0).toUpperCase(),
      content: content.trim(),
      likes: 0,
      likedBy: [],
      comments: [],
      createdAt: new Date().toISOString(),
    };
    db.community.unshift(post);
    writeDb(db);
    return send(res, 201, post);
  }

  const likeMatch = route.match(/^\/api\/community\/posts\/([^/]+)\/like$/);
  if (likeMatch && method === "POST") {
    const db = readDb();
    const post = db.community.find((p) => p.id === likeMatch[1]);
    if (!post) return send(res, 404, { error: "Post not found." });
    const key = body.username || "anonymous";
    if (post.likedBy.includes(key)) {
      post.likedBy = post.likedBy.filter((u) => u !== key);
      post.likes = Math.max(0, post.likes - 1);
    } else {
      post.likedBy.push(key);
      post.likes += 1;
    }
    writeDb(db);
    return send(res, 200, post);
  }

  const commentMatch = route.match(/^\/api\/community\/posts\/([^/]+)\/comments$/);
  if (commentMatch && method === "POST") {
    if (!body.content?.trim()) {
      return send(res, 400, { error: "Comment content required." });
    }
    const db = readDb();
    const post = db.community.find((p) => p.id === commentMatch[1]);
    if (!post) return send(res, 404, { error: "Post not found." });
    post.comments.push({
      id: uid("c"),
      username: body.username || "Guest",
      content: body.content.trim(),
      createdAt: new Date().toISOString(),
    });
    writeDb(db);
    return send(res, 200, post);
  }

  send(res, 404, { error: "Not found" });
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.url.startsWith("/api/")) {
      await handleApi(req, res);
    } else {
      serveStatic(req, res);
    }
  } catch (err) {
    console.error(err);
    send(res, 500, { error: "Server error" });
  }
});

server.listen(PORT, () => {
  console.log(`AI Arena running at http://localhost:${PORT}`);
});
