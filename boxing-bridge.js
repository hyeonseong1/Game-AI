/**
 * Persistent Python bridge for PettingZoo boxing_v2 (Atari Boxing).
 */
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const readline = require("readline");

const VENV_PYTHON = path.join(__dirname, "services", "boxing", ".venv", "bin", "python");
const PYTHON =
  process.env.BOXING_PYTHON ||
  (fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3");
const SCRIPT = path.join(__dirname, "services", "boxing", "boxing_web_service.py");

let proc = null;
let rl = null;
let nextId = 0;
const pending = new Map();
let bootError = null;

function startProcess() {
  if (proc) return;
  proc = spawn(PYTHON, [SCRIPT], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  rl = readline.createInterface({ input: proc.stdout });

  rl.on("line", (line) => {
    let data;
    try {
      data = JSON.parse(line);
    } catch {
      return;
    }
    const id = data._rpcId;
    if (id == null || !pending.has(id)) return;
    const { resolve, reject, timer } = pending.get(id);
    clearTimeout(timer);
    pending.delete(id);
    delete data._rpcId;
    if (data.ok === false) reject(new Error(data.error || "Boxing RPC failed"));
    else resolve(data);
  });

  proc.stderr.on("data", (chunk) => {
    const msg = chunk.toString();
    if (msg.trim()) console.error("[boxing]", msg.trim());
  });

  proc.on("error", (err) => {
    bootError = err;
    proc = null;
  });

  proc.on("exit", (code) => {
    bootError = new Error(`Boxing service exited (${code})`);
    proc = null;
    rl = null;
    for (const [, { reject, timer }] of pending) {
      clearTimeout(timer);
      reject(bootError);
    }
    pending.clear();
  });
}

function rpc(payload, timeoutMs = 12000) {
  startProcess();
  if (bootError) return Promise.reject(bootError);
  if (!proc) return Promise.reject(new Error("Boxing service unavailable"));

  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error("Boxing request timed out"));
    }, timeoutMs);

    pending.set(id, { resolve, reject, timer });
    proc.stdin.write(JSON.stringify({ ...payload, _rpcId: id }) + "\n");
  });
}

function ping() {
  return rpc({ cmd: "ping" }, 15000);
}

function createSession(difficulty) {
  const sessionId = `bx-${Date.now().toString(36)}`;
  return rpc({ cmd: "create", sessionId, difficulty });
}

function stepSession(sessionId, keys) {
  return rpc({ cmd: "step", sessionId, keys });
}

function destroySession(sessionId) {
  return rpc({ cmd: "destroy", sessionId }).catch(() => {});
}

module.exports = {
  ping,
  createSession,
  stepSession,
  destroySession,
};
