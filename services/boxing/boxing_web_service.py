#!/usr/bin/env python3
"""
JSON-lines RPC service for PettingZoo boxing_v2 (Atari Boxing).
Upstream reference: https://github.com/bosakad/DQN-Multiagent-ATARI-Boxing
stdin: one JSON object per line
stdout: one JSON response per line
"""
from __future__ import annotations

import base64
import io
import json
import random
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np

# Atari action IDs (boxing_v2 minimal action space)
ACTION = {
    "NOOP": 0,
    "FIRE": 1,
    "UP": 2,
    "RIGHT": 3,
    "LEFT": 4,
    "DOWN": 5,
    "UPRIGHT": 6,
    "UPLEFT": 7,
    "DOWNRIGHT": 8,
    "DOWNLEFT": 9,
    "FIRE_UP": 10,
    "FIRE_RIGHT": 11,
    "FIRE_LEFT": 12,
    "FIRE_DOWN": 13,
    "FIRE_UPRIGHT": 14,
    "FIRE_UPLEFT": 15,
    "FIRE_DOWNRIGHT": 16,
    "FIRE_DOWNLEFT": 17,
}

DIFFICULTY = {
    "easy": {"mistake": 0.4, "aggressive": 0.15},
    "medium": {"mistake": 0.18, "aggressive": 0.35},
    "hell": {"mistake": 0.05, "aggressive": 0.6},
}

SESSIONS: Dict[str, "BoxingSession"] = {}


def reply(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def keys_to_action(keys: Dict[str, bool]) -> int:
    up = bool(keys.get("up"))
    down = bool(keys.get("down"))
    left = bool(keys.get("left"))
    right = bool(keys.get("right"))
    fire = bool(keys.get("fire"))

    if fire:
        if up and left:
            return ACTION["FIRE_UPLEFT"]
        if up and right:
            return ACTION["FIRE_UPRIGHT"]
        if down and left:
            return ACTION["FIRE_DOWNLEFT"]
        if down and right:
            return ACTION["FIRE_DOWNRIGHT"]
        if up:
            return ACTION["FIRE_UP"]
        if down:
            return ACTION["FIRE_DOWN"]
        if left:
            return ACTION["FIRE_LEFT"]
        if right:
            return ACTION["FIRE_RIGHT"]
        return ACTION["FIRE"]

    if up and left:
        return ACTION["UPLEFT"]
    if up and right:
        return ACTION["UPRIGHT"]
    if down and left:
        return ACTION["DOWNLEFT"]
    if down and right:
        return ACTION["DOWNRIGHT"]
    if up:
        return ACTION["UP"]
    if down:
        return ACTION["DOWN"]
    if left:
        return ACTION["LEFT"]
    if right:
        return ACTION["RIGHT"]
    return ACTION["NOOP"]


class BoxingSession:
    def __init__(self, session_id: str, difficulty: str = "medium"):
        from pettingzoo.atari import boxing_v2

        self.session_id = session_id
        self.difficulty = difficulty if difficulty in DIFFICULTY else "medium"
        self.env = boxing_v2.parallel_env(render_mode="rgb_array")
        self.player_id = "first_0"
        self.ai_id = "second_0"
        self.player_score = 0.0
        self.ai_score = 0.0
        self.done = False
        self.frame_count = 0
        self._reset()

    def _reset(self) -> None:
        self.env.reset()
        self.player_score = 0.0
        self.ai_score = 0.0
        self.done = False
        self.frame_count = 0

    def _encode_frame(self) -> Optional[str]:
        frame = self.env.render()
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        try:
            from PIL import Image
        except ImportError:
            # Fallback: raw PNG via pygame if available on env
            return None
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _ai_action(self) -> int:
        cfg = DIFFICULTY[self.difficulty]
        if random.random() < cfg["mistake"]:
            return self.env.action_space(self.ai_id).sample()

        # Aggressive corner-seeking + punch (similar to project random-policy tests)
        if random.random() < cfg["aggressive"]:
            return random.choice(
                [
                    ACTION["FIRE_UPLEFT"],
                    ACTION["FIRE_UPRIGHT"],
                    ACTION["FIRE_DOWNLEFT"],
                    ACTION["FIRE_DOWNRIGHT"],
                    ACTION["FIRE_UP"],
                    ACTION["FIRE_DOWN"],
                ]
            )

        return random.choice(
            [
                ACTION["UPLEFT"],
                ACTION["UPRIGHT"],
                ACTION["DOWNLEFT"],
                ACTION["DOWNRIGHT"],
                ACTION["LEFT"],
                ACTION["RIGHT"],
            ]
        )

    def step(self, keys: Dict[str, bool]) -> Dict[str, Any]:
        if self.done:
            return self.status(frame_b64=self._encode_frame())

        player_action = keys_to_action(keys)
        ai_action = self._ai_action()
        actions = {self.player_id: player_action, self.ai_id: ai_action}

        _, rewards, terminations, truncations, _ = self.env.step(actions)
        self.frame_count += 1

        self.player_score += float(rewards.get(self.player_id, 0))
        self.ai_score += float(rewards.get(self.ai_id, 0))

        if not self.env.agents:
            self.done = True
        if terminations.get(self.player_id) or terminations.get(self.ai_id):
            self.done = True
        if truncations.get(self.player_id) or truncations.get(self.ai_id):
            self.done = True
        # Atari Boxing typically ends at 100 points
        if self.player_score >= 100 or self.ai_score >= 100:
            self.done = True

        return self.status(frame_b64=self._encode_frame())

    def status(self, frame_b64: Optional[str] = None) -> Dict[str, Any]:
        return {
            "ok": True,
            "sessionId": self.session_id,
            "frame": frame_b64,
            "playerScore": int(round(self.player_score)),
            "aiScore": int(round(self.ai_score)),
            "done": self.done,
            "winner": (
                1
                if self.done and self.player_score > self.ai_score
                else 2
                if self.done and self.ai_score > self.player_score
                else 0
                if self.done
                else None
            ),
        }

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    rpc_id = req.get("_rpcId")
    cmd = req.get("cmd")
    session_id = req.get("sessionId")

    def wrap(result: Dict[str, Any]) -> Dict[str, Any]:
        if rpc_id is not None:
            result["_rpcId"] = rpc_id
        return result

    try:
        if cmd == "ping":
            return wrap({"ok": True, "pong": True})

        if cmd == "create":
            sid = session_id or f"bx-{random.randint(10000, 99999)}"
            if sid in SESSIONS:
                SESSIONS[sid].close()
            SESSIONS[sid] = BoxingSession(sid, req.get("difficulty", "medium"))
            out = SESSIONS[sid].status(frame_b64=SESSIONS[sid]._encode_frame())
            out["sessionId"] = sid
            return wrap(out)

        if cmd == "step":
            if not session_id or session_id not in SESSIONS:
                return wrap({"ok": False, "error": "Unknown session"})
            keys = req.get("keys") or {}
            return wrap(SESSIONS[session_id].step(keys))

        if cmd == "destroy":
            if session_id and session_id in SESSIONS:
                SESSIONS[session_id].close()
                del SESSIONS[session_id]
            return wrap({"ok": True})

        return wrap({"ok": False, "error": f"Unknown command: {cmd}"})
    except Exception as exc:
        return wrap({
            "ok": False,
            "error": str(exc),
            "trace": traceback.format_exc(),
        })


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            reply({"ok": False, "error": f"Invalid JSON: {exc}"})
            continue
        reply(handle(req))


if __name__ == "__main__":
    main()
