/**
 * Pong PPO ActorCritic inference — runs entirely in the browser.
 *
 * Network: Linear(6,256,Tanh) → Linear(256,256,Tanh) → actor Linear(256,3)
 * Matches ppo_agent.py ActorCritic architecture exactly.
 *
 * Usage:
 *   const agent = new PongRLAgent();
 *   await agent.load('/models/pong_best.json');
 *   const action = agent.predict(obs);  // 0=stay 1=up 2=down
 */
class PongRLAgent {
  constructor() {
    this._weights = null;
  }

  async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load model: ${url}`);
    this._weights = await res.json();
    return this;
  }

  predict(obs) {
    if (!this._weights) return 0;
    const w = this._weights;
    let h = this._linear(obs, w.shared_0_weight, w.shared_0_bias);
    h = h.map(Math.tanh);
    h = this._linear(h, w.shared_2_weight, w.shared_2_bias);
    h = h.map(Math.tanh);
    const logits = this._linear(h, w.actor_weight, w.actor_bias);
    // greedy: argmax
    let best = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    return best;
  }

  _linear(x, W, b) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = b[i];
      const row = W[i];
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }
}

window.PongRLAgent = PongRLAgent;
