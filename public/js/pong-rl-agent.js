/**
 * PPO ActorCritic inference — runs entirely in the browser.
 *
 * Supports the legacy 2-layer MLP JSON and the SimbaV2 discrete PPO JSON.
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
    const arch = w.architecture || w.meta?.architecture || "legacy_mlp";
    if (arch === "rainbow_feature_c51") return this._predictRainbowFeatureC51(obs, w);
    if (arch === "simba_v2_discrete") return this._predictSimbaV2(obs, w);

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

  _predictRainbowFeatureC51(obs, w) {
    const layers = w.layers || {};
    let h = this._linear(obs, layers.fc.weight, layers.fc.bias).map(v => Math.max(0, v));

    let value = this._linear(h, layers.value_hidden.weight, layers.value_hidden.bias)
      .map(v => Math.max(0, v));
    value = this._linear(value, layers.value_out.weight, layers.value_out.bias);

    let adv = this._linear(h, layers.adv_hidden.weight, layers.adv_hidden.bias)
      .map(v => Math.max(0, v));
    adv = this._linear(adv, layers.adv_out.weight, layers.adv_out.bias);

    const actDim = w.act_dim || w.meta?.act_dim || 6;
    const atoms = w.atoms || 51;
    const support = w.support || this._linspace(w.v_min ?? -10, w.v_max ?? 10, atoms);
    const qValues = new Array(actDim).fill(0);

    for (let a = 0; a < actDim; a++) {
      const logits = new Array(atoms);
      for (let z = 0; z < atoms; z++) {
        let advMean = 0;
        for (let aa = 0; aa < actDim; aa++) advMean += adv[aa * atoms + z];
        advMean /= actDim;
        logits[z] = value[z] + adv[a * atoms + z] - advMean;
      }
      const probs = this._softmax(logits);
      let q = 0;
      for (let z = 0; z < atoms; z++) q += probs[z] * support[z];
      qValues[a] = q;
    }
    return this._argmax(qValues);
  }

  _predictSimbaV2(obs, w) {
    const mean = w.obs_mean || [];
    const variance = w.obs_var || [];
    const clip = w.obs_clip ?? 5;
    let h = obs.map((v, i) => {
      const n = (v - (mean[i] ?? 0)) / Math.sqrt((variance[i] ?? 1) + 1e-6);
      return Math.max(-clip, Math.min(clip, n));
    });

    h.push(w.c_shift ?? 3);
    h = this._l2(h);
    h = this._linearNoBias(h, w.embed.w);
    h = this._scale(h, w.embed.scale);
    h = this._l2(h);

    for (const block of w.blocks || []) {
      const y = this._hyperMlp(h, block.mlp);
      h = this._l2(h.map((v, i) => v + (block.alpha[i] ?? 0) * (y[i] - v)));
    }

    const logits = this._hyperMlp(h, w.actor).map((v, i) => v + (w.actor.bias[i] ?? 0));
    return this._argmax(logits);
  }

  _hyperMlp(x, spec) {
    let h = this._linearNoBias(x, spec.w1);
    h = this._scale(h, spec.scale);
    h = h.map(v => Math.max(0, v) + 1e-8);
    h = this._linearNoBias(h, spec.w2);
    return spec.normalize_out ? this._l2(h) : h;
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

  _linearNoBias(x, W) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = 0;
      const row = W[i];
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  _scale(x, scale) {
    return x.map((v, i) => v * (scale[i] ?? 1));
  }

  _l2(x) {
    let norm = 0;
    for (const v of x) norm += v * v;
    norm = Math.sqrt(Math.max(norm, 1e-8));
    return x.map(v => v / norm);
  }

  _argmax(values) {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }

  _softmax(values) {
    let maxValue = -Infinity;
    for (const v of values) if (v > maxValue) maxValue = v;
    const out = new Array(values.length);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      const e = Math.exp(values[i] - maxValue);
      out[i] = e;
      sum += e;
    }
    const denom = Math.max(sum, 1e-12);
    return out.map(v => v / denom);
  }

  _linspace(min, max, count) {
    if (count <= 1) return [min];
    const out = new Array(count);
    const step = (max - min) / (count - 1);
    for (let i = 0; i < count; i++) out[i] = min + step * i;
    return out;
  }
}

window.PongRLAgent = PongRLAgent;
