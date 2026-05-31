/**
 * Canvas game mount helper for browser-games ports
 */
const CanvasArena = {
  mountSingle(container, options = {}) {
    const { title = "Game", hint = "", aspectRatio = "16 / 10", createGame } = options;
    const uid = "cv-" + Date.now();

    container.innerHTML =
      '<div class="canvas-game-wrap">' +
      '<div class="game-panel canvas-panel">' +
      "<h4>" +
      title +
      "</h4>" +
      '<div id="' +
      uid +
      '-mount" class="canvas-mount"></div>' +
      (hint ? '<p class="controls-hint">' + hint + "</p>" : "") +
      "</div>" +
      "</div>";

    const mountEl = document.getElementById(uid + "-mount");
    const canvas = document.createElement("canvas");
    canvas.id = uid + "-canvas";
    canvas.className = "browser-game-canvas";
    canvas.style.aspectRatio = aspectRatio;
    mountEl.appendChild(canvas);

    const runtime = createGame(canvas);
    return {
      unmount() {
        runtime.destroy?.();
        if (runtime.rafId) cancelAnimationFrame(runtime.rafId);
        container.innerHTML = "";
      },
    };
  },

  runLoop(game, renderFn, timing) {
    const frameTime = timing.FRAME_TIME || 1000 / 60;
    const maxDelta = timing.MAX_DELTA || 200;
    let lastTime = performance.now();
    let accumulator = 0;
    let rafId = 0;
    let stopped = false;

    function loop(timestamp) {
      if (stopped) return;
      let delta = timestamp - lastTime;
      lastTime = timestamp;
      if (delta > maxDelta) delta = maxDelta;
      accumulator += delta;
      while (accumulator >= frameTime) {
        game.update();
        accumulator -= frameTime;
      }
      renderFn();
      rafId = requestAnimationFrame(loop);
    }

    rafId = requestAnimationFrame(loop);

    return {
      rafId,
      stop() {
        stopped = true;
        cancelAnimationFrame(rafId);
      },
    };
  },
};
