/** ASCII-only source — icons via Unicode escapes (encoding-safe on Windows) */
const GAME_ICONS = Object.freeze({
  boxing: "\uD83E\uDD4A",
  "space-invaders": "\uD83D\uDC7E",
  chess: "\u265F\uFE0F",
  gomoku: "\u26AB",
});

/** Tetris-style block icon (T-piece + I-piece + S-piece), centered in 48×48 */
const TETRIS_ICON_SVG =
  '<svg class="game-icon-svg game-icon-svg--tetris" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" overflow="hidden" aria-hidden="true">' +
  /* T-piece (purple): top col1 / mid cols 0-2  — rows: y=10, y=20 */
  '<rect x="15" y="10" width="9" height="9" rx="1.5" fill="#a855f7" stroke="#e9d5ff" stroke-width="0.5"/>' +
  '<rect x="5"  y="20" width="9" height="9" rx="1.5" fill="#a855f7" stroke="#e9d5ff" stroke-width="0.5"/>' +
  '<rect x="15" y="20" width="9" height="9" rx="1.5" fill="#a855f7" stroke="#e9d5ff" stroke-width="0.5"/>' +
  '<rect x="25" y="20" width="9" height="9" rx="1.5" fill="#a855f7" stroke="#e9d5ff" stroke-width="0.5"/>' +
  /* S-piece (green): top-right area — rows: y=10 */
  '<rect x="25" y="10" width="9" height="9" rx="1.5" fill="#4ade80" stroke="#bbf7d0" stroke-width="0.5"/>' +
  '<rect x="35" y="10" width="9" height="9" rx="1.5" fill="#4ade80" stroke="#bbf7d0" stroke-width="0.5"/>' +
  /* I-piece (cyan): bottom row — y=30, cols 0-3 */
  '<rect x="5"  y="30" width="9" height="9" rx="1.5" fill="#22d3ee" stroke="#a5f3fc" stroke-width="0.5"/>' +
  '<rect x="15" y="30" width="9" height="9" rx="1.5" fill="#22d3ee" stroke="#a5f3fc" stroke-width="0.5"/>' +
  '<rect x="25" y="30" width="9" height="9" rx="1.5" fill="#22d3ee" stroke="#a5f3fc" stroke-width="0.5"/>' +
  '<rect x="35" y="30" width="9" height="9" rx="1.5" fill="#22d3ee" stroke="#a5f3fc" stroke-width="0.5"/>' +
  "</svg>";

/** Atari Pong-style court (was \uD83C\uDDF3 = regional "N", not a game icon) */
const PONG_ICON_SVG =
  '<svg class="game-icon-svg game-icon-svg--pong" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" aria-hidden="true">' +
  '<rect x="4" y="4" width="40" height="40" rx="5" fill="rgba(15,23,42,0.55)" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>' +
  '<line x1="24" y1="10" x2="24" y2="38" stroke="rgba(255,255,255,0.22)" stroke-width="1.5" stroke-dasharray="2 3" stroke-linecap="round"/>' +
  '<rect x="9" y="14" width="3.5" height="16" rx="1.25" fill="#fff"/>' +
  '<rect x="35.5" y="18" width="3.5" height="12" rx="1.25" fill="rgba(255,255,255,0.9)"/>' +
  '<rect x="21.5" y="21.5" width="5" height="5" rx="0.5" fill="#fff"/>' +
  "</svg>";

const MEDAL_ICONS = Object.freeze([
  "\uD83E\uDD47",
  "\uD83E\uDD48",
  "\uD83E\uDD49",
]);

function getGameIcon(gameId) {
  if (gameId === "pong") return PONG_ICON_SVG;
  if (gameId === "tetris") return TETRIS_ICON_SVG;
  return GAME_ICONS[gameId] || "\u25C6";
}
