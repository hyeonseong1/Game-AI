/**
 * Tetris — Dual-Screen PvAI
 *
 * Left screen  : Player (keyboard)
 * Right screen : AI (heuristic from tetrisai/js/ai.js)
 *
 * AI difficulty:
 *   easy   — random placement
 *   medium — 1-piece lookahead with tuned weights
 *   hell   — 2-piece lookahead with tuned weights + faster gravity
 *
 * Fully self-contained (no iframe, no external dependencies).
 * Ported from: github.com/LeeYiyuan/tetrisai (MIT)
 */
const TetrisGame = (function () {
'use strict';

// ═══════════════════════════════════════════════════════
// SECTION 1: GRID  (adapted from tetrisai/js/grid.js)
// ═══════════════════════════════════════════════════════
function Grid(rows, cols) {
    this.rows = rows; this.cols = cols;
    this.cells = Array.from({length: rows}, () => new Array(cols).fill(0));
}
Grid.prototype.clone = function () {
    const g = new Grid(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) g.cells[r] = [...this.cells[r]];
    return g;
};
Grid.prototype.valid = function (piece) {
    for (let r = 0; r < piece.cells.length; r++)
        for (let c = 0; c < piece.cells[r].length; c++)
            if (piece.cells[r][c] !== 0) {
                const gr = piece.row + r, gc = piece.col + c;
                if (gr < 0 || gr >= this.rows || gc < 0 || gc >= this.cols) return false;
                if (this.cells[gr][gc] !== 0) return false;
            }
    return true;
};
Grid.prototype.addPiece = function (piece) {
    for (let r = 0; r < piece.cells.length; r++)
        for (let c = 0; c < piece.cells[r].length; c++)
            if (piece.cells[r][c] !== 0) {
                const gr = piece.row + r, gc = piece.col + c;
                if (gr >= 0) this.cells[gr][gc] = piece.cells[r][c];
            }
};
Grid.prototype.clearLines = function () {
    let cleared = 0;
    for (let r = this.rows - 1; r >= 0; r--) {
        if (this.cells[r].every(v => v !== 0)) {
            this.cells.splice(r, 1);
            this.cells.unshift(new Array(this.cols).fill(0));
            cleared++;
            r++; // re-check same index
        }
    }
    return cleared;
};
Grid.prototype.exceeded = function () {
    return this.cells[0].some(v => v !== 0) || this.cells[1].some(v => v !== 0);
};
// Heuristic helpers
Grid.prototype.aggregateHeight = function () {
    let total = 0;
    for (let c = 0; c < this.cols; c++) total += this._colHeight(c);
    return total;
};
Grid.prototype._colHeight = function (col) {
    for (let r = 0; r < this.rows; r++) if (this.cells[r][col] !== 0) return this.rows - r;
    return 0;
};
Grid.prototype.lines = function () {
    return this.cells.filter(row => row.every(v => v !== 0)).length;
};
Grid.prototype.holes = function () {
    let count = 0;
    for (let c = 0; c < this.cols; c++) {
        let block = false;
        for (let r = 0; r < this.rows; r++) {
            if (this.cells[r][c] !== 0) block = true;
            else if (block) count++;
        }
    }
    return count;
};
Grid.prototype.bumpiness = function () {
    let total = 0;
    for (let c = 0; c < this.cols - 1; c++)
        total += Math.abs(this._colHeight(c) - this._colHeight(c + 1));
    return total;
};

// ═══════════════════════════════════════════════════════
// SECTION 2: PIECE  (adapted from tetrisai/js/piece.js)
// ═══════════════════════════════════════════════════════
// Colors: modern Tetris palette (one per piece type 0-6)
const PIECE_COLORS = [
    '#FFD700', // O  yellow
    '#3B82F6', // J  blue
    '#F97316', // L  orange
    '#EF4444', // Z  red
    '#22C55E', // S  green
    '#A855F7', // T  purple
    '#06B6D4', // I  cyan
];

function Piece(cells, typeId) {
    this.cells = cells.map(r => [...r]);
    this.typeId = typeId;
    this.dimension = cells.length;
    this.row = 0; this.col = 0;
}
Piece.fromIndex = function (idx) {
    const defs = [
        // 0: O
        [[1,1],[1,1]],
        // 1: J
        [[1,0,0],[1,1,1],[0,0,0]],
        // 2: L
        [[0,0,1],[1,1,1],[0,0,0]],
        // 3: Z
        [[1,1,0],[0,1,1],[0,0,0]],
        // 4: S
        [[0,1,1],[1,1,0],[0,0,0]],
        // 5: T
        [[0,1,0],[1,1,1],[0,0,0]],
        // 6: I
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    ];
    // Convert 1s to type color index (1-indexed so 0 = empty)
    const raw = defs[idx];
    const cells = raw.map(r => r.map(v => v ? idx + 1 : 0));
    const p = new Piece(cells, idx);
    p.col = Math.floor((10 - p.dimension) / 2);
    return p;
};
Piece.prototype.clone = function () {
    const p = new Piece(this.cells, this.typeId);
    p.row = this.row; p.col = this.col;
    return p;
};
Piece.prototype.rotateCW = function () {
    const n = this.dimension;
    const next = Array.from({length: n}, () => new Array(n).fill(0));
    for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++)
            next[c][n - 1 - r] = this.cells[r][c];
    this.cells = next;
};
Piece.prototype.rotate = function (grid) {
    const orig = this.cells.map(r => [...r]);
    const oc = this.col;
    this.rotateCW();
    if (grid.valid(this)) return;
    // wall kicks: try offsets
    for (const dc of [1, -1, 2, -2]) {
        this.col = oc + dc;
        if (grid.valid(this)) return;
    }
    this.col = oc;
    this.cells = orig; // revert
};
Piece.prototype.moveLeft  = function (grid) { this.col--; if (!grid.valid(this)) { this.col++; return false; } return true; };
Piece.prototype.moveRight = function (grid) { this.col++; if (!grid.valid(this)) { this.col--; return false; } return true; };
Piece.prototype.moveDown  = function (grid) { this.row++; if (!grid.valid(this)) { this.row--; return false; } return true; };

// ═══════════════════════════════════════════════════════
// SECTION 3: BAG RANDOMIZER
// ═══════════════════════════════════════════════════════
function RPG() {
    this._bag = [0,1,2,3,4,5,6];
    this._idx = 7;
}
RPG.prototype.next = function () {
    if (this._idx >= this._bag.length) {
        for (let i = 6; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this._bag[i], this._bag[j]] = [this._bag[j], this._bag[i]];
        }
        this._idx = 0;
    }
    return Piece.fromIndex(this._bag[this._idx++]);
};

// ═══════════════════════════════════════════════════════
// SECTION 4: AI  (adapted from tetrisai/js/ai.js)
// ═══════════════════════════════════════════════════════
// Tuned weights from the genetic tuner (README defaults)
const WEIGHTS_HELL   = { h: 0.510066, l: 0.760666, o: 0.35663,  b: 0.184483 };
const WEIGHTS_MEDIUM = { h: 0.510066, l: 0.600000, o: 0.45000,  b: 0.300000 };

function scoreGrid(grid, w) {
    return -w.h * grid.aggregateHeight()
           +w.l * grid.lines()
           -w.o * grid.holes()
           -w.b * grid.bumpiness();
}

function aiBest(grid, pieces, idx, w) {
    const piece = pieces[idx];
    let bestScore = -Infinity, bestPiece = null;
    const tryPiece = piece.clone();
    for (let rot = 0; rot < 4; rot++) {
        if (rot > 0) tryPiece.rotateCW();
        // Move all the way left
        while (tryPiece.moveLeft(grid));
        while (grid.valid(tryPiece)) {
            // Hard-drop clone
            const dropped = tryPiece.clone();
            while (dropped.moveDown(grid));
            const testGrid = grid.clone();
            testGrid.addPiece(dropped);
            testGrid.clearLines();
            let score;
            if (idx === pieces.length - 1) {
                score = scoreGrid(testGrid, w);
            } else {
                const sub = aiBest(testGrid, pieces, idx + 1, w);
                score = sub ? sub.score : -Infinity;
            }
            if (score > bestScore) { bestScore = score; bestPiece = tryPiece.clone(); }
            tryPiece.col++;
        }
    }
    return bestPiece ? { piece: bestPiece, score: bestScore } : null;
}

// ═══════════════════════════════════════════════════════
// SECTION 5: GAME ENGINE
// ═══════════════════════════════════════════════════════
const ROWS = 22, COLS = 10;
const VISIBLE_ROWS = 20;    // rows 2..21 are visible
const SCORE_TABLE = [0, 100, 300, 500, 800]; // 0-4 lines

// Gravity: ms between automatic drops (indexed by level 1-15+)
function gravityMs(level) {
    const base = 1000 - (level - 1) * 80;
    return Math.max(50, base);
}

class TetrisEngine {
    constructor(opts = {}) {
        this.onScore    = opts.onScore    || (() => {});
        this.onGameOver = opts.onGameOver || (() => {});
        this.aiMode     = opts.aiMode || null; // null=human, 'random', 'medium', 'hell'

        this.reset();
    }

    reset() {
        this.grid    = new Grid(ROWS, COLS);
        this.rpg     = new RPG();
        this.score   = 0;
        this.level   = 1;
        this.linesCleared = 0;
        this.over    = false;

        this.current = this._spawn();
        this.next    = this._spawn();

        this.dropAcc    = 0;
        this.lockDelay  = 0;
        this.grounded   = false;

        // AI state
        this._aiTarget  = null;
        this._aiTimer   = 0;
        this._aiDecided = false;
        this._aiRandCol = null;   // target column for random mode
    }

    _spawn() {
        const p = this.rpg.next();
        p.row = 0;
        return p;
    }

    _lock() {
        this.grid.addPiece(this.current);
        const lines = this.grid.clearLines();
        this.linesCleared += lines;
        this.level = Math.floor(this.linesCleared / 10) + 1;
        this.score += SCORE_TABLE[lines] * this.level;
        this.onScore(this.score, this.level, lines);

        this.current = this.next;
        this.next    = this._spawn();
        this._aiTarget  = null;
        this._aiDecided = false;
        this._aiRandCol = null;

        if (!this.grid.valid(this.current)) {
            this.over = true;
            this.onGameOver(this.score, this.level);
        }
    }

    // ── Human input ───────────────────────────────────────────────────
    keyLeft()   { if (!this.over) this.current.moveLeft(this.grid); }
    keyRight()  { if (!this.over) this.current.moveRight(this.grid); }
    keyDown()   { if (!this.over && !this.current.moveDown(this.grid)) this._lock(); else this.dropAcc = 0; }
    keyRotate() { if (!this.over) this.current.rotate(this.grid); }
    keyDrop()   { if (!this.over) { while (this.current.moveDown(this.grid)); this._lock(); } }

    // ── AI move decisions (rotation + lateral only; dropping via gravity) ──
    _aiMoveDecide() {
        // On new piece: compute target once
        if (!this._aiDecided) {
            if (this.aiMode === 'random') {
                // Apply random rotations immediately (visual consistency)
                const rot = Math.floor(Math.random() * 4);
                for (let i = 0; i < rot; i++) {
                    const saved = this.current.cells.map(r => [...r]);
                    this.current.rotateCW();
                    if (!this.grid.valid(this.current)) {
                        this.current.cells = saved; break;
                    }
                }
                const maxCol = COLS - this.current.dimension;
                this._aiRandCol = Math.max(0, Math.floor(Math.random() * (maxCol + 1)));
            } else {
                const w = this.aiMode === 'hell' ? WEIGHTS_HELL : WEIGHTS_MEDIUM;
                const lookahead = this.aiMode === 'hell' ? 2 : 1;
                const pieces = [this.current, this.next].slice(0, lookahead);
                const result = aiBest(this.grid, pieces, 0, w);
                this._aiTarget = result ? result.piece : null;
            }
            this._aiDecided = true;
            this._aiTimer   = 0;
        }

        // One lateral/rotation move per N frames
        // easy(random)=8≈133ms, medium=5≈83ms, hell=2≈33ms
        const interval = this.aiMode === 'hell' ? 2 : this.aiMode === 'medium' ? 5 : 8;
        this._aiTimer++;
        if (this._aiTimer < interval) return;
        this._aiTimer = 0;

        if (this.aiMode === 'random') {
            if      (this.current.col < this._aiRandCol) this.current.moveRight(this.grid);
            else if (this.current.col > this._aiRandCol) this.current.moveLeft(this.grid);
            // Aligned → gravity does the rest
            return;
        }

        if (!this._aiTarget) return;
        const rot = this._computeRotations(this.current, this._aiTarget);
        if (rot > 0)                                     { this.current.rotate(this.grid); return; }
        if (this.current.col < this._aiTarget.col)       { this.current.moveRight(this.grid); return; }
        if (this.current.col > this._aiTarget.col)       { this.current.moveLeft(this.grid); return; }
        // Hell: hard-drop immediately once aligned
        if (this.aiMode === 'hell') { while (this.current.moveDown(this.grid)); this._lock(); return; }
        // medium/random: let gravity handle the drop naturally
    }

    _computeRotations(current, target) {
        // Count CW rotations to match target orientation
        const tc = current.clone();
        for (let rot = 0; rot < 4; rot++) {
            if (tc.cells.toString() === target.cells.toString()) return rot;
            tc.rotateCW();
        }
        return 0;
    }

    // ── Update (called every frame with dt in ms) ─────────────────────
    update(dt) {
        if (this.over) return;

        // AI makes rotation/lateral decisions (gravity handles dropping)
        if (this.aiMode) this._aiMoveDecide();

        // Gravity — hell falls fast during positioning; medium/random are more natural
        const gravMult = this.aiMode === 'hell'   ? 0.15
                       : this.aiMode === 'medium' ? 0.65
                       : this.aiMode === 'random' ? 0.80
                       : 1.0;
        this.dropAcc += dt;
        const grav = gravityMs(this.level) * gravMult;
        while (this.dropAcc >= grav) {
            this.dropAcc -= grav;
            if (!this.current.moveDown(this.grid)) {
                this._lock();
                return;
            }
        }
    }

    // ── State snapshot for rendering ──────────────────────────────────
    snapshot() {
        return {
            cells:   this.grid.cells,
            current: this.current,
            next:    this.next,
            score:   this.score,
            level:   this.level,
            lines:   this.linesCleared,
            over:    this.over,
        };
    }
}

// ═══════════════════════════════════════════════════════
// SECTION 6: RENDERER
// ═══════════════════════════════════════════════════════
const CS = 22;   // cell size px
const BW = COLS * CS;          // board width  = 220
const BH = VISIBLE_ROWS * CS;  // board height = 440
const PANEL_W = BW + 80;       // +80 for next/info panel

const COLOR_BG   = '#111';
const COLOR_GRID = '#222';
const COLOR_TEXT = '#ddd';

function cellColor(v) {
    return v > 0 ? PIECE_COLORS[v - 1] : null;
}

function renderBoard(ctx, snap, ox, oy, label) {
    // Background
    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(ox, oy, BW, BH);

    // Grid lines
    ctx.strokeStyle = COLOR_GRID;
    ctx.lineWidth = 0.5;
    for (let r = 0; r <= VISIBLE_ROWS; r++) {
        ctx.beginPath();
        ctx.moveTo(ox, oy + r * CS);
        ctx.lineTo(ox + BW, oy + r * CS);
        ctx.stroke();
    }
    for (let c = 0; c <= COLS; c++) {
        ctx.beginPath();
        ctx.moveTo(ox + c * CS, oy);
        ctx.lineTo(ox + c * CS, oy + BH);
        ctx.stroke();
    }

    // Locked cells (skip hidden rows 0-1)
    for (let r = 2; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const v = snap.cells[r][c];
            if (v) _drawCell(ctx, ox + c * CS, oy + (r - 2) * CS, CS, cellColor(v));
        }
    }

    // Ghost piece (use a temporary Grid to validate moves)
    if (!snap.over && snap.current) {
        const tmpGrid = new Grid(ROWS, COLS);
        tmpGrid.cells = snap.cells.map(r => [...r]);
        const ghost = snap.current.clone();
        while (ghost.moveDown(tmpGrid));
        const col = cellColor(snap.current.cells.flat().find(v => v) || 1);
        for (let r = 0; r < ghost.cells.length; r++)
            for (let c = 0; c < ghost.cells[r].length; c++)
                if (ghost.cells[r][c] && ghost.row + r >= 2) {
                    const px = ox + (ghost.col + c) * CS;
                    const py = oy + (ghost.row + r - 2) * CS;
                    ctx.strokeStyle = col;
                    ctx.lineWidth = 1;
                    ctx.strokeRect(px + 1, py + 1, CS - 2, CS - 2);
                }
    }

    // Current piece
    if (!snap.over && snap.current) {
        const p = snap.current;
        for (let r = 0; r < p.cells.length; r++)
            for (let c = 0; c < p.cells[r].length; c++)
                if (p.cells[r][c] && p.row + r >= 2) {
                    _drawCell(ctx, ox + (p.col + c) * CS, oy + (p.row + r - 2) * CS, CS, cellColor(p.cells[r][c]));
                }
    }

    // Border
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 2;
    ctx.strokeRect(ox, oy, BW, BH);

    // Label
    ctx.fillStyle = COLOR_TEXT;
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, ox + BW / 2, oy - 8);
}

function renderSidePanel(ctx, snap, ox, oy) {
    // Next piece preview
    ctx.fillStyle = COLOR_TEXT;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('NEXT', ox + 4, oy + 14);

    const nx = ox + 8, ny = oy + 20;
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(nx - 4, ny - 4, 64, 64);
    if (snap.next) {
        const p = snap.next;
        const off = p.dimension === 4 ? 0 : p.dimension === 2 ? 10 : 5;
        for (let r = 0; r < p.cells.length; r++)
            for (let c = 0; c < p.cells[r].length; c++)
                if (p.cells[r][c])
                    _drawCell(ctx, nx + off + c * 14, ny + off + r * 14, 14, cellColor(p.cells[r][c]));
    }

    // Score / Level / Lines
    const tx = ox + 4, sy = ny + 76;
    ctx.fillStyle = '#888';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('SCORE', tx, sy);
    ctx.fillStyle = COLOR_TEXT;
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.fillText(snap.score, tx, sy + 16);

    ctx.fillStyle = '#888';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText('LEVEL', tx, sy + 38);
    ctx.fillStyle = COLOR_TEXT;
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.fillText(snap.level, tx, sy + 54);

    ctx.fillStyle = '#888';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText('LINES', tx, sy + 76);
    ctx.fillStyle = COLOR_TEXT;
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.fillText(snap.lines, tx, sy + 92);

    // Game over overlay
    if (snap.over) {
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.fillRect(ox - BW, oy - 20, BW + 80, BH + 30);
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 15px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('TOPPED', ox - BW / 2, oy + BH / 2 - 10);
        ctx.fillText('OUT', ox - BW / 2, oy + BH / 2 + 10);
    }
}

function _drawCell(ctx, x, y, size, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x + 1, y + 1, size - 2, size - 2);
    // Highlight top-left
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.fillRect(x + 1, y + 1, size - 2, 3);
    ctx.fillRect(x + 1, y + 1, 3, size - 2);
    // Shadow bottom-right
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.fillRect(x + 1, y + size - 4, size - 2, 3);
    ctx.fillRect(x + size - 4, y + 1, 3, size - 2);
}

// ═══════════════════════════════════════════════════════
// SECTION 7: DUAL MOUNT
// ═══════════════════════════════════════════════════════
function mountDualTetris(container, ctx, difficulty) {
    // Canvas dimensions: two boards side by side
    const GAP = 16;
    const PAD_X = 8;
    const BOARD_Y = 30;
    const PANEL = 72;   // side panel width
    const CW = BW + PANEL;     // canvas width per player: 292
    const CH = BH + 40;        // canvas height: 480
    const TOTAL_W = CW * 2 + GAP + PAD_X * 2;

    container.innerHTML = `
      <div class="tetris-dual-wrap" style="--tetris-canvas-width:${TOTAL_W}px">
        <canvas id="td-canvas" class="tetris-dual-canvas" width="${TOTAL_W}" height="${CH}"></canvas>
        <p class="controls-hint tetris-controls-hint">
          ← → move &nbsp;|&nbsp; ↑ / Z rotate &nbsp;|&nbsp; ↓ soft-drop &nbsp;|&nbsp; Space hard-drop
        </p>
      </div>`;

    const canvas = document.getElementById('td-canvas');
    const c = canvas.getContext('2d');

    let p1Score = 0, p2Score = 0, p1Done = false, p2Done = false;

    const checkEnd = () => {
        if (!p1Done || !p2Done) return;
        ctx.onEnd({
            playerScore: p1Score,
            aiScore:     p2Score,
            message: p1Score > p2Score
                ? `You win! You ${p1Score} — AI ${p2Score}`
                : p2Score > p1Score
                ? `AI wins! AI ${p2Score} — You ${p1Score}`
                : `Draw! Both scored ${p1Score}`,
        });
    };

    // Player 1 — human
    const eng1 = new TetrisEngine({
        aiMode: null,
        onScore(score) { p1Score = score; ctx.onScore(p1Score, p2Score); },
        onGameOver(score) { p1Score = score; p1Done = true; ctx.onScore(p1Score, p2Score); checkEnd(); },
    });

    // Player 2 — AI
    const aiMode = difficulty === 'hell' ? 'hell' : difficulty === 'medium' ? 'medium' : 'random';
    const eng2 = new TetrisEngine({
        aiMode,
        onScore(score) { p2Score = score; ctx.onScore(p1Score, p2Score); },
        onGameOver(score) { p2Score = score; p2Done = true; ctx.onScore(p1Score, p2Score); checkEnd(); },
    });

    ctx.onScore(0, 0);

    // ── Keyboard handling ──────────────────────────────────────────────
    const keys = {};
    const DAS = 200, ARR = 50; // ms
    const dasState = { left: 0, right: 0, down: 0 };

    const onKeyDown = (e) => {
        if (keys[e.code]) return;
        keys[e.code] = true;
        switch (e.code) {
            case 'ArrowLeft':  e.preventDefault(); eng1.keyLeft();   dasState.left  = Date.now() + DAS; break;
            case 'ArrowRight': e.preventDefault(); eng1.keyRight();  dasState.right = Date.now() + DAS; break;
            case 'ArrowDown':  e.preventDefault(); eng1.keyDown();   dasState.down  = Date.now() + DAS; break;
            case 'ArrowUp':    e.preventDefault(); eng1.keyRotate(); break;
            case 'KeyZ':       e.preventDefault(); eng1.keyRotate(); break;
            case 'Space':      e.preventDefault(); eng1.keyDrop();   break;
        }
    };
    const onKeyUp = (e) => { keys[e.code] = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup',   onKeyUp);

    // ── Game loop ─────────────────────────────────────────────────────
    let last = performance.now(), rafId = 0, stopped = false;

    const loop = (ts) => {
        if (stopped) return;
        const dt = Math.min(ts - last, 80);
        last = ts;

        // DAS auto-repeat
        const now = Date.now();
        if (keys['ArrowLeft']  && now > dasState.left  + ARR) { eng1.keyLeft();  dasState.left  = now - ARR + ARR; }
        if (keys['ArrowRight'] && now > dasState.right + ARR) { eng1.keyRight(); dasState.right = now - ARR + ARR; }
        if (keys['ArrowDown']  && now > dasState.down  + ARR) { eng1.keyDown();  dasState.down  = now - ARR + ARR; }

        eng1.update(dt);
        eng2.update(dt);

        // Render
        c.fillStyle = '#0a0a0a';
        c.fillRect(0, 0, TOTAL_W, CH);

        const s1 = eng1.snapshot(), s2 = eng2.snapshot();

        // Player 1
        const x1 = PAD_X;
        renderBoard(c, s1, x1, BOARD_Y, '🎮 You (Player 1)');
        renderSidePanel(c, s1, x1 + BW + 4, BOARD_Y);

        // Player 2
        const x2 = x1 + CW + GAP;
        renderBoard(c, s2, x2, BOARD_Y, '🤖 AI (Player 2)');
        renderSidePanel(c, s2, x2 + BW + 4, BOARD_Y);

        rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);

    return {
        unmount() {
            stopped = true;
            cancelAnimationFrame(rafId);
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup',   onKeyUp);
            container.innerHTML = '';
        },
    };
}

// ═══════════════════════════════════════════════════════
// SECTION 8: PUBLIC API
// ═══════════════════════════════════════════════════════
let _runtime = null;

return {
    id: 'tetris',
    title: 'Tetris',

    mount(container, ctx) {
        const difficulty = ctx.difficulty || 'medium';
        _runtime = mountDualTetris(container, ctx, difficulty);
    },

    unmount() {
        _runtime?.unmount();
        _runtime = null;
    },
};

})();
