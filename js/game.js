/*
 * Block Out Sort - core engine
 *
 * Coordinate system: grid cells addressed by row (r, top->bottom) and
 * column (c, left->right). Blocks are sets of absolute cells. Movement is
 * axis-aligned sliding; a block stops when the next cell would leave the
 * board or overlap another block / non-playable cell.
 *
 * A block is "cleared" when it slides off the board through an exit gate of
 * the same color whose opening fully covers the block's crossing lane.
 */

const PALETTE = {
  red:    { base: "#ff5d6c", dark: "#e23a4d", light: "#ff97a1" },
  blue:   { base: "#4d8af0", dark: "#2f6bd6", light: "#8fb6f8" },
  green:  { base: "#3ec46d", dark: "#27a455", light: "#85e0a2" },
  yellow: { base: "#ffcb3d", dark: "#e8ab12", light: "#ffe08a" },
  purple: { base: "#a667f0", dark: "#8744d8", light: "#cda3f7" },
  orange: { base: "#ff924c", dark: "#ed6f25", light: "#ffbd8c" },
  teal:   { base: "#27c6c0", dark: "#13a39e", light: "#7fe3df" },
  pink:   { base: "#ff70b8", dark: "#ec479b", light: "#ffa8d4" },
};

const DIRS = {
  up:    { dr: -1, dc: 0, edge: "top" },
  down:  { dr: 1,  dc: 0, edge: "bottom" },
  left:  { dr: 0,  dc: -1, edge: "left" },
  right: { dr: 0,  dc: 1,  edge: "right" },
};

const key = (r, c) => r + "," + c;

let _uid = 1;

class Game {
  constructor(level) {
    this.load(level);
  }

  load(level) {
    this.level = level;
    this.cols = level.cols;
    this.rows = level.rows;
    this.moves = 0;
    this.cleared = 0;

    // Permanent holes (non-rectangular boards).
    this.holes = new Set((level.holes || []).map(([r, c]) => key(r, c)));

    // Locked regions: cells blocked until `cleared >= unlockAt`.
    this.lockedRegions = (level.lockedRegions || []).map((reg) => ({
      cells: reg.cells.map(([r, c]) => ({ r, c })),
      unlockAt: reg.unlockAt || 0,
    }));

    // Blocks.
    this.blocks = level.blocks.map((b) => ({
      id: _uid++,
      color: b.color,
      cells: b.cells.map(([r, c]) => ({ r, c })),
      unlockAt: b.unlockAt || 0, // frozen until this many cleared
    }));

    this.totalBlocks = this.blocks.length;

    // Exits.
    this.exits = level.exits.map((e) => ({
      edge: e.edge,
      start: e.start,
      length: e.length,
      color: e.color,
      unlockAt: e.unlockAt || 0,
    }));
  }

  /* ---- queries ---- */

  inBounds(r, c) {
    return r >= 0 && r < this.rows && c >= 0 && c < this.cols;
  }

  isRegionLocked(r, c) {
    for (const reg of this.lockedRegions) {
      if (this.cleared >= reg.unlockAt) continue;
      for (const cell of reg.cells) {
        if (cell.r === r && cell.c === c) return true;
      }
    }
    return false;
  }

  // A cell that a block may occupy/pass into.
  isPlayable(r, c) {
    if (!this.inBounds(r, c)) return false;
    if (this.holes.has(key(r, c))) return false;
    if (this.isRegionLocked(r, c)) return false;
    return true;
  }

  isBlockFrozen(block) {
    return this.cleared < block.unlockAt;
  }

  isExitActive(exit) {
    return this.cleared >= exit.unlockAt;
  }

  // Map of occupied cells -> block, excluding `exceptId`.
  occupancy(exceptId) {
    const map = new Map();
    for (const b of this.blocks) {
      if (b.id === exceptId) continue;
      for (const cell of b.cells) map.set(key(cell.r, cell.c), b);
    }
    return map;
  }

  /*
   * Compute how far `block` can slide in `dirName`.
   * Returns { steps, reason: 'block' | 'edge', canExit }.
   *  - steps: max whole cells it can advance while staying on the board.
   *  - reason 'edge': it reached the board boundary in that direction.
   *  - reason 'block': it was stopped by another block / locked cell.
   *  - canExit: true when, at the boundary, a matching active exit fully
   *    covers the block's crossing lane (so it may slide off entirely).
   */
  computeSlide(block, dirName) {
    const dir = DIRS[dirName];
    const occ = this.occupancy(block.id);
    let steps = 0;
    let reason = "edge";

    // Safety bound: never loop more than board span.
    const limit = this.rows + this.cols + 2;

    while (steps < limit) {
      let offBoard = false;
      let blocked = false;
      for (const cell of block.cells) {
        const nr = cell.r + dir.dr * (steps + 1);
        const nc = cell.c + dir.dc * (steps + 1);
        if (!this.inBounds(nr, nc)) {
          offBoard = true;
        } else if (this.holes.has(key(nr, nc)) || this.isRegionLocked(nr, nc) || occ.has(key(nr, nc))) {
          blocked = true;
        }
      }
      if (blocked) {
        reason = "block";
        break;
      }
      if (offBoard) {
        reason = "edge";
        break;
      }
      steps++;
    }

    let canExit = false;
    if (reason === "edge") canExit = this.canExitThrough(block, dirName, occ);
    return { steps, reason, canExit };
  }

  /*
   * Can `block` fully and legally leave the board in `dirName`?
   *
   * Corrected exit rule (was previously a gate-coverage check only):
   *  1. An active matching gate must cover the block's whole crossing lane.
   *  2. The block must be translatable far enough for EVERY cell to leave the
   *     board, and during that entire extrusion every cell still inside the
   *     board must remain collision-free (no other block / hole / locked cell).
   *
   * Without (2) an irregular piece could be cleared even though a trailing cell
   * would pass through another block while the piece is partially off-board.
   * `occ` is the occupancy map excluding this block (as built by computeSlide);
   * it is rebuilt here if not supplied so the method is safe to call directly.
   */
  canExitThrough(block, dirName, occ) {
    const dir = DIRS[dirName];
    if (!occ) occ = this.occupancy(block.id);

    // (1) Gate must cover the whole crossing lane.
    const lane = new Set();
    for (const cell of block.cells) {
      lane.add(dir.dr !== 0 ? cell.c : cell.r);
    }
    let gateCovers = false;
    for (const exit of this.exits) {
      if (exit.edge !== dir.edge) continue;
      if (exit.color !== block.color) continue;
      if (!this.isExitActive(exit)) continue;
      let covers = true;
      for (const x of lane) {
        if (x < exit.start || x >= exit.start + exit.length) {
          covers = false;
          break;
        }
      }
      if (covers) { gateCovers = true; break; }
    }
    if (!gateCovers) return false;

    // (2) The block must extrude fully without any on-board cell colliding.
    const limit = this.rows + this.cols + 2;
    for (let s = 1; s <= limit; s++) {
      let allOff = true;
      for (const cell of block.cells) {
        const nr = cell.r + dir.dr * s;
        const nc = cell.c + dir.dc * s;
        if (this.inBounds(nr, nc)) {
          allOff = false;
          if (this.holes.has(key(nr, nc)) || this.isRegionLocked(nr, nc) || occ.has(key(nr, nc))) {
            return false;
          }
        }
      }
      if (allOff) return true;
    }
    return false;
  }

  /* ---- mutations ---- */

  translateBlock(block, dirName, steps) {
    if (steps === 0) return;
    const dir = DIRS[dirName];
    for (const cell of block.cells) {
      cell.r += dir.dr * steps;
      cell.c += dir.dc * steps;
    }
  }

  removeBlock(block) {
    const idx = this.blocks.indexOf(block);
    if (idx >= 0) {
      this.blocks.splice(idx, 1);
      this.cleared++;
    }
  }

  isWon() {
    return this.blocks.length === 0;
  }
}
