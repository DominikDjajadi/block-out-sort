/*
 * Block Out Sort - solver / validator
 *
 * Breadth-first search over board configurations. Because a drag can stop at
 * any reachable cell along its slide, every intermediate stop (and every legal
 * exit) is treated as a distinct successor with cost 1. BFS therefore returns
 * the minimum number of moves (drags) to clear the board, which we use as the
 * difficulty signal.
 *
 * The "cleared" count is derived from how many blocks remain
 * (cleared = totalBlocks - remaining), so a state is fully described by the
 * set of remaining blocks and their positions. That keeps the search state
 * compact and lets us hash/dedupe configurations.
 *
 * Reuses Game (geometry, sliding, exit rules) and DIRS from game.js.
 * Works in the browser (exposed on window) and under Node via eval.
 */

(function (root) {
  "use strict";

  function cloneBlocks(blocks) {
    return blocks.map((b) => ({
      id: b.id,
      color: b.color,
      unlockAt: b.unlockAt,
      cells: b.cells.map((c) => ({ r: c.r, c: c.c })),
    }));
  }

  function translate(block, dirName, steps) {
    const d = DIRS[dirName];
    for (const cell of block.cells) {
      cell.r += d.dr * steps;
      cell.c += d.dc * steps;
    }
  }

  // Canonical, order-independent key. Same-color/shape blocks are
  // interchangeable, so sorting collapses equivalent configurations.
  function stateKey(blocks) {
    const parts = blocks.map((b) => {
      const cells = b.cells
        .slice()
        .sort((a, z) => a.r - z.r || a.c - z.c)
        .map((c) => c.r + "," + c.c)
        .join(";");
      return b.color + "#" + (b.unlockAt || 0) + "#" + cells;
    });
    parts.sort();
    return parts.join("|");
  }

  // All single-move successors of a configuration.
  function expand(g, blocks, total) {
    g.blocks = blocks;
    g.cleared = total - blocks.length;
    const out = [];
    for (const b of blocks) {
      if (g.isBlockFrozen(b)) continue;
      for (const dirName of ["up", "down", "left", "right"]) {
        const s = g.computeSlide(b, dirName);
        for (let step = 1; step <= s.steps; step++) {
          const nb = cloneBlocks(blocks);
          translate(nb.find((x) => x.id === b.id), dirName, step);
          out.push({ blocks: nb, move: { color: b.color, dir: dirName, steps: step, exit: false } });
        }
        if (s.reason === "edge" && s.canExit) {
          const nb = cloneBlocks(blocks).filter((x) => x.id !== b.id);
          out.push({ blocks: nb, move: { color: b.color, dir: dirName, steps: s.steps, exit: true } });
        }
      }
    }
    return out;
  }

  function reconstruct(visited, goalKey) {
    const path = [];
    let k = goalKey;
    // The cleared board hashes to "", so guard on null (start marker) rather
    // than truthiness.
    while (k !== null && k !== undefined) {
      const e = visited.get(k);
      if (!e) break; // reached the start state (value is null)
      path.push(e.move);
      k = e.parent;
    }
    return path.reverse();
  }

  function solveLevel(level, opts) {
    opts = opts || {};
    const maxNodes = opts.maxNodes || 300000;

    const g = new Game(level);
    const total = g.totalBlocks;
    const start = cloneBlocks(g.blocks);

    if (start.length === 0) return { solvable: true, moves: 0, path: [], explored: 0 };

    const startKey = stateKey(start);
    const visited = new Map([[startKey, null]]);
    const queue = [{ blocks: start, key: startKey, depth: 0 }];
    let head = 0;
    let explored = 0;

    while (head < queue.length) {
      const cur = queue[head++];
      explored++;
      if (explored > maxNodes) {
        return { solvable: false, exhausted: true, explored, moves: Infinity };
      }
      for (const sc of expand(g, cur.blocks, total)) {
        const k = stateKey(sc.blocks);
        if (visited.has(k)) continue;
        visited.set(k, { parent: cur.key, move: sc.move });
        if (sc.blocks.length === 0) {
          return {
            solvable: true,
            moves: cur.depth + 1,
            path: reconstruct(visited, k),
            explored,
          };
        }
        queue.push({ blocks: sc.blocks, key: k, depth: cur.depth + 1 });
      }
    }
    return { solvable: false, exhausted: false, explored, moves: Infinity };
  }

  /*
   * Fast solvability check that uses ONLY exit moves (no shuffling blocks
   * aside). Branching is just "which block exits next", so this is far cheaper
   * than the full BFS. It is complete for any level that can be solved by
   * exiting blocks in some order - which every generator output is, by
   * construction - and for those the minimum move count is exactly the block
   * count (one drag per block). It may report a false negative for hand-made
   * puzzles that genuinely require sliding a block out of the way first; use
   * solveLevel for those.
   */
  function solveExitOnly(level) {
    const g = new Game(level);
    const total = g.totalBlocks;
    const start = cloneBlocks(g.blocks);
    if (start.length === 0) return { solvable: true, moves: 0, order: [], exitOnly: true };

    const dead = new Set();
    const order = [];

    function dfs(blocks) {
      if (blocks.length === 0) return true;
      const k = stateKey(blocks);
      if (dead.has(k)) return false;
      for (const b of blocks) {
        g.blocks = blocks;
        g.cleared = total - blocks.length;
        if (g.isBlockFrozen(b)) continue;
        for (const dir of ["up", "down", "left", "right"]) {
          const s = g.computeSlide(b, dir);
          if (s.reason === "edge" && s.canExit) {
            order.push({ color: b.color, dir, exit: true });
            if (dfs(blocks.filter((x) => x.id !== b.id))) return true;
            order.pop();
            break; // one exit direction per block is enough
          }
        }
      }
      dead.add(k);
      return false;
    }

    const ok = dfs(start);
    return {
      solvable: ok,
      moves: ok ? start.length : Infinity,
      order: ok ? order.slice() : null,
      exitOnly: true,
    };
  }

  /* ---- A* (optimal min-moves, faster than plain BFS) ---- */

  class MinHeap {
    constructor() { this.a = []; }
    size() { return this.a.length; }
    push(n) {
      const a = this.a;
      a.push(n);
      let i = a.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (less(a[i], a[p])) { [a[i], a[p]] = [a[p], a[i]]; i = p; } else break;
      }
    }
    pop() {
      const a = this.a;
      const top = a[0];
      const last = a.pop();
      if (a.length) {
        a[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = 2 * i + 2;
          let s = i;
          if (l < a.length && less(a[l], a[s])) s = l;
          if (r < a.length && less(a[r], a[s])) s = r;
          if (s === i) break;
          [a[i], a[s]] = [a[s], a[i]];
          i = s;
        }
      }
      return top;
    }
  }
  // lower f first; tie-break toward fewer remaining blocks (closer to goal)
  function less(x, y) { return x.f !== y.f ? x.f < y.f : x.h < y.h; }

  function solveAStar(level, opts) {
    opts = opts || {};
    const maxNodes = opts.maxNodes || 250000;
    const g = new Game(level);
    const total = g.totalBlocks;
    const start = cloneBlocks(g.blocks);
    if (start.length === 0) return { solvable: true, moves: 0, path: [], explored: 0 };

    const startKey = stateKey(start);
    const gScore = new Map([[startKey, 0]]);
    const came = new Map([[startKey, null]]);
    const heap = new MinHeap();
    heap.push({ key: startKey, blocks: start, g: 0, h: start.length, f: start.length });
    let explored = 0;

    while (heap.size()) {
      const cur = heap.pop();
      if (cur.g > (gScore.get(cur.key) ?? Infinity)) continue; // stale
      explored++;
      if (explored > maxNodes) return { solvable: false, exhausted: true, explored, moves: Infinity };
      if (cur.blocks.length === 0) {
        return { solvable: true, moves: cur.g, path: reconstruct(came, cur.key), explored };
      }
      for (const sc of expand(g, cur.blocks, total)) {
        const k = stateKey(sc.blocks);
        const ng = cur.g + 1;
        if (ng < (gScore.get(k) ?? Infinity)) {
          gScore.set(k, ng);
          came.set(k, { parent: cur.key, move: sc.move });
          const h = sc.blocks.length;
          heap.push({ key: k, blocks: sc.blocks, g: ng, h, f: ng + h });
        }
      }
    }
    return { solvable: false, exhausted: false, explored, moves: Infinity };
  }

  function difficultyLabel(moves) {
    if (!isFinite(moves)) return "unknown";
    if (moves <= 4) return "trivial";
    if (moves <= 8) return "easy";
    if (moves <= 14) return "medium";
    if (moves <= 22) return "hard";
    return "expert";
  }

  // Friendly wrapper for level authoring (uses A* for speed + optimal moves).
  function validateLevel(level, opts) {
    const r = solveAStar(level, opts);
    return {
      name: level.name,
      solvable: r.solvable,
      minMoves: r.moves,
      difficulty: r.solvable ? difficultyLabel(r.moves) : null,
      exhausted: !!r.exhausted,
      explored: r.explored,
      path: r.path || null,
    };
  }

  function describePath(path) {
    if (!path) return "";
    return path
      .map((m, i) => `${i + 1}. ${m.color} ${m.dir}${m.exit ? " -> EXIT" : " " + m.steps}`)
      .join("\n");
  }

  const api = { solveLevel, solveAStar, solveExitOnly, validateLevel, difficultyLabel, describePath };
  root.Solver = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
