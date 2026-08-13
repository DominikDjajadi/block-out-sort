/*
 * Block Out Sort - level generator (reverse construction)
 *
 * Two-phase build:
 *
 * 1) Gates. Exactly ONE gate per color. Gates never overlap (each color gets a
 *    disjoint span; same-edge gates are packed into separate slots).
 *
 * 2) Blocks. Each block is created just inside ITS COLOR'S gate and slid inward
 *    until it stops against a wall or an already-placed block. A block is only
 *    accepted if it can still slide back OUT to its gate against everything
 *    placed before it - which, because blocks exit in reverse placement order,
 *    guarantees a valid exit-only solution. So every base level is solvable by
 *    construction.
 *
 * Difficulty (see notes above labelByExtra) is measured by forced extra moves
 * and tuned by board density + reverse-slide scramble depth.
 *
 * Reuses Game / DIRS from game.js and Solver from solver.js.
 */

(function (root) {
  "use strict";

  const SHAPES = {
    mono:   [[0, 0]],
    domino: [[0, 0], [0, 1]],
    triI:   [[0, 0], [0, 1], [0, 2]],
    triL:   [[0, 0], [1, 0], [1, 1]],
    tetO:   [[0, 0], [0, 1], [1, 0], [1, 1]],
    tetI:   [[0, 0], [0, 1], [0, 2], [0, 3]],
    tetT:   [[0, 0], [0, 1], [0, 2], [1, 1]],
    tetL:   [[0, 0], [1, 0], [2, 0], [2, 1]],
    tetS:   [[0, 1], [0, 2], [1, 0], [1, 1]],
    plus:   [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]],
  };
  const DEFAULT_SHAPES = ["domino", "domino", "triI", "triL", "tetO", "tetT", "tetL", "tetS", "plus"];
  const DEFAULT_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "teal", "pink"];
  const INWARD = { top: "down", bottom: "up", left: "right", right: "left" };
  const OUTWARD = { top: "up", bottom: "down", left: "left", right: "right" };
  const OPP = { up: "down", down: "up", left: "right", right: "left" };
  const EDGES = ["top", "bottom", "left", "right"];

  /* ---- rng ---- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)];
  const randInt = (rng, lo, hi) => lo + Math.floor(rng() * (hi - lo + 1));
  function shuffle(rng, arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  /* ---- shapes ---- */
  function normalize(cells) {
    const mr = Math.min(...cells.map((c) => c[0]));
    const mc = Math.min(...cells.map((c) => c[1]));
    return cells.map(([r, c]) => [r - mr, c - mc]);
  }
  function rotate(cells) { return normalize(cells.map(([r, c]) => [c, -r])); }
  function randomOrientation(rng, name) {
    let cells = SHAPES[name].map((c) => c.slice());
    const turns = randInt(rng, 0, 3);
    for (let i = 0; i < turns; i++) cells = rotate(cells);
    return normalize(cells);
  }

  /* ---- gates: one per color, non-overlapping ---- */
  function assignGates(cfg, rng) {
    const colors = shuffle(rng, cfg.colors).slice(0, cfg.colorCount);
    const perEdge = {};
    for (const e of EDGES) perEdge[e] = [];
    const edgeBag = shuffle(rng, EDGES);
    colors.forEach((col, i) => perEdge[edgeBag[i % EDGES.length]].push(col));

    const gates = [];
    const colorGate = {};
    for (const edge of EDGES) {
      const cols = perEdge[edge];
      if (!cols.length) continue;
      const edgeLen = edge === "top" || edge === "bottom" ? cfg.cols : cfg.rows;
      const slot = Math.floor(edgeLen / cols.length);
      if (slot < 1) return null; // too many colors on one edge for this board
      cols.forEach((col, i) => {
        const maxLen = Math.min(cfg.maxGateLen, slot);
        const len = Math.max(1, randInt(rng, Math.min(cfg.minGateLen, maxLen), maxLen));
        const slotStart = i * slot;
        const start = slotStart + randInt(rng, 0, slot - len);
        const gate = { edge, color: col, start, length: len };
        gates.push(gate);
        colorGate[col] = gate;
      });
    }
    return { gates, colorGate, colors };
  }

  /* ---- place one block through its color's gate ---- */
  function placeColored(work, placed, gate, cfg, rng, id) {
    const edge = gate.edge;
    const inward = INWARD[edge];
    const shape = randomOrientation(rng, pick(rng, cfg.shapes));
    const H = Math.max(...shape.map((c) => c[0])) + 1;
    const W = Math.max(...shape.map((c) => c[1])) + 1;

    let cells;
    if (edge === "top" || edge === "bottom") {
      if (W > gate.length || W > cfg.cols) return null; // lane must fit the gate
      const c0 = gate.start + randInt(rng, 0, gate.length - W);
      const rOff = edge === "top" ? 0 : cfg.rows - H;
      if (rOff < 0) return null;
      cells = shape.map(([r, c]) => ({ r: r + rOff, c: c + c0 }));
    } else {
      if (H > gate.length || H > cfg.rows) return null;
      const r0 = gate.start + randInt(rng, 0, gate.length - H);
      const cOff = edge === "left" ? 0 : cfg.cols - W;
      if (cOff < 0) return null;
      cells = shape.map(([r, c]) => ({ r: r + r0, c: c + cOff }));
    }

    const taken = new Set();
    for (const b of placed) for (const cl of b.cells) taken.add(cl.r + "," + cl.c);
    for (const cl of cells) {
      if (!work.isPlayable(cl.r, cl.c)) return null;
      if (taken.has(cl.r + "," + cl.c)) return null;
    }

    const cand = { id, color: gate.color, unlockAt: 0, cells: cells.map((c) => ({ r: c.r, c: c.c })) };
    work.blocks = placed.concat([cand]);

    const slide = work.computeSlide(cand, inward);
    let steps;
    if (slide.steps <= 0) steps = 0;
    else if (rng() < cfg.deepBias) steps = slide.steps;
    else steps = randInt(rng, 0, slide.steps);
    const d = DIRS[inward];
    for (const cl of cand.cells) { cl.r += d.dr * steps; cl.c += d.dc * steps; }

    // Must be able to slide back out to the edge against earlier blocks.
    work.blocks = placed.concat([cand]);
    if (work.computeSlide(cand, OUTWARD[edge]).reason !== "edge") return null;

    return cand;
  }

  function countPlayable(cfg) {
    return cfg.cols * cfg.rows - (cfg.holes ? cfg.holes.length : 0);
  }

  // Pack the board up to a target fill fraction (density drives congestion).
  function buildBase(cfg, rng) {
    const ga = assignGates(cfg, rng);
    if (!ga) return null;
    const work = new Game({ cols: cfg.cols, rows: cfg.rows, holes: cfg.holes || [], blocks: [], exits: ga.gates });
    const placed = [];
    let id = 1;
    let filled = 0;
    const target = Math.round(cfg.density * countPlayable(cfg));

    while (filled < target && placed.length < cfg.maxBlocks) {
      let ok = false;
      for (let t = 0; t < cfg.placeTries; t++) {
        const gate = ga.colorGate[pick(rng, ga.colors)];
        const cand = placeColored(work, placed, gate, cfg, rng, id);
        if (cand) { placed.push(cand); id++; filled += cand.cells.length; ok = true; break; }
      }
      if (!ok) break; // board effectively full
    }
    if (placed.length < 3) return null;
    return { placed, gates: ga.gates, filled };
  }

  /*
   * Cheap human-difficulty proxy: of many RANDOM "just exit whatever can exit"
   * playthroughs, what fraction actually clear the board? If almost any greedy
   * order works the puzzle feels obvious; if few/none do, you must plan.
   */
  function exitOrderSuccess(level, trials, rng) {
    const g = new Game(level);
    const total = g.totalBlocks;
    const base = g.blocks.map((b) => ({ id: b.id, color: b.color, unlockAt: 0, cells: b.cells.map((c) => ({ r: c.r, c: c.c })) }));
    let success = 0;
    for (let t = 0; t < trials; t++) {
      let blocks = base.map((b) => ({ id: b.id, color: b.color, unlockAt: 0, cells: b.cells.map((c) => ({ r: c.r, c: c.c })) }));
      let stuck = false;
      while (blocks.length) {
        g.blocks = blocks;
        g.cleared = total - blocks.length;
        const exitable = [];
        for (const b of blocks) {
          for (const dir of ["up", "down", "left", "right"]) {
            const s = g.computeSlide(b, dir);
            if (s.reason === "edge" && s.canExit) { exitable.push(b); break; }
          }
        }
        if (!exitable.length) { stuck = true; break; }
        const b = exitable[Math.floor(rng() * exitable.length)];
        blocks = blocks.filter((x) => x.id !== b.id);
      }
      if (!stuck) success++;
    }
    return success / trials;
  }

  /* ---- reverse-slide (inverse of a forward slide); preserves solvability ---- */
  function reverseSlide(g, rng) {
    g.cleared = 0;
    for (const b of shuffle(rng, g.blocks)) {
      for (const d of shuffle(rng, ["up", "down", "left", "right"])) {
        const fwd = g.computeSlide(b, d);
        if (fwd.steps !== 0) continue;                       // b must already be at a stop in d
        if (fwd.reason === "edge" && fwd.canExit) continue;  // a stop, not an exit
        const back = OPP[d];
        const bs = g.computeSlide(b, back);
        if (bs.steps <= 0) continue;
        const m = randInt(rng, 1, bs.steps);
        const dd = DIRS[back];
        for (const cl of b.cells) { cl.r += dd.dr * m; cl.c += dd.dc * m; }
        return true;
      }
    }
    return false;
  }

  function assemble(placed, gates, cfg, name) {
    const used = new Set(placed.map((b) => b.color));
    const level = {
      name: name || "Generated Level",
      cols: cfg.cols,
      rows: cfg.rows,
      blocks: placed.map((b) => ({ color: b.color, cells: b.cells.map((c) => [c.r, c.c]) })),
      exits: gates.filter((g) => used.has(g.color)), // drop gates with no blocks
    };
    if (cfg.holes && cfg.holes.length) level.holes = cfg.holes;
    return level;
  }

  function interlockOf(level) {
    const g = new Game(level);
    let immediate = 0;
    for (const b of g.blocks) {
      for (const dir of ["up", "down", "left", "right"]) {
        const s = g.computeSlide(b, dir);
        if (s.reason === "edge" && s.canExit) { immediate++; break; }
      }
    }
    return g.blocks.length - immediate;
  }

  /* ---- top-level ---- */
  function defaults(opts) {
    const cols = opts.cols || 6;
    const rows = opts.rows || 6;
    const diff = opts.difficulty || "hard";
    // Fewer gates => more blocks per color => more competition for each exit.
    const colorCount = opts.colorCount || (diff === "easy" ? 4 : 3);
    // Density is the biggest lever for "congested, must-plan" boards.
    const density = opts.density != null ? opts.density : (diff === "easy" ? 0.34 : diff === "medium" ? 0.5 : 0.62);
    const maxBlocks = opts.blocks || Math.floor(cols * rows * 0.8);
    return {
      name: opts.name,
      cols, rows,
      difficulty: diff,
      density,
      maxBlocks,
      colors: opts.colors || DEFAULT_COLORS,
      colorCount,
      shapes: opts.shapes || DEFAULT_SHAPES,
      holes: opts.holes || [],
      minGateLen: opts.minGateLen || 2,
      maxGateLen: opts.maxGateLen || 3,
      deepBias: opts.deepBias != null ? opts.deepBias : (diff === "easy" ? 0.4 : diff === "hard" ? 0.9 : 0.65),
      placeTries: opts.placeTries || 80,
      // reverse-slide budget (drives the difficulty progression). Easy stays at
      // the sparse base; medium/hard scramble to force intermediate moves.
      slideBudget: opts.slideBudget != null ? opts.slideBudget : (diff === "easy" ? 0 : maxBlocks * 4),
      // target number of forced non-exit ("shuffle") moves in the solution
      targetExtra: opts.targetExtra != null ? opts.targetExtra : (diff === "easy" ? 0 : diff === "medium" ? 2 : 4),
    };
  }

  function snapshot(blocks) {
    return blocks.map((b) => ({ color: b.color, cells: b.cells.map((c) => ({ r: c.r, c: c.c })) }));
  }

  /*
   * Difficulty is measured by FORCED EXTRA MOVES = (optimal solution length) -
   * (block count). Since one block exits per drag, any "extra" move is a
   * non-exit shuffle the player is forced to make. Boards jump straight from
   * "any exit order works" to "needs a shuffle", and each further reverse-slide
   * tends to add another forced move - so scramble DEPTH is the difficulty dial:
   *   easy   = 0 extra (just exit things; obvious)
   *   medium = 1-2 extra (a shuffle or two)
   *   hard   = 3+ extra (multiple forced maneuvers)
   */
  function labelByExtra(extra) {
    if (extra == null) return "hard"; // solver capped on a scrambled board
    if (extra <= 0) return "easy";
    if (extra <= 2) return "medium";
    return "hard";
  }

  /*
   * opts.measure (default true): verify the optimal move count with A* and label
   * difficulty exactly. Set false for a FAST path (used by the in-game button):
   * it skips A* and uses scramble depth past the forced-move threshold as the
   * difficulty estimate. Solvability is guaranteed by construction either way.
   */
  function generate(opts) {
    opts = opts || {};
    const seed = opts.seed != null ? (opts.seed >>> 0) : (Math.random() * 4294967296) >>> 0;
    const cfg = defaults(opts);
    const rng = mulberry32(seed);
    const measure = opts.measure !== false;
    const maxAttempts = opts.maxAttempts || (measure ? 60 : 200);
    const TRIALS = 20;
    const playable = countPlayable(cfg);
    const want = cfg.difficulty;
    const cap = want === "hard" ? 350000 : 200000;

    let best = null;       // closest fallback by |estimate - target|
    let bestErr = Infinity;

    const consider = (level, density, depth, forcedAt) => {
      let extra = null, minMoves = null, capped = false;
      if (measure && root.Solver) {
        const v = root.Solver.solveAStar(level, { maxNodes: cap });
        if (v.exhausted) capped = true;
        else { minMoves = v.moves; extra = v.moves - level.blocks.length; }
      }
      // estimate forced moves when not measured (or capped): how far past the
      // exit-only-unsolvable threshold we scrambled.
      const estimate = extra != null ? extra : (forcedAt >= 0 ? depth - forcedAt + 1 : 0);
      const err = Math.abs(estimate - cfg.targetExtra);
      if (err < bestErr) { bestErr = err; best = { level, density, extra, minMoves, capped, estimate }; }
      return estimate;
    };

    // How far past the "needs a shuffle" threshold to scramble in fast mode.
    // Forced moves plateau (the optimal solver finds shortcuts), so for hard we
    // scramble generously toward that ceiling; medium just needs a shuffle or so.
    const fastExtra = want === "hard" ? cfg.maxBlocks * 2 : Math.max(1, cfg.targetExtra);

    for (let a = 0; a < maxAttempts && bestErr > 0; a++) {
      const base = buildBase(cfg, rng);
      if (!base) continue;
      const density = base.filled / playable;

      if (want === "easy") {
        const level = assemble(base.placed, base.gates, cfg, cfg.name);
        if (!measure || exitOrderSuccess(level, TRIALS, rng) >= 0.8) consider(level, density, 0, -1);
        continue;
      }

      const g = new Game(assemble(base.placed, base.gates, cfg));
      let forcedAt = -1;

      if (!measure) {
        // FAST path: scramble to "needs shuffle" + a generous margin, trust
        // construction for solvability, skip the (expensive) optimal solver.
        for (let s = 1; s <= cfg.slideBudget; s++) {
          if (!reverseSlide(g, rng)) break;
          if (forcedAt < 0) {
            if (!Solver.solveExitOnly(assemble(g.blocks, base.gates, cfg)).solvable) forcedAt = s;
            continue;
          }
          if (s - forcedAt >= fastExtra) break;
        }
        if (forcedAt < 0) continue; // couldn't force a shuffle; try another base
        best = { level: assemble(g.blocks, base.gates, cfg, cfg.name), density,
                 extra: null, minMoves: null, capped: false, estimate: fastExtra };
        break;
      }

      // MEASURED path: scramble until exits alone can't solve it, then keep
      // scrambling and verify forced moves with A* up to the target.
      for (let s = 1; s <= cfg.slideBudget; s++) {
        if (!reverseSlide(g, rng)) break;
        if (forcedAt < 0) {
          if (!Solver.solveExitOnly(assemble(g.blocks, base.gates, cfg)).solvable) forcedAt = s;
          continue;
        }
        const slack = want === "hard" ? 2 : 0;
        if (s - forcedAt >= cfg.targetExtra - 1 + slack) {
          const level = assemble(g.blocks, base.gates, cfg, cfg.name);
          if (consider(level, density, s, forcedAt) >= cfg.targetExtra) break;
        }
      }
    }

    if (!best) return null;

    const level = best.level;
    level._meta = {
      seed,
      density: +best.density.toFixed(2),
      solvable: true, // guaranteed by construction
      minMoves: best.minMoves,
      extraMoves: best.extra,
      difficulty: best.extra != null ? labelByExtra(best.extra) : want,
      estimated: best.extra == null,
    };
    if (best.capped) level._meta.note = "solver-capped";
    return level;
  }

  const api = { generate, interlockOf, exitOrderSuccess, SHAPES };
  root.Generator = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
