/* Block Out Sort - rendering, input, animation loop, UI */

(function () {
  "use strict";

  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");

  // Drag tuning (in cells).
  const EXIT_EXTRA = 0.9;    // how far past the wall the player may drag a clearable block
  const EXIT_TRIGGER = 0.45; // push past the wall beyond this to actually clear
  const AXIS_LOCK_PX = 6;

  let levelIndex = 0;
  let game = null;

  // Layout (CSS pixels).
  let cellSize = 64;
  let margin = 36;          // room around the board for gates
  let origin = { x: margin, y: margin };
  let dpr = window.devicePixelRatio || 1;

  // Animations.
  let exitAnims = [];       // pieces flying off the board

  // Active drag.
  let drag = null;

  /* ----------------------------------------------------------------- */
  /* Setup & layout                                                    */
  /* ----------------------------------------------------------------- */

  let currentLabel = "1";
  let currentLevelObj = null;
  let isGenerated = false;

  function startLevel(idx) {
    levelIndex = (idx + LEVELS.length) % LEVELS.length;
    isGenerated = false;
    loadLevelObject(LEVELS[levelIndex], String(levelIndex + 1));
  }

  function loadLevelObject(levelObj, label) {
    game = new Game(levelObj);
    currentLevelObj = levelObj;
    currentLabel = label;
    for (const b of game.blocks) b.vis = { dx: 0, dy: 0, shake: 0 };
    exitAnims = [];
    drag = null;
    document.getElementById("win-overlay").classList.add("hidden");
    document.getElementById("ui-name").textContent = game.level.name;
    layout();
    updateHud();
  }

  function randomLevel() {
    const tiers = ["medium", "hard", "hard"]; // lean harder
    const difficulty = tiers[Math.floor(Math.random() * tiers.length)];
    const cols = 6 + Math.floor(Math.random() * 2); // 6 or 7
    // measure:false uses the fast construction path (no solver during generation)
    const lvl = Generator.generate({ cols, rows: cols, difficulty, measure: false });
    if (!lvl) return;
    const m = lvl._meta || {};
    const extra = m.extraMoves != null ? m.extraMoves : "~" + (difficulty === "hard" ? 4 : 2);
    lvl.name = "Random " + (m.difficulty || difficulty) + " — " + lvl.blocks.length +
      " blocks, " + extra + " forced shuffle moves.";
    isGenerated = true;
    loadLevelObject(lvl, "\u2605");
  }

  function layout() {
    const wrap = canvas.parentElement;
    const availW = Math.min(560, window.innerWidth - 24);
    const availH = window.innerHeight - 250;

    // Reserve ~0.5 cell of margin on each side for gates.
    const sizeW = availW / (game.cols + 1.0);
    const sizeH = availH / (game.rows + 1.0);
    cellSize = Math.max(30, Math.min(82, Math.floor(Math.min(sizeW, sizeH))));
    margin = Math.round(cellSize * 0.5);
    origin = { x: margin, y: margin };

    const w = game.cols * cellSize + margin * 2;
    const h = game.rows * cellSize + margin * 2;

    dpr = window.devicePixelRatio || 1;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    wrap.style.width = w + "px";
    wrap.style.height = h + "px";
  }

  function cellToPx(r, c) {
    return { x: origin.x + c * cellSize, y: origin.y + r * cellSize };
  }

  function pxToCell(x, y) {
    return {
      r: Math.floor((y - origin.y) / cellSize),
      c: Math.floor((x - origin.x) / cellSize),
    };
  }

  /* ----------------------------------------------------------------- */
  /* Drawing helpers                                                   */
  /* ----------------------------------------------------------------- */

  function roundRect(x, y, w, h, r) {
    if (typeof r === "number") r = { tl: r, tr: r, br: r, bl: r };
    ctx.beginPath();
    ctx.moveTo(x + r.tl, y);
    ctx.lineTo(x + w - r.tr, y);
    ctx.arcTo(x + w, y, x + w, y + r.tr, r.tr);
    ctx.lineTo(x + w, y + h - r.br);
    ctx.arcTo(x + w, y + h, x + w - r.br, y + h, r.br);
    ctx.lineTo(x + r.bl, y + h);
    ctx.arcTo(x, y + h, x, y + h - r.bl, r.bl);
    ctx.lineTo(x, y + r.tl);
    ctx.arcTo(x, y, x + r.tl, y, r.tl);
    ctx.closePath();
  }

  function cellSet(cells) {
    const s = new Set();
    for (const c of cells) s.add(c.r + "," + c.c);
    return s;
  }

  // Draw a connected piece (blocks and exit animations share this).
  function drawPiece(cells, colorName, off, alpha) {
    const col = PALETTE[colorName];
    const set = cellSet(cells);
    const R = cellSize * 0.28;
    const inset = Math.max(1, cellSize * 0.04);

    ctx.save();
    ctx.globalAlpha = alpha;

    for (const cell of cells) {
      const p = cellToPx(cell.r, cell.c);
      const x = p.x + inset + off.x;
      const y = p.y + inset + off.y;
      const s = cellSize - inset * 2;

      const up = set.has((cell.r - 1) + "," + cell.c);
      const dn = set.has((cell.r + 1) + "," + cell.c);
      const lf = set.has(cell.r + "," + (cell.c - 1));
      const rt = set.has(cell.r + "," + (cell.c + 1));
      const rad = {
        tl: !up && !lf ? R : 3,
        tr: !up && !rt ? R : 3,
        br: !dn && !rt ? R : 3,
        bl: !dn && !lf ? R : 3,
      };

      const g = ctx.createLinearGradient(x, y, x, y + s);
      g.addColorStop(0, col.light);
      g.addColorStop(0.45, col.base);
      g.addColorStop(1, col.dark);
      ctx.fillStyle = g;
      roundRect(x, y, s, s, rad);
      ctx.fill();

      // glossy top highlight
      ctx.globalAlpha = alpha * 0.35;
      ctx.fillStyle = "#ffffff";
      roundRect(x + s * 0.16, y + s * 0.12, s * 0.68, s * 0.26, s * 0.13);
      ctx.fill();
      ctx.globalAlpha = alpha;
    }
    ctx.restore();
  }

  function drawSlot(r, c) {
    const p = cellToPx(r, c);
    const inset = cellSize * 0.07;
    ctx.fillStyle = "rgba(0,0,0,0.22)";
    roundRect(p.x + inset, p.y + inset, cellSize - inset * 2, cellSize - inset * 2, cellSize * 0.22);
    ctx.fill();
  }

  function drawIceCell(r, c, label) {
    const p = cellToPx(r, c);
    const inset = cellSize * 0.06;
    ctx.fillStyle = "rgba(160,210,245,0.30)";
    roundRect(p.x + inset, p.y + inset, cellSize - inset * 2, cellSize - inset * 2, cellSize * 0.2);
    ctx.fill();
    ctx.strokeStyle = "rgba(220,240,255,0.55)";
    ctx.lineWidth = 2;
    ctx.stroke();
    drawLock(p.x + cellSize / 2, p.y + cellSize / 2, cellSize * 0.18);
    if (label) {
      ctx.fillStyle = "#eaf6ff";
      ctx.font = "700 " + Math.round(cellSize * 0.24) + "px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, p.x + cellSize / 2, p.y + cellSize * 0.74);
    }
  }

  function drawLock(cx, cy, s) {
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    roundRect(cx - s * 0.7, cy - s * 0.15, s * 1.4, s * 1.1, s * 0.25);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = Math.max(2, s * 0.28);
    ctx.beginPath();
    ctx.arc(cx, cy - s * 0.15, s * 0.55, Math.PI, 0);
    ctx.stroke();
    ctx.restore();
  }

  function drawGate(exit) {
    const active = game.isExitActive(exit);
    const col = PALETTE[exit.color];
    const depth = cellSize * 0.42;
    let x, y, w, h, arrow;

    if (exit.edge === "top") {
      x = origin.x + exit.start * cellSize; w = exit.length * cellSize;
      y = origin.y - depth; h = depth; arrow = "up";
    } else if (exit.edge === "bottom") {
      x = origin.x + exit.start * cellSize; w = exit.length * cellSize;
      y = origin.y + game.rows * cellSize; h = depth; arrow = "down";
    } else if (exit.edge === "left") {
      y = origin.y + exit.start * cellSize; h = exit.length * cellSize;
      x = origin.x - depth; w = depth; arrow = "left";
    } else {
      y = origin.y + exit.start * cellSize; h = exit.length * cellSize;
      x = origin.x + game.cols * cellSize; w = depth; arrow = "right";
    }

    const pad = cellSize * 0.08;
    ctx.save();
    if (active) {
      const g = ctx.createLinearGradient(x, y, x, y + h);
      g.addColorStop(0, col.light);
      g.addColorStop(1, col.dark);
      ctx.fillStyle = g;
    } else {
      ctx.fillStyle = "rgba(120,130,160,0.5)";
    }
    roundRect(x + pad, y + pad, w - pad * 2, h - pad * 2, depth * 0.35);
    ctx.fill();

    const cx = x + w / 2, cy = y + h / 2;
    if (active) {
      drawArrow(cx, cy, arrow, depth * 0.34);
    } else {
      drawLock(cx, cy - depth * 0.05, depth * 0.2);
      const remaining = exit.unlockAt - game.cleared;
      ctx.fillStyle = "#eef1ff";
      ctx.font = "700 " + Math.round(depth * 0.36) + "px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(remaining > 0 ? remaining : "", cx, cy + depth * 0.32);
    }
    ctx.restore();
  }

  function drawArrow(cx, cy, dir, s) {
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.beginPath();
    if (dir === "up") {
      ctx.moveTo(cx, cy - s); ctx.lineTo(cx + s, cy + s * 0.5); ctx.lineTo(cx - s, cy + s * 0.5);
    } else if (dir === "down") {
      ctx.moveTo(cx, cy + s); ctx.lineTo(cx + s, cy - s * 0.5); ctx.lineTo(cx - s, cy - s * 0.5);
    } else if (dir === "left") {
      ctx.moveTo(cx - s, cy); ctx.lineTo(cx + s * 0.5, cy + s); ctx.lineTo(cx + s * 0.5, cy - s);
    } else {
      ctx.moveTo(cx + s, cy); ctx.lineTo(cx - s * 0.5, cy + s); ctx.lineTo(cx - s * 0.5, cy - s);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  /* ----------------------------------------------------------------- */
  /* Frame                                                             */
  /* ----------------------------------------------------------------- */

  let lastT = performance.now();

  function frame(now) {
    const dt = Math.min(50, now - lastT);
    lastT = now;
    update(dt);
    render();
    requestAnimationFrame(frame);
  }

  function update(dt) {
    // ease block visual offsets back to rest (except the one being dragged)
    for (const b of game.blocks) {
      if (drag && drag.block === b) continue;
      const k = Math.pow(0.0025, dt / 1000); // smooth decay
      b.vis.dx *= k;
      b.vis.dy *= k;
      if (Math.abs(b.vis.dx) < 0.3) b.vis.dx = 0;
      if (Math.abs(b.vis.dy) < 0.3) b.vis.dy = 0;
      if (b.vis.shake > 0) b.vis.shake = Math.max(0, b.vis.shake - dt);
    }
    // advance exit animations
    for (const a of exitAnims) a.t += dt;
    exitAnims = exitAnims.filter((a) => a.t < a.dur);
  }

  function render() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // board panel
    const bw = game.cols * cellSize, bh = game.rows * cellSize;
    ctx.fillStyle = "rgba(255,255,255,0.05)";
    roundRect(origin.x - cellSize * 0.16, origin.y - cellSize * 0.16,
      bw + cellSize * 0.32, bh + cellSize * 0.32, cellSize * 0.3);
    ctx.fill();

    // slots / holes / locked regions
    for (let r = 0; r < game.rows; r++) {
      for (let c = 0; c < game.cols; c++) {
        if (game.holes.has(r + "," + c)) continue;
        if (game.isRegionLocked(r, c)) {
          const reg = lockedRegionAt(r, c);
          drawIceCell(r, c, reg ? String(reg.unlockAt - game.cleared) : "");
        } else {
          drawSlot(r, c);
        }
      }
    }

    // gates
    for (const e of game.exits) drawGate(e);

    // blocks
    for (const b of game.blocks) {
      const off = {
        x: b.vis.dx + shakeOffset(b),
        y: b.vis.dy,
      };
      drawPiece(b.cells, b.color, off, 1);
      if (game.isBlockFrozen(b)) drawFrozenOverlay(b);
    }

    // exit animations
    for (const a of exitAnims) {
      const p = a.t / a.dur;
      const eased = p * p;
      const dist = eased * a.leave * cellSize;
      const off = { x: a.dir.dc * dist, y: a.dir.dr * dist };
      drawPiece(a.cells, a.color, off, 1 - p * p);
    }
  }

  function shakeOffset(b) {
    if (b.vis.shake <= 0) return 0;
    return Math.sin(b.vis.shake / 22) * (b.vis.shake / 120) * 7;
  }

  function lockedRegionAt(r, c) {
    for (const reg of game.lockedRegions) {
      if (game.cleared >= reg.unlockAt) continue;
      for (const cell of reg.cells) if (cell.r === r && cell.c === c) return reg;
    }
    return null;
  }

  function drawFrozenOverlay(b) {
    const set = cellSet(b.cells);
    const inset = Math.max(1, cellSize * 0.04);
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = "#d6f0ff";
    for (const cell of b.cells) {
      const p = cellToPx(cell.r, cell.c);
      roundRect(p.x + inset, p.y + inset, cellSize - inset * 2, cellSize - inset * 2, cellSize * 0.24);
      ctx.fill();
    }
    ctx.restore();
    // remaining count at piece center
    let sr = 0, sc = 0;
    for (const cell of b.cells) { sr += cell.r; sc += cell.c; }
    const cr = sr / b.cells.length, cc = sc / b.cells.length;
    const p = cellToPx(cr, cc);
    drawLock(p.x + cellSize / 2, p.y + cellSize * 0.42, cellSize * 0.16);
    ctx.fillStyle = "#10324a";
    ctx.font = "800 " + Math.round(cellSize * 0.3) + "px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(b.unlockAt - game.cleared), p.x + cellSize / 2, p.y + cellSize * 0.72);
  }

  /* ----------------------------------------------------------------- */
  /* Input                                                             */
  /* ----------------------------------------------------------------- */

  function localPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function blockAt(r, c) {
    for (const b of game.blocks) {
      for (const cell of b.cells) if (cell.r === r && cell.c === c) return b;
    }
    return null;
  }

  function onDown(e) {
    if (!game || game.isWon()) return;
    const pt = localPoint(e);
    const { r, c } = pxToCell(pt.x, pt.y);
    const b = blockAt(r, c);
    if (!b) return;
    if (game.isBlockFrozen(b)) {
      b.vis.shake = 240;
      return;
    }
    drag = {
      block: b,
      startX: pt.x,
      startY: pt.y,
      axis: null,
      offset: 0,
      pos: null,
      neg: null,
    };
    canvas.setPointerCapture(e.pointerId);
  }

  function onMove(e) {
    if (!drag) return;
    const pt = localPoint(e);
    const dx = pt.x - drag.startX;
    const dy = pt.y - drag.startY;

    if (!drag.axis) {
      if (Math.max(Math.abs(dx), Math.abs(dy)) < AXIS_LOCK_PX) return;
      drag.axis = Math.abs(dx) >= Math.abs(dy) ? "h" : "v";
      const posName = drag.axis === "h" ? "right" : "down";
      const negName = drag.axis === "h" ? "left" : "up";
      drag.pos = game.computeSlide(drag.block, posName);
      drag.neg = game.computeSlide(drag.block, negName);
    }

    const raw = (drag.axis === "h" ? dx : dy) / cellSize;
    const upper = drag.pos.steps + (drag.pos.canExit ? EXIT_EXTRA : 0);
    const lower = -(drag.neg.steps + (drag.neg.canExit ? EXIT_EXTRA : 0));
    drag.offset = Math.max(lower, Math.min(upper, raw));

    if (drag.axis === "h") { drag.block.vis.dx = drag.offset * cellSize; drag.block.vis.dy = 0; }
    else { drag.block.vis.dy = drag.offset * cellSize; drag.block.vis.dx = 0; }
  }

  function onUp(e) {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (!d.axis) return;

    const block = d.block;
    const posName = d.axis === "h" ? "right" : "down";
    const negName = d.axis === "h" ? "left" : "up";
    const offset = d.offset;

    let exit = null;
    let committed = 0;
    if (offset > 0 && d.pos.canExit && offset >= d.pos.steps + EXIT_TRIGGER) {
      exit = { dir: posName, steps: d.pos.steps };
    } else if (offset < 0 && d.neg.canExit && -offset >= d.neg.steps + EXIT_TRIGGER) {
      exit = { dir: negName, steps: d.neg.steps };
    } else {
      committed = clamp(Math.round(offset), -d.neg.steps, d.pos.steps);
    }

    if (exit) {
      game.translateBlock(block, exit.dir, exit.steps);
      spawnExitAnim(block, exit.dir);
      game.removeBlock(block);
      game.moves++;
      block.vis.dx = 0; block.vis.dy = 0;
      afterChange();
    } else {
      game.translateBlock(block, posName, committed);
      if (committed !== 0) game.moves++;
      const restPx = (offset - committed) * cellSize;
      if (d.axis === "h") { block.vis.dx = restPx; block.vis.dy = 0; }
      else { block.vis.dy = restPx; block.vis.dx = 0; }
      afterChange();
    }
  }

  function spawnExitAnim(block, dirName) {
    const dir = DIRS[dirName];
    const cells = block.cells.map((c) => ({ r: c.r, c: c.c }));
    let leave;
    if (dirName === "up") leave = Math.max(...cells.map((c) => c.r)) + 1;
    else if (dirName === "down") leave = game.rows - Math.min(...cells.map((c) => c.r));
    else if (dirName === "left") leave = Math.max(...cells.map((c) => c.c)) + 1;
    else leave = game.cols - Math.min(...cells.map((c) => c.c));
    exitAnims.push({ cells, color: block.color, dir, leave, t: 0, dur: 280 });
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /* ----------------------------------------------------------------- */
  /* UI / state                                                        */
  /* ----------------------------------------------------------------- */

  function afterChange() {
    updateHud();
    if (game.isWon()) {
      setTimeout(showWin, 320);
    }
  }

  function updateHud() {
    document.getElementById("ui-level").textContent = currentLabel;
    document.getElementById("ui-moves").textContent = game.moves;
    document.getElementById("ui-cleared").textContent =
      game.cleared + "/" + game.totalBlocks;
  }

  function showWin() {
    document.getElementById("win-stats").textContent =
      "Cleared in " + game.moves + " move" + (game.moves === 1 ? "" : "s") + ".";
    document.getElementById("win-overlay").classList.remove("hidden");
  }

  /* ----------------------------------------------------------------- */
  /* Wiring                                                            */
  /* ----------------------------------------------------------------- */

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);

  const replay = () => (isGenerated ? loadLevelObject(currentLevelObj, currentLabel) : startLevel(levelIndex));
  document.getElementById("btn-restart").addEventListener("click", replay);
  document.getElementById("btn-prev").addEventListener("click", () => startLevel(levelIndex - 1));
  document.getElementById("btn-next").addEventListener("click", () => startLevel(levelIndex + 1));
  document.getElementById("btn-random").addEventListener("click", randomLevel);
  document.getElementById("btn-replay").addEventListener("click", replay);
  document.getElementById("btn-continue").addEventListener("click", () =>
    isGenerated ? randomLevel() : startLevel(levelIndex + 1));

  window.addEventListener("resize", () => { if (game) layout(); });

  startLevel(0);
  requestAnimationFrame(frame);
})();
