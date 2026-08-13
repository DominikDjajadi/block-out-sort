/*
 * Conformance runner (JavaScript side).
 *
 * Reads the language-neutral fixtures in fixtures/conformance/*.json and
 * evaluates them with the real Game engine, checking that the JS engine agrees
 * with the expected legal actions, canonical keys, cleared counts, terminal
 * flags, and (il)legal action applications.
 *
 *   node tools/run_conformance.js            run every fixture
 *   node tools/run_conformance.js basic_moves run one fixture (by file stem)
 *
 * Exits non-zero if any check fails.
 */

const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const src =
  fs.readFileSync(path.join(root, "js/game.js"), "utf8") + "\n" +
  "global.Game = Game; global.DIRS = DIRS;";
eval(src);

const DIR_NAMES = ["up", "down", "left", "right"];

/* ---- helpers shared with the Python conformance module ---- */

function sortedCells(block) {
  return block.cells
    .slice()
    .sort((a, z) => a.r - z.r || a.c - z.c)
    .map((c) => [c.r, c.c]);
}

function canonicalKey(blocks) {
  const parts = blocks.map((b) => {
    const cells = sortedCells(b).map(([r, c]) => r + "," + c).join(";");
    return b.color + "#" + (b.unlockAt || 0) + "#" + cells;
  });
  parts.sort();
  return parts.join("|");
}

function makeBlocks(specs) {
  return specs.map((b, i) => ({
    id: i + 1,
    color: b.color,
    unlockAt: b.unlockAt || 0,
    cells: b.cells.map(([r, c]) => ({ r, c })),
  }));
}

function cloneBlocks(blocks) {
  return blocks.map((b) => ({
    id: b.id,
    color: b.color,
    unlockAt: b.unlockAt,
    cells: b.cells.map((c) => ({ r: c.r, c: c.c })),
  }));
}

function setState(g, blocks, total) {
  g.blocks = blocks;
  g.cleared = total - blocks.length;
}

function actionKey(norm) {
  const cells = norm.cells
    .slice()
    .sort((a, z) => a[0] - z[0] || a[1] - z[1])
    .map((c) => c[0] + "," + c[1])
    .join(";");
  return [norm.color, cells, norm.dir, norm.distance, !!norm.exit].join("|");
}

function normalizedActions(g, blocks, total) {
  setState(g, blocks, total);
  const out = [];
  for (const b of blocks) {
    if (g.isBlockFrozen(b)) continue;
    for (const dir of DIR_NAMES) {
      const s = g.computeSlide(b, dir);
      for (let step = 1; step <= s.steps; step++) {
        out.push({ color: b.color, cells: sortedCells(b), dir, distance: step, exit: false });
      }
      if (s.reason === "edge" && s.canExit) {
        out.push({ color: b.color, cells: sortedCells(b), dir, distance: s.steps, exit: true });
      }
    }
  }
  return out;
}

class IllegalAction extends Error {}

function findBlock(blocks, norm) {
  const want = norm.cells
    .slice()
    .sort((a, z) => a[0] - z[0] || a[1] - z[1])
    .map((c) => c[0] + "," + c[1])
    .join(";");
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    if (b.color !== norm.color) continue;
    if (sortedCells(b).map(([r, c]) => r + "," + c).join(";") === want) return i;
  }
  return -1;
}

// Apply one normalized action; returns a NEW blocks array. Throws IllegalAction.
function applyAction(g, blocks, total, norm) {
  setState(g, blocks, total);
  const idx = findBlock(blocks, norm);
  if (idx < 0) throw new IllegalAction("no such block");
  const block = blocks[idx];
  if (g.isBlockFrozen(block)) throw new IllegalAction("frozen block");
  const s = g.computeSlide(block, norm.dir);

  if (norm.exit) {
    const legal = s.reason === "edge" && s.canExit && norm.distance === s.steps;
    if (!legal) throw new IllegalAction("illegal exit");
    return cloneBlocks(blocks.filter((_, i) => i !== idx));
  }
  if (!(norm.distance >= 1 && norm.distance <= s.steps)) {
    throw new IllegalAction("illegal slide distance");
  }
  const next = cloneBlocks(blocks);
  const d = DIRS[norm.dir];
  for (const c of next[idx].cells) { c.r += d.dr * norm.distance; c.c += d.dc * norm.distance; }
  return next;
}

/* ---- checking ---- */

let failures = 0;
const fails = [];

function check(cond, label) {
  if (!cond) { failures++; fails.push(label); }
}

function actionsEqual(a, b) {
  const sa = new Set(a.map(actionKey));
  const sb = new Set(b.map(actionKey));
  if (sa.size !== sb.size) return false;
  for (const k of sa) if (!sb.has(k)) return false;
  return true;
}

function runTest(fixtureName, level, test) {
  const g = new Game(level);
  const startSpecs = test.setup && test.setup.blocks ? test.setup.blocks : level.blocks;
  const total = startSpecs.length;
  let blocks = makeBlocks(startSpecs);

  const where = `${fixtureName} :: ${test.name}`;

  if (test.expect) {
    const e = test.expect;
    if (e.canonicalKey !== undefined) {
      check(canonicalKey(blocks) === e.canonicalKey,
        `${where}: canonicalKey (got "${canonicalKey(blocks)}", want "${e.canonicalKey}")`);
    }
    if (e.cleared !== undefined) {
      check((total - blocks.length) === e.cleared, `${where}: cleared`);
    }
    if (e.terminal !== undefined) {
      check((blocks.length === 0) === e.terminal, `${where}: terminal`);
    }
    if (e.legalActions !== undefined) {
      const got = normalizedActions(g, blocks, total);
      check(actionsEqual(got, e.legalActions),
        `${where}: legalActions mismatch\n    got:  ${got.map(actionKey).sort().join("\n          ")}\n    want: ${e.legalActions.map(actionKey).sort().join("\n          ")}`);
    }
  }

  if (test.apply) {
    let threw = false;
    try {
      for (const norm of test.apply) {
        blocks = applyAction(g, blocks, total, norm);
      }
    } catch (err) {
      if (err instanceof IllegalAction) threw = true;
      else throw err;
    }
    if (test.expectError) {
      check(threw, `${where}: expected an illegal action but all applied`);
    } else {
      check(!threw, `${where}: unexpected illegal action`);
      if (test.after) {
        const a = test.after;
        if (a.canonicalKey !== undefined) {
          check(canonicalKey(blocks) === a.canonicalKey,
            `${where}: after.canonicalKey (got "${canonicalKey(blocks)}", want "${a.canonicalKey}")`);
        }
        if (a.cleared !== undefined) check((total - blocks.length) === a.cleared, `${where}: after.cleared`);
        if (a.terminal !== undefined) check((blocks.length === 0) === a.terminal, `${where}: after.terminal`);
        if (a.legalActions !== undefined) {
          const got = normalizedActions(g, blocks, total);
          check(actionsEqual(got, a.legalActions), `${where}: after.legalActions mismatch`);
        }
      }
    }
  }
}

function main() {
  const dir = path.join(root, "fixtures", "conformance");
  const only = process.argv[2];
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json"))
    .filter((f) => !only || f === only || f === only + ".json");

  let testCount = 0;
  for (const file of files) {
    const fixture = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
    for (const test of fixture.tests) {
      runTest(fixture.name, fixture.level, test);
      testCount++;
    }
  }

  if (failures === 0) {
    console.log(`OK - ${testCount} conformance tests passed in JavaScript.`);
    process.exit(0);
  } else {
    console.log(`FAIL - ${failures} check(s) failed:`);
    for (const f of fails) console.log("  - " + f);
    process.exit(1);
  }
}

main();
