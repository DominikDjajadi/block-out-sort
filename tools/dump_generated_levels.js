/*
 * Emit a JSON array of procedurally generated levels (fixed seeds) so the
 * Python/JS parity tests have reproducible small generated levels to solve.
 *
 *   node tools/dump_generated_levels.js [count] [--out fixtures/generated_levels.json]
 *
 * Uses the fast generation path (measure:false) and strips the _meta field.
 */

const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const src =
  fs.readFileSync(path.join(root, "js/game.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(root, "js/solver.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(root, "js/generator.js"), "utf8") + "\n" +
  "global.Game = Game; global.Solver = Solver; global.Generator = Generator;";
eval(src);

const count = process.argv[2] && !process.argv[2].startsWith("--")
  ? parseInt(process.argv[2], 10) : 6;
const outIdx = process.argv.indexOf("--out");
const outPath = outIdx >= 0 ? process.argv[outIdx + 1]
  : path.join(root, "fixtures", "generated_levels.json");

const levels = [];
for (let i = 0; i < count; i++) {
  // Small boards keep exact A* cheap; vary difficulty deterministically.
  const difficulty = i % 3 === 0 ? "easy" : i % 3 === 1 ? "medium" : "hard";
  const lvl = Generator.generate({ cols: 5, rows: 5, difficulty, seed: 1000 + i, measure: false });
  if (!lvl) continue;
  delete lvl._meta;
  lvl.name = `Generated ${difficulty} #${i} (seed ${1000 + i})`;
  levels.push(lvl);
}

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(levels, null, 2) + "\n");
console.log(`wrote ${path.relative(root, outPath)} (${levels.length} levels)`);
