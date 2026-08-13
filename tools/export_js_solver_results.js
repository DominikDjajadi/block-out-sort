/*
 * Export JavaScript solver results as JSON, for Python/JS parity checks.
 *
 *   node tools/export_js_solver_results.js <levels.json> [maxNodes]
 *
 * Reads a JSON file containing a level object or an array of level objects and
 * writes a JSON array of per-level results to stdout (nothing else, so it is
 * machine-parseable):
 *
 *   [{ index, name, solvable, minMoves, exhausted, exitOnly, explored }, ...]
 *
 * `solvable` is true / false / null (null = search budget hit; never reported
 * as unsolvable). `minMoves` is null unless solved optimally.
 */

const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const src =
  fs.readFileSync(path.join(root, "js/game.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(root, "js/solver.js"), "utf8") + "\n" +
  "global.Game = Game; global.Solver = Solver; global.DIRS = DIRS;";
eval(src);

const levelsPath = process.argv[2];
if (!levelsPath) {
  console.error("usage: node tools/export_js_solver_results.js <levels.json> [maxNodes]");
  process.exit(2);
}
const maxNodes = process.argv[3] ? parseInt(process.argv[3], 10) : 250000;

let data = JSON.parse(fs.readFileSync(levelsPath, "utf8"));
if (!Array.isArray(data)) {
  if (data && Array.isArray(data.levels)) data = data.levels;
  else data = [data];
}

const out = data.map((level, index) => {
  const astar = Solver.solveAStar(level, { maxNodes });
  const exitOnly = Solver.solveExitOnly(level);
  const solvable = astar.exhausted ? null : !!astar.solvable;
  const minMoves = astar.solvable ? astar.moves : null;
  return {
    index,
    name: level.name || null,
    solvable,
    minMoves,
    exhausted: !!astar.exhausted,
    exitOnly: !!exitOnly.solvable,
    explored: astar.explored,
  };
});

process.stdout.write(JSON.stringify(out));
