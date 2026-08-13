/*
 * Level validator CLI.
 *
 *   node tools/validate.js          validate every level in js/levels.js
 *   node tools/validate.js 3        validate only level 3 and print its solution
 *
 * Reports solvability, minimum moves (difficulty), and search size.
 */

const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const src =
  fs.readFileSync(path.join(root, "js/levels.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(root, "js/game.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(root, "js/solver.js"), "utf8") + "\n" +
  "global.LEVELS = LEVELS; global.Solver = Solver;";
eval(src);

const only = process.argv[2] ? parseInt(process.argv[2], 10) : null;

let allOk = true;
LEVELS.forEach((lv, i) => {
  const num = i + 1;
  if (only && num !== only) return;

  const t0 = Date.now();
  const res = Solver.validateLevel(lv);
  const ms = Date.now() - t0;

  const status = res.solvable
    ? `OK  min-moves=${res.minMoves} (${res.difficulty})`
    : res.exhausted
      ? "UNKNOWN (search limit hit)"
      : "UNSOLVABLE";
  if (!res.solvable) allOk = false;

  console.log(`Level ${num}: ${status}  [${res.explored} states, ${ms}ms]`);
  console.log(`  ${lv.name}`);

  if (only && res.path) {
    console.log("  Solution:");
    console.log(Solver.describePath(res.path).split("\n").map((l) => "    " + l).join("\n"));
  }
});

process.exit(allOk ? 0 : 1);
