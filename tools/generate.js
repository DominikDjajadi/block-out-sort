/*
 * Level generator CLI.
 *
 *   node tools/generate.js
 *   node tools/generate.js --cols 7 --rows 7 --blocks 6 --difficulty hard --seed 42
 *   node tools/generate.js --count 5            generate & validate a batch (smoke test)
 *
 * Prints a paste-ready level object plus a validation report.
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

function parseArgs() {
  const a = process.argv.slice(2);
  const o = {};
  for (let i = 0; i < a.length; i++) {
    if (a[i].startsWith("--")) {
      const key = a[i].slice(2);
      const val = a[i + 1];
      if (val === undefined || val.startsWith("--")) { o[key] = true; }
      else { o[key] = /^\d+$/.test(val) ? parseInt(val, 10) : val; i++; }
    }
  }
  return o;
}

function fmtLevel(lvl) {
  const blocks = lvl.blocks
    .map((b) => `    { color: "${b.color}", cells: ${JSON.stringify(b.cells)} },`)
    .join("\n");
  const exits = lvl.exits
    .map((e) => `    { edge: "${e.edge}", start: ${e.start}, length: ${e.length}, color: "${e.color}" },`)
    .join("\n");
  return [
    "{",
    `  name: ${JSON.stringify(lvl.name)},`,
    `  cols: ${lvl.cols}, rows: ${lvl.rows},`,
    "  blocks: [",
    blocks,
    "  ],",
    "  exits: [",
    exits,
    "  ],",
    "},",
  ].join("\n");
}

const args = parseArgs();
const baseOpts = {
  cols: args.cols || 6,
  rows: args.rows || 6,
  difficulty: args.difficulty || "hard",
  seed: args.seed != null && args.seed !== true ? args.seed : undefined,
};
if (args.blocks) baseOpts.blocks = args.blocks;
if (args.density) baseOpts.density = args.density / 100; // pass as percent, e.g. --density 60

if (args.count) {
  let ok = 0;
  for (let i = 0; i < args.count; i++) {
    const lvl = Generator.generate(baseOpts);
    if (!lvl) { console.log(`#${i + 1}: FAILED to build`); continue; }
    const m = lvl._meta;
    if (m.solvable) ok++;
    console.log(
      `#${i + 1}: ${m.solvable ? "solvable" : "UNSOLVABLE"} | ${lvl.blocks.length} blocks | ` +
      `min-moves=${m.minMoves} extra=${m.extraMoves} (${m.difficulty}) | density=${m.density} | seed=${m.seed}`
    );
  }
  console.log(`\n${ok}/${args.count} solvable.`);
  process.exit(ok === args.count ? 0 : 1);
} else {
  const lvl = Generator.generate(baseOpts);
  if (!lvl) { console.error("Failed to generate a level with these constraints."); process.exit(1); }
  console.log(fmtLevel(lvl));
  console.log("\n// meta:", JSON.stringify(lvl._meta));
}
