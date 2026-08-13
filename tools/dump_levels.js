/*
 * Dump the handcrafted LEVELS (js/levels.js) to JSON so non-JS tooling (e.g.
 * the Python test-suite) can load the exact same level definitions.
 *
 *   node tools/dump_levels.js          writes fixtures/levels.json
 */

const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
eval(fs.readFileSync(path.join(root, "js/levels.js"), "utf8") + "\nglobal.LEVELS = LEVELS;");

const out = path.join(root, "fixtures", "levels.json");
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(LEVELS, null, 2) + "\n");
console.log(`wrote ${path.relative(root, out)} (${LEVELS.length} levels)`);
