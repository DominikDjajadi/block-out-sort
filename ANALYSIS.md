# Block Out Sort — engine analysis (pre-port)

This document records the findings that motivated the Python port and the one
JavaScript engine patch. It is the "inspect and report" deliverable.

## Architecture

- `js/game.js` is the **authoritative** rules engine. Coordinates are `(r, c)`
  with `r` top→bottom, `c` left→right. A `Game` holds the loaded level plus
  mutable runtime state (`blocks`, `cleared`, `moves`).
- `js/solver.js` reuses `Game` geometry for BFS / A\* / exit-only search. It
  derives `cleared = totalBlocks − remaining` and dedupes states with an
  order-independent `stateKey` that excludes `moves`.
- `js/generator.js` reverse-constructs solvable levels and applies
  reversibility-preserving scrambles.
- `js/main.js` is presentation only (canvas render, pointer drag, animation,
  HUD). The only "rules" it owns are drag-gesture thresholds.
- `tools/validate.js` / `tools/generate.js` `eval` the browser scripts under
  Node. There was **no structural level validator** before this work.

## Move lifecycle

`pointerdown` → `blockAt` → `pointermove` axis-locks and caches
`game.computeSlide` per direction → visible offset clamped to
`steps + (canExit ? EXIT_EXTRA : 0)` → `pointerup` commits either an **exit**
(`translateBlock(steps)` + `spawnExitAnim` + `removeBlock`, which does
`cleared++`) or a rounded slide → `updateHud` / `isWon`.

## Level schema

Grid `cols`×`rows`; `holes` permanently non-playable; `blocks`
(`{color, cells, unlockAt?}`, frozen until `cleared ≥ unlockAt`); `exits`
(`{edge, start, length, color, unlockAt?}`, `start/length` index columns for
top/bottom and rows for left/right, locked until `cleared ≥ unlockAt`);
`lockedRegions` (`{cells, unlockAt}`, walls until `cleared ≥ unlockAt`).

## Bugs / ambiguities

1. **Irregular-exit extrusion bug (real, fixed).** `computeSlide` stops at the
   first step a cell leaves the board, and `canExitThrough` only checks gate
   coverage. The extrusion itself is never collision-tested, so an irregular
   piece can be cleared even though a trailing cell would pass through another
   block / hole / locked cell while the piece is partially off the board. See
   `fixtures/conformance/irregular_exit.json` for the canonical failing case.
2. **No structural validator** — added in `python/blocksort/validation.py`.
3. **`unlockAt` reachability unchecked** — a block needing `unlockAt ≥ total`
   can never thaw; flagged by the new validator.
4. **Disconnected blocks** are accepted by the engine but unsupported by the
   renderer and never generated. The validator rejects them (documented).

## Corrected exit rule (authoritative)

A block may exit in a direction iff:

1. a matching **active** exit on that edge covers **every** lane coordinate the
   block crosses, **and**
2. the block can translate far enough that **every** cell leaves the board,
   **and**
3. throughout that translation **every still-on-board cell is collision-free**
   (no other block, hole, or locked cell).

The `game.js` patch implements (2)+(3) by continuing the extrusion past the
boundary and checking each still-on-board cell. No browser drag-math change is
required: the corrected rule only ever turns a previously-`true` `canExit` into
`false`, so the existing gesture thresholds and exit animation remain correct.
