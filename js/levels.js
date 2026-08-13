/*
 * Level format
 * ------------
 * cols, rows        : board dimensions
 * holes             : [[r,c], ...] cells that are never playable (board shape)
 * blocks            : [{ color, cells:[[r,c],...], unlockAt? }]
 *                       unlockAt = block is frozen until this many cleared
 * exits             : [{ edge:'top'|'bottom'|'left'|'right',
 *                        start, length, color, unlockAt? }]
 *                       start/length index columns (top/bottom) or rows (left/right)
 *                       unlockAt = gate is locked until this many cleared
 * lockedRegions     : [{ cells:[[r,c],...], unlockAt }] blocked until cleared
 *
 * A block clears when it slides off the board through a matching-color exit
 * whose opening fully covers the block's crossing lane.
 */

const LEVELS = [
  {
    name: "Warm Up — slide each block to its matching gate.",
    cols: 6,
    rows: 6,
    blocks: [
      { color: "red",    cells: [[2, 1], [2, 2]] },
      { color: "blue",   cells: [[1, 3], [1, 4], [2, 3], [2, 4]] },
      { color: "green",  cells: [[4, 3], [4, 4]] },
      { color: "yellow", cells: [[3, 2], [4, 2]] },
    ],
    exits: [
      { edge: "top",    start: 1, length: 2, color: "red" },
      { edge: "right",  start: 1, length: 2, color: "blue" },
      { edge: "bottom", start: 3, length: 2, color: "green" },
      { edge: "left",   start: 3, length: 2, color: "yellow" },
    ],
  },

  {
    name: "Traffic — clear a lane before the next block can move.",
    cols: 6,
    rows: 6,
    blocks: [
      { color: "orange", cells: [[0, 0], [1, 0], [2, 0]] },
      { color: "teal",   cells: [[0, 1], [0, 2]] },
      { color: "purple", cells: [[3, 2], [3, 3], [4, 2], [4, 3]] },
      { color: "pink",   cells: [[5, 4], [5, 5]] },
      { color: "green",  cells: [[2, 4], [3, 4]] },
    ],
    exits: [
      { edge: "left",   start: 0, length: 3, color: "orange" },
      { edge: "top",    start: 1, length: 2, color: "teal" },
      { edge: "bottom", start: 2, length: 2, color: "purple" },
      { edge: "right",  start: 4, length: 2, color: "pink" },
      { edge: "right",  start: 2, length: 2, color: "green" },
    ],
  },

  {
    name: "Frozen — clear blocks to thaw ice and unlock locked gates.",
    cols: 6,
    rows: 7,
    blocks: [
      { color: "red",    cells: [[3, 1], [3, 2]] },
      { color: "yellow", cells: [[1, 3], [2, 3]] },
      { color: "green",  cells: [[5, 2], [5, 3]] },
      { color: "blue",   cells: [[4, 4], [4, 5], [5, 4], [5, 5]], unlockAt: 2 },
      { color: "purple", cells: [[0, 0], [0, 1]], unlockAt: 1 },
    ],
    exits: [
      { edge: "top",    start: 1, length: 2, color: "red" },
      { edge: "left",   start: 1, length: 2, color: "yellow" },
      { edge: "bottom", start: 2, length: 2, color: "green", unlockAt: 1 },
      { edge: "right",  start: 4, length: 2, color: "blue" },
      { edge: "top",    start: 0, length: 2, color: "purple" },
    ],
    lockedRegions: [
      { cells: [[2, 5], [3, 5]], unlockAt: 3 },
    ],
  },

  {
    name: "Shapes — T, plus, L and Z pieces all slide and clear.",
    cols: 7,
    rows: 7,
    blocks: [
      { color: "purple", cells: [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]] }, // plus
      { color: "red",    cells: [[4, 0], [4, 1], [4, 2], [5, 1]] },          // T
      { color: "teal",   cells: [[1, 5], [2, 5], [3, 5], [3, 6]] },          // L
      { color: "orange", cells: [[5, 4], [5, 5], [6, 5], [6, 6]] },          // Z
    ],
    exits: [
      { edge: "top",    start: 0, length: 3, color: "purple" },
      { edge: "bottom", start: 0, length: 3, color: "red" },
      { edge: "right",  start: 1, length: 3, color: "teal" },
      { edge: "bottom", start: 4, length: 3, color: "orange" },
    ],
  },
];
