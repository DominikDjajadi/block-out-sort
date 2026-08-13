# Neural state & action encoding (Block Out Sort)

This document specifies the **stable** tensor encoding consumed by the
supervised policy-value network (`blocksort.training`). It is the source of
truth: encoders, the model, checkpoints, and tests all follow it. Nothing here
depends on runtime object identity (Python `id`, list position, etc.).

All coordinates are `(r, c)` with `r` top→bottom, `c` left→right, matching the
engine.

## Fixed orderings (must never be reordered)

These orderings are stored in every checkpoint so inference is reproducible.

- **Colors** (`COLOR_ORDER`, 8): `red, blue, green, yellow, purple, orange, teal, pink`.
- **Directions** (`DIRECTION_ORDER`, 4): `up=0, down=1, left=2, right=3`.

## Encoding configuration

```python
@dataclass(frozen=True)
class EncodingConfig:
    max_rows: int = 8          # padded board height
    max_cols: int = 8          # padded board width
    max_slide_distance: int = 8
    max_blocks: int = 16       # supported total and count normalization limit
    colors: tuple[str, ...] = COLOR_ORDER
```

A level whose `rows > max_rows`, `cols > max_cols`, whose total block count is
greater than `max_blocks`, or whose required slide distance is greater than
`max_slide_distance` raises `EncodingError`. **Inputs are never silently
truncated or normalized outside the checkpoint's declared range.**

Count/threshold scalars are normalized by `max_blocks` (a fixed constant in the
config, not data-dependent), so they are stable across datasets. Designer
generation is capped at the smaller `max_blocks` limit of the participating
designer and protagonist checkpoints.

## State tensor

Channel-first `board` tensor of shape `[C, max_rows, max_cols]`,
`C = 4 + num_colors + 5 + 2 + 4*(num_colors + 3) + 2 = 65` for the default
8-color palette, zero-padded outside the real `rows × cols` board. A separate
`valid_cell_mask` of shape `[max_rows, max_cols]` marks the real board extent so
padded cells can never be confused with playable cells.

### Channel list (`C = 65`, 8 colors)

| idx | name | meaning |
| --- | --- | --- |
| 0 | `board_extent` | 1 inside the real `rows × cols` board (= `valid_cell_mask`) |
| 1 | `hole` | 1 on permanent holes (non-playable structural cells) |
| 2 | `locked_region_now` | 1 where a locked region is **currently** locked (`cleared < unlock_at`) |
| 3 | `locked_region_remaining` | `(unlock_at - cleared)/max_blocks` on currently-locked region cells, else 0 |
| 4–11 | `occupancy[color]` | 1 where a block of `COLOR_ORDER[k]` occupies the cell (8 channels) |
| 12 | `anchor` | 1 at each block's canonical anchor cell (lexicographically smallest cell) |
| 13 | `connect_up` | 1 if the cell above belongs to the **same block** |
| 14 | `connect_down` | 1 if the cell below belongs to the same block |
| 15 | `connect_left` | 1 if the cell to the left belongs to the same block |
| 16 | `connect_right` | 1 if the cell to the right belongs to the same block |
| 17 | `frozen_now` | 1 on cells of blocks that are **currently** frozen (`cleared < unlock_at`) |
| 18 | `frozen_remaining` | `(unlock_at - cleared)/max_blocks` on frozen-block cells, else 0 |
| 19–62 | gate planes | per direction `d ∈ (up, down, left, right)`, a block of `num_colors + 3` planes |
| | `gate_{d}_color[color]` | 1 on a gate's adjacent border cell, by gate color (8 planes per direction) |
| | `gate_{d}_active` | 1 on border cells where a gate exiting `d` is active (`cleared >= unlock_at`) |
| | `gate_{d}_locked` | 1 on border cells where a gate exiting `d` is still locked |
| | `gate_{d}_remaining` | `(unlock_at - cleared)/max_blocks` on locked gates exiting `d`, else 0 |
| 63 | `cleared_plane` | constant plane = `cleared / max_blocks` (on the real board) |
| 64 | `remaining_plane` | constant plane = `remaining / max_blocks` (on the real board) |

Gate planes are laid out as 4 contiguous direction blocks, each
`[color_0..color_7, active, locked, remaining]`, so `gate_color(d, k) = gate0 +
d*(num_colors+3) + k`.

**Block identity.** Color occupancy alone cannot separate two touching
same-color blocks. The `anchor` channel plus the four `connect_*` channels make
block membership explicit: a single combined block has connectivity across its
internal boundary and one anchor; two touching blocks have a connectivity gap at
the boundary and two anchors. Connectivity is computed per actual block, so it
is independent of block ordering in memory.

**Gate direction.** Gate color is encoded **per outward direction**, so a corner
border cell carrying gates of different colors and directions (common in
procedurally generated levels) stays fully distinguishable — e.g. a top-red and
right-blue gate set `gate_up_color[red]` and `gate_right_color[blue]` at the same
cell. A gate maps to the adjacent in-board cell: top→row 0, bottom→row `rows-1`,
left→col 0, right→col `cols-1`, over its `start..start+length` span. *Residual
limitation:* two gates of the **same direction** overlapping a single cell with
**different unlock thresholds** (one active, one locked) share that direction's
`active`/`locked` planes, so their individual status is not separable; this is
extremely rare and does not affect any current level.

**Order invariance.** Every channel is an aggregate over cells/blocks that does
not depend on block list order, so reordering interchangeable blocks yields an
identical tensor.

### Global feature vector

`global_features`, shape `[6]`:

```
[ cleared/max_blocks,
  remaining/max_blocks,
  total_blocks/max_blocks,
  rows/max_rows,
  cols/max_cols,
  active_gate_count/max_blocks ]
```

Scalars are provided both as constant planes (channels 34–35) and in the global
vector; the value head consumes the global vector. This is deterministic and
documented.

## Fixed action space

Actions use a **spatial anchor**, not block IDs:

```
action identity = (anchor_row, anchor_col, direction, move_code)
```

- `anchor_row, anchor_col`: the block's canonical anchor (min `(r,c)` cell)
  **before** the move. Anchors uniquely locate blocks in a valid state because
  blocks cannot overlap.
- `direction`: index into `DIRECTION_ORDER`.
- `move_code`: with `D = max_slide_distance` and `M = D + 1` codes:
  - slide of distance `d ∈ [1, D]` → `move_code_index = d - 1` (slots `0..D-1`)
  - **EXIT** → `move_code_index = D` (the final slot)

EXIT and slides never collide (slides use slots `0..D-1`, EXIT uses slot `D`).
A required slide distance `> D` raises `EncodingError`.

### Shape and flat index

Spatial action shape `[max_rows, max_cols, 4, M]`; flattened action-space size
`A = max_rows * max_cols * 4 * M` (defaults: `8*8*4*9 = 2304`).

```
move_code_count M = max_slide_distance + 1
index = ((anchor_row * max_cols + anchor_col) * 4 + direction_index) * M
        + move_code_index
```

Decoding inverts this:

```
move_code_index = index % M
rest            = index // M
direction_index = rest % 4
rest            = rest // 4
anchor_col      = rest % max_cols
anchor_row      = rest // max_cols
```

Then the block with anchor `(anchor_row, anchor_col)` is located in the state;
`move_code_index == D` means EXIT (the exit distance is recomputed from the
environment's slide result), otherwise the slide distance is
`move_code_index + 1`.

The policy head emits logits shaped `[4*M, max_rows, max_cols]` where
`channel = direction_index * M + move_code_index`. Reshaping/permuting to
`[max_rows, max_cols, 4, M]` and flattening matches the index formula exactly, so
logits, masks, and targets share one ordering.

### Legal-action mask

`build_legal_action_mask(env, state, config)` returns a `[A]` float tensor with
1 at `action_index(state, a)` for every `a` in `env.legal_actions(state)`, else
0. Padded anchors (outside the real board) are always 0 because no legal action
anchors there. Illegal logits are set to `-inf` before softmax.

## Targets

- **Policy target** (sparse → dense `[A]`): the record stores a probability per
  legal action (uniform over optimal, or soft-regret). Each is placed at its
  action index; all illegal indices are 0; the vector sums to ≈1. Multiple
  optimal actions each get nonzero mass.
- **Value target** (scalar): `normalized_value = -raw_optimal_moves / constant`
  (default `constant = 20`). Terminal value is 0; more negative = farther from
  done; higher = better. The raw move count is kept for denormalized metrics.
  The normalization scheme + constant are stored in the record and in
  checkpoints.

## Unsupported / explicit limits

- Boards larger than `max_rows × max_cols` → error.
- Required slide distance `> max_slide_distance` → error.
- More than `len(COLOR_ORDER)` distinct colors / unknown color → error.
- The dataset contains **no terminal states** (the generator never labels them);
  terminal handling in losses/metrics exists only for model robustness.
