# Block Out Sort

[Play the browser version](https://dominikdjajadi.github.io/block-out-sort/) · [![CI](https://github.com/DominikDjajadi/block-out-sort/actions/workflows/ci.yml/badge.svg)](https://github.com/DominikDjajadi/block-out-sort/actions/workflows/ci.yml)

![Block Out Sort gameplay](docs/assets/gameplay.png)

Block Out Sort is a puzzle game about sliding colored blocks through matching gates. I started it as a small browser game and later used it to experiment with procedural generation, exact solvers, and neural-guided search.

The browser game is playable now. The solver and training work are still in progress.

## Playing the game

Drag a block horizontally or vertically. A block is cleared when it crosses a gate of the same color; clear every block to finish the level. Later levels add frozen blocks, locked gates, and locked regions.

To run it locally, serve the repository root with any static file server. For example:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>. There is no frontend build step or runtime dependency.

## How the project works

There are matching JavaScript and Python implementations of the game rules. The JavaScript version runs the game, while the Python version is used for generation, solving, training, and evaluation. Shared fixtures test that both implementations behave the same way.

Levels are generated backwards from a completed board. This gives the generator a known solution instead of relying on random boards that may be impossible.

The Python side includes:

- A* and BFS solvers for exact solutions
- a PyTorch policy/value model
- PUCT search guided by the model
- expert-iteration and co-training experiments
- an adversarial level designer
- resumable runs and paired model evaluation

The longer explanation is in [the architecture notes](docs/architecture.md). The model input and output format is documented in [the encoding notes](docs/neural_encoding.md).

## Current solver result

The latest confirmed experiment compared the previous solver with a cumulatively trained learner on 500 newly generated levels. Across search budgets of 20, 34, 57, 95, and 160 simulations, the weighted solve rate increased from `0.6104` to `0.6260`. The paired-bootstrap 95% lower bound was `+0.0044`, and none of the individual budgets regressed in that test.

That is an encouraging result, but it is not a finished production model. The checkpoint is not included in this repository, and search difficulty is not necessarily the same as difficulty for a human player. The experiment and its limitations are described in [research results](docs/research-results.md).

## Python setup

Python 3.10 or newer is required.

```bash
cd python
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

The JavaScript/Python conformance tests run from the repository root:

```bash
node tools/run_conformance.js
```

To solve the first bundled level with BFS:

```bash
cd python
python -m blocksort.cli.solve \
  --levels ../fixtures/levels.json \
  --level-index 0 \
  --show-solution \
  --bfs
```

## Repository layout

- `js/` contains the browser game, rendering, rules, and generator.
- `python/blocksort/` contains the Python environment, solvers, model, search, and training code.
- `python/tests/` contains the Python test suite.
- `fixtures/` contains levels and cross-language conformance cases.
- `docs/` contains the architecture, model encoding, and research notes.

Large replay buffers, experiment directories, evaluation pools, disposable checkpoints, and machine-specific logs are not included. Three small versioned datasets are kept in the repository so the test suite can run on its own.

## License

Copyright © 2026 Dominik Djajadi. All rights reserved. The code is available here for viewing and technical discussion; permission to copy, modify, use, or distribute it is not granted. See [LICENSE](LICENSE).
