# Confirmed research result

## Summary

On August 13, 2026, a cumulative shadow-learner checkpoint passed an independent, candidate-blind promotion evaluation on 500 newly generated levels.

| Metric | Previous champion | Candidate | Difference |
| --- | ---: | ---: | ---: |
| Weighted solve rate | 0.6104 | 0.6260 | **+0.0156** |
| One-sided 95% paired-bootstrap lower bound |  |  | **+0.0044** |

The promotion contract used equal weights across simulation budgets 20, 34, 57, 95, and 160. All preregistered per-budget regression guards passed.

| Simulation budget | Previous champion | Candidate | Difference |
| ---: | ---: | ---: | ---: |
| 20 | 0.312 | 0.354 | +0.042 |
| 34 | 0.436 | 0.454 | +0.018 |
| 57 | 0.620 | 0.622 | +0.002 |
| 95 | 0.796 | 0.812 | +0.016 |
| 160 | 0.888 | 0.888 | 0.000 |

The uncertainty calculation used 10,000 paired, level-level bootstrap replicates. Pairing matters because both checkpoints were evaluated on the same levels.

## Why the confirmation was candidate-blind

The checkpoint had previously been selected using an opened diagnostic pool. That pool was therefore ineligible to confirm the checkpoint. A fresh pool was generated and sealed without evaluating the candidate, the evaluation contract was fixed, and only then were champion and candidate compared.

The fresh confirmation produced:

- 500 levels;
- five predefined bounded-search budgets;
- equal budget weights;
- a minimum aggregate-improvement requirement;
- per-budget regression guards;
- a strictly positive paired-bootstrap lower-bound requirement.

The candidate passed every condition and became the confirmed incumbent.

## What changed in the training system

Earlier rounds trained each candidate from the champion and discarded it whenever it missed the strict promotion threshold. That protected the official checkpoint, but it also prevented small improvements from accumulating.

The revised system separates two roles:

- **Champion:** the officially evaluated model used for comparison and reporting.
- **Shadow learner:** a checkpoint that can continue receiving safe cumulative updates.

A looser continuation decision protects the learner against catastrophic regressions, while the stricter promotion decision protects the champion. This change produced reproducible medium- and high-budget improvements and ultimately the independently confirmed promotion above.

## What this result does not establish

- The absolute gain is modest; this is evidence that the pipeline can improve the solver, not that learning is complete.
- The evaluation measures bounded solver performance, not human-perceived puzzle quality or fun.
- The result applies to the registered generator distribution and tested budgets.
- The model checkpoint and sealed pool are intentionally not bundled in this source repository.
- Continued co-training may expose new stability/plasticity tradeoffs, so later checkpoints require the same independent evaluation discipline.

## Reproducibility boundary

The implementation needed to construct pools, train candidates, evaluate bounded search, enforce regression guards, and calculate paired uncertainty is public here. Large replay buffers, disposable checkpoints, full run directories, and still-sealed evaluation data are excluded from Git. Public research artifacts can be added as compact, immutable release assets after they are retired from future model selection.
