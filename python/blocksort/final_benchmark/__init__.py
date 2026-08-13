"""Milestone 8: final benchmark, longer co-training experiment, and ablations.

This package contains *experiment drivers* (not part of the training library):

* ``harder``     -- build a harder, non-saturated frozen benchmark + held-out
                    promotion levels; verify the initial protagonist is not
                    saturated.
* ``solver``     -- compare solver variants (raw policy, graph search at fixed
                    budgets, exact A*) on identical frozen states.
* ``generators`` -- compare level generators (random, BC designer, adversarial
                    designer, co-trained designer) over multiple seeds.
* ``ablations``  -- compact ablations for the most important design choices.
* ``report``     -- assemble machine-readable artifacts + a Markdown report.
* ``run``        -- orchestrate everything under ``runs/final_benchmark/``.

No web serving, browser, telemetry, distributed training, new architectures, new
mechanics, or new automated tests (see milestone scope).
"""
