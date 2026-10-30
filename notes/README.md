# Notes

## Design decisions
- **Feasibility is a gate, not a trade-off.** An infeasible solution cannot be compared on distance at all, so every move checks capacity first.
- **Run many strategies instead of tuning one.** Each construction method fails on a different instance shape (clustered vs. spread out vs. ring-shaped). Running all of them and keeping the best was more reliable than tuning a single method.
- **Never trust the solver's own numbers.** The validator recomputes loads and distances from the original file, so a bug in the solver cannot hide behind a "PASS".
- **Vehicle count comes from the data, not the file name.** `k5` in `A-n32-k5` is a hint, not an input.

## What changed between versions
| Version | Change | Why |
|---|---|---|
| v1 → v2 | Added 2-opt, Or-opt, Relocate, Swap, Cross-exchange and more construction methods | Nearest Neighbor alone left large gaps |
| v2 → v3 | Added Sweep, Perturbation, time limit, auto vehicle count | Local search got stuck in local optima; large instances could run too long |
| v3 → v4 | Cleaner output, skip messages for methods that do not apply | Easier for judges and users to read |

## Lessons learned
- Brute force grows as n!, so even 30 customers is out of reach; heuristics + local search are the practical answer.
- On P-n40-k5 the single-start NN + local search answer was 11.6% from optimal, while multi-start reached 0%. The starting point matters as much as the improvement step.
- In the self-created case, fewer empty-return kilometres was the main source of CO₂ savings (37.1% of the reduction came from empty trips).

## Limitations / future work
- Distances are straight-line (EUC_2D); real roads, traffic and time windows are not modelled.
- Only one vehicle type; a heterogeneous fleet (6-, 10-, 12-wheel trucks) would be closer to real operations.
- Results should be validated with real traffic and operational data before deployment.
