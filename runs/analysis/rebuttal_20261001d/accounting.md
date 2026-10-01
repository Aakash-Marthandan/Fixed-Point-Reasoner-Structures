# P6 — the common accounting (saved records; every gate reproduced)

Identity: 1 − A = (1 − U) + (U − C) + (C − A). Fractions of the configuration's problems. 'unmeasured' = only endpoints were saved.

## Fixed-start trajectories (one candidate: C = A)
| configuration | n | 1 − U | U − C | C − A | 1 − A | any temporary loss |
|---|---|---|---|---|---|---|
| Attention 128, full test, one fixed-start trajectory, depth 64 | 422786 | 0.52 % | 0.00 % | 0 | 0.52 % | 0 |
| Attention 192, full test, one fixed-start trajectory, depth 64 | 422786 | 0.38 % | 0.00 % | 0 | 0.38 % | 4 |
| Attention 256, full test, one fixed-start trajectory, depth 64 | 422786 | 0.38 % | 0.00 % | 0 | 0.38 % | 7 |
| MLP 192 at 46k, 128 validation puzzles, start 'cold', depth 16 | 128 | 1.56 % | 0.00 % | 0 | 1.56 % | — |
| MLP 192 at 46k, 128 validation puzzles, start 'ri', depth 16 | 128 | 3.12 % | 0.00 % | 0 | 3.12 % | — |
| MLP 192 at 46k, 128 validation puzzles, start 'rifix', depth 16 | 128 | 0.78 % | 0.00 % | 0 | 0.78 % | — |
| MLP 192 at 46k, 128 validation puzzles, start 'sym', depth 16 | 128 | 3.91 % | 0.00 % | 0 | 3.91 % | — |
| MLP 192 at 46k, 128 validation puzzles, start 'anchor', depth 16 | 128 | 0.00 % | 0.00 % | 0 | 0.00 % | — |
| MLP 192 at 94k, 128 validation puzzles, start 'cold', depth 16 | 128 | 46.88 % | 44.53 % | 0 | 91.41 % | — |
| MLP 192 at 94k, 128 validation puzzles, start 'ri', depth 16 | 128 | 3.12 % | 0.00 % | 0 | 3.12 % | — |
| MLP 192 at 94k, 128 validation puzzles, start 'rifix', depth 16 | 128 | 3.12 % | 0.00 % | 0 | 3.12 % | — |
| MLP 192 at 94k, 128 validation puzzles, start 'sym', depth 16 | 128 | 2.34 % | 3.12 % | 0 | 5.47 % | — |
| MLP 192 at 94k, 128 validation puzzles, start 'anchor', depth 16 | 128 | 0.00 % | 0.00 % | 0 | 0.00 % | — |
| Released TRM on ARC, 419 queries, fixed start, depth 16 | 419 | 67.06 % | 2.39 % | 0 | 69.45 % | — |

## Candidate banks (U unmeasured; the residual selector's gap C − A; beside it the verifier selector, which returns the fixed start if exact or else any draw among the first k that passes the constraint check)
| configuration | k | 1 − C | C − A | 1 − A | verifier: 1 − A_verif |
|---|---|---|---|---|---|
| Our TRM, baseline seed 0 (50k) | 1 | 7.88 % | 0.00 % | 7.88 % | 5.00 % |
| Our TRM, baseline seed 0 (50k) | 8 | 2.54 % | 1.97 % | 4.50 % | 2.43 % |
| Our TRM, baseline seed 0 (50k) | 32 | 1.17 % | 3.75 % | 4.92 % | 1.16 % |
| Our TRM, baseline seed 0 (50k) | 128 | 0.29 % | 7.22 % | 7.52 % | 0.29 % |
| Our TRM, baseline seed 1 (40k) | 1 | 41.51 % | 0.00 % | 41.51 % | 4.33 % |
| Our TRM, baseline seed 1 (40k) | 8 | 2.90 % | 36.49 % | 39.38 % | 2.21 % |
| Our TRM, baseline seed 1 (40k) | 32 | 1.09 % | 41.84 % | 42.93 % | 1.08 % |
| Our TRM, baseline seed 1 (40k) | 128 | 0.24 % | 46.12 % | 46.36 % | 0.24 % |
| Our TRM, answer anchors only (42k) | 1 | 46.69 % | 0.00 % | 46.69 % | 4.67 % |
| Our TRM, answer anchors only (42k) | 8 | 4.01 % | 47.82 % | 51.83 % | 2.19 % |
| Our TRM, answer anchors only (42k) | 32 | 0.98 % | 56.07 % | 57.05 % | 0.92 % |
| Our TRM, answer anchors only (42k) | 128 | 0.14 % | 61.63 % | 61.77 % | 0.14 % |
| Our TRM, random starts only (48k) | 1 | 8.01 % | 0.00 % | 8.01 % | 5.82 % |
| Our TRM, random starts only (48k) | 8 | 3.81 % | 0.06 % | 3.88 % | 3.58 % |
| Our TRM, random starts only (48k) | 32 | 2.42 % | 0.03 % | 2.44 % | 2.35 % |
| Our TRM, random starts only (48k) | 128 | 1.69 % | 0.02 % | 1.71 % | 1.68 % |
| Attention 128, 5,000 puzzles | 1 | 0.58 % | 0.00 % | 0.58 % | 0.20 % |
| Attention 128, 5,000 puzzles | 8 | 0.06 % | 0.02 % | 0.08 % | 0.06 % |
| Attention 128, 5,000 puzzles | 32 | 0.06 % | 0.00 % | 0.06 % | 0.06 % |
| Attention 128, 5,000 puzzles | 128 | 0.04 % | 0.00 % | 0.04 % | 0.04 % |
| Attention 192, 5,000 puzzles | 1 | 0.40 % | 0.00 % | 0.40 % | 0.22 % |
| Attention 192, 5,000 puzzles | 8 | 0.12 % | 0.00 % | 0.12 % | 0.12 % |
| Attention 192, 5,000 puzzles | 32 | 0.02 % | 0.00 % | 0.02 % | 0.02 % |
| Attention 192, 5,000 puzzles | 128 | 0.00 % | 0.02 % | 0.02 % | 0.00 % |
| Attention 256, 5,000 puzzles | 1 | 0.38 % | 0.00 % | 0.38 % | 0.22 % |
| Attention 256, 5,000 puzzles | 8 | 0.06 % | 0.00 % | 0.06 % | 0.06 % |
| Attention 256, 5,000 puzzles | 32 | 0.04 % | 0.02 % | 0.06 % | 0.04 % |
| Attention 256, 5,000 puzzles | 128 | 0.02 % | 0.06 % | 0.08 % | 0.02 % |
| Earlier Attention 256, reference (28k) | 1 | 0.42 % | 0.00 % | 0.42 % | 0.26 % |
| Earlier Attention 256, reference (28k) | 8 | 0.16 % | 0.02 % | 0.18 % | 0.16 % |
| Earlier Attention 256, reference (28k) | 32 | 0.06 % | 0.02 % | 0.08 % | 0.06 % |
| Earlier Attention 256, no starts or anchors (30k) | 1 | 0.88 % | 0.00 % | 0.88 % | 0.08 % |
| Earlier Attention 256, no starts or anchors (30k) | 8 | 0.12 % | 0.58 % | 0.70 % | 0.04 % |
| Earlier Attention 256, no starts or anchors (30k) | 32 | 0.04 % | 1.38 % | 1.42 % | 0.02 % |
| Earlier Attention 256, changed batch and rate (22k) | 1 | 8.26 % | 0.00 % | 8.26 % | 6.44 % |
| Earlier Attention 256, changed batch and rate (22k) | 8 | 4.06 % | 0.04 % | 4.10 % | 3.98 % |
| Earlier Attention 256, changed batch and rate (22k) | 32 | 2.64 % | 0.02 % | 2.66 % | 2.62 % |
| Earlier Attention 256, no damping or noise (20k) | 1 | 0.46 % | 0.00 % | 0.46 % | 0.22 % |
| Earlier Attention 256, no damping or noise (20k) | 8 | 0.06 % | 0.00 % | 0.06 % | 0.06 % |
| Earlier Attention 256, no damping or noise (20k) | 32 | 0.04 % | 0.00 % | 0.04 % | 0.04 % |

## ARC bank (eight Gaussian starts, depth 16)
| k | 1 − C | C − A | 1 − A |
|---|---|---|---|
| 1 | 68.02 % | 0.00 % | 68.02 % |
| 2 | 68.02 % | 0.24 % | 68.26 % |
| 4 | 67.78 % | 0.48 % | 68.26 % |
| 8 | 67.54 % | 1.19 % | 68.74 % |

Union of the fixed trajectory's 16 readouts and the eight endpoints: 146 of 419 covered.

Gates (17): all reproduced.
