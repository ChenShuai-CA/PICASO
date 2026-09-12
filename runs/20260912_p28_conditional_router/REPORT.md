# P2.8 conditional prototype routing

The P2.7 four-prototype library is frozen. A scenario-level ridge router uses only the first five actor-visible observation frames to rank those prototypes. All outcome labels and evaluations use `abd_supported_v1_partial`; heldout was not read.

Selected P2.7 seed: 47. Training eligible: single 110, dual 100.

## Formal screen

| branch | N | router-1 | fixed-1 | router-2 | fixed-2 | shuffled-2 | script | delta router2-fixed2 [95% CI] | delta router2-shuffled2 [95% CI] | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 103 | 0.346 | 0.315 | 0.365 | 0.324 | 0.313 | 0.000 | 0.041 [0.012, 0.078] | 0.052 [0.014, 0.099] | True |
| dual | 101 | 0.273 | 0.226 | 0.301 | 0.232 | 0.263 | 0.028 | 0.069 [0.034, 0.113] | 0.038 [-0.006, 0.083] | False |

Both-branch mechanism gate: **False**.

The gate did not pass both branches, so the preregistered fresh development set was not evaluated.

The router is a centralized scenario-level candidate selector over actor-visible histories. It is not a decentralized closed-loop actor policy. Success rates estimate dangerous-and-valid coverage over five paired perturbation draws per condition.
