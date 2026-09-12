# P2.12 one-time heldout confirmation

The frozen P2.11 router selects one prototype from the first five actor-visible frames and executes one candidate episode. The perturbations are a numerical sensitivity domain, not an AEB-calibrated distribution.

| branch | N | router-1 | fixed-1 | script | router1-fixed1 [95% CI] | router1-script [95% CI] | perm delta | perm p | valid | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 222 | 0.368 | 0.327 | 0.000 | 0.041 [0.012, 0.071] | 0.368 [0.310, 0.428] | 0.114 | 0.0002 | 1.000 | True |
| dual | 227 | 0.204 | 0.174 | 0.022 | 0.029 [0.001, 0.058] | 0.181 [0.135, 0.230] | 0.057 | 0.0002 | 0.942 | True |

Both-branch final gate: **True**.

This is the sole final heldout attempt under the frozen P2.12 protocol. Its result is reported as observed and does not trigger model, budget, seed, denominator, or gate changes.
