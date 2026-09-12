# P2.11 frozen single-candidate router confirmation

The frozen ridge router observes the first five actor-visible frames, selects one frozen prototype, and executes exactly one candidate episode. This is a single-candidate conditional selection experiment, not a continuously reactive actor. The perturbations form a numerical sensitivity domain and are not an AEB-calibrated distribution.

| branch | N | router-1 | fixed-1 | script | router1-fixed1 [95% CI] | router1-script [95% CI] | perm delta | perm p | valid | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 239 | 0.352 | 0.308 | 0.000 | 0.044 [0.012, 0.078] | 0.352 [0.295, 0.410] | 0.110 | 0.0002 | 1.000 | True |
| dual | 232 | 0.253 | 0.216 | 0.023 | 0.037 [0.003, 0.073] | 0.230 [0.178, 0.284] | 0.068 | 0.0002 | 0.959 | True |

Both-branch gate: **True**.

The one-candidate router becomes the preferred final method and is eligible for a later user-authorized one-time heldout evaluation. Heldout was not read in P2.11.

The primary comparison is equal-budget router-1 versus fixed-1. The script comparison also uses one episode. Results remain limited to dangerous-and-valid scenario coverage.
