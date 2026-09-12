# P2.10 frozen-router independent confirmation

The P2.7 four-prototype library and P2.8 branch-specific ridge routers are frozen. This run uses a new seed-75000 confirmatory screen and a 5000-permutation alignment test. The numerical perturbation envelope is retained only as a reproducible sensitivity domain; this is not AEB-calibrated validation.

## Independent screen

| branch | N | router-1 | router-2 | fixed-2 | script | router2-fixed2 [95% CI] | router2-script [95% CI] | perm delta | perm p | valid | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 121 | 0.380 | 0.388 | 0.357 | 0.000 | 0.031 [0.008, 0.060] | 0.388 [0.306, 0.471] | 0.043 | 0.0008 | 1.000 | True |
| dual | 104 | 0.323 | 0.373 | 0.290 | 0.015 | 0.083 [0.035, 0.135] | 0.358 [0.271, 0.444] | 0.051 | 0.0018 | 0.969 | True |

Router-2 interaction cost (mean decision steps until success or two-candidate exhaustion): single 119.3; dual 120.9.

Both-branch screen gate: **True**.

## Conditionally opened fresh development

| branch | N | router-1 | router-2 | fixed-2 | script | router2-fixed2 [95% CI] | router2-script [95% CI] | perm delta | perm p | valid | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 107 | 0.305 | 0.325 | 0.279 | 0.000 | 0.047 [0.021, 0.077] | 0.325 [0.243, 0.409] | 0.043 | 0.0034 | 1.000 | True |
| dual | 116 | 0.305 | 0.345 | 0.267 | 0.034 | 0.078 [0.038, 0.124] | 0.310 [0.234, 0.391] | 0.042 | 0.0066 | 0.963 | True |

Router-2 interaction cost (mean decision steps until success or two-candidate exhaustion): single 124.9; dual 125.7.

Both-branch fresh-development gate: **True**.

The frozen two-candidate router passed both independent stages and is eligible for a later, explicitly authorized one-time heldout evaluation. P2.10 did not read heldout.

The result concerns dangerous-and-valid coverage with up to two candidate rollouts per condition. It is not evidence that one closed-loop policy execution beats the script.
