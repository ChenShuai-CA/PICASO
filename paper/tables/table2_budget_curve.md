# K=1/K=2 budget curve on the P2.11 independent screen

| branch | method | K | coverage [95% CI] | mean rollouts | mean decision steps | valid |
|---|---|---:|---:|---:|---:|---:|
| single | router | 1 | 0.352 [0.295, 0.408] | 1.000 | 70.1 | 1.000 |
| single | router | 2 | 0.385 [0.326, 0.443] | 1.648 | 120.7 | 1.000 |
| single | fixed | 1 | 0.308 [0.251, 0.366] | 1.000 | 70.2 | 1.000 |
| single | fixed | 2 | 0.314 [0.256, 0.372] | 1.692 | 125.3 | 1.000 |
| dual | router | 1 | 0.253 [0.202, 0.307] | 1.000 | 73.9 | 0.959 |
| dual | router | 2 | 0.307 [0.253, 0.363] | 1.747 | 132.6 | 0.977 |
| dual | fixed | 1 | 0.216 [0.166, 0.268] | 1.000 | 74.8 | 1.000 |
| dual | fixed | 2 | 0.222 [0.172, 0.276] | 1.784 | 137.5 | 1.000 |

Paired K=2 minus K=1 increments:

| branch | method | coverage gain [95% CI] | extra rollouts | extra steps | gain per extra rollout |
|---|---|---:|---:|---:|---:|
| single | router | 0.033 [0.013, 0.056] | 0.648 | 50.6 | 0.050 |
| single | fixed | 0.006 [0.000, 0.014] | 0.692 | 55.1 | 0.008 |
| dual | router | 0.053 [0.028, 0.082] | 0.747 | 58.6 | 0.072 |
| dual | fixed | 0.005 [0.001, 0.011] | 0.784 | 62.7 | 0.007 |
