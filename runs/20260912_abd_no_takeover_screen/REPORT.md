# ABD no-takeover candidate screen

- TXT files encountered: **5543**; channel data exports: **4121**; configuration metadata TXT: **1422**.
- AEB-path exports screened: **1258**.
- BR-zero exports with an observed braking event: **892**.
- Manual-review queue: **40** runs across **10** vehicle folders and **13** scenario labels.

Dynamic target reference/actual channels are usable in **781** BR-zero braking runs. The median per-run P95 aligned position residual is **0.002809 m** and its across-run 95th percentile is **0.022391 m**. The maximum is **1.157118 m**, so outliers require a quality gate before fitting a distribution.

The queue is ranked from four operator-reviewed 14-BZ3X examples. With only four positive examples and no independently instrumented manual-braking negatives, the score is a review-priority measure. Only the operator may change `driver_intervention` from `unknown`.

The detected time is `observed_braking_onset`: a sustained measured deceleration threshold backtracked to -0.3 m/s2. It cannot establish ECU AEB request time without vehicle CAN or another authoritative trigger channel.

Target `reference` and `actual` X/Y channels share the exported `Time` rows. Their tracking errors are reported where the target trajectory is dynamic. This supports target execution-error analysis, while LaunchPad low-level actuator channels remain available only when captured by the corresponding target system.
