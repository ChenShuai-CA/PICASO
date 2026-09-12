# ABD historical smoke selection: 14-BZ3X

Selected for parser, event-window and brake-source classification smoke testing. The project test operator reviewed manual takeover using Robot Controller Check Paths and Motion Pack forward/lateral velocity curves. The four BR-zero runs have no observed takeover signature and may enter observed braking response analysis. They are not AEB request-timing samples because the exports contain no direct vehicle AEB request/status.

| run | scenario | role | driver review | onset s | onset TTC s | peak m/s2 | BR cmd range | class |
|---|---|---|---|---:|---:|---:|---:|---|
| V14_T56_R1.txt | CCRs | zero_br_unknown_repeat_1 | none_confirmed | 24.792 | 0.884 | -11.419 | 0.000..0.000 | observed_braking_no_takeover_signature_aeb_unconfirmed |
| V14_T56_R2.txt | CCRs | zero_br_unknown_repeat_2 | none_confirmed | 25.162 | 0.689 | -10.388 | 0.000..0.000 | observed_braking_no_takeover_signature_aeb_unconfirmed |
| V14_T133_R2.txt | CCFT | zero_br_unknown_turning | none_confirmed | 37.883 | 0.173 | -10.024 | 0.000..0.000 | observed_braking_no_takeover_signature_aeb_unconfirmed |
| V14_T503_R4.txt | CPTA | zero_br_unknown_turning | none_confirmed | 41.444 | 0.695 | -6.119 | 0.000..0.000 | observed_braking_no_takeover_signature_aeb_unconfirmed |
| V14_T133_R1.txt | CCFT | positive_br_robot_negative_control | n/a | 11.671 | 16.942 | -4.476 | 18.162..47.329 | robot_brake_negative_control |

All five exports contain 415 channels at approximately 100 Hz and have matching `.spec`, `.log` and `.CRUN` files. The manifest records SHA256 values and exact relative paths; raw files were not copied or modified.

Before using new runs for response calibration, add an independent driver-brake marker and synchronised target command/actual logs. Without vehicle CAN, retain the Post Processor threshold time as `observed_braking_onset`, not AEB request time.

Human review is entered only in the `driver_intervention` column of `manual_intervention_review.csv`. Allowed values are `none_confirmed`, `manual`, and `unknown`. The shared review method and its limitation are stored in `manifest.json`. Re-run this script after editing; it validates the four paths and preserves the review file.
