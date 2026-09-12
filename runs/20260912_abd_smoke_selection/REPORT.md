# ABD historical smoke selection: 14-BZ3X

Selected for parser, event-window and brake-source classification smoke testing. These are not AEB calibration samples because the exports cannot separate vehicle AEB from driver braking.

| run | scenario | role | onset s | onset TTC s | peak m/s2 | BR cmd range | class |
|---|---|---|---:|---:|---:|---:|---|
| V14_T56_R1.txt | CCRs | zero_br_unknown_repeat_1 | 24.792 | 0.884 | -11.419 | 0.000..0.000 | unknown_aeb_or_driver_brake |
| V14_T56_R2.txt | CCRs | zero_br_unknown_repeat_2 | 25.162 | 0.689 | -10.388 | 0.000..0.000 | unknown_aeb_or_driver_brake |
| V14_T133_R2.txt | CCFT | zero_br_unknown_turning | 37.883 | 0.173 | -10.024 | 0.000..0.000 | unknown_aeb_or_driver_brake |
| V14_T503_R4.txt | CPTA | zero_br_unknown_turning | 41.444 | 0.695 | -6.119 | 0.000..0.000 | unknown_aeb_or_driver_brake |
| V14_T133_R1.txt | CCFT | positive_br_robot_negative_control | 11.671 | 16.942 | -4.476 | 18.162..47.329 | robot_brake_negative_control |

All five exports contain 415 channels at approximately 100 Hz and have matching `.spec`, `.log` and `.CRUN` files. The manifest records SHA256 values and exact relative paths; raw files were not copied or modified.

Before using new runs for response calibration, add an independent driver-brake marker and synchronised target command/actual logs. Without vehicle CAN, retain the Post Processor threshold time as `observed_braking_onset`, not AEB request time.
