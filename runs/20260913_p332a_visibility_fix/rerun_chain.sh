#!/usr/bin/env bash
# P3.3.2a rerun chain: M2 x2 (cross-process bitwise check), CV, M3, teacher probe.
set -e
cd /mnt/d/Projects/Scenario_Generation_Research
P=/home/shuai/.venvs/scenario-gpu/bin/python
D=runs/20260913_p332a_visibility_fix

"$P" research_tasks/train_p33_nominal.py --mode smoke > "$D/m2_run.log" 2>&1 && echo M2_RUN1_DONE
"$P" research_tasks/train_p33_nominal.py --mode smoke --updates 200 > "$D/m2_run2.log" 2>&1 && echo M2_RUN2_DONE
"$P" -c "import numpy as np; a=np.loadtxt('$D/M2_smoke_loss_curve.csv',delimiter=','); b=np.loadtxt('$D/M2_smoke_loss_curve_probe.csv',delimiter=','); print('CROSS_PROCESS_BITWISE', bool((a==b).all()))" | tee "$D/cross_process_check.txt"
"$P" research_tasks/evaluate_p33_t1.py --mode cv > "$D/cv_run.log" 2>&1 && echo CV_DONE
"$P" research_tasks/evaluate_p33_t1.py --mode model > "$D/m3_run.log" 2>&1 && echo M3_DONE
"$P" "$D/teacher_decode_probe.py" > "$D/teacher_decode_probe.log" 2>&1 && echo TEACHER_PROBE_DONE
