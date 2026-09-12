"""P3: build aeb_dataset_v1 from v0 by removing residual foot-confusion risk.

Operator decisions (2026-09-13):
1. The 46-row REVIEW_QUEUE (foot 33 / adjudicate 6 / mid-window 7) is
   permanently excluded instead of awaiting review. The earlier proposal to
   admit the 5 small-cmd mid-window runs after tail truncation is withdrawn
   (operator cannot confirm them manually either).
2. Rows whose only evidence for "no driver foot" is the machine signature
   carry a residual risk: a very light foot (<=50 N) is indistinguishable
   from an AEB pedal drag. Remove the quantifiable risk bands:
   - foot_pre_event=1  (pedal moving in the 2 s before the event; the
     operator-confirmed AEB set is 44/44 foot_pre_event=0)
   - no tension AND force_max_win >= 30 N  (above the confirmed set's main
     band: median 12.7 N, p90 ~25 N). Operator-confirmed rows are exempt
     from this rule -- the removal's purpose is "cannot be ruled out
     manually", and those rows were ruled out manually. (Confirmed-set
     counterexamples V2_T76_R1 37.6 N / V2_T1170_R2 32.5 N stay in.)

Input : runs/20260912_abd_aeb_pedal_screen/{aeb_dataset_v0.csv,
        per_run_signature.csv}
Output: runs/20260912_abd_aeb_pedal_screen/{aeb_dataset_v1.csv,
        aeb_dataset_v1_removed.csv}
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/20260912_abd_aeb_pedal_screen'

NO_TENSION_FORCE_N = 30.0
CONFIRMED_LABELS = ('none_confirmed', 'manual_after_aeb_stop')


def main():
    v0 = list(csv.DictReader(
        (BASE / 'aeb_dataset_v0.csv').open(encoding='utf-8-sig')))
    sig = {r['run']: r for r in csv.DictReader(
        (BASE / 'per_run_signature.csv').open(encoding='utf-8-sig'))}

    kept, removed = [], []
    for r in v0:
        s = sig[r['run']]
        reason = ''
        if s['foot_pre_event'] == '1':
            reason = 'foot_pre_event'
        elif (s['tension'] in ('', '0')
                and float(s['force_max_win'] or 0) >= NO_TENSION_FORCE_N
                and r['operator_label'] not in CONFIRMED_LABELS):
            reason = f'no_tension_force_ge_{NO_TENSION_FORCE_N:.0f}n'
        row = dict(r, removal_reason=reason)
        (kept if not reason else removed).append(row)

    with (BASE / 'aeb_dataset_v1.csv').open('w', encoding='utf-8-sig',
                                            newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kept[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    with (BASE / 'aeb_dataset_v1_removed.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(removed)

    from collections import Counter
    print(f"v0 {len(v0)} -> v1 {len(kept)} (removed {len(removed)})")
    print("removal reasons:", dict(Counter(
        r['removal_reason'] for r in removed)))
    print("v1 signature:", dict(Counter(r['signature'] for r in kept)))
    print("v1 unique sha256:", len({r['sha256'] for r in kept}))
    print("v1 operator-confirmed:", sum(
        1 for r in kept if r['operator_label'] in CONFIRMED_LABELS))


if __name__ == '__main__':
    main()
