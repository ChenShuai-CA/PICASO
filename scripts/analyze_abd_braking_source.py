"""Compare operator-confirmed AEB and manual-braking event shapes in ABD data.

This is a diagnostic separability study, not a validated automatic labeller.
The positive examples are 44 reviewed AEB-test braking windows.  The manual
examples are 33 FCW-only runs where the operator confirmed braking after the
audible warning.  Robot-active runs are excluded by construction.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from calibrate_abd_v1 import features as response_features, load_run  # noqa: E402
from finalize_abd_proxy_v2 import read_csv_auto, normalize_run  # noqa: E402
from screen_abd_no_takeover import analyze_candidate  # noqa: E402

ABD_ROOT = ROOT / 'Data/ABD_Data'
AEB_REVIEWED = ROOT / 'runs/20260912_abd_proxy_calibration_v2/reviewed_aeb_runs.csv'
FCW_REVIEW = ROOT / 'runs/20260912_abd_fcw_audio_audit/manual_intervention_review.csv'
OUT = ROOT / 'runs/20260912_abd_braking_source_analysis'

FEATURES = (
    'peak_deceleration_magnitude_mps2',
    'effective_constant_deceleration_distance_mps2',
    'time_onset_to_peak_s',
    'event_duration_s',
    'jerk_p95_mps3',
    'lateral_velocity_delta_peak_mps',
    'yaw_rate_delta_peak_dps',
    'heading_change_deg',
    'speed_rebound_kph',
    'positive_speed_step_fraction',
    'brake_pulse_count',
    'br_position_travel_mm',
    'brake_force_max_n',
)


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ['run'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def manual_review_rows() -> list[dict]:
    rows = []
    for row in read_csv_auto(FCW_REVIEW):
        note = row.get('driver_intervention', '')
        if '人工接管' not in note and 'manual' not in note.casefold():
            continue
        rows.append({
            'run': normalize_run(row['run']),
            'vehicle': row['vehicle'],
            'label': 'manual_after_fcw_audio',
            'y_aeb': 0,
        })
    return rows


def aeb_review_rows() -> list[dict]:
    return [{
        'run': normalize_run(row['run']),
        'vehicle': row['vehicle'],
        'label': 'operator_confirmed_aeb_response',
        'y_aeb': 1,
    } for row in read_csv_auto(AEB_REVIEWED)]


def extract(row: dict) -> dict:
    path = ABD_ROOT / row['run']
    screened = analyze_candidate(path, ABD_ROOT)
    arrays, _ = load_run(path)
    response = response_features(arrays)
    if response is None:
        raise RuntimeError(f'no braking response in reviewed run: {row["run"]}')
    if screened['event_status'] != 'br_zero_observed_braking':
        raise RuntimeError(
            f'reviewed human/AEB comparison run has robot activity: {row["run"]}')
    return {
        **row,
        'scenario': screened['scenario'],
        'onset_speed_kph': response['onset_speed_kph'],
        'peak_deceleration_magnitude_mps2': -response['peak_deceleration_mps2'],
        'effective_constant_deceleration_distance_mps2':
            response['effective_constant_deceleration_distance_mps2'],
        'time_onset_to_peak_s': response['time_onset_to_peak_s'],
        'event_duration_s': response['event_duration_s'],
        'br_position_travel_mm': response['br_position_travel_mm'],
        'brake_force_max_n': response['brake_force_max_n'],
        **{name: screened[name] for name in (
            'jerk_p95_mps3', 'lateral_velocity_delta_peak_mps',
            'yaw_rate_delta_peak_dps', 'heading_change_deg',
            'speed_rebound_kph', 'positive_speed_step_fraction',
            'brake_pulse_count')},
    }


def describe(values) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        'n': int(len(array)),
        'p10': round(float(np.quantile(array, .10)), 4),
        'median': round(float(np.median(array)), 4),
        'p90': round(float(np.quantile(array, .90)), 4),
    }


def auc_score(y, scores) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = y == 1, y == 0
    ranks = rankdata(scores, method='average')
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def balanced_accuracy(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    sensitivity = float(np.mean(pred[y == 1] == 1))
    specificity = float(np.mean(pred[y == 0] == 0))
    return {
        'sensitivity_aeb': round(sensitivity, 4),
        'specificity_manual': round(specificity, 4),
        'balanced_accuracy': round((sensitivity + specificity) / 2, 4),
    }


def fit_threshold(y, values) -> tuple[float, str]:
    y, values = np.asarray(y, dtype=int), np.asarray(values, dtype=float)
    unique = np.unique(values)
    if len(unique) == 1:
        return float(unique[0]), 'aeb_if_high'
    thresholds = np.r_[unique[0] - 1e-9, (unique[:-1] + unique[1:]) / 2,
                       unique[-1] + 1e-9]
    best = (-1., None, None)
    for threshold in thresholds:
        for direction in ('aeb_if_high', 'aeb_if_low'):
            pred = values >= threshold if direction == 'aeb_if_high' else values <= threshold
            score = balanced_accuracy(y, pred.astype(int))['balanced_accuracy']
            candidate = (score, -abs(float(threshold) - float(np.median(values))), direction)
            if candidate > (best[0], best[1] or -np.inf, best[2] or ''):
                best = (score, -abs(float(threshold) - float(np.median(values))), direction)
                best_threshold = float(threshold)
    return best_threshold, best[2]


def cross_validated_threshold(rows: list[dict], feature: str, groups=None) -> dict | None:
    available = [r for r in rows if r.get(feature) not in (None, '')
                 and np.isfinite(float(r[feature]))]
    y = np.asarray([r['y_aeb'] for r in available], dtype=int)
    values = np.asarray([float(r[feature]) for r in available])
    if min(np.sum(y == 0), np.sum(y == 1)) < 2:
        return None
    pred = np.zeros(len(available), dtype=int)
    if groups is None:
        folds = [np.asarray([i]) for i in range(len(available))]
        scheme = 'leave_one_run_out'
    else:
        group_values = np.asarray([r[groups] for r in available])
        common = [g for g in np.unique(group_values)
                  if len(np.unique(y[group_values == g])) == 2]
        keep = np.isin(group_values, common)
        y, values, group_values = y[keep], values[keep], group_values[keep]
        available = [r for r, k in zip(available, keep) if k]
        if len(np.unique(group_values)) < 3:
            return None
        pred = np.zeros(len(available), dtype=int)
        folds = [np.flatnonzero(group_values == g) for g in np.unique(group_values)]
        scheme = f'leave_one_{groups}_out_on_class-overlap_groups'
    for test in folds:
        train = np.ones(len(y), dtype=bool)
        train[test] = False
        if len(np.unique(y[train])) < 2:
            return None
        threshold, direction = fit_threshold(y[train], values[train])
        pred[test] = ((values[test] >= threshold) if direction == 'aeb_if_high'
                      else (values[test] <= threshold)).astype(int)
    return {'scheme': scheme, 'n': len(y), **balanced_accuracy(y, pred)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [extract(row) for row in aeb_review_rows() + manual_review_rows()]
    write_csv(OUT / 'labelled_event_features.csv', rows)
    y = np.asarray([r['y_aeb'] for r in rows], dtype=int)
    comparisons = []
    for feature in FEATURES:
        available = [r for r in rows if r.get(feature) not in (None, '')
                     and np.isfinite(float(r[feature]))]
        fy = np.asarray([r['y_aeb'] for r in available], dtype=int)
        values = np.asarray([float(r[feature]) for r in available])
        raw_auc = auc_score(fy, values)
        threshold, direction = fit_threshold(fy, values)
        predictions = ((values >= threshold) if direction == 'aeb_if_high'
                       else (values <= threshold)).astype(int)
        comparisons.append({
            'feature': feature,
            'aeb': describe(values[fy == 1]),
            'manual': describe(values[fy == 0]),
            'univariate_auc_aeb_if_value_high': round(raw_auc, 4),
            'univariate_separability_auc': round(max(raw_auc, 1 - raw_auc), 4),
            'in_sample_best_threshold': round(threshold, 4),
            'in_sample_direction': direction,
            'in_sample': balanced_accuracy(fy, predictions),
            'leave_one_run_out': cross_validated_threshold(available, feature),
            'leave_one_vehicle_out': cross_validated_threshold(
                available, feature, groups='vehicle'),
        })
    comparisons.sort(key=lambda x: x['univariate_separability_auc'], reverse=True)

    aeb_speeds = [r['onset_speed_kph'] for r in rows if r['y_aeb'] == 1]
    manual_speeds = [r['onset_speed_kph'] for r in rows if r['y_aeb'] == 0]
    common_speed_low = max(min(aeb_speeds), min(manual_speeds))
    common_speed_high = min(max(aeb_speeds), max(manual_speeds))
    common_speed_rows = [r for r in rows
                         if common_speed_low <= r['onset_speed_kph'] <= common_speed_high]
    speed_overlap_comparisons = []
    for feature in FEATURES:
        available = [r for r in common_speed_rows if r.get(feature) not in (None, '')
                     and np.isfinite(float(r[feature]))]
        fy = np.asarray([r['y_aeb'] for r in available], dtype=int)
        if len(np.unique(fy)) < 2:
            continue
        values = np.asarray([float(r[feature]) for r in available])
        raw_auc = auc_score(fy, values)
        speed_overlap_comparisons.append({
            'feature': feature,
            'aeb': describe(values[fy == 1]),
            'manual': describe(values[fy == 0]),
            'univariate_auc_aeb_if_value_high': round(raw_auc, 4),
            'univariate_separability_auc': round(max(raw_auc, 1 - raw_auc), 4),
        })
    speed_overlap_comparisons.sort(
        key=lambda x: x['univariate_separability_auc'], reverse=True)

    overlap_vehicles = sorted(
        set(r['vehicle'] for r in rows if r['y_aeb'] == 1)
        & set(r['vehicle'] for r in rows if r['y_aeb'] == 0))
    summary = {
        'label_counts': dict(Counter(r['label'] for r in rows)),
        'vehicle_counts_by_label': {
            label: dict(Counter(r['vehicle'] for r in rows if r['label'] == label))
            for label in sorted({r['label'] for r in rows})
        },
        'class_overlap_vehicles': overlap_vehicles,
        'onset_speed_kph_by_label': {
            'aeb': describe(aeb_speeds),
            'manual': describe(manual_speeds),
        },
        'common_speed_support_kph': {
            'low': common_speed_low,
            'high': common_speed_high,
            'aeb_n': sum(r['y_aeb'] == 1 for r in common_speed_rows),
            'manual_n': sum(r['y_aeb'] == 0 for r in common_speed_rows),
        },
        'feature_comparisons': comparisons,
        'common_speed_support_feature_comparisons': speed_overlap_comparisons,
        'interpretation': {
            'use': ('Kinematic features may rank AEB-path BR-zero runs for manual review; '
                    'they are not yet a validated automatic source label.'),
            'selection_bias': ('40 of 44 AEB examples were selected using similarity to '
                               'four confirmed AEB prototypes, so apparent separation is '
                               'optimistically biased.'),
            'domain_confounding': ('Manual examples are FCW-only post-warning takeovers, '
                                   'not early safety takeovers during failed AEB tests. '
                                   'Test intent, speed, vehicle and scenario differ.'),
            'speed_overlap_limit': ('Only four reviewed AEB events fall inside the manual '
                                    'class onset-speed range, so common-speed comparisons '
                                    'are descriptive and underpowered.'),
            'manual_requirement': ('A validated classifier requires operator-labelled '
                                   'manual takeovers drawn from AEB-path BR-zero runs.'),
        },
    }
    (OUT / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    top = comparisons[:5]
    speed_overlap_by_feature = {
        item['feature']: item for item in speed_overlap_comparisons}
    matched_peak = speed_overlap_by_feature['peak_deceleration_magnitude_mps2']
    matched_ttp = speed_overlap_by_feature['time_onset_to_peak_s']
    report = [
        '# ABD AEB-response versus manual-braking diagnostic', '',
        '- Compared 44 operator-confirmed AEB response windows with 33 operator-confirmed '
        'manual braking windows from FCW-only tests; all have zero event-window BR Command.',
        f"- Vehicles present in both classes: {', '.join(overlap_vehicles)}.", '',
        f"- AEB median onset speed: **{summary['onset_speed_kph_by_label']['aeb']['median']:.1f} km/h**; "
        f"manual median: **{summary['onset_speed_kph_by_label']['manual']['median']:.1f} km/h**. "
        f"Only **{summary['common_speed_support_kph']['aeb_n']} AEB** and "
        f"**{summary['common_speed_support_kph']['manual_n']} manual** events lie in the "
        f"common {common_speed_low:.1f}-{common_speed_high:.1f} km/h support.", '',
        '| Feature | AEB median | Manual median | Separation AUC | LORO balanced acc. | LOVO balanced acc. |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for item in top:
        loro = item['leave_one_run_out'] or {}
        lovo = item['leave_one_vehicle_out'] or {}
        report.append(
            f"| {item['feature']} | {item['aeb']['median']:.4f} | "
            f"{item['manual']['median']:.4f} | {item['univariate_separability_auc']:.3f} | "
            f"{loro.get('balanced_accuracy', float('nan')):.3f} | "
            f"{lovo.get('balanced_accuracy', float('nan')):.3f} |")
    report += [
        '',
        'The manuals support BR Command as the direct robot-source discriminator and Motion '
        'Pack channels as braking-shape evidence. They also explicitly warn that acceleration '
        'thresholds can detect pre-brake, driver intervention, or other external deceleration.',
        'Manual evidence: RC Software Manual RM-S-01 Issue 23, sections 6.12.11.11.5 and '
        '6.12.11.11.10 (PDF pp.107 and 110-111); Post Processor User Guide AN-6083 Issue 9, '
        '"Falsely Identified AEB Events" (document p.18 / PDF p.19).',
        '',
        'These results therefore support automated ranking and evidence display. They do not '
        'yet support automatic labelling of the remaining BR-zero AEB-path records, because '
        'the manual comparison class comes from a different FCW-only test domain and 40 AEB '
        'examples were preselected by AEB-prototype similarity.',
        '',
        'The apparent full-sample separation in duration and time-to-peak is strongly '
        'confounded by onset speed. In the common-speed subset, reviewed AEB and manual '
        f"events have nearly equal peak-deceleration medians (AEB {matched_peak['aeb']['median']:.3f} "
        f"vs manual {matched_peak['manual']['median']:.3f} m/s2) and similar onset-to-peak "
        f"times (AEB {matched_ttp['aeb']['median']:.2f} vs manual "
        f"{matched_ttp['manual']['median']:.2f} s). The AEB subset has only four runs. "
        'A rule such as "harder braking means AEB" is not supported by these data.',
        '',
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({
        'labels': summary['label_counts'],
        'overlap_vehicles': overlap_vehicles,
        'top_features': top,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
