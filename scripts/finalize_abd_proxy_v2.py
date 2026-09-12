"""Finalize operator-reviewed ABD evidence for proxy-supported calibration v2.

The AEB review tables identify braking events whose main braking window is not
contaminated by driver or robot intervention.  The FCW table has different test
semantics: those runs are FCW-only tests and the driver normally brakes after
the audible warning.  Consequently their audio edge remains usable FCW
evidence, while their subsequent braking is never used as an AEB response.

This script preserves those two evidence roles, re-extracts AEB response
features from raw exports, de-duplicates by file hash, and writes a calibration
evidence package.  The resulting v2 JSON is deliberately marked incompatible
with the current simulator until its combined ``response_delay`` is split into
trigger-policy and actuation-delay terms.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_abd_v1 import features, load_run  # noqa: E402
from screen_abd_no_takeover import aeb_request_proxy  # noqa: E402

ABD_ROOT = ROOT / "Data/ABD_Data"
SCREEN_REVIEW = ROOT / "runs/20260912_abd_no_takeover_screen/manual_intervention_review.csv"
ANCHOR_REVIEW = ROOT / "runs/20260912_abd_smoke_selection/manual_intervention_review.csv"
FCW_REVIEW = ROOT / "runs/20260912_abd_fcw_audio_audit/manual_intervention_review.csv"
FCW_MAPPINGS = ROOT / "runs/20260912_abd_fcw_audio_audit/fcw_audio_mappings.csv"
OUT = ROOT / "runs/20260912_abd_proxy_calibration_v2"


def read_csv_auto(path: Path) -> list[dict]:
    """Read Excel-edited review CSVs without silently replacing Chinese text."""
    raw = path.read_bytes()
    last_error = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            return list(csv.DictReader(text.splitlines()))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeError(f"cannot decode review CSV {path}: {last_error}")


def normalize_run(value: str) -> str:
    run = value.strip().replace("\\", "/")
    prefix = "Data/ABD_Data/"
    return run[len(prefix):] if run.startswith(prefix) else run


def classify_aeb_review(value: str) -> dict:
    note = value.strip()
    folded = note.casefold()
    if folded in {"none", "none_confirmed"}:
        return {
            "intervention_class": "none_confirmed",
            "aeb_response_eligibility": "eligible_full_braking_window",
            "analysis_window_rule": "detected_braking_event",
        }
    if ("aeb" in folded and "刹停避撞之后" in note and "人工接管" in note):
        return {
            "intervention_class": "manual_after_aeb_stop",
            "aeb_response_eligibility": "eligible_through_first_stop_only",
            "analysis_window_rule": "detected_braking_event_ending_at_first_stop",
        }
    if "manual" in folded or "人工" in note:
        return {
            "intervention_class": "manual_timing_not_resolved",
            "aeb_response_eligibility": "excluded_manual_intervention",
            "analysis_window_rule": "none",
        }
    return {
        "intervention_class": "unresolved",
        "aeb_response_eligibility": "excluded_unresolved",
        "analysis_window_rule": "none",
    }


def classify_fcw_review(value: str) -> str:
    note = value.strip()
    folded = note.casefold()
    if "人工接管" in note or "manual" in folded:
        return "manual_after_fcw_audio_expected_by_test_procedure"
    if folded in {"none", "none_confirmed"}:
        return "none_confirmed"
    return "unresolved"


def stat(values) -> dict | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "n": int(len(array)),
        "min": round(float(array.min()), 4),
        "p05": round(float(np.quantile(array, 0.05)), 4),
        "median": round(float(np.median(array)), 4),
        "p95": round(float(np.quantile(array, 0.95)), 4),
        "max": round(float(array.max()), 4),
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std(ddof=1)), 4) if len(array) > 1 else None,
    }


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["run"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reviewed_aeb_records() -> tuple[list[dict], list[dict]]:
    source_rows = []
    for source, path in (("operator_smoke_anchor", ANCHOR_REVIEW),
                         ("operator_expanded_review", SCREEN_REVIEW)):
        for row in read_csv_auto(path):
            decision = classify_aeb_review(row.get("driver_intervention", ""))
            source_rows.append({
                "run": normalize_run(row["run"]),
                "vehicle": row.get("vehicle", ""),
                "scenario": row.get("scenario", ""),
                "review_source": source,
                "operator_note": row.get("driver_intervention", "").strip(),
                **decision,
            })

    eligible = [row for row in source_rows
                if row["aeb_response_eligibility"].startswith("eligible_")]
    unresolved = [row for row in source_rows
                  if row["aeb_response_eligibility"] == "excluded_unresolved"]
    if unresolved:
        raise RuntimeError(f"unresolved AEB operator reviews: {len(unresolved)}")
    records = []
    for row in eligible:
        raw_path = ABD_ROOT / row["run"]
        arrays, digest = load_run(raw_path)
        response = features(arrays)
        if response is None:
            raise RuntimeError(f"reviewed AEB run has no response event: {row['run']}")
        proxy = aeb_request_proxy(response["onset_time_s"])
        records.append({
            **row,
            "sha256": digest,
            **response,
            **proxy,
        })

    unique, duplicate_rows = [], []
    seen = {}
    for row in records:
        if row["sha256"] in seen:
            duplicate_rows.append({
                "run": row["run"], "sha256": row["sha256"],
                "duplicate_of": seen[row["sha256"]],
            })
        else:
            seen[row["sha256"]] = row["run"]
            unique.append(row)
    return unique, duplicate_rows


def reviewed_fcw_records() -> list[dict]:
    review = {
        normalize_run(row["run"]): {
            "operator_note": row.get("driver_intervention", "").strip(),
            "intervention_class": classify_fcw_review(
                row.get("driver_intervention", "")),
        }
        for row in read_csv_auto(FCW_REVIEW)
    }
    mappings = read_csv_auto(FCW_MAPPINGS)
    records = []
    for row in mappings:
        if row.get("trace_status") != "fcw_audio_rising_edge_found":
            continue
        run = normalize_run(row["run"])
        reviewed = review.get(run, {})
        braking_source = row.get("braking_source_status", "no_observed_braking")
        if braking_source == "robot_channel_active":
            braking_interpretation = "robot_braking_after_fcw"
        elif reviewed.get("intervention_class", "").startswith("manual_after_fcw"):
            braking_interpretation = "manual_braking_after_fcw"
        elif braking_source == "no_observed_braking" or not row.get(
                "observed_braking_onset_s", "").strip():
            braking_interpretation = "no_observed_braking"
        else:
            braking_interpretation = "unresolved_non_aeb_braking"
        records.append({
            "run": run,
            "vehicle": row["vehicle"],
            "ttt_index": row["ttt_index"],
            "can_user_defined_index": row["can_user_defined_index"],
            "configured_label": row["configured_label"],
            "t_fcw_audio_observed_s": row["t_fcw_audio_observed_s"],
            "test_intent": "fcw_only_no_aeb",
            "eligible_for_fcw_audio_timing": True,
            "eligible_for_aeb_response_or_proxy": False,
            "braking_interpretation": braking_interpretation,
            "intervention_class": reviewed.get("intervention_class", "not_reviewed"),
            "operator_note": reviewed.get("operator_note", ""),
        })
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aeb, duplicates = reviewed_aeb_records()
    fcw = reviewed_fcw_records()
    unresolved_aeb = [r for r in aeb if r["intervention_class"] == "unresolved"]
    if unresolved_aeb:
        raise RuntimeError(f"unresolved eligible AEB reviews: {len(unresolved_aeb)}")

    effective = stat(r["effective_constant_deceleration_distance_mps2"] for r in aeb)
    peak = stat(r["peak_deceleration_mps2"] for r in aeb)
    build_up = stat(r["time_onset_to_peak_s"] for r in aeb)
    margin = stat(r["margin_time_s"] for r in aeb)
    aeb_counts = Counter(r["intervention_class"] for r in aeb)
    fcw_braking = Counter(r["braking_interpretation"] for r in fcw)

    config = {
        "version": "abd_proxy_calibrated_v2",
        "created": "2026-09-12",
        "purpose": "ABD-supported proxy calibration evidence for the split AEB response model",
        "current_scenario_lab_compatible": False,
        "compatibility_blocker": (
            "scenario_lab.response_delay currently affects both stopping-trigger preview "
            "and the brake queue; v2 requires separate controller_trigger_margin and "
            "aeb_actuation_delay fields before use"
        ),
        "eligible_unique_aeb_runs": len(aeb),
        "vehicles": len({r["vehicle"] for r in aeb}),
        "scenarios": sorted({r["scenario"] for r in aeb}),
        "parameters": {
            "brake_deceleration": {
                "status": "abd_supported_observed_response_envelope",
                "dist": "uniform_sensitivity_envelope",
                "low": effective["min"], "high": effective["max"],
                "summary_distance_equivalent_mps2": effective,
                "peak_deceleration_diagnostic_signed_mps2": peak,
            },
            "aeb_actuation_delay": {
                "status": "operator_engineering_prior_not_observed_ecu_signal",
                "dist": "uniform",
                "low": 0.15, "nominal": 0.25, "high": 0.35,
                "unit": "s",
                "mapping": "T_AEB_proxy = T_dec_03 - aeb_actuation_delay",
            },
            "brake_build_up": {
                "status": "abd_supported_observed_onset_to_peak_diagnostic",
                "summary_s": build_up,
                "note": "Retained as an empirical diagnostic pending a ramp-model fit.",
            },
            "controller_trigger_margin": {
                "status": "not_identified_separately_from_geometry_and_trigger_policy",
                "observed_margin_time_diagnostic_s": margin,
            },
        },
        "fcw": {
            "observed_audio_edges": len(fcw),
            "test_intent": "fcw_only_no_aeb",
            "aeb_linkage_eligible_runs": 0,
            "braking_interpretation_counts": dict(fcw_braking),
        },
        "provenance": {
            "aeb_activation": "engineering-prior proxy, never an observed ECU signal",
            "fcw_activation": "AVAD3-observed audible warning edge",
            "driver_intervention": "operator review of Robot Controller paths and velocity traces",
        },
        "source_runs": [{"run": r["run"], "sha256": r["sha256"],
                         "intervention_class": r["intervention_class"]} for r in aeb],
    }
    summary = {
        "aeb_review_rows_including_anchors": len(aeb) + len(duplicates),
        "aeb_unique_eligible_runs": len(aeb),
        "aeb_duplicate_files_removed": len(duplicates),
        "aeb_intervention_classes": dict(aeb_counts),
        "aeb_vehicle_count": len({r["vehicle"] for r in aeb}),
        "aeb_scenario_count": len({r["scenario"] for r in aeb}),
        "fcw_observed_audio_runs": len(fcw),
        "fcw_reviewed_manual_after_audio": sum(
            r["intervention_class"].startswith("manual_after_fcw") for r in fcw),
        "fcw_braking_interpretation_counts": dict(fcw_braking),
        "fcw_runs_eligible_for_aeb_response_or_proxy": 0,
        "brake_deceleration_distance_equivalent_mps2": effective,
        "peak_deceleration_mps2": peak,
        "brake_build_up_onset_to_peak_s": build_up,
    }

    write_csv(OUT / "reviewed_aeb_runs.csv", aeb)
    write_csv(OUT / "reviewed_fcw_runs.csv", fcw)
    write_csv(OUT / "duplicate_aeb_files.csv", duplicates)
    (OUT / "abd_proxy_calibrated_v2.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# ABD reviewed evidence finalization and proxy calibration v2", "",
        f"- Operator-reviewed AEB response records: **{len(aeb)}** unique runs "
        f"across **{summary['aeb_vehicle_count']}** vehicles and "
        f"**{summary['aeb_scenario_count']}** scenario labels.",
        f"- Review decisions: **{aeb_counts.get('none_confirmed', 0)}** no takeover; "
        f"**{aeb_counts.get('manual_after_aeb_stop', 0)}** manual takeover only after "
        "the AEB stop (analysis censored at first stop).",
        f"- FCW audio edges: **{len(fcw)}** FCW-only runs. The **"
        f"{summary['fcw_reviewed_manual_after_audio']}** BR-zero paired cases were all "
        "confirmed as expected manual braking after the warning.",
        "- No FCW-only run is used for AEB response, AEB proxy timing, or FCW-to-AEB delay.",
        "- `T_AEB_proxy = T_dec_03 - U(0.15, 0.35 s)` is retained as an operator-prior "
        "latent activation proxy, not an observed ECU request/active transition.", "",
        "The v2 evidence package is intentionally blocked from direct simulator use until "
        "the current combined `response_delay` is split into trigger-policy and actuation "
        "delay terms. This prevents the 0.15-0.35 s prior from being counted twice.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
