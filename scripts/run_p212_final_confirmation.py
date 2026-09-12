"""Run the explicitly authorized, one-time P2.12 heldout confirmation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_p211_single_candidate as p211  # noqa: E402


BRANCHES = ("single", "dual")
HELDOUT_SEED = 77000
COUNT_PER_BRANCH = 360
PERTURBATIONS = 5


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_inputs(prereg):
    for item in prereg["frozen_inputs"].values():
        path = ROOT / item["path"]
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {item['path']}")
    for relative, expected in prereg["frozen_source_hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")


def ensure_unopened_preflight(output):
    result_names = (
        "attempt_started.json", "screen_dataset.npz", "screen_eligible.json",
        "candidate_attempts.jsonl", "script_attempts.jsonl", "summary.json",
        "REPORT.md", "completed.json", "condition_fingerprint_audit.json",
    )
    present = [name for name in result_names if (output / name).exists()]
    condition_path = output / "conditions/final_confirmation_seed77000.json"
    if condition_path.exists():
        present.append(str(condition_path.relative_to(output)))
    if present:
        raise RuntimeError(
            "P2.12 artifacts already exist; inspect authorization/audit state: "
            + ", ".join(present))


def write_report(path, results, passed):
    lines = [
        "# P2.12 one-time heldout confirmation", "",
        "The frozen P2.11 router selects one prototype from the first five actor-visible frames "
        "and executes one candidate episode. The perturbations are a numerical sensitivity "
        "domain, not an AEB-calibrated distribution.", "",
        "| branch | N | router-1 | fixed-1 | script | router1-fixed1 [95% CI] | "
        "router1-script [95% CI] | perm delta | perm p | valid | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['branch']} | {row['conditions']} | {row['router1_rate']:.3f} | "
            f"{row['fixed1_rate']:.3f} | {row['script_rate']:.3f} | "
            f"{p211.fmt(row['router1_minus_fixed1'], row['router1_minus_fixed1_ci95'])} | "
            f"{p211.fmt(row['router1_minus_script'], row['router1_minus_script_ci95'])} | "
            f"{row['router1_minus_permutation_mean']:.3f} | "
            f"{row['permutation_p_one_sided']:.4f} | "
            f"{row['candidate_valid_rate']:.3f} | {row['pass']} |")
    lines += [
        "", f"Both-branch final gate: **{passed}**.", "",
        "This is the sole final heldout attempt under the frozen P2.12 protocol. Its result is "
        "reported as observed and does not trigger model, budget, seed, denominator, or gate changes.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="runs/20260912_p212_final_confirmation")
    parser.add_argument(
        "--authorize-one-time-heldout", action="store_true",
        help="confirm explicit user authorization for the irreversible heldout attempt")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("run inside WSL2 Ubuntu")

    output = (ROOT / args.output).resolve()
    prereg_path = output / "preregistration.json"
    prereg = read_json(prereg_path)
    if not prereg.get("created_before_p212_conditions_and_outcomes"):
        raise ValueError("P2.12 preregistration is not marked pre-outcome")
    if prereg["heldout"]["seed"] != HELDOUT_SEED:
        raise ValueError("heldout seed differs from the frozen script")
    verify_frozen_inputs(prereg)

    if not args.authorize_one_time_heldout:
        ensure_unopened_preflight(output)
        print(json.dumps({
            "preflight": "ready",
            "frozen_inputs_verified": True,
            "heldout_seed": HELDOUT_SEED,
            "heldout_generated_or_read": False,
            "next_required_flag": "--authorize-one-time-heldout",
        }, indent=2))
        return

    completed_path = output / "completed.json"
    if completed_path.exists():
        raise RuntimeError("P2.12 already completed; rerun is forbidden")
    marker_path = output / "attempt_started.json"
    prereg_hash = file_hash(prereg_path)
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["preregistration_sha256"] != prereg_hash:
            raise RuntimeError("preregistration changed after the heldout attempt started")
    else:
        marker_path.write_text(json.dumps({
            "protocol": prereg["protocol"],
            "preregistration_sha256": prereg_hash,
            "authorized_cli_flag_received": True,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "started_or_resumable",
        }, indent=2), encoding="utf-8")

    from scenario_lab.sampling import export_conditions, load_conditions
    from scenario_lab.teacher import spec_fingerprint
    from scenario_lab.train import load_perturb_config

    frozen = prereg["frozen_inputs"]
    library_file = read_json(ROOT / frozen["prototype_library"]["path"])
    library = {
        branch: np.asarray(library_file["parameters"][branch], dtype=np.float32)
        for branch in BRANCHES}
    models = {
        branch: p211.load_router(ROOT / frozen[f"{branch}_ridge"]["path"])
        for branch in BRANCHES}
    perturb_config = load_perturb_config(
        ROOT / frozen["perturbation_config"]["path"])

    condition_dir = output / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    heldout_path = condition_dir / "final_confirmation_seed77000.json"
    if not heldout_path.exists():
        export_conditions(
            heldout_path, HELDOUT_SEED, COUNT_PER_BRANCH, BRANCHES,
            sampler_version=2, role="reference", purpose="heldout")

    prior_paths = [
        ROOT / "runs/20260912_p28_conditional_router/conditions/training_seed72000.json",
        ROOT / "runs/20260912_p28_conditional_router/conditions/screen_seed73000.json",
        ROOT / "runs/20260912_p28_conditional_router/conditions/development_seed74000.json",
        ROOT / "runs/20260912_p210_confirmatory_router/conditions/confirmatory_screen_seed75000.json",
        ROOT / "runs/20260912_p211_single_candidate/conditions/single_candidate_screen_seed76000.json",
    ]
    prior_fingerprints = set()
    prior_counts = {}
    for path in prior_paths:
        specs, _ = load_conditions(path)
        fingerprints = {spec_fingerprint(spec) for spec in specs}
        prior_counts[path.name] = len(fingerprints)
        prior_fingerprints.update(fingerprints)
    heldout_specs, heldout_header = load_conditions(heldout_path)
    heldout_fingerprints = {spec_fingerprint(spec) for spec in heldout_specs}
    overlap = len(heldout_fingerprints & prior_fingerprints)
    if overlap:
        raise ValueError("P2.12 heldout overlaps a prior development condition set")

    p211.SCREEN_SEED = HELDOUT_SEED
    p211.COUNT_PER_BRANCH = COUNT_PER_BRANCH
    data, candidate_rows, script_rows, manifest = p211.prepare_screen(
        heldout_path, library, models, perturb_config, output)
    results = p211.evaluate(data, candidate_rows, prereg)
    passed = all(row["pass"] for row in results)

    summary = {
        "protocol": prereg["protocol"],
        "preregistration_sha256": prereg_hash,
        "paper_scope": prereg["paper_scope"],
        "frozen_input_hashes_verified": True,
        "heldout_manifest": p211.compact_manifest(manifest),
        "results": results,
        "both_branch_pass": passed,
        "final_confirmation": "PASS" if passed else "FAIL",
        "preferred_final_budget": 1,
        "method_changed_after_heldout": False,
        "heldout_read": True,
    }
    audit = {
        "prior_condition_counts": prior_counts,
        "heldout_condition_set_version": heldout_header["condition_set_version"],
        "heldout_conditions": len(heldout_fingerprints),
        "overlap_with_all_prior_development_conditions": overlap,
        "legacy_seed41000_read": False,
        "one_time_attempt_marker_sha256": file_hash(marker_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (output / "condition_fingerprint_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    write_report(output / "REPORT.md", results, passed)

    raw = [
        heldout_path, output / "screen_dataset.npz", output / "screen_eligible.json",
        output / "candidate_attempts.jsonl", output / "script_attempts.jsonl"]
    (output / "artifact_manifest.json").write_text(json.dumps({
        "raw_artifacts": [{
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        } for path in raw],
        "large_attempt_jsonl_committed": False,
    }, indent=2), encoding="utf-8")
    completed_path.write_text(json.dumps({
        "protocol": prereg["protocol"],
        "final_confirmation": "PASS" if passed else "FAIL",
        "both_branch_pass": passed,
        "preferred_final_budget": 1,
        "method_changed_after_heldout": False,
        "heldout_read": True,
        "rerun_forbidden": True,
    }, indent=2), encoding="utf-8")
    marker = read_json(marker_path)
    marker.update(status="completed", completed_sha256=file_hash(completed_path))
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(json.dumps({
        "final_confirmation": "PASS" if passed else "FAIL",
        "results": [{
            "branch": row["branch"],
            "conditions": row["conditions"],
            "router1": row["router1_rate"],
            "fixed1": row["fixed1_rate"],
            "script": row["script_rate"],
            "pass": row["pass"],
        } for row in results],
        "legacy_seed41000_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
