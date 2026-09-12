import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper/evidence"


def read_csv(name):
    with (EVIDENCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_budget_k1_matches_independent_screen_summary():
    budget = read_csv("budget_curve.csv")
    summary = json.loads(
        (ROOT / "runs/20260912_p211_single_candidate/summary.json").read_text(
            encoding="utf-8"
        )
    )
    by_branch = {row["branch"]: row for row in summary["results"]}
    for branch in ("single", "dual"):
        router = next(
            row for row in budget
            if row["branch"] == branch
            and row["method"] == "router"
            and row["budget_k"] == "1"
        )
        fixed = next(
            row for row in budget
            if row["branch"] == branch
            and row["method"] == "fixed"
            and row["budget_k"] == "1"
        )
        assert float(router["coverage"]) == pytest.approx(
            by_branch[branch]["router1_rate"]
        )
        assert float(fixed["coverage"]) == pytest.approx(
            by_branch[branch]["fixed1_rate"]
        )


def test_budget_increment_is_paired_and_monotone():
    budget = read_csv("budget_curve.csv")
    increments = read_csv("budget_increments.csv")
    lookup = {
        (row["branch"], row["method"], int(row["budget_k"])): row
        for row in budget
    }
    for row in increments:
        key = (row["branch"], row["method"])
        one = lookup[key + (1,)]
        two = lookup[key + (2,)]
        observed = float(two["coverage"]) - float(one["coverage"])
        assert observed >= 0
        assert float(row["coverage_gain"]) == pytest.approx(observed)
        assert float(row["extra_candidate_rollouts"]) > 0
        assert float(row["extra_decision_steps"]) > 0


def test_manifest_keeps_heldout_k2_out_of_confirmatory_evidence():
    manifest = json.loads(
        (EVIDENCE / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["heldout_rerun"] is False
    assert "not computed as confirmatory" in manifest["p212_k2_policy"]
    input_paths = {row["path"] for row in manifest["inputs"]}
    assert "runs/20260912_p212_final_confirmation/summary.json" in input_paths
    assert all("screen_dataset.npz" not in path or "p212" not in path
               for path in input_paths)
