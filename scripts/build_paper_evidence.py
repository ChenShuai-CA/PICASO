"""Build frozen paper tables and publication figures from completed P2 artifacts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario_lab.pulse_router import paired_bootstrap, portfolio_scores  # noqa: E402


PAPER = ROOT / "paper"
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
P211 = ROOT / "runs/20260912_p211_single_candidate"
P212 = ROOT / "runs/20260912_p212_final_confirmation"
P210 = ROOT / "runs/20260912_p210_confirmatory_router"
P28 = ROOT / "runs/20260912_p28_conditional_router"
BRANCHES = ("single", "dual")
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
    "yellow": "#F0E442", "black": "#000000", "gray": "#777777",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate_ci(condition_scores, rounds=5000, seed=2051):
    values = np.asarray(condition_scores, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (rounds, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), np.quantile(samples, [0.025, 0.975]).tolist()


def method_metrics(data, rows, order, budget, branch):
    select = data["branch"] == branch
    global_indices = np.flatnonzero(select)
    outcomes = data["outcomes"][select]
    valid = data["valid"][select]
    steps = data["steps"][select]
    local_order = order[select]
    condition_scores = portfolio_scores(outcomes, local_order, budget)
    row_map = {
        (int(row["condition_index"]), int(row["head"]), int(row["perturbation"])): row
        for row in rows
    }
    executed_valid = []
    costs = []
    episode_counts = []
    role_invalid = 0
    total_steps = 0
    for local_index, global_index in enumerate(global_indices):
        for perturbation in range(outcomes.shape[2]):
            cost = 0
            attempts = 0
            for head in local_order[local_index, :budget]:
                head = int(head)
                row = row_map[(int(global_index), head, perturbation)]
                attempts += 1
                cost += int(steps[local_index, head, perturbation])
                executed_valid.append(bool(valid[local_index, head, perturbation]))
                role_invalid += sum(
                    reason in ("pedestrian_role", "occluder_role")
                    for reason in row.get("invalid_reasons", []))
                if outcomes[local_index, head, perturbation]:
                    break
            costs.append(cost)
            episode_counts.append(attempts)
            total_steps += cost
    rate, ci = rate_ci(condition_scores, seed=2051 + 100 * budget + (branch == "dual"))
    return condition_scores, {
        "branch": branch,
        "method": "",
        "budget_k": budget,
        "conditions": int(select.sum()),
        "perturbations": int(outcomes.shape[2]),
        "coverage": rate,
        "coverage_ci_low": ci[0],
        "coverage_ci_high": ci[1],
        "mean_candidate_rollouts": float(np.mean(episode_counts)),
        "mean_decision_steps": float(np.mean(costs)),
        "total_decision_steps": int(total_steps),
        "executed_candidate_valid_rate": float(np.mean(executed_valid)),
        "role_invalid": int(role_invalid),
    }


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt_ci(value, low, high):
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def write_markdown_tables(main_rows, budget_rows, increments, failures):
    main = [
        "# Main results", "",
        "| split | branch | N | router-1 | fixed-1 | script | router-fixed [95% CI] | router-script [95% CI] | permutation delta (p) | valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        main.append(
            f"| {row['split']} | {row['branch']} | {row['conditions']} | "
            f"{row['router1_rate']:.3f} | {row['fixed1_rate']:.3f} | "
            f"{row['script_rate']:.3f} | "
            f"{fmt_ci(row['router1_minus_fixed1'], row['fixed_ci_low'], row['fixed_ci_high'])} | "
            f"{fmt_ci(row['router1_minus_script'], row['script_ci_low'], row['script_ci_high'])} | "
            f"{row['permutation_delta']:.3f} ({row['permutation_p']:.4f}) | "
            f"{row['candidate_valid_rate']:.3f} |")
    (TABLES / "table1_main_results.md").write_text("\n".join(main) + "\n", encoding="utf-8")

    budget = [
        "# K=1/K=2 budget curve on the P2.11 independent screen", "",
        "| branch | method | K | coverage [95% CI] | mean rollouts | mean decision steps | valid |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in budget_rows:
        budget.append(
            f"| {row['branch']} | {row['method']} | {row['budget_k']} | "
            f"{fmt_ci(row['coverage'], row['coverage_ci_low'], row['coverage_ci_high'])} | "
            f"{row['mean_candidate_rollouts']:.3f} | {row['mean_decision_steps']:.1f} | "
            f"{row['executed_candidate_valid_rate']:.3f} |")
    budget += ["", "Paired K=2 minus K=1 increments:", "",
               "| branch | method | coverage gain [95% CI] | extra rollouts | extra steps | gain per extra rollout |",
               "|---|---|---:|---:|---:|---:|"]
    for row in increments:
        budget.append(
            f"| {row['branch']} | {row['method']} | "
            f"{fmt_ci(row['coverage_gain'], row['gain_ci_low'], row['gain_ci_high'])} | "
            f"{row['extra_candidate_rollouts']:.3f} | {row['extra_decision_steps']:.1f} | "
            f"{row['coverage_gain_per_extra_rollout']:.3f} |")
    (TABLES / "table2_budget_curve.md").write_text("\n".join(budget) + "\n", encoding="utf-8")

    failure = [
        "# Mechanism ablation and failed-route audit", "",
        "| stage | tested mechanism | single result | dual result | gate | inference |",
        "|---|---|---|---|---|---|",
    ]
    for row in failures:
        failure.append(
            f"| {row['stage']} | {row['mechanism']} | {row['single']} | "
            f"{row['dual']} | {row['gate']} | {row['inference']} |")
    (TABLES / "table3_failed_routes.md").write_text("\n".join(failure) + "\n", encoding="utf-8")


def configure_plotting():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "scenario-paper-evidence-v1",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def export_figure(fig, stem):
    fixed_date = datetime(2026, 9, 12, tzinfo=timezone.utc)
    fig.savefig(FIGURES / f"{stem}.pdf", metadata={
        "Creator": "Scenario Generation Research",
        "CreationDate": fixed_date,
        "ModDate": fixed_date,
    })
    svg_path = FIGURES / f"{stem}.svg"
    fig.savefig(svg_path, metadata={
        "Creator": "Scenario Generation Research", "Date": "2026-09-12",
    })
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n", encoding="utf-8")
    fig.savefig(FIGURES / f"{stem}.png", dpi=400, metadata={
        "Software": "Scenario Generation Research", "Creation Time": "2026-09-12",
    })
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, color, fontsize=8, lw=1.1):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=color, edgecolor="#333333", linewidth=lw)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, linespacing=1.25)
    return patch


def arrow(ax, start, end, color="#444444"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.1, color=color))


def plot_graphical_abstract(main_rows):
    held = {row["branch"]: row for row in main_rows if row["split"] == "Heldout"}
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(6, 4.72, "Conditional safety-critical scenario generation with one-candidate selection",
            ha="center", va="center", fontsize=12, fontweight="bold")

    boxes = [
        ((0.15, 1.8), 1.85, "Scenario\nSingle: ego +\npedestrian\nDual: + occluder", "#EAF3F8"),
        ((2.35, 1.8), 1.75, "Observable\nprefix\n5 visible frames\n+ masks", "#E6F5EE"),
        ((4.45, 1.8), 1.65, "Frozen library\n4 pulse\nprototypes", "#FFF4D6"),
        ((6.45, 1.8), 1.65, "Ridge router\ncondition-specific\nranking", "#F4EAF7"),
        ((8.45, 1.8), 1.35, "Execute\ntop-1\nK = 1", "#FBE9E4"),
        ((10.15, 1.8), 1.65, "Valid-dangerous\ncoverage", "#E7F4EA"),
    ]
    for xy, width, label, color in boxes:
        rounded_box(ax, xy, width, 1.7, label, color, fontsize=6.3)
    for start, end in ((2.03, 2.31), (4.13, 4.41), (6.13, 6.41),
                       (8.13, 8.41), (9.83, 10.11)):
        arrow(ax, (start, 2.65), (end, 2.65))
    ax.text(6, 1.15, "One-time heldout improvement over the strongest fixed single prototype",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.text(4.35, 0.55,
            f"Single  +{100 * held['single']['router1_minus_fixed1']:.1f} pp  "
            f"[{100 * held['single']['fixed_ci_low']:.1f}, {100 * held['single']['fixed_ci_high']:.1f}]",
            ha="center", fontsize=9, color=OKABE_ITO["blue"])
    ax.text(8.0, 0.55,
            f"Dual  +{100 * held['dual']['router1_minus_fixed1']:.1f} pp  "
            f"[{100 * held['dual']['fixed_ci_low']:.1f}, {100 * held['dual']['fixed_ci_high']:.1f}]",
            ha="center", fontsize=9, color=OKABE_ITO["vermillion"])
    ax.text(6, 0.08, "95% condition-bootstrap intervals; numerical sensitivity domain",
            ha="center", fontsize=7, color="#555555")
    export_figure(fig, "graphical_abstract")


def plot_method_overview():
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(0.2, 7.55, "A  Offline library construction", fontsize=10, fontweight="bold")
    ax.text(0.2, 3.75, "B  Frozen conditional inference", fontsize=10, fontweight="bold")
    train_boxes = [
        (0.3, "Script-safe\nconditions +\nlane-locked CEM", "#EAF3F8"),
        (3.1, "Multi-success\npulse sets", "#FFF4D6"),
        (5.9, "Four-head\nset predictor", "#F4EAF7"),
        (8.7, "Branch medians\n→ frozen\nlibrary", "#E6F5EE"),
    ]
    for x, text, color in train_boxes:
        rounded_box(ax, (x, 5.4), 2.2, 1.25, text, color, fontsize=6.8)
    for x in (2.53, 5.33, 8.13):
        arrow(ax, (x, 6.02), (x + 0.5, 6.02))
    ax.text(1.4, 5.08, "CEM interaction cost is paid offline", fontsize=6.8,
            color=OKABE_ITO["vermillion"], ha="center")

    infer_boxes = [
        (0.3, "New single/dual\ncondition", "#EAF3F8"),
        (3.1, "Five actor-visible\nframes\n+ masks", "#E6F5EE"),
        (5.9, "Frozen ridge router\nranks four\nprototypes", "#F4EAF7"),
        (8.7, "Execute top K\nstop at first\nsuccess", "#FBE9E4"),
    ]
    for x, text, color in infer_boxes:
        rounded_box(ax, (x, 1.9), 2.2, 1.25, text, color, fontsize=6.8)
    for x in (2.53, 5.33, 8.13):
        arrow(ax, (x, 2.52), (x + 0.5, 2.52))
    arrow(ax, (9.8, 5.35), (9.8, 3.22), color=OKABE_ITO["green"])
    ax.text(10.0, 4.25, "freeze", rotation=90, va="center", fontsize=7,
            color=OKABE_ITO["green"])
    ax.text(6, 1.1,
            "Primary metric: condition-level P(complete ∧ valid ∧ dangerous) across paired perturbations",
            ha="center", fontsize=8)
    ax.text(6, 0.55,
            "K=1 is the heldout-confirmed final method; K=2 is a development-only budget extension",
            ha="center", fontsize=8, color="#555555")
    export_figure(fig, "fig1_method_overview")


def plot_budget_curve(rows, increments):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    styles = {
        "router": (OKABE_ITO["blue"], "o", "-"),
        "fixed": (OKABE_ITO["orange"], "s", "--"),
    }
    for ax, branch, panel in zip(axes, BRANCHES, ("A", "B")):
        local = [row for row in rows if row["branch"] == branch]
        for method in ("router", "fixed"):
            selected = sorted((row for row in local if row["method"] == method),
                              key=lambda row: row["budget_k"])
            x = np.array([row["mean_candidate_rollouts"] for row in selected])
            y = np.array([row["coverage"] for row in selected])
            low = y - np.array([row["coverage_ci_low"] for row in selected])
            high = np.array([row["coverage_ci_high"] for row in selected]) - y
            color, marker, line = styles[method]
            ax.errorbar(x, y, yerr=np.vstack((low, high)), color=color, marker=marker,
                        linestyle=line, capsize=3, linewidth=1.4, markersize=5,
                        label="Conditional router" if method == "router" else "Fixed order")
            for row in selected:
                yoff = 8 if method == "router" else -24
                ax.annotate(f"K={row['budget_k']}\n{row['mean_decision_steps']:.0f} steps",
                            (row["mean_candidate_rollouts"], row["coverage"]),
                            xytext=(5, yoff), textcoords="offset points", fontsize=6.5)
        inc = next(row for row in increments if row["branch"] == branch and row["method"] == "router")
        ax.text(0.38, 0.94,
                f"Router K2−K1 = {100 * inc['coverage_gain']:.1f} pp\n"
                f"95% CI [{100 * inc['gain_ci_low']:.1f}, {100 * inc['gain_ci_high']:.1f}]",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=.25", facecolor="white", edgecolor="#BBBBBB"))
        ax.set_title(f"{panel}  {branch.capitalize()} branch")
        ax.set_xlabel("Mean candidate rollouts per perturbation")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.94, 1.86)
    axes[0].set_ylabel("Valid-dangerous condition coverage")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Development budget curve on the P2.11 independent screen", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    export_figure(fig, "fig2_budget_curve")


def plot_main_forest(main_rows):
    order = [("Heldout", "dual"), ("Heldout", "single"),
             ("Independent screen", "dual"), ("Independent screen", "single")]
    rows = [next(row for row in main_rows if row["split"] == split and row["branch"] == branch)
            for split, branch in order]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    y = np.arange(len(rows))
    colors = [OKABE_ITO["vermillion"] if row["branch"] == "dual" else OKABE_ITO["blue"]
              for row in rows]
    values = np.array([100 * row["router1_minus_fixed1"] for row in rows])
    low = values - np.array([100 * row["fixed_ci_low"] for row in rows])
    high = np.array([100 * row["fixed_ci_high"] for row in rows]) - values
    for index, row in enumerate(rows):
        filled = row["split"] == "Heldout"
        ax.errorbar(values[index], y[index], xerr=[[low[index]], [high[index]]],
                    fmt="o", color=colors[index], markerfacecolor=colors[index] if filled else "white",
                    markersize=6, capsize=3, linewidth=1.4)
        ax.text(max(8.3, 100 * row["fixed_ci_high"] + 0.3), y[index],
                f"N={row['conditions']}", va="center", fontsize=7)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, [f"{row['split']} · {row['branch']}" for row in rows])
    ax.set_xlabel("Router-1 minus fixed-1 coverage (percentage points)")
    ax.set_title("Condition-bootstrap effects across independent confirmation stages")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    export_figure(fig, "fig3_main_effects")


def plot_failure_routes():
    route_rows = [
        ("P2.1 no-prior robust", 0.000, 0.000, 0.000, 0.000, -0.020, 0.030, False),
        ("P2.2 mixed MAPPO", 0.040, 0.000, 0.1051, -0.005, -0.075, 0.060, False),
        ("P2.3 equal-branch MAPPO", 0.000, -0.100, 0.090, 0.005, -0.090, 0.095, False),
        ("P2.5 BC→MAPPO", 0.055, -0.110, 0.205, -0.010, -0.110, 0.090, False),
        ("P2.7 multi-solution K=4", 0.270, 0.100, 0.440, 0.200, 0.095, 0.325, True),
        ("P2.11 router K=1", 0.352, 0.295, 0.410, 0.230, 0.178, 0.284, True),
        ("P2.12 heldout router K=1", 0.368, 0.310, 0.428, 0.181, 0.135, 0.230, True),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(route_rows))
    offsets = {"single": -0.13, "dual": 0.13}
    for branch, color in (("single", OKABE_ITO["blue"]),
                          ("dual", OKABE_ITO["vermillion"])):
        if branch == "single":
            vals = np.array([row[1] for row in route_rows])
            lows = np.array([row[2] for row in route_rows])
            highs = np.array([row[3] for row in route_rows])
        else:
            vals = np.array([row[4] for row in route_rows])
            lows = np.array([row[5] for row in route_rows])
            highs = np.array([row[6] for row in route_rows])
        ax.errorbar(vals, y + offsets[branch], xerr=np.vstack((vals - lows, highs - vals)),
                    fmt="o", color=color, capsize=2.5, markersize=4.5, linewidth=1.1,
                    label=branch.capitalize())
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, [row[0] for row in route_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Coverage difference versus script")
    ax.set_title("Failed-route audit and mechanism transition")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.19, 0.03,
            "P2.7 uses K=4 and is not an equal-budget comparison; P2.11/P2.12 use K=1.",
            fontsize=6.8, color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    export_figure(fig, "fig4_failed_route_audit")


def main():
    for path in (EVIDENCE, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    p211_data = load_npz(P211 / "screen_dataset.npz")
    p211_rows = load_jsonl(P211 / "candidate_attempts.jsonl")
    p28_summary = read_json(P28 / "summary.json")
    fixed_by_branch = {
        row["branch"]: np.asarray(row["fixed_order"], dtype=int)
        for row in p28_summary["screen_results"]}
    router_order = p211_data["router_order"].astype(int)
    fixed_order = np.zeros_like(router_order)
    for branch in BRANCHES:
        fixed_order[p211_data["branch"] == branch] = fixed_by_branch[branch]

    budget_rows = []
    score_map = {}
    for branch in BRANCHES:
        for method, order in (("router", router_order), ("fixed", fixed_order)):
            for budget in (1, 2):
                scores, report = method_metrics(
                    p211_data, p211_rows, order, budget, branch)
                report["method"] = method
                budget_rows.append(report)
                score_map[(branch, method, budget)] = scores
    increments = []
    for branch in BRANCHES:
        for method in ("router", "fixed"):
            one = next(row for row in budget_rows
                       if row["branch"] == branch and row["method"] == method and row["budget_k"] == 1)
            two = next(row for row in budget_rows
                       if row["branch"] == branch and row["method"] == method and row["budget_k"] == 2)
            delta, ci = paired_bootstrap(
                score_map[(branch, method, 2)], score_map[(branch, method, 1)],
                rounds=5000, seed=2061 + (branch == "dual") + 10 * (method == "fixed"))
            extra_rollouts = two["mean_candidate_rollouts"] - one["mean_candidate_rollouts"]
            increments.append({
                "branch": branch, "method": method,
                "coverage_gain": delta, "gain_ci_low": ci[0], "gain_ci_high": ci[1],
                "extra_candidate_rollouts": extra_rollouts,
                "extra_decision_steps": two["mean_decision_steps"] - one["mean_decision_steps"],
                "coverage_gain_per_extra_rollout": delta / extra_rollouts,
            })

    main_rows = []
    for label, path in (("Independent screen", P211 / "summary.json"),
                        ("Heldout", P212 / "summary.json")):
        for row in read_json(path)["results"]:
            main_rows.append({
                "split": label, "branch": row["branch"], "conditions": row["conditions"],
                "router1_rate": row["router1_rate"], "fixed1_rate": row["fixed1_rate"],
                "script_rate": row["script_rate"],
                "router1_minus_fixed1": row["router1_minus_fixed1"],
                "fixed_ci_low": row["router1_minus_fixed1_ci95"][0],
                "fixed_ci_high": row["router1_minus_fixed1_ci95"][1],
                "router1_minus_script": row["router1_minus_script"],
                "script_ci_low": row["router1_minus_script_ci95"][0],
                "script_ci_high": row["router1_minus_script_ci95"][1],
                "permutation_delta": row["router1_minus_permutation_mean"],
                "permutation_p": row["permutation_p_one_sided"],
                "candidate_valid_rate": row["candidate_valid_rate"],
                "role_invalid": row["role_invalid"], "pass": row["pass"],
            })

    p210 = read_json(P210 / "summary.json")
    k2_rows = []
    for split, rows in (("Independent screen", p210["screen_results"]),
                        ("Fresh development", p210["development_results"])):
        for row in rows:
            k2_rows.append({
                "split": split, "branch": row["branch"], "conditions": row["conditions"],
                "router2_rate": row["router2"]["rate"],
                "fixed2_rate": row["fixed2"]["rate"],
                "router2_minus_fixed2": row["router2_minus_fixed2"],
                "ci_low": row["router2_minus_fixed2_ci95"][0],
                "ci_high": row["router2_minus_fixed2_ci95"][1],
                "mean_decision_steps": row["router2"]["mean_steps_until_success_or_exhaustion"],
                "candidate_valid_rate": row["router2"]["candidate_valid_rate"],
                "role_invalid": row["router2"]["role_invalid"],
                "permutation_delta": row["router2_minus_permutation_mean"],
                "permutation_p": row["permutation_p_one_sided"],
            })

    failures = [
        {"stage": "P2/P2.1", "mechanism": "PPO/MAPPO with prior and robust variants",
         "single": "no reliable gain; best stable no-prior variant Δ=0.000",
         "dual": "prior variants unstable; best stable no-prior variant Δ≈0",
         "gate": "FAIL", "inference": "More training under the same recipe was unsupported."},
        {"stage": "P2.2/P2.3", "mechanism": "Shared/specialized networks and equal branch budget",
         "single": "Δ=0.000 to 0.040; CIs include zero",
         "dual": "Δ=-0.005 to 0.005; CIs include zero",
         "gate": "FAIL superiority", "inference": "Architecture and exposure were not the main bottleneck."},
        {"stage": "P2.4", "mechanism": "Lane-locked per-condition CEM feasibility",
         "single": "parameter Δ=0.440; trajectory Δ=0.595",
         "dual": "parameter Δ=0.335; trajectory Δ=0.530",
         "gate": "PASS feasibility", "inference": "Solutions existed, but required many simulator interactions."},
        {"stage": "P2.5", "mechanism": "Single-teacher BC followed by MAPPO",
         "single": "Δ=0.055 [-0.110, 0.205]",
         "dual": "Δ=-0.010 [-0.110, 0.090]",
         "gate": "FAIL", "inference": "Open-loop teachers did not transfer reliably to closed-loop execution."},
        {"stage": "P2.6", "mechanism": "Direct condition-to-pulse parameter prediction",
         "single": "rate 0.269; required >0.50",
         "dual": "rate 0.358; required >0.25",
         "gate": "FAIL both-branch", "inference": "Single-output regression discarded solution multimodality."},
        {"stage": "P2.7", "mechanism": "Multi-success set supervision, K=4",
         "single": "dev 0.520 vs script 0.250",
         "dual": "dev 0.550 vs script 0.350",
         "gate": "PASS", "inference": "A compact executable solution set restored coverage at higher cost."},
        {"stage": "P2.8", "mechanism": "Conditional ridge routing, K=2",
         "single": "router-fixed Δ=0.041 [0.012, 0.078]",
         "dual": "router-fixed passed; shuffled-control CI crossed zero",
         "gate": "FAIL mechanism", "inference": "Weak alignment required an independent confirmation set."},
        {"stage": "P2.10–P2.12", "mechanism": "Frozen conditional router, then K=1 confirmation",
         "single": "heldout Δ=0.041 [0.012, 0.071]",
         "dual": "heldout Δ=0.029 [0.001, 0.058]",
         "gate": "PASS", "inference": "Condition-dependent top-1 selection was sufficient."},
    ]

    write_csv(EVIDENCE / "main_results.csv", main_rows)
    write_csv(EVIDENCE / "budget_curve.csv", budget_rows)
    write_csv(EVIDENCE / "budget_increments.csv", increments)
    write_csv(EVIDENCE / "k2_development_confirmation.csv", k2_rows)
    write_csv(EVIDENCE / "failed_route_audit.csv", failures)
    write_markdown_tables(main_rows, budget_rows, increments, failures)

    plot_graphical_abstract(main_rows)
    plot_method_overview()
    plot_budget_curve(budget_rows, increments)
    plot_main_forest(main_rows)
    plot_failure_routes()

    input_paths = [
        P211 / "summary.json", P211 / "screen_dataset.npz",
        P211 / "candidate_attempts.jsonl", P212 / "summary.json",
        P210 / "summary.json", P28 / "summary.json",
    ]
    outputs = sorted([
        path for folder in (EVIDENCE, FIGURES, TABLES) for path in folder.iterdir()
        if path.name != "evidence_manifest.json"
    ])
    manifest = {
        "protocol": "paper evidence freeze v1",
        "scope": "read-only synthesis of completed P2 artifacts; no training or simulation",
        "rq3_primary_split": "P2.11 independent screen",
        "p212_k2_policy": "not computed as confirmatory evidence after heldout unblinding",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                   for path in input_paths],
        "outputs": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                     "sha256": sha256(path)} for path in outputs],
        "heldout_rerun": False,
    }
    (EVIDENCE / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "main_results": main_rows,
        "budget_increments": increments,
        "figures": [path.name for path in sorted(FIGURES.iterdir())],
        "heldout_rerun": False,
    }, indent=2))


if __name__ == "__main__":
    main()
