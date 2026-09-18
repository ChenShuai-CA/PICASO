"""P3.3.4-R1 gate table: R0 bitwise recheck (orig/proj/C) + C-v2 SPEC_R1 S6 gates.

Run: python research_tasks/p334_r1_gates.py
Reads runs/20260917_p334_r1_eval/METRICS_V2.json (R1 four-arm) and
runs/20260916_p334_review_closure/METRICS_V2.json (R0 sealed three-arm).
Emits a compact text report; no files modified.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R1 = REPO / "runs/20260917_p334_r1_eval/METRICS_V2.json"
R0 = REPO / "runs/20260916_p334_review_closure/METRICS_V2.json"


def diff_tree(a, b, path=""):
    """Yield (path, a_val, b_val) for every leaf mismatch; a is the reference."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a:
            if key not in b:
                yield (f"{path}.{key}", a[key], "<missing>")
                continue
            yield from diff_tree(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (path, f"len={len(a)}", f"len={len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from diff_tree(x, y, f"{path}[{i}]")
    elif a != b:
        yield (path, a, b)


def macro(group_level, key):
    return sum(group_level[s][key] for s in group_level) / len(group_level)


def fmt_pct(x):
    return f"{100 * x:+.2f}%"


def main() -> None:
    r1 = json.loads(R1.read_text(encoding="utf-8"))
    r0 = json.loads(R0.read_text(encoding="utf-8"))

    print("=" * 72)
    print("1. R0 BITWISE RECHECK (keys present in R0, arms orig/proj/C)")
    print("=" * 72)
    total_diffs = 0
    for arm in ("orig", "proj", "C"):
        diffs = list(diff_tree(r0["arms"][arm], r1["arms"][arm], arm))
        added = sorted(set(r1["arms"][arm]) - set(r0["arms"][arm]))
        total_diffs += len(diffs)
        print(f"  {arm:5s}: {len(diffs)} mismatched leaves"
              + (f"  FIRST: {diffs[0][0]} {diffs[0][1]} != {diffs[0][2]}"
                 if diffs else "")
              + (f"  | R1-added keys: {added}" if added else ""))
    for key in ("samples", "outlier_samples_excluded_in_clean_stratum",
                "sampling", "solver_status"):
        if key in r0:
            diffs = list(diff_tree(r0[key], r1.get(key), key))
            total_diffs += len(diffs)
            print(f"  {key}: {len(diffs)} mismatches"
                  + (f"  FIRST: {diffs[0][0]}" if diffs else ""))
    for name, blob in (("R0", r0), ("R1", r1)):
        print(f"  sample_id_sorted_sha256 {name}: "
              f"{blob.get('sample_id_sorted_sha256', '<absent>')}")
    print(f"  VERDICT: {'BITWISE IDENTICAL' if total_diffs == 0 else 'DIFFERENCES PRESENT'}")

    print()
    print("=" * 72)
    print("2. C-v2 GATE TABLE (SPEC_R1 section 6, vs orig, full denominator)")
    print("=" * 72)
    orig = r1["arms"]["orig"]["group_level"]
    c2 = r1["arms"]["C-v2"]["group_level"]
    c1 = r0["arms"]["C"]["group_level"]

    def row(label, key, src=c2, ref=orig, extra=None):
        val = macro(src, key)
        base = macro(ref, key)
        print(f"  {label:34s} {val:8.4f} vs {base:8.4f}  {fmt_pct((val - base) / base):>8s}"
              + (f"   [C-v1 {macro(c1, key):.4f} {fmt_pct((macro(c1, key) - base) / base)}]"
                 if extra else ""))

    print("  -- gate 1: error metrics <= +5% vs orig")
    row("macro minADE@6", "min_ade", extra=True)
    for src_name in ("interaction", "waymo"):
        print(f"  {src_name} minADE@6:".ljust(36)
              + f"{c2[src_name]['min_ade']:.4f} vs {orig[src_name]['min_ade']:.4f}"
              + f"  {fmt_pct((c2[src_name]['min_ade'] - orig[src_name]['min_ade']) / orig[src_name]['min_ade']):>8s}"
              + f"   [C-v1 {c1[src_name]['min_ade']:.4f}]")
        print(f"  {src_name} minFDE@6:".ljust(36)
              + f"{c2[src_name]['min_fde']:.4f} vs {orig[src_name]['min_fde']:.4f}"
              + f"  {fmt_pct((c2[src_name]['min_fde'] - orig[src_name]['min_fde']) / orig[src_name]['min_fde']):>8s}"
              + f"   [C-v1 {c1[src_name]['min_fde']:.4f}]")
    row("macro joint minADE@6", "min_ade_joint", extra=True)
    row("macro joint minFDE@6", "min_fde_joint", extra=True)

    print("  -- gate 2: diversity_v2 / orig >= 80%")
    for arm in ("proj", "C", "C-v2"):
        ratio = (r1["arms"][arm]["diversity_v2_mean"]
                 / r1["arms"]["orig"]["diversity_v2_mean"])
        print(f"  {arm:5s} diversity_v2 {r1['arms'][arm]['diversity_v2_mean']:.4f}"
              f"  ratio {ratio:.1%}")

    print("  -- gate 3a: frozen step violations <= GT floor +3pp")
    for arm in ("orig", "proj", "C", "C-v2"):
        worst = 0.0
        detail = []
        worst_jump = 0.0
        for group, meters in r1["arms"][arm]["kinematics_frozen"].items():
            for metric, payload in meters.items():
                if metric == "continuity":
                    worst_jump = max(worst_jump,
                                     payload["implied_accel_over_limit_rate"])
                    continue
                worst = max(worst, payload["step_rate"])
                if payload["step_rate"] > 0:
                    detail.append(f"{group}/{metric}={payload['step_rate']:.2%}")
        print(f"  {arm:5s} worst frozen step_rate {worst:.4%}"
              + ("  " + ", ".join(detail[:6]) if detail else "  (all zero)")
              + f" | worst start-jump {worst_jump:.4%}")

    print("  -- gate 3b: anchored kinematic_valid ratio >= C-v1 99.997%,"
          " no all-invalid scenes")
    for arm in ("C", "C-v2"):
        v = r1["arms"][arm]["validity"]
        att = sum(s["attempted"] for s in v.values())
        kv = sum(s["kinematic_valid"] for s in v.values())
        none_agents = sum(s["agents_none"] for s in v.values())
        print(f"  {arm:5s} kinematic_valid {kv}/{att} = {kv / att:.5%}"
              f"  agents_none(all-6-invalid) {none_agents}"
              f"  scenes_no_valid_agent {r1['arms'][arm]['validity_totals']['scenes_no_valid_agent']}")

    print("  -- gate 4: v0 over-limit accounting (recorded v0, R1)")
    print(f"  {json.dumps(r1['v0_recorded_over_limit'])}")

    print()
    print("=" * 72)
    print("3. SUPPLEMENTARY: per-source minADE + clean stratum (R0 section 5 style)")
    print("=" * 72)
    for arm in ("orig", "proj", "C", "C-v2"):
        gl = r1["arms"][arm]["group_level"]
        clean = r1["arms"][arm]["group_level_clean_stratum"]
        base_clean = macro(clean, "min_ade")
        base_full = macro(gl, "min_ade")
        print(f"  {arm:5s} full {base_full:.4f}  clean {base_clean:.4f}")
    base_clean_orig = macro(r1["arms"]["orig"]["group_level_clean_stratum"], "min_ade")
    for arm in ("C", "C-v2"):
        clean = macro(r1["arms"][arm]["group_level_clean_stratum"], "min_ade")
        print(f"  {arm:5s} clean-stratum vs orig-clean: {fmt_pct((clean - base_clean_orig) / base_clean_orig)}")

    print()
    print("  latency:", json.dumps(r1["latency"])[:400])


if __name__ == "__main__":
    main()
