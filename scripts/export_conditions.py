"""Export an explicit, versioned condition manifest shared by every method."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scenario_lab.sampling import export_conditions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--branches', nargs='+', default=['single', 'dual'], choices=['single', 'dual'])
    p.add_argument('--sampler-version', type=int, default=2, choices=[1, 2])
    p.add_argument('--role', default='reference', choices=['reference', 'stress'])
    p.add_argument('--purpose', default='diagnostic',
                   choices=['training', 'diagnostic', 'development', 'heldout'])
    p.add_argument('--controller', default='stopping', choices=['stopping', 'ttc'])
    a = p.parse_args()
    manifest = export_conditions(a.output, seed=a.seed, count=a.count, branches=tuple(a.branches),
                                 sampler_version=a.sampler_version, role=a.role,
                                 purpose=a.purpose, controller=a.controller)
    infeasible = sum(1 for c in manifest['conditions'] if c['spec']['condition_role'] == 'reference_infeasible')
    print(f"wrote {a.output}: {len(manifest['conditions'])} conditions, "
          f"version={manifest['condition_set_version']}, "
          f"resamples={manifest['resample_counts']}, reference_infeasible={infeasible}")


if __name__ == '__main__':
    main()
