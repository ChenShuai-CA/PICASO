# P3.3.2a independent review (2026-09-13)

Reviewed commit: `88033f7`.

Decision: **ENGINEERING_GO for the bounded 100-shard nominal architecture
experiment**, with the corrections and scope below. This is not G1 acceptance,
model superiority, or approval of a closed-loop actor implementation.

## Evidence independently checked

- Full WSL test rerun: **183 passed in 46.08 s**, no warning in this run.
- All 10 artifacts match the SHA-256 values in ARTIFACT_MANIFEST.json. The two
  independent-process loss CSV files have the same SHA-256. This review did
  not rerun the two full M2 training jobs; it inspected their code and evidence.
- The encoder now builds per-query source keys from per-frame visibility;
  synthetic partial-history tests exercise both invariance and positive
  sensitivity to visible history. The previous pooled-history defect is fixed.
- Shard rows are copied; the weakref/gc test verifies backing-array release.
- Optimizer state is cloned after loading, and strict determinism disables
  memory-efficient/flash SDPA during training. The rebuilt probe no longer
  shares mutable optimizer state between replicas. Recorded resumed loss and
  final-weight differences are zero; the earlier allocator attribution is
  correctly withdrawn.
- Joint-scene selection uses one ADE-minimizing sample per scene for all scored
  agents. Its reported FDE is **FDE at the ADE-selected sample**, not an
  independently FDE-minimizing metric; preserve that definition in tables.
- Four plots per source exist; an INTERACTION overlay was visually inspected.
  Rollouts remain visibly poor, consistent with the reported metrics.
- No heldout trajectory content was read by this review.

## Corrections made during review

1. `rerun_chain.sh` still contained the original header-parsing failure and
   a `| tee` pipeline without pipefail, despite the completion report saying
   it had been fixed. It now uses `set -euo pipefail`, checked `cmp`, and
   SHA-256 output. Existing matching loss files remain valid evidence.
2. REPORT section 6 incorrectly called Waymo teacher-argmax ADE 4.03 better
   than CV 3.969 and the old model 3.76. Both comparisons go the other way.
   Corrected those statements and removed the unsupported causal explanation.

## Additional interpretation limits for the next phase

- Sampled NLL is E under the top-p sampling distribution of minus log
  probability under the original model. Calling it a Monte Carlo estimate of
  sampling entropy is still incorrect when those distributions differ. The
  formula in the report is usable; the entropy shorthand in
  REPORT/SMOKE_VALIDATION/docs was corrected during this review.
- Teacher forcing versus rollout is a diagnostic contrast, not an isolation
  experiment proving a unique cause. The data do not establish that more
  training will close the gap.
- A 100% acceleration/jerk violation diagnostic must not be dismissed as
  harmless simply because discretization contributes. Quantify validity under
  the final reconstruction/execution scheme before safety-critical use.
- The current memory figure is a smoke configuration measurement. It is not
  an 11-fold guarantee for batch-size or model-size scaling.

## Visibility: conditional prediction versus joint rollout

The fixed-teacher counterfactual test and end-to-end joint-rollout invariance
are different claims. The archived `visibility_rollout_review.py` uses the
submitted checkpoint on a synthetic fixture, holds all query-visible history
and masks fixed, and perturbs source 6's hidden past (t=0 remains visible).
On CPU with seed 123, fixed-teacher query-0 logits were bitwise equal, while
joint rollout changed 5 query-0 tokens across six samples and had maximum
query-0 coordinate difference 11.2695045 m. This is an adversarial synthetic
diagnostic with a large clipped input perturbation, not a measured dataset
failure rate or realistic effect-size estimate.

The route is other agents' generated actions entering later chunk prefixes.
Generated visible actions can legitimately transmit causal effects in a joint
scene model; therefore this result alone does **not** prove an illegal
hidden-state leak. Conversely, the phrase "identity keys contain no history"
does not prove full-rollout observation isolation: token identities can depend
on their generating agents' private histories.

Accept the current repair for conditional nominal scene generation. Before
closed-loop actor use, specify which generated actions are observable at each
step and test policy invariance with the observable action history held fixed,
using dynamic environment masks. Do not silently turn conditional-logit tests
into a claim of globally invariant joint trajectories.

## Next phase

Proceed to the previously planned 100-shard architecture tier, with group-level
split isolation, bounded loading, source-specific exposure accounting, fixed
training budgets and validation checkpoints, CV comparison, and both per-agent
and joint-scene metrics. Keep final heldout sealed. Treat rollout quality,
kinematics, and INTERACTION independent-group coverage as measured questions;
the present engineering GO makes no promise of scientific success.
