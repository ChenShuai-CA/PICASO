# P3.3.2 independent review

- Review date: 2026-09-13
- Reviewed commit: `ba8c4e9`
- Status: **REQUIRES_CORRECTION**
- Decision: retain the data/CV/training infrastructure, but do not promote the
  current checkpoint or model metrics to the 100-shard architecture stage.
  Complete a bounded P3.3.2a correction and rerun M1/M2/M3 first.

## Verified items

1. Commit scope matches the completion report: 35 files and 5,451 insertions,
   confined to the P3.3.2 implementation, tests, documentation, and result
   bundle.
2. Independent full-suite rerun with the required WSL Python completed with
   `177 passed, 1 warning`.
3. The independently recomputed trainable parameter count is 13,126,282.
4. All seven files listed in `ARTIFACT_MANIFEST.json` exist and match their
   recorded SHA-256 hashes.
5. The CV baseline, group-level aggregation, full-dev evaluation, M1 overfit,
   and the honest finding that the current rollout is worse than CV are
   supported by the committed artifacts.
6. No evidence of heldout trajectory-content access was found in this review.

## Blocking findings

### 1. Temporal actor-visibility leakage

`ARSceneV1._agent_static_tokens()` pools every source agent's complete
11-frame history using only `state_valid_mask`. `_encoder_mask()` then allows
that already-pooled token according to visibility at the final history frame
only. A source that is visible at t=0 can therefore disclose states from an
earlier frame in which it was hidden from the querying actor. This violates the
frozen per-query, per-time actor-visibility contract.

The existing counterfactual test hides a source for every history frame and
does not exercise this case. A direct partial-history counterfactual changed
query-0 logits by `2.3406744e-4` after modifying a source state at one hidden
past frame while keeping that source visible at t=0.

Dataset scan:

| split | samples | affected samples | current-visible pairs with hidden past | primary-query pairs |
|---|---:|---:|---:|---:|
| train | 26,799 | 25,329 | 350,726 | 248,186 |
| dev | 4,416 | 3,793 | 45,142 | 37,595 |

Consequence: the `actor_visible_counterfactual=pass` gate is a false pass. The
current checkpoint and model-derived M2/M3 diagnostics must be superseded after
the visibility fix. CV results are unaffected.

### 2. The allocator mechanism claim is not established

Training enables deterministic algorithms with `warn_only=True`, while the M2
log reports that memory-efficient attention uses a nondeterministic algorithm.
The claim that strict deterministic mode raised no error and ruled out kernel
nondeterminism is therefore false.

The allocator probe also loads the same in-memory `snapshot["optimizer"]`
payload into R0 and R2 sequentially. PyTorch optimizer loading on the same
device shares tensor storage with that payload. An independent pointer/mutation
check on the committed checkpoint returned `same_data_ptr=True`, and mutating
the loaded optimizer state changed the source payload. R0 training therefore
pollutes the optimizer state later used to initialize R2. This explains the
large R0/R2 divergence without requiring allocator-dependent GEMM selection.

Consequence: exact restore at the resume point remains useful evidence, but the
allocator-layout-to-cuBLAS attribution and the long-horizon determinism wording
must be withdrawn and retested with independent deep-copied states under
`warn_only=False` or a deterministic attention backend.

### 3. Resident-memory gate measures the reader pointer, not live arrays

`ShardReader.get()` returns NumPy row views. If a batch crosses a shard
boundary, those views keep the prior shard's full backing arrays alive after
the reader loads the next shard. The implementation can therefore transiently
retain two shards per source even though `reader.resident` names only one.

Consequence: the pipeline remains bounded, but the exact claim of one resident
shard per source and `max_resident_examples=4096` is not demonstrated. Return
copied rows or collate before switching shards, then test backing-array
lifetime rather than only the reader property.

## Required report corrections

1. `role=0` means padding, not `other`. `other` is `agent_type=4`; the dev
   evaluation contains 8,692 type-4 records, so that real-data path was
   exercised.
2. Source-balanced M2 exposure is 25,600 samples per source. Relative to the
   smoke train inventory this is about 5.69 Waymo passes and 1.15 INTERACTION
   passes. The repeated `~1.1 Waymo epoch` statement is mislabeled.
3. M2 elapsed time and throughput include the 200-update uninterrupted branch
   plus the 100-update resumed tail, i.e. 300 executed update-equivalents.
4. `sampled_token_nll` is computed from full-softmax log probabilities for
   samples drawn after top-p truncation/renormalization. It is not exactly the
   entropy of the sampling distribution, so `exp(1.872)` should not be called
   its effective vocabulary size.
5. Teacher-forced oracle/argmax decoding supports a diagnosis of exposure
   error, but does not uniquely rule out representation or decoder defects
   under off-distribution prefixes. The `0.7^10` calculation is a heuristic,
   not a measured rollout all-correct probability.
6. All eight committed overlays come from the initial Waymo dev batches. Add
   at least one source-stratified INTERACTION example as required by the
   P3.3.2 handoff.

## Metric decision needed before scale-up

The current `minADE@6` and `minFDE@6` select the best sample separately for
each agent. For a joint multi-agent generator, different agents can therefore
be scored from different sampled joint scenes. Before G1, freeze and report
both the existing per-agent oracle metric and a joint-scene best-of-6 metric
that uses one sample index for all evaluated agents in a scene/group, with one
declared primary endpoint.

## P3.3.2a exit conditions

1. Apply per-query, per-time visibility before any cross-agent pooling or
   attention, and add a single-hidden-past-frame counterfactual test.
2. Correct the shard lifetime behavior and add a backing-array lifetime/RSS
   test.
3. Rebuild the resume probe with independent deep copies and an actually strict
   deterministic configuration; downgrade any mechanism that remains
   unproven.
4. Correct role/type, exposure, timing, sampled-NLL, and attribution wording.
5. Produce source-stratified overlays and freeze the joint-scene best-of-K
   metric.
6. Rerun M1, M2, and full-dev M3 using the corrected model. Preserve the CV
   artifacts and keep heldout sealed.

Only after these conditions pass should the project enter the 100-shard
architecture experiment.
