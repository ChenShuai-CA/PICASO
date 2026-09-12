# Codex audit of commit 6a3e908

Date: 2026-09-13

## Decision

Adopt `aeb_dataset_v1` and the observed braking-response features.  Adopt the
revised `abd_derived_v2_sensitivity.json` for new numerical-sensitivity runs.
Do not describe the 680 rows as ECU-confirmed AEB events, the 0.15-0.35 s band
as measured delay, or the pooled uniform envelope as a fitted fleet
distribution.

## Reproduced checks

- `aeb_dataset_v0`: 707 unique run/hash pairs.
- `aeb_dataset_v1`: 680 unique run/hash pairs.
- Added residual-risk removals: 27 = 12 `foot_pre_event` + 15
  `no_tension_force_ge_30n`; kept and removed sets are disjoint and exactly
  partition v0.
- Review queue: 46 path records, 34 unique file hashes.  It is disjoint from
  v0/v1 because it was excluded before v0 was formed.
- Raw provenance: all 680 source TXT files exist and all 680 SHA-256 values
  match the recorded hashes.
- Timing: 680/680 rows align with v1 and parse successfully; 670 stop within
  10 s; TTC is available for 677; one 0.35 s back-calculation precedes the log
  start and remains explicitly negative.
- Outcomes: clear 516, contact-and-stopped proxy 56, near-contact proxy 24,
  passed-or-swept proxy 84.
- Response envelope source: 612 stopped rows outside the longitudinal contact
  proxy, covering 10 vehicle directories.  The two largest directories supply
  388/612 rows, so the result is not fleet-balanced.

## Corrections made during audit

1. Replaced the hand-written percentile indices with standard linear
   quantiles.  The response envelope changed from U(5.772, 8.952) to
   **U(5.801, 8.951) m/s2**; the substantive conclusion is unchanged.
2. Replaced ambiguous `active_s=onset` output with a directly observed
   `observed_response_onset_s` and separate synthetic request/active proxy
   columns.  Request and ECU-active proxies are identical because the historic
   logs cannot distinguish them.
3. Split simulator `response_delay` into `controller_preview_delay` and
   `aeb_actuation_delay` for new configs.  Legacy specs still fall back to the
   old combined field and keep their replay behavior.  The revised config fixes
   controller preview at the nominal 0.25 s while sampling actuation delay from
   U(0.15, 0.35), so the controller no longer knows each execution-delay draw.
4. Corrected `action_delay_steps=0-2` from the reported 0-40 ms to **0-200 ms**:
   the queue advances once per 0.1 s decision step, although physics integrates
   internally at 0.02 s.
5. Clarified evidence strength: 43/680 rows are operator-confirmed and 637 are
   conservative machine-screened response candidates based on ABD pedal
   mechanics.

## Statistical scope

The 612-row pooled distribution varies by vehicle and scenario.  Vehicle-level
medians range from 6.678 to 8.368 m/s2 among groups with available rows; the
largest groups dominate the pooled quantiles.  Uniform sampling between pooled
p5 and p95 is therefore an ABD-supported sensitivity design, not empirical
resampling and not an estimate of population probabilities.

The contact labels are longitudinal-distance proxies without full lateral
geometry.  Excluding `contact_and_stopped` avoids obvious collision-dynamics
contamination, but “non-contact” remains proxy-defined.

## Validation

- Artifact partition, timing semantics, standard quantiles and split-config
  consumption are covered by regression tests.
- The revised config passes `load_perturb_config`.
- Full repository result: `133 passed` with the WSL project environment.
