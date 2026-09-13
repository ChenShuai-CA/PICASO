# P3.3.1 smoke data pipeline report

Status: **G0 PASS; model not trained; G1 not evaluated**.

The frozen smoke selection was converted into 26 resumable source units and 31,215
`scene-shard-v1` examples: 4,982 Waymo Scenario examples and 26,233 INTERACTION window/anchor
examples. The split contains 26,799 train and 4,416 dev examples. The statistical units are 4,982
Waymo Scenario IDs and 16 INTERACTION recording cases, not the overlapping window count.

All source units were resumed after the post-processing interruption. `VERIFICATION.json` confirms
that every final shard hash and example count matches its manifest, sample IDs are unique, metadata
and motion-token contracts pass, all 128 tokens are occupied, and no independent group crosses
train/dev. The maximum coordinate round-trip error is 4.56e-12 m. Heldout trajectory content was
not read.

The codebook was fit on train only with bounded source-by-agent-type reservoirs. It labeled
3,371,044 valid targets; normalized token entropy is 0.840 and the largest token share is 21.58%.
These are engineering diagnostics, not evidence of predictive quality.

Known limitations: 7,423 examples truncate agents at 16 and 30,685 truncate map polylines at 64;
the visibility field is a dynamic-box geometric proxy without static-building occlusion; ambiguous
INTERACTION pedestrian/bicycle rows are typed as `other`. Smoke dev has 481 independent Waymo
Scenario IDs and four INTERACTION cases, below the frozen G1 minimum of 5,000 and 100. P3.3.2 must
therefore remain a model/dataloader smoke stage before the 100-shard architecture experiment.
