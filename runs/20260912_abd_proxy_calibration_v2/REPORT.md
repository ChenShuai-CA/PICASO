# ABD reviewed evidence finalization and proxy calibration v2

- Operator-reviewed AEB response records: **44** unique runs across **10** vehicles and **14** scenario labels.
- Review decisions: **43** no takeover; **1** manual takeover only after the AEB stop (analysis censored at first stop).
- FCW audio edges: **188** FCW-only runs. The **33** BR-zero paired cases were all confirmed as expected manual braking after the warning.
- No FCW-only run is used for AEB response, AEB proxy timing, or FCW-to-AEB delay.
- `T_AEB_proxy = T_dec_03 - U(0.15, 0.35 s)` is retained as an operator-prior latent activation proxy, not an observed ECU request/active transition.

The v2 evidence package is intentionally blocked from direct simulator use until the current combined `response_delay` is split into trigger-policy and actuation delay terms. This prevents the 0.15-0.35 s prior from being counted twice.
