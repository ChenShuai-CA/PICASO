# ABD AVAD3 / FCW audio timing audit

- SPEC mappings to CAN User Defined 1/2: **201**.
- Mappings supported as FCW audio by an explicit sound label or FCW-context operator convention: **201**.
- Runs with an observed 0-to-1 audio edge: **188**, across **10** vehicle folders.

- Runs that also contain an observed braking onset: **186**. These are FCW-only tests, so the subsequent braking is not an AEB response.

- Of the paired runs, **153** have robot braking and **33** BR-zero runs were confirmed as the expected manual takeover after the warning.

- All recovered audio edges remain usable FCW timing observations. No historical FCW-only run is eligible for AEB response, AEB proxy timing, or FCW-to-AEB delay.

`T_FCW_audio_observed` is the first rising edge of `Time tolerance X (within tolerances)` after the companion SPEC maps that trigger input to CAN User Defined 1 or 2. This recovers the AVAD3-observed audible warning time even when the raw CAN User Defined column was not included in the TXT export.

`Time tolerance X (excl. delay)` completes the configured TTT true-time condition; `Time tolerance X` additionally includes the configured trigger delay. Neither should replace the within-tolerance first edge when estimating audible onset.

The recovered time is an external acoustic observation. It includes the vehicle warning generation, speaker/acoustic propagation, AVAD3 detection, and RC sampling chain; it is not the ECU internal FCW request transition.
