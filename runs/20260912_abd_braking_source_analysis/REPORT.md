# ABD AEB-response versus manual-braking diagnostic

- Compared 44 operator-confirmed AEB response windows with 33 operator-confirmed manual braking windows from FCW-only tests; all have zero event-window BR Command.
- Vehicles present in both classes: 9-BZ7, Guang_Qi_S9_data, XPengP7+_data.

- AEB median onset speed: **20.8 km/h**; manual median: **59.3 km/h**. Only **4 AEB** and **25 manual** events lie in the common 50.3-64.2 km/h support.

| Feature | AEB median | Manual median | Separation AUC | LORO balanced acc. | LOVO balanced acc. |
|---|---:|---:|---:|---:|---:|
| speed_rebound_kph | 2.4679 | 3.1707 | 0.924 | 0.901 | 0.791 |
| event_duration_s | 0.8350 | 1.6600 | 0.916 | 0.943 | 0.929 |
| time_onset_to_peak_s | 0.5000 | 1.3100 | 0.915 | 0.886 | 0.847 |
| positive_speed_step_fraction | 0.0730 | 0.0571 | 0.856 | 0.788 | 0.791 |
| effective_constant_deceleration_distance_mps2 | 7.4985 | 8.9860 | 0.835 | 0.776 | 0.459 |

The manuals support BR Command as the direct robot-source discriminator and Motion Pack channels as braking-shape evidence. They also explicitly warn that acceleration thresholds can detect pre-brake, driver intervention, or other external deceleration.
Manual evidence: RC Software Manual RM-S-01 Issue 23, sections 6.12.11.11.5 and 6.12.11.11.10 (PDF pp.107 and 110-111); Post Processor User Guide AN-6083 Issue 9, "Falsely Identified AEB Events" (document p.18 / PDF p.19).

These results therefore support automated ranking and evidence display. They do not yet support automatic labelling of the remaining BR-zero AEB-path records, because the manual comparison class comes from a different FCW-only test domain and 40 AEB examples were preselected by AEB-prototype similarity.

The apparent full-sample separation in duration and time-to-peak is strongly confounded by onset speed. In the common-speed subset, reviewed AEB and manual events have nearly equal peak-deceleration medians (AEB 11.098 vs manual 11.012 m/s2) and similar onset-to-peak times (AEB 1.37 vs manual 1.27 s). The AEB subset has only four runs. A rule such as "harder braking means AEB" is not supported by these data.
