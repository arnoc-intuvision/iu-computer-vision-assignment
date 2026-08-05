# Stage C — kappa Sweep (Phase 7)

Sweep over kappa on clean + motionblur. kappa frozen by pooled AMOTA.

| kappa | condition | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | mAVE |
|---|---|---|---|---|---|---|---|---|
| 1.0 | clean | 0.6324 | 0.6887 | 0.5835 | 0.3488 | 260 | 175 | 0.900 |
| 1.0 | motionblur sev1 | 0.6045 | 0.7408 | 0.5521 | 0.3331 | 230 | 167 | 0.941 |
| 1.0 | motionblur sev2 | 0.5403 | 0.8362 | 0.4889 | 0.3636 | 236 | 186 | 0.974 |
| 1.0 | motionblur sev3 | 0.4752 | 0.8971 | 0.4251 | 0.4196 | 304 | 245 | 1.041 |
| 3.0 | clean | 0.6300 | 0.6918 | 0.5811 | 0.3534 | 257 | 179 | 0.900 |
| 3.0 | motionblur sev1 | 0.5975 | 0.7577 | 0.5504 | 0.3489 | 243 | 181 | 0.938 |
| 3.0 | motionblur sev2 | 0.5188 | 0.8536 | 0.4674 | 0.3806 | 284 | 231 | 0.956 |
| 3.0 | motionblur sev3 | 0.4490 | 0.9447 | 0.4113 | 0.4196 | 297 | 252 | 1.011 |
| 5.0 | clean | 0.6287 | 0.6938 | 0.5812 | 0.3519 | 276 | 185 | 0.901 |
| 5.0 | motionblur sev1 | 0.5853 | 0.7769 | 0.5401 | 0.3596 | 247 | 204 | 0.943 |
| 5.0 | motionblur sev2 | 0.5127 | 0.8796 | 0.4707 | 0.4117 | 306 | 251 | 0.965 |
| 5.0 | motionblur sev3 | 0.4347 | 0.9585 | 0.3989 | 0.4424 | 311 | 283 | 0.968 |

**Pooled AMOTA by kappa:** k=1.0 -> 0.5631, k=3.0 -> 0.5488, k=5.0 -> 0.5404

**Frozen kappa = 1.0** (highest pooled AMOTA).

