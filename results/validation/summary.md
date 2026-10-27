| experiment | set | grid | thetas | insample_f1 | theta_loco_f1 | late_f1 | base_theta_loco_f1 | base_late_f1 | boot_diff | boot_ci_lo | boot_ci_hi | adopt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joint | joint.enabled=true |  | 3.0/3.5/4.0/4.5 | 0.525 | 0.498 | 0.527 | 0.479 | 0.527 | 0.046 | -0.008 | 0.104 | False |
| lof_cheap | lof.enabled=true lof.ref_max=200 lof.paa=4 |  | 3.0/3.5/4.0/4.5 | 0.517 | 0.517 | 0.492 | 0.479 | 0.527 | 0.038 | -0.011 | 0.094 | False |
| lof_full | lof.enabled=true |  | 3.0/3.5/4.0/4.5 | 0.523 | 0.510 | 0.577 | 0.479 | 0.527 | 0.043 | -0.015 | 0.105 | True |
| shift | shift.enabled=true | shift.window=24,48,96 | 3.0/3.5/4.0/4.5 | 0.478 | 0.435 | 0.505 | 0.479 | 0.527 | -0.001 | -0.054 | 0.061 | False |
| subspace | subspace.enabled=true | subspace.lag=16,32,64 | 3.0/3.5/4.0/4.5 | 0.530 | 0.488 | 0.538 | 0.479 | 0.527 | 0.051 | -0.005 | 0.116 | True |
| w_counts | transform.counts=anscombe |  | 3.5/4.0/4.5/5.0/5.5 | 0.389 | 0.389 | 0.449 | 0.479 | 0.527 | -0.090 | -0.145 | -0.038 | False |
| w_forecast | forecast.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.482 | 0.464 | 0.541 | 0.479 | 0.527 | 0.003 | -0.021 | 0.025 | False |
| w_joint | joint.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.541 | 0.485 | 0.527 | 0.479 | 0.527 | 0.062 | -0.004 | 0.125 | False |
| w_lof_full | lof.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.523 | 0.500 | 0.523 | 0.479 | 0.527 | 0.043 | -0.015 | 0.105 | False |
| w_neural | neural.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.469 | 0.469 | 0.514 | 0.479 | 0.527 | -0.010 | -0.037 | 0.013 | False |
| w_subspace | subspace.enabled=true | subspace.lag=16,32,64 | 3.5/4.0/4.5/5.0/5.5 | 0.533 | 0.476 | 0.538 | 0.479 | 0.527 | 0.054 | -0.011 | 0.126 | False |
| w_subspace32 | subspace.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.532 | 0.509 | 0.538 | 0.479 | 0.527 | 0.053 | -0.004 | 0.115 | True |
| w_subspace32_neural | subspace.enabled=true neural.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.530 | 0.504 | 0.528 | 0.479 | 0.527 | 0.051 | -0.005 | 0.111 | True |
| w_subspace_joint | subspace.enabled=true joint.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.523 | 0.496 | 0.531 | 0.479 | 0.527 | 0.043 | -0.020 | 0.109 | True |
| w_subspace_lof | subspace.enabled=true lof.enabled=true |  | 3.5/4.0/4.5/5.0/5.5 | 0.541 | 0.482 | 0.571 | 0.479 | 0.527 | 0.062 | -0.001 | 0.129 | True |
