# STOCK-axis KEY-SVD forensics (cross-stock common-mode probe)

- backbone: `mlruns/716849652326531066/2e8e24fe541a4a31949275b387da2ef2/artifacts/model` (g012_anchor_seed42)
- device: CPU (CUDA_VISIBLE_DEVICES="")
- D=64, n_heads=4, d_head=16, scale=0.2500, n_layers=2, num_alphas=158
- valid_stride=3, n_factor_slots=8
- **faithfulness gate**: valid daily rank_ic = **0.076381** over 203 days (g012 band ~0.078-0.082) -> PASS

## Per-layer cross-stock geometry (mean over slots/days)

| layer | s1/common | common_energy_frac | eff_rank | s2/s1 | peak_ent@T0.5 | @T1 | @T2 | @T3 | @T5 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7851 | 0.6049 | 1.336 | 0.3027 | 0.9788 | 0.9276 | 0.8352 | 0.7847 | 0.7276 | not_common_mode_dominated |
| 1 | 0.8588 | 0.5914 | 2.100 | 0.4517 | 0.9758 | 0.9057 | 0.7774 | 0.7156 | 0.6509 | not_common_mode_dominated |

## Thresholds / reading
- **common-mode dominated** if `s1_over_common <~ 0.1` OR `common_energy_frac > ~0.8` (cross-stock keys ~= constant -> uniform attention).
- **residual peakable** if `peak_ent@T3 < ~0.9` (a perfectly-aligned warm query COULD un-uniform the residual).
- **world-A** = common-mode + peakable residual -> key-centering + warm query fixes it.
- **world-B** = residual structureless / not peakable -> abandon stock attention.

## Overall verdict (final layer 1): **not_common_mode_dominated**
- s1_over_common=0.8588, common_energy_frac=0.5914, peak_ent@T3=0.7156

