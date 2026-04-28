"""CSI300 tri-scope HSF v6: stock-time relation + mHC-lite GRU + simple static pooling.

This keeps the stock_time_factor_v1 inner relation path from the legacy-summary
probe, but makes the post-MoE aggregation deliberately simple:
- temporal pooling: bounded convex GRU/h_last mixing
- factor pooling: fixed-alpha attention/mean over h_last only
"""

from __future__ import annotations

import copy

from .ref_all_hsf_csi300_tri_scope_resgru_mean_legacy_summary_stockrel import data_conf as _base_data_conf
from .ref_all_hsf_csi300_tri_scope_resgru_mean_legacy_summary_stockrel import model_conf as _base_model_conf
from .ref_all_hsf_csi300_tri_scope_resgru_mean_legacy_summary_stockrel import port_conf as _base_port_conf


data_conf = copy.deepcopy(_base_data_conf)
model_conf = copy.deepcopy(_base_model_conf)
port_conf = copy.deepcopy(_base_port_conf)

model_cfg = model_conf["kwargs"]["model_config"]
trainer_cfg = model_conf["kwargs"]["trainer_config"]

model_cfg.update(
    {
        "temporal_pooling_mode": "gru_mhc_lite_v1",
        "temporal_mhc_mix_init": 0.05,
        "temporal_mhc_mix_max": 0.25,
        "pooling_mode": "simple_static",
        "pooling_num_queries": 1,
        "pooling_alpha": 0.7,
        "pooling_alpha_scale": 0.0,
        "pooling_use_global_state": False,
        "pooling_use_local_state": False,
        "pooling_summary_source": "none",
    }
)

trainer_cfg.update(
    {
        "seed": 42,
        "enable_local_counterfactual_diag": True,
        "enable_expert_advantage_diag": True,
    }
)
