"""CSI300 tri-scope HSF v4: residual GRU, mean pooling, bounded local FiLM, full-x/internal state in HSF."""

from __future__ import annotations

import copy

from .ref_all_hsf_csi300_tri_scope_gru_mean_inner_router import data_conf as _base_data_conf
from .ref_all_hsf_csi300_tri_scope_gru_mean_inner_router import model_conf as _base_model_conf
from .ref_all_hsf_csi300_tri_scope_gru_mean_inner_router import port_conf as _base_port_conf


data_conf = copy.deepcopy(_base_data_conf)
model_conf = copy.deepcopy(_base_model_conf)
port_conf = copy.deepcopy(_base_port_conf)

model_cfg = model_conf["kwargs"]["model_config"]
trainer_cfg = model_conf["kwargs"]["trainer_config"]

model_cfg.update(
    {
        "temporal_pooling_mode": "gru_residual_v1",
        "temporal_gru_residual_scale_init": 0.05,
        "temporal_gru_residual_scale_learnable": True,
        "pooling_mode": "mean_only",
        "pooling_num_queries": 1,
        "pooling_alpha": 0.0,
        "pooling_alpha_scale": 0.0,
        "pooling_use_global_state": False,
        "pooling_use_local_state": False,
        "pooling_summary_source": "none",
        "state_fusion_mode": "bounded_sum_v1",
        "local_router_scale_init": 0.1,
        "local_film_scale_init": 0.05,
        "local_scale_learnable": True,
        "film_use_global_state": True,
        "film_use_local_state": True,
        "router_aux_loss_type": "usage_entropy",
        "router_usage_entropy_coef": 0.05,
        "router_z_stability_coef": 1e-3,
        "router_use_internal_batch_state": True,
        "router_internal_scale_init": 1.0,
        "router_internal_scale_learnable": False,
        "d_internal_state_input": 4,
        "global_state_use_internal_state": True,
        "global_state_use_x": True,
        "local_state_use_internal_state": True,
    }
)

trainer_cfg.update(
    {
        "seed": 42,
        "enable_local_counterfactual_diag": True,
        "enable_expert_advantage_diag": True,
    }
)
