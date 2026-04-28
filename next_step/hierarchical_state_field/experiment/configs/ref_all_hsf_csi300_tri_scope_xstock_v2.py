"""CSI300 tri-scope HSF v2 with cross-stock representation and denoised pooling."""

from __future__ import annotations

import copy

from .ref_all_hsf_csi300_tri_scope_relative_momentum import data_conf as _base_data_conf
from .ref_all_hsf_csi300_tri_scope_relative_momentum import model_conf as _base_model_conf
from .ref_all_hsf_csi300_tri_scope_relative_momentum import port_conf as _base_port_conf
from .ref_all_hsf_csi300_tri_scope_relative_momentum import (
    MARKET_DAY_SUMMARY_PATH,
    MARKET_STATE_PATH,
)


data_conf = copy.deepcopy(_base_data_conf)
model_conf = copy.deepcopy(_base_model_conf)
port_conf = copy.deepcopy(_base_port_conf)

model_cfg = model_conf["kwargs"]["model_config"]
trainer_cfg = model_conf["kwargs"]["trainer_config"]

model_cfg.update(
    {
        "d_model": 64,
        "n_heads": 4,
        "d_ff": 128,
        "main_loss": "mse",
        "mse_normalize": False,
        "value_embedding_type": "feature_tokenizer",
        "use_hierarchical_state_field": True,
        "router_use_local_state": True,
        "router_use_stock_token": True,
        "router_stock_scale_init": 0.1,
        "router_stock_scale_learnable": True,
        "router_aux_loss_type": "usage_entropy",
        "router_usage_entropy_coef": 0.01,
        "temporal_pooling_mode": "learned_query_mha_v1",
        "temporal_pooling_heads": 4,
        "use_cross_stock_attention": True,
        "cross_stock_attention_heads": 4,
        "cross_stock_residual_scale_init": 0.1,
        "cross_stock_scale_learnable": True,
        "pooling_mode": "full",
        "pooling_num_queries": 4,
        "pooling_alpha": 0.25,
        "pooling_alpha_scale": 0.15,
        "pooling_use_local_state": True,
    }
)

trainer_cfg.update(
    {
        "batch_size": 300,
        "eval_batch_size": 300,
        "precision": "amp_fp16",
        "grad_accum_steps": 1,
        "market_state_path": MARKET_STATE_PATH,
        "market_day_summary_path": MARKET_DAY_SUMMARY_PATH,
        "seed": 42,
        "enable_local_counterfactual_diag": True,
        "enable_expert_advantage_diag": True,
    }
)
