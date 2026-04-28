"""CSI300 tri-scope HSF v3: GRU temporal pooling, mean pooling, inner cross-stock, router internal stats."""

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
        "d_macro_input": 603,
        "d_day_summary_input": 419,
        "global_state_use_macro": True,
        "global_state_use_day_summary": True,
        "local_state_input_mode": "last_mean_std_trend_vol",
        "state_fusion_mode": "sum_norm",
        "router_use_global_state": True,
        "router_use_local_state": True,
        "router_use_stock_token": False,
        "router_aux_loss_type": "usage_entropy",
        "router_usage_entropy_coef": 0.01,
        "router_use_internal_batch_state": True,
        "router_internal_mode": "long",
        "router_internal_lag": 5,
        "router_internal_use_batch_stats": True,
        "router_internal_tail_threshold": 2.0,
        "router_internal_fusion": "concat_v1",
        "router_internal_scale_init": 0.1,
        "router_internal_scale_learnable": True,
        "film_use_global_state": True,
        "film_use_local_state": False,
        "temporal_pooling_mode": "gru_v1",
        "temporal_pooling_heads": 4,
        "pooling_mode": "mean_only",
        "pooling_num_queries": 1,
        "pooling_alpha": 0.0,
        "pooling_alpha_scale": 0.0,
        "pooling_use_global_state": False,
        "pooling_use_local_state": False,
        "pooling_summary_source": "none",
        "use_inner_cross_stock_attention": True,
        "inner_cross_stock_attention_heads": 4,
        "inner_cross_stock_residual_scale_init": 0.05,
        "inner_cross_stock_scale_learnable": True,
        "use_cross_stock_attention": False,
        "router_summary_source": "day_asset",
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
        "market_state_shift": 0,
        "market_day_summary_shift": 0,
        "market_state_strict": True,
        "market_day_summary_strict": True,
        "seed": 42,
        "enable_local_counterfactual_diag": True,
        "enable_expert_advantage_diag": True,
    }
)
