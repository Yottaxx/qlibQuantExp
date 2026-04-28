"""CSI300 tri-scope HSF v5: legacy layer-summary concat + stock-level inner relation.

This config keeps the v4 stable skeleton as the rollback point:
- residual GRU temporal pooling
- mean-only factor pooling
- bounded local FiLM/router
- usage-entropy router auxiliary with small z-stability term
- internal state injected into both GlobalStateEncoder and LocalStateEncoder

The only intended deltas are:
- stronger factor gate scale/shift
- per-layer batch layer summary concatenated to the router regime embedding
- inner cross-stock upgraded from a day token broadcast to stock-level relation attention
- full-x global branch encoded through XPanelStateEncoder
"""

from __future__ import annotations

import copy

from .ref_all_hsf_csi300_tri_scope_resgru_mean_bounded_inner import data_conf as _base_data_conf
from .ref_all_hsf_csi300_tri_scope_resgru_mean_bounded_inner import model_conf as _base_model_conf
from .ref_all_hsf_csi300_tri_scope_resgru_mean_bounded_inner import port_conf as _base_port_conf


data_conf = copy.deepcopy(_base_data_conf)
model_conf = copy.deepcopy(_base_model_conf)
port_conf = copy.deepcopy(_base_port_conf)

model_cfg = model_conf["kwargs"]["model_config"]
trainer_cfg = model_conf["kwargs"]["trainer_config"]

model_cfg.update(
    {
        "factor_gate_scale": 1.0,
        "factor_gate_shift_scale": 0.2,
        "router_use_layer_summary": True,
        "router_summary_source": "batch",
        "router_summary_fusion_mode": "legacy_concat_v1",
        "use_inner_cross_stock_attention": True,
        "inner_cross_stock_mode": "stock_time_factor_v1",
        "inner_cross_stock_diag_factor_samples": 8,
    }
)

trainer_cfg.update(
    {
        "seed": 42,
        "enable_local_counterfactual_diag": True,
        "enable_expert_advantage_diag": True,
    }
)
