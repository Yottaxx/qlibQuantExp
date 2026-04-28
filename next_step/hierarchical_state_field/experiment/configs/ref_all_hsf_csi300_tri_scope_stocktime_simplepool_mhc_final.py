"""CSI300 tri-scope legacy-regime v7 ablation: no inner cross-stock.

Derived from v6:
- Dropout increased: 0.1 -> 0.3 (to fight overfitting)
- Learning Rate decreased: 5e-5 -> 1e-5 (for more stable convergence)
- Epochs extended: 20 -> 40
- All fixes applied: RMSNorm, Pre-Norm, MHA Scaled Init.
- HSF disabled: use the legacy RegimeContextEncoder branch.
- Inner cross-stock attention disabled.
- Factor ID added to feature-tokenizer tokens.
- Router auxiliary loss switched back to z-loss.
"""

from __future__ import annotations

import copy

from .ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc import data_conf as _base_data_conf
from .ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc import model_conf as _base_model_conf
from .ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc import port_conf as _base_port_conf


data_conf = copy.deepcopy(_base_data_conf)
model_conf = copy.deepcopy(_base_model_conf)
port_conf = copy.deepcopy(_base_port_conf)

model_cfg = model_conf["kwargs"]["model_config"]
trainer_cfg = model_conf["kwargs"]["trainer_config"]

# 1. Increase Dropout to combat the 0.25 Train IC vs 0.02 Valid IC gap
model_cfg.update(
    {
        "dropout": 0.3,
        "regime_macro_dropout": 0.2,  # Also increase macro dropout
        "feature_tokenizer_add_factor_id": True,
        "router_aux_loss_type": "z_loss",
        "use_hierarchical_state_field": False,
        "router_summary_source": "batch",
        "router_summary_fusion_mode": "default",
        "pooling_summary_source": "none",
        "use_inner_cross_stock_attention": False,
        "inner_cross_stock_mode": "day_token",
    }
)

# 2. Refine Trainer settings: run the full 40 epochs without early stopping.
trainer_cfg.update(
    {
        "lr": 1e-5,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "adam_betas": (0.9, 0.998),
        "adam_eps": 1e-8,
        "adam_amsgrad": False,
        "adam_foreach": True,
        "adam_fused": None,
        "adamw_decay_matrix_only": True,
        "n_epochs": 40,
        "early_stop": 0,
        "train_stop_threshold": None,
        "seed": 42,
    }
)
