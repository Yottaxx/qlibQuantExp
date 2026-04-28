"""CSI300 tri-scope HSF v4 ablation: local state enters FiLM only."""

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
        "router_use_local_state": False,
        "film_use_local_state": True,
        "pooling_use_local_state": False,
        "pooling_summary_source": "none",
        "local_film_scale_init": 0.05,
    }
)

trainer_cfg.update(
    {
        "seed": 42,
        "enable_local_counterfactual_diag": True,
        "enable_expert_advantage_diag": True,
    }
)
