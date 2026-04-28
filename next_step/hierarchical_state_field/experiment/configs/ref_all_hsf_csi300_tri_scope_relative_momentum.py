"""CSI300 HSF config wired to tri-scope day summary and relative-momentum field."""

from __future__ import annotations

import copy

from .ref_all_hsf_csi300 import data_conf as _base_data_conf
from .ref_all_hsf_csi300 import model_conf as _base_model_conf
from .ref_all_hsf_csi300 import port_conf as _base_port_conf


MARKET_STATE_PATH = "artifacts/market_state/daily_market_field_ref_all_SH000300_tri_scope_relative_momentum_v1.pkl"
MARKET_DAY_SUMMARY_PATH = "artifacts/market_state/daily_market_observation_tri_scope_v1.pkl"


data_conf = copy.deepcopy(_base_data_conf)
model_conf = copy.deepcopy(_base_model_conf)
port_conf = copy.deepcopy(_base_port_conf)

model_cfg = model_conf["kwargs"]["model_config"]
trainer_cfg = model_conf["kwargs"]["trainer_config"]

model_cfg.update(
    {
        "use_hierarchical_state_field": True,
        "global_state_use_macro": True,
        "global_state_use_day_summary": True,
        "local_state_input_mode": "last_mean_std_trend_vol",
        "state_fusion_mode": "sum_norm",
        "router_use_global_state": True,
        "router_use_local_state": True,
        "film_use_global_state": True,
        "film_use_local_state": True,
        "pooling_use_global_state": True,
        "pooling_use_local_state": True,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
        "value_embedding_type": "feature_tokenizer",
        "pooling_mode": "full",
        "main_loss": "mse",
        "mse_normalize": False,
    }
)

trainer_cfg.update(
    {
        "market_state_path": MARKET_STATE_PATH,
        "market_day_summary_path": MARKET_DAY_SUMMARY_PATH,
        "market_state_shift": 0,
        "market_day_summary_shift": 0,
        "market_state_strict": True,
        "market_day_summary_strict": True,
        "seed": 42,
    }
)
