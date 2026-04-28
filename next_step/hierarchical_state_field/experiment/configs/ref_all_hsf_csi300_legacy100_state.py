"""CSI300 HSF control: keep model local/fusion default, swap global state to 100D legacy ref-all."""

from __future__ import annotations

import copy

from next_step.hierarchical_state_field.experiment.configs import (
    ref_all_hsf_csi300_e395_trainproto as base,
)


DATA_START = base.DATA_START
DATA_END = base.DATA_END
FIT_START = base.FIT_START
FIT_END = base.FIT_END
TRAIN_START = base.TRAIN_START
TRAIN_END = base.TRAIN_END
EVAL_START = base.EVAL_START
EVAL_END = base.EVAL_END

MARKET_STATE_PATH = "artifacts/market_state/daily_market_ref_all_SH000300_legacy.pkl"
MARKET_DAY_SUMMARY_PATH = base.MARKET_DAY_SUMMARY_PATH


data_conf = copy.deepcopy(base.data_conf)
model_conf = copy.deepcopy(base.model_conf)
port_conf = copy.deepcopy(base.port_conf)

model_config = model_conf["kwargs"]["model_config"]
trainer_config = model_conf["kwargs"]["trainer_config"]

model_config["value_embedding_type"] = "feature_tokenizer"
model_config["pooling_mode"] = "full"
model_config["factor_gate_scale"] = 1.0
model_config["factor_gate_shift_scale"] = 0.2
model_config["local_state_input_mode"] = "last_mean_std_trend_vol"
model_config["state_fusion_mode"] = "sum_norm"
model_config["router_summary_source"] = "day_asset"
model_config["pooling_summary_source"] = "day_asset"

trainer_config["market_state_path"] = MARKET_STATE_PATH
trainer_config["seed"] = 42
trainer_config["enable_local_counterfactual_diag"] = True
