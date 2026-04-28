"""CSI300 HSF: ref-all relative-complexity macro state with macro + day summary."""

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

MARKET_STATE_PATH = "artifacts/market_state/daily_market_field_ref_all_SH000300_relative_complexity_v1.pkl"
MARKET_DAY_SUMMARY_PATH = "artifacts/market_state/daily_market_observation_ref_all_pti_v1.pkl"


data_conf = copy.deepcopy(base.data_conf)
model_conf = copy.deepcopy(base.model_conf)
port_conf = copy.deepcopy(base.port_conf)

model_config = model_conf["kwargs"]["model_config"]
trainer_config = model_conf["kwargs"]["trainer_config"]

model_config["main_loss"] = "mse"
model_config["mse_normalize"] = False
model_config["value_embedding_type"] = "feature_tokenizer"
model_config["pooling_mode"] = "full"
model_config["local_state_input_mode"] = "last_mean_std_trend_vol"
model_config["state_fusion_mode"] = "sum_norm"
model_config["global_state_use_macro"] = True
model_config["global_state_use_day_summary"] = True
model_config["router_summary_source"] = "day_asset"
model_config["pooling_summary_source"] = "day_asset"
model_config["router_use_global_state"] = True
model_config["router_use_local_state"] = True
model_config["film_use_global_state"] = True
model_config["film_use_local_state"] = True
model_config["pooling_use_global_state"] = True
model_config["pooling_use_local_state"] = True

trainer_config["market_state_path"] = MARKET_STATE_PATH
trainer_config["market_day_summary_path"] = MARKET_DAY_SUMMARY_PATH
trainer_config["seed"] = 42
trainer_config["enable_local_counterfactual_diag"] = True
trainer_config["enable_expert_advantage_diag"] = True

