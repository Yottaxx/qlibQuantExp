"""CSI300 HSF relative-complexity control: bounded local residual fusion."""

from __future__ import annotations

import copy

from next_step.hierarchical_state_field.experiment.configs import (
    ref_all_hsf_csi300_relative_complexity_macro_plus_day_summary as base,
)


DATA_START = base.DATA_START
DATA_END = base.DATA_END
FIT_START = base.FIT_START
FIT_END = base.FIT_END
TRAIN_START = base.TRAIN_START
TRAIN_END = base.TRAIN_END
EVAL_START = base.EVAL_START
EVAL_END = base.EVAL_END
MARKET_STATE_PATH = base.MARKET_STATE_PATH
MARKET_DAY_SUMMARY_PATH = base.MARKET_DAY_SUMMARY_PATH

data_conf = copy.deepcopy(base.data_conf)
model_conf = copy.deepcopy(base.model_conf)
port_conf = copy.deepcopy(base.port_conf)

model_config = model_conf["kwargs"]["model_config"]
model_config["state_fusion_mode"] = "bounded_sum_v1"
model_config["local_router_scale_init"] = 0.1
model_config["local_pooling_scale_init"] = 0.1
model_config["local_pooling_alpha_scale_init"] = 0.1
model_config["local_film_scale_init"] = 0.1
model_config["local_scale_learnable"] = True

