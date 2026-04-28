"""CSI300 HSF experiment: factor_projection_v1 local encoder with branch_mlp_v1 fusion."""

from __future__ import annotations

import copy

from next_step.hierarchical_state_field.experiment.configs import (
    ref_all_hsf_csi300_factorproj as base,
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
trainer_config = model_conf["kwargs"]["trainer_config"]

model_config["state_fusion_mode"] = "branch_mlp_v1"
trainer_config["enable_local_counterfactual_diag"] = True
