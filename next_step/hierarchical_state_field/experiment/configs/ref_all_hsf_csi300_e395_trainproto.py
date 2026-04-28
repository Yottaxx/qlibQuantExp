"""CSI300 ref-all HSF config with e395-like training and FiLM strength.

This run intentionally keeps valid/test on the same 2020-07-01..2022-12-31
window per the current experiment request, while aligning the non-date
settings that materially differed from the e395 baseline.
"""

from __future__ import annotations

import copy

from next_step.hierarchical_state_field.experiment.configs import ref_all_hsf_csi300 as base


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

# Keep the requested validation/test overlap for this diagnostic experiment.
data_conf["kwargs"]["segments"]["valid"] = (EVAL_START, EVAL_END)
data_conf["kwargs"]["segments"]["test"] = (EVAL_START, EVAL_END)
port_conf["backtest"]["start_time"] = EVAL_START
port_conf["backtest"]["end_time"] = EVAL_END

# Align the factor FiLM amplitude with the e395 local baseline.
model_config["factor_gate_scale"] = 1.0
model_config["factor_gate_shift_scale"] = 0.2

# Keep HSF target routing/pooling semantics, but train with e395-like protocol.
trainer_config["n_epochs"] = 40
trainer_config["precision"] = "amp_fp16"
trainer_config["early_stop"] = 0
trainer_config["train_stop_key"] = "loss_main"
trainer_config["train_stop_threshold"] = 1.35
trainer_config["min_epochs"] = 5
trainer_config["consecutive_k"] = 2
