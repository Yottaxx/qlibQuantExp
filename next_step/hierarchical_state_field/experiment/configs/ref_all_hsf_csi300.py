"""Standalone CSI300 config for the ref-all hierarchical state field target run.

This config intentionally does not import work_flow.py.  It freezes the current
workflow semantics while switching the state assets to the ref-all shared
observation plus SH000300 benchmark-specific field.
"""

DATA_START = "2008-01-01"
DATA_END = "2022-12-31"
FIT_START = "2008-01-01"
FIT_END = "2020-03-31"
TRAIN_START = "2008-01-30"
TRAIN_END = "2020-03-31"
EVAL_START = "2020-07-01"
EVAL_END = "2022-12-31"

MARKET_STATE_PATH = "artifacts/market_state/daily_market_field_ref_all_SH000300.pkl"
MARKET_DAY_SUMMARY_PATH = "artifacts/market_state/daily_market_observation_ref_all_pti_v1.pkl"


data_conf = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "step_len": 8,
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": DATA_START,
                "end_time": DATA_END,
                "fit_start_time": FIT_START,
                "fit_end_time": FIT_END,
                "instruments": "csi300",
                "infer_processors": [
                    {
                        "class": "RobustZScoreNorm",
                        "kwargs": {
                            "fields_group": "feature",
                            "clip_outlier": True,
                        },
                    },
                    {
                        "class": "Fillna",
                        "kwargs": {
                            "fields_group": "feature",
                        },
                    },
                ],
                "learn_processors": [
                    {
                        "class": "DropnaLabel",
                    },
                    {
                        "class": "CSZScoreNorm",
                        "kwargs": {
                            "fields_group": "label",
                            "method": "robust",
                        },
                    },
                ],
                "label": ["Ref($close, -5) / Ref($close, -1) - 1"],
            },
        },
        "segments": {
            "train": (TRAIN_START, TRAIN_END),
            "valid": (EVAL_START, EVAL_END),
            "test": (EVAL_START, EVAL_END),
        },
    },
}


model_conf = {
    "class": "QlibQuantMoE",
    "module_path": "module.model_adapter",
    "kwargs": {
        "model_config": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
            "dropout": 0.1,
            "initializer_range": 0.02,
            "num_alphas": 158,
            "context_len": 8,
            "value_embedding_type": "feature_tokenizer",
            "feature_tokenizer_bias": True,
            "feature_tokenizer_add_factor_id": False,
            "feature_tokenizer_init_std": 0.02,
            "use_regime_time_embedding": True,
            "time_tau_min": 0.5,
            "time_tau_max": 50.0,
            "time_tau_init": 5.0,
            "time_emb_init_std": 0.02,
            "time_decay_normalize": True,
            "use_regime_factor_gate": True,
            "factor_gate_scale": 0.5,
            "factor_gate_shift_scale": 0.0,
            "router_noise": 0.01,
            "router_temperature": 1.0,
            "router_z_loss_coef": 0.01,
            "router_use_layer_summary": False,
            "router_summary_source": "day_asset",
            "use_alibi": False,
            "use_feature_selection": False,
            "selection_reg_lambda": 1e-5,
            "selection_temperature": 0.1,
            "selection_noise_std": 0.5,
            "main_loss": "mse",
            "loss_weights": {
                "listmle": 1.0,
                "mse": 1.0,
                "ic": 1.0,
                "rank": 0.0,
                "huber": 0.0,
            },
            "mse_normalize": False,
            "rank_topk": 5,
            "huber_delta": 1.0,
            "listmle_tau": 0.8,
            "use_external_macro": True,
            "d_macro_input": 0,
            "d_day_summary_input": 0,
            "regime_macro_dropout": 0.05,
            "use_hierarchical_state_field": True,
            "d_global_state": 64,
            "d_local_state": 64,
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
            "regime_internal_mode": "long",
            "regime_internal_lag": 5,
            "regime_internal_use_batch_stats": False,
            "regime_internal_tail_threshold": 2.0,
            "pooling_alpha": 0.7,
            "pooling_mode": "full",
            "pooling_alpha_scale": 0.3,
            "pooling_d_ff": None,
            "pooling_use_layer_summary": False,
            "pooling_summary_source": "day_asset",
        },
        "trainer_config": {
            "lr": 5e-5,
            "n_epochs": 20,
            "batch_size": 300,
            "eval_batch_size": 300,
            "precision": "fp32",
            "grad_accum_steps": 1,
            "seed": 42,
            "early_stop": 5,
            "train_stop_key": "loss_main",
            "train_stop_threshold": None,
            "min_epochs": 1,
            "consecutive_k": 1,
            "num_workers": 0,
            "market_state_path": MARKET_STATE_PATH,
            "market_state_shift": 0,
            "market_state_strict": True,
            "market_state_missing_policy": "ffill",
            "market_day_summary_path": MARKET_DAY_SUMMARY_PATH,
            "market_day_summary_shift": 0,
            "market_day_summary_strict": True,
            "market_day_summary_missing_policy": "ffill",
            "use_warmup": True,
            "warmup_ratio": 0.05,
            "warmup_steps": 0,
            "debug_sanity_check": True,
            "strict_valid_data_key": True,
            "fill_nonfinite_feature": True,
            "enable_local_counterfactual_diag": False,
        },
    },
}


port_conf = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 30,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": EVAL_START,
        "end_time": EVAL_END,
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            "deal_price": "close",
        },
    },
}
