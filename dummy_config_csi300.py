data_conf = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "step_len": 8,  # æ—¶åºçª—å£ï¼Œå¯¹åº”æ¨¡åž‹ context_len
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2022-12-31",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2020-03-31",
                "instruments": "csi300",
                # æŽ¨ç†é¢„å¤„ç†ï¼ˆDK_Iï¼Œç”¨äºŽç‰¹å¾é¢„å¤„ç†ï¼‰ï¼š
                # - ç‰¹å¾ï¼šåŽ»æžå€¼ + å¡«å……
                # æ³¨æ„ï¼šDropnaLabel ä¸èƒ½æ”¾åœ¨ infer_processors ä¸­ï¼ˆQlib é™åˆ¶ï¼‰
                "infer_processors": [
                    {
                        "class": "RobustZScoreNorm",
                        "kwargs": {"fields_group": "feature", "clip_outlier": True},
                    },
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                # è®­ç»ƒé¢„å¤„ç†ï¼ˆtrain ä½¿ç”¨ DK_Lï¼‰ï¼š
                # - DropnaLabel: ç§»é™¤ NaN æ ‡ç­¾
                # - DropExtremeLabel: ç§»é™¤æˆªé¢ top/bottom 2.5% æžç«¯å€¼ï¼ˆå¤„ç†æ¶¨è·Œåœï¼Œå¯¹é½ MASTERï¼‰
                # - CSZScoreNorm: æˆªé¢ ZScore æ ‡å‡†åŒ–
                # æ³¨æ„ï¼švalid/test ä½¿ç”¨ DK_Iï¼Œä¸ä¼š drop extremeï¼Œè¯„ä¼°åœ¨å…¨éƒ¨æ•°æ®ä¸Šè¿›è¡Œ
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label", "method": "robust"}},
                ],
                # Label: ä¸‹äº”æ—¥æ”¶ç›Š
                "label": ["Ref($close, -5) / Ref($close, -1) - 1"],
            },
        },
        "segments": {
            "train": ("2008-01-01", "2020-03-31"),
            "test": ("2020-07-01", "2022-12-31"),
            "test": ("2020-07-01", "2022-12-31"),
        },
    },
}

# =============================================================================
# 2. Model Config (RST-MoE)
# =============================================================================
model_conf = {
    "class": "QlibQuantMoE",
    "module_path": "module.model_adapter",
    "kwargs": {
        "model_config": {
            # ---- Architecture ----
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
            "dropout": 0.1,
            "initializer_range": 0.02,
            # NOTE: context_len / num_alphas will be overwritten by data-driven values in QlibQuantMoE._init_net
            "context_len": 8,
            "num_alphas": 158,
            # ---- Value embedding ----
            "value_embedding_type": "feature_tokenizer",  # shared_linear | feature_tokenizer
            "feature_tokenizer_bias": True,
            "feature_tokenizer_add_factor_id": False,
            "feature_tokenizer_init_std": 0.02,
            # ---- Regime-adaptive time embedding ----
            "use_regime_time_embedding": True,
            "time_tau_min": 0.5,
            "time_tau_max": 50.0,
            "time_tau_init": 5.0,
            "time_emb_init_std": 0.02,
            "time_decay_normalize": True,
            # ---- Regime-adaptive factor gate (FiLM) ----
            "use_regime_factor_gate": True,
            "factor_gate_scale": 1.0,
            "factor_gate_shift_scale": 0.2,
            # ---- MoE router ----
            "router_noise": 0.01,
            "router_temperature": 1.0,
            "router_z_loss_coef": 0.01,
            "router_use_layer_summary": False,
            "router_summary_source": "day_asset",
            # ---- Positional/feature selection ----
            "use_alibi": False,  # recommended default (time embedding already provides position signal)
            "use_feature_selection": False,
            "selection_reg_lambda": 1e-5,
            "selection_temperature": 0.1,
            "selection_noise_std": 0.5,
            # ---- Loss ----
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
            # ---- Macro / regime context ----
            "use_external_macro": True,
            "d_macro_input": 0,
            "d_day_summary_input": 0,
            "use_hierarchical_state_field": False,
            "d_global_state": 64,
            "d_local_state": 64,
            "global_state_use_macro": True,
            "global_state_use_day_summary": True,
            "local_state_input_mode": "last_mean_std_trend_vol",
            "router_use_global_state": True,
            "router_use_local_state": True,
            "film_use_global_state": True,
            "film_use_local_state": True,
            "pooling_use_global_state": True,
            "pooling_use_local_state": True,
            "regime_macro_dropout": 0.1,
            "regime_internal_mode": "long",
            "regime_internal_lag": 5,
            "regime_internal_use_batch_stats": False,
            "regime_internal_tail_threshold": 2.0,
            # ---- Pooling ----
            "pooling_alpha": 0.7,
            "pooling_summary_source": "day_asset",
        },
        "trainer_config": {
            "lr": 5e-5,
            "n_epochs": 20,
            "batch_size": 300,  # å¯¹åº” FixedDailyBatchSampler çš„æ—¥åº¦ batch
            "eval_batch_size": 300,
            # Mixed precision:
            # - "amp_fp16": recommended on RTX 4070S (fastest, needs GradScaler)
            # - "amp_bf16": more stable, usually no GradScaler (requires BF16 support)
            # - "fp32": baseline
            "precision": "amp_fp16",
            # Gradient accumulation across K (shuffled) daily microbatches (K dates per optimizer step)
            "grad_accum_steps": 1,
            # [Safety Check] Internal Regime Encoder requires sufficient batch size (e.g. > 100)
            # to estimate covariance matrix. If using internal_mode, ensure batch_size is large enough.
            # "assert_batch_size_min": 100,
            "seed": 42,
            # "early_stop": 5,
            "train_stop_key": "loss_main",
            "train_stop_threshold": 1.33,
            "min_epochs": 5,
            "consecutive_k": 2,
            "num_workers": 0,  # debug æ—¶ç”¨ 0ï¼Œæ­£å¼è®­ç»ƒå¯ä»¥æ‹‰é«˜
            # Optional: precomputed market daily state as macro_features (recommended for longer horizons)
            "market_state_path": "artifacts/market_state/daily_market_field_csi300.pkl",
            "market_state_shift": 0,
            "market_state_strict": True,
            "market_day_summary_path": None,
            "market_day_summary_shift": 0,
            "market_day_summary_strict": True,
            # Warmup é…ç½®ï¼ˆä¸Ž adapter ä¸­çš„é»˜è®¤å€¼ä¸€è‡´ï¼‰ï¼š
            "use_warmup": True,
            "warmup_ratio": 0.05,
            "warmup_steps": 0,
            "debug_sanity_check":True,
            # Valid åªå…è®¸ DK_Iï¼š
            "strict_valid_data_key":True
        },
    },
}

# =============================================================================
# 3. Strategy & Backtest Config (å®˜æ–¹ port_analysis_config)
# =============================================================================
port_conf = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",  # å ä½ç¬¦ï¼ŒSignalRecord ä¼šè‡ªåŠ¨æ›¿æ¢
            "topk": 30,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2020-07-01",
        "end_time": "2022-12-31",
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            # "limit_threshold": 0.095,
            "deal_price": "close",
            # "open_cost": 0.0005,
            # "close_cost": 0.0015,
            # "min_cost": 5,
        },
    },
}

