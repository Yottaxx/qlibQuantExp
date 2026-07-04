# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import os
import random
from collections import defaultdict
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

from transformers.optimization import get_cosine_schedule_with_warmup

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model
from qlib.workflow import R

from module.dataloader.sampler import FixedDailyBatchSampler, DailyChunkBatchSampler
from module.quant_moe_model import QuantMoEModel
from module.utils.model_configuration import QuantMoEConfig
from module.utils.market_state import (
    MarketStateLookup,
    load_market_state_df,
    lookup_market_state,
    make_market_state_lookup,
)


class QlibQuantMoE(Model):
    """
    Qlib adapter for QuantMoEModel with TSDatasetH.

    Design:
    - 日度截面 batch: FixedDailyBatchSampler（为 CS / ListMLE 损失准备）。
    - 显式 schema 校验：
        * 探测 label 是否单独输出；否则检测是否 pack 在 x 的最后 label_dim 个通道。
    - 训练使用 DK_L，验证/测试/预测使用 DK_I（对齐 Qlib 官方工作流）。
    - Label NaN 不会进 loss/metric（先 mask 再计算）。
    - Warmup + cosine LR scheduler (transformers.get_cosine_schedule_with_warmup)。
    - Stopping & best checkpoint:
        * If trainer_config['early_stop']>0: legacy valid-metric based early stop (patience).
        * Else: train-metric (default loss_main) min-selection + optional threshold stop:
            - train_stop_key, train_stop_threshold, min_epochs, consecutive_k
    - Recorder logs:
        * train/* & valid/*：
            - loss_total / loss_main / loss_listmle / loss_mse / loss_ic / loss_aux / loss_sparsity
            - ic_pearson_batch / rank_ic_batch（训练：batch 内现算）
            - ic_pearson_daily / rank_ic_daily（验证：按日全截面聚合后现算）
            - ic_raw / rank_ic（兼容旧名字：训练时=*_batch，验证时=*_daily）
            - gate_entropy / time_ratio / active_feat_ratio
        * train_curve 对象：
            - epoch, train_main, train_listmle, train_mse, train_ic, valid_main, valid_rank_ic, valid_ic
    """

    def __init__(self, model_config: dict = None, trainer_config: dict = None, **kwargs):
        self.model_config = dict(model_config or {})
        self.trainer_config = dict(trainer_config or {})

        # Optimizer / schedule
        self.lr = float(self.trainer_config.get("lr", 5e-4))
        self.epochs = int(self.trainer_config.get("n_epochs", 20))
        self.batch_size = int(self.trainer_config.get("batch_size", 1024))
        self.num_workers = int(self.trainer_config.get("num_workers", 4))
        self.random_seed = self.trainer_config.get("seed", 42)
        self._warned_keys: set[str] = set()

        # Runtime reproducibility / device policy. Defaults are diagnostic-first:
        # keep training compatible, but make every source of randomness explicit.
        self.device_request = str(self.trainer_config.get("device", "auto") or "auto").strip().lower()
        self.deterministic_mode = str(self.trainer_config.get("deterministic_mode", "warn") or "warn").strip().lower()
        det_alias = {
            "true": "warn",
            "1": "warn",
            "yes": "warn",
            "false": "off",
            "0": "off",
            "no": "off",
            "warn_only": "warn",
            "warning": "warn",
            "strict": "strict",
            "error": "strict",
            "off": "off",
            "none": "off",
        }
        self.deterministic_mode = det_alias.get(self.deterministic_mode, self.deterministic_mode)
        if self.deterministic_mode not in {"off", "warn", "strict"}:
            raise ValueError(
                "deterministic_mode must be one of off/warn/strict, "
                f"got {self.deterministic_mode!r}"
            )
        self.seed_workers = bool(self.trainer_config.get("seed_workers", True))
        self.train_sampler_mode = str(
            self.trainer_config.get("train_sampler_mode", "sampled_daily") or "sampled_daily"
        ).strip().lower()
        sampler_alias = {
            "sampled": "sampled_daily",
            "fixed": "sampled_daily",
            "fixed_daily": "sampled_daily",
            "daily": "sampled_daily",
            "full": "full_daily",
            "full_day": "full_daily",
            "full_daily": "full_daily",
        }
        self.train_sampler_mode = sampler_alias.get(self.train_sampler_mode, self.train_sampler_mode)
        if self.train_sampler_mode not in {"sampled_daily", "full_daily"}:
            raise ValueError(
                "train_sampler_mode must be sampled_daily or full_daily, "
                f"got {self.train_sampler_mode!r}"
            )
        self.sampler_diag = bool(self.trainer_config.get("sampler_diag", True))
        self._runtime_flags: Dict[str, Any] = {}
        self._last_train_sampler_stats: Dict[str, Any] = {}

        self.early_stop = int(self.trainer_config.get("early_stop", 0) or 0)
        self.min_delta = float(self.trainer_config.get("min_delta", 1e-6))
        # Train-loss threshold stopping (used when early_stop is disabled)
        # - train_stop_key: metric key in train metrics dict (default: loss_main)
        # - train_stop_threshold: stop when metric <= threshold for `consecutive_k` epochs (after `min_epochs`)
        self.train_stop_key = str(self.trainer_config.get("train_stop_key", "loss_main") or "loss_main")
        _thr = self.trainer_config.get("train_stop_threshold", None)
        if _thr is None or (isinstance(_thr, str) and _thr.strip().lower() in {"", "none", "null"}):
            self.train_stop_threshold: Optional[float] = None
        else:
            self.train_stop_threshold = float(_thr)
        self.min_epochs = max(1, int(self.trainer_config.get("min_epochs", 1) or 1))
        self.consecutive_k = max(1, int(self.trainer_config.get("consecutive_k", 1) or 1))
        self.checkpoint_metric = str(self.trainer_config.get("checkpoint_metric", "") or "").strip()
        self.checkpoint_mode = str(self.trainer_config.get("checkpoint_mode", "max") or "max").strip().lower()
        if self.checkpoint_mode not in {"max", "min"}:
            raise ValueError(f"checkpoint_mode must be 'max' or 'min', got {self.checkpoint_mode!r}")
        self.checkpoint_min_delta = float(self.trainer_config.get("checkpoint_min_delta", self.min_delta))
        # Gradient accumulation (micro-batch = one date cross-section; accumulate across K dates)
        self.grad_accum_steps = max(1, int(self.trainer_config.get("grad_accum_steps", 5)))
        # Validation data_key policy
        # - Default: DK_I (infer), aligned with Qlib official workflows
        # - If strict_valid_data_key=True, validation will NOT fall back to DK_L.
        self.valid_data_key = str(self.trainer_config.get("valid_data_key", DataHandlerLP.DK_I))
        self.strict_valid_data_key = bool(self.trainer_config.get("strict_valid_data_key", False))

        # Optional: precomputed market daily state as macro_features
        # - market_state_path: path to DataFrame(index=datetime, columns=state_dims)
        # - shift: optionally shift state by k days (within provided index) to avoid look-ahead
        self.market_state_path = self.trainer_config.get("market_state_path", None)
        self.market_state_shift = int(self.trainer_config.get("market_state_shift", 0) or 0)
        self.market_state_strict = bool(self.trainer_config.get("market_state_strict", True))
        self._market_state: Optional[MarketStateLookup] = None

        # If TSDataSampler packs label into x: last `label_dim` channels are labels.
        # 对 Alpha158 + 单一 label，一般 label_dim=1。
        self.label_dim = int(self.trainer_config.get("label_dim", 1))

        # Strict feature-schema check (L-4 guard): when True, a feature-dim mismatch that is NOT
        # the packed-label case raises instead of silently truncating trailing channels. This
        # defeats the silent no-op where appended CS-rank channels (train==infer) would be dropped
        # at predict/eval if train and infer schemas ever diverge.
        self.strict_feature_schema = bool(self.trainer_config.get("strict_feature_schema", False))

        # Warmup scheduler config
        self.use_warmup = bool(self.trainer_config.get("use_warmup", True))
        self.warmup_ratio = float(self.trainer_config.get("warmup_ratio", 0.1))
        self.warmup_steps = int(self.trainer_config.get("warmup_steps", 0))

        # tqdm progress
        self.use_tqdm = bool(self.trainer_config.get("use_tqdm", True))
        self.tqdm_update_every = int(self.trainer_config.get("tqdm_update_every", 10))
        self.tqdm_mininterval = float(self.trainer_config.get("tqdm_mininterval", 0.3))

        self.device = self._resolve_device(self.device_request)
        self._configure_reproducible_runtime()
        self.net: Optional[QuantMoEModel] = None

        # Mixed precision / AMP
        self.precision = str(self.trainer_config.get("precision", "fp32") or "fp32").strip().lower()
        self.amp_enabled: bool = False
        self.amp_dtype: Optional[torch.dtype] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        prec_alias = {
            "32": "fp32",
            "float32": "fp32",
            "fp32": "fp32",
            "amp": "amp_fp16",
            "mixed": "amp_fp16",
            "fp16": "amp_fp16",
            "float16": "amp_fp16",
            "amp_fp16": "amp_fp16",
            "mixed_fp16": "amp_fp16",
            "bf16": "amp_bf16",
            "bfloat16": "amp_bf16",
            "amp_bf16": "amp_bf16",
            "mixed_bf16": "amp_bf16",
        }
        self.precision = prec_alias.get(self.precision, self.precision)

        if self.precision == "fp32":
            pass
        elif self.precision == "amp_fp16":
            if self.device.type == "cuda":
                self.amp_enabled = True
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            else:
                print(">>> [AMP] Requested amp_fp16 but CUDA is unavailable; falling back to fp32.")
                self.precision = "fp32"
        elif self.precision == "amp_bf16":
            if self.device.type == "cuda":
                is_supported = True
                try:
                    is_supported = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    is_supported = False
                if is_supported:
                    self.amp_enabled = True
                    self.amp_dtype = torch.bfloat16
                    self.scaler = None  # BF16 typically does not need GradScaler
                else:
                    print(">>> [AMP] Requested amp_bf16 but BF16 is not supported on this CUDA device; falling back to amp_fp16.")
                    self.precision = "amp_fp16"
                    self.amp_enabled = True
                    self.amp_dtype = torch.float16
                    self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            elif self.device.type == "cpu":
                # CPU autocast supports bf16 on many ops (PyTorch >= 1.10).
                self.amp_enabled = True
                self.amp_dtype = torch.bfloat16
                self.scaler = None
            else:
                print(">>> [AMP] Requested amp_bf16 but neither CUDA nor CPU autocast is available; falling back to fp32.")
                self.precision = "fp32"
        else:
            print(f">>> [AMP] Unknown precision='{self.precision}', supported: fp32/amp_fp16/amp_bf16; falling back to fp32.")
            self.precision = "fp32"

        # global step for scheduler
        self.global_step: int = 0

        # ---- diagnostics ----
        # Enable a one-batch backward sanity check at the beginning of fit()
        self.debug_sanity_check = bool(self.trainer_config.get("debug_sanity_check", False))
        # Thresholds for "almost constant" detection
        self.debug_std_eps = float(self.trainer_config.get("debug_std_eps", 1e-8))
        self.debug_grad_eps = float(self.trainer_config.get("debug_grad_eps", 1e-12))

    def _seed_int(self) -> Optional[int]:
        if self.random_seed is None:
            return None
        try:
            return int(self.random_seed)
        except Exception:
            return None

    def _resolve_device(self, requested: str) -> torch.device:
        req = str(requested or "auto").strip().lower()
        if req == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if req.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f">>> [Runtime] Requested device={requested!r} but CUDA is unavailable; falling back to CPU.")
                return torch.device("cpu")
            dev = torch.device(req)
            if dev.index is not None:
                try:
                    n_cuda = int(torch.cuda.device_count())
                except Exception:
                    n_cuda = 0
                if int(dev.index) >= n_cuda:
                    print(
                        f">>> [Runtime] Requested device={requested!r} but only {n_cuda} CUDA device(s) exist; "
                        "falling back to cuda:0."
                    )
                    return torch.device("cuda:0")
            return dev
        return torch.device(req)

    def _configure_reproducible_runtime(self) -> None:
        seed = self._seed_int()
        if seed is not None:
            self._set_global_seed(seed)

        deterministic_enabled = self.deterministic_mode != "off"
        warn_only = self.deterministic_mode == "warn"
        cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")

        if deterministic_enabled:
            if not cublas_workspace:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                cublas_workspace = ":4096:8"
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except TypeError:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                print(f">>> [Runtime] torch deterministic algorithms setup failed: {e}")

        self._runtime_flags = {
            "device_requested": self.device_request,
            "device_resolved": str(self.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "seed": seed,
            "deterministic_mode": self.deterministic_mode,
            "deterministic_algorithms": bool(deterministic_enabled),
            "deterministic_warn_only": bool(warn_only),
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
            "cublas_workspace_config": cublas_workspace,
            "seed_workers": bool(self.seed_workers),
            "train_sampler_mode": self.train_sampler_mode,
            "sampler_diag": bool(self.sampler_diag),
        }

    def _autocast_ctx(self):
        if not self.amp_enabled or self.amp_dtype is None:
            return nullcontext()
        # Prefer torch.autocast; fall back to cuda.amp.autocast on older PyTorch.
        try:
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=True)
        except Exception:
            if self.device.type == "cuda":
                try:
                    return torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True)
                except TypeError:
                    # Older PyTorch may not accept `dtype=` here (defaults to fp16).
                    return torch.cuda.amp.autocast(enabled=True)
            return nullcontext()

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(msg)

    def _set_global_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _loader_generator(self, offset: int = 0) -> Optional[torch.Generator]:
        seed = self._seed_int()
        if seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(int(seed) + int(offset))
        return g

    def _worker_init_fn(self):
        if not self.seed_workers:
            return None
        seed = self._seed_int()
        if seed is None:
            return None

        def _init(worker_id: int) -> None:
            worker_seed = int(seed) + int(worker_id)
            random.seed(worker_seed)
            np.random.seed(worker_seed % (2**32))
            torch.manual_seed(worker_seed)

        return _init

    def log_config_summary(
        self,
        *,
        stage: str = "resolved",
        total_steps: Optional[int] = None,
        warmup_steps: Optional[int] = None,
    ) -> None:
        stage = (stage or "resolved").strip().lower()
        tag = "planned" if stage.startswith("plan") else "resolved"

        mc = self.model_config
        resolved = None
        if self.net is not None:
            resolved = getattr(self.net, "config", None)
        if resolved is None:
            try:
                resolved = QuantMoEConfig(**mc)
            except Exception:
                resolved = None

        def cfg_get(key, default=None):
            if resolved is not None and hasattr(resolved, key):
                return getattr(resolved, key)
            return mc.get(key, default)

        on_off = lambda v: "on" if bool(v) else "off"

        # For "resolved" stage, print full config table directly (no redundant summary)
        if tag == "resolved" and resolved is not None:
            self._print_full_config(resolved, total_steps=total_steps, warmup_steps=warmup_steps)
            return

        # For "planned" stage (before training), print brief summary
        dims = []
        if "context_len" in mc:
            dims.append(f"context_len={mc.get('context_len')}")
        if "num_alphas" in mc:
            dims.append(f"num_alphas={mc.get('num_alphas')}")
        if not dims:
            dims.append("context_len=auto")
            dims.append("num_alphas=auto")

        if self.market_state_path:
            macro_dim = cfg_get("d_macro_input", 0)
            dim_str = str(macro_dim) if int(macro_dim or 0) > 0 else "auto"
            macro_desc = (
                f"external({Path(self.market_state_path).name}, dim={dim_str}, "
                f"shift={self.market_state_shift}, strict={self.market_state_strict})"
            )
        else:
            mode = cfg_get("regime_internal_mode", "long")
            lag = cfg_get("regime_internal_lag", 5)
            macro_desc = f"internal({mode}, lag={lag})"

        emb_type = str(cfg_get("value_embedding_type", "shared_linear"))
        if emb_type == "feature_tokenizer":
            add_id = bool(cfg_get("feature_tokenizer_add_factor_id", False))
            emb_type = f"{emb_type}+id" if add_id else emb_type

        model_parts = [
            f"loss={cfg_get('main_loss', 'mse')}",
            f"d_model={cfg_get('d_model', 'n/a')}",
            f"layers={cfg_get('n_layers', 'n/a')}",
            f"heads={cfg_get('n_heads', 'n/a')}",
            ", ".join(dims),
            f"val_emb={emb_type}",
            f"time_emb={on_off(cfg_get('use_regime_time_embedding', False))}",
            f"factor_gate={on_off(cfg_get('use_regime_factor_gate', False))}",
            f"router={cfg_get('router_mode', 'learned')}",
            f"feat_sel={on_off(cfg_get('use_feature_selection', False))}",
            f"alibi={on_off(cfg_get('use_alibi', False))}",
            f"macro={macro_desc}",
        ]
        macro_drop = float(cfg_get("regime_macro_dropout", 0.0) or 0.0)
        if macro_drop > 0:
            model_parts.append(f"macro_drop={macro_drop:g}")
        print(f">>> [Config:{tag}] model: " + ", ".join(model_parts))

        if self.use_warmup:
            warmup_desc = f"on(ratio={self.warmup_ratio:g})"
        else:
            warmup_desc = "off"

        trainer_parts = [
            f"lr={self.lr:g}",
            f"epochs={self.epochs}",
            f"batch={self.batch_size}",
            f"accum={self.grad_accum_steps}",
            f"early_stop={self.early_stop}",
            f"warmup={warmup_desc}",
            f"precision={self.precision}",
            f"seed={self.random_seed}",
        ]
        if self.checkpoint_metric:
            trainer_parts.append(f"checkpoint={self.checkpoint_metric}:{self.checkpoint_mode}")
        if not (self.early_stop and self.early_stop > 0):
            thr_desc = "off" if self.train_stop_threshold is None else f"{self.train_stop_key}<={self.train_stop_threshold:g}"
            trainer_parts.append(f"train_stop={thr_desc}")
            trainer_parts.append(f"min_epochs={self.min_epochs}")
            trainer_parts.append(f"k={self.consecutive_k}")
        print(f">>> [Config:{tag}] trainer: " + ", ".join(trainer_parts))
    
    def _print_full_config(
        self,
        config,
        *,
        total_steps: Optional[int] = None,
        warmup_steps: Optional[int] = None,
    ) -> None:
        """Print complete model configuration in a structured table format."""
        print("\n" + "=" * 80)
        print("COMPLETE MODEL CONFIGURATION (Resolved)")
        print("=" * 80)
        
        # Group config attributes by category
        categories = {
            "Architecture": [
                ("d_model", "Hidden dimension"),
                ("n_heads", "Attention heads"),
                ("n_layers", "Number of layers"),
                ("d_ff", "FFN dimension"),
                ("num_alphas", "Number of factors (N)"),
                ("context_len", "Sequence length (T)"),
                ("dropout", "Dropout rate"),
                ("initializer_range", "Weight init std"),
            ],
            "Value Embedding": [
                ("value_embedding_type", "Value embedding type"),
                ("feature_tokenizer_bias", "FT bias"),
                ("feature_tokenizer_add_factor_id", "FT add factor ID"),
                ("feature_tokenizer_init_std", "FT init std"),
            ],
            "Loss & Training": [
                ("main_loss", "Primary loss function"),
                ("loss_weights", "Loss weight dict"),
                ("listmle_tau", "ListMLE temperature"),
                ("rank_topk", "RankNet top-k"),
                ("huber_delta", "Huber delta"),
                ("mse_normalize", "MSE normalize flag"),
            ],
            "Regime-Adaptive Time Embedding": [
                ("use_regime_time_embedding", "Enable time embedding"),
                ("time_tau_min", "Min tau (short memory)"),
                ("time_tau_max", "Max tau (long memory)"),
                ("time_tau_init", "Initial tau"),
                ("time_emb_init_std", "Time emb init std"),
                ("time_decay_normalize", "Normalize decay weights"),
            ],
            "Regime-Adaptive Factor Gate": [
                ("use_regime_factor_gate", "Enable factor gate (FiLM)"),
                ("factor_gate_scale", "Gate scale (gamma range)"),
                ("factor_gate_shift_scale", "Gate shift scale (beta)"),
            ],
            "MoE Router": [
                ("router_mode", "Router mode"),
                ("router_noise", "Logit noise std"),
                ("router_temperature", "Softmax temperature"),
                ("router_z_loss_coef", "Z-loss coefficient"),
                ("router_use_layer_summary", "Use layer summary token"),
            ],
            "Feature Selection": [
                ("use_feature_selection", "Enable feature selection"),
                ("selection_reg_lambda", "Sparsity regularization"),
                ("selection_temperature", "Gumbel temperature"),
                ("selection_noise_std", "Selection noise std"),
            ],
            "Positional Encoding": [
                ("use_alibi", "Use ALiBi bias"),
            ],
            "Regime Context Encoder": [
                ("use_external_macro", "Use external macro features"),
                ("d_macro_input", "Macro input dimension"),
                ("regime_macro_dropout", "Macro dropout"),
                ("regime_internal_mode", "Internal mode (short/long)"),
                ("regime_internal_lag", "Internal lag steps"),
                ("regime_internal_use_batch_stats", "Use batch statistics"),
                ("regime_internal_tail_threshold", "Tail threshold"),
            ],
            "Pooling": [
                ("pooling_alpha", "Attention vs mean weight"),
            ],
        }
        
        for cat_name, attrs in categories.items():
            print(f"\n--- {cat_name} ---")
            print(f"{'Parameter':<35} {'Value':<30} {'Description':<25}")
            print("-" * 90)
            for attr_name, desc in attrs:
                value = getattr(config, attr_name, "N/A")
                # Format value for display
                if isinstance(value, dict):
                    value_str = str({k: v for k, v in value.items() if v != 0})
                elif isinstance(value, float):
                    value_str = f"{value:g}"
                elif isinstance(value, bool):
                    value_str = "[+] ON" if value else "[-] OFF"
                else:
                    value_str = str(value)
                print(f"{attr_name:<35} {value_str:<30} {desc:<25}")
        
        # Print trainer config
        print(f"\n--- Trainer Configuration ---")
        print(f"{'Parameter':<35} {'Value':<30} {'Description':<25}")
        print("-" * 90)
        
        # Build warmup description
        if self.use_warmup:
            if total_steps is not None and warmup_steps is not None:
                warmup_val = f"{warmup_steps}/{total_steps} steps"
            elif self.warmup_steps > 0:
                warmup_val = f"{self.warmup_steps} steps"
            else:
                warmup_val = f"{self.warmup_ratio:g} ratio"
        else:
            warmup_val = "disabled"
        
        trainer_attrs = [
            ("lr", self.lr, "Learning rate"),
            ("epochs", self.epochs, "Number of epochs"),
            ("batch_size", self.batch_size, "Batch size (stocks/day)"),
            ("grad_accum_steps", self.grad_accum_steps, "Accumulate K days/step"),
            ("early_stop", self.early_stop, "Early stop patience"),
            ("train_stop_key", self.train_stop_key, "Train stop metric key"),
            ("train_stop_threshold", self.train_stop_threshold, "Train stop threshold"),
            ("min_epochs", self.min_epochs, "Min epochs before stop"),
            ("consecutive_k", self.consecutive_k, "Consecutive epochs to stop"),
            ("use_warmup", self.use_warmup, "Enable LR warmup"),
            ("warmup_config", warmup_val, "Warmup steps/ratio"),
            ("total_steps", total_steps or "N/A", "Total training steps"),
            ("random_seed", self.random_seed, "Random seed"),
            ("num_workers", self.num_workers, "DataLoader workers"),
            ("label_dim", self.label_dim, "Label dimension"),
            ("market_state_path", Path(self.market_state_path).name if self.market_state_path else "None", "Market state file"),
            ("market_state_shift", self.market_state_shift, "Market state shift"),
            ("market_state_strict", self.market_state_strict, "Strict market state"),
        ]
        for name, value, desc in trainer_attrs:
            if isinstance(value, float):
                value_str = f"{value:g}"
            elif isinstance(value, bool):
                value_str = "[+] ON" if value else "[-] OFF"
            else:
                value_str = str(value)
            print(f"{name:<35} {value_str:<30} {desc:<25}")
        
        print("=" * 80 + "\n")

    def _sanity_check_batch(
        self,
        bx: torch.Tensor,
        by: Optional[torch.Tensor],
        bmacro: Optional[torch.Tensor],
        *,
        f_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.net is None:
            return
        self.net.train(True)

        bx_t = torch.nan_to_num(bx, 0.0).to(self.device)
        by_t = None if by is None else by.to(self.device).float()
        macro_t = None if bmacro is None else torch.nan_to_num(bmacro, 0.0).to(self.device).float()

        optimizer.zero_grad(set_to_none=True)
        with self._autocast_ctx():
            out = self.net(bx_t, f_ids, labels=by_t, macro_features=macro_t)
            loss = getattr(out, "loss", None)
        if loss is None:
            self._warn_once(
                "sanity_no_loss",
                ">>> [Sanity] out.loss is None; no backward/step will happen. Check label pipeline & masking.",
            )
            return
        if not torch.isfinite(loss):
            self._warn_once("sanity_nan_loss", f">>> [Sanity] loss is NaN/Inf: {loss}")
            return

        # Important: do NOT call GradScaler.unscale_ here. This sanity check runs before training,
        # and leaving the scaler in an "unscaled" stage can break the first real optimizer step.
        loss_to_backprop = loss.float() if self.scaler is not None else loss
        loss_to_backprop.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0))

        score_std = float("nan")
        if getattr(out, "scores", None) is not None:
            score_std = float(out.scores.detach().float().std(unbiased=False).item())

        label_std = float("nan")
        if by_t is not None:
            label_std = float(by_t.detach().float().std(unbiased=False).item())

        x_last_std = float("nan")
        if bx_t.ndim == 3 and bx_t.shape[0] > 0:
            x_last_std = float(bx_t[:, -1, :].detach().float().std(unbiased=False).item())

        print(
            ">>> [Sanity] "
            f"loss={float(loss.detach().item()):.6f} "
            f"grad_norm={grad_norm:.3e} "
            f"score_std={score_std:.3e} "
            f"label_std={label_std:.3e} "
            f"x_last_std={x_last_std:.3e}"
        )
        optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _extract_datetime(s: Any) -> Optional[pd.Timestamp]:
        if isinstance(s, dict):
            dt = s.get("_datetime", None)
            return None if dt is None else pd.Timestamp(dt)
        return None

    @staticmethod
    def _extract_pos(s: Any) -> Optional[int]:
        if isinstance(s, dict):
            p = s.get("_pos", None)
            return None if p is None else int(p)
        return None

    @staticmethod
    def _wrap_with_datetime(tsds):
        """
        Wrap Qlib TS dataset so each sample carries its own datetime (for macro_features lookup).
        """
        idx = tsds.get_index()
        dates = pd.to_datetime(idx.get_level_values("datetime")).to_numpy()

        class _Wrapped:
            def __init__(self, base, dates_arr):
                self._base = base
                self._dates = dates_arr

            def __len__(self):
                return len(self._base)

            def __getitem__(self, i: int):
                s = self._base[int(i)]
                dt = self._dates[int(i)]
                if isinstance(s, dict):
                    d = dict(s)
                    d.setdefault("_datetime", dt)
                    d.setdefault("_pos", int(i))
                    return d
                if isinstance(s, (tuple, list)):
                    x = s[0] if len(s) > 0 else None
                    y = s[1] if len(s) > 1 else None
                    return {"x": x, "y": y, "_datetime": dt, "_pos": int(i)}
                return {"x": s, "_datetime": dt, "_pos": int(i)}

            def get_index(self):
                return self._base.get_index()

            @property
            def data(self):
                return getattr(self._base, "data", None)

        return _Wrapped(tsds, dates)

    def _ensure_market_state(self) -> None:
        if self._market_state is not None or not self.market_state_path:
            return
        df = load_market_state_df(self.market_state_path)
        self._market_state = make_market_state_lookup(df, shift=self.market_state_shift)
        print(
            ">>> [MarketState] loaded "
            f"{self.market_state_path} (dates={len(df)}, dim={df.shape[1]}), "
            f"shift={self.market_state_shift}, strict={self.market_state_strict}"
        )
        # auto-config model to accept macro features
        self.model_config["use_external_macro"] = True
        self.model_config["d_macro_input"] = int(self._market_state.dim)

    def _macro_from_dates(self, dates: List[pd.Timestamp]) -> Optional[torch.Tensor]:
        if self._market_state is None:
            return None
        arr = lookup_market_state(self._market_state, dates, strict=self.market_state_strict)
        return torch.from_numpy(arr).float()

    # ---------- helpers ----------
    def _make_pbar(self, it, *, desc: str, total: Optional[int] = None, leave: bool = False):
        if not self.use_tqdm:
            return None, it
        if tqdm is None:
            raise RuntimeError("tqdm is not installed. Install tqdm or set trainer_config.use_tqdm=False")
        return tqdm(
            it, desc=desc, total=total, leave=leave, dynamic_ncols=True, mininterval=self.tqdm_mininterval
        ), None

    @staticmethod
    def _as_numpy(x: Any) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _coerce_feature_dim(self, x_np: np.ndarray, *, expected_dim: int, context: str) -> np.ndarray:
        """
        Ensure x has feature dim == expected_dim (num_alphas).

        Some handlers may leak extra channels into x (e.g., packed label) on DK_I/diagnostic paths.
        The core model expects x shaped as [T, num_alphas].
        """
        if x_np.ndim != 2:
            raise RuntimeError(f"{context}: expected x to be 2D [T,F], got shape {tuple(x_np.shape)}")

        expected_dim = int(expected_dim)
        if expected_dim <= 0:
            return x_np

        f_dim = int(x_np.shape[1])
        if f_dim == expected_dim:
            return x_np

        if f_dim > expected_dim:
            if self.label_dim > 0 and f_dim == expected_dim + self.label_dim:
                self._warn_once(
                    f"schema:strip_packed_label:{context}",
                    ">>> [Schema] Detected extra channel(s) in x for "
                    f"{context}: x_dim={f_dim} vs expected num_alphas={expected_dim}. "
                    f"Treating the last {self.label_dim} channel(s) as packed label and stripping them.",
                )
            elif self.strict_feature_schema:
                raise RuntimeError(
                    f"{context}: x_dim={f_dim} > expected num_alphas={expected_dim} and the "
                    f"difference ({f_dim - expected_dim}) is not the packed-label count "
                    f"({self.label_dim}). strict_feature_schema=True forbids silent truncation "
                    "(train==infer feature schema mismatch — check that all feature processors, "
                    "e.g. CSRankAppend, are applied identically on train and inference)."
                )
            else:
                self._warn_once(
                    f"schema:truncate_extra_channels:{context}",
                    ">>> [Schema] Detected extra channel(s) in x for "
                    f"{context}: x_dim={f_dim} vs expected num_alphas={expected_dim}. "
                    "Truncating extra channels to match the trained model.",
                )
            return np.asarray(x_np[:, :expected_dim])

        # f_dim < expected_dim
        raise RuntimeError(
            f"{context}: x_dim={f_dim} < expected num_alphas={expected_dim}. "
            "Check handler schema / feature set consistency between train and inference."
        )

    def _coerce_bx_feature_dim(self, bx: torch.Tensor, *, expected_dim: int, context: str) -> torch.Tensor:
        """Torch variant of `_coerce_feature_dim` for bx shaped as [B,T,F]."""
        if bx.ndim != 3:
            raise RuntimeError(f"{context}: expected bx to be 3D [B,T,F], got shape {tuple(bx.shape)}")

        expected_dim = int(expected_dim)
        if expected_dim <= 0:
            return bx

        f_dim = int(bx.shape[2])
        if f_dim == expected_dim:
            return bx

        if f_dim > expected_dim:
            if self.label_dim > 0 and f_dim == expected_dim + self.label_dim:
                self._warn_once(
                    f"schema:strip_packed_label:{context}",
                    ">>> [Schema] Detected extra channel(s) in bx for "
                    f"{context}: x_dim={f_dim} vs expected num_alphas={expected_dim}. "
                    f"Treating the last {self.label_dim} channel(s) as packed label and stripping them.",
                )
            elif self.strict_feature_schema:
                raise RuntimeError(
                    f"{context}: bx feature dim={f_dim} > expected num_alphas={expected_dim} and the "
                    f"difference ({f_dim - expected_dim}) is not the packed-label count "
                    f"({self.label_dim}). strict_feature_schema=True forbids silent truncation "
                    "(train==infer feature schema mismatch — check that all feature processors, "
                    "e.g. CSRankAppend, are applied identically on train and inference)."
                )
            else:
                self._warn_once(
                    f"schema:truncate_extra_channels:{context}",
                    ">>> [Schema] Detected extra channel(s) in bx for "
                    f"{context}: x_dim={f_dim} vs expected num_alphas={expected_dim}. "
                    "Truncating extra channels to match the trained model.",
                )
            return bx[:, :, :expected_dim]

        raise RuntimeError(
            f"{context}: x_dim={f_dim} < expected num_alphas={expected_dim}. "
            "Check handler schema / feature set consistency between train and inference."
        )

    def _get_num_alphas(self) -> int:
        n = None
        if self.net is not None:
            n = getattr(getattr(self.net, "config", None), "num_alphas", None)
        if n is None:
            n = self.model_config.get("num_alphas", 0)
        return int(n or 0)

    def _macro_tensor_for_day(self, dt: pd.Timestamp, bsz: int) -> Optional[torch.Tensor]:
        if self._market_state is None:
            return None
        bsz = int(bsz)
        if bsz <= 0:
            return None
        m = self._macro_from_dates([pd.Timestamp(dt)] * bsz)
        if m is None:
            return None
        return torch.nan_to_num(m, 0.0).to(self.device).float()

    def _stack_feature_batch_from_row_indices(
        self,
        tsds,
        row_idx: np.ndarray,
        *,
        max_samples: int | None,
        num_alphas: int,
        context: str,
    ) -> Optional[torch.Tensor]:
        row_idx = np.asarray(row_idx, dtype=int)
        if row_idx.size <= 0:
            return None

        if max_samples is not None and int(max_samples) > 0:
            row_idx = row_idx[: int(max_samples)]

        xs: List[torch.Tensor] = []
        for i in row_idx:
            raw_x, _ = self._extract_sample(tsds[int(i)])
            x_np = self._as_numpy(raw_x)
            x_np = self._coerce_feature_dim(x_np, expected_dim=num_alphas, context=context)
            xs.append(torch.from_numpy(np.asarray(x_np)).float())

        if not xs:
            return None

        bx = torch.stack(xs, dim=0)  # [B,T,N]
        return torch.nan_to_num(bx, 0.0).to(self.device)

    @staticmethod
    def _select_spaced_dates(dates: List[pd.Timestamp], k: int) -> List[pd.Timestamp]:
        """
        Select k dates spread over the span (deterministic, avoids consecutive days when possible).
        """
        k = int(k)
        if k <= 0 or not dates:
            return []
        if k >= len(dates):
            return list(dates)

        n = len(dates)
        if k == 1:
            return [dates[-1]]

        # Farthest-point sampling on indices; always include last day (and first if k>1).
        selected = {n - 1, 0}
        while len(selected) < k:
            best_i = None
            best_dist = -1
            for i in range(n):
                if i in selected:
                    continue
                dist = min(abs(i - s) for s in selected)
                if dist > best_dist:
                    best_dist = dist
                    best_i = i
            if best_i is None:
                break
            selected.add(best_i)

        return [dates[i] for i in sorted(selected)]

    def _maybe_warn_market_state(self):
        if self.market_state_path and self._market_state is None:
            print(">>> [MarketState] market_state_path is set but not loaded yet; call fit/predict will load it.")

    def _print_model_summary(self) -> None:
        """Print model architecture and parameter statistics after initialization."""
        if self.net is None:
            return
        
        print("\n" + "=" * 80)
        print("MODEL ARCHITECTURE SUMMARY")
        print("=" * 80)
        
        # Print full model structure
        print(self.net)
        
        print("\n" + "-" * 80)
        print("PARAMETER STATISTICS BY MODULE")
        print("-" * 80)
        
        # Collect parameter stats by top-level module
        module_stats = {}
        total_params = 0
        trainable_params = 0
        
        for name, param in self.net.named_parameters():
            # Extract top-level module name
            top_level = name.split(".")[0]
            num_params = param.numel()
            is_trainable = param.requires_grad
            
            if top_level not in module_stats:
                module_stats[top_level] = {"total": 0, "trainable": 0, "layers": []}
            
            module_stats[top_level]["total"] += num_params
            module_stats[top_level]["layers"].append((name, list(param.shape), num_params, is_trainable))
            
            total_params += num_params
            if is_trainable:
                module_stats[top_level]["trainable"] += num_params
                trainable_params += num_params
        
        # Print summary table
        print(f"{'Module':<25} {'Params':<15} {'Trainable':<15} {'%Total':<10}")
        print("-" * 65)
        
        for mod_name, stats in sorted(module_stats.items(), key=lambda x: -x[1]["total"]):
            pct = 100.0 * stats["total"] / total_params if total_params > 0 else 0
            print(f"{mod_name:<25} {stats['total']:<15,} {stats['trainable']:<15,} {pct:<10.1f}%")
        
        print("-" * 65)
        print(f"{'TOTAL':<25} {total_params:<15,} {trainable_params:<15,} {'100.0':<10}%")
        
        # Print detailed layer breakdown (optional, can be verbose)
        print("\n" + "-" * 80)
        print("DETAILED LAYER BREAKDOWN")
        print("-" * 80)
        print(f"{'Layer Name':<50} {'Shape':<25} {'Params':<12}")
        print("-" * 87)
        
        for name, param in self.net.named_parameters():
            shape_str = str(list(param.shape))
            print(f"{name:<50} {shape_str:<25} {param.numel():<12,}")
        
        print("=" * 80 + "\n")

    @staticmethod
    def _extract_sample(s: Any) -> Tuple[Any, Optional[Any]]:
        """Extract (raw_x, raw_y_or_None) from a single TSDataSampler sample.

        Compatible formats:
        - dict:   {"feature"/"data"/"x": x, "label"/"y": y}
        - tuple:  (x, y, *...)
        - tensor/ndarray: x only
        """
        y = None
        if isinstance(s, dict):
            x = s.get("feature", None) or s.get("data", None) or s.get("x", None)
            y = s.get("label", None)
            if y is None:
                y = s.get("y", None)
        elif isinstance(s, (tuple, list)):
            x = s[0] if len(s) > 0 else None
            y = s[1] if len(s) > 1 else None
        else:
            x = s

        if x is None:
            raise ValueError("TSDataSampler returned a sample with x=None.")
        return x, y

    def _split_packed_label(self, x_np: np.ndarray, y_np: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """If y_np is None but label is packed into x_np, split it explicitly.

        Convention:
        - x_np: [T, F + label_dim]
        - y_np: [label_dim] taken from last timestep, last label_dim channels.
        """
        if y_np is not None:
            return x_np, y_np

        if self.label_dim <= 0:
            # adapter 配置成了“没有 pack label”
            return x_np, None

        if x_np.ndim != 2:
            raise RuntimeError(f"Expect x to be 2D [T,C], got shape {x_np.shape}")

        if x_np.shape[1] <= self.label_dim:
            raise RuntimeError(
                f"Packed-label assumption violated: got x_dim={x_np.shape[1]} <= label_dim={self.label_dim}. "
                f"Please set trainer_config['label_dim']=0 or fix handler schema."
            )

        # 从最后一个 timestep 抽 label，避免 future leak
        y_np = x_np[-1, -self.label_dim:]
        x_np = x_np[:, :-self.label_dim]
        return x_np, y_np

    # ---------- schema validation ----------
    def _validate_train_schema(self, dataset: DatasetH):
        """Validate train schema & return a TS dataset ready for DataLoader.

        检查：
        - label 是否显式存在；
        - 若不存在，则是否可以从 x 的最后 label_dim 个通道中安全拆出。
        """
        ts = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        s0 = ts[0]
        raw_x, raw_y = self._extract_sample(s0)
        x_np = self._as_numpy(raw_x)
        y_np = None if raw_y is None else self._as_numpy(raw_y)

        if x_np.ndim != 2:
            raise RuntimeError(f"Train sample x must be 2D [T,C]; got shape {x_np.shape}")

        if y_np is not None:
            # 显式 label 模式
            if self.label_dim > 0:
                # 允许 [label_dim] 或 [..., label_dim]
                if y_np.ndim == 1 and y_np.shape[0] != self.label_dim:
                    raise RuntimeError(
                        f"Explicit label dim mismatch: y.shape={y_np.shape}, label_dim={self.label_dim}. "
                        f"Set trainer_config['label_dim'] accordingly."
                    )
                if y_np.ndim > 1 and y_np.shape[-1] != self.label_dim:
                    raise RuntimeError(
                        f"Explicit label last-dim mismatch: y.shape={y_np.shape}, label_dim={self.label_dim}. "
                        f"Set trainer_config['label_dim'] accordingly."
                    )
            print(
                f">>> [Schema] Train sample has explicit label: x_dim={x_np.shape[1]}, "
                f"label_dim={self.label_dim} (y.shape={y_np.shape})."
            )
            return ts

        # y_np is None: pack-label 模式
        if self.label_dim <= 0:
            raise RuntimeError(
                "Train sample has no explicit label and label_dim<=0, so adapter will never see labels. "
                "Either: (1) expose label via handler, or (2) set label_dim>0 if label is packed into x."
            )

        if x_np.shape[1] <= self.label_dim:
            raise RuntimeError(
                f"Train sample appears to have no room for packed label: x_dim={x_np.shape[1]}, "
                f"label_dim={self.label_dim}. Check handler output or adjust label_dim."
            )

        feat_dim = x_np.shape[1] - self.label_dim
        print(
            f">>> [Schema] Packed-label detected on train segment: total_dim={x_np.shape[1]}, "
            f"feature_dim={feat_dim}, label_dim={self.label_dim}. "
            f"Adapter will treat the last {self.label_dim} channel(s) at the final time step as label."
        )
        return ts

    # ---------- collate ----------
    @staticmethod
    def _stack_labels_or_none(ys: List[Optional[torch.Tensor]] | None) -> Optional[torch.Tensor]:
        if not ys:
            return None
        if any(t is None for t in ys):
            if all(t is None for t in ys):
                return None
            raise RuntimeError(
                "Mixed None and tensor labels inside a batch. "
                "Check handler DropnaLabel / schema / packed-label settings."
            )
        by = torch.stack([t.view(-1) for t in ys], dim=0)  # [B,label_dim]
        if by.ndim == 2 and by.shape[1] == 1:
            by = by.squeeze(1)  # [B]
        return by

    def _collect_batch_fields(
        self,
        samples: List[Any],
        *,
        with_label: bool,
        require_datetime: bool,
        require_pos: bool,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]] | None, List[pd.Timestamp] | None, List[int] | None]:
        xs: List[torch.Tensor] = []
        ys: List[Optional[torch.Tensor]] | None = [] if with_label else None
        dts: List[pd.Timestamp] | None = [] if require_datetime else None
        pos: List[int] | None = [] if require_pos else None

        for s in samples:
            raw_x, raw_y = self._extract_sample(s)
            x_np = self._as_numpy(raw_x)

            if with_label:
                y_np = None if raw_y is None else self._as_numpy(raw_y)
                x_np, y_np = self._split_packed_label(x_np, y_np)
                assert ys is not None
                ys.append(None if y_np is None else torch.from_numpy(np.asarray(y_np)).float())

            xs.append(torch.from_numpy(np.asarray(x_np)).float())

            if require_datetime:
                dt = self._extract_datetime(s)
                if dt is None:
                    raise RuntimeError("Daily loader requires _datetime; wrap dataset via _wrap_with_datetime.")
                assert dts is not None
                dts.append(dt)

            if require_pos:
                p = self._extract_pos(s)
                if p is None:
                    raise RuntimeError("Daily loader requires _pos; wrap dataset via _wrap_with_datetime.")
                assert pos is not None
                pos.append(p)

        bx = torch.stack(xs, dim=0)
        return bx, ys, dts, pos

    def _collate_train(self, samples: List[Any]):
        need_dt = self._market_state is not None
        bx, ys, dts, _ = self._collect_batch_fields(
            samples,
            with_label=True,
            require_datetime=need_dt,
            require_pos=False,
        )

        bx = bx  # [B,T,F]
        by = self._stack_labels_or_none(ys)
        if self._market_state is None:
            return bx, by
        assert dts is not None
        bmacro = self._macro_from_dates(dts)
        return bx, by, bmacro

    def _collate_eval_daily(self, samples: List[Any]):
        """
        Eval/predict collate for daily samplers that guarantee single-day batches.
        Returns a normalized day key so evaluation can aggregate across chunks of the same day.
        """
        bx, ys, dts, _ = self._collect_batch_fields(
            samples,
            with_label=True,
            require_datetime=True,
            require_pos=False,
        )
        assert dts is not None
        day = pd.to_datetime(dts[0]).normalize()
        if any(pd.to_datetime(dt).normalize() != day for dt in dts[1:]):
            raise RuntimeError("Eval daily sampler produced a batch with mixed dates; expected a single day per batch.")

        bx = bx  # [B,T,F]
        by = self._stack_labels_or_none(ys)

        # macro is optional; if enabled, we look up per-sample datetime then return it alongside day key
        bmacro = None
        if self._market_state is not None:
            bmacro = self._macro_from_dates(dts)

        return bx, by, bmacro, day

    def _collate_feat(self, samples: List[Any]):
        need_dt = self._market_state is not None
        bx, _, dts, _ = self._collect_batch_fields(
            samples,
            with_label=False,
            require_datetime=need_dt,
            require_pos=False,
        )
        if self._market_state is None:
            return bx
        assert dts is not None
        bmacro = self._macro_from_dates(dts)
        return bx, bmacro

    def _collate_feat_with_pos(self, samples: List[Any]):
        """
        Predict-only collate that returns positional indices so we can scatter predictions
        back to `tsds.get_index()` order when using a `batch_sampler` that reorders samples.
        """
        need_dt = self._market_state is not None
        bx, _, dts, pos = self._collect_batch_fields(
            samples,
            with_label=False,
            require_datetime=need_dt,
            require_pos=True,
        )
        assert pos is not None
        pos_t = torch.tensor(pos, dtype=torch.long)
        if self._market_state is None:
            return bx, pos_t
        assert dts is not None
        bmacro = self._macro_from_dates(dts)
        return bx, bmacro, pos_t

    def _daily_chunk_max_bs(self) -> int:
        """Effective max_batch_size for DailyChunkBatchSampler.

        Readout cross-stock attention (use_readout_stock_attn) requires each batch to be ONE FULL
        trading day — otherwise a day with > batch_size stocks is chunk-split and attention would only
        mix within a fragment, silently corrupting predictions. When the flag is on, return an
        effectively-unbounded chunk size so every day is a single batch (B^2 attention at B~300-500 is
        cheap). Otherwise preserve the legacy batch_size chunking.
        """
        cfg = getattr(getattr(self, "net", None), "config", None)
        if bool(getattr(cfg, "use_readout_stock_attn", False)):
            return 1_000_000
        return self.batch_size

    def _make_daily_loader(self, tsds, *, shuffle: bool, train: bool) -> DataLoader:
        """
        使用 FixedDailyBatchSampler 做日度截面 batch.
        - train=True/False: 都用 _collate_train（valid 也需要 label 做监控）。
        """
        if self._market_state is not None:
            tsds = self._wrap_with_datetime(tsds)
        if train and self.train_sampler_mode == "full_daily":
            sampler = DailyChunkBatchSampler(
                tsds,
                max_batch_size=self._daily_chunk_max_bs(),
                shuffle=shuffle,
                seed=self.random_seed,
            )
        else:
            sampler = FixedDailyBatchSampler(tsds, self.batch_size, shuffle=shuffle, seed=self.random_seed)
        return DataLoader(
            dataset=tsds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_train,
            worker_init_fn=self._worker_init_fn(),
            generator=self._loader_generator(offset=17 if train else 23),
        )

    def _make_daily_chunk_loader(self, tsds, *, with_label: bool) -> DataLoader:
        """
        Deterministic daily loader without sampling:
        - each day is fully covered (split into chunks if needed)
        - no up/down-sampling
        """
        tsds = self._wrap_with_datetime(tsds)
        sampler = DailyChunkBatchSampler(tsds, max_batch_size=self._daily_chunk_max_bs())
        return DataLoader(
            dataset=tsds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_eval_daily if with_label else self._collate_feat,
            worker_init_fn=self._worker_init_fn(),
            generator=self._loader_generator(offset=31),
        )

    def _train_sampler_epoch_metrics(self, loader: DataLoader) -> Dict[str, float]:
        if not self.sampler_diag:
            return {}
        sampler = getattr(loader, "batch_sampler", None)
        stats = getattr(sampler, "last_epoch_stats", None)
        if not isinstance(stats, dict) or not stats:
            if isinstance(sampler, DailyChunkBatchSampler):
                groups = getattr(sampler, "daily_groups", []) or []
                total = int(sum(len(g) for g in groups))
                stats = {
                    "epoch": 0,
                    "num_days": int(len(groups)),
                    "num_batches": int(len(sampler)),
                    "batch_size": int(getattr(sampler, "max_batch_size", self.batch_size)),
                    "total_source_samples": total,
                    "total_draws": total,
                    "unique_samples": total,
                    "coverage_ratio": 1.0 if total > 0 else float("nan"),
                    "coverage_gap_vs_full": 0.0 if total > 0 else float("nan"),
                    "duplicate_draws": 0,
                    "duplicate_rate": 0.0,
                    "downsample_days": 0,
                    "upsample_days": 0,
                    "exact_days": int(len(groups)),
                    "downsample_day_ratio": 0.0,
                    "upsample_day_ratio": 0.0,
                }
            else:
                return {}

        self._last_train_sampler_stats = dict(stats)
        out: Dict[str, float] = {}
        for k, v in stats.items():
            if isinstance(v, bool):
                out[f"sampler_{k}"] = float(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    out[f"sampler_{k}"] = fv
        return out

    # ---------- net init & metrics ----------
    def _init_net(self, bx: torch.Tensor) -> None:
        # bx: [B,T,F]
        T, F = int(bx.shape[1]), int(bx.shape[2])
        self.model_config.update({"context_len": T, "num_alphas": F})
        conf = QuantMoEConfig(**self.model_config)
        self.net = QuantMoEModel(conf).to(self.device)
        
        # Print model architecture and parameter statistics
        self._print_model_summary()

    @staticmethod
    def _avg(meters: Dict[str, float], n: int) -> Dict[str, float]:
        n = max(1, int(n))
        return {k: v / n for k, v in meters.items()}

    def _log_metrics(self, step: int, prefix: str, metrics: Dict[str, float]) -> None:
        """
        - loss_main    → 额外记 {prefix}/main
        - loss_listmle → 额外记 {prefix}/listmle
        - loss_mse     → 额外记 {prefix}/mse
        - loss_ic      → 额外记 {prefix}/ic = -loss_ic （loss_ic = -IC）
        """
        m = dict(metrics)
        if "loss_main" in m:
            m.setdefault("main", m["loss_main"])
        if "loss_listmle" in m:
            m.setdefault("listmle", m["loss_listmle"])
        if "loss_mse" in m:
            m.setdefault("mse", m["loss_mse"])
        if "loss_ic" in m:
            # loss_ic = -IC, so monitored IC should be -loss_ic within [-1, 1]
            m.setdefault("ic", -float(m["loss_ic"]))

        # Required diagnostic matrix prefixes. These mirrors are intentionally
        # omitted when the source metric is absent, so disabled modules do not
        # create NaN placeholders (e.g. no time/tau_* when time embedding is off).
        primary: Dict[str, float] = {}
        grouped: Dict[str, float] = {}
        for k, v in m.items():
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue

            primary[f"{prefix}/{k}"] = fv

            if k.startswith("router_layer_"):
                grouped[f"router/{prefix}_{k}"] = fv
            elif k.startswith("time_tau") or k == "time_half_life":
                grouped[f"time/{prefix}_{k}"] = fv
            elif k.startswith("factor_gate") or k.startswith("film_"):
                grouped[f"film/{prefix}_{k}"] = fv
            elif k.startswith("expert_"):
                grouped[f"expert/{prefix}_{k}"] = fv
            elif k.startswith("sampler_"):
                grouped[f"sampler/{prefix}_{k}"] = fv
            elif k.startswith("time_embedding_") or k.startswith("temporal_"):
                grouped[f"temporal/{prefix}_{k}"] = fv
            elif k in {
                "loss_gap",
                "rank_ic_gap",
                "post_peak_decay",
                "score_std",
                "label_std",
                "score_label_std_ratio",
                "grad_norm",
                "optimizer_steps",
                "grad_attempt_steps",
                "grad_skipped_steps",
                "grad_nonfinite_steps",
                "grad_skipped_rate",
                "grad_nonfinite_rate",
                "grad_nonfinite_tensors",
                "grad_nonfinite_values",
                "lr",
            } or k.startswith("grad_norm_"):
                grouped[f"opt/{prefix}_{k}"] = fv

        if primary:
            try:
                R.log_metrics(step=step, **primary)
            except Exception:
                pass

        if grouped:
            try:
                R.log_metrics(step=step, **grouped)
            except Exception:
                pass

    def _monitor(self, valid_metrics: Dict[str, float]) -> float:
        """用于 early stopping 的单一 score（越大越好）."""
        # Prefer daily full-cross-section RankIC when available (valid uses daily-chunk loader).
        if "rank_ic_daily" in valid_metrics:
            return float(valid_metrics["rank_ic_daily"])
        if "rank_ic" in valid_metrics:
            return float(valid_metrics["rank_ic"])
        if "loss_ic" in valid_metrics:
            return -float(valid_metrics["loss_ic"])
        return -float(valid_metrics.get("loss_total", 0.0))

    def _checkpoint_metric_value(
        self,
        train_metrics: Dict[str, float],
        valid_metrics: Optional[Dict[str, float]],
    ) -> Optional[float]:
        metric = str(self.checkpoint_metric or "").strip().lower()
        if not metric:
            return None
        metric = metric.replace("/", "_").replace(".", "_").replace("-", "_")

        source = valid_metrics if valid_metrics is not None else train_metrics
        if metric in {"valid_rank_ic", "valid_rankic", "valid_daily_rank_ic", "daily_rank_ic"}:
            source = valid_metrics
            keys = ["rank_ic_daily", "rank_ic"]
        elif metric in {"valid_ic", "valid_daily_ic", "daily_ic"}:
            source = valid_metrics
            keys = ["ic_pearson_daily", "ic_raw", "loss_ic"]
        elif metric.startswith("valid_"):
            source = valid_metrics
            keys = [metric[len("valid_"):]]
        elif metric.startswith("train_"):
            source = train_metrics
            keys = [metric[len("train_"):]]
        elif metric in {"rank_ic", "rankic"}:
            keys = ["rank_ic_daily", "rank_ic"]
        elif metric in {"ic", "ic_raw"}:
            keys = ["ic_pearson_daily", "ic_raw", "loss_ic"]
        else:
            keys = [metric]

        if source is None:
            return None
        for key in keys:
            if key not in source:
                continue
            try:
                val = float(source[key])
            except Exception:
                continue
            if key == "loss_ic" and metric in {"valid_ic", "valid_daily_ic", "daily_ic", "ic", "ic_raw"}:
                val = -val
            if np.isfinite(val):
                return val
        return None

    def _checkpoint_improved(self, value: float, best: float) -> bool:
        if self.checkpoint_mode == "min":
            return value < best - self.checkpoint_min_delta
        return value > best + self.checkpoint_min_delta

    def _grad_group_name(self, name: str) -> str:
        if name.startswith("regime_encoder."):
            return "grad_norm_regime_encoder"
        if name.startswith("time_embedding."):
            return "grad_norm_time_embedding"
        if name.startswith("factor_gate."):
            return "grad_norm_factor_film"
        if ".router." in name or ".layer_summary_proj." in name:
            return "grad_norm_router"
        if ".time_expert." in name:
            return "grad_norm_time_expert"
        if ".factor_expert." in name:
            return "grad_norm_factor_expert"
        if name.startswith("factor_pooling.") or name.startswith("head."):
            return "grad_norm_pooling_head"
        return "grad_norm_other"

    def _grad_norm_stats(self) -> Tuple[float, Dict[str, float], int, int]:
        """Return pre-clipping grad norms and non-finite counts for diagnostics.

        Norms are accumulated with FP64 reductions. This avoids monitor-only
        overflow in the L2 sum for large-but-finite FP32/AMP gradients.
        Non-finite tensors are reported separately and are not folded into the
        finite norm average.
        """
        assert self.net is not None
        group_sq: Dict[str, float] = defaultdict(float)
        total_sq = 0.0
        nonfinite_tensors = 0
        nonfinite_values = 0

        for name, p in self.net.named_parameters():
            if p.grad is None:
                continue
            try:
                g = p.grad.detach()
                finite = torch.isfinite(g)
                if not bool(finite.all().item()):
                    nonfinite_tensors += 1
                    try:
                        nonfinite_values += int((~finite).sum().item())
                    except Exception:
                        pass
                    continue

                # `dtype=torch.float64` computes the norm in double precision
                # without changing the gradient tensor itself.
                n = torch.linalg.vector_norm(g, ord=2, dtype=torch.float64)
                nsq = float((n * n).item())
                if not math.isfinite(nsq):
                    nonfinite_tensors += 1
                    continue
                total_sq += nsq
                group_sq[self._grad_group_name(name)] += nsq
            except Exception:
                continue

        total_norm = float(math.sqrt(max(total_sq, 0.0)))
        group_norms = {k: float(math.sqrt(max(v, 0.0))) for k, v in group_sq.items()}
        return total_norm, group_norms, nonfinite_tensors, nonfinite_values

    def _clip_grads_by_total_norm(self, total_norm: float, max_norm: float = 1.0) -> None:
        """Clip gradients using a precomputed finite total norm."""
        if not math.isfinite(total_norm) or total_norm <= 0:
            return
        clip_coef = float(max_norm) / (total_norm + 1e-6)
        if clip_coef >= 1.0:
            return
        assert self.net is not None
        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

    def _finish_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        opt_meters: Dict[str, float],
        opt_counts: Dict[str, int],
    ) -> bool:
        """Unscale/clip/step and record optimization diagnostics.

        Returns True only when an optimizer step was actually applied.
        AMP overflow steps are intentionally excluded from grad_norm averages;
        they are exposed through grad_nonfinite_* metrics instead.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

        grad_norm, module_grad_norms, nonfinite_tensors, nonfinite_values = self._grad_norm_stats()
        grad_finite = (
            nonfinite_tensors == 0
            and math.isfinite(grad_norm)
        )

        opt_meters["grad_attempt_steps"] += 1.0
        opt_counts["grad_attempt_steps"] += 1

        if grad_finite:
            self._clip_grads_by_total_norm(grad_norm, max_norm=1.0)
            opt_meters["grad_norm"] += float(grad_norm)
            opt_counts["grad_norm"] += 1
            for gk, gv in module_grad_norms.items():
                opt_meters[gk] += float(gv)
                opt_counts[gk] += 1
        else:
            opt_meters["grad_nonfinite_steps"] += 1.0
            opt_counts["grad_nonfinite_steps"] += 1
            opt_meters["grad_nonfinite_tensors"] += float(nonfinite_tensors)
            opt_counts["grad_nonfinite_tensors"] += 1
            opt_meters["grad_nonfinite_values"] += float(nonfinite_values)
            opt_counts["grad_nonfinite_values"] += 1

        step_skipped = False
        if self.scaler is not None:
            prev_scale = float(self.scaler.get_scale())
            self.scaler.step(optimizer)
            self.scaler.update()
            new_scale = float(self.scaler.get_scale())
            step_skipped = (new_scale < prev_scale) or (not grad_finite)
            if step_skipped:
                opt_meters["grad_skipped_steps"] += 1.0
                opt_counts["grad_skipped_steps"] += 1
        else:
            if grad_finite:
                optimizer.step()
            else:
                step_skipped = True
                opt_meters["grad_skipped_steps"] += 1.0
                opt_counts["grad_skipped_steps"] += 1

        if step_skipped:
            if not grad_finite:
                self._warn_once(
                    "nonfinite_grad_norm",
                    f">>> [Warn] non-finite gradients detected "
                    f"(tensors={nonfinite_tensors}, values={nonfinite_values}); "
                    "optimizer step skipped and excluded from grad_norm metrics.",
                )
            return False

        if grad_norm <= self.debug_grad_eps:
            self._warn_once(
                "zero_grad_norm",
                f">>> [Warn] grad_norm approx 0 ({grad_norm:.3e}); parameters may not be updating.",
            )

        if scheduler is not None:
            scheduler.step()
        self.global_step += 1
        return True

    # ---------- epoch loop ----------
    def _run_epoch(
        self,
        loader: DataLoader,
        f_ids: torch.Tensor,
        *,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        train: bool,
        desc: str,
    ) -> Dict[str, float]:
        assert self.net is not None
        self.net.train(train)

        meters = defaultdict(float)
        n_batches = 0

        try:
            total = len(loader)
        except Exception:
            total = None
        pbar, _ = self._make_pbar(loader, desc=desc, total=total, leave=False)

        skip_invalid_label = 0
        skip_nan_loss = 0
        skip_no_loss = 0
        opt_steps = 0
        opt_meters = defaultdict(float)
        opt_counts = defaultdict(int)
        daily_buffer = defaultdict(lambda: {"p": [], "y": []}) if not train else None
        accum_steps = self.grad_accum_steps if (train and optimizer is not None) else 1
        accum_count = 0

        iterator = pbar if pbar is not None else loader
        for step, batch in enumerate(iterator):
            day_key = None
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                bx, by, bmacro, day_key = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 3:
                bx, by, bmacro = batch
            else:
                bx, by = batch
                bmacro = None
            bx_t = torch.nan_to_num(bx, 0.0).to(self.device)  # [B,T,F]
            if self.label_dim > 0 and by is None:
                raise RuntimeError(
                    "by is None in training/valid loop while label_dim>0; "
                    "check schema validation and label_dim."
                )
            by_t = None if by is None else by.to(self.device).float()
            macro_t = None if bmacro is None else torch.nan_to_num(bmacro, 0.0).to(self.device).float()

            # 屏蔽非法 label
            if by_t is not None:
                valid = torch.isfinite(by_t)
                if valid.sum().item() < 2:
                    skip_invalid_label += 1
                    if pbar is not None and (step + 1) % max(1, self.tqdm_update_every) == 0:
                        pbar.set_postfix({"skip_lbl": skip_invalid_label}, refresh=False)
                    continue
                bx_t = bx_t[valid]
                by_t = by_t[valid]
                if macro_t is not None:
                    macro_t = macro_t[valid]

            if train and optimizer is not None and accum_count == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with self._autocast_ctx():
                    # Note: date_ids removed - regime signal is computed from internal statistics
                    out = self.net(bx_t, f_ids, labels=by_t, macro_features=macro_t)
                    loss = getattr(out, "loss", None)

                if train and optimizer is not None and loss is not None:
                    if not torch.isfinite(loss):
                        skip_nan_loss += 1
                        continue
                    # Accumulate gradients across K (shuffled) daily cross-section microbatches.
                    # Scale to approximate mean gradient (large-batch) rather than sum.
                    loss_to_backprop = loss / float(accum_steps)
                    if self.scaler is not None:
                        self.scaler.scale(loss_to_backprop).backward()
                    else:
                        loss_to_backprop.backward()
                    accum_count += 1

                    do_step = accum_count >= accum_steps
                    if do_step:
                        # If the last update has fewer than K microbatches, rescale grads so the
                        # effective gradient is still an average over the available microbatches.
                        rem = accum_count
                        if rem > 0 and rem < accum_steps:
                            scale = float(accum_steps) / float(rem)
                            for p in self.net.parameters():
                                if p.grad is None:
                                    continue
                                p.grad.mul_(scale)

                        step_taken = self._finish_optimizer_step(
                            optimizer,
                            scheduler,
                            opt_meters,
                            opt_counts,
                        )
                        if step_taken:
                            opt_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                elif train and optimizer is not None and loss is None:
                    skip_no_loss += 1

            n_batches += 1

            # ---- one-time diagnostics for "loss never moves" ----
            # If scores/x/labels are almost constant in a batch, list-wise ranking loss can produce near-zero updates.
            if train and optimizer is not None:
                try:
                    if by_t is not None:
                        y_std = float(by_t.detach().float().std(unbiased=False).item())
                        if y_std <= self.debug_std_eps:
                            self._warn_once(
                                "label_almost_constant",
                                f">>> [Warn] label std approx 0 ({y_std:.3e}); ranking loss has little signal in-batch.",
                            )
                    if getattr(out, "scores", None) is not None:
                        p_std = float(out.scores.detach().float().std(unbiased=False).item())
                        if p_std <= self.debug_std_eps:
                            self._warn_once(
                                "score_almost_constant",
                                f">>> [Warn] score std approx 0 ({p_std:.3e}); check data variability / model wiring.",
                            )
                    if bx_t.ndim == 3 and bx_t.shape[0] > 0:
                        x_std = float(bx_t[:, -1, :].detach().float().std(unbiased=False).item())
                        if x_std <= self.debug_std_eps:
                            self._warn_once(
                                "x_almost_constant",
                                f">>> [Warn] x(last-step) std approx 0 ({x_std:.3e}); features may be all-0 after Fillna.",
                            )
                except Exception:
                    pass

            # 聚合模型内部的 metrics（loss_total / loss_listmle / loss_ic / aux / sparsity ...）
            if getattr(out, "metrics", None):
                for k, v in out.metrics.items():
                    meters[k] += float(v)

            # 适配器侧计算 IC / RankIC（基于当前 label）
            if by_t is not None and getattr(out, "scores", None) is not None:
                stock_scores = out.scores
                p_vec = stock_scores.view(-1).detach().cpu().numpy()
                y_vec = by_t.view(-1).detach().cpu().numpy()
                if p_vec.size >= 2:
                    p_std = float(np.std(p_vec))
                    meters["score_std"] += p_std
                else:
                    p_std = np.nan
                if y_vec.size >= 2:
                    y_std = float(np.std(y_vec))
                    meters["label_std"] += y_std
                else:
                    y_std = np.nan
                if np.isfinite(p_std) and np.isfinite(y_std) and y_std > 0:
                    meters["score_label_std_ratio"] += float(p_std / y_std)

                # For eval loaders that split a day into chunks, aggregate (p,y) by date first,
                # then compute IC / RankIC on the full daily cross-section.
                if not train and day_key is not None and daily_buffer is not None:
                    buf = daily_buffer[pd.to_datetime(day_key).normalize()]
                    buf["p"].append(p_vec)
                    buf["y"].append(y_vec)
                else:
                    if p_vec.size >= 2 and y_vec.size >= 2:
                        if np.std(p_vec) > 0 and np.std(y_vec) > 0:
                            ic = np.corrcoef(p_vec, y_vec)[0, 1]
                            meters["ic_pearson_batch"] += float(ic)
                            meters["ic_raw"] += float(ic)  # backward-compatible alias (batch-level)

                        rank_p = pd.Series(p_vec).rank().to_numpy()
                        rank_y = pd.Series(y_vec).rank().to_numpy()
                        if np.std(rank_p) > 0 and np.std(rank_y) > 0:
                            ric = np.corrcoef(rank_p, rank_y)[0, 1]
                            meters["rank_ic_batch"] += float(ric)
                            meters["rank_ic"] += float(ric)  # backward-compatible alias (batch-level)

            if getattr(out, "avg_gate_entropy", None) is not None:
                meters["gate_entropy"] += float(out.avg_gate_entropy)
            if getattr(out, "avg_time_ratio", None) is not None:
                meters["time_ratio"] += float(out.avg_time_ratio)
                # Router Collapse Warning: detect extreme time_ratio
                tr = float(out.avg_time_ratio)
                if train and (tr < 0.1 or tr > 0.9):
                    self._warn_once(
                        "router_collapse",
                        f">>> [Warn] Router may be collapsing: time_ratio={tr:.3f}. "
                        f"Expected range [0.2, 0.8]. Consider increasing router_z_loss_coef.",
                    )
            if getattr(out, "avg_gate_entropy", None) is not None:
                ge = float(out.avg_gate_entropy)
                if train and ge < 0.2:
                    self._warn_once(
                        "router_low_entropy",
                        f">>> [Warn] Router entropy too low: {ge:.3f}. "
                        f"May indicate collapse to single expert. Check router_z_loss_coef.",
                    )
            if getattr(out, "selected_mask", None) is not None:
                meters["active_feat_ratio"] += float(out.selected_mask.mean().item())

            if pbar is not None and ((step + 1) % max(1, self.tqdm_update_every) == 0):
                avg = {k: meters[k] / max(1, n_batches) for k in meters}
                if train and optimizer is not None:
                    avg["lr"] = optimizer.param_groups[0]["lr"]
                avg["skip_lbl"] = skip_invalid_label
                avg["skip_nan"] = skip_nan_loss
                avg["skip_noloss"] = skip_no_loss
                if train and optimizer is not None:
                    avg["opt"] = opt_steps
                pbar.set_postfix(avg, refresh=False)

        if pbar is not None:
            pbar.close()

        # Flush last partial gradient accumulation (if the epoch ends before reaching K).
        if train and optimizer is not None and accum_count > 0:
            scale = float(accum_steps) / float(accum_count) if accum_count < accum_steps else 1.0
            if scale != 1.0:
                for p in self.net.parameters():
                    if p.grad is None:
                        continue
                    p.grad.mul_(scale)
            step_taken = self._finish_optimizer_step(
                optimizer,
                scheduler,
                opt_meters,
                opt_counts,
            )
            if step_taken:
                opt_steps += 1
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        avg = self._avg(meters, n_batches)
        step_count_keys = {
            "grad_attempt_steps",
            "grad_skipped_steps",
            "grad_nonfinite_steps",
        }
        for k, v in opt_meters.items():
            if k in step_count_keys:
                continue
            c = int(opt_counts.get(k, 0))
            if c > 0:
                avg[k] = float(v / c)
        if train and optimizer is not None:
            grad_attempt_steps = int(opt_meters.get("grad_attempt_steps", 0.0))
            grad_skipped_steps = int(opt_meters.get("grad_skipped_steps", 0.0))
            grad_nonfinite_steps = int(opt_meters.get("grad_nonfinite_steps", 0.0))
            avg["optimizer_steps"] = float(opt_steps)
            avg["grad_attempt_steps"] = float(grad_attempt_steps)
            avg["grad_skipped_steps"] = float(grad_skipped_steps)
            avg["grad_nonfinite_steps"] = float(grad_nonfinite_steps)
            if grad_attempt_steps > 0:
                avg["grad_skipped_rate"] = float(grad_skipped_steps / grad_attempt_steps)
                avg["grad_nonfinite_rate"] = float(grad_nonfinite_steps / grad_attempt_steps)
        if train and optimizer is not None:
            try:
                avg["lr"] = float(optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        if train and optimizer is not None and opt_steps == 0:
            self._warn_once(
                "no_optimizer_steps",
                ">>> [Warn] No optimizer steps happened in this epoch (opt_steps=0). "
                "Common causes: labels missing/filtered, loss=None/NaN, or gradients are zero.",
            )

        # If we buffered daily chunks, compute daily IC / RankIC on the full cross-section per day.
        # Expose explicit names to avoid confusion with batch-level metrics.
        if not train and daily_buffer:
            ics = []
            rics = []
            for _, v in daily_buffer.items():
                p = np.concatenate(v["p"], axis=0) if v["p"] else None
                y = np.concatenate(v["y"], axis=0) if v["y"] else None
                if p is None or y is None or p.size < 2 or y.size < 2:
                    continue
                if np.std(p) > 0 and np.std(y) > 0:
                    ics.append(float(np.corrcoef(p, y)[0, 1]))
                rp = pd.Series(p).rank().to_numpy()
                ry = pd.Series(y).rank().to_numpy()
                if np.std(rp) > 0 and np.std(ry) > 0:
                    rics.append(float(np.corrcoef(rp, ry)[0, 1]))

            ic_daily = float(np.mean(ics)) if ics else float("nan")
            ric_daily = float(np.mean(rics)) if rics else float("nan")
            avg["ic_pearson_daily"] = ic_daily
            avg["rank_ic_daily"] = ric_daily

            # Backward-compatible aliases (these are daily-level on valid/test with chunk loader)
            avg["ic_raw"] = ic_daily
            avg["rank_ic"] = ric_daily

        return avg

    # ---------- Qlib API ----------
    def fit(self, dataset: DatasetH, evals_result=dict()):
        if "seed" in self.trainer_config:
            self._set_global_seed(self.random_seed)
        self._ensure_market_state()
        # 1) Train schema & TSDS
        train_tsds = self._validate_train_schema(dataset)
        train_loader = self._make_daily_loader(train_tsds, shuffle=True, train=True)

        # 2) Valid set (default: DK_I)
        valid_loader = None
        valid_err: Optional[Exception] = None
        valid_data_keys: List[str] = [self.valid_data_key]
        if not self.strict_valid_data_key:
            for k in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
                k = str(k)
                if k not in valid_data_keys:
                    valid_data_keys.append(k)

        for data_key in valid_data_keys:
            try:
                valid_tsds = dataset.prepare("valid", col_set=["feature", "label"], data_key=data_key)
                # Sanity check: ensure label exists (or is packed in x) when we want valid metrics.
                s0 = valid_tsds[0]
                raw_x, raw_y = self._extract_sample(s0)
                x_np = self._as_numpy(raw_x)
                y_np = None if raw_y is None else self._as_numpy(raw_y)
                _, y_np = self._split_packed_label(x_np, y_np)
                if self.label_dim > 0 and y_np is None:
                    raise RuntimeError(f"Valid segment has no label under data_key={data_key}")

                # Valid should be evaluated on full daily cross-sections without sampling.
                valid_loader = self._make_daily_chunk_loader(valid_tsds, with_label=True)
                if data_key != DataHandlerLP.DK_I:
                    print(
                        ">>> [Valid] DK_I does not provide label; falling back to DK_L "
                        "(valid labels may be learn-processed by the handler)."
                    )
                break
            except Exception as e:
                valid_err = e
                valid_loader = None

        if valid_loader is None and valid_err is not None:
            if self.strict_valid_data_key:
                print(
                    f">>> [Valid] disabled: failed to prepare valid loader under data_key={self.valid_data_key} "
                    f"({valid_err})"
                )
            else:
                print(f">>> [Valid] disabled: failed to prepare valid loader ({valid_err})")

        # 3) Init network from first batch
        first = next(iter(train_loader))
        if isinstance(first, (tuple, list)) and len(first) == 3:
            bx0, by0, bmacro0 = first
        else:
            bx0, by0 = first
            bmacro0 = None
        if self.label_dim > 0 and by0 is None:
            raise RuntimeError(
                "First training batch has by=None while label_dim>0. "
                "Check handler schema or set trainer_config['label_dim']=0 if there is truly no label."
            )

        if self.net is None:
            self._init_net(bx0)

        assert self.net is not None
        # h-20260610-002 follow-up: optionally exclude the stock expert's projections from weight
        # decay so its query (Wq) can WARM (grow) under the de-mean gradient instead of being
        # pinned to ~0 by wd (the cold-query cause of the uniform-attention collapse; KEY-SVD
        # showed keys are structured/peakable, so the blocker is operator-state, not input).
        # tau and ||Wq|| are redundant, so freeing Wq from wd IS "set a suitable temperature",
        # done the only way that survives training. Default off => optimizer byte-identical.
        _adamw_wd = 0.01  # torch AdamW default, preserved for the decayed group
        _no_wd_scope = str(self.trainer_config.get("expert_no_wd_scope", "") or "").strip().lower()
        if bool(self.trainer_config.get("stock_expert_no_wd", False)) and not _no_wd_scope:
            _no_wd_scope = "stock_expert"
        if _no_wd_scope:
            _match = ["stock_expert", "factor_expert", "time_expert"] if _no_wd_scope in {"all", "all_experts"} \
                else [_no_wd_scope]
            no_wd, decayed = [], []
            for _name, _p in self.net.named_parameters():
                if not _p.requires_grad:
                    continue
                (no_wd if any(m in _name for m in _match) else decayed).append(_p)
            optimizer = optim.AdamW(
                [
                    {"params": decayed, "weight_decay": _adamw_wd},
                    {"params": no_wd, "weight_decay": 0.0},
                ],
                lr=self.lr,
            )
            print(f">>> [wd-exclusion] scope={_match} -> {len(no_wd)} param tensors excluded from weight_decay "
                  f"(decayed={len(decayed)})")
        else:
            optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        f_ids = torch.arange(int(self.model_config["num_alphas"]), device=self.device)

        if self.debug_sanity_check:
            self._sanity_check_batch(bx0, by0, bmacro0, f_ids=f_ids, optimizer=optimizer)

        # Warmup scheduler
        total_training_steps = None
        warmup_steps = None
        if self.use_warmup:
            try:
                num_update_steps_per_epoch = len(train_loader)
            except Exception:
                num_update_steps_per_epoch = 1
            # Scheduler steps should match optimizer.step() calls (not microbatches) when using grad accumulation.
            num_optimizer_steps_per_epoch = int(math.ceil(float(num_update_steps_per_epoch) / float(self.grad_accum_steps)))
            total_training_steps = max(1, self.epochs * max(1, num_optimizer_steps_per_epoch))

            if self.warmup_steps > 0:
                warmup_steps = self.warmup_steps
            else:
                warmup_steps = int(total_training_steps * self.warmup_ratio)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
            )
        else:
            scheduler = None

        self.log_config_summary(
            stage="resolved",
            total_steps=total_training_steps,
            warmup_steps=warmup_steps,
        )

        use_valid_early_stop = bool(self.early_stop and self.early_stop > 0)
        if use_valid_early_stop and valid_loader is None:
            self._warn_once(
                "early_stop_no_valid",
                ">>> [Warn] early_stop is set but valid loader is unavailable; "
                "falling back to train-loss threshold stopping.",
            )
            use_valid_early_stop = False

        best_state = None
        best_score = float("-inf")  # used in valid early stop mode (maximize)
        best_train = float("inf")  # used in train threshold mode (minimize)
        checkpoint_enabled = bool(self.checkpoint_metric)
        checkpoint_state = None
        checkpoint_best = float("inf") if self.checkpoint_mode == "min" else float("-inf")
        checkpoint_epoch: Optional[int] = None
        bad = 0
        good = 0

        # 训练曲线缓存，用于报告里的“训练过程诊断”
        rec = R.get_recorder()
        if rec is not None:
            try:
                rec.save_objects(runtime_flags=dict(self._runtime_flags))
            except Exception as e:
                print(f">>> [Runtime] save runtime_flags failed: {e}")
        train_curve = None
        main_loss = str(self.model_config.get("main_loss", "mse")).lower()
        if main_loss == "mle":
            main_loss = "listmle"
        if rec is not None:
            train_curve = {
                "epoch": [],
                "train_main": [],
                "train_listmle": [],
                "train_mse": [],
                "train_ic": [],
                "train_rank_ic": [],
                "train_score_std": [],
                "train_label_std": [],
                "train_grad_norm": [],
                "train_grad_nonfinite_steps": [],
                "train_grad_nonfinite_rate": [],
                "train_grad_skipped_steps": [],
                "train_grad_skipped_rate": [],
                "train_optimizer_steps": [],
                "train_lr": [],
                "valid_main": [],
                "valid_listmle": [],
                "valid_mse": [],
                "valid_rank_ic": [],
                "valid_ic": [],
                "valid_score_std": [],
                "valid_label_std": [],
            }

        if use_valid_early_stop:
            stop_desc = f"valid_early_stop(patience={self.early_stop})"
        else:
            thr_desc = "off" if self.train_stop_threshold is None else f"{self.train_stop_key}<={self.train_stop_threshold:g}"
            stop_desc = f"train_threshold({thr_desc}, min_epochs={self.min_epochs}, k={self.consecutive_k})"
        print(f">>> [Train] epochs={self.epochs}, stop={stop_desc}")
        epoch_iter = range(self.epochs)
        if self.use_tqdm and trange is not None:
            epoch_iter = trange(self.epochs, desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_iter:
            train_sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                try:
                    train_sampler.set_epoch(epoch)
                except Exception:
                    pass
            tr = self._run_epoch(
                train_loader,
                f_ids,
                optimizer=optimizer,
                scheduler=scheduler,
                train=True,
                desc=f"Train e{epoch + 1:02d}",
            )
            tr.update(self._train_sampler_epoch_metrics(train_loader))
            self._log_metrics(epoch, "train", tr)
            print(f"| Train {epoch + 1:02d} | " + " | ".join(f"{k}:{v:.6f}" for k, v in tr.items()))

            va = None
            if valid_loader is not None:
                va = self._run_epoch(
                    valid_loader,
                    f_ids,
                    optimizer=None,
                    scheduler=None,
                    train=False,
                    desc=f"Valid e{epoch + 1:02d}",
                )
                self._log_metrics(epoch, "valid", va)
                print(f"| Valid {epoch + 1:02d} | " + " | ".join(f"{k}:{v:.6f}" for k, v in va.items()))

            # 记录训练曲线
            if train_curve is not None:
                train_curve["epoch"].append(int(epoch + 1))
                # main loss
                main_key = f"loss_{main_loss}"
                train_curve["train_main"].append(
                    float(tr.get("loss_main", tr.get(main_key, np.nan)))
                )
                # train
                train_curve["train_listmle"].append(float(tr.get("loss_listmle", np.nan)))
                train_curve["train_mse"].append(float(tr.get("loss_mse", np.nan)))
                train_curve["train_ic"].append(
                    float(tr.get("ic_raw", -tr["loss_ic"] if "loss_ic" in tr else np.nan))
                )
                train_curve["train_rank_ic"].append(float(tr.get("rank_ic", np.nan)))
                train_curve["train_score_std"].append(float(tr.get("score_std", np.nan)))
                train_curve["train_label_std"].append(float(tr.get("label_std", np.nan)))
                train_curve["train_grad_norm"].append(float(tr.get("grad_norm", np.nan)))
                train_curve["train_grad_nonfinite_steps"].append(float(tr.get("grad_nonfinite_steps", np.nan)))
                train_curve["train_grad_nonfinite_rate"].append(float(tr.get("grad_nonfinite_rate", np.nan)))
                train_curve["train_grad_skipped_steps"].append(float(tr.get("grad_skipped_steps", np.nan)))
                train_curve["train_grad_skipped_rate"].append(float(tr.get("grad_skipped_rate", np.nan)))
                train_curve["train_optimizer_steps"].append(float(tr.get("optimizer_steps", np.nan)))
                train_curve["train_lr"].append(float(tr.get("lr", np.nan)))
                # valid
                if va is not None:
                    train_curve["valid_main"].append(
                        float(va.get("loss_main", va.get(main_key, np.nan)))
                    )
                    train_curve["valid_listmle"].append(float(va.get("loss_listmle", np.nan)))
                    train_curve["valid_mse"].append(float(va.get("loss_mse", np.nan)))
                    train_curve["valid_rank_ic"].append(
                        float(va.get("rank_ic", np.nan)) if "rank_ic" in va else float("nan")
                    )
                    train_curve["valid_ic"].append(
                        float(va.get("ic_pearson_daily", va.get("ic_raw", -va["loss_ic"] if "loss_ic" in va else np.nan)))
                    )
                    train_curve["valid_score_std"].append(float(va.get("score_std", np.nan)))
                    train_curve["valid_label_std"].append(float(va.get("label_std", np.nan)))
                else:
                    train_curve["valid_main"].append(float("nan"))
                    train_curve["valid_listmle"].append(float("nan"))
                    train_curve["valid_mse"].append(float("nan"))
                    train_curve["valid_rank_ic"].append(float("nan"))
                    train_curve["valid_ic"].append(float("nan"))
                    train_curve["valid_score_std"].append(float("nan"))
                    train_curve["valid_label_std"].append(float("nan"))

                # Dynamic per-layer router diagnostics for the matrix/report.
                # Keep vectors rectangular even if a future config omits a key.
                cur_len = len(train_curve["epoch"])
                dyn_vals: Dict[str, float] = {}
                for k, v in tr.items():
                    if str(k).startswith(("router_layer_", "expert_layer_", "sampler_")):
                        dyn_vals[f"train_{k}"] = float(v)
                if va is not None:
                    for k, v in va.items():
                        if str(k).startswith(("router_layer_", "expert_layer_")):
                            dyn_vals[f"valid_{k}"] = float(v)
                dyn_keys = {
                    k
                    for k in train_curve.keys()
                    if k.startswith("train_router_layer_")
                    or k.startswith("valid_router_layer_")
                    or k.startswith("train_expert_layer_")
                    or k.startswith("valid_expert_layer_")
                    or k.startswith("train_sampler_")
                } | set(dyn_vals.keys())
                for k in sorted(dyn_keys):
                    if k not in train_curve:
                        train_curve[k] = [float("nan")] * (cur_len - 1)
                    train_curve[k].append(float(dyn_vals.get(k, np.nan)))

            if checkpoint_enabled:
                ckpt_value = self._checkpoint_metric_value(tr, va)
                if ckpt_value is not None and self._checkpoint_improved(ckpt_value, checkpoint_best):
                    checkpoint_best = float(ckpt_value)
                    checkpoint_epoch = int(epoch + 1)
                    checkpoint_state = copy.deepcopy(self.net.state_dict())

            # Stopping & best checkpoint selection:
            # - If early_stop>0: use valid metrics (legacy behavior)
            # - Else: use train metric (min) + optional threshold stop
            if use_valid_early_stop and va is not None:
                score = self._monitor(va)
                if self.use_tqdm and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(
                        {"best": best_score if best_score != float("-inf") else None, "bad": bad},
                        refresh=False,
                    )

                if score > best_score + self.min_delta:
                    best_score = score
                    best_state = copy.deepcopy(self.net.state_dict())
                    bad = 0
                else:
                    bad += 1

                if self.early_stop and self.early_stop > 0 and bad >= self.early_stop:
                    print(f">>> [EarlyStop] epoch={epoch + 1}, best_score={best_score:.6f}")
                    break
            elif not use_valid_early_stop:
                tv = tr.get(self.train_stop_key, None)
                if tv is None and self.train_stop_key != "loss_main":
                    tv = tr.get("loss_main", None)
                if tv is not None:
                    tv = float(tv)
                    if np.isfinite(tv):
                        # best checkpoint by train-loss(min)
                        if tv < best_train - self.min_delta:
                            best_train = tv
                            best_state = copy.deepcopy(self.net.state_dict())
                        # threshold stop
                        if self.train_stop_threshold is not None and (epoch + 1) >= self.min_epochs:
                            if tv <= self.train_stop_threshold:
                                good += 1
                            else:
                                good = 0
                            if good >= self.consecutive_k:
                                print(
                                    f">>> [TrainStop] epoch={epoch + 1}, {self.train_stop_key}={tv:.6f} "
                                    f"<= {self.train_stop_threshold:g} (k={self.consecutive_k})"
                                )
                                break

        restored_checkpoint = False
        if checkpoint_enabled and checkpoint_state is not None:
            self.net.load_state_dict(checkpoint_state)
            restored_checkpoint = True
            print(
                f">>> [Train] restored best checkpoint "
                f"({self.checkpoint_metric}={checkpoint_best:.6f}, epoch={checkpoint_epoch})"
            )
        elif best_state is not None:
            self.net.load_state_dict(best_state)
            if use_valid_early_stop:
                print(f">>> [Train] restored best (score={best_score:.6f})")
            else:
                print(f">>> [Train] restored best ({self.train_stop_key}={best_train:.6f})")

        # 保存训练曲线
        best_checkpoint_info = {
            "enabled": bool(checkpoint_enabled),
            "metric": self.checkpoint_metric,
            "mode": self.checkpoint_mode,
            "score": (
                float(checkpoint_best)
                if checkpoint_enabled and checkpoint_epoch is not None and np.isfinite(checkpoint_best)
                else None
            ),
            "epoch": checkpoint_epoch,
            "restored": bool(restored_checkpoint),
        }

        if train_curve is not None:
            try:
                rec.save_objects(
                    train_curve=train_curve,
                    best_checkpoint_info=best_checkpoint_info,
                    sampler_epoch_stats=dict(self._last_train_sampler_stats),
                )
            except Exception as e:
                print(f">>> [Train] save train_curve failed: {e}")

        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        assert self.net is not None
        self.net.eval()

        self._ensure_market_state()
        tsds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        # Always wrap so we can scatter predictions back to the correct index order when using batch_sampler.
        tsds = self._wrap_with_datetime(tsds)
        # Inference should be "intra-day batches" without any up/down-sampling:
        # - every sample enters the model exactly once
        # - each batch contains a single trading day (split into chunks if needed)
        sampler = DailyChunkBatchSampler(tsds, max_batch_size=self._daily_chunk_max_bs())
        loader = DataLoader(
            dataset=tsds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_feat_with_pos,
            worker_init_fn=self._worker_init_fn(),
            generator=self._loader_generator(offset=43),
        )

        num_alphas = self._get_num_alphas()
        f_ids = torch.arange(num_alphas, device=self.device)
        idx = tsds.get_index()
        pred = np.full((len(idx),), np.nan, dtype=float)

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    bx, bmacro, bpos = batch
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    bx, bpos = batch
                    bmacro = None
                else:
                    raise RuntimeError(f"Unexpected predict batch format: {type(batch)}")

                bx = self._coerce_bx_feature_dim(bx, expected_dim=num_alphas, context="predict")
                bx_t = torch.nan_to_num(bx, 0.0).to(self.device)
                macro_t = None if bmacro is None else torch.nan_to_num(bmacro, 0.0).to(self.device).float()
                with self._autocast_ctx():
                    # Note: date_ids removed - regime signal is computed from internal statistics
                    out = self.net(bx_t, f_ids, macro_features=macro_t)
                score = out.scores.detach().cpu().numpy()
                pos = bpos.detach().cpu().numpy().astype(int)
                if score.shape[0] != pos.shape[0]:
                    raise RuntimeError(f"Predict scatter mismatch: score={score.shape}, pos={pos.shape}")
                pred[pos] = score

        if not np.isfinite(pred).all():
            bad = int(np.sum(~np.isfinite(pred)))
            raise RuntimeError(f"Predict produced {bad} NaN/Inf entries; check data/filters.")
        return pd.Series(pred, index=idx).sort_index()

    def get_feature_importance(self):
        if self.net is not None and getattr(self.net, "feature_selector", None) is not None:
            mu = self.net.feature_selector.mu.detach().sigmoid().cpu().numpy()
            return pd.Series(mu)
        return None

    # ---------- Spatio-Temporal Visualization ----------
    def _collect_daily_diag_series(self, dataset: DatasetH, segment: str = "test") -> Dict[str, pd.Series]:
        """
        Collect daily diagnostic series on the given segment.

        Returns
        -------
        Dict[str, pd.Series]
            Keys (if available):
            - time_ratio: router time-expert ratio (avg over layers & samples per day)
            - gate_entropy: router gate entropy (avg over layers & samples per day)
            - time_tau / time_half_life: regime-adaptive time-scale diagnostics
            - factor_gate_*: regime-adaptive factor gate diagnostics
        """
        assert self.net is not None
        self._ensure_market_state()

        tsds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        idx = tsds.get_index()
        if not isinstance(idx, pd.MultiIndex) or "datetime" not in (idx.names or []):
            return {}
        dates = pd.to_datetime(idx.get_level_values("datetime")).normalize()

        num_alphas = self._get_num_alphas()
        f_ids = torch.arange(num_alphas, device=self.device)

        # Router can use batch-level "layer summary" (market mean/std). Therefore we must NOT mix dates
        # inside a batch; otherwise day-level diagnostics become meaningless.
        df_idx = pd.DataFrame({"datetime": dates})
        df_idx["int_idx"] = np.arange(len(df_idx), dtype=int)
        by_day = df_idx.groupby("datetime", sort=True)["int_idx"].apply(lambda x: x.to_numpy(dtype=int))

        metric_keys = (
            "time_embedding_norm",
            "time_embedding_to_value_norm_ratio",
            "time_tau",
            "time_half_life",
            "time_tau_std",
            "time_tau_p10",
            "time_tau_p90",
            "time_tau_range_util",
            "film_gamma_strength",
            "film_beta_strength",
            "factor_gate_mean",
            "factor_gate_std",
            "factor_gate_entropy",
            "factor_gate_topk_mass_5",
            "factor_gate_topk_mass_10",
        )
        n_layers = len(getattr(self.net, "layers", [])) if self.net is not None else 0
        expert_metric_keys: List[str] = []
        for layer_idx in range(int(n_layers)):
            for suffix in (
                "time_expert_norm",
                "factor_expert_norm",
                "time_contrib_norm",
                "factor_contrib_norm",
                "contrib_norm_ratio",
                "expert_cosine",
                "time_winner_ratio",
            ):
                expert_metric_keys.append(f"expert_layer_{layer_idx}_{suffix}")
        metric_keys = tuple(list(metric_keys) + expert_metric_keys)

        sum_by_day: Dict[pd.Timestamp, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        cnt_by_day: Dict[pd.Timestamp, int] = defaultdict(int)

        for dt, row_idx in by_day.items():
            row_idx = np.asarray(row_idx, dtype=int)
            n = int(row_idx.size)
            if n <= 0:
                continue

            for start in range(0, n, self.batch_size):
                chunk = row_idx[start : start + self.batch_size]
                bx_t = self._stack_feature_batch_from_row_indices(
                    tsds,
                    chunk,
                    max_samples=None,
                    num_alphas=num_alphas,
                    context="_collect_daily_diag_series",
                )
                if bx_t is None:
                    continue

                bsz = int(bx_t.shape[0])
                if bsz <= 0:
                    continue

                macro_t = self._macro_tensor_for_day(pd.Timestamp(dt), bsz)

                with torch.no_grad():
                    with self._autocast_ctx():
                        out = self.net(bx_t, f_ids, macro_features=macro_t)

                cnt_by_day[pd.Timestamp(dt)] += bsz

                # Router diagnostics (time_ratio / gate_entropy)
                if getattr(out, "gate_weights", None):
                    try:
                        gw = torch.stack(out.gate_weights, dim=0)  # [L,B,E] (E=2 or 3)
                        tr = gw[:, :, 0].mean(dim=0).detach().cpu().numpy()  # [B]
                        sum_by_day[pd.Timestamp(dt)]["time_ratio"] += float(np.sum(tr))
                        if gw.shape[-1] >= 3:
                            sr = gw[:, :, 2].mean(dim=0).detach().cpu().numpy()
                            sum_by_day[pd.Timestamp(dt)]["stock_ratio"] += float(np.sum(sr))
                    except Exception:
                        pass
                elif getattr(out, "avg_time_ratio", None) is not None:
                    sum_by_day[pd.Timestamp(dt)]["time_ratio"] += float(out.avg_time_ratio) * bsz

                if getattr(out, "avg_gate_entropy", None) is not None:
                    sum_by_day[pd.Timestamp(dt)]["gate_entropy"] += float(out.avg_gate_entropy) * bsz

                # Model-provided diagnostics (batch-mean scalars) -> accumulate by sample count
                m = getattr(out, "metrics", None) or {}
                for k in metric_keys:
                    if k in m:
                        sum_by_day[pd.Timestamp(dt)][k] += float(m[k]) * bsz

        if not cnt_by_day:
            return {}

        dts_sorted = sorted(cnt_by_day.keys())
        out_series: Dict[str, pd.Series] = {}
        for k in ("time_ratio", "gate_entropy", *metric_keys):
            data = {dt: (sum_by_day[dt][k] / max(1, cnt_by_day[dt])) for dt in dts_sorted if k in sum_by_day[dt]}
            if data:
                out_series[k] = pd.Series(data, dtype=float).sort_index()
        return out_series

    def _collect_gate_series(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """
        收集指定 segment 上的日度 gate time_ratio 序列。
        为了和报告里的 keyed object 对齐：
        - 返回 Series，index 为自然日期（datetime），value 为该日的平均 time_ratio。
        """
        series_map = self._collect_daily_diag_series(dataset, segment=segment)
        return series_map.get("time_ratio", pd.Series(dtype=float))

    def collect_router_oracle_diagnostics(self, dataset: DatasetH, segment: str = "test") -> Dict[str, float]:
        """
        Diagnostic-only forced-router evaluation.

        Replays the segment with default routing, forced time expert, forced
        factor expert, and uniform routing. This is not used during training.
        """
        assert self.net is not None
        self._ensure_market_state()

        try:
            tsds = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
            raw_x, raw_y = self._extract_sample(tsds[0])
            x_np = self._as_numpy(raw_x)
            y_np = None if raw_y is None else self._as_numpy(raw_y)
            _, y_np = self._split_packed_label(x_np, y_np)
            if self.label_dim > 0 and y_np is None:
                return {}
        except Exception as e:
            self._warn_once(
                f"router_oracle_prepare_{segment}",
                f">>> [RouterOracle] skipped for segment={segment}: {e}",
            )
            return {}

        loader = self._make_daily_chunk_loader(tsds, with_label=True)
        num_alphas = self._get_num_alphas()
        f_ids = torch.arange(num_alphas, device=self.device)
        modes = {"default": None, "time": "time", "factor": "factor", "uniform": "uniform"}
        if bool(getattr(getattr(self.net, "config", None), "use_stock_expert", False)):
            # h-20260610-002 probes: solo-stock + leave-one-out (the marginal-contribution reading)
            modes["stock"] = "stock"
            modes["no_stock"] = "no_stock"
        buffers: Dict[str, Dict[pd.Timestamp, Dict[str, List[np.ndarray]]]] = {
            name: defaultdict(lambda: {"p": [], "y": []}) for name in modes
        }

        was_training = bool(self.net.training)
        self.net.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    if not (isinstance(batch, (tuple, list)) and len(batch) == 4):
                        continue
                    bx, by, bmacro, day_key = batch
                    bx = self._coerce_bx_feature_dim(bx, expected_dim=num_alphas, context="router_oracle")
                    bx_t = torch.nan_to_num(bx, 0.0).to(self.device)
                    by_t = None if by is None else by.to(self.device).float()
                    if by_t is None:
                        continue
                    valid = torch.isfinite(by_t)
                    if valid.sum().item() < 2:
                        continue
                    bx_t = bx_t[valid]
                    by_t = by_t[valid]
                    macro_t = None if bmacro is None else torch.nan_to_num(bmacro, 0.0).to(self.device).float()
                    if macro_t is not None:
                        macro_t = macro_t[valid]
                    y_vec = by_t.view(-1).detach().cpu().numpy()
                    dt_key = pd.to_datetime(day_key).normalize()
                    for mode_name, override in modes.items():
                        with self._autocast_ctx():
                            out = self.net(
                                bx_t,
                                f_ids,
                                labels=None,
                                macro_features=macro_t,
                                router_override=override,
                            )
                        p_vec = out.scores.view(-1).detach().cpu().numpy()
                        buffers[mode_name][dt_key]["p"].append(p_vec)
                        buffers[mode_name][dt_key]["y"].append(y_vec)
        finally:
            self.net.train(was_training)

        def _daily_corr(mode_name: str) -> Tuple[pd.Series, pd.Series]:
            ic_vals: Dict[pd.Timestamp, float] = {}
            ric_vals: Dict[pd.Timestamp, float] = {}
            for dt, parts in buffers[mode_name].items():
                p = np.concatenate(parts["p"], axis=0) if parts["p"] else None
                y = np.concatenate(parts["y"], axis=0) if parts["y"] else None
                if p is None or y is None or p.size < 2 or y.size < 2:
                    continue
                if np.std(p) > 0 and np.std(y) > 0:
                    ic_vals[dt] = float(np.corrcoef(p, y)[0, 1])
                rp = pd.Series(p).rank().to_numpy()
                ry = pd.Series(y).rank().to_numpy()
                if np.std(rp) > 0 and np.std(ry) > 0:
                    ric_vals[dt] = float(np.corrcoef(rp, ry)[0, 1])
            return pd.Series(ic_vals, dtype=float).sort_index(), pd.Series(ric_vals, dtype=float).sort_index()

        ic_by_mode: Dict[str, pd.Series] = {}
        ric_by_mode: Dict[str, pd.Series] = {}
        out_metrics: Dict[str, float] = {}
        for mode_name in modes:
            ic_s, ric_s = _daily_corr(mode_name)
            ic_by_mode[mode_name] = ic_s
            ric_by_mode[mode_name] = ric_s
            if not ic_s.empty:
                out_metrics[f"{mode_name}_ic_mean"] = float(ic_s.mean())
            if not ric_s.empty:
                out_metrics[f"{mode_name}_rank_ic_mean"] = float(ric_s.mean())

        joined = pd.concat(
            [
                ric_by_mode.get("default", pd.Series(dtype=float)).rename("default"),
                ric_by_mode.get("time", pd.Series(dtype=float)).rename("time"),
                ric_by_mode.get("factor", pd.Series(dtype=float)).rename("factor"),
                ric_by_mode.get("uniform", pd.Series(dtype=float)).rename("uniform"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if not joined.empty:
            oracle_expert = joined[["time", "factor"]].max(axis=1)
            out_metrics["oracle_expert_rank_ic_mean"] = float(oracle_expert.mean())
            out_metrics["router_oracle_gap"] = float((oracle_expert - joined["default"]).mean())
            out_metrics["router_time_advantage"] = float((joined["time"] - joined["default"]).mean())
            out_metrics["router_factor_advantage"] = float((joined["factor"] - joined["default"]).mean())
            out_metrics["router_uniform_advantage"] = float((joined["uniform"] - joined["default"]).mean())
            out_metrics["time_beats_factor_day_ratio"] = float((joined["time"] > joined["factor"]).mean())
            out_metrics["default_beats_oracle_expert_day_ratio"] = float((joined["default"] > oracle_expert).mean())
            out_metrics["n_days"] = float(len(joined))
        # h-20260610-002: stock solo + leave-one-out marginal contribution (aligned on common days)
        if "stock" in ric_by_mode or "no_stock" in ric_by_mode:
            j2 = pd.concat(
                [
                    ric_by_mode.get("default", pd.Series(dtype=float)).rename("default"),
                    ric_by_mode.get("stock", pd.Series(dtype=float)).rename("stock"),
                    ric_by_mode.get("no_stock", pd.Series(dtype=float)).rename("no_stock"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            if not j2.empty:
                out_metrics["router_stock_advantage"] = float((j2["stock"] - j2["default"]).mean())
                # THE mechanism reading: how much rank_ic the model LOSES when the stock gate is
                # zeroed (time/factor renormalized). ~0 with high stock share = used-but-useless.
                out_metrics["router_no_stock_delta"] = float((j2["default"] - j2["no_stock"]).mean())

        return out_metrics

    def _collect_daily_factor_profiles(
        self,
        dataset: DatasetH,
        segment: str = "test",
        *,
        topk: int = 10,
    ) -> Dict[str, Any]:
        """
        Collect daily factor-gate and pooling distributions for focused diagnostics.

        Returns only what exists: FiLM-related entries are absent when factor gate
        is disabled, which keeps the MLflow/report validity checks clean.
        """
        assert self.net is not None
        self._ensure_market_state()

        tsds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        idx = tsds.get_index()
        if not isinstance(idx, pd.MultiIndex) or "datetime" not in (idx.names or []):
            return {}

        dates = pd.to_datetime(idx.get_level_values("datetime")).normalize()
        num_alphas = self._get_num_alphas()
        f_ids = torch.arange(num_alphas, device=self.device)

        df_idx = pd.DataFrame({"datetime": dates})
        df_idx["int_idx"] = np.arange(len(df_idx), dtype=int)
        by_day = df_idx.groupby("datetime", sort=True)["int_idx"].apply(lambda x: x.to_numpy(dtype=int))

        gate_sum: Dict[pd.Timestamp, np.ndarray] = {}
        pool_sum: Dict[pd.Timestamp, np.ndarray] = {}
        gate_cnt: Dict[pd.Timestamp, int] = defaultdict(int)
        pool_cnt: Dict[pd.Timestamp, int] = defaultdict(int)

        for dt, row_idx in by_day.items():
            row_idx = np.asarray(row_idx, dtype=int)
            for start in range(0, int(row_idx.size), self.batch_size):
                chunk = row_idx[start : start + self.batch_size]
                bx_t = self._stack_feature_batch_from_row_indices(
                    tsds,
                    chunk,
                    max_samples=None,
                    num_alphas=num_alphas,
                    context="_collect_daily_factor_profiles",
                )
                if bx_t is None:
                    continue
                bsz = int(bx_t.shape[0])
                if bsz <= 0:
                    continue

                macro_t = self._macro_tensor_for_day(pd.Timestamp(dt), bsz)
                with torch.no_grad():
                    with self._autocast_ctx():
                        out = self.net(bx_t, f_ids, macro_features=macro_t)

                dt_key = pd.Timestamp(dt)
                gate_imp = getattr(out, "factor_gate_importance", None)
                if isinstance(gate_imp, torch.Tensor) and gate_imp.ndim == 2:
                    arr = gate_imp.detach().float().cpu().numpy()
                    gate_sum.setdefault(dt_key, np.zeros((arr.shape[1],), dtype=float))
                    gate_sum[dt_key] += np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=0)
                    gate_cnt[dt_key] += int(arr.shape[0])

                pool_w = getattr(out, "factor_pool_weights", None)
                if isinstance(pool_w, torch.Tensor) and pool_w.ndim == 2:
                    arr = pool_w.detach().float().cpu().numpy()
                    pool_sum.setdefault(dt_key, np.zeros((arr.shape[1],), dtype=float))
                    pool_sum[dt_key] += np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=0)
                    pool_cnt[dt_key] += int(arr.shape[0])

        def _profile(sum_map: Dict[pd.Timestamp, np.ndarray], cnt_map: Dict[pd.Timestamp, int]) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            for dt in sorted(sum_map.keys()):
                cnt = max(1, int(cnt_map.get(dt, 0)))
                w = np.asarray(sum_map[dt], dtype=float) / float(cnt)
                w = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
                s = float(w.sum())
                if s > 0:
                    w = w / s
                k = min(max(1, int(topk)), int(w.shape[0]))
                ids = np.argsort(w)[::-1][:k]
                out[pd.Timestamp(dt).strftime("%Y-%m-%d")] = {
                    "weights": [float(x) for x in w],
                    "top_ids": [int(i) for i in ids],
                    "top_weights": [float(w[i]) for i in ids],
                }
            return out

        gate_profile = _profile(gate_sum, gate_cnt)
        pool_profile = _profile(pool_sum, pool_cnt)

        overlap_data: Dict[pd.Timestamp, float] = {}
        for dt_str, g in gate_profile.items():
            p = pool_profile.get(dt_str, None)
            if not p:
                continue
            a = set(g.get("top_ids", [])[:topk])
            b = set(p.get("top_ids", [])[:topk])
            if a or b:
                overlap_data[pd.Timestamp(dt_str)] = float(len(a & b) / max(1, len(a | b)))

        result: Dict[str, Any] = {}
        if gate_profile:
            result["factor_gate"] = gate_profile
        if pool_profile:
            result["pooling"] = pool_profile
        if overlap_data:
            result["film_pool_topk_overlap"] = pd.Series(overlap_data, dtype=float).sort_index()
        return result

    def _collect_attention_maps(
        self,
        dataset: DatasetH,
        segment: str = "test",
        target_dates: List[Union[str, pd.Timestamp]] | None = None,
        *,
        max_dates: int = 5,
        attn_layer: int = -1,
        factor_use_last_time: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        抽若干交易日, 提取最后一层 attention maps（time & factor）:

        Returns
        -------
        Dict[date_str, Dict[str, np.ndarray]]
            - date_str: "YYYY-MM-DD"
            - "time":   [T, T]  (avg over batch × factor × heads)
            - "factor": [N, N]  (avg over batch × heads, default uses last time step of window)
            - "factor_pool": [N] (avg over batch; attention pooling weights from model output factor_pool_weights)

        Notes
        -----
        - time expert attention is computed on shape [B*N, H, T, T] (or compatible variants)
        - factor expert attention is computed on shape [B*T, H, N, N] (or compatible variants)
        - if target_dates is None, dates are selected to be spread over the segment span (avoid consecutive days)
        """
        self._ensure_market_state()
        try:
            tsds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        except Exception as e:
            print(f">>> [Warn] _collect_attention_maps failed to prepare tsds for segment={segment} ({e})")
            return {}

        # 1) choose dates
        try:
            idx = tsds.get_index()
        except Exception as e:
            # best-effort fallback for older TSDataSampler implementations
            try:
                idx = getattr(getattr(tsds, "data", None), "index", None)
            except Exception:
                idx = None

        if not isinstance(idx, pd.MultiIndex) or "datetime" not in (idx.names or []):
            print(
                f">>> [Warn] _collect_attention_maps failed to resolve 'datetime' index; "
                f"idx_type={type(idx).__name__}, idx_names={getattr(idx, 'names', None)}"
            )
            return {}

        dates = pd.to_datetime(idx.get_level_values("datetime")).normalize()

        if target_dates:
            chosen_dates = [pd.Timestamp(pd.to_datetime(d)).normalize() for d in target_dates]
            # Keep backward-compatible behavior for explicit target_dates.
            if max_dates is not None and max_dates > 0 and len(chosen_dates) > int(max_dates):
                chosen_dates = chosen_dates[-int(max_dates) :]
        else:
            chosen_dates = [pd.Timestamp(d).normalize() for d in sorted(pd.unique(dates))]
            if max_dates is not None and max_dates > 0 and len(chosen_dates) > int(max_dates):
                chosen_dates = self._select_spaced_dates(chosen_dates, int(max_dates))

        if not chosen_dates:
            return {}

        # 2) prepare factor ids
        num_alphas = self._get_num_alphas()
        f_ids = torch.arange(num_alphas, device=self.device)

        def _reduce_time_attn(time_attn: torch.Tensor, *, B: int, T: int, N: int) -> np.ndarray:
            """Return [T,T]."""
            if time_attn is None:
                raise ValueError("time_attn is None")
            a = time_attn
            # [BN, H, T, T] or [B, H, T, T]
            if a.dim() == 4:
                bn, H, t1, t2 = a.shape
                if bn == B * N:
                    a = a.view(B, N, H, t1, t2).mean(dim=(0, 1, 2))
                elif bn == B:
                    a = a.mean(dim=(0, 1))
                else:
                    a = a.mean(dim=0).mean(dim=0)
            # [BN, T, T] or [B, T, T]
            elif a.dim() == 3:
                bn, t1, t2 = a.shape
                if bn == B * N:
                    a = a.view(B, N, t1, t2).mean(dim=(0, 1))
                elif bn == B:
                    a = a.mean(dim=0)
                else:
                    a = a.mean(dim=0)
            else:
                raise ValueError(f"Unexpected time_attn ndim={a.dim()}")
            return a.detach().cpu().numpy()

        def _reduce_factor_attn(factor_attn: torch.Tensor, *, B: int, T: int, N: int, use_last_time: bool) -> np.ndarray:
            """Return [N,N]."""
            if factor_attn is None:
                raise ValueError("factor_attn is None")
            a = factor_attn
            # [B*T, H, N, N] or [B, H, N, N]
            if a.dim() == 4:
                bth, H, n1, n2 = a.shape
                if bth == B * T:
                    a = a.view(B, T, H, n1, n2)
                    if use_last_time:
                        a = a[:, -1]  # [B, H, N, N]
                        a = a.mean(dim=(0, 1))
                    else:
                        a = a.mean(dim=(0, 1, 2))
                elif bth == B:
                    a = a.mean(dim=(0, 1))
                else:
                    a = a.mean(dim=0).mean(dim=0)
            # [B*T, N, N] or [B, N, N]
            elif a.dim() == 3:
                bth, n1, n2 = a.shape
                if bth == B * T:
                    a = a.view(B, T, n1, n2)
                    if use_last_time:
                        a = a[:, -1].mean(dim=0)
                    else:
                        a = a.mean(dim=(0, 1))
                elif bth == B:
                    a = a.mean(dim=0)
                else:
                    a = a.mean(dim=0)
            else:
                raise ValueError(f"Unexpected factor_attn ndim={a.dim()}")
            return a.detach().cpu().numpy()

        # 3) collect
        attn_maps: Dict[str, Dict[str, np.ndarray]] = {}

        for dt in chosen_dates:
            row_idx = np.where(dates == dt)[0]
            if len(row_idx) == 0:
                continue

            bx_t = self._stack_feature_batch_from_row_indices(
                tsds,
                row_idx,
                max_samples=self.batch_size,
                num_alphas=num_alphas,
                context="_collect_attention_maps",
            )
            if bx_t is None:
                continue

            B = int(bx_t.shape[0])
            T = int(bx_t.shape[1])
            N = int(bx_t.shape[2])
            macro_t = self._macro_tensor_for_day(pd.Timestamp(dt), B)

            n_layers = len(getattr(self.net, "layers", [])) if self.net is not None else 0
            if n_layers <= 0:
                continue
            if attn_layer >= 0:
                layer_idx = min(int(attn_layer), n_layers - 1)
            else:
                layer_idx = n_layers - 1

            with torch.no_grad():
                with self._autocast_ctx():
                    # Note: date_ids removed - regime signal is computed from internal statistics
                    out = self.net(
                        bx_t,
                        f_ids[:N],
                        macro_features=macro_t,
                        return_attn=True,
                        attn_layers=[layer_idx],
                    )

            maps_one: Dict[str, np.ndarray] = {}

            # Attention-pooling weights over factors (for interpretability): [B, N] -> mean over batch => [N]
            try:
                pool_w = getattr(out, "factor_pool_weights", None)
                if isinstance(pool_w, torch.Tensor) and pool_w.dim() == 2 and int(pool_w.shape[-1]) == N:
                    maps_one["factor_pool"] = pool_w.detach().float().mean(dim=0).cpu().numpy()
            except Exception as e:
                print(f">>> [Warn] _collect_attention_maps (factor_pool) failed at {dt}: {e}")

            layer_attn = None
            if getattr(out, "attn_maps", None):
                key = f"layer_{layer_idx}"
                layer_attn = out.attn_maps.get(key, None)
            if isinstance(layer_attn, dict) and len(layer_attn) > 0:
                if "time" in layer_attn and layer_attn["time"] is not None:
                    try:
                        maps_one["time"] = _reduce_time_attn(layer_attn["time"], B=B, T=T, N=N)
                    except Exception as e:
                        print(f">>> [Warn] _collect_attention_maps (time) failed at {dt}: {e}")
                if "factor" in layer_attn and layer_attn["factor"] is not None:
                    try:
                        maps_one["factor"] = _reduce_factor_attn(
                            layer_attn["factor"], B=B, T=T, N=N, use_last_time=factor_use_last_time
                        )
                    except Exception as e:
                        print(f">>> [Warn] _collect_attention_maps (factor) failed at {dt}: {e}")

            if maps_one:
                attn_maps[dt.strftime("%Y-%m-%d")] = maps_one

        return attn_maps

    @staticmethod
    def _plot_gate_series(gate_series: pd.Series, title: str = "Gate Time Ratio"):
        fig, ax = plt.subplots(figsize=(8, 3))
        gate_series.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("date")
        ax.set_ylabel("time_expert_ratio")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_series(series: pd.Series, *, title: str, y_label: str):
        fig, ax = plt.subplots(figsize=(8, 3))
        series.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("date")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_tau_vs_time_ratio(time_ratio: pd.Series, time_tau: pd.Series, *, title: str):
        df = pd.concat(
            [
                pd.Series(time_ratio, name="time_ratio"),
                pd.Series(time_tau, name="time_tau"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        df = df.sort_index()

        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax2 = ax1.twinx()

        l1 = ax1.plot(df.index, df["time_ratio"], color="C0", label="time_ratio")
        l2 = ax2.plot(df.index, df["time_tau"], color="C1", label="time_tau")

        ax1.set_title(title)
        ax1.set_xlabel("date")
        ax1.set_ylabel("time_ratio")
        ax2.set_ylabel("time_tau")
        ax1.grid(True, alpha=0.3)

        lines = (l1 or []) + (l2 or [])
        labels = [ln.get_label() for ln in lines]
        if lines:
            ax1.legend(lines, labels, loc="upper left", frameon=False)

        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_attention_map(
        attn: np.ndarray,
        *,
        title: str,
        x_label: str,
        y_label: str,
        figsize: Tuple[float, float] = (4, 4),
    ):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attn, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def export_visuals(
        self,
        dataset: DatasetH,
        segment: str = "test",
        *,
        max_attn_days: int = 4,
        attn_layer: int = -1,
        target_dates: List[Union[str, pd.Timestamp]] | None = None,
        prefix: str = "st_disentangle",
        factor_use_last_time: bool = True,
        factor_topk: int = 10,
        save_png: bool = True,
        ):
        """
        在当前 Qlib Recorder 中导出:
        1) router / regime-adaptive diagnostics 日度序列 (Series + Fig + optional PNG)
           - gate time_ratio (backward compatible key: f"{prefix}_gate_series")
           - optional: gate_entropy / time_tau / time_half_life / factor_gate_* (if enabled in model)
        2) 若干日期的 attention heatmaps (raw dict + figs + optional PNGs)

        - raw attention stored as:   f"{prefix}_attn_maps"
          format: {date_str: {"time": [T,T], "factor": [N,N], "factor_pool": [N]}}
        - optional PNG filenames stored as: f"{prefix}_attn_pngs"
          format: {date_str: {"time": "...png", "factor": "...png", "pool": "...png"}}
        """
        recorder = R.get_recorder()
        local_dir = None
        if save_png:
            try:
                local_dir = Path(recorder.get_local_dir())
                local_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                local_dir = None

        daily_series = self._collect_daily_diag_series(dataset, segment=segment)
        gate_series = daily_series.get("time_ratio", pd.Series(dtype=float))
        attn_maps = self._collect_attention_maps(
            dataset,
            segment=segment,
            target_dates=target_dates,
            max_dates=max_attn_days,
            attn_layer=attn_layer,
            factor_use_last_time=factor_use_last_time,
        )
        factor_profiles = self._collect_daily_factor_profiles(
            dataset,
            segment=segment,
            topk=factor_topk or 10,
        )

        factor_topk_map: Dict[str, Dict[str, List[int] | List[float]]] = {}
        factor_pool_topk_map: Dict[str, Dict[str, List[int] | List[float]]] = {}
        if factor_topk and int(factor_topk) > 0 and isinstance(attn_maps, dict):
            def _factor_importance(attn: np.ndarray) -> np.ndarray:
                a = np.asarray(attn, dtype=float)
                if a.ndim == 3:
                    a = a.mean(axis=0)
                if a.ndim != 2:
                    raise ValueError(f"factor attn must be 2D, got shape {a.shape}")
                # Normalize rows, then use column mean as "attention received".
                row_sum = a.sum(axis=-1, keepdims=True) + 1e-12
                a = a / row_sum
                return a.mean(axis=0)

            for dt_str, maps in attn_maps.items():
                if not isinstance(maps, dict):
                    continue
                f_map = maps.get("factor", None)
                if f_map is None:
                    continue
                try:
                    scores = _factor_importance(f_map)
                except Exception as e:
                    print(f">>> [Visual] factor_topk failed at {dt_str}: {e}")
                    continue
                n = int(scores.shape[0])
                k = min(int(factor_topk), n)
                if k <= 0:
                    continue
                idx = np.argsort(scores)[::-1][:k]
                factor_topk_map[dt_str] = {
                    "ids": [int(i) for i in idx],
                    "weights": [float(scores[i]) for i in idx],
                }

            for dt_str, maps in attn_maps.items():
                if not isinstance(maps, dict):
                    continue
                pool_w = maps.get("factor_pool", None)
                if pool_w is None:
                    continue
                try:
                    scores = np.asarray(pool_w, dtype=float).reshape(-1)
                except Exception as e:
                    print(f">>> [Visual] factor_pool_topk failed at {dt_str}: {e}")
                    continue
                if scores.ndim != 1:
                    continue
                n = int(scores.shape[0])
                k = min(int(factor_topk), n)
                if k <= 0:
                    continue
                idx = np.argsort(scores)[::-1][:k]
                factor_pool_topk_map[dt_str] = {
                    "ids": [int(i) for i in idx],
                    "weights": [float(scores[i]) for i in idx],
                }

        # save raw objects first (so report can still work even if fig saving fails)
        try:
            extra_series_objs = {
                f"{prefix}_{k}_series": v for k, v in daily_series.items() if k != "time_ratio" and v is not None
            }
            if factor_topk_map:
                extra_series_objs[f"{prefix}_factor_topk"] = factor_topk_map
            if factor_pool_topk_map:
                extra_series_objs[f"{prefix}_factor_pool_topk"] = factor_pool_topk_map
            if factor_profiles:
                if "factor_gate" in factor_profiles:
                    extra_series_objs[f"{prefix}_factor_gate_profiles"] = factor_profiles["factor_gate"]
                if "pooling" in factor_profiles:
                    extra_series_objs[f"{prefix}_pool_profiles"] = factor_profiles["pooling"]
                if "film_pool_topk_overlap" in factor_profiles:
                    extra_series_objs[f"{prefix}_film_pool_topk_overlap_series"] = factor_profiles[
                        "film_pool_topk_overlap"
                    ]
            recorder.save_objects(
                **{
                    f"{prefix}_gate_series": gate_series,
                    f"{prefix}_attn_maps": attn_maps,
                    **extra_series_objs,
                }
            )
        except Exception as e:
            print(f">>> [Visual] save_objects(raw) failed: {e}")

        # gate curve: fig + png
        fig_gate = None
        try:
            fig_gate = self._plot_gate_series(gate_series, title=f"Gate Time Ratio ({segment})")
            try:
                recorder.save_objects(**{f"{prefix}_gate_series_fig": fig_gate})
            except Exception:
                pass
            if local_dir is not None:
                gate_png = f"{prefix}_gate_series_{segment}.png"
                try:
                    fig_gate.savefig(local_dir / gate_png, dpi=150, bbox_inches="tight")
                    try:
                        recorder.save_objects(**{f"{prefix}_gate_png": gate_png})
                    except Exception:
                        pass
                except Exception as e:
                    print(f">>> [Visual] save gate png failed: {e}")
        except Exception as e:
            print(f">>> [Visual] build gate fig failed: {e}")
        finally:
            if fig_gate is not None:
                plt.close(fig_gate)

        # other daily diagnostics (if any): figs + pngs
        try:
            for k, s in daily_series.items():
                if k == "time_ratio" or s is None or len(s) == 0:
                    continue
                title = f"{k} ({segment})"
                y_label = k
                fig = None
                try:
                    fig = self._plot_series(s, title=title, y_label=y_label)
                    try:
                        recorder.save_objects(**{f"{prefix}_{k}_series_fig": fig})
                    except Exception:
                        pass
                    if local_dir is not None:
                        fn = f"{prefix}_{k}_series_{segment}.png"
                        try:
                            fig.savefig(local_dir / fn, dpi=150, bbox_inches="tight")
                            try:
                                recorder.save_objects(**{f"{prefix}_{k}_png": fn})
                            except Exception:
                                pass
                        except Exception as e:
                            print(f">>> [Visual] save diag png failed ({k}): {e}")
                finally:
                    if fig is not None:
                        plt.close(fig)
        except Exception as e:
            print(f">>> [Visual] save diag figs failed: {e}")

        # tau vs time_ratio combined plot (for paper narrative)
        try:
            tr = daily_series.get("time_ratio", None)
            tau = daily_series.get("time_tau", None)
            if tr is not None and tau is not None and len(tr) > 0 and len(tau) > 0:
                fig = None
                try:
                    fig = self._plot_tau_vs_time_ratio(tr, tau, title=f"tau vs time_ratio ({segment})")
                    try:
                        recorder.save_objects(**{f"{prefix}_tau_vs_time_ratio_fig": fig})
                    except Exception:
                        pass
                    if local_dir is not None:
                        fn = f"{prefix}_tau_vs_time_ratio_{segment}.png"
                        try:
                            fig.savefig(local_dir / fn, dpi=150, bbox_inches="tight")
                            try:
                                recorder.save_objects(**{f"{prefix}_tau_vs_time_ratio_png": fn})
                            except Exception:
                                pass
                        except Exception as e:
                            print(f">>> [Visual] save tau_vs_time_ratio png failed: {e}")
                finally:
                    if fig is not None:
                        plt.close(fig)
        except Exception as e:
            print(f">>> [Visual] save tau_vs_time_ratio fig failed: {e}")

        # attention heatmaps: figs + pngs
        attn_pngs: Dict[str, Dict[str, str]] = {}
        for dt_str, maps in (attn_maps or {}).items():
            try:
                # backward compatibility: maps might be [T,T]
                if isinstance(maps, dict):
                    time_attn = maps.get("time", None)
                    factor_attn = maps.get("factor", None)
                    pool_attn = maps.get("factor_pool", None)
                else:
                    time_attn = maps
                    factor_attn = None
                    pool_attn = None

                if time_attn is not None:
                    fig_t = None
                    try:
                        fig_t = self._plot_attention_map(
                            np.asarray(time_attn),
                            title=f"Time Attention ({dt_str})",
                            x_label="time (j)",
                            y_label="time (i)",
                            figsize=(4, 4),
                        )
                        key_t = f"{prefix}_attn_time_{dt_str}"
                        try:
                            recorder.save_objects(**{key_t: fig_t})
                        except Exception:
                            pass
                        if local_dir is not None:
                            fn_t = f"{prefix}_attn_time_{dt_str}.png"
                            try:
                                fig_t.savefig(local_dir / fn_t, dpi=150, bbox_inches="tight")
                                attn_pngs.setdefault(dt_str, {})["time"] = fn_t
                            except Exception as e:
                                print(f">>> [Visual] save attn png failed (time, {dt_str}): {e}")
                    finally:
                        if fig_t is not None:
                            plt.close(fig_t)

                if factor_attn is not None:
                    fig_f = None
                    try:
                        fig_f = self._plot_attention_map(
                            np.asarray(factor_attn),
                            title=f"Factor Attention ({dt_str})",
                            x_label="factor (j)",
                            y_label="factor (i)",
                            figsize=(6, 6),
                        )
                        key_f = f"{prefix}_attn_factor_{dt_str}"
                        try:
                            recorder.save_objects(**{key_f: fig_f})
                        except Exception:
                            pass
                        if local_dir is not None:
                            fn_f = f"{prefix}_attn_factor_{dt_str}.png"
                            try:
                                fig_f.savefig(local_dir / fn_f, dpi=150, bbox_inches="tight")
                                attn_pngs.setdefault(dt_str, {})["factor"] = fn_f
                            except Exception as e:
                                print(f">>> [Visual] save attn png failed (factor, {dt_str}): {e}")
                    finally:
                        if fig_f is not None:
                            plt.close(fig_f)

                if pool_attn is not None:
                    fig_p = None
                    try:
                        w = np.asarray(pool_attn, dtype=float).reshape(1, -1)
                        fig_p = self._plot_attention_map(
                            w,
                            title=f"Factor Pooling Attention ({dt_str})",
                            x_label="factor (j)",
                            y_label="pool",
                            figsize=(10, 2),
                        )
                        key_p = f"{prefix}_attn_pool_factor_{dt_str}"
                        try:
                            recorder.save_objects(**{key_p: fig_p})
                        except Exception:
                            pass
                        if local_dir is not None:
                            fn_p = f"{prefix}_attn_pool_factor_{dt_str}.png"
                            try:
                                fig_p.savefig(local_dir / fn_p, dpi=150, bbox_inches="tight")
                                attn_pngs.setdefault(dt_str, {})["pool"] = fn_p
                            except Exception as e:
                                print(f">>> [Visual] save attn png failed (pool, {dt_str}): {e}")
                    finally:
                        if fig_p is not None:
                            plt.close(fig_p)
            except Exception as e:
                print(f">>> [Visual] save attn figs failed ({dt_str}): {e}")

        if attn_pngs:
            try:
                recorder.save_objects(**{f"{prefix}_attn_pngs": attn_pngs})
            except Exception:
                pass
