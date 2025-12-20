# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from collections import defaultdict
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

        self.early_stop = int(self.trainer_config.get("early_stop", 0) or 0)
        self.min_delta = float(self.trainer_config.get("min_delta", 1e-6))

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

        # Warmup scheduler config
        self.use_warmup = bool(self.trainer_config.get("use_warmup", True))
        self.warmup_ratio = float(self.trainer_config.get("warmup_ratio", 0.1))
        self.warmup_steps = int(self.trainer_config.get("warmup_steps", 0))

        # tqdm progress
        self.use_tqdm = bool(self.trainer_config.get("use_tqdm", True))
        self.tqdm_update_every = int(self.trainer_config.get("tqdm_update_every", 10))
        self.tqdm_mininterval = float(self.trainer_config.get("tqdm_mininterval", 0.3))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[QuantMoEModel] = None

        # global step for scheduler
        self.global_step: int = 0

        # ---- diagnostics ----
        # Enable a one-batch backward sanity check at the beginning of fit()
        self.debug_sanity_check = bool(self.trainer_config.get("debug_sanity_check", False))
        # Thresholds for "almost constant" detection
        self.debug_std_eps = float(self.trainer_config.get("debug_std_eps", 1e-8))
        self.debug_grad_eps = float(self.trainer_config.get("debug_grad_eps", 1e-12))
        self._warned_keys: set[str] = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(msg)

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

        loss.backward()
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

    def _maybe_warn_market_state(self):
        if self.market_state_path and self._market_state is None:
            print(">>> [MarketState] market_state_path is set but not loaded yet; call fit/predict will load it.")

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

    def _make_daily_loader(self, tsds, *, shuffle: bool, train: bool) -> DataLoader:
        """
        使用 FixedDailyBatchSampler 做日度截面 batch.
        - train=True/False: 都用 _collate_train（valid 也需要 label 做监控）。
        """
        if self._market_state is not None:
            tsds = self._wrap_with_datetime(tsds)
        sampler = FixedDailyBatchSampler(tsds, self.batch_size, shuffle=shuffle, seed=self.random_seed)
        return DataLoader(
            dataset=tsds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_train,
        )

    def _make_daily_chunk_loader(self, tsds, *, with_label: bool) -> DataLoader:
        """
        Deterministic daily loader without sampling:
        - each day is fully covered (split into chunks if needed)
        - no up/down-sampling
        """
        tsds = self._wrap_with_datetime(tsds)
        sampler = DailyChunkBatchSampler(tsds, max_batch_size=self.batch_size)
        return DataLoader(
            dataset=tsds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_eval_daily if with_label else self._collate_feat,
        )

    # ---------- net init & metrics ----------
    def _init_net(self, bx: torch.Tensor) -> None:
        # bx: [B,T,F]
        T, F = int(bx.shape[1]), int(bx.shape[2])
        self.model_config.update({"context_len": T, "num_alphas": F})
        conf = QuantMoEConfig(**self.model_config)
        self.net = QuantMoEModel(conf).to(self.device)
        print(f">>> [Auto-Config] context_len={T}, num_alphas={F}")

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

        try:
            R.log_metrics(step=step, **{f"{prefix}/{k}": float(v) for k, v in m.items()})
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
        daily_buffer = defaultdict(lambda: {"p": [], "y": []}) if not train else None

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

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                # Note: date_ids removed - regime signal is computed from internal statistics
                out = self.net(bx_t, f_ids, labels=by_t, macro_features=macro_t)
                loss = getattr(out, "loss", None)

                if train and optimizer is not None and loss is not None:
                    if not torch.isfinite(loss):
                        skip_nan_loss += 1
                        continue
                    loss.backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0))
                    optimizer.step()
                    opt_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                    self.global_step += 1
                    if not np.isfinite(grad_norm) or grad_norm <= self.debug_grad_eps:
                        self._warn_once(
                            "zero_grad_norm",
                            f">>> [Warn] grad_norm≈0 ({grad_norm:.3e}); parameters may not be updating.",
                        )
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
                                f">>> [Warn] label std≈0 ({y_std:.3e}); ranking loss has little signal in-batch.",
                            )
                    if getattr(out, "scores", None) is not None:
                        p_std = float(out.scores.detach().float().std(unbiased=False).item())
                        if p_std <= self.debug_std_eps:
                            self._warn_once(
                                "score_almost_constant",
                                f">>> [Warn] score std≈0 ({p_std:.3e}); check data variability / model wiring.",
                            )
                    if bx_t.ndim == 3 and bx_t.shape[0] > 0:
                        x_std = float(bx_t[:, -1, :].detach().float().std(unbiased=False).item())
                        if x_std <= self.debug_std_eps:
                            self._warn_once(
                                "x_almost_constant",
                                f">>> [Warn] x(last-step) std≈0 ({x_std:.3e}); features may be all-0 after Fillna.",
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

        avg = self._avg(meters, n_batches)
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
        self._ensure_market_state()
        # 1) Train schema & TSDS
        train_tsds = self._validate_train_schema(dataset)
        train_loader = self._make_daily_loader(train_tsds, shuffle=True, train=True)

        # 2) Valid set (DK_I)
        valid_loader = None
        valid_err: Optional[Exception] = None
        for data_key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
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
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        f_ids = torch.arange(int(self.model_config["num_alphas"]), device=self.device)

        if self.debug_sanity_check:
            self._sanity_check_batch(bx0, by0, bmacro0, f_ids=f_ids, optimizer=optimizer)

        # Warmup scheduler
        if self.use_warmup:
            try:
                num_update_steps_per_epoch = len(train_loader)
            except Exception:
                num_update_steps_per_epoch = 1
            total_training_steps = max(1, self.epochs * num_update_steps_per_epoch)

            if self.warmup_steps > 0:
                warmup_steps = self.warmup_steps
            else:
                warmup_steps = int(total_training_steps * self.warmup_ratio)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
            )
            print(
                f">>> [Schedule] use warmup: total_steps={total_training_steps}, "
                f"warmup_steps={warmup_steps}"
            )
        else:
            scheduler = None
            print(">>> [Schedule] warmup disabled")

        best_state = None
        best_score = float("-inf")
        bad = 0

        # 训练曲线缓存，用于报告里的“训练过程诊断”
        rec = R.get_recorder()
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
                "valid_main": [],
                "valid_listmle": [],
                "valid_mse": [],
                "valid_rank_ic": [],
                "valid_ic": [],
            }

        print(f">>> [Train] epochs={self.epochs}, early_stop={self.early_stop}")
        epoch_iter = range(self.epochs)
        if self.use_tqdm and trange is not None:
            epoch_iter = trange(self.epochs, desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_iter:
            tr = self._run_epoch(
                train_loader,
                f_ids,
                optimizer=optimizer,
                scheduler=scheduler,
                train=True,
                desc=f"Train e{epoch + 1:02d}",
            )
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
                    float(-tr["loss_ic"]) if "loss_ic" in tr else float("nan")
                )
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
                        float(-va["loss_ic"]) if "loss_ic" in va else float("nan")
                    )
                else:
                    train_curve["valid_main"].append(float("nan"))
                    train_curve["valid_listmle"].append(float("nan"))
                    train_curve["valid_mse"].append(float("nan"))
                    train_curve["valid_rank_ic"].append(float("nan"))
                    train_curve["valid_ic"].append(float("nan"))

            # early stopping 监控
            if va is not None:
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

        if best_state is not None:
            self.net.load_state_dict(best_state)
            print(f">>> [Train] restored best (score={best_score:.6f})")

        # 保存训练曲线
        if train_curve is not None:
            try:
                rec.save_objects(train_curve=train_curve)
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
        sampler = DailyChunkBatchSampler(tsds, max_batch_size=self.batch_size)
        loader = DataLoader(
            dataset=tsds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_feat_with_pos,
        )

        f_ids = torch.arange(int(self.model_config["num_alphas"]), device=self.device)
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
                bx_t = torch.nan_to_num(bx, 0.0).to(self.device)
                macro_t = None if bmacro is None else torch.nan_to_num(bmacro, 0.0).to(self.device).float()
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

        f_ids = torch.arange(int(self.model_config["num_alphas"]), device=self.device)

        # Router can use batch-level "layer summary" (market mean/std). Therefore we must NOT mix dates
        # inside a batch; otherwise day-level diagnostics become meaningless.
        df_idx = pd.DataFrame({"datetime": dates})
        df_idx["int_idx"] = np.arange(len(df_idx), dtype=int)
        by_day = df_idx.groupby("datetime", sort=True)["int_idx"].apply(lambda x: x.to_numpy(dtype=int))

        metric_keys = (
            "time_tau",
            "time_half_life",
            "factor_gate_mean",
            "factor_gate_std",
            "factor_gate_entropy",
            "factor_gate_topk_mass_5",
            "factor_gate_topk_mass_10",
        )

        sum_by_day: Dict[pd.Timestamp, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        cnt_by_day: Dict[pd.Timestamp, int] = defaultdict(int)

        for dt, row_idx in by_day.items():
            row_idx = np.asarray(row_idx, dtype=int)
            n = int(row_idx.size)
            if n <= 0:
                continue

            for start in range(0, n, self.batch_size):
                chunk = row_idx[start : start + self.batch_size]
                xs: List[torch.Tensor] = []
                for i in chunk:
                    raw_x, _ = self._extract_sample(tsds[int(i)])
                    x_np = self._as_numpy(raw_x)
                    xs.append(torch.from_numpy(np.asarray(x_np)).float())

                if not xs:
                    continue

                bx = torch.stack(xs, dim=0)
                bx_t = torch.nan_to_num(bx, 0.0).to(self.device)
                bsz = int(bx_t.shape[0])
                if bsz <= 0:
                    continue

                macro_t = None
                if self._market_state is not None:
                    m = self._macro_from_dates([pd.Timestamp(dt)] * bsz)
                    macro_t = torch.nan_to_num(m, 0.0).to(self.device).float()

                with torch.no_grad():
                    out = self.net(bx_t, f_ids, macro_features=macro_t)

                cnt_by_day[pd.Timestamp(dt)] += bsz

                # Router diagnostics (time_ratio / gate_entropy)
                if getattr(out, "gate_weights", None):
                    try:
                        gw = torch.stack(out.gate_weights, dim=0)  # [L,B,2]
                        tr = gw[:, :, 0].mean(dim=0).detach().cpu().numpy()  # [B]
                        sum_by_day[pd.Timestamp(dt)]["time_ratio"] += float(np.sum(tr))
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

        Notes
        -----
        - time expert attention is computed on shape [B*N, H, T, T] (or compatible variants)
        - factor expert attention is computed on shape [B*T, H, N, N] (or compatible variants)
        """
        self._ensure_market_state()
        try:
            tsds = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        except Exception:
            tsds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        if not hasattr(tsds, "data"):
            return {}

        # 1) choose dates
        try:
            idx_df = tsds.data.index.to_frame(index=False)
            date_series = pd.to_datetime(idx_df["datetime"])
        except Exception:
            # fallback: cannot resolve date index
            return {}

        if target_dates:
            chosen_dates = []
            for d in target_dates:
                d = pd.to_datetime(d)
                chosen_dates.append(d.normalize())
        else:
            chosen_dates = sorted(date_series.unique())

        if not chosen_dates:
            return {}

        if max_dates is not None and max_dates > 0:
            chosen_dates = chosen_dates[-max_dates:]

        # 2) prepare factor ids
        num_alphas = int(self.model_config["num_alphas"])
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
            mask = date_series == dt
            row_idx = np.where(mask.values)[0]
            if len(row_idx) == 0:
                continue

            xs: List[torch.Tensor] = []
            for i in row_idx[: self.batch_size]:
                raw_x, _ = self._extract_sample(tsds[int(i)])
                x_np = self._as_numpy(raw_x)
                xs.append(torch.from_numpy(np.asarray(x_np)).float())

            if not xs:
                continue

            bx = torch.stack(xs, dim=0).to(self.device)  # [B, T, N]
            B = int(bx.shape[0])
            T = int(bx.shape[1])
            macro_t = None
            if self._market_state is not None:
                m = self._macro_from_dates([dt] * B)
                macro_t = torch.nan_to_num(m, 0.0).to(self.device).float()

            layer_idx = attn_layer if attn_layer >= 0 else (len(self.net.layers) - 1)

            with torch.no_grad():
                # Note: date_ids removed - regime signal is computed from internal statistics
                out = self.net(
                    bx,
                    f_ids,
                    macro_features=macro_t,
                    return_attn=True,
                    attn_layers=[layer_idx],
                )

            if not getattr(out, "attn_maps", None):
                continue

            key = f"layer_{layer_idx}"
            layer_attn = out.attn_maps.get(key, None)
            if not isinstance(layer_attn, dict) or len(layer_attn) == 0:
                continue

            maps_one: Dict[str, np.ndarray] = {}
            if "time" in layer_attn and layer_attn["time"] is not None:
                try:
                    maps_one["time"] = _reduce_time_attn(layer_attn["time"], B=B, T=T, N=num_alphas)
                except Exception:
                    pass
            if "factor" in layer_attn and layer_attn["factor"] is not None:
                try:
                    maps_one["factor"] = _reduce_factor_attn(
                        layer_attn["factor"], B=B, T=T, N=num_alphas, use_last_time=factor_use_last_time
                    )
                except Exception:
                    pass

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
        save_png: bool = True,
    ):
        """
        在当前 Qlib Recorder 中导出:
        1) router / regime-adaptive diagnostics 日度序列 (Series + Fig + optional PNG)
           - gate time_ratio (backward compatible key: f"{prefix}_gate_series")
           - optional: gate_entropy / time_tau / time_half_life / factor_gate_* (if enabled in model)
        2) 若干日期的 attention heatmaps (raw dict + figs + optional PNGs)

        - raw attention stored as:   f"{prefix}_attn_maps"
          format: {date_str: {"time": [T,T], "factor": [N,N]}}
        - optional PNG filenames stored as: f"{prefix}_attn_pngs"
          format: {date_str: {"time": "...png", "factor": "...png"}}
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

        # save raw objects first (so report can still work even if fig saving fails)
        try:
            extra_series_objs = {
                f"{prefix}_{k}_series": v for k, v in daily_series.items() if k != "time_ratio" and v is not None
            }
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
        try:
            fig_gate = self._plot_gate_series(gate_series, title=f"Gate Time Ratio ({segment})")
            recorder.save_objects(**{f"{prefix}_gate_series_fig": fig_gate})
            if local_dir is not None:
                gate_png = f"{prefix}_gate_series_{segment}.png"
                fig_gate.savefig(local_dir / gate_png, dpi=150, bbox_inches="tight")
                recorder.save_objects(**{f"{prefix}_gate_png": gate_png})
            plt.close(fig_gate)
        except Exception as e:
            print(f">>> [Visual] save gate fig failed: {e}")

        # other daily diagnostics (if any): figs + pngs
        try:
            for k, s in daily_series.items():
                if k == "time_ratio" or s is None or len(s) == 0:
                    continue
                title = f"{k} ({segment})"
                y_label = k
                fig = self._plot_series(s, title=title, y_label=y_label)
                recorder.save_objects(**{f"{prefix}_{k}_series_fig": fig})
                if local_dir is not None:
                    fn = f"{prefix}_{k}_series_{segment}.png"
                    fig.savefig(local_dir / fn, dpi=150, bbox_inches="tight")
                    recorder.save_objects(**{f"{prefix}_{k}_png": fn})
                plt.close(fig)
        except Exception as e:
            print(f">>> [Visual] save diag figs failed: {e}")

        # tau vs time_ratio combined plot (for paper narrative)
        try:
            tr = daily_series.get("time_ratio", None)
            tau = daily_series.get("time_tau", None)
            if tr is not None and tau is not None and len(tr) > 0 and len(tau) > 0:
                fig = self._plot_tau_vs_time_ratio(tr, tau, title=f"tau vs time_ratio ({segment})")
                recorder.save_objects(**{f"{prefix}_tau_vs_time_ratio_fig": fig})
                if local_dir is not None:
                    fn = f"{prefix}_tau_vs_time_ratio_{segment}.png"
                    fig.savefig(local_dir / fn, dpi=150, bbox_inches="tight")
                    recorder.save_objects(**{f"{prefix}_tau_vs_time_ratio_png": fn})
                plt.close(fig)
        except Exception as e:
            print(f">>> [Visual] save tau_vs_time_ratio fig failed: {e}")

        # attention heatmaps: figs + pngs
        attn_pngs: Dict[str, Dict[str, str]] = {}
        try:
            for dt_str, maps in attn_maps.items():
                # backward compatibility: maps might be [T,T]
                if isinstance(maps, dict):
                    time_attn = maps.get("time", None)
                    factor_attn = maps.get("factor", None)
                else:
                    time_attn = maps
                    factor_attn = None

                if time_attn is not None:
                    fig_t = self._plot_attention_map(
                        np.asarray(time_attn),
                        title=f"Time Attention ({dt_str})",
                        x_label="time (j)",
                        y_label="time (i)",
                        figsize=(4, 4),
                    )
                    key_t = f"{prefix}_attn_time_{dt_str}"
                    recorder.save_objects(**{key_t: fig_t})
                    if local_dir is not None:
                        fn_t = f"{prefix}_attn_time_{dt_str}.png"
                        fig_t.savefig(local_dir / fn_t, dpi=150, bbox_inches="tight")
                        attn_pngs.setdefault(dt_str, {})["time"] = fn_t
                    plt.close(fig_t)

                if factor_attn is not None:
                    fig_f = self._plot_attention_map(
                        np.asarray(factor_attn),
                        title=f"Factor Attention ({dt_str})",
                        x_label="factor (j)",
                        y_label="factor (i)",
                        figsize=(6, 6),
                    )
                    key_f = f"{prefix}_attn_factor_{dt_str}"
                    recorder.save_objects(**{key_f: fig_f})
                    if local_dir is not None:
                        fn_f = f"{prefix}_attn_factor_{dt_str}.png"
                        fig_f.savefig(local_dir / fn_f, dpi=150, bbox_inches="tight")
                        attn_pngs.setdefault(dt_str, {})["factor"] = fn_f
                    plt.close(fig_f)
        except Exception as e:
            print(f">>> [Visual] save attn figs failed: {e}")

        if attn_pngs:
            try:
                recorder.save_objects(**{f"{prefix}_attn_pngs": attn_pngs})
            except Exception:
                pass
