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

from module.dataloader.sampler import FixedDailyBatchSampler
from module.quant_moe_model import QuantMoEModel
from module.utils.model_configuration import QuantMoEConfig


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
            - loss_total / loss_listmle / loss_ic / loss_aux / loss_sparsity
            - ic_raw / rank_ic (adapter 侧现算)
            - gate_entropy / time_ratio / active_feat_ratio
        * train_curve 对象：
            - epoch, train_listmle, train_ic, valid_listmle, valid_rank_ic, valid_ic
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
    def _collate_train(self, samples: List[Any]):
        xs: List[torch.Tensor] = []
        ys: List[Optional[torch.Tensor]] = []

        for s in samples:
            raw_x, raw_y = self._extract_sample(s)
            x_np = self._as_numpy(raw_x)
            y_np = None if raw_y is None else self._as_numpy(raw_y)
            x_np, y_np = self._split_packed_label(x_np, y_np)

            xs.append(torch.from_numpy(np.asarray(x_np)).float())
            ys.append(None if y_np is None else torch.from_numpy(np.asarray(y_np)).float())

        bx = torch.stack(xs, dim=0)  # [B,T,F]
        by = None
        if ys and ys[0] is not None:
            by = torch.stack([t.view(-1) for t in ys], dim=0)  # [B,label_dim]
            if by.shape[1] == 1:
                by = by.squeeze(1)  # [B]
        return bx, by

    def _collate_feat(self, samples: List[Any]):
        xs: List[torch.Tensor] = []
        for s in samples:
            raw_x, _ = self._extract_sample(s)
            x_np = self._as_numpy(raw_x)
            xs.append(torch.from_numpy(np.asarray(x_np)).float())
        return torch.stack(xs, dim=0)

    def _make_daily_loader(self, tsds, *, shuffle: bool, train: bool) -> DataLoader:
        """
        使用 FixedDailyBatchSampler 做日度截面 batch.
        - train=True/False: 都用 _collate_train（valid 也需要 label 做监控）。
        """
        sampler = FixedDailyBatchSampler(tsds, self.batch_size, shuffle=shuffle, seed=self.random_seed)
        return DataLoader(
            dataset=tsds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_train if train else self._collate_train,
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
        - loss_listmle → 额外记 {prefix}/listmle
        - loss_ic      → 额外记 {prefix}/ic = 1 - loss_ic （loss_ic = -IC）
        """
        m = dict(metrics)
        if "loss_listmle" in m:
            m.setdefault("listmle", m["loss_listmle"])
        if "loss_ic" in m:
            # loss_ic = -IC, so monitored IC should be -loss_ic within [-1, 1]
            m.setdefault("ic", -float(m["loss_ic"]))

        try:
            R.log_metrics(step=step, **{f"{prefix}/{k}": float(v) for k, v in m.items()})
        except Exception:
            pass

    def _monitor(self, valid_metrics: Dict[str, float]) -> float:
        """用于 early stopping 的单一 score（越大越好）."""
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

        iterator = pbar if pbar is not None else loader
        for step, (bx, by) in enumerate(iterator):
            bx_t = torch.nan_to_num(bx, 0.0).to(self.device)  # [B,T,F]
            if self.label_dim > 0 and by is None:
                raise RuntimeError(
                    "by is None in training/valid loop while label_dim>0; "
                    "check schema validation and label_dim."
                )
            by_t = None if by is None else by.to(self.device).float()

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

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                # Note: date_ids removed - regime signal is computed from internal statistics
                out = self.net(bx_t, f_ids, labels=by_t)
                loss = getattr(out, "loss", None)

                if train and optimizer is not None and loss is not None:
                    if not torch.isfinite(loss):
                        skip_nan_loss += 1
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.global_step += 1

            n_batches += 1

            # 聚合模型内部的 metrics（loss_total / loss_listmle / loss_ic / aux / sparsity ...）
            if getattr(out, "metrics", None):
                for k, v in out.metrics.items():
                    meters[k] += float(v)

            # 适配器侧计算 IC / RankIC（基于当前 label）
            if by_t is not None and getattr(out, "scores", None) is not None:
                stock_scores = out.scores
                p_vec = stock_scores.view(-1).detach().cpu().numpy()
                y_vec = by_t.view(-1).detach().cpu().numpy()

                if p_vec.size >= 2 and y_vec.size >= 2:
                    if np.std(p_vec) > 0 and np.std(y_vec) > 0:
                        ic = np.corrcoef(p_vec, y_vec)[0, 1]
                        meters["ic_raw"] += float(ic)

                    rank_p = pd.Series(p_vec).rank().to_numpy()
                    rank_y = pd.Series(y_vec).rank().to_numpy()
                    if np.std(rank_p) > 0 and np.std(rank_y) > 0:
                        ric = np.corrcoef(rank_p, rank_y)[0, 1]
                        meters["rank_ic"] += float(ric)

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
                pbar.set_postfix(avg, refresh=False)

        if pbar is not None:
            pbar.close()

        return self._avg(meters, n_batches)

    # ---------- Qlib API ----------
    def fit(self, dataset: DatasetH, evals_result=dict()):
        # 1) Train schema & TSDS
        train_tsds = self._validate_train_schema(dataset)
        train_loader = self._make_daily_loader(train_tsds, shuffle=True, train=True)

        # 2) Valid set (DK_I)
        try:
            valid_tsds = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
            valid_loader = self._make_daily_loader(valid_tsds, shuffle=False, train=True)
        except Exception:
            valid_loader = None

        # 3) Init network from first batch
        bx0, by0 = next(iter(train_loader))
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
        if rec is not None:
            train_curve = {
                "epoch": [],
                "train_listmle": [],
                "train_ic": [],
                "valid_listmle": [],
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
                # train
                train_curve["train_listmle"].append(float(tr.get("loss_listmle", np.nan)))
                train_curve["train_ic"].append(
                    float(1.0 - tr["loss_ic"]) if "loss_ic" in tr else float("nan")
                )
                # valid
                if va is not None:
                    train_curve["valid_listmle"].append(float(va.get("loss_listmle", np.nan)))
                    train_curve["valid_rank_ic"].append(
                        float(va.get("rank_ic", np.nan)) if "rank_ic" in va else float("nan")
                    )
                    train_curve["valid_ic"].append(
                        float(1.0 - va["loss_ic"]) if "loss_ic" in va else float("nan")
                    )
                else:
                    train_curve["valid_listmle"].append(float("nan"))
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

        tsds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        loader = DataLoader(
            dataset=tsds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_feat,
        )

        f_ids = torch.arange(int(self.model_config["num_alphas"]), device=self.device)
        preds: List[np.ndarray] = []

        with torch.no_grad():
            for bx in loader:
                bx_t = torch.nan_to_num(bx, 0.0).to(self.device)
                # Note: date_ids removed - regime signal is computed from internal statistics
                out = self.net(bx_t, f_ids)
                score = out.scores.detach().cpu().numpy()
                preds.append(score)

        pred = np.concatenate(preds, axis=0)
        idx = tsds.get_index()
        if len(pred) != len(idx):
            pred = pred[: len(idx)]
        return pd.Series(pred, index=idx).sort_index()

    def get_feature_importance(self):
        if self.net is not None and getattr(self.net, "feature_selector", None) is not None:
            mu = self.net.feature_selector.mu.detach().sigmoid().cpu().numpy()
            return pd.Series(mu)
        return None

    # ---------- Spatio-Temporal Visualization ----------
    def _collect_gate_series(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """
        收集指定 segment 上的日度 gate time_ratio 序列。
        为了和报告里的 keyed object 对齐：
        - 返回 Series，index 为自然日期（datetime），value 为该日的平均 time_ratio。
        """
        assert self.net is not None

        tsds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        idx = tsds.get_index()
        dates = pd.to_datetime(idx.get_level_values("datetime"))

        f_ids = torch.arange(int(self.model_config["num_alphas"]), device=self.device)

        all_dates: List[pd.Timestamp] = []
        all_time_ratio: List[float] = []

        batch_x: List[torch.Tensor] = []
        batch_dates: List[pd.Timestamp] = []

        for i in range(len(tsds)):
            raw_x, _ = self._extract_sample(tsds[i])
            x_np = self._as_numpy(raw_x)
            batch_x.append(torch.from_numpy(np.asarray(x_np)).float())
            batch_dates.append(dates[i])

            if len(batch_x) == self.batch_size or i == len(tsds) - 1:
                bx = torch.stack(batch_x, dim=0).to(self.device)
                
                with torch.no_grad():
                    # Note: date_ids removed - regime signal is computed from internal statistics
                    out = self.net(bx, f_ids)

                if out.gate_weights:
                    # gate_weights: List[num_layers] of [B, 2]
                    gw = torch.stack(out.gate_weights, dim=0)  # [L,B,2]
                    tr = gw[:, :, 0].mean(dim=0).detach().cpu().numpy()  # [B]
                    all_time_ratio.extend(tr.tolist())
                    all_dates.extend(batch_dates)

                batch_x.clear()
                batch_dates.clear()

        if not all_dates:
            return pd.Series(dtype=float)

        df = pd.DataFrame({"datetime": all_dates, "time_ratio": all_time_ratio})
        gate_series = df.groupby("datetime")["time_ratio"].mean().sort_index()
        return gate_series

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

            layer_idx = attn_layer if attn_layer >= 0 else (len(self.net.layers) - 1)

            with torch.no_grad():
                # Note: date_ids removed - regime signal is computed from internal statistics
                out = self.net(
                    bx,
                    f_ids,
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
        1) gate time_ratio 日度序列 (Series + Fig + optional PNG)
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

        gate_series = self._collect_gate_series(dataset, segment=segment)
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
            recorder.save_objects(
                **{
                    f"{prefix}_gate_series": gate_series,
                    f"{prefix}_attn_maps": attn_maps,
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
