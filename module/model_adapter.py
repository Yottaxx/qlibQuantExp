
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm, trange
except Exception:  # pragma: no cover
    tqdm = None
    trange = None

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
    - Daily cross-sectional batching via FixedDailyBatchSampler (needed for CS losses).
    - Explicit schema validation on the train segment:
        * Checks whether TSDataSampler returns (x, y) or packs label into x.
        * Enforces label_dim consistency.
    - Uses DK_L for train, DK_I for valid/test/predict (aligned with official workflow).
    - Label NaNs are masked (not coerced to 0) before loss computation.
    """

    def __init__(self, model_config: dict = None, trainer_config: dict = None, **kwargs):
        self.model_config = dict(model_config or {})
        self.trainer_config = dict(trainer_config or {})

        self.lr = float(self.trainer_config.get("lr", 5e-4))
        self.epochs = int(self.trainer_config.get("n_epochs", 20))
        self.batch_size = int(self.trainer_config.get("batch_size", 1024))
        self.num_workers = int(self.trainer_config.get("num_workers", 4))

        self.early_stop = int(self.trainer_config.get("early_stop", 0) or 0)
        self.min_delta = float(self.trainer_config.get("min_delta", 1e-6))

        # If TSDataSampler packs label into x: last `label_dim` channels are labels.
        # For Alpha158 + single label, label_dim=1 is typical.
        self.label_dim = int(self.trainer_config.get("label_dim", 1))

        # tqdm progress
        self.use_tqdm = bool(self.trainer_config.get("use_tqdm", True))
        self.tqdm_update_every = int(self.trainer_config.get("tqdm_update_every", 10))
        self.tqdm_mininterval = float(self.trainer_config.get("tqdm_mininterval", 0.3))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[QuantMoEModel] = None

    def _make_pbar(self, it, *, desc: str, total: Optional[int] = None, leave: bool = False):
        if not self.use_tqdm:
            return None, it
        if tqdm is None:
            raise RuntimeError('tqdm is not installed. Install tqdm or set trainer_config.use_tqdm=False')
        return tqdm(it, desc=desc, total=total, leave=leave, dynamic_ncols=True, mininterval=self.tqdm_mininterval), None

    # ---------- low-level helpers ----------
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
            # Explicitly configured as "no label in x".
            return x_np, None

        if x_np.ndim != 2:
            raise RuntimeError(f"Expect x to be 2D [T,C], got shape {x_np.shape}")

        if x_np.shape[1] <= self.label_dim:
            raise RuntimeError(
                f"Packed-label assumption violated: got x_dim={x_np.shape[1]} <= label_dim={self.label_dim}. "
                f"Please set trainer_config['label_dim']=0 or fix handler schema."
            )

        # Take label from the *last timestep*; avoid leaking future information.
        y_np = x_np[-1, -self.label_dim:]
        x_np = x_np[:, :-self.label_dim]
        return x_np, y_np

    # ---------- schema validation ----------
    def _validate_train_schema(self, dataset: DatasetH):
        """Validate train schema & return a TS dataset ready for DataLoader.

        This method checks:
        - Whether label is present explicitly.
        - If not, whether it can be safely inferred from the last channels of x.
        - That label_dim is consistent with the observed shapes.
        """
        ts = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        s0 = ts[0]
        raw_x, raw_y = self._extract_sample(s0)
        x_np = self._as_numpy(raw_x)
        y_np = None if raw_y is None else self._as_numpy(raw_y)

        if x_np.ndim != 2:
            raise RuntimeError(f"Train sample x must be 2D [T,C]; got shape {x_np.shape}")

        if y_np is not None:
            # Label provided explicitly
            if self.label_dim > 0:
                # allow [label_dim] or [..., label_dim]
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

        # y_np is None: try packed-label mode
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

    # ---------- collate functions ----------
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

        bx = torch.stack(xs, dim=0)  # [B, T, F]
        by = None
        if ys and ys[0] is not None:
            by = torch.stack([t.view(-1) for t in ys], dim=0)  # [B, label_dim]
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
        return DataLoader(
            dataset=tsds,
            batch_sampler=FixedDailyBatchSampler(tsds, self.batch_size, shuffle=shuffle),
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate_train if train else self._collate_train,
        )

    # ---------- model init & metrics ----------
    def _init_net(self, bx: torch.Tensor) -> None:
        # bx: [B, T, F]
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
        try:
            R.log_metrics(step=step, **{f"{prefix}/{k}": float(v) for k, v in metrics.items()})
        except Exception:
            pass

    def _monitor(self, valid_metrics: Dict[str, float]) -> float:
        # higher is better; prefer ic proxy if present
        if "loss_ic" in valid_metrics:
            return 1.0 - float(valid_metrics["loss_ic"])
        return -float(valid_metrics.get("loss_total", 0.0))

    # ---------- epoch loop ----------
    def _run_epoch(
        self,
        loader: DataLoader,
        f_ids: torch.Tensor,
        *,
        optimizer: Optional[torch.optim.Optimizer],
        train: bool,
        desc: str,
    ) -> Dict[str, float]:
        assert self.net is not None

        self.net.train(train)
        meters = defaultdict(float)
        n_batches = 0

        # tqdm wrapper (per-epoch)
        total = None
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
                # If this happens, schema validation should have already raised.
                raise RuntimeError("by is None in training/valid loop; check schema validation and label_dim.")

            by_t = None if by is None else by.to(self.device).float()

            # mask invalid labels (loss impls must not see NaN/Inf)
            if by_t is not None:
                valid = torch.isfinite(by_t)
                vr = float(valid.float().mean().item())
                meters["valid_ratio"] += vr
                if valid.sum().item() < 2:
                    skip_invalid_label += 1
                    if pbar is not None and (step + 1) % max(1, self.tqdm_update_every) == 0:
                        pbar.set_postfix({"skip_lbl": skip_invalid_label}, refresh=False)
                    continue
                bx_t = bx_t[valid]
                by_t = by_t[valid]

            bd_t = torch.zeros(bx_t.shape[0], dtype=torch.long, device=self.device)  # date_ids placeholder

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                out = self.net(bx_t, f_ids, bd_t, labels=by_t)
                loss = getattr(out, "loss", None)

                if train and loss is not None:
                    if not torch.isfinite(loss):
                        skip_nan_loss += 1
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()

            n_batches += 1

            if getattr(out, "metrics", None):
                for k, v in out.metrics.items():
                    meters[k] += float(v)

            if getattr(out, "avg_gate_entropy", None) is not None:
                meters["gate_entropy"] += float(out.avg_gate_entropy)
            if getattr(out, "avg_time_ratio", None) is not None:
                meters["time_ratio"] += float(out.avg_time_ratio)
            if getattr(out, "selected_mask", None) is not None:
                meters["active_feat_ratio"] += float(out.selected_mask.mean().item())

            if pbar is not None and ((step + 1) % max(1, self.tqdm_update_every) == 0):
                avg = {k: meters[k] / max(1, n_batches) for k in ("loss_total", "loss_ic", "loss_rank", "loss_huber") if k in meters}
                avg.update({k: meters[k] / max(1, n_batches) for k in ("gate_entropy", "time_ratio", "active_feat_ratio") if k in meters})
                if "valid_ratio" in meters:
                    avg["valid"] = meters["valid_ratio"] / max(1, n_batches)
                if train and optimizer is not None:
                    avg["lr"] = optimizer.param_groups[0]["lr"]
                if skip_invalid_label:
                    avg["skip_lbl"] = skip_invalid_label
                if skip_nan_loss:
                    avg["skip_nan"] = skip_nan_loss
                pbar.set_postfix(avg, refresh=False)

        if pbar is not None:
            pbar.close()

        return self._avg(meters, n_batches)

    # ---------- Qlib API ----------
    def fit(self, dataset: DatasetH, evals_result=dict()):
        # 1) Train schema validation + TSDS
        train_tsds = self._validate_train_schema(dataset)
        train_loader = self._make_daily_loader(train_tsds, shuffle=True, train=True)

        # 2) Valid set (DK_I, aligned with official inference flow)
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

        best_state = None
        best_score = float("-inf")
        bad = 0

        print(f">>> [Train] epochs={self.epochs}, early_stop={self.early_stop}")
        epoch_iter = range(self.epochs)
        if self.use_tqdm and trange is not None:
            epoch_iter = trange(self.epochs, desc='Epochs', dynamic_ncols=True)
        for epoch in epoch_iter:
            tr = self._run_epoch(train_loader, f_ids, optimizer=optimizer, train=True, desc=f'Train e{epoch+1:02d}')
            self._log_metrics(epoch, "train", tr)
            print(f"| Train {epoch + 1:02d} | " + " | ".join(f"{k}:{v:.6f}" for k, v in tr.items()))

            if valid_loader is None:
                continue

            va = self._run_epoch(valid_loader, f_ids, optimizer=None, train=False, desc=f'Valid e{epoch+1:02d}')
            self._log_metrics(epoch, "valid", va)
            print(f"| Valid {epoch + 1:02d} | " + " | ".join(f"{k}:{v:.6f}" for k, v in va.items()))

            score = self._monitor(va)

            # update epoch-level progress bar
            if self.use_tqdm and hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix({'best': best_score if best_score != float('-inf') else None, 'bad': bad}, refresh=False)
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
                bd_t = torch.zeros(bx_t.shape[0], dtype=torch.long, device=self.device)
                out = self.net(bx_t, f_ids, bd_t)
                score = out.logits.mean(dim=1).detach().cpu().numpy()
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
