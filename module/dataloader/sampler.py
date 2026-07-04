import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from typing import Any, Dict, Optional


class FixedDailyBatchSampler(Sampler):
    """
    SOTA Sampling Strategy:
    1. Group data by date.
    2. Randomly select a date.
    3. Randomly sample 'batch_size' instruments from that date.
       - If daily_count > batch_size: Downsample (Unbiased estimator of gradients)
       - If daily_count < batch_size: Upsample (Padding to ensure tensor shape stability)
    """

    def __init__(self, data_source, batch_size, shuffle=True, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = None if seed is None else int(seed)
        self.epoch = 0
        self.last_epoch_stats: Dict[str, Any] = {}

        # Optimize: Access index directly without loading full data
        try:
            # Qlib TSDatasetH/DatasetH usually stores index in .handler or directly accessible
            index = data_source.get_index()
        except AttributeError:
            # Fallback for wrappers
            index = data_source.dataset.get_index()

        # Build Date -> [Indices] Map
        # Reset index to get integer locations
        df_idx = pd.DataFrame(index=index).reset_index()
        df_idx['int_idx'] = np.arange(len(df_idx))

        # Grouping (This is fast enough for < 10M rows)
        self.daily_groups = df_idx.groupby('datetime')['int_idx'].apply(np.array).tolist()
        self.num_batches = len(self.daily_groups)
        self.total_source_samples = int(sum(len(g) for g in self.daily_groups))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_epoch(self) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(self.seed) + int(self.epoch))

    def __iter__(self):
        rng = self._rng_for_epoch()
        indices = np.arange(self.num_batches)
        if self.shuffle:
            indices = rng.permutation(indices)

        total_draws = 0
        unique_selected = set()
        duplicate_draws = 0
        downsample_days = 0
        upsample_days = 0
        exact_days = 0

        for day_i in indices:
            daily_indices = self.daily_groups[day_i]
            n_samples = len(daily_indices)

            if n_samples > self.batch_size:
                # Downsample: Randomly pick subset
                batch_indices = rng.choice(daily_indices, self.batch_size, replace=False)
                downsample_days += 1
            elif n_samples < self.batch_size:
                # Upsample: Randomly pick with replacement to fill batch
                batch_indices = rng.choice(daily_indices, self.batch_size, replace=True)
                upsample_days += 1
            else:
                batch_indices = np.asarray(daily_indices, dtype=int).copy()
                exact_days += 1

            # Shuffle in-batch order for better gradient diversity
            if self.shuffle:
                batch_indices = rng.permutation(batch_indices)

            arr = np.asarray(batch_indices, dtype=int)
            total_draws += int(arr.size)
            unique_in_batch = set(int(x) for x in np.unique(arr))
            unique_selected.update(unique_in_batch)
            duplicate_draws += int(arr.size - len(unique_in_batch))

            yield batch_indices

        num_days = max(1, int(self.num_batches))
        total_source = max(1, int(self.total_source_samples))
        self.last_epoch_stats = {
            "epoch": int(self.epoch),
            "num_days": int(self.num_batches),
            "num_batches": int(self.num_batches),
            "batch_size": int(self.batch_size),
            "total_source_samples": int(self.total_source_samples),
            "total_draws": int(total_draws),
            "unique_samples": int(len(unique_selected)),
            "coverage_ratio": float(len(unique_selected) / total_source),
            "coverage_gap_vs_full": float(1.0 - (len(unique_selected) / total_source)),
            "duplicate_draws": int(duplicate_draws),
            "duplicate_rate": float(duplicate_draws / max(1, total_draws)),
            "downsample_days": int(downsample_days),
            "upsample_days": int(upsample_days),
            "exact_days": int(exact_days),
            "downsample_day_ratio": float(downsample_days / num_days),
            "upsample_day_ratio": float(upsample_days / num_days),
        }

    def __len__(self):
        return self.num_batches


class DailyChunkBatchSampler(Sampler):
    """
    Deterministic daily batch sampler (no sampling, no up/down-sampling).

    - Groups samples by their `datetime` index level.
    - For each day, yields consecutive chunks of indices with size <= max_batch_size.
    - Guarantees every sample appears exactly once per epoch.
    - Preserves chronological/sample order by default; train callers may enable
      epoch-aware deterministic shuffling of day order and in-day order.

    This is suitable for inference/predict where we want "intra-day batches" while
    preserving full coverage and the original sample order.
    """

    def __init__(
        self,
        data_source,
        max_batch_size: int,
        *,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        if max_batch_size is None or int(max_batch_size) <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.data_source = data_source
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = None if seed is None else int(seed)
        self.epoch = 0
        self.last_epoch_stats: Dict[str, Any] = {}

        try:
            index = data_source.get_index()
        except AttributeError:
            index = data_source.dataset.get_index()

        if not isinstance(index, pd.MultiIndex) or "datetime" not in (index.names or []):
            raise RuntimeError(
                "DailyChunkBatchSampler requires a MultiIndex with a 'datetime' level. "
                f"Got index type={type(index)}, names={getattr(index, 'names', None)}"
            )

        # NOTE:
        # Qlib datasets are not guaranteed to be ordered by datetime (often instrument-major),
        # so we must group by the datetime level rather than assuming contiguity.
        dts = pd.to_datetime(index.get_level_values("datetime"))
        df_idx = pd.DataFrame({"datetime": dts})
        df_idx["int_idx"] = np.arange(len(df_idx), dtype=int)
        self.daily_groups = (
            df_idx.groupby("datetime", sort=True)["int_idx"].apply(lambda x: x.to_numpy(dtype=int)).tolist()
        )

        self._num_batches = int(
            sum(int(np.ceil(len(g) / self.max_batch_size)) for g in self.daily_groups)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_epoch(self) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(self.seed) + int(self.epoch))

    def __iter__(self):
        rng = self._rng_for_epoch()
        day_order = np.arange(len(self.daily_groups))
        if self.shuffle:
            day_order = rng.permutation(day_order)

        total_draws = 0
        unique_selected = set()
        short_days = 0
        exact_days = 0
        split_days = 0

        for day_i in day_order:
            g = np.asarray(self.daily_groups[int(day_i)], dtype=int)
            n = int(len(g))
            if n < self.max_batch_size:
                short_days += 1
            elif n == self.max_batch_size:
                exact_days += 1
            else:
                split_days += 1

            if self.shuffle and n > 1:
                g = rng.permutation(g)

            for i in range(0, n, self.max_batch_size):
                batch = np.asarray(g[i : i + self.max_batch_size], dtype=int)
                total_draws += int(batch.size)
                unique_selected.update(int(x) for x in batch)
                yield batch

        num_days = max(1, int(len(self.daily_groups)))
        total_source = int(sum(len(g) for g in self.daily_groups))
        total_source_safe = max(1, total_source)
        self.last_epoch_stats = {
            "epoch": int(self.epoch),
            "num_days": int(len(self.daily_groups)),
            "num_batches": int(self._num_batches),
            "batch_size": int(self.max_batch_size),
            "total_source_samples": int(total_source),
            "total_draws": int(total_draws),
            "unique_samples": int(len(unique_selected)),
            "coverage_ratio": float(len(unique_selected) / total_source_safe),
            "coverage_gap_vs_full": float(1.0 - (len(unique_selected) / total_source_safe)),
            "duplicate_draws": 0,
            "duplicate_rate": 0.0,
            "downsample_days": 0,
            "upsample_days": 0,
            "exact_days": int(exact_days),
            "short_days": int(short_days),
            "split_days": int(split_days),
            "downsample_day_ratio": 0.0,
            "upsample_day_ratio": 0.0,
            "short_day_ratio": float(short_days / num_days),
            "split_day_ratio": float(split_days / num_days),
            "full_coverage_day_ratio": 1.0 if total_source > 0 else float("nan"),
        }

    def __len__(self):
        return int(self._num_batches)
