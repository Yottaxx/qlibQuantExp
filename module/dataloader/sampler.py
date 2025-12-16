import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from typing import Optional


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
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

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

    def __iter__(self):
        indices = np.arange(self.num_batches)
        if self.shuffle:
            indices = self.rng.permutation(indices)

        for day_i in indices:
            daily_indices = self.daily_groups[day_i]
            n_samples = len(daily_indices)

            if n_samples >= self.batch_size:
                # Downsample: Randomly pick subset
                batch_indices = self.rng.choice(daily_indices, self.batch_size, replace=False)
            else:
                # Upsample: Randomly pick with replacement to fill batch
                batch_indices = self.rng.choice(daily_indices, self.batch_size, replace=True)

            yield batch_indices

    def __len__(self):
        return self.num_batches


class DailyChunkBatchSampler(Sampler):
    """
    Deterministic daily batch sampler (no sampling, no up/down-sampling).

    - Groups samples by their `datetime` index level.
    - For each day, yields consecutive chunks of indices with size <= max_batch_size.
    - Guarantees every sample appears exactly once per epoch.

    This is suitable for inference/predict where we want "intra-day batches" while
    preserving full coverage and the original sample order.
    """

    def __init__(self, data_source, max_batch_size: int):
        if max_batch_size is None or int(max_batch_size) <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.data_source = data_source
        self.max_batch_size = int(max_batch_size)

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

    def __iter__(self):
        for g in self.daily_groups:
            n = int(len(g))
            for i in range(0, n, self.max_batch_size):
                yield g[i : i + self.max_batch_size]

    def __len__(self):
        return int(self._num_batches)
