import math
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedDailyBatchSampler(Sampler):
    """
    SOTA Distributed Sampler for Quant:
    1. [Logic]: Splits data by DATE across GPUs (Rank 0 gets Day A, Rank 1 gets Day B).
       - Ensures RankNet/IC Loss validity (comparisons within the same day).
       - Provides Regime Diversity (different GPUs learn different market states).

    2. [Memory]: Enforces Fixed Batch Size via Upsampling/Downsampling.
       - Prevents OOM on days with 4000+ stocks.
       - Stabilizes BatchNorm/LayerNorm statistics.

    3. [DDP]: Handles Epoch Shuffling & Padding.
       - Globally shuffles dates every epoch so GPUs see different days over time.
       - Pads the number of days so all GPUs have equal #batches (prevents DDP hanging).
    """

    def __init__(self, data_source, batch_size: int, num_replicas: int = None, rank: int = None, shuffle: bool = True,
                 seed: int = 0):
        """
        Args:
            data_source: Qlib TSDataset or object with .get_index()
            batch_size: Fixed tensor size for GPU
            num_replicas: World size (default: auto-detect)
            rank: Current GPU rank (default: auto-detect)
            shuffle: Whether to shuffle dates globally
            seed: Random seed for reproducibility
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # 1. Auto-detect DDP settings
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        # 2. Build Date -> Indices Map efficiently
        try:
            # Qlib TSDatasetH usually has index access
            index = data_source.get_index()
        except AttributeError:
            # Fallback for subsets or wrappers
            index = data_source.dataset.get_index()

        # Reset index to get integer locations [0, 1, 2, ...]
        df_idx = pd.DataFrame(index=index).reset_index()
        df_idx['int_idx'] = np.arange(len(df_idx))

        # Group by datetime to get list of [array(indices_day_1), array(indices_day_2), ...]
        # This list represents "All available trading days"
        self.daily_groups = df_idx.groupby('datetime')['int_idx'].apply(np.array).tolist()

        # 3. Handle DDP Partitioning Logic
        # We need total_size to be divisible by num_replicas to ensure equal batches
        self.num_samples = int(math.ceil(len(self.daily_groups) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 1. Deterministic shuffling based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Indices of days: [0, 1, ..., num_days-1]
        indices = torch.arange(len(self.daily_groups)).tolist()

        if self.shuffle:
            indices = torch.randperm(len(self.daily_groups), generator=g).tolist()

        # 2. Padding (Ensure all GPUs have same number of days)
        # If dates are not enough, loop back to the beginning
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        # 3. Subsample for current Rank
        # Rank 0 gets: [0, 4, 8...], Rank 1 gets: [1, 5, 9...]
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        # 4. Generate Batches
        for day_idx in indices:
            # Retrieve the actual sample indices for this day
            daily_indices = self.daily_groups[day_idx]
            n_samples = len(daily_indices)

            # 5. Fixed Batch Size Logic (Upsample/Downsample)
            if n_samples >= self.batch_size:
                # Downsample: Randomly pick subset (Unbiased)
                # Note: We use numpy for fast choice, seeding is controlled by process
                batch_indices = np.random.choice(daily_indices, self.batch_size, replace=False)
            else:
                # Upsample: Randomly pick with replacement (Padding)
                batch_indices = np.random.choice(daily_indices, self.batch_size, replace=True)

            yield batch_indices

    def __len__(self):
        # Return the number of batches (days) this GPU will process per epoch
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler.
        When :attr:`shuffle=True`, this ensures all replicas use a different random ordering
        for each epoch. Otherwise, the next iteration of this sampler will yield the same ordering.
        """
        self.epoch = epoch