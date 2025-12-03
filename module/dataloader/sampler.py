import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class FixedDailyBatchSampler(Sampler):
    """
    SOTA Sampling Strategy:
    1. Group data by date.
    2. Randomly select a date.
    3. Randomly sample 'batch_size' instruments from that date.
       - If daily_count > batch_size: Downsample (Unbiased estimator of gradients)
       - If daily_count < batch_size: Upsample (Padding to ensure tensor shape stability)
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

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
            np.random.shuffle(indices)

        for day_i in indices:
            daily_indices = self.daily_groups[day_i]
            n_samples = len(daily_indices)

            if n_samples >= self.batch_size:
                # Downsample: Randomly pick subset
                batch_indices = np.random.choice(daily_indices, self.batch_size, replace=False)
            else:
                # Upsample: Randomly pick with replacement to fill batch
                batch_indices = np.random.choice(daily_indices, self.batch_size, replace=True)

            yield batch_indices

    def __len__(self):
        return self.num_batches