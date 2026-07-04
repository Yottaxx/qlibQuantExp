# module/utils/processors.py
"""Custom qlib processors (h-20260610-001).

DropExtremeLabel: per-day cross-sectional drop of the top/bottom `percent` extreme LABELS,
TRAIN-ONLY (wire it into handler `learn_processors` => applies to DK_L only; valid/test use DK_I
untouched, so evaluation stays on the full universe — no eval-side selection).

Motivation: limit-up/limit-down and extreme-event labels are noise, not learnable signal
(MASTER-aligned). Was once enabled at 5%/side (d6e82a9), stripped in 8afb0d0 (crash-cleanup,
not falsification). Order in learn_processors: DropnaLabel -> DropExtremeLabel -> CSZScoreNorm
(drop extremes BEFORE the z-score so the per-day stats aren't polluted by the tails).

CSRankAppend (L-4): per-day cross-sectional rank-percentile of each feature, APPENDED as new
feature channels (the global-normalized features are kept intact so regime info stays available;
this is NOT wholesale CS-normalization). Hands the CS-blind backbone the day-relative coordinate
that defines a cross-section, with zero inter-stock message passing. Wire into `infer_processors`
so it applies identically to train (DK_L) and eval (DK_I) => train==infer feature schema.
"""
from __future__ import annotations

import pandas as pd

from qlib.data.dataset.processor import Processor, get_group_columns


class DropExtremeLabel(Processor):
    """Drop per-day cross-sectional extreme labels (both tails), train-only.

    Args:
        fields_group: column group holding the label (default "label").
        percent: per-side drop fraction (0.025 => keep the middle 95% each day).
    """

    def __init__(self, fields_group: str = "label", percent: float = 0.025):
        super().__init__()
        self.fields_group = fields_group
        self.percent = float(percent)
        if not 0.0 <= self.percent < 0.5:
            raise ValueError(f"percent must be in [0, 0.5), got {self.percent}")

    def __call__(self, df):
        if self.percent <= 0.0:
            return df
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        label = df[cols[0]]
        # per-day cross-sectional percentile rank in (0,1]
        pct = label.groupby(level="datetime", group_keys=False).rank(pct=True)
        keep = (pct > self.percent) & (pct <= (1.0 - self.percent))
        return df[keep.values]

    def is_for_infer(self) -> bool:
        # learn-side only; never applied to inference data
        return False

    def readonly(self) -> bool:
        return False


class CSRankAppend(Processor):
    """Append per-day cross-sectional rank-percentile of each feature as NEW channels (L-4).

    For every feature column c and every trading day, compute the point-in-time
    cross-sectional percentile rank of c across that day's peers (rank(pct=True)),
    optionally centered to [-0.5, 0.5], and append it as a new feature column
    ``<c><suffix>``. The original (global-normalized) features are kept unchanged, so the
    regime/common-mode information consumed by RegimeContextEncoder/layer_summary is preserved;
    only NEW columns are added.

    Leakage / correctness invariants:
      - Point-in-time: rank is computed within a single ``datetime`` group only (same-day peers),
        with NO shifting and NO use of any other day => strictly past-safe for a T+k label.
      - Universe-consistent: applies identically on train (DK_L) and eval (DK_I) when wired into
        ``infer_processors`` => the full evaluation universe is ranked the same way as train, and
        the train==infer feature schema is preserved (both gain the same appended columns).
      - Rank is monotonic in the input, so it is invariant to the preceding per-feature
        RobustZScoreNorm for all non-clipped values. Place this AFTER RobustZScoreNorm+Fillna so
        (a) there are no NaNs to rank and (b) the appended rank columns are NOT re-normalized.
        Caveat: clipped tail values (|z|>clip) tie at the boundary, slightly compressing tail
        ranks; a pre-clip raw rank is a future refinement (matches ceiling_probe_csrank.py).

    Args:
        fields_group: column group to rank and append (default "feature").
        suffix: appended-column name suffix (default "__csr").
        center: subtract 0.5 so the coordinate is symmetric around 0 (default True).
    """

    def __init__(self, fields_group: str = "feature", suffix: str = "__csr", center: bool = True):
        super().__init__()
        self.fields_group = fields_group
        self.suffix = str(suffix)
        self.center = bool(center)

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        feat = df.loc[:, cols]
        # per-day cross-sectional percentile rank in (0,1]; same-day peers only (point-in-time).
        ranked = feat.groupby(level="datetime", group_keys=False).rank(pct=True)
        if self.center:
            ranked = ranked - 0.5
        # Defensive: post-Fillna there are no NaNs to rank, but if the processor order ever changes,
        # a NaN feature would yield a NaN rank. 0.0 == the centered median rank (neutral coordinate).
        ranked = ranked.fillna(0.0)
        # Preserve the column index structure (MultiIndex ("feature", name) for Alpha158) and keep
        # the appended rank columns INSIDE the feature group so all feature-group columns stay
        # contiguous (defends against any positional feature/label slicing downstream).
        is_multi = isinstance(cols[0], tuple)
        new_cols = [
            (tuple(list(c[:-1]) + [str(c[-1]) + self.suffix]) if is_multi else str(c) + self.suffix)
            for c in cols
        ]
        ranked.columns = pd.MultiIndex.from_tuples(new_cols) if is_multi else new_cols
        out = pd.concat([df, ranked], axis=1)
        # Reorder: original feature cols + appended rank cols, then everything else (e.g. label).
        other_cols = [c for c in df.columns if c not in set(cols)]
        ordered = list(cols) + list(ranked.columns) + list(other_cols)
        return out.loc[:, ordered]

    def is_for_infer(self) -> bool:
        # MUST run on inference too: train and eval need the identical appended schema.
        return True

    def readonly(self) -> bool:
        return False
