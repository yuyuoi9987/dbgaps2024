import gaps_config as gcfg
import fn_config as fncfg

import pandas as pd
import numpy as np

class Alpha:
    def __init__(self, author, weights_df):
        self.author = author
        self.weights_df = weights_df

        self._validate_format()
        self._validate_weights()
        self._validate_constraints()
        self._sort()

    def get_alpha(self):
        return self.weights_df
    
    def _validate_format(self):
        assert self.weights_df.columns in fncfg.ASSET_ALIASES.values(), "Invalid asset names"
        assert self.weights_df.index.is_monotonic_increasing, "Index is not sorted"

    def _validate_weights(self):
        assert self.weights_df.sum(axis=1).eq(1).all(), "Weights do not sum to 1"

    def _validate_constraints(self):
        # individual asset constraints
        assert self.weights_df.apply(lambda col: col.between( *gcfg.ASSET_WEIGHT_CONSTRAINTS[col.name] ), axis=0).all().all(), "Individual asset weight constraints violated"

        # group constraints
        for group, assets in gcfg.GROUP_ASSETS.items():
            assert self.weights_df[assets].sum(axis=1).between( *gcfg.GROUP_WEIGHT_CONSTRAINTS[group] ).all(), f"Group constraint violated for {group}"

    def _sort(self):
        self.weights_df = self.weights_df[fncfg.ASSET_ALIASES.values()]

    def to_numpy(self):
        return self.weights_df.to_numpy()