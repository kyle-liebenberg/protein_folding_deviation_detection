"""Feature and target transforms, learned from TRAINING data only.

Feature steps, in order (justified in docs/01-data-preparation.md):
    1. select the input features (F2 is dropped, see src/data.py)
    2. log1p on the right-skewed features   -> roughly symmetric distributions
    3. clip to the training 0.5th-99.5th percentiles   -> outliers (mainly F7) can't dominate
    4. standardise to mean 0, std 1   -> every input on the same scale for gradient descent

Target: standardised for training (mean 0, std 1), converted back to Å for reporting.

Why fit on training data only: if the test data influenced the means, stds or clip bounds, the test
score would no longer be an honest estimate of performance on unseen data (data leakage).
"""

import numpy as np
import pandas as pd

from src.data import INPUT_FEATURES, TARGET

# Features with skewness > 1 in the raw data (F3 and F9 are already roughly symmetric).
# F2 is listed so the 9-feature comparison in Phase 2 transforms it too.
LOG_FEATURES = ["F1", "F2", "F4", "F5", "F6", "F7", "F8"]
CLIP_PERCENTILES = (0.5, 99.5)


class Preprocessor:
    """Call `fit` on training data, then `transform_X` / `transform_y` on any split.

    `features` defaults to our 8 chosen inputs. Pass a different list to re-check the feature decision.
    """

    def __init__(self, features=INPUT_FEATURES):
        self.features = list(features)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = self._log(df[self.features])
        self.clip_low = X.quantile(CLIP_PERCENTILES[0] / 100)
        self.clip_high = X.quantile(CLIP_PERCENTILES[1] / 100)
        X = X.clip(self.clip_low, self.clip_high, axis=1)
        self.x_mean, self.x_std = X.mean(), X.std()
        self.y_mean, self.y_std = df[TARGET].mean(), df[TARGET].std()
        return self

    def transform_X(self, df: pd.DataFrame) -> np.ndarray:
        X = self._log(df[self.features])
        X = X.clip(self.clip_low, self.clip_high, axis=1)
        X = (X - self.x_mean) / self.x_std
        return X.to_numpy(dtype=np.float32)

    def transform_y(self, df: pd.DataFrame) -> np.ndarray:
        """RMSD (Å) -> standardised target, shape (n, 1)."""
        y = (df[TARGET] - self.y_mean) / self.y_std
        return y.to_numpy(dtype=np.float32).reshape(-1, 1)

    def inverse_transform_y(self, y_scaled) -> np.ndarray:
        """Standardised predictions -> RMSD in Å, shape (n,)."""
        return np.asarray(y_scaled).reshape(-1) * self.y_std + self.y_mean

    @staticmethod
    def _log(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        to_log = [f for f in LOG_FEATURES if f in X.columns]
        X[to_log] = np.log1p(X[to_log])  # log1p(x) = log(1 + x): safe for the zeros in F7/F8
        return X
