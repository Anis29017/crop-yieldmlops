"""Model wrapper: predict yield as a change from the recent average.

Year-to-year yield changes are small, and yields drift upwards over time.
Tree models cannot predict values outside what they saw in training, and on
raw t/ha sugarcane (~60) would dominate the other crops (~0.5-3). So the
wrapped model learns

    log( yield / mean(yield of the previous 3 years) )

and predict() turns that back into t/ha. The saved model therefore takes the
normal feature table and returns yield in t/ha directly (the API needs nothing
special).
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

RECENT_COLS = ["prev_yield", "prev2_yield", "prev3_yield"]


def recent_average(X):
    """Mean yield of the previous 3 years (t/ha) for each row."""
    return X[RECENT_COLS].mean(axis=1).to_numpy(dtype=float)


class RecentYieldRatioRegressor(BaseEstimator, RegressorMixin):
    """Fits `estimator` on log(yield / recent average); predicts yield in t/ha."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        ref = recent_average(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.log(np.asarray(y, dtype=float) / ref))
        return self

    def predict(self, X):
        return recent_average(X) * np.exp(self.estimator_.predict(X))
