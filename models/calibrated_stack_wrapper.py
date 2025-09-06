"""Pickle-safe calibrated stacked wrapper used by the pipeline.

This module holds the CalibratedStacked class so it can be imported and the
class referenced by a stable module path when joblib pickles the object.
"""
import numpy as np


class CalibratedStacked:
    """Top-level calibrated stacked wrapper that is pickleable."""
    def __init__(self, base_models, calibrator, feat_cols, classes_, base_names):
        self.base_models = base_models
        self.calibrator = calibrator
        self.feature_cols = feat_cols
        self.classes_ = classes_
        self._base_names = list(base_names)

    def predict_proba(self, X):
        # Accept DataFrame-like X; reindex and fill missing
        Xsub = X.reindex(columns=self.feature_cols).fillna(0.0)
        probs = [self.base_models[name].predict_proba(Xsub) for name in self._base_names]
        stacked = np.hstack(probs)
        return self.calibrator.predict_proba(stacked)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

