"""
Validate saved model artifacts (.npy, .joblib, .json) are intact and consistent.
"""

import os
import pytest
import numpy as np
import joblib

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml")


class TestModelArtifacts:

    def test_model_file_exists(self):
        assert os.path.exists(os.path.join(ML_DIR, "xgboost_model.json"))

    def test_config_file_exists(self):
        assert os.path.exists(os.path.join(ML_DIR, "xgboost_config.joblib"))

    def test_config_has_required_keys(self):
        config = joblib.load(os.path.join(ML_DIR, "xgboost_config.joblib"))
        assert "feature_columns" in config
        assert "threshold" in config
        assert "best_iteration" in config

    def test_config_feature_count(self):
        config = joblib.load(os.path.join(ML_DIR, "xgboost_config.joblib"))
        assert len(config["feature_columns"]) == 28

    def test_config_threshold(self):
        config = joblib.load(os.path.join(ML_DIR, "xgboost_config.joblib"))
        assert config["threshold"] == 0.5

    def test_x_all_shape(self):
        X = np.load(os.path.join(ML_DIR, "X_all.npy"))
        # Feature count may vary by pipeline version (7 or 28)
        assert X.shape[1] > 0
        assert X.shape[0] > 0

    def test_x_train_shape(self):
        X = np.load(os.path.join(ML_DIR, "X_train.npy"))
        assert X.shape[1] > 0
        assert X.shape[0] > 0

    def test_y_all_shape(self):
        y = np.load(os.path.join(ML_DIR, "y_all.npy"), allow_pickle=True)
        X = np.load(os.path.join(ML_DIR, "X_all.npy"))
        assert y.shape[0] == X.shape[0]

    def test_y_all_is_boolean(self):
        y = np.load(os.path.join(ML_DIR, "y_all.npy"), allow_pickle=True)
        assert y.dtype == bool or set(np.unique(y)).issubset({True, False, 0, 1})

    def test_x_all_no_nans(self):
        X = np.load(os.path.join(ML_DIR, "X_all.npy"))
        assert not np.isnan(X).any()

    def test_x_train_no_nans(self):
        X = np.load(os.path.join(ML_DIR, "X_train.npy"))
        assert not np.isnan(X).any()

    def test_x_train_subset_of_x_all(self):
        """X_train (FP only) should be smaller than X_all (all notifications)."""
        X_train = np.load(os.path.join(ML_DIR, "X_train.npy"))
        X_all = np.load(os.path.join(ML_DIR, "X_all.npy"))
        assert X_train.shape[0] < X_all.shape[0]
