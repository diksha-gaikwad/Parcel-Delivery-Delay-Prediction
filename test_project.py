import pandas as pd
import pytest
from src.config import FEATURES
from src.train_model import convert_target


def test_target_conversion():
    assert convert_target(pd.Series(["yes", "no"])).tolist() == [1, 0]


def test_unknown_target_value_is_rejected():
    with pytest.raises(ValueError):
        convert_target(pd.Series(["yes", "unknown"]))


def test_outcome_fields_are_not_features():
    for field in ["delivery_id", "delivery_status", "delivery_time_hours", "delivery_rating"]:
        assert field not in FEATURES


def test_expected_features_are_present():
    assert "distance_km" in FEATURES
    assert "package_weight_kg" in FEATURES
    assert "weather_condition" in FEATURES
