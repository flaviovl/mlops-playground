from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import (
    LagFeatures,
    WindowFeatures,
)
from sklearn.pipeline import Pipeline


def create_datetime_features():
    return DatetimeFeatures(
        variables="index",
        features_to_extract=[
            "month",
            "week",
            "day_of_week",
            "day_of_month",
            "hour",
            "weekend",
        ],
    )


def create_lag_features():
    return LagFeatures(
        variables=["co_sensor", "rh"],
        freq=["1H", "24H"],
        missing_values="ignore",
    )


def create_window_features():
    return WindowFeatures(
        variables=["co_sensor", "rh"],
        window="3H",
        freq="1H",
        missing_values="ignore",
    )


def create_cyclical_features():
    return CyclicalFeatures(
        variables=["month", "hour"],
        drop_original=False,
    )


def create_imputer():
    return DropMissingData()


def create_pipeline(features):
    return Pipeline(
        [
            ("datetime", create_datetime_features()),
            ("lag", create_lag_features()),
            ("window", create_window_features()),
            ("cyclical", create_cyclical_features()),
            ("impute", create_imputer()),
            ("drop", DropFeatures(columns=["index"])),
        ]
    )
