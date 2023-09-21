import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def load_data():
    file = "../datasets/air_quality_uci.csv"

    data = pd.read_csv(
        file,
        sep=",",
        usecols=["Date_Time", "CO_sensor", "RH"],
        index_col=["Date_Time"],
    )
    data.index = pd.to_datetime(data.index, format="%d/%m/%Y %H:%M:%S")
    data.columns = data.columns.str.lower()
    data.index.name = data.index.name.lower()

    data = data.sort_index()
    data = data.loc["2004-04-01":"2005-04-30"]
    data = data.loc[(data["co_sensor"] >= 0) & (data["rh"] >= 0)]

    return data


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy["year"] = X_copy.index.year
        X_copy["month"] = X_copy.index.month
        X_copy["day"] = X_copy.index.day
        X_copy["hour"] = X_copy.index.hour
        X_copy["minute"] = X_copy.index.minute
        X_copy["second"] = X_copy.index.second
        X_copy["day_of_week"] = X_copy.index.dayofweek
        X_copy["day_of_year"] = X_copy.index.dayofyear

        return X_copy


if __name__ == "__main__":
    df = load_data()
    print("=" * 80)
    print("Antes de adicionar as features de data e hora")
    print(df.head())
    print("=" * 80)

    datetime_transformer = DatetimeFeatures()
    df = datetime_transformer.fit_transform(df)

    print("Apos de adicionar as features de data e hora")
    print(df.head())
    print("=" * 80)
