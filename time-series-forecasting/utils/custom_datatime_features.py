import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def load_data():
    file = "../datasets/air_quality_uci.csv"

    data = pd.read_csv(
        file,
        sep=",",
        usecols=["Date_Time", "CO_sensor", "RH"],
        indedf_col=["Date_Time"],
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

    def transform(self, df):
        df_copy = df.copy()

        df_copy["year"] = df_copy.index.year
        df_copy["month"] = df_copy.index.month
        df_copy["day"] = df_copy.index.day
        df_copy["hour"] = df_copy.index.hour
        df_copy["minute"] = df_copy.index.minute
        df_copy["second"] = df_copy.index.second
        df_copy["day_of_week"] = df_copy.index.dayofweek
        df_copy["day_of_year"] = df_copy.index.dayofyear

        return df_copy


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
