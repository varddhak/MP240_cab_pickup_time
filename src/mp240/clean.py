import pandas as pd

from .config import load_params


def clean_train(df, params=None):
    p = params or load_params()
    c = p["cleaning"]
    df = df.drop_duplicates().copy()

    # Old City typed rain in cm, so divide by the factor to get mm
    u = c["unit_fix"]
    fix = df[u["group_col"]] == u["group_value"]
    df.loc[fix, u["column"]] = (df.loc[fix, u["column"]] / u["factor"]).round(2)

    # stuck surge: blank those rows, then fill with the median of the rest
    s = c["stuck"]
    dates = pd.to_datetime(df[p["data"]["date_col"]])
    stuck = (dates >= s["start_date"]) & (dates <= s["end_date"])
    df.loc[stuck, s["column"]] = float("nan")
    df[s["column"]] = df[s["column"]].fillna(df[s["column"]].median())

    return df.reset_index(drop=True)