from .config import load_params


def get_xy(df):
    p = load_params()
    f = p["features"]
    return df[f["numeric"] + f["categorical"]], df[p["data"]["target"]]