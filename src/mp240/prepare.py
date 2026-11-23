import pandas as pd

from .clean import clean_train
from .config import ROOT, load_params


def main():
    p = load_params()
    raw = pd.read_csv(ROOT / p["data"]["train"])
    out = clean_train(raw, p)

    path = ROOT / p["data"]["train_clean"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    print("rows:", len(raw), "->", len(out))
    print(out.groupby(p["data"]["group_col"])["rain_mm"].mean().round(1))
    print("surge missing:", int(out["surge"].isna().sum()))


if __name__ == "__main__":
    main()