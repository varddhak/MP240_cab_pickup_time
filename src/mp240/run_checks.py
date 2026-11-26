import pandas as pd

from .checks import (duplicate_rows, leaking_column, sentinel_value,
                     spelling_variants, stuck_sensor, target_typos, unit_mixup)
from .config import ROOT, load_params


def main():
    p = load_params()
    d, ck = p["data"], p["checks"]
    train = pd.read_csv(ROOT / d["train"])
    later = pd.read_csv(ROOT / d["later"])
    hold = pd.read_csv(ROOT / d["holdout"])
    target, group, date = d["target"], d["group_col"], d["date_col"]

    num_cols = list(train.select_dtypes("number").columns)
    text_cols = [c for c in train.select_dtypes("object").columns
                 if c not in (d["id_col"], date)]

    print("rows:", len(train), "\n")
    print("1 leaking_column\n", leaking_column(train, hold, target), "\n")
    print("2 unit_mixup (top 6)\n", unit_mixup(train, group, num_cols).head(6), "\n")
    print("3 sentinel_value:", sentinel_value(train, num_cols, ck["sentinels"]), "\n")
    print("4 duplicate_rows (extra copies):", duplicate_rows(train), "\n")
    print("5 stuck_sensor:", stuck_sensor(train, date, num_cols, ck["stuck_min_rows"]), "\n")
    print("6 target_typos:", target_typos(train, target), "\n")
    print("7 spelling_variants:", spelling_variants(train, later, text_cols))


if __name__ == "__main__":
    main()