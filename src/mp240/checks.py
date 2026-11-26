import numpy as np
import pandas as pd


def leaking_column(train, holdout, target):
    """A leak is missing from holdout and almost equals the target."""
    rows = []
    for c in train.columns:
        if c == target:
            continue
        numeric = pd.api.types.is_numeric_dtype(train[c])
        corr = train[c].corr(train[target]) if numeric else None
        rows.append((c, c in holdout.columns, corr))
    return pd.DataFrame(rows, columns=["column", "in_holdout", "corr_with_target"])


def unit_mixup(df, group_col, num_cols):
    """Ratio of a group's median to everyone else's median, per column."""
    rows = []
    for c in num_cols:
        for g in df[group_col].unique():
            inside = df.loc[df[group_col] == g, c].median()
            outside = df.loc[df[group_col] != g, c].median()
            if outside > 0 and inside > 0:
                rows.append((c, g, round(inside / outside, 3)))
    out = pd.DataFrame(rows, columns=["column", "group", "ratio"])
    out["off_by"] = np.abs(np.log10(out["ratio"]))
    return out.sort_values("off_by", ascending=False)


def sentinel_value(df, num_cols, sentinels):
    found = []
    for c in num_cols:
        for v in sentinels:
            n = int((df[c] == v).sum())
            if n > 0:
                found.append((c, v, n))
    return found


def duplicate_rows(df):
    return int(df.duplicated().sum())


def stuck_sensor(df, date_col, num_cols, min_rows):
    """Long run of the exact same value when rows are sorted by date."""
    d = df.sort_values(date_col, kind="stable")
    found = []
    for c in num_cols:
        run_id = (d[c] != d[c].shift()).cumsum()
        for _, block in d.groupby(run_id):
            if len(block) >= min_rows:
                found.append(dict(column=c, value=block[c].iloc[0], rows=len(block),
                                  start_date=block[date_col].iloc[0],
                                  end_date=block[date_col].iloc[-1]))
    return found


def target_typos(df, target):
    """Values far above the rest (more than 5x the median)."""
    s = df[target].dropna()
    big = s[s > 5 * s.median()]
    return dict(count=len(big), largest_normal=s[s <= 5 * s.median()].max())


def spelling_variants(train, later, text_cols):
    """later.csv has the clean spellings; anything extra in train is a variant."""
    out = {}
    for c in text_cols:
        extra = sorted(set(train[c].dropna().unique()) - set(later[c].dropna().unique()))
        out[c] = dict(present=len(extra) > 0, odd_values=extra,
                      true_levels=int(later[c].nunique()))
    return out