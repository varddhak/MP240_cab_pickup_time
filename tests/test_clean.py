import pandas as pd
import pytest

from mp240.checks import duplicate_rows
from mp240.clean import clean_train


def make_df():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R2", "R3", "R4", "R5"],
        "date": ["2026-03-01", "2026-03-08", "2026-03-08",
                 "2026-03-10", "2026-03-30", "2026-04-01"],
        "zone": ["Airport", "Old City", "Old City", "Tech Park", "Old City", "Airport"],
        "rain_mm": [30.0, 3.0, 3.0, 20.0, 4.0, 10.0],
        "surge": [1.5, 2.26, 2.26, 2.26, 1.9, 2.1],
    })


def test_checks_count_duplicates():
    assert duplicate_rows(make_df()) == 1


def test_clean_drops_duplicates():
    assert len(clean_train(make_df())) == 5


def test_clean_fixes_old_city_rain_only():
    out = clean_train(make_df())
    assert out.loc[out.zone == "Old City", "rain_mm"].tolist() == pytest.approx([30.0, 40.0])
    assert out.loc[out.zone == "Airport", "rain_mm"].tolist() == pytest.approx([30.0, 10.0])


def test_clean_replaces_stuck_surge():
    out = clean_train(make_df())
    assert out["surge"].isna().sum() == 0
    assert (out["surge"] == 2.26).sum() == 0


def test_clean_does_not_modify_input():
    df = make_df()
    clean_train(df)
    assert len(df) == 6
    assert df["rain_mm"].tolist()[1] == 3.0