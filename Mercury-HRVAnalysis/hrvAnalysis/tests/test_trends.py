from datetime import date, timedelta

import pandas as pd

from hrv_platform.trends import regression_stats


def test_regression_trend_improving():
    start = date(2026, 1, 1)
    df = pd.DataFrame(
        {
            "measurement_date": [start + timedelta(days=i) for i in range(10)],
            "rmssd": [10 + i for i in range(10)],
        }
    )
    stats = regression_stats(df, "rmssd")
    assert stats["trend_direction"] == "improving"
    assert stats["slope"] > 0
