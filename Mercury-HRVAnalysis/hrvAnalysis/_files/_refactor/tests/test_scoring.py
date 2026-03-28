from hrv_platform.scoring import compute_ms_recovery_score


def test_ms_score_is_bounded():
    score = compute_ms_recovery_score(
        {
            "rmssd": 42,
            "sdnn": 50,
            "pNN50": 13,
            "SD1": 30,
            "SD2": 40,
            "LF": 1050,
            "HF": 820,
        }
    )
    assert 0 <= score <= 100
    assert round(score, 2) == 50.0


def test_ms_score_high_profile_clips():
    score = compute_ms_recovery_score(
        {
            "rmssd": 420,
            "sdnn": 500,
            "pNN50": 130,
            "SD1": 300,
            "SD2": 400,
            "LF": 10500,
            "HF": 8200,
        }
    )
    assert score == 100.0
