from scripts.calibrate_global_search_thresholds import LabeledScore, calibrate_resource


def test_calibration_selects_thresholds_that_reject_negative_samples() -> None:
    samples = [
        LabeledScore("users", True, 0.80, None),
        LabeledScore("users", True, 0.70, 0.20),
        LabeledScore("users", False, 0.10, 0.01),
        LabeledScore("users", False, 0.20, 0.02),
    ]

    result = calibrate_resource(samples, target_precision=1.0)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["false_positives"] == 0
