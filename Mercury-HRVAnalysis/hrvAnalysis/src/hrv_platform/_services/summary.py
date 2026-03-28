from __future__ import annotations

from sqlalchemy.orm import Session

from ..repository import HRVRepository
from ..scoring import MSRecoveryScorer
from ..trends import TrendAnalyzer
from ..anomalies import AnomalyDetector


def build_summary(session: Session, source_name: str = "MyHRV_import") -> dict:
    repo = HRVRepository(session)
    scorer = MSRecoveryScorer()
    trend_analyzer = TrendAnalyzer()
    anomaly_detector = AnomalyDetector()

    measurements = repo.get_recent_measurements(source_name=source_name, limit=90)

    if not measurements:
        return {
            "data_points": 0,
            "date_range": {"start": None, "end": None},
            "current_values": {},
            "recovery_scores": {"ms": 0.0},
            "baselines": {},
            "alerts": [],
            "anomalies": [],
            "trends": {},
        }

    baselines = repo.get_baselines(source_name=source_name)
    latest = measurements[0]

    current_values = {
        "SD1": latest.SD1,
        "SD2": latest.SD2,
        "sdnn": latest.sdnn,
        "rmssd": latest.rmssd,
        "pNN50": latest.pNN50,
        "VLF": latest.VLF,
        "LF": latest.LF,
        "HF": latest.HF,
    }

    ms_score = scorer.calculate(current_values)

    trend_data = trend_analyzer.analyze(measurements)
    anomalies = anomaly_detector.detect(measurements, baselines)

    return {
        "data_points": len(measurements),
        "date_range": {
            "start": measurements[-1].measurement_date.isoformat() if measurements[-1].measurement_date else None,
            "end": measurements[0].measurement_date.isoformat() if measurements[0].measurement_date else None,
        },
        "current_values": current_values,
        "recovery_scores": {"ms": ms_score},
        "baselines": baselines,
        "alerts": repo.get_recent_alerts(source_name=source_name),
        "anomalies": anomalies,
        "trends": trend_data,
    }