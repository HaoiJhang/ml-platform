import json

import pandas as pd

from ml_platform.artifacts import artifact_to_dict
from ml_platform.data_flow import DataFlowTracker


def test_data_flow_tracker_serializes_preview_and_column_diffs() -> None:
    tracker = DataFlowTracker(target="target", preview_limit=2)
    first = pd.DataFrame(
        {
            "target": [1, 0, 1],
            "signup_at": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "segment": ["a", None, "c"],
        }
    )
    second = first.drop(columns=["segment"]).copy()
    second["score"] = [0.1, 0.2, 0.3]

    tracker.snapshot_dataframe("raw_dataset", "Raw dataset", "intake", first, preview=True)
    tracker.snapshot_dataframe("feature_subset", "Feature subset", "intake", second)

    payload = artifact_to_dict(tracker.to_trace())
    raw_snapshot = payload["snapshots"][0]
    next_snapshot = payload["snapshots"][1]

    assert raw_snapshot["preview"]["truncated"] is True
    assert raw_snapshot["preview"]["rows"][0]["signup_at"] == "2024-01-01T00:00:00"
    assert raw_snapshot["preview"]["rows"][1]["segment"] is None
    assert next_snapshot["columns_added"] == ["score"]
    assert next_snapshot["columns_removed"] == ["segment"]
    json.dumps(payload, ensure_ascii=False)
