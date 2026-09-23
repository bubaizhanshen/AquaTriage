import pandas as pd

from scripts.run_echa_pmra_external_validation import (
    RELIABILITY_METHODS,
    chemical_level_fixed_workload,
)


def test_external_chemical_level_uses_fixed_chemical_counts() -> None:
    rows = []
    for chemical_id in range(15):
        for endpoint in ("fish", "daphnid"):
            row = {
                "chemical_id": f"C{chemical_id:02d}",
                "endpoint": endpoint,
                "abs_error": float(chemical_id),
            }
            for score_col in RELIABILITY_METHODS.values():
                row[score_col] = float(chemical_id)
            row["ad_similarity_risk"] = float(14 - chemical_id)
            rows.append(row)

    result = chemical_level_fixed_workload(40, pd.DataFrame(rows)).set_index("method")

    assert (result["n_chemicals"] == 15).all()
    assert (result["review_n"] == 4).all()
    assert (result["high_error_n"] == 2).all()
    assert result.loc["prediction_error_risk", "high_error_capture_rate"] == 1.0
    assert result.loc["similarity_ad_risk", "high_error_capture_rate"] == 0.0
