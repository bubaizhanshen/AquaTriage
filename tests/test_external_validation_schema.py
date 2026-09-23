import numpy as np
import pandas as pd
import pytest

from scripts.run_echa_pmra_external_validation import align_external_input_columns


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chemical_id": ["external-1"],
            "chemical_name": ["Example chemical"],
            "casrn": ["10-00-0"],
            "smiles": ["CCO"],
            "endpoint": ["fish_96h_lc50"],
            "species": ["Oryzias latipes"],
            "target_log_molar": [-2.0],
        }
    )


def test_external_panel_missing_optional_columns_are_explicitly_added() -> None:
    train = _panel().assign(mech_proxy=[1.0], ctx_hardness=[20.0])
    aligned, audit = align_external_input_columns(train, _panel())

    assert "mech_proxy" in aligned.columns
    assert "ctx_hardness" in aligned.columns
    assert aligned.loc[0, "mech_proxy"] is np.nan or pd.isna(aligned.loc[0, "mech_proxy"])
    assert set(audit["column"]) == {"mech_proxy", "ctx_hardness"}
    assert (audit["action"] == "added_as_missing_and_training_partition_imputed").all()


def test_external_panel_missing_core_field_is_rejected() -> None:
    train = _panel().assign(mech_proxy=[1.0])
    panel = _panel().drop(columns="smiles")

    with pytest.raises(ValueError, match="smiles"):
        align_external_input_columns(train, panel)
