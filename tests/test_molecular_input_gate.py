from __future__ import annotations

import pandas as pd
import pytest

from ecoood.features import binary_fingerprints


@pytest.mark.parametrize("smiles", [None, "", "not-a-smiles"])
def test_fingerprint_builder_rejects_unresolved_molecular_input(smiles: str | None) -> None:
    with pytest.raises(ValueError, match="resolve or reject"):
        binary_fingerprints(pd.Series([smiles]), n_bits=64)
