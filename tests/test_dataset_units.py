from __future__ import annotations

import pytest

import pandas as pd

from scripts.build_ecotox_dataset import (
    clean_concentration_unit,
    concentration_to_molar,
    deterministic_rejection_flag,
    pubchem_url,
)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("nM", 1e-9),
        ("umol/dm3", 1e-6),
        ("AI mg/L", 1e-5),
        ("mg/dm3", 1e-5),
        ("ug/mL", 1e-5),
        ("ng/mL", 1e-8),
    ],
)
def test_concentration_unit_equivalents(unit: str, expected: float) -> None:
    assert concentration_to_molar(1.0, unit, 100.0) == pytest.approx(expected)


def test_unit_normalization_handles_micro_and_spacing() -> None:
    assert clean_concentration_unit("  µg / L ") == "ug/l"


@pytest.mark.parametrize("unit", ["AE mg/L", "TOT mg/L"])
def test_qualified_mass_units_are_not_converted_as_parent_compound(unit: str) -> None:
    assert concentration_to_molar(1.0, unit, 100.0) is None


def test_carbon_free_inorganic_structure_triggers_deterministic_rejection() -> None:
    row = pd.Series(
        {
            "chemical_class": "unclassified",
            "chemical_name": "Disodium hexafluorosilicate(2-)",
            "smiles": "[Na+].[Na+].F[Si--](F)(F)(F)(F)F",
        }
    )
    assert deterministic_rejection_flag(row)


@pytest.mark.parametrize("name", ["Lauryl betaine, inner salt", "Lauryl betaine, INNER-SALT"])
def test_resolved_single_molecule_inner_salt_is_not_rejected_by_name(name: str) -> None:
    row = pd.Series(
        {
            "chemical_class": "unclassified",
            "chemical_name": name,
            "smiles": "CCCCCCCCCCCC[N+](C)(C)CC([O-])=O",
        }
    )
    assert not deterministic_rejection_flag(row)


@pytest.mark.parametrize(
    ("name", "smiles"),
    [
        ("Unresolved salt", "CCO"),
        ("Sodium acetate salt", "[Na+].CC([O-])=O"),
        ("Claimed inner salt", "CCO"),
        ("Claimed inner salt", "C[N+](C)(C)C.[Cl-]"),
        ("Claimed inner salt", "C[N+](C)(C)C"),
        ("Claimed inner salt", "[Na+].[Cl-]"),
        ("Claimed inner salt", "not-a-smiles"),
        ("Inner salt mixture", "CCCCCCCCCCCC[N+](C)(C)CC([O-])=O"),
        ("Unknown inner salt", "CCCCCCCCCCCC[N+](C)(C)CC([O-])=O"),
    ],
)
def test_inner_salt_name_does_not_bypass_other_eligibility_checks(
    name: str, smiles: str,
) -> None:
    row = pd.Series(
        {"chemical_class": "unclassified", "chemical_name": name, "smiles": smiles}
    )
    assert deterministic_rejection_flag(row)


def test_inner_salt_exception_does_not_override_metal_class_rejection() -> None:
    row = pd.Series(
        {
            "chemical_class": "metals",
            "chemical_name": "Inner salt",
            "smiles": "CCCCCCCCCCCC[N+](C)(C)CC([O-])=O",
        }
    )
    assert deterministic_rejection_flag(row)


def test_pubchem_lookup_uses_hyphenated_casrn() -> None:
    assert "/3240-78-6/" in pubchem_url("3240786")
    assert "/3240-78-6/" in pubchem_url("3240-78-6")
    assert "/Antimycin%20A/" in pubchem_url("Antimycin A")


@pytest.mark.parametrize("verified", [None, False, "false", "", "pending"])
def test_pubchem_structure_without_identity_verification_is_not_eligible(verified) -> None:
    row = pd.Series({"chemical_name": "Resolved-looking name", "smiles": "CCO",
        "structure_source": "pubchem", "identity_verified": verified})
    assert deterministic_rejection_flag(row)


def test_verified_pubchem_identity_still_requires_a_usable_structure() -> None:
    row = pd.Series({"chemical_name": "Ethanol", "smiles": "CCO",
        "structure_source": "pubchem", "identity_verified": True,
        "identity_evidence_url": "https://comptox.epa.gov/dashboard/chemical/details/DTXSID_TEST"})
    assert not deterministic_rejection_flag(row)
    row["smiles"] = "not-a-smiles"
    assert deterministic_rejection_flag(row)


def test_pubchem_verified_flag_without_source_evidence_is_not_eligible() -> None:
    row = pd.Series({"chemical_name": "ethanol", "smiles": "CCO",
                     "structure_source": "pubchem", "identity_verified": True})
    assert deterministic_rejection_flag(row)


@pytest.mark.parametrize("smiles", ["CC*", "CCCCCCCCC*.O*.C1=CC=CC=C1", "[*]C1CCCCC1"])
def test_generalized_structure_is_not_a_fixed_molecular_input(smiles: str) -> None:
    row = pd.Series({"chemical_name": "Registered chemical", "smiles": smiles,
                     "structure_source": "dsstox"})
    assert deterministic_rejection_flag(row)


def test_concrete_cxsmiles_stereochemistry_is_not_a_generalized_structure() -> None:
    row = pd.Series({"chemical_name": "Cyclohexene", "smiles": "C1=CCCCC1 |c:0|",
                     "structure_source": "dsstox"})
    assert not deterministic_rejection_flag(row)
