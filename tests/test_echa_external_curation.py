import pandas as pd

from scripts.build_echa_pmra_clean_panel import (
    has_resolved_molecular_identity,
    has_usable_molecular_representation,
    is_acceptable_study_record,
)
from scripts.build_echa_pmra_external_rows import parse_effect_value


def test_parse_effect_value_excludes_explicitly_censored_results() -> None:
    value, unit, is_censored = parse_effect_value("> 100 mg/L")

    assert value is None
    assert unit == "mg/L"
    assert is_censored


def test_parse_effect_value_supports_commas_and_scientific_notation() -> None:
    value, unit, is_censored = parse_effect_value("1,200e-3 ug/L")

    assert value == 1.2
    assert unit == "µg/L"
    assert not is_censored

    molar_value, molar_unit, molar_is_censored = parse_effect_value("0.25 µmol/L")
    assert molar_value == 0.25
    assert molar_unit == "µmol/L"
    assert not molar_is_censored


def test_external_study_adequacy_filter() -> None:
    assert is_acceptable_study_record("001 | Key | Experimental study")
    assert is_acceptable_study_record("002 | Supporting | Experimental study")
    assert is_acceptable_study_record(
        "003 | Weight of evidence | Experimental study"
    )
    assert not is_acceptable_study_record(
        "004 | Disregarded | Experimental study"
    )
    assert not is_acceptable_study_record("005 | Other | Calculation")
    assert not is_acceptable_study_record(
        "006 | No specified study adequacy | Experimental study"
    )


def test_external_molecular_eligibility_matches_application_gate() -> None:
    assert has_usable_molecular_representation("CCO")
    assert not has_usable_molecular_representation("[Na+].[Cl-]")
    assert not has_usable_molecular_representation("")
    assert not has_usable_molecular_representation("not-a-smiles")


def test_external_identity_gate_rejects_variable_composition_surrogates() -> None:
    resolved = pd.Series(
        {
            "chemical_name": "sodium acetate",
            "smiles": "[Na+].CC([O-])=O",
            "inchikey": "VMHLLURERBWHNL-UHFFFAOYSA-M",
        }
    )
    reaction_product = resolved.copy()
    reaction_product["chemical_name"] = "Reaction products of substance A and substance B"
    cxsmiles = resolved.copy()
    cxsmiles["smiles"] = "CCCCN(C)C |LN:3:1.2|"
    no_inchikey = resolved.copy()
    no_inchikey["inchikey"] = ""

    assert has_resolved_molecular_identity(resolved)
    assert not has_resolved_molecular_identity(reaction_product)
    assert not has_resolved_molecular_identity(cxsmiles)
    assert not has_resolved_molecular_identity(no_inchikey)
