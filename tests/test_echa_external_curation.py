import pandas as pd

from scripts.build_echa_pmra_clean_panel import (
    has_resolved_molecular_identity,
    has_usable_molecular_representation,
    is_acceptable_study_record,
)
from scripts.build_echa_pmra_external_rows import (
    parse_effect_value, parse_duration_hours, section_anchor_documents, linked_field_documents,
)


def test_study_links_without_optional_icons_are_retained() -> None:
    html = '''<div id="section">
      <a class="das-leaf" rel="host" href="doc_one">001 | Key | Experimental study</a>
      <a class="das-leaf" rel="host" href="doc_two">Endpoint summary</a>
      <a class="das-leaf" rel="host" href="doc_three"><i class="icon-item icon-ENDPOINT_SUMMARY"></i>002 | Summary</a>
      <a class="das-leaf" rel="host" href="doc_four"><i class="icon-item icon-ENDPOINT_STUDY_RECORD"></i>Study</a>
    </div>'''
    assert [d['is_study_record'] for d in section_anchor_documents(html, 'section')] == [True, False, False, True]


def test_duration_accepts_days_but_not_ranges() -> None:
    assert parse_duration_hours('4 d') == 96
    assert parse_duration_hours('48 hours') == 48
    assert parse_duration_hours('72 - 96 h') is None


def test_nested_reference_links_are_preserved_without_treating_labels_as_content() -> None:
    html = '''<div class="das-field"><div class="das-field_label">Reference</div>
    <div class="das-field_value"><a class="das-field_reference-link" href="ref_uuid">Reference</a></div></div>
    <div class="das-field"><div class="das-field_label">Test material information</div>
    <a class="das-field_reference-link" href="material_uuid">Test material information</a></div>'''
    assert linked_field_documents(html, 'Reference') == ['ref_uuid']
    assert linked_field_documents(html, 'Test material information') == ['material_uuid']


def test_effect_ranges_and_approximations_are_not_exact_values() -> None:
    for text in ['1 - 5 mg/L', '1 to 5 mg/L', '~ 5 mg/L', 'ca. 5 mg/L']:
        value, unit, not_exact = parse_effect_value(text)
        assert value is None
        assert unit == 'mg/L'
        assert not_exact
    assert parse_effect_value('1e-3 mg/L') == (0.001, 'mg/L', False)


def test_effect_parser_does_not_invent_positive_or_integer_concentrations() -> None:
    assert parse_effect_value('-1 mg/L')[0] is None
    assert parse_effect_value('0 mg/L')[0] is None
    assert parse_effect_value('1,2 mg/L')[0] is None
    assert parse_effect_value('.5 mg/L') == (0.5, 'mg/L', False)


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
