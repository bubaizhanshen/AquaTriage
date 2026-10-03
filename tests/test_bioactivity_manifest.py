from scripts.build_bioactivity_manifest import definitions


def test_bioactivity_manifest_covers_assay_inputs_not_coverage_count():
    fields = definitions()
    assert len(fields) == 34
    assert "mech_feature_count" not in fields
    assert fields["mech_cytotox_median_um"]["units"] == "micromol/L"
    assert fields["mech_h295r_bmd_min"]["units"] == "micromol/L"
    assert "not a validated" in fields["mech_h295r_active_endpoint_count"]["definition"]
