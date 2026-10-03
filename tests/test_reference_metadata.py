import pandas as pd

from scripts.run_reference_holdout_validation import (
    ReferenceAnnotations,
    _attach_reference_metadata,
)


def test_ambiguous_reference_matches_share_connected_group():
    def row(cas, value):
        return dict(casrn=cas,species='species A',endpoint_code='LC50',effect='MOR',
                    measurement='MORT',duration_h=96.0,medium='FW',study_year=2000.0,
                    doi='',toxicity_value=value,toxicity_unit='mg/L')
    processed=pd.DataFrame([row('1',1.0),row('2',2.0),row('3',3.0)])
    raw=[]
    for i,references in enumerate([['A','B'],['B','C'],['D']]):
        for reference in references:
            item=processed.iloc[i].to_dict()
            item.update(casrn_key=item['casrn'],reference_number=reference,
                        test_id=f'test-{i}-{reference}',study_type='LAB',
                        test_location='LAB',exposure_type='STATIC')
            raw.append(item)
    result,metadata=_attach_reference_metadata(processed,
        ReferenceAnnotations(pd.DataFrame(raw),{}))
    assert len(result)==len(processed)
    assert result.reference_number.iloc[0]==result.reference_number.iloc[1]
    assert result.reference_number.iloc[2]!=result.reference_number.iloc[0]
    assert metadata.reference_numbers.tolist()==['A;B','B;C','D']
