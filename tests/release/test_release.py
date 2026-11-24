from pathlib import Path
import hashlib
import json
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from export_attention_case import compile_case,REGISTRY


def test_completed_registry_has_44_cases():assert len(REGISTRY)==44


@pytest.mark.parametrize('case',['p1_forward_cpu','p2_backward_gpu','paper3_grouped_openmp','p4_block_gpu'])
def test_case_export_reuses_closed_proof(tmp_path,case):
    result=compile_case(case,tmp_path/case)
    assert result['historical_case_certificate']['end_DNF_equal']
    for name,sha in result['files_sha256'].items():assert hashlib.sha256((tmp_path/case/name).read_bytes()).hexdigest()==sha
    assert json.loads((tmp_path/case/'03_lifted_onf.json').read_text())['physical_execution_certified'] is False
    if case=='p4_block_gpu':assert result['entry_profile']=='corrected fresh-device entry'


def test_unknown_case_fails_before_writing(tmp_path):
    out=tmp_path/'bad'
    with pytest.raises(ValueError):compile_case('not_a_case',out)
    assert not out.exists()
