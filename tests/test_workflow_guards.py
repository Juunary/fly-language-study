import json
from pathlib import Path
import pytest
from flystudy.protocol import write_json, file_hash
from flystudy.workflow import verified_evidence
from flystudy.cli import main
from flystudy import cli
from flystudy.budget import Ledger
from flystudy.train import account_attempt


def test_stale_nested_evidence_is_rejected(tmp_path):
    source=tmp_path/'evidence.json'; write_json(source,{'passed':True})
    report={'evidence_files':{str(source):file_hash(source)}}
    assert verified_evidence(report)==report['evidence_files']
    write_json(source,{'passed':False})
    with pytest.raises(ValueError,match='Stale evidence'): verified_evidence(report)


@pytest.mark.parametrize('reason',['budget_interruption','technical_failure','debug_interruption'])
def test_campaign_child_cannot_succeed_after_interruption(monkeypatch,reason):
    monkeypatch.setattr(cli,'dispatch',lambda args:{'stop_reason':reason})
    assert main(['init','--output','unused']) == 2


def test_setup_failure_charged_once_and_retry_separate(tmp_path):
    ledger=Ledger(tmp_path/'ledger.json')
    ledger.reserve_bundle([dict(run_id='attempt-1',category='reserve',hours=.1),
                           dict(run_id='attempt-2',category='reserve',hours=.1)])
    @account_attempt
    def fails(output,run_id='original',cohort='pilot',seed=10001,mode='mono',order=('en',),
              reservation_id='attempt-1',resume=None,ledger_path=None,smoke=False,device='cuda:0'):
        raise RuntimeError('setup failed before model was ready')
    for rid in ('attempt-1','attempt-2'):
        with pytest.raises(RuntimeError,match='setup failed'):
            fails(tmp_path/'run',ledger_path=ledger.path,reservation_id=rid)
    records=json.loads(ledger.path.read_text())['runs']
    assert all(r['status']=='technical_failure' and r['used']>=0 for r in records.values())
    with pytest.raises(ValueError,match='not active'): ledger.reservation('attempt-1')


def test_terminal_resume_is_immutable(tmp_path):
    output=tmp_path/'run'; write_json(output/'summary.json',{'stop_reason':'mastered'})
    before=file_hash(output/'summary.json')
    @account_attempt
    def terminal(output,run_id='original',cohort='smoke',seed=7,mode='mono',order=('en',),
                 reservation_id=None,resume='primary.pt',ledger_path=None,smoke=True,device='cpu'):
        pytest.fail('Completed run must not be executed again')
    with pytest.raises(ValueError,match='terminal'): terminal(output)
    assert file_hash(output/'summary.json')==before


def test_unfilled_human_review_cannot_pass(tmp_path):
    from flystudy.data import generate
    from flystudy.gates import certify_review
    generate(tmp_path/'data',18,2)
    write_json(tmp_path/'attest.json',{})
    with pytest.raises(ValueError,match='reviewer'):
        certify_review(tmp_path/'data',tmp_path/'data'/'audit-sample.csv',tmp_path/'attest.json',tmp_path/'review.json')


def test_real_g0_cannot_pass_without_two_cuda_devices(tmp_path,monkeypatch):
    import torch
    from flystudy.graph import synthetic
    from flystudy.gates import g0
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    graph=synthetic(); graph.n=5000; graph.provenance['condition']='real'
    result=g0(graph,tmp_path/'g0.json')
    assert result['status']=='blocked' and result['checks']==[]


def test_auxiliary_shuffle_reserves_all_orders_both_topologies(tmp_path):
    from dataclasses import asdict
    from flystudy.auxplan import reserve_auxiliary
    from flystudy.protocol import Protocol,ORDERS
    from flystudy.graph import synthetic,scramble
    from flystudy.gates import code_hash
    p=Protocol(); graph=synthetic(); graph.save(tmp_path/'real.npz')
    other=scramble(graph,991,swaps_per_edge=1); other.save(tmp_path/'shuffled.npz')
    token=tmp_path/'token.json'; token.write_text('{}')
    write_json(tmp_path/'data'/'manifest.json',{'dataset_hash':'test-only'})
    write_json(tmp_path/'main.json',dict(status='frozen',protocol=asdict(p),code_hash=code_hash(),
        graph_hash=graph.hash,tokenizer_hash=file_hash(token),dataset_hash='test-only',evidence_files={}))
    result=reserve_auxiliary(tmp_path/'main.json','shuffle',tmp_path/'real.npz',tmp_path/'shuffled.npz',
        token,tmp_path/'data',tmp_path/'aux',tmp_path/'ledger.json',.1)
    assert len(result['runs'])==24
    for seed in (7001,7002):
        for variant in ('real','shuffle'):
            assert {tuple(r['order']) for r in result['runs'] if r['seed']==seed and r['variant']==variant}==set(ORDERS)
    assert len(json.loads((tmp_path/'ledger.json').read_text())['runs'])==24


def test_contingency_transfer_preserves_total_and_reservations(tmp_path):
    ledger=Ledger(tmp_path/'ledger.json')
    ledger.reserve_bundle([dict(run_id='retry',category='reserve',hours=20)])
    with pytest.raises(ValueError): ledger.transfer_reserve('g0_g1',13)
    ledger.transfer_reserve('g0_g1',12)
    state=json.loads(ledger.path.read_text())
    assert sum(state['allocations'].values())==672 and state['allocations']['reserve']==20
