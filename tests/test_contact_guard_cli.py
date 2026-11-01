"""CLI guard opt-in and final-file status; numerical tests run remotely."""
import json

import pytest

from scripts.pdm_control import refine_geometry_structures as cli
from test_pdm_geometry_refine_cli import setup_case, stub, rows, assert_all_rows
from test_pdm_geometry_contacts_cli import stage_stub, assert_final_pdm


def guard_receipt():
    return dict(enabled=True, initial_feasible=True, ever_feasible=True,
                final_feasible=True, rejected_proposals=4, history_complete=True)


def test_cli_guard_requires_contact_weight_before_output_creation(tmp_path):
    source, model, out, profile, args = setup_case(tmp_path)
    assert cli.main(args+['--preserve-feasible-contacts']) == 1
    assert not out.exists()


def test_cli_programmatic_nonboolean_guard_is_rejected(tmp_path):
    source, model, out, profile, args = setup_case(tmp_path)
    parsed = cli.parser().parse_args(args+['--contact-weight', '1'])
    parsed.preserve_feasible_contacts = 1
    with pytest.raises(ValueError, match='boolean'):
        cli.run(parsed)
    assert not out.exists()


def test_cli_default_has_no_guard_fields(tmp_path, monkeypatch):
    source, model, out, profile, args = setup_case(tmp_path)
    seen = []
    def fake(*inputs, **kwargs):
        seen.append(kwargs)
        return stub(status='stationary')(*inputs, **kwargs)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', fake)
    assert cli.parser().parse_args(args).preserve_feasible_contacts is False
    assert cli.main(args) == 0
    assert 'preserve_feasible_contacts' not in seen[0]
    assert not any(key.startswith('contact_guard_') for key in rows(out/'status.csv')[0])
    assert 'preserve_feasible_contacts' not in json.loads((out/'run.json').read_text())


@pytest.mark.parametrize('stages', [False, True])
def test_cli_guard_forwarding_counts_and_partial_coverage(stages, tmp_path, monkeypatch):
    source, model, out, profile, args = setup_case(tmp_path)
    seen = []; audit = tmp_path/'audit'
    if stages:
        base = stage_stub(seen=seen)
        def fake(*inputs, **kwargs):
            result = base(*inputs, **kwargs)
            result.update(preserve_feasible_contacts=True, contact_feasibility_guard=guard_receipt())
            for stage in result['stages']:
                stage['contact_feasibility_guard'] = guard_receipt()
            return result
        monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh', fake)
    else:
        def fake(*inputs, **kwargs):
            seen.append((None, None, kwargs))
            result = stub(status='stationary')(*inputs, **kwargs)
            result['reports'][0]['contact_feasibility_guard'] = guard_receipt()
            result['reports'][0]['history'][0]['contact_guard_evidence'] = {'proposal_frac_coords': [[.1, .2, .3]]}
            return result
        monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', fake)
    extra = ['--topology-stage-steps', '1', '1'] if stages else []
    assert cli.main(args+['--contact-weight', '1', '--preserve-feasible-contacts', '--audit-dir', str(audit)]+extra) == 1
    assert seen[0][2]['preserve_feasible_contacts'] is True
    state = rows(out/'status.csv')[0]
    assert state['contact_guard_enabled'] == 'True'
    assert state['contact_guard_rejected_proposals'] == '4'
    assert state['contact_guard_evidence_complete'] == 'True'
    assert state['contact_guard_final_feasible'] == 'True'
    assert state['status'] == 'partial_contact_reference_coverage'
    assert json.loads((out/'run.json').read_text())['preserve_feasible_contacts'] is True
    public = ''.join((out/name).read_text() for name in ('run.json', 'status.csv', 'geometry.csv', 'pdm.csv'))
    assert 'proposal_frac_coords' not in public
    assert (audit/'0.vasp.json').exists()
    assert_final_pdm(out, model)


def test_cli_missing_guard_receipt_retains_error_and_all_rows(tmp_path, monkeypatch):
    source, model, out, profile, args = setup_case(tmp_path, ('0.vasp', '1.vasp'))
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', stub(status='stationary'))
    assert cli.main(args+['--contact-weight', '1', '--preserve-feasible-contacts']) == 1
    assert_all_rows(out, ('0.vasp', '1.vasp'))
    for row in rows(out/'status.csv'):
        assert row['status'] == 'error' and row['coordinate_state'] == 'last_iterate'
        assert 'guard evidence' in row['error']
        assert row['contact_guard_evidence_complete'] == 'False'
    assert_final_pdm(out, model)


def test_cli_export_recheck_does_not_mask_guard_regression(tmp_path, monkeypatch):
    source, model, out, profile, args = setup_case(tmp_path)
    def fake(*inputs, **kwargs):
        result = stub(status='stationary')(*inputs, **kwargs)
        result['frac_coords'][1] = result['frac_coords'][0]+.01
        result['reports'][0]['contact_feasibility_guard'] = guard_receipt()
        return result
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', fake)
    assert cli.main(args+['--contact-weight', '1', '--preserve-feasible-contacts']) == 1
    state = rows(out/'status.csv')[0]
    assert state['status'] == 'error' and state['coordinate_state'] == 'last_iterate'
    assert state['contact_guard_final_feasible'] == 'False'
    assert 'preserve latched' in state['error']
    assert rows(out/'pdm.csv')[0]['pdm_status'] == 'computed'
