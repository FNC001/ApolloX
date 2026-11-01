"""Explicit projection forwarding and retained final-only failure rows."""
import json

import pytest

from scripts.pdm_control import refine_geometry_structures as cli
from test_contact_guard_cli import guard_receipt
from test_pdm_geometry_refine_cli import setup_case, stub, rows, assert_all_rows
from test_pdm_geometry_contacts_cli import stage_stub, assert_final_pdm


@pytest.mark.parametrize('extra', [['--contact-projection-sweeps', '32'],
    ['--contact-weight', '1', '--contact-projection-sweeps', '32'],
    ['--contact-weight', '1', '--preserve-feasible-contacts', '--contact-projection-sweeps', '33']])
def test_projection_requires_guard_and_bounded_budget(tmp_path, extra):
    source, model, out, profile, args = setup_case(tmp_path)
    assert cli.main(args+extra) == 1
    assert not out.exists()


def test_default_does_not_forward_projection_or_add_metadata(tmp_path, monkeypatch):
    source, model, out, profile, args = setup_case(tmp_path); seen = []
    def fake(*inputs, **kwargs):
        seen.append(kwargs)
        return stub(status='stationary')(*inputs, **kwargs)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', fake)
    assert cli.parser().parse_args(args).contact_projection_sweeps is None
    assert cli.main(args) == 0
    assert 'contact_projection_sweeps' not in seen[0]
    assert 'contact_projection_sweeps' not in json.loads((out/'run.json').read_text())


@pytest.mark.parametrize('stages', [False, True])
def test_explicit_forwarding_and_audit_not_public_raw(stages, tmp_path, monkeypatch):
    source, model, out, profile, args = setup_case(tmp_path); seen = []; audit = tmp_path/'audit'
    summary = dict(max_sweeps_per_direction=32, directions_attempted=1, actual_sweeps=2,
                   direction_statuses=['numerical_fixed_point'], nearest_projection_optimality_claimed=False)
    if stages:
        base = stage_stub(seen=seen)
        def fake(*inputs, **kwargs):
            result = base(*inputs, **kwargs)
            result.update(preserve_feasible_contacts=True, contact_projection_sweeps=32, contact_feasibility_guard=guard_receipt())
            for stage in result['stages']:
                stage.update(contact_feasibility_guard=guard_receipt(), contact_projection=summary)
            return result
        monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh', fake)
    else:
        def fake(*inputs, **kwargs):
            seen.append((None, None, kwargs))
            result = stub(status='stationary')(*inputs, **kwargs)
            result['reports'][0].update(contact_feasibility_guard=guard_receipt(), contact_projection=summary)
            result['reports'][0]['history'][0]['contact_projection'] = {'raw_cartesian_displacement': [[.01, 0, 0]]}
            return result
        monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', fake)
    extra = ['--topology-stage-steps', '1', '1'] if stages else []
    assert cli.main(args+['--contact-weight', '1', '--preserve-feasible-contacts', '--contact-projection-sweeps', '32', '--audit-dir', str(audit)]+extra) == 1
    assert seen[0][2]['contact_projection_sweeps'] == 32
    assert json.loads((out/'run.json').read_text())['contact_projection_sweeps'] == 32
    public = ''.join((out/name).read_text() for name in ('run.json', 'status.csv', 'geometry.csv', 'pdm.csv'))
    assert 'raw_cartesian_displacement' not in public
    assert (audit/'0.vasp.json').exists()
    assert_final_pdm(out, model)


def test_projection_failure_retains_every_input_and_last_coordinates(tmp_path, monkeypatch):
    source, model, out, profile, args = setup_case(tmp_path, ('0.vasp', '1.vasp'))
    def fake(*inputs, **kwargs):
        result = stub(status='contact_projection_failed')(*inputs, **kwargs)
        result['reports'][0].update(contact_feasibility_guard=guard_receipt(),
            error='Finite projection residual remains', contact_projection=dict(max_sweeps_per_direction=1,
                directions_attempted=1, actual_sweeps=1, direction_statuses=['projection_residual_infeasible']))
        return result
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', fake)
    assert cli.main(args+['--contact-weight', '1', '--preserve-feasible-contacts', '--contact-projection-sweeps', '1']) == 1
    assert_all_rows(out, ('0.vasp', '1.vasp'))
    for row in rows(out/'status.csv'):
        assert row['status'] == 'error' and row['coordinate_state'] == 'last_iterate'
        assert 'projection' in row['error'].lower()
    assert_final_pdm(out, model)
