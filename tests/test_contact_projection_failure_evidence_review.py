"""CLI evidence for a failed projection after a nontrivial accepted iterate.

This is a serialization/delivery fixture, not a numerical projection test.
Run it only with the separately approved remote test suite.
"""
import json

import numpy as np
from pymatgen.io.vasp import Poscar

from scripts.pdm_control import refine_geometry_structures as cli
from test_contact_guard_cli import guard_receipt
from test_pdm_geometry_contacts_cli import assert_final_pdm
from test_pdm_geometry_refine_cli import assert_all_rows, rows, setup_case, stub


def test_failed_projection_preserves_last_iterate_and_raw_audit_for_every_input(tmp_path, monkeypatch):
    names = ('0.vasp', '1.vasp')
    source, model, out, profile, args = setup_case(tmp_path, names)
    audit_dir = tmp_path/'projection-audit'
    original_bytes = {name: (source/name).read_bytes() for name in names}
    captured = []
    failure_message = 'Finite projection residual remains after one sweep'

    def failed_projection(*inputs, **kwargs):
        frac, _, _, cells = inputs[:4]
        assert kwargs['preserve_feasible_contacts'] is True
        assert kwargs['contact_projection_sweeps'] == 1
        # Supply a last-accepted-coordinate sentinel, distinct from both the
        # input and the failing direction. The outward atom shift also changes
        # hard PDM counts so reusing input metrics cannot pass this CLI test.
        # This stub does not assert that a numerical optimizer found this state.
        accepted = frac.clone()+frac.new_tensor([.0078125, -.00390625, .001953125])
        accepted[1, 0] += .125
        requested = frac.new_zeros(frac.shape)
        requested[0, 0], requested[1, 0] = .03, -.03
        projected = requested/2
        inverse = cells[0].inverse()
        raw_candidate = (accepted+requested @ inverse).remainder(1.)
        projected_candidate = (accepted+projected @ inverse).remainder(1.)
        residual = {
            'finite': True, 'feasible': False,
            'maximum_halfspace_violation_A': .0125,
            'maximum_ball_violation_A': 0.,
            'violated_halfspace_indices': [0], 'violated_atom_balls': [],
        }
        sweep = {
            'sweep': 0, **residual,
            'maximum_primal_dual_change_A': .015,
            'projected_cartesian_displacement': projected,
            'projected_candidate_frac_coords': projected_candidate,
            'halfspace_corrections': [-.015],
            'ball_corrections': requested-projected,
        }
        projection = {
            'status': 'projection_residual_infeasible', 'usable': False,
            'error': failure_message, 'max_sweeps': 1, 'actual_sweeps': 1,
            'input_frac_coords': accepted, 'cell': cells[0],
            'raw_cartesian_displacement': requested,
            'raw_candidate_frac_coords': raw_candidate,
            'projected_cartesian_displacement': projected,
            'projected_candidate_frac_coords': projected_candidate,
            'returned_dtype_residual': residual, 'sweeps': [sweep],
        }
        result = stub(status='contact_projection_failed', coords=accepted)(*inputs, **kwargs)
        result['reports'][0].update(
            error=failure_message,
            contact_feasibility_guard=guard_receipt(),
            contact_projection={
                'max_sweeps_per_direction': 1, 'directions_attempted': 1,
                'actual_sweeps': 1, 'direction_statuses': ['projection_residual_infeasible'],
                'nearest_projection_optimality_claimed': False,
            },
        )
        result['reports'][0]['history'][0].update(accepted=False, proposals=[], contact_projection=projection)
        captured.append({
            'accepted': accepted.detach().cpu().numpy().copy(),
            'raw': requested.detach().cpu().numpy().copy(),
            'projected': projected.detach().cpu().numpy().copy(),
            'raw_candidate': raw_candidate.detach().cpu().numpy().copy(),
            'projected_candidate': projected_candidate.detach().cpu().numpy().copy(),
            'ball_corrections': (requested-projected).detach().cpu().numpy().copy(),
        })
        return result

    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry', failed_projection)
    assert cli.main(args+[
        '--contact-weight', '1', '--preserve-feasible-contacts',
        '--contact-projection-sweeps', '1', '--audit-dir', str(audit_dir),
    ]) == 1
    assert len(captured) == len(names)
    assert_all_rows(out, names)
    assert {path.name for path in audit_dir.glob('*.json')} == {name+'.json' for name in names}
    status_by_name = {row['filename']: row for row in rows(out/'status.csv')}
    pdm_by_name = {row['label']: row for row in rows(out/'pdm.csv')}
    schema = json.loads((model/'pdm_schema.json').read_text())

    for name, expected in zip(names, captured):
        original = Poscar.from_file(source/name, check_for_potcar=False).structure
        delivered = Poscar.from_file(out/'gen'/name, check_for_potcar=False).structure
        np.testing.assert_allclose(delivered.frac_coords, expected['accepted'], rtol=0, atol=1e-14)
        assert not np.allclose(delivered.frac_coords, original.frac_coords, rtol=0, atol=1e-8)
        assert not np.allclose(delivered.frac_coords, expected['projected_candidate'], rtol=0, atol=1e-8)
        assert (source/name).read_bytes() == original_bytes[name]

        state = status_by_name[name]
        assert state['status'] == 'error' and state['coordinate_state'] == 'last_iterate'
        assert state['optimizer_status'] == 'contact_projection_failed'
        assert state['audit_status'] == 'written' and failure_message in state['error']
        assert state['output_sha256'] == cli.sha(out/'gen'/name)

        evidence = json.loads((audit_dir/(name+'.json')).read_text())
        assert evidence['status']['status'] == 'error'
        assert evidence['status']['coordinate_state'] == 'last_iterate'
        np.testing.assert_allclose(evidence['delivered_frac_coords'], expected['accepted'], rtol=0, atol=1e-14)
        projection = evidence['refinement']['reports'][0]['history'][0]['contact_projection']
        assert projection['status'] == 'projection_residual_infeasible' and projection['usable'] is False
        assert projection['actual_sweeps'] == 1
        np.testing.assert_array_equal(projection['input_frac_coords'], expected['accepted'])
        np.testing.assert_array_equal(projection['raw_cartesian_displacement'], expected['raw'])
        np.testing.assert_array_equal(projection['projected_cartesian_displacement'], expected['projected'])
        np.testing.assert_array_equal(projection['raw_candidate_frac_coords'], expected['raw_candidate'])
        np.testing.assert_array_equal(projection['projected_candidate_frac_coords'], expected['projected_candidate'])
        residual = projection['returned_dtype_residual']
        assert residual['finite'] is True and residual['feasible'] is False
        assert residual['maximum_halfspace_violation_A'] > 0
        assert residual['violated_halfspace_indices'] == [0]
        assert len(projection['sweeps']) == 1
        sweep = projection['sweeps'][0]
        assert sweep['halfspace_corrections'] == [-.015]
        np.testing.assert_array_equal(sweep['ball_corrections'], expected['ball_corrections'])
        np.testing.assert_array_equal(sweep['projected_candidate_frac_coords'], expected['projected_candidate'])
        assert sweep['maximum_primal_dual_change_A'] > 0 and sweep['feasible'] is False

        # The shared stub deliberately returns fake final_hard_raw values.
        # Neither that field nor the rejected projection candidate is delivered.
        assert pdm_by_name[name]['pdm_status'] == 'computed'
        assert all(pdm_by_name[name][feature] != '999' for feature in schema['features'])
        assert cli.hard_pdm(delivered, schema) != cli.hard_pdm(original, schema)
    assert_final_pdm(out, model)
    run = json.loads((out/'run.json').read_text())
    assert run['status'] == 'COMPLETE_WITH_ERRORS'
    assert run['errors'] == run['retained_files'] == len(names)
    public = ''.join((out/name).read_text() for name in ('run.json', 'status.csv', 'geometry.csv', 'pdm.csv'))
    assert all(key not in public for key in (
        'raw_cartesian_displacement', 'projected_candidate_frac_coords',
        'halfspace_corrections', 'ball_corrections',
    ))
