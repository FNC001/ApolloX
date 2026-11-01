"""All-pair lower bounds join the fixed-cell PDM and local-angle objective."""
import copy
import json

import pytest
import torch

from cdvae.common.reference_geometry_profile import POLICY
from test_pdm_geometry_guidance import example, run_case


def reference():
    return {'version': 1, 'source': 'md', 'fit_split': 'train', 'policy': copy.deepcopy(POLICY),
        'bond_groups': {'C|C': {'covered': True, 'bounds': [2., 2.5], 'reasons': [],
            'structure_count': 48, 'observation_count': 300}}, 'angle_groups': {}}


def contacts(**kwargs):
    return dict(contact_profile=reference(), contact_source='md', contact_weight=5., **kwargs)


def empty(args):
    args = list(args)
    args[4] = dict(bond_pairs=[], bond_bounds_A=[], angle_triples=[], angle_bounds_deg=[])
    return args


def test_contact_only_refines_new_pair_with_no_original_local_bond():
    args = empty(example(angle=90., first=1.5, second=1.5))
    result = run_case(args, steps=30, **contacts())
    report = result['reports'][0]
    assert report['iterations'] > 0 and not report['has_local_constraints']
    assert report['initial_contact_audit']['covered_pair_count'] == 3
    before = report['initial_contact_audit']['pairs']
    after = report['final_contact_audit']['pairs']
    assert after[0]['distance_A'] > before[0]['distance_A']
    assert report['final_components']['contact_lower_bound_loss'] < report['initial_components']['contact_lower_bound_loss']
    assert result['final_hard_raw'].tolist() == [[3.]]
    assert not report['profile_completeness_checked']


def test_contact_objective_contains_pdm_bond_angle_repulsion_and_bounds():
    result = run_case(example(), steps=8, **contacts())
    report = result['reports'][0]
    for row in [report['initial_components'], report['final_components'], *report['history']]:
        components = [row, *row.get('proposals', [])]
        for item in components:
            if 'loss' not in item:
                continue
            assert item['loss'] == pytest.approx(item['soft_pdm_loss'] + item['weighted_repulsion_loss']
                + item['weighted_bond_loss'] + item['weighted_angle_loss'] + item['weighted_contact_lower_bound_loss'])
            assert item['weighted_contact_lower_bound_loss'] == pytest.approx(5. * item['contact_lower_bound_loss'])
            if item.get('accepted') and 'backtrack' in item:
                assert item['loss'] <= row['loss']
        assert len(row.get('proposals', [])) <= result['config']['line_search_steps']
        assert 'pairs' not in row
    assert report['iterations'] <= 8
    assert report['initial_contact_audit']['pairs'] and report['final_contact_audit']['pairs']
    json.dumps(result['reports'], allow_nan=False)


def test_targets_still_change_generated_geometry_with_contacts_enabled():
    args = empty(example(angle=90., first=2.9, second=2.9))
    low = run_case(args, target=0., steps=50, **contacts())
    high = run_case(args, target=3., steps=50, **contacts())
    assert float(low['final_hard_raw'][0,0]) < float(high['final_hard_raw'][0,0])
    assert not torch.equal(low['frac_coords'], high['frac_coords'])


def test_contact_pairs_are_all_graph_local_and_local_constraints_rebase():
    a = example(); b = example(angle=120., first=1.8, second=2.3)
    constraints = {'bond_pairs': [[0,1],[1,2],[3,4],[4,5]],
        'bond_bounds_A': a[4]['bond_bounds_A'] + b[4]['bond_bounds_A'],
        'angle_triples': [[0,1,2],[3,4,5]],
        'angle_bounds_deg': a[4]['angle_bounds_deg'] + b[4]['angle_bounds_deg']}
    batch = (torch.cat([a[0],b[0]]), torch.cat([a[1],b[1]]), torch.tensor([3,3]), torch.cat([a[3],b[3]]), constraints)
    combined = run_case(batch, steps=3, **contacts())
    first = run_case(a, steps=3, **contacts()); second = run_case(b, steps=3, **contacts())
    assert torch.equal(combined['frac_coords'], torch.cat([first['frac_coords'],second['frac_coords']]))
    for index, row in enumerate(combined['reports']):
        assert row['global_atom_offset'] == 3*index
        assert row['local_constraints']['bond_pairs'] == [[0,1],[1,2]]
        assert {tuple(p['pair']) for p in row['final_contact_audit']['pairs']} == {(0,1),(0,2),(1,2)}


def test_undefined_new_contact_retains_graph_and_continues_other_graph():
    a = empty(example()); a[0][0] = a[0][1]
    b = empty(example(angle=90.,first=1.5,second=1.5))
    batch = (torch.cat([a[0],b[0]]),torch.cat([a[1],b[1]]),torch.tensor([3,3]),torch.cat([a[3],b[3]]),a[4])
    result = run_case(batch, steps=3, **contacts())
    failed, complete = result['reports']
    assert failed['status'] == 'undefined_pair_contacts'
    assert failed['initial_contact_audit']['undefined_pair_indices'] == [0]
    assert failed['final_contact_audit']['planned_distinct_site_pair_count'] == 3
    assert torch.equal(result['frac_coords'][:3],a[0])
    assert complete['iterations'] == 3 and not torch.equal(result['frac_coords'][3:],b[0])
    json.dumps(result['reports'],allow_nan=False)


def test_missing_contact_group_without_local_constraints_is_not_success():
    args = empty(example())
    profile = reference(); profile['bond_groups'] = {}
    result = run_case(args, steps=3, contact_profile=profile, contact_source='md', contact_weight=1.)
    report = result['reports'][0]
    assert report['status'] == 'missing_local_constraints' and report['iterations'] == 0
    assert report['final_contact_audit']['lower_bound_constraints_satisfied'] is None
    assert report['final_contact_audit']['uncovered_pair_count'] == 3
    assert torch.equal(result['frac_coords'],args[0])


@pytest.mark.parametrize('kwargs', [
    {'contact_profile': reference()}, {'contact_source': 'md'}, {'contact_weight': 1.},
    {'contact_profile': reference(), 'contact_source': 'md', 'contact_weight': True},
    {'contact_profile': reference(), 'contact_source': 'md', 'contact_weight': 0.},
    {'contact_profile': reference(), 'contact_source': 'md', 'contact_weight': float('nan')},
    {'contact_profile': reference(), 'contact_source': 'other', 'contact_weight': 1.},
])
def test_contact_configuration_is_explicit_and_source_matched(kwargs):
    with pytest.raises(ValueError):
        run_case(example(),steps=1,**kwargs)


def test_default_disabled_path_keeps_original_outputs_and_fields():
    plain = run_case(example(),steps=3)
    disabled = run_case(example(),steps=3, contact_profile=None, contact_source=None, contact_weight=None)
    assert plain.keys() == disabled.keys()
    assert torch.equal(plain['frac_coords'],disabled['frac_coords'])
    assert plain['reports'] == disabled['reports']
    assert 'contact_weight' not in plain
    assert 'initial_contact_audit' not in plain['reports'][0]
    assert 'contact_lower_bound_loss' not in plain['reports'][0]['initial_components']
