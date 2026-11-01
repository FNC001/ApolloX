"""Finite cancellation regressions; numerical execution is remote-only."""
import json
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

from cdvae.common.feasible_displacement import _contract_toward_origin
from test_contact_projection_interior import contract


SAVED_FAILURES = [{'name': 'input0-contacts_fixed_topology.pt', 'raw': [[6.273390893426871e-05, -0.018857365777862796, -0.008772431702241599], [0.0008819462573412583, -0.01982422982256672, -0.011835538139248642]], 'normal': [-0.7588936385663957, -0.6511120790489808, 0.011554473591234767], 'rhs': -2.7550763969763636e-05, 'previous_scale': 0.9999999999999453}, {'name': 'input1-contacts_fixed_topology.pt', 'raw': [[-0.010175195880517022, -0.01611481929621204, 0.007866306143666768], [-0.0031317973305631753, -0.016376174560842633, -0.0052710533759662]], 'normal': [-0.18984299223057807, -0.9787099365833319, -0.07801601331772334], 'rhs': -5.6424446480018986e-05, 'previous_scale': 0.9999999999999666}, {'name': 'input2-contacts_fixed_topology.pt', 'raw': [[0.011113495227108056, -0.0032728484079264275, -0.002459561509151369], [0.009534887273434855, 0.0023545933470036115, -0.007200524355549057]], 'normal': [-0.8991012530580745, 0.10385860337102078, 0.42524149286637264], 'rhs': -1.2267487972339453e-05, 'previous_scale': 0.999999999999703}, {'name': 'input3-contacts_fixed_topology.pt', 'raw': [[0.0015746814823075182, -0.022961922738781164, 0.003151810229982647], [0.0024158370389655015, -0.022736346233944917, -0.00278735682332262]], 'normal': [0.8414932700167344, 0.5218119048122254, 0.14000504459761595], 'rhs': -5.978102536330354e-06, 'previous_scale': 0.9999999999990609}]


@pytest.mark.parametrize('case', SAVED_FAILURES, ids=lambda x:x['name'])
def test_saved_cancelled_faces_fail_old_relative_margin_but_pass_absolute_budget(case):
    raw=np.asarray(case['raw'],dtype=np.float64);normal=np.asarray(case['normal'],dtype=np.float64)
    old=raw*case['previous_scale']
    assert float(np.sum(normal*(old[1]-old[0]))) < case['rhs']
    returned,record=contract(case['raw'],[case['normal']],[case['rhs']],.04)
    actual=returned.double().numpy()
    assert float(np.sum(normal*(actual[1]-actual[0]))) >= case['rhs']
    assert np.max(np.linalg.norm(actual,axis=-1)) <= .04
    assert record['roundoff_epsilon_multiplier']==64
    assert record['geometry_thresholds_relaxed'] is False
    assert record['extra_solver_sweeps']==0
    assert 0 < record['alpha_upper_bound'] <= 1
    assert record['halfspace_conservative_dot_lower_bounds_A'][0] < record['halfspace_dot_displacements_A'][0]
    assert not record['zero_direction']
    # A geometry-only adjustment does not fabricate an objective decrease.
    assert record['finite_roundoff_guarantee_claimed'] is False
    json.dumps(record,allow_nan=False)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_absolute_budget_includes_cancelled_terms_and_actual_output_precision(dtype):
    raw=[[.03125,.03125,0.],[.031250000001,.031249999998,0.]]
    normal=[.6,.8,0.]
    returned,record=contract(raw,[normal],[-1e-10],.04,dtype)
    with localcontext() as ctx:
        ctx.prec=80
        exactS=sum(abs(Decimal.from_float(n))*(abs(Decimal.from_float(a))+abs(Decimal.from_float(b)))
                   for n,a,b in zip(normal,raw[0],raw[1]))
        exactdot=sum(Decimal.from_float(n)*(Decimal.from_float(b)-Decimal.from_float(a))
                   for n,a,b in zip(normal,raw[0],raw[1]))
    savedS=record['halfspace_absolute_product_sums_A'][0]
    delta=record['halfspace_roundoff_allowances_A'][0]
    assert abs(savedS-float(exactS)) <= 2*np.finfo(np.float64).eps*float(exactS)
    expected=64*torch.finfo(dtype).eps*savedS
    assert delta==np.nextafter(expected,np.inf)
    assert record['halfspace_conservative_dot_lower_bounds_A'][0] <= float(exactdot)
    assert delta > abs(record['halfspace_dot_displacements_A'][0])*64*torch.finfo(dtype).eps
    actual=returned.double().numpy()
    assert float(np.sum(np.asarray(normal)*(actual[1]-actual[0]))) >= -1e-10
    assert np.max(np.linalg.norm(actual,axis=-1))<=.04
    assert returned.dtype==dtype


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_zero_slack_common_translation_is_honestly_conservative_stationary(dtype):
    returned,record=contract([[.01,.02,0.],[.01,.02,0.]],[[.6,.8,0.]],[0.],.04,dtype)
    assert record['halfspace_dot_displacements_A']==[0.]
    assert record['halfspace_absolute_product_sums_A'][0]>0
    assert record['halfspace_conservative_dot_lower_bounds_A'][0]<0
    assert record['alpha_upper_bound']==0
    assert torch.count_nonzero(returned)==0
    assert record['zero_direction'] and record['stationary_reason']=='zero_contracted_direction'
    assert record['nearest_projection_optimality_claimed'] is False


def test_positive_halfspace_bound_is_explicitly_rounded_inward():
    _,record=contract([[.02,0.,0.],[-.02,0.,0.]],[[1.,0.,0.]],[-.01],.04)
    safe=record['halfspace_conservative_dot_lower_bounds_A'][0]
    expected=np.nextafter(-.01/safe,0.)
    assert record['halfspace_alpha_upper_bounds_capped_at_one']==[expected]
    assert record['positive_halfspace_bounds_rounded_toward_zero']
    assert record['positive_roundoff_allowances_rounded_up'] and record['conservative_dots_rounded_down']


def test_absolute_sum_overflow_is_not_silently_converted_to_zero_step():
    raw=torch.tensor([[1.,1.,1.],[1.,1.,1.]],dtype=torch.float64)
    with pytest.raises(ValueError,match='absolute product sum'):
        _contract_toward_origin(raw,torch.tensor([[0,1]]),torch.full((1,3),1e308,dtype=torch.float64),
            torch.tensor([-1.],dtype=torch.float64),2.,torch.float64)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_subnormal_face_retains_strict_bound_or_exact_stationary(dtype):
    rhs=-float(np.nextafter(np.float64(0.),np.float64(1.)))
    returned,record=contract([[.01,0.,0.],[-.01,0.,0.]],[[1.,0.,0.]],[rhs],.04,dtype)
    actual=returned.double().numpy()
    assert float(actual[1,0]-actual[0,0]) >= rhs
    assert not record['finite_roundoff_guarantee_claimed']
    if np.all(actual==0):assert record['zero_direction']
    json.dumps(record,allow_nan=False)
