"""Deterministic target refinement contracts; these are not model efficacy tests."""
import unittest
import torch
from cdvae.common.pdm_guidance import refine_pdm


SCHEMA = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2',
          'modes': ['pair'], 'elements': ['Al'], 'features': ['AlAl'], 'cutoff': 5.0}


class PDMGuidanceTests(unittest.TestCase):
    def inputs(self, distance=5.1):
        coords = torch.tensor([[.1, .1, .1], [.1 + distance / 12., .1, .1]], dtype=torch.float64)
        return coords, torch.tensor([13, 13]), torch.tensor([2]), torch.eye(3, dtype=torch.float64)[None] * 12.

    def test_same_start_can_follow_two_different_targets_without_randomness(self):
        args = self.inputs()
        originals = [x.clone() for x in args]
        rng = torch.random.get_rng_state().clone()
        near = refine_pdm(*args, torch.tensor([[1.]]), [1.], SCHEMA, {'steps': 80})
        far = refine_pdm(*args, torch.tensor([[0.]]), [1.], SCHEMA, {'steps': 80})
        self.assertEqual(float(near['final_hard_raw'][0, 0]), 1.)
        self.assertEqual(float(far['final_hard_raw'][0, 0]), 0.)
        self.assertFalse(torch.equal(near['frac_coords'], far['frac_coords']))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for original, now in zip(originals, args):
            self.assertTrue(torch.equal(original, now))
        for output in (near, far):
            self.assertTrue(torch.equal(output['cell_matrices'], args[3]))
            accepted = [r['accepted_loss'] for r in output['reports'][0]['history'] if r['accepted']]
            self.assertTrue(all(a >= b for a, b in zip(accepted, accepted[1:])))

    def test_short_distance_penalty_can_separate_overlapping_atoms(self):
        args = self.inputs(.8)
        out = refine_pdm(*args, torch.tensor([[1.]]), [1.], SCHEMA, {'steps': 80})
        delta = out['frac_coords'][1] - out['frac_coords'][0]
        delta = delta - delta.round()
        self.assertGreater(float(torch.linalg.vector_norm(delta @ args[3][0])), .99)

    def test_batch_does_not_couple_optimization_or_loss_scaling(self):
        args = self.inputs()
        single = refine_pdm(*args, torch.tensor([[1.]]), [1.], SCHEMA, {'steps': 5})
        batch = refine_pdm(args[0].repeat(2, 1), args[1].repeat(2), args[2].repeat(2),
                           args[3].repeat(2, 1, 1), torch.tensor([[1.], [0.]]), [1.], SCHEMA, {'steps': 5})
        self.assertTrue(torch.equal(single['frac_coords'], batch['frac_coords'][:2]))

    def test_works_inside_no_grad_and_rejects_invalid_scales(self):
        with torch.no_grad():
            result = refine_pdm(*self.inputs(), torch.tensor([[1.]]), [1.], SCHEMA, {'steps': 1})
        self.assertEqual(result['reports'][0]['iterations'], 1)
        with self.assertRaises(ValueError):
            refine_pdm(*self.inputs(), torch.tensor([[1.]]), [0.], SCHEMA)
        with self.assertRaises(ValueError):
            refine_pdm(*self.inputs(), torch.tensor([[1.]]), [1.], SCHEMA, {'random_restarts': 5})


class PDMGuidanceReviewTests(unittest.TestCase):
    def test_cartesian_gradient_transform_on_skew_cell(self):
        from cdvae.common.pdm_guidance import _objective, _config
        from cdvae.common.differentiable_pdm import PairPDMDescriptor
        cell=torch.tensor([[12.,0,0],[3.,10.,0],[1.,2.,14.]],dtype=torch.float64)
        inverse=torch.linalg.inv(cell)
        cart=torch.tensor([[1.,1.,1.],[5.8,1.5,1.2]],dtype=torch.float64,requires_grad=True)
        types=torch.tensor([13,13]);target=torch.tensor([0.],dtype=torch.float64);std=torch.ones(1,dtype=torch.float64)
        descriptor=PairPDMDescriptor(SCHEMA);config=_config({'steps':2})
        fn=lambda xyz:_objective(xyz@inverse,types,cell,target,std,descriptor,config)[0]
        self.assertTrue(torch.autograd.gradcheck(fn,(cart,),eps=1e-6,atol=1e-5))
        frac=(cart@inverse).detach().requires_grad_(True)
        gf,=torch.autograd.grad(_objective(frac,types,cell,target,std,descriptor,config)[0],frac)
        gc,=torch.autograd.grad(fn(cart),cart)
        torch.testing.assert_close(gf@inverse.T,gc,atol=1e-10,rtol=1e-10)

    def test_rejects_known_impossible_targets_and_missing_species_counts(self):
        args=PDMGuidanceTests().inputs()
        for target in (-1.,2.):
            with self.assertRaises(ValueError):
                refine_pdm(*args,torch.tensor([[target]]),[1.],SCHEMA,{'steps':1})
        schema=dict(SCHEMA,elements=['Al','B'],features=['AlAl','AlB','BB'])
        with self.assertRaisesRegex(ValueError,'composition'):
            refine_pdm(*args,torch.tensor([[1.,.1,0.]]),[1.,1.,1.],schema,{'steps':1})
        # Continuous mean targets are allowed; they are not asserted realizable.
        refine_pdm(*args,torch.tensor([[.5]]),[1.],SCHEMA,{'steps':1})

    def test_fixed_cell_obvious_count_infeasibility_is_rejected(self):
        coords,types,count,cell=PDMGuidanceTests().inputs()
        with self.assertRaisesRegex(ValueError,'fixed cell'):
            refine_pdm(coords,types,count,cell/4,torch.tensor([[0.]]),[1.],SCHEMA,{'steps':1})

    def test_exact_collision_retained_and_explicitly_reported(self):
        args=PDMGuidanceTests().inputs(0.)
        before=args[0].clone()
        out=refine_pdm(*args,torch.tensor([[1.]]),[1.],SCHEMA,{'steps':4})
        self.assertTrue(torch.equal(out['frac_coords'],before))
        self.assertEqual(out['reports'][0]['status'],'stationary_with_collisions')
        self.assertGreater(out['reports'][0]['final_repulsion_loss'],0.)

    def test_nonfinite_objective_retains_every_graph_and_json_safe_failure_trace(self):
        import json
        args=PDMGuidanceTests().inputs()
        out=refine_pdm(args[0].repeat(2,1),args[1].repeat(2),args[2].repeat(2),args[3].repeat(2,1,1),
                       torch.tensor([[1.],[0.]]),[1e-300],SCHEMA,{'steps':2})
        self.assertEqual(len(out['reports']),2)
        self.assertEqual(out['frac_coords'].shape,(4,3))
        self.assertTrue(torch.equal(out['frac_coords'],args[0].repeat(2,1)))
        self.assertTrue(all(r['status']=='nonfinite_objective' for r in out['reports']))
        json.dumps(out['reports'],allow_nan=False)


    def test_repulsion_gradient_points_apart_and_cartesian_step_is_bounded(self):
        from cdvae.common.pdm_guidance import _objective, _config
        from cdvae.common.differentiable_pdm import PairPDMDescriptor, nearest_image_displacements
        coords,types,count,cell=PDMGuidanceTests().inputs(.8)
        coords.requires_grad_(True)
        objective,_,repulsion=_objective(coords,types,cell[0],torch.ones(1),torch.ones(1),PairPDMDescriptor(SCHEMA),_config(None))
        gradient,=torch.autograd.grad(repulsion,coords)
        cart_gradient=gradient@torch.linalg.inv(cell[0]).T
        # d=.8, r_min=1, N=2: +/-2*(1-.8)/N = +/- .2.
        self.assertAlmostEqual(float(cart_gradient[0,0]),.2,places=10)
        self.assertAlmostEqual(float(cart_gradient[1,0]),-.2,places=10)
        out=refine_pdm(coords.detach(),types,count,cell,torch.ones(1,1),[1.],SCHEMA,{'steps':1,'max_step_A':.04})
        moved=nearest_image_displacements(out['frac_coords']-coords.detach(),cell[0])
        self.assertLessEqual(float(torch.linalg.vector_norm(moved,dim=-1).max()),.04+1e-12)


if __name__ == '__main__':
    unittest.main()
