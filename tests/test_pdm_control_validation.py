"""Portable CPU fixtures verify validation arithmetic, never model quality."""
import json,random,shutil,subprocess,sys
from pathlib import Path
from itertools import combinations_with_replacement
import numpy as np,pandas as pd,pytest,torch
from pymatgen.core import Structure,Lattice
from pymatgen.io.vasp import Poscar
SCRIPT_ROOT=Path(__file__).resolve().parents[1]/'scripts/pdm_control'
sys.path.insert(0,str(SCRIPT_ROOT))
from validation_common import *
from generate_validation import native_arms,validated_lattice,condition_arms
from evaluate_validation import cluster_interval,describe

def seal(root):
 files=[{'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='GENERATION_COMPLETE.json']
 frame=pd.read_csv(root/'generation_manifest.csv');write_json(root/'GENERATION_COMPLETE.json',{'status':'ALL_PLANNED_ATTEMPTS_COMMITTED_NOT_ACCURACY_PASS','planned':len(frame),'exports':int((frame.status=='ok').sum()),'errors':int((frame.status=='error').sum()),'fixture':True,'files':files})

def create(target_root,protocol_path,out):
 target_root=Path(target_root);protocol_path=Path(protocol_path);out=Path(out);require(not out.exists(),'Newfixturepathrequired');inputs=target_root/'stages/pilot12/generator_inputs';protocol=json.loads(protocol_path.read_text());settings=stage_settings(protocol,'pilot12')
 pp=pd.read_csv(inputs/'pair_plan.csv',dtype=str);sp=pd.read_csv(inputs/'singleton_plan.csv',dtype=str);raw=pd.read_csv(inputs/'target_raw.csv').set_index('label');schema=json.loads((inputs/'pdm_schema.json').read_text());names=schema['features']
 plan,layout=planned_attempts(pp,sp,settings,'UNIT_FIXTURE_NOT_TRAINED_MODEL','pilot12');out.mkdir(parents=True)
 for d in ['inputs','model_contract','structures','tensors','rng']:(out/d).mkdir()
 for p in inputs.iterdir():shutil.copy2(p,out/'inputs'/p.name)
 for name in ['pdm_schema.json','scaler_stats.txt']:shutil.copy2(inputs/name,out/'model_contract'/name)
 (out/'model_contract/hparams.yaml').write_text('unit_fixture: true\n')
 torch.save({'unit_fixture':True},out/'model_contract/prop_scalers.pt');torch.save({'unit_fixture':True},out/'model_contract/lattice_scaler.pt')
 shutil.copy2(protocol_path,out/'protocol.json');write_json(out/'guidance_config.json',{'steps':200,'temperature':.1,'max_step_A':.04,'min_distance_A':1.0,'geometry_weight':10,'line_search_steps':6})
 write_csv(plan,out/'expected_attempts.csv');write_json(out/'batch_layout.json',layout)
 lines=(inputs/'scaler_stats.txt').read_text().splitlines();mean=np.fromstring(next(x for x in lines if x.startswith('Train Mean:')).split(':')[1],sep=',');std=np.fromstring(next(x for x in lines if x.startswith('Train Std:')).split(':')[1],sep=',')
 rows=plan.to_dict('records');byid={r['generated_id']:r for r in rows};batch_records=[]
 for seed_index,seed in enumerate(settings['sampling_seeds']):
  for batch in layout:
   arms=['left','right']+(['repeat_left'] if batch['repeat'] else []) if batch['batch_kind']=='pair' else ['single']
   state=capture_rng(torch,np);torch.save({'start':state,'end':{arm:state for arm in arms}},out/f'rng/s{seed_index}_{batch["batch_id"]}.pt')
   batch_records.append({'seed_index':seed_index,'seed':seed,'batch_id':batch['batch_id'],'rng_equal':True,'all_native_success':True})
 for row in rows:
  tid=row['target_id'];gid=row['generated_id'];s=Structure.from_file(target_root/'evaluator_only/witness_structures'/f'{tid}.vasp');target=raw.loc[tid,names].to_numpy(float);scaled=(target-mean)/std
  sample={'num_atoms':torch.tensor([len(s)]),'atom_types':torch.tensor([x.specie.Z for x in s]),'frac_coords':torch.tensor(s.frac_coords,dtype=torch.float64),'lengths':torch.tensor([s.lattice.abc],dtype=torch.float64),'angles':torch.tensor([s.lattice.angles],dtype=torch.float64),'is_traj':False}
  embedding=torch.nn.functional.pad(torch.tensor(scaled,dtype=torch.float32).reshape(1,21),(0,29));z=torch.zeros(1,256)
  payload={'target_id':tid,'pair_id':row['pair_id'],'guidance':row['guidance'],'batch_id':row['batch_id'],'checkpoint_sha256':'0'*64,'z':z,'composition_embedding':torch.zeros(1,16),'cond_z':torch.cat([z,embedding],dim=-1),'embedded_pdm':embedding,'model_standardized_target':torch.tensor(scaled,dtype=torch.float32).reshape(1,21),'external_standardized_target':scaled.tolist(),'raw_target':target.tolist(),'initial_frac_coords':torch.zeros(len(s),3),'samples':sample,'unit_fixture':True}
  row.update(status='ok',filename=f'structures/{gid}.vasp',tensor_file=f'tensors/{gid}.pt');shutil.copy2(target_root/'evaluator_only/witness_structures'/f'{tid}.vasp',out/row['filename'])
  if row['guidance']=='ON':
   off=byid[row['off_generated_id']];payload.update(off_tensor_sha256=sha(out/off['tensor_file']),off_tensor_file=off['tensor_file'],guidance_report={'final_hard_raw':torch.tensor(target).reshape(1,21)})
  torch.save(payload,out/row['tensor_file'])
 write_csv(pd.DataFrame(rows),out/'generation_manifest.csv');write_json(out/'batch_records.json',batch_records)
 contract={p.name:sha(p) for p in (out/'model_contract').iterdir()}
 write_json(out/'provenance.json',{'candidate_id':'UNIT_FIXTURE_NOT_TRAINED_MODEL','stage':'pilot12','checkpoint_sha256':'0'*64,'checkpoint_path':'UNIT_FIXTURE_NO_MODEL','prior_only':True,'reference_geometry_used':False,'cpu_fixture_only':True,
  'model_condition_scaler':{'means':[0.]*21,'stds':[1.]*21},'model_contract_sha256':contract,'input_sha256':{p.name:sha(p) for p in (out/'inputs').iterdir()},
  'protocol_sha256':sha(out/'protocol.json'),'guidance_config_sha256':sha(out/'guidance_config.json'),'expected_attempts_sha256':sha(out/'expected_attempts.csv'),'batch_layout_sha256':sha(out/'batch_layout.json'),
  'fixture_truth':'Allcoordinates arecopiedfromreference witnesses tovalidate arithmetic. Not sampled bymodel, not empirical evidence. Evaluator mustrefuse scientificacceptance.'})
 seal(out);return out

@pytest.fixture(scope='module')
def fixture_root(tmp_path_factory):
 root=tmp_path_factory.mktemp('validation-unit-fixture')
 targets=root/'target-source';inputs=targets/'stages/pilot12/generator_inputs';inputs.mkdir(parents=True)
 witnesses=targets/'evaluator_only/witness_structures';witnesses.mkdir(parents=True)
 elements=['B','Co','Fe','Mo','Ni','O'];names=sorted(''.join(x) for x in combinations_with_replacement(elements,2))
 schema={'schema_version':2,'features':names,'elements':elements,'modes':['pair'],'cutoff':5.,'geometry':'periodic-minimum-image-clique-v2'}
 write_json(inputs/'pdm_schema.json',schema)
 (inputs/'scaler_stats.txt').write_text('Train Mean:'+','.join(['0']*21)+'\nTrain Std:'+','.join(['1']*21)+'\n')
 write_json(inputs/'GENERATOR_INPUT_CONTRACT.json',{'unit_fixture':True})
 formula='BCoFeMoNiO';raw=[]
 for tid,coords in [('A',[[.1,.1,.1],[.2,.1,.1],[.3,.1,.1],[.1,.2,.1],[.2,.2,.1],[.3,.2,.1]]),('B',[[.1,.1,.1],[.4,.1,.1],[.7,.1,.1],[.1,.5,.1],[.4,.5,.1],[.7,.5,.1]]),('C',[[.1,.1,.1],[.2,.1,.1],[.3,.1,.1],[.1,.2,.1],[.2,.2,.1],[.3,.2,.1]])]:
  Poscar(Structure(Lattice.cubic(20),elements,coords)).write_file(witnesses/f'{tid}.vasp')
  vector,_,_=describe(witnesses/f'{tid}.vasp',names,5,formula);raw.append({'label':tid,'formula':formula,**dict(zip(names,vector))})
 pd.DataFrame(raw).to_csv(inputs/'target_raw.csv',index=False)
 pd.DataFrame([{'pair_id':'P','left_target_id':'A','right_target_id':'B','formula':formula,'source_domain':'md_labeled_endpoint'}]).to_csv(inputs/'pair_plan.csv',index=False)
 pd.DataFrame([{'target_id':'C','formula':formula,'source_domain':'legacy_variable_composition'}]).to_csv(inputs/'singleton_plan.csv',index=False)
 protocol={'stages':{'pilot12':{'MD_real_pairs':1,'legacy_real_singletons':1,'sampler':{'n_step_each':10,'sigma_levels':50,'step_lr':.0001,'min_sigma':0.},'sampling_seeds':[20261101],'pair_batch_size':1,'singleton_batch_size':1}}}
 write_json(root/'protocol.json',protocol);create(targets,root/'protocol.json',root/'generation')
 outcome=evaluate(root/'generation',root/'evaluation');assert outcome.returncode==0,outcome.stderr
 return root

def evaluate(generation,out):
 return subprocess.run([sys.executable,str(SCRIPT_ROOT/'evaluate_validation.py'),'--generation',str(generation),'--out',str(out),'--allow-fixture'],capture_output=True,text=True)

def test_actual_real_structure_fixture_has_exact_targets_and_no_scientific_acceptance(fixture_root):
 d=json.loads((fixture_root/'evaluation/validation_summary.json').read_text())
 assert d['classification']=='UNIT_FIXTURE_ONLY' and d['counts']=={'all_attempts':8,'primary_attempts':6,'repeat_attempts':2,'exports':8,'failed_exports':0}
 for mode in ['OFF','ON']:
  assert not d['modes'][mode]['decision']['eligible']
  assert d['modes'][mode]['MD_real_paired_control']['mean_advantage_over_gap_valid']==pytest.approx(1.)
  for source in ['md_labeled_endpoint','legacy_variable_composition']:assert d['modes'][mode]['accuracy_by_source'][source]['valid_mean_mre']==0

def test_missing_planned_repeat_is_rejected_even_with_updated_artifact_seal(tmp_path,fixture_root):
 r=tmp_path/'g';shutil.copytree(fixture_root/'generation',r)
 p=r/'generation_manifest.csv';d=pd.read_csv(p,keep_default_na=False);d=d.drop(d[d.arm=='repeat_left'].index[0]);d.to_csv(p,index=False);seal(r)
 result=evaluate(r,tmp_path/'e');assert result.returncode!=0 and 'denominator' in result.stderr

def test_changed_requested_PDM_tensor_is_rejected(tmp_path,fixture_root):
 r=tmp_path/'g';shutil.copytree(fixture_root/'generation',r);d=pd.read_csv(r/'generation_manifest.csv');p=r/d.tensor_file.iloc[0];v=torch.load(p,weights_only=False);v['raw_target'][0]+=1;torch.save(v,p);seal(r)
 result=evaluate(r,tmp_path/'e');assert result.returncode!=0 and 'Raw input target differs' in result.stderr

def test_failed_ON_attempt_remains_in_joint_denominator(tmp_path,fixture_root):
 r=tmp_path/'g';shutil.copytree(fixture_root/'generation',r);p=r/'generation_manifest.csv';d=pd.read_csv(p,keep_default_na=False);idx=d[(d.guidance=='ON')&(d.arm=='left')].index[0];d.loc[idx,['status','filename','error']]=['error','','Fixture simulated refinementfailure'];d.to_csv(p,index=False);seal(r)
 result=evaluate(r,tmp_path/'e');assert result.returncode==0,result.stderr
 summary=json.loads((tmp_path/'e/validation_summary.json').read_text());assert summary['counts']['all_attempts']==8 and summary['counts']['failed_exports']==1
 acc=summary['modes']['ON']['accuracy_by_source']['md_labeled_endpoint'];assert acc['attempts']==2 and acc['joint_valid_mre_under20_count']==1
 assert summary['modes']['ON']['MD_real_paired_control']['all_pairs_invalid_penalized_mean_advantage_over_gap']==pytest.approx(-1.)

def test_native_arms_share_initial_coordinates_and_rng_but_change_condition():
 class Fake:
  def langevin_dynamics(self,c,ld,counts,types):
   coords=torch.rand((int(counts.sum()),3));noise=torch.randn_like(coords)
   return {'num_atoms':counts,'atom_types':types,'frac_coords':coords+.001*noise+.01*c[0,0], 'lengths':torch.ones(len(counts),3)*10,'angles':torch.ones(len(counts),3)*90}
 random.seed(4);np.random.seed(4);torch.manual_seed(4);state=capture_rng(torch,np);initial=torch.rand((4,3));restore_rng(state,torch,np)
 outcomes=native_arms(Fake(),{'left':torch.zeros(1,2),'right':torch.ones(1,2),'repeat_left':torch.zeros(1,2)},torch.tensor([4]),torch.tensor([5,5,8,8]),state,initial,torch,np,None)
 assert all(x['status']=='ok' for x in outcomes.values())
 assert equal_rng(outcomes['left']['rng_end'],outcomes['right']['rng_end'],torch,np)
 assert torch.equal(outcomes['left']['samples'][0]['frac_coords'],outcomes['repeat_left']['samples'][0]['frac_coords'])
 assert not torch.equal(outcomes['left']['samples'][0]['frac_coords'],outcomes['right']['samples'][0]['frac_coords'])

def test_cluster_CI_keeps_zero_valid_batch_and_repeated_seed_unit():
 f=pd.DataFrame({'pair_id':['p0','p0','p1','p1'],'batch_id':['b0','b0','b1','b1'],'advantage':[1.,1.,-1.,-1.],'pair_geometry_valid':[True,True,False,False]})
 r=cluster_interval(f,'advantage','batch_id',repetitions=1000)
 assert r['clusters']==2 and r['empty_resamples']>0 and r['ci95']==[1.,1.]


def test_nonfinite_condition_is_rejected(tmp_path,fixture_root):
 r=tmp_path/'g';shutil.copytree(fixture_root/'generation',r)
 d=pd.read_csv(r/'generation_manifest.csv');p=r/d.tensor_file.iloc[0];v=torch.load(p,weights_only=False);v['cond_z'][0,0]=float('nan');torch.save(v,p);seal(r)
 result=evaluate(r,tmp_path/'e');assert result.returncode!=0 and 'Nonfiniteconditiontensor' in result.stderr

def test_ON_cannot_change_OFF_cell(tmp_path,fixture_root):
 r=tmp_path/'g';shutil.copytree(fixture_root/'generation',r)
 d=pd.read_csv(r/'generation_manifest.csv');p=r/d[d.guidance=='ON'].tensor_file.iloc[0];v=torch.load(p,weights_only=False);v['samples']['lengths']+=1;torch.save(v,p);seal(r)
 result=evaluate(r,tmp_path/'e');assert result.returncode!=0 and 'ON changedfrozencell' in result.stderr


@pytest.mark.parametrize('lengths,angles,reason', [
 ([-1,12,12],[90,90,90],'lengths must be positive'),
 ([12,12,12],[200,90,90],'angles must lie'),
 ([12,12,12],[10,10,170],'Gram determinant'),
 ([12,12,12],[90,90,float('nan')],'Nonfinite')])
def test_invalid_sampler_cells_are_rejected_before_normalizing_export(lengths,angles,reason):
 sample={'lengths':torch.tensor([lengths],dtype=torch.float64),'angles':torch.tensor([angles],dtype=torch.float64)}
 with pytest.raises(ValueError,match=reason):validated_lattice(sample)

def test_valid_skew_sampler_cell_preserves_parameters():
 lattice=validated_lattice({'lengths':torch.tensor([[11.,12.,13.]]),'angles':torch.tensor([[75.,80.,100.]])})
 assert lattice.abc==pytest.approx([11.,12.,13.]) and lattice.angles==pytest.approx([75.,80.,100.])


def test_GPU_development_protocol_cannot_count_as_scientific_acceptance(tmp_path,fixture_root):
 r=tmp_path/'g';shutil.copytree(fixture_root/'generation',r)
 protocol=json.loads((r/'protocol.json').read_text());protocol['development_only']=True;write_json(r/'protocol.json',protocol)
 provenance=json.loads((r/'provenance.json').read_text());provenance['cpu_fixture_only']=False;provenance['development_only']=True;provenance['protocol_sha256']=sha(r/'protocol.json')
 for field,name in [('source_contract_sha256','source_contract.json'),('runner_sha256','executed_generate_validation.py'),('common_sha256','executed_validation_common.py')]:
  (r/name).write_text('unit fixture implementation provenance\n');provenance[field]=sha(r/name)
 write_json(r/'provenance.json',provenance);seal(r)
 result=evaluate(r,tmp_path/'e');assert result.returncode==0,result.stderr
 summary=json.loads((tmp_path/'e/validation_summary.json').read_text());assert summary['classification']=='DEVELOPMENT_SMOKE_ONLY'
 assert not any(v['decision']['eligible'] for v in summary['modes'].values())


def test_condition_assembly_calls_nondeterministic_composition_once_and_reuses_repeat():
 from types import SimpleNamespace
 class NoisyComposition:
  calls=0
  def __call__(self,inputs):
   self.calls+=1
   return torch.ones(len(inputs[1]),2)+self.calls*1e-7
 class PDMLayer:
  calls=0
  def __call__(self,x):
   self.calls+=1
   return x*2
 class NoisyConditionHead:
  calls=0
  def __call__(self,z,values):
   self.calls+=1
   return torch.cat([z,*values.values()],dim=-1)+self.calls*1e-7
 comp,pdm,head=NoisyComposition(),PDMLayer(),NoisyConditionHead()
 model=SimpleNamespace(multiemb=SimpleNamespace(cond_keys=['element_values','composition'],sub_emb_list=[pdm,comp]),zgivenc=head)
 conditions,embeddings,shared=condition_arms(model,torch.zeros(2,3),torch.tensor([5,8,5,8]),torch.tensor([2,2]),{'left':torch.zeros(2,21),'right':torch.ones(2,21)},repeat_left=True)
 assert comp.calls==1 and pdm.calls==2 and head.calls==2
 assert torch.equal(conditions['left'],conditions['repeat_left'])
 assert conditions['left'].data_ptr()==conditions['repeat_left'].data_ptr()
 assert embeddings['left'].data_ptr()==embeddings['repeat_left'].data_ptr()
 assert shared.shape==(2,2) and not torch.equal(conditions['left'],conditions['right'])

def test_changed_composition_embedding_is_rejected_as_confounded_pair(tmp_path,fixture_root):
 r=tmp_path/'g';shutil.copytree(fixture_root/'generation',r)
 frame=pd.read_csv(r/'generation_manifest.csv');off=frame[(frame.arm=='right')&(frame.guidance=='OFF')].iloc[0];on=frame[(frame.arm=='right')&(frame.guidance=='ON')].iloc[0]
 op=r/off.tensor_file;v=torch.load(op,weights_only=False);v['composition_embedding']+=1e-7;torch.save(v,op)
 npth=r/on.tensor_file;v=torch.load(npth,weights_only=False);v['composition_embedding']+=1e-7;v['off_tensor_sha256']=sha(op);torch.save(v,npth);seal(r)
 result=evaluate(r,tmp_path/'e');assert result.returncode!=0 and 'Paired composition embedding differs' in result.stderr
