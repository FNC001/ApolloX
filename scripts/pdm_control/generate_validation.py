#!/usr/bin/env python3
"""New PDM model prior generation; ON refines the identical OFF structure.
No reference CIF/coordinates/lattice or test inputs are accepted by this CLI.
All predeclared attempts persist, including failed exports and pending work.
"""
import argparse,importlib,json,os,random,shutil,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from validation_common import *


def validated_lattice(sample):
 """Reject impossible sampler cells before pymatgen can normalize their parameters."""
 import numpy as np
 from pymatgen.core import Lattice
 lengths=np.asarray(sample['lengths'].detach().cpu(),dtype=float).reshape(3)
 angles=np.asarray(sample['angles'].detach().cpu(),dtype=float).reshape(3)
 require(np.isfinite(lengths).all() and np.isfinite(angles).all(),'Nonfinite cell parameters')
 require((lengths>0).all(),'Generated cell lengths must be positive')
 require(((angles>0)&(angles<180)).all(),'Generated cell angles must lie strictly between 0 and 180 degrees')
 a,b,c=np.cos(np.deg2rad(angles));gram=np.array([[1,c,b],[c,1,a],[b,a,1]])
 require(float(np.linalg.det(gram))>1e-12,'Generated cell angles have a nonpositive or degenerate Gram determinant')
 lattice=Lattice.from_parameters(*lengths,*angles)
 require(np.isfinite(lattice.matrix).all() and np.isfinite(lattice.volume) and lattice.volume>0,'Nonpositive generated cell')
 require(np.allclose(lattice.abc,lengths,rtol=1e-7,atol=1e-7) and np.allclose(lattice.angles,angles,rtol=1e-7,atol=1e-7),'Constructed lattice changed sampler cell parameters')
 return lattice


def split_samples(samples,counts,atom_types,torch):
 result=cpu_tree(samples,torch);expected=counts.cpu();n=int(expected.sum());b=len(counts)
 require(torch.equal(result['num_atoms'],expected),'Sampler changed atom counts')
 require(torch.equal(result['atom_types'],atom_types.cpu()),'Sampler changed atom identity/order')
 for key,shape in [('lengths',(b,3)),('angles',(b,3)),('frac_coords',(n,3))]:
  require(tuple(result[key].shape)==shape,'Invalid sampler output shape '+key)
 rows=[];start=0
 for index,count in enumerate(expected.tolist()):
  rows.append({'num_atoms':expected[index:index+1],'lengths':result['lengths'][index:index+1],'angles':result['angles'][index:index+1],
   'frac_coords':result['frac_coords'][start:start+count],'atom_types':result['atom_types'][start:start+count],'is_traj':False});start+=count
 return rows

def native_arms(model,conditions,counts,atom_types,rng_start,initial_frac,torch,np,ld):
 outcomes={}
 for arm,cond in conditions.items():
  restore_rng(rng_start,torch,np)
  # Native inherited sampler draws initial frac coordinates before its first noise.
  require(torch.equal(torch.rand((int(counts.sum()),3),device=atom_types.device),initial_frac),'Initial RNG coordinates differ')
  restore_rng(rng_start,torch,np);started=time.monotonic()
  try:
   with torch.no_grad():samples=model.langevin_dynamics(cond,ld,counts,atom_types)
   outcomes[arm]={'status':'ok','samples':split_samples(samples,counts,atom_types,torch),'error':''}
  except Exception as error:
   outcomes[arm]={'status':'error','error':type(error).__name__+': '+str(error),'traceback':traceback.format_exc()}
  outcomes[arm].update(rng_end=capture_rng(torch,np),seconds=time.monotonic()-started)
 return outcomes

def source_contract(repo):
 roots=[repo/'cond-cdvae/cdvae',repo/'cond-cdvae/conf',repo/'apollox_common'];paths=[]
 for root in roots:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.yaml','.yml','.json'})
 paths += [repo/'cond-cdvae/scripts/eval_utils.py',repo/'cond-cdvae/scripts/extract_gen.py']
 return [{'path':str(p.relative_to(repo)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))]

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ['repo','model','generator-inputs','protocol','guidance-config','out']:parser.add_argument('--'+name,required=True,type=Path)
 parser.add_argument('--candidate-id',required=True);parser.add_argument('--stage',choices=['pilot12','gate64','all_validation'],required=True)
 parser.add_argument('--checkpoint-sha256',required=True);parser.add_argument('--gpu',default='0');parser.add_argument('--allow-cpu-fixture',action='store_true')
 parser.add_argument('--guidance-module',default='cdvae.common.pdm_guidance')
 parser.add_argument('--snapshot-name',default='unspecified');parser.add_argument('--expected-cost-pdm',type=float)
 args=parser.parse_args();repo=args.repo.resolve();model_path=args.model.resolve();inputs=args.generator_inputs.resolve();out=args.out.resolve()
 require(not out.exists(),'Fresh output required; never overwrite/retry failed samples silently')
 os.environ['CUDA_VISIBLE_DEVICES']='' if args.allow_cpu_fixture else args.gpu;os.environ['PROJECT_ROOT']=str(repo/'cond-cdvae')
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
 sys.path[:0]=[str(repo/'cond-cdvae/scripts'),str(repo/'cond-cdvae'),str(repo)]
 import numpy as np,pandas as pd,torch
 from pymatgen.core import Composition,Lattice,Structure
 from pymatgen.io.vasp import Poscar
 from cdvae.common.condition_utils import prepare_pdm_values
 from cdvae.common.data_utils import lattice_params_to_matrix_torch
 from eval_utils import load_model
 refine=importlib.import_module(args.guidance_module).refine_pdm
 protocol=json.loads(args.protocol.read_text());settings=stage_settings(protocol,args.stage);guidance=json.loads(args.guidance_config.read_text())
 if not args.allow_cpu_fixture:require(torch.cuda.is_available(),'CUDA not available')
 device=torch.device('cpu' if args.allow_cpu_fixture else 'cuda:0');torch.set_num_threads(2)
 torch.backends.cudnn.benchmark=False
 if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 pair_plan=pd.read_csv(inputs/'pair_plan.csv',dtype=str,keep_default_na=False);singles=pd.read_csv(inputs/'singleton_plan.csv',dtype=str,keep_default_na=False)
 raw=pd.read_csv(inputs/'target_raw.csv',dtype={'label':str}).set_index('label');schema=json.loads((inputs/'pdm_schema.json').read_text());names=schema['features']
 require(len(pair_plan)==settings['MD_real_pairs'] and len(singles)==settings['legacy_real_singletons'],'Stage counts differ from frozen protocol')
 require(set(raw.index)==set(pair_plan.left_target_id)|set(pair_plan.right_target_id)|set(singles.target_id) and raw.index.is_unique,'Target identities differ')
 require(set(raw.columns)=={'formula',*names},'Generator inputs must contain onlycomposition/PDM')
 for name in ['pdm_schema.json','scaler_stats.txt']:require(sha(inputs/name)==sha(model_path/name),'Model/validation contract mismatch '+name)
 lines=(inputs/'scaler_stats.txt').read_text().splitlines();mean=np.fromstring(next(x for x in lines if x.startswith('Train Mean:')).split(':')[1],sep=',');std=np.fromstring(next(x for x in lines if x.startswith('Train Std:')).split(':')[1],sep=',')
 checkpoints=list(model_path.glob('*.ckpt'));require(len(checkpoints)==1 and sha(checkpoints[0])==args.checkpoint_sha256,'Ambiguous or changed checkpoint')
 contract={name:sha(model_path/name) for name in ['hparams.yaml','prop_scalers.pt','lattice_scaler.pt','pdm_schema.json','scaler_stats.txt']}
 sources=source_contract(repo)
 model,_,cfg=load_model(model_path,load_data=False)
 require(type(model).__name__=='PDMControlCDVAE','New candidate must retain explicitPDMControlCDVAE class')
 require(set(model.multiemb.cond_keys)=={'composition','element_values'},'Unexpected condition keys')
 require(model.pdm_control_config['version']==1 and model.pdm_control_config['schema']==schema,'Candidate schema/version differs')
 require(np.allclose(model.pdm_control_config['external_mean'],mean,rtol=1e-10,atol=1e-9) and np.allclose(model.pdm_control_config['external_std'],std,rtol=1e-10,atol=1e-9),'Candidate external normalizer differs')
 if args.expected_cost_pdm is not None:require(float(model.pdm_control_config['cost'])==args.expected_cost_pdm,'Candidate PDM loss weight differs from declared candidate')
 model.to(device).eval();require(len(model.sigmas)==50,'Unexpected noise schedule')
 require(model.langevin_dynamics.__func__.__module__=='cdvae.pl_modules.model','Runner assumes verified native initial-coordinate RNG order')
 attempts,layout=planned_attempts(pair_plan,singles,settings,args.candidate_id,args.stage)
 out.mkdir(parents=True)
 for dirname in ['structures','tensors','rng','inputs','model_contract']:(out/dirname).mkdir()
 for name in ['pair_plan.csv','singleton_plan.csv','target_raw.csv','pdm_schema.json','scaler_stats.txt','GENERATOR_INPUT_CONTRACT.json']:shutil.copy2(inputs/name,out/'inputs'/name)
 shutil.copy2(args.protocol,out/'protocol.json');shutil.copy2(args.guidance_config,out/'guidance_config.json');shutil.copy2(__file__,out/'executed_generate_validation.py');shutil.copy2(Path(__file__).with_name('validation_common.py'),out/'executed_validation_common.py')
 for name in contract:shutil.copy2(model_path/name,out/'model_contract'/name)
 write_csv(attempts,out/'expected_attempts.csv');write_csv(attempts,out/'generation_manifest.csv');write_json(out/'batch_layout.json',layout);write_json(out/'source_contract.json',sources)
 provenance={'candidate_id':args.candidate_id,'snapshot_name':args.snapshot_name,'stage':args.stage,'checkpoint_path':str(checkpoints[0]),'checkpoint_sha256':args.checkpoint_sha256,'model_contract_sha256':contract,
  'source_root':str(repo),'source_contract_sha256':sha(out/'source_contract.json'),'input_sha256':{p.name:sha(p) for p in (out/'inputs').iterdir()},
  'protocol_sha256':sha(out/'protocol.json'),'guidance_config_sha256':sha(out/'guidance_config.json'),'runner_sha256':sha(__file__),'common_sha256':sha(Path(__file__).with_name('validation_common.py')),
  'expected_attempts_sha256':sha(out/'expected_attempts.csv'),'batch_layout_sha256':sha(out/'batch_layout.json'),'model_cost_pdm':float(model.pdm_control_config['cost']),
  'model_condition_scaler':{'means':torch.as_tensor(model.prop_scalers[list(cfg.data.prop).index('element_values')].means).cpu().tolist(),'stds':torch.as_tensor(model.prop_scalers[list(cfg.data.prop).index('element_values')].stds).cpu().tolist()},'development_only':bool(protocol.get('development_only',False)),'prior_only':True,'reference_geometry_used':False,'guidance_ON_definition':'Fixed refinement of the identical correspondingOFF structure; not a separate learned-model response.',
  'torch':torch.__version__,'device':str(device),'settings':settings,'cpu_fixture_only':args.allow_cpu_fixture}
 write_json(out/'provenance.json',provenance)
 records=attempts.to_dict('records');byid={r['generated_id']:r for r in records};batch_records=[]
 ld=SimpleNamespace(n_step_each=settings['sampler']['n_step_each'],step_lr=settings['sampler']['step_lr'],min_sigma=settings['sampler']['min_sigma'],save_traj=False,disable_bar=True)
 def export(sample,path):
  for key in ['frac_coords','lengths','angles']:require(bool(torch.isfinite(sample[key]).all()),'Nonfinite structure '+key)
  lattice=validated_lattice(sample)
  Poscar(Structure(lattice,sample['atom_types'].tolist(),sample['frac_coords'].numpy(),coords_are_cartesian=False,to_unit_cell=True)).write_file(path)
 for seed_index,seed in enumerate(settings['sampling_seeds']):
  for batch_index,batch in enumerate(layout):
   paired=batch['batch_kind']=='pair';segment=(pair_plan if paired else singles).iloc[batch['start']:batch['stop']];batch_seed=int(seed)+batch_index*1009;sampler_seed=batch_seed+1000003
   random.seed(batch_seed);np.random.seed(batch_seed%2**32);torch.manual_seed(batch_seed)
   if device.type=='cuda':torch.cuda.manual_seed_all(batch_seed)
   counts_list=[];types=[]
   for formula in segment.formula:
    composition=Composition(formula).get_el_amt_dict();require(all(v>0 and float(v).is_integer() for v in composition.values()),'Nonintegercomposition')
    atoms=[Composition(symbol).elements[0].Z for symbol,count in sorted(composition.items()) for _ in range(int(count))];counts_list.append(len(atoms));types+=atoms
   counts=torch.tensor(counts_list,dtype=torch.long,device=device);atom_types=torch.tensor(types,dtype=torch.long,device=device);z=torch.randn(len(segment),model.hparams.latent_dim,device=device)
   arms=['left','right']+(['repeat_left'] if batch['repeat'] else []) if paired else ['single'];conditions={};extra={}
   with torch.no_grad():
    for arm in arms:
     side='left' if arm=='repeat_left' else arm;ids=segment[side+'_target_id' if paired else 'target_id'].tolist();scaled=(raw.loc[ids,names].to_numpy(float)-mean)/std
     pdm=prepare_pdm_values(model,scaled.tolist(),cfg.data.prop,model.prop_scalers,batched=True).to(device)
     emb=model.multiemb({'composition':(atom_types,counts),'element_values':pdm});conditions[arm]=model.zgivenc(z,emb)
     extra[arm]={'ids':ids,'pdm':pdm.detach().cpu(),'embedded_pdm':emb['element_values'].detach().cpu(),'scaled':scaled}
   random.seed(sampler_seed);np.random.seed(sampler_seed%2**32);torch.manual_seed(sampler_seed)
   if device.type=='cuda':torch.cuda.manual_seed_all(sampler_seed)
   start=capture_rng(torch,np);initial_frac=torch.rand((int(counts.sum()),3),device=device);restore_rng(start,torch,np)
   outcomes=native_arms(model,conditions,counts,atom_types,start,initial_frac,torch,np,ld)
   endpoints=[x['rng_end'] for x in outcomes.values()];rng_equal=all(equal_rng(endpoints[0],v,torch,np) for v in endpoints[1:])
   rng_rel=f'rng/s{seed_index}_{batch["batch_id"]}.pt';torch.save({'start':start,'end':{k:v['rng_end'] for k,v in outcomes.items()},'z':z.detach().cpu(),'initial_frac_coords':initial_frac.detach().cpu(),'num_atoms':counts.cpu(),'atom_types':atom_types.cpu()},out/rng_rel)
   batch_records.append({'seed_index':seed_index,'seed':seed,'batch_id':batch['batch_id'],'rng_equal':rng_equal,'all_native_success':all(x['status']=='ok' for x in outcomes.values()),'arms':{k:{key:v for key,v in x.items() if key not in ['samples','rng_end']} for k,x in outcomes.items()}})
   offsets=np.r_[0,np.cumsum(counts_list)]
   for arm,item in outcomes.items():
    for index,target_id in enumerate(extra[arm]['ids']):
     base=f's{seed_index}_{batch["batch_id"]}_{arm}_r{index:03d}';off=byid[base+'_OFF'];on=byid[base+'_ON']
     payload={'target_id':target_id,'pair_id':off['pair_id'],'guidance':'OFF','checkpoint_sha256':args.checkpoint_sha256,'seed':seed,'batch_id':batch['batch_id'],'z':z[index:index+1].detach().cpu(),
      'cond_z':conditions[arm][index:index+1].detach().cpu(),'embedded_pdm':extra[arm]['embedded_pdm'][index:index+1],'model_standardized_target':extra[arm]['pdm'][index:index+1],
      'external_standardized_target':extra[arm]['scaled'][index].tolist(),'raw_target':raw.loc[target_id,names].to_numpy(float).tolist(),'initial_frac_coords':initial_frac[offsets[index]:offsets[index+1]].detach().cpu()}
     off.update(status='error',tensor_file='tensors/'+base+'_OFF.pt');on.update(status='error',tensor_file='tensors/'+base+'_ON.pt')
     off_payload=dict(payload);on_payload={**payload,'guidance':'ON','off_tensor_file':off['tensor_file'],'guidance_config':guidance}
     try:
      require(item['status']=='ok',item.get('error','Native generationfailed'))
      sample=item['samples'][index];off_payload['samples']=sample;off_file='structures/'+base+'_OFF.vasp';export(sample,out/off_file);off.update(status='ok',filename=off_file)
     except Exception as exc:off['error']=type(exc).__name__+': '+str(exc)
     torch.save(off_payload,out/off['tensor_file'])
     on_payload['off_tensor_sha256']=sha(out/off['tensor_file'])
     try:
      require(off['status']=='ok','Refinementnotrun becauseOFF exportfailed: '+off['error'])
      sample=off_payload['samples'];frac=sample['frac_coords'].to(device);at=sample['atom_types'].to(device);na=sample['num_atoms'].to(device)
      cells=lattice_params_to_matrix_torch(sample['lengths'].to(device),sample['angles'].to(device))
      target=torch.as_tensor(raw.loc[target_id,names].to_numpy(float),device=device,dtype=frac.dtype).reshape(1,-1);scale=torch.as_tensor(std,device=device,dtype=frac.dtype)
      refined=refine(frac,at,na,cells,target,scale,schema,config=guidance)
      require(torch.equal(refined['cell_matrices'],cells),'Guidance unexpectedlychanged nativecell')
      guided={**sample,'frac_coords':refined['frac_coords'].detach().cpu()};on_payload.update(samples=guided,guidance_report=cpu_tree(refined,torch))
      on_file='structures/'+base+'_ON.vasp';export(guided,out/on_file);on.update(status='ok',filename=on_file)
     except Exception as exc:on['error']=type(exc).__name__+': '+str(exc)
     torch.save(on_payload,out/on['tensor_file'])
   write_csv(pd.DataFrame(records),out/'generation_manifest.csv');write_json(out/'batch_records.json',batch_records)
   print(json.dumps({'event':'batch_committed','candidate_id':args.candidate_id,'stage':args.stage,'seed_index':seed_index,'batch_id':batch['batch_id'],'committed':sum(x['status']!='pending' for x in records),'planned':len(records)}),flush=True)
 require(all(r['status'] in ['ok','error'] for r in records),'Pending outputs remain')
 for name,digest in contract.items():require(sha(model_path/name)==digest,'Model contract changed duringgeneration')
 require(sha(checkpoints[0])==args.checkpoint_sha256 and source_contract(repo)==sources,'Model/source changed duringgeneration')
 files=[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]
 write_json(out/'GENERATION_COMPLETE.json',{'status':'ALL_PLANNED_ATTEMPTS_COMMITTED_NOT_ACCURACY_PASS','planned':len(records),'exports':sum(r['status']=='ok' for r in records),'errors':sum(r['status']=='error' for r in records),'files':files})
if __name__=='__main__':main()
