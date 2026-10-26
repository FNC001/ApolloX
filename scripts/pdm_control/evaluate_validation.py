#!/usr/bin/env python3
"""Independent hard-PDM evaluation of predeclared prior generation and refinement.
No model descriptor or differentiable/guidance implementation is used as gold.
"""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd,torch
from pymatgen.core import Structure,Composition
from validation_common import *


def describe(path,names,cutoff,formula):
 s=Structure.from_file(path)
 require(np.isfinite(s.cart_coords).all() and np.isfinite(s.lattice.matrix).all() and s.volume>0,'Nonfinite/nonpositive generated geometry')
 i,j=np.triu_indices(len(s),1);require(len(i)>0,'Insufficientatoms')
 d=s.distance_matrix;require(np.isfinite(d[i,j]).all(),'Nonfiniteperiodic distances')
 symbols=[x.specie.symbol for x in s];counts=Counter(''.join(sorted((symbols[a],symbols[b]))) for a,b in zip(i,j) if d[a,b]<cutoff)
 comp=s.composition.as_dict()==Composition(formula).as_dict();minimum=float(d[i,j].min())
 return np.array([counts[n] for n in names],float),{'composition_matches':comp,'minimum_distance_A':minimum,'geometry_valid':bool(comp and minimum>=.5),'near_training_distance_screen_ge1A':bool(comp and minimum>=1.),'volume_A3':float(s.volume)},s

def finite_mean(values):
 a=np.asarray(values,float);a=a[np.isfinite(a)];return float(a.mean()) if len(a) else None

def cluster_interval(frame,column,cluster,valid_only=True,seed=20261103,repetitions=10000):
 ids=frame[cluster].unique();eligible=frame[frame.pair_geometry_valid] if valid_only else frame
 table=eligible.groupby(cluster)[column].agg(['sum','count']).reindex(ids,fill_value=0)
 if len(ids)<2:return {'ci95':None,'clusters':len(ids),'empty_resamples':0}
 rng=np.random.default_rng(seed);idx=rng.integers(len(ids),size=(repetitions,len(ids)));n=table['count'].to_numpy()[idx].sum(1);total=table['sum'].to_numpy()[idx].sum(1)
 v=total[n>0]/n[n>0]
 return {'ci95':np.quantile(v,[.025,.975]).tolist() if len(v) else None,'clusters':len(ids),'empty_resamples':int((n==0).sum()),'replications':repetitions}

def accuracy(frame):
 valid=frame[frame.geometry_valid];success=valid[valid.mre<.2]
 return {'attempts':len(frame),'exports':int(frame.export_ok.sum()),'errors':int((frame.status=='error').sum()),'geometry_valid':len(valid),
  'geometry_valid_rate':float(len(valid)/len(frame)),'joint_valid_mre_under20_count':len(success),'joint_valid_mre_under20_rate':float(len(success)/len(frame)),
  'valid_mean_mre':finite_mean(valid.mre),'valid_median_mre':float(valid.mre.median()) if len(valid) else None,
  'all_export_mean_mre_diagnostic':finite_mean(frame.mre),'all_export_median_mre_diagnostic':float(frame.mre.dropna().median()) if frame.mre.notna().any() else None,
  'valid_mean_normalized_mae':finite_mean(valid.assigned_mae_std),'all_export_mean_normalized_mae_diagnostic':finite_mean(frame.assigned_mae_std),
  'zero_target_component_attempts':int((frame.zero_target_features>0).sum()),'minimum_distance_below1A_exports':int(((frame.minimum_distance_A<1)&frame.export_ok).sum())}

def control_summary(frame):
 valid=frame[frame.pair_geometry_valid]
 return {'pair_seed_attempts':len(frame),'unique_pairs':int(frame.pair_id.nunique()),'valid_pair_seed_attempts':len(valid),'pair_geometry_valid_rate':len(valid)/len(frame),
  'mean_symmetric_advantage_valid':finite_mean(valid.advantage),'mean_advantage_over_gap_valid':finite_mean(valid.advantage_over_gap),
  'all_pairs_invalid_penalized_mean_advantage_over_gap':float(frame.invalid_penalized_advantage_over_gap.mean()),
  'pair_bootstrap':cluster_interval(frame,'advantage','pair_id'),'batch_bootstrap':cluster_interval(frame,'advantage','batch_id'),
  'all_pairs_invalid_penalized_pair_bootstrap':cluster_interval(frame,'invalid_penalized_advantage_over_gap','pair_id',False),
  'same_z_pair_seed_attempts':int(frame.same_z.sum()),'different_condition_pair_seed_attempts':int(frame.different_cond_z.sum())}

def gate_decision(stage,protocol,acc,control,fixture=False):
 if fixture:return {'eligible':False,'status':'CPU_FIXTURE_NOT_SCIENTIFIC_ACCEPTANCE'}
 md=acc['md_labeled_endpoint'];legacy=acc['legacy_variable_composition'];tests={}
 if stage=='pilot12':
  g=protocol['stages']['pilot12']['screen_to_full_gate']
  tests={'MD_geometry':md['geometry_valid_rate']>=g['MD_geometry_valid_rate_min'],'MD_joint_precision':md['joint_valid_mre_under20_rate']>=g['MD_joint_valid_MRE_under20_min'],
    'positive_paired_point_estimate':control['mean_symmetric_advantage_valid'] is not None and control['mean_symmetric_advantage_valid']>g['mean_symmetric_advantage_min_strict']}
  status='CHEAP_SCREEN_ONLY_NOT_CONTROL_ACCURACY_ACCEPTANCE'
 else:
  g=protocol['stages']['gate64']['go_long_training'];pairci=control['pair_bootstrap']['ci95'];batchci=control['batch_bootstrap']['ci95']
  tests={'MD_geometry':md['geometry_valid_rate']>=g['MD_geometry_valid_rate_min'],'MD_joint_precision':md['joint_valid_mre_under20_rate']>=g['MD_joint_valid_MRE_under20_min'],
   'paired_relative_advantage':control['mean_advantage_over_gap_valid'] is not None and control['mean_advantage_over_gap_valid']>=g['MD_mean_symmetric_advantage_over_target_gap_min'],
   'pair_CI_positive':pairci is not None and pairci[0]>g['MD_valid_pair_bootstrap_lower95_strictly_above'],
   'batch_CI_positive':batchci is not None and batchci[0]>g['MD_batch_bootstrap_lower95_strictly_above'],
   'failures_penalized_advantage_positive':control['all_pairs_invalid_penalized_mean_advantage_over_gap']>g['MD_all_pairs_invalid_penalized_mean_advantage_over_gap_strictly_above'],
   'legacy_geometry':legacy['geometry_valid_rate']>=g['legacy_geometry_valid_rate_min'],'legacy_joint_precision':legacy['joint_valid_mre_under20_rate']>=g['legacy_joint_valid_MRE_under20_min']}
  status='PREDECLARED_DEVELOPMENT_GATE_NOT_INDEPENDENT_TEST'
 return {'eligible':all(tests.values()),'status':status,'checks':tests}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generation',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--allow-fixture',action='store_true')
 a=parser.parse_args();root=a.generation.resolve();out=a.out.resolve();require(not out.exists(),'Fresh evaluation outputrequired');require(not out.is_relative_to(root),'Evaluation output must be outside immutable generation evidence')
 complete=json.loads((root/'GENERATION_COMPLETE.json').read_text());require(complete['status']=='ALL_PLANNED_ATTEMPTS_COMMITTED_NOT_ACCURACY_PASS','Generationnotcomplete')
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()};expected={r['path'] for r in complete['files']}|{'GENERATION_COMPLETE.json'};require(actual==expected,'Generation inventory changed')
 for r in complete['files']:
  p=safe_child(root,r['path']);require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Generationbytes changed '+r['path'])
 prov=json.loads((root/'provenance.json').read_text());fixture=prov.get('cpu_fixture_only',False);require(not fixture or a.allow_fixture,'CPUfixture requires explicitallowance')
 require(prov['prior_only'] and not prov['reference_geometry_used'],'Reference-conditioned generation cannot countasprior')
 for name,digest in prov['input_sha256'].items():require(sha(root/'inputs'/name)==digest,'Generatorinputs changed')
 for name,digest in prov['model_contract_sha256'].items():require(sha(root/'model_contract'/name)==digest,'Modelcontract changed')
 for field,name in [('protocol_sha256','protocol.json'),('guidance_config_sha256','guidance_config.json'),('expected_attempts_sha256','expected_attempts.csv'),('batch_layout_sha256','batch_layout.json')]:require(prov[field]==sha(root/name),'Changed executioncontract '+name)
 if not fixture:
  for field,name in [('source_contract_sha256','source_contract.json'),('runner_sha256','executed_generate_validation.py'),('common_sha256','executed_validation_common.py')]:require(prov[field]==sha(root/name),'Changed implementation provenance '+name)
 for name in ['pdm_schema.json','scaler_stats.txt']:require(sha(root/'inputs'/name)==sha(root/'model_contract'/name),'Input/model schema orscaler mismatch')
 protocol=json.loads((root/'protocol.json').read_text());settings=stage_settings(protocol,prov['stage']);schema=json.loads((root/'inputs/pdm_schema.json').read_text());names=schema['features'];require(schema['schema_version']==2 and schema['modes']==['pair'] and schema['cutoff']==5,'UnexpectedPDMschema')
 pp=pd.read_csv(root/'inputs/pair_plan.csv',dtype=str,keep_default_na=False);sp=pd.read_csv(root/'inputs/singleton_plan.csv',dtype=str,keep_default_na=False);raw=pd.read_csv(root/'inputs/target_raw.csv',dtype={'label':str}).set_index('label')
 require(len(pp)==settings['MD_real_pairs'] and len(sp)==settings['legacy_real_singletons'],'Stage counts differ from declared protocol')
 require(raw.index.is_unique and set(raw.index)==set(pp.left_target_id)|set(pp.right_target_id)|set(sp.target_id),'Target identities differ from plan')
 plan,layout=planned_attempts(pp,sp,settings,prov['candidate_id'],prov['stage']);saved=pd.read_csv(root/'expected_attempts.csv',dtype=str,keep_default_na=False);frame=pd.read_csv(root/'generation_manifest.csv',dtype=str,keep_default_na=False)
 require(json.loads((root/'batch_layout.json').read_text())==layout,'Batch layout differsfrominputs')
 stable=[x for x in plan if x not in ['status','error','filename','tensor_file']]
 require(plan[stable].astype(str).equals(saved[stable]) and saved[stable].equals(frame[stable]) and len(frame)==len(plan),'Attempt identity/denominator differsfromfrozenplan')
 require(frame.generated_id.is_unique and set(frame.status)<={'ok','error'},'Missing/pending/duplicate attempts')
 lines=(root/'inputs/scaler_stats.txt').read_text().splitlines();mean=np.fromstring(next(x for x in lines if x.startswith('Train Mean:')).split(':')[1],sep=',');std=np.fromstring(next(x for x in lines if x.startswith('Train Std:')).split(':')[1],sep=',')
 require(len(std)==21 and np.isfinite(std).all() and (std>0).all(),'Invalidscaler')
 ms=prov['model_condition_scaler'];model_mean=np.array(ms['means']);model_std=np.array(ms['stds']);require((model_std>0).all(),'Invalidmodelscaler')
 batches=json.loads((root/'batch_records.json').read_text());bybatch={(int(x['seed_index']),x['batch_id']):x for x in batches};require(len(bybatch)==len(settings['sampling_seeds'])*len(layout),'Missingcommittedbatch')
 rng_audit=[]
 for (si,bid),batch in bybatch.items():
  rows=frame[(frame.seed_index==str(si))&(frame.batch_id==bid)];paths=rows.rng_file.unique();require(len(paths)==1,'Batch RNGfile mismatch')
  evidence=torch.load(safe_child(root,paths[0]),map_location='cpu',weights_only=False);ends=list(evidence['end'].values());same=all(equal_rng(ends[0],v,torch,np) for v in ends[1:])
  require(same==batch['rng_equal'],'Actual/claimed RNG endpoints differ')
  if batch['all_native_success']:require(same,'Successful matchedarms useddifferentRNGstreams')
  rng_audit.append({'seed_index':si,'batch_id':bid,'actual_end_equal':same,'all_native_success':batch['all_native_success']})
 rows=[];payloads={};vectors={}
 for row in frame.to_dict('records'):
  gid=row['generated_id'];target=raw.loc[row['target_id'],names].to_numpy(float);other=raw.loc[row['swapped_target_id'],names].to_numpy(float) if row['swapped_target_id'] else None
  payload=torch.load(safe_child(root,row['tensor_file']),map_location='cpu',weights_only=False);payloads[gid]=payload
  for key in ['target_id','pair_id','guidance','batch_id']:require(payload[key]==row[key],'Tensor identitydiffers '+key)
  require(payload['checkpoint_sha256']==prov['checkpoint_sha256'],'Tensor checkpointdiffers')
  require(np.array_equal(payload['raw_target'],target),'Raw input target differs')
  scaled=(target-mean)/std;require(np.allclose(payload['external_standardized_target'],scaled,rtol=1e-10,atol=1e-10),'ExternalPDMtransform differs')
  require(np.allclose(payload['model_standardized_target'].numpy().reshape(21),(scaled-model_mean)/model_std,rtol=1e-5,atol=1e-5),'ModelPDMtransform differs')
  for key in ['z','embedded_pdm','cond_z','model_standardized_target','initial_frac_coords']:require(bool(torch.isfinite(payload[key]).all()),'Nonfiniteconditiontensor')
  item={**row,'geometry_valid':False,'export_ok':False,'mre':None,'assigned_mae_std':None,'swapped_mae_std':None,'advantage':None,'minimum_distance_A':None,'volume_A3':None,'zero_target_features':int((target==0).sum()),'hard_descriptor_guidance_discrepancy_max':None}
  if row['guidance']=='ON':
   offrow=frame[frame.generated_id==row['off_generated_id']].iloc[0];require(payload['off_tensor_sha256']==sha(safe_child(root,offrow.tensor_file)),'ON isnotboundtotheidenticalOFFtensor')
   off=torch.load(safe_child(root,offrow.tensor_file),map_location='cpu',weights_only=False)
   for key in ['z','cond_z','embedded_pdm','model_standardized_target','initial_frac_coords']:require(torch.equal(payload[key],off[key]),'ON changedmodelinput '+key)
   if row['status']=='ok':
    require(offrow.status=='ok','ON cannot succeedwithoutOFFexport')
    for key in ['num_atoms','atom_types','lengths','angles']:require(torch.equal(payload['samples'][key],off['samples'][key]),'ON changedfrozencell/atoms')
  if row['status']=='ok':
   vector,geometry,structure=describe(safe_child(root,row['filename']),names,5,row['formula']);vectors[gid]=vector;sample=payload['samples'];delta=structure.frac_coords-sample['frac_coords'].numpy();delta-=np.round(delta)
   require(np.max(np.abs(delta))<1e-6 and [x.specie.Z for x in structure]==sample['atom_types'].tolist(),'Tensor/POSCARcoordinates/speciesmismatch')
   require(np.allclose(structure.lattice.abc,sample['lengths'].numpy().reshape(3),atol=1e-6) and np.allclose(structure.lattice.angles,sample['angles'].numpy().reshape(3),atol=1e-6),'Tensor/POSCARcellmismatch')
   errors=np.abs(vector-target);item.update(geometry,export_ok=True,mre=float((errors/(np.abs(target)+1e-8)).mean()),assigned_mae_std=float((errors/std).mean()))
   if other is not None:item.update(swapped_mae_std=float((np.abs(vector-other)/std).mean()),advantage=float(((np.abs(vector-other)-errors)/std).mean()))
   if row['guidance']=='ON' and 'final_hard_raw' in payload.get('guidance_report',{}):item['hard_descriptor_guidance_discrepancy_max']=float(np.abs(vector-payload['guidance_report']['final_hard_raw'].numpy().reshape(21)).max())
  rows.append(item)
 result=pd.DataFrame(rows);primary=result[result.arm!='repeat_left'];pair_rows=[]
 for (mode,pid,seed),g in primary[primary.pair_id!=''].groupby(['guidance','pair_id','seed']):
  require(len(g)==2 and set(g.target_id)==set(g.swapped_target_id),'Incompletepairattempt')
  left=g[g.arm=='left'].iloc[0];right=g[g.arm=='right'].iloc[0];lp,rp=payloads[left.generated_id],payloads[right.generated_id]
  require(torch.equal(lp['z'],rp['z']) and torch.equal(lp['initial_frac_coords'],rp['initial_frac_coords']),'Pairedprior/initialcoordinatesdiffer')
  gap=float((np.abs(raw.loc[left.target_id,names].to_numpy(float)-raw.loc[right.target_id,names].to_numpy(float))/std).mean());require(gap>0,'Zerotargetgap')
  valid=bool(g.geometry_valid.all());adv=finite_mean(g.advantage) if bool(g.export_ok.all()) else None
  pair_rows.append({'guidance':mode,'pair_id':pid,'seed':seed,'batch_id':left.batch_id,'source_domain':left.source_domain,'pair_geometry_valid':valid,'same_z':True,
   'different_cond_z':not torch.equal(lp['cond_z'],rp['cond_z']),'target_gap':gap,'advantage':adv,'advantage_over_gap':adv/gap if adv is not None else None,'invalid_penalized_advantage_over_gap':adv/gap if valid else -1.})
 pairs=pd.DataFrame(pair_rows);repeat_rows=[]
 for row in result[result.arm=='repeat_left'].itertuples():
  original=primary[(primary.target_id==row.target_id)&(primary.guidance==row.guidance)&(primary.seed==row.seed)];require(len(original)==1,'Repeatpairmissing')
  o=original.iloc[0];l,r=payloads[o.generated_id],payloads[row.generated_id]
  for key in ['z','cond_z','embedded_pdm','model_standardized_target','initial_frac_coords']:require(torch.equal(l[key],r[key]),'Sameconditionrepeatdiffers '+key)
  rms=None;pdm=None
  if o.export_ok and row.export_ok:
   delta=l['samples']['frac_coords'].double()-r['samples']['frac_coords'].double();delta-=delta.round();rms=float(delta.square().mean().sqrt());pdm=float((np.abs(vectors[o.generated_id]-vectors[row.generated_id])/std).mean())
  repeat_rows.append({'guidance':row.guidance,'target_id':row.target_id,'pair_id':row.pair_id,'seed':row.seed,'same_inputs':True,'both_exports':bool(o.export_ok and row.export_ok),'wrapped_fractional_rms':rms,'repeat_hard_PDM_normalized_mae':pdm})
 modes={}
 for mode,g in primary.groupby('guidance'):
  acc={source:accuracy(x) for source,x in g.groupby('source_domain')};control=control_summary(pairs[pairs.guidance==mode]);decision=gate_decision(prov['stage'],protocol,acc,control,fixture or bool(protocol.get('development_only',False)))
  modes[mode]={'accuracy_by_source':acc,'MD_real_paired_control':control,'decision':decision,'claim_scope':'Learnedmodelpriorgeneration' if mode=='OFF' else 'WholemodelplusfixedPDMrefinementpipeline; notnetworkalone'}
 classification='NEITHER_PROTOCOL_MEETS_DEVELOPMENT_GATE'
 if modes['OFF']['decision']['eligible']:classification='OFF_MEETS_CURRENT_STAGE_GATE'
 elif modes['ON']['decision']['eligible']:classification='ONLY_GUIDED_PIPELINE_MEETS_CURRENT_STAGE_GATE'
 if prov['stage']=='pilot12':classification+='__CHEAP_SCREEN_NOT_FINAL_ACCURACY'
 if protocol.get('development_only',False):classification='DEVELOPMENT_SMOKE_ONLY'
 if fixture:classification='UNIT_FIXTURE_ONLY'
 summary={'status':'COMPLETE_DECLARED_VALIDATION_ATTEMPTS_AUDITED','scientific_success_not_implied':True,'development_only':bool(protocol.get('development_only',False)),'candidate_id':prov['candidate_id'],'snapshot_name':prov.get('snapshot_name','unspecified'),'model_cost_pdm':prov.get('model_cost_pdm'),'stage':prov['stage'],'checkpoint_sha256':prov['checkpoint_sha256'],'checkpoint_path':prov['checkpoint_path'],
  'generation_seal_sha256':sha(root/'GENERATION_COMPLETE.json'),'target_raw_sha256':sha(root/'inputs/target_raw.csv'),'protocol_sha256':sha(root/'protocol.json'),'evaluator_sha256':sha(__file__),
  'counts':{'all_attempts':len(result),'primary_attempts':len(primary),'repeat_attempts':len(repeat_rows),'exports':int(result.export_ok.sum()),'failed_exports':int((result.status=='error').sum())},
  'modes':modes,'classification':classification,'repeat_summary':{mode:{'repeats':len(g),'mean_fractional_rms':finite_mean(g.wrapped_fractional_rms),'mean_hard_PDM_normalized_mae':finite_mean(g.repeat_hard_PDM_normalized_mae)} for mode,g in pd.DataFrame(repeat_rows).groupby('guidance')},
  'definitions':{'MRE':'mean(abs(g-t)/(abs(t)+1e-8)), all21featuresincludingzeros','advantage':'symmetric swappedminusassigned meanabsolutePDMdistance dividedbytrainStd; positivefavorsassigned','geometry':'exactcomposition, finitepositivevolume, minperiodicdistance>=0.5A; notphysicalstability','bootstrap':'PairID orlogicalbatchcluster resampling retainsbothseeds,10000replications; developmentonly'},
  'limitations':['Validation is reused fordevelopment, neverafinaluntouchedtest.','MD highgap subsetdoesnotrepresentallcompositions; legacysingletons giveaccuracyonly.','OFFandONarenotinterchangeableclaims; ONhasexplicitoptimizeraccess torequestedPDM.','Wrappedfractionalrepeatdifferencesarenotfixedcellphysicaldisplacements.','Noenergy/stabilityinference.','Pilot500updates areonlyacheapscreen, notfull5000updateaccuracy.']}
 out.mkdir(parents=True);result.to_csv(out/'per_attempt_metrics.csv',index=False);pairs.to_csv(out/'paired_metrics.csv',index=False);pd.DataFrame(repeat_rows).to_csv(out/'repeat_metrics.csv',index=False);write_json(out/'rng_audit.json',rng_audit);write_json(out/'validation_summary.json',summary)
 for r in complete['files']:require(sha(safe_child(root,r['path']))==r['sha256'],'Evidencechanged duringevaluation')
 print(json.dumps({'classification':classification,'counts':summary['counts'],'modes':modes},indent=2,allow_nan=False))
if __name__=='__main__':main()
