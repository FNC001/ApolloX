"""Shared immutable identities only; scientific descriptors live in the evaluator."""
import hashlib,json,os,random
from pathlib import Path

def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)
def write_csv(frame,path):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);os.replace(tmp,path)
def safe_child(root,relative):
 p=(Path(root)/str(relative)).resolve();require(p.is_relative_to(Path(root).resolve()),'Path escapes evidence root');return p
def capture_rng(torch,np):
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.random.get_rng_state().clone(),'torch_cuda':[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
def restore_rng(state,torch,np):
 random.setstate(state['python']);np.random.set_state(state['numpy']);torch.random.set_rng_state(state['torch_cpu'])
 if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])
def equal_rng(a,b,torch,np):
 return a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['torch_cpu'],b['torch_cpu']) and len(a['torch_cuda'])==len(b['torch_cuda']) and all(torch.equal(x,y) for x,y in zip(a['torch_cuda'],b['torch_cuda']))
def cpu_tree(x,torch):
 if isinstance(x,torch.Tensor):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu_tree(v,torch) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cpu_tree(v,torch) for v in x]
 return x

def stage_settings(protocol,stage):
 st=dict(protocol['stages'][stage])
 if stage=='all_validation':
  gate=protocol['stages']['gate64'];st.update(sampler=dict(gate['sampler']),sampling_seeds=list(gate['sampling_seeds']),pair_batch_size=gate['pair_batch_size'],singleton_batch_size=gate['singleton_batch_size'])
 require(isinstance(st['sampler'],dict),'Stage sampler not frozen')
 require(st['sampler']['n_step_each']==(10 if stage=='pilot12' else 100),'Unexpected native sampling fidelity')
 require(st['sampler']['sigma_levels']==50,'Expected50 sigma levels')
 st['repeat_pairs']=4 if stage=='pilot12' else 8
 return st

def build_layout(pair_plan,singleton_plan,settings):
 batches=[]
 for kind,frame,size in [('pair',pair_plan,settings['pair_batch_size']),('singleton',singleton_plan,settings['singleton_batch_size'])]:
  require(size>0,'Invalid batch size')
  for start in range(0,len(frame),size):
   chunk=frame.iloc[start:start+size]
   # Repeats use a complete first batch; target order and RNG layout remain identical.
   repeat=kind=='pair' and start<settings['repeat_pairs']
   if repeat:require(start+len(chunk)<=settings['repeat_pairs'],'Repeat count must align with complete batches')
   batches.append({'batch_id':f'{kind}-b{start//size:04d}','batch_kind':kind,'start':start,'stop':start+len(chunk),'repeat':repeat,
    'ids':chunk['pair_id' if kind=='pair' else 'target_id'].tolist()})
 return batches

def planned_attempts(pair_plan,singleton_plan,settings,candidate_id,stage):
 import pandas as pd
 layout=build_layout(pair_plan,singleton_plan,settings);rows=[]
 for seed_index,seed in enumerate(settings['sampling_seeds']):
  for batch_index,batch in enumerate(layout):
   paired=batch['batch_kind']=='pair';segment=(pair_plan if paired else singleton_plan).iloc[batch['start']:batch['stop']]
   arms=['left','right']+(['repeat_left'] if batch['repeat'] else []) if paired else ['single']
   batch_seed=int(seed)+batch_index*1009;sample_seed=batch_seed+1000003
   for arm in arms:
    side='left' if arm=='repeat_left' else arm
    for row_index,item in enumerate(segment.to_dict('records')):
     target=item[side+'_target_id'] if paired else item['target_id'];other=item[('right' if side=='left' else 'left')+'_target_id'] if paired else ''
     base=f's{seed_index}_{batch["batch_id"]}_{arm}_r{row_index:03d}'
     for mode in ['OFF','ON']:
      gid=base+'_'+mode
      rows.append({'generated_id':gid,'candidate_id':candidate_id,'stage':stage,'guidance':mode,'seed':seed,'seed_index':seed_index,'batch_seed':batch_seed,'sampler_seed':sample_seed,
       'batch_id':batch['batch_id'],'batch_kind':batch['batch_kind'],'batch_row':row_index,'pair_id':item['pair_id'] if paired else '',
       'target_id':target,'swapped_target_id':other,'reference_id':target,'formula':item['formula'],'source_domain':item['source_domain'],'target_role':'real',
       'arm':arm,'is_repeat':arm=='repeat_left','off_generated_id':base+'_OFF','status':'pending','error':'','filename':'','tensor_file':'','rng_file':f'rng/s{seed_index}_{batch["batch_id"]}.pt'})
 return pd.DataFrame(rows),layout
