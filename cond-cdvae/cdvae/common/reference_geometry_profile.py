"""Source-specific training-reference intervals and frozen local topology."""
from itertools import combinations
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list


POLICY = {'radius_multiplier':1.2, 'min_structures':5, 'min_observations':30,
          'quantiles':[.01,.99], 'quantile_method':'linear'}


class UnsupportedTopology(ValueError):
    def __init__(self, reason, detail):
        self.reason, self.detail = reason, detail
        super().__init__(reason)


def _source(source):
    if not isinstance(source,str) or not source.strip():
        raise ValueError('An explicit nonempty source is required')


def _bond_key(a,b):
    return '|'.join(sorted((a,b)))


def _angle_key(center,a,b,coordination):
    return '|'.join((center,*sorted((a,b)),f'CN={coordination}'))


def _topology(atoms):
    if not isinstance(atoms,Atoms) or not len(atoms):
        raise UnsupportedTopology('invalid_atoms',{})
    cell=np.asarray(atoms.cell,dtype=float);positions=np.asarray(atoms.positions,dtype=float)
    numbers=atoms.numbers.tolist();symbols=atoms.get_chemical_symbols()
    determinant=float(np.linalg.det(cell))
    if (not np.isfinite(cell).all() or not np.isfinite(positions).all()
            or not math.isfinite(determinant) or determinant<=0 or not atoms.pbc.all()):
        raise UnsupportedTopology('finite_positive_fully_periodic_cell_required',{})
    if any(z<=0 or z>=len(covalent_radii) or not math.isfinite(float(covalent_radii[z])) or covalent_radii[z]<=0 for z in numbers):
        raise UnsupportedTopology('missing_covalent_radius',{'atomic_numbers':numbers})
    cutoff=POLICY['radius_multiplier']*covalent_radii[atoms.numbers]
    try:
        left,right,shifts,vectors,distances=neighbor_list('ijSDd',atoms,cutoff,self_interaction=False)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
        raise UnsupportedTopology('neighbor_list_failed',{'error':str(error)}) from error
    self_images=[{'site':int(i),'shift':s.tolist()} for i,j,s in zip(left,right,shifts) if i==j]
    if self_images:
        raise UnsupportedTopology('self_periodic_images',{'images':self_images,'directed_neighbor_count':len(left)})
    canonical={}
    for i,j,s,v,d in zip(left,right,shifts,vectors,distances):
        i,j=int(i),int(j)
        if i>j:i,j,s,v=j,i,-s,-v
        key=(i,j);shift=tuple(int(x) for x in s)
        canonical.setdefault(key,{})[shift]=(np.asarray(v,dtype=float),float(d))
    repeated=[{'pair':list(pair),'shifts':[list(s) for s in images]} for pair,images in canonical.items() if len(images)!=1]
    if repeated:
        raise UnsupportedTopology('multiple_periodic_images_for_site_pair',{'pairs':repeated,'directed_neighbor_count':len(left)})
    bonds=[];neighbors=[[] for _ in atoms];oriented={}
    for (i,j),images in sorted(canonical.items()):
        shift,(vector,distance)=next(iter(images.items()))
        try:
            mic,minimum=find_mic(positions[j]-positions[i],cell,pbc=True)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            raise UnsupportedTopology('minimum_image_failed',{'pair':[i,j],'error':str(error)}) from error
        if not np.allclose(vector,mic,rtol=1e-10,atol=1e-10):
            raise UnsupportedTopology('neighbor_image_not_unique_mic',{'pair':[i,j],'neighbor_shift':list(shift)})
        if not math.isfinite(distance) or distance<=0:
            raise UnsupportedTopology('coincident_bonded_sites',{'pair':[i,j],'distance_A':distance if math.isfinite(distance) else None})
        bonds.append({'pair':[i,j],'image_shift':list(shift),'length_A':float(minimum),
                      'key':_bond_key(symbols[i],symbols[j]),'elements':sorted((symbols[i],symbols[j]))})
        neighbors[i].append(j);neighbors[j].append(i)
        oriented[i,j]=mic;oriented[j,i]=-mic
    counts=[len(x) for x in neighbors]
    angles=[]
    for center,adjacent in enumerate(neighbors):
        for first,second in combinations(sorted(adjacent),2):
            a,b=oriented[center,first],oriented[center,second]
            denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
            if not math.isfinite(denominator) or denominator<=0:
                raise UnsupportedTopology('undefined_local_angle',{'triple':[first,center,second]})
            cosine=float(np.clip(np.dot(a,b)/denominator,-1.,1.))
            degrees=float(np.degrees(np.arccos(cosine)))
            angles.append({'triple':[first,center,second],'angle_deg':degrees,
                'key':_angle_key(symbols[center],symbols[first],symbols[second],counts[center]),
                'center_element':symbols[center],'neighbor_elements':sorted((symbols[first],symbols[second])),
                'coordination':counts[center]})
    return {'supported':True,'topology_rule':'ASE pair cutoff = 1.2*(covalent_radius_i+covalent_radius_j)',
            'representation':'unique periodic MIC site pairs; angle center is middle index',
            'atomic_numbers':numbers,'cell':cell.tolist(),'pbc':atoms.pbc.tolist(),
            'bonds':bonds,'angles':angles,'coordination_numbers':counts,
            'zero_coordination_atoms':[i for i,n in enumerate(counts) if n==0],
            'directed_neighbor_count':int(len(left))}


def _group_profile(observations,kind):
    groups={}
    for key,rows in sorted(observations.items()):
        ids=sorted({row['structure_id'] for row in rows})
        values=[row['value'] for row in rows]
        interval=np.quantile(values,POLICY['quantiles'],method=POLICY['quantile_method']).tolist()
        reasons=[]
        if len(ids)<POLICY['min_structures']:reasons.append('insufficient_structures')
        if len(values)<POLICY['min_observations']:reasons.append('insufficient_observations')
        if not all(math.isfinite(v) and v>0 for v in interval):reasons.append('nonpositive_or_nonfinite_interval')
        if interval[0]>=interval[1]:reasons.append('degenerate_interval')
        if kind=='angle' and interval[1]>180:reasons.append('angle_interval_out_of_range')
        groups[key]={'covered':not reasons,'reasons':reasons,'structure_count':len(ids),
                     'observation_count':len(values),'structure_ids':ids,
                     'observed_quantiles':interval,'bounds':list(interval) if not reasons else None}
    return groups


def fit_reference_profile(records,*,source):
    """Fit fixed q01/q99 intervals from explicit same-source train records.

    Every record supplies ``id``, ``source``, ``split='train'`` and ASE ``atoms``.
    Supported reference observations are grouped by element pair or by center
    element, unordered neighbor elements and center coordination number. Every
    supplied structure retains its reference status and observation counts.
    """
    _source(source)
    records=list(records)
    ids=[]
    for record in records:
        if not isinstance(record,dict) or not {'id','source','split','atoms'}<=set(record):
            raise ValueError('Reference records need id, source, split and atoms')
        if not isinstance(record['id'],str) or not record['id'].strip():
            raise ValueError('Reference IDs must be nonempty strings')
        if record['source']!=source or record['split']!='train':
            raise ValueError('Reference fitting requires only explicit same-source training records')
        ids.append(record['id'])
    if len(ids)!=len(set(ids)):raise ValueError('Reference IDs must be unique')
    bonds,angles,receipts={},{},[]
    for record in records:
        receipt={'id':record['id'],'source':source,'split':'train'}
        try:
            topology=_topology(record['atoms'])
        except UnsupportedTopology as error:
            receipt.update(status='unsupported',reason=error.reason,detail=error.detail,
                           bond_observation_count=0,angle_observation_count=0)
        else:
            for item in topology['bonds']:
                bonds.setdefault(item['key'],[]).append({'structure_id':record['id'],'value':item['length_A']})
            for item in topology['angles']:
                angles.setdefault(item['key'],[]).append({'structure_id':record['id'],'value':item['angle_deg']})
            receipt.update(status='used' if topology['bonds'] else 'no_local_observations',
                bond_observation_count=len(topology['bonds']),angle_observation_count=len(topology['angles']),
                zero_coordination_atoms=topology['zero_coordination_atoms'],
                coordination_numbers=topology['coordination_numbers'])
        receipts.append(receipt)
    return {'version':1,'source':source,'fit_split':'train','policy':copy.deepcopy(POLICY),
        'reference_ids':ids,'reference_count':len(receipts),
        'usable_reference_count':sum(r['status']=='used' for r in receipts),
        'unsupported_reference_count':sum(r['status']=='unsupported' for r in receipts),
        'no_local_observation_reference_count':sum(r['status']=='no_local_observations' for r in receipts),
        'references':receipts,'bond_groups':_group_profile(bonds,'bond'),
        'angle_groups':_group_profile(angles,'angle')}


def _validate_profile(profile,source):
    _source(source)
    if (not isinstance(profile,dict) or type(profile.get('version')) is not int or profile['version']!=1
            or profile.get('source')!=source or profile.get('fit_split')!='train' or profile.get('policy')!=POLICY):
        raise ValueError('A matching same-source version 1 training profile is required')
    for group_name in ('bond_groups','angle_groups'):
        if not isinstance(profile.get(group_name),dict):raise ValueError('Missing profile groups')
        for group in profile[group_name].values():
            if not isinstance(group,dict) or type(group.get('covered')) is not bool:
                raise ValueError('Explicit group coverage is required')
            if (not isinstance(group.get('reasons'),list)
                    or any(not isinstance(reason,str) or not reason for reason in group['reasons'])
                    or group['covered']!= (len(group['reasons'])==0)):
                raise ValueError('Coverage reasons must match the group coverage state')
            if group['covered']:
                bounds=group.get('bounds')
                if (not isinstance(bounds,list) or len(bounds)!=2
                        or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in bounds)
                        or bounds[0]>=bounds[1] or (group_name=='angle_groups' and bounds[1]>180)
                        or type(group.get('structure_count')) is not int or group['structure_count']<5
                        or type(group.get('observation_count')) is not int or group['observation_count']<30):
                    raise ValueError('Covered groups require supported positive nondegenerate reference intervals')


def load_reference_profile(path,*,source,expected_sha256):
    """Load a same-source training profile after checking its exact file SHA256."""
    if (not isinstance(expected_sha256,str) or len(expected_sha256)!=64
            or any(char not in '0123456789abcdef' for char in expected_sha256)):
        raise ValueError('expected_sha256 must be an explicit lowercase SHA256 digest')
    content=Path(path).read_bytes()
    actual=hashlib.sha256(content).hexdigest()
    if actual!=expected_sha256:raise ValueError('Reference profile SHA256 does not match expected_sha256')
    profile=json.loads(content)
    _validate_profile(profile,source)
    return profile


def build_local_constraints(atoms,profile,*,source,structure_id):
    """Freeze initial topology and emit covered reference constraints per structure."""
    _validate_profile(profile,source)
    if not isinstance(structure_id,str) or not structure_id.strip():raise ValueError('An explicit structure_id is required')
    empty={'bond_pairs':[],'bond_bounds_A':[],'angle_triples':[],'angle_bounds_deg':[]}
    base={'version':1,'structure_id':structure_id,'source':source,'constraints':empty,
          'coverage_complete':False,'has_constraints':False,
          'coverage':{'total_bonds':None,'covered_bonds':0,'total_angles':None,'covered_angles':0,
                      'site_count':len(atoms) if isinstance(atoms,Atoms) else None,'zero_coordination_count':None}}
    try:topology=_topology(atoms)
    except UnsupportedTopology as error:
        return dict(base,status='unsupported_periodic_topology',reason=error.reason,detail=error.detail,
                    frozen_topology=None,missing_bonds=[],omitted_angles=[],zero_coordination_atoms=[])
    base['frozen_topology']=topology
    missing=[];omitted=[];covered_bonds=set()
    for index,bond in enumerate(topology['bonds']):
        group=profile['bond_groups'].get(bond['key'])
        if group is None or not group['covered']:
            missing.append({'bond_index':index,'pair':bond['pair'],'key':bond['key'],
                            'reasons':['missing_reference_group'] if group is None else list(group['reasons'])})
        else:
            empty['bond_pairs'].append(list(bond['pair']));empty['bond_bounds_A'].append(list(group['bounds']))
            covered_bonds.add(tuple(bond['pair']))
    for index,angle in enumerate(topology['angles']):
        group=profile['angle_groups'].get(angle['key'])
        first,center,second=angle['triple']
        missing_arms=[list(pair) for pair in (tuple(sorted((first,center))),tuple(sorted((second,center)))) if pair not in covered_bonds]
        reasons=[]
        if group is None:reasons.append('missing_reference_group')
        elif not group['covered']:reasons.extend(group['reasons'])
        if missing_arms:reasons.append('unsupported_bond_arm')
        if reasons:
            omitted.append({'angle_index':index,'triple':angle['triple'],'key':angle['key'],
                'reasons':reasons,'omitted_angle_reason':'unsupported_bond_arm' if missing_arms else 'missing_angle_reference',
                'unsupported_bond_arms':missing_arms})
        else:
            empty['angle_triples'].append(list(angle['triple']));empty['angle_bounds_deg'].append(list(group['bounds']))
    has=bool(empty['bond_pairs'] or empty['angle_triples'])
    complete=has and not missing and not omitted and not topology['zero_coordination_atoms']
    base.update(status='reference_coverage_complete' if complete else ('partial_reference_coverage' if has else 'missing_local_constraints'),
        coverage_complete=bool(complete),has_constraints=has,missing_bonds=missing,omitted_angles=omitted,
        zero_coordination_atoms=topology['zero_coordination_atoms'],
        coverage={'total_bonds':len(topology['bonds']),'covered_bonds':len(empty['bond_pairs']),
                  'total_angles':len(topology['angles']),'covered_angles':len(empty['angle_triples']),
                  'site_count':len(atoms),'zero_coordination_count':len(topology['zero_coordination_atoms'])})
    return base


def audit_frozen_topology(atoms,frozen_topology):
    """Compare saved topology using a 1e-10 Angstrom cell readback tolerance.

    Cell diagnostics use zero relative tolerance and report exact equality
    separately. Atomic numbers and periodic flags require exact equality.
    """
    if not isinstance(frozen_topology,dict) or frozen_topology.get('supported') is not True:
        raise ValueError('A supported frozen initial topology is required')
    original_edges={tuple(row['pair']) for row in frozen_topology['bonds']}
    original_angles={tuple(row['triple']) for row in frozen_topology['angles']}
    try:current=_topology(atoms)
    except UnsupportedTopology as error:
        return {'status':'unsupported_new_topology','reason':error.reason,'detail':error.detail,
                'original_bond_count':len(original_edges),'original_angle_count':len(original_angles),
                'constraints_replaced':False,'current_topology':None}
    cell=np.asarray(current['cell'],dtype=float)
    frozen_cell=np.asarray(frozen_topology['cell'],dtype=float)
    cell_abs_tolerance_A=1e-10
    cell_exact_match=bool(np.array_equal(cell,frozen_cell))
    cell_max_abs_difference_A=float(np.max(np.abs(cell-frozen_cell)))
    fixed=(current['atomic_numbers']==frozen_topology['atomic_numbers']
           and bool(np.allclose(cell,frozen_cell,atol=cell_abs_tolerance_A,rtol=0.))
           and current['pbc']==frozen_topology['pbc'])
    edges={tuple(row['pair']) for row in current['bonds']};angles={tuple(row['triple']) for row in current['angles']}
    return {'status':'fixed_inputs_changed' if not fixed else ('unchanged_site_topology' if edges==original_edges else 'changed_site_topology'),
            'fixed_inputs_match':fixed,'constraints_replaced':False,'edge_identity':'unique unordered site pair',
            'cell_exact_match':cell_exact_match,'cell_max_abs_difference_A':cell_max_abs_difference_A,
            'cell_abs_tolerance_A':cell_abs_tolerance_A,'cell_relative_tolerance':0.,
            'added_bonds':[list(x) for x in sorted(edges-original_edges)],
            'removed_bonds':[list(x) for x in sorted(original_edges-edges)],
            'added_angles':[list(x) for x in sorted(angles-original_angles)],
            'removed_angles':[list(x) for x in sorted(original_angles-angles)],
            'original_coordination_numbers':list(frozen_topology['coordination_numbers']),
            'current_coordination_numbers':current['coordination_numbers'],
            'original_zero_coordination_atoms':list(frozen_topology['zero_coordination_atoms']),
            'current_zero_coordination_atoms':current['zero_coordination_atoms'],'current_topology':current}
