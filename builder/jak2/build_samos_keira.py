"""Build private Jak II Samos and Keira rig candidates from local disc assets.

This stage preserves authored facial shapes for native-channel fitting. It
does not install, patch an active runtime, or claim live eye/face readiness.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

import numpy as np

RELEASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RELEASE),str(RELEASE/'stages')]
from retarget_glb import load,retarget

def configure_readers(work):
 os.environ['CINEMATIC_BUILD_WORK']=str(work.resolve())
 global decompress,model,DRAW,buffers,textures
 from native_io import activate
 activate()
 from fr3_models import decompress,model,DRAW
 from inspect_faces import buffers
 from inspect_transformations import textures

ALIASES={
 'samos':{'beardDriver':'head','Birdfeather':'BIRDhead1','Birdjaw':'BIRDhead1',
          'BIRDR2':'BIRDR1','BIRDL2':'BIRDL1','Birdtail':'BIRD1',
          'RhairBASE':'head','Rhair':'head','LthumbA':'LthumbB','RthumbA':'RthumbB'},
 'keira':{'rEara':'rEar1','rEarb':'rEar2','lEara':'lEar1','lEarb':'lEar2',
          'Rmidhaira':'Rhair1','Rmidhairb':'Rhair2','Lmidhaira':'Lhair1','Lmidhairb':'Lhair2',
          **{s+'bang'+a:'flip'+s+str(i) for s in 'LR' for i,a in enumerate('abc',1)},
          'lTopStrap':'lTopStrap2','rTopStrap':'rTopStrap2',
          'lBotStrap':'lBotStrap1','rBotStrap':'rBotStrap1',
          'Lanklestrap':'LfootFlaps','Ranklestrap':'RfootFlaps',
          'LpantFlap':'lKneeFlap','RpantFlap':'rKneeFlap',
          # Temporary mappings disappear with the unsupported mask surfaces.
          'maskstrap':'gogglesBase','mask':'gogglesBase'}}
KEIRA_MASK=(0,1,27)
KEIRA_TORCH=(2,3,4,5)
TORCH_ROTATION=np.array([[0,0,-1],[0,1,0],[1,0,0]],dtype='<f4')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(doc,binary,path):
 binary=bytearray(binary);binary+=b'\0'*(-len(binary)%4)
 encoded=json.dumps(doc,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%4)
 path.write_bytes(struct.pack('<III',0x46546c67,2,28+len(encoded)+len(binary))+
  struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary)

def correct_props(path):
 doc,binary,a=load(path);binary=bytearray(binary);primitives=doc['meshes'][0]['primitives']
 assert len(primitives)==28
 assert [doc['materials'][primitives[i]['material']]['name'] for i in KEIRA_MASK]==['keira-brownstraps-new','keira-maskbolt','keira-mask']
 written=set()
 for pi in KEIRA_TORCH:
  p=primitives[pi];fields=[p['attributes'][key] for key in ('POSITION','NORMAL')]
  fields.extend(ai for target in p['targets'] for ai in target.values())
  for ai in fields:
   if ai in written:continue
   written.add(ai);entry=doc['accessors'][ai];view=doc['bufferViews'][entry['bufferView']]
   assert entry['componentType']==5126 and entry['type']=='VEC3' and view.get('byteStride',12)==12
   value=np.asarray(a(ai)@TORCH_ROTATION.T,dtype='<f4');start=view.get('byteOffset',0)+entry.get('byteOffset',0)
   binary[start:start+value.nbytes]=value.tobytes();entry.update(min=value.min(0).tolist(),max=value.max(0).tolist())
 doc['meshes'][0]['primitives']=[p for i,p in enumerate(primitives) if i not in KEIRA_MASK]
 save(doc,binary,path)

def native_first_pose(doc,accessor,animation_name):
 """Evaluate an exported native clip at its authored first frame."""
 from scipy.spatial.transform import Rotation
 animation=next(a for a in doc['animations'] if a['name']==animation_name)
 local=[np.asarray(n.get('matrix',np.eye(4).T.ravel()),dtype=float).reshape(4,4).T.copy() for n in doc['nodes']]
 curves={}
 for channel in animation['channels']:
  key=channel['target']['path']
  if key=='weights':continue
  sampler=animation['samplers'][channel['sampler']]
  assert sampler.get('interpolation','LINEAR')=='LINEAR'
  assert float(accessor(sampler['input']).ravel()[0])==0
  value=accessor(sampler['output'])[0].astype(float);assert np.isfinite(value).all()
  curves.setdefault(channel['target']['node'],{})[key]=value
 for node,paths in curves.items():
  assert set(paths)=={'translation','rotation','scale'}
  local[node]=np.eye(4)
  local[node][:3,:3]=Rotation.from_quat(paths['rotation']).as_matrix()@np.diag(paths['scale'])
  local[node][:3,3]=paths['translation']
 parents={child:i for i,n in enumerate(doc['nodes']) for child in n.get('children',[])};world={}
 def visit(node):
  if node not in world:world[node]=visit(parents[node])@local[node] if node in parents else local[node]
  return world[node]
 return np.asarray([visit(j) for j in doc['skins'][0]['joints']])

def articulate_samos_beard(path,native_path):
 """Fit the cinematic beard to native animation's neutral skin field.

 Native beard inverse binds contain a correction already compensated by the
 original mesh. Cancel that correction in the cinematic bind vertices, while
 retaining both native beard joints, their weights and animated movement.
 """
 low,_,la=load(native_path);doc,binary,a=load(path)
 names=[low['nodes'][i]['name'] for i in low['skins'][0]['joints']]
 assert names==[doc['nodes'][i]['name'] for i in doc['skins'][0]['joints']]
 ib=la(low['skins'][0]['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).astype(float)
 assert np.array_equal(la(low['skins'][0]['inverseBindMatrices']),a(doc['skins'][0]['inverseBindMatrices']))
 skin=native_first_pose(low,la,'samos-idle')@ib;head=names.index('head')
 relative=np.linalg.inv(skin[head])@skin
 beard=[names.index('beard_lip'),names.index('beard')]
 for animation in ('samos-idle-bots','samos-stand0','samos-raise-ship'):
  animated=native_first_pose(low,la,animation)@ib;other=np.linalg.inv(animated[head])@animated
  assert np.allclose(other[beard],relative[beard],atol=1e-7,rtol=0)
 def normal(values):return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-20)
 binary=bytearray(binary);changed=[];maximum_error=0.;maximum_normal_error=0.
 def write(index,values):
  entry=doc['accessors'][index];view=doc['bufferViews'][entry['bufferView']]
  assert entry['componentType']==5126 and entry['type']=='VEC3' and view.get('byteStride',12)==12
  values=np.asarray(values,dtype='<f4');start=view.get('byteOffset',0)+entry.get('byteOffset',0)
  binary[start:start+values.nbytes]=values.tobytes();entry.update(min=values.min(0).tolist(),max=values.max(0).tolist())
  return values.astype(float)
 for pi,p in enumerate(doc['meshes'][0]['primitives']):
  j=a(p['attributes']['JOINTS_0']).astype(int);w=a(p['attributes']['WEIGHTS_0']).astype(float)
  mask=(np.isin(j,beard)&(w>0)).any(axis=1)
  if not mask.any():continue
  assert pi==6 and int(mask.sum())==326
  assert set(j[mask][w[mask]>0].tolist())<={head,*beard}
  xyz=a(p['attributes']['POSITION']).astype(float);n=a(p['attributes']['NORMAL']).astype(float)
  blended=np.einsum('nkij,nk->nij',relative[j],w);inverse=np.linalg.inv(blended[mask])
  assert np.linalg.cond(blended[mask,:3,:3]).max()<1.01
  fixed=xyz.copy();fixed[mask]=np.einsum('nij,nj->ni',inverse,np.c_[xyz[mask],np.ones(mask.sum())])[:,:3]
  # merc2.vert transforms normals with the weighted skin matrix directly.
  fixed_n=n.copy();fixed_n[mask]=normal(np.einsum('nij,nj->ni',inverse[:,:3,:3],n[mask]))
  written=write(p['attributes']['POSITION'],fixed);written_n=write(p['attributes']['NORMAL'],fixed_n)
  for target in p['targets']:
   delta=a(target['POSITION']).astype(float);fixed_delta=delta.copy()
   fixed_delta[mask]=np.einsum('nij,nj->ni',inverse[:,:3,:3],delta[mask]);write(target['POSITION'],fixed_delta)
   delta=a(target['NORMAL']).astype(float);fixed_delta=delta.copy()
   fixed_delta[mask]=normal(np.einsum('nij,nj->ni',inverse[:,:3,:3],n[mask]+delta[mask]))-fixed_n[mask]
   write(target['NORMAL'],fixed_delta)
  restored=np.einsum('nij,nj->ni',blended,np.c_[written,np.ones(len(written))])[:,:3]
  restored_n=normal(np.einsum('nij,nj->ni',blended[:,:3,:3],written_n))
  error=float(np.linalg.norm(restored[mask]-xyz[mask],axis=1).max())
  normal_error=float(np.linalg.norm(restored_n[mask]-normal(n[mask]),axis=1).max())
  assert error<2e-7 and normal_error<2e-7
  assert np.array_equal(written[~mask],xyz[~mask]) and np.array_equal(written_n[~mask],n[~mask])
  maximum_error=max(maximum_error,error);maximum_normal_error=max(maximum_normal_error,normal_error)
  changed.append(dict(primitive=pi,vertices=int(mask.sum())))
 assert len(changed)==1
 save(doc,binary,path)
 return dict(method='Inverse blended native neutral beard skin field relative to head',
  reference='samos-idle frame 0',matchingReferences=['samos-idle-bots','samos-stand0','samos-raise-ship'],
  changed=changed,nativeJointWeightsAndInverseBindsUnchanged=True,liveBeardArticulationRetained=True,
  authoredCinematicNeutralShapeRestored=True,allOtherGeometryUVTopologyColoursUnchanged=True,
  normalsAndAuthoredMorphDeltasRebased=True,maximumNeutralReconstructionErrorMetres=maximum_error,
  maximumNeutralNormalError=maximum_normal_error)

def articulate_keira_cuffs(path,native_path,donor_path):
 """Transfer Jak II's native bend field without altering cinematic clothing."""
 from jak2.mesh_topology import components,nearest_triangle
 doc,binary,a=load(path);native,_,na=load(native_path);donor,_,da=load(donor_path)
 pi=next(i for i,p in enumerate(doc['meshes'][0]['primitives']) if doc['materials'][p['material']]['name']=='keira-pantslarge')
 ni=next(i for i,p in enumerate(native['meshes'][0]['primitives']) if native['materials'][p['material']]['name']=='keira-pants')
 p=doc['meshes'][0]['primitives'][pi];np_=native['meshes'][0]['primitives'][ni]
 xyz=a(p['attributes']['POSITION']);nxyz=na(np_['attributes']['POSITION']).astype(float)
 tri,parts=components(doc,a,p)
 selected=np.unique(tri[np.concatenate([part for part in parts if xyz[np.unique(tri[part]),1].max()<.8])])
 assert len(selected)==120
 native_tri=na(np_['indices']).reshape(-1,3)
 nj,nw=na(np_['attributes']['JOINTS_0']),na(np_['attributes']['WEIGHTS_0'])
 joints,weights=a(p['attributes']['JOINTS_0']),a(p['attributes']['WEIGHTS_0'])
 oldj,oldw=joints.copy(),weights.copy()
 source_ids=np.unique(da(donor['meshes'][0]['primitives'][21]['indices']).ravel())
 records=[];maximum_error=0.
 for vertex in selected:
  sign=1 if xyz[vertex,0]>0 else -1;side='L' if sign==1 else 'R'
  eligible=np.where((nxyz[native_tri,1].max(1)<.8)&np.all(nxyz[native_tri,0]*sign>0,axis=1))[0]
  assert len(eligible)==66
  local,bary,distance=nearest_triangle(xyz[vertex].astype(float),nxyz[native_tri[eligible]])
  bary=np.clip(bary,0,1);bary/=bary.sum();triangle=int(eligible[local]);dense=np.zeros(len(doc['skins'][0]['joints']))
  for v,b in zip(native_tri[triangle],bary):np.add.at(dense,nj[v],nw[v]*b)
  dense[dense<1e-8]=0;dense/=dense.sum();active=np.flatnonzero(dense>0)
  assert 0<len(active)<=3
  joints[vertex],weights[vertex]=0,0
  for column,j in enumerate(sorted(active,key=lambda j:-dense[j])):joints[vertex,column]=j;weights[vertex,column]=dense[j]
  actual=np.zeros_like(dense);np.add.at(actual,joints[vertex],weights[vertex])
  error=float(np.max(np.abs(actual-dense)));maximum_error=max(maximum_error,error);assert error<1e-6
  records.append(dict(vertex=int(vertex),sourceVertex=int(source_ids[vertex]),side=side,nativePrimitive=ni,nativeTriangle=triangle,
    nativeVertices=native_tri[triangle].tolist(),barycentric=bary.tolist(),nativeSurfaceDistanceMetres=distance))
 outside=np.setdiff1d(np.arange(len(xyz)),selected)
 assert np.array_equal(joints[outside],oldj[outside]) and np.array_equal(weights[outside],oldw[outside])
 binary=bytearray(binary)
 for key,values in [('JOINTS_0',joints),('WEIGHTS_0',weights)]:
  ac=doc['accessors'][p['attributes'][key]];bv=doc['bufferViews'][ac['bufferView']]
  assert 'byteStride' not in bv;start=bv.get('byteOffset',0)+ac.get('byteOffset',0)
  binary[start:start+values.nbytes]=values.tobytes()
 save(doc,binary,path)
 return dict(method='Closest original Jak II cuff triangle, barycentric native skin weights',outputPrimitive=pi,
  vertices=120,geometryHeightUVTopologyNormalsAndMorphsUnchanged=True,originalBootsPreserved=True,
  maximumReconstructedJointWeightError=maximum_error,maximumNativeSurfaceDistanceMetres=max(r['nativeSurfaceDistanceMetres'] for r in records),verticesDetail=records)

def source_draws(raw,name):
 tx,_=textures(raw);merc=model(raw,name);rows=[]
 for ei,e in enumerate(merc['effects']):
  for di,packed in enumerate(e['draws']):
   mode,texture,eye,first,count,tris,no_strip=DRAW.unpack(packed)
   rows.append(dict(sourceEffect=ei,sourceDraw=di,sourceTexture=tx[texture]['name'],
     sourceEyeId=eye,sourceDrawMode=mode,sourceHasEnvmap=bool(e['tail'][8]),triangles=tris))
 return merc,rows

def morph_inventory(doc,a):
 count=len(doc['meshes'][0]['primitives'][0].get('targets',[]));channels=[]
 for channel in range(count):
  parts=[]
  for pi,p in enumerate(doc['meshes'][0]['primitives']):
   ids=np.unique(a(p['indices']));delta=a(p['targets'][channel]['POSITION'])[ids]
   lengths=np.linalg.norm(delta,axis=1);moving=int(np.count_nonzero(lengths>1e-7))
   if moving:parts.append(dict(primitive=pi,material=doc['materials'][p['material']]['name'],
     movingVertices=moving,maximumDisplacementMetres=float(lengths.max())))
  channels.append(dict(channel=channel,movingParts=parts))
 return channels

def material_target(label,row):
 """Provisional native semantic groups; live visibility must still be checked."""
 effect,texture=row['sourceEffect'],row['sourceTexture']
 if label=='samos':
  if effect==6 and texture=='samos-teeth2':return 4
  return {0:3,1:2,2:3,3:0,4:2,5:1,6:2}[effect]
 if effect in (0,5):return None
 if effect==1:return 1 if texture=='keira-torch-guard-01' else 4
 if effect==2:return 2 if texture.startswith('programmer_eye_') else 1
 return {3:3,4:0}[effect]

def build_one(label,data,work,source,output):
 configure_readers(work)
 native=source/f'decompiler_out/jak2/levels/stadblmp/{label}-lod0.glb'
 donor=data/f'decompiler_out/jak2/levels/lintcstb/{label}-highres-lod0.glb'
 native_fr3=work/'import/out/jak2/fr3/stadblmp.fr3';donor_fr3=data/'out/jak2/fr3/lintcstb.fr3'
 destination=output/f'{label}-lod0.glb'
 retarget(native,donor,destination,donor_path=donor_fr3,joint_aliases=ALIASES[label],preserve_morph_targets=True,use_default_aliases=False)
 cuff_report=None
 if label=='keira':
  correct_props(destination)
  cuff_report=articulate_keira_cuffs(destination,native,donor)
 low,_,la=load(native);high,_,ha=load(donor);doc,_,a=load(destination)
 nr=decompress(native_fr3);hr=decompress(donor_fr3)
 native_record,native_draws=source_draws(nr,label+'-lod0')
 _,donor_draws=source_draws(hr,label+'-highres-lod0')
 _,_,_,native_vb=buffers(nr,label+'-lod0');nv=np.frombuffer(native_vb,'<f4').reshape(-1,16);nb=np.frombuffer(native_vb,'u1').reshape(-1,64)
 _,_,_,donor_vb=buffers(hr,label+'-highres-lod0');db=np.frombuffer(donor_vb,'u1').reshape(-1,64)
 native_names=[low['nodes'][i]['name'] for i in low['skins'][0]['joints']]
 donor_names=[high['nodes'][i]['name'] for i in high['skins'][0]['joints']]
 assert native_names==[doc['nodes'][i]['name'] for i in doc['skins'][0]['joints']]
 assert np.array_equal(la(low['skins'][0]['inverseBindMatrices']),a(doc['skins'][0]['inverseBindMatrices']))
 native_used=set()
 for p in low['meshes'][0]['primitives']:
  ids=np.unique(la(p['indices']));native_used.update(la(p['attributes']['JOINTS_0'])[ids][la(p['attributes']['WEIGHTS_0'])[ids]>0].tolist())
  assert np.array_equal(la(p['attributes']['POSITION'])[ids]*4096,nv[ids,:3]),'Selective GLB geometry differs from full original FR3'
 original_indices=[i for i in range(len(high['meshes'][0]['primitives'])) if label!='keira' or i not in KEIRA_MASK]
 assert len(original_indices)==len(doc['meshes'][0]['primitives'])
 rows=[];all_used=set();triangles=0;mapping={}
 for pi,(si,p) in enumerate(zip(original_indices,doc['meshes'][0]['primitives'])):
  sp=high['meshes'][0]['primitives'][si];used,inverse=np.unique(ha(sp['indices']).ravel(),return_inverse=True)
  assert np.array_equal(a(p['indices']).ravel(),inverse)
  expected=ha(sp['attributes']['POSITION'])[used]
  normal=ha(sp['attributes']['NORMAL'])[used];normal=normal/np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-20)
  rotated=label=='keira' and si in KEIRA_TORCH
  if rotated:expected=expected@TORCH_ROTATION.T;normal=normal@TORCH_ROTATION.T
  assert np.array_equal(expected,a(p['attributes']['POSITION']))
  assert np.allclose(normal,a(p['attributes']['NORMAL']),atol=1e-7)
  assert np.array_equal(ha(sp['attributes']['TEXCOORD_0'])[used],a(p['attributes']['TEXCOORD_0']))
  assert np.array_equal(db[used,56:60],a(p['attributes']['COLOR_0']))
  assert len(p['targets'])==len(sp['targets'])
  for t,st in zip(p['targets'],sp['targets']):
   for key in t:
    values=ha(st[key])[used]
    if rotated:values=values@TORCH_ROTATION.T
    assert np.array_equal(values,a(t[key]))
  joints=a(p['attributes']['JOINTS_0']).astype(int);weights=a(p['attributes']['WEIGHTS_0'])
  assert np.isfinite(weights).all() and (weights>=0).all() and np.allclose(weights.sum(1),1,atol=1e-6)
  assert ((weights>0).sum(1)<=3).all()
  active=set(joints[weights>0].tolist());assert active<=native_used,(label,active-native_used)
  all_used|=active
  source_j=ha(sp['attributes']['JOINTS_0'])[used];source_w=ha(sp['attributes']['WEIGHTS_0'])[used]
  for ji in np.unique(source_j[source_w>0]):
   name=donor_names[int(ji)];mapping[name]=ALIASES[label].get(name,name)
  entry=copy.deepcopy(donor_draws[si]);entry.update(outputPrimitive=pi,sourcePrimitive=si,sourceVertices=used.tolist(),
    material=doc['materials'][p['material']]['name'],provisionalNativeEffect=material_target(label,entry),
    torchAxesCorrected=rotated,localLiveEyeSide=(entry['sourceEyeId']%2 if entry['sourceEyeId']!=255 else None))
  rows.append(entry);triangles+=len(inverse)//3
 max_evaluated=struct.unpack_from('<I',native_record['tail'],4)[0]-1
 assert max(all_used)<=max_evaluated==len(native_names)-1
 guard=None
 if label=='keira':
  def points(d,accessor,needle):
   p=next(p for p in d['meshes'][0]['primitives'] if needle in d['materials'][p['material']]['name'])
   return accessor(p['attributes']['POSITION'])[np.unique(accessor(p['indices']))]
  lp=points(low,la,'torch-guard');hp=points(doc,a,'torch-guard')
  dist=np.linalg.norm(lp[:,None]-hp[None,:],axis=-1);guard=float(max(dist.min(0).max(),dist.min(1).max()))
  assert guard<.006
 beard_report=articulate_samos_beard(destination,native) if label=='samos' else None
 if beard_report:doc,_,a=load(destination)
 native_channels=morph_inventory(low,la);donor_channels=morph_inventory(doc,a)
 report=dict(character=label,model=label+'-lod0',status='Private rig prototype; native facial fitting, live eye adapter, material reconstruction and in-game validation pending',
  output=destination.name,sha256=sha(destination),sourceNativeGLB=sha(native),sourceDonorGLB=sha(donor),nativeFR3=sha(native_fr3),donorFR3=sha(donor_fr3),
  triangles=triangles,primitives=len(rows),nativeJoints=len(native_names),nativeOriginalMaximumWeightedGLBJoint=max(native_used),maximumWeightedGLBJoint=max(all_used),
  nativeMercNumJoints=max_evaluated+1,allActiveJointsWithinOriginalSubmission=True,nativeJointOrderAndInverseBindsExact=True,
  gameplayMorphChannels=len(native_channels),preservedAuthoredCinematicShapes=len(donor_channels),nativeChannels=native_channels,authoredShapes=donor_channels,
  faceFitting=dict(nativeTargetCount=len(native_channels),cinematicShapeCount=len(donor_channels),
    cinematicToNativePrimitiveSets=({'10':[2,3,4],'11':[23],'6':[3,4,18]} if label=='samos' else {'7':[5],'8':[7]}),
    sourceDonorFacePrimitives=([10,11,6] if label=='samos' else [9,10]),
    nativeEyePrimitives=([1] if label=='samos' else [4]),
    prototypeEyePrimitives=([7,8] if label=='samos' else [5,6]),
    sourceMorphExportWarning=('Original Samos FR3 and GLB contain common near-metre morph displacements. Validate actual native gameplay use before attaching a cinematic bank; never blindly subtract or rebase this source data.' if label=='samos' else None)),
  nativeEyeDrawIDs=sorted({x['sourceEyeId'] for x in native_draws}),liveEyeAdapterRequired=True,staticEyesAccepted=False,
  nativeAnimationClips=[x['name'] for x in low.get('animations',[])],aliases=ALIASES[label],usedJointMapping=mapping,
  nativeEffectCount=len(native_record['effects']),nativeEffectDraws=native_draws,primitiveProvenance=rows,
  accessoryCorrections=(dict(omittedUnsupportedMaskPrimitives=list(KEIRA_MASK),torchPrimitiveSourceIndices=list(KEIRA_TORCH),torchRotation=TORCH_ROTATION.tolist(),
    maximumNativeTorchGuardAlignmentErrorMetres=guard,cinematicClosedFootwearPreserved=True,cinematicCuffHeightUnchanged=True,noJak1FootOrClothingSubstitutions=True) if label=='keira' else {}),
  nativeGeometryMatchesFullOriginalFR3=True,authoredGeometryUVTopologyColoursAndMorphsPreservedExceptDocumentedTorchRotation=(label!='samos'),
  nativeBeardArticulation=beard_report,
  nativeCuffArticulation=cuff_report,
  runtimeArchives=['stadblmp.fr3'],
  remaining=[('Audit native facial-channel use before enabling the anomalous legacy Samos bank' if label=='samos' else 'Fit native gameplay facial channels with explicit body, hand, hair and eye exceptions'),'Install actual live eye controller and route local eye sides without static textures','Restore native material visibility and reflection controls from provisional semantic map','Verify shadows, props, cloth and expressions in the game before installation'])
 destination.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data',required=True,type=Path);parser.add_argument('--work',required=True,type=Path)
 parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 os.environ['CINEMATIC_BUILD_WORK']=str(args.work.resolve())
 report=[build_one(label,args.data,args.work,args.source,args.output) for label in ('samos','keira')]
 (args.output/'prototype-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps([{key:r[key] for key in ('character','triangles','primitives','nativeJoints','gameplayMorphChannels','preservedAuthoredCinematicShapes','maximumWeightedGLBJoint','sha256')} for r in report],indent=2))

if __name__=='__main__':main()
