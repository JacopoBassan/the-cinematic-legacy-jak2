"""Build Jak II cinematic NPC geometry with caller-supplied native hooks.

The Jak II stage supplies verified source paths, joint tables and material maps.
All geometry, texture and bind data come from the local extraction.
"""
import copy
import hashlib
import re
import struct
from pathlib import Path

import numpy as np
from fr3_models import Reader, DRAW, decompress, compress, model, serialize
from inspect_faces import buffers
from inspect_transformations import textures

ROOT = None
OUT = None
CAT = {}
FULLCAT = {}
EMPTY_MOD = struct.pack('<5QI2Q', 0,0,0,0,0,0,0,0)


def skeleton(art_name, expected):
    raise RuntimeError('Configure the native skeleton reader before building NPCs')


def donor_effect(low_name, effect, texture):
    raise RuntimeError('Configure the native material map before building NPCs')


def native_draw_kept(low_name, effect, texture, draw):
    raise RuntimeError('Configure native draw retention before building NPCs')


def vec(b, size): return struct.pack('<Q',len(b)//size)+b
def texture_bytes(t):
    def s(x): b=x.encode(); return struct.pack('<Q',len(b))+b
    return struct.pack('<HHI',t['width'],t['height'],t['combo'])+vec(t['pixels'],4)+s(t['name'])+s(t['page'])+t['flag']

def used_indices(m, ix):
    lists=[ix[d[3]:d[3]+d[4]] for e in m['effects'] for d in map(DRAW.unpack,e['draws'])]
    ids=np.unique(np.concatenate(lists));return ids[ids!=0xffffffff]

def active_joints(m, ix, vb):
    used=used_indices(m,ix); b=np.frombuffer(vb,'u1').reshape(-1,64)[used]; f=b.copy().view('<f4').reshape(-1,16)
    return set(int(x) for k in range(3) for x in b[f[:,8+k]>0,60+k])

def choose_map(name, supported, low_name):
    if name in supported: return name, name
    if re.match('[LR](index|ring|pinky)[ABC]$',name):
        side=name[0]; part=name[-1]; candidate=side+'middle'+part
        if candidate not in supported: candidate=side+'middleB'
        if candidate in supported: return candidate, candidate
    if re.match('[LR](middle|thumb)C$',name):
        candidate=name[:-1]+'B'; return candidate,candidate
    if name in {'HairBase','HairMid','LearBase','LearMid','RearBase','RearMid','learA','learB','rearA','rearB','Ponytail','jaw','browL','browR','Learbase','Learmid','Rearbase','Rearmid','neck'}:
        return 'head','head'
    if name in {'rearflap1','rearflap2','rearflap3'}: return 'rearflapbase','rearflapbase'
    if name=='holster':return 'hips','hips'
    if name=='chest' and 'realChest' in supported:return 'realChest','chest'
    if name in {'Lball','Rball'} and name[0]+'ankle' in supported:return name[0]+'ankle',name[0]+'ankle'
    if name=='hose':return 'realChest','chest'
    if name in {'Ropea','Ropeb'}:return 'hips','hips'
    if name in {'Knife','Cigar'}:return ('Lhand','Lhand') if name=='Knife' else ('Rhand','Rhand')
    raise AssertionError(('unmapped weighted joint',low_name,name))


def build_one(low_name, high_name, *, source_dir=None, eye_slot_map=None):
    lc,hc=CAT[low_name],CAT[high_name]
    hr=decompress(ROOT/hc['files'][0]);hm=model(hr,high_name)
    _,hi,_,hv=buffers(hr,high_name);ht,_=textures(hr)
    hs=skeleton(high_name.removesuffix('-lod0'),hc['joints'])
    ls=skeleton(low_name.removesuffix('-lod0'),lc['joints'])
    reports=[]
    for filename in lc['files']:
        original=(Path(source_dir) if source_dir is not None else ROOT)/filename;source_digest=hashlib.sha256(original.read_bytes()).hexdigest();lr=decompress(original);lm=model(lr,low_name)
        istart,li,_,lv=buffers(lr,low_name);lt,tend=textures(lr)
        supported_ids=active_joints(lm,li,lv)
        supported={name for name,item in ls.items() if item['id'] in supported_ids}
        hb=np.frombuffer(hv,'u1').reshape(-1,64)
        selections=[];removed_tris=0
        for ei,e in enumerate(hm['effects']):
            for packed in e['draws']:
                d=list(DRAW.unpack(packed));te=donor_effect(low_name,ei,ht[d[1]]['name'])
                if te is None:continue
                ix=hi[d[3]:d[3]+d[4]]
                selections.append((te,d,ix))
        used=np.unique(np.concatenate([ix[ix!=0xffffffff] for _,_,ix in selections]))
        source=hb[used].copy();sf=source.view('<f4').reshape(-1,16)
        weights=sf[:,8:11].copy();beforepos=sf[:,:3].copy();beforenorm=sf[:,4:7].copy();original_joints=source[:,60:63].copy()
        assert np.allclose(weights.sum(1),1,atol=1e-5)
        spos=np.c_[beforepos,np.ones(len(source))];nresult=np.zeros_like(beforenorm,dtype=float);presult=np.zeros_like(beforepos,dtype=float)
        byid={v['id']:k for k,v in hs.items()};mapping={};transforms={}
        for k in range(3):
            for joint in np.unique(original_joints[:,k]):
                active=(original_joints[:,k]==joint)&(weights[:,k]>0)
                if not active.any():continue
                name=byid[int(joint)];target,anchor=choose_map(name,supported,low_name)
                assert target in supported and anchor in hs
                matrix=np.linalg.inv(ls[target]['inv_bind'])@hs[anchor]['inv_bind']
                normal_matrix=np.linalg.inv(matrix[:3,:3]).T
                presult[active]+=(spos[active]@matrix.T)[:,:3]*weights[active,k,None]
                nresult[active]+=(beforenorm[active]@normal_matrix.T)*weights[active,k,None]
                source[active,60+k]=ls[target]['id']
                mapping[name]=target;transforms[name]=matrix.tolist()
        # Preserve source normals under exact identity transforms; changed bind
        # frames use normalized inverse-transpose directions.
        lengths=np.linalg.norm(nresult,axis=1);assert np.all(lengths>1e-7)
        sf[:,:3]=presult;sf[:,4:7]=nresult/lengths[:,None]
        torso_fit=None
        hand_fit=None
        assert np.all(np.isfinite(sf[:,:3])) and np.all(np.isfinite(sf[:,4:7]))
        # Catch accidental matrix-ID remapping collisions or unit errors before
        # a staged asset can be accepted for a runtime test.
        native_used=used_indices(lm,li);native_pos=np.frombuffer(lv,'<f4').reshape(-1,16)[native_used,:3]
        assert np.ptp(sf[:,1])<np.ptp(native_pos[:,1])*1.2,(low_name,'unexpected height')
        assert np.max(np.abs(sf[:,:3]))<np.max(np.abs(native_pos))*1.5,(low_name,'unexpected bound')
        assert all(int(j) in supported_ids for k in range(3) for j in source[sf[:,8+k]>0,60+k])
        region_report=None
        new=copy.deepcopy(lm)
        for e in new['effects']:
            assert e['mod']==EMPTY_MOD
            e['draws']=[]
        native_kept=0
        for ei,e in enumerate(lm['effects']):
            for packed in e['draws']:
                d=DRAW.unpack(packed)
                if native_draw_kept(low_name,ei,lt[d[1]]['name'],d):
                    new['effects'][ei]['draws'].append(packed);native_kept+=d[5]
        txmap={};added=[];existing={texture_bytes(t):i for i,t in enumerate(lt)}
        for i in sorted({d[1] for _,d,_ in selections}):
            key=texture_bytes(ht[i]);index=existing.get(key)
            if index is None:index=len(lt)+len(added);added.append(ht[i]);existing[key]=index
            txmap[i]=index
        vmap=np.full(len(hb),0xffffffff,dtype='<u4');vmap[used]=np.arange(len(used),dtype='<u4')+len(lv)//64
        app=[];nextidx=len(li)
        for ei,d,ix in selections:
            mapped=ix.copy();valid=ix!=0xffffffff;mapped[valid]=vmap[ix[valid]]
            assert np.all(mapped[valid]!=0xffffffff)
            app.append(mapped);d[1]=txmap[d[1]];d[2]=(eye_slot_map or {}).get(d[2],255);d[3]=nextidx;nextidx+=len(mapped)
            new['effects'][ei]['draws'].append(DRAW.pack(*d))
        assert all(e['draws'] for e in new['effects'])
        tail=bytearray(new['tail']);struct.pack_into('<I',tail,0,sum(len(e['draws']) for e in new['effects']));new['tail']=bytes(tail)
        allix=np.concatenate([li]+app);allvb=lv+source.tobytes()
        fixed=lr[:istart]+vec(allix.tobytes(),4)+vec(allvb,64)+lr[-2:]
        fixed=fixed[:lm['start']]+serialize(new)+fixed[lm['end']:]
        r=Reader(lr,2);r.take(r.num('Q'));tstart=r.pos
        fixed=fixed[:tstart]+struct.pack('<Q',len(lt)+len(added))+b''.join(texture_bytes(t) for t in lt+added)+fixed[tend:]
        checked=model(fixed,low_name);_,ci,_,cv=buffers(fixed,low_name);ct,_=textures(fixed)
        assert serialize(checked)==serialize(new) and cv==allvb and np.array_equal(ci,allix)
        assert cv[:len(lv)]==lv and np.array_equal(ci[:len(li)],li)
        assert [texture_bytes(t) for t in ct[:len(lt)]]==[texture_bytes(t) for t in lt]
        preserved=0
        for name,entries in FULLCAT.items():
            if name!=low_name and any(x['file']==filename for x in entries):
                assert serialize(model(lr,name))==serialize(model(fixed,name)),name;preserved+=1
        eye_report=None
        OUT.mkdir(parents=True,exist_ok=True);out=OUT/filename;compress(out,fixed);assert decompress(out)==fixed
        report=dict(file=filename,gameplay=low_name,donor=high_name,triangles=sum(DRAW.unpack(d)[5] for e in new['effects'] for d in e['draws']),native_preserved_triangles=native_kept,removed_staff_triangles=removed_tris,effects=len(new['effects']),native_effect_tails_preserved=all(a['tail']==b['tail'] for a,b in zip(lm['effects'],new['effects'])),matrix_ids_within_original_submission=True,joint_mapping=mapping,bind_transforms=transforms,body_extent_m=(np.ptp(sf[:,:3],axis=0)/4096).tolist(),source_sha256=source_digest,output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),output_bytes=out.stat().st_size,other_models_preserved=preserved,guard_material_effect_assignment=region_report,status='staged; requires runtime animation, attachment, eye and material checks')
        report['eye_surface_transfer']=eye_report
        report['torso_fit']=torso_fit
        report['hand_fit']=hand_fit
        report['static_eye_layers']=None
        report['requires_full_joint_submission']=False
        report['matrix_ids_within_original_submission']=True
        reports.append(report)
        print(filename,low_name,report['triangles'],'triangles',native_kept,'native preserved triangles',removed_tris,'staff triangles removed')
    return reports
