"""Attach native Jak II gameplay morph channels and live eye surfaces."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--work',required=True,type=Path)
    args=parser.parse_args()
    os.environ['CINEMATIC_BUILD_WORK']=str(args.work.resolve())
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stages'))
    from fix_faces import closest_surface,fix_model,vec
    from inspect_faces import glb,buffers
    from fr3_models import decompress,compress,serialize,model,DRAW
    from eye_alignment import calibrated_eye_uvs,write_eye_uvs
    models=args.data/'decompiler_out/jak2/levels'
    raw=decompress(args.work/'GAME-materials.fr3')
    start,indices,_,original_vertices=buffers(raw,'jakb-lod0')
    vertices=bytearray(original_vertices);new_indices=indices;updates=[];reports=[]
    for name,high,transfer in [
        ('jakb-lod0','jak-highres-lod0',{3:[5],4:[6],8:[9,11],24:[9,11]}),
        ('daxter-lod0','daxter-highres-lod0',{}),
    ]:
        ld,la=glb(models/f'common/{name}.glb')
        hd,ha=glb(args.work/f'rig-prototypes/{name}.glb')
        lp,hp=ld['meshes'][0]['primitives'],hd['meshes'][0]['primitives']
        count=len(lp[0]['targets'])
        coefficients=np.load(args.work/f'face-refined/{name}-face-coefficients.npy')
        deformations={}
        for pi,p in enumerate(hp):
            dp=np.einsum('ct,tnj->cnj',coefficients,np.array([ha(t['POSITION']) for t in p['targets']]))
            dn=np.einsum('ct,tnj->cnj',coefficients,np.array([ha(t['NORMAL']) for t in p['targets']]))
            if pi in transfer:
                triangles=[];delta=[];normal=[]
                for si in transfer[pi]:
                    source=lp[si];ids=la(source['indices']).reshape(-1,3)
                    triangles.append(la(source['attributes']['POSITION'])[ids])
                    delta.append(np.array([la(t['POSITION'])[ids] for t in source['targets']]))
                    normal.append(np.array([la(t['NORMAL'])[ids] for t in source['targets']]))
                ti,bary,_=closest_surface(ha(p['attributes']['POSITION']),np.concatenate(triangles))
                # Jak II channels 0..3 control fingers and hair, independently of expression.
                dp[:4]=np.einsum('tnij,ni->tnj',np.concatenate(delta,axis=1)[:4,ti],bary)
                dn[:4]=np.einsum('tnij,ni->tnj',np.concatenate(normal,axis=1)[:4,ti],bary)
            if np.max(np.abs(dp))>1e-7:
                deformations[pi]=(dp,dn)
        def eyes(_):
            out={}
            for pi,p in enumerate(hp):
                mat=hd['materials'][p['material']]
                ti=mat.get('pbrMetallicRoughness',{}).get('baseColorTexture',{}).get('index')
                if ti is not None:
                    image=hd['images'][hd['textures'][ti]['source']]['name']
                    if image in ('programmer_eye_left','programmer_eye_right'):
                        out[pi]=int(image.endswith('right'))
            assert len(out)==2
            return out
        high_eyes=eyes(high)
        for eye in high_eyes:deformations.pop(eye,None)
        def adapter(*_):return hd,ha,deformations,count,[]
        low_eyes={0:3,1:2} if name.startswith('jak') else {0:2,1:1}
        eye_uv,eye_report=calibrated_eye_uvs(lp,la,hp,ha,high_eyes,low_eyes,name)
        changed_eye_vertices=write_eye_uvs(raw,name,hd,ha,new_indices,vertices,eye_uv)
        native_record=model(decompress(args.data/'out/jak2/fr3/GAME.fr3'),name)
        native_eye_ids=sorted({DRAW.unpack(d)[2] for e in native_record['effects'] for d in e['draws'] if DRAW.unpack(d)[2]!=255})
        assert len(native_eye_ids)==2 and native_eye_ids[0]%2==0 and native_eye_ids[1]==native_eye_ids[0]+1
        target,added,report=fix_model(raw,name,high,{},new_indices,vertices,
                                     retarget_fn=adapter,eye_fn=eyes,
                                     eye_slot=native_eye_ids[0]//2)
        report['native_eye_ids']=native_eye_ids
        report['eye_alignment']=eye_report
        report['eye_alignment']['uvVerticesWritten']=len(changed_eye_vertices)
        target['tail']=target['tail'][:8]+native_record['tail'][8:]
        new_indices=np.concatenate([new_indices]+added);updates.append(target);reports.append(report)
    fixed=raw[:start]+vec(new_indices.astype('<u4').tobytes(),4)+vec(vertices,64)+raw[-2:]
    for target in sorted(updates,key=lambda x:x['start'],reverse=True):
        fixed=fixed[:target['start']]+serialize(target)+fixed[target['end']:]
    _,check_indices,_,check_vertices=buffers(fixed,'jakb-lod0')
    assert np.array_equal(check_indices[:len(indices)],indices)
    before=np.frombuffer(original_vertices,'u1').reshape(-1,64)
    after=np.frombuffer(check_vertices,'u1').reshape(-1,64)
    assert np.array_equal(before[:,:48],after[:,:48]) and np.array_equal(before[:,56:],after[:,56:])
    for target in updates:assert serialize(model(fixed,target['name']))==serialize(target)
    compress(args.work/'GAME-faces.fr3',fixed)
    (args.work/'native-faces-report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
