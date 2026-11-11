"""Verify unrelated native models survive selective cinematic NPC extraction."""
import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import npc_materials as io


def texture_equal(a,b):
    return all(a[k]==b[k] for k in ('name','width','height','pixels'))


def compare_draws(old,new,oi,ni,ov,nv,ot,nt,*,local=False):
    assert len(old)==len(new)
    for packed_a,packed_b in zip(old,new):
        a,b=io.DRAW.unpack(packed_a),io.DRAW.unpack(packed_b)
        assert (a[0],a[2],a[4:])==(b[0],b[2],b[4:])
        if a[1]<0 or b[1]<0:
            assert a[1]==b[1]
        else:
            assert texture_equal(ot[a[1]],nt[b[1]])
        ax,bx=oi[a[3]:a[3]+a[4]],ni[b[3]:b[3]+b[4]]
        assert np.array_equal(ax==0xffffffff,bx==0xffffffff)
        valid=ax!=0xffffffff
        assert np.array_equal(ov[ax[valid]],nv[bx[valid]])


def payload(effect):
    r=io.Reader(effect['mod'])
    mod,fix=r.draws(),r.draws()
    vertices=r.vector(64)
    remainder=r.take(len(effect['mod'])-r.pos)
    return mod,fix,np.frombuffer(vertices,'u1').reshape(-1,64),remainder


def review(work,directory):
    io.configure(work)
    exclusions={'atollext.fr3':{'sig-lod0','ashelin-lod0'},
        'underb.fr3':{'sig-lod0'},'ctyasha.fr3':{'ashelin-lod0'},
        'stadblmp.fr3':{'samos-lod0','keira-lod0'}}
    rows=[]
    for filename,replaced in exclusions.items():
        old=io.decompress(work/'import/out/jak2/fr3'/filename)
        new=io.decompress(directory/filename)
        ot,_=io.textures(old);nt,_=io.textures(new)
        _,oi,_,ov=io.buffers(old,next(iter(replaced)))
        _,ni,_,nv=io.buffers(new,next(iter(replaced)))
        ov=np.frombuffer(ov,'u1').reshape(-1,64)
        nv=np.frombuffer(nv,'u1').reshape(-1,64)
        names=sorted({m.group().decode() for m in re.finditer(rb'[a-zA-Z0-9+_.-]+-lod[0-9]',old)}-replaced)
        checked=[]
        for name in names:
            try:
                a,b=io.model(old,name),io.model(new,name)
            except AssertionError:
                continue
            assert a['tail']==b['tail'],(filename,name,'model decode fields')
            assert len(a['effects'])==len(b['effects'])
            for ae,be in zip(a['effects'],b['effects']):
                assert ae['tail'][:4]+ae['tail'][8:]==be['tail'][:4]+be['tail'][8:]
                if ae['tail'][8]:
                    ar,br=struct.unpack_from('<I',ae['tail'],4)[0],struct.unpack_from('<I',be['tail'],4)[0]
                    assert texture_equal(ot[ar],nt[br])
                compare_draws(ae['draws'],be['draws'],oi,ni,ov,nv,ot,nt)
                am,af,av,ap=payload(ae);bm,bf,bv,bp=payload(be)
                assert np.array_equal(av,bv) and ap==bp
                compare_draws(af,bf,oi,ni,ov,nv,ot,nt)
                compare_draws(am,bm,oi,ni,av,bv,ot,nt,local=True)
            checked.append(name)
        rows.append(dict(file=filename,unchangedModels=checked,count=len(checked),
            nativeGeometrySkinUVsColoursMorphsAndMaterialsExact=True,
            offsetsMayDifferAfterSelectiveImport=True))
    report=dict(subjects=rows,totalUnchangedModels=sum(r['count'] for r in rows))
    (directory/'native-context-regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',required=True,type=Path)
    p.add_argument('--directory',required=True,type=Path)
    a=p.parse_args();review(a.work,a.directory)
