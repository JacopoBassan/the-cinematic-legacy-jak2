"""Check NPC render metadata, source pixels and untouched geometry independently."""
import argparse
import hashlib
import json
from pathlib import Path

import npc_materials as io


def review(data, work, directory, baseline):
    io.configure(work)
    reports = json.loads((directory/'npc-materials.json').read_text())
    inventory = json.loads((work/'variant-audit.json').read_text())['modelInventory']
    files = sorted({r['file'] for r in reports})
    rows = []
    for filename in files:
        original = io.decompress(baseline/filename)
        current = io.decompress(directory/filename)
        changed = {r['model'] for r in reports if r['file']==filename}
        _, oi, _, ov = io.buffers(original, next(iter(changed)))
        _, ni, _, nv = io.buffers(current, next(iter(changed)))
        assert oi.tobytes()==ni.tobytes() and ov==nv
        old_textures, _ = io.textures(original)
        new_textures, _ = io.textures(current)
        assert new_textures[:len(old_textures)]==old_textures
        other = 0
        for name, entries in inventory.items():
            if name not in changed and any(e['file']==filename for e in entries):
                assert io.serialize(io.model(original,name))==io.serialize(io.model(current,name)),name
                other += 1
        for row in [r for r in reports if r['file']==filename]:
            donor_file = {'sig-lod0':'atollext.fr3','ashelin-lod0':'atollext.fr3',
                'samos-lod0':'lintcstb.fr3','keira-lod0':'lintcstb.fr3'}[row['model']]
            donor_path = data/'out/jak2/fr3'/donor_file
            if not donor_path.is_file():
                donor_path = work/'import/out/jak2/fr3'/donor_file
            donor_raw = io.decompress(donor_path)
            donor = io.model(donor_raw, row['donor'])
            dt, _ = io.textures(donor_raw)
            source = [(ei,e,io.DRAW.unpack(d)) for ei,e in enumerate(donor['effects']) for d in e['draws']]
            current_model = io.model(current,row['model'])
            actual = sorted([(ei,e,io.DRAW.unpack(d)) for ei,e in enumerate(current_model['effects']) for d in e['draws']],key=lambda x:x[2][3])
            assert len(actual)==len(row['primitiveProvenance'])
            for manifest,(ei,e,d) in zip(row['primitiveProvenance'],actual):
                si,se,sd = source[manifest['sourcePrimitive']]
                assert ei==manifest['effect'] and d[0]==sd[0] and d[2]==sd[2] and d[5]==sd[5]
                assert e['tail'][:4]==se['tail'][:4] and e['tail'][8]==se['tail'][8]
                nt, st = new_textures[d[1]], dt[sd[1]]
                assert (nt['width'],nt['height'],nt['pixels'])==(st['width'],st['height'],st['pixels'])
                if e['tail'][8]:
                    import struct
                    nr=new_textures[struct.unpack_from('<I',e['tail'],4)[0]]
                    sr=dt[struct.unpack_from('<I',se['tail'],4)[0]]
                    assert (nr['width'],nr['height'],nr['pixels'])==(sr['width'],sr['height'],sr['pixels'])
            native=io.model(io.decompress(work/'import/out/jak2/fr3'/filename),row['model'])
            assert current_model['tail'][8:]==native['tail'][8:]
            rows.append(dict(file=filename,model=row['model'],triangles=sum(d[2][5] for d in actual),
                effects=len(current_model['effects']),fullGeometryExact=True,sourceDrawModesAndPixelsExact=True,
                nativeModelTailRestored=True,otherModelsExact=other,
                sha256=hashlib.sha256((directory/filename).read_bytes()).hexdigest()))
    report=dict(subjects=rows,status='Static material checks passed; live eye, expression, prop and animation tests remain required')
    (directory/'npc-material-regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','work','directory','baseline'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    review(a.data,a.work,a.directory,a.baseline)
