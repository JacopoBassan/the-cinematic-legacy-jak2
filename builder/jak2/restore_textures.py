"""Restore PS2 alpha and isolate custom texture names from animated global names."""
import argparse,json,os,struct,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Private alternate output for staged validation')
    a=p.parse_args();os.environ['CINEMATIC_BUILD_WORK']=str(a.work.resolve())
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stages'))
    from fr3_models import Reader,decompress,compress
    from inspect_transformations import textures
    raw=decompress(a.work/'GAME-faces.fr3');target,end=textures(raw)
    donors=[]
    for name in ['GAME','hiphog']:donors+=textures(decompress(a.data/f'out/jak2/fr3/{name}.fr3'))[0]
    restored=[]
    for t in target:
        if t['page']!='custom-level':continue
        candidates=[s for s in donors if (s['name'],s['width'],s['height'])==(t['name'],t['width'],t['height'])
                    and np.array_equal(np.frombuffer(s['pixels'],'u1').reshape(-1,4)[:,:3],np.frombuffer(t['pixels'],'u1').reshape(-1,4)[:,:3])]
        variants={bytes(s['pixels']) for s in candidates}
        if len(variants)==1:t['pixels']=next(iter(variants))
        elif t['name']!='white':raise ValueError(f'Missing or ambiguous native texture: {t["name"]}')
        restored.append(t['name']);t['name']='j2-cinematic/'+t['name']
    r=Reader(raw,2);r.take(r.num('Q'));out=bytearray(raw[:r.pos])+struct.pack('<Q',len(target))
    for t in target:
        out+=struct.pack('<HHI',t['width'],t['height'],t['combo'])+struct.pack('<Q',len(t['pixels'])//4)+t['pixels']
        for key in ['name','page']:
            encoded=t[key].encode();out+=struct.pack('<Q',len(encoded))+encoded
        out+=t['flag']
    out+=raw[end:]
    assert textures(bytes(out))[0]==target
    from transformations import connect
    fixed,transformation=connect(bytes(out),decompress(a.data/'out/jak2/fr3/hiphog.fr3'),a.work/'rig-prototypes/jakb-lod0.glb')
    from native_horns import retain_native_horns
    fixed,horns=retain_native_horns(fixed,decompress(a.data/'out/jak2/fr3/GAME.fr3'),a.work/'rig-prototypes/jakb-lod0.glb')
    output=a.output or a.work/'GAME-cinematic.fr3'
    compress(output,fixed)
    assert decompress(output)==fixed
    report=json.dumps({'restored':restored,'globalNamesPreserved':True,'transformation':transformation,'hornCompatibility':horns},indent=2)+'\n'
    output.with_suffix('.textures.json').write_text(report)
    if a.output is None:(a.work/'texture-report.json').write_text(report)
    print('Restored',len(restored),'cinematic textures and isolated custom names')

if __name__=='__main__':main()
