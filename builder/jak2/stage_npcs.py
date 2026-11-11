"""Assemble private Jak II NPC runtime candidates and live eye resources."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','work','geometry','faces','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    output=a.output.resolve()
    if not output.is_relative_to(a.work.resolve()) or output.is_relative_to((a.work/'import').resolve()):
        raise ValueError('Staging output must remain private and outside the active project')
    if output.exists():
        raise FileExistsError('Choose a fresh staging directory')
    output.mkdir(parents=True)
    for name in ('atollext.fr3','underb.fr3','ctyasha.fr3'):
        shutil.copy2(a.geometry/name,output/name)
    shutil.copy2(a.faces/'stadblmp.fr3',output/'stadblmp.fr3')
    os.environ['CINEMATIC_BUILD_WORK']=str(a.work.resolve())
    from cinematic_eyes import generate
    generate(a.data,a.work,output,output/'cinematic-eyes.gc',None)
    # These generated sources are explicitly staged. The caller decides when
    # to replace the private game's files and compile its GAME archive.
    from patch_runtime import patch
    foreground=Path('goal_src/jak2/engine/gfx/foreground/foreground.gc')
    (output/'foreground-cinematic.gc').write_text(patch((a.data/foreground).read_text()))
    from transformations import patch_texture_animation
    texture=Path('goal_src/jak2/engine/gfx/texture/texture-anim.gc')
    (output/'texture-anim-cinematic.gc').write_text(patch_texture_animation((a.data/texture).read_text()))
    (output/'npc-assembly.json').write_text(json.dumps(dict(
        files=['atollext.fr3','underb.fr3','ctyasha.fr3','stadblmp.fr3'],
        nativeControlsRequired=True,facesInput='stadblmp.fr3',
        samosMorphBankStatus='Original bank is anomalous; no unsafe cinematic deformation attached',
        status='Private candidate; requires compilation and in-game validation'),indent=2)+'\n')
    snapshot=dict(latestKeiraFaceInput=hashlib.sha256((a.faces/'stadblmp.fr3').read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.iterdir()) if p.suffix in ('.fr3','.gc')})
    (output/'staging-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    print('Assembled four NPC containers and generated isolated runtime sources')


if __name__=='__main__':
    main()
