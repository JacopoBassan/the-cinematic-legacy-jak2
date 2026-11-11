"""Read-only Jak II transformation and native model inventory from local data."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'stages'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    os.environ['CINEMATIC_BUILD_WORK'] = str(a.work.resolve())
    from fr3_models import decompress, model, DRAW
    from inspect_transformations import textures
    from retarget_glb import load
    rows = []
    for archive, name in [('GAME', 'jakb-lod0'), ('hiphog', 'jak-highres-lod0'),
                          ('oracle', 'darkjak-highres-lod0'),
                          ('ldjakbrn', 'jak-highres-prison-lod0')]:
        raw = decompress(a.data / f'out/jak2/fr3/{archive}.fr3')
        m = model(raw, name)
        tex, _ = textures(raw)
        d, _, acc = load(a.data / f'decompiler_out/jak2/levels/{"common" if archive == "GAME" else archive}/{name}.glb')
        rows.append(dict(archive=archive, name=name, joints=[d['nodes'][j]['name'] for j in d['skins'][0]['joints']],
                         effects=[dict(triangles=sum(DRAW.unpack(x)[5] for x in e['draws']),
                                       draws=[dict(textureIndex=v[1], texture=tex[v[1]]['name'] if v[1] >= 0 else None,
                                                   triangles=v[5], eye=v[2]) for v in map(DRAW.unpack, e['draws'])]) for e in m['effects']],
                         textures=[dict(name=t['name'], width=t['width'], height=t['height'])
                                   for t in tex if 'jakb' in t['name'] or 'jak-orig' in t['name']]))
    names = {}
    for file in sorted((a.work / 'import/out/jak2/fr3').glob('*.fr3')):
        raw = decompress(file)
        for match in re.finditer(rb'([a-zA-Z0-9_-]+-lod[0-9])', raw):
            name = match.group().decode()
            if any(k in name for k in ('highres', 'sig', 'ashelin', 'guard', 'torn', 'samos', 'keira', 'kor', 'kid', 'baron', 'errol', 'tess', 'transport')):
                try:
                    m = model(raw, name)
                except (AssertionError, ValueError):
                    continue
                names.setdefault(name, []).append(dict(file=file.name, effects=len(m['effects']),
                    triangles=sum(DRAW.unpack(x)[5] for e in m['effects'] for x in e['draws'])))
    report = dict(variants=rows, modelInventory=names)
    (a.work / 'variant-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'variants': [{k:v for k,v in row.items() if k != 'textures'} for row in rows], 'models': names}, indent=2))


if __name__ == '__main__':
    main()
