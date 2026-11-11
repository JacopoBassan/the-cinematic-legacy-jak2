"""Restore authored Jak II cinematic effect boundaries after GLB import.

This private stage restores rendering metadata only. Runtime visibility control
mapping and facial buffers must be completed before installing its output.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Imported player FR3, separate from the original extraction')
    args = parser.parse_args()
    os.environ['CINEMATIC_BUILD_WORK'] = str(args.work.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'stages'))
    from dax_effects import restore
    from fr3_models import DRAW, compress, decompress, model, serialize
    raw = decompress(args.input or args.work/'import/out/jak2/fr3/GAME.fr3')
    donor = decompress(args.data/'out/jak2/fr3/hiphog.fr3')
    baseline = decompress(args.data/'out/jak2/fr3/GAME.fr3')
    rows = []
    for name, source, counts in [
        ('jakb-lod0', 'jak-highres-lod0', [192,216,540,628,1402,7812,96]),
        ('daxter-lod0', 'daxter-highres-lod0', [312,40,692,64,7050,336,1016]),
    ]:
        raw = restore(raw, donor, (args.work/f'rig-prototypes/{name}.glb').read_bytes(),
                      name=name, donor_name=source, expected_triangles=counts)
        target, original = model(raw, name), model(donor, source)
        target['tail'] = target['tail'][:8] + model(baseline, name)['tail'][8:]
        for te, oe in zip(target['effects'], original['effects']):
            for index, (td, od) in enumerate(zip(te['draws'], oe['draws'])):
                values, authored = list(DRAW.unpack(td)), DRAW.unpack(od)
                # Importer defaults do not preserve every original blend/depth mode.
                values[0] = authored[0]
                te['draws'][index] = DRAW.pack(*values)
        raw = raw[:target['start']] + serialize(target) + raw[target['end']:]
        rows.append(dict(model=name, effects=len(counts), triangles=sum(counts),
                         authoredDrawAndReflectionModes=True, geometryUnchanged=True))
    output = args.work/'GAME-materials.fr3'
    compress(output, raw)
    assert decompress(output) == raw
    report = dict(subjects=rows, status='Private render metadata candidate; runtime controls and faces pending')
    (args.work/'material-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
