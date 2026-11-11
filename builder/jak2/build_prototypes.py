"""Private Jak II rig prototypes. Outputs are not installable mod payloads.

Read geometry and textures exclusively from a local OpenGOAL extraction.
Keep authored morph targets for fitting to native gameplay channels later.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget_glb import load, retarget


def verify(low_path, high_path, output):
    low, _, la = load(low_path)
    high, _, ha = load(high_path)
    result, _, ra = load(output)
    native_names = [low['nodes'][i]['name'] for i in low['skins'][0]['joints']]
    result_names = [result['nodes'][i]['name'] for i in result['skins'][0]['joints']]
    assert native_names == result_names, 'Gameplay joint order changed'
    assert np.array_equal(la(low['skins'][0]['inverseBindMatrices']),
                          ra(result['skins'][0]['inverseBindMatrices']))
    originals = high['meshes'][0]['primitives']
    rebuilt = result['meshes'][0]['primitives']
    assert len(originals) == len(rebuilt)
    triangles = vertices = 0
    for source, dest in zip(originals, rebuilt):
        used, inverse = np.unique(ha(source['indices']).ravel(), return_inverse=True)
        assert np.array_equal(inverse, ra(dest['indices']).ravel())
        for attribute in ('POSITION', 'TEXCOORD_0'):
            assert np.array_equal(ha(source['attributes'][attribute])[used],
                                  ra(dest['attributes'][attribute])), attribute
        assert len(source.get('targets', [])) == len(dest.get('targets', []))
        for st, dt in zip(source.get('targets', []), dest.get('targets', [])):
            for attribute in st:
                assert np.array_equal(ha(st[attribute])[used], ra(dt[attribute]))
        weights = ra(dest['attributes']['WEIGHTS_0'])
        joints = ra(dest['attributes']['JOINTS_0'])
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert np.allclose(weights.sum(axis=1), 1, atol=1e-6)
        assert ((weights > 0).sum(axis=1) <= 3).all()
        assert joints.max() < len(native_names)
        triangles += len(inverse) // 3
        vertices += len(used)
    return dict(output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                drawVertices=vertices, triangles=triangles, nativeJoints=len(native_names),
                authoredMorphTargets=max(len(p.get('targets', [])) for p in rebuilt),
                exactGeometryUVTopologyAndMorphs=True, nativeJointOrderAndBindMatrices=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.environ['CINEMATIC_BUILD_WORK'] = str(args.output.resolve())
    models = args.data / 'decompiler_out/jak2/levels'
    fr3 = args.data / 'out/jak2/fr3'
    jobs = [
        ('jakb-lod0', 'hiphog/jak-highres-lod0', 'hiphog',
         {'packStrapTop': 'upper_body', 'pantsLthigh': 'hips'}),
        ('daxter-lod0', 'hiphog/daxter-highres-lod0', 'hiphog', {}),
    ]
    results = []
    for native, cinematic, donor, aliases in jobs:
        low = models / f'common/{native}.glb'
        high = models / f'{cinematic}.glb'
        output = args.output / f'{native}.glb'
        retarget(low, high, output, donor_path=fr3 / f'{donor}.fr3',
                 joint_aliases=aliases, preserve_morph_targets=True)
        results.append(verify(low, high, output) | {'provisionalClothingAliases': aliases})
    report = dict(status='Private rig prototypes, not gameplay-ready', subjects=results,
                  remaining=['Fit authored facial shapes to native gameplay channels',
                             'Restore native eye compositor and cinematic material effects',
                             'Validate clothing attachment through gameplay motion',
                             'Test dark and prison forms, EU and USA installation',
                             'Capture matched in-game comparisons'])
    (args.output / 'prototype-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
