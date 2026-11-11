"""Static regression checks for staged Jak II Sig and Ashelin replacements."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / 'stages'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    os.environ['CINEMATIC_BUILD_WORK'] = str(a.work.resolve())
    from fr3_models import decompress, model, DRAW, serialize
    from inspect_faces import buffers
    from inspect_transformations import textures
    from npc_builder import active_joints, texture_bytes, used_indices
    from build_npcs import native_joint_limit
    subjects = {
        'sig-lod0': ('sig-highres-lod0', 10750, [208], ['atollext.fr3', 'underb.fr3']),
        'ashelin-lod0': ('ashelin-highres-lod0', 10223, [200, 201], ['atollext.fr3', 'ctyasha.fr3']),
    }
    inventory = json.loads((a.work / 'variant-audit.json').read_text())['modelInventory']
    reports = []
    for filename in sorted({f for _, _, _, files in subjects.values() for f in files}):
        baseline = decompress(a.work / 'import/out/jak2/fr3' / filename)
        output = a.directory / filename
        candidate = decompress(output)
        first = next(name for name, (_, _, _, files) in subjects.items() if filename in files)
        _, ni, _, nv = buffers(baseline, first)
        _, ci, _, cv = buffers(candidate, first)
        assert np.array_equal(ci[:len(ni)], ni)
        assert cv[:len(nv)] == nv
        nt, _ = textures(baseline)
        ct, _ = textures(candidate)
        assert [texture_bytes(t) for t in nt] == [texture_bytes(t) for t in ct[:len(nt)]]
        originals = {name for name, entries in inventory.items() if any(e['file'] == filename for e in entries)}
        replaced = {name for name, (_, _, _, files) in subjects.items() if filename in files}
        preserved = 0
        for name in originals - replaced:
            assert serialize(model(baseline, name)) == serialize(model(candidate, name)), name
            preserved += 1
        for name in replaced:
            native, result = model(baseline, name), model(candidate, name)
            donor, expected, eyes, _ = subjects[name]
            assert len(native['effects']) == len(result['effects'])
            assert all(x['tail'] == y['tail'] for x, y in zip(native['effects'], result['effects']))
            assert result['tail'][8:] == native['tail'][8:]
            assert sum(DRAW.unpack(d)[5] for e in result['effects'] for d in e['draws']) == expected
            actual_eyes = sorted({DRAW.unpack(d)[2] for e in result['effects'] for d in e['draws'] if DRAW.unpack(d)[2] != 255})
            assert actual_eyes == eyes, (name, actual_eyes)
            supported = active_joints(native, ni, nv)
            weighted = active_joints(result, ci, cv)
            assert weighted <= supported, (name, weighted - supported)
            limit = native_joint_limit(a.work / 'import/decompiler_out/jak2/raw_obj', name.removesuffix('-lod0'), name)
            assert max(weighted) <= limit
            ids = used_indices(result, ci)
            values = np.frombuffer(cv, '<f4').reshape(-1, 16)[ids]
            assert np.isfinite(values).all()
            assert np.allclose(values[:, 8:11].sum(1), 1, atol=1e-5)
            assert np.allclose(np.linalg.norm(values[:, 4:7], axis=1), 1, atol=1e-5)
            reports.append(dict(file=filename, model=name, donor=donor, triangles=expected,
                eyeIds=actual_eyes, activeJointIds=sorted(weighted),
                nativeEffectControlsAndTailsPreserved=True, originalVertexAndIndexBuffersPreserved=True,
                originalTexturesPreserved=True, otherIndexedModelsPreserved=preserved,
                noNewJointEvaluationDependencies=True, nativeEvaluatedMaxJoint=limit,
                outputSha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                status='Static regression passed; live blink, gaze, grip, material and motion checks pending'))
    report = dict(subjects=reports, activeRuntimeUntouched=True)
    (a.directory / 'npc-regression.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([dict(file=r['file'], model=r['model'], triangles=r['triangles'], status=r['status']) for r in reports], indent=2))


if __name__ == '__main__':
    main()
