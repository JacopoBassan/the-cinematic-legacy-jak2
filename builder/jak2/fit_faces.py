"""Fit Jak II gameplay channels to authored cinematic shapes for review.

This produces private numeric candidates, not an installable facial patch.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
from retarget_glb import load


def fit(low_path, high_path, parts, channels, landmarks=()):
    from fix_faces import closest_surface
    low, _, la = load(low_path)
    high, _, ha = load(high_path)
    lp, hp = low['meshes'][0]['primitives'], high['meshes'][0]['primitives']
    native_count = len(lp[0]['targets'])
    cinematic_count = len(hp[0]['targets'])
    matrices, targets, correspondence = [], [], []
    for high_part, low_parts in parts.items():
        p = hp[high_part]
        used = np.unique(ha(p['indices']).ravel())
        positions = ha(p['attributes']['POSITION'])[used]
        basis = np.array([ha(t['POSITION'])[used] for t in p['targets']])
        moving = np.linalg.norm(basis, axis=2).max(axis=0) > 1e-6
        triangles, deltas = [], []
        for low_part in low_parts:
            q = lp[low_part]
            ids = la(q['indices']).reshape(-1, 3)
            triangles.append(la(q['attributes']['POSITION'])[ids])
            deltas.append(np.array([la(t['POSITION'])[ids] for t in q['targets']]))
        ti, bary, distance = closest_surface(positions, np.concatenate(triangles))
        desired = np.einsum('tnij,ni->tnj', np.concatenate(deltas, axis=1)[:, ti], bary)
        matrices.append(basis[:, moving].reshape(cinematic_count, -1).T)
        targets.append(desired[:, moving].reshape(native_count, -1).T)
        correspondence.append(dict(cinematicPart=high_part, gameplayParts=low_parts,
                                   fittedVertices=int(moving.sum()),
                                   maxDistance=float(distance[moving].max())))
    A, B = np.concatenate(matrices), np.concatenate(targets)
    coefficient = np.zeros((native_count, cinematic_count))
    errors = []
    regularizer = 0.025
    landmark_a, landmark_b = [], []
    for low_part, low_id, high_part, high_id in landmarks:
        source, dest = lp[low_part], hp[high_part]
        assert low_id in la(source['indices']) and high_id in ha(dest['indices'])
        assert abs(float(la(source['attributes']['POSITION'])[low_id,0])) < 1e-5
        assert abs(float(ha(dest['attributes']['POSITION'])[high_id,0])) < 1e-5
        landmark_a.append(np.array([ha(t['POSITION'])[high_id] for t in dest['targets']]).T)
        landmark_b.append(np.array([la(t['POSITION'])[low_id] for t in source['targets']]).T)
    LA = np.concatenate(landmark_a) if landmarks else np.zeros((0, cinematic_count))
    LB = np.concatenate(landmark_b) if landmarks else np.zeros((0, native_count))
    for channel in channels:
        desired = B[:, channel]
        initial = lsq_linear(np.vstack([A, np.eye(cinematic_count)*regularizer]),
                             np.r_[desired, np.zeros(cinematic_count)], bounds=(0,1.5), tol=1e-9)
        result = lsq_linear(np.vstack([A, np.eye(cinematic_count)*regularizer, LA*32]),
                            np.r_[desired, np.zeros(cinematic_count), LB[:,channel]*32],
                            bounds=(0, 1.5), tol=1e-9)
        if not result.success:
            raise RuntimeError(f'Facial fit failed for channel {channel}: {result.message}')
        coefficient[channel] = result.x
        actual = A @ result.x
        errors.append(dict(channel=channel,
                           mouthErrorBefore=float(np.linalg.norm(LA@initial.x-LB[:,channel])),
                           mouthErrorAfter=float(np.linalg.norm(LA@result.x-LB[:,channel])),
                           relativeSurfaceFitError=float(np.linalg.norm(actual-desired)/max(np.linalg.norm(desired),1e-12)),
                           nativeMaxDisplacement=float(np.linalg.norm(desired.reshape(-1,3),axis=1).max()),
                           candidateMaxDisplacement=float(np.linalg.norm(actual.reshape(-1,3),axis=1).max())))
    assert np.isfinite(coefficient).all()
    return coefficient, dict(gameplayChannels=native_count, cinematicShapes=cinematic_count,
                             facialChannels=channels, correspondence=correspondence, errors=errors,
                             mouthLandmarks=landmarks,
                             status='Numerical candidate; mouth, symmetry and motion require visual validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.environ['CINEMATIC_BUILD_WORK'] = str(args.output.resolve())
    sys.path.insert(0, str(RELEASE/'stages'))
    models = args.data/'decompiler_out/jak2/levels'
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, high, parts, channels, landmarks in [
        ('jakb-lod0', 'jak-highres-lod0', {8:[9], 10:[4,8], 28:[8,9]}, list(range(4,12)),
         [(9,4217,8,11292),(9,4240,8,11341)]),
        ('daxter-lod0', 'daxter-highres-lod0', {5:[4,5,6], 6:[4,5,6,7]}, list(range(9)),
         [(5,6281,6,2387),(6,6327,6,1703),(6,6349,6,1755)]),
    ]:
        coefficient, metrics = fit(models/f'common/{name}.glb',
                                   models/f'hiphog/{high}.glb', parts, channels, landmarks)
        np.save(args.output/f'{name}-face-coefficients.npy', coefficient)
        report[name] = metrics
    (args.output/'face-fit-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
