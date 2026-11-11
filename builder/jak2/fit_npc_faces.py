"""Fit private Jak II NPC facial candidates from native and cinematic data.

The correspondence is by anatomical surface and native channel displacement,
never by assuming that native and cinematic channel numbers mean the same thing.
Samos's original bank is retained as a diagnostic until its runtime semantics
are established. It is deliberately not written to a usable coefficient file.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
from retarget_glb import load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topology_components(primitive, accessor):
    """Keep coincident upper/lower mouth contact vertices separate."""
    triangles = accessor(primitive['indices']).reshape(-1, 3)
    used, compact = np.unique(triangles, return_inverse=True)
    triangles = compact.reshape(-1, 3)
    left, right = triangles.ravel(), triangles[:, [1, 2, 0]].ravel()
    graph = coo_matrix((np.ones(len(left)), (left, right)), shape=(len(used), len(used)))
    count, labels = connected_components(graph, directed=False)
    position = accessor(primitive['attributes']['POSITION'])
    groups = [used[labels == i] for i in range(count)]
    return sorted(groups, key=lambda ids: float(position[ids, 1].mean()))


def basis_at(primitive, accessor, ids):
    return np.array([accessor(t['POSITION'])[ids].mean(axis=0) for t in primitive['targets']]).T


def keira_constraints(lp, la, hp, ha):
    """Use explicit upper/lower lip and topologically separate tooth surfaces."""
    rows, desired, notes = [], [], []
    # These contact loops are distinct even where the quantized native mouth is
    # closed. Channel 10 moves the lower loop, while channel 9 moves the upper.
    landmarks = [
        ('lower lip contact', 32317, 700, [0, 2.256417, .333634], [0, 2.255288, .340229]),
        ('upper lip contact', 32222, 750, [0, 2.256417, .342413], [0, 2.259363, .354490]),
    ]
    for label, lo, hi, expected_lo, expected_hi in landmarks:
        assert lo in la(lp[5]['indices']) and hi in ha(hp[7]['indices'])
        assert np.allclose(la(lp[5]['attributes']['POSITION'])[lo], expected_lo, atol=2e-6)
        assert np.allclose(ha(hp[7]['attributes']['POSITION'])[hi], expected_hi, atol=2e-6)
        rows.append(basis_at(hp[7], ha, [hi]))
        desired.append(basis_at(lp[5], la, [lo]))
        notes.append(dict(label=label, nativePrimitive=5, nativeVertices=[lo],
                          cinematicPrimitive=7, cinematicVertices=[hi]))
    native = topology_components(lp[7], la)
    cinematic = topology_components(hp[8], ha)
    assert len(native) == len(cinematic) == 2, 'Unexpected upper/lower teeth topology'
    for label, lo, hi in zip(('lower teeth', 'upper teeth'), native, cinematic):
        rows.append(basis_at(hp[8], ha, hi))
        desired.append(basis_at(lp[7], la, lo))
        notes.append(dict(label=label, nativePrimitive=7, nativeVertices=lo.tolist(),
                          cinematicPrimitive=8, cinematicVertices=hi.tolist()))
    return np.concatenate(rows), np.concatenate(desired), notes


def keira_motion_regions(lp, la, hp, ha):
    """Separate mouth and upper-face motions by their measured spatial energy."""
    fractions = []
    for primitive, accessor in ((lp[5], la), (hp[7], ha)):
        used = np.unique(accessor(primitive['indices']))
        position = accessor(primitive['attributes']['POSITION'])[used]
        delta = np.array([accessor(t['POSITION'])[used] for t in primitive['targets']])
        energy = np.sum(delta * delta, axis=2)
        fractions.append(energy[:, position[:, 1] < 2.31].sum(axis=1) /
                         np.maximum(energy.sum(axis=1), 1e-20))
    native, cinematic = fractions
    # These extracted banks have two distinct anatomical clusters. Keep the
    # measurement explicit so a changed source cannot silently use this split.
    assert np.all((native < .01) | (native > .97))
    assert np.all((cinematic < .2) | (cinematic > .97))
    allowed = [np.flatnonzero((cinematic > .5) == (value > .5)) for value in native]
    return allowed, dict(boundaryHeightMetres=2.31, nativeMouthEnergyFraction=native.tolist(),
                         cinematicMouthEnergyFraction=cinematic.tolist(),
                         criterion='Same anatomical motion region, derived from displacement energy',
                         nativeChannelNumbersAssumedEquivalent=False)


def fit_model(low_path, high_path, parts, *, name):
    from fix_faces import closest_surface
    low, _, la = load(low_path)
    high, _, ha = load(high_path)
    lp, hp = low['meshes'][0]['primitives'], high['meshes'][0]['primitives']
    native_count, cinematic_count = len(lp[0]['targets']), len(hp[0]['targets'])
    matrices, targets, correspondence = [], [], []
    for high_part, low_parts in parts.items():
        primitive = hp[high_part]
        used = np.unique(ha(primitive['indices']))
        positions = ha(primitive['attributes']['POSITION'])[used]
        basis = np.array([ha(t['POSITION'])[used] for t in primitive['targets']])
        moving = np.linalg.norm(basis, axis=2).max(axis=0) > 1e-6
        triangles, deltas = [], []
        for low_part in low_parts:
            source = lp[low_part]
            ids = la(source['indices']).reshape(-1, 3)
            triangles.append(la(source['attributes']['POSITION'])[ids])
            deltas.append(np.array([la(t['POSITION'])[ids] for t in source['targets']]))
        ti, bary, distance = closest_surface(positions, np.concatenate(triangles))
        desired = np.einsum('tnij,ni->tnj', np.concatenate(deltas, axis=1)[:, ti], bary)
        matrices.append(basis[:, moving].reshape(cinematic_count, -1).T)
        targets.append(desired[:, moving].reshape(native_count, -1).T)
        correspondence.append(dict(cinematicPrimitive=high_part, nativePrimitives=low_parts,
                                   fittedVertices=int(moving.sum()),
                                   surfaceDistanceMedianMetres=float(np.median(distance[moving])),
                                   surfaceDistanceMaximumMetres=float(distance[moving].max())))
    A, B = np.concatenate(matrices), np.concatenate(targets)
    if name == 'keira-lod0':
        LA, LB, constraints = keira_constraints(lp, la, hp, ha)
        allowed, regions = keira_motion_regions(lp, la, hp, ha)
    else:
        LA, LB, constraints = np.zeros((0, cinematic_count)), np.zeros((0, native_count)), []
        allowed, regions = [np.arange(cinematic_count)] * native_count, None
    coefficients, errors = np.zeros((native_count, cinematic_count)), []
    regularizer, landmark_weight = .025, 32.0
    base = np.vstack((A, np.eye(cinematic_count) * regularizer))
    constrained = np.vstack((base, LA * landmark_weight))
    for channel in range(native_count):
        desired = B[:, channel]
        shapes = allowed[channel]
        initial = lsq_linear(base[:, shapes], np.r_[desired, np.zeros(cinematic_count)],
                             bounds=(0, 1.5), tol=1e-9, method='bvls')
        result = lsq_linear(constrained[:, shapes],
                            np.r_[desired, np.zeros(cinematic_count), LB[:, channel] * landmark_weight],
                            bounds=(0, 1.5), tol=1e-9, method='bvls')
        if not result.success:
            raise RuntimeError(f'{name} channel {channel}: {result.message}')
        coefficients[channel, shapes] = result.x
        actual = A[:, shapes] @ result.x
        residual = (LA[:, shapes] @ result.x - LB[:, channel]).reshape(-1, 3)
        errors.append(dict(channel=channel,
            relativeSurfaceError=float(np.linalg.norm(actual-desired) / max(np.linalg.norm(desired), 1e-12)),
            nativeMaximumMetres=float(np.linalg.norm(desired.reshape(-1, 3), axis=1).max()),
            candidateMaximumMetres=float(np.linalg.norm(actual.reshape(-1, 3), axis=1).max()),
            constraintMaximumBeforeMetres=float(np.linalg.norm((LA[:, shapes]@initial.x-LB[:, channel]).reshape(-1, 3), axis=1).max(initial=0)),
            constraintMaximumAfterMetres=float(np.linalg.norm(residual, axis=1).max(initial=0)),
            constraintHorizontalMaximumMetres=float(np.abs(residual[:, 0]).max(initial=0)),
            authoredCoefficientSum=float(result.x.sum()), allowedAuthoredShapes=shapes.tolist()))
    assert np.isfinite(coefficients).all()
    common = B.mean(axis=1)
    baseline_ratio = float(np.linalg.norm(common) / max(np.linalg.norm(B) / np.sqrt(native_count), 1e-12))
    unsafe = name == 'samos-lod0'
    report = dict(model=name, nativeChannels=native_count, cinematicShapes=cinematic_count,
                  nativeGLBSha256=sha(low_path), cinematicGLBSha256=sha(high_path),
                  correspondence=correspondence, constraints=constraints, errors=errors,
                  anatomicalMotionRegions=regions,
                  regularizer=regularizer, constraintWeight=landmark_weight,
                  sharedMeanToChannelRMSRatio=baseline_ratio,
                  anyNativeDisplacementOverHalfMetre=any(e['nativeMaximumMetres'] > .5 for e in errors),
                  rebasedNativeDeltas=False, nativeChannelNumbersAssumedEquivalent=False,
                  installable=False,
                  status=('Held diagnostic: original Samos bank has verified common large offsets; '
                          'ordinary native clips have no blend tracks; runtime weight semantics pending'
                          if unsafe else 'Numerical candidate; native expressions and mouth contacts require live review'))
    return coefficients, report, (low, la, high, ha)


def deformations(high, ha, coefficients):
    """One authored coefficient vector drives all cinematic material parts."""
    result = {}
    for pi, p in enumerate(high['meshes'][0]['primitives']):
        # Animated eye composition owns the sclera and gaze, not face fitting.
        material = high['materials'][p['material']]['name']
        if 'programmer_eye_' in material:
            continue
        dp = np.einsum('ct,tnj->cnj', coefficients, np.array([ha(t['POSITION']) for t in p['targets']]))
        dn = np.einsum('ct,tnj->cnj', coefficients, np.array([ha(t['NORMAL']) for t in p['targets']]))
        if np.max(np.abs(dp)) > 1e-7 or np.max(np.abs(dn)) > 1e-7:
            result[pi] = (dp, dn)
    return result


def keira_mouth_qa(asset, coefficients):
    low, la, high, ha = asset
    lp, hp = low['meshes'][0]['primitives'], high['meshes'][0]['primitives']
    native, cinematic = lp[5], hp[7]
    native_delta = np.array([la(t['POSITION']) for t in native['targets']])
    cinematic_delta = np.einsum('ct,tnj->cnj', coefficients,
                               np.array([ha(t['POSITION']) for t in cinematic['targets']]))
    native_base = la(native['attributes']['POSITION'])
    cinematic_base = ha(cinematic['attributes']['POSITION'])
    channels = []
    for channel in range(len(coefficients)):
        native_gap = native_base[32222] - native_base[32317]
        cinematic_gap = cinematic_base[750] - cinematic_base[700]
        native_change = native_delta[channel, 32222] - native_delta[channel, 32317]
        cinematic_change = cinematic_delta[channel, 750] - cinematic_delta[channel, 700]
        lip_residual = np.array([cinematic_delta[channel, hi] - native_delta[channel, lo]
                                 for lo, hi in ((32317, 700), (32222, 750))])
        channels.append(dict(channel=channel,
            nativeVerticalLipGapMetres=float((native_gap + native_change)[1]),
            candidateVerticalLipGapMetres=float((cinematic_gap + cinematic_change)[1]),
            verticalOpeningChangeErrorMetres=float(abs(cinematic_change[1] - native_change[1])),
            lipDisplacementMaximumErrorMetres=float(np.linalg.norm(lip_residual, axis=1).max()),
            centerlineHorizontalMotionMaximumMetres=float(np.abs(cinematic_delta[channel, [700, 750], 0]).max())))
    assert max(c['centerlineHorizontalMotionMaximumMetres'] for c in channels) < 1e-5
    return dict(neutralCinematicMeshAltered=False, lipCorrespondence='Separate native upper/lower contact loops',
                forcedLiveReviewChannels=[1, 9, 10, 21, 22], channels=channels)


def stage_keira(input_path, output_path, asset, coefficients):
    """Write only a private alternate FR3; Samos diagnostic bank is excluded."""
    from fix_faces import fix_model, vec
    from inspect_faces import buffers
    from fr3_models import decompress, compress, serialize, model, DRAW
    low, la, high, ha = asset
    raw = decompress(input_path)
    start, indices, _, original_vertices = buffers(raw, 'keira-lod0')
    vertices = bytearray(original_vertices)
    motion = deformations(high, ha, coefficients)
    count = len(low['meshes'][0]['primitives'][0]['targets'])
    def adapter(*_):
        return high, ha, motion, count, [dict(sharedAuthoredBasis=True, directEyeDeformation=False)]
    def eyes(_):
        return {5: 0, 6: 1}
    target, extra, report = fix_model(raw, 'keira-lod0', 'keira-highres-lod0', {}, indices, vertices,
                                     retarget_fn=adapter, eye_fn=eyes, eye_slot=72)
    combined = np.concatenate([indices] + extra)
    fixed = raw[:start] + vec(combined.astype('<u4').tobytes(), 4) + vec(vertices, 64) + raw[-2:]
    fixed = fixed[:target['start']] + serialize(target) + fixed[target['end']:]
    _, checked_indices, _, checked_vertices = buffers(fixed, 'keira-lod0')
    assert checked_vertices == original_vertices, 'Facial staging changed neutral vertices'
    assert np.array_equal(checked_indices[:len(indices)], indices)
    assert serialize(model(fixed, 'keira-lod0')) == serialize(target)
    assert serialize(model(fixed, 'samos-lod0')) == serialize(model(raw, 'samos-lod0'))
    eye_ids = sorted({DRAW.unpack(d)[2] for effect in target['effects'] for d in effect['draws']
                      if DRAW.unpack(d)[2] != 255})
    assert eye_ids == [144, 145]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compress(output_path, fixed)
    return dict(inputSha256=sha(input_path), outputSha256=sha(output_path),
                neutralVerticesByteExact=True, samosModelByteExact=True,
                cinematicEyeIds=eye_ids,
                cinematicMovingPrimitives=sorted(motion), **report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=('all', 'samos', 'keira'), default='all')
    parser.add_argument('--stage-input', type=Path)
    parser.add_argument('--stage-output', type=Path)
    args = parser.parse_args()
    if bool(args.stage_input) != bool(args.stage_output):
        parser.error('--stage-input and --stage-output are required together')
    if args.stage_output and args.stage_input.resolve() == args.stage_output.resolve():
        parser.error('Staging requires a distinct private output file')
    os.environ['CINEMATIC_BUILD_WORK'] = str(args.work.resolve())
    sys.path.insert(0, str(RELEASE / 'stages'))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for key in ('samos', 'keira'):
        if args.model not in ('all', key):
            continue
        name = key + '-lod0'
        prototype = args.work / 'npc-samos-keira-prototypes' / (name + '.glb')
        native = args.work / 'npc-samos-keira-source/decompiler_out/jak2/levels/stadblmp' / (name + '.glb')
        provenance = json.loads(prototype.with_suffix('.provenance.json').read_text())
        parts = {int(i): ids for i, ids in provenance['faceFitting']['cinematicToNativePrimitiveSets'].items()}
        coefficients, metrics, asset = fit_model(native, prototype, parts, name=name)
        filename = name + ('-diagnostic-UNVALIDATED-coefficients.npy' if key == 'samos' else '-face-coefficients.npy')
        np.save(args.output / filename, coefficients)
        metrics['coefficientFile'] = filename
        if key == 'keira':
            metrics['mouthQA'] = keira_mouth_qa(asset, coefficients)
        if key == 'keira' and args.stage_output:
            metrics['privateStaging'] = stage_keira(args.stage_input, args.stage_output, asset, coefficients)
        report[name] = metrics
        print(name, metrics['status'], flush=True)
    (args.output / 'npc-face-fit-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: dict(status=v['status'], maximumConstraintErrorMetres=max(e['constraintMaximumAfterMetres'] for e in v['errors']),
                             medianRelativeSurfaceError=float(np.median([e['relativeSurfaceError'] for e in v['errors']])))
                      for k, v in report.items()}, indent=2))


if __name__ == '__main__':
    main()
