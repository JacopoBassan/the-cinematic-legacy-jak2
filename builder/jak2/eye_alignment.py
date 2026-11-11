"""Align live Jak II eye coordinates without distorting cinematic eye surfaces.

Model XY projection confuses the different cinematic eyeball shapes with gaze.
Map the native eye's normal/UV field onto the cinematic normals instead. This
retains Jak's native gaze. Daxter keeps his authored cinematic UVs: matching
his different eyeball normals collapses most of the surface onto a UV boundary.
Only UVs change. Native eye IDs, gaze, eyelid controls and geometry stay intact.
"""
from __future__ import annotations

import math
import struct

import numpy as np


def angle_between(left, right):
    return math.degrees(math.acos(float(np.clip(np.dot(left, right), -1, 1))))


def sample_eye(primitive, accessor, uv, point):
    normals = accessor(primitive['attributes']['NORMAL'])
    positions = accessor(primitive['attributes']['POSITION'])
    hits = []
    target = np.r_[point, 1.0]
    for ids in accessor(primitive['indices']).reshape(-1, 3):
        matrix = np.vstack([uv[ids].T, np.ones(3)])
        bary = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if np.linalg.norm(matrix@bary-target) < 1e-6 and bary.min() >= -1e-6:
            direction = normals[ids].T@bary
            direction /= np.linalg.norm(direction)
            hits.append((float(direction[2]), positions[ids].T@bary, direction))
    if not hits:
        raise ValueError(f'Eye does not cover compositor point {point}')
    _, position, direction = max(hits, key=lambda hit: hit[0])
    return {'position': position.tolist(), 'direction': direction.tolist()}


def native_xy_projection(source, source_accessor, destination, destination_accessor):
    used = np.unique(source_accessor(source['indices']))
    position = source_accessor(source['attributes']['POSITION'])[used]
    uv = source_accessor(source['attributes']['TEXCOORD_0'])[used]
    projection = np.linalg.lstsq(np.c_[position[:, :2], np.ones(len(used))], uv, rcond=None)[0]
    position = destination_accessor(destination['attributes']['POSITION'])
    return np.clip(np.c_[position[:, :2], np.ones(len(position))]@projection, 0, 1)


def directional_uv(source, source_accessor, destination, destination_accessor):
    from fix_faces import closest_surface

    triangles = source_accessor(source['indices']).reshape(-1, 3)
    normals = source_accessor(source['attributes']['NORMAL'])
    normals = normals/np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    native_uv = source_accessor(source['attributes']['TEXCOORD_0'])
    target_normals = destination_accessor(destination['attributes']['NORMAL'])
    target_normals = target_normals/np.maximum(np.linalg.norm(target_normals, axis=1, keepdims=True), 1e-12)
    uv = destination_accessor(destination['attributes']['TEXCOORD_0']).copy()
    used = np.unique(destination_accessor(destination['indices']))
    boundary = []
    for vertex in used:
        direction = target_normals[vertex]
        hits = []
        for ids in triangles:
            # Interpolated source normal = scale * target direction. Solve the
            # ray/normal-triangle intersection, retaining the native UV field.
            matrix = np.block([[normals[ids].T, -direction[:, None]],
                               [np.ones((1, 3)), np.zeros((1, 1))]])
            try:
                result = np.linalg.solve(matrix, np.array([0, 0, 0, 1.0]))
            except np.linalg.LinAlgError:
                continue
            if result[:3].min() >= -1e-6 and result[3] > 0:
                hits.append((float(result[3]), native_uv[ids].T@result[:3]))
        if hits:
            uv[vertex] = max(hits, key=lambda hit: hit[0])[1]
        else:
            boundary.append(vertex)
    if boundary:
        # The cinematic mesh covers more of the eyeball than the gameplay
        # patch. Extend only to the closest authored native UV boundary.
        ti, bary, _ = closest_surface(target_normals[boundary], normals[triangles])
        uv[boundary] = np.einsum('nij,ni->nj', native_uv[triangles[ti]], bary)
    assert np.isfinite(uv[used]).all()
    assert uv[used].min() >= -1e-6 and uv[used].max() <= 1+1e-6
    uv[used] = np.clip(uv[used], 0, 1)
    return uv.astype('<f4'), {'usedVertices': len(used), 'boundaryVertices': len(boundary)}


def calibrated_eye_uvs(lp, la, hp, ha, high_eyes, low_eyes, name):
    assert set(high_eyes.values()) == {0, 1} and set(low_eyes) == {0, 1}
    if name == 'daxter-lod0':
        return authored_daxter_eye_uvs(lp, la, hp, ha, high_eyes, low_eyes)
    by_side = {side: part for part, side in high_eyes.items()}
    corrected, mapping = {}, {}
    for side, part in by_side.items():
        p, source = hp[part], lp[low_eyes[side]]
        used = np.unique(ha(p['indices']))
        native_used = np.unique(la(source['indices']))
        assert (np.median(ha(p['attributes']['POSITION'])[used, 0]) > 0) == (side == 0)
        assert (np.median(la(source['attributes']['POSITION'])[native_used, 0]) > 0) == (side == 0)
        corrected[part], mapping[part] = directional_uv(source, la, p, ha)
    samples = []
    for label, gaze in [('neutral', (0, 0)), ('left', (-.25, 0)), ('right', (.25, 0)),
                        ('up', (0, -.2)), ('down', (0, .2)),
                        ('upper-left', (-.15, -.1)), ('lower-right', (.15, .1))]:
        native, actual, errors = [], [], []
        for side in (0, 1):
            source, p = lp[low_eyes[side]], hp[by_side[side]]
            native.append(sample_eye(source, la, la(source['attributes']['TEXCOORD_0']), np.array(gaze)+.5))
            actual.append(sample_eye(p, ha, corrected[by_side[side]], np.array(gaze)+.5))
            errors.append(angle_between(native[-1]['direction'], actual[-1]['direction']))
        assert max(errors) < 2.0, (name, label, errors)
        samples.append({'gaze': label, 'compositorOffset': list(gaze), 'native': native,
                        'cinematic': actual, 'directionErrorDegrees': errors,
                        'nativeBinocularAngle': angle_between(native[0]['direction'], native[1]['direction']),
                        'cinematicBinocularAngle': angle_between(actual[0]['direction'], actual[1]['direction'])})
    previous = [sample_eye(hp[by_side[side]], ha,
                native_xy_projection(lp[low_eyes[side]], la, hp[by_side[side]], ha), [.5, .5])
                for side in (0, 1)]
    report = {'method': 'Native anatomical eye normal-to-UV field transfer',
              'previousXYProjectionNeutralBinocularAngle': angle_between(previous[0]['direction'], previous[1]['direction']),
              'eyePrimitiveSides': high_eyes, 'mapping': mapping, 'gazeSamples': samples,
              'maximumGazeDirectionErrorDegrees': max(max(sample['directionErrorDegrees']) for sample in samples),
              'nativeControlsPreserved': ['x', 'y', 'lid', 'iris-scale', 'pupil-scale', 'lid-scale', 'blink'],
              'nativeControlSource': 'goal_src/jak2/engine/gfx/foreground/eye.gc:1182-1298',
              'neutralGeometryChanged': False, 'facialShapesChanged': False, 'staticEyes': False,
              'validation': 'Numerical gaze passed; live gaze, eyelid and blink validation required'}
    return corrected, report


def authored_daxter_eye_uvs(lp, la, hp, ha, high_eyes, low_eyes):
    """Preserve the donor's full iris area while using native live eye controls."""
    corrected, mapping = {}, {}
    for part, side in high_eyes.items():
        p, source = hp[part], lp[low_eyes[side]]
        used = np.unique(ha(p['indices']))
        uv = ha(p['attributes']['TEXCOORD_0']).astype('<f4')
        assert np.isfinite(uv[used]).all()
        assert uv[used].min() >= 0 and uv[used].max() <= 1
        assert (np.median(ha(p['attributes']['POSITION'])[used, 0]) > 0) == (side == 0)
        triangles = ha(p['indices']).reshape(-1, 3)
        ab = uv[triangles[:, 1]] - uv[triangles[:, 0]]
        ac = uv[triangles[:, 2]] - uv[triangles[:, 0]]
        area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
        assert (area > 1e-12).all(), 'Collapsed authored Daxter eye UV triangle'
        native_uv = la(source['attributes']['TEXCOORD_0'])
        native_center = np.array(sample_eye(source, la, native_uv, [.5, .5])['position'])
        authored_center = np.array(sample_eye(p, ha, uv, [.5, .5])['position'])
        footprints = []
        for primitive, accessor, coordinates in ((source, la, native_uv), (p, ha, uv)):
            points = [np.array(sample_eye(primitive, accessor, coordinates, point)['position'])
                      for point in ((.4, .5), (.6, .5), (.5, .4), (.5, .6))]
            footprints.append(np.array([np.linalg.norm(points[1] - points[0]),
                                        np.linalg.norm(points[3] - points[2])]))
        ratio = footprints[1] / footprints[0]
        # Validate an area around the pupil, not just its central normal.
        assert ((ratio > .75) & (ratio < 1.3)).all(), 'Distorted central eye footprint'
        offset = float(np.linalg.norm(authored_center - native_center))
        assert offset < float(footprints[0].min()) * .5, 'Displaced neutral Daxter pupil'
        corrected[part] = uv
        mapping[part] = {'usedVertices': len(used), 'boundaryVertices': 0,
                         'collapsedUvTriangles': 0, 'triangles': len(triangles),
                         'authoredUvPreserved': True,
                         'centralUvFootprintNative': footprints[0].tolist(),
                         'centralUvFootprintCinematic': footprints[1].tolist(),
                         'centralUvFootprintRatio': ratio.tolist(),
                         'neutralCenterOffsetModelUnits': offset}
    by_side = {side: part for part, side in high_eyes.items()}
    samples = []
    for label, gaze in [('neutral', (0, 0)), ('left', (-.25, 0)), ('right', (.25, 0)),
                        ('up', (0, -.2)), ('down', (0, .2)),
                        ('upper-left', (-.15, -.1)), ('lower-right', (.15, .1))]:
        native, actual = [], []
        for side in (0, 1):
            source, part = lp[low_eyes[side]], by_side[side]
            native.append(sample_eye(source, la, la(source['attributes']['TEXCOORD_0']), np.array(gaze) + .5))
            actual.append(sample_eye(hp[part], ha, corrected[part], np.array(gaze) + .5))
        samples.append({'gaze': label, 'compositorOffset': list(gaze),
                        'native': native, 'cinematic': actual})
    return corrected, {'method': 'Authored cinematic Daxter eye UVs with native live compositor',
                       'eyePrimitiveSides': high_eyes, 'mapping': mapping, 'gazeSamples': samples,
                       'footprintDefinition': 'Physical surface span of UV .4 to .6, not texture alpha bounds',
                       'nativeControlsPreserved': ['x', 'y', 'lid', 'iris-scale', 'pupil-scale', 'lid-scale', 'blink'],
                       'neutralGeometryChanged': False, 'facialShapesChanged': False, 'staticEyes': False,
                       'validation': 'Full UV area and neutral placement passed; live gaze and blink checks required'}


def write_eye_uvs(raw, name, doc, accessor, indices, vertices, uv_by_part):
    from fr3_models import model, DRAW

    target = model(raw, name)
    draws = sorted([draw for effect in target['effects'] for draw in effect['draws']],
                   key=lambda draw: DRAW.unpack(draw)[3])
    prims = doc['meshes'][0]['primitives']
    assert len(draws) == len(prims)
    changed = set()
    for part, uv in uv_by_part.items():
        values = DRAW.unpack(draws[part])
        ids = indices[values[3]:values[3]+values[4]]
        local = accessor(prims[part]['indices']).ravel()
        offset = int(ids[0])-int(local[0])
        assert np.array_equal(ids, local+offset)
        for vertex in np.unique(ids):
            struct.pack_into('<2f', vertices, int(vertex)*64+48, *uv[int(vertex)-offset])
            changed.add(int(vertex))
    return changed
