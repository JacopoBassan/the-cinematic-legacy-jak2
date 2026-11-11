"""Retain authored gameplay horns when the cinematic donor contains collapsed ones.

Both local Jak II cinematic donors store their horn surfaces collapsed to a line.
The gameplay horn mesh is therefore retained, with its exact native vertex records,
triangle strip, material and skin weights. Native target-darkjak scaling controls it.
Nothing in this module contains or distributes game geometry.
"""
import copy
import hashlib
import struct

import numpy as np


def retain_native_horns(raw, native_raw, prototype):
    from fr3_models import DRAW, Reader, model, serialize
    from inspect_faces import buffers
    from inspect_transformations import textures
    from retarget_glb import load

    target = model(raw, 'jakb-lod0')
    original = model(native_raw, 'jakb-lod0')
    donor_doc, _, donor_acc = load(prototype)
    names = [donor_doc['nodes'][i]['name'] for i in donor_doc['skins'][0]['joints']]
    assert names.index('horns') == 41 and len(names) == 63
    primitives = donor_doc['meshes'][0]['primitives']
    horn_parts = []
    for pi, primitive in enumerate(primitives):
        ids = np.unique(donor_acc(primitive['indices']).ravel())
        joints = donor_acc(primitive['attributes']['JOINTS_0'])[ids]
        weights = donor_acc(primitive['attributes']['WEIGHTS_0'])[ids]
        horn = ((joints == 41) & (weights > 0)).any(axis=1)
        if not horn.any():
            continue
        assert horn.all(), 'Do not remove a mixed horn/body primitive'
        assert np.allclose(weights[(joints == 41) & (weights > 0)], 1.0)
        positions = donor_acc(primitive['attributes']['POSITION'])[ids]
        assert np.ptp(positions[:, 1]) == 0 and np.ptp(positions[:, 2]) == 0
        for shape in primitive.get('targets', []):
            assert not np.any(donor_acc(shape['POSITION'])[ids])
        horn_parts.append(pi)
    assert horn_parts == [16, 25], ('Unsupported cinematic horn layout', horn_parts)

    native_textures, _ = textures(native_raw)
    target_textures, _ = textures(raw)
    native_draws = [d for effect in original['effects'] for d in effect['draws']]
    native_draw = list(DRAW.unpack(native_draws[22]))
    assert native_textures[native_draw[1]]['name'] == 'jakbsmall-horn'
    assert native_draw[2] == 255 and native_draw[5] == 50 and native_draw[6] == 0
    texture = native_textures[native_draw[1]]
    matches = [i for i, tex in enumerate(target_textures) if tex == texture]
    assert matches, 'Exact native horn image must remain in the archive'
    native_draw[1] = native_draw[1] if native_draw[1] in matches else matches[0]

    _, native_indices, _, native_vertices = buffers(native_raw, 'jakb-lod0')
    strip = native_indices[native_draw[3]:native_draw[3] + native_draw[4]].copy()
    used = np.unique(strip[strip != 0xffffffff])
    assert len(used) == 38
    source_records = np.frombuffer(native_vertices, 'u1').reshape(-1, 64)[used].copy()
    positions = source_records[:, :12].copy().view('<f4').reshape(-1, 3) / 4096
    assert np.ptp(positions[:, 1]) > 0.45, 'Native source horns must have full height'
    # The source and replacement share the exact native joint palette. Every
    # 64-byte record, including normals, weights, joint indices, UV and color,
    # is copied rather than interpolated or regenerated.
    index_start, old_indices, _, old_vertices = buffers(raw, 'jakb-lod0')
    first_vertex = len(old_vertices) // 64
    remapped = strip.copy()
    valid = strip != 0xffffffff
    remapped[valid] = np.searchsorted(used, strip[valid]) + first_vertex
    native_draw[3] = len(old_indices)
    replacement = DRAW.pack(*native_draw)

    old_model = copy.deepcopy(target)
    flat = [d for effect in target['effects'] for d in effect['draws']]
    assert len(flat) == len(primitives)
    old_main, old_extra = flat[16], flat[25]
    assert DRAW.unpack(old_main)[5] == 112 and DRAW.unpack(old_extra)[5] == 40
    assert target_textures[DRAW.unpack(old_main)[1]]['name'] == 'j2-cinematic/jakb-horn'
    assert target_textures[DRAW.unpack(old_extra)[1]]['name'] == 'j2-cinematic/common-white'
    touched_effects = []
    for ei, effect in enumerate(target['effects']):
        if old_main not in effect['draws'] and old_extra not in effect['draws']:
            continue
        reader = Reader(effect['mod'])
        modified, fixed = reader.draws(), reader.draws()
        assert old_main not in modified and old_extra not in modified
        assert old_main in fixed and old_extra in fixed
        def patch(draws):
            return [replacement if draw == old_main else draw
                    for draw in draws if draw != old_extra]
        effect['draws'] = patch(effect['draws'])
        fixed = patch(fixed)
        effect['mod'] = (struct.pack('<Q', len(modified)) + b''.join(modified)
                         + struct.pack('<Q', len(fixed)) + b''.join(fixed)
                         + effect['mod'][reader.pos:])
        # Effect visibility/reflection flags and all morph data stay untouched.
        assert effect['tail'] == old_model['effects'][ei]['tail']
        touched_effects.append(ei)
    assert touched_effects == [5]
    assert target['tail'] == old_model['tail']

    indices = np.concatenate([old_indices, remapped]).astype('<u4')
    vertices = old_vertices + source_records.tobytes()
    updated = (raw[:index_start] + struct.pack('<Q', len(indices)) + indices.tobytes()
               + struct.pack('<Q', len(vertices) // 64) + vertices + raw[-2:])
    updated = updated[:target['start']] + serialize(target) + updated[target['end']:]
    checked = model(updated, 'jakb-lod0')
    _, check_indices, _, check_vertices = buffers(updated, 'jakb-lod0')
    assert serialize(checked) == serialize(target)
    assert np.array_equal(check_indices[:len(old_indices)], old_indices)
    assert check_vertices[:len(old_vertices)] == old_vertices
    assert check_vertices[len(old_vertices):] == source_records.tobytes()
    assert updated[:target['start']] == raw[:target['start']]
    new_index_start = buffers(updated, 'jakb-lod0')[0]
    assert updated[checked['end']:new_index_start] == raw[target['end']:index_start]
    assert textures(updated)[0] == target_textures
    triangles = sum(DRAW.unpack(d)[5] for e in checked['effects'] for d in e['draws'])
    assert triangles == 10784
    return updated, dict(strategy='Exact native horn compatibility geometry',
        reason='Both cinematic donors collapse their unused horn surfaces to a line',
        replacedCinematicPrimitive=16, removedCollapsedPrimitive=25,
        nativePrimitive=22, nativeVertices=38, nativeTriangles=50,
        retainedHornHeightMetres=float(np.ptp(positions[:, 1])),
        nativeVertexRecordsSha256=hashlib.sha256(source_records.tobytes()).hexdigest(),
        nativeTriangleStripSha256=hashlib.sha256(strip.tobytes()).hexdigest(),
        exactNativeNormalsWeightsJointsUVColors=True, exactNativeMaterialAndTexture=True,
        nativeHornScaleControllerUnchanged=True, effectCount=len(checked['effects']),
        totalTriangles=triangles, unrelatedGlobalVertexAndIndexBytesUnchanged=True,
        unrelatedModelRecordsAndMorphPayloadsUnchanged=True)
