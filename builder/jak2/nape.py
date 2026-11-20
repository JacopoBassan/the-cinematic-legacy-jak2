"""Fit the original Jak II nape surface beneath the cinematic hair."""
import struct

import numpy as np


NAPE_TRIANGLES = 7
NAPE_TEXTURE_SLOT = -2


def triangle_strip(indices):
    """Expand a native Merc strip, resetting winding at each restart."""
    strip = np.lib.stride_tricks.sliding_window_view(indices, 3)
    starts = np.maximum.accumulate(np.where(indices == 0xffffffff, np.arange(len(indices)) + 1, 0))
    winding = (np.arange(len(strip)) - starts[:len(strip)]) % 2
    valid = ((strip != 0xffffffff).all(1) & (strip[:, 0] != strip[:, 1])
             & (strip[:, 1] != strip[:, 2]) & (strip[:, 2] != strip[:, 0]))
    faces = strip[valid].copy()
    faces[winding[valid] == 1] = faces[winding[valid] == 1][:, (1, 0, 2)]
    return faces


def fitted_nape(native_ids, native_vertices, skin_ids, skin_vertices, neck):
    source = np.frombuffer(native_vertices, 'u1').reshape(-1, 64)
    target = np.frombuffer(skin_vertices, 'u1').reshape(-1, 64)
    native = source.view('<f4')
    cinematic = target.view('<f4')

    def rigid(records, floats, ids):
        return np.where(records[ids, 60:63] == neck, floats[ids, 8:11], 0).sum(-1) > .999

    faces = triangle_strip(native_ids)
    points = native[faces, :3] / 4096
    # The native face draw includes a neck-bound patch below its scalp.
    cover = (rigid(source, native, faces).all(1) & (points[:, :, 1].min(1) > 2.49)
             & (points[:, :, 2].mean(1) < .075))
    faces = faces[cover]
    assert len(faces) == NAPE_TRIANGLES, 'Unexpected native nape topology'
    used, triangles = np.unique(faces, return_inverse=True)
    records = source[used].copy()
    fitted = records.view('<f4')
    points = fitted[:, :3] / 4096
    seam = np.flatnonzero((points[:, 1] < 2.60) & (np.abs(points[:, 0]) < .18))
    seam = seam[np.argsort(points[seam, 0])]

    skin = cinematic[skin_ids, :3] / 4096
    on_seam = (rigid(target, cinematic, skin_ids) & (np.abs(skin[:, 0]) < .151)
               & (skin[:, 1] > 2.499) & (skin[:, 1] < 2.517) & (skin[:, 2] < -.012))
    rim, indices = np.unique(skin[on_seam], return_index=True, axis=0)
    anchors = skin_ids[on_seam][indices[np.argsort(rim[:, 0])]]
    assert len(seam) == len(anchors) == 5 and len(used) == 9, 'Unexpected cinematic nape seam'
    fitted[seam, :3] = cinematic[anchors, :3]
    fitted[seam, 4:7] = cinematic[anchors, 4:7]
    records[seam, 56:60] = target[anchors, 56:60]
    # Continue down the stock atlas to the skin below the hairline at the seam.
    fitted[seam, 13] += np.interp(np.abs(points[seam, 0]), [0, .108, .162], [.31, .275, .105])
    # Follow the wider cinematic ear roots on both sides. Pull the top of the
    # patch into the hair shell so it stays overlapped at the highest lift.
    sides = np.abs(points[:, 0]) > .18
    fitted[sides, 0] *= 1.015
    fitted[sides, 1] += .016 * 4096
    fitted[points[:, 1] > 2.65, 1] += .024 * 4096
    fitted[:, 4:7] /= np.linalg.norm(fitted[:, 4:7], axis=1, keepdims=True)
    return records.tobytes(), triangles.astype('<u4')


def restore_nape(raw, native_raw, prototype):
    from fr3_models import DRAW, Reader, model, serialize
    from inspect_faces import buffers
    from retarget_glb import load
    from fix_faces import draw_list, vec

    name = 'jakb-lod0'
    doc, _, _ = load(prototype)
    joint_names = [doc['nodes'][i]['name'] for i in doc['skins'][0]['joints']]
    neck = joint_names.index('neckB') + 1
    original = model(native_raw, name)
    target = model(raw, name)
    source_draw = list(DRAW.unpack(original['effects'][6]['draws'][1]))
    assert source_draw[1:3] == [NAPE_TEXTURE_SLOT, 255] and source_draw[6] == 0
    face_effect = target['effects'][5]
    face_draw = DRAW.unpack(face_effect['draws'][0])
    assert face_draw[1:3] == (-10, 255) and face_draw[6] == 1
    _, native_indices, _, native_vertices = buffers(native_raw, name)
    start, indices, _, vertices = buffers(raw, name)
    records, faces = fitted_nape(
        native_indices[source_draw[3]:source_draw[3] + source_draw[4]], native_vertices,
        np.unique(indices[face_draw[3]:face_draw[3] + face_draw[4]]), vertices, neck)
    faces += len(vertices) // 64
    source_draw[3:7] = len(indices), faces.size, NAPE_TRIANGLES, 1
    draw = DRAW.pack(*source_draw)
    face_effect['draws'].append(draw)
    reader = Reader(face_effect['mod'])
    modified, fixed = reader.draws(), reader.draws()
    face_effect['mod'] = draw_list(modified) + draw_list([*fixed, draw]) + face_effect['mod'][reader.pos:]
    target['tail'] = struct.pack('<I', struct.unpack_from('<I', target['tail'])[0] + 2) + target['tail'][4:]

    packed = (raw[:target['start']] + serialize(target) + raw[target['end']:start]
              + vec(indices.astype('<u4').tobytes() + faces.tobytes(), 4)
              + vec(vertices + records, 64) + raw[-2:])
    checked = model(packed, name)
    _, new_indices, _, new_vertices = buffers(packed, name)
    assert serialize(checked) == serialize(target)
    assert np.array_equal(new_indices[:len(indices)], indices)
    assert new_vertices[:len(vertices)] == vertices
    return packed, {'triangles': NAPE_TRIANGLES, 'vertices': len(records) // 64, 'joint': 'neckB'}
