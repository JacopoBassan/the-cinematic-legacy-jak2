"""Reconnect Jak II cinematic skin to the game's existing animated outputs.

Normal and Dark Jak gameplay share one native rig. Cinematic palettes already
exist in OpenGOAL's oracle CLUT group. Slots are derived from the local donor.
"""
import struct


def connect(raw, donor_raw, glb_path):
    import numpy as np
    from fr3_models import DRAW, Reader, model, serialize
    from inspect_faces import buffers
    from inspect_transformations import textures
    from retarget_glb import load
    doc, _, _ = load(glb_path)
    source = model(donor_raw, 'jak-highres-lod0')
    donor_draws = [DRAW.unpack(d) for e in source['effects'] for d in e['draws']]
    prims = doc['meshes'][0]['primitives']
    assert len(donor_draws) == len(prims)
    slots = {}
    for primitive, draw in zip(prims, donor_draws):
        if draw[1] >= 0:
            continue
        material = doc['materials'][primitive['material']]
        texture = doc['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
        name = doc['images'][texture['source']]['name']
        assert name not in slots or slots[name] == draw[1]
        slots[name] = draw[1]
    # Source-layout guard. These materials include the hands, so no unrelated
    # low-resolution finger image is substituted.
    assert slots == {'jakb-hairtrans': -11, 'jakb-facert': -10,
                     'jakb-eyebrow': -14, 'jakb-facelft': -9}, slots
    tex, _ = textures(raw)
    target = model(raw, 'jakb-lod0')
    counts = {'normal': 0, 'modified': 0, 'fixed': 0}

    def patch(draw, kind):
        values = list(DRAW.unpack(draw))
        if values[1] >= 0:
            name = tex[values[1]]['name'].removeprefix('j2-cinematic/')
            if name in slots:
                assert values[2] == 255, 'Eye surface must retain its live eye slot'
                values[1] = slots[name]
                counts[kind] += 1
        return DRAW.pack(*values)

    for effect in target['effects']:
        effect['draws'] = [patch(d, 'normal') for d in effect['draws']]
        r = Reader(effect['mod'])
        modified, fixed = r.draws(), r.draws()
        effect['mod'] = (struct.pack('<Q', len(modified)) + b''.join(patch(d, 'modified') for d in modified)
                         + struct.pack('<Q', len(fixed)) + b''.join(patch(d, 'fixed') for d in fixed)
                         + effect['mod'][r.pos:])
    assert counts['normal'] >= 5 and counts['modified'] > 0, counts
    result = raw[:target['start']] + serialize(target) + raw[target['end']:]
    assert len(result) == len(raw)
    assert result[:target['start']] == raw[:target['start']]
    assert result[target['end']:] == raw[target['end']:]
    _, old_indices, _, old_vertices = buffers(raw, target['name'])
    _, new_indices, _, new_vertices = buffers(result, target['name'])
    assert np.array_equal(old_indices, new_indices) and old_vertices == new_vertices
    return result, dict(animatedMaterialSlots=slots, connectedDraws=counts,
                        nativeHornAndBodyScalingPreserved=True, nativeEyeControllerPreserved=True,
                        geometryAndMorphBuffersUnchanged=True,
                        requiresCinematicOracleClutRuntimeUpdate=True)


def patch_texture_animation(source):
    """Update cinematic CLUTs with the native gameplay transformation amount.

    Five false destinations produce the renderer's expected 48-byte payload.
    Outputs use animated slots and never remap unrelated native VRAM entries.
    No new engine slots, texture names, or palette copies are required.
    """
    old = '''    ((= anim-array *darkjak-texture-anim-array*)
     (pc-clut-blender bucket (texture-anim-pc darkjak) anim-array)
     (return #f)
     )'''
    new = '''    ((= anim-array *darkjak-texture-anim-array*)
     (pc-clut-blender bucket (texture-anim-pc darkjak) anim-array)
     ;; Cinematic skin follows the native transformation on the same rig.
     ;; Oracle has five outputs: its payload is 48 bytes, not the native 32.
     ;; False destinations prevent VRAM aliasing with unrelated materials.
     (let ((cinematic (new 'static 'texture-anim-array :type texture-anim
                        (new 'static 'texture-anim :tex #f)
                        (new 'static 'texture-anim :tex #f)
                        (new 'static 'texture-anim :tex #f)
                        (new 'static 'texture-anim :tex #f)
                        (new 'static 'texture-anim :tex #f))))
       (set! (-> cinematic array-data 0 frame-time)
             (-> anim-array array-data 0 frame-time))
       (pc-clut-blender bucket (texture-anim-pc oracle-jak) cinematic))
     (return #f)
     )'''
    if source.count(old) != 1:
        raise ValueError('Unsupported native dark texture update anchor')
    return source.replace(old, new, 1)
