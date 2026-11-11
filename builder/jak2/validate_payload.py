"""Validate generated Jak II geometry, controls and untouched native context."""
import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def validate(data, work):
    import npc_materials as io
    io.configure(work)
    from jak2.build_npcs import native_joint_limit
    from jak2.patch_runtime import EFFECTS, REFLECTION_EFFECTS
    from npc_builder import texture_bytes
    from review_npc_context import compare_draws, payload, texture_equal
    from jak2.restore_context import indexed_textures
    specs = {
        'GAME.fr3': {'jakb-lod0': (10784, [0, 1], True), 'daxter-lod0': (9510, [2, 3], True)},
        'atollext.fr3': {'sig-lod0': (10750, [208], False), 'ashelin-lod0': (10223, [200, 201], False)},
        'underb.fr3': {'sig-lod0': (10750, [208], False)},
        'ctyasha.fr3': {'ashelin-lod0': (10223, [200, 201], False)},
        'stadblmp.fr3': {'samos-lod0': (10434, [216, 217], False), 'keira-lod0': (10528, [144, 145], True)},
    }
    report = []
    for filename, replacements in specs.items():
        path = work / 'GAME-context.fr3' if filename == 'GAME.fr3' else work / 'npc-ready' / filename
        raw = io.decompress(path)
        baseline = io.decompress(data / 'out/jak2/fr3' / filename)
        native_textures, native_texture_end = io.textures(baseline)
        textures, texture_end = io.textures(raw)
        native_indexed, _ = indexed_textures(baseline, native_texture_end)
        indexed, _ = indexed_textures(raw, texture_end)
        assert indexed[:len(native_indexed)] == native_indexed, (filename, 'Lost ordered native indexed textures')
        native_index_names = {t['name'] for t in native_indexed}
        assert all(t['name'] not in native_index_names for t in indexed[len(native_indexed):]), (filename, 'Ambiguous indexed texture alias')
        current_texture_keys = {texture_bytes(t) for t in textures}
        assert all(texture_bytes(t) in current_texture_keys for t in native_textures), 'Lost native texture context'
        first = next(iter(replacements))
        _, indices, _, vertices = io.buffers(raw, first)
        _, old_indices, _, old_vertices = io.buffers(baseline, first)
        vertex_bytes = np.frombuffer(vertices, 'u1').reshape(-1, 64)
        old_vertex_bytes = np.frombuffer(old_vertices, 'u1').reshape(-1, 64)
        rows = []
        for name, (triangle_count, eyes, has_morphs) in replacements.items():
            target, native = io.model(raw, name), io.model(baseline, name)
            limit = native_joint_limit(work / 'import/decompiler_out/jak2/raw_obj', name.removesuffix('-lod0'), name)
            assert target['tail'][8:] == native['tail'][8:], (name, 'native decode configuration')
            assert len(target['effects']) == len(EFFECTS[name])
            assert all(0 <= control < len(native['effects']) for control in EFFECTS[name] + REFLECTION_EFFECTS[name])
            assert sum(io.DRAW.unpack(d)[5] for e in target['effects'] for d in e['draws']) == triangle_count
            actual_eyes = set()
            active_joints = set()
            float_vertices = 0
            animated_slots = set()
            def check(draws, source):
                for packed in draws:
                    mode, texture, eye, start, count, _, _ = io.DRAW.unpack(packed)
                    if texture >= 0:
                        assert texture < len(textures)
                    else:
                        assert name == 'jakb-lod0' and texture in (-14, -11, -10, -9)
                        animated_slots.add(texture)
                    if eye != 255:
                        actual_eyes.add(eye)
                    strip = indices[start:start + count]
                    assert len(strip) == count
                    used = strip[strip != 0xffffffff]
                    assert len(used) and int(used.max()) < len(source)
                    data = source[used]
                    floats = data.copy().view('<f4').reshape(-1, 16)
                    assert np.isfinite(floats[:, :14]).all()
                    weights = floats[:, 8:11]
                    assert (weights >= 0).all() and np.allclose(weights.sum(1), 1, atol=1e-5)
                    lengths = np.linalg.norm(floats[:, 4:7], axis=1)
                    nonunit = np.abs(lengths - 1) > 3e-5
                    if nonunit.any():
                        # The retained native PS2 horn records use quantized
                        # normals. native_horns verifies their full bytes.
                        assert name == 'jakb-lod0'
                        assert (np.abs(lengths[nonunit] - 1) < .02).all()
                        assert (((data[nonunit, 60:63] == 42) & (weights[nonunit] > 0)).any(axis=1)).all()
                    active_joints.update(int(x) for x in data[:, 60:63][weights > 0])
            for effect in target['effects']:
                assert effect['draws'], (name, 'Empty effect can crash Merc2')
                if effect['tail'][8]:
                    assert struct.unpack_from('<I', effect['tail'], 4)[0] < len(textures)
                check(effect['draws'], vertex_bytes)
                modified, fixed, float_data, remainder = payload(effect)
                check(modified, float_data)
                check(fixed, vertex_bytes)
                float_vertices += len(float_data)
            assert sorted(actual_eyes) == eyes
            assert active_joints and max(active_joints) <= limit
            assert bool(float_vertices) == has_morphs, (name, float_vertices, has_morphs)
            if name == 'jakb-lod0':
                assert animated_slots == {-14, -11, -10, -9}
            rows.append({'model': name, 'triangles': triangle_count, 'eyeIds': eyes,
                         'maximumWeightedJoint': max(active_joints), 'nativeJointLimit': limit,
                         'floatMorphVertices': float_vertices, 'effectControls': EFFECTS[name],
                         'reflectionControls': REFLECTION_EFFECTS[name]})
        # Importing geometry can shift numeric offsets. Compare actual indexed
        # data, textures and morph buffers of every other native Merc model.
        names = sorted({m.decode() for m in re.findall(rb'[A-Za-z0-9+_.-]+-lod[0-9]', baseline)} - replacements.keys())
        checked = []
        for name in names:
            try:
                old = io.model(baseline, name)
            except AssertionError:
                continue
            new = io.model(raw, name)
            assert old['tail'] == new['tail'], (filename, name, 'decode fields')
            assert len(old['effects']) == len(new['effects'])
            for before, after in zip(old['effects'], new['effects']):
                assert before['tail'][:4] + before['tail'][8:] == after['tail'][:4] + after['tail'][8:]
                if before['tail'][8]:
                    a, b = (struct.unpack_from('<I', e['tail'], 4)[0] for e in (before, after))
                    assert texture_equal(native_textures[a], textures[b])
                compare_draws(before['draws'], after['draws'], old_indices, indices, old_vertex_bytes, vertex_bytes, native_textures, textures)
                om, of, ov, op = payload(before)
                nm, nf, nv, npayload = payload(after)
                assert np.array_equal(ov, nv) and op == npayload
                compare_draws(of, nf, old_indices, indices, old_vertex_bytes, vertex_bytes, native_textures, textures)
                compare_draws(om, nm, old_indices, indices, ov, nv, native_textures, textures)
            checked.append(name)
        report.append({'file': filename, 'subjects': rows, 'nativeGlobalTexturesRetained': len(native_textures),
                       'orderedNativeIndexedTexturesExact': len(native_indexed),
                       'additionalIndexedTextures': len(indexed) - len(native_indexed),
                       'unrelatedModelsSemanticallyExact': checked})
    result = {'status': 'Static asset/control/context validation passed', 'files': report,
              'allSixModelsValidated': True, 'runtimeCoverage': 'See separate bounded runtime review; no universal campaign claim'}
    (work / 'payload-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.data, args.work), indent=2))
