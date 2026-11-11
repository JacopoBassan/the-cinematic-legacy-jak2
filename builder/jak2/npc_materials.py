"""Restore cinematic NPC material groups with explicit native control maps.

Only private FR3 candidates are written. Source pixels and render modes come
from the user's extraction; none are embedded in this source distribution.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RELEASE/'stages'), str(RELEASE)]


def configure(work):
    os.environ['CINEMATIC_BUILD_WORK'] = str(work.resolve())
    from native_io import activate
    activate()
    global DRAW, Reader, model, serialize, compress, decompress, textures, buffers, texture_bytes
    from fr3_models import DRAW, Reader, model, serialize, compress, decompress
    from inspect_transformations import textures
    from inspect_faces import buffers
    from npc_builder import texture_bytes


def group(name, source_effect, source_draw, texture):
    if name in ('sig-lod0', 'ashelin-lod0'):
        return source_effect
    if name == 'samos-lod0':
        return 7 if texture == 'samos-teeth2' else source_effect
    if name == 'keira-lod0':
        if source_effect == 1:
            return 0 if texture == 'keira-torch-guard-01' else 1
        if source_effect == 2:
            return 3 if source_draw[2] != 255 else 2
        return {3:4, 4:5}[source_effect]
    raise ValueError(name)


def restore(raw, donor_raw, name, donor_name, omitted=(), native_tail=None):
    spec = importlib.util.spec_from_file_location('jak2_npc_runtime', Path(__file__).with_name('patch_runtime.py'))
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    EFFECTS = runtime.EFFECTS
    before = raw
    target, donor = model(raw, name), model(donor_raw, donor_name)
    original_buffers = buffers(raw, name)[1:]
    target_draws = sorted((DRAW.unpack(d) for e in target['effects'] for d in e['draws']), key=lambda d:d[3])
    source_textures, _ = textures(donor_raw)
    target_textures, texture_end = textures(raw)
    existing = {texture_bytes(t):i for i,t in enumerate(target_textures)}
    added = []
    def texture(source_index):
        item = copy.deepcopy(source_textures[source_index])
        item['name'] = 'j2-cinematic/'+donor_name+'/'+item['name']
        key = texture_bytes(item)
        if key not in existing:
            existing[key] = len(target_textures)+len(added)
            added.append(item)
        return existing[key]
    effects = [None] * len(EFFECTS[name])
    primitive = output_primitive = 0
    rows = []
    for ei, source in enumerate(donor['effects']):
        for packed in source['draws']:
            si = primitive
            primitive += 1
            if si in omitted:
                continue
            sd = DRAW.unpack(packed)
            td = list(target_draws[output_primitive])
            assert sd[5] == td[5], (name, si, sd[5], td[5])
            source_texture = source_textures[sd[1]]['name']
            destination = group(name, ei, sd, source_texture)
            if effects[destination] is None:
                effect = copy.deepcopy(target['effects'][0])
                # Geometry has not yet received native facial buffers.
                r = Reader(effect['mod'])
                assert not r.draws() and not r.draws() and not r.vector(64)
                effect['draws'] = []
                env = source['tail'][8]
                reflection = texture(struct.unpack_from('<I', source['tail'], 4)[0]) if env else 0
                effect['tail'] = source['tail'][:4]+struct.pack('<I', reflection)+source['tail'][8:9]+b'\0'
                effects[destination] = effect
            else:
                assert effects[destination]['tail'][:4] == source['tail'][:4]
                assert effects[destination]['tail'][8] == source['tail'][8]
            td[0] = sd[0]
            td[1] = texture(sd[1])
            td[2] = sd[2]
            effects[destination]['draws'].append(DRAW.pack(*td))
            rows.append(dict(primitive=output_primitive, sourcePrimitive=si,
                sourceEffect=ei, effect=destination, nativeControl=EFFECTS[name][destination],
                texture=source_texture, eye=td[2], triangles=td[5],
                reflective=bool(source['tail'][8])))
            output_primitive += 1
    assert output_primitive == len(target_draws)
    assert all(e and e['draws'] for e in effects), (name, [bool(e) for e in effects])
    target['effects'] = effects
    tail = bytearray(target['tail'])
    struct.pack_into('<I', tail, 0, output_primitive)
    if native_tail is not None:
        tail[8:] = native_tail[8:]
    target['tail'] = bytes(tail)
    raw = raw[:target['start']]+serialize(target)+raw[target['end']:]
    r = Reader(before, 2)
    r.take(r.num('Q'))
    raw = raw[:r.pos]+struct.pack('<Q', len(target_textures)+len(added))+b''.join(texture_bytes(t) for t in target_textures+added)+raw[texture_end:]
    # Both model serialization and the texture table grew, but their payload
    # offsets are reconstructed from the parsed record boundaries.
    after_buffers = buffers(raw, name)[1:]
    assert original_buffers[0].tobytes() == after_buffers[0].tobytes()
    assert original_buffers[2] == after_buffers[2]
    assert textures(raw)[0][:len(target_textures)] == target_textures
    checked = model(raw, name)
    assert len(checked['effects']) == len(EFFECTS[name])
    assert sum(DRAW.unpack(d)[5] for e in checked['effects'] for d in e['draws']) == sum(row['triangles'] for row in rows)
    return raw, dict(model=name, donor=donor_name, effects=len(effects),
        effectControls=EFFECTS[name], primitiveProvenance=rows,
        geometryAndNativeTexturePrefixExact=True, appendedTextures=len(added))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'work', 'input', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--only', nargs='+', choices=['sig','ashelin','samos','keira'], default=['sig','ashelin','samos','keira'])
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(a.work.resolve()) or a.output.resolve().is_relative_to((a.work/'import').resolve()):
        raise ValueError('Output must remain inside private work and outside the active project')
    configure(a.work)
    jobs = dict(sig=('atollext.fr3','sig-highres-lod0',(),['atollext.fr3','underb.fr3']),
        ashelin=('atollext.fr3','ashelin-highres-lod0',(),['atollext.fr3','ctyasha.fr3']),
        samos=('lintcstb.fr3','samos-highres-lod0',(),['stadblmp.fr3']),
        keira=('lintcstb.fr3','keira-highres-lod0',(0,1,27),['stadblmp.fr3']))
    a.output.mkdir(parents=True, exist_ok=True)
    report = []
    for character in a.only:
        donor_file, donor_name, omitted, files = jobs[character]
        donor_path = a.data/'out/jak2/fr3'/donor_file
        if not donor_path.is_file():
            donor_path = a.work/'import/out/jak2/fr3'/donor_file
        donor_raw = decompress(donor_path)
        for filename in files:
            output = a.output/filename
            source = output if output.is_file() else a.input/filename
            raw = decompress(source)
            native = model(decompress(a.work/'import/out/jak2/fr3'/filename), character+'-lod0')
            fixed, row = restore(raw, donor_raw, character+'-lod0', donor_name, omitted, native['tail'])
            compress(output, fixed)
            assert decompress(output) == fixed
            row.update(file=filename, sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            report.append(row)
    (a.output/'npc-materials.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('model','file','effects','appendedTextures')} for r in report], indent=2))


if __name__ == '__main__':
    main()
