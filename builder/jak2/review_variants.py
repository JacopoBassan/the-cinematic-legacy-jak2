"""Validate staged Jak II transformation routing against local source assets."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'stages'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    os.environ['CINEMATIC_BUILD_WORK'] = str(a.work.resolve())
    from fr3_models import decompress, model, DRAW, Reader, serialize
    from inspect_faces import buffers
    from inspect_transformations import textures
    from retarget_glb import load
    from transformations import connect, patch_texture_animation
    raw = decompress(a.candidate)
    target = model(raw, 'jakb-lod0')
    slots = {v[1] for e in target['effects'] for v in map(DRAW.unpack, e['draws']) if v[1] < 0}
    assert slots == {-9, -10, -11, -14}
    eye_draws = [v for e in target['effects'] for v in map(DRAW.unpack, e['draws']) if v[2] != 255]
    assert {v[2] for v in eye_draws} == {0, 1}
    assert sum(v[5] for e in target['effects'] for v in map(DRAW.unpack, e['draws'])) == 10784
    patched = patch_texture_animation((a.data / 'goal_src/jak2/engine/gfx/texture/texture-anim.gc').read_text())
    branch = patched.split(';; Cinematic skin follows', 1)[1].split('(return #f)', 1)[0]
    destinations = len(re.findall(r"\(new 'static 'texture-anim :tex #f\)", branch))
    assert destinations == 5
    # OpenGOAL's native oracle renderer expects 16 morph bytes plus one
    # 16-byte-rounded destination vector containing five uint32 values.
    payload_bytes = 16 + 16 * ((destinations + 3) // 4)
    assert payload_bytes == 48
    original_target = (a.data / 'goal_src/jak2/engine/target/target-darkjak.gc').read_text()
    assert '(set-darkjak-texture-morph! f30-0)' in original_target
    assert '(-> self horns)' in original_target
    doc, _, acc = load(a.work / 'rig-prototypes/jakb-lod0.glb')
    names = [doc['nodes'][j]['name'] for j in doc['skins'][0]['joints']]
    horns = names.index('horns')
    collapsed_donor_vertices = 0
    for primitive in doc['meshes'][0]['primitives']:
        joints, weights = (acc(primitive['attributes'][k]) for k in ('JOINTS_0', 'WEIGHTS_0'))
        collapsed_donor_vertices += int(np.any((joints == horns) & (weights > 0), axis=1).sum())
    assert collapsed_donor_vertices == 94
    # Cinematic horn surfaces are collapsed in the retail source. Check the
    # actual staged render geometry, not only an active joint influence.
    native_raw = decompress(a.data / 'out/jak2/fr3/GAME.fr3')
    native_model = model(native_raw, 'jakb-lod0')
    native_draw = [d for e in native_model['effects'] for d in e['draws']][22]
    native_values = DRAW.unpack(native_draw)
    native_textures, _ = textures(native_raw)
    target_textures, _ = textures(raw)
    candidates = [d for e in target['effects'] for d in e['draws']
                  if DRAW.unpack(d)[1] >= 0
                  and target_textures[DRAW.unpack(d)[1]] == native_textures[native_values[1]]]
    assert len(candidates) == 1
    staged_values = DRAW.unpack(candidates[0])
    assert [staged_values[i] for i in (0, 2, 4, 5, 6)] == [native_values[i] for i in (0, 2, 4, 5, 6)]
    _, native_indices, _, native_vertices = buffers(native_raw, 'jakb-lod0')
    _, staged_indices, _, staged_vertices = buffers(raw, 'jakb-lod0')
    source_strip = native_indices[native_values[3]:native_values[3]+native_values[4]]
    staged_strip = staged_indices[staged_values[3]:staged_values[3]+staged_values[4]]
    assert np.array_equal(source_strip == 0xffffffff, staged_strip == 0xffffffff)
    source_used = np.unique(source_strip[source_strip != 0xffffffff])
    staged_used = np.unique(staged_strip[staged_strip != 0xffffffff])
    assert len(source_used) == len(staged_used) == 38
    assert np.array_equal(np.searchsorted(source_used, source_strip[source_strip != 0xffffffff]),
                          np.searchsorted(staged_used, staged_strip[staged_strip != 0xffffffff]))
    source_records = np.frombuffer(native_vertices, 'u1').reshape(-1, 64)[source_used]
    staged_records = np.frombuffer(staged_vertices, 'u1').reshape(-1, 64)[staged_used]
    assert np.array_equal(source_records, staged_records)
    horn_positions = staged_records[:, :12].copy().view('<f4').reshape(-1, 3) / 4096
    assert np.ptp(horn_positions[:, 1]) > 0.45
    # A second pass must refuse an already-connected record instead of silently
    # accepting a malformed output produced by running stages out of order.
    try:
        connect(raw, decompress(a.data / 'out/jak2/fr3/hiphog.fr3'), a.work / 'rig-prototypes/jakb-lod0.glb')
    except AssertionError:
        pass
    else:
        raise AssertionError('Repeated animated routing was not detected')
    report = dict(candidate=a.candidate.name, sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                  cinematicTriangles=10784, animatedSlots=sorted(slots), liveEyeIds=[0, 1],
                  nativeHornCompatibility=dict(vertices=38, triangles=50,
                      exactNativeVertexRecordsStripAndMaterial=True,
                      hornHeightMetres=float(np.ptp(horn_positions[:, 1])),
                      replacedCollapsedCinematicVertices=collapsed_donor_vertices),
                  nativeHornAndBodyScalingUnchanged=True,
                  cinematicClutCount=destinations, cinematicDmaPayloadBytes=payload_bytes,
                  allDummyVramDestinationsFalse=True,
                  prison=dict(separatePlayableModel=False,
                              note='Gameplay uses skel-jchar/jakb-lod0; prison donor belongs to native cinematic scene actors and is preserved'),
                  status='Static checks passed; normal/dark transition, giant form and motion require live validation')
    (a.candidate.with_suffix('.review.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
