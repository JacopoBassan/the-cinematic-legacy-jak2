"""Stage Jak II cinematic NPC geometry against native runtime controls.

This reads the user's extracted art groups and native FR3s. Outputs stay in a
private directory. Live eye adaptation and runtime captures remain required.
"""
import argparse
import json
import os
import re
import shutil
import sys
import struct
from pathlib import Path

import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / 'stages'))


def skeleton_v4(directory, art_name, expected):
    """Decode and validate the original contiguous Jak II v4 joint table."""
    raw = (directory / (art_name + '-ag.go')).read_bytes()
    assert struct.unpack_from('<H', raw, 8)[0] == 4
    # v4 pointers are relative to the data segment immediately after its
    # 16-byte header. Header offset 4 is a link-table size, unlike v5.
    base = 16
    by_id = {int(i) - 1: name for name, i in expected.items()}
    matches = []
    for start in range(base, min(len(raw) - 80 * len(expected), base + 65536), 16):
        if struct.unpack_from('<III', raw, start)[::2] != (0xffffffff, 0):
            continue
        rows = {}
        for i in range(len(expected)):
            at = start + 80 * i
            tag, pointer, index, parent = struct.unpack_from('<4I', raw, at)
            if tag != 0xffffffff or index != i or pointer + base + 4 > len(raw):
                break
            length = struct.unpack_from('<I', raw, pointer + base)[0]
            if not 0 < length < 100:
                break
            name = raw[pointer + base + 4:pointer + base + 4 + length].decode('ascii', errors='replace')
            if name != by_id[i]:
                break
            matrix = np.asarray(struct.unpack_from('<16f', raw, at + 16)).reshape(4, 4).T
            if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
                break
            rows[name] = dict(id=i + 1, inv_bind=matrix,
                             parent=(parent + base - start - 4) // 80 + 1 if parent != 0xffffffff else None)
        if len(rows) == len(expected):
            matches.append(rows)
    assert len(matches) == 1, (art_name, len(matches))
    return matches[0]


def native_joint_limit(directory, art_name, model_name):
    """Read the actual native merc joint-evaluation bound, not node-list length."""
    raw = (directory / (art_name + '-ag.go')).read_bytes()
    assert struct.unpack_from('<H', raw, 8)[0] == 4
    u32 = lambda at: struct.unpack_from('<I', raw, at)[0]
    base = 16
    merc = base + u32(base + 36) - 4
    at = base + u32(merc + 8)
    name = raw[at + 4:at + 4 + u32(at)].decode('ascii')
    assert name == model_name, (art_name, name, model_name)
    return u32(merc + 20) + 2


def build(data, work, output):
    if not output.resolve().is_relative_to(work.resolve()) or output.resolve().is_relative_to((work / 'import').resolve()):
        raise ValueError('NPC output must be a private work subdirectory outside the active import project')
    os.environ['CINEMATIC_BUILD_WORK'] = str(work.resolve())
    import npc_builder as npc
    from fr3_models import decompress, model, DRAW
    base = work / 'import'
    source = base / 'out/jak2/fr3'
    text = (data / 'goal_src/jak2/engine/data/joint-nodes.gc').read_text()
    inventory = json.loads((work / 'variant-audit.json').read_text())['modelInventory']
    jobs = [('sig-lod0', 'sig-highres-lod0', {208: 208}),
            ('ashelin-lod0', 'ashelin-highres-lod0', {200: 200, 201: 201})]
    names = sorted({name for low, high, _ in jobs for name in (low, high)})
    catalog = {}
    for name in names:
        joints = dict(re.findall(r'\(def-joint-node ' + re.escape(name) + r'-jg "([^"]+)" (\d+)\)', text))
        if not joints:
            raise ValueError('Missing native joint metadata: ' + name)
        catalog[name] = dict(name=name, files=[x['file'] for x in inventory[name]], joints=joints)
    npc.ROOT = source
    npc.OUT = output
    npc.CAT = catalog
    npc.FULLCAT = {name: entries for name, entries in inventory.items()}
    raw_objects = base / 'decompiler_out/jak2/raw_obj'
    npc.skeleton = lambda name, expected: skeleton_v4(raw_objects, name, expected)

    original_map = npc.choose_map

    def choose_map(name, supported, low_name):
        return original_map(name, supported, low_name)

    # The native effect layout preserves gameplay visibility, color fades,
    # reflection modes and lens controls. Original Sig has no facial morph
    # channels, so cinematic speech shapes are not attached to unrelated data.
    def donor_effect(low_name, effect, texture):
        if low_name == 'sig-lod0':
            return {0: 3, 1: 3, 2: 0, 3: 2, 4: 1, 5: 3, 6: 4}[effect]
        if low_name == 'ashelin-lod0':
            return {0: 0, 1: 1, 2: 3, 3: 3, 4: 2, 5: 2}[effect]
        raise ValueError(low_name)

    def native_draw_kept(low_name, effect, texture, draw):
        assert low_name in ('sig-lod0', 'ashelin-lod0')
        return False

    npc.choose_map = choose_map
    npc.donor_effect = donor_effect
    npc.native_draw_kept = native_draw_kept
    # Jak II uses fixed cinematic eye banks. Sig's original eye controller is
    # slot104, hence draw eyeID208. Slot0 belongs to Jak and must not be reused.
    output.mkdir(parents=True, exist_ok=True)
    for filename in sorted({file for low, _, _ in jobs for file in catalog[low]['files']}):
        shutil.copy2(source / filename, output / filename)
    rows = []
    for low, high, eyes in jobs:
        rows.extend(npc.build_one(low, high, source_dir=output, eye_slot_map=eyes))
    for row in rows:
        from inspect_faces import buffers
        from npc_builder import active_joints
        raw = decompress(output / row['file'])
        _, indices, _, vertices = buffers(raw, row['gameplay'])
        limit = native_joint_limit(raw_objects, row['gameplay'].removesuffix('-lod0'), row['gameplay'])
        weighted = active_joints(model(raw, row['gameplay']), indices, vertices)
        assert max(weighted) <= limit, (row['gameplay'], max(weighted), limit)
        row['nativeEvaluatedMaxJoint'] = limit
        row['maxWeightedJoint'] = max(weighted)
        row['liveEyeAdapterRequired'] = True
        row['nativeFaceHasMorphChannels'] = False
        row['status'] = 'Private rig candidate; live eyes and in-game animation validation pending'
    (output / 'npc-build-report.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ('file', 'gameplay', 'donor', 'triangles', 'effects', 'status')} for r in rows], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.data, a.work, a.output)


if __name__ == '__main__':
    main()
