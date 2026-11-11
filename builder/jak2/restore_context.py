"""Retain native RGBA and indexed global textures after a selective import."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys


def indexed_textures(raw, start):
    """Read format43 IndexTexture records without repacking their CLUT data."""
    position = start
    def take(size):
        nonlocal position
        value = raw[position:position + size]
        if len(value) != size:
            raise ValueError('Truncated indexed texture metadata')
        position += size
        return value
    def count():
        return struct.unpack('<Q', take(8))[0]
    def string():
        return take(count()).decode('utf-8')
    records = []
    for _ in range(count()):
        record_start = position
        width, height, combo = struct.unpack('<HHI', take(8))
        indices = take(count())
        clut = take(256 * 4)
        name, page = string(), string()
        levels = [string() for _ in range(count())]
        records.append({'name': name, 'page': page, 'levels': levels,
                        'width': width, 'height': height, 'combo': combo,
                        'indices': indices, 'clut': clut,
                        'packed': raw[record_start:position]})
    return records, position


def restore(candidate, original, output, work):
    if Path(output).resolve() in (Path(candidate).resolve(), Path(original).resolve()):
        raise ValueError('Use a distinct context output')
    os.environ['CINEMATIC_BUILD_WORK'] = str(Path(work).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stages'))
    from fr3_models import Reader, compress, decompress
    from inspect_transformations import textures
    from npc_builder import texture_bytes
    raw = decompress(candidate)
    current, end = textures(raw)
    original_raw = decompress(original)
    native, native_end = textures(original_raw)
    current_indices, index_end = indexed_textures(raw, end)
    native_indices, native_index_end = indexed_textures(original_raw, native_end)
    # Animated native texture setup looks these records up by name. Preserve
    # the whole ordered native vector, including index bytes, CLUTs and level
    # provenance. Donor-only records follow it, with no ambiguous name aliases.
    native_names = {t['name'] for t in native_indices}
    extras = [t for t in current_indices if t['name'] not in native_names]
    index_vector = (original_raw[native_end:native_index_end] if not extras else
                    struct.pack('<Q', len(native_indices) + len(extras))
                    + b''.join(t['packed'] for t in native_indices + extras))
    existing = {texture_bytes(texture) for texture in current}
    missing = []
    for texture in native:
        packed = texture_bytes(texture)
        if packed in existing:
            continue
        # Native texture IDs must never be stolen by another global image.
        collisions = [t for t in current + missing if t['combo'] == texture['combo'] and t['page'] != 'custom-level']
        if collisions:
            raise ValueError('A native global texture ID has conflicting data: ' + texture['name'])
        missing.append(texture)
        existing.add(packed)
    reader = Reader(raw, 2)
    reader.take(reader.num('Q'))
    fixed = (raw[:reader.pos] + struct.pack('<Q', len(current) + len(missing))
             + b''.join(texture_bytes(t) for t in current + missing)
             + index_vector + raw[index_end:])
    compress(output, fixed)
    checked, checked_end = textures(decompress(output))
    assert checked == current + missing
    checked_indices, checked_index_end = indexed_textures(fixed, checked_end)
    assert checked_indices == native_indices + extras
    assert [t['packed'] for t in checked_indices[:len(native_indices)]] == [t['packed'] for t in native_indices]
    assert fixed[checked_index_end:] == raw[index_end:], 'Context stage changed geometry or morph data'
    checked_keys = {texture_bytes(t) for t in checked}
    assert all(texture_bytes(t) in checked_keys for t in native)
    return {'nativeTextures': len(native), 'existingTextures': len(current), 'appendedNativeTextures': len(missing),
            'allNativeGlobalTexturesPreserved': True, 'existingTextureIndicesUnchanged': True,
            'nativeIndexedTextures': len(native_indices), 'donorOnlyIndexedTextures': len(extras),
            'orderedNativeIndexRecordsExact': True,
            'indexNamesPalettesPixelsAndLevelProvenanceExact': True,
            'geometryMaterialsEyesMorphsAndHornsUnchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'original', 'output', 'work'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    report = restore(args.candidate, args.original, args.output, args.work)
    args.output.with_suffix('.context.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
