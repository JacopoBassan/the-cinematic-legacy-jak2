"""Generate live Jak II NPC eyes using locally extracted cinematic resources.

The generated GOAL and copied textures are private game assets. This file ships
only extraction and control logic. Jak II uses fixed native cinematic eye banks.
"""
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / 'stages'))
SPECS = [
    ('sig', 'sig-highres', 'atollext.fr3', 'sig-eye', 'sig-eyelid', ['sig-lod0'], 104),
    ('ashelin', 'ashelin-highres', 'atollext.fr3', 'ashelin-eye', 'ashelin-eyelid', ['ashelin-lod0'], 100),
    ('samos', 'samos-highres', 'stadblmp.fr3', 'samos-eye', 'samos-eyelid', ['samos-lod0'], 108),
    ('keira', 'keira-highres', 'garage.fr3', 'keira-iris-64x64', 'keira-eyelid', ['keira-lod0'], 72),
]


def automatic_blink_lid(lid):
    """Keep the authored closure while selecting the native blink convention.

    Native negative lids mean closure=1+lid, blended toward 1 by the blink.
    Nonnegative lids are direct closures, so encode them as closure-1 instead.
    A fully closed lid remains closed, independent of the blink coefficient.
    """
    if not math.isfinite(lid) or not -1.0 <= lid <= 1.0:
        raise ValueError(f'Unsupported authored eyelid value: {lid}')
    return lid if lid < 0.0 or lid == 1.0 else lid - 1.0


def one_frame_blink_lids(frames):
    """Read each signed lid byte without changing the private packed track."""
    if len(frames) != 2:
        raise ValueError('Automatic idle blink requires one left/right frame')
    authored = [struct.unpack('<b', bytes([(word >> 16) & 255]))[0] / 128.0
                for word in frames]
    return authored, [automatic_blink_lid(lid) for lid in authored]


def eye_data(path):
    raw = path.read_bytes()
    assert struct.unpack_from('<H', raw, 8)[0] == 4
    base = 16
    u32 = lambda at: struct.unpack_from('<I', raw, at)[0]

    def name(pointer):
        at = base + pointer
        size = u32(at)
        assert 0 < size < 256
        return raw[at + 4:at + 4 + size].decode('ascii')

    merc = base + u32(base + 36) - 4
    ctrl = base + u32(merc + 96)
    assert base < ctrl < len(raw) - 496
    slot, _, count = struct.unpack_from('<3B', raw, ctrl)
    assert count in (3, 6)
    shaders = list(struct.unpack_from('<60Q', raw, ctrl + 16))
    candidates = []
    for i in range(3, u32(base + 12)):
        at = base + u32(base + 32 + 4 * i) - 4
        if not base <= at <= len(raw) - 48:
            continue
        label = name(u32(at + 8))
        # art-joint-anim::eye-anim occupies its original reserved header word.
        ep = u32(at + 4)
        if ep in (0, 0xffffffff) or 'idle' not in label:
            continue
        assert base <= base + ep <= len(raw) - 8
        maximum = struct.unpack_from('<h', raw, base + ep)[0]
        assert 0 <= maximum < 2000
        frames = struct.unpack_from('<' + str((maximum + 1) * 2) + 'Q', raw, base + ep + 8)
        candidates.append((label, frames))
    assert candidates, (path, 'No cinematic idle eye data')
    label, frames = min(candidates, key=lambda x: (x[0] != path.stem.removesuffix('-ag') + '-idle', len(x[0])))
    return slot, count, shaders, frames, label


def patch_source(source):
    old = '''    (if (logtest? (-> this skel status) (joint-control-status eye-anim-valid eye-anim))
        (merc-eye-anim this)
        )'''
    if source.count(old) != 1:
        raise ValueError('Unsupported Jak II process-drawable eye dispatch')
    return source.replace('(in-package goal)', '(in-package goal)\n(define-extern j2-cinematic-eye-anim (function process-drawable none))', 1).replace(old, '    (j2-cinematic-eye-anim this)', 1)


def generate(data, work, directory, output, only, source_only=False):
    if not directory.resolve().is_relative_to(work.resolve()) or directory.resolve().is_relative_to((work / 'import').resolve()):
        raise ValueError('Eye staging must remain outside the active import project')
    from fr3_models import Reader, decompress, compress
    from inspect_transformations import textures
    from npc_builder import texture_bytes
    source = work / 'import/out/jak2/fr3'
    raw_dir = work / 'import/decompiler_out/jak2/raw_obj'
    inventory = json.loads((work / 'variant-audit.json').read_text())['modelInventory']
    lines = ['(in-package goal)',
        ';; GAME links this module before texture pages. Bind only from a live update.',
        ';; Never add these private shaders to native texture-page link chains.',
        '(defun j2-cinematic-eye-ready? ((p process-drawable))',
        '  (and (valid? p process-drawable (the-as string #f) #f 0)',
        '       (valid? (-> p draw) draw-control (the-as string #f) #f 0)',
        '       (valid? (-> p draw mgeo) merc-ctrl (the-as string #f) #f 0)',
        '       (valid? (-> p skel) joint-control (the-as string #f) #f 0)',
        '       (valid? *level* level-group (the-as string #f) #f 0)',
        '       (valid? *texture-page-dir* texture-page-dir (the-as string #f) #f 0)',
        '       (valid? *eye-control-arrays* eye-control-arrays',
        '               (the-as string #f) #f 0)',
        '       (< (-> p draw level-index) 7)',
        '       (valid? (-> *level* level (-> p draw level-index)) level',
        '               (the-as string #f) #f 0)))']
    report, dispatch = [], []
    for si, (key, art, filename, iris, lid, actors, expected_slot) in enumerate(SPECS):
        if only and key not in only:
            continue
        tx, _ = textures(decompress(source / filename))
        selected = [next(t for t in tx if t['name'] == name) for name in (iris, lid)]
        slot, count, shaders, frames, label = eye_data(raw_dir / (art + '-ag.go'))
        assert slot == expected_slot and slot // 20 < 6 and slot % 20 < 16
        for t in selected:
            assert t['width'] in (64, 128, 256) and t['height'] in (64, 128, 256)
            assert len(t['pixels']) == t['width'] * t['height'] * 4
        pages = {}
        for ti, t in enumerate(selected):
            pages.setdefault(t['combo'] >> 16, []).append((t['combo'] & 65535, t, ti))
        for page, items in pages.items():
            length = max(i for i, _, _ in items) + 1
            name = f'j2-cin-{key}-page-{page}'
            lines.append(f'(define {name} (new \'static \'texture-page :name "{items[0][1]["page"]}" :id {page} :length {length} :data (new \'static \'array texture {length} ' + ' '.join(['#f'] * length) + ')))')
            for idx, t, ti in items:
                tbp = 0x3000 + si * 8 + ti * 4
                lines.append(f'(set! (-> {name} data {idx}) (new \'static \'texture :w {t["width"]} :h {t["height"]} :num-mips 1 :tex1-control 9 :psm 19 :name "{t["name"]}" :dest (new \'static \'array uint16 7 {tbp} 0 0 0 0 0 0) :width (new \'static \'array uint8 7 {max(2, t["width"] // 64)} 0 0 0 0 0 0)))')
        for suffix, values in [('shaders', shaders), ('frames', frames)]:
            lines.append(f'(define j2-cin-{key}-{suffix} (new \'static \'array uint64 {len(values)} ' + ' '.join(f'#x{x:x}' for x in values) + '))')
        lines.extend([f'(define j2-cin-{key}-initialized #f)',
            ';; Symbol cells are 32-bit. Owner handles and load-order need 64 bits.',
            f'(define j2-cin-{key}-owner-cache (new \'static \'array uint64 1 0))',
            f'(define j2-cin-{key}-level-slot -1)',
            f'(define j2-cin-{key}-order-cache (new \'static \'array int64 1 -1))',
            f'(define j2-cin-{key}-login (the-as bsp-header #f))',
            f'(define j2-cin-{key}-clock 0.0)',
            f'(defun j2-cin-{key}-initialize ((p process-drawable))',
            '  (let ((lev (-> *level* level (-> p draw level-index)))',
            '        (owner (process->handle p))',
            f'        (ctrl (-> *eye-control-arrays* data {slot // 20} data {slot % 20})))',
            f'    (when (or (!= (-> j2-cin-{key}-owner-cache 0) (the-as uint owner))',
            '              (!= (-> ctrl process) owner)',
            f'              (!= j2-cin-{key}-level-slot (-> p draw level-index))',
            f'              (!= (-> j2-cin-{key}-order-cache 0) (-> lev load-order))',
            f'              (!= j2-cin-{key}-login (-> *level* log-in-level-bsp)))',
            f'      (set! j2-cin-{key}-initialized #f))',
            f'    (unless j2-cin-{key}-initialized',
            '      (let ((pupil (lookup-texture-by-id-fast',
            f'              (-> (the-as (inline-array adgif-shader) j2-cin-{key}-shaders)',
            '                  1 texture-id))))',
            '        (when (valid? pupil texture (the-as string #f) #f 0)'])
        for i in range(count):
            shader = f'(-> (the-as (inline-array adgif-shader) j2-cin-{key}-shaders) {i})'
            lines.append(f'          (logclear! (-> {shader} link-test) (link-test-flags needs-log-in bit-9))')
            if i % 3 == 1:
                lines.append(f'          (adgif-shader<-texture! {shader} pupil)')
            else:
                t = selected[0 if i % 3 == 0 else 1]
                lines.append(f'          (adgif-shader<-texture! {shader} (-> j2-cin-{key}-page-{t["combo"] >> 16} data {t["combo"] & 65535}))')
        lines.extend([f'          (set! j2-cin-{key}-level-slot (the-as int (-> p draw level-index)))',
            f'          (set! (-> j2-cin-{key}-owner-cache 0) (the-as uint owner))',
            f'          (set! (-> j2-cin-{key}-order-cache 0) (-> lev load-order))',
            f'          (set! j2-cin-{key}-login (-> *level* log-in-level-bsp))',
            '          ;; Native uint16 countdown must not begin at zero.',
            '          (set! (-> ctrl random-time) (the-as uint 60))',
            '          (set! (-> ctrl blink) 0.0)',
            '          (set! (-> ctrl process) owner)',
            f'          (set! j2-cin-{key}-clock 0.0)',
            f'          (set! j2-cin-{key}-initialized #t)))))',
            f'  j2-cin-{key}-initialized)',
            f'(defun j2-cin-{key}-update ((p process-drawable))',
            f'  (when (j2-cin-{key}-initialize p)',
            f'  (let ((ctrl (-> *eye-control-arrays* data {slot // 20} data {slot % 20})))',
            '    (set! (-> ctrl process) (process->handle p))',
            '    (set! (-> ctrl draw-flag) #t)',
            f'    (set! (-> ctrl different-eyes) {"#t" if count == 6 else "#f"})',
            f'    (set! (-> ctrl shaders) (the-as (inline-array adgif-shader) j2-cin-{key}-shaders))',
            '    (set! (-> ctrl bucket) (the-as uint (vu1-bucket-map',
            '      (the-as int (-> *level* draw-index-map (-> p draw level-index)))',
            '      (the-as int (-> p draw default-texture-page)) (merc-mode eye))))',
            '    (logior! (-> p skel status) (joint-control-status eye-anim))',
            f'    (+! j2-cin-{key}-clock (* 30.0 (-> p clock seconds-per-frame)))',
            f'    (let ((frame (mod (the int j2-cin-{key}-clock) {len(frames) // 2})))',
            f'      (convert-eye-data (-> ctrl left) (-> j2-cin-{key}-frames (* frame 2)))',
            f'      (convert-eye-data (-> ctrl right) (-> j2-cin-{key}-frames (+ 1 (* frame 2)))))'])
        blink_lids = None
        if len(frames) == 2:
            authored_lids, blink_lids = one_frame_blink_lids(frames)
            for side, lid in zip(('left', 'right'), blink_lids):
                lines.append(f'    (set! (-> ctrl {side} lid) {lid!r})')
        lines.append('  )')
        for page in pages:
            lines.append(f'  (__pc-texture-upload-now j2-cin-{key}-page-{page} (tex-upload-mode seg0-1-2))')
        lines.append('  ) (none))')
        for actor in actors:
            dispatch.append(f'      ((string= name "{actor}") (j2-cin-{key}-update p))')
        touched = sorted({entry['file'] for actor in actors for entry in inventory[actor]})
        for filename in ([] if source_only else touched):
            path = directory / filename
            raw = decompress(path)
            current, end = textures(raw)
            missing = [t for t in selected if not any(t['combo'] == x['combo'] and t['pixels'] == x['pixels'] for x in current)]
            for t in missing:
                assert not any(t['combo'] == x['combo'] for x in current), 'Eye texture ID collision'
            reader = Reader(raw, 2)
            reader.take(reader.num('Q'))
            start = reader.pos
            fixed = raw[:start] + struct.pack('<Q', len(current) + len(missing)) + b''.join(texture_bytes(t) for t in current + missing) + raw[end:]
            compress(path, fixed)
            assert decompress(path) == fixed
        report.append(dict(character=key, originalSlot=slot, availableDrawEyeIds=[slot * 2] if count == 3 else [slot * 2, slot * 2 + 1],
                           controllerBank=slot // 20, controllerIndex=slot % 20, animation=label, frames=len(frames) // 2,
                           autoBlink=len(frames) == 2, textures=[t['name'] for t in selected], files=touched,
                           authoredRestingLids=authored_lids if blink_lids is not None else None,
                           automaticBlinkLids=blink_lids,
                           lazyShaderBinding=True, sourceOnly=source_only))
    lines.extend(['(defun j2-cinematic-eye-anim ((p process-drawable))',
        '  (when (j2-cinematic-eye-ready? p)',
        '    (let ((name (-> p draw mgeo name)))', '      (cond'] + dispatch + [
        '      (else (when (logtest? (-> p skel status) (joint-control-status eye-anim-valid eye-anim))',
        '        (merc-eye-anim p))))))', '  (none))'])
    output.write_text('\n'.join(lines) + '\n')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    relative = Path('goal_src/jak2/engine/process-drawable/process-drawable.gc')
    output.with_name('process-drawable-cinematic.gc').write_text(patch_source((data / relative).read_text()))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', nargs='+', choices=[x[0] for x in SPECS])
    p.add_argument('--source-only', action='store_true', help='Write GOAL only; never recompress or modify model files')
    a = p.parse_args()
    os.environ['CINEMATIC_BUILD_WORK'] = str(a.work.resolve())
    print(json.dumps(generate(a.data, a.work, a.directory, a.output, a.only, a.source_only), indent=2))


if __name__ == '__main__':
    main()
