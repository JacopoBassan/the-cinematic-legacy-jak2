"""Assemble only the four cinematic adapters into an owned Jak II archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

REPLACEMENTS = ('foreground', 'texture-anim', 'process-drawable')
HELPER = 'cinematic-eyes'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entries(raw):
    if len(raw) < 64:
        raise ValueError('Incomplete archive header')
    count = struct.unpack_from('<I', raw)[0]
    if not 1 <= count <= 2000:
        raise ValueError('Unexpected archive object count')
    cursor, result = 64, []
    for _ in range(count):
        if cursor + 64 > len(raw):
            raise ValueError('Incomplete object header')
        size = struct.unpack_from('<I', raw, cursor)[0]
        end = cursor + 64 + ((size + 15) & ~15)
        if not size or end > len(raw):
            raise ValueError('Incomplete object payload')
        encoded = raw[cursor + 4:cursor + 64].split(b'\0', 1)[0]
        try:
            name = encoded.decode('ascii')
        except UnicodeDecodeError as error:
            raise ValueError('Invalid object name') from error
        if not name:
            raise ValueError('Missing object name')
        result.append((name, raw[cursor:end]))
        cursor = end
    return raw[:64], result, raw[cursor:]


def object_record(name, payload):
    if not payload or len(payload) > 16 * 1024 * 1024:
        raise ValueError('Unexpected compiled object size')
    encoded = name.encode('ascii')
    return (struct.pack('<I', len(payload)) + encoded.ljust(60, b'\0')
            + payload + b'\0' * (-len(payload) % 16))


def assemble(original, objects, output):
    original, objects, output = map(lambda p: Path(p).resolve(), (original, objects, output))
    if output == original or output.exists():
        raise ValueError('Use a new output archive outside the original')
    raw = original.read_bytes()
    header, baseline, tail = entries(raw)
    names = [name for name, _ in baseline]
    if HELPER in names or any(names.count(n) != 1 for n in (*REPLACEMENTS, 'eye')):
        raise ValueError('Unsupported original GAME archive')
    replacements = {n: object_record(n, (objects / (n + '.o')).read_bytes())
                    for n in (*REPLACEMENTS, HELPER)}
    result = []
    for name, record in baseline:
        result.append((name, replacements.get(name, record)))
        if name == 'eye':
            result.append((HELPER, replacements[HELPER]))
    new_header = bytearray(header)
    struct.pack_into('<I', new_header, 0, len(result))
    packed = bytes(new_header) + b''.join(record for _, record in result) + tail
    verified_header, verified, verified_tail = entries(packed)
    without_helper = [(n, record) for n, record in verified if n != HELPER]
    unchanged = [(n, record) for n, record in baseline if n not in REPLACEMENTS]
    if (len(without_helper) != len(baseline)
            or any(n != old_n or (n not in REPLACEMENTS and record != old_record)
                   for (n, record), (old_n, old_record) in zip(without_helper, baseline))
            or verified_tail != tail or verified_header[4:] != header[4:]
            or [n for n, _ in verified if n != HELPER] != [n for n, _ in baseline]):
        raise ValueError('An unrelated object, archive field, or order changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(packed)
        if Path(temporary).read_bytes() != packed or original.read_bytes() != raw:
            raise ValueError('Archive write verification failed')
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'sourceSha256': digest(raw), 'outputSha256': digest(packed),
            'sourceObjects': len(baseline), 'outputObjects': len(verified),
            'replacedObjects': list(REPLACEMENTS), 'addedObject': HELPER,
            'unchangedObjects': len(unchanged), 'unrelatedObjectBytesPreserved': True,
            'nativeDuplicateRecordsPreserved': {n: names.count(n) for n in names if names.count(n) > 1},
            'nativeObjectOrderPreserved': True, 'archiveFieldsAndTailPreserved': True,
            'objectSha256': {n: digest(replacements[n]) for n in replacements},
            'runtimeValidation': 'pending'}


def report_path(original, objects, output, report):
    """Reject destructive report aliases before creating the archive."""
    report = Path(report).resolve()
    protected = {Path(original).resolve(), Path(output).resolve()}
    protected.update((Path(objects) / (n + '.o')).resolve()
                     for n in (*REPLACEMENTS, HELPER))
    if report in protected or report.exists():
        raise ValueError('Use a new report path distinct from the archive and compiled inputs')
    return report


def publish_report(path, report):
    """Publish a complete report without replacing a concurrently created file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        # Windows rename refuses an existing target. POSIX rename replaces it,
        # so publish a link to the completed temporary file there instead.
        if os.name == 'nt':
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    destination = report_path(args.original, args.objects, args.output, args.report)
    report = assemble(args.original, args.objects, args.output)
    publish_report(destination, report)
    print(json.dumps(report, indent=2))
