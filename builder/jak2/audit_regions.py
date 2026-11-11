"""Compare selected Jak II EU/USA archives without extracting a second game."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

ARCHIVES = ['ATE.DGO', 'UNB.DGO', 'CTYASHA.DGO', 'STADBLMP.DGO',
            'GARAGE.DGO', 'LINTCSTB.DGO', 'HIPHOG.DGO', 'ORACLE.DGO', 'LDJAKBRN.DGO']


def iso_records(stream, extent, length):
    stream.seek(extent * 2048)
    raw = stream.read(length)
    assert len(raw) == length
    cursor = 0
    records = {}
    while cursor < length:
        size = raw[cursor]
        if size == 0:
            cursor = ((cursor // 2048) + 1) * 2048
            continue
        assert size >= 34 and cursor + size <= length
        record = raw[cursor:cursor + size]
        label = record[33:33 + record[32]]
        if label not in (b'\0', b'\1'):
            name = label.decode('ascii').split(';', 1)[0]
            assert name not in records
            records[name] = (struct.unpack_from('<I', record, 2)[0],
                             struct.unpack_from('<I', record, 10)[0], bool(record[25] & 2))
        cursor += size
    return records


def audit(eu, usa_iso):
    rows = []
    with usa_iso.open('rb') as source:
        source.seek(16 * 2048)
        volume = source.read(2048)
        assert volume[:7] == b'\x01CD001\x01', 'Unsupported ISO primary volume'
        root = volume[156:190]
        directories = iso_records(source, struct.unpack_from('<I', root, 2)[0], struct.unpack_from('<I', root, 10)[0])
        extent, length, folder = directories['DGO']
        assert folder
        records = iso_records(source, extent, length)
        for name in ARCHIVES:
            extent, length, folder = records[name]
            assert not folder
            source.seek(extent * 2048)
            remaining = length
            digest = hashlib.sha256()
            while remaining:
                block = source.read(min(remaining, 1024 ** 2))
                assert block
                digest.update(block)
                remaining -= len(block)
            native = eu / 'DGO' / name
            eu_digest = hashlib.sha256(native.read_bytes()).hexdigest()
            rows.append(dict(file='DGO/' + name, eu=eu_digest, usa=digest.hexdigest(),
                             usaBytes=length, identical=eu_digest == digest.hexdigest()))
    return dict(archives=rows, allIdentical=all(r['identical'] for r in rows),
                secondInstallationOrExtractedArchiveCopiesCreated=False,
                scope='Archive provenance comparison; full region installation and gameplay validation remain separate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eu', type=Path, required=True)
    p.add_argument('--usa-iso', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit(a.eu, a.usa_iso)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
