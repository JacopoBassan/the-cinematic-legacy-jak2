"""Import native-rig Jak II NPC prototypes into a private selective project."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'disc', 'work', 'extractor'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--resume-failed', action='store_true', help='Resume configuration-only failed setup without replacing inputs')
    a = p.parse_args()
    project = a.work.resolve() / 'npc-import'
    if project.exists() and not (a.resume_failed and not (project/'out').exists()):
        raise FileExistsError('Private NPC import project already exists; inspect before rerunning')
    project.mkdir(exist_ok=True)
    if not (project/'decompiler').exists():
        shutil.copytree(a.data / 'decompiler', project / 'decompiler')
    replacements = project / 'custom_assets/jak2/merc_replacements'
    replacements.mkdir(parents=True, exist_ok=True)
    for name in ('samos-lod0', 'keira-lod0'):
        source = a.work / 'npc-samos-keira-prototypes' / (name + '.glb')
        manifest = json.loads(source.with_suffix('.provenance.json').read_text())
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['sha256']
        assert manifest['allActiveJointsWithinOriginalSubmission']
        assert manifest['nativeJointOrderAndInverseBindsExact']
        shutil.copy2(source, replacements / source.name)
    inputs = dict(dgo_names=['CGO/KERNEL.CGO', 'CGO/GAME.CGO', 'DGO/STADBLMP.DGO'],
                  levels_to_extract=['STADBLMP.DGO'], common_tpages=[3219],
                  str_texture_file_names=[], str_file_names=[], object_file_names=[],
                  streamed_audio_file_names=[])
    input_path = project / 'selected-inputs.json'
    input_path.write_text(json.dumps(inputs))
    override = dict(inputs_file=input_path.as_posix(), rip_levels=False, dump_objs=False,
                    process_game_text=False, process_game_count=False,
                    process_part_group_table=False, find_functions=False, extract_collision=False)
    log_path = a.work / 'npc-import.log'
    with log_path.open('w') as log:
        result = subprocess.run([str(a.extractor.resolve()), str(a.disc.resolve()),
            '--folder', '--game', 'jak2', '--proj-path', str(project), '--decompile',
            '--decomp-config-override', json.dumps(override), '--disable-ansi'],
            stdout=log, stderr=subprocess.STDOUT)
    print(log_path.read_text(errors='replace')[-4000:])
    if result.returncode:
        raise SystemExit(result.returncode)
    assert (project / 'out/jak2/fr3/stadblmp.fr3').is_file()
    print('Private native-rig NPC import complete; materials and faces remain separate stages')


if __name__ == '__main__':
    main()
