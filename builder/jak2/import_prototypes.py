"""Import private Jak II candidates into an isolated extraction workspace."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--disc', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--extractor', type=Path, required=True)
    parser.add_argument('--project', type=Path, help='New selective import directory, outside the original extraction')
    args = parser.parse_args()
    work = args.work.resolve()
    project = args.project.resolve() if args.project else work/'import'
    if project.exists():
        raise FileExistsError('Import workspace already exists; inspect it before rerunning')
    if shutil.disk_usage(work).free < 1024**3:
        raise RuntimeError('At least 1 GiB free is required for the isolated import')
    project.mkdir()
    for folder in ('goal_src', 'decompiler', 'game'):
        shutil.copytree(args.data/folder, project/folder)
    replacements = project/'custom_assets/jak2/merc_replacements'
    replacements.mkdir(parents=True)
    for name in ('jakb-lod0', 'daxter-lod0'):
        shutil.copy2(work/f'rig-prototypes/{name}.glb', replacements/f'{name}.glb')
    inputs = dict(dgo_names=['CGO/KERNEL.CGO', 'CGO/GAME.CGO', 'CGO/COMMON.CGO'],
                  levels_to_extract=[], str_texture_file_names=[], str_file_names=[],
                  object_file_names=[], streamed_audio_file_names=[])
    input_path = work/'import-inputs.json'
    input_path.write_text(json.dumps(inputs))
    override = dict(inputs_file=input_path.as_posix(), rip_levels=False,
                    dump_objs=False, process_game_text=False, process_game_count=False,
                    process_part_group_table=False, extract_collision=False)
    with (work/'import.log').open('w') as log:
        result = subprocess.run([str(args.extractor.resolve()), str(args.disc.resolve()),
                                 '--folder', '--game', 'jak2', '--proj-path', str(project),
                                 '--decompile', '--decomp-config-override', json.dumps(override),
                                 '--disable-ansi'], stdout=log, stderr=subprocess.STDOUT)
    print((work/'import.log').read_text(errors='replace')[-4000:])
    if result.returncode:
        raise SystemExit(result.returncode)
    if not (project/'out/jak2/fr3/GAME.fr3').is_file():
        raise RuntimeError('Extractor succeeded without producing the expected GAME.fr3')
    print('Isolated import complete; materials and native facial channels still require reconstruction')


if __name__ == '__main__':
    main()
