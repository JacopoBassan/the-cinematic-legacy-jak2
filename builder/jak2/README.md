# Jak II cinematic builder

This game-specific builder is called by the shared OpenGOAL Launcher installer.
It rebuilds Jak, Daxter, Sig, Ashelin, stadium Samos and stadium Keira from the
user's original local disc assets. Ordinary Samos/Keira scenes already use their
cinematic models, so only their stadium gameplay models are replaced.

The distributed files contain conversion code and compatibility hashes. They
contain no game models, textures, animations, compiled game code or fitted
coefficient arrays. The builder generates those locally and produces exactly
five FR3 files and one scoped GAME.CGO update. Generated payloads must not be
published as source releases.

The builder uses the official OpenGOAL **v0.3.9** toolchain for Windows x64
or Apple Silicon macOS. Both platforms are supported. Intel Macs and Linux
are unsupported.
Compatibility profiles distinguish PAL SCES-51608 and USA SCUS-97265 v2.01 by
the actual ELF and consumed archive hashes. PAL and USA native GAME archives
are pinned separately, including their native territory settings. Other disc
revisions and tool releases are rejected until tested.

The conversion preserves native eye IDs and live gaze/eyelid controls, Daxter's
authored cinematic eye UVs, the original animated Dark Jak horn geometry, the
corrected animated Samos beard, Keira's cuff bindings and native facial channels.
NPC eye resources bind lazily and retain 64-bit owner/load-order caches. Native
Samos's unused legacy stadium morph bank is not activated or rebased.

`build_release.py` exposes `preflight(data, tools, config)` and
`build(data, tools, work, output, config)`. Both require the original installation
and pinned official tools. Build destinations must be new, separate directories
outside the installation. The build never launches the game, edits installed
files, or accesses saves. Temporary selective imports are asset-processing
projects without a game executable or full audio copy.

For a developer rebuild, run from the shared source directory:

```powershell
python -m jak2.build_release --game-data C:\OpenGOAL\active\jak2\data --tools C:\OpenGOAL\active\bin --check
python -m jak2.build_release --game-data C:\OpenGOAL\active\jak2\data --tools C:\OpenGOAL\active\bin --work C:\Builds\jak2-work --output C:\Builds\jak2-payload
```

Offline validation can pass explicit `--source-data` and `--disc-data` roots.
The shipped configuration contains no machine-specific paths. Exact source
dependencies are listed in `source-closure.json`.

Every build validates native texture context, all six models' active bone
bounds and weights, material/visibility/reflection controls, live eye routing,
morph buffers, Dark slots and unrelated native model preservation. Static and
offline build validation do not establish full campaign or platform coverage.
