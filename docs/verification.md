# Jak II verification

Release `0.3.7` targets Windows x64, [Apple Silicon macOS](macos.md) and [experimental Linux x64](linux.md) with official OpenGOAL **v0.3.9**. The launcher protocol was checked with **v2.11.2**. Intel Macs are unsupported. Real Linux launcher import, model conversion and gameplay are not yet validated.

## Refreshed HUD

Refreshed HUD 1.2 texture preservation was verified with the supported PAL and USA discs on Windows using OpenGOAL v0.3.9. The same texture handling runs in the macOS and Linux installers.

## Source checks

On 2026-11-11, the game-specific suite completed 66 tests on Apple Silicon: 62 passed and four Windows-only cases were skipped. Twelve portable filesystem fixtures also passed on this Mac.

On Windows, all 66 tests and 13 installer filesystem checks passed on 2026-11-15.

Coverage includes wrong-game inputs, corrupt packages, source inventories, staged writes, rollback, interrupted transactions, hardlinks and lock ownership. The Windows checks also cover junctions, paths and child-process cleanup. These source checks are separate from player testing.

[`builder/source-files.json`](../builder/source-files.json) separates the complete `sourceFiles` developer ZIP from the smaller `runtimeFiles` installer source set. Package checks verify each manifest against its own file set. They also check pinned dependencies, runtime versions, Windows adapter provenance and Mac/Linux executable modes. Game models and textures are excluded from distributable source and installer archives.

The fresh per-game layout has not had a new full player session. Historical coverage follows.

## Linux package and installer checks

Original input checks passed for the supported EU and USA discs with the pinned Linux x64 tools in Ubuntu 24.04 under Rosetta. The required native input fingerprints are recorded in [the Linux profile](../builder/platforms/linux-x64.json). These checks did not run cinematic conversion or gameplay.

The Linux workflow uses Ubuntu 24.04 x64. It builds the final TAR.GZ and checks archive hashes, executable modes, source files, the bundled Python runtime and native dependency operations.

Launcher builds prefer the source shipped with the pinned OpenGOAL tools. The launcher merges source files into the active game folder, where older files can remain. The builder checks the selected source, the player’s disc and generated game files.

Asset-free installer fixtures cover first install, updates, input and save preservation, wrong-game and running-game refusal, cancellation, rollback, child-process cleanup and retry. They use synthetic files and do not drive the real launcher or convert real game assets. They also check transaction marker ownership and inherited runtime settings. The workflow stores a `verification-linux-x64` receipt.

Release assembly passed with both builder and standard archive names. Checks covered one download per platform, metadata, source files and release checksums. Missing Linux input and a changed Linux runtime pin were rejected before output was written. These archive fixtures use synthetic runtime files.

## Model checks

The builder produces replacements for Jak, Daxter, Sig, Ashelin, stadium Samos and stadium Keira. Checks cover eye routing, horn transitions, morphs and texture context. The five graphics outputs retain the 233 native indexed textures. Unrelated models and regional archive objects are checked for preservation.

Jak, Sig and Samos use native upper shadow-volume clipping to prevent their gameplay shadow proxies from cutting through the cinematic models. EU v0.3.9 checks covered normal and Dark Jak, Sig and Samos from multiple angles, with ground shadows retained. The correction changes no geometry, textures, materials or animation controls.

The builder uses OpenGOAL’s `buildinfo.json` to select the disc revision. For older extractions, it reads the `BOOT2` entry in `SYSTEM.CNF`. It checks the original tools, selected executable and required archives before conversion. It validates the generated model data and records payload checksums. Earlier conversions covered the supported EU and USA profiles.

## Historical player coverage

These checks were recorded before the 2026-11-11 repository split.

- **Windows:** Earlier Windows USA checks covered startup, prison loading, running, jumping, Jak and Daxter blink cycles, and Dark Jak with both horns. Targeted NPC checks covered resting lids and blinking for Samos, Keira, Sig and Ashelin. EU conversion and model context were checked offline.
- **Apple Silicon:** Earlier Apple Silicon checks included native builds from supported local inputs and a complete Samos blink cycle.

## Comparisons and image policy

The [gallery](comparisons/jak2.md) contains 19 matched comparisons for Jak II. Lossless compression checks preserved the decoded RGB pixels and image dimensions. The [comparison manifest](comparisons/manifest.json) records file hashes and capture details.

Before-and-after images, including launcher thumbnails, must use unchanged screenshot pixels. Only cropping, placement and lossless compression are allowed. Do not use image generation, redraw details, retouch colors or upscale captures. Thumbnail source hashes and crop rectangles are recorded in [`mod-source-provenance.json`](../mod-source-provenance.json).

## Limits

- Shadows retain the original gameplay hulls and can differ from cinematic contours.
- NPCs keep gameplay expression and gaze behavior. The mod adds no gaze-tracking AI or cinematic dialogue lip-sync.
- Player checks cover representative scenes and animations, not a full campaign or every NPC.
- The complete normal/Dark/normal transition was not sampled in the earlier v0.3.9 player test.
