# Jak II verification

Candidate `0.3.0-rc.1` targets Windows x64 and [Apple Silicon macOS](macos.md) with official OpenGOAL **v0.3.9**. The launcher protocol was checked with **v2.11.2**. Intel Macs and Linux are unsupported.

## Current source checks, 2026-11-11

The game-specific suite completed 66 tests on Apple Silicon: 62 passed and four Windows-only cases were skipped. Twelve portable filesystem fixtures also passed on this Mac. All 13 Windows filesystem fixtures remain configured for native Windows checks.

Coverage includes wrong-game inputs, corrupt packages, source inventories, staged writes, rollback, interrupted transactions, hardlinks and lock ownership. Native Windows junction, path and child-process checks require the Windows run. These source checks are separate from player testing.

[`builder/source-files.json`](../builder/source-files.json) separates the complete `sourceFiles` developer ZIP from the smaller `runtimeFiles` installer source set. Package checks verify each manifest against its own file set. They also check pinned dependencies, runtime versions, Windows adapter provenance and Mac executable modes. Game models and textures are excluded from distributable source and installer archives.

The fresh per-game layout has not had a new full player session. Historical coverage follows.

## Model checks

The builder produces replacements for Jak, Daxter, Sig, Ashelin, stadium Samos and stadium Keira. Checks cover eye routing, horn transitions, morphs and texture context. The five graphics outputs retain the 233 native indexed textures. Unrelated models and regional archive objects are checked for preservation.

The builder checks original tools, disc revisions and consumed assets before conversion. It validates the generated model data and records payload checksums. Earlier conversions covered the supported EU and USA profiles.

## Historical player coverage

These checks were recorded before the 2026-11-11 repository split.

- **Windows:** Earlier Windows USA checks covered startup, prison loading, running, jumping, Jak and Daxter blink cycles, and Dark Jak with both horns. Targeted NPC checks covered resting lids and blinking for Samos, Keira, Sig and Ashelin. EU conversion and model context were checked offline.
- **Apple Silicon:** Earlier Apple Silicon checks included native builds from supported local inputs and a complete Samos blink cycle.

## Comparisons and image policy

The [gallery](comparisons/jak2.md) contains 19 matched comparisons for Jak II. Lossless compression checks preserved the decoded RGB pixels and image dimensions. The [comparison manifest](comparisons/manifest.json) records file hashes and capture details.

Before-and-after images, including launcher thumbnails, must use unchanged screenshot pixels. Only cropping, placement and lossless compression are allowed. Do not use image generation, redraw details, retouch colours or upscale captures. Thumbnail source hashes and crop rectangles are recorded in [`mod-source-provenance.json`](../mod-source-provenance.json).

## Limits

- Shadows retain the original gameplay hulls and can differ from cinematic contours.
- NPCs keep gameplay expression and gaze behavior. The mod adds no gaze-tracking AI or cinematic dialogue lip-sync.
- Player checks cover representative scenes and animations, not a full campaign or every NPC.
- The complete normal/Dark/normal transition was not sampled in the earlier v0.3.9 player test.
