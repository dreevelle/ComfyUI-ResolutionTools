# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file ComfyUI custom node package. All code lives in `__init__.py`; there
is no build step, test suite, or linter config. Two nodes,
`ResolutionSelectorMP` and `ResolutionAlignToGrid`, both pure integer maths with
no torch/numpy involvement.

Registered through ComfyUI's **V3 schema API**, not the legacy
`NODE_CLASS_MAPPINGS` dict: `comfy_entrypoint()` returns a `ComfyExtension` whose
`get_node_list()` yields `IO.ComfyNode` subclasses. Inputs are declared in
`define_schema()` and arrive as named kwargs to `execute()`.

## Local environment

- ComfyUI checkout: `/home/dreevelle/comfy/ComfyUI`
- Python venv: `/home/dreevelle/comfy-env/bin/python`
- The package is published, and `ComfyUI/custom_nodes/comfyui-resolution-tools`
  (lowercase, the registry id) is a **ComfyUI-Manager install** that tracks
  registry releases. This working tree is not wired into ComfyUI.

To test a change before releasing it, symlink this tree in under its CamelCase
name so it doesn't collide with the Manager install:

```bash
ln -sfn /home/dreevelle/Projects/ComfyUI-ResolutionTools \
        /home/dreevelle/comfy/ComfyUI/custom_nodes/ComfyUI-ResolutionTools
```

**Remove that symlink before doing anything in ComfyUI-Manager for this
package.** Manager's remove/replace path does `shutil.rmtree` on the *resolved*
path, so it deletes the symlink target's contents instead of unlinking — on
2026-08-09 that wiped this working tree, `.git` included, and it had to be
restored by re-cloning. Commit and push before any Manager operation, and never
leave a symlinked dev copy in place while installing or updating the published
version.

## Verifying changes

`__init__.py` imports `comfy_api.latest` at module scope, so it needs ComfyUI on
`sys.path` — but the maths functions are module-level and dependency-free, so
they can be exercised directly:

```bash
cd /home/dreevelle/comfy/ComfyUI && PYTHONPATH=/home/dreevelle/comfy/ComfyUI \
/home/dreevelle/comfy-env/bin/python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('rt', '/home/dreevelle/Projects/ComfyUI-ResolutionTools/__init__.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.solve_exact(9, 16, 16, 2.36e6))   # -> (1152, 2048)
"
```

Two invariants any change to `solve_exact` must preserve, and which are worth
re-fuzzing over random `(ratio, alignment, target)` triples:

1. `width % alignment == 0 and height % alignment == 0`
2. `width * h_ratio == height * w_ratio` (ratio held exactly, no drift)

Known-good anchors: `9:16 @ m=16, 2.36 MP -> 1152x2048`;
`16:9 @ m=32, 1.33 MP -> 1536x864`; `21:9 @ m=16, 2.37 MP -> 2352x1008`.

## Why the maths is what it is

**Real megapixels.** Core's `comfy_extras/nodes_resolution.py:78` uses
`megapixels * 1024 * 1024`. That's the SDXL-era convention where "1 MP" meant
1024x1024. This package deliberately diverges to `* 1_000_000`. Do not "fix"
this to match core — the divergence is the point, and the README documents it.

**Lattice solving.** `lattice_step()` computes the smallest scale increment that
keeps both axes on the alignment grid while holding an exact ratio. Core rounds
each axis independently, which drifts the ratio (9:16 at 2.36 MP becomes 0.5649
instead of 0.5625). `solve_nearest()` preserves core's behaviour for when the
lattice is too coarse; it is opt-in via `exact_ratio=False`.

**Alignment = VAE spatial downscale x DiT patch size.** Verified against the
ComfyUI source rather than assumed:

- Krea 2 -> 16. `latent_formats.LatentFormat.spacial_downscale_ratio = 8`
  (inherited by `Wan21`, which `supported_models.Krea2` uses) x `patch=2` in
  `comfy/ldm/krea2/model.py`.
- MiniMax H3 -> 32. `latent_formats.MiniMaxH3Video.spacial_downscale_ratio = 16`
  x `patch_size=(1, 2, 2)` in `comfy/ldm/minimax/model.py`.
- SDXL/SD1.5 -> 64. UNet, no patchify; three downsample stages on an 8x VAE.

The failure mode when unaligned is `comfy.ldm.common_dit.pad_to_patch_size`,
which pads with `padding_mode="circular"` and appends only at the end — hence a
bottom/right band, never top/left. If a future model changes VAE or patch size,
the alignment table in the README and `ALIGNMENT_TOOLTIP` both need updating.

## Conventions

- Version is duplicated in `__version__` (`__init__.py`) and `[project].version`
  (`pyproject.toml`) — bump both.
- `dependencies` stays empty; this package is stdlib-only by design.
- Node IDs (`ResolutionSelectorMP`, `ResolutionAlignToGrid`) are the public API.
  Renaming one breaks every saved workflow that uses it.

## Releasing

Intended for the Comfy Registry as `comfyui-resolution-tools` under publisher
`dreevelle`, same setup as ComfyUI-FastImageSequence.

**Changing `[project].version` in `pyproject.toml` on `main` publishes a
release.** `.github/workflows/publish_action.yml` watches that path and runs
`Comfy-Org/publish-node-action` with the `REGISTRY_ACCESS_TOKEN` repo secret.
Don't touch the version field unless a release is intended; republishing an
existing version fails with `400 The node version already exists`.

`.comfyignore` keeps `CLAUDE.md` and `.github/` out of the published archive.
`comfy node validate` runs the registry checks locally; `comfy node pack`
produces the exact archive that would be uploaded.
