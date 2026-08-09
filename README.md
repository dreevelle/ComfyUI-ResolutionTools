# ComfyUI-ResolutionTools

Resolution maths for latent-space models, done exactly.

Two nodes, no dependencies:

| Node | Does |
|---|---|
| **Resolution Selector (Real MP)** | aspect ratio + real megapixels → grid-aligned width/height, ratio held exact |
| **Align Resolution to Grid** | snap an arbitrary size onto the grid (1920×1080 → 1920×1088) |

## Why

### 1. "Megapixels" usually isn't megapixels

ComfyUI's built-in Resolution Selector computes `megapixels * 1024 * 1024`. That's
**mebipixels** — every value you type is inflated 4.86% in area. Ask for 2.36 MP at
9:16 and you get 1184×2096 = 2.48 real MP.

This node uses `megapixels * 1_000_000`. Ask for 2.36 and you get 2.36.

### 2. Rounding each axis independently drifts the aspect ratio

That same 1184×2096 has a ratio of 0.5649. True 9:16 is 0.5625. Both axes were
individually snapped to a multiple of 16, and the ratio fell between the cracks.

With `exact_ratio` on (the default), this node solves on the **lattice of sizes that
are simultaneously grid-aligned and exactly on-ratio**. For a reduced ratio `a:b` and
alignment `m`, those are exactly `(a·L·n, b·L·n)` where

```
L = lcm(m / gcd(a, m), m / gcd(b, m))
```

so it picks `n = round(sqrt(target_px / (a·b·L²)))`. For 9:16 at m=16 the lattice is
`144n × 256n`, and n=8 gives **1152×2048** — exact 9:16, on the grid, 2.36 MP.

Turn `exact_ratio` off to get the old independent-rounding behaviour when you'd rather
hit the area target precisely and don't care about a hair of ratio drift.

### 3. Getting alignment wrong bands the bottom of your image

A latent-space DiT reaches pixels through two reductions:

```
pixels ──/ VAE spatial downscale ──> latent ──/ DiT patch size ──> tokens
```

Both axes must be multiples of `vae_downscale × patch_size`:

| Models | Alignment | |
|---|---|---|
| Krea 2, Flux, SD3, Qwen-Image, Wan | **16** | 8 × 2 |
| MiniMax H3 | **32** | 16 × 2 |
| SDXL, SD1.5 (UNet, 3 downsample stages) | **64** | universally safe |

Miss it and `comfy.ldm.common_dit.pad_to_patch_size` pads the latent up to the patch
grid using **circular** padding — the pad row is a wrapped copy of the *top* of the
image. That fabricated row gets folded into the same patch token as the real bottom
row, a combination the model never saw in training, so the final latent row decodes to
garbage: a band along the bottom edge, 8 px tall on Krea 2, 16 px on H3. Padding is
only ever appended, so the artifact is always bottom/right, never top/left.

This is why 1920×1080 bands on Krea 2 (1080/8 = 135 latent rows, odd) and 1920×1088
doesn't (1088/8 = 136, even). The built-in selector defaults `multiple` to 8, which
walks you straight into it.

## Reference resolutions

Krea 2 (alignment 16, ~2.36 MP — Turbo's sanctioned 1K–2K band):

| Ratio | Resolution | Real MP |
|---|---|---|
| 1:1 | 1536 × 1536 | 2.36 |
| 16:9 | 2048 × 1152 | 2.36 |
| 9:16 | 1152 × 2048 | 2.36 |
| 3:2 | 1824 × 1216 | 2.22 |
| 4:3 | 1728 × 1296 | 2.24 |
| 21:9 | 2352 × 1008 | 2.37 |

MiniMax H3 (alignment 32; native canvas is 768 short edge / 1.03 MP, so treat anything
above ~1.3 MP as extrapolation):

| Ratio | Resolution | Real MP |
|---|---|---|
| 16:9 | 1024 × 576 | 0.59 |
| 16:9 | 1536 × 864 | 1.33 |
| ~1.75:1 | 1344 × 768 | 1.03 (node default, exactly the trained canvas) |

Note that on a 32-grid, exact 16:9 only exists at `512n × 288n` — none of the commonly
circulated H3 resolution tables are actually 16:9.

## Nodes

### Resolution Selector (Real MP)

**Inputs** — `aspect_ratio` (11 presets + Custom), `megapixels` (real, 10⁶),
`alignment` (default 16), `exact_ratio` (default on), `custom_w` / `custom_h` (advanced,
for the Custom ratio).

**Outputs** — `width`, `height`, `megapixels` (what it actually produced), `label`
(`"1152x2048"`, useful for filename prefixes).

### Align Resolution to Grid

**Inputs** — `width`, `height`, `alignment`, `mode` (`round` / `up` / `down`).

**Outputs** — `width`, `height`, `changed` (true if anything was off-grid), `label`.

Use `up` when you must not lose framing, `down` when you must not exceed a budget.

## Install

Via ComfyUI-Manager, or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dreevelle/ComfyUI-ResolutionTools
```

No dependencies, no restart quirks — it's pure stdlib maths.

## License

MIT
