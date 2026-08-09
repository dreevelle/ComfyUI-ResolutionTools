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

All values below are **real** megapixels. If you're migrating from a table computed
with the built-in selector, every label in it is ~4.86% low — a row marked "2.0 MP"
there is really 2.09.

### Exact-ratio tables (`exact_ratio = true`)

These are the *complete* set of true-16:9 grid-aligned resolutions. There is nothing
valid between the rows: the lattice is `256n × 144n` at alignment 16 and
`512n × 288n` at alignment 32. Type the listed megapixels and you get that row back.

Krea 2 — `alignment = 16`:

| megapixels | Aspect | Output (alignment=16) |
|---|---|---|
| 0.15 | 16:9 | 512 x 288 |
| 0.33 | 16:9 | 768 x 432 |
| 0.59 | 16:9 | 1024 x 576 |
| 0.92 | 16:9 | 1280 x 720 |
| 1.33 | 16:9 | 1536 x 864 |
| 1.81 | 16:9 | 1792 x 1008 |
| 2.36 | 16:9 | 2048 x 1152 |
| 2.99 | 16:9 | 2304 x 1296 |
| 3.69 | 16:9 | 2560 x 1440 |
| 4.46 | 16:9 | 2816 x 1584 |

MiniMax H3 — `alignment = 32`:

| megapixels | Aspect | Output (alignment=32) |
|---|---|---|
| 0.15 | 16:9 | 512 x 288 |
| 0.59 | 16:9 | 1024 x 576 |
| 1.33 | 16:9 | 1536 x 864 |
| 2.36 | 16:9 | 2048 x 1152 |
| 3.69 | 16:9 | 2560 x 1440 |
| 5.31 | 16:9 | 3072 x 1728 |

Five rows to 4 MP is genuinely all there is on a 32-grid — which is why none of the
commonly circulated H3 resolution tables are actually 16:9.

### Area-target tables (`exact_ratio = false`)

For when you want a specific pixel budget more than a perfect ratio. Each axis is
rounded independently, so the ratio drifts a little.

Krea 2 — `alignment = 16`:

| megapixels | Aspect | Output (alignment=16) |
|---|---|---|
| 0.2 | 16:9 | 592 x 336 |
| 0.3 | 16:9 | 736 x 416 |
| 0.4 | 16:9 | 848 x 480 |
| 0.5 | 16:9 | 944 x 528 |
| 0.6 | 16:9 | 1040 x 576 |
| 0.7 | 16:9 | 1120 x 624 |
| 0.8 | 16:9 | 1200 x 672 |
| 0.9 | 16:9 | 1264 x 704 |
| 1.0 | 16:9 | 1328 x 752 |
| 1.2 | 16:9 | 1456 x 816 |
| 1.4 | 16:9 | 1584 x 880 |
| 1.6 | 16:9 | 1680 x 944 |
| 1.8 | 16:9 | 1792 x 1008 |
| 2.0 | 16:9 | 1888 x 1056 |
| 2.25 | 16:9 | 2000 x 1120 |
| 2.5 | 16:9 | 2112 x 1184 |
| 2.75 | 16:9 | 2208 x 1248 |
| 3.0 | 16:9 | 2304 x 1296 |
| 3.25 | 16:9 | 2400 x 1360 |
| 3.5 | 16:9 | 2496 x 1408 |
| 3.75 | 16:9 | 2576 x 1456 |
| 4.0 | 16:9 | 2672 x 1504 |

MiniMax H3 — `alignment = 32`:

| megapixels | Aspect | Output (alignment=32) |
|---|---|---|
| 0.2 | 16:9 | 608 x 320 |
| 0.3 | 16:9 | 736 x 416 |
| 0.4 | 16:9 | 832 x 480 |
| 0.5 | 16:9 | 928 x 544 |
| 0.6 | 16:9 | 1024 x 576 |
| 0.7 | 16:9 | 1120 x 640 |
| 0.8 | 16:9 | 1184 x 672 |
| 0.9 | 16:9 | 1280 x 704 |
| 1.0 | 16:9 | 1344 x 736 |
| 1.2 | 16:9 | 1472 x 832 |
| 1.4 | 16:9 | 1568 x 896 |
| 1.6 | 16:9 | 1696 x 960 |
| 1.8 | 16:9 | 1792 x 992 |
| 2.0 | 16:9 | 1888 x 1056 |
| 2.25 | 16:9 | 1984 x 1120 |
| 2.5 | 16:9 | 2112 x 1184 |
| 2.75 | 16:9 | 2208 x 1248 |
| 3.0 | 16:9 | 2304 x 1312 |
| 3.25 | 16:9 | 2400 x 1344 |
| 3.5 | 16:9 | 2496 x 1408 |
| 3.75 | 16:9 | 2592 x 1440 |
| 4.0 | 16:9 | 2656 x 1504 |

Ratio drift is under 1% for most Krea 2 rows, but gets bad on H3 at the low end —
6.9% at 0.2 MP, where a 32 px step is a large fraction of a 320 px axis. Below about
0.6 MP on H3, use the exact lattice instead.

### Other ratios at ~2.36 MP (Krea 2, `alignment = 16`)

| megapixels | Aspect | Output (alignment=16) |
|---|---|---|
| 2.36 | 1:1 | 1536 x 1536 |
| 2.36 | 16:9 | 2048 x 1152 |
| 2.36 | 9:16 | 1152 x 2048 |
| 2.22 | 3:2 | 1824 x 1216 |
| 2.22 | 2:3 | 1216 x 1824 |
| 2.24 | 4:3 | 1728 x 1296 |
| 2.24 | 3:4 | 1296 x 1728 |
| 2.37 | 21:9 | 2352 x 1008 |
| 2.37 | 9:21 | 1008 x 2352 |

### Picking a row

**Krea 2** — the official envelope is a 1K–2K long edge, so 2.36 MP (2048×1152) is the
top of sanctioned territory. The 2.99 and 3.69 rows are extrapolation: expect detail
repetition rather than sharper eyes.

**MiniMax H3** — the native canvas is a 768 short edge, 768×1344 ≈ 1.03 MP
(`BASE_SHORT_EDGE` / `MAX_PIXELS` in `comfy_extras/nodes_minimax_h3.py`), and 1344×768
is the node default. 1.33 MP (1536×864) is a mild ~29% stretch and the highest row
worth running by default. 2.36 MP is 2.3× the trained area and 3.69 MP is 3.6×; motion
coherence degrades before sharpness improves.

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
