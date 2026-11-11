# ComfyUI-ResolutionTools

Resolution maths for latent-space models, done exactly.

Three nodes, no dependencies:

| Node | Does |
|---|---|
| **Resolution Preset** | pick a known-good resolution from a list — start here |
| **Resolution Selector (Real MP)** | aspect ratio + real megapixels → grid-aligned width/height, ratio held exact |
| **Align Resolution to Grid** | snap an arbitrary size onto the grid (1920×1080 → 1920×1088) |

```
Resolution Preset
┌────────────────────────────────────────────────────┐
│ preset   16:9 · 2048 × 1152 · 2.36 MP · K2+H3+SDXL │
└────────────────────────────────────────────────────┘
   ↳ width 2048   height 1152   megapixels 2.36   label "2048x1152"
```

## Usage

Pick a preset, wire `width` and `height` into whatever takes dimensions. That's the
whole thing:

```
Resolution Preset ──┬── width ──▶ Empty Latent · width
                    └── height ─▶ Empty Latent · height
```

Either empty-latent node works. `EmptySD3LatentImage` is the nominally correct one for
Krea 2 (`[B, 16, H//8, W//8]`, matching its 16-channel Wan21 format), but the legacy
4-channel `EmptyLatentImage` is fine too: `comfy/sample.py` runs
`fix_empty_latent_channels`, which detects an all-zero latent and repeats it out to the
model's channel count. Same for the MiniMax H3 nodes — feed `width`/`height` straight
into `MiniMax H3 Image to Video`.

`megapixels` and `label` are there when useful and cost nothing left unconnected.
`label` gives `"2048x1152"`, handy wired into a `filename_prefix` so saved files carry
their resolution.

### Three concrete pipelines

**Stills for a 2560×1440 screen** — generate above the delivery size on the exact-16:9
lattice, then downscale once at the very end. With P-H3 Image, pick its 16:9 8 MP bucket:

```
Resolution Preset  16:9 · 3584 × 2016 · 7.23 MP · K2+H3 · P-H3 8 MP
        │ width, height
        ▼
p-h3-image Conditioning → KSampler → VAEDecode
        ▼
Upscale Image      lanczos · 2560 × 1440 · crop disabled
        ▼
Save Image
```

With Krea 2, 2560×1440 is already past its sanctioned 2K long edge, so there is no
over-size row to supersample from: either generate `16:9 · 2560 × 1440` natively and
accept the extrapolation, or generate `2048 × 1152` and upscale ×1.25 with an upscale
model. See [Picking a row](#picking-a-row).

**MiniMax H3 video** — same front end into the H3 nodes, on `1536 × 864` (nearest H3's
1.03 MP native canvas, and the longest clips) or `2048 × 1152` (more detail, shorter
ceiling). Both sit *below* 1440p, so a clip is fitted up ×5/3 or ×1.25 — at display
time, or with an upscale model; generating H3 at 2560 × 1440 is 3.6× its trained area.
A P-H3 still makes a clean first frame for either row: see
[the hand-off table](#why-it-matters-the-hand-off-to-h3-video-and-25601440).

**Krea 2 Edit character reference** — pick a portrait preset and feed it at native
resolution, with **no** `Upscale Image` on that branch:

```
Resolution Preset  9:16 · 1152 × 2048 · 2.36 MP · K2+H3+SDXL
        │ width, height
        ▼
   (generate) → VAEEncode → ReferenceLatent → Edit Model Reference Method
```

Reference and generation resolutions are independent — the DiT gives each reference its
own slot in the positional ids, so a portrait reference conditioning a landscape
generation is the intended design. Do not downscale a reference to your delivery size:
`ReferenceLatent` never resizes, so whatever you hand it is what conditions the model.

### P-H3 image generation and editing

In **Resolution Preset**, choose an entry ending with `P-H3 8 MP` or `P-H3 4 MP edits`
and wire `width` and `height` into `p-h3-image Conditioning`. Use the 8 MP entries for
generation or editing, and the 4 MP entries for editing.

The buckets come from
[Pruna's recommended canvases](https://huggingface.co/PrunaAI/P-H3-Image-Comfy#recommended-resolutions),
but several of Pruna's published shapes are not the ratio they're listed under:
3872 × 2176 is 1.7794, not 16:9, and 3360 × 2496 is 35:26, not 4:3. So each preset is the
**largest exact-ratio size on P-H3's grid (32) that does not exceed Pruna's area** for
that bucket:

| Ratio | Pruna 8 MP | **P-H3 8 MP** | Pruna 4 MP edits | **P-H3 4 MP edits** |
|---|---|---|---|---|
| 1:1 | 2912 × 2912 | **2912 × 2912** · 8.48 MP | 2048 × 2048 | **2048 × 2048** · 4.19 MP |
| 4:3 | 3360 × 2496 ✗ | **3328 × 2496** · 8.31 MP | 2368 × 1760 ✗ | **2304 × 1728** · 3.98 MP |
| 3:2 | 3552 × 2368 | **3552 × 2368** · 8.41 MP | 2496 × 1664 | **2496 × 1664** · 4.15 MP |
| 16:9 | 3872 × 2176 ✗ | **3584 × 2016** · 7.23 MP | 2720 × 1536 ✗ | **2560 × 1440** · 3.69 MP |

Portrait entries are the same sizes transposed. ✗ marks a published shape that is off-ratio.

#### Why it matters: the hand-off to H3 video and 2560×1440

A P-H3 still is usually not the end of the pipeline. Two things consume it, and both need
it exactly 16:9:

- **`MiniMax H3 Image to Video`** resizes `first_frame` with crop *disabled* — a straight
  stretch onto the video canvas. Every H3 video row (1536 × 864, 2048 × 1152, …) is on the
  exact-16:9 lattice `512n × 288n`, so an off-ratio seed gets squashed to fit.
- **A fixed 2560 × 1440 target** (e.g. an asset pipeline that rejects anything not exactly
  16:9) is on that same lattice: 2560 × 1440 is n = 5.

Every 16:9 P-H3 preset is on that lattice too, so the whole chain is uniform scaling, same
factor on both axes, no crop, no stretch:

| P-H3 still | → 1536 × 864 (H3) | → 2048 × 1152 (H3) | → 2560 × 1440 |
|---|---|---|---|
| 2560 × 1440 · `P-H3 4 MP edits` | ×3/5 | ×4/5 | native |
| 3072 × 1728 · `K2+H3+SDXL` row | **×1/2** | ×2/3 | ×5/6 |
| 3584 × 2016 · `P-H3 8 MP` | ×3/7 | ×4/7 | ×5/7 |
| ~~3872 × 2176~~ (Pruna, old preset) | stretched 0.09% | stretched 0.09% | rejected / cropped |

`3584 × 2016` is the default 16:9 pick: the closest exact 16:9 to P-H3's trained 8 MP area
without going over. `3072 × 1728` (in the regular list, not tagged P-H3) is worth knowing
about when the still is mostly a seed for the 1536 × 864 video row — it halves exactly.
`2560 × 1440` is the edit size, and is already the delivery size.

All P-H3 sizes land on grid 32. The MiniMax H3 video sizing advice elsewhere in this
README applies to the base video model, not P-H3 Image. Compatibility tags describe grid
alignment, not a model's recommended canvas.

> **Changed in 1.5.0.** Earlier versions listed Pruna's shapes verbatim, with `(approx)`
> on the off-ratio ones. Saved workflows using a `16:9 (approx)`, `9:16 (approx)`,
> `4:3 (approx)` or `3:4 (approx)` P-H3 entry need the preset re-selected.

### When to use the other two nodes

**Resolution Selector (Real MP)** — a ratio or size the preset list doesn't carry: a
custom ratio, an exact pixel budget, or 21:9 at 2352×1008 (on grid 16, while the 21:9
presets use grid 64). It's an alternative to the preset node, not a stage after it;
chaining them is meaningless, since it takes a ratio and a budget rather than
dimensions.

**Align Resolution to Grid** — dimensions you didn't choose. An img2img or ControlNet
source, or an upscale chain where multiplying by 1.5 lands off-grid.

## Why

### 1. "Megapixels" usually isn't megapixels

ComfyUI's built-in Resolution Selector computes `megapixels * 1024 * 1024`. That's
**mebipixels** — every value you type is inflated 4.86% in area. Ask for 2.36 MP at
9:16 and you get 1184×2096 = 2.48 real MP.

This node uses `megapixels * 1_000_000`. Ask for 2.36 and you get 2.36.

**So convert before you copy a number out of a model's docs.** A megapixel figure in
someone else's guide is almost always in the built-in node's unit; multiply by 1.049 to
get the value to type here. ComfyUI's own MiniMax H3 tutorial, for instance, says to set
Megapixels to `0.98` — that is **1.03 real MP**. With `exact_ratio` on, the lattice
usually absorbs an error that small; with it off, 4.86% is exactly the margin you miss
by. At 16:9 and `alignment = 32`, typing 1.03 gives 1344×768 and typing 0.98 gives
1312×736.

Worth noticing what that setting produces in the first place: 1344×768 is **7:4**
(1.750), not the 16:9 (1.778) it is offered under. It reaches H3's documented canvas
only because the two axes rounded apart — see below. The exact-16:9 lattice at
`alignment = 32` is `512n × 288n`, which steps 1024×576 → 1536×864 with nothing in
between, so there is no on-ratio ~1 MP frame to be had; pick a row instead.

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
| MiniMax H3, P-H3 Image | **32** | 16 × 2 |
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

## Delivering to a fixed screen size

If your output has to land on a fixed canvas — a 2560×1440 game screen, for instance —
generate on the exact-ratio lattice and resize once, at the end, to the delivery size.

2560×1440 is exactly 16:9 and is itself a lattice row on both grids (n = 10 of
`256n × 144n` at alignment 16, n = 5 of `512n × 288n` at alignment 32). So every 16:9
row resizes onto it with **no crop and no stretch**, and — unlike 1920×1080, whose 1080
is off-grid for every model here — you can also generate at the delivery size directly:

| Generate | MP | Grid | → 2560×1440 |
|---|---|---|---|
| 1536 × 864 | 1.33 | 32 | ×5/3 upscale |
| 2048 × 1152 | 2.36 | 64 | ×5/4 upscale |
| **2560 × 1440** | 3.69 | 32 | **native** |
| 2816 × 1584 | 4.46 | 16 | ×10/11 downscale |
| 3072 × 1728 | 5.31 | 64 | ×5/6 downscale |
| 3328 × 1872 | 6.23 | 16 | ×10/13 downscale |
| **3584 × 2016** | 7.23 | 32 | **×5/7 downscale** — P-H3 8 MP |
| 3840 × 2160 | 8.29 | 16 | ×2/3 downscale — Selector only, above the preset ceiling |

Generating over-size and downscaling is **supersampling** — free anti-aliasing on hair,
eyelashes and fabric that a 1:1 render never gets. 3584 × 2016 is the pick when the
model can take it (P-H3 Image); 2560 × 1440 native when it can't, rather than upscaling
something smaller, since an upscale cannot add detail the generation didn't have. The
rows below 2560 are there for video, where the model's native canvas wins.

Grid matters for which model can produce the row, not for the resize: a grid-16 row like
2816 × 1584 is fine for Krea 2 but off-grid for H3 / P-H3.

### Which filter

Measured on a zone plate (a chirp whose frequency rises toward the corners), downscaled
to 2560×1440 through the same code paths `common_upscale` uses. Two numbers per cell:
**passband** error, where the detail is below 1440p's Nyquist limit and should survive
untouched, and **aliasing**, the residue left where the detail is above that limit and
should average to flat grey:

| Filter | 3072 × 1728 → 2560×1440 (×5/6) | 3584 × 2016 → 2560×1440 (×5/7) |
|---|---|---|
| | passband / aliasing | passband / aliasing |
| **lanczos** | **0.0044** / 0.0020 | **0.0043** / **0.0429** |
| bicubic | 0.0101 / 0.0000 | 0.0100 / 0.1872 |
| bilinear | 0.0429 / 0.0000 | 0.0316 / 0.1423 |
| area | 0.1154 / 0.0000 | 0.0871 / 0.0438 |
| nearest-exact | 0.1162 / 0.0000 | 0.0942 / 0.2527 |

(At ×5/6 the source's own resolution limit sits below the aliasing band, so nothing can
alias there; the column only separates filters on passband.)

**Use `lanczos`.** It is the only filter that is best on both counts. Three caveats:

- **`bicubic` and `bilinear` alias at real reductions.** ComfyUI calls
  `torch.nn.functional.interpolate` without `antialias`, so they don't prefilter. Near
  ×1 that barely matters; at ×5/7 bicubic leaves over 4× lanczos's aliasing — moiré on
  fine fabric and hair.
- **`area` is a trap.** It aliases less than bicubic, but at a non-integer factor its
  box averages one input pixel for some outputs and two for others — uneven weighting,
  visible as blotchiness, and the worst passband error here bar nearest.
- **`lanczos` quantises to 8-bit.** `comfy/utils.py` routes it through PIL via
  `Image.fromarray(np.clip(255. * ..., 0, 255).astype(np.uint8))`. Irrelevant for 8-bit
  delivery (PNG / WebP / VP9), but if you keep a 16-bit master, save it *before* the
  scale node and do the resize outside ComfyUI (ffmpeg or ImageMagick lanczos, both full
  precision) — at these reductions `bicubic` is not a safe float fallback.

### Use `Upscale Image`, not `Upscale Image By`

Pin the delivery size, not the scale factor. `ImageScaleBy` computes
`round(dim * scale_by)` and then calls the same `common_upscale` as `ImageScale`, so the
pixels are identical — but its `scale_by` widget has `step: 0.01`, and the rounded
factors are wrong for almost the whole table:

| Source | typed at step 0.01 | result |
|---|---|---|
| 1536 × 864 | 1.67 | 2565 × 1443 ✗ |
| 2816 × 1584 | 0.91 | 2563 × 1441 ✗ |
| 3072 × 1728 | 0.83 | 2550 × 1434 ✗ |
| 3328 × 1872 | 0.77 | 2563 × 1441 ✗ |
| 3584 × 2016 | 0.71 | 2545 × 1431 ✗ |

Only 2048 × 1152 at 1.25 lands. Feeding the factor from a full-precision float node
avoids that, but buys nothing: `ImageScale` at `2560 × 1440` is exact by construction,
needs no extra node, and is correct for *every* lattice row without edits — whereas a
hardcoded factor is correct for exactly one source resolution and goes silently stale
the moment you change the preset.

The lattice already did the work upstream by making the source exactly 16:9. `5/7` is
just what `2560/3584` happens to equal; it isn't a number you need to carry around.

### Wiring it up

Graphs for the common cases are under [Usage](#usage). The delivery-side detail:

`Upscale Image` handles a `[B,H,W,C]` video batch frame by frame, but if you are dumping
a PNG sequence and encoding with ffmpeg anyway, fold the resize into that pass instead —
it avoids an intermediate 8-bit round trip and ffmpeg's lanczos is full precision:

```bash
ffmpeg -framerate 24 -i raw_%05d.png \
  -vf "scale=2560:1440:flags=lanczos,format=yuv420p" \
  -c:v libvpx-vp9 -crf 28 -b:v 0 -row-mt 1 \
  out.webm
```

24 fps matches H3's internal `FPS = 24`.

### One simplification

Set **`alignment = 32` for everything.** 32 is a multiple of 16, so every size it
produces is legal on Krea 2 *and* MiniMax H3 — one configuration for a whole project
instead of tracking which model needs which grid. The lattice is coarser, but at 16:9
the useful rows are all still there (1024×576, 1536×864, 2048×1152, 2560×1440), and
2048×1152 comes out identical either way.

## Nodes

### Resolution Preset

One combo of 136 presets: 122 generated exact-ratio entries and 14
[P-H3 Image buckets](#p-h3-image-generation-and-editing). Every entry lands on the
patch grid and holds its ratio exactly.

**Inputs** — `preset`, formatted `ratio · width × height · real MP · compatible models`.
P-H3 entries append their nominal bucket after the compatibility tag. The compatibility
tag comes from the resolution's divisibility, not a hand-kept table:

| Tag | Grid | Models |
|---|---|---|
| `K2` | 16 | Krea 2, Flux, SD3, Qwen-Image, Wan |
| `K2+H3` | 32 | the above, plus MiniMax H3 / P-H3 Image |
| `K2+H3+SDXL` | 64 | the above, plus SDXL / SD1.5 — universally safe |

**Outputs** — `width`, `height`, `megapixels`, `label`.

The exact-ratio list is **generated from the lattice at import time**, so it
cannot drift from what `Resolution Selector (Real MP)` produces. Per ratio it uses the
coarsest grid that still offers at least six sizes up to 4.3 MP: fine-lattice ratios
like 1:1 (where L=16, so every 16 px is legal) would otherwise contribute a hundred
near-identical entries, and the coarse grid lands them on canonical sizes — 512, 768,
1024, 1536, 2048 — instead. Coarse-lattice ratios like 16:9 fall through to grid 16
and keep their full useful set. Larger sizes extend the same grid up to 8 MP,
preserving all existing preset labels for saved workflows.

The exact-ratio list spans 0.15 to 8 MP; each ratio stops at its largest grid-aligned
size within that ceiling. The bottom end is there for runs where the canvas is
overhead rather than output — an audio-only MiniMax H3 render, say, where `16:9 · 512 ×
288` is the smallest exact 16:9 the model's 32 px grid admits.

Why a list and not a number: with `exact_ratio` on, the valid outputs *are* a discrete
lattice. A continuous megapixel input advertises precision that does not exist, and
pushes the lookup table out of the node and into your head (or a sticky note next to
the graph). Reach for `Resolution Selector (Real MP)` when you need a ratio or a size
this list does not carry — 21:9 at exactly 2352×1008, say, which falls on grid 16 while
the 21:9 presets use grid 64.

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
