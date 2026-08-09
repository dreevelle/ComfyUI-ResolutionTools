"""ComfyUI-ResolutionTools.

Resolution maths for latent-space models, done exactly.

Three nodes:

``ResolutionPreset``
    The everyday one. A single list of known-good resolutions, each exactly on
    its aspect ratio and on the model's patch grid, tagged with the model
    families it is legal for. Generated from the lattice at import time, so it
    cannot drift from what the solver below produces.

``ResolutionSelectorMP``
    The parametric one, for custom ratios and arbitrary budgets. Aspect ratio +
    a *real* megapixel target (10^6 pixels, not 1024^2) -> width/height that
    land on the model's patch grid **without drifting the aspect ratio**.
    Rounding each axis independently -- what the built-in Resolution Selector
    does -- turns 9:16 into 0.5649 instead of 0.5625. This solves on the
    lattice of grid-aligned exact-ratio sizes instead.

``ResolutionAlignToGrid``
    Snap an arbitrary width/height (e.g. 1920x1080) to the same grid
    (-> 1920x1088), which is what stops the bottom-edge banding described below.

Why alignment matters
---------------------
A latent-space DiT reaches pixels through two reductions::

    pixels --/ VAE spatial downscale --> latent --/ DiT patch size --> tokens

so both axes must be multiples of ``vae_downscale * patch_size``:

    ==========================================  =========
    Krea 2 / Flux / SD3 / Qwen-Image / Wan       16   (8 x 2)
    MiniMax H3                                   32   (16 x 2)
    SDXL / SD1.5 (UNet, 3 downsample stages)     64
    ==========================================  =========

Miss it and ComfyUI's ``comfy.ldm.common_dit.pad_to_patch_size`` pads the latent
up to the patch grid with **circular** padding -- the pad row is a wrapped copy
of the top of the image. That fake row gets folded into the same patch token as
the real bottom row, which the model never saw in training, so the final latent
row decodes to a band of garbage at the bottom edge. One latent row = 8 px on
Krea 2, 16 px on H3. Padding is only ever appended, so the artifact is always
bottom/right, never top/left.

Self-contained on purpose (nothing imported from ``comfy_extras``) so a ComfyUI
core update can't silently change the maths.
"""

import math
from enum import Enum

from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

__version__ = "1.1.0"


# ---------------------------------------------------------------------------
# Aspect ratios. Values are (w, h) integer pairs, reduced before use.
# ---------------------------------------------------------------------------
class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_V = "2:3 (Portrait Photo)"
    PHOTO_H = "3:2 (Photo)"
    STANDARD_V = "3:4 (Portrait Standard)"
    STANDARD_H = "4:3 (Standard)"
    SOCIAL_V = "4:5 (Portrait Social)"
    SOCIAL_H = "5:4 (Social)"
    WIDESCREEN_V = "9:16 (Portrait Widescreen)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_V = "9:21 (Portrait Ultrawide)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"
    CUSTOM = "Custom (use custom_w / custom_h)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.SOCIAL_V: (4, 5),
    AspectRatio.SOCIAL_H: (5, 4),
    AspectRatio.WIDESCREEN_V: (9, 16),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.ULTRAWIDE_V: (9, 21),
    AspectRatio.ULTRAWIDE_H: (21, 9),
}


ALIGNMENT_TOOLTIP = (
    "Pixel grid both axes must land on = VAE spatial downscale x DiT patch size. "
    "16: Krea 2 / Flux / SD3 / Qwen-Image / Wan. "
    "32: MiniMax H3. "
    "64: SDXL / SD1.5, and universally safe. "
    "Getting this wrong causes a band of artifacts along the bottom/right edge."
)


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------
def lattice_step(w_ratio: int, h_ratio: int, alignment: int) -> int:
    """Smallest scale increment keeping an exact ratio on the alignment grid.

    For a reduced ratio a:b, sizes are (a*s, b*s). ``a*s`` is a multiple of
    ``m`` only when ``s`` is a multiple of ``m // gcd(a, m)``; same for ``b``.
    Both hold together on multiples of the lcm of those two, so the valid
    resolutions are exactly ``(a*L*n, b*L*n)`` for integer n >= 1.

    9:16 at m=16  -> L=16 -> 144n x 256n  (n=8 gives 1152x2048)
    16:9 at m=32  -> L=32 -> 512n x 288n  (n=3 gives 1536x864)
    """
    return math.lcm(alignment // math.gcd(w_ratio, alignment),
                    alignment // math.gcd(h_ratio, alignment))


def solve_exact(w_ratio: int, h_ratio: int, alignment: int, target_pixels: float) -> tuple[int, int]:
    """Grid-aligned size with the ratio held exactly, area nearest the target."""
    step = lattice_step(w_ratio, h_ratio, alignment)
    # area(n) = w_ratio*h_ratio*step^2 * n^2, so solve for n and round.
    n = max(1, round(math.sqrt(target_pixels / (w_ratio * h_ratio * step * step))))
    return w_ratio * step * n, h_ratio * step * n


def solve_nearest(w_ratio: int, h_ratio: int, alignment: int, target_pixels: float) -> tuple[int, int]:
    """Grid-aligned size with each axis rounded independently.

    Hits the area target more closely than :func:`solve_exact` at the cost of a
    small aspect-ratio drift. Useful when the exact lattice is coarse.
    """
    scale = math.sqrt(target_pixels / (w_ratio * h_ratio))
    width = max(alignment, round(w_ratio * scale / alignment) * alignment)
    height = max(alignment, round(h_ratio * scale / alignment) * alignment)
    return width, height


def align_value(value: int, alignment: int, mode: str) -> int:
    """Snap one dimension onto the grid, never returning less than ``alignment``."""
    if mode == "up":
        snapped = math.ceil(value / alignment) * alignment
    elif mode == "down":
        snapped = math.floor(value / alignment) * alignment
    else:
        snapped = round(value / alignment) * alignment
    return max(alignment, snapped)


# ---------------------------------------------------------------------------
# Preset generation
#
# With exact_ratio on, the valid outputs are a discrete lattice, so a
# continuous megapixel input advertises precision that does not exist. These
# presets are the lattice, rendered readable -- derived at import time, never
# hand-maintained, so they cannot drift from what the solver produces.
# ---------------------------------------------------------------------------
PRESET_MIN_MP = 0.25
PRESET_MAX_MP = 4.3
PRESET_MIN_ENTRIES = 6   # coarsest grid that still offers this many sizes wins
PRESET_MAX_ENTRIES = 9   # thin geometrically beyond this, keeping both ends

# 16:9 and 9:16 lead: the common case for video and for delivery to a screen.
PRESET_RATIO_ORDER = [
    AspectRatio.WIDESCREEN_H, AspectRatio.WIDESCREEN_V,
    AspectRatio.SQUARE,
    AspectRatio.PHOTO_H, AspectRatio.PHOTO_V,
    AspectRatio.STANDARD_H, AspectRatio.STANDARD_V,
    AspectRatio.SOCIAL_H, AspectRatio.SOCIAL_V,
    AspectRatio.ULTRAWIDE_H, AspectRatio.ULTRAWIDE_V,
]


def compat_tag(width: int, height: int) -> str:
    """Which model families a resolution is legal for, from its divisibility."""
    if width % 64 == 0 and height % 64 == 0:
        return "K2+H3+SDXL"
    if width % 32 == 0 and height % 32 == 0:
        return "K2+H3"
    return "K2"


def _lattice_points(w_ratio, h_ratio, alignment):
    step = lattice_step(w_ratio, h_ratio, alignment)
    out, n = [], 1
    while True:
        w, h = w_ratio * step * n, h_ratio * step * n
        mp = w * h / 1_000_000
        if mp > PRESET_MAX_MP:
            return out
        if mp >= PRESET_MIN_MP:
            out.append((w, h, mp))
        n += 1


def build_presets() -> dict[str, tuple[int, int]]:
    """Readable label -> (width, height), covering every aspect ratio.

    Per ratio, take the *coarsest* grid that still yields PRESET_MIN_ENTRIES
    sizes. Fine-lattice ratios (1:1 has L=16, so 16n x 16n) would otherwise
    flood the list with a hundred near-identical entries; the coarse grid
    lands them on canonical sizes instead (512, 768, 1024, ...). Coarse-lattice
    ratios like 16:9 fall through to grid 16 and keep their full useful set.
    """
    presets = {}
    for ratio in PRESET_RATIO_ORDER:
        w_ratio, h_ratio = ASPECT_RATIOS[ratio]
        divisor = math.gcd(w_ratio, h_ratio)
        a, b = w_ratio // divisor, h_ratio // divisor

        points = []
        for alignment in (256, 128, 64, 32, 16):
            points = _lattice_points(a, b, alignment)
            if len(points) >= PRESET_MIN_ENTRIES:
                break

        if len(points) > PRESET_MAX_ENTRIES:
            span = len(points) - 1
            keep = {round(i * span / (PRESET_MAX_ENTRIES - 1)) for i in range(PRESET_MAX_ENTRIES)}
            points = [points[i] for i in sorted(keep)]

        # label with the declared ratio (21:9), not the reduced pair (7:3)
        name = ratio.value.split(" ")[0]
        for w, h, mp in points:
            presets[f"{name} · {w} × {h} · {mp:.2f} MP · {compat_tag(w, h)}"] = (w, h)
    return presets


PRESETS = build_presets()
PRESET_LABELS = list(PRESETS)
DEFAULT_PRESET = next(k for k, v in PRESETS.items() if v == (2048, 1152))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class ResolutionSelectorMP(IO.ComfyNode):
    """Aspect ratio + real megapixels -> grid-aligned width/height."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ResolutionSelectorMP",
            display_name="Resolution Selector (Real MP)",
            category="utilities",
            description=(
                "Width/height from an aspect ratio and a real megapixel target "
                "(1 MP = 1,000,000 pixels), snapped to the model's patch grid "
                "with the aspect ratio held exactly."
            ),
            inputs=[
                IO.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.WIDESCREEN_H,
                    tooltip="Aspect ratio of the output. Portrait entries are listed before their landscape counterparts.",
                ),
                IO.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.01,
                    max=64.0,
                    step=0.01,
                    tooltip=(
                        "Target area in real megapixels: 1.0 = 1,000,000 pixels. "
                        "Note this is NOT the built-in Resolution Selector's unit, which is 1024x1024 = 1.049 MP."
                    ),
                ),
                IO.Int.Input(
                    "alignment",
                    default=16,
                    min=1,
                    max=256,
                    step=8,
                    tooltip=ALIGNMENT_TOOLTIP,
                ),
                IO.Boolean.Input(
                    "exact_ratio",
                    default=True,
                    tooltip=(
                        "On: hold the aspect ratio exactly by snapping to the lattice of valid sizes, "
                        "accepting a slightly different area. Off: round each axis independently, "
                        "hitting the area target more closely but drifting the ratio."
                    ),
                ),
                IO.Int.Input(
                    "custom_w",
                    default=16,
                    min=1,
                    max=4096,
                    tooltip="Width term of the ratio, used only when aspect_ratio is Custom.",
                    advanced=True,
                ),
                IO.Int.Input(
                    "custom_h",
                    default=9,
                    min=1,
                    max=4096,
                    tooltip="Height term of the ratio, used only when aspect_ratio is Custom.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Int.Output(display_name="width", tooltip="Width in pixels, a multiple of alignment."),
                IO.Int.Output(display_name="height", tooltip="Height in pixels, a multiple of alignment."),
                IO.Float.Output(display_name="megapixels", tooltip="Real megapixels actually produced."),
                IO.String.Output(display_name="label", tooltip='Dimensions as "1152x2048", handy for filename prefixes.'),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio: str, megapixels: float, alignment: int,
                exact_ratio: bool, custom_w: int, custom_h: int) -> IO.NodeOutput:
        if aspect_ratio == AspectRatio.CUSTOM:
            w_ratio, h_ratio = custom_w, custom_h
        else:
            w_ratio, h_ratio = ASPECT_RATIOS[AspectRatio(aspect_ratio)]

        # Reduce, so 16:9 and 32:18 give the same lattice.
        divisor = math.gcd(w_ratio, h_ratio)
        w_ratio, h_ratio = w_ratio // divisor, h_ratio // divisor

        target_pixels = megapixels * 1_000_000
        solve = solve_exact if exact_ratio else solve_nearest
        width, height = solve(w_ratio, h_ratio, alignment, target_pixels)

        return IO.NodeOutput(width, height, (width * height) / 1_000_000, f"{width}x{height}")


class ResolutionPreset(IO.ComfyNode):
    """Pick a known-good resolution from a list instead of computing one."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ResolutionPreset",
            display_name="Resolution Preset",
            category="utilities",
            description=(
                "Known-good resolutions, picked from a list. Every entry is exactly "
                "on its aspect ratio and on the model's patch grid. The tag says which "
                "models it is legal for: K2 = Krea 2 / Flux / SD3 / Qwen-Image / Wan "
                "(grid 16), K2+H3 adds MiniMax H3 (grid 32), K2+H3+SDXL adds the UNet "
                "models (grid 64)."
            ),
            inputs=[
                IO.Combo.Input(
                    "preset",
                    options=PRESET_LABELS,
                    default=DEFAULT_PRESET,
                    tooltip=(
                        "ratio · width × height · real megapixels · compatible models. "
                        "For a ratio or size not listed, use Resolution Selector (Real MP)."
                    ),
                ),
            ],
            outputs=[
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
                IO.Float.Output(display_name="megapixels", tooltip="Real megapixels (10^6 pixels)."),
                IO.String.Output(display_name="label", tooltip='Dimensions as "2048x1152".'),
            ],
        )

    @classmethod
    def execute(cls, preset: str) -> IO.NodeOutput:
        width, height = PRESETS[preset]
        return IO.NodeOutput(width, height, (width * height) / 1_000_000, f"{width}x{height}")


class ResolutionAlignToGrid(IO.ComfyNode):
    """Snap an arbitrary width/height onto the model's patch grid."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ResolutionAlignToGrid",
            display_name="Align Resolution to Grid",
            category="utilities",
            description=(
                "Snap width/height to a multiple of the model's patch grid, e.g. "
                "1920x1080 -> 1920x1088 for Krea 2. Unaligned dimensions get "
                "circular-padded in the DiT and band along the bottom/right edge."
            ),
            inputs=[
                IO.Int.Input("width", default=1920, min=1, max=16384),
                IO.Int.Input("height", default=1080, min=1, max=16384),
                IO.Int.Input("alignment", default=16, min=1, max=256, step=8, tooltip=ALIGNMENT_TOOLTIP),
                IO.Combo.Input(
                    "mode",
                    options=["round", "up", "down"],
                    default="round",
                    tooltip="round: nearest multiple. up: never shrink (may add letterbox). down: never grow (may crop framing).",
                ),
            ],
            outputs=[
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
                IO.Boolean.Output(display_name="changed", tooltip="True if either dimension was off the grid."),
                IO.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, width: int, height: int, alignment: int, mode: str) -> IO.NodeOutput:
        new_w = align_value(width, alignment, mode)
        new_h = align_value(height, alignment, mode)
        changed = (new_w != width) or (new_h != height)
        return IO.NodeOutput(new_w, new_h, changed, f"{new_w}x{new_h}")


class ResolutionToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [ResolutionPreset, ResolutionSelectorMP, ResolutionAlignToGrid]


async def comfy_entrypoint() -> ResolutionToolsExtension:
    return ResolutionToolsExtension()
