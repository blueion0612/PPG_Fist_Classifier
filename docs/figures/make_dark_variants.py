"""Make a dark-theme variant of a matplotlib figure that has a white ground.

Only achromatic pixels are touched: white ground becomes near-black, black text
and axes become light. Colored pixels (the bars, the ROC line) keep their hue,
so the data is untouched.
"""
import sys
import numpy as np
from PIL import Image

DARK_BG = np.array([13, 17, 23], dtype=np.float32)      # GitHub dark canvas
LIGHT_INK = np.array([201, 209, 217], dtype=np.float32)  # GitHub dark foreground


def darkify(src, dst, sat_thresh=26):
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(np.float32)

    mx = a.max(axis=2)
    mn = a.min(axis=2)
    achromatic = (mx - mn) < sat_thresh          # gray, white or black
    lum = a.mean(axis=2) / 255.0                 # 1.0 = white ground, 0.0 = ink

    # white ground -> DARK_BG, black ink -> LIGHT_INK, linear in between
    mapped = DARK_BG[None, None, :] * lum[..., None] + LIGHT_INK[None, None, :] * (1.0 - lum[..., None])

    out = a.copy()
    out[achromatic] = mapped[achromatic]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(dst)
    frac = achromatic.mean()
    print(f"  {dst}   achromatic pixels remapped: {frac*100:.1f}%")


if __name__ == "__main__":
    import glob, os
    here = os.path.dirname(os.path.abspath(__file__))
    targets = sys.argv[1:] or [
        p for p in sorted(glob.glob(os.path.join(here, "fig*.png")))
        if not p.endswith("-dark.png")
        # the confusion matrix uses a colormap that spans white to blue, so
        # inverting its achromatic pixels would flip what the light cells mean
        and "confusion_matrix" not in p
    ]
    for src in targets:
        darkify(src, src.replace(".png", "-dark.png"))
