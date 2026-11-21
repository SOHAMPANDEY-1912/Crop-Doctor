"""Leaf-centred crop, used IDENTICALLY for training data, test photos and the app.

Why: the public datasets differ in how big the leaf is in the frame and what surrounds it (black cloth, soil, white sheet).
A classifier can learn those differences instead of the disease. Cropping to the leaf makes the scale and surroundings
more alike across datasets, phone photos and Pi camera photos. If no trustworthy leaf outline is found the full image is used.
"""
import cv2
import numpy as np


def _fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return (padded | cv2.bitwise_not(flood))[1:-1, 1:-1]


def leaf_bbox(bgr, margin=0.10, min_frac=0.004, max_frac=0.90, work=384, square=True):
    """(x0, y0, x1, y1) in original pixels, or None when there is no trustworthy leaf outline."""
    H0, W0 = bgr.shape[:2]
    s = min(1.0, work / float(max(H0, W0)))
    img = cv2.resize(bgr, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    Hh, S, V = cv2.split(hsv)
    green = ((((Hh >= 30) & (Hh <= 90) & (S >= 30) & (V >= 25)).astype(np.uint8)) * 255)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    b = max(3, int(0.04 * min(h, w)))
    bm = np.concatenate([green[:b].ravel(), green[-b:].ravel(), green[:, :b].ravel(), green[:, -b:].ravel()])
    if (bm > 0).mean() > 0.3:                                  # border is leaf: the leaf already fills the frame
        return None
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab[:, :, 1:] - bg[1:], axis=2)       # colour only, so shadows do not count
    if np.percentile(dist, 97) < 10 and not (green > 0).any():
        return None
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.bitwise_or(fg, green)
    k = max(5, int(0.03 * max(h, w))) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None
    overlap = np.bincount(lab_[green > 0], minlength=n)
    overlap[0] = 0
    pick = int(np.argmax(overlap)) if overlap.max() > 0 else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = st[pick]
    if not (min_frac <= area / float(h * w) <= max_frac):
        return None
    mx, my = int(margin * bw) + 2, int(margin * bh) + 2
    x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my)
    if (x1 - x0) < 24 or (y1 - y0) < 24:
        return None
    X0, Y0, X1, Y1 = x0 / s, y0 / s, x1 / s, y1 / s
    if square:                                  # keep the aspect ratio: grow the short side around the leaf centre
        side = max(X1 - X0, Y1 - Y0)
        cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
        X0, X1 = cx - side / 2, cx + side / 2
        Y0, Y1 = cy - side / 2, cy + side / 2
        if X0 < 0: X1, X0 = min(W0, X1 - X0), 0
        if Y0 < 0: Y1, Y0 = min(H0, Y1 - Y0), 0
        if X1 > W0: X0, X1 = max(0, X0 - (X1 - W0)), W0
        if Y1 > H0: Y0, Y1 = max(0, Y0 - (Y1 - H0)), H0
    return int(X0), int(Y0), int(min(W0, X1)), int(min(H0, Y1))


def leaf_crop(bgr, **kw):
    """Returns (cropped_bgr, used_crop)."""
    box = leaf_bbox(bgr, **kw)
    if box is None:
        return bgr, False
    x0, y0, x1, y1 = box
    return bgr[y0:y1, x0:x1], True


def leaf_crop_pil(im, **kw):
    """PIL RGB in, PIL RGB out."""
    from PIL import Image
    bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    out, used = leaf_crop(bgr, **kw)
    return (Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)) if used else im), used
