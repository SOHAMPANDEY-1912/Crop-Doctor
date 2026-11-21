"""Approximate leaf-area severity with plain OpenCV (no deep learning).

Idea: healthy tissue is green. Anything not green that lies inside the leaf (or next to the green
tissue and brown/yellow/orange) counts as affected. severity % = affected / (green + affected).
This is a rough colour-based estimate, not a clinical score. The mild/moderate/severe cut-offs are
placeholders (BUCKETS) that should be agreed with an agronomist.
"""
import sys
import cv2
import numpy as np

BUCKETS = (10.0, 30.0)      # percent of leaf area: mild < 10, moderate 10-30, severe >= 30
MIN_LEAF_FRAC = 0.08        # if less than 8% of the picture looks like leaf, say "unclear"


def _fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return (padded | cv2.bitwise_not(flood))[1:-1, 1:-1]


def _foreground_by_border(bgr, k, green):
    """Leaf mask for leaves with almost no green left: everything that differs from the border colour."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(3, int(0.04 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bmask = np.concatenate([green[:b].ravel(), green[-b:].ravel(), green[:, :b].ravel(), green[:, -b:].ravel()])
    if (bmask > 0).mean() > 0.3:                           # border is mostly green: the leaf fills the frame
        return None
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    if np.percentile(dist, 95) < 20:                       # nothing stands out from the background
        return None
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return _fill_holes(((lab_ == biggest).astype(np.uint8)) * 255)


def estimate_severity(bgr, max_side=512, buckets=BUCKETS):
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    k = max(5, int(0.03 * max(h, w))) | 1
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    green = (((H >= 30) & (H <= 90) & (S >= 40) & (V >= 40)).astype(np.uint8)) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    keep = np.zeros_like(green)
    for i in range(1, n):                                   # drop tiny green specks
        if stats[i, cv2.CC_STAT_AREA] >= 0.002 * h * w:
            keep[lab == i] = 255
    green = keep

    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    enclosed = _fill_holes(closed)                          # leaf region = green plus holes inside it
    # leaf outline = convex hull of the main green pieces (pieces smaller than 20% of the biggest are ignored)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    hull = np.zeros_like(green)
    if n2 > 1:
        biggest = st2[1:, cv2.CC_STAT_AREA].max()
        main = np.isin(lab2, [i for i in range(1, n2) if st2[i, cv2.CC_STAT_AREA] >= 0.2 * biggest])
        pts = cv2.findNonZero(main.astype(np.uint8))
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 255)
    near = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 3) | 1, max(3, k // 3) | 1)))
    lesion_colour = (((H >= 8) & (H < 30) & (S >= 50) & (V >= 40)).astype(np.uint8)) * 255   # brown / orange / yellow-brown

    lesion = (((enclosed > 0) & (green == 0)) | ((near > 0) & (lesion_colour > 0) & (green == 0))) & (V >= 30)
    lesion = cv2.morphologyEx(lesion.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    g, l = int((green > 0).sum()), int(lesion.sum())
    leaf_frac = (g + l) / float(h * w)
    overlay = bgr.copy()
    overlay[lesion] = (0.45 * overlay[lesion] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    method = "green"
    fg = _foreground_by_border(bgr, k, green)               # leaf mask from the background colour, if trustworthy
    frac = 0.0 if fg is None else float((fg > 0).mean())
    if fg is not None and 0.08 <= frac <= 0.85:
        lesion = (fg > 0) & (green == 0)
        g, l = int(((fg > 0) & (green > 0)).sum()), int(lesion.sum())
        leaf_frac, method = frac, "border"
        overlay = bgr.copy()
        overlay[lesion] = (0.45 * overlay[lesion] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    elif leaf_frac < MIN_LEAF_FRAC or (g + l) == 0:
        return dict(pct=None, bucket="unclear", leaf_frac=leaf_frac, overlay=overlay, method="none")
    pct = 100.0 * l / (g + l)
    bucket = "mild" if pct < buckets[0] else ("moderate" if pct < buckets[1] else "severe")
    return dict(pct=pct, bucket=bucket, leaf_frac=leaf_frac, overlay=overlay, method=method)


def annotate(result):
    img = result["overlay"].copy()
    text = "unclear: retake photo" if result["pct"] is None else f"{result['pct']:.0f}% affected - {result['bucket']}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    im = cv2.imread(src)
    if im is None:
        raise SystemExit(f"cannot read {src}")
    r = estimate_severity(im)
    cv2.imwrite(dst, annotate(r))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "overlay"})
