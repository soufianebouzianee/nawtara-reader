"""Dark borders cut off before the page is read (0.5.3).

Installed by page.setup() right after `import homr.main`; homr's files are not edited.

Why: a phone screenshot of a sheet shown in a dark app (Facebook, WhatsApp, Instagram) has wide black bands above
and below the page, often with the status bar or the app's buttons in them. homr's autocrop keeps them (its page
contour starts at the image's edge), and the staff spacing estimate before segmentation (page_scale.py) then votes on
the bands' noise: on a 1170x2532 screenshot whose bands covered 40 % of it, it found 6 px instead of 13, the page was
enlarged 3.3x instead of 1.6x, and three of seven staves fell apart (22 of 26 bars, owner's test of 2026-11-26).
The same screenshot with the bands cut off read 26 of 26.

What it does, after homr's autocrop: the page is the rows from the first to the last row that is mostly white
(at least half of it above 180 grey). The rows above it go when they are dark (mean under 60) and at least 3 % of
the image's height, and the page between them is paper (median above 180); the same for the rows below, then for
the columns at the left and right of what is left.
The result is a view of the image, so layout.json maps every box back to the upload as for homr's own crop.
An image with no such band (every scan and photo of a page on paper) is returned as it is, and read as before.
"""
import cv2
import numpy as np

BRIGHT = 180       # grey above which a pixel is paper
PAPER_SHARE = 0.5  # a row (or column) of the page: at least this share of paper pixels
DARK = 60          # a band that goes: its mean grey is under this
MIN_BAND = 0.03    # and it is at least this share of the image's height (or width)
MIN_KEEP = 0.25    # never keep less than this share of the image


def _span(grey, axis):
    """(first, end) of the rows (axis=1) or columns (axis=0) that are mostly paper, or None."""
    paper = np.flatnonzero((grey > BRIGHT).mean(axis=axis) >= PAPER_SHARE)
    return None if len(paper) == 0 else (int(paper[0]), int(paper[-1]) + 1)


def _cut(grey, axis):
    """(start, end) to keep along one axis: only dark bands at the two ends go."""
    n = grey.shape[0] if axis == 1 else grey.shape[1]
    span = _span(grey, axis)
    if span is None or span[1] - span[0] < MIN_KEEP * n:
        return 0, n
    first, end = span
    page = grey[first:end] if axis == 1 else grey[:, first:end]
    if np.median(page) <= BRIGHT:
        return 0, n            # not a page of paper (white music on black, a dark photo): left as it is
    head = grey[:first] if axis == 1 else grey[:, :first]
    tail = grey[end:] if axis == 1 else grey[:, end:]
    start = first if first >= MIN_BAND * n and head.mean() < DARK else 0
    stop = end if n - end >= MIN_BAND * n and tail.mean() < DARK else n
    return start, stop


def trim(img):
    """The image without its dark bands, as a view of it; the image itself when it has none."""
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top, bottom = _cut(grey, 1)
    left, right = _cut(grey[top:bottom], 0)
    if (top, bottom, left, right) == (0, grey.shape[0], 0, grey.shape[1]):
        return img
    return img[top:bottom, left:right]


def install(M):
    """Patch homr.main (imported as M): dark bands go after homr's autocrop."""
    _autocrop = M.autocrop

    def autocrop(img):
        return trim(_autocrop(img))

    M.autocrop = autocrop
