"""The page's scale before homr finds staves (0.5.1).

Installed by page.setup() right after `import homr.main as M`; homr's files are not edited.

Why: homr resizes every image to 1920 px wide before its segmentation net.
Music that is small relative to the image's width (a page padded or framed wide, two pages side by side, a desktop or
video frame, a one-line strip) then reaches the net with a staff spacing of 3.5-6 px; at about 6 px staves start to
drop out, at 3.5-4 px homr says "No staffs found". Measured on 12 corpus pages padded 2-5x wide: staves found
102/83/35/0 of 103 -> 103/103/103/103 with this change; one-staff strips 31 -> 35 of 40 pages read.

What it does:
  1. a short image (height < 1/4 of its width: one or two staves cropped tight) gets a white margin of one image
     height above and below and 5 % left and right;
  2. the staff spacing is estimated from the image (run-length vote: commonest dark run = line, commonest light run
     between dark runs = space); when it would be under 12 px at homr's 1920 px width, the image is resized to the
     width that brings it to 20 px instead (at most 6000 px wide). Otherwise homr's own resize runs unchanged, so a
     page read today is read exactly as today.
"""
import cv2
import numpy as np
from PIL import Image

TARGET_SPACE = 20.0
MIN_SPACE_AT_1920 = 12.0
MAX_WIDTH = 6000


def staff_space(img):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.median(g) < 110: g = 255 - g
    _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = b.shape
    dark, light = np.zeros(h + 1, np.int64), np.zeros(h + 1, np.int64)
    for c in b[:, :: max(1, w // 400)].T:
        d = np.diff(np.concatenate([[0], c, [0]]).astype(np.int8)); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        if len(st) < 3: continue
        np.add.at(dark, np.clip(en - st, 0, h), 1); np.add.at(light, np.clip(st[1:] - en[:-1], 0, h), 1)
    if light[1:].sum() < 50: return None
    t = int(np.argmax(dark[1: max(2, h // 20)]) + 1); sp = int(np.argmax(light[2: max(3, h // 10)]) + 2)
    if light[sp] < 0.02 * light[1:].sum(): return None
    return t + sp


def add_margin(img):
    h, w = img.shape[:2]
    if h >= 0.25 * w: return img
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); bg = int(np.percentile(g, 90))
    return cv2.copyMakeBorder(img, h, h, int(0.05 * w), int(0.05 * w), cv2.BORDER_CONSTANT, value=(bg, bg, bg))


def install(M):
    """Patch homr.main (imported as M): margin at load, spacing-aware resize."""
    _imread, _resize = M.cv2.imread, M.resize_image

    def imread(path, *a):
        im = _imread(path, *a)
        return im if im is None else add_margin(im)

    def resize_image(im):
        sp = staff_space(im)
        if sp and sp * 1920 / im.shape[1] < MIN_SPACE_AT_1920:
            tw = int(min(MAX_WIDTH, 1920 * TARGET_SPACE / (sp * 1920 / im.shape[1])))
            return np.array(Image.fromarray(im).resize((tw, round(im.shape[0] * tw / im.shape[1]))))
        return _resize(im)

    M.cv2.imread = imread
    M.resize_image = resize_image
