"""A staff's crop is never thinner than the staff itself (0.5.2).

Installed by page.setup() right after `import homr.staff_parsing`; homr's files are not edited.

Why: homr cuts each staff out of the page between the end of the staff above and the start of the staff below
(`staff_parsing._calculate_region`). When its page model finds two staves that overlap (a staff found twice, or two
staves whose bounds cross on a small or tilted image), that window can close to a pixel or less. The crop is then
scaled to a height of 0 and OpenCV stops the whole page ("inv_scale_x > 0" in resize): the page is not read at all.
Seen on two user photos on 2026-11-25; reproduced with a one-pixel region.

What it does: the window always keeps the staff's own lines plus one staff space above and below. A window that
already does (every page whose staves do not overlap) is returned unchanged, so such a page is read exactly as before.
"""
import numpy as np


def install(SP):
    """Patch homr.staff_parsing (imported as SP)."""
    _region = SP._calculate_region

    def _calculate_region(staff, regions):
        r = _region(staff, regions)
        u = float(staff.average_unit_size)
        top, bottom = staff.min_y - u, staff.max_y + u
        if r[1] <= top and r[3] >= bottom:
            return r
        return np.array([int(r[0]), int(min(r[1], top)), int(r[2]), int(max(r[3], bottom))])

    SP._calculate_region = _calculate_region
