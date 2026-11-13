"""Where each bar of the reading sits on the page image -> layout.json.

The staff model gives symbols without positions. Three things are already known while a page is read,
and this module only listens to them (it changes no value homr or the model works with, so
page.musicxml and conf.json are the same bytes with or without it):

  - homr's page pipeline: the frame it works in (the upload, cropped to the paper when homr finds its
    edges, then scaled to 1920 px wide), the five lines of every staff it cuts in that frame, and the
    thin vertical strokes it takes for bar lines;
  - the cut it hands the model for each staff (x in the cut is a straight scale of x on the page:
    homr's straightening moves pixels up and down only);
  - the decoder's `attention` output: for each symbol it writes, roughly where on the cut it looked.

A bar's box is its staff's band between two bar lines. Each bar-line symbol the model wrote is placed
by its attention, then snapped to the nearest vertical stroke found on the staff when one is close;
see `place_staff`. Everything is given in pixels of the uploaded image (upright, as OpenCV loads it:
EXIF orientation applied) and the file says how sure each bar is.

    install(h)                 once, after page.setup(); wraps homr functions
    attach(inference)          once, on the OnnxStaff2Score instance
    reset()                    before each page
    write(path, xml_path)      after the page was read: writes layout.json (or nothing on failure)
"""
import json
import math
import xml.etree.ElementTree as ET

LAYOUT_VERSION = 1

RAW = {}          # what was heard while the current page was read
_STATE = {}       # hooks' scratch: current staff, attention of the decoding in progress


def reset():
    RAW.clear()
    RAW.update(staves=[], barLines=[], measures=None)
    _STATE.clear()
    _STATE.update(attention=[], crops=[], canvas=None)


def _view_offset(whole, part):
    """(x, y) of `part` inside `whole` when it is a view of it, else (0, 0)."""
    try:
        import numpy as np
        if not np.shares_memory(whole, part):
            return 0, 0
        a, b = whole.__array_interface__["data"][0], part.__array_interface__["data"][0]
        row = whole.strides[0]
        off = b - a
        if off < 0 or part.shape[0] > whole.shape[0] or part.shape[1] > whole.shape[1]:
            return 0, 0
        return int((off % row) // whole.strides[1]), int(off // row)
    except Exception:
        return 0, 0


def install(h, inference=None):
    """Listen to homr's page pipeline. Call after page.setup() (which wraps some of the same
    functions: ours go on top and hand every value through untouched)."""
    import homr.staff_parsing as SP
    M, G = h.M, h.G
    reset()
    if inference is not None:
        attach(inference)

    autocrop_ = M.autocrop
    def autocrop(img):
        out = autocrop_(img)
        try:
            x, y = _view_offset(img, out) if out is not img else (0, 0)
            RAW["upload"] = [int(img.shape[1]), int(img.shape[0])]
            RAW["crop"] = [x, y, int(out.shape[1]), int(out.shape[0])]
        except Exception:
            RAW["failed"] = "crop"
        return out
    M.autocrop = autocrop

    detect_ = M.detect_bar_lines
    def detect_bar_lines(lines, unit):
        out = detect_(lines, unit)
        try:
            RAW["barLines"] = [[round(float(b.center[0]), 1), round(float(b.center[1]), 1),
                                round(float(max(b.size)), 1), round(float(min(b.size)), 1)] for b in out]
        except Exception:
            RAW["failed"] = "barLines"
        return out
    M.detect_bar_lines = detect_bar_lines

    crop_ = SP.crop_image_and_return_new_top
    def crop(image, *box):
        out, top = crop_(image, *box)
        try:
            _STATE["crops"].append((float(top[0]), float(top[1]), int(out.shape[1]), int(out.shape[0])))
        except Exception:
            pass
        return out, top
    SP.crop_image_and_return_new_top = crop

    center_ = SP.center_image_on_canvas
    def center(image, canvas_size, *a, **k):
        try:
            _STATE["canvas"] = (int(image.shape[1]), int(image.shape[0]), int(canvas_size[0]), int(canvas_size[1]))
        except Exception:
            pass
        return center_(image, canvas_size, *a, **k)
    SP.center_image_on_canvas = center

    prepare_ = SP.prepare_staff_image
    def prepare_staff_image(debug, index, staff, staff_image, regions):
        _STATE["crops"], _STATE["canvas"] = [], None
        out = prepare_(debug, index, staff, staff_image, regions)
        try:
            heard_staff(SP, staff, staff_image, regions)
        except Exception:
            RAW["failed"] = "staff"
        return out
    SP.prepare_staff_image = prepare_staff_image

    def heard_staff(SP, staff, staff_image, regions):
        region = SP._calculate_region(staff, regions)
        dims = SP.get_tr_omr_canvas_size((int(region[3] - region[1]), int(region[2] - region[0])))
        scale = float(dims[1] / (region[3] - region[1]))
        crops, canvas = _STATE["crops"], _STATE["canvas"]
        row = dict(
            frame=[int(staff_image.shape[1]), int(staff_image.shape[0])],
            unit=round(float(staff.average_unit_size), 2),
            # the five lines at each grid point: x, top line y, bottom line y
            grid=[[round(float(p.x), 1), round(float(min(p.y)), 1), round(float(max(p.y)), 1)] for p in staff.grid],
            cut=None, symbols=[],
            # the note heads homr's page model saw on this staff (x, y), left to right
            heads=sorted([round(float(n.center[0]), 1), round(float(n.center[1]), 1)]
                         for n in staff.get_notes()),
        )
        if len(crops) == 2 and canvas and canvas[0] > 0:
            # x on the cut -> x in the frame:  (x_cut * cropW / canvasW + left) / scale
            row["cut"] = dict(scale=scale, left=crops[0][0] + crops[1][0], width=canvas[0], canvas=canvas[2],
                              top=crops[0][1] + crops[1][1], height=canvas[1], canvasHeight=canvas[3])
        row["ink"] = ink_profile(staff_image, row["grid"], row["unit"])
        RAW["staves"].append(row)
        _STATE["image"] = staff_image

    build_ = G.build_measures
    def build_measures(args, voice, is_first_part, has_two_staves=False):
        measures = build_(args, voice, is_first_part, has_two_staves)
        try:
            if is_first_part and RAW.get("measures") is None:
                RAW["measures"] = dict(count=len(measures), segments=_segments(voice))
        except Exception:
            RAW["failed"] = "measures"
        return measures
    G.build_measures = build_measures


class _Tap:
    """The decoder session, with the `attention` output of every step kept."""

    def __init__(self, session, sink, index):
        self._session, self._sink, self._index = session, sink, index

    def run_with_iobinding(self, binding, *a, **k):
        out = self._session.run_with_iobinding(binding, *a, **k)
        try:
            self._sink.append(binding.get_outputs()[self._index].numpy().reshape(-1).astype(float).tolist())
        except Exception:
            self._sink.append(None)
        return out

    def __getattr__(self, name):
        return getattr(self._session, name)


def attach(inference):
    """Tag every symbol the staff model writes with (staff, step) and keep its attention."""
    reader = inference.reader
    try:
        index = list(reader.out_names).index("attention")
    except (AttributeError, ValueError):
        return
    reader.dec = _Tap(reader.dec, _STATE.setdefault("attention", []), index)

    read_ = reader.predict
    def read(x):
        _STATE["attention"].clear()
        symbols, probs = read_(x)
        _STATE["raw"] = [s[0] for s in symbols]
        return symbols, probs
    reader.predict = read

    predict_ = inference.predict
    def predict(image):
        out = predict_(image)
        try:
            tag(out)
        except Exception:
            RAW["failed"] = "symbols"
        return out
    inference.predict = predict

    def tag(out):
        if not RAW.get("staves"):
            return
        staff_no = len(RAW["staves"]) - 1
        raw, att = _STATE.get("raw") or [], _STATE.get("attention") or []
        rows, k, in_key = [], 0, False
        for step, rhythm in enumerate(raw):
            a = att[step] if step < len(att) else None
            rows.append([rhythm, None if not a or len(a) < 2 else round(a[0], 2), None if not a or len(a) < 2 else round(a[1], 2)])
            # the same walk as page.collapse_keys: a run of keySign is one symbol of `out`
            if rhythm == "keySign":
                if not in_key:
                    in_key = True
                    if k < len(out):
                        out[k]._lay = (staff_no, step)
                    k += 1
                continue
            in_key = False
            if k < len(out):
                out[k]._lay = (staff_no, step)
            k += 1
        RAW["staves"][staff_no]["symbols"] = rows
        RAW["staves"][staff_no]["tagged"] = k == len(out)


CLOSES_AFTER = ("repeatEnd", "repeatEndStart")


def _segments(voice):
    """The bars homr's writer cuts the first part into (music_xml_generator.build_measures), each as
    the (staff, step) tags of its symbols: `open`, the sign that opened it when that is a
    start-repeat; `close`, the bar line or repeat sign that closed it; `staves`, how many of its
    notes and rests came from each staff; `first` and `last`, its first and last tagged symbol."""
    def new():
        return dict(open=None, close=None, closeSign=None, staves={}, events=0, n=0, first=None, last=None)

    segs, cur = [], new()
    for s in voice:
        r, tag = s.rhythm, getattr(s, "_lay", None)
        if r == "repeatStart":
            segs.append(cur)
            cur = new()
            cur["open"] = tag
        cur["n"] += 1
        if tag is not None and r not in ("chord", "newline"):
            cur["first"] = cur["first"] or tag
            cur["last"] = tag
        if r.startswith(("note", "rest")):
            cur["events"] += 1
            if tag is not None:
                cur["staves"][tag[0]] = cur["staves"].get(tag[0], 0) + 1
        if "barline" in r or r in CLOSES_AFTER:
            cur["close"], cur["closeSign"] = tag, r
            segs.append(cur)
            cur = new()
    if cur["n"]:
        segs.append(cur)
    return segs


# ---- from what was heard to boxes ---------------------------------------------------------------

# In staff spaces (the gap between two staff lines, `unit`):
MERGE = 2.0         # strokes closer than this are one sign (a repeat sign, a double bar)
HEADER = 4.0        # no bar line this close to the staff's left end (the clef)
TAIL = 2.0          # a stroke this close to the staff's right end is the staff's closing bar line
COVER = 0.7         # a bar line covers at least this much of the staff's height
HEAD_COLUMN = 0.8   # note heads closer than this in x are one chord
BRACKET_SLACK = 0.5
HEADS_OFF = 2       # note heads seen and notes read may differ by this many on a staff
SURE = ("staff", "line", "ink")


STROKE = 0.8        # a stroke is dark over at least this much of the staff's height
STROKE_CLEAR = 1.0  # and at least this far from the note heads on either side


def ink_profile(image, grid, unit):
    """For each pixel column of the staff, left end to right end: how much of the height between
    the top and the bottom staff line is dark (0 to 1, in hundredths). A bar line is a column that
    is dark all the way; a stem rarely is, and a note head never. The page may lean a little, so a
    row counts as dark when the column or one of its two neighbours is."""
    import numpy as np
    x0, x1 = int(round(grid[0][0])), int(round(grid[-1][0]))
    x0, x1 = max(1, x0), min(image.shape[1] - 2, x1)
    if x1 <= x0:
        return dict(x0=x0, values=[])
    xs, gx = np.arange(x0, x1 + 1), [g[0] for g in grid]
    tops = np.interp(xs, gx, [g[1] for g in grid])
    bottoms = np.interp(xs, gx, [g[2] for g in grid])
    ya = max(0, int(tops.min() - 2 * unit))
    yb = min(image.shape[0], int(bottoms.max() + 2 * unit) + 1)
    strip = image[ya:yb, x0 - 1:x1 + 2]
    strip = (strip if strip.ndim == 2 else strip[:, :, :3].mean(axis=2)).astype(np.float32)
    # dark = clearly darker than the paper around it (the paper's shade drifts across a photo)
    paper = np.percentile(strip, 80, axis=0)
    kernel = max(3, int(4 * unit) | 1)
    paper = np.convolve(np.pad(paper, kernel // 2, mode="edge"), np.ones(kernel) / kernel, mode="valid")
    dark = strip < (paper * 0.72)[None, :]
    wide = dark[:, :-2] | dark[:, 1:-1] | dark[:, 2:]
    values = []
    for i in range(x1 - x0 + 1):
        a = int(round(tops[i] + 0.15 * unit)) - ya
        b = int(round(bottoms[i] - 0.15 * unit)) - ya
        col = wide[max(a, 0):max(b, a + 1), i]
        values.append(int(round(100 * float(col.mean()))) if col.size else 0)
    return dict(x0=x0, values=values)


def strokes_between(staff, lo, hi):
    """Full-height dark strokes of the staff between x = lo and x = hi (two note heads), as sign
    positions: columns at least STROKE dark, joined when close."""
    ink, u = staff.get("ink"), staff["unit"]
    if not ink or not ink["values"]:
        return []
    a = int(math.ceil(lo + STROKE_CLEAR * u)) - ink["x0"]
    b = int(math.floor(hi - STROKE_CLEAR * u)) - ink["x0"]
    xs = [ink["x0"] + i for i in range(max(a, 0), min(b, len(ink["values"]) - 1) + 1)
          if ink["values"][i] >= 100 * STROKE]
    signs = []
    for x in xs:
        if signs and x - signs[-1][-1] <= MERGE * u:
            signs[-1].append(x)
        else:
            signs.append([x])
    return [sum(g) / len(g) for g in signs]


def _is_sep(r):
    return "barline" in r or r in ("repeatStart", "repeatEnd", "repeatEndStart")


def _is_event(r):
    return r.startswith(("note", "rest"))


def _at(grid, x, col):
    """Line `col` (1 top, 2 bottom) of the staff at x, between grid points; flat outside them."""
    if x <= grid[0][0]:
        return grid[0][col]
    for a, b in zip(grid, grid[1:]):
        if x <= b[0]:
            t = (x - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            return a[col] + t * (b[col] - a[col])
    return grid[-1][col]


def strokes_on(staff, bar_lines):
    """homr's bar-line strokes that cross this staff, as sign positions (x), left to right."""
    grid, u = staff["grid"], staff["unit"]
    xs = []
    for cx, cy, long_side, _short in bar_lines:
        if not grid[0][0] - u <= cx <= grid[-1][0] + 2 * u:
            continue
        top, bottom = _at(grid, cx, 1), _at(grid, cx, 2)
        covered = min(cy + long_side / 2, bottom) - max(cy - long_side / 2, top)
        if bottom > top and covered >= COVER * (bottom - top):
            xs.append(cx)
    xs.sort()
    signs = []
    for x in xs:
        if signs and x - signs[-1][-1] <= MERGE * u:
            signs[-1].append(x)
        else:
            signs.append([x])
    return [sum(g) / len(g) for g in signs]


def boundaries_of(staff):
    """The places where the model wrote a bar line or repeat sign on this staff: signs with no note
    or rest between them are one place. Each: its steps, and how many notes and rests (`before`,
    `after`) and note onsets (`headsBefore`: chords once, rests not) the staff holds on each side."""
    groups, cur, events, onsets, in_chord = [], None, 0, 0, False
    for step, (r, _x, _y) in enumerate(staff["symbols"]):
        if _is_sep(r):
            if cur is None:
                cur = dict(steps=[step], before=events, headsBefore=onsets)
                groups.append(cur)
            else:
                cur["steps"].append(step)
        elif _is_event(r):
            events += 1
            if r.startswith("note") and not in_chord:
                onsets += 1
            cur = None
        in_chord = r == "chord"
    for g in groups:
        g["after"] = events - g["before"]
    return groups, events, onsets


def place_staff(staff, bar_lines):
    """Give each bar-line place of the staff an x in the frame, and say how it was found:

      line     a stroke homr saw, and the strokes inside the staff are as many as the places the
               model wrote (taken in order), or it is the only stroke between the two note heads
               the place must lie between
      ink      homr saw no stroke there, but the staff image has exactly one full-height dark
               stroke between those two note heads (`strokes_between`)
      between  no stroke to take: the place is only known to lie between two note heads
               (`lo`, `hi`); x is their middle
      share    neither (homr's note heads and the model's notes disagree, and no single stroke
               stands where the place should be): x is the place's share of the staff by the count
               of notes and rests; a bar with such an edge is given its staff's box
    Places before the first note or rest (`lead`) and after the last (`trail`) are the staff's ends.
    """
    grid, u = staff["grid"], staff["unit"]
    x0, x1 = grid[0][0], grid[-1][0]
    groups, events, onsets = boundaries_of(staff)
    strokes = strokes_on(staff, bar_lines)
    inner = [x for x in strokes if x0 + HEADER * u < x < x1 - TAIL * u]
    columns = []
    for hx, _hy in staff.get("heads") or []:
        if columns and hx - columns[-1][-1] <= HEAD_COLUMN * u:
            columns[-1].append(hx)
        else:
            columns.append([hx])
    columns = [sum(c) / len(c) for c in columns]
    heads_agree = bool(columns) and len(columns) == onsets

    for g in groups:
        g["kind"] = "lead" if not g["before"] else "trail" if not g["after"] else "inner"
    want = [g for g in groups if g["kind"] == "inner"]
    # A stroke left of the first note head is no bar line when the model wrote a note before its
    # first bar line: it is the start-repeat after the clef (or a stroke of the clef or the time).
    # The same on the right of the last head.
    if want and columns:
        if want[0]["headsBefore"] > 0:
            inner = [x for x in inner if x >= columns[0]]
        if onsets - want[-1]["headsBefore"] > 0:
            inner = [x for x in inner if x <= columns[-1]]

    # Each place lies between two note heads: the last of the bar before and the first of the bar
    # after. When homr saw one or two heads more or fewer than the model wrote, the two heads are
    # only known to within that many, and the gap is taken that much wider (`loose`).
    off = abs(len(columns) - onsets)
    loose = 0 < off <= HEADS_OFF and len(columns) >= 4
    for g in want:
        k = g["headsBefore"]
        if heads_agree and 0 < k < len(columns):
            g["lo"], g["hi"] = columns[k - 1], columns[k]
        elif loose and 0 < k < onsets:
            g["wlo"] = columns[max(0, min(len(columns) - 1, k - 1 - off))]
            g["whi"] = columns[max(0, min(len(columns) - 1, k + off))]

    def inside(g, lo, hi):
        return [x for x in inner if lo - BRACKET_SLACK * u <= x <= hi + BRACKET_SLACK * u]

    if len(inner) == len(want):
        for g, x in zip(want, inner):
            g["x"], g["how"] = x, "line"
            if "lo" in g and not inside(g, g["lo"], g["hi"]).count(x):
                g["how"] = "line?"   # in order, but not between the two heads it should be between
    else:
        first = columns[0] if columns else None
        for g in want:
            if "lo" in g:
                seen = inside(g, g["lo"], g["hi"])
                own = strokes_between(staff, g["lo"], g["hi"]) if len(seen) != 1 else []
                if len(seen) == 1:
                    g["x"], g["how"] = seen[0], "line"
                elif len(own) == 1:
                    g["x"], g["how"] = own[0], "ink"
                else:
                    g["x"], g["how"] = (g["lo"] + g["hi"]) / 2, "between"
                continue
            seen = inside(g, g["wlo"], g["whi"]) if "wlo" in g else []
            if len(seen) == 1:
                g["x"], g["how"] = seen[0], "line"
            else:
                start = (first - 1.5 * u) if first is not None else x0 + 6 * u
                g["x"], g["how"] = start + (x1 - start) * g["before"] / max(events, 1), "share"
        # a place cannot come before the one on its left, nor two places take one stroke
        last = None
        for g in want:
            if last is not None and g["x"] <= last["x"]:
                g["how"] = last["how"] = "share"
            last = g
    return dict(groups=groups, strokes=len(strokes), inner=len(inner), wanted=len(want),
                heads=len(columns), onsets=onsets, headsAgree=heads_agree)


def build(raw):
    """layout.json's content from what was heard (RAW)."""
    staves_raw = raw.get("staves") or []
    meas = raw.get("measures")
    crop = raw.get("crop")
    if not staves_raw or not meas or not crop or not raw.get("upload") or raw.get("failed"):
        return None   # something was not heard: no layout.json rather than a wrong one
    fw, fh = staves_raw[0]["frame"]
    sx, sy = crop[2] / fw, crop[3] / fh

    def up(x, y):
        return round(crop[0] + x * sx, 1), round(crop[1] + y * sy, 1)

    def band(staff, xa, xb):
        """The staff's five lines between xa and xb as a box of the upload: [x0, y0, x1, y1]."""
        grid = staff["grid"]
        xs = [xa, xb] + [g[0] for g in grid if xa < g[0] < xb]
        top = min(_at(grid, x, 1) for x in xs)
        bottom = max(_at(grid, x, 2) for x in xs)
        (ax, ay), (bx, by) = up(xa, top), up(xb, bottom)
        return [ax, ay, bx, by]

    plans = [place_staff(s, raw.get("barLines") or []) for s in staves_raw]
    where = {}   # (staff, step) -> its place
    for n, plan in enumerate(plans):
        for g in plan["groups"]:
            for step in g["steps"]:
                where[(n, step)] = g

    segs = list(meas["segments"])
    while len(segs) > meas["count"] and segs and segs[-1]["events"] == 0:
        segs.pop()
    aligned = len(segs) == meas["count"]
    bars = []
    for i in range(meas["count"]):
        row = dict(index=i, staff=None, box=None, trust=None, left=None, right=None)
        bars.append(row)
        seg = segs[i] if aligned else None
        if not seg:
            continue
        on = {int(k): v for k, v in seg["staves"].items()}
        if not on:
            continue   # nothing in it: it has no place of its own (the server drops such a bar)
        n = min(on)    # a bar that runs over a line break is shown on the staff where it starts
        staff = staves_raw[n]
        grid = staff["grid"]
        left = right = None
        if seg["open"] and seg["open"][0] == n:
            left = where.get(tuple(seg["open"]))
        elif i and segs[i - 1]["close"] and segs[i - 1]["close"][0] == n:
            left = where.get(tuple(segs[i - 1]["close"]))
        if seg["close"] and seg["close"][0] == n:
            right = where.get(tuple(seg["close"]))
        elif i + 1 < len(segs) and segs[i + 1]["open"] and segs[i + 1]["open"][0] == n:
            right = where.get(tuple(segs[i + 1]["open"]))
        xa, how_a = (grid[0][0], "staff") if left is None or left["kind"] != "inner" else (left["x"], left["how"])
        xb, how_b = (grid[-1][0], "staff") if right is None or right["kind"] != "inner" else (right["x"], right["how"])
        hows = (how_a, how_b)
        if len(on) > 1 or xb <= xa or "share" in hows:
            trust = "staff"
        elif "between" in hows:
            trust = "near"
        else:
            trust = "bar"
        if trust == "staff":
            xa, xb = grid[0][0], grid[-1][0]
        row.update(staff=n, box=band(staff, xa, xb), trust=trust, left=how_a, right=how_b)
        if len(on) > 1:
            row["overLineBreak"] = True
        if trust == "near":
            # the widest the bar can be: out to the note head an unsure edge lies before or after
            row["reach"] = band(staff, left["lo"] if how_a == "between" else xa,
                                right["hi"] if how_b == "between" else xb)
    staves = []
    for n, (staff, plan) in enumerate(zip(staves_raw, plans)):
        grid = staff["grid"]
        staves.append(dict(index=n, box=band(staff, grid[0][0], grid[-1][0]),
                           bars=[b["index"] for b in bars if b["staff"] == n],
                           barLinesSeen=plan["strokes"], barLinesInside=plan["inner"], barLinesRead=plan["wanted"],
                           noteHeadsSeen=plan["heads"], noteHeadsRead=plan["onsets"]))
    return dict(layout=LAYOUT_VERSION, page=dict(width=raw["upload"][0], height=raw["upload"][1]),
                frame=dict(crop=crop, width=fw, height=fh), aligned=aligned, staves=staves, bars=bars)


def write(path, xml_path=None, version=None):
    """Write layout.json for the page just read. Never raises: a page whose layout cannot be worked
    out has no layout.json, and its reading is not touched."""
    try:
        path.unlink(missing_ok=True)
        doc = build(RAW)
        if doc is None:
            return False
        if xml_path is not None:
            part = ET.parse(xml_path).getroot().find("part")
            if part is None or len(part.findall("measure")) != len(doc["bars"]):
                return False
        doc = dict(reader="nawtara-reader", version=version, **doc)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(path)
        return True
    except Exception:
        return False
