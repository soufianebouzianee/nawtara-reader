"""One page image -> MusicXML + per-bar confidence, with homr's page pipeline and our ONNX model.

The runtime changes to homr are listed in NOTICE. The staff model is swapped for the ONNX loop of
onnx_loop.py. homr's sources (vendor/homr) are not edited: every change is made here, at import time.

  - vocabulary: lifts qb and q#, rhythm token keySign (must be patched before homr.transformer.configs)
  - Staff2Score.predict -> our encoder/decoder through onnxruntime, empty-context loop
  - LIFT_TO_ALTER gains qb -0.5 and q# +0.5
  - a run of keySign symbols becomes one key with <key-step>/<key-alter>
  - where homr's cutter joins a braced/bracketed pair of staves, the model is given the upper staff
    alone; a page of pairs where homr missed one pair is completed by spacing (split_systems)
  - per measure of part 1: the lowest probability of the tokens the model chose, and the lowest
    accidental-branch probability among the notes that carry a sign
  - title OCR off (rapidocr is not installed), every download raises, no teaser image is written
No PyTorch: nothing here or in homr's inference path imports it (cli.py refuses to run if it got loaded).
"""
import json
import math
import os
import sys
import types
import xml.etree.ElementTree as ET
from concurrent.futures import Future
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOMR_SRC = Path(os.environ.get("NAWTARA_READER_HOMR") or HERE.parent / "vendor")

STAVES_MODE = "top"   # "merge" = homr's own joining of staff pairs (research only)
ALTER = {"b": -1, "#": 1, "qb": -0.5, "q#": 0.5, "bb": -2, "##": 2, "N": 0}
SIGNS = frozenset(ALTER)   # lift classes that are an accidental sign (a natural included)

STAFF_KEYS = []   # key token each staff read, in reading order
STAFFS = []       # what homr's cutter found and what was fed to the model
MEASURES = []     # per-measure confidence rows of part 1
STAFF_SECONDS = []


def setup(threads, verbose=False):
    """Import homr with the patches. Returns the modules the caller needs.
    homr logs the whole reading to stderr; unless verbose, its log is dropped (a caller's log file
    must not hold a user's music)."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HOMR_SRC))
    import cv2
    import numpy as np  # noqa: F401
    import onnxruntime as ort
    cv2.setNumThreads(threads)

    # homr's segmentation net opens its session with onnxruntime's default (every core): give it ours
    _session = ort.InferenceSession
    def session(path, sess_options=None, *a, **k):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = 1
        return _session(path, sess_options, *a, **k)
    ort.InferenceSession = session

    # not installed and never used: title OCR and the weight downloader
    sys.modules["rapidocr"] = types.SimpleNamespace(RapidOCR=type("RapidOCR", (), {}))
    sys.modules["requests"] = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(RequestException=Exception), get=_blocked)

    if not verbose:
        import homr.simple_logging as SL
        SL.eprint = lambda *a, **k: None

    import homr.transformer.vocabulary as V  # patch before configs is imported
    _old_rhythm = V.build_rhythm
    V.build_lift = lambda: V.build_dict([V.nonote, V.empty, "#", "##", "N", "b", "bb", "qb", "q#"])
    V.build_rhythm = lambda: {**_old_rhythm(), "keySign": len(_old_rhythm())}
    V.has_rhythm_symbol_a_position = lambda r: r == "keySign" or r.startswith(("note", "rest", "clef"))

    import homr.download_utils as DU
    DU.download_file = _blocked
    import homr.main as M
    import homr.model as HM
    import homr.brace_dot_detection as BD
    import homr.music_xml_generator as G
    import homr.staff_parsing_tromr as SPT
    import homr.debug as DBG
    import homr.segmentation.inference_segnet as SEG
    import musicxml.xmlelement.xmlelement as mxl
    from homr.transformer.configs import Config
    from homr.transformer.vocabulary import EncodedSymbol

    segnet = os.environ.get("NAWTARA_READER_SEGNET")
    if segnet:
        SEG.segnet_path_onnx = segnet

    G.LIFT_TO_ALTER.update({"qb": -0.5, "q#": 0.5})

    def build_key(model_key, attributes):
        key = mxl.XMLKey()
        code = model_key.rhythm.split("_", 1)[1]
        if code.lstrip("-").isdigit():
            key.add_child(mxl.XMLFifths(value_=int(code)))
        else:
            for item in code.split(","):
                if not item:
                    continue
                step, alt = item.split(":")
                key.add_child(mxl.XMLKeyStep(value_=step))
                key.add_child(mxl.XMLKeyAlter(value_=float(alt)))
            if not code:
                key.add_child(mxl.XMLFifths(value_=0))
        attributes.add_child(key)
    G.build_key = build_key

    def collapse_keys(symbols):
        out, run = [], []
        def flush():
            if run:
                signs = []
                for s in run:
                    if s.lift in ALTER and ALTER[s.lift] and s.pitch and s.pitch[0] in "ABCDEFG":
                        signs.append(f"{s.pitch[0]}:{ALTER[s.lift]}")
                k = EncodedSymbol("keySignature_" + ",".join(signs), position=run[0].position)
                k.conf = min(getattr(s, "conf", 1.0) for s in run)
                out.append(k)
                run.clear()
        for s in symbols:
            if s.rhythm == "keySign":
                run.append(s)
                continue
            flush()
            out.append(s)
        flush()
        return out

    # ---- per-measure confidence of part 1 -----------------------------------------------------
    _build_measures = G.build_measures
    def build_measures(args, voice, is_first_part, has_two_staves=False):
        measures = _build_measures(args, voice, is_first_part, has_two_staves)
        if is_first_part:
            segs, cur = [], []
            for s in voice:
                r = s.rhythm
                if r in ("repeatStart",):
                    segs.append(cur)
                    cur = [s]
                    continue
                cur.append(s)
                if "barline" in r or r in ("repeatEnd", "repeatEndStart"):
                    segs.append(cur)
                    cur = []
            if cur:
                segs.append(cur)
            rows = []
            for seg in segs:
                real = [s for s in seg if s.rhythm not in ("chord", "newline")]
                cs = [getattr(s, "conf", None) for s in real]
                known = [c for c in cs if c is not None]
                notes = [s for s in seg if s.rhythm.startswith(("note", "rest"))]
                signs = [s.sign for s in seg if getattr(s, "sign", None) is not None]
                rows.append(dict(
                    min=round(min(known), 4) if known else None,
                    sign=round(min(signs), 4) if signs else None,
                    logp=round(sum(-math.log(max(c, 1e-9)) for c in known), 4),
                    low=sum(1 for c in known if c < 0.9), n=len(known), unknown=len(cs) - len(known),
                    events=len(notes),
                    worst=min(((getattr(s, "conf", 2), f"{s.rhythm} {s.pitch} {s.lift}") for s in real),
                              default=(None, ""))[1]))
            # a trailing segment with no note may not have become a measure
            while len(rows) > len(measures) and rows and rows[-1]["events"] == 0:
                rows.pop()
            MEASURES.append(dict(measures=len(measures), segments=len(rows), rows=rows))
        return measures
    G.build_measures = build_measures

    # ---- one staff at a time where homr joins a pair ------------------------------------------
    _merge = HM.Staff.merge
    def merge(self, other):
        res = _merge(self, other)
        res._pair = tuple(sorted((self, other), key=lambda s: s.min_y))
        return res
    HM.Staff.merge = merge

    def box(s):
        return dict(min_y=round(float(s.min_y)), max_y=round(float(s.max_y)), min_x=round(float(s.min_x)),
                    max_x=round(float(s.max_x)), unit=round(float(s.average_unit_size), 2))

    def split_systems(multi):
        """homr's MultiStaff list -> the list fed to the model.

        A joined pair is given as its upper staff alone. When most staves of a page sit in joined
        pairs (a page of two-staff systems), a staff left single is matched with its neighbour when
        the gap between them is no wider than the widest gap inside a joined pair (+15%); of such a
        matched pair the upper staff is read. Otherwise single staves are read as homr reads them."""
        mode = STAVES_MODE
        pairs_gap, flat = [], []
        for m in multi:
            for s in m.staffs[:1] if mode == "top" else m.staffs:  # a bracketed group: its top staff
                if getattr(s, "_pair", None):
                    up, lo = s._pair
                    pairs_gap.append(lo.min_y - up.max_y)
                    flat.append(("pair", up, lo))
                else:
                    flat.append(("single", s, None))
        STAFFS.append(dict(groups=[[box(x) for x in m.staffs] for m in multi],
                           pairs=[dict(upper=box(u), lower=box(l)) for k, u, l in flat if k == "pair"],
                           singles=[box(s) for k, s, _ in flat if k == "single"]))
        if mode != "top" or not pairs_gap:
            return multi
        n_pair, n_single = len(pairs_gap), sum(1 for k, *_ in flat if k == "single")
        flat.sort(key=lambda t: t[1].min_y)
        fed = []
        if 2 * n_pair >= n_single:  # a page of two-staff systems: complete the pairs homr missed
            limit = max(pairs_gap) * 1.15
            i = 0
            while i < len(flat):
                k, a, b = flat[i]
                if (k == "single" and i + 1 < len(flat) and flat[i + 1][0] == "single"
                        and 0 < flat[i + 1][1].min_y - a.max_y <= limit):
                    fed.append(a)
                    STAFFS[-1].setdefault("paired_by_gap", []).append([box(a), box(flat[i + 1][1])])
                    i += 2
                    continue
                fed.append(a)
                i += 1
        else:
            fed = [a for k, a, b in flat]
        STAFFS[-1]["fed"] = [box(s) for s in fed]
        return [HM.MultiStaff([s], []) for s in fed]

    _find = BD.find_braces_brackets_and_grand_staff_lines
    def find(debug, staffs, brace_dot):
        return split_systems(_find(debug, staffs, brace_dot))
    M.find_braces_brackets_and_grand_staff_lines = find

    done = Future()
    done.set_result("")
    M.detect_title = lambda *a, **k: done
    DBG.Debug.write_teaser = lambda self, *a, **k: None

    return types.SimpleNamespace(M=M, G=G, SPT=SPT, Config=Config, EncodedSymbol=EncodedSymbol,
                                 collapse_keys=collapse_keys)


def _blocked(*a, **k):
    raise RuntimeError("nawtara-reader: network use is blocked")


class OnnxStaff2Score:
    """What homr calls for each staff image (the role of homr's Staff2Score)."""
    MEAN, STD = 0.7931, 0.1738

    def __init__(self, h, encoder, decoder, threads, dump=None):
        from .onnx_loop import OnnxStaffReader
        self.h = h
        self.reader = OnnxStaffReader(encoder, decoder, h.Config(), threads)
        self.dump = dump

    @classmethod
    def tensor(cls, image):
        """The model input: float32 [1, 1, H, W], numpy only (same arithmetic as the research path,
        which did it in float64 and cast at the end)."""
        import numpy as np
        x = (np.array(image) / 255 - cls.MEAN) / cls.STD
        return x[np.newaxis, np.newaxis].astype(np.float32)

    def predict(self, image):
        import time
        x = self.tensor(image)
        if self.dump is not None:
            self.dump.append(x)
        t0 = time.perf_counter()
        symbols, probs = self.reader.predict(x)
        STAFF_SECONDS.append(round(time.perf_counter() - t0, 3))
        raw = []
        for s, (r, p, l) in zip(symbols, probs):
            e = self.h.EncodedSymbol(*s)
            e.conf = min(r, p, l) if e.rhythm.startswith("note") else r
            if e.rhythm.startswith("note") and e.lift in SIGNS:
                e.sign = l   # how sure the model is of the sign it chose for this note
            raw.append(e)
        out = self.h.collapse_keys(raw)
        STAFF_KEYS.append(next((o.rhythm for o in out if o.rhythm.startswith("keySignature_")), None))
        return out


# ---- does each bar add up to the metre? --------

def metre_fit(path, staff="1"):
    """Per measure of part 1: (quarter length the last <time> asks for, length of the staff's
    voice-1 notes and rests, grace and chord notes left out)."""
    part = ET.parse(path).getroot().find("part")
    div, expect, out = 1, None, []
    for meas in part.findall("measure"):
        for e in meas.iter("divisions"):
            div = int(e.text)
        t = meas.find("attributes/time")
        if t is not None and t.findtext("beats"):
            try:
                expect = Fraction(int(t.findtext("beats").split("+")[0]) * 4, int(t.findtext("beat-type")))
            except ValueError:
                pass
        total = Fraction(0)
        for n in meas.findall("note"):
            if n.find("grace") is not None or n.find("chord") is not None:
                continue
            if staff is not None and (n.findtext("staff") or "1") != staff:
                continue
            if (n.findtext("voice") or "1") != "1":
                continue
            total += Fraction(int(n.findtext("duration") or 0), div)
        new_system = any(p.get("new-system") == "yes" for p in meas.findall("print"))
        out.append((expect, total, meas.get("number"), new_system))
    return out


def adds_up(fit, i):
    """True / False, or None when the reader read no time signature. A short first or last bar
    counts as adding up (a pickup, a closing bar)."""
    exp, got = fit[i][:2]
    if exp is None:
        return None
    if got == exp:
        return True
    if (i == 0 or i == len(fit) - 1) and 0 < got < exp:
        return True
    return False


def bar_rows(xml_path, conf):
    """conf.json "bars": one row per <measure> of part 1, in file order."""
    fit = metre_fit(xml_path)
    rows = conf["rows"] if conf and conf["segments"] == len(fit) else None
    out = []
    for i, (exp, got, number, new_system) in enumerate(fit):
        row = rows[i] if rows else {}
        out.append(dict(index=i, number=number, minProb=row.get("min"),
                        minSignProb=row.get("sign"), addsUp=adds_up(fit, i),
                        events=row.get("events"), worst=row.get("worst"),
                        startsStaff=i == 0 or new_system,
                        length=str(got), expected=None if exp is None else str(exp)))
    return out
