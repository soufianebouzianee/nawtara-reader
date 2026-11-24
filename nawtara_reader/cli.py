#!/usr/bin/env python3
"""nawtara-reader PAGE_IMAGE OUT_DIR
nawtara-reader PAGE_IMAGE PAGE_IMAGE [...] OUT_DIR

Reads one page image (photo or scan of sheet music) and writes into OUT_DIR:
  page.musicxml   the reading (homr's MusicXML; part 1 staff 1 is the melody)
  conf.json       per bar of part 1: lowest token probability, lowest probability of an accidental
                  sign, whether the bar adds up to the metre; "signatures": every time signature the
                  model read on part 1 (bar, staff, lower figure), the ones page.musicxml leaves out too
  layout.json     where each bar of part 1 is on the image (nawtara_reader/layout.py); absent when
                  it could not be worked out

Given several page images, it reads them one after the other in the order given, with the models
loaded once, and writes for the Nth page (N = 001, 002, ...):
  page-N.musicxml, conf-N.json and layout-N.json   the same files as for one page, or
  page-N.error                                     one line saying why that page was not read; the run goes on
conf-N.json is written last (after layout-N.json, when there is one), so a page is complete once it exists. Each page is read on its own:
nothing (key, metre, repeats) is carried from one page to the next.

Model files: --models DIR or NAWTARA_READER_MODELS (a folder with encoder.onnx and
decoder_fused.onnx); --encoder / --decoder name single files. homr's segmentation model is taken from
vendor/homr/segmentation, or from NAWTARA_READER_SEGNET. Threads: --threads or NAWTARA_READER_THREADS
(default 2). No network is used; nothing is downloaded.

Exit codes: 0 read (several pages: every page was read); 2 bad arguments or missing files; 3 the
page could not be read (no staff found, or homr failed; several pages: at least one was not read,
see the .error files); 4 PyTorch got loaded (must not happen).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

VERSION = "0.5.1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def in_background(fn):
    """Start fn() in a thread; the returned function waits for it and gives its value (or raises its
    error). Used for the model files' hashes, which take as long as loading the models."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # handed to the caller below
            box["error"] = e
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def result():
        thread.join()
        if "error" in box:
            raise box["error"]
        return box["value"]
    return result


def read_page(h, image, out, xml_name):
    """One page image through homr -> OUT_DIR/xml_name. Returns (exit code, message)."""
    # homr writes next to the image it is given: hand it a link in a private folder
    work = Path(tempfile.mkdtemp(prefix=".reader-", dir=out))
    try:
        link = work / ("page" + image.suffix.lower())
        try:
            link.symlink_to(image.resolve())
        except OSError:
            shutil.copyfile(image, link)
        cfg = h.M.ProcessingConfig(False, False, False, False, -1, False, False, False)
        try:
            h.M.process_image(str(link), cfg, h.G.XmlGeneratorArguments(False, None, None))
        except Exception as e:  # homr raises on a page with no staff
            return 3, f"page not read: {type(e).__name__}: {e}"
        xml = work / "page.musicxml"
        if not xml.is_file():
            return 3, "homr wrote no MusicXML"
        shutil.move(str(xml), out / xml_name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if "torch" in sys.modules:
        return 4, "PyTorch was imported"
    return 0, ""


def write_conf(P, out, xml_name, conf_name, a, models, seconds):
    """conf.json of the page just read (P holds what the reader kept of it)."""
    conf = P.MEASURES[0] if P.MEASURES else None
    bars = P.bar_rows(out / xml_name, conf)
    import onnxruntime
    doc = dict(
        reader="nawtara-reader", version=VERSION,
        models=models,
        runtime=dict(onnxruntime=onnxruntime.__version__, threads=a.threads),
        staves=dict(mode=a.staves, read=len(P.STAFF_KEYS), keys=P.STAFF_KEYS,
                    pairs=len(P.STAFFS[0]["pairs"]) if P.STAFFS else 0,
                    pairedByGap=len(P.STAFFS[0].get("paired_by_gap", [])) if P.STAFFS else 0),
        aligned=bool(conf and conf["segments"] == len(bars)),
        measures=conf["measures"] if conf else len(bars),
        segments=conf["segments"] if conf else None,
        bars=bars,
        signatures=signatures(P, out / xml_name),
        rows=conf["rows"] if conf else None,
        seconds=seconds(),
    )
    tmp = out / (conf_name + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(out / conf_name)


def signatures(P, xml_path):
    """conf.json "signatures" (page.signature_rows); None when they could not be placed. Never fails
    the page: the MusicXML is the reading, this list only says where signatures were seen."""
    try:
        return P.signature_rows(xml_path)
    except Exception:
        return None


def one_line(text, limit=300):
    return " ".join(str(text).split())[:limit]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nawtara-reader", description=__doc__.split("\n\n", 1)[1].split("\n\nModel files")[0],
                                 usage="%(prog)s [options] PAGE_IMAGE [PAGE_IMAGE ...] OUT_DIR",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", metavar="PAGE_IMAGE ... OUT_DIR",
                    help="one or more page images, then the folder to write into")
    ap.add_argument("--models", default=os.environ.get("NAWTARA_READER_MODELS"))
    ap.add_argument("--encoder")
    ap.add_argument("--decoder")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NAWTARA_READER_THREADS") or 2))
    ap.add_argument("--staves", choices=("top", "merge"), default="top",
                    help="top (default): of a joined pair of staves read the upper one; merge: homr's own")
    ap.add_argument("--dump-tensors", help="research, one page only: save every staff's input tensor to this .npz")
    ap.add_argument("--verbose", action="store_true", help="keep homr's own log on stderr (it prints the reading)")
    ap.add_argument("--version", action="version", version=VERSION)
    a = ap.parse_intermixed_args(argv)
    if len(a.paths) < 2:
        ap.error("the following arguments are required: OUT_DIR")
    images = [Path(p) for p in a.paths[:-1]]
    out = Path(a.paths[-1])
    several = len(images) > 1
    if several and a.dump_tensors:
        ap.error("--dump-tensors takes one page")

    encoder = Path(a.encoder) if a.encoder else Path(a.models or "") / "encoder.onnx"
    decoder = Path(a.decoder) if a.decoder else Path(a.models or "") / "decoder_fused.onnx"
    needed = ((encoder, "encoder model"), (decoder, "decoder model"))
    if not several:
        needed = ((images[0], "page image"),) + needed
    for f, what in needed:
        if not f.is_file():
            print(f"nawtara-reader: {what} not found: {f}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    if several:
        names = [(f"page-{n:03d}.musicxml", f"conf-{n:03d}.json", f"page-{n:03d}.error", f"layout-{n:03d}.json")
                 for n in range(1, len(images) + 1)]
    else:
        names = [("page.musicxml", "conf.json", None, "layout.json")]
    for group in names:
        for name in group:
            if name:
                (out / name).unlink(missing_ok=True)

    t0 = time.perf_counter()
    models = in_background(lambda: dict(encoder=sha12(encoder), decoder=sha12(decoder)))
    from nawtara_reader import page as P
    P.STAVES_MODE = a.staves
    h = P.setup(a.threads, a.verbose)
    dump = [] if a.dump_tensors else None
    h.SPT.inference = P.OnnxStaff2Score(h, encoder, decoder, a.threads, dump)
    from nawtara_reader import layout as L   # where each bar is on the page (layout.json); listens only
    L.install(h, h.SPT.inference)
    t_load = time.perf_counter() - t0

    if not several:
        xml_name, conf_name, _, layout_name = names[0]
        code, message = read_page(h, images[0], out, xml_name)
        if code:
            print(f"nawtara-reader: {message}", file=sys.stderr)
            return code
        if dump is not None:
            import numpy as np
            np.savez(a.dump_tensors, **{f"staff{i}": x for i, x in enumerate(dump)})
        L.write(out / layout_name, out / xml_name, VERSION)   # never an error: no file when not worked out
        write_conf(P, out, xml_name, conf_name, a, models(), lambda: dict(
            total=round(time.perf_counter() - t0, 2), load=round(t_load, 2), staves=P.STAFF_SECONDS))
        return 0

    # several pages: each on its own, a failure is written down and the run goes on
    failed = 0
    for n, (image, (xml_name, conf_name, error_name, layout_name)) in enumerate(zip(images, names), 1):
        t_page = time.perf_counter()
        P.reset()
        L.reset()
        try:
            if not image.is_file():
                code, message = 2, f"page image not found: {image}"
            else:
                code, message = read_page(h, image, out, xml_name)
            if code == 0:
                L.write(out / layout_name, out / xml_name, VERSION)
                # total: this page alone; load: the one load of the run, the same figure on every page
                write_conf(P, out, xml_name, conf_name, a, models(), lambda: dict(
                    total=round(time.perf_counter() - t_page, 2), load=round(t_load, 2), staves=P.STAFF_SECONDS))
        except Exception as e:
            code, message = 3, f"page not read: {type(e).__name__}: {e}"
        if code:
            failed += 1
            (out / xml_name).unlink(missing_ok=True)
            (out / layout_name).unlink(missing_ok=True)
            (out / error_name).write_text(one_line(message) + "\n")
            print(f"nawtara-reader: page {n}: {one_line(message)}", file=sys.stderr)
            if code == 4:
                return 4
    return 3 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
