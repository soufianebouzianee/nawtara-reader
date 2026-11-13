#!/usr/bin/env python3
"""nawtara-reader PAGE_IMAGE OUT_DIR

Reads one page image (photo or scan of sheet music) and writes into OUT_DIR:
  page.musicxml   the reading (homr's MusicXML; part 1 staff 1 is the melody)
  conf.json       per bar of part 1: lowest token probability, lowest probability of an accidental
                  sign, whether the bar adds up to the metre
  layout.json     where each bar of part 1 is on the image (nawtara_reader/layout.py); absent when
                  it could not be worked out

Model files: --models DIR or NAWTARA_READER_MODELS (a folder with encoder.onnx and
decoder_fused.onnx); --encoder / --decoder name single files. homr's segmentation model is taken from
vendor/homr/segmentation, or from NAWTARA_READER_SEGNET. Threads: --threads or NAWTARA_READER_THREADS
(default 2). No network is used; nothing is downloaded.

Exit codes: 0 read; 2 bad arguments or missing files; 3 the page could not be read (no staff found,
or homr failed); 4 PyTorch got loaded (must not happen).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

VERSION = "0.2.0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nawtara-reader", description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--models", default=os.environ.get("NAWTARA_READER_MODELS"))
    ap.add_argument("--encoder")
    ap.add_argument("--decoder")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NAWTARA_READER_THREADS") or 2))
    ap.add_argument("--staves", choices=("top", "merge"), default="top",
                    help="top (default): of a joined pair of staves read the upper one; merge: homr's own")
    ap.add_argument("--dump-tensors", help="research: save every staff's input tensor to this .npz")
    ap.add_argument("--verbose", action="store_true", help="keep homr's own log on stderr (it prints the reading)")
    ap.add_argument("--version", action="version", version=VERSION)
    a = ap.parse_args(argv)

    image = Path(a.image)
    out = Path(a.out_dir)
    encoder = Path(a.encoder) if a.encoder else Path(a.models or "") / "encoder.onnx"
    decoder = Path(a.decoder) if a.decoder else Path(a.models or "") / "decoder_fused.onnx"
    for f, what in ((image, "page image"), (encoder, "encoder model"), (decoder, "decoder model")):
        if not f.is_file():
            print(f"nawtara-reader: {what} not found: {f}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    for name in ("page.musicxml", "conf.json", "layout.json"):
        (out / name).unlink(missing_ok=True)

    t0 = time.perf_counter()
    from nawtara_reader import page as P
    P.STAVES_MODE = a.staves
    h = P.setup(a.threads, a.verbose)
    dump = [] if a.dump_tensors else None
    h.SPT.inference = P.OnnxStaff2Score(h, encoder, decoder, a.threads, dump)
    from nawtara_reader import layout as L   # where each bar is on the page (layout.json); listens only
    L.install(h, h.SPT.inference)
    t_load = time.perf_counter() - t0

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
            print(f"nawtara-reader: page not read: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        xml = work / "page.musicxml"
        if not xml.is_file():
            print("nawtara-reader: homr wrote no MusicXML", file=sys.stderr)
            return 3
        shutil.move(str(xml), out / "page.musicxml")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if "torch" in sys.modules:
        print("nawtara-reader: PyTorch was imported", file=sys.stderr)
        return 4
    conf = P.MEASURES[0] if P.MEASURES else None
    bars = P.bar_rows(out / "page.musicxml", conf)
    if dump is not None:
        import numpy as np
        np.savez(a.dump_tensors, **{f"staff{i}": x for i, x in enumerate(dump)})
    import onnxruntime
    doc = dict(
        reader="nawtara-reader", version=VERSION,
        models=dict(encoder=sha12(encoder), decoder=sha12(decoder)),
        runtime=dict(onnxruntime=onnxruntime.__version__, threads=a.threads),
        staves=dict(mode=a.staves, read=len(P.STAFF_KEYS), keys=P.STAFF_KEYS,
                    pairs=len(P.STAFFS[0]["pairs"]) if P.STAFFS else 0,
                    pairedByGap=len(P.STAFFS[0].get("paired_by_gap", [])) if P.STAFFS else 0),
        aligned=bool(conf and conf["segments"] == len(bars)),
        measures=conf["measures"] if conf else len(bars),
        segments=conf["segments"] if conf else None,
        bars=bars,
        rows=conf["rows"] if conf else None,
        seconds=dict(total=round(time.perf_counter() - t0, 2), load=round(t_load, 2),
                     staves=P.STAFF_SECONDS),
    )
    tmp = out / "conf.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(out / "conf.json")
    L.write(out / "layout.json", out / "page.musicxml", VERSION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
