# nawtara-reader

A command that reads one page image of sheet music, including Arabic quarter
tones (half-flats and half-sharps), and writes MusicXML.

    nawtara-reader PAGE_IMAGE OUT_DIR

It is built on [homr](https://github.com/liebharc/homr) 0.7.0 (page layout,
staff finding, dewarping, MusicXML writer), with a different staff model in
place of homr's, run on onnxruntime. See `NOTICE` for what is homr's, what is
ours, and the full list of run-time changes to homr.

## Important: the model files are not included

The staff model (`encoder.onnx` and `decoder_fused.onnx`) is **not** in this
repository. The command cannot read anything without it. Point to the folder
holding both files with `NAWTARA_READER_MODELS` (or `--models DIR`; or name the
two files with `--encoder` and `--decoder`).

homr's own segmentation model (about 55 MB, from homr's repository, 0.7.0) is
also not included here; see "Segmentation model" in `NOTICE`. Put it at
`vendor/homr/segmentation/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx`
or name your copy with `NAWTARA_READER_SEGNET`.

## Install

Python 3.11. PyTorch is not needed.

    python3.11 -m venv .venv
    .venv/bin/pip install -r requirements.txt

(or with uv: `uv venv --python 3.11 .venv` and
`uv pip install --python .venv/bin/python -r requirements.txt`).

## Run

    export NAWTARA_READER_MODELS=/path/to/folder/with/the/two/models
    ./nawtara-reader page.jpg out/

The launcher uses `NAWTARA_READER_PYTHON` if set, otherwise `.venv/bin/python`
beside it, otherwise `python3`.

Options: `--threads N` (or `NAWTARA_READER_THREADS`; default 2) sets the threads
of the model, the segmentation net and OpenCV. `--staves top|merge`: of a
braced pair of staves, read the upper one (default) or use homr's own joining.
`--verbose` keeps homr's own log, which prints the whole reading to stderr.
`--version`. No network is used; nothing is downloaded.

Exit codes: 0 read; 2 bad arguments or a missing file; 3 the page could not be
read (no staff found, or homr failed); 4 PyTorch was loaded.

## Output files

Written into `OUT_DIR`:

- `page.musicxml`: the reading. Part 1, staff 1 is the melody.
- `conf.json`: one row per bar of part 1 (`bars`), in file order: `minProb`
  (lowest probability among the symbols the model chose in that bar),
  `minSignProb` (lowest probability, in the model's accidental branch, of the
  sign chosen for a note that carries one: flat, sharp, natural, half-flat,
  half-sharp or a double; null when no note in the bar carries a sign), `addsUp`
  (true or false; null if no time signature was read; a short first or last bar
  counts as adding up), `startsStaff`, `length` and `expected` (in quarter
  notes), plus the model files' short hashes, the thread count and timings.
- `layout.json`: where each bar is on the image. `page` is the image's size in
  pixels, upright (EXIF orientation applied). `staves`: each staff read, with
  its `box` `[x0, y0, x1, y1]` in those pixels and the bars on it. `bars`: one
  row per bar of part 1, in file order like `conf.json`: `staff`, `box` (the
  staff's five lines between the bar's two bar lines) and `trust`: `bar` (both
  edges are the staff's end or a bar line found on the page), `near` (an edge
  is only known to lie between two note heads; `reach` is the widest the bar
  can be) or `staff` (no edge found: `box` is the whole staff). A bar with
  nothing in it has no box. The file is not written when the layout could not
  be worked out; `page.musicxml` and `conf.json` never depend on it.

## Reading the MusicXML

Every note states its sounding alteration in `<alter>`, the key's included, and
there is no `<accidental>` element: a bare E under a key that flattens E by a
half is a natural E. Do not apply the key signature to bare notes again. A
start-repeat sign closes the bar it is met in, so a staff that begins with one
has an empty bar in front.

## Licence

GNU Affero General Public License v3.0 (`LICENSE`), because it contains homr,
which is under the same licence (`vendor/HOMR-LICENSE`). If you run a modified
version as a network service you must offer its source to its users. See
`NOTICE`.
