# nawtara-reader

A command that reads page images of sheet music, including Arabic quarter
tones (half-flats and half-sharps), and writes MusicXML.

    nawtara-reader PAGE_IMAGE OUT_DIR
    nawtara-reader PAGE_IMAGE PAGE_IMAGE [...] OUT_DIR

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

## Several pages in one run

    ./nawtara-reader page1.jpg page2.jpg page3.jpg out/

The models are loaded once and the pages are read one after the other, in the
order given. For the Nth page (`001`, `002`, ...) `OUT_DIR` gets:

- `page-N.musicxml`, `conf-N.json` and `layout-N.json`: the same files as for
  one page, with the same content a run on that page alone would write (apart
  from the seconds). `layout-N.json` is missing when the layout could not be
  worked out; the page still counts as read. `conf-N.json` is written last,
  after the layout, so a page is complete once its conf exists.
- or `page-N.error`: one line saying why the page was not read (no staff
  found, the image file is missing, homr failed). The run goes on with the
  next page.

Exit code 0 when every page was read, 3 when at least one was not (2 for bad
arguments or missing model files, 4 if PyTorch was loaded, as for one page). A
page with neither a conf nor an error file was not reached: the run stopped before it.

Each page is read on its own. Nothing is carried from one page to the next:
a page with no printed time signature has none in its file,
and repeats are not matched across pages. Joining the pages is the caller's
work. Files of the same names left by an earlier run are removed first; files
with other numbers are not, so give each run its own folder.

In `conf-N.json`, `seconds.total` is that page alone and `seconds.load` the
one model load of the run (the same figure on every page).

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
  `signatures` (from 0.4.0): every time signature the model read on part 1,
  in reading order: `bar` (the bar it opens, an index into `bars`), `staff`
  (the staff it was read on, from 1, top to bottom), `lowerFigure`, `prob`
  (the model's probability), `startsStaff`, `atBarStart` and `inFile`. The
  model reads only the lower figure of a time signature, and the MusicXML
  writer states one only where that figure changes, so a signature read with
  the same lower figure as the last (3/4 after 4/4) has `inFile` false: it is
  not in `page.musicxml`, but it was seen there. `signatures` is null when
  the signatures could not be placed on bars.
  `notes` (from 0.5.0, on each row of `bars`): the bar's notes in reading
  order, chord notes included: `pitch` (as `A4`), `lift` (the alteration the
  model chose: `_` none, `N` natural, `b`, `#`, `bb`, `##`, `qb` half-flat,
  `q#` half-sharp; the sounding one, the key's included), `p` (its
  probability) and `alts` (every other lift given at least 0.005). Null when the model's bars could not be
  matched to the file's measures (like `minProb`). A bar may list a note the
  writer left out of `page.musicxml`. `page.musicxml` does not change.
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

## Small music on a wide image

From 0.5.1, a one-line strip or a page whose music is small for the image's
width (wide margins, two pages side by side, a screen capture) is padded or
scaled before homr looks for staves, so its staff spacing reaches about 20 px
at the net. Other images are read as before (NOTICE, item 17).

## Licence

GNU Affero General Public License v3.0 (`LICENSE`), because it contains homr,
which is under the same licence (`vendor/HOMR-LICENSE`). If you run a modified
version as a network service you must offer its source to its users. See
`NOTICE`.
