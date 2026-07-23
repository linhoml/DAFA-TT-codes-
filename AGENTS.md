# AGENTS.md

## Cursor Cloud specific instructions

This is a PySide6 + Matplotlib desktop GUI app for CRISM Mars hyperspectral image
analysis. There is a single product: the desktop app at `app/spectral_app.py`. The
`app/disort/` package is a library used by the app's `Tools → DISORT correction`
menu (Python port of Fortran; `PythonicDISORT` is the RT solver).

### Services / how to run

- Run the app (GUI): `source .venv/bin/activate && DISPLAY=:1 python app/spectral_app.py`
  - The update script creates/uses a `.venv` in the repo root; dependencies are from `requirements.txt`.
  - `DISPLAY=:1` targets the VNC desktop already running in the cloud VM. Without a display the app aborts (Qt xcb).
- There is no lint config and no automated test suite in this repo. A basic syntax
  check is `python -m compileall app`. The `$\mu$` `SyntaxWarning`s at import time
  are pre-existing and harmless.
- Headless import check: `QT_QPA_PLATFORM=offscreen python -c "import sys; sys.path.insert(0,'app'); import disort"`.

### Gotchas

- System libraries (installed once outside the update script): PySide6 needs Qt/GL
  libs (`libegl1`, `libgl1`, `libxcb*`, `libxkbcommon*`, etc.) or import fails with
  `libEGL.so.1: cannot open shared object`.
- The app's `plt.rcParams` hardcodes Chinese font names (SimSun/SimHei/...) that are
  not present, so Chinese titles render as boxes and matplotlib logs `findfont`
  warnings. This is cosmetic only — menus and axis labels are English and all
  functionality works. Installing `fonts-noto-cjk` does NOT fix it because the font
  family names in the code don't match Noto's names.
- Data format: the app opens ENVI hyperspectral data (`.hdr` + data file) via
  `File → Open`. Wavelengths >100 in the header are treated as nm and divided by
  1000 to μm. Filenames containing `fr`/`hr` trigger automatic edge NoData cropping.
- No sample ENVI data ships in the repo (the `DAFATT for CRISM.zip` holds MATLAB/IDL
  scripts, not a cube). A synthetic CRISM-like test cube can be generated with
  `python testdata/make_test_cube.py`, which writes `testdata/crism_fr_synthetic.hdr`
  (gitignored). Use it for `File → Open` to exercise RGB display, pixel-spectrum
  plotting, and `Spectral parameter` (BD1400/BD1900/...) maps.
