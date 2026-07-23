"""Generate a synthetic CRISM-like ENVI hyperspectral cube for manual testing.

Produces `testdata/crism_fr_synthetic.hdr` + `.img` with wavelengths (nm) covering
~1.0-2.6 um so the app can build the false-color RGB, plot spectra, and compute
CRISM spectral parameters (BD1400, BD1900, ...). The filename contains "fr" so the
app's FR edge-cropping branch is exercised.
"""
import os
import numpy as np
import spectral.io.envi as envi

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS, COLS = 80, 90

# CRISM-like wavelength grid in nm (stored in header; app converts >100 -> um).
wavelengths = np.arange(1000.0, 2601.0, 6.55)  # ~245 bands
bands = wavelengths.size


def gaussian_band(wl, center, depth, width):
    return depth * np.exp(-0.5 * ((wl - center) / width) ** 2)


def make_spectrum(base, features):
    spec = np.full(bands, base, dtype=np.float32)
    # gentle continuum slope
    spec += (wavelengths - wavelengths.mean()) / np.ptp(wavelengths) * 0.05
    for center, depth, width in features:
        spec -= gaussian_band(wavelengths, center, depth, width)
    return spec


# Two mineral-like endmembers with absorption features at CRISM param positions.
hydrated = make_spectrum(0.30, [(1400, 0.05, 25), (1900, 0.09, 40),
                                (2210, 0.06, 30), (2300, 0.05, 30)])
dust = make_spectrum(0.42, [(1900, 0.03, 40), (2500, 0.04, 50)])

cube = np.empty((ROWS, COLS, bands), dtype=np.float32)
rng = np.random.default_rng(42)
for r in range(ROWS):
    for c in range(COLS):
        # left half hydrated mineral, right half dust, blended in the middle
        frac = np.clip((c - COLS * 0.35) / (COLS * 0.3), 0, 1)
        spec = (1 - frac) * hydrated + frac * dust
        cube[r, c] = spec + rng.normal(0, 0.004, bands).astype(np.float32)

metadata = {
    'wavelength units': 'Nanometers',
    'wavelength': [f'{w:.3f}' for w in wavelengths],
    'description': 'Synthetic CRISM-like test cube',
}

hdr = os.path.join(HERE, 'crism_fr_synthetic.hdr')
envi.save_image(hdr, cube, dtype=np.float32, force=True,
                interleave='bil', metadata=metadata)
print('Wrote', hdr, 'shape', cube.shape)
