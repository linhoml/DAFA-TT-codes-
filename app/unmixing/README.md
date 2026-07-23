# Unmixing

Linear / sparse spectral unmixing and Hapke-SSA unmixing for SpectralApp.

## Menu (Unmixing)

1. **加载端元光谱库…** — `.mat` (DAFA/TT `TargetLibrary_paper.mat`), `.txt`, or a folder of txt spectra  
2. **Hapke model** — convert reflectance ↔ single-scattering albedo, then linear unmix in SSA space  
3. **Sparse unmixing** — NNLS / OMP / FCLS / UCLS in reflectance (or I/F) space  
4. **显示丰度图… / 显示 RMSE 图** — after whole-image mode  

Default library (if present): `data/libraries/TargetLibrary_paper.mat`  
(54 serpentine + 77 carbonate endmembers from Lin et al. DAFA/TT).

## Typical workflow

1. Open a CRISM cube (File → Open)  
2. Click a pixel (optional window average via `像元窗口`)  
3. Unmixing → load library (or accept the default MAT)  
4. Run Hapke or Sparse → choose 当前像元 or 整图  

Observed vs modeled spectra are drawn on the raw-spectrum panel; abundance maps use the result panel.
