from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter
import ccd_utils as cu   #this needs to be uploaded directly when I used Notebook

MIN_SNR = 5.0 # Workbook2:threshold above background
PEAK_BOX = 10 # Workbook2:size of the local-max
R_AP, R_IN, R_OUT = 3.0, 5.0, 8.0 # Workbook3: aperture / annulus radii
ZP_U, ZP_V = 25.0, 25.0 # Workbook3: simple relative zeropoint

DATA = Path("/content/hst-hrd-assignment/data")
OUT  = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)

def _list_aligned(folder: Path):
    aligned = sorted(folder.glob("*_c0m_aligned.fits"))
    return aligned if aligned else sorted(folder.glob("*.fits"))

F336W = _list_aligned(DATA / "F336W")
F555W = _list_aligned(DATA / "F555W")

def read_image_any_hdu(path: Path):
    with fits.open(path) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None:
                return h.data.astype(float), h.header
    raise ValueError(f"No image data in {path}")

def median_combine(files, out_fits: Path):
    stack, hdr0 = [], None
    for f in files:
        data, hdr = read_image_any_hdu(f)
        stack.append(data)
        if hdr0 is None: hdr0 = hdr
    med = np.nanmedian(np.stack(stack, axis=0), axis=0)
    fits.writeto(out_fits, med.astype("float32"), hdr0, overwrite=True)
    print("wrote", out_fits)
    return med

img_u = median_combine(F336W, OUT / "combined_F336W.fits")
img_v = median_combine(F555W, OUT / "combined_F555W.fits")

mean, med, std = sigma_clipped_stats(img_v, sigma=3.0)
work = img_v - med
thresh_mask = work > (MIN_SNR * std)

def find_local_peaks(data, nsize=PEAK_BOX):
    """Workbook2 helper: return (y,x) coords of local maxima in an nsize window."""
    lm = maximum_filter(data, size=nsize)
    return np.argwhere(lm == data)

coords_all = find_local_peaks(work, nsize=PEAK_BOX)     # (y,x) everywhere

coords = np.array([c for c in coords_all if thresh_mask[c[0], c[1]]], dtype=int) # keep only those also above the SNR threshold


catalog = Table()
if len(coords) > 0:
    ys, xs = coords[:,0].astype(float), coords[:,1].astype(float) # build the star catalog (id, x, y)
    catalog['id'] = np.arange(1, len(xs)+1, dtype=int)
    catalog['x']  = xs
    catalog['y']  = ys
else:
    catalog['id'] = []; catalog['x'] = []; catalog['y'] = []

print(f"[] Workbook2 stars found: {len(catalog)}")

from astropy.stats import sigma_clipped_stats as _scs  # for an. median

def do_photometry(image, catalog, r_ap=R_AP, r_in=R_IN, r_out=R_OUT):
    """Workbook3: circular aperture flux; background=σ-clipped median in annulus."""
    fluxes = np.zeros(len(catalog), dtype=float)
    ok = np.zeros(len(catalog), dtype=bool)

    for i, (x, y) in enumerate(np.column_stack([catalog['x'], catalog['y']])):
        cut_src, mask_src = cu.circular_aperture(image, center=(y, x), radius=r_ap)
        if cut_src.size == 0 or mask_src.sum() == 0:
            continue
        src_sum = cut_src[mask_src].sum()
        cut_out, _ = cu.circular_aperture(image, center=(y, x), radius=r_out)
        if cut_out.size == 0:
            continue
        oy, ox = cut_out.shape
        yy, xx = np.indices((oy, ox))
        cy, cx = oy // 2, ox // 2
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        ann_mask = (dist >= r_in) & (dist <= r_out)

        vals = cut_out[ann_mask]
        bkg_med = 0.0 if vals.size == 0 else _scs(vals, sigma=3.0)[1]

        flux = src_sum - bkg_med * mask_src.sum()
        fluxes[i] = flux
        ok[i] = flux > 0      # from Workbook3: reject negative flux
    return ok, fluxes

ok_v, flux_v = do_photometry(img_v, catalog, r_ap=R_AP, r_in=R_IN, r_out=R_OUT)
ok_u, flux_u = do_photometry(img_u, catalog, r_ap=R_AP, r_in=R_IN, r_out=R_OUT)
print(f"[] F555W positive flux: {ok_v.sum()} / {len(catalog)}")
print(f"[] F336W positive flux: {ok_u.sum()} / {len(catalog)}")

keep = ok_v & ok_u # keep only stars measured in both bands
cat2 = catalog[keep]
flux_v2 = flux_v[keep]
flux_u2 = flux_u[keep]
print(f"[] matched in both filters: {len(cat2)}")

def flux_to_magnitude(flux, zeropoint=25.0):
    # workbook formula
    return -2.5 * np.log10(flux) - abs(zeropoint)

mag_V = flux_to_magnitude(flux_v2, ZP_V)
mag_U = flux_to_magnitude(flux_u2, ZP_U)
mag_tab = Table()
mag_tab['id'] = cat2['id']
mag_tab['x'] = cat2['x']
mag_tab['y'] = cat2['y']
mag_tab['aperture_radius'] = np.full(len(cat2), R_AP)
mag_tab['mag_F555W'] = mag_V
mag_tab['mag_F336W'] = mag_U
mag_tab.write(OUT / "mag_catalog.csv", overwrite=True)
print("wrote", OUT / "mag_catalog.csv")

plt.figure(figsize=(5, 6))
plt.style.use('dark_background')
plt.scatter(mag_V - mag_U, mag_V, s=3, color='w', alpha=0.6)
plt.gca().invert_yaxis()
plt.xlabel("F555W-F336W (Colour)", fontsize=12)
plt.ylabel("F555W (magnitude)", fontsize=12)
plt.title("Hertzsprung–Russell (Colour–Magnitude) Diagram", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "cmd.png", dpi=200)
plt.show()

plt.figure(figsize=(5, 6))
color = mag_V - mag_U
sc2 = plt.scatter(color, mag_V, c=color, cmap='jet', s=10, alpha=0.8, edgecolors='none')
plt.gca().invert_yaxis()
plt.xlabel("F555W-F336W (Colour)", fontsize=12)
plt.ylabel("F555W (Magnitude)", fontsize=12)
plt.title("Hertzsprung–Russell Diagram", fontsize=13)

cb = plt.colorbar(sc2)
cb.set_label("F555W-F336W", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "cmd_colour.png", dpi=200)
plt.show()
