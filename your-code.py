# I used the worked Brightspace workbooks 1-3 for most of this 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter
import ccd_utils as cu  # this needed to be uploaded separately in Google Colab for it to work

MIN_SNR = 5.0 # defining the constants (from workbook)
PEAK_BOX = 10
R_AP = 3.0
R_IN = 5.0
R_OUT = 8.0
ZP_U = 25.0
ZP_V = 25.0

BASE = Path(__file__).resolve().parent # I got help with this part
DATA = BASE / "data"
OUT  = BASE / "outputs"; OUT.mkdir(exist_ok=True)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

F336W = sorted((DATA / "F336W").glob("*_aligned.fits")) #list the image files (think workbook 1)
F555W = sorted((DATA / "F555W").glob("*_aligned.fits"))
print("files found:", len(F336W), len(F555W))

def read_any_fits(path): # read & combine data
    hdul = fits.open(path)  # directly from workbook 1
    for h in hdul:
        if getattr(h, "data", None) is not None:
            data = h.data.astype(float)
            hdr = h.header
            hdul.close()
            return data, hdr
    hdul.close()
    raise ValueError("no image data found??")

def median_combine(files, outfile):
    imgs = []
    hdr0 = None
    for f in files:
        data, hdr = read_any_fits(f)
        imgs.append(data)
        if hdr0 is None:
            hdr0 = hdr
    print("combining", len(imgs), "images")
    med = np.nanmedian(np.stack(imgs), axis=0)
    fits.writeto(outfile, med.astype("float32"), hdr0, overwrite=True)
    print("saved:", outfile)
    return med

img_u = median_combine(F336W, OUT / "combined_F336W.fits")
img_v = median_combine(F555W, OUT / "combined_F555W.fits")


mean, med, std = sigma_clipped_stats(img_v, sigma=3.0) # finding stars (Workbook 2)
print("mean/med/std = ", mean, med, std)

work = img_v - med
mask = work > (MIN_SNR * std)

lm = maximum_filter(work, size=PEAK_BOX)
coords_all = np.argwhere(lm == work)

coords = []
for c in coords_all:
    y, x = c
    if mask[y, x]:
        coords.append((y, x))
coords = np.array(coords)
print("num stars maybe:", len(coords))

cat = Table() # make table for the stars, cat for catalog
if len(coords) > 0:
    cat['id'] = np.arange(1, len(coords) + 1)
    cat['x'] = coords[:, 1].astype(float)
    cat['y'] = coords[:, 0].astype(float)
else:
    cat['id'] = []
    cat['x'] = []
    cat['y'] = []
print("catalog done", len(cat))

if len(cat) < 5: # checking all okay
    print(cat)

from astropy.stats import sigma_clipped_stats as scs # photometry bit (Workbook 3)

def aperture_phot(image, catalog, r_ap, r_in, r_out):
    fluxes = np.zeros(len(catalog))
    ok = np.zeros(len(catalog), dtype=bool)
    for i in range(len(catalog)):
        x = catalog['x'][i]
        y = catalog['y'][i]
        cut_src, mask_src = cu.circular_aperture(image, (y, x), r_ap)
        if cut_src.size == 0:
            continue
        src_sum = cut_src[mask_src].sum()

        cut_out, _ = cu.circular_aperture(image, (y, x), r_out)
        if cut_out.size == 0:
            continue

        oy, ox = cut_out.shape
        yy, xx = np.indices((oy, ox))
        cy, cx = oy // 2, ox // 2
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        ann_mask = (dist >= r_in) & (dist <= r_out)
        vals = cut_out[ann_mask]

        if vals.size == 0:
            bkg_med = 0.0
        else:
            bkg_med = scs(vals, sigma=3.0)[1]

        flux = src_sum - bkg_med * mask_src.sum()
        fluxes[i] = flux
        if flux > 0:
            ok[i] = True

        if i % 50 == 0:
            print("flux", i, "=", flux) # to check if it’s working
    return ok, fluxes

ok_v, flux_v = aperture_phot(img_v, cat, R_AP, R_IN, R_OUT)
ok_u, flux_u = aperture_phot(img_u, cat, R_AP, R_IN, R_OUT)

print("ok_v:", ok_v.sum(), " ok_u:", ok_u.sum())

keep = ok_v & ok_u
cat2 = cat[keep]
flux_v2 = flux_v[keep]
flux_u2 = flux_u[keep]
print("matched stars in both:", len(cat2))

def flux_to_mag(flux, zp): # mags (Workbook 3 again)
    return -2.5 * np.log10(flux) - abs(zp) # same equation from workbook

mag_V = flux_to_mag(flux_v2, ZP_V)
mag_U = flux_to_mag(flux_u2, ZP_U)

tab = Table() # save results
tab['id'] = cat2['id']
tab['x'] = cat2['x']
tab['y'] = cat2['y']
tab['mag_F555W'] = mag_V
tab['mag_F336W'] = mag_U
tab.write(OUT / "mag_catalog.csv", overwrite=True)
print("created mag_catalog.csv")

plt.figure() #plot hr diagram bit (Workbook 3 last part)
plt.style.use('dark_background')
plt.scatter(mag_V - mag_U, mag_V, s=1, color='white', alpha=0.3)
plt.gca().invert_yaxis()
plt.xlabel("F555W - F336W colour")
plt.ylabel("F555W magnitude")
plt.title("Hertzbrung-Russell Diagram")
plt.tight_layout()
plt.savefig(OUT / "cmd.png", dpi=200)
plt.show()

plt.figure(figsize=(5,5)) # I want a plot with more colour and similarity to a real H-R diagram
color = mag_V - mag_U
sc2 = plt.scatter(color, mag_V, c=color, cmap='jet', s=5, alpha=0.8, edgecolors='none')
plt.gca().invert_yaxis()
plt.xlabel("F555W-F336W (Colour)", fontsize=12)
plt.ylabel("F555W (Magnitude)", fontsize=12)
plt.title("Hertzsprung–Russell Diagram", fontsize=13)

cb = plt.colorbar(sc2)
cb.set_label("F555W-F336W", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "cmd_colour.png", dpi=200)
plt.show()
