"""
astro_utils.py
Andrew Kirwan, UCD
26/09/2025

A collection of functions and classes to help with data analysis
in the optical detector portion of the Space Detector Lab module.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipywidgets as widgets
from IPython.display import display


def _sigma_clip(sample, sigma=3.0, niter=10):
    """
    A recursive sigma-clipping function.

    Parameters:
    -----------
    sample : `np.ndarray` or `list`
        the input data to clip
    sigma : float
        the number of standard deviations to use
    niter : int
        the maximum number of clipping iterations

    Returns:
    -----------
    mask : array
        a boolean array representing the sigma-clipped data
    """

    mask = np.ones_like(sample, dtype=bool)

    for _ in range(niter):
        filtered = sample[mask]

        # make sure there is something to clip
        if filtered.size == 0:
            break

        median = np.median(filtered)
        stddev = np.std(filtered)

        # if there is no deviation, break
        if stddev == 0:
            break

        new_mask = np.abs(sample - median) < sigma * stddev

        if np.all(new_mask == mask):
            break

        return mask


def sigma_clip(sample, sigma=3.0, niter=5):
    """
    The public sigma clipping routine to calculate statistics

    Parameters:
    -----------
    sample : `np.ndarray`
        input data to clip
    sigma : float
        the number of standard deviations to use
    niter: int
        the maximum number of clipping iterations

    Returns:
    -----------
    mean : float
        the mean of the clipped data
    median : float
        the median of the clipped data
    stddev : float
        the standard deviation of the clipped data
    """

    mask = _sigma_clip(sample, sigma=sigma, niter=niter)
    clipped = sample[mask]

    return np.mean(clipped), np.median(clipped), np.std(clipped)


def normalize_image(x, pmin=1, pmax=99):
    """
    normalize data between 0 and 1, optionally clipping the original data
    between user-specified min/max values, or percentiles

    Parameters:
    -----------
    x: np.ndarray
        The data to normalize
    pmin: float or int
        the lower percentile to use for normalization
    pmax : float or int
        the upper percentile to use for normalization

    Returns:
    -----------
    norm: `matplotlib.colors.Normalize` object
        the normalized data in the interval [0, 1]
    """

    vmin, vmax = np.percentile(x, [pmin, pmax])
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    return norm(x)


def transform_by(func, x, pmin=1, pmax=99):
    """
    transform/stretch the data by a named function,
    normalizing the data using given percentiles

    Parameters:
    -----------
    func: str
        one of 'linear', 'sqrt' or 'log'
    x: np.ndarray
        data to transform
    pmin : float or int
        lower percentile
    pmax : float or int
        upper percentile
    Returns:
    ----------
    res : `np.array`
        the stretched data
    """

    x_norm = normalize_image(x, pmin=pmin, pmax=pmax)

    if func == "sqrt":
        return np.sqrt(x_norm)
    elif func == "log":
        eps = 1e-6
        return np.ma.log10(x + eps) / np.ma.log10(1 + eps)
    elif func == "linear":
        return x_norm
    else:
        return ValueError(f"Unsupported function: {func}")


def square_aperture(data, center, box_size):
    """
    Extract a square cutout from the input data

    Parameters:
    -----------
    data : `np.ndarray`
        the input data
    center : list or tuple
        the center of the box to extract, in y-x order
    box_size : int
        the size of the box in pixels

    Returns:
    -----------
    res : `np.ndarray`
        a cutout of the data
    """

    y, x = map(int, center)

    # ensure the box_size is odd
    if box_size % 2 == 0:
        box_size += 1

    half = box_size // 2

    # use slices
    sly = slice(y - half, y + half + 1)
    slx = slice(x - half, x + half + 1)

    return data[sly, slx]


def circular_aperture(data, center, radius):
    """
    Extract a circular cutout from the input data

    Parameters:
    -----------
    data : `np.ndarray`
        the input data
    center : list or tuple
        the center of the circle to extract, in y-x order
    radius : int or float
        the radius of the circle in pixels

    Returns:
    -----------
    cutout : `np.ndarray`
        a square cutout of the data
    mask : `np.ndarray`
        the circular mask used for the operation
    """

    # use the radius to establish a box size and take a cutout
    box_size = int(np.ceil(2 * radius))
    cutout = square_aperture(data, center, box_size)

    ys, xs = np.indices(cutout.shape)

    # get the new center of the cutout
    xc = cutout.shape[1] // 2
    yc = cutout.shape[0] // 2

    # get the pixel start position for the mask
    y0 = yc - box_size // 2
    x0 = xc - box_size // 2

    x_coords = xs + x0
    y_coords = ys + y0

    distance = np.sqrt((x_coords - xc) ** 2 + (y_coords - yc) ** 2)

    # finally the mask...
    mask = distance <= radius

    return cutout, mask


def histogram(
    ax, x, bins=100, range=None, density=False, normed=True, histtype="step", **kwargs
):
    """
    compute and plot a histogram from some given data, correctly
    handling normalization

    Parameters:
    -----------
    ax: matplotlib.axes.Axes instance
        where to plot the data
    x: array
        input values
    bins: int
        number of bins to use
    range: tuple or None
        bin range
    density: bool (default False)
        if True, return a noramlized probability density
    normed: bool, default True
        if True, return a normalized histogram
    histtype: one of 'bar', 'stacked', 'step', 'stepfilled'; default 'step'
        what type of histogram to draw

    Returns:
    ----------
    n, bins, patches:
        see Axes.hist for more information
    """

    counts, bin_edges = np.histogram(x, bins, range)
    widths = bin_edges[1:] - bin_edges[:-1]
    if density:
        counts = counts / (widths * x.size)
    elif normed:
        counts = counts / widths

    return ax.hist(
        bin_edges[:-1], bin_edges, weights=counts, histtype=histtype, **kwargs
    )


class Coordinates:
    """
    A simple class to allow a user to interactively select and store
    pixel coordinates from an image.

    Parameters:
    -----------
    data : 2D `np.ndarray`
        the image you want to plot and use to extract coordinates
    stretch : str
        can be one of 'linear', 'sqrt', or 'log'; the input
        image is transformed by this function for better visualization
    pmin : int, float
        the lower percentile to use for normalizing the data
    pmax : int, float
        the upper percentile to use for normalizing the data
    """

    def __init__(self, data, stretch="sqrt", pmin=1, pmax=99):
        self.image = data
        self.stretch = stretch
        self.pmin = pmin
        self.pmax = pmax
        self.coords = []
        self.interactive = False

        self.scaled_image = self.prepare_image()

        self.fig, self.ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
        self.plot_image()
        self.connect_events()

        self.toggle = widgets.ToggleButton(
            value=False,
            description="Interactive",
            tooltip="Toggle interactive coordinate selection",
        )

        self.toggle.observe(self.on_toggle, names="value")
        display(self.toggle)

    def prepare_image(self):
        stretched = transform_by(self.stretch, self.image, self.pmin, self.pmax)
        return stretched

    def plot_image(self):
        self.ax.imshow(self.scaled_image, origin="lower", cmap="gray")
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        plt.show()

    def connect_events(self):
        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        if not self.interactive or event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        self.coords.append((y, x))
        self.ax.plot(x, y, "r+")
        self.fig.canvas.draw()

    def on_toggle(self, change):
        self.interactive = change["new"]
        state = "ON" if self.interactive else "OFF"
        print(f"Interactive mode {state}")

    def get_coordinates(self):
        return np.asarray(self.coords)

    def save_coordinates(self, filename):
        if os.path.exists(filename):
            skip_header = True
        else:
            skip_header = False

        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            if not skip_header:
                writer.writerow(["y", "x"])
            writer.writerows(self.get_coordinates())
