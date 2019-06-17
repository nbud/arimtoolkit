"""
Interactively locate artefact based on times of flight.

Select a signal in a view or a Bscan, show how it appears in the other views/Bscan.
"""
import logging
import itertools
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.ray
import arim.im
import arim.models.block_in_immersion as bim
import arim.plot as aplt

from . import common

# %%

USE_DYNAMIC_SCALE = True
PLOT_DEFECT_BOX = False

# %% Load configuration

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("arim").setLevel(logging.INFO)


def locate_artefact(dataset_name, save):
    # %%
    conf = arim.io.load_conf(dataset_name)
    # conf['grid']['pixel_size'] = 2e-3  # debug
    conf["grid"]["pixel_size"] = 0.5e-3  # hardcode to make faster
    aplt.conf["savefig"] = False

    result_dir = conf["result_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    # Load frame
    frame = common.load_frame(conf, apply_filter=True, expand=True)

    # Make grid
    z_backwall = conf["backwall"]["z"]
    assert not np.isnan(z_backwall)

    grid = common.make_grid_tfm(conf)
    grid_p = grid.to_oriented_points()
    probe_p = frame.probe.to_oriented_points()

    # Make views
    views = bim.make_views(
        frame.examination_object,
        probe_p,
        grid_p,
        tfm_unique_only=True,
        max_number_of_reflection=1,
    )

    # %% Plot Bscan
    bscan_ax, _ = aplt.plot_bscan_pulse_echo(
        frame, clim=[-60, -20], interpolation="bilinear"
    )
    bscan_ax.figure.canvas.set_window_title("Bscan")

    # %% Perform ray tracing

    arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

    # %% Run TFM

    tfms = OrderedDict()
    for i, view in enumerate(views.values()):
        with arim.helpers.timeit("TFM {}".format(view.name), logger=logger):
            tfms[view.name] = arim.im.tfm.tfm_for_view(frame, grid, view, fillvalue=0.0)

    # %% Plot all TFM

    try:
        reference_rect = common.reference_rect(conf)
        reference_area = grid.points_in_rectbox(**reference_rect)
    except common.NoDefect:
        reference_rect = None
        reference_area = None

    if USE_DYNAMIC_SCALE:
        scale = aplt.common_dynamic_db_scale(
            [tfm.res for tfm in tfms.values()], reference_area
        )
    else:
        scale = itertools.repeat((None, None))

    tfm_axes = {}

    ncols = 3
    nrows = 7
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(9, 12), sharex=False, sharey=False
    )
    xmin = grid.xmin
    xmax = grid.xmax
    zmin = conf["frontwall"]["z"]
    zmax = conf["backwall"]["z"]

    for i, ((viewname, tfm), ax) in enumerate(zip(tfms.items(), axes.ravel())):
        ref_db, clim = next(scale)
        clim = [-40, 0.0]
        ax, im = aplt.plot_tfm(
            tfm,
            ax=ax,
            scale="db",
            ref_db=ref_db,
            clim=clim,
            interpolation="none",
            savefig=False,
            draw_cbar=False,
        )
        ax.set_title(viewname, y=0.9, size="small")
        ax.set_adjustable("box")
        ax.axis([xmin, xmax, zmax, zmin])
        tfm_axes[viewname] = ax

        if ax in axes[-1, :]:
            ax.set_xlabel("x (mm)")
            ax.set_xticks([xmin, xmax, np.round((xmin + xmax) / 2, decimals=3)])
        else:
            ax.set_xlabel("")
            ax.set_xticks([])
        if ax in axes[:, 0]:
            ax.set_ylabel("z (mm)")
            ax.set_yticks([zmax, zmin, np.round((zmin + zmax) / 2, decimals=3)])
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), location="top", fraction=0.05, aspect=40, pad=0.03
    )
    cbar.ax.set_ylabel("dB")
    ax.figure.canvas.set_window_title(f"TFMs")

    return bscan_ax, tfm_axes, tfms, views


# %%
ALPHA_MAX = 0.5
_magenta_alpha_data = dict(
    red=[(0.0, None, 1.0), (1.0, 1.0, None)],
    green=[(0.0, None, 0.0), (1.0, 0.0, None)],
    blue=[(0.0, None, 1.0), (1.0, 1.0, None)],
    alpha=[(0.0, None, 0.0), (1.0, ALPHA_MAX, None)],
)
magenta_alpha = mpl.colors.LinearSegmentedColormap("MagentaAlpha", _magenta_alpha_data)
magenta_alpha.set_bad(color="w", alpha=1.0)
magenta_alpha.set_over(color="xkcd:bright green", alpha=1.0)
magenta_alpha.set_under(color="xkcd:bright green", alpha=1.0)

# %%


class TimeOfFlightSelector:
    def __init__(self, bscan_ax, tfms_axes, tfms, views):
        children = []
        for key, tfm in tfms.items():
            view = views[key]
            ax = tfms_axes[key]
            children.append(TfmSelector(self, tfm, view, ax))
        children.append(BscanSelector(self, bscan_ax))
        self.children = children

    def callback_select_points(self, tmin, tmax):
        for child in self.children:
            child.show_points_in_range(tmin, tmax)

    def clear_masks(self):
        for child in self.children:
            child.clear_mask()

    def disconnect(self):
        self.clear_mask()
        self.lasso.disconnect_events()
        self.ax.figure.canvas.mpl_disconnect(self._cid)
        self.ax.figure.canvas.draw_idle()

    def save(self):
        figs = set(child.ax.figure for child in self.children)
        for i, fig in enumerate(figs):
            fig.savefig(f"artefact_locator_{i:02}")


class BaseSelector:
    def __init__(self, parent, ax):
        self.ax = ax
        self.parent = parent
        self._cid = ax.figure.canvas.mpl_connect("button_press_event", self.onclick)

    def clear_mask(self):
        ax = self.ax
        for patch in ax.patches:
            patch.remove()
        for im in ax.images[1:]:
            im.remove()
        ax.figure.canvas.draw_idle()

    def onclick(self, event):
        if event.button == 3:  # right click
            self.parent.clear_masks()

    def disconnect(self):
        self.clear_mask()
        self.lasso.disconnect_events()
        self.ax.figure.canvas.mpl_disconnect(self._cid)
        self.ax.figure.canvas.draw_idle()


class TfmSelector(BaseSelector):
    def __init__(self, parent, tfm, view, ax):
        self.tfm = tfm
        self.view = view

        self.grid = tfm.grid
        self.grid_coords = tfm.grid.to_1d_points().coords[:, [0, 2]]

        super().__init__(parent, ax)

        lineprops = dict(color="magenta")
        lasso = mpl.widgets.LassoSelector(
            ax, self.onselect, button=1, lineprops=lineprops
        )
        self.lasso = lasso

    def show_points_in_range(self, tmin, tmax):
        in_range_points = self.get_points_from_tmin_tmax(tmin, tmax)
        assert in_range_points.shape == (self.grid.numpoints,)
        numelements = self.view.tx_path.rays.times.shape[0]
        assert in_range_points.shape == (self.grid.numpoints,)

        #        mask = (mask == numelements)
        mask = in_range_points / numelements
        #        mask = (in_range_points > 1)
        mask = np.rot90(mask.reshape((self.grid.numx, self.grid.numz)))
        mask2 = np.zeros_like(mask)
        mask2[mask > 0.7] = np.nan
        cmap = magenta_alpha
        #        cmap = 'Oranges'
        self.ax.imshow(
            mask2,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            extent=(self.grid.xmin, self.grid.xmax, self.grid.zmax, self.grid.zmin),
            origin="lower",
            interpolation="nearest",
        )
        self.ax.figure.canvas.draw_idle()

    def get_tmin_tmax_from_points(self, selected_points):
        """
        Returns min and max pulse-echo times for a cloud of points.
        Returns
        -------
        tmin, tmax:
            shape (numelements,)
        """
        # pulse-echo times
        times = self.view.tx_path.rays.times + self.view.rx_path.rays.times

        subtimes = times[:, selected_points]
        return np.min(subtimes, axis=1), np.max(subtimes, axis=1)

    def get_points_from_tmin_tmax(self, tmin, tmax):
        """
        Returns for each grid point the number of pulse-echo scanlines (between
        zero and numelements) whose times of flight are in the range.
        """
        # pulse-echo times
        times = self.view.tx_path.rays.times + self.view.rx_path.rays.times

        # shape: (numelements, numgrid)
        cond1 = tmin[:, np.newaxis] <= times
        cond2 = times <= tmax[:, np.newaxis]
        points = cond1 & cond2

        return np.sum(points, axis=0)

    def onselect(self, verts):
        path = mpl.path.Path(verts)
        selected_points = path.contains_points(self.grid_coords)
        tmin, tmax = self.get_tmin_tmax_from_points(selected_points)
        self.parent.callback_select_points(tmin, tmax)


class BscanSelector(BaseSelector):
    def __init__(self, parent, ax):
        super().__init__(parent, ax)

        im = ax.images[0]

        numelements, numsamples = im.get_size()
        tmin, tmax = im.get_extent()[:2]

        samples = np.linspace(tmin, tmax, numsamples)
        elements = np.arange(numelements)

        self._extent = im.get_extent()
        self._samples = samples

        xx, yy = np.meshgrid(samples, elements, indexing="ij")
        self.grid_coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self._shape = xx.shape

        lineprops = dict(color="magenta")
        lasso = mpl.widgets.LassoSelector(
            ax, self.onselect, button=1, lineprops=lineprops
        )
        self.lasso = lasso

    def get_tmin_tmax_from_points(self, selected_points):
        """
        Returns min and max pulse-echo times for a cloud of points.
        Parameters
        ----------
        selected_points : 
            Shape : numgridopoints

        Returns
        -------
        tmin, tmax:
            shape (numelements,)
        """

        # pulse-echo times
        a = selected_points.reshape(self._shape)

        # np.argmax returns the first occurence of the max so we can use it
        # to detect the first and the last "True"
        tmin_indices = np.argmax(a, axis=0)
        tmax_indices = a.shape[0] - 1 - np.argmax(a[::-1, :], axis=0)

        tmin = self._samples[tmin_indices]
        tmax = self._samples[tmax_indices]

        # case where no point selected in the scanline
        no_selection = np.max(a, axis=0) == False
        tmin[no_selection] = 0.0
        tmax[no_selection] = 0.0

        return tmin, tmax

    def get_points_from_tmin_tmax(self, tmin, tmax):
        """
        Returns for each grid point the number of pulse-echo scanlines (between
        zero and numelements) whose times of flight are in the range.
        """
        times = self._samples[:, np.newaxis]

        # shape: (numelements, numgrid)
        cond1 = tmin[np.newaxis, :] <= times
        cond2 = times <= tmax[np.newaxis, :]
        points = cond1 & cond2

        return points

    def show_points_in_range(self, tmin, tmax):
        in_range_points = self.get_points_from_tmin_tmax(tmin, tmax)

        mask = np.rot90(in_range_points)
        cmap = magenta_alpha
        # cmap = 'Oranges'
        self.ax.imshow(
            mask,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            extent=self._extent,
            origin="upper",
            interpolation="nearest",
        )
        self.ax.figure.canvas.draw_idle()
        self.ax.axis("auto")

    def onselect(self, verts):
        path = mpl.path.Path(verts)
        selected_points = path.contains_points(self.grid_coords)
        tmin, tmax = self.get_tmin_tmax_from_points(selected_points)
        self.parent.callback_select_points(tmin, tmax)
        self.ax.figure.canvas.draw_idle()


# %%
if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save
    if save:
        logger.warning("Script argument 'save' is ignored")

    bscan_ax, tfm_axes, tfms, views = locate_artefact(dataset_name, save)

    selector = TimeOfFlightSelector(bscan_ax, tfm_axes, tfms, views)
    bscan_selector = selector.children[-1]
    """
    To restart:
    selector.disconnect(); selector = TimeOfFlightSelector(tfm_axes, tfms, views)
    
    To save:
    selector.save()    
    """

    # Block script until windows are closed.
    plt.show()
