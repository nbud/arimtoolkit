"""Plot Ascan and Bscan and show wall and defect modes on top.
    
"""
import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt
import arim
import arim.model
import arim.plot as aplt
import arim.models.block_in_immersion as bim

from . import common


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FIGSIZE = [9.6, 7.2]


def plot_timetraces_full(dataset_name, save, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    frame = common.load_frame(conf)
    examination_object = arim.io.block_in_immersion_from_conf(conf)
    probe_p = frame.probe.to_oriented_points()
    scatterers = common.defect_oriented_point(conf)
    views = bim.make_views(
        examination_object,
        probe_p,
        scatterers,
        max_number_of_reflection=1,
        tfm_unique_only=False,
    )
    arim.ray.ray_tracing(views.values())

    frontwall_path = bim.frontwall_path(
        examination_object.couplant_material,
        examination_object.block_material,
        *frame.probe.to_oriented_points(),
        *examination_object.frontwall,
    )
    backwall_paths = bim.backwall_paths(
        examination_object.couplant_material,
        examination_object.block_material,
        frame.probe.to_oriented_points(),
        examination_object.frontwall,
        examination_object.backwall,
        max_number_of_reflection=1,
    )
    wall_paths = [frontwall_path] + list(backwall_paths.values())
    arim.ray.ray_tracing_for_paths(wall_paths)

    def make_linestyles():
        linestyles = itertools.product(
            ["-", "--", "-.", ":"], plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )
        return itertools.cycle(linestyles)

    #%% Ascan
    def plot_ascan(tx, rx):
        scanline = np.abs(frame.get_scanline(tx, rx))
        plt.figure(figsize=FIGSIZE)
        plt.plot(frame.time.samples, scanline, "k")
        plt.xlabel("Time (µs)")
        plt.title(f"tx={tx}, rx={rx}")

        linestyle_cycler = make_linestyles()
        for path, (ls, color) in zip(wall_paths, linestyle_cycler):
            plt.axvline(
                path.rays.times[tx, rx],
                label=path.name,
                ymin=0,
                color=color,
                ls=ls,
                lw=2.5,
            )
        for view, (ls, color) in zip(views.values(), linestyle_cycler):
            plt.axvline(
                view.tx_path.rays.times[tx, 0] + view.rx_path.rays.times[rx, 0],
                label=view.name,
                ymin=0,
                color=color,
                ls=ls,
            )
        plt.gca().xaxis.set_major_formatter(aplt.micro_formatter)
        plt.gca().xaxis.set_minor_formatter(aplt.micro_formatter)
        plt.ylim([scanline.min(), scanline.max()])
        plt.legend()
        if save:
            plt.savefig(result_dir / f"ascan_full_{tx}_{rx}")

    plot_ascan(0, 0)
    plot_ascan(0, frame.probe.numelements - 1)
    #%% Bscan
    def plot_bscan(scanlines_idx, filename, **kwargs):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        aplt.plot_bscan(
            frame, scanlines_idx, ax=ax, savefig=False, cmap="Greys", **kwargs
        )
        linestyle_cycler = make_linestyles()
        tx_arr = frame.tx[scanlines_idx]
        rx_arr = frame.rx[scanlines_idx]
        y = np.arange(len(tx_arr))
        for path, (ls, color) in zip(wall_paths, linestyle_cycler):
            plt.plot(
                path.rays.times[tx_arr, rx_arr],
                y,
                label=path.name,
                color=color,
                ls=ls,
                lw=3.0,
            )
        for view, (ls, color) in zip(views.values(), linestyle_cycler):
            plt.plot(
                view.tx_path.rays.times[tx_arr, 0] + view.rx_path.rays.times[rx_arr, 0],
                y,
                label=view.name,
                color=color,
                ls=ls,
            )
        plt.legend()
        if save:
            plt.savefig(result_dir / filename)

    plot_bscan(frame.tx == frame.rx, "bscan_full_1")
    plot_bscan(frame.tx == 0, "bscan_full_2")
    plot_bscan(frame.tx == frame.probe.numelements - 1, "bscan_full_3")

    #%%
    if noshow:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    argparser = common.argparser(__doc__)
    args = argparser.parse_args()
    plot_timetraces_full(args.dataset_name, args.save, args.noshow)
