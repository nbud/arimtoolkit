"""
Measure experimental TFM intensities of defect

Output
------
{result_directory}/tfm_near_defect.png
{result_directory}/intensities_experimental.csv
    Columns: view, Experimental

"""
import math
import pickle
import logging

import arim
import arim.ray
import arim.im
import arim.plot as aplt
import arim.models.block_in_immersion as bim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def measure_tfm_intensity(dataset_name, save, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    frame = common.load_frame(conf, apply_filter=True, expand=True)

    grid = common.grid_near_defect(conf)
    grid_p = grid.to_oriented_points()

    probe_p = frame.probe.to_oriented_points()
    views = bim.make_views(
        frame.examination_object,
        probe_p,
        grid_p,
        max_number_of_reflection=1,
        tfm_unique_only=True,
    )
    views = common.filter_views(views, conf)

    # %%
    arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

    # %% Run TFM
    tfms = dict()
    for viewname, view in views.items():
        tfms[viewname] = arim.im.tfm.tfm_for_view(
            frame, grid, view, interpolation=common.TFM_FINE_INTERP, fillvalue=np.nan
        )

    # %% Save raw TFM results
    if save:
        with open(result_dir / "tfm_experimental.pickle", "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(tfms, f, pickle.HIGHEST_PROTOCOL)

    # %% Plot
    scale_tfm = aplt.common_dynamic_db_scale([tfm.res for tfm in tfms.values()])
    # scale_tfm = itertools.repeat((None, None))

    ncols = 6
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=math.ceil(len(tfms) / ncols),
        figsize=(16, 9),
        sharex=True,
        sharey=True,
    )

    for (viewname, tfm), ax in zip(tfms.items(), axes.ravel()):
        ref_db, clim = next(scale_tfm)
        aplt.plot_tfm(
            tfm,
            ax=ax,
            scale="db",
            ref_db=ref_db,
            clim=clim,
            interpolation="none",
            savefig=False,
        )
        ax.set_title(viewname)

        if ax in axes[-1, :]:
            ax.set_xlabel("x (mm)")
        else:
            ax.set_xlabel("")
        if ax in axes[:, 0]:
            ax.set_ylabel("z (mm)")
        else:
            ax.set_ylabel("")

    if save:
        fig.savefig(str(result_dir / "tfm_near_defect"))
    if noshow:
        plt.close("all")
    else:
        plt.show()

    # %% Measure amplitudes and save and save as csv
    data = []
    for viewname, tfm in tfms.items():
        data.append((viewname, np.max(np.abs(tfm.res))))

    intensities = pd.DataFrame(data, columns=("view", "Experimental")).set_index("view")
    if save:
        intensities.to_csv(result_dir / "intensities_experimental.csv")
    return tfms, intensities


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    if args.dataset_name == "*":
        for dataset_name in common.all_datasets():
            tfms, intensities = measure_tfm_intensity(
                dataset_name, args.save, noshow=args.noshow
            )
            print(intensities)
            plt.close("all")
    else:
        tfms, intensities = measure_tfm_intensity(
            args.dataset_name, args.save, noshow=args.noshow
        )
        print(intensities)
