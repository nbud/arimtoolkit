"""
Run full forward model, compute TFM on created data and save TFM intensities.

Measurements are made on a fine grid near the defect with Lanczos interpolation.
TFM images of the whole block are performed on a coarser grid with no
interpolation for comparison with experimental images (1).

(1): only if full_tfm=True

Output
------
tfm_singlef.pickle
    Dict of TfmResults
tfm_multif.pickle
    Dict of TfmResults
tfm_singlef_large.pickle
    Dict of TfmResults (1)
tfm_multif_large.pickle
    Dict of TfmResults (1)
tfm_{i:02}_{viewname}_singlef
    TFM images
tfm_{i:02}_{viewname}_multif
    TFM images
intensities_singlef_unscaled.csv
    Columns: view, Model_SingleFreq_Centre, Model_SingleFreq_Max
intensities_multif_unscaled.csv
    Columns: view, Model_SingleFreq_Centre, Model_SingleFreq_Max
    
"""
import math
import logging
from collections import OrderedDict
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import hilbert
import scipy.fftpack

import arim
import arim.model
import arim.scat
import arim.plot as aplt
import arim.models.block_in_immersion as bim
import arim.im
import arim.signal  # for imaging
import arim.scat
import plot_tfms

from . import common

save = True
aplt.conf["savefig"] = True

# For TFM plot
USE_DYNAMIC_SCALE = True
PLOT_DEFECT_BOX = True


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("arim").setLevel(logging.INFO)
logging.getLogger("arim.models").setLevel(logging.DEBUG)

# %%


def model_full(dataset_name, use_multifreq, full_tfm=True):
    # %%
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    probe = common.load_probe(conf)
    examination_object = arim.io.block_in_immersion_from_conf(conf)
    tx, rx = arim.ut.fmc(probe.numelements)
    numscanlines = len(tx)

    scatterers = common.defect_oriented_point(conf)
    grid = common.grid_near_defect(conf)
    grid_p = grid.to_oriented_points()

    probe_p = probe.to_oriented_points()
    views = bim.make_views(
        examination_object,
        probe_p,
        scatterers,
        max_number_of_reflection=1,
        tfm_unique_only=True,
    )
    views_imaging = bim.make_views(
        examination_object,
        probe_p,
        grid_p,
        max_number_of_reflection=1,
        tfm_unique_only=True,
    )

    arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)
    arim.ray.ray_tracing(views_imaging.values(), convert_to_fortran_order=True)

    if full_tfm:
        grid_large = common.make_grid_tfm(conf)
        grid_large_p = grid_large.to_oriented_points()
        views_imaging_large = bim.make_views(
            examination_object,
            probe_p,
            grid_large_p,
            max_number_of_reflection=1,
            tfm_unique_only=True,
        )
        arim.ray.ray_tracing(
            views_imaging_large.values(), convert_to_fortran_order=True
        )

    if use_multifreq:
        multifreq_key = "multif"
        multifreq_key_title = "MultiFreq"
    else:
        multifreq_key = "singlef"
        multifreq_key_title = "SingleFreq"

    # %% Toneburst and time vector

    max_delay = max(
        (
            view.tx_path.rays.times.max() + view.rx_path.rays.times.max()
            for view in views.values()
        )
    )

    numcycles = conf["model"]["toneburst"]["numcycles"]
    centre_freq = common.get_centre_freq(conf, probe)
    dt = 0.25 / centre_freq  # to adjust so that the whole toneburst is sampled
    _tmax = max_delay + 4 * numcycles / centre_freq

    numsamples = scipy.fftpack.next_fast_len(math.ceil(_tmax / dt))
    time = arim.Time(0.0, dt, numsamples)
    freq_array = np.fft.rfftfreq(len(time), dt)
    numfreq = len(freq_array)

    toneburst = arim.model.make_toneburst(
        numcycles, centre_freq, dt, numsamples, wrap=True
    )
    toneburst *= 1.0 / np.abs(hilbert(toneburst)[0])
    toneburst_f = np.fft.rfft(toneburst)

    # plot toneburst
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(1e6 * time.samples, toneburst)
    plt.title("toneburst (time domain)")
    plt.xlabel("time (µs)")

    plt.subplot(1, 2, 2)
    plt.plot(1e-6 * np.fft.rfftfreq(len(toneburst), dt), abs(toneburst_f))
    plt.title("toneburst (frequency domain)")
    plt.xlabel("frequency (MHz)")

    # %% Compute transfer function
    numangles_for_scat_precomp = 180  # 0 to disable

    model_options = dict(
        probe_element_width=probe.dimensions.x[0],
        numangles_for_scat_precomp=numangles_for_scat_precomp,
    )

    scat_obj = arim.scat.scat_factory(
        material=examination_object.block_material, **conf["scatterer"]["specs"]
    )

    scat_angle = np.deg2rad(conf["scatterer"]["angle_deg"])

    transfer_function_f = np.zeros((numscanlines, numfreq), np.complex_)
    tfms = OrderedDict()
    if full_tfm:
        tfms_large = OrderedDict()
    else:
        tfms_large = None

    if use_multifreq:
        # Multi frequency model
        transfer_function_iterator = bim.multifreq_scat_transfer_functions(
            views,
            tx,
            rx,
            freq_array=freq_array,
            scat_obj=scat_obj,
            scat_angle=scat_angle,
            **model_options,
        )
    else:
        # Single frequency model
        transfer_function_iterator = bim.singlefreq_scat_transfer_functions(
            views,
            tx,
            rx,
            freq_array=freq_array,
            scat_obj=scat_obj,
            scat_angle=scat_angle,
            **model_options,
            frequency=common.get_centre_freq(conf, probe),
        )

    with arim.helpers.timeit("Main loop"):
        for viewname, partial_transfer_func in transfer_function_iterator:
            transfer_function_f += partial_transfer_func

            # imaging:
            partial_response = arim.signal.rfft_to_hilbert(
                partial_transfer_func * toneburst_f, numsamples
            )
            partial_frame = arim.Frame(
                partial_response, time, tx, rx, probe, examination_object
            )

            tfms[viewname] = arim.im.tfm.tfm_for_view(
                partial_frame,
                grid,
                views_imaging[viewname],
                interpolation=common.TFM_FINE_INTERP,
                fillvalue=0.0,
            )
            if full_tfm:
                tfms_large[viewname] = arim.im.tfm.tfm_for_view(
                    partial_frame,
                    grid_large,
                    views_imaging_large[viewname],
                    fillvalue=0.0,
                )

    # %% Save raw TFM results
    if save:
        with open(result_dir / f"tfm_{multifreq_key}.pickle", "wb") as f:
            pickle.dump(tfms, f, pickle.HIGHEST_PROTOCOL)
        if full_tfm:
            with open(result_dir / f"tfm_{multifreq_key}_large.pickle", "wb") as f:
                pickle.dump(tfms_large, f, pickle.HIGHEST_PROTOCOL)

    # %% Measure TFM intensities

    tmp = []
    scatterer_idx = grid.closest_point(*scatterers.points[0])

    for viewname, tfm in tfms.items():
        max_tfm_idx = np.argmax(np.abs(tfm.res))
        tmp.append(
            (
                viewname,
                np.abs(tfm.res.flat[scatterer_idx]),
                np.abs(tfm.res.flat[max_tfm_idx]),
                grid.x.flat[max_tfm_idx],
                grid.y.flat[max_tfm_idx],
                grid.z.flat[max_tfm_idx],
            )
        )
    intensities_df = pd.DataFrame(
        tmp,
        columns=[
            "view",
            f"Model_{multifreq_key_title}_Centre",
            f"Model_{multifreq_key_title}_Max",
            "x_max_intensity",
            "y_max_intensity",
            "z_max_intensity",
        ],
    ).set_index("view")

    if save:
        intensities_df.to_csv(
            str(result_dir / f"intensities_{multifreq_key}_unscaled.csv")
        )

    # %% Plot TFM (defect only)
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
        amp = intensities_df.loc[viewname]
        ax.plot(amp["x_max_intensity"], amp["z_max_intensity"], "1m")
        ax.plot(scatterers.points.x, scatterers.points.z, "dm")

    fig.savefig(str(result_dir / f"tfm_model_{multifreq_key}"))

    # %%

    return tfms, tfms_large, intensities_df


if __name__ == "__main__":
    argparser = common.argparser(__doc__)
    argparser.add_argument(
        "--skip-full-tfm",
        action="store_true",
        default=False,
        help="Skip calculation of full TFM images",
    )
    args = argparser.parse_args()

    # flag to run full tfm:
    full_tfm = not (args.skip_full_tfm)

    tfms, tfms_large, intensities_singlef = model_full(
        args.dataset_name, use_multifreq=False, full_tfm=full_tfm
    )
    print(intensities_singlef)
    if full_tfm:
        plot_tfms.plot_tfms_singlef(
            args.dataset_name,
            tfms_large,
            args.save,
            is_paper=args.paper,
            noshow=args.noshow,
        )

    tfms, tfms_large, intensities_multif = model_full(
        args.dataset_name, use_multifreq=True, full_tfm=full_tfm
    )
    print(intensities_multif)
    if full_tfm:
        plot_tfms.plot_tfms_multif(
            args.dataset_name,
            tfms_large,
            args.save,
            is_paper=args.paper,
            noshow=args.noshow,
        )
