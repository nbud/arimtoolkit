# -*- coding: utf-8 -*-
"""
Measure TFM walls intensities from experimental data and model

Intensities are scaled to reference amplitudes.

Output
------
wall_intensities_model.csv
wall_intensities_exp.csv
    Index: frontwall, backwall_LL, ...
    Columns: l1, l2, max
wall_intensities.png

"""
import math
import logging

import numpy as np
import scipy
import pandas as pd
import arim
import arim.ray
import arim.im
import arim.plot as aplt
import arim.models.block_in_immersion as bim
import matplotlib.pyplot as plt

from . import common

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# supported walls
WALL_KEYS = frozenset(("frontwall", "backwall_LL", "backwall_LT", "backwall_LLLL"))


def wall_views(conf, use_line_grids=False, wall_keys=WALL_KEYS):
    """
    Create views for imaging walls

    Returns
    -------
    grids : list of Grid
    views : list of Views
        keys: frontwall, backwall_LL, backwall_LT, backwall_LLLL
    """
    wall_keys = set(wall_keys)
    unknown_wall_keys = wall_keys - WALL_KEYS
    if unknown_wall_keys:
        raise ValueError(f"Unknown wall keys: {unknown_wall_keys}")

    probe = common.load_probe(conf)
    probe_p = probe.to_oriented_points()

    examination_object = arim.io.examination_object_from_conf(conf)

    if use_line_grids:
        dz = 0.0
    else:
        dz = 1.5e-3
    z_backwall = conf["backwall"]["z"]
    grid_kwargs = dict(
        xmin=probe.locations.x.min(),
        xmax=probe.locations.x.max(),
        ymin=0.0,
        ymax=0.0,
        pixel_size=0.15e-3,
    )
    grid_front = arim.geometry.Grid(**grid_kwargs, zmin=-dz, zmax=dz)
    grid_back = arim.geometry.Grid(
        **grid_kwargs, zmin=z_backwall - dz, zmax=z_backwall + dz
    )
    assert not use_line_grids or grid_front.shape[2] == 1
    assert not use_line_grids or grid_back.shape[2] == 1

    front_views = bim.make_views(
        examination_object,
        probe_p,
        grid_front.to_oriented_points(),
        max_number_of_reflection=0,
        tfm_unique_only=True,
    )
    back_views = bim.make_views(
        examination_object,
        probe_p,
        grid_back.to_oriented_points(),
        max_number_of_reflection=2,
        tfm_unique_only=True,
    )
    views = {
        "frontwall": front_views["L-L"],
        "backwall_LL": back_views["L-L"],
        "backwall_LT": back_views["L-T"],
        "backwall_LLLL": back_views["LLL-L"],
    }
    grids = {
        "frontwall": grid_front,
        "backwall_LL": grid_back,
        "backwall_LT": grid_back,
        "backwall_LLLL": grid_back,
    }
    grids = {k: v for k, v in grids.items() if k in wall_keys}
    views = {k: v for k, v in views.items() if k in wall_keys}
    return grids, views


def make_model_walls(conf, use_multifreq=False, wall_keys=WALL_KEYS):
    """
    Generate a FMC containing the walls

    Parameters
    ----------
    conf : dict
    wall_keys : set
    use_multifreq : bool

    Returns
    -------
    Frame
    """
    probe = common.load_probe(conf)
    examination_object = arim.io.examination_object_from_conf(conf)
    tx_list, rx_list = arim.ut.fmc(probe.numelements)
    wall_keys = set(wall_keys)

    model_options = dict(
        probe_element_width=probe.dimensions.x[0],
        use_directivity=True,
        use_beamspread=True,
        use_transrefl=True,
        use_attenuation=True,
    )

    unknown_wall_keys = wall_keys - WALL_KEYS
    if unknown_wall_keys:
        raise ValueError(f"Unknown wall keys: {unknown_wall_keys}")

    # used paths:
    wall_paths = []

    if "frontwall" in wall_keys:
        # Frontwall path
        frontwall_path = bim.frontwall_path(
            examination_object.couplant_material,
            examination_object.block_material,
            *probe.to_oriented_points(),
            *examination_object.frontwall,
        )
        wall_paths.append(frontwall_path)

    # Backwall paths
    backwall_paths = bim.backwall_paths(
        examination_object.couplant_material,
        examination_object.block_material,
        probe.to_oriented_points(),
        examination_object.frontwall,
        examination_object.backwall,
        max_number_of_reflection=2,
    )
    if "backwall_LL" in wall_keys:
        wall_paths.append(backwall_paths["LL"])
    if "backwall_LT" in wall_keys:
        wall_paths.append(backwall_paths["LT"])
        wall_paths.append(backwall_paths["TL"])
    if "backwall_LLLL" in wall_keys:
        wall_paths.append(backwall_paths["LLLL"])

    arim.ray.ray_tracing_for_paths(wall_paths)

    # Toneburst
    numcycles = conf["model"]["toneburst"]["numcycles"]
    centre_freq = common.get_centre_freq(conf, probe)
    max_delay = max(path.rays.times.max() for path in wall_paths)
    dt = 0.25 / centre_freq  # to adjust so that the whole toneburst is sampled
    _tmax = max_delay + 4 * numcycles / centre_freq
    numsamples = scipy.fftpack.next_fast_len(math.ceil(_tmax / dt))
    time = arim.Time(0.0, dt, numsamples)
    freq_array = np.fft.rfftfreq(len(time), dt)
    toneburst = arim.model.make_toneburst(
        numcycles, centre_freq, dt, numsamples, wrap=True
    )
    toneburst_f = np.fft.rfft(toneburst)

    # convert to dict due to bim API
    wall_paths_dict = dict(zip(range(len(wall_paths)), wall_paths))
    if use_multifreq:
        transfer_function_iterator = bim.multifreq_wall_transfer_functions(
            wall_paths_dict, tx_list, rx_list, freq_array, **model_options
        )
    else:
        transfer_function_iterator = bim.singlefreq_wall_transfer_functions(
            wall_paths_dict, tx_list, rx_list, centre_freq, freq_array, **model_options
        )

    scanlines = None
    for _, transfer_function_wall_f in transfer_function_iterator:
        tmp_scanlines = arim.signal.rfft_to_hilbert(
            transfer_function_wall_f * toneburst_f, numsamples, axis=-1
        )
        if scanlines is None:
            scanlines = tmp_scanlines
        else:
            scanlines += tmp_scanlines

    return arim.Frame(scanlines, time, tx_list, rx_list, probe, examination_object)


def tfm_walls(dataset_name, save, use_multifreq, noshow=False):
    """
    Measure TFM intensities of walls for experimental and modelled data
    """
    logger.info(f"dataset_name: {dataset_name}")
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    grids, views = wall_views(conf, use_line_grids=True)
    arim.ray.ray_tracing(views.values())

    # %% Prepare frames
    exp_frame = common.load_frame(conf, apply_filter=True, expand=True)
    model_frame = make_model_walls(conf, use_multifreq=use_multifreq)

    try:
        exp_ref_amp = conf["experimental"]["reference_amplitude"]
        model_ref_amp = conf["model"]["reference_amplitude"]
    except KeyError:
        exp_ref_amp = 1.0
        model_ref_amp = 1 / conf["model"]["scaling"]
        logger.warning(
            "Missing reference amplitude model scaling. Use model scaling instead."
        )

    # %% Calculate TFM
    exp_tfms = {}
    model_tfms = {}

    for wall_key, view in views.items():
        grid = grids[wall_key]
        exp_tfm = arim.im.tfm.tfm_for_view(
            exp_frame, grid, view, interpolation=common.TFM_FINE_INTERP
        )
        exp_tfms[wall_key] = exp_tfm
        model_tfm = arim.im.tfm.tfm_for_view(
            model_frame, grid, view, interpolation=common.TFM_FINE_INTERP
        )
        model_tfms[wall_key] = model_tfm

    # %% Plot wall intensities
    fig, axes = plt.subplots(nrows=len(exp_tfms), squeeze=True, sharex=True)

    for i, (wall_key, exp_tfm) in enumerate(exp_tfms.items()):
        model_tfm = model_tfms[wall_key]
        x = exp_tfm.grid.x.ravel()

        exp_tfm_amp = common.db(exp_tfm.res.ravel() / exp_ref_amp)
        model_tfm_amp = common.db(model_tfm.res.ravel() / model_ref_amp)

        plt.sca(axes[i])
        plt.plot(x, exp_tfm_amp, label="experimental")
        plt.plot(x, model_tfm_amp, label="modelled")
        plt.title(wall_key)
        if i == 0:
            plt.legend()
    plt.suptitle("TFM intensity (dB)")
    plt.xlabel("x (mm)")
    plt.gca().xaxis.set_major_formatter(aplt.milli_formatter)
    if save:
        plt.savefig(result_dir / "wall_intensities")

    # %% Calculate metrics and save
    def make_metrics(tfms, ref_amp, suffix):
        res = {}
        for wall_key, tfm in tfms.items():
            amp = np.abs(tfm.res.ravel()) / ref_amp
            res[wall_key] = {"rms": common.rms(amp), "max": np.max(amp)}
        df = pd.DataFrame(res)
        if save:
            df.to_csv(result_dir / f"wall_intensities_{suffix}.csv")
        return df

    exp_wall_df = make_metrics(exp_tfms, exp_ref_amp, "exp")
    model_wall_df = make_metrics(model_tfms, model_ref_amp, "model")

    if noshow:
        plt.close("all")
    else:
        plt.show()

    return exp_wall_df, model_wall_df


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    use_multifreq = False
    exp_wall_df, model_wall_df = tfm_walls(
        args.dataset_name, args.save, use_multifreq=use_multifreq, noshow=args.noshow
    )
    print("=== Model (dB):")
    print(common.db(model_wall_df))
    print()
    print("=== Experimental (dB):")
    print(common.db(exp_wall_df))
    print()
    print("=== Exp - Model (dB):")
    print(common.db(exp_wall_df / model_wall_df))
