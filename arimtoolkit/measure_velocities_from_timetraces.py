"""
Calculate L and T velocities from LL and LT backwalls.

Raises
------
IndefiniteVelocityError

Output
------
conf.d/30_block_velocities.yaml
velocity_L.png
velocity_T.png

"""
import logging

import numpy as np
import matplotlib.pyplot as plt
import arim
import arim.ray
from arim.im.das import lanczos_interpolation
from tqdm import tqdm
import pandas as pd
import yaml
import arim.models.block_in_immersion as bim
import numba

from . import common

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("arim").setLevel(logging.WARNING)


class IndefiniteVelocityError(RuntimeError):
    pass


def _time_of_flights_backwall_LL(conf):
    probe = common.load_probe(conf)
    examination_object = arim.io.examination_object_from_conf(conf)
    tx_list, rx_list = arim.ut.fmc(probe.numelements)

    # Backwall paths
    backwall_paths = bim.backwall_paths(
        examination_object.couplant_material,
        examination_object.block_material,
        probe.to_oriented_points(),
        examination_object.frontwall,
        examination_object.backwall,
    )

    path = backwall_paths["LL"]
    arim.ray.ray_tracing_for_paths([path])
    return path.rays.times


def _time_of_flights_backwall_LT(conf):
    probe = common.load_probe(conf)
    examination_object = arim.io.examination_object_from_conf(conf)
    tx_list, rx_list = arim.ut.fmc(probe.numelements)

    # Backwall paths
    backwall_paths = bim.backwall_paths(
        examination_object.couplant_material,
        examination_object.block_material,
        probe.to_oriented_points(),
        examination_object.frontwall,
        examination_object.backwall,
    )

    path = backwall_paths["LT"]
    arim.ray.ray_tracing_for_paths([path])
    return path.rays.times


@numba.njit(parallel=True)
def _wall_intensities_lanczos(scanlines, tof_arr, tx, rx, t0, invdt, a):
    res = 0.0
    for scan in range(scanlines.shape[0]):
        tof = tof_arr[tx[scan], rx[scan]]
        tof_idx = (tof - t0) * invdt
        res += lanczos_interpolation(tof_idx, scanlines[scan], a)
    return res


def _wall_intensities(frame, tof_arr):
    return _wall_intensities_lanczos(
        frame.scanlines,
        tof_arr,
        frame.tx,
        frame.rx,
        frame.time.start,
        1 / frame.time.step,
        a=3,
    )


def _measure_l_vel(conf, frame, l_vel_range):
    intensities = []

    for l_vel in tqdm(l_vel_range, desc="L velocity"):
        conf["block_material"]["longitudinal_vel"] = l_vel
        tof = _time_of_flights_backwall_LL(conf)
        intensities.append(_wall_intensities(frame, tof))
    return pd.Series(intensities, index=l_vel_range)


def _measure_t_vel(conf, frame, t_vel_range):
    intensities = []

    for t_vel in tqdm(t_vel_range, desc="T velocity"):
        conf["block_material"]["transverse_vel"] = t_vel
        tof = _time_of_flights_backwall_LT(conf)
        intensities.append(_wall_intensities(frame, tof))
    return pd.Series(intensities, index=t_vel_range)


def measure_velocities_from_timetraces(dataset_name, save, noshow=False):
    """
    maximise Sum_i(Envelope(TimeTrace[tof_backwall_i]))
    """
    conf = arim.io.load_conf(dataset_name)
    # conf["frontwall"]["numpoints"] = 1000
    # conf["backwall"]["numpoints"] = 1000
    root_dir = conf["root_dir"]
    result_dir = conf["result_dir"]

    frame = common.load_frame(
        conf, apply_filter=True, expand=True, warn_if_fallback_vel=False
    )
    frame.scanlines = np.abs(frame.scanlines)

    # === L velocity ===
    # First pass
    base_l_vel = (
        conf["block_material"]["longitudinal_vel"] // 10
    ) * 10  # make round numbers
    l_vel_range_1 = np.arange(base_l_vel - 100, base_l_vel + 100.1, 10.0)
    intensities_1 = _measure_l_vel(conf, frame, l_vel_range_1)
    l_vel_1_idx = intensities_1.values.argmax()

    if l_vel_1_idx == 0 or l_vel_1_idx == (len(l_vel_range_1) - 1):
        # we're on a bound, that's bad
        raise IndefiniteVelocityError

    # Second pass
    l_vel_range_2 = np.arange(
        l_vel_range_1[l_vel_1_idx - 1] + 1, l_vel_range_1[l_vel_1_idx + 1], 1.0
    )
    intensities_2 = _measure_l_vel(conf, frame, l_vel_range_2)

    # agregate results
    intensities = pd.concat([intensities_1, intensities_2]).sort_index()
    l_vel_opt = intensities.idxmax()
    logger.info(f"Optimal L velocitiy: {l_vel_opt} m/s")
    conf["block_material"]["longitudinal_vel"] = l_vel_opt

    # plot
    plt.figure()
    plt.plot(intensities.index, intensities, ".-")
    plt.xlabel("L velocitiy (m/s)")
    plt.ylabel("Backwall LL intensity")
    plt.title(f"Optimum: {l_vel_opt}")
    if save:
        plt.savefig(result_dir / "velocity_L")

    # === T velocity ===
    # First pass
    base_t_vel = (
        conf["block_material"]["transverse_vel"] // 10
    ) * 10  # make round numbers
    t_vel_range_1 = np.arange(base_t_vel - 100, base_t_vel + 100.1, 10.0)
    intensities_1 = _measure_t_vel(conf, frame, t_vel_range_1)
    t_vel_1_idx = intensities_1.values.argmax()

    if t_vel_1_idx == 0 or t_vel_1_idx == (len(t_vel_range_1) - 1):
        # we're on a bound, that's bad
        raise IndefiniteVelocityError

    # Second pass
    t_vel_range_2 = np.arange(
        t_vel_range_1[t_vel_1_idx - 1] + 1, t_vel_range_1[t_vel_1_idx + 1], 1.0
    )
    intensities_2 = _measure_t_vel(conf, frame, t_vel_range_2)

    # agregate results
    intensities = pd.concat([intensities_1, intensities_2]).sort_index()
    t_vel_opt = intensities.idxmax()
    logger.info(f"Optimal T velocitiy: {t_vel_opt} m/s")
    conf["block_material"]["transverse_vel"] = t_vel_opt

    # plot
    plt.figure()
    plt.plot(intensities.index, intensities, ".-")
    plt.xlabel("T velocitiy (m/s)")
    plt.ylabel("Backwall LT intensity")
    plt.title(f"Optimum: {t_vel_opt}")
    if save:
        plt.savefig(result_dir / "velocity_T")

    if save:
        # Save velocities as conf file
        block_conf = dict(
            longitudinal_vel=float(l_vel_opt),
            transverse_vel=float(t_vel_opt),
            metadata=dict(source="Velocities measured from TFM", is_fallback=False),
        )
        block_conf2 = dict(block_material=block_conf)

        with (root_dir / "conf.d/30_block_velocities.yaml").open("w") as f:
            f.write("# generated by measure_velocities_from_timetraces.py\n")
            yaml.dump(block_conf2, f, default_flow_style=False)

    if noshow:
        plt.close("all")
    else:
        plt.show()
    return l_vel_opt, t_vel_opt


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save
    l_vel_opt, t_vel_opt = measure_velocities_from_timetraces(
        dataset_name, save, noshow=args.noshow
    )
