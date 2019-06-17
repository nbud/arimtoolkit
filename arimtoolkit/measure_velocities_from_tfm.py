"""
Calculate L and T velocities from LL and LT backwalls.

Raises
------
IndefiniteAttenuationError

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
from tqdm import tqdm
import pandas as pd
import yaml

from . import common, tfm_walls

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("arim").setLevel(logging.WARNING)


class IndefiniteVelocityError(RuntimeError):
    pass


def _measure_l_vel(conf, frame, l_vel_range):
    intensities = []

    for l_vel in tqdm(l_vel_range, desc="L velocity"):
        # make backwall LL view
        conf["block_material"]["longitudinal_vel"] = l_vel
        grids, views = tfm_walls.wall_views(conf, use_line_grids=True)
        grid = grids["backwall_LL"]
        view = views["backwall_LL"]
        arim.ray.ray_tracing([view])

        tfm = arim.im.tfm.tfm_for_view(
            frame, grid, view, interpolation=common.TFM_FINE_INTERP
        )
        intensities.append(np.sum(np.abs(tfm.res)))
    return pd.Series(intensities, index=l_vel_range)


def _measure_t_vel(conf, frame, t_vel_range):
    intensities = []

    for t_vel in tqdm(t_vel_range, desc="T velocity"):
        # make backwall LL view
        conf["block_material"]["transverse_vel"] = t_vel
        grids, views = tfm_walls.wall_views(conf, use_line_grids=True)
        grid = grids["backwall_LT"]
        view = views["backwall_LT"]
        arim.ray.ray_tracing([view])

        tfm = arim.im.tfm.tfm_for_view(
            frame, grid, view, interpolation=common.TFM_FINE_INTERP
        )
        intensities.append(np.sum(np.abs(tfm.res)))
    return pd.Series(intensities, index=t_vel_range)


def measure_velocities_from_tfm(dataset_name, save, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    root_dir = conf["root_dir"]
    result_dir = conf["result_dir"]

    frame = common.load_frame(
        conf, apply_filter=True, expand=True, warn_if_fallback_vel=False
    )

    # === L velocity ===
    # First pass
    base_l_vel = (
        conf["block_material"]["longitudinal_vel"] // 10
    ) * 10  # make round numbers
    l_vel_range_1 = np.arange(base_l_vel - 200., base_l_vel + 200.1, 10.0)
    intensities_1 = _measure_l_vel(conf, frame, l_vel_range_1)
    l_vel_1_idx = intensities_1.values.argmax()

    if l_vel_1_idx == 0 or l_vel_1_idx == (len(l_vel_range_1) - 1):
        # we're on a bound, that's bad
        plt.figure()
        plt.plot(l_vel_range_1, intensities_1, ".-")
        plt.xlabel("L velocitiy (m/s)")
        plt.ylabel("Backwall LL intensity")
        plt.title(f"Cannot find optimum")
        if save:
            plt.savefig(result_dir / "velocity_L")
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
        plt.figure()
        plt.plot(t_vel_range_1, intensities_1, ".-")
        plt.xlabel("T velocitiy (m/s)")
        plt.ylabel("Backwall LT intensity")
        plt.title(f"Cannot find optimum")
        if save:
            plt.savefig(result_dir / "velocity_L")
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
            f.write("# generated by measure_velocities_from_tfm.py\n")
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
    l_vel_opt, t_vel_opt = measure_velocities_from_tfm(
        dataset_name, save, noshow=args.noshow
    )
