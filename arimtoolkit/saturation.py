# -*- coding: utf-8 -*-
"""
Plot a Bscan showing the extrema values to detect saturation.
"""

import numpy as np
import matplotlib.pyplot as plt
import arim
from . import common


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save
    noshow = args.noshow

    conf = arim.io.load_conf(dataset_name)
    root_dir = conf["root_dir"]
    result_dir = conf["result_dir"]

    frame = common.load_frame(
        conf, apply_filter=False, expand=False, warn_if_fallback_vel=False
    )
    scanlines = frame.scanlines[frame.tx == frame.rx]

    cmin = frame.scanlines.min()
    cmax = frame.scanlines.max()

    satmap = np.zeros_like(scanlines)
    satmap[scanlines == cmin] = -1
    satmap[scanlines == cmax] = +1

    sat_ratio = np.sum(np.abs(satmap)) / satmap.size

    plt.figure()
    plt.imshow(satmap, interpolation="none", cmap="coolwarm")
    plt.colorbar()
    plt.axis("auto")
    plt.clim([-1.0, 1.0])
    plt.title(f"Saturated values in Bscan: {sat_ratio*100:0.1f}%")
    plt.xlabel("Time index")
    plt.ylabel("Channel index")
    if save:
        plt.savefig(result_dir / "saturation_map")

    if noshow:
        plt.close("all")
    else:
        plt.show()
