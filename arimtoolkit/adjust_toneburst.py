"""
Adjust the modelled toneburst

Output
------
toneburst.png
conf.d/20_toneburst.yaml

"""

import matplotlib.pyplot as plt
import numpy as np
import logging

import yaml
import arim
import arim.ray
import arim.signal
import arim.im
import arim.plot as aplt
from scipy.signal import welch
from scipy.interpolate import InterpolatedUnivariateSpline

from . import common

# %% helpers

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def norm(x):
    return x / np.max(np.abs(x))


def welch_frame(frame, return_normalised=False, selector=slice(None)):
    x = frame.scanlines[selector]
    f, pxx = welch(x, 1 / frame.time.step)
    if pxx.ndim == 2:
        pxx = np.mean(pxx, axis=0)
    if return_normalised:
        pxx = norm(pxx)
    return f, pxx


def adjust_toneburst(dataset_name, save, numcycles_range=None, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]
    frame_raw = common.load_frame(conf, apply_filter=False, expand=False)
    frame = frame_raw.apply_filter(
        arim.signal.ButterworthBandpass(**conf["filter_for_tfm"], time=frame_raw.time)
    )

    # find centre freq
    exp_f, exp_pxx = welch_frame(frame)
    exp_pxx = norm(exp_pxx)
    centre_freq = exp_f[np.argmax(exp_pxx)]
    logger.info(f"exp centre freq {centre_freq/1e6:.2f} MHz")

    # %% find optimal numcycles
    numsamples = 300
    dt = frame.time.step
    pxx_list = []
    pxx_errors = []
    numcycles_range = np.arange(2.0, 10.0, 0.5)
    for numcycles in numcycles_range:
        toneburst = arim.model.make_toneburst(numcycles, centre_freq, dt, numsamples)
        f, pxx = welch(toneburst, 1 / dt)
        pxx = norm(pxx)
        pxx_list.append(pxx)

        pxx_f = InterpolatedUnivariateSpline(f, pxx, ext=2)
        pxx_errors.append(np.sum(np.abs((pxx_f(exp_f) - exp_pxx))))

    opt_idx = np.argmin(pxx_errors)
    opt_numcycles = numcycles_range[opt_idx]
    opt_pxx = pxx_list[opt_idx]
    logger.info(f"optimal numcycles: {opt_numcycles}")

    # %% plots
    plt.figure()
    plt.plot(numcycles_range, pxx_errors, ".-")
    plt.plot(opt_numcycles, pxx_errors[opt_idx], "d")
    plt.xlabel("numcycles")
    plt.ylabel("errors")

    plt.figure()
    plt.plot(exp_f, exp_pxx, label="exp filtered")
    plt.plot(f, opt_pxx, label=f"model {opt_numcycles} cycles")
    plt.xlabel("freq (MHz)")
    plt.ylabel("normalised spectral density")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(aplt.mega_formatter)
    plt.title(f"modelled centre freq {centre_freq/1e6:.2f} MHz")
    if save:
        plt.savefig(result_dir / "toneburst")
    if noshow:
        plt.close("all")
    else:
        plt.show()

    # %% to yaml
    toneburst_conf = dict(
        model=dict(
            toneburst=dict(
                numcycles=float(opt_numcycles), centre_freq=float(centre_freq)
            )
        )
    )
    if save:
        with (result_dir / "conf.d/20_toneburst.yaml").open("w") as f:
            yaml.dump(toneburst_conf, f, default_flow_style=False)
            logger.info("Wrote conf.d/20_toneburst.yaml")

    return opt_numcycles, centre_freq


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save

    opt_numcycles, centre_freq = adjust_toneburst(
        dataset_name, save, noshow=args.noshow
    )
