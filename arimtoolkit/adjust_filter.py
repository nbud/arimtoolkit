"""
Check the filter

Output
------
filter.png
filtered_ascan.png

"""

import matplotlib.pyplot as plt
import logging
import numpy as np

import arim
import arim.ray
import arim.signal
import arim.im
import arim.plot as aplt

from . import common, adjust_toneburst

# %% helpers

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def adjust_filter(dataset_name, save, conf_filter=None, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    # Load frame
    frame_raw = common.load_frame(conf, apply_filter=False, expand=False)

    if conf_filter is None:
        conf_filter = conf["filter_for_tfm"]
    frame = frame_raw.apply_filter(
        arim.signal.ButterworthBandpass(**conf_filter, time=frame_raw.time)
    )
    # aplt.plot_bscan_pulse_echo(frame)

    kwargs = dict(return_normalised=False)
    # kwargs = dict(return_normalised=False, selector=frame.tx==frame.rx)
    # kwargs = dict(return_normalised=False, selector=(slice(None), slice(None, 600)))
    f, pxx_raw = adjust_toneburst.welch_frame(frame_raw, **kwargs)
    f, pxx = adjust_toneburst.welch_frame(frame, **kwargs)

    # %% Plot PSD
    plt.figure()
    plt.plot(f, pxx_raw, label="raw")
    plt.plot(f, pxx, label="filtered")
    plt.legend()
    plt.title(repr(conf_filter))
    plt.gca().xaxis.set_major_formatter(aplt.mega_formatter)
    plt.xlabel("frequency (MHz)")
    plt.ylabel("power spectral density")
    if save:
        plt.savefig(result_dir / "filter")
    if noshow:
        plt.close("all")
    else:
        plt.show()

    # %% Plot Ascan
    max_idx = np.argmax(np.abs(frame_raw.scanlines[0]))
    inds = slice(max(0, max_idx - 50), max_idx + 150)
    plt.figure()
    plt.plot(
        frame_raw.time.samples[inds], frame_raw.scanlines.real[0][inds], label="raw"
    )
    plt.plot(frame.time.samples[inds], frame.scanlines.real[0][inds], label="filtered")
    plt.gca().xaxis.set_major_formatter(aplt.micro_formatter)
    plt.legend()
    plt.xlabel("time (µs)")
    plt.ylabel("Ascan amplitude")

    if save:
        plt.savefig(result_dir / "filtered_ascan")
    if noshow:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save
    noshow = args.noshow

    conf_filter = None
    # conf_filter = conf_filter = dict(order=4, cutoff_min=1e6, cutoff_max=4e6)
    adjust_filter(dataset_name, save, conf_filter, noshow)
