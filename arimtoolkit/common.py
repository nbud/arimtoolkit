"""
Helpers and useful functions
"""
import argparse
import numpy as np
import arim
import arim.signal
import arim.measurement
import logging
import matplotlib as mpl

from .downsample_frame import downsample_frame

MEASUREMENT_DX = 4.0e-3
MEASUREMENT_DZ = 4.0e-3

TFM_FINE_INTERP = ("lanczos", 3)
TFM_FAST_INTERP = "nearest"

logging.getLogger("arim").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def argparser(doc=None):
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("dataset_name")
    parser.add_argument(
        "-s", "--save", action="store_true", default=False, help="save results"
    )
    parser.add_argument(
        "--paper", action="store_true", default=False, help="paper output"
    )
    parser.add_argument(
        "--noshow", action="store_true", default=False, help="do no show figures"
    )
    return parser


def load_frame(
    conf,
    apply_filter=True,
    expand=True,
    warn_if_fallback_vel=True,
    use_probe_from_conf=True,
):
    frame = arim.io.frame_from_conf(conf, use_probe_from_conf=use_probe_from_conf)
    try:
        elements_idx = conf["subprobe"]
    except KeyError:
        pass
    else:
        frame = frame.subframe_from_probe_elements(elements_idx, True)
    block = frame.examination_object.block_material
    if warn_if_fallback_vel:
        if block.metadata.get("is_fallback", False):
            logger.warning(
                "Accurate block material data not available, using fallback data"
            )
    if apply_filter:
        frame = frame.apply_filter(
            (
                arim.signal.Hilbert()
                + arim.signal.ButterworthBandpass(
                    **conf["filter_for_tfm"], time=frame.time
                )
            )
        )
    if expand:
        frame = frame.expand_frame_assuming_reciprocity()

    downsample_frame_k = conf.get("downsample_frame", None)
    if downsample_frame_k is not None:
        frame = downsample_frame(frame, downsample_frame_k)

    return frame


def load_probe(conf, **kwargs):
    probe = arim.io.probe_from_conf(conf, **kwargs)
    try:
        elements_idx = conf["subprobe"]
    except KeyError:
        return probe
    else:
        return probe.subprobe(elements_idx)


def make_grid_tfm(conf, extra=5e-3):
    grid_conf = conf["grid"].copy()
    pixel_size = grid_conf["pixel_size"]
    if grid_conf.get("zmin", None) is None:
        grid_conf["zmin"] = conf["frontwall"]["z"] + pixel_size
    if grid_conf.get("zmax", None) is None:
        grid_conf["zmax"] = conf["backwall"]["z"] - pixel_size + extra
    return arim.Grid(**grid_conf, ymin=0.0, ymax=0.0)


def rect_to_patch(rect, fill=False, edgecolor="magenta"):
    return mpl.patches.Rectangle(
        (rect["xmin"], rect["zmin"]),
        rect["xmax"] - rect["xmin"],
        rect["zmax"] - rect["zmin"],
        fill=fill,
        edgecolor=edgecolor,
    )


class NoDefect(KeyError):
    pass


def defect_oriented_point(conf):
    try:
        defect_centre = conf["scatterer"]["location"]
    except KeyError as e:
        raise NoDefect from e

    return arim.geometry.default_oriented_points(
        arim.geometry.Points(
            [[defect_centre["x"], defect_centre["y"], defect_centre["z"]]],
            name="Defect",
        )
    )


def grid_near_defect(conf, pixel_size=0.15e-3):
    """
    return a grid centered at the defect
    """
    defect_centre = conf["scatterer"]["location"]
    return arim.Grid.grid_centred_at_point(
        defect_centre["x"],
        defect_centre["y"],
        defect_centre["z"],
        MEASUREMENT_DX,
        0.0,
        MEASUREMENT_DZ,
        pixel_size,
    )


def reference_rect(conf):
    try:
        defect_centre = conf["scatterer"]["location"]
    except KeyError as e:
        raise NoDefect from e

    return dict(
        xmin=defect_centre["x"] - MEASUREMENT_DX / 2,
        xmax=defect_centre["x"] + MEASUREMENT_DX / 2,
        zmin=defect_centre["z"] - MEASUREMENT_DZ / 2,
        zmax=defect_centre["z"] + MEASUREMENT_DZ / 2,
    )


def get_centre_freq(conf, probe):
    try:
        return conf["model"]["toneburst"]["centre_freq"]
    except KeyError:
        logger.warning(
            "missing toneburst centre frequency, use probe centre freq as fallback"
        )
        return probe.frequency


def rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def db(x):
    return 20 * np.log10(np.abs(x))


def filter_views(views_dict, conf):
    views_to_use = conf.get("views_to_use", "all")
    if views_to_use == "all":
        return views_dict
    else:
        return {
            viewname: val
            for viewname, val in views_dict.items()
            if viewname in views_to_use
        }
