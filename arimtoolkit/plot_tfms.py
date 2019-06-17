# -*- coding: utf-8 -*-
"""
Create figures for paper

Input
=====
tfm_experimental_large.pickle
    Dict of TfmResults
tfm_singlef_large.pickle
    Dict of TfmResults
tfm_multif_large.pickle
    Dict of TfmResults

Output
======
tfm_{i:02}_{viewname}
    TFM images
tfm_{i:02}_{viewname}_singlef
    TFM images
tfm_{i:02}_{viewname}_multif
    TFM images

"""

import functools
import contextlib
import itertools
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import arim.plot as aplt
import arim

from . import common

USE_DYNAMIC_SCALE = True
PLOT_DEFECT_BOX = True


def plot_tfms(
    dataset_name, tfms, save, suffix, is_paper=False, noshow=False, db_range=40.0
):
    """
    Plot TFMs and save them in result directory
    """
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    if is_paper:
        style_context = mpl.style.context("to_pdf.mplstyle")
        figsize = (5, 2.5)
        use_cbar = False
        title = None
        interpolation = "none"
    else:
        style_context = contextlib.suppress()  # nullcontext
        figsize = (10, 5)
        use_cbar = True
        interpolation = None

    # get an element
    tfm = next(iter(tfms.values()))
    grid = tfm.grid
    try:
        reference_rect = common.reference_rect(conf)
        reference_area = grid.points_in_rectbox(**reference_rect)
    except common.NoDefect:
        reference_rect = None
        reference_area = None
    if USE_DYNAMIC_SCALE:
        scale = aplt.common_dynamic_db_scale(
            [tfm.res for tfm in tfms.values()], reference_area, db_range=db_range
        )
    else:
        scale = itertools.repeat((None, [-db_range, 0.0]))

    with style_context:
        for i, (viewname, tfm) in enumerate(tfms.items()):
            assert tfm.grid is grid

            ref_db, clim = next(scale)

            if PLOT_DEFECT_BOX and reference_rect is not None:
                patches = [common.rect_to_patch(reference_rect)]
            else:
                patches = None

            if not (is_paper):
                title = f"TFM {viewname}"

            ax, _ = aplt.plot_tfm(
                tfm,
                clim=clim,
                scale="db",
                ref_db=ref_db,
                title=title,
                savefig=False,
                figsize=figsize,
                patches=patches,
                draw_cbar=use_cbar,
                interpolation=interpolation,
            )
            ax.set_adjustable("box")
            ax.axis([grid.xmin, grid.xmax, grid.zmax, 0])
            if save:
                ax.figure.savefig(str(result_dir / f"tfm_{i:02}_{viewname}{suffix}"))
            if noshow:
                plt.close(ax.figure)
    if noshow:
        plt.close("all")
    else:
        plt.show()


plot_tfms_experimental = functools.partial(plot_tfms, suffix="")
plot_tfms_singlef = functools.partial(plot_tfms, suffix="_singlef")
plot_tfms_multif = functools.partial(plot_tfms, suffix="_multif")


def _load_tfms(dataset_name, save, pickle_filename):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    with open(result_dir / pickle_filename, "rb") as f:
        tfms = pickle.load(f)

    return tfms


def load_and_plot_tfms_experimental(dataset_name, save, is_paper=False, noshow=False):
    tfms = _load_tfms(dataset_name, save, "tfm_experimental_large.pickle")
    plot_tfms_experimental(dataset_name, tfms, save, is_paper=is_paper, noshow=noshow)


def load_and_plot_tfms_singlef(dataset_name, save, is_paper=False, noshow=False):
    tfms = _load_tfms(dataset_name, save, "tfm_singlef_large.pickle")
    plot_tfms_singlef(dataset_name, tfms, save, is_paper=is_paper, noshow=noshow)


def load_and_plot_tfms_multif(dataset_name, save, is_paper=False, noshow=False):
    tfms = _load_tfms(dataset_name, save, "tfm_multif_large.pickle")
    plot_tfms_multif(dataset_name, tfms, save, is_paper=is_paper, noshow=noshow)


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()

    dataset_name = args.dataset_name
    save = args.save
    is_paper = args.paper
    noshow = args.noshow
    load_and_plot_tfms_experimental(dataset_name, save, is_paper, noshow)
    load_and_plot_tfms_singlef(dataset_name, save, is_paper, noshow)
    load_and_plot_tfms_multif(dataset_name, save, is_paper, noshow)
