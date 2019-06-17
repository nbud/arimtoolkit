"""
Computes for a block in immersion multi-view TFM with uniform amplitudes (21 views).

Input
-----
Conf file

Output
------
tfm_{i:02}_{viewname}
    TFM images
tfm_experimental_large.pickle
    Dict of TfmResults
    
"""
import logging
from collections import OrderedDict
import pickle

import numpy as np
import arim
import arim.ray
import arim.im
import arim.models.block_in_immersion as bim
import arim.plot as aplt

from . import common, plot_tfms

# %% Load configuration

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("arim").setLevel(logging.INFO)


def uniform_tfm(dataset_name, save):
    # %%
    conf = arim.io.load_conf(dataset_name)
    #    conf['grid']['pixel_size'] = 2e-3  # debug
    aplt.conf["savefig"] = save

    result_dir = conf["result_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    # Load frame
    frame = common.load_frame(conf, apply_filter=True, expand=True)

    # Make grid
    z_backwall = conf["backwall"]["z"]
    assert not np.isnan(z_backwall)

    grid = common.make_grid_tfm(conf)
    grid_p = grid.to_oriented_points()
    probe_p = frame.probe.to_oriented_points()

    # Make views
    views = bim.make_views(
        frame.examination_object,
        probe_p,
        grid_p,
        tfm_unique_only=True,
        max_number_of_reflection=1,
    )
    views = common.filter_views(views, conf)

    # %% Perform ray tracing

    arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

    # %% Run TFM

    tfms = OrderedDict()
    for i, view in enumerate(views.values()):
        with arim.helpers.timeit("TFM {}".format(view.name), logger=logger):
            tfms[view.name] = arim.im.tfm.tfm_for_view(frame, grid, view, fillvalue=0.0)

    # %% Save
    if save:
        with open(result_dir / "tfm_experimental_large.pickle", "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(tfms, f, pickle.HIGHEST_PROTOCOL)

    return tfms


# %%
if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save

    tfms = uniform_tfm(dataset_name, save)
    plot_tfms.plot_tfms_experimental(
        dataset_name, tfms, save, is_paper=args.paper, noshow=args.noshow
    )
