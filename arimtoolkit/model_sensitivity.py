"""
Run sensitivity forward model and save TFM intensities.

Output
------
intensities_sensitivity_unscaled.csv
    Columns: view, Model_Sensitivity
"""
import logging

import arim
import arim.ray
import arim.im
import arim.models.block_in_immersion as bim
import numpy as np
import pandas as pd
import arim.scat

from . import common

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%


def model_sensitivity(dataset_name, save):
    # %%
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    probe = common.load_probe(conf)
    examination_object = arim.io.block_in_immersion_from_conf(conf)
    tx, rx = arim.ut.fmc(probe.numelements)
    numscanlines = len(tx)

    model_options = dict(
        frequency=common.get_centre_freq(conf, probe),
        probe_element_width=probe.dimensions.x[0],
    )

    grid_p = common.defect_oriented_point(conf)
    probe_p = probe.to_oriented_points()
    views = bim.make_views(
        examination_object,
        probe_p,
        grid_p,
        max_number_of_reflection=1,
        tfm_unique_only=True,
    )
    arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

    scat_obj = arim.scat.scat_factory(
        **conf["scatterer"]["specs"], material=examination_object.block_material
    )
    scat_angle = np.deg2rad(conf["scatterer"]["angle_deg"])
    with arim.helpers.timeit("Scattering matrices", logger=logger):
        # scat_mat = scat_obj.as_single_freq_matrices(model_options['frequency'], 360) # use precomputation
        scat_mat = scat_obj.as_angles_funcs(
            model_options["frequency"]
        )  # no precomputation

    with arim.helpers.timeit("Computation of ray weights for all paths", logger=logger):
        ray_weights = bim.ray_weights_for_views(views, **model_options)

    theoretical_intensities_dict = dict()
    scanline_weights = np.ones(numscanlines)

    for viewname, view in views.items():
        model_coefficients = arim.model.model_amplitudes_factory(
            tx.astype(int),
            rx.astype(int),
            view,
            ray_weights,
            scat_mat,
            scat_angle=scat_angle,
        )

        # shape: numpoints, numscanlines
        theoretical_intensities_dict[
            viewname
        ] = model_coefficients.sensitivity_uniform_tfm(scanline_weights)
        # ax, _ = aplt.plot_oxz(np.abs(theoretical_intensities_dict[viewname]), grid, scale='linear', title=viewname)

    # %%
    data = []
    for viewname, th_amp in theoretical_intensities_dict.items():
        data.append((viewname, np.abs(th_amp[0])))

    intensities = pd.DataFrame(data, columns=("view", "Model_Sensitivity")).set_index(
        "view"
    )
    if save:
        intensities.to_csv(result_dir / "intensities_sensitivity_unscaled.csv")
    # %%
    return intensities


# %%
if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save
    intensities = model_sensitivity(dataset_name, save)
    print(intensities)
