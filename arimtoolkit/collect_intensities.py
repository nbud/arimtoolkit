"""
Input
-----
intensities_experimental.csv
intensities_sensitivity_unscaled.csv
intensities_singlef_unscaled.csv
intensities_multif_unscaled.csv
conf[model][reference_amplitude]
conf[experimental][reference_amplitude]

Output
------
intensities.csv :
    columns: view, Experimental, Model_SingleFreq_Max, Model_SingleFreq_Centre, 

"""
import logging

import pandas as pd
import arim

from . import common

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def finalise_intensities(dataset_name, save):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    try:
        exp_ref_amp = conf["experimental"]["reference_amplitude"]
        model_ref_amp = conf["model"]["reference_amplitude"]
    except KeyError:
        exp_ref_amp = 1.0
        model_ref_amp = 1 / conf["model"]["scaling"]
        logger.warning(
            "Missing reference amplitude model scaling. Use model scaling instead."
        )

    df_exp = pd.read_csv(result_dir / "intensities_experimental.csv", index_col=0)
    intensities = df_exp / exp_ref_amp

    df2 = pd.read_csv(result_dir / "intensities_sensitivity_unscaled.csv", index_col=0)
    # scale and save
    intensities["Model_Sensitivity"] = df2["Model_Sensitivity"] / model_ref_amp

    try:
        df3 = pd.read_csv(result_dir / "intensities_singlef_unscaled.csv", index_col=0)
    except FileNotFoundError:
        logger.info("Could not find intensities_singlef_unscaled.csv")
    else:
        # scale and save
        intensities["Model_SingleFreq_Centre"] = (
            df3["Model_SingleFreq_Centre"] / model_ref_amp
        )
        intensities["Model_SingleFreq_Max"] = (
            df3["Model_SingleFreq_Max"] / model_ref_amp
        )

    try:
        df4 = pd.read_csv(result_dir / "intensities_multif_unscaled.csv", index_col=0)
    except FileNotFoundError:
        logger.info("Could not find intensities_multif_unscaled.csv")
    else:
        # ignore useless model Model_MultiFreq_Max
        intensities["Model_MultiFreq_Max"] = df4["Model_MultiFreq_Max"] / model_ref_amp

    if save:
        intensities.to_csv(result_dir / "intensities.csv")

    return intensities


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    intensities = finalise_intensities(args.dataset_name, args.save)
    print(intensities)
