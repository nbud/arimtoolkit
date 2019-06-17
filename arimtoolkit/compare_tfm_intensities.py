"""
Generate figures to compare TFM intensities between experiments and models
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arim

from . import common


def compare_tfm_intensities(dataset_name, save, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    intensities = pd.read_csv(result_dir / "intensities.csv", index_col=0)
    viewnames = list(intensities.index)

    intensities_db = 20 * np.log10(np.abs(intensities))
    intensities_db.sort_values(by="Experimental", ascending=False)
    print("===== Summary of intensities_db =====")
    print(intensities_db.describe())

    # %% Count views within 3 dB agreement
    err_1 = (intensities_db.Model_MultiFreq_Max - intensities_db.Experimental).abs()
    agreement_1 = err_1[~err_1.isna()] < 3
    print(
        "=== 3 dB agreement between Model_MultiFreq_Max "
        f"and Experimental: {agreement_1.sum()}/{agreement_1.count()}"
    )

    err_2 = (
        intensities_db.Model_MultiFreq_Max - intensities_db.Model_SingleFreq_Centre
    ).abs()
    agreement_2 = err_2[~err_2.isna()] < 3
    print(
        "=== 3 dB agreement between Model_MultiFreq_Max "
        f"and Model_SingleFreq_Centre: {agreement_2.sum()}/{agreement_2.count()}"
    )

    # %% Check Model_Sensitivity and Model_SingleFreq_Centre are equal

    intensities_db[["Model_Sensitivity", "Model_SingleFreq_Centre"]].plot(kind="bar")
    plt.legend()
    plt.ylabel("amplitude dB")
    plt.xticks(range(len(viewnames)), viewnames, rotation=60)

    err = np.max(
        np.abs(
            (intensities_db.Model_Sensitivity - intensities_db.Model_SingleFreq_Centre)
        )
    )
    plt.title(f"Max error due to TFM interpolation: {err:.3f} dB")

    # %% Model error boxplot

    errors = pd.DataFrame(
        {
            model_key[6:]: intensities_db[model_key] - intensities_db.Experimental
            for model_key in [
                "Model_SingleFreq_Centre",
                "Model_SingleFreq_Max",
                "Model_MultiFreq_Max",
            ]
        }
    )

    errors.plot(kind="box")
    plt.title("db(Experimental / Model)")
    if save:
        plt.savefig(str(result_dir / "model_errors_box"))

    desc = errors.describe()
    if save:
        desc.to_csv(result_dir / "model_errors.csv")
    print("===== Error vs Experimental =====")
    print(desc)

    # %% Model error barplot

    errors.plot.bar()
    plt.xticks(range(len(viewnames)), viewnames, rotation=60)
    plt.title("Errors between model and experimental results (dB)")
    if save:
        plt.savefig(str(result_dir / "model_errors"))

    # %% Is sensitivity model appropriate?

    plt.figure()
    x = intensities_db["Model_SingleFreq_Centre"]
    y = intensities_db["Model_MultiFreq_Max"]

    plt.plot(x, y, ".")
    for (viewname, x_val, y_val) in zip(intensities_db.index, x, y):
        plt.text(x_val, y_val, viewname, horizontalalignment="center")
    amin = np.min([x, y])
    amax = np.max([x, y])
    plt.plot([amin, amax], [amin, amax], "-k", linewidth=1, zorder=0)
    plt.plot([amin, amax], [amin - 3, amax - 3], "--C3", zorder=0)
    plt.plot([amin, amax], [amin + 3, amax + 3], "--C3", zorder=0)
    plt.xlabel("Sensitivity model (dB)")
    plt.ylabel("Multi-frequency model (dB)")
    plt.grid()
    if save:
        plt.savefig(str(result_dir / f"validity_sensitivity"))

    # %% Is SingleFreq_Max appropriate?

    plt.figure()
    x = intensities_db["Model_SingleFreq_Max"]
    y = intensities_db["Model_MultiFreq_Max"]

    plt.plot(x, y, ".")
    for (viewname, x_val, y_val) in zip(intensities_db.index, x, y):
        plt.text(x_val, y_val, viewname, horizontalalignment="center")
    amin = np.min([x, y])
    amax = np.max([x, y])
    plt.plot([amin, amax], [amin, amax], "-k", linewidth=1, zorder=0)
    plt.plot([amin, amax], [amin - 3, amax - 3], "--C3", zorder=0)
    plt.plot([amin, amax], [amin + 3, amax + 3], "--C3", zorder=0)
    plt.xlabel("SingleFreq Max model (dB)")
    plt.ylabel("Multi-frequency model (dB)")
    plt.grid()
    if save:
        plt.savefig(str(result_dir / f"validity_singlefreq_max"))

    # %%
    print("===== Error Senstivity vs MultiFreq_Max =====")
    error_sens = (
        intensities_db["Model_SingleFreq_Centre"]
        - intensities_db["Model_MultiFreq_Max"]
    )
    print(error_sens.describe())
    # %% XY plots scaled by wall

    model_keys = [
        "Model_SingleFreq_Centre",
        "Model_SingleFreq_Max",
        "Model_MultiFreq_Max",
    ]

    def plot_vs_experiment(model_key):
        x = intensities_db[model_key]
        y = intensities_db["Experimental"]

        is_not_nan = ~np.isnan(y)
        x = x[is_not_nan]
        y = y[is_not_nan]

        fig = plt.figure()
        plt.plot(x, y, ".")
        for (viewname, x_val, y_val) in zip(x.index, x, y):
            plt.text(x_val, y_val, viewname, horizontalalignment="center")

        ax = plt.gca()
        amin = np.min([x, y])
        amax = np.max([x, y])
        plt.plot([amin, amax], [amin, amax], "-k", linewidth=1, zorder=0)
        plt.plot([amin, amax], [amin - 3, amax - 3], "--C3", zorder=0)
        plt.plot([amin, amax], [amin + 3, amax + 3], "--C3", zorder=0)
        plt.xlabel(model_key.replace("_", " ") + " (dB)")
        plt.ylabel("Experimental (dB)")
        plt.text(
            0.95,
            0.1,
            "Model overestimates",
            transform=ax.transAxes,
            horizontalalignment="right",
        )
        plt.text(0.1, 0.9, "Model underestimates", transform=ax.transAxes)
        plt.grid()
        if save:
            fig.savefig(str(result_dir / f"experiment_vs_{model_key}"))

    for model_key in model_keys:
        plot_vs_experiment(model_key)

    # %% XY plots scaled by brighest view

    model_keys = [
        "Model_SingleFreq_Centre",
        "Model_SingleFreq_Max",
        "Model_MultiFreq_Max",
    ]
    scale_with = intensities_db.Experimental.idxmax()
    print(f"View for scaling: {scale_with}")

    def plot_vs_experiment_rescaled(model_key):
        x = (
            intensities_db[model_key]
            - intensities_db[model_key][scale_with]
            + intensities_db["Experimental"][scale_with]
        )
        y = intensities_db["Experimental"]

        is_not_nan = ~np.isnan(y)
        x = x[is_not_nan]
        y = y[is_not_nan]

        fig = plt.figure()
        plt.plot(x, y, ".")
        for (viewname, x_val, y_val) in zip(intensities_db.index, x, y):
            plt.text(x_val, y_val, viewname, horizontalalignment="center")

        ax = plt.gca()
        amin = np.min([x, y])
        amax = np.max([x, y])
        plt.plot([amin, amax], [amin, amax], "-k", linewidth=1, zorder=0)
        plt.plot([amin, amax], [amin - 3, amax - 3], "--C3", zorder=0)
        plt.plot([amin, amax], [amin + 3, amax + 3], "--C3", zorder=0)
        plt.xlabel(model_key.replace("_", " ") + " (dB)")
        plt.ylabel("Experimental (dB)")
        plt.text(
            0.95,
            0.1,
            "Model overestimates",
            transform=ax.transAxes,
            horizontalalignment="right",
        )
        plt.text(0.1, 0.9, "Model underestimates", transform=ax.transAxes)
        plt.grid()
        # if save:
        #     fig.savefig(str(result_dir / f'experiment_vs_{model_key}_rescaled'))

    for model_key in model_keys:
        plot_vs_experiment_rescaled(model_key)

    if noshow:
        plt.close("all")
    else:
        plt.show()


# %%
if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    dataset_name = args.dataset_name
    save = args.save

    if args.paper:
        with plt.style.context(["grayscale", "to_svg.mplstyle"]):
            compare_tfm_intensities(dataset_name, save, noshow=args.noshow)
    else:
        compare_tfm_intensities(dataset_name, save, noshow=args.noshow)
