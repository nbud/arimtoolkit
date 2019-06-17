"""
Measure model scaling coefficient from wall echoes in timestraces
with ordinary least square.

LEGACY code. Use measure_model_scaling_tfm

Scaling methods:
    
- frontwall_all
- frontwall_pulse_echo
- backwall_LL_all
- backwall_LL_pulse_echo
- backwall_LT_all
- backwall_LT_pulse_echo
- backwall_TT_all
- backwall_TT_pulse_echo

Known issues:
- The measurement using the backwall LT and TL is unreliable because
these signals strongly interfere with each other; they cannot be considered
as independent.
- The model scaling using the frontwall is unreliable if the frontwall is
saturated.


Output
------
model_scalings.csv
    Columns: model_scaling, rsquared, ref_wall_key, method
wall_amps_*.png
conf.d/22_model_scaling.yaml

"""
import logging

import arim
import arim.ray
import arim.im
import arim.plot as aplt
import arim.models.block_in_immersion as bim
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
import yaml

from . import common

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

methods = [
    "frontwall_all",
    "frontwall_pulse_echo",
    "backwall_LL_all",
    "backwall_LL_pulse_echo",
    "backwall_LT_all",
    "backwall_LT_pulse_echo",
    "backwall_TT_all",
    "backwall_TT_pulse_echo",
]


def measure_model_scaling(
    dataset_name, save, method="frontwall_pulse_echo", noshow=False
):
    conf = arim.io.load_conf(dataset_name)
    aplt.conf["savefig"] = save
    result_dir = conf["result_dir"]
    root_dir = conf["root_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    frame = common.load_frame(conf, apply_filter=True, expand=True)

    model_options = dict(
        frequency=common.get_centre_freq(conf, frame.probe),
        probe_element_width=frame.probe.dimensions.x[0],
    )

    def db(x):
        return 20 * np.log10(x)

    model_amps = {}
    model_times = {}
    exp_amps = {}
    exp_times = {}

    def to_2d(x):
        n = frame.probe.numelements
        if len(x) == (n * (n + 1)) // 2:
            # hmc
            y = np.full((n, n), np.nan, x.dtype)
            y[np.triu_indices(n)] = x
            y = np.fmax(y, y.T)  # fill lower part
            np.testing.assert_allclose(y, y.T)
        elif len(x) == n ** 2:
            # fmc
            y = x.reshape((n, n))
        else:
            raise ValueError
        return y

    # %% Frontwall model

    frontwall_path = bim.frontwall_path(
        frame.examination_object.couplant_material,
        frame.examination_object.block_material,
        *frame.probe.to_oriented_points(),
        *frame.examination_object.frontwall,
    )

    frontwall_ray_weights, frontwall_ray_weights_dict = bim.ray_weights_for_wall(
        frontwall_path, **model_options
    )

    model_times["frontwall"] = frontwall_path.rays.times[frame.tx, frame.rx]
    model_amps["frontwall"] = np.abs(frontwall_ray_weights[frame.tx, frame.rx])

    # %% Backwalls model

    backwall_paths = bim.backwall_paths(
        frame.examination_object.couplant_material,
        frame.examination_object.block_material,
        frame.probe.to_oriented_points(),
        frame.examination_object.frontwall,
        frame.examination_object.backwall,
    )

    arim.ray.ray_tracing_for_paths(backwall_paths.values())

    for wall_key_, backwall_path in backwall_paths.items():
        wall_key = f"backwall_{wall_key_}"
        backwall_ray_weights, backwall_ray_weights_dict = bim.ray_weights_for_wall(
            backwall_path, **model_options
        )

        model_times[wall_key] = backwall_path.rays.times[frame.tx, frame.rx]
        model_amps[wall_key] = np.abs(backwall_ray_weights[frame.tx, frame.rx])

    # %% Measure

    window_size = int(0.5e-6 / frame.time.step)
    window_size_t = window_size * frame.time.step

    for wall_key in model_times.keys():
        exp_times_tmp = np.zeros(frame.numscanlines)
        exp_amps_tmp = np.zeros(frame.numscanlines)

        for i in range(frame.numscanlines):
            time_idx = np.searchsorted(frame.time.samples, model_times[wall_key][i])
            window = slice(time_idx - window_size, time_idx + window_size)

            max_idx = np.argmax(np.abs(frame.scanlines[i, window]))

            exp_times_tmp[i] = frame.time.samples[window][max_idx]
            exp_amps_tmp[i] = np.abs(frame.scanlines[i, window])[max_idx]

        exp_amps[wall_key] = exp_amps_tmp
        exp_times[wall_key] = exp_times_tmp

    # %% Ascan

    tx_idx = 0
    rx_idx = frame.probe.numelements - 1

    idx = np.flatnonzero((frame.tx == tx_idx) & (frame.rx == rx_idx))[0]

    plt.figure()
    plt.plot(frame.time.samples, np.abs(frame.scanlines[idx]), "-")
    for i, (wall_key, t) in enumerate(model_times.items()):
        #    plt.axvline(t[idx], label=wall_key, color=f"C{i+1}", zorder=-1)
        plt.axvspan(
            t[idx] - window_size_t,
            t[idx] + window_size_t,
            label=wall_key,
            color=f"C{i+1}",
            alpha=0.25,
            zorder=-1,
        )
        plt.plot(exp_times[wall_key][idx], exp_amps[wall_key][idx], f"C{i+1}o")

    plt.gca().xaxis.set_major_formatter(aplt.micro_formatter)
    plt.title(f"Ascan tx={tx_idx}, rx={rx_idx}")

    _t_walls = np.array([t[idx] for t in model_times.values()])
    _tmin = max(frame.time.start, _t_walls.min() - 5e-6)
    _tmax = min(frame.time.end, _t_walls.max() + 5e-6)
    plt.xlim([_tmin, _tmax])
    plt.legend()
    plt.xlabel("time (µs)")
    plt.ylabel("amplitude (linear)")
    if save:
        plt.savefig(str(result_dir / f"ascan_{tx_idx}_{rx_idx}"))

    # %% Compare all models

    def make_model_scalings():
        tmp = []
        scan_to_use_dict = {
            "all": np.ones(frame.numscanlines, bool),
            "pulse_echo": frame.tx == frame.rx,
        }

        for ref_wall_key in ["frontwall", "backwall_LL", "backwall_LT", "backwall_TT"]:
            for to_use_key, scan_to_use in scan_to_use_dict.items():
                x = model_amps[ref_wall_key][scan_to_use]
                y = exp_amps[ref_wall_key][scan_to_use]
                ols_res = sm.OLS(y, x).fit()
                model_scaling = ols_res.params[0]

                tmp.append(
                    (
                        f"{ref_wall_key}_{to_use_key}",
                        model_scaling,
                        ols_res.rsquared,
                        ref_wall_key,
                    )
                )

        return pd.DataFrame(
            tmp, columns=("method", "model_scaling", "rsquared", "ref_wall_key")
        ).set_index("method")

    model_scaling_df = make_model_scalings()
    if save:
        # save csv
        model_scaling_df.to_csv(result_dir / "model_scalings.csv")

        # save yaml
        model_scaling = float(model_scaling_df.model_scaling[method])
        model_conf = dict(model=dict(scaling=model_scaling))
        with (root_dir / "conf.d/22_model_scaling.yaml").open("w") as f:
            f.write("# generated by measure_model_scaling\n")
            f.write(f"# method={method}\n")
            yaml.dump(model_conf, f, default_flow_style=False)

    # %% Select one method

    is_pulse_echo = frame.tx == frame.rx
    model_scaling = model_scaling_df.loc[method, "model_scaling"]

    # %% line plots for pulse-echo

    # Model predicts no amplitude for LT, TL and TT
    for wall_key in ["frontwall", "backwall_LL"]:
        plt.figure()

        plt.plot(
            db(model_amps[wall_key][is_pulse_echo] * model_scaling),
            "--",
            label=f" model",
        )
        plt.plot(db(exp_amps[wall_key][is_pulse_echo]), label=f"exp")

        plt.xlabel("element index")
        plt.ylabel("amplitude dB ")
        plt.title(f"{wall_key} amplitudes\npulse-echo - scaling method {method}")
        plt.legend()
        if save:
            plt.savefig(str(result_dir / f"wall_amps_{wall_key}_{method}_pulse_echo"))

    # %% line plots for a given tx

    idx = frame.probe.numelements - 1

    for wall_key in ["frontwall", "backwall_LL", "backwall_LT", "backwall_TT"]:
        plt.figure()

        plt.plot(
            db(to_2d(model_amps[wall_key] * model_scaling)[idx]), "--", label=f" model"
        )
        plt.plot(db(to_2d(exp_amps[wall_key])[idx]), label=f"exp")

        plt.xlabel("rx index")
        plt.ylabel("amplitude dB ")
        plt.title(f"{wall_key} amplitudes\ntx={idx} - scaling method {method}")
        plt.legend()
        if save:
            plt.savefig(str(result_dir / f"wall_amps_{wall_key}_{method}_{idx}"))

    if noshow:
        plt.close("all")
    else:
        plt.show()

    return model_scaling_df


if __name__ == "__main__":
    parser = common.argparser(__doc__)
    parser.add_argument(
        "--method",
        choices=methods,
        help="Scaling method",
        default="frontwall_pulse_echo",
    )
    args = parser.parse_args()
    model_scaling_df = measure_model_scaling(
        args.dataset_name, args.save, args.method, noshow=args.noshow
    )
    print(model_scaling_df)
