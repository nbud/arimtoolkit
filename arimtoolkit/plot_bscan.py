import arim
import arim.plot as aplt
import matplotlib.pyplot as plt

from . import common


def plot_bscan(dataset_name, save=False, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    frame = common.load_frame(
        conf, apply_filter=True, expand=False, warn_if_fallback_vel=True
    )
    result_dir = conf["result_dir"]
    # plot bscan
    ax, imag = aplt.plot_bscan_pulse_echo(
        frame, clim=[-40, 0], filename=str(result_dir / "bscan"), savefig=save
    )
    if noshow:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    plot_bscan(args.dataset_name, args.save, noshow=args.noshow)
