import logging

import arim
import arim.ray
import arim.im
import arim.plot as aplt
import matplotlib.pyplot as plt

from . import common

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_interfaces(dataset_name, save, noshow=False):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    logger.info(f"dataset_name: {dataset_name}")

    probe = common.load_probe(conf)
    probe_p = probe.to_oriented_points()

    examination_object = arim.io.block_in_immersion_from_conf(conf)

    frontwall = examination_object.frontwall
    backwall = examination_object.backwall

    all_interfaces = [probe_p, frontwall, backwall]
    markers = [".", "-", "-"]

    try:
        all_interfaces.append(common.defect_oriented_point(conf))
        markers.append("d")
    except common.NoDefect:
        pass

    aplt.plot_interfaces(
        all_interfaces,
        show_orientations=False,
        show_last=True,
        markers=markers,
        filename=str(result_dir / "interfaces"),
        savefig=save,
    )
    if noshow:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()

    if args.dataset_name == "*":
        for dataset_name in common.all_datasets():
            plot_interfaces(dataset_name, args.save, noshow=args.noshow)
            # plt.close('all')
    else:
        plot_interfaces(args.dataset_name, args.save, noshow=args.noshow)
