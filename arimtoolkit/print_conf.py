# -*- coding: utf-8 -*-
import arim
import yaml
from . import common


def print_conf(dataset_name):
    conf = arim.io.load_conf(dataset_name)
    print(yaml.dump(dict(conf), default_flow_style=False))


if __name__ == "__main__":
    args = common.argparser(__doc__).parse_args()
    print_conf(args.dataset_name)
