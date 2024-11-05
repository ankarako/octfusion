import argparse

import utils
import mv_fitter

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Multi-view RGB-D data model fitting")
    parser.add_argument("--conf", type=str, help="Path to the configuration file to load.")
    args = parser.parse_args()

    # read configuration file
    conf = utils.conf.read_conf(args.conf)

    # initialize fitting algorithm
    fitter = mv_fitter.MVFitter(**conf)
    fitter.fit()
