import argparse
import utils
from sampler import Sampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sample from a trained OctFusion model")
    parser.add_argument("--conf", type=str, help="Path to the configuration file to load.")
    args = parser.parse_args()

    # read configuration
    conf = utils.conf.read_conf(args.conf)

    # initialize sampler
    sampler = Sampler(**conf)
    sampler.sample()

