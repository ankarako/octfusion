import argparse
import utils.conf as conf

from trainer.registry import get_trainer
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train OctFusion via trainers")
    parser.add_argument("--conf", type=str, help="The path to the configuration file to load")
    args = parser.parse_args()

    # read configuration file
    config = conf.read_conf(args.conf)

    # instantiate trainer
    trainer = get_trainer(config.key, **config.kwargs)
    try:
        trainer.train()
    except KeyboardInterrupt:
        utils.log.WARN("Training interrupted.")
        utils.log.INFO("Saving checkpoint...")
        fpath = trainer.save_chkp()
        utils.log.INFO(f"Checkpoint saved at: {fpath}")