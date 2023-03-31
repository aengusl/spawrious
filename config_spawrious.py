from ml_collections import ConfigDict
import argparse


def get_config():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='path to data directory')
    parser.add_argument('--dataset_name', type=str, help='dataset challenge to download. Options are: o2o_easy, o2o_medium, o2o_hard, m2m (which accounts for all many to many challenges), entire_dataset')

    config = ConfigDict(vars(parser.parse_args()))
    return config