import argparse
import logging
import ssl
import neptune.new as neptune

from training_handler import TrainingHandler as TrHandler
from testing_handler import TestingHandler as TsHandler
from dynamic_object import DynamicObject as Meta
from constants import *

ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_path', type=str, required=True)
    # parser.add_argument('--test_config_path', type=str, required=True)
    parser.add_argument('--slurm_task_id', type=str, default='default_id', required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    train_config = Meta(args.train_config_path)
    # test_config = Meta(args.test_config_path)

    run = neptune.init(
        name=str(train_config[TRAIN_STEP]) + '_' + train_config[MODEL_CLASS] + '_' + str(args.slurm_task_id),
        project="eran.zecharia/thesis",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTY0MThmNy04MTdiLTRjMmYtOWExZi0zNjlhMmZkZjM1MGYifQ==",
    )

    train_config['neptune'] = run

    # Set Slurm task ID for
    train_config[SLURM_TASK_ID] = args.slurm_task_id

    # Train model
    TrHandler.train(train_config)

    # Test model
    # TsHandler.test(test_config)


if __name__ == '__main__':
    main()

