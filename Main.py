import argparse
import logging

from TrainingHandler import TrainingHandler as TrHandler
from TestingHandler import TestingHandler as TsHandler
from DynamicObject import DynamicObject as Meta
from Constants import *


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

    # Set Slurm task ID for
    train_config[SLURM_TASK_ID] = args.slurm_task_id

    # Train model
    TrHandler.train(train_config)

    # Test model
    # TsHandler.test(test_config)


if __name__ == '__main__':
    main()


