import logging
import training_variables as train_vars_module
import datasets_manager
import os
import neptune.new as neptune

from constants import *
from dynamic_object import DynamicObject
from model import Model
from tqdm import tqdm


# static class which handles the training of a model
class TrainingHandler:
    logger = logging.getLogger('TrainingHandler')

    @staticmethod
    def init_log_object():
        log_values = DynamicObject()
        log_values['correct'] = 0
        log_values['running_loss'] = 0

        return log_values

    @staticmethod
    def log_accuracy_and_loss(log_values, train_vars, config):
        run = config['neptune']
        params = {"learning_rate": config[LEARNING_RATE], "optimizer": config[OPTIMIZER]}
        run["parameters"] = params

        denominator = len(train_vars[LOADER]) * config[BATCH_SIZE]
        epoch_loss = log_values['running_loss'] / denominator
        epoch_accuracy = 100 * log_values['correct'] / denominator

        print('\rEpoch Accuracy %.2f percent' % epoch_accuracy)
        print('\rEpoch loss: %.5f ' % epoch_loss)

        run["train/loss"].log(epoch_loss)
        run["train/accuracy"].log(epoch_accuracy)

    @staticmethod
    def save_trained_model(epoch, config, train_vars):
        if (epoch + 1) in config[EPOCHS_TO_SAVE]:
            full_path = os.path.join(config[MODEL_SAVE_DIR], config[ENSEMBLE_NAME], str(epoch + 1))
            train_vars[MODEL].save_model(full_path, 'model_' + str(config[SLURM_TASK_ID]))

    @staticmethod
    def train(config: DynamicObject):
        if not config[SHOULD_TRAIN_MODEL]:
            return

        train_vars = DynamicObject()

        train_vars[MODEL] = Model(config[MODEL_CLASS], config[NUM_CLASSES], config[MODEL_PATH])
        train_vars[OPTIMIZER] = train_vars_module.custom_optimizers[config[OPTIMIZER]](train_vars[MODEL].net.parameters(), lr=config[LEARNING_RATE])
        train_vars[CRITERION] = train_vars_module.custom_losses[config[CRITERION]]()
        train_vars[LOADER] = datasets_manager.get_loader(config[DATASET_NAME], config[BATCH_SIZE], config[NUM_LOADER_WORKERS])
        train_vars[TRAIN_STEP] = train_vars_module.custom_training_steps[config[TRAIN_STEP]]

        train_vars[MODEL].net.to(train_vars_module.device)

        for epoch in range(config[NUM_EPOCHS]):
            log_values = TrainingHandler.init_log_object()

            with tqdm(train_vars[LOADER], unit='batch') as data_loader:
                for batch_data in data_loader:
                    data_loader.set_description(f"Epoch {epoch + 1}")
                    train_vars[TRAIN_STEP](batch_data, train_vars, log_values)
                    data_loader.set_postfix(log_values['postfix_descriptor'])
            TrainingHandler.save_trained_model(epoch, config, train_vars)
            TrainingHandler.log_accuracy_and_loss(log_values, train_vars, config)
        TrainingHandler.logger.warning('Finished training!')
