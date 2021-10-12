import torch
import datasets_manager

from constants import *
from tqdm import tqdm
from dynamic_object import DynamicObject
from model import Model


# static class which handles the training of a model
class TestingHandler:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def init_test_variables(config):
        test_vars = DynamicObject()

        # should set model class before ensemble
        test_vars[MODEL_CLASS] = Model.model_class_map[config[MODEL_CLASS]]
        test_vars[LOADER] = datasets_manager.get_loader(config[DATASET_NAME], config[BATCH_SIZE], config[NUM_LOADER_WORKERS], train=False)
        if config[ENSEMBLE]:
            test_vars[ENSEMBLE] = Model.load_multiple_models(config[SAVED_PATH], test_vars)
        else:
            test_vars[MODEL] = Model(config[MODEL_CLASS], test_vars[SAVED_PATH])
        return test_vars

    @staticmethod
    def test(config):
        test_vars = TestingHandler.init_test_variables(config)

        if config[TEST_ENSEMBLE]:
            TestingHandler.test_ensemble(test_vars)
        else:
            TestingHandler.test_single_model(test_vars)

    @staticmethod
    def test_ensemble(test_vars):
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(test_vars[LOADER], unit=" image") as tbatch:
                for data in tbatch:
                    images, labels = data
                    images, labels = images.to(TestingHandler.device), labels.to(TestingHandler.device)
                    nets_outputs = []
                    for net in test_vars[ENSEMBLE]:
                        nets_outputs.append(net(images))
                    outputs = sum(nets_outputs) / len(test_vars[ENSEMBLE])
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.10f %%' % (
                100 * correct / total))
        return 100 * correct / total

    @staticmethod
    def test_single_model(test_vars):
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(test_vars[LOADER], unit=" image") as tbatch:
                for data in tbatch:
                    images, labels = data
                    images, labels = images.to(TestingHandler.device), labels.to(TestingHandler.device)
                    outputs = test_vars[MODEL].net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.10f %%' % (
                100 * correct / total))
        return 100 * correct / total
