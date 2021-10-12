import os
import time

import torchvision.models

from network_models import *
from logging import getLogger
from constants import *


class Model:
    # Static variable
    model_class_map = {
        'TransResNet18': TransResNet18,
        'RotResNet18': RotResNet18,
        'ResNet18': torchvision.models.resnet18
    }

    def __init__(self, model_class, num_classes, path=None):
        # Create model take weights from path if exists
        self.net = Model.model_class_map[model_class](num_classes)
        self.logger = getLogger('model-logger')
        self.load_weights(path)

    def load_weights(self, path):
        if not path:
            self.logger.warning('Path was not given, initializing weights randomly')
        elif os.path.exists(path):
            self.logger.error('No such file or directory ' + path)
        else:
            self.logger.info('Model was loaded from path successfully')
            self.net.load_state_dict(path)

    def save_model(self, model_dir, model_name):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        time.sleep(2)
        torch.save(self.net.state_dict(), os.path.join(model_dir, model_name))

    @staticmethod
    def load_multiple_models(models_dir, test_vars):
        models = []
        for filename in os.listdir(models_dir):
            if filename.endswith(".pth"):
                model = Model(test_vars[MODEL_CLASS], os.path.join(models_dir, filename))
                models.append(model)
        return models

