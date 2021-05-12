import torch

from torch.optim import Adam, SGD
from torch.nn import MSELoss, CrossEntropyLoss
from constants import *
from datasets_manager import ImageTransforms


# Training steps definitions
def default_train_step(batch_data, train_vars, log_values):
    # initialize descriptor for batch loss
    log_values['postfix_descriptor'] = {}

    inputs, labels = batch_data
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    #
    # # zero the parameter gradients
    train_vars[OPTIMIZER].zero_grad()

    # forward + backward + optimize
    outputs = train_vars[MODEL].net(inputs)
    loss = train_vars[CRITERION](outputs, labels)
    loss.backward()
    train_vars[OPTIMIZER].step()

    # update loss and accuracy logs
    log_values['correct'] += (torch.argmax(outputs, dim=1) == labels).sum().item()
    log_values['running_loss'] += loss.item()
    log_values['postfix_descriptor']['loss'] = loss.item()


def transnet_train_step(batch_data, train_vars, log_values):
    # initialize descriptor for batch loss
    log_values['postfix_descriptor'] = {}

    inputs, labels = batch_data
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    # zero the parameter gradients
    train_vars[OPTIMIZER].zero_grad()

    # Calculate loss for rot0
    outputs = train_vars[MODEL].net(inputs)
    loss_rot0 = train_vars[CRITERION](outputs, labels)

    # Calculate loss for rot90
    rot90_inputs = ImageTransforms.rot90_batch(inputs)
    outputs_90 = train_vars[MODEL].net(rot90_inputs, rot_type=1)
    loss_rot90 = train_vars[CRITERION](outputs_90, labels)

    # Calculate loss for rot180
    rot180_inputs = ImageTransforms.rot180_batch(inputs)
    outputs_180 = train_vars[MODEL].net(rot180_inputs, rot_type=2)
    loss_rot180 = train_vars[CRITERION](outputs_180, labels)

    # Calculate loss for rot270
    rot270_inputs = ImageTransforms.rot270_batch(inputs)
    outputs_270 = train_vars[MODEL].net(rot270_inputs, rot_type=3)
    loss_rot270 = train_vars[CRITERION](outputs_270, labels)

    total_loss = loss_rot0 + loss_rot90 + loss_rot180 + loss_rot270
    total_loss.backward()
    train_vars[OPTIMIZER].step()

    # update loss and accuracy logs
    log_values['correct'] += (torch.argmax(outputs, dim=1) == labels).sum().item()
    log_values['running_loss'] += loss_rot0.item()
    log_values['postfix_descriptor']['loss'] = loss_rot0.item()


custom_training_steps = {
    'transnet': transnet_train_step,
    'default': default_train_step
}

custom_optimizers = {
    'adam': Adam,
    'sgd': SGD
}

custom_losses = {
    'mse': MSELoss,
    'cross_entropy': CrossEntropyLoss
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
