import numpy as np
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


def simclr_train_step(batch_data, train_vars, log_values, tau=0.1):
    # initialize descriptor for batch loss
    log_values['postfix_descriptor'] = {}

    inputs, labels = batch_data
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    # Zero the parameters gradients for regular loss
    train_vars[OPTIMIZER].zero_grad()

    outputs = train_vars[MODEL].net(inputs)
    loss = train_vars[CRITERION](outputs, labels)
    loss.backward()
    train_vars[OPTIMIZER].step()

    # Zero the parameters gradients for simclr loss
    train_vars[OPTIMIZER].zero_grad()

    first_representation = train_vars[MODEL]()


def rotnet_train_step(batch_data, train_vars, log_values):
    # initialize descriptor for batch loss
    inputs, labels = batch_data
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    # calculate rotation
    inputs_rotation, labels_rotation = ImageTransforms.random_rotate_batch(inputs)

    # Zero the parameter gradients for rotation loss
    train_vars[OPTIMIZER].zero_grad()

    outputs_rotation = train_vars[MODEL].net(inputs_rotation, rot_task=True)
    loss_rot = train_vars[CRITERION](outputs_rotation, labels_rotation)
    loss_rot.backward()
    train_vars[OPTIMIZER].step()

    # Zero the parameter gradients for regular loss
    train_vars[OPTIMIZER].zero_grad()

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


def simclr_loss(batch_proj1, batch_proj2):
    proj1_norm = torch.sqrt((batch_proj1*batch_proj1).sum(axis=1))
    proj2_norm = torch.sqrt((batch_proj2*batch_proj2).sum(axis=1))
    s = (batch_proj1*batch_proj2).sum(axis=1) / (proj1_norm * proj2_norm)
    return s


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