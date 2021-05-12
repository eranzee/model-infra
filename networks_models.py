import torch.nn as nn
import torch.nn.functional as F
import torch


class VanillaNet(nn.Module):
    def __init__(self):
        super(VanillaNet, self).__init__()
        self.model_index = 0

        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc3_rot = nn.Linear(84, 4)

    def forward(self, x, rot_task=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Use -1 because we have to infer the batch size
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if rot_task:
            return self.fc3_rot(x)
        return self.fc3(x)


class LinearVanillaNet(nn.Module):
    def __init__(self):
        super(LinearVanillaNet, self).__init__()
        self.name = 'Linear Net'

        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        # Use -1 because we have to infer the batch size
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ExtraFullyConnectedOneVanillaNet(nn.Module):
    def __init__(self):
        super(ExtraFullyConnectedOneVanillaNet, self).__init__()
        self.name = 'Extra Fully Connected Net - 1 layer'

        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 42)
        self.fc4 = nn.Linear(42, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Use -1 because we have to infer the batch size
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class ExtraFullyConnectedTwoVanillaNet(nn.Module):
    def __init__(self):
        super(ExtraFullyConnectedTwoVanillaNet, self).__init__()
        self.name = 'Extra Fully Connected Net - 2 layers'

        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 42)
        self.fc4 = nn.Linear(42, 21)
        self.fc5 = nn.Linear(21, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Use -1 because we have to infer the batch size
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


class ExtraConvolutionDownSampleVanillaNet(nn.Module):
    def __init__(self):
        super(ExtraConvolutionDownSampleVanillaNet, self).__init__()
        self.name = 'Vanilla Net'

        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.fc1 = nn.Linear(64 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        # Use -1 because we have to infer the batch size
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNetStyleOneBlockVanillaNet(nn.Module):
    def __init__(self):
        super(ResNetStyleOneBlockVanillaNet, self).__init__()
        self.name = 'resnet-style-one-block'
        self.model_index = 0

        self.zero_pad = nn.ZeroPad2d(2)
        self.identity_conv1 = nn.Conv2d(3, 64, (1, 1))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))

        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        output = F.relu(self.conv1(self.zero_pad(x)))
        output = self.conv2(self.zero_pad(output))
        x1 = F.relu(self.pool(output) + self.pool(self.identity_conv1(x)))

        # Use -1 because we have to infer the batch size
        output = x1.view(-1, 64 * 16 * 16)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class ResNetStyleTwoBlocksVanillaNet(nn.Module):
    def __init__(self):
        super(ResNetStyleTwoBlocksVanillaNet, self).__init__()
        self.name = 'two-blocks-resnet'
        self.zero_pad = nn.ZeroPad2d(2)

        self.identity_conv1 = nn.Conv2d(3, 64, (1, 1))
        self.identity_conv2 = nn.Conv2d(64, 128, (1, 1))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(5, 5))

        self.fc1 = nn.Linear(128 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.fc3_rot = nn.Linear(32, 4)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, rot_task=False):
        output = F.relu(self.conv1(self.zero_pad(x)))
        output = self.conv2(self.zero_pad(output))
        x1 = F.relu(self.pool(output) + self.pool(self.identity_conv1(x)))

        output = F.relu(self.conv3(self.zero_pad(x1)))
        output = self.conv4(self.zero_pad(output))
        x2 = F.relu(self.pool(output) + self.pool(self.identity_conv2(x1)))

        # Use -1 because we have to infer the batch size
        output = x2.view(-1, 128 * 8 * 8)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        if rot_task:
            return self.fc3_rot(output)
        return self.fc3(output)


class ResNetStyleTwoBlocksVanillaNetTransnet(nn.Module):
    def __init__(self):
        super(ResNetStyleTwoBlocksVanillaNetTransnet, self).__init__()
        self.name = 'two-blocks-resnet'
        self.zero_pad = nn.ZeroPad2d(2)

        self.identity_conv1 = nn.Conv2d(3, 64, (1, 1))
        self.identity_conv2 = nn.Conv2d(64, 128, (1, 1))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(5, 5))

        self.fc1 = nn.Linear(128 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.fc3_rot1 = nn.Linear(32, 10)
        self.fc3_rot2 = nn.Linear(32, 10)
        self.fc3_rot3 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, rot_type=0):
        output = F.relu(self.conv1(self.zero_pad(x)))
        output = self.conv2(self.zero_pad(output))
        x1 = F.relu(self.pool(output) + self.pool(self.identity_conv1(x)))

        output = F.relu(self.conv3(self.zero_pad(x1)))
        output = self.conv4(self.zero_pad(output))
        x2 = F.relu(self.pool(output) + self.pool(self.identity_conv2(x1)))

        # Use -1 because we have to infer the batch size
        output = x2.view(-1, 128 * 8 * 8)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))

        if rot_type == 1:
            return self.fc3_rot1(output)
        if rot_type == 2:
            return self.fc3_rot2(output)
        if rot_type == 3:
            return self.fc3_rot3(output)
        return self.fc3(output)


class ResNetStyleThreeBlocksVanillaNet(nn.Module):
    def __init__(self):
        super(ResNetStyleThreeBlocksVanillaNet, self).__init__()
        self.name = 'Resnet Style Three Blocks'
        self.zero_pad = nn.ZeroPad2d(2)
        self.pool = nn.MaxPool2d(2, 2)

        self.identity_conv1 = nn.Conv2d(3, 64, (1, 1))
        self.identity_conv2 = nn.Conv2d(64, 128, (1, 1))
        self.identity_conv3 = nn.Conv2d(128, 256, (1, 1))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(5, 5))

        self.conv5 = nn.Conv2d(128, 128, kernel_size=(5, 5))
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(5, 5))

        self.fc1 = nn.Linear(256 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.fc3_rot = nn.Linear(32, 4)

    def forward(self, x, rot_task=False):
        output = F.relu(self.conv1(self.zero_pad(x)))
        output = self.conv2(self.zero_pad(output))
        x1 = F.relu(self.pool(output) + self.pool(self.identity_conv1(x)))

        output = F.relu(self.conv3(self.zero_pad(x1)))
        output = self.conv4(self.zero_pad(output))
        x2 = F.relu(self.pool(output) + self.pool(self.identity_conv2(x1)))

        output = F.relu(self.conv5(self.zero_pad(x2)))
        output = F.relu(self.conv6(self.zero_pad(output)))
        x3 = F.relu(self.pool(output) + self.pool(self.identity_conv3(x2)))

        # Use -1 because we have to infer the batch size
        output = x3.view(-1, 256 * 4 * 4)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))

        if rot_task:
            return self.fc3_rot(output)
        return self.fc3(output)


class ResNetStyleFourBlocksVanillaNet(nn.Module):
    def __init__(self):
        super(ResNetStyleFourBlocksVanillaNet, self).__init__()
        self.name = 'Resnet Style Four Blocks'
        self.zero_pad = nn.ZeroPad2d(2)

        self.identity_conv1 = nn.Conv2d(3, 64, (1, 1))
        self.identity_conv2 = nn.Conv2d(64, 128, (1, 1))
        self.identity_conv3 = nn.Conv2d(128, 256, (1, 1))
        self.identity_conv4 = nn.Conv2d(256, 512, (1, 1))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(5, 5))

        self.conv5 = nn.Conv2d(128, 128, kernel_size=(5, 5))
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(5, 5))

        self.conv7 = nn.Conv2d(256, 256, kernel_size=(5, 5))
        self.conv8 = nn.Conv2d(256, 512, kernel_size=(5, 5))

        self.fc1 = nn.Linear(512 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 32 features
        output = F.relu(self.conv1(self.zero_pad(x)))
        output = self.conv2(self.zero_pad(output))
        x1 = F.relu(self.pool(output) + self.pool(self.identity_conv1(x)))

        # 64 features
        output = F.relu(self.conv3(self.zero_pad(x1)))
        output = self.conv4(self.zero_pad(output))
        x2 = F.relu(self.pool(output) + self.pool(self.identity_conv2(x1)))

        # 128 features
        output = F.relu(self.conv5(self.zero_pad(x2)))
        output = F.relu(self.conv6(self.zero_pad(output)))
        x3 = F.relu(self.pool(output) + self.pool(self.identity_conv3(x2)))

        # 256 features
        output = F.relu(self.conv7(self.zero_pad(x3)))
        output = F.relu(self.conv8(self.zero_pad(output)))
        x4 = F.relu(self.pool(output) + self.pool(self.identity_conv4(x3)))

        # 512 features
        # Use -1 because we have to infer the batch size
        output = x4.view(-1, 512 * 2 * 2)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class StVgg(nn.Module):
    def __init__(self):
        super(StVgg, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.flatten = torch.flatten
        self.fc0 = nn.Linear(2048, 10)
        self.fc1 = nn.Linear(2048, 10)
        self.fc2 = nn.Linear(2048, 10)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x, rotation_type=0):
        output = self.conv1_1(x)
        output = self.conv1_2(output)
        output = self.pool1(output)
        output = self.dropout1(output)
        output = self.bn1(output)

        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.pool2(output)
        output = self.dropout2(output)
        output = self.bn2(output)

        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.pool3(output)
        output = self.dropout3(output)
        output = self.bn3(output)

        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.pool4(output)
        output = self.dropout4(output)
        output = self.bn4(output)

        output = self.flatten(output, start_dim=1)

        if rotation_type == 0:
            output = self.fc0(output)
        elif rotation_type == 1:
            output = self.fc1(output)
        elif rotation_type == 2:
            output = self.fc2(output)
        elif rotation_type == 3:
            output = self.fc3(output)

        return output
