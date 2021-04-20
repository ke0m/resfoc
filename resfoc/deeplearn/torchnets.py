import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):

  def __init__(self):
    super(Unet,self).__init__()
    # Maxpool
    self.pool = nn.MaxPool2d(2,2)
    # Upsampling
    self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)

    # Convolutional blocks
    self.conv1a = nn.Conv2d(     1, 16,3,padding=(1,1))
    self.conv1b = nn.Conv2d(    16, 16,3,padding=(1,1))

    self.conv2a = nn.Conv2d(    16, 32,3,padding=(1,1))
    self.conv2b = nn.Conv2d(    32, 32,3,padding=(1,1))

    self.conv3a = nn.Conv2d(    32, 64,3,padding=(1,1))
    self.conv3b = nn.Conv2d(    64, 64,3,padding=(1,1))

    self.conv4a = nn.Conv2d(    64,512,3,padding=(1,1))
    self.conv4b = nn.Conv2d(   512,512,3,padding=(1,1))

    # 64 channels comes from conv3 output
    self.conv5a = nn.Conv2d(64+512, 64,3,padding=(1,1))
    self.conv5b = nn.Conv2d(    64, 64,3,padding=(1,1))

    # 32 channels comes from conv2 output
    self.conv6a = nn.Conv2d(32+ 64, 32,3,padding=(1,1))
    self.conv6b = nn.Conv2d(    32, 32,3,padding=(1,1))

    # 16 channels comes from conv1 output
    self.conv7a = nn.Conv2d(16+ 32, 16,3,padding=(1,1))
    self.conv7b = nn.Conv2d(    16, 16,3,padding=(1,1))

    self.conv8  = nn.Conv2d(    16,  1,3,padding=(1,1))

  def forward(self,x):
    """ Forward pass of the network """
    # Contracting portion
    x1  = F.relu(self.conv1b(F.relu(self.conv1a(x  ))))
    x1d = self.pool(x1)
    x2  = F.relu(self.conv2b(F.relu(self.conv2a(x1d))))
    x2d = self.pool(x2)
    x3  = F.relu(self.conv3b(F.relu(self.conv3a(x2d))))
    x3d = self.pool(x3)
    x4  = F.relu(self.conv4b(F.relu(self.conv4a(x3d))))
    # Expanding portion
    x5  = torch.cat([x3,self.up(x4)],dim=1)
    x6  = F.relu(self.conv5b(F.relu(self.conv5a(x5 ))))
    x7  = torch.cat([x2,self.up(x6)],dim=1)
    x8  = F.relu(self.conv6b(F.relu(self.conv6a(x7 ))))
    x9  = torch.cat([x1,self.up(x8)],dim=1)
    x10 = F.relu(self.conv7b(F.relu(self.conv7a(x9 ))))
    x11 = self.conv8(x10)

    return x11

class Vgg3_3d(nn.Module):

  def __init__(self):
    super(Vgg3_3d,self).__init__()
    # Maxpool
    self.pool = nn.MaxPool3d(2)

    # Convolutional blocks
    self.conv1 = nn.Conv3d(  1, 32,3,padding=(1,1,1))
    self.conv2 = nn.Conv3d( 32, 64,3,padding=(1,1,1))
    self.conv3 = nn.Conv3d( 64,128,3,padding=(1,1,1))

    # Linear
    self.fc1 = nn.Linear(128 * 8 * 8 * 8,128)
    self.fc2 = nn.Linear(128,1)

  def forward(self,x):
    """ Forward pass of the network """
    # Convolutional layers
    x1  = F.relu(self.conv1(x  ))
    x1d = self.pool(x1)
    x2  = F.relu(self.conv2(x1d))
    x2d = self.pool(x2)
    x3  = F.relu(self.conv3(x2d))
    x3d = self.pool(x3)
    # Flatten
    x3f = x3d.view(-1,128 * 8 * 8 * 8)
    x4  = F.relu(self.fc1(x3f))
    x5 = self.fc2(x4)

    return x5

class Vgg3a_3d(nn.Module):

  def __init__(self):
    super(Vgg3a_3d,self).__init__()
    # Maxpool
    self.pool1 = nn.MaxPool3d((1,2,2))
    self.pool2 = nn.MaxPool3d((2,2,2))

    # Convolutional blocks
    self.conv1 = nn.Conv3d(  1, 32,3,padding=(1,1,1))
    self.conv2 = nn.Conv3d( 32, 64,3,padding=(1,1,1))
    self.conv3 = nn.Conv3d( 64,128,3,padding=(1,1,1))

    # Linear
    self.fc1 = nn.Linear(128 * 8 * 8 * 8,128)
    self.fc2 = nn.Linear(128,1)

  def forward(self,x):
    """ Forward pass of the network """
    # Convolutional layers
    x1  = F.relu(self.conv1(x  ))
    x1d = self.pool1(x1)
    x2  = F.relu(self.conv2(x1d))
    x2d = self.pool2(x2)
    x3  = F.relu(self.conv3(x2d))
    x3d = self.pool2(x3)
    # Flatten
    x3f = x3d.view(-1,128 * 8 * 8 * 8)
    x4  = F.relu(self.fc1(x3f))
    x5  = self.fc2(x4)

    return x5

def save_torchnet(net,path) -> None:
  """
  Saves a torch net. Handles the case when the network
  is wrapped with data parallel

  Parameters:
    net  - the torch network
    path - the path to the network
  """
  # Get the parameter dictionary
  try:
    state_dict = net.module.state_dict()
  except AttributeError:
    state_dict = net.state_dict()

  torch.save(state_dict, path)

