import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleDownConv(nn.Module):
  def __init__(self, c_in, c_out):
    super().__init__()
    
    self.double_conv = nn.Sequential(
      nn.Conv2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, c_out),
      nn.GELU(),
      
      nn.Conv2d(
        in_channels=c_out,
        out_channels=c_out,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, c_out),
      nn.GELU(),
      
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
  def forward(self, X):
    return self.double_conv(X)


class DoubleUpConv(nn.Module):
  def __init__(self, c_in, c_out):
    super().__init__()
    
    self.double_conv = nn.Sequential(
      nn.Conv2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, c_out),
      nn.GELU(),
      
      nn.Conv2d(
        in_channels=c_out,
        out_channels=c_out,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, c_out),
      nn.GELU(),
      
      nn.ConvTranspose2d(
        in_channels=c_out,
        out_channels=c_out,
        kernel_size=2,
        stride=2,
        padding=0,
        bias=True
      ),
      nn.GroupNorm(8, c_out)
    )
    
    
  def forward(self, X):
    return self.double_conv(X)



class VAE(nn.Module):
  def __init__(self, num_layers=4, hid_c_dim=256, z_dim=128):
    super().__init__()
    
    self.encoder = nn.Sequential(
      DoubleDownConv(3, 32),
      DoubleDownConv(32, 64),
      DoubleDownConv(64, 128),
      DoubleDownConv(128, 256),
      DoubleDownConv(256, 512, last=True)
    )
    
    self.hid_2_mean = nn.Sequential(
      nn.Linear(512 * 4**2, z_dim*4),
      nn.GELU(),
      nn.LayerNorm(z_dim*4),
      
      nn.Linear(z_dim*4, z_dim)
    )
    
    self.hid_2_std = nn.Sequential(
      nn.Linear(512 * 4**2, z_dim*4),
      nn.GELU(),
      nn.LayerNorm(z_dim*4),
      
      nn.Linear(z_dim*4, z_dim)
    )
    
    self.z_2_hid = nn.Sequential(
      nn.Linear(z_dim, 128//2**5)
    )
    
    self.decoder = nn.Sequential(
      DoubleUpConv(512, 256),
      DoubleUpConv(256, 128),
      DoubleUpConv(128, 64),
      DoubleUpConv(64, 32),
      DoubleUpConv(32, 3, last=True)
    )
    
  
  def encode(self, img):
    B = img.shape[0]
    
    h = self.decoder(img).view(B, -1)
    mean, std = self.hid_2_mean(h), self.hid_2_std(h)
    
    return mean, std    
  
  def decode(self, z):
    h = self.z_2_hid(z)
    out = self.decoder(z)
    
    return F.tanh(out)
  
  def forward(self, X):
    mean, std = self.encode(X)
    
    z_reparam = mean + std * torch.randn_like(mean)