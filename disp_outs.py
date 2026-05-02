import torch
import torchvision.transforms as TT
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

from model import VAE

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'

def show_samples(n=10):
  model = VAE().to(DEVICE)
  model.eval()
  
  model_state_dict = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=True)
  
  model.load_state_dict(model_state_dict)
  
  image_paths = [
      os.path.join('data/', f)
      for f in os.listdir('data/')
      if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]
    
  disp_paths = random.sample(image_paths, n)
  
  imgs = []
  transforms = TT.Compose([
    TT.Resize((140, 140)),
    TT.CenterCrop((128, 128)),
    TT.ToTensor(),
    TT.Normalize([0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225])
  ])
  
  for p in disp_paths:
    imgs.append(transforms(Image.open(p).convert("RGB")))
    
  imgs = torch.stack(imgs).to(DEVICE)

  means, logvars = [], []
  
  with torch.no_grad():
    means, logvars = model.encode(imgs)
  
  stds = torch.exp(logvars * 0.5)
  
  z_all = means + stds * torch.randn_like(means)
  
  with torch.no_grad():
    outs = model.decode(z_all)
    
    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    
    outs = outs * std_tensor + mean_tensor
    outs = torch.clamp(outs, 0, 1)
    
  for i in range(n):
    s = outs[i].squeeze().cpu().numpy().transpose(1, 2, 0)
    
    plt.imshow(s)
    plt.show()
    
if __name__ == '__main__':
  show_samples()