import torch
import torchvision.transforms as TT
from PIL import Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

from model import VAE
from config import (DEVICE, RESIZE, CROP_SIZE, NORM_MEAN, NORM_STD,
                    Z_DIM, SAVE_DIR, DISP_EPOCH, N_SAMPLES)

def show_samples(n=N_SAMPLES):
    model = VAE(z_dim=Z_DIM).to(DEVICE)
    model.eval()

    model_state_dict = torch.load(f'{SAVE_DIR}/vae_epoch_{DISP_EPOCH}.pth', map_location='cpu', weights_only=True)['model_state_dict']

    model.load_state_dict(model_state_dict)

    image_paths = [
        os.path.join('data/', f)
        for f in os.listdir('data/')
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]

    disp_paths = random.sample(image_paths, n)

    imgs = []
    transforms = TT.Compose([
        TT.Resize(RESIZE),
        TT.CenterCrop(CROP_SIZE),
        TT.ToTensor(),
        TT.Normalize(NORM_MEAN, NORM_STD)
    ])

    for p in disp_paths:
        imgs.append(transforms(Image.open(p).convert("RGB")))

    imgs = torch.stack(imgs).to(DEVICE)

    means, logvars = [], []

    with torch.no_grad():
        means, logvars = model.encode(imgs)

    stds = torch.exp(logvars * 0.5)

    z_all = means + stds * torch.randn_like(means)

    mean_tensor = torch.tensor(NORM_MEAN).view(1, 3, 1, 1).to(DEVICE)
    std_tensor = torch.tensor(NORM_STD).view(1, 3, 1, 1).to(DEVICE)

    with torch.no_grad():
        outs = model.decode(z_all)
        outs = outs * std_tensor + mean_tensor
        outs = torch.clamp(outs, 0, 1)

    for i in range(n):
        s = outs[i].squeeze().cpu().numpy().transpose(1, 2, 0)

        plt.imshow(s)
        plt.show()

if __name__ == '__main__':
    show_samples()