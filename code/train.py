import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
import os
from PIL import Image
from tqdm.auto import tqdm

from model import VAE
from config import (DEVICE, DATA_DIR, RESIZE, CROP_SIZE, NORM_MEAN, NORM_STD,
                    HFLIP_PROB, SATURATION_FACTOR, COLOR_JITTER,
                    BATCH_SIZE, LEARNING_RATE, TOTAL_EPOCHS, KL_BETA,
                    VGG_LOSS_WEIGHT, Z_DIM, SAVE_DIR, RESUME_EPOCH,
                    LOAD_WEIGHTS, SAVE_WEIGHTS, SEED)


class VGGLoss(nn.Module):
    def __init__(self, weight, device):
        super().__init__()
        self.weight = weight

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

        self.selected_layers = nn.Sequential(*list(vgg.children())[:23])

        for p in self.selected_layers.parameters():
            p.requires_grad_(False)

    def forward(self, x, y):
        x_features = self.selected_layers(x)
        y_features = self.selected_layers(y)

        return self.weight * F.mse_loss(x_features, y_features)


class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def main():
    torch.manual_seed(SEED)

    transforms = TT.Compose([
        TT.Resize(RESIZE),
        TT.CenterCrop(CROP_SIZE),
        TT.RandomHorizontalFlip(HFLIP_PROB),
        TT.Lambda(lambda x: TT.functional.adjust_saturation(x, saturation_factor=SATURATION_FACTOR)),
        TT.ColorJitter(*COLOR_JITTER),
        TT.ToTensor(),
        TT.Normalize(NORM_MEAN, NORM_STD)
    ])

    train_ds = FlowerDataset(DATA_DIR, transforms)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = VAE(z_dim=Z_DIM).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    feat_loss = VGGLoss(weight=VGG_LOSS_WEIGHT, device=DEVICE)
    def loss_fn(y_preds, y, mean, logvar, beta=KL_BETA):
        batch_size = y.shape[0]
        return F.mse_loss(y_preds, y, reduction='sum') / batch_size, beta * ((-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))) / batch_size), feat_loss(y_preds, y)


    total_epochs = TOTAL_EPOCHS

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    start_epoch = 0
    best_loss = float('inf')


    if LOAD_WEIGHTS:
        resume_path = os.path.join(save_dir, f"vae_epoch_{RESUME_EPOCH}.pth")

        if os.path.exists(resume_path):
            print(f"Loading checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming from Epoch {start_epoch} with previous loss {best_loss:.5f}\n\n")

    for epoch in tqdm(range(start_epoch, total_epochs)):
        model.train()

        rec_losses = []
        kl_losses = []
        p_losses = []

        for n_batch, X in enumerate(train_dl):
            X = X.to(DEVICE)

            y_preds, mean, logvar = model(X)

            rec_loss, kl_loss, p_loss = loss_fn(y_preds, X, mean, logvar)

            loss = rec_loss + kl_loss + p_loss

            rec_losses.append(rec_loss.item())
            kl_losses.append(kl_loss.item())
            p_losses.append(p_loss.item())

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


        avg_rec_loss = sum(rec_losses) / len(rec_losses)
        avg_kl_loss = sum(kl_losses) / len(kl_losses)
        avg_p_loss = sum(p_losses) / len(p_losses)
        total_epoch_loss = avg_rec_loss + avg_kl_loss + avg_p_loss

        print(f"\nEpoch: {epoch}")
        print(f"Rec Loss: {avg_rec_loss:.5f} | Feat Loss: {avg_p_loss:.5f} | KL Loss: {avg_kl_loss:.5f} ")

        if SAVE_WEIGHTS:
            epoch_checkpoint_path = os.path.join(save_dir, f"vae_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_epoch_loss,
            }, epoch_checkpoint_path)


        if total_epoch_loss < best_loss:
            best_loss = total_epoch_loss
            if SAVE_WEIGHTS:
                best_model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"##########################################\n!!!NEW BEST MODEL SAVED!!! (Loss: {best_loss:.5f})\n##########################################\n")


if __name__ == '__main__':
    main()
