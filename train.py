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

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'


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
  transforms = TT.Compose([
    TT.Resize((200, 200)),
    TT.CenterCrop((128, 128)),
    TT.RandomHorizontalFlip(0.5),
    TT.ColorJitter(0.15, 0.1, 0.1, 0.05),
    TT.ToTensor(),
    TT.Normalize([0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225])
  ])
  
  train_ds = FlowerDataset('data/', transforms)
  train_dl = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True
  )
  
  model = VAE().to(DEVICE)
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  feat_loss = VGGLoss(weight=0.6, device=DEVICE)
  def loss_fn (y_preds, y, mean, logvar, beta=1.0):
    batch_size = y.shape[0]
    return F.mse_loss(y_preds, y, reduction='sum') / batch_size, beta * ((-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))) / batch_size), feat_loss(y_preds, y)
  
  
  epochs = 128
  
  save_dir = "checkpoints"
  os.makedirs(save_dir, exist_ok=True)
  best_loss = float('inf')
  
  for epoch in tqdm(range(epochs)):
    model.train()
    
    rec_losses = []
    kl_losses = []
    feat_losses = []
    
    for n_batch, X in enumerate(train_dl):
      X = X.to(DEVICE)
      
      y_preds, mean, logvar = model(X)
      
      rec_loss, kl_loss, feat_loss = loss_fn(y_preds, X, mean, logvar)

      loss = rec_loss + kl_loss + feat_loss
      
      rec_losses.append(rec_loss.item())
      kl_losses.append(kl_loss.item())
      feat_losses.append(feat_loss.item())

      optimizer.zero_grad()
      loss.backward()
      #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      

    avg_rec_loss = sum(rec_losses) / len(rec_losses)
    avg_kl_loss = sum(kl_losses) / len(kl_losses)
    avg_feat_loss = sum(feat_losses) / len(feat_losses)
    total_epoch_loss = avg_rec_loss + avg_kl_loss + avg_feat_loss
    
    print(f"\nEpoch: {epoch}")
    print(f"Rec Loss: {avg_rec_loss:.5f} | KL Loss: {avg_kl_loss:.5f} | Feat Loss: {avg_feat_loss:.5f}")
    
    epoch_checkpoint_path = os.path.join(save_dir, f"vae_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_epoch_loss,
    }, epoch_checkpoint_path)
    
    
    if total_epoch_loss < best_loss:
        best_loss = total_epoch_loss
        best_model_path = os.path.join(save_dir, "best_model.pth")
        
        torch.save(model.state_dict(), best_model_path)
        print(f"##########################################\n!!!NEW BEST MODEL SAVED!!! (Loss: {best_loss:.5f})\n##########################################\n")
        
        
if __name__ == '__main__':
  main()