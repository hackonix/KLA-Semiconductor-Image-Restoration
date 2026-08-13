"""Train a residual U-Net for paired grayscale image restoration.

Expected dataset layout:
    data_dir/degraded/
    data_dir/gt/

Images are paired by identical filenames.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class PairedImageDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.degraded_dir, self.gt_dir = root / "degraded", root / "gt"
        gt_names = {p.name for p in self.gt_dir.iterdir() if p.suffix.lower() in EXTS}
        self.items = [p for p in sorted(self.degraded_dir.iterdir())
                      if p.suffix.lower() in EXTS and p.name in gt_names]
        if not self.items:
            raise RuntimeError("No matching image pairs found in degraded/ and gt/.")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p = self.items[i]
        x = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        y = np.asarray(Image.open(self.gt_dir / p.name).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c,c,3,padding=1), nn.ReLU(inplace=True), nn.Conv2d(c,c,3,padding=1))
    def forward(self,x): return x + self.net(x)

class ResidualUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.in_conv=nn.Conv2d(1,base,3,padding=1)
        self.e1=nn.Sequential(ResidualBlock(base),ResidualBlock(base))
        self.down1=nn.Conv2d(base,base*2,3,stride=2,padding=1)
        self.e2=nn.Sequential(ResidualBlock(base*2),ResidualBlock(base*2))
        self.down2=nn.Conv2d(base*2,base*4,3,stride=2,padding=1)
        self.b=nn.Sequential(ResidualBlock(base*4),ResidualBlock(base*4))
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,stride=2)
        self.c2=nn.Conv2d(base*4,base*2,3,padding=1)
        self.up1=nn.ConvTranspose2d(base*2,base,2,stride=2)
        self.c1=nn.Conv2d(base*2,base,3,padding=1)
        self.out=nn.Conv2d(base,1,3,padding=1)
    def forward(self,x,output_size=None):
        if output_size is None: output_size=(x.shape[-2]*2,x.shape[-1]*2)
        base=F.interpolate(x,size=output_size,mode="bicubic",align_corners=False)
        e1=self.e1(self.in_conv(base)); e2=self.e2(self.down1(e1)); b=self.b(self.down2(e2))
        z=self.c2(torch.cat([self.up2(b),e2],1)); z=F.relu(z)
        z=self.c1(torch.cat([self.up1(z),e1],1)); z=F.relu(z)
        return torch.clamp(base+self.out(z),0,1)

def edge_loss(p,t):
    return F.l1_loss(p[..., :,1:]-p[..., :,:-1], t[..., :,1:]-t[..., :,:-1]) + F.l1_loss(p[...,1:,:]-p[...,:-1,:], t[...,1:,:]-t[...,:-1,:])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',required=True); ap.add_argument('--output_dir',default='checkpoints'); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch_size',type=int,default=16); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--val_fraction',type=float,default=.2); a=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    ds=PairedImageDataset(a.data_dir); nv=max(1,int(len(ds)*a.val_fraction)); nt=len(ds)-nv
    tr,va=random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(42)); tl=DataLoader(tr,a.batch_size,shuffle=True); vl=DataLoader(va,a.batch_size,shuffle=False)
    model=ResidualUNet().to(device); opt=torch.optim.Adam(model.parameters(),lr=a.lr); best=float('inf')
    for ep in range(a.epochs):
        model.train(); tsum=0
        for x,y in tl:
            x,y=x.to(device),y.to(device); p=model(x,y.shape[-2:]); loss=F.mse_loss(p,y)+.05*edge_loss(p,y); opt.zero_grad(); loss.backward(); opt.step(); tsum+=loss.item()*x.size(0)
        model.eval(); vsum=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(device),y.to(device); p=model(x,y.shape[-2:]); vsum+=(F.mse_loss(p,y)+.05*edge_loss(p,y)).item()*x.size(0)
        tr_loss=tsum/nt; va_loss=vsum/nv; state={'epoch':ep+1,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'train_loss':tr_loss,'val_loss':va_loss,'model_name':'ResidualUNet'}
        torch.save(state,out/f'epoch_{ep+1:02d}.pth'); torch.save(state,out/'last_model.pth')
        if va_loss<best: best=va_loss; torch.save(state,out/'final_model.pth')
        print(f'Epoch [{ep+1}/{a.epochs}] Train: {tr_loss:.6f} Val: {va_loss:.6f}')
    print(f'Best validation loss: {best:.8f} | Device: {device}')

if __name__=='__main__': main()
