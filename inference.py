"""Run restoration on every image in an input directory.

Usage:
python inference.py --input_dir <test_images> --output_dir <restored_outputs> --weights weights/final_model.pth
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
class ResidualBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x): return x+self.net(x)
class ResidualUNet(nn.Module):
    def __init__(self,base=32):
        super().__init__(); self.in_conv=nn.Conv2d(1,base,3,padding=1); self.e1=nn.Sequential(ResidualBlock(base),ResidualBlock(base)); self.down1=nn.Conv2d(base,base*2,3,stride=2,padding=1); self.e2=nn.Sequential(ResidualBlock(base*2),ResidualBlock(base*2)); self.down2=nn.Conv2d(base*2,base*4,3,stride=2,padding=1); self.b=nn.Sequential(ResidualBlock(base*4),ResidualBlock(base*4)); self.up2=nn.ConvTranspose2d(base*4,base*2,2,stride=2); self.c2=nn.Conv2d(base*4,base*2,3,padding=1); self.up1=nn.ConvTranspose2d(base*2,base,2,stride=2); self.c1=nn.Conv2d(base*2,base,3,padding=1); self.out=nn.Conv2d(base,1,3,padding=1)
    def forward(self,x):
        size=(x.shape[-2]*2,x.shape[-1]*2); base=F.interpolate(x,size=size,mode='bicubic',align_corners=False); e1=self.e1(self.in_conv(base)); e2=self.e2(self.down1(e1)); b=self.b(self.down2(e2)); z=F.relu(self.c2(torch.cat([self.up2(b),e2],1))); z=F.relu(self.c1(torch.cat([self.up1(z),e1],1))); return torch.clamp(base+self.out(z),0,1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_dir',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--weights',default='weights/final_model.pth'); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=ResidualUNet().to(device); ckpt=torch.load(args.weights,map_location=device,weights_only=False); model.load_state_dict(ckpt.get('model_state_dict',ckpt)); model.eval(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in EXTS)
    with torch.inference_mode():
        for p in files:
            arr=np.asarray(Image.open(p).convert('L'),dtype=np.float32)/255.0; pred=model(torch.from_numpy(arr)[None,None].to(device))[0,0].cpu().numpy(); Image.fromarray((pred*255).round().astype(np.uint8)).save(out/f'{p.stem}.png'); print(f'{p.name} -> {p.stem}.png')
    print(f'Processed {len(files)} images on {device}.')
if __name__=='__main__': main()
