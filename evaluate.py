"""Standalone evaluation entry point for the KLA restoration model.

Required arguments:
  --input_dir  directory containing degraded test images
  --output_dir directory where restored images are written

Optional:
  --weights checkpoint path (default: weights/final_model.pth)
  --gt_dir ground-truth directory for PSNR/SSIM calculation
"""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

def read_gray(p): return np.asarray(Image.open(p).convert('L'),dtype=np.float32)/255.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_dir',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--weights',default='weights/final_model.pth'); ap.add_argument('--gt_dir',default=None); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=ResidualUNet().to(device)
    ckpt=torch.load(args.weights,map_location=device,weights_only=False); model.load_state_dict(ckpt.get('model_state_dict',ckpt)); model.eval()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); files=sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in EXTS)
    psnr_values=[]; ssim_values=[]; times=[]
    with torch.inference_mode():
        for p in files:
            x=torch.from_numpy(read_gray(p))[None,None].to(device)
            if device.type=='cuda': torch.cuda.synchronize()
            t=time.perf_counter(); pred=model(x)
            if device.type=='cuda': torch.cuda.synchronize()
            times.append(time.perf_counter()-t); pred_np=pred[0,0].cpu().numpy(); Image.fromarray((pred_np*255).round().astype(np.uint8)).save(out/f'{p.stem}.png')
            if args.gt_dir:
                gp=Path(args.gt_dir)/p.name
                if gp.exists():
                    gt=read_gray(gp); psnr_values.append(peak_signal_noise_ratio(gt,pred_np,data_range=1.0)); ssim_values.append(structural_similarity(gt,pred_np,data_range=1.0))
    print(f'Images processed: {len(files)}')
    if times: print(f'Mean inference time/image: {np.mean(times):.6f} s')
    if psnr_values: print(f'PSNR: {np.mean(psnr_values):.6f} dB\nSSIM: {np.mean(ssim_values):.6f}')

if __name__=='__main__': main()
