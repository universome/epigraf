"""
A debugging script for INR fitting
"""
import torch
from torch.optim import Adam
from PIL import Image
import torchvision.transforms.functional as TVF
from tqdm import tqdm
import numpy as np

from src.dnnlib import EasyDict
from src.training.networks_inr_gan import SynthesisNetwork


cfg = EasyDict({
    "output_channels": 3,
    "coord_dim": 2,
    "num_blocks": 2,
    "cbase": 32768,
    "cmax": 128,
    "num_fp16_blocks": 0,
    "fmm": EasyDict({"enabled": False}),
    "posenc_period_len": 128.0,
})
device = 'cuda'
m = SynthesisNetwork(cfg, 1).to(device)
trg = TVF.to_tensor(Image.open('target_img.jpg').crop((0, 0, 256, 256)))
trg = trg.to(device)
num_iters = 10000
h, w = trg.shape[1:]

x_coords = list(range(w))
y_coords = list(range(h))
batch_size = 2048
ws = torch.zeros(batch_size, m.num_ws, 1, device=device)

optim = Adam(m.parameters(), lr=1e-3)

for i in tqdm(range(num_iters)):
    curr_x_coords = torch.tensor(np.random.choice(x_coords, size=batch_size))
    curr_y_coords = torch.tensor(np.random.choice(y_coords, size=batch_size))
    curr_trg = trg[:, curr_y_coords, curr_x_coords] # [3, batch_size]

    inputs = torch.stack([curr_x_coords / w, curr_y_coords / h], dim=1).float().to(device).unsqueeze(2)

    optim.zero_grad()
    preds = m(inputs, ws) # [batch_size, c, num_samples]
    loss = (preds.squeeze(2) - curr_trg.t()).abs().sum()
    loss.backward()
    optim.step()

    if i % 250 == 0:
        print('Curr loss', loss.item())


curr_x_coords = torch.tensor(x_coords).repeat(h).to(device) / w
curr_y_coords = torch.tensor(y_coords).repeat_interleave(w).to(device) / h
with torch.no_grad():
    inputs = torch.stack([curr_x_coords, curr_y_coords], dim=1).float().to(device).unsqueeze(2)
    preds = m(inputs, torch.zeros(inputs.shape[0], m.num_ws, 1, device=device)).squeeze(2)
    preds = preds.reshape(w, h, 3).permute(2, 0, 1).cpu().clamp(0, 1)
TVF.to_pil_image(preds).save('result.jpg', q=95)
