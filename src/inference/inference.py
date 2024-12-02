# support COCO txt file

import sys
if './' not in sys.path:
    sys.path.append('./')
import os
import torch
import random
import numpy as np
from PIL import Image
import einops
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from src.datasets.dataset_display import MyDataset
from torch.utils.data import DataLoader

original_path = 'demo/txt'
ckpt_file_path = './ckpt/aerogen_diorr_last.ckpt'
resolution = 512
mask_size = 64
mode = 'test'
batch_size = 1
num_samples = 2
ddim_steps = 50
H = 512
W = 512

dataset = MyDataset(original_path, resolution, mask_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

model = create_model('configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml').cuda()
model.load_state_dict(load_state_dict(ckpt_file_path, location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

for batch in tqdm(dataloader):
    filenames = batch['filename']
    prompts = batch['txt']
    
    with torch.no_grad():
        mask_controls = batch['mask_conditions'].float().cuda()
        category_controls = batch['category_conditions'].float().cuda()
        bbox_controls = batch['bboxes'].float().cuda()
        mask_vectors = batch['mask_vector'].float().cuda()
        cond = {
            "c_crossattn": [model.get_learned_conditioning(prompts * num_samples)],
            "bbox_control": [torch.cat([bbox_controls for _ in range(num_samples)], dim=0)],
            "category_control": [torch.cat([category_controls for _ in range(num_samples)], dim=0)],
            "mask_control": [torch.cat([mask_controls for _ in range(num_samples)], dim=0)],
            "mask_vector": [torch.cat([mask_vectors for _ in range(num_samples)], dim=0)],
        }
        un_cond = {
            "c_crossattn": [model.get_learned_conditioning([""] * num_samples * batch_size)],
            "bbox_control": [torch.cat([bbox_controls for _ in range(num_samples)], dim=0)],
            "category_control": [torch.cat([category_controls for _ in range(num_samples)], dim=0)],
            "mask_control": [torch.cat([mask_controls for _ in range(num_samples)], dim=0)],
            "mask_vector": [torch.cat([mask_vectors for _ in range(num_samples)], dim=0)],
        }

        shape = (4, H // 8, W // 8)
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples * batch_size, shape, cond, verbose=False, eta=0.2,
                                         unconditional_guidance_scale=7.5,
                                         unconditional_conditioning=un_cond)


    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    for i, filename in enumerate(filenames):
        results = [x_samples[i * num_samples + j] for j in range(num_samples)]
        
        for idx, image_data in enumerate(results):
            output_dir = os.path.join('./demo/img', str(idx))
            os.makedirs(output_dir, exist_ok=True)
            
            image = Image.fromarray(image_data)
            output_file = os.path.join(output_dir, filename.replace('txt', 'jpg'))
            image.save(output_file)
            
            print(f'Generation images saved at {output_file}')
            print('ok')
