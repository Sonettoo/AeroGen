# support VOC2007 xml file

import sys
if './' not in sys.path:
    sys.path.append('./')
import os
import torch
import numpy as np
from PIL import Image
import einops
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from src.datasets.dataset_test import MyDataset

# Define paths and parameters
original_path = './dataset/DIOR-VOC/Annotations/Oriented_Bounding_Boxes/'
image_file_path = './dataset/DIOR-VOC/VOC2007/JPEGImages/'
ckpt_file_path = './ckpt/aerogen_diorr_last.ckpt'
output_dir = 'path/to/save/images'
resolution = 512
mask_size = 64
num_samples = 1  # Number of samples per input
ddim_steps = 50
H = 512
W = 512
batch_size = 10  # Set your desired batch size here

# Create dataset
dataset = MyDataset(
    original_path, image_file_path, resolution, mask_size,
    mode='test', rotate_once=False, rotate_twice=False
)

# Helper functions
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

# Load model and sampler
model = create_model('configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml').cuda()
model.load_state_dict(load_state_dict(ckpt_file_path, location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# Helper function for repeating tensors
def repeat_tensor(tensor, num_repeats):
    expanded_tensor = tensor.unsqueeze(1).repeat(1, num_repeats, *[1 for _ in range(tensor.dim() - 1)])
    return expanded_tensor.view(-1, *tensor.shape[1:])

# Process dataset in batches
num = 0
for i in tqdm(range(0, len(dataset), batch_size)):
    # Prepare batch data
    batch_infodicts = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
    batch_filenames = [infodict['filename'] for infodict in batch_infodicts]
    batch_prompts = [infodict['txt'] for infodict in batch_infodicts]

    # Check if the output files already exist
    files_to_skip = []
    for idx, filename in enumerate(batch_filenames):
        save_path = os.path.join(output_dir, filename.replace('.xml', '.jpg'))
        if os.path.exists(save_path):
            files_to_skip.append(idx)
            print(f'Skipping {save_path}, already exists.')

    if len(files_to_skip) == len(batch_filenames):
        continue  # Skip entire batch if all files are already generated

    valid_batch_infodicts = [batch_infodicts[idx] for idx in range(len(batch_infodicts)) if idx not in files_to_skip]
    valid_batch_prompts = [batch_prompts[idx] for idx in range(len(batch_prompts)) if idx not in files_to_skip]

    with torch.no_grad():
        # Prepare conditioning inputs
        batch_mask_control = torch.stack(
            [torch.from_numpy(infodict['mask_conditions']).float() for infodict in valid_batch_infodicts], dim=0
        ).cuda()
        batch_category_control = torch.stack(
            [torch.from_numpy(infodict['category_conditions']).float() for infodict in valid_batch_infodicts], dim=0
        ).cuda()
        batch_bbox_control = torch.stack(
            [torch.from_numpy(infodict['bboxes']).float() for infodict in valid_batch_infodicts], dim=0
        ).cuda()
        batch_mask_vector = torch.stack(
            [torch.from_numpy(infodict['mask_vector']).float() for infodict in valid_batch_infodicts], dim=0
        ).cuda()

        # Repeat each item `num_samples` times
        total_samples = len(valid_batch_infodicts) * num_samples
        batch_prompts_repeated = []
        for prompt in valid_batch_prompts:
            batch_prompts_repeated.extend([prompt] * num_samples)

        # Expand and repeat tensors
        batch_mask_control_repeated = repeat_tensor(batch_mask_control, num_samples)
        batch_category_control_repeated = repeat_tensor(batch_category_control, num_samples)
        batch_bbox_control_repeated = repeat_tensor(batch_bbox_control, num_samples)
        batch_mask_vector_repeated = repeat_tensor(batch_mask_vector, num_samples)

        # Get conditioning
        c_crossattn = model.get_learned_conditioning(batch_prompts_repeated)
        un_c_crossattn = model.get_learned_conditioning([""] * total_samples)

        # Prepare conditioning dictionaries
        cond = {
            "c_crossattn": [c_crossattn],
            "bbox_control": [batch_bbox_control_repeated],
            "category_control": [batch_category_control_repeated],
            "mask_control": [batch_mask_control_repeated],
            "mask_vector": [batch_mask_vector_repeated],
        }
        un_cond = {
            "c_crossattn": [un_c_crossattn],
            "bbox_control": [batch_bbox_control_repeated],
            "category_control": [batch_category_control_repeated],
            "mask_control": [batch_mask_control_repeated],
            "mask_vector": [batch_mask_vector_repeated],
        }
        shape = (4, H // 8, W // 8)
        samples, _ = ddim_sampler.sample(
            ddim_steps, total_samples, shape, cond, verbose=False, eta=0.2,
            unconditional_guidance_scale=1,
            unconditional_conditioning=un_cond
        )

    # Decode and save samples for non-skipped items
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
                ).cpu().numpy().clip(0, 255).astype(np.uint8)

    for idx, image_data in enumerate(x_samples):
        img_index = idx // num_samples
        sample_index = idx % num_samples
        filename_idx = [i for i in range(len(batch_filenames)) if i not in files_to_skip][img_index]
        filename = batch_filenames[filename_idx]
        save_path = os.path.join(output_dir, filename.replace('.xml', '.jpg'))
        image = Image.fromarray(image_data)
        image.save(save_path)

    num += len(valid_batch_infodicts)
